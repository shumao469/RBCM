"""Train an RBCM edge detection model.

PyCharm usage:
1. Edit `DEFAULT_ARGS` below if needed.
2. Right-click this file and choose Run.

Command-line usage:
```powershell
python edge_model\\train.py --config edge_model\\configs\\rbcm\\nyudv2_strict.yaml
```
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from edge_model.core.env import append_workspace_local_packages

append_workspace_local_packages(PROJECT_ROOT)

from edge_model.core.checkpoint_io import load_checkpoint
from edge_model.core.config import deep_update, load_config, project_path, save_config
from edge_model.core.paths import make_run_paths
from edge_model.core.seed import seed_everything
from edge_model.data.build import make_dataset, make_loader
from edge_model.engine.train_loop import (
    append_metrics_csv,
    average_train_epoch_state,
    evaluate,
    train_one_epoch,
)
from edge_model.models.build import build_model
from edge_model.tools.analyze_training_log import analyze_training_log
from rbcm_edge.models.losses import EdgeDetectionLoss

STOP_REQUESTED = False

DEFAULT_ARGS = {
    "config": PROJECT_ROOT / "edge_model" / "configs" / "rbcm" / "nyudv2_strict.yaml",
    "experiment_name": None,
    "train_dataset": None,
    "val_dataset": None,
    "epochs": None,
    "batch_size": None,
    "input_size": None,
    "learning_rate": None,
    "time_budget_minutes": None,
    "device": None,
    "periodic_checkpoint_minutes": None,
    "periodic_checkpoint_keep": None,
}


def parse_args() -> argparse.Namespace:
    """Parse training arguments with editable defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_ARGS["config"])
    parser.add_argument("--experiment-name", default=DEFAULT_ARGS["experiment_name"])
    parser.add_argument("--train-dataset", default=DEFAULT_ARGS["train_dataset"])
    parser.add_argument("--val-dataset", default=DEFAULT_ARGS["val_dataset"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_ARGS["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_ARGS["batch_size"])
    parser.add_argument("--input-size", type=int, default=DEFAULT_ARGS["input_size"])
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_ARGS["learning_rate"])
    parser.add_argument("--time-budget-minutes", type=float, default=DEFAULT_ARGS["time_budget_minutes"])
    parser.add_argument("--device", default=DEFAULT_ARGS["device"])
    parser.add_argument(
        "--periodic-checkpoint-minutes",
        type=float,
        default=DEFAULT_ARGS["periodic_checkpoint_minutes"],
        help="Save a resumable checkpoint about this often, in wall-clock minutes. <=0 disables it.",
    )
    parser.add_argument(
        "--periodic-checkpoint-keep",
        type=int,
        default=DEFAULT_ARGS["periodic_checkpoint_keep"],
        help="Keep this many numbered periodic checkpoints in checkpoints/periodic. <=0 keeps all.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoints/last.pt under the configured output_root/experiment_name if it exists.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume from an explicit checkpoint path. Overrides --resume.",
    )
    parser.add_argument(
        "--stop-file",
        type=Path,
        default=None,
        help="If this file exists, stop after the current update and keep last.pt resumable.",
    )
    return parser.parse_args()


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply command-line overrides to the YAML config."""
    updates = {
        "experiment_name": args.experiment_name,
        "device": args.device,
        "dataset": {
            "train_dataset": args.train_dataset,
            "val_dataset": args.val_dataset,
            "input_size": args.input_size,
        },
        "loader": {"batch_size": args.batch_size},
        "train": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "time_budget_minutes": args.time_budget_minutes,
            "stop_file": str(args.stop_file) if args.stop_file else None,
            "periodic_checkpoint_minutes": args.periodic_checkpoint_minutes,
            "periodic_checkpoint_keep": args.periodic_checkpoint_keep,
        },
    }
    return deep_update(config, updates)


def enforce_training_protocol(config: dict, config_path: Path) -> None:
    """Reject archived configurations whose validation and test sets overlap."""
    protocol = config.get("paper_protocol", {}) or {}
    if bool(protocol.get("training_disabled", False)):
        replacement = protocol.get("replacement", "a strict V5 configuration")
        reason = protocol.get("reason", "the protocol is archived and non-strict")
        raise RuntimeError(
            f"Training is disabled for archived config {config_path}: {reason}. "
            f"Use {replacement}."
        )


def request_stop(signum, _frame) -> None:
    """Signal handler that asks training to checkpoint after the current update."""
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"Received signal {signum}; training will save a resumable checkpoint after the current update.")


def register_stop_signal_handlers() -> None:
    """Register best-effort signal handlers for graceful long-run interruption."""
    for signal_name in ("SIGINT", "SIGTERM", "SIGHUP"):
        signum = getattr(signal, signal_name, None)
        if signum is None:
            continue
        try:
            signal.signal(signum, request_stop)
        except (OSError, RuntimeError, ValueError):
            continue


def resolve_resume_checkpoint(args: argparse.Namespace, config: dict, run_paths) -> Path | None:
    """Resolve the checkpoint used for a resumable training launch."""
    if args.resume_from is not None:
        path = args.resume_from if args.resume_from.is_absolute() else PROJECT_ROOT / args.resume_from
        if not path.exists():
            raise FileNotFoundError(f"Requested resume checkpoint does not exist: {path}")
        return path
    if not args.resume:
        return None
    path = run_paths.checkpoints / "last.pt"
    if path.exists():
        return path
    configured = config.get("train", {}).get("resume_from")
    if configured:
        path = project_path(config, configured)
        if not path.exists():
            raise FileNotFoundError(f"Configured resume checkpoint does not exist: {path}")
        return path
    print(f"--resume was requested, but no checkpoint exists at {run_paths.checkpoints / 'last.pt'}; starting fresh.")
    return None


def load_training_state(
    *,
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    device: torch.device,
    config: dict,
) -> tuple[int, float, dict | None]:
    """Load full continuation state, including an optional mid-epoch cursor."""
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    validate_resume_config(checkpoint.get("config"), config)
    model.load_state_dict(checkpoint["model"], strict=True)
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    epoch = int(checkpoint.get("epoch", 0))
    metrics = checkpoint.get("metrics", {}) or {}
    best_score = checkpoint.get("best_score")
    if best_score is None and metrics:
        best_score = checkpoint_score(metrics, config)
    if best_score is None:
        best_score = -float("inf")
    epoch_complete = bool(checkpoint.get("epoch_complete", True))
    epoch_state = checkpoint.get("epoch_state") if not epoch_complete else None
    restore_rng_state(checkpoint.get("rng_state"))
    if epoch_complete:
        print(f"Resumed training from {checkpoint_path} at epoch {epoch}; next epoch is {epoch + 1}.")
        return epoch + 1, float(best_score), None
    step = int(checkpoint.get("step_in_epoch", (epoch_state or {}).get("step_in_epoch", 0)))
    print(
        f"Resumed training from {checkpoint_path} inside epoch {epoch} at step {step}; "
        "the same epoch order will be replayed and completed batches will be skipped."
    )
    return epoch, float(best_score), epoch_state


def validate_resume_config(saved_config: dict | None, current_config: dict) -> None:
    """Reject continuation when a change would alter the optimized experiment."""
    if not isinstance(saved_config, dict):
        return

    train_keys = (
        "learning_rate",
        "weight_decay",
        "scheduler",
        "scheduler_t_max_epochs",
        "scheduler_milestones",
        "scheduler_gamma",
        "gradient_clip_norm",
        "mixed_precision",
        "resumable_mid_epoch",
    )
    saved_train = saved_config.get("train", {})
    current_train = current_config.get("train", {})
    saved_signature = {
        "seed": saved_config.get("seed"),
        "dataset": saved_config.get("dataset"),
        "loader": saved_config.get("loader"),
        "model": saved_config.get("model"),
        "loss": saved_config.get("loss"),
        "train": {key: saved_train.get(key) for key in train_keys},
    }
    current_signature = {
        "seed": current_config.get("seed"),
        "dataset": current_config.get("dataset"),
        "loader": current_config.get("loader"),
        "model": current_config.get("model"),
        "loss": current_config.get("loss"),
        "train": {key: current_train.get(key) for key in train_keys},
    }
    if saved_signature != current_signature:
        changed = [
            key
            for key in saved_signature
            if saved_signature.get(key) != current_signature.get(key)
        ]
        raise ValueError(
            "Refusing to resume with experiment-defining config changes in: "
            + ", ".join(changed)
        )


def make_training_checkpoint(
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    config: dict,
    metrics: dict,
    best_score: float,
    epoch_complete: bool = True,
    step_in_epoch: int | None = None,
    epoch_state: dict | None = None,
) -> dict:
    """Build a resumable training checkpoint payload."""
    return {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "config": config,
        "metrics": metrics,
        "best_score": float(best_score),
        "epoch_complete": bool(epoch_complete),
        "step_in_epoch": None if step_in_epoch is None else int(step_in_epoch),
        "epoch_state": epoch_state,
        "rng_state": capture_rng_state(),
    }


def capture_rng_state() -> dict:
    """Capture RNG state so a resumed run stays as close as possible to the interrupted run."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict | None) -> None:
    """Restore RNG state from a training checkpoint when available."""
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"].cpu())
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_state])


def save_checkpoint_atomic(checkpoint: dict, path: Path) -> None:
    """Write a checkpoint through a temporary file to avoid half-written artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    torch.save(checkpoint, tmp_path)
    tmp_path.replace(path)


def stop_reason_after_epoch(config: dict) -> str | None:
    """Return the reason training should stop after the current epoch, if any."""
    if STOP_REQUESTED:
        return "received an interrupt/termination signal"
    stop_file = config.get("train", {}).get("stop_file")
    if not stop_file:
        return None
    path = project_path(config, stop_file)
    if path.exists():
        return f"stop file exists at {path}"
    return None


def periodic_checkpoint_interval_seconds(config: dict) -> float:
    """Return configured wall-clock interval for lightweight periodic checkpointing."""
    train_cfg = config.get("train", {})
    minutes = train_cfg.get("periodic_checkpoint_minutes")
    if minutes is None:
        minutes = train_cfg.get("periodic_checkpoint_interval_minutes", 0.0)
    if minutes is None:
        return 0.0
    return max(0.0, float(minutes) * 60.0)


def periodic_checkpoint_keep(config: dict) -> int:
    """Return how many numbered periodic checkpoints should be retained."""
    keep = config.get("train", {}).get("periodic_checkpoint_keep", 3)
    return int(keep) if keep is not None else 3


def maybe_save_periodic_checkpoint(
    *,
    checkpoint: dict,
    run_paths,
    config: dict,
    epoch: int,
    last_save_time: float,
    now: float,
    source_checkpoint_path: Path | None = None,
) -> float:
    """Save a resumable checkpoint snapshot when the wall-clock interval has elapsed."""
    interval_seconds = periodic_checkpoint_interval_seconds(config)
    if interval_seconds <= 0.0 or (now - last_save_time) < interval_seconds:
        return last_save_time

    periodic_dir = run_paths.checkpoints / "periodic"
    periodic_path = periodic_dir / f"epoch_{epoch:03d}.pt"
    if source_checkpoint_path is not None and source_checkpoint_path.exists():
        periodic_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_checkpoint_path, periodic_path)
    else:
        save_checkpoint_atomic(checkpoint, periodic_path)
    shutil.copy2(periodic_path, run_paths.checkpoints / "periodic_latest.pt")

    keep = periodic_checkpoint_keep(config)
    if keep > 0:
        snapshots = sorted(periodic_dir.glob("epoch_*.pt"))
        for old_path in snapshots[:-keep]:
            old_path.unlink(missing_ok=True)

    print(f"Saved periodic checkpoint for mid-run evaluation: {periodic_path}")
    return now


def save_mid_epoch_checkpoint(
    *,
    checkpoint: dict,
    run_paths,
    config: dict,
    epoch: int,
    step: int,
) -> None:
    """Atomically refresh last.pt and retain a small rotating recovery history."""
    last_path = run_paths.checkpoints / "last.pt"
    save_checkpoint_atomic(checkpoint, last_path)

    periodic_dir = run_paths.checkpoints / "periodic"
    periodic_dir.mkdir(parents=True, exist_ok=True)
    periodic_path = periodic_dir / f"epoch_{epoch:03d}_step_{step:05d}.pt"
    shutil.copy2(last_path, periodic_path)
    shutil.copy2(last_path, run_paths.checkpoints / "periodic_latest.pt")

    keep = periodic_checkpoint_keep(config)
    if keep > 0:
        snapshots = sorted(periodic_dir.glob("epoch_*_step_*.pt"))
        for old_path in snapshots[:-keep]:
            old_path.unlink(missing_ok=True)
    print(f"Saved mid-epoch recovery checkpoint: epoch={epoch}, step={step}", flush=True)


def resumable_epoch_seed(base_seed: int, epoch: int) -> int:
    """Return a stable per-epoch seed for sample order and worker transforms."""
    return int((int(base_seed) * 1_000_003 + int(epoch) * 97_409) % (2**63 - 1))


def main(args: argparse.Namespace) -> None:
    """Run the complete training workflow."""
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = apply_overrides(load_config(config_path), args)
    enforce_training_protocol(config, config_path.resolve())
    config["paths"]["project_root"] = str(PROJECT_ROOT)
    seed_everything(
        int(config.get("seed", 42)),
        deterministic=bool(config.get("deterministic", True)),
    )
    register_stop_signal_handlers()

    device_name = config.get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is unavailable; falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    output_root = project_path(config, config["paths"].get("output_root", "edge_outputs/rbcm/runs"))
    run_paths = make_run_paths(output_root, config.get("experiment_name", "edge_experiment"))
    save_config(run_paths.root / "config.yaml", config)

    dataset_cfg = config["dataset"]
    train_dataset = make_dataset(
        config,
        dataset_name=dataset_cfg["train_dataset"],
        split=dataset_cfg.get("train_split", "train"),
        training=True,
    )
    val_dataset = make_dataset(
        config,
        dataset_name=dataset_cfg["val_dataset"],
        split=dataset_cfg.get("val_split", "all"),
        training=False,
    )
    train_cfg = config.get("train", {})
    mid_epoch_resume_enabled = bool(train_cfg.get("resumable_mid_epoch", False))
    train_loader = None if mid_epoch_resume_enabled else make_loader(train_dataset, config, shuffle=True)
    val_loader = make_loader(val_dataset, config, shuffle=False)

    model = build_model(config).to(device)
    loss_cfg = config.get("loss", {})
    criterion = EdgeDetectionLoss(
        dice_weight=float(loss_cfg.get("dice_weight", 1.0)),
        local_weight=float(loss_cfg.get("local_weight", 0.3)),
        side_weight=float(loss_cfg.get("side_weight", 0.4)),
        context_weight=float(loss_cfg.get("context_weight", 0.0)),
        context_dilation=int(loss_cfg.get("context_dilation", 3)),
        context_gamma=float(loss_cfg.get("context_gamma", 1.0)),
        gate_sparsity_weight=float(loss_cfg.get("gate_sparsity_weight", 1e-4)),
        density_weight=float(loss_cfg.get("density_weight", 0.0)),
        density_target_multiplier=float(loss_cfg.get("density_target_multiplier", 2.0)),
        density_floor=float(loss_cfg.get("density_floor", 0.005)),
        tversky_weight=float(loss_cfg.get("tversky_weight", 0.0)),
        tversky_alpha=float(loss_cfg.get("tversky_alpha", 0.7)),
        tversky_beta=float(loss_cfg.get("tversky_beta", 0.3)),
        tversky_gamma=float(loss_cfg.get("tversky_gamma", 1.0)),
        far_background_weight=float(loss_cfg.get("far_background_weight", 0.0)),
        far_background_dilation=int(loss_cfg.get("far_background_dilation", 3)),
        far_background_gamma=float(loss_cfg.get("far_background_gamma", 1.5)),
        background_blob_weight=float(loss_cfg.get("background_blob_weight", 0.0)),
        background_blob_dilation=int(loss_cfg.get("background_blob_dilation", 3)),
        background_blob_pool_size=int(loss_cfg.get("background_blob_pool_size", 9)),
        background_blob_margin=float(loss_cfg.get("background_blob_margin", 0.06)),
        background_blob_gamma=float(loss_cfg.get("background_blob_gamma", 2.0)),
        aux_tversky_weight=float(loss_cfg.get("aux_tversky_weight", 0.0)),
        aux_far_background_weight=float(loss_cfg.get("aux_far_background_weight", 0.0)),
        aux_far_background_dilation=loss_cfg.get("aux_far_background_dilation"),
        aux_far_background_gamma=loss_cfg.get("aux_far_background_gamma"),
        mix_balance_weight=float(loss_cfg.get("mix_balance_weight", 0.0)),
        mix_prior=loss_cfg.get("mix_prior"),
        separation_weight=float(loss_cfg.get("separation_weight", loss_cfg.get("sep_weight", 0.0))),
        uncertainty_prior_weight=float(loss_cfg.get("uncertainty_prior_weight", 0.0)),
        uncertainty_prior=float(loss_cfg.get("uncertainty_prior", 0.35)),
        hf_residual_weight=float(loss_cfg.get("hf_residual_weight", 0.0)),
        hf_gate_sparsity_weight=float(loss_cfg.get("hf_gate_sparsity_weight", 0.0)),
        continuity_residual_weight=float(loss_cfg.get("continuity_residual_weight", 0.0)),
        continuity_gate_sparsity_weight=float(loss_cfg.get("continuity_gate_sparsity_weight", 0.0)),
        continuity_isolated_weight=float(loss_cfg.get("continuity_isolated_weight", 0.0)),
        continuity_isolated_dilation=loss_cfg.get("continuity_isolated_dilation"),
        continuity_isolated_gamma=float(loss_cfg.get("continuity_isolated_gamma", 1.5)),
        continuity_support_weight=float(loss_cfg.get("continuity_support_weight", 0.0)),
        continuity_support_target=float(loss_cfg.get("continuity_support_target", 0.14)),
        continuity_support_dilation=int(loss_cfg.get("continuity_support_dilation", 1)),
        continuity_support_gamma=float(loss_cfg.get("continuity_support_gamma", 1.0)),
        calibration_weight=float(loss_cfg.get("calibration_weight", 0.0)),
        calibration_edge_margin=float(loss_cfg.get("calibration_edge_margin", 0.55)),
        calibration_background_margin=float(loss_cfg.get("calibration_background_margin", 0.04)),
        calibration_background_dilation=int(loss_cfg.get("calibration_background_dilation", 3)),
        calibration_edge_weight=float(loss_cfg.get("calibration_edge_weight", 0.35)),
        calibration_background_weight=float(loss_cfg.get("calibration_background_weight", 1.0)),
        active_density_weight=float(loss_cfg.get("active_density_weight", 0.0)),
        active_density_min_multiplier=float(loss_cfg.get("active_density_min_multiplier", 0.35)),
        active_density_max_multiplier=float(loss_cfg.get("active_density_max_multiplier", 1.45)),
        active_density_floor=float(loss_cfg.get("active_density_floor", 0.002)),
        active_density_threshold=float(loss_cfg.get("active_density_threshold", 0.5)),
        active_density_temperature=float(loss_cfg.get("active_density_temperature", 0.06)),
        state_supervision_weight=float(loss_cfg.get("state_supervision_weight", 0.0)),
        state_edge_dilation=int(loss_cfg.get("state_edge_dilation", 1)),
        state_suppress_dilation=int(loss_cfg.get("state_suppress_dilation", 5)),
        target_threshold=float(loss_cfg.get("target_threshold", 0.01)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"].get("learning_rate", 1e-4)),
        weight_decay=float(config["train"].get("weight_decay", 1e-4)),
    )
    use_amp = bool(config["train"].get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp) if use_amp else None
    scheduler = build_scheduler(optimizer, config, epochs=int(config["train"].get("epochs", 20)))

    strict_selection_cfg = normalize_strict_selection_config(config)
    strict_selection_enabled = bool(strict_selection_cfg.get("enabled", False))
    save_eval_candidates = bool(train_cfg.get("save_eval_checkpoints", strict_selection_enabled))
    candidate_dir = run_paths.checkpoints / "eval_candidates"
    if save_eval_candidates:
        candidate_dir.mkdir(parents=True, exist_ok=True)

    best_score = -float("inf")
    start_epoch = 1
    resume_epoch_state: dict | None = None
    resume_checkpoint = resolve_resume_checkpoint(args, config, run_paths)
    if resume_checkpoint is not None:
        start_epoch, best_score, resume_epoch_state = load_training_state(
            checkpoint_path=resume_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            config=config,
        )

    stale_evals = 0
    epochs = int(config["train"].get("epochs", 20))
    early_stop_patience = config["train"].get("early_stop_patience")
    early_stop_min_delta = float(config["train"].get("early_stop_min_delta", 0.0))
    start_time = time.monotonic()
    last_periodic_save_time = start_time
    time_budget_minutes = config["train"].get("time_budget_minutes")
    time_budget_seconds = None if time_budget_minutes is None else float(time_budget_minutes) * 60.0
    if start_epoch > epochs:
        print(f"Checkpoint already reached epoch {start_epoch - 1}; configured epochs={epochs}. Nothing to train.")
        return

    for epoch in range(start_epoch, epochs + 1):
        latest_metrics: dict = {}
        stop_after_epoch_reason: str | None = None
        if mid_epoch_resume_enabled:
            epoch_generator = torch.Generator()
            epoch_generator.manual_seed(resumable_epoch_seed(int(config.get("seed", 42)), epoch))
            active_train_loader = make_loader(
                train_dataset,
                config,
                shuffle=True,
                generator=epoch_generator,
                # Recreate workers from the same epoch seed after a restart so
                # random crops/flips replay before the saved batch cursor.
                persistent_workers=False,
            )
        else:
            if train_loader is None:
                raise RuntimeError("Training loader was not initialized.")
            active_train_loader = train_loader

        active_resume_state = resume_epoch_state if epoch == start_epoch else None

        def save_recovery_checkpoint(step: int, epoch_state: dict) -> None:
            partial_metrics = average_train_epoch_state(epoch_state)
            checkpoint = make_training_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config,
                metrics=partial_metrics,
                best_score=best_score,
                epoch_complete=False,
                step_in_epoch=step,
                epoch_state=epoch_state,
            )
            save_mid_epoch_checkpoint(
                checkpoint=checkpoint,
                run_paths=run_paths,
                config=config,
                epoch=epoch,
                step=step,
            )

        train_result = train_one_epoch(
            model=model,
            loader=active_train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            epoch=epoch,
            log_interval=int(config["train"].get("log_interval", 10)),
            gradient_clip_norm=positive_optional_float(config["train"].get("gradient_clip_norm")),
            diagnostic_interval=int(train_cfg.get("diagnostic_interval", 10)),
            resume_state=active_resume_state,
            checkpoint_interval_steps=(
                int(train_cfg.get("checkpoint_interval_steps", 0))
                if mid_epoch_resume_enabled
                else 0
            ),
            checkpoint_interval_seconds=(
                periodic_checkpoint_interval_seconds(config)
                if mid_epoch_resume_enabled
                else 0.0
            ),
            checkpoint_callback=save_recovery_checkpoint if mid_epoch_resume_enabled else None,
            stop_requested=(
                (lambda: stop_reason_after_epoch(config) is not None)
                if mid_epoch_resume_enabled
                else None
            ),
        )
        if not bool(train_result.get("completed", False)):
            step = int(train_result.get("state", {}).get("step_in_epoch", 0))
            print(
                f"Stopped inside epoch {epoch} after step {step}; "
                "resume with the same command plus --resume."
            )
            return
        train_metrics = train_result["metrics"]
        resume_epoch_state = None

        row = {"epoch": epoch, "split": "train", "lr": optimizer.param_groups[0]["lr"], **train_metrics}
        append_metrics_csv(run_paths.logs / "train_log.csv", row)
        print(row)
        latest_metrics = train_metrics

        if epoch % int(config["train"].get("eval_interval", 1)) == 0:
            save_visuals = epoch % int(config["train"].get("save_visual_interval", 1)) == 0
            save_gate_heatmaps = bool(config["train"].get("save_gate_heatmaps", False))
            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                visual_dir=run_paths.visualizations / f"epoch_{epoch:03d}" if save_visuals else None,
                pred_dir=None,
                gate_dir=run_paths.gate_heatmaps / f"epoch_{epoch:03d}" if save_visuals and save_gate_heatmaps else None,
                max_visual_samples=int(config["train"].get("max_visual_samples", 8)),
                metric_mode=str(train_cfg.get("metric_mode", "fast_gpu")),
                fast_metric_thresholds=int(train_cfg.get("fast_metric_thresholds", 49)),
                fast_metric_tolerance_pixels=int(train_cfg.get("fast_metric_tolerance_pixels", 4)),
                apply_nms=bool(train_cfg.get("apply_nms", False)),
                nms_low_threshold=float(train_cfg.get("nms_low_threshold", 0.0)),
                diagnostic_interval=int(train_cfg.get("diagnostic_interval", 10)),
            )
            current_score = checkpoint_score(val_metrics, config)
            val_metrics["checkpoint_score"] = current_score
            val_row = {"epoch": epoch, "split": "val", **val_metrics}
            append_metrics_csv(run_paths.logs / "train_log.csv", val_row)
            print(val_row)
            latest_metrics = val_metrics

            if current_score > best_score + early_stop_min_delta:
                best_score = current_score
                stale_evals = 0
                checkpoint = make_training_checkpoint(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    config=config,
                    metrics=val_metrics,
                    best_score=best_score,
                )
                if save_eval_candidates:
                    save_checkpoint_atomic(checkpoint, candidate_dir / f"epoch_{epoch:03d}.pt")
                save_checkpoint_atomic(checkpoint, run_paths.checkpoints / "best_validation_metric.pt")
                save_checkpoint_atomic(checkpoint, run_paths.checkpoints / "best.pt")
                label = "provisional best" if strict_selection_enabled else "best"
                print(
                    f"Saved {label} checkpoint with validation score={best_score:.4f}, "
                    f"ODS={float(val_metrics.get('ODS', 0.0)):.4f}"
                )
            else:
                stale_evals += 1
                checkpoint = make_training_checkpoint(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    config=config,
                    metrics=val_metrics,
                    best_score=best_score,
                )
                if save_eval_candidates:
                    save_checkpoint_atomic(checkpoint, candidate_dir / f"epoch_{epoch:03d}.pt")

            if early_stop_patience is not None and stale_evals >= int(early_stop_patience):
                stop_after_epoch_reason = (
                    "no validation checkpoint-score improvement "
                    f"for {stale_evals} evaluations"
                )

        if scheduler is not None:
            scheduler.step()

        latest_checkpoint = make_training_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            metrics=latest_metrics,
            best_score=best_score,
        )
        last_checkpoint_path = run_paths.checkpoints / "last.pt"
        save_checkpoint_atomic(latest_checkpoint, last_checkpoint_path)
        now = time.monotonic()
        last_periodic_save_time = maybe_save_periodic_checkpoint(
            checkpoint=latest_checkpoint,
            run_paths=run_paths,
            config=config,
            epoch=epoch,
            last_save_time=last_periodic_save_time,
            now=now,
            source_checkpoint_path=last_checkpoint_path,
        )

        if stop_after_epoch_reason is not None:
            print(f"Stopping after epoch {epoch}: {stop_after_epoch_reason}.")
            break

        manual_stop_reason = stop_reason_after_epoch(config)
        if manual_stop_reason is not None:
            print(f"Stopping after epoch {epoch}: {manual_stop_reason}.")
            break

        if time_budget_seconds is not None and (time.monotonic() - start_time) >= time_budget_seconds:
            elapsed_minutes = (time.monotonic() - start_time) / 60.0
            print(f"Stopping after epoch {epoch}: reached time budget ({elapsed_minutes:.1f} min).")
            break

    if strict_selection_enabled:
        select_checkpoint_with_strict_eval(
            model=model,
            base_config=config,
            run_paths=run_paths,
            criterion=criterion,
            device=device,
            selection_cfg=strict_selection_cfg,
        )

    if bool(config["train"].get("auto_plot_log", True)):
        log_csv = run_paths.logs / "train_log.csv"
        plot_dir = run_paths.root / "plots"
        print(f"Generating training trend plots from {log_csv}")
        analyze_training_log(log_csv=log_csv, output_dir=plot_dir)


def normalize_strict_selection_config(config: dict) -> dict:
    """Return normalized strict checkpoint-selection settings.

    The training loop may still use a fast validation metric for quick feedback,
    but paper-facing checkpoints should be selected by the same strict protocol
    used for final evaluation. This block enables a post-training strict subset
    or full-val re-evaluation over saved candidate checkpoints.
    """
    train_cfg = config.get("train", {})
    raw_cfg = train_cfg.get("strict_checkpoint_selection", {})
    if isinstance(raw_cfg, bool):
        raw_cfg = {"enabled": raw_cfg}
    elif raw_cfg is None:
        raw_cfg = {}
    elif not isinstance(raw_cfg, dict):
        raise TypeError("train.strict_checkpoint_selection must be a mapping, boolean, or null.")

    mode = str(raw_cfg.get("mode", "strict_subset")).lower()
    enabled = bool(raw_cfg.get("enabled", mode not in {"", "off", "none", "false", "disabled", "train_metric"}))
    max_samples = raw_cfg.get("max_samples", 32 if mode in {"strict_subset", "subset"} else None)
    if max_samples is not None and int(max_samples) <= 0:
        max_samples = None

    return {
        "enabled": enabled,
        "mode": mode,
        "metric_mode": str(raw_cfg.get("metric_mode", "fast_gpu")),
        "max_samples": max_samples,
        "batch_size": raw_cfg.get("batch_size"),
        "num_workers": raw_cfg.get("num_workers"),
        "pin_memory": raw_cfg.get("pin_memory"),
        "apply_nms": raw_cfg.get("apply_nms", train_cfg.get("apply_nms", True)),
        "nms_low_threshold": raw_cfg.get("nms_low_threshold", train_cfg.get("nms_low_threshold", 0.0)),
        "fast_metric_thresholds": raw_cfg.get("fast_metric_thresholds", train_cfg.get("fast_metric_thresholds", 49)),
        "fast_metric_tolerance_pixels": raw_cfg.get(
            "fast_metric_tolerance_pixels",
            train_cfg.get("fast_metric_tolerance_pixels", 2),
        ),
        "score": raw_cfg.get("score", {"mode": "ODS"}),
        "replace_best": bool(raw_cfg.get("replace_best", True)),
        "output_name": str(raw_cfg.get("output_name", "best_strict.pt")),
        "candidate_dir": str(raw_cfg.get("candidate_dir", "eval_candidates")),
        "include_last": bool(raw_cfg.get("include_last", True)),
        "include_validation_metric_best": bool(
            raw_cfg.get(
                "include_validation_metric_best",
                raw_cfg.get("include_train_metric_best", True),
            )
        ),
    }


def positive_optional_float(value) -> float | None:
    """Return a positive float config value, or None when disabled."""
    if value is None:
        return None
    parsed = float(value)
    return parsed if parsed > 0 else None


def select_checkpoint_with_strict_eval(
    *,
    model: torch.nn.Module,
    base_config: dict,
    run_paths,
    criterion: EdgeDetectionLoss,
    device: torch.device,
    selection_cfg: dict,
) -> None:
    """Re-rank saved candidate checkpoints with strict or strict-subset metrics."""
    candidates = collect_selection_candidates(run_paths.checkpoints, selection_cfg)
    if not candidates:
        print("Strict checkpoint selection skipped: no candidate checkpoints were saved.")
        return

    strict_loader = make_strict_selection_loader(base_config, selection_cfg)
    rows: list[dict[str, object]] = []
    best_row: dict[str, object] | None = None
    best_checkpoint: dict | None = None
    best_score = -float("inf")
    score_config = strict_selection_score_config(base_config, selection_cfg)

    print(
        f"Running strict checkpoint selection over {len(candidates)} candidates "
        f"({len(strict_loader.dataset)} validation samples)."
    )
    for checkpoint_path in candidates:
        checkpoint = load_checkpoint(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"], strict=True)
        metrics = evaluate(
            model=model,
            loader=strict_loader,
            criterion=criterion,
            device=device,
            visual_dir=None,
            pred_dir=None,
            gate_dir=None,
            max_visual_samples=0,
            metric_mode=str(selection_cfg.get("metric_mode", "fast_gpu")),
            fast_metric_thresholds=int(selection_cfg.get("fast_metric_thresholds", 49)),
            fast_metric_tolerance_pixels=int(selection_cfg.get("fast_metric_tolerance_pixels", 2)),
            apply_nms=bool(selection_cfg.get("apply_nms", True)),
            nms_low_threshold=float(selection_cfg.get("nms_low_threshold", 0.0)),
            diagnostic_interval=int(selection_cfg.get("diagnostic_interval", base_config.get("train", {}).get("diagnostic_interval", 10))),
        )
        score = checkpoint_score(metrics, score_config)
        row = {
            "candidate": str(checkpoint_path),
            "source_epoch": int(checkpoint.get("epoch", -1)),
            "strict_selection_score": score,
            **metrics,
        }
        rows.append(row)
        print(
            f"Strict selection candidate epoch={row['source_epoch']}: "
            f"score={score:.4f}, ODS={float(metrics.get('ODS', 0.0)):.4f}, "
            f"OIS={float(metrics.get('OIS', 0.0)):.4f}, AP={float(metrics.get('AP', 0.0)):.4f}"
        )
        if score > best_score:
            best_score = score
            best_row = row
            best_checkpoint = checkpoint

    write_strict_selection_report(run_paths.logs / "strict_checkpoint_selection.csv", rows)
    if best_checkpoint is None or best_row is None:
        print("Strict checkpoint selection skipped: no checkpoint could be evaluated.")
        return

    selected = dict(best_checkpoint)
    selected["metrics"] = {
        key: value for key, value in best_row.items() if key not in {"candidate", "source_epoch"}
    }
    selected["strict_checkpoint_selection"] = {
        "candidate": str(best_row["candidate"]),
        "source_epoch": int(best_row["source_epoch"]),
        "score": float(best_row["strict_selection_score"]),
        "metric_mode": str(selection_cfg.get("metric_mode", "strict")),
        "max_samples": selection_cfg.get("max_samples"),
        "replace_best": bool(selection_cfg.get("replace_best", True)),
    }
    output_path = run_paths.checkpoints / str(selection_cfg.get("output_name", "best_strict.pt"))
    save_checkpoint_atomic(selected, output_path)
    if bool(selection_cfg.get("replace_best", True)):
        save_checkpoint_atomic(selected, run_paths.checkpoints / "best.pt")

    write_strict_selection_summary(run_paths.logs / "strict_checkpoint_selection.md", best_row, output_path, selection_cfg)
    print(
        f"Selected strict checkpoint epoch={best_row['source_epoch']} with "
        f"score={float(best_row['strict_selection_score']):.4f}, "
        f"ODS={float(best_row.get('ODS', 0.0)):.4f}; wrote {output_path.name}"
        + (" and replaced best.pt." if bool(selection_cfg.get("replace_best", True)) else ".")
    )


def collect_selection_candidates(checkpoint_dir: Path, selection_cfg: dict) -> list[Path]:
    """Collect saved checkpoints for strict re-ranking, preserving epoch order."""
    candidate_dir = checkpoint_dir / str(selection_cfg.get("candidate_dir", "eval_candidates"))
    candidates: list[Path] = []
    if candidate_dir.exists():
        candidates.extend(sorted(candidate_dir.glob("epoch_*.pt")))
    # When per-epoch candidates are saved, the validation-best and last.pt are
    # usually copies of one of those epoch checkpoints. Falling back to them only
    # when no epoch candidates exist avoids repeated strict re-evaluation.
    if not candidates:
        if bool(selection_cfg.get("include_validation_metric_best", True)):
            validation_best = checkpoint_dir / "best_validation_metric.pt"
            legacy_best = checkpoint_dir / "best_train_metric.pt"
            candidates.append(validation_best if validation_best.exists() else legacy_best)
        if bool(selection_cfg.get("include_last", True)):
            candidates.append(checkpoint_dir / "last.pt")

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        if not path.exists():
            continue
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(path)
    return deduped


def make_strict_selection_loader(base_config: dict, selection_cfg: dict):
    """Build the validation loader used only for strict checkpoint selection."""
    dataset_cfg = base_config.get("dataset", {})
    loader_cfg = base_config.get("loader", {})
    dataset_update: dict[str, object] = {}
    max_samples = selection_cfg.get("max_samples")
    if max_samples is not None:
        dataset_update["max_eval_samples"] = int(max_samples)
        dataset_update["max_val_samples"] = int(max_samples)

    loader_update: dict[str, object] = {}
    if selection_cfg.get("batch_size") is not None:
        loader_update["batch_size"] = int(selection_cfg["batch_size"])
    else:
        loader_update["batch_size"] = int(loader_cfg.get("batch_size", 4))
    if selection_cfg.get("num_workers") is not None:
        loader_update["num_workers"] = int(selection_cfg["num_workers"])
    if selection_cfg.get("pin_memory") is not None:
        loader_update["pin_memory"] = bool(selection_cfg["pin_memory"])

    selection_config = deep_update(base_config, {"dataset": dataset_update, "loader": loader_update})
    dataset = make_dataset(
        selection_config,
        dataset_name=dataset_cfg["val_dataset"],
        split=dataset_cfg.get("val_split", "val"),
        training=False,
    )
    return make_loader(dataset, selection_config, shuffle=False)


def strict_selection_score_config(base_config: dict, selection_cfg: dict) -> dict:
    """Build a temporary config for scoring strict-selection metrics."""
    score_cfg = selection_cfg.get("score") or {"mode": "ODS"}
    if isinstance(score_cfg, str):
        score_cfg = {"mode": score_cfg}
    if not isinstance(score_cfg, dict):
        raise TypeError("train.strict_checkpoint_selection.score must be a mapping or metric name.")
    return deep_update(base_config, {"train": {"checkpoint_score": score_cfg}})


def write_strict_selection_report(path: Path, rows: list[dict[str, object]]) -> None:
    """Write per-candidate strict checkpoint selection metrics."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_strict_selection_summary(path: Path, best_row: dict[str, object], output_path: Path, selection_cfg: dict) -> None:
    """Write a small human-readable note describing checkpoint re-ranking."""
    metric_mode = str(selection_cfg.get("metric_mode", "strict"))
    strict_mode = metric_mode not in {"fast", "fast_gpu", "approximate"}
    title = "Strict Checkpoint Selection" if strict_mode else "Approximate Checkpoint Re-ranking"
    interpretation = (
        "This checkpoint was selected by the configured strict validation evaluator."
        if strict_mode
        else "This checkpoint was re-ranked by an approximate validation metric; run the near-official evaluator separately for paper-facing reporting."
    )
    lines = [
        f"# {title}",
        "",
        f"- Selected checkpoint: `{output_path}`",
        f"- Source candidate: `{best_row['candidate']}`",
        f"- Source epoch: `{best_row['source_epoch']}`",
        f"- Metric mode: `{metric_mode}`",
        f"- Max validation samples: `{selection_cfg.get('max_samples')}`",
        f"- Selection score: `{float(best_row['strict_selection_score']):.6f}`",
        f"- ODS: `{float(best_row.get('ODS', 0.0)):.6f}`",
        f"- OIS: `{float(best_row.get('OIS', 0.0)):.6f}`",
        f"- AP: `{float(best_row.get('AP', 0.0)):.6f}`",
        "",
        interpretation,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict, epochs: int):
    """Create an optional learning-rate scheduler from config."""
    train_cfg = config.get("train", {})
    name = str(train_cfg.get("scheduler", "none")).lower()
    if name in {"", "none", "off"}:
        return None
    if name == "cosine":
        t_max = int(train_cfg.get("scheduler_t_max_epochs", epochs))
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, t_max),
            eta_min=float(train_cfg.get("min_learning_rate", 1e-6)),
        )
    if name in {"multistep", "multi_step"}:
        raw_steps = train_cfg.get("scheduler_milestones", train_cfg.get("lr_steps", []))
        if isinstance(raw_steps, str):
            milestones = [int(item) for item in re.split(r"[,\-\s]+", raw_steps.strip()) if item]
        else:
            milestones = [int(item) for item in raw_steps]
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=float(train_cfg.get("scheduler_gamma", train_cfg.get("lr_gamma", 0.1))),
        )
    raise ValueError(f"Unsupported scheduler: {name}")


def checkpoint_score(metrics: dict, config: dict) -> float:
    """Return the validation score used for best-checkpoint selection.

    By default this preserves the original behavior and selects by ODS. Long
    RBCM searches can opt into a density-aware composite score so a checkpoint
    with slightly higher ODS but excessive texture density does not silently win.
    """
    score_cfg = config.get("train", {}).get("checkpoint_score", {}) or {}
    mode = str(score_cfg.get("mode", score_cfg.get("metric", "ODS"))).lower()
    ods = float(metrics.get("ODS", 0.0))
    if mode in {"ods", "f1", "default"}:
        return ods
    if mode in {"ois"}:
        return float(metrics.get("OIS", ods))
    if mode in {"ap"}:
        return float(metrics.get("AP", ods))
    if mode not in {"composite", "density_aware", "density-aware"}:
        raise ValueError(f"Unsupported checkpoint score mode: {mode}")

    score = (
        float(score_cfg.get("ods_weight", 1.0)) * ods
        + float(score_cfg.get("ois_weight", 0.0)) * float(metrics.get("OIS", 0.0))
        + float(score_cfg.get("ap_weight", 0.0)) * float(metrics.get("AP", 0.0))
    )
    density_metric = str(score_cfg.get("density_metric", "density_ratio"))
    density_ratio = float(metrics.get(density_metric, metrics.get("density_ratio", 0.0)))
    density_target = float(score_cfg.get("density_target", score_cfg.get("target_density_ratio", 2.0)))
    density_power = float(score_cfg.get("density_power", 1.0))
    high_excess = max(0.0, density_ratio - density_target)
    score -= float(score_cfg.get("density_penalty", 0.0)) * (high_excess ** density_power)

    min_density = score_cfg.get("min_density_ratio")
    if min_density is not None:
        low_excess = max(0.0, float(min_density) - density_ratio)
        score -= float(score_cfg.get("low_density_penalty", 0.0)) * (low_excess ** density_power)
    return score


if __name__ == "__main__":
    main(parse_args())
