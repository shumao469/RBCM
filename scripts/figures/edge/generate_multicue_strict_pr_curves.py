"""Generate missing MultiCue-strict cross-domain PR curves.

The frozen MultiCue checkpoint and validation-selected calibration candidates
are applied unchanged to each target test split.  Prediction maps are then
evaluated with the same target-specific local evaluator used by the formal
score tables.  No target-domain parameter search is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = (
    ROOT
    / "edge_outputs"
    / "rbcm"
    / "predictions"
    / "multicue_strict_seed4517_generalization5"
)
MODES = ("plain_identity", "main_surround", "no_surround", "conv_control")


@dataclass(frozen=True)
class TargetSpec:
    name: str
    official_root_flag: str
    official_root: Path
    split_file: str = "splits/test.txt"
    match_tolerance: float = 0.0075
    gt_threshold: float = 0.0
    eval_gt_variant: str = "edge"
    eval_gt_mode: str = "binary"


TARGETS = {
    "BIPED": TargetSpec(
        "BIPED", "--biped-root", ROOT / "edge_data" / "official_rbcm" / "BIPED"
    ),
    "Multicue": TargetSpec(
        "Multicue",
        "--multicue-root",
        ROOT / "edge_data" / "official_rbcm" / "Multicue",
        split_file="splits/strict_68_12_20/test.txt",
        gt_threshold=0.3,
        eval_gt_variant="soft_vote",
        eval_gt_mode="soft",
    ),
    "NYUDv2": TargetSpec(
        "NYUDv2",
        "--nyud-root",
        ROOT / "edge_data" / "official_rbcm" / "NYUDv2",
        match_tolerance=0.011,
    ),
    "BSDS500": TargetSpec(
        "BSDS500",
        "--bsds-root",
        ROOT / "edge_data" / "official_repro" / "BSDS500",
    ),
    "UDED": TargetSpec(
        "UDED", "--uded-root", ROOT / "edge_data" / "official_repro" / "UDED"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--thresholds", type=int, default=49)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=None,
        help=(
            "Directory containing best.pt, config.yaml, and calibration_candidates.csv, "
            "or the release pretrained root containing multicue_strict/. If omitted, "
            "use the formal local MultiCue checkpoint directory."
        ),
    )
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--compute-internal-metrics",
        action="store_true",
        help="Also compute the calibration-path metrics used by the V5 same-domain table.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_checkpoint_dir(args: argparse.Namespace) -> Path:
    if args.checkpoint_root is None:
        return (
            ROOT
            / "edge_outputs"
            / "rbcm"
            / "checkpoints"
            / "multicue"
            / "main"
        )
    root = args.checkpoint_root.resolve()
    nested = root / "multicue_strict"
    return nested if nested.is_dir() else root


def target_split_path(spec: TargetSpec) -> Path:
    return ROOT / "edge_data" / "official_rbcm" / spec.name / spec.split_file


def official_dir_name(args: argparse.Namespace) -> str:
    return f"official{int(args.thresholds)}"


def run(command: list[str], log_path: Path, done_path: Path, force: bool) -> None:
    if done_path.exists() and not force:
        print(f"[skip] {done_path}", flush=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "src"), env.get("PYTHONPATH", "")]
    )
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print("[run] " + " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as handle:
        handle.write("# " + " ".join(command) + "\n\n")
        handle.flush()
        subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def apply_checkpoint(args: argparse.Namespace, spec: TargetSpec) -> Path:
    output_dir = args.output_root / "apply" / spec.name
    split_file = target_split_path(spec)
    checkpoint_dir = resolve_checkpoint_dir(args)
    command = [
        args.python,
        str(ROOT / "scripts" / "experiments" / "evaluate_generalization.py"),
        "--config",
        str(checkpoint_dir / "config.yaml"),
        "--checkpoint",
        str(checkpoint_dir / "best.pt"),
        "--candidate-csv",
        str(checkpoint_dir / "calibration_candidates.csv"),
        "--output-dir",
        str(output_dir),
        "--source-label",
        "Multicue_strict_68_12_20",
        "--dataset-name",
        spec.name,
        "--split",
        "test",
        "--split-file",
        str(split_file),
        "--edge-data-root",
        str(ROOT / "edge_data" / "official_rbcm"),
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--thresholds",
        str(args.thresholds),
        "--match-tolerance",
        str(spec.match_tolerance),
        "--gt-threshold",
        str(spec.gt_threshold),
        "--eval-gt-variant",
        spec.eval_gt_variant,
        "--eval-gt-mode",
        spec.eval_gt_mode,
        "--nms-low-threshold",
        "0.02",
        "--modes",
        "main_surround",
        "no_surround",
        "conv_control",
        "--candidate-split",
        "val",
        "--save-predictions",
    ]
    if not args.compute_internal_metrics:
        command.append("--skip-internal-metrics")
    if args.max_samples is not None:
        command.extend(["--max-samples", str(args.max_samples)])
    summary_csv = output_dir / "summary.csv"
    metrics_complete = False
    if summary_csv.is_file():
        with summary_csv.open(newline="", encoding="utf-8-sig") as handle:
            existing = list(csv.DictReader(handle))
        metrics_complete = bool(existing) and all(
            str(row.get("metrics_skipped", "")).lower() == "false"
            and row.get("ODS", "")
            and row.get("OIS", "")
            and row.get("AP", "")
            for row in existing
        )
    run(
        command,
        args.output_root / "logs" / f"apply_{spec.name}.log",
        output_dir / "summary.json",
        args.force or (args.compute_internal_metrics and not metrics_complete),
    )
    return output_dir


def evaluate_mode(
    args: argparse.Namespace,
    spec: TargetSpec,
    apply_dir: Path,
    mode: str,
) -> Path:
    output_dir = args.output_root / official_dir_name(args) / spec.name / mode
    split_file = target_split_path(spec)
    command = [
        args.python,
        str(ROOT / "scripts" / "baselines" / "evaluate_official_edges.py"),
        "--dataset",
        spec.name,
        "--prediction-dir",
        str(apply_dir / "predictions" / mode),
        "--output-dir",
        str(output_dir),
        "--split-file",
        str(split_file),
        "--orientation",
        "as_is",
        "--apply-nms",
        "--metric-backend",
        "dilation",
        "--thresholds",
        str(args.thresholds),
        "--match-tolerance",
        str(spec.match_tolerance),
        "--gt-threshold",
        str(spec.gt_threshold),
        "--device",
        args.device,
        spec.official_root_flag,
        str(spec.official_root),
    ]
    if args.max_samples is not None:
        command.extend(["--max-samples", str(args.max_samples), "--allow-extra"])
    run(
        command,
        args.output_root / "logs" / f"official_{spec.name}_{mode}.log",
        output_dir / "summary.json",
        args.force,
    )
    return output_dir / "summary.json"


def build_summary(args: argparse.Namespace) -> None:
    rows: list[dict[str, object]] = []
    # Rebuild from every completed target so an incremental rerun cannot
    # truncate the canonical summary to only the targets requested this time.
    for target in TARGETS:
        for mode in MODES:
            path = args.output_root / official_dir_name(args) / target / mode / "summary.json"
            if not path.exists():
                continue
            report = json.loads(path.read_text(encoding="utf-8"))
            selected = report["selected"]
            curve_path = (
                args.output_root
                / official_dir_name(args)
                / target
                / mode
                / "pr_curve_as_is.csv"
            )
            try:
                curve_file = curve_path.relative_to(ROOT)
            except ValueError:
                curve_file = curve_path
            rows.append(
                {
                    "training_source": "Multicue",
                    "target_dataset": target,
                    "mode": mode,
                    "ODS": selected["ODS"],
                    "OIS": selected["OIS"],
                    "AP": selected["AP"],
                    "n_images": selected["n_images"],
                    "thresholds": args.thresholds,
                    "curve_file": str(curve_file).replace("\\", "/"),
                }
            )
    if not rows:
        return
    summary_path = args.output_root / f"generalization_summary_official{int(args.thresholds)}.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    for target in args.targets:
        spec = TARGETS[target]
        apply_dir = apply_checkpoint(args, spec)
        for mode in MODES:
            evaluate_mode(args, spec, apply_dir, mode)
    build_summary(args)
    print(f"Wrote MultiCue-strict PR evidence to {args.output_root}")


if __name__ == "__main__":
    main()
