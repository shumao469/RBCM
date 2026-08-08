"""Smoke-test the minimal H-RBCM paper release and selected checkpoints."""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

sys.dont_write_bytecode = True


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEVELOPMENT_RELEASE_ROOT = PROJECT_ROOT / "dist" / "release"
if (DEVELOPMENT_RELEASE_ROOT / "github" / "RBCM-Edge").is_dir():
    ROOT = DEVELOPMENT_RELEASE_ROOT / "github" / "RBCM-Edge"
    DEFAULT_CHECKPOINT_ROOT = (
        DEVELOPMENT_RELEASE_ROOT
        / "checkpoints"
        / "RBCM-Edge-Checkpoints"
        / "pretrained"
    )
else:
    ROOT = PROJECT_ROOT
    DEFAULT_CHECKPOINT_ROOT = ROOT / "pretrained"
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from edge_model.models.build import build_model  # noqa: E402
from scripts.experiments.calibrate import (  # noqa: E402
    Candidate,
    apply_candidate,
    edge_energy_from_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT,
    )
    parser.add_argument(
        "--dataset",
        choices=("all", "biped", "multicue", "nyudv2"),
        default="all",
        help="Selected strict checkpoint package to smoke-test.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional config override. Only valid with one selected dataset.",
    )
    return parser.parse_args()


def load_candidate(path: Path, mode: str) -> Candidate:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("mode") == mode and row.get("split", "val") == "val":
                return Candidate(
                    mode=mode,
                    ring=row["ring"],
                    alpha=float(row["alpha"]),
                    edge_weight=float(row["edge_weight"]),
                    prob_weight=float(row["prob_weight"]),
                    uncertainty_power=float(row["uncertainty_power"]),
                    temperature=float(row["temperature"]),
                    bias=float(row["bias"]),
                    sharpen=float(row["sharpen"]),
                )
    raise RuntimeError(f"No validation candidate for {mode} in {path}")


def smoke_checkpoint(
    checkpoint_root: Path,
    dataset: str,
    config_override: Path | None,
) -> str:
    if dataset == "biped":
        package_options = (Path("biped/main"),)
    else:
        package_options = (Path(f"{dataset}_strict"), Path(f"{dataset}/main"))
    package_root = next(
        (checkpoint_root / option for option in package_options if (checkpoint_root / option).is_dir()),
        checkpoint_root / package_options[0],
    )
    package_name = package_root.relative_to(checkpoint_root).as_posix()
    checkpoint = package_root / "best.pt"
    if dataset == "biped":
        candidate_names = (
            "calibration_candidates.csv",
            "calibration_main_candidates.csv",
        )
    elif dataset == "multicue":
        candidate_names = ("calibration_candidates.csv", "fixed_candidates.csv")
    else:
        candidate_names = ("fixed_candidates.csv", "calibration_candidates.csv")
    candidates = next(
        (
            package_root / name
            for name in candidate_names
            if (package_root / name).is_file()
        ),
        package_root / candidate_names[0],
    )
    if not checkpoint.exists() or not candidates.exists():
        raise FileNotFoundError(
            f"Extract RBCM-Edge-Checkpoints so pretrained/{package_name} exists."
        )

    config_path = (
        config_override
        if config_override is not None
        else ROOT / (
            f"edge_model/configs/rbcm/{dataset}_strict.yaml"
            if dataset != "biped"
            else "edge_model/configs/rbcm/biped.yaml"
        )
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = build_model(config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()

    torch.manual_seed(4517)
    image = torch.rand(1, 3, 64, 96)
    with torch.inference_mode():
        output = model(image)
    logits = output["logits"]
    probability = torch.sigmoid(logits).squeeze().cpu().numpy()
    if probability.shape != (64, 96) or not np.isfinite(probability).all():
        raise RuntimeError(f"Unexpected anchor output: {probability.shape}")

    image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    with tempfile.TemporaryDirectory(prefix="rbcm_release_smoke_") as temp_dir:
        image_path = Path(temp_dir) / "synthetic.png"
        Image.fromarray(
            np.clip(image_np * 255.0, 0.0, 255.0).astype(np.uint8),
            mode="RGB",
        ).save(image_path)
        energy = edge_energy_from_image(image_path)
    main_candidate = load_candidate(candidates, "main_surround")
    calibrated = apply_candidate(
        {
            "sample_id": "synthetic",
            "prob": probability,
            "edge": energy,
            "target": np.zeros_like(probability, dtype=np.float32),
        },
        main_candidate,
    )
    if calibrated.shape != probability.shape or not np.isfinite(calibrated).all():
        raise RuntimeError("Non-finite or shape-changing RBCM calibration output")

    params = sum(parameter.numel() for parameter in model.parameters())
    return (
        f"dataset={dataset} "
        f"checkpoint_epoch={payload.get('epoch')} "
        f"params={params} output_shape={probability.shape}"
    )


def main() -> None:
    args = parse_args()
    if args.config is not None and args.dataset == "all":
        raise ValueError("--config requires one selected dataset")

    mea_scripts = sorted((ROOT / "MEA_analysis").glob("*.py")) + sorted(
        (ROOT / "MEA_model").glob("*/plot_*.py")
    )
    if len(mea_scripts) < 17:
        raise RuntimeError(
            f"Incomplete formal MEA source set: found {len(mea_scripts)} scripts"
        )
    for script in mea_scripts:
        compile(script.read_text(encoding="utf-8"), str(script), "exec")

    datasets = ("biped", "multicue", "nyudv2") if args.dataset == "all" else (args.dataset,)
    reports = [
        smoke_checkpoint(args.checkpoint_root, dataset, args.config)
        for dataset in datasets
    ]
    print(
        "release_smoke=PASS "
        f"mea_scripts={len(mea_scripts)} "
        + " | ".join(reports)
    )


if __name__ == "__main__":
    main()
