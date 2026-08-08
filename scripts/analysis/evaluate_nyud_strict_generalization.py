"""Evaluate the frozen strict-NYUD H-RBCM checkpoint on five target datasets.

The checkpoint and all post-hoc candidates are selected on the independent
NYUDv2 validation split. This runner applies them unchanged to BIPED,
MultiCue, NYUDv2, BSDS500, and UDED, then invokes the shared saved-prediction
evaluator. No target dataset is used for candidate or checkpoint selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "edge_model" / "configs" / "rbcm" / "nyudv2_strict.yaml"
CHECKPOINT = (
    ROOT
    / "pretrained"
    / "nyudv2_strict"
    / "best.pt"
)
FORMAL_SUMMARY = ROOT / "pretrained" / "nyudv2_strict" / "formal_summary.json"
DEFAULT_RUN_TAG = "nyudv2_strict_seed4517_20260725_generalization5"
MODES = ("plain_identity", "main_surround", "no_surround", "conv_control")
CALIBRATED_MODES = ("main_surround", "no_surround", "conv_control")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    split_file: str
    expected_count: int
    match_tolerance: float
    gt_threshold: float
    official_root_flag: str
    official_root: Path


DATASETS = {
    "BIPED": DatasetSpec(
        "BIPED",
        "splits/test.txt",
        50,
        0.0075,
        0.0,
        "--biped-root",
        ROOT / "edge_data" / "official_repro" / "BIPED",
    ),
    "Multicue": DatasetSpec(
        "Multicue",
        "splits/test.txt",
        20,
        0.0075,
        0.3,
        "--multicue-root",
        ROOT / "edge_data" / "official_rbcm" / "Multicue",
    ),
    "NYUDv2": DatasetSpec(
        "NYUDv2",
        "splits/strict_381_414/test.txt",
        654,
        0.011,
        0.0,
        "--nyud-root",
        ROOT / "edge_data" / "official_rbcm" / "NYUDv2",
    ),
    "BSDS500": DatasetSpec(
        "BSDS500",
        "splits/test.txt",
        200,
        0.0075,
        0.0,
        "--bsds-root",
        ROOT / "edge_data" / "official_repro" / "BSDS500",
    ),
    "UDED": DatasetSpec(
        "UDED",
        "splits/test.txt",
        30,
        0.0075,
        0.0,
        "--uded-root",
        ROOT / "edge_data" / "official_repro" / "UDED",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default=DEFAULT_RUN_TAG)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--formal-summary", type=Path, default=FORMAL_SUMMARY)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--thresholds", type=int, default=99)
    parser.add_argument("--official-dir-name", default="official")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--compute-internal-metrics",
        action="store_true",
        help="Also compute the calibration-path metrics used by the V5 same-domain table.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run_root(args: argparse.Namespace) -> Path:
    return ROOT / "edge_outputs" / "rbcm" / "predictions" / args.run_tag


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nonempty_line_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def target_split_path(spec: DatasetSpec) -> Path:
    return ROOT / "edge_data" / "official_rbcm" / spec.name / spec.split_file


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_fixed_candidates(path: Path, formal_summary: Path) -> list[dict[str, Any]]:
    summary = load_json(formal_summary)
    rows: list[dict[str, Any]] = []
    for mode in CALIBRATED_MODES:
        candidate = dict(summary["selected"][mode]["candidate"])
        candidate["split"] = "val"
        candidate["selection_dataset"] = "NYUDv2"
        candidate["selection_count"] = int(summary["val_count"])
        try:
            selection_summary = str(formal_summary.relative_to(ROOT))
        except ValueError:
            selection_summary = str(formal_summary)
        candidate["selection_summary"] = selection_summary
        rows.append(candidate)
    write_csv(path, rows)
    return rows


def preflight(args: argparse.Namespace, selected: list[DatasetSpec]) -> Path:
    config = args.config.resolve()
    checkpoint = args.checkpoint.resolve()
    formal_summary = args.formal_summary.resolve()
    required = [config, checkpoint, formal_summary]
    for spec in selected:
        required.extend([target_split_path(spec), spec.official_root])
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required assets:\n" + "\n".join(missing))

    counts = {spec.name: nonempty_line_count(target_split_path(spec)) for spec in selected}
    wrong = {
        spec.name: (counts[spec.name], spec.expected_count)
        for spec in selected
        if counts[spec.name] != spec.expected_count
    }
    if wrong:
        raise RuntimeError(f"Unexpected target split counts: {wrong}")

    out = run_root(args)
    out.mkdir(parents=True, exist_ok=True)
    fixed_path = out / "fixed_candidates.csv"
    candidates = write_fixed_candidates(fixed_path, formal_summary)
    def display_path(path: Path) -> str:
        try:
            return str(path.relative_to(ROOT))
        except ValueError:
            return str(path)

    manifest = {
        "source": "NYUDv2 strict 381/414/654, seed 4517",
        "selection_policy": (
            "Checkpoint and candidates selected on the 414-image NYUDv2 "
            "validation split only; fixed unchanged on all five targets."
        ),
        "config": display_path(config),
        "config_sha256": sha256(config),
        "checkpoint": display_path(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "formal_selection_summary": display_path(formal_summary),
        "formal_selection_summary_sha256": sha256(formal_summary),
        "fixed_candidates": candidates,
        "targets": {
            spec.name: {
                "split_file": str(target_split_path(spec).relative_to(ROOT)),
                "split_sha256": sha256(target_split_path(spec)),
                "count": counts[spec.name],
                "match_tolerance": spec.match_tolerance,
                "gt_threshold": spec.gt_threshold,
                "thresholds": args.thresholds,
                "apply_nms": True,
                "metric_backend": "dilation",
            }
            for spec in selected
        },
    }
    manifest_name = (
        "protocol_manifest.json"
        if args.thresholds == 99 and args.official_dir_name == "official"
        else f"protocol_manifest_{args.official_dir_name}_{args.thresholds}.json"
    )
    manifest_path = out / manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return fixed_path


def run_command(
    args: argparse.Namespace,
    name: str,
    command: list[str],
    *,
    done_path: Path | None = None,
) -> None:
    if done_path is not None and done_path.exists() and not args.force:
        print(f"[skip] {name}: {done_path}", flush=True)
        return
    print(f"[run] {name}", flush=True)
    print("+ " + " ".join(command), flush=True)
    if args.dry_run:
        return
    logs = run_root(args) / "launcher_logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "src"), env.get("PYTHONPATH", "")])
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    with (logs / f"{name}.log").open("w", encoding="utf-8", errors="replace") as handle:
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


def apply_dataset(
    args: argparse.Namespace,
    spec: DatasetSpec,
    fixed_candidates: Path,
) -> Path:
    out = run_root(args) / "apply" / spec.name
    command = [
        args.python,
        str(ROOT / "scripts" / "experiments" / "evaluate_generalization.py"),
        "--config",
        str(args.config.resolve()),
        "--checkpoint",
        str(args.checkpoint.resolve()),
        "--candidate-csv",
        str(fixed_candidates),
        "--output-dir",
        str(out),
        "--source-label",
        "nyudv2_strict_seed4517",
        "--dataset-name",
        spec.name,
        "--split",
        "test",
        "--split-file",
        str(target_split_path(spec)),
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
        "0.0",
        "--eval-gt-variant",
        "edge",
        "--eval-gt-mode",
        "binary",
        "--nms-low-threshold",
        "0.02",
        "--modes",
        *CALIBRATED_MODES,
        "--candidate-split",
        "val",
        "--save-predictions",
    ]
    if not args.compute_internal_metrics:
        command.append("--skip-internal-metrics")
    if args.max_samples is not None:
        command.extend(["--max-samples", str(args.max_samples)])
    summary_csv = out / "summary.csv"
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
    done_path = (
        out / "summary.json"
        if not args.compute_internal_metrics or metrics_complete
        else None
    )
    run_command(args, f"apply_{spec.name}", command, done_path=done_path)
    return out


def evaluate_mode(
    args: argparse.Namespace,
    spec: DatasetSpec,
    apply_dir: Path,
    mode: str,
) -> Path:
    out = run_root(args) / args.official_dir_name / spec.name / mode
    command = [
        args.python,
        str(ROOT / "scripts" / "baselines" / "evaluate_official_edges.py"),
        "--dataset",
        spec.name,
        "--prediction-dir",
        str(apply_dir / "predictions" / mode),
        "--output-dir",
        str(out),
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
        command.extend(["--max-samples", str(args.max_samples)])
    run_command(
        args,
        f"official_{spec.name}_{mode}",
        command,
        done_path=out / "summary.json",
    )
    return out / "summary.json"


def build_summary(args: argparse.Namespace, selected: list[DatasetSpec]) -> Path:
    rows: list[dict[str, Any]] = []
    for spec in selected:
        for mode in MODES:
            path = (
                run_root(args)
                / args.official_dir_name
                / spec.name
                / mode
                / "summary.json"
            )
            if not path.exists():
                continue
            payload = load_json(path)
            metrics = dict(payload["selected"])
            rows.append(
                {
                    "source": "NYUDv2_strict",
                    "dataset": spec.name,
                    "mode": mode,
                    "n_matched": payload["n_matched"],
                    "match_tolerance": payload["match_tolerance"],
                    "gt_threshold": payload["gt_threshold"],
                    "thresholds": args.thresholds,
                    **metrics,
                }
            )
    summary_name = (
        "generalization_summary.csv"
        if args.thresholds == 99 and args.official_dir_name == "official"
        else f"generalization_summary_{args.official_dir_name}_{args.thresholds}.csv"
    )
    out = run_root(args) / summary_name
    if rows:
        write_csv(out, rows)
    return out


def main() -> None:
    args = parse_args()
    unknown = [name for name in args.datasets if name not in DATASETS]
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    selected = [DATASETS[name] for name in args.datasets]
    fixed_candidates = preflight(args, selected)
    for spec in selected:
        apply_dir = apply_dataset(args, spec, fixed_candidates)
        for mode in MODES:
            evaluate_mode(args, spec, apply_dir, mode)
    summary = build_summary(args, selected)
    print(f"[done] summary={summary}", flush=True)


if __name__ == "__main__":
    main()
