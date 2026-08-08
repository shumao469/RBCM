"""Evaluation-only reproduction for the final RBCM BIPED/Multicue weights.

This driver intentionally does not train or tune anything. It consumes the
verified checkpoints and validation-frozen candidate CSVs from the public
``pretrained`` package or the equivalent development checkpoint tree.

Stages:
1. Re-apply the fixed candidates and evaluate BIPED 3-split stability.
2. Re-apply the fixed BIPED-selected and Multicue-selected candidates to the
   five target test datasets.
3. Compare regenerated metrics against the immutable local reference summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_TAG = "biped_multicue"
REFERENCE_ROOT = PROJECT_ROOT / "results" / "rbcm" / "reference"
REFERENCE_GENERALIZATION = REFERENCE_ROOT / "generalization_biped_multicue.csv"
REFERENCE_BIPED_3SPLIT = REFERENCE_ROOT / "biped_stability.csv"

MODES = ["plain_identity", "main_surround", "no_surround", "conv_control"]
TARGETS = ["BIPED", "Multicue", "NYUDv2", "BSDS500", "UDED"]
WEIGHT_ROOT = PROJECT_ROOT / "pretrained"

TARGET_SPLITS = {
    "BIPED": "splits/protocols/main/test.txt",
    "Multicue": "splits/strict_68_12_20/test.txt",
    "NYUDv2": "splits/strict_381_414/test.txt",
    "BSDS500": "splits/test.txt",
    "UDED": "splits/test.txt",
}


def biped_source_artifacts() -> tuple[Path, Path, list[Path]]:
    train_root = WEIGHT_ROOT / "biped" / "main"
    merged = train_root / "calibration_candidates.csv"
    candidates = (
        [merged]
        if merged.is_file()
        else [
            train_root / "calibration_main_candidates.csv",
            train_root / "calibration_control_candidates.csv",
        ]
    )
    return (
        train_root / "config.yaml",
        train_root / "best.pt",
        candidates,
    )


def multicue_source_artifacts() -> tuple[Path, Path, list[Path]]:
    public_root = WEIGHT_ROOT / "multicue_strict"
    train_root = public_root if public_root.is_dir() else WEIGHT_ROOT / "multicue" / "main"
    return (
        train_root / "config.yaml",
        train_root / "best.pt",
        [
            train_root / "calibration_candidates.csv"
        ],
    )


def biped_split_artifacts(split: int) -> tuple[Path, Path, list[Path]]:
    train_root = WEIGHT_ROOT / "biped" / f"split{split}"
    return (
        train_root / "config.yaml",
        train_root / "best.pt",
        [
            train_root / "calibration_candidates.csv"
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["smoke", "biped-3split", "generalization", "compare", "all"],
        default="all",
    )
    parser.add_argument("--run-tag", default=DEFAULT_RUN_TAG)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=PROJECT_ROOT / "pretrained",
        help="Public pretrained/ root or development edge_outputs/rbcm/checkpoints root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "edge_outputs" / "rbcm" / "predictions",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=("biped", "multicue"),
        default=["biped"],
        help="Frozen training sources to evaluate during the generalization stage.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--biped-batch-size", type=int, default=4)
    parser.add_argument("--multicue-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Regenerate completed outputs.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_root(args: argparse.Namespace) -> Path:
    return args.output_root.resolve() / args.run_tag


def log_dir(args: argparse.Namespace) -> Path:
    path = run_root(args) / "launcher_logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def py(args: argparse.Namespace, *parts: str | Path) -> list[str]:
    return [args.python, *[str(part) for part in parts]]


def run_command(
    args: argparse.Namespace,
    name: str,
    command: list[str],
    *,
    done_path: Path | None = None,
    cwd: Path = PROJECT_ROOT,
) -> None:
    if done_path is not None and done_path.exists() and not args.force:
        print(f"[skip] {name}: {done_path}", flush=True)
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir(args) / f"{stamp}_{name}.log"
    printable = " ".join(str(item) for item in command)
    print(f"[run] {name}", flush=True)
    print(f"+ {printable}", flush=True)
    if args.dry_run:
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), env.get("PYTHONPATH", "")]
    )
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    with log_path.open("w", encoding="utf-8", errors="replace") as handle:
        handle.write(f"# {name}\n# cwd={cwd}\n# command={printable}\n\n")
        handle.flush()
        subprocess.run(command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_paths(paths: list[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required RBCM release artifacts:\n" + "\n".join(map(str, missing)))


def choose_device(requested: str) -> str:
    if requested != "cuda":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    print("[warn] CUDA is not available locally; falling back to CPU for evaluation.", flush=True)
    return "cpu"


def validate_artifacts(args: argparse.Namespace) -> None:
    paths: list[Path] = []
    selected_sources = []
    if "biped" in args.sources:
        selected_sources.append(biped_source_artifacts())
    if "multicue" in args.sources:
        selected_sources.append(multicue_source_artifacts())
    for cfg, ckpt, csvs in selected_sources:
        paths.extend([cfg, ckpt, *csvs])
    if args.stage in {"biped-3split", "all"}:
        for split in range(3):
            cfg, ckpt, csvs = biped_split_artifacts(split)
            paths.extend([cfg, ckpt, *csvs])
    require_paths(paths)


def target_split_path(dataset: str) -> Path:
    return PROJECT_ROOT / "edge_data" / "official_rbcm" / dataset / TARGET_SPLITS[dataset]


def ensure_uded_layout(args: argparse.Namespace) -> None:
    target = PROJECT_ROOT / "edge_data" / "official_rbcm" / "UDED" / "splits" / "test.txt"
    require_paths([target])


def apply_candidates(
    args: argparse.Namespace,
    *,
    name: str,
    cfg_path: Path,
    ckpt_path: Path,
    candidate_csvs: list[Path],
    output_dir: Path,
    source_label: str,
    dataset: str,
    split_file: Path,
    edge_data_root: Path,
    input_size: int,
    batch_size: int,
) -> Path:
    summary = output_dir / "summary.csv"
    command = py(
        args,
        PROJECT_ROOT / "scripts" / "experiments" / "evaluate_generalization.py",
        "--config",
        cfg_path,
        "--checkpoint",
        ckpt_path,
        "--output-dir",
        output_dir,
        "--source-label",
        source_label,
        "--dataset-name",
        dataset,
        "--split",
        "test",
        "--split-file",
        split_file,
        "--edge-data-root",
        edge_data_root,
        "--input-size",
        str(input_size),
        "--device",
        args.device,
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(args.num_workers),
        "--thresholds",
        "49",
        "--candidate-split",
        "val",
        "--modes",
        "main_surround",
        "no_surround",
        "conv_control",
        "--save-predictions",
        "--skip-internal-metrics",
    )
    for csv_path in candidate_csvs:
        command.extend(["--candidate-csv", str(csv_path)])
    run_command(args, name, command, done_path=summary)
    return summary


def official_eval(
    args: argparse.Namespace,
    *,
    source_label: str,
    dataset: str,
    mode: str,
    pred_dir: Path,
    output_dir: Path,
    biped_root: Path | None = None,
) -> Path:
    summary = output_dir / "summary.json"
    command = py(
        args,
        PROJECT_ROOT / "scripts" / "baselines" / "evaluate_official_edges.py",
        "--dataset",
        dataset,
        "--prediction-dir",
        pred_dir,
        "--output-dir",
        output_dir,
        "--orientation",
        "as_is",
        "--apply-nms",
        "--metric-backend",
        "dilation",
        "--thresholds",
        "49",
        "--device",
        args.device,
    )
    official_repro = PROJECT_ROOT / "edge_data" / "official_repro"
    official_rbcm = PROJECT_ROOT / "edge_data" / "official_rbcm"
    if dataset == "BIPED":
        command.extend(["--biped-root", str(biped_root or official_repro / "BIPED")])
    elif dataset == "BSDS500":
        command.extend(["--bsds-root", str(official_repro / "BSDS500")])
    elif dataset == "Multicue":
        command.extend(["--multicue-root", str(official_rbcm / "Multicue"), "--gt-threshold", "0.3"])
    elif dataset == "NYUDv2":
        command.extend(["--nyud-root", str(official_rbcm / "NYUDv2"), "--match-tolerance", "0.011"])
    elif dataset == "UDED":
        command.extend(["--uded-root", str(official_repro / "UDED")])
    else:
        raise ValueError(dataset)
    run_command(args, f"official_{source_label}_{dataset}_{mode}", command, done_path=summary)
    return summary


def run_apply_and_official(
    args: argparse.Namespace,
    *,
    cfg_path: Path,
    ckpt_path: Path,
    candidate_csvs: list[Path],
    source_label: str,
    dataset: str,
    apply_root: Path,
    official_root: Path,
    edge_data_root: Path,
    input_size: int,
    batch_size: int,
    biped_root: Path | None = None,
) -> None:
    apply_candidates(
        args,
        name=f"apply_{source_label}_{dataset}",
        cfg_path=cfg_path,
        ckpt_path=ckpt_path,
        candidate_csvs=candidate_csvs,
        output_dir=apply_root / source_label / dataset,
        source_label=source_label,
        dataset=dataset,
        split_file=target_split_path(dataset),
        edge_data_root=edge_data_root,
        input_size=input_size,
        batch_size=batch_size,
    )
    for mode in MODES:
        official_eval(
            args,
            source_label=source_label,
            dataset=dataset,
            mode=mode,
            pred_dir=apply_root / source_label / dataset / "predictions" / mode,
            output_dir=official_root / source_label / dataset / mode,
            biped_root=biped_root,
        )


def stage_smoke(args: argparse.Namespace) -> None:
    validate_artifacts(args)
    run_command(
        args,
        "py_compile_h_eval_only",
        py(
            args,
            "-m",
            "py_compile",
            PROJECT_ROOT / "scripts" / "experiments" / "evaluate_generalization.py",
            PROJECT_ROOT / "scripts" / "experiments" / "prepare_biped_splits.py",
            PROJECT_ROOT / "scripts" / "experiments" / "summarize_biped_splits.py",
            PROJECT_ROOT / "scripts" / "baselines" / "evaluate_official_edges.py",
            Path(__file__),
        ),
        done_path=run_root(args) / ".smoke_py_compile_done",
    )
    if not args.dry_run:
        (run_root(args) / ".smoke_py_compile_done").write_text("ok\n", encoding="utf-8")


def stage_stability(args: argparse.Namespace) -> None:
    out_root = run_root(args) / "stability"
    data_root = PROJECT_ROOT / "edge_data" / "official_rbcm"
    biped_eval_root = PROJECT_ROOT / "edge_data" / "official_repro" / "BIPED"
    for split in range(3):
        cfg, ckpt, candidate_csvs = biped_split_artifacts(split)
        apply_dir = out_root / "apply" / f"split{split}"
        apply_candidates(
            args,
            name=f"apply_biped3split_s{split}",
            cfg_path=cfg,
            ckpt_path=ckpt,
            candidate_csvs=candidate_csvs,
            output_dir=apply_dir,
            source_label=f"split{split}",
            dataset="BIPED",
            split_file=(
                PROJECT_ROOT
                / "edge_data"
                / "official_rbcm"
                / "BIPED"
                / "splits"
                / "protocols"
                / f"split{split}"
                / "test.txt"
            ),
            edge_data_root=data_root,
            input_size=300,
            batch_size=args.biped_batch_size,
        )
        for mode in MODES:
            official_eval(
                args,
                source_label=f"split{split}",
                dataset="BIPED",
                mode=mode,
                pred_dir=apply_dir / "predictions" / mode,
                output_dir=out_root / "official" / f"split{split}" / mode,
                biped_root=biped_eval_root,
            )
    run_command(
        args,
        "summarize_stability",
        py(
            args,
            PROJECT_ROOT / "scripts" / "experiments" / "summarize_biped_splits.py",
            "--run-root",
            out_root,
            "--output-dir",
            out_root / "summary",
        ),
        done_path=out_root / "summary" / "aggregate_metrics.csv",
    )


def stage_generalization(args: argparse.Namespace) -> None:
    ensure_uded_layout(args)
    out_root = run_root(args) / "generalization"
    input_sizes = {"BIPED": 300, "Multicue": 500, "NYUDv2": 500, "BSDS500": 300, "UDED": 300}
    batch_sizes = {
        "BIPED": args.biped_batch_size,
        "Multicue": args.multicue_batch_size,
        "NYUDv2": max(1, min(args.biped_batch_size, 2)),
        "BSDS500": args.biped_batch_size,
        "UDED": args.biped_batch_size,
    }
    sources = {}
    if "biped" in args.sources:
        sources["biped_selected"] = biped_source_artifacts()
    if "multicue" in args.sources:
        sources["multicue_selected"] = multicue_source_artifacts()
    for source_label, (cfg, ckpt, candidate_csvs) in sources.items():
        for dataset in TARGETS:
            run_apply_and_official(
                args,
                cfg_path=cfg,
                ckpt_path=ckpt,
                candidate_csvs=candidate_csvs,
                source_label=source_label,
                dataset=dataset,
                apply_root=out_root / "apply",
                official_root=out_root / "official",
                edge_data_root=PROJECT_ROOT / "edge_data" / "official_rbcm",
                input_size=input_sizes[dataset],
                batch_size=batch_sizes[dataset],
            )
    write_official_summary(out_root / "official", out_root / "generalization_summary.csv")


def write_official_summary(official_root: Path, output_csv: Path) -> None:
    rows: list[dict[str, Any]] = []
    for summary_json in sorted(official_root.glob("*/*/*/summary.json")):
        source_label, dataset, mode = summary_json.parts[-4:-1]
        summary = read_json(summary_json)
        selected = dict(summary.get("selected", {}))
        rows.append(
            {
                **selected,
                "source_label": source_label,
                "source": source_label,
                "dataset": dataset,
                "mode": mode,
                "n_matched": summary.get("n_matched"),
            }
        )
    write_csv(output_csv, rows)


def _float_or_none(value: str | float | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _row_key(row: dict[str, str], key_fields: list[str]) -> tuple[str, ...]:
    values: list[str] = []
    for field in key_fields:
        if field == "source_label":
            values.append(row.get("source_label") or row.get("source") or "")
        else:
            values.append(row.get(field, ""))
    return tuple(values)


def compare_metric_tables(
    *,
    local_csv: Path,
    reference_csv: Path,
    key_fields: list[str],
    metrics: list[str],
    output_csv: Path,
) -> None:
    if not local_csv.exists() or not reference_csv.exists():
        print(f"[compare-skip] missing {local_csv} or {reference_csv}", flush=True)
        return
    local = {_row_key(row, key_fields): row for row in load_csv(local_csv)}
    reference = {_row_key(row, key_fields): row for row in load_csv(reference_csv)}
    rows: list[dict[str, Any]] = []
    for key in sorted(set(local) | set(reference)):
        row: dict[str, Any] = {field: value for field, value in zip(key_fields, key, strict=True)}
        local_row = local.get(key, {})
        reference_row = reference.get(key, {})
        row["local_present"] = bool(local_row)
        row["reference_present"] = bool(reference_row)
        for metric in metrics:
            local_value = _float_or_none(local_row.get(metric))
            reference_value = _float_or_none(reference_row.get(metric))
            row[f"local_{metric}"] = local_value
            row[f"reference_{metric}"] = reference_value
            row[f"delta_{metric}"] = (
                None if local_value is None or reference_value is None else local_value - reference_value
            )
        rows.append(row)
    write_csv(output_csv, rows)


def stage_compare(args: argparse.Namespace) -> None:
    out = run_root(args) / "comparison_to_server"
    compare_metric_tables(
        local_csv=run_root(args) / "stability" / "summary" / "aggregate_metrics.csv",
        reference_csv=REFERENCE_BIPED_3SPLIT,
        key_fields=["mode"],
        metrics=["ODS_mean", "OIS_mean", "AP_mean"],
        output_csv=out / "stability_aggregate_delta.csv",
    )
    compare_metric_tables(
        local_csv=run_root(args) / "generalization" / "generalization_summary.csv",
        reference_csv=REFERENCE_GENERALIZATION,
        key_fields=["source_label", "dataset", "mode"],
        metrics=["ODS", "OIS", "AP"],
        output_csv=out / "generalization_delta.csv",
    )


def write_protocol_note(args: argparse.Namespace) -> None:
    note = run_root(args) / "LOCAL_EVAL_ONLY_PROTOCOL.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        "\n".join(
            [
                "# Local RBCM Evaluation-Only Reproduction",
                "",
                "This run does not train, resume training, or search post-hoc candidates.",
                "",
                f"- run_tag: `{args.run_tag}`",
                f"- weight_root: `{WEIGHT_ROOT}`",
                f"- sources: `{', '.join(args.sources)}`",
                "- inputs: verified `config.yaml`, `best.pt`, and fixed calibration candidate CSV files",
                "- outputs: regenerated predictions, official-style metrics, and local-vs-reference delta CSV files",
                "",
                "Historical diagnostic runs under `backup/` are not used by this reproduction entry point.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    global WEIGHT_ROOT
    args = parse_args()
    WEIGHT_ROOT = args.checkpoint_root.resolve()
    args.device = choose_device(str(args.device))
    run_root(args).mkdir(parents=True, exist_ok=True)
    validate_artifacts(args)
    write_protocol_note(args)
    stages = [args.stage] if args.stage != "all" else ["smoke", "biped-3split", "generalization", "compare"]
    for stage in stages:
        print(f"\n=== STAGE {stage} ===", flush=True)
        if stage == "smoke":
            stage_smoke(args)
        elif stage == "biped-3split":
            stage_stability(args)
        elif stage == "generalization":
            stage_generalization(args)
        elif stage == "compare":
            stage_compare(args)
        else:
            raise ValueError(stage)
    print(f"\nDONE evaluation-only reproduction root={run_root(args)}", flush=True)


if __name__ == "__main__":
    main()
