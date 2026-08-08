"""Audit the formal edge-detection pipeline without retraining.

The audit is deliberately conservative: it reports protocol limitations as
warnings and reserves errors for conditions that can silently change scores or
make a paper-facing result incomplete.  It checks normalized datasets, split
membership, GT semantics, checkpoints, calibration candidates, stored
near-official summaries, and AP aggregation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from edge_model.core.checkpoint_io import load_checkpoint  # noqa: E402
from edge_model.data.build import make_dataset  # noqa: E402
from edge_model.engine.metrics import average_precision_from_curve  # noqa: E402
from edge_model.models.build import build_model  # noqa: E402
from scripts.experiments.calibrate import parse_ring  # noqa: E402


CONFIGS = {
    "BIPED": PROJECT_ROOT / "edge_model" / "configs" / "rbcm" / "biped.yaml",
    "Multicue": PROJECT_ROOT / "edge_model" / "configs" / "rbcm" / "multicue_strict.yaml",
    # The strict 381/414/654 split is the only paper-facing NYUDv2 protocol.
    "NYUDv2": PROJECT_ROOT / "edge_model" / "configs" / "rbcm" / "nyudv2_strict.yaml",
}

DEVELOPMENT_WEIGHT_ROOTS = {
    "BIPED": PROJECT_ROOT / "edge_outputs" / "rbcm" / "checkpoints" / "biped",
    "Multicue": PROJECT_ROOT / "edge_outputs" / "rbcm" / "checkpoints" / "multicue" / "main",
    "NYUDv2": PROJECT_ROOT / "edge_outputs" / "rbcm" / "checkpoints" / "nyudv2" / "main",
}

EXPECTED_EVAL_GT = {
    "BIPED": ("edge", "soft"),
    "Multicue": ("soft_vote", "soft"),
    "NYUDv2": ("edge", "binary"),
}

VALID_MODES = {"plain_identity", "main_surround", "no_surround", "conv_control"}


@dataclass
class Issue:
    severity: str
    area: str
    code: str
    message: str
    path: str = ""
    score_impact: str = "none"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "rbcm" / "audits" / "full_pipeline_20260723",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=None,
        help=(
            "Checkpoint package root. Accepts the public pretrained/ layout or "
            "the development edge_outputs/rbcm/checkpoints layout. Auto-detect by default."
        ),
    )
    parser.add_argument(
        "--require-results",
        action="store_true",
        help="Treat missing generated result tables as errors after a full reproduction run.",
    )
    parser.add_argument(
        "--train-dimension-samples",
        type=int,
        default=512,
        help="Maximum train pairs per dataset whose raw image/GT dimensions are checked.",
    )
    return parser.parse_args()


def resolve_weight_roots(checkpoint_root: Path | None) -> dict[str, Path]:
    if checkpoint_root is None:
        public_root = PROJECT_ROOT / "pretrained"
        if (public_root / "multicue_strict").is_dir():
            checkpoint_root = public_root
        else:
            return DEVELOPMENT_WEIGHT_ROOTS
    root = checkpoint_root.resolve()
    public = {
        "BIPED": root / "biped",
        "Multicue": root / "multicue_strict",
        "NYUDv2": root / "nyudv2_strict",
    }
    if all(path.is_dir() for path in public.values()):
        return public
    development = {
        "BIPED": root / "biped",
        "Multicue": root / "multicue" / "main",
        "NYUDv2": root / "nyudv2" / "main",
    }
    if all(path.is_dir() for path in development.values()):
        return development
    raise FileNotFoundError(
        "Checkpoint root is neither a public pretrained/ package nor a development "
        f"checkpoint tree: {root}"
    )


def add_issue(
    issues: list[Issue],
    severity: str,
    area: str,
    code: str,
    message: str,
    path: Path | str | None = None,
    score_impact: str = "none",
) -> None:
    issues.append(
        Issue(
            severity=severity,
            area=area,
            code=code,
            message=message,
            path=str(path or ""),
            score_impact=score_impact,
        )
    )


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError(f"Configuration is not a mapping: {path}")
    return loaded


def target_stats(path: Path) -> tuple[tuple[int, int], float, float, int]:
    with Image.open(path) as image:
        arr = np.asarray(image.convert("L"), dtype=np.uint8)
    return (int(arr.shape[0]), int(arr.shape[1])), float(arr.min()), float(arr.max()), int(np.count_nonzero(arr))


def raw_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return int(image.height), int(image.width)


def choose_evenly(items: list, limit: int) -> list:
    if limit <= 0 or len(items) <= limit:
        return items
    indices = np.linspace(0, len(items) - 1, limit, dtype=np.int64)
    return [items[int(index)] for index in indices]


def audit_datasets(issues: list[Issue], train_dimension_samples: int) -> dict:
    report: dict[str, dict] = {}
    for dataset_name, config_path in CONFIGS.items():
        entry: dict = {"config": str(config_path)}
        try:
            config = load_yaml(config_path)
            dataset_cfg = config.get("dataset", {})
            gt_cfg = dataset_cfg.get("gt", {}) or {}
            expected_variant, expected_mode = EXPECTED_EVAL_GT[dataset_name]
            actual_variant = str(gt_cfg.get("eval_variant", "edge"))
            actual_mode = str(gt_cfg.get("eval_mode", "binary"))
            entry["configured_gt"] = {
                "train_variant": str(gt_cfg.get("train_variant", "edge")),
                "eval_variant": actual_variant,
                "train_mode": str(gt_cfg.get("train_mode", "binary")),
                "eval_mode": actual_mode,
                "loss_weight_variant": str(gt_cfg.get("loss_weight_variant", "none")),
            }
            if (actual_variant, actual_mode) != (expected_variant, expected_mode):
                add_issue(
                    issues,
                    "ERROR",
                    "data",
                    "GT_SEMANTICS_MISMATCH",
                    f"{dataset_name} eval GT is {actual_variant}/{actual_mode}, expected {expected_variant}/{expected_mode}.",
                    config_path,
                    "direct",
                )

            datasets = {
                "train": make_dataset(config, dataset_name, "train", training=True),
                "val": make_dataset(config, dataset_name, "val", training=False),
                "test": make_dataset(config, dataset_name, "test", training=False),
            }
            split_ids = {name: [pair.sample_id for pair in ds.pairs] for name, ds in datasets.items()}
            entry["split_counts"] = {name: len(ids) for name, ids in split_ids.items()}
            entry["split_unique_counts"] = {name: len(set(ids)) for name, ids in split_ids.items()}
            for split, ids in split_ids.items():
                duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
                if duplicates:
                    add_issue(
                        issues,
                        "ERROR",
                        "data",
                        "DUPLICATE_SPLIT_IDS",
                        f"{dataset_name}/{split} has duplicate IDs: {duplicates[:10]}",
                        config_path,
                        "direct",
                    )

            overlaps = {}
            for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
                overlap = sorted(set(split_ids[left]) & set(split_ids[right]))
                overlaps[f"{left}_{right}"] = {"count": len(overlap), "first10": overlap[:10]}
                if overlap:
                    if left == "val" and right == "test":
                        add_issue(
                            issues,
                            "WARNING",
                            "protocol",
                            "VAL_TEST_OVERLAP",
                            f"{dataset_name} validation and test are identical/overlapping ({len(overlap)} IDs); checkpoint or calibration selection is not independent of reporting.",
                            config_path,
                            "optimistic_same_domain",
                        )
                    else:
                        add_issue(
                            issues,
                            "ERROR",
                            "data",
                            "TRAIN_EVAL_OVERLAP",
                            f"{dataset_name} {left}/{right} overlap by {len(overlap)} IDs.",
                            config_path,
                            "direct",
                        )
            entry["overlaps"] = overlaps

            pair_checks = []
            train_pairs = choose_evenly(list(datasets["train"].pairs), train_dimension_samples)
            eval_pairs = list(datasets["val"].pairs) + list(datasets["test"].pairs)
            unique_pairs = {}
            for pair in train_pairs + eval_pairs:
                unique_pairs[(str(pair.image_path), str(pair.edge_path))] = pair
            zero_gt = []
            size_mismatch = []
            missing = []
            variants = Counter()
            weight_variants = Counter()
            for pair in unique_pairs.values():
                variants[pair.edge_variant] += 1
                weight_variants[str(pair.weight_variant or "none")] += 1
                if not pair.image_path.exists() or not pair.edge_path.exists():
                    missing.append(pair.sample_id)
                    continue
                image_size = raw_size(pair.image_path)
                edge_size, _, edge_max, nonzero = target_stats(pair.edge_path)
                if image_size != edge_size:
                    size_mismatch.append((pair.sample_id, image_size, edge_size))
                if edge_max <= 0.0 or nonzero == 0:
                    zero_gt.append(pair.sample_id)
                if pair.weight_path is not None:
                    if not pair.weight_path.exists():
                        missing.append(f"{pair.sample_id}:weight")
                    elif raw_size(pair.weight_path) != edge_size:
                        size_mismatch.append((f"{pair.sample_id}:weight", raw_size(pair.weight_path), edge_size))
                pair_checks.append(pair.sample_id)
            entry["pair_audit"] = {
                "checked": len(pair_checks),
                "edge_variants": dict(variants),
                "weight_variants": dict(weight_variants),
                "missing_count": len(missing),
                "size_mismatch_count": len(size_mismatch),
                "zero_gt_count": len(zero_gt),
                "zero_gt_first10": zero_gt[:10],
            }
            if missing:
                add_issue(issues, "ERROR", "data", "MISSING_PAIR_FILE", f"{dataset_name} missing files: {missing[:10]}", config_path, "direct")
            if size_mismatch:
                add_issue(issues, "ERROR", "preprocess", "RAW_SIZE_MISMATCH", f"{dataset_name} raw image/GT size mismatches: {size_mismatch[:5]}", config_path, "direct")
            if zero_gt:
                add_issue(issues, "WARNING", "data", "EMPTY_GT", f"{dataset_name} has {len(zero_gt)} checked empty GT maps: {zero_gt[:10]}", config_path, "possible")
        except Exception as exc:
            entry["exception"] = repr(exc)
            add_issue(issues, "ERROR", "data", "DATASET_AUDIT_EXCEPTION", f"{dataset_name}: {exc}", config_path, "unknown")
        report[dataset_name] = entry
    return report


def audit_checkpoints(issues: list[Issue], weight_roots: dict[str, Path]) -> dict:
    report: dict[str, list[dict]] = {}
    for dataset_name, root in weight_roots.items():
        rows = []
        config = load_yaml(CONFIGS[dataset_name])
        reference_model = build_model(config)
        expected_keys = set(reference_model.state_dict())
        for checkpoint_path in sorted(root.rglob("*.pt")):
            row = {"path": str(checkpoint_path), "size": checkpoint_path.stat().st_size}
            selected_eval_package = (
                (checkpoint_path.parent / "protocol_manifest.json").is_file()
                and (checkpoint_path.parent / "checkpoint_provenance.json").is_file()
            )
            try:
                checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
                state = checkpoint.get("model")
                if not isinstance(state, dict):
                    raise KeyError("checkpoint['model'] is absent or invalid")
                actual_keys = set(state)
                row.update(
                    {
                        "epoch": int(checkpoint.get("epoch", -1)),
                        "model_keys": len(actual_keys),
                        "optimizer": checkpoint.get("optimizer") is not None,
                        "scheduler": checkpoint.get("scheduler") is not None,
                        "scaler": checkpoint.get("scaler") is not None,
                        "rng_state": checkpoint.get("rng_state") is not None,
                    }
                )
                if actual_keys != expected_keys:
                    missing = sorted(expected_keys - actual_keys)
                    extra = sorted(actual_keys - expected_keys)
                    raise RuntimeError(f"state keys differ; missing={missing[:5]}, extra={extra[:5]}")
                model = build_model(checkpoint.get("config", config))
                model.load_state_dict(state, strict=True)
                row["strict_load"] = True
                for key in ("optimizer", "scheduler", "rng_state"):
                    if checkpoint.get(key) is None:
                        add_issue(
                            issues,
                            "INFO" if selected_eval_package else "WARNING",
                            "checkpoint",
                            (
                                "EVAL_PACKAGE_NO_RESUME_STATE"
                                if selected_eval_package
                                else "INCOMPLETE_RESUME_STATE"
                            ),
                            (
                                f"{checkpoint_path.name} is a provenance-tracked "
                                f"evaluation package and has no {key} state."
                                if selected_eval_package
                                else f"{checkpoint_path.name} has no {key} state."
                            ),
                            checkpoint_path,
                            "none_for_eval",
                        )
            except Exception as exc:
                row["strict_load"] = False
                row["exception"] = repr(exc)
                add_issue(issues, "ERROR", "checkpoint", "CHECKPOINT_LOAD_FAILURE", str(exc), checkpoint_path, "direct")
            rows.append(row)
        if not rows:
            add_issue(issues, "ERROR", "checkpoint", "NO_CHECKPOINT", f"No checkpoints below {root}", root, "direct")
        report[dataset_name] = rows
    return report


def finite_or_blank(value: str) -> bool:
    if value is None or not str(value).strip():
        return True
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def audit_candidates(issues: list[Issue], weight_roots: dict[str, Path]) -> dict:
    report: dict[str, dict] = {}
    candidate_paths = sorted(
        csv_path
        for root in weight_roots.values()
        for csv_path in root.rglob("*candidates.csv")
    )
    for csv_path in candidate_paths:
        key = str(csv_path.relative_to(PROJECT_ROOT))
        entry = {"rows": 0, "modes": [], "splits": [], "rings": []}
        try:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            entry["rows"] = len(rows)
            modes = sorted({str(row.get("mode", "")) for row in rows})
            splits = sorted({str(row.get("split", "")) for row in rows})
            rings = sorted({str(row.get("ring", "")) for row in rows if str(row.get("ring", "")).strip()})
            entry.update({"modes": modes, "splits": splits, "rings": rings})
            invalid_modes = sorted(set(modes) - VALID_MODES)
            if invalid_modes:
                add_issue(issues, "ERROR", "calibration", "INVALID_MODE", f"Invalid modes: {invalid_modes}", csv_path, "direct")
            for ring in rings:
                parse_ring(ring)
            numeric_columns = {
                "AP", "ODS", "ODS_threshold", "OIS", "alpha", "bias", "edge_weight",
                "precision_at_ODS", "prob_weight", "recall_at_ODS", "sharpen", "temperature",
                "uncertainty_power",
            }
            bad_cells = [
                (index + 2, column, row.get(column, ""))
                for index, row in enumerate(rows)
                for column in numeric_columns
                if column in row and not finite_or_blank(row.get(column, ""))
            ]
            if bad_cells:
                add_issue(issues, "ERROR", "calibration", "NONFINITE_CANDIDATE", f"Invalid numeric cells: {bad_cells[:10]}", csv_path, "direct")
            if "val" in splits and "test" in splits:
                val_signatures = {
                    tuple((column, row.get(column, "")) for column in sorted(row) if column != "split")
                    for row in rows if row.get("split") == "val"
                }
                test_signatures = {
                    tuple((column, row.get(column, "")) for column in sorted(row) if column != "split")
                    for row in rows if row.get("split") == "test"
                }
                if val_signatures == test_signatures and val_signatures:
                    add_issue(
                        issues,
                        "WARNING",
                        "protocol",
                        "IDENTICAL_VAL_TEST_CANDIDATES",
                        "Validation and test calibration rows are identical; this is expected only when val and test use the same samples.",
                        csv_path,
                        "optimistic_same_domain",
                    )
        except Exception as exc:
            entry["exception"] = repr(exc)
            add_issue(issues, "ERROR", "calibration", "CANDIDATE_AUDIT_EXCEPTION", str(exc), csv_path, "unknown")
        report[key] = entry
    return report


def audit_summaries(issues: list[Issue]) -> dict:
    report = {"summary_files": 0, "with_complete_count_metadata": 0, "incomplete": [], "protocols": {}}
    protocol_counter: Counter[str] = Counter()
    diagnostic_root = (PROJECT_ROOT / "results" / "rbcm" / "audits").resolve()
    for summary_path in sorted((PROJECT_ROOT / "results" / "rbcm").rglob("summary.json")):
        # Audit regression fixtures intentionally exercise incomplete/extra
        # prediction handling. They are diagnostics, not stored formal
        # evaluations, and must not make the audit fail on its own fixtures.
        if diagnostic_root in summary_path.resolve().parents:
            continue
        report["summary_files"] += 1
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                continue
            missing_count = data.get("missing_count")
            extra_count = data.get("extra_count")
            requested = data.get("n_requested")
            matched = data.get("n_matched")
            if missing_count is not None and requested is not None and matched is not None:
                report["with_complete_count_metadata"] += 1
                if int(missing_count) != 0 or int(requested) != int(matched):
                    report["incomplete"].append(str(summary_path))
                    add_issue(issues, "ERROR", "evaluation", "INCOMPLETE_STORED_EVAL", f"requested={requested}, matched={matched}, missing={missing_count}", summary_path, "direct")
            if extra_count not in (None, 0):
                report["incomplete"].append(str(summary_path))
                add_issue(issues, "ERROR", "evaluation", "EXTRA_STORED_PREDICTIONS", f"extra_count={extra_count}", summary_path, "possible")
            signature = json.dumps(
                {
                    "apply_nms": data.get("apply_nms"),
                    "metric_backend": data.get("metric_backend"),
                    "gt_threshold": data.get("gt_threshold"),
                    "match_tolerance": data.get("match_tolerance"),
                    "threshold_count": data.get("threshold_count"),
                },
                sort_keys=True,
            )
            protocol_counter[signature] += 1
        except Exception as exc:
            add_issue(issues, "ERROR", "evaluation", "SUMMARY_PARSE_FAILURE", str(exc), summary_path, "unknown")
    report["protocols"] = dict(protocol_counter)
    if report["summary_files"] and report["with_complete_count_metadata"] < report["summary_files"]:
        add_issue(
            issues,
            "INFO",
            "evaluation",
            "LEGACY_SUMMARY_METADATA",
            f"{report['summary_files'] - report['with_complete_count_metadata']} legacy summary files predate complete-set metadata; current evaluator now enforces it.",
            PROJECT_ROOT / "results" / "rbcm",
            "none_historical",
        )
    return report


def audit_ap(issues: list[Issue], require_results: bool = False) -> dict:
    synthetic = [
        {"recall": 0.2, "precision": 0.9},
        {"recall": 0.2, "precision": 0.8},
        {"recall": 0.6, "precision": 0.7},
    ]
    value = average_precision_from_curve(synthetic)
    expected = 0.65
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1.0e-12):
        add_issue(issues, "ERROR", "metrics", "AP_REGRESSION", f"Synthetic AP={value}, expected={expected}", score_impact="direct")

    tables = {}
    for table_path in (
        PROJECT_ROOT / "edge_outputs" / "rbcm" / "tables" / "biped_stability_ap_corrected.csv",
        PROJECT_ROOT / "edge_outputs" / "rbcm" / "tables" / "self_test_ap_corrected.csv",
    ):
        rows = []
        if table_path.exists():
            with table_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            for index, row in enumerate(rows, start=2):
                for metric in ("ODS", "OIS", "AP"):
                    if metric in row and not finite_or_blank(row[metric]):
                        add_issue(issues, "ERROR", "metrics", "NONFINITE_TABLE_METRIC", f"row={index}, metric={metric}, value={row[metric]}", table_path, "direct")
        elif require_results:
            add_issue(issues, "ERROR", "metrics", "MISSING_CORRECTED_TABLE", "Corrected paper table is missing.", table_path, "direct")
        tables[str(table_path.relative_to(PROJECT_ROOT))] = {"rows": len(rows)}
    return {"synthetic_ap": value, "synthetic_expected": expected, "tables": tables}


def render_report(report: dict, issues: list[Issue], chinese: bool) -> str:
    counts = Counter(issue.severity for issue in issues)
    title = "# 边缘检测全链路完整性审计" if chinese else "# Edge Pipeline Integrity Audit"
    lines = [title, ""]
    if chinese:
        lines.extend(
            [
                f"- 审计错误：{counts.get('ERROR', 0)}",
                f"- 协议警告：{counts.get('WARNING', 0)}",
                f"- 信息提示：{counts.get('INFO', 0)}",
                "- 说明：ERROR 表示可能直接改变分数或破坏复现；WARNING 表示已知协议限制；INFO 不影响当前分数。",
            ]
        )
    else:
        lines.extend(
            [
                f"- Errors: {counts.get('ERROR', 0)}",
                f"- Protocol warnings: {counts.get('WARNING', 0)}",
                f"- Informational notes: {counts.get('INFO', 0)}",
                "- Interpretation: ERROR can change scores or break reproduction; WARNING is a disclosed protocol limitation; INFO does not change current scores.",
            ]
        )
    lines.extend(["", "## Issues" if not chinese else "## 问题清单", ""])
    if not issues:
        lines.append("No issues detected." if not chinese else "未检测到问题。")
    for issue in issues:
        lines.append(f"- **{issue.severity} / {issue.code}** [{issue.area}] {issue.message}")
        if issue.path:
            lines.append(f"  - Path: `{issue.path}`")
        lines.append(f"  - Score impact: `{issue.score_impact}`")
    lines.extend(["", "## Dataset Summary" if not chinese else "## 数据集摘要", ""])
    for name, entry in report.get("datasets", {}).items():
        lines.append(f"### {name}")
        lines.append(f"- Splits: `{entry.get('split_counts', {})}`")
        lines.append(f"- Configured GT: `{entry.get('configured_gt', {})}`")
        lines.append(f"- Pair audit: `{entry.get('pair_audit', {})}`")
        lines.append("")
    lines.extend(["## Reproduction Assets" if not chinese else "## 复现资产", ""])
    for name, rows in report.get("checkpoints", {}).items():
        loaded = sum(bool(row.get("strict_load")) for row in rows)
        lines.append(f"- {name}: {loaded}/{len(rows)} checkpoints strict-loaded")
    lines.extend(
        [
            "",
            "## Protocol Boundary" if not chinese else "## 协议边界",
            "",
            (
                "Stored paper-facing scores use the project's uniform near-official evaluator. The dilation backend is not the exact BSDS benchmark bipartite matcher. BIPED, strict MultiCue, and strict NYUDv2 use independent train/validation/test partitions. The legacy MultiCue and BSDS configurations reused reporting samples for validation; those records are excluded from the formal strict result."
                if not chinese
                else "已存论文分数使用工程内统一的 near-official 评估器。dilation 后端不是 BSDS 官方的精确二分匹配器。BIPED、严格 MultiCue 与严格 NYUDv2 均使用相互独立的训练集、验证集和测试集。旧 MultiCue 与 BSDS 配置曾复用验证和报告样本，这些记录不进入正式严格结果。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    issues: list[Issue] = []
    weight_roots = resolve_weight_roots(args.checkpoint_root)
    report = {
        "project_root": str(PROJECT_ROOT),
        "checkpoint_roots": {key: str(value) for key, value in weight_roots.items()},
        "datasets": audit_datasets(issues, int(args.train_dimension_samples)),
        "checkpoints": audit_checkpoints(issues, weight_roots),
        "calibration_candidates": audit_candidates(issues, weight_roots),
        "stored_evaluations": audit_summaries(issues),
        "ap_regression": audit_ap(issues, require_results=bool(args.require_results)),
    }
    report["issue_counts"] = dict(Counter(issue.severity for issue in issues))
    report["issues"] = [asdict(issue) for issue in issues]

    (output_dir / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output_dir / "issues.csv", [asdict(issue) for issue in issues])
    (output_dir / "README.md").write_text(render_report(report, issues, chinese=False), encoding="utf-8")
    (output_dir / "README.zh-CN.md").write_text(render_report(report, issues, chinese=True), encoding="utf-8")

    print(json.dumps(report["issue_counts"], indent=2))
    print(f"Audit written to {output_dir}")
    if any(issue.severity == "ERROR" for issue in issues):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
