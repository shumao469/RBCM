"""Normalize regenerated V5 evaluator outputs for the manuscript table builder.

This script contains no paper scores. It combines the three frozen internal
sources after inference and official-style evaluation, and records hashes for
every input file. External baselines remain a separate, explicitly supplied
evidence package.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from edge_model.engine.metrics import average_precision_from_curve
MODES = ("plain_identity", "no_surround", "conv_control", "main_surround")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--biped-stability",
        type=Path,
        default=ROOT / (
            "edge_outputs/rbcm/predictions/biped_multicue/stability/"
            "summary/aggregate_metrics.csv"
        ),
    )
    parser.add_argument(
        "--biped-transfer",
        type=Path,
        default=ROOT / (
            "edge_outputs/rbcm/predictions/biped_multicue/generalization/"
            "generalization_summary.csv"
        ),
    )
    parser.add_argument(
        "--biped-stability-curves",
        type=Path,
        default=ROOT / (
            "edge_outputs/rbcm/predictions/biped_multicue/stability/official"
        ),
    )
    parser.add_argument(
        "--biped-transfer-curves",
        type=Path,
        default=ROOT / (
            "edge_outputs/rbcm/predictions/biped_multicue/generalization/official"
        ),
    )
    parser.add_argument(
        "--multicue-same-domain",
        type=Path,
        default=ROOT / (
            "edge_outputs/rbcm/predictions/multicue_strict_seed4517_same_domain99/"
            "generalization_summary_official99.csv"
        ),
    )
    parser.add_argument(
        "--multicue-transfer",
        type=Path,
        default=ROOT / (
            "edge_outputs/rbcm/predictions/multicue_strict_seed4517_generalization5/"
            "generalization_summary_official49.csv"
        ),
    )
    parser.add_argument(
        "--nyud-same-domain",
        type=Path,
        default=ROOT / (
            "edge_outputs/rbcm/predictions/"
            "nyudv2_strict_seed4517_20260725_generalization5/apply/NYUDv2/summary.csv"
        ),
    )
    parser.add_argument(
        "--nyud-transfer",
        type=Path,
        default=ROOT / (
            "edge_outputs/rbcm/predictions/"
            "nyudv2_strict_seed4517_20260725_generalization5/"
            "generalization_summary_official49_49.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "edge_outputs/rbcm/reproduction/v5/inputs",
    )
    parser.add_argument(
        "--multicue-reference",
        type=Path,
        default=ROOT / "edge_outputs/rbcm/checkpoints/multicue/main/formal_summary.json",
    )
    parser.add_argument(
        "--nyud-reference",
        type=Path,
        default=ROOT / "edge_outputs/rbcm/checkpoints/nyudv2/main/formal_summary.json",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty reproduction input: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_name(row: dict[str, str]) -> str:
    value = row.get("target_dataset") or row.get("dataset") or ""
    return "MultiCue" if value.lower() == "multicue" else value


def standardized_transfer(
    rows: list[dict[str, str]], source: str, *, source_label: str | None = None
) -> list[dict[str, object]]:
    selected = rows
    if source_label is not None:
        selected = [
            row
            for row in rows
            if (row.get("source_label") or row.get("source")) == source_label
        ]
    normalized: list[dict[str, object]] = []
    for row in selected:
        target = dataset_name(row)
        if not target or row.get("mode") not in MODES:
            continue
        normalized.append(
            {
                "source_dataset": source,
                "target_dataset": target,
                "mode": row["mode"],
                "ODS": row["ODS"],
                "OIS": row["OIS"],
                "AP": row["AP"],
                "n_images": row.get("n_images") or row.get("n_matched") or "",
                "thresholds": row.get("thresholds", ""),
                "metric_backend": row.get("metric_backend", "dilation"),
            }
        )
    return sorted(normalized, key=lambda row: (str(row["target_dataset"]), MODES.index(str(row["mode"]))))


def endpoint_ap(path: Path) -> float:
    rows = read_csv(path)
    curve = [
        {"recall": float(row["recall"]), "precision": float(row["precision"])}
        for row in rows
    ]
    return float(average_precision_from_curve(curve))


def biped_same_domain(
    rows: list[dict[str, str]], curve_root: Path
) -> tuple[list[dict[str, object]], list[Path]]:
    by_mode = {row["mode"]: row for row in rows}
    used_curves: list[Path] = []
    normalized = [
        {
            "dataset": "BIPED",
            "mode": mode,
            "ODS": by_mode[mode]["ODS_mean"],
            "OIS": by_mode[mode]["OIS_mean"],
            "AP": sum(
                endpoint_ap(curve_root / split / mode / "pr_curve_as_is.csv")
                for split in ("split0", "split1", "split2")
            )
            / 3.0,
            "n_images": 50,
            "n_runs": by_mode[mode].get("n", "3"),
        }
        for mode in MODES
    ]
    for mode in MODES:
        used_curves.extend(
            curve_root / split / mode / "pr_curve_as_is.csv"
            for split in ("split0", "split1", "split2")
        )
    return normalized, used_curves


def correct_biped_transfer_ap(
    rows: list[dict[str, object]], curve_root: Path
) -> tuple[list[dict[str, object]], list[Path]]:
    used_curves: list[Path] = []
    for row in rows:
        target = "Multicue" if row["target_dataset"] == "MultiCue" else str(row["target_dataset"])
        path = curve_root / "biped_selected" / target / str(row["mode"]) / "pr_curve_as_is.csv"
        row["AP"] = endpoint_ap(path)
        used_curves.append(path)
    return rows, used_curves


def one_source_same_domain(
    rows: list[dict[str, str]], dataset: str, n_images: int
) -> list[dict[str, object]]:
    standardized = standardized_transfer(rows, dataset)
    selected = [row for row in standardized if row["target_dataset"] == dataset]
    by_mode = {str(row["mode"]): row for row in selected}
    return [
        {
            "dataset": dataset,
            "mode": mode,
            "ODS": by_mode[mode]["ODS"],
            "OIS": by_mode[mode]["OIS"],
            "AP": by_mode[mode]["AP"],
            "n_images": by_mode[mode]["n_images"] or n_images,
            "n_runs": 1,
        }
        for mode in MODES
    ]


def require_complete(rows: list[dict[str, object]], expected: int, label: str) -> None:
    keys = {(str(row.get("target_dataset", row.get("dataset", ""))), str(row["mode"])) for row in rows}
    if len(rows) != expected or len(keys) != expected:
        raise RuntimeError(f"Incomplete {label}: rows={len(rows)}, unique dataset/mode={len(keys)}, expected={expected}")


def verify_same_domain_reference(
    rows: list[dict[str, object]], dataset: str, path: Path
) -> dict[str, float]:
    report = json.loads(path.read_text(encoding="utf-8"))
    expected: dict[str, dict[str, float]] = {}
    if dataset == "MultiCue":
        expected = {
            mode: report["modes"][mode]["selected"] for mode in MODES
        }
    elif dataset == "NYUDv2":
        expected = {
            "plain_identity": report["baseline"]["test"],
            **{
                mode: report["selected"][mode]["test"]
                for mode in MODES
                if mode != "plain_identity"
            },
        }
    observed = {
        str(row["mode"]): row for row in rows if row["dataset"] == dataset
    }
    deltas: dict[str, float] = {}
    for mode in MODES:
        for metric in ("ODS", "OIS", "AP"):
            delta = abs(float(observed[mode][metric]) - float(expected[mode][metric]))
            deltas[f"{mode}.{metric}"] = delta
            tolerance = 1.0e-4 if dataset == "MultiCue" else 5.0e-8
            if delta > tolerance:
                raise RuntimeError(
                    f"{dataset} regenerated {mode}/{metric} differs from frozen "
                    f"validation-selected reference by {delta:.3g}"
                )
        if dataset == "MultiCue":
            # CUDA kernels plus uint8 prediction export can move a score across
            # a fourth-decimal rounding boundary. Only after all three metrics
            # pass the explicit tolerance do we retain the frozen full-precision
            # value recorded with the released checkpoint.
            for metric in ("ODS", "OIS", "AP"):
                observed[mode][metric] = float(expected[mode][metric])
    return deltas


def main() -> None:
    args = parse_args()
    sources = {
        "biped_stability": args.biped_stability.resolve(),
        "biped_transfer": args.biped_transfer.resolve(),
        "multicue_same_domain": args.multicue_same_domain.resolve(),
        "multicue_transfer": args.multicue_transfer.resolve(),
        "nyud_same_domain": args.nyud_same_domain.resolve(),
        "nyud_transfer": args.nyud_transfer.resolve(),
    }
    raw = {name: read_csv(path) for name, path in sources.items()}

    same, biped_stability_curves = biped_same_domain(
        raw["biped_stability"], args.biped_stability_curves.resolve()
    )
    same.extend(one_source_same_domain(raw["multicue_same_domain"], "MultiCue", 20))
    same.extend(one_source_same_domain(raw["nyud_same_domain"], "NYUDv2", 654))
    biped = standardized_transfer(raw["biped_transfer"], "BIPED", source_label="biped_selected")
    biped, biped_transfer_curves = correct_biped_transfer_ap(
        biped, args.biped_transfer_curves.resolve()
    )
    multicue = standardized_transfer(raw["multicue_transfer"], "MultiCue")
    nyud = standardized_transfer(raw["nyud_transfer"], "NYUDv2")

    require_complete(same, 12, "same-domain ablation")
    require_complete(biped, 20, "BIPED transfer")
    require_complete(multicue, 20, "MultiCue transfer")
    require_complete(nyud, 20, "NYUDv2 transfer")
    reference_deltas = {
        "MultiCue": verify_same_domain_reference(
            same, "MultiCue", args.multicue_reference.resolve()
        ),
        "NYUDv2": verify_same_domain_reference(
            same, "NYUDv2", args.nyud_reference.resolve()
        ),
    }

    output = args.output_dir.resolve()
    write_csv(output / "same_domain.csv", same)
    write_csv(output / "biped_transfer.csv", biped)
    write_csv(output / "multicue_transfer.csv", multicue)
    write_csv(output / "nyud_transfer.csv", nyud)
    manifest = {
        "selection_boundary": (
            "All checkpoints and calibration candidates were frozen on source-domain "
            "validation data before these evaluator outputs were created."
        ),
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in sources.items()
        },
        "curve_inputs": [
            {"path": str(path.resolve()), "sha256": sha256(path.resolve())}
            for path in biped_stability_curves + biped_transfer_curves
        ],
        "same_domain_reference_max_abs_delta": {
            dataset: max(deltas.values()) for dataset, deltas in reference_deltas.items()
        },
        "canonicalization": (
            "Fresh MultiCue saved-map metrics must agree with the released frozen "
            "reference within 1e-4 before its full-precision values are used, avoiding "
            "hardware/PNG fourth-decimal boundary drift. NYUDv2 must match within 5e-8."
        ),
        "outputs": {
            "same_domain_rows": len(same),
            "biped_transfer_rows": len(biped),
            "multicue_transfer_rows": len(multicue),
            "nyud_transfer_rows": len(nyud),
        },
    }
    (output / "reproduction_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Wrote V5 normalized reproduction inputs to {output}")


if __name__ == "__main__":
    main()
