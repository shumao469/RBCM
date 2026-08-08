"""Summarize BIPED official metrics across deterministic split repeats."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


MODES = ["plain_identity", "main_surround", "no_surround", "conv_control"]
METRICS = ["ODS", "OIS", "AP"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def read_best(path: Path) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty summary: {path}")
    return max(rows, key=lambda row: (float(row.get("ODS", "nan")), float(row.get("AP", "nan"))))


def fmt(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.6f}"


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    output_dir = (args.output_dir or run_root / "summary").resolve()
    rows: list[dict[str, object]] = []
    for split_dir in sorted((run_root / "official").glob("split*")):
        split = split_dir.name
        for mode in MODES:
            summary = split_dir / mode / "summary.csv"
            if not summary.exists():
                continue
            best = read_best(summary)
            row: dict[str, object] = {"split": split, "mode": mode, "summary_csv": str(summary)}
            for metric in METRICS:
                row[metric] = float(best[metric])
            rows.append(row)

    aggregate_rows: list[dict[str, object]] = []
    by_mode = {mode: [row for row in rows if row["mode"] == mode] for mode in MODES}
    for mode, mode_rows in by_mode.items():
        aggregate: dict[str, object] = {"mode": mode, "n": len(mode_rows)}
        for metric in METRICS:
            values = [float(row[metric]) for row in mode_rows]
            mean, std = mean_std(values)
            aggregate[f"{metric}_mean"] = mean
            aggregate[f"{metric}_std"] = std
            aggregate[f"{metric}_values"] = ";".join(fmt(value) for value in values)
        aggregate_rows.append(aggregate)

    delta_rows: list[dict[str, object]] = []
    split_names = sorted({str(row["split"]) for row in rows})
    for split in split_names:
        split_rows = {str(row["mode"]): row for row in rows if str(row["split"]) == split}
        main = split_rows.get("main_surround")
        if not main:
            continue
        for control in ["plain_identity", "no_surround", "conv_control"]:
            other = split_rows.get(control)
            if not other:
                continue
            row = {"split": split, "comparison": f"main_surround-minus-{control}"}
            for metric in METRICS:
                row[f"d_{metric}"] = float(main[metric]) - float(other[metric])
            delta_rows.append(row)
    for control in ["plain_identity", "no_surround", "conv_control"]:
        control_rows = [row for row in delta_rows if str(row["comparison"]).endswith(control)]
        if not control_rows:
            continue
        row = {"split": "mean_std", "comparison": f"main_surround-minus-{control}"}
        for metric in METRICS:
            values = [float(item[f"d_{metric}"]) for item in control_rows]
            mean, std = mean_std(values)
            row[f"d_{metric}_mean"] = mean
            row[f"d_{metric}_std"] = std
        delta_rows.append(row)

    write_csv(output_dir / "split_metrics.csv", rows)
    write_csv(output_dir / "aggregate_metrics.csv", aggregate_rows)
    write_csv(output_dir / "delta_metrics.csv", delta_rows)

    lines = ["# BIPED 3-Split Mean/Std", ""]
    lines.append("| mode | n | ODS mean | ODS std | OIS mean | OIS std | AP mean | AP std |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in aggregate_rows:
        lines.append(
            "| {mode} | {n} | {ods_m} | {ods_s} | {ois_m} | {ois_s} | {ap_m} | {ap_s} |".format(
                mode=row["mode"],
                n=row["n"],
                ods_m=fmt(float(row["ODS_mean"])),
                ods_s=fmt(float(row["ODS_std"])),
                ois_m=fmt(float(row["OIS_mean"])),
                ois_s=fmt(float(row["OIS_std"])),
                ap_m=fmt(float(row["AP_mean"])),
                ap_s=fmt(float(row["AP_std"])),
            )
        )
    lines.extend(["", "## Main Minus Controls", ""])
    lines.append("| comparison | dODS mean | dODS std | dOIS mean | dOIS std | dAP mean | dAP std |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in delta_rows:
        if row["split"] != "mean_std":
            continue
        lines.append(
            "| {comparison} | {dods_m} | {dods_s} | {dois_m} | {dois_s} | {dap_m} | {dap_s} |".format(
                comparison=row["comparison"],
                dods_m=fmt(float(row["d_ODS_mean"])),
                dods_s=fmt(float(row["d_ODS_std"])),
                dois_m=fmt(float(row["d_OIS_mean"])),
                dois_s=fmt(float(row["d_OIS_std"])),
                dap_m=fmt(float(row["d_AP_mean"])),
                dap_s=fmt(float(row["d_AP_std"])),
            )
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
