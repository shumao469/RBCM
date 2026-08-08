"""Rebuild the seven machine-readable result tables used by manuscript V5.

The script contains no paper scores. It reads the canonical evaluator outputs
produced by the frozen formal checkpoints and writes compact CSV tables. The
generated files are ignored by Git; only this reconstruction logic is public.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]

MODE_LABELS = {
    "plain_identity": "Anchor",
    "no_surround": "No-surround",
    "conv_control": "Conv-control",
    "main_surround": "H-RBCM",
}
MODE_ORDER = ("plain_identity", "no_surround", "conv_control", "main_surround")

EXTERNAL_KEEP = (
    ("MultiCue->BIPED", "PiDiNet official Table-7 (MultiCue)"),
    ("MultiCue->UDED", "PiDiNet official Table-7 (MultiCue)"),
    ("NYUDv2->BIPED", "PiDiNet official Table-6 (NYUDv2 RGB)"),
    ("NYUDv2->BIPED", "CATS official (NYUDv2)"),
    ("NYUDv2->UDED", "PiDiNet official Table-6 (NYUDv2 RGB)"),
    ("NYUDv2->UDED", "CATS official (NYUDv2)"),
    ("BIPED->UDED", "TEED local reproduction (BIPED)"),
)

EXTERNAL_LABELS = {
    "PiDiNet official Table-7 (MultiCue)": "PiDiNet-MultiCue",
    "PiDiNet official Table-6 (NYUDv2 RGB)": "PiDiNet-NYUDv2",
    "CATS official (NYUDv2)": "CATS-NYUDv2",
    "TEED local reproduction (BIPED)": "TEED-local-BIPED",
}

PR_PROTOCOL_SHORT = {
    "fixed non-trained operator": "No training",
    "released BSDS+PASCAL checkpoint": "BSDS+PASCAL",
    "released BIPED checkpoint": "BIPED",
    "released BSDS checkpoint": "BSDS",
    "released MultiCue checkpoint": "MultiCue",
    "released NYUDv2 RGB checkpoint": "NYUDv2 RGB",
    "BIPED three-fixed-split protocol": "BIPED (3 splits)",
    "strict MultiCue 68/12/20 protocol": "MultiCue (68/12/20)",
    "strict NYUDv2 381/414/654 protocol": "NYUDv2 RGB (381/414/654)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        choices=("internal", "full"),
        default="full",
        help=(
            "Internal writes Tables 1-3 from released H-RBCM assets only. "
            "Full additionally requires separately obtained external evidence for Tables 4-5."
        ),
    )
    parser.add_argument(
        "--same-domain",
        type=Path,
        default=ROOT / "edge_outputs/rbcm/tables/strict_protocols/same_domain_strict.csv",
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
        "--nyud-transfer",
        type=Path,
        default=ROOT / "edge_outputs/rbcm/tables/strict_protocols/nyudv2_transfer_internal.csv",
    )
    parser.add_argument(
        "--biped-transfer",
        type=Path,
        default=ROOT / "edge_outputs/rbcm/tables/fair_generalization/h_all_modes.csv",
    )
    parser.add_argument(
        "--external-transfer",
        type=Path,
        default=ROOT / (
            "edge_outputs/rbcm/tables/requested_cross_domain_external/"
            "source_matched_learned.csv"
        ),
    )
    parser.add_argument(
        "--pr-manifest",
        type=Path,
        default=ROOT / (
            "edge_outputs/rbcm/figures/publication/pr_source_target_matrix/"
            "curve_manifest.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "edge_outputs/rbcm/tables/v5_manuscript",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write an empty V5 table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def f4(value: str | float) -> str:
    return f"{float(value):.4f}"


def normalize_dataset(name: str) -> str:
    return {"Multicue": "MultiCue", "multicue": "MultiCue"}.get(name, name)


def internal_rows(
    raw: Iterable[dict[str, str]],
    group_key: str,
    groups: tuple[str, ...],
    group_label: str,
) -> list[dict[str, str]]:
    lookup = {
        (normalize_dataset(row[group_key]), row["mode"]): row
        for row in raw
    }
    rows: list[dict[str, str]] = []
    for group in groups:
        for mode in MODE_ORDER:
            item = lookup[(group, mode)]
            rows.append(
                {
                    group_label: group,
                    "Method": MODE_LABELS[mode],
                    "ODS": f4(item["ODS"]),
                    "OIS": f4(item["OIS"]),
                    "AP": f4(item["AP"]),
                }
            )
    return rows


def table_2(
    biped: list[dict[str, str]],
    multicue: list[dict[str, str]],
    nyud: list[dict[str, str]],
) -> list[dict[str, str]]:
    sources = (
        (
            "BIPED",
            [row for row in biped if row["source_dataset"] == "BIPED" and row["target_dataset"] == "UDED"],
        ),
        ("MultiCue", [row for row in multicue if row["target_dataset"] == "UDED"]),
        ("NYUDv2", [row for row in nyud if row["target_dataset"] == "UDED"]),
    )
    rows: list[dict[str, str]] = []
    for source, source_rows in sources:
        by_mode = {row["mode"]: row for row in source_rows}
        for mode in MODE_ORDER:
            item = by_mode[mode]
            rows.append(
                {
                    "Training source": source,
                    "Method": MODE_LABELS[mode],
                    "ODS": f4(item["ODS"]),
                    "OIS": f4(item["OIS"]),
                    "AP": f4(item["AP"]),
                }
            )
    return rows


def table_4(raw: list[dict[str, str]]) -> list[dict[str, str]]:
    lookup = {(row["experiment"], row["external_method"]): row for row in raw}
    rows: list[dict[str, str]] = []
    for index, key in enumerate(EXTERNAL_KEEP, start=1):
        item = lookup[key]
        pair = f"C{index}"
        rows.extend(
            [
                {
                    "Comparison": pair,
                    "Transfer": item["experiment"].replace("->", " -> "),
                    "Method": "H-RBCM",
                    "ODS": f4(item["H_ODS"]),
                    "OIS": f4(item["H_OIS"]),
                    "AP": f4(item["H_AP"]),
                    "H wins": item["H_win_count"] + "/3",
                },
                {
                    "Comparison": pair,
                    "Transfer": item["experiment"].replace("->", " -> "),
                    "Method": EXTERNAL_LABELS[item["external_method"]],
                    "ODS": f4(item["external_ODS"]),
                    "OIS": f4(item["external_OIS"]),
                    "AP": f4(item["external_AP"]),
                    "H wins": "-",
                },
            ]
        )
    return rows


def pr_table(raw: list[dict[str, str]], panel: int) -> list[dict[str, str]]:
    selected = [row for row in raw if int(row["panel_index"]) == panel]
    ranked = sorted(selected, key=lambda row: float(row["display_f"]), reverse=True)
    rank_lookup = {row["model"]: str(index) for index, row in enumerate(ranked, start=1)}
    family_labels = {"traditional": "Traditional", "external": "External", "internal": "Internal"}
    rows: list[dict[str, str]] = []
    for item in sorted(selected, key=lambda row: int(row["order"])):
        rows.append(
            {
                "Rank": rank_lookup[item["model"]],
                "Method": item["model"].replace("No surround", "No-surround").replace("Conv control", "Conv-control"),
                "Type": family_labels[item["family"]],
                "Training source / protocol": PR_PROTOCOL_SHORT.get(
                    item["source_protocol"], item["source_protocol"]
                ),
                "F": f4(item["display_f"]),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    same = read_csv(args.same_domain.resolve())
    multicue = read_csv(args.multicue_transfer.resolve())
    nyud = read_csv(args.nyud_transfer.resolve())
    biped = read_csv(args.biped_transfer.resolve())
    tables = {
        "01_same_domain_ablation.csv": internal_rows(
            same, "dataset", ("BIPED", "MultiCue", "NYUDv2"), "Dataset"
        ),
        "02_three_sources_to_uded.csv": table_2(biped, multicue, nyud),
        "03_multicue_four_target_generalization.csv": internal_rows(
            multicue,
            "target_dataset",
            ("BIPED", "NYUDv2", "BSDS500", "UDED"),
            "Target",
        ),
    }
    if args.scope == "full":
        external = read_csv(args.external_transfer.resolve())
        pr_manifest = read_csv(args.pr_manifest.resolve())
        tables.update(
            {
                "04_selected_external_cross_domain.csv": table_4(external),
                "05a_biped_same_target_pr_models.csv": pr_table(pr_manifest, 1),
                "05b_multicue_same_target_pr_models.csv": pr_table(pr_manifest, 7),
                "05c_nyudv2_same_target_pr_models.csv": pr_table(pr_manifest, 13),
            }
        )

    output = args.output_dir.resolve()
    for filename, rows in tables.items():
        write_csv(output / filename, rows)
    counts = ", ".join(f"{name}={len(rows)}" for name, rows in tables.items())
    print(f"Wrote manuscript V5 result tables to {output}: {counts}")


if __name__ == "__main__":
    main()
