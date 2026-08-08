"""Run the V5 release from frozen checkpoints through tables and Figure 5.

The default ``all`` stage rebuilds every internal H-RBCM result that can be
computed from the released code, datasets, MEA inputs, and checkpoint package.
External-model Tables 4-5 require a separately supplied evidence package and
are enabled with ``--external-evidence-root``. No stage searches candidates on
test or target-domain data.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS = ROOT / "edge_outputs/rbcm/reproduction/v5/inputs"
DEFAULT_TABLES = ROOT / "edge_outputs/rbcm/reproduction/v5/tables"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("verify", "protocol", "inference", "inputs", "tables", "figure5", "audit", "all"),
        default="all",
    )
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "pretrained")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--external-evidence-root", type=Path, default=None)
    parser.add_argument("--skip-mea", action="store_true", help="Skip Figure 5 when authorized MEA inputs are unavailable.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the complete command plan without running it.")
    return parser.parse_args()


def run(args: argparse.Namespace, relative: str, *command_args: str) -> None:
    command = [sys.executable, str(ROOT / relative), *command_args]
    print("+", " ".join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def require(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Missing {label}: {resolved}")
    return resolved


def resolve_source_checkpoint_dir(checkpoint_root: Path, source: str) -> Path:
    """Resolve one frozen source checkpoint in release or development layouts."""
    root = require(checkpoint_root, "checkpoint package")
    candidates = {
        "multicue": (root / "multicue_strict", root / "multicue" / "main"),
        "nyudv2": (root / "nyudv2_strict", root / "nyudv2" / "main"),
    }
    if source not in candidates:
        raise ValueError(f"Unsupported source checkpoint: {source}")
    for candidate in candidates[source]:
        if (candidate / "best.pt").is_file():
            return candidate.resolve()
    if (root / "best.pt").is_file():
        return root
    searched = ", ".join(str(path) for path in candidates[source])
    raise FileNotFoundError(f"Missing frozen {source} checkpoint; searched: {searched}")


def optional_force(args: argparse.Namespace) -> tuple[str, ...]:
    return ("--force",) if args.force else ()


def stage_verify(args: argparse.Namespace) -> None:
    if (ROOT / "MANIFEST_SHA256.csv").is_file():
        run(args, "scripts/release/verify_paper_release.py", "--code-root", str(ROOT))
    else:
        print("[info] Full development tree detected; release-manifest membership check is deferred to the generated clean tree.")
    run(
        args,
        "scripts/release/smoke_paper_release.py",
        "--checkpoint-root",
        str(require(args.checkpoint_root, "checkpoint package")),
        "--dataset",
        "all",
    )


def stage_protocol(args: argparse.Namespace) -> None:
    run(
        args,
        "scripts/checks/audit_multicue_strict_protocol.py",
        "--config",
        "edge_model/configs/rbcm/multicue_strict.yaml",
    )
    run(
        args,
        "scripts/checks/audit_nyud_strict_protocol.py",
        "--config",
        "edge_model/configs/rbcm/nyudv2_strict.yaml",
    )
    run(args, "scripts/checks/run_rbcm_paper_preflight.py", "--check-data")


def stage_inference(args: argparse.Namespace) -> None:
    checkpoint_root = require(args.checkpoint_root, "checkpoint package")
    multicue_checkpoint = resolve_source_checkpoint_dir(checkpoint_root, "multicue")
    nyud_checkpoint = resolve_source_checkpoint_dir(checkpoint_root, "nyudv2")
    biped_common = (
        "--checkpoint-root",
        str(checkpoint_root),
        "--device",
        args.device,
        "--num-workers",
        str(args.num_workers),
    )
    run(
        args,
        "scripts/experiments/evaluate_biped_multicue.py",
        "--stage",
        "all",
        *biped_common,
        "--sources",
        "biped",
        "--biped-batch-size",
        str(args.batch_size),
        *optional_force(args),
    )
    run(
        args,
        "scripts/figures/edge/generate_multicue_strict_pr_curves.py",
        "--checkpoint-root",
        str(multicue_checkpoint),
        "--device",
        args.device,
        "--num-workers",
        str(args.num_workers),
        "--batch-size",
        str(args.batch_size),
        "--thresholds",
        "49",
        "--targets",
        "BIPED",
        "Multicue",
        "NYUDv2",
        "BSDS500",
        "UDED",
        *optional_force(args),
    )
    run(
        args,
        "scripts/figures/edge/generate_multicue_strict_pr_curves.py",
        "--checkpoint-root",
        str(multicue_checkpoint),
        "--device",
        args.device,
        "--num-workers",
        str(args.num_workers),
        "--batch-size",
        str(args.batch_size),
        "--thresholds",
        "99",
        "--targets",
        "Multicue",
        "--output-root",
        "edge_outputs/rbcm/predictions/multicue_strict_seed4517_same_domain99",
        *optional_force(args),
    )
    nyud_common = (
        "--config",
        "edge_model/configs/rbcm/nyudv2_strict.yaml",
        "--checkpoint",
        str(nyud_checkpoint / "best.pt"),
        "--formal-summary",
        str(nyud_checkpoint / "formal_summary.json"),
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
    )
    run(
        args,
        "scripts/analysis/evaluate_nyud_strict_generalization.py",
        *nyud_common,
        "--thresholds",
        "49",
        "--official-dir-name",
        "official49",
        "--datasets",
        "BIPED",
        "Multicue",
        "NYUDv2",
        "BSDS500",
        "UDED",
        *optional_force(args),
    )
    run(
        args,
        "scripts/analysis/evaluate_nyud_strict_generalization.py",
        *nyud_common,
        "--thresholds",
        "99",
        "--official-dir-name",
        "official",
        "--compute-internal-metrics",
        "--datasets",
        "NYUDv2",
        *optional_force(args),
    )


def stage_inputs(args: argparse.Namespace) -> None:
    multicue_checkpoint = resolve_source_checkpoint_dir(args.checkpoint_root, "multicue")
    nyud_checkpoint = resolve_source_checkpoint_dir(args.checkpoint_root, "nyudv2")
    run(
        args,
        "scripts/analysis/build_v5_reproduction_inputs.py",
        "--multicue-reference",
        str(multicue_checkpoint / "formal_summary.json"),
        "--nyud-reference",
        str(nyud_checkpoint / "formal_summary.json"),
    )


def stage_tables(args: argparse.Namespace) -> None:
    command = (
        "--scope",
        "internal",
        "--same-domain",
        str(DEFAULT_INPUTS / "same_domain.csv"),
        "--biped-transfer",
        str(DEFAULT_INPUTS / "biped_transfer.csv"),
        "--multicue-transfer",
        str(DEFAULT_INPUTS / "multicue_transfer.csv"),
        "--nyud-transfer",
        str(DEFAULT_INPUTS / "nyud_transfer.csv"),
        "--output-dir",
        str(DEFAULT_TABLES),
    )
    run(args, "scripts/analysis/build_v5_result_tables.py", *command)
    if args.external_evidence_root is not None:
        external_root = require(args.external_evidence_root, "external evidence package")
        run(
            args,
            "scripts/analysis/build_v5_result_tables.py",
            *command[2:-2],
            "--scope",
            "full",
            "--external-transfer",
            str(require(external_root / "source_matched_learned.csv", "external comparison table")),
            "--pr-manifest",
            str(require(external_root / "curve_manifest.csv", "external PR manifest")),
            "--output-dir",
            str(DEFAULT_TABLES),
        )


def stage_figure5(args: argparse.Namespace) -> None:
    if args.skip_mea:
        print("[skip] Figure 5 was explicitly skipped.")
        return
    run(args, "scripts/analysis/run_mea_pipeline.py")
    run(
        args,
        "scripts/analysis/reproduce_figure5_relative_statistics.py",
        "--candidate-csv",
        str(
            resolve_source_checkpoint_dir(args.checkpoint_root, "multicue")
            / "calibration_candidates.csv"
        ),
    )
    run(
        args,
        "scripts/figures/bridge/render_figure5_relative_panels.py",
        "--source-dir",
        "edge_outputs/rbcm/analyses/mea_rbcm_bridge/figure5_relative",
        "--output-dir",
        "edge_outputs/rbcm/figures/mea_rbcm_bridge/figure5_relative",
    )


def stage_audit(args: argparse.Namespace) -> None:
    run(
        args,
        "scripts/checks/audit_edge_pipeline_integrity.py",
        "--checkpoint-root",
        str(args.checkpoint_root.resolve()),
        "--output-dir",
        "edge_outputs/rbcm/reproduction/v5/audit",
    )


def main() -> None:
    args = parse_args()
    stages = (
        ("verify", "protocol", "inference", "inputs", "tables", "figure5", "audit")
        if args.stage == "all"
        else (args.stage,)
    )
    handlers = {
        "verify": stage_verify,
        "protocol": stage_protocol,
        "inference": stage_inference,
        "inputs": stage_inputs,
        "tables": stage_tables,
        "figure5": stage_figure5,
        "audit": stage_audit,
    }
    for stage in stages:
        print(f"\n=== V5 STAGE: {stage} ===", flush=True)
        handlers[stage](args)
    print("\nV5 release reproduction pipeline completed.", flush=True)


if __name__ == "__main__":
    main()
