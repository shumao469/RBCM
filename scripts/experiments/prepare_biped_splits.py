"""Prepare deterministic BIPED train/val splits for stability reporting."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path, default=PROJECT_ROOT / "edge_data" / "official_repro")
    parser.add_argument(
        "--normalized-root",
        type=Path,
        default=PROJECT_ROOT / "edge_data" / "official_rbcm" / "BIPED",
    )
    parser.add_argument("--protocol-prefix", default="split")
    parser.add_argument("--num-splits", type=int, default=3)
    parser.add_argument("--val-count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=4517)
    parser.add_argument("--overwrite-splits", action="store_true")
    return parser.parse_args()


def read_pairs(path: Path) -> list[tuple[str, str]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty pair list: {path}")
    if text.startswith("["):
        raw = json.loads(text)
        pairs = [(str(item[0]), str(item[1])) for item in raw if len(item) >= 2]
        if not pairs:
            raise ValueError(f"No JSON pairs found in {path}")
        return pairs
    pairs: list[tuple[str, str]] = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            pairs.append((parts[0], parts[1]))
    if not pairs:
        raise ValueError(f"No pairs found in {path}")
    return pairs


def write_split(path: Path, stems: list[str], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        expected = "\n".join(stems) + "\n"
        if path.read_text(encoding="utf-8") != expected:
            raise ValueError(f"Existing split differs from the deterministic protocol: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(stems) + "\n", encoding="utf-8")


def verify_normalized_pairs(root: Path, pairs: list[tuple[str, str]]) -> None:
    """Check that the normalized release tree contains every official pair."""
    image_stems = {path.stem for path in (root / "image").iterdir() if path.is_file()}
    edge_stems = {path.stem for path in (root / "edge").iterdir() if path.is_file()}
    expected = {Path(image_rel).stem for image_rel, _ in pairs}
    missing_images = sorted(expected - image_stems)
    missing_edges = sorted(expected - edge_stems)
    if missing_images or missing_edges:
        raise FileNotFoundError(
            f"Incomplete normalized BIPED tree: missing_images={missing_images[:5]}, "
            f"missing_edges={missing_edges[:5]}"
        )


def main() -> None:
    args = parse_args()
    src_root = args.official_root.resolve() / "BIPED"
    train_pairs = read_pairs(src_root / "train_pair0.lst")
    test_pairs = read_pairs(src_root / "test_pair.lst")
    train_stems = [Path(image_rel).stem for image_rel, _ in train_pairs]
    test_stems = [Path(image_rel).stem for image_rel, _ in test_pairs]
    if args.num_splits * args.val_count > len(train_stems):
        raise ValueError("num_splits * val_count exceeds the number of BIPED training samples")

    shuffled = list(train_stems)
    random.Random(int(args.seed)).shuffle(shuffled)
    normalized_root = args.normalized_root.resolve()
    protocol_root = normalized_root / "splits" / "protocols"
    verify_normalized_pairs(normalized_root, train_pairs + test_pairs)
    manifest = {
        "official_root": str(src_root),
        "normalized_root": str(normalized_root),
        "protocol_root": str(protocol_root),
        "protocol_prefix": str(args.protocol_prefix),
        "num_splits": int(args.num_splits),
        "val_count": int(args.val_count),
        "seed": int(args.seed),
        "splits": [],
    }
    for split_idx in range(int(args.num_splits)):
        split_root = protocol_root / f"{args.protocol_prefix}{split_idx}"
        val_stems = sorted(shuffled[split_idx * args.val_count : (split_idx + 1) * args.val_count])
        val_set = set(val_stems)
        split_train = [stem for stem in train_stems if stem not in val_set]
        write_split(split_root / "splits" / "train.txt", split_train, overwrite=bool(args.overwrite_splits))
        write_split(split_root / "splits" / "val.txt", val_stems, overwrite=bool(args.overwrite_splits))
        write_split(split_root / "splits" / "test.txt", test_stems, overwrite=bool(args.overwrite_splits))
        write_split(
            split_root / "splits" / "all.txt",
            sorted({*split_train, *val_stems, *test_stems}),
            overwrite=bool(args.overwrite_splits),
        )
        manifest["splits"].append(
            {
                "split": split_idx,
                "root": str(split_root),
                "train_count": len(split_train),
                "val_count": len(val_stems),
                "test_count": len(test_stems),
                "val_stems": val_stems,
            }
        )
    protocol_root.mkdir(parents=True, exist_ok=True)
    (protocol_root / "stability_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
