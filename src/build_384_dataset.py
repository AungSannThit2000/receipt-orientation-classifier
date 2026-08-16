from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image

from receipt_preprocessing import open_rgb_image, orientation_sample


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "generated_manifest.csv"
DEFAULT_OUTPUT_MANIFEST = (
    PROJECT_ROOT / "data" / "manifests" / "generated_manifest_384.csv"
)
DEFAULT_CANONICAL_DIR = PROJECT_ROOT / "data" / "canonical"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed_384"


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def validate_group_splits(rows: list[dict[str, str]]) -> None:
    splits_by_source: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        splits_by_source[row["source_image"]].add(row["split"])
    leakage = {
        source: sorted(splits)
        for source, splits in splits_by_source.items()
        if len(splits) != 1
    }
    if leakage:
        sample = list(leakage.items())[:5]
        raise ValueError(f"Source receipts cross dataset splits: {sample}")


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild the verified orientation dataset at a higher resolution."
    )
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--canonical-dir", type=Path, default=DEFAULT_CANONICAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.image_size < 224:
        raise ValueError("Draft 3 training images must be at least 224 px.")
    rows = read_manifest(args.source_manifest)
    if not rows:
        raise ValueError("The source manifest is empty.")
    validate_group_splits(rows)

    canonical_cache: dict[Path, Image.Image] = {}
    output_rows: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        canonical_path = args.canonical_dir / row["split"] / row["source_image"]
        if canonical_path not in canonical_cache:
            canonical_cache[canonical_path] = open_rgb_image(canonical_path)

        output_path = args.output_dir / row["generated_image"]
        if args.overwrite or not output_path.is_file():
            sample = orientation_sample(
                canonical_cache[canonical_path],
                clockwise_degrees=float(row["clockwise_angle"]),
                size=args.image_size,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sample.save(
                output_path,
                format="JPEG",
                quality=args.jpeg_quality,
                optimize=True,
            )
        else:
            with Image.open(output_path) as existing:
                if existing.size != (args.image_size, args.image_size):
                    raise ValueError(
                        f"Existing output has the wrong size: {output_path} {existing.size}"
                    )

        output_rows.append({**row, "image_size": str(args.image_size)})
        if index % 200 == 0 or index == len(rows):
            print(f"generated {index}/{len(rows)}", flush=True)

    write_manifest(args.output_manifest, output_rows)
    print(f"output_dir={args.output_dir}", flush=True)
    print(f"manifest={args.output_manifest}", flush=True)


if __name__ == "__main__":
    main()
