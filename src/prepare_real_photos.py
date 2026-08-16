from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageOps

from receipt_preprocessing import model_canvas
from realtime_receipt_detection import YOLOWorldReceiptDetector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "real_photo_manifest.csv"
DEFAULT_OUTPUT_MANIFEST = (
    PROJECT_ROOT / "data" / "manifests" / "real_photo_prepared_384.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "real_processed_384"
CLASS_NAMES = {"upright", "tilted_right", "upside_down", "tilted_left"}
VALID_SPLITS = {"train", "val", "test", "external_holdout"}


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def validate_manifest(rows: list[dict[str, str]]) -> None:
    splits_by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["label"] not in CLASS_NAMES:
            raise ValueError(f"Unsupported label: {row['label']}")
        if row["split"] not in VALID_SPLITS:
            raise ValueError(f"Unsupported split: {row['split']}")
        if not row["receipt_group"].strip():
            raise ValueError("Every real photo needs a physical receipt_group.")
        splits_by_group[row["receipt_group"]].add(row["split"])
    leakage = {
        group: sorted(splits)
        for group, splits in splits_by_group.items()
        if len(splits) != 1
    }
    if leakage:
        raise ValueError(f"Physical receipts cross splits: {leakage}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and normalize labeled real receipt photos for Draft 3."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_INPUT_MANIFEST)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-size", type=int, default=384)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.manifest)
    if not rows:
        raise ValueError("The real-photo manifest is empty.")
    validate_manifest(rows)

    detector = YOLOWorldReceiptDetector()
    prepared_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        image_path = PROJECT_ROOT / row["image"]
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        detection = detector.detect(image)
        normalized = model_canvas(
            detection.extraction.image,
            size=args.image_size,
        )
        output_name = f"{row['receipt_group']}__{image_path.stem}.jpg"
        output_relative = Path(row["split"]) / row["label"] / output_name
        output_path = args.output_dir / output_relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
        normalized.save(output_path, format="JPEG", quality=92, optimize=True)
        prepared_rows.append(
            {
                **row,
                "processed_image": output_relative.as_posix(),
                "detected": detection.detected,
                "detector_confidence": f"{detection.detector_confidence:.6f}",
                "crop_method": detection.extraction.method,
                "crop_area_ratio": f"{detection.extraction.area_ratio:.6f}",
                "image_size": args.image_size,
            }
        )
        print(
            f"[{index}/{len(rows)}] {image_path.name}: "
            f"{detection.extraction.method} -> {output_relative}",
            flush=True,
        )

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(prepared_rows[0]))
        writer.writeheader()
        writer.writerows(prepared_rows)
    print(f"manifest={args.output_manifest}", flush=True)


if __name__ == "__main__":
    main()
