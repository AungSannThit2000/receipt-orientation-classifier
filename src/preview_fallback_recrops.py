from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from prepare_dataset import tile_image
from receipt_preprocessing import extract_receipt, open_rgb_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT.parents[1] / "large-receipt-image-dataset-SRD"
ANALYSIS_PATH = PROJECT_ROOT / "data" / "manifests" / "source_analysis.csv"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "fallback_recrop_preview.jpg"


def main() -> None:
    with ANALYSIS_PATH.open(newline="", encoding="utf-8") as file:
        fallback_rows = [
            row for row in csv.DictReader(file) if row["crop_method"] == "full_frame_fallback"
        ]

    tile_width, tile_height = 230, 285
    columns = 6
    rows = (len(fallback_rows) + columns - 1) // columns
    grid = Image.new("RGB", (columns * tile_width, rows * tile_height), (245, 245, 245))
    draw = ImageDraw.Draw(grid)
    method_counts: Counter[str] = Counter()

    for index, row in enumerate(fallback_rows):
        source_image = str(row["source_image"])
        extraction = extract_receipt(open_rgb_image(SOURCE_DIR / source_image))
        method_counts[extraction.method] += 1
        left = (index % columns) * tile_width
        top = (index // columns) * tile_height
        grid.paste(tile_image(extraction.image, (210, 210)), (left + 10, top + 42))
        draw.text((left + 10, top + 8), source_image, fill=(20, 20, 20))
        draw.text(
            (left + 10, top + 25),
            f"{extraction.method} conf={extraction.confidence:.2f}",
            fill=(70, 70, 70),
        )
        draw.text(
            (left + 10, top + 257),
            f"candidate area={extraction.area_ratio:.2f}",
            fill=(70, 70, 70),
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    grid.save(OUTPUT_PATH, format="JPEG", quality=91, optimize=True)
    print(f"Preview saved to: {OUTPUT_PATH}")
    print(f"Fallback sources checked: {len(fallback_rows)}")
    for method, count in sorted(method_counts.items()):
        print(f"{method}: {count}")


if __name__ == "__main__":
    main()
