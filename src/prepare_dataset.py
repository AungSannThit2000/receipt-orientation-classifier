from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import stat
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from ocr_orientation import EasyOCROrientationDetector, OrientationDecision
from receipt_preprocessing import (
    MODEL_CANVAS_COLOR,
    draw_extraction_overlay,
    extract_receipt,
    open_rgb_image,
    orientation_sample,
    rotate_with_fill,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_SOURCE_DIR = WORKSPACE_ROOT / "large-receipt-image-dataset-SRD"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_CANONICAL_DIR = PROJECT_ROOT / "data" / "canonical"
DEFAULT_MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"
DEFAULT_DOCS_DIR = PROJECT_ROOT / "docs"
DEFAULT_OCR_MODEL_DIR = PROJECT_ROOT / "models" / "easyocr"
DEFAULT_OVERRIDES_PATH = PROJECT_ROOT / "config" / "orientation_overrides.csv"

CLASS_DEFINITIONS = (
    ("upright", 0.0),
    ("tilted_right", 90.0),
    ("upside_down", 180.0),
    ("tilted_left", 270.0),
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class SourceImage:
    path: Path
    relative_path: str
    sha256: str
    width: int
    height: int
    image_format: str
    mode: str
    exif_orientation: int | None


@dataclass(frozen=True)
class OrientationOverride:
    rotation_ccw_to_upright: int
    deskew_ccw: float
    note: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_source_images(source_dir: Path) -> tuple[list[SourceImage], list[dict[str, str]]]:
    images: list[SourceImage] = []
    errors: list[dict[str, str]] = []

    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative_path = path.relative_to(source_dir).as_posix()
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                images.append(
                    SourceImage(
                        path=path,
                        relative_path=relative_path,
                        sha256=sha256_file(path),
                        width=image.width,
                        height=image.height,
                        image_format=image.format or "",
                        mode=image.mode,
                        exif_orientation=image.getexif().get(274),
                    )
                )
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            errors.append(
                {
                    "source_image": relative_path,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    return images, errors


def deduplicate_sources(images: list[SourceImage]) -> tuple[list[SourceImage], list[dict[str, str]]]:
    by_hash: dict[str, list[SourceImage]] = {}
    for image in images:
        by_hash.setdefault(image.sha256, []).append(image)

    unique: list[SourceImage] = []
    duplicates: list[dict[str, str]] = []
    for group in by_hash.values():
        ordered = sorted(group, key=lambda item: item.relative_path)
        kept = ordered[0]
        unique.append(kept)
        for duplicate in ordered[1:]:
            duplicates.append(
                {
                    "kept_source_image": kept.relative_path,
                    "excluded_duplicate_image": duplicate.relative_path,
                    "sha256": duplicate.sha256,
                }
            )
    return sorted(unique, key=lambda item: item.relative_path), duplicates


def split_sources(
    images: list[SourceImage],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[SourceImage]]:
    shuffled = list(images)
    random.Random(seed).shuffle(shuffled)
    train_count = round(len(shuffled) * train_ratio)
    val_count = round(len(shuffled) * val_ratio)
    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }


def load_overrides(path: Path) -> dict[str, OrientationOverride]:
    if not path.exists():
        return {}
    overrides: dict[str, OrientationOverride] = {}
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            source_image = (row.get("source_image") or "").strip()
            rotation_text = (row.get("rotation_ccw_to_upright") or "").strip()
            if not source_image or not rotation_text:
                continue
            rotation = int(float(rotation_text)) % 360
            if rotation not in {0, 90, 180, 270}:
                raise ValueError(f"Override rotation must be 0, 90, 180, or 270: {source_image}")
            overrides[source_image] = OrientationOverride(
                rotation_ccw_to_upright=rotation,
                deskew_ccw=float((row.get("deskew_ccw") or "0").strip() or 0),
                note=(row.get("note") or "").strip(),
            )
    return overrides


def load_analysis_cache(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Cannot reuse analysis because the manifest does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    cache: dict[str, dict[str, object]] = {}
    for row in rows:
        row["review_required"] = str(row["review_required"]).strip().lower() == "true"
        cache[str(row["source_image"])] = row
    return cache


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_jpeg(image: Image.Image, path: Path, quality: int = 92, maximum_dimension: int | None = None) -> None:
    output = image.convert("RGB")
    if maximum_dimension is not None and max(output.size) > maximum_dimension:
        output = ImageOps.contain(
            output,
            (maximum_dimension, maximum_dimension),
            method=Image.Resampling.LANCZOS,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    output.save(path, format="JPEG", quality=quality, optimize=True)


def ensure_safe_generated_path(path: Path) -> None:
    resolved_root = PROJECT_ROOT.resolve()
    resolved_path = path.resolve()
    if resolved_path == resolved_root or resolved_root not in resolved_path.parents:
        raise ValueError(f"Refusing to reset a path outside the Draft 2 project: {resolved_path}")


def reset_generated_directories(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"Generated output already exists. Use --overwrite to replace:\n{joined}")
    for path in existing:
        ensure_safe_generated_path(path)
        shutil.rmtree(path, onerror=_remove_readonly)
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def _remove_readonly(function, path: str, _error_info) -> None:
    os.chmod(path, stat.S_IWRITE)
    function(path)


def disabled_ocr_decision() -> OrientationDecision:
    return OrientationDecision(
        best_rotation_ccw=0,
        deskew_rotation_ccw=0.0,
        best_score=0.0,
        second_score=0.0,
        confidence_margin=1.0,
        review_required=False,
        scores={0: 0.0, 90: 0.0, 180: 0.0, 270: 0.0},
        keyword_hits={0: 0, 90: 0, 180: 0, 270: 0},
        text_previews={0: "", 90: "", 180: "", 270: ""},
    )


def tile_image(image: Image.Image, size: tuple[int, int], fill=(225, 225, 225)) -> Image.Image:
    fitted = ImageOps.contain(image.convert("RGB"), size, Image.Resampling.LANCZOS)
    tile = Image.new("RGB", size, fill)
    tile.paste(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
    return tile


def create_crop_qa_grid(
    source_rows: list[dict[str, object]],
    source_dir: Path,
    canonical_dir: Path,
    destination: Path,
    seed: int,
    maximum_items: int = 24,
) -> None:
    if not source_rows:
        return
    fallback = [row for row in source_rows if row["crop_method"] == "full_frame_fallback"]
    detected = [row for row in source_rows if row["crop_method"] != "full_frame_fallback"]
    rng = random.Random(seed)
    rng.shuffle(fallback)
    rng.shuffle(detected)
    selected = (fallback[: maximum_items // 2] + detected[: maximum_items // 2])[:maximum_items]
    if len(selected) < maximum_items:
        selected_paths = {str(row["source_image"]) for row in selected}
        remaining = [row for row in source_rows if str(row["source_image"]) not in selected_paths]
        selected.extend(remaining[: maximum_items - len(selected)])

    tile_width, tile_height = 430, 250
    columns = 3
    rows = (len(selected) + columns - 1) // columns
    grid = Image.new("RGB", (columns * tile_width, rows * tile_height), (245, 245, 245))
    draw = ImageDraw.Draw(grid)

    for index, row in enumerate(selected):
        left = (index % columns) * tile_width
        top = (index // columns) * tile_height
        source_path = source_dir / str(row["source_image"])
        canonical_path = canonical_dir / str(row["canonical_image"])
        original = open_rgb_image(source_path)
        canonical = open_rgb_image(canonical_path)
        grid.paste(tile_image(original, (195, 190)), (left + 10, top + 38))
        grid.paste(tile_image(canonical, (195, 190)), (left + 220, top + 38))
        draw.text((left + 10, top + 8), str(row["source_image"]), fill=(20, 20, 20))
        draw.text(
            (left + 170, top + 8),
            f"{row['crop_method']} {float(row['crop_confidence']):.2f}",
            fill=(80, 80, 80),
        )
        draw.text((left + 10, top + 226), "original", fill=(80, 80, 80))
        draw.text((left + 220, top + 226), "cropped + canonicalized", fill=(80, 80, 80))

    destination.parent.mkdir(parents=True, exist_ok=True)
    grid.save(destination, format="JPEG", quality=90, optimize=True)


def create_review_grid(
    review_rows: list[dict[str, object]],
    canonical_dir: Path,
    destination: Path,
    maximum_items: int = 36,
) -> None:
    if not review_rows:
        if destination.exists():
            destination.unlink()
        return
    selected = review_rows[:maximum_items]
    tile_width, tile_height = 230, 270
    columns = 6
    rows = (len(selected) + columns - 1) // columns
    grid = Image.new("RGB", (columns * tile_width, rows * tile_height), (245, 245, 245))
    draw = ImageDraw.Draw(grid)
    for index, row in enumerate(selected):
        left = (index % columns) * tile_width
        top = (index // columns) * tile_height
        image = open_rgb_image(canonical_dir / str(row["canonical_image"]))
        grid.paste(tile_image(image, (210, 210)), (left + 10, top + 36))
        draw.text((left + 10, top + 8), str(row["source_image"]), fill=(20, 20, 20))
        draw.text(
            (left + 112, top + 8),
            f"best={row['ocr_best_rotation_ccw']} margin={float(row['ocr_margin']):.2f}",
            fill=(80, 80, 80),
        )
        draw.text((left + 10, top + 248), "shown with conservative 0-degree default", fill=(80, 80, 80))
    destination.parent.mkdir(parents=True, exist_ok=True)
    grid.save(destination, format="JPEG", quality=90, optimize=True)


def create_fallback_grid(
    source_rows: list[dict[str, object]],
    canonical_dir: Path,
    destination: Path,
) -> None:
    fallback_rows = [row for row in source_rows if row["crop_method"] == "full_frame_fallback"]
    if not fallback_rows:
        if destination.exists():
            destination.unlink()
        return
    tile_width, tile_height = 230, 270
    columns = 6
    rows = (len(fallback_rows) + columns - 1) // columns
    grid = Image.new("RGB", (columns * tile_width, rows * tile_height), (245, 245, 245))
    draw = ImageDraw.Draw(grid)
    for index, row in enumerate(fallback_rows):
        left = (index % columns) * tile_width
        top = (index // columns) * tile_height
        image = open_rgb_image(canonical_dir / str(row["canonical_image"]))
        grid.paste(tile_image(image, (210, 210)), (left + 10, top + 36))
        draw.text((left + 10, top + 8), str(row["source_image"]), fill=(20, 20, 20))
        draw.text(
            (left + 10, top + 248),
            f"full-frame confidence={float(row['crop_confidence']):.2f}",
            fill=(80, 80, 80),
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    grid.save(destination, format="JPEG", quality=90, optimize=True)


def create_orientation_grid(
    manifest_rows: list[dict[str, object]],
    output_dir: Path,
    destination: Path,
    seed: int,
    samples_per_class: int = 5,
) -> None:
    if not manifest_rows:
        return
    rng = random.Random(seed)
    tile_size = 224
    label_height = 34
    rows = len(CLASS_DEFINITIONS)
    columns = samples_per_class
    grid = Image.new("RGB", (columns * tile_size, rows * (tile_size + label_height)), (245, 245, 245))
    draw = ImageDraw.Draw(grid)

    for row_index, (label, _) in enumerate(CLASS_DEFINITIONS):
        choices = [row for row in manifest_rows if row["label"] == label]
        selected = rng.sample(choices, min(columns, len(choices)))
        for column_index, row in enumerate(selected):
            image = open_rgb_image(output_dir / str(row["generated_image"]))
            x = column_index * tile_size
            y = row_index * (tile_size + label_height)
            grid.paste(image, (x, y))
            draw.text(
                (x + 6, y + tile_size + 7),
                f"{label} {float(row['clockwise_angle']):.1f} deg",
                fill=(20, 20, 20),
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    grid.save(destination, format="JPEG", quality=91, optimize=True)


def create_markdown_report(summary: dict[str, object], destination: Path) -> None:
    orientation_counts = summary["ocr_best_rotation_counts"]
    crop_counts = summary["crop_method_counts"]
    split_counts = summary["split_counts"]
    class_counts = summary["class_counts"]
    lines = [
        "# Draft 2 - Receipt Extraction and Relabelling Report",
        "",
        "## Label definition",
        "",
        "| Label | Clockwise range from upright |",
        "| --- | --- |",
        "| `upright` | 315 to 360 degrees, or 0 to less than 45 degrees |",
        "| `tilted_right` | 45 to less than 135 degrees |",
        "| `upside_down` | 135 to less than 225 degrees |",
        "| `tilted_left` | 225 to less than 315 degrees |",
        "",
        "Generated samples use class centers with controlled random jitter. Exact class-boundary images are not generated.",
        "",
        "## Source processing",
        "",
        f"- Source files found: {summary['source_images_found']}",
        f"- Unique sources after exact deduplication: {summary['unique_source_images_used']}",
        f"- Duplicate files excluded: {summary['excluded_duplicate_images']}",
        f"- Orientation-approved sources: {summary['approved_source_images']}",
        f"- Sources requiring orientation review: {summary['review_required_images']}",
        f"- Generated 224 x 224 images: {summary['generated_images']}",
        "",
        "## Receipt extraction methods",
        "",
        "| Method | Sources |",
        "| --- | ---: |",
    ]
    for method, count in crop_counts.items():
        lines.append(f"| `{method}` | {count} |")

    lines.extend(
        [
            "",
            "A full-frame fallback means the detector could not confidently separate a paper boundary. These images are retained because many dataset files are already tightly cropped to the receipt.",
            "",
            "## OCR top-side decisions",
            "",
            "| Counter-clockwise correction | Sources |",
            "| --- | ---: |",
        ]
    )
    for rotation in (0, 90, 180, 270):
        lines.append(f"| {rotation} degrees | {orientation_counts.get(str(rotation), 0)} |")

    lines.extend(
        [
            "",
            "OCR first compares upright with upside down, then checks the two sideways rotations when that result is weak or ambiguous. Low-score or closely tied decisions are excluded until manually reviewed.",
            "",
            "## Generated dataset",
            "",
            "| Split | Approved sources | Generated images |",
            "| --- | ---: | ---: |",
        ]
    )
    for split in ("train", "val", "test"):
        lines.append(
            f"| `{split}` | {split_counts[split]['approved_sources']} | {split_counts[split]['generated_images']} |"
        )

    lines.extend(["", "| Split | Upright | Right | Upside down | Left |", "| --- | ---: | ---: | ---: | ---: |"])
    for split in ("train", "val", "test"):
        counts = class_counts[split]
        lines.append(
            f"| `{split}` | {counts['upright']} | {counts['tilted_right']} | {counts['upside_down']} | {counts['tilted_left']} |"
        )

    lines.extend(
        [
            "",
            "## Quality-control files",
            "",
            "- `docs/crop_qa_grid.jpg`: original versus extracted/canonical receipt examples.",
            "- `docs/orientation_sample_grid.jpg`: examples of the four final labels.",
            "- `docs/review_required_grid.jpg`: uncertain OCR decisions, when any exist.",
            "- `docs/full_frame_fallback_grid.jpg`: sources where no separate paper boundary was applied.",
            "- `data/manifests/source_analysis.csv`: crop and OCR measurements for every source.",
            "- `data/manifests/review_required.csv`: sources blocked from training pending review.",
            "",
            "## Inference rule",
            "",
            "Streamlit must call the same receipt extraction and 224 x 224 canvas functions. It must preserve the uploaded receipt's observed rotation; OCR must not rotate the test image before the classifier receives it.",
            "",
        ]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Draft 2 receipt-orientation dataset.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--canonical-dir", type=Path, default=DEFAULT_CANONICAL_DIR)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument("--ocr-model-dir", type=Path, default=DEFAULT_OCR_MODEL_DIR)
    parser.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES_PATH)
    parser.add_argument("--ocr-mode", choices=("verify", "off"), default="verify")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--samples-per-class", type=int, default=3)
    parser.add_argument("--jitter", type=float, default=40.0)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=4201)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--max-sources", type=int)
    parser.add_argument("--reuse-analysis", action="store_true")
    parser.add_argument("--refresh-fallback-crops", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {args.source_dir}")
    if not 0 < args.jitter < 45:
        raise ValueError("--jitter must be greater than 0 and less than 45 degrees")

    analysis_cache: dict[str, dict[str, object]] = {}
    if args.reuse_analysis:
        analysis_cache = load_analysis_cache(args.manifest_dir / "source_analysis.csv")
        reset_generated_directories([args.output_dir], overwrite=args.overwrite)
        args.canonical_dir.mkdir(parents=True, exist_ok=True)
        args.manifest_dir.mkdir(parents=True, exist_ok=True)
        args.report_dir.mkdir(parents=True, exist_ok=True)
    else:
        reset_generated_directories(
            [args.output_dir, args.canonical_dir, args.manifest_dir, args.report_dir],
            overwrite=args.overwrite,
        )

    images, unreadable_rows = scan_source_images(args.source_dir)
    unique_images, duplicate_rows = deduplicate_sources(images)
    if args.max_sources is not None:
        unique_images = unique_images[: args.max_sources]
    splits = split_sources(unique_images, args.train_ratio, args.val_ratio, args.seed)
    source_to_split = {
        source.relative_path: split
        for split, split_images in splits.items()
        for source in split_images
    }
    overrides = load_overrides(args.overrides)

    ocr_detector = None
    if args.ocr_mode == "verify" and not args.reuse_analysis:
        ocr_detector = EasyOCROrientationDetector(args.ocr_model_dir)

    source_rows: list[dict[str, object]] = []
    accepted_sources: list[tuple[SourceImage, str, Path]] = []
    total_sources = len(unique_images)

    for source_index, source in enumerate(unique_images, start=1):
        split = source_to_split[source.relative_path]
        if args.reuse_analysis:
            if source.relative_path not in analysis_cache:
                raise KeyError(f"Source is missing from cached analysis: {source.relative_path}")
            row = dict(analysis_cache[source.relative_path])
            override = overrides.get(source.relative_path)
            if override is not None:
                cached_rotation = int(float(str(row["applied_cardinal_rotation_ccw"])))
                cached_deskew = float(str(row["applied_deskew_ccw"]))
                if cached_rotation != override.rotation_ccw_to_upright or abs(cached_deskew - override.deskew_ccw) > 0.001:
                    raise ValueError(
                        f"Cached canonical image does not match the override for {source.relative_path}; "
                        "run without --reuse-analysis"
                    )
                row["review_required"] = False
                row["orientation_source"] = "manual_override"
                row["manual_override_note"] = override.note
            canonical_path = args.canonical_dir / str(row["canonical_image"])
            if not canonical_path.exists():
                raise FileNotFoundError(f"Cached canonical image is missing: {canonical_path}")

            if args.refresh_fallback_crops and row["crop_method"] == "full_frame_fallback":
                original = open_rgb_image(source.path)
                refreshed_extraction = extract_receipt(original)
                if refreshed_extraction.method != "full_frame_fallback":
                    cardinal_rotation = int(float(str(row["applied_cardinal_rotation_ccw"])))
                    deskew_rotation = float(str(row["applied_deskew_ccw"]))
                    canonical = rotate_with_fill(
                        refreshed_extraction.image,
                        cardinal_rotation + deskew_rotation,
                        MODEL_CANVAS_COLOR,
                    )
                    save_jpeg(
                        canonical,
                        canonical_path,
                        quality=args.jpeg_quality,
                        maximum_dimension=1400,
                    )
                    overlay_path = (
                        args.report_dir
                        / "crop_overlays"
                        / f"{Path(source.relative_path).stem}.jpg"
                    )
                    save_jpeg(
                        draw_extraction_overlay(original, refreshed_extraction),
                        overlay_path,
                        quality=88,
                        maximum_dimension=900,
                    )
                    row["crop_method"] = refreshed_extraction.method
                    row["crop_confidence"] = f"{refreshed_extraction.confidence:.4f}"
                    row["crop_area_ratio"] = f"{refreshed_extraction.area_ratio:.4f}"
                    row["crop_bbox"] = ",".join(
                        str(value) for value in refreshed_extraction.bbox
                    )
            source_rows.append(row)
            if not row["review_required"]:
                accepted_sources.append((source, split, canonical_path))
            status = "REVIEW" if row["review_required"] else "accepted"
            print(
                f"[{source_index:03d}/{total_sources:03d}] {source.relative_path}: "
                f"reused analysis {status}",
                flush=True,
            )
            continue

        original = open_rgb_image(source.path)
        extraction = extract_receipt(original)
        decision = ocr_detector.detect(extraction.image) if ocr_detector else disabled_ocr_decision()
        override = overrides.get(source.relative_path)

        if override is not None:
            cardinal_rotation = override.rotation_ccw_to_upright
            deskew_rotation = override.deskew_ccw
            review_required = False
            orientation_source = "manual_override"
        elif decision.review_required:
            cardinal_rotation = 0
            deskew_rotation = 0.0
            review_required = True
            orientation_source = "review_required_default"
        else:
            cardinal_rotation = decision.best_rotation_ccw
            deskew_rotation = decision.deskew_rotation_ccw
            review_required = False
            orientation_source = "easyocr"

        canonical = rotate_with_fill(
            extraction.image,
            cardinal_rotation + deskew_rotation,
            MODEL_CANVAS_COLOR,
        )
        canonical_relative = Path(split) / f"{Path(source.relative_path).stem}.jpg"
        canonical_path = args.canonical_dir / canonical_relative
        save_jpeg(canonical, canonical_path, quality=args.jpeg_quality, maximum_dimension=1400)

        overlay_path = args.report_dir / "crop_overlays" / f"{Path(source.relative_path).stem}.jpg"
        save_jpeg(draw_extraction_overlay(original, extraction), overlay_path, quality=88, maximum_dimension=900)

        row: dict[str, object] = {
            "source_image": source.relative_path,
            "split": split,
            "source_sha256": source.sha256,
            "original_width": source.width,
            "original_height": source.height,
            "exif_orientation": source.exif_orientation or "",
            "crop_method": extraction.method,
            "crop_confidence": f"{extraction.confidence:.4f}",
            "crop_area_ratio": f"{extraction.area_ratio:.4f}",
            "crop_bbox": ",".join(str(value) for value in extraction.bbox),
            "ocr_best_rotation_ccw": decision.best_rotation_ccw,
            "ocr_best_score": f"{decision.best_score:.4f}",
            "ocr_second_score": f"{decision.second_score:.4f}",
            "ocr_margin": f"{decision.confidence_margin:.4f}",
            "ocr_score_0": f"{decision.scores[0]:.4f}",
            "ocr_score_90": f"{decision.scores[90]:.4f}",
            "ocr_score_180": f"{decision.scores[180]:.4f}",
            "ocr_score_270": f"{decision.scores[270]:.4f}",
            "ocr_keyword_hits_0": decision.keyword_hits[0],
            "ocr_keyword_hits_90": decision.keyword_hits[90],
            "ocr_keyword_hits_180": decision.keyword_hits[180],
            "ocr_keyword_hits_270": decision.keyword_hits[270],
            "ocr_best_text_preview": decision.text_previews[decision.best_rotation_ccw],
            "applied_cardinal_rotation_ccw": cardinal_rotation,
            "applied_deskew_ccw": f"{deskew_rotation:.4f}",
            "orientation_source": orientation_source,
            "manual_override_note": override.note if override else "",
            "review_required": review_required,
            "canonical_image": canonical_relative.as_posix(),
        }
        source_rows.append(row)
        if not review_required:
            accepted_sources.append((source, split, canonical_path))

        status = "REVIEW" if review_required else "accepted"
        print(
            f"[{source_index:03d}/{total_sources:03d}] {source.relative_path}: "
            f"crop={extraction.method} ocr={decision.best_rotation_ccw} "
            f"margin={decision.confidence_margin:.2f} {status}",
            flush=True,
        )

    rng = random.Random(args.seed)
    manifest_rows: list[dict[str, object]] = []
    for source, split, canonical_path in accepted_sources:
        canonical = open_rgb_image(canonical_path)
        source_stem = Path(source.relative_path).stem
        for label, center_angle in CLASS_DEFINITIONS:
            for sample_index in range(1, args.samples_per_class + 1):
                jitter = rng.uniform(-args.jitter, args.jitter)
                clockwise_angle = (center_angle + jitter) % 360.0
                generated_name = (
                    f"{source_stem}__cw_{clockwise_angle:06.2f}__sample_{sample_index}.jpg"
                )
                generated_relative = Path(split) / label / generated_name
                generated_path = args.output_dir / generated_relative
                sample = orientation_sample(
                    canonical,
                    clockwise_degrees=clockwise_angle,
                    size=args.image_size,
                )
                save_jpeg(sample, generated_path, quality=args.jpeg_quality)
                manifest_rows.append(
                    {
                        "generated_image": generated_relative.as_posix(),
                        "source_image": source.relative_path,
                        "split": split,
                        "label": label,
                        "class_center_clockwise": f"{center_angle:.1f}",
                        "jitter_degrees": f"{jitter:.4f}",
                        "clockwise_angle": f"{clockwise_angle:.4f}",
                        "image_size": args.image_size,
                        "source_sha256": source.sha256,
                    }
                )

    source_fields = list(source_rows[0].keys()) if source_rows else []
    write_csv(args.manifest_dir / "source_analysis.csv", source_rows, source_fields)
    write_csv(
        args.manifest_dir / "review_required.csv",
        [row for row in source_rows if row["review_required"]],
        source_fields,
    )
    write_csv(
        args.manifest_dir / "generated_manifest.csv",
        manifest_rows,
        [
            "generated_image",
            "source_image",
            "split",
            "label",
            "class_center_clockwise",
            "jitter_degrees",
            "clockwise_angle",
            "image_size",
            "source_sha256",
        ],
    )
    write_csv(
        args.manifest_dir / "excluded_duplicates.csv",
        duplicate_rows,
        ["kept_source_image", "excluded_duplicate_image", "sha256"],
    )
    write_csv(
        args.manifest_dir / "unreadable_images.csv",
        unreadable_rows,
        ["source_image", "error_type", "error"],
    )

    review_rows = [row for row in source_rows if row["review_required"]]
    crop_counts = Counter(str(row["crop_method"]) for row in source_rows)
    orientation_counts = Counter(str(row["ocr_best_rotation_ccw"]) for row in source_rows)
    split_counts: dict[str, dict[str, int]] = {}
    class_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        split_counts[split] = {
            "source_images": sum(1 for row in source_rows if row["split"] == split),
            "approved_sources": sum(
                1 for row in source_rows if row["split"] == split and not row["review_required"]
            ),
            "generated_images": sum(1 for row in manifest_rows if row["split"] == split),
        }
        class_counts[split] = {
            label: sum(
                1 for row in manifest_rows if row["split"] == split and row["label"] == label
            )
            for label, _ in CLASS_DEFINITIONS
        }

    summary: dict[str, object] = {
        "project_root": str(PROJECT_ROOT),
        "source_dir": str(args.source_dir),
        "output_dir": str(args.output_dir),
        "image_size": args.image_size,
        "samples_per_class": args.samples_per_class,
        "jitter_degrees": args.jitter,
        "seed": args.seed,
        "ocr_mode": "reused_verified_analysis" if args.reuse_analysis else args.ocr_mode,
        "source_images_found": len(images),
        "unique_source_images_used": len(unique_images),
        "excluded_duplicate_images": len(duplicate_rows),
        "unreadable_images": len(unreadable_rows),
        "approved_source_images": len(accepted_sources),
        "review_required_images": len(review_rows),
        "generated_images": len(manifest_rows),
        "crop_method_counts": dict(sorted(crop_counts.items())),
        "ocr_best_rotation_counts": dict(sorted(orientation_counts.items())),
        "split_counts": split_counts,
        "class_counts": class_counts,
        "classes": [label for label, _ in CLASS_DEFINITIONS],
    }
    (args.manifest_dir / "generation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    create_crop_qa_grid(source_rows, args.source_dir, args.canonical_dir, args.docs_dir / "crop_qa_grid.jpg", args.seed)
    create_review_grid(review_rows, args.canonical_dir, args.docs_dir / "review_required_grid.jpg")
    create_fallback_grid(
        source_rows,
        args.canonical_dir,
        args.docs_dir / "full_frame_fallback_grid.jpg",
    )
    create_orientation_grid(
        manifest_rows,
        args.output_dir,
        args.docs_dir / "orientation_sample_grid.jpg",
        args.seed,
    )
    create_markdown_report(summary, args.docs_dir / "phase2b_preprocessing_report.md")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
