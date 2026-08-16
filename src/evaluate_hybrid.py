from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score

from hybrid_orientation import (
    DEFAULT_HYBRID_CONFIG,
    DEFAULT_OCR_MODEL_DIR,
    HybridOCRConfig,
    ROTATION_TO_LABEL,
    candidate_rotations,
    resolve_decision,
)
from inference import DEFAULT_CHECKPOINT, ReceiptOrientationClassifier
from ocr_orientation import EasyOCROrientationDetector
from receipt_preprocessing import rotate_with_fill


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "generated_manifest.csv"
DEFAULT_CANONICAL_DIR = PROJECT_ROOT / "data" / "canonical"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports" / "hybrid"
CLASS_NAMES = ["upright", "tilted_right", "upside_down", "tilted_left"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write an empty evaluation file.")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect_predictions(
    split: str,
    manifest_path: Path,
    canonical_dir: Path,
    checkpoint_path: Path,
    ocr_model_dir: Path,
    output_path: Path,
    chunk_size: int,
    limit: int | None,
    ocr_image_size: int,
    resume: bool,
) -> list[dict[str, object]]:
    manifest_rows = [row for row in read_csv(manifest_path) if row["split"] == split]
    if limit is not None:
        manifest_rows = manifest_rows[:limit]

    collected: list[dict[str, object]] = []
    if resume and output_path.is_file():
        collected = list(read_csv(output_path))
        completed_images = {str(row["generated_image"]) for row in collected}
        manifest_rows = [
            row for row in manifest_rows if row["generated_image"] not in completed_images
        ]
        print(f"{split}: resuming after {len(collected)} completed images", flush=True)

    classifier = ReceiptOrientationClassifier(checkpoint_path)
    detector = EasyOCROrientationDetector(
        model_storage_directory=ocr_model_dir,
        image_size=ocr_image_size,
        download_enabled=False,
    )
    canonical_cache: dict[str, Image.Image] = {}
    started = time.perf_counter()

    for chunk_start in range(0, len(manifest_rows), chunk_size):
        chunk_rows = manifest_rows[chunk_start : chunk_start + chunk_size]
        model_records: list[tuple[dict[str, str], object]] = []
        ocr_requests: list[tuple[Image.Image, tuple[int, ...]]] = []
        for row in chunk_rows:
            source_name = row["source_image"]
            if source_name not in canonical_cache:
                with Image.open(canonical_dir / split / source_name) as image:
                    canonical_cache[source_name] = image.convert("RGB")
            clockwise_angle = float(row["clockwise_angle"])
            observed = rotate_with_fill(
                canonical_cache[source_name],
                -clockwise_angle,
                fill_color=classifier.fill_color,
            )
            model_result = classifier.predict(observed)
            model_records.append((row, model_result))
            ocr_requests.append(
                (
                    model_result.extraction.image,
                    candidate_rotations(model_result.label),
                )
            )

        decisions = detector.compare_rotations_batch(ocr_requests, batch_size=16)
        for (row, model_result), decision in zip(model_records, decisions):
            ocr_label = ROTATION_TO_LABEL[decision.best_rotation_ccw]
            collected.append(
                {
                    "generated_image": row["generated_image"],
                    "source_image": row["source_image"],
                    "actual_label": row["label"],
                    "clockwise_angle": f"{float(row['clockwise_angle']):.4f}",
                    "model_label": model_result.label,
                    "model_confidence": f"{model_result.confidence:.6f}",
                    "model_margin": f"{model_result.margin:.6f}",
                    "ocr_label": ocr_label,
                    "ocr_best_rotation_ccw": decision.best_rotation_ccw,
                    "ocr_best_score": f"{decision.best_score:.6f}",
                    "ocr_second_score": f"{decision.second_score:.6f}",
                    "ocr_margin": f"{decision.confidence_margin:.6f}",
                    "ocr_keyword_hits": decision.keyword_hits[decision.best_rotation_ccw],
                    "ocr_text_preview": decision.text_previews[decision.best_rotation_ccw],
                    "ocr_score_0": f"{decision.scores[0]:.6f}",
                    "ocr_score_90": f"{decision.scores[90]:.6f}",
                    "ocr_score_180": f"{decision.scores[180]:.6f}",
                    "ocr_score_270": f"{decision.scores[270]:.6f}",
                }
            )
        write_csv(output_path, collected)
        elapsed = time.perf_counter() - started
        print(
            f"{split}: {len(collected)} saved, "
            f"{chunk_start + len(chunk_rows)}/{len(manifest_rows)} new "
            f"({elapsed:.1f}s elapsed)",
            flush=True,
        )
    return collected


def accuracy(rows: list[dict[str, object]], key: str) -> float:
    return sum(row[key] == row["actual_label"] for row in rows) / len(rows)


def apply_config(
    rows: list[dict[str, object]],
    config: HybridOCRConfig,
) -> list[dict[str, object]]:
    resolved: list[dict[str, object]] = []
    for row in rows:
        reliable = (
            float(row["ocr_best_score"]) >= config.minimum_score
            and float(row["ocr_margin"]) >= config.minimum_margin
        )
        final_label = str(row["ocr_label"]) if reliable else str(row["model_label"])
        source = (
            "model_ocr_inconclusive"
            if not reliable
            else "ocr_confirmed"
            if final_label == row["model_label"]
            else "ocr_override"
        )
        resolved.append(
            {
                **row,
                "final_label": final_label,
                "decision_source": source,
                "ocr_reliable": reliable,
            }
        )
    return resolved


def tune_thresholds(
    rows: list[dict[str, object]],
    output_path: Path,
) -> HybridOCRConfig:
    score_thresholds = (0.0, 10.0, 15.0, 18.0, 25.0, 35.0, 50.0, 75.0, 100.0)
    margin_thresholds = (0.0, 0.10, 0.15, 0.20, 0.22, 0.25, 0.30, 0.40, 0.50, 0.60)
    candidates: list[dict[str, object]] = []
    for minimum_score in score_thresholds:
        for minimum_margin in margin_thresholds:
            config = HybridOCRConfig(minimum_score, minimum_margin)
            resolved = apply_config(rows, config)
            overrides = [row for row in resolved if row["decision_source"] == "ocr_override"]
            candidates.append(
                {
                    "minimum_score": minimum_score,
                    "minimum_margin": minimum_margin,
                    "accuracy": accuracy(resolved, "final_label"),
                    "macro_f1": f1_score(
                        [row["actual_label"] for row in resolved],
                        [row["final_label"] for row in resolved],
                        labels=CLASS_NAMES,
                        average="macro",
                        zero_division=0,
                    ),
                    "reliable_count": sum(bool(row["ocr_reliable"]) for row in resolved),
                    "override_count": len(overrides),
                    "helpful_overrides": sum(
                        row["final_label"] == row["actual_label"]
                        and row["model_label"] != row["actual_label"]
                        for row in overrides
                    ),
                    "harmful_overrides": sum(
                        row["final_label"] != row["actual_label"]
                        and row["model_label"] == row["actual_label"]
                        for row in overrides
                    ),
                }
            )
    candidates.sort(
        key=lambda row: (
            float(row["accuracy"]),
            float(row["macro_f1"]),
            -int(row["harmful_overrides"]),
            -int(row["override_count"]),
            float(row["minimum_margin"]),
            float(row["minimum_score"]),
        ),
        reverse=True,
    )
    write_csv(output_path, candidates)
    best = candidates[0]
    return HybridOCRConfig(
        minimum_score=float(best["minimum_score"]),
        minimum_margin=float(best["minimum_margin"]),
    )


def plot_confusion(path: Path, rows: list[dict[str, object]], key: str, title: str) -> None:
    matrix = confusion_matrix(
        [row["actual_label"] for row in rows],
        [row[key] for row in rows],
        labels=CLASS_NAMES,
    )
    figure, axis = plt.subplots(figsize=(7.4, 6.4))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_title(title)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=25, ha="right")
    axis.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = int(matrix[row_index, column_index])
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > matrix.max() * 0.55 else "black",
            )
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def write_report(
    path: Path,
    split: str,
    resolved: list[dict[str, object]],
    config: HybridOCRConfig,
) -> None:
    model_accuracy = accuracy(resolved, "model_label")
    ocr_accuracy = accuracy(resolved, "ocr_label")
    hybrid_accuracy = accuracy(resolved, "final_label")
    counts = Counter(str(row["decision_source"]) for row in resolved)
    overrides = [row for row in resolved if row["decision_source"] == "ocr_override"]
    helpful = sum(
        row["final_label"] == row["actual_label"]
        and row["model_label"] != row["actual_label"]
        for row in overrides
    )
    harmful = sum(
        row["final_label"] != row["actual_label"]
        and row["model_label"] == row["actual_label"]
        for row in overrides
    )
    lines = [
        f"# Hybrid OCR Evaluation - {split.title()}",
        "",
        f"- Samples: {len(resolved)}",
        f"- Model-only accuracy: {model_accuracy:.2%}",
        f"- Pairwise OCR-only accuracy: {ocr_accuracy:.2%}",
        f"- Hybrid accuracy: {hybrid_accuracy:.2%}",
        f"- OCR minimum score: {config.minimum_score:.2f}",
        f"- OCR minimum margin: {config.minimum_margin:.2f}",
        f"- OCR confirmations: {counts['ocr_confirmed']}",
        f"- OCR overrides: {counts['ocr_override']}",
        f"- OCR inconclusive fallbacks: {counts['model_ocr_inconclusive']}",
        f"- Helpful overrides: {helpful}",
        f"- Harmful overrides: {harmful}",
        "",
        "The model selects the vertical or horizontal orientation pair. OCR then compares only the two opposite directions in that pair. Weak or closely tied OCR results cannot override the model.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model plus pairwise OCR verification.")
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--canonical-dir", type=Path, default=DEFAULT_CANONICAL_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--ocr-model-dir", type=Path, default=DEFAULT_OCR_MODEL_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_HYBRID_CONFIG)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--ocr-image-size", type=int, default=384)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.reports_dir / f"hybrid_{args.split}_ocr_predictions.csv"
    if args.reuse and predictions_path.is_file():
        rows: list[dict[str, object]] = list(read_csv(predictions_path))
    else:
        rows = collect_predictions(
            split=args.split,
            manifest_path=args.manifest,
            canonical_dir=args.canonical_dir,
            checkpoint_path=args.checkpoint,
            ocr_model_dir=args.ocr_model_dir,
            output_path=predictions_path,
            chunk_size=args.chunk_size,
            limit=args.limit,
            ocr_image_size=args.ocr_image_size,
            resume=args.resume,
        )

    if args.split == "val":
        config = tune_thresholds(
            rows,
            args.reports_dir / "hybrid_threshold_search.csv",
        )
        args.config.parent.mkdir(parents=True, exist_ok=True)
        args.config.write_text(
            json.dumps(
                {
                    "minimum_score": config.minimum_score,
                    "minimum_margin": config.minimum_margin,
                    "ocr_image_size": args.ocr_image_size,
                    "tuned_on_split": "val",
                    "validation_samples": len(rows),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        config = HybridOCRConfig.from_path(args.config)

    resolved = apply_config(rows, config)
    write_csv(args.reports_dir / f"hybrid_{args.split}_resolved.csv", resolved)
    plot_confusion(
        args.reports_dir / f"confusion_hybrid_{args.split}.png",
        resolved,
        "final_label",
        f"Hybrid model + OCR - {args.split}",
    )
    write_report(
        args.reports_dir / f"hybrid_{args.split}_report.md",
        args.split,
        resolved,
        config,
    )
    print(
        f"{args.split}: model={accuracy(resolved, 'model_label'):.4f} "
        f"ocr={accuracy(resolved, 'ocr_label'):.4f} "
        f"hybrid={accuracy(resolved, 'final_label'):.4f} "
        f"score>={config.minimum_score:.2f} margin>={config.minimum_margin:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
