from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageOps

from hybrid_orientation import HybridOCRConfig, HybridOrientationVerifier
from inference import ReceiptOrientationClassifier
from realtime_receipt_detection import YOLOWorldReceiptDetector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "real_photo_manifest.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "draft3" / "external_holdout_results.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate grouped external receipt photos.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as file:
        rows = [
            row for row in csv.DictReader(file) if row["split"] == "external_holdout"
        ]
    if not rows:
        raise ValueError("No external_holdout rows were found.")

    classifier = ReceiptOrientationClassifier()
    detector = YOLOWorldReceiptDetector()
    verifier = HybridOrientationVerifier(HybridOCRConfig.from_path())
    results = []
    for row in rows:
        path = PROJECT_ROOT / row["image"]
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        detection = detector.detect(image)
        prediction = classifier.predict_extraction(detection.extraction)
        verification = verifier.verify(
            prediction.extraction.image,
            prediction.label,
            model_confidence=prediction.confidence,
            model_margin=prediction.margin,
        )
        results.append(
            {
                **row,
                "model_label": prediction.label,
                "model_confidence": prediction.confidence,
                "ocr_label": verification.ocr_label,
                "ocr_score": verification.best_score,
                "ocr_margin": verification.margin,
                "final_label": verification.final_label,
                "decision_source": verification.decision_source,
                "correct": verification.final_label == row["label"],
                "abstained": verification.final_label is None,
                "crop_method": detection.extraction.method,
                "detector_confidence": detection.detector_confidence,
                "ocr_seconds": verification.ocr_ms / 1000.0,
            }
        )
        print(
            f"{path.name}: model={prediction.label} ocr={verification.ocr_label} "
            f"final={verification.final_label}",
            flush=True,
        )

    resolved = [row for row in results if not row["abstained"]]
    summary = {
        "samples": len(results),
        "unique_receipt_groups": len({row["receipt_group"] for row in results}),
        "coverage": len(resolved) / len(results),
        "model_accuracy": sum(row["model_label"] == row["label"] for row in results)
        / len(results),
        "resolved_accuracy": (
            sum(bool(row["correct"]) for row in resolved) / len(resolved)
            if resolved
            else None
        ),
        "warning": "Both samples show one physical receipt; this is a regression check, not an accuracy estimate.",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
