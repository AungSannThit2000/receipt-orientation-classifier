# Model card: Receipt Orientation Hybrid Classifier

## Summary

This educational system classifies receipt images into upright, tilted right, upside down, or tilted left. It combines a four-class MobileNetV3 Small model with receipt localization and pairwise Thai/English OCR verification.

## Intended use

- Demonstrating image classification, transfer learning, preprocessing, OCR, and local/cloud deployment in an AI concepts course.
- Detecting quarter-turn orientation before a downstream receipt-reading workflow.
- Comparing CNN-only and hybrid decision systems.

## Out-of-scope use

- Financial validation, tax compliance, fraud detection, or receipt authenticity.
- Extracting or storing payment information.
- Safety-critical automatic document processing without human review.
- Continuous exact-angle estimation.

## Inputs and outputs

Input: one JPG, PNG, or WebP photograph up to 20 MB and 40 megapixels.

Output: one of four orientation labels or `Uncertain`, model probabilities, OCR evidence, crop diagnostics, and an optional corrected receipt image.

## Model architecture

- Backbone: MobileNetV3 Small initialized from ImageNet.
- Fine-tuning: final three feature blocks and classifier.
- Input: 224 x 224 RGB with ImageNet normalization.
- Classes: `upright`, `tilted_right`, `upside_down`, `tilted_left`.
- Checkpoint size: 5.94 MB.
- Localization: fixed-vocabulary YOLO-World plus OpenCV boundary refinement.
- Verification: EasyOCR Thai and English, pairwise opposite-direction scoring.

## Training data

The private source collection contained 198 unique upright receipt images after exact deduplication. Controlled rotation generated 2,376 balanced samples. Splits were grouped by source receipt to prevent leakage.

The dataset is excluded from this repository. Users retraining the project are responsible for dataset licensing, consent, and removal of sensitive information.

## Performance

- Selected training experiment full test: 92.24% accuracy and 92.26% macro F1.
- Production-path model-only test during OCR evaluation: 92.82%.
- Hybrid test: 98.28% on 348 generated, source-group-held-out samples.
- External regression: two upside-down Thai views of one receipt corrected from wrong model predictions to correct hybrid results.

Generated-domain metrics should not be interpreted as unrestricted camera-photo accuracy.

## Known risks

- Domain shift across languages, printers, paper shapes, cameras, and backgrounds.
- OCR latency and cold-start download time.
- Pairwise OCR cannot repair a model-selected wrong axis.
- Receipt images may contain personal or financial information.
- Current OCR thresholds are provisional because the real-photo validation set is too small.

## Mitigations

- Object detection removes background clutter before classification.
- OCR provides independent text-direction evidence.
- Strict override thresholds prevent weak OCR from changing a model result.
- A high-confidence agreement rule accepts matching model/OCR directions only when model confidence, model margin, and OCR relative margin all pass strict gates.
- Explicit abstention prevents automatic rotation when evidence is inadequate.
- Uploaded data is not intentionally persisted by application code.
- The UI exposes crop and decision evidence for human inspection.

## Recommended evaluation before broader use

Collect at least 50 physical receipts across languages and conditions, preserve group-separated train/validation/test splits, calibrate thresholds only on validation, and report per-class recall, confusion matrix, latency distribution, OCR coverage, and abstention rate on the untouched real-photo test set.
