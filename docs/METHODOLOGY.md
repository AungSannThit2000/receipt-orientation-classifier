# Methodology

## 1. Problem definition

The task is a four-class image classification problem. The target is the receipt's observed top direction, not its exact deskew angle:

- `upright`: centered around 0 degrees clockwise.
- `tilted_right`: centered around 90 degrees clockwise.
- `upside_down`: centered around 180 degrees clockwise.
- `tilted_left`: centered around 270 degrees clockwise.

Class boundaries occur at 45, 135, 225, and 315 degrees. Generated training examples avoid exact boundaries because those cases are visually ambiguous.

## 2. Why classification alone was insufficient

A receipt silhouette is approximately rectangular and often symmetric after resizing. At 224 x 224 pixels, faint text can lose the detail needed to distinguish opposite directions. The initial model therefore learned the vertical/horizontal axis reliably but made more mistakes within opposite pairs, especially `upright` versus `upside_down`.

The final system separates these two decisions:

1. A CNN predicts one of the four classes and therefore selects an axis.
2. OCR compares only the two directions on that axis.
3. A calibrated rule confirms, overrides, or abstains.

This uses the CNN for layout and visual structure while using readable text as independent evidence for polarity.

## 3. Data preparation

### 3.1 Source audit

The source folder contained 200 receipt images. SHA-256 exact deduplication removed two files, leaving 198 unique sources.

### 3.2 Receipt extraction

`src/receipt_preprocessing.py` builds candidate masks from grayscale brightness, low saturation, neutral bright pixels, and edges. Candidate contours are scored using:

- area relative to the image;
- rectangularity;
- convexity;
- aspect ratio;
- distance from the image center;
- border contact.

The best valid quadrilateral is perspective-corrected. If no candidate passes the quality rules, the full frame is retained instead of inventing an unreliable crop.

### 3.3 Canonical top-side verification

OCR was used during data preparation to check whether each original source was upright. Manual overrides are supported through `config/orientation_overrides.csv`. OCR does not rotate test images before classification because that would erase the target being predicted.

### 3.4 Grouped split

The physical source receipt is the split unit. All generated rotations from one source remain together in train, validation, or test.

| Split | Source receipts | Generated images | Images per class |
| --- | ---: | ---: | ---: |
| Train | 139 | 1,668 | 417 |
| Validation | 30 | 360 | 90 |
| Test | 29 | 348 | 87 |

This prevents leakage that would occur if nearly identical rotations of one receipt appeared in both training and evaluation.

### 3.5 Rotation generation

Each source generates three samples for every class, for 12 samples per source and 2,376 samples overall. Rotations use class-centered random jitter. The rotated receipt is placed on a neutral gray square canvas without stretching its aspect ratio.

## 4. Training experiments

Four experiments compared architecture and crop policy:

| Experiment | Data policy | Validation | Full test | Checkpoint |
| --- | --- | ---: | ---: | ---: |
| Simple CNN | All approved crops | 79.72% | 87.36% | 1.22 MB |
| Simple CNN | Strict crops only | 83.67% | 88.79% | 1.22 MB |
| MobileNetV3 Small | All approved crops | **93.33%** | **92.24%** | 5.94 MB |
| MobileNetV3 Small | Strict crops only | 92.33% | 88.79% | 5.94 MB |

The selected network starts from ImageNet MobileNetV3 Small weights. The final three feature blocks and classifier are fine-tuned. Training uses cross-entropy with 0.05 label smoothing, AdamW, cosine learning-rate scheduling, deterministic seed 4201, and early stopping.

The full-data MobileNet performed best. Keeping fallback crops added appearance variation that was useful despite weaker boundary confidence.

## 5. Real-time localization

Uploaded photographs contain tables, hands, and other background objects that were less common in the source dataset. The production path therefore adds a YOLO-World receipt/document detector before OpenCV refinement.

1. Detect the most suitable receipt or document box.
2. Pad the box by 2%.
3. Run the contour extractor inside that proposal.
4. Map the refined polygon back to original coordinates.
5. Fall back to full-image contour extraction if object detection fails.

The detector checkpoint contains its fixed vocabulary, so the large CLIP text encoder is not needed during inference.

## 6. Model inference

The extracted receipt is padded to a 224 x 224 gray canvas and normalized with the ImageNet mean and standard deviation stored in the checkpoint. Softmax produces probabilities for all four classes. The top probability is confidence; the difference between the top two probabilities is the model margin.

No quarter-turn normalization occurs before this prediction.

## 7. OCR verification

EasyOCR uses Thai and English recognition at a 640-pixel square input. For speed, the verifier:

- detects text once for the model-selected pair;
- keeps the 24 largest text regions;
- reuses transformed text boxes for the opposite rotation;
- recognizes regions in a CPU batch;
- scores readable length, OCR confidence, alphabetic content, and receipt keywords.

For each candidate direction, the OCR score is approximately:

```text
sum(confidence * min(text_length, 30)^0.85)
+ alphabetic bonuses
+ 7.5 * receipt_keyword_hits
```

The relative OCR margin is:

```text
(best_score - second_score) / max(best_score, 1)
```

## 8. Decision policy

The current runtime configuration is provisional:

- strict override: OCR score at least 5.0 and margin at least 0.35;
- same-label consensus: OCR score at least 4.0, OCR margin at least 0.50, model confidence at least 0.55, and model margin at least 0.15;
- otherwise: return `Uncertain` and do not auto-rotate.

A weak OCR result cannot override a disagreeing model. A slightly weaker result may confirm the model only when both sources independently have adequate evidence.

## 9. Correction output

After a final label exists, the extracted receipt is rotated to upright:

| Final label | Correction |
| --- | --- |
| Upright | None |
| Tilted right | 90 degrees counter-clockwise |
| Upside down | 180 degrees |
| Tilted left | 90 degrees clockwise |

An uncertain result is never automatically corrected.

## 10. Evaluation design

Three evaluation layers are reported:

1. Generated source-group validation and test splits for controlled comparison.
2. Dataset-free generated demo receipts for public reproducibility.
3. External real-photo regression checks that are kept outside training.

The external set currently contains two photos of one physical Thai receipt, so it verifies a known failure but cannot estimate deployment accuracy.
