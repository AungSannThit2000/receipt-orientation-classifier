# Phase 6 Real-Time Receipt Detection Report

## Scope

This phase adds receipt object detection only to uploaded-image inference. Training images, generated labels, manifests, validation/test splits, and orientation checkpoints were not changed.

## Pipeline

1. YOLO-World searches the uploaded image using a persisted vocabulary containing `receipt`, `paper receipt`, `printed receipt`, and `document`.
2. The highest-confidence suitable box is padded by 2% of the image dimension.
3. The existing OpenCV detector refines the receipt inside that smaller proposal.
4. The refined receipt is sent directly to the existing MobileNetV3 classifier and OCR verifier.
5. If YOLO-World finds no suitable object, the original full-image contour detector is used.

The detector checkpoint is `models/detection/receipt_yolov8s_worldv2.pt`. Its vocabulary is embedded, so the large CLIP text encoder is not required during application inference.

## External Photo Result

Test image: `external_thai_view_1.jpg`

| Measurement | Original contour path | Object-detection path |
| --- | ---: | ---: |
| Crop area relative to photo | 59.7% | 19.7% |
| Extracted size | 2218 x 2055 | 872 x 1735 |
| Model result | Upright, 78.3% | Upright, 96.8% |
| Actual orientation | Upside down | Upside down |

YOLO-World detected the document at `(845, 600, 1677, 2346)` with 26.3% confidence. Detector inference took approximately 0.35 seconds on CPU after model initialization.

The new crop is materially better: the receipt fills the model input and most hand/background pixels are removed. It does not correct this orientation prediction because the unchanged model still does not generalize to this handheld Thai receipt. OCR weakly preferred upside down but remained below its validated override margin.

## Regression Check

The four quarter-turn variants of a held-out receipt image remained correctly classified. YOLO-World detected one variant and the other three used the contour fallback, which confirms that a detector miss does not break the established path.

## Application Changes

The Streamlit inspection view now includes:

- Uploaded image
- Object-detection overlay
- Refined receipt crop
- Final 224 x 224 model input
- Object status and confidence
- Detector time and crop method

The sidebar `Receipt object detection` toggle permits direct comparison with the previous real-time preprocessing path.

## Limitation

Object detection solves localization, not text-direction generalization. Correcting the remaining example will require real Thai/handheld receipt training data, multilingual OCR, a higher-resolution orientation method, or a combination of those changes.
