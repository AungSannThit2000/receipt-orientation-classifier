# Phase 5 Hybrid OCR Report

> Phase 6 adds inference-only receipt object detection before this classifier and OCR pipeline. See `phase6_realtime_detection_report.md`.

## Objective

The classifier occasionally confused opposite orientations: right with left, or upright with upside down. Receipt text gives OCR an independent direction signal, but unrestricted four-way OCR can also choose the wrong vertical or horizontal axis.

The hybrid pipeline therefore separates the decision into two parts:

1. MobileNetV3 selects the vertical pair (`upright`, `upside_down`) or horizontal pair (`tilted_right`, `tilted_left`).
2. EasyOCR scores only the two opposite text directions in that pair.
3. OCR overrides the model only when its relative score margin passes a threshold tuned on validation data.
4. An inconclusive OCR result falls back to the model.

OCR runs after classification on the extracted, high-resolution receipt. It never normalizes the uploaded image before the model sees it.

## Validation tuning

The threshold search used all 360 validation samples and selected:

- Minimum OCR score: `0.00`
- Minimum relative score margin: `0.22`
- OCR canvas size: `384 x 384`

Validation performance:

| Method | Accuracy |
| --- | ---: |
| Model only | 92.50% |
| Pairwise OCR only | 96.94% |
| Hybrid | 97.22% |

The hybrid method made 21 helpful overrides, 4 harmful overrides, and retained the model for 18 inconclusive OCR comparisons.

## Untouched test result

The validation-selected rule was locked before evaluating the 348-image source-held-out test split.

| Method | Correct | Accuracy |
| --- | ---: | ---: |
| Model only | 323 / 348 | 92.82% |
| Pairwise OCR only | 335 / 348 | 96.26% |
| Hybrid | 342 / 348 | 98.28% |

The hybrid test decisions included 307 OCR confirmations, 22 OCR overrides, and 19 inconclusive fallbacks. Nineteen overrides corrected model errors and none changed a correct model prediction into an incorrect result.

## Remaining limitation

The six remaining test errors are mainly cases where the classifier selected the wrong axis. Pairwise OCR intentionally cannot cross from the vertical pair to the horizontal pair, or vice versa. This restriction prevents OCR from taking over the whole classification task and was more reliable on the held-out data.

## Application behavior

The Streamlit app displays the final hybrid orientation, its decision source, the original four model probabilities, both pairwise OCR scores, OCR margin, recognized-text preview, and the final correction. If OCR cannot initialize or its score margin is too low, the classifier result remains available.

Evaluation artifacts are stored under `reports/hybrid`, including resolved predictions, threshold search results, Markdown summaries, and confusion matrices.
