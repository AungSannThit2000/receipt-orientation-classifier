# Results and limitations

## Architecture comparison

| Model | Training data | Validation | Full test | Macro F1 | Size |
| --- | --- | ---: | ---: | ---: | ---: |
| Simple CNN | Full | 79.72% | 87.36% | 87.31% | 1.22 MB |
| Simple CNN | Strict | 83.67% | 88.79% | 88.78% | 1.22 MB |
| MobileNetV3 Small | Full | **93.33%** | **92.24%** | **92.26%** | 5.94 MB |
| MobileNetV3 Small | Strict | 92.33% | 88.79% | 88.85% | 5.94 MB |

The selected model is `mobilenet_v3_small_finetune_full_best.pt`.

## Selected model by class

| Class | Full-test accuracy |
| --- | ---: |
| Upright | 91.95% |
| Tilted right | 94.25% |
| Upside down | 87.36% |
| Tilted left | 95.40% |

The largest model-only error was upside down being predicted as upright. This is consistent with rectangular symmetry and small text after 224-pixel resizing.

## Hybrid OCR evaluation

The production inference path was evaluated again while tuning and testing pairwise OCR:

| Split | Samples | Model only | Pairwise OCR only | Hybrid |
| --- | ---: | ---: | ---: | ---: |
| Validation | 360 | 92.50% | 96.94% | **97.22%** |
| Test | 348 | 92.82% | 96.26% | **98.28%** |

On the test split, OCR made 22 overrides. Nineteen corrected model errors and none changed a correct model prediction to an incorrect result under the validation-selected threshold policy. Nineteen OCR comparisons were inconclusive and fell back to the model in that evaluation.

## External real-photo regression

Two upside-down Thai photographs of one physical receipt were held outside training:

| Photo | Model | OCR | Final | Result |
| --- | --- | --- | --- | --- |
| External photo 1 | Upright, 96.8% | Upside down | Upside down | Correct |
| External photo 2 | Upright, 82.2% | Upside down | Upside down | Correct |

Object detection isolated the receipt, but model-only classification remained wrong. Thai/English OCR provided the polarity evidence needed to override it.

Because the two images show the same physical receipt, this is one regression case with two views. Reporting it as 100% real-world accuracy would be invalid.

## Public synthetic examples

All four generated demo images resolve correctly after OCR. The base model is low-confidence and wrong on three of the four, which makes these examples useful for demonstrating the hybrid method rather than inflating model-only performance.

## Current limitations

### Domain shift

Most training sources are upright English receipts with generated rotations. Handheld Thai receipts, patterned backgrounds, glare, folds, and strong perspective are underrepresented.

### Axis dependence

OCR compares only the model-selected pair. If the model chooses the wrong axis, pairwise OCR cannot cross from vertical to horizontal or vice versa.

### OCR latency

Thai/English OCR is CPU intensive. The current optimized external-photo checks took about 24-28 seconds locally. First cloud use is slower because language models must download and initialize.

### Threshold calibration

The current runtime thresholds were adjusted using a very small external regression case and are marked provisional. They need a larger grouped real-photo validation set before any production claim.

### No exact-angle estimate

The classifier returns a quadrant, not continuous skew. OCR line angles are diagnostic and limited to mild deskew evidence.

## Recommended next experiment

Collect at least 50 physical receipts with all four orientations in two scenes each. Split by receipt identity, fine-tune the 384-pixel model, then recalibrate OCR override and abstention thresholds on real-photo validation data. Report the final untouched real-photo test result with per-class recall, confusion matrix, latency percentiles, and abstention coverage.
