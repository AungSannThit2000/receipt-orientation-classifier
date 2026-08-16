# Phase 3 Training Report

## Goal

Compare two model architectures and two dataset policies for the four receipt orientation classes:

- `upright`
- `tilted_right`
- `upside_down`
- `tilted_left`

The source receipt, rather than each generated rotation, is the split unit. This prevents rotations of the same receipt from appearing in both training and evaluation data.

## Experiments

| Experiment | Training variant | Epochs run | Best epoch | Best validation | Full test | Macro F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `simple_cnn_full` | All approved crops | 18 | 18 | 79.72% | 87.36% | 87.31% |
| `simple_cnn_strict` | Excludes full-frame fallbacks | 18 | 17 | 83.67% | 88.79% | 88.78% |
| `mobilenet_v3_small_finetune_full` | All approved crops | 10 | 7 | 93.33% | **92.24%** | **92.26%** |
| `mobilenet_v3_small_finetune_strict` | Excludes full-frame fallbacks | 6 | 3 | 92.33% | 88.79% | 88.85% |

All four checkpoints were also evaluated on the same 300-image strict test subset and 48-image fallback-only subset. The fallback-only subset is too small for use as the headline metric.

## Selected Model

Use `models/trained/mobilenet_v3_small_finetune_full_best.pt` for the application.

It achieved:

- Full test accuracy: 92.24% (321 of 348 images)
- Full test macro F1: 92.26%
- Strict test accuracy: 93.00%
- Fallback-only test accuracy: 87.50%
- CPU model evaluation time: about 5.8 ms per prepared 224 x 224 image
- Checkpoint size: 5.94 MB

Per-class full-test accuracy:

| Class | Accuracy | Correct / total |
| --- | ---: | ---: |
| `upright` | 91.95% | 80 / 87 |
| `tilted_right` | 94.25% | 82 / 87 |
| `upside_down` | 87.36% | 76 / 87 |
| `tilted_left` | 95.40% | 83 / 87 |

The largest remaining error is `upside_down` being predicted as `upright` (11 images). This is the expected difficult case when the receipt shape is nearly symmetric and the printed text is faint or too small for the visual model to distinguish its direction.

## Interpretation

Partial MobileNetV3 fine-tuning was more effective than training the small CNN from scratch. Training MobileNet on all approved receipts was also better than removing the full-frame fallback sources. Those sources add useful appearance variation even though their receipt boundary was not detected confidently.

OCR was useful during dataset preparation to establish the canonical top side and reduce label noise. OCR should not automatically rotate an uploaded image before classification because that would remove the orientation the classifier is intended to predict. It can later be used as an optional confidence check after model prediction.

## Reproducibility

- Random seed: 4201
- Input size: 224 x 224 RGB
- Normalization: ImageNet mean and standard deviation
- Loss: cross entropy with 0.05 label smoothing
- Optimizer: AdamW with cosine learning-rate scheduling
- MobileNet policy: ImageNet initialization, final three feature blocks and classifier fine-tuned
- Evaluation sets: 348 full, 300 strict, and 48 fallback-only images

Detailed metrics, training histories, curves, and confusion matrices are in `reports/training`.
