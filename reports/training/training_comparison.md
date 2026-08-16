# Draft 2 Training Comparison

## Evaluation sets

- Full test set: 348 images.
- Strict-crop test set: 300 images.
- Full-frame fallback test set: 48 images.

The fallback test set is small, so its accuracy is diagnostic rather than a stable headline metric.

## Results

| Experiment | Training data | Best validation | Full test | Strict test | Fallback test | Size (MB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `mobilenet_v3_small_finetune_full` | full | 93.33% | 92.24% | 93.00% | 87.50% | 5.94 |
| `simple_cnn_strict` | strict | 83.67% | 88.79% | 89.00% | 87.50% | 1.22 |
| `mobilenet_v3_small_finetune_strict` | strict | 92.33% | 88.79% | 88.33% | 91.67% | 5.94 |
| `simple_cnn_full` | full | 79.72% | 87.36% | 87.00% | 89.58% | 1.22 |

## Best checkpoints

- Best full-test accuracy: `mobilenet_v3_small_finetune_full` at 92.24%.
- Best strict-test accuracy: `mobilenet_v3_small_finetune_full` at 93.00%.
- Best fallback-test accuracy: `mobilenet_v3_small_finetune_strict` at 91.67%.

Every checkpoint stores the class order, normalization values, image size, architecture, and dataset variant required for inference.
