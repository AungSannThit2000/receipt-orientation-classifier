# Draft 2 - Receipt Extraction and Relabelling Report

## Label definition

| Label | Clockwise range from upright |
| --- | --- |
| `upright` | 315 to 360 degrees, or 0 to less than 45 degrees |
| `tilted_right` | 45 to less than 135 degrees |
| `upside_down` | 135 to less than 225 degrees |
| `tilted_left` | 225 to less than 315 degrees |

Generated samples use class centers with controlled random jitter. Exact class-boundary images are not generated.

## Source processing

- Source files found: 200
- Unique sources after exact deduplication: 198
- Duplicate files excluded: 2
- Orientation-approved sources: 198
- Sources requiring orientation review: 0
- Generated 224 x 224 images: 2376

## Receipt extraction methods

| Method | Sources |
| --- | ---: |
| `edges` | 5 |
| `full_frame_fallback` | 20 |
| `low_saturation` | 36 |
| `neutral_bright` | 54 |
| `otsu` | 83 |

A full-frame fallback means the detector could not confidently separate a paper boundary. These images are retained because many dataset files are already tightly cropped to the receipt.

## OCR top-side decisions

| Counter-clockwise correction | Sources |
| --- | ---: |
| 0 degrees | 198 |
| 90 degrees | 0 |
| 180 degrees | 0 |
| 270 degrees | 0 |

OCR first compares upright with upside down, then checks the two sideways rotations when that result is weak or ambiguous. Low-score or closely tied decisions are excluded until manually reviewed.

## Generated dataset

| Split | Approved sources | Generated images |
| --- | ---: | ---: |
| `train` | 139 | 1668 |
| `val` | 30 | 360 |
| `test` | 29 | 348 |

| Split | Upright | Right | Upside down | Left |
| --- | ---: | ---: | ---: | ---: |
| `train` | 417 | 417 | 417 | 417 |
| `val` | 90 | 90 | 90 | 90 |
| `test` | 87 | 87 | 87 | 87 |

## Quality-control files

- `docs/crop_qa_grid.jpg`: original versus extracted/canonical receipt examples.
- `docs/orientation_sample_grid.jpg`: examples of the four final labels.
- `docs/review_required_grid.jpg`: uncertain OCR decisions, when any exist.
- `docs/full_frame_fallback_grid.jpg`: sources where no separate paper boundary was applied.
- `data/manifests/source_analysis.csv`: crop and OCR measurements for every source.
- `data/manifests/review_required.csv`: sources blocked from training pending review.

## Inference rule

Streamlit must call the same receipt extraction and 224 x 224 canvas functions. It must preserve the uploaded receipt's observed rotation; OCR must not rotate the test image before the classifier receives it.
