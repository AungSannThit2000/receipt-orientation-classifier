# Draft 3 Report

## Problem found

The selected 224 px classifier learned the generated English-receipt domain well but did not generalize to the photographed Thai receipt. Object detection produced a useful receipt crop in both supplied photos, yet the classifier still predicted `upright` for the upside-down receipt. The failure is therefore not only cropping: text became small at 224 px, the training augmentation did not represent camera scenes well, and the original training set contained no Thai receipts.

## Implemented inference changes

- Kept YOLO-World plus OpenCV receipt extraction before classification.
- Changed EasyOCR from English-only to Thai + English.
- Uses a 640 px OCR input as the tested speed/quality compromise.
- Counted Unicode and Thai text when scoring orientation evidence.
- Detects text once and reuses those boxes for the opposite rotation.
- Recognizes only the 24 strongest text regions with larger CPU batches.
- Added an absolute OCR score threshold as well as a relative margin threshold.
- Uses separate strict override and model-plus-OCR consensus thresholds.
- Added an explicit `Uncertain` result that prevents automatic rotation when OCR is weak.

## External regression result

| Photo | Base model | OCR | Final | OCR score | Margin |
| --- | --- | --- | --- | ---: | ---: |
| `external_thai_view_1.jpg` | upright | upside down | upside down | 5.45 | 82.38% |
| `external_thai_view_2.jpg` | upright | upside down | upside down | 38.96 | 96.39% |

Both corrections are successful, but the two images show the same physical receipt. This is a regression check with one unique receipt group, not a valid estimate of accuracy. The optimized checks took 27.8 and 24.5 seconds, compared with approximately 36-44 seconds before optimization. The current score threshold of 5 and margin threshold of 0.35 are provisional.

An additional upside-down photo produced model confidence 58.9%, model margin 21.6%, OCR score 4.64, and OCR margin 77.4%. Because the model and OCR agree, the consensus rule accepts `upside_down`; the same OCR score would remain insufficient to override a disagreeing model. Consensus currently requires OCR score 4, OCR margin 50%, model confidence 55%, and model margin 15%.

## Prepared retraining path

- Rebuilt all 2,376 generated samples at 384 x 384.
- Preserved 198 source receipt groups and verified that no source crosses splits.
- Added stronger camera-condition augmentation without invalid quarter-turn transformations.
- Added a dedicated upright-versus-upside-down MobileNetV3 trainer.
- Added a grouped real-photo intake manifest and detector-based normalization script.
- Kept the two current Thai images in `external_holdout`, outside training.

Full retraining has intentionally not been presented as the fix yet. Training another model from the same synthetic English receipt sources would mostly reproduce the same domain limitation. The 384 px trainers are ready for use after collecting diverse real photos.

## Required real dataset

Target at least 50 different physical receipts. Capture each receipt in all four orientations and in two varied scenes, giving about 400 labeled photos. Include Thai and English receipts, several phones, close and distant framing, shadows, glare, hands, patterned backgrounds, and mild perspective distortion.

Assign one `receipt_group` per physical receipt and split by that group, approximately 70% train, 15% validation, and 15% test. Never place different photos of the same receipt in multiple splits. Keep a final external holdout untouched until model and OCR thresholds are locked.

## Verification completed

- 18 unit tests pass.
- Four-class 384 px camera-augmentation smoke test passes.
- Two-class vertical-direction 384 px smoke test passes.
- All sampled rebuilt images are 384 x 384.
- Both supplied external images are corrected to `upside_down` by the production inference path.
