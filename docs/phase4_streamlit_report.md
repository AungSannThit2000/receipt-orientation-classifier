# Phase 4 Streamlit Report

> Phase 5 extends this application with post-classification OCR verification. See `phase5_hybrid_ocr_report.md` for the current decision pipeline and accuracy.

## Application

The local Streamlit application is implemented in `app.py`. It loads the selected `mobilenet_v3_small_finetune_full_best.pt` checkpoint through `src/inference.py`.

For every uploaded JPG, PNG, or WebP image, the application:

1. Applies EXIF orientation and converts the image to RGB.
2. Detects, masks, and crops the receipt with the same OpenCV code used to prepare the training dataset.
3. Places the extracted receipt on the same 224 x 224 gray model canvas.
4. Applies the normalization values stored in the checkpoint.
5. Reports the predicted orientation, confidence, all four class probabilities, crop diagnostics, and model execution time.
6. Produces an optional quarter-turn corrected receipt for preview and PNG download.

OCR does not rotate uploaded images before classification. This preserves the orientation that the model is intended to predict.

## Run

From the project root:

```powershell
python -m streamlit run .\app.py
```

The default local URL is `http://localhost:8501`.

## Validation

- Python compilation passed for `app.py`, `src/inference.py`, and the test module.
- Three inference tests passed, including all four quarter-turn rotations of a held-out receipt image.
- Streamlit's application test loaded the empty state with one uploader and no exceptions.
- A live upload of the held-out receipt image produced `Upright` at 95.2% confidence.
- A live held-out tilted-right image produced `Tilted right` at 98.3% confidence and the correct counter-clockwise correction action.
- Desktop, 390 px mobile, and 320 px narrow-mobile layouts had no horizontal overflow.
- The live browser console contained no errors or warnings.

## Files

- `app.py`: Streamlit interface and upload handling.
- `src/inference.py`: checkpoint loading, preprocessing, prediction, and correction logic.
- `tests/test_inference.py`: checkpoint and end-to-end quarter-turn tests.
- `.streamlit/config.toml`: local theme and upload configuration.
