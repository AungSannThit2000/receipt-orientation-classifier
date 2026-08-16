# Code guide

This guide describes the intention, inputs, outputs, and execution order of the project files.

## Application entry point

### `app.py`

Runs the Streamlit interface. It validates uploads, caches model resources, executes object detection, classification, and OCR verification, then presents:

- final label and decision source;
- four model probabilities;
- object-detection overlay and extracted crop;
- OCR direction scores and recognized text preview;
- processing time;
- corrected receipt preview and PNG download.

Streamlit session keys are based on the upload hash and enabled pipeline options so changing tabs does not repeat expensive inference.

## Runtime modules

### `src/receipt_preprocessing.py`

Shared preprocessing for dataset preparation and inference. It applies EXIF orientation, generates OpenCV masks, scores receipt contours, performs perspective extraction, rotates with a fixed fill color, and creates the square model canvas. Sharing this code reduces train/serve skew.

### `src/realtime_receipt_detection.py`

Loads the persisted YOLO-World detector, chooses a suitable receipt/document box, refines it with the shared OpenCV extractor, maps coordinates to the original image, and draws the inspection overlay. It returns a contour fallback when no object passes the threshold.

### `src/inference.py`

Loads checkpoint metadata and model weights, validates the class order, rebuilds MobileNetV3 or the small CNN, prepares the tensor, runs softmax inference, calculates confidence and margin, and creates the correction preview.

### `src/ocr_orientation.py`

Wraps EasyOCR. It prepares color or enhanced OCR inputs, limits recognition to the largest text regions, transforms boxes for the opposite rotation, scores recognized text, and returns the best direction with its evidence.

### `src/hybrid_orientation.py`

Defines candidate opposite-direction pairs and the final decision rules. It converts OCR rotations to class labels and returns one of: OCR confirmed, OCR override, model/OCR consensus, model fallback, or uncertain.

## Dataset preparation

### `src/prepare_dataset.py`

The original pipeline for raw receipts. It scans readable images, computes hashes, removes exact duplicates, assigns source-group splits, extracts paper boundaries, evaluates canonical orientation, applies manual overrides, creates balanced rotations, writes manifests, and builds QA grids.

Main outputs, which are intentionally excluded from GitHub:

```text
data/canonical/{train,val,test}/
data/processed/{train,val,test}/{class}/
data/manifests/source_analysis.csv
data/manifests/generated_manifest.csv
```

### `src/build_384_dataset.py`

Rebuilds already-approved canonical crops at 384 x 384 using camera-style augmentations. It validates that no source receipt crosses a split before writing `generated_manifest_384.csv`.

### `src/prepare_real_photos.py`

Reads a manually maintained real-photo manifest, checks required labels and physical receipt group consistency, detects/crops each receipt, and prepares real photos for future grouped retraining. External holdout rows remain excluded from training.

### `src/preview_fallback_recrops.py`

Creates a visual comparison of old full-frame fallback inputs and candidate recrops. It is a QA utility, not part of application inference.

## Training and evaluation

### `src/train_models.py`

Trains the four experiment combinations. It constructs transforms and loaders from a manifest, builds models, freezes or unfreezes the intended layers, runs early-stopped training, evaluates all common test subsets, saves self-describing checkpoints, and writes curves, confusion matrices, CSV, JSON, and Markdown comparisons.

### `src/evaluate_hybrid.py`

Runs classifier and OCR predictions on a labeled split, searches OCR thresholds on validation data, locks the selected policy, evaluates the untouched test set, and writes resolved predictions and confusion matrices.

### `src/train_vertical_direction.py`

Optional two-class research path for only `upright` versus `upside_down`. It was prepared for future real-photo retraining but is not used by the deployed app.

### `src/evaluate_external_photos.py`

Runs the exact production path on manually labeled real photographs and records model, detector, OCR, final decision, correctness, and timing. The current rows form a regression check, not a representative test set.

## Repository and report utilities

### `scripts/generate_demo_assets.py`

Creates the four fictional public receipt examples under `docs/assets`. The generated samples contain no source-dataset or personal receipt content and are safe to publish with the documentation.

### `scripts/build_project_report.py`

Builds `docs/Receipt_Orientation_Classifier_Project_Report.docx` from the aggregate metrics, charts, screenshot, methodology, and code inventory. It applies the report styles, tables, captions, page numbering, and image alt text programmatically.

## Configuration

### `config/hybrid_ocr_config.json`

Runtime OCR language, image size, region limit, batch size, strict override thresholds, consensus thresholds, and abstention policy.

### `config/draft3_training.json`

Prepared 384-pixel training plan and grouped real-photo collection targets.

### `config/orientation_overrides.csv`

Auditable manual correction layer for canonical source orientation.

### `.streamlit/config.toml`

App theme, headless server behavior, upload limit, and usage-statistics setting.

## Tests

### `tests/test_inference.py`

Checks checkpoint metadata, complete probability output, correction artifacts, and inference on the public generated examples.

### `tests/test_hybrid_orientation.py`

Checks candidate pairs, opposite-direction overrides, confirmations, consensus, abstention, and fallback behavior without loading OCR models.

### `tests/test_ocr_orientation.py`

Checks OCR region selection, enhanced preprocessing, strong-result fast paths, and multilingual text scoring without invoking full OCR.

### `tests/test_realtime_receipt_detection.py`

Checks detector candidate selection, coordinate mapping, and proposal fallback using deterministic fake boxes.

## Typical execution order

For a fresh research run:

```powershell
python src/prepare_dataset.py --source-dir <dataset> --overwrite
python src/train_models.py
python src/evaluate_hybrid.py
python -m unittest discover -s tests -v
python scripts/build_project_report.py
python -m streamlit run app.py
```

For application use with the included checkpoints, only the final two commands are required.
