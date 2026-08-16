# Streamlit Community Cloud deployment

## Repository requirements

The deployable files are at repository root:

```text
app.py
requirements.txt
packages.txt
.streamlit/config.toml
models/detection/receipt_yolov8s_worldv2.pt
models/trained/mobilenet_v3_small_finetune_full_best.pt
```

The receipt dataset is not required for inference and is excluded by `.gitignore`.

## Deploy

1. Sign in to Streamlit Community Cloud with the GitHub account that administers the repository.
2. Choose **Create app** and select the repository.
3. Set the branch to `main` and entrypoint to `app.py`.
4. Open **Advanced settings** and select Python 3.10.
5. No secrets are required.
6. Deploy and watch the build logs until the app is healthy.

## First-run behavior

The classifier and receipt detector are committed with the repository. EasyOCR model files are not committed because the Thai recognizer is larger than GitHub's normal file limit. The first OCR-enabled prediction downloads missing Thai, English, and text-detection weights into `models/easyocr`. Streamlit caches the initialized verifier for the process lifetime.

The first prediction can therefore take several minutes. Later predictions in the same running instance avoid download and initialization work but OCR itself remains CPU intensive.

## Dependency design

`requirements.txt` contains only application dependencies. Training-only Matplotlib and scikit-learn are separated into `requirements-training.txt` to reduce cloud build time and memory.

The application declares `opencv-python-headless` for server use. Ultralytics also installs `opencv-python` transitively, so `packages.txt` installs the Linux `libgl1` runtime library required when that OpenCV wheel is imported on Streamlit Community Cloud.

## Resource considerations

- Streamlit caches the classifier, detector, and OCR reader with `st.cache_resource`.
- Uploads are limited to 20 MB and 40 megapixels.
- Object detection and OCR can be disabled independently for diagnostics.
- OCR runs only after a new image or relevant pipeline setting changes.
- A cold instance must redownload OCR files because Community Cloud storage is ephemeral.

## Troubleshooting

### Missing checkpoint

Confirm both committed `.pt` files exist at the paths shown above. The classifier cannot run without its selected checkpoint. If the detector is unavailable, the app warns and falls back to OpenCV contour extraction.

### OCR download or initialization failure

Check outbound network access and available disk. The app will keep the classifier result and display an OCR fallback warning.

### Out of memory

Disable OCR for a model-only diagnostic. If memory remains insufficient, use English-only OCR for a separate deployment or host the full multilingual service on a larger instance. Do not silently remove Thai OCR while presenting the existing multilingual metrics.

### Slow prediction

Cold-start downloads, CPU OCR, and a large uploaded image are the main causes. The app displays detector, classifier, and OCR timing separately so the bottleneck can be identified.

## Updating

Community Cloud watches the GitHub branch. Source changes update the app automatically; dependency changes trigger a full rebuild. Revalidate all public sample images after changing preprocessing, class order, checkpoints, or OCR thresholds.
