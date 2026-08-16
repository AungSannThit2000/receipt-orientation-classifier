from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hybrid_orientation import (  # noqa: E402
    DEFAULT_HYBRID_CONFIG,
    HybridOCRConfig,
    HybridOrientationVerifier,
    ROTATION_TO_LABEL,
)
from inference import (  # noqa: E402
    CORRECTION_LABELS,
    DEFAULT_CHECKPOINT,
    DISPLAY_NAMES,
    ReceiptOrientationClassifier,
    correct_receipt,
)
from realtime_receipt_detection import (  # noqa: E402
    ReceiptDetectionResult,
    YOLOWorldReceiptDetector,
)


MAX_IMAGE_PIXELS = 40_000_000
CLASS_COLORS = {
    "upright": "#0f766e",
    "tilted_right": "#2563eb",
    "upside_down": "#b42318",
    "tilted_left": "#b45309",
}
CROP_METHOD_NAMES = {
    "otsu": "Otsu",
    "low_saturation": "Low saturation",
    "neutral_bright": "Neutral",
    "edges": "Edges",
    "full_frame_fallback": "Full frame",
    "yolo_world_box": "YOLO-World box",
}
DECISION_SOURCE_NAMES = {
    "ocr_override": "OCR override",
    "ocr_confirmed": "OCR confirmed",
    "ocr_consensus": "Model + OCR",
    "uncertain_ocr_inconclusive": "Uncertain",
    "model_ocr_inconclusive": "Model fallback",
    "model_only": "Model only",
    "model_ocr_error": "Model fallback",
}


st.set_page_config(
    page_title="Receipt Orientation",
    page_icon=":material/receipt_long:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #17212b;
        --muted: #60707c;
        --line: #d9e0e3;
        --panel: #f4f6f7;
        --teal: #0f766e;
        --amber: #b45309;
        --red: #b42318;
    }
    [data-testid="stAppViewContainer"] > .main .block-container {
        max-width: 1160px;
        padding-top: 2.1rem;
        padding-bottom: 3rem;
    }
    [data-testid="stSidebar"] {
        border-right: 1px solid var(--line);
    }
    h1, h2, h3, p, label, button {
        letter-spacing: 0 !important;
    }
    h1 {
        font-size: 2rem !important;
        line-height: 1.15 !important;
        margin-bottom: 0.3rem !important;
    }
    h3 {
        font-size: 1.2rem !important;
        line-height: 1.3 !important;
    }
    .app-kicker {
        color: var(--teal);
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
    }
    .app-subtitle {
        color: var(--muted);
        font-size: 1rem;
        margin: 0 0 1.4rem 0;
    }
    .result-band {
        background: #eef6f4;
        border: 1px solid #bad8d2;
        border-left: 5px solid var(--teal);
        border-radius: 6px;
        padding: 1.1rem 1.25rem;
        margin: 1.15rem 0 1rem 0;
    }
    .result-eyebrow {
        color: #40635f;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }
    .result-title {
        color: var(--ink);
        font-size: 1.65rem;
        font-weight: 720;
        line-height: 1.2;
    }
    .result-detail {
        color: #475c59;
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }
    .probability-row {
        display: grid;
        grid-template-columns: minmax(120px, 170px) minmax(120px, 1fr) 64px;
        gap: 0.8rem;
        align-items: center;
        margin: 0.75rem 0;
    }
    .probability-label {
        color: var(--ink);
        font-size: 0.9rem;
        font-weight: 600;
        white-space: nowrap;
    }
    .probability-swatch {
        display: inline-block;
        width: 9px;
        height: 9px;
        border-radius: 2px;
        margin-right: 0.5rem;
    }
    .probability-track {
        background: #e5eaec;
        border-radius: 3px;
        height: 10px;
        overflow: hidden;
    }
    .probability-fill {
        height: 100%;
        border-radius: 3px;
    }
    .probability-value {
        color: #42515b;
        font-variant-numeric: tabular-nums;
        font-size: 0.85rem;
        text-align: right;
    }
    [data-testid="stFileUploaderDropzone"] {
        border: 1px dashed #9aa8af;
        border-radius: 6px;
        background: #fafbfb;
    }
    [data-testid="stMetric"] {
        border-top: 1px solid var(--line);
        padding-top: 0.7rem;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.55rem;
        line-height: 1.2;
        white-space: normal;
        overflow: visible;
        text-overflow: clip;
    }
    [data-testid="stImage"] img {
        border: 1px solid var(--line);
        border-radius: 4px;
    }
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 1.2rem;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        padding-left: 0;
        padding-right: 0;
    }
    @media (max-width: 700px) {
        [data-testid="stAppViewContainer"] > .main .block-container {
            padding-top: 1.2rem;
        }
        h1 { font-size: 1.65rem !important; }
        .result-title { font-size: 1.35rem; }
        .probability-row {
            grid-template-columns: minmax(105px, 135px) minmax(80px, 1fr) 52px;
            gap: 0.5rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_classifier() -> ReceiptOrientationClassifier:
    return ReceiptOrientationClassifier(DEFAULT_CHECKPOINT)


@st.cache_resource(show_spinner=False)
def load_ocr_verifier() -> HybridOrientationVerifier:
    config = HybridOCRConfig.from_path(DEFAULT_HYBRID_CONFIG)
    return HybridOrientationVerifier(config)


@st.cache_resource(show_spinner=False)
def load_receipt_detector() -> YOLOWorldReceiptDetector:
    return YOLOWorldReceiptDetector()


def read_uploaded_image(contents: bytes) -> Image.Image:
    if not contents:
        raise ValueError("The uploaded file is empty.")
    try:
        with Image.open(io.BytesIO(contents)) as image:
            width, height = image.size
            if width * height > MAX_IMAGE_PIXELS:
                raise ValueError("The image is too large. Use an image below 40 megapixels.")
            return image.copy()
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("The file is not a readable JPG, PNG, or WebP image.") from error


def image_as_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def probability_row(label: str, value: float) -> str:
    display_name = DISPLAY_NAMES[label]
    color = CLASS_COLORS[label]
    width = max(0.0, min(100.0, value * 100.0))
    return f"""
        <div class="probability-row">
            <div class="probability-label">
                <span class="probability-swatch" style="background:{color}"></span>{display_name}
            </div>
            <div class="probability-track" aria-label="{display_name} probability">
                <div class="probability-fill" style="width:{width:.2f}%;background:{color}"></div>
            </div>
            <div class="probability-value">{width:.1f}%</div>
        </div>
    """


def crop_method_name(method: str) -> str:
    if method.startswith("yolo_world+"):
        refinement = method.split("+", maxsplit=1)[1]
        refinement_name = CROP_METHOD_NAMES.get(
            refinement,
            refinement.replace("_", " ").title(),
        )
        return f"YOLO-World + {refinement_name}"
    return CROP_METHOD_NAMES.get(method, method.replace("_", " ").title())


with st.sidebar:
    st.markdown("### Decision pipeline")
    object_detection_enabled = st.toggle(
        "Receipt object detection",
        value=True,
        help="Locate the receipt before creating the classifier input.",
    )
    ocr_enabled = st.toggle(
        "OCR verification",
        value=True,
        help="Compare the two opposite text directions selected by the classifier.",
    )
    st.metric("Draft 3 calibration", "Provisional")
    st.caption("Thai + English OCR with an explicit uncertain result.")
    st.divider()
    st.markdown("### Base model")
    st.write("MobileNetV3 Small")
    st.metric("Model-only accuracy", "92.82%")
    st.caption("App-path benchmark before OCR verification.")
    st.divider()
    st.markdown("### Orientation classes")
    for class_name in ("upright", "tilted_right", "upside_down", "tilted_left"):
        st.markdown(
            f'<span class="probability-swatch" style="background:{CLASS_COLORS[class_name]}"></span>'
            f"{DISPLAY_NAMES[class_name]}",
            unsafe_allow_html=True,
        )

st.markdown('<div class="app-kicker">Computer vision workspace</div>', unsafe_allow_html=True)
st.title("Receipt orientation")
st.markdown(
    '<p class="app-subtitle">Classify the quarter-turn orientation of a receipt photo.</p>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Receipt image",
    type=("jpg", "jpeg", "png", "webp"),
    help="Maximum image size: 40 megapixels and 20 MB.",
)

if uploaded_file is None:
    st.info("Upload a receipt image to begin.", icon=":material/upload_file:")
    st.stop()

uploaded_bytes = uploaded_file.getvalue()
upload_key = hashlib.sha256(uploaded_bytes).hexdigest()

try:
    uploaded_image = read_uploaded_image(uploaded_bytes)
    classifier = load_classifier()
    prediction_cache_key = (
        f"{upload_key}:object_detection={int(object_detection_enabled)}"
    )
    if st.session_state.get("prediction_upload_key") != prediction_cache_key:
        with st.spinner("Detecting receipt and classifying orientation..."):
            detection_result: ReceiptDetectionResult | None = None
            detection_error: str | None = None
            if object_detection_enabled:
                try:
                    detection_result = load_receipt_detector().detect(uploaded_image)
                except (
                    ImportError,
                    FileNotFoundError,
                    RuntimeError,
                    ValueError,
                    OSError,
                ) as error:
                    detection_error = str(error)

            if detection_result is not None:
                prediction = classifier.predict_extraction(detection_result.extraction)
            else:
                prediction = classifier.predict(uploaded_image)

            st.session_state["prediction_result"] = prediction
            st.session_state["detection_result"] = detection_result
            st.session_state["detection_error"] = detection_error
            st.session_state["prediction_upload_key"] = prediction_cache_key
    result = st.session_state["prediction_result"]
    detection_result = st.session_state.get("detection_result")
    detection_error = st.session_state.get("detection_error")
except (ValueError, FileNotFoundError, RuntimeError) as error:
    st.error(str(error), icon=":material/error:")
    st.stop()

ocr_result = None
ocr_error = None
if ocr_enabled:
    ocr_cache_key = f"{prediction_cache_key}:{result.label}"
    try:
        if st.session_state.get("ocr_upload_key") != ocr_cache_key:
            with st.spinner("Model is extracting features from photo....."):
                verifier = load_ocr_verifier()
                st.session_state["ocr_result"] = verifier.verify(
                    result.extraction.image,
                    result.label,
                    model_confidence=result.confidence,
                    model_margin=result.margin,
                )
                st.session_state["ocr_upload_key"] = ocr_cache_key
        ocr_result = st.session_state["ocr_result"]
    except (ImportError, FileNotFoundError, RuntimeError, ValueError, OSError) as error:
        ocr_error = str(error)

final_label = ocr_result.final_label if ocr_result is not None else result.label
final_display_label = DISPLAY_NAMES[final_label] if final_label is not None else "Uncertain"
if final_label is None:
    final_corrected_receipt = result.extraction.image
else:
    final_corrected_receipt = correct_receipt(
        result.extraction.image,
        final_label,
        fill_color=classifier.fill_color,
    )
if ocr_result is not None:
    decision_source = ocr_result.decision_source
elif ocr_error:
    decision_source = "model_ocr_error"
else:
    decision_source = "model_only"

confidence_percent = result.confidence * 100.0
if decision_source == "ocr_override":
    result_detail = (
        f"OCR changed the model result from {result.display_label} "
        f"&middot; {ocr_result.margin * 100:.1f}% OCR margin"
    )
elif decision_source == "ocr_confirmed":
    result_detail = (
        f"OCR confirmed the model result &middot; {confidence_percent:.1f}% model confidence"
    )
elif decision_source == "ocr_consensus":
    result_detail = (
        f"Model and OCR agree &middot; {ocr_result.margin * 100:.1f}% OCR margin"
    )
elif decision_source == "model_ocr_inconclusive":
    result_detail = (
        f"OCR was inconclusive &middot; using {confidence_percent:.1f}% model confidence"
    )
elif decision_source == "uncertain_ocr_inconclusive":
    result_detail = (
        f"OCR evidence was too weak to verify {result.display_label} "
        f"&middot; no correction applied"
    )
elif decision_source == "model_ocr_error":
    result_detail = f"OCR unavailable &middot; using {confidence_percent:.1f}% model confidence"
else:
    result_detail = f"Model only &middot; {confidence_percent:.1f}% confidence"

st.markdown(
    f"""
    <div class="result-band">
        <div class="result-eyebrow">Detected orientation</div>
        <div class="result-title">{final_display_label}</div>
        <div class="result-detail">{result_detail}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if ocr_error:
    st.warning(
        "OCR verification could not run, so this result uses the classifier only.",
        icon=":material/warning:",
    )
if detection_error:
    st.warning(
        "Receipt object detection could not run, so the contour detector was used.",
        icon=":material/warning:",
    )
elif (
    object_detection_enabled
    and detection_result is not None
    and not detection_result.detected
):
    st.info(
        "No receipt object passed the detector threshold; contour fallback was used.",
        icon=":material/info:",
    )
elif (
    decision_source in {"model_only", "model_ocr_inconclusive"}
    and (result.confidence < 0.65 or result.margin < 0.20)
):
    st.warning(
        "This prediction is uncertain. Check the receipt crop and decision evidence.",
        icon=":material/warning:",
    )

metric_columns = st.columns(4)
metric_columns[0].metric("Final orientation", final_display_label)
metric_columns[1].metric("Decision", DECISION_SOURCE_NAMES[decision_source])
metric_columns[2].metric("Model confidence", f"{confidence_percent:.1f}%")
processing_ms = (
    result.inference_ms
    + (detection_result.detector_ms if detection_result else 0.0)
    + (ocr_result.ocr_ms if ocr_result else 0.0)
)
metric_columns[3].metric("Processing time", f"{processing_ms / 1000:.2f} s")

inspection_tab, evidence_tab, correction_tab = st.tabs(
    ["Image inspection", "Decision evidence", "Correction"]
)

with inspection_tab:
    inspection_images = [
        ("Uploaded image", uploaded_image),
    ]
    if detection_result is not None:
        inspection_images.append(("Object detection", detection_result.overlay))
    inspection_images.extend(
        [
            ("Detected receipt", result.extraction.image),
            ("Model input", result.model_input),
        ]
    )
    image_columns = st.columns(len(inspection_images))
    for column, (caption, image) in zip(image_columns, inspection_images):
        with column:
            st.caption(caption)
            st.image(image, width="stretch")

    with st.expander("Detection details"):
        details = st.columns(4)
        if detection_result is not None:
            detector_status = "Detected" if detection_result.detected else "Fallback"
            details[0].metric("Object detector", detector_status)
            details[1].metric(
                "Object confidence",
                f"{detection_result.detector_confidence * 100:.1f}%",
            )
            details[2].metric(
                "Receipt area", f"{result.extraction.area_ratio * 100:.1f}%"
            )
            details[3].metric(
                "Detector time", f"{detection_result.detector_ms:.1f} ms"
            )
        else:
            details[0].metric("Object detector", "Off")
            details[1].metric(
                "Crop confidence", f"{result.extraction.confidence * 100:.1f}%"
            )
            details[2].metric(
                "Receipt area", f"{result.extraction.area_ratio * 100:.1f}%"
            )
            details[3].metric("Input size", f"{classifier.image_size} x {classifier.image_size}")
        st.caption(f"Crop method: {crop_method_name(result.extraction.method)}")

with evidence_tab:
    st.subheader("Model probabilities")
    for class_name in classifier.class_names:
        st.markdown(
            probability_row(class_name, result.probabilities[class_name]),
            unsafe_allow_html=True,
        )
    st.caption(f"Top-two probability margin: {result.margin * 100:.1f} percentage points")

    st.divider()
    st.subheader("OCR direction check")
    if ocr_result is not None:
        ocr_metrics = st.columns(4)
        ocr_metrics[0].metric("OCR result", DISPLAY_NAMES[ocr_result.ocr_label])
        ocr_metrics[1].metric("Relative margin", f"{ocr_result.margin * 100:.1f}%")
        ocr_metrics[2].metric("Keyword hits", str(ocr_result.keyword_hits))
        ocr_metrics[3].metric("OCR time", f"{ocr_result.ocr_ms / 1000:.2f} s")

        score_columns = st.columns(len(ocr_result.rotation_candidates))
        for column, rotation in zip(score_columns, ocr_result.rotation_candidates):
            label = DISPLAY_NAMES[ROTATION_TO_LABEL[rotation]]
            column.metric(label, f"{ocr_result.scores[rotation]:.1f}")

        if decision_source == "ocr_override":
            st.success(
                f"OCR overrode {result.display_label} with {final_display_label}.",
                icon=":material/published_with_changes:",
            )
        elif decision_source == "ocr_confirmed":
            st.success("OCR confirmed the model prediction.", icon=":material/check_circle:")
        elif decision_source == "ocr_consensus":
            st.success(
                "Model and OCR agreed with sufficient combined evidence.",
                icon=":material/check_circle:",
            )
        elif decision_source == "uncertain_ocr_inconclusive":
            st.warning(
                "OCR could not reliably distinguish the two opposite directions. The app will not force a result.",
                icon=":material/help:",
            )
        else:
            st.info(
                "The OCR score margin was below the validated threshold, so the model result was retained.",
                icon=":material/info:",
            )

        if ocr_result.text_preview:
            with st.expander("Recognized text preview"):
                st.write(ocr_result.text_preview)
        enhanced_scores = {
            rotation: scores
            for rotation, scores in ocr_result.variant_scores.items()
            if len(scores) > 1
        }
        if enhanced_scores:
            with st.expander("OCR preprocessing evidence"):
                for rotation in ocr_result.rotation_candidates:
                    scores = enhanced_scores.get(rotation)
                    if not scores:
                        continue
                    label = DISPLAY_NAMES[ROTATION_TO_LABEL[rotation]]
                    st.caption(label)
                    st.write(
                        " · ".join(
                            f"{name.replace('_', ' ').title()}: {score:.1f}"
                            for name, score in scores.items()
                        )
                    )
    elif ocr_error:
        st.warning("OCR evidence is unavailable for this image.", icon=":material/warning:")
        with st.expander("OCR error details"):
            st.code(ocr_error)
    else:
        st.info("OCR verification is disabled.", icon=":material/info:")

with correction_tab:
    correction_columns = st.columns([1.2, 1])
    with correction_columns[0]:
        st.caption(
            "Quarter-turn corrected receipt"
            if final_label is not None
            else "Detected receipt crop"
        )
        st.image(final_corrected_receipt, width="stretch")
    with correction_columns[1]:
        if final_label is None:
            st.subheader("No correction applied")
            st.warning(
                "Capture a clearer, closer photo or review the orientation manually.",
                icon=":material/help:",
            )
        else:
            st.subheader(CORRECTION_LABELS[final_label])
            st.download_button(
                "Download corrected receipt",
                data=image_as_png(final_corrected_receipt),
                file_name="corrected_receipt.png",
                mime="image/png",
                icon=":material/download:",
                width="stretch",
            )
