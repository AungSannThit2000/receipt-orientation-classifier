from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps


MODEL_CANVAS_COLOR = (190, 190, 190)


@dataclass(frozen=True)
class ReceiptExtraction:
    image: Image.Image
    method: str
    confidence: float
    area_ratio: float
    bbox: tuple[int, int, int, int]
    polygon: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _Candidate:
    method: str
    score: float
    area_ratio: float
    polygon: np.ndarray


def open_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def _odd_kernel_size(value: int) -> int:
    value = max(3, value)
    return value if value % 2 == 1 else value + 1


def _build_detection_masks(rgb: np.ndarray) -> list[tuple[str, np.ndarray]]:
    height, width = rgb.shape[:2]
    minimum_dimension = min(height, width)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu_bright = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    saturation_limit = int(np.clip(np.percentile(saturation, 70), 45, 105))
    value_limit = int(np.clip(np.percentile(value, 48), 105, 220))
    low_saturation_bright = np.where(
        (saturation <= saturation_limit) & (value >= value_limit), 255, 0
    ).astype(np.uint8)

    a_channel = lab[:, :, 1].astype(np.float32) - 128.0
    b_channel = lab[:, :, 2].astype(np.float32) - 128.0
    chroma = np.sqrt(a_channel * a_channel + b_channel * b_channel)
    chroma_limit = float(np.clip(np.percentile(chroma, 68), 12, 45))
    light_limit = int(np.clip(np.percentile(gray, 42), 90, 210))
    neutral_bright = np.where((chroma <= chroma_limit) & (gray >= light_limit), 255, 0).astype(np.uint8)

    median = float(np.median(blur))
    lower = int(max(20, 0.60 * median))
    upper = int(min(255, max(lower + 25, 1.35 * median)))
    edges = cv2.Canny(blur, lower, upper)

    close_size = _odd_kernel_size(round(minimum_dimension * 0.018))
    open_size = _odd_kernel_size(round(minimum_dimension * 0.006))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))

    masks: list[tuple[str, np.ndarray]] = []
    for method, mask in (
        ("otsu", otsu_bright),
        ("low_saturation", low_saturation_bright),
        ("neutral_bright", neutral_bright),
    ):
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel, iterations=1)
        masks.append((method, cleaned))

    edge_mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    edge_mask = cv2.dilate(edge_mask, open_kernel, iterations=1)
    masks.append(("edges", edge_mask))
    return masks


def _score_contour(
    contour: np.ndarray,
    method: str,
    gray: np.ndarray,
) -> _Candidate | None:
    height, width = gray.shape
    image_area = float(height * width)
    contour_area = float(cv2.contourArea(contour))
    if contour_area < image_area * 0.035:
        return None

    rect = cv2.minAreaRect(contour)
    rect_width, rect_height = rect[1]
    if rect_width < 2 or rect_height < 2:
        return None

    rect_area = float(rect_width * rect_height)
    area_ratio = rect_area / image_area
    if area_ratio < 0.07 or area_ratio > 1.08:
        return None

    polygon = np.rint(cv2.boxPoints(rect)).astype(np.int32)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)

    hull = cv2.convexHull(contour)
    hull_area = max(float(cv2.contourArea(hull)), 1.0)
    rectangularity = float(np.clip(contour_area / max(rect_area, 1.0), 0.0, 1.0))
    solidity = float(np.clip(contour_area / hull_area, 0.0, 1.0))

    center_x, center_y = rect[0]
    center_distance = np.hypot(center_x - width / 2.0, center_y - height / 2.0)
    center_quality = float(np.clip(1.0 - center_distance / (0.62 * np.hypot(width, height)), 0.0, 1.0))

    long_side = max(rect_width, rect_height)
    short_side = max(min(rect_width, rect_height), 1.0)
    aspect_ratio = long_side / short_side
    if 1.15 <= aspect_ratio <= 5.5:
        aspect_quality = 1.0
    elif aspect_ratio <= 7.0:
        aspect_quality = 0.65
    else:
        aspect_quality = 0.30

    candidate_mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.fillConvexPoly(candidate_mask, polygon, 255)
    dilation_size = _odd_kernel_size(round(min(height, width) * 0.025))
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
    ring = cv2.subtract(cv2.dilate(candidate_mask, dilation_kernel), candidate_mask)
    inside_values = gray[candidate_mask > 0]
    outside_values = gray[ring > 0]
    inside_mean = float(inside_values.mean()) if inside_values.size else 0.0
    outside_mean = float(outside_values.mean()) if outside_values.size else float(gray.mean())
    contrast_quality = float(np.clip((inside_mean - outside_mean + 18.0) / 75.0, 0.0, 1.0))

    area_quality = float(np.clip(area_ratio / 0.48, 0.0, 1.0))
    score = (
        0.30 * area_quality
        + 0.21 * rectangularity
        + 0.12 * solidity
        + 0.14 * center_quality
        + 0.18 * contrast_quality
        + 0.05 * aspect_quality
    )

    border_margin = max(2, round(min(height, width) * 0.008))
    border_points = sum(
        x <= border_margin
        or y <= border_margin
        or x >= width - 1 - border_margin
        or y >= height - 1 - border_margin
        for x, y in polygon
    )
    if border_points >= 3 and area_ratio > 0.82:
        score -= 0.10

    return _Candidate(method=method, score=float(np.clip(score, 0.0, 1.0)), area_ratio=area_ratio, polygon=polygon)


def extract_receipt(
    image: Image.Image,
    fill_color: tuple[int, int, int] = MODEL_CANVAS_COLOR,
    max_detection_dimension: int = 1200,
) -> ReceiptExtraction:
    rgb_image = image.convert("RGB")
    original = np.asarray(rgb_image)
    original_height, original_width = original.shape[:2]

    scale = min(1.0, max_detection_dimension / max(original_height, original_width))
    if scale < 1.0:
        detection_rgb = cv2.resize(
            original,
            (round(original_width * scale), round(original_height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        detection_rgb = original

    gray = cv2.cvtColor(detection_rgb, cv2.COLOR_RGB2GRAY)
    candidates: list[_Candidate] = []
    for method, mask in _build_detection_masks(detection_rgb):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
            candidate = _score_contour(contour, method, gray)
            if candidate is not None:
                candidates.append(candidate)

    if not candidates:
        return _full_frame_extraction(rgb_image)

    overall_best = max(candidates, key=lambda candidate: candidate.score)
    if overall_best.area_ratio > 0.96:
        # A full-frame threshold can hide a useful paper contour, but small
        # alternatives are often only text blocks. Require broad frame coverage.
        alternatives = [
            candidate for candidate in candidates if 0.55 <= candidate.area_ratio <= 0.96
        ]
        if not alternatives:
            return _full_frame_extraction(rgb_image, confidence=overall_best.score)
        best = max(alternatives, key=lambda candidate: candidate.score)
        if best.score < 0.42:
            return _full_frame_extraction(rgb_image, confidence=overall_best.score)
    else:
        best = overall_best
        if best.score < 0.46:
            return _full_frame_extraction(rgb_image, confidence=best.score)

    polygon = np.rint(best.polygon.astype(np.float64) / scale).astype(np.int32)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, original_width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, original_height - 1)

    mask = np.zeros((original_height, original_width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon, 255)
    composite = np.full_like(original, fill_color, dtype=np.uint8)
    composite[mask > 0] = original[mask > 0]

    x, y, width, height = cv2.boundingRect(polygon)
    padding = max(2, round(max(original_height, original_width) * 0.012))
    left = max(0, x - padding)
    top = max(0, y - padding)
    right = min(original_width, x + width + padding)
    bottom = min(original_height, y + height + padding)
    cropped = Image.fromarray(composite[top:bottom, left:right], mode="RGB")

    polygon_points = tuple((int(px), int(py)) for px, py in polygon)
    return ReceiptExtraction(
        image=cropped,
        method=best.method,
        confidence=best.score,
        area_ratio=float(cv2.contourArea(polygon) / (original_height * original_width)),
        bbox=(left, top, right, bottom),
        polygon=polygon_points,
    )


def _full_frame_extraction(image: Image.Image, confidence: float = 0.0) -> ReceiptExtraction:
    width, height = image.size
    return ReceiptExtraction(
        image=image.copy(),
        method="full_frame_fallback",
        confidence=float(confidence),
        area_ratio=1.0,
        bbox=(0, 0, width, height),
        polygon=((0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)),
    )


def rotate_with_fill(
    image: Image.Image,
    counter_clockwise_degrees: float,
    fill_color: tuple[int, int, int] = MODEL_CANVAS_COLOR,
) -> Image.Image:
    return image.rotate(
        counter_clockwise_degrees,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=fill_color,
    )


def model_canvas(
    image: Image.Image,
    size: int = 224,
    margin_ratio: float = 0.055,
    fill_color: tuple[int, int, int] = MODEL_CANVAS_COLOR,
) -> Image.Image:
    usable_size = max(1, round(size * (1.0 - 2.0 * margin_ratio)))
    fitted = ImageOps.contain(image.convert("RGB"), (usable_size, usable_size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), fill_color)
    offset = ((size - fitted.width) // 2, (size - fitted.height) // 2)
    canvas.paste(fitted, offset)
    return canvas


def orientation_sample(
    canonical_receipt: Image.Image,
    clockwise_degrees: float,
    size: int = 224,
    fill_color: tuple[int, int, int] = MODEL_CANVAS_COLOR,
) -> Image.Image:
    rotated = rotate_with_fill(canonical_receipt, -clockwise_degrees, fill_color)
    return model_canvas(rotated, size=size, fill_color=fill_color)


def draw_extraction_overlay(image: Image.Image, extraction: ReceiptExtraction) -> Image.Image:
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    points = list(extraction.polygon)
    points.append(points[0])
    line_width = max(2, round(max(image.size) / 300))
    draw.line(points, fill=(220, 35, 35), width=line_width)
    draw.rectangle(
        (8, 8, min(image.width - 8, 420), 38),
        fill=(255, 255, 255),
    )
    draw.text(
        (14, 14),
        f"{extraction.method}  confidence={extraction.confidence:.2f}",
        fill=(20, 20, 20),
    )
    return overlay
