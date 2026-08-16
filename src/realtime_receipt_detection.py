from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from receipt_preprocessing import (
    MODEL_CANVAS_COLOR,
    ReceiptExtraction,
    draw_extraction_overlay,
    extract_receipt,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DETECTOR_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "detection"
    / "receipt_yolov8s_worldv2.pt"
)


@dataclass(frozen=True)
class ReceiptDetectionResult:
    extraction: ReceiptExtraction
    overlay: Image.Image
    detected: bool
    detector_label: str | None
    detector_confidence: float
    detector_bbox: tuple[int, int, int, int] | None
    detector_ms: float
    refinement_method: str | None


class YOLOWorldReceiptDetector:
    def __init__(
        self,
        checkpoint_path: Path = DEFAULT_DETECTOR_CHECKPOINT,
        confidence_threshold: float = 0.08,
        image_size: int = 640,
        padding_ratio: float = 0.02,
        device: str = "cpu",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Receipt detector checkpoint not found: {self.checkpoint_path}"
            )

        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise ImportError(
                "Receipt object detection requires the ultralytics package."
            ) from error

        self.model = YOLO(str(self.checkpoint_path), task="detect")
        self.confidence_threshold = float(confidence_threshold)
        self.image_size = int(image_size)
        self.padding_ratio = float(padding_ratio)
        self.device = device

    def detect(self, image: Image.Image) -> ReceiptDetectionResult:
        normalized = ImageOps.exif_transpose(image).convert("RGB")
        rgb = np.asarray(normalized)
        started = perf_counter()
        result = self.model.predict(
            source=rgb,
            conf=self.confidence_threshold,
            imgsz=self.image_size,
            device=self.device,
            max_det=10,
            verbose=False,
        )[0]
        detector_ms = (perf_counter() - started) * 1000.0

        detection = self._select_detection(result, normalized.size)
        if detection is None:
            extraction = extract_receipt(normalized, fill_color=MODEL_CANVAS_COLOR)
            return ReceiptDetectionResult(
                extraction=extraction,
                overlay=draw_extraction_overlay(normalized, extraction),
                detected=False,
                detector_label=None,
                detector_confidence=0.0,
                detector_bbox=None,
                detector_ms=detector_ms,
                refinement_method=None,
            )

        bbox, confidence, label = detection
        padded_bbox = self._pad_bbox(bbox, normalized.size)
        proposal = normalized.crop(padded_bbox)
        local = extract_receipt(proposal, fill_color=MODEL_CANVAS_COLOR)
        extraction = self._map_extraction(
            local,
            proposal,
            padded_bbox,
            normalized.size,
        )
        overlay = self._draw_overlay(
            normalized,
            bbox=bbox,
            extraction=extraction,
            label=label,
            confidence=confidence,
        )
        return ReceiptDetectionResult(
            extraction=extraction,
            overlay=overlay,
            detected=True,
            detector_label=label,
            detector_confidence=confidence,
            detector_bbox=bbox,
            detector_ms=detector_ms,
            refinement_method=local.method,
        )

    def _select_detection(
        self,
        result: object,
        image_size: tuple[int, int],
    ) -> tuple[tuple[int, int, int, int], float, str] | None:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return None

        width, height = image_size
        image_area = float(width * height)
        names = getattr(result, "names", self.model.names)
        candidates: list[
            tuple[float, tuple[int, int, int, int], float, str]
        ] = []
        for raw_box, raw_confidence, raw_class in zip(
            boxes.xyxy.cpu().tolist(),
            boxes.conf.cpu().tolist(),
            boxes.cls.cpu().tolist(),
        ):
            left, top, right, bottom = self._clamp_bbox(raw_box, image_size)
            box_width = right - left
            box_height = bottom - top
            if box_width < 2 or box_height < 2:
                continue

            area_ratio = box_width * box_height / image_area
            if area_ratio < 0.015 or area_ratio > 0.98:
                continue

            center_x = (left + right) / 2.0
            center_y = (top + bottom) / 2.0
            center_distance = np.hypot(
                center_x - width / 2.0,
                center_y - height / 2.0,
            )
            center_quality = float(
                np.clip(
                    1.0 - center_distance / max(np.hypot(width, height) * 0.7, 1.0),
                    0.0,
                    1.0,
                )
            )
            confidence = float(raw_confidence)
            score = confidence * (0.85 + 0.15 * center_quality)
            label = str(names[int(raw_class)])
            candidates.append((score, (left, top, right, bottom), confidence, label))

        if not candidates:
            return None
        _, bbox, confidence, label = max(candidates, key=lambda item: item[0])
        return bbox, confidence, label

    @staticmethod
    def _clamp_bbox(
        raw_box: list[float],
        image_size: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        width, height = image_size
        left = int(np.clip(round(raw_box[0]), 0, width - 1))
        top = int(np.clip(round(raw_box[1]), 0, height - 1))
        right = int(np.clip(round(raw_box[2]), left + 1, width))
        bottom = int(np.clip(round(raw_box[3]), top + 1, height))
        return left, top, right, bottom

    def _pad_bbox(
        self,
        bbox: tuple[int, int, int, int],
        image_size: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        width, height = image_size
        padding = max(2, round(max(width, height) * self.padding_ratio))
        left, top, right, bottom = bbox
        return (
            max(0, left - padding),
            max(0, top - padding),
            min(width, right + padding),
            min(height, bottom + padding),
        )

    @staticmethod
    def _map_extraction(
        local: ReceiptExtraction,
        proposal: Image.Image,
        proposal_bbox: tuple[int, int, int, int],
        original_size: tuple[int, int],
    ) -> ReceiptExtraction:
        offset_x, offset_y = proposal_bbox[:2]
        original_width, original_height = original_size

        use_refinement = (
            local.method != "full_frame_fallback"
            and 0.32 <= local.area_ratio <= 0.98
        )
        if use_refinement:
            polygon = tuple(
                (x + offset_x, y + offset_y) for x, y in local.polygon
            )
            left, top, right, bottom = local.bbox
            global_bbox = (
                left + offset_x,
                top + offset_y,
                right + offset_x,
                bottom + offset_y,
            )
            polygon_array = np.asarray(polygon, dtype=np.int32)
            area_ratio = float(
                cv2.contourArea(polygon_array) / (original_width * original_height)
            )
            return ReceiptExtraction(
                image=local.image,
                method=f"yolo_world+{local.method}",
                confidence=local.confidence,
                area_ratio=area_ratio,
                bbox=global_bbox,
                polygon=polygon,
            )

        left, top, right, bottom = proposal_bbox
        polygon = (
            (left, top),
            (right - 1, top),
            (right - 1, bottom - 1),
            (left, bottom - 1),
        )
        return ReceiptExtraction(
            image=proposal.copy(),
            method="yolo_world_box",
            confidence=0.0,
            area_ratio=((right - left) * (bottom - top))
            / (original_width * original_height),
            bbox=proposal_bbox,
            polygon=polygon,
        )

    @staticmethod
    def _draw_overlay(
        image: Image.Image,
        bbox: tuple[int, int, int, int],
        extraction: ReceiptExtraction,
        label: str,
        confidence: float,
    ) -> Image.Image:
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        line_width = max(3, round(max(image.size) / 320))
        draw.rectangle(bbox, outline=(15, 118, 110), width=line_width)
        polygon = list(extraction.polygon)
        polygon.append(polygon[0])
        draw.line(polygon, fill=(180, 83, 9), width=line_width)
        draw.rectangle((8, 8, min(image.width - 8, 520), 42), fill=(255, 255, 255))
        draw.text(
            (14, 14),
            f"YOLO-World {label}  confidence={confidence:.2f}",
            fill=(20, 20, 20),
        )
        return overlay
