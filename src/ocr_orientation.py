from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


RECEIPT_KEYWORDS = {
    "amount",
    "balance",
    "card",
    "cash",
    "change",
    "date",
    "invoice",
    "order",
    "payment",
    "price",
    "purchase",
    "receipt",
    "subtotal",
    "tax",
    "thank",
    "total",
    "visa",
}
THAI_RECEIPT_KEYWORDS = {
    "เงินสด",
    "ทอน",
    "ใบเสร็จ",
    "ภาษี",
    "รวม",
    "วันที่",
    "ยอดรวม",
}
OCR_VARIANTS = ("color", "sharpened", "threshold")


@dataclass(frozen=True)
class OrientationDecision:
    best_rotation_ccw: int
    deskew_rotation_ccw: float
    best_score: float
    second_score: float
    confidence_margin: float
    review_required: bool
    scores: dict[int, float]
    keyword_hits: dict[int, int]
    text_previews: dict[int, str]
    variant_scores: dict[int, dict[str, float]] = field(default_factory=dict)


class EasyOCROrientationDetector:
    def __init__(
        self,
        model_storage_directory: Path,
        image_size: int = 512,
        minimum_score: float = 18.0,
        minimum_margin: float = 0.22,
        download_enabled: bool = True,
        languages: tuple[str, ...] = ("th", "en"),
        enhance_when_ambiguous: bool = True,
        fast_pairwise: bool = True,
        max_text_regions: int = 24,
        recognition_batch_size: int = 32,
    ) -> None:
        import easyocr

        self.image_size = image_size
        self.minimum_score = minimum_score
        self.minimum_margin = minimum_margin
        self.languages = languages
        self.enhance_when_ambiguous = enhance_when_ambiguous
        self.fast_pairwise = fast_pairwise
        self.max_text_regions = max_text_regions
        self.recognition_batch_size = recognition_batch_size
        model_storage_directory.mkdir(parents=True, exist_ok=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.reader = easyocr.Reader(
                list(languages),
                gpu=False,
                model_storage_directory=str(model_storage_directory),
                download_enabled=download_enabled,
                verbose=False,
            )

    def compare_rotations(
        self,
        image: Image.Image,
        rotations: tuple[int, ...],
    ) -> OrientationDecision:
        normalized = tuple(dict.fromkeys(int(rotation) % 360 for rotation in rotations))
        if (
            self.fast_pairwise
            and len(normalized) == 2
            and (normalized[1] - normalized[0]) % 360 == 180
        ):
            return self._compare_opposite_rotations_fast(image, normalized)
        return self.compare_rotations_batch([(image, rotations)])[0]

    def _compare_opposite_rotations_fast(
        self,
        image: Image.Image,
        rotations: tuple[int, int],
    ) -> OrientationDecision:
        base_rotation, opposite_rotation = rotations
        prepared = self._prepare_variant(
            image.rotate(
                base_rotation,
                resample=Image.Resampling.BICUBIC,
                expand=True,
                fillcolor=(245, 245, 245),
            ),
            "color",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            horizontal_lists, free_lists = self.reader.detect(
                prepared,
                min_size=20,
                text_threshold=0.55,
                low_text=0.25,
                link_threshold=0.30,
                canvas_size=self.image_size + 128,
                reformat=True,
            )
        horizontal = list(horizontal_lists[0])
        free = list(free_lists[0])
        horizontal, free = self._largest_text_regions(horizontal, free)

        height, width = prepared.shape[:2]
        opposite_image = np.rot90(prepared, 2).copy()
        opposite_horizontal = [
            [width - x_max, width - x_min, height - y_max, height - y_min]
            for x_min, x_max, y_min, y_max in horizontal
        ]
        opposite_free = [
            [[width - x, height - y] for x, y in box] for box in free
        ]

        candidate_inputs = (
            (base_rotation, prepared, horizontal, free),
            (
                opposite_rotation,
                opposite_image,
                opposite_horizontal,
                opposite_free,
            ),
        )
        scored: dict[int, tuple[float, int, str, list[tuple[float, float]]]] = {}
        for rotation, candidate_image, candidate_horizontal, candidate_free in candidate_inputs:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = self.reader.recognize(
                    candidate_image,
                    candidate_horizontal,
                    candidate_free,
                    decoder="greedy",
                    batch_size=self.recognition_batch_size,
                    workers=0,
                    detail=1,
                    paragraph=False,
                    reformat=True,
                )
            scored[rotation] = self._score_results(results)

        scores = {rotation: 0.0 for rotation in (0, 90, 180, 270)}
        hits = {rotation: 0 for rotation in (0, 90, 180, 270)}
        previews = {rotation: "" for rotation in (0, 90, 180, 270)}
        angles = {rotation: [] for rotation in (0, 90, 180, 270)}
        variant_scores = {rotation: {} for rotation in (0, 90, 180, 270)}
        for rotation, (score, keyword_hits, preview, line_angles) in scored.items():
            scores[rotation] = score
            hits[rotation] = keyword_hits
            previews[rotation] = preview
            angles[rotation] = line_angles
            variant_scores[rotation] = {"color": score}

        ranked = sorted(
            ((rotation, scores[rotation]) for rotation in rotations),
            key=lambda item: item[1],
            reverse=True,
        )
        best_rotation, best_score = ranked[0]
        second_score = ranked[1][1]
        margin = (best_score - second_score) / max(best_score, 1.0)
        return OrientationDecision(
            best_rotation_ccw=best_rotation,
            deskew_rotation_ccw=self._weighted_median_angle(angles[best_rotation]),
            best_score=best_score,
            second_score=second_score,
            confidence_margin=margin,
            review_required=(
                best_score < self.minimum_score or margin < self.minimum_margin
            ),
            scores=scores,
            keyword_hits=hits,
            text_previews=previews,
            variant_scores=variant_scores,
        )

    def _largest_text_regions(
        self,
        horizontal: list,
        free: list,
    ) -> tuple[list, list]:
        ranked_regions: list[tuple[float, str, object]] = []
        for box in horizontal:
            x_min, x_max, y_min, y_max = box
            area = max(float(x_max - x_min), 1.0) * max(float(y_max - y_min), 1.0)
            ranked_regions.append((area, "horizontal", box))
        for box in free:
            area = abs(float(cv2.contourArea(np.asarray(box, dtype=np.float32))))
            ranked_regions.append((area, "free", box))
        selected = sorted(ranked_regions, key=lambda item: item[0], reverse=True)[
            : self.max_text_regions
        ]
        return (
            [box for _, kind, box in selected if kind == "horizontal"],
            [box for _, kind, box in selected if kind == "free"],
        )

    def compare_rotations_batch(
        self,
        requests: list[tuple[Image.Image, tuple[int, ...]]],
        batch_size: int = 16,
    ) -> list[OrientationDecision]:
        if not requests:
            return []

        normalized_requests: list[tuple[Image.Image, tuple[int, ...]]] = []
        prepared: list[np.ndarray] = []
        locations: list[tuple[int, int, str]] = []
        for request_index, (image, rotations) in enumerate(requests):
            normalized = tuple(dict.fromkeys(int(rotation) % 360 for rotation in rotations))
            if len(normalized) < 2 or any(rotation not in {0, 90, 180, 270} for rotation in normalized):
                raise ValueError(
                    "Rotation comparisons require at least two unique values from 0, 90, 180, and 270."
                )
            normalized_requests.append((image, normalized))
            for rotation in normalized:
                prepared.append(
                    self._prepare_variant(
                        image.rotate(
                            rotation,
                            resample=Image.Resampling.BICUBIC,
                            expand=True,
                            fillcolor=(245, 245, 245),
                        ),
                        "color",
                    )
                )
                locations.append((request_index, rotation, "color"))

        details_by_request: list[dict[int, dict[str, tuple]]] = [
            {rotation: {} for rotation in (0, 90, 180, 270)} for _ in requests
        ]
        self._store_results(
            locations,
            self._run_ocr(prepared, batch_size=batch_size),
            details_by_request,
        )

        enhanced_prepared: list[np.ndarray] = []
        enhanced_locations: list[tuple[int, int, str]] = []
        if self.enhance_when_ambiguous:
            for request_index, (image, rotations) in enumerate(normalized_requests):
                color_scores = [
                    details_by_request[request_index][rotation]["color"][0]
                    for rotation in rotations
                ]
                ranked_color = sorted(color_scores, reverse=True)
                color_margin = (ranked_color[0] - ranked_color[1]) / max(
                    ranked_color[0], 1.0
                )
                if (
                    ranked_color[0] >= self.minimum_score
                    and color_margin >= self.minimum_margin
                ):
                    continue
                for rotation in rotations:
                    rotated = image.rotate(
                        rotation,
                        resample=Image.Resampling.BICUBIC,
                        expand=True,
                        fillcolor=(245, 245, 245),
                    )
                    for variant in OCR_VARIANTS[1:]:
                        enhanced_prepared.append(self._prepare_variant(rotated, variant))
                        enhanced_locations.append((request_index, rotation, variant))

        if enhanced_prepared:
            self._store_results(
                enhanced_locations,
                self._run_ocr(enhanced_prepared, batch_size=batch_size),
                details_by_request,
            )

        decisions: list[OrientationDecision] = []
        for request_index, (_, rotations) in enumerate(normalized_requests):
            scores = {rotation: 0.0 for rotation in (0, 90, 180, 270)}
            hits = {rotation: 0 for rotation in (0, 90, 180, 270)}
            previews = {rotation: "" for rotation in (0, 90, 180, 270)}
            angles = {rotation: [] for rotation in (0, 90, 180, 270)}
            variant_scores: dict[int, dict[str, float]] = {
                rotation: {} for rotation in (0, 90, 180, 270)
            }
            for rotation in rotations:
                variant_details = details_by_request[request_index][rotation]
                variant_scores[rotation] = {
                    name: float(details[0]) for name, details in variant_details.items()
                }
                ranked_variants = sorted(
                    variant_details.items(),
                    key=lambda item: item[1][0],
                    reverse=True,
                )
                best_variant, best_details = ranked_variants[0]
                second_variant_score = (
                    ranked_variants[1][1][0] if len(ranked_variants) > 1 else 0.0
                )
                scores[rotation] = float(best_details[0] + 0.35 * second_variant_score)
                hits[rotation] = int(best_details[1])
                previews[rotation] = str(best_details[2])
                angles[rotation] = best_details[3]

            ranked = sorted(
                ((rotation, scores[rotation]) for rotation in rotations),
                key=lambda item: item[1],
                reverse=True,
            )
            best_rotation, best_score = ranked[0]
            second_score = ranked[1][1]
            margin = (best_score - second_score) / max(best_score, 1.0)
            decisions.append(
                OrientationDecision(
                    best_rotation_ccw=best_rotation,
                    deskew_rotation_ccw=self._weighted_median_angle(
                        angles[best_rotation]
                    ),
                    best_score=best_score,
                    second_score=second_score,
                    confidence_margin=margin,
                    review_required=(
                        best_score < self.minimum_score or margin < self.minimum_margin
                    ),
                    scores=scores,
                    keyword_hits=hits,
                    text_previews=previews,
                    variant_scores=variant_scores,
                )
            )
        return decisions

    def detect(self, image: Image.Image) -> OrientationDecision:
        primary = self.compare_rotations(image, (0, 180))
        if not primary.review_required:
            return primary
        return self.compare_rotations(image, (0, 90, 180, 270))

    def _run_ocr(
        self,
        prepared: list[np.ndarray],
        batch_size: int,
    ) -> list:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.reader.readtext_batched(
                prepared,
                batch_size=batch_size,
                workers=0,
                detail=1,
                decoder="greedy",
                paragraph=False,
                canvas_size=self.image_size + 128,
                text_threshold=0.55,
                low_text=0.25,
                link_threshold=0.30,
            )

    def _store_results(
        self,
        locations: list[tuple[int, int, str]],
        results: list,
        details_by_request: list[dict[int, dict[str, tuple]]],
    ) -> None:
        for (request_index, rotation, variant), rotation_results in zip(
            locations, results
        ):
            score, hits, preview, angles = self._score_results(rotation_results)
            details_by_request[request_index][rotation][variant] = (
                score,
                hits,
                preview,
                angles,
            )

    def _prepare_variant(self, image: Image.Image, variant: str) -> np.ndarray:
        padded = ImageOps.pad(
            image.convert("RGB"),
            (self.image_size, self.image_size),
            method=Image.Resampling.LANCZOS,
            color=(245, 245, 245),
            centering=(0.5, 0.5),
        )
        if variant == "color":
            return np.asarray(padded)

        grayscale = cv2.cvtColor(np.asarray(padded), cv2.COLOR_RGB2GRAY)
        if variant == "sharpened":
            blurred = cv2.GaussianBlur(grayscale, (0, 0), sigmaX=1.5)
            return cv2.addWeighted(grayscale, 2.8, blurred, -1.8, 0)
        if variant == "threshold":
            return cv2.adaptiveThreshold(
                grayscale,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                11,
            )
        raise ValueError(f"Unsupported OCR preprocessing variant: {variant}")

    @staticmethod
    def _score_results(results: list) -> tuple[float, int, str, list[tuple[float, float]]]:
        total_score = 0.0
        all_text: list[str] = []
        line_angles: list[tuple[float, float]] = []

        for box, text, confidence in results:
            normalized = " ".join(
                "".join(
                    character.lower()
                    if character.isalnum() or character == "$"
                    else " "
                    for character in text
                ).split()
            )
            alphanumeric_count = sum(character.isalnum() for character in normalized)
            if alphanumeric_count < 2:
                continue

            confidence = float(np.clip(confidence, 0.0, 1.0))
            length_weight = min(alphanumeric_count, 30) ** 0.85
            total_score += confidence * length_weight
            if any(character.isalpha() for character in normalized):
                total_score += 0.6 * confidence
            all_text.append(normalized)

            if len(box) >= 2:
                top_left = box[0]
                top_right = box[1]
                delta_x = float(top_right[0] - top_left[0])
                delta_y = float(top_right[1] - top_left[1])
                if abs(delta_x) > 2:
                    angle = math.degrees(math.atan2(delta_y, delta_x))
                    while angle > 45:
                        angle -= 90
                    while angle < -45:
                        angle += 90
                    if abs(angle) <= 18:
                        line_angles.append((angle, max(confidence * alphanumeric_count, 0.1)))

        combined_text = " ".join(all_text)
        hits = sum(
            1
            for keyword in RECEIPT_KEYWORDS
            if re.search(rf"\b{re.escape(keyword)}\b", combined_text)
        )
        hits += sum(1 for keyword in THAI_RECEIPT_KEYWORDS if keyword in combined_text)
        total_score += hits * 7.5
        preview = " | ".join(all_text[:6])[:300]
        return total_score, hits, preview, line_angles

    @staticmethod
    def _weighted_median_angle(values: list[tuple[float, float]]) -> float:
        if len(values) < 3:
            return 0.0
        ordered = sorted(values, key=lambda item: item[0])
        total_weight = sum(weight for _, weight in ordered)
        running_weight = 0.0
        for angle, weight in ordered:
            running_weight += weight
            if running_weight >= total_weight / 2.0:
                return float(np.clip(angle, -12.0, 12.0))
        return 0.0
