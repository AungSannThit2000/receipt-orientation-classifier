from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from PIL import Image

from ocr_orientation import EasyOCROrientationDetector, OrientationDecision


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OCR_MODEL_DIR = PROJECT_ROOT / "models" / "easyocr"
DEFAULT_HYBRID_CONFIG = PROJECT_ROOT / "config" / "hybrid_ocr_config.json"

VERTICAL_CLASSES = {"upright", "upside_down"}
HORIZONTAL_CLASSES = {"tilted_right", "tilted_left"}
PAIR_ROTATIONS = {
    "vertical": (0, 180),
    "horizontal": (90, 270),
}
ROTATION_TO_LABEL = {
    0: "upright",
    90: "tilted_right",
    180: "upside_down",
    270: "tilted_left",
}


@dataclass(frozen=True)
class HybridOCRConfig:
    minimum_score: float
    minimum_margin: float
    ocr_image_size: int = 768
    languages: tuple[str, ...] = ("th", "en")
    uncertain_on_inconclusive: bool = True
    fast_pairwise: bool = True
    max_text_regions: int = 24
    recognition_batch_size: int = 32
    confirmation_minimum_score: float = 4.0
    confirmation_minimum_margin: float = 0.50
    confirmation_model_confidence: float = 0.55
    confirmation_model_margin: float = 0.15
    strong_consensus_minimum_score: float = 2.0
    strong_consensus_minimum_margin: float = 0.50
    strong_consensus_model_confidence: float = 0.95
    strong_consensus_model_margin: float = 0.80

    @classmethod
    def from_path(cls, path: Path = DEFAULT_HYBRID_CONFIG) -> "HybridOCRConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            minimum_score=float(payload["minimum_score"]),
            minimum_margin=float(payload["minimum_margin"]),
            ocr_image_size=int(payload.get("ocr_image_size", 768)),
            languages=tuple(payload.get("languages", ["th", "en"])),
            uncertain_on_inconclusive=bool(
                payload.get("uncertain_on_inconclusive", True)
            ),
            fast_pairwise=bool(payload.get("fast_pairwise", True)),
            max_text_regions=int(payload.get("max_text_regions", 24)),
            recognition_batch_size=int(payload.get("recognition_batch_size", 32)),
            confirmation_minimum_score=float(
                payload.get("confirmation_minimum_score", 4.0)
            ),
            confirmation_minimum_margin=float(
                payload.get("confirmation_minimum_margin", 0.50)
            ),
            confirmation_model_confidence=float(
                payload.get("confirmation_model_confidence", 0.55)
            ),
            confirmation_model_margin=float(
                payload.get("confirmation_model_margin", 0.15)
            ),
            strong_consensus_minimum_score=float(
                payload.get("strong_consensus_minimum_score", 2.0)
            ),
            strong_consensus_minimum_margin=float(
                payload.get("strong_consensus_minimum_margin", 0.50)
            ),
            strong_consensus_model_confidence=float(
                payload.get("strong_consensus_model_confidence", 0.95)
            ),
            strong_consensus_model_margin=float(
                payload.get("strong_consensus_model_margin", 0.80)
            ),
        )


@dataclass(frozen=True)
class OCRVerificationResult:
    final_label: str | None
    model_label: str
    ocr_label: str
    decision_source: str
    reliable: bool
    best_score: float
    second_score: float
    margin: float
    keyword_hits: int
    text_preview: str
    scores: dict[int, float]
    variant_scores: dict[int, dict[str, float]]
    rotation_candidates: tuple[int, ...]
    ocr_ms: float

    @property
    def overridden(self) -> bool:
        return self.decision_source == "ocr_override"


def candidate_rotations(model_label: str) -> tuple[int, ...]:
    if model_label in VERTICAL_CLASSES:
        return PAIR_ROTATIONS["vertical"]
    if model_label in HORIZONTAL_CLASSES:
        return PAIR_ROTATIONS["horizontal"]
    raise ValueError(f"Unsupported model label: {model_label}")


def resolve_decision(
    model_label: str,
    decision: OrientationDecision,
    config: HybridOCRConfig,
    ocr_ms: float = 0.0,
    model_confidence: float = 0.0,
    model_margin: float = 0.0,
) -> OCRVerificationResult:
    ocr_label = ROTATION_TO_LABEL[decision.best_rotation_ccw]
    strict_reliable = (
        decision.best_score >= config.minimum_score
        and decision.confidence_margin >= config.minimum_margin
    )
    consensus_reliable = (
        not strict_reliable
        and ocr_label == model_label
        and decision.best_score >= config.confirmation_minimum_score
        and decision.confidence_margin >= config.confirmation_minimum_margin
        and model_confidence >= config.confirmation_model_confidence
        and model_margin >= config.confirmation_model_margin
    )
    strong_consensus_reliable = (
        not strict_reliable
        and not consensus_reliable
        and ocr_label == model_label
        and decision.best_score >= config.strong_consensus_minimum_score
        and decision.confidence_margin >= config.strong_consensus_minimum_margin
        and model_confidence >= config.strong_consensus_model_confidence
        and model_margin >= config.strong_consensus_model_margin
    )
    reliable = strict_reliable or consensus_reliable or strong_consensus_reliable
    if consensus_reliable:
        final_label = model_label
        source = "ocr_consensus"
    elif strong_consensus_reliable:
        final_label = model_label
        source = "high_confidence_consensus"
    elif not reliable:
        final_label = None if config.uncertain_on_inconclusive else model_label
        source = (
            "uncertain_ocr_inconclusive"
            if config.uncertain_on_inconclusive
            else "model_ocr_inconclusive"
        )
    elif ocr_label == model_label:
        final_label = model_label
        source = "ocr_confirmed"
    else:
        final_label = ocr_label
        source = "ocr_override"

    return OCRVerificationResult(
        final_label=final_label,
        model_label=model_label,
        ocr_label=ocr_label,
        decision_source=source,
        reliable=reliable,
        best_score=decision.best_score,
        second_score=decision.second_score,
        margin=decision.confidence_margin,
        keyword_hits=decision.keyword_hits[decision.best_rotation_ccw],
        text_preview=decision.text_previews[decision.best_rotation_ccw],
        scores=decision.scores,
        variant_scores=decision.variant_scores,
        rotation_candidates=candidate_rotations(model_label),
        ocr_ms=ocr_ms,
    )


class HybridOrientationVerifier:
    def __init__(
        self,
        config: HybridOCRConfig,
        model_storage_directory: Path = DEFAULT_OCR_MODEL_DIR,
    ) -> None:
        self.config = config
        self.detector = EasyOCROrientationDetector(
            model_storage_directory=model_storage_directory,
            image_size=config.ocr_image_size,
            minimum_score=config.minimum_score,
            minimum_margin=config.minimum_margin,
            # Community Cloud starts without the local EasyOCR cache. EasyOCR
            # reuses existing files locally and downloads only missing weights.
            download_enabled=True,
            languages=config.languages,
            enhance_when_ambiguous=True,
            fast_pairwise=config.fast_pairwise,
            max_text_regions=config.max_text_regions,
            recognition_batch_size=config.recognition_batch_size,
        )

    def verify(
        self,
        image: Image.Image,
        model_label: str,
        model_confidence: float = 0.0,
        model_margin: float = 0.0,
    ) -> OCRVerificationResult:
        rotations = candidate_rotations(model_label)
        started = perf_counter()
        decision = self.detector.compare_rotations(image, rotations)
        ocr_ms = (perf_counter() - started) * 1000.0
        return resolve_decision(
            model_label,
            decision,
            self.config,
            ocr_ms=ocr_ms,
            model_confidence=model_confidence,
            model_margin=model_margin,
        )
