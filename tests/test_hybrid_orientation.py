from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hybrid_orientation import (  # noqa: E402
    HybridOCRConfig,
    candidate_rotations,
    resolve_decision,
)
from ocr_orientation import OrientationDecision  # noqa: E402


def ocr_decision(
    best_rotation: int,
    best_score: float = 80.0,
    second_score: float = 20.0,
) -> OrientationDecision:
    scores = {rotation: 0.0 for rotation in (0, 90, 180, 270)}
    scores[best_rotation] = best_score
    other_rotation = (best_rotation + 180) % 360
    scores[other_rotation] = second_score
    margin = (best_score - second_score) / max(best_score, 1.0)
    return OrientationDecision(
        best_rotation_ccw=best_rotation,
        deskew_rotation_ccw=0.0,
        best_score=best_score,
        second_score=second_score,
        confidence_margin=margin,
        review_required=False,
        scores=scores,
        keyword_hits={rotation: 0 for rotation in scores},
        text_previews={rotation: "" for rotation in scores},
    )


class HybridOrientationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = HybridOCRConfig(minimum_score=0.0, minimum_margin=0.22)

    def test_candidate_pair_preserves_model_axis(self) -> None:
        self.assertEqual(candidate_rotations("upright"), (0, 180))
        self.assertEqual(candidate_rotations("upside_down"), (0, 180))
        self.assertEqual(candidate_rotations("tilted_right"), (90, 270))
        self.assertEqual(candidate_rotations("tilted_left"), (90, 270))

    def test_ocr_overrides_opposite_horizontal_direction(self) -> None:
        result = resolve_decision(
            "tilted_right",
            ocr_decision(best_rotation=270),
            self.config,
        )
        self.assertEqual(result.final_label, "tilted_left")
        self.assertEqual(result.decision_source, "ocr_override")
        self.assertTrue(result.overridden)

    def test_ocr_overrides_opposite_vertical_direction(self) -> None:
        result = resolve_decision(
            "upright",
            ocr_decision(best_rotation=180),
            self.config,
        )
        self.assertEqual(result.final_label, "upside_down")
        self.assertEqual(result.decision_source, "ocr_override")

    def test_ocr_confirms_model_when_directions_agree(self) -> None:
        result = resolve_decision(
            "upright",
            ocr_decision(best_rotation=0),
            self.config,
        )
        self.assertEqual(result.final_label, "upright")
        self.assertEqual(result.decision_source, "ocr_confirmed")

    def test_low_margin_returns_uncertain(self) -> None:
        result = resolve_decision(
            "tilted_right",
            ocr_decision(best_rotation=270, best_score=50.0, second_score=42.0),
            self.config,
        )
        self.assertIsNone(result.final_label)
        self.assertEqual(result.decision_source, "uncertain_ocr_inconclusive")
        self.assertFalse(result.reliable)

    def test_inconclusive_can_fall_back_to_model_when_configured(self) -> None:
        config = HybridOCRConfig(
            minimum_score=0.0,
            minimum_margin=0.22,
            uncertain_on_inconclusive=False,
        )
        result = resolve_decision(
            "tilted_right",
            ocr_decision(best_rotation=270, best_score=50.0, second_score=42.0),
            config,
        )
        self.assertEqual(result.final_label, "tilted_right")
        self.assertEqual(result.decision_source, "model_ocr_inconclusive")

    def test_model_and_ocr_consensus_uses_confirmation_thresholds(self) -> None:
        config = HybridOCRConfig(minimum_score=5.0, minimum_margin=0.35)
        result = resolve_decision(
            "upside_down",
            ocr_decision(best_rotation=180, best_score=4.64, second_score=1.05),
            config,
            model_confidence=0.59,
            model_margin=0.22,
        )
        self.assertEqual(result.final_label, "upside_down")
        self.assertEqual(result.decision_source, "ocr_consensus")
        self.assertTrue(result.reliable)

    def test_weak_model_does_not_pass_consensus_threshold(self) -> None:
        config = HybridOCRConfig(minimum_score=5.0, minimum_margin=0.35)
        result = resolve_decision(
            "upside_down",
            ocr_decision(best_rotation=180, best_score=4.64, second_score=1.05),
            config,
            model_confidence=0.51,
            model_margin=0.22,
        )
        self.assertIsNone(result.final_label)
        self.assertEqual(result.decision_source, "uncertain_ocr_inconclusive")


if __name__ == "__main__":
    unittest.main()
