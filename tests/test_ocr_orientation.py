from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ocr_orientation import EasyOCROrientationDetector  # noqa: E402


def scored(score: float, preview: str = "text") -> tuple:
    return score, 0, preview, []


class EasyOCROrientationDetectorTests(unittest.TestCase):
    def detector_without_reader(self, batches: list[list[tuple]]) -> object:
        detector = object.__new__(EasyOCROrientationDetector)
        detector.image_size = 64
        detector.minimum_score = 18.0
        detector.minimum_margin = 0.22
        detector.enhance_when_ambiguous = True
        detector.fast_pairwise = False
        detector._prepare_variant = lambda image, variant: np.zeros((64, 64), dtype=np.uint8)
        detector._score_results = lambda result: result
        detector._run_ocr = lambda prepared, batch_size: batches.pop(0)
        return detector

    def test_region_limit_keeps_largest_text_boxes(self) -> None:
        detector = object.__new__(EasyOCROrientationDetector)
        detector.max_text_regions = 2
        horizontal, free = detector._largest_text_regions(
            [[0, 5, 0, 5], [0, 20, 0, 10]],
            [[[0, 0], [12, 0], [12, 8], [0, 8]]],
        )
        self.assertEqual(horizontal, [[0, 20, 0, 10]])
        self.assertEqual(len(free), 1)

    def test_ambiguous_color_pass_runs_enhanced_variants(self) -> None:
        detector = self.detector_without_reader(
            [
                [scored(5.0, "wrong"), scored(15.0, "correct")],
                [scored(4.0), scored(3.0), scored(12.0), scored(11.0)],
            ]
        )
        decision = detector.compare_rotations(Image.new("RGB", (80, 160)), (0, 180))
        self.assertEqual(decision.best_rotation_ccw, 180)
        self.assertAlmostEqual(decision.best_score, 19.2)
        self.assertFalse(decision.review_required)
        self.assertEqual(
            set(decision.variant_scores[180]),
            {"color", "sharpened", "threshold"},
        )

    def test_strong_color_pass_skips_enhanced_variants(self) -> None:
        batches = [[scored(4.0), scored(30.0)]]
        detector = self.detector_without_reader(batches)
        decision = detector.compare_rotations(Image.new("RGB", (80, 160)), (0, 180))
        self.assertEqual(decision.best_rotation_ccw, 180)
        self.assertEqual(decision.variant_scores[180], {"color": 30.0})
        self.assertEqual(batches, [])

    def test_thai_characters_contribute_to_direction_score(self) -> None:
        box = [[0, 0], [40, 0], [40, 10], [0, 10]]
        score, hits, preview, _ = EasyOCROrientationDetector._score_results(
            [(box, "ยอดรวม เงินสด", 0.9)]
        )
        self.assertGreater(score, 10.0)
        self.assertEqual(hits, 2)
        self.assertIn("ยอดรวม", preview)


if __name__ == "__main__":
    unittest.main()
