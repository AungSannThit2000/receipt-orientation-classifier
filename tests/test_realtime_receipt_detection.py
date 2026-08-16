from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from receipt_preprocessing import ReceiptExtraction  # noqa: E402
from realtime_receipt_detection import YOLOWorldReceiptDetector  # noqa: E402


class FakeBoxes(SimpleNamespace):
    def __len__(self) -> int:
        return len(self.conf)


class ReceiptObjectDetectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = object.__new__(YOLOWorldReceiptDetector)
        self.detector.padding_ratio = 0.02
        self.detector.model = SimpleNamespace(names={0: "receipt", 1: "document"})

    def test_select_detection_prefers_confident_central_receipt(self) -> None:
        boxes = FakeBoxes(
            xyxy=torch.tensor(
                [
                    [10.0, 10.0, 70.0, 100.0],
                    [230.0, 90.0, 430.0, 510.0],
                ]
            ),
            conf=torch.tensor([0.40, 0.55]),
            cls=torch.tensor([0.0, 1.0]),
        )
        result = SimpleNamespace(boxes=boxes, names={0: "receipt", 1: "document"})

        selected = self.detector._select_detection(result, (640, 640))

        self.assertIsNotNone(selected)
        bbox, confidence, label = selected
        self.assertEqual(bbox, (230, 90, 430, 510))
        self.assertAlmostEqual(confidence, 0.55, places=5)
        self.assertEqual(label, "document")

    def test_map_refined_crop_back_to_original_coordinates(self) -> None:
        local = ReceiptExtraction(
            image=Image.new("RGB", (180, 380), "white"),
            method="edges",
            confidence=0.9,
            area_ratio=0.72,
            bbox=(10, 10, 190, 390),
            polygon=((10, 10), (190, 10), (190, 390), (10, 390)),
        )
        proposal = Image.new("RGB", (200, 400), "white")

        mapped = self.detector._map_extraction(
            local,
            proposal,
            proposal_bbox=(100, 200, 300, 600),
            original_size=(800, 800),
        )

        self.assertEqual(mapped.method, "yolo_world+edges")
        self.assertEqual(mapped.bbox, (110, 210, 290, 590))
        self.assertEqual(mapped.polygon[0], (110, 210))
        self.assertEqual(mapped.image.size, (180, 380))

    def test_rejected_refinement_uses_the_full_proposal(self) -> None:
        local = ReceiptExtraction(
            image=Image.new("RGB", (20, 20), "white"),
            method="edges",
            confidence=0.5,
            area_ratio=0.05,
            bbox=(0, 0, 20, 20),
            polygon=((0, 0), (19, 0), (19, 19), (0, 19)),
        )
        proposal = Image.new("RGB", (200, 400), "white")

        mapped = self.detector._map_extraction(
            local,
            proposal,
            proposal_bbox=(100, 200, 300, 600),
            original_size=(800, 800),
        )

        self.assertEqual(mapped.method, "yolo_world_box")
        self.assertEqual(mapped.image.size, (200, 400))
        self.assertEqual(mapped.bbox, (100, 200, 300, 600))


if __name__ == "__main__":
    unittest.main()
