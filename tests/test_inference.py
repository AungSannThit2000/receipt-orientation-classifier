from __future__ import annotations

import sys
import unittest
from pathlib import Path

from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inference import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    EXPECTED_CLASSES,
    ReceiptOrientationClassifier,
)


SOURCE_IMAGE = PROJECT_ROOT / "docs" / "assets" / "demo_upright.jpg"


class ReceiptOrientationClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.classifier = ReceiptOrientationClassifier(DEFAULT_CHECKPOINT)
        cls.source = ImageOps.exif_transpose(Image.open(SOURCE_IMAGE)).convert("RGB")

    def test_checkpoint_contract(self) -> None:
        self.assertEqual(self.classifier.class_names, EXPECTED_CLASSES)
        self.assertEqual(self.classifier.image_size, 224)
        self.assertEqual(self.classifier.architecture, "mobilenet_v3_small_finetune")

    def test_prediction_is_complete(self) -> None:
        result = self.classifier.predict(self.source)
        self.assertIn(result.label, EXPECTED_CLASSES)
        self.assertEqual(result.model_input.size, (224, 224))
        self.assertEqual(set(result.probabilities), set(EXPECTED_CLASSES))
        self.assertAlmostEqual(sum(result.probabilities.values()), 1.0, places=5)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        self.assertGreater(result.corrected_receipt.width, 0)
        self.assertGreater(result.corrected_receipt.height, 0)

        extracted_result = self.classifier.predict_extraction(result.extraction)
        self.assertEqual(extracted_result.label, result.label)
        self.assertEqual(extracted_result.model_input.size, (224, 224))

    def test_all_public_demo_images_complete_inference(self) -> None:
        for path in sorted((PROJECT_ROOT / "docs" / "assets").glob("demo_*.jpg")):
            with self.subTest(path=path.name):
                result = self.classifier.predict(Image.open(path))
                self.assertIn(result.label, EXPECTED_CLASSES)
                self.assertAlmostEqual(sum(result.probabilities.values()), 1.0, places=5)
                self.assertEqual(result.model_input.size, (224, 224))


if __name__ == "__main__":
    unittest.main()
