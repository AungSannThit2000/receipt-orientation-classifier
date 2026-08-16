from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import models, transforms

from receipt_preprocessing import (
    MODEL_CANVAS_COLOR,
    ReceiptExtraction,
    extract_receipt,
    model_canvas,
    rotate_with_fill,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "trained"
    / "mobilenet_v3_small_finetune_full_best.pt"
)

EXPECTED_CLASSES = ("upright", "tilted_right", "upside_down", "tilted_left")
DISPLAY_NAMES = {
    "upright": "Upright",
    "tilted_right": "Tilted right",
    "upside_down": "Upside down",
    "tilted_left": "Tilted left",
}
CORRECTION_DEGREES = {
    "upright": 0,
    "tilted_right": 90,
    "upside_down": 180,
    "tilted_left": -90,
}
CORRECTION_LABELS = {
    "upright": "No quarter-turn needed",
    "tilted_right": "Rotate 90 degrees counter-clockwise",
    "upside_down": "Rotate 180 degrees",
    "tilted_left": "Rotate 90 degrees clockwise",
}


def correct_receipt(
    image: Image.Image,
    label: str,
    fill_color: tuple[int, int, int] = MODEL_CANVAS_COLOR,
) -> Image.Image:
    """Return the extracted receipt rotated upright for the supplied label."""
    try:
        correction_degrees = CORRECTION_DEGREES[label]
    except KeyError as error:
        raise ValueError(f"Unsupported orientation label: {label}") from error

    if correction_degrees:
        return rotate_with_fill(
            image,
            correction_degrees,
            fill_color=fill_color,
        )
    return image.copy()


@dataclass(frozen=True)
class PredictionResult:
    label: str
    confidence: float
    probabilities: dict[str, float]
    margin: float
    extraction: ReceiptExtraction
    model_input: Image.Image
    corrected_receipt: Image.Image
    inference_ms: float

    @property
    def display_label(self) -> str:
        return DISPLAY_NAMES[self.label]

    @property
    def correction_label(self) -> str:
        return CORRECTION_LABELS[self.label]


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(192, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def _build_model(architecture: str, num_classes: int) -> nn.Module:
    if architecture == "mobilenet_v3_small_finetune":
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    if architecture == "simple_cnn":
        return SimpleCNN(num_classes)
    raise ValueError(f"Unsupported checkpoint architecture: {architecture}")


class ReceiptOrientationClassifier:
    def __init__(
        self,
        checkpoint_path: Path = DEFAULT_CHECKPOINT,
        device: str | torch.device = "cpu",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {self.checkpoint_path}")

        self.device = torch.device(device)
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        required_keys = {
            "architecture",
            "class_names",
            "image_size",
            "normalization_mean",
            "normalization_std",
            "state_dict",
        }
        missing_keys = sorted(required_keys.difference(checkpoint))
        if missing_keys:
            raise ValueError(
                "Checkpoint is missing required metadata: " + ", ".join(missing_keys)
            )

        self.architecture = str(checkpoint["architecture"])
        self.class_names = tuple(str(name) for name in checkpoint["class_names"])
        if self.class_names != EXPECTED_CLASSES:
            raise ValueError(
                f"Unexpected checkpoint class order: {self.class_names}; "
                f"expected {EXPECTED_CLASSES}"
            )

        self.image_size = int(checkpoint["image_size"])
        self.fill_color = tuple(
            int(value) for value in checkpoint.get("receipt_fill_color", MODEL_CANVAS_COLOR)
        )
        mean = [float(value) for value in checkpoint["normalization_mean"]]
        std = [float(value) for value in checkpoint["normalization_std"]]
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        self.model = _build_model(self.architecture, len(self.class_names))
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.best_epoch = int(checkpoint.get("best_epoch", 0))
        self.validation_accuracy = float(checkpoint.get("best_val_accuracy", 0.0))

    def predict(self, image: Image.Image) -> PredictionResult:
        normalized_image = ImageOps.exif_transpose(image).convert("RGB")
        extraction = extract_receipt(normalized_image, fill_color=self.fill_color)
        return self.predict_extraction(extraction)

    def predict_extraction(self, extraction: ReceiptExtraction) -> PredictionResult:
        prepared = model_canvas(
            extraction.image,
            size=self.image_size,
            fill_color=self.fill_color,
        )
        inputs = self.transform(prepared).unsqueeze(0).to(self.device)

        started = perf_counter()
        with torch.inference_mode():
            logits = self.model(inputs)
            probability_tensor = torch.softmax(logits, dim=1)[0].cpu()
        inference_ms = (perf_counter() - started) * 1000.0

        probabilities = {
            name: float(probability_tensor[index])
            for index, name in enumerate(self.class_names)
        }
        ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        label, confidence = ranked[0]
        margin = confidence - ranked[1][1]

        corrected = correct_receipt(
            extraction.image,
            label,
            fill_color=self.fill_color,
        )

        return PredictionResult(
            label=label,
            confidence=confidence,
            probabilities=probabilities,
            margin=margin,
            extraction=extraction,
            model_input=prepared,
            corrected_receipt=corrected,
            inference_ms=inference_ms,
        )
