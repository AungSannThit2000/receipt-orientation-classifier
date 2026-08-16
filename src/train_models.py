from __future__ import annotations

import argparse
import csv
import io
import json
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageFile
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode


ImageFile.LOAD_TRUNCATED_IMAGES = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "generated_manifest.csv"
DEFAULT_ANALYSIS_PATH = PROJECT_ROOT / "data" / "manifests" / "source_analysis.csv"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models" / "trained"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports" / "training"

CLASS_NAMES = ["upright", "tilted_right", "upside_down", "tilted_left"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
FILL_COLOR = (190, 190, 190)


@dataclass(frozen=True)
class Experiment:
    name: str
    architecture: str
    dataset_variant: str
    max_epochs: int
    patience: int


class ImageByteCache:
    def __init__(self) -> None:
        self._items: dict[Path, bytes] = {}

    def open_rgb(self, path: Path) -> Image.Image:
        if path not in self._items:
            self._items[path] = path.read_bytes()
        with Image.open(io.BytesIO(self._items[path])) as image:
            return image.convert("RGB")

    def __len__(self) -> int:
        return len(self._items)


class ManifestDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        data_dir: Path,
        transform,
        cache: ImageByteCache,
    ) -> None:
        self.rows = rows
        self.data_dir = data_dir
        self.transform = transform
        self.cache = cache

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        image = self.cache.open_rgb(self.data_dir / row["generated_image"])
        return self.transform(image), CLASS_TO_INDEX[row["label"]]


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def build_transforms(
    augmentation_profile: str = "baseline",
) -> tuple[transforms.Compose, transforms.Compose]:
    if augmentation_profile == "baseline":
        augmentation_steps = [
            transforms.RandomAffine(
                degrees=3,
                translate=(0.02, 0.02),
                scale=(0.97, 1.03),
                interpolation=InterpolationMode.BILINEAR,
                fill=FILL_COLOR,
            ),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.14, contrast=0.14, saturation=0.08)],
                p=0.55,
            ),
        ]
    elif augmentation_profile == "camera":
        augmentation_steps = [
            transforms.RandomPerspective(
                distortion_scale=0.14,
                p=0.30,
                interpolation=InterpolationMode.BILINEAR,
                fill=FILL_COLOR,
            ),
            transforms.RandomAffine(
                degrees=4.5,
                translate=(0.08, 0.08),
                scale=(0.80, 1.15),
                shear=(-2.0, 2.0),
                interpolation=InterpolationMode.BILINEAR,
                fill=FILL_COLOR,
            ),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.28,
                        contrast=0.30,
                        saturation=0.12,
                        hue=0.02,
                    )
                ],
                p=0.75,
            ),
            transforms.RandomGrayscale(p=0.08),
            transforms.RandomAutocontrast(p=0.20),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.3))],
                p=0.20,
            ),
        ]
    else:
        raise ValueError(f"Unsupported augmentation profile: {augmentation_profile}")

    tensor_steps = [transforms.ToTensor()]
    if augmentation_profile == "camera":
        tensor_steps.append(
            transforms.RandomErasing(
                p=0.18,
                scale=(0.01, 0.08),
                ratio=(0.3, 3.3),
                value="random",
            )
        )
    tensor_steps.append(transforms.Normalize(mean=MEAN, std=STD))
    train_transform = transforms.Compose(augmentation_steps + tensor_steps)
    evaluation_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )
    return train_transform, evaluation_transform


def rows_for_subset(
    manifest_rows: list[dict[str, str]],
    source_methods: dict[str, str],
    split: str,
    subset: str,
) -> list[dict[str, str]]:
    rows = [row for row in manifest_rows if row["split"] == split]
    if subset == "strict":
        rows = [row for row in rows if source_methods[row["source_image"]] != "full_frame_fallback"]
    elif subset == "fallback":
        rows = [row for row in rows if source_methods[row["source_image"]] == "full_frame_fallback"]
    elif subset != "full":
        raise ValueError(f"Unsupported dataset subset: {subset}")
    return rows


def create_loader(
    rows: list[dict[str, str]],
    data_dir: Path,
    transform,
    cache: ImageByteCache,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = ManifestDataset(rows, data_dir, transform, cache)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def build_loaders(
    manifest_rows: list[dict[str, str]],
    source_methods: dict[str, str],
    data_dir: Path,
    variant: str,
    cache: ImageByteCache,
    batch_size: int,
    seed: int,
    augmentation_profile: str = "baseline",
) -> tuple[dict[str, DataLoader], dict[str, int]]:
    train_transform, evaluation_transform = build_transforms(augmentation_profile)
    train_rows = rows_for_subset(manifest_rows, source_methods, "train", variant)
    val_rows = rows_for_subset(manifest_rows, source_methods, "val", variant)
    test_rows = rows_for_subset(manifest_rows, source_methods, "test", "full")
    strict_test_rows = rows_for_subset(manifest_rows, source_methods, "test", "strict")
    fallback_test_rows = rows_for_subset(manifest_rows, source_methods, "test", "fallback")

    loaders = {
        "train": create_loader(
            train_rows, data_dir, train_transform, cache, batch_size, True, seed
        ),
        "val": create_loader(
            val_rows, data_dir, evaluation_transform, cache, batch_size, False, seed
        ),
        "test_full": create_loader(
            test_rows, data_dir, evaluation_transform, cache, batch_size, False, seed
        ),
        "test_strict": create_loader(
            strict_test_rows, data_dir, evaluation_transform, cache, batch_size, False, seed
        ),
        "test_fallback": create_loader(
            fallback_test_rows, data_dir, evaluation_transform, cache, batch_size, False, seed
        ),
    }
    counts = {name: len(loader.dataset) for name, loader in loaders.items()}
    return loaders, counts


def build_model(architecture: str) -> tuple[nn.Module, dict[str, object]]:
    if architecture == "simple_cnn":
        model = SimpleCNN(len(CLASS_NAMES))
        return model, {
            "architecture": architecture,
            "pretrained": False,
            "fine_tuned_feature_blocks": 0,
        }

    if architecture == "mobilenet_v3_small_finetune":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
        for parameter in model.parameters():
            parameter.requires_grad = False

        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, len(CLASS_NAMES))
        for feature_block in model.features[-3:]:
            for parameter in feature_block.parameters():
                parameter.requires_grad = True
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True

        return model, {
            "architecture": architecture,
            "pretrained": True,
            "weights": str(weights),
            "fine_tuned_feature_blocks": 3,
        }

    raise ValueError(f"Unsupported architecture: {architecture}")


def build_optimizer(model: nn.Module, architecture: str) -> torch.optim.Optimizer:
    if architecture == "simple_cnn":
        return torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    feature_parameters = [
        parameter for parameter in model.features[-3:].parameters() if parameter.requires_grad
    ]
    classifier_parameters = [
        parameter for parameter in model.classifier.parameters() if parameter.requires_grad
    ]
    return torch.optim.AdamW(
        [
            {"params": feature_parameters, "lr": 1e-4},
            {"params": classifier_parameters, "lr": 5e-4},
        ],
        weight_decay=1e-4,
    )


def set_model_mode(model: nn.Module, training: bool) -> None:
    model.train(training)
    if not training:
        return
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            own_parameters = list(module.parameters(recurse=False))
            if own_parameters and not any(parameter.requires_grad for parameter in own_parameters):
                module.eval()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    set_model_mode(model, training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if training:
                loss.backward()
                optimizer.step()

            batch_size = inputs.size(0)
            total_loss += float(loss.item()) * batch_size
            total_correct += int((outputs.argmax(dim=1) == targets).sum().item())
            total_samples += batch_size

    if total_samples == 0:
        return 0.0, 0.0
    return total_loss / total_samples, total_correct / total_samples


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[dict[str, object], list[int], list[int]]:
    set_model_mode(model, False)
    y_true: list[int] = []
    y_pred: list[int] = []
    total_loss = 0.0
    total_samples = 0
    start = time.perf_counter()

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets_on_device = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets_on_device)
            predictions = outputs.argmax(dim=1).cpu()

            batch_size = inputs.size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            y_true.extend(targets.tolist())
            y_pred.extend(predictions.tolist())

    elapsed = time.perf_counter() - start
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0,
    )
    correct = sum(actual == predicted for actual, predicted in zip(y_true, y_pred))
    metrics: dict[str, object] = {
        "samples": total_samples,
        "loss": total_loss / total_samples if total_samples else 0.0,
        "accuracy": correct / total_samples if total_samples else 0.0,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "ms_per_image": (elapsed / total_samples) * 1000 if total_samples else 0.0,
        "per_class_accuracy": {},
    }
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_total = sum(actual == class_index for actual in y_true)
        class_correct = sum(
            actual == class_index and predicted == class_index
            for actual, predicted in zip(y_true, y_pred)
        )
        metrics["per_class_accuracy"][class_name] = (
            class_correct / class_total if class_total else 0.0
        )
    return metrics, y_true, y_pred


def parameter_counts(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


def write_history(path: Path, history: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "learning_rate",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
            ],
        )
        writer.writeheader()
        writer.writerows(history)


def plot_history(path: Path, history: list[dict[str, float]], title: str) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_acc"] for row in history], label="train")
    axes[1].plot(epochs, [row["val_acc"] for row in history], label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(
    path: Path,
    y_true: list[int],
    y_pred: list[int],
    title: str,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=25, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    threshold = matrix.max() / 2 if matrix.size and matrix.max() else 0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = int(matrix[row_index, column_index])
            ax.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    experiment: Experiment,
    model_metadata: dict[str, object],
    best_epoch: int,
    best_val_accuracy: float,
    best_val_loss: float,
    image_size: int,
    augmentation_profile: str,
) -> None:
    checkpoint = {
        "experiment_name": experiment.name,
        "architecture": experiment.architecture,
        "dataset_variant": experiment.dataset_variant,
        "class_names": CLASS_NAMES,
        "image_size": image_size,
        "normalization_mean": MEAN,
        "normalization_std": STD,
        "receipt_fill_color": FILL_COLOR,
        "excluded_crop_method": (
            "full_frame_fallback" if experiment.dataset_variant == "strict" else None
        ),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "best_val_loss": best_val_loss,
        "augmentation_profile": augmentation_profile,
        "model_metadata": model_metadata,
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, path)


def train_experiment(
    experiment: Experiment,
    manifest_rows: list[dict[str, str]],
    source_methods: dict[str, str],
    data_dir: Path,
    cache: ImageByteCache,
    batch_size: int,
    seed: int,
    device: torch.device,
    models_dir: Path,
    reports_dir: Path,
    image_size: int,
    augmentation_profile: str,
) -> dict[str, object]:
    set_seed(seed)
    loaders, dataset_counts = build_loaders(
        manifest_rows,
        source_methods,
        data_dir,
        experiment.dataset_variant,
        cache,
        batch_size,
        seed,
        augmentation_profile,
    )
    model, model_metadata = build_model(experiment.architecture)
    model = model.to(device)
    total_parameters, trainable_parameters = parameter_counts(model)
    optimizer = build_optimizer(model, experiment.architecture)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=experiment.max_epochs, eta_min=5e-6
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    checkpoint_path = models_dir / f"{experiment.name}_best.pt"
    history_path = reports_dir / f"history_{experiment.name}.csv"
    curves_path = reports_dir / f"curves_{experiment.name}.png"
    history: list[dict[str, float]] = []
    best_val_accuracy = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    training_start = time.perf_counter()

    print(
        f"\n=== {experiment.name} ===\n"
        f"architecture={experiment.architecture} variant={experiment.dataset_variant} "
        f"train={dataset_counts['train']} val={dataset_counts['val']} "
        f"trainable_parameters={trainable_parameters:,}",
        flush=True,
    )

    for epoch in range(1, experiment.max_epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, loaders["train"], criterion, device, optimizer
        )
        val_loss, val_accuracy = run_epoch(model, loaders["val"], criterion, device)
        learning_rate = optimizer.param_groups[-1]["lr"]
        history.append(
            {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_loss": train_loss,
                "train_acc": train_accuracy,
                "val_loss": val_loss,
                "val_acc": val_accuracy,
            }
        )

        improved = val_accuracy > best_val_accuracy + 1e-8 or (
            abs(val_accuracy - best_val_accuracy) <= 1e-8 and val_loss < best_val_loss
        )
        if improved:
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path,
                model,
                experiment,
                model_metadata,
                best_epoch,
                best_val_accuracy,
                best_val_loss,
                image_size,
                augmentation_profile,
            )
        else:
            epochs_without_improvement += 1

        print(
            f"{experiment.name} epoch {epoch:02d}/{experiment.max_epochs} "
            f"train_acc={train_accuracy:.4f} val_acc={val_accuracy:.4f} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}",
            flush=True,
        )
        scheduler.step()
        if epochs_without_improvement >= experiment.patience:
            print(
                f"{experiment.name} early stopping after {epoch} epochs; "
                f"best epoch was {best_epoch}",
                flush=True,
            )
            break

    training_seconds = time.perf_counter() - training_start
    write_history(history_path, history)
    plot_history(curves_path, history, experiment.name)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    evaluations: dict[str, dict[str, object]] = {}
    confusion_paths: dict[str, str] = {}
    for subset_name in ("test_full", "test_strict", "test_fallback"):
        metrics, y_true, y_pred = evaluate(model, loaders[subset_name], criterion, device)
        evaluations[subset_name] = metrics
        confusion_path = reports_dir / f"confusion_{experiment.name}_{subset_name}.png"
        plot_confusion(
            confusion_path,
            y_true,
            y_pred,
            f"{experiment.name} - {subset_name}",
        )
        confusion_paths[subset_name] = str(confusion_path)

    primary_subset = (
        "test_strict" if experiment.dataset_variant == "strict" else "test_full"
    )
    result: dict[str, object] = {
        "experiment_name": experiment.name,
        "architecture": experiment.architecture,
        "dataset_variant": experiment.dataset_variant,
        "image_size": image_size,
        "augmentation_profile": augmentation_profile,
        "max_epochs": experiment.max_epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "best_val_loss": best_val_loss,
        "primary_test_subset": primary_subset,
        "primary_test_accuracy": evaluations[primary_subset]["accuracy"],
        "test_full_accuracy": evaluations["test_full"]["accuracy"],
        "test_full_macro_f1": evaluations["test_full"]["macro_f1"],
        "test_strict_accuracy": evaluations["test_strict"]["accuracy"],
        "test_strict_macro_f1": evaluations["test_strict"]["macro_f1"],
        "test_fallback_accuracy": evaluations["test_fallback"]["accuracy"],
        "test_fallback_macro_f1": evaluations["test_fallback"]["macro_f1"],
        "full_test_ms_per_image": evaluations["test_full"]["ms_per_image"],
        "training_seconds": training_seconds,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "checkpoint_size_mb": checkpoint_path.stat().st_size / (1024 * 1024),
        "checkpoint": str(checkpoint_path),
        "history_csv": str(history_path),
        "curves_png": str(curves_path),
        "confusion_matrices": confusion_paths,
        "dataset_counts": dataset_counts,
        "evaluations": evaluations,
        "model_metadata": model_metadata,
    }
    print(
        f"{experiment.name} results: full={result['test_full_accuracy']:.4f} "
        f"strict={result['test_strict_accuracy']:.4f} "
        f"fallback={result['test_fallback_accuracy']:.4f}",
        flush=True,
    )
    return result


def write_comparison_csv(path: Path, results: list[dict[str, object]]) -> None:
    fieldnames = [
        "experiment_name",
        "architecture",
        "dataset_variant",
        "best_epoch",
        "epochs_completed",
        "best_val_accuracy",
        "primary_test_accuracy",
        "test_full_accuracy",
        "test_full_macro_f1",
        "test_strict_accuracy",
        "test_strict_macro_f1",
        "test_fallback_accuracy",
        "test_fallback_macro_f1",
        "full_test_ms_per_image",
        "training_seconds",
        "total_parameters",
        "trainable_parameters",
        "checkpoint_size_mb",
        "checkpoint",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result[field] for field in fieldnames})


def write_markdown_report(
    path: Path,
    results: list[dict[str, object]],
    subset_counts: dict[str, int],
) -> None:
    best_full = max(results, key=lambda result: result["test_full_accuracy"])
    best_strict = max(results, key=lambda result: result["test_strict_accuracy"])
    best_fallback = max(results, key=lambda result: result["test_fallback_accuracy"])
    lines = [
        "# Draft 2 Training Comparison",
        "",
        "## Evaluation sets",
        "",
        f"- Full test set: {subset_counts['test_full']} images.",
        f"- Strict-crop test set: {subset_counts['test_strict']} images.",
        f"- Full-frame fallback test set: {subset_counts['test_fallback']} images.",
        "",
        "The fallback test set is small, so its accuracy is diagnostic rather than a stable headline metric.",
        "",
        "## Results",
        "",
        "| Experiment | Training data | Best validation | Full test | Strict test | Fallback test | Size (MB) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in sorted(results, key=lambda item: item["test_full_accuracy"], reverse=True):
        lines.append(
            f"| `{result['experiment_name']}` | {result['dataset_variant']} | "
            f"{result['best_val_accuracy']:.2%} | {result['test_full_accuracy']:.2%} | "
            f"{result['test_strict_accuracy']:.2%} | {result['test_fallback_accuracy']:.2%} | "
            f"{result['checkpoint_size_mb']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Best checkpoints",
            "",
            f"- Best full-test accuracy: `{best_full['experiment_name']}` at {best_full['test_full_accuracy']:.2%}.",
            f"- Best strict-test accuracy: `{best_strict['experiment_name']}` at {best_strict['test_strict_accuracy']:.2%}.",
            f"- Best fallback-test accuracy: `{best_fallback['experiment_name']}` at {best_fallback['test_fallback_accuracy']:.2%}.",
            "",
            "Every checkpoint stores the class order, normalization values, image size, architecture, and dataset variant required for inference.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def smoke_test(
    experiments: list[Experiment],
    manifest_rows: list[dict[str, str]],
    source_methods: dict[str, str],
    data_dir: Path,
    batch_size: int,
    seed: int,
    device: torch.device,
    augmentation_profile: str,
) -> None:
    cache = ImageByteCache()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    for experiment in experiments:
        set_seed(seed)
        loaders, counts = build_loaders(
            manifest_rows,
            source_methods,
            data_dir,
            experiment.dataset_variant,
            cache,
            batch_size,
            seed,
            augmentation_profile,
        )
        model, _ = build_model(experiment.architecture)
        model = model.to(device)
        optimizer = build_optimizer(model, experiment.architecture)
        inputs, targets = next(iter(loaders["train"]))
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        print(
            f"smoke {experiment.name}: batch={inputs.shape[0]} output={tuple(outputs.shape)} "
            f"loss={loss.item():.4f} train_rows={counts['train']}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train receipt orientation models.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--source-analysis", type=Path, default=DEFAULT_ANALYSIS_PATH)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--simple-epochs", type=int, default=18)
    parser.add_argument("--mobile-epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=4201)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--augmentation-profile",
        choices=("baseline", "camera"),
        default="baseline",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=(
            "simple_cnn_full",
            "simple_cnn_strict",
            "mobilenet_v3_small_finetune_full",
            "mobilenet_v3_small_finetune_strict",
        ),
    )
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_csv(args.manifest)
    analysis_rows = read_csv(args.source_analysis)
    source_methods = {row["source_image"]: row["crop_method"] for row in analysis_rows}
    missing_methods = sorted(
        {row["source_image"] for row in manifest_rows} - set(source_methods)
    )
    if missing_methods:
        raise ValueError(f"Sources are missing crop analysis: {missing_methods[:5]}")

    configured = {
        "simple_cnn_full": Experiment(
            "simple_cnn_full", "simple_cnn", "full", args.simple_epochs, 5
        ),
        "simple_cnn_strict": Experiment(
            "simple_cnn_strict", "simple_cnn", "strict", args.simple_epochs, 5
        ),
        "mobilenet_v3_small_finetune_full": Experiment(
            "mobilenet_v3_small_finetune_full",
            "mobilenet_v3_small_finetune",
            "full",
            args.mobile_epochs,
            3,
        ),
        "mobilenet_v3_small_finetune_strict": Experiment(
            "mobilenet_v3_small_finetune_strict",
            "mobilenet_v3_small_finetune",
            "strict",
            args.mobile_epochs,
            3,
        ),
    }
    experiment_names = args.experiments or list(configured)
    experiments = [configured[name] for name in experiment_names]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_test_count = len(rows_for_subset(manifest_rows, source_methods, "test", "full"))
    strict_test_count = len(rows_for_subset(manifest_rows, source_methods, "test", "strict"))
    fallback_test_count = len(rows_for_subset(manifest_rows, source_methods, "test", "fallback"))
    subset_counts = {
        "test_full": full_test_count,
        "test_strict": strict_test_count,
        "test_fallback": fallback_test_count,
    }

    print(
        f"device={device} manifest_rows={len(manifest_rows)} "
        f"test_full={full_test_count} test_strict={strict_test_count} "
        f"test_fallback={fallback_test_count}",
        flush=True,
    )
    if args.smoke_test:
        smoke_test(
            experiments,
            manifest_rows,
            source_methods,
            args.data_dir,
            args.batch_size,
            args.seed,
            device,
            args.augmentation_profile,
        )
        return

    config = {
        "data_dir": str(args.data_dir),
        "manifest": str(args.manifest),
        "source_analysis": str(args.source_analysis),
        "models_dir": str(args.models_dir),
        "reports_dir": str(args.reports_dir),
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "augmentation_profile": args.augmentation_profile,
        "seed": args.seed,
        "device": str(device),
        "class_names": CLASS_NAMES,
        "experiments": [experiment.__dict__ for experiment in experiments],
        "evaluation_subset_counts": subset_counts,
    }
    (args.reports_dir / "training_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    cache = ImageByteCache()
    results: list[dict[str, object]] = []
    for experiment in experiments:
        results.append(
            train_experiment(
                experiment,
                manifest_rows,
                source_methods,
                args.data_dir,
                cache,
                args.batch_size,
                args.seed,
                device,
                args.models_dir,
                args.reports_dir,
                args.image_size,
                args.augmentation_profile,
            )
        )
        (args.reports_dir / "training_comparison.partial.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )

    comparison_csv = args.reports_dir / "training_comparison.csv"
    comparison_json = args.reports_dir / "training_comparison.json"
    comparison_report = args.reports_dir / "training_comparison.md"
    write_comparison_csv(comparison_csv, results)
    comparison_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_markdown_report(comparison_report, results, subset_counts)
    partial_path = args.reports_dir / "training_comparison.partial.json"
    if partial_path.exists():
        partial_path.unlink()

    print("\n=== Final comparison ===", flush=True)
    for result in sorted(results, key=lambda item: item["test_full_accuracy"], reverse=True):
        print(
            f"{result['experiment_name']}: full={result['test_full_accuracy']:.4f} "
            f"strict={result['test_strict_accuracy']:.4f} "
            f"fallback={result['test_fallback_accuracy']:.4f} "
            f"best_val={result['best_val_accuracy']:.4f}",
            flush=True,
        )
    print(f"Image byte cache contains {len(cache)} files", flush=True)
    print(f"Report: {comparison_report}", flush=True)


if __name__ == "__main__":
    main()
