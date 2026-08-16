from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from train_models import (
    FILL_COLOR,
    MEAN,
    STD,
    ImageByteCache,
    build_transforms,
    run_epoch,
    set_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed_384"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "generated_manifest_384.csv"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "trained" / "vertical_direction_384_best.pt"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "training_384" / "vertical_direction.json"
CLASS_NAMES = ["upright", "upside_down"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}


class VerticalDirectionDataset(Dataset):
    def __init__(self, rows, data_dir: Path, transform, cache: ImageByteCache) -> None:
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


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return [row for row in rows if row["label"] in CLASS_TO_INDEX]


def validate_source_splits(rows: list[dict[str, str]]) -> None:
    splits_by_source: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        splits_by_source[row["source_image"]].add(row["split"])
    leakage = {
        source: sorted(splits)
        for source, splits in splits_by_source.items()
        if len(splits) != 1
    }
    if leakage:
        raise ValueError(f"Source receipt leakage detected: {list(leakage.items())[:5]}")


def create_loader(rows, data_dir, transform, cache, batch_size, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        VerticalDirectionDataset(rows, data_dir, transform, cache),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def build_model() -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for block in model.features[-3:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASS_NAMES))
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    return model


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [
            {"params": model.features[-3:].parameters(), "lr": 1e-4},
            {"params": model.classifier.parameters(), "lr": 5e-4},
        ],
        weight_decay=1e-4,
    )


def evaluate_predictions(model, loader, device) -> tuple[float, dict[str, dict[str, int]]]:
    model.eval()
    total = 0
    correct = 0
    matrix = {actual: Counter() for actual in CLASS_NAMES}
    with torch.no_grad():
        for inputs, targets in loader:
            predictions = model(inputs.to(device)).argmax(dim=1).cpu()
            for actual_index, predicted_index in zip(targets, predictions):
                actual = CLASS_NAMES[int(actual_index)]
                predicted = CLASS_NAMES[int(predicted_index)]
                matrix[actual][predicted] += 1
                correct += int(actual == predicted)
                total += 1
    return correct / max(total, 1), {
        actual: {predicted: matrix[actual][predicted] for predicted in CLASS_NAMES}
        for actual in CLASS_NAMES
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a dedicated upright-versus-upside-down direction head."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=4201)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rows = read_manifest(args.manifest)
    validate_source_splits(rows)
    rows_by_split = {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "val", "test")
    }
    train_transform, evaluation_transform = build_transforms("camera")
    cache = ImageByteCache()
    loaders = {
        "train": create_loader(
            rows_by_split["train"], args.data_dir, train_transform, cache,
            args.batch_size, True, args.seed,
        ),
        "val": create_loader(
            rows_by_split["val"], args.data_dir, evaluation_transform, cache,
            args.batch_size, False, args.seed,
        ),
        "test": create_loader(
            rows_by_split["test"], args.data_dir, evaluation_transform, cache,
            args.batch_size, False, args.seed,
        ),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    optimizer = build_optimizer(model)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    if args.smoke_test:
        inputs, targets = next(iter(loaders["train"]))
        outputs = model(inputs.to(device))
        loss = criterion(outputs, targets.to(device))
        loss.backward()
        print(
            f"smoke input={tuple(inputs.shape)} output={tuple(outputs.shape)} "
            f"loss={loss.item():.4f}",
            flush=True,
        )
        return

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, loaders["train"], criterion, device, optimizer
        )
        val_loss, val_accuracy = run_epoch(model, loaders["val"], criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )
        improved = val_accuracy > best_accuracy or (
            val_accuracy == best_accuracy and val_loss < best_loss
        )
        if improved:
            best_accuracy = val_accuracy
            best_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "task": "vertical_direction",
                    "architecture": "mobilenet_v3_small_finetune",
                    "class_names": CLASS_NAMES,
                    "image_size": args.image_size,
                    "normalization_mean": MEAN,
                    "normalization_std": STD,
                    "receipt_fill_color": FILL_COLOR,
                    "augmentation_profile": "camera",
                    "best_epoch": best_epoch,
                    "best_val_accuracy": best_accuracy,
                    "state_dict": model.state_dict(),
                },
                args.checkpoint,
            )
        else:
            stale_epochs += 1
        print(
            f"epoch {epoch:02d}: train={train_accuracy:.4f} val={val_accuracy:.4f}",
            flush=True,
        )
        if stale_epochs >= args.patience:
            break

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    test_accuracy, confusion = evaluate_predictions(model, loaders["test"], device)
    report = {
        "task": "vertical_direction",
        "class_names": CLASS_NAMES,
        "image_size": args.image_size,
        "augmentation_profile": "camera",
        "dataset_counts": {split: len(loader.dataset) for split, loader in loaders.items()},
        "best_epoch": best_epoch,
        "best_val_accuracy": best_accuracy,
        "test_accuracy": test_accuracy,
        "test_confusion": confusion,
        "training_seconds": time.perf_counter() - started,
        "history": history,
        "checkpoint": str(args.checkpoint),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
