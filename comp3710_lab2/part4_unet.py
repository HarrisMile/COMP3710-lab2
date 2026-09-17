from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import (
    dataloader_kwargs,
    load_model_checkpoint,
    make_output_dir,
    save_checkpoint,
    save_json,
    select_device,
    seed_everything,
)
from .metrics import categorical_cross_entropy, soft_dice_loss
from .models import UNet
from .oasis import (
    OASIS_LABEL_VALUES,
    OASIS_NUM_CLASSES,
    OASISSegmentationDataset,
    resolve_oasis_root,
)


def make_loader(
    root: Path,
    split: str,
    image_size: int,
    batch_size: int,
    workers: int,
    device: torch.device,
    limit: int | None,
) -> DataLoader:
    dataset = OASISSegmentationDataset(
        root,
        split,
        image_size=image_size,
        horizontal_flip=split == "train",
        limit=limit,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        **dataloader_kwargs(device, workers),
    )


def _dice_counts(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    intersections = []
    denominators = []
    for class_index in range(OASIS_NUM_CLASSES):
        predicted = predictions == class_index
        expected = targets == class_index
        intersections.append((predicted & expected).sum(dtype=torch.float64))
        denominators.append(
            predicted.sum(dtype=torch.float64) + expected.sum(dtype=torch.float64)
        )
    return torch.stack(intersections), torch.stack(denominators)


def run_epoch(
    model: UNet,
    loader: DataLoader,
    device: torch.device,
    dice_weight: float,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
) -> dict[str, float | list[float]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct_pixels = 0
    pixel_count = 0
    intersections = torch.zeros(OASIS_NUM_CLASSES, dtype=torch.float64, device=device)
    denominators = torch.zeros(OASIS_NUM_CLASSES, dtype=torch.float64, device=device)
    sample_count = 0
    for images, one_hot_masks, _ in loader:
        images = images.to(device, non_blocking=True)
        one_hot_masks = one_hot_masks.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if scaler.is_enabled()
            else nullcontext()
        )
        with torch.set_grad_enabled(training), context:
            logits = model(images)
            categorical_loss = categorical_cross_entropy(logits, one_hot_masks)
            dice_loss = soft_dice_loss(logits, one_hot_masks)
            loss = categorical_loss + dice_weight * dice_loss
        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        predictions = logits.argmax(dim=1)
        targets = one_hot_masks.argmax(dim=1)
        batch_intersections, batch_denominators = _dice_counts(predictions, targets)
        intersections += batch_intersections
        denominators += batch_denominators
        correct_pixels += (predictions == targets).sum().item()
        pixel_count += targets.numel()
        total_loss += loss.item() * len(images)
        sample_count += len(images)
    dice = (2 * intersections + 1e-6) / (denominators + 1e-6)
    return {
        "loss": total_loss / sample_count,
        "pixel_accuracy": correct_pixels / pixel_count,
        "dice_per_class": dice.float().cpu().tolist(),
        "mean_dice": dice.mean().item(),
    }


@torch.inference_mode()
def save_predictions(
    model: UNet,
    loader: DataLoader,
    device: torch.device,
    output_path: Path,
) -> None:
    model.eval()
    images, masks, names = next(iter(loader))
    images = images.to(device)[:4]
    masks = masks[:4]
    predictions = model(images).argmax(dim=1).cpu()
    targets = masks.argmax(dim=1)
    rows = len(images)
    figure, axes = plt.subplots(rows, 3, figsize=(9, 3 * rows), squeeze=False)
    for row in range(rows):
        axes[row, 0].imshow(images[row, 0].cpu(), cmap="gray")
        axes[row, 0].set_title(names[row])
        axes[row, 1].imshow(targets[row], vmin=0, vmax=OASIS_NUM_CLASSES - 1)
        axes[row, 1].set_title("Ground truth")
        axes[row, 2].imshow(predictions[row], vmin=0, vmax=OASIS_NUM_CLASSES - 1)
        axes[row, 2].set_title("Prediction")
        for axis in axes[row]:
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_history(history: list[dict], output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Validation")
    axes[0].set_title("Categorical CE + Dice loss")
    for class_index, label_value in enumerate(OASIS_LABEL_VALUES):
        axes[1].plot(
            epochs,
            [row["val_dice_per_class"][class_index] for row in history],
            label=f"Label {label_value}",
        )
    axes[1].axhline(0.9, color="black", linestyle="--", label="DSC 0.9 target")
    axes[1].set_ylim(0, 1.01)
    axes[1].set_title("Validation DSC by label")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def flatten_metrics(prefix: str, metrics: dict) -> dict:
    return {
        f"{prefix}_loss": metrics["loss"],
        f"{prefix}_pixel_accuracy": metrics["pixel_accuracy"],
        f"{prefix}_dice_per_class": metrics["dice_per_class"],
        f"{prefix}_mean_dice": metrics["mean_dice"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Part 4 Task 2: OASIS UNet")
    parser.add_argument("--data-root")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", default="results/part4_unet")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = select_device(args.device)
    root = resolve_oasis_root(args.data_root)
    output_dir = make_output_dir(args.output_dir)
    train_loader = make_loader(
        root,
        "train",
        args.image_size,
        args.batch_size,
        args.workers,
        device,
        args.limit,
    )
    val_loader = make_loader(
        root,
        "validate",
        args.image_size,
        args.batch_size,
        args.workers,
        device,
        args.limit,
    )
    test_loader = make_loader(
        root,
        "test",
        args.image_size,
        args.batch_size,
        args.workers,
        device,
        args.limit,
    )
    model = UNet(OASIS_NUM_CLASSES, args.base_channels).to(device)
    if args.checkpoint:
        load_model_checkpoint(args.checkpoint, model, device)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and not args.no_amp
    )
    if args.eval_only:
        metrics = run_epoch(model, test_loader, device, args.dice_weight, None, scaler)
        save_predictions(model, test_loader, device, output_dir / "test_predictions.png")
        save_json(metrics, output_dir / "test_metrics.json")
        print(metrics)
        return

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4
    )
    history = []
    best_mean_dice = -1.0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, device, args.dice_weight, optimizer, scaler
        )
        val_metrics = run_epoch(
            model, val_loader, device, args.dice_weight, None, scaler
        )
        scheduler.step(val_metrics["mean_dice"])
        row = {
            "epoch": epoch,
            **flatten_metrics("train", train_metrics),
            **flatten_metrics("val", val_metrics),
        }
        history.append(row)
        print(row)
        save_predictions(
            model, val_loader, device, output_dir / "validation_predictions_latest.png"
        )
        if val_metrics["mean_dice"] > best_mean_dice:
            best_mean_dice = val_metrics["mean_dice"]
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                {"val_mean_dice": best_mean_dice},
                image_size=args.image_size,
                num_classes=OASIS_NUM_CLASSES,
                label_values=OASIS_LABEL_VALUES,
            )
            save_predictions(
                model, val_loader, device, output_dir / "validation_predictions_best.png"
            )
    load_model_checkpoint(output_dir / "best.pt", model, device)
    test_metrics = run_epoch(model, test_loader, device, args.dice_weight, None, scaler)
    save_predictions(model, test_loader, device, output_dir / "test_predictions.png")
    plot_history(history, output_dir / "training_curves.png")
    save_json(
        {
            "device": str(device),
            "data_root": str(root),
            "label_values": list(OASIS_LABEL_VALUES),
            "best_val_mean_dice": best_mean_dice,
            "test": test_metrics,
            "all_test_labels_above_0_9": all(
                score > 0.9 for score in test_metrics["dice_per_class"]
            ),
            "total_seconds": time.perf_counter() - started,
            "history": history,
        },
        output_dir / "metrics.json",
    )


if __name__ == "__main__":
    main()
