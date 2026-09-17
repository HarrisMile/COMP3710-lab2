from __future__ import annotations

import argparse
import math
import time
from contextlib import nullcontext
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .common import (
    dataloader_kwargs,
    load_model_checkpoint,
    make_output_dir,
    save_checkpoint,
    save_json,
    select_device,
    seed_everything,
)
from .models import CIFARResNet18

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def make_loaders(
    data_dir: str | Path,
    batch_size: int,
    workers: int,
    device: torch.device,
    max_train_samples: int | None,
    max_test_samples: int | None,
    fake_data: bool = False,
) -> tuple[DataLoader, DataLoader]:
    training_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.15), value="random"),
        ]
    )
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]
    )
    if fake_data:
        train_data = datasets.FakeData(
            size=max_train_samples or 64,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=training_transform,
        )
        test_data = datasets.FakeData(
            size=max_test_samples or 32,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=test_transform,
            random_offset=10_000,
        )
    else:
        train_data = datasets.CIFAR10(
            data_dir, train=True, download=True, transform=training_transform
        )
        test_data = datasets.CIFAR10(
            data_dir, train=False, download=True, transform=test_transform
        )
    if max_train_samples:
        train_data = Subset(train_data, range(min(max_train_samples, len(train_data))))
    if max_test_samples:
        test_data = Subset(test_data, range(min(max_test_samples, len(test_data))))
    options = dataloader_kwargs(device, workers)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **options,
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, **options
    )
    return train_loader, test_loader


def mixup(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        return inputs, labels, labels, 1.0
    weight = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(len(inputs), device=inputs.device)
    mixed = weight * inputs + (1 - weight) * inputs[permutation]
    return mixed, labels, labels[permutation], weight


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    mixup_alpha: float,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    weighted_correct = 0.0
    sample_count = 0
    amp_enabled = scaler.is_enabled()
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if device.type == "cuda":
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        inputs, labels_a, labels_b, weight = mixup(inputs, labels, mixup_alpha)
        optimizer.zero_grad(set_to_none=True)
        context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if amp_enabled
            else nullcontext()
        )
        with context:
            logits = model(inputs)
            loss = weight * loss_function(logits, labels_a) + (1 - weight) * loss_function(
                logits, labels_b
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        predictions = logits.argmax(dim=1)
        weighted_correct += weight * (predictions == labels_a).sum().item()
        weighted_correct += (1 - weight) * (predictions == labels_b).sum().item()
        sample_count += len(inputs)
        total_loss += loss.item() * len(inputs)
    return total_loss / sample_count, weighted_correct / sample_count


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    sample_count = 0
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if device.type == "cuda":
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        logits = model(inputs)
        loss = loss_function(logits, labels)
        total_loss += loss.item() * len(inputs)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        sample_count += len(inputs)
    return total_loss / sample_count, correct / sample_count


def cosine_with_warmup(epoch: int, warmup_epochs: int, total_epochs: int) -> float:
    if epoch < warmup_epochs:
        return (epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return 0.5 * (1 + math.cos(math.pi * progress))


def plot_history(history: list[dict[str, float]], output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0].plot(epochs, [row["test_loss"] for row in history], label="Test")
    axes[0].set_title("Cross-entropy loss")
    axes[1].plot(epochs, [row["train_accuracy"] for row in history], label="Train")
    axes[1].plot(epochs, [row["test_accuracy"] for row in history], label="Test")
    axes[1].axhline(0.9, color="tab:orange", linestyle="--", label="90% target")
    axes[1].axhline(0.94, color="tab:red", linestyle="--", label="94% target")
    axes[1].set_title("Accuracy")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Part 3.2: CIFAR-10 ResNet-18")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument(
        "--fake-data",
        action="store_true",
        help="Use synthetic images only to smoke-test the training pipeline",
    )
    parser.add_argument("--output-dir", default="results/part3_cifar_resnet")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = select_device(args.device)
    output_dir = make_output_dir(args.output_dir)
    train_loader, test_loader = make_loaders(
        args.data_dir,
        args.batch_size,
        args.workers,
        device,
        args.max_train_samples,
        args.max_test_samples,
        args.fake_data,
    )
    model = CIFARResNet18().to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if args.checkpoint:
        load_model_checkpoint(args.checkpoint, model, device)
    if args.compile:
        model = torch.compile(model)
    loss_function = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    if args.eval_only:
        loss, accuracy = evaluate(model, test_loader, loss_function, device)
        print(f"test_loss={loss:.4f} test_accuracy={accuracy:.4f}")
        return

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: cosine_with_warmup(epoch, args.warmup_epochs, args.epochs),
    )
    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history = []
    best_accuracy = -1.0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_function,
            scaler,
            device,
            args.mixup_alpha,
        )
        test_loss, test_accuracy = evaluate(model, test_loader, loss_function, device)
        elapsed = time.perf_counter() - started
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": elapsed,
        }
        history.append(row)
        print(row)
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                {"test_accuracy": test_accuracy},
                architecture="CIFARResNet18",
            )
        scheduler.step()
    save_checkpoint(
        output_dir / "last.pt",
        model,
        optimizer,
        args.epochs,
        {"test_accuracy": history[-1]["test_accuracy"]},
        architecture="CIFARResNet18",
    )
    plot_history(history, output_dir / "training_curves.png")
    save_json(
        {
            "device": str(device),
            "amp": amp_enabled,
            "best_test_accuracy": best_accuracy,
            "total_seconds": time.perf_counter() - started,
            "history": history,
        },
        output_dir / "metrics.json",
    )


if __name__ == "__main__":
    main()
