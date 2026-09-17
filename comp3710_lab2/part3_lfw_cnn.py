from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .common import (
    dataloader_kwargs,
    load_model_checkpoint,
    make_output_dir,
    save_checkpoint,
    save_json,
    select_device,
    seed_everything,
)
from .models import LFWConvNet
from .part2_eigenfaces import load_lfw


def make_loaders(
    data_home: str | Path,
    min_faces: int,
    batch_size: int,
    workers: int,
    device: torch.device,
    seed: int,
) -> tuple[DataLoader, DataLoader, int, list[str]]:
    images, _, labels, target_names = load_lfw(data_home, min_faces)
    images = images.astype(np.float32)
    if images.max() > 1:
        images /= 255.0
    train_x, test_x, train_y, test_y = train_test_split(
        images,
        labels,
        test_size=0.25,
        random_state=seed,
        stratify=labels,
    )
    train_data = TensorDataset(
        torch.from_numpy(train_x[:, None]), torch.from_numpy(train_y).long()
    )
    test_data = TensorDataset(
        torch.from_numpy(test_x[:, None]), torch.from_numpy(test_y).long()
    )
    loader_options = dataloader_kwargs(device, workers)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, **loader_options
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, **loader_options
    )
    return train_loader, test_loader, len(target_names), target_names.tolist()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    sample_count = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(inputs)
            loss = loss_function(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(inputs)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        sample_count += len(inputs)
    return total_loss / sample_count, correct / sample_count


def plot_history(history: list[dict[str, float]], output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0].plot(epochs, [row["test_loss"] for row in history], label="Test")
    axes[0].set_title("Loss")
    axes[1].plot(epochs, [row["train_accuracy"] for row in history], label="Train")
    axes[1].plot(epochs, [row["test_accuracy"] for row in history], label="Test")
    axes[1].set_title("Accuracy")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


@torch.inference_mode()
def evaluate_predictions(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    sample_count = 0
    labels_all = []
    predictions_all = []
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs)
        total_loss += loss_function(logits, labels).item() * len(inputs)
        sample_count += len(inputs)
        labels_all.append(labels.cpu())
        predictions_all.append(logits.argmax(dim=1).cpu())
    labels_array = torch.cat(labels_all).numpy()
    predictions_array = torch.cat(predictions_all).numpy()
    accuracy = float((labels_array == predictions_array).mean())
    return total_loss / sample_count, accuracy, labels_array, predictions_array


def plot_confusion_matrix(
    matrix: np.ndarray,
    target_names: list[str],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(target_names)), target_names, rotation=45, ha="right")
    axis.set_yticks(range(len(target_names)), target_names)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title("LFW CNN confusion matrix")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


@torch.inference_mode()
def plot_predictions(
    model: nn.Module,
    loader: DataLoader,
    target_names: list[str],
    device: torch.device,
    output_path: Path,
    count: int = 8,
) -> None:
    model.eval()
    inputs, labels = next(iter(loader))
    inputs = inputs.to(device)
    predictions = model(inputs).argmax(dim=1).cpu()
    inputs = inputs.cpu()
    count = min(count, len(inputs))
    rows = 2
    columns = 4
    figure, axes = plt.subplots(rows, columns, figsize=(12, 6))
    for index, axis in enumerate(axes.flat):
        if index >= count:
            axis.axis("off")
            continue
        label = int(labels[index])
        prediction = int(predictions[index])
        axis.imshow(inputs[index, 0], cmap="gray")
        axis.set_title(
            f"True: {target_names[label]}\nPred: {target_names[prediction]}",
            color="green" if label == prediction else "red",
            fontsize=9,
        )
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Part 3.1: two-layer CNN on LFW")
    parser.add_argument("--data-home", default="data")
    parser.add_argument("--min-faces", type=int, default=70)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--rf-metrics",
        default="results/part2_eigenfaces/metrics.json",
        help="Part 2 metrics used for the required CNN-versus-RF comparison",
    )
    parser.add_argument("--output-dir", default="results/part3_lfw_cnn")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = select_device(args.device)
    output_dir = make_output_dir(args.output_dir)
    train_loader, test_loader, num_classes, target_names = make_loaders(
        args.data_home,
        args.min_faces,
        args.batch_size,
        args.workers,
        device,
        args.seed,
    )
    model = LFWConvNet(num_classes).to(device)
    if args.checkpoint:
        load_model_checkpoint(args.checkpoint, model, device)
    loss_function = nn.CrossEntropyLoss()
    if args.eval_only:
        loss, accuracy = run_epoch(model, test_loader, loss_function, device)
        plot_predictions(
            model,
            test_loader,
            target_names,
            device,
            output_dir / "test_predictions.png",
        )
        print(f"test_loss={loss:.4f} test_accuracy={accuracy:.4f}")
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history = []
    best_accuracy = -1.0
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, loss_function, device, optimizer
        )
        test_loss, test_accuracy = run_epoch(model, test_loader, loss_function, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
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
                target_names=target_names,
            )
    duration = time.perf_counter() - start
    load_model_checkpoint(output_dir / "best.pt", model, device)
    best_loss, best_accuracy, labels, predictions = evaluate_predictions(
        model,
        test_loader,
        loss_function,
        device,
    )
    report = classification_report(
        labels,
        predictions,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(labels, predictions)
    plot_confusion_matrix(matrix, target_names, output_dir / "confusion_matrix.png")

    rf_accuracy = None
    rf_metrics_path = Path(args.rf_metrics)
    if rf_metrics_path.is_file():
        rf_accuracy = float(json.loads(rf_metrics_path.read_text())["accuracy"])
    accuracy_difference = None if rf_accuracy is None else best_accuracy - rf_accuracy

    plot_history(history, output_dir / "training_curves.png")
    save_json(
        {
            "device": str(device),
            "duration_seconds": duration,
            "best_test_loss": best_loss,
            "best_test_accuracy": best_accuracy,
            "random_forest_accuracy": rf_accuracy,
            "cnn_minus_random_forest_accuracy": accuracy_difference,
            "cnn_outperformed_random_forest": (
                None if rf_accuracy is None else best_accuracy > rf_accuracy
            ),
            "architecture": {
                "convolution_layers": 2,
                "filters_per_layer": 32,
                "kernel_size": [3, 3],
                "classifier": "dense",
            },
            "target_names": target_names,
            "classification_report": report,
            "confusion_matrix": matrix.tolist(),
            "history": history,
        },
        output_dir / "metrics.json",
    )
    print(
        {
            "best_cnn_accuracy": best_accuracy,
            "random_forest_accuracy": rf_accuracy,
            "cnn_minus_random_forest_accuracy": accuracy_difference,
        }
    )


if __name__ == "__main__":
    main()
