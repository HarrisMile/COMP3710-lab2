from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from .common import (
    dataloader_kwargs,
    load_model_checkpoint,
    make_output_dir,
    save_checkpoint,
    save_json,
    select_device,
    seed_everything,
)
from .models import ConvVAE
from .oasis import OASISImageDataset, resolve_oasis_root


def vae_loss(
    reconstruction: torch.Tensor,
    inputs: torch.Tensor,
    mean: torch.Tensor,
    log_variance: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loss_context = (
        torch.autocast(device_type="cuda", enabled=False)
        if reconstruction.device.type == "cuda"
        else nullcontext()
    )
    with loss_context:
        reconstruction = reconstruction.float()
        inputs = inputs.float()
        mean = mean.float()
        log_variance = log_variance.float()
        reconstruction_loss = F.binary_cross_entropy(
            reconstruction, inputs, reduction="sum"
        ) / len(inputs)
        kl_loss = -0.5 * torch.sum(
            1 + log_variance - mean.square() - log_variance.exp()
        )
        kl_loss = kl_loss / len(inputs)
    return reconstruction_loss + beta * kl_loss, reconstruction_loss, kl_loss


def make_loader(
    root: Path,
    split: str,
    image_size: int,
    batch_size: int,
    workers: int,
    device: torch.device,
    limit: int | None,
) -> DataLoader:
    dataset = OASISImageDataset(
        root,
        split,
        image_size=image_size,
        value_range="zero_one",
        horizontal_flip=split == "train",
        limit=limit,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        **dataloader_kwargs(device, workers),
    )


def run_epoch(
    model: ConvVAE,
    loader: DataLoader,
    device: torch.device,
    beta: float,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "reconstruction": 0.0, "kl": 0.0}
    sample_count = 0
    for inputs in loader:
        inputs = inputs.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if scaler.is_enabled()
            else nullcontext()
        )
        with torch.set_grad_enabled(training), context:
            reconstruction, mean, log_variance = model(inputs)
            loss, reconstruction_loss, kl_loss = vae_loss(
                reconstruction, inputs, mean, log_variance, beta
            )
        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        batch_size = len(inputs)
        totals["loss"] += loss.item() * batch_size
        totals["reconstruction"] += reconstruction_loss.item() * batch_size
        totals["kl"] += kl_loss.item() * batch_size
        sample_count += batch_size
    return {key: value / sample_count for key, value in totals.items()}


@torch.inference_mode()
def save_reconstructions(
    model: ConvVAE,
    loader: DataLoader,
    device: torch.device,
    output_path: Path,
) -> None:
    model.eval()
    inputs = next(iter(loader)).to(device)[:8]
    reconstruction, _, _ = model(inputs)
    comparison = torch.cat([inputs, reconstruction]).cpu()
    save_image(comparison, output_path, nrow=8, padding=2)


@torch.inference_mode()
def save_manifold(
    model: ConvVAE,
    device: torch.device,
    output_path: Path,
    grid_size: int = 15,
) -> None:
    if model.latent_dim != 2:
        return
    model.eval()
    axis = torch.linspace(-2.5, 2.5, grid_size, device=device)
    latent = torch.cartesian_prod(axis.flip(0), axis)
    generated = model.decode(latent).cpu()
    save_image(generated, output_path, nrow=grid_size, padding=1)


def plot_history(history: list[dict[str, float]], output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Validation")
    axes[0].set_title("VAE objective")
    axes[1].plot(epochs, [row["train_kl"] for row in history], label="Train KL")
    axes[1].plot(epochs, [row["val_kl"] for row in history], label="Validation KL")
    axes[1].set_title("KL divergence")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Part 4 Task 1: OASIS VAE")
    parser.add_argument("--data-root")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", default="results/part4_vae")
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
    model = ConvVAE(args.image_size, args.latent_dim, args.base_channels).to(device)
    if args.checkpoint:
        load_model_checkpoint(args.checkpoint, model, device)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and not args.no_amp
    )
    if args.eval_only:
        metrics = run_epoch(model, val_loader, device, args.beta, None, scaler)
        save_reconstructions(model, val_loader, device, output_dir / "reconstructions.png")
        save_manifold(model, device, output_dir / "manifold.png")
        print(metrics)
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history = []
    best_loss = float("inf")
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, device, args.beta, optimizer, scaler
        )
        val_metrics = run_epoch(model, val_loader, device, args.beta, None, scaler)
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        print(row)
        save_reconstructions(
            model, val_loader, device, output_dir / "reconstructions_latest.png"
        )
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                {"val_loss": best_loss},
                image_size=args.image_size,
                latent_dim=args.latent_dim,
                beta=args.beta,
            )
            save_manifold(model, device, output_dir / "manifold_best.png")
    plot_history(history, output_dir / "training_curves.png")
    save_json(
        {
            "device": str(device),
            "data_root": str(root),
            "image_size": args.image_size,
            "latent_dim": args.latent_dim,
            "base_channels": args.base_channels,
            "beta": args.beta,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "amp": scaler.is_enabled(),
            "best_val_loss": best_loss,
            "total_seconds": time.perf_counter() - started,
            "history": history,
        },
        output_dir / "metrics.json",
    )


if __name__ == "__main__":
    main()
