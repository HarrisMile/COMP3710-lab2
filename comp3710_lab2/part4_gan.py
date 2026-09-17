from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from .common import (
    dataloader_kwargs,
    make_output_dir,
    save_json,
    select_device,
    seed_everything,
)
from .models import Discriminator, Generator
from .oasis import OASISImageDataset, resolve_oasis_root


def make_loader(
    root: Path,
    image_size: int,
    batch_size: int,
    workers: int,
    device: torch.device,
    limit: int | None,
) -> DataLoader:
    dataset = OASISImageDataset(
        root,
        "train",
        image_size=image_size,
        value_range="minus_one_one",
        horizontal_flip=True,
        limit=limit,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **dataloader_kwargs(device, workers),
    )


def train_epoch(
    generator: Generator,
    discriminator: Discriminator,
    loader: DataLoader,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    scaler_g: torch.amp.GradScaler,
    scaler_d: torch.amp.GradScaler,
    device: torch.device,
    latent_dim: int,
) -> tuple[float, float, float, float]:
    generator.train()
    discriminator.train()
    total_generator_loss = 0.0
    total_discriminator_loss = 0.0
    total_real_score = 0.0
    total_fake_score = 0.0
    batch_count = 0
    amp_enabled = scaler_g.is_enabled()
    for real_images in loader:
        real_images = real_images.to(device, non_blocking=True)
        batch_size = len(real_images)
        real_targets = torch.full((batch_size,), 0.9, device=device)
        fake_targets = torch.zeros(batch_size, device=device)
        discriminator_optimizer.zero_grad(set_to_none=True)
        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        discriminator_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if amp_enabled
            else nullcontext()
        )
        with discriminator_context:
            generated_images = generator(noise)
            real_logits = discriminator(real_images)
            fake_logits = discriminator(generated_images.detach())
            discriminator_loss = loss_function(real_logits, real_targets)
            discriminator_loss += loss_function(fake_logits, fake_targets)
        scaler_d.scale(discriminator_loss).backward()
        scaler_d.step(discriminator_optimizer)
        scaler_d.update()

        generator_optimizer.zero_grad(set_to_none=True)
        generator_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if amp_enabled
            else nullcontext()
        )
        with generator_context:
            generator_logits = discriminator(generated_images)
            generator_loss = loss_function(
                generator_logits, torch.ones(batch_size, device=device)
            )
        scaler_g.scale(generator_loss).backward()
        scaler_g.step(generator_optimizer)
        scaler_g.update()

        total_generator_loss += generator_loss.item()
        total_discriminator_loss += discriminator_loss.item()
        total_real_score += torch.sigmoid(real_logits).mean().item()
        total_fake_score += torch.sigmoid(fake_logits).mean().item()
        batch_count += 1
    return (
        total_generator_loss / batch_count,
        total_discriminator_loss / batch_count,
        total_real_score / batch_count,
        total_fake_score / batch_count,
    )


@torch.inference_mode()
def generate_samples(
    generator: Generator,
    noise: torch.Tensor,
) -> torch.Tensor:
    generator.eval()
    return generator(noise).add(1).div(2).clamp(0, 1)


def sample_diversity_metrics(samples: torch.Tensor) -> dict[str, float]:
    flattened = samples.float().flatten(1)
    pairwise_l1 = torch.pdist(flattened, p=1) / flattened.shape[1]
    return {
        "sample_pixel_std": float(samples.float().std(dim=0).mean().item()),
        "mean_pairwise_l1": float(pairwise_l1.mean().item()),
        "minimum_pairwise_l1": float(pairwise_l1.min().item()),
    }


def diversity_ratio(
    generated: dict[str, float], real: dict[str, float]
) -> dict[str, float]:
    return {
        key: generated[key] / real[key] if real[key] > 0 else float("nan")
        for key in real
    }


def save_sample_grid(samples: torch.Tensor, output_path: Path) -> None:
    save_image(samples.cpu(), output_path, nrow=8, padding=2)


@torch.inference_mode()
def save_latent_interpolation(
    generator: Generator,
    latent_dim: int,
    device: torch.device,
    output_path: Path,
    pairs: int = 8,
    steps: int = 8,
) -> None:
    generator.eval()
    starts = torch.randn(pairs, latent_dim, 1, 1, device=device)
    ends = torch.randn(pairs, latent_dim, 1, 1, device=device)
    alphas = torch.linspace(0, 1, steps, device=device)
    latent = torch.stack(
        [
            (1 - alpha) * starts[pair] + alpha * ends[pair]
            for pair in range(pairs)
            for alpha in alphas
        ]
    )
    save_sample_grid(generate_samples(generator, latent), output_path)


def plot_history(history: list[dict[str, float]], output_path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["generator_loss"] for row in history], label="Generator")
    axes[0].plot(
        epochs, [row["discriminator_loss"] for row in history], label="Discriminator"
    )
    axes[0].set_title("GAN losses")
    axes[1].plot(epochs, [row["real_score"] for row in history], label="D(real)")
    axes[1].plot(epochs, [row["fake_score"] for row in history], label="D(fake)")
    axes[1].set_title("Discriminator confidence")
    axes[2].plot(
        epochs,
        [row["mean_pairwise_l1"] for row in history],
        label="Mean pairwise L1",
    )
    axes[2].plot(
        epochs,
        [row["sample_pixel_std"] for row in history],
        label="Pixel std",
    )
    axes[2].set_title("Generated-sample diversity")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Part 4 Task 3: OASIS DCGAN")
    parser.add_argument("--data-root")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--sample-only", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", default="results/part4_gan")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = select_device(args.device)
    output_dir = make_output_dir(args.output_dir)
    generator = Generator(args.image_size, args.latent_dim, args.base_channels).to(device)
    discriminator = Discriminator(args.image_size, args.base_channels).to(device)
    checkpoint = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if checkpoint.get("image_size", args.image_size) != args.image_size:
            raise ValueError("Checkpoint image size does not match --image-size")
        if checkpoint.get("latent_dim", args.latent_dim) != args.latent_dim:
            raise ValueError("Checkpoint latent dimension does not match --latent-dim")
        generator.load_state_dict(checkpoint["generator_state"])
        discriminator.load_state_dict(checkpoint["discriminator_state"])
    if checkpoint is not None and "fixed_noise" in checkpoint:
        fixed_noise = checkpoint["fixed_noise"].to(device)
    else:
        fixed_noise = torch.randn(64, args.latent_dim, 1, 1, device=device)
    if args.sample_only:
        samples = generate_samples(generator, fixed_noise)
        save_sample_grid(samples, output_dir / "generated_samples.png")
        save_latent_interpolation(
            generator,
            args.latent_dim,
            device,
            output_dir / "latent_interpolation.png",
        )
        print(sample_diversity_metrics(samples))
        return

    root = resolve_oasis_root(args.data_root)
    loader = make_loader(
        root,
        args.image_size,
        args.batch_size,
        args.workers,
        device,
        args.limit,
    )
    real_samples = next(iter(loader))[:64].add(1).div(2).clamp(0, 1)
    save_sample_grid(real_samples, output_dir / "real_samples.png")
    real_diversity = sample_diversity_metrics(real_samples)
    print({"real_reference_diversity": real_diversity}, flush=True)
    generator_optimizer = torch.optim.Adam(
        generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999)
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999)
    )
    start_epoch = 1
    history = []
    if checkpoint is not None:
        if "generator_optimizer_state" in checkpoint:
            generator_optimizer.load_state_dict(checkpoint["generator_optimizer_state"])
        if "discriminator_optimizer_state" in checkpoint:
            discriminator_optimizer.load_state_dict(
                checkpoint["discriminator_optimizer_state"]
            )
        start_epoch = int(checkpoint["epoch"]) + 1
        history = list(checkpoint.get("history", []))
    if start_epoch > args.epochs:
        raise ValueError(
            f"Checkpoint is already at epoch {start_epoch - 1}; "
            f"--epochs must be at least {start_epoch}"
        )
    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler_g = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    scaler_d = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    loss_function = nn.BCEWithLogitsLoss()
    started = time.perf_counter()
    elapsed_offset = history[-1]["elapsed_seconds"] if history else 0.0
    for epoch in range(start_epoch, args.epochs + 1):
        generator_loss, discriminator_loss, real_score, fake_score = train_epoch(
            generator,
            discriminator,
            loader,
            generator_optimizer,
            discriminator_optimizer,
            loss_function,
            scaler_g,
            scaler_d,
            device,
            args.latent_dim,
        )
        samples = generate_samples(generator, fixed_noise)
        diversity = sample_diversity_metrics(samples)
        row = {
            "epoch": epoch,
            "generator_loss": generator_loss,
            "discriminator_loss": discriminator_loss,
            "real_score": real_score,
            "fake_score": fake_score,
            **diversity,
            "elapsed_seconds": elapsed_offset + time.perf_counter() - started,
        }
        history.append(row)
        print(row, flush=True)
        save_sample_grid(samples, output_dir / "samples_latest.png")
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            save_sample_grid(samples, output_dir / f"samples_epoch_{epoch:03d}.png")
            torch.save(
                {
                    "epoch": epoch,
                    "generator_state": generator.state_dict(),
                    "discriminator_state": discriminator.state_dict(),
                    "generator_optimizer_state": generator_optimizer.state_dict(),
                    "discriminator_optimizer_state": discriminator_optimizer.state_dict(),
                    "image_size": args.image_size,
                    "latent_dim": args.latent_dim,
                    "base_channels": args.base_channels,
                    "fixed_noise": fixed_noise.detach().cpu(),
                    "history": history,
                },
                output_dir / "checkpoint_latest.pt",
            )
    save_latent_interpolation(
        generator,
        args.latent_dim,
        device,
        output_dir / "latent_interpolation.png",
    )
    plot_history(history, output_dir / "training_curves.png")
    save_json(
        {
            "device": str(device),
            "data_root": str(root),
            "image_size": args.image_size,
            "latent_dim": args.latent_dim,
            "base_channels": args.base_channels,
            "batch_size": args.batch_size,
            "epochs_completed": history[-1]["epoch"],
            "amp": amp_enabled,
            "real_reference_diversity": real_diversity,
            "final_diversity": {
                key: history[-1][key]
                for key in (
                    "sample_pixel_std",
                    "mean_pairwise_l1",
                    "minimum_pairwise_l1",
                )
            },
            "final_to_real_diversity_ratio": diversity_ratio(
                history[-1], real_diversity
            ),
            "total_seconds": history[-1]["elapsed_seconds"],
            "history": history,
        },
        output_dir / "metrics.json",
    )


if __name__ == "__main__":
    main()
