from __future__ import annotations

import argparse
import copy
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets

from .common import (
    load_model_checkpoint,
    make_output_dir,
    save_checkpoint,
    save_json,
    seed_everything,
    select_device,
)
from .models import CIFARResNet18
from .part3_cifar_resnet import CIFAR_MEAN, CIFAR_STD, mixup, plot_history


@dataclass
class CachedCIFAR10:
    train_images: torch.Tensor
    train_labels: torch.Tensor
    test_images: torch.Tensor
    test_labels: torch.Tensor


def _normalise_images(
    images: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    images = images.to(device=device, dtype=dtype).div_(255.0)
    mean = torch.tensor(CIFAR_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    return images.sub_(mean).div_(std)


def _dataset_tensors(
    dataset: datasets.CIFAR10,
    limit: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = len(dataset) if limit is None else min(limit, len(dataset))
    images = torch.from_numpy(np.asarray(dataset.data[:count])).permute(0, 3, 1, 2)
    labels = torch.tensor(dataset.targets[:count], dtype=torch.long)
    return images.contiguous(), labels


def load_cached_cifar10(
    data_dir: str | Path,
    device: torch.device,
    dtype: torch.dtype,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
    fake_data: bool = False,
    prepad: int = 4,
) -> CachedCIFAR10:
    if fake_data:
        train_count = max_train_samples or 128
        test_count = max_test_samples or 64
        train_images = torch.randint(0, 256, (train_count, 3, 32, 32), dtype=torch.uint8)
        test_images = torch.randint(0, 256, (test_count, 3, 32, 32), dtype=torch.uint8)
        train_labels = torch.randint(0, 10, (train_count,))
        test_labels = torch.randint(0, 10, (test_count,))
    else:
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True)
        train_images, train_labels = _dataset_tensors(train_dataset, max_train_samples)
        test_images, test_labels = _dataset_tensors(test_dataset, max_test_samples)

    normalised_train_images = _normalise_images(train_images, device, dtype)
    if prepad > 0:
        normalised_train_images = F.pad(
            normalised_train_images,
            (prepad, prepad, prepad, prepad),
            mode="reflect",
        )

    return CachedCIFAR10(
        train_images=normalised_train_images,
        train_labels=train_labels.to(device),
        test_images=_normalise_images(test_images, device, dtype),
        test_labels=test_labels.to(device),
    )


def augment_cifar_batch(
    images: torch.Tensor,
    padding: int = 4,
    cutout_size: int = 8,
    crop_size: int | None = None,
) -> torch.Tensor:
    """Apply independent random crops, flips, and cutout entirely on the GPU."""
    batch_size, channels, input_height, _ = images.shape
    if crop_size is None:
        crop_size = input_height
    if padding > 0:
        images = F.pad(images, (padding, padding, padding, padding), mode="reflect")
    source_height, source_width = images.shape[-2:]
    if crop_size > source_height or crop_size > source_width:
        raise ValueError("crop_size cannot exceed the padded image dimensions")
    if crop_size != source_height or crop_size != source_width:
        top = torch.randint(
            0, source_height - crop_size + 1, (batch_size,), device=images.device
        )
        left = torch.randint(
            0, source_width - crop_size + 1, (batch_size,), device=images.device
        )
        rows = top[:, None] + torch.arange(crop_size, device=images.device)[None, :]
        columns = left[:, None] + torch.arange(crop_size, device=images.device)[None, :]
        linear_indices = (
            rows[:, :, None] * source_width + columns[:, None, :]
        ).reshape(batch_size, -1)
        images = images.flatten(2).gather(
            2, linear_indices[:, None, :].expand(-1, channels, -1)
        )
        images = images.reshape(batch_size, channels, crop_size, crop_size)

    flip_mask = torch.rand(batch_size, device=images.device) < 0.5
    images = torch.where(flip_mask[:, None, None, None], images.flip(-1), images)

    if cutout_size > 0:
        height, width = images.shape[-2:]
        if cutout_size > height or cutout_size > width:
            raise ValueError("cutout_size cannot exceed the cropped image dimensions")
        top = torch.randint(
            0, height - cutout_size + 1, (batch_size, 1, 1), device=images.device
        )
        left = torch.randint(
            0, width - cutout_size + 1, (batch_size, 1, 1), device=images.device
        )
        rows = torch.arange(height, device=images.device).view(1, height, 1)
        columns = torch.arange(width, device=images.device).view(1, 1, width)
        cutout_mask = ((rows >= top) & (rows < top + cutout_size)) & (
            (columns >= left) & (columns < left + cutout_size)
        )
        images = images.masked_fill(cutout_mask[:, None, :, :], 0.0)
    return images


def _autocast_context(device: torch.device, enabled: bool):
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _synchronise(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _model_state_on_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    model_to_save = getattr(model, "_orig_mod", model)
    return {
        name: value.detach().cpu().clone()
        for name, value in model_to_save.state_dict().items()
    }


def save_dawnbench_log(history: list[dict[str, float]], path: Path) -> None:
    lines = ["epoch\thours\ttop1Accuracy"]
    for row in history:
        hours = row["cumulative_model_update_seconds"] / 3600.0
        accuracy_percent = 100.0 * row["test_accuracy"]
        lines.append(f"{int(row['epoch'])}\t{hours:.9f}\t{accuracy_percent:.4f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def _update_ema(model: nn.Module, ema_model: nn.Module, decay: float) -> None:
    model_to_average = getattr(model, "_orig_mod", model)
    for current, averaged in zip(
        model_to_average.state_dict().values(), ema_model.state_dict().values()
    ):
        if averaged.is_floating_point():
            averaged.lerp_(current.detach(), 1.0 - decay)
        else:
            averaged.copy_(current)


def train_epoch_cached(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss_function: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    mixup_alpha: float,
    cutout_size: int,
    ema_model: nn.Module,
    ema_decay: float,
    ema_update_every: int,
    global_step: int,
) -> tuple[float, float, int]:
    model.train()
    order = torch.randperm(len(images), device=device)
    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    weighted_correct = torch.zeros((), device=device, dtype=torch.float32)
    sample_count = 0
    amp_enabled = scaler.is_enabled()
    usable_count = (len(images) // batch_size) * batch_size
    if usable_count == 0:
        usable_count = len(images)
    order = order[:usable_count]

    for indices in order.split(batch_size):
        inputs = augment_cifar_batch(
            images[indices], padding=0, cutout_size=cutout_size, crop_size=32
        )
        inputs = inputs.contiguous(memory_format=torch.channels_last)
        batch_labels = labels[indices]
        inputs, labels_a, labels_b, weight = mixup(inputs, batch_labels, mixup_alpha)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, amp_enabled):
            logits = model(inputs)
            if weight == 1.0:
                loss = loss_function(logits, labels_a)
            else:
                loss = weight * loss_function(logits, labels_a)
                loss += (1 - weight) * loss_function(logits, labels_b)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        global_step += 1
        if global_step % ema_update_every == 0:
            _update_ema(model, ema_model, ema_decay**ema_update_every)

        predictions = logits.argmax(dim=1)
        weighted_correct += weight * (predictions == labels_a).sum()
        weighted_correct += (1 - weight) * (predictions == labels_b).sum()
        sample_count += len(indices)
        total_loss += loss.detach().float() * len(indices)

    return (
        (total_loss / sample_count).item(),
        (weighted_correct / sample_count).item(),
        global_step,
    )


@torch.inference_mode()
def evaluate_cached(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    loss_function: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    flip_tta: bool,
) -> tuple[float, float]:
    model.eval()
    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    correct = torch.zeros((), device=device, dtype=torch.long)
    for start in range(0, len(images), batch_size):
        stop = min(start + batch_size, len(images))
        inputs = images[start:stop].contiguous(memory_format=torch.channels_last)
        batch_labels = labels[start:stop]
        with _autocast_context(device, amp_enabled):
            logits = model(inputs)
            if flip_tta:
                logits = 0.5 * (logits + model(inputs.flip(-1)))
            loss = loss_function(logits, batch_labels)
        total_loss += loss.detach().float() * len(inputs)
        correct += (logits.argmax(dim=1) == batch_labels).sum()
    return (total_loss / len(images)).item(), (correct / len(images)).item()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Part 3.2: GPU-cached CIFAR-10 ResNet-18 speed challenge"
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--epochs", type=int, default=72)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--cutout-size", type=int, default=8)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--ema-update-every", type=int, default=5)
    parser.add_argument("--no-flip-tta", action="store_true")
    parser.add_argument("--target-accuracy", type=float, default=0.94)
    parser.add_argument("--stop-on-target", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--fake-data", action="store_true")
    parser.add_argument("--output-dir", default="results/part3_cifar_resnet_fast")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_started = time.perf_counter()
    seed_everything(args.seed)
    device = select_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    amp_enabled = device.type == "cuda" and not args.no_amp
    cache_dtype = torch.float16 if amp_enabled else torch.float32
    output_dir = make_output_dir(args.output_dir)
    data = load_cached_cifar10(
        args.data_dir,
        device,
        cache_dtype,
        args.max_train_samples,
        args.max_test_samples,
        args.fake_data,
        prepad=4,
    )

    model = CIFARResNet18().to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if args.checkpoint:
        load_model_checkpoint(args.checkpoint, model, device)
    ema_model = copy.deepcopy(model).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    loss_function = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    if args.eval_only:
        test_loss, test_accuracy = evaluate_cached(
            ema_model,
            data.test_images,
            data.test_labels,
            args.batch_size,
            loss_function,
            device,
            amp_enabled,
            not args.no_flip_tta,
        )
        print(f"test_loss={test_loss:.4f} test_accuracy={test_accuracy:.4f}")
        return
    if args.compile:
        model = torch.compile(model, mode="reduce-overhead")

    optimizer_options: dict[str, object] = {
        "lr": args.learning_rate,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "nesterov": True,
    }
    if device.type == "cuda":
        optimizer_options["fused"] = True
    optimizer = torch.optim.SGD(model.parameters(), **optimizer_options)
    steps_per_epoch = max(1, len(data.train_images) // args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.15,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=1_000.0,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    history: list[dict[str, float]] = []
    best_accuracy = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    target_epoch: int | None = None
    time_to_target: float | None = None
    model_update_time_to_target: float | None = None
    global_step = 0
    cumulative_model_update_seconds = 0.0
    setup_seconds = time.perf_counter() - process_started
    _synchronise(device)
    training_started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        _synchronise(device)
        epoch_started = time.perf_counter()
        model_update_started = time.perf_counter()
        train_loss, train_accuracy, global_step = train_epoch_cached(
            model,
            data.train_images,
            data.train_labels,
            args.batch_size,
            optimizer,
            scheduler,
            loss_function,
            scaler,
            device,
            args.mixup_alpha,
            args.cutout_size,
            ema_model,
            args.ema_decay,
            args.ema_update_every,
            global_step,
        )
        _synchronise(device)
        model_update_seconds = time.perf_counter() - model_update_started
        cumulative_model_update_seconds += model_update_seconds
        test_loss, test_accuracy = evaluate_cached(
            ema_model,
            data.test_images,
            data.test_labels,
            args.batch_size,
            loss_function,
            device,
            amp_enabled,
            not args.no_flip_tta,
        )
        _synchronise(device)
        epoch_seconds = time.perf_counter() - epoch_started
        elapsed = time.perf_counter() - training_started
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "model_update_seconds": model_update_seconds,
            "cumulative_model_update_seconds": cumulative_model_update_seconds,
            "epoch_seconds": epoch_seconds,
            "elapsed_seconds": elapsed,
        }
        history.append(row)
        print(row, flush=True)

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch
            best_state = _model_state_on_cpu(ema_model)

        if target_epoch is None and test_accuracy >= args.target_accuracy:
            target_epoch = epoch
            time_to_target = elapsed
            model_update_time_to_target = cumulative_model_update_seconds
            print(
                f"target_reached accuracy={test_accuracy:.4f} "
                f"epoch={epoch} wall_seconds={elapsed:.3f} "
                f"model_update_seconds={cumulative_model_update_seconds:.3f}",
                flush=True,
            )
            if args.stop_on_target:
                break

    _synchronise(device)
    training_seconds = time.perf_counter() - training_started
    final_epoch = history[-1]["epoch"]
    save_checkpoint(
        output_dir / "last.pt",
        model,
        optimizer,
        final_epoch,
        {"test_accuracy": history[-1]["test_accuracy"]},
        architecture="CIFARResNet18",
    )
    if best_state is None:
        raise RuntimeError("No training epoch completed")
    model_to_save = getattr(model, "_orig_mod", model)
    model_to_save.load_state_dict(best_state)
    save_checkpoint(
        output_dir / "best.pt",
        model,
        None,
        best_epoch,
        {"test_accuracy": best_accuracy},
        architecture="CIFARResNet18",
    )
    plot_history(history, output_dir / "training_curves.png")
    save_dawnbench_log(history, output_dir / "dawnbench.tsv")
    save_json(
        {
            "device": str(device),
            "amp": amp_enabled,
            "compiled": args.compile,
            "gpu_cached_data": device.type == "cuda",
            "ema_decay": args.ema_decay,
            "ema_update_every": args.ema_update_every,
            "flip_tta": not args.no_flip_tta,
            "batch_size": args.batch_size,
            "epochs_planned": args.epochs,
            "epochs_completed": final_epoch,
            "target_accuracy": args.target_accuracy,
            "target_reached": target_epoch is not None,
            "target_epoch": target_epoch,
            "time_to_target_seconds": time_to_target,
            "model_update_time_to_target_seconds": model_update_time_to_target,
            "benchmark_time_to_target_seconds": (
                None
                if model_update_time_to_target is None
                else setup_seconds + model_update_time_to_target
            ),
            "best_test_accuracy": best_accuracy,
            "best_epoch": best_epoch,
            "setup_seconds": setup_seconds,
            "training_seconds": training_seconds,
            "model_update_seconds": cumulative_model_update_seconds,
            "total_process_seconds": time.perf_counter() - process_started,
            "history": history,
        },
        output_dir / "metrics.json",
    )


if __name__ == "__main__":
    main()
