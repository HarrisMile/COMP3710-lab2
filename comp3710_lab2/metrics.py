from __future__ import annotations

import torch
import torch.nn.functional as F


def categorical_cross_entropy(
    logits: torch.Tensor,
    one_hot_target: torch.Tensor,
) -> torch.Tensor:
    if logits.shape != one_hot_target.shape:
        raise ValueError(
            f"logits and one_hot_target must match: {logits.shape} != {one_hot_target.shape}"
        )
    return -(one_hot_target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def soft_dice_loss(
    logits: torch.Tensor,
    one_hot_target: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    probabilities = F.softmax(logits, dim=1)
    dimensions = (0, 2, 3)
    intersection = (probabilities * one_hot_target).sum(dim=dimensions)
    denominator = probabilities.sum(dim=dimensions) + one_hot_target.sum(dim=dimensions)
    dice = (2 * intersection + epsilon) / (denominator + epsilon)
    return 1 - dice.mean()


@torch.no_grad()
def dice_per_class(
    logits_or_predictions: torch.Tensor,
    one_hot_target: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    if logits_or_predictions.ndim == 4:
        predictions = logits_or_predictions.argmax(dim=1)
    elif logits_or_predictions.ndim == 3:
        predictions = logits_or_predictions
    else:
        raise ValueError("predictions must have shape NCHW or NHW")
    targets = one_hot_target.argmax(dim=1)
    num_classes = one_hot_target.shape[1]
    scores = []
    for class_index in range(num_classes):
        predicted = predictions == class_index
        expected = targets == class_index
        intersection = (predicted & expected).sum(dtype=torch.float64)
        denominator = predicted.sum(dtype=torch.float64) + expected.sum(dtype=torch.float64)
        scores.append((2 * intersection + epsilon) / (denominator + epsilon))
    return torch.stack(scores).to(dtype=torch.float32)


@torch.no_grad()
def pixel_accuracy(logits: torch.Tensor, one_hot_target: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    targets = one_hot_target.argmax(dim=1)
    return (predictions == targets).float().mean().item()
