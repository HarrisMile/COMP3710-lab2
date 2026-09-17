import torch

from comp3710_lab2.metrics import (
    categorical_cross_entropy,
    dice_per_class,
    soft_dice_loss,
)


def test_perfect_dice_scores_are_one() -> None:
    target_classes = torch.tensor([[[0, 1], [2, 3]]])
    target = torch.nn.functional.one_hot(target_classes, 4).permute(0, 3, 1, 2).float()
    logits = target * 20 - 10
    assert torch.allclose(dice_per_class(logits, target), torch.ones(4))
    assert soft_dice_loss(logits, target).item() < 1e-5
    assert categorical_cross_entropy(logits, target).item() < 1e-5
