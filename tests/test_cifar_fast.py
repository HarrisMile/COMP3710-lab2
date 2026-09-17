import torch

from comp3710_lab2.part3_cifar_resnet_fast import augment_cifar_batch


def test_gpu_style_augmentation_shape_and_cutout() -> None:
    torch.manual_seed(42)
    images = torch.ones(4, 3, 32, 32)
    augmented = augment_cifar_batch(images, padding=4, cutout_size=8)

    assert augmented.shape == images.shape
    assert torch.isfinite(augmented).all()
    expected_zero_values = 3 * 8 * 8
    assert torch.all((augmented == 0).sum(dim=(1, 2, 3)) == expected_zero_values)
