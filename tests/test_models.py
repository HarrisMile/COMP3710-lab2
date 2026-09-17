import torch

from comp3710_lab2.models import (
    CIFARResNet18,
    ConvVAE,
    Discriminator,
    Generator,
    LFWConvNet,
    UNet,
)
from comp3710_lab2.part4_gan import diversity_ratio, sample_diversity_metrics


@torch.inference_mode()
def test_classification_model_shapes() -> None:
    assert LFWConvNet(7)(torch.randn(2, 1, 50, 37)).shape == (2, 7)
    assert CIFARResNet18()(torch.randn(2, 3, 32, 32)).shape == (2, 10)


@torch.inference_mode()
def test_vae_shape_and_latent_dimension() -> None:
    model = ConvVAE(image_size=64, latent_dim=2, base_channels=8).eval()
    reconstruction, mean, log_variance = model(torch.rand(2, 1, 64, 64))
    assert reconstruction.shape == (2, 1, 64, 64)
    assert mean.shape == log_variance.shape == (2, 2)


@torch.inference_mode()
def test_vae_supports_native_oasis_resolution() -> None:
    model = ConvVAE(image_size=256, latent_dim=2, base_channels=4).eval()
    reconstruction, mean, log_variance = model(torch.rand(1, 1, 256, 256))
    assert reconstruction.shape == (1, 1, 256, 256)
    assert mean.shape == log_variance.shape == (1, 2)


@torch.inference_mode()
def test_unet_shape() -> None:
    model = UNet(num_classes=4, base_channels=4).eval()
    assert model(torch.rand(2, 1, 64, 64)).shape == (2, 4, 64, 64)


@torch.inference_mode()
def test_gan_shapes() -> None:
    generator = Generator(image_size=64, latent_dim=16, base_channels=8).eval()
    discriminator = Discriminator(image_size=64, base_channels=8).eval()
    generated = generator(torch.randn(2, 16, 1, 1))
    assert generated.shape == (2, 1, 64, 64)
    assert discriminator(generated).shape == (2,)


def test_gan_diversity_metrics_detect_collapsed_samples() -> None:
    collapsed = torch.zeros(8, 1, 8, 8)
    varied = torch.arange(8, dtype=torch.float32)[:, None, None, None].expand(-1, 1, 8, 8)
    collapsed_metrics = sample_diversity_metrics(collapsed)
    varied_metrics = sample_diversity_metrics(varied)
    assert collapsed_metrics["mean_pairwise_l1"] == 0
    assert collapsed_metrics["sample_pixel_std"] == 0
    assert varied_metrics["mean_pairwise_l1"] > 0
    assert varied_metrics["sample_pixel_std"] > 0
    ratios = diversity_ratio(varied_metrics, varied_metrics)
    assert all(value == 1 for value in ratios.values())
