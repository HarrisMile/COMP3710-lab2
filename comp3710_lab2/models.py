from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class LFWConvNet(nn.Module):
    """The required two-layer, 32-filter, 3x3 CNN for LFW."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)
        if stride != 1 or in_channels != channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(inputs)
        output = F.relu(self.bn1(self.conv1(inputs)), inplace=True)
        output = self.bn2(self.conv2(output))
        return F.relu(output + residual, inplace=True)


class CIFARResNet18(nn.Module):
    """ResNet-18 built from scratch with a CIFAR-sized stem."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.in_channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)
        self._reset_parameters()

    def _make_layer(self, channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(self.in_channels, channels, stride)]
        self.in_channels = channels
        layers.extend(BasicBlock(channels, channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.stem(inputs)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.pool(output).flatten(1)
        return self.classifier(output)


class ConvVAE(nn.Module):
    def __init__(
        self,
        image_size: int = 128,
        latent_dim: int = 2,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        if image_size < 32 or image_size & (image_size - 1):
            raise ValueError("image_size must be a power of two and at least 32")
        stages = int(math.log2(image_size)) - 3
        channels = [min(base_channels * (2**index), base_channels * 8) for index in range(stages)]
        encoder_layers: list[nn.Module] = []
        in_channels = 1
        for out_channels in channels:
            encoder_layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            in_channels = out_channels
        self.encoder = nn.Sequential(*encoder_layers)
        self.feature_shape = (channels[-1], 8, 8)
        feature_count = math.prod(self.feature_shape)
        self.to_mean = nn.Linear(feature_count, latent_dim)
        self.to_log_variance = nn.Linear(feature_count, latent_dim)
        self.from_latent = nn.Linear(latent_dim, feature_count)

        decoder_layers: list[nn.Module] = []
        current_channels = channels[-1]
        for out_channels in reversed(channels[:-1]):
            decoder_layers.extend(
                [
                    nn.ConvTranspose2d(
                        current_channels, out_channels, 4, 2, 1, bias=False
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            current_channels = out_channels
        decoder_layers.extend(
            [
                nn.ConvTranspose2d(current_channels, 1, 4, 2, 1),
                nn.Sigmoid(),
            ]
        )
        self.decoder = nn.Sequential(*decoder_layers)
        self.latent_dim = latent_dim

    def encode(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(inputs).flatten(1)
        return self.to_mean(features), self.to_log_variance(features)

    @staticmethod
    def reparameterize(mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
        standard_deviation = torch.exp(0.5 * log_variance)
        return mean + torch.randn_like(standard_deviation) * standard_deviation

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        features = self.from_latent(latent).view(-1, *self.feature_shape)
        return self.decoder(features)

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_variance = self.encode(inputs)
        latent = self.reparameterize(mean, log_variance)
        return self.decode(latent), mean, log_variance


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        inputs = self.up(inputs)
        if inputs.shape[-2:] != skip.shape[-2:]:
            inputs = F.interpolate(inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, inputs], dim=1))


class UNet(nn.Module):
    def __init__(self, num_classes: int = 4, base_channels: int = 32) -> None:
        super().__init__()
        b = base_channels
        self.enc1 = DoubleConv(1, b)
        self.enc2 = DoubleConv(b, b * 2)
        self.enc3 = DoubleConv(b * 2, b * 4)
        self.enc4 = DoubleConv(b * 4, b * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(b * 8, b * 16)
        self.up4 = UpBlock(b * 16, b * 8, b * 8)
        self.up3 = UpBlock(b * 8, b * 4, b * 4)
        self.up2 = UpBlock(b * 4, b * 2, b * 2)
        self.up1 = UpBlock(b * 2, b, b)
        self.output = nn.Conv2d(b, num_classes, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(inputs)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        bottleneck = self.bottleneck(self.pool(enc4))
        output = self.up4(bottleneck, enc4)
        output = self.up3(output, enc3)
        output = self.up2(output, enc2)
        output = self.up1(output, enc1)
        return self.output(output)


def _validate_gan_image_size(image_size: int) -> None:
    if image_size < 32 or image_size & (image_size - 1):
        raise ValueError("image_size must be a power of two and at least 32")


class Generator(nn.Module):
    def __init__(
        self,
        image_size: int = 128,
        latent_dim: int = 128,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        _validate_gan_image_size(image_size)
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(latent_dim, base_channels * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
        ]
        size = 4
        channels = base_channels * 8
        while size < image_size // 2:
            next_channels = max(base_channels, channels // 2)
            layers.extend(
                [
                    nn.ConvTranspose2d(channels, next_channels, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(next_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            channels = next_channels
            size *= 2
        layers.extend(
            [
                nn.ConvTranspose2d(channels, 1, 4, 2, 1, bias=False),
                nn.Tanh(),
            ]
        )
        self.layers = nn.Sequential(*layers)
        self.latent_dim = latent_dim
        self.apply(dcgan_weights_init)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.layers(latent)


class Discriminator(nn.Module):
    def __init__(self, image_size: int = 128, base_channels: int = 64) -> None:
        super().__init__()
        _validate_gan_image_size(image_size)
        layers: list[nn.Module] = [
            nn.Conv2d(1, base_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        size = image_size // 2
        channels = base_channels
        while size > 4:
            next_channels = min(base_channels * 8, channels * 2)
            layers.extend(
                [
                    nn.Conv2d(channels, next_channels, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(next_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            channels = next_channels
            size //= 2
        layers.append(nn.Conv2d(channels, 1, 4, 1, 0, bias=False))
        self.layers = nn.Sequential(*layers)
        self.apply(dcgan_weights_init)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs).flatten()


def dcgan_weights_init(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)
