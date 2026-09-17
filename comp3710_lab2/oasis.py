from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

OASIS_LABEL_VALUES = (0, 85, 170, 255)
OASIS_NUM_CLASSES = len(OASIS_LABEL_VALUES)


def resolve_oasis_root(value: str | Path | None = None) -> Path:
    candidates = []
    if value:
        candidates.append(Path(value).expanduser())
    if os.environ.get("OASIS_ROOT"):
        candidates.append(Path(os.environ["OASIS_ROOT"]).expanduser())
    candidates.extend(
        [
            Path("data/OASIS"),
            Path("/home/groups/comp3710/OASIS"),
        ]
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"OASIS dataset not found. Checked: {checked}")


def split_paths(root: Path, split: str) -> tuple[Path, Path]:
    image_dir = root / f"keras_png_slices_{split}"
    mask_dir = root / f"keras_png_slices_seg_{split}"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    return image_dir, mask_dir


def image_to_mask_name(image_name: str) -> str:
    if not image_name.startswith("case_"):
        raise ValueError(f"Unexpected OASIS image filename: {image_name}")
    return "seg_" + image_name.removeprefix("case_")


def _limited(paths: list[Path], limit: int | None) -> list[Path]:
    if limit is None:
        return paths
    if limit <= 0:
        raise ValueError("limit must be positive")
    return paths[:limit]


class OASISImageDataset(Dataset[torch.Tensor]):
    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: int = 128,
        value_range: str = "zero_one",
        horizontal_flip: bool = False,
        limit: int | None = None,
    ) -> None:
        self.root = Path(root)
        image_dir, _ = split_paths(self.root, split)
        self.files = _limited(sorted(image_dir.glob("*.png")), limit)
        if not self.files:
            raise RuntimeError(f"No PNG images found in {image_dir}")
        if value_range not in {"zero_one", "minus_one_one"}:
            raise ValueError("value_range must be 'zero_one' or 'minus_one_one'")
        self.image_size = image_size
        self.value_range = value_range
        self.horizontal_flip = horizontal_flip

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.files[index]).convert("L")
        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        if self.horizontal_flip and random.random() < 0.5:
            image = TF.hflip(image)
        tensor = TF.to_tensor(image)
        if self.value_range == "minus_one_one":
            tensor = tensor.mul(2).sub(1)
        return tensor


class OASISSegmentationDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, str]]
):
    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: int = 256,
        horizontal_flip: bool = False,
        limit: int | None = None,
    ) -> None:
        self.root = Path(root)
        image_dir, mask_dir = split_paths(self.root, split)
        if not mask_dir.is_dir():
            raise FileNotFoundError(f"Missing mask directory: {mask_dir}")
        images = _limited(sorted(image_dir.glob("*.png")), limit)
        self.pairs = [(path, mask_dir / image_to_mask_name(path.name)) for path in images]
        missing = [mask for _, mask in self.pairs if not mask.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} masks; first: {missing[0]}")
        if not self.pairs:
            raise RuntimeError(f"No image/mask pairs found in {image_dir}")
        self.image_size = image_size
        self.horizontal_flip = horizontal_flip

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        image_path, mask_path = self.pairs[index]
        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")
        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        mask = TF.resize(
            mask,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST,
        )
        if self.horizontal_flip and random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        image_tensor = TF.to_tensor(image)
        raw_mask = torch.from_numpy(np.array(mask, dtype=np.uint8).copy()).long()
        class_mask = torch.full_like(raw_mask, -1)
        for class_index, value in enumerate(OASIS_LABEL_VALUES):
            class_mask[raw_mask == value] = class_index
        if (class_mask < 0).any():
            unknown = torch.unique(raw_mask[class_mask < 0]).tolist()
            raise ValueError(f"Unknown OASIS mask values: {unknown}")
        one_hot = F.one_hot(class_mask, num_classes=OASIS_NUM_CLASSES)
        one_hot = one_hot.permute(2, 0, 1).float()
        return image_tensor, one_hot, image_path.name


def inspect_mask_values(
    root: str | Path,
    split: str,
    limit: int | None = None,
) -> set[int]:
    _, mask_dir = split_paths(Path(root), split)
    values: set[int] = set()
    for index, path in enumerate(sorted(mask_dir.glob("*.png"))):
        if limit is not None and index >= limit:
            break
        values.update(np.unique(np.asarray(Image.open(path).convert("L"))).tolist())
    return values
