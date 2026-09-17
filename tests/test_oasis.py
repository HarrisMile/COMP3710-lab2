from pathlib import Path

import numpy as np
import torch
from PIL import Image

from comp3710_lab2.oasis import (
    OASISImageDataset,
    OASISSegmentationDataset,
    image_to_mask_name,
)


def make_tiny_oasis(root: Path) -> None:
    image_dir = root / "keras_png_slices_train"
    mask_dir = root / "keras_png_slices_seg_train"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    image = np.arange(64, dtype=np.uint8).reshape(8, 8)
    mask = np.tile(np.array([0, 85, 170, 255], dtype=np.uint8), (8, 2))
    Image.fromarray(image).save(image_dir / "case_001_slice_0.nii.png")
    Image.fromarray(mask).save(mask_dir / "seg_001_slice_0.nii.png")


def test_image_mask_name_mapping() -> None:
    assert image_to_mask_name("case_050_slice_1.nii.png") == "seg_050_slice_1.nii.png"


def test_oasis_image_and_one_hot_mask(tmp_path: Path) -> None:
    make_tiny_oasis(tmp_path)
    image_dataset = OASISImageDataset(tmp_path, "train", image_size=16)
    image = image_dataset[0]
    assert image.shape == (1, 16, 16)
    segmentation_dataset = OASISSegmentationDataset(tmp_path, "train", image_size=8)
    image, one_hot, name = segmentation_dataset[0]
    assert image.shape == (1, 8, 8)
    assert one_hot.shape == (4, 8, 8)
    assert torch.all(one_hot.sum(dim=0) == 1)
    assert set(one_hot.argmax(dim=0).unique().tolist()) == {0, 1, 2, 3}
    assert name == "case_001_slice_0.nii.png"
