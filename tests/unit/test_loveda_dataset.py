"""
Unit tests for LoveDADataset.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from datasets.loveda.dataset import LoveDADataset, LoveDADatasetError
from datasets.loveda.taxonomy import IGNORE_INDEX
from datasets.loveda.transforms import eval_transform, train_transform


def _write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _build_fake_loveda(root: Path) -> None:
    rng = np.random.default_rng(1)

    layout = [
        ("Train", "Urban", 4, True),
        ("Val", "Rural", 2, True),
        ("Test", "Urban", 2, False),
    ]

    for split_dir, domain_dir, n, with_masks in layout:
        images_dir = root / split_dir / domain_dir / "images_png"
        masks_dir = root / split_dir / domain_dir / "masks_png"

        for i in range(n):
            image = (rng.random((16, 16, 3)) * 255).astype(np.uint8)
            _write_png(images_dir / f"{i}.png", image)

            if with_masks:
                mask = rng.integers(0, 8, size=(16, 16)).astype(np.uint8)
                _write_png(masks_dir / f"{i}.png", mask)


def test_train_split_returns_image_and_mask_tensors(tmp_path):
    _build_fake_loveda(tmp_path)
    dataset = LoveDADataset(root=tmp_path, split="train", transform=eval_transform())

    assert len(dataset) == 4

    item = dataset[0]
    assert item["image"].shape == (3, 16, 16)
    assert item["image"].dtype == torch.float32
    assert item["mask"].shape == (16, 16)
    assert item["mask"].dtype == torch.long
    assert item["split"] == "train"
    assert item["domain"] == "urban"


def test_test_split_does_not_require_masks(tmp_path):
    _build_fake_loveda(tmp_path)
    dataset = LoveDADataset(root=tmp_path, split="test", transform=eval_transform())

    assert len(dataset) == 2
    item = dataset[0]
    assert torch.all(item["mask"] == IGNORE_INDEX)


def test_domain_filter(tmp_path):
    _build_fake_loveda(tmp_path)
    dataset = LoveDADataset(
        root=tmp_path, split="val", domain="rural", transform=eval_transform()
    )
    assert len(dataset) == 2


def test_empty_split_domain_combination_raises(tmp_path):
    _build_fake_loveda(tmp_path)
    # Val only has Rural samples in this fixture.
    with pytest.raises(LoveDADatasetError):
        LoveDADataset(root=tmp_path, split="val", domain="urban")


def test_invalid_split_name_raises(tmp_path):
    _build_fake_loveda(tmp_path)
    with pytest.raises(LoveDADatasetError):
        LoveDADataset(root=tmp_path, split="bogus")


def test_missing_root_and_manifest_raises(tmp_path):
    with pytest.raises(LoveDADatasetError):
        LoveDADataset(split="train")


def test_reuses_prescanned_manifest(tmp_path):
    from datasets.loveda.manifest import scan_dataset

    _build_fake_loveda(tmp_path)
    manifest = scan_dataset(tmp_path)

    dataset = LoveDADataset(manifest=manifest, split="train", transform=eval_transform())
    assert len(dataset) == 4


def test_train_transform_produces_valid_tensors(tmp_path):
    _build_fake_loveda(tmp_path)
    dataset = LoveDADataset(
        root=tmp_path, split="train", transform=train_transform(size=16)
    )

    item = dataset[0]
    assert item["image"].shape == (3, 16, 16)
    assert item["mask"].shape == (16, 16)


def test_dataloader_batches_samples(tmp_path):
    _build_fake_loveda(tmp_path)
    dataset = LoveDADataset(root=tmp_path, split="train", transform=eval_transform())
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    batch = next(iter(loader))
    assert batch["image"].shape == (2, 3, 16, 16)
    assert batch["mask"].shape == (2, 16, 16)
    assert len(batch["sample_id"]) == 2
