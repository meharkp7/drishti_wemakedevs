"""
Unit tests for the LoveDA manifest scanner.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from datasets.loveda.manifest import ManifestError, scan_dataset


def _write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _build_fake_loveda(root: Path) -> None:
    rng = np.random.default_rng(0)

    layout = {
        ("Train", "Urban"): 3,
        ("Train", "Rural"): 2,
        ("Val", "Urban"): 1,
        ("Test", "Urban"): 2,  # no masks
    }

    for (split_dir, domain_dir), n in layout.items():
        images_dir = root / split_dir / domain_dir / "images_png"
        masks_dir = root / split_dir / domain_dir / "masks_png"

        for i in range(n):
            image = (rng.random((8, 8, 3)) * 255).astype(np.uint8)
            _write_png(images_dir / f"{i}.png", image)

            if split_dir != "Test":
                mask = rng.integers(0, 8, size=(8, 8)).astype(np.uint8)
                _write_png(masks_dir / f"{i}.png", mask)


def test_scan_dataset_finds_all_samples(tmp_path):
    _build_fake_loveda(tmp_path)

    manifest = scan_dataset(tmp_path)

    assert len(manifest) == 3 + 2 + 1 + 2
    stats = manifest.stats()
    assert stats.by_split == {"train": 5, "val": 1, "test": 2}
    assert stats.by_domain == {"urban": 6, "rural": 2}
    assert stats.missing_masks == 2


def test_scan_dataset_tolerates_one_extra_nesting_level(tmp_path):
    nested_root = tmp_path / "LoveDA"
    _build_fake_loveda(nested_root)

    manifest = scan_dataset(tmp_path)

    assert len(manifest) == 8


def test_scan_dataset_rejects_train_sample_missing_mask(tmp_path):
    images_dir = tmp_path / "Train" / "Urban" / "images_png"
    _write_png(images_dir / "0.png", np.zeros((4, 4, 3), dtype=np.uint8))
    # Deliberately no masks_png directory at all.

    with pytest.raises(ManifestError):
        scan_dataset(tmp_path)


def test_scan_dataset_rejects_missing_root(tmp_path):
    with pytest.raises(ManifestError):
        scan_dataset(tmp_path / "does_not_exist")


def test_scan_dataset_rejects_empty_images_dir(tmp_path):
    (tmp_path / "Train" / "Urban" / "images_png").mkdir(parents=True)

    with pytest.raises(ManifestError):
        scan_dataset(tmp_path)


def test_manifest_filter_by_split_and_domain(tmp_path):
    _build_fake_loveda(tmp_path)
    manifest = scan_dataset(tmp_path)

    train_only = manifest.filter(split="train")
    assert len(train_only) == 5
    assert all(sample.split == "train" for sample in train_only.samples)

    urban_only = manifest.filter(domain="urban")
    assert all(sample.domain == "urban" for sample in urban_only.samples)

    train_rural = manifest.filter(split="train", domain="rural")
    assert len(train_rural) == 2


def test_manifest_filter_rejects_unknown_split_or_domain(tmp_path):
    _build_fake_loveda(tmp_path)
    manifest = scan_dataset(tmp_path)

    with pytest.raises(ManifestError):
        manifest.filter(split="bogus")

    with pytest.raises(ManifestError):
        manifest.filter(domain="bogus")


def test_sample_ids_are_stable_and_unique(tmp_path):
    _build_fake_loveda(tmp_path)
    manifest = scan_dataset(tmp_path)

    sample_ids = [sample.sample_id for sample in manifest.samples]
    assert len(sample_ids) == len(set(sample_ids))
