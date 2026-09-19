"""
Unit tests for LoveDA taxonomy and the dataset validator.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from datasets.loveda.manifest import scan_dataset
from datasets.loveda.taxonomy import (
    IGNORE_INDEX,
    NUM_CLASSES,
    LoveDATaxonomyError,
    class_by_id,
    class_by_name,
    class_names,
    is_valid_mask_value,
)
from datasets.loveda.validate import validate_manifest


def test_num_classes_is_seven():
    assert NUM_CLASSES == 7


def test_ignore_index_is_zero_and_not_a_class():
    assert IGNORE_INDEX == 0
    with pytest.raises(LoveDATaxonomyError):
        class_by_id(0)


def test_class_by_id_and_name_roundtrip():
    building = class_by_id(2)
    assert building.name == "building"
    assert class_by_name("building").id == 2


def test_unknown_class_lookup_raises():
    with pytest.raises(LoveDATaxonomyError):
        class_by_id(99)
    with pytest.raises(LoveDATaxonomyError):
        class_by_name("skyscraper")


def test_is_valid_mask_value():
    assert is_valid_mask_value(0)
    assert is_valid_mask_value(7)
    assert not is_valid_mask_value(8)
    assert not is_valid_mask_value(-1)


def test_class_names_ordering():
    names = class_names()
    assert names[0] == "background"
    assert names[-1] == "agriculture"
    assert class_names(include_ignore=True)[0] == "ignore"


def _write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def test_validate_manifest_flags_invalid_pixel_values_and_size_mismatch(tmp_path):
    images_dir = tmp_path / "Train" / "Urban" / "images_png"
    masks_dir = tmp_path / "Train" / "Urban" / "masks_png"

    # Sample 0: valid.
    _write_png(images_dir / "0.png", np.zeros((8, 8, 3), dtype=np.uint8))
    _write_png(masks_dir / "0.png", np.ones((8, 8), dtype=np.uint8))

    # Sample 1: mask has an out-of-range pixel value (9).
    _write_png(images_dir / "1.png", np.zeros((8, 8, 3), dtype=np.uint8))
    bad_mask = np.ones((8, 8), dtype=np.uint8)
    bad_mask[0, 0] = 9
    _write_png(masks_dir / "1.png", bad_mask)

    # Sample 2: mask dimensions don't match the image.
    _write_png(images_dir / "2.png", np.zeros((8, 8, 3), dtype=np.uint8))
    _write_png(masks_dir / "2.png", np.ones((4, 4), dtype=np.uint8))

    manifest = scan_dataset(tmp_path)
    report = validate_manifest(manifest)

    assert not report.ok
    problems = {issue.sample_id: issue.problem for issue in report.issues}
    assert any("invalid mask pixel values" in p for p in problems.values())
    assert any("size mismatch" in p for p in problems.values())
    # Ignore-index pixels must never appear in the class histogram.
    assert IGNORE_INDEX not in report.class_pixel_counts


def test_validate_manifest_reports_clean_dataset(tmp_path):
    images_dir = tmp_path / "Train" / "Urban" / "images_png"
    masks_dir = tmp_path / "Train" / "Urban" / "masks_png"

    _write_png(images_dir / "0.png", np.zeros((8, 8, 3), dtype=np.uint8))
    _write_png(masks_dir / "0.png", np.full((8, 8), 3, dtype=np.uint8))

    manifest = scan_dataset(tmp_path)
    report = validate_manifest(manifest)

    assert report.ok
    assert report.class_pixel_counts == {3: 64}
