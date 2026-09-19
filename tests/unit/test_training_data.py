"""
Adversarial tests for DRISHTI's LoveDA training-data layer.

Covers:
- dataset/sample-ID contract
- statistics correctness
- ignore handling
- unknown labels
- split/domain filtering
- class/sample frequencies
- sampling strategies
- sampling bounds
- deterministic sampler behavior
- missing statistics
- invalid configuration
- DataLoader tensor contracts
- validation determinism
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from datasets.loveda.dataset import (
    LoveDADataset,
    LoveDADatasetError,
)
from training.data.dataloader import (
    LoveDALoaderConfig,
    build_loveda_loaders,
)
from training.data.sampler import (
    LoveDASamplerFactory,
    SamplingError,
    SamplingStrategy,
)
from training.data.statistics import (
    LoveDAStatisticsAnalyzer,
    LoveDAStatisticsError,
)


def _make_mask(
    path: Path,
    values: np.ndarray,
) -> None:
    Image.fromarray(
        values.astype(np.uint8),
        mode="L",
    ).save(path)


def _make_sample(
    root: Path,
    *,
    split: str,
    domain: str,
    sample_id: str,
    mask: np.ndarray,
) -> None:
    """
    Create one synthetic LoveDA-style sample.

    ``sample_id`` is deliberately the filename stem only. The real manifest
    owns canonical sample-ID construction from split/domain/stem.
    """
    image_dir = (
        root
        / split
        / domain
        / "images"
    )

    mask_dir = (
        root
        / split
        / domain
        / "masks"
    )

    image_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mask_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = np.zeros(
        (mask.shape[0], mask.shape[1], 3),
        dtype=np.uint8,
    )

    Image.fromarray(
        image,
        mode="RGB",
    ).save(
        image_dir / f"{sample_id}.png"
    )

    _make_mask(
        mask_dir / f"{sample_id}.png",
        mask,
    )


@pytest.fixture
def synthetic_manifest(tmp_path):
    from datasets.loveda.manifest import scan_dataset

    # 16 pixels:
    #
    # class 1 -> 3
    # class 2 -> 2
    # class 3 -> 2
    # class 4 -> 2
    # class 5 -> 2
    # class 6 -> 2
    # class 7 -> 2
    # ignore   -> 1
    #
    # Total = 16
    train_mask_a = np.array(
        [
            [1, 1, 2, 2],
            [1, 3, 3, 0],
            [4, 4, 5, 5],
            [6, 6, 7, 7],
        ],
        dtype=np.uint8,
    )

    # 16 pixels:
    #
    # class 1 -> 3
    # class 2 -> 2
    # class 3 -> 3
    # class 4 -> 0
    # class 5 -> 0
    # class 6 -> 2
    # class 7 -> 4
    # ignore   -> 2
    #
    # Total = 16
    train_mask_b = np.array(
        [
            [1, 1, 1, 2],
            [2, 3, 3, 3],
            [6, 6, 7, 7],
            [7, 7, 0, 0],
        ],
        dtype=np.uint8,
    )

    val_mask = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 0],
            [1, 2, 3, 4],
            [5, 6, 7, 0],
        ],
        dtype=np.uint8,
    )

    _make_sample(
        tmp_path,
        split="train",
        domain="urban",
        sample_id="001",
        mask=train_mask_a,
    )

    _make_sample(
        tmp_path,
        split="train",
        domain="rural",
        sample_id="001",
        mask=train_mask_b,
    )

    _make_sample(
        tmp_path,
        split="val",
        domain="urban",
        sample_id="001",
        mask=val_mask,
    )

    manifest = scan_dataset(tmp_path)

    return manifest


def test_dataset_exposes_stable_sample_ids(
    synthetic_manifest,
):
    dataset = LoveDADataset(
        manifest=synthetic_manifest,
        split="train",
    )

    assert dataset.sample_ids == (
        "train-urban-001",
        "train-rural-001",
    )

    assert len(dataset.sample_ids) == len(dataset)

    assert len(
        set(dataset.sample_ids)
    ) == len(dataset.sample_ids)


def test_sample_ids_match_dataset_index_order(
    synthetic_manifest,
):
    dataset = LoveDADataset(
        manifest=synthetic_manifest,
        split="train",
    )

    for index, sample_id in enumerate(
        dataset.sample_ids
    ):
        item = dataset[index]

        assert item["sample_id"] == sample_id


def test_statistics_counts_ignore_correctly(
    synthetic_manifest,
):
    stats = (
        LoveDAStatisticsAnalyzer(
            synthetic_manifest
        ).analyze(split="train")
    )

    assert stats.total_samples == 2
    assert stats.labeled_samples == 2

    assert stats.total_pixels == 32

    # First mask: 1 ignored pixel.
    # Second mask: 2 ignored pixels.
    assert stats.ignored_pixels == 3

    assert stats.valid_pixels == 29

    assert sum(
        stats.class_pixel_counts.values()
    ) == stats.valid_pixels


def test_statistics_class_frequencies_sum_to_one(
    synthetic_manifest,
):
    stats = (
        LoveDAStatisticsAnalyzer(
            synthetic_manifest
        ).analyze(split="train")
    )

    frequencies = (
        stats.class_pixel_frequencies
    )

    assert sum(
        frequencies.values()
    ) == pytest.approx(1.0)


def test_statistics_class_pixel_counts(
    synthetic_manifest,
):
    stats = (
        LoveDAStatisticsAnalyzer(
            synthetic_manifest
        ).analyze(split="train")
    )

    assert stats.class_pixel_counts == {
        1: 6,
        2: 4,
        3: 5,
        4: 2,
        5: 2,
        6: 4,
        7: 6,
    }


def test_statistics_class_sample_counts(
    synthetic_manifest,
):
    stats = (
        LoveDAStatisticsAnalyzer(
            synthetic_manifest
        ).analyze(split="train")
    )

    assert stats.class_sample_counts[1] == 2
    assert stats.class_sample_counts[2] == 2
    assert stats.class_sample_counts[3] == 2
    assert stats.class_sample_counts[4] == 1
    assert stats.class_sample_counts[5] == 1
    assert stats.class_sample_counts[6] == 2
    assert stats.class_sample_counts[7] == 2


def test_statistics_split_filtering(
    synthetic_manifest,
):
    stats = (
        LoveDAStatisticsAnalyzer(
            synthetic_manifest
        ).analyze(split="val")
    )

    assert stats.total_samples == 1
    assert stats.labeled_samples == 1
    assert stats.by_split == {"val": 1}
    assert stats.by_domain == {"urban": 1}


def test_statistics_domain_filtering(
    synthetic_manifest,
):
    stats = (
        LoveDAStatisticsAnalyzer(
            synthetic_manifest
        ).analyze(
            split="train",
            domain="rural",
        )
    )

    assert stats.total_samples == 1
    assert stats.by_domain == {"rural": 1}


def test_unknown_mask_label_is_rejected(
    synthetic_manifest,
    tmp_path,
):
    bad_mask = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 99],
        ],
        dtype=np.uint8,
    )

    _make_sample(
        tmp_path,
        split="train",
        domain="urban",
        sample_id="bad-label",
        mask=bad_mask,
    )

    from datasets.loveda.manifest import scan_dataset

    manifest = scan_dataset(tmp_path)

    with pytest.raises(
        LoveDAStatisticsError,
        match="unknown LoveDA mask value",
    ):
        LoveDAStatisticsAnalyzer(
            manifest
        ).analyze(split="train")


def test_uniform_weights_are_equal(
    synthetic_manifest,
):
    stats = (
        LoveDAStatisticsAnalyzer(
            synthetic_manifest
        ).analyze(split="train")
    )

    factory = LoveDASamplerFactory(
        stats,
        strategy=SamplingStrategy.UNIFORM,
    )

    weights = factory.build_weights(
        list(stats.sample_statistics)
    )

    assert torch.allclose(
        weights,
        torch.ones_like(weights),
    )


@pytest.mark.parametrize(
    "strategy",
    [
        SamplingStrategy.CLASS_BALANCED,
        SamplingStrategy.DOMAIN_BALANCED,
        SamplingStrategy.CLASS_DOMAIN_BALANCED,
    ],
)
def test_non_uniform_sampling_produces_valid_weights(
    synthetic_manifest,
    strategy,
):
    stats = (
        LoveDAStatisticsAnalyzer(
            synthetic_manifest
        ).analyze(split="train")
    )

    factory = LoveDASamplerFactory(
        stats,
        strategy=strategy,
    )

    weights = factory.build_weights(
        list(stats.sample_statistics)
    )

    assert weights.numel() == 2

    assert torch.all(
        torch.isfinite(weights)
    )

    assert torch.all(
        weights > 0
    )


def test_sampling_weights_respect_positive_constraint(
    synthetic_manifest,
):
    stats = (
        LoveDAStatisticsAnalyzer(
            synthetic_manifest
        ).analyze(split="train")
    )

    factory = LoveDASamplerFactory(
        stats,
        strategy=SamplingStrategy.CLASS_BALANCED,
        min_weight=0.5,
        max_weight=2.0,
    )

    weights = factory.build_weights(
        list(stats.sample_statistics)
    )

    assert torch.all(
        weights > 0
    )


def test_sampling_weights_are_normalized_to_mean_one(
    synthetic_manifest,
):
    stats = (
        LoveDAStatisticsAnalyzer(
            synthetic_manifest
        ).analyze(split="train")
    )

    factory = LoveDASamplerFactory(
        stats,
        strategy=SamplingStrategy.CLASS_BALANCED,
    )

    weights = factory.build_weights(
        list(stats.sample_statistics)
    )

    assert weights.mean().item() == pytest.approx(
        1.0
    )


def test_missing_sample_statistics_rejected(
    synthetic_manifest,
):
    stats = (
        LoveDAStatisticsAnalyzer(
            synthetic_manifest
        ).analyze(split="train")
    )

    factory = LoveDASamplerFactory(
        stats,
        strategy=SamplingStrategy.CLASS_BALANCED,
    )

    with pytest.raises(
        SamplingError,
        match="No statistics found",
    ):
        factory.build_weights(
            ["does-not-exist"]
        )


def test_invalid_sampler_configuration_rejected(
    synthetic_manifest,
):
    stats = (
        LoveDAStatisticsAnalyzer(
            synthetic_manifest
        ).analyze(split="train")
    )

    with pytest.raises(
        SamplingError,
        match="class_power",
    ):
        LoveDASamplerFactory(
            stats,
            class_power=-1.0,
        )


def test_loader_config_rejects_persistent_workers_without_workers():
    with pytest.raises(
        ValueError,
        match="persistent_workers",
    ):
        LoveDALoaderConfig(
            num_workers=0,
            persistent_workers=True,
        )


def test_loader_config_rejects_invalid_batch_size():
    with pytest.raises(
        ValueError,
        match="batch_size",
    ):
        LoveDALoaderConfig(
            batch_size=0,
        )


def test_loader_config_rejects_invalid_worker_count():
    with pytest.raises(
        ValueError,
        match="num_workers",
    ):
        LoveDALoaderConfig(
            num_workers=-1,
        )


def test_loader_config_rejects_invalid_sampling_power():
    with pytest.raises(
        ValueError,
        match="class_sampling_power",
    ):
        LoveDALoaderConfig(
            class_sampling_power=-0.1,
        )


def test_train_and_val_loader_contract(
    synthetic_manifest,
):
    loaders = build_loveda_loaders(
        synthetic_manifest,
        config=LoveDALoaderConfig(
            batch_size=1,
            num_workers=0,
            sampling_strategy=(
                SamplingStrategy.UNIFORM
            ),
        ),
        include_test=False,
    )

    train_batch = next(
        iter(loaders.train)
    )

    val_batch = next(
        iter(loaders.val)
    )

    assert train_batch["image"].ndim == 4
    assert train_batch["image"].shape[1] == 3

    assert train_batch["mask"].ndim == 3
    assert train_batch["mask"].shape[0] == 1

    assert val_batch["image"].ndim == 4
    assert val_batch["image"].shape[1] == 3

    assert val_batch["mask"].ndim == 3


def test_train_loader_sample_ids_belong_to_train_split(
    synthetic_manifest,
):
    loaders = build_loveda_loaders(
        synthetic_manifest,
        config=LoveDALoaderConfig(
            batch_size=1,
            num_workers=0,
        ),
        include_test=False,
    )

    expected = {
        "train-urban-001",
        "train-rural-001",
    }

    observed = {
        sample_id
        for batch in loaders.train
        for sample_id in batch["sample_id"]
    }

    assert observed <= expected


def test_validation_loader_is_not_randomized(
    synthetic_manifest,
):
    loaders = build_loveda_loaders(
        synthetic_manifest,
        config=LoveDALoaderConfig(
            batch_size=1,
            num_workers=0,
        ),
        include_test=False,
    )

    first = [
        item
        for batch in loaders.val
        for item in batch["sample_id"]
    ]

    second = [
        item
        for batch in loaders.val
        for item in batch["sample_id"]
    ]

    assert first == second


def test_statistics_report_is_json_serializable(
    synthetic_manifest,
):
    stats = (
        LoveDAStatisticsAnalyzer(
            synthetic_manifest
        ).analyze(split="train")
    )

    encoded = json.dumps(
        stats.to_dict()
    )

    assert isinstance(encoded, str)


def test_sampler_is_reproducible_with_same_generator_seed(
    synthetic_manifest,
):
    stats = (
        LoveDAStatisticsAnalyzer(
            synthetic_manifest
        ).analyze(split="train")
    )

    factory = LoveDASamplerFactory(
        stats,
        strategy=SamplingStrategy.CLASS_BALANCED,
    )

    sample_ids = list(
        stats.sample_statistics
    )

    generator_a = torch.Generator()
    generator_a.manual_seed(123)

    generator_b = torch.Generator()
    generator_b.manual_seed(123)

    sampler_a = factory.build_sampler(
        sample_ids,
        num_samples=20,
        generator=generator_a,
    )

    sampler_b = factory.build_sampler(
        sample_ids,
        num_samples=20,
        generator=generator_b,
    )

    assert list(sampler_a) == list(sampler_b)


def test_loveda_dataset_rejects_empty_split(
    synthetic_manifest,
):
    with pytest.raises(
        LoveDADatasetError,
        match="No LoveDA samples found",
    ):
        LoveDADataset(
            manifest=synthetic_manifest,
            split="test",
        )


def test_manifest_rejects_missing_required_train_masks(
    tmp_path,
):
    from datasets.loveda.manifest import (
        ManifestError,
        scan_dataset,
    )

    image_dir = (
        tmp_path
        / "train"
        / "urban"
        / "images"
    )

    image_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = Image.fromarray(
        np.zeros(
            (4, 4, 3),
            dtype=np.uint8,
        ),
        mode="RGB",
    )

    image.save(
        image_dir / "001.png"
    )

    with pytest.raises(
        ManifestError,
        match="no matching mask",
    ):
        scan_dataset(tmp_path)