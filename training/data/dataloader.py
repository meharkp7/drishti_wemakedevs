"""
training/data/dataloader.py

Production DataLoader construction for DRISHTI's LoveDA ML pipeline.

Responsibilities:

    manifest
        ↓
    LoveDADataset
        ↓
    deterministic / augmented transforms
        ↓
    optional class/domain-aware sampler
        ↓
    PyTorch DataLoader

The module deliberately does not contain model or loss logic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.loveda.dataset import LoveDADataset
from datasets.loveda.manifest import LoveDAManifest
from datasets.loveda.transforms import (
    eval_transform,
    train_transform,
)
from training.data.sampler import (
    LoveDASamplerFactory,
    SamplingStrategy,
)
from training.data.statistics import (
    LoveDAStatistics,
    LoveDAStatisticsAnalyzer,
)


class LoveDALoaderError(ValueError):
    """Raised when DataLoader configuration is invalid."""


@dataclass(frozen=True)
class LoveDALoaderConfig:
    """Validated configuration for LoveDA DataLoader construction."""

    batch_size: int = 4

    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False

    drop_last_train: bool = False

    seed: int = 42

    image_size: Optional[int] = None

    sampling_strategy: SamplingStrategy = (
        SamplingStrategy.UNIFORM
    )

    class_sampling_power: float = 0.5
    domain_sampling_power: float = 1.0

    sampling_min_weight: float = 0.25
    sampling_max_weight: float = 4.0

    train_samples_per_epoch: Optional[int] = None

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise LoveDALoaderError(
                "batch_size must be > 0."
            )

        if self.num_workers < 0:
            raise LoveDALoaderError(
                "num_workers must be >= 0."
            )

        if self.persistent_workers and self.num_workers == 0:
            raise LoveDALoaderError(
                "persistent_workers requires num_workers > 0."
            )

        if self.seed < 0:
            raise LoveDALoaderError(
                "seed must be >= 0."
            )

        if self.image_size is not None and self.image_size <= 0:
            raise LoveDALoaderError(
                "image_size must be > 0 when provided."
            )

        if self.class_sampling_power < 0:
            raise LoveDALoaderError(
                "class_sampling_power must be >= 0."
            )

        if self.domain_sampling_power < 0:
            raise LoveDALoaderError(
                "domain_sampling_power must be >= 0."
            )

        if self.sampling_min_weight <= 0:
            raise LoveDALoaderError(
                "sampling_min_weight must be > 0."
            )

        if (
            self.sampling_max_weight
            < self.sampling_min_weight
        ):
            raise LoveDALoaderError(
                "sampling_max_weight must be >= "
                "sampling_min_weight."
            )

        if (
            self.train_samples_per_epoch is not None
            and self.train_samples_per_epoch <= 0
        ):
            raise LoveDALoaderError(
                "train_samples_per_epoch must be > 0 "
                "when provided."
            )


@dataclass(frozen=True)
class LoveDALoaders:
    """Complete train/validation/test loader bundle."""

    train: DataLoader
    val: DataLoader
    test: Optional[DataLoader]

    statistics: LoveDAStatistics


def seed_worker(worker_id: int) -> None:
    """
    Seed Python, NumPy, and PyTorch RNGs inside each DataLoader worker.

    PyTorch derives worker-specific initial seeds from the DataLoader
    generator. Propagating that seed into other RNGs prevents duplicated
    augmentation streams across workers.
    """
    del worker_id

    worker_seed = torch.initial_seed() % (
        2**32
    )

    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loveda_loaders(
    manifest: LoveDAManifest,
    *,
    config: LoveDALoaderConfig = LoveDALoaderConfig(),
    include_test: bool = True,
) -> LoveDALoaders:
    """
    Build production train/val/test DataLoaders from one manifest.

    The same manifest is reused for every split so the filesystem is scanned
    exactly once by the caller.
    """
    if not isinstance(
        manifest,
        LoveDAManifest,
    ):
        raise LoveDALoaderError(
            "manifest must be LoveDAManifest."
        )

    train_dataset = LoveDADataset(
        manifest=manifest,
        split="train",
        transform=train_transform(
            size=config.image_size
        ),
        require_masks=True,
    )

    val_dataset = LoveDADataset(
        manifest=manifest,
        split="val",
        transform=eval_transform(
            size=config.image_size
        ),
        require_masks=True,
    )

    test_dataset: Optional[LoveDADataset] = None

    if include_test:
        try:
            test_dataset = LoveDADataset(
                manifest=manifest,
                split="test",
                transform=eval_transform(
                    size=config.image_size
                ),
                require_masks=False,
            )
        except Exception:
            # A local development copy may intentionally contain only
            # train/val. Test availability is therefore optional.
            test_dataset = None

    # --------------------------------------------------------------
    # Statistics
    # --------------------------------------------------------------

    statistics = (
        LoveDAStatisticsAnalyzer(
            manifest
        ).analyze()
    )

    # Statistics used for sampling must correspond exactly to the
    # training subset.
    train_statistics = (
        LoveDAStatisticsAnalyzer(
            manifest
        ).analyze(split="train")
    )

    train_sample_ids = train_dataset.sample_ids

    generator = torch.Generator()
    generator.manual_seed(config.seed)

    # --------------------------------------------------------------
    # Training sampler
    # --------------------------------------------------------------

    sampler_factory = LoveDASamplerFactory(
        train_statistics,
        strategy=config.sampling_strategy,
        class_power=config.class_sampling_power,
        domain_power=config.domain_sampling_power,
        min_weight=config.sampling_min_weight,
        max_weight=config.sampling_max_weight,
    )

    sampler = sampler_factory.build_sampler(
        train_sample_ids,
        num_samples=(
            config.train_samples_per_epoch
            or len(train_sample_ids)
        ),
        replacement=(
            config.sampling_strategy
            != SamplingStrategy.UNIFORM
        ),
        generator=generator,
    )

    # For uniform sampling we use the sampler too, which gives us one
    # deterministic sampling interface across all strategies.
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        drop_last=config.drop_last_train,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    # --------------------------------------------------------------
    # Validation / test loaders
    # --------------------------------------------------------------

    val_generator = torch.Generator()
    val_generator.manual_seed(
        config.seed + 1
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=val_generator,
    )

    test_loader: Optional[DataLoader] = None

    if test_dataset is not None:
        test_generator = torch.Generator()
        test_generator.manual_seed(
            config.seed + 2
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers,
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=test_generator,
        )

    return LoveDALoaders(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        statistics=statistics,
    )