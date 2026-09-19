"""Canonical LoveDA DataLoader construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from torch.utils.data import DataLoader

from datasets.loveda.dataset import LoveDADataset
from datasets.loveda.manifest import LoveDAManifest
from datasets.loveda.transforms import eval_transform, train_transform
from training.data.sampler import LoveDASamplerFactory, SamplingStrategy
from training.data.statistics import LoveDAStatisticsAnalyzer


@dataclass(frozen=True)
class LoveDALoaderConfig:
    batch_size: int = 4
    num_workers: int = 0
    persistent_workers: bool = False
    pin_memory: bool = False
    drop_last: bool = False
    image_size: Optional[int] = None
    sampling_strategy: SamplingStrategy = SamplingStrategy.CLASS_BALANCED
    class_sampling_power: float = 1.0
    domain_sampling_power: float = 1.0
    min_sampling_weight: float = 1e-6
    max_sampling_weight: float = float("inf")
    seed: int = 26137

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0")
        if self.persistent_workers and self.num_workers == 0:
            raise ValueError("persistent_workers requires num_workers > 0")
        if self.class_sampling_power < 0:
            raise ValueError("class_sampling_power must be >= 0")
        if self.domain_sampling_power < 0:
            raise ValueError("domain_sampling_power must be >= 0")
        if self.image_size is not None and self.image_size <= 0:
            raise ValueError("image_size must be > 0")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")


@dataclass(frozen=True)
class LoveDALoaders:
    train: DataLoader
    val: DataLoader
    test: Optional[DataLoader] = None


def build_loveda_loaders(
    manifest: LoveDAManifest,
    *,
    config: LoveDALoaderConfig | None = None,
    include_test: bool = False,
    train_domain: str | None = None,
    val_domain: str | None = None,
    test_domain: str | None = None,
) -> LoveDALoaders:
    config = config or LoveDALoaderConfig()

    train_dataset = LoveDADataset(
        manifest=manifest,
        split="train",
        domain=train_domain,
        transform=train_transform(config.image_size),
    )
    val_dataset = LoveDADataset(
        manifest=manifest,
        split="val",
        domain=val_domain,
        transform=eval_transform(config.image_size),
    )

    statistics = LoveDAStatisticsAnalyzer(manifest).analyze(
        split="train",
        domain=train_domain,
    )

    try:
        strategy = (
            config.sampling_strategy
            if isinstance(config.sampling_strategy, SamplingStrategy)
            else SamplingStrategy(config.sampling_strategy)
        )
    except ValueError as exc:
        raise ValueError(
            f"Unknown sampling_strategy: {config.sampling_strategy!r}"
        ) from exc

    sampler_factory = LoveDASamplerFactory(
        statistics,
        strategy=strategy,
        class_power=config.class_sampling_power,
        domain_power=config.domain_sampling_power,
        min_weight=config.min_sampling_weight,
        max_weight=config.max_sampling_weight,
    )

    train_sampler = None
    shuffle = True
    if strategy is not SamplingStrategy.UNIFORM:
        generator = __import__("torch").Generator()
        generator.manual_seed(config.seed)
        train_sampler = sampler_factory.build_sampler(
            train_dataset.sample_ids,
            num_samples=len(train_dataset),
            generator=generator,
        )
        shuffle = False

    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "persistent_workers": config.persistent_workers,
    }

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        shuffle=shuffle,
        drop_last=config.drop_last,
        **common,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **common,
    )

    test_loader = None
    if include_test:
        test_dataset = LoveDADataset(
            manifest=manifest,
            split="test",
            domain=test_domain,
            transform=eval_transform(config.image_size),
            require_masks=False,
        )
        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            drop_last=False,
            **common,
        )

    return LoveDALoaders(
        train=train_loader,
        val=val_loader,
        test=test_loader,
    )
