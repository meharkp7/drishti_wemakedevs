"""Reproducible sample weighting for LoveDA training."""

from __future__ import annotations

from enum import Enum
from typing import Iterable, Sequence

import torch
from torch.utils.data import WeightedRandomSampler

from training.data.statistics import LoveDAStatistics, LoveDASampleStatistics


class SamplingError(ValueError):
    """Raised for invalid or incomplete sampling configuration."""


class SamplingStrategy(str, Enum):
    UNIFORM = "uniform"
    CLASS_BALANCED = "class_balanced"
    DOMAIN_BALANCED = "domain_balanced"
    CLASS_DOMAIN_BALANCED = "class_domain_balanced"


class LoveDASamplerFactory:
    """Build normalized sample weights and PyTorch weighted samplers."""

    def __init__(
        self,
        statistics: LoveDAStatistics,
        *,
        strategy: SamplingStrategy = SamplingStrategy.UNIFORM,
        class_power: float = 1.0,
        domain_power: float = 1.0,
        min_weight: float = 1e-6,
        max_weight: float = float("inf"),
    ) -> None:
        if not isinstance(strategy, SamplingStrategy):
            try:
                strategy = SamplingStrategy(strategy)
            except ValueError as exc:
                raise SamplingError(f"Unknown sampling strategy: {strategy!r}") from exc
        if class_power < 0:
            raise SamplingError("class_power must be >= 0")
        if domain_power < 0:
            raise SamplingError("domain_power must be >= 0")
        if min_weight <= 0:
            raise SamplingError("min_weight must be > 0")
        if max_weight <= 0 or max_weight < min_weight:
            raise SamplingError("max_weight must be >= min_weight and > 0")

        self.statistics = statistics
        self.strategy = strategy
        self.class_power = float(class_power)
        self.domain_power = float(domain_power)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)

    def build_weights(self, sample_ids: Sequence[str] | Iterable[str]) -> torch.Tensor:
        sample_ids = list(sample_ids)
        if not sample_ids:
            return torch.empty(0, dtype=torch.double)

        weights = []
        for sample_id in sample_ids:
            sample = self.statistics.sample_statistics.get(sample_id)
            if sample is None:
                raise SamplingError(f"No statistics found for sample {sample_id!r}")
            weights.append(self._raw_weight(sample))

        tensor = torch.tensor(weights, dtype=torch.double)
        tensor = tensor.clamp(min=self.min_weight, max=self.max_weight)
        mean = tensor.mean()
        if not torch.isfinite(mean) or mean <= 0:
            raise SamplingError("Sampling weights must have a finite positive mean")
        tensor = tensor / mean
        tensor = tensor.clamp(min=self.min_weight, max=self.max_weight)
        # Re-normalize after clipping so the public contract remains mean=1.
        tensor = tensor / tensor.mean()
        return tensor

    def build_sampler(
        self,
        sample_ids: Sequence[str] | Iterable[str],
        *,
        num_samples: int | None = None,
        replacement: bool = True,
        generator: torch.Generator | None = None,
    ) -> WeightedRandomSampler:
        sample_ids = list(sample_ids)
        if num_samples is None:
            num_samples = len(sample_ids)
        if num_samples < 0:
            raise SamplingError("num_samples must be >= 0")
        weights = self.build_weights(sample_ids)
        return WeightedRandomSampler(
            weights=weights,
            num_samples=int(num_samples),
            replacement=replacement,
            generator=generator,
        )

    def _raw_weight(self, sample: LoveDASampleStatistics) -> float:
        if self.strategy is SamplingStrategy.UNIFORM:
            return 1.0

        class_factor = 1.0
        if self.strategy in {
            SamplingStrategy.CLASS_BALANCED,
            SamplingStrategy.CLASS_DOMAIN_BALANCED,
        }:
            present = sample.present_classes
            if present:
                frequencies = [
                    self.statistics.class_pixel_frequencies[class_id]
                    for class_id in present
                    if self.statistics.class_pixel_frequencies[class_id] > 0
                ]
                if frequencies:
                    # A sample containing rarer classes gets more weight.
                    class_factor = (
                        sum((1.0 / f) ** self.class_power for f in frequencies)
                        / len(frequencies)
                    )

        domain_factor = 1.0
        if self.strategy in {
            SamplingStrategy.DOMAIN_BALANCED,
            SamplingStrategy.CLASS_DOMAIN_BALANCED,
        }:
            domain_count = self.statistics.by_domain.get(sample.domain, 0)
            total = max(self.statistics.total_samples, 1)
            if domain_count:
                domain_frequency = domain_count / total
                domain_factor = (1.0 / domain_frequency) ** self.domain_power

        return float(class_factor * domain_factor)
