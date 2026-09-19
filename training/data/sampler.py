"""
training/data/sampler.py

Sampling strategies for LoveDA semantic segmentation.

The sampler operates on already-computed per-sample statistics.

Supported strategies:

    uniform
        Every sample receives equal probability.

    class_balanced
        Rare semantic classes increase the sampling probability of samples
        containing those classes.

    domain_balanced
        Urban/rural domains are balanced at the sample level.

    class_domain_balanced
        Combines class rarity and domain rarity.

The implementation deliberately avoids modifying LoveDADataset itself.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Optional, Sequence

import torch
from torch.utils.data import Sampler, WeightedRandomSampler

from training.data.statistics import LoveDAStatistics


class SamplingError(ValueError):
    """Raised when sampling configuration is invalid."""


class SamplingStrategy(str, Enum):
    """Supported LoveDA sampling strategies."""

    UNIFORM = "uniform"
    CLASS_BALANCED = "class_balanced"
    DOMAIN_BALANCED = "domain_balanced"
    CLASS_DOMAIN_BALANCED = "class_domain_balanced"


class LoveDASamplerFactory:
    """
    Build deterministic sample weights from LoveDA statistics.

    Parameters
    ----------
    statistics:
        Statistics computed from the same manifest/subset used by the dataset.

    strategy:
        Sampling strategy.

    class_power:
        Controls how aggressively rare classes influence sample weights.

        0.0 -> no class effect
        0.5 -> square-root rarity
        1.0 -> full inverse-frequency effect

    domain_power:
        Equivalent control for urban/rural balancing.

    min_weight / max_weight:
        Hard bounds preventing pathological oversampling.
    """

    def __init__(
        self,
        statistics: LoveDAStatistics,
        *,
        strategy: SamplingStrategy = SamplingStrategy.UNIFORM,
        class_power: float = 0.5,
        domain_power: float = 1.0,
        min_weight: float = 0.25,
        max_weight: float = 4.0,
    ) -> None:
        if not isinstance(
            statistics,
            LoveDAStatistics,
        ):
            raise SamplingError(
                "statistics must be LoveDAStatistics."
            )

        if class_power < 0:
            raise SamplingError(
                "class_power must be >= 0."
            )

        if domain_power < 0:
            raise SamplingError(
                "domain_power must be >= 0."
            )

        if min_weight <= 0:
            raise SamplingError(
                "min_weight must be > 0."
            )

        if max_weight < min_weight:
            raise SamplingError(
                "max_weight must be >= min_weight."
            )

        self.statistics = statistics
        self.strategy = SamplingStrategy(strategy)
        self.class_power = float(class_power)
        self.domain_power = float(domain_power)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)

        self._class_weights = (
            self._build_class_weights()
        )
        self._domain_weights = (
            self._build_domain_weights()
        )

    def build_weights(
        self,
        sample_ids: Sequence[str],
    ) -> torch.DoubleTensor:
        """
        Return one positive sampling weight per sample ID.

        The order exactly matches ``sample_ids``.
        """
        weights = []

        for sample_id in sample_ids:
            if sample_id not in self.statistics.sample_statistics:
                raise SamplingError(
                    f"No statistics found for sample "
                    f"{sample_id!r}."
                )

            stats = (
                self.statistics.sample_statistics[
                    sample_id
                ]
            )

            weight = 1.0

            if self.strategy in (
                SamplingStrategy.CLASS_BALANCED,
                SamplingStrategy.CLASS_DOMAIN_BALANCED,
            ):
                weight *= self._class_sample_weight(
                    stats.class_pixel_counts
                )

            if self.strategy in (
                SamplingStrategy.DOMAIN_BALANCED,
                SamplingStrategy.CLASS_DOMAIN_BALANCED,
            ):
                weight *= self._domain_weights[
                    stats.domain
                ]

            weight = min(
                self.max_weight,
                max(self.min_weight, weight),
            )

            weights.append(weight)

        tensor = torch.tensor(
            weights,
            dtype=torch.double,
        )

        if not torch.all(
            torch.isfinite(tensor)
        ):
            raise SamplingError(
                "Computed sampling weights contain "
                "non-finite values."
            )

        if torch.any(tensor <= 0):
            raise SamplingError(
                "All sampling weights must be positive."
            )

        # Normalize around mean=1. This keeps the numerical scale stable
        # while preserving relative probabilities.
        tensor /= tensor.mean()

        return tensor

    def build_sampler(
        self,
        sample_ids: Sequence[str],
        *,
        num_samples: Optional[int] = None,
        replacement: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> WeightedRandomSampler:
        """
        Construct a PyTorch WeightedRandomSampler.

        ``replacement=True`` is recommended for class-balanced sampling,
        because otherwise the sampler cannot guarantee sufficient exposure
        to rare classes.
        """
        if not sample_ids:
            raise SamplingError(
                "sample_ids must not be empty."
            )

        if num_samples is None:
            num_samples = len(sample_ids)

        if num_samples <= 0:
            raise SamplingError(
                "num_samples must be > 0."
            )

        if not replacement and self.strategy != SamplingStrategy.UNIFORM:
            raise SamplingError(
                "Non-replacement sampling is not supported for "
                "non-uniform strategies."
            )

        weights = self.build_weights(
            sample_ids
        )

        return WeightedRandomSampler(
            weights=weights,
            num_samples=num_samples,
            replacement=replacement,
            generator=generator,
        )

    def _build_class_weights(
        self,
    ) -> Dict[int, float]:
        """
        Compute inverse sample-frequency class weights.

        Sample frequency is used instead of raw pixel frequency because
        sampling acts on images, not individual pixels.
        """
        frequencies = (
            self.statistics.class_sample_frequencies
        )

        positive = [
            value
            for value in frequencies.values()
            if value > 0
        ]

        if not positive:
            raise SamplingError(
                "No semantic classes were observed."
            )

        reference = max(positive)

        weights = {}

        for class_id, frequency in frequencies.items():
            if frequency <= 0:
                weights[class_id] = 1.0
            else:
                rarity = reference / frequency
                weights[class_id] = (
                    rarity ** self.class_power
                )

        return weights

    def _build_domain_weights(
        self,
    ) -> Dict[str, float]:
        """Compute inverse-frequency urban/rural sample weights."""
        counts = self.statistics.by_domain

        total = sum(counts.values())

        if total <= 0:
            raise SamplingError(
                "No samples available for domain balancing."
            )

        frequencies = {
            domain: count / total
            for domain, count in counts.items()
            if count > 0
        }

        reference = max(
            frequencies.values()
        )

        return {
            domain: (
                (reference / frequency)
                ** self.domain_power
            )
            for domain, frequency
            in frequencies.items()
        }

    def _class_sample_weight(
        self,
        class_counts: Dict[int, int],
    ) -> float:
        """
        Aggregate class rarity for one sample.

        A sample's class weight is the mean rarity of the semantic classes
        actually present. This avoids allowing one tiny rare-class pixel to
        completely dominate the sampling distribution.
        """
        present = [
            class_id
            for class_id, count
            in class_counts.items()
            if count > 0
        ]

        if not present:
            return 1.0

        weights = [
            self._class_weights[class_id]
            for class_id in present
        ]

        return float(
            sum(weights) / len(weights)
        )