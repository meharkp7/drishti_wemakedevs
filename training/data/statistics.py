"""
training/data/statistics.py

Dataset-level and sample-level statistics for semantic segmentation.

The statistics layer is intentionally independent from the model, loss, and
sampler. It consumes the already-validated LoveDA manifest and reads masks
directly from disk so that:

    dataset statistics
        -> sampling weights
        -> class weighting
        -> experiment reports

all originate from the same deterministic source.

Important LoveDA convention:

    0 = IGNORE_INDEX
    1 = background
    2 = building
    3 = road
    4 = water
    5 = barren
    6 = forest
    7 = agriculture

Ignore pixels are excluded from all class statistics.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
from PIL import Image

from datasets.loveda.manifest import LoveDAManifest, LoveDASample
from datasets.loveda.taxonomy import (
    IGNORE_INDEX,
    LOVEDA_CLASSES,
    NUM_CLASSES,
)


class LoveDAStatisticsError(ValueError):
    """Raised when dataset statistics cannot be computed safely."""


@dataclass(frozen=True)
class SampleStatistics:
    """Statistics for one labeled LoveDA sample."""

    sample_id: str
    split: str
    domain: str

    total_pixels: int
    ignored_pixels: int
    valid_pixels: int

    class_pixel_counts: Dict[int, int]

    @property
    def class_presence(self) -> Tuple[int, ...]:
        """Return semantic class IDs appearing in the sample."""
        return tuple(
            class_id
            for class_id, count in self.class_pixel_counts.items()
            if count > 0
        )

    @property
    def valid_fraction(self) -> float:
        """Fraction of pixels participating in supervised learning."""
        if self.total_pixels == 0:
            return 0.0
        return self.valid_pixels / self.total_pixels


@dataclass(frozen=True)
class LoveDAStatistics:
    """
    Immutable dataset statistics snapshot.

    All class dictionaries use LoveDA's external class IDs (1..7), not model
    indices (0..6).
    """

    root: str
    total_samples: int
    labeled_samples: int

    by_split: Dict[str, int]
    by_domain: Dict[str, int]

    total_pixels: int
    ignored_pixels: int
    valid_pixels: int

    class_pixel_counts: Dict[int, int]
    class_sample_counts: Dict[int, int]

    sample_statistics: Dict[str, SampleStatistics]

    @property
    def class_pixel_frequencies(self) -> Dict[int, float]:
        """Fraction of valid pixels belonging to each class."""
        if self.valid_pixels == 0:
            return {
                class_id: 0.0
                for class_id in self.class_pixel_counts
            }

        return {
            class_id: count / self.valid_pixels
            for class_id, count in self.class_pixel_counts.items()
        }

    @property
    def class_sample_frequencies(self) -> Dict[int, float]:
        """Fraction of labeled samples containing each class."""
        if self.labeled_samples == 0:
            return {
                class_id: 0.0
                for class_id in self.class_sample_counts
            }

        return {
            class_id: count / self.labeled_samples
            for class_id, count in self.class_sample_counts.items()
        }

    def to_dict(self) -> dict:
        """Return a JSON-serializable statistics report."""
        return {
            "root": self.root,
            "total_samples": self.total_samples,
            "labeled_samples": self.labeled_samples,
            "by_split": dict(self.by_split),
            "by_domain": dict(self.by_domain),
            "total_pixels": self.total_pixels,
            "ignored_pixels": self.ignored_pixels,
            "valid_pixels": self.valid_pixels,
            "class_pixel_counts": dict(self.class_pixel_counts),
            "class_pixel_frequencies": self.class_pixel_frequencies,
            "class_sample_counts": dict(self.class_sample_counts),
            "class_sample_frequencies": self.class_sample_frequencies,
        }


class LoveDAStatisticsAnalyzer:
    """
    Compute deterministic statistics from a LoveDA manifest.

    Parameters
    ----------
    manifest:
        Already-scanned LoveDAManifest.

    cache:
        Optional in-memory cache. The analyzer itself does not write cache
        files; persistence belongs to the experiment/reporting layer.
    """

    def __init__(
        self,
        manifest: LoveDAManifest,
    ) -> None:
        if not isinstance(manifest, LoveDAManifest):
            raise LoveDAStatisticsError(
                "manifest must be a LoveDAManifest."
            )

        self.manifest = manifest

    def analyze(
        self,
        *,
        split: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> LoveDAStatistics:
        """
        Analyze the requested manifest subset.

        Test samples without masks are excluded from pixel-level statistics,
        but remain represented in total sample counts.
        """
        subset = self.manifest.filter(
            split=split,
            domain=domain,
        )

        by_split: Counter[str] = Counter()
        by_domain: Counter[str] = Counter()

        class_pixel_counts = {
            cls.id: 0
            for cls in LOVEDA_CLASSES
        }

        class_sample_counts = {
            cls.id: 0
            for cls in LOVEDA_CLASSES
        }

        sample_statistics: Dict[str, SampleStatistics] = {}

        total_pixels = 0
        ignored_pixels = 0
        valid_pixels = 0
        labeled_samples = 0

        for sample in subset.samples:
            by_split[sample.split] += 1
            by_domain[sample.domain] += 1

            if not sample.has_mask():
                continue

            stats = self._analyze_sample(sample)

            sample_statistics[sample.sample_id] = stats
            labeled_samples += 1

            total_pixels += stats.total_pixels
            ignored_pixels += stats.ignored_pixels
            valid_pixels += stats.valid_pixels

            for class_id, count in stats.class_pixel_counts.items():
                class_pixel_counts[class_id] += count

                if count > 0:
                    class_sample_counts[class_id] += 1

        return LoveDAStatistics(
            root=subset.root,
            total_samples=len(subset),
            labeled_samples=labeled_samples,
            by_split=dict(by_split),
            by_domain=dict(by_domain),
            total_pixels=total_pixels,
            ignored_pixels=ignored_pixels,
            valid_pixels=valid_pixels,
            class_pixel_counts=class_pixel_counts,
            class_sample_counts=class_sample_counts,
            sample_statistics=sample_statistics,
        )

    def _analyze_sample(
        self,
        sample: LoveDASample,
    ) -> SampleStatistics:
        if not sample.mask_path:
            raise LoveDAStatisticsError(
                f"Sample {sample.sample_id!r} has no mask."
            )

        mask_path = Path(sample.mask_path)

        try:
            with Image.open(mask_path) as mask:
                mask_array = np.asarray(
                    mask.convert("L"),
                    dtype=np.int64,
                )
        except Exception as exc:
            raise LoveDAStatisticsError(
                f"Failed to read mask for sample "
                f"{sample.sample_id!r} at {mask_path}: {exc}"
            ) from exc

        if mask_array.ndim != 2:
            raise LoveDAStatisticsError(
                f"Mask for sample {sample.sample_id!r} must be 2D; "
                f"got shape {mask_array.shape}."
            )

        total = int(mask_array.size)

        ignored = int(
            np.count_nonzero(
                mask_array == IGNORE_INDEX
            )
        )

        valid = total - ignored

        class_counts: Dict[int, int] = {}

        valid_values = mask_array[
            mask_array != IGNORE_INDEX
        ]

        if valid_values.size:
            unique, counts = np.unique(
                valid_values,
                return_counts=True,
            )

            for class_id, count in zip(
                unique.tolist(),
                counts.tolist(),
            ):
                if class_id not in {
                    cls.id for cls in LOVEDA_CLASSES
                }:
                    raise LoveDAStatisticsError(
                        f"Sample {sample.sample_id!r} contains "
                        f"unknown LoveDA mask value {class_id}."
                    )

                class_counts[int(class_id)] = int(count)

        # Keep every semantic class explicitly represented.
        class_counts = {
            class_id: class_counts.get(class_id, 0)
            for class_id in range(1, NUM_CLASSES + 1)
        }

        return SampleStatistics(
            sample_id=sample.sample_id,
            split=sample.split,
            domain=sample.domain,
            total_pixels=total,
            ignored_pixels=ignored,
            valid_pixels=valid,
            class_pixel_counts=class_counts,
        )


def analyze_loveda(
    manifest: LoveDAManifest,
    *,
    split: Optional[str] = None,
    domain: Optional[str] = None,
) -> LoveDAStatistics:
    """Convenience wrapper around LoveDAStatisticsAnalyzer."""
    return LoveDAStatisticsAnalyzer(manifest).analyze(
        split=split,
        domain=domain,
    )