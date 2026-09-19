"""Deterministic statistics over a staged LoveDA manifest.

The statistics layer deliberately consumes the manifest rather than walking the
filesystem itself.  This keeps sample identity and dataset structure in one
place and makes sampling reproducible from a fixed manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

from datasets.loveda.manifest import LoveDAManifest
from datasets.loveda.taxonomy import IGNORE_INDEX, LOVEDA_CLASSES


class LoveDAStatisticsError(ValueError):
    """Raised when LoveDA statistics cannot be computed safely."""


@dataclass(frozen=True)
class LoveDASampleStatistics:
    """Statistics for one LoveDA sample."""

    sample_id: str
    split: str
    domain: str
    total_pixels: int
    valid_pixels: int
    ignored_pixels: int
    class_pixel_counts: Dict[int, int]
    present_classes: Tuple[int, ...]


@dataclass(frozen=True)
class LoveDAStatistics:
    """Immutable aggregate report used by samplers and experiment logging."""

    split: Optional[str]
    domain: Optional[str]
    total_samples: int
    labeled_samples: int
    total_pixels: int
    valid_pixels: int
    ignored_pixels: int
    class_pixel_counts: Dict[int, int]
    class_pixel_frequencies: Dict[int, float]
    class_sample_counts: Dict[int, int]
    by_split: Dict[str, int]
    by_domain: Dict[str, int]
    sample_statistics: Dict[str, LoveDASampleStatistics]

    def to_dict(self) -> dict:
        return {
            "split": self.split,
            "domain": self.domain,
            "total_samples": self.total_samples,
            "labeled_samples": self.labeled_samples,
            "total_pixels": self.total_pixels,
            "valid_pixels": self.valid_pixels,
            "ignored_pixels": self.ignored_pixels,
            "class_pixel_counts": dict(self.class_pixel_counts),
            "class_pixel_frequencies": dict(self.class_pixel_frequencies),
            "class_sample_counts": dict(self.class_sample_counts),
            "by_split": dict(self.by_split),
            "by_domain": dict(self.by_domain),
            "sample_statistics": {
                sample_id: {
                    "sample_id": sample.sample_id,
                    "split": sample.split,
                    "domain": sample.domain,
                    "total_pixels": sample.total_pixels,
                    "valid_pixels": sample.valid_pixels,
                    "ignored_pixels": sample.ignored_pixels,
                    "class_pixel_counts": dict(sample.class_pixel_counts),
                    "present_classes": list(sample.present_classes),
                }
                for sample_id, sample in self.sample_statistics.items()
            },
        }


class LoveDAStatisticsAnalyzer:
    """Compute validated pixel/class statistics for a manifest subset."""

    def __init__(self, manifest: LoveDAManifest) -> None:
        self.manifest = manifest

    def analyze(
        self,
        *,
        split: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> LoveDAStatistics:
        try:
            filtered = self.manifest.filter(split=split, domain=domain)
        except Exception as exc:
            raise LoveDAStatisticsError(str(exc)) from exc

        sample_stats: Dict[str, LoveDASampleStatistics] = {}
        class_pixel_counts = {cls.id: 0 for cls in LOVEDA_CLASSES}
        class_sample_counts = {cls.id: 0 for cls in LOVEDA_CLASSES}
        by_split: Dict[str, int] = {}
        by_domain: Dict[str, int] = {}
        total_pixels = valid_pixels = ignored_pixels = 0
        labeled_samples = 0

        for sample in filtered.samples:
            by_split[sample.split] = by_split.get(sample.split, 0) + 1
            by_domain[sample.domain] = by_domain.get(sample.domain, 0) + 1

            if sample.mask_path is None:
                # The public LoveDA test split is unlabeled. It is useful for
                # manifest construction but cannot contribute supervised stats.
                sample_stats[sample.sample_id] = LoveDASampleStatistics(
                    sample_id=sample.sample_id,
                    split=sample.split,
                    domain=sample.domain,
                    total_pixels=0,
                    valid_pixels=0,
                    ignored_pixels=0,
                    class_pixel_counts={cls.id: 0 for cls in LOVEDA_CLASSES},
                    present_classes=(),
                )
                continue

            labeled_samples += 1
            mask_path = Path(sample.mask_path)
            try:
                with Image.open(mask_path) as mask_image:
                    values = np.asarray(mask_image.convert("L"), dtype=np.int64)
            except Exception as exc:
                raise LoveDAStatisticsError(
                    f"Failed to read mask for sample {sample.sample_id!r}: {exc}"
                ) from exc

            sample_total = int(values.size)
            sample_ignored = int(np.count_nonzero(values == IGNORE_INDEX))
            unknown = sorted(
                int(value)
                for value in np.unique(values)
                if int(value) != IGNORE_INDEX
                and int(value) not in class_pixel_counts
            )
            if unknown:
                raise LoveDAStatisticsError(
                    f"unknown LoveDA mask value(s) {unknown} in sample "
                    f"{sample.sample_id!r}"
                )

            sample_counts = {
                class_id: int(np.count_nonzero(values == class_id))
                for class_id in class_pixel_counts
            }
            present = tuple(
                class_id for class_id, count in sample_counts.items() if count > 0
            )
            sample_valid = sample_total - sample_ignored

            for class_id, count in sample_counts.items():
                class_pixel_counts[class_id] += count
                if count > 0:
                    class_sample_counts[class_id] += 1

            total_pixels += sample_total
            ignored_pixels += sample_ignored
            valid_pixels += sample_valid

            sample_stats[sample.sample_id] = LoveDASampleStatistics(
                sample_id=sample.sample_id,
                split=sample.split,
                domain=sample.domain,
                total_pixels=sample_total,
                valid_pixels=sample_valid,
                ignored_pixels=sample_ignored,
                class_pixel_counts=sample_counts,
                present_classes=present,
            )

        if valid_pixels:
            frequencies = {
                class_id: count / valid_pixels
                for class_id, count in class_pixel_counts.items()
            }
        else:
            frequencies = {class_id: 0.0 for class_id in class_pixel_counts}

        return LoveDAStatistics(
            split=split,
            domain=domain,
            total_samples=len(filtered.samples),
            labeled_samples=labeled_samples,
            total_pixels=total_pixels,
            valid_pixels=valid_pixels,
            ignored_pixels=ignored_pixels,
            class_pixel_counts=class_pixel_counts,
            class_pixel_frequencies=frequencies,
            class_sample_counts=class_sample_counts,
            by_split=by_split,
            by_domain=by_domain,
            sample_statistics=sample_stats,
        )
