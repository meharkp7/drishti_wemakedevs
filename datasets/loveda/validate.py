"""
datasets/loveda/validate.py

Dataset integrity validation for a scanned LoveDA manifest.

Run this once after downloading/staging LoveDA, and before pointing any
training or evaluation job at it:

    python -m datasets.loveda.validate --root data/loveda

This is deliberately NOT run automatically inside
``LoveDADataset.__getitem__`` -- per-sample validation on every epoch would
be needlessly slow. This is a one-time (or CI) gate, matching the QC-vs-
inference split used elsewhere in DRISHTI's pipeline.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from datasets.loveda.manifest import LoveDAManifest, scan_dataset
from datasets.loveda.taxonomy import IGNORE_INDEX, is_valid_mask_value

_MAX_ISSUES_SHOWN = 20


@dataclass(frozen=True)
class ValidationIssue:
    sample_id: str
    problem: str


@dataclass
class ValidationReport:
    total_samples: int
    checked_samples: int
    issues: List[ValidationIssue] = field(default_factory=list)
    class_pixel_counts: Dict[int, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.issues

    def summary(self) -> str:
        lines = [
            f"Checked {self.checked_samples}/{self.total_samples} samples.",
            f"Issues found: {len(self.issues)}",
        ]

        if self.class_pixel_counts:
            total_px = sum(self.class_pixel_counts.values())
            lines.append("Class pixel distribution (excludes ignore/0):")
            for class_id in sorted(self.class_pixel_counts):
                count = self.class_pixel_counts[class_id]
                pct = 100.0 * count / total_px if total_px else 0.0
                lines.append(f"  class {class_id}: {count} px ({pct:.2f}%)")

        for issue in self.issues[:_MAX_ISSUES_SHOWN]:
            lines.append(f"  [{issue.sample_id}] {issue.problem}")

        if len(self.issues) > _MAX_ISSUES_SHOWN:
            lines.append(f"  ... and {len(self.issues) - _MAX_ISSUES_SHOWN} more")

        return "\n".join(lines)


def validate_manifest(
    manifest: LoveDAManifest,
    *,
    max_samples: Optional[int] = None,
    check_pixel_values: bool = True,
) -> ValidationReport:
    """
    Check every (or up to ``max_samples``) sample for:
      - readable image/mask files
      - matching image/mask pixel dimensions
      - mask pixel values restricted to {0} union {1..NUM_CLASSES}

    Also accumulates a class pixel histogram as a side effect, which is
    useful for spotting class imbalance before training.
    """
    samples = manifest.samples
    if max_samples is not None:
        samples = samples[:max_samples]

    issues: List[ValidationIssue] = []
    class_pixel_counts: Dict[int, int] = {}
    checked = 0

    for sample in samples:
        checked += 1

        try:
            with Image.open(sample.image_path) as img:
                img.verify()
            with Image.open(sample.image_path) as img:
                image_size = img.size
        except Exception as exc:
            issues.append(
                ValidationIssue(sample.sample_id, f"unreadable image: {exc}")
            )
            continue

        if not sample.has_mask():
            continue

        try:
            with Image.open(sample.mask_path) as mask_img:
                mask_img.verify()
        except Exception as exc:
            issues.append(
                ValidationIssue(sample.sample_id, f"unreadable mask: {exc}")
            )
            continue

        with Image.open(sample.mask_path) as mask_img:
            mask_size = mask_img.size
            mask_arr = (
                np.array(mask_img.convert("L")) if check_pixel_values else None
            )

        if mask_size != image_size:
            issues.append(
                ValidationIssue(
                    sample.sample_id,
                    f"image/mask size mismatch: {image_size} vs {mask_size}",
                )
            )

        if check_pixel_values and mask_arr is not None:
            unique_values = np.unique(mask_arr)
            invalid = [
                int(v) for v in unique_values if not is_valid_mask_value(int(v))
            ]
            if invalid:
                issues.append(
                    ValidationIssue(
                        sample.sample_id,
                        f"invalid mask pixel values: {invalid}",
                    )
                )

            for value in unique_values:
                class_id = int(value)
                if class_id == IGNORE_INDEX:
                    continue
                class_pixel_counts[class_id] = class_pixel_counts.get(
                    class_id, 0
                ) + int((mask_arr == class_id).sum())

    return ValidationReport(
        total_samples=len(manifest.samples),
        checked_samples=checked,
        issues=issues,
        class_pixel_counts=class_pixel_counts,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a staged LoveDA dataset directory."
    )
    parser.add_argument("--root", required=True, help="Path to staged LoveDA.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap samples checked (omit to check the whole manifest).",
    )
    parser.add_argument(
        "--skip-pixel-check",
        action="store_true",
        help="Skip the (slower) per-pixel mask value / histogram check.",
    )
    args = parser.parse_args()

    manifest = scan_dataset(args.root)
    stats = manifest.stats()

    print(f"Scanned manifest: {len(manifest)} samples.")
    print(f"By split:  {stats.by_split}")
    print(f"By domain: {stats.by_domain}")
    print(f"Missing masks (expected only for 'test'): {stats.missing_masks}")
    print()

    report = validate_manifest(
        manifest,
        max_samples=args.max_samples,
        check_pixel_values=not args.skip_pixel_check,
    )
    print(report.summary())

    if not report.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
