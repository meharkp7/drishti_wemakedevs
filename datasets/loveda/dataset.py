"""
datasets/loveda/dataset.py

PyTorch-facing LoveDA dataset.

This is the only class in the ML Data Layer that training/evaluation code
should import directly. It wraps a LoveDAManifest and a paired transform and
yields DRISHTI's frozen per-sample contract (see ``to_batch_dict``), so
swapping augmentation, splits, or even swapping LoveDA for a different
dataset later never touches model training code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from datasets.loveda.manifest import (
    LoveDAManifest,
    ManifestError,
    scan_dataset,
)
from datasets.loveda.taxonomy import IGNORE_INDEX
from datasets.loveda.transforms import PairedTransform, eval_transform


class LoveDADatasetError(ValueError):
    """Raised for invalid LoveDADataset construction or sample decoding failures."""


@dataclass(frozen=True)
class LoveDASampleItem:
    """DRISHTI's frozen per-sample training/evaluation contract for LoveDA."""

    sample_id: str
    split: str
    domain: str
    image: torch.Tensor  # (3, H, W) float, normalized
    mask: Optional[torch.Tensor]  # (H, W) long, or None for unlabeled test samples

    def to_batch_dict(self) -> Dict[str, Any]:
        """
        JSON/collate-friendly representation.

        Unlabeled (test-split) samples get an all-``IGNORE_INDEX`` mask
        rather than ``None`` here, so that a DataLoader can always collate
        a batch of these into a single tensor without a custom collate_fn.
        """
        mask = self.mask
        if mask is None:
            mask = torch.full(
                self.image.shape[-2:],
                IGNORE_INDEX,
                dtype=torch.long,
            )

        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "domain": self.domain,
            "image": self.image,
            "mask": mask,
        }


class LoveDADataset(Dataset):
    """
    LoveDA dataset for a single split (optionally restricted to one domain).

    Parameters
    ----------
    root:
        Path to a staged LoveDA directory. Ignored if ``manifest`` is given.

    split:
        One of "train", "val", "test".

    domain:
        One of "urban", "rural", or None to use both domains.

    manifest:
        A pre-scanned LoveDAManifest, to avoid re-scanning the filesystem
        when constructing multiple splits/domains from the same root.

    transform:
        Paired ``(image, mask) -> (image_tensor, mask_tensor)`` callable.

    require_masks:
        If True, every matched sample must have a mask. Defaults to True for
        train/val and False for test.
    """

    def __init__(
        self,
        root: Optional[str] = None,
        *,
        split: str,
        domain: Optional[str] = None,
        manifest: Optional[LoveDAManifest] = None,
        transform: Optional[PairedTransform] = None,
        require_masks: Optional[bool] = None,
    ) -> None:
        if manifest is None:
            if root is None:
                raise LoveDADatasetError(
                    "Either `root` or `manifest` must be provided."
                )

            try:
                manifest = scan_dataset(root)
            except ManifestError as exc:
                raise LoveDADatasetError(
                    str(exc)
                ) from exc

        try:
            filtered = manifest.filter(
                split=split,
                domain=domain,
            )
        except ManifestError as exc:
            raise LoveDADatasetError(
                str(exc)
            ) from exc

        if len(filtered) == 0:
            raise LoveDADatasetError(
                f"No LoveDA samples found for split={split!r}, "
                f"domain={domain!r} under {manifest.root!r}."
            )

        if require_masks is None:
            require_masks = split != "test"

        if require_masks:
            missing = [
                sample.sample_id
                for sample in filtered.samples
                if not sample.has_mask()
            ]

            if missing:
                raise LoveDADatasetError(
                    f"{len(missing)} sample(s) in split={split!r} are "
                    f"missing masks (e.g. {missing[0]}). Pass "
                    f"require_masks=False if this is expected."
                )

        self._manifest = filtered
        self._transform = (
            transform
            or eval_transform()
        )
        self._split = split
        self._domain = domain

    def __len__(self) -> int:
        return len(self._manifest)

    @property
    def sample_ids(self) -> Tuple[str, ...]:
        """
        Stable ordered sample IDs.

        This is the public bridge between Dataset and training infrastructure.

        The order is exactly the order used by ``__getitem__`` and therefore
        can safely be used to align sampler weights with dataset indices.
        """
        return tuple(
            sample.sample_id
            for sample in self._manifest.samples
        )

    @property
    def split(self) -> str:
        """Dataset split represented by this instance."""
        return self._split

    @property
    def domain(self) -> Optional[str]:
        """Optional domain restriction represented by this instance."""
        return self._domain

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self._manifest.samples[index]

        try:
            image = (
                Image.open(
                    sample.image_path
                ).convert("RGB")
            )
        except Exception as exc:
            raise LoveDADatasetError(
                f"Failed to load image for sample "
                f"{sample.sample_id!r} at "
                f"{sample.image_path!r}: {exc}"
            ) from exc

        mask: Optional[Image.Image] = None

        if sample.has_mask():
            try:
                mask = Image.open(
                    sample.mask_path
                )

                if mask.mode != "L":
                    mask = mask.convert("L")

            except Exception as exc:
                raise LoveDADatasetError(
                    f"Failed to load mask for sample "
                    f"{sample.sample_id!r} at "
                    f"{sample.mask_path!r}: {exc}"
                ) from exc

        image_tensor, mask_tensor = self._transform(
            image,
            mask,
        )

        item = LoveDASampleItem(
            sample_id=sample.sample_id,
            split=sample.split,
            domain=sample.domain,
            image=image_tensor,
            mask=mask_tensor,
        )

        return item.to_batch_dict()