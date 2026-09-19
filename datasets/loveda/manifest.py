"""
datasets/loveda/manifest.py

Deterministic scanning + structural validation of a LoveDA dataset
directory into a typed manifest of (image, mask) sample pairs.

This is the ML Data Layer's contract boundary: everything downstream
(dataset.py, training, evaluation) consumes a LoveDAManifest, never the
filesystem directly. That keeps "where LoveDA happens to live on disk"
fully isolated from "how LoveDA is consumed."

Expected on-disk layout (the standard LoveDA release layout):

    <root>/
      Train/
        Urban/
          images_png/*.png
          masks_png/*.png
        Rural/
          images_png/*.png
          masks_png/*.png
      Val/
        Urban/...
        Rural/...
      Test/
        Urban/
          images_png/*.png        # Test has no public masks
        Rural/
          images_png/*.png

Kaggle mirrors of LoveDA sometimes rename ``images_png``/``masks_png`` to
``images``/``masks``, use lowercase split/domain folder names, nest
everything under one extra top-level ``LoveDA/`` folder, or double up each
split folder's name (``Train/Train/Urban/...`` instead of
``Train/Urban/...``). ``scan_dataset`` tolerates these by checking a short
list of known aliases and up to one extra level of nesting at both the
root and the split level, rather than assuming a single exact layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

SPLITS: Tuple[str, ...] = ("train", "val", "test")
DOMAINS: Tuple[str, ...] = ("urban", "rural")

_SPLIT_DIR_ALIASES: Dict[str, Tuple[str, ...]] = {
    "train": ("Train", "train"),
    "val": ("Val", "val", "Valid", "valid", "validation"),
    "test": ("Test", "test"),
}
_DOMAIN_DIR_ALIASES: Dict[str, Tuple[str, ...]] = {
    "urban": ("Urban", "urban"),
    "rural": ("Rural", "rural"),
}
_IMAGE_DIR_ALIASES: Tuple[str, ...] = ("images_png", "images", "img")
_MASK_DIR_ALIASES: Tuple[str, ...] = ("masks_png", "masks", "labels", "gt")
_IMAGE_SUFFIXES: Tuple[str, ...] = (".png", ".tif", ".tiff", ".jpg", ".jpeg")


class ManifestError(ValueError):
    """Raised when the LoveDA directory layout or a sample pair is invalid."""


@dataclass(frozen=True)
class LoveDASample:
    """One (image[, mask]) pair, fully identified and traceable."""

    sample_id: str
    split: str  # "train" | "val" | "test"
    domain: str  # "urban" | "rural"
    image_path: str
    mask_path: Optional[str]  # None only for the held-out test split

    def has_mask(self) -> bool:
        return self.mask_path is not None


@dataclass(frozen=True)
class ManifestStats:
    """Summary counts over a LoveDAManifest."""

    total: int
    by_split: Dict[str, int]
    by_domain: Dict[str, int]
    missing_masks: int


@dataclass(frozen=True)
class LoveDAManifest:
    """Deterministic, ordered collection of LoveDA samples."""

    root: str
    samples: Tuple[LoveDASample, ...]

    def __len__(self) -> int:
        return len(self.samples)

    def filter(
        self,
        *,
        split: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> "LoveDAManifest":
        """Return a new manifest restricted to the given split/domain."""
        if split is not None and split not in SPLITS:
            raise ManifestError(
                f"Unknown split: {split!r}. Valid splits are {SPLITS}."
            )

        if domain is not None and domain not in DOMAINS:
            raise ManifestError(
                f"Unknown domain: {domain!r}. Valid domains are {DOMAINS}."
            )

        filtered = tuple(
            sample
            for sample in self.samples
            if (split is None or sample.split == split)
            and (domain is None or sample.domain == domain)
        )

        return LoveDAManifest(root=self.root, samples=filtered)

    def stats(self) -> ManifestStats:
        by_split: Dict[str, int] = {}
        by_domain: Dict[str, int] = {}
        missing_masks = 0

        for sample in self.samples:
            by_split[sample.split] = by_split.get(sample.split, 0) + 1
            by_domain[sample.domain] = by_domain.get(sample.domain, 0) + 1
            if not sample.has_mask():
                missing_masks += 1

        return ManifestStats(
            total=len(self.samples),
            by_split=by_split,
            by_domain=by_domain,
            missing_masks=missing_masks,
        )


def _find_dir(parent: Path, aliases: Tuple[str, ...]) -> Optional[Path]:
    for alias in aliases:
        candidate = parent / alias
        if candidate.is_dir():
            return candidate
    return None


def _has_any_split_dir(candidate: Path) -> bool:
    return any(
        _find_dir(candidate, aliases) is not None
        for aliases in _SPLIT_DIR_ALIASES.values()
    )


def _has_any_domain_dir(candidate: Path) -> bool:
    return any(
        _find_dir(candidate, aliases) is not None
        for aliases in _DOMAIN_DIR_ALIASES.values()
    )


def _resolve_root(root: Path) -> Path:
    """Find the directory directly containing Train/Val/Test, tolerating
    one extra level of nesting (e.g. ``<root>/LoveDA/Train/...``)."""
    if _has_any_split_dir(root):
        return root

    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if _has_any_split_dir(child):
            return child

    raise ManifestError(
        f"Could not find LoveDA split directories (Train/Val/Test) under "
        f"{root}, including one level of nesting."
    )


def _resolve_split_dir(split_dir: Path) -> Path:
    """Find the directory directly containing Urban/Rural, tolerating one
    extra level of nesting. Some Kaggle mirrors double up the split folder
    name (e.g. ``Train/Train/Urban/...``, ``Val/Val/Rural/...``) instead of
    the plain ``Train/Urban/...`` layout the official release uses."""
    if _has_any_domain_dir(split_dir):
        return split_dir

    for child in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        if _has_any_domain_dir(child):
            return child

    raise ManifestError(
        f"Could not find domain directories (Urban/Rural) under "
        f"{split_dir}, including one level of nesting."
    )


def scan_dataset(root: str | Path) -> LoveDAManifest:
    """
    Scan a staged LoveDA directory and build a deterministic manifest.

    Every ``train``/``val`` image must have a matching mask; ``test`` may be
    mask-free (LoveDA's public test split has no released labels). A domain
    or split that is entirely absent locally (e.g. you only downloaded
    Train+Val) is skipped rather than treated as an error -- a partial local
    copy is still a valid manifest, callers just get fewer samples.
    """
    root_path = Path(root)

    if not root_path.is_dir():
        raise ManifestError(
            f"LoveDA root does not exist or is not a directory: {root_path}"
        )

    resolved_root = _resolve_root(root_path)

    samples = []

    for split in SPLITS:
        split_dir = _find_dir(resolved_root, _SPLIT_DIR_ALIASES[split])
        if split_dir is None:
            continue

        split_dir = _resolve_split_dir(split_dir)

        for domain in DOMAINS:
            domain_dir = _find_dir(split_dir, _DOMAIN_DIR_ALIASES[domain])
            if domain_dir is None:
                continue

            image_dir = _find_dir(domain_dir, _IMAGE_DIR_ALIASES)
            if image_dir is None:
                raise ManifestError(
                    f"No images directory found under {domain_dir} "
                    f"(looked for {_IMAGE_DIR_ALIASES})."
                )

            mask_dir = _find_dir(domain_dir, _MASK_DIR_ALIASES)

            image_paths = sorted(
                path
                for path in image_dir.iterdir()
                if path.suffix.lower() in _IMAGE_SUFFIXES
            )

            if not image_paths:
                raise ManifestError(f"No image files found in {image_dir}.")

            for image_path in image_paths:
                stem = image_path.stem
                mask_path: Optional[Path] = None

                if mask_dir is not None:
                    exact = mask_dir / f"{stem}{image_path.suffix}"
                    if exact.exists():
                        mask_path = exact
                    else:
                        # Tolerate a differing extension between image/mask
                        # (some mirrors ship .tif images with .png masks).
                        matches = sorted(mask_dir.glob(f"{stem}.*"))
                        mask_path = matches[0] if matches else None

                if mask_path is None and split != "test":
                    raise ManifestError(
                        f"Image {image_path} has no matching mask under "
                        f"{mask_dir!r} -- split {split!r} is expected to be "
                        f"fully labeled. Only 'test' may be mask-free."
                    )

                samples.append(
                    LoveDASample(
                        sample_id=f"{split}-{domain}-{stem}",
                        split=split,
                        domain=domain,
                        image_path=str(image_path),
                        mask_path=(
                            str(mask_path) if mask_path is not None else None
                        ),
                    )
                )

    if not samples:
        raise ManifestError(f"No LoveDA samples found under {resolved_root}.")

    return LoveDAManifest(root=str(resolved_root), samples=tuple(samples))