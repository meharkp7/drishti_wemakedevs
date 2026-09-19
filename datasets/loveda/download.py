"""
datasets/loveda/download.py

Fetches a LoveDA mirror from Kaggle via ``kagglehub`` and stages it where
``datasets.loveda.manifest.scan_dataset`` expects it.

Requires:
    pip install kagglehub
    A Kaggle API token configured -- either ``~/.kaggle/kaggle.json``, or
    the ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` environment variables.

Usage:
    python -m datasets.loveda.download --dataset <owner>/<dataset-slug> --dest data/loveda
    python -m datasets.loveda.validate --root data/loveda

We deliberately do NOT hardcode a Kaggle dataset slug here. Several
independent Kaggle mirrors of LoveDA exist under different slugs, and the
original LoveDA release (RSIDEA, Wuhan University) is CC BY-NC-SA /
academic-use-only -- pick the mirror whose license you're actually
operating under and pass its slug explicitly. Search
https://www.kaggle.com/datasets?search=loveda to find one, then verify its
internal folder layout matches (or is close to) what
``datasets/loveda/manifest.py`` documents; ``scan_dataset`` tolerates a few
common naming variations but not arbitrary ones.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


class LoveDADownloadError(RuntimeError):
    """Raised when the Kaggle download or staging step fails."""


def download(dataset: str, dest: str) -> Path:
    """Download ``dataset`` (a Kaggle 'owner/slug' identifier) into ``dest``."""
    try:
        import kagglehub
    except ImportError as exc:
        raise LoveDADownloadError(
            "kagglehub is required for this download helper. Install it "
            "with `pip install kagglehub` and configure a Kaggle API token "
            "first (see this module's docstring)."
        ) from exc

    print(f"Downloading Kaggle dataset '{dataset}' via kagglehub ...")

    try:
        cache_path = Path(kagglehub.dataset_download(dataset))
    except Exception as exc:
        raise LoveDADownloadError(
            f"kagglehub failed to download '{dataset}': {exc}"
        ) from exc

    print(f"kagglehub cached the dataset at: {cache_path}")

    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    for item in sorted(cache_path.iterdir()):
        target = dest_path / item.name
        if target.exists():
            print(f"Skipping {item.name} (already exists at {target}).")
            continue
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)

    print(f"LoveDA staged at: {dest_path}")
    print(f"Verify the layout with: python -m datasets.loveda.validate --root {dest_path}")
    return dest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a LoveDA mirror from Kaggle and stage it locally."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Kaggle dataset identifier, e.g. 'owner-name/loveda-dataset'.",
    )
    parser.add_argument(
        "--dest",
        default="data/loveda",
        help="Local directory to stage the dataset in (default: data/loveda).",
    )
    args = parser.parse_args()
    download(args.dataset, args.dest)


if __name__ == "__main__":
    main()
