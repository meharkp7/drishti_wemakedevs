"""
datasets/loveda/taxonomy.py

Canonical LoveDA land-cover class taxonomy.

Source of truth: Wang et al., "LoveDA: A Remote Sensing Land-Cover Dataset
for Domain Adaptive Semantic Segmentation" (NeurIPS 2021 Datasets &
Benchmarks track) and the RSIDEA / Wuhan University release notes.

Mask pixel value -> class:

    0   ignore / no-data   (excluded from loss and from every metric)
    1   background
    2   building
    3   road
    4   water
    5   barren
    6   forest
    7   agriculture

This is DRISHTI-internal and scoped to "whatever LoveDA labels mean" --
it is deliberately NOT the same thing as DRISHTI's eventual production
land-cover taxonomy (that is a Phase-3+ decision). LoveDA is a training/
evaluation data source feeding the ML Data Layer, not the product schema
that the API or the review queue speak.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


class LoveDATaxonomyError(ValueError):
    """Raised when a LoveDA class id/name lookup or mask value is invalid."""


@dataclass(frozen=True)
class LoveDAClass:
    """One labeled LoveDA class."""

    id: int
    name: str
    color: Tuple[int, int, int]  # RGB, for mask visualization only


# Pixel value 0 means "no-data" and must be excluded from loss and metrics,
# never treated as a real class -- this is the single most common LoveDA
# training bug and the reason it has its own named constant rather than a
# bare literal scattered through training/eval code.
IGNORE_INDEX = 0

LOVEDA_CLASSES: Tuple[LoveDAClass, ...] = (
    LoveDAClass(1, "background", (255, 255, 255)),
    LoveDAClass(2, "building", (255, 0, 0)),
    LoveDAClass(3, "road", (255, 255, 0)),
    LoveDAClass(4, "water", (0, 0, 255)),
    LoveDAClass(5, "barren", (159, 129, 183)),
    LoveDAClass(6, "forest", (0, 255, 0)),
    LoveDAClass(7, "agriculture", (255, 195, 128)),
)

# 7 labeled classes. Index 0 (IGNORE_INDEX) is not a class and is not
# counted here -- model output heads should have NUM_CLASSES channels,
# with ignore handled via the loss's ignore_index, not an 8th channel.
NUM_CLASSES = len(LOVEDA_CLASSES)

_BY_ID: Dict[int, LoveDAClass] = {cls.id: cls for cls in LOVEDA_CLASSES}
_BY_NAME: Dict[str, LoveDAClass] = {cls.name: cls for cls in LOVEDA_CLASSES}


def class_by_id(class_id: int) -> LoveDAClass:
    """Look up a LoveDA class by its mask pixel value (1-7)."""
    try:
        return _BY_ID[class_id]
    except KeyError as exc:
        raise LoveDATaxonomyError(
            f"Unknown LoveDA class id: {class_id!r}. "
            f"Valid ids are {sorted(_BY_ID)} (0 is IGNORE_INDEX, not a class)."
        ) from exc


def class_by_name(name: str) -> LoveDAClass:
    """Look up a LoveDA class by name (e.g. 'building')."""
    try:
        return _BY_NAME[name]
    except KeyError as exc:
        raise LoveDATaxonomyError(
            f"Unknown LoveDA class name: {name!r}. "
            f"Valid names are {sorted(_BY_NAME)}."
        ) from exc


def is_valid_mask_value(value: int) -> bool:
    """True if ``value`` is a legal LoveDA mask pixel value (0 or 1-7)."""
    return value == IGNORE_INDEX or value in _BY_ID


def class_names(*, include_ignore: bool = False) -> List[str]:
    """Ordered class names, id 1 first. Optionally prefix with 'ignore'."""
    names = [cls.name for cls in LOVEDA_CLASSES]
    return (["ignore"] + names) if include_ignore else names
