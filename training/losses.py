"""
Production-grade semantic segmentation losses for DRISHTI.

LoveDA external labels
----------------------
0 = ignore
1..7 = semantic classes

Model logits
------------
channel 0..6 correspond to external classes 1..7.

This module centralizes:
- label remapping
- class weighting
- CE / Weighted CE
- Dice
- CE + Dice
- Focal
- Focal + Dice
- Tversky
- Tversky + Focal
- validated factory APIs
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ============================================================
# Exceptions
# ============================================================


class LossError(ValueError):
    """Raised when loss configuration or tensor contracts are invalid."""


# ============================================================
# Enumerations
# ============================================================


class LossName(str, Enum):
    CROSS_ENTROPY = "cross_entropy"
    WEIGHTED_CROSS_ENTROPY = "weighted_cross_entropy"
    DICE = "dice"
    CE_DICE = "ce_dice"
    FOCAL = "focal"
    FOCAL_DICE = "focal_dice"
    TVERSKY = "tversky"
    TVERSKY_FOCAL = "tversky_focal"


class ClassWeightingStrategy(str, Enum):
    NONE = "none"
    INVERSE_FREQUENCY = "inverse_frequency"
    MEDIAN_FREQUENCY = "median_frequency"
    EFFECTIVE_NUMBER = "effective_number"


# ============================================================
# Configuration
# ============================================================


@dataclass(frozen=True)
class LossConfig:
    """Immutable validated segmentation loss configuration."""

    name: str = LossName.CE_DICE.value

    class_ids: tuple[int, ...] = (
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    )

    ignore_index: int = 0
    internal_ignore_index: int = -100

    class_weighting: str = ClassWeightingStrategy.NONE.value

    gamma: float = 2.0

    tversky_alpha: float = 0.5
    tversky_beta: float = 0.5

    smooth: float = 1.0

    ce_weight: float = 1.0
    dice_weight: float = 1.0
    focal_weight: float = 1.0
    tversky_weight: float = 1.0

    effective_number_beta: float = 0.9999

    def __post_init__(self):

        ids = tuple(self.class_ids)

        if len(ids) == 0:
            raise LossError("class_ids must not be empty.")

        if len(set(ids)) != len(ids):
            raise LossError("class_ids must be unique.")

        if any(
            not isinstance(i, int)
            or isinstance(i, bool)
            for i in ids
        ):
            raise LossError("class_ids must contain integers.")

        if self.ignore_index in ids:
            raise LossError(
                "ignore_index cannot overlap class_ids."
            )

        if self.internal_ignore_index in range(len(ids)):
            raise LossError(
                "internal_ignore_index overlaps model channels."
            )

        if self.name not in {x.value for x in LossName}:
            raise LossError(
                f"Unsupported loss '{self.name}'."
            )

        if self.class_weighting not in {
            x.value for x in ClassWeightingStrategy
        }:
            raise LossError(
                f"Unsupported weighting '{self.class_weighting}'."
            )

        numeric = {
            "gamma": self.gamma,
            "smooth": self.smooth,
            "ce_weight": self.ce_weight,
            "dice_weight": self.dice_weight,
            "focal_weight": self.focal_weight,
            "tversky_weight": self.tversky_weight,
        }

        for name, value in numeric.items():

            value = float(value)

            if not torch.isfinite(torch.tensor(value)):
                raise LossError(
                    f"{name} must be finite."
                )

            if value < 0:
                raise LossError(
                    f"{name} must be >= 0."
                )

        if not 0 <= self.tversky_alpha <= 1:
            raise LossError(
                "tversky_alpha must lie in [0,1]."
            )

        if not 0 <= self.tversky_beta <= 1:
            raise LossError(
                "tversky_beta must lie in [0,1]."
            )

        if self.tversky_alpha + self.tversky_beta == 0:
            raise LossError(
                "At least one Tversky coefficient must be positive."
            )

        if not 0 <= self.effective_number_beta < 1:
            raise LossError(
                "effective_number_beta must lie in [0,1)."
            )

        if self.name == LossName.CE_DICE.value:

            if self.ce_weight + self.dice_weight == 0:
                raise LossError(
                    "CE + Dice requires positive total weight."
                )

        if self.name == LossName.FOCAL_DICE.value:

            if self.focal_weight + self.dice_weight == 0:
                raise LossError(
                    "Focal + Dice requires positive total weight."
                )

        if self.name == LossName.TVERSKY_FOCAL.value:

            if self.tversky_weight + self.focal_weight == 0:
                raise LossError(
                    "Tversky + Focal requires positive total weight."
                )

        object.__setattr__(self, "class_ids", ids)


# ============================================================
# Class weighting
# ============================================================


def compute_class_weights(
    class_counts: Mapping[int, int | float] | None,
    *,
    class_ids: Sequence[int],
    strategy: str = "none",
    effective_number_beta: float = 0.9999,
) -> Tensor:

    ids = tuple(class_ids)

    if len(ids) == 0:
        raise LossError("class_ids cannot be empty.")

    if strategy not in {
        x.value for x in ClassWeightingStrategy
    }:
        raise LossError(
            f"Unsupported class weighting '{strategy}'."
        )

    if strategy == "none":
        return torch.ones(
            len(ids),
            dtype=torch.float32,
        )

    if class_counts is None:
        raise LossError(
            f"class_counts required for '{strategy}'."
        )

    counts = torch.tensor(
        [
            float(class_counts.get(cid, 0))
            for cid in ids
        ],
        dtype=torch.float64,
    )

    if not torch.isfinite(counts).all():
        raise LossError(
            "class_counts must be finite."
        )

    if torch.any(counts <= 0):
        raise LossError(
            "All class counts must be positive."
        )

    if strategy == "inverse_frequency":

        weights = 1.0 / counts

    elif strategy == "median_frequency":

        weights = torch.median(counts) / counts

    else:

        beta = torch.tensor(
            effective_number_beta,
            dtype=torch.float64,
        )

        effective = (
            1.0
            - torch.pow(beta, counts)
        )

        weights = (
            1.0 - beta
        ) / effective

    weights = weights / weights.mean()

    if not torch.isfinite(weights).all():
        raise LossError(
            "Invalid class weights."
        )

    return weights.float()


# ============================================================
# Validation
# ============================================================


def _validate_inputs(
    logits: Tensor,
    target: Tensor,
    num_classes: int,
):

    if not isinstance(logits, Tensor):
        raise LossError(
            "logits must be Tensor."
        )

    if not isinstance(target, Tensor):
        raise LossError(
            "target must be Tensor."
        )

    if logits.ndim != 4:
        raise LossError(
            "logits must have shape [N,C,H,W]."
        )

    if target.ndim != 3:
        raise LossError(
            "target must have shape [N,H,W]."
        )

    if logits.shape[0] != target.shape[0]:
        raise LossError(
            "Batch dimension mismatch."
        )

    if logits.shape[-2:] != target.shape[-2:]:
        raise LossError(
            "Spatial dimensions mismatch."
        )

    if logits.shape[1] != num_classes:
        raise LossError(
            f"Expected {num_classes} channels."
        )

    if not logits.is_floating_point():
        raise LossError(
            "logits must be floating point."
        )

    if not torch.isfinite(logits).all():
        raise LossError(
            "logits contain NaN or Inf."
        )

    if target.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise LossError(
            "target must use integer dtype."
        )


# ============================================================
# Label remapping
# ============================================================


def _map_external_to_internal(
    target: Tensor,
    class_ids: Sequence[int],
    ignore_index: int,
    internal_ignore_index: int,
):

    mapped = torch.full(
        target.shape,
        internal_ignore_index,
        dtype=torch.long,
        device=target.device,
    )

    valid = target != ignore_index
    unknown = valid.clone()

    for internal, external in enumerate(class_ids):

        hit = target == external

        mapped[hit] = internal

        unknown &= ~hit

    if torch.any(unknown):

        bad = torch.unique(
            target[unknown]
        ).tolist()

        raise LossError(
            f"Target contains unknown class IDs {bad}"
        )

    return mapped, valid


# ============================================================
# Loss module
# ============================================================


class SegmentationLoss(nn.Module):

    def __init__(
        self,
        config: LossConfig,
        class_weights: Tensor | None = None,
    ):
        super().__init__()

        self.config = config

        if class_weights is None:

            self.register_buffer(
                "class_weights",
                torch.empty(0),
            )

        else:

            if class_weights.ndim != 1:
                raise LossError(
                    "class_weights must be 1D."
                )

            if class_weights.numel() != len(config.class_ids):
                raise LossError(
                    "class_weights length mismatch."
                )

            if not torch.isfinite(class_weights).all():
                raise LossError(
                    "class_weights contain NaN/Inf."
                )

            if torch.any(class_weights <= 0):
                raise LossError(
                    "class_weights must be positive."
                )

            self.register_buffer(
                "class_weights",
                class_weights.float().clone(),
            )

    def _weights(self, logits):

        if self.class_weights.numel() == 0:
            return None

        return self.class_weights.to(
            logits.device,
            logits.dtype,
        )

    def _prepare(self, logits, target):

        _validate_inputs(
            logits,
            target,
            len(self.config.class_ids),
        )

        return _map_external_to_internal(
            target,
            self.config.class_ids,
            self.config.ignore_index,
            self.config.internal_ignore_index,
        )

    def _cross_entropy(self, logits, target):

        valid = (
            target
            != self.config.internal_ignore_index
        )

        if not torch.any(valid):
            return logits.sum() * 0.0

        return F.cross_entropy(
            logits,
            target,
            weight=self._weights(logits),
            ignore_index=self.config.internal_ignore_index,
        )

    def _dice(self, logits, target):

        valid = (
            target
            != self.config.internal_ignore_index
        )

        if not torch.any(valid):
            return logits.sum() * 0.0

        probs = logits.softmax(dim=1)

        safe = target.clamp_min(0)

        one_hot = F.one_hot(
            safe,
            len(self.config.class_ids),
        ).permute(0, 3, 1, 2)

        one_hot = one_hot.to(probs.dtype)

        mask = valid.unsqueeze(1)

        probs = probs * mask
        one_hot = one_hot * mask

        dims = (0, 2, 3)

        intersection = (
            probs * one_hot
        ).sum(dims)

        denominator = (
            probs.sum(dims)
            + one_hot.sum(dims)
        )

        dice = (
            2 * intersection
            + self.config.smooth
        ) / (
            denominator
            + self.config.smooth
        )

        present = one_hot.sum(dims) > 0

        if not torch.any(present):
            return logits.sum() * 0.0

        dice = dice[present]

        weights = self._weights(logits)

        if weights is not None:

            weights = weights[present]
            weights = weights / weights.sum()

            return (
                (1 - dice) * weights
            ).sum()

        return (1 - dice).mean()

    def _focal(self, logits, target):

        valid = (
            target
            != self.config.internal_ignore_index
        )

        if not torch.any(valid):
            return logits.sum() * 0.0

        log_prob = F.log_softmax(
            logits,
            dim=1,
        )

        safe = target.clamp_min(0)

        nll = -torch.gather(
            log_prob,
            1,
            safe.unsqueeze(1),
        ).squeeze(1)

        pt = torch.exp(-nll)

        focal = (
            1 - pt
        ).pow(self.config.gamma)

        loss = focal * nll

        weights = self._weights(logits)

        if weights is not None:
            loss = loss * weights[safe]

        return loss[valid].mean()

    def _tversky(self, logits, target):

        valid = (
            target
            != self.config.internal_ignore_index
        )

        if not torch.any(valid):
            return logits.sum() * 0.0

        probs = logits.softmax(dim=1)

        safe = target.clamp_min(0)

        one_hot = F.one_hot(
            safe,
            len(self.config.class_ids),
        ).permute(0, 3, 1, 2)

        one_hot = one_hot.to(probs.dtype)

        mask = valid.unsqueeze(1)

        probs = probs * mask
        one_hot = one_hot * mask

        dims = (0, 2, 3)

        tp = (
            probs * one_hot
        ).sum(dims)

        fp = (
            probs * (1 - one_hot)
        ).sum(dims)

        fn = (
            (1 - probs) * one_hot
        ).sum(dims)

        score = (
            tp + self.config.smooth
        ) / (
            tp
            + self.config.tversky_alpha * fp
            + self.config.tversky_beta * fn
            + self.config.smooth
        )

        present = one_hot.sum(dims) > 0

        if not torch.any(present):
            return logits.sum() * 0.0

        score = score[present]

        weights = self._weights(logits)

        if weights is not None:

            weights = weights[present]
            weights = weights / weights.sum()

            return (
                (1 - score) * weights
            ).sum()

        return (1 - score).mean()

    def forward(self, logits, target):

        mapped, _ = self._prepare(
            logits,
            target,
        )

        name = self.config.name

        if name == "cross_entropy":
            return self._cross_entropy(logits, mapped)

        if name == "weighted_cross_entropy":

            if self.class_weights.numel() == 0:
                raise LossError(
                    "weighted_cross_entropy requires class weights."
                )

            return self._cross_entropy(logits, mapped)

        if name == "dice":
            return self._dice(logits, mapped)

        if name == "ce_dice":

            ce = self._cross_entropy(logits, mapped)
            dice = self._dice(logits, mapped)

            return (
                self.config.ce_weight * ce
                + self.config.dice_weight * dice
            )

        if name == "focal":
            return self.config.focal_weight * self._focal(logits, mapped)

        if name == "focal_dice":

            focal = self._focal(logits, mapped)
            dice = self._dice(logits, mapped)

            return (
                self.config.focal_weight * focal
                + self.config.dice_weight * dice
            )

        if name == "tversky":
            return self.config.tversky_weight * self._tversky(logits, mapped)

        if name == "tversky_focal":

            tv = self._tversky(logits, mapped)
            focal = self._focal(logits, mapped)

            return (
                self.config.tversky_weight * tv
                + self.config.focal_weight * focal
            )

        raise LossError(
            f"Unsupported loss {name}"
        )


# ============================================================
# Factory
# ============================================================


def build_loss(
    config: LossConfig,
    *,
    class_counts: Mapping[int, int | float] | None = None,
    class_weights: Tensor | None = None,
) -> SegmentationLoss:

    if not isinstance(config, LossConfig):
        raise LossError(
            "config must be LossConfig."
        )

    resolved_weights = class_weights

    weighting = config.class_weighting

    if (
        config.name == "weighted_cross_entropy"
        and weighting == "none"
        and resolved_weights is None
    ):
        weighting = "inverse_frequency"

    if resolved_weights is None and weighting != "none":

        resolved_weights = compute_class_weights(
            class_counts,
            class_ids=config.class_ids,
            strategy=weighting,
            effective_number_beta=config.effective_number_beta,
        )

    if (
        config.name == "weighted_cross_entropy"
        and resolved_weights is None
    ):
        raise LossError(
            "weighted_cross_entropy requires class weights."
        )

    return SegmentationLoss(
        config,
        resolved_weights,
    )


def build_loss_from_name(
    name: str,
    *,
    class_ids: Sequence[int] = (
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    ),
    ignore_index: int = 0,
    internal_ignore_index: int = -100,
    class_weighting: str = "none",
    class_counts: Mapping[int, int | float] | None = None,
    class_weights: Tensor | None = None,
    gamma: float = 2.0,
    tversky_alpha: float = 0.5,
    tversky_beta: float = 0.5,
    smooth: float = 1.0,
    ce_weight: float = 1.0,
    dice_weight: float = 1.0,
    focal_weight: float = 1.0,
    tversky_weight: float = 1.0,
    effective_number_beta: float = 0.9999,
) -> SegmentationLoss:

    config = LossConfig(
        name=name,
        class_ids=tuple(class_ids),
        ignore_index=ignore_index,
        internal_ignore_index=internal_ignore_index,
        class_weighting=class_weighting,
        gamma=gamma,
        tversky_alpha=tversky_alpha,
        tversky_beta=tversky_beta,
        smooth=smooth,
        ce_weight=ce_weight,
        dice_weight=dice_weight,
        focal_weight=focal_weight,
        tversky_weight=tversky_weight,
        effective_number_beta=effective_number_beta,
    )

    return build_loss(
        config,
        class_counts=class_counts,
        class_weights=class_weights,
    )


def build_loss_from_training_config(
    training_config,
    *,
    class_counts=None,
):

    return build_loss_from_name(
        training_config.loss,
        class_weighting=training_config.class_weighting,
        class_counts=class_counts,
    )


__all__ = [
    "LossError",
    "LossName",
    "ClassWeightingStrategy",
    "LossConfig",
    "compute_class_weights",
    "SegmentationLoss",
    "build_loss",
    "build_loss_from_name",
    "build_loss_from_training_config",
]