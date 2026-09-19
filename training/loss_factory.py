"""
Factory for constructing DRISHTI segmentation losses.

This module keeps experiment configuration separate from the actual
loss mathematics.

Typical usage
-------------
    config = LossConfig(
        name="ce_dice",
        class_weighting="effective_number",
    )

    loss_fn = build_loss(
        config,
        class_counts=training_pixel_counts,
    )

    loss = loss_fn(logits, target)
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor

from training.losses import (
    ClassWeightingStrategy,
    LossConfig,
    LossError,
    LossName,
    SegmentationLoss,
    compute_class_weights,
)


def build_loss(
    config: LossConfig,
    *,
    class_counts: Mapping[int, int | float] | None = None,
    class_weights: Tensor | None = None,
) -> SegmentationLoss:
    """
    Construct a fully configured SegmentationLoss.

    Parameters
    ----------
    config:
        Immutable loss configuration.

    class_counts:
        Training-set pixel counts by external class ID.
        Required when config.class_weighting is not "none".

    class_weights:
        Optional explicit weights. Explicit weights are useful for
        controlled experiments or externally calibrated weighting.

        Explicit weights take precedence over computed weights.

    Returns
    -------
    SegmentationLoss

    Raises
    ------
    LossError
        If the requested weighting configuration cannot be satisfied.
    """
    if not isinstance(config, LossConfig):
        raise LossError(
            "config must be an instance of LossConfig."
        )

    computed_weights: Tensor | None = None

    if class_weights is not None:
        if config.class_weighting != (
            ClassWeightingStrategy.NONE.value
        ):
            # Explicit weights intentionally take precedence.
            # This allows reproducible experiments with externally
            # specified weighting while retaining the configuration
            # metadata describing the experiment.
            pass

        computed_weights = class_weights.detach().float()

    elif config.class_weighting != (
        ClassWeightingStrategy.NONE.value
    ):
        computed_weights = compute_class_weights(
            class_counts,
            class_ids=config.class_ids,
            strategy=config.class_weighting,
            effective_number_beta=config.effective_number_beta,
        )

    if (
        config.name
        == LossName.WEIGHTED_CROSS_ENTROPY.value
        and computed_weights is None
    ):
        raise LossError(
            "weighted_cross_entropy requires class weighting "
            "or explicit class_weights."
        )

    return SegmentationLoss(
        config=config,
        class_weights=computed_weights,
    )


def build_loss_from_name(
    name: str,
    *,
    class_ids: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7),
    ignore_index: int = 0,
    class_weighting: str = (
        ClassWeightingStrategy.NONE.value
    ),
    class_counts: Mapping[int, int | float] | None = None,
    class_weights: Tensor | None = None,
    gamma: float = 2.0,
    tversky_alpha: float = 0.5,
    tversky_beta: float = 0.5,
    smooth: float = 1.0,
    ce_weight: float = 1.0,
    dice_weight: float = 1.0,
    tversky_weight: float = 1.0,
    focal_weight: float = 1.0,
    effective_number_beta: float = 0.9999,
) -> SegmentationLoss:
    """
    Convenience constructor for experiment runners.

    All important experiment parameters remain explicit rather than
    hidden in global state.
    """
    config = LossConfig(
        name=name,
        class_ids=class_ids,
        ignore_index=ignore_index,
        class_weighting=class_weighting,
        gamma=gamma,
        tversky_alpha=tversky_alpha,
        tversky_beta=tversky_beta,
        smooth=smooth,
        ce_weight=ce_weight,
        dice_weight=dice_weight,
        tversky_weight=tversky_weight,
        focal_weight=focal_weight,
        effective_number_beta=effective_number_beta,
    )

    return build_loss(
        config,
        class_counts=class_counts,
        class_weights=class_weights,
    )


__all__ = [
    "build_loss",
    "build_loss_from_name",
]