"""
training/metrics.py

Production-grade semantic-segmentation metric accumulation.

The metric engine separates:
    external dataset labels
        ↓
    canonical contiguous class indices
        ↓
    confusion matrix
        ↓
    reported metrics

This allows datasets such as LoveDA, whose semantic labels are 1..7 with
0 reserved for IGNORE_INDEX, to coexist with model outputs whose channels
are always 0..C-1.

Supported:
- confusion matrix
- per-class IoU / Jaccard
- mean IoU
- frequency-weighted IoU
- per-class precision
- per-class recall
- per-class F1 / Dice
- macro precision / recall / F1
- pixel accuracy
- mean class accuracy
- ignore-index handling
- arbitrary external class IDs
- streaming batch accumulation
- logits or class-index predictions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch


class SegmentationMetricError(ValueError):
    """Raised for invalid segmentation metric inputs."""


@dataclass(frozen=True)
class SegmentationMetrics:
    """Immutable snapshot of accumulated segmentation metrics."""

    pixel_accuracy: float
    mean_class_accuracy: float
    mean_iou: float
    frequency_weighted_iou: float

    macro_precision: float
    macro_recall: float
    macro_f1: float

    per_class_iou: Dict[int, float]
    per_class_precision: Dict[int, float]
    per_class_recall: Dict[int, float]
    per_class_f1: Dict[int, float]

    valid_pixels: int
    confusion_matrix: torch.Tensor

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation."""
        return {
            "pixel_accuracy": self.pixel_accuracy,
            "mean_class_accuracy": self.mean_class_accuracy,
            "mean_iou": self.mean_iou,
            "frequency_weighted_iou": self.frequency_weighted_iou,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "per_class_iou": self.per_class_iou,
            "per_class_precision": self.per_class_precision,
            "per_class_recall": self.per_class_recall,
            "per_class_f1": self.per_class_f1,
            "valid_pixels": self.valid_pixels,
            "confusion_matrix": self.confusion_matrix.tolist(),
        }


class SegmentationMetricAccumulator:
    """
    Streaming semantic-segmentation metric accumulator.

    Parameters
    ----------
    num_classes:
        Number of semantic classes.

    ignore_index:
        Dataset label that should be excluded from evaluation.

    class_ids:
        External dataset class IDs corresponding to model channels.

        Example for LoveDA:

            num_classes = 7
            ignore_index = 0
            class_ids = range(1, 8)

        This means:

            external 1 -> internal 0
            external 2 -> internal 1
            ...
            external 7 -> internal 6

        The confusion matrix therefore always remains C x C while the
        public per-class metrics use the original external IDs.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        ignore_index: int = 0,
        class_ids: Optional[Sequence[int]] = None,
    ) -> None:
        if num_classes <= 0:
            raise SegmentationMetricError(
                "num_classes must be greater than zero."
            )

        if class_ids is None:
            class_ids = tuple(range(num_classes))
        else:
            class_ids = tuple(class_ids)

        if len(class_ids) != num_classes:
            raise SegmentationMetricError(
                "class_ids length must equal num_classes."
            )

        if len(set(class_ids)) != num_classes:
            raise SegmentationMetricError(
                "class_ids must be unique."
            )

        if ignore_index in class_ids:
            raise SegmentationMetricError(
                "ignore_index cannot also be a valid class ID."
            )

        self.num_classes = num_classes
        self.ignore_index = int(ignore_index)
        self.class_ids = class_ids

        self._external_to_internal = {
            int(external_id): internal_id
            for internal_id, external_id in enumerate(class_ids)
        }

        self._confusion = torch.zeros(
            (num_classes, num_classes),
            dtype=torch.int64,
        )

    @property
    def confusion_matrix(self) -> torch.Tensor:
        """Return a defensive copy of the accumulated confusion matrix."""
        return self._confusion.clone()

    @property
    def valid_pixels(self) -> int:
        """Return the number of non-ignored pixels accumulated."""
        return int(self._confusion.sum().item())

    def reset(self) -> None:
        """Reset all accumulated statistics."""
        self._confusion.zero_()

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """
        Accumulate a prediction/target batch.

        Predictions may be either:

        1. Model logits:
               (N, C, H, W)

           In this case argmax produces canonical class indices
           0..C-1.

        2. Class-index predictions:
               (...,)

           These are interpreted using the configured external class IDs.

        Targets are always interpreted as external dataset labels.

        For LoveDA:
            target labels = 0..7
            0 = ignore
            1..7 = semantic classes
        """
        self._validate_tensor_inputs(predictions, targets)

        # --------------------------------------------------------------
        # Logits → canonical model indices
        # --------------------------------------------------------------

        predictions_are_logits = (
            predictions.ndim == targets.ndim + 1
        )

        if predictions_are_logits:
            if predictions.shape[1] != self.num_classes:
                raise SegmentationMetricError(
                    "Logit channel count does not match num_classes."
                )

            predictions = predictions.argmax(dim=1)

            # Logits produce canonical indices 0..C-1.
            prediction_indices = predictions.to(torch.int64)

        else:
            # Class-index predictions are interpreted as external IDs.
            prediction_indices = self._map_external_labels(
                predictions,
                name="predictions",
            )

        # --------------------------------------------------------------
        # Targets → canonical model indices
        # --------------------------------------------------------------

        target_indices = self._map_external_labels(
            targets,
            name="targets",
            allow_ignore=True,
        )

        if prediction_indices.shape != target_indices.shape:
            raise SegmentationMetricError(
                "Prediction and target shapes must match after "
                "canonicalization: "
                f"predictions={tuple(prediction_indices.shape)}, "
                f"targets={tuple(target_indices.shape)}."
            )

        prediction_indices = prediction_indices.reshape(-1)
        target_indices = target_indices.reshape(-1)

        # Ignore pixels are represented as -1 internally.
        valid = target_indices != -1

        if not torch.any(valid):
            return

        prediction_indices = prediction_indices[valid]
        target_indices = target_indices[valid]

        invalid_predictions = (
            (prediction_indices < 0)
            | (prediction_indices >= self.num_classes)
        )

        if torch.any(invalid_predictions):
            bad = prediction_indices[
                invalid_predictions
            ][:10].tolist()

            raise SegmentationMetricError(
                "Predictions contain invalid class IDs: "
                f"{bad}."
            )

        encoded = (
            target_indices * self.num_classes
            + prediction_indices
        )

        batch_confusion = torch.bincount(
            encoded,
            minlength=self.num_classes * self.num_classes,
        ).reshape(
            self.num_classes,
            self.num_classes,
        )

        self._confusion += batch_confusion.cpu()

    def compute(self) -> SegmentationMetrics:
        """Compute a consistent snapshot of all accumulated metrics."""
        cm = self._confusion.to(torch.float64)

        true_positive = torch.diag(cm)

        actual = cm.sum(dim=1)
        predicted = cm.sum(dim=0)

        total = cm.sum()

        class_accuracy = self._safe_divide(
            true_positive,
            actual,
        )

        precision = self._safe_divide(
            true_positive,
            predicted,
        )

        union = (
            actual
            + predicted
            - true_positive
        )

        iou = self._safe_divide(
            true_positive,
            union,
        )

        recall = class_accuracy

        f1 = self._safe_divide(
            2.0 * precision * recall,
            precision + recall,
        )

        if total > 0:
            pixel_accuracy = float(
                (true_positive.sum() / total).item()
            )
        else:
            pixel_accuracy = float("nan")

        mean_class_accuracy = self._nanmean(
            class_accuracy
        )

        mean_iou = self._nanmean(iou)

        macro_precision = self._nanmean(
            precision
        )

        macro_recall = self._nanmean(
            recall
        )

        macro_f1 = self._nanmean(f1)

        if total > 0:
            frequencies = actual / total

            frequency_weighted_iou = float(
                (
                    frequencies
                    * torch.nan_to_num(
                        iou,
                        nan=0.0,
                    )
                ).sum().item()
            )
        else:
            frequency_weighted_iou = float("nan")

        def keyed(
            values: torch.Tensor,
        ) -> Dict[int, float]:
            return {
                int(class_id): float(value)
                for class_id, value
                in zip(self.class_ids, values.tolist())
            }

        return SegmentationMetrics(
            pixel_accuracy=pixel_accuracy,
            mean_class_accuracy=mean_class_accuracy,
            mean_iou=mean_iou,
            frequency_weighted_iou=frequency_weighted_iou,
            macro_precision=macro_precision,
            macro_recall=macro_recall,
            macro_f1=macro_f1,
            per_class_iou=keyed(iou),
            per_class_precision=keyed(precision),
            per_class_recall=keyed(recall),
            per_class_f1=keyed(f1),
            valid_pixels=self.valid_pixels,
            confusion_matrix=self.confusion_matrix,
        )

    def _map_external_labels(
        self,
        values: torch.Tensor,
        *,
        name: str,
        allow_ignore: bool = False,
    ) -> torch.Tensor:
        """
        Convert external dataset labels into contiguous internal indices.

        Valid class:
            external ID → 0..C-1

        Ignore:
            ignore_index → -1

        Unknown IDs are rejected rather than silently interpreted.
        """
        values = values.detach().to(torch.int64)

        result = torch.full_like(
            values,
            fill_value=-2,
        )

        for external_id, internal_id in (
            self._external_to_internal.items()
        ):
            result[values == external_id] = internal_id

        if allow_ignore:
            result[
                values == self.ignore_index
            ] = -1

        unknown = result == -2

        if torch.any(unknown):
            bad = (
                torch.unique(values[unknown])
                .tolist()
            )

            raise SegmentationMetricError(
                f"{name} contain unknown external class IDs: "
                f"{bad}."
            )

        if not allow_ignore:
            ignored = values == self.ignore_index

            if torch.any(ignored):
                raise SegmentationMetricError(
                    f"{name} contain ignore_index={self.ignore_index}. "
                    "Ignore labels are only valid in targets."
                )

        return result

    @staticmethod
    def _validate_tensor_inputs(
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        if not isinstance(predictions, torch.Tensor):
            raise SegmentationMetricError(
                "predictions must be a torch.Tensor."
            )

        if not isinstance(targets, torch.Tensor):
            raise SegmentationMetricError(
                "targets must be a torch.Tensor."
            )

        if predictions.ndim == 0:
            raise SegmentationMetricError(
                "predictions must have at least one dimension."
            )

        if targets.ndim == 0:
            raise SegmentationMetricError(
                "targets must have at least one dimension."
            )

    @staticmethod
    def _safe_divide(
        numerator: torch.Tensor,
        denominator: torch.Tensor,
    ) -> torch.Tensor:
        result = torch.full_like(
            numerator,
            float("nan"),
            dtype=torch.float64,
        )

        valid = denominator != 0

        result[valid] = (
            numerator[valid]
            / denominator[valid]
        )

        return result

    @staticmethod
    def _nanmean(
        values: torch.Tensor,
    ) -> float:
        valid = ~torch.isnan(values)

        if not torch.any(valid):
            return float("nan")

        return float(
            values[valid].mean().item()
        )