"""
DRISHTI segmentation model contract.

All trainable segmentation architectures must conform to this module.

Canonical input:
    [N, 3, H, W]

Canonical output:
    [N, C, H, W]

For LoveDA:
    C = 7

External dataset labels:
    0 = ignore
    1 = background
    2 = building
    3 = road
    4 = water
    5 = barren
    6 = forest
    7 = agriculture

Model outputs use canonical internal channel indices:
    0..6
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn


class ModelContractError(ValueError):
    """Raised when a model violates the DRISHTI contract."""


DEFAULT_CLASS_IDS: tuple[int, ...] = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
)


@dataclass(frozen=True)
class ModelSpec:
    """
    Immutable metadata describing a segmentation architecture.

    This metadata is configuration/provenance, not runtime inference
    state. Runtime inference metadata such as device/backend belongs
    to the inference layer.
    """

    model_id: str
    model_version: str
    architecture: str

    num_classes: int = len(DEFAULT_CLASS_IDS)
    in_channels: int = 3

    class_ids: tuple[int, ...] = DEFAULT_CLASS_IDS

    input_size: int | None = None

    pretrained_encoder: bool = False
    pretrained_source: str | None = None

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ModelContractError(
                "model_id must be non-empty."
            )

        if not self.model_version.strip():
            raise ModelContractError(
                "model_version must be non-empty."
            )

        if not self.architecture.strip():
            raise ModelContractError(
                "architecture must be non-empty."
            )

        if self.num_classes <= 0:
            raise ModelContractError(
                "num_classes must be > 0."
            )

        if self.in_channels <= 0:
            raise ModelContractError(
                "in_channels must be > 0."
            )

        if len(self.class_ids) != self.num_classes:
            raise ModelContractError(
                "len(class_ids) must equal num_classes."
            )

        if len(set(self.class_ids)) != len(self.class_ids):
            raise ModelContractError(
                "class_ids must contain unique values."
            )

        if any(
            not isinstance(class_id, int)
            or isinstance(class_id, bool)
            for class_id in self.class_ids
        ):
            raise ModelContractError(
                "class_ids must contain integers."
            )

        if self.input_size is not None:
            if self.input_size <= 0:
                raise ModelContractError(
                    "input_size must be > 0 when provided."
                )

        if self.pretrained_encoder and not self.pretrained_source:
            raise ModelContractError(
                "pretrained_source is required when "
                "pretrained_encoder=True."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible model metadata."""
        result = asdict(self)
        result["class_ids"] = list(self.class_ids)
        return result


class SegmentationModel(nn.Module, ABC):
    """
    Abstract base class for every DRISHTI segmentation architecture.

    Subclasses implement `_forward_impl`; the public `forward` method
    validates the resulting tensor against the canonical contract.

    This prevents architecture-specific output quirks from leaking
    into the trainer.
    """

    spec: ModelSpec

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()

        if not isinstance(spec, ModelSpec):
            raise ModelContractError(
                "spec must be a ModelSpec."
            )

        self.spec = spec

    @property
    def model_id(self) -> str:
        return self.spec.model_id

    @property
    def model_version(self) -> str:
        return self.spec.model_version

    @property
    def architecture(self) -> str:
        return self.spec.architecture

    @property
    def num_classes(self) -> int:
        return self.spec.num_classes

    @property
    def in_channels(self) -> int:
        return self.spec.in_channels

    @property
    def class_ids(self) -> tuple[int, ...]:
        return self.spec.class_ids

    def parameter_count(
        self,
        *,
        trainable_only: bool = False,
    ) -> int:
        """
        Return the number of parameters.

        Parameters
        ----------
        trainable_only:
            If True, only parameters with requires_grad=True are counted.
        """
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if not trainable_only
            or parameter.requires_grad
        )

    def freeze_encoder(self) -> None:
        """
        Freeze parameters belonging to an encoder when the concrete
        architecture exposes one.

        Architectures without an encoder simply remain unchanged.
        """
        encoder = getattr(self, "encoder", None)

        if encoder is None:
            raise ModelContractError(
                f"{self.architecture} does not expose an encoder."
            )

        for parameter in encoder.parameters():
            parameter.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze parameters belonging to an exposed encoder."""
        encoder = getattr(self, "encoder", None)

        if encoder is None:
            raise ModelContractError(
                f"{self.architecture} does not expose an encoder."
            )

        for parameter in encoder.parameters():
            parameter.requires_grad = True

    def forward(self, x: Tensor) -> Tensor:
        """
        Validate input, execute architecture, then validate logits.

        Returns:
            Tensor with shape [N, C, H, W].
        """
        self._validate_input(x)

        logits = self._forward_impl(x)

        self._validate_output(
            logits,
            input_tensor=x,
        )

        return logits

    @abstractmethod
    def _forward_impl(self, x: Tensor) -> Tensor:
        """Architecture-specific forward implementation."""
        raise NotImplementedError

    def _validate_input(self, x: Tensor) -> None:
        if not isinstance(x, Tensor):
            raise ModelContractError(
                "Model input must be a torch.Tensor."
            )

        if x.ndim != 4:
            raise ModelContractError(
                "Model input must have shape [N, C, H, W]."
            )

        if x.shape[1] != self.in_channels:
            raise ModelContractError(
                f"Expected {self.in_channels} input channels, "
                f"got {x.shape[1]}."
            )

        if x.shape[0] <= 0:
            raise ModelContractError(
                "Batch dimension must be > 0."
            )

        if x.shape[-2] <= 0 or x.shape[-1] <= 0:
            raise ModelContractError(
                "Spatial dimensions must be > 0."
            )

        if not x.is_floating_point():
            raise ModelContractError(
                "Model input must use a floating-point dtype."
            )

        if not torch.isfinite(x).all():
            raise ModelContractError(
                "Model input contains NaN or infinite values."
            )

    def _validate_output(
        self,
        logits: Tensor,
        *,
        input_tensor: Tensor,
    ) -> None:
        if not isinstance(logits, Tensor):
            raise ModelContractError(
                f"{self.architecture} must return a torch.Tensor."
            )

        if logits.ndim != 4:
            raise ModelContractError(
                "Model output must have shape [N, C, H, W]."
            )

        if logits.shape[0] != input_tensor.shape[0]:
            raise ModelContractError(
                "Model output batch dimension does not match input."
            )

        if logits.shape[1] != self.num_classes:
            raise ModelContractError(
                f"Expected {self.num_classes} output channels, "
                f"got {logits.shape[1]}."
            )

        if logits.shape[-2:] != input_tensor.shape[-2:]:
            raise ModelContractError(
                "Model output spatial dimensions must match input "
                "spatial dimensions."
            )

        if not logits.is_floating_point():
            raise ModelContractError(
                "Model output must use a floating-point dtype."
            )

        if not torch.isfinite(logits).all():
            raise ModelContractError(
                "Model output contains NaN or infinite values."
            )

    def metadata(self) -> Mapping[str, Any]:
        """
        Return immutable model metadata plus parameter statistics.
        """
        return {
            **self.spec.to_dict(),
            "parameter_count": self.parameter_count(),
            "trainable_parameter_count": self.parameter_count(
                trainable_only=True
            ),
        }