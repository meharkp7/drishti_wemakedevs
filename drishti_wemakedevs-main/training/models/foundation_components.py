from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class FoundationComponentError(ValueError):
    """Raised for invalid foundation-model components."""


@dataclass(frozen=True)
class FeaturePyramid:
    """
    Ordered multi-scale feature representation.

    Features are ordered from highest spatial resolution to lowest.
    """

    features: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        if not self.features:
            raise FoundationComponentError("Feature pyramid cannot be empty.")

        reference_batch = self.features[0].shape[0]

        previous_h = None
        previous_w = None

        for index, feature in enumerate(self.features):
            if not isinstance(feature, torch.Tensor):
                raise FoundationComponentError(
                    f"Feature {index} must be a Tensor."
                )

            if feature.ndim != 4:
                raise FoundationComponentError(
                    f"Feature {index} must be [B,C,H,W]."
                )

            if feature.shape[0] != reference_batch:
                raise FoundationComponentError(
                    "All pyramid features must have the same batch size."
                )

            if feature.shape[1] <= 0:
                raise FoundationComponentError(
                    f"Feature {index} has invalid channel count."
                )

            height, width = feature.shape[-2:]

            if height <= 0 or width <= 0:
                raise FoundationComponentError(
                    f"Feature {index} has invalid spatial dimensions."
                )

            if not torch.is_floating_point(feature):
                raise FoundationComponentError(
                    f"Feature {index} must be floating point."
                )

            if not torch.isfinite(feature).all():
                raise FoundationComponentError(
                    f"Feature {index} contains non-finite values."
                )

            if previous_h is not None:
                if height > previous_h or width > previous_w:
                    raise FoundationComponentError(
                        "Feature pyramid must progress from high to low "
                        "spatial resolution."
                    )

            previous_h = height
            previous_w = width

    @property
    def batch_size(self) -> int:
        return self.features[0].shape[0]

    @property
    def channels(self) -> tuple[int, ...]:
        return tuple(feature.shape[1] for feature in self.features)

    @property
    def resolutions(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (feature.shape[-2], feature.shape[-1])
            for feature in self.features
        )


class FoundationEncoder(nn.Module, ABC):
    """
    Stable interface for foundation/self-supervised image encoders.

    The encoder exposes a multi-scale feature pyramid rather than logits.
    """

    def __init__(
        self,
        *,
        name: str,
        feature_channels: Sequence[int],
        pretrained: bool,
        checkpoint: str | None,
    ) -> None:
        super().__init__()

        if not name.strip():
            raise FoundationComponentError("Encoder name cannot be empty.")

        channels = tuple(int(value) for value in feature_channels)

        if not channels or any(value <= 0 for value in channels):
            raise FoundationComponentError(
                "feature_channels must contain positive integers."
            )

        if pretrained and not checkpoint:
            raise FoundationComponentError(
                "A checkpoint/source is required for pretrained encoders."
            )

        self._name = name
        self._feature_channels = channels
        self._pretrained = pretrained
        self._checkpoint = checkpoint

    @property
    def name(self) -> str:
        return self._name

    @property
    def feature_channels(self) -> tuple[int, ...]:
        return self._feature_channels

    @property
    def pretrained(self) -> bool:
        return self._pretrained

    @property
    def checkpoint(self) -> str | None:
        return self._checkpoint

    @abstractmethod
    def extract_features(self, x: torch.Tensor) -> FeaturePyramid:
        """Extract a validated multi-scale feature pyramid."""

    def forward(self, x: torch.Tensor) -> FeaturePyramid:
        if not isinstance(x, torch.Tensor):
            raise FoundationComponentError("Input must be a Tensor.")

        if x.ndim != 4:
            raise FoundationComponentError("Input must have shape [B,C,H,W].")

        if x.shape[1] != 3:
            raise FoundationComponentError("Foundation encoder expects RGB input.")

        if x.shape[0] <= 0 or x.shape[-2] <= 0 or x.shape[-1] <= 0:
            raise FoundationComponentError("Input dimensions must be positive.")

        if not torch.is_floating_point(x):
            raise FoundationComponentError("Input must be floating point.")

        if not torch.isfinite(x).all():
            raise FoundationComponentError("Input contains non-finite values.")

        pyramid = self.extract_features(x)

        if pyramid.batch_size != x.shape[0]:
            raise FoundationComponentError(
                "Encoder changed the input batch size."
            )

        return pyramid


class HierarchicalFoundationEncoder(FoundationEncoder):
    """
    Dependency-free hierarchical encoder.

    This serves as the executable foundation-encoder reference implementation.
    A real DINOv2/remote-sensing foundation backend can later implement the
    same FoundationEncoder contract.
    """

    def __init__(
        self,
        *,
        base_channels: int = 32,
        pretrained: bool = False,
        checkpoint: str | None = None,
    ) -> None:
        if base_channels < 8:
            raise FoundationComponentError(
                "base_channels must be >= 8."
            )

        channels = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        )

        super().__init__(
            name="hierarchical_foundation_reference",
            feature_channels=channels,
            pretrained=pretrained,
            checkpoint=checkpoint,
        )

        self.stage1 = self._stage(3, channels[0], stride=2)
        self.stage2 = self._stage(channels[0], channels[1], stride=2)
        self.stage3 = self._stage(channels[1], channels[2], stride=2)
        self.stage4 = self._stage(channels[2], channels[3], stride=2)

    @staticmethod
    def _stage(
        in_channels: int,
        out_channels: int,
        *,
        stride: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def extract_features(self, x: torch.Tensor) -> FeaturePyramid:
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)

        return FeaturePyramid(
            features=(f1, f2, f3, f4)
        )


class GeospatialFeatureAdapter(nn.Module):
    """
    Projects heterogeneous encoder features into a common decoder dimension.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int = 128,
    ) -> None:
        super().__init__()

        channels = tuple(int(value) for value in in_channels)

        if not channels or any(value <= 0 for value in channels):
            raise FoundationComponentError(
                "in_channels must contain positive integers."
            )

        if out_channels <= 0:
            raise FoundationComponentError(
                "out_channels must be positive."
            )

        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        channels_in,
                        out_channels,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU(),
                )
                for channels_in in channels
            ]
        )

    def forward(
        self,
        pyramid: FeaturePyramid,
    ) -> tuple[torch.Tensor, ...]:
        if len(pyramid.features) != len(self.projections):
            raise FoundationComponentError(
                "Feature count does not match adapter configuration."
            )

        return tuple(
            projection(feature)
            for projection, feature in zip(
                self.projections,
                pyramid.features,
            )
        )


class MultiScaleGeospatialDecoder(nn.Module):
    """
    Fuses a feature pyramid and progressively reconstructs full resolution.
    """

    def __init__(
        self,
        channels: int,
        num_classes: int,
    ) -> None:
        super().__init__()

        if channels <= 0 or num_classes <= 0:
            raise FoundationComponentError(
                "channels and num_classes must be positive."
            )

        self.refine = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(
                channels,
                num_classes,
                kernel_size=1,
            ),
        )

    def forward(
        self,
        features: tuple[torch.Tensor, ...],
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        if not features:
            raise FoundationComponentError(
                "Decoder received no features."
            )

        target_size = features[0].shape[-2:]

        fused = features[0]

        for feature in features[1:]:
            upsampled = F.interpolate(
                feature,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
            fused = fused + upsampled

        logits = self.refine(fused)

        return F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )