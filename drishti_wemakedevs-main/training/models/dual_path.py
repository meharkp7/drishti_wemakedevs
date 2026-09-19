from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .base import DEFAULT_CLASS_IDS, ModelSpec, SegmentationModel
from .frequency import FrequencyFeatureExtractor


class DualPathArchitectureError(ValueError):
    """Raised for invalid DualPathMamba configurations."""


class ConvBlock(nn.Module):
    """Residual convolutional block for local spatial representation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.BatchNorm2d(out_channels)

        self.activation = nn.GELU()

        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)

        x = self.activation(
            self.norm1(self.conv1(x))
        )

        x = self.norm2(
            self.conv2(x)
        )

        return self.activation(x + residual)


class SpatialStateMixer(nn.Module):
    """
    Pure-PyTorch bidirectional spatial state mixer.

    This deliberately avoids an external CUDA-specific Mamba dependency.
    """

    def __init__(
        self,
        channels: int,
        *,
        state_dim: int = 16,
    ) -> None:
        super().__init__()

        if channels <= 0 or state_dim <= 0:
            raise DualPathArchitectureError(
                "channels and state_dim must be positive."
            )

        self.norm = nn.LayerNorm(channels)

        self.input_projection = nn.Linear(
            channels,
            channels * 2,
        )

        self.state_projection = nn.Linear(
            channels,
            state_dim,
        )

        self.state_reconstruction = nn.Linear(
            state_dim,
            channels,
        )

        self.gate_projection = nn.Linear(
            channels,
            channels,
        )

        self.output_projection = nn.Linear(
            channels,
            channels,
        )

    def _scan(
        self,
        sequence: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, channels = sequence.shape

        projected_state = self.state_projection(
            sequence
        )

        state_dim = projected_state.shape[-1]

        state = torch.zeros(
            batch,
            state_dim,
            device=sequence.device,
            dtype=sequence.dtype,
        )

        outputs = []

        for index in range(length):
            token = sequence[:, index]

            update = torch.sigmoid(
                token.mean(
                    dim=-1,
                    keepdim=True,
                )
            )

            state = (
                (1.0 - update) * state
                + update * projected_state[:, index]
            )

            reconstructed = self.state_reconstruction(
                state
            )

            outputs.append(
                token + reconstructed
            )

        return torch.stack(outputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.norm(x)

        value, gate = self.input_projection(x).chunk(
            2,
            dim=-1,
        )

        value = F.silu(value)
        gate = torch.sigmoid(
            self.gate_projection(gate)
        )

        forward = self._scan(value)

        backward = torch.flip(
            self._scan(
                torch.flip(value, dims=[1])
            ),
            dims=[1],
        )

        mixed = 0.5 * (forward + backward)

        mixed = mixed * gate

        mixed = self.output_projection(mixed)

        return residual + mixed


class SpatialMambaPath(nn.Module):
    """Global-context path based on hierarchical CNN + state mixing."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        *,
        state_dim: int,
    ) -> None:
        super().__init__()

        self.stage1 = ConvBlock(
            in_channels,
            base_channels,
            stride=2,
        )

        self.stage2 = ConvBlock(
            base_channels,
            base_channels * 2,
            stride=2,
        )

        self.stage3 = ConvBlock(
            base_channels * 2,
            base_channels * 4,
            stride=2,
        )

        self.mixer = SpatialStateMixer(
            base_channels * 4,
            state_dim=state_dim,
        )

        self.channels = base_channels * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        batch, channels, height, width = x.shape

        tokens = x.flatten(2).transpose(1, 2)

        tokens = self.mixer(tokens)

        return tokens.transpose(1, 2).reshape(
            batch,
            channels,
            height,
            width,
        )


class LocalCNNPath(nn.Module):
    """High-resolution local-detail path."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
    ) -> None:
        super().__init__()

        self.stage1 = ConvBlock(
            in_channels,
            base_channels,
            stride=2,
        )

        self.stage2 = ConvBlock(
            base_channels,
            base_channels * 2,
            stride=2,
        )

        self.stage3 = ConvBlock(
            base_channels * 2,
            base_channels * 4,
            stride=2,
        )

        self.channels = base_channels * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        return self.stage3(x)


class CrossPathFusion(nn.Module):
    """
    Adaptive fusion of local and global representations.

    A learned gate determines how much information is retained from each
    path for every spatial location.
    """

    def __init__(
        self,
        channels: int,
    ) -> None:
        super().__init__()

        self.gate = nn.Sequential(
            nn.Conv2d(
                channels * 2,
                channels,
                kernel_size=1,
            ),
            nn.Sigmoid(),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(
                channels * 2,
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(
        self,
        local: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        if local.shape[-2:] != global_features.shape[-2:]:
            global_features = F.interpolate(
                global_features,
                size=local.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        combined = torch.cat(
            [local, global_features],
            dim=1,
        )

        gate = self.gate(combined)

        gated_local = local * gate
        gated_global = global_features * (1.0 - gate)

        return self.projection(
            torch.cat(
                [gated_local, gated_global],
                dim=1,
            )
        )


class DualPathDecoder(nn.Module):
    """Multi-stage decoder returning full-resolution segmentation logits."""

    def __init__(
        self,
        channels: int,
        num_classes: int,
    ) -> None:
        super().__init__()

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
                channels // 2,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels // 2),
            nn.GELU(),
            nn.Conv2d(
                channels // 2,
                num_classes,
                kernel_size=1,
            ),
        )

    def forward(
        self,
        features: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        logits = self.refine(features)

        return F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )


class DualPathMamba(SegmentationModel):
    """
    DRISHTI M7.

    Local CNN representation
        +
    global bidirectional state-space representation
        +
    adaptive cross-path fusion
        +
    optional frequency-domain representation.
    """

    def __init__(
        self,
        *,
        base_channels: int = 24,
        state_dim: int = 16,
        frequency_fusion: bool = True,
        frequency_channels: int | None = None,
        model_version: str = "1.0.0",
    ) -> None:
        spec = ModelSpec(
            model_id="dual_path_mamba",
            model_version=model_version,
            architecture=(
                "DualPath-CNN-Mamba-Frequency"
                if frequency_fusion
                else "DualPath-CNN-Mamba"
            ),
            num_classes=len(DEFAULT_CLASS_IDS),
            in_channels=3,
            class_ids=DEFAULT_CLASS_IDS,
            input_size=None,
            pretrained_encoder=False,
            pretrained_source=None,
        )

        super().__init__(spec)

        if base_channels < 8:
            raise DualPathArchitectureError(
                "base_channels must be >= 8."
            )

        if frequency_channels is None:
            frequency_channels = base_channels * 4

        if frequency_channels <= 0:
            raise DualPathArchitectureError(
                "frequency_channels must be positive."
            )

        self.frequency_fusion_enabled = frequency_fusion

        self.local_path = LocalCNNPath(
            spec.in_channels,
            base_channels,
        )

        self.global_path = SpatialMambaPath(
            spec.in_channels,
            base_channels,
            state_dim=state_dim,
        )

        feature_channels = base_channels * 4

        self.cross_fusion = CrossPathFusion(
            feature_channels
        )

        if frequency_fusion:
            self.frequency_path = FrequencyFeatureExtractor(
                in_channels=spec.in_channels,
                out_channels=frequency_channels,
            )

            self.frequency_projection = nn.Sequential(
                nn.Conv2d(
                    frequency_channels,
                    feature_channels,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(feature_channels),
                nn.GELU(),
            )

            self.final_fusion = nn.Sequential(
                nn.Conv2d(
                    feature_channels * 2,
                    feature_channels,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(feature_channels),
                nn.GELU(),
            )
        else:
            self.frequency_path = None
            self.frequency_projection = None
            self.final_fusion = nn.Identity()

        self.decoder = DualPathDecoder(
            feature_channels,
            spec.num_classes,
        )

    def _forward_impl(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        output_size = x.shape[-2:]

        local = self.local_path(x)
        global_features = self.global_path(x)

        fused = self.cross_fusion(
            local,
            global_features,
        )

        if self.frequency_fusion_enabled:
            assert self.frequency_path is not None
            assert self.frequency_projection is not None

            frequency = self.frequency_path(x)

            frequency = F.interpolate(
                frequency,
                size=fused.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            frequency = self.frequency_projection(
                frequency
            )

            fused = self.final_fusion(
                torch.cat(
                    [fused, frequency],
                    dim=1,
                )
            )

        return self.decoder(
            fused,
            output_size,
        )


def dual_path_mamba(
    *,
    base_channels: int = 24,
    state_dim: int = 16,
    frequency_fusion: bool = True,
    frequency_channels: int | None = None,
) -> SegmentationModel:
    """Factory for the DRISHTI M7 research candidate."""

    return DualPathMamba(
        base_channels=base_channels,
        state_dim=state_dim,
        frequency_fusion=frequency_fusion,
        frequency_channels=frequency_channels,
    )


DUAL_PATH_FACTORIES = {
    "dual_path_mamba": dual_path_mamba,
}