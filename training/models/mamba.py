from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .base import DEFAULT_CLASS_IDS, ModelSpec, SegmentationModel


class MambaArchitectureError(ValueError):
    """Raised when the CNN-Mamba architecture is invalid."""


class ConvNormAct(nn.Module):
    """Convolution + normalization + activation block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
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
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SelectiveStateSpaceBlock(nn.Module):
    """
    Dependency-free selective state-space block.

    This is a Mamba-style sequence mixer implemented entirely in PyTorch.
    It provides:
      - input-dependent gating
      - recurrent state propagation
      - bidirectional spatial context
      - residual connection
      - normalization

    The implementation intentionally avoids mamba-ssm so that the model
    remains executable on CPU and Apple Silicon.
    """

    def __init__(
        self,
        dim: int,
        *,
        state_dim: int = 16,
        expansion: int = 2,
    ) -> None:
        super().__init__()

        if dim <= 0:
            raise MambaArchitectureError("dim must be positive.")
        if state_dim <= 0:
            raise MambaArchitectureError("state_dim must be positive.")
        if expansion <= 0:
            raise MambaArchitectureError("expansion must be positive.")

        hidden_dim = dim * expansion

        self.norm = nn.LayerNorm(dim)

        self.in_proj = nn.Linear(dim, hidden_dim * 2)

        self.state_projection = nn.Linear(
            hidden_dim,
            state_dim,
        )

        self.state_to_hidden = nn.Linear(
            state_dim,
            hidden_dim,
        )

        self.delta_projection = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.out_proj = nn.Linear(hidden_dim, dim)

        self.activation = nn.SiLU()

    def _scan(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sequential state propagation.

        x:
            [B, L, D]
        """

        batch, length, hidden_dim = x.shape

        state_input = self.state_projection(x)
        state = torch.zeros(
            batch,
            state_input.shape[-1],
            device=x.device,
            dtype=x.dtype,
        )

        outputs: list[torch.Tensor] = []

        for index in range(length):
            token = x[:, index]

            delta = torch.sigmoid(
                self.delta_projection(token)
            )

            candidate = state_input[:, index]

            state = (
                (1.0 - delta.mean(dim=-1, keepdim=True)) * state
                + delta.mean(dim=-1, keepdim=True) * candidate
            )

            state_features = self.state_to_hidden(state)

            outputs.append(
                token + state_features
            )

        return torch.stack(outputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.norm(x)

        projected = self.in_proj(x)

        value, gate = projected.chunk(2, dim=-1)

        value = self.activation(value)
        gate = torch.sigmoid(gate)

        forward = self._scan(value)

        backward = torch.flip(
            self._scan(torch.flip(value, dims=[1])),
            dims=[1],
        )

        mixed = 0.5 * (forward + backward)

        mixed = mixed * gate

        return residual + self.out_proj(mixed)


class SpatialMambaBlock(nn.Module):
    """
    Applies the state-space block over raster tokens.

    [B, C, H, W]
        ↓
    [B, H*W, C]
        ↓
    bidirectional state-space mixer
        ↓
    [B, C, H, W]
    """

    def __init__(
        self,
        channels: int,
        *,
        state_dim: int = 16,
    ) -> None:
        super().__init__()

        self.mixer = SelectiveStateSpaceBlock(
            channels,
            state_dim=state_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape

        tokens = x.flatten(2).transpose(1, 2)

        tokens = self.mixer(tokens)

        return tokens.transpose(1, 2).reshape(
            batch,
            channels,
            height,
            width,
        )


class CNNMambaEncoder(nn.Module):
    """Hierarchical CNN encoder followed by global state-space mixing."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        *,
        state_dim: int,
    ) -> None:
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.stage1 = nn.Sequential(
            ConvNormAct(in_channels, c1, stride=2),
            ConvNormAct(c1, c1),
        )

        self.stage2 = nn.Sequential(
            ConvNormAct(c1, c2, stride=2),
            ConvNormAct(c2, c2),
        )

        self.stage3 = nn.Sequential(
            ConvNormAct(c2, c3, stride=2),
            ConvNormAct(c3, c3),
        )

        self.global_mamba = SpatialMambaBlock(
            c3,
            state_dim=state_dim,
        )

        self.channels = (c1, c2, c3)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)

        f3 = self.global_mamba(f3)

        return f1, f2, f3


class MambaDecoder(nn.Module):
    """Lightweight multi-scale decoder."""

    def __init__(
        self,
        channels: tuple[int, int, int],
        num_classes: int,
    ) -> None:
        super().__init__()

        c1, c2, c3 = channels

        decoder_channels = c2

        self.deep = nn.Sequential(
            nn.Conv2d(c3, decoder_channels, 1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU(),
        )

        self.mid = nn.Sequential(
            nn.Conv2d(c2, decoder_channels, 1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU(),
        )

        self.low = nn.Sequential(
            nn.Conv2d(c1, decoder_channels // 2, 1, bias=False),
            nn.BatchNorm2d(decoder_channels // 2),
            nn.GELU(),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(
                decoder_channels + decoder_channels // 2,
                decoder_channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU(),
            nn.Conv2d(
                decoder_channels,
                num_classes,
                1,
            ),
        )

    def forward(
        self,
        f1: torch.Tensor,
        f2: torch.Tensor,
        f3: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        deep = self.deep(f3)

        deep = F.interpolate(
            deep,
            size=f2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        mid = self.mid(f2)

        merged = deep + mid

        merged = F.interpolate(
            merged,
            size=f1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        low = self.low(f1)

        merged = torch.cat(
            [merged, low],
            dim=1,
        )

        logits = self.refine(merged)

        return F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )


class CNNMamba(SegmentationModel):
    """
    DRISHTI M5.

    CNN local representation
            +
    bidirectional state-space global representation
            +
    multi-scale decoder.
    """

    def __init__(
        self,
        *,
        base_channels: int = 32,
        state_dim: int = 16,
        model_version: str = "1.0.0",
    ) -> None:
        spec = ModelSpec(
            model_id="cnn_mamba",
            model_version=model_version,
            architecture="CNN-Mamba",
            num_classes=len(DEFAULT_CLASS_IDS),
            in_channels=3,
            class_ids=DEFAULT_CLASS_IDS,
            input_size=None,
            pretrained_encoder=False,
            pretrained_source=None,
        )

        super().__init__(spec)

        if base_channels < 8:
            raise MambaArchitectureError(
                "base_channels must be >= 8."
            )

        self.encoder = CNNMambaEncoder(
            in_channels=spec.in_channels,
            base_channels=base_channels,
            state_dim=state_dim,
        )

        self.decoder = MambaDecoder(
            self.encoder.channels,
            spec.num_classes,
        )

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        original_size = x.shape[-2:]

        f1, f2, f3 = self.encoder(x)

        return self.decoder(
            f1,
            f2,
            f3,
            original_size,
        )


def cnn_mamba(
    *,
    base_channels: int = 32,
    state_dim: int = 16,
) -> SegmentationModel:
    """Factory for the DRISHTI CNN-Mamba candidate."""

    return CNNMamba(
        base_channels=base_channels,
        state_dim=state_dim,
    )


MAMBA_FACTORIES = {
    "cnn_mamba": cnn_mamba,
}