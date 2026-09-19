from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FrequencyComponentError(ValueError):
    """Raised for invalid frequency-domain components."""


class HaarWaveletTransform(nn.Module):
    """
    Lightweight 2D Haar wavelet decomposition.

    For every input channel the transform produces:
        LL: low-low
        LH: low-high
        HL: high-low
        HH: high-high

    The transform is implemented with fixed analytical filters and does not
    introduce trainable parameters.
    """

    def __init__(self) -> None:
        super().__init__()

        sqrt_half = 2.0 ** -0.5

        ll = torch.tensor(
            [[1.0, 1.0], [1.0, 1.0]],
            dtype=torch.float32,
        ) * 0.5

        lh = torch.tensor(
            [[-1.0, -1.0], [1.0, 1.0]],
            dtype=torch.float32,
        ) * 0.5

        hl = torch.tensor(
            [[-1.0, 1.0], [-1.0, 1.0]],
            dtype=torch.float32,
        ) * 0.5

        hh = torch.tensor(
            [[1.0, -1.0], [-1.0, 1.0]],
            dtype=torch.float32,
        ) * 0.5

        filters = torch.stack(
            [ll, lh, hl, hh],
            dim=0,
        ).unsqueeze(1)

        self.register_buffer(
            "filters",
            filters,
            persistent=False,
        )

        self._normalization = sqrt_half

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if not isinstance(x, torch.Tensor):
            raise FrequencyComponentError("Input must be a Tensor.")

        if x.ndim != 4:
            raise FrequencyComponentError(
                "Input must have shape [B,C,H,W]."
            )

        if not torch.is_floating_point(x):
            raise FrequencyComponentError(
                "Input must be floating point."
            )

        if not torch.isfinite(x).all():
            raise FrequencyComponentError(
                "Input contains non-finite values."
            )

        _, channels, height, width = x.shape

        if height < 2 or width < 2:
            raise FrequencyComponentError(
                "Spatial dimensions must be at least 2."
            )

        # Replicate the analytical filters independently for every channel.
        filters = self.filters.to(
            device=x.device,
            dtype=x.dtype,
        ).repeat(channels, 1, 1, 1)

        transformed = F.conv2d(
            x,
            filters,
            stride=2,
            groups=channels,
        )

        # [B, 4C, H/2, W/2] -> four [B,C,H/2,W/2] components.
        components = transformed.chunk(4, dim=1)

        return tuple(
            component * self._normalization
            for component in components
        )


class FrequencyFeatureExtractor(nn.Module):
    """
    Learnable frequency feature extractor.

    Wavelet coefficients are first projected into a compact representation,
    then fused with a residual spatial projection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise FrequencyComponentError(
                "in_channels must be positive."
            )

        if out_channels <= 0:
            raise FrequencyComponentError(
                "out_channels must be positive."
            )

        self.wavelet = HaarWaveletTransform()

        self.wavelet_projection = nn.Sequential(
            nn.Conv2d(
                in_channels * 4,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        self.spatial_projection = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(
                out_channels * 2,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low_low, low_high, high_low, high_high = self.wavelet(x)

        wavelet_features = torch.cat(
            [
                low_low,
                low_high,
                high_low,
                high_high,
            ],
            dim=1,
        )

        frequency = self.wavelet_projection(
            wavelet_features
        )

        spatial = self.spatial_projection(x)

        # Haar decomposition naturally follows floor(H/2), floor(W/2),
        # while the strided convolution follows ceil(H/2), ceil(W/2).
        # Align both branches explicitly before fusion so odd-sized inputs
        # remain fully supported.
        if frequency.shape[-2:] != spatial.shape[-2:]:
            frequency = F.interpolate(
                frequency,
                size=spatial.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        fused = torch.cat(
            [frequency, spatial],
            dim=1,
        )

        return self.fusion(fused)