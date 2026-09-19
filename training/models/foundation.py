from __future__ import annotations

from torch import Tensor

from .base import DEFAULT_CLASS_IDS, ModelSpec, SegmentationModel
from .foundation_components import (
    FoundationComponentError,
    GeospatialFeatureAdapter,
    HierarchicalFoundationEncoder,
    MultiScaleGeospatialDecoder,
)


class FoundationArchitectureError(ValueError):
    """Raised when the foundation segmentation model is invalid."""


class FoundationGeospatialModel(SegmentationModel):
    """
    M6: Foundation Encoder + Geospatial Decoder.

    The encoder is intentionally separated from the decoder so that a real
    pretrained foundation model can replace the reference encoder without
    changing the segmentation contract.
    """

    def __init__(
        self,
        *,
        base_channels: int = 32,
        decoder_channels: int = 128,
        pretrained: bool = False,
        checkpoint: str | None = None,
        model_version: str = "1.0.0",
    ) -> None:
        if pretrained and not checkpoint:
            raise FoundationArchitectureError(
                "checkpoint is required when pretrained=True."
            )

        spec = ModelSpec(
            model_id="foundation_geospatial",
            model_version=model_version,
            architecture="FoundationEncoder-GeospatialDecoder",
            num_classes=len(DEFAULT_CLASS_IDS),
            in_channels=3,
            class_ids=DEFAULT_CLASS_IDS,
            input_size=None,
            pretrained_encoder=pretrained,
            pretrained_source=checkpoint,
        )

        super().__init__(spec)

        try:
            self.foundation_encoder = HierarchicalFoundationEncoder(
                base_channels=base_channels,
                pretrained=pretrained,
                checkpoint=checkpoint,
            )
        except FoundationComponentError as exc:
            raise FoundationArchitectureError(str(exc)) from exc

        self.feature_adapter = GeospatialFeatureAdapter(
            self.foundation_encoder.feature_channels,
            out_channels=decoder_channels,
        )

        self.decoder = MultiScaleGeospatialDecoder(
            channels=decoder_channels,
            num_classes=spec.num_classes,
        )

    @property
    def encoder(self):
        return self.foundation_encoder

    def _forward_impl(self, x: Tensor) -> Tensor:
        pyramid = self.foundation_encoder(x)

        adapted = self.feature_adapter(pyramid)

        return self.decoder(
            adapted,
            output_size=x.shape[-2:],
        )

    def freeze_encoder(self) -> None:
        for parameter in self.foundation_encoder.parameters():
            parameter.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for parameter in self.foundation_encoder.parameters():
            parameter.requires_grad = True


def foundation_geospatial(
    *,
    base_channels: int = 32,
    decoder_channels: int = 128,
    pretrained: bool = False,
    checkpoint: str | None = None,
) -> SegmentationModel:
    """Factory for the M6 foundation-encoder candidate."""

    return FoundationGeospatialModel(
        base_channels=base_channels,
        decoder_channels=decoder_channels,
        pretrained=pretrained,
        checkpoint=checkpoint,
    )