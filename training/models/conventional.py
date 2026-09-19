from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from .base import DEFAULT_CLASS_IDS, ModelSpec, SegmentationModel


class ModelArchitectureError(ValueError):
    """Raised when a model architecture cannot be constructed."""


def _make_spec(
    *,
    model_id: str,
    architecture: str,
    encoder: str,
    model_version: str = "1.0.0",
    pretrained: bool = True,
    pretrained_source: str | None = "imagenet",
) -> ModelSpec:
    return ModelSpec(
        model_id=model_id,
        model_version=model_version,
        architecture=architecture,
        num_classes=len(DEFAULT_CLASS_IDS),
        in_channels=3,
        class_ids=DEFAULT_CLASS_IDS,
        input_size=None,
        pretrained_encoder=pretrained,
        pretrained_source=pretrained_source if pretrained else None,
    )


class SMPArchitecture(SegmentationModel):
    """
    Segmentation Models PyTorch adapter.

    Keeps the DRISHTI model contract independent of the third-party
    segmentation library.
    """

    def __init__(
        self,
        *,
        spec: ModelSpec,
        architecture_name: str,
        encoder_name: str,
        pretrained: bool,
    ) -> None:
        super().__init__(spec)

        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise ModelArchitectureError(
                "segmentation_models_pytorch is required for conventional "
                "architectures. Install the pinned DRISHTI ML dependencies."
            ) from exc

        architecture_cls = getattr(smp, architecture_name, None)
        if architecture_cls is None:
            raise ModelArchitectureError(
                f"SMP architecture '{architecture_name}' is unavailable."
            )

        encoder_weights = "imagenet" if pretrained else None

        try:
            self.network = architecture_cls(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=spec.in_channels,
                classes=spec.num_classes,
                activation=None,
            )
        except Exception as exc:
            raise ModelArchitectureError(
                f"Failed to construct {architecture_name} "
                f"with encoder '{encoder_name}'."
            ) from exc

    @property
    def encoder(self) -> nn.Module:
        encoder = getattr(self.network, "encoder", None)
        if encoder is None:
            raise ModelArchitectureError(
                f"Architecture '{self.spec.architecture}' does not expose "
                "an encoder."
            )
        return encoder

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        output = self.network(x)

        if isinstance(output, (tuple, list)):
            if len(output) != 1:
                raise ModelArchitectureError(
                    f"{self.spec.model_id} returned {len(output)} outputs; "
                    "DRISHTI requires one segmentation tensor."
                )
            output = output[0]

        return output


def deeplabv3plus_resnet50(
    *,
    pretrained: bool = True,
) -> SegmentationModel:
    spec = _make_spec(
        model_id="deeplabv3plus_resnet50",
        architecture="DeepLabV3Plus",
        encoder="resnet50",
        pretrained=pretrained,
    )
    return SMPArchitecture(
        spec=spec,
        architecture_name="DeepLabV3Plus",
        encoder_name="resnet50",
        pretrained=pretrained,
    )


def deeplabv3plus_resnet101(
    *,
    pretrained: bool = True,
) -> SegmentationModel:
    spec = _make_spec(
        model_id="deeplabv3plus_resnet101",
        architecture="DeepLabV3Plus",
        encoder="resnet101",
        pretrained=pretrained,
    )
    return SMPArchitecture(
        spec=spec,
        architecture_name="DeepLabV3Plus",
        encoder_name="resnet101",
        pretrained=pretrained,
    )


def unetplusplus_efficientnet_b4(
    *,
    pretrained: bool = True,
) -> SegmentationModel:
    spec = _make_spec(
        model_id="unetplusplus_efficientnet_b4",
        architecture="UnetPlusPlus",
        encoder="efficientnet-b4",
        pretrained=pretrained,
    )
    return SMPArchitecture(
        spec=spec,
        architecture_name="UnetPlusPlus",
        encoder_name="timm-efficientnet-b4",
        pretrained=pretrained,
    )


def fpn_resnet50(
    *,
    pretrained: bool = True,
) -> SegmentationModel:
    spec = _make_spec(
        model_id="fpn_resnet50",
        architecture="FPN",
        encoder="resnet50",
        pretrained=pretrained,
    )
    return SMPArchitecture(
        spec=spec,
        architecture_name="FPN",
        encoder_name="resnet50",
        pretrained=pretrained,
    )


CONVENTIONAL_FACTORIES: dict[str, Callable[..., SegmentationModel]] = {
    "deeplabv3plus_resnet50": deeplabv3plus_resnet50,
    "deeplabv3plus_resnet101": deeplabv3plus_resnet101,
    "unetplusplus_efficientnet_b4": unetplusplus_efficientnet_b4,
    "fpn_resnet50": fpn_resnet50,
}