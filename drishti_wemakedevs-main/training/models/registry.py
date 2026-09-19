"""
DRISHTI segmentation model registry.

The registry provides a single architecture-independent construction
interface for the training and inference layers.

Architecture implementations are loaded lazily so importing DRISHTI
does not require every optional model dependency to be installed.

Current candidate architectures:
    - deeplabv3plus_resnet50
    - deeplabv3plus_resnet101
    - unetplusplus_efficientnet_b4
    - fpn_resnet50

All candidates use:
    in_channels = 3
    classes = 7

The registry is deliberately explicit. Unknown architecture names,
duplicate registrations, malformed factories, and incompatible model
objects fail immediately.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from torch import nn

from training.models.base import (
    DEFAULT_CLASS_IDS,
    ModelContractError,
    ModelSpec,
    SegmentationModel,
)


class ModelRegistryError(RuntimeError):
    """Raised for model registry failures."""


ModelFactory = Callable[..., SegmentationModel]


@dataclass(frozen=True)
class RegisteredModel:
    """Immutable registry entry."""

    name: str
    description: str
    factory: ModelFactory

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ModelRegistryError(
                "Registered model name must be non-empty."
            )

        if not callable(self.factory):
            raise ModelRegistryError(
                f"Factory for '{self.name}' is not callable."
            )


class _SMPModel(SegmentationModel):
    """
    Thin adapter around segmentation_models_pytorch.

    The third-party architecture is intentionally isolated here.
    Training code interacts only with SegmentationModel.
    """

    def __init__(
        self,
        *,
        spec: ModelSpec,
        network: nn.Module,
    ) -> None:
        super().__init__(spec)
        self.network = network

    def _forward_impl(self, x):
        return self.network(x)

    @property
    def encoder(self):
        return getattr(
            self.network,
            "encoder",
            None,
        )


def _require_smp():
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ModelRegistryError(
            "segmentation_models_pytorch is required for the "
            "registered DRISHTI architectures. Install the project "
            "ML dependencies before constructing these models."
        ) from exc

    return smp


def _build_smp(
    *,
    architecture: str,
    model_id: str,
    model_version: str,
    encoder_name: str,
    encoder_weights: str | None,
    in_channels: int,
    num_classes: int,
    class_ids: tuple[int, ...],
    input_size: int | None,
) -> SegmentationModel:
    if in_channels <= 0:
        raise ModelRegistryError(
            "in_channels must be > 0."
        )

    if num_classes <= 0:
        raise ModelRegistryError(
            "num_classes must be > 0."
        )

    smp = _require_smp()

    if architecture == "deeplabv3plus":
        network = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
        )

    elif architecture == "unetplusplus":
        network = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
        )

    elif architecture == "fpn":
        network = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
        )

    else:
        raise ModelRegistryError(
            f"Unsupported SMP architecture: {architecture}"
        )

    spec = ModelSpec(
        model_id=model_id,
        model_version=model_version,
        architecture=architecture,
        num_classes=num_classes,
        in_channels=in_channels,
        class_ids=class_ids,
        input_size=input_size,
        pretrained_encoder=encoder_weights is not None,
        pretrained_source=encoder_weights,
    )

    return _SMPModel(
        spec=spec,
        network=network,
    )


def _deeplabv3plus_resnet50(
    *,
    model_id: str = "drishti_deeplabv3plus_r50",
    model_version: str = "0.1.0",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
    num_classes: int = 7,
    class_ids: tuple[int, ...] = DEFAULT_CLASS_IDS,
    input_size: int | None = None,
    **_: Any,
) -> SegmentationModel:
    return _build_smp(
        architecture="deeplabv3plus",
        model_id=model_id,
        model_version=model_version,
        encoder_name="resnet50",
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        num_classes=num_classes,
        class_ids=class_ids,
        input_size=input_size,
    )


def _deeplabv3plus_resnet101(
    *,
    model_id: str = "drishti_deeplabv3plus_r101",
    model_version: str = "0.1.0",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
    num_classes: int = 7,
    class_ids: tuple[int, ...] = DEFAULT_CLASS_IDS,
    input_size: int | None = None,
    **_: Any,
) -> SegmentationModel:
    return _build_smp(
        architecture="deeplabv3plus",
        model_id=model_id,
        model_version=model_version,
        encoder_name="resnet101",
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        num_classes=num_classes,
        class_ids=class_ids,
        input_size=input_size,
    )


def _unetplusplus_efficientnet_b4(
    *,
    model_id: str = "drishti_unetplusplus_effb4",
    model_version: str = "0.1.0",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
    num_classes: int = 7,
    class_ids: tuple[int, ...] = DEFAULT_CLASS_IDS,
    input_size: int | None = None,
    **_: Any,
) -> SegmentationModel:
    return _build_smp(
        architecture="unetplusplus",
        model_id=model_id,
        model_version=model_version,
        encoder_name="timm-efficientnet-b4",
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        num_classes=num_classes,
        class_ids=class_ids,
        input_size=input_size,
    )


def _fpn_resnet50(
    *,
    model_id: str = "drishti_fpn_r50",
    model_version: str = "0.1.0",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
    num_classes: int = 7,
    class_ids: tuple[int, ...] = DEFAULT_CLASS_IDS,
    input_size: int | None = None,
    **_: Any,
) -> SegmentationModel:
    return _build_smp(
        architecture="fpn",
        model_id=model_id,
        model_version=model_version,
        encoder_name="resnet50",
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        num_classes=num_classes,
        class_ids=class_ids,
        input_size=input_size,
    )


class ModelRegistry:
    """
    Explicit registry of segmentation architectures.

    The registry is instance-based so tests and future deployments can
    maintain isolated registries rather than mutating global state.
    """

    def __init__(
        self,
        *,
        include_builtin: bool = True,
    ) -> None:
        self._models: dict[str, RegisteredModel] = {}

        if include_builtin:
            self._register_builtins()

    def _register_builtins(self) -> None:
        self.register(
            name="deeplabv3plus_resnet50",
            factory=_deeplabv3plus_resnet50,
            description=(
                "DeepLabV3+ with a ResNet-50 encoder."
            ),
        )

        self.register(
            name="deeplabv3plus_resnet101",
            factory=_deeplabv3plus_resnet101,
            description=(
                "DeepLabV3+ with a ResNet-101 encoder."
            ),
        )

        self.register(
            name="unetplusplus_efficientnet_b4",
            factory=_unetplusplus_efficientnet_b4,
            description=(
                "UNet++ with an EfficientNet-B4 encoder."
            ),
        )

        self.register(
            name="fpn_resnet50",
            factory=_fpn_resnet50,
            description=(
                "Feature Pyramid Network with a ResNet-50 encoder."
            ),
        )

    def register(
        self,
        *,
        name: str,
        factory: ModelFactory,
        description: str,
        overwrite: bool = False,
    ) -> None:
        """
        Register a model factory.

        Duplicate registration is rejected unless overwrite=True.
        """
        if not name.strip():
            raise ModelRegistryError(
                "Model name must be non-empty."
            )

        if not callable(factory):
            raise ModelRegistryError(
                f"Factory for '{name}' is not callable."
            )

        if not description.strip():
            raise ModelRegistryError(
                "Model description must be non-empty."
            )

        if name in self._models and not overwrite:
            raise ModelRegistryError(
                f"Model '{name}' is already registered."
            )

        self._models[name] = RegisteredModel(
            name=name,
            description=description,
            factory=factory,
        )

    def unregister(self, name: str) -> None:
        """Remove a registered architecture."""
        if name not in self._models:
            raise ModelRegistryError(
                f"Model '{name}' is not registered."
            )

        del self._models[name]

    def contains(self, name: str) -> bool:
        """Return whether a model name is registered."""
        return name in self._models

    def names(self) -> tuple[str, ...]:
        """Return registered names in deterministic order."""
        return tuple(sorted(self._models))

    def describe(self, name: str) -> Mapping[str, str]:
        """Return human-readable metadata for a registered model."""
        entry = self._models.get(name)

        if entry is None:
            raise ModelRegistryError(
                f"Unknown model architecture: '{name}'."
            )

        return {
            "name": entry.name,
            "description": entry.description,
        }

    def build(
        self,
        name: str,
        **kwargs: Any,
    ) -> SegmentationModel:
        """
        Construct a registered segmentation model.

        The returned object must satisfy the SegmentationModel
        contract. Invalid factories fail immediately rather than
        allowing incompatible objects into the trainer.
        """
        entry = self._models.get(name)

        if entry is None:
            available = ", ".join(self.names())

            raise ModelRegistryError(
                f"Unknown model architecture: '{name}'. "
                f"Available: [{available}]"
            )

        try:
            model = entry.factory(**kwargs)
        except ModelRegistryError:
            raise
        except Exception as exc:
            raise ModelRegistryError(
                f"Failed to construct model '{name}': "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if not isinstance(model, SegmentationModel):
            raise ModelContractError(
                f"Factory '{name}' returned "
                f"{type(model).__name__}, expected "
                "SegmentationModel."
            )

        return model

    def __len__(self) -> int:
        return len(self._models)


DEFAULT_MODEL_REGISTRY = ModelRegistry(
    include_builtin=True,
)


def build_model(
    name: str,
    **kwargs: Any,
) -> SegmentationModel:
    """
    Convenience API backed by the default registry.
    """
    return DEFAULT_MODEL_REGISTRY.build(
        name,
        **kwargs,
    )


__all__ = [
    "DEFAULT_MODEL_REGISTRY",
    "ModelFactory",
    "ModelRegistry",
    "ModelRegistryError",
    "RegisteredModel",
    "build_model",
]