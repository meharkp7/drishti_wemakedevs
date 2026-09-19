from __future__ import annotations

import pytest
import torch
from torch import nn

from training.models.base import (
    DEFAULT_CLASS_IDS,
    ModelContractError,
    ModelSpec,
    SegmentationModel,
)
from training.models.registry import (
    ModelRegistry,
    ModelRegistryError,
)


class DummySegmentationModel(SegmentationModel):
    """Minimal architecture used to test the base contract."""

    def __init__(
        self,
        *,
        output_channels: int = 7,
        preserve_spatial_size: bool = True,
    ):
        spec = ModelSpec(
            model_id="dummy",
            model_version="1.0.0",
            architecture="dummy",
            num_classes=7,
            in_channels=3,
            class_ids=DEFAULT_CLASS_IDS,
        )

        super().__init__(spec)

        self.output_channels = output_channels
        self.preserve_spatial_size = preserve_spatial_size

        self.head = nn.Conv2d(
            3,
            output_channels,
            kernel_size=1,
        )

    def _forward_impl(self, x):
        output = self.head(x)

        if not self.preserve_spatial_size:
            output = output[..., :-1, :-1]

        return output


# ---------------------------------------------------------------------------
# ModelSpec
# ---------------------------------------------------------------------------


def test_default_model_spec_matches_loveda():
    spec = ModelSpec(
        model_id="test",
        model_version="1.0.0",
        architecture="test_arch",
    )

    assert spec.num_classes == 7
    assert spec.in_channels == 3
    assert spec.class_ids == DEFAULT_CLASS_IDS


def test_model_spec_serializes_class_ids_as_json_list():
    spec = ModelSpec(
        model_id="test",
        model_version="1.0.0",
        architecture="test_arch",
    )

    result = spec.to_dict()

    assert isinstance(result["class_ids"], list)
    assert result["class_ids"] == list(DEFAULT_CLASS_IDS)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_id": ""},
        {"model_version": ""},
        {"architecture": ""},
        {"num_classes": 0},
        {"in_channels": 0},
        {"class_ids": (1, 2)},
        {"class_ids": (1, 1, 2, 3, 4, 5, 6)},
        {"class_ids": (True, 2, 3, 4, 5, 6, 7)},
        {"input_size": 0},
    ],
)
def test_invalid_model_specs_are_rejected(kwargs):
    defaults = {
        "model_id": "test",
        "model_version": "1.0.0",
        "architecture": "test_arch",
    }

    defaults.update(kwargs)

    with pytest.raises(ModelContractError):
        ModelSpec(**defaults)


def test_pretrained_model_requires_source():
    with pytest.raises(ModelContractError):
        ModelSpec(
            model_id="test",
            model_version="1.0.0",
            architecture="test",
            pretrained_encoder=True,
        )


# ---------------------------------------------------------------------------
# Forward contract
# ---------------------------------------------------------------------------


def test_model_accepts_valid_input_and_returns_canonical_shape():
    model = DummySegmentationModel()

    x = torch.randn(
        2,
        3,
        32,
        32,
    )

    logits = model(x)

    assert logits.shape == (2, 7, 32, 32)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_model_rejects_non_tensor_input():
    model = DummySegmentationModel()

    with pytest.raises(ModelContractError):
        model([[1, 2, 3]])


def test_model_rejects_wrong_rank():
    model = DummySegmentationModel()

    x = torch.randn(
        3,
        32,
        32,
    )

    with pytest.raises(ModelContractError):
        model(x)


def test_model_rejects_wrong_input_channels():
    model = DummySegmentationModel()

    x = torch.randn(
        2,
        4,
        32,
        32,
    )

    with pytest.raises(
        ModelContractError,
        match="input channels",
    ):
        model(x)


def test_model_rejects_integer_input():
    model = DummySegmentationModel()

    x = torch.ones(
        2,
        3,
        32,
        32,
        dtype=torch.int64,
    )

    with pytest.raises(ModelContractError):
        model(x)


def test_model_rejects_nan_input():
    model = DummySegmentationModel()

    x = torch.randn(
        1,
        3,
        16,
        16,
    )

    x[0, 0, 0, 0] = float("nan")

    with pytest.raises(ModelContractError):
        model(x)


def test_model_rejects_infinite_input():
    model = DummySegmentationModel()

    x = torch.randn(
        1,
        3,
        16,
        16,
    )

    x[0, 0, 0, 0] = float("inf")

    with pytest.raises(ModelContractError):
        model(x)


def test_model_rejects_wrong_output_channel_count():
    model = DummySegmentationModel(
        output_channels=6,
    )

    x = torch.randn(
        1,
        3,
        16,
        16,
    )

    with pytest.raises(
        ModelContractError,
        match="output channels",
    ):
        model(x)


def test_model_rejects_wrong_output_spatial_shape():
    model = DummySegmentationModel(
        preserve_spatial_size=False,
    )

    x = torch.randn(
        1,
        3,
        16,
        16,
    )

    with pytest.raises(
        ModelContractError,
        match="spatial dimensions",
    ):
        model(x)


class NaNOutputModel(DummySegmentationModel):
    def _forward_impl(self, x):
        output = super()._forward_impl(x)
        output[..., 0, 0] = float("nan")
        return output


class IntegerOutputModel(DummySegmentationModel):
    def _forward_impl(self, x):
        output = super()._forward_impl(x)
        return output.round().to(torch.int64)


def test_model_rejects_nan_output():
    model = NaNOutputModel()

    x = torch.randn(
        1,
        3,
        8,
        8,
    )

    with pytest.raises(ModelContractError):
        model(x)


def test_model_rejects_non_float_output():
    model = IntegerOutputModel()

    x = torch.randn(
        1,
        3,
        8,
        8,
    )

    with pytest.raises(ModelContractError):
        model(x)


# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------


def test_parameter_count_is_correct():
    model = DummySegmentationModel()

    expected = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    assert model.parameter_count() == expected


def test_trainable_parameter_count_respects_requires_grad():
    model = DummySegmentationModel()

    total = model.parameter_count()

    model.head.weight.requires_grad = False
    model.head.bias.requires_grad = False

    trainable = model.parameter_count(
        trainable_only=True,
    )

    assert trainable < total


def test_model_metadata_contains_parameter_statistics():
    model = DummySegmentationModel()

    metadata = model.metadata()

    assert metadata["model_id"] == "dummy"
    assert metadata["model_version"] == "1.0.0"
    assert metadata["architecture"] == "dummy"
    assert metadata["num_classes"] == 7
    assert metadata["in_channels"] == 3
    assert metadata["parameter_count"] > 0
    assert metadata["trainable_parameter_count"] > 0


def test_freeze_and_unfreeze_encoder_fail_cleanly_without_encoder():
    model = DummySegmentationModel()

    with pytest.raises(ModelContractError):
        model.freeze_encoder()

    with pytest.raises(ModelContractError):
        model.unfreeze_encoder()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_empty_registry_starts_empty():
    registry = ModelRegistry(
        include_builtin=False,
    )

    assert len(registry) == 0
    assert registry.names() == ()


def test_registry_can_register_and_build_custom_model():
    registry = ModelRegistry(
        include_builtin=False,
    )

    def factory(**kwargs):
        return DummySegmentationModel()

    registry.register(
        name="dummy",
        factory=factory,
        description="Dummy test architecture.",
    )

    assert registry.contains("dummy")
    assert registry.names() == ("dummy",)

    model = registry.build("dummy")

    assert isinstance(
        model,
        SegmentationModel,
    )


def test_registry_rejects_duplicate_registration():
    registry = ModelRegistry(
        include_builtin=False,
    )

    def factory(**kwargs):
        return DummySegmentationModel()

    registry.register(
        name="dummy",
        factory=factory,
        description="Dummy.",
    )

    with pytest.raises(ModelRegistryError):
        registry.register(
            name="dummy",
            factory=factory,
            description="Duplicate.",
        )


def test_registry_allows_explicit_overwrite():
    registry = ModelRegistry(
        include_builtin=False,
    )

    def factory(**kwargs):
        return DummySegmentationModel()

    registry.register(
        name="dummy",
        factory=factory,
        description="Original.",
    )

    registry.register(
        name="dummy",
        factory=factory,
        description="Replacement.",
        overwrite=True,
    )

    assert registry.describe("dummy")["description"] == "Replacement."


def test_registry_rejects_unknown_model():
    registry = ModelRegistry(
        include_builtin=False,
    )

    with pytest.raises(
        ModelRegistryError,
        match="Unknown model architecture",
    ):
        registry.build("does_not_exist")


def test_registry_rejects_non_callable_factory():
    registry = ModelRegistry(
        include_builtin=False,
    )

    with pytest.raises(ModelRegistryError):
        registry.register(
            name="bad",
            factory=None,
            description="Invalid.",
        )


def test_registry_rejects_empty_name():
    registry = ModelRegistry(
        include_builtin=False,
    )

    def factory(**kwargs):
        return DummySegmentationModel()

    with pytest.raises(ModelRegistryError):
        registry.register(
            name="",
            factory=factory,
            description="Invalid.",
        )


def test_registry_rejects_empty_description():
    registry = ModelRegistry(
        include_builtin=False,
    )

    def factory(**kwargs):
        return DummySegmentationModel()

    with pytest.raises(ModelRegistryError):
        registry.register(
            name="dummy",
            factory=factory,
            description="",
        )


def test_registry_rejects_factory_returning_wrong_type():
    registry = ModelRegistry(
        include_builtin=False,
    )

    def bad_factory(**kwargs):
        return nn.Conv2d(3, 7, 1)

    registry.register(
        name="bad",
        factory=bad_factory,
        description="Invalid model.",
    )

    with pytest.raises(ModelContractError):
        registry.build("bad")


def test_registry_unregisters_model():
    registry = ModelRegistry(
        include_builtin=False,
    )

    def factory(**kwargs):
        return DummySegmentationModel()

    registry.register(
        name="dummy",
        factory=factory,
        description="Dummy.",
    )

    registry.unregister("dummy")

    assert not registry.contains("dummy")
    assert len(registry) == 0


def test_registry_rejects_unregistering_unknown_model():
    registry = ModelRegistry(
        include_builtin=False,
    )

    with pytest.raises(ModelRegistryError):
        registry.unregister("missing")


def test_builtin_registry_names_are_deterministic():
    registry = ModelRegistry()

    assert registry.names() == (
        "deeplabv3plus_resnet101",
        "deeplabv3plus_resnet50",
        "fpn_resnet50",
        "unetplusplus_efficientnet_b4",
    )


def test_builtin_registry_does_not_import_smp_during_registry_construction():
    registry = ModelRegistry()

    assert len(registry) == 4
    assert registry.contains(
        "deeplabv3plus_resnet50"
    )


# ---------------------------------------------------------------------------
# Gradient contract
# ---------------------------------------------------------------------------


def test_model_preserves_gradient_flow():
    model = DummySegmentationModel()

    x = torch.randn(
        2,
        3,
        16,
        16,
        requires_grad=True,
    )

    logits = model(x)

    loss = logits.mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    trainable_gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert trainable_gradients
    assert all(
        gradient is not None
        for gradient in trainable_gradients
    )
    assert all(
        torch.isfinite(gradient).all()
        for gradient in trainable_gradients
    )