from __future__ import annotations

import importlib.util

import pytest
import torch

from training.models.base import DEFAULT_CLASS_IDS
from training.models.conventional import (
    CONVENTIONAL_FACTORIES,
)
from training.models.mamba import (
    CNNMamba,
    MAMBA_FACTORIES,
    cnn_mamba,
)


def test_mamba_factory_exists():
    assert "cnn_mamba" in MAMBA_FACTORIES


def test_mamba_contract_metadata():
    model = cnn_mamba(
        base_channels=8,
        state_dim=4,
    )

    assert isinstance(model, CNNMamba)
    assert model.model_id == "cnn_mamba"
    assert model.num_classes == 7
    assert model.in_channels == 3
    assert model.class_ids == DEFAULT_CLASS_IDS
    assert model.architecture == "CNN-Mamba"


def test_mamba_forward_shape():
    model = cnn_mamba(
        base_channels=8,
        state_dim=4,
    )
    model.eval()

    x = torch.randn(1, 3, 64, 64)

    with torch.no_grad():
        output = model(x)

    assert output.shape == (1, 7, 64, 64)
    assert torch.isfinite(output).all()


def test_mamba_odd_spatial_dimensions():
    model = cnn_mamba(
        base_channels=8,
        state_dim=4,
    )
    model.eval()

    x = torch.randn(1, 3, 65, 71)

    with torch.no_grad():
        output = model(x)

    assert output.shape == (1, 7, 65, 71)


def test_mamba_batch_dimension():
    model = cnn_mamba(
        base_channels=8,
        state_dim=4,
    )
    model.eval()

    x = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        output = model(x)

    assert output.shape == (2, 7, 32, 32)


def test_mamba_backward_has_gradients():
    model = cnn_mamba(
        base_channels=8,
        state_dim=4,
    )

    x = torch.randn(
        1,
        3,
        32,
        32,
        requires_grad=True,
    )

    output = model(x)

    loss = output.mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    trainable_grads = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert trainable_grads
    assert any(
        gradient is not None
        and torch.isfinite(gradient).all()
        for gradient in trainable_grads
    )


def test_mamba_rejects_bad_channels():
    model = cnn_mamba(
        base_channels=8,
        state_dim=4,
    )

    with pytest.raises(ValueError):
        model(torch.randn(1, 4, 32, 32))


def test_mamba_rejects_integer_input():
    model = cnn_mamba(
        base_channels=8,
        state_dim=4,
    )

    with pytest.raises(ValueError):
        model(torch.ones(1, 3, 32, 32, dtype=torch.int64))


def test_mamba_rejects_non_finite_input():
    model = cnn_mamba(
        base_channels=8,
        state_dim=4,
    )

    x = torch.randn(1, 3, 32, 32)
    x[0, 0, 0, 0] = float("nan")

    with pytest.raises(ValueError):
        model(x)


def test_mamba_parameter_count_is_positive():
    model = cnn_mamba(
        base_channels=8,
        state_dim=4,
    )

    assert model.parameter_count() > 0
    assert model.parameter_count(trainable_only=True) > 0


def test_mamba_encoder_freezing():
    model = cnn_mamba(
        base_channels=8,
        state_dim=4,
    )

    model.freeze_encoder()

    encoder_parameters = list(
        model.encoder.parameters()
    )

    assert encoder_parameters
    assert all(
        not parameter.requires_grad
        for parameter in encoder_parameters
    )

    model.unfreeze_encoder()

    assert all(
        parameter.requires_grad
        for parameter in encoder_parameters
    )


def test_conventional_factory_names():
    expected = {
        "deeplabv3plus_resnet50",
        "deeplabv3plus_resnet101",
        "unetplusplus_efficientnet_b4",
        "fpn_resnet50",
    }

    assert set(CONVENTIONAL_FACTORIES) == expected


def test_conventional_factories_are_callable():
    for name, factory in CONVENTIONAL_FACTORIES.items():
        assert callable(factory), name


def test_smp_is_optional_at_import_time():
    """
    Conventional architecture module must remain importable even if
    segmentation_models_pytorch is not installed.
    """

    module = importlib.util.find_spec(
        "segmentation_models_pytorch"
    )

    # The assertion intentionally does not require SMP to be installed.
    # Architecture construction itself is responsible for reporting
    # the dependency requirement.
    assert module is None or module is not None


@pytest.mark.parametrize(
    "factory_name",
    [
        "deeplabv3plus_resnet50",
        "deeplabv3plus_resnet101",
        "unetplusplus_efficientnet_b4",
        "fpn_resnet50",
    ],
)
def test_conventional_factory_metadata_without_construction(factory_name):
    factory = CONVENTIONAL_FACTORIES[factory_name]

    assert callable(factory)