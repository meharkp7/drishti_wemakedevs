from __future__ import annotations

import pytest
import torch

from training.models.base import DEFAULT_CLASS_IDS
from training.models.dual_path import (
    DUAL_PATH_FACTORIES,
    DualPathArchitectureError,
    DualPathMamba,
    dual_path_mamba,
)
from training.models.frequency import (
    FrequencyComponentError,
    FrequencyFeatureExtractor,
    HaarWaveletTransform,
)


def make_model(
    *,
    frequency_fusion: bool = True,
) -> DualPathMamba:
    return DualPathMamba(
        base_channels=8,
        state_dim=4,
        frequency_fusion=frequency_fusion,
        frequency_channels=16,
    )


def test_factory_registered():
    assert "dual_path_mamba" in DUAL_PATH_FACTORIES
    assert callable(DUAL_PATH_FACTORIES["dual_path_mamba"])


def test_model_metadata():
    model = make_model()

    assert model.model_id == "dual_path_mamba"
    assert model.num_classes == 7
    assert model.in_channels == 3
    assert model.class_ids == DEFAULT_CLASS_IDS
    assert model.frequency_fusion_enabled is True
    assert model.architecture == "DualPath-CNN-Mamba-Frequency"


def test_frequency_disabled_metadata():
    model = make_model(
        frequency_fusion=False,
    )

    assert model.frequency_fusion_enabled is False
    assert model.architecture == "DualPath-CNN-Mamba"


def test_forward_with_frequency():
    model = make_model()
    model.eval()

    x = torch.randn(1, 3, 64, 64)

    with torch.no_grad():
        output = model(x)

    assert output.shape == (1, 7, 64, 64)
    assert torch.isfinite(output).all()


def test_forward_without_frequency():
    model = make_model(
        frequency_fusion=False,
    )
    model.eval()

    x = torch.randn(1, 3, 64, 64)

    with torch.no_grad():
        output = model(x)

    assert output.shape == (1, 7, 64, 64)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    "height,width",
    [
        (32, 48),
        (63, 65),
        (71, 79),
        (96, 80),
    ],
)
def test_arbitrary_spatial_dimensions(
    height: int,
    width: int,
):
    model = make_model()
    model.eval()

    x = torch.randn(
        1,
        3,
        height,
        width,
    )

    with torch.no_grad():
        output = model(x)

    assert output.shape == (
        1,
        7,
        height,
        width,
    )


def test_batch_dimension():
    model = make_model()
    model.eval()

    x = torch.randn(
        2,
        3,
        32,
        40,
    )

    with torch.no_grad():
        output = model(x)

    assert output.shape == (
        2,
        7,
        32,
        40,
    )


def test_backward_gradient_flow():
    model = make_model()

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

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert gradients

    assert any(
        gradient is not None
        and torch.isfinite(gradient).all()
        for gradient in gradients
    )


def test_frequency_branch_receives_gradients():
    model = make_model()

    x = torch.randn(
        1,
        3,
        32,
        32,
    )

    output = model(x)
    output.mean().backward()

    frequency_parameters = [
        parameter
        for parameter in model.frequency_path.parameters()
        if parameter.requires_grad
    ]

    assert frequency_parameters

    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        for parameter in frequency_parameters
    )


def test_frequency_disabled_has_no_frequency_parameters():
    model = make_model(
        frequency_fusion=False,
    )

    assert model.frequency_path is None
    assert model.frequency_projection is None


def test_bad_input_channels():
    model = make_model()

    with pytest.raises(ValueError):
        model(torch.randn(1, 4, 32, 32))


def test_bad_input_dtype():
    model = make_model()

    with pytest.raises(ValueError):
        model(
            torch.ones(
                1,
                3,
                32,
                32,
                dtype=torch.int64,
            )
        )


def test_nonfinite_input():
    model = make_model()

    x = torch.randn(
        1,
        3,
        32,
        32,
    )

    x[0, 0, 0, 0] = float("nan")

    with pytest.raises(ValueError):
        model(x)


def test_positive_parameter_count():
    model = make_model()

    assert model.parameter_count() > 0
    assert model.parameter_count(
        trainable_only=True
    ) > 0


def test_wavelet_output_shapes():
    transform = HaarWaveletTransform()

    x = torch.randn(
        2,
        3,
        32,
        40,
    )

    outputs = transform(x)

    assert len(outputs) == 4

    for output in outputs:
        assert output.shape == (
            2,
            3,
            16,
            20,
        )
        assert torch.isfinite(output).all()


def test_wavelet_preserves_channel_independence():
    transform = HaarWaveletTransform()

    x = torch.zeros(
        1,
        2,
        16,
        16,
    )

    x[:, 0] = 1.0

    outputs = transform(x)

    for output in outputs:
        assert torch.allclose(
            output[:, 1],
            torch.zeros_like(output[:, 1]),
        )


def test_frequency_extractor_shape():
    extractor = FrequencyFeatureExtractor(
        in_channels=3,
        out_channels=12,
    )

    x = torch.randn(
        1,
        3,
        32,
        40,
    )

    output = extractor(x)

    assert output.shape == (
        1,
        12,
        16,
        20,
    )

    assert torch.isfinite(output).all()


def test_frequency_extractor_backward():
    extractor = FrequencyFeatureExtractor(
        in_channels=3,
        out_channels=12,
    )

    x = torch.randn(
        1,
        3,
        16,
        16,
        requires_grad=True,
    )

    output = extractor(x)
    output.mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_wavelet_rejects_too_small_input():
    transform = HaarWaveletTransform()

    with pytest.raises(FrequencyComponentError):
        transform(
            torch.randn(
                1,
                3,
                1,
                8,
            )
        )


def test_wavelet_rejects_nonfinite_input():
    transform = HaarWaveletTransform()

    x = torch.randn(
        1,
        3,
        8,
        8,
    )

    x[0, 0, 0, 0] = float("inf")

    with pytest.raises(FrequencyComponentError):
        transform(x)


def test_invalid_base_channels():
    with pytest.raises(DualPathArchitectureError):
        DualPathMamba(
            base_channels=4,
        )


def test_invalid_frequency_channels():
    with pytest.raises(DualPathArchitectureError):
        DualPathMamba(
            base_channels=8,
            frequency_channels=0,
        )


def test_factory_output():
    model = dual_path_mamba(
        base_channels=8,
        state_dim=4,
    )

    assert isinstance(
        model,
        DualPathMamba,
    )