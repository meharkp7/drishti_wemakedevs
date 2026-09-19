from __future__ import annotations
from xml.parsers.expat import model

import pytest
import torch

from training.models.base import DEFAULT_CLASS_IDS
from training.models.foundation import (
    FoundationArchitectureError,
    FoundationGeospatialModel,
    foundation_geospatial,
)
from training.models.foundation_components import (
    FeaturePyramid,
    FoundationComponentError,
    GeospatialFeatureAdapter,
    HierarchicalFoundationEncoder,
)


def make_model() -> FoundationGeospatialModel:
    return FoundationGeospatialModel(
        base_channels=8,
        decoder_channels=32,
    )


def test_factory_metadata():
    model = foundation_geospatial(
        base_channels=8,
        decoder_channels=32,
    )

    assert model.model_id == "foundation_geospatial"
    assert model.architecture == "FoundationEncoder-GeospatialDecoder"
    assert model.num_classes == 7
    assert model.in_channels == 3
    assert model.class_ids == DEFAULT_CLASS_IDS
    assert model.spec.pretrained_encoder is False


def test_forward_shape():
    model = make_model()
    model.eval()

    x = torch.randn(1, 3, 64, 64)

    with torch.no_grad():
        output = model(x)

    assert output.shape == (1, 7, 64, 64)
    assert torch.isfinite(output).all()


def test_forward_odd_resolution():
    model = make_model()
    model.eval()

    x = torch.randn(1, 3, 65, 71)

    with torch.no_grad():
        output = model(x)

    assert output.shape == (1, 7, 65, 71)


def test_forward_batch():
    model = make_model()
    model.eval()

    x = torch.randn(2, 3, 48, 52)

    with torch.no_grad():
        output = model(x)

    assert output.shape == (2, 7, 48, 52)


def test_encoder_feature_pyramid():
    encoder = HierarchicalFoundationEncoder(
        base_channels=8,
    )

    x = torch.randn(2, 3, 64, 64)

    pyramid = encoder(x)

    assert isinstance(pyramid, FeaturePyramid)
    assert pyramid.batch_size == 2
    assert pyramid.channels == (8, 16, 32, 64)
    assert pyramid.resolutions == (
        (32, 32),
        (16, 16),
        (8, 8),
        (4, 4),
    )


def test_feature_adapter():
    encoder = HierarchicalFoundationEncoder(
        base_channels=8,
    )

    pyramid = encoder(
        torch.randn(1, 3, 64, 64)
    )

    adapter = GeospatialFeatureAdapter(
        pyramid.channels,
        out_channels=24,
    )

    adapted = adapter(pyramid)

    assert len(adapted) == 4
    assert all(feature.shape[1] == 24 for feature in adapted)
    assert all(torch.isfinite(feature).all() for feature in adapted)


def test_freeze_and_unfreeze_encoder():
    model = make_model()

    model.freeze_encoder()

    assert all(
        not parameter.requires_grad
        for parameter in model.foundation_encoder.parameters()
    )

    model.unfreeze_encoder()

    assert all(
        parameter.requires_grad
        for parameter in model.foundation_encoder.parameters()
    )


def test_decoder_remains_trainable_when_encoder_frozen():
    model = make_model()

    model.freeze_encoder()

    encoder_parameters = list(
        model.foundation_encoder.parameters()
    )

    decoder_parameters = list(
        model.decoder.parameters()
    )

    assert encoder_parameters
    assert decoder_parameters

    assert all(
        not parameter.requires_grad
        for parameter in encoder_parameters
    )

    assert all(
        parameter.requires_grad
        for parameter in decoder_parameters
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


def test_invalid_input_channels():
    model = make_model()

    with pytest.raises(ValueError):
        model(torch.randn(1, 4, 32, 32))


def test_invalid_input_dtype():
    model = make_model()

    with pytest.raises(ValueError):
        model(torch.ones(1, 3, 32, 32, dtype=torch.int64))


def test_nonfinite_input():
    model = make_model()

    x = torch.randn(1, 3, 32, 32)
    x[0, 0, 0, 0] = float("inf")

    with pytest.raises(ValueError):
        model(x)


def test_pretrained_requires_checkpoint():
    with pytest.raises(FoundationArchitectureError):
        FoundationGeospatialModel(
            base_channels=8,
            decoder_channels=32,
            pretrained=True,
        )


def test_pretrained_configuration_metadata():
    model = FoundationGeospatialModel(
        base_channels=8,
        decoder_channels=32,
        pretrained=True,
        checkpoint="reference-checkpoint",
    )

    assert model.spec.pretrained_encoder is True
    assert model.spec.pretrained_source == "reference-checkpoint"


def test_invalid_pyramid_feature_rank():
    with pytest.raises(FoundationComponentError):
        FeaturePyramid(
            features=(
                torch.randn(1, 8, 8),
            )
        )


def test_invalid_pyramid_feature_dtype():
    with pytest.raises(FoundationComponentError):
        FeaturePyramid(
            features=(
                torch.ones(1, 8, 8, 8, dtype=torch.int64),
            )
        )


def test_invalid_pyramid_batch_size():
    with pytest.raises(FoundationComponentError):
        FeaturePyramid(
            features=(
                torch.randn(1, 8, 16, 16),
                torch.randn(2, 16, 8, 8),
            )
        )


def test_invalid_pyramid_resolution_order():
    with pytest.raises(FoundationComponentError):
        FeaturePyramid(
            features=(
                torch.randn(1, 8, 8, 8),
                torch.randn(1, 16, 16, 16),
            )
        )


def test_positive_parameter_count():
    model = make_model()

    assert model.parameter_count() > 0
    assert model.parameter_count(trainable_only=True) > 0