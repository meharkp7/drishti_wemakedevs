from __future__ import annotations

import pytest
import torch

from training.losses import (
    ClassWeightingStrategy,
    LossConfig,
    LossError,
    LossName,
    SegmentationLoss,
    build_loss,
    build_loss_from_name,
    compute_class_weights,
)


CLASS_IDS = (1,2,3,4,5,6,7)

COUNTS = {
    1: 1000,
    2: 500,
    3: 100,
    4: 250,
    5: 50,
    6: 700,
    7: 900,
}


def batch():

    torch.manual_seed(42)

    logits = torch.randn(
        2,
        7,
        8,
        8,
        requires_grad=True,
    )

    target = torch.randint(
        0,
        8,
        (2,8,8),
    )

    return logits,target


# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------

@pytest.mark.parametrize(
    "name",
    [x.value for x in LossName],
)
def test_all_names(name):

    cfg = LossConfig(name=name)

    assert cfg.name == name


def test_default_contract():

    cfg = LossConfig()

    assert cfg.class_ids == CLASS_IDS
    assert cfg.ignore_index == 0


def test_duplicate_ids():

    with pytest.raises(LossError):
        LossConfig(class_ids=(1,1,2))


def test_invalid_ignore_overlap():

    with pytest.raises(LossError):
        LossConfig(
            class_ids=(0,1,2),
            ignore_index=0,
        )


# ----------------------------------------------------------
# WEIGHTS
# ----------------------------------------------------------

@pytest.mark.parametrize(
    "strategy",
    [x.value for x in ClassWeightingStrategy],
)
def test_weight_modes(strategy):

    w = compute_class_weights(
        COUNTS if strategy!="none" else None,
        class_ids=CLASS_IDS,
        strategy=strategy,
    )

    assert w.shape == (7,)
    assert torch.isfinite(w).all()
    assert torch.all(w>0)


def test_inverse_prefers_rare():

    w = compute_class_weights(
        COUNTS,
        class_ids=CLASS_IDS,
        strategy="inverse_frequency",
    )

    assert w[4] > w[0]


def test_missing_counts():

    with pytest.raises(LossError):

        compute_class_weights(
            None,
            class_ids=CLASS_IDS,
            strategy="inverse_frequency",
        )


# ----------------------------------------------------------
# FACTORY
# ----------------------------------------------------------

def test_factory():

    cfg = LossConfig()

    loss = build_loss(cfg)

    assert isinstance(loss,SegmentationLoss)


def test_factory_weighted():

    cfg = LossConfig(
        name="weighted_cross_entropy"
    )

    loss = build_loss(
        cfg,
        class_counts=COUNTS,
    )

    assert loss.class_weights.numel()==7


def test_explicit_weights():

    weights=torch.ones(7)

    loss=build_loss_from_name(
        "dice",
        class_weights=weights,
    )

    assert torch.allclose(
        loss.class_weights,
        weights,
    )


# ----------------------------------------------------------
# ALL LOSSES
# ----------------------------------------------------------

@pytest.mark.parametrize(
    "name",
    [x.value for x in LossName],
)
def test_every_loss_forward_backward(name):

    logits,target=batch()

    kwargs={}

    if name=="weighted_cross_entropy":

        kwargs={
            "class_counts":COUNTS,
        }

    loss_fn=build_loss_from_name(
        name,
        **kwargs,
    )

    loss=loss_fn(logits,target)

    assert loss.ndim==0
    assert torch.isfinite(loss)

    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


# ----------------------------------------------------------
# IGNORE
# ----------------------------------------------------------

@pytest.mark.parametrize(
    "name",
    [
        "cross_entropy",
        "dice",
        "ce_dice",
        "focal",
        "focal_dice",
        "tversky",
        "tversky_focal",
    ],
)
def test_all_ignore(name):

    logits=torch.randn(
        1,7,4,4,
        requires_grad=True,
    )

    target=torch.zeros(
        1,4,4,
        dtype=torch.long,
    )

    loss=build_loss_from_name(name)

    value=loss(logits,target)

    assert value.item()==pytest.approx(0.0)

    value.backward()

    assert torch.isfinite(logits.grad).all()


# ----------------------------------------------------------
# LABEL CONTRACT
# ----------------------------------------------------------

def test_external_ids():

    logits=torch.randn(
        1,7,2,4,
        requires_grad=True,
    )

    target=torch.tensor(
        [[[1,2,3,4],
          [5,6,7,0]]]
    )

    loss=build_loss_from_name(
        "cross_entropy"
    )

    value=loss(logits,target)

    assert torch.isfinite(value)


def test_unknown_class():

    logits=torch.randn(
        1,7,2,2
    )

    target=torch.tensor(
        [[[1,2],
          [3,99]]]
    )

    loss=build_loss_from_name("dice")

    with pytest.raises(LossError):
        loss(logits,target)


# ----------------------------------------------------------
# VALIDATION
# ----------------------------------------------------------

def test_invalid_logits_channels():

    logits=torch.randn(1,6,2,2)

    target=torch.ones(
        1,2,2,
        dtype=torch.long,
    )

    with pytest.raises(LossError):
        build_loss_from_name("dice")(logits,target)


def test_invalid_target_dtype():

    logits=torch.randn(1,7,2,2)

    target=torch.ones(
        1,2,2,
        dtype=torch.float32,
    )

    with pytest.raises(LossError):
        build_loss_from_name("dice")(logits,target)


def test_nan_logits():

    logits=torch.randn(1,7,2,2)

    logits[0,0,0,0]=float("nan")

    target=torch.ones(
        1,2,2,
        dtype=torch.long,
    )

    with pytest.raises(LossError):
        build_loss_from_name("cross_entropy")(logits,target)


# ----------------------------------------------------------
# EXTREME NUMERICAL STABILITY
# ----------------------------------------------------------

def test_extreme_logits():

    logits=torch.zeros(
        1,7,2,2,
        requires_grad=True,
    )

    logits.data[0,0]=1000
    logits.data[0,1]=-1000

    target=torch.ones(
        1,2,2,
        dtype=torch.long,
    )

    for name in [
        "cross_entropy",
        "dice",
        "ce_dice",
        "focal",
        "focal_dice",
        "tversky",
        "tversky_focal",
    ]:

        logits.grad=None

        loss=build_loss_from_name(name)

        value=loss(logits,target)

        assert torch.isfinite(value)

        value.backward()

        assert torch.isfinite(logits.grad).all()


# ----------------------------------------------------------
# PRESENT CLASS ONLY
# ----------------------------------------------------------

def test_single_class_dice():

    logits=torch.randn(
        2,7,8,8,
        requires_grad=True,
    )

    target=torch.full(
        (2,8,8),
        3,
        dtype=torch.long,
    )

    loss=build_loss_from_name("dice")

    value=loss(logits,target)

    assert torch.isfinite(value)

    value.backward()

    assert torch.isfinite(logits.grad).all()