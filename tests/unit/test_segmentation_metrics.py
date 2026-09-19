import math

import pytest
import torch

from training.metrics import (
    SegmentationMetricAccumulator,
    SegmentationMetricError,
)


def test_perfect_prediction():
    metric = SegmentationMetricAccumulator(
        3,
        ignore_index=255,
    )

    target = torch.tensor(
        [[0, 1, 2],
         [2, 1, 0]]
    )

    metric.update(target, target)

    result = metric.compute()

    assert result.pixel_accuracy == pytest.approx(1.0)
    assert result.mean_iou == pytest.approx(1.0)
    assert result.macro_f1 == pytest.approx(1.0)

    assert all(
        value == pytest.approx(1.0)
        for value in result.per_class_iou.values()
    )


def test_known_confusion_matrix():
    metric = SegmentationMetricAccumulator(
        2,
        ignore_index=255,
    )

    target = torch.tensor(
        [[0, 0, 1, 1]]
    )

    prediction = torch.tensor(
        [[0, 1, 1, 0]]
    )

    metric.update(prediction, target)

    result = metric.compute()

    assert result.confusion_matrix.tolist() == [
        [1, 1],
        [1, 1],
    ]

    assert result.per_class_iou[0] == pytest.approx(1 / 3)
    assert result.per_class_iou[1] == pytest.approx(1 / 3)
    assert result.mean_iou == pytest.approx(1 / 3)

    assert result.pixel_accuracy == pytest.approx(0.5)


def test_ignore_index_is_excluded():
    metric = SegmentationMetricAccumulator(
        2,
        ignore_index=255,
    )

    target = torch.tensor(
        [[0, 1, 255, 255]]
    )

    prediction = torch.tensor(
        [[0, 0, 1, 0]]
    )

    metric.update(prediction, target)

    result = metric.compute()

    assert result.valid_pixels == 2
    assert result.pixel_accuracy == pytest.approx(0.5)

    assert result.confusion_matrix.tolist() == [
        [1, 0],
        [1, 0],
    ]


def test_all_ignored_batch_is_safe():
    metric = SegmentationMetricAccumulator(
        2,
        ignore_index=255,
    )

    target = torch.full(
        (2, 4),
        255,
        dtype=torch.long,
    )

    prediction = torch.zeros_like(target)

    metric.update(prediction, target)

    result = metric.compute()

    assert result.valid_pixels == 0
    assert math.isnan(result.pixel_accuracy)
    assert math.isnan(result.mean_iou)


def test_absent_class_is_not_counted_in_macro_metrics():
    metric = SegmentationMetricAccumulator(
        3,
        ignore_index=255,
    )

    target = torch.tensor(
        [[0, 0, 1, 1]]
    )

    prediction = target.clone()

    metric.update(prediction, target)

    result = metric.compute()

    assert result.mean_iou == pytest.approx(1.0)
    assert result.macro_f1 == pytest.approx(1.0)

    assert math.isnan(result.per_class_iou[2])
    assert math.isnan(result.per_class_f1[2])


def test_logits_are_supported():
    metric = SegmentationMetricAccumulator(
        3,
        ignore_index=255,
    )

    target = torch.tensor(
        [[0, 1],
         [2, 0]]
    )

    logits = torch.full(
        (1, 3, 2, 2),
        -10.0,
    )

    logits[0, 0, 0, 0] = 10
    logits[0, 1, 0, 1] = 10
    logits[0, 2, 1, 0] = 10
    logits[0, 0, 1, 1] = 10

    metric.update(logits, target.unsqueeze(0))

    result = metric.compute()

    assert result.pixel_accuracy == pytest.approx(1.0)


def test_batch_accumulation_equals_single_update():
    target_a = torch.tensor(
        [[0, 1],
         [1, 0]]
    )

    pred_a = torch.tensor(
        [[0, 0],
         [1, 1]]
    )

    target_b = torch.tensor(
        [[1, 1],
         [0, 0]]
    )

    pred_b = torch.tensor(
        [[1, 0],
         [0, 0]]
    )

    first = SegmentationMetricAccumulator(
        2,
        ignore_index=255,
    )

    first.update(pred_a, target_a)
    first.update(pred_b, target_b)

    combined = SegmentationMetricAccumulator(
        2,
        ignore_index=255,
    )

    combined.update(
        torch.cat([pred_a, pred_b]),
        torch.cat([target_a, target_b]),
    )

    assert torch.equal(
        first.confusion_matrix,
        combined.confusion_matrix,
    )

    assert first.compute().mean_iou == pytest.approx(
        combined.compute().mean_iou
    )


def test_loveda_external_class_ids():
    metric = SegmentationMetricAccumulator(
        7,
        ignore_index=0,
        class_ids=range(1, 8),
    )

    target = torch.tensor(
        [[1, 2],
         [3, 7]]
    )

    metric.update(target, target)

    result = metric.compute()

    assert set(result.per_class_iou) == set(range(1, 8))
    assert result.per_class_iou[1] == pytest.approx(1.0)
    assert result.per_class_iou[7] == pytest.approx(1.0)

    # Classes absent from this particular sample are undefined, not zero.
    assert math.isnan(result.per_class_iou[4])


def test_invalid_prediction_class_is_rejected():
    metric = SegmentationMetricAccumulator(
        2,
        ignore_index=255,
    )

    with pytest.raises(SegmentationMetricError):
        metric.update(
            torch.tensor([[0, 2]]),
            torch.tensor([[0, 1]]),
        )


def test_invalid_target_class_is_rejected():
    metric = SegmentationMetricAccumulator(
        2,
        ignore_index=255,
    )

    with pytest.raises(SegmentationMetricError):
        metric.update(
            torch.tensor([[0, 1]]),
            torch.tensor([[0, 2]]),
        )


def test_shape_mismatch_is_rejected():
    metric = SegmentationMetricAccumulator(
        2,
        ignore_index=255,
    )

    with pytest.raises(SegmentationMetricError):
        metric.update(
            torch.zeros((2, 2), dtype=torch.long),
            torch.zeros((2, 3), dtype=torch.long),
        )


def test_invalid_logit_channel_count_is_rejected():
    metric = SegmentationMetricAccumulator(
        3,
        ignore_index=255,
    )

    with pytest.raises(SegmentationMetricError):
        metric.update(
            torch.zeros((1, 2, 4, 4)),
            torch.zeros((1, 4, 4), dtype=torch.long),
        )


def test_duplicate_class_ids_are_rejected():
    with pytest.raises(SegmentationMetricError):
        SegmentationMetricAccumulator(
            3,
            class_ids=[1, 1, 2],
        )


def test_metric_output_is_json_serializable():
    metric = SegmentationMetricAccumulator(
        2,
        ignore_index=255,
    )

    target = torch.tensor([[0, 1]])
    metric.update(target, target)

    output = metric.compute().to_dict()

    assert isinstance(output, dict)
    assert isinstance(output["confusion_matrix"], list)
    assert isinstance(output["per_class_iou"], dict)


def test_reset_clears_accumulation():
    metric = SegmentationMetricAccumulator(
        2,
        ignore_index=255,
    )

    target = torch.tensor([[0, 1]])
    metric.update(target, target)

    assert metric.valid_pixels == 2

    metric.reset()

    assert metric.valid_pixels == 0
    assert torch.equal(
        metric.confusion_matrix,
        torch.zeros((2, 2), dtype=torch.int64),
    )