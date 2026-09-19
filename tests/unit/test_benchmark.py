from __future__ import annotations

import json

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from training.benchmark import (
    LOVE_DA_CLASS_IDS,
    BenchmarkError,
    BenchmarkRunner,
)


class TinyModel(nn.Module):
    model_version = "test-1.0"
    architecture = "Tiny-Test"

    def __init__(
        self,
        num_classes=7,
        in_channels=3,
        **_,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            num_classes,
            kernel_size=1,
        )

    @property
    def metadata(self):
        return {
            "model_id": "tiny",
            "model_version": self.model_version,
            "architecture": self.architecture,
            "parameter_count": sum(
                p.numel()
                for p in self.parameters()
            ),
        }

    def forward(self, x):
        return self.conv(x)


class TinyLoss(nn.Module):
    def forward(self, logits, target):
        target = target.clone()

        # LoveDA external labels 1..7 need canonical
        # internal labels 0..6 for CrossEntropy.
        target = target - 1
        target[target < 0] = -100

        return nn.functional.cross_entropy(
            logits,
            target,
            ignore_index=-100,
        )


class FakeConfig:
    seed = 26137
    deterministic = True
    device = "cpu"

    epochs = 1
    learning_rate = 0.01
    weight_decay = 0.0

    optimizer = "sgd"
    scheduler = "none"

    gradient_accumulation_steps = 1
    gradient_clip_norm = None

    amp = False
    amp_dtype = "float16"

    monitor_metric = "mean_iou"
    monitor_mode = "max"

    early_stopping_patience = None

    def to_dict(self):
        return {
            "seed": self.seed,
            "deterministic": self.deterministic,
            "device": self.device,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "gradient_accumulation_steps": 1,
            "gradient_clip_norm": None,
            "amp": False,
            "amp_dtype": "float16",
            "monitor_metric": "mean_iou",
            "monitor_mode": "max",
            "early_stopping_patience": None,
        }


def make_loader():
    images = torch.randn(
        4,
        3,
        8,
        8,
    )

    masks = torch.tensor(
        [
            [[1] * 8] * 8,
            [[2] * 8] * 8,
            [[6] * 8] * 8,
            [[7] * 8] * 8,
        ],
        dtype=torch.long,
    )

    domains = [
        "urban",
        "urban",
        "rural",
        "rural",
    ]

    dataset = [
        {
            "image": images[i],
            "mask": masks[i],
            "domains": domains[i],
        }
        for i in range(4)
    ]

    return DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )


def make_runner(tmp_path):
    return BenchmarkRunner(
        benchmark_id="test-benchmark",
        training_config=FakeConfig(),
        train_loader_factory=make_loader,
        val_loader_factory=make_loader,
        model_ids=("tiny",),
        output_dir=tmp_path,
        model_builder=lambda *args, **kwargs: TinyModel(),
        loss_factory=lambda *args, **kwargs: TinyLoss(),
        seed=26137,
    )


def test_constants_match_loveda_contract():
    assert LOVE_DA_CLASS_IDS == (
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    )


def test_empty_benchmark_id_rejected(tmp_path):
    with pytest.raises(BenchmarkError):
        BenchmarkRunner(
            benchmark_id="",
            training_config=FakeConfig(),
            train_loader_factory=make_loader,
            val_loader_factory=make_loader,
            model_ids=("tiny",),
            output_dir=tmp_path,
        )


def test_empty_model_ids_rejected(tmp_path):
    with pytest.raises(BenchmarkError):
        BenchmarkRunner(
            benchmark_id="x",
            training_config=FakeConfig(),
            train_loader_factory=make_loader,
            val_loader_factory=make_loader,
            model_ids=(),
            output_dir=tmp_path,
        )


def test_duplicate_model_ids_rejected(tmp_path):
    with pytest.raises(BenchmarkError):
        BenchmarkRunner(
            benchmark_id="x",
            training_config=FakeConfig(),
            train_loader_factory=make_loader,
            val_loader_factory=make_loader,
            model_ids=("tiny", "tiny"),
            output_dir=tmp_path,
        )


def test_negative_seed_rejected(tmp_path):
    with pytest.raises(BenchmarkError):
        BenchmarkRunner(
            benchmark_id="x",
            training_config=FakeConfig(),
            train_loader_factory=make_loader,
            val_loader_factory=make_loader,
            model_ids=("tiny",),
            output_dir=tmp_path,
            seed=-1,
        )


def test_optimizer_uses_canonical_learning_rate(
    tmp_path,
):
    runner = make_runner(tmp_path)

    model = TinyModel()

    optimizer = runner._build_optimizer(
        model
    )

    assert optimizer.param_groups[0]["lr"] == pytest.approx(
        0.01
    )


def test_optimizer_rejects_invalid_learning_rate(
    tmp_path,
):
    runner = make_runner(tmp_path)

    runner.training_config.learning_rate = 0.0

    with pytest.raises(BenchmarkError):
        runner._build_optimizer(
            TinyModel()
        )


def test_cosine_scheduler_is_supported(
    tmp_path,
):
    config = FakeConfig()
    config.scheduler = "cosine"

    runner = BenchmarkRunner(
        benchmark_id="x",
        training_config=config,
        train_loader_factory=make_loader,
        val_loader_factory=make_loader,
        model_ids=("tiny",),
        output_dir=tmp_path,
    )

    optimizer = runner._build_optimizer(
        TinyModel()
    )

    scheduler = runner._build_scheduler(
        optimizer
    )

    assert isinstance(
        scheduler,
        torch.optim.lr_scheduler.CosineAnnealingLR,
    )


def test_plateau_scheduler_is_supported(
    tmp_path,
):
    config = FakeConfig()
    config.scheduler = "plateau"

    runner = BenchmarkRunner(
        benchmark_id="x",
        training_config=config,
        train_loader_factory=make_loader,
        val_loader_factory=make_loader,
        model_ids=("tiny",),
        output_dir=tmp_path,
    )

    optimizer = runner._build_optimizer(
        TinyModel()
    )

    scheduler = runner._build_scheduler(
        optimizer
    )

    assert isinstance(
        scheduler,
        torch.optim.lr_scheduler.ReduceLROnPlateau,
    )


def test_none_scheduler_returns_none(
    tmp_path,
):
    runner = make_runner(tmp_path)

    optimizer = runner._build_optimizer(
        TinyModel()
    )

    assert (
        runner._build_scheduler(
            optimizer
        )
        is None
    )


def test_monitor_configuration_is_canonical(
    tmp_path,
):
    runner = make_runner(tmp_path)

    assert (
        runner._config_value(
            "monitor_metric",
            None,
        )
        == "mean_iou"
    )

    assert (
        runner._config_value(
            "monitor_mode",
            None,
        )
        == "max"
    )


def test_metric_factory_uses_loveda_taxonomy(
    tmp_path,
):
    runner = make_runner(tmp_path)

    accumulator = runner._metric_factory()

    assert accumulator.num_classes == 7
    assert tuple(
        accumulator.class_ids
    ) == LOVE_DA_CLASS_IDS
    assert accumulator.ignore_index == 0


def test_model_metadata_is_extracted(
    tmp_path,
):
    runner = make_runner(tmp_path)

    metadata = runner._model_metadata(
        TinyModel(),
        "tiny",
    )

    assert metadata["model_id"] == "tiny"
    assert metadata["architecture"] == "Tiny-Test"
    assert metadata["parameter_count"] > 0


def test_json_safe_converts_nested_values():
    value = {
        "path": __file__,
        "tuple": (1, 2),
        "tensor_scalar": torch.tensor(2),
    }

    converted = runner_json_safe(value)

    assert converted["tuple"] == [1, 2]
    assert converted["tensor_scalar"] == 2


def runner_json_safe(value):
    from training.benchmark import _json_safe

    return _json_safe(value)


def test_full_benchmark_runs(
    tmp_path,
):
    runner = make_runner(tmp_path)

    result = runner.run()

    assert result.dataset == "LoveDA"
    assert len(result.models) == 1

    model = result.models[0]

    assert model.model_id == "tiny"
    assert model.best_epoch is not None
    assert model.best_metric is not None

    assert model.evaluation.valid_pixels > 0

    overall = model.evaluation.overall

    assert "pixel_accuracy" in overall
    assert "mean_iou" in overall
    assert "macro_precision" in overall
    assert "macro_recall" in overall
    assert "macro_f1" in overall
    assert "macro_dice" in overall
    assert "frequency_weighted_iou" in overall

    assert set(
        model.evaluation.per_class.keys()
    ) == {
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
    }


def test_per_class_metrics_have_expected_fields(
    tmp_path,
):
    runner = make_runner(tmp_path)

    result = runner.run()

    per_class = (
        result.models[0]
        .evaluation
        .per_class
    )

    for values in per_class.values():
        assert "class_name" in values
        assert "iou" in values
        assert "precision" in values
        assert "recall" in values
        assert "f1" in values


def test_confusion_matrix_has_canonical_shape(
    tmp_path,
):
    runner = make_runner(tmp_path)

    result = runner.run()

    matrix = (
        result.models[0]
        .evaluation
        .confusion_matrix
    )

    assert len(matrix) == 7

    for row in matrix:
        assert len(row) == 7
        assert all(
            isinstance(value, int)
            for value in row
        )


def test_domain_breakdown_contains_urban_and_rural(
    tmp_path,
):
    runner = make_runner(tmp_path)

    result = runner.run()

    domains = (
        result.models[0]
        .evaluation
        .domain_metrics
    )

    assert set(domains) == {
        "urban",
        "rural",
    }

    for metrics in domains.values():
        assert "mean_iou" in metrics
        assert "macro_f1" in metrics
        assert "pixel_accuracy" in metrics


def test_benchmark_artifact_is_persisted(
    tmp_path,
):
    runner = make_runner(tmp_path)

    runner.run()

    artifact = (
        tmp_path
        / "test-benchmark.json"
    )

    assert artifact.exists()

    with artifact.open(
        encoding="utf-8"
    ) as handle:
        payload = json.load(handle)

    assert payload["dataset"] == "LoveDA"
    assert len(payload["models"]) == 1

    model = payload["models"][0]

    assert "evaluation" in model
    assert "per_class" in (
        model["evaluation"]
    )
    assert "domain_metrics" in (
        model["evaluation"]
    )


def test_artifact_is_strict_json(
    tmp_path,
):
    runner = make_runner(tmp_path)

    runner.run()

    artifact = (
        tmp_path
        / "test-benchmark.json"
    )

    text = artifact.read_text(
        encoding="utf-8"
    )

    payload = json.loads(text)

    assert payload["benchmark_id"] == (
        "test-benchmark"
    )


def test_missing_validation_image_key_is_rejected(
    tmp_path,
):
    def bad_loader():
        return [
            {
                "mask": torch.ones(
                    1,
                    4,
                    4,
                    dtype=torch.long,
                )
            }
        ]

    runner = BenchmarkRunner(
        benchmark_id="bad",
        training_config=FakeConfig(),
        train_loader_factory=make_loader,
        val_loader_factory=bad_loader,
        model_ids=("tiny",),
        output_dir=tmp_path,
        model_builder=lambda *args, **kwargs: TinyModel(),
        loss_factory=lambda *args, **kwargs: TinyLoss(),
    )

    with pytest.raises(BenchmarkError):
        runner.run()


def test_missing_validation_mask_key_is_rejected(
    tmp_path,
):
    def bad_loader():
        return [
            {
                "image": torch.randn(
                    1,
                    3,
                    4,
                    4,
                )
            }
        ]

    runner = BenchmarkRunner(
        benchmark_id="bad",
        training_config=FakeConfig(),
        train_loader_factory=make_loader,
        val_loader_factory=bad_loader,
        model_ids=("tiny",),
        output_dir=tmp_path,
        model_builder=lambda *args, **kwargs: TinyModel(),
        loss_factory=lambda *args, **kwargs: TinyLoss(),
    )

    with pytest.raises(BenchmarkError):
        runner.run()


def test_domain_metadata_length_is_validated(
    tmp_path,
):
    def bad_loader():
        images = torch.randn(
            2,
            3,
            4,
            4,
        )

        masks = torch.ones(
            2,
            4,
            4,
            dtype=torch.long,
        )

        return [
            {
                "image": images,
                "mask": masks,
                "domains": ["urban"],
            }
        ]

    runner = BenchmarkRunner(
        benchmark_id="bad-domain",
        training_config=FakeConfig(),
        train_loader_factory=make_loader,
        val_loader_factory=bad_loader,
        model_ids=("tiny",),
        output_dir=tmp_path,
        model_builder=lambda *args, **kwargs: TinyModel(),
        loss_factory=lambda *args, **kwargs: TinyLoss(),
    )

    with pytest.raises(BenchmarkError):
        runner.run()


def test_unknown_domain_is_rejected(
    tmp_path,
):
    def bad_loader():
        images = torch.randn(
            1,
            3,
            4,
            4,
        )

        masks = torch.ones(
            1,
            4,
            4,
            dtype=torch.long,
        )

        return [
            {
                "image": images,
                "mask": masks,
                "domains": ["unknown"],
            }
        ]

    runner = BenchmarkRunner(
        benchmark_id="bad-domain",
        training_config=FakeConfig(),
        train_loader_factory=make_loader,
        val_loader_factory=bad_loader,
        model_ids=("tiny",),
        output_dir=tmp_path,
        model_builder=lambda *args, **kwargs: TinyModel(),
        loss_factory=lambda *args, **kwargs: TinyLoss(),
    )

    with pytest.raises(BenchmarkError):
        runner.run()


def test_empty_validation_loader_is_rejected(
    tmp_path,
):
    def empty_loader():
        return []

    runner = BenchmarkRunner(
        benchmark_id="empty-val",
        training_config=FakeConfig(),
        train_loader_factory=make_loader,
        val_loader_factory=empty_loader,
        model_ids=("tiny",),
        output_dir=tmp_path,
        model_builder=lambda *args, **kwargs: TinyModel(),
        loss_factory=lambda *args, **kwargs: TinyLoss(),
    )

    with pytest.raises(BenchmarkError):
        runner.run()


def test_callback_receives_completed_model(
    tmp_path,
):
    runner = make_runner(tmp_path)

    received = []

    runner.run(
        on_model_complete=received.append
    )

    assert len(received) == 1
    assert (
        received[0].model_id
        == "tiny"
    )


def test_runner_can_be_reused(
    tmp_path,
):
    runner = make_runner(tmp_path)

    first = runner.run()
    second = runner.run()

    assert len(first.models) == 1
    assert len(second.models) == 1
    assert (
        second.models[0].model_id
        == "tiny"
    )