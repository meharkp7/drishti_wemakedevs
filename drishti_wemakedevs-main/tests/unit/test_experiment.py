from __future__ import annotations

import json

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from training.checkpoint import CheckpointManager
from training.engine import TrainingEngine
from training.experiment import ExperimentError, ExperimentRunner


class TinySegmentationModel(nn.Module):
    def __init__(self, classes: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(3, classes, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class TinyMetricState:
    def __init__(self):
        self.count = 0

    def update(self, logits, targets):
        self.count += int(targets.numel())

    def compute(self):
        return {"pixel_count": float(self.count)}


class ConstantLoss(nn.Module):
    """
    Deterministic non-improving loss.

    It remains connected to the graph so backward() is valid.
    """

    def forward(self, logits, targets):
        return logits.mean() * 0.0 + 1.0


class CountingScheduler:
    def __init__(self):
        self.steps = 0

    def step(self, *args):
        self.steps += 1

    def state_dict(self):
        return {"steps": self.steps}

    def load_state_dict(self, state):
        self.steps = int(state["steps"])


@pytest.fixture
def loader():
    torch.manual_seed(7)

    images = torch.randn(4, 3, 8, 8)
    masks = torch.randint(
        0,
        2,
        (4, 8, 8),
        dtype=torch.long,
    )

    dataset = TensorDataset(images, masks)

    def collate(batch):
        xs, ys = zip(*batch)
        return {
            "image": torch.stack(xs),
            "mask": torch.stack(ys),
        }

    return DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate,
    )


@pytest.fixture
def components():
    model = TinySegmentationModel()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
    )

    return model, criterion, optimizer


def make_engine(
    model,
    criterion,
    optimizer,
    **kwargs,
):
    return TrainingEngine(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device="cpu",
        metrics_factory=TinyMetricState,
        **kwargs,
    )


def test_single_epoch_runs(loader, components, tmp_path):
    model, criterion, optimizer = components

    runner = ExperimentRunner(
        experiment_id="exp-001",
        engine=make_engine(model, criterion, optimizer),
        train_loader=loader,
        val_loader=loader,
        checkpoint_manager=CheckpointManager(tmp_path),
        max_epochs=1,
    )

    result = runner.run()

    assert result.completed_epochs == 1
    assert result.best_epoch == 0
    assert result.best_metric is not None
    assert result.best_checkpoint is not None
    assert result.latest_checkpoint is not None


def test_history_is_json_serializable(
    loader,
    components,
    tmp_path,
):
    model, criterion, optimizer = components

    runner = ExperimentRunner(
        experiment_id="exp-json",
        engine=make_engine(model, criterion, optimizer),
        train_loader=loader,
        val_loader=loader,
        checkpoint_manager=CheckpointManager(tmp_path),
        max_epochs=1,
    )

    result = runner.run()

    json.dumps(result.to_dict())

    history = json.loads(
        (tmp_path / "checkpoint-history.json").read_text()
    )

    assert history["experiment_id"] == "exp-json"
    assert len(history["history"]) == 1


def test_max_epochs_is_respected(
    loader,
    components,
    tmp_path,
):
    model, criterion, optimizer = components

    runner = ExperimentRunner(
        experiment_id="exp-max",
        engine=make_engine(model, criterion, optimizer),
        train_loader=loader,
        val_loader=None,
        checkpoint_manager=CheckpointManager(tmp_path),
        max_epochs=3,
    )

    result = runner.run()

    assert result.completed_epochs == 3
    assert result.best_epoch in {0, 1, 2}
    assert result.best_epoch < 3


def test_patience_stops_on_deterministic_plateau(
    loader,
    components,
    tmp_path,
):
    model, _, optimizer = components

    runner = ExperimentRunner(
        experiment_id="exp-patience",
        engine=make_engine(
            model,
            ConstantLoss(),
            optimizer,
        ),
        train_loader=loader,
        val_loader=loader,
        checkpoint_manager=CheckpointManager(tmp_path),
        max_epochs=10,
        patience=0,
    )

    result = runner.run()

    assert result.completed_epochs == 2
    assert result.best_epoch == 0


def test_patience_allows_configured_non_improving_epochs(
    loader,
    components,
    tmp_path,
):
    model, _, optimizer = components

    runner = ExperimentRunner(
        experiment_id="exp-patience-2",
        engine=make_engine(
            model,
            ConstantLoss(),
            optimizer,
        ),
        train_loader=loader,
        val_loader=loader,
        checkpoint_manager=CheckpointManager(tmp_path),
        max_epochs=10,
        patience=2,
    )

    result = runner.run()

    assert result.completed_epochs == 4
    assert result.best_epoch == 0


def test_min_mode_detects_improvement(
    loader,
    components,
    tmp_path,
):
    model, criterion, optimizer = components

    runner = ExperimentRunner(
        experiment_id="exp-min",
        engine=make_engine(model, criterion, optimizer),
        train_loader=loader,
        val_loader=None,
        checkpoint_manager=CheckpointManager(tmp_path),
        max_epochs=1,
        monitor="loss",
        monitor_mode="min",
    )

    assert runner._is_improvement(1.0)

    runner.best_metric = 2.0

    assert runner._is_improvement(1.0)
    assert not runner._is_improvement(3.0)


def test_max_mode_detects_improvement(
    loader,
    components,
    tmp_path,
):
    model, criterion, optimizer = components

    runner = ExperimentRunner(
        experiment_id="exp-max-mode",
        engine=make_engine(model, criterion, optimizer),
        train_loader=loader,
        val_loader=None,
        checkpoint_manager=CheckpointManager(tmp_path),
        max_epochs=1,
        monitor="pixel_count",
        monitor_mode="max",
    )

    runner.best_metric = 10.0

    assert runner._is_improvement(11.0)
    assert not runner._is_improvement(9.0)


def test_min_delta_is_respected(
    loader,
    components,
    tmp_path,
):
    model, criterion, optimizer = components

    runner = ExperimentRunner(
        experiment_id="exp-delta",
        engine=make_engine(model, criterion, optimizer),
        train_loader=loader,
        val_loader=None,
        checkpoint_manager=CheckpointManager(tmp_path),
        max_epochs=1,
        min_delta=0.1,
    )

    runner.best_metric = 1.0

    assert not runner._is_improvement(0.95)
    assert runner._is_improvement(0.89)


def test_invalid_configuration_is_rejected(
    loader,
    components,
    tmp_path,
):
    model, criterion, optimizer = components

    with pytest.raises(ExperimentError):
        ExperimentRunner(
            experiment_id="",
            engine=make_engine(model, criterion, optimizer),
            train_loader=loader,
            val_loader=None,
            checkpoint_manager=CheckpointManager(tmp_path),
            max_epochs=1,
        )

    with pytest.raises(ExperimentError):
        ExperimentRunner(
            experiment_id="bad",
            engine=make_engine(model, criterion, optimizer),
            train_loader=loader,
            val_loader=None,
            checkpoint_manager=CheckpointManager(tmp_path),
            max_epochs=0,
        )

    with pytest.raises(ExperimentError):
        ExperimentRunner(
            experiment_id="bad-mode",
            engine=make_engine(model, criterion, optimizer),
            train_loader=loader,
            val_loader=None,
            checkpoint_manager=CheckpointManager(tmp_path),
            max_epochs=1,
            monitor_mode="sideways",
        )


def test_start_epoch_bounds_are_checked(
    loader,
    components,
    tmp_path,
):
    model, criterion, optimizer = components

    runner = ExperimentRunner(
        experiment_id="exp-start",
        engine=make_engine(model, criterion, optimizer),
        train_loader=loader,
        val_loader=None,
        checkpoint_manager=CheckpointManager(tmp_path),
        max_epochs=2,
    )

    with pytest.raises(ExperimentError):
        runner.run(start_epoch=2)


def test_callback_receives_epoch_record(
    loader,
    components,
    tmp_path,
):
    model, criterion, optimizer = components

    records = []

    runner = ExperimentRunner(
        experiment_id="exp-callback",
        engine=make_engine(model, criterion, optimizer),
        train_loader=loader,
        val_loader=None,
        checkpoint_manager=CheckpointManager(tmp_path),
        max_epochs=2,
    )

    runner.run(on_epoch_end=records.append)

    assert len(records) == 2
    assert records[0]["experiment_id"] == "exp-callback"
    assert "train" in records[0]


def test_scheduler_is_stepped(
    loader,
    components,
    tmp_path,
):
    model, criterion, optimizer = components

    scheduler = CountingScheduler()

    engine = make_engine(
        model,
        criterion,
        optimizer,
        scheduler=scheduler,
    )

    runner = ExperimentRunner(
        experiment_id="exp-scheduler",
        engine=engine,
        train_loader=loader,
        val_loader=loader,
        checkpoint_manager=CheckpointManager(tmp_path),
        max_epochs=3,
    )

    runner.run()

    assert scheduler.steps == 3


def test_resume_rejects_wrong_experiment(
    loader,
    components,
    tmp_path,
):
    model, criterion, optimizer = components

    manager = CheckpointManager(tmp_path)
    engine = make_engine(model, criterion, optimizer)

    manager.save(
        path=tmp_path / "foreign.pt",
        model=model,
        optimizer=optimizer,
        engine_state=engine.state_dict(),
        epoch=0,
        global_step=1,
        extra={
            "experiment_id": "other-experiment",
        },
    )

    runner = ExperimentRunner(
        experiment_id="my-experiment",
        engine=engine,
        train_loader=loader,
        val_loader=None,
        checkpoint_manager=manager,
        max_epochs=2,
    )

    with pytest.raises(ExperimentError):
        runner.resume(tmp_path / "foreign.pt")


def test_resume_restores_history_and_counters(
    loader,
    components,
    tmp_path,
):
    model, criterion, optimizer = components

    manager = CheckpointManager(tmp_path)

    engine = make_engine(
        model,
        criterion,
        optimizer,
    )

    runner = ExperimentRunner(
        experiment_id="resume-exp",
        engine=engine,
        train_loader=loader,
        val_loader=None,
        checkpoint_manager=manager,
        max_epochs=1,
    )

    first_result = runner.run()

    assert first_result.completed_epochs == 1

    checkpoint = tmp_path / "checkpoint-latest.pt"

    new_model = TinySegmentationModel()
    new_optimizer = torch.optim.SGD(
        new_model.parameters(),
        lr=0.01,
    )

    new_engine = make_engine(
        new_model,
        criterion,
        new_optimizer,
    )

    resumed = ExperimentRunner(
        experiment_id="resume-exp",
        engine=new_engine,
        train_loader=loader,
        val_loader=None,
        checkpoint_manager=manager,
        max_epochs=2,
    )

    result = resumed.resume(checkpoint)

    assert len(result.history) >= 1
    assert result.history[0]["experiment_id"] == "resume-exp"
    assert new_engine.global_step > 0


def test_best_checkpoint_tracks_improvement(
    loader,
    components,
    tmp_path,
):
    model, criterion, optimizer = components

    runner = ExperimentRunner(
        experiment_id="exp-best",
        engine=make_engine(model, criterion, optimizer),
        train_loader=loader,
        val_loader=loader,
        checkpoint_manager=CheckpointManager(tmp_path),
        max_epochs=2,
    )

    result = runner.run()

    assert result.best_checkpoint is not None
    assert result.latest_checkpoint is not None
    assert result.best_epoch is not None


def test_non_finite_monitor_is_rejected(
    loader,
    components,
    tmp_path,
):
    model, criterion, optimizer = components

    runner = ExperimentRunner(
        experiment_id="exp-finite",
        engine=make_engine(model, criterion, optimizer),
        train_loader=loader,
        val_loader=None,
        checkpoint_manager=CheckpointManager(tmp_path),
        max_epochs=1,
    )

    fake_result = type(
        "Result",
        (),
        {
            "loss": float("nan"),
            "metrics": {},
        },
    )()

    with pytest.raises(ExperimentError):
        runner._extract_monitor_value(
            {"train": fake_result}
        )


def test_history_checkpoint_contains_resume_state(
    loader,
    components,
    tmp_path,
):
    model, criterion, optimizer = components

    runner = ExperimentRunner(
        experiment_id="exp-state",
        engine=make_engine(model, criterion, optimizer),
        train_loader=loader,
        val_loader=None,
        checkpoint_manager=CheckpointManager(tmp_path),
        max_epochs=1,
    )

    runner.run()

    loaded = CheckpointManager(tmp_path).load(
        tmp_path / "checkpoint-latest.pt"
    )

    assert loaded.extra["experiment_id"] == "exp-state"
    assert isinstance(loaded.extra["history"], list)
    assert loaded.extra["best_epoch"] == 0