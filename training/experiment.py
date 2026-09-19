from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import json
import os
import random

import numpy as np
import torch

from training.checkpoint import CheckpointManager
from training.engine import EpochResult, TrainingEngine


class ExperimentError(RuntimeError):
    """Raised when experiment orchestration fails."""


@dataclass(frozen=True)
class ExperimentResult:
    """Immutable summary of an experiment run."""

    experiment_id: str
    completed_epochs: int
    best_metric: Optional[float]
    best_epoch: Optional[int]
    history: tuple[Mapping[str, Any], ...]
    best_checkpoint: Optional[str]
    latest_checkpoint: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "completed_epochs": self.completed_epochs,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "history": [dict(item) for item in self.history],
            "best_checkpoint": self.best_checkpoint,
            "latest_checkpoint": self.latest_checkpoint,
        }


@dataclass
class ExperimentRunner:
    """
    Production-oriented experiment orchestration.

    Responsibilities:
      - deterministic seeding
      - epoch orchestration
      - monitored metric handling
      - best/latest checkpoints
      - early stopping
      - persistent experiment history
      - resume support
    """

    experiment_id: str
    engine: TrainingEngine
    train_loader: Any
    val_loader: Optional[Any]
    checkpoint_manager: CheckpointManager

    max_epochs: int
    monitor: str = "loss"
    monitor_mode: str = "min"
    patience: Optional[int] = None
    min_delta: float = 0.0
    seed: Optional[int] = None

    model_metadata: Mapping[str, Any] = field(default_factory=dict)
    training_config: Mapping[str, Any] = field(default_factory=dict)

    history: list[Mapping[str, Any]] = field(default_factory=list)

    best_metric: Optional[float] = None
    best_epoch: Optional[int] = None
    epochs_without_improvement: int = 0

    def __post_init__(self) -> None:
        if not self.experiment_id.strip():
            raise ExperimentError("experiment_id cannot be empty")

        if self.max_epochs < 1:
            raise ExperimentError("max_epochs must be >= 1")

        if self.monitor_mode not in {"min", "max"}:
            raise ExperimentError(
                "monitor_mode must be 'min' or 'max'"
            )

        if self.patience is not None and self.patience < 0:
            raise ExperimentError("patience must be >= 0")

        if not np.isfinite(self.min_delta) or self.min_delta < 0:
            raise ExperimentError(
                "min_delta must be finite and >= 0"
            )

        if self.seed is not None:
            self._seed_everything(self.seed)

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        start_epoch: Optional[int] = None,
        on_epoch_end: Optional[
            Callable[[Mapping[str, Any]], None]
        ] = None,
    ) -> ExperimentResult:
        if start_epoch is None:
            start_epoch = self.engine.epoch

        if start_epoch < 0:
            raise ExperimentError("start_epoch must be >= 0")

        if start_epoch >= self.max_epochs:
            raise ExperimentError(
                f"start_epoch {start_epoch} >= max_epochs {self.max_epochs}"
            )

        for epoch in range(start_epoch, self.max_epochs):
            results = self.engine.fit_epoch(
                self.train_loader,
                self.val_loader,
                epoch=epoch,
            )

            record = self._build_epoch_record(results)
            self.history.append(record)

            monitored_value = self._extract_monitor_value(results)
            improved = self._is_improvement(monitored_value)

            if improved:
                self.best_metric = monitored_value
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            self._save_latest(record)

            if improved:
                self._save_best(record)

            self._persist_history()

            if on_epoch_end is not None:
                on_epoch_end(record)

            if self._should_stop():
                break

        return self._result()

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    def resume(
        self,
        checkpoint_path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
        strict_model: bool = True,
    ) -> ExperimentResult:
        loaded = self.checkpoint_manager.load(
            checkpoint_path,
            model=self.engine.model,
            optimizer=self.engine.optimizer,
            scheduler=self.engine.scheduler,
            scaler=self.engine.scaler,
            engine=self.engine,
            map_location=map_location,
            strict_model=strict_model,
        )

        saved_experiment = loaded.extra.get("experiment_id")

        if (
            saved_experiment is not None
            and saved_experiment != self.experiment_id
        ):
            raise ExperimentError(
                "Checkpoint belongs to a different experiment: "
                f"{saved_experiment!r}"
            )

        saved_history = loaded.extra.get("history")

        if saved_history is not None:
            if not isinstance(saved_history, list):
                raise ExperimentError(
                    "Checkpoint history must be a list"
                )
            self.history = list(saved_history)

        self.best_metric = loaded.metadata.best_metric

        saved_best_epoch = loaded.extra.get("best_epoch")
        if saved_best_epoch is not None:
            self.best_epoch = int(saved_best_epoch)
        elif self.best_metric is not None:
            self.best_epoch = loaded.metadata.epoch

        self.epochs_without_improvement = int(
            loaded.extra.get(
                "epochs_without_improvement",
                0,
            )
        )

        next_epoch = self.engine.epoch + 1

        if next_epoch >= self.max_epochs:
            return self._result()

        return self.run(start_epoch=next_epoch)

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _checkpoint_extra(
        self,
        *,
        record: Mapping[str, Any],
        is_best: bool = False,
    ) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "is_best": is_best,
            "best_epoch": self.best_epoch,
            "epochs_without_improvement":
                self.epochs_without_improvement,
            "history": [dict(item) for item in self.history],
            "last_epoch_record": dict(record),
        }

    def _save_latest(self, record: Mapping[str, Any]) -> Path:
        return self.checkpoint_manager.save_latest(
            model=self.engine.model,
            optimizer=self.engine.optimizer,
            scheduler=self.engine.scheduler,
            scaler=self.engine.scaler,
            engine_state=self.engine.state_dict(),
            epoch=self.engine.epoch,
            global_step=self.engine.global_step,
            best_metric=self.best_metric,
            metric_name=self.monitor,
            model_metadata=self.model_metadata,
            training_config=self.training_config,
            metrics=record,
            extra=self._checkpoint_extra(record=record),
        )

    def _save_best(self, record: Mapping[str, Any]) -> Path:
        return self.checkpoint_manager.save_best(
            model=self.engine.model,
            optimizer=self.engine.optimizer,
            scheduler=self.engine.scheduler,
            scaler=self.engine.scaler,
            engine_state=self.engine.state_dict(),
            epoch=self.engine.epoch,
            global_step=self.engine.global_step,
            best_metric=self.best_metric,
            metric_name=self.monitor,
            model_metadata=self.model_metadata,
            training_config=self.training_config,
            metrics=record,
            extra=self._checkpoint_extra(
                record=record,
                is_best=True,
            ),
        )

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    def _extract_monitor_value(
        self,
        results: Mapping[str, EpochResult],
    ) -> float:
        source = results.get("val") or results.get("train")

        if source is None:
            raise ExperimentError("No epoch result available")

        if self.monitor == "loss":
            value = source.loss
        elif self.monitor in source.metrics:
            value = source.metrics[self.monitor]
        else:
            raise ExperimentError(
                f"Monitored metric {self.monitor!r} was not produced"
            )

        value = float(value)

        if not np.isfinite(value):
            raise ExperimentError(
                f"Monitored metric {self.monitor!r} is non-finite"
            )

        return value

    def _is_improvement(self, value: float) -> bool:
        if self.best_metric is None:
            return True

        if self.monitor_mode == "min":
            return value < self.best_metric - self.min_delta

        return value > self.best_metric + self.min_delta

    def _should_stop(self) -> bool:
        if self.patience is None:
            return False

        return self.epochs_without_improvement > self.patience

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def _build_epoch_record(
        self,
        results: Mapping[str, EpochResult],
    ) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "epoch": self.engine.epoch,
            "global_step": self.engine.global_step,
            "train": (
                results["train"].to_dict()
                if "train" in results
                else None
            ),
            "val": (
                results["val"].to_dict()
                if "val" in results
                else None
            ),
        }

    def _persist_history(self) -> None:
        path = (
            self.checkpoint_manager.directory
            / f"{self.checkpoint_manager.prefix}-history.json"
        )

        payload = {
            "experiment_id": self.experiment_id,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "epochs_without_improvement":
                self.epochs_without_improvement,
            "history": self.history,
        }

        temp_path = path.with_name(
            f".{path.name}.tmp"
        )

        try:
            with temp_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    payload,
                    handle,
                    indent=2,
                    sort_keys=True,
                )
                handle.flush()
                os.fsync(handle.fileno())

            os.replace(temp_path, path)

            try:
                directory_fd = os.open(
                    path.parent,
                    os.O_RDONLY,
                )
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                pass

        except Exception as exc:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass

            raise ExperimentError(
                f"Failed to persist experiment history: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------

    def _result(self) -> ExperimentResult:
        best_path = (
            self.checkpoint_manager.directory
            / f"{self.checkpoint_manager.prefix}-best.pt"
        )

        latest_path = (
            self.checkpoint_manager.directory
            / f"{self.checkpoint_manager.prefix}-latest.pt"
        )

        return ExperimentResult(
            experiment_id=self.experiment_id,
            completed_epochs=len(self.history),
            best_metric=self.best_metric,
            best_epoch=self.best_epoch,
            history=tuple(self.history),
            best_checkpoint=(
                str(best_path)
                if self.best_epoch is not None
                and best_path.exists()
                else None
            ),
            latest_checkpoint=(
                str(latest_path)
                if latest_path.exists()
                else None
            ),
        )

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------

    @staticmethod
    def _seed_everything(seed: int) -> None:
        if seed < 0:
            raise ExperimentError("seed must be >= 0")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        try:
            torch.use_deterministic_algorithms(
                True,
                warn_only=True,
            )
        except Exception:
            pass

        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False