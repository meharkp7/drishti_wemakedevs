from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from torch import nn


class CheckpointError(RuntimeError):
    """Raised when checkpoint persistence or restoration fails."""


CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CheckpointMetadata:
    """Immutable metadata describing one checkpoint."""

    epoch: int
    global_step: int
    best_metric: Optional[float]
    metric_name: Optional[str]
    schema_version: int = CHECKPOINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.epoch < 0:
            raise ValueError("epoch must be >= 0")
        if self.global_step < 0:
            raise ValueError("global_step must be >= 0")

        if self.best_metric is not None:
            value = float(self.best_metric)
            if not torch.isfinite(torch.tensor(value)):
                raise ValueError("best_metric must be finite")

        if self.metric_name is not None and not self.metric_name.strip():
            raise ValueError("metric_name cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "metric_name": self.metric_name,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True)
class LoadedCheckpoint:
    """Validated checkpoint contents."""

    metadata: CheckpointMetadata
    model_metadata: Mapping[str, Any]
    training_config: Mapping[str, Any]
    metrics: Mapping[str, Any]
    extra: Mapping[str, Any]


class CheckpointManager:
    """
    Atomic, versioned PyTorch checkpoint manager.

    A checkpoint contains:
      - schema version
      - model state
      - optimizer state
      - scheduler state, when present
      - AMP scaler state, when present
      - engine state
      - model metadata
      - training configuration
      - epoch metrics
      - best metric
      - optional extra metadata

    Files are first written to a temporary file in the same directory and
    then atomically replaced into the requested destination.
    """

    def __init__(
        self,
        directory: str | os.PathLike[str],
        *,
        prefix: str = "checkpoint",
    ) -> None:
        self.directory = Path(directory)
        self.prefix = prefix

        if not prefix or "/" in prefix or "\\" in prefix:
            raise CheckpointError(
                "prefix must be a non-empty filename component"
            )

        self.directory.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        *,
        path: str | os.PathLike[str] | None = None,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        engine_state: Optional[Mapping[str, Any]] = None,
        epoch: int,
        global_step: int,
        best_metric: Optional[float] = None,
        metric_name: Optional[str] = None,
        model_metadata: Optional[Mapping[str, Any]] = None,
        training_config: Optional[Mapping[str, Any]] = None,
        metrics: Optional[Mapping[str, Any]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        if not isinstance(model, nn.Module):
            raise CheckpointError("model must be a torch.nn.Module")

        metadata = CheckpointMetadata(
            epoch=int(epoch),
            global_step=int(global_step),
            best_metric=best_metric,
            metric_name=metric_name,
        )

        if path is None:
            path = self.directory / (
                f"{self.prefix}-epoch-{metadata.epoch:04d}.pt"
            )
        else:
            path = Path(path)

        if not path.is_absolute():
            path = self.directory / path

        path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "metadata": metadata.to_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": (
                optimizer.state_dict()
                if optimizer is not None
                else None
            ),
            "scheduler_state": (
                scheduler.state_dict()
                if scheduler is not None
                and hasattr(scheduler, "state_dict")
                else None
            ),
            "scaler_state": (
                scaler.state_dict()
                if scaler is not None
                and hasattr(scaler, "state_dict")
                else None
            ),
            "engine_state": dict(engine_state or {}),
            "model_metadata": dict(model_metadata or {}),
            "training_config": dict(training_config or {}),
            "metrics": dict(metrics or {}),
            "extra": dict(extra or {}),
        }

        self._validate_payload(payload)

        self._atomic_torch_save(payload, path)

        return path

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(
        self,
        path: str | os.PathLike[str],
        *,
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        engine: Optional[Any] = None,
        map_location: str | torch.device = "cpu",
        strict_model: bool = True,
    ) -> LoadedCheckpoint:
        path = Path(path)

        if not path.exists():
            raise CheckpointError(
                f"Checkpoint does not exist: {path}"
            )

        if not path.is_file():
            raise CheckpointError(
                f"Checkpoint path is not a file: {path}"
            )

        try:
            payload = torch.load(
                path,
                map_location=map_location,
                weights_only=False,
            )
        except Exception as exc:
            raise CheckpointError(
                f"Failed to read checkpoint {path}: {exc}"
            ) from exc

        self._validate_payload(payload)

        if model is not None:
            self._restore_model(
                model,
                payload["model_state"],
                strict=strict_model,
            )

        if optimizer is not None:
            saved_optimizer = payload["optimizer_state"]
            if saved_optimizer is None:
                raise CheckpointError(
                    "Checkpoint does not contain optimizer state"
                )
            try:
                optimizer.load_state_dict(saved_optimizer)
            except Exception as exc:
                raise CheckpointError(
                    f"Failed to restore optimizer state: {exc}"
                ) from exc

        if scheduler is not None:
            saved_scheduler = payload["scheduler_state"]
            if saved_scheduler is None:
                raise CheckpointError(
                    "Checkpoint does not contain scheduler state"
                )
            try:
                scheduler.load_state_dict(saved_scheduler)
            except Exception as exc:
                raise CheckpointError(
                    f"Failed to restore scheduler state: {exc}"
                ) from exc

        if scaler is not None:
            saved_scaler = payload["scaler_state"]
            if saved_scaler is None:
                raise CheckpointError(
                    "Checkpoint does not contain scaler state"
                )
            try:
                scaler.load_state_dict(saved_scaler)
            except Exception as exc:
                raise CheckpointError(
                    f"Failed to restore AMP scaler state: {exc}"
                ) from exc

        if engine is not None:
            saved_engine = payload["engine_state"]
            if not hasattr(engine, "load_state_dict"):
                raise CheckpointError(
                    "engine must expose load_state_dict()"
                )
            try:
                engine.load_state_dict(saved_engine)
            except Exception as exc:
                raise CheckpointError(
                    f"Failed to restore engine state: {exc}"
                ) from exc

        metadata = payload["metadata"]

        return LoadedCheckpoint(
            metadata=CheckpointMetadata(
                epoch=int(metadata["epoch"]),
                global_step=int(metadata["global_step"]),
                best_metric=metadata.get("best_metric"),
                metric_name=metadata.get("metric_name"),
                schema_version=int(metadata["schema_version"]),
            ),
            model_metadata=dict(payload["model_metadata"]),
            training_config=dict(payload["training_config"]),
            metrics=dict(payload["metrics"]),
            extra=dict(payload["extra"]),
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def save_latest(
        self,
        **kwargs: Any,
    ) -> Path:
        return self.save(
            path=self.directory / f"{self.prefix}-latest.pt",
            **kwargs,
        )

    def save_best(
        self,
        **kwargs: Any,
    ) -> Path:
        return self.save(
            path=self.directory / f"{self.prefix}-best.pt",
            **kwargs,
        )

    @staticmethod
    def read_metadata(
        path: str | os.PathLike[str],
        *,
        map_location: str | torch.device = "cpu",
    ) -> CheckpointMetadata:
        try:
            payload = torch.load(
                Path(path),
                map_location=map_location,
                weights_only=False,
            )
        except Exception as exc:
            raise CheckpointError(
                f"Failed to read checkpoint metadata: {exc}"
            ) from exc

        CheckpointManager._validate_payload(payload)

        metadata = payload["metadata"]

        return CheckpointMetadata(
            epoch=int(metadata["epoch"]),
            global_step=int(metadata["global_step"]),
            best_metric=metadata.get("best_metric"),
            metric_name=metadata.get("metric_name"),
            schema_version=int(metadata["schema_version"]),
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_payload(payload: Any) -> None:
        if not isinstance(payload, Mapping):
            raise CheckpointError(
                "Checkpoint root must be a mapping"
            )

        required = {
            "schema_version",
            "metadata",
            "model_state",
            "optimizer_state",
            "scheduler_state",
            "scaler_state",
            "engine_state",
            "model_metadata",
            "training_config",
            "metrics",
            "extra",
        }

        missing = required - set(payload.keys())
        if missing:
            raise CheckpointError(
                f"Checkpoint missing keys: {sorted(missing)}"
            )

        version = payload["schema_version"]
        if int(version) != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointError(
                f"Unsupported checkpoint schema version: {version}"
            )

        metadata = payload["metadata"]

        if not isinstance(metadata, Mapping):
            raise CheckpointError("Checkpoint metadata must be a mapping")

        for key in (
            "epoch",
            "global_step",
            "best_metric",
            "metric_name",
            "schema_version",
        ):
            if key not in metadata:
                raise CheckpointError(
                    f"Checkpoint metadata missing {key!r}"
                )

        if int(metadata["epoch"]) < 0:
            raise CheckpointError("Checkpoint epoch must be >= 0")

        if int(metadata["global_step"]) < 0:
            raise CheckpointError(
                "Checkpoint global_step must be >= 0"
            )

        if int(metadata["schema_version"]) != version:
            raise CheckpointError(
                "Checkpoint metadata schema version mismatch"
            )

        if not isinstance(payload["model_state"], Mapping):
            raise CheckpointError(
                "model_state must be a mapping"
            )

        for key in (
            "engine_state",
            "model_metadata",
            "training_config",
            "metrics",
            "extra",
        ):
            if not isinstance(payload[key], Mapping):
                raise CheckpointError(
                    f"{key} must be a mapping"
                )

    @staticmethod
    def _restore_model(
        model: nn.Module,
        state: Mapping[str, Any],
        *,
        strict: bool,
    ) -> None:
        try:
            result = model.load_state_dict(
                state,
                strict=strict,
            )
        except Exception as exc:
            raise CheckpointError(
                f"Model state is incompatible with current model: {exc}"
            ) from exc

        if not strict:
            missing = list(getattr(result, "missing_keys", []))
            unexpected = list(getattr(result, "unexpected_keys", []))

            # Non-strict loading is explicitly allowed, but the caller
            # can inspect these keys through normal PyTorch semantics.
            if missing or unexpected:
                return

    # ------------------------------------------------------------------
    # Atomic persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _atomic_torch_save(
        payload: Mapping[str, Any],
        destination: Path,
    ) -> None:
        destination = destination.resolve()
        parent = destination.parent
        parent.mkdir(parents=True, exist_ok=True)

        temp_path: Optional[Path] = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=parent,
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())

            os.replace(temp_path, destination)

            # Best-effort directory durability.
            try:
                dir_fd = os.open(
                    parent,
                    os.O_RDONLY,
                )
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except OSError:
                pass

        except Exception as exc:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass

            raise CheckpointError(
                f"Atomic checkpoint write failed: {exc}"
            ) from exc


def save_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    engine_state: Optional[Mapping[str, Any]] = None,
    epoch: int,
    global_step: int,
    best_metric: Optional[float] = None,
    metric_name: Optional[str] = None,
    model_metadata: Optional[Mapping[str, Any]] = None,
    training_config: Optional[Mapping[str, Any]] = None,
    metrics: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Stateless convenience wrapper around CheckpointManager."""

    path = Path(path)

    manager = CheckpointManager(path.parent)

    return manager.save(
        path=path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        engine_state=engine_state,
        epoch=epoch,
        global_step=global_step,
        best_metric=best_metric,
        metric_name=metric_name,
        model_metadata=model_metadata,
        training_config=training_config,
        metrics=metrics,
        extra=extra,
    )