from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import nullcontext
from typing import Any, Dict, Iterable, Mapping, Optional

import torch
from torch import Tensor, nn

from training.metrics import SegmentationMetrics


class TrainingEngineError(RuntimeError):
    """Raised when the training engine encounters an invalid state or batch."""


@dataclass(frozen=True)
class EpochResult:
    """Immutable result of one train or validation epoch."""

    epoch: int
    phase: str
    loss: float
    metrics: Mapping[str, float]
    samples: int
    batches: int
    optimizer_steps: int
    global_step: int
    skipped_batches: int = 0

    def __post_init__(self) -> None:
        if self.phase not in {"train", "val"}:
            raise ValueError("phase must be 'train' or 'val'")
        if self.epoch < 0:
            raise ValueError("epoch must be >= 0")
        if self.samples < 0 or self.batches < 0:
            raise ValueError("samples and batches must be >= 0")
        if self.optimizer_steps < 0 or self.global_step < 0:
            raise ValueError("step counters must be >= 0")
        if self.skipped_batches < 0:
            raise ValueError("skipped_batches must be >= 0")

        for name, value in self.metrics.items():
            if not isinstance(name, str):
                raise ValueError("metric names must be strings")
            if not isinstance(value, (int, float)):
                raise ValueError(f"metric {name!r} must be numeric")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "phase": self.phase,
            "loss": self.loss,
            "metrics": dict(self.metrics),
            "samples": self.samples,
            "batches": self.batches,
            "optimizer_steps": self.optimizer_steps,
            "global_step": self.global_step,
            "skipped_batches": self.skipped_batches,
        }


@dataclass
class _EpochAccumulator:
    loss_sum: float = 0.0
    samples: int = 0
    batches: int = 0
    skipped_batches: int = 0

    def update(self, loss: float, batch_size: int) -> None:
        self.loss_sum += float(loss) * batch_size
        self.samples += batch_size
        self.batches += 1

    @property
    def mean_loss(self) -> float:
        if self.samples == 0:
            return 0.0
        return self.loss_sum / self.samples


@dataclass
class TrainingEngine:
    """
    Production-oriented segmentation training engine.

    The engine deliberately keeps orchestration separate from:
      - model construction
      - loss construction
      - dataset construction
      - checkpoint persistence

    Expected batch contract:
        {
            "image": Tensor[B, C, H, W],
            "mask": Tensor[B, H, W],
            ...
        }

    The model must return:
        Tensor[B, num_classes, H, W]

    The loss callable must accept:
        loss(logits, target)
    """

    model: nn.Module
    criterion: nn.Module
    optimizer: torch.optim.Optimizer
    device: torch.device | str
    scheduler: Optional[Any] = None
    scaler: Optional[torch.amp.GradScaler] = None
    gradient_accumulation_steps: int = 1
    gradient_clip_norm: Optional[float] = None
    use_amp: bool = False
    amp_dtype: torch.dtype = torch.float16
    metrics_factory: Optional[Any] = None
    epoch: int = 0
    global_step: int = 0

    _optimizer_steps_this_epoch: int = field(
        default=0, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)

        if self.gradient_accumulation_steps < 1:
            raise TrainingEngineError(
                "gradient_accumulation_steps must be >= 1"
            )

        if self.gradient_clip_norm is not None:
            if (
                self.gradient_clip_norm <= 0
                or not torch.isfinite(
                    torch.tensor(float(self.gradient_clip_norm))
                )
            ):
                raise TrainingEngineError(
                    "gradient_clip_norm must be finite and > 0"
                )

        if self.epoch < 0 or self.global_step < 0:
            raise TrainingEngineError(
                "epoch and global_step must be >= 0"
            )

        self.model.to(self.device)

        if self.use_amp and self.device.type not in {
            "cuda",
            "cpu",
            "mps",
        }:
            raise TrainingEngineError(
                f"AMP is unsupported for device type {self.device.type!r}"
            )

        if self.scaler is None and self.use_amp and self.device.type == "cuda":
            self.scaler = torch.amp.GradScaler("cuda")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        dataloader: Iterable[Mapping[str, Any]],
        epoch: Optional[int] = None,
    ) -> EpochResult:
        """Run one complete training epoch."""

        resolved_epoch = self.epoch if epoch is None else int(epoch)
        if resolved_epoch < 0:
            raise TrainingEngineError("epoch must be >= 0")

        self.model.train()
        self._optimizer_steps_this_epoch = 0

        accumulator = _EpochAccumulator()
        metric_state = self._create_metric_state()

        self.optimizer.zero_grad(set_to_none=True)

        pending_accumulation = 0

        for batch_index, batch in enumerate(dataloader):
            try:
                images, targets = self._prepare_batch(batch)
                batch_size = int(images.shape[0])

                with self._autocast_context():
                    logits = self.model(images)
                    loss = self.criterion(logits, targets)

                self._validate_loss(loss, batch_index)

                scaled_loss = loss / self.gradient_accumulation_steps

                if self.scaler is not None and self.use_amp:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                pending_accumulation += 1

                self._update_metrics(
                    metric_state,
                    logits.detach(),
                    targets.detach(),
                )

                accumulator.update(
                    float(loss.detach().cpu()),
                    batch_size,
                )

                should_step = (
                    pending_accumulation
                    >= self.gradient_accumulation_steps
                )

                if should_step:
                    self._optimizer_step()
                    pending_accumulation = 0

            except TrainingEngineError:
                raise
            except (RuntimeError, ValueError, TypeError) as exc:
                raise TrainingEngineError(
                    f"Training failed on batch {batch_index}: {exc}"
                ) from exc

        # Flush a final partial accumulation window.
        if pending_accumulation > 0:
            self._optimizer_step()

        metrics = self._finalize_metrics(metric_state)

        self.epoch = resolved_epoch

        return EpochResult(
            epoch=resolved_epoch,
            phase="train",
            loss=accumulator.mean_loss,
            metrics=metrics,
            samples=accumulator.samples,
            batches=accumulator.batches,
            optimizer_steps=self._optimizer_steps_this_epoch,
            global_step=self.global_step,
            skipped_batches=accumulator.skipped_batches,
        )

    @torch.no_grad()
    def validate_epoch(
        self,
        dataloader: Iterable[Mapping[str, Any]],
        epoch: Optional[int] = None,
    ) -> EpochResult:
        """Run one complete validation epoch without parameter updates."""

        resolved_epoch = self.epoch if epoch is None else int(epoch)
        if resolved_epoch < 0:
            raise TrainingEngineError("epoch must be >= 0")

        self.model.eval()

        accumulator = _EpochAccumulator()
        metric_state = self._create_metric_state()

        for batch_index, batch in enumerate(dataloader):
            try:
                images, targets = self._prepare_batch(batch)

                with self._autocast_context():
                    logits = self.model(images)
                    loss = self.criterion(logits, targets)

                self._validate_loss(loss, batch_index)

                self._update_metrics(
                    metric_state,
                    logits,
                    targets,
                )

                accumulator.update(
                    float(loss.detach().cpu()),
                    int(images.shape[0]),
                )

            except TrainingEngineError:
                raise
            except (RuntimeError, ValueError, TypeError) as exc:
                raise TrainingEngineError(
                    f"Validation failed on batch {batch_index}: {exc}"
                ) from exc

        metrics = self._finalize_metrics(metric_state)

        return EpochResult(
            epoch=resolved_epoch,
            phase="val",
            loss=accumulator.mean_loss,
            metrics=metrics,
            samples=accumulator.samples,
            batches=accumulator.batches,
            optimizer_steps=0,
            global_step=self.global_step,
            skipped_batches=accumulator.skipped_batches,
        )

    def fit_epoch(
        self,
        train_loader: Iterable[Mapping[str, Any]],
        val_loader: Optional[Iterable[Mapping[str, Any]]] = None,
        epoch: Optional[int] = None,
    ) -> Dict[str, EpochResult]:
        """
        Execute training and optional validation for one epoch.

        Scheduler stepping is intentionally performed after validation
        when a validation loader exists.
        """

        resolved_epoch = self.epoch if epoch is None else int(epoch)

        train_result = self.train_epoch(
            train_loader,
            epoch=resolved_epoch,
        )

        results: Dict[str, EpochResult] = {
            "train": train_result,
        }

        if val_loader is not None:
            val_result = self.validate_epoch(
                val_loader,
                epoch=resolved_epoch,
            )
            results["val"] = val_result
            self._step_scheduler(val_result.loss, val_result.metrics)
        else:
            self._step_scheduler(None, None)

        self.epoch = resolved_epoch + 1

        return results

    # ------------------------------------------------------------------
    # Batch / model validation
    # ------------------------------------------------------------------

    def _prepare_batch(
        self,
        batch: Mapping[str, Any],
    ) -> tuple[Tensor, Tensor]:
        if not isinstance(batch, Mapping):
            raise TrainingEngineError(
                f"Expected mapping batch, got {type(batch).__name__}"
            )

        if "image" not in batch:
            raise TrainingEngineError("Batch is missing 'image'")
        if "mask" not in batch:
            raise TrainingEngineError("Batch is missing 'mask'")

        images = batch["image"]
        targets = batch["mask"]

        if not isinstance(images, Tensor):
            raise TrainingEngineError("'image' must be a torch.Tensor")
        if not isinstance(targets, Tensor):
            raise TrainingEngineError("'mask' must be a torch.Tensor")

        if images.ndim != 4:
            raise TrainingEngineError(
                f"Expected image [B,C,H,W], got {tuple(images.shape)}"
            )

        if targets.ndim != 3:
            raise TrainingEngineError(
                f"Expected mask [B,H,W], got {tuple(targets.shape)}"
            )

        if images.shape[0] != targets.shape[0]:
            raise TrainingEngineError("Image/mask batch sizes differ")

        if images.shape[-2:] != targets.shape[-2:]:
            raise TrainingEngineError(
                "Image/mask spatial dimensions differ"
            )

        if images.shape[0] == 0:
            raise TrainingEngineError("Empty batches are not supported")

        if not images.is_floating_point():
            images = images.float()

        images = images.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)

        return images, targets

    @staticmethod
    def _validate_loss(loss: Tensor, batch_index: int) -> None:
        if not isinstance(loss, Tensor):
            raise TrainingEngineError(
                f"Loss on batch {batch_index} is not a Tensor"
            )

        if loss.ndim != 0:
            raise TrainingEngineError(
                f"Loss must be scalar, got shape {tuple(loss.shape)}"
            )

        if not torch.isfinite(loss.detach()):
            raise TrainingEngineError(
                f"Non-finite loss encountered on batch {batch_index}"
            )

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------

    def _optimizer_step(self) -> None:
        if self.scaler is not None and self.use_amp:
            self.scaler.unscale_(self.optimizer)

        if self.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.gradient_clip_norm,
                error_if_nonfinite=True,
            )

        if self.scaler is not None and self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)

        self._optimizer_steps_this_epoch += 1
        self.global_step += 1

    def _step_scheduler(
        self,
        val_loss: Optional[float],
        metrics: Optional[Mapping[str, float]],
    ) -> None:
        if self.scheduler is None:
            return

        # ReduceLROnPlateau requires a monitored value.
        if isinstance(
            self.scheduler,
            torch.optim.lr_scheduler.ReduceLROnPlateau,
        ):
            if val_loss is None:
                raise TrainingEngineError(
                    "ReduceLROnPlateau requires validation loss"
                )
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()

    # ------------------------------------------------------------------
    # AMP
    # ------------------------------------------------------------------

    def _autocast_context(self):
        if not self.use_amp:
            return nullcontext()

        if self.device.type == "cuda":
            return torch.autocast(
                device_type="cuda",
                dtype=self.amp_dtype,
            )

        if self.device.type == "cpu":
            return torch.autocast(
                device_type="cpu",
                dtype=self.amp_dtype,
            )

        if self.device.type == "mps":
            return torch.autocast(
                device_type="mps",
                dtype=self.amp_dtype,
            )

        return nullcontext()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _create_metric_state(self) -> Any:
        if self.metrics_factory is None:
            return None
        return self.metrics_factory()

    @staticmethod
    def _update_metrics(
        metric_state: Any,
        logits: Tensor,
        targets: Tensor,
    ) -> None:
        if metric_state is None:
            return

        if hasattr(metric_state, "update"):
            metric_state.update(logits, targets)
            return

        raise TrainingEngineError(
            "metrics_factory must return an object exposing update()"
        )

    @staticmethod
    def _finalize_metrics(metric_state: Any) -> Dict[str, float]:
        if metric_state is None:
            return {}

        if hasattr(metric_state, "compute"):
            computed = metric_state.compute()
        elif hasattr(metric_state, "result"):
            computed = metric_state.result()
        else:
            raise TrainingEngineError(
                "Metric state must expose compute() or result()"
            )

        if computed is None:
            return {}

        result: Dict[str, float] = {}

        if not isinstance(computed, Mapping):
            raise TrainingEngineError(
                "Metric result must be a mapping"
            )

        for key, value in computed.items():
            if isinstance(value, Tensor):
                if value.numel() != 1:
                    raise TrainingEngineError(
                        f"Metric {key!r} must be scalar"
                    )
                value = value.detach().cpu().item()

            if not isinstance(value, (int, float)):
                raise TrainingEngineError(
                    f"Metric {key!r} is not numeric"
                )

            result[str(key)] = float(value)

        return result

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "gradient_clip_norm": self.gradient_clip_norm,
            "use_amp": self.use_amp,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise TrainingEngineError("Engine state must be a mapping")

        required = {"epoch", "global_step"}
        missing = required - set(state.keys())
        if missing:
            raise TrainingEngineError(
                f"Engine state missing keys: {sorted(missing)}"
            )

        epoch = int(state["epoch"])
        global_step = int(state["global_step"])

        if epoch < 0 or global_step < 0:
            raise TrainingEngineError(
                "Checkpoint engine counters must be >= 0"
            )

        saved_accumulation = state.get(
            "gradient_accumulation_steps",
            self.gradient_accumulation_steps,
        )

        if int(saved_accumulation) != self.gradient_accumulation_steps:
            raise TrainingEngineError(
                "Gradient accumulation configuration differs "
                "from the checkpoint"
            )

        self.epoch = epoch
        self.global_step = global_step