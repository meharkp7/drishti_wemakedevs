from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import json
import math
import os
import random

import numpy as np
import torch
from torch import nn

from training.checkpoint import CheckpointManager
from training.config import TrainingConfig
from training.engine import TrainingEngine
from training.experiment import ExperimentRunner
from training.losses import build_loss_from_training_config
from training.metrics import SegmentationMetricAccumulator
from training.models.registry import build_model


LOVE_DA_CLASS_IDS = (1, 2, 3, 4, 5, 6, 7)
LOVE_DA_IGNORE_INDEX = 0
LOVE_DA_CLASS_NAMES = {
    1: "background",
    2: "building",
    3: "road",
    4: "water",
    5: "barren",
    6: "forest",
    7: "agriculture",
}


class BenchmarkError(RuntimeError):
    """Raised when benchmark configuration or execution fails."""


@dataclass(frozen=True)
class BenchmarkEvaluation:
    """Scientific evaluation artifact for one trained model."""

    overall: Mapping[str, Any]
    per_class: Mapping[str, Mapping[str, float]]
    confusion_matrix: tuple[tuple[int, ...], ...]
    domain_metrics: Mapping[str, Mapping[str, Any]]
    valid_pixels: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": _json_safe(self.overall),
            "per_class": _json_safe(self.per_class),
            "confusion_matrix": [
                list(row)
                for row in self.confusion_matrix
            ],
            "domain_metrics": _json_safe(
                self.domain_metrics
            ),
            "valid_pixels": self.valid_pixels,
        }


@dataclass(frozen=True)
class BenchmarkModelResult:
    """Immutable result for one benchmarked model."""

    model_id: str
    model_version: str
    architecture: str
    experiment_id: str

    parameter_count: Optional[int]

    best_epoch: Optional[int]
    best_metric: Optional[float]
    checkpoint_path: Optional[str]

    history: tuple[Mapping[str, Any], ...]

    evaluation: BenchmarkEvaluation

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "architecture": self.architecture,
            "experiment_id": self.experiment_id,
            "parameter_count": self.parameter_count,
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "checkpoint_path": self.checkpoint_path,
            "history": [
                _json_safe(item)
                for item in self.history
            ],
            "evaluation": self.evaluation.to_dict(),
        }


@dataclass(frozen=True)
class BenchmarkResult:
    """Immutable benchmark-level result."""

    benchmark_id: str
    dataset: str
    seed: Optional[int]
    class_ids: tuple[int, ...]
    class_names: Mapping[int, str]
    models: tuple[BenchmarkModelResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "dataset": self.dataset,
            "seed": self.seed,
            "class_ids": list(self.class_ids),
            "class_names": {
                str(k): v
                for k, v in self.class_names.items()
            },
            "models": [
                model.to_dict()
                for model in self.models
            ],
        }


@dataclass
class BenchmarkRunner:
    """
    Reproducible LoveDA model benchmark.

    Responsibilities:
      - build models through the registry
      - build losses through the loss factory
      - build optimizer/scheduler
      - train through TrainingEngine + ExperimentRunner
      - monitor the canonical configured validation metric
      - evaluate the selected best checkpoint
      - produce overall, per-class and urban/rural metrics
      - persist an atomic JSON benchmark artifact

    Dataset taxonomy is intentionally fixed to LoveDA for this benchmark.
    """

    benchmark_id: str
    training_config: TrainingConfig

    train_loader_factory: Callable[[], Any]
    val_loader_factory: Callable[[], Any]

    model_ids: Sequence[str]

    output_dir: str | Path

    loss_factory: Callable[..., nn.Module] = (
        build_loss_from_training_config
    )

    model_builder: Callable[..., nn.Module] = build_model

    seed: Optional[int] = None

    results: list[BenchmarkModelResult] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        if not isinstance(self.benchmark_id, str):
            raise BenchmarkError(
                "benchmark_id must be a string."
            )

        if not self.benchmark_id.strip():
            raise BenchmarkError(
                "benchmark_id cannot be empty."
            )

        if not self.model_ids:
            raise BenchmarkError(
                "At least one model_id is required."
            )

        normalized = tuple(
            str(model_id).strip()
            for model_id in self.model_ids
        )

        if any(not item for item in normalized):
            raise BenchmarkError(
                "model_ids cannot contain empty values."
            )

        if len(set(normalized)) != len(normalized):
            raise BenchmarkError(
                "model_ids must be unique."
            )

        self.model_ids = normalized
        self.output_dir = Path(self.output_dir)

        if self.seed is not None:
            if not isinstance(self.seed, int):
                raise BenchmarkError(
                    "seed must be an integer or None."
                )

            if self.seed < 0:
                raise BenchmarkError(
                    "seed must be >= 0."
                )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        on_model_complete: Optional[
            Callable[[BenchmarkModelResult], None]
        ] = None,
    ) -> BenchmarkResult:
        """Run all configured models sequentially."""

        self.results.clear()

        self._apply_seed()

        for model_id in self.model_ids:
            result = self._run_model(model_id)

            self.results.append(result)

            if on_model_complete is not None:
                on_model_complete(result)

            self._persist_result()

        return BenchmarkResult(
            benchmark_id=self.benchmark_id,
            dataset="LoveDA",
            seed=self.seed,
            class_ids=LOVE_DA_CLASS_IDS,
            class_names=LOVE_DA_CLASS_NAMES,
            models=tuple(self.results),
        )

    # ------------------------------------------------------------------
    # Model execution
    # ------------------------------------------------------------------

    def _run_model(
        self,
        model_id: str,
    ) -> BenchmarkModelResult:
        self._apply_seed()

        model = self._build_model(model_id)

        metadata = self._model_metadata(
            model,
            model_id,
        )

        criterion = self._build_loss()

        optimizer = self._build_optimizer(model)

        scheduler = self._build_scheduler(
            optimizer
        )

        train_loader = self.train_loader_factory()
        val_loader = self.val_loader_factory()

        if train_loader is None:
            raise BenchmarkError(
                f"Train loader factory returned None "
                f"for {model_id!r}."
            )

        if val_loader is None:
            raise BenchmarkError(
                f"Validation loader factory returned None "
                f"for {model_id!r}."
            )

        device = self._resolve_device()

        engine = TrainingEngine(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            gradient_accumulation_steps=int(
                self._config_value(
                    "gradient_accumulation_steps",
                    1,
                )
            ),
            gradient_clip_norm=self._config_value(
                "gradient_clip_norm",
                None,
            ),
            use_amp=self._effective_amp(
                device
            ),
            amp_dtype=self._amp_dtype(),
            metrics_factory=self._metric_factory,
        )

        experiment_id = (
            f"{self.benchmark_id}-{model_id}"
        )

        checkpoint_dir = (
            self.output_dir / model_id
        )

        checkpoint_manager = CheckpointManager(
            checkpoint_dir,
            prefix="checkpoint",
        )

        runner = ExperimentRunner(
            experiment_id=experiment_id,
            engine=engine,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_manager=checkpoint_manager,
            max_epochs=int(
                self._config_value(
                    "epochs",
                    1,
                )
            ),
            monitor=str(
                self._config_value(
                    "monitor_metric",
                    "mean_iou",
                )
            ),
            monitor_mode=str(
                self._config_value(
                    "monitor_mode",
                    "max",
                )
            ),
            patience=self._config_value(
                "early_stopping_patience",
                None,
            ),
            min_delta=float(
                self._config_value(
                    "min_delta",
                    0.0,
                )
            ),
            seed=self.seed,
            model_metadata=metadata,
            training_config=self._config_to_dict(),
        )

        try:
            experiment_result = runner.run()
        except Exception as exc:
            raise BenchmarkError(
                f"Benchmark training failed for "
                f"{model_id!r}: {exc}"
            ) from exc

        checkpoint_path = (
            experiment_result.best_checkpoint
        )

        if not checkpoint_path:
            raise BenchmarkError(
                f"No best checkpoint was produced for "
                f"{model_id!r}."
            )

        self._restore_model(
            model,
            checkpoint_path,
        )

        evaluation = self._evaluate(
            model,
            val_loader,
            device,
        )

        return BenchmarkModelResult(
            model_id=model_id,
            model_version=str(
                getattr(
                    model,
                    "model_version",
                    "unknown",
                )
            ),
            architecture=str(
                getattr(
                    model,
                    "architecture",
                    model_id,
                )
            ),
            experiment_id=experiment_id,
            parameter_count=self._parameter_count(
                model
            ),
            best_epoch=experiment_result.best_epoch,
            best_metric=experiment_result.best_metric,
            checkpoint_path=str(
                checkpoint_path
            ),
            history=tuple(
                _json_safe(item)
                for item in experiment_result.history
            ),
            evaluation=evaluation,
        )

    # ------------------------------------------------------------------
    # Model / loss
    # ------------------------------------------------------------------

    def _build_model(
        self,
        model_id: str,
    ) -> nn.Module:
        try:
            model = self.model_builder(
                model_id,
                num_classes=len(
                    LOVE_DA_CLASS_IDS
                ),
                in_channels=3,
                class_ids=LOVE_DA_CLASS_IDS,
            )
        except Exception as exc:
            raise BenchmarkError(
                f"Failed to build model "
                f"{model_id!r}: {exc}"
            ) from exc

        if not isinstance(model, nn.Module):
            raise BenchmarkError(
                f"Model builder returned "
                f"{type(model).__name__}, not nn.Module."
            )

        return model

    def _build_loss(self) -> nn.Module:
        try:
            return self.loss_factory(
                self.training_config
            )
        except TypeError:
            try:
                return self.loss_factory()
            except Exception as exc:
                raise BenchmarkError(
                    f"Failed to build loss: {exc}"
                ) from exc
        except Exception as exc:
            raise BenchmarkError(
                f"Failed to build loss: {exc}"
            ) from exc

    @staticmethod
    def _metric_factory() -> "_BenchmarkMetricAdapter":
        """Create the canonical LoveDA metric adapter for TrainingEngine."""
        return _BenchmarkMetricAdapter()

    @staticmethod
    def _parameter_count(
        model: nn.Module,
    ) -> int:
        return sum(
            parameter.numel()
            for parameter in model.parameters()
        )

    def _evaluate(
        self,
        model: nn.Module,
        loader: Any,
        device: torch.device,
    ) -> BenchmarkEvaluation:
        model.eval()

        overall = _BenchmarkMetricAdapter()

        domain_accumulators: dict[
            str,
            Any,
        ] = {}

        batches_seen = 0

        try:
            with torch.inference_mode():
                for batch in loader:
                    batches_seen += 1

                    if not isinstance(
                        batch,
                        Mapping,
                    ):
                        raise BenchmarkError(
                            "Validation batch must be a mapping."
                        )

                    if "image" not in batch:
                        raise BenchmarkError(
                            "Validation batch is missing "
                            "'image'."
                        )

                    if "mask" not in batch:
                        raise BenchmarkError(
                            "Validation batch is missing "
                            "'mask'."
                        )

                    images = batch["image"]
                    masks = batch["mask"]

                    if not isinstance(
                        images,
                        torch.Tensor,
                    ):
                        raise BenchmarkError(
                            "'image' must be a tensor."
                        )

                    if not isinstance(
                        masks,
                        torch.Tensor,
                    ):
                        raise BenchmarkError(
                            "'mask' must be a tensor."
                        )

                    images = images.to(
                        device,
                        non_blocking=True,
                    )
                    masks = masks.to(
                        device,
                        non_blocking=True,
                    )

                    logits = model(images)

                    overall.update(
                        logits,
                        masks,
                    )

                    domains = batch.get(
                        "domains"
                    )

                    if domains is None:
                        continue

                    if isinstance(
                        domains,
                        str,
                    ):
                        domains = [domains]

                    domains = list(domains)

                    if len(domains) != (
                        images.shape[0]
                    ):
                        raise BenchmarkError(
                            "Domain metadata length does not "
                            "match validation batch size."
                        )

                    unique_domains = {
                        str(domain).strip().lower()
                        for domain in domains
                    }

                    if "" in unique_domains:
                        raise BenchmarkError(
                            "Validation batch contains "
                            "an empty domain."
                        )

                    for domain in unique_domains:
                        if domain not in {
                            "urban",
                            "rural",
                        }:
                            raise BenchmarkError(
                                "Unsupported LoveDA domain: "
                                f"{domain!r}"
                            )

                        if domain not in domain_accumulators:
                            domain_accumulators[
                                domain
                            ] = _BenchmarkMetricAdapter()

                    for index, domain in enumerate(
                        domains
                    ):
                        domain = (
                            str(domain)
                            .strip()
                            .lower()
                        )

                        domain_accumulators[
                            domain
                        ].update(
                            logits[
                                index:index + 1
                            ],
                            masks[
                                index:index + 1
                            ],
                        )

        except BenchmarkError:
            raise
        except Exception as exc:
            raise BenchmarkError(
                f"Validation evaluation failed: {exc}"
            ) from exc

        if batches_seen == 0:
            raise BenchmarkError(
                "Validation loader produced zero batches."
            )

        overall_raw = overall.raw

        overall_metrics = self._metrics_to_dict(
            overall_raw
        )

        per_class = self._per_class_metrics(
            overall_raw
        )

        confusion = overall.confusion_matrix

        confusion_matrix = tuple(
            tuple(
                int(value)
                for value in row
            )
            for row in confusion.tolist()
        )

        domain_metrics = {
            domain: self._metrics_to_dict(
                accumulator.raw
            )
            for domain, accumulator
            in sorted(
                domain_accumulators.items()
            )
        }

        return BenchmarkEvaluation(
            overall=overall_metrics,
            per_class=per_class,
            confusion_matrix=confusion_matrix,
            domain_metrics=domain_metrics,
            valid_pixels=overall.valid_pixels,
        )

    @staticmethod
    def _metrics_to_dict(
        metrics: Any,
    ) -> dict[str, Any]:
        """
        Convert SegmentationMetrics into JSON-safe scalar metrics.

        Per-class fields and confusion_matrix are deliberately kept in their
        dedicated evaluation sections.
        """
        result: dict[str, Any] = {}

        scalar_fields = (
            "pixel_accuracy",
            "mean_class_accuracy",
            "mean_iou",
            "frequency_weighted_iou",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "valid_pixels",
        )

        for field_name in scalar_fields:
            if not hasattr(
                metrics,
                field_name,
            ):
                continue

            value = getattr(
                metrics,
                field_name,
            )

            if isinstance(
                value,
                (int, float),
            ):
                result[field_name] = (
                    float(value)
                    if field_name != "valid_pixels"
                    else int(value)
                )

        if (
            "macro_dice"
            not in result
        ):
            f1 = result.get(
                "macro_f1"
            )

            if f1 is not None:
                result["macro_dice"] = f1

        return _json_safe(result)

    @staticmethod
    def _per_class_metrics(
        metrics: Any,
    ) -> dict[str, dict[str, float]]:
        result: dict[
            str,
            dict[str, float],
        ] = {}

        field_mapping = {
            "iou": "per_class_iou",
            "precision": "per_class_precision",
            "recall": "per_class_recall",
            "f1": "per_class_f1",
        }

        for class_id in LOVE_DA_CLASS_IDS:
            result[str(class_id)] = {
                "class_name": LOVE_DA_CLASS_NAMES[
                    class_id
                ],
            }

            for output_name, field_name in (
                field_mapping.items()
            ):
                values = getattr(
                    metrics,
                    field_name,
                    {},
                )

                value = values.get(
                    class_id,
                    float("nan"),
                )

                result[str(class_id)][
                    output_name
                ] = float(value)

        return result

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def _build_optimizer(
        self,
        model: nn.Module,
    ) -> torch.optim.Optimizer:
        parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ]

        if not parameters:
            raise BenchmarkError(
                "Model has no trainable parameters."
            )

        optimizer_name = str(
            self._config_value(
                "optimizer",
                "adamw",
            )
        ).lower()

        lr = float(
            self._config_value(
                "learning_rate",
                3e-4,
            )
        )

        weight_decay = float(
            self._config_value(
                "weight_decay",
                0.0,
            )
        )

        if not math.isfinite(lr) or lr <= 0:
            raise BenchmarkError(
                "learning_rate must be finite and > 0."
            )

        if (
            not math.isfinite(weight_decay)
            or weight_decay < 0
        ):
            raise BenchmarkError(
                "weight_decay must be finite and >= 0."
            )

        if optimizer_name == "adamw":
            return torch.optim.AdamW(
                parameters,
                lr=lr,
                weight_decay=weight_decay,
            )

        if optimizer_name == "sgd":
            momentum = float(
                self._config_value(
                    "momentum",
                    0.9,
                )
            )

            if (
                not math.isfinite(momentum)
                or momentum < 0
                or momentum >= 1
            ):
                raise BenchmarkError(
                    "SGD momentum must be in [0, 1)."
                )

            return torch.optim.SGD(
                parameters,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )

        raise BenchmarkError(
            f"Unsupported optimizer: "
            f"{optimizer_name!r}"
        )

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------

    def _build_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> Optional[Any]:
        name = str(
            self._config_value(
                "scheduler",
                "none",
            )
        ).lower()

        if name in {
            "",
            "none",
            "null",
        }:
            return None

        epochs = int(
            self._config_value(
                "epochs",
                1,
            )
        )

        if epochs <= 0:
            raise BenchmarkError(
                "epochs must be > 0."
            )

        if name == "cosine":
            min_lr = float(
                self._config_value(
                    "min_learning_rate",
                    1e-6,
                )
            )

            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=min_lr,
            )

        if name == "plateau":
            mode = str(
                self._config_value(
                    "monitor_mode",
                    "max",
                )
            ).lower()

            if mode not in {
                "min",
                "max",
            }:
                raise BenchmarkError(
                    "monitor_mode must be "
                    "'min' or 'max'."
                )

            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=mode,
                factor=0.1,
                patience=2,
            )

        if name == "one_cycle":
            steps_per_epoch = self._loader_length_hint(
                self.train_loader_factory()
            )

            if steps_per_epoch is None:
                raise BenchmarkError(
                    "one_cycle scheduler requires a sized "
                    "training loader."
                )

            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=float(
                    self._config_value(
                        "learning_rate",
                        3e-4,
                    )
                ),
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
            )

        if name == "cosine_warmup":
            warmup_epochs = int(
                self._config_value(
                    "warmup_epochs",
                    0,
                )
            )

            warmup_epochs = max(
                0,
                min(
                    warmup_epochs,
                    epochs,
                ),
            )

            min_lr = float(
                self._config_value(
                    "min_learning_rate",
                    1e-6,
                )
            )

            def lr_lambda(epoch: int) -> float:
                if warmup_epochs > 0 and (
                    epoch < warmup_epochs
                ):
                    return float(
                        epoch + 1
                    ) / float(
                        warmup_epochs
                    )

                remaining = max(
                    1,
                    epochs - warmup_epochs,
                )

                progress = min(
                    1.0,
                    max(
                        0.0,
                        (
                            epoch
                            - warmup_epochs
                        )
                        / remaining,
                    ),
                )

                cosine = (
                    0.5
                    * (
                        1.0
                        + math.cos(
                            math.pi * progress
                        )
                    )
                )

                base_lr = float(
                    self._config_value(
                        "learning_rate",
                        3e-4,
                    )
                )

                floor = min_lr / base_lr

                return max(
                    floor,
                    cosine,
                )

            return torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lr_lambda,
            )

        raise BenchmarkError(
            f"Unsupported scheduler: {name!r}"
        )

    @staticmethod
    def _loader_length_hint(
        loader: Any,
    ) -> Optional[int]:
        try:
            length = len(loader)
        except (TypeError, AttributeError):
            return None

        if length <= 0:
            return None

        return int(length)

    # ------------------------------------------------------------------
    # Checkpoint restore
    # ------------------------------------------------------------------

    @staticmethod
    def _restore_model(
        model: nn.Module,
        checkpoint_path: str | Path,
    ) -> None:
        path = Path(checkpoint_path)

        if not path.exists():
            raise BenchmarkError(
                f"Best checkpoint does not exist: "
                f"{path}"
            )

        try:
            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(
                path,
                map_location="cpu",
            )
        except Exception as exc:
            raise BenchmarkError(
                f"Failed to load checkpoint "
                f"{path}: {exc}"
            ) from exc

        state_dict = None

        if isinstance(
            checkpoint,
            Mapping,
        ):
            for key in (
                "model_state",
                "model_state_dict",
                "model",
                "state_dict",
            ):
                candidate = checkpoint.get(
                    key
                )

                if isinstance(
                    candidate,
                    Mapping,
                ):
                    state_dict = candidate
                    break

        if state_dict is None:
            if isinstance(
                checkpoint,
                Mapping,
            ) and checkpoint:
                tensor_values = all(
                    isinstance(
                        value,
                        torch.Tensor,
                    )
                    for value in checkpoint.values()
                )

                if tensor_values:
                    state_dict = checkpoint

        if state_dict is None:
            raise BenchmarkError(
                "Checkpoint does not contain a "
                "recognizable model state_dict."
            )

        try:
            model.load_state_dict(
                state_dict,
                strict=True,
            )
        except Exception as exc:
            raise BenchmarkError(
                "Best checkpoint state_dict could "
                "not be loaded into the benchmark "
                f"model: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Configuration / device
    # ------------------------------------------------------------------

    def _config_value(
        self,
        name: str,
        default: Any,
    ) -> Any:
        config = self.training_config

        if hasattr(
            config,
            name,
        ):
            return getattr(
                config,
                name,
            )

        if isinstance(
            config,
            Mapping,
        ):
            return config.get(
                name,
                default,
            )

        return default

    def _config_to_dict(self) -> dict[str, Any]:
        config = self.training_config

        if hasattr(
            config,
            "to_dict",
        ):
            try:
                return _json_safe(
                    config.to_dict()
                )
            except Exception:
                pass

        if isinstance(
            config,
            Mapping,
        ):
            return _json_safe(
                dict(config)
            )

        return _json_safe(
            vars(config)
            if hasattr(
                config,
                "__dict__",
            )
            else {}
        )

    def _resolve_device(
        self,
    ) -> torch.device:
        config = self.training_config

        if hasattr(
            config,
            "resolve_device",
        ):
            try:
                return config.resolve_device()
            except Exception as exc:
                raise BenchmarkError(
                    f"Device resolution failed: {exc}"
                ) from exc

        requested = str(
            self._config_value(
                "device",
                "auto",
            )
        ).lower()

        if requested == "cpu":
            return torch.device("cpu")

        if requested == "cuda":
            if not torch.cuda.is_available():
                raise BenchmarkError(
                    "CUDA requested but unavailable."
                )
            return torch.device("cuda")

        if requested == "mps":
            mps = getattr(
                torch.backends,
                "mps",
                None,
            )

            if (
                mps is None
                or not mps.is_available()
            ):
                raise BenchmarkError(
                    "MPS requested but unavailable."
                )

            return torch.device("mps")

        if requested != "auto":
            raise BenchmarkError(
                f"Unsupported device: "
                f"{requested!r}"
            )

        if torch.cuda.is_available():
            return torch.device("cuda")

        mps = getattr(
            torch.backends,
            "mps",
            None,
        )

        if (
            mps is not None
            and mps.is_available()
        ):
            return torch.device("mps")

        return torch.device("cpu")

    def _effective_amp(
        self,
        device: torch.device,
    ) -> bool:
        config = self.training_config

        if hasattr(
            config,
            "effective_amp",
        ):
            try:
                return bool(
                    config.effective_amp()
                )
            except Exception:
                pass

        enabled = bool(
            self._config_value(
                "amp",
                False,
            )
        )

        if not enabled:
            return False

        return device.type in {
            "cuda",
            "mps",
        }

    def _amp_dtype(self) -> torch.dtype:
        config = self.training_config

        if hasattr(
            config,
            "resolved_amp_dtype",
        ):
            return config.resolved_amp_dtype()

        value = str(
            self._config_value(
                "amp_dtype",
                "float16",
            )
        ).lower()

        if value == "bfloat16":
            return torch.bfloat16

        return torch.float16

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @staticmethod
    def _model_metadata(
        model: nn.Module,
        model_id: str,
    ) -> dict[str, Any]:
        metadata = getattr(
            model,
            "metadata",
            None,
        )

        if callable(metadata):
            metadata = metadata()

        if isinstance(
            metadata,
            Mapping,
        ):
            return _json_safe(
                dict(metadata)
            )

        return {
            "model_id": model_id,
            "model_version": str(
                getattr(
                    model,
                    "model_version",
                    "unknown",
                )
            ),
            "architecture": str(
                getattr(
                    model,
                    "architecture",
                    model_id,
                )
            ),
            "parameter_count": sum(
                parameter.numel()
                for parameter in model.parameters()
            ),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_result(self) -> Path:
        path = (
            self.output_dir
            / f"{self.benchmark_id}.json"
        )

        payload = BenchmarkResult(
            benchmark_id=self.benchmark_id,
            dataset="LoveDA",
            seed=self.seed,
            class_ids=LOVE_DA_CLASS_IDS,
            class_names=LOVE_DA_CLASS_NAMES,
            models=tuple(self.results),
        ).to_dict()

        temp_path = path.with_name(
            f".{path.name}.tmp"
        )

        try:
            with temp_path.open(
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    payload,
                    handle,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                handle.flush()
                os.fsync(
                    handle.fileno()
                )

            os.replace(
                temp_path,
                path,
            )

        except Exception as exc:
            try:
                temp_path.unlink(
                    missing_ok=True
                )
            except OSError:
                pass

            raise BenchmarkError(
                f"Failed to persist benchmark "
                f"result: {exc}"
            ) from exc

        return path

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------

    def _apply_seed(self) -> None:
        if self.seed is None:
            return

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                self.seed
            )

        deterministic = bool(
            self._config_value(
                "deterministic",
                True,
            )
        )

        if deterministic:
            try:
                torch.use_deterministic_algorithms(
                    True,
                    warn_only=True,
                )
            except Exception:
                pass

            if hasattr(
                torch.backends,
                "cudnn",
            ):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False



class _BenchmarkMetricAdapter:
    """
    Adapter between SegmentationMetricAccumulator and TrainingEngine.

    SegmentationMetricAccumulator.compute() returns the rich
    SegmentationMetrics object. TrainingEngine deliberately requires
    compute() to return a scalar Mapping for epoch logging/checkpoint
    monitoring.
    """

    def __init__(self) -> None:
        self.accumulator = SegmentationMetricAccumulator(
            num_classes=len(LOVE_DA_CLASS_IDS),
            class_ids=LOVE_DA_CLASS_IDS,
            ignore_index=LOVE_DA_IGNORE_INDEX,
        )

    @property
    def num_classes(self) -> int:
        return len(LOVE_DA_CLASS_IDS)

    @property
    def class_ids(self) -> tuple[int, ...]:
        return LOVE_DA_CLASS_IDS

    @property
    def ignore_index(self) -> int:
        return LOVE_DA_IGNORE_INDEX

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        self.accumulator.update(
            logits,
            targets,
        )

    def compute(self) -> Mapping[str, float]:
        metrics = self.accumulator.compute()

        result: dict[str, float] = {}

        for name in (
            "pixel_accuracy",
            "mean_class_accuracy",
            "mean_iou",
            "frequency_weighted_iou",
            "macro_precision",
            "macro_recall",
            "macro_f1",
        ):
            value = getattr(
                metrics,
                name,
                None,
            )

            if value is None:
                continue

            if isinstance(
                value,
                torch.Tensor,
            ):
                if value.numel() != 1:
                    continue
                value = value.detach().cpu().item()

            if isinstance(
                value,
                (int, float),
            ):
                result[name] = float(value)

        # Dice is equivalent to F1 for the per-class
        # segmentation formulation used by this metrics layer.
        if "macro_dice" not in result:
            if "macro_f1" in result:
                result["macro_dice"] = result[
                    "macro_f1"
                ]

        return result

    @property
    def raw(self) -> Any:
        """Expose the rich metric object for final evaluation."""
        return self.accumulator.compute()

    @property
    def confusion_matrix(self) -> Any:
        return self.accumulator.confusion_matrix

    @property
    def valid_pixels(self) -> int:
        return int(
            self.accumulator.valid_pixels
        )
def _json_safe(value: Any) -> Any:
    """Recursively convert values to strict JSON-compatible objects."""

    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()

        return [
            _json_safe(item)
            for item in value.detach().cpu().tolist()
        ]
    
    if isinstance(
        value,
        Mapping,
    ):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            _json_safe(item)
            for item in value
        ]

    if isinstance(
        value,
        Path,
    ):
        return str(value)

    if isinstance(
        value,
        torch.device,
    ):
        return str(value)

    if isinstance(
        value,
        torch.dtype,
    ):
        return str(value)

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if isinstance(
        value,
        float,
    ):
        if math.isnan(value):
            return None

        if math.isinf(value):
            return None

        return value

    if isinstance(
        value,
        (str, int, bool),
    ) or value is None:
        return value

    return str(value)


def run_benchmark(
    *,
    benchmark_id: str,
    training_config: TrainingConfig,
    train_loader_factory: Callable[[], Any],
    val_loader_factory: Callable[[], Any],
    model_ids: Sequence[str],
    output_dir: str | Path,
    seed: Optional[int] = None,
) -> BenchmarkResult:
    """Convenience API for one complete benchmark."""

    runner = BenchmarkRunner(
        benchmark_id=benchmark_id,
        training_config=training_config,
        train_loader_factory=train_loader_factory,
        val_loader_factory=val_loader_factory,
        model_ids=model_ids,
        output_dir=output_dir,
        seed=seed,
    )

    return runner.run()