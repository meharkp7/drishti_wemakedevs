"""
training/config.py

Canonical, validated configuration for DRISHTI's supervised segmentation
experiments.

This module intentionally contains experiment/training configuration only.
Dataset taxonomy remains owned by datasets.loveda.taxonomy.

Design goals:
- reproducibility
- explicit configuration
- no magic training constants
- safe defaults
- deterministic experiment identity
- CPU/GPU portability
- future compatibility with multiple model families
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional
import os
import random

import numpy as np
import torch


DeviceType = Literal["auto", "cpu", "cuda", "mps"]


class TrainingConfigError(ValueError):
    """Raised when training configuration is invalid."""


@dataclass(frozen=True)
class TrainingConfig:
    """
    Immutable configuration for one segmentation experiment.

    The configuration is deliberately model-agnostic. Architecture-specific
    parameters belong to the model configuration, not this global training
    contract.
    """

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------

    seed: int = 26137
    deterministic: bool = True

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------

    device: DeviceType = "auto"
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    epochs: int = 50
    batch_size: int = 8

    learning_rate: float = 3e-4
    weight_decay: float = 1e-4

    gradient_accumulation_steps: int = 1
    gradient_clip_norm: Optional[float] = 1.0

    # ------------------------------------------------------------------
    # Mixed precision
    # ------------------------------------------------------------------

    amp: bool = True
    amp_dtype: Literal["float16", "bfloat16"] = "float16"

    # ------------------------------------------------------------------
    # Optimization / scheduling
    # ------------------------------------------------------------------

    optimizer: Literal["adamw", "sgd"] = "adamw"
    scheduler: Literal[
        "none",
        "cosine",
        "cosine_warmup",
        "one_cycle",
        "plateau",
    ] = "cosine"

    warmup_epochs: int = 3
    min_learning_rate: float = 1e-6

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    loss: Literal[
        "cross_entropy",
        "weighted_cross_entropy",
        "dice",
        "ce_dice",
        "focal",
        "focal_dice",
        "tversky",
        "tversky_focal",
    ] = "ce_dice"

    class_weighting: Literal[
        "none",
        "inverse_frequency",
        "median_frequency",
        "effective_number",
    ] = "none"

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    eval_every_epoch: int = 1

    # A class may be absent from a validation image/batch. Metrics handle
    # this explicitly; this threshold controls only experiment-level
    # checkpoint selection.
    monitor_metric: Literal[
        "mean_iou",
        "macro_f1",
        "macro_dice",
        "pixel_accuracy",
    ] = "mean_iou"

    monitor_mode: Literal["max", "min"] = "max"

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    checkpoint_dir: Path = Path("artifacts/checkpoints")
    experiment_dir: Path = Path("artifacts/experiments")

    save_best_only: bool = False
    save_last: bool = True

    early_stopping_patience: Optional[int] = 10

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    image_size: Optional[int] = None
    crop_size: Optional[int] = None

    # ------------------------------------------------------------------
    # Optional metadata
    # ------------------------------------------------------------------

    experiment_name: str = "loveda_segmentation"

    notes: str = ""

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.seed < 0:
            raise TrainingConfigError("seed must be >= 0.")

        if self.epochs <= 0:
            raise TrainingConfigError("epochs must be > 0.")

        if self.batch_size <= 0:
            raise TrainingConfigError("batch_size must be > 0.")

        if self.num_workers < 0:
            raise TrainingConfigError("num_workers must be >= 0.")

        if self.num_workers == 0 and self.persistent_workers:
            raise TrainingConfigError(
                "persistent_workers=True requires num_workers > 0."
            )

        if self.learning_rate <= 0:
            raise TrainingConfigError("learning_rate must be > 0.")

        if self.weight_decay < 0:
            raise TrainingConfigError("weight_decay must be >= 0.")

        if self.gradient_accumulation_steps <= 0:
            raise TrainingConfigError(
                "gradient_accumulation_steps must be > 0."
            )

        if self.gradient_clip_norm is not None:
            if self.gradient_clip_norm <= 0:
                raise TrainingConfigError(
                    "gradient_clip_norm must be > 0 or None."
                )

        if self.warmup_epochs < 0:
            raise TrainingConfigError("warmup_epochs must be >= 0.")

        if self.warmup_epochs > self.epochs:
            raise TrainingConfigError(
                "warmup_epochs cannot exceed epochs."
            )

        if self.min_learning_rate <= 0:
            raise TrainingConfigError("min_learning_rate must be > 0.")

        if self.min_learning_rate > self.learning_rate:
            raise TrainingConfigError(
                "min_learning_rate cannot exceed learning_rate."
            )

        if self.eval_every_epoch <= 0:
            raise TrainingConfigError("eval_every_epoch must be > 0.")

        if self.image_size is not None and self.image_size <= 0:
            raise TrainingConfigError("image_size must be > 0 or None.")

        if self.crop_size is not None and self.crop_size <= 0:
            raise TrainingConfigError("crop_size must be > 0 or None.")

        if self.early_stopping_patience is not None:
            if self.early_stopping_patience < 0:
                raise TrainingConfigError(
                    "early_stopping_patience must be >= 0 or None."
                )

        if not self.experiment_name.strip():
            raise TrainingConfigError(
                "experiment_name must not be empty."
            )

        if self.amp and self.device == "cpu":
            # AMP is technically available on some CPU configurations,
            # but float16 CPU training is not a portable production default.
            if self.amp_dtype == "float16":
                raise TrainingConfigError(
                    "float16 AMP on an explicitly selected CPU is unsupported "
                    "by this portable training contract. Use bfloat16 or "
                    "disable AMP."
                )

    def resolve_device(self) -> torch.device:
        """
        Resolve the configured device deterministically.

        ``auto`` prefers CUDA, then Apple MPS, then CPU.
        Explicit unavailable accelerators fail loudly rather than silently
        changing experiment semantics.
        """
        if self.device == "cpu":
            return torch.device("cpu")

        if self.device == "cuda":
            if not torch.cuda.is_available():
                raise TrainingConfigError(
                    "device='cuda' requested but CUDA is unavailable."
                )
            return torch.device("cuda")

        if self.device == "mps":
            mps = getattr(torch.backends, "mps", None)
            if mps is None or not mps.is_available():
                raise TrainingConfigError(
                    "device='mps' requested but MPS is unavailable."
                )
            return torch.device("mps")

        if torch.cuda.is_available():
            return torch.device("cuda")

        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    def resolved_amp_dtype(self) -> torch.dtype:
        """Return the torch dtype corresponding to the AMP configuration."""
        return (
            torch.float16
            if self.amp_dtype == "float16"
            else torch.bfloat16
        )

    def effective_amp(self) -> bool:
        """
        Return whether AMP should actually be enabled on the resolved device.

        This is deliberately conservative: MPS support differs from CUDA,
        while CPU float16 is excluded by validation.
        """
        if not self.amp:
            return False

        device = self.resolve_device()

        if device.type == "cuda":
            return True

        if device.type == "mps":
            # Keep the flag available for future tested MPS support, but don't
            # silently claim CUDA-equivalent AMP semantics.
            return self.amp_dtype == "float16"

        return self.amp_dtype == "bfloat16"

    def apply_seed(self) -> None:
        """
        Apply reproducibility controls to Python, NumPy and PyTorch.

        Deterministic execution may reduce performance. It is therefore
        explicitly controlled by ``deterministic``.
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        if self.deterministic:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

            try:
                torch.use_deterministic_algorithms(True)
            except RuntimeError:
                # Some operators/backends do not support deterministic
                # execution. Configuration should not fail merely because
                # the runtime cannot provide a stronger guarantee.
                pass

            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

    def ensure_output_dirs(self) -> None:
        """Create experiment/checkpoint directories if necessary."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        """Return a JSON-serializable configuration representation."""
        return {
            "seed": self.seed,
            "deterministic": self.deterministic,
            "device": self.device,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "gradient_clip_norm": self.gradient_clip_norm,
            "amp": self.amp,
            "amp_dtype": self.amp_dtype,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "warmup_epochs": self.warmup_epochs,
            "min_learning_rate": self.min_learning_rate,
            "loss": self.loss,
            "class_weighting": self.class_weighting,
            "eval_every_epoch": self.eval_every_epoch,
            "monitor_metric": self.monitor_metric,
            "monitor_mode": self.monitor_mode,
            "checkpoint_dir": str(self.checkpoint_dir),
            "experiment_dir": str(self.experiment_dir),
            "save_best_only": self.save_best_only,
            "save_last": self.save_last,
            "early_stopping_patience": self.early_stopping_patience,
            "image_size": self.image_size,
            "crop_size": self.crop_size,
            "experiment_name": self.experiment_name,
            "notes": self.notes,
        }