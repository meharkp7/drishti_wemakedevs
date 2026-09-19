"""Run a reproducible LoveDA training/benchmark experiment.

Examples
--------
python scripts/run_loveda_benchmark.py \
  --loveda-root "C:\\Users\\siddh\\Downloads\\LoveDA" \
  --config configs/experiments/loveda_m1_baseline.json

To benchmark all registered candidates with the same protocol:
python scripts/run_loveda_benchmark.py \
  --loveda-root "C:\\Users\\siddh\\Downloads\\LoveDA" \
  --config configs/experiments/loveda_m1_baseline.json \
  --models all

The script intentionally does not download LoveDA or silently change the
configured experiment. Dataset and model weights must be available to the
local environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import sys

# Allow direct execution as `python scripts/run_loveda_benchmark.py` from the
# repository root without requiring an editable install first.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.loveda.manifest import scan_dataset
from training.benchmark import BenchmarkRunner
from training.config import TrainingConfig
from training.data.dataloader import LoveDALoaderConfig, build_loveda_loaders
from training.data.sampler import SamplingStrategy
from training.models.registry import DEFAULT_MODEL_REGISTRY


DEFAULT_MODELS = (
    "deeplabv3plus_resnet50",
    "deeplabv3plus_resnet101",
    "unetplusplus_efficientnet_b4",
    "fpn_resnet50",
    "cnn_mamba",
    "foundation_geospatial",
    "dual_path_mamba",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration root must be an object: {path}")
    return payload


def _build_training_config(payload: dict[str, Any]) -> TrainingConfig:
    values = dict(payload)
    # These are benchmark-level paths and are not needed by TrainingConfig's
    # runtime execution, but keeping them in JSON is useful for provenance.
    values["checkpoint_dir"] = Path(values.get("checkpoint_dir", "artifacts/checkpoints"))
    values["experiment_dir"] = Path(values.get("experiment_dir", "artifacts/experiments"))
    return TrainingConfig(**values)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loveda-root", required=True, help="Staged LoveDA root directory.")
    parser.add_argument("--config", required=True, type=Path, help="JSON benchmark configuration.")
    parser.add_argument(
        "--models",
        default="configured",
        help="configured, all, or a comma-separated list of registry model names.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional override for benchmark output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the dataset/config/model registry without training.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = Path(args.loveda_root).expanduser().resolve()
    config_payload = _load_json(args.config)

    if not root.exists() or not root.is_dir():
        raise SystemExit(f"LoveDA root does not exist or is not a directory: {root}")

    manifest = scan_dataset(root)
    training_config = _build_training_config(config_payload.get("training", {}))

    data_payload = dict(config_payload.get("data", {}))
    strategy = SamplingStrategy(data_payload.pop("sampling_strategy", "class_balanced"))
    loader_config = LoveDALoaderConfig(
        sampling_strategy=strategy,
        seed=int(config_payload.get("seed", training_config.seed)),
        **data_payload,
    )

    configured_models = tuple(config_payload.get("model_ids", ()))
    requested = args.models.strip().lower()
    if requested == "configured":
        model_ids = configured_models
    elif requested == "all":
        model_ids = DEFAULT_MODELS
    else:
        model_ids = tuple(item.strip() for item in args.models.split(",") if item.strip())

    if not model_ids:
        raise SystemExit("No model IDs configured.")

    unknown = [name for name in model_ids if not DEFAULT_MODEL_REGISTRY.contains(name)]
    if unknown:
        raise SystemExit(
            "Unknown model(s): " + ", ".join(unknown) +
            f". Available: {', '.join(DEFAULT_MODEL_REGISTRY.names())}"
        )

    output_payload = config_payload.get("benchmark", {})
    output_dir = args.output_dir or Path(output_payload.get("output_dir", "artifacts/benchmarks/loveda"))

    print(f"LoveDA root: {root}")
    print(f"Samples: {len(manifest.samples)}")
    print(f"Models: {', '.join(model_ids)}")
    print(f"Device: {training_config.resolve_device()}")
    print(f"Output: {output_dir}")

    if args.dry_run:
        # Building loaders validates split/domain/sample contracts without
        # starting a potentially hours-long training job.
        loaders = build_loveda_loaders(manifest, config=loader_config)
        print(f"Train batches: {len(loaders.train)}")
        print(f"Validation batches: {len(loaders.val)}")
        print("DRY RUN OK")
        return 0

    def train_loader_factory():
        return build_loveda_loaders(manifest, config=loader_config).train

    def val_loader_factory():
        return build_loveda_loaders(manifest, config=loader_config).val

    runner = BenchmarkRunner(
        benchmark_id=str(config_payload.get("benchmark_id", "loveda_benchmark")),
        training_config=training_config,
        train_loader_factory=train_loader_factory,
        val_loader_factory=val_loader_factory,
        model_ids=model_ids,
        output_dir=output_dir,
        seed=int(config_payload.get("seed", training_config.seed)),
        latency_warmup_batches=int(output_payload.get("latency_warmup_batches", 2)),
        latency_measurement_batches=int(output_payload.get("latency_measurement_batches", 10)),
    )

    result = runner.run(
        on_model_complete=lambda model: print(
            f"Completed {model.model_id}: "
            f"best mIoU={model.best_metric} at epoch={model.best_epoch}"
        )
    )

    result_path = output_dir / f"{result.benchmark_id}.json"
    print(f"Benchmark complete: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
