# Milestone B — LoveDA Experiments and Benchmark Protocol

This milestone converts DRISHTI's existing training/model code into a reproducible empirical benchmark. It does not change the production geospatial taxonomy: LoveDA remains an ML training/evaluation taxonomy only.

## Fixed protocol

All M1–M7 candidates use the same dataset split, seed, augmentation, loader policy, optimizer family, learning-rate schedule, loss, checkpoint-selection metric, and early-stopping policy unless an experiment explicitly records a justified exception.

- Dataset: LoveDA
- Classes: 7 semantic classes; mask value `0` is ignore/no-data
- Train/validation: the staged LoveDA manifest's `train` and `val` splits
- Seed: `26137`
- Sampling: class-balanced, with domain balancing enabled in the sampler
- Batch size: `4` by default; change only when hardware requires it and record the change
- Optimizer: AdamW
- Initial learning rate: `3e-4`
- Weight decay: `1e-4`
- Scheduler: cosine, 3 warm-up epochs
- Loss: CE + Dice
- Class-weighted loss: disabled for the first locked protocol; sampling handles the initial imbalance intervention
- Maximum epochs: `50`
- Early stopping: patience `10`
- Model selection: validation mean IoU (`mIoU`), maximize
- Augmentation: paired horizontal/vertical flips, right-angle rotation, bounded image-only color jitter
- Validation transforms: deterministic
- Input normalization: ImageNet mean/std
- Checkpoints: best + latest experiment checkpoints
- Reproducibility: deterministic seed controls

## Candidate mapping

| ID | Registry model | Architecture |
|---|---|---|
| M1 | `deeplabv3plus_resnet50` | DeepLabV3+ + ResNet-50 |
| M2 | `deeplabv3plus_resnet101` | DeepLabV3+ + ResNet-101 |
| M3 | `unetplusplus_efficientnet_b4` | UNet++ + EfficientNet-B4 |
| M4 | `fpn_resnet50` | FPN + ResNet-50 |
| M5 | `cnn_mamba` | CNN + Mamba/state-space mixing |
| M6 | `foundation_geospatial` | Foundation-style encoder + geospatial decoder |
| M7 | `dual_path_mamba` | Dual-path CNN/Mamba + frequency branch |

## Required evidence per model

The benchmark artifact records:

1. Best validation mIoU and epoch
2. Validation loss/history
3. Overall pixel accuracy, mIoU, frequency-weighted IoU, macro precision/recall/F1/Dice
4. Per-class IoU/precision/recall/F1
5. Urban and rural validation metrics
6. Confusion matrix
7. Parameter count
8. Representative inference latency: mean, median and P95
9. Images/second for the benchmark batch
10. Peak accelerator memory when available
11. Exact checkpoint path and model metadata
12. Exact training configuration and seed

## Running M1

```powershell
python scripts/run_loveda_benchmark.py `
  --loveda-root "C:\Users\siddh\Downloads\LoveDA" `
  --config configs/experiments/loveda_m1_baseline.json
```

Before training, use `--dry-run` to validate the dataset, loader, configuration and model registry without starting a long run:

```powershell
python scripts/run_loveda_benchmark.py `
  --loveda-root "C:\Users\siddh\Downloads\LoveDA" `
  --config configs/experiments/loveda_m1_baseline.json `
  --dry-run
```

## Running the complete M1–M7 benchmark

```powershell
python scripts/run_loveda_benchmark.py `
  --loveda-root "C:\Users\siddh\Downloads\LoveDA" `
  --config configs/experiments/loveda_full_benchmark_v1.json `
  --models all
```

This is intentionally a long-running experiment. Run M1 first and inspect the resulting artifact before launching all seven models.

## Dependency requirement

M1–M4 use `segmentation_models_pytorch`. Install the optional training dependencies from the project root:

```powershell
python -m pip install -e ".[training]"
```

The script does not download LoveDA automatically and does not silently substitute a random model. The dataset path and optional pretrained encoder weights must be available in the environment.

## What this milestone does not claim yet

- It does not claim a best model before measured results exist.
- It does not turn LoveDA labels into DRISHTI's final production taxonomy.
- It does not replace the current production inference baseline yet.
- It does not imply that on-device inference, remote imagery, WMS, human review, frontend, or AI assistant features are implemented.
