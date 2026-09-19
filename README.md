# DRISHTI

AI-assisted geospatial intelligence pipeline: imagery -> semantic segmentation ->
confidence-driven human verification -> temporal change detection -> GIS-ready export.

## Development setup

DRISHTI currently supports Python 3.10--3.13. Create and activate a virtual
environment, then install the CPU development dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install optional candidate-model and LoveDA download support when needed:

```powershell
python -m pip install -e ".[dev,training]"
```

Run the complete suite from the repository root:

```powershell
python -m pytest -q
```

The tests require `rasterio`; it is a core dependency because AOI clipping,
tiling, vectorization, and change detection operate on georeferenced rasters.
The legacy `requirements.txt` is retained for reference only; `pyproject.toml`
is the supported install contract.

## Current capability boundary

The repository contains a tested LoveDA training foundation and local-raster
geospatial primitives. The runtime inference module is still a Phase-2 VOC
baseline and must not be used as a land-cover model until a checkpoint-backed
LoveDA inference adapter is introduced.

The baseline is offline-first. It uses a randomly initialized plumbing model
by default; setting `DRISHTI_ALLOW_BASELINE_MODEL_DOWNLOAD=true` explicitly
permits torchvision to download VOC weights. Neither mode is a DRISHTI
production land-cover model.

## Repository layout

frontend/       React + map UI, WebGL overlay rendering, review UI, dashboard
backend/        API, services, background workers, schemas, core utilities
adapters/       Interaction Adapter - normalizes map/text/voice/sms/email into one Command
inference/      Model registry, preprocessing/postprocessing, device-aware Inference Router
geospatial/     Imagery abstraction, tiling, vectorization, change detection, validation
crowdsource/    Citizen/field-worker submission intake, dedup, geotag validation
agent/          LLM natural-language command layer (5-tool allowlist)
notifications/  Templated email/SMS dispatch on job-state transitions
storage/        Storage clients/adapters (object store + metadata DB)
tests/          unit / integration / e2e
configs/        Centralized configuration (no magic constants in code)
scripts/        One-off/dev scripts

## LoveDA experiment milestone

The repository now includes a locked, reproducible LoveDA benchmark protocol and runnable experiment entrypoint:

- `configs/experiments/loveda_m1_baseline.json` — first M1 baseline
- `configs/experiments/loveda_full_benchmark_v1.json` — M1–M7 protocol
- `scripts/run_loveda_benchmark.py` — dataset validation, dry-run, training and benchmark execution
- `docs/MILESTONE_B_LOVEDA.md` — protocol, model mapping and evidence requirements

Start with a dry run:

```powershell
python scripts/run_loveda_benchmark.py --loveda-root "C:\Users\siddh\Downloads\LoveDA" --config configs/experiments/loveda_m1_baseline.json --dry-run
```

Then run M1:

```powershell
python scripts/run_loveda_benchmark.py --loveda-root "C:\Users\siddh\Downloads\LoveDA" --config configs/experiments/loveda_m1_baseline.json
```

The benchmark records validation metrics, per-class results, urban/rural breakdowns, confusion matrix, parameter count, representative inference latency and accelerator memory where available. No model is declared the winner until measured results exist.
