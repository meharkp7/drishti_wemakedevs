"""
Centralized DRISHTI configuration.
No magic constants should be scattered through the codebase — add new settings here.
"""

import os


def _env(key: str, default=None):
    return os.environ.get(key, default)


# --- Model ---
MODEL_NAME = _env("DRISHTI_MODEL_NAME", "placeholder-segmentation-model")
MODEL_VERSION = _env("DRISHTI_MODEL_VERSION", "0.1.0")

# --- Tiling ---
TILE_SIZE = int(_env("DRISHTI_TILE_SIZE", 512))
TILE_OVERLAP = int(_env("DRISHTI_TILE_OVERLAP", 32))

# --- Confidence thresholds ---
CONFIDENCE_THRESHOLDS = {
    "high": float(_env("DRISHTI_CONF_HIGH", 0.85)),
    "medium": float(_env("DRISHTI_CONF_MEDIUM", 0.60)),
    # below "medium" is treated as low confidence
}

# --- AOI limits ---
MAX_AOI_AREA_M2 = float(_env("DRISHTI_MAX_AOI_AREA_M2", 5_000_000))  # 5 km^2 default cap
SUPPORTED_CRS = ["EPSG:4326", "EPSG:3857"]

# --- Storage ---
STORAGE_BUCKET = _env("DRISHTI_STORAGE_BUCKET", "drishti-local-dev")

# --- Datasets ---
# Local staging directory for a downloaded LoveDA copy. See
# datasets/loveda/download.py and datasets/loveda/manifest.py.
LOVEDA_ROOT = _env("DRISHTI_LOVEDA_ROOT", "data/loveda")

# --- v2 additions (Inference Router / Crowdsource layer) ---
INFERENCE_BACKENDS_AVAILABLE = _env(
    "DRISHTI_INFERENCE_BACKENDS", "cpu"
).split(",")  # e.g. "gpu,cpu" or "gpu,npu,cpu,cloud"

CROWDSOURCE_INTAKE_ENABLED = _env("DRISHTI_CROWDSOURCE_ENABLED", "true").lower() == "true"