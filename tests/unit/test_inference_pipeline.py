"""
tests/unit/test_inference_pipeline.py

Phase 2 gate test, per the implementation plan:
    "Do not continue until image -> model -> segmentation -> overlay works reliably."

This test does not assert on prediction *correctness* (the model may be running as an
untrained fallback in offline/sandboxed environments — see inference/models/
segmentation_model.py). It asserts the pipeline is mechanically sound: every artifact is
produced, with the right shapes and a valid confidence range.
"""

import json
import os

import numpy as np
import pytest
from PIL import Image

from inference.infer import segment
from inference.postprocessing.visualize import generate_visualizations

FIXTURE_IMAGE = os.path.join(os.path.dirname(__file__), "..", "fixtures", "demo_aoi.png")


@pytest.fixture()
def output_dir(tmp_path):
    return str(tmp_path / "phase2_output")


def test_segment_produces_full_prediction_artifact(output_dir):
    result = segment(FIXTURE_IMAGE, output_dir)

    # Contract fields the rest of the system depends on
    for key in ("classes", "confidence", "mask_path", "model_id", "model_version",
                "inference_backend", "inference_time_ms"):
        assert key in result

    assert os.path.exists(result["mask_path"])
    assert os.path.exists(result["confidence_map_path"])
    assert os.path.exists(os.path.join(output_dir, "result.json"))

    assert 0.0 <= result["confidence"]["mean"] <= 1.0
    assert 0.0 <= result["confidence"]["min"] <= result["confidence"]["max"] <= 1.0
    assert len(result["classes"]) >= 1


def test_mask_matches_original_dimensions(output_dir):
    result = segment(FIXTURE_IMAGE, output_dir)
    original = Image.open(FIXTURE_IMAGE)
    mask = Image.open(result["mask_path"])
    assert mask.size == original.size


def test_visualizations_are_generated(output_dir):
    result = segment(FIXTURE_IMAGE, output_dir)
    paths = generate_visualizations(FIXTURE_IMAGE, result, output_dir)

    for key in ("original", "segmentation", "overlay", "confidence_heatmap"):
        assert key in paths
        assert os.path.exists(paths[key])

    original = Image.open(paths["original"])
    overlay = Image.open(paths["overlay"])
    assert overlay.size == original.size


def test_result_json_is_serializable(output_dir):
    result = segment(FIXTURE_IMAGE, output_dir)
    with open(os.path.join(output_dir, "result.json")) as f:
        loaded = json.load(f)
    assert loaded["model_id"] == result["model_id"]