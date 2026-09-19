import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from geospatial.change import detect_semantic_change
from geospatial.vectorization import mask_to_geojson


def _write_mask(path: Path, values: np.ndarray):
    profile = {
        "driver": "GTiff",
        "height": values.shape[0],
        "width": values.shape[1],
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:4326",
        "transform": from_origin(77.0, 28.0, 0.001, 0.001),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(values, 1)


def test_mask_to_geojson(tmp_path):
    mask = tmp_path / "mask.tif"
    _write_mask(mask, np.array([[1, 1, 0], [1, 2, 2], [0, 2, 2]], dtype=np.uint8))
    result = mask_to_geojson(mask, tmp_path / "mask.geojson")
    assert result.feature_count == 2
    payload = json.loads(Path(result.output_path).read_text())
    assert len(payload["features"]) == 2
    assert set(result.class_area_m2) == {1, 2}


def test_change_detection(tmp_path):
    before = tmp_path / "before.tif"
    after = tmp_path / "after.tif"
    _write_mask(before, np.array([[1, 1, 0], [1, 2, 2], [0, 2, 2]], dtype=np.uint8))
    _write_mask(after, np.array([[1, 2, 0], [1, 2, 1], [0, 2, 2]], dtype=np.uint8))
    result = detect_semantic_change(before, after, tmp_path / "changes")
    assert result.changed_pixels == 2
    assert result.transitions == {"1->2": 1, "2->1": 1}
    assert 0 < result.changed_fraction < 1
    assert Path(result.output_mask).exists()
    assert Path(result.output_geojson).exists()
