from pathlib import Path
from uuid import uuid4

import numpy as np
import rasterio
from rasterio.transform import from_origin

from backend.core.job_state import AnalysisJob, AnalysisJobRequest
from backend.workers.analysis_pipeline import LocalAnalysisPipeline


def _write_rgb_raster(path: Path):
    data = np.zeros((3, 96, 128), dtype=np.uint8)
    data[0, 10:45, 10:50] = 180
    data[1, 50:80, 20:110] = 120
    data[2, 20:70, 60:115] = 200
    profile = {
        "driver": "GTiff",
        "height": 96,
        "width": 128,
        "count": 3,
        "dtype": "uint8",
        "crs": "EPSG:4326",
        "transform": from_origin(77.0, 28.0, 0.0001, 0.0001),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


def test_local_analysis_pipeline_runs(tmp_path):
    raster = tmp_path / "20250101.tif"
    _write_rgb_raster(raster)

    request = AnalysisJobRequest(
        aoi={
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [77.0001, 27.9905],
                    [77.0115, 27.9905],
                    [77.0115, 27.9999],
                    [77.0001, 27.9999],
                    [77.0001, 27.9905],
                ]],
            },
            "crs": "EPSG:4326",
        },
        imagery_source={"path": str(raster)},
    )
    job = AnalysisJob.create(request=request, job_id=uuid4())

    result = LocalAnalysisPipeline(tile_size=64, overlap=8).run(job, tmp_path / "output")

    assert result.tile_count > 0
    assert Path(result.clipped_raster).exists()
    assert Path(result.mask_path).exists()
    assert Path(result.confidence_path).exists()
    assert Path(result.vector_path).exists()
    assert result.statistics["vector_feature_count"] >= 0
