"""
End-to-end unit coverage for the DRISHTI Phase-3 geospatial core.

Pipeline under test:

AOI
  -> raster coverage
  -> AOI clipping
  -> deterministic tiling
  -> geospatial metadata preservation
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from geospatial.aoi import AOI
from geospatial.imagery.raster import (
    check_aoi_coverage,
    clip_raster_to_aoi,
    read_raster_metadata,
)
from geospatial.tiling import tile_raster


def _create_source_raster(path: Path) -> None:
    """Create a deterministic georeferenced GeoTIFF."""
    width = 1000
    height = 1000

    transform = from_origin(
        500000.0,
        3100000.0,
        10.0,
        10.0,
    )

    data = np.arange(
        width * height,
        dtype=np.uint16,
    ).reshape(height, width)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype=data.dtype,
        crs="EPSG:3857",
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(data, 1)


def test_aoi_to_clipped_raster_to_tiles(tmp_path):
    source_path = tmp_path / "source.tif"
    clipped_path = tmp_path / "clipped.tif"
    tile_dir = tmp_path / "tiles"

    _create_source_raster(source_path)

    # AOI expressed in the same CRS as the source raster.
    aoi = AOI.from_coordinates(
        [
            [500500.0, 3097500.0],
            [502500.0, 3097500.0],
            [502500.0, 3099500.0],
            [500500.0, 3099500.0],
            [500500.0, 3097500.0],
        ],
        crs="EPSG:3857",
    )

    # 1. Confirm source coverage.
    assert check_aoi_coverage(source_path, aoi) is True

    # 2. Read source metadata.
    source_metadata = read_raster_metadata(source_path)

    assert source_metadata.crs == "EPSG:3857"
    assert source_metadata.width == 1000
    assert source_metadata.height == 1000
    assert source_metadata.band_count == 1
    assert source_metadata.pixel_size_x == 10.0
    assert source_metadata.pixel_size_y == 10.0

    # 3. Clip source raster to AOI.
    clipped_metadata = clip_raster_to_aoi(
        source_path,
        aoi,
        clipped_path,
    )

    assert clipped_path.exists()
    assert clipped_metadata.crs == "EPSG:3857"
    assert clipped_metadata.width > 0
    assert clipped_metadata.height > 0
    assert clipped_metadata.width <= source_metadata.width
    assert clipped_metadata.height <= source_metadata.height

    # Clipping must preserve source resolution.
    assert clipped_metadata.pixel_size_x == 10.0
    assert clipped_metadata.pixel_size_y == 10.0

    # 4. Tile the clipped raster.
    artifacts = tile_raster(
        clipped_path,
        tile_dir,
        tile_size=256,
        overlap=32,
    )

    assert artifacts
    assert all(
        Path(artifact.path).exists()
        for artifact in artifacts
    )

    # 5. Every tile remains georeferenced.
    for artifact in artifacts:
        with rasterio.open(artifact.path) as tile:
            assert tile.crs.to_string() == "EPSG:3857"
            assert tile.width == artifact.width
            assert tile.height == artifact.height
            assert tuple(tile.transform) == artifact.transform

    # 6. Tile ordering is deterministic.
    assert [
        artifact.tile_id for artifact in artifacts
    ] == sorted(
        artifact.tile_id for artifact in artifacts
    )


def test_aoi_outside_raster_is_rejected(tmp_path):
    source_path = tmp_path / "source.tif"

    _create_source_raster(source_path)

    # Completely outside the source raster.
    aoi = AOI.from_coordinates(
        [
            [600000.0, 3000000.0],
            [601000.0, 3000000.0],
            [601000.0, 3001000.0],
            [600000.0, 3001000.0],
            [600000.0, 3000000.0],
        ],
        crs="EPSG:3857",
    )

    assert check_aoi_coverage(
        source_path,
        aoi,
    ) is False