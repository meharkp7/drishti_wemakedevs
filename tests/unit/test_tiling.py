"""
Unit tests for DRISHTI geospatial raster tiling.
"""

from pathlib import Path

import numpy as np
import rasterio
import pytest
from rasterio.transform import from_origin

from geospatial.tiling import tile_raster
from geospatial.tiling.tiler import TilingError, _tile_starts


def _create_test_raster(path: Path) -> None:
    """Create a deterministic georeferenced GeoTIFF."""
    width = 1000
    height = 700

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
        crs="EPSG:32643",
        transform=transform,
    ) as dst:
        dst.write(data, 1)


def test_tile_starts_are_deterministic():
    assert _tile_starts(1000, 256, 32) == [
        0,
        224,
        448,
        672,
        744,
    ]

    assert _tile_starts(700, 256, 32) == [
        0,
        224,
        444,
    ]

    assert _tile_starts(200, 256, 32) == [0]


def test_tiles_are_created_with_geospatial_metadata(tmp_path):
    raster_path = tmp_path / "source.tif"
    output_dir = tmp_path / "tiles"

    _create_test_raster(raster_path)

    artifacts = tile_raster(
        raster_path,
        output_dir,
        tile_size=256,
        overlap=32,
    )

    assert len(artifacts) == 15

    for artifact in artifacts:
        tile_path = Path(artifact.path)

        assert tile_path.exists()
        assert artifact.crs == "EPSG:32643"
        assert 0 < artifact.width <= 256
        assert 0 < artifact.height <= 256

        with rasterio.open(tile_path) as tile:
            assert tile.crs.to_string() == "EPSG:32643"
            assert tile.width == artifact.width
            assert tile.height == artifact.height
            assert tuple(tile.transform) == artifact.transform


def test_tile_order_is_row_major_and_deterministic(tmp_path):
    raster_path = tmp_path / "source.tif"

    _create_test_raster(raster_path)

    first_dir = tmp_path / "tiles_first"
    second_dir = tmp_path / "tiles_second"

    first = tile_raster(
        raster_path,
        first_dir,
        tile_size=256,
        overlap=32,
    )

    second = tile_raster(
        raster_path,
        second_dir,
        tile_size=256,
        overlap=32,
    )

    first_ids = [artifact.tile_id for artifact in first]
    second_ids = [artifact.tile_id for artifact in second]

    assert first_ids == second_ids

    assert first_ids[:5] == [
        "tile_r0000_c0000",
        "tile_r0000_c0001",
        "tile_r0000_c0002",
        "tile_r0000_c0003",
        "tile_r0000_c0004",
    ]

    assert first_ids[-1] == "tile_r0002_c0004"

    first_bounds = [artifact.bounds for artifact in first]
    second_bounds = [artifact.bounds for artifact in second]

    assert first_bounds == second_bounds


def test_edge_tiles_reach_raster_boundary(tmp_path):
    raster_path = tmp_path / "source.tif"
    output_dir = tmp_path / "tiles"

    _create_test_raster(raster_path)

    artifacts = tile_raster(
        raster_path,
        output_dir,
        tile_size=256,
        overlap=32,
    )

    with rasterio.open(raster_path) as source:
        source_bounds = source.bounds

    last_column = max(artifact.col for artifact in artifacts)
    last_row = max(artifact.row for artifact in artifacts)

    bottom_right = next(
        artifact
        for artifact in artifacts
        if artifact.row == last_row
        and artifact.col == last_column
    )

    assert bottom_right.bounds[2] == pytest.approx(
        source_bounds.right
    )

    assert bottom_right.bounds[1] == pytest.approx(
        source_bounds.bottom
    )


def test_tile_pixels_match_source_window(tmp_path):
    raster_path = tmp_path / "source.tif"
    output_dir = tmp_path / "tiles"

    _create_test_raster(raster_path)

    artifacts = tile_raster(
        raster_path,
        output_dir,
        tile_size=256,
        overlap=32,
    )

    first = artifacts[0]

    with rasterio.open(raster_path) as source:
        expected = source.read(
            window=rasterio.windows.Window(
                col_off=0,
                row_off=0,
                width=256,
                height=256,
            )
        )

    with rasterio.open(first.path) as tile:
        actual = tile.read()

    np.testing.assert_array_equal(actual, expected)


def test_invalid_overlap_is_rejected(tmp_path):
    raster_path = tmp_path / "source.tif"
    output_dir = tmp_path / "tiles"

    _create_test_raster(raster_path)

    with pytest.raises(TilingError):
        tile_raster(
            raster_path,
            output_dir,
            tile_size=256,
            overlap=256,
        )


def test_invalid_tile_size_is_rejected(tmp_path):
    raster_path = tmp_path / "source.tif"
    output_dir = tmp_path / "tiles"

    _create_test_raster(raster_path)

    with pytest.raises(TilingError):
        tile_raster(
            raster_path,
            output_dir,
            tile_size=0,
            overlap=32,
        )