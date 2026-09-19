from datetime import datetime

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

from geospatial.aoi import AOI
from geospatial.imagery.provider import (
    ImageryProviderError,
    LocalImageryProvider,
    RasterWindowArtifact,
)


def _create_source_raster(path):
    width = 512
    height = 512

    data = np.arange(
        width * height,
        dtype=np.uint16,
    ).reshape(height, width)

    transform = from_origin(
        500000,
        3100000,
        10,
        10,
    )

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint16",
        crs="EPSG:3857",
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(data, 1)


def _create_aoi():
    return AOI.from_coordinates(
        [
            [500500.0, 3099500.0],
            [502500.0, 3099500.0],
            [502500.0, 3100000.0],
            [500500.0, 3100000.0],
            [500500.0, 3099500.0],
        ],
        crs="EPSG:3857",
    )


def test_local_provider_metadata(tmp_path):
    raster_path = tmp_path / "imagery_20260917.tif"
    _create_source_raster(raster_path)

    provider = LocalImageryProvider(raster_path)

    metadata = provider.get_metadata()

    assert metadata.width == 512
    assert metadata.height == 512
    assert metadata.band_count == 1
    assert metadata.crs == "EPSG:3857"
    assert metadata.pixel_size_x == 10.0
    assert metadata.pixel_size_y == 10.0


def test_local_provider_coverage(tmp_path):
    raster_path = tmp_path / "imagery.tif"
    _create_source_raster(raster_path)

    provider = LocalImageryProvider(raster_path)

    assert provider.check_coverage(_create_aoi()) is True

    outside_aoi = AOI.from_coordinates(
        [
            [600000.0, 3000000.0],
            [601000.0, 3000000.0],
            [601000.0, 3001000.0],
            [600000.0, 3001000.0],
            [600000.0, 3000000.0],
        ],
        crs="EPSG:3857",
    )

    assert provider.check_coverage(outside_aoi) is False


def test_local_provider_fetch_window_returns_materialized_artifact(
    tmp_path,
):
    raster_path = tmp_path / "imagery.tif"
    _create_source_raster(raster_path)

    provider = LocalImageryProvider(raster_path)

    window = Window(
        col_off=10,
        row_off=20,
        width=64,
        height=32,
    )

    artifact = provider.fetch_window(window)

    assert isinstance(artifact, RasterWindowArtifact)
    assert artifact.window == window
    assert artifact.data.shape == (1, 32, 64)
    assert artifact.width == 64
    assert artifact.height == 32
    assert artifact.band_count == 1
    assert artifact.crs == "EPSG:3857"


def test_local_provider_fetch_window_preserves_spatial_metadata(
    tmp_path,
):
    raster_path = tmp_path / "imagery.tif"
    _create_source_raster(raster_path)

    provider = LocalImageryProvider(raster_path)

    window = Window(
        col_off=10,
        row_off=20,
        width=64,
        height=32,
    )

    artifact = provider.fetch_window(window)

    expected_transform = rasterio.Affine(
        10.0,
        0.0,
        500100.0,
        0.0,
        -10.0,
        3099800.0,
    )

    assert artifact.transform == expected_transform
    assert artifact.bounds == (
        500100.0,
        3099480.0,
        500740.0,
        3099800.0,
    )


def test_local_provider_fetch_window_rejects_out_of_bounds_window(
    tmp_path,
):
    raster_path = tmp_path / "imagery.tif"
    _create_source_raster(raster_path)

    provider = LocalImageryProvider(raster_path)

    window = Window(
        col_off=500,
        row_off=0,
        width=32,
        height=32,
    )

    try:
        provider.fetch_window(window)
        assert False, "Expected ImageryProviderError"
    except ImageryProviderError:
        pass


def test_local_provider_fetch_tiles(tmp_path):
    raster_path = tmp_path / "imagery.tif"
    tile_dir = tmp_path / "tiles"

    _create_source_raster(raster_path)

    provider = LocalImageryProvider(raster_path)

    tiles = provider.fetch_tiles(
        tile_dir,
        tile_size=128,
        overlap=16,
    )

    assert tiles
    assert all(tile.path for tile in tiles)

    for tile in tiles:
        tile_path = tile_dir / tile.path.split("/")[-1]

        assert tile_path.exists()
        assert tile.crs == "EPSG:3857"
        assert tile.width <= 128
        assert tile.height <= 128


def test_local_provider_timestamp_from_filename(tmp_path):
    raster_path = tmp_path / "imagery_20260917T143000.tif"
    _create_source_raster(raster_path)

    provider = LocalImageryProvider(raster_path)

    assert provider.get_timestamp() == datetime(
        2026,
        9,
        17,
        14,
        30,
        0,
    )


def test_local_provider_accepts_explicit_timestamp(tmp_path):
    raster_path = tmp_path / "imagery.tif"
    _create_source_raster(raster_path)

    timestamp = datetime(2025, 1, 15, 10, 30)

    provider = LocalImageryProvider(
        raster_path,
        timestamp=timestamp,
    )

    assert provider.get_timestamp() == timestamp


def test_local_provider_rejects_missing_raster(tmp_path):
    missing_path = tmp_path / "does_not_exist.tif"

    try:
        LocalImageryProvider(missing_path)
        assert False, "Expected ImageryProviderError"
    except ImageryProviderError:
        pass