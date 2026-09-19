"""
Raster metadata and AOI-aware operations for DRISHTI.

This module intentionally does not implement an imagery provider yet.
It provides the low-level raster operations that the ImageryProvider
layer will use later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_bounds
from shapely.geometry import mapping

from geospatial.aoi import AOI


class RasterError(RuntimeError):
    """Raised when a raster cannot be inspected or processed."""


@dataclass(frozen=True)
class RasterMetadata:
    """Metadata required by downstream DRISHTI geospatial processing."""

    path: str
    width: int
    height: int
    band_count: int
    dtype: str
    crs: str
    bounds: tuple[float, float, float, float]
    resolution: tuple[float, float]
    nodata: Optional[float]
    driver: str

    @property
    def pixel_size_x(self) -> float:
        """Pixel width in raster CRS units."""
        return float(self.resolution[0])

    @property
    def pixel_size_y(self) -> float:
        """Pixel height in raster CRS units."""
        return float(self.resolution[1])


def read_raster_metadata(path: str | Path) -> RasterMetadata:
    """
    Read raster metadata without loading the complete raster into memory.
    """
    raster_path = Path(path)

    if not raster_path.exists():
        raise RasterError(f"Raster does not exist: {raster_path}")

    if not raster_path.is_file():
        raise RasterError(f"Raster path is not a file: {raster_path}")

    try:
        with rasterio.open(raster_path) as src:
            if src.crs is None:
                raise RasterError(
                    f"Raster has no CRS: {raster_path}"
                )

            bounds = (
                float(src.bounds.left),
                float(src.bounds.bottom),
                float(src.bounds.right),
                float(src.bounds.top),
            )

            resolution = (
                float(abs(src.transform.a)),
                float(abs(src.transform.e)),
            )

            return RasterMetadata(
                path=str(raster_path),
                width=src.width,
                height=src.height,
                band_count=src.count,
                dtype=str(src.dtypes[0]),
                crs=src.crs.to_string(),
                bounds=bounds,
                resolution=resolution,
                nodata=src.nodata,
                driver=src.driver,
            )

    except RasterError:
        raise
    except Exception as exc:
        raise RasterError(
            f"Failed to read raster metadata: {raster_path}"
        ) from exc


def check_aoi_coverage(
    raster_path: str | Path,
    aoi: AOI,
) -> bool:
    """
    Check whether the raster spatial extent intersects the AOI.

    The AOI is transformed into the raster CRS before comparison.
    """
    raster_path = Path(raster_path)

    if not raster_path.exists():
        raise RasterError(f"Raster does not exist: {raster_path}")

    try:
        with rasterio.open(raster_path) as src:
            if src.crs is None:
                raise RasterError(
                    f"Raster has no CRS: {raster_path}"
                )

            raster_bounds = src.bounds

            aoi_bounds = transform_bounds(
                aoi.crs,
                src.crs,
                *aoi.bounds,
                densify_pts=21,
            )

            aoi_left, aoi_bottom, aoi_right, aoi_top = aoi_bounds

            return not (
                aoi_right <= raster_bounds.left
                or aoi_left >= raster_bounds.right
                or aoi_top <= raster_bounds.bottom
                or aoi_bottom >= raster_bounds.top
            )

    except RasterError:
        raise
    except Exception as exc:
        raise RasterError(
            f"Failed to check AOI coverage for: {raster_path}"
        ) from exc


def clip_raster_to_aoi(
    raster_path: str | Path,
    aoi: AOI,
    output_path: str | Path,
) -> RasterMetadata:
    """
    Clip a raster to an AOI and write the result to ``output_path``.

    The output retains:
    - raster CRS
    - transform
    - resolution
    - band structure
    - nodata value
    - source driver where supported

    Returns
    -------
    RasterMetadata
        Metadata of the generated clipped raster.
    """
    raster_path = Path(raster_path)
    output_path = Path(output_path)

    if not raster_path.exists():
        raise RasterError(f"Raster does not exist: {raster_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with rasterio.open(raster_path) as src:
            if src.crs is None:
                raise RasterError(
                    f"Raster has no CRS: {raster_path}"
                )

            if not check_aoi_coverage(raster_path, aoi):
                raise RasterError(
                    "AOI does not intersect the raster extent."
                )

            geometry = aoi.geometry

            if aoi.crs != src.crs.to_string():
                from rasterio.warp import transform_geom

                geometry = transform_geom(
                    aoi.crs,
                    src.crs,
                    mapping(aoi.geometry),
                )

            clipped, clipped_transform = mask(
                src,
                [geometry],
                crop=True,
                filled=True,
                nodata=src.nodata,
            )

            profile = src.profile.copy()

            profile.update(
                {
                    "height": clipped.shape[1],
                    "width": clipped.shape[2],
                    "transform": clipped_transform,
                }
            )

            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(clipped)

    except RasterError:
        raise
    except ValueError as exc:
        raise RasterError(
            f"AOI clipping failed for raster: {raster_path}"
        ) from exc
    except Exception as exc:
        raise RasterError(
            f"Failed to clip raster: {raster_path}"
        ) from exc

    return read_raster_metadata(output_path)