"""
Imagery provider abstractions for DRISHTI.

Phase 4 introduces a stable interface between DRISHTI's geospatial
processing pipeline and the source of imagery.

The first implementation is intentionally local-first:
    LocalImageryProvider -> local raster/GeoTIFF

Future providers can implement the same interface for:
    - WMS / WMTS
    - Sentinel or other remote imagery
    - cloud-backed imagery
    - cached imagery services

The provider layer is responsible for obtaining imagery.
Geospatial processing remains responsible for clipping, tiling,
and downstream analysis.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import rasterio
from rasterio.windows import Window
from shapely.geometry import box

from geospatial.aoi import AOI
from geospatial.imagery.raster import (
    RasterError,
    RasterMetadata,
    check_aoi_coverage,
    read_raster_metadata,
)
from geospatial.tiling import TileArtifact, tile_raster


class ImageryProviderError(RuntimeError):
    """Raised when an imagery provider cannot fulfill a request."""


@dataclass(frozen=True)
class RasterWindowArtifact:
    """
    Materialized raster window returned by an imagery provider.

    Unlike an open Rasterio dataset, this artifact owns plain raster
    data and its spatial metadata. It is therefore safe to pass between
    processing stages without keeping a file handle open.
    """

    data: object
    window: Window
    transform: object
    crs: str
    bounds: tuple[float, float, float, float]

    @property
    def height(self) -> int:
        """Number of rows in the materialized window."""
        return int(self.data.shape[-2])

    @property
    def width(self) -> int:
        """Number of columns in the materialized window."""
        return int(self.data.shape[-1])

    @property
    def band_count(self) -> int:
        """Number of raster bands in the materialized window."""
        if self.data.ndim == 2:
            return 1

        return int(self.data.shape[0])


class ImageryProvider(ABC):
    """
    Abstract interface for imagery acquisition.

    Implementations must provide the same contract regardless of
    where imagery is actually stored or retrieved from.
    """

    @abstractmethod
    def get_metadata(self) -> RasterMetadata:
        """Return metadata describing the available imagery."""

    @abstractmethod
    def check_coverage(self, aoi: AOI) -> bool:
        """Return whether imagery intersects the requested AOI."""

    @abstractmethod
    def fetch_window(
        self,
        window: Window,
    ) -> RasterWindowArtifact:
        """Materialize and return the requested raster window."""

    @abstractmethod
    def fetch_tiles(
        self,
        output_dir: str | Path,
        tile_size: int,
        overlap: int,
    ) -> List[TileArtifact]:
        """Materialize imagery tiles into ``output_dir``."""

    @abstractmethod
    def get_timestamp(self) -> datetime:
        """Return the acquisition timestamp of the imagery."""


class LocalImageryProvider(ImageryProvider):
    """
    Imagery provider backed by a local raster file.

    This is the Phase-4 reference implementation. It deliberately
    delegates raster inspection and tiling to the existing geospatial
    core instead of duplicating those operations.
    """

    def __init__(
        self,
        raster_path: str | Path,
        timestamp: datetime | None = None,
    ) -> None:
        self._raster_path = Path(raster_path)

        if not self._raster_path.exists():
            raise ImageryProviderError(
                f"Imagery raster does not exist: {self._raster_path}"
            )

        if not self._raster_path.is_file():
            raise ImageryProviderError(
                f"Imagery raster path is not a file: {self._raster_path}"
            )

        try:
            self._metadata = read_raster_metadata(self._raster_path)
        except RasterError as exc:
            raise ImageryProviderError(
                f"Unable to initialize imagery provider: "
                f"{self._raster_path}"
            ) from exc

        self._timestamp = timestamp or self._infer_timestamp()

    def get_metadata(self) -> RasterMetadata:
        """Return metadata for the configured local raster."""
        return self._metadata

    def check_coverage(self, aoi: AOI) -> bool:
        """Check whether the configured raster covers the AOI."""
        try:
            return check_aoi_coverage(self._raster_path, aoi)
        except RasterError as exc:
            raise ImageryProviderError(
                f"Unable to check imagery coverage: "
                f"{self._raster_path}"
            ) from exc

    def fetch_window(
        self,
        window: Window,
    ) -> RasterWindowArtifact:
        """
        Read and materialize a raster window.

        The returned artifact contains no open file handle.
        """
        if (
            window.col_off < 0
            or window.row_off < 0
            or window.width <= 0
            or window.height <= 0
        ):
            raise ImageryProviderError(
                f"Invalid raster window: {window}"
            )

        try:
            with rasterio.open(self._raster_path) as src:
                if window.col_off + window.width > src.width:
                    raise ImageryProviderError(
                        f"Raster window exceeds raster width: {window}"
                    )

                if window.row_off + window.height > src.height:
                    raise ImageryProviderError(
                        f"Raster window exceeds raster height: {window}"
                    )

                data = src.read(window=window)
                transform = src.window_transform(window)

                bounds = tuple(
                    float(value)
                    for value in rasterio.windows.bounds(
                        window,
                        src.transform,
                    )
                )

                return RasterWindowArtifact(
                    data=data,
                    window=window,
                    transform=transform,
                    crs=src.crs.to_string(),
                    bounds=bounds,
                )

        except ImageryProviderError:
            raise
        except Exception as exc:
            raise ImageryProviderError(
                f"Unable to fetch raster window: {window}"
            ) from exc

    def fetch_tiles(
        self,
        output_dir: str | Path,
        tile_size: int,
        overlap: int,
    ) -> List[TileArtifact]:
        """Generate georeferenced tiles from the local imagery."""
        try:
            return tile_raster(
                self._raster_path,
                output_dir,
                tile_size=tile_size,
                overlap=overlap,
            )
        except Exception as exc:
            raise ImageryProviderError(
                f"Unable to fetch imagery tiles from: "
                f"{self._raster_path}"
            ) from exc

    def get_timestamp(self) -> datetime:
        """Return the configured or inferred acquisition timestamp."""
        return self._timestamp

    def _infer_timestamp(self) -> datetime:
        """
        Infer a timestamp from the raster filename when possible.

        Supported filename convention:
            YYYYMMDD
            YYYYMMDDTHHMMSS

        If no timestamp can be inferred, use the file modification time.
        """
        stem = self._raster_path.stem

        candidates = [
            ("%Y%m%dT%H%M%S", 15),
            ("%Y%m%d", 8),
        ]

        for fmt, length in candidates:
            for start in range(len(stem) - length + 1):
                candidate = stem[start : start + length]

                try:
                    return datetime.strptime(candidate, fmt)
                except ValueError:
                    continue

        return datetime.fromtimestamp(
            self._raster_path.stat().st_mtime
        )