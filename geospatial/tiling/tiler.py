"""
Deterministic geospatial raster tiling for DRISHTI.

The tiler converts an AOI-clipped raster into fixed-size tiles while
preserving the spatial reference of every tile.

Key guarantees:
- deterministic tile ordering
- configurable tile size
- configurable overlap
- preserved CRS
- preserved affine transform
- no spatially meaningless image-only crops
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

from configs.config import TILE_OVERLAP, TILE_SIZE


class TilingError(ValueError):
    """Raised when raster tiling parameters or operations are invalid."""


@dataclass(frozen=True)
class TileArtifact:
    """Metadata describing one generated geospatial tile."""

    tile_id: str
    path: str

    row: int
    col: int

    width: int
    height: int

    crs: str

    bounds: tuple[float, float, float, float]

    transform: tuple[float, ...]


def _validate_parameters(tile_size: int, overlap: int) -> None:
    """Validate tiling parameters before touching the raster."""
    if not isinstance(tile_size, int) or tile_size <= 0:
        raise TilingError("tile_size must be a positive integer.")

    if not isinstance(overlap, int) or overlap < 0:
        raise TilingError("overlap must be a non-negative integer.")

    if overlap >= tile_size:
        raise TilingError(
            "overlap must be smaller than tile_size."
        )


def _tile_starts(
    dimension: int,
    tile_size: int,
    overlap: int,
) -> List[int]:
    """
    Generate deterministic pixel offsets for one raster dimension.

    Tiles advance using ``tile_size - overlap`` until the remaining
    edge would create an unnecessarily small tile. In that case the
    final tile is shifted backwards so that it ends exactly at the
    raster boundary.

    This guarantees:
    - complete raster coverage
    - no tile larger than ``tile_size``
    - deterministic ordering
    - reasonably sized edge tiles
    """
    if dimension <= 0:
        raise TilingError("Raster dimension must be positive.")

    if dimension <= tile_size:
        return [0]

    stride = tile_size - overlap

    starts: List[int] = [0]
    position = 0

    while True:
        next_position = position + stride

        if next_position + tile_size >= dimension:
            final_position = dimension - tile_size

            if final_position > starts[-1]:
                starts.append(final_position)

            break

        starts.append(next_position)
        position = next_position

    return starts


def tile_raster(
    raster_path: str | Path,
    output_dir: str | Path,
    *,
    tile_size: int = TILE_SIZE,
    overlap: int = TILE_OVERLAP,
) -> List[TileArtifact]:
    """
    Tile a raster into deterministic geospatial windows.

    Parameters
    ----------
    raster_path:
        Input raster, normally the AOI-clipped raster.

    output_dir:
        Directory in which generated GeoTIFF tiles are stored.

    tile_size:
        Maximum width/height of a tile in pixels.

    overlap:
        Number of overlapping pixels between adjacent tiles.

    Returns
    -------
    list[TileArtifact]
        Tiles ordered deterministically in row-major order.

    Notes
    -----
    Tiles retain the input raster's CRS and receive a correctly
    calculated affine transform for their spatial window.
    """
    _validate_parameters(tile_size, overlap)

    raster_path = Path(raster_path)
    output_dir = Path(output_dir)

    if not raster_path.exists():
        raise TilingError(f"Raster does not exist: {raster_path}")

    if not raster_path.is_file():
        raise TilingError(f"Raster path is not a file: {raster_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: List[TileArtifact] = []

    try:
        with rasterio.open(raster_path) as src:
            if src.crs is None:
                raise TilingError(
                    "Input raster must have a CRS."
                )

            row_starts = _tile_starts(
                src.height,
                tile_size,
                overlap,
            )

            col_starts = _tile_starts(
                src.width,
                tile_size,
                overlap,
            )

            for row_index, row_start in enumerate(row_starts):
                for col_index, col_start in enumerate(col_starts):

                    width = min(
                        tile_size,
                        src.width - col_start,
                    )

                    height = min(
                        tile_size,
                        src.height - row_start,
                    )

                    window = Window(
                        col_off=col_start,
                        row_off=row_start,
                        width=width,
                        height=height,
                    )

                    data = src.read(window=window)

                    tile_transform = window_transform(
                        window,
                        src.transform,
                    )

                    tile_id = (
                        f"tile_r{row_index:04d}"
                        f"_c{col_index:04d}"
                    )

                    tile_path = output_dir / f"{tile_id}.tif"

                    profile = src.profile.copy()
                    profile.update(
                        {
                            "height": height,
                            "width": width,
                            "transform": tile_transform,
                        }
                    )

                    with rasterio.open(
                        tile_path,
                        "w",
                        **profile,
                    ) as dst:
                        dst.write(data)

                    bounds = rasterio.windows.bounds(
                        window,
                        src.transform,
                    )

                    artifact = TileArtifact(
                        tile_id=tile_id,
                        path=str(tile_path),
                        row=row_index,
                        col=col_index,
                        width=width,
                        height=height,
                        crs=src.crs.to_string(),
                        bounds=(
                            float(bounds[0]),
                            float(bounds[1]),
                            float(bounds[2]),
                            float(bounds[3]),
                        ),
                        transform=tuple(
                            float(value)
                            for value in tile_transform
                        ),
                    )

                    artifacts.append(artifact)

    except TilingError:
        raise
    except Exception as exc:
        raise TilingError(
            f"Failed to tile raster: {raster_path}"
        ) from exc

    return artifacts