"""Semantic temporal change detection over aligned classification rasters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, mapping


class ChangeDetectionError(RuntimeError):
    """Raised when temporal comparison inputs are incompatible."""


@dataclass(frozen=True)
class ChangeSummary:
    changed_pixels: int
    total_valid_pixels: int
    changed_fraction: float
    transitions: dict[str, int]
    output_mask: str
    output_geojson: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "changed_pixels": self.changed_pixels,
            "total_valid_pixels": self.total_valid_pixels,
            "changed_fraction": self.changed_fraction,
            "transitions": dict(self.transitions),
            "output_mask": self.output_mask,
            "output_geojson": self.output_geojson,
        }


def detect_semantic_change(
    before_mask_path: str | Path,
    after_mask_path: str | Path,
    output_dir: str | Path,
    *,
    ignore_class_ids: tuple[int, ...] = (0,),
) -> ChangeSummary:
    """Compare two aligned class masks and emit a change mask + polygons.

    Pixel value 0 is treated as no-data by default. A changed pixel is encoded
    as 1 in the output mask; unchanged/ignored pixels are 0.
    """
    before_mask_path = Path(before_mask_path)
    after_mask_path = Path(after_mask_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with rasterio.open(before_mask_path) as before_src:
            before = before_src.read(1)
            before_profile = before_src.profile.copy()
            before_transform = before_src.transform
            before_crs = before_src.crs
            before_shape = before.shape

        with rasterio.open(after_mask_path) as after_src:
            after = after_src.read(1)
            after_transform = after_src.transform
            after_crs = after_src.crs

        if before_shape != after.shape:
            raise ChangeDetectionError("Before/after masks have different shapes")
        if before_crs != after_crs:
            raise ChangeDetectionError("Before/after masks use different CRS")
        if before_transform != after_transform:
            raise ChangeDetectionError("Before/after masks are not spatially aligned")

        valid = np.ones(before.shape, dtype=bool)
        for ignore_id in ignore_class_ids:
            valid &= before != ignore_id
            valid &= after != ignore_id

        changed = valid & (before != after)
        changed_pixels = int(changed.sum())
        valid_pixels = int(valid.sum())

        transitions: dict[str, int] = {}
        pairs = np.stack((before[changed], after[changed]), axis=1) if changed_pixels else np.empty((0, 2), dtype=before.dtype)
        for from_id, to_id in pairs.tolist():
            key = f"{int(from_id)}->{int(to_id)}"
            transitions[key] = transitions.get(key, 0) + 1

        mask_path = output_dir / "change_mask.tif"
        profile = before_profile.copy()
        profile.update(count=1, dtype="uint8", nodata=0)
        with rasterio.open(mask_path, "w", **profile) as dst:
            dst.write(changed.astype(np.uint8), 1)

        features = []
        for geometry, value in shapes(changed.astype(np.uint8), mask=changed, transform=before_transform):
            if int(value) != 1:
                continue
            geom = shape(geometry)
            if geom.is_empty:
                continue
            features.append({
                "type": "Feature",
                "geometry": mapping(geom),
                "properties": {"changed": True},
            })

        geojson_path = output_dir / "changes.geojson"
        geojson_path.write_text(
            json.dumps({
                "type": "FeatureCollection",
                "features": features,
                "properties": {
                    "feature_count": len(features),
                    "transitions": transitions,
                    "crs": before_crs.to_string() if before_crs else None,
                },
            }, indent=2),
            encoding="utf-8",
        )

        return ChangeSummary(
            changed_pixels=changed_pixels,
            total_valid_pixels=valid_pixels,
            changed_fraction=(changed_pixels / valid_pixels) if valid_pixels else 0.0,
            transitions=transitions,
            output_mask=str(mask_path),
            output_geojson=str(geojson_path),
        )
    except ChangeDetectionError:
        raise
    except Exception as exc:
        raise ChangeDetectionError(f"Semantic change detection failed: {exc}") from exc
