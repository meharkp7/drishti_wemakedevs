"""Convert a georeferenced semantic mask into GIS-ready polygons."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, mapping
from shapely.ops import transform
from pyproj import Transformer


class VectorizationError(RuntimeError):
    """Raised when a raster mask cannot be converted to valid vectors."""


@dataclass(frozen=True)
class VectorFeature:
    class_id: int
    area_m2: float
    confidence: float | None
    geometry: dict[str, Any]


@dataclass(frozen=True)
class VectorizationResult:
    output_path: str
    feature_count: int
    class_area_m2: dict[int, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_path": self.output_path,
            "feature_count": self.feature_count,
            "class_area_m2": dict(self.class_area_m2),
        }


def _metric_transformer(crs: str):
    source = crs
    if str(source).upper() in {"EPSG:3857"}:
        return None
    # Equal-area calculations are approximated in a local UTM CRS selected
    # from the raster centroid; geometry coordinates remain in raster CRS in
    # the exported GeoJSON.
    return None


def mask_to_geojson(
    mask_path: str | Path,
    output_path: str | Path,
    *,
    confidence_path: str | Path | None = None,
    min_area_m2: float = 0.0,
    ignore_class_ids: tuple[int, ...] = (0,),
) -> VectorizationResult:
    """Polygonize a single-band integer mask while preserving its CRS."""
    mask_path = Path(mask_path)
    output_path = Path(output_path)
    if not mask_path.is_file():
        raise VectorizationError(f"Mask does not exist: {mask_path}")
    if min_area_m2 < 0:
        raise VectorizationError("min_area_m2 must be >= 0")

    try:
        with rasterio.open(mask_path) as src:
            if src.count != 1:
                raise VectorizationError("Mask raster must contain exactly one band")
            mask_array = src.read(1)
            transform_affine = src.transform
            crs = src.crs
            if crs is None:
                raise VectorizationError("Mask raster must have a CRS")

        confidence = None
        if confidence_path is not None:
            confidence_path = Path(confidence_path)
            if confidence_path.suffix.lower() in {".npy", ".npz"}:
                confidence = np.load(confidence_path)
                if hasattr(confidence, "files"):
                    confidence = confidence["arr_0"]
            else:
                with rasterio.open(confidence_path) as confidence_src:
                    if confidence_src.count != 1:
                        raise VectorizationError("Confidence raster must contain exactly one band")
                    confidence = confidence_src.read(1)
                    if confidence_src.transform != transform_affine or confidence_src.crs != crs:
                        raise VectorizationError("Confidence raster is not spatially aligned with mask")
            if confidence.shape != mask_array.shape:
                raise VectorizationError("Confidence map shape does not match mask")

        features = []
        class_area: dict[int, float] = {}

        # Work in a metric CRS for area filtering/reporting when the source is
        # geographic.  Exported geometries remain in the source CRS.
        source_crs = crs.to_string()
        to_metric = None
        if crs.is_geographic:
            from pyproj import CRS
            from rasterio.warp import transform_bounds
            bounds = src.bounds if 'src' in locals() else None
            if bounds is not None:
                lon = (bounds.left + bounds.right) / 2
                lat = (bounds.bottom + bounds.top) / 2
                zone = int((lon + 180) / 6) + 1
                epsg = 32600 + zone if lat >= 0 else 32700 + zone
                to_metric = Transformer.from_crs(
                    source_crs, f"EPSG:{epsg}", always_xy=True
                )

        for raw_geometry, raw_value in shapes(mask_array, transform=transform_affine):
            class_id = int(raw_value)
            if class_id in ignore_class_ids:
                continue
            geometry = shape(raw_geometry)
            if geometry.is_empty:
                continue

            metric_geometry = geometry
            if to_metric is not None:
                metric_geometry = transform(to_metric.transform, geometry)
            area_m2 = float(metric_geometry.area)
            if area_m2 < min_area_m2:
                continue

            mean_conf = None
            if confidence is not None:
                # Rasterio's shapes() emits connected components; use the
                # component geometry as a mask to estimate mean confidence.
                from rasterio.features import geometry_mask
                component_mask = geometry_mask(
                    [raw_geometry],
                    out_shape=mask_array.shape,
                    transform=transform_affine,
                    invert=True,
                )
                values = confidence[component_mask]
                if values.size:
                    mean_conf = float(np.mean(values))

            class_area[class_id] = class_area.get(class_id, 0.0) + area_m2
            features.append({
                "type": "Feature",
                "geometry": mapping(geometry),
                "properties": {
                    "class_id": class_id,
                    "area_m2": area_m2,
                    "mean_confidence": mean_conf,
                    "crs": source_crs,
                },
            })

        output_path.parent.mkdir(parents=True, exist_ok=True)
        collection = {
            "type": "FeatureCollection",
            "features": features,
            "properties": {
                "crs": source_crs,
                "feature_count": len(features),
            },
        }
        output_path.write_text(json.dumps(collection, indent=2), encoding="utf-8")

        return VectorizationResult(
            output_path=str(output_path),
            feature_count=len(features),
            class_area_m2=class_area,
        )
    except VectorizationError:
        raise
    except Exception as exc:
        raise VectorizationError(f"Vectorization failed for {mask_path}: {exc}") from exc
