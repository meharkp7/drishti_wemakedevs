"""
AOI (Area of Interest) primitives and validation for DRISHTI.

An AOI is represented as a GeoJSON-like polygon and carries the CRS in
which its coordinates are expressed.

Responsibilities:
- validate polygon geometry
- validate CRS
- calculate area in square metres
- normalize geometry representation
- provide deterministic serialization
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Dict, Mapping

from pyproj import CRS, Transformer
from shapely.geometry import Polygon, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

from configs.config import MAX_AOI_AREA_M2, SUPPORTED_CRS


class AOIError(ValueError):
    """Raised when an AOI fails validation."""


@dataclass(frozen=True)
class AOI:
    """
    Immutable Area of Interest.

    Parameters
    ----------
    geometry:
        Shapely Polygon or GeoJSON geometry mapping.
    crs:
        Coordinate reference system, e.g. ``EPSG:4326``.
    """

    geometry: Polygon
    crs: str

    def __post_init__(self) -> None:
        normalized_crs = self._normalize_crs(self.crs)

        if not isinstance(self.geometry, Polygon):
            raise AOIError("AOI geometry must be a Polygon.")

        if self.geometry.is_empty:
            raise AOIError("AOI geometry cannot be empty.")

        if not self.geometry.is_valid:
            raise AOIError(
                f"AOI geometry is invalid: {self.geometry.explain_validity()}"
            )

        if self.geometry.area <= 0:
            raise AOIError("AOI geometry must have non-zero area.")

        if normalized_crs not in SUPPORTED_CRS:
            raise AOIError(
                f"Unsupported AOI CRS '{normalized_crs}'. "
                f"Supported CRS: {SUPPORTED_CRS}"
            )

        object.__setattr__(self, "crs", normalized_crs)

        area_m2 = self.area_m2

        if area_m2 > MAX_AOI_AREA_M2:
            raise AOIError(
                f"AOI area ({area_m2:.2f} m²) exceeds the configured "
                f"maximum of {MAX_AOI_AREA_M2:.2f} m²."
            )

    @staticmethod
    def _normalize_crs(crs: str) -> str:
        """Normalize a CRS into the canonical EPSG representation."""
        if not isinstance(crs, str) or not crs.strip():
            raise AOIError("CRS must be a non-empty string.")

        try:
            parsed = CRS.from_user_input(crs)
        except Exception as exc:
            raise AOIError(f"Invalid CRS '{crs}'.") from exc

        authority = parsed.to_authority()

        if authority is None:
            raise AOIError(
                f"CRS '{crs}' does not resolve to a supported authority code."
            )

        return f"{authority[0]}:{authority[1]}"

    @classmethod
    def from_geojson(
        cls,
        geometry: Mapping[str, Any],
        crs: str = "EPSG:4326",
    ) -> "AOI":
        """Construct an AOI from a GeoJSON geometry mapping."""
        try:
            geom = shape(geometry)
        except Exception as exc:
            raise AOIError("Invalid GeoJSON geometry.") from exc

        if not isinstance(geom, Polygon):
            raise AOIError("GeoJSON AOI must be a Polygon.")

        return cls(geometry=geom, crs=crs)

    @classmethod
    def from_coordinates(
        cls,
        coordinates: list[list[float]],
        crs: str = "EPSG:4326",
    ) -> "AOI":
        """
        Construct an AOI from polygon coordinates.

        The first and last coordinate should represent the same point.
        """
        try:
            polygon = Polygon(coordinates)
        except Exception as exc:
            raise AOIError("Invalid polygon coordinates.") from exc

        return cls(geometry=polygon, crs=crs)

    @property
    def area_m2(self) -> float:
        """
        Return AOI area in square metres.

        Geographic CRS coordinates such as EPSG:4326 are transformed to
        a local metric CRS before calculating area.
        """
        if self.crs == "EPSG:3857":
            return float(self.geometry.area)

        if self.crs == "EPSG:4326":
            metric_crs = self._metric_crs_for_geometry()

            transformer = Transformer.from_crs(
                self.crs,
                metric_crs,
                always_xy=True,
            )

            projected = transform(transformer.transform, self.geometry)

            return float(projected.area)

        raise AOIError(f"Area calculation is not implemented for {self.crs}.")

    def _metric_crs_for_geometry(self) -> str:
        """
        Select a UTM CRS appropriate for the AOI centroid.

        This is used only for local area calculation.
        """
        centroid = self.geometry.centroid

        longitude = centroid.x
        latitude = centroid.y

        zone = int((longitude + 180) / 6) + 1

        if latitude >= 0:
            epsg = 32600 + zone
        else:
            epsg = 32700 + zone

        return f"EPSG:{epsg}"

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return (min_x, min_y, max_x, max_y)."""
        return tuple(float(value) for value in self.geometry.bounds)

    def to_geojson(self) -> Dict[str, Any]:
        """Return a deterministic GeoJSON geometry mapping."""
        return {
            "type": "Polygon",
            "coordinates": [
                [
                    [float(x), float(y)]
                    for x, y in ring.coords
                ]
                for ring in [self.geometry.exterior]
            ],
        }

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable AOI representation."""
        return {
            "geometry": self.to_geojson(),
            "crs": self.crs,
            "area_m2": self.area_m2,
            "bounds": list(self.bounds),
        }

    def to_json(self) -> str:
        """Return deterministic JSON representation."""
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )

    def reproject(self, target_crs: str) -> "AOI":
        """Return a new AOI reprojected into ``target_crs``."""
        normalized_target = self._normalize_crs(target_crs)

        if normalized_target not in SUPPORTED_CRS:
            raise AOIError(
                f"Unsupported target CRS '{normalized_target}'. "
                f"Supported CRS: {SUPPORTED_CRS}"
            )

        if normalized_target == self.crs:
            return self

        transformer = Transformer.from_crs(
            self.crs,
            normalized_target,
            always_xy=True,
        )

        projected = transform(transformer.transform, self.geometry)

        return AOI(
            geometry=projected,
            crs=normalized_target,
        )