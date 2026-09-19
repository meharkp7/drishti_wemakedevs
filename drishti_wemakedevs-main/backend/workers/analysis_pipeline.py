"""End-to-end local DRISHTI analysis pipeline.

This is the first vertical slice that connects the already-built geospatial
and inference primitives:

AOI -> local imagery -> clip -> tiles -> segmentation -> confidence-aware
mosaic -> GIS vectorization -> result artifacts.

Remote imagery, trained LoveDA checkpoints, human review, and temporal
comparison remain swappable stages rather than being hard-coded into the
pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from backend.core.job_state import AnalysisJob
from configs.config import TILE_OVERLAP, TILE_SIZE
from geospatial.aoi import AOI
from geospatial.imagery.provider import LocalImageryProvider
from geospatial.imagery.raster import clip_raster_to_aoi
from geospatial.vectorization import mask_to_geojson
from inference.infer import segment


class AnalysisPipelineError(RuntimeError):
    """Raised when an analysis vertical slice cannot be completed."""


@dataclass(frozen=True)
class AnalysisResult:
    job_id: str
    clipped_raster: str
    tile_count: int
    mask_path: str
    confidence_path: str
    vector_path: str
    statistics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "clipped_raster": self.clipped_raster,
            "tile_count": self.tile_count,
            "mask_path": self.mask_path,
            "confidence_path": self.confidence_path,
            "vector_path": self.vector_path,
            "statistics": self.statistics,
        }


class LocalAnalysisPipeline:
    """Execute a deterministic local analysis for one persisted job."""

    def __init__(
        self,
        *,
        tile_size: int = TILE_SIZE,
        overlap: int = TILE_OVERLAP,
    ) -> None:
        if tile_size <= 0:
            raise ValueError("tile_size must be > 0")
        if overlap < 0 or overlap >= tile_size:
            raise ValueError("overlap must satisfy 0 <= overlap < tile_size")
        self.tile_size = tile_size
        self.overlap = overlap

    def run(self, job: AnalysisJob, output_dir: str | Path) -> AnalysisResult:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            aoi = AOI.from_geojson(
                job.request.aoi["geometry"],
                crs=str(job.request.aoi.get("crs", "EPSG:4326")),
            )
            imagery_path = self._imagery_path(job)
            provider = LocalImageryProvider(imagery_path)

            if not provider.check_coverage(aoi):
                raise AnalysisPipelineError("AOI does not intersect supplied imagery")

            clipped_path = output_dir / "clipped.tif"
            clip_raster_to_aoi(imagery_path, aoi, clipped_path)

            tiles_dir = output_dir / "tiles"
            tiles = provider_for(clipped_path).fetch_tiles(
                tiles_dir,
                tile_size=self.tile_size,
                overlap=self.overlap,
            )
            if not tiles:
                raise AnalysisPipelineError("No imagery tiles were produced")

            mask, confidence, profile = self._run_tiles(tiles, clipped_path, output_dir)

            mask_path = output_dir / "prediction_mask.tif"
            confidence_path = output_dir / "confidence.tif"
            self._write_raster(mask_path, mask, profile, "uint8", nodata=0)
            self._write_raster(confidence_path, confidence, profile, "float32", nodata=0.0)

            vector_path = output_dir / "prediction.geojson"
            vector_result = mask_to_geojson(
                mask_path,
                vector_path,
                confidence_path=confidence_path,
                min_area_m2=0.0,
            )

            class_pixel_counts: dict[str, int] = {}
            for class_id in np.unique(mask):
                class_pixel_counts[str(int(class_id))] = int((mask == class_id).sum())

            result = AnalysisResult(
                job_id=str(job.job_id),
                clipped_raster=str(clipped_path),
                tile_count=len(tiles),
                mask_path=str(mask_path),
                confidence_path=str(confidence_path),
                vector_path=vector_result.output_path,
                statistics={
                    "class_pixel_counts": class_pixel_counts,
                    "mean_confidence": float(confidence.mean()),
                    "vector_feature_count": vector_result.feature_count,
                    "class_area_m2": vector_result.class_area_m2,
                },
            )

            (output_dir / "result.json").write_text(
                json.dumps(result.to_dict(), indent=2),
                encoding="utf-8",
            )
            return result
        except AnalysisPipelineError:
            raise
        except Exception as exc:
            raise AnalysisPipelineError(
                f"Analysis pipeline failed for job {job.job_id}: {exc}"
            ) from exc

    @staticmethod
    def _imagery_path(job: AnalysisJob) -> Path:
        source = job.request.imagery_source
        raw_path = source.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise AnalysisPipelineError(
                "Local analysis requires imagery_source.path"
            )
        path = Path(raw_path)
        if not path.is_file():
            raise AnalysisPipelineError(f"Imagery file does not exist: {path}")
        return path

    def _run_tiles(self, tiles, clipped_path: Path, output_dir: Path):
        with rasterio.open(clipped_path) as src:
            profile = src.profile.copy()
            base_transform = src.transform
            height, width = src.height, src.width

        mosaic_mask = np.zeros((height, width), dtype=np.uint8)
        mosaic_confidence = np.full((height, width), -1.0, dtype=np.float32)

        inference_dir = output_dir / "inference"
        inference_dir.mkdir(parents=True, exist_ok=True)

        for tile in tiles:
            tile_result = segment(tile.path, str(inference_dir / tile.tile_id))
            # The segmentation mask is produced from the tile's RGB image;
            # use its saved prediction artifact rather than the source raster.
            from PIL import Image
            predicted_mask = np.asarray(Image.open(tile_result["mask_path"]))
            tile_confidence = np.load(tile_result["confidence_map_path"]).astype(np.float32)

            a = base_transform
            t = rasterio.Affine(*tile.transform)
            x_offset = int(round((t.c - a.c) / a.a))
            y_offset = int(round((t.f - a.f) / a.e))
            y_end = min(y_offset + predicted_mask.shape[0], height)
            x_end = min(x_offset + predicted_mask.shape[1], width)
            local_h = max(0, y_end - y_offset)
            local_w = max(0, x_end - x_offset)
            if local_h == 0 or local_w == 0:
                continue

            pred = predicted_mask[:local_h, :local_w]
            conf = tile_confidence[:local_h, :local_w]
            target_conf = mosaic_confidence[y_offset:y_end, x_offset:x_end]
            replace = conf > target_conf
            target_mask = mosaic_mask[y_offset:y_end, x_offset:x_end]
            target_mask[replace] = pred[replace]
            target_conf[replace] = conf[replace]

        mosaic_confidence[mosaic_confidence < 0] = 0.0
        return mosaic_mask, mosaic_confidence, profile

    @staticmethod
    def _write_raster(path: Path, array: np.ndarray, profile: dict, dtype: str, nodata):
        output_profile = profile.copy()
        output_profile.update(
            count=1,
            dtype=dtype,
            nodata=nodata,
            compress="deflate",
            tiled=False,
        )
        output_profile.pop("blockxsize", None)
        output_profile.pop("blockysize", None)
        with rasterio.open(path, "w", **output_profile) as dst:
            dst.write(array.astype(dtype), 1)


def provider_for(raster_path: Path) -> LocalImageryProvider:
    """Create a provider for an already-clipped raster."""
    return LocalImageryProvider(raster_path)


def execute_job(
    job_service,
    job_id,
    output_root: str | Path,
    *,
    pipeline: LocalAnalysisPipeline | None = None,
) -> AnalysisResult:
    """Execute one persisted local job while updating its state machine."""
    pipeline = pipeline or LocalAnalysisPipeline()
    job = job_service.get_job(job_id)
    job_service.prepare(job_id)
    job_service.start_imagery_fetch(job_id)
    job_service.start_tiling(job_id)
    job_service.start_inference(job_id)
    job_service.start_postprocessing(job_id)

    output_dir = Path(output_root) / str(job_id)
    try:
        result = pipeline.run(job, output_dir)
    except Exception as exc:
        try:
            job_service.fail(job_id, reason=str(exc)[:500])
        except Exception:
            pass
        if isinstance(exc, AnalysisPipelineError):
            raise
        raise AnalysisPipelineError(str(exc)) from exc

    job_service.complete(job_id)
    return result
