"""
inference/infer.py

Canonical Phase 2 inference entrypoint:

    image -> preprocess -> model -> prediction -> confidence -> mask

Returns the PredictionArtifact contract that every later phase (Inference Router,
confidence calibration, vectorization, review queue) builds on top of. Nothing
downstream should ever need to know this is a placeholder VOC-class model instead of
the final land-cover model — that's the whole point of freezing this contract now.
"""

import json
import os
import time

import numpy as np
import torch
from PIL import Image

from inference.models.segmentation_model import load_model
from inference.preprocessing.preprocess import load_image, preprocess


def segment(image_path: str, output_dir: str) -> dict:
    """
    Run segmentation on a single image and write the mask + metadata to output_dir.

    Returns a dict matching the PredictionArtifact contract:
        {
          "classes": [...],
          "confidence": {...summary...},
          "mask_path": "...",
          "model_id": "...",
          "model_version": "...",
          "inference_backend": "cpu",
          "inference_time_ms": ...
        }
    """
    os.makedirs(output_dir, exist_ok=True)

    loaded = load_model()
    image = load_image(image_path)
    input_tensor = preprocess(image, loaded.preprocess_transform)

    start = time.time()
    with torch.no_grad():
        output = loaded.model(input_tensor)["out"][0]  # (num_classes, H, W)
    elapsed_ms = (time.time() - start) * 1000

    probabilities = torch.softmax(output, dim=0)          # (num_classes, H, W)
    confidence_map, class_map = torch.max(probabilities, dim=0)  # each (H, W)

    # The model's preprocessing transform resizes the input (e.g. shorter side to 520px),
    # so its output resolution does NOT match the original image. Every downstream
    # consumer (vectorization, georeferencing, overlay) needs the mask aligned pixel-
    # for-pixel with the ORIGINAL image, so resize back here rather than downstream.
    original_size = image.size  # PIL: (width, height)
    class_map_np = class_map.byte().cpu().numpy()
    confidence_map_np = confidence_map.cpu().numpy()

    class_map_img = Image.fromarray(class_map_np).resize(original_size, resample=Image.NEAREST)
    class_map_np = np.array(class_map_img)

    confidence_img = Image.fromarray(confidence_map_np.astype(np.float32), mode="F").resize(
        original_size, resample=Image.BILINEAR
    )
    confidence_map_np = np.array(confidence_img)

    # Per-class presence + mean confidence, for the "classes": [...] summary the
    # inference doc's example output describes.
    present_class_ids = sorted(set(np.unique(class_map_np).tolist()))
    classes_summary = []
    for class_id in present_class_ids:
        pixel_mask = class_map_np == class_id
        pixel_count = int(pixel_mask.sum())
        mean_conf = float(confidence_map_np[pixel_mask].mean())
        classes_summary.append({
            "class_id": class_id,
            "class_name": loaded.classes[class_id] if class_id < len(loaded.classes) else str(class_id),
            "pixel_count": pixel_count,
            "pixel_fraction": round(pixel_count / class_map_np.size, 4),
            "mean_confidence": round(mean_conf, 4),
        })
    # Largest classes first — mirrors how a reviewer would want to scan results.
    classes_summary.sort(key=lambda c: c["pixel_count"], reverse=True)

    mask_path = os.path.join(output_dir, "mask.png")
    Image.fromarray(class_map_np).save(mask_path)

    confidence_path = os.path.join(output_dir, "confidence.npy")
    np.save(confidence_path, confidence_map_np)

    result = {
        "classes": classes_summary,
        "confidence": {
            "mean": round(float(confidence_map_np.mean()), 4),
            "min": round(float(confidence_map_np.min()), 4),
            "max": round(float(confidence_map_np.max()), 4),
        },
        "mask_path": mask_path,
        "confidence_map_path": confidence_path,
        "model_id": loaded.model_id,
        "model_version": str(loaded.model_version),
        "inference_backend": "cpu",
        "inference_time_ms": round(elapsed_ms, 2),
    }

    with open(os.path.join(output_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python -m inference.infer <image_path> <output_dir>")
        sys.exit(1)

    result = segment(sys.argv[1], sys.argv[2])
    print(json.dumps(result, indent=2))