"""
inference/postprocessing/visualize.py

Generates the four Phase 2.4 artifacts:

    original -> segmentation (colored mask) -> overlay -> confidence map

per the implementation plan's "Do not continue until image -> model -> segmentation ->
overlay works reliably" gate.
"""

import os

import numpy as np
from PIL import Image


def _color_palette(num_colors: int) -> np.ndarray:
    """Deterministic distinct-ish colors per class id (simple HSV sweep)."""
    import colorsys

    colors = []
    for i in range(num_colors):
        hue = i / max(num_colors, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
        colors.append([int(r * 255), int(g * 255), int(b * 255)])
    colors[0] = [0, 0, 0]  # background stays black
    return np.array(colors, dtype=np.uint8)


def colorize_mask(class_map: np.ndarray, num_classes: int) -> Image.Image:
    palette = _color_palette(num_classes)
    colored = palette[class_map]
    return Image.fromarray(colored, mode="RGB")


def make_overlay(original: Image.Image, colored_mask: Image.Image, alpha: float = 0.5) -> Image.Image:
    original_rgba = original.convert("RGBA").resize(colored_mask.size)
    mask_rgba = colored_mask.convert("RGBA")
    return Image.blend(original_rgba, mask_rgba, alpha)


def confidence_heatmap(confidence_map: np.ndarray) -> Image.Image:
    """Simple grayscale heatmap: brighter = more confident."""
    normalized = ((confidence_map - confidence_map.min()) /
                  (confidence_map.max() - confidence_map.min() + 1e-8) * 255).astype(np.uint8)
    return Image.fromarray(normalized, mode="L")


def generate_visualizations(image_path: str, result: dict, output_dir: str) -> dict:
    """
    Reads the mask.png / confidence.npy that inference.infer.segment() already wrote,
    and produces original.png, segmentation.png, overlay.png, confidence.png alongside
    them in the same output_dir.
    """
    original = Image.open(image_path).convert("RGB")
    class_map = np.array(Image.open(result["mask_path"]))
    confidence_map = np.load(result["confidence_map_path"])
    num_classes = int(class_map.max()) + 1

    colored_mask = colorize_mask(class_map, num_classes)
    overlay = make_overlay(original, colored_mask)
    conf_img = confidence_heatmap(confidence_map)

    original_out = os.path.join(output_dir, "original.png")
    segmentation_out = os.path.join(output_dir, "segmentation.png")
    overlay_out = os.path.join(output_dir, "overlay.png")
    confidence_out = os.path.join(output_dir, "confidence.png")

    original.save(original_out)
    colored_mask.save(segmentation_out)
    overlay.save(overlay_out)
    conf_img.save(confidence_out)

    return {
        "original": original_out,
        "segmentation": segmentation_out,
        "overlay": overlay_out,
        "confidence_heatmap": confidence_out,
    }