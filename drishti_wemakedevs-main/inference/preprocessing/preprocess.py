"""
inference/preprocessing/preprocess.py

image -> resize/normalize -> tensor

Kept deterministic per the implementation plan (Phase 2.2): same input always produces
the same tensor, no randomness, no augmentation. Augmentation belongs to training, not
inference.
"""

from PIL import Image
import torch


def load_image(image_path: str) -> Image.Image:
    """Load an image from disk and force RGB (drops alpha, handles grayscale sources)."""
    return Image.open(image_path).convert("RGB")


def preprocess(image: Image.Image, transform) -> torch.Tensor:
    """
    Apply the model's expected preprocessing transform and add a batch dimension.

    `transform` is the torchvision weight-specific transform returned by
    `LoadedModel.preprocess_transform` — this keeps preprocessing coupled to whichever
    model is currently loaded, rather than hardcoding resize/normalize constants here.
    """
    tensor = transform(image)
    return tensor.unsqueeze(0)  # (1, C, H, W)