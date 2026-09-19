"""
datasets/loveda/transforms.py

Paired image+mask transforms for LoveDA.

Segmentation transforms must apply identical geometric operations to the
image and its mask -- a flip or resize that only touches the image silently
corrupts every label in that sample. Every transform factory here returns a
callable of the shape ``(image, mask) -> (image_tensor, mask_tensor)`` for
that reason; there is no "transform an image alone" entrypoint in this
module.

``eval_transform`` is deterministic (no randomness), mirroring the
determinism guarantee in ``inference/preprocessing/preprocess.py`` -- LoveDA
validation numbers must be reproducible run to run.

``train_transform`` adds bounded random flips/rotation and image-only color
jitter. It intentionally avoids random resize/crop, which could push a rare
class fully out of view without any signal that it happened -- augmentation
can grow once first training numbers exist.
"""

from __future__ import annotations

import random
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

# ImageNet normalization statistics. Kept consistent with the general
# pretrained model DRISHTI's Phase-2 inference pipeline currently loads
# (torchvision DeepLabV3), so LoveDA-trained weights and the current
# placeholder model use the same input convention until a LoveDA-specific
# model replaces it.
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

PairedTransform = Callable[
    [Image.Image, Optional[Image.Image]],
    Tuple[torch.Tensor, Optional[torch.Tensor]],
]


def _to_mask_tensor(mask: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(mask, dtype=np.int64))


def _to_image_tensor(image: Image.Image) -> torch.Tensor:
    tensor = TF.to_tensor(image)
    return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)


def eval_transform(size: Optional[int] = None) -> PairedTransform:
    """Deterministic transform: optional resize, then normalize. No randomness."""

    def _apply(
        image: Image.Image,
        mask: Optional[Image.Image],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if size is not None:
            image = image.resize((size, size), resample=Image.BILINEAR)
            if mask is not None:
                mask = mask.resize((size, size), resample=Image.NEAREST)

        image_tensor = _to_image_tensor(image)
        mask_tensor = _to_mask_tensor(mask) if mask is not None else None
        return image_tensor, mask_tensor

    return _apply


def train_transform(
    size: Optional[int] = None,
    *,
    hflip_prob: float = 0.5,
    vflip_prob: float = 0.5,
    rotate_degrees: Sequence[int] = (0, 90, 180, 270),
    color_jitter: bool = True,
) -> PairedTransform:
    """Modest, label-safe augmentation for training."""
    jitter = None
    if color_jitter:
        from torchvision.transforms import ColorJitter

        jitter = ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10)

    def _apply(
        image: Image.Image,
        mask: Optional[Image.Image],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if size is not None:
            image = image.resize((size, size), resample=Image.BILINEAR)
            if mask is not None:
                mask = mask.resize((size, size), resample=Image.NEAREST)

        if random.random() < hflip_prob:
            image = TF.hflip(image)
            if mask is not None:
                mask = TF.hflip(mask)

        if random.random() < vflip_prob:
            image = TF.vflip(image)
            if mask is not None:
                mask = TF.vflip(mask)

        angle = random.choice(rotate_degrees)
        if angle:
            image = image.rotate(angle)
            if mask is not None:
                mask = mask.rotate(angle)

        if jitter is not None:
            image = jitter(image)  # color-only -- the mask must stay untouched

        image_tensor = _to_image_tensor(image)
        mask_tensor = _to_mask_tensor(mask) if mask is not None else None
        return image_tensor, mask_tensor

    return _apply
