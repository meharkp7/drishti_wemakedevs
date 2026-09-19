"""
inference/models/segmentation_model.py

Phase 2 baseline model.

DRISHTI's real target is a land-cover class set (building / road / vegetation / water).
Training or fine-tuning that model is out of scope for Phase 2 — per the implementation
plan, Phase 2's only job is to prove the mechanical pipeline:

    image -> preprocess -> model -> prediction -> confidence -> mask

So this loads a general-purpose, ImageNet/COCO-pretrained semantic segmentation model
(DeepLabV3 + ResNet-50 backbone, shipped by torchvision) purely to validate that the
pipeline works end to end. Swapping in a geospatial-specific model later is a drop-in
replacement: everything downstream only depends on the PredictionArtifact contract
below, not on which model produced it.
"""

import warnings
from dataclasses import dataclass
from functools import lru_cache

import torch
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

# COCO-with-VOC-labels class set that ships with this torchvision weight set.
# NOTE: placeholder classes for Phase 2 plumbing only — not the final DRISHTI class set.
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


@dataclass
class LoadedModel:
    model: torch.nn.Module
    classes: list
    model_id: str
    model_version: str
    preprocess_transform: object


@lru_cache(maxsize=1)
def load_model() -> LoadedModel:
    """
    Loads the pretrained segmentation model once per process (cached).
    Runs on CPU — this sandbox / most hackathon laptops have no GPU, and the
    Inference Router (Phase 13) is what will later pick GPU/NPU/CPU/cloud per request.
    """
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    transform = weights.transforms()

    try:
        model = deeplabv3_resnet50(weights=weights)
        model_id = "deeplabv3_resnet50_coco_voc"
        model_version = weights.value if hasattr(weights, "value") else "default"
    except Exception as exc:
        # Pretrained weights live on download.pytorch.org, which some sandboxed/offline
        # environments (including this one) don't have network access to. Fall back to
        # a randomly-initialized model of the identical architecture so the PIPELINE
        # (preprocess -> model -> mask -> confidence -> overlay) can still be validated
        # end to end. Predictions will be meaningless until real weights load — this is
        # a plumbing fallback, not a substitute for the pretrained model in the demo.
        warnings.warn(
            f"Could not download pretrained weights ({exc}). "
            "Falling back to a randomly-initialized model for pipeline validation only. "
            "Re-run with internet access to download.pytorch.org to get real predictions."
        )
        model = deeplabv3_resnet50(weights=None, weights_backbone=None)
        model_id = "deeplabv3_resnet50_UNTRAINED_FALLBACK"
        model_version = "random_init"

    model.eval()

    return LoadedModel(
        model=model,
        classes=VOC_CLASSES,
        model_id=model_id,
        model_version=str(model_version),
        preprocess_transform=transform,
    )