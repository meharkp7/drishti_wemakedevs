"""Device-aware inference routing public API."""

from inference.router.router import (
    BackendInfo,
    InferenceRequest,
    InferenceRouter,
    InferenceRouterError,
    discover_backends,
)

__all__ = [
    "BackendInfo",
    "InferenceRequest",
    "InferenceRouter",
    "InferenceRouterError",
    "discover_backends",
]
