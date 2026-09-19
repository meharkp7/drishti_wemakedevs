"""Device-aware inference routing for DRISHTI.

The router separates *where* inference runs from *what* model performs it.
Every backend returns the same PredictionArtifact-shaped dictionary so later
stages never need to know whether inference used CPU, CUDA, or Apple MPS.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Mapping, Any

import torch


class InferenceRouterError(RuntimeError):
    """Raised when no suitable inference backend is available."""


@dataclass(frozen=True)
class InferenceRequest:
    """Validated routing request."""

    preferred_backend: str | None = None
    allow_fallback: bool = True


@dataclass(frozen=True)
class BackendInfo:
    name: str
    available: bool
    reason: str


def discover_backends() -> tuple[BackendInfo, ...]:
    """Return deterministic availability information for local backends."""
    cuda = torch.cuda.is_available()
    mps = bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )
    return (
        BackendInfo("cuda", cuda, "CUDA available" if cuda else "CUDA unavailable"),
        BackendInfo("mps", mps, "MPS available" if mps else "MPS unavailable"),
        BackendInfo("cpu", True, "CPU is always available"),
    )


class InferenceRouter:
    """Select a local device and execute a supplied inference callable."""

    _PRIORITY = ("cuda", "mps", "cpu")

    def __init__(self) -> None:
        self._backends = {info.name: info for info in discover_backends()}

    @property
    def backends(self) -> tuple[BackendInfo, ...]:
        return tuple(self._backends[name] for name in self._PRIORITY)

    def select_backend(self, request: InferenceRequest | None = None) -> str:
        request = request or InferenceRequest()
        preferred = request.preferred_backend

        if preferred is not None:
            preferred = preferred.lower().strip()
            if preferred not in self._backends:
                raise InferenceRouterError(
                    f"Unknown inference backend {preferred!r}. "
                    f"Available: {', '.join(self._backends)}"
                )
            if self._backends[preferred].available:
                return preferred
            if not request.allow_fallback:
                raise InferenceRouterError(
                    f"Requested backend {preferred!r} is unavailable."
                )

        for name in self._PRIORITY:
            if self._backends[name].available:
                return name

        raise InferenceRouterError("No inference backend is available.")

    def device_for(self, backend: str) -> torch.device:
        backend = backend.lower()
        if backend == "cuda":
            return torch.device("cuda")
        if backend == "mps":
            return torch.device("mps")
        if backend == "cpu":
            return torch.device("cpu")
        raise InferenceRouterError(f"Unknown inference backend: {backend!r}")

    def run(
        self,
        inference_fn: Callable[[torch.device], Mapping[str, Any]],
        *,
        request: InferenceRequest | None = None,
    ) -> dict[str, Any]:
        """Run inference and attach backend/timing metadata to the result."""
        backend = self.select_backend(request)
        device = self.device_for(backend)
        start = perf_counter()
        try:
            result = dict(inference_fn(device))
        except Exception as exc:
            if request and request.allow_fallback and backend != "cpu":
                start = perf_counter()
                try:
                    result = dict(inference_fn(torch.device("cpu")))
                    backend = "cpu"
                    device = torch.device("cpu")
                except Exception as cpu_exc:
                    raise InferenceRouterError(
                        f"Inference failed on {backend} and CPU fallback also failed: {cpu_exc}"
                    ) from cpu_exc
            else:
                raise InferenceRouterError(
                    f"Inference failed on backend {backend}: {exc}"
                ) from exc

        elapsed_ms = (perf_counter() - start) * 1000.0
        result.setdefault("inference_backend", backend)
        result.setdefault("device", str(device))
        result.setdefault("inference_time_ms", round(elapsed_ms, 2))
        return result
