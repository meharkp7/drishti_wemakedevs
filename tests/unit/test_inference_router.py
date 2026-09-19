import pytest
import torch

from inference.router import InferenceRequest, InferenceRouter, InferenceRouterError


def test_cpu_is_always_available():
    router = InferenceRouter()
    assert any(info.name == "cpu" and info.available for info in router.backends)


def test_unknown_backend_rejected():
    with pytest.raises(InferenceRouterError):
        InferenceRouter().select_backend(InferenceRequest(preferred_backend="quantum"))


def test_cpu_request_is_honored():
    router = InferenceRouter()
    assert router.select_backend(InferenceRequest(preferred_backend="cpu")) == "cpu"


def test_run_attaches_backend_and_device():
    router = InferenceRouter()
    result = router.run(lambda device: {"value": str(device)}, request=InferenceRequest(preferred_backend="cpu"))
    assert result["value"] == "cpu"
    assert result["inference_backend"] == "cpu"
    assert result["device"] == "cpu"
    assert result["inference_time_ms"] >= 0
