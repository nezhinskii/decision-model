"""Bridge layout/state contract plus real FLA kernel parity when CUDA is available."""
import inspect
from types import SimpleNamespace
import pytest
import torch

from src.native_lm import kernels


class CUDAView:
    """Exercise CUDA dispatch using CPU arithmetic in the contract test."""
    device = SimpleNamespace(type="cuda")

    def __init__(self, tensor):
        self.tensor = tensor

    def transpose(self, *axes):
        return self.tensor.transpose(*axes)


@pytest.mark.parametrize("real_cuda", [False, True])
def test_fla_bridge_preserves_outputs_and_state(monkeypatch, real_cuda):
    if real_cuda and not torch.cuda.is_available():
        pytest.skip("CUDA required for Triton parity")
    import fla.modules.causal_conv1d as fla
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    reference = inspect.unwrap(qwen.causal_conv1d_fn)
    reference_update = inspect.unwrap(qwen.causal_conv1d_update)
    if not real_cuda:
        def convolution(x, weight, bias, activation, backend):
            assert backend == "triton"
            return reference(x.transpose(1, 2), weight, bias, activation).transpose(1, 2), None
        def update(x, cache, weight, bias, activation):
            return reference_update(x.transpose(1, 2), cache, weight, bias, activation).transpose(1, 2), cache
        monkeypatch.setattr(fla, "causal_conv1d", convolution)
        monkeypatch.setattr(fla, "causal_conv1d_update", update)
    module = SimpleNamespace(__name__="test_qwen3_5", causal_conv1d_fn=reference,
                             causal_conv1d_update=reference_update)
    monkeypatch.setattr(kernels, "importlib", SimpleNamespace(import_module=lambda _: module))
    monkeypatch.setattr(kernels, "_is_reference", lambda f: not hasattr(f, "kernel_provider"))
    model = SimpleNamespace(get_input_embeddings=lambda: SimpleNamespace(
        weight=SimpleNamespace(device=SimpleNamespace(type="cuda"))))
    kernels.install_qwen_convolutions(model)
    installed = module.causal_conv1d_update
    kernels.install_qwen_convolutions(model)
    assert module.causal_conv1d_update is installed
    device, dtype = ("cuda", torch.bfloat16) if real_cuda else ("cpu", torch.float32)
    torch.manual_seed(4)
    weight = torch.randn(32, 4, device=device, dtype=dtype)
    bias = torch.randn(32, device=device, dtype=dtype)
    wrap = (lambda x: x) if real_cuda else CUDAView
    tolerance = dict(atol=.04, rtol=.04) if real_cuda else dict(atol=1e-6, rtol=1e-5)
    for length in (1, 2, 17):
        x = torch.randn(1, 32, length, device=device, dtype=dtype)
        actual = module.causal_conv1d_fn(wrap(x), weight, bias, activation="silu")
        torch.testing.assert_close(actual, reference(x, weight, bias, activation="silu"), **tolerance)
    state = torch.randn(1, 32, 4, device=device, dtype=dtype)
    expected_state = state.clone()
    for _ in range(5):
        x = torch.randn(1, 32, 1, device=device, dtype=dtype)
        actual = module.causal_conv1d_update(wrap(x), state, weight, bias, "silu")
        expected = reference_update(x, expected_state, weight, bias, "silu")
        torch.testing.assert_close(state, expected_state)
        torch.testing.assert_close(actual, expected, **tolerance)


def test_reference_detection_preserves_existing_optimized_dispatch():
    def dispatcher(optimized):
        is_new_implementation = optimized
        def function():
            return is_new_implementation
        return function
    assert kernels._is_reference(dispatcher(False))
    assert not kernels._is_reference(dispatcher(True))
    assert not kernels._is_reference(lambda: None)
