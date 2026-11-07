import time

import pytest
import torch

from model_core.ops import _ema_simple, _ema_simple_reference


def _cases(dtype=torch.float32):
    torch.manual_seed(20260903)
    return {
        "random": torch.randn(3, 4096, dtype=dtype),
        "constant": torch.full((2, 4096), 3.25, dtype=dtype),
        "trend": torch.linspace(-5.0, 5.0, 4096, dtype=dtype).repeat(2, 1),
        "alternating": (((torch.arange(4096) % 2) * 2 - 1).to(dtype)).repeat(2, 1),
        "tiny": torch.randn(2, 4096, dtype=dtype) * 1e-7,
        "large": torch.randn(2, 4096, dtype=dtype) * 1e7,
    }


@pytest.mark.parametrize("span", [5, 20])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_compiled_cpu_ema_matches_upstream_reference(span, dtype):
    atol = 2e-7 if dtype == torch.float32 else 2e-14
    for name, x in _cases(dtype).items():
        ref = _ema_simple_reference(x, span)
        got = _ema_simple(x, span)
        diff = (got - ref).abs().max().item()
        assert diff <= atol, f"{name} span={span} dtype={dtype}: max diff={diff}"
        assert got.shape == x.shape
        assert got.dtype == x.dtype
        assert got.device == x.device
        assert torch.equal(got[:, 0], x[:, 0])


def test_empty_and_singleton_contract():
    empty = torch.empty(2, 0, dtype=torch.float32)
    assert _ema_simple(empty, 5).shape == (2, 0)
    one = torch.tensor([[2.5], [-3.0]], dtype=torch.float32)
    assert torch.equal(_ema_simple(one, 20), one)


def test_autograd_tensor_uses_reference_semantics():
    x = torch.randn(1, 32, requires_grad=True)
    got = _ema_simple(x, 5)
    ref = _ema_simple_reference(x, 5)
    assert torch.equal(got, ref)
    got.sum().backward()
    assert x.grad is not None


def test_long_cpu_backend_removes_python_runtime_tail():
    # Guard against accidental return to the ~6s Python recurrence.  This is a
    # generous threshold so normal CI variance does not make the test flaky.
    x = torch.randn(1, 100_000, dtype=torch.float32)
    _ema_simple(x[:, :100], 5)  # warm import/cache
    t0 = time.perf_counter()
    _ema_simple(x, 5)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.5, f"compiled EMA unexpectedly slow: {elapsed:.3f}s"
