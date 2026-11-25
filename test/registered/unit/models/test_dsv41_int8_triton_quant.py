import importlib

import pytest
import torch


def _import_dsv41_int8():
    return importlib.import_module("sglang.kernels.ops.gemm.dsv41_int8")


def test_triton_quantize_gate_and_block_r(monkeypatch):
    mod = _import_dsv41_int8()

    assert mod._select_triton_quant_block_r(128) == 8
    assert mod._select_triton_quant_block_r(512) == 8
    assert mod._select_triton_quant_block_r(768) == 4
    assert mod._select_triton_quant_block_r(1024) == 4
    assert mod._select_triton_quant_block_r(4096) == 1

    monkeypatch.setenv("SGLANG_DSV41_INT8_TRITON_QUANT", "0")
    assert mod._triton_quantize_enabled() is False

    monkeypatch.setenv("SGLANG_DSV41_INT8_TRITON_QUANT", "1")
    assert mod._triton_quantize_enabled() is True

    # CPU tensor must never take the GPU Triton path.
    cpu_x = torch.randn((512, 512), dtype=torch.bfloat16)
    assert mod._can_use_triton_quantize(cpu_x) is False


    # Below threshold (e.g. decode or small prefill) must stay on native C++.
    if torch.cuda.is_available() and torch.version.hip is not None:
        small_gpu = torch.empty((512, 8, 4096), dtype=torch.bfloat16, device="cuda")
        big_gpu = torch.empty((1024, 8, 4096), dtype=torch.bfloat16, device="cuda")
        assert mod._can_use_triton_quantize(small_gpu) is False
        assert mod._can_use_triton_quantize(big_gpu) is True


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="Requires HCU/ROCm GPU",
)
@pytest.mark.parametrize(
    "shape",
    [
        (1024, 8, 4096),
        (4096, 8, 4096),
        (65, 8, 512),  # non-multiple of BLOCK_R=8 (direct kernel check)
        (131, 4, 768),  # non-power-of-2 K and non-multiple of BLOCK_R=4
    ],
)
def test_triton_quantize_bitwise_exact_vs_native(shape, monkeypatch):
    mod = _import_dsv41_int8()
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=torch.bfloat16, device="cuda") * 3.5
    # Include an all-zero row and a tiny-magnitude row to verify scale floor.
    x[0, 0].zero_()
    x[1, 0].fill_(1e-8)

    monkeypatch.setenv("SGLANG_DSV41_INT8_TRITON_QUANT", "0")
    q_ref, s_ref = mod.quantize(x)

    monkeypatch.setenv("SGLANG_DSV41_INT8_TRITON_QUANT", "1")
    if x.numel() >= mod._TRITON_QUANT_MIN_ELEMENTS:
        q_tri, s_tri = mod.quantize(x)
    else:
        q_tri = torch.empty_like(q_ref)
        s_tri = torch.empty_like(s_ref)
        mod._quantize_triton(x, q_tri, s_tri)

    assert torch.equal(q_tri, q_ref), (
        f"q mismatch for {shape}: max_diff={(q_tri.int() - q_ref.int()).abs().max().item()}"
    )
    assert torch.equal(s_tri, s_ref), (
        f"scale mismatch for {shape}: max_diff={(s_tri - s_ref).abs().max().item()}"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="Requires HCU/ROCm GPU",
)
def test_hip_graph_capture_across_row_count_change_zero_recompile(monkeypatch):
    """Regression test: warming up (1024, 8, 4096) and capturing (1025, 8, 4096)
    must not trigger a new Triton M-divisibility specialization during capture."""
    mod = _import_dsv41_int8()
    mod._TRITON_QUANT_WARMED.clear()
    monkeypatch.setenv("SGLANG_DSV41_INT8_TRITON_QUANT", "1")

    def _triton_cache_entries():
        fn = mod._dsv41_int8_quantize_triton_kernel
        caches = getattr(fn, "device_caches", None)
        if caches is not None:
            return sum(len(v[0]) for v in caches.values())
        return len(getattr(fn, "cache", {}))

    # Warm up M=1024 (m=8192, divisible by 16).
    x_warm = torch.randn((1024, 8, 4096), dtype=torch.bfloat16, device="cuda")
    mod.quantize(x_warm)
    torch.cuda.synchronize()
    cached_after_warm = _triton_cache_entries()

    # Capture M=1025 (m=8200, not divisible by 16) inside HIP Graph.
    x_cap = torch.randn((1025, 8, 4096), dtype=torch.bfloat16, device="cuda") * 2.0
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        q_cap, s_cap = mod.quantize(x_cap)
    g.replay()
    torch.cuda.synchronize()

    assert _triton_cache_entries() == cached_after_warm, (
        "Unexpected Triton recompilation triggered during HIP Graph capture"
    )

    monkeypatch.setenv("SGLANG_DSV41_INT8_TRITON_QUANT", "0")
    q_ref, s_ref = mod.quantize(x_cap)
    assert torch.equal(q_cap, q_ref)
    assert torch.equal(s_cap, s_ref)
