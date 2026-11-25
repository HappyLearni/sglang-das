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


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="Requires HCU/ROCm GPU",
)
@pytest.mark.parametrize(
    "shape",
    [
        (4096, 8, 512),
        (4096, 16, 512),
        (4096, 8, 4096),
        (65, 8, 512),  # non-multiple of BLOCK_R=8
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
    q_tri, s_tri = mod.quantize(x)

    assert torch.equal(q_tri, q_ref), (
        f"q mismatch for {shape}: max_diff={(q_tri.int() - q_ref.int()).abs().max().item()}"
    )
    assert torch.equal(s_tri, s_ref), (
        f"scale mismatch for {shape}: max_diff={(s_tri - s_ref).abs().max().item()}"
    )
