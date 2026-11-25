import os
import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.layernorm import mhc_boundary_hip as mod
from sglang.srt.utils import is_hcu, is_hip


class TestMhcBoundaryHipBf16x2Cache(unittest.TestCase):
    def setUp(self):
        mod._HC_FN_BF16X2_CACHE.clear()

    def tearDown(self):
        mod._HC_FN_BF16X2_CACHE.clear()

    def test_bf16x2_cache_reuses_and_invalidates_on_mutation(self):
        w = torch.randn((24, 256), dtype=torch.float32) * 0.05
        hi1, lo1 = mod._get_hc_fn_bf16x2(w)
        hi2, lo2 = mod._get_hc_fn_bf16x2(w)
        self.assertIs(hi1, hi2)
        self.assertIs(lo1, lo2)
        recon = hi1.float() + lo1.float()
        self.assertLess((recon - w).abs().max().item(), 1e-5)

        w.mul_(2.0)
        hi3, lo3 = mod._get_hc_fn_bf16x2(w)
        self.assertIsNot(hi1, hi3)
        self.assertLess(((hi3.float() + lo3.float()) - w).abs().max().item(), 1e-5)

    def test_env_flag_gates_bf16x2(self):
        with (
            patch.object(mod, "_is_hcu", True),
            patch.object(mod, "is_batch_invariant_mode_enabled", return_value=False),
            patch.dict(os.environ, {"SGLANG_HCU_MHC_BOUNDARY_BF16X2": "0"}, clear=False),
        ):
            self.assertFalse(mod._hcu_hc_boundary_bf16x2_enabled())
        with (
            patch.object(mod, "_is_hcu", True),
            patch.object(mod, "is_batch_invariant_mode_enabled", return_value=False),
            patch.dict(os.environ, {"SGLANG_HCU_MHC_BOUNDARY_BF16X2": "1"}, clear=False),
        ):
            self.assertTrue(mod._hcu_hc_boundary_bf16x2_enabled())


@unittest.skipUnless(
    torch.cuda.is_available() and is_hip() and is_hcu(),
    "HCU GPU required",
)
class TestMhcBoundaryHipBf16x2Device(unittest.TestCase):
    def setUp(self):
        mod._HC_FN_BF16X2_CACHE.clear()

    def test_accuracy_and_hip_graph_replay(self):
        device = torch.device("cuda")
        gen = torch.Generator(device=device).manual_seed(7)
        hc_fn = torch.randn((24, 20480), device=device, generator=gen, dtype=torch.float32) * 0.02
        hc_scale = torch.ones((3,), device=device, dtype=torch.float32)
        hc_base = torch.zeros((24,), device=device, dtype=torch.float32)

        for m in (6, 16, 48, 64):
            x = torch.randn((m, 5120), device=device, generator=gen).to(torch.bfloat16)
            res = torch.randn((m, 4, 5120), device=device, generator=gen).to(torch.bfloat16)
            post = torch.rand((m, 4), device=device, generator=gen, dtype=torch.float32)
            comb = torch.rand((m, 4, 4), device=device, generator=gen, dtype=torch.float32)
            pre = torch.rand((m, 4), device=device, generator=gen, dtype=torch.float32)

            with patch.dict(
                os.environ, {"SGLANG_HCU_MHC_BOUNDARY_BF16X2": "0"}, clear=False
            ):
                ref_ro, ref_y, ref_c = mod.hc_boundary_fused_deferred(
                    x, res, post, comb, pre, hc_fn, hc_scale, hc_base, 4, 20, 1e-6, 1e-6
                )
            with patch.dict(
                os.environ, {"SGLANG_HCU_MHC_BOUNDARY_BF16X2": "1"}, clear=False
            ):
                out_ro, out_y, out_c = mod.hc_boundary_fused_deferred(
                    x, res, post, comb, pre, hc_fn, hc_scale, hc_base, 4, 20, 1e-6, 1e-6
                )

            self.assertTrue(torch.equal(out_ro, ref_ro), f"M={m} residual_out mismatch")
            self.assertTrue(torch.equal(out_y, ref_y), f"M={m} y mismatch")
            self.assertTrue(torch.equal(out_c.part_sq, ref_c.part_sq), f"M={m} sq mismatch")
            mix_err = (out_c.part_mix.sum(0) - ref_c.part_mix.sum(0)).abs().max().item()
            self.assertLess(mix_err, 1e-4, f"M={m} mix_err={mix_err}")

        # Verify HIP Graph capture + replay uses cached bf16x2 weights without allocating.
        x = torch.randn((16, 5120), device=device, generator=gen).to(torch.bfloat16)
        res = torch.randn((16, 4, 5120), device=device, generator=gen).to(torch.bfloat16)
        post = torch.rand((16, 4), device=device, generator=gen, dtype=torch.float32)
        comb = torch.rand((16, 4, 4), device=device, generator=gen, dtype=torch.float32)
        pre = torch.rand((16, 4), device=device, generator=gen, dtype=torch.float32)
        with patch.dict(
            os.environ, {"SGLANG_HCU_MHC_BOUNDARY_BF16X2": "1"}, clear=False
        ):
            ref_ro, ref_y, ref_c = mod.hc_boundary_fused_deferred(
                x, res, post, comb, pre, hc_fn, hc_scale, hc_base, 4, 20, 1e-6, 1e-6
            )
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                g_ro, g_y, g_c = mod.hc_boundary_fused_deferred(
                    x, res, post, comb, pre, hc_fn, hc_scale, hc_base, 4, 20, 1e-6, 1e-6
                )
            g.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(g_ro, ref_ro))
            self.assertTrue(torch.equal(g_y, ref_y))
            self.assertTrue(torch.equal(g_c.part_mix, ref_c.part_mix))


if __name__ == "__main__":
    unittest.main()
