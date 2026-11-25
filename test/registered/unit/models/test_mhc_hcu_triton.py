"""Unit tests for the HCU single-pass Triton MFMA BF16x3 mHC mix_stats kernel."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.layernorm import mhc_hcu_triton
from sglang.srt.models import deepseek_v4_mhc as mhc
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _reference_mix_stats(
    x_flat: torch.Tensor,
    fn_w: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    sinkhorn_iters: int = 20,
    rms_eps: float = 1e-20,
    hc_eps: float = 1e-6,
):
    xf = x_flat.float()
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + rms_eps)
    mix = torch.nn.functional.linear(xf, fn_w.float()) * rsqrt
    pre = torch.sigmoid(mix[:, :4] * hc_scale[0] + hc_base[:4]) + hc_eps
    post = 2.0 * torch.sigmoid(mix[:, 4:8] * hc_scale[1] + hc_base[4:8])
    comb = (
        mix[:, 8:].view(x_flat.shape[0], 4, 4) * hc_scale[2]
        + hc_base[8:].view(1, 4, 4)
    )
    comb = torch.exp(comb - comb.amax(dim=-1, keepdim=True))
    if sinkhorn_iters > 0:
        comb = comb / comb.sum(dim=-1, keepdim=True) + hc_eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
        for _ in range(sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    return pre, post, comb


def _split_bf16_parts(fn_w: torch.Tensor):
    w0 = fn_w.to(torch.bfloat16)
    rem1 = fn_w - w0.float()
    w1 = rem1.to(torch.bfloat16)
    w2 = (rem1 - w1.float()).to(torch.bfloat16)
    return (w0.contiguous(), w1.contiguous(), w2.contiguous())


class TestMhcHcuTriton(CustomTestCase):
    def tearDown(self):
        mhc_hcu_triton._weight_cache.clear()

    def test_launch_config_selection(self):
        self.assertEqual(mhc_hcu_triton._select_launch_config(1024, 20480), (128, 64, 8, 8))
        self.assertEqual(mhc_hcu_triton._select_launch_config(1536, 20480), (128, 64, 8, 8))
        self.assertEqual(mhc_hcu_triton._select_launch_config(4096, 20480), (128, 64, 8, 8))
        self.assertEqual(mhc_hcu_triton._select_launch_config(1024, 512), (128, 64, 8, 8))
        self.assertEqual(mhc_hcu_triton._select_launch_config(1024, 256), (64, 64, 4, 4))
        self.assertIsNone(mhc_hcu_triton._select_launch_config(1024, 100))

    def test_padded_weight_cache_and_version_invalidation(self):
        fn_w = torch.randn(24, 512, dtype=torch.float32) * 0.02
        parts = _split_bf16_parts(fn_w)
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=False):
            w1 = mhc_hcu_triton._get_padded_bf16x3_weight(parts)
            w2 = mhc_hcu_triton._get_padded_bf16x3_weight(parts)
            self.assertIs(w1, w2)
            self.assertEqual(tuple(w1.shape), (512, 128))
            self.assertTrue(torch.equal(w1[:, :24], parts[0].t()))
            self.assertTrue(torch.equal(w1[:, 24:48], parts[1].t()))
            self.assertTrue(torch.equal(w1[:, 48:72], parts[2].t()))
            self.assertTrue(torch.all(w1[:, 72:] == 0))
            parts[0].mul_(2.0)
            w3 = mhc_hcu_triton._get_padded_bf16x3_weight(parts)
            self.assertIsNot(w1, w3)

    def test_dispatch_gate_respects_threshold_and_batch_invariance(self):
        cfg = mhc.HcConfig(
            mult=4,
            sinkhorn_iters=20,
            eps=1e-6,
            rms_eps=1e-20,
            hidden=128,
            pre_from_prev=True,
            cp_prefill=False,
        )
        fn = torch.randn(24, 512, dtype=torch.float32) * 0.02
        parts = _split_bf16_parts(fn)
        hc = mhc.HcSubLayer(
            cfg=cfg,
            fn=fn,
            scale=torch.ones(3, dtype=torch.float32) * 0.1,
            base=torch.zeros(24, dtype=torch.float32),
            norm=SimpleNamespace(),
            bf16_parts=parts,
        )
        sentinel = (torch.zeros(1), torch.zeros(1), torch.zeros(1))
        x_big = torch.randn(1536, 4, 128, dtype=torch.bfloat16)
        with (
            patch.object(mhc, "_use_hcu_hc_mix_stats_triton", True),
            patch.object(mhc, "_use_hcu_hc_mix_stats", False),
            patch.object(mhc, "_is_gfx95_supported", False),
            patch.object(mhc, "get_platform", return_value=SimpleNamespace(is_sm90=False, is_sm100=False, is_blackwell=False)),
            patch(
                "sglang.srt.batch_invariant_ops.is_batch_invariant_mode_enabled",
                return_value=False,
            ),
            patch.object(
                mhc_hcu_triton,
                "can_use_hc_mix_stats_sinkhorn_bf16x3_hcu",
                return_value=True,
            ),
            patch.object(
                mhc_hcu_triton,
                "hc_mix_stats_sinkhorn_bf16x3_hcu",
                return_value=sentinel,
            ) as mock_kernel,
        ):
            out = mhc._mix_stats_impl(hc, x_big)
            self.assertIs(out, sentinel)
            mock_kernel.assert_called_once()

            # Below threshold (including 768 and 1024) must NOT call Triton kernel.
            for small_m in (256, 768, 1024, 1535):
                mock_kernel.reset_mock()
                with patch(
                    "sglang.kernels.ops.layernorm.mhc.hc_split_sinkhorn",
                    return_value=(torch.zeros(1, 1), torch.zeros(1, 1), torch.zeros(1, 1)),
                ):
                    mhc._mix_stats_impl(
                        hc, torch.randn(small_m, 4, 128, dtype=torch.bfloat16)
                    )
                mock_kernel.assert_not_called()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA/HIP device required")
    def test_gpu_accuracy_matches_fp32_reference(self):
        device = torch.device("cuda:0")
        torch.manual_seed(0)
        k = 20480
        fn_w = torch.randn(24, k, device=device, dtype=torch.float32) * 0.02
        hc_scale = torch.ones(3, device=device, dtype=torch.float32) * 0.1
        hc_base = torch.randn(24, device=device, dtype=torch.float32) * 0.05
        parts = _split_bf16_parts(fn_w)
        for m in (769, 1024, 1536, 2048, 3072, 4096, 4097):
            x = torch.randn(m, k, device=device, dtype=torch.bfloat16) * 0.5
            pre_ref, post_ref, comb_ref = _reference_mix_stats(
                x, fn_w, hc_scale, hc_base
            )
            pre, post, comb = mhc_hcu_triton.hc_mix_stats_sinkhorn_bf16x3_hcu(
                x, parts, hc_scale, hc_base, 20, 1e-20, 1e-6
            )
            self.assertEqual(pre.shape, (m, 4))
            self.assertEqual(post.shape, (m, 4))
            self.assertEqual(comb.shape, (m, 4, 4))
            self.assertLess((pre - pre_ref).abs().max().item(), 5e-6)
            self.assertLess((post - post_ref).abs().max().item(), 5e-6)
            self.assertLess((comb - comb_ref).abs().max().item(), 5e-6)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA/HIP device required")
    def test_gpu_high_logit_sinkhorn_stability_no_nan(self):
        """Regression test: comb logits > 88.7 (e.g., 160.0) must not overflow to NaN."""
        device = torch.device("cuda:0")
        torch.manual_seed(7)
        k = 20480
        m = 4096
        fn_w = torch.randn(24, k, device=device, dtype=torch.float32) * 0.02
        hc_scale = torch.tensor([0.1, 0.1, 1.5], device=device, dtype=torch.float32)
        hc_base = torch.randn(24, device=device, dtype=torch.float32) * 0.05
        # Force logits well beyond fp32 exp overflow (~88.72) and underflow (-103.97).
        hc_base[8] = 160.0
        hc_base[13] = -120.0
        hc_base[18] = 148.9
        parts = _split_bf16_parts(fn_w)
        x = torch.randn(m, k, device=device, dtype=torch.bfloat16) * 2.0

        pre_ref, post_ref, comb_ref = _reference_mix_stats(
            x, fn_w, hc_scale, hc_base
        )
        pre, post, comb = mhc_hcu_triton.hc_mix_stats_sinkhorn_bf16x3_hcu(
            x, parts, hc_scale, hc_base, 20, 1e-20, 1e-6
        )
        self.assertTrue(torch.isfinite(pre).all())
        self.assertTrue(torch.isfinite(post).all())
        self.assertTrue(torch.isfinite(comb).all())
        self.assertLess((pre - pre_ref).abs().max().item(), 5e-6)
        self.assertLess((post - post_ref).abs().max().item(), 5e-6)
        self.assertLess((comb - comb_ref).abs().max().item(), 5e-6)

        try:
            import lightop
        except ImportError:
            lightop = None
        if lightop is not None and hasattr(lightop, "hc_mix_stats_sinkhorn_bf16x3"):
            pre_lo, post_lo, comb_lo = lightop.hc_mix_stats_sinkhorn_bf16x3(
                x,
                parts,
                hc_scale,
                hc_base,
                20,
                1e-20,
                1e-6,
            )
            self.assertLess((pre - pre_lo).abs().max().item(), 5e-6)
            self.assertLess((post - post_lo).abs().max().item(), 5e-6)
            self.assertLess((comb - comb_lo).abs().max().item(), 5e-6)


if __name__ == "__main__":
    unittest.main()
