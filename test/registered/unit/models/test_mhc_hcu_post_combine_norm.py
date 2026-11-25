import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.layernorm import mhc_hcu_post_combine_norm as mod
from sglang.srt.utils import is_hcu, is_hip


class TestMhcHcuPostCombineNormDispatch(unittest.TestCase):
    def _make_cfg(self, *, cp_prefill: bool = True, pre_from_prev: bool = True):
        return SimpleNamespace(
            pre_from_prev=pre_from_prev,
            hidden=5120,
            mult=4,
            cp_prefill=cp_prefill,
        )

    def _make_fake_tensor(self, shape, *, dtype=torch.bfloat16, is_cuda=True, stride=(-1, 1)):
        t = SimpleNamespace(
            shape=shape,
            dtype=dtype,
            is_cuda=is_cuda,
            stride=lambda dim: stride[dim],
        )
        return t

    def test_gate_allows_cp_prefill_and_respects_bounds_and_env(self):
        cfg = self._make_cfg(cp_prefill=True)
        next_ok = SimpleNamespace(norm_fusable=True)
        next_no = SimpleNamespace(norm_fusable=False)
        y_ok = self._make_fake_tensor((4096, 5120))
        r_ok = self._make_fake_tensor((4096, 4, 5120), stride=(20480, 5120, 1))
        y_small = self._make_fake_tensor((64, 5120))
        r_small = self._make_fake_tensor((64, 4, 5120), stride=(20480, 5120, 1))

        with (
            patch.object(mod, "is_hcu", return_value=True),
            patch.object(mod, "is_batch_invariant_mode_enabled", return_value=False),
            patch.dict(
                os.environ, {"SGLANG_HCU_FUSED_MHC_POST_COMBINE_NORM": "1"}, clear=False
            ),
        ):
            self.assertTrue(
                mod.can_use_hcu_mhc_post_combine_norm(cfg, y_ok, r_ok, next_ok)
            )
            self.assertFalse(
                mod.can_use_hcu_mhc_post_combine_norm(cfg, y_ok, r_ok, next_no)
            )
            self.assertFalse(
                mod.can_use_hcu_mhc_post_combine_norm(cfg, y_small, r_small, next_ok)
            )

        # Batch-invariant mode must disable fused dispatch across threshold boundaries.
        with (
            patch.object(mod, "is_hcu", return_value=True),
            patch.object(mod, "is_batch_invariant_mode_enabled", return_value=True),
            patch.dict(
                os.environ, {"SGLANG_HCU_FUSED_MHC_POST_COMBINE_NORM": "1"}, clear=False
            ),
        ):
            self.assertFalse(
                mod.can_use_hcu_mhc_post_combine_norm(cfg, y_ok, r_ok, next_ok)
            )

        with (
            patch.object(mod, "is_hcu", return_value=True),
            patch.object(mod, "is_batch_invariant_mode_enabled", return_value=False),
            patch.dict(
                os.environ, {"SGLANG_HCU_FUSED_MHC_POST_COMBINE_NORM": "0"}, clear=False
            ),
        ):
            self.assertFalse(
                mod.can_use_hcu_mhc_post_combine_norm(cfg, y_ok, r_ok, next_ok)
            )


@unittest.skipUnless(
    torch.cuda.is_available() and is_hip() and is_hcu(),
    "HCU GPU required",
)
class TestMhcHcuPostCombineNormDevice(unittest.TestCase):
    def _ref_bf16_staged(self, x, residual, post, comb, pre, weight, eps):
        # Match unfused bf16-staged path: mhc_post -> bf16 -> hc_combine -> bf16 -> rms_norm
        post_f = post.float().unsqueeze(-1)
        comb_f = comb.float().unsqueeze(-1)
        updated = (
            post_f * x.float().unsqueeze(1)
            + (comb_f * residual.float().unsqueeze(2)).sum(dim=1)
        ).to(torch.bfloat16)
        combined = (pre.float().unsqueeze(-1) * updated.float()).sum(dim=1).to(
            torch.bfloat16
        )
        h = combined.float()
        inv_rms = torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + eps)
        normed = (h * inv_rms * weight.float()).to(torch.bfloat16)
        return updated, normed

    def test_accuracy_across_prefill_shapes(self):
        device = torch.device("cuda")
        gen = torch.Generator(device=device).manual_seed(123)
        weight = (
            torch.randn((5120,), device=device, generator=gen) * 0.25 + 1.0
        ).to(torch.bfloat16)
        eps = 1e-6

        for m in (512, 1023, 2048, 4096):
            x = (torch.randn((m, 5120), device=device, generator=gen) * 0.5).to(
                torch.bfloat16
            )
            residual = (
                torch.randn((m, 4, 5120), device=device, generator=gen) * 0.5
            ).to(torch.bfloat16)
            post = torch.rand((m, 4), device=device, generator=gen, dtype=torch.float32)
            comb = torch.softmax(
                torch.randn((m, 4, 4), device=device, generator=gen, dtype=torch.float32),
                dim=-1,
            )
            pre = torch.rand((m, 4), device=device, generator=gen, dtype=torch.float32)

            ref_r, ref_y = self._ref_bf16_staged(x, residual, post, comb, pre, weight, eps)
            out_r, out_y = mod.mhc_post_combine_norm_hcu(
                x, residual, post, comb, pre, weight, eps
            )

            self.assertEqual(out_r.shape, (m, 4, 5120))
            self.assertEqual(out_y.shape, (m, 5120))
            self.assertEqual(out_r.dtype, torch.bfloat16)
            self.assertEqual(out_y.dtype, torch.bfloat16)
            r_err = (out_r.float() - ref_r.float()).abs().max().item()
            y_err = (out_y.float() - ref_y.float()).abs().max().item()
            self.assertLessEqual(r_err, 1.6e-2, f"M={m} residual max err={r_err}")
            self.assertLessEqual(y_err, 3.2e-2, f"M={m} norm max err={y_err}")

    def test_batch_invariant_mode_is_bitwise_identical_across_511_to_512_threshold(self):
        """Regression test: expanding 511 -> 512 rows under batch-invariant mode
        must keep the first 511 rows bitwise identical."""
        from sglang.srt.layers.layernorm import RMSNorm
        from sglang.srt.models import deepseek_v4_mhc as mhc

        device = torch.device("cuda")
        gen = torch.Generator(device=device).manual_seed(77)
        norm = RMSNorm(5120, eps=1e-6).to(device=device, dtype=torch.bfloat16)
        norm.weight.data.copy_(
            (torch.randn((5120,), device=device, generator=gen) * 0.25 + 1.0).to(
                torch.bfloat16
            )
        )
        cfg = mhc.HcConfig(
            mult=4,
            sinkhorn_iters=20,
            eps=1e-6,
            rms_eps=1e-20,
            hidden=5120,
            pre_from_prev=True,
            cp_prefill=True,
        )
        hc = mhc.HcSubLayer(
            cfg=cfg,
            fn=torch.empty(0, device=device),
            scale=torch.empty(0, device=device),
            base=torch.empty(0, device=device),
            norm=norm,
        )
        next_b = mhc.make_boundary(norm, accepts_mxfp8=False)

        x512 = (torch.randn((512, 5120), device=device, generator=gen) * 0.5).to(
            torch.bfloat16
        )
        res512 = (torch.randn((512, 4, 5120), device=device, generator=gen) * 0.5).to(
            torch.bfloat16
        )
        post512 = torch.rand((512, 4), device=device, generator=gen, dtype=torch.float32)
        comb512 = torch.softmax(
            torch.randn((512, 4, 4), device=device, generator=gen, dtype=torch.float32),
            dim=-1,
        )
        pre512 = torch.rand((512, 4), device=device, generator=gen, dtype=torch.float32)

        with (
            patch.object(mod, "is_batch_invariant_mode_enabled", return_value=True),
            patch(
                "sglang.srt.batch_invariant_ops.is_batch_invariant_mode_enabled",
                return_value=True,
            ),
            patch(
                "sglang.kernels.ops.layernorm.mhc.is_dsa_prefill_cp_interleave",
                return_value=False,
            ),
            patch.dict(
                os.environ, {"SGLANG_HCU_FUSED_MHC_POST_COMBINE_NORM": "1"}, clear=False
            ),
        ):
            st511 = mhc._post_fusion(
                hc,
                x512[:511],
                res512[:511],
                (pre512[:511], post512[:511], comb512[:511]),
                next_b,
            )
            st512 = mhc._post_fusion(
                hc,
                x512,
                res512,
                (pre512, post512, comb512),
                next_b,
            )
            # Neither should take the fused path under batch-invariant mode.
            self.assertIsNone(st511.input)
            self.assertIsNone(st512.input)
            y511 = mhc.combine(hc, st511)
            y512 = mhc.combine(hc, st512)
            self.assertTrue(torch.equal(st511.residual, st512.residual[:511]))
            self.assertTrue(torch.equal(y511, y512[:511]))


if __name__ == "__main__":
    unittest.main()
