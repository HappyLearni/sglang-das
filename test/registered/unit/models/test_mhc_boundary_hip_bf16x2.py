import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.layernorm import mhc_boundary_hip as mod
from sglang.srt.utils import is_hcu, is_hip


def _make_direct_weight_updater(model: torch.nn.Module, device: str):
    from sglang.srt.model_executor.model_runner_components.weight_updater import (
        WeightUpdater,
    )

    return WeightUpdater(
        tp_rank=0,
        device=device,
        gpu_id=0,
        model_config=SimpleNamespace(),
        custom_weight_loaders={},
        get_model=lambda: model,
        update_model_fields=lambda **_: None,
        recapture_cuda_graph=lambda: None,
        get_model_runner=lambda: SimpleNamespace(),
    )


class TestMhcBoundaryHipBf16x2Cache(unittest.TestCase):
    def setUp(self):
        mod._HC_FN_BF16X2_CACHE.clear()

    def tearDown(self):
        mod._HC_FN_BF16X2_CACHE.clear()

    def test_bf16x2_cache_reuses_and_refreshes_on_mutation_and_data_copy(self):
        w = torch.nn.Parameter(torch.randn((24, 256), dtype=torch.float32) * 0.05)
        hi1, lo1 = mod._get_hc_fn_bf16x2(w)
        hi2, lo2 = mod._get_hc_fn_bf16x2(w)
        self.assertIs(hi1, hi2)
        self.assertIs(lo1, lo2)
        recon = hi1.float() + lo1.float()
        self.assertLess((recon - w).abs().max().item(), 1e-5)

        # Standard in-place mutation increments _version and updates buffers in place.
        with torch.no_grad():
            w.mul_(2.0)
        hi3, lo3 = mod._get_hc_fn_bf16x2(w)
        self.assertIs(hi1, hi3)
        self.assertLess(((hi3.float() + lo3.float()) - w).abs().max().item(), 1e-5)

        # Online weight reload uses param.data.copy_(), which does NOT bump _version.
        ver_before = w._version
        new_w = torch.randn((24, 256), dtype=torch.float32) * 0.08
        w.data.copy_(new_w)
        self.assertEqual(w._version, ver_before)
        mod.refresh_hc_fn_bf16x2_cache(w)
        hi4, lo4 = mod._get_hc_fn_bf16x2(w)
        self.assertIs(hi1, hi4)
        self.assertIs(lo1, lo4)
        self.assertLess(((hi4.float() + lo4.float()) - new_w).abs().max().item(), 1e-5)

    def test_direct_weight_updater_refreshes_bf16x2_cache_in_place(self):
        from sglang.srt.model_executor.model_runner_components import (
            weight_updater as wu_mod,
        )

        class _TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.hc_attn_fn = torch.nn.Parameter(
                    torch.randn((24, 256), dtype=torch.float32) * 0.05
                )

        model = _TinyModel()
        hi1, lo1 = mod._get_hc_fn_bf16x2(model.hc_attn_fn)
        ver_before = model.hc_attn_fn._version
        new_w = torch.randn((24, 256), dtype=torch.float32) * 0.09
        updater = _make_direct_weight_updater(model, device="cpu")

        with (
            patch.object(wu_mod, "get_parallel", return_value=SimpleNamespace(tp_rank=0)),
            patch.object(
                wu_mod,
                "get_model",
                return_value=SimpleNamespace(weight_cache_mode="off"),
            ),
            patch.object(
                wu_mod, "_unsupported_derived_weight_cache_error", return_value=None
            ),
            patch.object(
                torch,
                "get_device_module",
                return_value=SimpleNamespace(current_device=lambda: torch.device("cpu")),
            ),
        ):
            ok, msg = updater.update_weights_from_tensor(
                [("hc_attn_fn", new_w)], load_format="direct"
            )

        self.assertTrue(ok, msg)
        self.assertEqual(model.hc_attn_fn._version, ver_before)
        self.assertTrue(torch.equal(model.hc_attn_fn.data, new_w))
        hi2, lo2 = mod._get_hc_fn_bf16x2(model.hc_attn_fn)
        self.assertIs(hi1, hi2)
        self.assertIs(lo1, lo2)
        self.assertLess(((hi2.float() + lo2.float()) - new_w).abs().max().item(), 1e-5)

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

        for m in (64, 512, 1024):
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
            if m < mod._HCU_MHC_BOUNDARY_BF16X2_MIN_M:
                self.assertEqual(mix_err, 0.0, f"M={m} below threshold must use FP32 path")
            else:
                self.assertLess(mix_err, 1e-4, f"M={m} mix_err={mix_err}")

        # Verify HIP Graph capture + replay at M >= threshold uses cached bf16x2 weights.
        x = torch.randn((512, 5120), device=device, generator=gen).to(torch.bfloat16)
        res = torch.randn((512, 4, 5120), device=device, generator=gen).to(torch.bfloat16)
        post = torch.rand((512, 4), device=device, generator=gen, dtype=torch.float32)
        comb = torch.rand((512, 4, 4), device=device, generator=gen, dtype=torch.float32)
        pre = torch.rand((512, 4), device=device, generator=gen, dtype=torch.float32)
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

            # Regression test: online weight reload via param.data.copy_() +
            # refresh_hc_fn_bf16x2_cache() must update the ALREADY-CAPTURED HIP
            # Graph's buffers in place with zero recapture.
            hc_param = torch.nn.Parameter(hc_fn.clone())
            mod._HC_FN_BF16X2_CACHE.clear()
            _, _, _ = mod.hc_boundary_fused_deferred(
                x, res, post, comb, pre, hc_param, hc_scale, hc_base, 4, 20, 1e-6, 1e-6
            )
            torch.cuda.synchronize()
            g_reload = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g_reload):
                _, _, g_reload_c = mod.hc_boundary_fused_deferred(
                    x, res, post, comb, pre, hc_param, hc_scale, hc_base, 4, 20, 1e-6, 1e-6
                )
            ver_before = hc_param._version
            new_fn = (
                torch.randn((24, 20480), device=device, generator=gen, dtype=torch.float32)
                * 0.03
            )
            hc_param.data.copy_(new_fn)
            self.assertEqual(hc_param._version, ver_before)
            mod.refresh_hc_fn_bf16x2_cache(hc_param)
            g_reload.replay()
            torch.cuda.synchronize()

            with patch.dict(
                os.environ, {"SGLANG_HCU_MHC_BOUNDARY_BF16X2": "0"}, clear=False
            ):
                _, _, new_ref_c = mod.hc_boundary_fused_deferred(
                    x, res, post, comb, pre, hc_param, hc_scale, hc_base, 4, 20, 1e-6, 1e-6
                )
            reload_err = (
                g_reload_c.part_mix.sum(0) - new_ref_c.part_mix.sum(0)
            ).abs().max().item()
            self.assertLess(reload_err, 1e-4, f"post-reload graph mix_err={reload_err}")

    def test_direct_weight_updater_refreshes_eager_and_captured_hip_graph_at_m512(self):
        from sglang.srt.model_executor.model_runner_components import (
            weight_updater as wu_mod,
        )

        device = torch.device("cuda")
        gen = torch.Generator(device=device).manual_seed(19)

        class _TinyMhcModel(torch.nn.Module):
            def __init__(self, init_fn: torch.Tensor):
                super().__init__()
                self.hc_attn_fn = torch.nn.Parameter(init_fn)

        init_fn = (
            torch.randn((24, 20480), device=device, generator=gen, dtype=torch.float32)
            * 0.02
        )
        new_fn = (
            torch.randn((24, 20480), device=device, generator=gen, dtype=torch.float32)
            * 0.04
        )
        model = _TinyMhcModel(init_fn.clone())
        hc_scale = torch.ones((3,), device=device, dtype=torch.float32)
        hc_base = torch.zeros((24,), device=device, dtype=torch.float32)
        x = torch.randn((512, 5120), device=device, generator=gen).to(torch.bfloat16)
        res = torch.randn((512, 4, 5120), device=device, generator=gen).to(
            torch.bfloat16
        )
        post = torch.rand((512, 4), device=device, generator=gen, dtype=torch.float32)
        comb = torch.rand(
            (512, 4, 4), device=device, generator=gen, dtype=torch.float32
        )
        pre = torch.rand((512, 4), device=device, generator=gen, dtype=torch.float32)

        with patch.dict(
            os.environ, {"SGLANG_HCU_MHC_BOUNDARY_BF16X2": "1"}, clear=False
        ):
            _, _, pre_eager_c = mod.hc_boundary_fused_deferred(
                x, res, post, comb, pre, model.hc_attn_fn, hc_scale, hc_base, 4, 20, 1e-6, 1e-6
            )
            pre_mix = pre_eager_c.part_mix.sum(0).clone()
            torch.cuda.synchronize()

            g_direct = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g_direct):
                _, _, g_direct_c = mod.hc_boundary_fused_deferred(
                    x, res, post, comb, pre, model.hc_attn_fn, hc_scale, hc_base, 4, 20, 1e-6, 1e-6
                )
            torch.cuda.synchronize()

            updater = _make_direct_weight_updater(model, device="cuda")
            ver_before = model.hc_attn_fn._version
            with (
                patch.object(
                    wu_mod, "get_parallel", return_value=SimpleNamespace(tp_rank=0)
                ),
                patch.object(
                    wu_mod,
                    "get_model",
                    return_value=SimpleNamespace(weight_cache_mode="off"),
                ),
                patch.object(
                    wu_mod,
                    "_unsupported_derived_weight_cache_error",
                    return_value=None,
                ),
            ):
                ok, msg = updater.update_weights_from_tensor(
                    [("hc_attn_fn", new_fn)], load_format="direct"
                )
            self.assertTrue(ok, msg)
            self.assertEqual(model.hc_attn_fn._version, ver_before)
            self.assertTrue(torch.equal(model.hc_attn_fn.data, new_fn))

            # Both eager and already-captured HIP Graph replay must reflect new_fn.
            _, _, post_eager_c = mod.hc_boundary_fused_deferred(
                x, res, post, comb, pre, model.hc_attn_fn, hc_scale, hc_base, 4, 20, 1e-6, 1e-6
            )
            g_direct.replay()
            torch.cuda.synchronize()
            eager_mix = post_eager_c.part_mix.sum(0).clone()
            graph_mix = g_direct_c.part_mix.sum(0).clone()

        with patch.dict(
            os.environ, {"SGLANG_HCU_MHC_BOUNDARY_BF16X2": "0"}, clear=False
        ):
            _, _, ref_new_c = mod.hc_boundary_fused_deferred(
                x, res, post, comb, pre, model.hc_attn_fn, hc_scale, hc_base, 4, 20, 1e-6, 1e-6
            )
            ref_new_mix = ref_new_c.part_mix.sum(0)

        self.assertGreater((ref_new_mix - pre_mix).abs().max().item(), 1e-2)
        self.assertLess((eager_mix - ref_new_mix).abs().max().item(), 1e-4)
        self.assertLess((graph_mix - ref_new_mix).abs().max().item(), 1e-4)


if __name__ == "__main__":
    unittest.main()
