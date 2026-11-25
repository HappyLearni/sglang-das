import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.gemm import router_gemv_hip
from sglang.kernels.ops.moe import rocm_router_gate as rocm_router_gate_mod
from sglang.srt.utils import is_hcu, is_hip


class TestHcuRouterSplitKDispatch(unittest.TestCase):
    def test_rocm_router_max_tokens_opt_in_on_hcu(self):
        with (
            patch.object(router_gemv_hip, "is_hip", return_value=True),
            patch.object(router_gemv_hip, "is_gfx95_supported", return_value=False),
            patch.object(router_gemv_hip, "is_hcu", return_value=True),
            patch.dict(os.environ, {"SGLANG_HCU_ROCM_ROUTER_SPLIT_K": "1"}, clear=False),
        ):
            # Non-router callers (allow_hcu=False) remain disabled on HCU.
            self.assertEqual(
                router_gemv_hip.rocm_gemv_split_k_max_tokens(
                    n=32, k=5120, weight_dtype=torch.bfloat16
                ),
                -1,
            )
            # Router caller (allow_hcu=True via rocm_router_max_tokens) is enabled.
            self.assertEqual(
                rocm_router_gate_mod.rocm_router_max_tokens(
                    num_experts=384,
                    hidden_size=5120,
                    topk=6,
                    weight_dtype=torch.bfloat16,
                ),
                router_gemv_hip.ROCM_ROUTER_MAX_TOKENS,
            )

    def test_rocm_router_max_tokens_env_disable_on_hcu(self):
        with (
            patch.object(router_gemv_hip, "is_hip", return_value=True),
            patch.object(router_gemv_hip, "is_gfx95_supported", return_value=False),
            patch.object(router_gemv_hip, "is_hcu", return_value=True),
            patch.dict(os.environ, {"SGLANG_HCU_ROCM_ROUTER_SPLIT_K": "0"}, clear=False),
        ):
            self.assertEqual(
                rocm_router_gate_mod.rocm_router_max_tokens(
                    num_experts=384,
                    hidden_size=5120,
                    topk=6,
                    weight_dtype=torch.bfloat16,
                ),
                -1,
            )

    def test_rocm_dsv3_router_split_k_guards_dtype_and_rows(self):
        from sglang.srt.layers import rocm_linear_utils

        gate = SimpleNamespace(
            rocm_router_max_tokens=64,
            weight=torch.zeros((384, 5120), dtype=torch.bfloat16),
        )
        fake_exec = SimpleNamespace(
            deterministic=SimpleNamespace(enable_deterministic_inference=False)
        )
        with patch.object(rocm_linear_utils, "get_exec", return_value=fake_exec):
            # M > max_tokens -> fallback
            self.assertIsNone(
                rocm_linear_utils.rocm_dsv3_router_split_k(
                    gate, torch.zeros((65, 5120), dtype=torch.bfloat16)
                )
            )
            # Non-bf16 activation -> fallback
            self.assertIsNone(
                rocm_linear_utils.rocm_dsv3_router_split_k(
                    gate, torch.zeros((8, 5120), dtype=torch.float16)
                )
            )
            # Non-contiguous inner stride -> fallback
            strided_x = torch.zeros((5120, 8), dtype=torch.bfloat16).t()
            self.assertIsNone(
                rocm_linear_utils.rocm_dsv3_router_split_k(gate, strided_x)
            )

    def test_biased_topk_jit_kernel_does_not_double_scale_when_apply_on_output_false(
        self,
    ):
        from sglang.srt.layers.moe import topk as topk_mod

        captured = {}

        def fake_rocm_router_gate(
            gating_output,
            correction_bias,
            topk,
            renormalize,
            routed_scaling_factor,
            **_kwargs,
        ):
            captured["scale"] = routed_scaling_factor
            return (
                torch.ones((gating_output.shape[0], topk), dtype=torch.float32),
                torch.zeros((gating_output.shape[0], topk), dtype=torch.int32),
            )

        hidden = torch.zeros((4, 5120), dtype=torch.bfloat16)
        logits = torch.zeros((4, 384), dtype=torch.float32)
        bias = torch.zeros((384,), dtype=torch.float32)
        partials = torch.zeros((10, 4, 384), dtype=torch.float32)
        with (
            patch.object(topk_mod, "_use_aiter", False),
            patch.object(topk_mod, "_is_hcu", True),
            patch.object(topk_mod, "rocm_router_gate", side_effect=fake_rocm_router_gate),
        ):
            topk_mod.biased_topk_jit_kernel_impl(
                hidden_states=hidden,
                gating_output=logits,
                correction_bias=bias,
                topk=6,
                renormalize=True,
                scoring_func="sqrtsoftplus",
                routed_scaling_factor=1.5,
                apply_routed_scaling_factor_on_output=False,
                router_logits_partials=partials,
            )
            self.assertEqual(captured["scale"], 1.0)

            topk_mod.biased_topk_jit_kernel_impl(
                hidden_states=hidden,
                gating_output=logits,
                correction_bias=bias,
                topk=6,
                renormalize=True,
                scoring_func="sqrtsoftplus",
                routed_scaling_factor=1.5,
                apply_routed_scaling_factor_on_output=True,
                router_logits_partials=partials,
            )
            self.assertEqual(captured["scale"], 1.5)


@unittest.skipUnless(
    torch.cuda.is_available() and is_hip() and is_hcu(),
    "HCU GPU required",
)
class TestHcuRouterSplitKDevice(unittest.TestCase):
    def _ref_sqrtsoftplus_topk(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        bias: torch.Tensor,
        topk: int,
        rsf: float,
    ):
        logits = x.float() @ w.float().t()
        scores = torch.sqrt(torch.nn.functional.softplus(logits))
        biased = scores + bias.float()
        _, ids = torch.topk(biased, k=topk, dim=-1, largest=True, sorted=True)
        weights = scores.gather(-1, ids)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-20) * rsf
        return logits, weights, ids.to(torch.int32)

    def test_split_k_gemv_and_fused_gate_accuracy(self):
        device = torch.device("cuda")
        gen = torch.Generator(device=device).manual_seed(42)
        w = (torch.randn((384, 5120), device=device, generator=gen) * 0.02).to(
            torch.bfloat16
        )
        bias = torch.randn((384,), device=device, generator=gen, dtype=torch.float32) * 0.2
        rsf = 1.5
        for m in (1, 6, 16, 48, 64):
            x = torch.randn((m, 5120), device=device, generator=gen).to(torch.bfloat16)
            ref_logits, ref_w, ref_ids = self._ref_sqrtsoftplus_topk(x, w, bias, 6, rsf)

            partials = router_gemv_hip.rocm_router_gemv_split_k(x, w)
            out_logits = torch.empty((m, 384), dtype=torch.float32, device=device)
            out_w, out_ids = rocm_router_gate_mod.rocm_router_gate(
                out_logits, bias, 6, True, rsf, partials=partials
            )

            self.assertTrue(torch.equal(out_ids, ref_ids), f"M={m} topk ids mismatch")
            self.assertLess((out_logits - ref_logits).abs().max().item(), 1e-4)
            self.assertLess((out_w - ref_w).abs().max().item(), 1e-5)

    def test_split_k_router_hip_graph_replay(self):
        device = torch.device("cuda")
        gen = torch.Generator(device=device).manual_seed(7)
        x = torch.randn((6, 5120), device=device, generator=gen).to(torch.bfloat16)
        w = (torch.randn((384, 5120), device=device, generator=gen) * 0.02).to(
            torch.bfloat16
        )
        bias = torch.randn((384,), device=device, generator=gen, dtype=torch.float32) * 0.2

        def run():
            partials = router_gemv_hip.rocm_router_gemv_split_k(x, w)
            logits = torch.empty((6, 384), dtype=torch.float32, device=device)
            weights, ids = rocm_router_gate_mod.rocm_router_gate(
                logits, bias, 6, True, 1.5, partials=partials
            )
            return logits, weights, ids

        ref_logits, ref_w, ref_ids = run()
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            g_logits, g_w, g_ids = run()
        g.replay()
        torch.cuda.synchronize()

        self.assertTrue(torch.equal(g_ids, ref_ids))
        self.assertTrue(torch.equal(g_w, ref_w))
        self.assertTrue(torch.equal(g_logits, ref_logits))

    def test_split_k_router_deepep_finalize_padding_and_int64(self):
        device = torch.device("cuda")
        gen = torch.Generator(device=device).manual_seed(99)
        x = torch.randn((8, 5120), device=device, generator=gen).to(torch.bfloat16)
        w = (torch.randn((384, 5120), device=device, generator=gen) * 0.02).to(
            torch.bfloat16
        )
        bias = torch.randn((384,), device=device, generator=gen, dtype=torch.float32) * 0.2
        n_valid = torch.tensor([5], dtype=torch.int32, device=device)

        _, ref_w, ref_ids = self._ref_sqrtsoftplus_topk(x[:5], w, bias, 6, 2.5)
        partials = router_gemv_hip.rocm_router_gemv_split_k(x, w)
        logits = torch.empty((8, 384), dtype=torch.float32, device=device)
        out_w, out_ids = rocm_router_gate_mod.rocm_router_gate(
            logits,
            bias,
            6,
            True,
            2.5,
            partials=partials,
            sqrtsoftplus_log1p=True,
            num_token_non_padded=n_valid,
            output_int64=True,
        )
        self.assertEqual(out_ids.dtype, torch.int64)
        self.assertTrue(torch.equal(out_ids[:5], ref_ids.to(torch.int64)))
        self.assertLess((out_w[:5] - ref_w).abs().max().item(), 1e-5)
        self.assertTrue(torch.all(out_ids[5:] == -1))
        self.assertTrue(torch.all(out_w[5:] == 0.0))

    def test_select_experts_ids_and_weights_match_baseline_both_scaling_modes(self):
        """Regression test: select_experts must not double-multiply routed_scaling_factor."""
        from sglang.srt.layers.moe import topk as topk_mod

        device = torch.device("cuda")
        gen = torch.Generator(device=device).manual_seed(123)
        m = 8
        x = torch.randn((m, 5120), device=device, generator=gen).to(torch.bfloat16)
        w = (torch.randn((384, 5120), device=device, generator=gen) * 0.02).to(
            torch.bfloat16
        )
        bias = torch.randn((384,), device=device, generator=gen, dtype=torch.float32) * 0.2
        partials = router_gemv_hip.rocm_router_gemv_split_k(x, w)
        logits_ref = partials.sum(dim=0)

        fake_a2a = SimpleNamespace(is_deepep=lambda: False)
        fake_runner = SimpleNamespace(
            is_deep_gemm=lambda: False,
            is_flashinfer_trtllm_routed=lambda: False,
        )
        fake_recorder = SimpleNamespace(on_select_experts=lambda **_: None)

        for apply_on_output in (False, True):
            expected_sum = 1.5 if apply_on_output else 1.0
            cfg = topk_mod.TopKConfig(
                top_k=6,
                use_grouped_topk=False,
                renormalize=True,
                correction_bias=bias,
                routed_scaling_factor=1.5,
                apply_routed_scaling_factor_on_output=apply_on_output,
                scoring_func="sqrtsoftplus",
                sqrtsoftplus_log1p=True,
            )
            logits_split = torch.empty((m, 384), dtype=torch.float32, device=device)
            with (
                patch.object(topk_mod, "get_moe_a2a_backend", return_value=fake_a2a),
                patch.object(topk_mod, "get_moe_runner_backend", return_value=fake_runner),
                patch.object(
                    topk_mod,
                    "get_global_expert_distribution_recorder",
                    return_value=fake_recorder,
                ),
            ):
                base_out = topk_mod.select_experts(
                    hidden_states=x,
                    router_logits=logits_ref.clone(),
                    topk_config=cfg,
                )
                split_out = topk_mod.select_experts(
                    hidden_states=x,
                    router_logits=logits_split,
                    topk_config=cfg,
                    router_logits_partials=partials,
                )

            self.assertTrue(
                torch.equal(split_out.topk_ids, base_out.topk_ids),
                f"apply_on_output={apply_on_output}: topk_ids mismatch",
            )
            self.assertLess(
                (split_out.topk_weights - base_out.topk_weights).abs().max().item(),
                1e-5,
                f"apply_on_output={apply_on_output}: topk_weights mismatch "
                f"(split sum={split_out.topk_weights[0].sum().item():.4f} vs "
                f"base sum={base_out.topk_weights[0].sum().item():.4f})",
            )
            self.assertAlmostEqual(
                split_out.topk_weights[0].sum().item(), expected_sum, places=4
            )


if __name__ == "__main__":
    unittest.main()
