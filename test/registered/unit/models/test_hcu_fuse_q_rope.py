import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models.deepseek_common.amd import deepseek_v4_hip as mod
from sglang.srt.utils import is_hcu, is_hip


class TestHcuFuseQRopeDispatch(unittest.TestCase):
    def _make_attn(self):
        return SimpleNamespace(
            head_dim=512,
            rope_head_dim=64,
            q_head_norm=False,
            n_local_heads=16,
            wq_b=lambda x: (torch.zeros((x.shape[0], 16 * 512), dtype=torch.bfloat16), None),
            _compute_q_b=lambda q, pos, q_out: torch.ones((q.shape[0], 16, 512), dtype=torch.bfloat16),
        )

    def test_hcu_gate_checks_kv_pool_and_env(self):
        attn = self._make_attn()
        pool_ok = SimpleNamespace(is_bf16_attention_kv_cache=False, uniform_fp8=False)
        pool_bf16 = SimpleNamespace(is_bf16_attention_kv_cache=True, uniform_fp8=False)
        pool_cp_ls = SimpleNamespace(
            is_bf16_attention_kv_cache=False,
            uniform_fp8=False,
            _owns_swa_layer_id=lambda _: True,
        )

        with (
            patch.object(mod, "_is_hcu", True),
            patch.object(mod, "_is_gfx95_supported", False),
            patch.object(mod, "in_replay_tail_graph", return_value=False),
            patch.dict(os.environ, {"SGLANG_HCU_FUSE_Q_ROPE": "1"}, clear=False),
        ):
            with patch.object(mod, "get_token_to_kv_pool", return_value=pool_ok):
                self.assertTrue(mod._hcu_kv_pool_supports_fused_q_rope(attn))
            with patch.object(mod, "get_token_to_kv_pool", return_value=pool_bf16):
                self.assertFalse(mod._hcu_kv_pool_supports_fused_q_rope(attn))
            with patch.object(mod, "get_token_to_kv_pool", return_value=pool_cp_ls):
                self.assertFalse(mod._hcu_kv_pool_supports_fused_q_rope(attn))

        with (
            patch.object(mod, "_is_hcu", True),
            patch.object(mod, "_is_gfx95_supported", False),
            patch.object(mod, "in_replay_tail_graph", return_value=False),
            patch.object(mod, "get_token_to_kv_pool", return_value=pool_ok),
            patch.dict(os.environ, {"SGLANG_HCU_FUSE_Q_ROPE": "0"}, clear=False),
        ):
            self.assertFalse(mod._hcu_kv_pool_supports_fused_q_rope(attn))

    def test_compute_q_b_preserves_bf16_q_lora_on_hcu(self):
        attn = self._make_attn()
        pool_ok = SimpleNamespace(is_bf16_attention_kv_cache=False, uniform_fp8=False)
        q_lora = torch.zeros((6, 1536), dtype=torch.bfloat16)
        q_for_wqb = SimpleNamespace(shape=(6, 1536))  # simulated Int8PerTokenInput
        with (
            patch.object(mod, "_is_hcu", True),
            patch.object(mod, "_is_gfx95_supported", False),
            patch.object(mod, "in_replay_tail_graph", return_value=False),
            patch.object(mod, "get_token_to_kv_pool", return_value=pool_ok),
            patch.dict(os.environ, {"SGLANG_HCU_FUSE_Q_ROPE": "1"}, clear=False),
        ):
            q, idx_q_lora, fused = mod.compute_q_b(
                attn, q_lora, q_for_wqb, torch.arange(6), None, unified=False, use_cp=False
            )
            self.assertTrue(fused)
            self.assertIs(idx_q_lora, q_lora)
            self.assertEqual(q.shape, (6, 16, 512))

    def test_fp16_q_or_kv_falls_back_to_unfused_rope_path(self):
        pool_ok = SimpleNamespace(is_bf16_attention_kv_cache=False, uniform_fp8=False)
        attn_bf16 = self._make_attn()
        attn_fp16_kv_norm = self._make_attn()
        attn_fp16_kv_norm.kv_norm = SimpleNamespace(
            weight=torch.ones((512,), dtype=torch.float16)
        )
        attn_fp16_wqb = self._make_attn()
        attn_fp16_wqb.wq_b = lambda x: (
            torch.zeros((x.shape[0], 16 * 512), dtype=torch.float16),
            None,
        )
        attn_fp16_wqb._compute_q_b = lambda q, pos, q_out: torch.ones(
            (q.shape[0], 16, 512), dtype=torch.float16
        )

        with (
            patch.object(mod, "_is_hcu", True),
            patch.object(mod, "_is_gfx95_supported", False),
            patch.object(mod, "in_replay_tail_graph", return_value=False),
            patch.object(mod, "get_token_to_kv_pool", return_value=pool_ok),
            patch.dict(os.environ, {"SGLANG_HCU_FUSE_Q_ROPE": "1"}, clear=False),
        ):
            # FP16 q_lora -> unfused path.
            q_fp16 = torch.zeros((6, 1536), dtype=torch.float16)
            q, _, fused = mod.compute_q_b(
                attn_bf16,
                q_fp16,
                q_fp16,
                torch.arange(6),
                None,
                unified=False,
                use_cp=False,
                kv_dtype=torch.float16,
            )
            self.assertFalse(fused)
            self.assertTrue(torch.all(q == 1))

            # BF16 q_lora with FP16 kv_dtype -> unfused path.
            q_bf16 = torch.zeros((6, 1536), dtype=torch.bfloat16)
            _, _, fused_kv = mod.compute_q_b(
                attn_bf16,
                q_bf16,
                q_bf16,
                torch.arange(6),
                None,
                unified=False,
                use_cp=False,
                kv_dtype=torch.float16,
            )
            self.assertFalse(fused_kv)

            # FP16 kv_norm weight -> unfused path.
            self.assertFalse(
                mod._hcu_kv_pool_supports_fused_q_rope(
                    attn_fp16_kv_norm,
                    q_dtype=torch.bfloat16,
                    kv_dtype=torch.bfloat16,
                )
            )

            # wq_b unexpectedly returning FP16 -> falls back to _compute_q_b.
            q_out_fp16, _, fused_wqb = mod.compute_q_b(
                attn_fp16_wqb,
                q_bf16,
                q_bf16,
                torch.arange(6),
                None,
                unified=False,
                use_cp=False,
                kv_dtype=torch.bfloat16,
            )
            self.assertFalse(fused_wqb)
            self.assertEqual(q_out_fp16.dtype, torch.float16)


@unittest.skipUnless(
    torch.cuda.is_available() and is_hip() and is_hcu(),
    "HCU GPU required",
)
class TestHcuFuseQRopeDevice(unittest.TestCase):
    def test_fused_k_norm_rope_with_q_matches_reference_and_hip_graph(self):
        from sglang.kernels.ops.attention.dsv4.elementwise import (
            KVLayout,
            fused_k_norm_rope_flashmla,
        )

        device = torch.device("cuda")
        gen = torch.Generator(device=device).manual_seed(11)
        B, H_q, D, RD, PS = 16, 16, 512, 64, 64
        kv = torch.randn((B, D), device=device, generator=gen).to(torch.bfloat16)
        kvw = (torch.randn((D,), device=device, generator=gen) * 0.2 + 1.0).to(
            torch.bfloat16
        )
        q0 = torch.randn((B, H_q, D), device=device, generator=gen).to(torch.bfloat16)
        angles = torch.randn((128, RD // 2), device=device, generator=gen, dtype=torch.float32)
        freqs = torch.complex(torch.cos(angles), torch.sin(angles))
        pos = torch.arange(B, dtype=torch.int64, device=device)
        loc = torch.arange(B, dtype=torch.int32, device=device)
        kvcache_ref = torch.zeros((4, 37440), dtype=torch.uint8, device=device)
        kvcache_fused = torch.zeros((4, 37440), dtype=torch.uint8, device=device)

        q_ref = q0.clone()
        ql_c = torch.view_as_complex(q_ref[..., D - RD :].float().reshape(B, H_q, RD // 2, 2))
        q_ref[..., D - RD :] = (
            torch.view_as_real(ql_c * freqs[pos].unsqueeze(1)).flatten(-2).to(torch.bfloat16)
        )
        fused_k_norm_rope_flashmla(
            kv, kvw, 1e-6, freqs, pos, loc, kvcache_ref, PS, layout=KVLayout.V4, q=None
        )

        q_fused = q0.clone()
        fused_k_norm_rope_flashmla(
            kv, kvw, 1e-6, freqs, pos, loc, kvcache_fused, PS, layout=KVLayout.V4, q=q_fused
        )
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(kvcache_ref, kvcache_fused))
        self.assertTrue(torch.equal(q_ref, q_fused))

        # HIP Graph replay check
        q_graph = q0.clone()
        kvcache_graph = torch.zeros_like(kvcache_ref)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fused_k_norm_rope_flashmla(
                kv, kvw, 1e-6, freqs, pos, loc, kvcache_graph, PS, layout=KVLayout.V4, q=q_graph
            )
        q_graph.copy_(q0)
        kvcache_graph.zero_()
        g.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(kvcache_ref, kvcache_graph))
        self.assertTrue(torch.equal(q_ref, q_graph))

    def test_fp16_kv_store_unfused_path_succeeds_without_bf16_q_rope(self):
        from sglang.kernels.ops.attention.dsv4.elementwise import (
            KVLayout,
            fused_k_norm_rope_flashmla,
        )

        device = torch.device("cuda")
        gen = torch.Generator(device=device).manual_seed(17)
        B, D, RD, PS = 8, 512, 64, 64
        kv = torch.randn((B, D), device=device, generator=gen).to(torch.float16)
        kvw = (torch.randn((D,), device=device, generator=gen) * 0.2 + 1.0).to(
            torch.float16
        )
        angles = torch.randn((32, RD // 2), device=device, generator=gen, dtype=torch.float32)
        freqs = torch.complex(torch.cos(angles), torch.sin(angles))
        pos = torch.arange(B, dtype=torch.int64, device=device)
        loc = torch.arange(B, dtype=torch.int32, device=device)
        kvcache = torch.zeros((4, 37440), dtype=torch.uint8, device=device)
        # Unfused FP16 K-store (q=None) succeeds cleanly when compute_q_b routes FP16 off the fused Q-RoPE path.
        fused_k_norm_rope_flashmla(
            kv, kvw, 1e-6, freqs, pos, loc, kvcache, PS, layout=KVLayout.V4, q=None
        )
        torch.cuda.synchronize()
        self.assertGreater(kvcache.int().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
