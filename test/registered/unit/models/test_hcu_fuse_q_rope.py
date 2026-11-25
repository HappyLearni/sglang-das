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


if __name__ == "__main__":
    unittest.main()
