import os
import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.compressed_tensors import quant_ops as mod
from sglang.srt.utils import is_hcu, is_hip


class _FakeEvent:
    _clock = 0.0

    def __init__(self, enable_timing: bool = True):
        self.t = 0.0

    def record(self, stream=None):
        self.t = _FakeEvent._clock

    def synchronize(self):
        pass

    def elapsed_time(self, other):
        return other.t - self.t


class TestHcuW8A8SmallMAutotune(unittest.TestCase):
    def setUp(self):
        mod._W8A8_SMALL_M_AUTOTUNE_CACHE.clear()
        _FakeEvent._clock = 0.0

    def tearDown(self):
        mod._W8A8_SMALL_M_AUTOTUNE_CACHE.clear()

    def test_selects_fastest_valid_candidate_and_caches(self):
        a = torch.zeros((6, 5120), dtype=torch.int8)
        b = torch.zeros((2112, 5120), dtype=torch.int8).t()
        scale_a = torch.ones((6, 1), dtype=torch.float32)
        scale_b = torch.ones((2112,), dtype=torch.float32)

        latencies = {
            ("lightop_channel", 0): None,  # unsupported shape returns None
            ("smooth_asm", 1): 0.012,
            ("smooth_asm", 2): 0.018,
            ("smooth_asm", 3): 0.025,
            ("smooth_asm", 4): 0.020,
            ("smooth_asm", 9): 0.016,
        }
        calls = []

        def fake_run(kind, tune, *_args):
            calls.append((kind, tune))
            dt = latencies[(kind, tune)]
            if dt is None:
                return None
            _FakeEvent._clock += dt
            return torch.zeros((6, 2112), dtype=torch.bfloat16)

        with (
            patch.object(mod, "_run_w8a8_small_m_candidate", side_effect=fake_run),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
            patch.object(torch.cuda, "current_stream", return_value=None),
            patch.object(torch.cuda, "Event", _FakeEvent),
        ):
            best = mod._select_hcu_w8a8_small_m_kernel(
                a, b, scale_a, scale_b, torch.bfloat16, 6, 2112, 5120
            )
            self.assertEqual(best, ("smooth_asm", 1))
            n_calls_first = len(calls)

            # Second call in the same M bucket (e.g., M=6 and M=7 -> bucket 8)
            # hits cache with zero extra candidate probes.
            best2 = mod._select_hcu_w8a8_small_m_kernel(
                a, b, scale_a, scale_b, torch.bfloat16, 7, 2112, 5120
            )
            self.assertEqual(best2, ("smooth_asm", 1))
            self.assertEqual(len(calls), n_calls_first)

    def test_selects_lightop_channel_when_fastest_and_supported(self):
        a = torch.zeros((6, 8192), dtype=torch.int8)
        b = torch.zeros((5120, 8192), dtype=torch.int8).t()
        scale_a = torch.ones((6, 1), dtype=torch.float32)
        scale_b = torch.ones((5120,), dtype=torch.float32)

        latencies = {
            ("lightop_channel", 0): 0.036,
            ("smooth_asm", 1): 0.060,
            ("smooth_asm", 2): 0.055,
            ("smooth_asm", 3): 0.050,
            ("smooth_asm", 4): 0.048,
            ("smooth_asm", 9): 0.043,
        }

        def fake_run(kind, tune, *_args):
            _FakeEvent._clock += latencies[(kind, tune)]
            return torch.zeros((6, 5120), dtype=torch.bfloat16)

        with (
            patch.object(mod, "_run_w8a8_small_m_candidate", side_effect=fake_run),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
            patch.object(torch.cuda, "current_stream", return_value=None),
            patch.object(torch.cuda, "Event", _FakeEvent),
        ):
            best = mod._select_hcu_w8a8_small_m_kernel(
                a, b, scale_a, scale_b, torch.bfloat16, 6, 5120, 8192
            )
            self.assertEqual(best, ("lightop_channel", 0))

    def test_skips_probing_during_graph_capture_unless_cached(self):
        a = torch.zeros((6, 5120), dtype=torch.int8)
        b = torch.zeros((2112, 5120), dtype=torch.int8).t()
        scale_a = torch.ones((6, 1), dtype=torch.float32)
        scale_b = torch.ones((2112,), dtype=torch.float32)

        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            self.assertIsNone(
                mod._select_hcu_w8a8_small_m_kernel(
                    a, b, scale_a, scale_b, torch.bfloat16, 6, 2112, 5120
                )
            )
            # Pre-populated bucket cache (M=6 -> bucket 8) returns cleanly during capture.
            mod._W8A8_SMALL_M_AUTOTUNE_CACHE[(None, 8, 2112, 5120, torch.bfloat16)] = (
                "smooth_asm",
                1,
            )
            self.assertEqual(
                mod._select_hcu_w8a8_small_m_kernel(
                    a, b, scale_a, scale_b, torch.bfloat16, 6, 2112, 5120
                ),
                ("smooth_asm", 1),
            )

    def test_env_disable_flag(self):
        with patch.dict(
            os.environ, {"SGLANG_HCU_W8A8_SMALL_M_AUTOTUNE": "0"}, clear=False
        ):
            self.assertFalse(mod._hcu_w8a8_small_m_autotune_enabled())
        with patch.dict(
            os.environ, {"SGLANG_HCU_W8A8_SMALL_M_AUTOTUNE": "1"}, clear=False
        ):
            self.assertTrue(mod._hcu_w8a8_small_m_autotune_enabled())


@unittest.skipUnless(
    torch.cuda.is_available() and is_hip() and is_hcu(),
    "HCU GPU required",
)
class TestHcuW8A8SmallMAutotuneDevice(unittest.TestCase):
    def setUp(self):
        mod._W8A8_SMALL_M_AUTOTUNE_CACHE.clear()

    def test_autotuned_output_matches_reference_and_hip_graph(self):
        device = torch.device("cuda")
        gen = torch.Generator(device=device).manual_seed(99)
        for m, n, k in ((6, 2112, 5120), (6, 5120, 8192), (16, 2048, 5120)):
            a = torch.randint(-64, 64, (m, k), device=device, generator=gen, dtype=torch.int8)
            b_nk = torch.randint(-64, 64, (n, k), device=device, generator=gen, dtype=torch.int8)
            b = b_nk.t()  # stride (1, k)
            scale_a = (torch.rand((m, 1), device=device, generator=gen) * 0.02 + 0.005).float()
            scale_b = (torch.rand((n,), device=device, generator=gen) * 0.02 + 0.005).float()

            ref = (
                (a.float() @ b_nk.float().t()) * scale_a * scale_b.unsqueeze(0)
            ).to(torch.bfloat16)

            out = mod.triton_scaled_mm(a, b, scale_a, scale_b, torch.bfloat16)
            self.assertEqual(out.shape, (m, n))
            self.assertEqual(out.dtype, torch.bfloat16)
            rel_err = (out.float() - ref.float()).abs().max().item() / (
                ref.float().abs().max().item() + 1e-6
            )
            self.assertLess(rel_err, 2e-2, f"shape={(m, n, k)} rel_err={rel_err}")

            # Verify HIP Graph capture + replay uses cached selection cleanly.
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                g_out = mod.triton_scaled_mm(a, b, scale_a, scale_b, torch.bfloat16)
            g.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(g_out, out))


if __name__ == "__main__":
    unittest.main()
