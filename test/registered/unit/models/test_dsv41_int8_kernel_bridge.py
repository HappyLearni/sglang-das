"""CPU checks for the consolidated API and installed kernel boundary."""
import builtins
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import torch
from sglang.kernels.ops.gemm import dsv41_int8
from sglang.test.ci.ci_register import register_cpu_ci
register_cpu_ci(est_time=10, suite="base-a-test-cpu")

class TestInt8KernelBridge(unittest.TestCase):
    def tearDown(self):
        for function in [dsv41_int8.backend, dsv41_int8._service_config, dsv41_int8._algorithms, dsv41_int8._scale_schedules]:
            function.cache_clear()
        dsv41_int8._algo_rank_cache.clear()

    def test_backend_lazily_binds_installed_kernel_once(self):
        ops = types.SimpleNamespace(**{name: Mock() for name in dsv41_int8._OPERATORS})
        with patch.dict(sys.modules, {"sgl_kernel": types.ModuleType("sgl_kernel")}), patch.object(torch.ops, "sgl_kernel", ops):
            first = dsv41_int8.backend()
            self.assertIs(first, dsv41_int8.backend())
            q, weight, raw = object(), object(), object()
            first.gemm(q, weight, raw, 17)
        ops.dsv41_int8_gemm_packed.assert_called_once_with(q, weight, raw, 17)

    def test_old_kernel_wheel_reports_required_upgrade(self):
        with patch.dict(sys.modules, {"sgl_kernel": types.ModuleType("sgl_kernel")}), patch.object(torch.ops, "sgl_kernel", types.SimpleNamespace()):
            with self.assertRaisesRegex(RuntimeError, "Rebuild the matching HCU"):
                dsv41_int8.backend()

    def test_native_bridge_keeps_plan_and_scale_arguments(self):
        ops = types.SimpleNamespace(dsv41_int8_prepare=Mock(return_value=17), dsv41_int8_gemm_packed=Mock(), dsv41_int8_scale_scheduled=Mock())
        api = dsv41_int8._NativeOps(ops)
        weight, q, raw, xs, ws, out = [object() for _ in range(6)]
        plan = api.prepare(weight, 40, 1, 16160, 5120, 1, 16160, 82739200, 5120, False, "")
        api.gemm(q, weight, raw, plan)
        api.scale_scheduled(raw, xs, ws, out, 2, 4, 128)
        ops.dsv41_int8_prepare.assert_called_once_with(weight, 40, 1, 16160, 5120, 1, 16160, 82739200, 5120, False, "")
        ops.dsv41_int8_gemm_packed.assert_called_once_with(q, weight, raw, 17)
        ops.dsv41_int8_scale_scheduled.assert_called_once_with(raw, xs, ws, out, 2, 4, 128)

    def test_embedded_defaults_and_explicit_configuration(self):
        with patch.dict(os.environ):
            os.environ.pop("SGLANG_DSV41_INT8_SERVICE_ALGO_CONFIG", None)
            self.assertIn("wo_a:1:4096:8:1024:4325376:0", dsv41_int8._algorithms())
        for function in [dsv41_int8._service_config, dsv41_int8._algorithms, dsv41_int8._scale_schedules]:
            function.cache_clear()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "algorithms.json"
            config = {"algorithms": {"custom": "1234"}, "scale_schedules": {"lm_head": {"4": [1, 2, 128]}}}
            path.write_text(json.dumps(config))
            with patch.dict(os.environ, {"SGLANG_DSV41_INT8_SERVICE_ALGO_CONFIG": str(path)}):
                self.assertEqual(dsv41_int8._algorithms(), config["algorithms"])
                self.assertEqual(dsv41_int8._scale_schedules(), config["scale_schedules"]["lm_head"])

    def test_import_does_not_load_native_backend(self):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name == "sgl_kernel":
                raise AssertionError("Importing the API must not load native operators")
            return original(name, *args, **kwargs)
        spec = importlib.util.spec_from_file_location("int8_import_probe", dsv41_int8.__file__)
        module = importlib.util.module_from_spec(spec)
        with patch("builtins.__import__", side_effect=guarded):
            spec.loader.exec_module(module)
        self.assertEqual(module.backend.cache_info().currsize, 0)

    def test_autotune_selects_fastest_rank_caches_and_stops_on_exhaustion(self):
        latencies = {0: 5.0, 1: 1.2, 2: 2.5}
        calls = []

        def fake_gemm_out(q, w, raw, groups, tile_n, group_major, algo_rank=0):
            calls.append(algo_rank)
            if algo_rank not in latencies:
                raise RuntimeError("invalid heuristic rank")

        class FakeEvent:
            last_rank = 0

            def __init__(self, enable_timing=False):
                self.rank = 0

            def record(self):
                self.rank = calls[-1] if calls else 0

            def synchronize(self):
                pass

            def elapsed_time(self, other):
                return latencies[other.rank]

        ops = types.SimpleNamespace(gemm_out=fake_gemm_out)
        fake_q = types.SimpleNamespace(
            shape=(4096, 8, 4096),
            is_cuda=True,
            device=types.SimpleNamespace(index=0),
        )
        with (
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
            patch.object(torch.cuda, "Event", FakeEvent),
        ):
            best = dsv41_int8._select_unpacked_algo_rank(
                ops, fake_q, object(), object(), 8, 1024, 4096, 0, False
            )
            self.assertEqual(best, 1)
            probe_calls = len(calls)
            # Second call hits cache with zero extra probes.
            best_cached = dsv41_int8._select_unpacked_algo_rank(
                ops, fake_q, object(), object(), 8, 1024, 4096, 0, False
            )
            self.assertEqual(best_cached, 1)
            self.assertEqual(len(calls), probe_calls)

    def test_autotune_skipped_during_graph_capture_without_poisoning_cache(self):
        ops = types.SimpleNamespace(gemm_out=Mock())
        fake_q = types.SimpleNamespace(
            shape=(6, 8, 4096),
            is_cuda=True,
            device=types.SimpleNamespace(index=0),
        )
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            rank = dsv41_int8._select_unpacked_algo_rank(
                ops, fake_q, object(), object(), 8, 1024, 4096, 0, False
            )
        self.assertEqual(rank, 0)
        self.assertEqual(len(dsv41_int8._algo_rank_cache), 0)
        ops.gemm_out.assert_not_called()

    def test_autotune_respects_disable_env_and_explicit_rank(self):
        with patch.dict(os.environ, {"SGLANG_DSV41_INT8_BLASLT_AUTOTUNE": "0"}):
            self.assertFalse(dsv41_int8._blaslt_autotune_enabled())
        with patch.dict(os.environ, {"SGLANG_DSV41_INT8_BLASLT_AUTOTUNE": "1"}):
            self.assertTrue(dsv41_int8._blaslt_autotune_enabled())

    def test_autotune_buckets_large_prefill_t_but_keeps_small_decode_exact(self):
        self.assertEqual(dsv41_int8._autotune_t_bucket(6), 6)
        self.assertEqual(dsv41_int8._autotune_t_bucket(48), 48)
        self.assertEqual(dsv41_int8._autotune_t_bucket(64), 64)
        self.assertEqual(dsv41_int8._autotune_t_bucket(65), 128)
        self.assertEqual(dsv41_int8._autotune_t_bucket(3652), 4096)
        self.assertEqual(dsv41_int8._autotune_t_bucket(3653), 4096)
        self.assertEqual(dsv41_int8._autotune_t_bucket(4096), 4096)

if __name__ == "__main__":
    unittest.main()
