import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation import utils as disagg_utils
from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.mem_cache.cp_cache_layer_split.validation import (
    validate_cp_cache_layer_split,
)
from sglang.srt.model_executor import pool_configurator as pools
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

register_cpu_ci(est_time=10, stage="base-a", runner_config="cpu")


class TestCpCacheLayerSplitDspark(CustomTestCase):
    def _args(self, **changes):
        values = dict(
            enable_dsa_cache_layer_split=False,
            disaggregation_mode="prefill",
            enable_prefill_cp=True,
            cp_strategy="interleave",
            attn_cp_size=8,
            pp_size=1,
            enable_two_batch_overlap=False,
            enable_single_batch_overlap=True,
            disaggregation_transfer_backend="mooncake",
            enable_hisparse=False,
            enable_hierarchical_cache=False,
            prefill_only_disable_kv_cache=False,
            enable_encoder_swa_bounded_replay=False,
            speculative_algorithm="DSPARK",
            dsv4_prefill_backend="flash_mla_sparse",
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend=Backend.DISABLED)
            ),
        )
        values.update(changes)
        return SimpleNamespace(**values)

    def _validate(self, args, target_layer_ids=(37, 38, 39)):
        config = SimpleNamespace(
            model_type="deepseek_v41",
            text_config=SimpleNamespace(
                compress_ratios=[0, 1, 2],
                dspark_target_layer_ids=list(target_layer_ids),
            ),
        )
        with ExitStack() as stack:
            for path, value in (
                ("sglang.srt.utils.is_cuda", False),
                ("sglang.srt.utils.is_hcu", True),
                ("sglang.srt.environ.envs.SGLANG_OPT_USE_COMPRESSOR_V2.get", True),
                ("sglang.srt.environ.envs.SGLANG_DISAGG_STAGING_BUFFER.get", False),
                (
                    "sglang.kernels.ops.attention.dsv4.unified_kv_kernels."
                    "env_gate.is_unified_kv_triton",
                    False,
                ),
                (
                    "sglang.srt.layers.attention.dsv4.sparse_prefill_utils."
                    "use_dsv4_q8kv8_sparse_prefill",
                    False,
                ),
            ):
                stack.enter_context(mock.patch(path, return_value=value))
            validate_cp_cache_layer_split(args, config)

    def test_dspark_and_sbo_allowed(self):
        self._validate(self._args())
        self._validate(self._args(speculative_algorithm=None), target_layer_ids=())

    def test_missing_dspark_config_rejected(self):
        with self.assertRaisesRegex(ValueError, "DSPARK model configuration"):
            self._validate(self._args(), target_layer_ids=())

    def test_other_unsupported_modes_remain_rejected(self):
        for changes, message in (
            ({"speculative_algorithm": "EAGLE"}, "DSPARK speculation only"),
            ({"disaggregation_mode": "decode"}, "disaggregation-mode prefill"),
            ({"enable_two_batch_overlap": True}, "two-batch overlap"),
        ):
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValueError, message):
                    self._validate(self._args(**changes))

    def _configurator(self, *, speculative, draft=False, ratio=None):
        schedule = SimpleNamespace(
            max_running_requests=64,
            _swa_full_tokens_ratio_explicitly_set=ratio is not None,
            swa_full_tokens_ratio=ratio,
            swa_prefix_tails=None,
            chunked_prefill_size=32768,
        )
        kvc = SimpleNamespace(
            kv_cache_dtype_str="fp8_e4m3",
            model_config=SimpleNamespace(
                qk_nope_head_dim=448,
                qk_rope_head_dim=64,
                index_head_dim=128,
                context_len=1048576,
                compress_ratios=[0, 0] + [2] * 18 + [1] * 20,
                window_size=128,
                hf_config=SimpleNamespace(kv_source_layer_ids=[2, 8, 14, 20]),
                hf_text_config=SimpleNamespace(dspark_target_layer_ids=[37, 38, 39]),
            ),
            layer_info=SimpleNamespace(start_layer=0, end_layer=40),
            pp_size=1,
            attn_dp_size=1,
            page_size=256,
            sliding_window_size=128,
            is_draft_worker=draft,
            spec_algorithm=(
                SpeculativeAlgorithm.DSPARK if speculative else SpeculativeAlgorithm.NONE
            ),
            # Bundled DSPARK's generic DFlash metadata reports the 40-layer
            # target; its real draft stage count comes from target_layer_ids.
            spec_aux_config=SimpleNamespace(dflash_draft_num_layers=40),
        )
        with ExitStack() as stack:
            for name, value in (
                ("get_schedule", schedule),
                (
                    "get_spec",
                    SimpleNamespace(
                        speculative_algorithm="DSPARK" if speculative else None
                    ),
                ),
                (
                    "get_parallel",
                    SimpleNamespace(enable_cp_cache_layer_split=True, attn_cp_size=8),
                ),
                (
                    "get_disagg",
                    SimpleNamespace(
                        disaggregation_mode="prefill",
                        disaggregation_decode_extra_slots=0,
                    ),
                ),
                (
                    "get_memory",
                    SimpleNamespace(enable_hisparse=False, disable_radix_cache=True),
                ),
                (
                    "get_exec",
                    SimpleNamespace(
                        kernel=SimpleNamespace(
                            enable_deepseek_v4_fp4_indexer=True,
                            dsv4_attn_backend="flashmla",
                        ),
                        features=SimpleNamespace(enable_encoder_swa_bounded_replay=False),
                    ),
                ),
                ("max_speculative_num_draft_tokens", 6 if speculative else 0),
            ):
                stack.enter_context(mock.patch.object(pools, name, return_value=value))
            for name in ("is_unified_kv_triton", "is_unified_kv_fp8"):
                stack.enter_context(
                    mock.patch(
                        "sglang.kernels.ops.attention.dsv4.unified_kv_kernels."
                        f"env_gate.{name}",
                        return_value=False,
                    )
                )
            stack.enter_context(
                mock.patch.object(
                    pools.envs.SGLANG_OPT_USE_ONLINE_COMPRESS, "get", return_value=False
                )
            )
            # Hold the allocator's request cap fixed to isolate draft pricing.
            stack.enter_context(
                mock.patch.object(pools, "compute_swa_request_cap", return_value=65536)
            )
            stack.enter_context(
                mock.patch(
                    "sglang.srt.mem_cache.deepseek_v4_memory_pool.select_dsv4_kv_layout",
                    return_value=(KVLayout.V4, None),
                )
            )
            stack.enter_context(
                mock.patch(
                    "sglang.srt.mem_cache.deepseek_v4_memory_pool.get_exec",
                    side_effect=lambda: pools.get_exec(),
                )
            )
            stack.enter_context(
                mock.patch(
                    "sglang.srt.mem_cache.deepseek_v4_memory_pool.get_spec",
                    # Hold target compressor ring pricing fixed in both cases;
                    # this test isolates the extra replicated draft KV budget.
                    return_value=SimpleNamespace(
                        speculative_algorithm="DSPARK", speculative_num_draft_tokens=6
                    ),
                )
            )
            return pools.DSV4PoolConfigurator(kvc)

    def test_replicated_draft_budget_cap_and_ratio(self):
        for ratio in (None, 0.1):
            with self.subTest(ratio=ratio):
                base = self._configurator(speculative=False, ratio=ratio)
                spec = self._configurator(speculative=True, ratio=ratio)
                self.assertEqual(spec.cp_draft_layers, 3)
                self.assertEqual(spec._spec_infl, 1.0)
                # Main now prices the physical page stride, including its pad.
                paged_kv_bytes = KVLayout.V4.page_bytes(spec.page_size) / spec.page_size
                self.assertEqual(
                    spec.bytes_per_swa_token - base.bytes_per_swa_token,
                    3 * paged_kv_bytes,
                )
                if ratio is None:
                    self.assertEqual(
                        spec.bytes_per_full_token, base.bytes_per_full_token
                    )
                    self.assertEqual(
                        spec._get_swa_fixed_bytes() - base._get_swa_fixed_bytes(),
                        spec.swa_cap_tokens * 3 * paged_kv_bytes,
                    )
                else:
                    self.assertAlmostEqual(
                        spec.bytes_per_full_token - base.bytes_per_full_token,
                        ratio * 3 * paged_kv_bytes,
                    )

    def test_draft_configurator_does_not_layer_shard(self):
        draft = self._configurator(speculative=True, draft=True)
        self.assertFalse(draft.use_cp_cache_layer_split)
        self.assertEqual(draft.cp_draft_layers, 0)

    def _transfer_args(self, rank=None, *, request_window=False):
        kv = []
        for ratio, sources in ((1, (20,)), (2, (2, 8, 14))):
            kv.extend(
                ((f"c{ratio}_kv", layer), 65536 // ratio)
                for layer in sources
                if rank is None or layer // 5 == rank
            )
            kv.extend(
                ((f"c{ratio}_indexer", layer), 16384 // ratio)
                for layer in sources
            )
        swa_layers = range(40) if rank is None else range(rank * 5, (rank + 1) * 5)
        states = [
            [(("swa_kv", layer), 32768) for layer in swa_layers],
            [(("c2_request_state", layer), 512) for layer in (2, 8, 14)],
            [(("draft_swa_kv", layer), 32768) for layer in range(3)],
        ]
        args = SimpleNamespace(
            mla_compression_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0] * 3,
            # The unsharded V4.1 pool has no legacy C4/C128 layer IDs,
            # whereas LayerSplit reports rank-local descriptor layer IDs.
            kv_layer_ids=[] if rank is None else [key[1] for key, _ in kv],
            kv_item_lens=[size for _, size in kv],
            state_types=[StateType.SWA, StateType.DSV4_REQUEST_STATE, StateType.SWA],
            state_item_lens=[[size for _, size in entries] for entries in states],
            cp_cache_layer_split=rank is not None,
            v4_transfer_metadata=dict(
                kv=[key for key, _ in kv],
                states=[[key for key, _ in entries] for entries in states],
            ),
        )
        if request_window:
            window = [
                ((family, layer), size)
                for family, size in (
                    ("dsv41_swa_window_kv", 32768),
                    ("dsv41_swa_window_tags", 2048),
                )
                for layer in swa_layers
            ]
            states = [states[1], window, states[2]]
            args.state_types = [
                StateType.DSV4_REQUEST_STATE,
                StateType.DSV41_SWA_WINDOW,
                StateType.SWA,
            ]
            args.state_item_lens = [[size for _, size in entries] for entries in states]
            args.v4_transfer_metadata["states"] = [
                [key for key, _ in entries] for entries in states
            ]
        return args, [kv, *states]

    def _check_handshake(self, *, request_window):
        peers = [
            self._transfer_args(rank, request_window=request_window)
            for rank in range(8)
        ]
        gather = mock.Mock(return_value=[entries for _, entries in peers])
        with (
            mock.patch.object(
                disagg_utils,
                "get_spec",
                return_value=SimpleNamespace(
                    speculative_algorithm="DSPARK", speculative_num_draft_tokens=6
                ),
            ),
            mock.patch.object(
                disagg_utils,
                "get_parallel",
                return_value=SimpleNamespace(
                    attn_cp_group=SimpleNamespace(all_gather_object=gather)
                ),
                create=True,
            ),
        ):
            decode, _ = self._transfer_args(request_window=request_window)
            expected = disagg_utils.get_dsv41_spec_layout(decode)
            self.assertEqual(
                expected["kv_layer_ids"],
                [key[1] for key, _ in self._transfer_args()[1][0]],
            )
            gather.assert_not_called()
            for args, _ in peers:
                self.assertEqual(disagg_utils.get_dsv41_spec_layout(args), expected)

    def test_dedup_replicated_indexer_pd_assigns_each_source_once_and_balances_ranks(self):
        import torch
        from sglang.srt.mem_cache.cp_cache_layer_split.deepseek_v4_pool import (
            CpCacheLayerSplitDeepSeekV4TokenToKVPool,
        )

        def make_fake_pool(cp_rank: int, cp_size: int, *, separate_scale: bool):
            pool = object.__new__(CpCacheLayerSplitDeepSeekV4TokenToKVPool)
            pool._init_cp_cache_layer_split(
                cp_rank=cp_rank,
                cp_size=cp_size,
                layer_shard_start_layer=0,
                layer_shard_layer_num=40,
            )
            pool._stage_start = 0
            pool._stage_end = 40
            pool._unified_kv = False
            pool.page_size = 256
            pool.compression_ratios = [0, 0] + [2] * 18 + [1] * 20
            pool.sources_by_ratio = {1: (20,), 2: (2, 8, 14)}
            pool.kv_pools = {}
            pool.index_pools = {}
            for ratio in (1, 2):
                owned_sources = [
                    s for s in pool.sources_by_ratio[ratio] if pool._is_layer_owned(s)
                ]
                kv_bufs = [
                    torch.zeros((4, 576 * (256 // ratio)), dtype=torch.uint8)
                    for _ in owned_sources
                ]
                pool.kv_pools[ratio] = SimpleNamespace(kv_buffer=kv_bufs)
                n_src = len(pool.sources_by_ratio[ratio])
                if separate_scale:
                    pool.index_pools[ratio] = SimpleNamespace(
                        page_size=64,
                        index_k_with_scale_buffer=None,
                        index_k_payload_buffer=[
                            torch.zeros((16, 64 * 64), dtype=torch.uint8)
                            for _ in range(n_src)
                        ],
                        index_k_scale_buffer=[
                            torch.zeros((16, 64 * 4), dtype=torch.uint8)
                            for _ in range(n_src)
                        ],
                    )
                else:
                    pool.index_pools[ratio] = SimpleNamespace(
                        page_size=64,
                        index_k_with_scale_buffer=[
                            torch.zeros((16, 64 * 68), dtype=torch.uint8)
                            for _ in range(n_src)
                        ],
                    )
            return pool

        for cp_size in (2, 4, 8):
            for separate_scale in (False, True):
                with self.subTest(cp_size=cp_size, separate_scale=separate_scale):
                    pools_list = [
                        make_fake_pool(r, cp_size, separate_scale=separate_scale)
                        for r in range(cp_size)
                    ]
                    seen = []
                    per_rank_counts = []
                    for pool in pools_list:
                        layout = pool.get_kv_transfer_layout()
                        ptrs, lens, item_lens = pool.get_contiguous_buf_infos()
                        self.assertEqual(len(layout), len(ptrs))
                        self.assertEqual(len(layout), len(lens))
                        self.assertEqual(len(layout), len(item_lens))
                        seen.extend(layout)
                        per_rank_counts.append(len(layout))

                    # Union covers every KV and indexer buffer with zero duplicates.
                    self.assertEqual(len(seen), len(set(seen)))
                    expected_keys = {
                        ("c1_kv", 20),
                        ("c2_kv", 2),
                        ("c2_kv", 8),
                        ("c2_kv", 14),
                        ("c1_indexer", 20),
                        ("c2_indexer", 2),
                        ("c2_indexer", 8),
                        ("c2_indexer", 14),
                    }
                    if separate_scale:
                        expected_keys |= {
                            ("c1_indexer_scale", 20),
                            ("c2_indexer_scale", 2),
                            ("c2_indexer_scale", 8),
                            ("c2_indexer_scale", 14),
                        }
                    self.assertEqual(set(seen), expected_keys)
                    if cp_size == 8 and not separate_scale:
                        # Every CP rank owns exactly 1 KV or indexer buffer.
                        self.assertEqual(per_rank_counts, [1] * 8)
                        self.assertEqual(
                            pools_list[3].get_kv_transfer_layout(),
                            [("c1_indexer", 20)],
                        )
                        self.assertEqual(
                            pools_list[4].get_kv_transfer_layout(),
                            [("c1_kv", 20)],
                        )

    def test_dedup_handshake_matches_unsharded_decode(self):
        # Simulate deduplicated CP8 where each replicated indexer is only on one rank.
        indexer_owner = {
            ("c1_indexer", 20): 3,
            ("c2_indexer", 2): 5,
            ("c2_indexer", 8): 6,
            ("c2_indexer", 14): 7,
        }
        peers = []
        for rank in range(8):
            args, entries = self._transfer_args(rank, request_window=True)
            filtered_kv = [
                (key, size)
                for key, size in entries[0]
                if key not in indexer_owner or indexer_owner[key] == rank
            ]
            args.kv_layer_ids = [key[1] for key, _ in filtered_kv]
            args.kv_item_lens = [size for _, size in filtered_kv]
            args.v4_transfer_metadata["kv"] = [key for key, _ in filtered_kv]
            peers.append((args, [filtered_kv, *entries[1:]]))

        gather = mock.Mock(return_value=[entries for _, entries in peers])
        with (
            mock.patch.object(
                disagg_utils,
                "get_spec",
                return_value=SimpleNamespace(
                    speculative_algorithm="DSPARK", speculative_num_draft_tokens=6
                ),
            ),
            mock.patch.object(
                disagg_utils,
                "get_parallel",
                return_value=SimpleNamespace(
                    attn_cp_group=SimpleNamespace(all_gather_object=gather)
                ),
                create=True,
            ),
        ):
            decode, _ = self._transfer_args(request_window=True)
            expected = disagg_utils.get_dsv41_spec_layout(decode)
            for args, _ in peers:
                self.assertEqual(disagg_utils.get_dsv41_spec_layout(args), expected)

    def test_layer_split_handshake_matches_unsharded_decode(self):
        self._check_handshake(request_window=False)

    def test_request_window_handshake_matches_unsharded_decode(self):
        self._check_handshake(request_window=True)

    def test_request_window_handshake_requires_paged_draft(self):
        args, _ = self._transfer_args(request_window=True)
        args.state_types[-1] = StateType.DSV4_REQUEST_STATE
        with (
            mock.patch.object(disagg_utils, "get_spec", return_value=SimpleNamespace(
                speculative_algorithm="DSPARK", speculative_num_draft_tokens=6,
            )),
            self.assertRaisesRegex(RuntimeError, "target and draft SWA state"),
        ):
            disagg_utils.get_dsv41_spec_layout(args)

    def test_layer_split_handshake_rejects_page_abi_mismatch(self):
        peers = [self._transfer_args(rank) for rank in range(8)]
        # The replicated indexer for layer 20 must have one identical ABI.
        key, size = peers[1][1][0][0]
        peers[1][1][0][0] = (key, size + 1)
        with (
            mock.patch.object(
                disagg_utils,
                "get_spec",
                return_value=SimpleNamespace(
                    speculative_algorithm="DSPARK", speculative_num_draft_tokens=6
                ),
            ),
            mock.patch.object(
                disagg_utils,
                "get_parallel",
                return_value=SimpleNamespace(
                    attn_cp_group=SimpleNamespace(
                        all_gather_object=mock.Mock(
                            return_value=[entries for _, entries in peers]
                        )
                    )
                ),
                create=True,
            ),
            self.assertRaisesRegex(RuntimeError, "page ABI differs"),
        ):
            disagg_utils.get_dsv41_spec_layout(peers[0][0])


if __name__ == "__main__":
    unittest.main()
