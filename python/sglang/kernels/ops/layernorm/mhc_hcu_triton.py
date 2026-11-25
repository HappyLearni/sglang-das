"""Single-pass split-K Triton MFMA BF16x3 mHC mix_stats kernel for HCU."""

from __future__ import annotations

from collections import OrderedDict
import threading
from typing import Sequence, Tuple
import weakref

import torch
import triton
import triton.language as tl

_N_PAD = 128
_HC_MULT = 4
_NUM_OUTPUTS = 24
_WEIGHT_CACHE_MAX_ENTRIES = 512
_weight_cache_lock = threading.Lock()
_weight_cache: OrderedDict[tuple, torch.Tensor] = OrderedDict()


@triton.jit
def _hc_bf16x3_proj_sq_splitk_kernel(
    X_ptr,
    W_ptr,
    Partial_ptr,
    Sq_ptr,
    M,
    K: tl.constexpr,
    N_PAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    offs_n = tl.arange(0, N_PAD)

    k_per_split = K // SPLIT_K
    k_start = pid_k * k_per_split
    k_end = k_start + k_per_split

    acc = tl.zeros((BLOCK_M, N_PAD), dtype=tl.float32)
    sq = tl.zeros((BLOCK_M,), dtype=tl.float32)

    m_i64 = offs_m.to(tl.int64)
    for k in range(k_start, k_end, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        x = tl.load(
            X_ptr + m_i64[:, None] * K + offs_k[None, :],
            mask=mask_m[:, None],
            other=0.0,
        )
        w = tl.load(W_ptr + offs_k[:, None] * N_PAD + offs_n[None, :])
        acc = tl.dot(x, w, acc, out_dtype=tl.float32)
        xf = x.to(tl.float32)
        sq += tl.sum(xf * xf, axis=1)

    pid_k_i64 = pid_k.to(tl.int64)
    m_total_i64 = tl.full((), M, tl.int64)
    partial_base = (pid_k_i64 * m_total_i64 + m_i64[:, None]) * N_PAD
    tl.store(Partial_ptr + partial_base + offs_n[None, :], acc, mask=mask_m[:, None])
    tl.store(Sq_ptr + pid_k_i64 * m_total_i64 + m_i64, sq, mask=mask_m)


@triton.jit
def _hc_bf16x3_reduce_sinkhorn_kernel(
    Partial_ptr,
    Sq_ptr,
    S_ptr,
    B_ptr,
    PRE_ptr,
    POST_ptr,
    COMB_ptr,
    M,
    K: tl.constexpr,
    N_PAD: tl.constexpr,
    SPLIT_K: tl.constexpr,
    RMS_EPS: tl.constexpr,
    HC_EPS: tl.constexpr,
    SINKHORN_ITERS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    m_total = tl.full((), M, tl.int64)

    sq_sum = 0.0
    idx4 = tl.arange(0, 4)
    pr = tl.zeros((4,), dtype=tl.float32)
    po = tl.zeros((4,), dtype=tl.float32)
    r4 = tl.arange(0, 4)[:, None]
    c4 = tl.arange(0, 4)[None, :]
    idx16 = 8 + r4 * 4 + c4
    cm_p = tl.zeros((4, 4), dtype=tl.float32)

    for sk in range(SPLIT_K):
        sk_i64 = tl.full((), sk, tl.int64)
        sq_sum += tl.load(Sq_ptr + sk_i64 * m_total + row)
        base_p = Partial_ptr + (sk_i64 * m_total + row) * N_PAD
        pr += tl.load(base_p + idx4) + tl.load(base_p + 24 + idx4) + tl.load(base_p + 48 + idx4)
        po += tl.load(base_p + 4 + idx4) + tl.load(base_p + 28 + idx4) + tl.load(base_p + 52 + idx4)
        cm_p += tl.load(base_p + idx16) + tl.load(base_p + 24 + idx16) + tl.load(base_p + 48 + idx16)

    rsqrt = tl.rsqrt(sq_sum / K + RMS_EPS)
    s0 = tl.load(S_ptr + 0)
    s1 = tl.load(S_ptr + 1)
    s2 = tl.load(S_ptr + 2)

    pre = tl.sigmoid(pr * rsqrt * s0 + tl.load(B_ptr + idx4)) + HC_EPS
    post = 2.0 * tl.sigmoid(po * rsqrt * s1 + tl.load(B_ptr + 4 + idx4))
    cm = cm_p * rsqrt * s2 + tl.load(B_ptr + idx16)
    cm = tl.exp(cm - tl.max(cm, axis=1)[:, None])
    # Match hc_split_sinkhorn_kernel: always run the initial row/col pass, then
    # max(SINKHORN_ITERS - 1, 0) additional passes (including SINKHORN_ITERS == 0).
    cm = cm / tl.sum(cm, axis=1)[:, None] + HC_EPS
    cm = cm / (tl.sum(cm, axis=0)[None, :] + HC_EPS)
    if SINKHORN_ITERS > 1:
        for _ in tl.static_range(SINKHORN_ITERS - 1):
            cm = cm / (tl.sum(cm, axis=1)[:, None] + HC_EPS)
            cm = cm / (tl.sum(cm, axis=0)[None, :] + HC_EPS)

    tl.store(PRE_ptr + row * 4 + idx4, pre)
    tl.store(POST_ptr + row * 4 + idx4, post)
    tl.store(COMB_ptr + row * 16 + r4 * 4 + c4, cm)


def _tensor_version(t: torch.Tensor) -> int:
    try:
        return t._version
    except RuntimeError:
        return -1


def _get_padded_bf16x3_weight(bf16_parts: Sequence[torch.Tensor]) -> torch.Tensor:
    key = (
        bf16_parts[0].device.index,
        tuple((p.data_ptr(), tuple(p.shape), _tensor_version(p)) for p in bf16_parts),
    )
    storages = tuple(p.untyped_storage() for p in bf16_parts)
    cached = _weight_cache.get(key)
    if cached is not None:
        refs, w_pad = cached
        if all(r() is s for r, s in zip(refs, storages)):
            return w_pad
    with _weight_cache_lock:
        cached = _weight_cache.get(key)
        if cached is not None:
            refs, w_pad = cached
            if all(r() is s for r, s in zip(refs, storages)):
                return w_pad
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "HCU Triton BF16x3 mHC weight cache must be initialized before graph capture"
            )
        k = bf16_parts[0].shape[1]
        w_pad = torch.zeros(
            (k, _N_PAD), device=bf16_parts[0].device, dtype=torch.bfloat16
        )
        w_pad[:, : 3 * _NUM_OUTPUTS] = torch.cat([p.t() for p in bf16_parts], dim=1)
        _weight_cache[key] = (tuple(weakref.ref(s) for s in storages), w_pad)
        if len(_weight_cache) > _WEIGHT_CACHE_MAX_ENTRIES:
            _weight_cache.popitem(last=False)
        return w_pad


def _select_launch_config(m: int, k: int) -> Tuple[int, int, int, int] | None:
    """Return (block_m, block_k, split_k, num_warps) for HCU gfx936 MFMA."""
    candidates = (
        (128, 64, 8, 8),
        (64, 128, 8, 4),
        (64, 64, 8, 4),
        (64, 64, 4, 4),
        (64, 64, 1, 4),
    )
    for block_m, block_k, split_k, num_warps in candidates:
        if k % (block_k * split_k) == 0:
            return block_m, block_k, split_k, num_warps
    return None


def can_use_hc_mix_stats_sinkhorn_bf16x3_hcu(
    x_flat: torch.Tensor,
    bf16_parts: object,
    mult: int,
) -> bool:
    if (
        mult != _HC_MULT
        or not x_flat.is_cuda
        or x_flat.ndim != 2
        or x_flat.dtype != torch.bfloat16
        or not x_flat.is_contiguous()
        or not isinstance(bf16_parts, (tuple, list))
        or len(bf16_parts) != 3
    ):
        return False
    m, k = x_flat.shape
    if m == 0 or _select_launch_config(m, k) is None:
        return False
    return all(
        isinstance(p, torch.Tensor)
        and p.device == x_flat.device
        and p.dtype == torch.bfloat16
        and p.shape == (_NUM_OUTPUTS, k)
        for p in bf16_parts
    )


def hc_mix_stats_sinkhorn_bf16x3_hcu(
    x_flat: torch.Tensor,
    bf16_parts: Sequence[torch.Tensor],
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    sinkhorn_iters: int,
    rms_eps: float,
    hc_eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute mHC (pre, post, comb) via single-pass split-K Triton MFMA BF16x3."""
    m, k = x_flat.shape
    cfg = _select_launch_config(m, k)
    if cfg is None:
        raise ValueError(f"Unsupported K={k} for HCU Triton BF16x3 mHC kernel")
    block_m, block_k, split_k, num_warps = cfg
    w_pad = _get_padded_bf16x3_weight(bf16_parts)
    scale = hc_scale.contiguous()
    base = hc_base.contiguous()

    device = x_flat.device
    partial = torch.empty((split_k, m, _N_PAD), device=device, dtype=torch.float32)
    sq = torch.empty((split_k, m), device=device, dtype=torch.float32)
    pre = torch.empty((m, _HC_MULT), device=device, dtype=torch.float32)
    post = torch.empty((m, _HC_MULT), device=device, dtype=torch.float32)
    comb = torch.empty((m, _HC_MULT, _HC_MULT), device=device, dtype=torch.float32)

    _hc_bf16x3_proj_sq_splitk_kernel[(triton.cdiv(m, block_m), split_k)](
        x_flat,
        w_pad,
        partial,
        sq,
        m,
        K=k,
        N_PAD=_N_PAD,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        SPLIT_K=split_k,
        num_warps=num_warps,
        num_stages=2,
    )
    _hc_bf16x3_reduce_sinkhorn_kernel[(m,)](
        partial,
        sq,
        scale,
        base,
        pre,
        post,
        comb,
        m,
        K=k,
        N_PAD=_N_PAD,
        SPLIT_K=split_k,
        RMS_EPS=rms_eps,
        HC_EPS=hc_eps,
        SINKHORN_ITERS=sinkhorn_iters,
        num_warps=1,
    )
    return pre, post, comb
