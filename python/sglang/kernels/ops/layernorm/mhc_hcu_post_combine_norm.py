# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Single-pass Triton fused mHC post + hc_combine + RMSNorm for HCU prefill.

Eliminates the intermediate HBM round-trips between ``mhc_post`` (writes
``R'[M, 4, H]``), ``hc_combine`` (re-reads ``R'``, writes ``h[M, H]``), and
``RMSNorm`` (re-reads ``h``) for ``H=5120, mult=4`` BF16 prefill batches.
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
from sglang.srt.utils import get_bool_env_var, is_hcu

_HCU_MHC_MIN_TOKENS = 512
_HCU_MHC_MAX_TOKENS = 65536
_HCU_MHC_HIDDEN = 5120
_HCU_MHC_MULT = 4


def is_hcu_mhc_post_combine_norm_enabled() -> bool:
    return (
        is_hcu()
        and get_bool_env_var("SGLANG_HCU_FUSED_MHC_POST_COMBINE_NORM", "true")
        and not is_batch_invariant_mode_enabled()
    )


def can_use_hcu_mhc_post_combine_norm(
    cfg,
    y: torch.Tensor,
    residual: torch.Tensor,
    next_boundary,
) -> bool:
    return (
        is_hcu_mhc_post_combine_norm_enabled()
        and next_boundary is not None
        and next_boundary.norm_fusable
        and cfg.pre_from_prev
        and cfg.hidden == _HCU_MHC_HIDDEN
        and cfg.mult == _HCU_MHC_MULT
        and _HCU_MHC_MIN_TOKENS <= y.shape[0] <= _HCU_MHC_MAX_TOKENS
        and y.is_cuda
        and residual.is_cuda
        and y.dtype == torch.bfloat16
        and residual.dtype == torch.bfloat16
        and y.stride(-1) == 1
        and residual.stride(-1) == 1
    )


@triton.jit
def _mhc_post_combine_norm_hcu_kernel(
    x_ptr,
    res_ptr,
    post_ptr,
    comb_ptr,
    pre_ptr,
    w_ptr,
    res_out_ptr,
    norm_out_ptr,
    stride_xm,
    stride_rm,
    stride_rk,
    stride_pm,
    stride_pk,
    stride_cm,
    stride_cj,
    stride_ci,
    stride_prem,
    stride_prek,
    stride_rom,
    stride_rok,
    stride_ym,
    eps,
    H: tl.constexpr,
    H_PAD: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, H_PAD)
    mask = offs < H

    # Load per-row coefficients (4 streams, 4x4 sinkhorn matrix).
    p0 = tl.load(post_ptr + row * stride_pm + 0 * stride_pk).to(tl.float32)
    p1 = tl.load(post_ptr + row * stride_pm + 1 * stride_pk).to(tl.float32)
    p2 = tl.load(post_ptr + row * stride_pm + 2 * stride_pk).to(tl.float32)
    p3 = tl.load(post_ptr + row * stride_pm + 3 * stride_pk).to(tl.float32)

    pr0 = tl.load(pre_ptr + row * stride_prem + 0 * stride_prek).to(tl.float32)
    pr1 = tl.load(pre_ptr + row * stride_prem + 1 * stride_prek).to(tl.float32)
    pr2 = tl.load(pre_ptr + row * stride_prem + 2 * stride_prek).to(tl.float32)
    pr3 = tl.load(pre_ptr + row * stride_prem + 3 * stride_prek).to(tl.float32)

    c_base = comb_ptr + row * stride_cm
    c00 = tl.load(c_base + 0 * stride_cj + 0 * stride_ci).to(tl.float32)
    c01 = tl.load(c_base + 0 * stride_cj + 1 * stride_ci).to(tl.float32)
    c02 = tl.load(c_base + 0 * stride_cj + 2 * stride_ci).to(tl.float32)
    c03 = tl.load(c_base + 0 * stride_cj + 3 * stride_ci).to(tl.float32)

    c10 = tl.load(c_base + 1 * stride_cj + 0 * stride_ci).to(tl.float32)
    c11 = tl.load(c_base + 1 * stride_cj + 1 * stride_ci).to(tl.float32)
    c12 = tl.load(c_base + 1 * stride_cj + 2 * stride_ci).to(tl.float32)
    c13 = tl.load(c_base + 1 * stride_cj + 3 * stride_ci).to(tl.float32)

    c20 = tl.load(c_base + 2 * stride_cj + 0 * stride_ci).to(tl.float32)
    c21 = tl.load(c_base + 2 * stride_cj + 1 * stride_ci).to(tl.float32)
    c22 = tl.load(c_base + 2 * stride_cj + 2 * stride_ci).to(tl.float32)
    c23 = tl.load(c_base + 2 * stride_cj + 3 * stride_ci).to(tl.float32)

    c30 = tl.load(c_base + 3 * stride_cj + 0 * stride_ci).to(tl.float32)
    c31 = tl.load(c_base + 3 * stride_cj + 1 * stride_ci).to(tl.float32)
    c32 = tl.load(c_base + 3 * stride_cj + 2 * stride_ci).to(tl.float32)
    c33 = tl.load(c_base + 3 * stride_cj + 3 * stride_ci).to(tl.float32)

    x = tl.load(x_ptr + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    r_base = res_ptr + row * stride_rm
    r0 = tl.load(r_base + 0 * stride_rk + offs, mask=mask, other=0.0).to(tl.float32)
    r1 = tl.load(r_base + 1 * stride_rk + offs, mask=mask, other=0.0).to(tl.float32)
    r2 = tl.load(r_base + 2 * stride_rk + offs, mask=mask, other=0.0).to(tl.float32)
    r3 = tl.load(r_base + 3 * stride_rk + offs, mask=mask, other=0.0).to(tl.float32)

    ro_base = res_out_ptr + row * stride_rom

    # Compute each updated stream R'[i], round to bf16 before store + combine to
    # preserve the exact intermediate bf16 semantics of the unfused path.
    u0 = (p0 * x + c00 * r0 + c10 * r1 + c20 * r2 + c30 * r3).to(tl.bfloat16)
    tl.store(ro_base + 0 * stride_rok + offs, u0, mask=mask)
    h = pr0 * u0.to(tl.float32)

    u1 = (p1 * x + c01 * r0 + c11 * r1 + c21 * r2 + c31 * r3).to(tl.bfloat16)
    tl.store(ro_base + 1 * stride_rok + offs, u1, mask=mask)
    h += pr1 * u1.to(tl.float32)

    u2 = (p2 * x + c02 * r0 + c12 * r1 + c22 * r2 + c32 * r3).to(tl.bfloat16)
    tl.store(ro_base + 2 * stride_rok + offs, u2, mask=mask)
    h += pr2 * u2.to(tl.float32)

    u3 = (p3 * x + c03 * r0 + c13 * r1 + c23 * r2 + c33 * r3).to(tl.bfloat16)
    tl.store(ro_base + 3 * stride_rok + offs, u3, mask=mask)
    h += pr3 * u3.to(tl.float32)

    # Match hc_combine's bf16 output rounding before RMSNorm.
    h_bf16 = h.to(tl.bfloat16)
    h_f32 = tl.where(mask, h_bf16.to(tl.float32), 0.0)
    var = tl.sum(h_f32 * h_f32, axis=0) / H
    inv_rms = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (h_f32 * inv_rms * w).to(tl.bfloat16)
    tl.store(norm_out_ptr + row * stride_ym + offs, y, mask=mask)


def mhc_post_combine_norm_hcu(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    pre: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused mHC post + hc_combine + RMSNorm for ``H=5120, mult=4`` BF16 rows.

    Returns ``(updated_residual [M, 4, 5120], normalized [M, 5120])``.
    """
    m, h = x.shape
    assert h == _HCU_MHC_HIDDEN, f"Expected hidden={_HCU_MHC_HIDDEN}, got {h}"
    assert residual.shape == (m, _HCU_MHC_MULT, h)
    assert post.shape == (m, _HCU_MHC_MULT)
    assert comb.shape == (m, _HCU_MHC_MULT, _HCU_MHC_MULT)
    assert pre.shape == (m, _HCU_MHC_MULT)
    assert weight.shape == (h,) and weight.is_contiguous()
    assert x.dtype == torch.bfloat16 and residual.dtype == torch.bfloat16
    assert weight.dtype == torch.bfloat16
    assert x.stride(-1) == 1 and residual.stride(-1) == 1

    updated = torch.empty((m, _HCU_MHC_MULT, h), dtype=torch.bfloat16, device=x.device)
    normalized = torch.empty((m, h), dtype=torch.bfloat16, device=x.device)
    if m == 0:
        return updated, normalized

    _mhc_post_combine_norm_hcu_kernel[(m,)](
        x,
        residual,
        post,
        comb,
        pre,
        weight,
        updated,
        normalized,
        x.stride(0),
        residual.stride(0),
        residual.stride(1),
        post.stride(0),
        post.stride(1),
        comb.stride(0),
        comb.stride(1),
        comb.stride(2),
        pre.stride(0),
        pre.stride(1),
        updated.stride(0),
        updated.stride(1),
        normalized.stride(0),
        float(eps),
        H=_HCU_MHC_HIDDEN,
        H_PAD=8192,
        num_warps=4,
    )
    return updated, normalized
