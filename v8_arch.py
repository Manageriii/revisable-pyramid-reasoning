"""RPR v8 Architectural Extensions & MuonClip Optimizer (`2.65B -> 2.90B` Total / `2.65B` Active):

1. `EngramConditionalMemory` (DeepSeek-V4 O(1) Hashed 2-Gram Conditional Memory, `+0.255B` params across L6 & L18):
   - Offloads lexical/API/mathematical n-gram idioms into two `61,440 x 2,048` `bfloat16` conditional lookup banks (`251.66M`
     table params) modulated by a context-aware SiLU gate and zero-initialized `out_scale` (`Step-0 logit diff = 0.000000`).
   - Uses `cache.offset`-indexed boundary token tracking so `L=N` prefill, `L=1` decode, and `L=1+K` MTP speculative
     rollbacks are 100% bit-identical.

2. `CrossLayerAttnRes` (Kimi K3 / K3.1 Pro Cross-Layer Attention Residuals on upper `full_attention` layers `26, 30, 32, 34`):
   - Computes a depth-axis softmax read over early/mid foundation checkpoints `(2, 8, 14, 20)` with zero-initialized
     `res_scale`, preventing `39k`-token middle logs from diluting Turn-1 constraints and option tables.

3. `MoBAPinnedSparseAttention` (Kimi MoBA + DeepSeek CSA2 Pinned-Sink + Top-K Block Attention for `L_kv > 4096`):
   - Pins Turn-1 foundation tokens (`[0 : 640]`) + causal local window (`1,024` tokens) + parameter-free Top-K (`K = 6`
     blocks of `256` tokens = `1,536` tokens) block-routed historical KV entries per KV head, keeping long-context
     attention sharp across `8k..128k` tokens.

4. `MuonClipOptimizer` (Kimi K2 / K3.1 Pro 5-Step Quintic Newton-Schulz Orthogonalized Momentum + AdamW + QK-Clip):
   - Applies 5-step Newton-Schulz orthogonalization (`a=3.4445, b=-4.7750, c=2.0315`) with RMS `0.2 * sqrt(max(m, n))`
     scaling to dense 2D projection/LoRA matrices, `AdamW` to 1D gates/norms and sparse `Engram` embedding rows, plus
     post-step `QK-Clip` spectral bounding on `q_proj` / `k_proj`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map


ENGRAM_LAYERS: tuple[int, ...] = (6, 18)
ATTN_RES_SOURCES: tuple[int, ...] = (2, 8, 14, 20)
ATTN_RES_TARGETS: tuple[int, ...] = (26, 30, 32, 34)
ATTN_RES_TARGETS_40L: tuple[int, ...] = (26, 30, 32, 36, 38)
UPCYCLED_40L_INDICES: tuple[int, ...] = (34, 35, 38, 39)
LATENT_RECUR_LAYERS_V12: tuple[int, ...] = (29, 35)
THOUGHT_ANCHOR_TEMP_LAYERS_40L: tuple[int, ...] = (26, 30, 32, 34, 36, 38)
DEFAULT_ENGRAM_BUCKETS = 61440  # 2 * 61,440 * 2,048 = 251,658,240 table params (+0.255B total with gates & AttnRes)


# ===========================================================================
# 1. DeepSeek-V4 Style O(1) Hashed 2-Gram Conditional Memory (`+0.255B`)
# ===========================================================================


@mx.compile
def _engram_fuse_compiled(
    h: mx.array,
    curr_ids: mx.array,
    prev_ids: mx.array,
    prime_prev: int,
    prime_curr: int,
    num_buckets: int,
    table_w: mx.array,
    norm_w: mx.array,
    norm_eps: float,
    w_down: mx.array,
    w_up: mx.array,
    out_scale: mx.array,
) -> mx.array:
    c64 = curr_ids.astype(mx.int64)
    p64 = prev_ids.astype(mx.int64)
    mixed = mx.abs(p64 * prime_prev + c64 * prime_curr + (p64 ^ (c64 * 40503)))
    b_ids = mx.remainder(mixed, num_buckets).astype(mx.int32)
    mem = table_w[b_ids].astype(h.dtype)
    h_n = mx.fast.rms_norm(h, norm_w, norm_eps)
    gate = mx.sigmoid(nn.silu(h_n @ w_down.T) @ w_up.T).astype(h.dtype)
    return h + (mem * gate * out_scale.astype(h.dtype)).astype(h.dtype)


class EngramConditionalMemory(nn.Module):
    """O(1) hashed 2-gram conditional memory lookup with context-dependent SiLU gating & zero-init residual scale."""

    def __init__(
        self,
        hidden_size: int = 2048,
        num_buckets: int = DEFAULT_ENGRAM_BUCKETS,
        gate_rank: int = 256,
        prime_prev: int = 1315423,
        prime_curr: int = 2654435,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_buckets = num_buckets
        self.prime_prev = prime_prev
        self.prime_curr = prime_curr
        self.norm_eps = norm_eps

        self.table = nn.Embedding(num_buckets, hidden_size)
        # Small initial scale on lookup vectors; exact Step-0 zero output guaranteed by `out_scale = 0`!
        self.table.weight = (mx.random.normal((num_buckets, hidden_size)) * 0.02).astype(mx.bfloat16)

        self.ctx_norm = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.gate_down = nn.Linear(hidden_size, gate_rank, bias=False)
        self.gate_up = nn.Linear(gate_rank, hidden_size, bias=False)
        self.out_scale = mx.zeros((hidden_size,), dtype=mx.bfloat16)

    def compute_bucket_ids(self, curr_ids: mx.array, prev_ids: mx.array) -> mx.array:
        c64 = curr_ids.astype(mx.int64)
        p64 = prev_ids.astype(mx.int64)
        mixed = mx.abs(p64 * self.prime_prev + c64 * self.prime_curr + (p64 ^ (c64 * 40503)))
        return mx.remainder(mixed, self.num_buckets).astype(mx.int32)

    def __call__(self, h: mx.array, curr_ids: mx.array, prev_ids: mx.array) -> mx.array:
        if not self.training:
            return _engram_fuse_compiled(
                h,
                curr_ids,
                prev_ids,
                self.prime_prev,
                self.prime_curr,
                self.num_buckets,
                self.table.weight,
                self.ctx_norm.weight,
                self.norm_eps,
                self.gate_down.weight,
                self.gate_up.weight,
                self.out_scale,
            )
        bucket_ids = self.compute_bucket_ids(curr_ids, prev_ids)
        mem = self.table(bucket_ids).astype(h.dtype)
        ctx_gate = mx.sigmoid(self.gate_up(nn.silu(self.gate_down(self.ctx_norm(h))))).astype(h.dtype)
        gated = (mem * ctx_gate * self.out_scale.astype(h.dtype)).astype(h.dtype)
        return h + gated


# ===========================================================================
# 2. Kimi K3 / K3.1 Pro Cross-Layer Attention Residuals (`AttnRes`)
# ===========================================================================


@mx.compile
def _attn_res_fuse4_compiled(
    h_curr: mx.array,
    s0: mx.array,
    s1: mx.array,
    s2: mx.array,
    s3: mx.array,
    q_norm_w: mx.array,
    src_norm_w: mx.array,
    norm_eps: float,
    wq: mx.array,
    wk: mx.array,
    wv_down: mx.array,
    wv_up: mx.array,
    res_scale: mx.array,
    scale: float,
) -> mx.array:
    src_stack = mx.stack(
        [s0.astype(h_curr.dtype), s1.astype(h_curr.dtype), s2.astype(h_curr.dtype), s3.astype(h_curr.dtype)],
        axis=2,
    )
    src_normed = mx.fast.rms_norm(src_stack, src_norm_w, norm_eps)
    q_n = mx.fast.rms_norm(h_curr, q_norm_w, norm_eps)
    q = (q_n @ wq.T)[..., None, :]
    k = src_normed @ wk.T
    scores = mx.matmul(q.astype(mx.float32), k.astype(mx.float32).transpose(0, 1, 3, 2)) * scale
    weights = mx.softmax(scores, axis=-1).astype(h_curr.dtype)
    blended = mx.matmul(weights, src_normed).squeeze(2)
    delta = ((nn.silu(blended @ wv_down.T) @ wv_up.T).astype(h_curr.dtype) * res_scale.astype(h_curr.dtype)).astype(
        h_curr.dtype
    )
    return h_curr + delta


class CrossLayerAttnRes(nn.Module):
    """Depth-axis softmax attention over early/mid foundation checkpoints `[h_2, h_8, h_14, h_20]` with zero-init gate."""

    def __init__(
        self,
        hidden_size: int = 2048,
        num_sources: int = len(ATTN_RES_SOURCES),
        bottleneck_dim: int = 256,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_sources = num_sources
        self.bottleneck_dim = bottleneck_dim
        self.norm_eps = norm_eps
        self.scale = bottleneck_dim**-0.5

        self.q_norm = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.src_norm = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.q_proj = nn.Linear(hidden_size, bottleneck_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, bottleneck_dim, bias=False)
        self.v_down = nn.Linear(hidden_size, bottleneck_dim, bias=False)
        self.v_up = nn.Linear(bottleneck_dim, hidden_size, bias=False)
        self.res_scale = mx.zeros((hidden_size,), dtype=mx.bfloat16)

    def __call__(self, h_curr: mx.array, source_states: list[mx.array]) -> mx.array:
        if not self.training and len(source_states) == 4:
            return _attn_res_fuse4_compiled(
                h_curr,
                source_states[0],
                source_states[1],
                source_states[2],
                source_states[3],
                self.q_norm.weight,
                self.src_norm.weight,
                self.norm_eps,
                self.q_proj.weight,
                self.k_proj.weight,
                self.v_down.weight,
                self.v_up.weight,
                self.res_scale,
                self.scale,
            )
        # `h_curr`: [B, L, D]; `source_states`: list of S tensors [B, L, D] -> stack [B, L, S, D]
        src_stack = mx.stack([s.astype(h_curr.dtype) for s in source_states], axis=2)
        src_normed = self.src_norm(src_stack)
        q = self.q_proj(self.q_norm(h_curr))[..., None, :]  # [B, L, 1, r]
        k = self.k_proj(src_normed)  # [B, L, S, r]
        scores = mx.matmul(q.astype(mx.float32), k.astype(mx.float32).transpose(0, 1, 3, 2)) * self.scale
        weights = mx.softmax(scores, axis=-1).astype(h_curr.dtype)  # [B, L, 1, S]
        blended = mx.matmul(weights, src_normed).squeeze(2)  # [B, L, D]
        delta = (self.v_up(nn.silu(self.v_down(blended))).astype(h_curr.dtype) * self.res_scale.astype(h_curr.dtype)).astype(h_curr.dtype)
        return h_curr + delta


# ===========================================================================
# 2B. Geiping et al. (2025/2026) Latent Recurrent Depth Block (`v10` 40L)
#     & LongRoPE2 / YaRN Continuous 256k Dual-Path Rotary Embedding
# ===========================================================================


@mx.compile
def _latent_recur_step_compiled(
    h: mx.array,
    anchor: mx.array,
    norm_w: mx.array,
    anc_norm_w: mx.array,
    norm_eps: float,
    w_gate: mx.array,
    w_up: mx.array,
    w_anc: mx.array,
    w_down: mx.array,
    step_bias: mx.array,
    loop_scale: mx.array,
) -> mx.array:
    h_n = mx.fast.rms_norm(h, norm_w, norm_eps)
    a_n = mx.fast.rms_norm(anchor.astype(h.dtype), anc_norm_w, norm_eps)
    g = nn.silu(h_n @ w_gate.T + step_bias.astype(h.dtype))
    u = (h_n @ w_up.T) + (a_n @ w_anc.T)
    delta = ((g * u) @ w_down.T) * loop_scale.astype(h.dtype)
    return h + delta.astype(h.dtype)


@mx.compile
def _latent_recur_entry_step_compiled(
    h: mx.array,
    entry_proj: mx.array,
    anc_proj: mx.array,
    norm_w: mx.array,
    norm_eps: float,
    w_gate: mx.array,
    w_up: mx.array,
    w_down: mx.array,
    step_bias: mx.array,
    loop_scale: mx.array,
) -> mx.array:
    h_n = mx.fast.rms_norm(h, norm_w, norm_eps)
    g = nn.silu(h_n @ w_gate.T + step_bias.astype(h.dtype))
    u = (h_n @ w_up.T) + entry_proj + anc_proj
    delta = ((g * u) @ w_down.T) * loop_scale.astype(h.dtype)
    return h + delta.astype(h.dtype)


class ThoughtAnchorHeadTemp(nn.Module):
    """Per-head learnable attention inverse-temperature log-scale (`Thermometer of Thoughts` + `Thought Anchors`).
    Initialized to zeros (`exp(0) == 1.000000` exact Step-0 identity) so upper receiver heads can sharpen
    onto pivotal `<think>` invariants (`[L0]`, `[L1]`, `[BACKTRACK]`) while exploratory heads stay broad.
    """

    def __init__(self, num_heads: int = 32) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.log_temp = mx.zeros((num_heads,), dtype=mx.bfloat16)

    def scale_queries(self, q: mx.array) -> mx.array:
        s = mx.exp(mx.clip(self.log_temp.astype(mx.float32), -0.45, 0.45)).astype(q.dtype)
        return q * s[None, :, None, None]


class LatentReasoningRecurrence(nn.Module):
    """Shared-weight multi-step continuous latent reasoning recurrence (`Geiping et al. Huginn Recurrent Depth`):
    iteratively refines hidden states against BOTH the pre-loop entry state (`entry_inject=True`) and the
    early-layer foundation anchor before synthesis layers, with zero-initialized `loop_scales` for exact
    `Step-0` identity preservation (`diff = 0.000000`).
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        latent_ffn_dim: int = 2560,
        num_loops: int = 2,
        norm_eps: float = 1e-5,
        entry_inject: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_ffn_dim = latent_ffn_dim
        self.num_loops = num_loops
        self.norm_eps = norm_eps
        self.entry_inject = entry_inject
        self.h_norm = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.anc_norm = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.w_gate = nn.Linear(hidden_size, latent_ffn_dim, bias=False)
        self.w_up = nn.Linear(hidden_size, latent_ffn_dim, bias=False)
        self.w_anc = nn.Linear(hidden_size, latent_ffn_dim, bias=False)
        self.w_down = nn.Linear(latent_ffn_dim, hidden_size, bias=False)
        if entry_inject:
            self.entry_norm = nn.RMSNorm(hidden_size, eps=norm_eps)
            self.w_entry = nn.Linear(hidden_size, latent_ffn_dim, bias=False)
        self.step_embed = mx.zeros((num_loops, latent_ffn_dim), dtype=mx.bfloat16)
        self.loop_scales = mx.zeros((num_loops, hidden_size), dtype=mx.bfloat16)

    def enable_entry_inject(self, zero_init_proj: bool = True) -> None:
        if not getattr(self, "entry_inject", False):
            self.entry_inject = True
            self.entry_norm = nn.RMSNorm(self.hidden_size, eps=self.norm_eps)
            self.entry_norm.weight = self.entry_norm.weight.astype(mx.bfloat16)
            self.w_entry = nn.Linear(self.hidden_size, self.latent_ffn_dim, bias=False)
            if zero_init_proj:
                self.w_entry.weight = mx.zeros((self.latent_ffn_dim, self.hidden_size), dtype=mx.bfloat16)
            else:
                self.w_entry.weight = self.w_entry.weight.astype(mx.bfloat16)

    def __call__(self, h: mx.array, source_states: list[mx.array]) -> mx.array:
        anchor = (
            (source_states[0].astype(h.dtype) + source_states[1].astype(h.dtype) + source_states[2].astype(h.dtype) + source_states[3].astype(h.dtype))
            * mx.array(0.25, dtype=h.dtype)
        )
        cur = h
        has_entry = bool(getattr(self, "entry_inject", False) and hasattr(self, "w_entry"))
        if not self.training:
            if has_entry:
                e_n = mx.fast.rms_norm(h, self.entry_norm.weight, self.norm_eps)
                entry_proj = e_n @ self.w_entry.weight.T
                a_n = mx.fast.rms_norm(anchor.astype(h.dtype), self.anc_norm.weight, self.norm_eps)
                anc_proj = a_n @ self.w_anc.weight.T
                for t in range(self.num_loops):
                    cur = _latent_recur_entry_step_compiled(
                        cur,
                        entry_proj,
                        anc_proj,
                        self.h_norm.weight,
                        self.norm_eps,
                        self.w_gate.weight,
                        self.w_up.weight,
                        self.w_down.weight,
                        self.step_embed[t],
                        self.loop_scales[t],
                    )
                return cur
            for t in range(self.num_loops):
                cur = _latent_recur_step_compiled(
                    cur,
                    anchor,
                    self.h_norm.weight,
                    self.anc_norm.weight,
                    self.norm_eps,
                    self.w_gate.weight,
                    self.w_up.weight,
                    self.w_anc.weight,
                    self.w_down.weight,
                    self.step_embed[t],
                    self.loop_scales[t],
                )
            return cur
        a_n = self.anc_norm(anchor)
        anc_proj = a_n @ self.w_anc.weight.T
        entry_proj = (self.entry_norm(h) @ self.w_entry.weight.T) if has_entry else None
        for t in range(self.num_loops):
            h_n = self.h_norm(cur)
            g = nn.silu(self.w_gate(h_n) + self.step_embed[t].astype(cur.dtype))
            u = self.w_up(h_n) + anc_proj
            if entry_proj is not None:
                u = u + entry_proj
            delta = self.w_down(g * u) * self.loop_scales[t].astype(cur.dtype)
            cur = cur + delta.astype(cur.dtype)
        return cur


class LongRoPE2YaRNRotary(nn.Module):
    """Continuous Dual-Path 256k (`262,144`) RoPE (`LongRoPE2` + `NTK-by-Parts/YaRN`):
    - Exact `1.0x` unmodified `inv_freq` for all positions `<= 32,768` (`0.000000` short-context drift!).
    - Continuous `C^0` band-wise position compression `(32768 + (pos - 32768) / band_scale)` for `pos > 32,768`
      (`1.0x` on local high-freq bands `0..7`, smooth ramp to `3.0x` on global low-freq bands `20..31`),
      preserving incremental KV cache validity across the `32k -> 256k` boundary!
    """

    def __init__(self, base_rope: nn.Module, pivot_pos: int = 32768, max_scale: float = 3.5) -> None:
        super().__init__()
        self.dim = int(base_rope.dim)
        self.base = float(base_rope.base)
        self.angles = int(base_rope.angles)
        self.pivot = float(pivot_pos)
        self.max_scale = float(max_scale)
        inv_f = self.base ** (-mx.arange(self.angles, dtype=mx.float32) * 2.0 / self.dim)
        inv_freq_pad = mx.pad(inv_f, [(0, self.dim // 2 - self.angles)])
        scales = []
        for i in range(self.dim // 2):
            if i >= self.angles or i < 8:
                scales.append(1.0)
            elif i < 20:
                alpha = (i - 8) / 12.0
                scales.append(1.0 + alpha * (self.max_scale - 1.0))
            else:
                scales.append(self.max_scale)
        inv_band_scale = mx.array([1.0 / s for s in scales], dtype=mx.float32)
        mx.eval(inv_freq_pad, inv_band_scale)
        # Store constant buffers in `__dict__` so MLX `nn.Module.parameters()` keeps zero parameter keys on `rope`!
        self.__dict__["_inv_freq"] = inv_freq_pad
        self.__dict__["_inv_band_scale"] = inv_band_scale

    def __call__(self, x: mx.array, offset: Any = 0, position_ids: mx.array | None = None) -> mx.array:
        if isinstance(offset, mx.array) and offset.ndim > 0:
            offset = offset[:, None]
        positions = (mx.arange(x.shape[-2]) + offset if position_ids is None else position_ids).astype(mx.float32)
        pos_exp = positions[..., None]
        # Fast exact path when entire batch/chunk is <= 32,768; seamless C^0 YaRN extension above 32,768!
        over = mx.maximum(pos_exp - self.pivot, 0.0)
        eff_pos = mx.minimum(pos_exp, self.pivot) + over * self._inv_band_scale
        ang = eff_pos * self._inv_freq
        ang = mx.concatenate([ang, ang], axis=-1)
        if ang.ndim == 3:
            ang = ang[:, None]
        cos, sin = mx.cos(ang).astype(x.dtype), mx.sin(ang).astype(x.dtype)
        a, b = mx.split(x, 2, axis=-1)
        return x * cos + mx.concatenate([-b, a], axis=-1) * sin


# ===========================================================================
# 3. Kimi MoBA + DeepSeek CSA2 Pinned-Sink + Top-K Block Attention (`L > 4k`)
# ===========================================================================


def moba_pinned_sparse_sdpa(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    scale: float,
    sink_tokens: int = 640,
    local_window: int = 1024,
    block_size: int = 64,
    top_k_blocks: int = 128,
) -> mx.array:
    """Exact causal Pinned-Sink + Top-K Block-Routed Attention for `full_attention` beyond dense threshold (`> 65k`).

    - `q`: `[B, H_q, L_q, D]`
    - `k`, `v`: `[B, H_kv, L_kv, D]`
    Always preserves:
      1. Turn-1 foundation tokens `[0 : sink_tokens]` (System + Rules + ALL `OPT-A..OPT-F` candidate options),
      2. Causal local window (`local_window` tokens),
      3. Top-`top_k_blocks` middle blocks (`block_size` tokens each) per KV head via block-mean query-key dot product.
    """
    B, H_q, L_q, D = q.shape
    _, H_kv, L_kv, _ = k.shape
    gqa_group = max(1, H_q // H_kv)

    def _gather_middle_blocks(q_blk: mx.array, k_mid: mx.array, v_mid: mx.array) -> tuple[mx.array, mx.array]:
        # `k_mid`: [B, H_kv, M, D] where M is an exact multiple of `block_size`
        M = int(k_mid.shape[2])
        n_blks = M // block_size
        if n_blks <= top_k_blocks:
            return k_mid, v_mid
        k_b = k_mid.reshape(B, H_kv, n_blks, block_size, D)
        v_b = v_mid.reshape(B, H_kv, n_blks, block_size, D)
        k_mean = mx.mean(k_b.astype(mx.float32), axis=3)  # [B, H_kv, n_blks, D]
        q_grp = mx.mean(q_blk.astype(mx.float32).reshape(B, H_kv, gqa_group, -1, D), axis=(2, 3))  # [B, H_kv, D]
        scores = mx.matmul(q_grp[:, :, None, :], k_mean.transpose(0, 1, 3, 2)).squeeze(2)  # [B, H_kv, n_blks]
        top_idx = mx.sort(mx.argpartition(-scores, kth=top_k_blocks - 1, axis=-1)[..., :top_k_blocks], axis=-1)
        idx_exp = top_idx[..., None, None]  # [B, H_kv, K, 1, 1]
        k_sel = mx.take_along_axis(k_b, idx_exp, axis=2).reshape(B, H_kv, top_k_blocks * block_size, D)
        v_sel = mx.take_along_axis(v_b, idx_exp, axis=2).reshape(B, H_kv, top_k_blocks * block_size, D)
        return k_sel, v_sel

    # Case A: Decode / Speculative Verify (`L_q <= 16` and `L_kv > 4096`)
    if L_q <= 16:
        raw_mid = max(0, (L_kv - local_window) - sink_tokens)
        aligned_mid = (raw_mid // block_size) * block_size
        mid_end = sink_tokens + aligned_mid
        if aligned_mid <= top_k_blocks * block_size:
            mask_dec = None
            if L_q > 1:
                q_pos = mx.arange(L_kv - L_q, L_kv)[:, None]
                k_pos = mx.arange(L_kv)[None, :]
                mask_dec = q_pos >= k_pos
            return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask_dec)

        k_sink, v_sink = k[:, :, :sink_tokens, :], v[:, :, :sink_tokens, :]
        k_sel, v_sel = _gather_middle_blocks(q, k[:, :, sink_tokens:mid_end, :], v[:, :, sink_tokens:mid_end, :])
        k_loc, v_loc = k[:, :, mid_end:, :], v[:, :, mid_end:, :]
        k_cat = mx.concatenate([k_sink, k_sel, k_loc], axis=2)
        v_cat = mx.concatenate([v_sink, v_sel, v_loc], axis=2)
        mask_dec = None
        if L_q > 1:
            # Prefix (`sink` + `sel` + `loc[:-L_q]`) is strictly < all `L_q` query positions; causal mask on tail `L_q`
            tot_k = int(k_cat.shape[2])
            q_rel = mx.arange(tot_k - L_q, tot_k)[:, None]
            k_rel = mx.arange(tot_k)[None, :]
            mask_dec = q_rel >= k_rel
        return mx.fast.scaled_dot_product_attention(q, k_cat, v_cat, scale=scale, mask=mask_dec)

    # Case B: Long-Context Prefill (`L_q > 4096`, `L_kv == L_q` or cached prefix + `L_q`)
    base_off = L_kv - L_q
    out_chunks: list[mx.array] = []
    chunk_q = 1024
    for qs in range(0, L_q, chunk_q):
        qe = min(L_q, qs + chunk_q)
        abs_qs = base_off + qs
        abs_qe = base_off + qe
        q_c = q[:, :, qs:qe, :]

        if abs_qe <= sink_tokens + local_window + top_k_blocks * block_size:
            k_sub = k[:, :, :abs_qe, :]
            v_sub = v[:, :, :abs_qe, :]
            q_pos = mx.arange(abs_qs, abs_qe)[:, None]
            k_pos = mx.arange(abs_qe)[None, :]
            c_mask = q_pos >= k_pos
            out_chunks.append(mx.fast.scaled_dot_product_attention(q_c, k_sub, v_sub, scale=scale, mask=c_mask))
            continue

        # Every key before `abs_qs - (local_window // 2)` is strictly causal (`< abs_qs`) for ALL queries in `q_c`!
        safe_hist_limit = max(sink_tokens, abs_qs - (local_window // 2))
        aligned_mid = ((safe_hist_limit - sink_tokens) // block_size) * block_size
        mid_end = sink_tokens + aligned_mid

        k_sink, v_sink = k[:, :, :sink_tokens, :], v[:, :, :sink_tokens, :]
        k_sel, v_sel = _gather_middle_blocks(q_c, k[:, :, sink_tokens:mid_end, :], v[:, :, sink_tokens:mid_end, :])
        k_tail, v_tail = k[:, :, mid_end:abs_qe, :], v[:, :, mid_end:abs_qe, :]

        k_cat = mx.concatenate([k_sink, k_sel, k_tail], axis=2)
        v_cat = mx.concatenate([v_sink, v_sel, v_tail], axis=2)

        n_free = int(k_sink.shape[2]) + int(k_sel.shape[2])
        tail_len = abs_qe - mid_end
        free_mask = mx.ones((qe - qs, n_free), dtype=mx.bool_)
        tail_q = mx.arange(abs_qs, abs_qe)[:, None]
        tail_k = mx.arange(mid_end, abs_qe)[None, :]
        c_mask = mx.concatenate([free_mask, tail_q >= tail_k], axis=1)
        out_chunks.append(mx.fast.scaled_dot_product_attention(q_c, k_cat, v_cat, scale=scale, mask=c_mask))

    return mx.concatenate(out_chunks, axis=2)


# ===========================================================================
# 4. Seamless `Lfm3Model` & `Attention` Attachment + Sidecar Checkpointing
# ===========================================================================


def attach_v8_extensions(
    vlm_model: Any,
    sidecar_dir: str | Path | None = None,
    *,
    enable_moba: bool = True,
    moba_threshold: int = 65536,
    num_engram_buckets: int = DEFAULT_ENGRAM_BUCKETS,
    enable_v12_deep: bool = False,
) -> dict[str, Any]:
    """Idempotently attach `Engram` (`+0.255B`), `CrossLayerAttnRes`, `LatentReasoningRecurrence` (`L29`/`L35`),
    `ThoughtAnchorHeadTemp`, `LongRoPE2YaRNRotary` (`256k`), and `MoBA` onto a loaded `36L` or `40L` `LFM3-VL` model.
    """
    from mlx_vlm.models.base import create_attention_mask, create_ssm_mask, scaled_dot_product_attention

    lfm3_core = vlm_model.language_model.model
    cfg = lfm3_core.config
    hidden_size = int(cfg.hidden_size)
    norm_eps = float(getattr(cfg, "norm_eps", 1e-5))
    is_40l = len(lfm3_core.layers) >= 40
    tgt_tuple = ATTN_RES_TARGETS_40L if is_40l else ATTN_RES_TARGETS

    sc_cfg: dict[str, Any] = {}
    if sidecar_dir is not None and (Path(sidecar_dir) / "v8_arch_config.json").exists():
        try:
            sc_cfg = json.loads((Path(sidecar_dir) / "v8_arch_config.json").read_text(encoding="utf-8"))
        except Exception:
            sc_cfg = {}

    if sc_cfg.get("moba_threshold") is not None:
        moba_threshold = max(int(moba_threshold), int(sc_cfg["moba_threshold"]))

    use_v12_deep = bool(enable_v12_deep or sc_cfg.get("thought_anchor_temp") or sc_cfg.get("entry_inject") or sc_cfg.get("arch_version") == "v12_40l_deep")
    lr_layers: list[int] = list(sc_cfg.get("latent_recur_layers", [29, 35] if (is_40l and use_v12_deep) else ([35] if is_40l else [])))
    init_loops = int(sc_cfg.get("latent_recur_loops", 4 if use_v12_deep else 3))
    use_entry_inj = bool(sc_cfg.get("entry_inject", use_v12_deep))
    use_ta_temp = bool(sc_cfg.get("thought_anchor_temp", use_v12_deep))

    if not hasattr(lfm3_core, "engram_6"):
        lfm3_core.engram_6 = EngramConditionalMemory(
            hidden_size=hidden_size, num_buckets=num_engram_buckets, prime_prev=1315423, prime_curr=2654435, norm_eps=norm_eps
        )
        lfm3_core.engram_18 = EngramConditionalMemory(
            hidden_size=hidden_size, num_buckets=num_engram_buckets, prime_prev=1984913, prime_curr=32452843, norm_eps=norm_eps
        )
        lfm3_core.engram_6.apply(lambda x: x.astype(mx.bfloat16) if mx.issubdtype(x.dtype, mx.floating) else x)
        lfm3_core.engram_18.apply(lambda x: x.astype(mx.bfloat16) if mx.issubdtype(x.dtype, mx.floating) else x)

    for t_idx in tgt_tuple:
        m_name = f"attn_res_{t_idx}"
        if not hasattr(lfm3_core, m_name):
            mod_ar = CrossLayerAttnRes(
                hidden_size=hidden_size, num_sources=len(ATTN_RES_SOURCES), bottleneck_dim=256, norm_eps=norm_eps
            )
            mod_ar.apply(lambda x: x.astype(mx.bfloat16) if mx.issubdtype(x.dtype, mx.floating) else x)
            setattr(lfm3_core, m_name, mod_ar)

    if is_40l:
        for lr_idx in lr_layers:
            lr_name = f"latent_recur_{lr_idx}"
            if not hasattr(lfm3_core, lr_name):
                mod_lr = LatentReasoningRecurrence(
                    hidden_size=hidden_size,
                    latent_ffn_dim=2560,
                    num_loops=init_loops,
                    norm_eps=norm_eps,
                    entry_inject=use_entry_inj,
                )
                mod_lr.apply(lambda x: x.astype(mx.bfloat16) if mx.issubdtype(x.dtype, mx.floating) else x)
                setattr(lfm3_core, lr_name, mod_lr)
            elif use_entry_inj:
                getattr(lfm3_core, lr_name).enable_entry_inject(zero_init_proj=True)

        if use_ta_temp:
            for ta_idx in THOUGHT_ANCHOR_TEMP_LAYERS_40L:
                if ta_idx < len(lfm3_core.layers) and lfm3_core.layers[ta_idx].layer_type == "full_attention":
                    ta_name = f"ta_temp_{ta_idx}"
                    n_h = int(lfm3_core.layers[ta_idx].self_attn.n_heads)
                    if not hasattr(lfm3_core, ta_name):
                        setattr(lfm3_core, ta_name, ThoughtAnchorHeadTemp(num_heads=n_h))
                    lfm3_core.layers[ta_idx].self_attn.__dict__["_ta_temp_mod"] = getattr(lfm3_core, ta_name)

    fa_first_idx = next(i for i, lyr in enumerate(lfm3_core.layers) if lyr.layer_type == "full_attention")
    lfm3_core.__dict__["_fa_first_idx"] = fa_first_idx

    # Patch `Lfm3Model` forward cleanly via dynamic subclass (preserving MLX `nn.Module` dict semantics!)
    base_cls = type(lfm3_core)
    if getattr(base_cls, "_is_v8_patched", False) is False:

        class Lfm3ModelV8(base_cls):  # type: ignore[valid-type,misc]
            _is_v8_patched = True

            def __call__(self, inputs=None, cache=None, input_embeddings=None, position_ids=None):
                h = self.embed_tokens(inputs) if input_embeddings is None else input_embeddings
                n_layers = len(self.layers)
                cache_list = [None] * n_layers if cache is None else cache

                prev_ids: mx.array | None = None
                curr_ids: mx.array | None = None
                if inputs is not None:
                    curr_ids = inputs
                    B, L = int(inputs.shape[0]), int(inputs.shape[1])
                    pos0 = 0
                    fa_idx0 = int(self.__dict__.get("_fa_first_idx", 2))
                    c_fa = cache_list[fa_idx0]
                    if c_fa is not None and hasattr(c_fa, "offset"):
                        off_raw = c_fa.offset
                        pos0 = int(off_raw.item()) if isinstance(off_raw, mx.array) else int(off_raw)
                    ring: dict[int, mx.array] | None = self.__dict__.get("_v8_tok_ring")
                    if ring is None or pos0 == 0:
                        ring = {}
                        self.__dict__["_v8_tok_ring"] = ring
                    if pos0 > 0 and (pos0 - 1) in ring and int(ring[pos0 - 1].shape[0]) == B:
                        first_prev = ring[pos0 - 1][:, None]
                    else:
                        first_prev = mx.zeros((B, 1), dtype=inputs.dtype)
                    prev_ids = first_prev if L == 1 else mx.concatenate([first_prev, inputs[:, :-1]], axis=1)
                    # Record trailing boundary tokens by exact cache offset for lossless MTP speculative rollback!
                    if cache is not None:
                        start_rec = max(0, L - 8)
                        for j in range(start_rec, L):
                            ring[pos0 + j] = inputs[:, j]
                        if len(ring) > 4096:
                            cutoff = pos0 - 32
                            for old_k in [k_p for k_p in ring if k_p < cutoff]:
                                del ring[old_k]

                src_bank: list[mx.array] = []
                src_set = set(ATTN_RES_SOURCES)
                tgt_set = set(ATTN_RES_TARGETS_40L if n_layers >= 40 else ATTN_RES_TARGETS)

                for idx, (layer, c) in enumerate(zip(self.layers, cache_list)):
                    if layer.is_attention_layer:
                        window = self.config.sliding_window if layer.layer_type == "sliding_attention" else None
                        mask = create_attention_mask(h, c, window_size=window)
                    else:
                        mask = create_ssm_mask(h, c)
                    h = layer(h, mask, c, position_ids)

                    if idx == 6 and curr_ids is not None and prev_ids is not None and "engram_6" in self:
                        h = self.engram_6(h, curr_ids, prev_ids)
                    elif idx == 18 and curr_ids is not None and prev_ids is not None and "engram_18" in self:
                        h = self.engram_18(h, curr_ids, prev_ids)

                    if idx in src_set:
                        src_bank.append(h)
                    elif idx in tgt_set and len(src_bank) == len(ATTN_RES_SOURCES) and f"attn_res_{idx}" in self:
                        attn_res_mod = getattr(self, f"attn_res_{idx}")
                        h = attn_res_mod(h, src_bank)

                    if idx in (29, 35) and n_layers >= 40 and len(src_bank) == len(ATTN_RES_SOURCES) and f"latent_recur_{idx}" in self:
                        h = getattr(self, f"latent_recur_{idx}")(h, src_bank)

                return self.embedding_norm(h)

        lfm3_core.__class__ = Lfm3ModelV8

    # Attach `LongRoPE2YaRNRotary` (`256k` / `262,144`) when `max_position_embeddings >= 262144`
    max_pos = int(getattr(cfg, "max_position_embeddings", 128000))
    longrope_enabled = max_pos >= 262144
    if longrope_enabled:
        for lyr in lfm3_core.layers:
            if lyr.layer_type == "full_attention" and not isinstance(lyr.self_attn.rope, LongRoPE2YaRNRotary):
                lyr.self_attn.rope = LongRoPE2YaRNRotary(lyr.self_attn.rope, pivot_pos=32768, max_scale=3.5)

    # Attach `MoBA` Pinned-Sink + Top-K Block Attention + `ThoughtAnchorHeadTemp` on `full_attention` layers (`12` in 36L, `14` in 40L)
    for idx_l, lyr in enumerate(lfm3_core.layers):
        if lyr.layer_type != "full_attention":
            continue
        att = lyr.self_attn
        if hasattr(lfm3_core, f"ta_temp_{idx_l}"):
            att.__dict__["_ta_temp_mod"] = getattr(lfm3_core, f"ta_temp_{idx_l}")
        if not enable_moba and att.__dict__.get("_ta_temp_mod") is None:
            continue
        att_cls = type(att)
        if getattr(att_cls, "_is_moba_patched", False):
            att.__dict__["_moba_threshold"] = moba_threshold if enable_moba else 10**9
            continue

        class FullAttentionMoBA(att_cls):  # type: ignore[valid-type,misc]
            _is_moba_patched = True

            def __call__(self, x, mask=None, cache=None, position_ids=None):
                batch, length, _ = x.shape
                q = self.q_norm(self.q_proj(x).reshape(batch, length, self.n_heads, self.head_dim))
                k = self.k_norm(self.k_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim))
                v = self.v_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
                offset = cache.offset if cache is not None else 0
                q = self.rope(q.transpose(0, 2, 1, 3), offset, position_ids)
                k = self.rope(k.transpose(0, 2, 1, 3), offset, position_ids)
                ta_mod = self.__dict__.get("_ta_temp_mod")
                if ta_mod is not None:
                    q = ta_mod.scale_queries(q)
                if cache is not None:
                    k, v = cache.update_and_fetch(k, v)
                thresh = int(self.__dict__.get("_moba_threshold", 65536))
                if int(k.shape[2]) > thresh and position_ids is None:
                    out = moba_pinned_sparse_sdpa(q, k, v, scale=self.scale)
                else:
                    out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
                out = out.transpose(0, 2, 1, 3)
                out = out * mx.sigmoid(self.gate_proj(x))[..., None]
                return self.o_proj(out.reshape(batch, length, -1))

        att.__class__ = FullAttentionMoBA
        att.__dict__["_moba_threshold"] = moba_threshold if enable_moba else 10**9

    if sidecar_dir is not None:
        load_v8_sidecar(vlm_model, Path(sidecar_dir))
    else:
        mx.eval(lfm3_core.parameters())

    active_lr_layers = [idx_r for idx_r in (29, 35) if hasattr(lfm3_core, f"latent_recur_{idx_r}")]
    ext_params = get_v8_extension_tensors(vlm_model)
    n_ext = sum(int(v.size) for v in ext_params.values())
    n_tot = sum(int(v.size) for _, v in tree_flatten(vlm_model.parameters()))
    return {
        "arch_version": "v12_40l_deep" if (is_40l and len(active_lr_layers) >= 2) else ("v10_40l" if is_40l else "v8"),
        "num_layers": len(lfm3_core.layers),
        "v8_extension_params": n_ext,
        "total_model_params": n_tot,
        "engram_layers": list(ENGRAM_LAYERS),
        "attn_res_sources": list(ATTN_RES_SOURCES),
        "attn_res_targets": list(tgt_tuple),
        "latent_recur_layers": active_lr_layers,
        "thought_anchor_temp_enabled": hasattr(lfm3_core, "ta_temp_26"),
        "longrope_256k_enabled": longrope_enabled,
        "max_position_embeddings": max_pos,
        "moba_enabled": enable_moba,
        "moba_threshold": moba_threshold,
    }


def get_v8_extension_tensors(vlm_model: Any) -> dict[str, mx.array]:
    """Return flat dictionary of ONLY the v8/v10/v12 sidecar tensors (`engram_*`, `attn_res_*`, `latent_recur_*`, `ta_temp_*`)."""
    flat = dict(tree_flatten(vlm_model.parameters()))
    return {
        k: v
        for k, v in flat.items()
        if ".engram_" in k or ".attn_res_" in k or ".latent_recur_" in k or ".ta_temp_" in k
    }


def save_v8_sidecar(vlm_model: Any, out_dir: str | Path, extra_meta: dict[str, Any] | None = None) -> Path:
    """Atomically save `v8_arch_ext.safetensors` and `v8_arch_config.json` into `out_dir`."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    ext = {
        k: (v.astype(mx.bfloat16) if mx.issubdtype(v.dtype, mx.floating) else v)
        for k, v in get_v8_extension_tensors(vlm_model).items()
    }
    mx.eval(list(ext.values()))
    core = vlm_model.language_model.model
    is_40l = len(core.layers) >= 40
    tgt_tuple = ATTN_RES_TARGETS_40L if is_40l else ATTN_RES_TARGETS
    active_lr = [idx_r for idx_r in (29, 35) if hasattr(core, f"latent_recur_{idx_r}")]
    is_v12 = is_40l and (len(active_lr) >= 2 or hasattr(core, "ta_temp_26"))
    tmp_st = d / "tmp_v8_arch_ext.safetensors"
    dst_st = d / "v8_arch_ext.safetensors"
    mx.save_safetensors(
        str(tmp_st),
        ext,
        metadata={"format": "mlx", "arch": "rpr_v12_40l_deep" if is_v12 else ("rpr_v10_40l_apex" if is_40l else "rpr_v8_engram_attnres")},
    )
    tmp_st.replace(dst_st)

    cfg = {
        "arch_version": "v12_40l_deep" if is_v12 else ("v10_40l" if is_40l else "v8"),
        "num_layers": len(core.layers),
        "engram_layers": list(ENGRAM_LAYERS),
        "engram_buckets": int(core.engram_6.num_buckets),
        "attn_res_sources": list(ATTN_RES_SOURCES),
        "attn_res_targets": list(tgt_tuple),
        "latent_recur_layers": active_lr,
        "latent_recur_loops": int(core.latent_recur_35.num_loops) if (is_40l and hasattr(core, "latent_recur_35")) else 0,
        "entry_inject": bool(getattr(core.latent_recur_35, "entry_inject", False)) if (is_40l and hasattr(core, "latent_recur_35")) else False,
        "thought_anchor_temp": hasattr(core, "ta_temp_26"),
        "max_position_embeddings": int(getattr(core.config, "max_position_embeddings", 128000)),
        "moba_threshold": 65536,
        "extension_param_count": sum(int(v.size) for v in ext.values()),
        **(extra_meta or {}),
    }
    tmp_cfg = d / "v8_arch_config.json.tmp"
    tmp_cfg.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    tmp_cfg.replace(d / "v8_arch_config.json")
    return dst_st


def load_v8_sidecar(vlm_model: Any, model_dir: str | Path) -> bool:
    """Load `v8_arch_ext.safetensors` into an attached `vlm_model` if present."""
    st_path = Path(model_dir) / "v8_arch_ext.safetensors"
    if not st_path.exists():
        return False
    weights = mx.load(str(st_path))
    core = vlm_model.language_model.model
    hidden_size = int(core.config.hidden_size)
    norm_eps = float(getattr(core.config, "norm_eps", 1e-5))

    for lr_idx in (29, 35):
        ls_key = f"language_model.model.latent_recur_{lr_idx}.loop_scales"
        ej_key = f"language_model.model.latent_recur_{lr_idx}.w_entry.weight"
        if ls_key in weights:
            n_lp = int(weights[ls_key].shape[0])
            has_ej = ej_key in weights
            lr_name = f"latent_recur_{lr_idx}"
            if not hasattr(core, lr_name):
                mod_lr = LatentReasoningRecurrence(
                    hidden_size=hidden_size, latent_ffn_dim=2560, num_loops=n_lp, norm_eps=norm_eps, entry_inject=has_ej
                )
                mod_lr.apply(lambda x: x.astype(mx.bfloat16) if mx.issubdtype(x.dtype, mx.floating) else x)
                setattr(core, lr_name, mod_lr)
            else:
                lr_mod = getattr(core, lr_name)
                if lr_mod.num_loops != n_lp:
                    lr_mod.num_loops = n_lp
                    lr_mod.step_embed = mx.zeros((n_lp, lr_mod.latent_ffn_dim), dtype=mx.bfloat16)
                    lr_mod.loop_scales = mx.zeros((n_lp, lr_mod.hidden_size), dtype=mx.bfloat16)
                if has_ej and not getattr(lr_mod, "entry_inject", False):
                    lr_mod.enable_entry_inject(zero_init_proj=True)

    for ta_idx in THOUGHT_ANCHOR_TEMP_LAYERS_40L:
        ta_key = f"language_model.model.ta_temp_{ta_idx}.log_temp"
        if ta_key in weights and ta_idx < len(core.layers):
            ta_name = f"ta_temp_{ta_idx}"
            if not hasattr(core, ta_name):
                setattr(core, ta_name, ThoughtAnchorHeadTemp(num_heads=int(weights[ta_key].shape[0])))
            core.layers[ta_idx].self_attn.__dict__["_ta_temp_mod"] = getattr(core, ta_name)

    vlm_model.load_weights(list(weights.items()), strict=False)
    mx.eval(vlm_model.parameters())
    return True


def unfreeze_v8_modules(vlm_model: Any) -> None:
    """Unfreeze `Engram`, `CrossLayerAttnRes`, `LatentReasoningRecurrence` (`L29`/`L35`), and `ThoughtAnchorHeadTemp` modules."""
    core = vlm_model.language_model.model
    core.engram_6.unfreeze()
    core.engram_18.unfreeze()
    tgt_tuple = ATTN_RES_TARGETS_40L if len(core.layers) >= 40 else ATTN_RES_TARGETS
    for t_idx in tgt_tuple:
        getattr(core, f"attn_res_{t_idx}").unfreeze()
    for lr_idx in (29, 35):
        if hasattr(core, f"latent_recur_{lr_idx}"):
            getattr(core, f"latent_recur_{lr_idx}").unfreeze()
    for ta_idx in THOUGHT_ANCHOR_TEMP_LAYERS_40L:
        if hasattr(core, f"ta_temp_{ta_idx}"):
            getattr(core, f"ta_temp_{ta_idx}").unfreeze()


def upcycle_v9_36l_to_v10_40l_identity(
    v9_dir: str | Path,
    out_40l_dir: str | Path,
    *,
    seed: int = 20261122,
    latent_loops: int = 3,
) -> dict[str, Any]:
    """Surgically upcycle `36L` `LFM3-VL-2.9B-RPR-V9-SEAL-BF16` -> `40L` `3.18B` pure `bfloat16` (`256k` context)
    with mathematically guaranteed `Step-0 max_abs_logit_diff == 0.000000`!

    - Layers `0..33`: copied unchanged from `v9`.
    - Layers `(34, 35)`: cloned `(full_attention, conv)` from `(32, 33)` with zeroed `post_operator_norm` & `post_ffn_norm`.
    - `latent_recur_35`: 2-step shared-weight `LatentReasoningRecurrence` (`loop_scales = 0`).
    - Layers `(36, 37)`: original `v9` `(34, 35)` + `attn_res_36` remapped from `v9`'s `attn_res_34`.
    - Layers `(38, 39)`: cloned `(full_attention, conv)` from original `(34, 35)` with zeroed `post_operator_norm` & `post_ffn_norm`
      + zero-gated 5th `CrossLayerAttnRes` (`attn_res_38`).
    - `text_config.max_position_embeddings = 262144` (`LongRoPE2YaRNRotary` `C^0` dual-path `256k` context).
    """
    import shutil

    src = Path(v9_dir)
    dst = Path(out_40l_dir)
    dst.mkdir(parents=True, exist_ok=True)

    skip_names = {
        "model.safetensors",
        "model.safetensors.index.json",
        "v8_arch_ext.safetensors",
        "v8_arch_config.json",
        "seal_nullspace_svf_bank.safetensors",
        "mtp_head.safetensors",
        "mtp_config.json",
        "stage123_done.json",
        "v9_stage1234_done.json",
        "upcycle_36l_verified.json",
        "__pycache__",
    }
    for item in src.iterdir():
        if item.name in skip_names or item.is_dir():
            continue
        shutil.copy2(item, dst / item.name)
    shutil.copy2(Path(__file__), dst / "v8_arch.py")

    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    tcfg = cfg["text_config"]
    orig_lt = list(tcfg["layer_types"])
    if len(orig_lt) != 36:
        raise ValueError(f"Expected 36-layer source checkpoint, got {len(orig_lt)} layers in {src}")
    new_lt = orig_lt[:34] + ["full_attention", "conv"] + orig_lt[34:36] + ["full_attention", "conv"]
    assert len(new_lt) == 40 and sum(1 for x in new_lt if x == "full_attention") == 14
    tcfg["num_hidden_layers"] = 40
    tcfg["layer_types"] = new_lt
    tcfg["max_position_embeddings"] = 262144
    (dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    # 1. Transform `model.safetensors` (`36L -> 40L` pure `bfloat16`)
    raw_w = mx.load(str(src / "model.safetensors"))
    new_w: dict[str, mx.array] = {}
    prefix = "language_model.model.layers."

    def _zero_if_post_norm(subkey: str, tensor: mx.array) -> mx.array:
        if subkey.endswith("post_operator_norm.weight") or subkey.endswith("post_ffn_norm.weight"):
            return mx.zeros_like(tensor).astype(mx.bfloat16)
        return tensor.astype(mx.bfloat16) if mx.issubdtype(tensor.dtype, mx.floating) else tensor

    for k, v in raw_w.items():
        v_bf16 = v.astype(mx.bfloat16) if mx.issubdtype(v.dtype, mx.floating) else v
        if not k.startswith(prefix):
            new_w[k] = v_bf16
            continue
        rest = k[len(prefix) :]
        dot = rest.index(".")
        l_idx = int(rest[:dot])
        sub = rest[dot + 1 :]
        if l_idx <= 33:
            new_w[f"{prefix}{l_idx}.{sub}"] = v_bf16
            if l_idx in (32, 33):
                clone_idx = l_idx + 2  # 32 -> 34, 33 -> 35
                new_w[f"{prefix}{clone_idx}.{sub}"] = _zero_if_post_norm(sub, v_bf16)
        elif l_idx in (34, 35):
            orig_shift_idx = l_idx + 2  # 34 -> 36, 35 -> 37 (exact original v9 weights!)
            clone_tail_idx = l_idx + 4  # 34 -> 38, 35 -> 39 (zero-gated residual clones!)
            new_w[f"{prefix}{orig_shift_idx}.{sub}"] = v_bf16
            new_w[f"{prefix}{clone_tail_idx}.{sub}"] = _zero_if_post_norm(sub, v_bf16)

    del raw_w
    mx.eval(list(new_w.values()))
    tmp_ms = dst / "tmp_model.safetensors"
    dst_ms = dst / "model.safetensors"
    mx.save_safetensors(str(tmp_ms), new_w, metadata={"format": "mlx", "arch": "lfm3_vl_40l_bf16"})
    tmp_ms.replace(dst_ms)

    total_bytes = sum(int(t.nbytes) for t in new_w.values())
    base_param_count = sum(int(t.size) for t in new_w.values())
    idx_doc = {
        "metadata": {"total_size": total_bytes},
        "weight_map": {k: "model.safetensors" for k in sorted(new_w.keys())},
    }
    (dst / "model.safetensors.index.json").write_text(json.dumps(idx_doc, indent=2) + "\n", encoding="utf-8")
    del new_w

    # 2. Remap `v8_arch_ext.safetensors` (`attn_res_34 -> attn_res_36`, init zero-gated `attn_res_38` & `latent_recur_35`)
    mx.random.seed(seed)
    old_ext = mx.load(str(src / "v8_arch_ext.safetensors"))
    new_ext: dict[str, mx.array] = {}
    for k, v in old_ext.items():
        v_bf16 = v.astype(mx.bfloat16) if mx.issubdtype(v.dtype, mx.floating) else v
        if ".attn_res_34." in k:
            new_ext[k.replace(".attn_res_34.", ".attn_res_36.")] = v_bf16
        else:
            new_ext[k] = v_bf16
    del old_ext

    hidden_size = int(tcfg["hidden_size"])
    norm_eps = float(tcfg.get("norm_eps", 1e-5))
    ar38 = CrossLayerAttnRes(hidden_size=hidden_size, num_sources=len(ATTN_RES_SOURCES), bottleneck_dim=256, norm_eps=norm_eps)
    lr35 = LatentReasoningRecurrence(hidden_size=hidden_size, latent_ffn_dim=2560, num_loops=latent_loops, norm_eps=norm_eps)
    for p_k, p_v in tree_flatten(ar38.parameters()):
        new_ext[f"language_model.model.attn_res_38.{p_k}"] = (
            p_v.astype(mx.bfloat16) if mx.issubdtype(p_v.dtype, mx.floating) else p_v
        )
    for p_k, p_v in tree_flatten(lr35.parameters()):
        new_ext[f"language_model.model.latent_recur_35.{p_k}"] = (
            p_v.astype(mx.bfloat16) if mx.issubdtype(p_v.dtype, mx.floating) else p_v
        )
    mx.eval(list(new_ext.values()))
    ext_param_count = sum(int(t.size) for t in new_ext.values())
    tmp_ext = dst / "tmp_v8_arch_ext.safetensors"
    mx.save_safetensors(str(tmp_ext), new_ext, metadata={"format": "mlx", "arch": "rpr_v11_40l_omega"})
    tmp_ext.replace(dst / "v8_arch_ext.safetensors")

    ext_cfg = {
        "arch_version": "v11_40l_omega",
        "num_layers": 40,
        "engram_layers": list(ENGRAM_LAYERS),
        "engram_buckets": DEFAULT_ENGRAM_BUCKETS,
        "attn_res_sources": list(ATTN_RES_SOURCES),
        "attn_res_targets": list(ATTN_RES_TARGETS_40L),
        "latent_recur_layers": [35],
        "latent_recur_loops": latent_loops,
        "upcycled_40l_indices": list(UPCYCLED_40L_INDICES),
        "max_position_embeddings": 262144,
        "moba_threshold": 65536,
        "extension_param_count": ext_param_count,
        "base_param_count": base_param_count,
        "total_param_count": base_param_count + ext_param_count,
    }
    (dst / "v8_arch_config.json").write_text(json.dumps(ext_cfg, indent=2) + "\n", encoding="utf-8")
    del new_ext
    mx.clear_cache()
    return ext_cfg


# ===========================================================================
# 5. Kimi K2 / K3.1 Pro `MuonClip` Optimizer (5-Step Newton-Schulz + QK-Clip)
# ===========================================================================


def zeropower_via_newtonschulz5(g: mx.array, steps: int = 5) -> mx.array:
    """Quintic 5-step Newton-Schulz orthogonalization (`Kimi K2` / Jordan 2024) in `bfloat16`."""
    a, b, c = 3.4445, -4.7750, 2.0315
    transposed = int(g.shape[0]) > int(g.shape[1])
    x = g.T if transposed else g
    norm = mx.sqrt(mx.sum(x.astype(mx.float32) * x.astype(mx.float32))) + 1e-7
    x = (x.astype(mx.float32) / norm).astype(mx.bfloat16)
    for _ in range(steps):
        at = x @ x.T
        bs = b * at + c * (at @ at)
        x = a * x + (bs @ x)
    return x.T if transposed else x


class Muon2D(optim.Optimizer):
    """Kimi K2 RMS-Scaled Nesterov Muon Optimizer for dense 2D projection & LoRA matrices."""

    def __init__(
        self,
        learning_rate: float | Any = 1.5e-4,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        ns_steps: int = 5,
    ) -> None:
        super().__init__()
        self._maybe_schedule("learning_rate", learning_rate)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.ns_steps = ns_steps

    def init_single(self, parameter: mx.array, state: dict[str, Any]) -> None:
        state["m"] = mx.zeros_like(parameter)

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict[str, Any]) -> mx.array:
        lr = self.learning_rate.astype(mx.float32)
        m_buf = state["m"]
        g_f32 = gradient.astype(mx.float32)
        m_new = self.momentum * m_buf.astype(mx.float32) + g_f32
        state["m"] = m_new.astype(parameter.dtype)
        u = g_f32 + self.momentum * m_new
        orth = zeropower_via_newtonschulz5(u, steps=self.ns_steps).astype(mx.float32)
        # Kimi K2 RMS alignment factor: `0.2 * sqrt(max(rows, cols))` aligns Muon LR directly with AdamW LR!
        rms_scale = 0.2 * math.sqrt(float(max(int(parameter.shape[0]), int(parameter.shape[1]))))
        p_f32 = parameter.astype(mx.float32)
        if self.weight_decay > 0.0:
            p_f32 = p_f32 * (1.0 - lr * self.weight_decay)
        updated = p_f32 - lr * rms_scale * orth
        return updated.astype(parameter.dtype)


def build_muon_clip_optimizer(
    *,
    lr_gates: Any,
    lr_muon_2d: Any,
    lr_engram_table: Any,
) -> optim.MultiOptimizer:
    """Build hybrid `MuonClip` (`Muon2D` on dense 2D matrices, `AdamW` on 1D gates/scales & sparse `Engram` rows)."""
    gate_keys = (
        "post_operator_norm",
        "post_ffn_norm",
        "out_scale",
        "res_scale",
        "loop_scales",
        "step_embed",
        "log_temp",
        "ctx_norm",
        "q_norm",
        "src_norm",
        "h_norm",
        "anc_norm",
        "entry_norm",
    )
    return optim.MultiOptimizer(
        [
            optim.AdamW(learning_rate=lr_gates, weight_decay=0.0),
            optim.AdamW(learning_rate=lr_engram_table, weight_decay=0.0),
            Muon2D(learning_rate=lr_muon_2d, momentum=0.95, weight_decay=0.01, ns_steps=5),
            optim.AdamW(learning_rate=lr_muon_2d, weight_decay=0.01),
        ],
        filters=[
            lambda k, p: p.ndim == 1 or any(gk in k for gk in gate_keys),
            lambda k, _: ".table.weight" in k,
            lambda k, p: p.ndim == 2 and ".table.weight" not in k,
        ],
    )


def apply_qk_clip(vlm_model: Any, max_lora_fro_norm: float = 6.0) -> int:
    """Kimi `QK-Clip` post-update spectral/Frobenius bound on `q_proj` and `k_proj` LoRA adapters."""
    clipped = 0
    for _, layer in enumerate(vlm_model.language_model.model.layers):
        if not layer.is_attention_layer:
            continue
        att = layer.self_attn
        for proj_name in ("q_proj", "k_proj"):
            proj = getattr(att, proj_name, None)
            if proj is not None and hasattr(proj, "lora_b") and hasattr(proj, "lora_a"):
                lb = proj.lora_b.astype(mx.float32)
                la = proj.lora_a.astype(mx.float32)
                eff_norm = float(mx.sqrt(mx.sum(lb * lb) * mx.sum(la * la)).item())
                if eff_norm > max_lora_fro_norm:
                    gamma = math.sqrt(max_lora_fro_norm / max(eff_norm, 1e-6))
                    proj.lora_b = (lb * gamma).astype(proj.lora_b.dtype)
                    proj.lora_a = (la * gamma).astype(proj.lora_a.dtype)
                    clipped += 1
    return clipped
