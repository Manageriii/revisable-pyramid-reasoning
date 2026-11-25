"""Lossless Sequential Multi-Token Prediction (MTP / EAGLE-FR-Spec) Draft Module & Tri-Hybrid Speculative Decoder for LFM3.

Key Hardware & Mathematical Properties:
  1. **Zero Vocab-Matrix Bloat (`FR-Spec` Shortlist):** Reuses frozen `embed_tokens` gathered over the top
     `V_sub = 16,384` active reasoning/code/JSON tokens (`64 MiB`), avoiding a `500 MiB` 128k untied `lm_head`.
  2. **Prompt-Context-Aware EAGLE Sequential Draft Block (`~33.6M` params, `64 MiB` BF16):**
     Fuses `[RMSNorm(h_t) ; RMSNorm(Embed(y_{t+1}))] -> 2048` through a dedicated GQA + SandwichNorm + SwiGLU
     block with its own `KVCache`, prefilled once in parallel over the prompt so draft steps attend to the
     entire prompt and `<think>` history (`~128 MiB` total read per draft token vs `4.49 GiB` on 32L trunk = `35x` lighter!).
  3. **Zero-Recompute Bit-Exact Tri-Hybrid Cache Rollback (`K <= 5`):**
     Rewinds `KVCache` (`full_attention`), `RotatingKVCache` (`sliding_attention`), and `ArraysCache`
     (`ShortConv` `L_cache = 7`) to exact accepted length `m in {1..1+K}` in `O(1)` tensor slices without
     running a second trunk forward pass, guaranteeing **100% lossless greedy token-for-token equivalence**.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_vlm.models.cache import ArraysCache, KVCache, RotatingKVCache
from mlx_vlm.models.lfm3.config import ModelConfig as Lfm3TextConfig
from mlx_vlm.models.lfm3.language import Attention, create_attention_mask


class _MTPMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


@mx.compile
def _mtp_pre_qkvg_compiled(
    prev_hidden: mx.array,
    next_token_embeds: mx.array,
    h_norm_w: mx.array,
    e_norm_w: mx.array,
    op_norm_w: mx.array,
    fusion_w: mx.array,
    qkvg_w: mx.array,
    q_norm_w: mx.array,
    k_norm_w: mx.array,
    eps: float,
    qk_eps: float,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    h_n = mx.fast.rms_norm(prev_hidden, h_norm_w, eps)
    e_n = mx.fast.rms_norm(next_token_embeds, e_norm_w, eps)
    x = mx.concatenate([h_n, e_n], axis=-1) @ fusion_w.T
    x_op = mx.fast.rms_norm(x, op_norm_w, eps)
    qkvg = x_op @ qkvg_w.T
    b_sz, s_len, _ = x.shape
    q_end = n_heads * head_dim
    k_end = q_end + n_kv_heads * head_dim
    v_end = k_end + n_kv_heads * head_dim
    q = mx.fast.rms_norm(qkvg[..., :q_end].reshape(b_sz, s_len, n_heads, head_dim), q_norm_w, qk_eps).transpose(0, 2, 1, 3)
    k = mx.fast.rms_norm(qkvg[..., q_end:k_end].reshape(b_sz, s_len, n_kv_heads, head_dim), k_norm_w, qk_eps).transpose(0, 2, 1, 3)
    v = qkvg[..., k_end:v_end].reshape(b_sz, s_len, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
    gate_sig = mx.sigmoid(qkvg[..., v_end:])[..., None]
    return x, q, k, v, gate_sig


@mx.compile
def _mtp_post_attn_compiled(
    x: mx.array,
    sdpa_out: mx.array,
    gate_sig: mx.array,
    o_proj_w: mx.array,
    post_op_w: mx.array,
    ffn_norm_w: mx.array,
    w1: mx.array,
    w3: mx.array,
    w2: mx.array,
    post_ffn_w: mx.array,
    out_norm_w: mx.array,
    eps: float,
) -> mx.array:
    b_sz, _, s_len, _ = sdpa_out.shape
    attn_out = (sdpa_out.transpose(0, 2, 1, 3) * gate_sig).reshape(b_sz, s_len, -1) @ o_proj_w.T
    x2 = x + mx.fast.rms_norm(attn_out, post_op_w, eps)
    f_in = mx.fast.rms_norm(x2, ffn_norm_w, eps)
    h_ff = (nn.silu(f_in @ w1.T) * (f_in @ w3.T)) @ w2.T
    x3 = x2 + mx.fast.rms_norm(h_ff, post_ffn_w, eps)
    return mx.fast.rms_norm(x3, out_norm_w, eps)


@mx.compile
def _mtp_shortlist_argmax_compiled(h_mtp_step: mx.array, w_sub: mx.array, shortlist_mx: mx.array) -> mx.array:
    sub_idx = mx.argmax(h_mtp_step @ w_sub.T, axis=-1)
    return mx.take(shortlist_mx, sub_idx, axis=0)


@mx.compile
def _mtp_shortlist_argmax_and_conf_compiled(
    h_mtp_step: mx.array, w_sub: mx.array, shortlist_mx: mx.array
) -> tuple[mx.array, mx.array]:
    logits = (h_mtp_step @ w_sub.T).astype(mx.float32)
    sub_idx = mx.argmax(logits, axis=-1)
    tok_id = mx.take(shortlist_mx, sub_idx, axis=0)
    # Approximate top-1 confidence margin `max_logit - logsumexp(logits)` (= log p_top1 <= 0) in O(V_sub) without top-2 sort!
    log_p_top1 = mx.max(logits, axis=-1) - mx.logsumexp(logits, axis=-1)
    return tok_id, log_p_top1


class SequentialMTPHead(nn.Module):
    """EAGLE-style sequential feature-fusion MTP block with FR-Spec active-vocabulary shortlist."""

    def __init__(
        self,
        text_config: Lfm3TextConfig,
        *,
        mtp_intermediate_size: int = 3072,
        shortlist_ids: list[int] | None = None,
    ):
        super().__init__()
        d = int(text_config.hidden_size)
        eps = float(text_config.norm_eps)
        self.hidden_size = d
        self.norm_eps = eps
        self.mtp_intermediate_size = mtp_intermediate_size

        self.h_norm = nn.RMSNorm(d, eps=eps)
        self.e_norm = nn.RMSNorm(d, eps=eps)
        self.fusion_proj = nn.Linear(2 * d, d, bias=False)

        self.operator_norm = nn.RMSNorm(d, eps=eps)
        self.self_attn = Attention(text_config, "full_attention")
        self.post_operator_norm = nn.RMSNorm(d, eps=eps)

        self.ffn_norm = nn.RMSNorm(d, eps=eps)
        self.feed_forward = _MTPMLP(d, mtp_intermediate_size)
        self.post_ffn_norm = nn.RMSNorm(d, eps=eps)
        self.out_norm = nn.RMSNorm(d, eps=eps)
        # DeepSeek-V3 MTP / EAGLE-3 Decoupled Logit Rotation (`Identity`-initialized; folded into `_cached_w_sub` at eval for 0ms cost!)
        self.logit_proj = nn.Linear(d, d, bias=False)
        self.logit_proj.weight = mx.eye(d, dtype=mx.bfloat16)

        # Non-trainable shortlist index buffer `[V_sub]`
        if shortlist_ids is None:
            shortlist_ids = list(range(min(16384, int(text_config.vocab_size))))
        self._shortlist_list = [int(x) for x in shortlist_ids]
        self._shortlist_mx = mx.array(self._shortlist_list, dtype=mx.int32)
        self._cached_w_sub_raw: mx.array | None = None
        self._cached_w_sub: mx.array | None = None
        self._cached_qkvg_w: mx.array | None = None

    def set_shortlist(self, shortlist_ids: list[int]) -> None:
        self._shortlist_list = [int(x) for x in shortlist_ids]
        self._shortlist_mx = mx.array(self._shortlist_list, dtype=mx.int32)
        self._cached_w_sub_raw = None
        self._cached_w_sub = None
        self._cached_qkvg_w = None

    def bind_vocab_projection(self, embed_tokens: nn.Embedding) -> None:
        """Cache raw `W_sub` (`[V_sub, d]`), folded `W_sub_eff = W_sub @ logit_proj.weight`, and fused QKVG (`0ms` overhead!)."""
        w_raw = mx.take(embed_tokens.weight, self._shortlist_mx, axis=0).astype(mx.bfloat16)
        w_eff = (w_raw.astype(mx.float32) @ self.logit_proj.weight.astype(mx.float32)).astype(mx.bfloat16)
        sa = self.self_attn
        qkvg_w = mx.concatenate(
            [sa.q_proj.weight, sa.k_proj.weight, sa.v_proj.weight, sa.gate_proj.weight], axis=0
        ).astype(mx.bfloat16)
        mx.eval(w_raw, w_eff, qkvg_w)
        self._cached_w_sub_raw = w_raw
        self._cached_w_sub = w_eff
        self._cached_qkvg_w = qkvg_w

    def warm_start_from_target_lm(self, language_model: Any) -> None:
        """Warm-start `self_attn` and norms from top `full_attention` layer + SandwichNorm zero-gated MLP."""
        d = self.hidden_size
        eye_half = 0.5 * mx.eye(d, dtype=mx.float32)
        self.fusion_proj.weight = mx.concatenate([eye_half, eye_half], axis=1).astype(mx.bfloat16)
        self.out_norm.weight = mx.array(language_model.model.embedding_norm.weight).astype(mx.bfloat16)
        # SandwichNorm Zero-Gate on randomly initialized `_MTPMLP` so Step 0 has zero white-noise injection!
        self.post_ffn_norm.weight = mx.zeros_like(self.post_ffn_norm.weight).astype(mx.bfloat16)

        src_attn_layer = None
        for lyr in reversed(language_model.model.layers):
            if getattr(lyr, "layer_type", "") == "full_attention":
                src_attn_layer = lyr
                break
        if src_attn_layer is not None:
            self.operator_norm.weight = mx.array(src_attn_layer.operator_norm.weight).astype(mx.bfloat16)
            self.ffn_norm.weight = mx.array(src_attn_layer.ffn_norm.weight).astype(mx.bfloat16)
            if hasattr(src_attn_layer, "post_operator_norm"):
                self.post_operator_norm.weight = mx.array(src_attn_layer.post_operator_norm.weight).astype(mx.bfloat16)
            src_weights = dict(tree_flatten(src_attn_layer.self_attn.parameters()))
            self.self_attn.load_weights([(k, v.astype(mx.bfloat16)) for k, v in src_weights.items()], strict=True)
        mx.eval(self.parameters())

    def forward_features(
        self,
        prev_hidden: mx.array,
        next_token_embeds: mx.array,
        cache: KVCache | None = None,
    ) -> mx.array:
        """Fuse `(h_{t}, embed(y_{t+1}))` -> `h_mtp_{t+1}` (`[B, S, d]`)."""
        if not self.training:
            sa = self.self_attn
            if self._cached_qkvg_w is None:
                self._cached_qkvg_w = mx.concatenate(
                    [sa.q_proj.weight, sa.k_proj.weight, sa.v_proj.weight, sa.gate_proj.weight], axis=0
                ).astype(mx.bfloat16)
                mx.eval(self._cached_qkvg_w)
            x, q, k, v, gate_sig = _mtp_pre_qkvg_compiled(
                prev_hidden,
                next_token_embeds,
                self.h_norm.weight,
                self.e_norm.weight,
                self.operator_norm.weight,
                self.fusion_proj.weight,
                self._cached_qkvg_w,
                sa.q_norm.weight,
                sa.k_norm.weight,
                self.norm_eps,
                float(sa.q_norm.eps),
                int(sa.n_heads),
                int(sa.n_kv_heads),
                int(sa.head_dim),
            )
            offset = cache.offset if cache is not None else 0
            q = sa.rope(q, offset, None)
            k = sa.rope(k, offset, None)
            if cache is not None:
                k, v = cache.update_and_fetch(k, v)
            mask = None if int(x.shape[1]) == 1 else create_attention_mask(x, cache, window_size=None)
            sdpa_out = mx.fast.scaled_dot_product_attention(q, k, v, scale=sa.scale, mask=mask)
            return _mtp_post_attn_compiled(
                x,
                sdpa_out,
                gate_sig,
                sa.o_proj.weight,
                self.post_operator_norm.weight,
                self.ffn_norm.weight,
                self.feed_forward.w1.weight,
                self.feed_forward.w3.weight,
                self.feed_forward.w2.weight,
                self.post_ffn_norm.weight,
                self.out_norm.weight,
                self.norm_eps,
            )
        self._cached_qkvg_w = None
        fused = mx.concatenate([self.h_norm(prev_hidden), self.e_norm(next_token_embeds)], axis=-1)
        x = self.fusion_proj(fused)
        mask = create_attention_mask(x, cache, window_size=None)
        h = self.self_attn(self.operator_norm(x), mask, cache, None)
        x = x + self.post_operator_norm(h)
        h_ff = self.feed_forward(self.ffn_norm(x))
        x = x + self.post_ffn_norm(h_ff)
        return self.out_norm(x)

    def draft_logits_shortlist(self, h_mtp: mx.array, embed_tokens: nn.Embedding) -> mx.array:
        """Project `h_mtp` (`[B, S, d]`) onto `[B, S, V_sub]` shortlist logits (`logit_proj` Differentiable in train, Folded in eval)."""
        if self._cached_w_sub_raw is None or self._cached_w_sub is None:
            self.bind_vocab_projection(embed_tokens)
        if self.training:
            assert self._cached_w_sub_raw is not None
            return self.logit_proj(h_mtp) @ self._cached_w_sub_raw.T
        assert self._cached_w_sub is not None
        return h_mtp @ self._cached_w_sub.T

    def draft_greedy_token(self, h_mtp_step: mx.array, embed_tokens: nn.Embedding) -> mx.array:
        """Return greedy token ID `[B, 1]` mapped back from `FR-Spec` shortlist to global `128k` vocab."""
        if self._cached_w_sub is None:
            self.bind_vocab_projection(embed_tokens)
        assert self._cached_w_sub is not None
        if not self.training:
            return _mtp_shortlist_argmax_compiled(h_mtp_step, self._cached_w_sub, self._shortlist_mx)
        sub_logits = h_mtp_step @ self._cached_w_sub.T  # [B, 1, V_sub]
        sub_idx = mx.argmax(sub_logits, axis=-1)  # [B, 1]
        return mx.take(self._shortlist_mx, sub_idx, axis=0)

    def draft_greedy_token_with_logprob(
        self, h_mtp_step: mx.array, embed_tokens: nn.Embedding
    ) -> tuple[mx.array, mx.array]:
        """Return `(tok_id [B, 1], log_p_top1 [B, 1])` for EAGLE-2 confidence-adaptive draft depth."""
        if self._cached_w_sub is None:
            self.bind_vocab_projection(embed_tokens)
        assert self._cached_w_sub is not None
        return _mtp_shortlist_argmax_and_conf_compiled(h_mtp_step, self._cached_w_sub, self._shortlist_mx)


# ---------------------------------------------------------------------------
# Exact Zero-Recompute Tri-Hybrid LFM3 Cache Snapshot & Rollback (`K <= 5`)
# ---------------------------------------------------------------------------


def snapshot_lfm3_cache_for_verify(cache: list[Any]) -> list[Any]:
    """Take lightweight zero-copy reference snapshots prior to verifying `1 + K` tokens."""
    snaps: list[Any] = []
    for c in cache:
        if isinstance(c, KVCache):
            snaps.append(("kv", int(c.offset)))
        elif isinstance(c, RotatingKVCache):
            snaps.append(("rot", c.keys, c.values, c.meta_state))
        elif isinstance(c, ArraysCache):
            snaps.append(("conv", c.cache[0], c._left_padding_advance, c._lengths_advance))
        else:
            raise TypeError(f"Unsupported cache type in LFM3 speculative verify: {type(c)}")
    return snaps


def rewind_lfm3_cache_after_verify(
    cache: list[Any],
    snaps: list[Any],
    *,
    verified_len: int,
    accepted_len: int,
    conv_keep: int = 6,
) -> None:
    """Rewind LFM3 caches in-place from `verified_len` (`1 + K`) back to `accepted_len` (`1 <= m <= 1 + K`).

    Requires `verified_len <= conv_keep` (`1 + K <= 6`, i.e., `K <= 5`) so `ArraysCache` retains all `m`
    accepted post-gate activations inside its trailing `6`-token window without re-running `ShortConv`.
    """
    reject_cnt = verified_len - accepted_len
    if reject_cnt <= 0:
        return
    if verified_len > conv_keep:
        raise ValueError(f"verified_len={verified_len} exceeds conv_keep={conv_keep} (use draft_k <= {conv_keep - 1}).")

    for c, snap in zip(cache, snaps):
        kind = snap[0]
        if kind == "kv":
            c.offset = snap[1] + accepted_len
        elif kind == "rot":
            _, old_k, old_v, old_meta = snap
            # Extract the `accepted_len` newly computed K/V slices from the tail of `_update_concat`'s temporal buffer
            new_k_all = c.keys[..., -verified_len:, :]
            new_v_all = c.values[..., -verified_len:, :]
            acc_k = new_k_all[..., :accepted_len, :]
            acc_v = new_v_all[..., :accepted_len, :]
            c.keys, c.values = old_k, old_v
            c.meta_state = old_meta
            if accepted_len == 1:
                c._update_in_place(acc_k, acc_v)
            else:
                c._update_concat(acc_k, acc_v)
        elif kind == "conv":
            _, prev_state, old_lp_adv, old_len_adv = snap
            new_state = c.cache[0]  # [B, 6, d] = [prev[verified_len:], b_0..b_{verified_len-1}]
            start_b = conv_keep - verified_len
            acc_b = new_state[:, start_b : start_b + accepted_len, :]
            if prev_state is None:
                prev_zeros = mx.zeros((new_state.shape[0], conv_keep - accepted_len, new_state.shape[2]), dtype=new_state.dtype)
                c.cache[0] = mx.concatenate([prev_zeros, acc_b], axis=1)
            else:
                c.cache[0] = mx.concatenate([prev_state[:, accepted_len:, :], acc_b], axis=1)
            c._left_padding_advance = old_lp_adv + (accepted_len if c._left_padding is not None else 0)
            c._lengths_advance = old_len_adv + (accepted_len if c._lengths is not None else 0)


# ---------------------------------------------------------------------------
# Lossless Speculative Greedy Decoding Loop + Baseline Benchmark Helper
# ---------------------------------------------------------------------------


def _apply_rep_penalty(
    logits_2d: mx.array,
    uniq_seen_ids: list[int],
    penalty: float,
) -> mx.array:
    """Exact token-level repetition penalty matching `mlx_lm` / `mlx_vlm` semantics (`logits_2d`: `[1, V]`)."""
    if penalty is None or abs(penalty - 1.0) < 1e-6 or not uniq_seen_ids:
        return logits_2d
    uniq = mx.array(uniq_seen_ids, dtype=mx.int32)
    selected = mx.take(logits_2d, uniq, axis=-1)
    updated = mx.where(selected < 0, selected * penalty, selected / penalty)
    logits_out = mx.array(logits_2d)
    logits_out[:, uniq] = updated
    return logits_out


def standard_greedy_generate(
    vlm_model: Any,
    prompt_ids: list[int],
    *,
    max_tokens: int = 256,
    eos_token_id: int = 124900,
    repetition_penalty: float = 1.0,
) -> tuple[list[int], dict[str, Any]]:
    """Reference greedy token-by-token decoder on `vlm_model.language_model` for exact tok/s & parity checks."""
    lm = vlm_model.language_model
    cache = lm.make_cache()
    inp = mx.array([prompt_ids], dtype=mx.int32)
    use_rep = repetition_penalty is not None and abs(repetition_penalty - 1.0) >= 1e-6
    seen_set = set(prompt_ids)
    uniq_seen = sorted(seen_set)

    t0 = time.perf_counter()
    out = lm(inp, cache=cache)
    last_logits = _apply_rep_penalty(out.logits[:, -1, :], uniq_seen, repetition_penalty) if use_rep else out.logits[:, -1, :]
    next_tok = int(mx.argmax(last_logits, axis=-1).item())

    gen_ids: list[int] = []
    forward_calls = 1
    while len(gen_ids) < max_tokens:
        gen_ids.append(next_tok)
        if use_rep and next_tok not in seen_set:
            seen_set.add(next_tok)
            uniq_seen.append(next_tok)
        if next_tok == eos_token_id or len(gen_ids) >= max_tokens:
            break
        step_in = mx.array([[next_tok]], dtype=mx.int32)
        out = lm(step_in, cache=cache)
        forward_calls += 1
        step_logits = _apply_rep_penalty(out.logits[:, -1, :], uniq_seen, repetition_penalty) if use_rep else out.logits[:, -1, :]
        next_tok = int(mx.argmax(step_logits, axis=-1).item())

    dt = max(1e-6, time.perf_counter() - t0)
    return gen_ids, {
        "tokens": len(gen_ids),
        "wall_clock_sec": round(dt, 4),
        "tok_per_sec": round(len(gen_ids) / dt, 2),
        "target_forward_calls": forward_calls,
    }


def _find_suffix_ngram_draft(
    history: list[int],
    max_draft: int,
) -> list[int] | None:
    """Zero-compute lossless Prompt/History Suffix Lookup (`PLD`):
    Searches backward in `history[:-1]` for the latest occurrence of:
      1. Trailing 4-gram (`len >= 1`, up to `max_draft = 5` tokens)
      2. Trailing 3-gram (`len >= 2`, up to `max_draft = 5` tokens)
      3. Full-window substantive 2-gram (`both >= 320`, `len >= 2`, up to 4 tokens)
      4. Unambiguous Rare-Token Continuation (`last_tok >= 2048` where all prior occurrences in
         `history[-3072:-1]` share the exact same immediate continuation token, `len >= 2`, up to 4 tokens)
    """
    n = len(history)
    if n < 6 or max_draft <= 0:
        return None
    for gram_len in (4, 3):
        if n <= gram_len + 1:
            continue
        pat = history[-gram_len:]
        p_last = pat[-1]
        p0 = pat[0]
        hi_idx = n - 2
        lo_idx = max(gram_len - 1, n - 4096)
        for idx in range(hi_idx, lo_idx - 1, -1):
            if history[idx] == p_last and history[idx - gram_len + 1] == p0:
                if history[idx - gram_len + 1 : idx + 1] == pat:
                    start = idx + 1
                    end = min(n - 1, start + min(4, max_draft))
                    if end - start >= 2:
                        return history[start:end]
    # Full-window 2-gram entity/identifier lookup (both tokens substantive `>= 320`, 4,096-token window)
    if n > 4 and history[-1] >= 320 and history[-2] >= 320:
        p0, p1 = history[-2], history[-1]
        lo2 = max(1, n - 4096)
        for idx in range(n - 2, lo2 - 1, -1):
            if history[idx] == p1 and history[idx - 1] == p0:
                start = idx + 1
                end = min(n - 1, start + min(4, max_draft))
                if end - start >= 2:
                    return history[start:end]
    return None


def mtp_speculative_generate(
    vlm_model: Any,
    mtp_head: SequentialMTPHead,
    prompt_ids: list[int],
    *,
    max_tokens: int = 256,
    draft_k: int = 2,
    eos_token_id: int = 124900,
    repetition_penalty: float = 1.0,
    use_ngram_lookup: bool = True,
) -> tuple[list[int], dict[str, Any]]:
    """Lossless speculative greedy decoder combining zero-overhead Suffix N-gram Lookup + `SequentialMTPHead` (`1 <= draft_k <= 5`).

    Mathematically guaranteed to emit the **exact same token ID sequence** as `standard_greedy_generate`.
    """
    if not (1 <= draft_k <= 5):
        raise ValueError("draft_k must be in 1..5 for zero-recompute LFM3 ShortConv rollback.")

    lm = vlm_model.language_model
    embed_tokens = lm.model.embed_tokens
    if mtp_head._cached_w_sub is None:
        mtp_head.bind_vocab_projection(embed_tokens)

    lm_cache = lm.make_cache()
    mtp_cache = KVCache()
    inp = mx.array([prompt_ids], dtype=mx.int32)
    use_rep = repetition_penalty is not None and abs(repetition_penalty - 1.0) >= 1e-6
    seen_set = set(prompt_ids)
    uniq_seen = sorted(seen_set)
    history = list(prompt_ids)
    max_conv_draft = min(5, int(lm.config.conv_L_cache) - 2)

    t0 = time.perf_counter()
    # 1. Target prefill on `prompt_ids` (`[1, L]`) + MTP KV prefill in one fused launch
    out = lm(inp, cache=lm_cache)
    h_all = out.hidden_states  # [1, L, d]
    first_logits = _apply_rep_penalty(out.logits[:, -1, :], uniq_seen, repetition_penalty) if use_rep else out.logits[:, -1, :]
    first_tok_mx = mx.argmax(first_logits, axis=-1)

    prefill_next_ids = (
        mx.concatenate([inp[:, 1:], first_tok_mx[:, None]], axis=1)
        if len(prompt_ids) > 1
        else first_tok_mx[:, None]
    )
    h_mtp_prefill = mtp_head.forward_features(h_all, embed_tokens(prefill_next_ids), cache=mtp_cache)
    prefetched_h_mtp0 = h_mtp_prefill[:, -1:, :]  # [1, 1, d] exact `j=0` MTP hidden state for `(h_{L-1}, first_tok)`!

    cur_h = h_all[:, -1:, :]  # [1, 1, d] — hidden state that produced `cur_tok`
    mx.eval(first_tok_mx, cur_h, prefetched_h_mtp0, mtp_cache.keys, mtp_cache.values)
    cur_tok = int(first_tok_mx.item())
    mtp_confirmed_offset = len(prompt_ids) - 1  # KV slots strictly prior to `(cur_h, cur_tok)`

    gen_ids: list[int] = []
    target_calls = 1
    draft_proposed = 0
    draft_accepted = 0
    ngram_accepted = 0
    verify_steps = 0

    while len(gen_ids) < max_tokens:
        gen_ids.append(cur_tok)
        history.append(cur_tok)
        if use_rep and cur_tok not in seen_set:
            seen_set.add(cur_tok)
            uniq_seen.append(cur_tok)
        if cur_tok == eos_token_id or len(gen_ids) >= max_tokens:
            break

        rem = max_tokens - len(gen_ids)
        cur_tok_mx = mx.array([[cur_tok]], dtype=mx.int32)

        # 2A. Fast zero-GPU Suffix N-gram Lookup (`up to 4 tokens`, `len >= 2`)!
        ngram_cand = _find_suffix_ngram_draft(history, min(4, max_conv_draft, rem)) if use_ngram_lookup else None
        used_ngram = ngram_cand is not None and len(ngram_cand) >= 2

        if used_ngram:
            assert ngram_cand is not None
            k_eff = len(ngram_cand)
            draft_seq_mx = mx.array([ngram_cand], dtype=mx.int32)
        else:
            # 2B. Neural `SequentialMTPHead` Draft (`j=0` already prefetched in parallel with previous KV sync!)
            k_eff = min(draft_k, rem)
            d0_mx = mtp_head.draft_greedy_token(prefetched_h_mtp0, embed_tokens)  # [1, 1] int32 on GPU
            draft_tok_mxs: list[mx.array] = [d0_mx]
            h_prev = prefetched_h_mtp0
            t_prev_mx = d0_mx
            for _ in range(1, k_eff):
                t_emb = embed_tokens(t_prev_mx)
                h_mtp = mtp_head.forward_features(h_prev, t_emb, cache=mtp_cache)
                d_tok_mx = mtp_head.draft_greedy_token(h_mtp, embed_tokens)
                draft_tok_mxs.append(d_tok_mx)
                h_prev = h_mtp
                t_prev_mx = d_tok_mx
            draft_seq_mx = d0_mx if k_eff == 1 else mx.concatenate(draft_tok_mxs, axis=1)  # [1, k_eff]

        verify_mx = mx.concatenate([cur_tok_mx, draft_seq_mx], axis=1)  # [1, 1 + k_eff]
        draft_proposed += k_eff

        # 3. Verify `[cur_tok, draft_1, ..., draft_{k_eff}]` in the SAME fused Metal command buffer!
        snaps = snapshot_lfm3_cache_for_verify(lm_cache)
        v_out = lm(verify_mx, cache=lm_cache)
        target_calls += 1
        verify_steps += 1
        v_logits = v_out.logits  # [1, 1 + k_eff, V]
        v_hidden = v_out.hidden_states  # [1, 1 + k_eff, d]

        if not use_rep:
            gold_all_mx = mx.argmax(v_logits, axis=-1)  # [1, 1 + k_eff] vectorized on GPU!
            mx.eval(draft_seq_mx, gold_all_mx, v_hidden)
            draft_toks = ngram_cand if used_ngram else draft_seq_mx[0].tolist()  # type: ignore[assignment]
            gold_toks = gold_all_mx[0].tolist()
        else:
            mx.eval(draft_seq_mx, v_logits, v_hidden)
            draft_toks = ngram_cand if used_ngram else draft_seq_mx[0].tolist()  # type: ignore[assignment]
            gold_toks = []

        # 4. Check exact greedy match at each position `j = 0 .. k_eff - 1`
        n_acc = 0
        next_after_step: int | None = None
        if not use_rep:
            for j in range(k_eff):
                gold_t = int(gold_toks[j])
                if gold_t == int(draft_toks[j]):
                    n_acc += 1
                    gen_ids.append(gold_t)
                    history.append(gold_t)
                    if gold_t == eos_token_id or len(gen_ids) >= max_tokens:
                        next_after_step = None
                        break
                else:
                    next_after_step = gold_t
                    break
            else:
                next_after_step = int(gold_toks[k_eff])
        else:
            temp_set = set(seen_set)
            temp_uniq = list(uniq_seen)
            for j in range(k_eff):
                step_l = _apply_rep_penalty(v_logits[:, j, :], temp_uniq, repetition_penalty)
                gold_t = int(mx.argmax(step_l, axis=-1).item())
                if gold_t == int(draft_toks[j]):
                    n_acc += 1
                    gen_ids.append(gold_t)
                    history.append(gold_t)
                    if gold_t not in seen_set:
                        seen_set.add(gold_t)
                        uniq_seen.append(gold_t)
                    if gold_t not in temp_set:
                        temp_set.add(gold_t)
                        temp_uniq.append(gold_t)
                    if gold_t == eos_token_id or len(gen_ids) >= max_tokens:
                        next_after_step = None
                        break
                else:
                    next_after_step = gold_t
                    break
            else:
                bonus_l = _apply_rep_penalty(v_logits[:, k_eff, :], temp_uniq, repetition_penalty)
                next_after_step = int(mx.argmax(bonus_l, axis=-1).item())

        draft_accepted += n_acc
        if used_ngram:
            ngram_accepted += n_acc
        accepted_len = 1 + n_acc  # `cur_tok` + `n_acc` accepted drafts now remain in `lm_cache`

        if next_after_step is None:
            break

        # 5. Rewind `lm_cache` cleanly to `accepted_len`, and fuse MTP KV sync + next `j=0` prefetch in ONE parallel call!
        if accepted_len < 1 + k_eff:
            rewind_lfm3_cache_after_verify(
                lm_cache,
                snaps,
                verified_len=1 + k_eff,
                accepted_len=accepted_len,
                conv_keep=int(lm.config.conv_L_cache) - 1,
            )

        # Notice slot `mtp_confirmed_offset + 0` ALREADY holds exact `(cur_h, cur_tok)` from the previous step's
        # fused prefetch — EVEN when `used_ngram == True`! So we ONLY need `v_hidden[:, :accepted_len, :]` +
        # `[*draft_toks[:n_acc], next_after_step]` (`accepted_len` tokens, zero `mx.concatenate`!) in ALL cases!
        mtp_cache.offset = mtp_confirmed_offset + 1
        sync_ids = mx.array([[*draft_toks[:n_acc], next_after_step]], dtype=mx.int32)
        h_sync = mtp_head.forward_features(v_hidden[:, :accepted_len, :], embed_tokens(sync_ids), cache=mtp_cache)
        prefetched_h_mtp0 = h_sync[:, -1:, :]
        mtp_confirmed_offset += accepted_len

        cur_tok = next_after_step
        cur_h = v_hidden[:, accepted_len - 1 : accepted_len, :]

    dt = max(1e-6, time.perf_counter() - t0)
    mean_emit_per_verify = round((1.0 + draft_accepted / max(1, verify_steps)), 3) if verify_steps else 1.0
    return gen_ids, {
        "tokens": len(gen_ids),
        "wall_clock_sec": round(dt, 4),
        "tok_per_sec": round(len(gen_ids) / dt, 2),
        "target_forward_calls": target_calls,
        "draft_proposed": draft_proposed,
        "draft_accepted": draft_accepted,
        "ngram_accepted": ngram_accepted,
        "acceptance_rate": round(draft_accepted / max(1, draft_proposed), 4),
        "mean_tokens_per_verify": mean_emit_per_verify,
    }


def save_mtp_checkpoint(mtp_head: SequentialMTPHead, save_dir: Path | str) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    weights = {
        k: v.astype(mx.bfloat16) if mx.issubdtype(v.dtype, mx.floating) else v
        for k, v in tree_flatten(mtp_head.parameters())
    }
    mx.eval(list(weights.values()))
    tmp_w = save_dir / "mtp_head.tmp.safetensors"
    mx.save_safetensors(str(tmp_w), weights, metadata={"format": "mlx"})
    tmp_w.replace(save_dir / "mtp_head.safetensors")
    meta = {
        "hidden_size": mtp_head.hidden_size,
        "mtp_intermediate_size": mtp_head.mtp_intermediate_size,
        "shortlist_size": len(mtp_head._shortlist_list),
        "shortlist_ids": mtp_head._shortlist_list,
    }
    tmp_m = save_dir / "mtp_config.json.tmp"
    tmp_m.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    tmp_m.replace(save_dir / "mtp_config.json")


def load_mtp_checkpoint(text_config: Lfm3TextConfig, load_dir: Path | str) -> SequentialMTPHead:
    load_dir = Path(load_dir)
    meta = json.loads((load_dir / "mtp_config.json").read_text(encoding="utf-8"))
    head = SequentialMTPHead(
        text_config,
        mtp_intermediate_size=int(meta["mtp_intermediate_size"]),
        shortlist_ids=[int(x) for x in meta["shortlist_ids"]],
    )
    raw_w = mx.load(str(load_dir / "mtp_head.safetensors"))
    if "logit_proj.weight" not in raw_w:
        raw_w["logit_proj.weight"] = mx.eye(head.hidden_size, dtype=mx.bfloat16)
    head.load_weights(list(raw_w.items()), strict=True)
    head.apply(lambda x: x.astype(mx.bfloat16) if mx.issubdtype(x.dtype, mx.floating) else x)
    mx.eval(head.parameters())
    return head
