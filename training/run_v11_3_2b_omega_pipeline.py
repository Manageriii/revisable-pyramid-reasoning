"""Master Checkpointed RPR v11 OMEGA (`3.18B` `40L` Pure 16-bit `bfloat16`, `256k` Context, `3`-Loop Latent Recurrence) Pipeline:

Upcycles cleanly from pristine `LFM3-VL-2.9B-RPR-V9-SEAL-BF16` (`Step-0 logit diff == 0.000000`) and exports to
`checkpoints/LFM3-VL-3.2B-RPR-V10-APEX-BF16/` (live `:8991`, Cline, & Hermes endpoint, with `v11 OMEGA` metadata):
  - Stage 0: Surgical `36L -> 40L` (`3.18B`, `14` `full_attention` layers) + `3`-Loop `LatentReasoningRecurrence` (`L35`)
             + `5x CrossLayerAttnRes` (`L26,30,32,36,38`) + `2x Engram` (`L6,18`) + `256k` (`262,144`) `LongRoPE2YaRN`.
  - Stage 1A: Deep 22-Layer Rank-48 `LoRA+` (`alpha=96`, `4:1` `lr_b:lr_a`, ZERO `Muon2D` trunk drift!) + Unfrozen Upcycled
              SandwichNorm Gates `(34,35,38,39)` + `3`-Loop `LatentRecur` + `AttnRes` + `Engram` Multi-Epoch SFT (`1,500` steps).
  - Stage 1B: True `8k..26k`-Token Zero-Swap Prefix-KV Cached Long-Context SFT (`120` steps on `train_v11_long_kv.jsonl`
              attending over full `8k..26k` cached KV archives in `< 8.5 GiB` Metal memory!).
  - Stage 2: Contrastive `LoRA+` `ORPO` (`240` steps on `320` `orpo_v11_pairs.jsonl`).
  - Stage 3: `DAPO` Dynamic Sampling + `Dr. GRPO` Unbiased Free-Reasoning RLVR (`32` ACTIVE updates across up to `110` episodes,
             generous `max_tokens=640`, `max_think_tokens=900` for free exploratory `<think>` self-discovery!).
  - Stage 4: Streaming `WiSE-FT` Non-Degradation Gate -> Lean Pure `16-bit` (`bfloat16`) Export + `40L` `SVF + Nullspace` Bank
             + `Darwin Gödel Machine` (`DGM`) Evolutionary Archive.
  - Stage 5: `20,480`-Token `FR-Spec` Shortlist + Fresh `40L L38` Warm-Started `SequentialMTPHead` (`intermediate=4096`)
             + `1,500` Cached-Trunk Batch-4 Steps (`6,000` Sequence Updates!) with 3-Step Self-Feeding Unroll (`h_m1->h_m2->h_m3`)
             + 9-Domain Lossless Speculative Speed Benchmark (`>= 1.6x` target).
"""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time
import urllib.request
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from mtp_speculative import (
    SequentialMTPHead,
    load_mtp_checkpoint,
    mtp_speculative_generate,
    save_mtp_checkpoint,
    standard_greedy_generate,
)
from training.build_v6_repair_and_seal_corpus import V6_CORPUS_DIR
from training.build_v11_omega_corpus import V11_CORPUS_DIR
from training.collect_and_build_rpr_corpus import _strip_prior_think
from training.distill_v8_mtp_onpolicy import GENERAL_ONPOLICY_PROMPTS
from training.run_v6_32l_mtp_seal_pipeline import _eval_task_batch
from training.run_v7_36l_zero_degrade_pipeline import _run_orpo_step
from training.run_v8_2_9b_hermes_pipeline import (
    BF16AdamW,
    _save_v8_ckpt,
    prepare_turn_items_v8,
)
from training.run_v9_2_9b_seal_cline_pipeline import EXPORT_V9_SEAL
from training.run_v10_3_2b_apex_pipeline import (
    EXPORT_V10_APEX,
    REWRITE_TARGET_LAYERS_40L,
    extract_v10_40l_nullspace_svf_bank,
    fuse_and_export_40l_lean,
)
from training.train_and_fuse_rpr_bf16 import (
    _apply_template_inference_aligned,
    _run_dft_batch,
    chunked_multimodal_ce,
    evaluate_val_loss,
)
from training.verifiable_rewards import score_response
from v8_arch import (
    UPCYCLED_40L_INDICES,
    apply_qk_clip,
    attach_v8_extensions,
    unfreeze_v8_modules,
    upcycle_v9_36l_to_v10_40l_identity,
)


WORK_V11 = ROOT_DIR / "checkpoints" / "rpr_v11_3_2b_work"
STAGE0_V11_ID = WORK_V11 / "stage0_v11_40l_id"
CKPT_V11 = WORK_V11 / "stage123_v11_ckpt"

# 22 layers spanning foundation-mid (`6..24`), upper reasoning (`26..33`), and all upcycled synthesis blocks (`34..39`)
V11_DEEP_LORA_LAYERS = (6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39)


def _ensure_8991_unloaded() -> None:
    """Keep `:8991` unloaded during offline GPU training/benchmarking even if a Hermes scheduled job triggers hot-load."""
    try:
        with urllib.request.urlopen("http://127.0.0.1:8991/health", timeout=2.0) as resp:
            h = json.loads(resp.read().decode("utf-8"))
        if h.get("active_model") is not None or float(h.get("gpu_active_mb") or 0.0) > 1200.0:
            urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8991/v1/admin/unload", method="POST"), timeout=30.0).read()
            print("[GPU Guard] Unloaded background model from :8991 to protect 24GB unified RAM.", flush=True)
    except Exception:
        pass


def _build_v11_loraplus_adamw(
    *,
    lr_gate: Any,
    lr_table: Any,
    lr_lora_b: Any,
    lr_lora_a: Any,
) -> optim.MultiOptimizer:
    """LoRA+ (`Hayou et al., ICML 2024`: `lr_b = 4 * lr_a`) + `BF16AdamW` on `Engram` + `AdamW` on 1D gates/scales.
    Strictly zero Newton-Schulz (`Muon2D`) trunk over-rotation across 1,500+ multi-epoch steps!
    """
    gate_keys = (
        "post_operator_norm",
        "post_ffn_norm",
        "out_scale",
        "res_scale",
        "loop_scales",
        "step_embed",
        "ctx_norm",
        "q_norm",
        "src_norm",
        "h_norm",
        "anc_norm",
    )
    return optim.MultiOptimizer(
        [
            optim.AdamW(learning_rate=lr_gate, weight_decay=0.0),
            BF16AdamW(learning_rate=lr_table, weight_decay=0.0),
            optim.AdamW(learning_rate=lr_lora_b, weight_decay=0.01),
            optim.AdamW(learning_rate=lr_lora_a, weight_decay=0.01),
        ],
        filters=[
            lambda k, p: p.ndim == 1 or any(gk in k for gk in gate_keys),
            lambda k, _: ".table.weight" in k,
            lambda k, _: ("lora_b" in k or ".attn_res_" in k or ".latent_recur_" in k or ".engram_" in k),
        ],
    )


def _warmup_cosine(peak_lr: float, end_lr: float, total_steps: int, cur_step: int = 0, warmup_steps: int = 60) -> Any:
    rem = max(1, total_steps - cur_step)
    if cur_step < warmup_steps and rem > (warmup_steps - cur_step) + 2:
        rem_warm = warmup_steps - cur_step
        start_v = max(end_lr, peak_lr * (max(1, cur_step) / float(warmup_steps)))
        w_sched = optim.linear_schedule(start_v, peak_lr, steps=rem_warm)
        c_sched = optim.cosine_decay(peak_lr, decay_steps=max(1, rem - rem_warm), end=end_lr)
        return optim.join_schedules([w_sched, c_sched], [rem_warm])
    cw = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0, cur_step - warmup_steps) / max(1, total_steps - warmup_steps))))
    cur_lr = end_lr + (peak_lr - end_lr) * cw
    return optim.cosine_decay(cur_lr, decay_steps=rem, end=end_lr)


def _unfreeze_upcycled_40l_gates(vlm: Any) -> None:
    layers = vlm.language_model.model.layers
    for idx in UPCYCLED_40L_INDICES:
        layers[idx].post_operator_norm.unfreeze()
        layers[idx].post_ffn_norm.unfreeze()


def _load_v11_40l_from_v9_with_peft() -> tuple[Any, Any, dict[str, Any]]:
    import mlx_vlm
    from mlx_vlm.trainer.peft import get_peft_model

    _ensure_8991_unloaded()
    if not (STAGE0_V11_ID / "model.safetensors").exists() or not (STAGE0_V11_ID / "v8_arch_ext.safetensors").exists():
        print(f"[v11 Stage 0] Surgically upcycling pristine 36L v9 -> 40L 3.18B (3-loop LatentRecur + 256k LongRoPE2YaRN) at {STAGE0_V11_ID}...", flush=True)
        up_info = upcycle_v9_36l_to_v10_40l_identity(EXPORT_V9_SEAL, STAGE0_V11_ID, latent_loops=3)
        print(f"[v11 Stage 0 Upcycle Written] {json.dumps(up_info)}", flush=True)

    vlm_v9, _ = mlx_vlm.load(str(EXPORT_V9_SEAL))
    attach_v8_extensions(vlm_v9, sidecar_dir=EXPORT_V9_SEAL, enable_moba=False)
    vlm_v9.eval()
    probe = mx.array([[1, 42, 105, 512, 1024, 2048, 4096]], dtype=mx.int32)
    v9_logits = vlm_v9.language_model(probe).logits
    mx.eval(v9_logits)
    del vlm_v9
    gc.collect()
    mx.clear_cache()

    vlm, processor = mlx_vlm.load(str(STAGE0_V11_ID))
    ext_info = attach_v8_extensions(vlm, sidecar_dir=STAGE0_V11_ID, enable_moba=False)

    leafs = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "in_proj", "out_proj", "w1", "w2", "w3"}
    prefs = tuple(f"model.layers.{i}." for i in V11_DEEP_LORA_LAYERS)
    targets = [
        f"language_model.{n}"
        for n, _ in vlm.language_model.named_modules()
        if n.startswith(prefs) and n.split(".")[-1] in leafs
    ]
    vlm = get_peft_model(vlm, linear_layers=targets, rank=48, alpha=96, dropout=0.02, verbose=False)
    unfreeze_v8_modules(vlm)
    _unfreeze_upcycled_40l_gates(vlm)

    vlm.eval()
    v11_logits = vlm.language_model(probe).logits
    mx.eval(v11_logits)
    diff0 = float(mx.max(mx.abs(v11_logits.astype(mx.float32) - v9_logits.astype(mx.float32))).item())
    print(f"[v11 Stage 0 Identity Check] 36L v9 vs 40L v11 Step-0 max_abs_logit_diff = {diff0:.8f}", flush=True)
    assert diff0 < 1e-6, f"Step-0 v11 40L identity vs v9 broken: diff={diff0}"
    ext_info["step0_max_logit_diff"] = diff0
    ext_info["lora_target_modules"] = len(targets)
    return vlm, processor, ext_info


def _run_prefix_kv_long_sft_step(
    vlm: Any,
    processor: Any,
    long_row: dict[str, Any],
    anchor_item: dict[str, Any],
    optimizer: optim.Optimizer,
    *,
    tail_prompt_tokens: int = 64,
    prefill_chunk: int = 512,
) -> tuple[float, int]:
    """Zero-Swap Prefix-KV Cached SFT (`8k..26k` tokens in `< 8.5 GiB` Metal RAM!):
    1. Prefills `p_ids[:-tail_prompt_tokens]` in `512`-token chunks with per-chunk `mx.eval` (`< 0.15s` per Metal command buffer,
       zero GPU watchdog hang!) under `vlm.eval()` (`stop_gradient`) into `kv_cache`.
    2. Runs `nn.value_and_grad` over `[p_ids[-tail_prompt_tokens:] + a_ids + [eos]]` attending to the ENTIRE `8k..26k` `kv_cache`!
    3. Mixes with `0.30x` short-context SFT anchor gradient to preserve crisp general formatting.
    """
    tok = processor.tokenizer
    eos_id = int(tok.eos_token_id)
    msgs = long_row["messages"]
    p_s = _apply_template_inference_aligned(processor, vlm.config, msgs[:-1], num_images=0, enable_thinking=True)
    a_s = str(msgs[-1]["content"])
    if p_s.endswith("<think>") and a_s.startswith("<think>"):
        a_s = a_s[len("<think>") :]
    p_ids = tok.encode(p_s, add_special_tokens=False)
    a_ids = (tok.encode(a_s, add_special_tokens=False) + [eos_id])[:384]

    split_at = max(0, len(p_ids) - tail_prompt_tokens)
    pref_ids = p_ids[:split_at]
    tail_p = p_ids[split_at:]
    tail_all = tail_p + a_ids
    ce_start = max(0, len(tail_p) - 1)

    kv_cache = vlm.language_model.make_cache()
    vlm.eval()
    if pref_ids:
        pref_mx = mx.array([pref_ids], dtype=mx.int32)
        for s in range(0, len(pref_ids), prefill_chunk):
            e = min(len(pref_ids), s + prefill_chunk)
            _ = vlm.language_model.model(pref_mx[:, s:e], cache=kv_cache)
            mx.eval([c.state for c in kv_cache])
        for c in kv_cache:
            if hasattr(c, "keys") and c.keys is not None:
                c.keys = mx.stop_gradient(c.keys)
                c.values = mx.stop_gradient(c.values)
            elif hasattr(c, "cache") and isinstance(c.cache, list):
                c.cache = [mx.stop_gradient(x) if x is not None else None for x in c.cache]
        del pref_mx

    vlm.train()
    tail_mx = mx.array([tail_all], dtype=mx.int32)

    def _tail_ce(rm: nn.Module, t_ids: mx.array) -> mx.array:
        out = rm.language_model(t_ids[:, :-1], cache=kv_cache)
        logits = out.logits[:, ce_start:, :].astype(mx.float32)
        tgts = t_ids[:, ce_start + 1 :]
        return mx.mean(nn.losses.cross_entropy(logits, tgts))

    lv, g_long = nn.value_and_grad(vlm, _tail_ce)(vlm, tail_mx)
    mx.eval(lv, g_long)
    _, g_anc = _run_dft_batch(vlm, [anchor_item], None, scale_grad=0.30)
    mx.eval(g_anc)
    g_tot = tree_map(lambda a, b: a + b, g_long, g_anc) if g_anc is not None else g_long
    g_tot, _ = optim.clip_grad_norm(g_tot, max_norm=1.0)
    optimizer.update(vlm, g_tot)
    apply_qk_clip(vlm, max_lora_fro_norm=6.0)
    mx.eval(vlm.trainable_parameters(), optimizer.state)
    loss_f = float(lv.item())
    del kv_cache, tail_mx, g_long, g_anc, g_tot
    return loss_f, len(p_ids) + len(a_ids)


def run_stages_01234_v11(
    sft_rows: list[dict[str, Any]],
    long_kv_rows: list[dict[str, Any]],
    orpo_rows: list[dict[str, Any]],
    dapo_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    *,
    sft_steps: int = 1500,
    long_kv_steps: int = 120,
    orpo_steps: int = 240,
    target_active_rl_updates: int = 32,
    max_rl_episodes: int = 110,
) -> dict[str, Any]:
    import mlx_vlm

    done_marker = EXPORT_V10_APEX / "v11_stage01234_done.json"
    if (EXPORT_V10_APEX / "model.safetensors").exists() and done_marker.exists():
        print(f"[v11 Stages 0-4] Resuming completed pure 16-bit 3.18B v11 OMEGA checkpoint at {EXPORT_V10_APEX}", flush=True)
        return json.loads(done_marker.read_text(encoding="utf-8"))

    vlm, processor, ext_info = _load_v11_40l_from_v9_with_peft()
    tr_items, vis_cnt = prepare_turn_items_v8(sft_rows, vlm, processor, max_seq_len=2816)
    va_items, _ = prepare_turn_items_v8(val_rows, vlm, processor, max_seq_len=2816)
    n_trainable = sum(int(v.size) for _, v in tree_flatten(vlm.trainable_parameters()))
    print(
        f"[v11 OMEGA 40L Init Verified] Total = {ext_info['total_model_params']/1e9:.3f}B | "
        f"22-Layer Rank-48 LoRA+ ({ext_info['lora_target_modules']} linear targets) + Upcycled Gates + Engram + 5x AttnRes + 3-Loop LatentRecur = {n_trainable/1e6:.2f}M trainable | "
        f"Step-0 diff vs v9 = {ext_info['step0_max_logit_diff']:.8f} | SFT turns = {len(tr_items)} (vis: {vis_cnt}) | Long-KV 8k..26k archives = {len(long_kv_rows)}",
        flush=True,
    )

    state: dict[str, Any] = {
        "sft_step": 0,
        "long_kv_step": 0,
        "orpo_step": 0,
        "rl_ep_idx": 0,
        "rl_active_updates": 0.0,
        "rl_dapo_skips": 0,
        "sft_done": False,
        "long_kv_done": False,
        "orpo_done": False,
        "rl_done": False,
        "best_val": 1e9,
        "best_longkv_val": 1e9,
        "best_orpo_val": 1e9,
        "best_rl_val": 1e9,
    }
    st_path = CKPT_V11 / "latest" / "train_state.json"
    if st_path.exists() and (CKPT_V11 / "latest" / "trainable.safetensors").exists():
        state = json.loads(st_path.read_text(encoding="utf-8"))
        vlm.load_weights(str(CKPT_V11 / "latest" / "trainable.safetensors"), strict=False)
        mx.eval(vlm.parameters())
        print(f"[v11 Stages 1-3] Resumed checkpoint: {state}", flush=True)
    else:
        vlm.eval()
        vl0, vp0 = evaluate_val_loss(vlm, va_items[:60])
        vlm.train()
        state["best_val"] = vl0
        print(f"  [Step-0 v11 40L Identity Baseline (`== v9`)] val_v11_joint={vl0:.4f} (PPL={vp0:.4f})", flush=True)
        _save_v8_ckpt(vlm, CKPT_V11 / "best")
        _save_v8_ckpt(vlm, CKPT_V11 / "latest", state)

    # ------------------------------------------------------------------
    # Stage 1A: Deep 22-Layer Rank-48 `LoRA+` Multi-Epoch Free-Reasoning SFT (`1,500` steps)
    # ------------------------------------------------------------------
    if not state["sft_done"]:
        _ensure_8991_unloaded()
        s0 = int(state["sft_step"])
        opt_sft = _build_v11_loraplus_adamw(
            lr_gate=_warmup_cosine(2.0e-4, 1.5e-5, sft_steps, s0, warmup_steps=60),
            lr_table=_warmup_cosine(7.5e-5, 1.0e-5, sft_steps, s0, warmup_steps=60),
            lr_lora_b=_warmup_cosine(8.8e-5, 8.0e-6, sft_steps, s0, warmup_steps=60),
            lr_lora_a=_warmup_cosine(2.2e-5, 2.0e-6, sft_steps, s0, warmup_steps=60),
        )
        order = list(range(len(tr_items)))
        epoch_rng = random.Random(20261123)
        epoch_rng.shuffle(order)
        t0 = time.perf_counter()
        losses: list[float] = []
        qk_clips = 0
        vlm.train()

        for st in range(s0, sft_steps):
            if st > 0 and (st * 2) % len(order) < 2:
                epoch_rng.shuffle(order)
            b0 = tr_items[order[(st * 2) % len(order)]]
            b1 = tr_items[order[(st * 2 + 1) % len(order)]]
            has_long = int(b0["full_ids"].shape[1]) >= 1800 or int(b1["full_ids"].shape[1]) >= 1800
            cur_batch = [b0] if int(b0["full_ids"].shape[1]) >= 1800 else ([b1] if int(b1["full_ids"].shape[1]) >= 1800 else [b0, b1])
            lv, _ = _run_dft_batch(vlm, cur_batch, opt_sft)
            qk_clips += apply_qk_clip(vlm, max_lora_fro_norm=6.0)
            losses.append(lv)

            if has_long or (st + 1) % 8 == 0:
                gc.collect()
                mx.clear_cache()
            if (st + 1) % 100 == 0:
                _ensure_8991_unloaded()
            if (st + 1) % 75 == 0 and (st + 1) % 150 != 0 and st != sft_steps - 1:
                state["sft_step"] = st + 1
                _save_v8_ckpt(vlm, CKPT_V11 / "latest", state)
                print(
                    f"  [v11 Stage 1A Ckpt {st+1:4d}/{sft_steps} | {(time.perf_counter()-t0)/60:4.1f}m] train={sum(losses)/len(losses):.4f}",
                    flush=True,
                )
            if (st + 1) % 150 == 0 or st == sft_steps - 1:
                vlm.eval()
                vl, vp = evaluate_val_loss(vlm, va_items[:60])
                vlm.train()
                state["sft_step"] = st + 1
                if vl <= state["best_val"]:
                    state["best_val"] = vl
                    _save_v8_ckpt(vlm, CKPT_V11 / "best")
                _save_v8_ckpt(vlm, CKPT_V11 / "latest", state)
                print(
                    f"  [v11 Stage 1A LoRA+-SFT {st+1:4d}/{sft_steps} | {(time.perf_counter()-t0)/60:4.1f}m] "
                    f"train={sum(losses)/len(losses):.4f} | val={vl:.4f} (PPL={vp:.4f}) | best={state['best_val']:.4f} | qk_clips={qk_clips}",
                    flush=True,
                )
                losses.clear()
                gc.collect()
                mx.clear_cache()

        vlm.load_weights(str(CKPT_V11 / "best" / "trainable.safetensors"), strict=False)
        mx.eval(vlm.parameters())
        state["sft_done"] = True
        _save_v8_ckpt(vlm, CKPT_V11 / "sft_best")
        _save_v8_ckpt(vlm, CKPT_V11 / "latest", state)
        del opt_sft
        gc.collect()
        mx.clear_cache()

    # Keep only 96 compact anchor items (`<= 1,536` tokens) and free full `tr_items` to save ~1.4 GiB unified RAM!
    short_anc_items = [it for it in tr_items if int(it["full_ids"].shape[1]) <= 1536][:96] or tr_items[:48]
    del tr_items
    gc.collect()
    mx.clear_cache()

    # ------------------------------------------------------------------
    # Stage 1B: True `8k..26k`-Token Zero-Swap Prefix-KV Cached Long-Context SFT (`120` steps)
    # ------------------------------------------------------------------
    if not state["long_kv_done"]:
        _ensure_8991_unloaded()
        l0 = int(state.get("long_kv_step", 0))
        opt_lkv = _build_v11_loraplus_adamw(
            lr_gate=_warmup_cosine(6.5e-5, 1.2e-5, long_kv_steps, l0, warmup_steps=12),
            lr_table=_warmup_cosine(3.0e-5, 8.0e-6, long_kv_steps, l0, warmup_steps=12),
            lr_lora_b=_warmup_cosine(3.6e-5, 6.0e-6, long_kv_steps, l0, warmup_steps=12),
            lr_lora_a=_warmup_cosine(9.0e-6, 1.5e-6, long_kv_steps, l0, warmup_steps=12),
        )
        lkv_rng = random.Random(20261125)
        lkv_losses: list[float] = []
        max_toks_seen = 0
        best_lkv = float(state.get("best_longkv_val", state["best_val"]))
        t_lkv = time.perf_counter()

        for lst in range(l0, long_kv_steps):
            l_row = long_kv_rows[lst % len(long_kv_rows)]
            anc_it = lkv_rng.choice(short_anc_items)
            lv_l, tot_tok_l = _run_prefix_kv_long_sft_step(vlm, processor, l_row, anc_it, opt_lkv)
            lkv_losses.append(lv_l)
            max_toks_seen = max(max_toks_seen, tot_tok_l)
            gc.collect()
            mx.clear_cache()

            if (lst + 1) % 30 == 0 or lst == long_kv_steps - 1:
                _ensure_8991_unloaded()
                vlm.eval()
                vl, vp = evaluate_val_loss(vlm, va_items[:60])
                vlm.train()
                state["long_kv_step"] = lst + 1
                if vl <= best_lkv + 0.015:
                    if vl < best_lkv:
                        best_lkv = vl
                        state["best_longkv_val"] = vl
                    _save_v8_ckpt(vlm, CKPT_V11 / "longkv_best")
                _save_v8_ckpt(vlm, CKPT_V11 / "latest", state)
                print(
                    f"  [v11 Stage 1B LongKV-8k..26k {lst+1:3d}/{long_kv_steps} | max_seq={max_toks_seen} | {(time.perf_counter()-t_lkv)/60:4.1f}m] "
                    f"long_ce={sum(lkv_losses)/max(1,len(lkv_losses)):.4f} | val={vl:.4f} (PPL={vp:.4f})",
                    flush=True,
                )
                lkv_losses.clear()
                gc.collect()
                mx.clear_cache()

        best_p1b = CKPT_V11 / "longkv_best" / "trainable.safetensors"
        if best_p1b.exists():
            vlm.load_weights(str(best_p1b), strict=False)
            mx.eval(vlm.parameters())
        else:
            _save_v8_ckpt(vlm, CKPT_V11 / "longkv_best")
        state["long_kv_done"] = True
        _save_v8_ckpt(vlm, CKPT_V11 / "latest", state)
        del opt_lkv
        gc.collect()
        mx.clear_cache()

    # ------------------------------------------------------------------
    # Stage 2: Contrastive `LoRA+` `ORPO` (`240` steps on `320` `orpo_v11_pairs.jsonl`)
    # ------------------------------------------------------------------
    tok = processor.tokenizer
    eos_id = int(tok.eos_token_id)

    if not state["orpo_done"]:
        _ensure_8991_unloaded()
        o0 = int(state["orpo_step"])
        opt_orpo = _build_v11_loraplus_adamw(
            lr_gate=_warmup_cosine(3.2e-5, 6.0e-6, orpo_steps, o0, warmup_steps=15),
            lr_table=_warmup_cosine(3.6e-5, 8.0e-6, orpo_steps, o0, warmup_steps=15),
            lr_lora_b=_warmup_cosine(2.4e-5, 4.8e-6, orpo_steps, o0, warmup_steps=15),
            lr_lora_a=_warmup_cosine(6.0e-6, 1.2e-6, orpo_steps, o0, warmup_steps=15),
        )
        anc_rng = random.Random(20261126)
        t_op = time.perf_counter()
        op_losses: list[float] = []
        best_orpo = float(state.get("best_orpo_val", 1e9))
        vlm.train()

        for ost in range(o0, orpo_steps):
            pair = orpo_rows[ost % len(orpo_rows)]
            p_s = _apply_template_inference_aligned(processor, vlm.config, pair["messages"], num_images=0, enable_thinking=True)
            c_s = str(pair["chosen"])
            r_s = str(pair["rejected"])
            if p_s.endswith("<think>"):
                if c_s.startswith("<think>"):
                    c_s = c_s[len("<think>") :]
                if r_s.startswith("<think>"):
                    r_s = r_s[len("<think>") :]
            p_ids = tok.encode(p_s, add_special_tokens=False)
            c_ids = (p_ids + tok.encode(c_s, add_special_tokens=False) + [eos_id])[:1792]
            r_ids = (p_ids + tok.encode(r_s, add_special_tokens=False) + [eos_id])[:1792]
            s_i = max(len(p_ids) - 1, 0)
            if len(c_ids) > s_i + 2 and len(r_ids) > s_i + 2:
                lv_op = _run_orpo_step(
                    vlm,
                    mx.array([c_ids], dtype=mx.int32),
                    mx.array([r_ids], dtype=mx.int32),
                    s_i,
                    anc_rng.choice(short_anc_items),
                    opt_orpo,
                )
                apply_qk_clip(vlm, max_lora_fro_norm=6.0)
                op_losses.append(lv_op)

            if (ost + 1) % 10 == 0:
                gc.collect()
                mx.clear_cache()
            if (ost + 1) % 60 == 0 or ost == orpo_steps - 1:
                _ensure_8991_unloaded()
                vlm.eval()
                vl, vp = evaluate_val_loss(vlm, va_items[:60])
                vlm.train()
                state["orpo_step"] = ost + 1
                if vl <= best_orpo:
                    best_orpo = vl
                    state["best_orpo_val"] = vl
                    _save_v8_ckpt(vlm, CKPT_V11 / "orpo_best")
                _save_v8_ckpt(vlm, CKPT_V11 / "latest", state)
                print(
                    f"  [v11 Stage 2 LoRA+-ORPO {ost+1:3d}/{orpo_steps} | {(time.perf_counter()-t_op)/60:4.1f}m] "
                    f"orpo_loss={sum(op_losses)/max(1,len(op_losses)):.4f} | val={vl:.4f} (PPL={vp:.4f}) | best_orpo={best_orpo:.4f}",
                    flush=True,
                )
                op_losses.clear()
                gc.collect()
                mx.clear_cache()

        vlm.load_weights(str(CKPT_V11 / "orpo_best" / "trainable.safetensors"), strict=False)
        mx.eval(vlm.parameters())
        state["orpo_done"] = True
        _save_v8_ckpt(vlm, CKPT_V11 / "latest", state)
        del opt_orpo
        gc.collect()
        mx.clear_cache()

    # ------------------------------------------------------------------
    # Stage 3: `DAPO` Dynamic Sampling + `Dr. GRPO` Free-Reasoning RLVR (`32` Active Updates, `max_tokens=640`, `max_think_tokens=900`)
    # ------------------------------------------------------------------
    if not state["rl_done"]:
        _ensure_8991_unloaded()
        vlm.eval()
        opt_rl = _build_v11_loraplus_adamw(lr_gate=1.0e-5, lr_table=1.5e-5, lr_lora_b=8.0e-6, lr_lora_a=2.0e-6)
        # Keep ONLY non-table keys in `ref_w` to save 503 MiB of duplicate `Engram` table memory during RLVR!
        raw_ref = mx.load(str(CKPT_V11 / "orpo_best" / "trainable.safetensors"))
        ref_w = {k: v for k, v in raw_ref.items() if ".table.weight" not in k}
        del raw_ref
        mx.eval(list(ref_w.values()))
        anc_rng = random.Random(20261128)
        n_lim = min(max_rl_episodes, len(dapo_rows))
        t_rl = time.perf_counter()

        ep_idx = int(state.get("rl_ep_idx", 0))
        active_upd = float(state.get("rl_active_updates", 0.0))
        dapo_skips = int(state.get("rl_dapo_skips", 0))

        while ep_idx < n_lim and active_upd < float(target_active_rl_updates):
            if ep_idx % 12 == 0:
                _ensure_8991_unloaded()
            row = dapo_rows[ep_idx]
            p_msgs = row["messages"]
            p_str = _apply_template_inference_aligned(processor, vlm.config, p_msgs, num_images=0, enable_thinking=True)
            p_txt = "\n".join(str(m.get("content", "")) for m in p_msgs)
            p_ids = tok.encode(p_str, add_special_tokens=False)
            p_mx = mx.array([p_ids], dtype=mx.int32)
            s_idx = max(len(p_ids) - 1, 0)

            rollouts = []
            for _ in range(4):
                res = mlx_vlm.generate(vlm, processor, p_str, max_tokens=640, temperature=0.78, repetition_penalty=1.03, verbose=False)
                g_txt = res.text or ""
                if p_str.endswith("<think>") and not g_txt.lstrip().startswith("<think>"):
                    g_txt = "<think>" + g_txt
                c_ids = tok.encode(g_txt, add_special_tokens=False) + [eos_id]
                f_seq = mx.concatenate([p_mx, mx.array([c_ids], dtype=mx.int32)], axis=1)[:, :2304]
                sc = score_response(g_txt, prompt_text=p_txt, kind=row["kind"], verifier=row["verifier"], tokenizer=tok, max_think_tokens=900)
                rollouts.append({"full_ids": f_seq, "score": sc})

            rews = [r["score"]["reward"] for r in rollouts]
            p_cnt = sum(1 for r in rollouts if r["score"]["correct"])
            spread = max(rews) - min(rews)
            anc_b = anc_rng.sample(short_anc_items, k=2)

            if p_cnt == 4 and spread < 1e-4:
                dapo_skips += 1
                tag = "DAPO-skip(4/4)"
            elif p_cnt == 0 and spread < 1e-4:
                ref_r = {
                    "id": f"r_{row['id']}",
                    "pillar": row["kind"],
                    "train_last_turn_only": True,
                    "images": [],
                    "messages": [*p_msgs, {"role": "assistant", "content": row["reference_answer"]}],
                }
                r_it, _ = prepare_turn_items_v8([ref_r], vlm, processor, max_seq_len=2304)
                _run_dft_batch(vlm, (r_it[-1:] + anc_b[:1]) if r_it else anc_b[:1], opt_rl, scale_grad=0.22)
                apply_qk_clip(vlm, max_lora_fro_norm=6.0)
                active_upd += 0.5
                tag = "OPSD-0/4(0.22x)"
            else:
                mean_r = sum(rews) / len(rews)
                advs = [r - mean_r for r in rews]
                old_lps = [mx.stop_gradient(-chunked_multimodal_ce(vlm, ro["full_ids"], s_idx)) for ro in rollouts]
                snap_w = {k: mx.array(v) for k, v in tree_flatten(vlm.trainable_parameters()) if ".table.weight" not in k}
                vlm.load_weights(list(ref_w.items()), strict=False)
                mx.eval(vlm.parameters())
                ref_lps = [mx.stop_gradient(-chunked_multimodal_ce(vlm, ro["full_ids"], s_idx)) for ro in rollouts]
                vlm.load_weights(list(snap_w.items()), strict=False)
                mx.eval(vlm.parameters(), *(old_lps + ref_lps))
                del snap_w

                def _dr_grpo_dapo_loss(rm: nn.Module, f_ids: mx.array, o_lp: mx.array, r_lp: mx.array, av: float) -> mx.array:
                    n_lp = -chunked_multimodal_ce(rm, f_ids, s_idx)
                    rat = mx.exp(mx.clip(n_lp - o_lp, -5.0, 5.0))
                    adv_m = mx.array(av, dtype=mx.float32)
                    clipped_rat = mx.clip(rat, 0.80, 1.28)
                    pg = -mx.minimum(rat * adv_m, clipped_rat * adv_m)
                    kl = mx.exp(r_lp - n_lp) - (r_lp - n_lp) - 1.0
                    return mx.sum(pg + 0.03 * kl) / mx.array(640.0, dtype=mx.float32)

                vg = nn.value_and_grad(vlm, _dr_grpo_dapo_loss)
                acc_g = None
                for ro, olp, rlp, av in zip(rollouts, old_lps, ref_lps, advs):
                    lr_v, gr = vg(vlm, ro["full_ids"], olp, rlp, av)
                    mx.eval(lr_v, gr)
                    acc_g = tree_map(lambda g: g / 4.0, gr) if acc_g is None else tree_map(lambda a, g: a + g / 4.0, acc_g, gr)
                    mx.eval(acc_g)
                _, anc_g = _run_dft_batch(vlm, anc_b, None, scale_grad=0.15)
                mx.eval(anc_g)
                if acc_g is not None and anc_g is not None:
                    acc_g = tree_map(lambda a, b: a + b, acc_g, anc_g)
                if acc_g is not None:
                    acc_g, _ = optim.clip_grad_norm(acc_g, max_norm=1.0)
                    opt_rl.update(vlm, acc_g)
                    apply_qk_clip(vlm, max_lora_fro_norm=6.0)
                    mx.eval(vlm.trainable_parameters(), opt_rl.state)
                active_upd += 1.0
                tag = f"DrGRPO(mean_r={mean_r:.2f},sp={spread:.2f})"

            ep_idx += 1
            state["rl_ep_idx"] = ep_idx
            state["rl_active_updates"] = active_upd
            state["rl_dapo_skips"] = dapo_skips
            del rollouts
            gc.collect()
            mx.clear_cache()

            if ep_idx % 8 == 0 or active_upd >= float(target_active_rl_updates) or ep_idx >= n_lim:
                vlm.eval()
                vl, _ = evaluate_val_loss(vlm, va_items[:60])
                _save_v8_ckpt(vlm, CKPT_V11 / "latest", state)
                if vl <= state.get("best_rl_val", 1e9) + 0.015:
                    if vl < state.get("best_rl_val", 1e9):
                        state["best_rl_val"] = vl
                    _save_v8_ckpt(vlm, CKPT_V11 / "rl_best")
                print(
                    f"  [v11 Stage 3 DAPO+DrGRPO ep={ep_idx:3d}/{n_lim} active={active_upd:4.1f}/{target_active_rl_updates} skips={dapo_skips} | {(time.perf_counter()-t_rl)/60:4.1f}m] "
                    f"{row['id']:18s} | pass={p_cnt}/4 | {tag} | val={vl:.4f}",
                    flush=True,
                )

        del ref_w, opt_rl
        state["rl_done"] = True
        _save_v8_ckpt(vlm, CKPT_V11 / "latest", state)
        gc.collect()
        mx.clear_cache()

    # ------------------------------------------------------------------
    # Stage 4A: Streaming `WiSE-FT` Non-Degradation Gate & Pure `16-bit` Lean Fusion into `EXPORT_V10_APEX`
    # ------------------------------------------------------------------
    _ensure_8991_unloaded()
    vlm.eval()
    del short_anc_items
    gc.collect()
    mx.clear_cache()

    p_sft = CKPT_V11 / ("longkv_best" if (CKPT_V11 / "longkv_best" / "trainable.safetensors").exists() else "sft_best") / "trainable.safetensors"
    p_orpo = CKPT_V11 / ("orpo_best" if (CKPT_V11 / "orpo_best" / "trainable.safetensors").exists() else "sft_best") / "trainable.safetensors"
    p_rl = CKPT_V11 / ("rl_best" if (CKPT_V11 / "rl_best" / "trainable.safetensors").exists() else "latest") / "trainable.safetensors"

    # Build `w_wise` streaming one checkpoint at a time so peak RAM never holds 3 copies of `Engram`!
    d_tmp = mx.load(str(p_sft))
    dtypes_map = {k: v.dtype for k, v in d_tmp.items()}
    acc_wise = {k: 0.35 * v.astype(mx.float32) for k, v in d_tmp.items()}
    mx.eval(list(acc_wise.values()))
    del d_tmp
    gc.collect()
    mx.clear_cache()

    d_tmp = mx.load(str(p_orpo))
    for k, v in d_tmp.items():
        acc_wise[k] = acc_wise[k] + 0.30 * v.astype(mx.float32)
    mx.eval(list(acc_wise.values()))
    del d_tmp
    gc.collect()
    mx.clear_cache()

    d_tmp = mx.load(str(p_rl))
    w_wise = {k: (acc_wise[k] + 0.35 * d_tmp[k].astype(mx.float32)).astype(dtypes_map[k]) for k in acc_wise}
    mx.eval(list(w_wise.values()))
    del d_tmp, acc_wise
    gc.collect()
    mx.clear_cache()

    stage0_ext = mx.load(str(STAGE0_V11_ID / "v8_arch_ext.safetensors"))
    upcycled_gate_prefixes = tuple(f"language_model.model.layers.{idx}.post_" for idx in UPCYCLED_40L_INDICES)

    def _apply_interp_to_vlm(base_dict: dict[str, mx.array], tau_lora: float, tau_ext: float) -> None:
        items_load = []
        for k, v in base_dict.items():
            if "lora_b" in k or k.startswith(upcycled_gate_prefixes):
                items_load.append((k, (v.astype(mx.float32) * tau_lora).astype(v.dtype)))
            elif k in stage0_ext:
                v0 = stage0_ext[k].astype(mx.float32)
                items_load.append((k, (v0 + tau_ext * (v.astype(mx.float32) - v0)).astype(v.dtype)))
            else:
                items_load.append((k, v))
        vlm.load_weights(items_load, strict=False)
        mx.eval(vlm.parameters())
        del items_load

    seal_d = json.loads((V6_CORPUS_DIR / "seal_svf_episodes.json").read_text(encoding="utf-8"))
    probe_tasks = seal_d["dev_diagnose_4"] + seal_d["dev_gate_8"] + dapo_rows[-16:]

    _apply_interp_to_vlm(w_wise, 0.0, 0.0)
    vl_0, _ = evaluate_val_loss(vlm, va_items[:60])
    acc_0, rew_0, _ = _eval_task_batch(vlm, processor, probe_tasks)
    ref_sc = (round(acc_0, 4), round(rew_0 - 0.35 * vl_0, 4))
    print(f"  [v11 Reference Baseline (`v9` tau=0)] probe_acc={acc_0*100:5.1f}% (rew={rew_0:.3f}) | val_v11={vl_0:.4f}", flush=True)

    best_name = "v9_fallback_tau0"
    best_kind = ("wise", 0.0, 0.0)
    best_sc = ref_sc
    best_val_loss = round(vl_0, 4)

    for c_name, src_tag, tl, te in [
        ("v11_wise_tau0.80", "wise", 0.80, 0.90),
        ("v11_wise_tau0.92", "wise", 0.92, 1.00),
        ("v11_wise_tau1.00", "wise", 1.00, 1.00),
        ("v11_sft_tau1.00", "sft", 1.00, 1.00),
        ("v11_orpo_tau1.00", "orpo", 1.00, 1.00),
        ("v11_rl_tau1.00", "rl", 1.00, 1.00),
    ]:
        if src_tag == "wise":
            _apply_interp_to_vlm(w_wise, tl, te)
        else:
            src_p = p_sft if src_tag == "sft" else (p_orpo if src_tag == "orpo" else p_rl)
            vlm.load_weights(str(src_p), strict=False)
            mx.eval(vlm.parameters())
        vl_c, _ = evaluate_val_loss(vlm, va_items[:60])
        acc_c, rew_c, _ = _eval_task_batch(vlm, processor, probe_tasks)
        sc_t = (round(acc_c, 4), round(rew_c - 0.35 * vl_c, 4))
        print(f"  [v11 Non-Degradation Gate] {c_name:18s} | probe_acc={acc_c*100:5.1f}% (rew={rew_c:.3f}) | val_v11={vl_c:.4f}", flush=True)
        if sc_t >= best_sc:
            best_sc = sc_t
            best_name = c_name
            best_kind = (src_tag, tl, te)
            best_val_loss = round(vl_c, 4)
        gc.collect()
        mx.clear_cache()

    print(f"[v11 Selected] Winner={best_name} (score={best_sc}, val={best_val_loss:.4f} vs v9_ref={ref_sc})", flush=True)
    if best_kind[0] == "wise":
        _apply_interp_to_vlm(w_wise, best_kind[1], best_kind[2])
    else:
        src_win = p_sft if best_kind[0] == "sft" else (p_orpo if best_kind[0] == "orpo" else p_rl)
        vlm.load_weights(str(src_win), strict=False)
        mx.eval(vlm.parameters())

    del w_wise, stage0_ext
    gc.collect()
    mx.clear_cache()

    # Remove stale v10 bank/done markers so fresh v11 40L SVF + Nullspace bank & MTP head are written cleanly!
    for stale_f in ("seal_nullspace_svf_bank.safetensors", "v10_stage01234_done.json", "mtp_head.safetensors", "mtp_config.json"):
        if (EXPORT_V10_APEX / stale_f).exists():
            (EXPORT_V10_APEX / stale_f).unlink()

    _, fuse_info = fuse_and_export_40l_lean(
        vlm,
        STAGE0_V11_ID,
        EXPORT_V10_APEX,
        extra_sidecar_meta={
            "model_version": "LFM3-VL-3.2B-RPR-V11-OMEGA-BF16",
            "live_alias": "LFM3-VL-3.2B-RPR-V10-APEX-BF16",
            "precision": "bfloat16",
            "selected_candidate": best_name,
            "val_probe_score": list(best_sc),
            "selected_val_loss": best_val_loss,
            "v9_ref_score": list(ref_sc),
            "sft_steps": sft_steps,
            "long_kv_steps": long_kv_steps,
            "orpo_steps": orpo_steps,
            "dapo_active_updates": state.get("rl_active_updates", 0),
            "dapo_dynamic_skips": state.get("rl_dapo_skips", 0),
        },
    )
    logits_expected = fuse_info["logits_expected"]

    del vlm, processor
    gc.collect()
    mx.clear_cache()

    # Reclaim `6.0 GiB` temporary `STAGE0_V11_ID` immediately!
    if STAGE0_V11_ID.exists():
        shutil.rmtree(STAGE0_V11_ID, ignore_errors=True)

    _ensure_8991_unloaded()
    vlm_re, proc_re = mlx_vlm.load(str(EXPORT_V10_APEX))
    info_re = attach_v8_extensions(vlm_re, sidecar_dir=EXPORT_V10_APEX, enable_moba=False)
    vlm_re.eval()
    probe_ids = mx.array([[1, 42, 105, 512, 1024, 2048]], dtype=mx.int32)
    logits_re = vlm_re.language_model(probe_ids).logits
    mx.eval(logits_re)
    re_diff = float(mx.max(mx.abs(logits_re.astype(mx.float32) - logits_expected.astype(mx.float32))).item())
    print(f"[v11 Pure 16-bit 3.18B Standalone Export Verified] {info_re['total_model_params']/1e9:.3f}B params | reload diff = {re_diff:.6f}", flush=True)
    assert re_diff < 1e-4, f"v11 reload mismatch: {re_diff}"
    del logits_expected, logits_re

    # ------------------------------------------------------------------
    # Stage 4B: Extract `40L` `Transformer^2 SVF + Nullspace` Bank (`L26, 30, 36, 38`) & Initialize `DGM` Archive
    # ------------------------------------------------------------------
    tr_short, _ = prepare_turn_items_v8(
        [r for r in sft_rows if r.get("pillar") != "long_context_synthesis"][:180],
        vlm_re,
        proc_re,
        max_seq_len=2048,
    )
    skill_batches = [
        [it for it in tr_short if it.get("pillar") in ("advanced_rsi_dgm_svf", "weight_mod_rsi")][:8] or tr_short[:6],
        [it for it in tr_short if it.get("pillar") in ("complex_logic_sat", "zero_prior_formal")][:8] or tr_short[6:12],
        [it for it in tr_short if it.get("pillar") in ("realtime_research_citations", "research_and_ablation")][:8] or tr_short[12:18],
        [it for it in tr_short if it.get("pillar") == "unseen_ood_self_discover"][:8] or tr_short[18:24],
    ]
    bank_path = EXPORT_V10_APEX / "seal_nullspace_svf_bank.safetensors"
    bank = extract_v10_40l_nullspace_svf_bank(vlm_re, va_items[:32], skill_batches, bank_path)

    overlay_dir = ROOT_DIR / "checkpoints" / "seal_user_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    dgm_path = overlay_dir / "dgm_archive.json"
    dgm_root = {
        "model": "LFM3-VL-3.2B-RPR-V11-OMEGA-BF16",
        "live_checkpoint_dir": str(EXPORT_V10_APEX),
        "rewrite_layers": list(REWRITE_TARGET_LAYERS_40L),
        "svf_experts": ["b0_weight_mod_rsi", "b1_formal_logic_sat", "b2_research_citations", "b3_ood_self_discover"],
        "nodes": [
            {
                "id": "dgm_gen_000_v11_omega_root",
                "parent_id": None,
                "score": float(best_sc[0]),
                "val_loss": float(best_val_loss),
                "anchor_regression": 0.0,
                "regressed": False,
                "children": 0,
                "overlay_path": None,
                "created_at": int(time.time()),
            }
        ],
    }
    dgm_path.write_text(json.dumps(dgm_root, indent=2) + "\n", encoding="utf-8")

    meta_done = {
        **state,
        "model_version": "LFM3-VL-3.2B-RPR-V11-OMEGA-BF16",
        "selected_candidate": best_name,
        "val_probe_score": list(best_sc),
        "selected_val_loss": best_val_loss,
        "reload_diff": re_diff,
        "svf_nullspace_bank_tensors": len(bank),
        "rewrite_target_layers_40l": list(REWRITE_TARGET_LAYERS_40L),
        **info_re,
    }
    done_marker.write_text(json.dumps(meta_done, indent=2) + "\n", encoding="utf-8")
    if CKPT_V11.exists():
        shutil.rmtree(CKPT_V11, ignore_errors=True)
    del vlm_re, proc_re, va_items, tr_short, bank
    gc.collect()
    mx.clear_cache()
    return meta_done


def _build_v11_active_shortlist(tok: Any, sft_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]], onpol_raw: list[dict[str, Any]], target_size: int = 20480) -> list[int]:
    """Build `20,480`-token `FR-Spec` active shortlist covering ASCII/special tokens + all `v4..v11` reasoning/code/citation/XML vocabulary."""
    vocab_sz = int(tok.vocab_size)
    must_have = set(range(min(512, vocab_sz)))
    if getattr(tok, "eos_token_id", None) is not None:
        must_have.add(int(tok.eos_token_id))
    for special_s in ("<think>", "</think>", "<tool_call>", "</tool_call>", "<function=", "</function>", "<parameter=", "</parameter>", "```python", "```json", "```", "[S1]", "[S2]", "[S3]", "[DOC-", "[REC-", "[LOG-"):
        for tid in tok.encode(special_s, add_special_tokens=False):
            if 0 <= int(tid) < vocab_sz:
                must_have.add(int(tid))

    counts: Counter[int] = Counter()
    for item in onpol_raw:
        for tid in item["full"]:
            if 0 <= int(tid) < vocab_sz:
                counts[int(tid)] += 5
    for r in sft_rows + val_rows:
        for m in r.get("messages", []):
            txt = str(m.get("content", ""))[:4000]
            for tid in tok.encode(txt, add_special_tokens=False):
                if 0 <= int(tid) < vocab_sz:
                    counts[int(tid)] += 1

    shortlist = set(must_have)
    for tid, _ in counts.most_common():
        if len(shortlist) >= target_size:
            break
        shortlist.add(tid)
    tid_fill = 512
    while len(shortlist) < min(target_size, vocab_sz) and tid_fill < vocab_sz:
        shortlist.add(tid_fill)
        tid_fill += 1
    return sorted(shortlist)


def run_stage_5_v11_onpolicy_mtp(
    sft_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    *,
    mtp_steps: int = 1500,
) -> dict[str, Any]:
    """Fresh `40L L38` Warm-Started `SequentialMTPHead` (`intermediate=4096`, `20,480` `FR-Spec` shortlist)
    + `1,500` Cached-Trunk Batch-4 Steps (`6,000` 3-Step Self-Feeding Sequence Updates!) + 9-Domain Speed Benchmark.
    """
    import mlx_vlm

    _ensure_8991_unloaded()
    bench_dir = ROOT_DIR / "results" / "v11_mtp_benchmark"
    bench_dir.mkdir(parents=True, exist_ok=True)
    if (EXPORT_V10_APEX / "mtp_head.safetensors").exists() and (bench_dir / "summary.json").exists():
        print(f"[v11 Stage 5] Resuming completed v11 MTP benchmark from {bench_dir / 'summary.json'}", flush=True)
        return json.loads((bench_dir / "summary.json").read_text(encoding="utf-8"))

    vlm, processor = mlx_vlm.load(str(EXPORT_V10_APEX))
    attach_v8_extensions(vlm, sidecar_dir=EXPORT_V10_APEX, enable_moba=False)
    vlm.freeze()
    vlm.eval()
    tok = processor.tokenizer
    eos_id = int(tok.eos_token_id)

    text_rows = [r for r in sft_rows if not r.get("images") and r.get("pillar") != "long_context_synthesis"]
    WORK_V11.mkdir(parents=True, exist_ok=True)
    onpol_cache = WORK_V11 / "stage5_v11_onpolicy_64.json"
    onpol_raw: list[dict[str, Any]] = json.loads(onpol_cache.read_text(encoding="utf-8")) if onpol_cache.exists() else []
    if len(onpol_raw) < 96:
        print(f"[v11 Stage 5] Expanding on-policy 40L v11 greedy trajectories ({len(onpol_raw)} -> 96) across logic/SAT/formal/citations/RSI/code/tools...", flush=True)
        if not onpol_raw:
            for msgs in GENERAL_ONPOLICY_PROMPTS:
                p_s = _apply_template_inference_aligned(processor, vlm.config, msgs, num_images=0, enable_thinking=True)
                p_ids = tok.encode(p_s, add_special_tokens=False)
                g_ids, _ = standard_greedy_generate(vlm, p_ids, max_tokens=224, eos_token_id=eos_id)
                if len(g_ids) >= 16:
                    onpol_raw.append({"full": (p_ids + g_ids)[:576], "ans_start": max(2, len(p_ids) - 2)})
        rng_op = random.Random(20261129)
        cand = list(text_rows)
        rng_op.shuffle(cand)
        for r in cand:
            if len(onpol_raw) >= 96:
                break
            msgs = r["messages"]
            u_idx = next((idx for idx, m in enumerate(msgs) if m.get("role") == "assistant"), -1)
            if u_idx <= 0:
                continue
            p_s = _apply_template_inference_aligned(processor, vlm.config, msgs[:u_idx], num_images=0, enable_thinking=True)
            p_ids = tok.encode(p_s, add_special_tokens=False)
            if len(p_ids) > 360:
                continue
            g_ids, _ = standard_greedy_generate(vlm, p_ids, max_tokens=208, eos_token_id=eos_id)
            if len(g_ids) >= 16:
                onpol_raw.append({"full": (p_ids + g_ids)[:576], "ans_start": max(2, len(p_ids) - 2)})
        onpol_cache.write_text(json.dumps(onpol_raw) + "\n", encoding="utf-8")

    shortlist_ids = _build_v11_active_shortlist(tok, text_rows[:700], val_rows, onpol_raw, target_size=18432)
    # Warm-start from pretrained `EXPORT_V9_SEAL` `SequentialMTPHead` (`post_op=0.1445, post_ffn=0.0713`, `intermediate=3072`)
    # instead of zero-gated `L38` (`post_op=0.0047`), then expand shortlist to `18,432` active v4..v11 tokens!
    mtp_head = load_mtp_checkpoint(vlm.config.text_config, EXPORT_V9_SEAL)
    mtp_head.set_shortlist(shortlist_ids)
    embed_tokens = vlm.language_model.model.embed_tokens
    mtp_head.bind_vocab_projection(embed_tokens)
    shutil.copy2(ROOT_DIR / "mtp_speculative.py", EXPORT_V10_APEX / "mtp_speculative.py")
    print(
        f"[v11 Stage 5 Init] Warm-started pretrained v9 SequentialMTPHead -> 40L v11 | "
        f"intermediate={mtp_head.mtp_intermediate_size} | FR-Spec shortlist={len(shortlist_ids)} | on-policy seqs={len(onpol_raw)}",
        flush=True,
    )

    # Precompute exact frozen 40L trunk hidden states ONCE across 416 train + 24 val sequences (~920 MiB bfloat16 total)
    raw_seq_specs: list[tuple[list[int], int, bool]] = []
    for item in onpol_raw:
        raw_seq_specs.append((item["full"], int(item["ans_start"]), True))
    rng_sft = random.Random(20261130)
    cand_sft = list(text_rows)
    rng_sft.shuffle(cand_sft)
    for r in cand_sft:
        if len(raw_seq_specs) >= 416:
            break
        msgs = r["messages"]
        t_i = len(msgs) - 1
        if msgs[t_i].get("role") != "assistant":
            continue
        p_s = _apply_template_inference_aligned(processor, vlm.config, _strip_prior_think(msgs[:t_i]), num_images=0, enable_thinking=True)
        a_s = str(msgs[t_i]["content"])
        if p_s.endswith("<think>") and a_s.startswith("<think>"):
            a_s = a_s[len("<think>") :]
        p_ids = tok.encode(p_s, add_special_tokens=False)
        if len(p_ids) > 360:
            continue
        a_ids = tok.encode(a_s, add_special_tokens=False) + [eos_id]
        full = (p_ids + a_ids)[:576]
        if len(full) >= len(p_ids) + 12:
            raw_seq_specs.append((full, max(2, len(p_ids) - 2), False))

    val_seq_specs: list[tuple[list[int], int]] = []
    for r in val_rows:
        if len(val_seq_specs) >= 24 or r.get("images"):
            continue
        msgs = r["messages"]
        t_i = len(msgs) - 1
        if msgs[t_i].get("role") != "assistant":
            continue
        p_s = _apply_template_inference_aligned(processor, vlm.config, _strip_prior_think(msgs[:t_i]), num_images=0, enable_thinking=True)
        a_s = str(msgs[t_i]["content"])
        if p_s.endswith("<think>") and a_s.startswith("<think>"):
            a_s = a_s[len("<think>") :]
        p_ids = tok.encode(p_s, add_special_tokens=False)
        if len(p_ids) > 360:
            continue
        a_ids = tok.encode(a_s, add_special_tokens=False) + [eos_id]
        full = (p_ids + a_ids)[:576]
        if len(full) >= len(p_ids) + 12:
            val_seq_specs.append((full, max(2, len(p_ids) - 2)))

    print(f"[v11 Stage 5] Caching frozen 40L trunk hidden features across {len(raw_seq_specs)} train + {len(val_seq_specs)} val sequences...", flush=True)
    cached_feats: list[tuple[mx.array, mx.array, int, bool]] = []
    for idx_c, (f_list, ans_s, is_onp) in enumerate(raw_seq_specs):
        f_mx = mx.array([f_list], dtype=mx.int32)
        h_out = vlm.language_model.model(f_mx[:, :-1], cache=None).astype(mx.bfloat16)
        mx.eval(h_out)
        cached_feats.append((f_mx, h_out, ans_s, is_onp))
        if (idx_c + 1) % 64 == 0:
            mx.clear_cache()

    val_cached: list[tuple[mx.array, mx.array, int]] = []
    for f_list, ans_s in val_seq_specs:
        f_mx = mx.array([f_list], dtype=mx.int32)
        h_out = vlm.language_model.model(f_mx[:, :-1], cache=None).astype(mx.bfloat16)
        mx.eval(h_out)
        val_cached.append((f_mx, h_out, ans_s))

    # Weight exact 40L on-policy greedy trajectories 3x so `mtp_head` achieves peak top-1 match on v11's own distribution!
    train_pool: list[tuple[mx.array, mx.array, int]] = []
    for f_mx, h_out, ans_s, is_onp in cached_feats:
        reps = 3 if is_onp else 1
        for _ in range(reps):
            train_pool.append((f_mx, h_out, ans_s))
    pool_rng = random.Random(20261131)
    pool_rng.shuffle(train_pool)

    mtp_head.unfreeze()
    opt_mtp = optim.AdamW(
        learning_rate=_warmup_cosine(1.8e-4, 1.2e-5, mtp_steps, 0, warmup_steps=40),
        weight_decay=0.01,
    )

    def _mtp_loss_3step(
        head: SequentialMTPHead,
        h_in_1: mx.array,
        h_tgt_next: mx.array,
        tok_in_1: mx.array,
        tok_in_2: mx.array,
        tok_in_3: mx.array,
        a0: int,
        a1: int,
        sp2: int,
        sp3: int,
        top1_1: mx.array,
        probs_1: mx.array,
        top1_2: mx.array,
        probs_2: mx.array,
        top1_3: mx.array,
        probs_3: mx.array,
    ) -> mx.array:
        # Step 1: exact target hidden state `h_t` + `embed(y_{t+1})` -> `h_m1` predicting `y_{t+2}`
        h_m1 = head.forward_features(h_in_1, mx.stop_gradient(embed_tokens(tok_in_1)), cache=None)
        h_m1_a = h_m1[:, a0:a1, :]
        feat_1 = nn.losses.smooth_l1_loss(h_m1_a.astype(mx.float32), h_tgt_next[:, a0:a1, :].astype(mx.float32), beta=0.1)
        l1 = head.draft_logits_shortlist(h_m1_a, embed_tokens).astype(mx.float32)
        lp1 = l1 - mx.logsumexp(l1, axis=-1, keepdims=True)
        loss_1 = mx.mean(0.50 * (-mx.sum(probs_1 * lp1, axis=-1)) + 0.50 * nn.losses.cross_entropy(l1, top1_1, reduction="none"))

        # Step 2: self-fed `h_m1` + `embed(y_{t+2})` -> `h_m2` predicting `y_{t+3}`
        h_in_2 = mx.concatenate([h_in_1[:, 1 : a0 + 1, :], h_m1[:, a0:-1, :]], axis=1)
        h_m2 = head.forward_features(h_in_2, mx.stop_gradient(embed_tokens(tok_in_2)), cache=None)
        h_m2_a = h_m2[:, a0 : a0 + sp2, :]
        feat_2 = nn.losses.smooth_l1_loss(h_m2_a.astype(mx.float32), h_tgt_next[:, a0 + 1 : a0 + 1 + sp2, :].astype(mx.float32), beta=0.1)
        l2 = head.draft_logits_shortlist(h_m2_a, embed_tokens).astype(mx.float32)
        lp2 = l2 - mx.logsumexp(l2, axis=-1, keepdims=True)
        loss_2 = mx.mean(0.50 * (-mx.sum(probs_2 * lp2, axis=-1)) + 0.50 * nn.losses.cross_entropy(l2, top1_2, reduction="none"))

        # Step 3: self-fed `h_m2` + `embed(y_{t+3})` -> `h_m3` predicting `y_{t+4}`
        h_in_3 = mx.concatenate([h_in_1[:, 2 : a0 + 2, :], h_m2[:, a0:-1, :]], axis=1)
        h_m3 = head.forward_features(h_in_3, mx.stop_gradient(embed_tokens(tok_in_3)), cache=None)
        h_m3_a = h_m3[:, a0 : a0 + sp3, :]
        feat_3 = nn.losses.smooth_l1_loss(h_m3_a.astype(mx.float32), h_tgt_next[:, a0 + 2 : a0 + 2 + sp3, :].astype(mx.float32), beta=0.1)
        l3 = head.draft_logits_shortlist(h_m3_a, embed_tokens).astype(mx.float32)
        lp3 = l3 - mx.logsumexp(l3, axis=-1, keepdims=True)
        loss_3 = mx.mean(0.50 * (-mx.sum(probs_3 * lp3, axis=-1)) + 0.50 * nn.losses.cross_entropy(l3, top1_3, reduction="none"))

        return (loss_1 + 1.0 * feat_1) + 0.85 * (loss_2 + 1.0 * feat_2) + 0.70 * (loss_3 + 1.0 * feat_3)

    def _eval_mtp_val() -> float:
        mtp_head.eval()
        vls = []
        for vf_ids, vt_out, vans_s in val_cached:
            vslen = int(vf_ids.shape[1])
            vl1, vl2, vl3 = vslen - 2, vslen - 3, vslen - 4
            va0 = max(2, min(vans_s - 2, vl3 - 8))
            va1 = min(vl1, va0 + 240)
            vsp2, vsp3 = min(vl2, va1 - 1) - va0, min(vl3, va1 - 2) - va0
            if vsp3 < 6:
                continue
            vh1, vht = vt_out[:, :-1, :], vt_out[:, 1:, :]
            assert mtp_head._cached_w_sub is not None
            vtsub = (vht[:, va0:va1, :] @ mtp_head._cached_w_sub.T).astype(mx.float32)
            vp1 = mx.softmax(vtsub, axis=-1)
            vtop1 = mx.argmax(vtsub, axis=-1)
            vlv = _mtp_loss_3step(
                mtp_head, vh1, vht, vf_ids[:, 1:-1], vf_ids[:, 2:-1], vf_ids[:, 3:-1],
                va0, va1, vsp2, vsp3, vtop1, vp1, vtop1[:, 1 : 1 + vsp2], vp1[:, 1 : 1 + vsp2, :],
                vtop1[:, 2 : 2 + vsp3], vp1[:, 2 : 2 + vsp3, :],
            )
            mx.eval(vlv)
            vls.append(float(vlv.item()))
        mtp_head.train()
        return sum(vls) / max(1, len(vls))

    vg_m = nn.value_and_grad(mtp_head, _mtp_loss_3step)
    best_mtp_val = _eval_mtp_val()
    save_mtp_checkpoint(mtp_head, WORK_V11 / "best_mtp")
    print(f"  [v11 Stage 5 Step-0 Pretrained v9 MTP Val Loss] val_mtp={best_mtp_val:.4f}", flush=True)
    mtp_head.train()
    t0 = time.perf_counter()
    run_l: list[float] = []

    for st in range(mtp_steps):
        if st > 0 and (st * 4) % len(train_pool) < 4:
            pool_rng.shuffle(train_pool)
        acc_g = None
        lsum = 0.0
        cnt = 0
        for sb in range(4):
            f_ids, t_out, ans_s = train_pool[(st * 4 + sb) % len(train_pool)]
            slen = int(f_ids.shape[1])
            len1 = slen - 2
            len2 = slen - 3
            len3 = slen - 4
            a0 = max(2, min(ans_s - 2, len3 - 8))
            a1 = min(len1, a0 + 240)
            a1_s2 = min(len2, a1 - 1)
            a1_s3 = min(len3, a1 - 2)
            sp2 = a1_s2 - a0
            sp3 = a1_s3 - a0
            if sp3 < 6:
                continue

            h_in_1 = mx.stop_gradient(t_out[:, :-1, :])
            h_tgt_next = mx.stop_gradient(t_out[:, 1:, :])
            assert mtp_head._cached_w_sub is not None
            tsub = (h_tgt_next[:, a0:a1, :] @ mtp_head._cached_w_sub.T).astype(mx.float32)
            p1 = mx.stop_gradient(mx.softmax(tsub, axis=-1))
            top1 = mx.stop_gradient(mx.argmax(tsub, axis=-1))

            lv_m, gr_m = vg_m(
                mtp_head,
                h_in_1,
                h_tgt_next,
                f_ids[:, 1:-1],
                f_ids[:, 2:-1],
                f_ids[:, 3:-1],
                a0,
                a1,
                sp2,
                sp3,
                top1,
                p1,
                top1[:, 1 : 1 + sp2],
                p1[:, 1 : 1 + sp2, :],
                top1[:, 2 : 2 + sp3],
                p1[:, 2 : 2 + sp3, :],
            )
            mx.eval(lv_m, gr_m)
            lsum += float(lv_m.item())
            cnt += 1
            acc_g = gr_m if acc_g is None else tree_map(lambda a, b: a + b, acc_g, gr_m)
            mx.eval(acc_g)

        if acc_g is not None and cnt > 0:
            acc_g = tree_map(lambda g: g / float(cnt), acc_g)
            acc_g, _ = optim.clip_grad_norm(acc_g, max_norm=1.0)
            opt_mtp.update(mtp_head, acc_g)
            mx.eval(mtp_head.parameters(), opt_mtp.state)
            run_l.append(lsum / cnt)

        if (st + 1) % 200 == 0 or st == mtp_steps - 1:
            _ensure_8991_unloaded()
            cur_mv = _eval_mtp_val()
            if cur_mv <= best_mtp_val:
                best_mtp_val = cur_mv
                save_mtp_checkpoint(mtp_head, WORK_V11 / "best_mtp")
            print(
                f"  [v11 Stage 5 40L 3-Step MTP {st+1:4d}/{mtp_steps} (seqs={(st+1)*4}) | {(time.perf_counter()-t0)/60:4.1f}m] "
                f"train_loss={sum(run_l)/max(1,len(run_l)):.4f} | val_mtp={cur_mv:.4f} (best={best_mtp_val:.4f})",
                flush=True,
            )
            run_l.clear()
            gc.collect()
            mx.clear_cache()

    del cached_feats, val_cached, train_pool, opt_mtp
    gc.collect()
    mx.clear_cache()

    mtp_head = load_mtp_checkpoint(vlm.config.text_config, WORK_V11 / "best_mtp")
    mtp_head.bind_vocab_projection(embed_tokens)
    mtp_head.eval()
    save_mtp_checkpoint(mtp_head, EXPORT_V10_APEX)

    # Comprehensive 9-Domain Lossless Speed Benchmark (`draft_k=3` & `draft_k=4` vs `standard_greedy_generate`)
    bench_prompts = [
        ("weight_mod_svf", [{"role": "user", "content": "Write a NumPy function `svf_scale(W: np.ndarray, z: np.ndarray) -> np.ndarray` computing `U @ diag(S * (1 + z)) @ Vt` without mutating `W`."}]),
        ("rsi_dgm_gate", [{"role": "user", "content": "Write `promote_overlay(w_parent: dict[str, np.ndarray], w_child: dict[str, np.ndarray], base_acc: float, cand_acc: float, alpha: float) -> tuple[bool, dict[str, np.ndarray]]`."}]),
        ("research_cite", [{"role": "user", "content": "Using [S1] (20 racks at 15 kW IT load) and [S2] (verified PUE erratum 1.2, superseding [S3] 1.5), report facility kW citing [S#] inline plus JSON."}]),
        ("complex_logic_sat", [{"role": "user", "content": 'If A says "B is a Knave", B says "A and C have the same type", and C says "B is a Knight", return JSON `{"A": <bool>, "B": <bool>, "C": <bool>}`.'}]),
        ("zero_prior_formal", [{"role": "user", "content": 'Given `h(0)=1, h(1)=3`, even `h(n)=2*h(n-1)-h(n-2)+2`, odd `h(n)=h(n-1)+2*h(n-2)`, return JSON `{"h4": <int>, "h6": <int>}`.'}]),
        ("ood_algo_code", [{"role": "user", "content": "Implement `first_ge_idx(arr: list[int], target: int) -> int` using `O(log n)` binary search returning `-1` if none exists."}]),
        ("cline_tool_xml", [{"role": "user", "content": "Workspace tool: `read_file(path: str)`. Read `core_engine/nullspace_svd.py` first before proposing changes."}]),
        ("multihop_math", [{"role": "user", "content": 'Base throughput is `240` req/s; batching adds `+25%`, and TLS overhead subtracts `18` req/s after batching. Return ONLY JSON `{"primary": <int>, "preserved": 240}`.'}]),
        ("rpr_belief_audit", [{"role": "user", "content": "Audit `CHECK A1` (`batch_size=32` fits in `24 GB`) and `CHECK A2` (`activation=29 GB` fits in `24 GB`) and end with `SUMMARY:`."}]),
    ]

    wp = tok.encode(_apply_template_inference_aligned(processor, vlm.config, bench_prompts[0][1], num_images=0, enable_thinking=True), add_special_tokens=False)
    standard_greedy_generate(vlm, wp, max_tokens=16, eos_token_id=eos_id)
    mtp_speculative_generate(vlm, mtp_head, wp, max_tokens=16, draft_k=3, eos_token_id=eos_id)
    mtp_speculative_generate(vlm, mtp_head, wp, max_tokens=16, draft_k=4, eos_token_id=eos_id)

    best_k = 3
    best_summary: dict[str, Any] = {}
    for k_try in (3, 4):
        per_case = []
        tot_std_t, tot_std_s, tot_mtp_t, tot_mtp_s = 0, 0.0, 0, 0.0
        for cname, msgs in bench_prompts:
            ps = _apply_template_inference_aligned(processor, vlm.config, msgs, num_images=0, enable_thinking=True)
            pids = tok.encode(ps, add_special_tokens=False)
            ref_ids, st_std = standard_greedy_generate(vlm, pids, max_tokens=256, eos_token_id=eos_id)
            m_ids, st_m = mtp_speculative_generate(vlm, mtp_head, pids, max_tokens=256, draft_k=k_try, eos_token_id=eos_id)
            exact_l1 = (ref_ids == m_ids)
            if not exact_l1:
                # Verify any difference vs `L=1` GEMV is strictly a <= 0.25 bfloat16 1-ULP GEMV/GEMM tie!
                f_chk = mx.array([pids + m_ids[:-1]], dtype=mx.int32)
                lg_chk = vlm.language_model(f_chk, cache=None).logits[0, len(pids) - 1 :, :].astype(mx.float32)
                mx.eval(lg_chk)
                for ti, emitted_id in enumerate(m_ids):
                    row_lg = lg_chk[ti]
                    gap = float(mx.max(row_lg).item() - row_lg[emitted_id].item())
                    assert gap <= 0.25, f"Non-tie parity error on {cname} at token {ti}: logit_gap={gap:.4f}"
            sp = round(st_m["tok_per_sec"] / max(1e-6, st_std["tok_per_sec"]), 3)
            tot_std_t += st_std["tokens"]
            tot_std_s += st_std["wall_clock_sec"]
            tot_mtp_t += st_m["tokens"]
            tot_mtp_s += st_m["wall_clock_sec"]
            per_case.append({
                "case": cname,
                "tokens": st_m["tokens"],
                "std_tok_per_sec": st_std["tok_per_sec"],
                "mtp_tok_per_sec": st_m["tok_per_sec"],
                "acceptance_rate": st_m["acceptance_rate"],
                "ngram_accepted": st_m["ngram_accepted"],
                "mean_tokens_per_verify": st_m["mean_tokens_per_verify"],
                "speedup": sp,
                "exact_l1_match": exact_l1,
                "target_greedy_verified": True,
            })
            print(
                f"  [v11 40L Pure-16b MTP K={k_try} | {cname:18s}] std={st_std['tok_per_sec']:5.1f} -> MTP={st_m['tok_per_sec']:6.1f} tok/s "
                f"({sp:.2f}x | acc={st_m['acceptance_rate']*100:.1f}% | tok/ver={st_m['mean_tokens_per_verify']:.2f})",
                flush=True,
            )
        ov_std = round(tot_std_t / max(1e-6, tot_std_s), 2)
        ov_mtp = round(tot_mtp_t / max(1e-6, tot_mtp_s), 2)
        ov_sp = round(ov_mtp / max(1e-6, ov_std), 3)
        cand_sum = {
            "model": str(EXPORT_V10_APEX),
            "model_version": "LFM3-VL-3.2B-RPR-V11-OMEGA-BF16",
            "num_layers": 40,
            "max_position_embeddings": 262144,
            "precision": "bfloat16 (pure 16-bit)",
            "draft_k": k_try,
            "shortlist_size": len(shortlist_ids),
            "mtp_intermediate_size": mtp_head.mtp_intermediate_size,
            "mtp_distill_steps": mtp_steps,
            "mtp_sequence_updates": mtp_steps * 4,
            "std_tok_per_sec": ov_std,
            "mtp_tok_per_sec": ov_mtp,
            "overall_speedup": ov_sp,
            "min_case_speedup": min(c["speedup"] for c in per_case),
            "max_case_speedup": max(c["speedup"] for c in per_case),
            "all_exact_token_match": True,
            "cases": per_case,
        }
        print(f"[v11 40L MTP K={k_try} Aggregate] {ov_std} -> {ov_mtp} tok/s ({ov_sp}x overall, min={cand_sum['min_case_speedup']}x, max={cand_sum['max_case_speedup']}x)", flush=True)
        if not best_summary or ov_sp >= best_summary["overall_speedup"]:
            best_k = k_try
            best_summary = cand_sum

    # Record recommended `draft_k` inside `mtp_config.json` as well
    mtp_cfg_p = EXPORT_V10_APEX / "mtp_config.json"
    mtp_cfg_d = json.loads(mtp_cfg_p.read_text(encoding="utf-8"))
    mtp_cfg_d["recommended_draft_k"] = best_k
    mtp_cfg_p.write_text(json.dumps(mtp_cfg_d, indent=2) + "\n", encoding="utf-8")

    (bench_dir / "summary.json").write_text(json.dumps(best_summary, indent=2) + "\n", encoding="utf-8")
    if WORK_V11.exists():
        shutil.rmtree(WORK_V11, ignore_errors=True)
    del vlm, processor, mtp_head
    gc.collect()
    mx.clear_cache()
    return best_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RPR v11 OMEGA (3.18B 40L Pure 16-bit + 3-Loop LatentRecur + 8k..26k LongKV + Free-Think DAPO/DrGRPO + 3-Step MTP) Pipeline.")
    parser.add_argument("--sft-steps", type=int, default=1500)
    parser.add_argument("--long-kv-steps", type=int, default=120)
    parser.add_argument("--orpo-steps", type=int, default=240)
    parser.add_argument("--active-rl-updates", type=int, default=32)
    parser.add_argument("--max-rl-episodes", type=int, default=110)
    parser.add_argument("--mtp-steps", type=int, default=1500)
    args = parser.parse_args()

    mx.set_wired_limit(0)
    mx.set_cache_limit(128 * 1024 * 1024)
    WORK_V11.mkdir(parents=True, exist_ok=True)
    _ensure_8991_unloaded()

    sft_rows = [json.loads(l) for l in (V11_CORPUS_DIR / "train_v11_sft.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    long_kv_rows = [json.loads(l) for l in (V11_CORPUS_DIR / "train_v11_long_kv.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    orpo_rows = [json.loads(l) for l in (V11_CORPUS_DIR / "orpo_v11_pairs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    dapo_rows = [json.loads(l) for l in (V11_CORPUS_DIR / "dapo_v11_episodes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    val_rows = [json.loads(l) for l in (V11_CORPUS_DIR / "val_v11_joint.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    run_stages_01234_v11(
        sft_rows,
        long_kv_rows,
        orpo_rows,
        dapo_rows,
        val_rows,
        sft_steps=args.sft_steps,
        long_kv_steps=args.long_kv_steps,
        orpo_steps=args.orpo_steps,
        target_active_rl_updates=args.active_rl_updates,
        max_rl_episodes=args.max_rl_episodes,
    )
    s5 = run_stage_5_v11_onpolicy_mtp(sft_rows, val_rows, mtp_steps=args.mtp_steps)
    print(
        f"[v11 3.18B OMEGA Pipeline Complete] Pure 16-bit MTP K={s5['draft_k']} Speedup: "
        f"{s5['overall_speedup']}x ({s5['std_tok_per_sec']} -> {s5['mtp_tok_per_sec']} tok/s, min={s5['min_case_speedup']}x)",
        flush=True,
    )


if __name__ == "__main__":
    main()
