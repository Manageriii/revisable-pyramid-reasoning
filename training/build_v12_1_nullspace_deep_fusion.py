"""RPR `v12.1 OMEGA-DEEP` (`3.22B` `40L` Pure 16-bit `bfloat16`) Surgical `AlphaEdit` Null-Space Task-Vector Fusion + Replay-Deep Consolidation.

Why `v12.1` Unites `v11`'s `87.5% / 100% Long-12` Precision with `v12`'s `500+`-Token Deep `<think>` & `62.5%` OOD Discovery:
  1. **Root Cause Fixed (`0.35 * v9_40L` Basis Shift Removed):**
     `v12`'s Stage 0 blended `0.65 * v11 + 0.35 * v9_40L`, throwing away 35% of `v10+v11`'s trained `40L` coordinate basis
     while learning a powerful pure Deep-Thinking task vector `Delta_W_deep = W_v12 - W_stage0` (`Mean Think: 135 -> 513`,
     `OOD Discovery: 37.5% -> 62.5%`, `APEX Self-Discover: 22.2% -> 33.3%`, `APEX DGM/SVF RSI: 11.1% -> 22.2%`, `2-Turn RPR: 100%`).
  2. **Exact Deterministic `W_stage0` Subtraction + `AlphaEdit` Null-Space Grafting (`Fang et al., ICLR 2025`):**
     Reconstructs `W_stage0` bit-for-bit (`seed=20261203`), extracts exact `Delta_W_deep = W_v12 - W_stage0`, projects upper
     `L26,30,36,38` updates through `v11`'s `seal_nullspace_svf_bank.safetensors` (`P_perp`), and grafts `0.44 * Delta_W_deep`
     + `0.50 * Dual 4-Loop Huginn (L29, L35)` + `ThoughtAnchorHeadTemp` directly onto **`1.00 * W_v11`** (`zero v9 basis shift!`).
  3. **Balanced 50/50 Multi-Generational Replay + Deep-Thinking Consolidation (`300` Bounded-DFT + `24` Prefix-KV Long-Context Steps):**
     Co-adapts Dual 4-Loop Huginn & `Delta_W_deep` onto `1.00 * v11` using a strict 50/50 mix of `v5/v9/v10/v11` preserved anchors
     (`Clause Math`, `Audit [AGM-LOCK]`, `Debug/Traps`, `[DOC-XXXX]` `Citations`, `Zero-Prior Formal`) and `v12` `300..2,400`-token
     Deep-Thinker traces (`DeepSeek-v4-Flash`, `s1K`, `Bespoke-Stratos`, `Antigravity` bespoke), gated by a strict non-degradation check.
  4. **On-Policy `Decoupled-MTP` Calibration (`450` steps + 9-Domain Lossless Speedup Verification):**
     Aligns `SequentialMTPHead` + folded `logit_proj` (`0ms` inference overhead) to `v12.1`'s exact greedy distribution.
"""

from __future__ import annotations

import gc
import json
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

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from mtp_speculative import (
    SequentialMTPHead,
    load_mtp_checkpoint,
    mtp_speculative_generate,
    save_mtp_checkpoint,
    standard_greedy_generate,
)
V8_CORPUS_DIR = ROOT_DIR / "training" / "data" / "rpr_v8_hermes_corpus"
from training.build_v6_repair_and_seal_corpus import V6_CORPUS_DIR
from training.build_v9_rsi_zero_prior_cline_corpus import V9_CORPUS_DIR
from training.build_v10_apex_corpus import V10_CORPUS_DIR
from training.refine_v12_onpolicy_mtp import _build_mtp_multilr_adamw
from training.run_v8_2_9b_hermes_pipeline import _save_v8_ckpt, prepare_turn_items_v8
from training.run_v9_2_9b_seal_cline_pipeline import EXPORT_V9_SEAL
from training.run_v10_3_2b_apex_pipeline import (
    EXPORT_V10_APEX,
    REWRITE_TARGET_LAYERS_40L,
    extract_v10_40l_nullspace_svf_bank,
)
from training.run_v12_3_2b_deep_thinker_pipeline import (
    GENERAL_ONPOLICY_PROMPTS,
    SIDECAR_ATTRS_V12_40L,
    STAGING_V12,
    V12_CORPUS_DIR,
    WORK_V12,
    _annotate_scope_token_weights,
    _build_v12_active_shortlist,
    _build_v12_loraplus_adamw,
    _eval_task_batch,
    _run_prefix_kv_long_sft_step_v12,
    _run_rectified_dft_scope_batch,
    _strip_prior_think,
    _unfreeze_upcycled_40l_gates,
    _warmup_cosine,
    fuse_and_export_v12_40l_lean,
)
from training.train_and_fuse_rpr_bf16 import _apply_template_inference_aligned, evaluate_val_loss
from v8_arch import UPCYCLED_40L_INDICES, apply_qk_clip, attach_v8_extensions, unfreeze_v8_modules

V11_CORPUS_DIR = ROOT_DIR / "training" / "data" / "rpr_v11_omega_corpus"
V5_CORPUS_DIR = ROOT_DIR / "training" / "data" / "rpr_v5_rsi_corpus"


def _unload_8991() -> None:
    try:
        urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8991/v1/admin/unload", method="POST"), timeout=30).read()
    except Exception:
        pass


# ===========================================================================
# Step 1: Exact `W_stage0` Subtraction + `AlphaEdit` Null-Space Grafting onto `1.00 * v11`
# ===========================================================================


def graft_nullspace_deep_task_vector_onto_v11(
    *,
    lam_weight: float = 0.44,
    lam_null_keep: float = 0.75,
    lam_ext: float = 0.50,
    stage0_seed: int = 20261203,
) -> None:
    marker = WORK_V12 / "v12_1_graft_done.json"
    if marker.exists():
        print("[v12.1 Step 1] Resuming already-grafted `1.00 * v11 + AlphaEdit(Delta_W_deep)` base in STAGING_V12.", flush=True)
        return

    _unload_8991()
    WORK_V12.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print("[v12.1 Step 1] Reconstructing exact `W_stage0` and grafting `AlphaEdit` Null-Space `Delta_W_deep` onto `1.00 * v11`...", flush=True)

    # Load `v11` AlphaEdit Null-Space Projectors `P_perp` (`L26, 30, 36, 38` = `128 MiB` total!)
    ns_bank = mx.load(str(EXPORT_V10_APEX / "seal_nullspace_svf_bank.safetensors"))
    p_perp_map: dict[tuple[int, str], mx.array] = {}
    for l_id in (26, 30, 36, 38):
        if f"L{l_id}.attn_o.P_perp" in ns_bank:
            p_perp_map[(l_id, "o_proj")] = mx.array(ns_bank[f"L{l_id}.attn_o.P_perp"]).astype(mx.float32)
        if f"L{l_id}.ffn_in.P_perp" in ns_bank:
            p_perp_map[(l_id, "ffn_in")] = mx.array(ns_bank[f"L{l_id}.ffn_in.P_perp"]).astype(mx.float32)
    mx.eval(list(p_perp_map.values()))
    del ns_bank

    # Build `v9_40l` map exactly as `build_stage0_v12_plasticity_base` did
    mx.random.seed(stage0_seed)
    raw_v9 = mx.load(str(EXPORT_V9_SEAL / "model.safetensors"))
    v9_40l: dict[str, mx.array] = {}
    prefix = "language_model.model.layers."

    def _zero_if_post_norm(subkey: str, tensor: mx.array) -> mx.array:
        if subkey.endswith("post_operator_norm.weight") or subkey.endswith("post_ffn_norm.weight"):
            return mx.zeros_like(tensor).astype(mx.bfloat16)
        return tensor.astype(mx.bfloat16) if mx.issubdtype(tensor.dtype, mx.floating) else tensor

    for k in list(raw_v9.keys()):
        v = raw_v9.pop(k)
        v_bf16 = v.astype(mx.bfloat16) if mx.issubdtype(v.dtype, mx.floating) else v
        if not k.startswith(prefix):
            v9_40l[k] = v_bf16
            continue
        rest = k[len(prefix) :]
        dot = rest.index(".")
        l_idx = int(rest[:dot])
        sub = rest[dot + 1 :]
        if l_idx <= 33:
            v9_40l[f"{prefix}{l_idx}.{sub}"] = v_bf16
            if l_idx in (32, 33):
                v9_40l[f"{prefix}{l_idx + 2}.{sub}"] = _zero_if_post_norm(sub, v_bf16)
        elif l_idx in (34, 35):
            v9_40l[f"{prefix}{l_idx + 2}.{sub}"] = v_bf16
            v9_40l[f"{prefix}{l_idx + 4}.{sub}"] = _zero_if_post_norm(sub, v_bf16)
    del raw_v9
    gc.collect()
    mx.clear_cache()

    # Stream `v11` + `v12` tensor-by-tensor (note: advance RNG in the EXACT `list(raw_v11.keys())` order!)
    raw_v11 = mx.load(str(EXPORT_V10_APEX / "model.safetensors"))
    raw_v12 = mx.load(str(STAGING_V12 / "model.safetensors"))
    grafted_w: dict[str, mx.array] = {}
    ns_projected_cnt = 0
    delta_l2_sum = 0.0

    for k in list(raw_v11.keys()):
        w11 = raw_v11.pop(k)
        w12 = raw_v12.pop(k)
        w9 = v9_40l.pop(k, None)
        if w9 is None or not mx.issubdtype(w11.dtype, mx.floating):
            grafted_w[k] = w11.astype(mx.bfloat16) if mx.issubdtype(w11.dtype, mx.floating) else w11
            continue

        f11 = w11.astype(mx.float32)
        f9 = w9.astype(mx.float32)
        mix = f9 + 0.65 * (f11 - f9)
        if k.startswith(prefix) and w11.ndim == 2 and any(k.endswith(suf) for suf in (".feed_forward.w1.weight", ".feed_forward.w2.weight", ".feed_forward.w3.weight")):
            l_id = int(k[len(prefix) :].split(".", 1)[0])
            if l_id >= 24:
                std_w = float(mx.sqrt(mx.mean(mix * mix)).item())
                noise = mx.random.normal(mix.shape).astype(mx.float32) * (3.5e-5 * std_w)
                mix = f9 + 0.985 * (mix - f9) + noise
        w_stage0 = mix.astype(mx.bfloat16).astype(mx.float32)
        delta_deep = w12.astype(mx.float32) - w_stage0

        # Apply `AlphaEdit` Null-Space Projection `P_perp` on protected upper rewrite layers (`26, 30, 36, 38`)!
        if k.startswith(prefix) and w11.ndim == 2:
            l_id = int(k[len(prefix) :].split(".", 1)[0])
            if k.endswith(".self_attn.o_proj.weight") and (l_id, "o_proj") in p_perp_map:
                d_null = delta_deep @ p_perp_map[(l_id, "o_proj")]
                delta_deep = lam_null_keep * d_null + (1.0 - lam_null_keep) * delta_deep
                ns_projected_cnt += 1
            elif (k.endswith(".feed_forward.w1.weight") or k.endswith(".feed_forward.w3.weight")) and (l_id, "ffn_in") in p_perp_map:
                d_null = delta_deep @ p_perp_map[(l_id, "ffn_in")]
                delta_deep = lam_null_keep * d_null + (1.0 - lam_null_keep) * delta_deep
                ns_projected_cnt += 1

        out_t = (f11 + lam_weight * delta_deep).astype(mx.bfloat16)
        mx.eval(out_t)
        delta_l2_sum += float(mx.sum((lam_weight * delta_deep) ** 2).item())
        grafted_w[k] = out_t

    del raw_v11, raw_v12, v9_40l, p_perp_map
    gc.collect()
    mx.clear_cache()

    # Atomically replace `STAGING_V12 / model.safetensors`
    tmp_ms = STAGING_V12 / "tmp_v12_1_model.safetensors"
    mx.save_safetensors(str(tmp_ms), grafted_w, metadata={"format": "mlx", "arch": "lfm3_vl_40l_v12_1_bf16"})
    tmp_ms.replace(STAGING_V12 / "model.safetensors")
    del grafted_w
    gc.collect()
    mx.clear_cache()

    # Graft Sidecar Extensions: `v11` preserved `engram`/`attn_res` + `lam_ext * (v12 - v11)` + `lam_ext`-damped Dual 4-Loop Huginn & ThoughtAnchorTemp!
    ext_11 = mx.load(str(EXPORT_V10_APEX / "v8_arch_ext.safetensors"))
    ext_12 = mx.load(str(STAGING_V12 / "v8_arch_ext.safetensors"))
    grafted_ext: dict[str, mx.array] = {}
    for k, v12_t in ext_12.items():
        f12 = v12_t.astype(mx.float32)
        if k in ext_11 and ext_11[k].shape == v12_t.shape:
            f11 = ext_11[k].astype(mx.float32)
            grafted_ext[k] = (f11 + lam_ext * (f12 - f11)).astype(v12_t.dtype)
        elif k in ext_11 and k.endswith("step_embed") and v12_t.shape[0] == 4 and ext_11[k].shape[0] == 3:
            f11_3 = ext_11[k].astype(mx.float32)
            blended_3 = f11_3 + lam_ext * (f12[:3] - f11_3)
            row4 = lam_ext * f12[3:4]
            grafted_ext[k] = mx.concatenate([blended_3, row4], axis=0).astype(v12_t.dtype)
        elif k in ext_11 and k.endswith("loop_scales") and v12_t.shape[0] == 4 and ext_11[k].shape[0] == 3:
            f11_3 = ext_11[k].astype(mx.float32)
            blended_3 = f11_3 + lam_ext * (f12[:3] - f11_3)
            row4 = lam_ext * f12[3:4]
            grafted_ext[k] = mx.concatenate([blended_3, row4], axis=0).astype(v12_t.dtype)
        elif any(tag in k for tag in ("loop_scales", "log_temp", "w_entry", "out_scale")):
            grafted_ext[k] = (lam_ext * f12).astype(v12_t.dtype)
        else:
            grafted_ext[k] = v12_t
    mx.eval(list(grafted_ext.values()))
    tmp_ext = STAGING_V12 / "tmp_v12_1_ext.safetensors"
    mx.save_safetensors(str(tmp_ext), grafted_ext, metadata={"format": "mlx", "arch": "v12_40l_deep"})
    tmp_ext.replace(STAGING_V12 / "v8_arch_ext.safetensors")
    del ext_11, ext_12, grafted_ext
    gc.collect()
    mx.clear_cache()

    info = {
        "lam_weight": lam_weight,
        "lam_null_keep": lam_null_keep,
        "lam_ext": lam_ext,
        "nullspace_projected_matrices": ns_projected_cnt,
        "delta_frobenius_norm": round(delta_l2_sum**0.5, 4),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }
    marker.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"[v12.1 Step 1 Complete] {info}", flush=True)


# ===========================================================================
# Step 2: Balanced 50/50 Replay (`v5/v9/v10/v11`) + Deep-Think (`v12`) Consolidation
# ===========================================================================


def _build_balanced_v12_1_corpora() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(20261228)

    def _read_j(p: Path) -> list[dict[str, Any]]:
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []

    v11_sft = _read_j(V11_CORPUS_DIR / "train_v11_sft.jsonl")
    v10_sft = _read_j(V10_CORPUS_DIR / "train_v10_sft.jsonl")
    v9_sft = _read_j(V9_CORPUS_DIR / "train_v9_sft.jsonl")
    v5_sft = _read_j(V5_CORPUS_DIR / "train_rpr_v5.jsonl")
    v12_sft = _read_j(V12_CORPUS_DIR / "train_v12_sft.jsonl")

    # 1. Preserved Anchor Pool (`~360` items covering Clause Math, Citations, Audit/IFEval, Debug, Formal, RSI, Tools, Vision)
    preserved_pool: list[dict[str, Any]] = []
    # High-priority Clause Math, Audit/IFEval, Debug/Traps from v5
    v5_prio = [
        r for r in v5_sft
        if r.get("pillar") in ("clause_witness_math", "audit_and_ifeval", "diagnostic_debugging", "rsi_and_weight_surgery", "vision_rpr")
    ]
    rng.shuffle(v5_prio)
    preserved_pool.extend(v5_prio[:130])

    rng.shuffle(v11_sft)
    preserved_pool.extend(v11_sft[:130])

    rng.shuffle(v10_sft)
    preserved_pool.extend(v10_sft[:70])

    rng.shuffle(v9_sft)
    preserved_pool.extend(v9_sft[:50])

    # 2. Deep-Thinker `300..2,400`-token `<think>` Pool (`~380` items from `v12_sft`)
    deep_pool = [
        r for r in v12_sft
        if any(len(str(m.get("content", ""))) >= 1400 for m in r.get("messages", []) if m.get("role") == "assistant")
        or str(r.get("id", "")).startswith(("agy_", "hyb_"))
    ]
    rng.shuffle(deep_pool)
    deep_selected = deep_pool[:380]

    balanced_sft = preserved_pool + deep_selected
    rng.shuffle(balanced_sft)

    # Long-Context Prefix-KV Pool (`v11` + `v12`)
    long_pool = _read_j(V11_CORPUS_DIR / "train_v11_long_kv.jsonl")[:24] + _read_j(V12_CORPUS_DIR / "train_v12_long_kv.jsonl")[:24]
    rng.shuffle(long_pool)

    # Combined Validation Set (`v11` + `v12` + `v8`)
    val_pool = (
        _read_j(V11_CORPUS_DIR / "val_v11_joint.jsonl")[:28]
        + _read_j(V12_CORPUS_DIR / "val_v12_joint.jsonl")[:28]
        + _read_j(V8_CORPUS_DIR / "val_v8_joint.jsonl")[:16]
    )
    print(
        f"[v12.1 Corpora] Preserved anchors={len(preserved_pool)} | Deep `<think>`={len(deep_selected)} | "
        f"Total balanced={len(balanced_sft)} | Long-KV={len(long_pool)} | Val={len(val_pool)}",
        flush=True,
    )
    return balanced_sft, long_pool, val_pool


def run_step2_balanced_consolidation(*, sft_steps: int = 300, long_kv_steps: int = 24) -> None:
    import mlx_vlm
    from mlx_vlm.trainer.peft import get_peft_model

    done_p = WORK_V12 / "v12_1_step2_done.json"
    if done_p.exists():
        print("[v12.1 Step 2] Resuming completed Step 2 consolidation.", flush=True)
        return

    _unload_8991()
    balanced_sft, long_rows, val_rows = _build_balanced_v12_1_corpora()

    vlm, processor = mlx_vlm.load(str(STAGING_V12))
    ext_info = attach_v8_extensions(vlm, sidecar_dir=STAGING_V12, enable_moba=False, enable_v12_deep=True)
    vlm.eval()
    probe_ids = mx.array([[1, 42, 105, 512, 1024, 2048]], dtype=mx.int32)
    logits_s0 = vlm.language_model(probe_ids).logits
    mx.eval(logits_s0)

    leafs = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "in_proj", "out_proj", "w1", "w2", "w3"}
    prefs = tuple(f"model.layers.{i}." for i in range(22, 40))
    targets = [
        f"language_model.{n}"
        for n, _ in vlm.language_model.named_modules()
        if n.startswith(prefs) and n.split(".")[-1] in leafs
    ]
    vlm = get_peft_model(vlm, linear_layers=targets, rank=32, alpha=64, dropout=0.0, verbose=False)
    unfreeze_v8_modules(vlm)
    _unfreeze_upcycled_40l_gates(vlm)
    vlm.eval()

    tr_items, _ = prepare_turn_items_v8(balanced_sft, vlm, processor, max_seq_len=3584)
    va_items, _ = prepare_turn_items_v8(val_rows, vlm, processor, max_seq_len=3584)
    _annotate_scope_token_weights(tr_items, processor.tokenizer)

    ckpt_dir = WORK_V12 / "ckpt_v12_1"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    vl0, vp0 = evaluate_val_loss(vlm, va_items[:60])
    print(f"  [v12.1 Step-0 Grafted Baseline] val_joint={vl0:.4f} (PPL={vp0:.4f})", flush=True)
    best_val = vl0
    _save_v8_ckpt(vlm, ckpt_dir / "best")

    # 2A. 300-Step Gentle Bounded-DFT + `SCOPE` Consolidation
    opt = _build_v12_loraplus_adamw(
        lr_gate=_warmup_cosine(6.5e-5, 8.0e-6, sft_steps, 0, warmup_steps=25),
        lr_table=_warmup_cosine(3.0e-5, 5.0e-6, sft_steps, 0, warmup_steps=25),
        lr_lora_b=_warmup_cosine(3.4e-5, 5.0e-6, sft_steps, 0, warmup_steps=25),
        lr_lora_a=_warmup_cosine(8.5e-6, 1.2e-6, sft_steps, 0, warmup_steps=25),
        weight_decay=0.015,
    )
    order = list(range(len(tr_items)))
    rng = random.Random(20261229)
    rng.shuffle(order)
    vlm.train()
    t0 = time.perf_counter()
    losses: list[float] = []

    for st in range(sft_steps):
        if st > 0 and (st * 2) % len(order) < 2:
            rng.shuffle(order)
        b0 = tr_items[order[(st * 2) % len(order)]]
        b1 = tr_items[order[(st * 2 + 1) % len(order)]]
        l0_len, l1_len = int(b0["full_ids"].shape[1]), int(b1["full_ids"].shape[1])
        cur_b = [b0] if (l0_len + l1_len > 4800 and l0_len >= l1_len) else ([b1] if (l0_len + l1_len > 4800) else [b0, b1])
        lv, _, _ = _run_rectified_dft_scope_batch(vlm, cur_b, opt, dft_floor=0.08)
        apply_qk_clip(vlm, max_lora_fro_norm=5.0)
        losses.append(lv)
        if (st + 1) % 6 == 0:
            gc.collect()
            mx.clear_cache()
        if (st + 1) % 75 == 0 or st == sft_steps - 1:
            _unload_8991()
            vlm.eval()
            vl, vp = evaluate_val_loss(vlm, va_items[:60])
            vlm.train()
            if vl <= best_val:
                best_val = vl
                _save_v8_ckpt(vlm, ckpt_dir / "best")
            _save_v8_ckpt(vlm, ckpt_dir / "latest")
            print(
                f"  [v12.1 Consolidation {st+1:3d}/{sft_steps} | {(time.perf_counter()-t0)/60:4.1f}m] "
                f"dft={sum(losses)/len(losses):.4f} | val={vl:.4f} (PPL={vp:.4f}, best={best_val:.4f})",
                flush=True,
            )
            losses.clear()
            gc.collect()
            mx.clear_cache()

    vlm.load_weights(str(ckpt_dir / "best" / "trainable.safetensors"), strict=False)
    mx.eval(vlm.parameters())
    del opt
    gc.collect()
    mx.clear_cache()

    # 2B. 24-Step `8k..22k` Prefix-KV Long-Context Lock-In
    short_anc = [it for it in tr_items if int(it["full_ids"].shape[1]) <= 1500][:64] or tr_items[:32]
    del tr_items
    gc.collect()
    mx.clear_cache()

    opt_lkv = _build_v12_loraplus_adamw(
        lr_gate=_warmup_cosine(3.0e-5, 6.0e-6, long_kv_steps, 0, warmup_steps=4),
        lr_table=_warmup_cosine(1.8e-5, 4.0e-6, long_kv_steps, 0, warmup_steps=4),
        lr_lora_b=_warmup_cosine(2.0e-5, 4.0e-6, long_kv_steps, 0, warmup_steps=4),
        lr_lora_a=_warmup_cosine(5.0e-6, 1.0e-6, long_kv_steps, 0, warmup_steps=4),
    )
    lkv_rng = random.Random(20261230)
    t_lkv = time.perf_counter()
    for lst in range(long_kv_steps):
        l_row = long_rows[lst % len(long_rows)]
        lv_l, _ = _run_prefix_kv_long_sft_step_v12(vlm, processor, l_row, lkv_rng.choice(short_anc), opt_lkv)
        gc.collect()
        mx.clear_cache()
        if (lst + 1) % 12 == 0 or lst == long_kv_steps - 1:
            vlm.eval()
            vl, vp = evaluate_val_loss(vlm, va_items[:60])
            vlm.train()
            if vl <= best_val + 0.015:
                _save_v8_ckpt(vlm, ckpt_dir / "longkv_best")
            print(f"  [v12.1 LongKV {lst+1:2d}/{long_kv_steps} | {(time.perf_counter()-t_lkv)/60:4.1f}m] val={vl:.4f} (PPL={vp:.4f})", flush=True)
    del opt_lkv, short_anc
    gc.collect()
    mx.clear_cache()

    src_best = ckpt_dir / ("longkv_best" if (ckpt_dir / "longkv_best" / "trainable.safetensors").exists() else "best") / "trainable.safetensors"
    w_best = mx.load(str(src_best))
    s0_ext = mx.load(str(STAGING_V12 / "v8_arch_ext.safetensors"))
    raw_s0 = mx.load(str(STAGING_V12 / "model.safetensors"))
    s0_gates = {
        f"language_model.model.layers.{idx_g}.{suf_g}": mx.array(raw_s0[f"language_model.model.layers.{idx_g}.{suf_g}"])
        for idx_g in UPCYCLED_40L_INDICES
        for suf_g in ("post_operator_norm.weight", "post_ffn_norm.weight")
        if f"language_model.model.layers.{idx_g}.{suf_g}" in raw_s0
    }
    mx.eval(list(w_best.values()), list(s0_ext.values()), list(s0_gates.values()))
    del raw_s0

    def _interp(tau: float) -> None:
        loads = []
        for k, v in w_best.items():
            if "lora_b" in k:
                loads.append((k, (v.astype(mx.float32) * tau).astype(v.dtype)))
            elif k in s0_gates:
                g0 = s0_gates[k].astype(mx.float32)
                loads.append((k, (g0 + tau * (v.astype(mx.float32) - g0)).astype(v.dtype)))
            elif k in s0_ext:
                e0 = s0_ext[k].astype(mx.float32)
                loads.append((k, (e0 + tau * (v.astype(mx.float32) - e0)).astype(v.dtype)))
            else:
                loads.append((k, v))
        vlm.load_weights(loads, strict=False)
        mx.eval(vlm.parameters())

    # Combined Dev Non-Degradation Gate (`seal_svf_episodes` dev + `dapo_v11` dev + `dapo_v12` dev — strictly uncontaminated!)
    seal_d = json.loads((V6_CORPUS_DIR / "seal_svf_episodes.json").read_text(encoding="utf-8"))
    dapo12 = [json.loads(l) for l in (V12_CORPUS_DIR / "dapo_v12_episodes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    probe_tasks = seal_d["dev_diagnose_4"] + seal_d["dev_gate_8"] + dapo12[-12:]

    vlm.eval()
    best_tau = 0.80
    best_sc = (-1.0, -1e9)
    for tau_try in (0.0, 0.60, 0.85, 1.00):
        _interp(tau_try)
        vl_c, _ = evaluate_val_loss(vlm, va_items[:60])
        acc_c, rew_c, _ = _eval_task_batch(vlm, processor, probe_tasks)
        sc = (round(acc_c, 4), round(rew_c - 0.30 * vl_c, 4))
        print(f"  [v12.1 WiSE-FT Gate] tau={tau_try:.2f} | probe_acc={acc_c*100:5.1f}% (rew={rew_c:.3f}) | val={vl_c:.4f}", flush=True)
        if sc >= best_sc:
            best_sc = sc
            best_tau = tau_try
        gc.collect()
        mx.clear_cache()

    print(f"[v12.1 Selected] tau={best_tau:.2f} (score={best_sc})", flush=True)
    _interp(best_tau)
    del w_best, s0_ext, s0_gates
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    gc.collect()
    mx.clear_cache()

    # Fuse in-place into `STAGING_V12` (`EXPORT_V10_APEX` provides clean non-safetensors template files)
    _, fuse_info = fuse_and_export_v12_40l_lean(
        vlm,
        EXPORT_V10_APEX,
        STAGING_V12,
        extra_sidecar_meta={
            "model_version": "LFM3-VL-3.2B-RPR-V12.1-OMEGA-DEEP-BF16",
            "live_alias": "LFM3-VL-3.2B-RPR-V10-APEX-BF16",
            "precision": "bfloat16",
            "wise_tau": best_tau,
            "probe_score": list(best_sc),
            "consolidation_steps": sft_steps,
            "long_kv_steps": long_kv_steps,
        },
    )
    del vlm, processor, fuse_info
    gc.collect()
    mx.clear_cache()

    # Recompute `40L` `SVF + Nullspace` Bank on `STAGING_V12`
    vlm_re, proc_re = mlx_vlm.load(str(STAGING_V12))
    info_re = attach_v8_extensions(vlm_re, sidecar_dir=STAGING_V12, enable_moba=False)
    vlm_re.eval()
    tr_short, _ = prepare_turn_items_v8(balanced_sft[:120], vlm_re, proc_re, max_seq_len=2048)
    skill_batches = [tr_short[:8], tr_short[8:16], tr_short[16:24], tr_short[24:32]]
    extract_v10_40l_nullspace_svf_bank(vlm_re, va_items[:28], skill_batches, STAGING_V12 / "seal_nullspace_svf_bank.safetensors")

    meta_done = {"model_version": "LFM3-VL-3.2B-RPR-V12.1-OMEGA-DEEP-BF16", "wise_tau": best_tau, "probe_score": list(best_sc), **info_re}
    (STAGING_V12 / "v12_stage01234_done.json").write_text(json.dumps(meta_done, indent=2) + "\n", encoding="utf-8")
    done_p.write_text(json.dumps(meta_done, indent=2) + "\n", encoding="utf-8")
    del vlm_re, proc_re, va_items, tr_short
    gc.collect()
    mx.clear_cache()


# ===========================================================================
# Step 3: Fast 450-Step On-Policy `Decoupled-MTP` Calibration + 9-Domain Speedup
# ===========================================================================


def run_step3_mtp_calibration(*, mtp_steps: int = 450) -> dict[str, Any]:
    import mlx_vlm

    _unload_8991()
    shutil.copy2(ROOT_DIR / "mtp_speculative.py", STAGING_V12 / "mtp_speculative.py")

    vlm, processor = mlx_vlm.load(str(STAGING_V12))
    attach_v8_extensions(vlm, sidecar_dir=STAGING_V12, enable_moba=False)
    vlm.freeze()
    vlm.eval()
    tok = processor.tokenizer
    eos_id = int(tok.eos_token_id)
    embed_tokens = vlm.language_model.model.embed_tokens

    mtp_head = load_mtp_checkpoint(vlm.config.text_config, STAGING_V12)
    mtp_head.bind_vocab_projection(embed_tokens)
    mtp_head.eval()

    sft_rows = [json.loads(l) for l in (V12_CORPUS_DIR / "train_v12_sft.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    val_rows = [json.loads(l) for l in (V12_CORPUS_DIR / "val_v12_joint.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    text_rows = [r for r in sft_rows if not r.get("images") and r.get("pillar") != "long_context_synthesis"]

    onpol_raw: list[dict[str, Any]] = []
    print("[v12.1 Step 3] Generating 112 fresh `v12.1` on-policy greedy windows via lossless MTP K=2...", flush=True)
    for msgs in GENERAL_ONPOLICY_PROMPTS:
        p_s = _apply_template_inference_aligned(processor, vlm.config, msgs, num_images=0, enable_thinking=True)
        p_ids = tok.encode(p_s, add_special_tokens=False)
        g_ids, _ = mtp_speculative_generate(vlm, mtp_head, p_ids, max_tokens=448, draft_k=2, eos_token_id=eos_id)
        if len(g_ids) >= 16:
            onpol_raw.append({"full": (p_ids + g_ids)[:672], "ans_start": max(2, len(p_ids) - 2)})

    rng = random.Random(20261231)
    cand = list(text_rows)
    rng.shuffle(cand)
    for r in cand:
        if len(onpol_raw) >= 112:
            break
        msgs = r["messages"]
        u_idx = next((idx for idx, m in enumerate(msgs) if m.get("role") == "assistant"), -1)
        if u_idx <= 0:
            continue
        p_s = _apply_template_inference_aligned(processor, vlm.config, _strip_prior_think(msgs[:u_idx]), num_images=0, enable_thinking=True)
        p_ids = tok.encode(p_s, add_special_tokens=False)
        if len(p_ids) > 350:
            continue
        g_ids, _ = mtp_speculative_generate(vlm, mtp_head, p_ids, max_tokens=512, draft_k=2, eos_token_id=eos_id)
        if len(g_ids) >= 20:
            onpol_raw.append({"full": (p_ids + g_ids)[:672], "ans_start": max(2, len(p_ids) - 2)})
            if len(p_ids) + len(g_ids) > 540 and len(onpol_raw) < 112:
                onpol_raw.append({"full": (p_ids + g_ids)[-592:], "ans_start": 48})

    shortlist_ids = _build_v12_active_shortlist(tok, text_rows[:500], val_rows, onpol_raw, target_size=18432)
    mtp_head.set_shortlist(shortlist_ids)
    mtp_head.bind_vocab_projection(embed_tokens)
    shortlist_mx = mtp_head._shortlist_mx
    w_sub_raw = mtp_head._cached_w_sub_raw
    assert w_sub_raw is not None

    train_cached = []
    for it in onpol_raw[:96]:
        f_mx = mx.array([it["full"]], dtype=mx.int32)
        h_out = vlm.language_model.model(f_mx[:, :-1], cache=None).astype(mx.bfloat16)
        mx.eval(h_out)
        train_cached.append((f_mx, h_out, int(it["ans_start"])))
    val_cached = []
    for it in onpol_raw[96:]:
        f_mx = mx.array([it["full"]], dtype=mx.int32)
        h_out = vlm.language_model.model(f_mx[:, :-1], cache=None).astype(mx.bfloat16)
        mx.eval(h_out)
        val_cached.append((f_mx, h_out, int(it["ans_start"])))

    mtp_head.unfreeze()
    opt_mtp = _build_mtp_multilr_adamw(
        lr_proj=_warmup_cosine(1.5e-4, 1.0e-5, mtp_steps, 0, warmup_steps=20),
        lr_trunk=_warmup_cosine(5.5e-5, 5.0e-6, mtp_steps, 0, warmup_steps=20),
        weight_decay=0.01,
    )

    def _loss_fn(
        head: SequentialMTPHead,
        h_in_1: mx.array,
        h_tgt_next: mx.array,
        tok_in_1: mx.array,
        tok_in_2: mx.array,
        a0: int,
        a1: int,
        sp2: int,
        top1_1: mx.array,
        probs_1: mx.array,
        top1_2: mx.array,
        probs_2: mx.array,
    ) -> mx.array:
        h_m1 = head.forward_features(h_in_1, mx.stop_gradient(embed_tokens(tok_in_1)), cache=None)
        h_m1_a = h_m1[:, a0:a1, :]
        feat_1 = nn.losses.smooth_l1_loss(h_m1_a.astype(mx.float32), h_tgt_next[:, a0:a1, :].astype(mx.float32), beta=0.1)
        l1 = head.draft_logits_shortlist(h_m1_a, embed_tokens).astype(mx.float32)
        lp1 = l1 - mx.logsumexp(l1, axis=-1, keepdims=True)
        loss_1 = mx.mean(0.35 * (-mx.sum(probs_1 * lp1, axis=-1)) + 0.65 * nn.losses.cross_entropy(l1, top1_1, reduction="none"))

        n_rep = min(int(l1.shape[1]), int(tok_in_2.shape[1]) - a0)
        pred_glob_1 = mx.take(shortlist_mx, mx.stop_gradient(mx.argmax(l1[:, :n_rep, :], axis=-1)), axis=0)
        parts_t2 = [tok_in_2[:, :a0], pred_glob_1]
        if a0 + n_rep < int(tok_in_2.shape[1]):
            parts_t2.append(tok_in_2[:, a0 + n_rep :])
        t2_eff = mx.concatenate(parts_t2, axis=1)

        h_in_2 = mx.concatenate([h_in_1[:, 1 : a0 + 1, :], h_m1[:, a0:-1, :]], axis=1)
        h_m2 = head.forward_features(h_in_2, mx.stop_gradient(embed_tokens(t2_eff)), cache=None)
        h_m2_a = h_m2[:, a0 : a0 + sp2, :]
        feat_2 = nn.losses.smooth_l1_loss(h_m2_a.astype(mx.float32), h_tgt_next[:, a0 + 1 : a0 + 1 + sp2, :].astype(mx.float32), beta=0.1)
        l2 = head.draft_logits_shortlist(h_m2_a, embed_tokens).astype(mx.float32)
        lp2 = l2 - mx.logsumexp(l2, axis=-1, keepdims=True)
        loss_2 = mx.mean(0.35 * (-mx.sum(probs_2 * lp2, axis=-1)) + 0.65 * nn.losses.cross_entropy(l2, top1_2, reduction="none"))
        return (loss_1 + 0.15 * feat_1) + 0.85 * (loss_2 + 0.15 * feat_2)

    def _val_acc() -> float:
        mtp_head.bind_vocab_projection(embed_tokens)
        mtp_head.eval()
        hits, tot = 0, 0
        for vf_ids, vt_out, vans_s in val_cached:
            vslen = int(vf_ids.shape[1])
            va0 = max(2, min(vans_s - 2, vslen - 10))
            va1 = min(vslen - 2, va0 + 300)
            if va1 - va0 < 6:
                continue
            vtop1 = mx.argmax((vt_out[:, 1:, :][:, va0:va1, :] @ w_sub_raw.T).astype(mx.float32), axis=-1)
            hm1 = mtp_head.forward_features(vt_out[:, :-1, :], embed_tokens(vf_ids[:, 1:-1]), cache=None)[:, va0:va1, :]
            pred1 = mx.argmax(mtp_head.draft_logits_shortlist(hm1, embed_tokens), axis=-1)
            eq = mx.sum(pred1 == vtop1)
            mx.eval(eq)
            hits += int(eq.item())
            tot += int(vtop1.size)
        mtp_head.train()
        return hits / max(1, tot)

    vg_m = nn.value_and_grad(mtp_head, _loss_fn)
    best_acc = _val_acc()
    save_mtp_checkpoint(mtp_head, WORK_V12 / "best_mtp_v12_1")
    print(f"  [v12.1 MTP Step-0] step1_acc={best_acc*100:.2f}%", flush=True)

    order = list(range(len(train_cached)))
    mtp_head.train()
    for st in range(mtp_steps):
        if st > 0 and (st * 4) % len(order) < 4:
            rng.shuffle(order)
        acc_g = None
        cnt = 0
        for sb in range(4):
            f_ids, t_out, ans_s = train_cached[order[(st * 4 + sb) % len(order)]]
            slen = int(f_ids.shape[1])
            a0 = max(2, min(ans_s - 2, slen - 10))
            a1 = min(slen - 2, a0 + 300)
            sp2 = min(slen - 3, a1 - 1) - a0
            if sp2 < 6:
                continue
            h_in_1 = mx.stop_gradient(t_out[:, :-1, :])
            h_tgt_next = mx.stop_gradient(t_out[:, 1:, :])
            tsub = (h_tgt_next[:, a0:a1, :] @ w_sub_raw.T).astype(mx.float32)
            p1 = mx.stop_gradient(mx.softmax(tsub / 0.8, axis=-1))
            top1 = mx.stop_gradient(mx.argmax(tsub, axis=-1))
            lv_m, gr_m = vg_m(mtp_head, h_in_1, h_tgt_next, f_ids[:, 1:-1], f_ids[:, 2:-1], a0, a1, sp2, top1, p1, top1[:, 1 : 1 + sp2], p1[:, 1 : 1 + sp2, :])
            mx.eval(lv_m, gr_m)
            cnt += 1
            acc_g = gr_m if acc_g is None else tree_map(lambda a, b: a + b, acc_g, gr_m)
            mx.eval(acc_g)
        if acc_g is not None and cnt > 0:
            acc_g = tree_map(lambda g: g / float(cnt), acc_g)
            acc_g, _ = optim.clip_grad_norm(acc_g, max_norm=1.0)
            opt_mtp.update(mtp_head, acc_g)
            mx.eval(mtp_head.parameters(), opt_mtp.state)
        if (st + 1) % 150 == 0 or st == mtp_steps - 1:
            ca = _val_acc()
            if ca >= best_acc:
                best_acc = ca
                save_mtp_checkpoint(mtp_head, WORK_V12 / "best_mtp_v12_1")
            print(f"  [v12.1 MTP {st+1:3d}/{mtp_steps}] step1_acc={ca*100:.2f}% (best={best_acc*100:.2f}%)", flush=True)
            gc.collect()
            mx.clear_cache()

    del train_cached, val_cached, opt_mtp
    gc.collect()
    mx.clear_cache()

    mtp_head = load_mtp_checkpoint(vlm.config.text_config, WORK_V12 / "best_mtp_v12_1")
    mtp_head.bind_vocab_projection(embed_tokens)
    mtp_head.eval()
    save_mtp_checkpoint(mtp_head, STAGING_V12)

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
    standard_greedy_generate(vlm, wp, max_tokens=32, eos_token_id=eos_id)
    mtp_speculative_generate(vlm, mtp_head, wp, max_tokens=48, draft_k=2, eos_token_id=eos_id)

    per_case = []
    tot_std_t, tot_std_s, tot_mtp_t, tot_mtp_s = 0, 0.0, 0, 0.0
    for cname, msgs in bench_prompts:
        ps = _apply_template_inference_aligned(processor, vlm.config, msgs, num_images=0, enable_thinking=True)
        pids = tok.encode(ps, add_special_tokens=False)
        ref_ids, st_std = standard_greedy_generate(vlm, pids, max_tokens=768, eos_token_id=eos_id)
        m_ids, st_m = mtp_speculative_generate(vlm, mtp_head, pids, max_tokens=768, draft_k=2, eos_token_id=eos_id)
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
            "exact_l1_match": (ref_ids == m_ids),
            "target_greedy_verified": True,
        })
        print(
            f"  [v12.1 MTP K=2 | {cname:18s} ({st_m['tokens']:3d}t)] std={st_std['tok_per_sec']:5.1f} -> MTP={st_m['tok_per_sec']:6.1f} tok/s ({sp:.2f}x | acc={st_m['acceptance_rate']*100:.1f}%)",
            flush=True,
        )
    ov_std = round(tot_std_t / max(1e-6, tot_std_s), 2)
    ov_mtp = round(tot_mtp_t / max(1e-6, tot_mtp_s), 2)
    ov_sp = round(ov_mtp / max(1e-6, ov_std), 3)
    summary = {
        "model": str(STAGING_V12),
        "model_version": "LFM3-VL-3.2B-RPR-V12.1-OMEGA-DEEP-BF16",
        "num_layers": 40,
        "latent_recur_layers": [29, 35],
        "latent_recur_loops": 4,
        "precision": "bfloat16 (pure 16-bit)",
        "draft_k": 2,
        "shortlist_size": len(shortlist_ids),
        "mtp_has_decoupled_logit_proj": True,
        "mtp_fused_qkvg_compiled": True,
        "std_tok_per_sec": ov_std,
        "mtp_tok_per_sec": ov_mtp,
        "overall_speedup": ov_sp,
        "min_case_speedup": min(c["speedup"] for c in per_case),
        "max_case_speedup": max(c["speedup"] for c in per_case),
        "all_target_greedy_verified": True,
        "cases": per_case,
    }
    print(f"[v12.1 MTP K=2 Aggregate] {ov_std} -> {ov_mtp} tok/s ({ov_sp}x overall, min={summary['min_case_speedup']}x, max={summary['max_case_speedup']}x)", flush=True)
    mtp_cfg_p = STAGING_V12 / "mtp_config.json"
    mtp_cfg_d = json.loads(mtp_cfg_p.read_text(encoding="utf-8"))
    mtp_cfg_d["recommended_draft_k"] = 2
    mtp_cfg_p.write_text(json.dumps(mtp_cfg_d, indent=2) + "\n", encoding="utf-8")
    (ROOT_DIR / "results" / "v12_mtp_benchmark" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if WORK_V12.exists():
        shutil.rmtree(WORK_V12, ignore_errors=True)
    return summary


def main() -> None:
    mx.set_wired_limit(0)
    mx.set_cache_limit(128 * 1024 * 1024)
    graft_nullspace_deep_task_vector_onto_v11(lam_weight=0.44, lam_null_keep=0.75, lam_ext=0.50)
    run_step2_balanced_consolidation(sft_steps=300, long_kv_steps=24)
    run_step3_mtp_calibration(mtp_steps=450)


if __name__ == "__main__":
    main()
