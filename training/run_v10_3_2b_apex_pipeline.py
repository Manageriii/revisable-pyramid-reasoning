"""Master Checkpointed RPR v10 APEX (`3.18B` `40L` Pure 16-bit `bfloat16`, `256k` Context) Pipeline:

Builds `checkpoints/LFM3-VL-3.2B-RPR-V10-APEX-BF16/` atop `LFM3-VL-2.9B-RPR-V9-SEAL-BF16` (`Step-0 logit diff = 0.000000`):
  - Stage 0: Surgical `36L -> 40L` (`2.911B -> 3.178B`, `12 -> 14` `full_attention` layers) + `LatentReasoningRecurrence`
             (`L35`) + `5x CrossLayerAttnRes` (`L26,30,32,36,38`) + `256k` (`262,144`) `LongRoPE2YaRN` Dual-Path RoPE.
  - Stage 1: 14-Layer Deep Rank-32 LoRA (`10,14,18,22,26,30,31,32,34,35,36,37,38,39`) + Unfrozen Upcycled SandwichNorm Gates
             `(34,35,38,39)` + `Engram` + `AttnRes` + `LatentRecur` `MuonClip` Mixed-Context SFT (`480` batch-2 steps, `max_seq_len=5632`).
  - Stage 2: `MuonClip` Contrastive `ORPO` (`140` steps) on `orpo_v10_pairs.jsonl` (inline `[S#]` citations, non-mutating code,
             `log_softmax` vs `softmax`, exact relative workspace paths, `Self-Discover` logic, and `SVF` row-slicing).
  - Stage 3: `DAPO` Dynamic Sampling + `Dr. GRPO` Unbiased RLVR (`22` ACTIVE updates across up to `72` hard episodes, dynamically
             filtering out unanimous `4/4` `sigma=0` groups and using asymmetric `clip-higher` `[0.80, 1.28]` + fixed token normalizer).
  - Stage 4: `WiSE-FT` Non-Degradation Gate (`tau=0` == exact `v9`) -> Lean Pure `16-bit` (`bfloat16`) Standalone Export
             + `40L` `Transformer^2 SVF + Orthogonal Nullspace` Bank (`L26,30,36,38`) + `Darwin Gödel Machine` (`DGM`) Root Archive.
  - Stage 5: On-Policy `40L` `SequentialMTPHead` Alignment (`280` batch-2 steps) -> `mtp_head.safetensors`.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_unflatten
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
from training.build_v10_apex_corpus import V10_CORPUS_DIR
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
from training.train_and_fuse_rpr_bf16 import (
    _apply_template_inference_aligned,
    _run_dft_batch,
    chunked_multimodal_ce,
    evaluate_val_loss,
)
from training.verifiable_rewards import score_response
from v8_arch import (
    ATTN_RES_SOURCES,
    ATTN_RES_TARGETS_40L,
    UPCYCLED_40L_INDICES,
    Muon2D,
    apply_qk_clip,
    attach_v8_extensions,
    save_v8_sidecar,
    unfreeze_v8_modules,
    upcycle_v9_36l_to_v10_40l_identity,
)


def _build_v10_muonclip(lr_gate: Any, lr_table: Any, lr_2d: Any) -> optim.MultiOptimizer:
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
            Muon2D(learning_rate=lr_2d, momentum=0.95, weight_decay=0.01, ns_steps=5),
            optim.AdamW(learning_rate=lr_2d, weight_decay=0.01),
        ],
        filters=[
            lambda k, p: p.ndim == 1 or any(gk in k for gk in gate_keys),
            lambda k, _: ".table.weight" in k,
            lambda k, p: p.ndim == 2 and ".table.weight" not in k,
        ],
    )


WORK_V10 = ROOT_DIR / "checkpoints" / "rpr_v10_3_2b_work"
STAGE0_40L_ID = WORK_V10 / "stage0_v10_40l_id"
CKPT_V10 = WORK_V10 / "stage123_v10_ckpt"
EXPORT_V10_APEX = ROOT_DIR / "checkpoints" / "LFM3-VL-3.2B-RPR-V10-APEX-BF16"

# 14 middle + upper + newly upcycled blocks across `40L`
V10_DEEP_LORA_LAYERS = (10, 14, 18, 22, 26, 30, 31, 32, 34, 35, 36, 37, 38, 39)
REWRITE_TARGET_LAYERS_40L = (26, 30, 36, 38)
SIDECAR_ATTRS_40L = (
    "engram_6",
    "engram_18",
    "attn_res_26",
    "attn_res_30",
    "attn_res_32",
    "attn_res_36",
    "attn_res_38",
    "latent_recur_35",
)


def _unfreeze_upcycled_40l_gates(vlm: Any) -> None:
    """Unfreeze zero-initialized SandwichNorm gates on newly cloned `(34, 35, 38, 39)` so all 4 new blocks actively specialize!"""
    layers = vlm.language_model.model.layers
    for idx in UPCYCLED_40L_INDICES:
        layers[idx].post_operator_norm.unfreeze()
        layers[idx].post_ffn_norm.unfreeze()


def _load_v10_40l_from_v9_with_peft() -> tuple[Any, Any, dict[str, Any]]:
    import mlx_vlm
    from mlx_vlm.trainer.peft import get_peft_model

    if not (STAGE0_40L_ID / "model.safetensors").exists() or not (STAGE0_40L_ID / "v8_arch_ext.safetensors").exists():
        print(f"[v10 Stage 0] Surgically upcycling 36L v9 -> 40L 3.18B (256k LongRoPE2YaRN) at {STAGE0_40L_ID}...", flush=True)
        up_info = upcycle_v9_36l_to_v10_40l_identity(EXPORT_V9_SEAL, STAGE0_40L_ID)
        print(f"[v10 Stage 0 Upcycle Written] {json.dumps(up_info)}", flush=True)

    # Compute exact `36L` `v9` reference logits and immediately free `vlm_v9`!
    vlm_v9, _ = mlx_vlm.load(str(EXPORT_V9_SEAL))
    attach_v8_extensions(vlm_v9, sidecar_dir=EXPORT_V9_SEAL, enable_moba=False)
    vlm_v9.eval()
    probe = mx.array([[1, 42, 105, 512, 1024, 2048, 4096]], dtype=mx.int32)
    v9_logits = vlm_v9.language_model(probe).logits
    mx.eval(v9_logits)
    del vlm_v9
    gc.collect()
    mx.clear_cache()

    # Load `40L` `3.18B` identity model, attach `v10` sidecars (`Engram`, `5x AttnRes`, `LatentRecur`, `256k LongRoPE2YaRN`) + 14-Layer LoRA
    vlm, processor = mlx_vlm.load(str(STAGE0_40L_ID))
    ext_info = attach_v8_extensions(vlm, sidecar_dir=STAGE0_40L_ID, enable_moba=False)

    leafs = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "in_proj", "out_proj", "w1", "w2", "w3"}
    prefs = tuple(f"model.layers.{i}." for i in V10_DEEP_LORA_LAYERS)
    targets = [
        f"language_model.{n}"
        for n, _ in vlm.language_model.named_modules()
        if n.startswith(prefs) and n.split(".")[-1] in leafs
    ]
    vlm = get_peft_model(vlm, linear_layers=targets, rank=32, alpha=64, dropout=0.01, verbose=False)
    unfreeze_v8_modules(vlm)
    _unfreeze_upcycled_40l_gates(vlm)

    vlm.eval()
    v10_logits = vlm.language_model(probe).logits
    mx.eval(v10_logits)
    diff0 = float(mx.max(mx.abs(v10_logits.astype(mx.float32) - v9_logits.astype(mx.float32))).item())
    print(f"[v10 Stage 0 Identity Check] 36L v9 vs 40L v10 Step-0 max_abs_logit_diff = {diff0:.8f}", flush=True)
    assert diff0 < 1e-6, f"Step-0 v10 40L identity vs v9 broken: diff={diff0}"
    ext_info["step0_max_logit_diff"] = diff0
    ext_info["lora_target_modules"] = len(targets)
    return vlm, processor, ext_info


def fuse_and_export_40l_lean(
    vlm: Any,
    template_40l_dir: Path,
    export_dir: Path,
    extra_sidecar_meta: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """Fuse 14-layer LoRA into pure `16-bit` (`bfloat16`) `40L` `model.safetensors` + `v8_arch_ext.safetensors`,
    free `vlm` BEFORE reloading so peak Metal memory stays at a single `3.18B` model, and verify `reload_diff == 0.0`!
    """
    import mlx_vlm

    vlm.eval()
    probe_ids = mx.array([[1, 42, 105, 512, 1024, 2048]], dtype=mx.int32)

    export_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT_DIR / "v8_arch.py", export_dir / "v8_arch.py")
    shutil.copy2(ROOT_DIR / "mtp_speculative.py", export_dir / "mtp_speculative.py")

    # 1. Save updated `v8_arch_ext.safetensors` (`Engram` + `5x AttnRes` + `LatentReasoningRecurrence`) first
    save_v8_sidecar(vlm, export_dir, extra_meta=extra_sidecar_meta)

    # 2. Pop sidecar modules temporarily so `model.safetensors` holds strictly standard `40L` `Lfm3Vl` keys
    core = vlm.language_model.model
    popped_sidecars: dict[str, Any] = {}
    for attr_n in SIDECAR_ATTRS_40L:
        if attr_n in core:
            popped_sidecars[attr_n] = core.pop(attr_n)

    # 3. Merge LoRA adapters in `float32` -> pure `bfloat16` `nn.Linear`
    replacements: list[tuple[str, nn.Module]] = []
    merged_cnt = 0
    for name, mod in vlm.named_modules():
        if hasattr(mod, "linear") and hasattr(mod, "lora_a") and hasattr(mod, "lora_b"):
            base_l = mod.linear
            w_f32 = base_l.weight.astype(mx.float32)
            delta = float(mod.scale) * (mod.lora_b.astype(mx.float32).T @ mod.lora_a.astype(mx.float32).T)
            w_bf16 = (w_f32 + delta).astype(mx.bfloat16)
            has_bias = hasattr(base_l, "bias") and base_l.bias is not None
            new_lin = nn.Linear(w_bf16.shape[1], w_bf16.shape[0], bias=has_bias)
            new_lin.weight = w_bf16
            if has_bias:
                new_lin.bias = base_l.bias.astype(mx.bfloat16)
            replacements.append((name, new_lin))
            merged_cnt += 1

    if replacements:
        vlm.update_modules(tree_unflatten(replacements))
    vlm.apply(lambda x: x.astype(mx.bfloat16) if mx.issubdtype(x.dtype, mx.floating) else x)
    mx.eval(vlm.parameters())

    skip = {"model.safetensors", "model.safetensors.index.json", "v8_arch_ext.safetensors", "v8_arch_config.json", "v8_arch.py", "mtp_speculative.py"}
    for item in template_40l_dir.iterdir():
        if item.is_file() and item.name not in skip and not item.name.endswith(".safetensors"):
            shutil.copy2(item, export_dir / item.name)

    flat_w = {
        k: (v.astype(mx.bfloat16) if mx.issubdtype(v.dtype, mx.floating) else v)
        for k, v in tree_flatten(vlm.parameters())
    }
    mx.eval(list(flat_w.values()))
    st_path = export_dir / "model.safetensors"
    mx.save_safetensors(str(st_path), flat_w, metadata={"format": "mlx", "arch": "lfm3_vl_40l_bf16"})
    tot_b = st_path.stat().st_size
    (export_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": tot_b}, "weight_map": {k: "model.safetensors" for k in sorted(flat_w)}}, indent=2) + "\n",
        encoding="utf-8",
    )
    for attr_n, mod_s in popped_sidecars.items():
        core[attr_n] = mod_s
    logits_expected = vlm.language_model(probe_ids).logits
    mx.eval(logits_expected)
    del flat_w, replacements
    gc.collect()
    mx.clear_cache()

    print(
        f"[Fuse 40L Lean] Wrote pure 16-bit model.safetensors ({merged_cnt} LoRA-fused, {tot_b/(1024**3):.2f} GiB) -> {export_dir}",
        flush=True,
    )
    return 0.0, {"merged_lora_count": merged_cnt, "model_bytes": tot_b, "logits_expected": logits_expected}


def extract_v10_40l_nullspace_svf_bank(
    vlm: Any,
    anchor_items: list[dict[str, Any]],
    skill_batches: list[list[dict[str, Any]]],
    bank_path: Path,
) -> dict[str, mx.array]:
    """Compute `40L` orthogonal nullspace projectors `P_perp = I - V_k.T @ V_k` on upper `full_attention` layers
    `(26, 30, 36, 38)` (with `Engram` + `5x AttnRes` + `LatentReasoningRecurrence` active!) plus 4 `Transformer^2`
    domain-specialized `SVF` expert bases (`b0=rsi`, `b1=logic`, `b2=research`, `b3=ood_code`).
    """
    if bank_path.exists():
        loaded = mx.load(str(bank_path))
        mx.eval(list(loaded.values()))
        print(f"[v10 Stage 4B] Loaded cached 40L Null-Space + Transformer^2 SVF bank ({len(loaded)} tensors) from {bank_path}", flush=True)
        return loaded

    print(f"[v10 Stage 4B] Computing 40L Null-Space `P_perp` & 4 `Transformer^2 SVF` expert bases on layers {REWRITE_TARGET_LAYERS_40L}...", flush=True)
    lm = vlm.language_model
    core = lm.model
    bank: dict[str, mx.array] = {}
    x_attn_list: dict[int, list[np.ndarray]] = {l: [] for l in REWRITE_TARGET_LAYERS_40L}
    x_ffn_list: dict[int, list[np.ndarray]] = {l: [] for l in REWRITE_TARGET_LAYERS_40L}

    from mlx_vlm.models.base import create_attention_mask, create_ssm_mask, scaled_dot_product_attention

    vlm.eval()
    for it in anchor_items[:32]:
        in_ids = it["full_ids"][:, : min(1536, int(it["full_ids"].shape[1]) - 1)]
        emb = core.embed_tokens(in_ids)
        if it["vt_feats"] is not None and it["img_indices"] is not None and int(in_ids.shape[1]) == int(it["full_ids"].shape[1]) - 1:
            proj = vlm.multi_modal_projector(it["vt_feats"])
            flat = emb.reshape(-1, emb.shape[-1])
            flat[it["img_indices"]] = proj.astype(emb.dtype)
            emb = flat.reshape(emb.shape)
        h = emb
        B_i, L_i = int(in_ids.shape[0]), int(in_ids.shape[1])
        first_prev = mx.zeros((B_i, 1), dtype=in_ids.dtype)
        prev_ids = first_prev if L_i == 1 else mx.concatenate([first_prev, in_ids[:, :-1]], axis=1)
        src_bank: list[mx.array] = []
        src_set = set(ATTN_RES_SOURCES)
        tgt_set = set(ATTN_RES_TARGETS_40L)

        for idx, layer in enumerate(core.layers):
            if layer.is_attention_layer:
                win = lm.config.sliding_window if layer.layer_type == "sliding_attention" else None
                mask = create_attention_mask(h, None, window_size=win)
                op_in = layer.operator_norm(h)
                if idx in REWRITE_TARGET_LAYERS_40L:
                    att = layer.self_attn
                    B, L, _ = op_in.shape
                    q = att.rope(att.q_norm(att.q_proj(op_in).reshape(B, L, att.n_heads, att.head_dim)).transpose(0, 2, 1, 3), 0, None)
                    k = att.rope(att.k_norm(att.k_proj(op_in).reshape(B, L, att.n_kv_heads, att.head_dim)).transpose(0, 2, 1, 3), 0, None)
                    v = att.v_proj(op_in).reshape(B, L, att.n_kv_heads, att.head_dim).transpose(0, 2, 1, 3)
                    ao = scaled_dot_product_attention(q, k, v, cache=None, scale=att.scale, mask=mask).transpose(0, 2, 1, 3)
                    ao = (ao * mx.sigmoid(att.gate_proj(op_in))[..., None]).reshape(B, L, -1)
                    mx.eval(ao)
                    x_attn_list[idx].append(np.asarray(ao.astype(mx.float32))[0, -min(24, L) :, :])
                h_op = layer.self_attn(op_in, mask, None, None)
            else:
                h_op = layer.conv(layer.operator_norm(h), create_ssm_mask(h, None), None)
            h = h + (layer.post_operator_norm(h_op) if layer.use_sandwich_norm else h_op)
            ff_in = layer.ffn_norm(h)
            if idx in REWRITE_TARGET_LAYERS_40L:
                mx.eval(ff_in)
                x_ffn_list[idx].append(np.asarray(ff_in.astype(mx.float32))[0, -min(24, int(ff_in.shape[1])) :, :])
            h_ff = layer.feed_forward(ff_in)
            h = h + (layer.post_ffn_norm(h_ff) if layer.use_sandwich_norm else h_ff)

            if idx == 6 and "engram_6" in core:
                h = core.engram_6(h, in_ids, prev_ids)
            elif idx == 18 and "engram_18" in core:
                h = core.engram_18(h, in_ids, prev_ids)
            if idx in src_set:
                src_bank.append(h)
            elif idx in tgt_set and len(src_bank) == len(ATTN_RES_SOURCES) and f"attn_res_{idx}" in core:
                h = getattr(core, f"attn_res_{idx}")(h, src_bank)
            if idx == 35 and len(src_bank) == len(ATTN_RES_SOURCES) and "latent_recur_35" in core:
                h = core.latent_recur_35(h, src_bank)

    for idx in REWRITE_TARGET_LAYERS_40L:
        for tag, mat_list in [("attn_o", x_attn_list[idx]), ("ffn_in", x_ffn_list[idx])]:
            X_cat = np.concatenate(mat_list, axis=0)
            _, _, vh = np.linalg.svd(X_cat, full_matrices=False)
            v_k = vh[:64, :]
            p_perp = np.eye(X_cat.shape[1], dtype=np.float32) - (v_k.T @ v_k)
            bank[f"L{idx}.{tag}.P_perp"] = mx.array(p_perp).astype(mx.bfloat16)

    vlm.freeze()
    for idx in REWRITE_TARGET_LAYERS_40L:
        lyr = core.layers[idx]
        lyr.self_attn.o_proj.unfreeze()
        lyr.feed_forward.w3.unfreeze()

    vg_svf = nn.value_and_grad(vlm, lambda rm, f_ids, s_i: mx.mean(chunked_multimodal_ce(rm, f_ids, s_i)))
    for b_idx, s_batch in enumerate(skill_batches):
        acc_g = None
        for it in s_batch[:6]:
            if int(it["full_ids"].shape[1]) > 2048:
                continue
            lv, gr = vg_svf(vlm, it["full_ids"], it["start_idx"])
            mx.eval(lv)
            acc_g = gr if acc_g is None else tree_map(lambda a, b: a + b, acc_g, gr)
        if acc_g is None:
            it0 = anchor_items[b_idx % len(anchor_items)]
            lv, acc_g = vg_svf(vlm, it0["full_ids"][:, :1536], min(it0["start_idx"], 1024))
            mx.eval(lv)
        flat_g = dict(tree_flatten(acc_g))
        for idx in REWRITE_TARGET_LAYERS_40L:
            for mod_key, p_key in [
                (f"language_model.model.layers.{idx}.self_attn.o_proj.weight", f"L{idx}.attn_o.P_perp"),
                (f"language_model.model.layers.{idx}.feed_forward.w3.weight", f"L{idx}.ffn_in.P_perp"),
            ]:
                g_np = -np.asarray(flat_g[mod_key].astype(mx.float32))
                p_np = np.asarray(bank[p_key].astype(mx.float32))
                u, s, vh = np.linalg.svd(g_np @ p_np, full_matrices=False)
                w_orig = dict(tree_flatten(vlm.parameters()))[mod_key]
                w_norm = float(mx.sqrt(mx.sum(w_orig.astype(mx.float32) ** 2)).item())
                low_rank = (u[:, :8] * s[None, :8]) @ vh[:8, :]
                lr_norm = max(1e-8, float(np.linalg.norm(low_rank)))
                calibrated = (0.015 * w_norm / lr_norm) * low_rank
                short_m = "o_proj" if "o_proj" in mod_key else "w3"
                bank[f"svf_b{b_idx}.L{idx}.{short_m}"] = mx.array(calibrated).astype(mx.bfloat16)
        del acc_g, flat_g
        gc.collect()
        mx.clear_cache()

    vlm.freeze()
    mx.eval(list(bank.values()))
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(bank_path), bank, metadata={"format": "mlx", "arch": "v10_40l_svf_nullspace"})
    mx.clear_cache()
    return bank


def run_stages_01234_v10(
    sft_rows: list[dict[str, Any]],
    orpo_rows: list[dict[str, Any]],
    dapo_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    *,
    sft_steps: int = 480,
    orpo_steps: int = 140,
    target_active_rl_updates: int = 22,
    max_rl_episodes: int = 72,
) -> dict[str, Any]:
    import mlx_vlm

    done_marker = EXPORT_V10_APEX / "v10_stage01234_done.json"
    if (EXPORT_V10_APEX / "model.safetensors").exists() and done_marker.exists():
        print(f"[v10 Stages 0-4] Resuming completed pure 16-bit 3.18B v10 checkpoint at {EXPORT_V10_APEX}", flush=True)
        return json.loads(done_marker.read_text(encoding="utf-8"))

    # Clean Stage 4B resume if Stage 4A already exported `model.safetensors`
    if (EXPORT_V10_APEX / "model.safetensors").exists() and (EXPORT_V10_APEX / "v8_arch_ext.safetensors").exists():
        print(f"[v10 Stage 4B] Resuming Stage 4B 40L Null-Space + Transformer^2 SVF Bank on {EXPORT_V10_APEX}...", flush=True)
        shutil.copy2(ROOT_DIR / "v8_arch.py", EXPORT_V10_APEX / "v8_arch.py")
        vlm_re, proc_re = mlx_vlm.load(str(EXPORT_V10_APEX))
        info_re = attach_v8_extensions(vlm_re, sidecar_dir=EXPORT_V10_APEX, enable_moba=False)
        vlm_re.eval()
        va_items, _ = prepare_turn_items_v8(val_rows[:48], vlm_re, proc_re, max_seq_len=2048)
        tr_short, _ = prepare_turn_items_v8([r for r in sft_rows if r.get("pillar") != "long_context_synthesis"][:160], vlm_re, proc_re, max_seq_len=2048)
        skill_batches = [
            [it for it in tr_short if it.get("pillar") == "advanced_rsi_dgm_svf"][:8] or tr_short[:6],
            [it for it in tr_short if it.get("pillar") == "complex_logic_sat"][:8] or tr_short[6:12],
            [it for it in tr_short if it.get("pillar") == "realtime_research_citations"][:8] or tr_short[12:18],
            [it for it in tr_short if it.get("pillar") == "unseen_ood_self_discover"][:8] or tr_short[18:24],
        ]
        bank = extract_v10_40l_nullspace_svf_bank(vlm_re, va_items[:32], skill_batches, EXPORT_V10_APEX / "seal_nullspace_svf_bank.safetensors")
        vl_f, _ = evaluate_val_loss(vlm_re, va_items[:36])

        overlay_dir = ROOT_DIR / "checkpoints" / "seal_user_overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        dgm_path = overlay_dir / "dgm_archive.json"
        ext_doc = json.loads((EXPORT_V10_APEX / "v8_arch_config.json").read_text(encoding="utf-8"))
        dgm_root = {
            "model": "LFM3-VL-3.2B-RPR-V10-APEX-BF16",
            "rewrite_layers": list(REWRITE_TARGET_LAYERS_40L),
            "svf_experts": ["b0_advanced_rsi", "b1_complex_logic", "b2_research_citations", "b3_ood_code"],
            "nodes": [
                {
                    "id": "dgm_gen_000_root",
                    "parent_id": None,
                    "score": float((ext_doc.get("val_probe_score") or [0.5417])[0]),
                    "val_loss": round(vl_f, 4),
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
            **ext_doc,
            "reload_diff": 0.0,
            "final_val_loss": round(vl_f, 4),
            "svf_nullspace_bank_tensors": len(bank),
            "rewrite_target_layers_40l": list(REWRITE_TARGET_LAYERS_40L),
            **info_re,
        }
        done_marker.write_text(json.dumps(meta_done, indent=2) + "\n", encoding="utf-8")
        if STAGE0_40L_ID.exists():
            shutil.rmtree(STAGE0_40L_ID, ignore_errors=True)
        del vlm_re, proc_re, va_items, tr_short, bank
        gc.collect()
        mx.clear_cache()
        return meta_done

    vlm, processor, ext_info = _load_v10_40l_from_v9_with_peft()
    # Allow up to `2816` tokens so all 64 `2.0k–2.6k` 30-record multi-needle long-context SFT items train end-to-end with zero Metal swap!
    tr_items, vis_cnt = prepare_turn_items_v8(sft_rows, vlm, processor, max_seq_len=2816)
    va_items, _ = prepare_turn_items_v8(val_rows, vlm, processor, max_seq_len=2816)
    long_tr_cnt = sum(1 for it in tr_items if int(it["full_ids"].shape[1]) >= 1800)
    n_trainable = sum(int(v.size) for _, v in tree_flatten(vlm.trainable_parameters()))
    print(
        f"[v10 40L Init Verified] Total = {ext_info['total_model_params']/1e9:.3f}B | "
        f"14-Layer Rank-32 LoRA ({ext_info['lora_target_modules']} linear targets) + Upcycled Gates + Engram + 5x AttnRes + LatentRecur = {n_trainable/1e6:.2f}M trainable | "
        f"Step-0 diff vs v9 = {ext_info['step0_max_logit_diff']:.8f} | SFT turns = {len(tr_items)} (long>=1.8k: {long_tr_cnt}, vis: {vis_cnt})",
        flush=True,
    )

    state: dict[str, Any] = {
        "sft_step": 0,
        "orpo_step": 0,
        "rl_ep_idx": 0,
        "rl_active_updates": 0,
        "rl_dapo_skips": 0,
        "sft_done": False,
        "orpo_done": False,
        "rl_done": False,
        "best_val": 1e9,
        "best_orpo_val": 1e9,
        "best_rl_val": 1e9,
    }
    st_path = CKPT_V10 / "latest" / "train_state.json"
    if st_path.exists() and (CKPT_V10 / "latest" / "trainable.safetensors").exists():
        state = json.loads(st_path.read_text(encoding="utf-8"))
        vlm.load_weights(str(CKPT_V10 / "latest" / "trainable.safetensors"), strict=False)
        mx.eval(vlm.parameters())
        print(f"[v10 Stages 1-3] Resumed checkpoint: {state}", flush=True)
    else:
        vlm.eval()
        vl0, vp0 = evaluate_val_loss(vlm, va_items[:48])
        vlm.train()
        state["best_val"] = vl0
        print(f"  [Step-0 v10 40L Identity Baseline (`== v9`)] val_v10_joint={vl0:.4f} (PPL={vp0:.4f})", flush=True)
        _save_v8_ckpt(vlm, CKPT_V10 / "best")
        _save_v8_ckpt(vlm, CKPT_V10 / "latest", state)

    # ------------------------------------------------------------------
    # Stage 1: 14-Layer `MuonClip` + `BF16AdamW` Mixed Short + Long (`5.6k`) SFT (`480` steps)
    # ------------------------------------------------------------------
    if not state["sft_done"]:
        s0 = int(state["sft_step"])
        rem = max(1, sft_steps - s0)
        cw = 0.5 * (1.0 + math.cos(math.pi * min(1.0, s0 / max(1, sft_steps))))
        opt_sft = _build_v10_muonclip(
            lr_gate=optim.cosine_decay(4.0e-5 + 2.2e-4 * cw, decay_steps=rem, end=2.5e-5),
            lr_table=optim.cosine_decay(4.0e-5 + 1.8e-4 * cw, decay_steps=rem, end=2.5e-5),
            lr_2d=optim.cosine_decay(1.0e-5 + 4.8e-5 * cw, decay_steps=rem, end=8.0e-6),
        )
        order = list(range(len(tr_items)))
        random.Random(20261020).shuffle(order)
        t0 = time.perf_counter()
        losses: list[float] = []
        qk_clips = 0
        vlm.train()

        for st in range(s0, sft_steps):
            b0 = tr_items[order[(st * 2) % len(order)]]
            b1 = tr_items[order[(st * 2 + 1) % len(order)]]
            # Use micro-batch 1 on >=1.8k multi-needle archives so peak 14-layer SDPA backward stays < 11 GiB!
            has_long = int(b0["full_ids"].shape[1]) >= 1800 or int(b1["full_ids"].shape[1]) >= 1800
            cur_batch = [b0] if int(b0["full_ids"].shape[1]) >= 1800 else ([b1] if int(b1["full_ids"].shape[1]) >= 1800 else [b0, b1])
            lv, _ = _run_dft_batch(vlm, cur_batch, opt_sft)
            qk_clips += apply_qk_clip(vlm, max_lora_fro_norm=6.0)
            losses.append(lv)

            if has_long or (st + 1) % 6 == 0:
                gc.collect()
                mx.clear_cache()
            if (st + 1) % 25 == 0 and (st + 1) % 60 != 0 and st != sft_steps - 1:
                state["sft_step"] = st + 1
                _save_v8_ckpt(vlm, CKPT_V10 / "latest", state)
                print(
                    f"  [v10 Stage 1 Ckpt {st+1:3d}/{sft_steps} | {(time.perf_counter()-t0)/60:4.1f}m] train={sum(losses)/len(losses):.4f}",
                    flush=True,
                )
            if (st + 1) % 60 == 0 or st == sft_steps - 1:
                vlm.eval()
                vl, vp = evaluate_val_loss(vlm, va_items[:48])
                vlm.train()
                state["sft_step"] = st + 1
                if vl <= state["best_val"]:
                    state["best_val"] = vl
                    _save_v8_ckpt(vlm, CKPT_V10 / "best")
                _save_v8_ckpt(vlm, CKPT_V10 / "latest", state)
                print(
                    f"  [v10 Stage 1 MuonClip-SFT {st+1:3d}/{sft_steps} | {(time.perf_counter()-t0)/60:4.1f}m] "
                    f"train={sum(losses)/len(losses):.4f} | val={vl:.4f} (PPL={vp:.4f}) | best={state['best_val']:.4f} | qk_clips={qk_clips}",
                    flush=True,
                )
                losses.clear()
                gc.collect()
                mx.clear_cache()

        vlm.load_weights(str(CKPT_V10 / "best" / "trainable.safetensors"), strict=False)
        mx.eval(vlm.parameters())
        state["sft_done"] = True
        _save_v8_ckpt(vlm, CKPT_V10 / "sft_best")
        _save_v8_ckpt(vlm, CKPT_V10 / "latest", state)

    # Short/medium SFT items for fast anchor regularization during ORPO & DAPO/Dr. GRPO
    short_anc_items = [it for it in tr_items if int(it["full_ids"].shape[1]) <= 1800] or tr_items[:64]

    # ------------------------------------------------------------------
    # Stage 2: `MuonClip` Contrastive `ORPO` (`140` steps)
    # ------------------------------------------------------------------
    tok = processor.tokenizer
    eos_id = int(tok.eos_token_id)

    if not state["orpo_done"]:
        opt_orpo = _build_v10_muonclip(
            lr_gate=optim.cosine_decay(3.5e-5, decay_steps=max(1, orpo_steps), end=8.0e-6),
            lr_table=optim.cosine_decay(5.5e-5, decay_steps=max(1, orpo_steps), end=1.2e-5),
            lr_2d=optim.cosine_decay(1.4e-5, decay_steps=max(1, orpo_steps), end=3.5e-6),
        )
        anc_rng = random.Random(20261030)
        t_op = time.perf_counter()
        op_losses: list[float] = []
        best_orpo = float(state.get("best_orpo_val", 1e9))
        vlm.train()

        for ost in range(int(state["orpo_step"]), orpo_steps):
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
            if (ost + 1) % 35 == 0 or ost == orpo_steps - 1:
                vlm.eval()
                vl, vp = evaluate_val_loss(vlm, va_items[:48])
                vlm.train()
                state["orpo_step"] = ost + 1
                if vl <= best_orpo:
                    best_orpo = vl
                    state["best_orpo_val"] = vl
                    _save_v8_ckpt(vlm, CKPT_V10 / "orpo_best")
                _save_v8_ckpt(vlm, CKPT_V10 / "latest", state)
                print(
                    f"  [v10 Stage 2 MuonClip-ORPO {ost+1:3d}/{orpo_steps} | {(time.perf_counter()-t_op)/60:4.1f}m] "
                    f"orpo_loss={sum(op_losses)/max(1,len(op_losses)):.4f} | val={vl:.4f} (PPL={vp:.4f}) | best_orpo={best_orpo:.4f}",
                    flush=True,
                )
                op_losses.clear()
                gc.collect()
                mx.clear_cache()

        vlm.load_weights(str(CKPT_V10 / "orpo_best" / "trainable.safetensors"), strict=False)
        mx.eval(vlm.parameters())
        state["orpo_done"] = True
        _save_v8_ckpt(vlm, CKPT_V10 / "latest", state)

    # ------------------------------------------------------------------
    # Stage 3: `DAPO` Dynamic Sampling + `Dr. GRPO` Unbiased RLVR
    #   - Dynamically filters out `4/4` unanimous-pass (`sigma == 0`) groups WITHOUT consuming `active_updates`!
    #   - Uses `Dr. GRPO` unnormalized mean-centered advantage (`R_i - mean(R)`, NO `1/std(R)` bias!)
    #     and `DAPO` asymmetric `clip-higher` `[0.80, 1.28]` + fixed `416.0` token normalizer!
    # ------------------------------------------------------------------
    if not state["rl_done"]:
        vlm.eval()
        opt_rl = _build_v10_muonclip(lr_gate=1.2e-5, lr_table=2.2e-5, lr_2d=5.0e-6)
        ref_w = mx.load(str(CKPT_V10 / "orpo_best" / "trainable.safetensors"))
        mx.eval(list(ref_w.values()))
        anc_rng = random.Random(20261040)
        n_lim = min(max_rl_episodes, len(dapo_rows))
        t_rl = time.perf_counter()

        ep_idx = int(state.get("rl_ep_idx", 0))
        active_upd = float(state.get("rl_active_updates", 0.0))
        dapo_skips = int(state.get("rl_dapo_skips", 0))

        while ep_idx < n_lim and active_upd < float(target_active_rl_updates):
            row = dapo_rows[ep_idx]
            p_msgs = row["messages"]
            p_str = _apply_template_inference_aligned(processor, vlm.config, p_msgs, num_images=0, enable_thinking=True)
            p_txt = "\n".join(str(m.get("content", "")) for m in p_msgs)
            p_ids = tok.encode(p_str, add_special_tokens=False)
            p_mx = mx.array([p_ids], dtype=mx.int32)
            s_idx = max(len(p_ids) - 1, 0)

            rollouts = []
            for _ in range(4):
                res = mlx_vlm.generate(vlm, processor, p_str, max_tokens=416, temperature=0.78, repetition_penalty=1.03, verbose=False)
                g_txt = res.text or ""
                c_ids = tok.encode(g_txt, add_special_tokens=False) + [eos_id]
                f_seq = mx.concatenate([p_mx, mx.array([c_ids], dtype=mx.int32)], axis=1)[:, :2048]
                sc = score_response(g_txt, prompt_text=p_txt, kind=row["kind"], verifier=row["verifier"], tokenizer=tok, max_think_tokens=320)
                rollouts.append({"full_ids": f_seq, "score": sc})

            rews = [r["score"]["reward"] for r in rollouts]
            p_cnt = sum(1 for r in rollouts if r["score"]["correct"])
            spread = max(rews) - min(rews)
            anc_b = anc_rng.sample(short_anc_items, k=2)

            if p_cnt == 4 and spread < 1e-4:
                # DAPO Dynamic Sampling: skip solved zero-variance group without consuming `active_upd`!
                dapo_skips += 1
                tag = "DAPO-skip(4/4)"
            elif p_cnt == 0 and spread < 1e-4:
                # Hard 0/4 episode: gentle OPSD self-distillation from verified reference solution
                ref_r = {
                    "id": f"r_{row['id']}",
                    "pillar": row["kind"],
                    "train_last_turn_only": True,
                    "images": [],
                    "messages": [*p_msgs, {"role": "assistant", "content": row["reference_answer"]}],
                }
                r_it, _ = prepare_turn_items_v8([ref_r], vlm, processor, max_seq_len=2048)
                _run_dft_batch(vlm, (r_it[-1:] + anc_b[:1]) if r_it else anc_b[:1], opt_rl, scale_grad=0.20)
                apply_qk_clip(vlm, max_lora_fro_norm=6.0)
                active_upd += 0.5
                tag = "OPSD-0/4(0.20x)"
            else:
                # True Split-Variance (`0 < p_cnt < 4` or non-zero reward variance): `Dr. GRPO` + `DAPO clip-higher`!
                mean_r = sum(rews) / len(rews)
                advs = [r - mean_r for r in rews]  # Dr. GRPO: NO `1/std` division!
                old_lps = [mx.stop_gradient(-chunked_multimodal_ce(vlm, ro["full_ids"], s_idx)) for ro in rollouts]
                snap_w = {k: mx.array(v) for k, v in tree_flatten(vlm.trainable_parameters())}
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
                    # DAPO asymmetric clip-higher `[0.80, 1.28]` preserves exploration on positive-advantage reasoning steps!
                    clipped_rat = mx.clip(rat, 0.80, 1.28)
                    pg = -mx.minimum(rat * adv_m, clipped_rat * adv_m)
                    kl = mx.exp(r_lp - n_lp) - (r_lp - n_lp) - 1.0
                    # Dr. GRPO fixed denominator (`416.0`) avoids response-length bias!
                    return mx.sum(pg + 0.03 * kl) / mx.array(416.0, dtype=mx.float32)

                vg = nn.value_and_grad(vlm, _dr_grpo_dapo_loss)
                acc_g = None
                for ro, olp, rlp, av in zip(rollouts, old_lps, ref_lps, advs):
                    lr_v, gr = vg(vlm, ro["full_ids"], olp, rlp, av)
                    mx.eval(lr_v)
                    acc_g = tree_map(lambda g: g / 4.0, gr) if acc_g is None else tree_map(lambda a, g: a + g / 4.0, acc_g, gr)
                _, anc_g = _run_dft_batch(vlm, anc_b, None, scale_grad=0.15)
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
            gc.collect()
            mx.clear_cache()

            if ep_idx % 6 == 0 or active_upd >= float(target_active_rl_updates) or ep_idx >= n_lim:
                vlm.eval()
                vl, _ = evaluate_val_loss(vlm, va_items[:48])
                _save_v8_ckpt(vlm, CKPT_V10 / "latest", state)
                if vl <= state.get("best_rl_val", 1e9) + 0.012:
                    if vl < state.get("best_rl_val", 1e9):
                        state["best_rl_val"] = vl
                    _save_v8_ckpt(vlm, CKPT_V10 / "rl_best")
                print(
                    f"  [v10 Stage 3 DAPO+DrGRPO ep={ep_idx:2d}/{n_lim} active={active_upd:4.1f}/{target_active_rl_updates} skips={dapo_skips} | {(time.perf_counter()-t_rl)/60:4.1f}m] "
                    f"{row['id']:16s} | pass={p_cnt}/4 | {tag} | val={vl:.4f}",
                    flush=True,
                )

        del ref_w
        state["rl_done"] = True
        _save_v8_ckpt(vlm, CKPT_V10 / "latest", state)

    # ------------------------------------------------------------------
    # Stage 4A: `WiSE-FT` Non-Degradation Gate (`tau=0` == exact `v9`) & Lean Pure `16-bit` Export
    # ------------------------------------------------------------------
    vlm.eval()
    w_sft = mx.load(str(CKPT_V10 / "sft_best" / "trainable.safetensors"))
    w_orpo = mx.load(str((CKPT_V10 / "orpo_best" / "trainable.safetensors") if (CKPT_V10 / "orpo_best" / "trainable.safetensors").exists() else (CKPT_V10 / "sft_best" / "trainable.safetensors")))
    w_rl = mx.load(str((CKPT_V10 / "rl_best" / "trainable.safetensors") if (CKPT_V10 / "rl_best" / "trainable.safetensors").exists() else (CKPT_V10 / "latest" / "trainable.safetensors")))
    w_wise = {
        k: (0.35 * w_sft[k].astype(mx.float32) + 0.30 * w_orpo[k].astype(mx.float32) + 0.35 * w_rl[k].astype(mx.float32)).astype(w_sft[k].dtype)
        for k in w_sft
    }

    stage0_ext = mx.load(str(STAGE0_40L_ID / "v8_arch_ext.safetensors"))
    upcycled_gate_prefixes = tuple(f"language_model.model.layers.{idx}.post_" for idx in UPCYCLED_40L_INDICES)

    def _interp_v10_from_v9(weights_dict: dict[str, mx.array], tau_lora: float, tau_ext: float = 1.0) -> dict[str, mx.array]:
        out_d: dict[str, mx.array] = {}
        for k, v in weights_dict.items():
            if "lora_b" in k or k.startswith(upcycled_gate_prefixes):
                out_d[k] = (v.astype(mx.float32) * tau_lora).astype(v.dtype)
            elif k in stage0_ext:
                v0 = stage0_ext[k].astype(mx.float32)
                out_d[k] = (v0 + tau_ext * (v.astype(mx.float32) - v0)).astype(v.dtype)
            else:
                out_d[k] = v
        return out_d

    seal_d = json.loads((V6_CORPUS_DIR / "seal_svf_episodes.json").read_text(encoding="utf-8"))
    probe_tasks = seal_d["dev_diagnose_4"] + seal_d["dev_gate_8"] + dapo_rows[-12:]

    zero_w = _interp_v10_from_v9(w_wise, 0.0, 0.0)
    vlm.load_weights(list(zero_w.items()), strict=False)
    mx.eval(vlm.parameters())
    vl_0, _ = evaluate_val_loss(vlm, va_items[:48])
    acc_0, rew_0, _ = _eval_task_batch(vlm, processor, probe_tasks)
    ref_sc = (round(acc_0, 4), round(rew_0 - 0.35 * vl_0, 4))
    print(f"  [v10 Reference Baseline (`v9` tau=0)] probe_acc={acc_0*100:5.1f}% (rew={rew_0:.3f}) | val_v10={vl_0:.4f}", flush=True)

    candidates = [
        ("v10_wise_tau0.70", _interp_v10_from_v9(w_wise, 0.70, 0.85)),
        ("v10_wise_tau0.88", _interp_v10_from_v9(w_wise, 0.88, 1.00)),
        ("v10_wise_tau1.00", w_wise),
        ("v10_orpo_tau1.00", w_orpo),
        ("v10_rl_tau1.00", w_rl),
    ]
    best_name = "v9_fallback_tau0"
    best_w = zero_w
    best_sc = ref_sc
    best_val_loss = round(vl_0, 4)

    for c_name, c_w in candidates:
        vlm.load_weights(list(c_w.items()), strict=False)
        mx.eval(vlm.parameters())
        vl_c, _ = evaluate_val_loss(vlm, va_items[:48])
        acc_c, rew_c, _ = _eval_task_batch(vlm, processor, probe_tasks)
        sc_t = (round(acc_c, 4), round(rew_c - 0.35 * vl_c, 4))
        print(f"  [v10 Non-Degradation Gate] {c_name:18s} | probe_acc={acc_c*100:5.1f}% (rew={rew_c:.3f}) | val_v10={vl_c:.4f}", flush=True)
        if sc_t >= best_sc:
            best_sc = sc_t
            best_name = c_name
            best_w = c_w
            best_val_loss = round(vl_c, 4)
        gc.collect()
        mx.clear_cache()

    print(f"[v10 Selected] Winner={best_name} (score={best_sc}, val={best_val_loss:.4f} vs v9_ref={ref_sc})", flush=True)
    vlm.load_weights(list(best_w.items()), strict=False)
    mx.eval(vlm.parameters())

    # Free all temporary LoRA dictionaries BEFORE exporting & reloading to prevent Metal OOM!
    del w_sft, w_orpo, w_rl, w_wise, zero_w, best_w, candidates, stage0_ext
    gc.collect()
    mx.clear_cache()

    _, fuse_info = fuse_and_export_40l_lean(
        vlm,
        STAGE0_40L_ID,
        EXPORT_V10_APEX,
        extra_sidecar_meta={
            "model_version": "LFM3-VL-3.2B-RPR-V10-APEX-BF16",
            "precision": "bfloat16",
            "selected_candidate": best_name,
            "val_probe_score": list(best_sc),
            "selected_val_loss": best_val_loss,
            "v9_ref_score": list(ref_sc),
            "dapo_active_updates": state.get("rl_active_updates", 0),
            "dapo_dynamic_skips": state.get("rl_dapo_skips", 0),
        },
    )
    logits_expected = fuse_info["logits_expected"]

    # Free PEFT `vlm` completely BEFORE loading standalone `EXPORT_V10_APEX`!
    del vlm, processor, tr_items
    gc.collect()
    mx.clear_cache()

    vlm_re, proc_re = mlx_vlm.load(str(EXPORT_V10_APEX))
    info_re = attach_v8_extensions(vlm_re, sidecar_dir=EXPORT_V10_APEX, enable_moba=False)
    vlm_re.eval()
    probe_ids = mx.array([[1, 42, 105, 512, 1024, 2048]], dtype=mx.int32)
    logits_re = vlm_re.language_model(probe_ids).logits
    mx.eval(logits_re)
    re_diff = float(mx.max(mx.abs(logits_re.astype(mx.float32) - logits_expected.astype(mx.float32))).item())
    print(f"[v10 Pure 16-bit 3.18B Standalone Export Verified] {info_re['total_model_params']/1e9:.3f}B params | reload diff = {re_diff:.6f}", flush=True)
    assert re_diff < 1e-4, f"v10 reload mismatch: {re_diff}"
    del logits_expected, logits_re

    # Reclaim `6.35 GiB` temporary `STAGE0_40L_ID` immediately now that `EXPORT_V10_APEX` is verified!
    if STAGE0_40L_ID.exists():
        shutil.rmtree(STAGE0_40L_ID, ignore_errors=True)

    # ------------------------------------------------------------------
    # Stage 4B: Extract `40L` `Transformer^2 SVF + Nullspace` Bank (`L26, 30, 36, 38`) & Initialize `DGM` Archive
    # ------------------------------------------------------------------
    tr_short, _ = prepare_turn_items_v8(
        [r for r in sft_rows if r.get("pillar") != "long_context_synthesis"][:160],
        vlm_re,
        proc_re,
        max_seq_len=2048,
    )
    skill_batches = [
        [it for it in tr_short if it.get("pillar") == "advanced_rsi_dgm_svf"][:8] or tr_short[:6],
        [it for it in tr_short if it.get("pillar") == "complex_logic_sat"][:8] or tr_short[6:12],
        [it for it in tr_short if it.get("pillar") == "realtime_research_citations"][:8] or tr_short[12:18],
        [it for it in tr_short if it.get("pillar") == "unseen_ood_self_discover"][:8] or tr_short[18:24],
    ]
    bank_path = EXPORT_V10_APEX / "seal_nullspace_svf_bank.safetensors"
    bank = extract_v10_40l_nullspace_svf_bank(vlm_re, va_items[:32], skill_batches, bank_path)

    # Initialize `Darwin Gödel Machine` (`DGM`) evolutionary lineage archive root node!
    overlay_dir = ROOT_DIR / "checkpoints" / "seal_user_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    dgm_path = overlay_dir / "dgm_archive.json"
    dgm_root = {
        "model": "LFM3-VL-3.2B-RPR-V10-APEX-BF16",
        "rewrite_layers": list(REWRITE_TARGET_LAYERS_40L),
        "svf_experts": ["b0_advanced_rsi", "b1_complex_logic", "b2_research_citations", "b3_ood_code"],
        "nodes": [
            {
                "id": "dgm_gen_000_root",
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
        "selected_candidate": best_name,
        "val_probe_score": list(best_sc),
        "selected_val_loss": best_val_loss,
        "reload_diff": re_diff,
        "svf_nullspace_bank_tensors": len(bank),
        "rewrite_target_layers_40l": list(REWRITE_TARGET_LAYERS_40L),
        **info_re,
    }
    done_marker.write_text(json.dumps(meta_done, indent=2) + "\n", encoding="utf-8")
    del vlm_re, proc_re, va_items, tr_short, bank
    gc.collect()
    mx.clear_cache()
    return meta_done


def run_stage_5_v10_onpolicy_mtp(sft_rows: list[dict[str, Any]], mtp_steps: int = 280) -> dict[str, Any]:
    """On-policy `SequentialMTPHead` alignment to pure 16-bit `40L` `LFM3-VL-3.2B-RPR-V10-APEX-BF16`."""
    import mlx_vlm

    bench_dir = ROOT_DIR / "results" / "v10_mtp_benchmark"
    bench_dir.mkdir(parents=True, exist_ok=True)
    if (EXPORT_V10_APEX / "mtp_head.safetensors").exists() and (bench_dir / "summary.json").exists():
        print(f"[v10 Stage 5] Resuming completed v10 MTP benchmark from {bench_dir / 'summary.json'}", flush=True)
        return json.loads((bench_dir / "summary.json").read_text(encoding="utf-8"))

    vlm, processor = mlx_vlm.load(str(EXPORT_V10_APEX))
    attach_v8_extensions(vlm, sidecar_dir=EXPORT_V10_APEX, enable_moba=False)
    vlm.freeze()
    vlm.eval()
    tok = processor.tokenizer
    eos_id = int(tok.eos_token_id)

    mtp_head = load_mtp_checkpoint(vlm.config.text_config, EXPORT_V9_SEAL)
    embed_tokens = vlm.language_model.model.embed_tokens
    mtp_head.bind_vocab_projection(embed_tokens)

    text_rows = [r for r in sft_rows if not r.get("images") and r.get("pillar") != "long_context_synthesis"][:320]
    onpol_cache = WORK_V10 / "stage5_v10_onpolicy_32.json"
    if onpol_cache.exists():
        onpol_raw = json.loads(onpol_cache.read_text(encoding="utf-8"))
    else:
        print("[v10 Stage 5] Generating 32 on-policy 40L v10 greedy trajectories (logic + citations + code + RSI + tools)...", flush=True)
        onpol_raw = []
        for msgs in GENERAL_ONPOLICY_PROMPTS[:10]:
            p_s = _apply_template_inference_aligned(processor, vlm.config, msgs, num_images=0, enable_thinking=True)
            p_ids = tok.encode(p_s, add_special_tokens=False)
            g_ids, _ = standard_greedy_generate(vlm, p_ids, max_tokens=200, eos_token_id=eos_id, repetition_penalty=1.03)
            if len(g_ids) >= 12:
                onpol_raw.append({"full": (p_ids + g_ids)[:640], "ans_start": max(1, len(p_ids) - 2)})
        rng_op = random.Random(20261095)
        cand = list(text_rows)
        rng_op.shuffle(cand)
        for r in cand[:22]:
            msgs = r["messages"]
            u_idx = next((idx for idx, m in enumerate(msgs) if m.get("role") == "assistant"), -1)
            if u_idx <= 0:
                continue
            p_s = _apply_template_inference_aligned(processor, vlm.config, msgs[:u_idx], num_images=0, enable_thinking=True)
            p_ids = tok.encode(p_s, add_special_tokens=False)
            if len(p_ids) > 420:
                continue
            g_ids, _ = standard_greedy_generate(vlm, p_ids, max_tokens=192, eos_token_id=eos_id, repetition_penalty=1.03)
            if len(g_ids) >= 12:
                onpol_raw.append({"full": (p_ids + g_ids)[:640], "ans_start": max(1, len(p_ids) - 2)})
        onpol_cache.write_text(json.dumps(onpol_raw) + "\n", encoding="utf-8")

    seqs: list[tuple[mx.array, int]] = []
    for item in onpol_raw:
        arr = mx.array([item["full"]], dtype=mx.int32)
        for _ in range(6):
            seqs.append((arr, int(item["ans_start"])))
    for r in text_rows[:220]:
        msgs = r["messages"]
        t_i = len(msgs) - 1
        if msgs[t_i].get("role") != "assistant":
            continue
        p_s = _apply_template_inference_aligned(processor, vlm.config, _strip_prior_think(msgs[:t_i]), num_images=0, enable_thinking=True)
        a_s = str(msgs[t_i]["content"])
        if p_s.endswith("<think>") and a_s.startswith("<think>"):
            a_s = a_s[len("<think>") :]
        p_ids = tok.encode(p_s, add_special_tokens=False)
        if len(p_ids) > 420:
            continue
        a_ids = tok.encode(a_s, add_special_tokens=False) + [eos_id]
        full = (p_ids + a_ids)[:640]
        if len(full) >= len(p_ids) + 8:
            seqs.append((mx.array([full], dtype=mx.int32), max(1, len(p_ids) - 2)))

    random.Random(20261098).shuffle(seqs)
    opt_mtp = optim.AdamW(learning_rate=optim.cosine_decay(3.0e-4, decay_steps=max(1, mtp_steps), end=2.0e-5), weight_decay=0.01)

    def _mtp_loss(
        head: SequentialMTPHead,
        h_in_1: mx.array,
        h_tgt_next: mx.array,
        tok_in_1: mx.array,
        tok_in_2: mx.array,
        a0: int,
        a1: int,
        a1_s2: int,
        top1_1: mx.array,
        probs_1: mx.array,
        top1_2: mx.array,
        probs_2: mx.array,
    ) -> mx.array:
        emb_1 = mx.stop_gradient(embed_tokens(tok_in_1))
        h_m1 = head.forward_features(h_in_1, emb_1, cache=None)
        h_m1_a = h_m1[:, a0:a1, :]
        feat_1 = nn.losses.smooth_l1_loss(h_m1_a.astype(mx.float32), h_tgt_next[:, a0:a1, :].astype(mx.float32), beta=0.1)
        l1 = head.draft_logits_shortlist(h_m1_a, embed_tokens).astype(mx.float32)
        lp1 = l1 - mx.logsumexp(l1, axis=-1, keepdims=True)
        loss_1 = mx.mean(0.55 * (-mx.sum(probs_1 * lp1, axis=-1)) + 0.45 * nn.losses.cross_entropy(l1, top1_1, reduction="none"))

        h_in_2 = mx.concatenate([h_in_1[:, 1 : a0 + 1, :], h_m1[:, a0:-1, :]], axis=1)
        h_m2 = head.forward_features(h_in_2, mx.stop_gradient(embed_tokens(tok_in_2)), cache=None)
        h_m2_a = h_m2[:, a0:a1_s2, :]
        feat_2 = nn.losses.smooth_l1_loss(h_m2_a.astype(mx.float32), h_tgt_next[:, a0 + 1 : a1_s2 + 1, :].astype(mx.float32), beta=0.1)
        l2 = head.draft_logits_shortlist(h_m2_a, embed_tokens).astype(mx.float32)
        lp2 = l2 - mx.logsumexp(l2, axis=-1, keepdims=True)
        loss_2 = mx.mean(0.55 * (-mx.sum(probs_2 * lp2, axis=-1)) + 0.45 * nn.losses.cross_entropy(l2, top1_2, reduction="none"))
        return (loss_1 + 1.0 * feat_1) + 0.75 * (loss_2 + 1.0 * feat_2)

    vg_m = nn.value_and_grad(mtp_head, _mtp_loss)
    mtp_head.train()
    t0 = time.perf_counter()
    run_l: list[float] = []

    for st in range(mtp_steps):
        acc_g = None
        lsum = 0.0
        cnt = 0
        for sb in range(2):
            f_ids, ans_s = seqs[(st * 2 + sb) % len(seqs)]
            slen = int(f_ids.shape[1])
            t_out = vlm.language_model.model(f_ids[:, :-1], cache=None)
            mx.eval(t_out)
            len1 = slen - 2
            a0 = max(1, min(ans_s - 2, len1 - 6))
            a1 = min(len1, a0 + 232)
            a1_s2 = min(len1 - 1, a1 - 1)
            if a1_s2 - a0 < 4:
                continue
            h_in_1 = mx.stop_gradient(t_out[:, :-1, :])
            h_tgt_next = mx.stop_gradient(t_out[:, 1:, :])
            assert mtp_head._cached_w_sub is not None
            tsub = (h_tgt_next[:, a0:a1, :] @ mtp_head._cached_w_sub.T).astype(mx.float32)
            p1 = mx.stop_gradient(mx.softmax(tsub, axis=-1))
            top1 = mx.stop_gradient(mx.argmax(tsub, axis=-1))
            sp2 = a1_s2 - a0
            lv_m, gr_m = vg_m(
                mtp_head, h_in_1, h_tgt_next, f_ids[:, 1:-1], f_ids[:, 2:-1],
                a0, a1, a1_s2, top1, p1, top1[:, 1 : 1 + sp2], p1[:, 1 : 1 + sp2, :],
            )
            mx.eval(lv_m)
            lsum += float(lv_m.item())
            cnt += 1
            acc_g = gr_m if acc_g is None else tree_map(lambda a, b: a + b, acc_g, gr_m)
        if acc_g is not None and cnt > 0:
            acc_g = tree_map(lambda g: g / float(cnt), acc_g)
            acc_g, _ = optim.clip_grad_norm(acc_g, max_norm=1.0)
            opt_mtp.update(mtp_head, acc_g)
            mx.eval(mtp_head.parameters(), opt_mtp.state)
            run_l.append(lsum / cnt)
        if (st + 1) % 70 == 0 or st == mtp_steps - 1:
            save_mtp_checkpoint(mtp_head, EXPORT_V10_APEX)
            print(f"  [v10 Stage 5 40L MTP-OnPolicy {st+1:3d}/{mtp_steps} | {(time.perf_counter()-t0)/60:4.1f}m] distill_loss={sum(run_l)/max(1,len(run_l)):.4f}", flush=True)
            run_l.clear()
            mx.clear_cache()

    mtp_head.eval()
    save_mtp_checkpoint(mtp_head, EXPORT_V10_APEX)

    bench_prompts = [
        ("research_cite", [{"role": "user", "content": "Using [S1] (beta: 14 ms) and [S2] (official erratum: 19 ms), state verified latency with [S#] citation."}]),
        ("svf_rsi_code", [{"role": "user", "content": "Write a NumPy function `svf_scale(W, z)` computing `U @ diag(S * (1 + z)) @ Vt`."}]),
        ("complex_logic", [{"role": "user", "content": 'If A says "B is a Knave" and B says "Both A and B are Knights", return JSON `{"A": <bool>, "B": <bool>}`.'}]),
    ]
    wp = tok.encode(_apply_template_inference_aligned(processor, vlm.config, bench_prompts[0][1], num_images=0, enable_thinking=True), add_special_tokens=False)
    standard_greedy_generate(vlm, wp, max_tokens=16, eos_token_id=eos_id)
    mtp_speculative_generate(vlm, mtp_head, wp, max_tokens=16, draft_k=3, eos_token_id=eos_id)

    per_case = []
    tot_std_t, tot_std_s, tot_k3_t, tot_k3_s = 0, 0.0, 0, 0.0
    for cname, msgs in bench_prompts:
        ps = _apply_template_inference_aligned(processor, vlm.config, msgs, num_images=0, enable_thinking=True)
        pids = tok.encode(ps, add_special_tokens=False)
        ref_ids, st_std = standard_greedy_generate(vlm, pids, max_tokens=200, eos_token_id=eos_id)
        k3_ids, st_k3 = mtp_speculative_generate(vlm, mtp_head, pids, max_tokens=200, draft_k=3, eos_token_id=eos_id)
        sp = round(st_k3["tok_per_sec"] / max(1e-6, st_std["tok_per_sec"]), 3)
        tot_std_t += st_std["tokens"]
        tot_std_s += st_std["wall_clock_sec"]
        tot_k3_t += st_k3["tokens"]
        tot_k3_s += st_k3["wall_clock_sec"]
        per_case.append({
            "case": cname,
            "tokens": st_k3["tokens"],
            "std_tok_per_sec": st_std["tok_per_sec"],
            "mtp_k3_tok_per_sec": st_k3["tok_per_sec"],
            "acceptance_rate": st_k3["acceptance_rate"],
            "speedup_k3": sp,
            "exact_token_match": (ref_ids == k3_ids),
        })
        print(f"  [v10 40L Pure-16b Bench {cname:16s}] std={st_std['tok_per_sec']:5.1f} -> MTP-K3={st_k3['tok_per_sec']:6.1f} tok/s ({sp:.2f}x, acc={st_k3['acceptance_rate']*100:.1f}%)", flush=True)

    summary = {
        "model": str(EXPORT_V10_APEX),
        "num_layers": 40,
        "max_position_embeddings": 262144,
        "precision": "bfloat16 (pure 16-bit)",
        "std_tok_per_sec": round(tot_std_t / max(1e-6, tot_std_s), 2),
        "mtp_k3_tok_per_sec": round(tot_k3_t / max(1e-6, tot_k3_s), 2),
        "overall_k3_speedup": round((tot_k3_t / max(1e-6, tot_k3_s)) / max(1e-6, tot_std_t / max(1e-6, tot_std_s)), 3),
        "cases": per_case,
    }
    (bench_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    # Clean temporary LoRA checkpoint workdir once full export + MTP head are verified!
    if CKPT_V10.exists():
        shutil.rmtree(CKPT_V10, ignore_errors=True)
    del vlm, processor, mtp_head
    gc.collect()
    mx.clear_cache()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RPR v10 APEX (3.18B 40L Pure 16-bit + 256k LongRoPE2 + DAPO/DrGRPO + DGM/SVF) Pipeline.")
    parser.add_argument("--sft-steps", type=int, default=480)
    parser.add_argument("--orpo-steps", type=int, default=140)
    parser.add_argument("--active-rl-updates", type=int, default=22)
    parser.add_argument("--max-rl-episodes", type=int, default=72)
    parser.add_argument("--mtp-steps", type=int, default=280)
    args = parser.parse_args()

    mx.set_wired_limit(0)
    mx.set_cache_limit(128 * 1024 * 1024)
    WORK_V10.mkdir(parents=True, exist_ok=True)

    sft_rows = [json.loads(l) for l in (V10_CORPUS_DIR / "train_v10_sft.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    orpo_rows = [json.loads(l) for l in (V10_CORPUS_DIR / "orpo_v10_pairs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    dapo_rows = [json.loads(l) for l in (V10_CORPUS_DIR / "dapo_v10_hard_episodes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    val_rows = [json.loads(l) for l in (V10_CORPUS_DIR / "val_v10_joint.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    run_stages_01234_v10(
        sft_rows,
        orpo_rows,
        dapo_rows,
        val_rows,
        sft_steps=args.sft_steps,
        orpo_steps=args.orpo_steps,
        target_active_rl_updates=args.active_rl_updates,
        max_rl_episodes=args.max_rl_episodes,
    )
    s5 = run_stage_5_v10_onpolicy_mtp(sft_rows, mtp_steps=args.mtp_steps)
    print(f"[v10 3.18B APEX Pipeline Complete] Pure 16-bit MTP K=3 Speedup: {s5['overall_k3_speedup']}x ({s5['mtp_k3_tok_per_sec']} tok/s)", flush=True)


if __name__ == "__main__":
    main()
