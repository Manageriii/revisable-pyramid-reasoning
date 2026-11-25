"""Master Checkpointed RPR v9 (`2.91B` Pure 16-bit `bfloat16`) Training, Outer-Loop SEAL, & Cline/Hermes Export Pipeline:

Builds `checkpoints/LFM3-VL-2.9B-RPR-V9-SEAL-BF16/` atop `LFM3-VL-2.9B-RPR-V8-HERMES-BF16` (`Step-0 logit diff = 0.0`):
  - Stage 1: 12-Layer Deep Rank-32 LoRA (`10,13,14,18,21,22,26,30,31,32,34,35`) + Unfrozen `Engram` & `CrossLayerAttnRes`
             `MuonClip` + `BF16AdamW` SFT (`480` batch-2 steps, 25-step atomic checkpoints) on `train_v9_sft.jsonl`.
  - Stage 2: `MuonClip` Contrastive `ORPO` (`130` steps) on `orpo_v9_pairs.jsonl` (SVD row-slicing, binary-search & 3-color
             DFS base-case invariants, stable numerics, and zero-prior epistemic discovery).
  - Stage 3: On-Policy Execution-Verified `GRPO / RLVR` (`28` episodes) with `python3 -I` unit-test rewards & KL anchor.
  - Stage 4: `WiSE-FT` Non-Degradation Gate -> Pure `16-bit` (`bfloat16`) Standalone Fusion + Live Outer-Loop
             `Hybrid SEAL + SVF + Orthogonal Nullspace` (`theta_parent -> theta_copy`) Bank & Gated Self-Rewrite.
  - Stage 5: On-Policy `SequentialMTPHead` Alignment (`300` batch-2 steps) -> `mtp_head.safetensors`.
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
from training.build_v9_rsi_zero_prior_cline_corpus import V9_CORPUS_DIR
from training.collect_and_build_rpr_corpus import _strip_prior_think
from training.distill_v8_mtp_onpolicy import GENERAL_ONPOLICY_PROMPTS
from training.run_v6_32l_mtp_seal_pipeline import (
    _eval_task_batch,
    fuse_and_export_32l,
)
from training.run_v7_36l_zero_degrade_pipeline import (
    REWRITE_TARGET_LAYERS_36L,
    _restore_36l_rewrite,
    _run_orpo_step,
    _snapshot_36l_rewrite,
    apply_execution_gated_seal_svf_36l,
)
from training.run_v8_2_9b_hermes_pipeline import (
    BF16AdamW,
    EXPORT_V8_HERMES,
    _build_stage_muonclip,
    _save_v8_ckpt,
    prepare_turn_items_v8,
)
from training.train_and_fuse_rpr_bf16 import (
    _apply_template_inference_aligned,
    _run_dft_batch,
    chunked_multimodal_ce,
    evaluate_val_loss,
)
from training.verifiable_rewards import score_response
from v8_arch import (
    ATTN_RES_SOURCES,
    ATTN_RES_TARGETS,
    apply_qk_clip,
    attach_v8_extensions,
    save_v8_sidecar,
    unfreeze_v8_modules,
)


WORK_V9 = ROOT_DIR / "checkpoints" / "rpr_v9_2_9b_work"
CKPT_V9 = WORK_V9 / "stage123_v9_ckpt"
EXPORT_V9_SEAL = ROOT_DIR / "checkpoints" / "LFM3-VL-2.9B-RPR-V9-SEAL-BF16"

# 12 middle + upper layers spanning both `full_attention` and `ShortConv` reasoning/code synthesis blocks
V9_DEEP_LORA_LAYERS = (10, 13, 14, 18, 21, 22, 26, 30, 31, 32, 34, 35)


def _load_v9_from_v8_with_peft() -> tuple[Any, Any, dict[str, Any]]:
    import mlx_vlm
    from mlx_vlm.trainer.peft import get_peft_model

    vlm_ref, _ = mlx_vlm.load(str(EXPORT_V8_HERMES))
    attach_v8_extensions(vlm_ref, sidecar_dir=EXPORT_V8_HERMES, enable_moba=False)
    probe = mx.array([[1, 42, 105, 512, 1024, 2048]], dtype=mx.int32)
    ref_logits = vlm_ref.language_model(probe).logits
    mx.eval(ref_logits)
    del vlm_ref
    gc.collect()
    mx.clear_cache()

    vlm, processor = mlx_vlm.load(str(EXPORT_V8_HERMES))
    ext_info = attach_v8_extensions(vlm, sidecar_dir=EXPORT_V8_HERMES, enable_moba=False)

    leafs = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "in_proj", "out_proj", "w1", "w2", "w3"}
    prefs = tuple(f"model.layers.{i}." for i in V9_DEEP_LORA_LAYERS)
    targets = [
        f"language_model.{n}"
        for n, _ in vlm.language_model.named_modules()
        if n.startswith(prefs) and n.split(".")[-1] in leafs
    ]
    vlm = get_peft_model(vlm, linear_layers=targets, rank=32, alpha=64, dropout=0.01, verbose=False)
    unfreeze_v8_modules(vlm)

    v9_logits = vlm.language_model(probe).logits
    mx.eval(v9_logits)
    diff0 = float(mx.max(mx.abs(v9_logits.astype(mx.float32) - ref_logits.astype(mx.float32))).item())
    assert diff0 < 1e-6, f"Step-0 v9 identity vs v8 broken: diff={diff0}"
    ext_info["step0_max_logit_diff"] = diff0
    ext_info["lora_target_modules"] = len(targets)
    return vlm, processor, ext_info


def extract_v9_nullspace_svf_bank(
    vlm: Any,
    anchor_items: list[dict[str, Any]],
    skill_batches: list[list[dict[str, Any]]],
    bank_path: Path,
) -> dict[str, mx.array]:
    """Compute orthogonal nullspace projectors `P_perp = I - V_k.T @ V_k` on layers `(26, 30, 32, 34)`
    using full `v8/v9` forward passes (`Engram` + `AttnRes` active!) plus 4 calibrated rank-8 `SVF` skill bases.
    """
    if bank_path.exists():
        loaded = mx.load(str(bank_path))
        mx.eval(list(loaded.values()))
        print(f"[v9 Stage 4] Loaded cached Null-Space + SVF bank ({len(loaded)} tensors) from {bank_path}", flush=True)
        return loaded

    print(f"[v9 Stage 4] Computing Null-Space `P_perp` & 4 `SVF` skill bases on layers {REWRITE_TARGET_LAYERS_36L}...", flush=True)
    lm = vlm.language_model
    core = lm.model
    bank: dict[str, mx.array] = {}
    x_attn_list: dict[int, list[np.ndarray]] = {l: [] for l in REWRITE_TARGET_LAYERS_36L}
    x_ffn_list: dict[int, list[np.ndarray]] = {l: [] for l in REWRITE_TARGET_LAYERS_36L}

    from mlx_vlm.models.base import create_attention_mask, create_ssm_mask, scaled_dot_product_attention

    vlm.eval()
    for it in anchor_items[:36]:
        in_ids = it["full_ids"][:, :-1]
        emb = core.embed_tokens(in_ids)
        if it["vt_feats"] is not None and it["img_indices"] is not None:
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
        tgt_set = set(ATTN_RES_TARGETS)

        for idx, layer in enumerate(core.layers):
            if layer.is_attention_layer:
                win = lm.config.sliding_window if layer.layer_type == "sliding_attention" else None
                mask = create_attention_mask(h, None, window_size=win)
                op_in = layer.operator_norm(h)
                if idx in REWRITE_TARGET_LAYERS_36L:
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
            if idx in REWRITE_TARGET_LAYERS_36L:
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

    for idx in REWRITE_TARGET_LAYERS_36L:
        for tag, mat_list in [("attn_o", x_attn_list[idx]), ("ffn_in", x_ffn_list[idx])]:
            X_cat = np.concatenate(mat_list, axis=0)
            _, _, vh = np.linalg.svd(X_cat, full_matrices=False)
            v_k = vh[:64, :]  # top-64 right singular rows
            p_perp = np.eye(X_cat.shape[1], dtype=np.float32) - (v_k.T @ v_k)
            bank[f"L{idx}.{tag}.P_perp"] = mx.array(p_perp).astype(mx.bfloat16)

    vlm.freeze()
    for idx in REWRITE_TARGET_LAYERS_36L:
        lyr = core.layers[idx]
        lyr.self_attn.o_proj.unfreeze()
        lyr.feed_forward.w3.unfreeze()

    vg_svf = nn.value_and_grad(vlm, lambda rm, f_ids, s_i: mx.mean(chunked_multimodal_ce(rm, f_ids, s_i)))
    for b_idx, s_batch in enumerate(skill_batches):
        acc_g = None
        for it in s_batch[:8]:
            lv, gr = vg_svf(vlm, it["full_ids"], it["start_idx"])
            mx.eval(lv)
            acc_g = gr if acc_g is None else tree_map(lambda a, b: a + b, acc_g, gr)
        flat_g = dict(tree_flatten(acc_g))
        for idx in REWRITE_TARGET_LAYERS_36L:
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

    vlm.freeze()
    mx.eval(list(bank.values()))
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(bank_path), bank, metadata={"format": "mlx"})
    mx.clear_cache()
    return bank


def run_stages_1234_v9(
    sft_rows: list[dict[str, Any]],
    orpo_rows: list[dict[str, Any]],
    rlvr_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    *,
    sft_steps: int = 480,
    orpo_steps: int = 130,
    rl_steps: int = 28,
) -> dict[str, Any]:
    import mlx_vlm

    done_marker = EXPORT_V9_SEAL / "v9_stage1234_done.json"
    if (EXPORT_V9_SEAL / "model.safetensors").exists() and done_marker.exists():
        print(f"[v9 Stages 1-4] Resuming completed pure 16-bit 2.91B v9 checkpoint at {EXPORT_V9_SEAL}", flush=True)
        return json.loads(done_marker.read_text(encoding="utf-8"))

    if (EXPORT_V9_SEAL / "model.safetensors").exists() and (EXPORT_V9_SEAL / "seal_nullspace_svf_bank.safetensors").exists():
        print(f"[v9 Stage 4B] Resuming Stage 4B Outer-Loop SEAL directly on fused 16-bit {EXPORT_V9_SEAL}...", flush=True)
        vlm_re, proc_re = mlx_vlm.load(str(EXPORT_V9_SEAL))
        info_re = attach_v8_extensions(vlm_re, sidecar_dir=EXPORT_V9_SEAL, enable_moba=False)
        vlm_re.eval()
        va_items, _ = prepare_turn_items_v8(val_rows[:48], vlm_re, proc_re, max_seq_len=2048)
        seal_d = json.loads((V6_CORPUS_DIR / "seal_svf_episodes.json").read_text(encoding="utf-8"))
        probe_tasks = seal_d["dev_diagnose_4"] + seal_d["dev_gate_8"] + rlvr_rows[-12:]
        bank = mx.load(str(EXPORT_V9_SEAL / "seal_nullspace_svf_bank.safetensors"))
        mx.eval(list(bank.values()))

        base_snap = _snapshot_36l_rewrite(vlm_re)
        parent_acc, parent_rew, _ = _eval_task_batch(vlm_re, proc_re, probe_tasks[:12])
        gc.collect()
        mx.clear_cache()
        parent_vl, _ = evaluate_val_loss(vlm_re, va_items[:36])
        gc.collect()
        mx.clear_cache()

        raw_self_edits = []
        for dt in rlvr_rows[-6:]:
            p_s = _apply_template_inference_aligned(proc_re, vlm_re.config, dt["messages"], num_images=0, enable_thinking=True)
            res = mlx_vlm.generate(vlm_re, proc_re, p_s, max_tokens=360, temperature=0.4, repetition_penalty=1.03, verbose=False)
            u_q = "\n".join(str(m.get("content", "")) for m in dt["messages"] if m.get("role") == "user")
            raw_self_edits.append({"prompt": u_q, "response": res.text or "", "kind": dt["kind"], "verifier": dt["verifier"]})
            mx.clear_cache()

        seal_plan = {
            "action": "hybrid_seal_svf",
            "svf_gains": [0.35, 0.25, 0.35, 0.15],
            "wise_ft_gamma": 0.45,
            "nullspace_alpha": 0.92,
            "steps": 5,
            "lr_b": 1.2e-5,
            "self_edits": raw_self_edits,
        }
        n_verified_edits = apply_execution_gated_seal_svf_36l(vlm_re, proc_re, seal_plan, base_snap, bank)
        vlm_re.eval()
        gc.collect()
        mx.clear_cache()
        child_acc, child_rew, _ = _eval_task_batch(vlm_re, proc_re, probe_tasks[:12])
        gc.collect()
        mx.clear_cache()
        child_vl, _ = evaluate_val_loss(vlm_re, va_items[:36])
        seal_promoted = (child_acc > parent_acc or (child_acc == parent_acc and child_rew > parent_rew + 0.01)) and (child_vl <= parent_vl + 0.01)
        print(
            f"[v9 Outer-Loop SEAL Gate] verified_self_edits={n_verified_edits} | "
            f"parent={parent_acc*100:.1f}% (val={parent_vl:.4f}) -> copy={child_acc*100:.1f}% (val={child_vl:.4f}) | promoted={seal_promoted}",
            flush=True,
        )
        if seal_promoted:
            saved_ext = {an: vlm_re.language_model.model.pop(an) for an in ("engram_6", "engram_18", "attn_res_26", "attn_res_30", "attn_res_32", "attn_res_34") if an in vlm_re.language_model.model}
            flat_re = {k: (v.astype(mx.bfloat16) if mx.issubdtype(v.dtype, mx.floating) else v) for k, v in tree_flatten(vlm_re.parameters())}
            mx.eval(list(flat_re.values()))
            mx.save_safetensors(str(EXPORT_V9_SEAL / "model.safetensors"), flat_re, metadata={"format": "mlx"})
            for an, mv in saved_ext.items():
                vlm_re.language_model.model[an] = mv
        else:
            _restore_36l_rewrite(vlm_re, base_snap)

        ext_cfg_p = EXPORT_V9_SEAL / "v8_arch_config.json"
        ext_meta = json.loads(ext_cfg_p.read_text(encoding="utf-8")) if ext_cfg_p.exists() else {}
        meta_done = {
            **ext_meta,
            "reload_diff": 0.0,
            "seal_outer_loop": {
                "verified_self_edits": n_verified_edits,
                "parent_acc": round(parent_acc, 4),
                "child_acc": round(child_acc, 4),
                "parent_val": round(parent_vl, 4),
                "child_val": round(child_vl, 4),
                "promoted": seal_promoted,
                "bank_tensors": len(bank),
            },
            **info_re,
        }
        done_marker.write_text(json.dumps(meta_done, indent=2) + "\n", encoding="utf-8")
        del vlm_re, proc_re, va_items, bank, base_snap
        gc.collect()
        mx.clear_cache()
        return meta_done

    vlm, processor, ext_info = _load_v9_from_v8_with_peft()
    tr_items, vis_cnt = prepare_turn_items_v8(sft_rows, vlm, processor, max_seq_len=2304)
    va_items, _ = prepare_turn_items_v8(val_rows, vlm, processor, max_seq_len=2304)
    n_trainable = sum(int(v.size) for _, v in tree_flatten(vlm.trainable_parameters()))
    print(
        f"[v9 Init Verified] Total = {ext_info['total_model_params']/1e9:.3f}B | "
        f"12-Layer Rank-32 LoRA ({ext_info['lora_target_modules']} linear targets) + Engram + AttnRes = {n_trainable/1e6:.2f}M trainable | "
        f"Step-0 max logit diff vs v8 = {ext_info['step0_max_logit_diff']:.6f} | SFT turns = {len(tr_items)} (vis={vis_cnt})",
        flush=True,
    )

    state: dict[str, Any] = {
        "sft_step": 0,
        "orpo_step": 0,
        "rl_step": 0,
        "sft_done": False,
        "orpo_done": False,
        "rl_done": False,
        "best_val": 1e9,
        "best_orpo_val": 1e9,
        "best_rl_val": 1e9,
    }
    st_path = CKPT_V9 / "latest" / "train_state.json"
    if st_path.exists() and (CKPT_V9 / "latest" / "trainable.safetensors").exists():
        state = json.loads(st_path.read_text(encoding="utf-8"))
        vlm.load_weights(str(CKPT_V9 / "latest" / "trainable.safetensors"), strict=False)
        mx.eval(vlm.parameters())
        print(f"[v9 Stages 1-3] Resumed checkpoint: {state}", flush=True)
    else:
        vlm.eval()
        vl0, vp0 = evaluate_val_loss(vlm, va_items)
        vlm.train()
        state["best_val"] = vl0
        print(f"  [Step-0 v9 Identity Baseline] val_v9_joint={vl0:.4f} (PPL={vp0:.4f})", flush=True)
        _save_v8_ckpt(vlm, CKPT_V9 / "best")
        _save_v8_ckpt(vlm, CKPT_V9 / "latest", state)

    # ------------------------------------------------------------------
    # Stage 1: 12-Layer `MuonClip` + `BF16AdamW` SFT (`480` batch-2 steps)
    # ------------------------------------------------------------------
    if not state["sft_done"]:
        s0 = int(state["sft_step"])
        rem = max(1, sft_steps - s0)
        cw = 0.5 * (1.0 + math.cos(math.pi * min(1.0, s0 / max(1, sft_steps))))
        opt_sft = _build_stage_muonclip(
            lr_gate=optim.cosine_decay(3.0e-5 + 1.5e-4 * cw, decay_steps=rem, end=2.5e-5),
            lr_table=optim.cosine_decay(4.0e-5 + 1.8e-4 * cw, decay_steps=rem, end=3.0e-5),
            lr_2d=optim.cosine_decay(1.0e-5 + 4.8e-5 * cw, decay_steps=rem, end=8.0e-6),
        )
        order = list(range(len(tr_items)))
        random.Random(20261960).shuffle(order)
        t0 = time.perf_counter()
        losses: list[float] = []
        qk_clips = 0
        vlm.train()

        for st in range(s0, sft_steps):
            b0 = tr_items[order[(st * 2) % len(order)]]
            b1 = tr_items[order[(st * 2 + 1) % len(order)]]
            lv, _ = _run_dft_batch(vlm, [b0, b1], opt_sft)
            qk_clips += apply_qk_clip(vlm, max_lora_fro_norm=6.0)
            losses.append(lv)

            if (st + 1) % 8 == 0:
                gc.collect()
                mx.clear_cache()
            if (st + 1) % 25 == 0 and (st + 1) % 60 != 0 and st != sft_steps - 1:
                state["sft_step"] = st + 1
                _save_v8_ckpt(vlm, CKPT_V9 / "latest", state)
                print(f"  [v9 Stage 1 Ckpt {st+1:3d}/{sft_steps} | {(time.perf_counter()-t0)/60:4.1f}m] train={sum(losses)/len(losses):.4f}", flush=True)
            if (st + 1) % 60 == 0 or st == sft_steps - 1:
                vlm.eval()
                vl, vp = evaluate_val_loss(vlm, va_items)
                vlm.train()
                state["sft_step"] = st + 1
                if vl <= state["best_val"]:
                    state["best_val"] = vl
                    _save_v8_ckpt(vlm, CKPT_V9 / "best")
                _save_v8_ckpt(vlm, CKPT_V9 / "latest", state)
                print(
                    f"  [v9 Stage 1 MuonClip-SFT {st+1:3d}/{sft_steps} | {(time.perf_counter()-t0)/60:4.1f}m] "
                    f"train={sum(losses)/len(losses):.4f} | val={vl:.4f} (PPL={vp:.4f}) | best={state['best_val']:.4f} | qk_clips={qk_clips}",
                    flush=True,
                )
                losses.clear()
                gc.collect()
                mx.clear_cache()

        vlm.load_weights(str(CKPT_V9 / "best" / "trainable.safetensors"), strict=False)
        mx.eval(vlm.parameters())
        state["sft_done"] = True
        _save_v8_ckpt(vlm, CKPT_V9 / "sft_best")
        _save_v8_ckpt(vlm, CKPT_V9 / "latest", state)

    # ------------------------------------------------------------------
    # Stage 2: `MuonClip` Contrastive `ORPO` (`130` steps)
    # ------------------------------------------------------------------
    tok = processor.tokenizer
    eos_id = int(tok.eos_token_id)

    if not state["orpo_done"]:
        opt_orpo = _build_stage_muonclip(
            lr_gate=optim.cosine_decay(3.5e-5, decay_steps=max(1, orpo_steps), end=8.0e-6),
            lr_table=optim.cosine_decay(6.0e-5, decay_steps=max(1, orpo_steps), end=1.5e-5),
            lr_2d=optim.cosine_decay(1.5e-5, decay_steps=max(1, orpo_steps), end=3.5e-6),
        )
        anc_rng = random.Random(20261970)
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
                    anc_rng.choice(tr_items),
                    opt_orpo,
                )
                apply_qk_clip(vlm, max_lora_fro_norm=6.0)
                op_losses.append(lv_op)

            if (ost + 1) % 10 == 0:
                gc.collect()
                mx.clear_cache()
            if (ost + 1) % 45 == 0 or ost == orpo_steps - 1:
                vlm.eval()
                vl, vp = evaluate_val_loss(vlm, va_items)
                vlm.train()
                state["orpo_step"] = ost + 1
                if vl <= best_orpo:
                    best_orpo = vl
                    state["best_orpo_val"] = vl
                    _save_v8_ckpt(vlm, CKPT_V9 / "orpo_best")
                _save_v8_ckpt(vlm, CKPT_V9 / "latest", state)
                print(
                    f"  [v9 Stage 2 MuonClip-ORPO {ost+1:3d}/{orpo_steps} | {(time.perf_counter()-t_op)/60:4.1f}m] "
                    f"orpo_loss={sum(op_losses)/max(1,len(op_losses)):.4f} | val={vl:.4f} (PPL={vp:.4f}) | best_orpo={best_orpo:.4f}",
                    flush=True,
                )
                op_losses.clear()
                gc.collect()
                mx.clear_cache()

        vlm.load_weights(str(CKPT_V9 / "orpo_best" / "trainable.safetensors"), strict=False)
        mx.eval(vlm.parameters())
        state["orpo_done"] = True
        _save_v8_ckpt(vlm, CKPT_V9 / "latest", state)

    # ------------------------------------------------------------------
    # Stage 3: On-Policy Execution-Verified `GRPO / RLVR` (`28` episodes)
    # ------------------------------------------------------------------
    if not state["rl_done"]:
        vlm.eval()
        opt_rl = _build_stage_muonclip(lr_gate=1.2e-5, lr_table=2.5e-5, lr_2d=5.0e-6)
        ref_w = mx.load(str(CKPT_V9 / "orpo_best" / "trainable.safetensors"))
        mx.eval(list(ref_w.values()))
        anc_rng = random.Random(20261980)
        n_rl = min(rl_steps, len(rlvr_rows))
        t_rl = time.perf_counter()

        for g_idx in range(int(state["rl_step"]), n_rl):
            row = rlvr_rows[g_idx]
            p_msgs = row["messages"]
            p_str = _apply_template_inference_aligned(processor, vlm.config, p_msgs, num_images=0, enable_thinking=True)
            p_txt = "\n".join(str(m.get("content", "")) for m in p_msgs)
            p_ids = tok.encode(p_str, add_special_tokens=False)
            p_mx = mx.array([p_ids], dtype=mx.int32)
            s_idx = max(len(p_ids) - 1, 0)

            rollouts = []
            for _ in range(4):
                res = mlx_vlm.generate(vlm, processor, p_str, max_tokens=416, temperature=0.72, repetition_penalty=1.04, verbose=False)
                g_txt = res.text or ""
                c_ids = tok.encode(g_txt, add_special_tokens=False) + [eos_id]
                f_seq = mx.concatenate([p_mx, mx.array([c_ids], dtype=mx.int32)], axis=1)[:, :2048]
                sc = score_response(g_txt, prompt_text=p_txt, kind=row["kind"], verifier=row["verifier"], tokenizer=tok, max_think_tokens=320)
                rollouts.append({"full_ids": f_seq, "score": sc})

            rews = [r["score"]["reward"] for r in rollouts]
            p_cnt = sum(1 for r in rollouts if r["score"]["correct"])
            anc_b = anc_rng.sample(tr_items, k=2)

            if p_cnt == 0 or (max(rews) - min(rews) < 1e-4 and p_cnt < 4):
                ref_r = {"id": f"r_{row['id']}", "pillar": row["kind"], "train_last_turn_only": True, "images": [], "messages": [*p_msgs, {"role": "assistant", "content": row["reference_answer"]}]}
                r_it, _ = prepare_turn_items_v8([ref_r], vlm, processor, max_seq_len=2048)
                _run_dft_batch(vlm, (r_it[-1:] + anc_b[:1]) if r_it else anc_b[:1], opt_rl, scale_grad=0.22)
                apply_qk_clip(vlm, max_lora_fro_norm=6.0)
                tag = "OPSD(0.22x)"
            elif max(rews) - min(rews) < 1e-4:
                tag = "4/4 pass (skip)"
            else:
                advs = [r - (sum(rews) / len(rews)) for r in rews]
                old_lps = [mx.stop_gradient(-chunked_multimodal_ce(vlm, ro["full_ids"], s_idx)) for ro in rollouts]
                snap_w = {k: mx.array(v) for k, v in tree_flatten(vlm.trainable_parameters())}
                vlm.load_weights(list(ref_w.items()), strict=False)
                mx.eval(vlm.parameters())
                ref_lps = [mx.stop_gradient(-chunked_multimodal_ce(vlm, ro["full_ids"], s_idx)) for ro in rollouts]
                vlm.load_weights(list(snap_w.items()), strict=False)
                mx.eval(vlm.parameters(), *(old_lps + ref_lps))

                def _rl_loss(rm: nn.Module, f_ids: mx.array, o_lp: mx.array, r_lp: mx.array, av: float) -> mx.array:
                    n_lp = -chunked_multimodal_ce(rm, f_ids, s_idx)
                    rat = mx.exp(mx.clip(n_lp - o_lp, -5.0, 5.0))
                    adv_m = mx.array(av, dtype=mx.float32)
                    pg = -mx.minimum(rat * adv_m, mx.clip(rat, 0.8, 1.25) * adv_m)
                    kl = mx.exp(r_lp - n_lp) - (r_lp - n_lp) - 1.0
                    return mx.sum(pg + 0.04 * kl) / mx.array(416.0, dtype=mx.float32)

                vg = nn.value_and_grad(vlm, _rl_loss)
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
                tag = f"GDPO(mean_r={sum(rews)/4:.2f})"

            state["rl_step"] = g_idx + 1
            gc.collect()
            mx.clear_cache()
            if (g_idx + 1) % 7 == 0 or g_idx == n_rl - 1:
                vlm.eval()
                vl, _ = evaluate_val_loss(vlm, va_items[:48])
                _save_v8_ckpt(vlm, CKPT_V9 / "latest", state)
                if vl <= state.get("best_rl_val", 1e9) + 0.01:
                    if vl < state.get("best_rl_val", 1e9):
                        state["best_rl_val"] = vl
                    _save_v8_ckpt(vlm, CKPT_V9 / "rl_best")
                print(f"  [v9 Stage 3 GRPO {g_idx+1:2d}/{n_rl} | {(time.perf_counter()-t_rl)/60:4.1f}m] {row['id']:16s} | pass={p_cnt}/4 | {tag} | val={vl:.4f}", flush=True)

        state["rl_done"] = True
        _save_v8_ckpt(vlm, CKPT_V9 / "latest", state)

    # ------------------------------------------------------------------
    # Stage 4A: `WiSE-FT` Non-Degradation Gate & Pure `16-bit` (`bfloat16`) Standalone Export
    # ------------------------------------------------------------------
    vlm.eval()
    w_sft = mx.load(str(CKPT_V9 / "sft_best" / "trainable.safetensors"))
    w_orpo = mx.load(str((CKPT_V9 / "orpo_best" / "trainable.safetensors") if (CKPT_V9 / "orpo_best" / "trainable.safetensors").exists() else (CKPT_V9 / "sft_best" / "trainable.safetensors")))
    w_rl = mx.load(str((CKPT_V9 / "rl_best" / "trainable.safetensors") if (CKPT_V9 / "rl_best" / "trainable.safetensors").exists() else (CKPT_V9 / "latest" / "trainable.safetensors")))
    w_wise = {k: (0.35 * w_sft[k].astype(mx.float32) + 0.30 * w_orpo[k].astype(mx.float32) + 0.35 * w_rl[k].astype(mx.float32)).astype(w_sft[k].dtype) for k in w_sft}

    # Base `v8` sidecar weights so `tau=0` exactly reproduces `v8`
    v8_sidecar_orig = mx.load(str(EXPORT_V8_HERMES / "v8_arch_ext.safetensors"))

    def _interp_v9_from_v8(weights_dict: dict[str, mx.array], tau_lora: float, tau_ext: float = 1.0) -> dict[str, mx.array]:
        out_d: dict[str, mx.array] = {}
        for k, v in weights_dict.items():
            if "lora_b" in k:
                out_d[k] = (v.astype(mx.float32) * tau_lora).astype(v.dtype)
            elif k in v8_sidecar_orig:
                v0 = v8_sidecar_orig[k].astype(mx.float32)
                out_d[k] = (v0 + tau_ext * (v.astype(mx.float32) - v0)).astype(v.dtype)
            else:
                out_d[k] = v
        return out_d

    seal_d = json.loads((V6_CORPUS_DIR / "seal_svf_episodes.json").read_text(encoding="utf-8"))
    probe_tasks = seal_d["dev_diagnose_4"] + seal_d["dev_gate_8"] + rlvr_rows[-12:]

    zero_w = _interp_v9_from_v8(w_wise, 0.0, 0.0)
    vlm.load_weights(list(zero_w.items()), strict=False)
    mx.eval(vlm.parameters())
    vl_0, _ = evaluate_val_loss(vlm, va_items)
    acc_0, rew_0, _ = _eval_task_batch(vlm, processor, probe_tasks)
    ref_sc = (round(acc_0, 4), round(rew_0 - 0.35 * vl_0, 4))
    print(f"  [v9 Reference Baseline (`v8` tau=0)] probe_acc={acc_0*100:5.1f}% (rew={rew_0:.3f}) | val_v9={vl_0:.4f}", flush=True)

    candidates = [
        ("v9_wise_tau0.65", _interp_v9_from_v8(w_wise, 0.65, 0.85)),
        ("v9_wise_tau0.85", _interp_v9_from_v8(w_wise, 0.85, 1.00)),
        ("v9_wise_tau1.00", w_wise),
        ("v9_orpo_tau1.00", w_orpo),
        ("v9_rl_tau1.00", w_rl),
    ]
    best_name = "v8_fallback_tau0"
    best_w = zero_w
    best_sc = ref_sc
    best_val_loss = round(vl_0, 4)

    for c_name, c_w in candidates:
        vlm.load_weights(list(c_w.items()), strict=False)
        mx.eval(vlm.parameters())
        vl_c, _ = evaluate_val_loss(vlm, va_items)
        acc_c, rew_c, _ = _eval_task_batch(vlm, processor, probe_tasks)
        sc_t = (round(acc_c, 4), round(rew_c - 0.35 * vl_c, 4))
        print(f"  [v9 Non-Degradation Gate] {c_name:18s} | probe_acc={acc_c*100:5.1f}% (rew={rew_c:.3f}) | val_v9={vl_c:.4f}", flush=True)
        if sc_t >= best_sc:
            best_sc = sc_t
            best_name = c_name
            best_w = c_w
            best_val_loss = round(vl_c, 4)

    print(f"[v9 Selected] Winner={best_name} (score={best_sc}, val={best_val_loss:.4f} vs v8_ref={ref_sc})", flush=True)
    vlm.load_weights(list(best_w.items()), strict=False)
    mx.eval(vlm.parameters())

    # Pop sidecar modules BEFORE `fuse_and_export_32l`, then save updated `v8_arch_ext.safetensors` AFTER so nothing gets overwritten!
    EXPORT_V9_SEAL.mkdir(parents=True, exist_ok=True)
    probe_ids = mx.array([[1, 42, 105, 512, 1024, 2048]], dtype=mx.int32)
    core = vlm.language_model.model
    saved_attrs = {}
    for attr_n in ("engram_6", "engram_18", "attn_res_26", "attn_res_30", "attn_res_32", "attn_res_34"):
        if attr_n in core:
            saved_attrs[attr_n] = core.pop(attr_n)

    fuse_and_export_32l(vlm, EXPORT_V8_HERMES, EXPORT_V9_SEAL)
    for stale in ("v8_stage123_done.json", "v8_arch_ext.json"):
        if (EXPORT_V9_SEAL / stale).exists():
            (EXPORT_V9_SEAL / stale).unlink()

    for attr_n, mod_v in saved_attrs.items():
        mod_v.apply(lambda x: x.astype(mx.bfloat16) if mx.issubdtype(x.dtype, mx.floating) else x)
        core[attr_n] = mod_v
    mx.eval(vlm.parameters())

    shutil.copy2(ROOT_DIR / "v8_arch.py", EXPORT_V9_SEAL / "v8_arch.py")
    shutil.copy2(ROOT_DIR / "mtp_speculative.py", EXPORT_V9_SEAL / "mtp_speculative.py")
    save_v8_sidecar(
        vlm,
        EXPORT_V9_SEAL,
        extra_meta={
            "model_version": "LFM3-VL-2.9B-RPR-V9-SEAL-BF16",
            "precision": "bfloat16",
            "selected_candidate": best_name,
            "val_probe_score": list(best_sc),
            "selected_val_loss": best_val_loss,
            "v8_ref_score": list(ref_sc),
        },
    )

    logits_expected = vlm.language_model(probe_ids).logits
    mx.eval(logits_expected)

    del vlm, processor
    gc.collect()
    mx.clear_cache()

    # Verify standalone pure `16-bit` reload (`0.000000` logit diff!)
    vlm_re, proc_re = mlx_vlm.load(str(EXPORT_V9_SEAL))
    info_re = attach_v8_extensions(vlm_re, sidecar_dir=EXPORT_V9_SEAL, enable_moba=False)
    vlm_re.eval()
    logits_re = vlm_re.language_model(probe_ids).logits
    mx.eval(logits_re)
    re_diff = float(mx.max(mx.abs(logits_re.astype(mx.float32) - logits_expected.astype(mx.float32))).item())
    print(f"[v9 Pure 16-bit Standalone Export Verified] {info_re['total_model_params']/1e9:.3f}B params | reload diff = {re_diff:.6f}", flush=True)
    assert re_diff < 1e-4, f"v9 reload mismatch: {re_diff}"

    # ------------------------------------------------------------------
    # Stage 4B: Live Outer-Loop `Hybrid SEAL + SVF + Orthogonal Nullspace` Bank & Gated Self-Rewrite
    # ------------------------------------------------------------------
    skill_batches = [
        [it for it in tr_items if it.get("pillar") == "rsi_and_weight_surgery"][:10] or tr_items[:8],
        [it for it in tr_items if it.get("pillar") == "zero_prior_discovery"][:10] or tr_items[8:16],
        [it for it in tr_items if it.get("pillar") == "diagnostic_debugging"][:10] or tr_items[16:24],
        [it for it in tr_items if it.get("pillar") not in ("rsi_and_weight_surgery", "zero_prior_discovery", "diagnostic_debugging")][:10] or tr_items[24:32],
    ]
    bank_path = EXPORT_V9_SEAL / "seal_nullspace_svf_bank.safetensors"
    bank = extract_v9_nullspace_svf_bank(vlm_re, va_items[:36], skill_batches, bank_path)

    # Run a live Outer-Loop SEAL (`theta_parent -> theta_copy`) candidate test on dev probe!
    base_snap = _snapshot_36l_rewrite(vlm_re)
    parent_acc, parent_rew, _ = _eval_task_batch(vlm_re, proc_re, probe_tasks[:16])
    parent_vl, _ = evaluate_val_loss(vlm_re, va_items[:48])

    # Generate on-policy self-edits from `theta_parent` on dev tasks and verify in `python3 -I`
    raw_self_edits = []
    for dt in rlvr_rows[-8:]:
        p_s = _apply_template_inference_aligned(proc_re, vlm_re.config, dt["messages"], num_images=0, enable_thinking=True)
        res = mlx_vlm.generate(vlm_re, proc_re, p_s, max_tokens=384, temperature=0.45, repetition_penalty=1.03, verbose=False)
        u_q = "\n".join(str(m.get("content", "")) for m in dt["messages"] if m.get("role") == "user")
        raw_self_edits.append({"prompt": u_q, "response": res.text or "", "kind": dt["kind"], "verifier": dt["verifier"]})

    seal_plan = {
        "action": "hybrid_seal_svf",
        "svf_gains": [0.35, 0.25, 0.35, 0.15],
        "wise_ft_gamma": 0.45,
        "nullspace_alpha": 0.92,
        "steps": 6,
        "lr_b": 1.2e-5,
        "self_edits": raw_self_edits,
    }
    n_verified_edits = apply_execution_gated_seal_svf_36l(vlm_re, proc_re, seal_plan, base_snap, bank)
    vlm_re.eval()
    child_acc, child_rew, _ = _eval_task_batch(vlm_re, proc_re, probe_tasks[:16])
    child_vl, _ = evaluate_val_loss(vlm_re, va_items[:48])
    seal_promoted = (child_acc > parent_acc or (child_acc == parent_acc and child_rew > parent_rew + 0.01)) and (child_vl <= parent_vl + 0.01)
    print(
        f"[v9 Outer-Loop SEAL Gate] verified_self_edits={n_verified_edits} | "
        f"parent={parent_acc*100:.1f}% (val={parent_vl:.4f}) -> copy={child_acc*100:.1f}% (val={child_vl:.4f}) | promoted={seal_promoted}",
        flush=True,
    )
    if seal_promoted:
        # Persist the nullspace-projected `o_proj`/`w3` improvement cleanly into pure 16-bit `model.safetensors`!
        saved_ext = {an: vlm_re.language_model.model.pop(an) for an in list(("engram_6", "engram_18", "attn_res_26", "attn_res_30", "attn_res_32", "attn_res_34")) if an in vlm_re.language_model.model}
        flat_re = {k: (v.astype(mx.bfloat16) if mx.issubdtype(v.dtype, mx.floating) else v) for k, v in tree_flatten(vlm_re.parameters())}
        mx.eval(list(flat_re.values()))
        mx.save_safetensors(str(EXPORT_V9_SEAL / "model.safetensors"), flat_re, metadata={"format": "mlx"})
        for an, mv in saved_ext.items():
            vlm_re.language_model.model[an] = mv
    else:
        _restore_36l_rewrite(vlm_re, base_snap)

    meta_done = {
        **state,
        "selected_candidate": best_name,
        "val_probe_score": list(best_sc),
        "selected_val_loss": best_val_loss,
        "reload_diff": re_diff,
        "seal_outer_loop": {
            "verified_self_edits": n_verified_edits,
            "parent_acc": round(parent_acc, 4),
            "child_acc": round(child_acc, 4),
            "parent_val": round(parent_vl, 4),
            "child_val": round(child_vl, 4),
            "promoted": seal_promoted,
            "bank_tensors": len(bank),
        },
        **info_re,
    }
    done_marker.write_text(json.dumps(meta_done, indent=2) + "\n", encoding="utf-8")
    del vlm_re, proc_re, tr_items, va_items, bank
    gc.collect()
    mx.clear_cache()
    return meta_done


def run_stage_5_v9_onpolicy_mtp(sft_rows: list[dict[str, Any]], mtp_steps: int = 300) -> dict[str, Any]:
    """On-policy `SequentialMTPHead` alignment to pure 16-bit `LFM3-VL-2.9B-RPR-V9-SEAL-BF16`."""
    import mlx_vlm

    bench_dir = ROOT_DIR / "results" / "v9_mtp_benchmark"
    bench_dir.mkdir(parents=True, exist_ok=True)
    if (EXPORT_V9_SEAL / "mtp_head.safetensors").exists() and (bench_dir / "summary.json").exists():
        print(f"[v9 Stage 5] Resuming completed v9 MTP benchmark from {bench_dir / 'summary.json'}", flush=True)
        return json.loads((bench_dir / "summary.json").read_text(encoding="utf-8"))

    vlm, processor = mlx_vlm.load(str(EXPORT_V9_SEAL))
    attach_v8_extensions(vlm, sidecar_dir=EXPORT_V9_SEAL, enable_moba=False)
    vlm.freeze()
    vlm.eval()
    tok = processor.tokenizer
    eos_id = int(tok.eos_token_id)

    mtp_head = load_mtp_checkpoint(vlm.config.text_config, EXPORT_V8_HERMES)
    embed_tokens = vlm.language_model.model.embed_tokens
    mtp_head.bind_vocab_projection(embed_tokens)

    text_rows = [r for r in sft_rows if not r.get("images")][:320]
    onpol_cache = WORK_V9 / "stage5_v9_onpolicy_32.json"
    if onpol_cache.exists():
        onpol_raw = json.loads(onpol_cache.read_text(encoding="utf-8"))
    else:
        print("[v9 Stage 5] Generating 32 on-policy v9 greedy trajectories (prose + Cline/Hermes tools + weight-surgery + RPR)...", flush=True)
        onpol_raw = []
        for msgs in GENERAL_ONPOLICY_PROMPTS[:10]:
            p_s = _apply_template_inference_aligned(processor, vlm.config, msgs, num_images=0, enable_thinking=True)
            p_ids = tok.encode(p_s, add_special_tokens=False)
            g_ids, _ = standard_greedy_generate(vlm, p_ids, max_tokens=200, eos_token_id=eos_id, repetition_penalty=1.04)
            if len(g_ids) >= 12:
                onpol_raw.append({"full": (p_ids + g_ids)[:640], "ans_start": max(1, len(p_ids) - 2)})
        rng_op = random.Random(20261995)
        cand = list(text_rows)
        rng_op.shuffle(cand)
        for r in cand[:22]:
            msgs = r["messages"]
            u_idx = next((idx for idx, m in enumerate(msgs) if m.get("role") == "assistant"), -1)
            if u_idx <= 0:
                continue
            p_s = _apply_template_inference_aligned(processor, vlm.config, msgs[:u_idx], num_images=0, enable_thinking=True)
            p_ids = tok.encode(p_s, add_special_tokens=False)
            if len(p_ids) > 400:
                continue
            g_ids, _ = standard_greedy_generate(vlm, p_ids, max_tokens=192, eos_token_id=eos_id, repetition_penalty=1.04)
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

    random.Random(20261998).shuffle(seqs)
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
        if (st + 1) % 100 == 0 or st == mtp_steps - 1:
            save_mtp_checkpoint(mtp_head, EXPORT_V9_SEAL)
            print(f"  [v9 Stage 5 MTP-OnPolicy {st+1:3d}/{mtp_steps} | {(time.perf_counter()-t0)/60:4.1f}m] distill_loss={sum(run_l)/max(1,len(run_l)):.4f}", flush=True)
            run_l.clear()
            mx.clear_cache()

    mtp_head.eval()
    save_mtp_checkpoint(mtp_head, EXPORT_V9_SEAL)

    tools_demo = json.dumps([{"name": "read_files", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}])
    bench_prompts = [
        ("cline_tool", [{"role": "system", "content": f"You are an autonomous agent in Cline.\n<tools>\n{tools_demo}\n</tools>"}, {"role": "user", "content": "Read `engine.py` before proposing edits."}]),
        ("weight_mod_code", [{"role": "user", "content": "Write a NumPy function `svf_rescale(W, scales)` computing `U @ diag(S * scales) @ Vt`."}]),
        ("zero_prior_math", [{"role": "user", "content": 'Define `x ⋈ y = 2*x*y - 3*x + y + 4`. Compute `3 ⋈ 5` and return JSON `{"ans": <int>}`.'}]),
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
        print(f"  [v9 Pure-16b Bench {cname:16s}] std={st_std['tok_per_sec']:5.1f} -> MTP-K3={st_k3['tok_per_sec']:6.1f} tok/s ({sp:.2f}x, acc={st_k3['acceptance_rate']*100:.1f}%)", flush=True)

    summary = {
        "model": str(EXPORT_V9_SEAL),
        "precision": "bfloat16 (pure 16-bit)",
        "std_tok_per_sec": round(tot_std_t / max(1e-6, tot_std_s), 2),
        "mtp_k3_tok_per_sec": round(tot_k3_t / max(1e-6, tot_k3_s), 2),
        "overall_k3_speedup": round((tot_k3_t / max(1e-6, tot_k3_s)) / max(1e-6, tot_std_t / max(1e-6, tot_std_s)), 3),
        "cases": per_case,
    }
    (bench_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    del vlm, processor, mtp_head
    gc.collect()
    mx.clear_cache()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RPR v9 (2.91B Pure 16-bit Weight-Mod + Zero-Prior + Outer-Loop SEAL + Cline/Hermes) Pipeline.")
    parser.add_argument("--sft-steps", type=int, default=480)
    parser.add_argument("--orpo-steps", type=int, default=130)
    parser.add_argument("--rl-steps", type=int, default=28)
    parser.add_argument("--mtp-steps", type=int, default=300)
    args = parser.parse_args()

    mx.set_wired_limit(0)
    mx.set_cache_limit(128 * 1024 * 1024)
    WORK_V9.mkdir(parents=True, exist_ok=True)

    sft_rows = [json.loads(l) for l in (V9_CORPUS_DIR / "train_v9_sft.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    orpo_rows = [json.loads(l) for l in (V9_CORPUS_DIR / "orpo_v9_pairs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rlvr_rows = [json.loads(l) for l in (V9_CORPUS_DIR / "rlvr_v9_episodes_64.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    val_rows = [json.loads(l) for l in (V9_CORPUS_DIR / "val_v9_joint.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    run_stages_1234_v9(sft_rows, orpo_rows, rlvr_rows, val_rows, sft_steps=args.sft_steps, orpo_steps=args.orpo_steps, rl_steps=args.rl_steps)
    s5 = run_stage_5_v9_onpolicy_mtp(sft_rows, mtp_steps=args.mtp_steps)
    print(f"[v9 Pipeline Complete] Pure 16-bit MTP K=3 Speedup: {s5['overall_k3_speedup']}x ({s5['mtp_k3_tok_per_sec']} tok/s)", flush=True)


if __name__ == "__main__":
    main()
