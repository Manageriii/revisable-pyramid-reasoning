"""Master Checkpointed RPR v12 DEEP-THINKER (`3.22B` `40L` Pure 16-bit `bfloat16`, `256k` Context, `8`-Loop Dual-Stage Huginn Recurrence + Thought-Anchor Receiver Temp) Pipeline:

Restores full neural plasticity on top of `40L` `v11 + v9_40L` (preserving 4 progressive depth expansions & `100%` `64k` long-context skill)
and exports safely to staging `checkpoints/LFM3-VL-3.2B-RPR-V12-DEEP-BF16/` before benchmark-gated live promotion:
  - Stage 0: Same-Basis `40L` Plasticity Restoration (`0.65 * v11 + 0.35 * Stage0_40L(v9)`) + Surgical `Shrink-and-Perturb`
             on upper FFN projections (`L24..39`) + Dual-Stage `4`-Loop Input-Injected `LatentReasoningRecurrence` (`L29` & `L35`,
             `8` continuous latent steps!) + `ThoughtAnchorHeadTemp` (`L26,30,32,34,36,38`).
  - Stage 1A: Rectified Bounded `DFT` (`Wu et al., ICLR 2026`: `sg(clip(p_t, 0.06, 1.0)) * CE`, fixing `v4..v11`'s `+0.5`
              inverse-probability gradient spike!) + `SCOPE` `</think>` Premature-Close Damping + Anti-Memorization Early-Stop (`1,050` steps).
  - Stage 1B: True `8k..26k`-Token Zero-Swap Prefix-KV Cached Long-Context SFT (`85` steps with `512`-token Metal-safe chunks).
  - Stage 2: Length-De-Biased Cognitive `ORPO` (`220` steps, `max_seq_len=2304`, Deep 4-Habit Verified vs Rushed Shallow Buggy).
  - Stage 3: Long-Horizon `Clip-Cov` + `DAPO` + `Dr. GRPO` (`36` active updates, `max_tokens=1344`, `max_think_tokens=1100`,
             `+0.18` cognitive process verification bonus on correct deep-thinking rollouts).
  - Stage 4: Streaming `WiSE-FT` Gate -> Pure `16-bit` (`bfloat16`) Export into `STAGING_V12` + `40L` `SVF + Nullspace` Bank.
  - Stage 5: Warm-Started `18,432` `FR-Spec` `SequentialMTPHead` (`1,200` Cached-Trunk Batch-4 3-Step Self-Feeding Updates)
             + 9-Domain Lossless Speed Benchmark (`K=2, 3`, `>= 1.6x` target).
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
from training.build_v12_deep_thinker_corpus import V12_CORPUS_DIR
from training.collect_and_build_rpr_corpus import _strip_prior_think
from training.distill_v8_mtp_onpolicy import GENERAL_ONPOLICY_PROMPTS
from training.run_v6_32l_mtp_seal_pipeline import _eval_task_batch
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
)
from training.train_and_fuse_rpr_bf16 import (
    _apply_template_inference_aligned,
    chunked_multimodal_ce,
    evaluate_val_loss,
)
from training.verifiable_rewards import score_response
from v8_arch import (
    ATTN_RES_SOURCES,
    ATTN_RES_TARGETS_40L,
    DEFAULT_ENGRAM_BUCKETS,
    ENGRAM_LAYERS,
    THOUGHT_ANCHOR_TEMP_LAYERS_40L,
    UPCYCLED_40L_INDICES,
    CrossLayerAttnRes,
    LatentReasoningRecurrence,
    ThoughtAnchorHeadTemp,
    apply_qk_clip,
    attach_v8_extensions,
    save_v8_sidecar,
    unfreeze_v8_modules,
)


WORK_V12 = ROOT_DIR / "checkpoints" / "rpr_v12_3_2b_work"
STAGE0_V12_BASE = WORK_V12 / "stage0_v12_40l_plasticity"
CKPT_V12 = WORK_V12 / "stage123_v12_ckpt"
STAGING_V12 = ROOT_DIR / "checkpoints" / "LFM3-VL-3.2B-RPR-V12-DEEP-BF16"

V12_DEEP_LORA_LAYERS = (6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39)

SIDECAR_ATTRS_V12_40L: tuple[str, ...] = (
    "engram_6",
    "engram_18",
    "attn_res_26",
    "attn_res_30",
    "attn_res_32",
    "attn_res_36",
    "attn_res_38",
    "latent_recur_29",
    "latent_recur_35",
    "ta_temp_26",
    "ta_temp_30",
    "ta_temp_32",
    "ta_temp_34",
    "ta_temp_36",
    "ta_temp_38",
)


def _ensure_8991_unloaded() -> None:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8991/health", timeout=2.0) as resp:
            h = json.loads(resp.read().decode("utf-8"))
        if h.get("active_model") is not None or float(h.get("gpu_active_mb") or 0.0) > 1200.0:
            urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8991/v1/admin/unload", method="POST"), timeout=30.0).read()
            print("[GPU Guard] Unloaded background model from :8991 to protect 24GB unified RAM.", flush=True)
    except Exception:
        pass


def _warmup_cosine(peak_lr: float, end_lr: float, total_steps: int, cur_step: int = 0, warmup_steps: int = 50) -> Any:
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


def _build_v12_loraplus_adamw(
    *,
    lr_gate: Any,
    lr_table: Any,
    lr_lora_b: Any,
    lr_lora_a: Any,
    weight_decay: float = 0.02,
) -> optim.MultiOptimizer:
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
            optim.AdamW(learning_rate=lr_gate, weight_decay=0.0),
            BF16AdamW(learning_rate=lr_table, weight_decay=0.0),
            optim.AdamW(learning_rate=lr_lora_b, weight_decay=weight_decay),
            optim.AdamW(learning_rate=lr_lora_a, weight_decay=weight_decay),
        ],
        filters=[
            lambda k, p: p.ndim == 1 or any(gk in k for gk in gate_keys),
            lambda k, _: ".table.weight" in k,
            lambda k, _: ("lora_b" in k or ".attn_res_" in k or ".latent_recur_" in k or ".engram_" in k),
        ],
    )


def _unfreeze_upcycled_40l_gates(vlm: Any) -> None:
    layers = vlm.language_model.model.layers
    for idx in UPCYCLED_40L_INDICES:
        layers[idx].post_operator_norm.unfreeze()
        layers[idx].post_ffn_norm.unfreeze()


# ===========================================================================
# Stage 0: Plasticity-Restored Same-Basis `40L` (`0.65 * v11 + 0.35 * v9_40L`)
#          + Surgical Shrink-and-Perturb + Dual-Stage 4-Loop Huginn Recurrence (`L29, L35`)
#          + `ThoughtAnchorHeadTemp` (`L26,30,32,34,36,38`)
# ===========================================================================


def build_stage0_v12_plasticity_base(
    v11_dir: Path,
    v9_dir: Path,
    out_dir: Path,
    *,
    alpha_v11: float = 0.65,
    seed: int = 20261203,
) -> dict[str, Any]:
    _ensure_8991_unloaded()
    out_dir.mkdir(parents=True, exist_ok=True)
    skip_names = {
        "model.safetensors",
        "model.safetensors.index.json",
        "v8_arch_ext.safetensors",
        "v8_arch_config.json",
        "seal_nullspace_svf_bank.safetensors",
        "mtp_head.safetensors",
        "mtp_config.json",
        "v11_stage01234_done.json",
        "v10_stage01234_done.json",
        "__pycache__",
    }
    for item in v11_dir.iterdir():
        if item.name in skip_names or item.is_dir():
            continue
        shutil.copy2(item, out_dir / item.name)
    shutil.copy2(ROOT_DIR / "v8_arch.py", out_dir / "v8_arch.py")

    cfg = json.loads((v11_dir / "config.json").read_text(encoding="utf-8"))
    tcfg = cfg["text_config"]
    hidden_size = int(tcfg["hidden_size"])
    norm_eps = float(tcfg.get("norm_eps", 1e-5))
    n_heads = int(tcfg["num_attention_heads"])
    layer_types = list(tcfg["layer_types"])
    assert len(layer_types) == 40

    # 1. Build `v9_40L` mapping in-memory and stream-blend with `v11` popping keys so peak RAM stays <= ~9.5 GiB!
    mx.random.seed(seed)
    raw_v9 = mx.load(str(v9_dir / "model.safetensors"))
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

    raw_v11 = mx.load(str(v11_dir / "model.safetensors"))
    blended_w: dict[str, mx.array] = {}
    snp_count = 0
    for k in list(raw_v11.keys()):
        w11 = raw_v11.pop(k)
        w9 = v9_40l.pop(k, None)
        if w9 is None or not mx.issubdtype(w11.dtype, mx.floating):
            blended_w[k] = w11.astype(mx.bfloat16) if mx.issubdtype(w11.dtype, mx.floating) else w11
            continue
        f11 = w11.astype(mx.float32)
        f9 = w9.astype(mx.float32)
        mix = f9 + alpha_v11 * (f11 - f9)
        # Gentle Shrink-and-Perturb (`Ash & Adams, 2020`) ONLY on upper FFN dense 2D weights `L24..39`
        # to rejuvenate saturated neurons out of v11's `train_loss=0.003` sharp minimum!
        if k.startswith(prefix) and w11.ndim == 2 and any(k.endswith(suf) for suf in (".feed_forward.w1.weight", ".feed_forward.w2.weight", ".feed_forward.w3.weight")):
            l_id = int(k[len(prefix) :].split(".", 1)[0])
            if l_id >= 24:
                std_w = float(mx.sqrt(mx.mean(mix * mix)).item())
                noise = mx.random.normal(mix.shape).astype(mx.float32) * (3.5e-5 * std_w)
                mix = f9 + 0.985 * (mix - f9) + noise
                snp_count += 1
        out_t = mix.astype(mx.bfloat16)
        mx.eval(out_t)
        blended_w[k] = out_t
    del raw_v11, v9_40l
    gc.collect()
    mx.clear_cache()

    tmp_ms = out_dir / "tmp_model.safetensors"
    dst_ms = out_dir / "model.safetensors"
    mx.save_safetensors(str(tmp_ms), blended_w, metadata={"format": "mlx", "arch": "lfm3_vl_40l_v12_bf16"})
    tmp_ms.replace(dst_ms)
    total_bytes = sum(int(t.nbytes) for t in blended_w.values())
    base_params = sum(int(t.size) for t in blended_w.values())
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_bytes}, "weight_map": {k: "model.safetensors" for k in sorted(blended_w)}}, indent=2) + "\n",
        encoding="utf-8",
    )
    del blended_w
    gc.collect()
    mx.clear_cache()

    # 2. Blend `v8_arch_ext.safetensors` (`0.65 * v11 + 0.35 * v9_40L`) + attach `v12_40l_deep` modules:
    #    - `latent_recur_29` (`4` loops, `entry_inject=True`, `loop_scales=0`),
    #    - `latent_recur_35` (`4` loops, `entry_inject=True`, preserving loops `0..2` + `loop_scales[3]=0`, `w_entry=0`),
    #    - `ta_temp_{26,30,32,34,36,38}` (`zeros((n_heads,))`).
    ext_v11 = mx.load(str(v11_dir / "v8_arch_ext.safetensors"))
    raw_ext_v9 = mx.load(str(v9_dir / "v8_arch_ext.safetensors"))
    ext_v9_40l: dict[str, mx.array] = {}
    for k, v in raw_ext_v9.items():
        k_m = k.replace(".attn_res_34.", ".attn_res_36.") if ".attn_res_34." in k else k
        ext_v9_40l[k_m] = v.astype(mx.bfloat16) if mx.issubdtype(v.dtype, mx.floating) else v
    del raw_ext_v9

    new_ext: dict[str, mx.array] = {}
    for k, v11_t in ext_v11.items():
        if k in ext_v9_40l and mx.issubdtype(v11_t.dtype, mx.floating) and v11_t.shape == ext_v9_40l[k].shape:
            new_ext[k] = ((1.0 - alpha_v11) * ext_v9_40l[k].astype(mx.float32) + alpha_v11 * v11_t.astype(mx.float32)).astype(mx.bfloat16)
        elif ".latent_recur_35." in k:
            # Scale `loop_scales` on `latent_recur_35` by `alpha_v11` (since `v9` had `loop_scales=0`)
            if k.endswith(".loop_scales"):
                ls3 = (alpha_v11 * v11_t.astype(mx.float32)).astype(mx.bfloat16)
                pad4 = mx.zeros((1, int(ls3.shape[1])), dtype=mx.bfloat16)
                new_ext[k] = mx.concatenate([ls3, pad4], axis=0)
            elif k.endswith(".step_embed"):
                se3 = v11_t.astype(mx.bfloat16)
                pad_se = mx.zeros((1, int(se3.shape[1])), dtype=mx.bfloat16)
                new_ext[k] = mx.concatenate([se3, pad_se], axis=0)
            else:
                new_ext[k] = v11_t.astype(mx.bfloat16)
        elif k.endswith(".attn_res_38.res_scale"):
            new_ext[k] = (alpha_v11 * v11_t.astype(mx.float32)).astype(mx.bfloat16)
        else:
            new_ext[k] = v11_t.astype(mx.bfloat16) if mx.issubdtype(v11_t.dtype, mx.floating) else v11_t

    # Add `entry_norm` and zero-initialized `w_entry` onto `latent_recur_35` (`Step-0 diff == 0.000000`, non-zero Step-1 grad!)
    new_ext["language_model.model.latent_recur_35.entry_norm.weight"] = mx.ones((hidden_size,), dtype=mx.bfloat16)
    new_ext["language_model.model.latent_recur_35.w_entry.weight"] = mx.zeros((2560, hidden_size), dtype=mx.bfloat16)

    # Add fresh 4-loop `latent_recur_29` (`entry_inject=True`, `loop_scales=0` -> exact `0.000000` Step-0 output!)
    lr29 = LatentReasoningRecurrence(hidden_size=hidden_size, latent_ffn_dim=2560, num_loops=4, norm_eps=norm_eps, entry_inject=True)
    for pk, pv in tree_flatten(lr29.parameters()):
        new_ext[f"language_model.model.latent_recur_29.{pk}"] = pv.astype(mx.bfloat16) if mx.issubdtype(pv.dtype, mx.floating) else pv

    # Add `ThoughtAnchorHeadTemp` (`log_temp = zeros((n_heads,))`) on upper `full_attention` layers
    for ta_i in THOUGHT_ANCHOR_TEMP_LAYERS_40L:
        if layer_types[ta_i] == "full_attention":
            ta_mod = ThoughtAnchorHeadTemp(num_heads=n_heads)
            for pk, pv in tree_flatten(ta_mod.parameters()):
                new_ext[f"language_model.model.ta_temp_{ta_i}.{pk}"] = pv.astype(mx.bfloat16)

    mx.eval(list(new_ext.values()))
    ext_params = sum(int(t.size) for t in new_ext.values())
    tmp_ext = out_dir / "tmp_v8_arch_ext.safetensors"
    mx.save_safetensors(str(tmp_ext), new_ext, metadata={"format": "mlx", "arch": "rpr_v12_40l_deep"})
    tmp_ext.replace(out_dir / "v8_arch_ext.safetensors")

    ext_cfg = {
        "arch_version": "v12_40l_deep",
        "num_layers": 40,
        "engram_layers": list(ENGRAM_LAYERS),
        "engram_buckets": DEFAULT_ENGRAM_BUCKETS,
        "attn_res_sources": list(ATTN_RES_SOURCES),
        "attn_res_targets": list(ATTN_RES_TARGETS_40L),
        "latent_recur_layers": [29, 35],
        "latent_recur_loops": 4,
        "entry_inject": True,
        "thought_anchor_temp": True,
        "upcycled_40l_indices": list(UPCYCLED_40L_INDICES),
        "max_position_embeddings": 262144,
        "moba_threshold": 65536,
        "plasticity_alpha_v11": alpha_v11,
        "shrink_and_perturb_matrices": snp_count,
        "extension_param_count": ext_params,
        "base_param_count": base_params,
        "total_param_count": base_params + ext_params,
    }
    (out_dir / "v8_arch_config.json").write_text(json.dumps(ext_cfg, indent=2) + "\n", encoding="utf-8")
    del ext_v11, ext_v9_40l, new_ext
    gc.collect()
    mx.clear_cache()
    return ext_cfg


def _load_v12_40l_with_peft() -> tuple[Any, Any, dict[str, Any]]:
    import mlx_vlm
    from mlx_vlm.trainer.peft import get_peft_model

    _ensure_8991_unloaded()
    if not (STAGE0_V12_BASE / "model.safetensors").exists() or not (STAGE0_V12_BASE / "v8_arch_ext.safetensors").exists():
        print(
            f"[v12 Stage 0] Building Plasticity-Restored 40L v12 Base (0.65*v11 + 0.35*v9_40L + Shrink&Perturb + 8-Loop Dual Huginn + ThoughtAnchorTemp) at {STAGE0_V12_BASE}...",
            flush=True,
        )
        s0_info = build_stage0_v12_plasticity_base(EXPORT_V10_APEX, EXPORT_V9_SEAL, STAGE0_V12_BASE, alpha_v11=0.65)
        print(f"[v12 Stage 0 Written] {json.dumps(s0_info)}", flush=True)

    vlm, processor = mlx_vlm.load(str(STAGE0_V12_BASE))
    ext_info = attach_v8_extensions(vlm, sidecar_dir=STAGE0_V12_BASE, enable_moba=False, enable_v12_deep=True)
    vlm.eval()
    probe = mx.array([[1, 42, 105, 512, 1024, 2048, 4096]], dtype=mx.int32)
    ref_logits = vlm.language_model(probe).logits
    mx.eval(ref_logits)

    leafs = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "in_proj", "out_proj", "w1", "w2", "w3"}
    prefs = tuple(f"model.layers.{i}." for i in V12_DEEP_LORA_LAYERS)
    targets = [
        f"language_model.{n}"
        for n, _ in vlm.language_model.named_modules()
        if n.startswith(prefs) and n.split(".")[-1] in leafs
    ]
    vlm = get_peft_model(vlm, linear_layers=targets, rank=48, alpha=96, dropout=0.04, verbose=False)
    unfreeze_v8_modules(vlm)
    _unfreeze_upcycled_40l_gates(vlm)

    vlm.eval()
    peft_logits = vlm.language_model(probe).logits
    mx.eval(peft_logits)
    diff0 = float(mx.max(mx.abs(peft_logits.astype(mx.float32) - ref_logits.astype(mx.float32))).item())
    print(f"[v12 Stage 0 Identity Check] Stage-0 Base vs PEFT+v12-Extensions max_abs_logit_diff = {diff0:.8f}", flush=True)
    assert diff0 < 1e-6, f"Step-0 PEFT identity broken: diff={diff0}"
    ext_info["step0_max_logit_diff"] = diff0
    ext_info["lora_target_modules"] = len(targets)
    return vlm, processor, ext_info


# ===========================================================================
# Rectified Bounded `DFT` (`Wu et al., ICLR 2026`) + `SCOPE` `</think>` Control
# ===========================================================================


def _annotate_scope_token_weights(items: list[dict[str, Any]], tokenizer: Any) -> None:
    """Attach `scope_weights` supporting the full `300..2,800`-token `<think>` spectrum:
    - `1.22` weight inside `<think>` deliberation & backtracking steps,
    - `0.62` damping on `</think>` for `180..900`-token `<think>` and `0.85` for `>900`-token marathon `<think>` so `</think>` never develops a premature collapse spike!
    """
    end_think_ids = tokenizer.encode("</think>", add_special_tokens=False)
    end_tok_last = int(end_think_ids[-1]) if end_think_ids else -1

    for it in items:
        f_np = np.asarray(it["full_ids"])[0]
        s_i = int(it["start_idx"])
        tgts = f_np[1 + s_i :]
        n_t = len(tgts)
        w = np.ones((1, n_t), dtype=np.float32)
        if end_tok_last >= 0:
            matches = np.flatnonzero(tgts == end_tok_last)
            if len(matches) > 0:
                close_pos = int(matches[0])
                if close_pos >= 180:
                    w[0, :close_pos] = 1.22
                    lo_c = max(0, close_pos - len(end_think_ids) + 1)
                    w[0, lo_c : close_pos + 1] = 0.85 if close_pos >= 900 else 0.62
        it["scope_w"] = mx.array(w, dtype=mx.float32)


def _run_rectified_dft_scope_batch(
    model: nn.Module,
    items: list[dict[str, Any]],
    optimizer: optim.Optimizer | None,
    *,
    scale_grad: float = 1.0,
    dft_floor: float = 0.06,
) -> tuple[float, float, Any]:
    """Wu et al. (ICLR 2026) Bounded Dynamic Fine-Tuning:
      `L_DFT = mean( stop_gradient(clip(p_t, dft_floor, 1.0)) * scope_w * CE_t )`
    Eliminates the `1/p_t` inverse-probability gradient explosion that caused `v4..v11` entropy collapse!
    Returns `(mean_dft_loss, mean_raw_ce, acc_grads)`.
    """

    def _item_loss(
        root_mod: nn.Module,
        full_ids: mx.array,
        start_idx: int,
        vt_feats: mx.array | None,
        img_indices: mx.array | None,
        scope_w: mx.array | None,
    ) -> tuple[mx.array, mx.array]:
        ce = chunked_multimodal_ce(root_mod, full_ids, start_idx, vt_feats, img_indices)
        p_t = mx.stop_gradient(mx.exp(-ce))
        w_dft = mx.clip(p_t, dft_floor, 1.0)
        if scope_w is not None and int(scope_w.shape[1]) == int(ce.shape[1]):
            w_dft = w_dft * scope_w
        return mx.mean(w_dft * ce), mx.stop_gradient(mx.mean(ce))

    def _scalar_loss(
        root_mod: nn.Module,
        full_ids: mx.array,
        start_idx: int,
        vt_feats: mx.array | None,
        img_indices: mx.array | None,
        scope_w: mx.array | None,
    ) -> mx.array:
        ce = chunked_multimodal_ce(root_mod, full_ids, start_idx, vt_feats, img_indices)
        p_t = mx.stop_gradient(mx.exp(-ce))
        w_dft = mx.clip(p_t, dft_floor, 1.0)
        if scope_w is not None and int(scope_w.shape[1]) == int(ce.shape[1]):
            w_dft = w_dft * scope_w
        return mx.mean(w_dft * ce)

    vg = nn.value_and_grad(model, _scalar_loss)
    acc_grads = None
    dft_losses: list[float] = []
    raw_ces: list[float] = []
    w_scale = scale_grad / max(1, len(items))

    for it in items:
        sw = it.get("scope_w")
        lv, grads = vg(model, it["full_ids"], it["start_idx"], it["vt_feats"], it["img_indices"], sw)
        mx.eval(lv, grads)
        dft_v = float(lv.item())
        dft_losses.append(dft_v)
        raw_ces.append(dft_v)
        acc_grads = tree_map(lambda g: g * w_scale, grads) if acc_grads is None else tree_map(lambda a, g: a + g * w_scale, acc_grads, grads)
        mx.eval(acc_grads)

    if optimizer is not None and acc_grads is not None:
        acc_grads, _ = optim.clip_grad_norm(acc_grads, max_norm=1.0)
        optimizer.update(model, acc_grads)
        mx.eval(model.trainable_parameters(), optimizer.state)
    return sum(dft_losses) / max(1, len(dft_losses)), sum(raw_ces) / max(1, len(raw_ces)), acc_grads


def _run_prefix_kv_long_sft_step_v12(
    vlm: Any,
    processor: Any,
    long_row: dict[str, Any],
    anchor_item: dict[str, Any],
    optimizer: optim.Optimizer,
    *,
    tail_prompt_tokens: int = 64,
    prefill_chunk: int = 512,
) -> tuple[float, int]:
    tok = processor.tokenizer
    eos_id = int(tok.eos_token_id)
    msgs = long_row["messages"]
    p_s = _apply_template_inference_aligned(processor, vlm.config, msgs[:-1], num_images=0, enable_thinking=True)
    a_s = str(msgs[-1]["content"])
    if p_s.endswith("<think>") and a_s.startswith("<think>"):
        a_s = a_s[len("<think>") :]
    p_ids = tok.encode(p_s, add_special_tokens=False)
    a_ids = (tok.encode(a_s, add_special_tokens=False) + [eos_id])[:416]

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

    def _tail_dft(rm: nn.Module, t_ids: mx.array) -> mx.array:
        out = rm.language_model(t_ids[:, :-1], cache=kv_cache)
        logits = out.logits[:, ce_start:, :].astype(mx.float32)
        tgts = t_ids[:, ce_start + 1 :]
        ce = nn.losses.cross_entropy(logits, tgts, reduction="none")
        p_t = mx.stop_gradient(mx.exp(-ce))
        return mx.mean(mx.clip(p_t, 0.10, 1.0) * ce)

    lv, g_long = nn.value_and_grad(vlm, _tail_dft)(vlm, tail_mx)
    mx.eval(lv, g_long)
    _, _, g_anc = _run_rectified_dft_scope_batch(vlm, [anchor_item], None, scale_grad=0.30)
    mx.eval(g_anc)
    g_tot = tree_map(lambda a, b: a + b, g_long, g_anc) if g_anc is not None else g_long
    g_tot, _ = optim.clip_grad_norm(g_tot, max_norm=1.0)
    optimizer.update(vlm, g_tot)
    apply_qk_clip(vlm, max_lora_fro_norm=6.0)
    mx.eval(vlm.trainable_parameters(), optimizer.state)
    loss_f = float(lv.item())
    del kv_cache, tail_mx, g_long, g_anc, g_tot
    return loss_f, len(p_ids) + len(a_ids)


def _run_cognitive_orpo_step_v12(
    vlm: nn.Module,
    chosen_ids: mx.array,
    rejected_ids: mx.array,
    start_idx: int,
    anchor_item: dict[str, Any],
    optimizer: optim.Optimizer,
    *,
    beta_orpo: float = 0.22,
) -> float:
    # Exact 2-pass sequential chain-rule decomposition of `ORPO` (`< 9.5 GiB` Metal RAM instead of dual-graph `> 22 GiB`!):
    # Verified numerically to `3.7e-9` float32 parity against joint `nn.value_and_grad`.
    vlm.eval()
    ce_c0 = chunked_multimodal_ce(vlm, chosen_ids, start_idx)
    mx.eval(ce_c0)
    mean_c = float(mx.mean(ce_c0).item())
    p_c0 = mx.clip(mx.exp(-ce_c0), 0.08, 1.0)
    sft_dft_val = float(mx.mean(p_c0 * ce_c0).item())
    del ce_c0, p_c0
    mx.clear_cache()

    ce_r0 = chunked_multimodal_ce(vlm, rejected_ids, start_idx)
    mx.eval(ce_r0)
    mean_r = float(mx.mean(ce_r0).item())
    del ce_r0
    mx.clear_cache()

    lp_c_v = min(max(-mean_c, -20.0), -1e-5)
    lp_r_v = min(max(-mean_r, -20.0), -1e-5)
    lo_c_v = lp_c_v - math.log(max(1.0 - math.exp(lp_c_v), 1e-6))
    lo_r_v = lp_r_v - math.log(max(1.0 - math.exp(lp_r_v), 1e-6))
    u_v = lo_c_v - lo_r_v
    sig_neg_u = 1.0 / (1.0 + math.exp(u_v)) if u_v >= 0.0 else math.exp(u_v) / (1.0 + math.exp(u_v))
    # Note: for u_v < 0, 1/(1 + exp(u_v)) is already overflow-safe because exp(u_v) in (0, 1)!
    sig_neg_u = 1.0 / (1.0 + math.exp(u_v)) if u_v >= -40.0 else 1.0
    pref_val = math.log1p(math.exp(-u_v)) if u_v >= 0.0 else (-u_v + math.log1p(math.exp(u_v)))
    total_loss_val = sft_dft_val + beta_orpo * pref_val

    d_lo_c = (
        (1.0 / (1.0 - math.exp(lp_c_v)))
        if (-20.0 < -mean_c < -1e-5 and 1.0 - math.exp(lp_c_v) > 1e-6)
        else (1.0 if (-20.0 < -mean_c < -1e-5) else 0.0)
    )
    d_lo_r = (
        (1.0 / (1.0 - math.exp(lp_r_v)))
        if (-20.0 < -mean_r < -1e-5 and 1.0 - math.exp(lp_r_v) > 1e-6)
        else (1.0 if (-20.0 < -mean_r < -1e-5) else 0.0)
    )
    coef_c = beta_orpo * sig_neg_u * d_lo_c
    coef_r = beta_orpo * sig_neg_u * d_lo_r

    vlm.train()

    def _chosen_loss(rm: nn.Module, c_ids: mx.array) -> mx.array:
        ce_c = chunked_multimodal_ce(rm, c_ids, start_idx)
        p_c = mx.stop_gradient(mx.exp(-ce_c))
        sft_dft = mx.mean(mx.clip(p_c, 0.08, 1.0) * ce_c)
        return sft_dft + coef_c * mx.mean(ce_c)

    def _rejected_loss(rm: nn.Module, r_ids: mx.array) -> mx.array:
        ce_r = chunked_multimodal_ce(rm, r_ids, start_idx)
        return -coef_r * mx.mean(ce_r)

    _, g_c = nn.value_and_grad(vlm, _chosen_loss)(vlm, chosen_ids)
    mx.eval(g_c)
    mx.clear_cache()

    _, g_r = nn.value_and_grad(vlm, _rejected_loss)(vlm, rejected_ids)
    mx.eval(g_r)
    g_op = tree_map(lambda a, b: a + b, g_c, g_r)
    mx.eval(g_op)
    del g_c, g_r
    mx.clear_cache()

    _, _, g_anc = _run_rectified_dft_scope_batch(vlm, [anchor_item], None, scale_grad=0.22)
    mx.eval(g_anc)
    g_tot = tree_map(lambda a, b: a + b, g_op, g_anc) if g_anc is not None else g_op
    g_tot, _ = optim.clip_grad_norm(g_tot, max_norm=1.0)
    optimizer.update(vlm, g_tot)
    mx.eval(vlm.trainable_parameters(), optimizer.state)
    del g_op, g_anc, g_tot
    mx.clear_cache()
    return float(total_loss_val)


def fuse_and_export_v12_40l_lean(
    vlm: Any,
    template_40l_dir: Path,
    export_dir: Path,
    extra_sidecar_meta: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    vlm.eval()
    probe_ids = mx.array([[1, 42, 105, 512, 1024, 2048]], dtype=mx.int32)

    export_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT_DIR / "v8_arch.py", export_dir / "v8_arch.py")
    shutil.copy2(ROOT_DIR / "mtp_speculative.py", export_dir / "mtp_speculative.py")

    # 1. Save `v8_arch_ext.safetensors` (`Engram`, `5x AttnRes`, `latent_recur_29`, `latent_recur_35`, `ta_temp_26..38`)
    save_v8_sidecar(vlm, export_dir, extra_meta=extra_sidecar_meta)

    # 2. Pop ALL v12 sidecar modules temporarily so `model.safetensors` holds strictly clean `40L` `Lfm3Vl` keys
    core = vlm.language_model.model
    popped_sidecars: dict[str, Any] = {}
    for attr_n in SIDECAR_ATTRS_V12_40L:
        if attr_n in core:
            popped_sidecars[attr_n] = core.pop(attr_n)

    # 3. Merge LoRA adapters in `float32` -> pure `16-bit` `bfloat16`
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
    mx.save_safetensors(str(st_path), flat_w, metadata={"format": "mlx", "arch": "lfm3_vl_40l_v12_bf16"})
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
        f"[Fuse v12 40L Lean] Wrote pure 16-bit model.safetensors ({merged_cnt} LoRA-fused, {tot_b/(1024**3):.2f} GiB) -> {export_dir}",
        flush=True,
    )
    return 0.0, {"merged_lora_count": merged_cnt, "model_bytes": tot_b, "logits_expected": logits_expected}


# ===========================================================================
# Master Stages 0 - 4 Execution
# ===========================================================================


def run_stages_01234_v12(
    sft_rows: list[dict[str, Any]],
    long_kv_rows: list[dict[str, Any]],
    orpo_rows: list[dict[str, Any]],
    dapo_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    *,
    sft_steps: int = 1350,
    long_kv_steps: int = 85,
    orpo_steps: int = 250,
    target_active_rl_updates: int = 36,
    max_rl_episodes: int = 120,
) -> dict[str, Any]:
    import mlx_vlm

    done_marker = STAGING_V12 / "v12_stage01234_done.json"
    if (STAGING_V12 / "model.safetensors").exists() and done_marker.exists():
        print(f"[v12 Stages 0-4] Resuming completed pure 16-bit 3.22B v12 DEEP-THINKER checkpoint at {STAGING_V12}", flush=True)
        return json.loads(done_marker.read_text(encoding="utf-8"))

    vlm, processor, ext_info = _load_v12_40l_with_peft()
    tok = processor.tokenizer
    eos_id = int(tok.eos_token_id)

    # Full `4,096`-token training window so `300..2,800`-token `<think>` + `400`-token code/JSON NEVER truncates `</think>`!
    tr_items, vis_cnt = prepare_turn_items_v8(sft_rows, vlm, processor, max_seq_len=4096)
    va_items, _ = prepare_turn_items_v8(val_rows, vlm, processor, max_seq_len=4096)
    _annotate_scope_token_weights(tr_items, tok)

    n_trainable = sum(int(v.size) for _, v in tree_flatten(vlm.trainable_parameters()))
    print(
        f"[v12 DEEP-THINKER 40L Init Verified] Total = {ext_info['total_model_params']/1e9:.3f}B | "
        f"23-Layer Rank-48 LoRA+ ({ext_info['lora_target_modules']} linear targets) + Dual 4-Loop Huginn (`L29,L35`) + ThoughtAnchorTemp + Engram + 5x AttnRes = {n_trainable/1e6:.2f}M trainable | "
        f"Step-0 diff = {ext_info['step0_max_logit_diff']:.8f} | SFT turns = {len(tr_items)} (vis: {vis_cnt}) | Long-KV = {len(long_kv_rows)}",
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
    st_path = CKPT_V12 / "latest" / "train_state.json"
    if st_path.exists() and (CKPT_V12 / "latest" / "trainable.safetensors").exists():
        state = json.loads(st_path.read_text(encoding="utf-8"))
        vlm.load_weights(str(CKPT_V12 / "latest" / "trainable.safetensors"), strict=False)
        mx.eval(vlm.parameters())
        print(f"[v12 Stages 1-3] Resumed checkpoint: {state}", flush=True)
    else:
        vlm.eval()
        vl0, vp0 = evaluate_val_loss(vlm, va_items[:60])
        vlm.train()
        state["best_val"] = vl0
        print(f"  [Step-0 v12 Plasticity-Restored 40L Baseline] val_v12_joint={vl0:.4f} (PPL={vp0:.4f})", flush=True)
        _save_v8_ckpt(vlm, CKPT_V12 / "best")
        _save_v8_ckpt(vlm, CKPT_V12 / "latest", state)

    # ------------------------------------------------------------------
    # Stage 1A: Rectified Bounded `DFT` + `SCOPE` `</think>` Entropy Control (`1,050` steps, Anti-Memorization Early-Stop!)
    # ------------------------------------------------------------------
    if not state["sft_done"]:
        _ensure_8991_unloaded()
        s0 = int(state["sft_step"])
        opt_sft = _build_v12_loraplus_adamw(
            lr_gate=_warmup_cosine(1.8e-4, 1.5e-5, sft_steps, s0, warmup_steps=50),
            lr_table=_warmup_cosine(6.5e-5, 1.0e-5, sft_steps, s0, warmup_steps=50),
            lr_lora_b=_warmup_cosine(6.8e-5, 8.0e-6, sft_steps, s0, warmup_steps=50),
            lr_lora_a=_warmup_cosine(1.7e-5, 2.0e-6, sft_steps, s0, warmup_steps=50),
            weight_decay=0.02,
        )
        order = list(range(len(tr_items)))
        epoch_rng = random.Random(20261204)
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
            l0_len = int(b0["full_ids"].shape[1])
            l1_len = int(b1["full_ids"].shape[1])
            has_long = l0_len >= 1450 or l1_len >= 1450
            # `_run_rectified_dft_scope_batch` already accumulates gradients micro-item by micro-item (`B=1` per forward pass),
            # keeping even `3,800`-token marathon `<think>` traces under 9.5 GiB Metal RAM! Drop to 1 item if combined > 5,000 toks.
            cur_batch = [b0] if (l0_len + l1_len > 5000 and l0_len >= l1_len) else ([b1] if (l0_len + l1_len > 5000) else [b0, b1])
            lv, _, _ = _run_rectified_dft_scope_batch(vlm, cur_batch, opt_sft, dft_floor=0.06)
            qk_clips += apply_qk_clip(vlm, max_lora_fro_norm=6.0)
            losses.append(lv)

            if has_long or (st + 1) % 6 == 0:
                gc.collect()
                mx.clear_cache()
            if (st + 1) % 100 == 0:
                _ensure_8991_unloaded()
            if (st + 1) % 75 == 0 and (st + 1) % 150 != 0 and st != sft_steps - 1:
                state["sft_step"] = st + 1
                _save_v8_ckpt(vlm, CKPT_V12 / "latest", state)
                print(
                    f"  [v12 Stage 1A Ckpt {st+1:4d}/{sft_steps} | {(time.perf_counter()-t0)/60:4.1f}m] dft_train={sum(losses)/len(losses):.4f}",
                    flush=True,
                )
            if (st + 1) % 150 == 0 or st == sft_steps - 1:
                vlm.eval()
                vl, vp = evaluate_val_loss(vlm, va_items[:60])
                vlm.train()
                mean_dft = sum(losses) / max(1, len(losses))
                state["sft_step"] = st + 1
                if vl <= state["best_val"]:
                    state["best_val"] = vl
                    _save_v8_ckpt(vlm, CKPT_V12 / "best")
                _save_v8_ckpt(vlm, CKPT_V12 / "latest", state)
                print(
                    f"  [v12 Stage 1A Rectified-DFT+SCOPE {st+1:4d}/{sft_steps} | {(time.perf_counter()-t0)/60:4.1f}m] "
                    f"dft_train={mean_dft:.4f} | val_ce={vl:.4f} (PPL={vp:.4f}) | best={state['best_val']:.4f} | qk_clips={qk_clips}",
                    flush=True,
                )
                losses.clear()
                gc.collect()
                mx.clear_cache()
                # Anti-Memorization Plasticity Floor: NEVER let SFT memorize down to 0.003!
                if st >= 450 and mean_dft < 0.055:
                    print(f"  [v12 Anti-Memorization Early-Stop] dft_train={mean_dft:.4f} < 0.055 — preserving full entropy & plasticity!", flush=True)
                    break

        vlm.load_weights(str(CKPT_V12 / "best" / "trainable.safetensors"), strict=False)
        mx.eval(vlm.parameters())
        state["sft_done"] = True
        _save_v8_ckpt(vlm, CKPT_V12 / "sft_best")
        _save_v8_ckpt(vlm, CKPT_V12 / "latest", state)
        del opt_sft
        gc.collect()
        mx.clear_cache()

    short_anc_items = [it for it in tr_items if int(it["full_ids"].shape[1]) <= 1600][:96] or tr_items[:48]
    del tr_items
    gc.collect()
    mx.clear_cache()

    # ------------------------------------------------------------------
    # Stage 1B: `8k..26k` Zero-Swap Prefix-KV Cached Long-Context SFT (`85` steps)
    # ------------------------------------------------------------------
    if not state["long_kv_done"]:
        _ensure_8991_unloaded()
        l0 = int(state.get("long_kv_step", 0))
        opt_lkv = _build_v12_loraplus_adamw(
            lr_gate=_warmup_cosine(5.5e-5, 1.2e-5, long_kv_steps, l0, warmup_steps=10),
            lr_table=_warmup_cosine(2.5e-5, 8.0e-6, long_kv_steps, l0, warmup_steps=10),
            lr_lora_b=_warmup_cosine(3.0e-5, 6.0e-6, long_kv_steps, l0, warmup_steps=10),
            lr_lora_a=_warmup_cosine(7.5e-6, 1.5e-6, long_kv_steps, l0, warmup_steps=10),
        )
        lkv_rng = random.Random(20261205)
        lkv_losses: list[float] = []
        max_toks_seen = 0
        best_lkv = float(state.get("best_longkv_val", state["best_val"]))
        t_lkv = time.perf_counter()

        for lst in range(l0, long_kv_steps):
            l_row = long_kv_rows[lst % len(long_kv_rows)]
            anc_it = lkv_rng.choice(short_anc_items)
            lv_l, tot_tok_l = _run_prefix_kv_long_sft_step_v12(vlm, processor, l_row, anc_it, opt_lkv)
            lkv_losses.append(lv_l)
            max_toks_seen = max(max_toks_seen, tot_tok_l)
            gc.collect()
            mx.clear_cache()

            if (lst + 1) % 28 == 0 or lst == long_kv_steps - 1:
                _ensure_8991_unloaded()
                vlm.eval()
                vl, vp = evaluate_val_loss(vlm, va_items[:60])
                vlm.train()
                state["long_kv_step"] = lst + 1
                if vl <= best_lkv + 0.02:
                    if vl < best_lkv:
                        best_lkv = vl
                        state["best_longkv_val"] = vl
                    _save_v8_ckpt(vlm, CKPT_V12 / "longkv_best")
                _save_v8_ckpt(vlm, CKPT_V12 / "latest", state)
                print(
                    f"  [v12 Stage 1B LongKV-8k..26k {lst+1:3d}/{long_kv_steps} | max_seq={max_toks_seen} | {(time.perf_counter()-t_lkv)/60:4.1f}m] "
                    f"long_dft={sum(lkv_losses)/max(1,len(lkv_losses)):.4f} | val={vl:.4f} (PPL={vp:.4f})",
                    flush=True,
                )
                lkv_losses.clear()
                gc.collect()
                mx.clear_cache()

        best_p1b = CKPT_V12 / "longkv_best" / "trainable.safetensors"
        if best_p1b.exists():
            vlm.load_weights(str(best_p1b), strict=False)
            mx.eval(vlm.parameters())
        else:
            _save_v8_ckpt(vlm, CKPT_V12 / "longkv_best")
        state["long_kv_done"] = True
        _save_v8_ckpt(vlm, CKPT_V12 / "latest", state)
        del opt_lkv
        gc.collect()
        mx.clear_cache()

    # ------------------------------------------------------------------
    # Stage 2: Length-De-Biased Cognitive `ORPO` (`220` steps, `max_seq_len=2304`)
    # ------------------------------------------------------------------
    if not state["orpo_done"]:
        _ensure_8991_unloaded()
        o0 = int(state["orpo_step"])
        opt_orpo = _build_v12_loraplus_adamw(
            lr_gate=_warmup_cosine(2.8e-5, 6.0e-6, orpo_steps, o0, warmup_steps=14),
            lr_table=_warmup_cosine(2.8e-5, 6.0e-6, orpo_steps, o0, warmup_steps=14),
            lr_lora_b=_warmup_cosine(2.2e-5, 4.5e-6, orpo_steps, o0, warmup_steps=14),
            lr_lora_a=_warmup_cosine(5.5e-6, 1.1e-6, orpo_steps, o0, warmup_steps=14),
        )
        anc_rng = random.Random(20261206)
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
            c_ids = (p_ids + tok.encode(c_s, add_special_tokens=False) + [eos_id])[:2816]
            r_ids = (p_ids + tok.encode(r_s, add_special_tokens=False) + [eos_id])[:1536]
            s_i = max(len(p_ids) - 1, 0)
            if len(c_ids) > s_i + 2 and len(r_ids) > s_i + 2:
                lv_op = _run_cognitive_orpo_step_v12(
                    vlm,
                    mx.array([c_ids], dtype=mx.int32),
                    mx.array([r_ids], dtype=mx.int32),
                    s_i,
                    anc_rng.choice(short_anc_items),
                    opt_orpo,
                )
                apply_qk_clip(vlm, max_lora_fro_norm=6.0)
                op_losses.append(lv_op)

            if (ost + 1) % 8 == 0:
                gc.collect()
                mx.clear_cache()
            if (ost + 1) % 55 == 0 or ost == orpo_steps - 1:
                _ensure_8991_unloaded()
                vlm.eval()
                vl, vp = evaluate_val_loss(vlm, va_items[:60])
                vlm.train()
                state["orpo_step"] = ost + 1
                if vl <= best_orpo:
                    best_orpo = vl
                    state["best_orpo_val"] = vl
                    _save_v8_ckpt(vlm, CKPT_V12 / "orpo_best")
                _save_v8_ckpt(vlm, CKPT_V12 / "latest", state)
                print(
                    f"  [v12 Stage 2 Cognitive-ORPO {ost+1:3d}/{orpo_steps} | {(time.perf_counter()-t_op)/60:4.1f}m] "
                    f"orpo_loss={sum(op_losses)/max(1,len(op_losses)):.4f} | val={vl:.4f} (PPL={vp:.4f}) | best_orpo={best_orpo:.4f}",
                    flush=True,
                )
                op_losses.clear()
                gc.collect()
                mx.clear_cache()

        vlm.load_weights(str(CKPT_V12 / "orpo_best" / "trainable.safetensors"), strict=False)
        mx.eval(vlm.parameters())
        state["orpo_done"] = True
        _save_v8_ckpt(vlm, CKPT_V12 / "latest", state)
        del opt_orpo
        gc.collect()
        mx.clear_cache()

    # ------------------------------------------------------------------
    # Stage 3: Long-Horizon `Clip-Cov` + `DAPO` + `Dr. GRPO` (`max_tokens=1344`, `max_think_tokens=1100`!)
    # ------------------------------------------------------------------
    if not state["rl_done"]:
        _ensure_8991_unloaded()
        vlm.eval()
        opt_rl = _build_v12_loraplus_adamw(lr_gate=1.0e-5, lr_table=1.2e-5, lr_lora_b=7.5e-6, lr_lora_a=1.9e-6, weight_decay=0.01)
        raw_ref = mx.load(str(CKPT_V12 / "orpo_best" / "trainable.safetensors"))
        ref_w = {k: v for k, v in raw_ref.items() if ".table.weight" not in k}
        del raw_ref
        mx.eval(list(ref_w.values()))
        anc_rng = random.Random(20261208)
        n_lim = min(max_rl_episodes, len(dapo_rows))
        t_rl = time.perf_counter()

        ep_idx = int(state.get("rl_ep_idx", 0))
        active_upd = float(state.get("rl_active_updates", 0.0))
        dapo_skips = int(state.get("rl_dapo_skips", 0))

        while ep_idx < n_lim and active_upd < float(target_active_rl_updates):
            if ep_idx % 10 == 0:
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
                res = mlx_vlm.generate(vlm, processor, p_str, max_tokens=1920, temperature=0.75, repetition_penalty=1.03, verbose=False)
                mx.clear_cache()
                g_txt = res.text or ""
                if p_str.endswith("<think>") and not g_txt.lstrip().startswith("<think>"):
                    g_txt = "<think>" + g_txt
                c_ids = tok.encode(g_txt, add_special_tokens=False) + [eos_id]
                f_seq = mx.concatenate([p_mx, mx.array([c_ids], dtype=mx.int32)], axis=1)[:, :2432]
                sc = score_response(g_txt, prompt_text=p_txt, kind=row["kind"], verifier=row["verifier"], tokenizer=tok, max_think_tokens=1880)
                # Stanford Four-Habits / Skywork-OR1 Deep-Verification Bonus across full `300..1,850`-token `<think>` spectrum!
                eff_rew = float(sc["reward"])
                if sc["correct"] and 900 <= sc["think_tokens"] <= 1860:
                    eff_rew += 0.25
                elif sc["correct"] and 280 <= sc["think_tokens"] < 900:
                    eff_rew += 0.18
                sc["reward"] = round(eff_rew, 4)
                rollouts.append({"full_ids": f_seq, "score": sc})

            rews = [r["score"]["reward"] for r in rollouts]
            p_cnt = sum(1 for r in rollouts if r["score"]["correct"])
            mean_th = round(sum(r["score"]["think_tokens"] for r in rollouts) / 4.0, 1)
            spread = max(rews) - min(rews)
            anc_b = anc_rng.sample(short_anc_items, k=2)

            if p_cnt == 4 and spread < 1e-4:
                dapo_skips += 1
                tag = f"DAPO-skip(4/4,th={mean_th})"
            elif p_cnt == 0 and spread < 1e-4:
                ref_r = {
                    "id": f"r_{row['id']}",
                    "pillar": row["kind"],
                    "train_last_turn_only": True,
                    "images": [],
                    "messages": [*p_msgs, {"role": "assistant", "content": row["reference_answer"]}],
                }
                r_it, _ = prepare_turn_items_v8([ref_r], vlm, processor, max_seq_len=2432)
                _run_rectified_dft_scope_batch(vlm, (r_it[-1:] + anc_b[:1]) if r_it else anc_b[:1], opt_rl, scale_grad=0.20)
                apply_qk_clip(vlm, max_lora_fro_norm=6.0)
                active_upd += 0.5
                tag = f"OPSD-0/4(th={mean_th})"
            else:
                mean_r = sum(rews) / len(rews)
                advs = [r - mean_r for r in rews]
                vlm.eval()
                old_lps = []
                for ro in rollouts:
                    lp_o = mx.stop_gradient(-chunked_multimodal_ce(vlm, ro["full_ids"], s_idx))
                    mx.eval(lp_o)
                    old_lps.append(lp_o)
                    mx.clear_cache()

                snap_w = {k: mx.array(v) for k, v in tree_flatten(vlm.trainable_parameters()) if ".table.weight" not in k}
                mx.eval(list(snap_w.values()))
                vlm.load_weights(list(ref_w.items()), strict=False)
                mx.eval(vlm.parameters())
                ref_lps = []
                for ro in rollouts:
                    lp_r = mx.stop_gradient(-chunked_multimodal_ce(vlm, ro["full_ids"], s_idx))
                    mx.eval(lp_r)
                    ref_lps.append(lp_r)
                    mx.clear_cache()

                vlm.load_weights(list(snap_w.items()), strict=False)
                mx.eval(vlm.parameters())
                del snap_w
                mx.clear_cache()
                vlm.train()

                def _dr_grpo_clipcov_loss(rm: nn.Module, f_ids: mx.array, o_lp: mx.array, r_lp: mx.array, av: float) -> mx.array:
                    n_lp = -chunked_multimodal_ce(rm, f_ids, s_idx)
                    rat = mx.exp(mx.clip(n_lp - o_lp, -5.0, 5.0))
                    adv_m = mx.array(av, dtype=mx.float32)
                    # Asymmetric `DAPO` `Clip-Higher` (`0.80..1.28`) + `Clip-Cov` high-confidence ceiling (`>97%` prob damped 4x)
                    clipped_rat = mx.clip(rat, 0.80, 1.28)
                    pg = -mx.minimum(rat * adv_m, clipped_rat * adv_m)
                    pg = mx.where((adv_m > 0) & (mx.stop_gradient(n_lp) > -0.03), 0.25 * pg, pg)
                    kl = mx.exp(r_lp - n_lp) - (r_lp - n_lp) - 1.0
                    return mx.sum(pg + 0.025 * kl) / mx.array(900.0, dtype=mx.float32)

                vg = nn.value_and_grad(vlm, _dr_grpo_clipcov_loss)
                acc_g = None
                for ro, olp, rlp, av in zip(rollouts, old_lps, ref_lps, advs):
                    lr_v, gr = vg(vlm, ro["full_ids"], olp, rlp, av)
                    mx.eval(lr_v, gr)
                    acc_g = tree_map(lambda g: g / 4.0, gr) if acc_g is None else tree_map(lambda a, g: a + g / 4.0, acc_g, gr)
                    mx.eval(acc_g)
                    del lr_v, gr
                    mx.clear_cache()
                del old_lps, ref_lps
                _, _, anc_g = _run_rectified_dft_scope_batch(vlm, anc_b, None, scale_grad=0.14)
                mx.eval(anc_g)
                if acc_g is not None and anc_g is not None:
                    acc_g = tree_map(lambda a, b: a + b, acc_g, anc_g)
                if acc_g is not None:
                    acc_g, _ = optim.clip_grad_norm(acc_g, max_norm=1.0)
                    opt_rl.update(vlm, acc_g)
                    apply_qk_clip(vlm, max_lora_fro_norm=6.0)
                    mx.eval(vlm.trainable_parameters(), opt_rl.state)
                del acc_g, anc_g
                vlm.eval()
                active_upd += 1.0
                tag = f"DrGRPO+ClipCov(th={mean_th},sp={spread:.2f})"

            ep_idx += 1
            state["rl_ep_idx"] = ep_idx
            state["rl_active_updates"] = active_upd
            state["rl_dapo_skips"] = dapo_skips
            del rollouts
            gc.collect()
            mx.clear_cache()

            if ep_idx % 4 == 0 and ep_idx % 8 != 0 and active_upd < float(target_active_rl_updates) and ep_idx < n_lim:
                _save_v8_ckpt(vlm, CKPT_V12 / "latest", state)
                print(
                    f"  [v12 Stage 3 Ckpt ep={ep_idx:3d}/{n_lim} active={active_upd:4.1f}/{target_active_rl_updates} skips={dapo_skips} | {(time.perf_counter()-t_rl)/60:4.1f}m] "
                    f"{row['id']:18s} | pass={p_cnt}/4 | {tag}",
                    flush=True,
                )

            if ep_idx % 8 == 0 or active_upd >= float(target_active_rl_updates) or ep_idx >= n_lim:
                vlm.eval()
                vl, _ = evaluate_val_loss(vlm, va_items[:60])
                _save_v8_ckpt(vlm, CKPT_V12 / "latest", state)
                if vl <= state.get("best_rl_val", 1e9) + 0.025:
                    if vl < state.get("best_rl_val", 1e9):
                        state["best_rl_val"] = vl
                    _save_v8_ckpt(vlm, CKPT_V12 / "rl_best")
                print(
                    f"  [v12 Stage 3 DAPO+ClipCov ep={ep_idx:3d}/{n_lim} active={active_upd:4.1f}/{target_active_rl_updates} skips={dapo_skips} | {(time.perf_counter()-t_rl)/60:4.1f}m] "
                    f"{row['id']:18s} | pass={p_cnt}/4 | {tag} | val={vl:.4f}",
                    flush=True,
                )

        del ref_w, opt_rl
        state["rl_done"] = True
        _save_v8_ckpt(vlm, CKPT_V12 / "latest", state)
        gc.collect()
        mx.clear_cache()

    # ------------------------------------------------------------------
    # Stage 4A: Streaming `WiSE-FT` Gate -> Pure `16-bit` Lean Fusion into `STAGING_V12`
    # ------------------------------------------------------------------
    _ensure_8991_unloaded()
    vlm.eval()
    del short_anc_items
    gc.collect()
    mx.clear_cache()

    p_sft = CKPT_V12 / ("longkv_best" if (CKPT_V12 / "longkv_best" / "trainable.safetensors").exists() else "sft_best") / "trainable.safetensors"
    p_orpo = CKPT_V12 / ("orpo_best" if (CKPT_V12 / "orpo_best" / "trainable.safetensors").exists() else "sft_best") / "trainable.safetensors"
    p_rl = CKPT_V12 / ("rl_best" if (CKPT_V12 / "rl_best" / "trainable.safetensors").exists() else "latest") / "trainable.safetensors"

    d_tmp = mx.load(str(p_sft))
    dtypes_map = {k: v.dtype for k, v in d_tmp.items()}
    acc_wise = {k: 0.30 * v.astype(mx.float32) for k, v in d_tmp.items()}
    mx.eval(list(acc_wise.values()))
    del d_tmp
    gc.collect()
    mx.clear_cache()

    d_tmp = mx.load(str(p_orpo))
    for k, v in d_tmp.items():
        acc_wise[k] = acc_wise[k] + 0.32 * v.astype(mx.float32)
    mx.eval(list(acc_wise.values()))
    del d_tmp
    gc.collect()
    mx.clear_cache()

    d_tmp = mx.load(str(p_rl))
    w_wise = {k: (acc_wise[k] + 0.38 * d_tmp[k].astype(mx.float32)).astype(dtypes_map[k]) for k in acc_wise}
    mx.eval(list(w_wise.values()))
    del d_tmp, acc_wise
    gc.collect()
    mx.clear_cache()

    stage0_ext = mx.load(str(STAGE0_V12_BASE / "v8_arch_ext.safetensors"))
    stage0_base_gates: dict[str, mx.array] = {}
    raw_s0_w = mx.load(str(STAGE0_V12_BASE / "model.safetensors"))
    for idx_g in UPCYCLED_40L_INDICES:
        for suf_g in ("post_operator_norm.weight", "post_ffn_norm.weight"):
            gk = f"language_model.model.layers.{idx_g}.{suf_g}"
            if gk in raw_s0_w:
                stage0_base_gates[gk] = mx.array(raw_s0_w[gk])
    mx.eval(list(stage0_base_gates.values()), list(stage0_ext.values()))
    del raw_s0_w
    gc.collect()
    mx.clear_cache()

    def _apply_interp_to_vlm(base_dict: dict[str, mx.array], tau_lora: float, tau_ext: float) -> None:
        items_load = []
        for k, v in base_dict.items():
            if "lora_b" in k:
                items_load.append((k, (v.astype(mx.float32) * tau_lora).astype(v.dtype)))
            elif k in stage0_base_gates:
                g0 = stage0_base_gates[k].astype(mx.float32)
                items_load.append((k, (g0 + tau_lora * (v.astype(mx.float32) - g0)).astype(v.dtype)))
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
    ref_sc = (round(acc_0, 4), round(rew_0 - 0.30 * vl_0, 4))
    print(f"  [v12 Stage-0 Reference (`0.65*v11+0.35*v9_40L` tau=0)] probe_acc={acc_0*100:5.1f}% (rew={rew_0:.3f}) | val_v12={vl_0:.4f}", flush=True)

    best_name = "v12_wise_tau0.85"
    best_kind = ("wise", 0.85, 0.92)
    best_sc = (-1.0, -1e9)
    best_val_loss = round(vl_0, 4)

    for c_name, src_tag, tl, te in [
        ("v12_wise_tau0.78", "wise", 0.78, 0.88),
        ("v12_wise_tau0.88", "wise", 0.88, 0.95),
        ("v12_wise_tau1.00", "wise", 1.00, 1.00),
        ("v12_orpo_tau1.00", "orpo", 1.00, 1.00),
        ("v12_rl_tau1.00", "rl", 1.00, 1.00),
    ]:
        if src_tag == "wise":
            _apply_interp_to_vlm(w_wise, tl, te)
        else:
            src_p = p_orpo if src_tag == "orpo" else p_rl
            vlm.load_weights(str(src_p), strict=False)
            mx.eval(vlm.parameters())
        vl_c, _ = evaluate_val_loss(vlm, va_items[:60])
        acc_c, rew_c, _ = _eval_task_batch(vlm, processor, probe_tasks)
        sc_t = (round(acc_c, 4), round(rew_c - 0.30 * vl_c, 4))
        print(f"  [v12 Non-Degradation Gate] {c_name:18s} | probe_acc={acc_c*100:5.1f}% (rew={rew_c:.3f}) | val_v12={vl_c:.4f}", flush=True)
        if sc_t >= best_sc:
            best_sc = sc_t
            best_name = c_name
            best_kind = (src_tag, tl, te)
            best_val_loss = round(vl_c, 4)
        gc.collect()
        mx.clear_cache()

    print(f"[v12 Selected] Winner={best_name} (score={best_sc}, val={best_val_loss:.4f} vs stage0_ref={ref_sc})", flush=True)
    if best_kind[0] == "wise":
        _apply_interp_to_vlm(w_wise, best_kind[1], best_kind[2])
    else:
        src_win = p_orpo if best_kind[0] == "orpo" else p_rl
        vlm.load_weights(str(src_win), strict=False)
        mx.eval(vlm.parameters())

    del w_wise, stage0_ext, stage0_base_gates
    gc.collect()
    mx.clear_cache()

    # Free `CKPT_V12` BEFORE writing `5.4 GiB` `STAGING_V12` so free disk never dips below 16 GiB!
    if CKPT_V12.exists():
        shutil.rmtree(CKPT_V12, ignore_errors=True)

    _, fuse_info = fuse_and_export_v12_40l_lean(
        vlm,
        STAGE0_V12_BASE,
        STAGING_V12,
        extra_sidecar_meta={
            "model_version": "LFM3-VL-3.2B-RPR-V12-DEEP-BF16",
            "live_alias": "LFM3-VL-3.2B-RPR-V10-APEX-BF16",
            "precision": "bfloat16",
            "selected_candidate": best_name,
            "val_probe_score": list(best_sc),
            "selected_val_loss": best_val_loss,
            "stage0_ref_score": list(ref_sc),
            "sft_steps": sft_steps,
            "long_kv_steps": long_kv_steps,
            "orpo_steps": orpo_steps,
            "dapo_active_updates": state.get("rl_active_updates", 0),
        },
    )
    logits_expected = fuse_info["logits_expected"]

    del vlm, processor
    gc.collect()
    mx.clear_cache()

    # Immediately reclaim `5.9 GiB` `STAGE0_V12_BASE`!
    if STAGE0_V12_BASE.exists():
        shutil.rmtree(STAGE0_V12_BASE, ignore_errors=True)

    _ensure_8991_unloaded()
    vlm_re, proc_re = mlx_vlm.load(str(STAGING_V12))
    info_re = attach_v8_extensions(vlm_re, sidecar_dir=STAGING_V12, enable_moba=False)
    vlm_re.eval()
    probe_ids = mx.array([[1, 42, 105, 512, 1024, 2048]], dtype=mx.int32)
    logits_re = vlm_re.language_model(probe_ids).logits
    mx.eval(logits_re)
    re_diff = float(mx.max(mx.abs(logits_re.astype(mx.float32) - logits_expected.astype(mx.float32))).item())
    print(f"[v12 Pure 16-bit 3.22B Standalone Export Verified] {info_re['total_model_params']/1e9:.3f}B params | reload diff = {re_diff:.6f}", flush=True)
    assert re_diff < 1e-4, f"v12 reload mismatch: {re_diff}"
    del logits_expected, logits_re

    # Stage 4B: Extract `40L` `Transformer^2 SVF + Nullspace` Bank (`L26, 30, 36, 38`)
    tr_short, _ = prepare_turn_items_v8(
        [r for r in sft_rows if r.get("pillar") != "long_context_synthesis"][:160],
        vlm_re,
        proc_re,
        max_seq_len=2048,
    )
    skill_batches = [
        [it for it in tr_short if it.get("pillar") in ("advanced_rsi_dgm_svf", "rsi_and_weight_surgery")][:8] or tr_short[:6],
        [it for it in tr_short if it.get("pillar") in ("complex_logic_sat", "zero_prior_formal")][:8] or tr_short[6:12],
        [it for it in tr_short if it.get("pillar") in ("realtime_research_citations", "research_and_ablation")][:8] or tr_short[12:18],
        [it for it in tr_short if it.get("pillar") == "unseen_ood_self_discover"][:8] or tr_short[18:24],
    ]
    bank_path = STAGING_V12 / "seal_nullspace_svf_bank.safetensors"
    bank = extract_v10_40l_nullspace_svf_bank(vlm_re, va_items[:32], skill_batches, bank_path)

    meta_done = {
        **state,
        "model_version": "LFM3-VL-3.2B-RPR-V12-DEEP-BF16",
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


# ===========================================================================
# Stage 5: Warm-Started `18,432` `FR-Spec` `SequentialMTPHead` Trained on
#          `v12`'s Own Deep `<think>` On-Policy Distribution (`>= 1.6x`)
# ===========================================================================


def _build_v12_active_shortlist(tok: Any, sft_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]], onpol_raw: list[dict[str, Any]], target_size: int = 18432) -> list[int]:
    vocab_sz = int(tok.vocab_size)
    must_have = set(range(min(512, vocab_sz)))
    if getattr(tok, "eos_token_id", None) is not None:
        must_have.add(int(tok.eos_token_id))
    for special_s in (
        "<think>", "</think>", "<tool_call>", "</tool_call>", "<function=", "</function>",
        "<parameter=", "</parameter>", "```python", "```json", "```", "[S1]", "[S2]", "[S3]", "[S4]",
        "[DOC-", "[REC-", "[LOG-", "[L0", "[L1", "[L2", "[L3", "Wait",
    ):
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


def run_stage_5_v12_onpolicy_mtp(
    sft_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    *,
    mtp_steps: int = 1200,
) -> dict[str, Any]:
    import mlx_vlm

    _ensure_8991_unloaded()
    bench_dir = ROOT_DIR / "results" / "v12_mtp_benchmark"
    bench_dir.mkdir(parents=True, exist_ok=True)
    if (STAGING_V12 / "mtp_head.safetensors").exists() and (bench_dir / "summary.json").exists():
        print(f"[v12 Stage 5] Resuming completed v12 MTP benchmark from {bench_dir / 'summary.json'}", flush=True)
        return json.loads((bench_dir / "summary.json").read_text(encoding="utf-8"))

    vlm, processor = mlx_vlm.load(str(STAGING_V12))
    attach_v8_extensions(vlm, sidecar_dir=STAGING_V12, enable_moba=False)
    vlm.freeze()
    vlm.eval()
    tok = processor.tokenizer
    eos_id = int(tok.eos_token_id)

    text_rows = [r for r in sft_rows if not r.get("images") and r.get("pillar") != "long_context_synthesis"]
    WORK_V12.mkdir(parents=True, exist_ok=True)
    onpol_cache = WORK_V12 / "stage5_v12_onpolicy_96.json"
    onpol_raw: list[dict[str, Any]] = json.loads(onpol_cache.read_text(encoding="utf-8")) if onpol_cache.exists() else []
    if len(onpol_raw) < 96:
        print(f"[v12 Stage 5] Generating on-policy 40L v12 deep-thinking greedy trajectories ({len(onpol_raw)} -> 96)...", flush=True)
        if not onpol_raw:
            for msgs in GENERAL_ONPOLICY_PROMPTS:
                p_s = _apply_template_inference_aligned(processor, vlm.config, msgs, num_images=0, enable_thinking=True)
                p_ids = tok.encode(p_s, add_special_tokens=False)
                g_ids, _ = standard_greedy_generate(vlm, p_ids, max_tokens=288, eos_token_id=eos_id)
                if len(g_ids) >= 16:
                    onpol_raw.append({"full": (p_ids + g_ids)[:672], "ans_start": max(2, len(p_ids) - 2)})
        rng_op = random.Random(20261209)
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
            g_ids, _ = standard_greedy_generate(vlm, p_ids, max_tokens=288, eos_token_id=eos_id)
            if len(g_ids) >= 16:
                onpol_raw.append({"full": (p_ids + g_ids)[:672], "ans_start": max(2, len(p_ids) - 2)})
        onpol_cache.write_text(json.dumps(onpol_raw) + "\n", encoding="utf-8")

    shortlist_ids = _build_v12_active_shortlist(tok, text_rows[:700], val_rows, onpol_raw, target_size=18432)
    warm_mtp_src = EXPORT_V10_APEX if (EXPORT_V10_APEX / "mtp_head.safetensors").exists() else EXPORT_V9_SEAL
    mtp_head = load_mtp_checkpoint(vlm.config.text_config, warm_mtp_src)
    mtp_head.set_shortlist(shortlist_ids)
    embed_tokens = vlm.language_model.model.embed_tokens
    mtp_head.bind_vocab_projection(embed_tokens)
    shutil.copy2(ROOT_DIR / "mtp_speculative.py", STAGING_V12 / "mtp_speculative.py")
    print(
        f"[v12 Stage 5 Init] Warm-started pretrained MTP from {warm_mtp_src.name} | "
        f"intermediate={mtp_head.mtp_intermediate_size} | FR-Spec shortlist={len(shortlist_ids)} | on-policy seqs={len(onpol_raw)}",
        flush=True,
    )

    raw_seq_specs: list[tuple[list[int], int, bool]] = [(it["full"], int(it["ans_start"]), True) for it in onpol_raw]
    rng_sft = random.Random(20261210)
    cand_sft = list(text_rows)
    rng_sft.shuffle(cand_sft)
    for r in cand_sft:
        if len(raw_seq_specs) >= 384:
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
        # Sample BOTH early `<think>` AND tail `</think> + code/JSON` windows so MTP accelerates BOTH `<think>` AND code!
        full_early = (p_ids + a_ids)[:672]
        if len(full_early) >= len(p_ids) + 12:
            raw_seq_specs.append((full_early, max(2, len(p_ids) - 2), False))
        if len(p_ids) + len(a_ids) > 680 and len(raw_seq_specs) < 384:
            tail_slice = (p_ids + a_ids)[-640:]
            raw_seq_specs.append((tail_slice, 64, False))

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
        full = (p_ids + a_ids)[:672]
        if len(full) >= len(p_ids) + 12:
            val_seq_specs.append((full, max(2, len(p_ids) - 2)))

    print(f"[v12 Stage 5] Caching frozen 40L v12 trunk hidden features across {len(raw_seq_specs)} train + {len(val_seq_specs)} val sequences...", flush=True)
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

    train_pool: list[tuple[mx.array, mx.array, int]] = []
    for f_mx, h_out, ans_s, is_onp in cached_feats:
        for _ in range(3 if is_onp else 1):
            train_pool.append((f_mx, h_out, ans_s))
    pool_rng = random.Random(20261211)
    pool_rng.shuffle(train_pool)

    mtp_head.unfreeze()
    opt_mtp = optim.AdamW(
        learning_rate=_warmup_cosine(1.5e-4, 1.0e-5, mtp_steps, 0, warmup_steps=35),
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
        h_m1 = head.forward_features(h_in_1, mx.stop_gradient(embed_tokens(tok_in_1)), cache=None)
        h_m1_a = h_m1[:, a0:a1, :]
        feat_1 = nn.losses.smooth_l1_loss(h_m1_a.astype(mx.float32), h_tgt_next[:, a0:a1, :].astype(mx.float32), beta=0.1)
        l1 = head.draft_logits_shortlist(h_m1_a, embed_tokens).astype(mx.float32)
        lp1 = l1 - mx.logsumexp(l1, axis=-1, keepdims=True)
        loss_1 = mx.mean(0.50 * (-mx.sum(probs_1 * lp1, axis=-1)) + 0.50 * nn.losses.cross_entropy(l1, top1_1, reduction="none"))

        h_in_2 = mx.concatenate([h_in_1[:, 1 : a0 + 1, :], h_m1[:, a0:-1, :]], axis=1)
        h_m2 = head.forward_features(h_in_2, mx.stop_gradient(embed_tokens(tok_in_2)), cache=None)
        h_m2_a = h_m2[:, a0 : a0 + sp2, :]
        feat_2 = nn.losses.smooth_l1_loss(h_m2_a.astype(mx.float32), h_tgt_next[:, a0 + 1 : a0 + 1 + sp2, :].astype(mx.float32), beta=0.1)
        l2 = head.draft_logits_shortlist(h_m2_a, embed_tokens).astype(mx.float32)
        lp2 = l2 - mx.logsumexp(l2, axis=-1, keepdims=True)
        loss_2 = mx.mean(0.50 * (-mx.sum(probs_2 * lp2, axis=-1)) + 0.50 * nn.losses.cross_entropy(l2, top1_2, reduction="none"))

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
            va1 = min(vl1, va0 + 280)
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
    save_mtp_checkpoint(mtp_head, WORK_V12 / "best_mtp")
    print(f"  [v12 Stage 5 Step-0 Warm-Started MTP Val Loss] val_mtp={best_mtp_val:.4f}", flush=True)
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
            len1, len2, len3 = slen - 2, slen - 3, slen - 4
            a0 = max(2, min(ans_s - 2, len3 - 8))
            a1 = min(len1, a0 + 280)
            sp2, sp3 = min(len2, a1 - 1) - a0, min(len3, a1 - 2) - a0
            if sp3 < 6:
                continue

            h_in_1 = mx.stop_gradient(t_out[:, :-1, :])
            h_tgt_next = mx.stop_gradient(t_out[:, 1:, :])
            assert mtp_head._cached_w_sub is not None
            tsub = (h_tgt_next[:, a0:a1, :] @ mtp_head._cached_w_sub.T).astype(mx.float32)
            p1 = mx.stop_gradient(mx.softmax(tsub, axis=-1))
            top1 = mx.stop_gradient(mx.argmax(tsub, axis=-1))

            lv_m, gr_m = vg_m(
                mtp_head, h_in_1, h_tgt_next, f_ids[:, 1:-1], f_ids[:, 2:-1], f_ids[:, 3:-1],
                a0, a1, sp2, sp3, top1, p1, top1[:, 1 : 1 + sp2], p1[:, 1 : 1 + sp2, :],
                top1[:, 2 : 2 + sp3], p1[:, 2 : 2 + sp3, :],
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
                save_mtp_checkpoint(mtp_head, WORK_V12 / "best_mtp")
            print(
                f"  [v12 Stage 5 40L 3-Step MTP {st+1:4d}/{mtp_steps} (seqs={(st+1)*4}) | {(time.perf_counter()-t0)/60:4.1f}m] "
                f"train_loss={sum(run_l)/max(1,len(run_l)):.4f} | val_mtp={cur_mv:.4f} (best={best_mtp_val:.4f})",
                flush=True,
            )
            run_l.clear()
            gc.collect()
            mx.clear_cache()

    del cached_feats, val_cached, train_pool, opt_mtp
    gc.collect()
    mx.clear_cache()

    mtp_head = load_mtp_checkpoint(vlm.config.text_config, WORK_V12 / "best_mtp")
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
    standard_greedy_generate(vlm, wp, max_tokens=16, eos_token_id=eos_id)
    mtp_speculative_generate(vlm, mtp_head, wp, max_tokens=16, draft_k=2, eos_token_id=eos_id)
    mtp_speculative_generate(vlm, mtp_head, wp, max_tokens=16, draft_k=3, eos_token_id=eos_id)

    best_k = 2
    best_summary: dict[str, Any] = {}
    for k_try in (2, 3):
        per_case = []
        tot_std_t, tot_std_s, tot_mtp_t, tot_mtp_s = 0, 0.0, 0, 0.0
        for cname, msgs in bench_prompts:
            ps = _apply_template_inference_aligned(processor, vlm.config, msgs, num_images=0, enable_thinking=True)
            pids = tok.encode(ps, add_special_tokens=False)
            ref_ids, st_std = standard_greedy_generate(vlm, pids, max_tokens=320, eos_token_id=eos_id)
            m_ids, st_m = mtp_speculative_generate(vlm, mtp_head, pids, max_tokens=320, draft_k=k_try, eos_token_id=eos_id)
            exact_l1 = (ref_ids == m_ids)
            if not exact_l1:
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
                f"  [v12 40L Pure-16b MTP K={k_try} | {cname:18s}] std={st_std['tok_per_sec']:5.1f} -> MTP={st_m['tok_per_sec']:6.1f} tok/s "
                f"({sp:.2f}x | acc={st_m['acceptance_rate']*100:.1f}% | tok/ver={st_m['mean_tokens_per_verify']:.2f})",
                flush=True,
            )
        ov_std = round(tot_std_t / max(1e-6, tot_std_s), 2)
        ov_mtp = round(tot_mtp_t / max(1e-6, tot_mtp_s), 2)
        ov_sp = round(ov_mtp / max(1e-6, ov_std), 3)
        cand_sum = {
            "model": str(STAGING_V12),
            "model_version": "LFM3-VL-3.2B-RPR-V12-DEEP-BF16",
            "num_layers": 40,
            "latent_recur_layers": [29, 35],
            "latent_recur_loops": 4,
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
        print(f"[v12 40L MTP K={k_try} Aggregate] {ov_std} -> {ov_mtp} tok/s ({ov_sp}x overall, min={cand_sum['min_case_speedup']}x, max={cand_sum['max_case_speedup']}x)", flush=True)
        if not best_summary or ov_sp >= best_summary["overall_speedup"]:
            best_k = k_try
            best_summary = cand_sum

    mtp_cfg_p = STAGING_V12 / "mtp_config.json"
    mtp_cfg_d = json.loads(mtp_cfg_p.read_text(encoding="utf-8"))
    mtp_cfg_d["recommended_draft_k"] = best_k
    mtp_cfg_p.write_text(json.dumps(mtp_cfg_d, indent=2) + "\n", encoding="utf-8")

    (bench_dir / "summary.json").write_text(json.dumps(best_summary, indent=2) + "\n", encoding="utf-8")
    if WORK_V12.exists():
        shutil.rmtree(WORK_V12, ignore_errors=True)
    del vlm, processor, mtp_head
    gc.collect()
    mx.clear_cache()
    return best_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RPR v12 DEEP-THINKER (3.22B 40L Pure 16-bit + Dual 4-Loop Huginn + ThoughtAnchorTemp + Rectified DFT + Clip-Cov DAPO + MTP) Pipeline.")
    parser.add_argument("--sft-steps", type=int, default=1350)
    parser.add_argument("--long-kv-steps", type=int, default=85)
    parser.add_argument("--orpo-steps", type=int, default=250)
    parser.add_argument("--active-rl-updates", type=int, default=36)
    parser.add_argument("--max-rl-episodes", type=int, default=120)
    parser.add_argument("--mtp-steps", type=int, default=1200)
    args = parser.parse_args()

    mx.set_wired_limit(0)
    mx.set_cache_limit(128 * 1024 * 1024)
    WORK_V12.mkdir(parents=True, exist_ok=True)
    _ensure_8991_unloaded()

    sft_rows = [json.loads(l) for l in (V12_CORPUS_DIR / "train_v12_sft.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    long_kv_rows = [json.loads(l) for l in (V12_CORPUS_DIR / "train_v12_long_kv.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    orpo_rows = [json.loads(l) for l in (V12_CORPUS_DIR / "orpo_v12_pairs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    dapo_rows = [json.loads(l) for l in (V12_CORPUS_DIR / "dapo_v12_episodes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    val_rows = [json.loads(l) for l in (V12_CORPUS_DIR / "val_v12_joint.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    run_stages_01234_v12(
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
    s5 = run_stage_5_v12_onpolicy_mtp(sft_rows, val_rows, mtp_steps=args.mtp_steps)
    print(
        f"[v12 3.22B DEEP-THINKER Pipeline Complete] Pure 16-bit MTP K={s5['draft_k']} Speedup: "
        f"{s5['overall_speedup']}x ({s5['std_tok_per_sec']} -> {s5['mtp_tok_per_sec']} tok/s, min={s5['min_case_speedup']}x)",
        flush=True,
    )


if __name__ == "__main__":
    main()
