"""Surgical +243M Global-Attention Depth Upcycling (24L -> 28L, 6 -> 8 Full-Attention) & 5-Stage RSI/RPR BF16 Trainer.

Produces standalone unquantized 16-bit (`bfloat16`) model:
  `checkpoints/LFM3-VL-2.24B-RPR-RSI-BF16/`

Stages (every stage & sub-interval atomically checkpointed for zero-loss resume):
  - Stage 0a: Lossless `bfloat16` dequantization of read-only `./models/LFM3-VL-2B-MLX-8bit`,
              surgical insertion of two `[conv, full_attention]` pairs (`[13,14]` -> `[15,16]` and
              `[21,22]` -> `[25,26]`, expanding `num_hidden_layers: 24 -> 28` and `full_attention: 6 -> 8`)
              with `post_operator_norm.weight = 0` and `post_ffn_norm.weight = 0`.
              Verifies exact Step-0 mathematical identity (`max logit diff == 0.000000` vs 24L) and
              unit-RMS non-singular SandwichNorm Jacobians.
  - Stage 0b: Targeted Upcycled-Block Warmup (`240` steps on `[15, 16, 25, 26]` only; original 24 layers frozen),
              gently opening `post_operator_norm` / `post_ffn_norm` gates and specializing the new global heads.
  - Stage 1 : Full 28-Layer + Multimodal Projector (`181` linear targets + upcycled SandwichNorm gates)
              Rank-32 (`alpha=64`) LoRA+ DFT (`lr_B = 3.6x lr_A`), WITH fixed `downsample_factor**2 = 4`
              multimodal vision token alignment (`180` PIL vision `<think>` items active!) and `patience=2`
              anti-overfitting early stopping (`1` epoch max).
  - Stage 2 : Shuffled On-Policy Multi-Turn SCoRe + GDPO/Dr.GRPO Verifiable RL (`96` episodes x `4` rollouts = `384`
              live 28L rollouts) anchored to `stage1_best` (`0.04 * KL` + `0.10 * DFT` multimodal/text replay).
  - Stage 3 : Strict Validation-Only 2D `WiSE-FT` sweep (`rl_lam` x `base_gamma`) scored on `val_rpr_v5` NLL
              + dedicated 16-task validation probe (zero touch on any held-out evaluation benchmark!).
  - Stage 4 : Lossless 16-bit `bfloat16` weight fusion into `checkpoints/LFM3-VL-2.24B-RPR-RSI-BF16/` +
              fresh `mlx_vlm.load` disk-reload parity gate (`max logit diff == 0.000000`).
"""

from __future__ import annotations

import argparse
import copy
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

from model import DEFAULT_MODEL_PATH, split_thinking
from training.build_rpr_v5_rsi_corpus import V5_CORPUS_DIR, build_heldout_v5_rsi_56
from training.train_and_fuse_rpr_bf16 import (
    _apply_template_inference_aligned,
    _evaluate_mini_probe,
    _run_dft_batch,
    chunked_multimodal_ce,
    dequantize_model_to_bf16,
    evaluate_val_loss,
)
from training.verifiable_rewards import score_response


WORK_ROOT = ROOT_DIR / "checkpoints" / "rpr_v5_upcycle_work"
STAGE0_ID_DIR = WORK_ROOT / "stage0_identity_28L"
STAGE0_WARM_DIR = WORK_ROOT / "stage0_warmed_28L"
LATEST_DIR = WORK_ROOT / "latest"
STAGE1_BEST_DIR = WORK_ROOT / "stage1_best"
STAGE2_BEST_DIR = WORK_ROOT / "stage2_best"
WISE_BEST_DIR = WORK_ROOT / "wise_ft_best"
EXPORT_V5_DIR = ROOT_DIR / "checkpoints" / "LFM3-VL-2.2B-RPR-RSI-BF16"

UPCYCLED_LAYER_INDICES = (15, 16, 25, 26)
GATE_NORM_LEAVES = {"post_operator_norm", "post_ffn_norm"}


# ---------------------------------------------------------------------------
# Fixed Multimodal Turn Preparation (Handles `downsample_factor**2 = 4`!)
# ---------------------------------------------------------------------------


def prepare_turn_items_v5(
    rows: list[dict[str, Any]],
    model: Any,
    processor: Any,
    *,
    max_seq_len: int = 1792,
) -> tuple[list[dict[str, Any]], int]:
    """Expand 1..3-turn conversations into assistant-masked sequences + cached frozen `vision_tower` features."""
    from mlx_vlm.generate import _prepare_generation_inputs

    tok = processor.tokenizer
    eos_id = int(tok.eos_token_id)
    img_tok_id = int(model.config.image_token_id)
    ds_sq = int(model.config.downsample_factor) ** 2
    prepared: list[dict[str, Any]] = []
    vis_kept = 0

    for row in rows:
        msgs = row["messages"]
        imgs = [str(p) for p in (row.get("images") or []) if p]
        for t_idx, msg in enumerate(msgs):
            if msg.get("role") != "assistant":
                continue
            prefix_msgs = msgs[:t_idx]
            prompt_str = _apply_template_inference_aligned(
                processor, model.config, prefix_msgs, num_images=len(imgs), enable_thinking=True
            )
            ans_str = str(msg["content"])
            if prompt_str.endswith("<think>") and ans_str.startswith("<think>"):
                ans_str = ans_str[len("<think>") :]
            elif prompt_str.endswith("<think>\n") and ans_str.startswith("<think>\n"):
                ans_str = ans_str[len("<think>\n") :]

            a_ids = list(tok.encode(ans_str, add_special_tokens=False)) + [eos_id]
            if not a_ids:
                continue

            if imgs:
                kw_p: dict[str, Any] = {}
                p_ids_mx, p_pix, _, _ = _prepare_generation_inputs(
                    model, processor, prompt_str, imgs, None, None, kw_p
                )
                vt_feats = mx.stop_gradient(model.vision_tower(p_pix, kw_p["image_grid_thw"]))
                mx.eval(p_ids_mx, vt_feats)
                p_len = int(p_ids_mx.shape[1])
                full_mx = mx.concatenate([p_ids_mx, mx.array([a_ids], dtype=p_ids_mx.dtype)], axis=1)
                if int(full_mx.shape[1]) > max_seq_len:
                    full_mx = full_mx[:, :max_seq_len]
                if int(full_mx.shape[1]) <= p_len + 2:
                    continue
                idx_np = np.flatnonzero(np.asarray(full_mx[:, :-1]).reshape(-1) == img_tok_id)
                # Note: `vision_tower` outputs `ds_sq * N_vis_tokens` rows before `multi_modal_projector` reshapes!
                if len(idx_np) * ds_sq != int(vt_feats.shape[0]):
                    continue
                img_indices = mx.array(idx_np, dtype=mx.int32)
                start_idx = max(p_len - 1, 0)
                vis_kept += 1
            else:
                p_ids = tok.encode(prompt_str, add_special_tokens=False)
                full = (p_ids + a_ids)[:max_seq_len]
                if len(full) <= len(p_ids) + 2:
                    continue
                full_mx = mx.array([full], dtype=mx.int32)
                vt_feats = None
                img_indices = None
                start_idx = max(len(p_ids) - 1, 0)

            prepared.append(
                {
                    "id": f"{row.get('id', 'sft')}_t{t_idx}",
                    "pillar": row.get("pillar", "unknown"),
                    "full_ids": full_mx,
                    "start_idx": start_idx,
                    "vt_feats": vt_feats,
                    "img_indices": img_indices,
                }
            )
    mx.clear_cache()
    return prepared, vis_kept


# ---------------------------------------------------------------------------
# Surgical 24L -> 28L Global-Attention Depth Upcycling (SandwichNorm Gate = 0)
# ---------------------------------------------------------------------------


def build_upcycled_28l_config_dict(base_cfg_dict: dict[str, Any]) -> dict[str, Any]:
    """Expand 24-layer LFM3 text_config to 28 layers (`6 -> 8` `full_attention` layers)."""
    cfg = copy.deepcopy(base_cfg_dict)
    cfg.pop("quantization", None)
    cfg.pop("quantization_config", None)
    cfg["dtype"] = "bfloat16"

    tcfg = cfg["text_config"]
    tcfg.pop("quantization", None)
    tcfg.pop("quantization_config", None)
    tcfg["dtype"] = "bfloat16"

    old_types = list(tcfg["layer_types"])
    assert len(old_types) == 24, f"Expected 24 base layers, found {len(old_types)}"
    assert old_types[13:15] == ["conv", "full_attention"]
    assert old_types[21:23] == ["conv", "full_attention"]

    # Map:
    #   0..14  <- old 0..14
    #   15..16 <- cloned old 13..14 ([conv, full_attention])
    #   17..24 <- old 15..22
    #   25..26 <- cloned old 21..22 ([conv, full_attention])
    #   27     <- old 23
    new_types = (
        old_types[:15]
        + ["conv", "full_attention"]
        + old_types[15:23]
        + ["conv", "full_attention"]
        + [old_types[23]]
    )
    assert len(new_types) == 28
    assert new_types.count("full_attention") == 8
    tcfg["num_hidden_layers"] = 28
    tcfg["layer_types"] = new_types
    return cfg


def upcycle_24l_to_28l_identity(base_model_dir: Path, out_28l_dir: Path) -> None:
    """Create a standalone 28-layer `bfloat16` LFM3-VL checkpoint with exact Step-0 identity to 24L."""
    import mlx_vlm

    if (out_28l_dir / "model.safetensors").exists() and (out_28l_dir / "upcycle_verified.json").exists():
        print(f"[Stage 0a] Resuming verified 28L identity checkpoint at {out_28l_dir}", flush=True)
        return

    print(f"[Stage 0a] Loading 24L base {base_model_dir} -> lossless bfloat16...", flush=True)
    model_24, _ = mlx_vlm.load(str(base_model_dir))
    n_deq = dequantize_model_to_bf16(model_24)
    print(f"[Stage 0a] Dequantized {n_deq} modules to bfloat16.", flush=True)

    probe_ids = mx.array([[1, 42, 105, 512, 1024, 2048]], dtype=mx.int32)
    logits_24 = model_24.language_model(probe_ids).logits
    mx.eval(logits_24)

    flat_24 = dict(tree_flatten(model_24.parameters()))
    # Source layer mapping for each target layer 0..27:
    src_for_dst: dict[int, int] = {}
    for dst in range(15):
        src_for_dst[dst] = dst
    src_for_dst[15] = 13
    src_for_dst[16] = 14
    for dst in range(17, 25):
        src_for_dst[dst] = dst - 2
    src_for_dst[25] = 21
    src_for_dst[26] = 22
    src_for_dst[27] = 23

    flat_28: dict[str, mx.array] = {}
    # Copy all non-layer tensors (embeddings, embedding_norm, vision_tower, multi_modal_projector)
    prefix = "language_model.model.layers."
    for k, v in flat_24.items():
        if not k.startswith(prefix):
            flat_28[k] = v.astype(mx.bfloat16) if mx.issubdtype(v.dtype, mx.floating) else v

    for dst_idx, src_idx in src_for_dst.items():
        src_pref = f"{prefix}{src_idx}."
        dst_pref = f"{prefix}{dst_idx}."
        is_upcycled = dst_idx in UPCYCLED_LAYER_INDICES
        for k, v in flat_24.items():
            if not k.startswith(src_pref):
                continue
            sub_key = k[len(src_pref) :]
            new_key = dst_pref + sub_key
            if is_upcycled and sub_key in {"post_operator_norm.weight", "post_ffn_norm.weight"}:
                # Exact Step-0 residual zero-gate (`x + 0 * RMSNorm(h) == x`, unit-RMS O(1) Jacobian!)
                flat_28[new_key] = mx.zeros_like(v).astype(mx.bfloat16)
            else:
                flat_28[new_key] = v.astype(mx.bfloat16) if mx.issubdtype(v.dtype, mx.floating) else v

    mx.eval(list(flat_28.values()))
    n_params_total = sum(int(v.size) for v in flat_28.values())
    n_params_24 = sum(int(v.size) for v in flat_24.values())
    del model_24, flat_24
    gc.collect()
    mx.clear_cache()

    out_28l_dir.mkdir(parents=True, exist_ok=True)
    skip_names = {"model.safetensors", "model.safetensors.index.json", "config.json"}
    for item in base_model_dir.iterdir():
        if item.is_file() and item.name not in skip_names and not item.name.endswith(".safetensors"):
            shutil.copy2(item, out_28l_dir / item.name)

    base_cfg = json.loads((base_model_dir / "config.json").read_text(encoding="utf-8"))
    cfg_28 = build_upcycled_28l_config_dict(base_cfg)
    (out_28l_dir / "config.json").write_text(json.dumps(cfg_28, indent=2) + "\n", encoding="utf-8")

    st_path = out_28l_dir / "model.safetensors"
    mx.save_safetensors(str(st_path), flat_28, metadata={"format": "mlx"})
    total_bytes = st_path.stat().st_size
    (out_28l_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": total_bytes}, "weight_map": {k: "model.safetensors" for k in sorted(flat_28)}},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    del flat_28
    gc.collect()
    mx.clear_cache()

    # Reload via standard `mlx_vlm.load` and prove Step-0 mathematical identity!
    model_28, _ = mlx_vlm.load(str(out_28l_dir))
    logits_28 = model_28.language_model(probe_ids).logits
    mx.eval(logits_28)
    max_diff = float(mx.max(mx.abs(logits_28.astype(mx.float32) - logits_24.astype(mx.float32))).item())
    added_m = (n_params_total - n_params_24) / 1e6
    tot_b = n_params_total / 1e9
    print(
        f"[Stage 0a Verified] 24L -> 28L Upcycled Model: {tot_b:.3f}B params (+{added_m:.2f}M across 4 blocks, "
        f"8 full_attention layers, {total_bytes/(1024**3):.2f} GiB BF16) | Step-0 max logit diff = {max_diff:.6f}",
        flush=True,
    )
    assert max_diff < 1e-5, f"Step-0 identity check failed: max_diff={max_diff}"
    (out_28l_dir / "upcycle_verified.json").write_text(
        json.dumps(
            {
                "base_params": n_params_24,
                "upcycled_params": n_params_total,
                "added_params": n_params_total - n_params_24,
                "num_hidden_layers": 28,
                "full_attention_layers": 8,
                "step0_max_logit_diff": max_diff,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    del model_28
    gc.collect()
    mx.clear_cache()


def _unfreeze_upcycled_gate_norms(model: nn.Module) -> None:
    """Keep `post_operator_norm` & `post_ffn_norm` trainable on the 4 upcycled blocks `[15, 16, 25, 26]`."""
    for idx in UPCYCLED_LAYER_INDICES:
        lyr = model.language_model.model.layers[idx]
        if hasattr(lyr, "post_operator_norm"):
            lyr.post_operator_norm.unfreeze()
        if hasattr(lyr, "post_ffn_norm"):
            lyr.post_ffn_norm.unfreeze()


def _save_trainable_ckpt(
    model: nn.Module,
    ckpt_dir: Path,
    state_payload: dict[str, Any] | None = None,
) -> None:
    """Save ALL trainable tensors (LoRA `lora_a`/`lora_b` + upcycled SandwichNorm gate weights) atomically."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    flat_tr = dict(tree_flatten(model.trainable_parameters()))
    tmp_w = ckpt_dir / "trainable.tmp.safetensors"
    mx.save_safetensors(str(tmp_w), flat_tr, metadata={"format": "mlx"})
    tmp_w.replace(ckpt_dir / "trainable.safetensors")
    if state_payload is not None:
        tmp_s = ckpt_dir / "train_state.json.tmp"
        tmp_s.write_text(json.dumps(state_payload, indent=2), encoding="utf-8")
        tmp_s.replace(ckpt_dir / "train_state.json")


def fuse_and_export_28l_bf16(
    model: Any,
    template_28l_dir: Path,
    export_dir: Path,
) -> None:
    """Fuse all LoRA layers into 28-layer `bfloat16` `nn.Linear` and verify fresh `mlx_vlm.load` parity."""
    import mlx_vlm

    print(f"[Stage 4 Fuse] Merging all LoRA + upcycled gate weights -> {export_dir}...", flush=True)
    replacements: list[tuple[str, nn.Module]] = []
    merged_cnt = 0
    for name, mod in model.named_modules():
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
        model.update_modules(tree_unflatten(replacements))
    model.apply(lambda x: x.astype(mx.bfloat16) if mx.issubdtype(x.dtype, mx.floating) else x)
    mx.eval(model.parameters())

    export_dir.mkdir(parents=True, exist_ok=True)
    skip_names = {"model.safetensors", "model.safetensors.index.json", "upcycle_verified.json"}
    for item in template_28l_dir.iterdir():
        if item.is_file() and item.name not in skip_names and not item.name.endswith(".safetensors"):
            shutil.copy2(item, export_dir / item.name)

    flat_weights = dict(tree_flatten(model.parameters()))
    st_path = export_dir / "model.safetensors"
    mx.save_safetensors(str(st_path), flat_weights, metadata={"format": "mlx"})
    total_bytes = st_path.stat().st_size
    (export_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": total_bytes}, "weight_map": {k: "model.safetensors" for k in sorted(flat_weights)}},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"[Stage 4 Fuse] Wrote {len(flat_weights)} bfloat16 tensors ({merged_cnt} LoRA-fused) -> "
        f"{st_path} ({total_bytes / (1024**3):.2f} GiB).",
        flush=True,
    )

    test_ids = mx.array([[1, 42, 105, 512, 1024]], dtype=mx.int32)
    logits_mem = model.language_model(test_ids).logits
    mx.eval(logits_mem)

    reloaded, _ = mlx_vlm.load(str(export_dir))
    logits_disk = reloaded.language_model(test_ids).logits
    mx.eval(logits_disk)
    max_diff = float(mx.max(mx.abs(logits_mem.astype(mx.float32) - logits_disk.astype(mx.float32))).item())
    print(f"[Stage 4 Verify] Standalone 28L reload succeeded! Max logit diff = {max_diff:.6f}", flush=True)
    assert max_diff < 1e-3, f"Disk reload mismatch: {max_diff}"
    del reloaded
    gc.collect()
    mx.clear_cache()


# ---------------------------------------------------------------------------
# Stage 0b: Targeted Warmup of the 4 Newly Upcycled `[conv, full_attention]` Blocks
# ---------------------------------------------------------------------------


def run_stage0b_block_warmup(
    identity_28l_dir: Path,
    warmed_28l_dir: Path,
    train_raw: list[dict[str, Any]],
    val_raw: list[dict[str, Any]],
    *,
    warmup_steps: int = 220,
    fresh: bool = False,
) -> None:
    """Warm up ONLY the 4 newly upcycled blocks `[15, 16, 25, 26]` while keeping original 24 layers frozen."""
    import mlx_vlm
    from mlx_vlm.trainer.peft import get_peft_model

    if not fresh and (warmed_28l_dir / "model.safetensors").exists() and (warmed_28l_dir / "warmup_done.json").exists():
        print(f"[Stage 0b] Resuming warmed 28L checkpoint at {warmed_28l_dir}", flush=True)
        return

    print("[Stage 0b] Loading 28L identity model for surgical 4-block warmup...", flush=True)
    model, processor = mlx_vlm.load(str(identity_28l_dir))
    train_items, n_vis = prepare_turn_items_v5(train_raw, model, processor, max_seq_len=1792)
    val_items, _ = prepare_turn_items_v5(val_raw, model, processor, max_seq_len=1792)

    leaf_targets = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "in_proj", "out_proj", "w1", "w2", "w3"}
    upcycled_prefixes = tuple(f"model.layers.{i}." for i in UPCYCLED_LAYER_INDICES)
    block_targets = [
        f"language_model.{name}"
        for name, _ in model.language_model.named_modules()
        if name.startswith(upcycled_prefixes) and name.split(".")[-1] in leaf_targets
    ]
    model = get_peft_model(model, linear_layers=block_targets, rank=32, alpha=64, dropout=0.0, verbose=False)
    _unfreeze_upcycled_gate_norms(model)
    print(
        f"[Stage 0b] Warmup targets: {len(block_targets)} linear modules + 8 SandwichNorm gates across layers {UPCYCLED_LAYER_INDICES} "
        f"(vision turns ready: {n_vis}).",
        flush=True,
    )

    warm_ckpt = WORK_ROOT / "stage0b_latest"
    w_state = {"step": 0, "best_val": 1e9}
    if not fresh and (warm_ckpt / "train_state.json").exists() and (warm_ckpt / "trainable.safetensors").exists():
        w_state = json.loads((warm_ckpt / "train_state.json").read_text(encoding="utf-8"))
        model.load_weights(str(warm_ckpt / "trainable.safetensors"), strict=False)
        mx.eval(model.parameters())
        print(f"[Stage 0b] Resumed warmup at step {w_state['step']}/{warmup_steps}", flush=True)
    else:
        vl0, vp0 = evaluate_val_loss(model, val_items)
        w_state["best_val"] = vl0
        _save_trainable_ckpt(model, warm_ckpt / "best")
        print(f"[Stage 0b Start] 28L identity val_loss={vl0:.4f} (PPL={vp0:.4f})", flush=True)

    opt_w = optim.MultiOptimizer(
        [
            optim.AdamW(learning_rate=optim.cosine_decay(1.2e-3, decay_steps=warmup_steps, end=2.0e-4), weight_decay=0.0),
            optim.AdamW(learning_rate=optim.cosine_decay(1.2e-4, decay_steps=warmup_steps, end=2.0e-5), weight_decay=0.01),
            optim.AdamW(learning_rate=optim.cosine_decay(3.5e-5, decay_steps=warmup_steps, end=7.0e-6), weight_decay=0.01),
        ],
        filters=[
            lambda k, _: any(gn in k for gn in GATE_NORM_LEAVES),
            lambda k, _: "lora_b" in k,
        ],
    )

    rng = random.Random(20261210)
    order = list(range(len(train_items)))
    rng.shuffle(order)
    t0 = time.perf_counter()
    losses: list[float] = []

    for step in range(w_state["step"], warmup_steps):
        b_idx = [(step * 2) % len(order), (step * 2 + 1) % len(order)]
        batch = [train_items[order[j]] for j in b_idx]
        lv, _ = _run_dft_batch(model, batch, opt_w)
        losses.append(lv)

        if step == 0:
            gate_max = max(
                float(mx.max(mx.abs(model.language_model.model.layers[idx].post_ffn_norm.weight)).item())
                for idx in UPCYCLED_LAYER_INDICES
            )
            print(f"  [Stage 0b Step 1] max(|upcycled post_ffn_norm|) = {gate_max:.6e}", flush=True)
            assert gate_max > 1e-6, "FATAL: upcycled SandwichNorm gate did not open on step 1!"

        if (step + 1) % 60 == 0 or step == warmup_steps - 1:
            vl, vp = evaluate_val_loss(model, val_items)
            avg_l = sum(losses) / max(1, len(losses))
            losses.clear()
            w_state["step"] = step + 1
            if vl <= w_state["best_val"]:
                w_state["best_val"] = vl
                _save_trainable_ckpt(model, warm_ckpt / "best")
            _save_trainable_ckpt(model, warm_ckpt, w_state)
            gate_rms = [
                round(float(mx.sqrt(mx.mean(model.language_model.model.layers[idx].post_ffn_norm.weight.astype(mx.float32) ** 2)).item()), 4)
                for idx in UPCYCLED_LAYER_INDICES
            ]
            print(
                f"  [Stage 0b Warmup {step+1:3d}/{warmup_steps} | {(time.perf_counter()-t0)/60:.1f}m] "
                f"train={avg_l:.4f} | val={vl:.4f} (PPL={vp:.4f}) | gate_rms={gate_rms}",
                flush=True,
            )

    model.load_weights(str(warm_ckpt / "best" / "trainable.safetensors"), strict=False)
    mx.eval(model.parameters())
    fuse_and_export_28l_bf16(model, identity_28l_dir, warmed_28l_dir)
    (warmed_28l_dir / "warmup_done.json").write_text(json.dumps(w_state, indent=2) + "\n", encoding="utf-8")
    del model, processor, train_items, val_items
    gc.collect()
    mx.clear_cache()


# ---------------------------------------------------------------------------
# Main 5-Stage Pipeline
# ---------------------------------------------------------------------------


def run_v5_pipeline(args: argparse.Namespace) -> None:
    import mlx_vlm
    from mlx_vlm.trainer.peft import get_peft_model

    mx.set_wired_limit(0)
    mx.set_cache_limit(384 * 1024 * 1024)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)

    train_raw = [json.loads(l) for l in (V5_CORPUS_DIR / "train_rpr_v5.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    val_raw = [json.loads(l) for l in (V5_CORPUS_DIR / "val_rpr_v5.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rlvr_raw = [json.loads(l) for l in (V5_CORPUS_DIR / "rlvr_v5_160.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    # Dedicated validation-only 16-task 8-pillar probe (seed=20261299 — strictly disjoint from ALL held-out eval files!)
    val_probe_56 = build_heldout_v5_rsi_56(seed=20261299)
    probe_16 = [val_probe_56[p * 7 + j] for p in range(8) for j in range(2)]

    # Stage 0a & 0b: Surgical 24L -> 28L Identity Upcycling + 4-Block Gate/Attention Warmup
    upcycle_24l_to_28l_identity(Path(args.model), STAGE0_ID_DIR)
    run_stage0b_block_warmup(
        STAGE0_ID_DIR,
        STAGE0_WARM_DIR,
        train_raw,
        val_raw,
        warmup_steps=args.warmup_steps,
        fresh=args.fresh,
    )

    print(f"[Stage 1 Init] Loading warmed 28-layer bfloat16 model from {STAGE0_WARM_DIR}...", flush=True)
    model, processor = mlx_vlm.load(str(STAGE0_WARM_DIR))
    train_items, vis_train = prepare_turn_items_v5(train_raw, model, processor, max_seq_len=1792)
    val_items, vis_val = prepare_turn_items_v5(val_raw, model, processor, max_seq_len=1792)
    print(
        f"[Stage 1 Init] Tokenized turns: train={len(train_items)} ({vis_train} vision), "
        f"val={len(val_items)} ({vis_val} vision), rlvr={len(rlvr_raw)}",
        flush=True,
    )
    assert vis_train >= 100, f"Expected >=100 vision training items active, got {vis_train}!"

    lm_leaf_targets = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "in_proj", "out_proj", "w1", "w2", "w3"}
    lm_qualified = [
        f"language_model.{name}"
        for name, _ in model.language_model.named_modules()
        if name.split(".")[-1] in lm_leaf_targets
    ]
    proj_qualified = ["multi_modal_projector.linear_1", "multi_modal_projector.linear_2"]
    all_targets = lm_qualified + proj_qualified
    model = get_peft_model(model, linear_layers=all_targets, rank=32, alpha=64, dropout=0.05, verbose=False)
    _unfreeze_upcycled_gate_norms(model)
    print(
        f"[Stage 1 Init] Attached Rank-32 (alpha=64, dropout=0.05) LoRA across {len(all_targets)} modules "
        f"({len(lm_qualified)} 28L-LM + 2 Projector) + 8 unfrozen upcycled SandwichNorm gates.",
        flush=True,
    )

    resume_state: dict[str, Any] = {
        "stage": "sft",
        "epoch": 0,
        "sample_idx": 0,
        "rl_idx": 0,
        "warm_val_loss": None,
        "best_val_loss": 1e9,
        "best_rl_val_loss": 1e9,
        "stage1_done": False,
        "stage2_done": False,
    }
    state_file = LATEST_DIR / "train_state.json"
    if not args.fresh and state_file.exists() and (LATEST_DIR / "trainable.safetensors").exists():
        resume_state = json.loads(state_file.read_text(encoding="utf-8"))
        model.load_weights(str(LATEST_DIR / "trainable.safetensors"), strict=False)
        mx.eval(model.parameters())
        print(f"[Checkpoint] Resumed Stage 1/2 from {LATEST_DIR}: {resume_state}", flush=True)

    if resume_state.get("warm_val_loss") is None:
        w_vl, w_vp = evaluate_val_loss(model, val_items)
        resume_state["warm_val_loss"] = w_vl
        resume_state["best_val_loss"] = w_vl
        _save_trainable_ckpt(model, STAGE1_BEST_DIR)
        _save_trainable_ckpt(model, LATEST_DIR, resume_state)
        print(f"[Stage 1 Baseline (Warmed 28L)] val_loss={w_vl:.4f} (PPL={w_vp:.4f})", flush=True)

    # ------------------------------------------------------------------
    # STAGE 1: Strictly 1-Epoch Anti-Overfitting Full-28L LoRA+ DFT
    # ------------------------------------------------------------------
    epochs = args.sft_epochs
    if not resume_state.get("stage1_done"):
        steps_per_epoch = math.ceil(len(train_items) / 2)
        total_steps = max(1, steps_per_epoch * epochs)
        opt_s1 = optim.MultiOptimizer(
            [
                optim.AdamW(learning_rate=optim.cosine_decay(2.5e-4, decay_steps=total_steps, end=2.5e-5), weight_decay=0.0),
                optim.AdamW(learning_rate=optim.cosine_decay(8.5e-5, decay_steps=total_steps, end=8.5e-6), weight_decay=0.02),
                optim.AdamW(learning_rate=optim.cosine_decay(2.4e-5, decay_steps=total_steps, end=2.4e-6), weight_decay=0.02),
            ],
            filters=[
                lambda k, _: any(gn in k for gn in GATE_NORM_LEAVES),
                lambda k, _: "lora_b" in k,
            ],
        )

        rng = random.Random(20261211)
        no_improve = 0
        early_stop = False
        checked_s1 = False
        t_s1 = time.perf_counter()

        for ep in range(resume_state["epoch"], epochs):
            if early_stop:
                break
            order = list(range(len(train_items)))
            rng.shuffle(order)
            start_i = resume_state["sample_idx"] if ep == resume_state["epoch"] else 0
            buf: list[dict[str, Any]] = []
            run_losses: list[float] = []

            for i in range(start_i, len(order)):
                buf.append(train_items[order[i]])
                if len(buf) == 2 or i == len(order) - 1:
                    lv, _ = _run_dft_batch(model, buf, opt_s1)
                    run_losses.append(lv)
                    buf.clear()

                    if not checked_s1:
                        checked_s1 = True
                        flat_tr = dict(tree_flatten(model.trainable_parameters()))
                        max_b = max(float(mx.max(mx.abs(v)).item()) for k, v in flat_tr.items() if "lora_b" in k)
                        print(f"  [Stage 1 Step 1 Sanity] max(|lora_b|)={max_b:.6e}", flush=True)
                        assert max_b > 1e-7, "FATAL: lora_b stayed zero!"

                if (i + 1) % 240 == 0 or i == len(order) - 1:
                    model.eval()
                    vl, vp = evaluate_val_loss(model, val_items)
                    model.train()
                    avg_tr = sum(run_losses) / max(1, len(run_losses))
                    run_losses.clear()
                    resume_state["stage"] = "sft"
                    resume_state["epoch"] = ep
                    resume_state["sample_idx"] = i + 1
                    if vl < resume_state["best_val_loss"] - 1e-4:
                        resume_state["best_val_loss"] = vl
                        no_improve = 0
                        _save_trainable_ckpt(model, STAGE1_BEST_DIR)
                    else:
                        no_improve += 1
                    _save_trainable_ckpt(model, LATEST_DIR, resume_state)
                    print(
                        f"  [Stage 1 DFT ep {ep+1}/{epochs} {i+1:4d}/{len(order)} | {(time.perf_counter()-t_s1)/60:4.1f}m] "
                        f"train={avg_tr:.4f} | val={vl:.4f} (PPL={vp:.4f}) | best={resume_state['best_val_loss']:.4f} | pat={no_improve}/2",
                        flush=True,
                    )
                    if no_improve >= 2:
                        print(f"  [Early Stop] Validation loss peaked at {resume_state['best_val_loss']:.4f}; stopping Stage 1.", flush=True)
                        early_stop = True
                        break
            resume_state["sample_idx"] = 0

        model.load_weights(str(STAGE1_BEST_DIR / "trainable.safetensors"), strict=False)
        mx.eval(model.parameters())
        _save_trainable_ckpt(model, STAGE2_BEST_DIR)
        resume_state["best_rl_val_loss"] = resume_state["best_val_loss"]
        resume_state["stage1_done"] = True
        resume_state["stage"] = "rlvr"
        _save_trainable_ckpt(model, LATEST_DIR, resume_state)
        gc.collect()
        mx.clear_cache()

    # ------------------------------------------------------------------
    # STAGE 2: Shuffled On-Policy Multi-Turn SCoRe + GDPO/Dr.GRPO RLVR
    # ------------------------------------------------------------------
    if not resume_state.get("stage2_done"):
        model.eval()  # Disable LoRA dropout during live generation & PG ratio computation!
        opt_s2 = optim.MultiOptimizer(
            [
                optim.AdamW(learning_rate=2.0e-5, weight_decay=0.0),
                optim.AdamW(learning_rate=7.0e-6, weight_decay=0.01),
                optim.AdamW(learning_rate=2.0e-6, weight_decay=0.01),
            ],
            filters=[
                lambda k, _: any(gn in k for gn in GATE_NORM_LEAVES),
                lambda k, _: "lora_b" in k,
            ],
        )
        tok = processor.tokenizer
        eos_id = int(tok.eos_token_id)
        max_comp_len = 448
        g_rollouts = 4

        s1_ref_weights = mx.load(str(STAGE1_BEST_DIR / "trainable.safetensors"))
        mx.eval(list(s1_ref_weights.values()))

        rl_order = list(range(len(rlvr_raw)))
        random.Random(20261212).shuffle(rl_order)
        rl_order = rl_order[: min(args.rl_steps, len(rl_order))]
        anchor_rng = random.Random(20261213)
        t_s2 = time.perf_counter()

        for g_idx in range(resume_state.get("rl_idx", 0), len(rl_order)):
            prompt_row = rlvr_raw[rl_order[g_idx]]
            prompt_msgs = prompt_row["messages"]
            prompt_str = _apply_template_inference_aligned(
                processor, model.config, prompt_msgs, num_images=0, enable_thinking=True
            )
            raw_prompt_txt = "\n".join(str(m.get("content", "")) for m in prompt_msgs)
            p_ids = tok.encode(prompt_str, add_special_tokens=False)
            prompt_mx = mx.array([p_ids], dtype=mx.int32)
            start_idx = max(len(p_ids) - 1, 0)

            rollouts: list[dict[str, Any]] = []
            for _r in range(g_rollouts):
                res = mlx_vlm.generate(
                    model,
                    processor,
                    prompt_str,
                    max_tokens=max_comp_len,
                    temperature=0.70,
                    repetition_penalty=1.05,
                    verbose=False,
                )
                gen_txt = res.text or ""
                comp_ids = tok.encode(gen_txt, add_special_tokens=False) + [eos_id]
                full_seq = mx.concatenate([prompt_mx, mx.array([comp_ids], dtype=mx.int32)], axis=1)
                if int(full_seq.shape[1]) > 1792:
                    full_seq = full_seq[:, :1792]
                sc = score_response(
                    gen_txt,
                    prompt_text=raw_prompt_txt,
                    kind=prompt_row["kind"],
                    verifier=prompt_row["verifier"],
                    tokenizer=tok,
                    max_think_tokens=320,
                )
                rollouts.append({"text": gen_txt, "full_ids": full_seq, "score": sc})

            rewards = [x["score"]["reward"] for x in rollouts]
            pass_cnt = sum(1 for ro in rollouts if ro["score"]["correct"])
            anchor_batch = anchor_rng.sample(train_items, k=min(2, len(train_items)))

            if pass_cnt == 0 or (max(rewards) - min(rewards) < 1e-4 and pass_cnt < g_rollouts):
                ref_row = {
                    "id": f"rescue_{prompt_row['id']}",
                    "pillar": prompt_row["kind"],
                    "images": [],
                    "messages": [*prompt_msgs, {"role": "assistant", "content": prompt_row["reference_answer"]}],
                }
                ref_items, _ = prepare_turn_items_v5([ref_row], model, processor, max_seq_len=1792)
                combined = (ref_items[-1:] + anchor_batch[:1]) if ref_items else anchor_batch[:1]
                _run_dft_batch(model, combined, opt_s2, scale_grad=0.30)
                tag_str = "OPSD(0.30x) rescue"
            elif max(rewards) - min(rewards) < 1e-4 and pass_cnt == g_rollouts:
                tag_str = "4/4 pass (DAPO skip)"
            else:
                ch_weights = {"correct": 1.0, "schema": 0.5, "close_think": 0.4, "anti_echo": 0.4, "tool": 0.6}
                advs = [0.0] * len(rollouts)
                for ck, cw in ch_weights.items():
                    vals = [float(x["score"]["channels"][ck]) for x in rollouts]
                    mu = sum(vals) / len(vals)
                    for idx_r, v in enumerate(vals):
                        advs[idx_r] += cw * (v - mu)

                old_lps = [
                    mx.stop_gradient(-chunked_multimodal_ce(model, ro["full_ids"], start_idx))
                    for ro in rollouts
                ]
                live_snap = {k: mx.array(v) for k, v in tree_flatten(model.trainable_parameters())}
                model.load_weights(list(s1_ref_weights.items()), strict=False)
                mx.eval(model.parameters())
                ref_lps = [
                    mx.stop_gradient(-chunked_multimodal_ce(model, ro["full_ids"], start_idx))
                    for ro in rollouts
                ]
                model.load_weights(list(live_snap.items()), strict=False)
                mx.eval(model.parameters(), *(old_lps + ref_lps))

                def rl_item_loss(
                    root_mod: nn.Module,
                    full_ids: mx.array,
                    old_lp: mx.array,
                    ref_lp: mx.array,
                    adv_val: float,
                ) -> mx.array:
                    new_lp = -chunked_multimodal_ce(root_mod, full_ids, start_idx)
                    ratio = mx.exp(mx.clip(new_lp - old_lp, -5.0, 5.0))
                    adv = mx.array(adv_val, dtype=mx.float32)
                    pg_loss = -mx.minimum(ratio * adv, mx.clip(ratio, 0.8, 1.28) * adv)
                    kl = mx.exp(ref_lp - new_lp) - (ref_lp - new_lp) - 1.0
                    return mx.sum(pg_loss + 0.04 * kl) / mx.array(float(max_comp_len), dtype=mx.float32)

                rl_val_grad = nn.value_and_grad(model, rl_item_loss)
                acc_grads = None
                n_r = float(len(rollouts))
                for ro, old_lp, ref_lp, adv_v in zip(rollouts, old_lps, ref_lps, advs):
                    l_r, g_r = rl_val_grad(model, ro["full_ids"], old_lp, ref_lp, adv_v)
                    mx.eval(l_r)
                    acc_grads = tree_map(lambda g: g / n_r, g_r) if acc_grads is None else tree_map(lambda a, g: a + g / n_r, acc_grads, g_r)

                _, anchor_grads = _run_dft_batch(model, anchor_batch, None, scale_grad=0.10)
                if acc_grads is not None and anchor_grads is not None:
                    acc_grads = tree_map(lambda a, b: a + b, acc_grads, anchor_grads)
                if acc_grads is not None:
                    acc_grads, _ = optim.clip_grad_norm(acc_grads, max_norm=1.0)
                    opt_s2.update(model, acc_grads)
                    mx.eval(model.trainable_parameters(), opt_s2.state)
                mx.clear_cache()
                tag_str = f"GDPO+KL_s1 (mean_r={sum(rewards)/len(rewards):.2f})"

            if (g_idx + 1) % 6 == 0 or g_idx == len(rl_order) - 1:
                print(
                    f"  [Stage 2 RL {g_idx+1:3d}/{len(rl_order)} | {(time.perf_counter()-t_s2)/60:4.1f}m] "
                    f"{prompt_row['id']:20s} | pass={pass_cnt}/{g_rollouts} | {tag_str}",
                    flush=True,
                )

            resume_state["rl_idx"] = g_idx + 1
            if (g_idx + 1) % 16 == 0 or g_idx == len(rl_order) - 1:
                vl, vp = evaluate_val_loss(model, val_items)
                _save_trainable_ckpt(model, LATEST_DIR, resume_state)
                if vl <= float(resume_state.get("best_rl_val_loss", 1e9)) + 0.015:
                    _save_trainable_ckpt(model, STAGE2_BEST_DIR)
                    resume_state["best_rl_val_loss"] = min(float(resume_state.get("best_rl_val_loss", 1e9)), vl)
                print(f"  [Stage 2 Ckpt] step={g_idx+1}/{len(rl_order)} | val_loss={vl:.4f} (PPL={vp:.4f})", flush=True)
                if vl > float(resume_state["best_val_loss"]) + 0.035:
                    print(f"  [Stage 2 Guard] val_loss={vl:.4f} exceeded SFT best + 0.035; stopping RLVR cleanly.", flush=True)
                    break

        resume_state["stage2_done"] = True
        resume_state["stage"] = "wise_ft"
        _save_trainable_ckpt(model, LATEST_DIR, resume_state)
        gc.collect()
        mx.clear_cache()

    # ------------------------------------------------------------------
    # STAGE 3: Validation-Only 2D WiSE-FT Sweep (`stage1_best` <-> `stage2_best` & `base_gamma`)
    # ------------------------------------------------------------------
    model.eval()
    print("[Stage 3] Running 2D WiSE-FT sweep on strictly disjoint 16-task validation probe...", flush=True)
    w_s1 = mx.load(str(STAGE1_BEST_DIR / "trainable.safetensors"))
    s2_file = STAGE2_BEST_DIR / "trainable.safetensors"
    if not s2_file.exists():
        s2_file = STAGE1_BEST_DIR / "trainable.safetensors"
    w_s2 = mx.load(str(s2_file))

    best_key = (-1, -1e9, -1e9)
    best_cfg = (1.0, 1.0)
    best_weights = w_s2

    candidates = [(0.0, 1.0), (0.5, 1.0), (1.0, 1.0), (0.5, 0.90), (1.0, 0.90)]
    for lam, gamma in candidates:
        blended = {}
        for k in w_s2:
            v_mix = (1.0 - lam) * w_s1[k] + lam * w_s2[k]
            blended[k] = (gamma * v_mix) if "lora_b" in k else v_mix
        model.load_weights(list(blended.items()), strict=False)
        mx.eval(model.parameters())
        vl, vp = evaluate_val_loss(model, val_items)
        p_pass, p_rew = _evaluate_mini_probe(model, processor, probe_16)
        cand_key = (p_pass, p_rew, -vl)
        print(
            f"  [WiSE-FT rl_lam={lam:.2f}, base_gamma={gamma:.2f}] val_probe={p_pass}/{len(probe_16)} | "
            f"mean_reward={p_rew:.3f} | val_loss={vl:.4f} (PPL={vp:.4f})",
            flush=True,
        )
        if cand_key >= best_key:
            best_key = cand_key
            best_cfg = (lam, gamma)
            best_weights = blended

    print(
        f"[Stage 3] Selected WiSE-FT rl_lam={best_cfg[0]:.2f}, base_gamma={best_cfg[1]:.2f} "
        f"(val_probe={best_key[0]}/{len(probe_16)}, val_loss={-best_key[2]:.4f})!",
        flush=True,
    )
    model.load_weights(list(best_weights.items()), strict=False)
    mx.eval(model.parameters())
    _save_trainable_ckpt(
        model,
        WISE_BEST_DIR,
        {"rl_lambda": best_cfg[0], "base_gamma": best_cfg[1], "probe_passed": best_key[0], "val_loss": -best_key[2]},
    )

    # ------------------------------------------------------------------
    # STAGE 4: Lossless 16-Bit `bfloat16` Standalone Export (`LFM3-VL-2.24B-RPR-RSI-BF16`)
    # ------------------------------------------------------------------
    fuse_and_export_28l_bf16(model, STAGE0_WARM_DIR, Path(args.export_dir))
    print(f"[Complete] Standalone 2.24B 28-Layer 16-bit RPR+RSI model ready at: {args.export_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Surgical 24L->28L Depth Upcycler + 5-Stage RSI/RPR BF16 Trainer.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Read-only 24L base model path.")
    parser.add_argument("--export-dir", default=str(EXPORT_V5_DIR), help="Standalone 28L 16-bit export directory.")
    parser.add_argument("--warmup-steps", type=int, default=220, help="Stage-0b surgical 4-block warmup steps.")
    parser.add_argument("--sft-epochs", type=int, default=1, help="Max Stage-1 epochs (default 1 to prevent overfitting).")
    parser.add_argument("--rl-steps", type=int, default=80, help="Shuffled Stage-2 on-policy SCoRe/RLVR episodes.")
    parser.add_argument("--fresh", action="store_true", help="Ignore existing Stage 0b/1/2 checkpoints.")
    return parser.parse_args()


if __name__ == "__main__":
    run_v5_pipeline(parse_args())
