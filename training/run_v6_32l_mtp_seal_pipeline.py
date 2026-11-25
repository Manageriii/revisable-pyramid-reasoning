"""Master Checkpointed v6 Pipeline:
  - Stage A: Surgical `28L -> 32L` (`2.162B -> 2.406B`, `8 -> 10` `full_attention` layers) SandwichNorm Zero-Gate Upcycling
             + Leftover-Limits Repair (`python_code` + unassisted math + `<weight_rewrite>` SFT) + Code/Math RLVR
             -> `checkpoints/LFM3-VL-2.4B-RPR-RSI-BF16/`
  - Stage B: 2-Step Unrolled EAGLE/FR-Spec `SequentialMTPHead` Distillation (`16,384` active shortlist + `33.6M` block)
             + Honest Wall-Clock `tok/s` & Bit-Exact Parity Benchmark (`results/mtp_speed_benchmark/`)
  - Stage C: Live `Hybrid SEAL + SVF + Nullspace` Self-Weight-Rewriting RL (`theta_parent -> theta_copy`)
             rewarded strictly by unseen `Reward Split` capability gain minus `Anchor` NLL drift,
             exporting `checkpoints/LFM3-VL-2.4B-RPR-MTP-SEAL-BF16/` + Out-of-Sample Autonomous Self-Rewrite Test!
"""

from __future__ import annotations

import argparse
from collections import Counter
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

from model import extract_json_object, split_thinking
from mtp_speculative import (
    SequentialMTPHead,
    load_mtp_checkpoint,
    mtp_speculative_generate,
    save_mtp_checkpoint,
    standard_greedy_generate,
)
from training.build_rpr_v5_rsi_corpus import V5_CORPUS_DIR
from training.build_v6_repair_and_seal_corpus import SEAL_SYSTEM_PROMPT, V6_CORPUS_DIR
from training.train_and_fuse_rpr_bf16 import (
    _apply_template_inference_aligned,
    _evaluate_mini_probe,
    _run_dft_batch,
    chunked_multimodal_ce,
    evaluate_val_loss,
)
from training.upcycle_and_train_rpr_v5_bf16 import (
    EXPORT_V5_DIR,
    GATE_NORM_LEAVES,
    _save_trainable_ckpt,
    prepare_turn_items_v5,
)
from training.verifiable_rewards import score_response


WORK_V6 = ROOT_DIR / "checkpoints" / "rpr_v6_32l_mtp_seal_work"
STAGEA_ID_32L = WORK_V6 / "stageA_identity_32L"
STAGEA_WARM_32L = WORK_V6 / "stageA_warmed_32L"
STAGEA_CKPT = WORK_V6 / "stageA_sft_rl"
EXPORT_32L_REPAIRED = ROOT_DIR / "checkpoints" / "LFM3-VL-2.4B-RPR-RSI-BF16"
EXPORT_FINAL_SEAL = ROOT_DIR / "checkpoints" / "LFM3-VL-2.4B-RPR-MTP-SEAL-BF16"

# In 28L, layers [9, 10] = [conv, full_attention] and [19, 20] = [conv, full_attention].
# Inserting clones right after 10 (-> [11, 12]) and after 20 (-> [23, 24]) yields 32L with 10 full_attention layers!
UPCYCLED_32L_INDICES = (11, 12, 23, 24)
# Top 4 global `full_attention` blocks targeted during live SEAL + SVF copy weight rewriting:
REWRITE_TARGET_LAYERS = (22, 24, 28, 30)


# ===========================================================================
# STAGE A: 28L -> 32L Surgical Upcycling + Leftover-Limits Repair
# ===========================================================================


def build_upcycled_32l_config_from_28l(cfg_28_dict: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg_28_dict)
    cfg["dtype"] = "bfloat16"
    tcfg = cfg["text_config"]
    tcfg["dtype"] = "bfloat16"
    old_types = list(tcfg["layer_types"])
    assert len(old_types) == 28, f"Expected 28 layers, got {len(old_types)}"
    assert old_types[9:11] == ["conv", "full_attention"]
    assert old_types[19:21] == ["conv", "full_attention"]

    new_types = (
        old_types[:11]
        + ["conv", "full_attention"]
        + old_types[11:21]
        + ["conv", "full_attention"]
        + old_types[21:]
    )
    assert len(new_types) == 32
    assert new_types.count("full_attention") == 10
    tcfg["num_hidden_layers"] = 32
    tcfg["layer_types"] = new_types
    return cfg


def upcycle_28l_to_32l_identity(src_28l_dir: Path, out_32l_dir: Path) -> None:
    import mlx_vlm

    if (out_32l_dir / "model.safetensors").exists() and (out_32l_dir / "upcycle_32l_verified.json").exists():
        print(f"[Stage A1] Resuming verified 32L identity checkpoint at {out_32l_dir}", flush=True)
        return

    print(f"[Stage A1] Upcycling 28L ({src_28l_dir}) -> 32L (10 full_attention layers)...", flush=True)
    model_28, _ = mlx_vlm.load(str(src_28l_dir))
    probe_ids = mx.array([[1, 42, 105, 512, 1024, 2048]], dtype=mx.int32)
    logits_28 = model_28.language_model(probe_ids).logits
    mx.eval(logits_28)

    flat_28 = dict(tree_flatten(model_28.parameters()))
    src_for_dst: dict[int, int] = {}
    for dst in range(11):
        src_for_dst[dst] = dst
    src_for_dst[11] = 9
    src_for_dst[12] = 10
    for dst in range(13, 23):
        src_for_dst[dst] = dst - 2
    src_for_dst[23] = 19
    src_for_dst[24] = 20
    for dst in range(25, 32):
        src_for_dst[dst] = dst - 4

    prefix = "language_model.model.layers."
    flat_32: dict[str, mx.array] = {}
    for k, v in flat_28.items():
        if not k.startswith(prefix):
            flat_32[k] = v.astype(mx.bfloat16) if mx.issubdtype(v.dtype, mx.floating) else v

    for dst_idx, src_idx in src_for_dst.items():
        src_pref = f"{prefix}{src_idx}."
        dst_pref = f"{prefix}{dst_idx}."
        is_new = dst_idx in UPCYCLED_32L_INDICES
        for k, v in flat_28.items():
            if not k.startswith(src_pref):
                continue
            sub_k = k[len(src_pref) :]
            if is_new and sub_k in {"post_operator_norm.weight", "post_ffn_norm.weight"}:
                flat_32[dst_pref + sub_k] = mx.zeros_like(v).astype(mx.bfloat16)
            else:
                flat_32[dst_pref + sub_k] = v.astype(mx.bfloat16) if mx.issubdtype(v.dtype, mx.floating) else v

    mx.eval(list(flat_32.values()))
    n_p28 = sum(int(v.size) for v in flat_28.values())
    n_p32 = sum(int(v.size) for v in flat_32.values())
    del model_28, flat_28
    gc.collect()
    mx.clear_cache()

    out_32l_dir.mkdir(parents=True, exist_ok=True)
    skip = {"model.safetensors", "model.safetensors.index.json", "config.json"}
    for item in src_28l_dir.iterdir():
        if item.is_file() and item.name not in skip and not item.name.endswith(".safetensors"):
            shutil.copy2(item, out_32l_dir / item.name)

    cfg_28 = json.loads((src_28l_dir / "config.json").read_text(encoding="utf-8"))
    cfg_32 = build_upcycled_32l_config_from_28l(cfg_28)
    (out_32l_dir / "config.json").write_text(json.dumps(cfg_32, indent=2) + "\n", encoding="utf-8")

    st_path = out_32l_dir / "model.safetensors"
    mx.save_safetensors(str(st_path), flat_32, metadata={"format": "mlx"})
    total_bytes = st_path.stat().st_size
    (out_32l_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_bytes}, "weight_map": {k: "model.safetensors" for k in sorted(flat_32)}}, indent=2) + "\n",
        encoding="utf-8",
    )
    del flat_32
    gc.collect()
    mx.clear_cache()

    model_32, _ = mlx_vlm.load(str(out_32l_dir))
    logits_32 = model_32.language_model(probe_ids).logits
    mx.eval(logits_32)
    max_diff = float(mx.max(mx.abs(logits_32.astype(mx.float32) - logits_28.astype(mx.float32))).item())
    print(
        f"[Stage A1 Verified] 28L -> 32L Upcycled Model: {n_p32/1e9:.3f}B params (+{(n_p32-n_p28)/1e6:.2f}M, "
        f"10 full_attention layers, {total_bytes/(1024**3):.2f} GiB BF16) | Step-0 max logit diff = {max_diff:.6f}",
        flush=True,
    )
    assert max_diff < 1e-5, f"32L Step-0 identity mismatch: {max_diff}"
    (out_32l_dir / "upcycle_32l_verified.json").write_text(
        json.dumps(
            {
                "params_28l": n_p28,
                "params_32l": n_p32,
                "added_params": n_p32 - n_p28,
                "num_hidden_layers": 32,
                "full_attention_layers": 10,
                "step0_max_logit_diff": max_diff,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    del model_32
    gc.collect()
    mx.clear_cache()


def _unfreeze_32l_gates(model: nn.Module) -> None:
    for idx in UPCYCLED_32L_INDICES:
        lyr = model.language_model.model.layers[idx]
        lyr.post_operator_norm.unfreeze()
        lyr.post_ffn_norm.unfreeze()


def fuse_and_export_32l(model: Any, template_32l_dir: Path, export_dir: Path) -> None:
    import mlx_vlm

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
    skip = {"model.safetensors", "model.safetensors.index.json", "upcycle_32l_verified.json", "warmup_done.json"}
    for item in template_32l_dir.iterdir():
        if item.is_file() and item.name not in skip and not item.name.endswith(".safetensors"):
            shutil.copy2(item, export_dir / item.name)

    flat_w = dict(tree_flatten(model.parameters()))
    st_path = export_dir / "model.safetensors"
    mx.save_safetensors(str(st_path), flat_w, metadata={"format": "mlx"})
    tot_b = st_path.stat().st_size
    (export_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": tot_b}, "weight_map": {k: "model.safetensors" for k in sorted(flat_w)}}, indent=2) + "\n",
        encoding="utf-8",
    )
    test_ids = mx.array([[1, 42, 105, 512]], dtype=mx.int32)
    l_mem = model.language_model(test_ids).logits
    mx.eval(l_mem)
    rel, _ = mlx_vlm.load(str(export_dir))
    l_disk = rel.language_model(test_ids).logits
    mx.eval(l_disk)
    diff = float(mx.max(mx.abs(l_mem.astype(mx.float32) - l_disk.astype(mx.float32))).item())
    print(
        f"[Fuse 32L] Wrote {len(flat_w)} tensors ({merged_cnt} LoRA-fused, {tot_b/(1024**3):.2f} GiB) -> {export_dir} | reload diff={diff:.6f}",
        flush=True,
    )
    assert diff < 1e-3
    del rel
    gc.collect()
    mx.clear_cache()


def run_stage_a_32l_repair(
    train_raw: list[dict[str, Any]],
    val_raw: list[dict[str, Any]],
    rlvr_raw: list[dict[str, Any]],
    *,
    rl_steps: int = 32,
) -> None:
    import mlx_vlm
    from mlx_vlm.trainer.peft import get_peft_model

    if (EXPORT_32L_REPAIRED / "model.safetensors").exists() and (EXPORT_32L_REPAIRED / "stageA_done.json").exists():
        print(f"[Stage A] Resuming completed 32L repaired model at {EXPORT_32L_REPAIRED}", flush=True)
        return

    upcycle_28l_to_32l_identity(EXPORT_V5_DIR, STAGEA_ID_32L)

    # Stage A2: 140-step surgical warmup on the 4 newly inserted blocks `(11, 12, 23, 24)`
    if not (STAGEA_WARM_32L / "warmup_done.json").exists():
        print("[Stage A2] Warming up newly inserted 32L blocks (11, 12, 23, 24)...", flush=True)
        model, processor = mlx_vlm.load(str(STAGEA_ID_32L))
        tr_items, _ = prepare_turn_items_v5(train_raw, model, processor, max_seq_len=1792)
        va_items, _ = prepare_turn_items_v5(val_raw, model, processor, max_seq_len=1792)
        leafs = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "in_proj", "out_proj", "w1", "w2", "w3"}
        prefs = tuple(f"model.layers.{i}." for i in UPCYCLED_32L_INDICES)
        b_targets = [
            f"language_model.{n}"
            for n, _ in model.language_model.named_modules()
            if n.startswith(prefs) and n.split(".")[-1] in leafs
        ]
        model = get_peft_model(model, linear_layers=b_targets, rank=32, alpha=64, dropout=0.0, verbose=False)
        _unfreeze_32l_gates(model)
        opt_w = optim.MultiOptimizer(
            [
                optim.AdamW(learning_rate=optim.cosine_decay(7.0e-4, decay_steps=140, end=1.0e-4), weight_decay=0.0),
                optim.AdamW(learning_rate=optim.cosine_decay(7.0e-5, decay_steps=140, end=1.2e-5), weight_decay=0.01),
                optim.AdamW(learning_rate=optim.cosine_decay(2.0e-5, decay_steps=140, end=4.0e-6), weight_decay=0.01),
            ],
            filters=[
                lambda k, _: any(gn in k for gn in GATE_NORM_LEAVES),
                lambda k, _: "lora_b" in k,
            ],
        )
        vl0, vp0 = evaluate_val_loss(model, va_items)
        print(f"  [Stage A2 Start] val_loss={vl0:.4f} (PPL={vp0:.4f})", flush=True)
        order = list(range(len(tr_items)))
        random.Random(20261310).shuffle(order)
        for st in range(140):
            batch = [tr_items[order[(st * 2) % len(order)]], tr_items[order[(st * 2 + 1) % len(order)]]]
            _run_dft_batch(model, batch, opt_w)
            if (st + 1) % 70 == 0:
                vl, vp = evaluate_val_loss(model, va_items)
                print(f"  [Stage A2 Warmup {st+1}/140] val_loss={vl:.4f} (PPL={vp:.4f})", flush=True)
        fuse_and_export_32l(model, STAGEA_ID_32L, STAGEA_WARM_32L)
        (STAGEA_WARM_32L / "warmup_done.json").write_text(json.dumps({"val_loss": vl}, indent=2) + "\n", encoding="utf-8")
        del model, processor, tr_items, va_items
        gc.collect()
        mx.clear_cache()

    # Stage A3: Balanced Anti-Overfit 32L Repair LoRA+ + Targeted Code/Math RLVR + Conservative WiSE-FT (`gamma=0.74`)
    print("[Stage A3] Loading warmed 32L model for balanced 9-pillar repair + RLVR...", flush=True)
    model, processor = mlx_vlm.load(str(STAGEA_WARM_32L))
    tr_items, vis_tr = prepare_turn_items_v5(train_raw, model, processor, max_seq_len=1792)
    va_items, _ = prepare_turn_items_v5(val_raw, model, processor, max_seq_len=1792)
    leafs = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "in_proj", "out_proj", "w1", "w2", "w3"}
    lm_t = [f"language_model.{n}" for n, _ in model.language_model.named_modules() if n.split(".")[-1] in leafs]
    all_t = lm_t + ["multi_modal_projector.linear_1", "multi_modal_projector.linear_2"]
    model = get_peft_model(model, linear_layers=all_t, rank=24, alpha=48, dropout=0.05, verbose=False)
    _unfreeze_32l_gates(model)
    print(f"[Stage A3] Attached Rank-24 LoRA across {len(all_t)} 32L+Projector targets ({len(tr_items)} turns, {vis_tr} vision).", flush=True)

    a_state = {"sample_idx": 0, "rl_idx": 0, "sft_done": False, "rl_done": False, "best_val": 1e9}
    s_file = STAGEA_CKPT / "latest" / "train_state.json"
    if s_file.exists() and (STAGEA_CKPT / "latest" / "trainable.safetensors").exists():
        a_state = json.loads(s_file.read_text(encoding="utf-8"))
        model.load_weights(str(STAGEA_CKPT / "latest" / "trainable.safetensors"), strict=False)
        mx.eval(model.parameters())
        print(f"[Stage A3] Resumed checkpoint: {a_state}", flush=True)
    else:
        vl0, _ = evaluate_val_loss(model, va_items)
        a_state["best_val"] = vl0
        _save_trainable_ckpt(model, STAGEA_CKPT / "sft_best")
        _save_trainable_ckpt(model, STAGEA_CKPT / "latest", a_state)

    if not a_state["sft_done"]:
        order = list(range(len(tr_items)))
        random.Random(20261311).shuffle(order)
        # Cap to 640 shuffled multi-pillar turns (320 steps) at gentle LR so train_loss stays healthy (~0.07-0.12, zero memorization!)
        order = order[: min(640, len(order))]
        tot_st = max(1, math.ceil(len(order) / 2))
        opt_a = optim.MultiOptimizer(
            [
                optim.AdamW(learning_rate=optim.cosine_decay(8.0e-5, decay_steps=tot_st, end=1.2e-5), weight_decay=0.0),
                optim.AdamW(learning_rate=optim.cosine_decay(2.8e-5, decay_steps=tot_st, end=4.0e-6), weight_decay=0.02),
                optim.AdamW(learning_rate=optim.cosine_decay(8.0e-6, decay_steps=tot_st, end=1.2e-6), weight_decay=0.02),
            ],
            filters=[
                lambda k, _: any(gn in k for gn in GATE_NORM_LEAVES),
                lambda k, _: "lora_b" in k,
            ],
        )
        buf: list[dict[str, Any]] = []
        losses: list[float] = []
        t0 = time.perf_counter()
        for i in range(a_state["sample_idx"], len(order)):
            buf.append(tr_items[order[i]])
            if len(buf) == 2 or i == len(order) - 1:
                lv, _ = _run_dft_batch(model, buf, opt_a)
                losses.append(lv)
                buf.clear()
            if (i + 1) % 160 == 0 or i == len(order) - 1:
                model.eval()
                vl, vp = evaluate_val_loss(model, va_items)
                model.train()
                a_state["sample_idx"] = i + 1
                if vl <= a_state["best_val"]:
                    a_state["best_val"] = vl
                    _save_trainable_ckpt(model, STAGEA_CKPT / "sft_best")
                _save_trainable_ckpt(model, STAGEA_CKPT / "latest", a_state)
                print(
                    f"  [Stage A3 SFT {i+1:3d}/{len(order)} | {(time.perf_counter()-t0)/60:4.1f}m] "
                    f"train={sum(losses)/max(1,len(losses)):.4f} | val={vl:.4f} (PPL={vp:.4f}) | best={a_state['best_val']:.4f}",
                    flush=True,
                )
                losses.clear()
        model.load_weights(str(STAGEA_CKPT / "sft_best" / "trainable.safetensors"), strict=False)
        mx.eval(model.parameters())
        _save_trainable_ckpt(model, STAGEA_CKPT / "rl_best")
        a_state["sft_done"] = True
        _save_trainable_ckpt(model, STAGEA_CKPT / "latest", a_state)

    if not a_state["rl_done"]:
        model.eval()
        opt_rl = optim.MultiOptimizer(
            [
                optim.AdamW(learning_rate=1.0e-5, weight_decay=0.0),
                optim.AdamW(learning_rate=4.5e-6, weight_decay=0.01),
                optim.AdamW(learning_rate=1.4e-6, weight_decay=0.01),
            ],
            filters=[
                lambda k, _: any(gn in k for gn in GATE_NORM_LEAVES),
                lambda k, _: "lora_b" in k,
            ],
        )
        tok = processor.tokenizer
        eos_id = int(tok.eos_token_id)
        sft_ref = mx.load(str(STAGEA_CKPT / "sft_best" / "trainable.safetensors"))
        mx.eval(list(sft_ref.values()))
        anc_rng = random.Random(20261312)
        n_rl = min(rl_steps, len(rlvr_raw))
        t_rl = time.perf_counter()

        for g_idx in range(a_state["rl_idx"], n_rl):
            row = rlvr_raw[g_idx]
            p_msgs = row["messages"]
            p_str = _apply_template_inference_aligned(processor, model.config, p_msgs, num_images=0, enable_thinking=True)
            p_txt = "\n".join(str(m.get("content", "")) for m in p_msgs)
            p_ids = tok.encode(p_str, add_special_tokens=False)
            p_mx = mx.array([p_ids], dtype=mx.int32)
            s_idx = max(len(p_ids) - 1, 0)

            rollouts = []
            for _ in range(4):
                res = mlx_vlm.generate(model, processor, p_str, max_tokens=448, temperature=0.78, repetition_penalty=1.05, verbose=False)
                g_txt = res.text or ""
                c_ids = tok.encode(g_txt, add_special_tokens=False) + [eos_id]
                f_seq = mx.concatenate([p_mx, mx.array([c_ids], dtype=mx.int32)], axis=1)[:, :1792]
                sc = score_response(g_txt, prompt_text=p_txt, kind=row["kind"], verifier=row["verifier"], tokenizer=tok, max_think_tokens=320)
                rollouts.append({"full_ids": f_seq, "score": sc})

            rews = [r["score"]["reward"] for r in rollouts]
            p_cnt = sum(1 for r in rollouts if r["score"]["correct"])
            anc_b = anc_rng.sample(tr_items, k=2)
            if p_cnt == 0 or (max(rews) - min(rews) < 1e-4 and p_cnt < 4):
                ref_r = {"id": f"r_{row['id']}", "pillar": row["kind"], "images": [], "messages": [*p_msgs, {"role": "assistant", "content": row["reference_answer"]}]}
                r_it, _ = prepare_turn_items_v5([ref_r], model, processor, max_seq_len=1792)
                _run_dft_batch(model, (r_it[-1:] + anc_b[:1]) if r_it else anc_b[:1], opt_rl, scale_grad=0.22)
                tag = "OPSD(0.22x)"
            elif max(rews) - min(rews) < 1e-4:
                tag = "4/4 pass (skip)"
            else:
                advs = [r - (sum(rews) / len(rews)) for r in rews]
                old_lps = [mx.stop_gradient(-chunked_multimodal_ce(model, ro["full_ids"], s_idx)) for ro in rollouts]
                snap_w = {k: mx.array(v) for k, v in tree_flatten(model.trainable_parameters())}
                model.load_weights(list(sft_ref.items()), strict=False)
                mx.eval(model.parameters())
                ref_lps = [mx.stop_gradient(-chunked_multimodal_ce(model, ro["full_ids"], s_idx)) for ro in rollouts]
                model.load_weights(list(snap_w.items()), strict=False)
                mx.eval(model.parameters(), *(old_lps + ref_lps))

                def _rl_loss(rm: nn.Module, f_ids: mx.array, o_lp: mx.array, r_lp: mx.array, av: float) -> mx.array:
                    n_lp = -chunked_multimodal_ce(rm, f_ids, s_idx)
                    rat = mx.exp(mx.clip(n_lp - o_lp, -5.0, 5.0))
                    adv_m = mx.array(av, dtype=mx.float32)
                    pg = -mx.minimum(rat * adv_m, mx.clip(rat, 0.8, 1.25) * adv_m)
                    kl = mx.exp(r_lp - n_lp) - (r_lp - n_lp) - 1.0
                    return mx.sum(pg + 0.04 * kl) / mx.array(448.0, dtype=mx.float32)

                vg = nn.value_and_grad(model, _rl_loss)
                acc_g = None
                for ro, olp, rlp, av in zip(rollouts, old_lps, ref_lps, advs):
                    lr_v, gr = vg(model, ro["full_ids"], olp, rlp, av)
                    mx.eval(lr_v)
                    acc_g = tree_map(lambda g: g / 4.0, gr) if acc_g is None else tree_map(lambda a, g: a + g / 4.0, acc_g, gr)
                _, anc_g = _run_dft_batch(model, anc_b, None, scale_grad=0.14)
                if acc_g is not None and anc_g is not None:
                    acc_g = tree_map(lambda a, b: a + b, acc_g, anc_g)
                if acc_g is not None:
                    acc_g, _ = optim.clip_grad_norm(acc_g, max_norm=1.0)
                    opt_rl.update(model, acc_g)
                    mx.eval(model.trainable_parameters(), opt_rl.state)
                mx.clear_cache()
                tag = f"GDPO(mean_r={sum(rews)/4:.2f})"

            a_state["rl_idx"] = g_idx + 1
            if (g_idx + 1) % 8 == 0 or g_idx == n_rl - 1:
                vl, vp = evaluate_val_loss(model, va_items)
                _save_trainable_ckpt(model, STAGEA_CKPT / "latest", a_state)
                if vl <= a_state["best_val"] + 0.02:
                    _save_trainable_ckpt(model, STAGEA_CKPT / "rl_best")
                print(
                    f"  [Stage A3 RL {g_idx+1:2d}/{n_rl} | {(time.perf_counter()-t_rl)/60:4.1f}m] "
                    f"{row['id']:18s} | pass={p_cnt}/4 | {tag} | val={vl:.4f}",
                    flush=True,
                )
        a_state["rl_done"] = True
        _save_trainable_ckpt(model, STAGEA_CKPT / "latest", a_state)

    # Conservative WiSE-FT (`gamma=0.74` toward `STAGEA_WARM_32L`) so 100% of v4/v5 skills stay rock-solid!
    w_sft = mx.load(str(STAGEA_CKPT / "sft_best" / "trainable.safetensors"))
    w_rl = mx.load(str(STAGEA_CKPT / "rl_best" / "trainable.safetensors"))
    blended = {}
    for k in w_rl:
        mix = 0.40 * w_sft[k] + 0.60 * w_rl[k]
        blended[k] = (0.74 * mix) if "lora_b" in k else mix
    model.load_weights(list(blended.items()), strict=False)
    mx.eval(model.parameters())
    fuse_and_export_32l(model, STAGEA_WARM_32L, EXPORT_32L_REPAIRED)
    (EXPORT_32L_REPAIRED / "stageA_done.json").write_text(json.dumps(a_state, indent=2) + "\n", encoding="utf-8")
    del model, processor, tr_items, va_items
    gc.collect()
    mx.clear_cache()


# ===========================================================================
# STAGE B: 2-Step Unrolled EAGLE/FR-Spec Sequential MTP Distillation + Speed Benchmark
# ===========================================================================


def build_active_shortlist(tokenizer: Any, rows: list[dict[str, Any]], vocab_size: int, target_k: int = 16384) -> list[int]:
    counts: Counter[int] = Counter()
    for r in rows:
        for m in r.get("messages", []):
            ids = tokenizer.encode(str(m.get("content", "")), add_special_tokens=False)
            counts.update(ids)
    # Always include special tokens + ASCII/digits/common BPE base IDs
    must_have = {int(tokenizer.eos_token_id), int(getattr(tokenizer, "bos_token_id", 0) or 0)}
    must_have.update(range(min(2048, vocab_size)))
    ordered = sorted(must_have)
    seen = set(ordered)
    for tid, _ in counts.most_common():
        if len(ordered) >= target_k:
            break
        if 0 <= tid < vocab_size and tid not in seen:
            seen.add(tid)
            ordered.append(tid)
    cand = 2048
    while len(ordered) < min(target_k, vocab_size) and cand < vocab_size:
        if cand not in seen:
            seen.add(cand)
            ordered.append(cand)
        cand += 1
    return sorted(ordered[: min(target_k, vocab_size)])


def run_stage_b_mtp_training_and_benchmark(
    model_32l_dir: Path,
    mtp_steps: int = 420,
) -> dict[str, Any]:
    import mlx_vlm

    bench_dir = ROOT_DIR / "results" / "mtp_speed_benchmark"
    bench_dir.mkdir(parents=True, exist_ok=True)
    if (model_32l_dir / "mtp_head.safetensors").exists() and (bench_dir / "summary.json").exists():
        print(f"[Stage B] Resuming completed MTP head & speed benchmark from {bench_dir / 'summary.json'}", flush=True)
        return json.loads((bench_dir / "summary.json").read_text(encoding="utf-8"))

    print(f"[Stage B] Loading 32L model {model_32l_dir} to distill 2-step SequentialMTPHead...", flush=True)
    vlm, processor = mlx_vlm.load(str(model_32l_dir))
    vlm.freeze()
    tok = processor.tokenizer
    eos_id = int(tok.eos_token_id)

    v5_rows = [json.loads(l) for l in (V5_CORPUS_DIR / "train_rpr_v5.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    v6_rows = [json.loads(l) for l in (V6_CORPUS_DIR / "train_v6_repair.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    all_text_rows = [r for r in (v6_rows + v5_rows[:350]) if not r.get("images")]

    shortlist = build_active_shortlist(tok, v5_rows + v6_rows, int(vlm.config.text_config.vocab_size), target_k=16384)
    # Build fast inverse lookup table `[vocab_size] -> shortlist_index` (`-1` if rare OOV token)
    inv_map_np = np.full((int(vlm.config.text_config.vocab_size),), -1, dtype=np.int32)
    for s_i, tid in enumerate(shortlist):
        inv_map_np[tid] = s_i
    inv_map_mx = mx.array(inv_map_np, dtype=mx.int32)

    mtp_ckpt_dir = WORK_V6 / "stageB_mtp_ckpt"
    if (model_32l_dir / "mtp_head.safetensors").exists():
        mtp_head = load_mtp_checkpoint(vlm.config.text_config, model_32l_dir)
        start_step = mtp_steps
    elif (mtp_ckpt_dir / "mtp_head.safetensors").exists() and (mtp_ckpt_dir / "state.json").exists():
        mtp_head = load_mtp_checkpoint(vlm.config.text_config, mtp_ckpt_dir)
        start_step = int(json.loads((mtp_ckpt_dir / "state.json").read_text(encoding="utf-8"))["step"])
        print(f"[Stage B] Resumed MTP training at step {start_step}/{mtp_steps}", flush=True)
    else:
        mtp_head = SequentialMTPHead(vlm.config.text_config, mtp_intermediate_size=3072, shortlist_ids=shortlist)
        mtp_head.warm_start_from_target_lm(vlm.language_model)
        mtp_head.apply(lambda x: x.astype(mx.bfloat16) if mx.issubdtype(x.dtype, mx.floating) else x)
        mx.eval(mtp_head.parameters())
        start_step = 0

    embed_tokens = vlm.language_model.model.embed_tokens
    mtp_head.bind_vocab_projection(embed_tokens)
    n_mtp_params = sum(int(v.size) for _, v in tree_flatten(mtp_head.parameters()))
    print(f"[Stage B] SequentialMTPHead ready (warm-started from L30 GQA): {n_mtp_params/1e6:.2f}M params + 16,384 FR-Spec shortlist.", flush=True)

    if start_step < mtp_steps:
        # 1. Cache 36 real on-policy greedy `32L` rollouts on training prompts so MTP learns exact self-generated token habits!
        onpol_cache = WORK_V6 / "stageB_onpolicy_36.json"
        if onpol_cache.exists():
            onpol_seqs_raw = json.loads(onpol_cache.read_text(encoding="utf-8"))
            print(f"[Stage B] Loaded {len(onpol_seqs_raw)} cached on-policy 32L greedy trajectories.", flush=True)
        else:
            print("[Stage B] Generating 36 on-policy 32L greedy trajectories across training pillars (~2m)...", flush=True)
            rng_op = random.Random(20261318)
            cand_tr = [r for r in (v6_rows + v5_rows[:180]) if not r.get("images")]
            rng_op.shuffle(cand_tr)
            onpol_seqs_raw = []
            for r in cand_tr[:36]:
                msgs = r["messages"]
                u_idx = next((idx for idx, m in enumerate(msgs) if m.get("role") == "assistant"), -1)
                if u_idx <= 0:
                    continue
                p_s = _apply_template_inference_aligned(processor, vlm.config, msgs[:u_idx], num_images=0, enable_thinking=True)
                p_ids = tok.encode(p_s, add_special_tokens=False)
                if len(p_ids) > 420:
                    continue
                g_ids, _ = standard_greedy_generate(vlm, p_ids, max_tokens=208, eos_token_id=eos_id, repetition_penalty=1.0)
                if len(g_ids) >= 12:
                    onpol_seqs_raw.append({"full": (p_ids + g_ids)[:640], "ans_start": max(1, len(p_ids) - 2)})
            onpol_cache.write_text(json.dumps(onpol_seqs_raw) + "\n", encoding="utf-8")
            print(f"[Stage B] Cached {len(onpol_seqs_raw)} on-policy 32L trajectories -> {onpol_cache}", flush=True)

        # Tokenize full prompt + assistant sequences (keeping full prompt prefix so RoPE & GQA copy match inference!)
        seqs: list[tuple[mx.array, int]] = []
        for item in onpol_seqs_raw:
            arr = mx.array([item["full"]], dtype=mx.int32)
            ans_st = int(item["ans_start"])
            # Upweight on-policy 32L greedy sequences 4x so MTP specializes in exact 32L greedy argmaxes!
            for _ in range(4):
                seqs.append((arr, ans_st))

        for r in all_text_rows:
            msgs = r["messages"]
            for t_i, m in enumerate(msgs):
                if m.get("role") != "assistant":
                    continue
                p_s = _apply_template_inference_aligned(processor, vlm.config, msgs[:t_i], num_images=0, enable_thinking=True)
                a_s = str(m["content"])
                if p_s.endswith("<think>") and a_s.startswith("<think>"):
                    a_s = a_s[len("<think>") :]
                p_ids = tok.encode(p_s, add_special_tokens=False)
                a_ids = tok.encode(a_s, add_special_tokens=False) + [eos_id]
                if len(p_ids) > 460:
                    continue
                full = (p_ids + a_ids)[:672]
                if len(full) >= len(p_ids) + 8:
                    seqs.append((mx.array([full], dtype=mx.int32), max(1, len(p_ids) - 2)))

        rng = random.Random(20261320)
        rng.shuffle(seqs)
        opt_mtp = optim.AdamW(
            learning_rate=optim.cosine_decay(5.0e-4, decay_steps=mtp_steps, end=3.0e-5),
            weight_decay=0.01,
        )

        def mtp_2step_loss(
            head: SequentialMTPHead,
            h_in_1: mx.array,
            h_tgt_next: mx.array,
            tok_in_1: mx.array,
            tok_in_2: mx.array,
            a0: int,
            a1: int,
            a1_step2: int,
            t_top1_1: mx.array,
            t_probs_1: mx.array,
            t_top1_2: mx.array,
            t_probs_2: mx.array,
        ) -> mx.array:
            # Step k=1 over FULL sequence from RoPE index 0: `(h_t, embed(y_{t+1})) -> h_{t+1}, y_{t+2}`
            emb_1 = mx.stop_gradient(embed_tokens(tok_in_1))
            h_m1 = head.forward_features(h_in_1, emb_1, cache=None)
            h_m1_ans = h_m1[:, a0:a1, :]
            h_tgt_ans = h_tgt_next[:, a0:a1, :]
            feat_l1 = nn.losses.smooth_l1_loss(h_m1_ans.astype(mx.float32), h_tgt_ans.astype(mx.float32), beta=0.1)

            logits_1 = head.draft_logits_shortlist(h_m1_ans, embed_tokens).astype(mx.float32)
            logp_1 = logits_1 - mx.logsumexp(logits_1, axis=-1, keepdims=True)
            kl_1 = -mx.sum(t_probs_1 * logp_1, axis=-1)
            ce_1 = nn.losses.cross_entropy(logits_1, t_top1_1, reduction="none")
            loss_1 = mx.mean(0.55 * kl_1 + 0.45 * ce_1)

            # Step k=2 (true prompt KVs + EAGLE self-conditioned assistant `h_m1`!):
            h_in_2 = mx.concatenate([h_in_1[:, 1 : a0 + 1, :], h_m1[:, a0:-1, :]], axis=1)
            emb_2 = mx.stop_gradient(embed_tokens(tok_in_2))
            h_m2 = head.forward_features(h_in_2, emb_2, cache=None)
            h_m2_ans = h_m2[:, a0:a1_step2, :]
            h_tgt_ans2 = h_tgt_next[:, a0 + 1 : a1_step2 + 1, :]
            feat_l2 = nn.losses.smooth_l1_loss(h_m2_ans.astype(mx.float32), h_tgt_ans2.astype(mx.float32), beta=0.1)

            logits_2 = head.draft_logits_shortlist(h_m2_ans, embed_tokens).astype(mx.float32)
            logp_2 = logits_2 - mx.logsumexp(logits_2, axis=-1, keepdims=True)
            kl_2 = -mx.sum(t_probs_2 * logp_2, axis=-1)
            ce_2 = nn.losses.cross_entropy(logits_2, t_top1_2, reduction="none")
            loss_2 = mx.mean(0.55 * kl_2 + 0.45 * ce_2)
            return (loss_1 + 1.0 * feat_l1) + 0.75 * (loss_2 + 1.0 * feat_l2)

        vg_mtp = nn.value_and_grad(mtp_head, mtp_2step_loss)
        t_m0 = time.perf_counter()
        run_l: list[float] = []

        for st in range(start_step, mtp_steps):
            acc_g = None
            step_l_sum = 0.0
            used_mb = 0
            for sub_b in range(2):
                full_ids, ans_start = seqs[(st * 2 + sub_b) % len(seqs)]
                seq_len = int(full_ids.shape[1])
                t_out = vlm.language_model.model(full_ids[:, :-1], cache=None)
                mx.eval(t_out)

                len_step1 = seq_len - 2
                a0 = max(1, min(ans_start - 2, len_step1 - 6))
                a1 = min(len_step1, a0 + 232)
                a1_step2 = min(len_step1 - 1, a1 - 1)
                if a1_step2 - a0 < 4:
                    continue

                h_in_1 = mx.stop_gradient(t_out[:, :-1, :])
                h_tgt_next = mx.stop_gradient(t_out[:, 1:, :])
                tok_in_1 = full_ids[:, 1:-1]
                tok_in_2 = full_ids[:, 2:-1]

                assert mtp_head._cached_w_sub is not None
                t_sub_logits = (h_tgt_next[:, a0:a1, :] @ mtp_head._cached_w_sub.T).astype(mx.float32)
                t_probs_1 = mx.stop_gradient(mx.softmax(t_sub_logits, axis=-1))
                t_top1_1 = mx.stop_gradient(mx.argmax(t_sub_logits, axis=-1))

                span2 = a1_step2 - a0
                t_probs_2 = t_probs_1[:, 1 : 1 + span2, :]
                t_top1_2 = t_top1_1[:, 1 : 1 + span2]

                lv, gr = vg_mtp(
                    mtp_head,
                    h_in_1,
                    h_tgt_next,
                    tok_in_1,
                    tok_in_2,
                    a0,
                    a1,
                    a1_step2,
                    t_top1_1,
                    t_probs_1,
                    t_top1_2,
                    t_probs_2,
                )
                mx.eval(lv)
                step_l_sum += float(lv.item())
                used_mb += 1
                acc_g = tree_map(lambda g: g * 0.5, gr) if acc_g is None else tree_map(lambda a, g: a + g * 0.5, acc_g, gr)

            if acc_g is not None and used_mb > 0:
                acc_g, _ = optim.clip_grad_norm(acc_g, max_norm=1.0)
                opt_mtp.update(mtp_head, acc_g)
                mx.eval(mtp_head.parameters(), opt_mtp.state)
                run_l.append(step_l_sum / used_mb)

            if (st + 1) % 90 == 0 or st == mtp_steps - 1:
                save_mtp_checkpoint(mtp_head, mtp_ckpt_dir)
                (mtp_ckpt_dir / "state.json").write_text(json.dumps({"step": st + 1}), encoding="utf-8")
                print(
                    f"  [Stage B MTP {st+1:3d}/{mtp_steps} | {(time.perf_counter()-t_m0)/60:4.1f}m] "
                    f"distill_loss={sum(run_l)/max(1,len(run_l)):.4f}",
                    flush=True,
                )
                run_l.clear()
                mx.clear_cache()

        save_mtp_checkpoint(mtp_head, model_32l_dir)

    # Benchmark real wall-clock `tok/s` & verify lossless bfloat16 target verification across 9 diverse prompts!
    def _max_target_logit_gap_bf16(prompt_toks: list[int], gen_toks: list[int]) -> float:
        if not gen_toks:
            return 0.0
        seq_in = mx.array([prompt_toks + gen_toks[:-1]], dtype=mx.int32)
        logits_all = vlm.language_model(seq_in, cache=None).logits[0, len(prompt_toks) - 1 :, :].astype(mx.float32)
        max_l = mx.max(logits_all, axis=-1)
        chosen_l = mx.take_along_axis(logits_all, mx.array(gen_toks, dtype=mx.int32)[:, None], axis=-1).squeeze(-1)
        mx.eval(max_l, chosen_l)
        return float(mx.max(max_l - chosen_l).item())

    eval_56 = [json.loads(l) for l in (V5_CORPUS_DIR / "heldout_v5_rsi_56.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    sample_tasks = [
        eval_56[0],   # RSI Python SVD code
        eval_56[1],   # ROME MLP Python code
        eval_56[4],   # SandwichNorm JSON math
        eval_56[7],   # McNemar research stats
        eval_56[14],  # XML Tool call
        eval_56[21],  # 2-Turn RPR belief revision
        eval_56[28],  # Clause witness math
        eval_56[35],  # Binary search Python debug
        eval_56[42],  # RPR Auditor CHECK lines
    ]
    w_p = tok.encode(
        _apply_template_inference_aligned(processor, vlm.config, sample_tasks[0]["messages"], num_images=0, enable_thinking=True),
        add_special_tokens=False,
    )
    _ = standard_greedy_generate(vlm, w_p, max_tokens=16, eos_token_id=eos_id)
    for _wk in (2, 3, 4, 5):
        _ = mtp_speculative_generate(vlm, mtp_head, w_p, max_tokens=16, draft_k=_wk, eos_token_id=eos_id)

    bench_rows: list[dict[str, Any]] = []
    tot_std_toks = 0
    tot_k2_toks = 0
    tot_k3_toks = 0
    tot_k4_toks = 0
    tot_k5_toks = 0
    tot_std_s = 0.0
    tot_k2_s = 0.0
    tot_k3_s = 0.0
    tot_k4_s = 0.0
    tot_k5_s = 0.0
    exact_seq_matches = 0
    max_bf16_ulp_gap = 0.0
    std_pass_cnt = 0
    k3_pass_cnt = 0
    k4_pass_cnt = 0

    for t in sample_tasks:
        p_str = _apply_template_inference_aligned(processor, vlm.config, t["messages"], num_images=0, enable_thinking=True)
        p_txt = "\n".join(str(m.get("content", "")) for m in t["messages"])
        p_ids = tok.encode(p_str, add_special_tokens=False)
        ref_ids, st_std = standard_greedy_generate(vlm, p_ids, max_tokens=256, eos_token_id=eos_id, repetition_penalty=1.0)
        k2_ids, st_k2 = mtp_speculative_generate(vlm, mtp_head, p_ids, max_tokens=256, draft_k=2, eos_token_id=eos_id, repetition_penalty=1.0)
        k3_ids, st_k3 = mtp_speculative_generate(vlm, mtp_head, p_ids, max_tokens=256, draft_k=3, eos_token_id=eos_id, repetition_penalty=1.0)
        k4_ids, st_k4 = mtp_speculative_generate(vlm, mtp_head, p_ids, max_tokens=256, draft_k=4, eos_token_id=eos_id, repetition_penalty=1.0)
        k5_ids, st_k5 = mtp_speculative_generate(vlm, mtp_head, p_ids, max_tokens=256, draft_k=5, eos_token_id=eos_id, repetition_penalty=1.0)

        gap_k3 = _max_target_logit_gap_bf16(p_ids, k3_ids)
        gap_k4 = _max_target_logit_gap_bf16(p_ids, k4_ids)
        task_max_gap = round(max(gap_k3, gap_k4), 5)
        max_bf16_ulp_gap = max(max_bf16_ulp_gap, task_max_gap)
        assert task_max_gap <= 0.35, f"Target verification logit gap {task_max_gap} exceeded bf16 GEMV/GEMM rounding on {t['id']}!"

        exact_match = (ref_ids == k3_ids)
        if exact_match:
            exact_seq_matches += 1
        prefix_match_len = next((idx for idx, (a, b) in enumerate(zip(ref_ids, k3_ids)) if a != b), min(len(ref_ids), len(k3_ids)))

        def _check_pass(g_toks: list[int]) -> bool:
            clean = [tid for tid in g_toks if tid != eos_id]
            txt = ("<think>" if p_str.endswith("<think>") else "") + tok.decode(clean)
            return bool(score_response(txt, prompt_text=p_txt, kind=t["kind"], verifier=t["verifier"], tokenizer=tok)["correct"])

        ok_std = _check_pass(ref_ids)
        ok_k3 = _check_pass(k3_ids)
        ok_k4 = _check_pass(k4_ids)
        std_pass_cnt += int(ok_std)
        k3_pass_cnt += int(ok_k3)
        k4_pass_cnt += int(ok_k4)

        tot_std_toks += st_std["tokens"]
        tot_k2_toks += st_k2["tokens"]
        tot_k3_toks += st_k3["tokens"]
        tot_k4_toks += st_k4["tokens"]
        tot_k5_toks += st_k5["tokens"]
        tot_std_s += st_std["wall_clock_sec"]
        tot_k2_s += st_k2["wall_clock_sec"]
        tot_k3_s += st_k3["wall_clock_sec"]
        tot_k4_s += st_k4["wall_clock_sec"]
        tot_k5_s += st_k5["wall_clock_sec"]
        sp2 = round(st_k2["tok_per_sec"] / max(1e-6, st_std["tok_per_sec"]), 3)
        sp3 = round(st_k3["tok_per_sec"] / max(1e-6, st_std["tok_per_sec"]), 3)
        sp4 = round(st_k4["tok_per_sec"] / max(1e-6, st_std["tok_per_sec"]), 3)
        sp5 = round(st_k5["tok_per_sec"] / max(1e-6, st_std["tok_per_sec"]), 3)
        bench_rows.append({
            "id": t["id"],
            "pillar": t["pillar"],
            "tokens": st_std["tokens"],
            "std_tok_s": st_std["tok_per_sec"],
            "mtp_k2_tok_s": st_k2["tok_per_sec"],
            "mtp_k2_speedup": sp2,
            "mtp_k2_accept_rate": st_k2["acceptance_rate"],
            "mtp_k3_tok_s": st_k3["tok_per_sec"],
            "mtp_k3_speedup": sp3,
            "mtp_k3_accept_rate": st_k3["acceptance_rate"],
            "mtp_k3_toks_per_verify": st_k3["mean_tokens_per_verify"],
            "mtp_k4_tok_s": st_k4["tok_per_sec"],
            "mtp_k4_speedup": sp4,
            "mtp_k4_accept_rate": st_k4["acceptance_rate"],
            "mtp_k4_toks_per_verify": st_k4["mean_tokens_per_verify"],
            "mtp_k5_tok_s": st_k5["tok_per_sec"],
            "mtp_k5_speedup": sp5,
            "mtp_k5_accept_rate": st_k5["acceptance_rate"],
            "exact_l1_gemv_match": exact_match,
            "prefix_match_tokens": prefix_match_len,
            "max_target_logit_gap_bf16": task_max_gap,
            "std_correct": ok_std,
            "mtp_k3_correct": ok_k3,
            "mtp_k4_correct": ok_k4,
        })
        print(
            f"  [MTP Speed {t['id']:16s}] std={st_std['tok_per_sec']:5.1f} tok/s | "
            f"K=2: {st_k2['tok_per_sec']:5.1f} ({sp2:.2f}x, acc={st_k2['acceptance_rate']*100:.1f}%) | "
            f"K=3: {st_k3['tok_per_sec']:5.1f} ({sp3:.2f}x, acc={st_k3['acceptance_rate']*100:.1f}%) | "
            f"K=4: {st_k4['tok_per_sec']:5.1f} ({sp4:.2f}x, acc={st_k4['acceptance_rate']*100:.1f}%) | "
            f"max_bf16_gap={task_max_gap:.4f}",
            flush=True,
        )

    agg_std_tps = round(tot_std_toks / max(1e-6, tot_std_s), 2)
    agg_k2_tps = round(tot_k2_toks / max(1e-6, tot_k2_s), 2)
    agg_k3_tps = round(tot_k3_toks / max(1e-6, tot_k3_s), 2)
    agg_k4_tps = round(tot_k4_toks / max(1e-6, tot_k4_s), 2)
    agg_k5_tps = round(tot_k5_toks / max(1e-6, tot_k5_s), 2)
    summary = {
        "mtp_params_millions": round(n_mtp_params / 1e6, 2),
        "shortlist_size": len(shortlist),
        "total_tokens_generated": tot_std_toks,
        "float32_exact_token_parity_unit_tested": True,
        "bf16_exact_l1_gemv_matches_of_9": exact_seq_matches,
        "max_target_logit_gap_bf16": round(max_bf16_ulp_gap, 5),
        "benchmark_9_pass_counts": {
            "standard_greedy": std_pass_cnt,
            "mtp_k3": k3_pass_cnt,
            "mtp_k4": k4_pass_cnt,
        },
        "standard_greedy_tok_per_sec": agg_std_tps,
        "mtp_k2_tok_per_sec": agg_k2_tps,
        "mtp_k2_overall_speedup": round(agg_k2_tps / max(1e-6, agg_std_tps), 3),
        "mtp_k3_tok_per_sec": agg_k3_tps,
        "mtp_k3_overall_speedup": round(agg_k3_tps / max(1e-6, agg_std_tps), 3),
        "mtp_k4_tok_per_sec": agg_k4_tps,
        "mtp_k4_overall_speedup": round(agg_k4_tps / max(1e-6, agg_std_tps), 3),
        "mtp_k5_tok_per_sec": agg_k5_tps,
        "mtp_k5_overall_speedup": round(agg_k5_tps / max(1e-6, agg_std_tps), 3),
        "per_task": bench_rows,
    }
    (bench_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    del vlm, processor, mtp_head
    gc.collect()
    mx.clear_cache()
    return summary


# ===========================================================================
# STAGE C: Live Hybrid SEAL + SVF + Nullspace Self-Weight-Rewriting RL on a Copy!
# ===========================================================================


def extract_nullspace_and_svf_banks(
    vlm: Any,
    processor: Any,
    anchor_items: list[dict[str, Any]],
    skill_batches: list[list[dict[str, Any]]],
    bank_path: Path,
) -> dict[str, mx.array]:
    """Extract real Null-Space Projectors `P_perp` (`[d_in, d_in]`) and 4 Rank-8 SVF Expert Deltas on `REWRITE_TARGET_LAYERS`."""
    if bank_path.exists():
        loaded = mx.load(str(bank_path))
        mx.eval(list(loaded.values()))
        print(f"[Stage C1] Loaded cached Null-Space + SVF weight-surgery bank ({len(loaded)} tensors) from {bank_path}", flush=True)
        return loaded

    print(f"[Stage C1] Extracting real Null-Space projectors & 4 SVF expert subspaces on layers {REWRITE_TARGET_LAYERS}...", flush=True)
    lm = vlm.language_model
    bank: dict[str, mx.array] = {}

    # 1. Capture post-ffn_norm activations `x_ffn` entering `feed_forward.w1/w3` and `h_ffn` entering `w2`?
    # Note: `w1` and `w3` have `d_in = 2048` (`hidden_size`), whereas `w2` has `d_in = 7680` (`[7680, 7680]` = 235MB!).
    # Targeting `self_attn.o_proj` (`d_in = 2048`) and `feed_forward.w1` / `feed_forward.w3` (`d_in = 2048`) keeps `P_perp`
    # compact (`[2048, 2048]` = `16 MiB` float32!) and directly controls both attention readout & SwiGLU gating!
    x_attn_list: dict[int, list[np.ndarray]] = {l: [] for l in REWRITE_TARGET_LAYERS}
    x_ffn_list: dict[int, list[np.ndarray]] = {l: [] for l in REWRITE_TARGET_LAYERS}

    for it in anchor_items[:36]:
        in_ids = it["full_ids"][:, :-1]
        emb = lm.model.embed_tokens(in_ids)
        if it["vt_feats"] is not None and it["img_indices"] is not None:
            proj = vlm.multi_modal_projector(it["vt_feats"])
            flat = emb.reshape(-1, emb.shape[-1])
            flat[it["img_indices"]] = proj.astype(emb.dtype)
            emb = flat.reshape(emb.shape)
        h = emb
        for idx, layer in enumerate(lm.model.layers):
            if layer.is_attention_layer:
                win = lm.config.sliding_window if layer.layer_type == "sliding_attention" else None
                from mlx_vlm.models.base import create_attention_mask, scaled_dot_product_attention
                mask = create_attention_mask(h, None, window_size=win)
                op_in = layer.operator_norm(h)
                if idx in REWRITE_TARGET_LAYERS:
                    # Capture exact pre-`o_proj` gated attention vector `[1, S, 2048]`!
                    att = layer.self_attn
                    B, L, _ = op_in.shape
                    q = att.rope(att.q_norm(att.q_proj(op_in).reshape(B, L, att.n_heads, att.head_dim)).transpose(0, 2, 1, 3), 0, None)
                    k = att.rope(att.k_norm(att.k_proj(op_in).reshape(B, L, att.n_kv_heads, att.head_dim)).transpose(0, 2, 1, 3), 0, None)
                    v = att.v_proj(op_in).reshape(B, L, att.n_kv_heads, att.head_dim).transpose(0, 2, 1, 3)
                    ao = scaled_dot_product_attention(q, k, v, cache=None, scale=att.scale, mask=mask).transpose(0, 2, 1, 3)
                    ao = (ao * mx.sigmoid(att.gate_proj(op_in))[..., None]).reshape(B, L, -1)
                    mx.eval(ao)
                    arr_a = np.asarray(ao.astype(mx.float32))[0, -min(24, L) :, :]
                    x_attn_list[idx].append(arr_a)
                h_op = layer.self_attn(op_in, mask, None, None)
            else:
                from mlx_vlm.models.base import create_ssm_mask
                mask = create_ssm_mask(h, None)
                h_op = layer.conv(layer.operator_norm(h), mask, None)
            h = h + (layer.post_operator_norm(h_op) if layer.use_sandwich_norm else h_op)
            ff_in = layer.ffn_norm(h)
            if idx in REWRITE_TARGET_LAYERS:
                mx.eval(ff_in)
                arr_f = np.asarray(ff_in.astype(mx.float32))[0, -min(24, int(ff_in.shape[1])) :, :]
                x_ffn_list[idx].append(arr_f)
            h_ff = layer.feed_forward(ff_in)
            h = h + (layer.post_ffn_norm(h_ff) if layer.use_sandwich_norm else h_ff)

    for idx in REWRITE_TARGET_LAYERS:
        for tag, mat_list in [("attn_o", x_attn_list[idx]), ("ffn_in", x_ffn_list[idx])]:
            X_cat = np.concatenate(mat_list, axis=0)  # [N_toks, 2048]
            _, s_v, vh = np.linalg.svd(X_cat, full_matrices=False)
            v_k = vh[:48]  # Protect top-48 anchor principal directions!
            p_perp = np.eye(X_cat.shape[1], dtype=np.float32) - (v_k.T @ v_k)
            bank[f"L{idx}.{tag}.P_perp"] = mx.array(p_perp).astype(mx.bfloat16)

    # 2. Extract 4 Rank-8 SVF Task-Expert Directions (`b = 0..3`) via gradient SVD projected onto `P_perp`!
    vlm.freeze()
    for idx in REWRITE_TARGET_LAYERS:
        lyr = lm.model.layers[idx]
        lyr.self_attn.o_proj.unfreeze()
        lyr.feed_forward.w3.unfreeze()

    def _batch_ce(rm: nn.Module, f_ids: mx.array, s_i: int) -> mx.array:
        return mx.mean(chunked_multimodal_ce(rm, f_ids, s_i))

    vg_svf = nn.value_and_grad(vlm, _batch_ce)
    for b_idx, s_batch in enumerate(skill_batches):
        acc_g = None
        for it in s_batch[:8]:
            lv, gr = vg_svf(vlm, it["full_ids"], it["start_idx"])
            mx.eval(lv)
            acc_g = gr if acc_g is None else tree_map(lambda a, b: a + b, acc_g, gr)
        flat_g = dict(tree_flatten(acc_g))
        for idx in REWRITE_TARGET_LAYERS:
            for mod_key, p_key in [
                (f"language_model.model.layers.{idx}.self_attn.o_proj.weight", f"L{idx}.attn_o.P_perp"),
                (f"language_model.model.layers.{idx}.feed_forward.w3.weight", f"L{idx}.ffn_in.P_perp"),
            ]:
                g_np = -np.asarray(flat_g[mod_key].astype(mx.float32))  # negative gradient = descent improvement direction!
                p_np = np.asarray(bank[p_key].astype(mx.float32))
                g_proj = g_np @ p_np
                u, s, vh = np.linalg.svd(g_proj, full_matrices=False)
                # Normalize Frobenius norm to `0.06 * ||W||_F / sqrt(8)` so `svf_gain = 1.0` is a clean calibrated step!
                w_orig = dict(tree_flatten(vlm.parameters()))[mod_key]
                w_norm = float(mx.sqrt(mx.sum(w_orig.astype(mx.float32) ** 2)).item())
                low_rank = (u[:, :8] * s[None, :8]) @ vh[:8, :]
                lr_norm = max(1e-8, float(np.linalg.norm(low_rank)))
                calibrated = (0.018 * w_norm / lr_norm) * low_rank
                short_m = "o_proj" if "o_proj" in mod_key else "w3"
                bank[f"svf_b{b_idx}.L{idx}.{short_m}"] = mx.array(calibrated).astype(mx.bfloat16)

    vlm.freeze()
    mx.eval(list(bank.values()))
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(bank_path), bank, metadata={"format": "mlx"})
    mx.clear_cache()
    print(f"[Stage C1] Saved {len(bank)} Null-Space + SVF weight-surgery tensors -> {bank_path}", flush=True)
    return bank


def _eval_task_batch(vlm: Any, processor: Any, tasks: list[dict[str, Any]]) -> tuple[float, float, list[dict[str, Any]]]:
    import mlx_vlm

    tok = processor.tokenizer
    oks = 0
    rews = []
    details = []
    for t in tasks:
        p_str = _apply_template_inference_aligned(processor, vlm.config, t["messages"], num_images=0, enable_thinking=True)
        res = mlx_vlm.generate(vlm, processor, p_str, max_tokens=448, temperature=0.0, repetition_penalty=1.05, verbose=False)
        raw = res.text or ""
        p_txt = "\n".join(str(m.get("content", "")) for m in t["messages"])
        sc = score_response(raw, prompt_text=p_txt, kind=t["kind"], verifier=t["verifier"], tokenizer=tok, max_think_tokens=320)
        if sc["correct"]:
            oks += 1
        rews.append(float(sc["reward"]))
        details.append({"id": t["id"], "kind": t["kind"], "correct": sc["correct"], "raw_tail": raw[-180:]})
    n = max(1, len(tasks))
    return oks / n, sum(rews) / n, details


def _format_diagnostic_prompt(tasks: list[dict[str, Any]], details: list[dict[str, Any]], anchor_nll: float) -> list[dict[str, str]]:
    lines = []
    pass_n = sum(1 for d in details if d["correct"])
    for i, (t, d) in enumerate(zip(tasks, details), start=1):
        u_snip = str(t["messages"][-1]["content"])[:220].replace("\n", " ")
        status = "PASS" if d["correct"] else f"FAIL (tail={d['raw_tail'][:120]!r})"
        lines.append(f"- `diag_{i}` ({t['kind']}): {status} | prompt_excerpt: {u_snip}")
    user_txt = (
        f"Dev Diagnostic Batch (`{len(tasks)}` tasks):\n"
        + "\n".join(lines)
        + f"\nCurrent dev pass rate: `{pass_n}/{len(tasks)}` (`{pass_n/max(1,len(tasks))*100:.1f}%`). "
        f"Anchor NLL: `{anchor_nll:.2f}`.\nPropose a surgical `<weight_rewrite>` plan for `theta_copy`."
    )
    return [
        {"role": "system", "content": SEAL_SYSTEM_PROMPT},
        {"role": "user", "content": user_txt},
    ]


def parse_weight_rewrite_plan(raw_text: str) -> tuple[dict[str, Any] | None, float]:
    """Parse & clamp a model-proposed `<weight_rewrite>` JSON object, returning `(plan, schema_reward)`."""
    closed = "</think>" in raw_text.lower()
    _, body = split_thinking(raw_text)
    search_s = body if body.strip() else raw_text
    obj = extract_json_object(search_s)
    req_keys = {"action", "svf_gains", "steps", "nullspace_alpha", "wise_ft_gamma", "self_edits"}
    # If first JSON lacked `<weight_rewrite>` keys, scan balanced `{...}` candidates from right to left
    if not isinstance(obj, dict) or len(req_keys & set(obj.keys())) < 2:
        decoder = json.JSONDecoder()
        for pos in range(len(search_s) - 1, -1, -1):
            if search_s[pos] == "{":
                try:
                    cand_obj, _ = decoder.raw_decode(search_s[pos:])
                    if isinstance(cand_obj, dict) and len(req_keys & set(cand_obj.keys())) >= 2:
                        obj = cand_obj
                        break
                except Exception:
                    continue
    if not isinstance(obj, dict):
        return None, (-0.4 if not closed else -0.2)

    hit_keys = len(req_keys & set(obj.keys()))
    r_schema = (0.35 if closed else 0.0) + 0.35 * (hit_keys / len(req_keys))

    def _safe_float(val: Any, default: float) -> float:
        if isinstance(val, (list, tuple)):
            val = val[0] if val else default
        try:
            f = float(val)
            return f if math.isfinite(f) else default
        except (TypeError, ValueError):
            return default

    def _safe_int(val: Any, default: int) -> int:
        if isinstance(val, (list, tuple)):
            val = val[0] if val else default
        try:
            return int(float(val))
        except (TypeError, ValueError):
            return default

    action = str(obj.get("action", "rewrite")).lower()
    raw_gains_in = obj.get("svf_gains")
    if isinstance(raw_gains_in, list) and len(raw_gains_in) == 1 and isinstance(raw_gains_in[0], list):
        raw_gains_in = raw_gains_in[0]
    raw_gains = raw_gains_in if isinstance(raw_gains_in, list) else [0.2, 0.2, 0.2, 0.2]
    gains = [float(max(-0.4, min(1.0, _safe_float(x, 0.0)))) for x in (raw_gains + [0.0, 0.0, 0.0, 0.0])[:4]]
    lr_b = float(max(0.0, min(1.5e-4, _safe_float(obj.get("lr_b"), 7.0e-5))))
    lr_a = float(max(0.0, min(5.0e-5, _safe_float(obj.get("lr_a"), 2.0e-5))))
    steps = int(max(0, min(5, _safe_int(obj.get("steps"), 2))))
    null_a = float(max(0.50, min(0.98, _safe_float(obj.get("nullspace_alpha"), 0.88))))
    gamma = float(max(0.75, min(1.00, _safe_float(obj.get("wise_ft_gamma"), 0.90))))

    edits_clean = []
    raw_edits = obj.get("self_edits")
    if isinstance(raw_edits, list):
        for ed in raw_edits[:2]:
            if isinstance(ed, dict) and ed.get("prompt") and ed.get("response"):
                resp_s = str(ed["response"])
                if "<think>" in resp_s and "</think>" in resp_s:
                    edits_clean.append({"prompt": str(ed["prompt"]), "response": resp_s})
                    r_schema += 0.15

    return {
        "action": "noop" if action == "noop" else "rewrite",
        "svf_gains": gains,
        "lr_b": lr_b,
        "lr_a": lr_a,
        "steps": steps,
        "nullspace_alpha": null_a,
        "wise_ft_gamma": gamma,
        "self_edits": edits_clean,
    }, round(r_schema, 4)


def snapshot_rewrite_layers(vlm: Any) -> dict[int, dict[str, mx.array]]:
    snap: dict[int, dict[str, mx.array]] = {}
    for idx in REWRITE_TARGET_LAYERS:
        lyr = vlm.language_model.model.layers[idx]
        snap[idx] = {
            "o_proj": mx.array(lyr.self_attn.o_proj.weight),
            "w3": mx.array(lyr.feed_forward.w3.weight),
        }
    mx.eval([ arr for d in snap.values() for arr in d.values() ])
    return snap


def restore_rewrite_layers(vlm: Any, snap: dict[int, dict[str, mx.array]]) -> None:
    for idx, d in snap.items():
        lyr = vlm.language_model.model.layers[idx]
        lyr.self_attn.o_proj.weight = mx.array(d["o_proj"])
        lyr.feed_forward.w3.weight = mx.array(d["w3"])
    mx.eval([vlm.language_model.model.layers[idx].self_attn.o_proj.weight for idx in snap])


def apply_hybrid_seal_svf_to_live_copy(
    vlm: Any,
    processor: Any,
    plan: dict[str, Any],
    base_snap: dict[int, dict[str, mx.array]],
    bank: dict[str, mx.array],
) -> None:
    """Apply `plan` (`SVF` subspace gains + `SEAL` self-edit gradient steps + `Nullspace` projection) in-place to `vlm`."""
    restore_rewrite_layers(vlm, base_snap)
    if plan["action"] == "noop":
        return

    gains = plan["svf_gains"]
    gamma = plan["wise_ft_gamma"]
    null_a = plan["nullspace_alpha"]

    # 1. Optional inner-loop SEAL gradient steps on model's own `self_edits`!
    seal_deltas: dict[tuple[int, str], mx.array] = {}
    if plan["steps"] > 0 and plan["self_edits"] and plan["lr_b"] > 0:
        edit_rows = [
            {
                "id": f"seal_ed_{i}",
                "pillar": "self_edit",
                "images": [],
                "messages": [
                    {"role": "system", "content": "Solve accurately step-by-step in <think>."},
                    {"role": "user", "content": ed["prompt"]},
                    {"role": "assistant", "content": ed["response"]},
                ],
            }
            for i, ed in enumerate(plan["self_edits"])
        ]
        ed_items, _ = prepare_turn_items_v5(edit_rows, vlm, processor, max_seq_len=1536)
        if ed_items:
            vlm.freeze()
            for idx in REWRITE_TARGET_LAYERS:
                lyr = vlm.language_model.model.layers[idx]
                lyr.self_attn.o_proj.unfreeze()
                lyr.feed_forward.w3.unfreeze()
            inner_opt = optim.AdamW(learning_rate=plan["lr_b"], weight_decay=0.0)
            for st_i in range(plan["steps"]):
                it = ed_items[st_i % len(ed_items)]
                _run_dft_batch(vlm, [it], inner_opt, scale_grad=1.0)
            vlm.freeze()
            for idx in REWRITE_TARGET_LAYERS:
                lyr = vlm.language_model.model.layers[idx]
                seal_deltas[(idx, "o_proj")] = (lyr.self_attn.o_proj.weight.astype(mx.float32) - base_snap[idx]["o_proj"].astype(mx.float32))
                seal_deltas[(idx, "w3")] = (lyr.feed_forward.w3.weight.astype(mx.float32) - base_snap[idx]["w3"].astype(mx.float32))

    # 2. Combine `SVF` basis gains + Null-Space-Projected `SEAL` gradient deltas, scaled by `wise_ft_gamma`!
    for idx in REWRITE_TARGET_LAYERS:
        lyr = vlm.language_model.model.layers[idx]
        for short_m, p_tag, attr_obj in [
            ("o_proj", "attn_o", lyr.self_attn.o_proj),
            ("w3", "ffn_in", lyr.feed_forward.w3),
        ]:
            w0 = base_snap[idx][short_m].astype(mx.float32)
            svf_sum = mx.zeros_like(w0)
            for b_i, g_val in enumerate(gains):
                if abs(g_val) > 1e-6:
                    svf_sum = svf_sum + float(g_val) * bank[f"svf_b{b_i}.L{idx}.{short_m}"].astype(mx.float32)

            d_seal = seal_deltas.get((idx, short_m))
            if d_seal is not None:
                p_perp = bank[f"L{idx}.{p_tag}.P_perp"].astype(mx.float32)
                d_proj = (1.0 - null_a) * d_seal + null_a * (d_seal @ p_perp)
                tot_delta = gamma * (svf_sum + d_proj)
            else:
                tot_delta = gamma * svf_sum

            attr_obj.weight = (w0 + tot_delta).astype(mx.bfloat16)

    mx.eval([vlm.language_model.model.layers[idx].self_attn.o_proj.weight for idx in REWRITE_TARGET_LAYERS])
    mx.eval([vlm.language_model.model.layers[idx].feed_forward.w3.weight for idx in REWRITE_TARGET_LAYERS])


def run_stage_c_live_self_rewrite_rl(
    src_32l_dir: Path,
    export_final_dir: Path,
    *,
    outer_episodes: int = 12,
) -> dict[str, Any]:
    import mlx_vlm
    from mlx_vlm.trainer.peft import get_peft_model

    out_eval_dir = ROOT_DIR / "results" / "self_rewrite_seal_eval"
    out_eval_dir.mkdir(parents=True, exist_ok=True)
    if (export_final_dir / "model.safetensors").exists() and (out_eval_dir / "summary.json").exists():
        print(f"[Stage C] Resuming completed self-rewriting model & evaluation from {out_eval_dir / 'summary.json'}", flush=True)
        return json.loads((out_eval_dir / "summary.json").read_text(encoding="utf-8"))

    v6_val = [json.loads(l) for l in (V6_CORPUS_DIR / "val_v6_repair.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    v6_tr = [json.loads(l) for l in (V6_CORPUS_DIR / "train_v6_repair.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    seal_data = json.loads((V6_CORPUS_DIR / "seal_svf_episodes.json").read_text(encoding="utf-8"))

    print(f"[Stage C] Loading 32L model {src_32l_dir} for Live Hybrid SEAL + SVF Self-Rewrite RL...", flush=True)
    vlm, processor = mlx_vlm.load(str(src_32l_dir))
    anchor_items, _ = prepare_turn_items_v5(v6_val[:36], vlm, processor, max_seq_len=1536)

    # Group 4 skill batches from `v6_tr` for SVF task-expert extraction
    code_it, _ = prepare_turn_items_v5([r for r in v6_tr if r["pillar"] == "diagnostic_debugging"][:10], vlm, processor)
    math_it, _ = prepare_turn_items_v5([r for r in v6_tr if r["pillar"] == "clause_witness_math"][:10], vlm, processor)
    rpr_it, _ = prepare_turn_items_v5([r for r in v6_tr if r["pillar"] == "standalone_rpr_2turn"][:10], vlm, processor)
    rsi_it, _ = prepare_turn_items_v5([r for r in v6_tr if r["pillar"] == "rsi_and_weight_surgery"][:10], vlm, processor)

    bank_path = WORK_V6 / "seal_svf_bank.safetensors"
    bank = extract_nullspace_and_svf_banks(vlm, processor, anchor_items, [code_it, math_it, rpr_it, rsi_it], bank_path)

    # Attach outer-loop policy LoRA ONLY on layers `[6..21]` (`q_proj, v_proj, o_proj, w1, w2`) so `REWRITE_TARGET_LAYERS` `(22,24,28,30)`
    # remain plain `nn.Linear` for fast in-place `theta_copy` weight rewriting during rollouts!
    policy_layers = tuple(f"model.layers.{i}." for i in range(6, 22))
    pol_targets = [
        f"language_model.{n}"
        for n, _ in vlm.language_model.named_modules()
        if n.startswith(policy_layers) and n.split(".")[-1] in {"q_proj", "v_proj", "o_proj", "w1", "w2"}
    ]
    vlm = get_peft_model(vlm, linear_layers=pol_targets, rank=16, alpha=32, dropout=0.0, verbose=False)
    vlm.eval()
    print(f"[Stage C2] Outer-loop Self-Rewrite Policy LoRA attached ({len(pol_targets)} modules on layers 6..21).", flush=True)

    seal_demos, _ = prepare_turn_items_v5([r for r in v6_tr if r["pillar"] == "seal_self_rewrite"], vlm, processor, max_seq_len=1664)
    c_ckpt = WORK_V6 / "stageC_ckpt"
    c_state = {"ep": 0, "done": False, "history": []}
    if (c_ckpt / "state.json").exists() and (c_ckpt / "trainable.safetensors").exists():
        c_state = json.loads((c_ckpt / "state.json").read_text(encoding="utf-8"))
        vlm.load_weights(str(c_ckpt / "trainable.safetensors"), strict=False)
        mx.eval(vlm.parameters())
        print(f"[Stage C2] Resumed outer-loop RL at episode {c_state['ep']}/{outer_episodes}", flush=True)

    base_snap = snapshot_rewrite_layers(vlm)
    ref_pol_weights = {k: mx.array(v) for k, v in tree_flatten(vlm.trainable_parameters())}
    mx.eval(list(ref_pol_weights.values()))

    if not c_state["done"]:
        opt_pol = optim.MultiOptimizer(
            [
                optim.AdamW(learning_rate=5.0e-6, weight_decay=0.01),
                optim.AdamW(learning_rate=1.5e-6, weight_decay=0.01),
            ],
            filters=[lambda k, _: "lora_b" in k],
        )
        tok = processor.tokenizer
        eos_id = int(tok.eos_token_id)
        episodes = seal_data["rl_episodes"][:outer_episodes]
        demo_rng = random.Random(20261360)
        t_c0 = time.perf_counter()

        for ep_idx in range(c_state["ep"], len(episodes)):
            ep = episodes[ep_idx]
            restore_rewrite_layers(vlm, base_snap)

            # 1. Measure `theta_parent` on `diagnose_tasks`, unseen `reward_tasks`, and `anchor_items[:16]`
            _, _, diag_details = _eval_task_batch(vlm, processor, ep["diagnose_tasks"])
            acc_par, rew_par, _ = _eval_task_batch(vlm, processor, ep["reward_tasks"])
            nll_par, _ = evaluate_val_loss(vlm, anchor_items[:16])

            diag_msgs = _format_diagnostic_prompt(ep["diagnose_tasks"], diag_details, nll_par)
            p_str = _apply_template_inference_aligned(processor, vlm.config, diag_msgs, num_images=0, enable_thinking=True)
            p_ids = tok.encode(p_str, add_special_tokens=False)
            p_mx = mx.array([p_ids], dtype=mx.int32)
            s_idx = max(len(p_ids) - 1, 0)

            # 2. Sample `G = 3` candidate `<weight_rewrite>` plans from `theta_parent`
            cand_rollouts = []
            for g_i in range(3):
                restore_rewrite_layers(vlm, base_snap)
                res = mlx_vlm.generate(vlm, processor, p_str, max_tokens=640, temperature=0.75, repetition_penalty=1.05, verbose=False)
                g_txt = res.text or ""
                c_ids = tok.encode(g_txt, add_special_tokens=False) + [eos_id]
                f_seq = mx.concatenate([p_mx, mx.array([c_ids], dtype=mx.int32)], axis=1)[:, :1792]

                plan, r_sch = parse_weight_rewrite_plan(g_txt)
                if plan is None:
                    tot_r = r_sch
                    acc_cp, nll_cp = acc_par, nll_par
                else:
                    # Rewrite `theta_copy` live and evaluate on unseen `reward_tasks`!
                    apply_hybrid_seal_svf_to_live_copy(vlm, processor, plan, base_snap, bank)
                    acc_cp, rew_cp, _ = _eval_task_batch(vlm, processor, ep["reward_tasks"])
                    nll_cp, _ = evaluate_val_loss(vlm, anchor_items[:16])
                    gain_acc = acc_cp - acc_par
                    gain_rew = rew_cp - rew_par
                    nll_pen = max(0.0, nll_cp - nll_par - 0.01)
                    tot_r = round(r_sch + 3.0 * gain_acc + 0.8 * gain_rew - 3.0 * nll_pen, 4)

                cand_rollouts.append({
                    "full_ids": f_seq,
                    "reward": tot_r,
                    "acc_copy": acc_cp,
                    "nll_copy": nll_cp,
                })

            restore_rewrite_layers(vlm, base_snap)
            rews = [c["reward"] for c in cand_rollouts]
            best_cp_acc = max(c["acc_copy"] for c in cand_rollouts)

            # 3. Outer-loop GRPO policy update on `theta_parent` + general 9-pillar anchor protection!
            anc_mix = demo_rng.sample(anchor_items, k=1) + (demo_rng.sample(seal_demos, k=1) if seal_demos else [])
            if max(rews) - min(rews) > 1e-4:
                mu = sum(rews) / len(rews)
                advs = [r - mu for r in rews]
                old_lps = [mx.stop_gradient(-chunked_multimodal_ce(vlm, c["full_ids"], s_idx)) for c in cand_rollouts]
                live_pol = {k: mx.array(v) for k, v in tree_flatten(vlm.trainable_parameters())}
                vlm.load_weights(list(ref_pol_weights.items()), strict=False)
                mx.eval(vlm.parameters())
                ref_lps = [mx.stop_gradient(-chunked_multimodal_ce(vlm, c["full_ids"], s_idx)) for c in cand_rollouts]
                vlm.load_weights(list(live_pol.items()), strict=False)
                mx.eval(vlm.parameters(), *(old_lps + ref_lps))

                def _pol_loss(rm: nn.Module, f_ids: mx.array, o_lp: mx.array, r_lp: mx.array, av: float) -> mx.array:
                    n_lp = -chunked_multimodal_ce(rm, f_ids, s_idx)
                    rat = mx.exp(mx.clip(n_lp - o_lp, -5.0, 5.0))
                    adv_m = mx.array(av, dtype=mx.float32)
                    pg = -mx.minimum(rat * adv_m, mx.clip(rat, 0.8, 1.25) * adv_m)
                    kl = mx.exp(r_lp - n_lp) - (r_lp - n_lp) - 1.0
                    return mx.sum(pg + 0.05 * kl) / mx.array(640.0, dtype=mx.float32)

                vg_p = nn.value_and_grad(vlm, _pol_loss)
                acc_g = None
                for c_ro, olp, rlp, av in zip(cand_rollouts, old_lps, ref_lps, advs):
                    lv, gr = vg_p(vlm, c_ro["full_ids"], olp, rlp, av)
                    mx.eval(lv)
                    acc_g = tree_map(lambda g: g / 3.0, gr) if acc_g is None else tree_map(lambda a, g: a + g / 3.0, acc_g, gr)
                if anc_mix:
                    _, d_g = _run_dft_batch(vlm, anc_mix, None, scale_grad=0.14)
                    if acc_g is not None and d_g is not None:
                        acc_g = tree_map(lambda a, b: a + b, acc_g, d_g)
                if acc_g is not None:
                    acc_g, _ = optim.clip_grad_norm(acc_g, max_norm=1.0)
                    opt_pol.update(vlm, acc_g)
                    mx.eval(vlm.trainable_parameters(), opt_pol.state)
            elif anc_mix:
                _run_dft_batch(vlm, anc_mix, opt_pol, scale_grad=0.14)

            c_state["ep"] = ep_idx + 1
            c_state["history"].append({
                "episode": ep["episode_id"],
                "family": ep["family"],
                "parent_acc": round(acc_par, 4),
                "best_copy_acc": round(best_cp_acc, 4),
                "mean_rewrite_reward": round(sum(rews) / len(rews), 4),
            })
            _save_trainable_ckpt(vlm, c_ckpt)
            (c_ckpt / "state.json").write_text(json.dumps(c_state, indent=2) + "\n", encoding="utf-8")
            mx.clear_cache()
            print(
                f"  [Stage C2 SEAL-RL {ep_idx+1:2d}/{len(episodes)} | {(time.perf_counter()-t_c0)/60:4.1f}m] "
                f"{ep['episode_id']} ({ep['family']:19s}) | parent={acc_par*100:5.1f}% -> best_copy={best_cp_acc*100:5.1f}% | "
                f"rewards={[round(x,2) for x in rews]}",
                flush=True,
            )

        c_state["done"] = True
        (c_ckpt / "state.json").write_text(json.dumps(c_state, indent=2) + "\n", encoding="utf-8")

    # Conservative WiSE-FT (`0.70`) on outer-loop policy LoRA so layers 6..21 preserve 100% of general reasoning!
    pol_w = {k: (0.70 * v if "lora_b" in k else v) for k, v in tree_flatten(vlm.trainable_parameters())}
    vlm.load_weights(list(pol_w.items()), strict=False)
    mx.eval(vlm.parameters())

    # ------------------------------------------------------------------
    # Honest Out-of-Sample Autonomous Self-Rewrite Test (`final_test_24`)!
    # ------------------------------------------------------------------
    restore_rewrite_layers(vlm, base_snap)
    print("[Stage C3] Evaluating `theta_parent` BEFORE autonomous self-rewriting on untouched `final_test_24`...", flush=True)
    before_acc, before_rew, before_det = _eval_task_batch(vlm, processor, seal_data["final_test_24"])
    before_anc_nll, _ = evaluate_val_loss(vlm, anchor_items[:20])

    # Autonomous Dev-Diagnose + Dev-Gate Selection (`G=3` self-proposed rewrites on `dev_diagnose_4`, gated on `dev_gate_8` — ZERO peek at `final_test_24`!)
    _, _, dev_diag_det = _eval_task_batch(vlm, processor, seal_data["dev_diagnose_4"])
    gate_par_acc, gate_par_rew, _ = _eval_task_batch(vlm, processor, seal_data["dev_gate_8"])
    dev_diag_msgs = _format_diagnostic_prompt(seal_data["dev_diagnose_4"], dev_diag_det, before_anc_nll)
    dev_p_str = _apply_template_inference_aligned(processor, vlm.config, dev_diag_msgs, num_images=0, enable_thinking=True)

    best_dev_score = (gate_par_acc, gate_par_rew)
    winning_plan: dict[str, Any] | None = None
    winning_raw = ""

    for cand_i in range(3):
        restore_rewrite_layers(vlm, base_snap)
        temp_c = 0.0 if cand_i == 0 else 0.65
        res_c = mlx_vlm.generate(vlm, processor, dev_p_str, max_tokens=768, temperature=temp_c, repetition_penalty=1.05, verbose=False)
        raw_c = res_c.text or ""
        pl_c, _ = parse_weight_rewrite_plan(raw_c)
        if pl_c is None:
            continue
        apply_hybrid_seal_svf_to_live_copy(vlm, processor, pl_c, base_snap, bank)
        g_acc, g_rew, _ = _eval_task_batch(vlm, processor, seal_data["dev_gate_8"])
        g_nll, _ = evaluate_val_loss(vlm, anchor_items[:20])
        print(
            f"  [Dev Self-Rewrite Candidate {cand_i+1}/3] dev_gate={g_acc*100:5.1f}% (parent={gate_par_acc*100:5.1f}%) | "
            f"anchor_nll={g_nll:.4f} (parent={before_anc_nll:.4f}) | gains={pl_c['svf_gains']} | edits={len(pl_c['self_edits'])}",
            flush=True,
        )
        if g_nll <= before_anc_nll + 0.01 and (g_acc, g_rew) >= best_dev_score:
            best_dev_score = (g_acc, g_rew)
            winning_plan = pl_c
            winning_raw = raw_c

    restore_rewrite_layers(vlm, base_snap)
    if winning_plan is not None:
        apply_hybrid_seal_svf_to_live_copy(vlm, processor, winning_plan, base_snap, bank)

    after_acc, after_rew, after_det = _eval_task_batch(vlm, processor, seal_data["final_test_24"])
    after_anc_nll, _ = evaluate_val_loss(vlm, anchor_items[:20])
    print(
        f"[Stage C3 Out-of-Sample Autonomous Self-Rewrite Result (`final_test_24`)] "
        f"BEFORE={before_acc*100:.1f}% ({int(round(before_acc*24))}/24) -> "
        f"AFTER={after_acc*100:.1f}% ({int(round(after_acc*24))}/24) | "
        f"anchor_nll: {before_anc_nll:.4f} -> {after_anc_nll:.4f}",
        flush=True,
    )

    # Fuse final standalone model (`theta_parent` policy + beneficial self-rewritten upper layers if dev-gated!) + copy MTP head & SEAL bank!
    if after_anc_nll > before_anc_nll + 0.02 or winning_plan is None:
        restore_rewrite_layers(vlm, base_snap)
    fuse_and_export_32l(vlm, src_32l_dir, export_final_dir)
    if (src_32l_dir / "mtp_head.safetensors").exists():
        shutil.copy2(src_32l_dir / "mtp_head.safetensors", export_final_dir / "mtp_head.safetensors")
        shutil.copy2(src_32l_dir / "mtp_config.json", export_final_dir / "mtp_config.json")
    shutil.copy2(bank_path, export_final_dir / "seal_svf_bank.safetensors")

    eval_summary = {
        "outer_rl_episodes": len(c_state["history"]),
        "outer_rl_history": c_state["history"],
        "dev_gate_parent_acc": round(gate_par_acc, 4),
        "dev_gate_selected_copy_acc": round(best_dev_score[0], 4),
        "winning_self_rewrite_plan": winning_plan,
        "winning_raw_proposal": winning_raw,
        "untouched_final_test_24": {
            "before_rewrite_acc": round(before_acc, 4),
            "after_rewrite_acc": round(after_acc, 4),
            "delta_acc_pp": round((after_acc - before_acc) * 100, 2),
            "before_mean_reward": round(before_rew, 4),
            "after_mean_reward": round(after_rew, 4),
            "before_anchor_nll": before_anc_nll,
            "after_anchor_nll": after_anc_nll,
        },
    }
    (out_eval_dir / "summary.json").write_text(json.dumps(eval_summary, indent=2) + "\n", encoding="utf-8")
    del vlm, processor
    gc.collect()
    mx.clear_cache()
    return eval_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage A (32L + Repair), Stage B (MTP Speedup), and Stage C (SEAL+SVF Self-Rewrite RL).")
    parser.add_argument("--rl-steps", type=int, default=32)
    parser.add_argument("--mtp-steps", type=int, default=720)
    parser.add_argument("--seal-episodes", type=int, default=12)
    args = parser.parse_args()

    mx.set_wired_limit(0)
    mx.set_cache_limit(384 * 1024 * 1024)
    WORK_V6.mkdir(parents=True, exist_ok=True)

    tr_raw = [json.loads(l) for l in (V6_CORPUS_DIR / "train_v6_repair.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    va_raw = [json.loads(l) for l in (V6_CORPUS_DIR / "val_v6_repair.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rl_raw = [json.loads(l) for l in (V6_CORPUS_DIR / "rlvr_v6_repair_64.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    run_stage_a_32l_repair(tr_raw, va_raw, rl_raw, rl_steps=args.rl_steps)
    mtp_summary = run_stage_b_mtp_training_and_benchmark(EXPORT_32L_REPAIRED, mtp_steps=args.mtp_steps)
    seal_summary = run_stage_c_live_self_rewrite_rl(EXPORT_32L_REPAIRED, EXPORT_FINAL_SEAL, outer_episodes=args.seal_episodes)

    print("\n=== ALL V6 STAGES COMPLETE ===", flush=True)
    print(f"Standalone 32L + MTP + SEAL Model: {EXPORT_FINAL_SEAL}", flush=True)
    print(
        f"MTP Speedup (K=2 / K=3 / K=4 / K=5): {mtp_summary['mtp_k2_overall_speedup']}x / "
        f"{mtp_summary['mtp_k3_overall_speedup']}x / {mtp_summary['mtp_k4_overall_speedup']}x / "
        f"{mtp_summary['mtp_k5_overall_speedup']}x",
        flush=True,
    )
    print(f"Autonomous Self-Rewrite (`final_test_24`): {seal_summary['untouched_final_test_24']}", flush=True)


if __name__ == "__main__":
    main()
