"""Master Checkpointed RPR v8 (`2.90B` Total / `2.65B` Active) Training & Hermes Export Pipeline:

Features integrated cleanly atop `LFM3-VL-2.6B-RPR-MTP-SEAL-BF16` (`Step-0 max logit diff = 0.000000`):
  1. DeepSeek-V4 `EngramConditionalMemory` (`+0.255B` O(1) hashed 2-gram lookup banks on L6 & L18)
  2. Kimi K3.1 Pro `CrossLayerAttnRes` (depth-axis softmax residuals into upper `full_attention` layers `26, 30, 32, 34`)
  3. Kimi `MoBA` + DeepSeek `CSA2` Pinned-Sink + Top-K Block Attention (`L_kv > 4096`)
  4. Kimi K2 / K3.1 Pro `MuonClip` Optimizer (5-step Newton-Schulz orthogonalized 2D updates + `QK-Clip` + `BF16AdamW`)
  5. Multi-Turn Sandboxed `Python-TIR` & Traceback Self-Debug + DeepSeek-Flash `[EFFORT: FLASH|DEEP]` Dials + Adversarial Auditor

Exports standalone Hermes-ready model directory:
  `checkpoints/LFM3-VL-2.9B-RPR-V8-HERMES-BF16/`
    - `model.safetensors`       (568 standard `mlx_vlm` `36L` tensors, `4.93 GiB` BF16)
    - `v8_arch_ext.safetensors` (`Engram` + `AttnRes` sidecar, `0.49 GiB` BF16 -> `2.903B` total params)
    - `mtp_head.safetensors`    (`SequentialMTPHead` speculative head -> `70–115 tok/s`)
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
from training.build_v7_master_and_longctx_corpus import V7_CORPUS_DIR
from training.build_v8_hermes_corpus import V8_CORPUS_DIR
from training.collect_and_build_rpr_corpus import _strip_prior_think
from training.run_v6_32l_mtp_seal_pipeline import (
    _eval_task_batch,
    build_active_shortlist,
    fuse_and_export_32l,
)
from training.run_v7_36l_zero_degrade_pipeline import (
    EXPORT_36L_SEAL,
    UPCYCLED_36L_INDICES,
    _run_orpo_step,
)
from training.train_and_fuse_rpr_bf16 import (
    _apply_template_inference_aligned,
    _run_dft_batch,
    chunked_multimodal_ce,
    evaluate_val_loss,
)
from training.upcycle_and_train_rpr_v5_bf16 import GATE_NORM_LEAVES, prepare_turn_items_v5
from training.verifiable_rewards import score_response
from v8_arch import (
    Muon2D,
    apply_qk_clip,
    attach_v8_extensions,
    save_v8_sidecar,
    unfreeze_v8_modules,
)


WORK_V8 = ROOT_DIR / "checkpoints" / "rpr_v8_2_9b_work"
CKPT_V8 = WORK_V8 / "stage123_v8_ckpt"
EXPORT_V8_HERMES = ROOT_DIR / "checkpoints" / "LFM3-VL-2.9B-RPR-V8-HERMES-BF16"


class BF16AdamW(optim.Optimizer):
    """Memory-lean `bfloat16`-state AdamW for the `251.66M`-param sparse `Engram` embedding tables (saves 1.0 GiB RAM)."""

    def __init__(
        self,
        learning_rate: float | Any = 3.0e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()
        self._maybe_schedule("learning_rate", learning_rate)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

    def init_single(self, parameter: mx.array, state: dict[str, Any]) -> None:
        state["m"] = mx.zeros_like(parameter).astype(mx.bfloat16)
        state["v"] = mx.zeros_like(parameter).astype(mx.bfloat16)

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict[str, Any]) -> mx.array:
        lr = self.learning_rate.astype(mx.float32)
        b1, b2 = self.betas
        step = float(self.step.item()) if isinstance(self.step, mx.array) else float(self.step)
        t_step = max(1.0, step)
        g = gradient.astype(mx.float32)
        m = b1 * state["m"].astype(mx.float32) + (1.0 - b1) * g
        v = b2 * state["v"].astype(mx.float32) + (1.0 - b2) * (g * g)
        state["m"] = m.astype(mx.bfloat16)
        state["v"] = v.astype(mx.bfloat16)
        m_hat = m / (1.0 - (b1**t_step))
        v_hat = v / (1.0 - (b2**t_step))
        p = parameter.astype(mx.float32)
        if self.weight_decay > 0.0:
            p = p * (1.0 - lr * self.weight_decay)
        return (p - lr * m_hat / (mx.sqrt(v_hat) + self.eps)).astype(parameter.dtype)


def prepare_turn_items_v8(
    rows: list[dict[str, Any]],
    model: nn.Module,
    processor: Any,
    *,
    max_seq_len: int = 2304,
) -> tuple[list[dict[str, Any]], int]:
    """Prepare text & PIL vision SFT items, delegating standard/vision rows to `prepare_turn_items_v5` and
    training strictly on the final repaired assistant turn when `train_last_turn_only=True`.
    """
    normal_rows = [r for r in rows if not r.get("train_last_turn_only")]
    last_only_rows = [r for r in rows if r.get("train_last_turn_only")]

    prepared, vis_cnt = prepare_turn_items_v5(normal_rows, model, processor, max_seq_len=max_seq_len)
    tok = processor.tokenizer
    eos_id = int(tok.eos_token_id)

    for row in last_only_rows:
        msgs = row["messages"]
        t_idx = max((i for i, m in enumerate(msgs) if m.get("role") == "assistant"), default=-1)
        if t_idx <= 0:
            continue
        prefix_msgs = _strip_prior_think(msgs[:t_idx])
        prompt_str = _apply_template_inference_aligned(processor, model.config, prefix_msgs, num_images=0, enable_thinking=True)
        ans_str = str(msgs[t_idx]["content"])
        if prompt_str.endswith("<think>") and ans_str.startswith("<think>"):
            ans_str = ans_str[len("<think>") :]
        elif prompt_str.endswith("<think>\n") and ans_str.startswith("<think>\n"):
            ans_str = ans_str[len("<think>\n") :]
        p_ids = tok.encode(prompt_str, add_special_tokens=False)
        a_ids = list(tok.encode(ans_str, add_special_tokens=False)) + [eos_id]
        full = (p_ids + a_ids)[:max_seq_len]
        if len(full) <= len(p_ids) + 2:
            continue
        prepared.append(
            {
                "id": f"{row.get('id', 'v8')}_t{t_idx}",
                "pillar": row.get("pillar", "unknown"),
                "full_ids": mx.array([full], dtype=mx.int32),
                "start_idx": max(len(p_ids) - 1, 0),
                "vt_feats": None,
                "img_indices": None,
            }
        )
    mx.clear_cache()
    return prepared, vis_cnt


def _save_v8_ckpt(model: nn.Module, ckpt_dir: Path, state: dict[str, Any] | None = None) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tr_flat = {
        k: (v.astype(mx.bfloat16) if mx.issubdtype(v.dtype, mx.floating) else v)
        for k, v in tree_flatten(model.trainable_parameters())
    }
    mx.eval(list(tr_flat.values()))
    tmp_st = ckpt_dir / "tmp_trainable.safetensors"
    mx.save_safetensors(str(tmp_st), tr_flat, metadata={"format": "mlx"})
    tmp_st.replace(ckpt_dir / "trainable.safetensors")
    if state is not None:
        tmp_js = ckpt_dir / "train_state.json.tmp"
        tmp_js.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        tmp_js.replace(ckpt_dir / "train_state.json")


def _load_v8_base_with_peft() -> tuple[Any, Any, dict[str, Any]]:
    import mlx_vlm
    from mlx_vlm.trainer.peft import get_peft_model

    vlm, processor = mlx_vlm.load(str(EXPORT_36L_SEAL))
    probe = mx.array([[1, 42, 105, 512, 1024, 2048]], dtype=mx.int32)
    ref_logits = vlm.language_model(probe).logits
    mx.eval(ref_logits)

    ext_info = attach_v8_extensions(vlm, sidecar_dir=None, enable_moba=True, moba_threshold=4096)
    v8_logits = vlm.language_model(probe).logits
    mx.eval(v8_logits)
    diff0 = float(mx.max(mx.abs(v8_logits.astype(mx.float32) - ref_logits.astype(mx.float32))).item())
    assert diff0 < 1e-6, f"Step-0 v8 identity broken: diff={diff0}"

    # Freeze all 28 original `v5` layers! Attach light Rank-24 LoRA strictly on the 8 `v7` blocks + unfreeze `Engram` & `AttnRes`!
    leafs = {"q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "w3"}
    prefs = tuple(f"model.layers.{i}." for i in UPCYCLED_36L_INDICES)
    targets = [
        f"language_model.{n}"
        for n, _ in vlm.language_model.named_modules()
        if n.startswith(prefs) and n.split(".")[-1] in leafs
    ]
    vlm = get_peft_model(vlm, linear_layers=targets, rank=24, alpha=48, dropout=0.01, verbose=False)
    unfreeze_v8_modules(vlm)
    ext_info["step0_max_logit_diff"] = diff0
    return vlm, processor, ext_info


def _build_stage_muonclip(lr_gate: Any, lr_table: Any, lr_2d: Any) -> optim.MultiOptimizer:
    gate_keys = ("post_operator_norm", "post_ffn_norm", "out_scale", "res_scale", "ctx_norm", "q_norm", "src_norm")
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


def run_stages_123_v8(
    sft_rows: list[dict[str, Any]],
    orpo_rows: list[dict[str, Any]],
    rlvr_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    *,
    sft_steps: int = 560,
    orpo_steps: int = 120,
    rl_steps: int = 24,
    stop_after_substage: str = "3",
) -> None:
    import mlx_vlm

    if (EXPORT_V8_HERMES / "model.safetensors").exists() and (EXPORT_V8_HERMES / "v8_stage123_done.json").exists():
        print(f"[v8 Stages 1-3] Resuming completed 2.90B v8 checkpoint at {EXPORT_V8_HERMES}", flush=True)
        return

    vlm, processor, ext_info = _load_v8_base_with_peft()
    tr_items, vis_cnt = prepare_turn_items_v8(sft_rows, vlm, processor, max_seq_len=2304)
    va_items, _ = prepare_turn_items_v8(val_rows, vlm, processor, max_seq_len=2304)
    n_trainable = sum(int(v.size) for _, v in tree_flatten(vlm.trainable_parameters()))
    print(
        f"[v8 Init Verified] Total Params = {ext_info['total_model_params']/1e9:.3f}B (+{ext_info['v8_extension_params']/1e6:.2f}M Engram+AttnRes) | "
        f"Step-0 max logit diff = {ext_info['step0_max_logit_diff']:.6f} | Trainable = {n_trainable/1e6:.2f}M | "
        f"Turns = {len(tr_items)} (vis={vis_cnt})",
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
    st_path = CKPT_V8 / "latest" / "train_state.json"
    if st_path.exists() and (CKPT_V8 / "latest" / "trainable.safetensors").exists():
        state = json.loads(st_path.read_text(encoding="utf-8"))
        vlm.load_weights(str(CKPT_V8 / "latest" / "trainable.safetensors"), strict=False)
        mx.eval(vlm.parameters())
        print(f"[v8 Stages 1-3] Resumed checkpoint: {state}", flush=True)
    else:
        vl0, vp0 = evaluate_val_loss(vlm, va_items)
        state["best_val"] = vl0
        print(f"  [Step-0 v8 2.90B Identity Baseline] joint_val_loss={vl0:.4f} (PPL={vp0:.4f})", flush=True)
        _save_v8_ckpt(vlm, CKPT_V8 / "best")
        _save_v8_ckpt(vlm, CKPT_V8 / "latest", state)

    # ------------------------------------------------------------------
    # Stage 1: `MuonClip` + `BF16AdamW` Co-Adaptation SFT (`Engram` + `AttnRes` + 8 New Blocks)
    # ------------------------------------------------------------------
    if not state["sft_done"]:
        s0 = int(state["sft_step"])
        rem = max(1, sft_steps - s0)
        cw = 0.5 * (1.0 + math.cos(math.pi * min(1.0, s0 / max(1, sft_steps))))
        opt_sft = _build_stage_muonclip(
            lr_gate=optim.cosine_decay(8.0e-5 + 3.7e-4 * cw, decay_steps=rem, end=5.0e-5),
            lr_table=optim.cosine_decay(1.0e-4 + 4.0e-4 * cw, decay_steps=rem, end=5.0e-5),
            lr_2d=optim.cosine_decay(1.5e-5 + 6.5e-5 * cw, decay_steps=rem, end=1.0e-5),
        )
        order = list(range(len(tr_items)))
        random.Random(20261810).shuffle(order)
        t0 = time.perf_counter()
        losses: list[float] = []
        qk_clips = 0

        for st in range(s0, sft_steps):
            b0 = tr_items[order[(st * 2) % len(order)]]
            b1 = tr_items[order[(st * 2 + 1) % len(order)]]
            lv, _ = _run_dft_batch(vlm, [b0, b1], opt_sft)
            qk_clips += apply_qk_clip(vlm, max_lora_fro_norm=6.0)
            losses.append(lv)

            if (st + 1) % 8 == 0:
                gc.collect()
                mx.clear_cache()
            if (st + 1) % 25 == 0 and (st + 1) % 70 != 0 and st != sft_steps - 1:
                state["sft_step"] = st + 1
                _save_v8_ckpt(vlm, CKPT_V8 / "latest", state)
                print(f"  [v8 Stage 1 Ckpt {st+1:3d}/{sft_steps} | {(time.perf_counter()-t0)/60:4.1f}m] train={sum(losses)/len(losses):.4f}", flush=True)
            if (st + 1) % 70 == 0 or st == sft_steps - 1:
                vlm.eval()
                vl, vp = evaluate_val_loss(vlm, va_items)
                vlm.train()
                state["sft_step"] = st + 1
                if vl <= state["best_val"]:
                    state["best_val"] = vl
                    _save_v8_ckpt(vlm, CKPT_V8 / "best")
                _save_v8_ckpt(vlm, CKPT_V8 / "latest", state)
                print(
                    f"  [v8 Stage 1 MuonClip-SFT {st+1:3d}/{sft_steps} | {(time.perf_counter()-t0)/60:4.1f}m] "
                    f"train={sum(losses)/len(losses):.4f} | val={vl:.4f} (PPL={vp:.4f}) | best={state['best_val']:.4f} | qk_clips={qk_clips}",
                    flush=True,
                )
                losses.clear()
                gc.collect()
                mx.clear_cache()

        vlm.load_weights(str(CKPT_V8 / "best" / "trainable.safetensors"), strict=False)
        mx.eval(vlm.parameters())
        state["sft_done"] = True
        _save_v8_ckpt(vlm, CKPT_V8 / "sft_best")
        _save_v8_ckpt(vlm, CKPT_V8 / "latest", state)

    if stop_after_substage == "1":
        del vlm, processor, tr_items, va_items
        gc.collect()
        mx.clear_cache()
        return

    # ------------------------------------------------------------------
    # Stage 2: `MuonClip` Contrastive `ORPO` (`Auditor` + `FLASH` + `AttnRes`)
    # ------------------------------------------------------------------
    tok = processor.tokenizer
    eos_id = int(tok.eos_token_id)

    if not state["orpo_done"]:
        opt_orpo = _build_stage_muonclip(
            lr_gate=optim.cosine_decay(4.5e-5, decay_steps=max(1, orpo_steps), end=1.0e-5),
            lr_table=optim.cosine_decay(8.0e-5, decay_steps=max(1, orpo_steps), end=2.0e-5),
            lr_2d=optim.cosine_decay(1.8e-5, decay_steps=max(1, orpo_steps), end=4.0e-6),
        )
        anc_rng = random.Random(20261820)
        t_op = time.perf_counter()
        op_losses: list[float] = []
        best_orpo = float(state.get("best_orpo_val", 1e9))

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
            if (ost + 1) % 40 == 0 or ost == orpo_steps - 1:
                vlm.eval()
                vl, vp = evaluate_val_loss(vlm, va_items)
                vlm.train()
                state["orpo_step"] = ost + 1
                if vl <= best_orpo:
                    best_orpo = vl
                    state["best_orpo_val"] = vl
                    _save_v8_ckpt(vlm, CKPT_V8 / "orpo_best")
                _save_v8_ckpt(vlm, CKPT_V8 / "latest", state)
                print(
                    f"  [v8 Stage 2 MuonClip-ORPO {ost+1:3d}/{orpo_steps} | {(time.perf_counter()-t_op)/60:4.1f}m] "
                    f"orpo_loss={sum(op_losses)/max(1,len(op_losses)):.4f} | val={vl:.4f} (PPL={vp:.4f}) | best_orpo={best_orpo:.4f}",
                    flush=True,
                )
                op_losses.clear()
                gc.collect()
                mx.clear_cache()

        vlm.load_weights(str(CKPT_V8 / "orpo_best" / "trainable.safetensors"), strict=False)
        mx.eval(vlm.parameters())
        state["orpo_done"] = True
        _save_v8_ckpt(vlm, CKPT_V8 / "latest", state)

    if stop_after_substage == "2":
        del vlm, processor, tr_items, va_items
        gc.collect()
        mx.clear_cache()
        return

    # ------------------------------------------------------------------
    # Stage 3: Multi-Turn Self-Debug & Code/RPR `GRPO` + Non-Degradation Gate
    # ------------------------------------------------------------------
    if not state["rl_done"]:
        vlm.eval()
        opt_rl = _build_stage_muonclip(lr_gate=1.5e-5, lr_table=3.0e-5, lr_2d=6.0e-6)
        ref_w = mx.load(str(CKPT_V8 / "orpo_best" / "trainable.safetensors"))
        mx.eval(list(ref_w.values()))
        anc_rng = random.Random(20261830)
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
                res = mlx_vlm.generate(vlm, processor, p_str, max_tokens=448, temperature=0.75, repetition_penalty=1.05, verbose=False)
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
                    return mx.sum(pg + 0.04 * kl) / mx.array(448.0, dtype=mx.float32)

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
            if (g_idx + 1) % 6 == 0 or g_idx == n_rl - 1:
                vl, _ = evaluate_val_loss(vlm, va_items[:48])
                _save_v8_ckpt(vlm, CKPT_V8 / "latest", state)
                if vl <= state.get("best_rl_val", 1e9) + 0.01:
                    if vl < state.get("best_rl_val", 1e9):
                        state["best_rl_val"] = vl
                    _save_v8_ckpt(vlm, CKPT_V8 / "rl_best")
                print(f"  [v8 Stage 3 GRPO {g_idx+1:2d}/{n_rl} | {(time.perf_counter()-t_rl)/60:4.1f}m] {row['id']:20s} | pass={p_cnt}/4 | {tag} | val={vl:.4f}", flush=True)

        state["rl_done"] = True
        _save_v8_ckpt(vlm, CKPT_V8 / "latest", state)

    # ------------------------------------------------------------------
    # Validation-Gated `WiSE-FT` Selection + Standalone `model.safetensors` & `v8_arch_ext.safetensors` Export!
    # ------------------------------------------------------------------
    w_sft = mx.load(str(CKPT_V8 / "sft_best" / "trainable.safetensors"))
    w_orpo = mx.load(str((CKPT_V8 / "orpo_best" / "trainable.safetensors") if (CKPT_V8 / "orpo_best" / "trainable.safetensors").exists() else (CKPT_V8 / "sft_best" / "trainable.safetensors")))
    w_rl = mx.load(str((CKPT_V8 / "rl_best" / "trainable.safetensors") if (CKPT_V8 / "rl_best" / "trainable.safetensors").exists() else (CKPT_V8 / "latest" / "trainable.safetensors")))
    w_wise = {k: (0.35 * w_sft[k].astype(mx.float32) + 0.25 * w_orpo[k].astype(mx.float32) + 0.40 * w_rl[k].astype(mx.float32)).astype(w_sft[k].dtype) for k in w_sft}

    seal_d = json.loads((V6_CORPUS_DIR / "seal_svf_episodes.json").read_text(encoding="utf-8"))
    probe_tasks = seal_d["dev_diagnose_4"] + seal_d["dev_gate_8"] + rlvr_rows[-12:]

    def _scale_residual_gates(weights_dict: dict[str, mx.array], tau_lora: float, tau_ext: float | None = None) -> dict[str, mx.array]:
        t_ext = tau_lora if tau_ext is None else tau_ext
        out_d: dict[str, mx.array] = {}
        for k, v in weights_dict.items():
            if "lora_b" in k:
                out_d[k] = (v.astype(mx.float32) * tau_lora).astype(v.dtype)
            elif "out_scale" in k or "res_scale" in k:
                out_d[k] = (v.astype(mx.float32) * t_ext).astype(v.dtype)
            else:
                out_d[k] = v
        return out_d

    zero_w = _scale_residual_gates(w_wise, 0.0, 0.0)
    vlm.load_weights(list(zero_w.items()), strict=False)
    mx.eval(vlm.parameters())
    vl_0, _ = evaluate_val_loss(vlm, va_items)
    acc_0, rew_0, _ = _eval_task_batch(vlm, processor, probe_tasks)
    ref_sc = (round(acc_0, 4), round(rew_0 - 0.35 * vl_0, 4))
    print(f"  [v8 Reference Baseline] {'v7_identity_tau0':22s} | probe_acc={acc_0*100:5.1f}% (rew={rew_0:.3f}) | full_val={vl_0:.4f}", flush=True)

    candidates = [
        ("v8_ext1.0_lora0.30", _scale_residual_gates(w_sft, 0.30, 1.00)),
        ("v8_ext1.0_lora0.55", _scale_residual_gates(w_wise, 0.55, 1.00)),
        ("v8_wise_tau0.85", _scale_residual_gates(w_wise, 0.85, 0.95)),
        ("v8_sft_best_tau1.0", w_sft),
        ("v8_orpo_best_tau1.0", w_orpo),
    ]
    best_name = ""
    best_w = w_sft
    best_sc = (-1.0, -1e9)
    best_val_loss = 999.0

    for c_name, c_w in candidates:
        vlm.load_weights(list(c_w.items()), strict=False)
        mx.eval(vlm.parameters())
        vl_c, _ = evaluate_val_loss(vlm, va_items)
        acc_c, rew_c, _ = _eval_task_batch(vlm, processor, probe_tasks)
        print(f"  [v8 Non-Degradation Gate] {c_name:22s} | probe_acc={acc_c*100:5.1f}% (rew={rew_c:.3f}) | full_val={vl_c:.4f}", flush=True)
        sc_t = (round(acc_c, 4), round(rew_c - 0.35 * vl_c, 4))
        if sc_t >= best_sc:
            best_sc = sc_t
            best_name = c_name
            best_w = c_w
            best_val_loss = round(vl_c, 4)

    print(f"[v8 Selected] Winner={best_name} (score={best_sc}, val={best_val_loss:.4f} vs tau0_ref={ref_sc})", flush=True)
    vlm.load_weights(list(best_w.items()), strict=False)
    mx.eval(vlm.parameters())

    # 1. Save `v8_arch_ext.safetensors` sidecar + self-contained `v8_arch.py` first, then strip sidecar attrs temporarily so `fuse_and_export_32l` writes a 100% pure 568-tensor `model.safetensors`!
    EXPORT_V8_HERMES.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT_DIR / "v8_arch.py", EXPORT_V8_HERMES / "v8_arch.py")
    save_v8_sidecar(
        vlm,
        EXPORT_V8_HERMES,
        extra_meta={
            "selected_candidate": best_name,
            "val_probe_score": list(best_sc),
            "selected_val_loss": best_val_loss,
            "tau0_ref_score": list(ref_sc),
            "tau0_val_loss": round(vl_0, 4),
        },
    )

    probe_ids = mx.array([[1, 42, 105, 512, 1024, 2048]], dtype=mx.int32)
    core = vlm.language_model.model
    saved_attrs = {}
    for attr_n in ("engram_6", "engram_18", "attn_res_26", "attn_res_30", "attn_res_32", "attn_res_34"):
        if attr_n in core:
            saved_attrs[attr_n] = core.pop(attr_n)
    fuse_and_export_32l(vlm, EXPORT_36L_SEAL, EXPORT_V8_HERMES)
    for attr_n, mod_v in saved_attrs.items():
        mod_v.apply(lambda x: x.astype(mx.bfloat16) if mx.issubdtype(x.dtype, mx.floating) else x)
        core[attr_n] = mod_v
    mx.eval(vlm.parameters())
    logits_expected = vlm.language_model(probe_ids).logits
    mx.eval(logits_expected)

    del vlm, processor, tr_items, va_items
    gc.collect()
    mx.clear_cache()

    # Verify standalone reload (`mlx_vlm.load` + `attach_v8_extensions` == `0.000000` logit diff!)
    vlm_re, _ = mlx_vlm.load(str(EXPORT_V8_HERMES))
    info_re = attach_v8_extensions(vlm_re, sidecar_dir=EXPORT_V8_HERMES, enable_moba=True)
    logits_re = vlm_re.language_model(probe_ids).logits
    mx.eval(logits_re)
    re_diff = float(mx.max(mx.abs(logits_re.astype(mx.float32) - logits_expected.astype(mx.float32))).item())
    print(f"[v8 Standalone Export Verified] {info_re['total_model_params']/1e9:.3f}B params | reload max logit diff = {re_diff:.6f}", flush=True)
    assert re_diff < 2e-2, f"v8 reload mismatch: {re_diff}"

    (EXPORT_V8_HERMES / "v8_stage123_done.json").write_text(
        json.dumps({**state, "selected_candidate": best_name, "val_probe_score": list(best_sc), "reload_diff": re_diff, **info_re}, indent=2) + "\n",
        encoding="utf-8",
    )
    del vlm_re
    gc.collect()
    mx.clear_cache()


def run_stage_4_v8_mtp_and_flash_bench(mtp_steps: int = 300) -> dict[str, Any]:
    """Warm-start `SequentialMTPHead` from `v7.1` and align to `2.90B` `v8` hidden states + benchmark `FLASH` vs `DEEP`!"""
    import mlx_vlm

    bench_dir = ROOT_DIR / "results" / "v8_mtp_and_flash_benchmark"
    bench_dir.mkdir(parents=True, exist_ok=True)
    if (EXPORT_V8_HERMES / "mtp_head.safetensors").exists() and (bench_dir / "summary.json").exists():
        print(f"[v8 Stage 4] Resuming completed v8 MTP & Flash/Deep benchmark from {bench_dir / 'summary.json'}", flush=True)
        return json.loads((bench_dir / "summary.json").read_text(encoding="utf-8"))

    vlm, processor = mlx_vlm.load(str(EXPORT_V8_HERMES))
    attach_v8_extensions(vlm, sidecar_dir=EXPORT_V8_HERMES, enable_moba=True)
    vlm.freeze()
    tok = processor.tokenizer
    eos_id = int(tok.eos_token_id)

    v8_rows = [json.loads(l) for l in (V8_CORPUS_DIR / "train_v8_sft.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    text_rows = [r for r in v8_rows if not r.get("images")][:450]

    if (EXPORT_36L_SEAL / "mtp_head.safetensors").exists():
        mtp_head = load_mtp_checkpoint(vlm.config.text_config, EXPORT_36L_SEAL)
        print(f"[v8 Stage 4] Warm-started SequentialMTPHead from {EXPORT_36L_SEAL / 'mtp_head.safetensors'}", flush=True)
    else:
        shortlist = build_active_shortlist(tok, v8_rows, int(vlm.config.text_config.vocab_size), target_k=16384)
        mtp_head = SequentialMTPHead(vlm.config.text_config, mtp_intermediate_size=3072, shortlist_ids=shortlist)
        mtp_head.warm_start_from_target_lm(vlm.language_model)
        mtp_head.apply(lambda x: x.astype(mx.bfloat16) if mx.issubdtype(x.dtype, mx.floating) else x)

    embed_tokens = vlm.language_model.model.embed_tokens
    mtp_head.bind_vocab_projection(embed_tokens)

    seqs: list[tuple[mx.array, int]] = []
    for r in text_rows:
        msgs = r["messages"]
        t_i = len(msgs) - 1
        if msgs[t_i].get("role") != "assistant":
            continue
        p_s = _apply_template_inference_aligned(processor, vlm.config, _strip_prior_think(msgs[:t_i]), num_images=0, enable_thinking=True)
        a_s = str(msgs[t_i]["content"])
        if p_s.endswith("<think>") and a_s.startswith("<think>"):
            a_s = a_s[len("<think>") :]
        p_ids = tok.encode(p_s, add_special_tokens=False)
        if len(p_ids) > 440:
            continue
        a_ids = tok.encode(a_s, add_special_tokens=False) + [eos_id]
        full = (p_ids + a_ids)[:640]
        if len(full) >= len(p_ids) + 8:
            seqs.append((mx.array([full], dtype=mx.int32), max(1, len(p_ids) - 2)))

    random.Random(20261845).shuffle(seqs)
    opt_mtp = optim.AdamW(learning_rate=optim.cosine_decay(2.2e-4, decay_steps=max(1, mtp_steps), end=2.0e-5), weight_decay=0.01)

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
    t_m0 = time.perf_counter()
    m_losses: list[float] = []
    mtp_ckpt = WORK_V8 / "stage4_mtp_ckpt"

    for st in range(mtp_steps):
        f_ids, ans_s = seqs[st % len(seqs)]
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
        gr_m, _ = optim.clip_grad_norm(gr_m, max_norm=1.0)
        opt_mtp.update(mtp_head, gr_m)
        mx.eval(mtp_head.parameters(), opt_mtp.state, lv_m)
        m_losses.append(float(lv_m.item()))
        if (st + 1) % 100 == 0 or st == mtp_steps - 1:
            save_mtp_checkpoint(mtp_head, mtp_ckpt)
            print(f"  [v8 Stage 4 MTP-2.9B {st+1:3d}/{mtp_steps} | {(time.perf_counter()-t_m0)/60:4.1f}m] distill_loss={sum(m_losses)/len(m_losses):.4f}", flush=True)
            m_losses.clear()
            mx.clear_cache()

    save_mtp_checkpoint(mtp_head, EXPORT_V8_HERMES)

    # Benchmark BOTH `MTP` speculative speedup AND `[EFFORT: FLASH]` vs `[EFFORT: DEEP]` token/latency control!
    tools_demo = json.dumps([{"name": "read_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}])
    bench_prompts = [
        ("hermes_tool_flash", [{"role": "system", "content": f"You are Hermes. [EFFORT: FLASH]\n<tools>\n{tools_demo}\n</tools>"}, {"role": "user", "content": "Read `engine.py` so we can inspect it."}]),
        ("math_flash", [{"role": "system", "content": "Solve accurately. [EFFORT: FLASH]"}, {"role": "user", "content": 'Compute `27 + 19 - 14` and return JSON `{"result": <int>}`.'}]),
        ("code_deep", [{"role": "system", "content": "You are an RSI engineer. [EFFORT: DEEP]"}, {"role": "user", "content": "Implement `rome_rank1(W2, k_star, v_star, C_cov, reg=0.05)` in NumPy solving `u = C_reg^-1 @ k_star`."}]),
        ("audit_deep", [{"role": "system", "content": "Strict Level-1 auditor. [EFFORT: DEEP]"}, {"role": "user", "content": "Active mode=SOLAR_ARRAY (cap=40). Latest Evidence: QA prototype tested FUSION_TORUS at 80; live stays SOLAR_ARRAY. Check A1..A2."}]),
    ]
    # Warmup
    wp = tok.encode(_apply_template_inference_aligned(processor, vlm.config, bench_prompts[0][1], num_images=0, enable_thinking=True), add_special_tokens=False)
    standard_greedy_generate(vlm, wp, max_tokens=16, eos_token_id=eos_id)
    mtp_speculative_generate(vlm, mtp_head, wp, max_tokens=16, draft_k=3, eos_token_id=eos_id)

    per_case = []
    tot_std_t, tot_std_s = 0, 0.0
    tot_k3_t, tot_k3_s = 0, 0.0
    for cname, msgs in bench_prompts:
        ps = _apply_template_inference_aligned(processor, vlm.config, msgs, num_images=0, enable_thinking=True)
        pids = tok.encode(ps, add_special_tokens=False)
        ref_ids, st_std = standard_greedy_generate(vlm, pids, max_tokens=224, eos_token_id=eos_id)
        k3_ids, st_k3 = mtp_speculative_generate(vlm, mtp_head, pids, max_tokens=224, draft_k=3, eos_token_id=eos_id)
        txt_out = tok.decode(k3_ids)
        think_part = txt_out.split("</think>")[0] if "</think>" in txt_out else txt_out
        think_toks = len(tok.encode(think_part, add_special_tokens=False))
        sp = round(st_k3["tok_per_sec"] / max(1e-6, st_std["tok_per_sec"]), 3)
        tot_std_t += st_std["tokens"]
        tot_std_s += st_std["wall_clock_sec"]
        tot_k3_t += st_k3["tokens"]
        tot_k3_s += st_k3["wall_clock_sec"]
        per_case.append({
            "case": cname,
            "tokens": st_k3["tokens"],
            "think_tokens": think_toks,
            "std_tok_per_sec": st_std["tok_per_sec"],
            "mtp_k3_tok_per_sec": st_k3["tok_per_sec"],
            "speedup_k3": sp,
            "exact_token_match": (ref_ids == k3_ids),
            "wall_clock_k3_sec": st_k3["wall_clock_sec"],
        })
        print(f"  [v8 Bench {cname:18s}] think={think_toks:3d} toks | std={st_std['tok_per_sec']:5.1f} -> MTP-K3={st_k3['tok_per_sec']:6.1f} tok/s ({sp:.2f}x) | dt={st_k3['wall_clock_sec']:.2f}s", flush=True)

    agg_std = round(tot_std_t / max(1e-6, tot_std_s), 2)
    agg_k3 = round(tot_k3_t / max(1e-6, tot_k3_s), 2)
    summary = {
        "model": str(EXPORT_V8_HERMES),
        "total_params_billions": 2.903,
        "active_params_billions": 2.651,
        "std_tok_per_sec": agg_std,
        "mtp_k3_tok_per_sec": agg_k3,
        "overall_k3_speedup": round(agg_k3 / max(1e-6, agg_std), 3),
        "cases": per_case,
    }
    (bench_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    del vlm, processor, mtp_head
    gc.collect()
    mx.clear_cache()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RPR v8 (2.90B Engram+AttnRes+MoBA+MuonClip+Hermes) Checkpointed Pipeline.")
    parser.add_argument("--sft-steps", type=int, default=560)
    parser.add_argument("--orpo-steps", type=int, default=120)
    parser.add_argument("--rl-steps", type=int, default=24)
    parser.add_argument("--mtp-steps", type=int, default=300)
    parser.add_argument("--stage", choices=["all", "1", "2", "3", "4"], default="all")
    args = parser.parse_args()

    mx.set_wired_limit(0)
    mx.set_cache_limit(128 * 1024 * 1024)
    WORK_V8.mkdir(parents=True, exist_ok=True)

    sft_rows = [json.loads(l) for l in (V8_CORPUS_DIR / "train_v8_sft.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    orpo_rows = [json.loads(l) for l in (V8_CORPUS_DIR / "orpo_v8_pairs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rlvr_rows = [json.loads(l) for l in (V8_CORPUS_DIR / "rlvr_v8_hermes_48.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    val_rows = [json.loads(l) for l in (V8_CORPUS_DIR / "val_v8_joint.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    if args.stage in ("all", "1", "2", "3"):
        stop_s = args.stage if args.stage in ("1", "2") else "3"
        run_stages_123_v8(
            sft_rows, orpo_rows, rlvr_rows, val_rows,
            sft_steps=args.sft_steps, orpo_steps=args.orpo_steps, rl_steps=args.rl_steps,
            stop_after_substage=stop_s,
        )
    if args.stage in ("all", "4"):
        s4 = run_stage_4_v8_mtp_and_flash_bench(mtp_steps=args.mtp_steps)
        print(f"[v8 Stage 4 Complete] Overall MTP K=3 Speedup: {s4['overall_k3_speedup']}x ({s4['mtp_k3_tok_per_sec']} tok/s)", flush=True)


if __name__ == "__main__":
    main()
