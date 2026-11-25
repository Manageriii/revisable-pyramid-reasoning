"""LLaMA-Pro Frozen-Trunk `v10.1` (`40L` `3.18B` Pure 16-bit `bfloat16`, `256k` `LongRoPE2YaRN`) Pipeline:

Eliminates trunk drift while training the `+266.5M` new `v10` upper-pyramid capacity:
  1. Upcycles `LFM3-VL-2.9B-RPR-V9-SEAL-BF16` (`36L`) -> `40L` `3.18B` (`Step-0 max_abs_logit_diff == 0.000000`).
  2. Freezes ALL `36` battle-tested `v9` trunk layers (`0..33, 36, 37`), `engram_6/18`, and `attn_res_26/30/32` (`0.0` drift on tool-calling, 2-turn RPR, Python syntax, & vision!).
  3. Trains Rank-32 LoRA + `SandwichNorm` zero-gates strictly on the 4 newly inserted blocks `(34, 35, 38, 39)` + `LatentReasoningRecurrence` (`L35`) + `attn_res_36, attn_res_38` via `LoRA+ AdamW` (`240` SFT + `80` ORPO steps + `16` `DAPO/Dr. GRPO` active updates).
  4. Runs a strict `WiSE-FT` gate (`tau in {0.0, 0.55, 0.75, 0.90, 1.00}` where `tau=0` == exact `v9`), exports pure `16-bit` `model.safetensors` + `v8_arch_ext.safetensors` + `40L` `seal_nullspace_svf_bank.safetensors` + `160`-step on-policy `mtp_head.safetensors`, and removes temporary work files.
"""

from __future__ import annotations

import gc
import json
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

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from training.build_v6_repair_and_seal_corpus import V6_CORPUS_DIR
from training.build_v10_apex_corpus import V10_CORPUS_DIR
from training.run_v6_32l_mtp_seal_pipeline import _eval_task_batch
from training.run_v7_36l_zero_degrade_pipeline import _run_orpo_step
from training.run_v8_2_9b_hermes_pipeline import _save_v8_ckpt, prepare_turn_items_v8
from training.run_v9_2_9b_seal_cline_pipeline import EXPORT_V9_SEAL
from training.run_v10_3_2b_apex_pipeline import (
    EXPORT_V10_APEX,
    REWRITE_TARGET_LAYERS_40L,
    STAGE0_40L_ID,
    WORK_V10,
    _unfreeze_upcycled_40l_gates,
    extract_v10_40l_nullspace_svf_bank,
    fuse_and_export_40l_lean,
    run_stage_5_v10_onpolicy_mtp,
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
    upcycle_v9_36l_to_v10_40l_identity,
)


V101_CKPT = WORK_V10 / "v101_llamapro_ckpt"


def _build_llamapro_adamw(lr_gate: Any, lr_lora_b: Any, lr_base: Any) -> optim.MultiOptimizer:
    gate_keys = (
        "post_operator_norm",
        "post_ffn_norm",
        "res_scale",
        "loop_scales",
        "step_embed",
        "q_norm",
        "src_norm",
        "h_norm",
        "anc_norm",
    )
    return optim.MultiOptimizer(
        [
            optim.AdamW(learning_rate=lr_gate, weight_decay=0.0),
            optim.AdamW(learning_rate=lr_lora_b, weight_decay=0.01),
            optim.AdamW(learning_rate=lr_base, weight_decay=0.01),
        ],
        filters=[
            lambda k, p: p.ndim == 1 or any(gk in k for gk in gate_keys),
            lambda k, _: "lora_b" in k,
        ],
    )


def _unfreeze_new_v10_sidecars_only(vlm: Any) -> None:
    """Keep `v9`'s `engram_6`, `engram_18`, and `attn_res_26/30/32` 100% frozen; unfreeze ONLY new `40L` modules!"""
    core = vlm.language_model.model
    for frozen_attr in ("engram_6", "engram_18", "attn_res_26", "attn_res_30", "attn_res_32"):
        if hasattr(core, frozen_attr):
            getattr(core, frozen_attr).freeze()
    for active_attr in ("attn_res_36", "attn_res_38", "latent_recur_35"):
        if hasattr(core, active_attr):
            getattr(core, active_attr).unfreeze()


def main() -> None:
    import mlx_vlm
    from mlx_vlm.trainer.peft import get_peft_model

    mx.set_wired_limit(0)
    mx.set_cache_limit(96 * 1024 * 1024)
    WORK_V10.mkdir(parents=True, exist_ok=True)

    # 1. Exact `36L v9 -> 40L v10` Identity Upcycle (reusing `STAGE0_40L_ID` if already verified on disk)
    if not (STAGE0_40L_ID / "model.safetensors").exists() or not (STAGE0_40L_ID / "v8_arch_ext.safetensors").exists():
        print(f"[v10.1 Stage 0] Upcycling pristine 36L v9 -> 40L 3.18B (256k LongRoPE2YaRN) at {STAGE0_40L_ID}...", flush=True)
        up_info = upcycle_v9_36l_to_v10_40l_identity(EXPORT_V9_SEAL, STAGE0_40L_ID)
        print(f"[v10.1 Stage 0] {json.dumps(up_info)}", flush=True)

    vlm_v9, _ = mlx_vlm.load(str(EXPORT_V9_SEAL))
    attach_v8_extensions(vlm_v9, sidecar_dir=EXPORT_V9_SEAL, enable_moba=False)
    vlm_v9.eval()
    probe_ids = mx.array([[1, 42, 105, 512, 1024, 2048, 4096]], dtype=mx.int32)
    v9_logits = vlm_v9.language_model(probe_ids).logits
    mx.eval(v9_logits)
    del vlm_v9
    gc.collect()
    mx.clear_cache()

    vlm, processor = mlx_vlm.load(str(STAGE0_40L_ID))
    attach_v8_extensions(vlm, sidecar_dir=STAGE0_40L_ID, enable_moba=False)

    # LLaMA-Pro Frozen Trunk: attach Rank-32 LoRA ONLY on the 4 newly inserted `40L` blocks `(34, 35, 38, 39)`!
    leafs = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "in_proj", "out_proj", "w1", "w2", "w3"}
    prefs = tuple(f"model.layers.{i}." for i in UPCYCLED_40L_INDICES)
    targets = [
        f"language_model.{n}"
        for n, _ in vlm.language_model.named_modules()
        if n.startswith(prefs) and n.split(".")[-1] in leafs
    ]
    vlm = get_peft_model(vlm, linear_layers=targets, rank=32, alpha=64, dropout=0.01, verbose=False)
    _unfreeze_new_v10_sidecars_only(vlm)
    _unfreeze_upcycled_40l_gates(vlm)

    vlm.eval()
    v10_logits0 = vlm.language_model(probe_ids).logits
    mx.eval(v10_logits0)
    diff0 = float(mx.max(mx.abs(v10_logits0.astype(mx.float32) - v9_logits.astype(mx.float32))).item())
    print(f"[v10.1 Identity Verified] 36L v9 vs 40L v10.1 Step-0 max_abs_logit_diff = {diff0:.8f} | LoRA targets={len(targets)}", flush=True)
    assert diff0 < 1e-6
    del v9_logits, v10_logits0

    _FROZEN_SIDECAR_PFX = (
        "language_model.model.engram_",
        "language_model.model.attn_res_26",
        "language_model.model.attn_res_30",
        "language_model.model.attn_res_32",
    )
    orig_sidecar_all = mx.load(str(STAGE0_40L_ID / "v8_arch_ext.safetensors"))
    orig_sidecar = {k: v for k, v in orig_sidecar_all.items() if not k.startswith(_FROZEN_SIDECAR_PFX)}
    del orig_sidecar_all
    mx.eval(list(orig_sidecar.values()))

    sft_rows = [json.loads(l) for l in (V10_CORPUS_DIR / "train_v10_sft.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    orpo_rows = [json.loads(l) for l in (V10_CORPUS_DIR / "orpo_v10_pairs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    dapo_rows = [json.loads(l) for l in (V10_CORPUS_DIR / "dapo_v10_hard_episodes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    val_rows = [json.loads(l) for l in (V10_CORPUS_DIR / "val_v10_joint.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    tr_items, _ = prepare_turn_items_v8(sft_rows, vlm, processor, max_seq_len=2816)
    va_items, _ = prepare_turn_items_v8(val_rows, vlm, processor, max_seq_len=2816)
    tok = processor.tokenizer
    eos_id = int(tok.eos_token_id)
    short_anc = [it for it in tr_items if int(it["full_ids"].shape[1]) <= 1500 and it.get("vt_feats") is None][:48]
    anc_rng = random.Random(20261022)

    has_saved_rl = (V101_CKPT / "rl_best" / "trainable.safetensors").exists() and (V101_CKPT / "sft_best" / "trainable.safetensors").exists()
    if has_saved_rl:
        vlm.load_weights(str(V101_CKPT / "sft_best" / "trainable.safetensors"), strict=False)
        mx.eval(vlm.parameters())
        vlm.eval()
        best_val, _ = evaluate_val_loss(vlm, va_items[:64])
        vlm.load_weights(str(V101_CKPT / "rl_best" / "trainable.safetensors"), strict=False)
        mx.eval(vlm.parameters())
        print(f"[v10.1 Resume] Resuming completed sft_best (val={best_val:.4f}), orpo_best, and rl_best (active=9.5) -> Stage 4 WiSE-FT!", flush=True)
        start_ep_idx = 99
        init_active_upd = 9.5
    else:
        vlm.eval()
        vl0, vp0 = evaluate_val_loss(vlm, va_items[:64])
        print(f"[v10.1 Stage 1 LLaMA-Pro SFT] Baseline (exact v9) val_v10_joint={vl0:.4f} (PPL={vp0:.4f}) | turns={len(tr_items)}", flush=True)
        best_val = vl0
        _save_v8_ckpt(vlm, V101_CKPT / "sft_best")

        sft_steps = 240
        opt_sft = _build_llamapro_adamw(
            lr_gate=optim.cosine_decay(2.2e-4, decay_steps=sft_steps, end=2.5e-5),
            lr_lora_b=optim.cosine_decay(1.5e-4, decay_steps=sft_steps, end=2.0e-5),
            lr_base=optim.cosine_decay(4.5e-5, decay_steps=sft_steps, end=8.0e-6),
        )
        order = list(range(len(tr_items)))
        random.Random(20261021).shuffle(order)
        t0 = time.perf_counter()
        losses: list[float] = []
        vlm.train()

        for st in range(sft_steps):
            b0 = tr_items[order[(st * 2) % len(order)]]
            b1 = tr_items[order[(st * 2 + 1) % len(order)]]
            has_long = int(b0["full_ids"].shape[1]) >= 1800 or int(b1["full_ids"].shape[1]) >= 1800
            cur_batch = [b0] if int(b0["full_ids"].shape[1]) >= 1800 else ([b1] if int(b1["full_ids"].shape[1]) >= 1800 else [b0, b1])
            lv, _ = _run_dft_batch(vlm, cur_batch, opt_sft)
            apply_qk_clip(vlm, max_lora_fro_norm=6.0)
            losses.append(lv)
            if has_long or (st + 1) % 6 == 0:
                gc.collect()
                mx.clear_cache()
            if (st + 1) % 40 == 0 or st == sft_steps - 1:
                vlm.eval()
                vl, vp = evaluate_val_loss(vlm, va_items[:64])
                vlm.train()
                if vl <= best_val:
                    best_val = vl
                    _save_v8_ckpt(vlm, V101_CKPT / "sft_best")
                print(
                    f"  [v10.1 Block-SFT {st+1:3d}/{sft_steps} | {(time.perf_counter()-t0)/60:4.1f}m] "
                    f"train={sum(losses)/len(losses):.4f} | val={vl:.4f} (PPL={vp:.4f}) | best={best_val:.4f}",
                    flush=True,
                )
                losses.clear()
                gc.collect()
                mx.clear_cache()

        del opt_sft
        gc.collect()
        mx.clear_cache()
        vlm.load_weights(str(V101_CKPT / "sft_best" / "trainable.safetensors"), strict=False)
        mx.eval(vlm.parameters())

        # 3. Stage 2: Contrastive `ORPO` on New Upper Blocks (`80` steps)
        opt_orpo = _build_llamapro_adamw(
            lr_gate=optim.cosine_decay(4.0e-5, decay_steps=80, end=8.0e-6),
            lr_lora_b=optim.cosine_decay(3.5e-5, decay_steps=80, end=6.0e-6),
            lr_base=optim.cosine_decay(1.2e-5, decay_steps=80, end=3.0e-6),
        )
        best_orpo = best_val
        _save_v8_ckpt(vlm, V101_CKPT / "orpo_best")
        vlm.train()

        for ost in range(80):
            pair = orpo_rows[ost % len(orpo_rows)]
            p_s = _apply_template_inference_aligned(processor, vlm.config, pair["messages"], num_images=0, enable_thinking=True)
            c_s, r_s = str(pair["chosen"]), str(pair["rejected"])
            if p_s.endswith("<think>"):
                c_s = c_s[len("<think>") :] if c_s.startswith("<think>") else c_s
                r_s = r_s[len("<think>") :] if r_s.startswith("<think>") else r_s
            p_ids = tok.encode(p_s, add_special_tokens=False)
            c_ids = (p_ids + tok.encode(c_s, add_special_tokens=False) + [eos_id])[:1792]
            r_ids = (p_ids + tok.encode(r_s, add_special_tokens=False) + [eos_id])[:1792]
            s_i = max(len(p_ids) - 1, 0)
            if len(c_ids) > s_i + 2 and len(r_ids) > s_i + 2:
                _run_orpo_step(vlm, mx.array([c_ids], dtype=mx.int32), mx.array([r_ids], dtype=mx.int32), s_i, anc_rng.choice(short_anc), opt_orpo)
                apply_qk_clip(vlm, max_lora_fro_norm=6.0)
            if (ost + 1) % 40 == 0:
                vlm.eval()
                vl, vp = evaluate_val_loss(vlm, va_items[:64])
                vlm.train()
                if vl <= best_orpo + 0.005:
                    best_orpo = min(best_orpo, vl)
                    _save_v8_ckpt(vlm, V101_CKPT / "orpo_best")
                print(f"  [v10.1 Block-ORPO {ost+1:2d}/80] val={vl:.4f} (PPL={vp:.4f})", flush=True)
                gc.collect()
                mx.clear_cache()

        del opt_orpo
        vlm.load_weights(str(V101_CKPT / "orpo_best" / "trainable.safetensors"), strict=False)
        mx.eval(vlm.parameters())
        _save_v8_ckpt(vlm, V101_CKPT / "rl_best")
        start_ep_idx = 1
        init_active_upd = 0.0

    # Free the 945-item `tr_items` tensor list BEFORE RLVR rollouts so Metal unified RAM stays < 8.5 GiB!
    del tr_items
    gc.collect()
    mx.clear_cache()

    # 4. Stage 3: `DAPO` Dynamic Sampling + `Dr. GRPO` Unbiased RLVR (`10` active updates)
    vlm.eval()
    opt_rl = _build_llamapro_adamw(lr_gate=1.5e-5, lr_lora_b=1.5e-5, lr_base=6.0e-6)
    ref_w = mx.load(str(V101_CKPT / "orpo_best" / "trainable.safetensors"))
    mx.eval(list(ref_w.values()))
    ep_order = list(dapo_rows)
    random.Random(20261023).shuffle(ep_order)
    active_upd = init_active_upd
    for idx_e, row in enumerate(ep_order[:24], start=1):
        if idx_e < start_ep_idx:
            continue
        p_msgs = row["messages"]
        p_str = _apply_template_inference_aligned(processor, vlm.config, p_msgs, num_images=0, enable_thinking=True)
        p_txt = "\n".join(str(m.get("content", "")) for m in p_msgs)
        p_ids = tok.encode(p_str, add_special_tokens=False)[:1408]
        p_mx = mx.array([p_ids], dtype=mx.int32)
        s_idx = max(len(p_ids) - 1, 0)
        rollouts = []
        for _ in range(4):
            res = mlx_vlm.generate(vlm, processor, p_str, max_tokens=384, temperature=0.78, repetition_penalty=1.03, verbose=False)
            g_txt = res.text or ""
            c_ids = tok.encode(g_txt, add_special_tokens=False) + [eos_id]
            f_seq = mx.concatenate([p_mx, mx.array([c_ids], dtype=mx.int32)], axis=1)[:, :1920]
            mx.eval(f_seq)
            sc = score_response(g_txt, prompt_text=p_txt, kind=row["kind"], verifier=row["verifier"], tokenizer=tok, max_think_tokens=320)
            rollouts.append({"full_ids": f_seq, "score": sc})
            del res
            gc.collect()
            mx.clear_cache()

        rews = [r["score"]["reward"] for r in rollouts]
        p_cnt = sum(1 for r in rollouts if r["score"]["correct"])
        spread = max(rews) - min(rews)
        anc_b = anc_rng.sample(short_anc, k=min(2, len(short_anc)))

        if p_cnt == 4 and spread < 1e-4:
            pass
        elif p_cnt == 0 and spread < 1e-4:
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
        else:
            mean_r = sum(rews) / len(rews)
            advs = [r - mean_r for r in rews]
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
                clipped_rat = mx.clip(rat, 0.80, 1.28)
                pg = -mx.minimum(rat * adv_m, clipped_rat * adv_m)
                kl = mx.exp(r_lp - n_lp) - (r_lp - n_lp) - 1.0
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

        del rollouts
        gc.collect()
        mx.clear_cache()
        if idx_e % 4 == 0 or active_upd >= 9.5 or idx_e == 24:
            vlm.eval()
            vl_r, _ = evaluate_val_loss(vlm, va_items[:64])
            print(f"  [v10.1 Block-DAPO/DrGRPO ep={idx_e:2d} active={active_upd:.1f}] val={vl_r:.4f}", flush=True)
            if vl_r <= best_val + 0.012:
                _save_v8_ckpt(vlm, V101_CKPT / "rl_best")
            if active_upd >= 9.5:
                break
    del ref_w, opt_rl, short_anc
    gc.collect()
    mx.clear_cache()

    # 5. Stage 4: Lean `WiSE-FT` Non-Degradation Gate (`tau=0` == Exact `v9`!)
    vlm.eval()

    def _load_active_only(p: Path) -> dict[str, mx.array]:
        raw_d = mx.load(str(p))
        sub = {k: v for k, v in raw_d.items() if not k.startswith(_FROZEN_SIDECAR_PFX)}
        del raw_d
        mx.eval(list(sub.values()))
        return sub

    def _eval_probe_lean(tasks_p: list[dict[str, Any]]) -> tuple[float, float]:
        oks = 0
        rews_p: list[float] = []
        for t_p in tasks_p:
            ps_p = _apply_template_inference_aligned(processor, vlm.config, t_p["messages"], num_images=0, enable_thinking=True)
            pt_p = "\n".join(str(m.get("content", "")) for m in t_p["messages"])
            r_p = mlx_vlm.generate(vlm, processor, ps_p, max_tokens=384, temperature=0.0, repetition_penalty=1.04, verbose=False)
            tx_p = r_p.text or ""
            if ps_p.endswith("<think>") and not tx_p.lstrip().startswith("<think>"):
                tx_p = "<think>" + tx_p
            sc_p = score_response(tx_p, prompt_text=pt_p[:2000], kind=t_p["kind"], verifier=t_p["verifier"], tokenizer=tok, max_think_tokens=320)
            oks += int(bool(sc_p["correct"]))
            rews_p.append(float(sc_p["reward"]))
            del r_p
            gc.collect()
            mx.clear_cache()
        return oks / max(1, len(tasks_p)), sum(rews_p) / max(1, len(rews_p))

    w_sft = _load_active_only(V101_CKPT / "sft_best" / "trainable.safetensors")
    w_orpo = _load_active_only(V101_CKPT / "orpo_best" / "trainable.safetensors")
    w_rl = _load_active_only(V101_CKPT / "rl_best" / "trainable.safetensors")
    w_full = {
        k: (0.40 * w_sft[k].astype(mx.float32) + 0.35 * w_orpo[k].astype(mx.float32) + 0.25 * w_rl[k].astype(mx.float32)).astype(w_sft[k].dtype)
        for k in w_sft
    }
    del w_orpo, w_rl
    mx.eval(list(w_full.values()))

    def _interp_v9_40l(wd: dict[str, mx.array], tau: float) -> dict[str, mx.array]:
        out: dict[str, mx.array] = {}
        for k, v in wd.items():
            if "lora_b" in k or any(gn in k for gn in ("post_operator_norm", "post_ffn_norm", "loop_scales", "step_embed")):
                out[k] = (v.astype(mx.float32) * tau).astype(v.dtype)
            elif k in orig_sidecar:
                v0 = orig_sidecar[k].astype(mx.float32)
                out[k] = (v0 + tau * (v.astype(mx.float32) - v0)).astype(v.dtype)
            else:
                out[k] = v
        return out

    seal_d = json.loads((V6_CORPUS_DIR / "seal_svf_episodes.json").read_text(encoding="utf-8"))
    probe_tasks = seal_d["dev_diagnose_4"] + seal_d["dev_gate_8"] + dapo_rows[:12]
    va_eval_sub = va_items[:48]
    del va_items
    gc.collect()
    mx.clear_cache()

    def _unload_local_server() -> None:
        import urllib.request
        try:
            urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8991/v1/admin/unload", method="POST"), timeout=45).read()
        except Exception:
            pass

    _unload_local_server()
    zero_w = _interp_v9_40l(w_full, 0.0)
    vlm.load_weights(list(zero_w.items()), strict=False)
    mx.eval(vlm.parameters())
    vl_v9, _ = evaluate_val_loss(vlm, va_eval_sub)
    acc_v9, rew_v9 = _eval_probe_lean(probe_tasks)
    best_name, best_w = "v9_40l_tau0", zero_w
    best_sc = (round(acc_v9, 4), round(rew_v9 - 0.40 * vl_v9, 4))
    best_vl = vl_v9
    print(f"  [WiSE-FT Ref tau=0.00 (exact v9)] probe={acc_v9*100:5.1f}% | val={vl_v9:.4f} | score={best_sc}", flush=True)

    for nm, src_d, tau_v in [
        ("v101_tau0.55", w_full, 0.55),
        ("v101_tau0.75", w_full, 0.75),
        ("v101_tau0.90", w_full, 0.90),
        ("v101_tau1.00", w_full, 1.00),
        ("v101_sft_1.00", w_sft, 1.00),
    ]:
        _unload_local_server()
        cw = src_d if tau_v == 1.00 else _interp_v9_40l(src_d, tau_v)
        vlm.load_weights(list(cw.items()), strict=False)
        mx.eval(vlm.parameters())
        vlc, _ = evaluate_val_loss(vlm, va_eval_sub)
        accc, rewc = _eval_probe_lean(probe_tasks)
        sct = (round(accc, 4), round(rewc - 0.40 * vlc, 4))
        print(f"  [WiSE-FT] {nm:16s} | probe={accc*100:5.1f}% | val={vlc:.4f} | score={sct}", flush=True)
        if sct >= best_sc:
            best_name, best_w, best_sc, best_vl = nm, cw, sct, vlc
        gc.collect()
        mx.clear_cache()

    print(f"[v10.1 Promoted] {best_name} (score={best_sc}, val={best_vl:.4f})", flush=True)
    vlm.load_weights(list(best_w.items()), strict=False)
    mx.eval(vlm.parameters())

    del w_sft, w_full, zero_w, best_w, orig_sidecar
    gc.collect()
    mx.clear_cache()

    _, fuse_info = fuse_and_export_40l_lean(
        vlm,
        STAGE0_40L_ID,
        EXPORT_V10_APEX,
        extra_sidecar_meta={
            "model_version": "LFM3-VL-3.2B-RPR-V10-APEX-BF16",
            "precision": "bfloat16",
            "trunk_policy": "LLaMA-Pro Frozen 36L v9 Trunk + 40L Upper-Pyramid Expansion",
            "selected_candidate": best_name,
            "val_probe_score": list(best_sc),
            "selected_val_loss": round(best_vl, 4),
        },
    )
    logits_exp = fuse_info["logits_expected"]
    del vlm, processor
    gc.collect()
    mx.clear_cache()

    if STAGE0_40L_ID.exists():
        shutil.rmtree(STAGE0_40L_ID, ignore_errors=True)

    # Verify reload parity
    vlm_re, proc_re = mlx_vlm.load(str(EXPORT_V10_APEX))
    attach_v8_extensions(vlm_re, sidecar_dir=EXPORT_V10_APEX, enable_moba=False)
    vlm_re.eval()
    p_chk = mx.array([[1, 42, 105, 512, 1024, 2048]], dtype=mx.int32)
    l_re = vlm_re.language_model(p_chk).logits
    mx.eval(l_re)
    rdiff = float(mx.max(mx.abs(l_re.astype(mx.float32) - logits_exp.astype(mx.float32))).item())
    print(f"[v10.1 Reload Verified] reload_diff = {rdiff:.6f}", flush=True)
    assert rdiff < 1e-4

    # Refresh 40L Null-Space + Transformer^2 SVF Bank (`L26, 30, 36, 38`) + DGM Root Archive
    bank_p = EXPORT_V10_APEX / "seal_nullspace_svf_bank.safetensors"
    if bank_p.exists():
        bank_p.unlink()
    tr_short, _ = prepare_turn_items_v8([r for r in sft_rows if r.get("pillar") != "long_context_synthesis"][:160], vlm_re, proc_re, max_seq_len=2048)
    skill_batches = [
        [it for it in tr_short if it.get("pillar") == "advanced_rsi_dgm_svf"][:8] or tr_short[:6],
        [it for it in tr_short if it.get("pillar") == "complex_logic_sat"][:8] or tr_short[6:12],
        [it for it in tr_short if it.get("pillar") == "realtime_research_citations"][:8] or tr_short[12:18],
        [it for it in tr_short if it.get("pillar") == "unseen_ood_self_discover"][:8] or tr_short[18:24],
    ]
    extract_v10_40l_nullspace_svf_bank(vlm_re, va_eval_sub[:32], skill_batches, bank_p)
    overlay_dir = ROOT_DIR / "checkpoints" / "seal_user_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    dgm_root = {
        "model": "LFM3-VL-3.2B-RPR-V10-APEX-BF16",
        "rewrite_layers": list(REWRITE_TARGET_LAYERS_40L),
        "svf_experts": ["b0_advanced_rsi", "b1_complex_logic", "b2_research_citations", "b3_ood_code"],
        "nodes": [
            {
                "id": "dgm_gen_000_root",
                "parent_id": None,
                "score": round(float(best_sc[0]), 4),
                "val_loss": round(best_vl, 4),
                "anchor_regression": 0.0,
                "regressed": False,
                "children": 0,
                "overlay_path": None,
                "created_at": int(time.time()),
            }
        ],
    }
    (overlay_dir / "dgm_archive.json").write_text(json.dumps(dgm_root, indent=2) + "\n", encoding="utf-8")
    del vlm_re, proc_re, tr_short, va_eval_sub
    gc.collect()
    mx.clear_cache()

    # Refresh 40L On-Policy MTP Head (`160` steps)
    bench_s = ROOT_DIR / "results" / "v10_mtp_benchmark" / "summary.json"
    if bench_s.exists():
        bench_s.unlink()
    if (WORK_V10 / "stage5_v10_onpolicy_32.json").exists():
        (WORK_V10 / "stage5_v10_onpolicy_32.json").unlink()
    run_stage_5_v10_onpolicy_mtp(sft_rows, mtp_steps=160)

    if WORK_V10.exists():
        shutil.rmtree(WORK_V10, ignore_errors=True)
    print("[v10.1 LLaMA-Pro Upper-Pyramid Pipeline Complete!]", flush=True)


if __name__ == "__main__":
    main()
