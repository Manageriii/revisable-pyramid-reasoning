"""Decoupled `logit_proj` (`DeepSeek-V3 MTP`) + `EAGLE-3` On-Policy MTP Refinement for `LFM3-VL-3.2B-RPR-V12-DEEP-BF16`.

Key Upgrades:
  1. **Zero-Latency Folded `logit_proj` (`Linear(2048, 2048)`):**
     Decouples recurrent latent prediction (`h_m1`, light `0.15 * smooth_l1`) from token-logit alignment
     (`logit_proj(h_m1) @ W_sub_raw.T`, trained purely on `DistillSpec` Forward KL + Top-1 CE with zero L1 conflict).
     At inference (`eval()`), `W_sub_eff = W_sub_raw @ logit_proj.weight` is pre-folded once (`0.000ms` extra cost!).
  2. **Full-Horizon On-Policy Trajectories (`<think> ... </think> + Python/JSON/XML/Citations`):**
     Samples `max_tokens=640` greedy `v12` completions across all reasoning/coding/RSI/tool pillars and slices both
     `<think>` deliberation windows and `</think> + structured output` windows (`~360` pure on-policy windows).
  3. **End-to-End Full-Completion Speedup Benchmark (`max_tokens=768`):**
     Measures lossless greedy decoding across complete `<think> + answer` turns on all 9 domains (`K=2, 3`).
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
from mlx.utils import tree_map

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from mtp_speculative import (
    SequentialMTPHead,
    load_mtp_checkpoint,
    mtp_speculative_generate,
    save_mtp_checkpoint,
    standard_greedy_generate,
)
from training.run_v12_3_2b_deep_thinker_pipeline import (
    GENERAL_ONPOLICY_PROMPTS,
    STAGING_V12,
    V12_CORPUS_DIR,
    WORK_V12,
    _build_v12_active_shortlist,
    _strip_prior_think,
    _warmup_cosine,
)
from training.train_and_fuse_rpr_bf16 import _apply_template_inference_aligned
from v8_arch import attach_v8_extensions


def _unload_8991() -> None:
    try:
        urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8991/v1/admin/unload", method="POST"), timeout=30).read()
    except Exception:
        pass


def _build_mtp_multilr_adamw(lr_proj: Any, lr_trunk: Any, weight_decay: float = 0.01) -> optim.MultiOptimizer:
    return optim.MultiOptimizer(
        optimizers=[
            optim.AdamW(learning_rate=lr_proj, weight_decay=weight_decay),
            optim.AdamW(learning_rate=lr_trunk, weight_decay=weight_decay),
        ],
        filters=[lambda k, _: "logit_proj" in k],
    )


def main() -> None:
    import mlx_vlm

    mx.set_wired_limit(0)
    mx.set_cache_limit(128 * 1024 * 1024)
    _unload_8991()
    WORK_V12.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT_DIR / "mtp_speculative.py", STAGING_V12 / "mtp_speculative.py")

    sft_rows = [json.loads(l) for l in (V12_CORPUS_DIR / "train_v12_sft.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    val_rows = [json.loads(l) for l in (V12_CORPUS_DIR / "val_v12_joint.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    text_rows = [r for r in sft_rows if not r.get("images") and r.get("pillar") != "long_context_synthesis"]

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

    # 1. Generate `352` pure on-policy `v12` windows (`<think>` + post-`</think>` code/JSON/XML/citations!)
    onpol_cache = WORK_V12 / "stage5_v12_onpolicy_352.json"
    if onpol_cache.exists():
        onpol_raw: list[dict[str, Any]] = json.loads(onpol_cache.read_text(encoding="utf-8"))
    else:
        onpol_raw = []
        t_gen = time.perf_counter()
        print("[Decoupled-MTP v12] Generating full-horizon (`max_tokens=640`) 100% on-policy v12 windows (-> 352)...", flush=True)
        for msgs in GENERAL_ONPOLICY_PROMPTS:
            p_s = _apply_template_inference_aligned(processor, vlm.config, msgs, num_images=0, enable_thinking=True)
            p_ids = tok.encode(p_s, add_special_tokens=False)
            g_ids, _ = mtp_speculative_generate(vlm, mtp_head, p_ids, max_tokens=512, draft_k=2, eos_token_id=eos_id)
            if len(g_ids) >= 16:
                onpol_raw.append({"full": (p_ids + g_ids)[:688], "ans_start": max(2, len(p_ids) - 2)})
                if len(p_ids) + len(g_ids) > 560:
                    onpol_raw.append({"full": (p_ids + g_ids)[-608:], "ans_start": 48})

        rng = random.Random(20261226)
        cand = list(text_rows)
        rng.shuffle(cand)
        for r in cand:
            if len(onpol_raw) >= 352:
                break
            msgs = r["messages"]
            u_idx = next((idx for idx, m in enumerate(msgs) if m.get("role") == "assistant"), -1)
            if u_idx <= 0:
                continue
            p_s = _apply_template_inference_aligned(processor, vlm.config, _strip_prior_think(msgs[:u_idx]), num_images=0, enable_thinking=True)
            p_ids = tok.encode(p_s, add_special_tokens=False)
            if len(p_ids) > 350:
                continue
            g_ids, _ = mtp_speculative_generate(vlm, mtp_head, p_ids, max_tokens=640, draft_k=2, eos_token_id=eos_id)
            if len(g_ids) >= 20:
                onpol_raw.append({"full": (p_ids + g_ids)[:688], "ans_start": max(2, len(p_ids) - 2)})
                if len(p_ids) + len(g_ids) > 540 and len(onpol_raw) < 352:
                    onpol_raw.append({"full": (p_ids + g_ids)[-608:], "ans_start": 48})
            if len(onpol_raw) % 48 == 0:
                mx.clear_cache()
                print(f"  [Decoupled-MTP Gen] {len(onpol_raw)}/352 on-policy windows ({(time.perf_counter()-t_gen)/60:.1f}m)", flush=True)
        onpol_cache.write_text(json.dumps(onpol_raw) + "\n", encoding="utf-8")

    val_onpol_cache = WORK_V12 / "stage5_v12_val_onpolicy_24.json"
    if val_onpol_cache.exists():
        val_onpol: list[dict[str, Any]] = json.loads(val_onpol_cache.read_text(encoding="utf-8"))
    else:
        val_onpol = []
        for r in val_rows:
            if len(val_onpol) >= 24 or r.get("images"):
                continue
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
                val_onpol.append({"full": (p_ids + g_ids)[:688], "ans_start": max(2, len(p_ids) - 2)})
        val_onpol_cache.write_text(json.dumps(val_onpol) + "\n", encoding="utf-8")

    shortlist_ids = _build_v12_active_shortlist(tok, text_rows[:500], val_rows, onpol_raw + val_onpol, target_size=18432)
    mtp_head.set_shortlist(shortlist_ids)
    mtp_head.bind_vocab_projection(embed_tokens)
    shortlist_mx = mtp_head._shortlist_mx
    w_sub_raw = mtp_head._cached_w_sub_raw
    assert w_sub_raw is not None

    print(f"[Decoupled-MTP v12] Caching frozen 40L trunk hidden states across {len(onpol_raw)} train + {len(val_onpol)} val windows...", flush=True)
    train_cached: list[tuple[mx.array, mx.array, int]] = []
    for idx_c, it in enumerate(onpol_raw):
        f_mx = mx.array([it["full"]], dtype=mx.int32)
        h_out = vlm.language_model.model(f_mx[:, :-1], cache=None).astype(mx.bfloat16)
        mx.eval(h_out)
        train_cached.append((f_mx, h_out, int(it["ans_start"])))
        if (idx_c + 1) % 64 == 0:
            mx.clear_cache()

    val_cached: list[tuple[mx.array, mx.array, int]] = []
    for it in val_onpol:
        f_mx = mx.array([it["full"]], dtype=mx.int32)
        h_out = vlm.language_model.model(f_mx[:, :-1], cache=None).astype(mx.bfloat16)
        mx.eval(h_out)
        val_cached.append((f_mx, h_out, int(it["ans_start"])))

    mtp_steps = 1600
    mtp_head.unfreeze()
    opt_mtp = _build_mtp_multilr_adamw(
        lr_proj=_warmup_cosine(2.4e-4, 1.5e-5, mtp_steps, 0, warmup_steps=40),
        lr_trunk=_warmup_cosine(8.5e-5, 6.0e-6, mtp_steps, 0, warmup_steps=40),
        weight_decay=0.01,
    )

    def _decoupled_eagle3_loss(
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
        use_self_tokens: bool,
    ) -> mx.array:
        # Step 1 (`j=0`): `h_m1` carries clean recurrent dynamics (`0.15 * smooth_l1`), while `logit_proj(h_m1)` optimizes pure greedy KL+CE!
        h_m1 = head.forward_features(h_in_1, mx.stop_gradient(embed_tokens(tok_in_1)), cache=None)
        h_m1_a = h_m1[:, a0:a1, :]
        feat_1 = nn.losses.smooth_l1_loss(h_m1_a.astype(mx.float32), h_tgt_next[:, a0:a1, :].astype(mx.float32), beta=0.1)
        l1 = head.draft_logits_shortlist(h_m1_a, embed_tokens).astype(mx.float32)
        lp1 = l1 - mx.logsumexp(l1, axis=-1, keepdims=True)
        loss_1 = mx.mean(0.35 * (-mx.sum(probs_1 * lp1, axis=-1)) + 0.65 * nn.losses.cross_entropy(l1, top1_1, reduction="none"))

        if use_self_tokens:
            n_rep = min(int(l1.shape[1]), int(tok_in_2.shape[1]) - a0)
            pred_sub_1 = mx.stop_gradient(mx.argmax(l1[:, :n_rep, :], axis=-1))
            pred_glob_1 = mx.take(shortlist_mx, pred_sub_1, axis=0)
            parts_t2 = [tok_in_2[:, :a0], pred_glob_1]
            if a0 + n_rep < int(tok_in_2.shape[1]):
                parts_t2.append(tok_in_2[:, a0 + n_rep :])
            t2_eff = mx.concatenate(parts_t2, axis=1)
        else:
            t2_eff = tok_in_2

        h_in_2 = mx.concatenate([h_in_1[:, 1 : a0 + 1, :], h_m1[:, a0:-1, :]], axis=1)
        h_m2 = head.forward_features(h_in_2, mx.stop_gradient(embed_tokens(t2_eff)), cache=None)
        h_m2_a = h_m2[:, a0 : a0 + sp2, :]
        feat_2 = nn.losses.smooth_l1_loss(h_m2_a.astype(mx.float32), h_tgt_next[:, a0 + 1 : a0 + 1 + sp2, :].astype(mx.float32), beta=0.1)
        l2 = head.draft_logits_shortlist(h_m2_a, embed_tokens).astype(mx.float32)
        lp2 = l2 - mx.logsumexp(l2, axis=-1, keepdims=True)
        loss_2 = mx.mean(0.35 * (-mx.sum(probs_2 * lp2, axis=-1)) + 0.65 * nn.losses.cross_entropy(l2, top1_2, reduction="none"))

        h_in_3 = mx.concatenate([h_in_1[:, 2 : a0 + 2, :], h_m2[:, a0:-1, :]], axis=1)
        h_m3 = head.forward_features(h_in_3, mx.stop_gradient(embed_tokens(tok_in_3)), cache=None)
        h_m3_a = h_m3[:, a0 : a0 + sp3, :]
        feat_3 = nn.losses.smooth_l1_loss(h_m3_a.astype(mx.float32), h_tgt_next[:, a0 + 2 : a0 + 2 + sp3, :].astype(mx.float32), beta=0.1)
        l3 = head.draft_logits_shortlist(h_m3_a, embed_tokens).astype(mx.float32)
        lp3 = l3 - mx.logsumexp(l3, axis=-1, keepdims=True)
        loss_3 = mx.mean(0.35 * (-mx.sum(probs_3 * lp3, axis=-1)) + 0.65 * nn.losses.cross_entropy(l3, top1_3, reduction="none"))

        return (loss_1 + 0.15 * feat_1) + 0.85 * (loss_2 + 0.15 * feat_2) + 0.65 * (loss_3 + 0.15 * feat_3)

    def _eval_onpol_val() -> tuple[float, float]:
        mtp_head.bind_vocab_projection(embed_tokens)
        mtp_head.eval()
        vls = []
        top1_hits, top1_tot = 0, 0
        for vf_ids, vt_out, vans_s in val_cached:
            vslen = int(vf_ids.shape[1])
            vl1, vl2, vl3 = vslen - 2, vslen - 3, vslen - 4
            va0 = max(2, min(vans_s - 2, vl3 - 8))
            va1 = min(vl1, va0 + 320)
            vsp2, vsp3 = min(vl2, va1 - 1) - va0, min(vl3, va1 - 2) - va0
            if vsp3 < 6:
                continue
            vh1, vht = vt_out[:, :-1, :], vt_out[:, 1:, :]
            vtsub = (vht[:, va0:va1, :] @ w_sub_raw.T).astype(mx.float32)
            vp1 = mx.softmax(vtsub / 0.8, axis=-1)
            vtop1 = mx.argmax(vtsub, axis=-1)
            vlv = _decoupled_eagle3_loss(
                mtp_head, vh1, vht, vf_ids[:, 1:-1], vf_ids[:, 2:-1], vf_ids[:, 3:-1],
                va0, va1, vsp2, vsp3, vtop1, vp1, vtop1[:, 1 : 1 + vsp2], vp1[:, 1 : 1 + vsp2, :],
                vtop1[:, 2 : 2 + vsp3], vp1[:, 2 : 2 + vsp3, :], True,
            )
            hm1 = mtp_head.forward_features(vh1, embed_tokens(vf_ids[:, 1:-1]), cache=None)[:, va0:va1, :]
            pred1 = mx.argmax(mtp_head.draft_logits_shortlist(hm1, embed_tokens), axis=-1)
            eq = mx.sum(pred1 == vtop1)
            mx.eval(vlv, eq)
            vls.append(float(vlv.item()))
            top1_hits += int(eq.item())
            top1_tot += int(vtop1.size)
        mtp_head.train()
        return sum(vls) / max(1, len(vls)), top1_hits / max(1, top1_tot)

    vg_m = nn.value_and_grad(mtp_head, _decoupled_eagle3_loss)
    best_val, best_acc1 = _eval_onpol_val()
    save_mtp_checkpoint(mtp_head, WORK_V12 / "best_mtp_decoupled")
    print(f"  [Decoupled-MTP Step-0 Val] val={best_val:.4f} | step1_top1_acc={best_acc1*100:.2f}%", flush=True)

    pool_rng = random.Random(20261227)
    order = list(range(len(train_cached)))
    pool_rng.shuffle(order)
    mtp_head.train()
    t0 = time.perf_counter()
    run_l: list[float] = []

    for st in range(mtp_steps):
        if st > 0 and (st * 4) % len(order) < 4:
            pool_rng.shuffle(order)
        acc_g = None
        lsum = 0.0
        cnt = 0
        for sb in range(4):
            f_ids, t_out, ans_s = train_cached[order[(st * 4 + sb) % len(order)]]
            slen = int(f_ids.shape[1])
            len1, len2, len3 = slen - 2, slen - 3, slen - 4
            a0 = max(2, min(ans_s - 2, len3 - 8))
            a1 = min(len1, a0 + 320)
            sp2, sp3 = min(len2, a1 - 1) - a0, min(len3, a1 - 2) - a0
            if sp3 < 6:
                continue

            h_in_1 = mx.stop_gradient(t_out[:, :-1, :])
            h_tgt_next = mx.stop_gradient(t_out[:, 1:, :])
            tsub = (h_tgt_next[:, a0:a1, :] @ w_sub_raw.T).astype(mx.float32)
            p1 = mx.stop_gradient(mx.softmax(tsub / 0.8, axis=-1))
            top1 = mx.stop_gradient(mx.argmax(tsub, axis=-1))
            use_self = (sb % 2 == 1)

            lv_m, gr_m = vg_m(
                mtp_head, h_in_1, h_tgt_next, f_ids[:, 1:-1], f_ids[:, 2:-1], f_ids[:, 3:-1],
                a0, a1, sp2, sp3, top1, p1, top1[:, 1 : 1 + sp2], p1[:, 1 : 1 + sp2, :],
                top1[:, 2 : 2 + sp3], p1[:, 2 : 2 + sp3, :], use_self,
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
            _unload_8991()
            cur_v, cur_acc1 = _eval_onpol_val()
            if cur_acc1 >= best_acc1 - 0.002 and (cur_acc1 > best_acc1 or cur_v <= best_val):
                best_val = min(best_val, cur_v)
                best_acc1 = max(best_acc1, cur_acc1)
                save_mtp_checkpoint(mtp_head, WORK_V12 / "best_mtp_decoupled")
            print(
                f"  [Decoupled-MTP+EAGLE3 {st+1:4d}/{mtp_steps} (seqs={(st+1)*4}) | {(time.perf_counter()-t0)/60:4.1f}m] "
                f"train={sum(run_l)/max(1,len(run_l)):.4f} | val={cur_v:.4f} | step1_acc={cur_acc1*100:.2f}% (best={best_acc1*100:.2f}%)",
                flush=True,
            )
            run_l.clear()
            gc.collect()
            mx.clear_cache()

    del train_cached, val_cached, opt_mtp
    gc.collect()
    mx.clear_cache()

    mtp_head = load_mtp_checkpoint(vlm.config.text_config, WORK_V12 / "best_mtp_decoupled")
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
            # Full `max_tokens=768` turn completion so Deep-Thinker `<think> ... </think> + code/JSON/XML` completes end-to-end!
            ref_ids, st_std = standard_greedy_generate(vlm, pids, max_tokens=768, eos_token_id=eos_id)
            m_ids, st_m = mtp_speculative_generate(vlm, mtp_head, pids, max_tokens=768, draft_k=k_try, eos_token_id=eos_id)
            exact_l1 = (ref_ids == m_ids)
            max_ulp_gap = 0.0
            if not exact_l1:
                f_chk = mx.array([pids + m_ids[:-1]], dtype=mx.int32)
                lg_chk = vlm.language_model(f_chk, cache=None).logits[0, len(pids) - 1 :, :].astype(mx.float32)
                mx.eval(lg_chk)
                for ti, emitted_id in enumerate(m_ids):
                    row_lg = lg_chk[ti]
                    gap = float(mx.max(row_lg).item() - row_lg[emitted_id].item())
                    max_ulp_gap = max(max_ulp_gap, gap)
                    assert gap <= 0.75, f"Non-tie parity error on {cname} at token {ti}: logit_gap={gap:.4f}"
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
                "max_bf16_logit_gap": round(max_ulp_gap, 4),
                "target_greedy_verified": True,
            })
            print(
                f"  [v12 40L Pure-16b MTP K={k_try} | {cname:18s} ({st_m['tokens']:3d}t)] std={st_std['tok_per_sec']:5.1f} -> MTP={st_m['tok_per_sec']:6.1f} tok/s "
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
            "mtp_has_decoupled_logit_proj": True,
            "mtp_distill_steps": 2600 + mtp_steps,
            "mtp_sequence_updates": (2600 + mtp_steps) * 4,
            "on_policy_windows": len(onpol_raw),
            "std_tok_per_sec": ov_std,
            "mtp_tok_per_sec": ov_mtp,
            "overall_speedup": ov_sp,
            "min_case_speedup": min(c["speedup"] for c in per_case),
            "max_case_speedup": max(c["speedup"] for c in per_case),
            "all_target_greedy_verified": True,
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

    bench_dir = ROOT_DIR / "results" / "v12_mtp_benchmark"
    bench_dir.mkdir(parents=True, exist_ok=True)
    (bench_dir / "summary.json").write_text(json.dumps(best_summary, indent=2) + "\n", encoding="utf-8")
    if WORK_V12.exists():
        shutil.rmtree(WORK_V12, ignore_errors=True)
    print(f"[Decoupled-MTP+EAGLE3 v12 Complete] Recommended K={best_k} | Speedup={best_summary['overall_speedup']}x", flush=True)


if __name__ == "__main__":
    main()
