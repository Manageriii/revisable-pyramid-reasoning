"""On-Policy SequentialMTPHead Alignment for `LFM3-VL-2.9B-RPR-V8-HERMES-BF16`:
Generates diverse on-policy greedy `v8` trajectories (general science/prose, Hermes `<tool_call>`, Python code, and RPR math)
and distills `mtp_head.safetensors` (`vlm.eval()` compiled `Engram`+`AttnRes` features) for high draft acceptance across all modes!
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
import random
import shutil
import sys
import time

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
from training.build_v8_hermes_corpus import V8_CORPUS_DIR
from training.collect_and_build_rpr_corpus import _strip_prior_think
from training.run_v8_2_9b_hermes_pipeline import EXPORT_V8_HERMES, WORK_V8
from training.train_and_fuse_rpr_bf16 import _apply_template_inference_aligned
from v8_arch import attach_v8_extensions


GENERAL_ONPOLICY_PROMPTS = [
    [{"role": "user", "content": "Could you tell me about black holes, event horizons, and Hawking radiation?"}],
    [{"role": "user", "content": "Explain how Apple Silicon unified memory bandwidth affects LLM token generation speed."}],
    [{"role": "user", "content": "Explain how B-trees and LSM-trees differ in write amplification and read latency."}],
    [{"role": "user", "content": "Explain how speculative decoding preserves exact target distribution lossless parity."}],
    [{"role": "user", "content": "Explain the difference between TCP congestion control (CUBIC vs BBR) in high-BDP networks."}],
    [{"role": "user", "content": "Write a clean Python function `toposort_kahn(adj: dict[str, list[str]]) -> list[str]` raising `ValueError` on cycles."}],
    [{"role": "user", "content": "Write a NumPy function `cosine_sim_matrix(A: np.ndarray, B: np.ndarray, eps: float = 1e-8) -> np.ndarray`."}],
    [{"role": "user", "content": "Implement `lru_cache_ttl` in Python using `OrderedDict` and `time.monotonic()`."}],
    [{"role": "system", "content": "You are Hermes.\n<tools>\n[{\"name\": \"read_file\", \"parameters\": {\"type\": \"object\", \"properties\": {\"path\": {\"type\": \"string\"}}, \"required\": [\"path\"]}}, {\"name\": \"search_files\", \"parameters\": {\"type\": \"object\", \"properties\": {\"pattern\": {\"type\": \"string\"}, \"path\": {\"type\": \"string\"}}, \"required\": [\"pattern\"]}}]\n</tools>"}, {"role": "user", "content": "Search for `SequentialMTPHead` across Python files in the current workspace."}],
    [{"role": "system", "content": "You are Hermes.\n<tools>\n[{\"name\": \"terminal\", \"parameters\": {\"type\": \"object\", \"properties\": {\"command\": {\"type\": \"string\"}}, \"required\": [\"command\"]}}]\n</tools>"}, {"role": "user", "content": "Run `pytest -q` in the current project using terminal."}],
    [{"role": "user", "content": "Summarize how General Relativity models gravitational lensing and Einstein rings."}],
    [{"role": "user", "content": "Explain how Rust ownership, borrowing, and lifetimes prevent data races at compile time."}],
]


def main() -> None:
    import mlx_vlm

    mx.set_wired_limit(0)
    mx.set_cache_limit(192 * 1024 * 1024)
    shutil.copy2(ROOT_DIR / "v8_arch.py", EXPORT_V8_HERMES / "v8_arch.py")

    vlm, processor = mlx_vlm.load(str(EXPORT_V8_HERMES))
    attach_v8_extensions(vlm, sidecar_dir=EXPORT_V8_HERMES, enable_moba=False)
    vlm.freeze()
    vlm.eval()
    tok = processor.tokenizer
    eos_id = int(tok.eos_token_id)

    mtp_head = load_mtp_checkpoint(vlm.config.text_config, EXPORT_V8_HERMES)
    embed_tokens = vlm.language_model.model.embed_tokens
    mtp_head.bind_vocab_projection(embed_tokens)

    v8_rows = [json.loads(l) for l in (V8_CORPUS_DIR / "train_v8_sft.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    text_rows = [r for r in v8_rows if not r.get("images")][:450]

    onpol_cache = WORK_V8 / "stage4_v8_onpolicy_36.json"
    WORK_V8.mkdir(parents=True, exist_ok=True)
    if onpol_cache.exists():
        onpol_raw = json.loads(onpol_cache.read_text(encoding="utf-8"))
    else:
        print("[MTP On-Policy] Generating 36 on-policy v8 greedy trajectories (chat + tools + code + math)...", flush=True)
        onpol_raw = []
        for msgs in GENERAL_ONPOLICY_PROMPTS:
            p_s = _apply_template_inference_aligned(processor, vlm.config, msgs, num_images=0, enable_thinking=True)
            p_ids = tok.encode(p_s, add_special_tokens=False)
            g_ids, _ = standard_greedy_generate(vlm, p_ids, max_tokens=220, eos_token_id=eos_id, repetition_penalty=1.05)
            if len(g_ids) >= 12:
                onpol_raw.append({"full": (p_ids + g_ids)[:640], "ans_start": max(1, len(p_ids) - 2)})
        rng_op = random.Random(20261905)
        cand = list(text_rows)
        rng_op.shuffle(cand)
        for r in cand[:24]:
            msgs = r["messages"]
            u_idx = next((idx for idx, m in enumerate(msgs) if m.get("role") == "assistant"), -1)
            if u_idx <= 0:
                continue
            p_s = _apply_template_inference_aligned(processor, vlm.config, msgs[:u_idx], num_images=0, enable_thinking=True)
            p_ids = tok.encode(p_s, add_special_tokens=False)
            if len(p_ids) > 400:
                continue
            g_ids, _ = standard_greedy_generate(vlm, p_ids, max_tokens=192, eos_token_id=eos_id, repetition_penalty=1.05)
            if len(g_ids) >= 12:
                onpol_raw.append({"full": (p_ids + g_ids)[:640], "ans_start": max(1, len(p_ids) - 2)})
        onpol_cache.write_text(json.dumps(onpol_raw) + "\n", encoding="utf-8")
        print(f"[MTP On-Policy] Cached {len(onpol_raw)} trajectories -> {onpol_cache}", flush=True)

    seqs: list[tuple[mx.array, int]] = []
    for item in onpol_raw:
        arr = mx.array([item["full"]], dtype=mx.int32)
        for _ in range(6):
            seqs.append((arr, int(item["ans_start"])))
    for r in text_rows[:240]:
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

    random.Random(20261910).shuffle(seqs)
    mtp_steps = 420
    opt_mtp = optim.AdamW(learning_rate=optim.cosine_decay(4.0e-4, decay_steps=mtp_steps, end=2.5e-5), weight_decay=0.01)

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
            lv, gr = vg_m(
                mtp_head, h_in_1, h_tgt_next, f_ids[:, 1:-1], f_ids[:, 2:-1],
                a0, a1, a1_s2, top1, p1, top1[:, 1 : 1 + sp2], p1[:, 1 : 1 + sp2, :],
            )
            mx.eval(lv)
            lsum += float(lv.item())
            cnt += 1
            acc_g = tree_map(lambda g: g * 0.5, gr) if acc_g is None else tree_map(lambda a, g: a + g * 0.5, acc_g, gr)
        if acc_g is not None and cnt > 0:
            acc_g, _ = optim.clip_grad_norm(acc_g, max_norm=1.0)
            opt_mtp.update(mtp_head, acc_g)
            mx.eval(mtp_head.parameters(), opt_mtp.state)
            run_l.append(lsum / cnt)
        if (st + 1) % 105 == 0 or st == mtp_steps - 1:
            print(f"  [MTP On-Policy {st+1:3d}/{mtp_steps} | {(time.perf_counter()-t0)/60:4.1f}m] distill_loss={sum(run_l)/max(1,len(run_l)):.4f}", flush=True)
            run_l.clear()
            mx.clear_cache()

    mtp_head.eval()
    save_mtp_checkpoint(mtp_head, EXPORT_V8_HERMES)

    # Benchmark cold-process tok/s across General Chat, Tool Call, Math, and Python Code!
    tools_demo = json.dumps([{"name": "read_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}])
    cases = [
        ("black_holes_chat", [{"role": "user", "content": "Could you tell me about black holes?"}]),
        ("hermes_tool_flash", [{"role": "system", "content": f"You are Hermes. [EFFORT: FLASH]\n<tools>\n{tools_demo}\n</tools>"}, {"role": "user", "content": "Read `engine.py` so we can inspect it."}]),
        ("math_flash", [{"role": "system", "content": "Solve accurately. [EFFORT: FLASH]"}, {"role": "user", "content": 'Compute `27 + 19 - 14` and return JSON `{"result": <int>}`.'}]),
        ("numpy_code_deep", [{"role": "system", "content": "You are an RSI engineer."}, {"role": "user", "content": "Implement `rome_rank1(W2, k_star, v_star, C_cov, reg=0.05)` in NumPy solving `u = C_reg^-1 @ k_star`."}]),
    ]
    wp = tok.encode(_apply_template_inference_aligned(processor, vlm.config, cases[0][1], num_images=0, enable_thinking=True), add_special_tokens=False)
    standard_greedy_generate(vlm, wp, max_tokens=16, eos_token_id=eos_id)
    mtp_speculative_generate(vlm, mtp_head, wp, max_tokens=16, draft_k=3, eos_token_id=eos_id)

    for cname, msgs in cases:
        ps = _apply_template_inference_aligned(processor, vlm.config, msgs, num_images=0, enable_thinking=True)
        pids = tok.encode(ps, add_special_tokens=False)
        ref_ids, st_std = standard_greedy_generate(vlm, pids, max_tokens=240, eos_token_id=eos_id)
        k3_ids, st_k3 = mtp_speculative_generate(vlm, mtp_head, pids, max_tokens=240, draft_k=3, eos_token_id=eos_id)
        print(
            f"  [MTP Bench {cname:18s}] std={st_std['tok_per_sec']:5.1f} -> MTP-K3={st_k3['tok_per_sec']:6.1f} tok/s "
            f"({st_k3['tok_per_sec']/max(1e-6,st_std['tok_per_sec']):.2f}x, acc={st_k3['acceptance_rate']*100:4.1f}%, match={ref_ids==k3_ids})",
            flush=True,
        )

    del vlm, processor, mtp_head
    gc.collect()
    mx.clear_cache()


if __name__ == "__main__":
    main()
