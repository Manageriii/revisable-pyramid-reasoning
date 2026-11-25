"""Checkpointed MLX post-training for LFM3-VL-2B-MLX-8bit (Stage 1 DFT -> Stage 2 GDPO/Dr.GRPO RLVR).

Runs strictly in this repo's `checkpoints/` directory using the base model read-only.
Never executes concurrently with other MLX jobs on Apple Silicon.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from model import DEFAULT_MODEL_PATH
from training.verifiable_rewards import score_response


DATA_DIR = Path(__file__).resolve().parent / "data"
CKPT_ROOT = ROOT_DIR / "checkpoints" / "lfm3_rpr_lora"
LATEST_DIR = CKPT_ROOT / "latest"
STAGE1_BEST_DIR = CKPT_ROOT / "stage1_best"
BEST_DIR = ROOT_DIR / "checkpoints" / "lfm3_rpr_lora_best"


def _apply_template_safe(
    processor: Any,
    config: Any,
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool = True,
) -> str:
    from mlx_vlm.prompt_utils import apply_chat_template

    try:
        return apply_chat_template(
            processor,
            config,
            messages,
            num_images=0,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            preserve_thinking=True,
        )
    except TypeError:
        return apply_chat_template(
            processor,
            config,
            messages,
            num_images=0,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )


def _tokenize_turn_examples(
    rows: list[dict[str, Any]],
    processor: Any,
    config: Any,
    *,
    max_seq_len: int = 1280,
) -> list[dict[str, Any]]:
    """Expand single-turn and multi-turn SFT rows into exact assistant-masked token sequences."""
    tok = processor.tokenizer
    eos_id = int(tok.eos_token_id)
    prepared: list[dict[str, Any]] = []

    for row in rows:
        msgs = row["messages"]
        for t_idx, msg in enumerate(msgs):
            if msg.get("role") != "assistant":
                continue
            prefix_msgs = msgs[:t_idx]
            prompt_str = _apply_template_safe(processor, config, prefix_msgs, enable_thinking=True)
            ans_str = str(msg["content"])
            if prompt_str.endswith("<think>") and ans_str.startswith("<think>"):
                ans_str = ans_str[len("<think>") :]
            elif prompt_str.endswith("<think>\n") and ans_str.startswith("<think>\n"):
                ans_str = ans_str[len("<think>\n") :]

            p_ids = tok.encode(prompt_str, add_special_tokens=False)
            a_ids = tok.encode(ans_str, add_special_tokens=False) + [eos_id]
            if not a_ids:
                continue
            full = (p_ids + a_ids)[:max_seq_len]
            if len(full) <= len(p_ids) + 2:
                continue
            start_idx = max(len(p_ids) - 1, 0)
            prepared.append(
                {
                    "id": f"{row.get('id', 'sft')}_t{t_idx}",
                    "pillar": row.get("pillar", "unknown"),
                    "full_ids": mx.array([full], dtype=mx.int32),
                    "start_idx": start_idx,
                }
            )
    return prepared


def chunked_token_ce(
    lm_model: nn.Module,
    full_ids: mx.array,
    start_idx: int,
    *,
    chunk_size: int = 256,
) -> mx.array:
    """Compute per-token cross-entropy `[1, n_tok]` in 256-token chunks without 2.4GB full logit allocation."""
    in_ids = full_ids[:, :-1]
    targets = full_ids[:, 1 + start_idx :]
    emb = mx.stop_gradient(lm_model.model.embed_tokens(in_ids))
    h = lm_model.model(in_ids, cache=None, input_embeddings=emb)[:, start_idx:, :]
    n_tok = int(targets.shape[1])
    chunks: list[mx.array] = []
    tied = bool(lm_model.config.tie_word_embeddings)
    for s in range(0, n_tok, chunk_size):
        e = min(s + chunk_size, n_tok)
        h_c = h[:, s:e, :]
        logits_c = (
            lm_model.model.embed_tokens.as_linear(h_c)
            if tied
            else lm_model.lm_head(h_c)
        )
        ce_c = nn.losses.cross_entropy(logits_c, targets[:, s:e], reduction="none").astype(mx.float32)
        chunks.append(ce_c)
    return mx.concatenate(chunks, axis=1)


def evaluate_val_loss(model: nn.Module, val_items: list[dict[str, Any]]) -> tuple[float, float]:
    lm = model.language_model
    total_nll = 0.0
    total_toks = 0
    for item in val_items:
        ce = chunked_token_ce(lm, item["full_ids"], item["start_idx"])
        loss_sum = mx.sum(ce)
        mx.eval(loss_sum)
        total_nll += float(loss_sum.item())
        total_toks += int(ce.shape[1])
    mx.clear_cache()
    avg = total_nll / max(1, total_toks)
    ppl = float(min(1e6, math.exp(min(avg, 20.0))))
    return round(avg, 4), round(ppl, 4)


def _save_checkpoint(
    model: nn.Module,
    ckpt_dir: Path,
    state_payload: dict[str, Any] | None = None,
) -> None:
    from mlx_vlm.trainer.peft import save_adapter

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_adapter(model, ckpt_dir / "adapters.safetensors")
    if state_payload is not None:
        tmp = ckpt_dir / "train_state.json.tmp"
        tmp.write_text(json.dumps(state_payload, indent=2), encoding="utf-8")
        tmp.replace(ckpt_dir / "train_state.json")


def _set_lora_scales(model: nn.Module, scale: float) -> dict[str, float]:
    prev: dict[str, float] = {}
    for name, mod in model.named_modules():
        if hasattr(mod, "scale") and hasattr(mod, "lora_a"):
            prev[name] = float(mod.scale)
            mod.scale = scale
    return prev


def _restore_lora_scales(model: nn.Module, prev: dict[str, float]) -> None:
    for name, mod in model.named_modules():
        if name in prev:
            mod.scale = prev[name]


def _run_dft_step(
    model: nn.Module,
    items: list[dict[str, Any]],
    optimizer: optim.Optimizer,
) -> float:
    """Accumulated Dynamic Fine-Tuning (DFT + CE blend) step over `items`."""
    lm = model.language_model

    def item_loss(root_mod: nn.Module, full_ids: mx.array, start_idx: int) -> mx.array:
        ce = chunked_token_ce(root_mod.language_model, full_ids, start_idx)
        p_t = mx.stop_gradient(mx.exp(-ce))
        return mx.mean((0.5 * p_t + 0.5) * ce)

    val_and_grad = nn.value_and_grad(model, item_loss)
    acc_grads = None
    losses: list[float] = []
    n = max(1, len(items))
    for it in items:
        loss, grads = val_and_grad(model, it["full_ids"], it["start_idx"])
        mx.eval(loss)
        losses.append(float(loss.item()))
        if acc_grads is None:
            acc_grads = tree_map(lambda g: g / n, grads)
        else:
            acc_grads = tree_map(lambda a, g: a + g / n, acc_grads, grads)
    if acc_grads is not None:
        acc_grads, _ = optim.clip_grad_norm(acc_grads, max_norm=1.0)
        optimizer.update(model, acc_grads)
        mx.eval(model.trainable_parameters(), optimizer.state)
    mx.clear_cache()
    return sum(losses) / max(1, len(losses))


def run_training(args: argparse.Namespace) -> None:
    import mlx_vlm
    from mlx_vlm.trainer.peft import get_peft_model, load_adapters

    mx.set_wired_limit(0)
    mx.set_cache_limit(256 * 1024 * 1024)

    train_raw = [json.loads(l) for l in (DATA_DIR / "train_sft.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    val_raw = [json.loads(l) for l in (DATA_DIR / "val_sft.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rlvr_raw = [json.loads(l) for l in (DATA_DIR / "rlvr_prompts.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    print(f"[Train] Loading read-only base model {args.model}...", flush=True)
    model, processor = mlx_vlm.load(args.model)
    tok = processor.tokenizer

    resume_state: dict[str, Any] = {
        "stage": "dft",
        "epoch": 0,
        "sample_idx": 0,
        "rl_idx": 0,
        "base_val_loss": None,
        "best_val_loss": 1e9,
        "stage1_done": False,
    }
    state_file = LATEST_DIR / "train_state.json"
    if not args.fresh and state_file.exists() and (LATEST_DIR / "adapters.safetensors").exists():
        resume_state = json.loads(state_file.read_text(encoding="utf-8"))
        model = load_adapters(model, str(LATEST_DIR))
        print(f"[Checkpoint] Resumed from {LATEST_DIR}: {resume_state}", flush=True)
    else:
        lm_leaf_targets = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "in_proj", "out_proj", "w1", "w2", "w3"}
        lm_qualified = [
            f"language_model.{name}"
            for name, _ in model.language_model.named_modules()
            if name.split(".")[-1] in lm_leaf_targets
        ]
        model = get_peft_model(
            model,
            linear_layers=lm_qualified,
            rank=16,
            alpha=32,
            dropout=0.0,
            verbose=False,
        )
        print(f"[Train] Initialized fresh rank-16 LoRA across {len(lm_qualified)} LM modules.", flush=True)

    train_items = _tokenize_turn_examples(train_raw, processor, model.config, max_seq_len=1280)
    val_items = _tokenize_turn_examples(val_raw, processor, model.config, max_seq_len=1280)
    print(f"[Train] Tokenized turns: train={len(train_items)}, val={len(val_items)}, rlvr={len(rlvr_raw)}", flush=True)

    if resume_state.get("base_val_loss") is None:
        base_vl, base_vp = evaluate_val_loss(model, val_items)
        resume_state["base_val_loss"] = base_vl
        resume_state["best_val_loss"] = base_vl
        _save_checkpoint(model, BEST_DIR)
        _save_checkpoint(model, LATEST_DIR, resume_state)
        print(f"[Stage 0 Base] val_loss={base_vl:.4f} (PPL={base_vp:.4f})", flush=True)

    # ------------------------------------------------------------------
    # STAGE 1: DFT + CE Warm-Start
    # ------------------------------------------------------------------
    epochs = args.sft_epochs
    if not resume_state.get("stage1_done"):
        steps_per_epoch = math.ceil(len(train_items) / 2)
        total_steps = max(1, steps_per_epoch * epochs)
        lr_sched = optim.cosine_decay(8.0e-5, decay_steps=total_steps, end=8.0e-6)
        opt_s1 = optim.AdamW(learning_rate=lr_sched, weight_decay=0.01)

        rng = random.Random(20261121)
        checked_first_step = False
        for ep in range(resume_state["epoch"], epochs):
            order = list(range(len(train_items)))
            rng.shuffle(order)
            start_i = resume_state["sample_idx"] if ep == resume_state["epoch"] else 0
            buf: list[dict[str, Any]] = []
            running_loss: list[float] = []

            for i in range(start_i, len(order)):
                buf.append(train_items[order[i]])
                if len(buf) == 2 or i == len(order) - 1:
                    l_val = _run_dft_step(model, buf, opt_s1)
                    running_loss.append(l_val)
                    buf.clear()
                    if not checked_first_step:
                        checked_first_step = True
                        max_lora_b = max(
                            float(mx.max(mx.abs(v)).item())
                            for k, v in tree_flatten(model.trainable_parameters())
                            if "lora_b" in k
                        )
                        print(f"  [Step 1 Sanity] max(|lora_b|)={max_lora_b:.6e}", flush=True)
                        assert max_lora_b > 1e-7, "FATAL: lora_b remained zero after first optimizer step!"

                if (i + 1) % 68 == 0 or i == len(order) - 1:
                    vl, vp = evaluate_val_loss(model, val_items)
                    avg_tr = sum(running_loss) / max(1, len(running_loss))
                    running_loss.clear()
                    resume_state["stage"] = "dft"
                    resume_state["epoch"] = ep
                    resume_state["sample_idx"] = i + 1
                    if vl < resume_state["best_val_loss"]:
                        resume_state["best_val_loss"] = vl
                        _save_checkpoint(model, BEST_DIR)
                    _save_checkpoint(model, LATEST_DIR, resume_state)
                    print(
                        f"  [DFT ep {ep+1}/{epochs} sample {i+1:3d}/{len(order)}] "
                        f"train_loss={avg_tr:.4f} | val_loss={vl:.4f} (PPL={vp:.4f}) | best={resume_state['best_val_loss']:.4f}",
                        flush=True,
                    )
            resume_state["sample_idx"] = 0

        if (BEST_DIR / "adapters.safetensors").exists():
            model.load_weights(str(BEST_DIR / "adapters.safetensors"), strict=False)
            mx.eval(model.parameters())
        _save_checkpoint(model, STAGE1_BEST_DIR)
        resume_state["stage1_done"] = True
        resume_state["stage"] = "rlvr"
        _save_checkpoint(model, LATEST_DIR, resume_state)
        gc.collect()
        mx.clear_cache()

    # ------------------------------------------------------------------
    # STAGE 2: Online Verifiable RLVR (GDPO + Dr. GRPO + DAPO + OPSD Rescue)
    # ------------------------------------------------------------------
    opt_s2 = optim.AdamW(learning_rate=1.2e-5, weight_decay=0.0)
    eos_id = int(tok.eos_token_id)
    max_comp_len = 384
    g_rollouts = 4

    rlvr_order = list(range(len(rlvr_raw)))
    random.Random(20261122).shuffle(rlvr_order)
    for g_idx in range(resume_state.get("rl_idx", 0), len(rlvr_order)):
        prompt_row = rlvr_raw[rlvr_order[g_idx]]
        prompt_msgs = prompt_row["messages"]
        prompt_str = _apply_template_safe(processor, model.config, prompt_msgs, enable_thinking=True)
        raw_prompt_txt = "\n".join(str(m.get("content", "")) for m in prompt_msgs)
        p_ids = tok.encode(prompt_str, add_special_tokens=False)
        prompt_mx = mx.array([p_ids], dtype=mx.int32)
        start_idx = max(len(p_ids) - 1, 0)

        # Rollout sampling (KV cache enabled)
        rollouts: list[dict[str, Any]] = []
        for _r in range(g_rollouts):
            res = mlx_vlm.generate(
                model,
                processor,
                prompt_str,
                max_tokens=max_comp_len,
                temperature=0.7,
                repetition_penalty=1.05,
                verbose=False,
            )
            gen_txt = res.text or ""
            comp_ids = tok.encode(gen_txt, add_special_tokens=False) + [eos_id]
            full_seq = mx.concatenate([prompt_mx, mx.array([comp_ids], dtype=mx.int32)], axis=1)
            if int(full_seq.shape[1]) > 1408:
                full_seq = full_seq[:, :1408]
            sc = score_response(
                gen_txt,
                prompt_text=raw_prompt_txt,
                kind=prompt_row["kind"],
                verifier=prompt_row["verifier"],
                tokenizer=tok,
                max_think_tokens=240,
            )
            rollouts.append({"text": gen_txt, "full_ids": full_seq, "score": sc})

        rewards = [x["score"]["reward"] for x in rollouts]
        # OPSD / DFT rescue if all rollouts scored low or zero reward spread
        if max(rewards) <= 0.25 or (max(rewards) - min(rewards) < 1e-4 and max(rewards) < 0.9):
            ref_row = {
                "id": f"rescue_{prompt_row['id']}",
                "pillar": prompt_row["kind"],
                "messages": [*prompt_msgs, {"role": "assistant", "content": prompt_row["reference_answer"]}],
            }
            ref_items = _tokenize_turn_examples([ref_row], processor, model.config, max_seq_len=1408)
            if ref_items:
                _run_dft_step(model, ref_items, opt_s2)
            print(
                f"  [RLVR {g_idx+1:2d}/{len(rlvr_raw)}] {prompt_row['id']} -> OPSD/DFT rescue "
                f"(rewards={[round(r,2) for r in rewards]})",
                flush=True,
            )
        else:
            # GDPO per-channel centered Dr. GRPO advantages (no sigma scaling!)
            ch_weights = {"correct": 1.0, "schema": 0.5, "close_think": 0.4, "anti_echo": 0.4, "tool": 0.6}
            advs = [0.0] * len(rollouts)
            for ck, cw in ch_weights.items():
                vals = [float(x["score"]["channels"][ck]) for x in rollouts]
                mu = sum(vals) / len(vals)
                for idx_r, v in enumerate(vals):
                    advs[idx_r] += cw * (v - mu)

            # Precompute old_logprobs (current) & ref_logprobs (scale=0.0) without graph overhead
            lm = model.language_model
            old_lps = [mx.stop_gradient(-chunked_token_ce(lm, ro["full_ids"], start_idx)) for ro in rollouts]
            prev_scales = _set_lora_scales(model, 0.0)
            ref_lps = [mx.stop_gradient(-chunked_token_ce(lm, ro["full_ids"], start_idx)) for ro in rollouts]
            _restore_lora_scales(model, prev_scales)
            mx.eval(*(old_lps + ref_lps))

            def rl_item_loss(
                root_mod: nn.Module,
                full_ids: mx.array,
                old_lp: mx.array,
                ref_lp: mx.array,
                adv_val: float,
            ) -> mx.array:
                new_lp = -chunked_token_ce(root_mod.language_model, full_ids, start_idx)
                ratio = mx.exp(mx.clip(new_lp - old_lp, -5.0, 5.0))
                adv = mx.array(adv_val, dtype=mx.float32)
                pg_unclipped = ratio * adv
                pg_clipped = mx.clip(ratio, 0.8, 1.28) * adv
                pg_loss = -mx.minimum(pg_unclipped, pg_clipped)
                kl = mx.exp(ref_lp - new_lp) - (ref_lp - new_lp) - 1.0
                return mx.sum(pg_loss + 0.025 * kl) / mx.array(float(max_comp_len), dtype=mx.float32)

            rl_val_grad = nn.value_and_grad(model, rl_item_loss)
            acc_grads = None
            n_r = float(len(rollouts))
            for ro, old_lp, ref_lp, adv_v in zip(rollouts, old_lps, ref_lps, advs):
                l_r, g_r = rl_val_grad(model, ro["full_ids"], old_lp, ref_lp, adv_v)
                mx.eval(l_r)
                if acc_grads is None:
                    acc_grads = tree_map(lambda g: g / n_r, g_r)
                else:
                    acc_grads = tree_map(lambda a, g: a + g / n_r, acc_grads, g_r)
            if acc_grads is not None:
                acc_grads, _ = optim.clip_grad_norm(acc_grads, max_norm=1.0)
                opt_s2.update(model, acc_grads)
                mx.eval(model.trainable_parameters(), opt_s2.state)
            mx.clear_cache()

            mean_r = sum(rewards) / len(rewards)
            pass_cnt = sum(1 for ro in rollouts if ro["score"]["correct"])
            print(
                f"  [RLVR {g_idx+1:2d}/{len(rlvr_raw)}] {prompt_row['id']} | pass={pass_cnt}/{len(rollouts)} | mean_r={mean_r:.3f}",
                flush=True,
            )

        resume_state["rl_idx"] = g_idx + 1
        if (g_idx + 1) % 6 == 0 or g_idx == len(rlvr_raw) - 1:
            vl, vp = evaluate_val_loss(model, val_items)
            max_allowed_vl = float(resume_state["best_val_loss"]) + 0.08
            _save_checkpoint(model, LATEST_DIR, resume_state)
            if vl <= max_allowed_vl:
                _save_checkpoint(model, BEST_DIR)
            print(f"  [RLVR Ckpt] step={g_idx+1} | val_loss={vl:.4f} (PPL={vp:.4f})", flush=True)

    print(f"[Train] Finished! Best adapter saved to {BEST_DIR}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LFM3-VL-2B LoRA for RPR, tool discipline, and schema following.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Read-only base model directory.")
    parser.add_argument("--sft-epochs", type=int, default=3, help="Stage-1 DFT epochs.")
    parser.add_argument("--fresh", action="store_true", help="Ignore existing training checkpoints.")
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
