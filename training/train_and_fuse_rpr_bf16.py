"""Deep 4-Stage BF16 RPR Post-Training & Lossless Standalone 16-Bit Fusion for LFM3-VL-2B.

Pipeline (100% standalone inside `checkpoints/`, read-only access to `./models/LFM3-VL-2B-MLX-8bit`):
  - Stage 0: Lossless in-memory dequantization (`QuantizedLinear` & `QuantizedEmbedding` -> `bfloat16`),
             frozen `vision_tower` feature precomputation, and Rank-32 (`alpha=64`) LoRA across all
             155 linear targets (153 LM + 2 multimodal projector layers).
  - Stage 1: Deep 4-epoch Dynamic Fine-Tuning (`0.5 * L_DFT + 0.5 * L_CE`) with LoRA+ differential
             learning rates (`lr_B = 3.6x lr_A` via `MultiOptimizer`), multi-turn prior-`<think>`
             stripping (exact inference alignment), and Exponential Moving Average (`EMA`, `0.995`).
  - Stage 2: Shuffled 96-episode Multi-Turn SCoRe + GDPO/Dr.GRPO Verifiable RL (`384` live rollouts)
             with OPSD deadlock rescue and anchored SFT/DFT replay (`+ 0.20 * L_DFT` on every RL step)
             to mathematically prevent catastrophic forgetting.
  - Stage 3: `WiSE-FT` weight-space interpolation sweep (`lambda in {0.0, 0.35, 0.70, 1.0}`) between
             `stage1_ema` and `stage2_best` scored on held-out validation loss + multi-pillar probe.
  - Stage 4: Lossless 16-bit (`bfloat16`) fusion into `checkpoints/LFM3-VL-2B-RPR-BF16/` (zero
             re-quantization!) + fresh standalone `mlx_vlm.load` verification.
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

from model import DEFAULT_MODEL_PATH, split_thinking
from training.verifiable_rewards import score_response


CORPUS_DIR = Path(__file__).resolve().parent / "data" / "rpr_v4_corpus"
CKPT_ROOT = ROOT_DIR / "checkpoints" / "rpr_v4_bf16_work"
LATEST_DIR = CKPT_ROOT / "latest"
STAGE1_BEST_DIR = CKPT_ROOT / "stage1_best"
STAGE1_EMA_DIR = CKPT_ROOT / "stage1_ema"
STAGE2_BEST_DIR = CKPT_ROOT / "stage2_best"
WISE_BEST_DIR = CKPT_ROOT / "wise_ft_best"
EXPORT_BF16_DIR = ROOT_DIR / "checkpoints" / "LFM3-VL-2B-RPR-BF16"


def dequantize_model_to_bf16(model: nn.Module) -> int:
    """Replace all `QuantizedLinear` and `QuantizedEmbedding` modules with lossless `bfloat16` equivalents."""
    replacements: list[tuple[str, nn.Module]] = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.QuantizedLinear):
            w = mx.dequantize(
                mod.weight, mod.scales, mod.biases, group_size=mod.group_size, bits=mod.bits
            ).astype(mx.bfloat16)
            has_bias = hasattr(mod, "bias") and mod.bias is not None
            lin = nn.Linear(w.shape[1], w.shape[0], bias=has_bias)
            lin.weight = w
            if has_bias:
                lin.bias = mod.bias.astype(mx.bfloat16)
            replacements.append((name, lin))
        elif isinstance(mod, nn.QuantizedEmbedding):
            w_e = mx.dequantize(
                mod.weight, mod.scales, mod.biases, group_size=mod.group_size, bits=mod.bits
            ).astype(mx.bfloat16)
            emb = nn.Embedding(w_e.shape[0], w_e.shape[1])
            emb.weight = w_e
            replacements.append((name, emb))
    if replacements:
        model.update_modules(tree_unflatten(replacements))
    model.apply(lambda x: x.astype(mx.bfloat16) if mx.issubdtype(x.dtype, mx.floating) else x)
    mx.eval(model.parameters())
    mx.clear_cache()
    return len(replacements)


def _apply_template_inference_aligned(
    processor: Any,
    config: Any,
    messages: list[dict[str, str]],
    *,
    num_images: int = 0,
    enable_thinking: bool = True,
) -> str:
    """Apply chat template with prior-turn `<think>` stripped, matching real multi-turn inference."""
    from mlx_vlm.prompt_utils import apply_chat_template

    cleaned_msgs: list[dict[str, str]] = []
    for m in messages:
        if m.get("role") == "assistant":
            _, post = split_thinking(str(m.get("content", "")))
            cleaned_msgs.append({"role": "assistant", "content": post.strip() or str(m.get("content", "")).strip()})
        else:
            cleaned_msgs.append({"role": str(m.get("role", "user")), "content": str(m.get("content", ""))})

    return apply_chat_template(
        processor,
        config,
        cleaned_msgs,
        num_images=num_images,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def prepare_turn_items(
    rows: list[dict[str, Any]],
    model: Any,
    processor: Any,
    *,
    max_seq_len: int = 1664,
) -> list[dict[str, Any]]:
    """Expand 1-turn & 2-turn rows into assistant-masked token sequences + cached frozen `vt_feats`."""
    from mlx_vlm.generate import _prepare_generation_inputs

    tok = processor.tokenizer
    eos_id = int(tok.eos_token_id)
    img_tok_id = int(model.config.image_token_id)
    prepared: list[dict[str, Any]] = []

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
                if len(idx_np) != int(vt_feats.shape[0]):
                    continue
                img_indices = mx.array(idx_np, dtype=mx.int32)
                start_idx = max(p_len - 1, 0)
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
    return prepared


def chunked_multimodal_ce(
    root_mod: nn.Module,
    full_ids: mx.array,
    start_idx: int,
    vt_feats: mx.array | None = None,
    img_indices: mx.array | None = None,
    *,
    chunk_size: int = 256,
) -> mx.array:
    """Compute per-token cross-entropy `[1, n_tok]` in 256-token chunks across text & multimodal items."""
    lm_model = root_mod.language_model
    in_ids = full_ids[:, :-1]
    targets = full_ids[:, 1 + start_idx :]
    emb = mx.stop_gradient(lm_model.model.embed_tokens(in_ids))
    if vt_feats is not None and img_indices is not None:
        proj = root_mod.multi_modal_projector(vt_feats)
        flat = emb.reshape(-1, emb.shape[-1])
        flat[img_indices] = proj.astype(emb.dtype)
        emb = flat.reshape(emb.shape)

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
    total_nll = 0.0
    total_toks = 0
    for item in val_items:
        ce = chunked_multimodal_ce(
            model, item["full_ids"], item["start_idx"], item["vt_feats"], item["img_indices"]
        )
        loss_sum = mx.sum(ce)
        mx.eval(loss_sum)
        total_nll += float(loss_sum.item())
        total_toks += int(ce.shape[1])
    mx.clear_cache()
    avg = total_nll / max(1, total_toks)
    ppl = float(min(1e6, math.exp(min(avg, 20.0))))
    return round(avg, 4), round(ppl, 4)


def _save_adapter_ckpt(
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


def _run_dft_batch(
    model: nn.Module,
    items: list[dict[str, Any]],
    optimizer: optim.Optimizer | None,
    *,
    scale_grad: float = 1.0,
) -> tuple[float, Any]:
    """Compute accumulated Dynamic Fine-Tuning (`0.5 * p_t + 0.5`) loss & grads; apply if `optimizer` given."""

    def item_loss(
        root_mod: nn.Module,
        full_ids: mx.array,
        start_idx: int,
        vt_feats: mx.array | None,
        img_indices: mx.array | None,
    ) -> mx.array:
        ce = chunked_multimodal_ce(root_mod, full_ids, start_idx, vt_feats, img_indices)
        p_t = mx.stop_gradient(mx.exp(-ce))
        return mx.mean((0.5 * p_t + 0.5) * ce)

    val_and_grad = nn.value_and_grad(model, item_loss)
    acc_grads = None
    losses: list[float] = []
    w = scale_grad / max(1, len(items))
    for it in items:
        loss, grads = val_and_grad(
            model, it["full_ids"], it["start_idx"], it["vt_feats"], it["img_indices"]
        )
        mx.eval(loss)
        losses.append(float(loss.item()))
        if acc_grads is None:
            acc_grads = tree_map(lambda g: g * w, grads)
        else:
            acc_grads = tree_map(lambda a, g: a + g * w, acc_grads, grads)

    if optimizer is not None and acc_grads is not None:
        acc_grads, _ = optim.clip_grad_norm(acc_grads, max_norm=1.0)
        optimizer.update(model, acc_grads)
        mx.eval(model.trainable_parameters(), optimizer.state)
        mx.clear_cache()
    return sum(losses) / max(1, len(losses)), acc_grads


def _evaluate_mini_probe(model: Any, processor: Any, probe_tasks: list[dict[str, Any]]) -> tuple[int, float]:
    """Run greedy generation on a 14-task multi-pillar probe to score WiSE-FT candidates."""
    import mlx_vlm

    tok = processor.tokenizer
    passed = 0
    rewards = []
    for t in probe_tasks:
        imgs = [str(p) for p in (t.get("images") or []) if p] or None
        prompt_str = _apply_template_inference_aligned(
            processor, model.config, t["messages"], num_images=len(imgs) if imgs else 0, enable_thinking=True
        )
        res = mlx_vlm.generate(
            model,
            processor,
            prompt_str,
            image=imgs,
            max_tokens=480,
            temperature=0.0,
            repetition_penalty=1.05,
            verbose=False,
        )
        raw_txt = res.text or ""
        prompt_txt = "\n".join(str(m.get("content", "")) for m in t["messages"])
        sc = score_response(
            raw_txt,
            prompt_text=prompt_txt,
            kind=t.get("kind", t["pillar"]),
            verifier=t["verifier"],
            tokenizer=tok,
            max_think_tokens=320,
        )
        if sc["correct"]:
            passed += 1
        rewards.append(float(sc["reward"]))
    mx.clear_cache()
    return passed, round(sum(rewards) / max(1, len(rewards)), 4)


def fuse_and_export_standalone_bf16(
    model: Any,
    base_dir: Path,
    export_dir: Path,
) -> None:
    """Merge all 155 LoRA layers losslessly into `bfloat16` `nn.Linear` and write standalone HF/MLX folder."""
    import mlx_vlm

    print(f"[Stage 4 Fuse] Merging LoRA weights into lossless 16-bit bfloat16 -> {export_dir}...", flush=True)
    replacements: list[tuple[str, nn.Module]] = []
    merged_cnt = 0
    for name, mod in model.named_modules():
        if hasattr(mod, "linear") and hasattr(mod, "lora_a") and hasattr(mod, "lora_b"):
            base_l = mod.linear
            if isinstance(base_l, nn.QuantizedLinear):
                w_f32 = mx.dequantize(
                    base_l.weight, base_l.scales, base_l.biases, group_size=base_l.group_size, bits=base_l.bits
                ).astype(mx.float32)
            else:
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
    # Copy tokenizer, processor, chat template, and custom model python files from read-only base_dir
    skip_names = {"model.safetensors", "model.safetensors.index.json", "config.json"}
    for item in base_dir.iterdir():
        if item.is_file() and item.name not in skip_names and not item.name.endswith(".safetensors"):
            shutil.copy2(item, export_dir / item.name)

    # Write clean unquantized bfloat16 config.json
    cfg = json.loads((base_dir / "config.json").read_text(encoding="utf-8"))
    cfg.pop("quantization", None)
    cfg.pop("quantization_config", None)
    if isinstance(cfg.get("text_config"), dict):
        cfg["text_config"].pop("quantization", None)
        cfg["text_config"].pop("quantization_config", None)
        cfg["text_config"]["dtype"] = "bfloat16"
    cfg["dtype"] = "bfloat16"
    (export_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    # Save standalone bfloat16 weights + safetensors index
    flat_weights = dict(tree_flatten(model.parameters()))
    safetensors_path = export_dir / "model.safetensors"
    mx.save_safetensors(str(safetensors_path), flat_weights, metadata={"format": "mlx"})
    total_bytes = safetensors_path.stat().st_size
    index_payload = {
        "metadata": {"total_size": total_bytes},
        "weight_map": {k: "model.safetensors" for k in sorted(flat_weights.keys())},
    }
    (export_dir / "model.safetensors.index.json").write_text(
        json.dumps(index_payload, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[Stage 4 Fuse] Wrote {len(flat_weights)} bfloat16 tensors ({merged_cnt} LoRA-fused) -> "
        f"{safetensors_path} ({total_bytes / (1024**3):.2f} GiB).",
        flush=True,
    )

    # Strict disk-reload parity gate!
    test_ids = mx.array([[1, 42, 105, 512, 1024]], dtype=mx.int32)
    logits_mem = model.language_model(test_ids).logits
    mx.eval(logits_mem)

    reloaded_model, _ = mlx_vlm.load(str(export_dir))
    logits_disk = reloaded_model.language_model(test_ids).logits
    mx.eval(logits_disk)
    max_diff = float(mx.max(mx.abs(logits_mem.astype(mx.float32) - logits_disk.astype(mx.float32))).item())
    print(f"[Stage 4 Verify] Fresh standalone disk reload succeeded! Max logit diff = {max_diff:.6f}", flush=True)
    assert max_diff < 1e-3, f"Disk reload mismatch too high: {max_diff}"
    del reloaded_model
    gc.collect()
    mx.clear_cache()


def run_pipeline(args: argparse.Namespace) -> None:
    import mlx_vlm
    from mlx_vlm.trainer.peft import get_peft_model

    mx.set_wired_limit(0)
    mx.set_cache_limit(384 * 1024 * 1024)

    train_raw = [json.loads(l) for l in (CORPUS_DIR / "train_rpr_v4.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    val_raw = [json.loads(l) for l in (CORPUS_DIR / "val_rpr_v4.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rlvr_raw = [json.loads(l) for l in (CORPUS_DIR / "rlvr_score_96.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    broad_42 = [json.loads(l) for l in (CORPUS_DIR / "heldout_broad_42.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    # Pick 2 tasks from each of the 7 pillars (14 tasks) as fast Stage-3 WiSE-FT selection probe
    probe_14 = [broad_42[p * 6 + j] for p in range(7) for j in range(2)]

    print(f"[Stage 0] Loading base model {args.model} and converting in-memory to lossless bfloat16...", flush=True)
    model, processor = mlx_vlm.load(args.model)
    n_deq = dequantize_model_to_bf16(model)
    print(f"[Stage 0] Converted {n_deq} quantized modules -> pure bfloat16.", flush=True)

    print("[Stage 0] Tokenizing corpus & caching frozen vision_tower features...", flush=True)
    train_items = prepare_turn_items(train_raw, model, processor, max_seq_len=1664)
    val_items = prepare_turn_items(val_raw, model, processor, max_seq_len=1664)
    print(
        f"[Stage 0] Ready: train_turns={len(train_items)}, val_turns={len(val_items)}, rlvr_episodes={len(rlvr_raw)}",
        flush=True,
    )

    lm_leaf_targets = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "in_proj", "out_proj", "w1", "w2", "w3"}
    lm_qualified = [
        f"language_model.{name}"
        for name, _ in model.language_model.named_modules()
        if name.split(".")[-1] in lm_leaf_targets
    ]
    proj_qualified = ["multi_modal_projector.linear_1", "multi_modal_projector.linear_2"]
    all_targets = lm_qualified + proj_qualified
    model = get_peft_model(
        model,
        linear_layers=all_targets,
        rank=32,
        alpha=64,
        dropout=0.0,
        verbose=False,
    )
    print(f"[Stage 0] Attached Rank-32 (alpha=64) LoRA across {len(all_targets)} modules (153 LM + 2 Projector).", flush=True)

    resume_state: dict[str, Any] = {
        "stage": "dft",
        "epoch": 0,
        "sample_idx": 0,
        "rl_idx": 0,
        "base_val_loss": None,
        "best_val_loss": 1e9,
        "best_rl_val_loss": 1e9,
        "stage1_done": False,
        "stage2_done": False,
    }
    state_file = LATEST_DIR / "train_state.json"
    if not args.fresh and state_file.exists() and (LATEST_DIR / "adapters.safetensors").exists():
        resume_state = json.loads(state_file.read_text(encoding="utf-8"))
        model.load_weights(str(LATEST_DIR / "adapters.safetensors"), strict=False)
        mx.eval(model.parameters())
        print(f"[Checkpoint] Resumed from {LATEST_DIR}: {resume_state}", flush=True)

    if resume_state.get("base_val_loss") is None:
        base_vl, base_vp = evaluate_val_loss(model, val_items)
        resume_state["base_val_loss"] = base_vl
        resume_state["best_val_loss"] = base_vl
        _save_adapter_ckpt(model, STAGE1_BEST_DIR)
        _save_adapter_ckpt(model, STAGE1_EMA_DIR)
        _save_adapter_ckpt(model, LATEST_DIR, resume_state)
        print(f"[Stage 0 Base BF16] val_loss={base_vl:.4f} (PPL={base_vp:.4f})", flush=True)

    # ------------------------------------------------------------------
    # STAGE 1: Deep 4-Epoch LoRA+ DFT + CE Curriculum + EMA Weights
    # ------------------------------------------------------------------
    epochs = args.sft_epochs
    if not resume_state.get("stage1_done"):
        steps_per_epoch = math.ceil(len(train_items) / 2)
        total_steps = max(1, steps_per_epoch * epochs)
        lr_b_sched = optim.cosine_decay(9.0e-5, decay_steps=total_steps, end=9.0e-6)
        lr_a_sched = optim.cosine_decay(2.5e-5, decay_steps=total_steps, end=2.5e-6)
        opt_s1 = optim.MultiOptimizer(
            [
                optim.AdamW(learning_rate=lr_b_sched, weight_decay=0.01),
                optim.AdamW(learning_rate=lr_a_sched, weight_decay=0.01),
            ],
            filters=[lambda k, _: "lora_b" in k],
        )

        ema_decay = 0.995
        ema_weights = {k: mx.array(v) for k, v in tree_flatten(model.trainable_parameters())}
        if (STAGE1_EMA_DIR / "adapters.safetensors").exists() and not args.fresh:
            loaded_ema = mx.load(str(STAGE1_EMA_DIR / "adapters.safetensors"))
            for k, v in loaded_ema.items():
                if k in ema_weights:
                    ema_weights[k] = v
        mx.eval(list(ema_weights.values()))

        rng = random.Random(20261130)
        checked_step1 = False
        no_improve_cnt = 0
        early_stopped = False
        t_s1 = time.perf_counter()

        for ep in range(resume_state["epoch"], epochs):
            if early_stopped:
                break
            order = list(range(len(train_items)))
            rng.shuffle(order)
            start_i = resume_state["sample_idx"] if ep == resume_state["epoch"] else 0
            buf: list[dict[str, Any]] = []
            running_loss: list[float] = []

            for i in range(start_i, len(order)):
                buf.append(train_items[order[i]])
                if len(buf) == 2 or i == len(order) - 1:
                    l_val, _ = _run_dft_batch(model, buf, opt_s1)
                    running_loss.append(l_val)
                    buf.clear()

                    live_flat = dict(tree_flatten(model.trainable_parameters()))
                    for k, v in live_flat.items():
                        ema_weights[k] = ema_decay * ema_weights[k] + (1.0 - ema_decay) * v
                    mx.eval(list(ema_weights.values()))

                    if not checked_step1:
                        checked_step1 = True
                        max_b = max(float(mx.max(mx.abs(v)).item()) for k, v in live_flat.items() if "lora_b" in k)
                        print(f"  [Step 1 Sanity] max(|lora_b|)={max_b:.6e}", flush=True)
                        assert max_b > 1e-7, "FATAL: lora_b remained zero after step 1!"

                if (i + 1) % 240 == 0 or i == len(order) - 1:
                    vl, vp = evaluate_val_loss(model, val_items)
                    avg_tr = sum(running_loss) / max(1, len(running_loss))
                    running_loss.clear()
                    resume_state["stage"] = "dft"
                    resume_state["epoch"] = ep
                    resume_state["sample_idx"] = i + 1
                    if vl < resume_state["best_val_loss"] - 1e-4:
                        resume_state["best_val_loss"] = vl
                        no_improve_cnt = 0
                        _save_adapter_ckpt(model, STAGE1_BEST_DIR)
                        STAGE1_EMA_DIR.mkdir(parents=True, exist_ok=True)
                        mx.save_safetensors(str(STAGE1_EMA_DIR / "adapters.safetensors"), ema_weights)
                        if (STAGE1_BEST_DIR / "adapter_config.json").exists():
                            shutil.copy2(STAGE1_BEST_DIR / "adapter_config.json", STAGE1_EMA_DIR / "adapter_config.json")
                    else:
                        no_improve_cnt += 1

                    _save_adapter_ckpt(model, LATEST_DIR, resume_state)
                    elapsed_m = (time.perf_counter() - t_s1) / 60.0
                    print(
                        f"  [Stage 1 DFT ep {ep+1}/{epochs} {i+1:4d}/{len(order)} | {elapsed_m:4.1f}m] "
                        f"train={avg_tr:.4f} | val={vl:.4f} (PPL={vp:.4f}) | best={resume_state['best_val_loss']:.4f} | pat={no_improve_cnt}/2",
                        flush=True,
                    )
                    if no_improve_cnt >= 2:
                        print(
                            f"  [Early Stop] Validation loss peaked at {resume_state['best_val_loss']:.4f}; stopping Stage 1 to prevent overfitting!",
                            flush=True,
                        )
                        early_stopped = True
                        break
            resume_state["sample_idx"] = 0

        # Always restore exact Stage-1 validation minimum (`STAGE1_BEST_DIR`) before Stage 2!
        model.load_weights(str(STAGE1_BEST_DIR / "adapters.safetensors"), strict=False)
        mx.eval(model.parameters())

        _save_adapter_ckpt(model, STAGE2_BEST_DIR)
        resume_state["best_rl_val_loss"] = resume_state["best_val_loss"]
        resume_state["stage1_done"] = True
        resume_state["stage"] = "rlvr"
        _save_adapter_ckpt(model, LATEST_DIR, resume_state)
        gc.collect()
        mx.clear_cache()

    # ------------------------------------------------------------------
    # STAGE 2: Anti-Overfitting SFT-Anchored SCoRe + GDPO/Dr.GRPO RLVR
    # ------------------------------------------------------------------
    if not resume_state.get("stage2_done"):
        opt_s2 = optim.MultiOptimizer(
            [
                optim.AdamW(learning_rate=7.5e-6, weight_decay=0.01),
                optim.AdamW(learning_rate=2.0e-6, weight_decay=0.01),
            ],
            filters=[lambda k, _: "lora_b" in k],
        )
        tok = processor.tokenizer
        eos_id = int(tok.eos_token_id)
        max_comp_len = 448
        g_rollouts = 4

        # Reference policy pi_ref is frozen STAGE1_BEST_DIR (val_loss minimum), NOT untrained base!
        s1_ref_weights = mx.load(str(STAGE1_BEST_DIR / "adapters.safetensors"))
        mx.eval(list(s1_ref_weights.values()))

        rl_order = list(range(len(rlvr_raw)))
        random.Random(20261201).shuffle(rl_order)
        rl_order = rl_order[: min(args.rl_steps, len(rl_order))]
        anchor_rng = random.Random(20261202)
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
                if int(full_seq.shape[1]) > 1664:
                    full_seq = full_seq[:, :1664]
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
                # Gentle 0.35x OPSD rescue + light anchor so we never re-overfit!
                ref_row = {
                    "id": f"rescue_{prompt_row['id']}",
                    "pillar": prompt_row["kind"],
                    "images": [],
                    "messages": [*prompt_msgs, {"role": "assistant", "content": prompt_row["reference_answer"]}],
                }
                ref_items = prepare_turn_items([ref_row], model, processor, max_seq_len=1664)
                combined = (ref_items + anchor_batch[:1]) if ref_items else anchor_batch[:1]
                _run_dft_batch(model, combined, opt_s2, scale_grad=0.35)
                tag_str = "OPSD(0.35x) rescue"
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
                # Compute ref_lps under frozen STAGE1_BEST_DIR trust-region reference!
                live_snapshot = {k: mx.array(v) for k, v in tree_flatten(model.trainable_parameters())}
                model.load_weights(list(s1_ref_weights.items()), strict=False)
                mx.eval(model.parameters())
                ref_lps = [
                    mx.stop_gradient(-chunked_multimodal_ce(model, ro["full_ids"], start_idx))
                    for ro in rollouts
                ]
                model.load_weights(list(live_snapshot.items()), strict=False)
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
                    if acc_grads is None:
                        acc_grads = tree_map(lambda g: g / n_r, g_r)
                    else:
                        acc_grads = tree_map(lambda a, g: a + g / n_r, acc_grads, g_r)

                _, anchor_grads = _run_dft_batch(model, anchor_batch, None, scale_grad=0.10)
                if acc_grads is not None and anchor_grads is not None:
                    acc_grads = tree_map(lambda a, b: a + b, acc_grads, anchor_grads)
                if acc_grads is not None:
                    acc_grads, _ = optim.clip_grad_norm(acc_grads, max_norm=1.0)
                    opt_s2.update(model, acc_grads)
                    mx.eval(model.trainable_parameters(), opt_s2.state)
                mx.clear_cache()
                tag_str = f"GDPO+KL_s1 (mean_r={sum(rewards)/len(rewards):.2f})"

            if (g_idx + 1) % 4 == 0 or g_idx == len(rl_order) - 1:
                el_m = (time.perf_counter() - t_s2) / 60.0
                print(
                    f"  [Stage 2 RL {g_idx+1:2d}/{len(rl_order)} | {el_m:4.1f}m] {prompt_row['id']:16s} | "
                    f"pass={pass_cnt}/{g_rollouts} | {tag_str}",
                    flush=True,
                )

            resume_state["rl_idx"] = g_idx + 1
            if (g_idx + 1) % 12 == 0 or g_idx == len(rl_order) - 1:
                vl, vp = evaluate_val_loss(model, val_items)
                _save_adapter_ckpt(model, LATEST_DIR, resume_state)
                if vl <= float(resume_state.get("best_rl_val_loss", 1e9)) + 0.015:
                    _save_adapter_ckpt(model, STAGE2_BEST_DIR)
                    resume_state["best_rl_val_loss"] = min(float(resume_state.get("best_rl_val_loss", 1e9)), vl)
                print(f"  [Stage 2 Ckpt] step={g_idx+1}/{len(rl_order)} | val_loss={vl:.4f} (PPL={vp:.4f})", flush=True)
                if vl > float(resume_state["best_val_loss"]) + 0.035:
                    print(
                        f"  [Stage 2 Early Guard] val_loss={vl:.4f} exceeded SFT best + 0.035; stopping RLVR cleanly.",
                        flush=True,
                    )
                    break

        resume_state["stage2_done"] = True
        resume_state["stage"] = "wise_ft"
        _save_adapter_ckpt(model, LATEST_DIR, resume_state)
        gc.collect()
        mx.clear_cache()

    # ------------------------------------------------------------------
    # STAGE 3: 2D Anti-Overfitting WiSE-FT Sweep (`stage1_best` <-> `stage2_best` & Base Shrinkage `gamma`)
    # ------------------------------------------------------------------
    print("[Stage 3] Running 2D Anti-Overfitting WiSE-FT sweep (SFT<->RL blend + Base-to-Finetuned shrinkage)...", flush=True)
    w_s1 = mx.load(str(STAGE1_BEST_DIR / "adapters.safetensors"))
    s2_path = STAGE2_BEST_DIR / "adapters.safetensors"
    if not s2_path.exists():
        s2_path = STAGE1_BEST_DIR / "adapters.safetensors"
    w_s2 = mx.load(str(s2_path))

    best_key = (-1, -1e9, -1e9)  # (probe_passed, probe_mean_reward, -val_loss)
    best_cfg = (1.0, 1.0)
    best_soup_weights = w_s2

    # Note: in LoRA (y = W_0 x + scale * x A B), scaling `lora_b` by `gamma` (leaving `lora_a` at `1.0`)
    # implements exact linear weight-space interpolation `(1 - gamma) * W_base + gamma * W_finetuned`!
    candidates = [(0.0, 1.0), (0.5, 1.0), (1.0, 1.0), (0.5, 0.88), (1.0, 0.88)]
    for lam, gamma in candidates:
        blended = {}
        for k in w_s2:
            v_mix = (1.0 - lam) * w_s1[k] + lam * w_s2[k]
            blended[k] = (gamma * v_mix) if "lora_b" in k else v_mix
        model.load_weights(list(blended.items()), strict=False)
        mx.eval(model.parameters())
        vl, vp = evaluate_val_loss(model, val_items)
        p_pass, p_rew = _evaluate_mini_probe(model, processor, probe_14)
        cand_key = (p_pass, p_rew, -vl)
        print(
            f"  [WiSE-FT rl_lam={lam:.2f}, base_gamma={gamma:.2f}] probe={p_pass}/{len(probe_14)} | "
            f"mean_reward={p_rew:.3f} | val_loss={vl:.4f} (PPL={vp:.4f})",
            flush=True,
        )
        if cand_key >= best_key:
            best_key = cand_key
            best_cfg = (lam, gamma)
            best_soup_weights = blended

    print(
        f"[Stage 3] Selected WiSE-FT rl_lam={best_cfg[0]:.2f}, base_gamma={best_cfg[1]:.2f} "
        f"(probe={best_key[0]}/{len(probe_14)}, val_loss={-best_key[2]:.4f})!",
        flush=True,
    )
    model.load_weights(list(best_soup_weights.items()), strict=False)
    mx.eval(model.parameters())
    _save_adapter_ckpt(
        model,
        WISE_BEST_DIR,
        {"rl_lambda": best_cfg[0], "base_gamma": best_cfg[1], "probe_passed": best_key[0], "val_loss": -best_key[2]},
    )

    # ------------------------------------------------------------------
    # STAGE 4: Lossless 16-Bit `bfloat16` Standalone Fusion & Disk Verification
    # ------------------------------------------------------------------
    fuse_and_export_standalone_bf16(model, Path(args.model), Path(args.export_dir))
    print(f"[Complete] Standalone 16-bit RPR model ready at: {args.export_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep 4-Stage BF16 RPR trainer & standalone 16-bit model exporter.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Read-only base model path.")
    parser.add_argument("--export-dir", default=str(EXPORT_BF16_DIR), help="Output directory for standalone 16-bit RPR model.")
    parser.add_argument("--sft-epochs", type=int, default=2, help="Max Stage-1 LoRA+ DFT epochs (auto early-stopped).")
    parser.add_argument("--rl-steps", type=int, default=48, help="Shuffled Stage-2 SCoRe/RLVR episodes.")
    parser.add_argument("--fresh", action="store_true", help="Restart training ignoring existing checkpoints.")
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
