"""Comprehensive RPR v12 DEEP-THINKER (`3.22B` `40L` Pure 16-bit `bfloat16`, `256k` Context, `8`-Loop Dual Huginn + ThoughtAnchorTemp) Evaluation Suite.

Evaluates `checkpoints/LFM3-VL-3.2B-RPR-V12-DEEP-BF16` (or `--model-dir`) across ALL 5 strictly held-out benchmarks (`140` unseen tasks, 0 contamination):
  1. `heldout_v5_rsi_56.jsonl`            (`56` 8-pillar tasks: `base_2b` -> `v9` -> `v10.1` -> `v11` -> `v12` & `v12+RPR`)
  2. `heldout_zero_prior_24.jsonl`        (`24` unseen Zero-Prior Formal, Weight-Mod & RSI, Research & OOD Discovery)
  3. `heldout_v10_apex_36.jsonl`          (`36` unseen Complex Logic/SAT, Research & Citations, Self-Discover OOD, DGM/SVF RSI)
  4. `heldout_long_256k_needles_12.jsonl` (`12` ultra-long `16k..64k` Multi-Hop Needle + Erratum + `[DOC-XXXX]` Citation tasks)
  5. `benchmark_100.jsonl --split test --limit 12` (`direct,direct_tir,rpr`)
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
from typing import Any

import mlx.core as mx

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from rpr_live_controller import run_live_rpr_pyramid_turn
from training.build_v9_rsi_zero_prior_cline_corpus import V9_CORPUS_DIR
from training.build_v10_apex_corpus import V10_CORPUS_DIR
from training.eval_rpr_bf16_suite import _summarize_broad
from training.eval_rpr_v5_rsi_suite import PILLAR_ORDER, V5_EVAL_PATH
from training.eval_v7_and_long_context import OUT_56_DIR
from training.eval_v9_comprehensive_suite import ZP24_PILLARS
from training.eval_v10_apex_suite import APEX36_PILLARS, LONG12_PILLARS
from training.eval_v11_omega_suite import _eval_long_context_v11
from training.run_v10_3_2b_apex_pipeline import EXPORT_V10_APEX
from training.run_v12_3_2b_deep_thinker_pipeline import STAGING_V12
from training.train_and_fuse_rpr_bf16 import _apply_template_inference_aligned
from training.verifiable_rewards import score_response
from v8_arch import attach_v8_extensions


def _load_cached_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _unload_8991() -> None:
    try:
        urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8991/v1/admin/unload", method="POST"), timeout=45).read()
    except Exception:
        pass


def _eval_long_context_v12(
    tasks: list[dict[str, Any]],
    *,
    label: str,
    model_path: str,
    cache_path: Path,
    max_tokens: int = 1600,
    use_rpr_controller: bool = False,
    fresh: bool = False,
) -> list[dict[str, Any]]:
    """Evaluate `16k..64k` Multi-Hop Needle + Erratum + `[DOC-XXXX]` Citation tasks (`prefill_step_size=1024`, deep `<think>`)."""
    import mlx_vlm

    cached: dict[str, dict[str, Any]] = {}
    if fresh and cache_path.exists():
        cache_path.unlink()
    elif not fresh and cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)
                cached[obj["id"]] = obj

    if all(t["id"] in cached for t in tasks):
        print(f"[long_ctx_v12] Loaded all {len(tasks)} cached runs for {label} from {cache_path}", flush=True)
        return [cached[t["id"]] for t in tasks]

    _unload_8991()
    print(f"[long_ctx_v12] Loading {label} (LongRoPE2YaRN + Dual 4-Loop Huginn + FlashSDPA<=64k) from {model_path}...", flush=True)
    model, processor = mlx_vlm.load(model_path)
    if (Path(model_path) / "v8_arch_ext.safetensors").exists():
        attach_v8_extensions(model, sidecar_dir=model_path, enable_moba=True, moba_threshold=65536)
    model.eval()
    tok = processor.tokenizer
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    def _gen_cb(msgs: list[dict[str, Any]], mt: int, en_th: bool) -> str:
        ps_i = _apply_template_inference_aligned(processor, model.config, msgs, num_images=0, enable_thinking=en_th)
        kw: dict[str, Any] = {
            "max_tokens": mt,
            "temperature": 0.0,
            "repetition_penalty": 1.02,
            "prefill_step_size": 1024,
            "enable_thinking": en_th,
            "verbose": False,
        }
        if en_th and len(ps_i) >= 20000:
            kw["thinking_budget"] = 960
        ri = mlx_vlm.generate(model, processor, ps_i, **kw)
        tx = ri.text or ""
        if en_th and ps_i.endswith("<think>") and not tx.lstrip().startswith("<think>"):
            tx = "<think>" + tx
        gc.collect()
        mx.clear_cache()
        return tx

    with cache_path.open("a", encoding="utf-8") as fh:
        for idx, task in enumerate(tasks, start=1):
            tid = task["id"]
            if tid in cached:
                results.append(cached[tid])
                continue

            _unload_8991()
            p_s = _apply_template_inference_aligned(processor, model.config, task["messages"], num_images=0, enable_thinking=True)
            prompt_tok_len = len(tok.encode(p_s, add_special_tokens=False))
            t0 = time.perf_counter()
            if use_rpr_controller:
                rpr_out = run_live_rpr_pyramid_turn(task["messages"], _gen_cb, max_tokens=max_tokens, has_images=False)
                raw_txt = rpr_out["text"]
                audit_note = rpr_out["audit_note"]
            else:
                raw_txt = _gen_cb(task["messages"], max_tokens, True)
                audit_note = "standalone"
            dt = round(time.perf_counter() - t0, 3)

            prompt_txt = "\n".join(str(m.get("content", "")) for m in task["messages"])
            sc = score_response(
                raw_txt,
                prompt_text=prompt_txt[:2500],
                kind=task.get("kind", "citation_grounded"),
                verifier=task["verifier"],
                tokenizer=tok,
                max_think_tokens=2000,
            )
            rec = {
                "id": tid,
                "pillar": task["pillar"],
                "kind": task.get("kind", "citation_grounded"),
                "prompt_tokens": prompt_tok_len,
                "label": label,
                "score": sc,
                "correct": bool(sc["correct"]),
                "reward": sc["reward"],
                "closed_think": bool(sc["closed_think"]),
                "think_tokens": sc["think_tokens"],
                "echo_overlap": sc["echo_overlap"],
                "tool_leaked": bool(sc["tool_leaked"]),
                "abstain_task": False,
                "audit_note": audit_note,
                "wall_clock_sec": dt,
                "raw_text": raw_txt,
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            results.append(rec)
            print(
                f"  [{label:16s} {idx:2d}/{len(tasks)} | prompt={prompt_tok_len:5d}t] {tid:16s} | "
                f"pass={rec['correct']} | think={sc['think_tokens']:3d} | closed={sc['closed_think']} | {audit_note} | {dt:.2f}s",
                flush=True,
            )

    del model, processor
    gc.collect()
    mx.clear_cache()
    return results


def _eval_standalone_deep_v12(
    tasks: list[dict[str, Any]],
    *,
    label: str,
    model_path: str,
    cache_path: Path,
    max_tokens: int = 3200,
    fresh: bool = False,
) -> list[dict[str, Any]]:
    import mlx_vlm

    cached: dict[str, dict[str, Any]] = {}
    if fresh and cache_path.exists():
        cache_path.unlink()
    elif not fresh and cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)
                cached[obj["id"]] = obj

    if all(t["id"] in cached for t in tasks):
        print(f"[deep_eval_v12] Loaded all {len(tasks)} cached runs for {label} from {cache_path}", flush=True)
        return [cached[t["id"]] for t in tasks]

    _unload_8991()
    print(f"[deep_eval_v12] Loading {label} from {model_path}...", flush=True)
    model, processor = mlx_vlm.load(model_path)
    if (Path(model_path) / "v8_arch_ext.safetensors").exists():
        attach_v8_extensions(model, sidecar_dir=model_path, enable_moba=False)
    model.eval()
    tok = processor.tokenizer
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    with cache_path.open("a", encoding="utf-8") as fh:
        for idx, task in enumerate(tasks, start=1):
            tid = task["id"]
            if tid in cached:
                results.append(cached[tid])
                continue
            imgs = [str(p) for p in (task.get("images") or []) if p] or None
            prompt_str = _apply_template_inference_aligned(
                processor, model.config, task["messages"], num_images=len(imgs) if imgs else 0, enable_thinking=True
            )
            t0 = time.perf_counter()
            res = mlx_vlm.generate(
                model,
                processor,
                prompt_str,
                image=imgs,
                max_tokens=max_tokens,
                temperature=0.0,
                repetition_penalty=1.03,
                verbose=False,
            )
            dt = round(time.perf_counter() - t0, 4)
            raw_txt = res.text or ""
            if prompt_str.endswith("<think>") and not raw_txt.lstrip().startswith("<think>"):
                raw_txt = "<think>" + raw_txt
            prompt_txt = "\n".join(str(m.get("content", "")) for m in task["messages"])
            sc = score_response(
                raw_txt,
                prompt_text=prompt_txt,
                kind=task.get("kind", task["pillar"]),
                verifier=task["verifier"],
                tokenizer=tok,
                max_think_tokens=3000,
            )
            row = {
                "id": tid,
                "pillar": task["pillar"],
                "kind": task.get("kind", task["pillar"]),
                "label": label,
                "raw_text": raw_txt,
                "score": sc,
                "wall_clock_sec": dt,
            }
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            results.append(row)
            mx.clear_cache()
            print(
                f"[{label:14s}] [{idx:2d}/{len(tasks)}] {tid:18s} ({task['pillar']:27s}) | "
                f"ok={sc['correct']!s:5s} | think={sc['think_tokens']:3d} | closed={sc['closed_think']}",
                flush=True,
            )

    del model, processor
    gc.collect()
    mx.clear_cache()
    return results


def _eval_rpr_pyramid_deep_v12(
    tasks: list[dict[str, Any]],
    *,
    label: str,
    model_path: str,
    cache_path: Path,
    max_tokens: int = 3200,
    fresh: bool = False,
) -> list[dict[str, Any]]:
    import mlx_vlm

    cached: dict[str, dict[str, Any]] = {}
    if fresh and cache_path.exists():
        cache_path.unlink()
    elif not fresh and cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)
                cached[obj["id"]] = obj

    if all(t["id"] in cached for t in tasks):
        print(f"[rpr_deep_v12] Loaded all {len(tasks)} cached runs for {label} from {cache_path}", flush=True)
        return [cached[t["id"]] for t in tasks]

    _unload_8991()
    print(f"[rpr_deep_v12] Loading {label} (pure 16-bit bfloat16 + Zero-Leakage Deep RPR Pyramid) from {model_path}...", flush=True)
    model, processor = mlx_vlm.load(model_path)
    if (Path(model_path) / "v8_arch_ext.safetensors").exists():
        attach_v8_extensions(model, sidecar_dir=model_path, enable_moba=False)
    model.eval()
    tok = processor.tokenizer
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    with cache_path.open("a", encoding="utf-8") as fh:
        for idx, task in enumerate(tasks, start=1):
            tid = task["id"]
            if tid in cached:
                results.append(cached[tid])
                continue
            imgs = [str(p) for p in (task.get("images") or []) if p] or None

            def _gen_cb(msgs: list[dict[str, Any]], max_toks: int, enable_think: bool) -> str:
                use_imgs = imgs if (imgs and msgs is task["messages"]) else None
                p_s = _apply_template_inference_aligned(
                    processor,
                    model.config,
                    msgs,
                    num_images=len(use_imgs) if use_imgs else 0,
                    enable_thinking=enable_think,
                )
                r = mlx_vlm.generate(
                    model,
                    processor,
                    p_s,
                    image=use_imgs,
                    max_tokens=max_toks,
                    temperature=0.0,
                    repetition_penalty=1.03,
                    verbose=False,
                )
                txt = r.text or ""
                if enable_think and p_s.endswith("<think>") and not txt.lstrip().startswith("<think>"):
                    txt = "<think>" + txt
                return txt

            t0 = time.perf_counter()
            rpr_out = run_live_rpr_pyramid_turn(
                task["messages"],
                _gen_cb,
                max_tokens=max_tokens,
                has_images=bool(imgs),
            )
            dt = round(time.perf_counter() - t0, 4)
            raw_txt = rpr_out["text"]
            prompt_txt = "\n".join(str(m.get("content", "")) for m in task["messages"])
            sc = score_response(
                raw_txt,
                prompt_text=prompt_txt,
                kind=task.get("kind", task["pillar"]),
                verifier=task["verifier"],
                tokenizer=tok,
                max_think_tokens=3000,
            )
            rec = {
                "id": tid,
                "pillar": task["pillar"],
                "kind": task.get("kind", task["pillar"]),
                "label": label,
                "score": sc,
                "correct": bool(sc["correct"]),
                "reward": sc["reward"],
                "closed_think": bool(sc["closed_think"]),
                "think_tokens": sc["think_tokens"],
                "echo_overlap": sc["echo_overlap"],
                "tool_leaked": bool(sc["tool_leaked"]),
                "abstain_task": bool(task["verifier"].get("abstain")),
                "wall_clock_sec": dt,
                "rpr_passes": rpr_out["passes"],
                "rpr_repaired": rpr_out["repaired"],
                "rpr_audit_note": rpr_out["audit_note"],
                "raw_text": raw_txt,
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            results.append(rec)
            mx.clear_cache()
            print(
                f"  [{label:16s} {idx:2d}/{len(tasks)}] {tid:16s} ({task['pillar']:24s}) | "
                f"pass={rec['correct']} | think={sc['think_tokens']:3d} | passes={rpr_out['passes']} ({rpr_out['audit_note']}) | {dt:.2f}s",
                flush=True,
            )

    del model, processor
    gc.collect()
    mx.clear_cache()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RPR v12 DEEP-THINKER across all 5 strictly held-out suites.")
    parser.add_argument("--model-dir", type=str, default="")
    parser.add_argument("--fresh-v12", action="store_true")
    parser.add_argument("--skip-bench12", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model_dir) if args.model_dir else (STAGING_V12 if (STAGING_V12 / "model.safetensors").exists() else EXPORT_V10_APEX)
    print(f"[eval_v12] Target checkpoint: {model_dir}", flush=True)

    out_56 = ROOT_DIR / "results" / "rpr_v12_3_2b_56"
    out_56.mkdir(parents=True, exist_ok=True)
    out_v11_56 = ROOT_DIR / "results" / "rpr_v11_3_2b_56"
    out_v10_56 = ROOT_DIR / "results" / "rpr_v10_3_2b_56"
    out_v9_56 = ROOT_DIR / "results" / "rpr_v9_2_9b_56"
    out_v8_56 = ROOT_DIR / "results" / "rpr_v8_2_9b_56"
    out_zp = ROOT_DIR / "results" / "zero_prior_ood_24"
    out_36 = ROOT_DIR / "results" / "v10_apex_36"
    out_long = ROOT_DIR / "results" / "v10_long_256k_needles_12"

    if args.fresh_v12:
        for stale_p in (
            out_56 / "rpr_v12_3_2b_preds.jsonl",
            out_56 / "rpr_v12_3_2b_rpr_preds.jsonl",
            out_zp / "rpr_v12_3_2b_zp24.jsonl",
            out_zp / "rpr_v12_3_2b_rpr_zp24.jsonl",
            out_36 / "rpr_v12_3_2b_apex36.jsonl",
            out_36 / "rpr_v12_3_2b_rpr_apex36.jsonl",
            out_long / "rpr_v12_3_2b_long12.jsonl",
            out_long / "rpr_v12_3_2b_rpr_long12.jsonl",
        ):
            if stale_p.exists():
                stale_p.unlink()

    mx.set_wired_limit(0)
    mx.set_cache_limit(192 * 1024 * 1024)

    # 1. Held-Out 56-Task 8-Pillar Benchmark
    tasks_56 = _load_cached_jsonl(V5_EVAL_PATH)
    sum_56: dict[str, Any] = {}
    for lbl, cpath in [
        ("base_2b", OUT_56_DIR / "base_2b_preds.jsonl"),
        ("rpr_v8_2.9b", out_v8_56 / "rpr_v8_2_9b_preds.jsonl"),
        ("rpr_v9_2.9b", out_v9_56 / "rpr_v9_2_9b_preds.jsonl"),
        ("rpr_v9_2.9b+RPR", out_v9_56 / "rpr_v9_2_9b_rpr_preds.jsonl"),
        ("rpr_v10.1_3.2b", out_v10_56 / "rpr_v10_3_2b_preds.jsonl"),
        ("rpr_v11_3.2b", out_v11_56 / "rpr_v11_3_2b_preds.jsonl"),
        ("rpr_v11_3.2b+RPR", out_v11_56 / "rpr_v11_3_2b_rpr_preds.jsonl"),
    ]:
        if cpath.exists():
            sum_56[lbl] = _summarize_broad(_load_cached_jsonl(cpath), lbl)

    v12_56 = _eval_standalone_deep_v12(
        tasks_56, label="rpr_v12_3.2b", model_path=str(model_dir), cache_path=out_56 / "rpr_v12_3_2b_preds.jsonl", max_tokens=3200
    )
    sum_56["rpr_v12_3.2b"] = _summarize_broad(v12_56, "rpr_v12_3.2b")

    v12_rpr_56 = _eval_rpr_pyramid_deep_v12(
        tasks_56, label="rpr_v12_3.2b+RPR", model_path=str(model_dir), cache_path=out_56 / "rpr_v12_3_2b_rpr_preds.jsonl", max_tokens=3200
    )
    sum_56["rpr_v12_3.2b+RPR"] = _summarize_broad(v12_rpr_56, "rpr_v12_3.2b+RPR")
    (out_56 / "summary.json").write_text(json.dumps(sum_56, indent=2) + "\n", encoding="utf-8")

    hdr56 = "| Model | Overall (`56`) | " + " | ".join(lbl for _, lbl in PILLAR_ORDER) + " | Closed `</think>` | Mean Think | Abstain |"
    sep56 = "| --- | ---: | " + " | ".join("---:" for _ in PILLAR_ORDER) + " | ---: | ---: | ---: |"
    md56 = ["# 56-Task 8-Pillar Generational Benchmark (`Base 2B -> v8 -> v9 -> v10.1 -> v11 -> v12 DEEP-THINKER`)", "", hdr56, sep56]
    for lbl, s in sum_56.items():
        p_cells = " | ".join(f"{s['pillars'][pk]['pass_rate']*100:.1f}%" for pk, _ in PILLAR_ORDER)
        abs_s = "-" if s["tool_abstention_rate"] is None else f"{s['tool_abstention_rate']*100:.1f}%"
        md56.append(
            f"| `{lbl}` | **{s['overall_pass_rate']*100:.1f}%** | {p_cells} | "
            f"{s['closed_think_rate']*100:.1f}% | {s['mean_think_tokens']} | {abs_s} |"
        )
    (out_56 / "report.md").write_text("\n".join(md56) + "\n", encoding="utf-8")
    print("\n".join(md56), flush=True)

    # 2. Held-Out 24-Task Zero-Prior Formal, Weight-Mod/RSI, & Research/OOD Discovery
    tasks_24 = _load_cached_jsonl(V9_CORPUS_DIR / "heldout_zero_prior_24.jsonl")
    sum_zp: dict[str, Any] = {}
    for lbl, cpath in [
        ("base_2b", out_zp / "base_2b_zp24.jsonl"),
        ("rpr_v9_2.9b+RPR", out_zp / "rpr_v9_2_9b_rpr_zp24.jsonl"),
        ("rpr_v11_3.2b", out_zp / "rpr_v11_3_2b_zp24.jsonl"),
        ("rpr_v11_3.2b+RPR", out_zp / "rpr_v11_3_2b_rpr_zp24.jsonl"),
    ]:
        if cpath.exists():
            sum_zp[lbl] = _summarize_broad(_load_cached_jsonl(cpath), lbl)

    v12_zp = _eval_standalone_deep_v12(
        tasks_24, label="rpr_v12_3.2b", model_path=str(model_dir), cache_path=out_zp / "rpr_v12_3_2b_zp24.jsonl", max_tokens=3200
    )
    sum_zp["rpr_v12_3.2b"] = _summarize_broad(v12_zp, "rpr_v12_3.2b")

    v12_rpr_zp = _eval_rpr_pyramid_deep_v12(
        tasks_24, label="rpr_v12_3.2b+RPR", model_path=str(model_dir), cache_path=out_zp / "rpr_v12_3_2b_rpr_zp24.jsonl", max_tokens=3200
    )
    sum_zp["rpr_v12_3.2b+RPR"] = _summarize_broad(v12_rpr_zp, "rpr_v12_3.2b+RPR")
    (out_zp / "summary_v12.json").write_text(json.dumps(sum_zp, indent=2) + "\n", encoding="utf-8")

    hdr24 = "| Model | Overall (`24`) | " + " | ".join(lbl for _, lbl in ZP24_PILLARS) + " | Closed `</think>` | Mean Think |"
    sep24 = "| --- | ---: | " + " | ".join("---:" for _ in ZP24_PILLARS) + " | ---: | ---: |"
    md24 = ["# 24-Task Zero-Prior-Knowledge (`OOD`) Formal Systems, Weight-Modification & Discovery Benchmark", "", hdr24, sep24]
    for lbl, s in sum_zp.items():
        p_cells = " | ".join(f"{s['pillars'][pk]['pass_rate']*100:.1f}%" for pk, _ in ZP24_PILLARS)
        md24.append(f"| `{lbl}` | **{s['overall_pass_rate']*100:.1f}%** | {p_cells} | {s['closed_think_rate']*100:.1f}% | {s['mean_think_tokens']} |")
    (out_zp / "report_v12.md").write_text("\n".join(md24) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md24), flush=True)

    # 3. Held-Out 36-Task Frontier APEX/OMEGA Suite
    tasks_36 = _load_cached_jsonl(V10_CORPUS_DIR / "heldout_v10_apex_36.jsonl")
    sum_36: dict[str, Any] = {}
    for lbl, cpath in [
        ("base_2b", out_36 / "base_2b_apex36.jsonl"),
        ("rpr_v9_2.9b", out_36 / "rpr_v9_2_9b_apex36.jsonl"),
        ("rpr_v11_3.2b", out_36 / "rpr_v11_3_2b_apex36.jsonl"),
        ("rpr_v11_3.2b+RPR", out_36 / "rpr_v11_3_2b_rpr_apex36.jsonl"),
    ]:
        if cpath.exists():
            sum_36[lbl] = _summarize_broad(_load_cached_jsonl(cpath), lbl)

    v12_36 = _eval_standalone_deep_v12(
        tasks_36, label="rpr_v12_3.2b", model_path=str(model_dir), cache_path=out_36 / "rpr_v12_3_2b_apex36.jsonl", max_tokens=3200
    )
    sum_36["rpr_v12_3.2b"] = _summarize_broad(v12_36, "rpr_v12_3.2b")

    v12_rpr_36 = _eval_rpr_pyramid_deep_v12(
        tasks_36, label="rpr_v12_3.2b+RPR", model_path=str(model_dir), cache_path=out_36 / "rpr_v12_3_2b_rpr_apex36.jsonl", max_tokens=3200
    )
    sum_36["rpr_v12_3.2b+RPR"] = _summarize_broad(v12_rpr_36, "rpr_v12_3.2b+RPR")
    (out_36 / "summary_v12.json").write_text(json.dumps(sum_36, indent=2) + "\n", encoding="utf-8")

    hdr36 = "| Model | Overall (`36`) | " + " | ".join(lbl for _, lbl in APEX36_PILLARS) + " | Closed `</think>` | Mean Think |"
    sep36 = "| --- | ---: | " + " | ".join("---:" for _ in APEX36_PILLARS) + " | ---: | ---: |"
    md36 = ["# 36-Task Frontier APEX/DEEP-THINKER Benchmark (Complex Logic/SAT, Citations, Self-Discover OOD, DGM/SVF RSI)", "", hdr36, sep36]
    for lbl, s in sum_36.items():
        p_cells = " | ".join(f"{s['pillars'][pk]['pass_rate']*100:.1f}%" for pk, _ in APEX36_PILLARS)
        md36.append(f"| `{lbl}` | **{s['overall_pass_rate']*100:.1f}%** | {p_cells} | {s['closed_think_rate']*100:.1f}% | {s['mean_think_tokens']} |")
    (out_36 / "report_v12.md").write_text("\n".join(md36) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md36), flush=True)

    # 4. Held-Out 12-Task Ultra-Long `16k..64k` Multi-Hop Needles + Errata
    tasks_long = _load_cached_jsonl(V10_CORPUS_DIR / "heldout_long_256k_needles_12.jsonl")
    sum_long: dict[str, Any] = {}
    for lbl, cpath in [
        ("rpr_v9_2.9b", out_long / "rpr_v9_2_9b_long12.jsonl"),
        ("rpr_v11_3.2b", out_long / "rpr_v11_3_2b_long12.jsonl"),
        ("rpr_v11_3.2b+RPR", out_long / "rpr_v11_3_2b_rpr_long12.jsonl"),
    ]:
        if cpath.exists():
            sum_long[lbl] = _summarize_broad(_load_cached_jsonl(cpath), lbl)

    v12_long = _eval_long_context_v12(
        tasks_long, label="rpr_v12_3.2b", model_path=str(model_dir), cache_path=out_long / "rpr_v12_3_2b_long12.jsonl", max_tokens=1600, use_rpr_controller=False
    )
    sum_long["rpr_v12_3.2b"] = _summarize_broad(v12_long, "rpr_v12_3.2b")

    v12_rpr_long = _eval_long_context_v12(
        tasks_long, label="rpr_v12_3.2b+RPR", model_path=str(model_dir), cache_path=out_long / "rpr_v12_3_2b_rpr_long12.jsonl", max_tokens=1600, use_rpr_controller=True
    )
    sum_long["rpr_v12_3.2b+RPR"] = _summarize_broad(v12_rpr_long, "rpr_v12_3.2b+RPR")
    (out_long / "summary_v12.json").write_text(json.dumps(sum_long, indent=2) + "\n", encoding="utf-8")

    hdrl = "| Model | Overall (`12`) | " + " | ".join(lbl for _, lbl in LONG12_PILLARS) + " | Closed `</think>` | Mean Think |"
    sepl = "| --- | ---: | " + " | ".join("---:" for _ in LONG12_PILLARS) + " | ---: | ---: |"
    mdl = ["# 12-Task Ultra-Long Context (`16k..64k` Tokens) Multi-Hop Needle + Erratum + `[DOC-XXXX]` Citation Benchmark", "", hdrl, sepl]
    for lbl, s in sum_long.items():
        p_cells = " | ".join(f"{s['pillars'][pk]['pass_rate']*100:.1f}%" for pk, _ in LONG12_PILLARS)
        mdl.append(f"| `{lbl}` | **{s['overall_pass_rate']*100:.1f}%** | {p_cells} | {s['closed_think_rate']*100:.1f}% | {s['mean_think_tokens']} |")
    (out_long / "report_v12.md").write_text("\n".join(mdl) + "\n", encoding="utf-8")
    print("\n" + "\n".join(mdl), flush=True)

    # 5. Held-Out `benchmark_100` Test Slice (`limit=12`)
    if not args.skip_bench12:
        _unload_8991()
        out_b12 = ROOT_DIR / "results" / "benchmark_test_12_rpr_v12_3_2b"
        subprocess.run(
            [
                sys.executable,
                str(ROOT_DIR / "evaluate.py"),
                "--dataset",
                str(ROOT_DIR / "data" / "benchmark_100.jsonl"),
                "--split",
                "test",
                "--limit",
                "12",
                "--model",
                str(model_dir),
                "--modes",
                "direct,direct_tir,rpr",
                "--per-call-max-tokens",
                "2816",
                "--token-budget",
                "49152",
                "--output-dir",
                str(out_b12),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
