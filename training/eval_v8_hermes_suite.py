"""Comprehensive RPR v8 (`2.90B` `Engram` + `AttnRes` + `MoBA` + `MuonClip` + `Hermes`) Verification Suite.

Evaluates `checkpoints/LFM3-VL-2.9B-RPR-V8-HERMES-BF16` across:
  1. `heldout_v5_rsi_56` (`56` unseen 8-pillar tasks: both single-shot `rpr_v8_2.9b` and zero-leakage `rpr_v8_2.9b+RPR`)
  2. `long_context_eval_16` (`16` unseen multi-turn sessions up to `39,245` prompt tokens with `AttnRes` + `MoBA` active!)
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import sys
import time
from typing import Any

import mlx.core as mx

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from model import DEFAULT_MODEL_PATH
from training.build_v7_master_and_longctx_corpus import V7_CORPUS_DIR
from training.eval_rpr_bf16_suite import _eval_standalone_model, _summarize_broad
from training.eval_rpr_v5_rsi_suite import PILLAR_ORDER, V5_EVAL_PATH
from training.eval_v7_and_long_context import (
    OUT_56_DIR,
    OUT_LC_DIR,
    _eval_standalone_model_rpr_verified,
)
from training.run_v6_32l_mtp_seal_pipeline import EXPORT_32L_REPAIRED, EXPORT_FINAL_SEAL
from training.run_v7_36l_zero_degrade_pipeline import EXPORT_36L_BASE, EXPORT_36L_SEAL
from training.run_v8_2_9b_hermes_pipeline import EXPORT_V8_HERMES
from training.train_and_fuse_rpr_bf16 import EXPORT_BF16_DIR, _apply_template_inference_aligned
from training.upcycle_and_train_rpr_v5_bf16 import EXPORT_V5_DIR
from training.verifiable_rewards import score_response
from v8_arch import attach_v8_extensions
from verifiers import parse_candidate_answer


def _eval_v8_rpr_verified_56(tasks: list[dict[str, Any]], cache_path: Path, fresh: bool = False) -> list[dict[str, Any]]:
    import mlx_vlm

    # Ensure `_eval_standalone_model_rpr_verified` also attaches `v8` extensions when loading `EXPORT_V8_HERMES`!
    orig_load = mlx_vlm.load

    def _patched_load(path_str: str, *a: Any, **kw: Any):
        m, p = orig_load(path_str, *a, **kw)
        if (Path(path_str) / "v8_arch_ext.safetensors").exists():
            attach_v8_extensions(m, sidecar_dir=path_str, enable_moba=True)
        return m, p

    mlx_vlm.load = _patched_load  # type: ignore[assignment]
    try:
        return _eval_standalone_model_rpr_verified(
            tasks,
            label="rpr_v8_2.9b+RPR",
            model_path=str(EXPORT_V8_HERMES),
            cache_path=cache_path,
            max_tokens=576,
            fresh=fresh,
        )
    finally:
        mlx_vlm.load = orig_load  # type: ignore[assignment]


def _eval_long_context_v8(tasks: list[dict[str, Any]], cache_path: Path, fresh: bool = False) -> dict[str, Any]:
    import mlx_vlm

    cached: dict[str, dict[str, Any]] = {}
    if not fresh and cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)
                cached[obj["id"]] = obj

    if len(cached) < len(tasks):
        print(f"[long_ctx_v8] Loading rpr_v8_2.9b_moba ({EXPORT_V8_HERMES}) with Engram + AttnRes + MoBA...", flush=True)
        mx.reset_peak_memory()
        vlm, proc = mlx_vlm.load(str(EXPORT_V8_HERMES))
        attach_v8_extensions(vlm, sidecar_dir=EXPORT_V8_HERMES, enable_moba=True, moba_threshold=4096)
        tok = proc.tokenizer

        for idx, t in enumerate(tasks, start=1):
            if t["id"] in cached:
                continue
            p_str = _apply_template_inference_aligned(proc, vlm.config, t["messages"], num_images=0, enable_thinking=True)
            prompt_toks = len(tok.encode(p_str, add_special_tokens=False))
            t0 = time.perf_counter()
            res = mlx_vlm.generate(vlm, proc, p_str, max_tokens=384, temperature=0.0, repetition_penalty=1.05, verbose=False)
            dt = round(time.perf_counter() - t0, 3)
            raw = res.text or ""
            full_raw = ("<think>" + raw) if (p_str.endswith("<think>") and not raw.startswith("<think>")) else raw
            p_txt = str(t["messages"][1]["content"]) + "\n" + str(t["messages"][-1]["content"])
            sc = score_response(full_raw, prompt_text=p_txt, kind="math_opt", verifier=t["verifier"], tokenizer=tok, max_think_tokens=320)
            pred_ans = parse_candidate_answer(full_raw) or {}
            pri_ok = pred_ans.get("primary") == t["verifier"]["primary"]
            pres_ok = pred_ans.get("preserved") == t["verifier"]["preserved"]
            peak_gb = round(mx.get_peak_memory() / (1024**3), 3)

            rec = {
                "id": t["id"],
                "target_tokens": t["target_tokens"],
                "actual_prompt_tokens": prompt_toks,
                "category": t["category"],
                "exact_correct": bool(sc["correct"]),
                "primary_correct": bool(pri_ok),
                "preserved_correct": bool(pres_ok),
                "closed_think": bool(sc["closed_think"]),
                "wall_clock_sec": dt,
                "peak_memory_gb": peak_gb,
                "pred": pred_ans,
                "expected": {"primary": t["verifier"]["primary"], "preserved": t["verifier"]["preserved"]},
            }
            cached[t["id"]] = rec
            all_ordered = [cached[x["id"]] for x in tasks if x["id"] in cached]
            tmp_p = cache_path.with_suffix(".tmp")
            tmp_p.write_text("\n".join(json.dumps(r) for r in all_ordered) + "\n", encoding="utf-8")
            tmp_p.replace(cache_path)
            mx.clear_cache()
            print(
                f"  [rpr_v8_2.9b_moba LongCtx {idx:2d}/{len(tasks)}] {t['id']} ({prompt_toks:5d} toks, {t['category']:17s}) | "
                f"exact={rec['exact_correct']} (pri={pri_ok}, pres={pres_ok}) | {dt:5.2f}s | peak={peak_gb:.2f} GiB",
                flush=True,
            )
        del vlm, proc
        gc.collect()
        mx.clear_cache()

    rows = [cached[t["id"]] for t in tasks]
    by_bucket: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_bucket[int(r["target_tokens"])].append(r)
        by_cat[str(r["category"])].append(r)

    def _rate(lst: list[dict[str, Any]], key: str = "exact_correct") -> float:
        return round(sum(1 for x in lst if x[key]) / max(1, len(lst)), 4)

    return {
        "label": "rpr_v8_2.9b_moba",
        "count": len(rows),
        "overall_exact_acc": _rate(rows, "exact_correct"),
        "overall_primary_acc": _rate(rows, "primary_correct"),
        "overall_preserved_acc": _rate(rows, "preserved_correct"),
        "closed_think_rate": _rate(rows, "closed_think"),
        "premise_change_acc": _rate(by_cat["premise_change"]),
        "irrelevant_update_acc": _rate(by_cat["irrelevant_update"]),
        "buckets": {
            f"{k//1024}k": {
                "mean_prompt_tokens": round(sum(x["actual_prompt_tokens"] for x in v) / len(v), 1),
                "exact_acc": _rate(v, "exact_correct"),
                "preserved_acc": _rate(v, "preserved_correct"),
                "mean_sec": round(sum(x["wall_clock_sec"] for x in v) / len(v), 2),
                "max_peak_gb": max(x["peak_memory_gb"] for x in v),
            }
            for k, v in sorted(by_bucket.items())
        },
        "max_prompt_tokens_tested": max(x["actual_prompt_tokens"] for x in rows),
        "max_peak_memory_gb": max(x["peak_memory_gb"] for x in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RPR v8 (2.90B Engram+AttnRes+MoBA+Hermes) across 56-task & 39k long-context suites.")
    parser.add_argument("--fresh-v8", action="store_true")
    args = parser.parse_args()

    mx.set_wired_limit(0)
    mx.set_cache_limit(256 * 1024 * 1024)

    out_v8_56 = ROOT_DIR / "results" / "rpr_v8_2_9b_56"
    out_v8_56.mkdir(parents=True, exist_ok=True)
    tasks_56 = [json.loads(l) for l in V5_EVAL_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]

    # Reuse cached generational baselines from `OUT_56_DIR`
    base_specs = [
        ("base_2b", DEFAULT_MODEL_PATH, OUT_56_DIR / "base_2b_preds.jsonl"),
        ("rpr_v4_2b", str(EXPORT_BF16_DIR), OUT_56_DIR / "rpr_v4_2b_preds.jsonl"),
        ("rpr_rsi_2.2b", str(EXPORT_V5_DIR), OUT_56_DIR / "rpr_rsi_2_2b_preds.jsonl"),
        ("rpr_seal_2.4b", str(EXPORT_FINAL_SEAL), OUT_56_DIR / "rpr_seal_2_4b_preds.jsonl"),
        ("rpr_seal_2.6b", str(EXPORT_36L_SEAL), OUT_56_DIR / "rpr_seal_2_6b_preds.jsonl"),
        ("rpr_v8_2.9b", str(EXPORT_V8_HERMES), out_v8_56 / "rpr_v8_2_9b_preds.jsonl"),
    ]
    summaries: dict[str, Any] = {}
    for lbl, mpath, cpath in base_specs:
        if not Path(mpath).exists():
            continue
        rows = _eval_standalone_model(
            tasks_56, label=lbl, model_path=mpath, cache_path=cpath, max_tokens=576,
            fresh=(args.fresh_v8 and lbl == "rpr_v8_2.9b"),
        )
        summaries[lbl] = _summarize_broad(rows, lbl)

    rpr_v8_rows = _eval_v8_rpr_verified_56(tasks_56, out_v8_56 / "rpr_v8_2_9b_rpr_preds.jsonl", fresh=args.fresh_v8)
    summaries["rpr_v8_2.9b+RPR"] = _summarize_broad(rpr_v8_rows, "rpr_v8_2.9b+RPR")

    (out_v8_56 / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    header = "| Model | Overall (`56`) | " + " | ".join(lbl for _, lbl in PILLAR_ORDER) + " | Closed `</think>` | Mean Think | Echo | Abstain |"
    sep = "| --- | ---: | " + " | ".join("---:" for _ in PILLAR_ORDER) + " | ---: | ---: | ---: | ---: |"
    md = ["# 56-Task 8-Pillar Generational Evaluation (`2B Base -> 2.90B v8 Engram+AttnRes+MoBA+Hermes`)", "", header, sep]
    for lbl, s in summaries.items():
        p_cells = " | ".join(f"{s['pillars'][pk]['pass_rate']*100:.1f}%" for pk, _ in PILLAR_ORDER)
        abs_str = "-" if s["tool_abstention_rate"] is None else f"{s['tool_abstention_rate']*100:.1f}%"
        md.append(
            f"| `{lbl}` | **{s['overall_pass_rate']*100:.1f}%** | {p_cells} | "
            f"{s['closed_think_rate']*100:.1f}% | {s['mean_think_tokens']} | {s['mean_echo_overlap']:.3f} | {abs_str} |"
        )
    (out_v8_56 / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md), flush=True)

    # Long-context 16-session benchmark (`4k..39k` tokens with `Engram` + `AttnRes` + `MoBA`!)
    lc_tasks = [json.loads(l) for l in (V7_CORPUS_DIR / "long_context_eval_16.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    lc_sum_path = OUT_LC_DIR / "summary.json"
    lc_summaries = json.loads(lc_sum_path.read_text(encoding="utf-8")) if lc_sum_path.exists() else {}
    lc_summaries["rpr_v8_2.9b_moba"] = _eval_long_context_v8(lc_tasks, OUT_LC_DIR / "rpr_v8_2_9b_moba_lc.jsonl", fresh=args.fresh_v8)
    lc_sum_path.write_text(json.dumps(lc_summaries, indent=2) + "\n", encoding="utf-8")

    md_lc = [
        "# Multi-Turn Long-Context Session Benchmark (`4k / 8k / 16k / 24k–39k` Tokens, `16` Unseen Sessions)",
        "",
        "| Model (`full_attention` layers) | Overall Exact (`16`) | Turn-1 `preserved` Recall | `4k` (`4`) | `8k` (`4`) | `16k` (`4`) | `24k–39k` (`4`) | Premise Switch | Irrel. Noise | Max Prompt Toks | Peak RAM |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lbl, s in lc_summaries.items():
        b = s["buckets"]
        md_lc.append(
            f"| `{lbl}` | **{s['overall_exact_acc']*100:.1f}%** | {s['overall_preserved_acc']*100:.1f}% | "
            f"{b['4k']['exact_acc']*100:.1f}% | {b['8k']['exact_acc']*100:.1f}% | "
            f"{b['16k']['exact_acc']*100:.1f}% | {b['24k']['exact_acc']*100:.1f}% | "
            f"{s['premise_change_acc']*100:.1f}% | {s['irrelevant_update_acc']*100:.1f}% | "
            f"{s['max_prompt_tokens_tested']} | {s['max_peak_memory_gb']:.2f} GiB |"
        )
    (OUT_LC_DIR / "report.md").write_text("\n".join(md_lc) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md_lc), flush=True)


if __name__ == "__main__":
    main()
