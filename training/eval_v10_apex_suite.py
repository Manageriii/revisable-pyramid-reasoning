"""Comprehensive RPR v10 APEX (`3.18B` `40L` Pure 16-bit `bfloat16`, `256k` `LongRoPE2YaRN`) Evaluation Suite.

Evaluates `checkpoints/LFM3-VL-3.2B-RPR-V10-APEX-BF16` across ALL 4 strictly held-out benchmarks (`128` unseen tasks, 0 contamination):
  1. `heldout_v5_rsi_56.jsonl`          (`56` 8-pillar tasks: `base_2b` -> `v7.1` -> `v8` -> `v9` -> `v9+RPR` -> `v10` -> `v10+RPR`)
  2. `heldout_zero_prior_24.jsonl`      (`24` unseen Zero-Prior Formal, Weight-Mod/RSI, & OOD Discovery tasks)
  3. `heldout_v10_apex_36.jsonl`        (`36` unseen Complex Logic/SAT, Real-Time Research/Citations, Self-Discover OOD, & DGM/SVF RSI)
  4. `heldout_long_256k_needles_12.jsonl` (`12` ultra-long `16k..64k` Multi-Hop Needle + Erratum + `[DOC-XXXX]` Citation tasks)
"""

from __future__ import annotations

import argparse
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
from rpr_live_controller import run_live_rpr_pyramid_turn
from training.build_v9_rsi_zero_prior_cline_corpus import V9_CORPUS_DIR
from training.build_v10_apex_corpus import V10_CORPUS_DIR
from training.eval_rpr_bf16_suite import _eval_standalone_model, _summarize_broad
from training.eval_rpr_v5_rsi_suite import PILLAR_ORDER, V5_EVAL_PATH
from training.eval_v7_and_long_context import OUT_56_DIR
from training.eval_v9_comprehensive_suite import ZP24_PILLARS, _eval_zero_leakage_rpr_pyramid
from training.run_v7_36l_zero_degrade_pipeline import EXPORT_36L_SEAL
from training.run_v8_2_9b_hermes_pipeline import EXPORT_V8_HERMES
from training.run_v9_2_9b_seal_cline_pipeline import EXPORT_V9_SEAL
from training.run_v10_3_2b_apex_pipeline import EXPORT_V10_APEX
from training.train_and_fuse_rpr_bf16 import _apply_template_inference_aligned
from training.verifiable_rewards import score_response
from v8_arch import attach_v8_extensions


APEX36_PILLARS = [
    ("complex_logic_sat", "Complex Logic & SAT (`9`)"),
    ("realtime_research_citations", "Research & Citations (`9`)"),
    ("unseen_ood_self_discover", "Self-Discover OOD (`9`)"),
    ("advanced_rsi_dgm_svf", "DGM/SVF/DrGRPO RSI (`9`)"),
]

LONG12_PILLARS = [
    ("long_context_16k", "16k Needles (`3`)"),
    ("long_context_32k", "32k Needles (`3`)"),
    ("long_context_48k", "48k Needles (`3`)"),
    ("long_context_64k", "64k Needles (`3`)"),
]


def _load_cached_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _eval_long_context_needles(
    tasks: list[dict[str, Any]],
    *,
    label: str,
    model_path: str,
    cache_path: Path,
    max_tokens: int = 640,
    use_rpr_controller: bool = False,
    fresh: bool = False,
) -> list[dict[str, Any]]:
    """Evaluate `16k..64k` Multi-Hop Needle + Erratum + `[DOC-XXXX]` Citation tasks (`prefill_step_size=1024`)."""
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
        print(f"[long_ctx_eval] Loaded all {len(tasks)} cached runs for {label} from {cache_path}", flush=True)
        return [cached[t["id"]] for t in tasks]

    print(f"[long_ctx_eval] Loading {label} (LongRoPE2YaRN + FlashSDPA<=64k / MoBA>64k) from {model_path}...", flush=True)
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
        if en_th:
            kw["thinking_budget"] = 260
        ri = mlx_vlm.generate(model, processor, ps_i, **kw)
        tx = ri.text or ""
        if ps_i.endswith("<think>") and not tx.lstrip().startswith("<think>"):
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

            p_s = _apply_template_inference_aligned(
                processor,
                model.config,
                task["messages"],
                num_images=0,
                enable_thinking=True,
            )
            prompt_tok_len = len(tok.encode(p_s, add_special_tokens=False))
            t0 = time.perf_counter()
            if use_rpr_controller:
                rpr_out = run_live_rpr_pyramid_turn(task["messages"], _gen_cb, max_tokens=max_tokens, has_images=False)
                raw_txt = rpr_out["text"]
            else:
                raw_txt = _gen_cb(task["messages"], max_tokens, True)
            dt = round(time.perf_counter() - t0, 3)

            prompt_txt = "\n".join(str(m.get("content", "")) for m in task["messages"])
            sc = score_response(
                raw_txt,
                prompt_text=prompt_txt[:2000],
                kind=task.get("kind", "citation_grounded"),
                verifier=task["verifier"],
                tokenizer=tok,
                max_think_tokens=360,
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
                "wall_clock_sec": dt,
                "raw_text": raw_txt,
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            results.append(rec)
            gc.collect()
            mx.clear_cache()
            print(
                f"  [{label} {idx:2d}/{len(tasks)}] {tid:18s} ({prompt_tok_len:6d} toks) | "
                f"pass={rec['correct']} | closed={rec['closed_think']} | {dt:.2f}s",
                flush=True,
            )

    del model, processor
    gc.collect()
    mx.clear_cache()
    return results


def _unload_8991() -> None:
    import urllib.request
    try:
        urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8991/v1/admin/unload", method="POST"), timeout=45).read()
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RPR v10 APEX (3.18B 40L Pure 16-bit 256k) across all 4 held-out suites.")
    parser.add_argument("--fresh-v10", action="store_true")
    parser.add_argument("--fresh-long", action="store_true")
    args = parser.parse_args()

    if args.fresh_v10:
        for stale_p in (
            ROOT_DIR / "results" / "rpr_v10_3_2b_56" / "rpr_v10_3_2b_preds.jsonl",
            ROOT_DIR / "results" / "rpr_v10_3_2b_56" / "rpr_v10_3_2b_rpr_preds.jsonl",
            ROOT_DIR / "results" / "zero_prior_ood_24" / "rpr_v10_3_2b_zp24.jsonl",
            ROOT_DIR / "results" / "zero_prior_ood_24" / "rpr_v10_3_2b_rpr_zp24.jsonl",
            ROOT_DIR / "results" / "v10_apex_36" / "rpr_v10_3_2b_apex36.jsonl",
            ROOT_DIR / "results" / "v10_apex_36" / "rpr_v10_3_2b_rpr_apex36.jsonl",
            ROOT_DIR / "results" / "v10_long_256k_needles_12" / "rpr_v10_3_2b_long12.jsonl",
            ROOT_DIR / "results" / "v10_long_256k_needles_12" / "rpr_v10_3_2b_rpr_long12.jsonl",
        ):
            if stale_p.exists():
                stale_p.unlink()

    mx.set_wired_limit(0)
    mx.set_cache_limit(192 * 1024 * 1024)

    # ------------------------------------------------------------------
    # 1. Held-Out 56-Task 8-Pillar Benchmark (`results/rpr_v10_3_2b_56`)
    # ------------------------------------------------------------------
    out_56 = ROOT_DIR / "results" / "rpr_v10_3_2b_56"
    out_56.mkdir(parents=True, exist_ok=True)
    out_v8_56 = ROOT_DIR / "results" / "rpr_v8_2_9b_56"
    out_v9_56 = ROOT_DIR / "results" / "rpr_v9_2_9b_56"
    tasks_56 = _load_cached_jsonl(V5_EVAL_PATH)

    sum_56: dict[str, Any] = {}
    for lbl, mpath, cpath in [
        ("base_2b", DEFAULT_MODEL_PATH, OUT_56_DIR / "base_2b_preds.jsonl"),
        ("rpr_seal_2.6b_v7.1", str(EXPORT_36L_SEAL), OUT_56_DIR / "rpr_seal_2_6b_preds.jsonl"),
        ("rpr_v8_2.9b", str(EXPORT_V8_HERMES), out_v8_56 / "rpr_v8_2_9b_preds.jsonl"),
        ("rpr_v9_2.9b", str(EXPORT_V9_SEAL), out_v9_56 / "rpr_v9_2_9b_preds.jsonl"),
    ]:
        if cpath.exists():
            sum_56[lbl] = _summarize_broad(_load_cached_jsonl(cpath), lbl)
        elif Path(mpath).exists():
            rows = _eval_standalone_model(tasks_56, label=lbl, model_path=mpath, cache_path=cpath, max_tokens=576)
            sum_56[lbl] = _summarize_broad(rows, lbl)

    v9_rpr_56_p = out_v9_56 / "rpr_v9_2_9b_rpr_preds.jsonl"
    if v9_rpr_56_p.exists():
        sum_56["rpr_v9_2.9b+RPR"] = _summarize_broad(_load_cached_jsonl(v9_rpr_56_p), "rpr_v9_2.9b+RPR")

    _unload_8991()
    v10_56 = _eval_standalone_model(
        tasks_56,
        label="rpr_v10_3.2b",
        model_path=str(EXPORT_V10_APEX),
        cache_path=out_56 / "rpr_v10_3_2b_preds.jsonl",
        max_tokens=576,
        fresh=args.fresh_v10,
    )
    sum_56["rpr_v10_3.2b"] = _summarize_broad(v10_56, "rpr_v10_3.2b")

    v10_rpr_56 = _eval_zero_leakage_rpr_pyramid(
        tasks_56,
        label="rpr_v10_3.2b+RPR",
        model_path=str(EXPORT_V10_APEX),
        cache_path=out_56 / "rpr_v10_3_2b_rpr_preds.jsonl",
        max_tokens=576,
        fresh=args.fresh_v10,
    )
    sum_56["rpr_v10_3.2b+RPR"] = _summarize_broad(v10_rpr_56, "rpr_v10_3.2b+RPR")
    (out_56 / "summary.json").write_text(json.dumps(sum_56, indent=2) + "\n", encoding="utf-8")

    hdr56 = "| Model | Overall (`56`) | " + " | ".join(lbl for _, lbl in PILLAR_ORDER) + " | Closed `</think>` | Mean Think | Abstain |"
    sep56 = "| --- | ---: | " + " | ".join("---:" for _ in PILLAR_ORDER) + " | ---: | ---: | ---: |"
    md56 = ["# 56-Task 8-Pillar Generational Benchmark (`Base 2B -> v7.1 -> v8 -> v9 -> Pure 16-bit 3.2B v10 & v10+RPR`)", "", hdr56, sep56]
    for lbl, s in sum_56.items():
        p_cells = " | ".join(f"{s['pillars'][pk]['pass_rate']*100:.1f}%" for pk, _ in PILLAR_ORDER)
        abs_s = "-" if s["tool_abstention_rate"] is None else f"{s['tool_abstention_rate']*100:.1f}%"
        md56.append(
            f"| `{lbl}` | **{s['overall_pass_rate']*100:.1f}%** | {p_cells} | "
            f"{s['closed_think_rate']*100:.1f}% | {s['mean_think_tokens']} | {abs_s} |"
        )
    (out_56 / "report.md").write_text("\n".join(md56) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md56), flush=True)

    # ------------------------------------------------------------------
    # 2. Held-Out 24-Task Zero-Prior OOD + Weight-Mod + Discovery Benchmark
    # ------------------------------------------------------------------
    out_zp = ROOT_DIR / "results" / "zero_prior_ood_24"
    out_zp.mkdir(parents=True, exist_ok=True)
    tasks_zp = _load_cached_jsonl(V9_CORPUS_DIR / "heldout_zero_prior_24.jsonl")

    sum_zp: dict[str, Any] = {}
    for lbl, mpath, cpath in [
        ("base_2b", DEFAULT_MODEL_PATH, out_zp / "base_2b_zp24.jsonl"),
        ("rpr_v8_2.9b", str(EXPORT_V8_HERMES), out_zp / "rpr_v8_2_9b_zp24.jsonl"),
        ("rpr_v9_2.9b", str(EXPORT_V9_SEAL), out_zp / "rpr_v9_2_9b_zp24.jsonl"),
    ]:
        if cpath.exists():
            sum_zp[lbl] = _summarize_broad(_load_cached_jsonl(cpath), lbl)
        elif Path(mpath).exists():
            rows = _eval_standalone_model(tasks_zp, label=lbl, model_path=mpath, cache_path=cpath, max_tokens=576)
            sum_zp[lbl] = _summarize_broad(rows, lbl)

    v9_rpr_zp_p = out_zp / "rpr_v9_2_9b_rpr_zp24.jsonl"
    if v9_rpr_zp_p.exists():
        sum_zp["rpr_v9_2.9b+RPR"] = _summarize_broad(_load_cached_jsonl(v9_rpr_zp_p), "rpr_v9_2.9b+RPR")

    _unload_8991()
    v10_zp = _eval_standalone_model(
        tasks_zp,
        label="rpr_v10_3.2b",
        model_path=str(EXPORT_V10_APEX),
        cache_path=out_zp / "rpr_v10_3_2b_zp24.jsonl",
        max_tokens=576,
        fresh=args.fresh_v10,
    )
    sum_zp["rpr_v10_3.2b"] = _summarize_broad(v10_zp, "rpr_v10_3.2b")

    v10_rpr_zp = _eval_zero_leakage_rpr_pyramid(
        tasks_zp,
        label="rpr_v10_3.2b+RPR",
        model_path=str(EXPORT_V10_APEX),
        cache_path=out_zp / "rpr_v10_3_2b_rpr_zp24.jsonl",
        max_tokens=576,
        fresh=args.fresh_v10,
    )
    sum_zp["rpr_v10_3.2b+RPR"] = _summarize_broad(v10_rpr_zp, "rpr_v10_3.2b+RPR")
    (out_zp / "summary_v10.json").write_text(json.dumps(sum_zp, indent=2) + "\n", encoding="utf-8")

    hdr24 = "| Model | Overall (`24`) | " + " | ".join(lbl for _, lbl in ZP24_PILLARS) + " | Closed `</think>` | Mean Think |"
    sep24 = "| --- | ---: | " + " | ".join("---:" for _ in ZP24_PILLARS) + " | ---: | ---: |"
    md24 = ["# 24-Task Zero-Prior-Knowledge (`OOD`) Formal Systems, Weight-Modification & Discovery Benchmark", "", hdr24, sep24]
    for lbl, s in sum_zp.items():
        p_cells = " | ".join(f"{s['pillars'][pk]['pass_rate']*100:.1f}%" for pk, _ in ZP24_PILLARS)
        md24.append(
            f"| `{lbl}` | **{s['overall_pass_rate']*100:.1f}%** | {p_cells} | "
            f"{s['closed_think_rate']*100:.1f}% | {s['mean_think_tokens']} |"
        )
    (out_zp / "report_v10.md").write_text("\n".join(md24) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md24), flush=True)

    # ------------------------------------------------------------------
    # 3. Held-Out 36-Task Frontier APEX Suite (`results/v10_apex_36`)
    # ------------------------------------------------------------------
    out_36 = ROOT_DIR / "results" / "v10_apex_36"
    out_36.mkdir(parents=True, exist_ok=True)
    tasks_36 = _load_cached_jsonl(V10_CORPUS_DIR / "heldout_v10_apex_36.jsonl")

    sum_36: dict[str, Any] = {}
    for lbl, mpath, cpath in [
        ("base_2b", DEFAULT_MODEL_PATH, out_36 / "base_2b_apex36.jsonl"),
        ("rpr_v9_2.9b", str(EXPORT_V9_SEAL), out_36 / "rpr_v9_2_9b_apex36.jsonl"),
        ("rpr_v10_3.2b", str(EXPORT_V10_APEX), out_36 / "rpr_v10_3_2b_apex36.jsonl"),
    ]:
        if not Path(mpath).exists() and not cpath.exists():
            continue
        rows = _eval_standalone_model(
            tasks_36,
            label=lbl,
            model_path=mpath,
            cache_path=cpath,
            max_tokens=576,
            fresh=(args.fresh_v10 and lbl == "rpr_v10_3.2b"),
        )
        sum_36[lbl] = _summarize_broad(rows, lbl)

    _unload_8991()
    v10_rpr_36 = _eval_zero_leakage_rpr_pyramid(
        tasks_36,
        label="rpr_v10_3.2b+RPR",
        model_path=str(EXPORT_V10_APEX),
        cache_path=out_36 / "rpr_v10_3_2b_rpr_apex36.jsonl",
        max_tokens=576,
        fresh=args.fresh_v10,
    )
    sum_36["rpr_v10_3.2b+RPR"] = _summarize_broad(v10_rpr_36, "rpr_v10_3.2b+RPR")
    (out_36 / "summary.json").write_text(json.dumps(sum_36, indent=2) + "\n", encoding="utf-8")

    hdr36 = "| Model | Overall (`36`) | " + " | ".join(lbl for _, lbl in APEX36_PILLARS) + " | Closed `</think>` | Mean Think |"
    sep36 = "| --- | ---: | " + " | ".join("---:" for _ in APEX36_PILLARS) + " | ---: | ---: |"
    md36 = ["# 36-Task v10 Frontier APEX Benchmark (Complex Logic/SAT, Citations, Self-Discover OOD, DGM/SVF RSI)", "", hdr36, sep36]
    for lbl, s in sum_36.items():
        p_cells = " | ".join(f"{s['pillars'][pk]['pass_rate']*100:.1f}%" for pk, _ in APEX36_PILLARS)
        md36.append(
            f"| `{lbl}` | **{s['overall_pass_rate']*100:.1f}%** | {p_cells} | "
            f"{s['closed_think_rate']*100:.1f}% | {s['mean_think_tokens']} |"
        )
    (out_36 / "report.md").write_text("\n".join(md36) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md36), flush=True)

    # ------------------------------------------------------------------
    # 4. Held-Out 12-Task Ultra-Long-Context `16k..64k` Multi-Hop Needle + Citation Suite
    # ------------------------------------------------------------------
    out_long = ROOT_DIR / "results" / "v10_long_256k_needles_12"
    out_long.mkdir(parents=True, exist_ok=True)
    tasks_long = _load_cached_jsonl(V10_CORPUS_DIR / "heldout_long_256k_needles_12.jsonl")

    sum_long: dict[str, Any] = {}
    for lbl, mpath, cpath, use_rpr_l in [
        ("rpr_v9_2.9b", str(EXPORT_V9_SEAL), out_long / "rpr_v9_2_9b_long12.jsonl", False),
        ("rpr_v10_3.2b", str(EXPORT_V10_APEX), out_long / "rpr_v10_3_2b_long12.jsonl", False),
        ("rpr_v10_3.2b+RPR", str(EXPORT_V10_APEX), out_long / "rpr_v10_3_2b_rpr_long12.jsonl", True),
    ]:
        if not Path(mpath).exists() and not cpath.exists():
            continue
        _unload_8991()
        rows_l = _eval_long_context_needles(
            tasks_long,
            label=lbl,
            model_path=mpath,
            cache_path=cpath,
            max_tokens=640,
            use_rpr_controller=use_rpr_l,
            fresh=(args.fresh_long or (args.fresh_v10 and lbl.startswith("rpr_v10_3.2b"))),
        )
        s_l = _summarize_broad(rows_l, lbl)
        s_l["max_prompt_tokens_tested"] = max((r.get("prompt_tokens", 0) for r in rows_l), default=0)
        s_l["mean_wall_clock_sec"] = round(sum(r.get("wall_clock_sec", 0.0) for r in rows_l) / max(1, len(rows_l)), 2)
        sum_long[lbl] = s_l

    (out_long / "summary.json").write_text(json.dumps(sum_long, indent=2) + "\n", encoding="utf-8")
    hdr_l = "| Model | Overall (`12`) | " + " | ".join(lbl for _, lbl in LONG12_PILLARS) + " | Max Tokens Tested | Mean Sec |"
    sep_l = "| --- | ---: | " + " | ".join("---:" for _ in LONG12_PILLARS) + " | ---: | ---: |"
    md_l = ["# 12-Task Ultra-Long-Context (`16k..64k`) Multi-Hop Needle + Erratum + Citation Benchmark", "", hdr_l, sep_l]
    for lbl, s in sum_long.items():
        p_cells = " | ".join(f"{s['pillars'][pk]['pass_rate']*100:.1f}%" for pk, _ in LONG12_PILLARS)
        md_l.append(
            f"| `{lbl}` | **{s['overall_pass_rate']*100:.1f}%** | {p_cells} | "
            f"`{s['max_prompt_tokens_tested']}` | {s['mean_wall_clock_sec']}s |"
        )
    (out_long / "report.md").write_text("\n".join(md_l) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md_l), flush=True)


if __name__ == "__main__":
    main()
