"""Comprehensive RPR v9 (`2.91B` Pure 16-bit `bfloat16` + Outer-Loop SEAL + Zero-Prior + Cline/Hermes) Evaluation Suite.

Evaluates `checkpoints/LFM3-VL-2.9B-RPR-V9-SEAL-BF16` across ALL 3 strictly held-out benchmarks (0 contamination):
  1. `heldout_v5_rsi_56.jsonl` (`56` unseen 8-pillar tasks: `base_2b` -> `v7.1` -> `v8` -> `v9` single-shot -> `v9+RPR`)
  2. `heldout_zero_prior_24.jsonl` (`24` unseen Zero-Prior Formal Systems, Weight-Mod/RSI, & OOD Discovery tasks)
  3. `benchmark_100.jsonl` (`--split test --limit 12`: `direct`, `direct_tir`, `rpr`)
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import mlx.core as mx

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from model import DEFAULT_MODEL_PATH
from rpr_live_controller import run_live_rpr_pyramid_turn
from training.build_v9_rsi_zero_prior_cline_corpus import V9_CORPUS_DIR
from training.eval_rpr_bf16_suite import _eval_standalone_model, _summarize_broad
from training.eval_rpr_v5_rsi_suite import PILLAR_ORDER, V5_EVAL_PATH
from training.eval_v7_and_long_context import OUT_56_DIR
from training.run_v7_36l_zero_degrade_pipeline import EXPORT_36L_SEAL
from training.run_v8_2_9b_hermes_pipeline import EXPORT_V8_HERMES
from training.run_v9_2_9b_seal_cline_pipeline import EXPORT_V9_SEAL
from training.train_and_fuse_rpr_bf16 import _apply_template_inference_aligned
from training.verifiable_rewards import score_response
from v8_arch import attach_v8_extensions


ZP24_PILLARS = [
    ("zero_prior_formal", "Zero-Prior Formal (`8`)"),
    ("weight_mod_rsi", "Weight-Mod & RSI (`8`)"),
    ("research_and_ood_discovery", "Research & OOD Discovery (`8`)"),
]


def _eval_zero_leakage_rpr_pyramid(
    tasks: list[dict[str, Any]],
    *,
    label: str,
    model_path: str,
    cache_path: Path,
    max_tokens: int = 1024,
    fresh: bool = False,
) -> list[dict[str, Any]]:
    """Run zero-leakage `rpr_live_controller.run_live_rpr_pyramid_turn` (strictly NEVER sees `task['verifier']` during generation!)."""
    import mlx_vlm

    cached: dict[str, dict[str, Any]] = {}
    if not fresh and cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)
                cached[obj["id"]] = obj

    if all(t["id"] in cached for t in tasks):
        print(f"[rpr_pyramid_eval] Loaded all {len(tasks)} cached runs for {label} from {cache_path}", flush=True)
        return [cached[t["id"]] for t in tasks]

    print(f"[rpr_pyramid_eval] Loading {label} (pure 16-bit bfloat16 + Zero-Leakage RPR Pyramid) from {model_path}...", flush=True)
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
                    repetition_penalty=1.08,
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
                max_think_tokens=900,
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
                f"  [{label} {idx:2d}/{len(tasks)}] {tid:16s} ({task['pillar']:24s}) | "
                f"pass={rec['correct']} | passes={rpr_out['passes']} ({rpr_out['audit_note']}) | {dt:.2f}s",
                flush=True,
            )

    del model, processor
    gc.collect()
    mx.clear_cache()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RPR v9 (2.91B Pure 16-bit) across 56-task, 24-task Zero-Prior, & 12-problem RPR suites.")
    parser.add_argument("--fresh-v9", action="store_true")
    parser.add_argument("--skip-bench12", action="store_true")
    args = parser.parse_args()

    mx.set_wired_limit(0)
    mx.set_cache_limit(256 * 1024 * 1024)

    # ------------------------------------------------------------------
    # 1. Held-Out 56-Task 8-Pillar Benchmark (`rpr_v9_2_9b_56`)
    # ------------------------------------------------------------------
    out_56 = ROOT_DIR / "results" / "rpr_v9_2_9b_56"
    out_56.mkdir(parents=True, exist_ok=True)
    out_v8_56 = ROOT_DIR / "results" / "rpr_v8_2_9b_56"
    tasks_56 = [json.loads(l) for l in V5_EVAL_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]

    specs_56 = [
        ("base_2b", DEFAULT_MODEL_PATH, OUT_56_DIR / "base_2b_preds.jsonl"),
        ("rpr_seal_2.6b_v7.1", str(EXPORT_36L_SEAL), OUT_56_DIR / "rpr_seal_2_6b_preds.jsonl"),
        ("rpr_v8_2.9b", str(EXPORT_V8_HERMES), out_v8_56 / "rpr_v8_2_9b_preds.jsonl"),
        ("rpr_v9_2.9b", str(EXPORT_V9_SEAL), out_56 / "rpr_v9_2_9b_preds.jsonl"),
    ]
    sum_56: dict[str, Any] = {}
    for lbl, mpath, cpath in specs_56:
        if not Path(mpath).exists() and not cpath.exists():
            continue
        rows = _eval_standalone_model(
            tasks_56,
            label=lbl,
            model_path=mpath,
            cache_path=cpath,
            max_tokens=576,
            fresh=(args.fresh_v9 and lbl == "rpr_v9_2.9b"),
        )
        sum_56[lbl] = _summarize_broad(rows, lbl)

    v9_rpr_56 = _eval_zero_leakage_rpr_pyramid(
        tasks_56,
        label="rpr_v9_2.9b+RPR",
        model_path=str(EXPORT_V9_SEAL),
        cache_path=out_56 / "rpr_v9_2_9b_rpr_preds.jsonl",
        max_tokens=576,
        fresh=args.fresh_v9,
    )
    sum_56["rpr_v9_2.9b+RPR"] = _summarize_broad(v9_rpr_56, "rpr_v9_2.9b+RPR")
    (out_56 / "summary.json").write_text(json.dumps(sum_56, indent=2) + "\n", encoding="utf-8")

    hdr56 = "| Model | Overall (`56`) | " + " | ".join(lbl for _, lbl in PILLAR_ORDER) + " | Closed `</think>` | Mean Think | Abstain |"
    sep56 = "| --- | ---: | " + " | ".join("---:" for _ in PILLAR_ORDER) + " | ---: | ---: | ---: |"
    md56 = ["# 56-Task 8-Pillar Generational Benchmark (`Base 2B -> v7.1 -> v8 -> Pure 16-bit v9 & v9+RPR`)", "", hdr56, sep56]
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
    tasks_zp = [json.loads(l) for l in (V9_CORPUS_DIR / "heldout_zero_prior_24.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    specs_zp = [
        ("base_2b", DEFAULT_MODEL_PATH, out_zp / "base_2b_zp24.jsonl"),
        ("rpr_v8_2.9b", str(EXPORT_V8_HERMES), out_zp / "rpr_v8_2_9b_zp24.jsonl"),
        ("rpr_v9_2.9b", str(EXPORT_V9_SEAL), out_zp / "rpr_v9_2_9b_zp24.jsonl"),
    ]
    sum_zp: dict[str, Any] = {}
    for lbl, mpath, cpath in specs_zp:
        if not Path(mpath).exists():
            continue
        rows = _eval_standalone_model(
            tasks_zp,
            label=lbl,
            model_path=mpath,
            cache_path=cpath,
            max_tokens=576,
            fresh=(args.fresh_v9 and lbl == "rpr_v9_2.9b"),
        )
        sum_zp[lbl] = _summarize_broad(rows, lbl)

    v9_rpr_zp = _eval_zero_leakage_rpr_pyramid(
        tasks_zp,
        label="rpr_v9_2.9b+RPR",
        model_path=str(EXPORT_V9_SEAL),
        cache_path=out_zp / "rpr_v9_2_9b_rpr_zp24.jsonl",
        max_tokens=576,
        fresh=args.fresh_v9,
    )
    sum_zp["rpr_v9_2.9b+RPR"] = _summarize_broad(v9_rpr_zp, "rpr_v9_2.9b+RPR")
    (out_zp / "summary.json").write_text(json.dumps(sum_zp, indent=2) + "\n", encoding="utf-8")

    hdr24 = "| Model | Overall (`24`) | " + " | ".join(lbl for _, lbl in ZP24_PILLARS) + " | Closed `</think>` | Mean Think |"
    sep24 = "| --- | ---: | " + " | ".join("---:" for _ in ZP24_PILLARS) + " | ---: | ---: |"
    md24 = ["# 24-Task Zero-Prior-Knowledge (`OOD`) Formal Systems, Weight-Modification & Discovery Benchmark", "", hdr24, sep24]
    for lbl, s in sum_zp.items():
        p_cells = " | ".join(f"{s['pillars'][pk]['pass_rate']*100:.1f}%" for pk, _ in ZP24_PILLARS)
        md24.append(
            f"| `{lbl}` | **{s['overall_pass_rate']*100:.1f}%** | {p_cells} | "
            f"{s['closed_think_rate']*100:.1f}% | {s['mean_think_tokens']} |"
        )
    (out_zp / "report.md").write_text("\n".join(md24) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md24), flush=True)

    # ------------------------------------------------------------------
    # 3. 12-Problem `benchmark_100` Test Slice (`direct,direct_tir,rpr`)
    # ------------------------------------------------------------------
    if not args.skip_bench12:
        out_b12 = ROOT_DIR / "results" / "benchmark_test_12_rpr_v9_2_9b"
        cmd = [
            sys.executable,
            str(ROOT_DIR / "evaluate.py"),
            "--model",
            str(EXPORT_V9_SEAL),
            "--dataset",
            str(ROOT_DIR / "data" / "benchmark_100.jsonl"),
            "--split",
            "test",
            "--limit",
            "12",
            "--modes",
            "direct,direct_tir,rpr",
            "--output-dir",
            str(out_b12),
        ]
        print(f"[bench12] Running {' '.join(cmd)}...", flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
