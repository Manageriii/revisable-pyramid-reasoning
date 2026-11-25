"""Checkpointed 42-task 7-pillar evaluation comparing Frozen Base vs Standalone 16-Bit RPR Model."""

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
from training.train_and_fuse_rpr_bf16 import EXPORT_BF16_DIR, _apply_template_inference_aligned
from training.verifiable_rewards import score_response


BROAD_EVAL_PATH = Path(__file__).resolve().parent / "data" / "rpr_v4_corpus" / "heldout_broad_42.jsonl"
DEFAULT_OUT_DIR = ROOT_DIR / "results" / "rpr_bf16_broad_42"


def _eval_standalone_model(
    tasks: list[dict[str, Any]],
    *,
    label: str,
    model_path: str,
    cache_path: Path,
    max_tokens: int = 1024,
    fresh: bool = False,
) -> list[dict[str, Any]]:
    import mlx_vlm

    cached: dict[str, dict[str, Any]] = {}
    if not fresh and cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                cached[row["id"]] = row
            except json.JSONDecodeError:
                continue

    if all(t["id"] in cached for t in tasks):
        print(f"[broad_eval] Loaded all {len(tasks)} cached runs for {label} from {cache_path}", flush=True)
        return [cached[t["id"]] for t in tasks]

    print(f"[broad_eval] Loading {label} from {model_path}...", flush=True)
    model, processor = mlx_vlm.load(model_path)
    if (Path(model_path) / "v8_arch_ext.safetensors").exists():
        from v8_arch import attach_v8_extensions

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
                repetition_penalty=1.08,
                verbose=False,
            )
            dt = round(time.perf_counter() - t0, 4)
            raw_txt = res.text or ""
            prompt_txt = "\n".join(str(m.get("content", "")) for m in task["messages"])
            sc = score_response(
                raw_txt,
                prompt_text=prompt_txt,
                kind=task.get("kind", task["pillar"]),
                verifier=task["verifier"],
                tokenizer=tok,
                max_think_tokens=900,
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
            print(
                f"[{label:8s}] [{idx:2d}/{len(tasks)}] {tid:18s} ({task['pillar']:27s}) | "
                f"ok={sc['correct']!s:5s} | think={sc['think_tokens']:3d} | echo={sc['echo_overlap']:.2f}",
                flush=True,
            )

    del model, processor
    gc.collect()
    mx.clear_cache()
    return results


def _summarize_broad(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    by_pillar: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_pillar[r["pillar"]].append(r)

    def _rate(items: list[dict[str, Any]], key: str) -> float:
        return round(sum(1 for it in items if it["score"][key]) / max(1, len(items)), 4)

    n = max(1, len(rows))
    neg_items = [r for r in rows if r.get("kind") == "tool_neg"]
    return {
        "label": label,
        "count": len(rows),
        "overall_pass_rate": _rate(rows, "correct"),
        "closed_think_rate": _rate(rows, "closed_think"),
        "mean_think_tokens": round(sum(r["score"]["think_tokens"] for r in rows) / n, 1),
        "mean_echo_overlap": round(sum(r["score"]["echo_overlap"] for r in rows) / n, 4),
        "tool_abstention_rate": round(1.0 - _rate(neg_items, "tool_leaked"), 4) if neg_items else None,
        "pillars": {
            p: {
                "count": len(items),
                "pass_rate": _rate(items, "correct"),
                "mean_think_tokens": round(sum(r["score"]["think_tokens"] for r in items) / len(items), 1),
            }
            for p, items in by_pillar.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Frozen Base vs Standalone 16-Bit RPR Model on 42 unseen tasks.")
    parser.add_argument("--base-model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--rpr-model", default=str(EXPORT_BF16_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [json.loads(l) for l in BROAD_EVAL_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]

    base_rows = _eval_standalone_model(
        tasks, label="base_8b", model_path=args.base_model, cache_path=out_dir / "base_preds.jsonl", fresh=args.fresh
    )
    rpr_rows = _eval_standalone_model(
        tasks, label="rpr_bf16", model_path=args.rpr_model, cache_path=out_dir / "rpr_bf16_preds.jsonl", fresh=args.fresh
    )

    base_sum = _summarize_broad(base_rows, "base_8b")
    rpr_sum = _summarize_broad(rpr_rows, "rpr_bf16")
    (out_dir / "summary.json").write_text(json.dumps({"base": base_sum, "rpr_bf16": rpr_sum}, indent=2), encoding="utf-8")

    pillar_order = [
        ("standalone_rpr_2turn", "2-Turn RPR"),
        ("diagnostic_debugging", "Debug/Traps"),
        ("ai_weights_and_self_improve", "AI/Weights/SelfImp"),
        ("multihop_math", "Multi-Hop Math"),
        ("audit_and_ifeval", "Audit/IFEval"),
        ("tool_calibration", "Tools Pos+Neg"),
        ("vision_multimodal", "Vision RPR"),
    ]
    header = "| Model | Overall | " + " | ".join(lbl for _, lbl in pillar_order) + " | Think Close | Mean Think | Echo |"
    sep = "| --- | ---: | " + " | ".join("---:" for _ in pillar_order) + " | ---: | ---: | ---: |"
    md = ["# 42-Task 7-Pillar Standalone RPR-BF16 Evaluation", "", header, sep]
    for s in (base_sum, rpr_sum):
        p_cells = " | ".join(f"{s['pillars'][pk]['pass_rate']*100:.1f}%" for pk, _ in pillar_order)
        md.append(
            f"| `{s['label']}` | **{s['overall_pass_rate']*100:.1f}%** | {p_cells} | "
            f"{s['closed_think_rate']*100:.1f}% | {s['mean_think_tokens']} | {s['mean_echo_overlap']:.3f} |"
        )
    (out_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md), flush=True)


if __name__ == "__main__":
    main()
