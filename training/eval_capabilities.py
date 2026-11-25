"""Evaluate base vs post-trained LFM3-VL-2B across held-out multi-skill gates with checkpoints."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from model import DEFAULT_MODEL_PATH, MLXModelClient
from training.verifiable_rewards import score_response


EVAL_PATH = Path(__file__).resolve().parent / "data" / "heldout_eval_20.jsonl"
DEFAULT_ADAPTER = ROOT_DIR / "checkpoints" / "lfm3_rpr_lora_best"
DEFAULT_OUT_DIR = ROOT_DIR / "results" / "capability_eval"


def _eval_one_model(
    tasks: list[dict[str, Any]],
    *,
    label: str,
    model_path: str,
    adapter_path: str | None,
    cache_path: Path,
    max_tokens: int = 512,
    fresh: bool = False,
) -> list[dict[str, Any]]:
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
        print(f"[capability_eval] Loaded all {len(tasks)} cached runs for {label} from {cache_path}")
        return [cached[t["id"]] for t in tasks]

    client = MLXModelClient(model_path, adapter_path=adapter_path, default_enable_thinking=True)
    tok = client._processor.tokenizer if hasattr(client._processor, "tokenizer") else client._processor

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with cache_path.open("a", encoding="utf-8") as fh:
        for idx, task in enumerate(tasks, start=1):
            tid = task["id"]
            if tid in cached:
                results.append(cached[tid])
                continue
            resp = client.generate(task["messages"], max_tokens=max_tokens, temperature=0.0, enable_thinking=True)
            prompt_txt = "\n".join(str(m.get("content", "")) for m in task["messages"])
            sc = score_response(
                resp.raw_text,
                prompt_text=prompt_txt,
                kind=task.get("kind", task["pillar"]),
                verifier=task["verifier"],
                tokenizer=tok,
                max_think_tokens=240,
            )
            row = {
                "id": tid,
                "pillar": task["pillar"],
                "label": label,
                "raw_text": resp.raw_text,
                "score": sc,
                "wall_clock_sec": round(resp.stats.wall_clock_sec, 4),
                "finish_reason": resp.stats.finish_reason,
            }
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            results.append(row)
            print(
                f"[{label}] [{idx}/{len(tasks)}] {tid:16s} | ok={sc['correct']!s:5s} | "
                f"closed={sc['closed_think']!s:5s} | think_toks={sc['think_tokens']:3d} | "
                f"echo={sc['echo_overlap']:.2f} | leaked={sc['tool_leaked']}",
                flush=True,
            )

    del client
    gc.collect()
    return results


def _summarize(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    by_pillar: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_pillar[r["pillar"]].append(r)

    def _rate(items: list[dict[str, Any]], key: str) -> float:
        return round(sum(1 for it in items if it["score"][key]) / max(1, len(items)), 4)

    n = max(1, len(rows))
    summary = {
        "label": label,
        "count": len(rows),
        "overall_pass_rate": _rate(rows, "correct"),
        "closed_think_rate": _rate(rows, "closed_think"),
        "mean_think_tokens": round(sum(r["score"]["think_tokens"] for r in rows) / n, 1),
        "mean_echo_overlap": round(sum(r["score"]["echo_overlap"] for r in rows) / n, 4),
        "tool_abstention_rate": round(
            1.0 - _rate(by_pillar["tool_abstention"], "tool_leaked"), 4
        )
        if by_pillar["tool_abstention"]
        else None,
        "pillars": {
            p: {
                "count": len(items),
                "pass_rate": _rate(items, "correct"),
                "mean_think_tokens": round(sum(r["score"]["think_tokens"] for r in items) / len(items), 1),
            }
            for p, items in by_pillar.items()
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate base vs post-trained capability gates.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [json.loads(l) for l in EVAL_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]

    base_rows = _eval_one_model(
        tasks,
        label="base",
        model_path=args.model,
        adapter_path=None,
        cache_path=out_dir / "base_preds.jsonl",
        fresh=args.fresh,
    )
    trained_rows = _eval_one_model(
        tasks,
        label="trained",
        model_path=args.model,
        adapter_path=args.adapter,
        cache_path=out_dir / "trained_preds.jsonl",
        fresh=args.fresh,
    )

    base_sum = _summarize(base_rows, "base")
    trained_sum = _summarize(trained_rows, "trained")
    payload = {"base": base_sum, "trained": trained_sum}
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = [
        "# LFM3-VL-2B Multi-Skill Capability Gate Report",
        "",
        "| Model | Overall Pass | Math/RPR | Schema/Audit | Tool Pos | Tool Abstain | Think Close | Mean Think Toks | Echo 5-gram |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in (base_sum, trained_sum):
        ps = s["pillars"]
        md.append(
            f"| `{s['label']}` | {s['overall_pass_rate']*100:.1f}% | "
            f"{ps['multihop_math']['pass_rate']*100:.1f}% | "
            f"{ps['schema_audit']['pass_rate']*100:.1f}% | "
            f"{ps['tool_positive']['pass_rate']*100:.1f}% | "
            f"{(s['tool_abstention_rate'] or 0)*100:.1f}% | "
            f"{s['closed_think_rate']*100:.1f}% | {s['mean_think_tokens']} | {s['mean_echo_overlap']:.3f} |"
        )
    (out_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
