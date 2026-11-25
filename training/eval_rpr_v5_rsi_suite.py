"""Checkpointed 3-Way Evaluation on the Unseen 56-Task 8-Pillar RPR + RSI + Weight-Surgery + Research + Vision Suite.

Compares standalone models (zero external adapters or controllers):
  1. `base_2b_8bit`      : `./models/LFM3-VL-2B-MLX-8bit` (24L, 6 full_attention)
  2. `rpr_v4_2b_bf16`    : `checkpoints/LFM3-VL-2B-RPR-BF16` (24L, 6 full_attention)
  3. `rpr_rsi_2.24b_bf16`: `checkpoints/LFM3-VL-2.24B-RPR-RSI-BF16` (28L, 8 full_attention)
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from model import DEFAULT_MODEL_PATH
from training.build_rpr_v5_rsi_corpus import V5_CORPUS_DIR
from training.eval_rpr_bf16_suite import _eval_standalone_model, _summarize_broad
from training.train_and_fuse_rpr_bf16 import EXPORT_BF16_DIR
from training.upcycle_and_train_rpr_v5_bf16 import EXPORT_V5_DIR


V5_EVAL_PATH = V5_CORPUS_DIR / "heldout_v5_rsi_56.jsonl"
DEFAULT_OUT_DIR = ROOT_DIR / "results" / "rpr_v5_rsi_56"

PILLAR_ORDER = [
    ("rsi_and_weight_surgery", "RSI & Weights"),
    ("research_and_ablation", "Research/Stats"),
    ("multistep_tool_use", "MultiStep Tools"),
    ("standalone_rpr_2turn", "2-Turn RPR"),
    ("clause_witness_math", "Clause Math"),
    ("diagnostic_debugging", "Debug/Traps"),
    ("audit_and_ifeval", "Audit/IFEval"),
    ("vision_multimodal", "Vision RPR"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Base 2B vs RPR-v4 2B vs RPR-RSI-v5 2.24B on 56 unseen tasks.")
    parser.add_argument("--base-model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--v4-model", default=str(EXPORT_BF16_DIR))
    parser.add_argument("--v5-model", default=str(EXPORT_V5_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [json.loads(l) for l in V5_EVAL_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]

    specs = [
        ("base_2b", args.base_model, out_dir / "base_2b_preds.jsonl"),
        ("rpr_v4_2b", args.v4_model, out_dir / "rpr_v4_2b_preds.jsonl"),
        ("rpr_rsi_2.2b", args.v5_model, out_dir / "rpr_rsi_2_2b_preds.jsonl"),
    ]
    summaries: dict[str, Any] = {}
    for lbl, mpath, cpath in specs:
        if not Path(mpath).exists():
            print(f"[Skip] Model path not found: {mpath}", flush=True)
            continue
        rows = _eval_standalone_model(
            tasks, label=lbl, model_path=mpath, cache_path=cpath, max_tokens=576, fresh=args.fresh
        )
        summaries[lbl] = _summarize_broad(rows, lbl)

    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")

    header = "| Model | Overall (`56`) | " + " | ".join(lbl for _, lbl in PILLAR_ORDER) + " | Closed `</think>` | Mean Think | Echo | Abstain |"
    sep = "| --- | ---: | " + " | ".join("---:" for _ in PILLAR_ORDER) + " | ---: | ---: | ---: | ---: |"
    md = ["# 56-Task 8-Pillar RPR + RSI + Weight-Surgery + Research + Vision Evaluation", "", header, sep]
    for lbl in summaries:
        s = summaries[lbl]
        p_cells = " | ".join(f"{s['pillars'][pk]['pass_rate']*100:.1f}%" for pk, _ in PILLAR_ORDER)
        abs_str = "-" if s["tool_abstention_rate"] is None else f"{s['tool_abstention_rate']*100:.1f}%"
        md.append(
            f"| `{lbl}` | **{s['overall_pass_rate']*100:.1f}%** | {p_cells} | "
            f"{s['closed_think_rate']*100:.1f}% | {s['mean_think_tokens']} | {s['mean_echo_overlap']:.3f} | {abs_str} |"
        )
    (out_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md), flush=True)


if __name__ == "__main__":
    main()
