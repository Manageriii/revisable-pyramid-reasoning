"""Checkpointed 5-Way Generational Evaluation on the Unseen 56-Task 8-Pillar RPR + RSI + Research + Vision Suite.

Compares standalone models across every architectural generation:
  1. `base_2b`       : `./models/LFM3-VL-2B-MLX-8bit` (`24L`, `6` `full_attention`)
  2. `rpr_v4_2b`     : `checkpoints/LFM3-VL-2B-RPR-BF16` (`24L`, `6` `full_attention`)
  3. `rpr_rsi_2.2b`  : `checkpoints/LFM3-VL-2.2B-RPR-RSI-BF16` (`28L`, `8` `full_attention`)
  4. `rpr_v6_2.4b`   : `checkpoints/LFM3-VL-2.4B-RPR-RSI-BF16` (`32L`, `10` `full_attention`, Stage A repaired)
  5. `rpr_seal_2.4b` : `checkpoints/LFM3-VL-2.4B-RPR-MTP-SEAL-BF16` (`32L`, `10` `full_attention` + MTP + SEAL+SVF self-rewrite)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from model import DEFAULT_MODEL_PATH
from training.build_rpr_v5_rsi_corpus import V5_CORPUS_DIR
from training.eval_rpr_bf16_suite import _eval_standalone_model, _summarize_broad
from training.eval_rpr_v5_rsi_suite import PILLAR_ORDER, V5_EVAL_PATH
from training.run_v6_32l_mtp_seal_pipeline import EXPORT_32L_REPAIRED, EXPORT_FINAL_SEAL
from training.train_and_fuse_rpr_bf16 import EXPORT_BF16_DIR
from training.upcycle_and_train_rpr_v5_bf16 import EXPORT_V5_DIR


DEFAULT_OUT_DIR = ROOT_DIR / "results" / "rpr_v6_seal_56"
PREV_V5_OUT_DIR = ROOT_DIR / "results" / "rpr_v5_rsi_56"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 2B Base -> 2B v4 -> 2.2B v5 -> 2.4B v6 -> 2.4B SEAL on 56 unseen tasks.")
    parser.add_argument("--base-model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--v4-model", default=str(EXPORT_BF16_DIR))
    parser.add_argument("--v5-model", default=str(EXPORT_V5_DIR))
    parser.add_argument("--v6-stagea-model", default=str(EXPORT_32L_REPAIRED))
    parser.add_argument("--v6-seal-model", default=str(EXPORT_FINAL_SEAL))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Seed cached prior-generation predictions if available and not running `--fresh`
    if not args.fresh and PREV_V5_OUT_DIR.exists():
        for fname in ("base_2b_preds.jsonl", "rpr_v4_2b_preds.jsonl", "rpr_rsi_2_2b_preds.jsonl"):
            src_f = PREV_V5_OUT_DIR / fname
            dst_f = out_dir / fname
            if src_f.exists() and not dst_f.exists():
                shutil.copy2(src_f, dst_f)

    tasks = [json.loads(l) for l in V5_EVAL_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]

    specs = [
        ("base_2b", args.base_model, out_dir / "base_2b_preds.jsonl"),
        ("rpr_v4_2b", args.v4_model, out_dir / "rpr_v4_2b_preds.jsonl"),
        ("rpr_rsi_2.2b", args.v5_model, out_dir / "rpr_rsi_2_2b_preds.jsonl"),
        ("rpr_v6_2.4b", args.v6_stagea_model, out_dir / "rpr_v6_2_4b_preds.jsonl"),
        ("rpr_seal_2.4b", args.v6_seal_model, out_dir / "rpr_seal_2_4b_preds.jsonl"),
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
    md = ["# 56-Task 8-Pillar Generational Evaluation (`2B Base -> 2.41B RPR+MTP+SEAL`)", "", header, sep]
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
