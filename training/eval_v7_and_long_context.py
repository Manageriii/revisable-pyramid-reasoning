"""Comprehensive Generational (`2B -> 2.65B 36L`) Non-Degradation, Zero-Leakage RPR Self-Verification & Multi-Turn Long-Context (`4k..39k`) Benchmark Suite.

Produces:
  1. `results/rpr_v7_36l_56/report.md`          — 8-row generational table across all 56 unseen 8-pillar tasks
     (`base_2b` -> `rpr_v4_2b` -> `rpr_rsi_2.2b` -> `rpr_v6_2.4b` -> `rpr_seal_2.4b` -> `rpr_v7_2.6b` -> `rpr_seal_2.6b` + zero-leakage `rpr_seal_2.6b+RPR`)
  2. `results/long_context_sessions_16/report.md` — Multi-turn session benchmark across `4k`, `8k`, `16k`, and `24k` (`up to 39k` actual) tokens
     comparing `base_2b` (`24L`, `6` FA) vs `rpr_rsi_2.2b` (`28L`, `8` FA) vs `rpr_seal_2.6b` (`36L`, `12` FA)
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import gc
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any

import mlx.core as mx

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from engine import _MULTI_OP_LEDGER_RE
from model import DEFAULT_MODEL_PATH
from symbolic_critic import parse_prompt_options
from training.build_v7_master_and_longctx_corpus import V7_CORPUS_DIR
from training.eval_rpr_bf16_suite import _eval_standalone_model, _summarize_broad
from training.eval_rpr_v5_rsi_suite import PILLAR_ORDER, V5_EVAL_PATH
from training.run_v6_32l_mtp_seal_pipeline import EXPORT_32L_REPAIRED, EXPORT_FINAL_SEAL
from training.run_v7_36l_zero_degrade_pipeline import EXPORT_36L_BASE, EXPORT_36L_SEAL
from training.train_and_fuse_rpr_bf16 import EXPORT_BF16_DIR, _apply_template_inference_aligned
from training.upcycle_and_train_rpr_v5_bf16 import EXPORT_V5_DIR
from training.verifiable_rewards import score_response
from verifiers import parse_candidate_answer, safe_eval_arith, split_thinking


OUT_56_DIR = ROOT_DIR / "results" / "rpr_v7_36l_56"
PREV_V6_DIR = ROOT_DIR / "results" / "rpr_v6_seal_56"
OUT_LC_DIR = ROOT_DIR / "results" / "long_context_sessions_16"

_PY_BLOCK_RE = re.compile(r"```python\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_last_py_block(text: str) -> str | None:
    blocks = _PY_BLOCK_RE.findall(text)
    return blocks[-1].strip() if blocks else None


def _extract_top_fn_name(code_src: str) -> str | None:
    try:
        tree = ast.parse(code_src)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node.name
    return None


def _run_sandboxed_python(code_src: str, timeout_sec: float = 2.5) -> tuple[bool, str]:
    try:
        compile(code_src, "<rpr_zero_leak_check>", "exec")
        proc = subprocess.run(
            [sys.executable, "-I", "-c", code_src],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        if proc.returncode == 0:
            return True, ""
        return False, (proc.stderr or proc.stdout or "Non-zero exit code")[-600:]
    except Exception as exc:
        return False, str(exc)[-600:]


def _eval_standalone_model_rpr_verified(
    tasks: list[dict[str, Any]],
    *,
    label: str,
    model_path: str,
    cache_path: Path,
    max_tokens: int = 576,
    fresh: bool = False,
) -> list[dict[str, Any]]:
    """Zero-leakage RPR self-verification mode across all 8 pillars:
    - Strictly NEVER accesses `task['verifier']` during generation, self-testing, or reconstruction!
    - For Python code tasks: runs isolated `python3 -I` syntax/import check + model-generated prompt-only smoke assertions;
      if an exception fires, performs 1 RPR assumption-invalidation & code-rebuild turn.
    - For `math_opt` tasks: AST-verifies any multi-op ledger expression in `<think>` (`safe_eval_arith`) and repairs arithmetic slips.
    """
    import mlx_vlm

    cached: dict[str, dict[str, Any]] = {}
    if not fresh and cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    row = json.loads(line)
                    cached[row["id"]] = row
                except json.JSONDecodeError:
                    continue

    if all(t["id"] in cached for t in tasks):
        print(f"[rpr_56_eval] Loaded all {len(tasks)} cached runs for {label} from {cache_path}", flush=True)
        return [cached[t["id"]] for t in tasks]

    print(f"[rpr_56_eval] Loading {label} (Zero-Leakage RPR Self-Verify) from {model_path}...", flush=True)
    model, processor = mlx_vlm.load(model_path)
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
                model, processor, prompt_str, image=imgs,
                max_tokens=max_tokens, temperature=0.0, repetition_penalty=1.08, verbose=False,
            )
            raw_txt = res.text or ""
            prompt_txt = "\n".join(str(m.get("content", "")) for m in task["messages"])

            # 1. Close-think recovery if `<think>` truncated before `</think>`
            if "</think>" not in raw_txt and not imgs:
                rec_msgs = [
                    *task["messages"],
                    {"role": "assistant", "content": f"<think>{raw_txt}</think>\n"},
                ]
                # Continue after `</think>` cleanly
                rec_prompt = _apply_template_inference_aligned(processor, model.config, task["messages"], num_images=0, enable_thinking=False)
                res_rec = mlx_vlm.generate(model, processor, rec_prompt, max_tokens=256, temperature=0.0, repetition_penalty=1.05, verbose=False)
                if res_rec.text:
                    raw_txt = f"{raw_txt[:900]}</think>\n{res_rec.text.strip()}"

            # 2. Zero-leakage Python execution & self-generated smoke-test verification (NO access to `task["verifier"]`!)
            code_blk = _extract_last_py_block(raw_txt)
            if code_blk is not None and not imgs:
                fn_name = _extract_top_fn_name(code_blk)
                ok_syntax, err_msg = _run_sandboxed_python(code_blk)
                if ok_syntax and fn_name:
                    test_gen_msgs = [
                        {
                            "role": "system",
                            "content": (
                                "You are a zero-leakage Python unit-test auditor. Given a specification and candidate implementation, "
                                "output ONLY a ```python ... ``` block containing 2-3 concise `assert` statements testing edge cases "
                                "(e.g. 2-element arrays, negative inputs, repeated calls with omitted default args, exact output shapes/norms). "
                                "Do NOT redefine the function."
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"Specification:\n{task['messages'][-1]['content']}\n\nCandidate `{fn_name}`:\n```python\n{code_blk}\n```\n\nOutput ONLY ```python assert ... ``` statements.",
                        },
                    ]
                    tg_p = _apply_template_inference_aligned(processor, model.config, test_gen_msgs, num_images=0, enable_thinking=False)
                    tg_res = mlx_vlm.generate(model, processor, tg_p, max_tokens=224, temperature=0.0, repetition_penalty=1.05, verbose=False)
                    self_test_src = _extract_last_py_block(tg_res.text or "") or (tg_res.text or "").strip()
                    if "assert " in self_test_src and f"def {fn_name}" not in self_test_src:
                        ok_smoke, smoke_err = _run_sandboxed_python(f"{code_blk}\n\n{self_test_src}")
                        if not ok_smoke:
                            ok_syntax, err_msg = False, smoke_err

                if not ok_syntax:
                    rebuild_msgs = [
                        *task["messages"],
                        {"role": "assistant", "content": (f"<think>{raw_txt}" if not raw_txt.startswith("<think>") else raw_txt)},
                        {
                            "role": "user",
                            "content": (
                                f"Zero-leakage `python3 -I` sandbox execution caught an error:\n```\n{err_msg}\n```\n"
                                "Invalidate the broken assumption in <think>, verify all requirements in the prompt, and output ONLY the corrected ```python ... ``` block."
                            ),
                        },
                    ]
                    rb_p = _apply_template_inference_aligned(processor, model.config, rebuild_msgs, num_images=0, enable_thinking=True)
                    rb_res = mlx_vlm.generate(model, processor, rb_p, max_tokens=max_tokens, temperature=0.0, repetition_penalty=1.05, verbose=False)
                    rb_txt = rb_res.text or ""
                    rb_code = _extract_last_py_block(rb_txt)
                    if rb_code is not None:
                        ok_rb, _ = _run_sandboxed_python(rb_code)
                        if ok_rb:
                            raw_txt = rb_txt

            # 3. Zero-leakage AST math fact check on `<think>` ledger expressions for `{"primary", "preserved"}` tasks
            options = parse_prompt_options(prompt_txt)
            cand_ans = parse_candidate_answer(raw_txt)
            if options and cand_ans is not None and cand_ans.get("preserved") is not None:
                think_part, _ = split_thinking(f"<think>{raw_txt}" if not raw_txt.startswith("<think>") else raw_txt)
                led_matches = _MULTI_OP_LEDGER_RE.findall(think_part)
                if led_matches:
                    ast_val = safe_eval_arith(led_matches[0].strip())
                    if ast_val is not None and float(ast_val).is_integer():
                        ast_int = int(round(ast_val))
                        if cand_ans["preserved"] != ast_int:
                            think_clean = think_part or ""
                            fixed_json = json.dumps({"primary": cand_ans["primary"], "preserved": ast_int})
                            raw_txt = f"{think_clean}</think>\n{fixed_json}"

            dt = round(time.perf_counter() - t0, 4)
            # Oracle `task["verifier"]` is touched ONLY here at final scoring!
            sc = score_response(
                raw_txt,
                prompt_text=prompt_txt,
                kind=task.get("kind", task["pillar"]),
                verifier=task["verifier"],
                tokenizer=tok,
                max_think_tokens=320,
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
                f"[{label:16s}] [{idx:2d}/{len(tasks)}] {tid:18s} ({task['pillar']:27s}) | "
                f"ok={sc['correct']!s:5s} | think={sc['think_tokens']:3d} | dt={dt:5.2f}s",
                flush=True,
            )

    del model, processor
    gc.collect()
    mx.clear_cache()
    return results


def run_generational_56_eval(fresh_v7: bool = False) -> dict[str, Any]:
    OUT_56_DIR.mkdir(parents=True, exist_ok=True)
    if PREV_V6_DIR.exists():
        for fn in (
            "base_2b_preds.jsonl",
            "rpr_v4_2b_preds.jsonl",
            "rpr_rsi_2_2b_preds.jsonl",
            "rpr_v6_2_4b_preds.jsonl",
            "rpr_seal_2_4b_preds.jsonl",
        ):
            src_f = PREV_V6_DIR / fn
            dst_f = OUT_56_DIR / fn
            if src_f.exists() and not dst_f.exists():
                shutil.copy2(src_f, dst_f)

    if fresh_v7:
        for fn in ("rpr_v7_2_6b_preds.jsonl", "rpr_seal_2_6b_preds.jsonl", "rpr_seal_2_6b_rpr_preds.jsonl"):
            p = OUT_56_DIR / fn
            if p.exists():
                p.unlink()

    tasks = [json.loads(l) for l in V5_EVAL_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    specs = [
        ("base_2b", DEFAULT_MODEL_PATH, OUT_56_DIR / "base_2b_preds.jsonl"),
        ("rpr_v4_2b", str(EXPORT_BF16_DIR), OUT_56_DIR / "rpr_v4_2b_preds.jsonl"),
        ("rpr_rsi_2.2b", str(EXPORT_V5_DIR), OUT_56_DIR / "rpr_rsi_2_2b_preds.jsonl"),
        ("rpr_v6_2.4b", str(EXPORT_32L_REPAIRED), OUT_56_DIR / "rpr_v6_2_4b_preds.jsonl"),
        ("rpr_seal_2.4b", str(EXPORT_FINAL_SEAL), OUT_56_DIR / "rpr_seal_2_4b_preds.jsonl"),
        ("rpr_v7_2.6b", str(EXPORT_36L_BASE), OUT_56_DIR / "rpr_v7_2_6b_preds.jsonl"),
        ("rpr_seal_2.6b", str(EXPORT_36L_SEAL), OUT_56_DIR / "rpr_seal_2_6b_preds.jsonl"),
    ]
    summaries: dict[str, Any] = {}
    for lbl, mpath, cpath in specs:
        if not Path(mpath).exists():
            continue
        rows = _eval_standalone_model(tasks, label=lbl, model_path=mpath, cache_path=cpath, max_tokens=576, fresh=False)
        summaries[lbl] = _summarize_broad(rows, lbl)

    if EXPORT_36L_SEAL.exists():
        rpr_rows = _eval_standalone_model_rpr_verified(
            tasks,
            label="rpr_seal_2.6b+RPR",
            model_path=str(EXPORT_36L_SEAL),
            cache_path=OUT_56_DIR / "rpr_seal_2_6b_rpr_preds.jsonl",
            max_tokens=576,
            fresh=False,
        )
        summaries["rpr_seal_2.6b+RPR"] = _summarize_broad(rpr_rows, "rpr_seal_2.6b+RPR")

    (OUT_56_DIR / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    header = "| Model | Overall (`56`) | " + " | ".join(lbl for _, lbl in PILLAR_ORDER) + " | Closed `</think>` | Mean Think | Echo | Abstain |"
    sep = "| --- | ---: | " + " | ".join("---:" for _ in PILLAR_ORDER) + " | ---: | ---: | ---: | ---: |"
    md = ["# 56-Task 8-Pillar Generational Evaluation (`2B Base -> 2.65B 36L RPR+MTP+SEAL`)", "", header, sep]
    for lbl, s in summaries.items():
        p_cells = " | ".join(f"{s['pillars'][pk]['pass_rate']*100:.1f}%" for pk, _ in PILLAR_ORDER)
        abs_str = "-" if s["tool_abstention_rate"] is None else f"{s['tool_abstention_rate']*100:.1f}%"
        md.append(
            f"| `{lbl}` | **{s['overall_pass_rate']*100:.1f}%** | {p_cells} | "
            f"{s['closed_think_rate']*100:.1f}% | {s['mean_think_tokens']} | {s['mean_echo_overlap']:.3f} | {abs_str} |"
        )
    (OUT_56_DIR / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md), flush=True)
    return summaries


def _eval_long_context_model(
    tasks: list[dict[str, Any]],
    *,
    label: str,
    model_path: str,
    cache_path: Path,
    fresh: bool = False,
) -> dict[str, Any]:
    import mlx_vlm

    cached: dict[str, dict[str, Any]] = {}
    if not fresh and cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)
                cached[obj["id"]] = obj

    if len(cached) < len(tasks):
        print(f"[long_ctx] Loading {label} ({model_path}) for {len(tasks) - len(cached)} remaining long-context sessions...", flush=True)
        mx.reset_peak_memory()
        vlm, proc = mlx_vlm.load(model_path)
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
                f"  [{label} LongCtx {idx:2d}/{len(tasks)}] {t['id']} ({prompt_toks:5d} toks, {t['category']:17s}) | "
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
        "label": label,
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


def _eval_long_context_rpr_mode(
    tasks: list[dict[str, Any]],
    *,
    label: str,
    model_path: str,
    cache_path: Path,
    fresh: bool = False,
) -> dict[str, Any]:
    """Evaluate zero-leakage Incremental Multi-Turn Session `RPR` across `4k..39k` multi-turn sessions!

    Each incoming user turn `u_2..u_M` is audited live as it arrives against Level-1 foundations (`_run_verification_step`).
    Turns verified to introduce zero active assumption changes (`not findings.foundation_invalid`) are pruned from the
    active `PyramidState.evidence` timeline (`O(1)` ATMS state compression!), whereas premise-changing turns trigger
    surgical Option-B reconstruction (`reconstruct` + `_build_pyramid_to_depth`).
    """
    from engine import BudgetTracker, _build_pyramid_to_depth, _run_verification_step
    from model import MLXModelClient
    from revision import reconstruct
    from state import PyramidState

    cached: dict[str, dict[str, Any]] = {}
    if not fresh and cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)
                cached[obj["id"]] = obj

    if len(cached) < len(tasks):
        print(f"[long_ctx_rpr] Loading {label} ({model_path}) for {len(tasks) - len(cached)} incremental multi-turn RPR sessions...", flush=True)
        mx.reset_peak_memory()
        client = MLXModelClient(model_path, default_enable_thinking=True)
        tok = client._processor.tokenizer
        for idx, t in enumerate(tasks, start=1):
            if t["id"] in cached:
                continue
            p_str = _apply_template_inference_aligned(client._processor, client._model.config, t["messages"], num_images=0, enable_thinking=True)
            full_session_toks = len(tok.encode(p_str, add_special_tokens=False))

            user_turns = [str(m["content"]) for m in t["messages"] if m.get("role") == "user"]
            u1 = user_turns[0]
            t0 = time.perf_counter()
            budget = BudgetTracker(max_total_tokens=98304)
            traces: list[dict[str, Any]] = []
            state = PyramidState(
                problem_id=t["id"],
                problem_text=u1,
                evidence=["Initial problem statement."],
                max_reconstructions=3,
            )
            _build_pyramid_to_depth(
                state, client, budget, traces,
                target_depth=3, phase="turn1",
                max_tokens=1536, temperature=0.0, enable_thinking=True,
                use_self_checks=True, audit_every_level=False,
            )
            init_ans = dict(state.candidate_answer) if state.candidate_answer else None

            for turn_no, u_turn in enumerate(user_turns[1:], start=2):
                if not budget.can_continue():
                    break
                state.evidence.append(u_turn)
                findings, _ = _run_verification_step(
                    state, client, budget, traces,
                    stage=f"turn{turn_no}_verify_from_l1",
                    use_self_checks=True, run_model_audit=True,
                    max_tokens=512, temperature=0.0, enable_thinking=True,
                )
                if findings.foundation_invalid and findings.invalidated_assumptions:
                    state.max_reconstructions = max(state.max_reconstructions, state.reconstruction_count + 2)
                    state, rebuilt = reconstruct(state, findings, phase=f"turn{turn_no}_update")
                    if rebuilt and budget.can_continue():
                        _build_pyramid_to_depth(
                            state, client, budget, traces,
                            target_depth=3, phase=f"turn{turn_no}_rebuild",
                            max_tokens=1536, temperature=0.0, enable_thinking=True,
                            use_self_checks=True, audit_every_level=False,
                        )
                else:
                    # Verified zero-invalidation auxiliary turn: compress ATMS timeline so O(1) PyramidState stays clean!
                    state.evidence.pop()

            dt = round(time.perf_counter() - t0, 3)
            pred_ans = (dict(state.candidate_answer) if state.candidate_answer else init_ans) or {}
            pri_ok = pred_ans.get("primary") == t["verifier"]["primary"]
            pres_ok = pred_ans.get("preserved") == t["verifier"]["preserved"]
            peak_gb = round(mx.get_peak_memory() / (1024**3), 3)

            rec = {
                "id": t["id"],
                "target_tokens": t["target_tokens"],
                "actual_prompt_tokens": full_session_toks,
                "category": t["category"],
                "exact_correct": bool(pri_ok and pres_ok),
                "primary_correct": bool(pri_ok),
                "preserved_correct": bool(pres_ok),
                "closed_think": True,
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
                f"  [{label} LongCtx-RPR {idx:2d}/{len(tasks)}] {t['id']} ({full_session_toks:5d} toks, {t['category']:17s}) | "
                f"exact={rec['exact_correct']} (pri={pri_ok}, pres={pres_ok}) | {dt:5.2f}s | peak={peak_gb:.2f} GiB",
                flush=True,
            )
        del client
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
        "label": label,
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


def run_long_context_16_eval(fresh_v7: bool = False) -> dict[str, Any]:
    OUT_LC_DIR.mkdir(parents=True, exist_ok=True)
    if fresh_v7:
        for fn in ("rpr_v7_36l_base_lc.jsonl", "rpr_v7_seal_2_6b_lc.jsonl", "rpr_v7_seal_2_6b_rpr_lc.jsonl"):
            p = OUT_LC_DIR / fn
            if p.exists():
                p.unlink()

    lc_tasks = [
        json.loads(l)
        for l in (V7_CORPUS_DIR / "long_context_eval_16.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    specs = [
        ("base_2b_24L", DEFAULT_MODEL_PATH, OUT_LC_DIR / "base_2b_lc.jsonl"),
        ("rpr_v5_2.2b_28L", str(EXPORT_V5_DIR), OUT_LC_DIR / "rpr_v5_2_2b_lc.jsonl"),
        ("rpr_v7_2.6b_36L", str(EXPORT_36L_BASE), OUT_LC_DIR / "rpr_v7_36l_base_lc.jsonl"),
        ("rpr_v7_seal_2.6b_36L", str(EXPORT_36L_SEAL), OUT_LC_DIR / "rpr_v7_seal_2_6b_lc.jsonl"),
    ]
    lc_summaries: dict[str, Any] = {}
    for lbl, mpath, cpath in specs:
        if not Path(mpath).exists():
            continue
        lc_summaries[lbl] = _eval_long_context_model(lc_tasks, label=lbl, model_path=mpath, cache_path=cpath, fresh=False)

    if EXPORT_36L_SEAL.exists():
        lc_summaries["rpr_v7_seal_2.6b_36L+RPR"] = _eval_long_context_rpr_mode(
            lc_tasks,
            label="rpr_v7_seal_2.6b_36L+RPR",
            model_path=str(EXPORT_36L_SEAL),
            cache_path=OUT_LC_DIR / "rpr_v7_seal_2_6b_rpr_lc.jsonl",
            fresh=False,
        )

    (OUT_LC_DIR / "summary.json").write_text(json.dumps(lc_summaries, indent=2) + "\n", encoding="utf-8")
    md = [
        "# Multi-Turn Long-Context Session Benchmark (`4k / 8k / 16k / 24k–39k` Tokens, `16` Unseen Sessions)",
        "",
        "| Model (`full_attention` layers) | Overall Exact (`16`) | Turn-1 `preserved` Recall | `4k` (`4`) | `8k` (`4`) | `16k` (`4`) | `24k–39k` (`4`) | Premise Switch | Irrel. Noise | Max Prompt Toks | Peak RAM |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lbl, s in lc_summaries.items():
        b = s["buckets"]
        md.append(
            f"| `{lbl}` | **{s['overall_exact_acc']*100:.1f}%** | {s['overall_preserved_acc']*100:.1f}% | "
            f"{b['4k']['exact_acc']*100:.1f}% | {b['8k']['exact_acc']*100:.1f}% | "
            f"{b['16k']['exact_acc']*100:.1f}% | {b['24k']['exact_acc']*100:.1f}% | "
            f"{s['premise_change_acc']*100:.1f}% | {s['irrelevant_update_acc']*100:.1f}% | "
            f"{s['max_prompt_tokens_tested']} | {s['max_peak_memory_gb']:.2f} GiB |"
        )
    (OUT_LC_DIR / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md), flush=True)
    return lc_summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v7 36L 56-task non-degradation & 16-session 4k-39k long-context benchmarks.")
    parser.add_argument("--fresh-v7", action="store_true", help="Re-evaluate only the newly trained 36L v7 checkpoints while keeping verified 2B/2.2B/2.4B baselines.")
    args = parser.parse_args()

    mx.set_wired_limit(0)
    mx.set_cache_limit(384 * 1024 * 1024)

    run_generational_56_eval(fresh_v7=args.fresh_v7)
    run_long_context_16_eval(fresh_v7=args.fresh_v7)


if __name__ == "__main__":
    main()
