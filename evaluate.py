"""Experiment runner and metric aggregator comparing Direct, Recheck, Restart, and RPR."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

from engine import BudgetTracker, ModeName, RunResult, run_mode
from model import DEFAULT_MODEL_PATH, MLXModelClient, MockModelClient, ModelClient
from state import Assumption, DependencyNode, PyramidState, RevisionRecord, VerifiedFact
from verifiers import Problem, ProblemScore, load_problems, score_problem_run


ALL_MODES: tuple[ModeName, ...] = ("direct", "direct_tir", "recheck", "restart", "rpr")


def summarize_mode_results(
    mode: ModeName,
    records: list[tuple[Problem, RunResult, ProblemScore]],
) -> dict[str, Any]:
    """Compute aggregate accuracy, belief-revision, and efficiency metrics for one mode."""
    n = len(records)
    if n == 0:
        return {"mode": mode, "count": 0}

    exact_ok = sum(1 for _, _, s in records if s.final_exact_correct)
    primary_ok = sum(1 for _, _, s in records if s.final_primary_correct)
    preserved_ok = sum(1 for _, _, s in records if s.final_preserved_correct)
    parsed_ok = sum(1 for _, _, s in records if s.parsed_successfully)

    by_cat: dict[str, list[tuple[Problem, RunResult, ProblemScore]]] = defaultdict(list)
    for item in records:
        by_cat[item[0].category].append(item)

    def _acc(items: list[tuple[Problem, RunResult, ProblemScore]]) -> float | None:
        if not items:
            return None
        return round(sum(1 for _, _, s in items if s.final_exact_correct) / len(items), 4)

    unnecessary_pool = by_cat["premise_change"] + by_cat["irrelevant_update"]
    unnecessary_rate = (
        round(
            sum(1 for _, _, s in unnecessary_pool if s.unnecessary_change)
            / len(unnecessary_pool),
            4,
        )
        if unnecessary_pool
        else None
    )

    total_prompt_toks = sum(r.budget.prompt_tokens for _, r, _ in records)
    total_gen_toks = sum(r.budget.generation_tokens for _, r, _ in records)
    total_toks = sum(r.budget.total_tokens for _, r, _ in records)
    total_sec = sum(r.budget.wall_clock_sec for _, r, _ in records)
    total_calls = sum(r.budget.model_calls for _, r, _ in records)
    peak_mem = max((r.budget.peak_memory_gb for _, r, _ in records), default=0.0)
    budget_hits = sum(1 for _, r, _ in records if r.budget.budget_exhausted)

    inv_checked = [
        s.invalidation_detected
        for _, _, s in by_cat["premise_change"]
        if s.invalidation_detected is not None
    ]
    fp_checked = [
        s.invalidation_false_positive
        for _, _, s in by_cat["irrelevant_update"]
        if s.invalidation_false_positive is not None
    ]

    return {
        "mode": mode,
        "count": n,
        "final_accuracy": round(exact_ok / n, 4),
        "primary_accuracy": round(primary_ok / n, 4),
        "preserved_accuracy": round(preserved_ok / n, 4),
        "parse_rate": round(parsed_ok / n, 4),
        "revision_accuracy": _acc(by_cat["premise_change"]),
        "irrelevant_accuracy": _acc(by_cat["irrelevant_update"]),
        "ordinary_accuracy": _acc(by_cat["ordinary"]),
        "unnecessary_change_rate": unnecessary_rate,
        "invalidation_detection_rate": (
            round(sum(1 for x in inv_checked if x) / len(inv_checked), 4)
            if inv_checked
            else None
        ),
        "false_reconstruction_rate": (
            round(sum(1 for x in fp_checked if x) / len(fp_checked), 4)
            if fp_checked
            else None
        ),
        "mean_prompt_tokens": round(total_prompt_toks / n, 1),
        "mean_generation_tokens": round(total_gen_toks / n, 1),
        "mean_total_tokens": round(total_toks / n, 1),
        "mean_wall_clock_sec": round(total_sec / n, 3),
        "mean_model_calls": round(total_calls / n, 2),
        "peak_memory_gb": round(peak_mem, 3),
        "tokens_per_correct": round(total_toks / exact_ok, 1) if exact_ok else None,
        "sec_per_correct": round(total_sec / exact_ok, 3) if exact_ok else None,
        "budget_exhausted_count": budget_hits,
    }


def _fmt_pct(val: float | None) -> str:
    return "-" if val is None else f"{val * 100:.1f}%"


def _fmt_num(val: float | int | None, suffix: str = "") -> str:
    return "-" if val is None else f"{val}{suffix}"


def render_markdown_report(
    summaries: list[dict[str, Any]],
    rpr_records: list[tuple[Problem, RunResult, ProblemScore]],
    config: dict[str, Any],
) -> str:
    """Create a human-readable Markdown summary table + RPR reconstruction log."""
    lines = [
        "# RPR Evaluation Report",
        "",
        f"- **Model**: `{config['model']}`",
        f"- **Dataset**: `{config['dataset']}` ({config['problem_count']} problems)",
        f"- **Token budget / problem**: `{config['token_budget']}`",
        f"- **Time budget / problem (s)**: `{config['time_budget']}`",
        f"- **Thinking enabled**: `{config['enable_thinking']}`",
        f"- **Self-consistency checks (zero-leakage)**: `{config['use_self_checks']}`",
        "",
        "## Comparison Table",
        "",
        "| Mode | Final Acc | Revision Acc | Irrel. Acc | Ordinary Acc | Unnec. Change | Inval. Detect | False Rebuild | Mean Tokens | Mean Calls | Mean Time (s) | Tokens / Correct |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in summaries:
        if s.get("count", 0) == 0:
            lines.append(f"| `{s['mode']}` | — | — | — | — | — | — | — | 0 | 0 | 0 | — |")
            continue
        lines.append(
            f"| `{s['mode']}` | {_fmt_pct(s['final_accuracy'])} | {_fmt_pct(s['revision_accuracy'])} | "
            f"{_fmt_pct(s['irrelevant_accuracy'])} | {_fmt_pct(s['ordinary_accuracy'])} | "
            f"{_fmt_pct(s['unnecessary_change_rate'])} | {_fmt_pct(s['invalidation_detection_rate'])} | "
            f"{_fmt_pct(s['false_reconstruction_rate'])} | {s['mean_total_tokens']} | "
            f"{s['mean_model_calls']} | {s['mean_wall_clock_sec']} | {_fmt_num(s['tokens_per_correct'])} |"
        )

    if rpr_records:
        lines.extend(["", "## RPR Revision & Reconstruction Log", ""])
        for prob, res, score in rpr_records:
            rev_log = res.pyramid_state.revision_log if res.pyramid_state else []
            status = "PASS" if score.final_exact_correct else "FAIL"
            lines.append(
                f"### `{prob.id}` ({prob.category}, {prob.domain}) — **{status}** "
                f"| pred=`{res.final_answer}` | expected=`{prob.expected_final}`"
            )
            if not rev_log:
                lines.append("- No reconstruction triggered.")
            else:
                for rec in rev_log:
                    lines.append(
                        f"- Step {rec.step} (`{rec.phase}`): invalidated={rec.invalidated_assumptions}, "
                        f"tainted={rec.tainted_nodes}, preserved={rec.preserved_facts} — "
                        f"reason: {rec.trigger_reason}"
                    )
            lines.append("")

    return "\n".join(lines)


def _load_checkpoint_rows(trace_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Load valid completed `(problem_id, mode)` rows from `traces.jsonl`, discarding partial lines."""
    saved: dict[tuple[str, str], dict[str, Any]] = {}
    if not trace_path.exists():
        return saved
    with trace_path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = row.get("problem_id")
            mode = row.get("mode")
            if isinstance(pid, str) and isinstance(mode, str) and isinstance(row.get("score"), dict):
                saved[(pid, mode)] = row
    return saved


def _rebuild_record_from_row(
    row: dict[str, Any],
    problem: Problem,
    *,
    token_budget: int | None,
    time_budget: float | None,
) -> tuple[Problem, RunResult, ProblemScore]:
    """Reconstruct `(Problem, RunResult, ProblemScore)` from a checkpoint JSONL row."""
    b = row.get("budget") or {}
    budget = BudgetTracker(
        max_total_tokens=token_budget,
        max_wall_clock_sec=time_budget,
        prompt_tokens=int(b.get("prompt_tokens", 0)),
        generation_tokens=int(b.get("generation_tokens", 0)),
        total_tokens=int(b.get("total_tokens", 0)),
        wall_clock_sec=float(b.get("wall_clock_sec", 0.0)),
        model_calls=int(b.get("model_calls", 0)),
        peak_memory_gb=float(b.get("peak_memory_gb", 0.0)),
        budget_exhausted=bool(b.get("budget_exhausted", False)),
    )
    ps_raw = row.get("pyramid_state")
    pyramid_state: PyramidState | None = None
    invalidated_statements: list[str] = []
    if isinstance(ps_raw, dict):
        assumptions = {
            k: Assumption(**v)
            for k, v in (ps_raw.get("assumptions") or {}).items()
            if isinstance(v, dict)
        }
        nodes = {
            k: DependencyNode(**v)
            for k, v in (ps_raw.get("nodes") or {}).items()
            if isinstance(v, dict)
        }
        verified_facts = {
            k: VerifiedFact(**v)
            for k, v in (ps_raw.get("verified_facts") or {}).items()
            if isinstance(v, dict)
        }
        revision_log = [
            RevisionRecord(**r)
            for r in (ps_raw.get("revision_log") or [])
            if isinstance(r, dict)
        ]
        invalidated_statements = [
            a.statement for a in assumptions.values() if a.status == "invalidated"
        ]
        pyramid_state = PyramidState(
            problem_id=str(ps_raw.get("problem_id") or problem.id),
            problem_text=problem.initial_prompt,
            evidence=list(ps_raw.get("evidence") or []),
            objective=str(ps_raw.get("objective") or ""),
            core_constraints=list(ps_raw.get("core_constraints") or []),
            assumptions=assumptions,
            nodes=nodes,
            verified_facts=verified_facts,
            candidate_answer=ps_raw.get("candidate_answer"),
            current_depth=int(ps_raw.get("current_depth", 0)),
            reconstruction_count=int(ps_raw.get("reconstruction_count", 0)),
            revision_log=revision_log,
        )
    mode_name: ModeName = row["mode"]
    res = RunResult(
        problem_id=problem.id,
        mode=mode_name,
        initial_answer=row.get("initial_answer"),
        final_answer=row.get("final_answer"),
        traces=list(row.get("traces") or []),
        budget=budget,
        pyramid_state=pyramid_state,
        invalidated_statements=invalidated_statements,
    )
    score = ProblemScore(**row["score"])
    return problem, res, score


def _write_checkpoint_snapshot(
    output_dir: Path,
    modes: list[ModeName],
    grouped: dict[ModeName, list[tuple[Problem, RunResult, ProblemScore]]],
    cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], Path]:
    """Atomically persist `summary.json` and `report.md` snapshots."""
    summaries = [summarize_mode_results(m, grouped[m]) for m in modes]
    completed_runs = sum(len(grouped[m]) for m in modes)
    cfg_snap = {**cfg, "completed_runs": completed_runs}

    summary_tmp = output_dir / "summary.json.tmp"
    summary_path = output_dir / "summary.json"
    summary_tmp.write_text(
        json.dumps({"config": cfg_snap, "summaries": summaries}, indent=2),
        encoding="utf-8",
    )
    summary_tmp.replace(summary_path)

    report_tmp = output_dir / "report.md.tmp"
    report_path = output_dir / "report.md"
    report_tmp.write_text(
        render_markdown_report(summaries, grouped.get("rpr", []), cfg_snap),
        encoding="utf-8",
    )
    report_tmp.replace(report_path)
    return summaries, report_path


def run_evaluation(
    problems: list[Problem],
    modes: list[ModeName],
    model: ModelClient,
    *,
    output_dir: Path,
    token_budget: int | None = None,
    time_budget: float | None = None,
    per_call_max_tokens: int = 1536,
    temperature: float = 0.0,
    enable_thinking: bool = True,
    use_self_checks: bool = True,
    audit_every_level: bool = False,
    dataset_label: str = "dataset",
    resume: bool = True,
) -> tuple[list[dict[str, Any]], Path]:
    """Run all requested modes over `problems`, persist incremental checkpoints, and return summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "traces.jsonl"
    grouped: dict[ModeName, list[tuple[Problem, RunResult, ProblemScore]]] = defaultdict(list)

    cfg = {
        "model": model.model_name,
        "dataset": dataset_label,
        "problem_count": len(problems),
        "expected_runs": len(problems) * len(modes),
        "token_budget": token_budget,
        "time_budget": time_budget,
        "per_call_max_tokens": per_call_max_tokens,
        "enable_thinking": enable_thinking,
        "use_self_checks": use_self_checks,
        "audit_every_level": audit_every_level,
    }

    saved: dict[tuple[str, str], dict[str, Any]] = {}
    if resume and trace_path.exists():
        saved = _load_checkpoint_rows(trace_path)
        clean_tmp = trace_path.with_suffix(".jsonl.tmp")
        with clean_tmp.open("w", encoding="utf-8") as tmp_fh:
            for row in saved.values():
                tmp_fh.write(json.dumps(row) + "\n")
        clean_tmp.replace(trace_path)
        if saved:
            print(f"[checkpoint] Loaded {len(saved)} completed run(s) from {trace_path}")
    elif not resume and trace_path.exists():
        trace_path.write_text("", encoding="utf-8")

    with trace_path.open("a", encoding="utf-8") as trace_fh:
        for idx, problem in enumerate(problems, start=1):
            for mode in modes:
                if resume and (problem.id, mode) in saved:
                    rec = _rebuild_record_from_row(
                        saved[(problem.id, mode)],
                        problem,
                        token_budget=token_budget,
                        time_budget=time_budget,
                    )
                    grouped[mode].append(rec)
                    _, res, score = rec
                    print(
                        f"[{idx}/{len(problems)}] {problem.id} | {mode:7s} | "
                        f"RESUMED from checkpoint | exact={score.final_exact_correct!s:5s} | "
                        f"toks={res.budget.total_tokens:4d} | pred={res.final_answer}"
                    )
                    continue

                res = run_mode(
                    mode,
                    problem,
                    model,
                    max_total_tokens=token_budget,
                    max_wall_clock_sec=time_budget,
                    per_call_max_tokens=per_call_max_tokens,
                    temperature=temperature,
                    enable_thinking=enable_thinking,
                    use_self_checks=use_self_checks,
                    audit_every_level=audit_every_level,
                )
                rev_log = (
                    [asdict(r) for r in res.pyramid_state.revision_log]
                    if res.pyramid_state
                    else None
                )
                score = score_problem_run(
                    problem,
                    mode,
                    initial_answer=res.initial_answer,
                    final_answer=res.final_answer,
                    revision_log=rev_log,
                    invalidated_statements=res.invalidated_statements,
                )
                grouped[mode].append((problem, res, score))

                row = {
                    "problem_id": problem.id,
                    "category": problem.category,
                    "domain": problem.domain,
                    "mode": mode,
                    "initial_answer": res.initial_answer,
                    "final_answer": res.final_answer,
                    "expected_initial": problem.expected_initial,
                    "expected_final": problem.expected_final,
                    "score": asdict(score),
                    "budget": {
                        "prompt_tokens": res.budget.prompt_tokens,
                        "generation_tokens": res.budget.generation_tokens,
                        "total_tokens": res.budget.total_tokens,
                        "wall_clock_sec": round(res.budget.wall_clock_sec, 4),
                        "model_calls": res.budget.model_calls,
                        "peak_memory_gb": round(res.budget.peak_memory_gb, 4),
                        "budget_exhausted": res.budget.budget_exhausted,
                    },
                    "pyramid_state": res.pyramid_state.to_dict() if res.pyramid_state else None,
                    "traces": res.traces,
                }
                trace_fh.write(json.dumps(row) + "\n")
                trace_fh.flush()
                _write_checkpoint_snapshot(output_dir, modes, grouped, cfg)
                print(
                    f"[{idx}/{len(problems)}] {problem.id} | {mode:7s} | "
                    f"exact={score.final_exact_correct!s:5s} | "
                    f"toks={res.budget.total_tokens:4d} | "
                    f"calls={res.budget.model_calls} | "
                    f"pred={res.final_answer}"
                )

    return _write_checkpoint_snapshot(output_dir, modes, grouped, cfg)


def _default_mock_responder(messages: list[dict[str, str]]) -> str:
    """Minimal fallback responder when --mock is used on CLI."""
    text = "\n".join(m["content"] for m in messages)
    if "strict logical consistency auditor" in text:
        return json.dumps(
            {
                "invalidated": [],
                "updated": {},
                "reason": "Mock audit saw no conflict.",
            }
        )
    if "Level 1 (Foundation)" in text:
        return json.dumps(
            {
                "assumptions": [{"id": "A1", "statement": "Initial assumption"}],
                "facts": [{"id": "F1", "statement": "Invariant fact", "expr": "6 * 10", "value": 60, "depends_on": []}],
            }
        )
    if "Level 2 (Decomposition" in text:
        return json.dumps(
            {
                "nodes": [{"id": "L2-1", "kind": "strategy", "statement": "Filter options", "depends_on": ["A1"]}],
            }
        )
    return json.dumps({"steps": ["Check options"], "primary": "OPT-A", "preserved": 60})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Direct, Recheck, Restart, and RPR.")
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).resolve().parent / "data" / "pilot_problems.jsonl"),
        help="Path to JSONL problem dataset.",
    )
    parser.add_argument(
        "--split",
        choices=["all", "pilot", "dev", "test"],
        default="all",
        help="Filter problems by split.",
    )
    parser.add_argument(
        "--category",
        choices=["all", "premise_change", "irrelevant_update", "ordinary"],
        default="all",
        help="Filter problems by category.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max problems to run.")
    parser.add_argument("--ids", default=None, help="Comma-separated problem IDs to run.")
    parser.add_argument(
        "--modes",
        default="direct,recheck,restart,rpr",
        help="Comma-separated modes to compare (direct,recheck,restart,rpr).",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Local MLX model path or HF repo.")
    parser.add_argument("--adapter", default=None, help="Optional MLX LoRA adapter directory.")
    parser.add_argument("--token-budget", type=int, default=28672, help="Matched max total tokens per problem.")
    parser.add_argument("--time-budget", type=float, default=None, help="Optional max wall-clock seconds per problem.")
    parser.add_argument("--per-call-max-tokens", type=int, default=1536, help="Max generation tokens per call.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--no-thinking", action="store_true", help="Disable <think> reasoning generation (enabled by default).")
    parser.add_argument("--no-self-checks", action="store_true", help="Disable zero-leakage AST math & self-consistency checks.")
    parser.add_argument("--audit-every-level", action="store_true", help="Run LLM audit after Level 2 as well as Level 3.")
    parser.add_argument("--mock", action="store_true", help="Use mock model (pipeline smoke test only).")
    parser.add_argument("--output-dir", default=None, help="Directory to store traces and report.")
    parser.add_argument("--fresh", action="store_true", help="Ignore existing checkpoints in --output-dir and overwrite from scratch.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    problems = load_problems(args.dataset)
    if args.split != "all":
        problems = [p for p in problems if p.split == args.split]
    if args.category != "all":
        problems = [p for p in problems if p.category == args.category]
    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
        problems = [p for p in problems if p.id in wanted]
    if args.limit is not None:
        problems = problems[: args.limit]
    if not problems:
        print("No problems matched the filters.", file=sys.stderr)
        sys.exit(1)

    modes: list[ModeName] = [m.strip() for m in args.modes.split(",") if m.strip()]  # type: ignore[misc]
    for m in modes:
        if m not in ALL_MODES:
            raise ValueError(f"Invalid mode '{m}'. Choose from {ALL_MODES}.")

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(__file__).resolve().parent / "results" / f"run_{stamp}"

    enable_thinking = not args.no_thinking
    resume = not args.fresh
    all_cached = False
    if not args.mock and resume and (out_dir / "traces.jsonl").exists():
        saved = _load_checkpoint_rows(out_dir / "traces.jsonl")
        all_cached = all((p.id, m) in saved for p in problems for m in modes)

    model_label = f"{args.model} + adapter:{Path(args.adapter).name}" if args.adapter else args.model
    model: ModelClient = (
        MockModelClient(responder=_default_mock_responder)
        if args.mock
        else (
            MockModelClient(responder=_default_mock_responder, model_name=model_label)
            if all_cached
            else MLXModelClient(
                args.model,
                adapter_path=args.adapter,
                default_enable_thinking=enable_thinking,
            )
        )
    )

    summaries, report_path = run_evaluation(
        problems,
        modes,
        model,
        output_dir=out_dir,
        token_budget=args.token_budget,
        time_budget=args.time_budget,
        per_call_max_tokens=args.per_call_max_tokens,
        temperature=args.temperature,
        enable_thinking=enable_thinking,
        use_self_checks=not args.no_self_checks,
        audit_every_level=args.audit_every_level,
        dataset_label=Path(args.dataset).name,
        resume=resume,
    )
    print("\n=== Summary ===")
    print(json.dumps(summaries, indent=2))
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
