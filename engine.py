"""Four-mode reasoning engine: Direct (A), Recheck (B), Restart (C), and RPR (D)."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
import subprocess
import sys
import time
from typing import Any, Literal

from model import (
    ModelClient,
    ModelResponse,
    build_recovery_messages,
    extract_json_object,
    needs_close_think_recovery,
    recover_close_think_json,
    split_thinking,
)
from revision import reconstruct, verify_from_level_one
from state import Assumption, DependencyNode, PyramidState, VerifiedFact
from symbolic_critic import (
    extract_l2_bounds,
    parse_prompt_options,
    repair_l2_cross_bindings,
    repair_l2_from_l1_assumptions,
    verify_options_against_bounds,
)
from verifiers import Problem, parse_candidate_answer, safe_eval_arith


ModeName = Literal["direct", "direct_tir", "recheck", "restart", "rpr"]

_TIR_SYSTEM_TOOLS = (
    "You are a careful analytical problem solver with access to a deterministic arithmetic calculator.\n"
    "<tools>\n"
    '[{"name": "calc", "description": "Evaluate a pure arithmetic expression using numbers and + - * / ( ) only.", '
    '"parameters": {"type": "object", "properties": {"expr": {"type": "string"}}, "required": ["expr"]}}]\n'
    "</tools>\n"
    "Before answering, you may call `calc` to verify exact ledger sums or constraint bounds by emitting:\n"
    "<tool_call>\n<function=calc>\n<parameter=expr>\n<numbers and + - * / ( ) only>\n</parameter>\n</function>\n</tool_call>\n"
    "Once verified (or if no calculation tool call is needed), output the final JSON.\n"
)
_TIR_CALL_RE = re.compile(
    r"<tool_call>\s*<function=calc>\s*<parameter=expr>\s*(.*?)\s*</parameter>\s*</function>\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
_PY_EXEC_CALL_RE = re.compile(
    r"<tool_call>\s*<function=python_exec>\s*<parameter=code>\s*(.*?)\s*</parameter>\s*</function>\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)

_ANSWER_FORMAT_NOTE = (
    'End with a valid JSON object containing exact keys "primary" and "preserved", '
    'formatted as {"primary": "OPT-X", "preserved": <integer>} where OPT-X is one of '
    'OPT-A, OPT-B, OPT-C, OPT-D, OPT-E, OPT-F and <integer> is a single evaluated number (never an expression).'
)


@dataclass
class BudgetTracker:
    """Tracks token, call, memory, and wall-clock consumption under optional caps."""

    max_total_tokens: int | None = None
    max_wall_clock_sec: float | None = None
    start_time: float = field(default_factory=time.perf_counter)
    prompt_tokens: int = 0
    generation_tokens: int = 0
    total_tokens: int = 0
    wall_clock_sec: float = 0.0
    model_calls: int = 0
    peak_memory_gb: float = 0.0
    budget_exhausted: bool = False

    def can_continue(self) -> bool:
        elapsed = time.perf_counter() - self.start_time
        if self.max_wall_clock_sec is not None and elapsed >= self.max_wall_clock_sec:
            self.budget_exhausted = True
            return False
        if self.max_total_tokens is not None and self.total_tokens >= self.max_total_tokens:
            self.budget_exhausted = True
            return False
        return True

    def allowed_generation_tokens(self, requested: int, messages: list[dict[str, str]]) -> int:
        if not self.can_continue():
            return 0
        if self.max_total_tokens is None:
            return requested
        est_prompt = int(sum(len(m["content"].split()) * 1.4 for m in messages)) + 32
        remaining = self.max_total_tokens - self.total_tokens - est_prompt
        if remaining <= 16:
            self.budget_exhausted = True
            return 0
        return min(requested, remaining)

    def record(self, resp: ModelResponse) -> None:
        self.prompt_tokens += resp.stats.prompt_tokens
        self.generation_tokens += resp.stats.generation_tokens
        self.total_tokens += resp.stats.total_tokens
        self.wall_clock_sec += resp.stats.wall_clock_sec
        self.model_calls += 1
        self.peak_memory_gb = max(self.peak_memory_gb, resp.stats.peak_memory_gb)
        if self.max_total_tokens is not None and self.total_tokens >= self.max_total_tokens:
            self.budget_exhausted = True

    def finalize_wall_clock(self) -> None:
        self.wall_clock_sec = max(self.wall_clock_sec, time.perf_counter() - self.start_time)


@dataclass
class RunResult:
    """Complete execution trace and telemetry for one problem and one mode."""

    problem_id: str
    mode: ModeName
    initial_answer: dict[str, Any] | None
    final_answer: dict[str, Any] | None
    traces: list[dict[str, Any]]
    budget: BudgetTracker
    pyramid_state: PyramidState | None = None
    invalidated_statements: list[str] = field(default_factory=list)


def _guarded_generate(
    model: ModelClient,
    budget: BudgetTracker,
    messages: list[dict[str, str]],
    *,
    stage: str,
    traces: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
    recovery_instruction: str = (
        "Output ONLY a compact valid JSON object summarizing your conclusion above "
        "(use evaluated numbers, not arithmetic expressions)."
    ),
    recovery_max_tokens: int = 512,
) -> ModelResponse | None:
    allowed = budget.allowed_generation_tokens(max_tokens, messages)
    if allowed <= 0:
        traces.append({"stage": stage, "skipped": "budget_exhausted"})
        return None
    resp = model.generate(
        messages,
        max_tokens=allowed,
        temperature=temperature,
        enable_thinking=enable_thinking,
    )
    budget.record(resp)
    traces.append(
        {
            "stage": stage,
            "prompt_tokens": resp.stats.prompt_tokens,
            "generation_tokens": resp.stats.generation_tokens,
            "wall_clock_sec": round(resp.stats.wall_clock_sec, 4),
            "finish_reason": resp.stats.finish_reason,
            "output": resp.raw_text,
        }
    )
    if needs_close_think_recovery(resp, enable_thinking=enable_thinking) and budget.can_continue():
        thinking_body = resp.thinking_text or resp.raw_text
        rec_messages = build_recovery_messages(messages, thinking_body, recovery_instruction)
        rec_allowed = budget.allowed_generation_tokens(recovery_max_tokens, rec_messages)
        if rec_allowed > 16:
            merged, rec_resp = recover_close_think_json(
                model,
                messages,
                resp,
                max_tokens=rec_allowed,
                temperature=temperature,
                instruction=recovery_instruction,
            )
            budget.record(rec_resp)
            traces.append(
                {
                    "stage": f"{stage}_close_think_recovery",
                    "prompt_tokens": rec_resp.stats.prompt_tokens,
                    "generation_tokens": rec_resp.stats.generation_tokens,
                    "wall_clock_sec": round(rec_resp.stats.wall_clock_sec, 4),
                    "finish_reason": rec_resp.stats.finish_reason,
                    "output": rec_resp.raw_text,
                }
            )
            return merged
    return resp


def run_mode(
    mode: ModeName,
    problem: Problem,
    model: ModelClient,
    *,
    max_total_tokens: int | None = None,
    max_wall_clock_sec: float | None = None,
    per_call_max_tokens: int = 1536,
    temperature: float = 0.0,
    enable_thinking: bool = True,
    use_self_checks: bool = True,
    audit_every_level: bool = False,
    max_reconstructions: int = 2,
) -> RunResult:
    """Dispatch execution to Mode A/B/C/D using shared telemetry and budgets."""
    budget = BudgetTracker(
        max_total_tokens=max_total_tokens,
        max_wall_clock_sec=max_wall_clock_sec,
    )
    if mode == "direct":
        res = _run_direct(
            problem, model, budget,
            max_tokens=per_call_max_tokens,
            temperature=temperature,
            enable_thinking=enable_thinking,
        )
    elif mode == "direct_tir":
        res = _run_direct_tir(
            problem, model, budget,
            max_tokens=per_call_max_tokens,
            temperature=temperature,
            enable_thinking=enable_thinking,
        )
    elif mode == "recheck":
        res = _run_recheck(
            problem, model, budget,
            max_tokens=per_call_max_tokens,
            temperature=temperature,
            enable_thinking=enable_thinking,
        )
    elif mode == "restart":
        res = _run_restart(
            problem, model, budget,
            max_tokens=per_call_max_tokens,
            temperature=temperature,
            enable_thinking=enable_thinking,
        )
    elif mode == "rpr":
        res = _run_rpr(
            problem, model, budget,
            max_tokens=per_call_max_tokens,
            temperature=temperature,
            enable_thinking=enable_thinking,
            use_self_checks=use_self_checks,
            audit_every_level=audit_every_level,
            max_reconstructions=max_reconstructions,
        )
    else:
        raise ValueError(f"Unknown reasoning mode: {mode}")
    budget.finalize_wall_clock()
    return res


def _run_direct(
    problem: Problem,
    model: ModelClient,
    budget: BudgetTracker,
    *,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
) -> RunResult:
    traces: list[dict[str, Any]] = []
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": "You are a careful analytical problem solver. Follow all constraints strictly.\n" + _ANSWER_FORMAT_NOTE,
        },
        {"role": "user", "content": problem.initial_prompt},
    ]
    resp1 = _guarded_generate(
        model, budget, messages,
        stage="phase1_direct", traces=traces,
        max_tokens=max_tokens, temperature=temperature, enable_thinking=enable_thinking,
    )
    initial_answer = parse_candidate_answer(resp1.raw_text) if resp1 else None
    final_answer = initial_answer

    if problem.midway_update and resp1 is not None:
        messages.append({"role": "assistant", "content": resp1.content_text or resp1.raw_text})
        messages.append(
            {
                "role": "user",
                "content": f"New evidence arrived:\n{problem.midway_update}\n\nProvide your updated answer.\n{_ANSWER_FORMAT_NOTE}",
            }
        )
        resp2 = _guarded_generate(
            model, budget, messages,
            stage="phase2_direct_update", traces=traces,
            max_tokens=max_tokens, temperature=temperature, enable_thinking=enable_thinking,
        )
        if resp2 is not None:
            parsed2 = parse_candidate_answer(resp2.raw_text)
            if parsed2 is not None:
                final_answer = parsed2

    return RunResult(
        problem_id=problem.id,
        mode="direct",
        initial_answer=initial_answer,
        final_answer=final_answer,
        traces=traces,
        budget=budget,
    )


def _execute_tir_phase(
    messages: list[dict[str, str]],
    model: ModelClient,
    budget: BudgetTracker,
    traces: list[dict[str, Any]],
    *,
    stage_prefix: str,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
    max_tool_rounds: int = 2,
) -> tuple[ModelResponse | None, dict[str, Any] | None]:
    """Run up to `max_tool_rounds` sandboxed `calc(expr)` calls before parsing the final JSON answer."""
    last_resp: ModelResponse | None = None
    for r_idx in range(max_tool_rounds + 1):
        resp = _guarded_generate(
            model, budget, messages,
            stage=f"{stage_prefix}_r{r_idx}", traces=traces,
            max_tokens=max_tokens, temperature=temperature, enable_thinking=enable_thinking,
        )
        if resp is None:
            break
        last_resp = resp
        body = resp.content_text or resp.raw_text
        exprs = [e.strip() for e in _TIR_CALL_RE.findall(body) if e.strip()]
        py_snippets = [c.strip() for c in _PY_EXEC_CALL_RE.findall(body) if c.strip()]
        if (not exprs and not py_snippets) or r_idx == max_tool_rounds:
            parsed = parse_candidate_answer(resp.raw_text)
            return resp, parsed

        results_payload = []
        for ex in exprs[:4]:
            val = safe_eval_arith(ex)
            if val is not None:
                clean_v: int | float = int(round(val)) if abs(val - round(val)) <= 1e-9 else round(val, 6)
                results_payload.append({"expr": ex, "result": clean_v})
            else:
                results_payload.append({"expr": ex, "error": "Use pure numbers and + - * / ( ) only."})
        for code_src in py_snippets[:2]:
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", "-c", code_src],
                    capture_output=True,
                    text=True,
                    timeout=2.5,
                    check=False,
                )
                if proc.returncode == 0:
                    results_payload.append({"python_stdout": (proc.stdout or "").strip()[:800]})
                else:
                    results_payload.append({"python_error": (proc.stderr or proc.stdout or "Exit != 0").strip()[-600:]})
            except Exception as exc:
                results_payload.append({"python_error": str(exc)[:400]})

        messages.append({"role": "assistant", "content": body.strip()})
        messages.append(
            {
                "role": "user",
                "content": (
                    "<tool_response>\n"
                    + json.dumps(results_payload if len(results_payload) > 1 else results_payload[0])
                    + "\n</tool_response>\nNow finish solving and output the verified final JSON.\n"
                    + _ANSWER_FORMAT_NOTE
                ),
            }
        )
    return last_resp, (parse_candidate_answer(last_resp.raw_text) if last_resp else None)


def _run_direct_tir(
    problem: Problem,
    model: ModelClient,
    budget: BudgetTracker,
    *,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
) -> RunResult:
    """Mode E (`direct_tir`): Tool-Integrated Reasoning with zero-leakage sandboxed `calc(expr)`."""
    traces: list[dict[str, Any]] = []
    messages: list[dict[str, str]] = [
        {"role": "system", "content": _TIR_SYSTEM_TOOLS + _ANSWER_FORMAT_NOTE},
        {"role": "user", "content": problem.initial_prompt},
    ]
    resp1, initial_answer = _execute_tir_phase(
        messages, model, budget, traces,
        stage_prefix="phase1_tir", max_tokens=max_tokens,
        temperature=temperature, enable_thinking=enable_thinking,
    )
    final_answer = initial_answer

    if problem.midway_update and resp1 is not None:
        messages.append({"role": "assistant", "content": (resp1.content_text or resp1.raw_text).strip()})
        messages.append(
            {
                "role": "user",
                "content": f"New evidence arrived:\n{problem.midway_update}\n\nProvide your updated answer (you may call `calc` first if helpful).\n{_ANSWER_FORMAT_NOTE}",
            }
        )
        _, parsed2 = _execute_tir_phase(
            messages, model, budget, traces,
            stage_prefix="phase2_tir_update", max_tokens=max_tokens,
            temperature=temperature, enable_thinking=enable_thinking,
        )
        if parsed2 is not None:
            final_answer = parsed2

    return RunResult(
        problem_id=problem.id,
        mode="direct_tir",
        initial_answer=initial_answer,
        final_answer=final_answer,
        traces=traces,
        budget=budget,
    )


def _run_recheck(
    problem: Problem,
    model: ModelClient,
    budget: BudgetTracker,
    *,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
) -> RunResult:
    traces: list[dict[str, Any]] = []
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": "You are a careful analytical problem solver. Follow all constraints strictly.\n" + _ANSWER_FORMAT_NOTE,
        },
        {"role": "user", "content": problem.initial_prompt},
    ]
    resp1 = _guarded_generate(
        model, budget, messages,
        stage="phase1_solve", traces=traces,
        max_tokens=max_tokens, temperature=temperature, enable_thinking=enable_thinking,
    )
    initial_answer = parse_candidate_answer(resp1.raw_text) if resp1 else None

    if resp1 is not None:
        messages.append({"role": "assistant", "content": resp1.content_text or resp1.raw_text})
        messages.append(
            {
                "role": "user",
                "content": (
                    "Review your previous solution step by step. Check every constraint and calculation. "
                    "Fix any mistakes and output the verified final JSON.\n" + _ANSWER_FORMAT_NOTE
                ),
            }
        )
        resp_rev1 = _guarded_generate(
            model, budget, messages,
            stage="phase1_recheck", traces=traces,
            max_tokens=max_tokens, temperature=temperature, enable_thinking=enable_thinking,
        )
        if resp_rev1 is not None:
            parsed_rev1 = parse_candidate_answer(resp_rev1.raw_text)
            if parsed_rev1 is not None:
                initial_answer = parsed_rev1
            messages.append(
                {"role": "assistant", "content": resp_rev1.content_text or resp_rev1.raw_text}
            )

    final_answer = initial_answer
    if problem.midway_update and resp1 is not None:
        messages.append(
            {
                "role": "user",
                "content": f"New evidence arrived:\n{problem.midway_update}\n\nUpdate your solution.\n{_ANSWER_FORMAT_NOTE}",
            }
        )
        resp2 = _guarded_generate(
            model, budget, messages,
            stage="phase2_update", traces=traces,
            max_tokens=max_tokens, temperature=temperature, enable_thinking=enable_thinking,
        )
        if resp2 is not None:
            parsed2 = parse_candidate_answer(resp2.raw_text)
            if parsed2 is not None:
                final_answer = parsed2
            messages.append({"role": "assistant", "content": resp2.content_text or resp2.raw_text})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Recheck your updated solution against ALL constraints (both unchanged rules and new evidence). "
                        "Ensure unchanged sub-answers were not accidentally altered.\n" + _ANSWER_FORMAT_NOTE
                    ),
                }
            )
            resp_rev2 = _guarded_generate(
                model, budget, messages,
                stage="phase2_recheck", traces=traces,
                max_tokens=max_tokens, temperature=temperature, enable_thinking=enable_thinking,
            )
            if resp_rev2 is not None:
                parsed_rev2 = parse_candidate_answer(resp_rev2.raw_text)
                if parsed_rev2 is not None:
                    final_answer = parsed_rev2

    return RunResult(
        problem_id=problem.id,
        mode="recheck",
        initial_answer=initial_answer,
        final_answer=final_answer,
        traces=traces,
        budget=budget,
    )


def _run_restart(
    problem: Problem,
    model: ModelClient,
    budget: BudgetTracker,
    *,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
) -> RunResult:
    traces: list[dict[str, Any]] = []
    init_messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": "You are a careful analytical problem solver. Follow all constraints strictly.\n" + _ANSWER_FORMAT_NOTE,
        },
        {"role": "user", "content": problem.initial_prompt},
    ]
    resp1 = _guarded_generate(
        model, budget, init_messages,
        stage="phase1_initial", traces=traces,
        max_tokens=max_tokens, temperature=temperature, enable_thinking=enable_thinking,
    )
    initial_answer = parse_candidate_answer(resp1.raw_text) if resp1 else None
    final_answer = initial_answer

    if problem.midway_update:
        # Discard Phase 1 conversation entirely and re-solve from scratch with all evidence.
        fresh_prompt = (
            f"{problem.initial_prompt}\n\n"
            f"IMPORTANT UPDATE / NEW EVIDENCE (supersedes conflicting earlier assumptions):\n"
            f"{problem.midway_update}"
        )
        restart_messages = [
            {
                "role": "system",
                "content": "You are a careful analytical problem solver. Solve cleanly from scratch using all current constraints.\n" + _ANSWER_FORMAT_NOTE,
            },
            {"role": "user", "content": fresh_prompt},
        ]
        resp2 = _guarded_generate(
            model, budget, restart_messages,
            stage="phase2_restart_fresh", traces=traces,
            max_tokens=max_tokens, temperature=temperature, enable_thinking=enable_thinking,
        )
        if resp2 is not None:
            parsed2 = parse_candidate_answer(resp2.raw_text)
            if parsed2 is not None:
                final_answer = parsed2

    return RunResult(
        problem_id=problem.id,
        mode="restart",
        initial_answer=initial_answer,
        final_answer=final_answer,
        traces=traces,
        budget=budget,
    )


def _run_rpr(
    problem: Problem,
    model: ModelClient,
    budget: BudgetTracker,
    *,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
    use_self_checks: bool,
    audit_every_level: bool,
    max_reconstructions: int,
) -> RunResult:
    traces: list[dict[str, Any]] = []
    state = PyramidState(
        problem_id=problem.id,
        problem_text=problem.initial_prompt,
        evidence=["Initial problem statement."],
        max_reconstructions=max_reconstructions,
    )

    _build_pyramid_to_depth(
        state, model, budget, traces,
        target_depth=3,
        phase="phase1",
        max_tokens=max_tokens,
        temperature=temperature,
        enable_thinking=enable_thinking,
        use_self_checks=use_self_checks,
        audit_every_level=audit_every_level,
    )
    initial_answer = dict(state.candidate_answer) if state.candidate_answer else None

    if problem.midway_update and budget.can_continue():
        state.evidence.append(problem.midway_update)

        findings, audit_resp = _run_verification_step(
            state, model, budget, traces,
            stage="phase2_verify_from_l1",
            use_self_checks=use_self_checks,
            run_model_audit=True,
            max_tokens=min(max_tokens, 512),
            temperature=temperature,
            enable_thinking=enable_thinking,
        )
        if findings.foundation_invalid:
            if max_reconstructions > 0 and findings.invalidated_assumptions:
                state.max_reconstructions = max(
                    state.max_reconstructions, state.reconstruction_count + 2
                )
            state, rebuilt = reconstruct(state, findings, phase="midway_update")
            if rebuilt and budget.can_continue():
                _build_pyramid_to_depth(
                    state, model, budget, traces,
                    target_depth=3,
                    phase="phase2_rebuild",
                    max_tokens=max_tokens,
                    temperature=temperature,
                    enable_thinking=enable_thinking,
                    use_self_checks=use_self_checks,
                    audit_every_level=audit_every_level,
                )
        else:
            traces.append({"stage": "phase2_no_rebuild", "reason": findings.trigger_reason})

    final_answer = dict(state.candidate_answer) if state.candidate_answer else initial_answer
    invalidated_statements = [
        a.statement for a in state.assumptions.values() if a.status == "invalidated"
    ]
    return RunResult(
        problem_id=problem.id,
        mode="rpr",
        initial_answer=initial_answer,
        final_answer=final_answer,
        traces=traces,
        budget=budget,
        pyramid_state=state,
        invalidated_statements=invalidated_statements,
    )


def _run_verification_step(
    state: PyramidState,
    model: ModelClient,
    budget: BudgetTracker,
    traces: list[dict[str, Any]],
    *,
    stage: str,
    use_self_checks: bool,
    run_model_audit: bool,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
):
    allowed = (
        budget.allowed_generation_tokens(max_tokens, [{"content": state.problem_text}])
        if run_model_audit
        else 0
    )
    rec_budget = (
        max(0, (budget.max_total_tokens - budget.total_tokens - allowed - 300))
        if budget.max_total_tokens is not None
        else 256
    )
    findings, resp = verify_from_level_one(
        state,
        model if allowed > 32 else None,
        use_self_checks=use_self_checks,
        run_model_audit=run_model_audit and allowed > 32,
        max_tokens=max(0, allowed),
        temperature=temperature,
        enable_thinking=enable_thinking,
        recovery_budget_tokens=min(256, rec_budget),
    )
    if resp is not None:
        budget.record(resp)
    traces.append(
        {
            "stage": stage,
            "foundation_invalid": findings.foundation_invalid,
            "invalidated_assumptions": findings.invalidated_assumptions,
            "violated_constraints": findings.violated_constraints,
            "trigger_reason": findings.trigger_reason,
            "preserved_facts": findings.preserved_facts,
            "output": resp.raw_text if resp else None,
        }
    )
    return findings, resp


_L1_ASSUMPTION_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(A\d+)\s*:\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)
_L1_FACT_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(F\d+)\s*(?:\[([^\]]*)\])?\s*:\s*([^=|\n]+?)(?:=[^|\n]*)?(?:\|\s*(.+))?$",
    re.IGNORECASE | re.MULTILINE,
)
_L2_NODE_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(L2-\d+)\s*(?:\[([^\]]*)\])?\s*:\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)


def _parse_deps(raw_bracket: str | None) -> list[str]:
    """Parse comma-separated dependency IDs from `[A1, A2]` or `[NONE]`."""
    if not raw_bracket:
        return []
    deps: list[str] = []
    for part in raw_bracket.split(","):
        cleaned = part.strip()
        if not cleaned or cleaned.upper() in {"NONE", "INDEPENDENT", "N/A"}:
            continue
        deps.append(cleaned)
    return deps


_PURE_ARITH_TAIL_RE = re.compile(
    r"([(\d][\d.+\-*/()\s]*[+\-*/][\d.+\-*/()\s]*[\d)])\s*$"
)
_MULTI_OP_LEDGER_RE = re.compile(
    r"(\d+\s*\*\s*\d+(?:\s*[+\-]\s*\d+(?:\s*\*\s*\d+)?)+)"
)


def _extract_pure_arith_expr(raw_expr: str) -> str:
    """Strip leading prose labels (e.g. `invariant ledger calculation: `) from an F1 math expression."""
    cleaned = raw_expr.strip()
    if safe_eval_arith(cleaned) is not None:
        return cleaned
    m = _PURE_ARITH_TAIL_RE.search(cleaned)
    if m and safe_eval_arith(m.group(1).strip()) is not None:
        return m.group(1).strip()
    return cleaned


def parse_level1_output(text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse line-oriented Level-1 assumptions & facts, with JSON fallback."""
    thinking, content = split_thinking(text)
    line_assumptions: list[dict[str, Any]] = []
    for raw_id, stmt in _L1_ASSUMPTION_RE.findall(content):
        cleaned = stmt.strip()
        if cleaned:
            line_assumptions.append({"id": raw_id.upper(), "statement": cleaned})

    line_facts: list[dict[str, Any]] = []
    for raw_id, raw_deps, raw_expr, raw_label in _L1_FACT_RE.findall(content):
        raw_trimmed = raw_expr.strip()
        expr = _extract_pure_arith_expr(raw_trimmed)
        label = raw_label.strip() if raw_label else raw_trimmed
        line_facts.append(
            {
                "id": raw_id.upper(),
                "statement": label,
                "expr": expr,
                "depends_on": _parse_deps(raw_deps),
            }
        )

    for fact in line_facts:
        has_op = any(op in fact["expr"] for op in "+-*/")
        if safe_eval_arith(fact["expr"]) is None or not has_op:
            recovered: str | None = None
            for item in line_assumptions:
                m_led = _MULTI_OP_LEDGER_RE.search(item["statement"])
                if m_led and safe_eval_arith(m_led.group(1).strip()) is not None:
                    recovered = m_led.group(1).strip()
                    break
            if recovered is None and thinking:
                for cand in reversed(_MULTI_OP_LEDGER_RE.findall(thinking)):
                    if safe_eval_arith(cand.strip()) is not None:
                        recovered = cand.strip()
                        break
            if recovered is not None:
                fact["expr"] = recovered

    if not line_facts and line_assumptions:
        for blob in (content, thinking):
            if not blob:
                continue
            for cand in reversed(_MULTI_OP_LEDGER_RE.findall(blob)):
                if safe_eval_arith(cand.strip()) is not None:
                    line_facts.append(
                        {
                            "id": "F1",
                            "statement": "invariant ledger",
                            "expr": cand.strip(),
                            "depends_on": [],
                        }
                    )
                    break
            if line_facts:
                break

    if line_assumptions or line_facts:
        return line_assumptions, line_facts

    parsed = extract_json_object(text)
    if parsed and isinstance(parsed.get("assumptions"), list):
        assumptions = [
            item if isinstance(item, dict) else {"id": f"A{i}", "statement": str(item).strip()}
            for i, item in enumerate(parsed["assumptions"], start=1)
        ]
        raw_facts = parsed.get("facts")
        if raw_facts is None:
            raw_facts = parsed.get("verified_facts")
        facts = [
            item if isinstance(item, dict) else {"id": f"F{i}", "statement": str(item).strip(), "depends_on": []}
            for i, item in enumerate(raw_facts or [], start=1)
        ]
        return assumptions, facts
    return [], []


def parse_level2_output(text: str) -> list[dict[str, Any]]:
    """Parse line-oriented Level-2 decomposition nodes, with JSON fallback."""
    _, content = split_thinking(text)
    line_nodes: list[dict[str, Any]] = []
    for raw_id, raw_deps, raw_stmt in _L2_NODE_RE.findall(content):
        stmt = raw_stmt.strip()
        if stmt:
            line_nodes.append(
                {
                    "id": raw_id.upper(),
                    "kind": "strategy",
                    "statement": stmt,
                    "depends_on": _parse_deps(raw_deps),
                }
            )
    if line_nodes:
        return line_nodes

    parsed = extract_json_object(text)
    if parsed and isinstance(parsed.get("nodes"), list):
        return [item for item in parsed["nodes"] if isinstance(item, dict)]
    return []


_L3_STEP_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(L3-\d+)\s*:\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)


def parse_level3_output(text: str) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Parse line-oriented L3-1/L3-2 steps + trailing answer JSON, with JSON-steps fallback."""
    _, content = split_thinking(text)
    steps: list[dict[str, Any]] = []
    for raw_id, stmt in _L3_STEP_RE.findall(content):
        cleaned = stmt.strip()
        if cleaned:
            steps.append({"id": raw_id.upper(), "statement": cleaned})

    parsed = extract_json_object(text)
    candidate: dict[str, Any] | None = None
    if parsed and ("primary" in parsed or "preserved" in parsed or "steps" in parsed):
        if not steps:
            for idx, item in enumerate(parsed.get("steps") or [], start=1):
                if isinstance(item, dict):
                    s = str(item.get("statement") or "").strip()
                    deps = item.get("depends_on")
                    if s:
                        steps.append({"id": f"L3-{idx}", "statement": s, "depends_on": deps})
                else:
                    s = str(item).strip()
                    if s:
                        steps.append({"id": f"L3-{idx}", "statement": s})
        candidate = parse_candidate_answer(json.dumps(parsed))
    if candidate is None:
        candidate = parse_candidate_answer(content) or parse_candidate_answer(text)
    return steps, candidate


def _build_pyramid_to_depth(
    state: PyramidState,
    model: ModelClient,
    budget: BudgetTracker,
    traces: list[dict[str, Any]],
    *,
    target_depth: int,
    phase: str,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
    use_self_checks: bool,
    audit_every_level: bool,
) -> None:
    """Advance pyramid Level 1 -> Level 2 -> Level 3 with Level-1 rechecking."""
    safety_iters = 0
    l2_self_corrected = False
    l3_self_corrected = False
    while state.current_depth < target_depth and budget.can_continue():
        safety_iters += 1
        if safety_iters > (target_depth + 2) * (state.max_reconstructions + 1):
            break

        next_depth = state.current_depth + 1
        if next_depth == 1:
            _advance_level_1(
                state, model, budget, traces,
                phase=phase,
                max_tokens=min(max_tokens, 384),
                temperature=temperature,
                enable_thinking=enable_thinking,
            )
        elif next_depth == 2:
            _advance_level_2(
                state, model, budget, traces,
                phase=phase,
                max_tokens=min(max_tokens, 448),
                temperature=temperature,
                enable_thinking=enable_thinking,
                use_self_checks=use_self_checks,
            )
        elif next_depth == 3:
            _advance_level_3(
                state, model, budget, traces,
                phase=phase,
                max_tokens=min(max_tokens, 1536),
                temperature=temperature,
                enable_thinking=enable_thinking,
                use_self_checks=use_self_checks,
            )

        # Revisit from Level 1 after advancing.
        # Run 0-token ATMS & AST math checks unless audit_every_level is enabled.
        do_model_audit = audit_every_level and next_depth >= 2
        findings, _ = _run_verification_step(
            state, model, budget, traces,
            stage=f"{phase}_verify_d{next_depth}",
            use_self_checks=use_self_checks,
            run_model_audit=do_model_audit,
            max_tokens=min(max_tokens, 512),
            temperature=temperature,
            enable_thinking=enable_thinking,
        )
        if findings.foundation_invalid:
            already_invalid = set(state.invalidated_assumption_ids())
            only_l2_unsat = (
                next_depth == 2
                and not findings.invalidated_assumptions
                and not state.compute_tainted_nodes(already_invalid)
                and any("UNSAT_L2_BOUNDS" in v for v in findings.violated_constraints)
                and not l2_self_corrected
            )
            if only_l2_unsat and budget.can_continue():
                l2_self_corrected = True
                _advance_level_2(
                    state, model, budget, traces,
                    phase=f"{phase}_l2_selfcheck",
                    max_tokens=min(max_tokens, 448),
                    temperature=temperature,
                    enable_thinking=enable_thinking,
                    use_self_checks=use_self_checks,
                    correction_note="; ".join(findings.violated_constraints),
                )
                findings2, _ = _run_verification_step(
                    state, model, budget, traces,
                    stage=f"{phase}_verify_d2_retry",
                    use_self_checks=use_self_checks,
                    run_model_audit=do_model_audit,
                    max_tokens=min(max_tokens, 512),
                    temperature=temperature,
                    enable_thinking=enable_thinking,
                )
                if not findings2.foundation_invalid:
                    continue
                findings = findings2
            l3_violations = (
                [v for v in findings.violated_constraints if not v.startswith("UNSAT_L2_BOUNDS")]
                if next_depth == 3
                else list(findings.violated_constraints)
            )
            if (
                next_depth == 3
                and not findings.invalidated_assumptions
                and not state.compute_tainted_nodes(already_invalid)
                and not l3_violations
            ):
                break
            only_l3_math_mismatch = (
                next_depth == 3
                and not findings.invalidated_assumptions
                and not state.compute_tainted_nodes(already_invalid)
                and bool(l3_violations)
                and state.candidate_answer is not None
                and state.candidate_answer.get("primary") is not None
                and not l3_self_corrected
            )
            if only_l3_math_mismatch and budget.can_continue():
                l3_self_corrected = True
                _advance_level_3(
                    state, model, budget, traces,
                    phase=f"{phase}_selfcheck",
                    max_tokens=min(max_tokens, 1536),
                    temperature=temperature,
                    enable_thinking=enable_thinking,
                    use_self_checks=use_self_checks,
                    correction_note="; ".join(l3_violations),
                )
                continue
            state, rebuilt = reconstruct(state, findings, phase=f"{phase}_d{next_depth}")
            if not rebuilt:
                if state.current_depth < target_depth:
                    continue
                break
            if state.current_depth < 2:
                l2_self_corrected = False


_INLINE_EQ_RE = re.compile(
    r"^([^=]+?)=\s*([0-9.+\-*/()\s]+?)(?:=\s*(-?\d+(?:\.\d+)?))?$"
)


def _verify_inline_equations(stmt: str) -> str:
    """AST-verify any `lhs = <pure_arith_expr> [= <num>]` semicolon-separated segments."""
    parts = [p.strip() for p in stmt.split(";")]
    out: list[str] = []
    for part in parts:
        m = _INLINE_EQ_RE.match(part)
        if m:
            lhs, expr, _ = m.groups()
            if any(op in expr for op in "+-*/"):
                val = safe_eval_arith(expr)
                if val is not None:
                    fmt = str(int(round(val))) if float(val).is_integer() else f"{val:g}"
                    out.append(f"{lhs.strip()} = {expr.strip()} = {fmt}")
                    continue
        out.append(part)
    return "; ".join(out)


def _problem_rules_without_options(problem_text: str) -> str:
    """Return problem rules without trailing Tasks / Candidate options sections, if present."""
    for marker in ("\nTasks:", "\nCandidate options"):
        idx = problem_text.find(marker)
        if idx != -1:
            return problem_text[:idx].strip()
    return problem_text


def _problem_options_or_full(problem_text: str) -> str:
    """Return only the Candidate options section when present, else full problem text."""
    marker = "\nCandidate options"
    idx = problem_text.find(marker)
    return problem_text[idx:].strip() if idx != -1 else problem_text


def _advance_level_1(
    state: PyramidState,
    model: ModelClient,
    budget: BudgetTracker,
    traces: list[dict[str, Any]],
    *,
    phase: str,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
) -> None:
    revision_context = ""
    if state.revision_log:
        last = state.revision_log[-1]
        invalidated_lines = "\n".join(
            f"- {a.id} (INVALIDATED): {a.statement}"
            for a in state.assumptions.values()
            if a.status == "invalidated"
        )
        revision_context = (
            f"\nIMPORTANT — Prior pyramid invalidated. Reason: {last.trigger_reason}\n"
            + (f"Invalidated assumptions (DO NOT RESTORE):\n{invalidated_lines}\n" if invalidated_lines else "")
            + "Re-derive ALL A1..A5 assumptions cleanly from Problem + Evidence timeline for the NEW active regime/constraints (update any regime-dependent rates, overheads, penalties, or caps)."
        )

    existing_facts = (
        "\nPreserved verified facts (keep their IDs if valid):\n"
        + "\n".join(f"- {f.id}: {f.statement}" for f in state.verified_facts.values())
        if state.verified_facts
        else ""
    )

    rules_text = _problem_rules_without_options(state.problem_text)
    sys_prompt = (
        "You are Level 1 (Foundation) of a Revisable Reasoning Pyramid.\n"
        "Identify the CURRENT active regime from Problem Rules + Evidence timeline, and plug in ONLY that active regime's concrete numbers (per-unit throughput, regime overhead, regime penalty, and whether its boolean feature is optional or MUST be true).\n"
        "Do NOT emit <tool_call> tags. Keep <think> concise, close </think>, and output ONLY tagged lines:\n"
        "A1: active_mode=<MODE>; boolean_feature=<optional, OR feature_name=true ONLY if active mode explicitly states MUST be true>\n"
        "A2: workload demand is <D>, and under <MODE> each unit handles <C>\n"
        "A3: (required units) * <cost_field> <= <total_budget_cap>\n"
        "A4: <base_latency_or_time_field> + <active_penalty> <= <gross_time_cap> (keep exact gross cap from Problem Rules on RHS; do NOT pre-subtract penalty!)\n"
        "A5: <resource_field> <= <gross_resource_limit> - <active_overhead> (use exact gross resource limit from Problem Rules, NOT A2 throughput)\n"
        "F1 [NONE]: <pure numbers and + - * / ( ) ONLY for invariant ledger> = <integer> | invariant ledger"
    )
    user_prompt = (
        f"Problem Rules:\n{rules_text}\n\n"
        f"Evidence timeline:\n" + "\n".join(f"{i+1}. {e}" for i, e in enumerate(state.evidence))
        + revision_context
        + existing_facts
        + "\n\nLEVEL 1 ONLY: Output A1..A5 with the active regime's exact numbers from Problem Rules plugged in, plus F1 for the invariant ledger calculation."
    )
    resp = _guarded_generate(
        model, budget,
        [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
        stage=f"{phase}_level1", traces=traces,
        max_tokens=max_tokens, temperature=temperature, enable_thinking=enable_thinking,
        recovery_instruction=(
            "Output ONLY tagged lines A1..A5 (with active regime numbers plugged in) and F1 (no JSON, no <tool_call>):\n"
            "A1: active_mode=<MODE>; boolean_feature=<optional OR feature_name=true>\n"
            "A2: demand is <D>, per-unit throughput under <MODE> is <C>\n"
            "A3: (required units) * <cost_field> <= <budget_cap>\n"
            "A4: <base_time> + <active_penalty> <= <time_cap>\n"
            "A5: <resource_field> <= <gross_limit> - <active_overhead>\n"
            "F1 [NONE]: <invariant ledger math expr with numbers and + - * / ( ) only> = <value> | invariant ledger"
        ),
        recovery_max_tokens=256,
    )
    parsed_assumptions, parsed_facts = parse_level1_output(resp.raw_text) if resp else ([], [])
    if parsed_assumptions:
        new_assumptions: dict[str, Assumption] = {}
        for aid, old_a in state.assumptions.items():
            if old_a.status == "invalidated":
                new_assumptions[aid] = old_a
        for idx, item in enumerate(parsed_assumptions, start=1):
            aid = str(item.get("id") or f"A{idx}")
            stmt = str(item.get("statement") or "").strip()
            if stmt:
                if aid in new_assumptions and new_assumptions[aid].status == "invalidated":
                    aid = f"{aid}_new"
                new_assumptions[aid] = Assumption(id=aid, statement=stmt, source="given")
        if new_assumptions:
            state.assumptions = new_assumptions

        for idx, item in enumerate(parsed_facts, start=1):
            fid = str(item.get("id") or f"F{idx}")
            stmt = str(item.get("statement") or "").strip()
            deps = [str(d) for d in (item.get("depends_on") or [])]
            expr_raw = item.get("expr")
            expr = str(expr_raw).strip() if isinstance(expr_raw, str) and expr_raw.strip() else None
            eval_val = safe_eval_arith(expr) if expr else None
            verified_by = "ast_math" if eval_val is not None else "given"
            existing_f = state.verified_facts.get(fid)
            if stmt and (existing_f is None or (existing_f.value is None and eval_val is not None)):
                state.verified_facts[fid] = VerifiedFact(
                    id=fid,
                    statement=stmt,
                    depends_on=deps,
                    verified_by=verified_by,
                    expr=expr,
                    value=eval_val,
                )
    has_verified_f1 = any(
        f.id == "F1" and f.verified_by == "ast_math" and f.value is not None
        for f in state.surviving_facts(set(state.invalidated_assumption_ids()))
    )
    if not has_verified_f1 and "invariant ledger" in rules_text.lower() and budget.can_continue():
        f1_resp = _guarded_generate(
            model, budget,
            [
                {
                    "role": "system",
                    "content": (
                        "Extract ONLY the invariant ledger arithmetic expression from Problem Rules using pure numbers and + - * / ( ). "
                        "Do NOT emit prose or JSON. Output ONE tagged line:\n"
                        "F1 [NONE]: <pure numbers and + - * / ( )> = <integer> | invariant ledger"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Problem Rules:\n{rules_text}\n\nOutput ONLY the F1 [NONE]: line for the invariant ledger calculation.",
                },
            ],
            stage=f"{phase}_level1_f1",
            traces=traces,
            max_tokens=min(max_tokens, 128),
            temperature=temperature,
            enable_thinking=False,
        )
        if f1_resp is not None:
            _, extra_facts = parse_level1_output(f1_resp.raw_text)
            for item in extra_facts:
                expr_raw = item.get("expr")
                expr = str(expr_raw).strip() if isinstance(expr_raw, str) and expr_raw.strip() else None
                eval_val = safe_eval_arith(expr) if expr else None
                if expr and eval_val is not None:
                    state.verified_facts["F1"] = VerifiedFact(
                        id="F1",
                        statement=str(item.get("statement") or "invariant ledger").strip(),
                        depends_on=[],
                        verified_by="ast_math",
                        expr=expr,
                        value=eval_val,
                    )
                    break
    if not state.active_assumptions():
        state.assumptions["A1"] = Assumption(
            id="A1", statement="Initial problem constraints as stated.", source="given"
        )
    state.current_depth = 1


def _advance_level_2(
    state: PyramidState,
    model: ModelClient,
    budget: BudgetTracker,
    traces: list[dict[str, Any]],
    *,
    phase: str,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
    use_self_checks: bool = True,
    correction_note: str | None = None,
) -> None:
    active_ids = [a.id for a in state.active_assumptions()]
    sample_a = ", ".join(active_ids[:2]) if active_ids else "A1"
    sample_b = ", ".join(active_ids) if active_ids else "A1"
    rules_text = _problem_rules_without_options(state.problem_text)
    sys_prompt = (
        "You are Level 2 (Decomposition & Subproblem Bounds) of a Revisable Reasoning Pyramid.\n"
        f"Using Problem Rules and Level-1 Active Assumptions ({', '.join(active_ids) or 'A1'}), solve the 5 subproblems for the CURRENT active regime (Latest evidence overrides any earlier active regime).\n"
        "Cite ONLY active assumption IDs in brackets. Write pure numbers and + - * / for arithmetic. Do NOT emit <tool_call> tags. After </think>, output ONLY 5 tagged lines:\n"
        f"L2-1 [{sample_a}]: required_units = <demand> / <active_per_unit_throughput> = <integer>\n"
        f"L2-2 [{sample_b}]: max_per_unit_cost = <total_budget_cap> / <required_units> = <number>\n"
        f"L2-3 [{sample_a}]: required_mode = <ACTIVE_MODE>; mandatory_feature = <optional, OR feature_name=true>\n"
        f"L2-4 [{sample_b}]: max_base_time_or_lat = <cap> - <active_penalty> = <number>\n"
        f"L2-5 [{sample_b}]: max_per_unit_resource = <gross_limit> - <active_overhead> = <number>"
    )
    correction_block = (
        f"\n\nSELF-CHECK CORRECTION REQUIRED:\n{correction_note}\n"
        "Re-read Problem Rules directly for the CURRENT active mode: "
        "(1) check if the boolean feature is 'optional' vs 'MUST be true' under the active mode; "
        "(2) check the exact gross resource limit (NOT A2 per-unit throughput!) minus active overhead, "
        "and exact gross latency/time cap minus active penalty."
        if correction_note
        else ""
    )
    user_prompt = (
        f"Problem Rules:\n{rules_text}\n\n"
        f"Latest evidence:\n{state.evidence[-1]}\n\n"
        f"{state.format_level_summary(max_level=1, include_invalidated=False)}"
        f"{correction_block}"
    )
    resp = _guarded_generate(
        model, budget,
        [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
        stage=f"{phase}_level2", traces=traces,
        max_tokens=max_tokens, temperature=temperature, enable_thinking=enable_thinking,
        recovery_instruction=(
            f"Output ONLY 5 tagged lines L2-1..L2-5 citing ONLY active IDs ({', '.join(active_ids)}):\n"
            f"L2-1 [{sample_a}]: required_units = <demand> / <per_unit_throughput> = <int>\n"
            f"L2-2 [{sample_b}]: max_per_unit_cost = <budget_cap> / <required_units> = <number>\n"
            f"L2-3 [{sample_a}]: required_mode = <ACTIVE_MODE>; mandatory_feature = <optional OR feature_name=true>\n"
            f"L2-4 [{sample_b}]: max_base_time_or_lat = <cap> - <active_penalty> = <number>\n"
            f"L2-5 [{sample_b}]: max_per_unit_resource = <gross_limit> - <active_overhead> = <number>"
        ),
        recovery_max_tokens=256,
    )
    parsed_nodes = parse_level2_output(resp.raw_text) if resp else []
    if not parsed_nodes and resp is not None and budget.can_continue():
        retry_resp = _guarded_generate(
            model, budget,
            [
                {
                    "role": "system",
                    "content": (
                        f"Output ONLY 5 tagged lines L2-1..L2-5 citing ONLY active IDs ({', '.join(active_ids) or 'A1'}):\n"
                        f"L2-1 [{sample_a}]: required_units = <demand> / <active_per_unit_throughput> = <int>\n"
                        f"L2-2 [{sample_b}]: max_per_unit_cost = <total_budget_cap> / <required_units> = <number>\n"
                        f"L2-3 [{sample_a}]: required_mode = <ACTIVE_MODE>; mandatory_feature = <optional OR feature_name=true>\n"
                        f"L2-4 [{sample_b}]: max_base_time_or_lat = <cap> - <active_penalty> = <number>\n"
                        f"L2-5 [{sample_b}]: max_per_unit_resource = <gross_limit> - <active_overhead> = <number>"
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            stage=f"{phase}_level2_format_retry",
            traces=traces,
            max_tokens=min(max_tokens, 256),
            temperature=temperature,
            enable_thinking=False,
        )
        if retry_resp is not None:
            parsed_nodes = parse_level2_output(retry_resp.raw_text)
    active_set = set(active_ids)
    for idx, item in enumerate(parsed_nodes, start=1):
        raw_id = str(item.get("id") or f"L2-{idx}")
        nid = raw_id if raw_id.upper().startswith("L2-") else f"L2-{idx}"
        kind = "strategy" if item.get("kind") == "strategy" else "subproblem"
        stmt = str(item.get("statement") or "").strip()
        raw_deps = [str(d) for d in (item.get("depends_on") or [])]
        deps: list[str] = []
        for d in raw_deps:
            if d in state.assumptions and state.assumptions[d].status == "invalidated":
                rep = state.assumptions[d].replaced_by
                if rep and rep in active_set:
                    deps.append(rep)
                elif f"{d}_new" in active_set:
                    deps.append(f"{d}_new")
                continue
            deps.append(d)
        if stmt:
            stmt = _verify_inline_equations(stmt)
            state.nodes[nid] = DependencyNode(
                id=nid, level=2, kind="strategy" if kind == "strategy" else "subproblem", statement=stmt, depends_on=list(dict.fromkeys(deps))
            )
    if use_self_checks:
        repair_l2_cross_bindings(state.nodes)
        repair_l2_from_l1_assumptions(
            state.active_assumptions(),
            state.nodes,
            options=parse_prompt_options(state.problem_text),
        )
    state.current_depth = 2


def _advance_level_3(
    state: PyramidState,
    model: ModelClient,
    budget: BudgetTracker,
    traces: list[dict[str, Any]],
    *,
    phase: str,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
    use_self_checks: bool = True,
    correction_note: str | None = None,
) -> None:
    sys_prompt = (
        "You are Level 3 (Detailed Reasoning & Option Selection) of a Revisable Reasoning Pyramid.\n"
        "Step 1: Verify the Level-2 derived numerical bounds (required mode, mandatory boolean feature, max cost/fee per unit, max base latency/time, max per-unit resource) against the active regime.\n"
        "Step 2: Check EVERY option OPT-A..OPT-F against ALL 5 bounds (1 short line per option checking mode, boolean feature, cost, latency/time, resource).\n"
        "Step 3: Use the verified F1 value for \"preserved\".\n"
        "Do NOT emit <tool_call> tags. After </think>, output ONLY 2 concise tagged lines followed by the final JSON object on the last line:\n"
        "L3-1: <verified active regime bounds: mode, mandatory feature, units, max_cost, max_base_lat, max_resource>\n"
        "L3-2: <OPT-A: pass/fail; OPT-B: pass/fail; OPT-C: pass/fail; OPT-D: pass/fail; OPT-E: pass/fail; OPT-F: pass/fail>\n"
        '{"primary": "OPT-X", "preserved": <integer>}'
    )
    correction_block = f"\n\nSELF-CHECK CORRECTION REQUIRED:\n{correction_note}" if correction_note else ""
    symbolic_hint = ""
    passing: list[str] = []
    unsat_l2_fallback = False
    if use_self_checks:
        options = parse_prompt_options(state.problem_text)
        bounds = extract_l2_bounds(state.active_nodes(level=2))
        if options and len(bounds) >= 3:
            passing, failures = verify_options_against_bounds(options, bounds)
            if len(passing) > 0:
                rows = [
                    f"- {opt}: PASS" if not failures[opt] else f"- {opt}: FAIL ({', '.join(failures[opt])})"
                    for opt in options
                ]
                symbolic_hint = "\n\nSymbolic L2 bound check on options:\n" + "\n".join(rows)
            else:
                unsat_l2_fallback = True
                near_opts = [opt for opt, errs in failures.items() if len(errs) == 1]
                near_str = f" Near-miss options failing only 1 derived bound: {', '.join(near_opts)}." if near_opts else ""
                symbolic_hint = (
                    "\n\nNote: Derived Level-2 bounds matched 0 options simultaneously (one L2 bound has an arithmetic slip)."
                    + near_str
                    + " Re-verify OPT-A..OPT-F directly against Problem Rules and Latest Evidence below, and select the unique valid option OPT-A..OPT-F (NEVER output placeholder OPT-X)."
                )

    problem_view = (
        state.problem_text
        if unsat_l2_fallback or not state.active_nodes(level=2)
        else _problem_options_or_full(state.problem_text)
    )

    user_prompt = (
        f"{problem_view}\n\n"
        f"Latest evidence:\n{state.evidence[-1]}\n\n"
        + state.format_level_summary(max_level=2, include_invalidated=False)
        + symbolic_hint
        + correction_block
        + f"\n\n{_ANSWER_FORMAT_NOTE}"
    )
    resp = _guarded_generate(
        model, budget,
        [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
        stage=f"{phase}_level3", traces=traces,
        max_tokens=max_tokens, temperature=temperature, enable_thinking=enable_thinking,
        recovery_instruction=(
            "Do NOT emit <tool_call> tags. Output ONLY 2 concise tagged lines followed by the final JSON object on the last line:\n"
            "L3-1: <derived active regime bounds>\n"
            "L3-2: <compact check of OPT-A..OPT-F>\n"
            '{"primary": "OPT-X", "preserved": <evaluated integer, never an expression>}'
        ),
        recovery_max_tokens=384,
    )
    l2_ids = [n.id for n in state.active_nodes(level=2)]
    steps, candidate = parse_level3_output(resp.raw_text) if resp else ([], None)
    for idx, item in enumerate(steps, start=1):
        nid = str(item.get("id") or f"L3-{idx}")
        stmt = str(item.get("statement") or "").strip()
        deps = [str(d) for d in (item.get("depends_on") or l2_ids)]
        if stmt:
            state.nodes[nid] = DependencyNode(
                id=nid, level=3, kind="reasoning_step", statement=stmt, depends_on=deps
            )
    if candidate is not None:
        state.candidate_answer = candidate
    if use_self_checks and len(passing) == 1:
        if state.candidate_answer is None:
            state.candidate_answer = {"primary": passing[0], "preserved": None}
        elif state.candidate_answer.get("primary") != passing[0]:
            state.candidate_answer["primary"] = passing[0]
    if use_self_checks:
        f1 = next(
            (
                f
                for f in state.surviving_facts(set(state.invalidated_assumption_ids()))
                if f.id == "F1" and not f.depends_on and f.verified_by == "ast_math" and f.value is not None
            ),
            None,
        )
        if f1 is not None and f1.value is not None:
            f1_clean = int(round(f1.value)) if float(f1.value).is_integer() else f1.value
            if state.candidate_answer is None:
                state.candidate_answer = {"primary": None, "preserved": f1_clean}
            else:
                state.candidate_answer["preserved"] = f1_clean
    state.current_depth = 3
