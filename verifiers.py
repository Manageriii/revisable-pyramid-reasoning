"""Problem schema, deterministic constraint checks, and evaluation scoring."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
import json
import operator
from pathlib import Path
import re
from typing import Any, Literal

from model import extract_json_object, split_thinking


ProblemCategory = Literal["premise_change", "irrelevant_update", "ordinary"]


@dataclass
class Problem:
    """Single benchmark problem with optional mid-reasoning evidence update."""

    id: str
    category: ProblemCategory
    domain: str
    split: Literal["pilot", "dev", "test"]
    initial_prompt: str
    midway_update: str | None
    expected_initial: dict[str, Any]
    expected_final: dict[str, Any]
    expected_invalidated_keywords: list[str] = field(default_factory=list)
    initial_constraints_spec: dict[str, Any] = field(default_factory=dict)
    final_constraints_spec: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Problem":
        return cls(
            id=str(raw["id"]),
            category=raw["category"],
            domain=str(raw.get("domain", "general")),
            split=raw.get("split", "pilot"),
            initial_prompt=str(raw["initial_prompt"]),
            midway_update=raw.get("midway_update"),
            expected_initial=dict(raw["expected_initial"]),
            expected_final=dict(raw["expected_final"]),
            expected_invalidated_keywords=list(
                raw.get("expected_invalidated_keywords", [])
            ),
            initial_constraints_spec=dict(raw.get("initial_constraints_spec", {})),
            final_constraints_spec=dict(raw.get("final_constraints_spec", {})),
        )


def load_problems(path: str | Path, *, limit: int | None = None) -> list[Problem]:
    """Load problems from a JSONL file."""
    problems: list[Problem] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            problems.append(Problem.from_dict(json.loads(line)))
            if limit is not None and len(problems) >= limit:
                break
    return problems


def parse_candidate_answer(text: str) -> dict[str, Any] | None:
    """Parse `{primary, preserved}` from model text, with safe regex fallbacks."""
    obj = extract_json_object(text)
    if obj is not None:
        primary = (
            obj.get("primary")
            if "primary" in obj
            else obj.get("primary_choice", obj.get("answer"))
        )
        if isinstance(primary, str) and primary.strip().upper() == "OPT-X":
            primary = None
        preserved = (
            obj.get("preserved")
            if "preserved" in obj
            else obj.get("preserved_decision", obj.get("invariant"))
        )
        if primary is not None or preserved is not None:
            return {"primary": primary, "preserved": preserved}

    _, content = split_thinking(text)
    scan = content or text
    primary_match = re.search(r'"primary"\s*:\s*"([^"]+)"', scan)
    preserved_match = re.search(
        r'"preserved"\s*:\s*(null|true|false|-?\d+(?:\.\d+)?|"[^"]*")',
        scan,
        re.IGNORECASE,
    )
    if primary_match or preserved_match:
        primary_val = primary_match.group(1) if primary_match else None
        if isinstance(primary_val, str) and primary_val.strip().upper() == "OPT-X":
            primary_val = None
        preserved_val: Any = None
        if preserved_match:
            raw_pres = preserved_match.group(1)
            try:
                preserved_val = json.loads(raw_pres)
            except json.JSONDecodeError:
                preserved_val = raw_pres.strip('"')
        if primary_val is not None or preserved_val is not None:
            return {"primary": primary_val, "preserved": preserved_val}

    line_primary = re.search(r"^\s*PRIMARY\s*:\s*(OPT-[A-F0-9]+)\b", scan, re.IGNORECASE | re.MULTILINE)
    line_preserved = re.search(r"^\s*PRESERVED\s*:\s*(-?\d+(?:\.\d+)?)\b", scan, re.IGNORECASE | re.MULTILINE)
    if line_primary or line_preserved:
        p_val = line_primary.group(1).upper() if line_primary else None
        if p_val == "OPT-X":
            p_val = None
        k_val: Any = None
        if line_preserved:
            raw_num = line_preserved.group(1)
            k_val = float(raw_num) if "." in raw_num else int(raw_num)
        if p_val is not None or k_val is not None:
            return {"primary": p_val, "preserved": k_val}

    opt_matches = [
        m.upper()
        for m in re.findall(r"\bOPT-[A-F0-9]+\b", scan, re.IGNORECASE)
        if m.upper() != "OPT-X"
    ]
    if opt_matches:
        return {"primary": opt_matches[-1], "preserved": None}
    return None


def _normalize_scalar(value: Any) -> Any:
    """Normalize answers across minor formatting variations without altering meaning."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        lower = s.lower()
        if lower in {"true", "yes"}:
            return True
        if lower in {"false", "no"}:
            return False
        opt_match = re.fullmatch(r"(?:option\s+)?(opt-[a-z0-9]+)", lower)
        if opt_match:
            return opt_match.group(1).upper()
        # Extract leading pure number if unit suffix like "15 GB" was appended.
        num_match = re.fullmatch(
            r"(-?\d+(?:\.\d+)?)\s*(?:gb|mb|ms|usd|hours?|hrs?|nodes?|days?|%|units?|tasks?|gb/s|s)?",
            lower,
        )
        if num_match:
            return float(num_match.group(1))
        return s.upper() if s.upper().startswith("OPT-") else lower
    if isinstance(value, list):
        return [_normalize_scalar(v) for v in value]
    return value


def values_match(predicted: Any, expected: Any) -> bool:
    """Compare predicted and expected answer fields with numeric tolerance."""
    if expected is None:
        return predicted is None
    p = _normalize_scalar(predicted)
    e = _normalize_scalar(expected)
    if p is None or e is None:
        return False
    if isinstance(p, float) and isinstance(e, float):
        return abs(p - e) <= 1e-6
    return p == e


_ARITH_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_ARITH_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_eval_arith(expr: str) -> float | None:
    """Safely evaluate a pure arithmetic expression string via AST.

    Supports only numeric literals (`int`/`float`, excluding `bool`), binary
    `+`, `-`, `*`, `/`, `//`, `%`, and unary `+`, `-`. Returns `None` on
    disallowed nodes, syntax errors, or division by zero.
    """
    if not isinstance(expr, str) or not expr.strip():
        return None
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        return None

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("Unsupported literal")
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _ARITH_BINOPS:
            left = _eval(node.left)
            right = _eval(node.right)
            if type(node.op) in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0.0:
                raise ZeroDivisionError("division by zero")
            return float(_ARITH_BINOPS[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ARITH_UNARYOPS:
            return float(_ARITH_UNARYOPS[type(node.op)](_eval(node.operand)))
        raise ValueError(f"Unsupported AST node: {type(node).__name__}")

    try:
        return _eval(tree)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def check_candidate_constraints(
    candidate: dict[str, Any] | None,
    constraints_spec: dict[str, Any],
) -> list[str]:
    """Deterministically verify a candidate choice against problem-stated rules.

    EXCLUSIVE USE: Offline dataset uniqueness validation (`data/generate_dataset.py`)
    and unit tests. Solvers (`engine.py`, `revision.py`) MUST NEVER call this with
    ground-truth problem specs during inference.
    """
    if not constraints_spec:
        return []
    if candidate is None:
        return ["No parseable candidate answer was produced."]

    violations: list[str] = []
    options = constraints_spec.get("options")
    rules = constraints_spec.get("rules", [])
    primary = _normalize_scalar(candidate.get("primary"))

    if options:
        if not isinstance(primary, str) or primary not in options:
            violations.append(
                f"Selected primary option '{candidate.get('primary')}' is not a valid option in {sorted(options)}."
            )
            return violations

        attrs = options[primary]
        for rule in rules:
            field_name = rule["field"]
            op = rule["op"]
            target = rule["value"]
            label = rule.get("label", f"{field_name} {op} {target}")
            actual = attrs.get(field_name)
            if actual is None:
                violations.append(f"Option {primary} missing attribute '{field_name}' for rule: {label}")
                continue
            if not _eval_rule(actual, op, target):
                violations.append(
                    f"Option {primary} ({field_name}={actual}) violates rule: {label}"
                )

    return violations


def _eval_rule(actual: Any, op: str, target: Any) -> bool:
    if op == "eq":
        return actual == target
    if op == "neq":
        return actual != target
    if op == "lte":
        return float(actual) <= float(target)
    if op == "lt":
        return float(actual) < float(target)
    if op == "gte":
        return float(actual) >= float(target)
    if op == "gt":
        return float(actual) > float(target)
    if op == "in":
        return actual in target
    raise ValueError(f"Unsupported constraint operator: {op}")


@dataclass
class ProblemScore:
    """Detailed evaluation outcome for one problem and one mode."""

    problem_id: str
    category: ProblemCategory
    mode: str
    initial_primary_correct: bool
    initial_preserved_correct: bool
    final_primary_correct: bool
    final_preserved_correct: bool
    final_exact_correct: bool
    unnecessary_change: bool
    invalidation_detected: bool | None
    invalidation_false_positive: bool | None
    reconstructions_triggered: int
    parsed_successfully: bool


def score_problem_run(
    problem: Problem,
    mode: str,
    *,
    initial_answer: dict[str, Any] | None,
    final_answer: dict[str, Any] | None,
    revision_log: list[dict[str, Any]] | None = None,
    invalidated_statements: list[str] | None = None,
) -> ProblemScore:
    """Score a single run across accuracy, belief retention, and invalidation metrics."""
    init_p = (
        values_match(initial_answer.get("primary"), problem.expected_initial.get("primary"))
        if initial_answer
        else False
    )
    init_k = (
        values_match(
            initial_answer.get("preserved"), problem.expected_initial.get("preserved")
        )
        if initial_answer and problem.expected_initial.get("preserved") is not None
        else True
    )

    final_p = (
        values_match(final_answer.get("primary"), problem.expected_final.get("primary"))
        if final_answer
        else False
    )
    has_preserved = problem.expected_final.get("preserved") is not None
    final_k = (
        values_match(
            final_answer.get("preserved"), problem.expected_final.get("preserved")
        )
        if final_answer and has_preserved
        else not has_preserved
    )
    final_exact = final_p and final_k

    # Unnecessary change check:
    # - For irrelevant_update: any change to primary OR preserved vs initial correct answer (or vs initial_answer)
    # - For premise_change: changing the invariant preserved sub-answer when it was valid initially (or getting preserved wrong when initial was right)
    unnecessary_change = False
    if problem.category == "irrelevant_update" and initial_answer and final_answer:
        changed_primary = not values_match(
            initial_answer.get("primary"), final_answer.get("primary")
        )
        changed_preserved = has_preserved and not values_match(
            initial_answer.get("preserved"), final_answer.get("preserved")
        )
        unnecessary_change = changed_primary or changed_preserved
    elif problem.category == "premise_change" and has_preserved and initial_answer:
        if init_k and not final_k:
            unnecessary_change = True

    reconstructions = len(revision_log) if revision_log else 0
    invalidation_detected: bool | None = None
    invalidation_false_positive: bool | None = None

    if mode == "rpr":
        if problem.category == "premise_change":
            if reconstructions == 0:
                invalidation_detected = False
            elif not problem.expected_invalidated_keywords:
                invalidation_detected = reconstructions > 0
            else:
                corpus = " ".join(invalidated_statements or []).lower()
                invalidation_detected = any(
                    kw.lower() in corpus for kw in problem.expected_invalidated_keywords
                )
        elif problem.category == "irrelevant_update":
            invalidation_false_positive = reconstructions > 0

    return ProblemScore(
        problem_id=problem.id,
        category=problem.category,
        mode=mode,
        initial_primary_correct=init_p,
        initial_preserved_correct=init_k,
        final_primary_correct=final_p,
        final_preserved_correct=final_k,
        final_exact_correct=final_exact,
        unnecessary_change=unnecessary_change,
        invalidation_detected=invalidation_detected,
        invalidation_false_positive=invalidation_false_positive,
        reconstructions_triggered=reconstructions,
        parsed_successfully=final_answer is not None,
    )
