"""Shared verifiable reward and capability scoring across math, RPR audits, schemas, and XML tool calls."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from typing import Any

from model import extract_json_object, split_thinking
from verifiers import values_match


_FN_RE = re.compile(
    r"<tool_call>\s*<function=([A-Za-z0-9_.-]+)>\s*(.*?)\s*</function>\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
_PARAM_RE = re.compile(
    r"<parameter=([A-Za-z0-9_.-]+)>\s*(.*?)\s*</parameter>",
    re.DOTALL | re.IGNORECASE,
)
_AUDIT_LINE_RE = re.compile(
    r"^\s*(?:[-*]\s*)?CHECK\s+(A\w+)\s*:\s*(VALID|INVALID|CONTRADICTED|REPLACED|KEEP)\b",
    re.IGNORECASE | re.MULTILINE,
)
_CODE_FENCE_RE = re.compile(r"```")
_REAL_TOOL_TAG_RE = re.compile(
    r"<function=|<parameter=|</tool_call>|<tool_call>\s*(?:<|\{|\n)",
    re.IGNORECASE,
)


def parse_xml_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse LFM3-VL XML `<tool_call><function=...>...</tool_call>` blocks."""
    calls: list[dict[str, Any]] = []
    for fn_name, body in _FN_RE.findall(text):
        params: dict[str, str] = {}
        for k, v in _PARAM_RE.findall(body):
            params[k.strip()] = v.strip()
        calls.append({"name": fn_name.strip(), "arguments": params})
    return calls


def compute_five_gram_echo(prompt_text: str, think_text: str) -> float:
    """Fraction of `<think>` word 5-grams copied verbatim from the prompt."""
    p_words = prompt_text.lower().split()
    t_words = think_text.lower().split()
    if len(p_words) < 5 or len(t_words) < 5:
        return 0.0
    p_grams = {tuple(p_words[i : i + 5]) for i in range(len(p_words) - 4)}
    t_grams = [tuple(t_words[i : i + 5]) for i in range(len(t_words) - 4)]
    hits = sum(1 for g in t_grams if g in p_grams)
    return hits / max(1, len(t_grams))


def _normalize_with_think(raw_text: str) -> str:
    stripped = raw_text.lstrip()
    if "</think>" in stripped.lower() and not stripped.lower().startswith("<think>"):
        return "<think>" + raw_text
    return raw_text


def _args_match(pred_args: dict[str, str], exp_args: dict[str, Any]) -> bool:
    for k, exp_v in exp_args.items():
        if k not in pred_args:
            return False
        if not values_match(pred_args[k], exp_v):
            return False
    return True


_PY_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_python_code(text: str) -> str:
    m = _PY_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    idx = text.find("def ")
    return text[idx:].strip() if idx != -1 else text.strip()


def _verify_python_code(code_str: str, test_str: str) -> tuple[bool, bool]:
    if not code_str:
        return False, False
    try:
        compile(code_str, "<rlvr_code>", "exec")
    except SyntaxError:
        return False, False
    runner = (
        "ns = {'__name__': 'solution_module'}\n"
        f"exec(compile({code_str!r}, '<solution>', 'exec'), ns)\n"
        f"exec(compile({test_str!r}, '<unit_test>', 'exec'), ns)\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", runner],
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
        )
        return proc.returncode == 0, True
    except Exception:
        return False, True


def score_response(
    raw_text: str,
    *,
    prompt_text: str,
    kind: str,
    verifier: dict[str, Any],
    tokenizer: Any = None,
    max_think_tokens: int = 320,
) -> dict[str, Any]:
    """Compute multi-channel verifiable rewards and exact pass flags."""
    norm = _normalize_with_think(raw_text)
    thinking, content = split_thinking(norm)
    closed_think = "</think>" in raw_text.lower()
    if not closed_think and not thinking:
        thinking = raw_text

    if tokenizer is not None:
        think_tokens = len(tokenizer.encode(thinking, add_special_tokens=False))
    else:
        think_tokens = int(len(thinking.split()) * 1.35)

    echo_overlap = compute_five_gram_echo(prompt_text, thinking)
    has_tool_tag = bool(
        (closed_think and "<tool_call" in content.lower())
        or _REAL_TOOL_TAG_RE.search(raw_text)
    )

    # Channel 1: Concise closed thinking
    if closed_think and think_tokens <= max_think_tokens:
        r_close = 0.4
    elif closed_think:
        r_close = max(-0.2, 0.2 - 0.002 * (think_tokens - max_think_tokens))
    else:
        r_close = -0.5

    # Channel 2: 5-gram anti-echo penalty
    r_anti_echo = -0.6 * max(0.0, echo_overlap - 0.20)

    # Channel 3: Tool-calling calibration
    tool_leaked = False
    r_tool = 0.0
    forbid_tool = bool(verifier.get("forbid_tool_call") or verifier.get("abstain"))
    if forbid_tool:
        tool_leaked = has_tool_tag
        r_tool = -1.5 if tool_leaked else 1.0
    elif verifier.get("tool"):
        calls = parse_xml_tool_calls(raw_text)
        exp_tool = str(verifier["tool"])
        exp_args = verifier.get("args") or {}
        if any(c["name"] == exp_tool and _args_match(c["arguments"], exp_args) for c in calls):
            r_tool = 1.0
        elif any(c["name"] == exp_tool for c in calls):
            r_tool = 0.2
        elif calls:
            r_tool = -0.3
        else:
            r_tool = -0.8

    # Channel 4 & 5: Correctness & schema compliance
    correct = False
    r_correct = 0.0
    r_schema = 0.0

    if kind == "tool_pos" or (verifier.get("tool") and not forbid_tool):
        calls = parse_xml_tool_calls(raw_text)
        exp_tool = str(verifier["tool"])
        exp_args = verifier.get("args") or {}
        exact_call = any(
            c["name"] == exp_tool and _args_match(c["arguments"], exp_args) for c in calls
        )
        correct = bool(exact_call and closed_think)
        r_correct = 1.0 if exact_call else 0.0
        r_schema = 0.3 if calls else 0.0

    elif kind in {"math_opt", "multihop_math"} or (
        forbid_tool and "primary" in verifier
    ):
        parsed = extract_json_object(norm)
        has_fence = bool(_CODE_FENCE_RE.search(content))
        if isinstance(parsed, dict) and {"primary", "preserved"}.issubset(parsed.keys()):
            r_schema = 0.4 if not has_fence else 0.2
        elif parsed is not None:
            r_schema = 0.1

        if isinstance(parsed, dict):
            pri_ok = values_match(parsed.get("primary"), verifier.get("primary"))
            pres_raw = parsed.get("preserved")
            pres_is_num = isinstance(pres_raw, (int, float)) and not isinstance(pres_raw, bool)
            pres_ok = pres_is_num and values_match(pres_raw, verifier.get("preserved"))
            anchor_pri = verifier.get("anchor_primary")
            stayed_anchored = (
                anchor_pri is not None
                and not pri_ok
                and values_match(parsed.get("primary"), anchor_pri)
            )
            if pri_ok and pres_ok:
                r_correct = 1.2 if anchor_pri is not None else 1.0
                correct = not tool_leaked
            elif stayed_anchored:
                r_correct = -0.4
            elif pri_ok or pres_ok:
                r_correct = 0.3

    elif kind in {"python_code", "python_coding"} or verifier.get("kind") == "python_code":
        code_str = _extract_python_code(content if closed_think else raw_text)
        passed_tests, syntax_ok = _verify_python_code(code_str, str(verifier.get("test", "")))
        r_schema = 0.4 if (syntax_ok and "def " in code_str) else (0.1 if syntax_ok else -0.2)
        if passed_tests:
            r_correct = 1.0
            correct = bool(not tool_leaked and closed_think)
        elif syntax_ok:
            r_correct = 0.15

    elif kind == "audit" or "verdict" in verifier:
        checks = _AUDIT_LINE_RE.findall(content)
        has_summary = "SUMMARY:" in content.upper()
        n_invalid = sum(1 for _, v in checks if v.upper() in {"INVALID", "CONTRADICTED", "REPLACED"})
        n_valid = sum(1 for _, v in checks if v.upper() == "VALID")
        if checks:
            r_schema = 0.25 + (0.15 if has_summary else 0.0)
        exp_verdict = str(verifier.get("verdict", "")).upper()
        if exp_verdict == "INVALID":
            verdict_ok = n_invalid >= 1 and len(checks) >= 2 and has_summary
        else:
            verdict_ok = n_invalid == 0 and n_valid >= 3 and has_summary
        r_correct = 1.0 if verdict_ok else 0.0
        correct = bool(verdict_ok and not tool_leaked)

    elif kind in {"tool_neg", "tool_abstention"} or (
        verifier.get("abstain") and isinstance(verifier.get("expected"), dict)
    ):
        exp = verifier.get("expected") or {}
        parsed = extract_json_object(norm)
        has_fence = bool(_CODE_FENCE_RE.search(content))
        if isinstance(parsed, dict) and set(parsed.keys()) == set(exp.keys()):
            r_schema = 0.4 if not has_fence else 0.2
            all_num_ok = all(
                not (isinstance(ev, (int, float)) and not isinstance(ev, bool) and isinstance(parsed.get(ek), str))
                for ek, ev in exp.items()
            )
            vals_ok = all(values_match(parsed.get(ek), ev) for ek, ev in exp.items())
            if all_num_ok and vals_ok:
                r_correct = 1.0
                correct = not tool_leaked
            elif any(values_match(parsed.get(ek), ev) for ek, ev in exp.items()):
                r_correct = 0.3
        else:
            r_schema = 0.0

    elif kind == "citation_grounded":
        req_cites = [str(c) for c in (verifier.get("required_citations") or [])]
        forbid_cites = [str(c) for c in (verifier.get("forbidden_citations") or [])]
        explicit_valid = verifier.get("valid_citations")
        if explicit_valid is not None:
            valid_cites = set(str(c) for c in explicit_valid)
        else:
            valid_cites = set(re.findall(r"\[(?:S\d+|DOC-\d{4}|REC-\d{4})\]", prompt_text))
            m_rng = re.search(r"\[(DOC|REC)-0001\]`\s*\.\.\s*`?\[\1-(\d{4})\]", prompt_text)
            if m_rng:
                pfx, n_max = m_rng.group(1), int(m_rng.group(2))
                valid_cites.update(f"[{pfx}-{i:04d}]" for i in range(1, n_max + 1))
            valid_cites.update(req_cites)
            valid_cites.update(forbid_cites)
        found_brackets = re.findall(r"\[(?:S\d+|DOC-\d{4}|REC-\d{4})\]", content)
        has_all_req = all(rc in content for rc in req_cites)
        has_forbid = any(fc in content for fc in forbid_cites)
        has_hallucinated = bool(valid_cites) and any(fb not in valid_cites for fb in found_brackets)

        exp_json = verifier.get("expected")
        req_substrings = [str(s) for s in (verifier.get("required_substrings") or [])]
        forbid_substrings = [str(s) for s in (verifier.get("forbidden_substrings") or [])]

        fact_ok = True
        if isinstance(exp_json, dict) and exp_json:
            parsed = extract_json_object(norm)
            if isinstance(parsed, dict) and set(exp_json.keys()).issubset(parsed.keys()):
                fact_ok = all(values_match(parsed.get(ek), ev) for ek, ev in exp_json.items())
            else:
                fact_ok = False
        if req_substrings and not all(sub.lower() in content.lower() for sub in req_substrings):
            fact_ok = False
        if forbid_substrings and any(fsub.lower() in content.lower() for fsub in forbid_substrings):
            fact_ok = False

        r_schema = 0.4 if (found_brackets and not has_hallucinated) else (-0.2 if has_hallucinated else 0.0)
        if fact_ok and has_all_req and not has_forbid and not has_hallucinated:
            r_correct = 1.0
            correct = bool(not tool_leaked and closed_think)
        elif fact_ok and not has_hallucinated:
            r_correct = 0.35
        elif has_all_req and not has_forbid and not has_hallucinated:
            r_correct = 0.25

    elif kind == "json_values":
        exp = verifier.get("expected") or {}
        parsed = extract_json_object(norm)
        if isinstance(parsed, dict) and set(exp.keys()).issubset(parsed.keys()):
            r_schema = 0.35
            vals_ok = all(values_match(parsed.get(ek), ev) for ek, ev in exp.items())
            if vals_ok:
                r_correct = 1.0
                correct = bool(not tool_leaked and closed_think)
            elif any(values_match(parsed.get(ek), ev) for ek, ev in exp.items()):
                r_correct = 0.3

    elif kind == "schema_json" or (
        isinstance(verifier.get("expected"), dict) and "lines" not in verifier.get("expected", {})
    ):
        exp = verifier.get("expected") or {}
        raw_body = content.strip()
        has_fence = bool(_CODE_FENCE_RE.search(raw_body))
        try:
            direct_obj = json.loads(raw_body)
        except json.JSONDecodeError:
            direct_obj = None
        parsed = direct_obj if isinstance(direct_obj, dict) else extract_json_object(norm)
        if isinstance(direct_obj, dict) and not has_fence and set(direct_obj.keys()) == set(exp.keys()):
            r_schema = 0.4
        elif isinstance(parsed, dict):
            r_schema = 0.1
        if isinstance(parsed, dict) and set(parsed.keys()) == set(exp.keys()):
            no_uneval = all(
                not (isinstance(ev, int) and not isinstance(ev, bool) and isinstance(parsed.get(ek), str))
                for ek, ev in exp.items()
            )
            vals_ok = all(values_match(parsed.get(ek), ev) for ek, ev in exp.items())
            if no_uneval and vals_ok and isinstance(direct_obj, dict) and not has_fence:
                r_correct = 1.0
                correct = not tool_leaked
            elif vals_ok:
                r_correct = 0.5

    elif kind == "schema_lines" or "lines" in verifier.get("expected", {}):
        exp_lines = [str(x).strip() for x in (verifier.get("expected") or {}).get("lines", [])]
        got_lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        matches = sum(1 for a, b in zip(got_lines, exp_lines) if a == b)
        r_schema = 0.4 * (matches / max(1, len(exp_lines))) if len(got_lines) == len(exp_lines) else 0.1
        if got_lines == exp_lines:
            r_correct = 1.0
            correct = not tool_leaked

    channels = {
        "correct": r_correct,
        "schema": r_schema,
        "close_think": r_close,
        "anti_echo": r_anti_echo,
        "tool": r_tool,
    }
    total_reward = round(sum(channels.values()), 4)
    return {
        "reward": total_reward,
        "channels": channels,
        "correct": bool(correct),
        "closed_think": closed_think,
        "think_tokens": think_tokens,
        "echo_overlap": round(echo_overlap, 4),
        "tool_leaked": tool_leaked,
    }
