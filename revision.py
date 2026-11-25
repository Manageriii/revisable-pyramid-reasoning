"""Verification from Level 1 and foundational reconstruction logic for RPR."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any

from model import (
    ModelClient,
    ModelResponse,
    extract_json_object,
    needs_close_think_recovery,
    recover_close_think_json,
    split_thinking,
)
from state import Assumption, PyramidState, RevisionRecord
from symbolic_critic import (
    extract_l2_bounds,
    format_bounds_readable,
    parse_prompt_options,
    verify_options_against_bounds,
)
from verifiers import values_match


@dataclass
class VerificationFindings:
    """Result of checking reasoning and new evidence against Level-1 foundations."""

    foundation_invalid: bool
    invalidated_assumptions: list[str] = field(default_factory=list)
    replacement_assumptions: dict[str, str] = field(default_factory=dict)
    tainted_nodes: list[str] = field(default_factory=list)
    preserved_facts: list[str] = field(default_factory=list)
    violated_constraints: list[str] = field(default_factory=list)
    trigger_reason: str = ""
    raw_audit_text: str | None = None


_AUDIT_SYSTEM_PROMPT = (
    "You are a strict logical consistency auditor for a hierarchical reasoning pyramid.\n"
    "Check EVERY active Level-1 assumption (A1..A5) against Latest Evidence and Problem Rules.\n"
    "Distinguish active production constraints from separate staging, annex, sandbox, prototype, or future-roadmap notes.\n"
    "Do NOT repeat or quote the problem text inside <think>. Keep <think> brief (1 short line per assumption), do NOT emit <tool_call> tags, then close </think> and output ONLY tagged lines:\n"
    "CHECK A1: <VALID or INVALID> | <short reason>\n"
    "CHECK A2: <VALID or INVALID> | <short reason>\n"
    "SUMMARY: <1-sentence summary>\n"
    "Rules:\n"
    "1. Mark an assumption INVALID (or REPLACED | <updated rule>) ONLY if Latest Evidence directly switches the active production regime, workload demand, or global limits.\n"
    "2. NEVER mark A1..A5 INVALID because per-option numbers are not in Problem Rules or because Latest Evidence mentions a separate sandbox, staging floor, annex, or prototype test while stating active production has not switched.\n"
    "3. If your reason says the active regime/value 'has not switched', 'stays', 'remains unchanged', or describes a non-active track/annex/sandbox, the verdict MUST be VALID."
)

_CHECK_LINE_RE = re.compile(
    r"^\s*(?:[-*]\s*)?CHECK\s+(A\w+)\s*:\s*(VALID|INVALID|CONTRADICTED|REPLACED|KEEP)\b(?:\s*\|\s*(.*))?$",
    re.IGNORECASE | re.MULTILINE,
)
_SUMMARY_LINE_RE = re.compile(
    r"^\s*(?:[-*]\s*)?SUMMARY\s*:\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)
_NON_CHANGE_PHRASES = (
    "has not switched",
    "not switched",
    "not the active",
    "stays ",
    "stay ",
    "remains unchanged",
    "is unchanged",
    "unchanged",
    "remains valid",
    "still active",
)
_ACTIVE_SWITCH_PHRASES = (
    "switched from",
    "switched to",
    "switching active",
    "changed from",
    "changed to",
    "->",
    "now online",
    "supersedes",
    "overrides",
)


_NEGATED_SWITCH_RE = re.compile(
    r"\b(?:has\s+not|have\s+not|did\s+not|does\s+not|not|never|no)\s+(?:mode\s+)?(?:switched|switch|changed|change)(?:\s+(?:from|to))?\b",
    re.IGNORECASE,
)
_NOOP_FROM_TO_RE = re.compile(r"\bfrom\s+([A-Za-z0-9_.-]+)\s+to\s+\1\b", re.IGNORECASE)


def _is_self_contradictory_invalid_note(note: str) -> bool:
    """Return True if an `INVALID` reason explicitly asserts that the active regime/value did NOT change."""
    low = note.lower()
    if not low:
        return False
    if _NOOP_FROM_TO_RE.search(low):
        return True
    has_non_change = any(p in low for p in _NON_CHANGE_PHRASES) or bool(_NEGATED_SWITCH_RE.search(low))
    scrubbed = _NEGATED_SWITCH_RE.sub(" ", low)
    has_active_switch = any(p in scrubbed for p in _ACTIVE_SWITCH_PHRASES)
    return has_non_change and not has_active_switch


def _summary_asserts_global_preservation(summary_text: str) -> bool:
    """Return True when the auditor's own `SUMMARY:` explicitly concludes no active change occurred."""
    low = summary_text.strip().lower()
    if not low:
        return False
    global_keep = (
        "no mode switch" in low
        or "preserves a1..a5" in low
        or "all remain valid" in low
        or "all assumptions remain valid" in low
        or "all assumptions valid" in low
        or bool(re.search(r"all\s+(?:active\s+)?assumptions\b[^.;]*\bremain\s+valid", low))
        or (("stays " in low or "does not alter live" in low) and "remain valid" in low)
    )
    if not global_keep:
        return False
    scrubbed = _NEGATED_SWITCH_RE.sub(" ", low)
    contradicts_keep = any(
        tok in scrubbed
        for tok in ("invalidat", "except", "however", "switched from", "switched to", "changed from", "changed to", "supersedes", "overrides")
    )
    return not contradicts_keep


_NON_PROD_EVIDENCE_TOKENS: tuple[str, ...] = (
    "prototype",
    "sandbox",
    "canary",
    "offline",
    "simulation",
    "bench-test",
    "bench test",
    "non-switching",
    "qa lab",
    "trial",
)


def parse_audit_output(
    text: str,
    valid_active_ids: set[str],
) -> tuple[list[str], dict[str, str], str]:
    """Parse line-oriented `CHECK A1: VALID|INVALID|REPLACED` audit replies with JSON fallback."""
    _, content = split_thinking(text)
    id_lookup = {aid.upper(): aid for aid in valid_active_ids}
    matches = _CHECK_LINE_RE.findall(content)
    if matches:
        sum_match = _SUMMARY_LINE_RE.search(content)
        raw_summary = sum_match.group(1).strip() if sum_match else ""
        if _summary_asserts_global_preservation(raw_summary):
            return [], {}, ""
        checked_valid = {
            id_lookup[rid.upper()]
            for rid, rv, _ in matches
            if rid.upper() in id_lookup and rv.upper() == "VALID"
        }
        checked_invalid_a1 = any(
            rid.upper().startswith("A1") and rv.upper() in {"INVALID", "CONTRADICTED", "REPLACED"}
            for rid, rv, _ in matches
            if rid.upper() in id_lookup
        )
        a1_kept_valid = any(k.upper().startswith("A1") for k in checked_valid) and not checked_invalid_a1
        sum_low = raw_summary.lower()
        if (
            a1_kept_valid
            and any(tok in sum_low for tok in _NON_PROD_EVIDENCE_TOKENS)
            and any(p in sum_low for p in ("unchanged", "stays under", "stays on", "stays in", "a1 is valid", "non-switching"))
        ):
            return [], {}, ""

        invalidated: list[str] = []
        replacements: dict[str, str] = {}
        seen: set[str] = set()
        reasons: list[str] = []
        for raw_id, raw_verdict, raw_reason in matches:
            canonical = id_lookup.get(raw_id.upper())
            verdict = raw_verdict.upper()
            note = raw_reason.strip()
            if canonical and verdict in {"INVALID", "CONTRADICTED", "REPLACED"}:
                if verdict in {"INVALID", "CONTRADICTED"} and _is_self_contradictory_invalid_note(note):
                    continue
                if a1_kept_valid and not canonical.upper().startswith("A1") and any(
                    tok in note.lower() for tok in _NON_PROD_EVIDENCE_TOKENS
                ):
                    continue
                if canonical not in seen:
                    invalidated.append(canonical)
                    seen.add(canonical)
                if verdict == "REPLACED" and note:
                    replacements[canonical] = note
                if note:
                    reasons.append(f"{canonical}: {note}")
        summary = raw_summary if (raw_summary and invalidated) else " | ".join(reasons)
        return invalidated, replacements, summary

    parsed = extract_json_object(text)
    if parsed:
        raw_inv = parsed.get("invalidated")
        if raw_inv is None:
            raw_inv = parsed.get("invalidated_assumptions")
        invalidated_ids: list[str] = []
        if isinstance(raw_inv, list):
            for x in raw_inv:
                canonical = id_lookup.get(str(x).upper())
                if canonical and canonical not in invalidated_ids:
                    invalidated_ids.append(canonical)
        raw_rep = parsed.get("updated")
        if raw_rep is None:
            raw_rep = parsed.get("replacement_assumptions")
        replacements = {}
        if isinstance(raw_rep, dict):
            for k, v in raw_rep.items():
                canonical = id_lookup.get(str(k).upper())
                if canonical and canonical in invalidated_ids:
                    replacements[canonical] = str(v)
        audit_reason = str(parsed.get("reason", "")).strip()
        return invalidated_ids, replacements, audit_reason

    return [], {}, ""


def _check_self_consistency(state: PyramidState, already_invalid: set[str]) -> list[str]:
    """Run zero-leakage self-checks on candidate structure, Level-2 bounds, and AST math facts."""
    violations: list[str] = []
    options = parse_prompt_options(state.problem_text)
    l2_nodes = state.active_nodes(level=2)
    bounds = extract_l2_bounds(l2_nodes)
    if options and state.current_depth >= 2 and not l2_nodes:
        violations.append(
            "UNSAT_L2_BOUNDS: Level 2 produced 0 active decomposition nodes; "
            "emit explicit L2-1..L2-5 lines for the active regime."
        )
    elif options and len(bounds) >= 3:
        passing, failures = verify_options_against_bounds(options, bounds)
        if len(passing) == 0:
            sample_reasons = "; ".join(
                f"{oid}: {', '.join(errs[:2])}" for oid, errs in list(failures.items())[:6] if errs
            )
            near_misses = [
                f"{oid} fails ONLY {errs[0]}"
                for oid, errs in failures.items()
                if len(errs) == 1 and not errs[0].startswith("mode=")
            ]
            unsat_hint = (
                f" Minimal conflict near-misses: [{'; '.join(near_misses)}]."
                if near_misses
                else ""
            )
            violations.append(
                f"UNSAT_L2_BOUNDS: 0 candidate options satisfy derived Level-2 bounds "
                f"({format_bounds_readable(bounds)}); failures=[{sample_reasons}].{unsat_hint} "
                f"Re-check Problem Rules for the active mode: verify whether the boolean feature is optional vs MUST be true, "
                f"and verify gross limit minus active overhead (not per-unit workload throughput)."
            )
        elif state.candidate_answer is not None and state.candidate_answer.get("primary") is not None:
            pri = str(state.candidate_answer["primary"]).upper()
            if pri in failures and failures[pri]:
                violations.append(
                    f"Selected {pri} violates derived Level-2 bounds: {', '.join(failures[pri])}. "
                    f"Options satisfying Level-2 bounds: {passing}."
                )

    if state.candidate_answer is None:
        return violations

    if state.candidate_answer.get("primary") is None:
        violations.append("Candidate answer is missing primary selection.")

    surviving_numeric = [
        f
        for f in state.surviving_facts(already_invalid)
        if f.value is not None and f.verified_by == "ast_math"
    ]
    target_fact = next((f for f in surviving_numeric if f.id == "F1"), None)
    if target_fact is None and len(surviving_numeric) == 1:
        target_fact = surviving_numeric[0]

    preserved = state.candidate_answer.get("preserved")
    if target_fact is not None and preserved is not None:
        if not values_match(preserved, target_fact.value):
            violations.append(
                f"Candidate preserved={preserved} disagrees with self-verified invariant "
                f"{target_fact.id} ({target_fact.expr} = {target_fact.value:g})."
            )
    return violations


def verify_from_level_one(
    state: PyramidState,
    model: ModelClient | None = None,
    *,
    use_self_checks: bool = True,
    run_model_audit: bool = True,
    max_tokens: int = 768,
    temperature: float = 0.0,
    enable_thinking: bool = True,
    recovery_budget_tokens: int = 256,
) -> tuple[VerificationFindings, ModelResponse | None]:
    """Verify Level 1 foundations, ATMS dependency graph, and self-extracted math facts."""
    already_invalid = set(state.invalidated_assumption_ids())
    dangling_tainted = sorted(state.compute_tainted_nodes(already_invalid))
    violated_constraints = (
        _check_self_consistency(state, already_invalid) if use_self_checks else []
    )

    invalidated_ids: list[str] = []
    replacements: dict[str, str] = {}
    audit_reason = ""
    model_resp: ModelResponse | None = None

    should_call_model = (
        model is not None
        and run_model_audit
        and len(state.active_assumptions()) > 0
        and max_tokens > 32
    )
    if should_call_model and model is not None:
        active_lines = "\n".join(
            f"- {a.id}: {a.statement}" for a in state.active_assumptions()
        )
        rules_ref = (
            state.problem_text.split("\nTasks:", 1)[0]
            .split("\nCandidate options", 1)[0]
            .strip()
        )
        prompt_parts = [
            f"Problem Rules:\n{rules_ref}",
            "Active Level-1 Assumptions:\n" + active_lines,
            f"Latest Evidence:\n{state.evidence[-1] if state.evidence else 'Initial problem only.'}",
        ]
        if violated_constraints:
            prompt_parts.append(
                "Self-consistency check failures:\n"
                + "\n".join(f"- {v}" for v in violated_constraints)
            )
        prompt_parts.append(
            "Check each active assumption A1, A2, ... against Latest Evidence. "
            "Output CHECK lines and SUMMARY only (no <tool_call> tags)."
        )
        messages = [
            {"role": "system", "content": _AUDIT_SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(prompt_parts)},
        ]
        model_resp = model.generate(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_thinking=enable_thinking,
        )
        if needs_close_think_recovery(model_resp, enable_thinking=enable_thinking) and recovery_budget_tokens > 16:
            model_resp, _ = recover_close_think_json(
                model,
                messages,
                model_resp,
                max_tokens=min(256, recovery_budget_tokens),
                temperature=temperature,
                instruction=(
                    "Output ONLY one line per active assumption in exact format:\n"
                    "CHECK A1: <VALID or INVALID> | <short reason>\n"
                    "CHECK A2: <VALID or INVALID> | <short reason>\n"
                    "SUMMARY: <1-sentence summary>"
                ),
            )

        valid_active_ids = {a.id for a in state.active_assumptions()}
        invalidated_ids, replacements, audit_reason = parse_audit_output(
            model_resp.raw_text, valid_active_ids
        )

    foundation_invalid = bool(
        invalidated_ids or violated_constraints or dangling_tainted
    )

    if violated_constraints and not invalidated_ids and state.active_assumptions():
        audit_reason = "; ".join(violated_constraints)

    invalid_roots = set(invalidated_ids) | already_invalid
    tainted_nodes = sorted(
        state.compute_tainted_nodes(invalid_roots) | set(dangling_tainted)
    )
    surviving = state.surviving_facts(invalid_roots)
    preserved_ids = [f.id for f in surviving]

    reason_parts = [p for p in [audit_reason, *violated_constraints] if p]
    if dangling_tainted:
        reason_parts.append(f"Nodes depend on invalidated assumptions: {dangling_tainted}")
    trigger_reason = " | ".join(dict.fromkeys(reason_parts)) if reason_parts else "No invalidation."

    findings = VerificationFindings(
        foundation_invalid=foundation_invalid,
        invalidated_assumptions=invalidated_ids,
        replacement_assumptions=replacements,
        tainted_nodes=tainted_nodes,
        preserved_facts=preserved_ids,
        violated_constraints=violated_constraints,
        trigger_reason=trigger_reason,
        raw_audit_text=model_resp.raw_text if model_resp else None,
    )
    return findings, model_resp


def can_reconstruct(state: PyramidState) -> tuple[bool, str | None]:
    """Check loop and max-reconstruction safety guards."""
    if state.reconstruction_count >= state.max_reconstructions:
        return False, (
            f"Reached max_reconstructions ({state.max_reconstructions}); "
            "stopping revision loop."
        )
    return True, None


def reconstruct(
    state: PyramidState,
    findings: VerificationFindings,
    *,
    phase: str = "revision",
) -> tuple[PyramidState, bool]:
    """Rebuild the reasoning pyramid from Level 1 while preserving valid verified facts."""
    allowed, block_reason = can_reconstruct(state)
    if not allowed:
        state.revision_log.append(
            RevisionRecord(
                step=len(state.revision_log) + 1,
                phase=phase,
                trigger_reason=block_reason or findings.trigger_reason,
                invalidated_assumptions=list(findings.invalidated_assumptions),
                tainted_nodes=list(findings.tainted_nodes),
                preserved_facts=list(findings.preserved_facts),
                rebuilt_from_level=0,
            )
        )
        return state, False

    pre_sig = f"ev={len(state.evidence)}|{state.foundation_signature()}"
    was_seen_before = pre_sig in state.seen_foundation_signatures
    state.seen_foundation_signatures.add(pre_sig)

    invalid_roots = set(findings.invalidated_assumptions)
    for aid in findings.invalidated_assumptions:
        if aid in state.assumptions:
            old = state.assumptions[aid]
            old.status = "invalidated"
            old.invalidation_reason = findings.trigger_reason
            rep_stmt = findings.replacement_assumptions.get(aid)
            if rep_stmt:
                new_id = f"{aid}_r{state.reconstruction_count + 1}"
                old.replaced_by = new_id
                state.assumptions[new_id] = Assumption(
                    id=new_id,
                    statement=rep_stmt,
                    source="given",
                    status="active",
                )

    surviving = state.surviving_facts(invalid_roots)
    state.verified_facts = {f.id: f for f in surviving}

    active_nodes_before = [nid for nid, n in state.nodes.items() if n.status != "discarded"]
    tainted_set = set(findings.tainted_nodes) | state.compute_tainted_nodes(invalid_roots)
    tainted_active = tainted_set & set(active_nodes_before)
    r_taint = len(tainted_active) / max(1, len(active_nodes_before))
    selective_repair = (
        r_taint <= 0.65
        and bool(active_nodes_before)
        and "UNSAT_L2_BOUNDS" not in findings.trigger_reason
    )

    discarded_ids: list[str] = []
    for nid, node in state.nodes.items():
        if node.status != "discarded":
            if selective_repair and nid not in tainted_set:
                continue
            node.status = "discarded"
            discarded_ids.append(nid)

    state.candidate_answer = None
    all_replaced = bool(findings.invalidated_assumptions) and all(
        aid in findings.replacement_assumptions for aid in findings.invalidated_assumptions
    )
    if selective_repair and all_replaced:
        state.current_depth = 1
        rebuilt_from_level = 2
    else:
        state.current_depth = 0
        rebuilt_from_level = 1
    state.reconstruction_count += 1

    post_sig = f"ev={len(state.evidence)}|{state.foundation_signature()}"
    is_repeat_sig = (
        (was_seen_before if post_sig == pre_sig else post_sig in state.seen_foundation_signatures)
        and not findings.invalidated_assumptions
        and not findings.replacement_assumptions
    )

    state.revision_log.append(
        RevisionRecord(
            step=len(state.revision_log) + 1,
            phase=phase,
            trigger_reason=findings.trigger_reason,
            invalidated_assumptions=list(findings.invalidated_assumptions),
            tainted_nodes=sorted(set(findings.tainted_nodes) | set(discarded_ids)),
            preserved_facts=[f.id for f in surviving],
            rebuilt_from_level=rebuilt_from_level,
        )
    )

    if is_repeat_sig:
        state.max_reconstructions = min(
            state.max_reconstructions, state.reconstruction_count
        )

    return state, True
