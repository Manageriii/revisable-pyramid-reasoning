"""Zero-leakage symbolic critic for prompt-visible options and Level-2 derived bounds."""

from __future__ import annotations

import re
from typing import Any

from state import Assumption, DependencyNode
from verifiers import safe_eval_arith


_OPTION_LINE_RE = re.compile(
    r"^\s*[-*]\s*(OPT-[A-Z0-9]+)\s*:\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)
_KV_PAIR_RE = re.compile(
    r'([A-Za-z_][A-Za-z0-9_]*)\s*=\s*("[^"]*"|\'[^\']*\'|true|false|-?\d+(?:\.\d+)?)',
    re.IGNORECASE,
)
_L2_REQ_UNITS_RE = re.compile(
    r"required_units\s*=\s*([^=;\n]+?)(?:=\s*(-?\d+(?:\.\d+)?))?\s*(?:;|$)",
    re.IGNORECASE,
)
_L2_COST_DIV_RE = re.compile(
    r"^(\s*max_per_unit_cost\s*=\s*)(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)(?:\s*=\s*-?\d+(?:\.\d+)?)?(.*)$",
    re.IGNORECASE,
)


def parse_prompt_options(problem_text: str) -> dict[str, dict[str, Any]]:
    """Deterministically parse candidate options `OPT-A..OPT-F` from raw problem text."""
    options: dict[str, dict[str, Any]] = {}
    for raw_opt, raw_rhs in _OPTION_LINE_RE.findall(problem_text):
        opt_id = raw_opt.upper()
        if opt_id == "OPT-X":
            continue
        attrs: dict[str, Any] = {}
        for key, raw_val in _KV_PAIR_RE.findall(raw_rhs):
            low = raw_val.lower()
            if (raw_val.startswith('"') and raw_val.endswith('"')) or (
                raw_val.startswith("'") and raw_val.endswith("'")
            ):
                attrs[key] = raw_val[1:-1]
            elif low == "true":
                attrs[key] = True
            elif low == "false":
                attrs[key] = False
            elif "." in raw_val:
                attrs[key] = float(raw_val)
            else:
                attrs[key] = int(raw_val)
        if attrs:
            options[opt_id] = attrs
    return options


def _format_num(val: float) -> str:
    return str(int(round(val))) if abs(val - round(val)) <= 1e-6 else f"{val:g}"


def repair_l2_cross_bindings(nodes: dict[str, DependencyNode]) -> list[str]:
    """Repair `L2-2` budget/unit division if it divided by demand instead of `L2-1` `required_units`."""
    repairs: list[str] = []
    req_units: float | None = None
    for nid, node in nodes.items():
        if node.level != 2 or node.status == "discarded":
            continue
        m = _L2_REQ_UNITS_RE.search(node.statement)
        if m:
            expr, rhs_num = m.group(1).strip(), m.group(2)
            eval_expr = safe_eval_arith(expr)
            if eval_expr is not None and eval_expr > 0:
                req_units = eval_expr
            elif rhs_num is not None and float(rhs_num) > 0:
                req_units = float(rhs_num)
            break

    if req_units is None or req_units <= 0:
        return repairs

    for nid, node in nodes.items():
        if node.level != 2 or node.status == "discarded":
            continue
        m = _L2_COST_DIV_RE.match(node.statement.strip())
        if not m:
            continue
        prefix, budget_s, denom_s, suffix = m.groups()
        budget_val = float(budget_s)
        denom_val = float(denom_s)
        if abs(denom_val - req_units) > 1e-6:
            corrected = budget_val / req_units
            u_str = _format_num(req_units)
            c_str = _format_num(corrected)
            node.statement = f"{prefix.strip()} {budget_s} / {u_str} = {c_str}{suffix}".strip()
            repairs.append(
                f"Repaired {nid} denominator {denom_s} -> {u_str} from L2-1 required_units ({budget_s} / {u_str} = {c_str})."
            )
    return repairs


_L1_LAT_CAP_RE = re.compile(
    r"\+\s*(-?\d+(?:\.\d+)?)\s*<=\s*(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_L1_RES_DIFF_RE = re.compile(
    r"(?:<=\s*|\bis\s+)(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_L1_BUDGET_CAP_RE = re.compile(
    r"\*\s*[A-Za-z_][A-Za-z0-9_]*\s*<=\s*(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_L2_SUB_BOUND_RE = re.compile(
    r"^(\s*(max_base_time_or_lat|max_per_unit_resource)\s*=\s*)(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)(?:\s*=\s*-?\d+(?:\.\d+)?)?(.*)$",
    re.IGNORECASE,
)
_L1_BOOL_CLAUSE_RE = re.compile(
    r"(?:boolean_feature|mandatory_feature)\s*=\s*([^;\n]+)",
    re.IGNORECASE,
)


def _parse_bool_clause(clause: str) -> tuple[str | None, bool] | None:
    """Parse `<feat> optional` or `<feat>=true` from an A1/L2 boolean clause."""
    cleaned = clause.strip().strip("()[]\"'")
    if not cleaned:
        return None
    low = cleaned.lower()
    if "or optional" in low or "mandatory=true | optional" in low:
        return None
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", cleaned)
    skip = {
        "optional", "none", "null", "rule", "true", "false",
        "mandatory", "required", "must", "be", "is", "or", "boolean_feature", "mandatory_feature",
    }
    feat_name = next((t for t in tokens if t.lower() not in skip), None)
    has_opt = (
        "optional" in low
        or "=false" in low
        or "= false" in low
        or ", false" in low
        or low in {"none", "false"}
    )
    if has_opt:
        return (feat_name, False)
    if (
        "=true" in low
        or "= true" in low
        or ", true" in low
        or "mandatory" in low
        or "must be true" in low
        or "required" in low
    ):
        return (feat_name, True) if feat_name else None
    return None


def repair_l2_from_l1_assumptions(
    assumptions: list[Assumption],
    nodes: dict[str, DependencyNode],
    options: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    """Repair stale Phase-1 bound carryovers in Level 2 using only active Level-1 assumptions.

    When `options` (parsed from prompt text) is supplied, enforces a monotonic satisfiability
    guard: if Level 2 before repair already satisfies >= 1 candidate option and overwriting from
    Level 1 would yield 0 satisfying options (`UNSAT`), the overwrite is rolled back.
    """
    snapshot = {nid: node.statement for nid, node in nodes.items()}
    passing_before: list[str] = []
    if options:
        bounds_before = extract_l2_bounds([n for n in nodes.values() if n.level == 2 and n.status != "discarded"])
        if len(bounds_before) >= 3:
            passing_before, _ = verify_options_against_bounds(options, bounds_before)

    repairs: list[str] = []
    l1_lat: tuple[float, float] | None = None  # (cap, pen)
    l1_res: tuple[float, float] | None = None  # (gross, over)
    l1_budget: float | None = None
    l1_bool: tuple[str | None, bool] | None = None

    for a in assumptions:
        if a.status != "active":
            continue
        stmt = a.statement
        m_bool = _L1_BOOL_CLAUSE_RE.search(stmt)
        if m_bool:
            parsed_b = _parse_bool_clause(m_bool.group(1))
            if parsed_b is not None:
                l1_bool = parsed_b
        m_lat = _L1_LAT_CAP_RE.search(stmt)
        if m_lat:
            pen_v, cap_v = float(m_lat.group(1)), float(m_lat.group(2))
            if cap_v > pen_v >= 0:
                l1_lat = (cap_v, pen_v)
        m_res = _L1_RES_DIFF_RE.search(stmt)
        if m_res and ("lat" not in stmt.lower() and "read_ms" not in stmt.lower()):
            gross_v, over_v = float(m_res.group(1)), float(m_res.group(2))
            if gross_v > over_v >= 0:
                l1_res = (gross_v, over_v)
        m_bud = _L1_BUDGET_CAP_RE.search(stmt)
        if m_bud:
            b_v = float(m_bud.group(1))
            if b_v > 0:
                l1_budget = b_v

    for nid, node in nodes.items():
        if node.level != 2 or node.status == "discarded":
            continue
        if l1_bool is not None and "mandatory_feature" in node.statement.lower():
            l1_feat, l1_req = l1_bool
            segs = [s.strip() for s in node.statement.split(";")]
            new_segs: list[str] = []
            changed_bool = False
            for seg in segs:
                fm = re.match(r"^mandatory_feature\s*=\s*(.+)$", seg, re.IGNORECASE)
                if fm:
                    l2_parsed = _parse_bool_clause(fm.group(1))
                    l2_feat = l2_parsed[0] if l2_parsed else None
                    l2_req = l2_parsed[1] if l2_parsed else None
                    eff_feat = l1_feat or l2_feat
                    if l2_req != l1_req:
                        if l1_req and eff_feat:
                            new_segs.append(f"mandatory_feature = {eff_feat}=true")
                            changed_bool = True
                            continue
                        if not l1_req:
                            rep_rhs = f"{eff_feat} optional" if eff_feat else "optional"
                            new_segs.append(f"mandatory_feature = {rep_rhs}")
                            changed_bool = True
                            continue
                new_segs.append(seg)
            if changed_bool:
                node.statement = "; ".join(new_segs)
                repairs.append(f"Repaired {nid} mandatory_feature from L1 A1 ({l1_bool}).")

        m_sub = _L2_SUB_BOUND_RE.match(node.statement.strip())
        if m_sub:
            prefix, key_name, lhs_s, rhs_s, suffix = m_sub.groups()
            lhs_v, rhs_v = float(lhs_s), float(rhs_s)
            key_low = key_name.lower()
            if key_low == "max_base_time_or_lat" and l1_lat is not None:
                cap_v, pen_v = l1_lat
                if abs(lhs_v - cap_v) > 1e-6 or abs(rhs_v - pen_v) > 1e-6:
                    res_v = cap_v - pen_v
                    c_s, p_s, r_s = _format_num(cap_v), _format_num(pen_v), _format_num(res_v)
                    node.statement = f"{prefix.strip()} {c_s} - {p_s} = {r_s}{suffix}".strip()
                    repairs.append(f"Repaired {nid} latency cap {lhs_s}-{rhs_s} -> {c_s}-{p_s}={r_s} from L1.")
            elif key_low == "max_per_unit_resource" and l1_res is not None:
                gross_v, over_v = l1_res
                if abs(lhs_v - gross_v) > 1e-6 or abs(rhs_v - over_v) > 1e-6:
                    res_v = gross_v - over_v
                    g_s, o_s, r_s = _format_num(gross_v), _format_num(over_v), _format_num(res_v)
                    node.statement = f"{prefix.strip()} {g_s} - {o_s} = {r_s}{suffix}".strip()
                    repairs.append(f"Repaired {nid} resource limit {lhs_s}-{rhs_s} -> {g_s}-{o_s}={r_s} from L1.")
            continue

        if l1_budget is not None:
            m_cost = _L2_COST_DIV_RE.match(node.statement.strip())
            if m_cost:
                prefix, budget_s, denom_s, suffix = m_cost.groups()
                budget_v, denom_v = float(budget_s), float(denom_s)
                if denom_v > 0 and abs(budget_v - l1_budget) > 1e-6:
                    corr = l1_budget / denom_v
                    b_s, d_s, c_s = _format_num(l1_budget), _format_num(denom_v), _format_num(corr)
                    node.statement = f"{prefix.strip()} {b_s} / {d_s} = {c_s}{suffix}".strip()
                    repairs.append(f"Repaired {nid} budget numerator {budget_s} -> {b_s} from L1.")

    if repairs and options and len(passing_before) >= 1:
        bounds_after = extract_l2_bounds([n for n in nodes.values() if n.level == 2 and n.status != "discarded"])
        passing_after, _ = verify_options_against_bounds(options, bounds_after)
        if len(passing_after) == 0:
            for nid, old_stmt in snapshot.items():
                nodes[nid].statement = old_stmt
            return []

    return repairs


def _extract_numeric_bound(rhs: str) -> float | None:
    """Extract evaluated numeric value from the right-hand side of `key = expr [= num]`."""
    cleaned = rhs.strip()
    if not cleaned:
        return None
    parts = [p.strip() for p in cleaned.split("=") if p.strip()]
    for part in reversed(parts):
        m = re.fullmatch(r"(-?\d+(?:\.\d+)?)\b.*", part)
        if m:
            num_head = re.match(r"^-?\d+(?:\.\d+)?$", part)
            if num_head:
                return float(part)
        num_m = re.match(r"^(-?\d+(?:\.\d+)?)\s*(?:[A-Za-z/%]+)?$", part)
        if num_m:
            return float(num_m.group(1))
        arith_head = re.match(r"^([0-9.+\-*/()\s]+)", part)
        if arith_head:
            val = safe_eval_arith(arith_head.group(1))
            if val is not None:
                return val
    return None


def extract_l2_bounds(nodes: list[DependencyNode]) -> dict[str, Any]:
    """Extract derived active-regime constraints from Level-2 dependency nodes."""
    bounds: dict[str, Any] = {}
    for node in nodes:
        if node.level != 2 or node.status == "discarded":
            continue
        for seg in node.statement.split(";"):
            s = seg.strip()
            if not s:
                continue
            mode_m = re.match(
                r"^(?:required_mode|active_mode|mode)\s*=\s*([A-Za-z0-9_]+)$",
                s,
                re.IGNORECASE,
            )
            if mode_m:
                mode_val = mode_m.group(1).upper()
                if mode_val != "ACTIVE_MODE":
                    bounds["required_mode"] = mode_val
                continue

            feat_m = re.match(r"^mandatory_feature\s*=\s*(.+)$", s, re.IGNORECASE)
            if feat_m:
                parsed_b = _parse_bool_clause(feat_m.group(1))
                if parsed_b is not None:
                    feat_name, is_req = parsed_b
                    bounds["mandatory_feature"] = (feat_name or "boolean_feature", is_req)
                continue

            for key in ("max_per_unit_cost", "max_base_time_or_lat", "max_per_unit_resource"):
                km = re.match(rf"^{key}\s*=\s*(.+)$", s, re.IGNORECASE)
                if km:
                    val = _extract_numeric_bound(km.group(1))
                    if val is not None:
                        bounds[key] = val
    return bounds


def format_bounds_readable(bounds: dict[str, Any]) -> str:
    """Format derived bounds without Python tuple syntax so retries never echo `(feat, True)`."""
    parts: list[str] = []
    if "required_mode" in bounds:
        parts.append(f"required_mode={bounds['required_mode']}")
    if "mandatory_feature" in bounds:
        feat_name, is_req = bounds["mandatory_feature"]
        parts.append(
            f"mandatory_feature={feat_name}=true"
            if is_req
            else f"mandatory_feature={feat_name} optional"
        )
    for k in ("max_per_unit_cost", "max_base_time_or_lat", "max_per_unit_resource"):
        if k in bounds and bounds[k] is not None:
            parts.append(f"{k}={_format_num(float(bounds[k]))}")
    return ", ".join(parts)


def _classify_option_attr(key: str, val: Any) -> str:
    low = key.lower()
    if low == "mode":
        return "mode"
    if isinstance(val, bool):
        return "bool"
    if low in {"cost", "price"} or "usd" in low or "cost" in low or "fee" in low or "toll" in low or "rent" in low or "grant" in low:
        return "cost"
    if low.startswith("base_") or "lat" in low or "wait" in low or "lag" in low or low.endswith(("_min", "_ms", "_hrs", "_sec")):
        return "latency"
    if isinstance(val, (int, float)):
        return "resource"
    return "other"


def verify_options_against_bounds(
    options: dict[str, dict[str, Any]],
    bounds: dict[str, Any],
) -> tuple[list[str], dict[str, list[str]]]:
    """Check parsed prompt options deterministically against model-derived Level-2 bounds."""
    passing: list[str] = []
    violations: dict[str, list[str]] = {}

    req_mode = bounds.get("required_mode")
    feat_spec = bounds.get("mandatory_feature")
    max_cost = bounds.get("max_per_unit_cost")
    max_lat = bounds.get("max_base_time_or_lat")
    max_res = bounds.get("max_per_unit_resource")

    for opt_id, attrs in options.items():
        errs: list[str] = []
        if req_mode and "mode" in attrs:
            actual_mode = str(attrs["mode"]).upper()
            if actual_mode != str(req_mode).upper():
                errs.append(f"mode={attrs['mode']} != {req_mode}")

        if isinstance(feat_spec, tuple) and len(feat_spec) == 2:
            feat_name, is_req = feat_spec
            if is_req:
                if feat_name in attrs:
                    actual_bool = attrs[feat_name]
                else:
                    actual_bool = next((v for v in attrs.values() if isinstance(v, bool)), None)
                if actual_bool is not True:
                    errs.append(f"{feat_name}={actual_bool} != True")

        for key, val in attrs.items():
            kind = _classify_option_attr(key, val)
            if kind == "cost" and max_cost is not None:
                if float(val) > float(max_cost) + 1e-6:
                    errs.append(f"{key}={val} > {_format_num(float(max_cost))}")
            elif kind == "latency" and max_lat is not None:
                if float(val) > float(max_lat) + 1e-6:
                    errs.append(f"{key}={val} > {_format_num(float(max_lat))}")
            elif kind == "resource" and max_res is not None:
                if float(val) > float(max_res) + 1e-6:
                    errs.append(f"{key}={val} > {_format_num(float(max_res))}")

        violations[opt_id] = errs
        if not errs:
            passing.append(opt_id)

    return passing, violations
