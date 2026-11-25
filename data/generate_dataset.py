"""Generate and verify hard 6-option conditional multi-hop RPR datasets (20 pilot + 100 benchmark)."""

from __future__ import annotations

import json
from pathlib import Path
import random
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verifiers import check_candidate_constraints


def _assert_unique_solution(problem: dict[str, Any]) -> None:
    """Offline dataset check ensuring initial and final 6-option tables yield unique valid choices."""
    for phase, spec_key, exp_key in [
        ("initial", "initial_constraints_spec", "expected_initial"),
        ("final", "final_constraints_spec", "expected_final"),
    ]:
        spec = problem[spec_key]
        options = spec.get("options", {})
        if not options:
            continue
        valid = [
            opt_id
            for opt_id in options
            if not check_candidate_constraints({"primary": opt_id, "preserved": 0}, spec)
        ]
        expected_opt = problem[exp_key]["primary"]
        if valid != [expected_opt]:
            raise AssertionError(
                f"Problem {problem['id']} ({phase}) expected unique [{expected_opt}], got valid={valid}"
            )


def _assert_anti_anchoring(
    problem: dict[str, Any],
    *,
    cost_field: str,
    lat_field: str,
    res_field: str,
    max_cost_r1: int,
    max_lat_r1: int,
    usable_res_r1: int,
) -> None:
    """Verify post-update valid option violates >= 2 stale R1 derived thresholds."""
    if problem["category"] != "premise_change":
        return
    init_opt = problem["expected_initial"]["primary"]
    rev_opt = problem["expected_final"]["primary"]
    assert init_opt != rev_opt
    rev_attrs = problem["final_constraints_spec"]["options"][rev_opt]
    stale_breaks = (
        int(rev_attrs[cost_field] > max_cost_r1)
        + int(rev_attrs[lat_field] > max_lat_r1)
        + int(rev_attrs[res_field] > usable_res_r1)
    )
    if stale_breaks < 2:
        raise AssertionError(
            f"Problem {problem['id']} rev option {rev_opt} only breaks {stale_breaks} stale thresholds."
        )


def _make_hard_conditional_problem(
    *,
    pid: str,
    category: str,
    domain: str,
    split: str,
    title: str,
    workload_label: str,
    workload: int,
    unit_name: str,
    r1_name: str,
    r2_name: str,
    cap_r1: int,
    cap_r2: int,
    raw_res: int,
    ovh_r1: int,
    ovh_r2: int,
    res_field: str,
    res_label: str,
    budget_field: str,
    budget_cap: int,
    lat_field: str,
    lat_cap: int,
    pen_r1: int,
    pen_r2: int,
    lat_label: str,
    flag_field: str,
    inv_prose: str,
    inv_a: int,
    inv_b: int,
    inv_c: int,
    inv_d: int,
    inv_e: int,
    update_text: str | None,
    invalidated_keywords: list[str],
    rng: random.Random,
    flip_lat_or_res_bad: bool = False,
) -> dict[str, Any]:
    """Construct a 6-option (OPT-A..OPT-F) conditional multi-hop problem."""
    assert workload % cap_r1 == 0
    assert workload % cap_r2 == 0
    units_r1 = workload // cap_r1
    units_r2 = workload // cap_r2

    max_cost_r1 = budget_cap // units_r1
    max_cost_r2 = budget_cap // units_r2
    usable_res_r1 = raw_res - ovh_r1
    usable_res_r2 = raw_res - ovh_r2
    max_lat_r1 = lat_cap - pen_r1
    max_lat_r2 = lat_cap - pen_r2

    assert max_cost_r2 > max_cost_r1 + 20
    assert usable_res_r2 > usable_res_r1 + 4
    assert max_lat_r2 > max_lat_r1 + 6

    preserved_val = (inv_a * inv_b) + (inv_c * inv_d) - inv_e
    assert preserved_val > 0

    def _mid(lo: int, hi: int) -> int:
        return lo + max(1, (hi - lo) // 2)

    if category == "premise_change":
        role_map = {
            "init_valid": {
                "mode": r1_name,
                flag_field: False,
                budget_field: max_cost_r1 - max(10, max_cost_r1 // 10),
                lat_field: max_lat_r1 - max(5, max_lat_r1 // 8),
                res_field: usable_res_r1 - max(2, usable_res_r1 // 10),
            },
            "rev_valid": {
                "mode": r2_name,
                flag_field: True,
                budget_field: _mid(max_cost_r1, max_cost_r2),
                lat_field: _mid(max_lat_r1, max_lat_r2),
                res_field: _mid(usable_res_r1, usable_res_r2),
            },
            "inertia_flag_trap": {
                "mode": r2_name,
                flag_field: False,
                budget_field: max_cost_r1 - max(12, max_cost_r1 // 8),
                lat_field: max_lat_r1 - max(8, max_lat_r1 // 6),
                res_field: usable_res_r1 - max(3, usable_res_r1 // 8),
            },
            "r2_bad_budget": {
                "mode": r2_name,
                flag_field: True,
                budget_field: max_cost_r2 + max(25, max_cost_r2 // 6),
                lat_field: max_lat_r2 - max(6, max_lat_r2 // 8),
                res_field: usable_res_r2 - max(2, usable_res_r2 // 10),
            },
            "r2_bad_metric": {
                "mode": r2_name,
                flag_field: True,
                budget_field: max_cost_r2 - max(15, max_cost_r2 // 10),
                lat_field: (max_lat_r2 + max(12, max_lat_r2 // 5)) if not flip_lat_or_res_bad else max_lat_r2 - 5,
                res_field: (usable_res_r2 - 3) if not flip_lat_or_res_bad else (usable_res_r2 + max(6, usable_res_r2 // 5)),
            },
            "r1_distractor": {
                "mode": r1_name,
                flag_field: True,
                budget_field: max_cost_r1 + max(18, max_cost_r1 // 7),
                lat_field: max_lat_r1 - max(6, max_lat_r1 // 8),
                res_field: usable_res_r1 - max(3, usable_res_r1 // 9),
            },
        }
    else:
        role_map = {
            "init_valid": {
                "mode": r1_name,
                flag_field: False,
                budget_field: max_cost_r1 - max(10, max_cost_r1 // 10),
                lat_field: max_lat_r1 - max(5, max_lat_r1 // 8),
                res_field: usable_res_r1 - max(2, usable_res_r1 // 10),
            },
            "r1_bad_budget": {
                "mode": r1_name,
                flag_field: True,
                budget_field: max_cost_r1 + max(20, max_cost_r1 // 6),
                lat_field: max_lat_r1 - max(6, max_lat_r1 // 8),
                res_field: usable_res_r1 - max(3, usable_res_r1 // 10),
            },
            "r1_bad_lat": {
                "mode": r1_name,
                flag_field: False,
                budget_field: max_cost_r1 - max(15, max_cost_r1 // 9),
                lat_field: max_lat_r1 + max(10, max_lat_r1 // 5),
                res_field: usable_res_r1 - max(2, usable_res_r1 // 10),
            },
            "r1_bad_res": {
                "mode": r1_name,
                flag_field: True,
                budget_field: max_cost_r1 - max(12, max_cost_r1 // 9),
                lat_field: max_lat_r1 - max(8, max_lat_r1 // 8),
                res_field: usable_res_r1 + max(5, usable_res_r1 // 5),
            },
            "r2_distractor_valid_if_r2": {
                "mode": r2_name,
                flag_field: True,
                budget_field: _mid(max_cost_r1, max_cost_r2),
                lat_field: _mid(max_lat_r1, max_lat_r2),
                res_field: _mid(usable_res_r1, usable_res_r2),
            },
            "r2_distractor_flag": {
                "mode": r2_name,
                flag_field: False,
                budget_field: max_cost_r1 - max(14, max_cost_r1 // 8),
                lat_field: max_lat_r1 - max(7, max_lat_r1 // 7),
                res_field: usable_res_r1 - max(4, usable_res_r1 // 8),
            },
        }

    labels = ["OPT-A", "OPT-B", "OPT-C", "OPT-D", "OPT-E", "OPT-F"]
    role_keys = list(role_map.keys())
    rng.shuffle(role_keys)

    options_dict: dict[str, dict[str, Any]] = {}
    role_to_label: dict[str, str] = {}
    for lbl, rname in zip(labels, role_keys):
        options_dict[lbl] = role_map[rname]
        role_to_label[rname] = lbl

    opt_lines = "\n".join(
        f"- {lbl}: " + ", ".join(f"{k}={json.dumps(v)}" for k, v in attrs.items())
        for lbl, attrs in options_dict.items()
    )

    initial_prompt = (
        f"{title}\n"
        f"Global production rules (apply under whichever active regime is selected):\n"
        f"- Active workload demand: {workload} {workload_label}.\n"
        f"- Total budget cap: (required {unit_name}s) * {budget_field} <= {budget_cap}.\n"
        f"- Effective {lat_label} cap: {lat_field} + (regime penalty) <= {lat_cap}.\n"
        f"- Per-{unit_name} {res_label} limit: {res_field} <= {raw_res} - (regime overhead).\n"
        f"Conditional regime specifications:\n"
        f"- If active mode is {r1_name}: each {unit_name} handles {cap_r1} {workload_label}, "
        f"regime overhead is {ovh_r1}, regime penalty is {pen_r1}, and {flag_field} is optional.\n"
        f"- If active mode is {r2_name}: each {unit_name} handles {cap_r2} {workload_label}, "
        f"regime overhead is {ovh_r2}, regime penalty is {pen_r2}, and {flag_field} MUST be true.\n"
        f"Current active production regime: {r1_name} (options must match the active mode).\n"
        f"Invariant ledger calculation: {inv_prose}.\n"
        f"Tasks:\n"
        f"1) Derive the required {unit_name} count, max allowable {budget_field}, usable {res_label}, "
        f"and max {lat_field} under the active regime, then pick the unique valid option OPT-A..OPT-F for primary.\n"
        f"2) Compute the exact integer value of the invariant ledger calculation for preserved.\n"
        f"Candidate options (values are per-{unit_name}):\n{opt_lines}"
    )

    init_rules = [
        {"field": "mode", "op": "eq", "value": r1_name, "label": f"Active mode = {r1_name}"},
        {"field": budget_field, "op": "lte", "value": max_cost_r1, "label": f"{units_r1}*{budget_field} <= {budget_cap}"},
        {"field": lat_field, "op": "lte", "value": max_lat_r1, "label": f"{lat_field}+{pen_r1} <= {lat_cap}"},
        {"field": res_field, "op": "lte", "value": usable_res_r1, "label": f"{res_field} <= {raw_res}-{ovh_r1}"},
    ]

    if category == "premise_change":
        final_rules = [
            {"field": "mode", "op": "eq", "value": r2_name, "label": f"Active mode = {r2_name}"},
            {"field": flag_field, "op": "eq", "value": True, "label": f"{flag_field}=true required in {r2_name}"},
            {"field": budget_field, "op": "lte", "value": max_cost_r2, "label": f"{units_r2}*{budget_field} <= {budget_cap}"},
            {"field": lat_field, "op": "lte", "value": max_lat_r2, "label": f"{lat_field}+{pen_r2} <= {lat_cap}"},
            {"field": res_field, "op": "lte", "value": usable_res_r2, "label": f"{res_field} <= {raw_res}-{ovh_r2}"},
        ]
        exp_final_opt = role_to_label["rev_valid"]
    else:
        final_rules = list(init_rules)
        exp_final_opt = role_to_label["init_valid"]

    prob = {
        "id": pid,
        "category": category,
        "domain": domain,
        "split": split,
        "initial_prompt": initial_prompt,
        "midway_update": update_text,
        "expected_initial": {"primary": role_to_label["init_valid"], "preserved": preserved_val},
        "expected_final": {"primary": exp_final_opt, "preserved": preserved_val},
        "expected_invalidated_keywords": invalidated_keywords,
        "initial_constraints_spec": {"options": options_dict, "rules": init_rules},
        "final_constraints_spec": {"options": options_dict, "rules": final_rules},
    }
    _assert_unique_solution(prob)
    _assert_anti_anchoring(
        prob,
        cost_field=budget_field,
        lat_field=lat_field,
        res_field=res_field,
        max_cost_r1=max_cost_r1,
        max_lat_r1=max_lat_r1,
        usable_res_r1=usable_res_r1,
    )
    return prob


def build_pilot_problems() -> list[dict[str, Any]]:
    """Build 20 hard conditional pilot problems: 10 premise_change, 5 irrelevant_update, 5 ordinary."""
    rng = random.Random(20261121)

    domain_templates = [
        {
            "domain": "architecture",
            "title": "Design a real-time inference serving cluster.",
            "w_lbl": "req/s",
            "u_name": "node",
            "r1": "CPU",
            "r2": "GPU",
            "res_f": "footprint_gb",
            "res_lbl": "RAM footprint (GB)",
            "b_f": "cost_usd",
            "lat_f": "base_lat_ms",
            "lat_lbl": "latency (ms)",
            "flag_f": "tensor_fp16",
            "inv_tpl": "{a} primary shards of {b} GB plus {c} replica logs of {d} GB, minus {e} GB deduplicated header savings",
            "pc_upd": "Infrastructure change: Dedicated GPU hardware is now online and mandatory, switching the active production regime from CPU to GPU.",
            "irr_upd": "Lab benchmark memo: Next quarter's experimental GPU prototype testbed hit 180 req/s per blade with 2 GB overhead across 3 sandbox units, and the separate QA log archive added 48 GB SSD space; production cluster deployment remains on its existing hardware.",
            "kw": ["cpu", "gpu", "regime", "mode"],
        },
        {
            "domain": "scheduling",
            "title": "Schedule multi-track conference workshop sessions.",
            "w_lbl": "attendees",
            "u_name": "hall",
            "r1": "IN_PERSON",
            "r2": "HYBRID_STUDIO",
            "res_f": "batch_seats",
            "res_lbl": "seat capacity",
            "b_f": "fee_usd",
            "lat_f": "base_min",
            "lat_lbl": "turnaround (min)",
            "flag_f": "recording_rig",
            "inv_tpl": "{a} print kits of {b} workbooks plus {c} badge trays of {d} lanyards, minus {e} canceled sponsor inserts",
            "pc_upd": "Format mandate: Broadcast streaming is now required for all tracks, switching the active event regime from IN_PERSON to HYBRID_STUDIO.",
            "irr_upd": "Facilities notice: Building Annex B tested a HYBRID_STUDIO stage setup for 90 VIP guests with a 12-minute changeover on a separate floor, and cafeteria trays were restocked by 240 units; the main conference track format has not switched.",
            "kw": ["in_person", "hybrid", "regime", "mode"],
        },
        {
            "domain": "routing",
            "title": "Configure an overnight inter-city cargo haul.",
            "w_lbl": "tons",
            "u_name": "carrier",
            "r1": "DIESEL_CONVOY",
            "r2": "RAIL_EXPRESS",
            "res_f": "module_tons",
            "res_lbl": "payload mass (tons)",
            "b_f": "toll_usd",
            "lat_f": "base_hrs",
            "lat_lbl": "transit (hrs)",
            "flag_f": "cold_telemetry",
            "inv_tpl": "{a} sealed containers of {b} crates plus {c} pallet boxes of {d} brackets, minus {e} empty dunnage spacers",
            "pc_upd": "Corridor switch: Port dispatch opened the dedicated freight spur, switching active haulage from DIESEL_CONVOY to RAIL_EXPRESS.",
            "irr_upd": "Regional bulletin: A pilot RAIL_EXPRESS spur 300 km north carried 45 tons per wagon during Tuesday's dry run and updated 18 yard signs, while this corridor's active production dispatch remains unchanged.",
            "kw": ["diesel", "rail", "regime", "mode"],
        },
        {
            "domain": "budget",
            "title": "Provision high-throughput research storage pools.",
            "w_lbl": "kIOPS",
            "u_name": "array",
            "r1": "STANDARD_HDD",
            "r2": "NVME_FABRIC",
            "res_f": "replica_tb",
            "res_lbl": "replica allocation (TB)",
            "b_f": "monthly_usd",
            "lat_f": "base_read_ms",
            "lat_lbl": "read delay (ms)",
            "flag_f": "hw_encrypt",
            "inv_tpl": "{a} compute racks with {b} backup tapes each plus {c} cold shelves of {d} cartridges, minus {e} decommissioned spares",
            "pc_upd": "Storage cutover: Flash fabric arrays passed acceptance, switching the active storage regime from STANDARD_HDD to NVME_FABRIC.",
            "irr_upd": "Vendor roadmap: Next year's NVME_FABRIC preview chassis claims 240 kIOPS and 4 TB parity in vendor lab slides, and office printer toner inventory grew by 32 boxes; active storage pool tier selection has not changed.",
            "kw": ["hdd", "nvme", "regime", "mode"],
        },
        {
            "domain": "security",
            "title": "Deploy perimeter VPN authentication gateways.",
            "w_lbl": "tunnels",
            "u_name": "gateway",
            "r1": "LEGACY_VPN",
            "r2": "ZERO_TRUST_ASIC",
            "res_f": "policy_mb",
            "res_lbl": "policy cache (MB)",
            "b_f": "license_usd",
            "lat_f": "base_auth_ms",
            "lat_lbl": "auth latency (ms)",
            "flag_f": "post_quantum",
            "inv_tpl": "{a} branch offices with {b} service certs each plus {c} enclave nodes of {d} root tokens, minus {e} revoked legacy keys",
            "pc_upd": "Security directive: Hardware cryptographic accelerators are now enforced, switching active gateway mode from LEGACY_VPN to ZERO_TRUST_ASIC.",
            "irr_upd": "Advisory note: A red-team sandbox tested ZERO_TRUST_ASIC firmware with 16 MB rule tables across 5 isolated test hosts and rotated 40 internal wiki tokens; production gateway enforcement mode remains as configured.",
            "kw": ["legacy", "zero_trust", "asic", "regime", "mode"],
        },
    ]

    # Pre-chosen diverse numerical parameters for the 10 premise_change, 5 irrelevant_update, 5 ordinary pilot problems
    pc_params = [
        (120, 30, 60, 64, 24, 8, 800, 120, 40, 10, 14, 18, 9, 12, 35, False),
        (180, 30, 45, 70, 25, 10, 1200, 60, 20, 5, 18, 15, 6, 14, 28, True),
        (120, 20, 30, 50, 18, 6, 420, 20, 9, 2, 9, 14, 8, 11, 19, False),
        (240, 40, 80, 80, 30, 12, 1200, 50, 20, 5, 12, 16, 7, 15, 42, True),
        (600, 100, 200, 96, 36, 12, 1800, 45, 18, 4, 16, 12, 11, 9, 27, False),
        (300, 50, 100, 60, 22, 8, 1500, 100, 35, 10, 13, 21, 5, 17, 33, True),
        (240, 60, 120, 72, 26, 8, 960, 70, 25, 7, 8, 19, 12, 13, 45, False),
        (360, 60, 90, 64, 24, 8, 1440, 30, 12, 3, 15, 14, 9, 16, 31, True),
        (480, 80, 160, 88, 32, 10, 1800, 80, 30, 8, 11, 22, 7, 18, 29, False),
        (6000, 1000, 2000, 128, 48, 16, 1800, 40, 15, 4, 24, 15, 10, 12, 54, True),
    ]

    problems: list[dict[str, Any]] = []
    for idx, params in enumerate(pc_params, start=1):
        tpl = domain_templates[(idx - 1) % len(domain_templates)]
        (w, c1, c2, raw_r, o1, o2, b_cap, l_cap, p1, p2, ia, ib, ic, id_, ie, flip_bad) = params
        inv_prose = tpl["inv_tpl"].format(a=ia, b=ib, c=ic, d=id_, e=ie)
        problems.append(
            _make_hard_conditional_problem(
                pid=f"pilot-pc-{idx:02d}",
                category="premise_change",
                domain=tpl["domain"],
                split="pilot",
                title=tpl["title"],
                workload_label=tpl["w_lbl"],
                workload=w,
                unit_name=tpl["u_name"],
                r1_name=tpl["r1"],
                r2_name=tpl["r2"],
                cap_r1=c1,
                cap_r2=c2,
                raw_res=raw_r,
                ovh_r1=o1,
                ovh_r2=o2,
                res_field=tpl["res_f"],
                res_label=tpl["res_lbl"],
                budget_field=tpl["b_f"],
                budget_cap=b_cap,
                lat_field=tpl["lat_f"],
                lat_cap=l_cap,
                pen_r1=p1,
                pen_r2=p2,
                lat_label=tpl["lat_lbl"],
                flag_field=tpl["flag_f"],
                inv_prose=inv_prose,
                inv_a=ia,
                inv_b=ib,
                inv_c=ic,
                inv_d=id_,
                inv_e=ie,
                update_text=tpl["pc_upd"],
                invalidated_keywords=tpl["kw"],
                rng=rng,
                flip_lat_or_res_bad=flip_bad,
            )
        )

    irr_params = [
        (160, 40, 80, 64, 20, 6, 1000, 90, 30, 8, 9, 14, 6, 11, 22),
        (240, 60, 120, 80, 28, 8, 1200, 50, 18, 5, 7, 18, 8, 12, 19),
        (180, 30, 60, 56, 20, 6, 900, 24, 10, 2, 11, 13, 5, 16, 27),
        (360, 60, 120, 72, 24, 8, 1800, 60, 22, 6, 13, 17, 9, 10, 31),
        (400, 100, 200, 96, 32, 10, 1600, 45, 16, 4, 15, 12, 7, 14, 25),
    ]
    for idx, params in enumerate(irr_params, start=1):
        tpl = domain_templates[(idx - 1) % len(domain_templates)]
        (w, c1, c2, raw_r, o1, o2, b_cap, l_cap, p1, p2, ia, ib, ic, id_, ie) = params
        inv_prose = tpl["inv_tpl"].format(a=ia, b=ib, c=ic, d=id_, e=ie)
        problems.append(
            _make_hard_conditional_problem(
                pid=f"pilot-irr-{idx:02d}",
                category="irrelevant_update",
                domain=tpl["domain"],
                split="pilot",
                title=tpl["title"],
                workload_label=tpl["w_lbl"],
                workload=w,
                unit_name=tpl["u_name"],
                r1_name=tpl["r1"],
                r2_name=tpl["r2"],
                cap_r1=c1,
                cap_r2=c2,
                raw_res=raw_r,
                ovh_r1=o1,
                ovh_r2=o2,
                res_field=tpl["res_f"],
                res_label=tpl["res_lbl"],
                budget_field=tpl["b_f"],
                budget_cap=b_cap,
                lat_field=tpl["lat_f"],
                lat_cap=l_cap,
                pen_r1=p1,
                pen_r2=p2,
                lat_label=tpl["lat_lbl"],
                flag_field=tpl["flag_f"],
                inv_prose=inv_prose,
                inv_a=ia,
                inv_b=ib,
                inv_c=ic,
                inv_d=id_,
                inv_e=ie,
                update_text=tpl["irr_upd"],
                invalidated_keywords=[],
                rng=rng,
            )
        )

    ord_params = [
        (140, 35, 70, 60, 18, 6, 1200, 80, 25, 8, 14, 25, 6, 13, 38),
        (192, 32, 64, 72, 24, 8, 1440, 55, 20, 6, 8, 19, 11, 9, 21),
        (150, 25, 50, 52, 16, 4, 1500, 28, 10, 3, 12, 16, 7, 15, 29),
        (300, 50, 100, 84, 28, 10, 2400, 75, 25, 7, 9, 24, 8, 12, 34),
        (600, 100, 200, 100, 36, 12, 2400, 50, 18, 5, 18, 11, 9, 14, 26),
    ]
    for idx, params in enumerate(ord_params, start=1):
        tpl = domain_templates[(idx - 1) % len(domain_templates)]
        (w, c1, c2, raw_r, o1, o2, b_cap, l_cap, p1, p2, ia, ib, ic, id_, ie) = params
        inv_prose = tpl["inv_tpl"].format(a=ia, b=ib, c=ic, d=id_, e=ie)
        problems.append(
            _make_hard_conditional_problem(
                pid=f"pilot-ord-{idx:02d}",
                category="ordinary",
                domain=tpl["domain"],
                split="pilot",
                title=tpl["title"],
                workload_label=tpl["w_lbl"],
                workload=w,
                unit_name=tpl["u_name"],
                r1_name=tpl["r1"],
                r2_name=tpl["r2"],
                cap_r1=c1,
                cap_r2=c2,
                raw_res=raw_r,
                ovh_r1=o1,
                ovh_r2=o2,
                res_field=tpl["res_f"],
                res_label=tpl["res_lbl"],
                budget_field=tpl["b_f"],
                budget_cap=b_cap,
                lat_field=tpl["lat_f"],
                lat_cap=l_cap,
                pen_r1=p1,
                pen_r2=p2,
                lat_label=tpl["lat_lbl"],
                flag_field=tpl["flag_f"],
                inv_prose=inv_prose,
                inv_a=ia,
                inv_b=ib,
                inv_c=ic,
                inv_d=id_,
                inv_e=ie,
                update_text=None,
                invalidated_keywords=[],
                rng=rng,
            )
        )

    return problems


def build_benchmark_100(seed: int = 20261121) -> list[dict[str, Any]]:
    """Generate 100 hard conditional multi-hop benchmark problems (50 ordinary, 35 premise_change, 15 irrelevant_update)."""
    rng = random.Random(seed)
    domain_templates = [
        ("architecture", "req/s", "node", "CPU", "GPU", "footprint_gb", "RAM (GB)", "cost_usd", "base_lat_ms", "latency (ms)", "tensor_fp16",
         "{a} primary shards of {b} GB plus {c} replica logs of {d} GB, minus {e} GB deduplicated header savings"),
        ("scheduling", "attendees", "hall", "IN_PERSON", "HYBRID_STUDIO", "batch_seats", "seat batch", "fee_usd", "base_min", "turnaround (min)", "recording_rig",
         "{a} print kits of {b} workbooks plus {c} badge trays of {d} lanyards, minus {e} canceled inserts"),
        ("routing", "tons", "carrier", "DIESEL_CONVOY", "RAIL_EXPRESS", "module_tons", "payload (tons)", "toll_usd", "base_hrs", "transit (hrs)", "cold_telemetry",
         "{a} containers of {b} crates plus {c} pallet boxes of {d} brackets, minus {e} empty spacers"),
        ("budget", "kIOPS", "array", "STANDARD_HDD", "NVME_FABRIC", "replica_tb", "replica (TB)", "monthly_usd", "base_read_ms", "read delay (ms)", "hw_encrypt",
         "{a} racks with {b} backup tapes each plus {c} shelves of {d} cartridges, minus {e} retired spares"),
        ("security", "tunnels", "gateway", "LEGACY_VPN", "ZERO_TRUST_ASIC", "policy_mb", "policy cache (MB)", "license_usd", "base_auth_ms", "auth delay (ms)", "post_quantum",
         "{a} offices with {b} certs each plus {c} enclaves of {d} root tokens, minus {e} revoked keys"),
    ]

    categories = ["ordinary"] * 50 + ["premise_change"] * 35 + ["irrelevant_update"] * 15
    rng.shuffle(categories)
    problems: list[dict[str, Any]] = []

    for idx, cat in enumerate(categories, start=1):
        dom, w_lbl, u_name, r1, r2, res_f, res_lbl, b_f, lat_f, lat_lbl, flag_f, inv_tpl = domain_templates[idx % len(domain_templates)]
        split = "dev" if idx % 2 == 0 else "test"

        cap_r1 = rng.choice([15, 20, 25, 30, 40])
        units_r1 = rng.choice([4, 6])
        workload = cap_r1 * units_r1
        cap_r2 = cap_r1 * 2

        unit_budget_r1 = rng.randint(80, 180)
        budget_cap = units_r1 * unit_budget_r1
        raw_res = rng.randint(60, 110)
        ovh_r1 = rng.randint(20, 32)
        ovh_r2 = rng.randint(6, 12)
        lat_cap = rng.randint(50, 95)
        pen_r1 = rng.randint(18, 30)
        pen_r2 = rng.randint(4, 10)

        ia, ib = rng.randint(6, 16), rng.randint(11, 24)
        ic, id_ = rng.randint(4, 12), rng.randint(8, 19)
        ie = rng.randint(10, 45)
        inv_prose = inv_tpl.format(a=ia, b=ib, c=ic, d=id_, e=ie)

        if cat == "premise_change":
            upd = f"Regime change: Active production mode switched from {r1} to {r2}."
            kw = [r1.lower(), r2.lower(), "regime", "mode"]
        elif cat == "irrelevant_update":
            d_val = rng.randint(25, 95)
            upd = (
                f"Staging bulletin: A separate QA prototype lab tested {r2} with {d_val} {w_lbl} per test {u_name} "
                f"and added {d_val + 12} log archives; active production {u_name} mode has not switched."
            )
            kw = []
        else:
            upd = None
            kw = []

        problems.append(
            _make_hard_conditional_problem(
                pid=f"bench-{idx:03d}",
                category=cat,
                domain=dom,
                split=split,
                title=f"Domain: {dom}. Configure production {u_name}s and select the valid option.",
                workload_label=w_lbl,
                workload=workload,
                unit_name=u_name,
                r1_name=r1,
                r2_name=r2,
                cap_r1=cap_r1,
                cap_r2=cap_r2,
                raw_res=raw_res,
                ovh_r1=ovh_r1,
                ovh_r2=ovh_r2,
                res_field=res_f,
                res_label=res_lbl,
                budget_field=b_f,
                budget_cap=budget_cap,
                lat_field=lat_f,
                lat_cap=lat_cap,
                pen_r1=pen_r1,
                pen_r2=pen_r2,
                lat_label=lat_lbl,
                flag_field=flag_f,
                inv_prose=inv_prose,
                inv_a=ia,
                inv_b=ib,
                inv_c=ic,
                inv_d=id_,
                inv_e=ie,
                update_text=upd,
                invalidated_keywords=kw,
                rng=rng,
                flip_lat_or_res_bad=(idx % 2 == 1),
            )
        )

    return problems


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def main() -> None:
    data_dir = Path(__file__).resolve().parent
    pilot = build_pilot_problems()
    bench = build_benchmark_100()
    write_jsonl(data_dir / "pilot_problems.jsonl", pilot)
    write_jsonl(data_dir / "benchmark_100.jsonl", bench)
    print(f"Wrote {len(pilot)} hard conditional pilot problems and {len(bench)} hard benchmark problems.")


if __name__ == "__main__":
    main()
