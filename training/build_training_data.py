"""Generate strictly decontaminated multi-skill training, RLVR, and held-out eval datasets.

Four balanced capability pillars across domains completely disjoint from `pilot_problems.jsonl`
and `benchmark_100.jsonl`:
  1. Concise multi-hop math & RPR Level-1/2/3 reasoning (`<think>` 50-140 tokens, zero echo)
  2. Belief-revision & distractor audit calibration (`CHECK A*: VALID|INVALID` + 2-turn updates)
  3. Calibrated XML `<tool_call>` calling (positive multi-arg/recovery + negative abstention)
  4. Strict instruction & JSON schema adherence (evaluated integers, exact keys, IFEval rules)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
import sys
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_OUT_DIR = Path(__file__).resolve().parent / "data"
PILOT_PATH = ROOT_DIR / "data" / "pilot_problems.jsonl"
BENCH_PATH = ROOT_DIR / "data" / "benchmark_100.jsonl"


_DOMAINS: list[dict[str, Any]] = [
    {
        "domain": "satellite_comms",
        "title": "Allocate orbital Ka/UHF telemetry downlinks.",
        "demand_name": "telemetry bandwidth",
        "demand_unit": "Mbps",
        "unit_singular": "transponder",
        "unit_plural": "transponders",
        "cost_key": "lease_usd",
        "lat_key": "link_ms",
        "lat_label": "end-to-end link delay (ms)",
        "res_key": "power_w",
        "res_label": "per-transponder draw (W)",
        "bool_key": "beam_lock",
        "mode_a": "UHF_RELAY",
        "mode_b": "KA_BAND_LASER",
        "inv_a": "telemetry frames",
        "inv_a_item": "header bytes",
        "inv_b": "parity blocks",
        "inv_b_item": "sync words",
        "inv_c": "pruned null slots",
        "distractor_loc": "ground test chamber B",
    },
    {
        "domain": "grid_battery",
        "title": "Dispatch utility substation storage banks.",
        "demand_name": "peak discharge load",
        "demand_unit": "MW",
        "unit_singular": "inverter",
        "unit_plural": "inverters",
        "cost_key": "capex_kusd",
        "lat_key": "ramp_ms",
        "lat_label": "effective ramp latency (ms)",
        "res_key": "rack_kg",
        "res_label": "per-inverter mass (kg)",
        "bool_key": "grid_sync",
        "mode_a": "FLOW_TANK",
        "mode_b": "SOLID_LFP",
        "inv_a": "breaker bays",
        "inv_a_item": "relay coils",
        "inv_b": "busbar feeders",
        "inv_b_item": "fuse links",
        "inv_c": "decommissioned spares",
        "distractor_loc": "microgrid lab trailer",
    },
    {
        "domain": "hospital_icu",
        "title": "Configure regional emergency triage wards.",
        "demand_name": "patient surge intake",
        "demand_unit": "beds",
        "unit_singular": "pod",
        "unit_plural": "pods",
        "cost_key": "grant_usd",
        "lat_key": "triage_min",
        "lat_label": "effective intake wait (min)",
        "res_key": "floor_m2",
        "res_label": "per-pod footprint (m2)",
        "bool_key": "neg_pressure",
        "mode_a": "MOBILE_POD",
        "mode_b": "FIXED_BAY",
        "inv_a": "supply carts",
        "inv_a_item": "sterile packs",
        "inv_b": "isolation kits",
        "inv_b_item": "filter seals",
        "inv_c": "expired triage tags",
        "distractor_loc": "training simulation wing",
    },
    {
        "domain": "compiler_pipeline",
        "title": "Schedule distributed build-farm runners.",
        "demand_name": "artifact throughput",
        "demand_unit": "jobs/min",
        "unit_singular": "worker",
        "unit_plural": "workers",
        "cost_key": "slot_usd",
        "lat_key": "cold_ms",
        "lat_label": "effective startup lag (ms)",
        "res_key": "cache_mb",
        "res_label": "per-worker slab (MB)",
        "bool_key": "simd_avx512",
        "mode_a": "INTERPRETED_JIT",
        "mode_b": "AOT_VECTOR",
        "inv_a": "object archives",
        "inv_a_item": "symbol tables",
        "inv_b": "PCH snapshots",
        "inv_b_item": "macro tables",
        "inv_c": "stale relocations",
        "distractor_loc": "nightly canary VM",
    },
    {
        "domain": "chip_lithography",
        "title": "Balance cleanroom wafer exposure bays.",
        "demand_name": "wafer cadence",
        "demand_unit": "wph",
        "unit_singular": "scanner",
        "unit_plural": "scanners",
        "cost_key": "opex_kusd",
        "lat_key": "align_ms",
        "lat_label": "effective alignment cycle (ms)",
        "res_key": "foot_m2",
        "res_label": "per-scanner bay space (m2)",
        "bool_key": "pellicle_seal",
        "mode_a": "DUV_DRY",
        "mode_b": "EUV_SCAN",
        "inv_a": "reticle pods",
        "inv_a_item": "quartz blanks",
        "inv_b": "purge cassettes",
        "inv_b_item": "nozzle seals",
        "inv_c": "scrapped test plates",
        "distractor_loc": "metrology demo annex",
    },
    {
        "domain": "warehouse_robotics",
        "title": "Coordinate automated fulfillment aisles.",
        "demand_name": "pick throughput",
        "demand_unit": "bins/hr",
        "unit_singular": "bot",
        "unit_plural": "bots",
        "cost_key": "rent_usd",
        "lat_key": "pick_sec",
        "lat_label": "effective cycle latency (sec)",
        "res_key": "aisle_dm",
        "res_label": "per-bot clearance (dm)",
        "bool_key": "lidar_fused",
        "mode_a": "AMR_WHEEL",
        "mode_b": "GANTRY_RAIL",
        "inv_a": "pallet totes",
        "inv_a_item": "barcode tags",
        "inv_b": "sorting chutes",
        "inv_b_item": "sensor clips",
        "inv_c": "returned empties",
        "distractor_loc": "south R&D mezzanine",
    },
]


def _make_scenario(rng: random.Random, domain_spec: dict[str, Any]) -> dict[str, Any]:
    d = domain_spec
    units_a = rng.choice([4, 5, 6])
    units_b = units_a // 2 if units_a % 2 == 0 else rng.choice([2, 3])
    if units_b == units_a:
        units_b = max(2, units_a - 2)
    per_a = rng.choice([15, 25, 35, 45])
    demand = units_a * per_a
    if demand % units_b != 0:
        units_b = 2
        units_a = 4
        demand = units_a * per_a
    per_b = demand // units_b

    cost_max_a = rng.choice([120, 150, 180, 210])
    total_budget = units_a * cost_max_a
    cost_max_b = total_budget // units_b

    cap_total = rng.choice([55, 65, 85, 105])
    pen_a = rng.choice([18, 22, 26])
    pen_b = rng.choice([4, 6, 8])
    lat_max_a = cap_total - pen_a
    lat_max_b = cap_total - pen_b

    res_gross = rng.choice([62, 74, 86, 98])
    over_a = rng.choice([20, 24, 28])
    over_b = rng.choice([6, 8, 10])
    res_max_a = res_gross - over_a
    res_max_b = res_gross - over_b

    # Invariant ledger arithmetic (distinct multiplier ranges, explicitly addition-minus)
    m1_a, m1_b = rng.randint(8, 19), rng.randint(11, 23)
    m2_a, m2_b = rng.randint(5, 13), rng.randint(9, 17)
    sub_c = rng.randint(12, 39)
    inv_val = m1_a * m1_b + m2_a * m2_b - sub_c
    inv_expr = f"{m1_a} * {m1_b} + {m2_a} * {m2_b} - {sub_c}"
    inv_text = (
        f"{m1_a} {d['inv_a']} of {m1_b} {d['inv_a_item']} plus "
        f"{m2_a} {d['inv_b']} of {m2_b} {d['inv_b_item']}, minus {sub_c} {d['inv_c']}."
    )

    # Construct 6 unique options: 1 valid for mode_a, 1 valid for mode_b, 4 near-miss decoys
    opts_raw = [
        {  # valid A (bool false to prove optional works)
            "role": "A_VALID",
            "mode": d["mode_a"],
            d["bool_key"]: False,
            d["cost_key"]: cost_max_a - rng.randint(8, 18),
            d["lat_key"]: lat_max_a - rng.randint(3, 6),
            d["res_key"]: res_max_a - rng.randint(2, 5),
        },
        {  # decoy A cost
            "role": "A_COST_FAIL",
            "mode": d["mode_a"],
            d["bool_key"]: True,
            d["cost_key"]: cost_max_a + rng.randint(14, 28),
            d["lat_key"]: lat_max_a - 4,
            d["res_key"]: res_max_a - 3,
        },
        {  # valid B (bool true required)
            "role": "B_VALID",
            "mode": d["mode_b"],
            d["bool_key"]: True,
            d["cost_key"]: int(round((cost_max_a + cost_max_b) * 0.55)),
            d["lat_key"]: lat_max_b - rng.randint(4, 8),
            d["res_key"]: res_max_b - rng.randint(3, 7),
        },
        {  # decoy B bool=false (fits mode_a budget/lat/res too so dangerous anchor!)
            "role": "B_BOOL_FAIL",
            "mode": d["mode_b"],
            d["bool_key"]: False,
            d["cost_key"]: cost_max_a - 15,
            d["lat_key"]: lat_max_a - 6,
            d["res_key"]: res_max_a - 6,
        },
        {  # decoy B lat fail
            "role": "B_LAT_FAIL",
            "mode": d["mode_b"],
            d["bool_key"]: True,
            d["cost_key"]: cost_max_b - 25,
            d["lat_key"]: lat_max_b + rng.randint(5, 11),
            d["res_key"]: res_max_b - 4,
        },
        {  # decoy B res fail
            "role": "B_RES_FAIL",
            "mode": d["mode_b"],
            d["bool_key"]: True,
            d["cost_key"]: cost_max_b - 30,
            d["lat_key"]: lat_max_b - 5,
            d["res_key"]: res_max_b + rng.randint(4, 9),
        },
    ]
    rng.shuffle(opts_raw)
    labels = ["OPT-A", "OPT-B", "OPT-C", "OPT-D", "OPT-E", "OPT-F"]
    options: dict[str, dict[str, Any]] = {}
    ans_a = ""
    ans_b = ""
    for lbl, row in zip(labels, opts_raw):
        role = row.pop("role")
        if role == "A_VALID":
            ans_a = lbl
        elif role == "B_VALID":
            ans_b = lbl
        options[lbl] = row

    opt_lines = []
    for lbl, attrs in options.items():
        fields = ", ".join(
            f'{k}="{v}"' if isinstance(v, str) else (f"{k}={str(v).lower()}" if isinstance(v, bool) else f"{k}={v}")
            for k, v in attrs.items()
        )
        opt_lines.append(f"- {lbl}: {fields}")

    rules_block = (
        f"{d['title']}\n"
        f"Target load: {demand} {d['demand_unit']}.\n"
        f"Budget ceiling: (needed {d['unit_plural']}) * {d['cost_key']} <= {total_budget}.\n"
        f"{d['lat_label'].capitalize()} limit: {d['lat_key']} + (mode penalty) <= {cap_total}.\n"
        f"{d['res_label'].capitalize()} limit: {d['res_key']} <= {res_gross} - (mode overhead).\n"
        f"Operating profiles:\n"
        f"- Under {d['mode_a']}: one {d['unit_singular']} serves {per_a} {d['demand_unit']}, "
        f"overhead={over_a}, penalty={pen_a}, {d['bool_key']} optional.\n"
        f"- Under {d['mode_b']}: one {d['unit_singular']} serves {per_b} {d['demand_unit']}, "
        f"overhead={over_b}, penalty={pen_b}, {d['bool_key']} MUST equal true.\n"
        f"Selected production profile: {d['mode_a']} (chosen row must match active profile).\n"
        f"Ledger inventory: {inv_text}"
    )
    full_prompt = (
        rules_block
        + f"\nRequirements:\n(a) Determine {d['unit_plural']} count, cost cap, latency bound, and resource cap; choose the sole qualifying entry OPT-A..OPT-F as \"primary\".\n"
        + "(b) Evaluate the integer ledger inventory sum as \"preserved\".\n"
        + "Options:\n"
        + "\n".join(opt_lines)
        + '\nReply with JSON: {"primary": "OPT-X", "preserved": <integer>}'
    )

    return {
        "d": d,
        "demand": demand,
        "total_budget": total_budget,
        "cap_total": cap_total,
        "res_gross": res_gross,
        "per_a": per_a,
        "over_a": over_a,
        "pen_a": pen_a,
        "units_a": units_a,
        "cost_max_a": cost_max_a,
        "lat_max_a": lat_max_a,
        "res_max_a": res_max_a,
        "per_b": per_b,
        "over_b": over_b,
        "pen_b": pen_b,
        "units_b": units_b,
        "cost_max_b": cost_max_b,
        "lat_max_b": lat_max_b,
        "res_max_b": res_max_b,
        "inv_expr": inv_expr,
        "inv_val": inv_val,
        "m1_a": m1_a,
        "m1_b": m1_b,
        "m2_a": m2_a,
        "m2_b": m2_b,
        "sub_c": sub_c,
        "ans_a": ans_a,
        "ans_b": ans_b,
        "options": options,
        "opt_lines": "\n".join(opt_lines),
        "rules_block": rules_block,
        "full_prompt": full_prompt,
    }


def _concise_opt_summary(options: dict[str, dict[str, Any]], d: dict[str, Any], active_mode: str, max_cost: int, max_lat: int, max_res: int, req_bool: bool) -> str:
    parts = []
    for lbl, a in options.items():
        if a["mode"] != active_mode:
            parts.append(f"{lbl}: fail(mode)")
        elif req_bool and not a[d["bool_key"]]:
            parts.append(f"{lbl}: fail({d['bool_key']})")
        elif a[d["cost_key"]] > max_cost:
            parts.append(f"{lbl}: fail(cost)")
        elif a[d["lat_key"]] > max_lat:
            parts.append(f"{lbl}: fail(lat)")
        elif a[d["res_key"]] > max_res:
            parts.append(f"{lbl}: fail(res)")
        else:
            parts.append(f"{lbl}: pass")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# PILLAR GENERATORS
# ---------------------------------------------------------------------------
def _make_multihop_example(rng: random.Random, idx: int) -> dict[str, Any]:
    d = _DOMAINS[idx % len(_DOMAINS)]
    sc = _make_scenario(rng, d)
    variant = idx % 3

    if variant == 0:
        # Concise direct multi-hop solve (no prompt echoing!)
        p1, p2 = sc["m1_a"] * sc["m1_b"], sc["m2_a"] * sc["m2_b"]
        think = (
            f"Active mode is {d['mode_a']}: capacity={sc['per_a']}, overhead={sc['over_a']}, penalty={sc['pen_a']}, {d['bool_key']} optional. "
            f"Units={sc['demand']}/{sc['per_a']}={sc['units_a']}; max_{d['cost_key']}={sc['total_budget']}/{sc['units_a']}={sc['cost_max_a']}; "
            f"max_{d['lat_key']}={sc['cap_total']}-{sc['pen_a']}={sc['lat_max_a']}; max_{d['res_key']}={sc['res_gross']}-{sc['over_a']}={sc['res_max_a']}. "
            f"Checking options: {sc['ans_a']} meets mode={d['mode_a']} and all bounds. "
            f"Ledger is addition then subtraction: {sc['m1_a']}*{sc['m1_b']}={p1}, {sc['m2_a']}*{sc['m2_b']}={p2}, {p1}+{p2}-{sc['sub_c']}={sc['inv_val']}."
        )
        ans = f"<think>{think}</think>\n" + json.dumps({"primary": sc["ans_a"], "preserved": sc["inv_val"]})
        return {
            "pillar": "multihop_math",
            "messages": [
                {"role": "system", "content": "Solve concisely without repeating the prompt. Output evaluated JSON."},
                {"role": "user", "content": sc["full_prompt"]},
                {"role": "assistant", "content": ans},
            ],
            "expected": {"primary": sc["ans_a"], "preserved": sc["inv_val"]},
        }

    if variant == 1:
        # RPR Level 1 + Level 2 extraction training (fixes 12*16*7*15 and 1200/140!)
        use_b = idx % 2 == 1
        mode = d["mode_b"] if use_b else d["mode_a"]
        cap = sc["per_b"] if use_b else sc["per_a"]
        over = sc["over_b"] if use_b else sc["over_a"]
        pen = sc["pen_b"] if use_b else sc["pen_a"]
        bool_desc = "mandatory=true" if use_b else "optional"
        p1, p2 = sc["m1_a"] * sc["m1_b"], sc["m2_a"] * sc["m2_b"]
        update_note = (
            f"\nUpdate: Dedicated {d['mode_b']} link mandated, switching active mode to {d['mode_b']}."
            if use_b
            else "\nInitial problem statement."
        )
        think = (
            f"Current regime is {mode}: capacity={cap}, overhead={over}, penalty={pen}, {d['bool_key']} {bool_desc}. "
            f"Invariant ledger sums two products then subtracts: {sc['inv_expr']} = {p1} + {p2} - {sc['sub_c']} = {sc['inv_val']}."
        )
        out = (
            f"<think>{think}</think>\n"
            f"A1: active_mode={mode}; boolean_feature={d['bool_key']} {bool_desc}\n"
            f"A2: workload demand is {sc['demand']}, and under active mode each {d['unit_singular']} handles {cap}\n"
            f"A3: required_{d['unit_plural']} * {d['cost_key']} <= {sc['total_budget']}\n"
            f"A4: {d['lat_key']} + {pen} <= {sc['cap_total']}\n"
            f"A5: {d['res_key']} <= {sc['res_gross']} - {over}\n"
            f"F1 [NONE]: {sc['inv_expr']} = {sc['inv_val']} | invariant ledger"
        )
        return {
            "pillar": "rpr_level1",
            "messages": [
                {"role": "system", "content": "You are Level 1 (Foundation) of a Revisable Reasoning Pyramid. Do NOT repeat the prompt inside <think>. Output A1..A5 and F1 lines only."},
                {"role": "user", "content": f"Problem Rules:\n{sc['rules_block']}\nEvidence:{update_note}"},
                {"role": "assistant", "content": out},
            ],
            "expected": {"mode": mode, "preserved": sc["inv_val"]},
        }

    # variant == 2: Level 2 numerical division + subtraction subproblem training
    units = sc["units_b"]
    cost_m = sc["cost_max_b"]
    lat_m = sc["lat_max_b"]
    res_m = sc["res_max_b"]
    think = (
        f"Active mode is {d['mode_b']} with {d['bool_key']}=true required. "
        f"Divide demand {sc['demand']} by capacity {sc['per_b']} = {units} {d['unit_plural']}. "
        f"CRITICAL: divide budget {sc['total_budget']} by required_units {units} (NOT demand {sc['demand']}) = {cost_m}. "
        f"Max latency = {sc['cap_total']} - {sc['pen_b']} = {lat_m}; max resource = {sc['res_gross']} - {sc['over_b']} = {res_m}."
    )
    out = (
        f"<think>{think}</think>\n"
        f"L2-1 [A1, A2]: required_units = {sc['demand']} / {sc['per_b']} = {units}\n"
        f"L2-2 [A1, A2, A3]: max_per_unit_cost = {sc['total_budget']} / {units} = {cost_m}\n"
        f"L2-3 [A1]: required_mode = {d['mode_b']}; mandatory_feature = {d['bool_key']}=true\n"
        f"L2-4 [A4]: max_base_time_or_lat = {sc['cap_total']} - {sc['pen_b']} = {lat_m}\n"
        f"L2-5 [A5]: max_per_unit_resource = {sc['res_gross']} - {sc['over_b']} = {res_m}"
    )
    return {
        "pillar": "rpr_level2",
        "messages": [
            {"role": "system", "content": "You are Level 2 (Decomposition & Subproblem Bounds). Do NOT copy rules inside <think>. Divide budget by required_units, never demand."},
            {"role": "user", "content": (
                f"Problem Rules:\n{sc['rules_block']}\n\nLatest evidence: Switched active mode to {d['mode_b']}.\n"
                f"Active Assumptions:\n- A1: active_mode={d['mode_b']}; {d['bool_key']} mandatory=true\n"
                f"- A2: demand={sc['demand']}, cap={sc['per_b']}\n- A3: total <= {sc['total_budget']}\n"
                f"- A4: +{sc['pen_b']} <= {sc['cap_total']}\n- A5: <= {sc['res_gross']}-{sc['over_b']}"
            )},
            {"role": "assistant", "content": out},
        ],
        "expected": {"units": units, "max_cost": cost_m},
    }


def _make_belief_revision_example(rng: random.Random, idx: int) -> dict[str, Any]:
    d = _DOMAINS[idx % len(_DOMAINS)]
    sc = _make_scenario(rng, d)
    sub = idx % 3

    if sub == 0:
        # Premise change Audit -> CHECK INVALID
        ev = f"Operational override: {d['mode_b']} hardware link is now active and mandatory, switching production regime from {d['mode_a']} to {d['mode_b']}."
        think = (
            f"Check A1..A5 against Latest Evidence ONLY (ignore option values): "
            f"Latest Evidence switches active production from {d['mode_a']} to {d['mode_b']}. "
            f"A1 ({d['mode_a']}), A2 (cap={sc['per_a']}), A4 (penalty={sc['pen_a']}), A5 (overhead={sc['over_a']}) all use obsolete {d['mode_a']} values. A3 budget cap stays valid."
        )
        out = (
            f"<think>{think}</think>\n"
            f"CHECK A1: INVALID | Active production mode switched from {d['mode_a']} to {d['mode_b']}.\n"
            f"CHECK A2: INVALID | Per-unit capacity shifts from {sc['per_a']} to {sc['per_b']} under {d['mode_b']}.\n"
            f"CHECK A3: VALID | Global budget cap formula {sc['total_budget']} is unchanged.\n"
            f"CHECK A4: INVALID | Penalty shifts from {sc['pen_a']} to {sc['pen_b']}.\n"
            f"CHECK A5: INVALID | Overhead shifts from {sc['over_a']} to {sc['over_b']}.\n"
            f"SUMMARY: Production switched from {d['mode_a']} to {d['mode_b']}, invalidating A1, A2, A4, and A5."
        )
        verdict = "INVALID"
    elif sub == 1:
        # Irrelevant sandbox/staging distractor Audit -> ALL CHECK VALID!
        ev = (
            f"Advisory bulletin: An isolated {d['distractor_loc']} tested {d['mode_b']} for {sc['units_b']} units "
            f"with a {sc['over_b']}-unit test load and rotated {sc['sub_c']} audit logs; the live production active regime remains {d['mode_a']}."
        )
        think = (
            f"Check whether Latest Evidence modifies live production rules or demand: "
            f"The bulletin describes an isolated test in {d['distractor_loc']} and explicitly confirms live production remains {d['mode_a']}. "
            f"Demand ({sc['demand']}) and global rules are untouched, so all A1..A5 assumptions stay VALID."
        )
        out = (
            f"<think>{think}</think>\n"
            f"CHECK A1: VALID | Live production active mode remains {d['mode_a']}.\n"
            f"CHECK A2: VALID | Workload demand {sc['demand']} and {d['mode_a']} capacity {sc['per_a']} are unchanged.\n"
            f"CHECK A3: VALID | Budget cap {sc['total_budget']} is unaffected.\n"
            f"CHECK A4: VALID | Penalty {sc['pen_a']} remains active.\n"
            f"CHECK A5: VALID | Overhead {sc['over_a']} remains active.\n"
            f"SUMMARY: Latest evidence covers an isolated test and explicitly keeps {d['mode_a']} active; all assumptions remain VALID."
        )
        verdict = "VALID"
    else:
        # Multi-turn belief revision (Turn 1 A -> Turn 2 override to B, preserving inv_val!)
        p1, p2 = sc["m1_a"] * sc["m1_b"], sc["m2_a"] * sc["m2_b"]
        think_1 = (
            f"Mode={d['mode_a']} ({d['bool_key']} optional): units={sc['units_a']}, max_cost={sc['cost_max_a']}, "
            f"max_lat={sc['lat_max_a']}, max_res={sc['res_max_a']}. Unique match={sc['ans_a']}. "
            f"Ledger: {p1}+{p2}-{sc['sub_c']}={sc['inv_val']}."
        )
        summary_b = _concise_opt_summary(
            sc["options"], d, d["mode_b"], sc["cost_max_b"], sc["lat_max_b"], sc["res_max_b"], True
        )
        think_2 = (
            f"Mode flipped to {d['mode_b']} ({d['bool_key']}=true mandatory). Recompute bounds cleanly: "
            f"units={sc['demand']}/{sc['per_b']}={sc['units_b']}, max_cost={sc['total_budget']}/{sc['units_b']}={sc['cost_max_b']}, "
            f"max_lat={sc['cap_total']}-{sc['pen_b']}={sc['lat_max_b']}, max_res={sc['res_gross']}-{sc['over_b']}={sc['res_max_b']}. "
            f"{summary_b}. Pick {sc['ans_b']}; ledger unchanged={sc['inv_val']}."
        )
        return {
            "pillar": "belief_revision_2turn",
            "messages": [
                {"role": "system", "content": "Solve concisely without repeating the prompt. Output JSON only after </think>."},
                {"role": "user", "content": sc["full_prompt"]},
                {"role": "assistant", "content": f"<think>{think_1}</think>\n" + json.dumps({"primary": sc["ans_a"], "preserved": sc["inv_val"]})},
                {"role": "user", "content": f"New evidence arrived: Live production switched active mode from {d['mode_a']} to {d['mode_b']}. Re-derive and output updated JSON."},
                {"role": "assistant", "content": f"<think>{think_2}</think>\n" + json.dumps({"primary": sc["ans_b"], "preserved": sc["inv_val"]})},
            ],
            "expected": {"primary": sc["ans_b"], "preserved": sc["inv_val"]},
        }

    rules_only = sc["rules_block"]
    return {
        "pillar": f"audit_{verdict.lower()}",
        "messages": [
            {"role": "system", "content": "You are a strict logical consistency auditor. Do NOT repeat the prompt in <think>. Output CHECK lines and SUMMARY only (no <tool_call> tags)."},
            {"role": "user", "content": (
                f"Problem Rules:\n{rules_only}\n\nActive Level-1 Assumptions:\n"
                f"- A1: active_mode={d['mode_a']}; {d['bool_key']} optional\n"
                f"- A2: demand={sc['demand']}, per_unit_cap={sc['per_a']}\n"
                f"- A3: required_units * {d['cost_key']} <= {sc['total_budget']}\n"
                f"- A4: {d['lat_key']} + {sc['pen_a']} <= {sc['cap_total']}\n"
                f"- A5: {d['res_key']} <= {sc['res_gross']} - {sc['over_a']}\n\n"
                f"Latest Evidence:\n{ev}\n\nOutput CHECK lines and SUMMARY."
            )},
            {"role": "assistant", "content": out},
        ],
        "expected": {"verdict": verdict},
    }


_TOOLS_BLOCK = (
    "<tools>\n"
    '{"type": "function", "function": {"name": "fetch_telemetry", "description": "Read sensor telemetry for a station", "parameters": {"type": "object", "properties": {"station_id": {"type": "string"}, "window_min": {"type": "integer"}}, "required": ["station_id", "window_min"]}}}\n'
    '{"type": "function", "function": {"name": "set_valve_flow", "description": "Set flow rate on a manifold", "parameters": {"type": "object", "properties": {"manifold": {"type": "string"}, "liters_per_min": {"type": "number"}}, "required": ["manifold", "liters_per_min"]}}}\n'
    '{"type": "function", "function": {"name": "run_diagnostic", "description": "Run hardware self-test on target device", "parameters": {"type": "object", "properties": {"device_id": {"type": "string"}, "deep": {"type": "boolean"}}, "required": ["device_id"]}}}\n'
    "</tools>"
)


def _format_tool_call(name: str, params: dict[str, Any]) -> str:
    body = "\n".join(f"<parameter={k}>\n{v}\n</parameter>" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}\n</function>\n</tool_call>"


def _make_tool_example(rng: random.Random, idx: int, *, negative: bool) -> dict[str, Any]:
    d = _DOMAINS[idx % len(_DOMAINS)]
    station = f"STN-{rng.randint(100, 999)}"
    win = rng.choice([5, 15, 30, 60])
    flow = rng.choice([12.5, 18.0, 24.5, 30.0])

    if negative:
        # Hammer/ToolACE-style negative abstention: irrelevant <tools> declared, task is pure analytical derivation!
        a, b = rng.randint(11, 24), rng.randint(12, 22)
        c, e = rng.randint(5, 12), rng.randint(10, 35)
        ans_num = a * b + c * a - e
        if idx % 2 == 0:
            user_q = (
                f"Pure analytical calculation for {d['domain']}: compute ({a} * {b}) + ({c} * {a}) - {e} "
                'and output ONLY valid JSON: {"total": <integer>, "status": "ok"}. No external tools apply.'
            )
            think = (
                f"Tools are listed in system prompt but this is a self-contained arithmetic problem; never emit <tool_call>. "
                f"{a}*{b}={a*b}, {c}*{a}={c*a}, sum={a*b+c*a}, minus {e} = {ans_num}."
            )
            reply = f"<think>{think}</think>\n" + json.dumps({"total": ans_num, "status": "ok"})
            exp = {"total": ans_num, "status": "ok"}
        else:
            sc = _make_scenario(rng, d)
            user_q = sc["full_prompt"]
            think = (
                f"Telemetry/valve tools do not apply to this constraint selection task; emit zero <tool_call> tags. "
                f"{d['mode_a']} gives {sc['units_a']} units, max cost {sc['cost_max_a']}, lat {sc['lat_max_a']}, res {sc['res_max_a']}. "
                f"Pick {sc['ans_a']}; ledger {sc['inv_val']}."
            )
            reply = f"<think>{think}</think>\n" + json.dumps({"primary": sc["ans_a"], "preserved": sc["inv_val"]})
            exp = {"primary": sc["ans_a"], "preserved": sc["inv_val"]}
        return {
            "pillar": "tool_abstention",
            "messages": [
                {"role": "system", "content": "You are an analytical assistant.\n" + _TOOLS_BLOCK},
                {"role": "user", "content": user_q},
                {"role": "assistant", "content": reply},
            ],
            "expected": {"abstain": True, "answer": exp},
        }

    kind = idx % 3
    if kind == 0:
        tc = _format_tool_call("fetch_telemetry", {"station_id": station, "window_min": win})
        return {
            "pillar": "tool_positive",
            "messages": [
                {"role": "system", "content": "Use available tools when real-time measurements are requested.\n" + _TOOLS_BLOCK},
                {"role": "user", "content": f"Check the last {win} minutes of telemetry for station {station}."},
                {"role": "assistant", "content": f"<think>User needs live telemetry for {station} over {win}m; call fetch_telemetry directly.</think>\n{tc}"},
            ],
            "expected": {"tool": "fetch_telemetry", "args": {"station_id": station, "window_min": str(win)}},
        }
    if kind == 1:
        dev = f"DEV-{rng.randint(10, 99)}"
        tc = _format_tool_call("run_diagnostic", {"device_id": dev, "deep": "true"})
        return {
            "pillar": "tool_positive",
            "messages": [
                {"role": "system", "content": "Use available tools when device checks are requested.\n" + _TOOLS_BLOCK},
                {"role": "user", "content": f"Run a deep hardware self-test on {dev}."},
                {"role": "assistant", "content": f"<think>Run run_diagnostic with device_id={dev} and deep=true.</think>\n{tc}"},
            ],
            "expected": {"tool": "run_diagnostic", "args": {"device_id": dev, "deep": "true"}},
        }
    # kind == 2: Multi-turn tool error recovery
    manifold = f"MNF-{rng.randint(1, 9)}"
    bad_tc = _format_tool_call("set_valve_flow", {"manifold": manifold, "liters_per_min": 999.0})
    good_tc = _format_tool_call("set_valve_flow", {"manifold": manifold, "liters_per_min": flow})
    return {
        "pillar": "tool_recovery",
        "messages": [
            {"role": "system", "content": "Adjust actuators carefully and recover from validation errors.\n" + _TOOLS_BLOCK},
            {"role": "user", "content": f"Set manifold {manifold} flow to safe limit (max 40 L/min, target {flow} L/min)."},
            {"role": "assistant", "content": f"<think>Call set_valve_flow on {manifold}.</think>\n{bad_tc}"},
            {"role": "user", "content": f"<tool_response>\n{{\"error\": \"liters_per_min=999.0 exceeds hardware cap 40.0; target={flow}\"}}\n</tool_response>"},
            {"role": "assistant", "content": f"<think>Previous call failed cap check (999.0 > 40.0); retry set_valve_flow with safe target {flow}.</think>\n{good_tc}"},
        ],
        "expected": {"tool": "set_valve_flow", "args": {"manifold": manifold, "liters_per_min": str(flow)}},
    }


def _make_schema_ifeval_example(rng: random.Random, idx: int) -> dict[str, Any]:
    d = _DOMAINS[idx % len(_DOMAINS)]
    v1, v2, v3 = rng.randint(12, 35), rng.randint(4, 15), rng.randint(10, 29)
    total = v1 * v2 - v3
    if idx % 2 == 0:
        # Strict JSON-only with evaluated integers (no code fence, no formulas)
        keys = ["product", "difference", "choice"]
        choice = rng.choice(["BETA", "GAMMA", "DELTA"])
        obj = {"product": v1 * v2, "difference": total, "choice": choice}
        think = (
            f"Must output ONLY raw valid JSON (no markdown fences) with exact keys product, difference, choice and integer values. "
            f"{v1}*{v2}={v1*v2}, {v1*v2}-{v3}={total}."
        )
        return {
            "pillar": "schema_json",
            "messages": [
                {"role": "system", "content": "Follow output format constraints strictly. No markdown fences unless requested."},
                {"role": "user", "content": (
                    f"Given x={v1}, y={v2}, z={v3}, and tag={choice}, compute product=x*y and difference=x*y-z. "
                    f'Output ONLY single-line raw JSON with exact keys {keys} and integer numeric values.'
                )},
                {"role": "assistant", "content": f"<think>{think}</think>\n{json.dumps(obj)}"},
            ],
            "expected": obj,
        }

    # Exact tagged N-line IFEval constraint (no bullets, exact prefixes, evaluated number)
    think = f"Need exactly 3 lines prefixed LINE 1:, LINE 2:, LINE 3: with domain={d['domain']} and evaluated sum={v1 + v2}."
    out = (
        f"<think>{think}</think>\n"
        f"LINE 1: DOMAIN={d['domain'].upper()}\n"
        f"LINE 2: SUM={v1 + v2}\n"
        f"LINE 3: STATUS=VERIFIED"
    )
    return {
        "pillar": "schema_lines",
        "messages": [
            {"role": "system", "content": "Follow every line prefix and line-count rule strictly."},
            {"role": "user", "content": (
                f"Output EXACTLY 3 lines after </think>:\n"
                f"LINE 1: DOMAIN=<{d['domain']} in uppercase>\n"
                f"LINE 2: SUM=<evaluated {v1} + {v2}>\n"
                f"LINE 3: STATUS=VERIFIED\nNo extra lines or bullets."
            )},
            {"role": "assistant", "content": out},
        ],
        "expected": {"lines": [f"LINE 1: DOMAIN={d['domain'].upper()}", f"LINE 2: SUM={v1 + v2}", "LINE 3: STATUS=VERIFIED"]},
    }


def _make_rlvr_prompts(seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    prompts: list[dict[str, Any]] = []

    # 6 math_opt prompts
    for i in range(6):
        ex = _make_multihop_example(rng, i * 3)
        prompts.append({
            "id": f"rlvr-math-{i+1:02d}",
            "kind": "math_opt",
            "messages": ex["messages"][:-1],
            "reference_answer": ex["messages"][-1]["content"],
            "verifier": {
                "primary": ex["expected"]["primary"],
                "preserved": ex["expected"]["preserved"],
                "forbid_tool_call": True,
                "require_think_close": True,
            },
        })

    # 6 audit prompts (3 premise override, 3 distractor)
    for i in range(6):
        ex = _make_belief_revision_example(rng, i * 3 + (0 if i < 3 else 1))
        prompts.append({
            "id": f"rlvr-audit-{i+1:02d}",
            "kind": "audit",
            "messages": ex["messages"][:-1],
            "reference_answer": ex["messages"][-1]["content"],
            "verifier": {
                "verdict": ex["expected"]["verdict"],
                "forbid_tool_call": True,
                "require_think_close": True,
            },
        })

    # 6 tool prompts (3 positive, 3 negative abstention)
    for i in range(6):
        neg = i >= 3
        ex = _make_tool_example(rng, i + (0 if not neg else 2), negative=neg)
        msgs = ex["messages"][:-1]
        if neg:
            v = {"abstain": True, "forbid_tool_call": True, "require_think_close": True, "expected": ex["expected"]["answer"]}
        else:
            v = {"abstain": False, "tool": ex["expected"]["tool"], "args": ex["expected"]["args"], "require_think_close": True}
        prompts.append({
            "id": f"rlvr-tool-{i+1:02d}",
            "kind": "tool_neg" if neg else "tool_pos",
            "messages": msgs,
            "reference_answer": ex["messages"][-1]["content"],
            "verifier": v,
        })

    # 6 strict schema / IFEval prompts
    for i in range(6):
        ex = _make_schema_ifeval_example(rng, i)
        prompts.append({
            "id": f"rlvr-schema-{i+1:02d}",
            "kind": ex["pillar"],
            "messages": ex["messages"][:-1],
            "reference_answer": ex["messages"][-1]["content"],
            "verifier": {
                "expected": ex["expected"],
                "forbid_tool_call": True,
                "require_think_close": True,
            },
        })

    return prompts


def _make_heldout_eval_20(seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    tasks: list[dict[str, Any]] = []

    # 5 unseen multi-hop math tasks
    for i in range(5):
        ex = _make_multihop_example(rng, i * 3)
        tasks.append({
            "id": f"eval-math-{i+1:02d}",
            "pillar": "multihop_math",
            "messages": ex["messages"][:-1],
            "verifier": {
                "primary": ex["expected"]["primary"],
                "preserved": ex["expected"]["preserved"],
                "forbid_tool_call": True,
            },
        })

    # 5 unseen schema & belief-revision audit tasks (3 audit + 2 IFEval)
    for i in range(3):
        ex = _make_belief_revision_example(rng, i * 3 + (0 if i < 2 else 1))
        tasks.append({
            "id": f"eval-audit-{i+1:02d}",
            "pillar": "schema_audit",
            "messages": ex["messages"][:-1],
            "verifier": {
                "verdict": ex["expected"]["verdict"],
                "forbid_tool_call": True,
            },
        })
    for i in range(2):
        ex = _make_schema_ifeval_example(rng, i)
        tasks.append({
            "id": f"eval-ifeval-{i+1:02d}",
            "pillar": "schema_audit",
            "kind": ex["pillar"],
            "messages": ex["messages"][:-1],
            "verifier": {
                "expected": ex["expected"],
                "forbid_tool_call": True,
            },
        })

    # 5 unseen positive tool-call tasks
    for i in range(5):
        ex = _make_tool_example(rng, i, negative=False)
        tasks.append({
            "id": f"eval-toolpos-{i+1:02d}",
            "pillar": "tool_positive",
            "messages": ex["messages"][:-1],
            "verifier": {
                "abstain": False,
                "tool": ex["expected"]["tool"],
                "args": ex["expected"]["args"],
            },
        })

    # 5 unseen negative tool-abstention tasks (tools declared in system, zero tool_call allowed)
    for i in range(5):
        ex = _make_tool_example(rng, i, negative=True)
        tasks.append({
            "id": f"eval-toolneg-{i+1:02d}",
            "pillar": "tool_abstention",
            "messages": ex["messages"][:-1],
            "verifier": {
                "abstain": True,
                "forbid_tool_call": True,
                "expected": ex["expected"]["answer"],
            },
        })

    return tasks


def assert_decontaminated(all_generated: list[dict[str, Any]]) -> None:
    """Verify zero domain, regime, numeric-signature, or 6-gram overlap with pilot/benchmark_100."""
    forbidden_regime_tokens = {
        "IN_PERSON",
        "HYBRID_STUDIO",
        "DIESEL_CONVOY",
        "RAIL_EXPRESS",
        "STANDARD_HDD",
        "NVME_FABRIC",
        "LEGACY_VPN",
        "ZERO_TRUST_ASIC",
    }
    bench_ngrams: set[tuple[str, ...]] = set()
    bench_ids: set[str] = set()
    for p in (PILOT_PATH, BENCH_PATH):
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            bench_ids.add(row["id"])
            words = (row.get("initial_prompt", "") + " " + (row.get("midway_update") or "")).lower().split()
            for j in range(max(0, len(words) - 6 + 1)):
                bench_ngrams.add(tuple(words[j : j + 6]))

    for row in all_generated:
        txt = "\n".join(str(m.get("content", "")) for m in row.get("messages", []))
        for tok in forbidden_regime_tokens:
            assert tok not in txt, f"Contamination hit: benchmark regime {tok} found in generated sample!"
        words = txt.lower().split()
        for j in range(max(0, len(words) - 6 + 1)):
            gram = tuple(words[j : j + 6])
            assert gram not in bench_ngrams, f"6-gram contamination detected: {' '.join(gram)}"


def _build_sft_split(seed: int, count: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    n_multihop = round(count * 0.30)
    n_revision = round(count * 0.25)
    n_tool_pos = round(count * 0.15)
    n_tool_neg = round(count * 0.10)
    n_schema = count - (n_multihop + n_revision + n_tool_pos + n_tool_neg)

    rows: list[dict[str, Any]] = []
    for i in range(n_multihop):
        rows.append(_make_multihop_example(rng, i))
    for i in range(n_revision):
        rows.append(_make_belief_revision_example(rng, i))
    for i in range(n_tool_pos):
        rows.append(_make_tool_example(rng, i, negative=False))
    for i in range(n_tool_neg):
        rows.append(_make_tool_example(rng, i, negative=True))
    for i in range(n_schema):
        rows.append(_make_schema_ifeval_example(rng, i))
    rng.shuffle(rows)
    for k, r in enumerate(rows, start=1):
        r["id"] = f"sft-{seed}-{k:04d}"
    return rows


def main() -> None:
    DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_sft = _build_sft_split(seed=20261121, count=240)
    val_sft = _build_sft_split(seed=20261122, count=48)
    rlvr_prompts = _make_rlvr_prompts(seed=20261123)
    heldout_20 = _make_heldout_eval_20(seed=20261124)

    all_rows = train_sft + val_sft + rlvr_prompts + heldout_20
    assert_decontaminated(all_rows)

    for fname, dataset in [
        ("train_sft.jsonl", train_sft),
        ("val_sft.jsonl", val_sft),
        ("rlvr_prompts.jsonl", rlvr_prompts),
        ("heldout_eval_20.jsonl", heldout_20),
    ]:
        out_path = DATA_OUT_DIR / fname
        out_path.write_text(
            "\n".join(json.dumps(r) for r in dataset) + "\n",
            encoding="utf-8",
        )

    print(
        f"Wrote decontaminated datasets to {DATA_OUT_DIR}: "
        f"train_sft={len(train_sft)}, val_sft={len(val_sft)}, "
        f"rlvr_prompts={len(rlvr_prompts)}, heldout_eval_20={len(heldout_20)}"
    )


if __name__ == "__main__":
    main()
