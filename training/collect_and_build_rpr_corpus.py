"""Assemble the 100%-frontier-authored RPR v4 + Diagnostic + AI/Weights/Self-Improvement corpus.

Combines:
  1. Bespoke non-templated frontier conversations (`frontier_ai_self_improve.jsonl`,
     `frontier_systems_debug.jsonl`, `frontier_math_logic_rpr.jsonl`)
  2. Verified parameterized first-principles pillars (`high_signal_pillars.py`: diagnostic bug-hunting,
     cognitive traps, AI/transformer/hybrid-conv weight mechanics, and post-training self-improvement)
  3. Grounded local-witness Single-Stream RPR (1-turn & 2-turn), RPR Controller stages, AGM Logic,
     Software Architecture & Python RPR, Calibrated XML Tools/Abstention, IFEval, and PIL Vision RPR.

Strictly enforces 100% Python unit-test execution and zero benchmark contamination against
`data/pilot_problems.jsonl` and `data/benchmark_100.jsonl`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

from PIL import Image, ImageDraw

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from engine import _ANSWER_FORMAT_NOTE
from training.build_training_data import _TOOLS_BLOCK, _format_tool_call, assert_decontaminated
from training.high_signal_pillars import (
    make_ai_weights_mechanics_item,
    make_diagnostic_debug_item,
    make_self_improvement_meta_item,
)
from verifiers import check_candidate_constraints


CORPUS_DIR = Path(__file__).resolve().parent / "data" / "rpr_v4_corpus"
IMG_DIR = CORPUS_DIR / "images"

FRONTIER_FILES = [
    "frontier_ai_self_improve.jsonl",
    "frontier_systems_debug.jsonl",
    "frontier_math_logic_rpr.jsonl",
]

_EXTENDED_DOMAINS: list[dict[str, Any]] = [
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
    {
        "domain": "subsea_cables",
        "title": "Provision transoceanic optical repeater spans.",
        "demand_name": "lit wavelength capacity",
        "demand_unit": "Tbps",
        "unit_singular": "repeater",
        "unit_plural": "repeaters",
        "cost_key": "tariff_usd",
        "lat_key": "prop_ms",
        "lat_label": "effective span delay (ms)",
        "res_key": "pump_ma",
        "res_label": "per-repeater pump current (mA)",
        "bool_key": "raman_amp",
        "mode_a": "COAX_COPPER",
        "mode_b": "DWDM_FIBER",
        "inv_a": "splice trays",
        "inv_a_item": "ferrule sleeves",
        "inv_b": "branching units",
        "inv_b_item": "shunt diodes",
        "inv_c": "retired couplings",
        "distractor_loc": "coastal cable depot",
    },
    {
        "domain": "bioreactor_ferm",
        "title": "Scale sterile upstream biologics vessels.",
        "demand_name": "feedstock harvest rate",
        "demand_unit": "L/hr",
        "unit_singular": "vessel",
        "unit_plural": "vessels",
        "cost_key": "media_usd",
        "lat_key": "equil_min",
        "lat_label": "effective equilibration time (min)",
        "res_key": "sparge_slpm",
        "res_label": "per-vessel gas flow (SLPM)",
        "bool_key": "inline_raman",
        "mode_a": "FED_BATCH",
        "mode_b": "PERFUSION_MEM",
        "inv_a": "seed flasks",
        "inv_a_item": "vial aliquots",
        "inv_b": "buffer carboys",
        "inv_b_item": "filter capsules",
        "inv_c": "quarantined retains",
        "distractor_loc": "benchtop QC hood",
    },
    {
        "domain": "quantum_cryo",
        "title": "Assign dilution refrigerator control lines.",
        "demand_name": "coherent gate throughput",
        "demand_unit": "kShots/s",
        "unit_singular": "cryostat",
        "unit_plural": "cryostats",
        "cost_key": "cryo_usd",
        "lat_key": "reset_ms",
        "lat_label": "effective readout cycle (ms)",
        "res_key": "heat_mw",
        "res_label": "per-cryostat thermal load (mW)",
        "bool_key": "twpa_chain",
        "mode_a": "CMOS_WARM",
        "mode_b": "SFQ_CRYO",
        "inv_a": "attenuator blocks",
        "inv_a_item": "coax jumpers",
        "inv_b": "resonator banks",
        "inv_b_item": "flux lines",
        "inv_c": "detuned couplers",
        "distractor_loc": "calibration annex C",
    },
]

_PALETTE = [
    ("crimson", (220, 48, 64)),
    ("emerald", (32, 188, 108)),
    ("cobalt", (48, 116, 236)),
    ("amber", (242, 178, 36)),
    ("violet", (156, 76, 228)),
    ("cyan", (36, 198, 218)),
]


def _decomp_mul(a: int, b: int) -> str:
    tens = (b // 10) * 10
    ones = b % 10
    if tens > 0 and ones > 0:
        return f"{a}*{b}={a*tens}+{a*ones}={a*b}"
    return f"{a}*{b}={a*b}"


def _decomp_ledger(m1_a: int, m1_b: int, m2_a: int, m2_b: int, sub_c: int) -> str:
    p1 = m1_a * m1_b
    p2 = m2_a * m2_b
    s12 = p1 + p2
    fin = s12 - sub_c
    return f"{_decomp_mul(m1_a, m1_b)}; {_decomp_mul(m2_a, m2_b)}; {p1}+{p2}={s12}; {s12}-{sub_c}={fin}"


def _make_rich_scenario(rng: random.Random, d: dict[str, Any]) -> dict[str, Any]:
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

    cost_max_a = rng.choice([120, 150, 180, 210, 240])
    total_budget = units_a * cost_max_a
    cost_max_b = total_budget // units_b

    cap_total = rng.choice([55, 65, 80, 95, 110])
    pen_a = rng.choice([18, 22, 26])
    pen_b = rng.choice([4, 6, 8])
    lat_max_a = cap_total - pen_a
    lat_max_b = cap_total - pen_b

    res_gross = rng.choice([62, 74, 86, 98, 110])
    over_a = rng.choice([20, 24, 28])
    over_b = rng.choice([6, 8, 12])
    res_max_a = res_gross - over_a
    res_max_b = res_gross - over_b

    m1_a, m1_b = rng.randint(8, 19), rng.randint(11, 23)
    m2_a, m2_b = rng.randint(5, 13), rng.randint(9, 17)
    sub_c = rng.randint(12, 39)
    inv_val = m1_a * m1_b + m2_a * m2_b - sub_c
    inv_expr = f"{m1_a} * {m1_b} + {m2_a} * {m2_b} - {sub_c}"
    inv_steps = _decomp_ledger(m1_a, m1_b, m2_a, m2_b, sub_c)
    inv_text = (
        f"{m1_a} {d['inv_a']} of {m1_b} {d['inv_a_item']} plus "
        f"{m2_a} {d['inv_b']} of {m2_b} {d['inv_b_item']}, minus {sub_c} {d['inv_c']}."
    )

    opts_raw = [
        {
            "role": "A_VALID",
            "mode": d["mode_a"],
            d["bool_key"]: rng.choice([False, True]),
            d["cost_key"]: cost_max_a - rng.randint(8, 22),
            d["lat_key"]: lat_max_a - rng.randint(3, 7),
            d["res_key"]: res_max_a - rng.randint(2, 6),
        },
        {
            "role": "A_RES_FAIL",
            "mode": d["mode_a"],
            d["bool_key"]: True,
            d["cost_key"]: cost_max_a - 10,
            d["lat_key"]: lat_max_a - 4,
            d["res_key"]: res_max_a + rng.randint(5, 11),
        },
        {
            "role": "B_VALID",
            "mode": d["mode_b"],
            d["bool_key"]: True,
            d["cost_key"]: int(round((cost_max_a + cost_max_b) * 0.54)),
            d["lat_key"]: lat_max_b - rng.randint(4, 9),
            d["res_key"]: res_max_b - rng.randint(3, 8),
        },
        {
            "role": "B_BOOL_FAIL",
            "mode": d["mode_b"],
            d["bool_key"]: False,
            d["cost_key"]: cost_max_a - 14,
            d["lat_key"]: lat_max_a - 5,
            d["res_key"]: res_max_a - 5,
        },
        {
            "role": "B_LAT_FAIL",
            "mode": d["mode_b"],
            d["bool_key"]: True,
            d["cost_key"]: cost_max_b - 24,
            d["lat_key"]: lat_max_b + rng.randint(5, 12),
            d["res_key"]: res_max_b - 4,
        },
        {
            "role": "B_COST_FAIL",
            "mode": d["mode_b"],
            d["bool_key"]: True,
            d["cost_key"]: cost_max_b + rng.randint(15, 35),
            d["lat_key"]: lat_max_b - 5,
            d["res_key"]: res_max_b - 5,
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
        + "\nTasks:\n1. Identify the unique qualifying row OPT-A..OPT-F as \"primary\" under the current profile.\n"
        + "2. Evaluate the integer ledger count as \"preserved\".\n"
        + "Candidate options:\n"
        + "\n".join(opt_lines)
        + '\nReply with JSON: {"primary": "OPT-X", "preserved": <integer>}'
    )
    premise_update = (
        f"Control-plane override: {d['mode_b']} hardware has been commissioned for live traffic, "
        f"shifting the operational profile from {d['mode_a']} over to {d['mode_b']}. "
        f"Target load ({demand} {d['demand_unit']}), budget ceiling ({total_budget}), and ledger inventory stay identical."
    )
    irrelevant_update = (
        f"Engineering memo: A sandboxed {d['distractor_loc']} bench-tested {d['mode_b']} across {units_b} units "
        f"at {over_b} trial overhead and purged {sub_c} temp files; live traffic stays on {d['mode_a']} with original rules."
    )

    def _spec_for(use_b: bool) -> dict[str, Any]:
        mode = d["mode_b"] if use_b else d["mode_a"]
        c_max = cost_max_b if use_b else cost_max_a
        l_max = lat_max_b if use_b else lat_max_a
        r_max = res_max_b if use_b else res_max_a
        rules_list: list[dict[str, Any]] = [
            {"field": "mode", "op": "eq", "value": mode},
            {"field": d["cost_key"], "op": "lte", "value": c_max},
            {"field": d["lat_key"], "op": "lte", "value": l_max},
            {"field": d["res_key"], "op": "lte", "value": r_max},
        ]
        if use_b:
            rules_list.append({"field": d["bool_key"], "op": "eq", "value": True})
        return {"options": options, "rules": rules_list}

    valid_a = [oid for oid in options if not check_candidate_constraints({"primary": oid, "preserved": inv_val}, _spec_for(False))]
    valid_b = [oid for oid in options if not check_candidate_constraints({"primary": oid, "preserved": inv_val}, _spec_for(True))]
    assert valid_a == [ans_a] and valid_b == [ans_b]

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
        "inv_steps": inv_steps,
        "inv_val": inv_val,
        "ans_a": ans_a,
        "ans_b": ans_b,
        "options": options,
        "opt_lines": "\n".join(opt_lines),
        "rules_block": rules_block,
        "full_prompt": full_prompt,
        "premise_update": premise_update,
        "irrelevant_update": irrelevant_update,
    }


def _witness_l3_lines(
    options: dict[str, dict[str, Any]],
    d: dict[str, Any],
    *,
    active_mode: str,
    req_bool: bool,
    max_cost: int,
    max_lat: int,
    max_res: int,
) -> str:
    lines: list[str] = []
    bk, ck, lk, rk = d["bool_key"], d["cost_key"], d["lat_key"], d["res_key"]
    for lbl, a in options.items():
        if a["mode"] != active_mode:
            lines.append(f"{lbl}: mode={a['mode']}!={active_mode}->FAIL")
            continue
        b_str = str(a[bk]).lower()
        if req_bool and not a[bk]:
            lines.append(f"{lbl}: mode={a['mode']}, {bk}={b_str}!=true->FAIL")
            continue
        if a[ck] > max_cost:
            lines.append(f"{lbl}: mode={a['mode']}, {bk}={b_str}, {ck}={a[ck]}>{max_cost}->FAIL")
            continue
        if a[lk] > max_lat:
            lines.append(f"{lbl}: {ck}={a[ck]}<={max_cost}, {lk}={a[lk]}>{max_lat}->FAIL")
            continue
        if a[rk] > max_res:
            lines.append(f"{lbl}: {ck}={a[ck]}<={max_cost}, {lk}={a[lk]}<={max_lat}, {rk}={a[rk]}>{max_res}->FAIL")
            continue
        lines.append(
            f"{lbl}: mode={a['mode']}, {bk}={b_str}, {ck}={a[ck]}<={max_cost}, {lk}={a[lk]}<={max_lat}, {rk}={a[rk]}<={max_res}->PASS"
        )
    return "; ".join(lines)


def _single_stream_turn1_think(sc: dict[str, Any]) -> str:
    d = sc["d"]
    w_a = _witness_l3_lines(
        sc["options"], d, active_mode=d["mode_a"], req_bool=False,
        max_cost=sc["cost_max_a"], max_lat=sc["lat_max_a"], max_res=sc["res_max_a"],
    )
    return (
        f"[L1] A1: mode={d['mode_a']} ({d['bool_key']} optional); A2: load={sc['demand']}/{sc['per_a']}; "
        f"A3: cost_cap<={sc['total_budget']}; A4: {d['lat_key']}+{sc['pen_a']}<={sc['cap_total']}; "
        f"A5: {d['res_key']}<={sc['res_gross']}-{sc['over_a']}; F1: {sc['inv_steps']}.\n"
        f"[L2] units={sc['demand']}/{sc['per_a']}={sc['units_a']}; max_{d['cost_key']}={sc['total_budget']}/{sc['units_a']}={sc['cost_max_a']}; "
        f"max_{d['lat_key']}={sc['cap_total']}-{sc['pen_a']}={sc['lat_max_a']}; max_{d['res_key']}={sc['res_gross']}-{sc['over_a']}={sc['res_max_a']}.\n"
        f"[L3] {w_a}. Select {sc['ans_a']}, preserved={sc['inv_val']}."
    )


def _single_stream_turn2_premise_think(sc: dict[str, Any]) -> str:
    d = sc["d"]
    w_b = _witness_l3_lines(
        sc["options"], d, active_mode=d["mode_b"], req_bool=True,
        max_cost=sc["cost_max_b"], max_lat=sc["lat_max_b"], max_res=sc["res_max_b"],
    )
    return (
        f"[AUDIT] Active profile switched {d['mode_a']}->{d['mode_b']}: A1,A2,A4,A5 INVALID; A3({sc['total_budget']}) VALID; F1={sc['inv_val']} PRESERVED.\n"
        f"[REBUILD L1] mode={d['mode_b']}, {d['bool_key']}=true required, cap={sc['per_b']}, pen={sc['pen_b']}, over={sc['over_b']}.\n"
        f"[REBUILD L2] units={sc['demand']}/{sc['per_b']}={sc['units_b']}; max_{d['cost_key']}={sc['total_budget']}/{sc['units_b']}={sc['cost_max_b']}; "
        f"max_{d['lat_key']}={sc['cap_total']}-{sc['pen_b']}={sc['lat_max_b']}; max_{d['res_key']}={sc['res_gross']}-{sc['over_b']}={sc['res_max_b']}.\n"
        f"[REBUILD L3] {w_b}. Select {sc['ans_b']}, preserved={sc['inv_val']}."
    )


def _single_stream_turn2_irrel_think(sc: dict[str, Any]) -> str:
    d = sc["d"]
    return (
        f"[AUDIT] Memo affects isolated {d['distractor_loc']} only; live traffic stays on {d['mode_a']}. "
        f"A1..A5 remain VALID; F1={sc['inv_val']} PRESERVED.\n"
        f"[KEEP] Retain primary={sc['ans_a']} and preserved={sc['inv_val']}."
    )


def _make_single_stream_rpr_item(rng: random.Random, idx: int) -> dict[str, Any]:
    d = _EXTENDED_DOMAINS[idx % len(_EXTENDED_DOMAINS)]
    sc = _make_rich_scenario(rng, d)
    sys_msg = "You are a careful analytical problem solver. Follow all constraints strictly.\n" + _ANSWER_FORMAT_NOTE
    t1 = "<think>" + _single_stream_turn1_think(sc) + "</think>\n" + json.dumps({"primary": sc["ans_a"], "preserved": sc["inv_val"]})

    mode_sel = idx % 5
    if mode_sel in {0, 1, 2}:
        u2 = f"New evidence arrived:\n{sc['premise_update']}\n\nProvide your updated answer.\n{_ANSWER_FORMAT_NOTE}"
        t2 = "<think>" + _single_stream_turn2_premise_think(sc) + "</think>\n" + json.dumps({"primary": sc["ans_b"], "preserved": sc["inv_val"]})
        return {
            "pillar": "single_stream_rpr_2turn",
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": sc["full_prompt"]},
                {"role": "assistant", "content": t1},
                {"role": "user", "content": u2},
                {"role": "assistant", "content": t2},
            ],
            "expected": {"primary": sc["ans_b"], "preserved": sc["inv_val"]},
        }
    if mode_sel == 3:
        u2 = f"New evidence arrived:\n{sc['irrelevant_update']}\n\nProvide your updated answer.\n{_ANSWER_FORMAT_NOTE}"
        t2 = "<think>" + _single_stream_turn2_irrel_think(sc) + "</think>\n" + json.dumps({"primary": sc["ans_a"], "preserved": sc["inv_val"]})
        return {
            "pillar": "single_stream_rpr_2turn",
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": sc["full_prompt"]},
                {"role": "assistant", "content": t1},
                {"role": "user", "content": u2},
                {"role": "assistant", "content": t2},
            ],
            "expected": {"primary": sc["ans_a"], "preserved": sc["inv_val"]},
        }
    return {
        "pillar": "single_stream_rpr_1turn",
        "messages": [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": sc["full_prompt"]},
            {"role": "assistant", "content": t1},
        ],
        "expected": {"primary": sc["ans_a"], "preserved": sc["inv_val"]},
    }


def _make_controller_stage_item(rng: random.Random, idx: int) -> dict[str, Any]:
    d = _EXTENDED_DOMAINS[idx % len(_EXTENDED_DOMAINS)]
    sc = _make_rich_scenario(rng, d)
    stage = idx % 4
    use_b = (idx // 4) % 2 == 1
    mode = d["mode_b"] if use_b else d["mode_a"]
    cap = sc["per_b"] if use_b else sc["per_a"]
    over = sc["over_b"] if use_b else sc["over_a"]
    pen = sc["pen_b"] if use_b else sc["pen_a"]
    units = sc["units_b"] if use_b else sc["units_a"]
    cost_m = sc["cost_max_b"] if use_b else sc["cost_max_a"]
    lat_m = sc["lat_max_b"] if use_b else sc["lat_max_a"]
    res_m = sc["res_max_b"] if use_b else sc["res_max_a"]
    ans = sc["ans_b"] if use_b else sc["ans_a"]
    bool_desc = "true required" if use_b else "optional"

    if stage == 0:
        ev_txt = sc["premise_update"] if use_b else "Initial problem statement."
        think = f"Active profile={mode} ({d['bool_key']} {bool_desc}), cap={cap}, pen={pen}, over={over}. Ledger: {sc['inv_steps']}."
        out = (
            f"<think>{think}</think>\n"
            f"A1: active_mode={mode}; boolean_feature={d['bool_key']} {bool_desc}\n"
            f"A2: workload demand is {sc['demand']}, and under active mode each {d['unit_singular']} handles {cap}\n"
            f"A3: total budget cap formula is (needed {d['unit_plural']}) * {d['cost_key']} <= {sc['total_budget']}\n"
            f"A4: effective latency/time cap formula is {d['lat_key']} + {pen} <= {sc['cap_total']}\n"
            f"A5: per-unit resource/capacity limit formula is {sc['res_gross']} - {over} = {res_m}\n"
            f"F1 [NONE]: {sc['inv_expr']} = {sc['inv_val']} | invariant ledger"
        )
        return {
            "pillar": "rpr_controller_l1",
            "messages": [
                {"role": "system", "content": "You are Level 1 (Foundation) of a Revisable Reasoning Pyramid. Do NOT emit <tool_call> tags. Output A1..A5 and F1 lines only."},
                {"role": "user", "content": f"Problem Rules:\n{sc['rules_block']}\n\nEvidence:\n- {ev_txt}"},
                {"role": "assistant", "content": out},
            ],
            "expected": {"mode": mode, "preserved": sc["inv_val"]},
        }

    if stage == 1:
        think = (
            f"Active mode={mode} ({d['bool_key']} {bool_desc}). "
            f"L2-1: {sc['demand']}/{cap}={units}. "
            f"L2-2: divide budget {sc['total_budget']} by units {units} = {cost_m}. "
            f"L2-4: gross latency ceiling {sc['cap_total']} minus active penalty {pen} = {lat_m} (never reuse stale bound). "
            f"L2-5: gross resource limit {sc['res_gross']} minus active overhead {over} = {res_m}."
        )
        out = (
            f"<think>{think}</think>\n"
            f"L2-1 [A1, A2]: required_units = {sc['demand']} / {cap} = {units}\n"
            f"L2-2 [A1, A2, A3]: max_per_unit_cost = {sc['total_budget']} / {units} = {cost_m}\n"
            f"L2-3 [A1]: required_mode = {mode}; mandatory_feature = {d['bool_key']}={bool_desc}\n"
            f"L2-4 [A1, A4]: max_base_time_or_lat = {sc['cap_total']} - {pen} = {lat_m}\n"
            f"L2-5 [A1, A5]: max_per_unit_resource = {sc['res_gross']} - {over} = {res_m}"
        )
        return {
            "pillar": "rpr_controller_l2",
            "messages": [
                {"role": "system", "content": "You are Level 2 (Decomposition & Subproblem Bounds) of a Revisable Reasoning Pyramid. Output ONLY L2-1..L2-5."},
                {"role": "user", "content": (
                    f"Problem Rules:\n{sc['rules_block']}\n\nLatest evidence:\n{sc['premise_update'] if use_b else 'Initial problem statement.'}\n\n"
                    f"Level 1 Active Assumptions:\n- A1: active_mode={mode}; boolean_feature={d['bool_key']} {bool_desc}\n"
                    f"- A2: demand={sc['demand']}, per_unit={cap}\n- A3: required * {d['cost_key']} <= {sc['total_budget']}\n"
                    f"- A4: {d['lat_key']} + {pen} <= {sc['cap_total']}\n- A5: {d['res_key']} <= {sc['res_gross']} - {over}"
                )},
                {"role": "assistant", "content": out},
            ],
            "expected": {"units": units, "lat": lat_m, "res": res_m},
        }

    if stage == 2:
        w = _witness_l3_lines(sc["options"], d, active_mode=mode, req_bool=use_b, max_cost=cost_m, max_lat=lat_m, max_res=res_m)
        out = (
            f"<think>Verify all 6 options against mode={mode}, {d['bool_key']}={bool_desc}, cost<={cost_m}, lat<={lat_m}, res<={res_m}: {w}. F1={sc['inv_val']}.</think>\n"
            f"L3-1: mode={mode}, {d['bool_key']}={bool_desc}, units={units}, max_cost={cost_m}, max_lat={lat_m}, max_res={res_m}\n"
            f"L3-2: {w}\n"
            + json.dumps({"primary": ans, "preserved": sc["inv_val"]})
        )
        return {
            "pillar": "rpr_controller_l3",
            "messages": [
                {"role": "system", "content": "You are Level 3 (Detailed Reasoning & Option Selection) of a Revisable Reasoning Pyramid. Output L3-1, L3-2, and JSON."},
                {"role": "user", "content": (
                    f"Candidate options:\n{sc['opt_lines']}\n\n"
                    f"Verified Invariant Facts:\n- F1: {sc['inv_expr']} = {sc['inv_val']}\n"
                    f"Level 2 Active Nodes:\n- L2-1: required_units = {units}\n- L2-2: max_per_unit_cost = {cost_m}\n"
                    f"- L2-3: required_mode = {mode}; mandatory_feature = {d['bool_key']}={bool_desc}\n"
                    f"- L2-4: max_base_time_or_lat = {lat_m}\n- L2-5: max_per_unit_resource = {res_m}"
                )},
                {"role": "assistant", "content": out},
            ],
            "expected": {"primary": ans, "preserved": sc["inv_val"]},
        }

    if use_b:
        ev = sc["premise_update"]
        think = f"Latest Evidence shifts live traffic from {d['mode_a']} to {d['mode_b']}. A1, A2, A4, A5 depend on {d['mode_a']} -> INVALID; A3 budget stays VALID."
        out = (
            f"<think>{think}</think>\n"
            f"CHECK A1: INVALID | Active mode switched from {d['mode_a']} to {d['mode_b']}.\n"
            f"CHECK A2: INVALID | Per-unit capacity changes from {sc['per_a']} to {sc['per_b']}.\n"
            f"CHECK A3: VALID | Budget cap {sc['total_budget']} is unchanged.\n"
            f"CHECK A4: INVALID | Penalty changes from {sc['pen_a']} to {sc['pen_b']}.\n"
            f"CHECK A5: INVALID | Overhead changes from {sc['over_a']} to {sc['over_b']}.\n"
            f"SUMMARY: Active profile switched from {d['mode_a']} to {d['mode_b']}, invalidating A1, A2, A4, and A5."
        )
        verdict = "INVALID"
    else:
        ev = sc["irrelevant_update"]
        think = f"Latest Evidence describes a sandboxed {d['distractor_loc']} trial while live traffic stays on {d['mode_a']}. A1..A5 remain VALID."
        out = (
            f"<think>{think}</think>\n"
            f"CHECK A1: VALID | Live traffic stays on {d['mode_a']}.\n"
            f"CHECK A2: VALID | Demand {sc['demand']} and {d['mode_a']} capacity {sc['per_a']} unchanged.\n"
            f"CHECK A3: VALID | Budget ceiling {sc['total_budget']} unchanged.\n"
            f"CHECK A4: VALID | Active penalty {sc['pen_a']} unchanged.\n"
            f"CHECK A5: VALID | Active overhead {sc['over_a']} unchanged.\n"
            f"SUMMARY: Sandboxed trial does not alter live {d['mode_a']} assumptions; all remain VALID."
        )
        verdict = "VALID"
    return {
        "pillar": "rpr_controller_audit",
        "messages": [
            {"role": "system", "content": "You are a strict logical consistency auditor. Output CHECK lines and SUMMARY only (no <tool_call> tags)."},
            {"role": "user", "content": (
                f"Problem Rules:\n{sc['rules_block']}\n\nActive Level-1 Assumptions:\n"
                f"- A1: active_mode={d['mode_a']}; {d['bool_key']} optional\n"
                f"- A2: demand={sc['demand']}, per_unit={sc['per_a']}\n"
                f"- A3: required * {d['cost_key']} <= {sc['total_budget']}\n"
                f"- A4: {d['lat_key']} + {sc['pen_a']} <= {sc['cap_total']}\n"
                f"- A5: {d['res_key']} <= {sc['res_gross']} - {sc['over_a']}\n\n"
                f"Latest Evidence:\n{ev}\n\nCheck each active assumption A1..A5 against Latest Evidence."
            )},
            {"role": "assistant", "content": out},
        ],
        "expected": {"verdict": verdict},
    }


def _make_clean_tool_item(rng: random.Random, idx: int, *, negative: bool) -> dict[str, Any]:
    d = _EXTENDED_DOMAINS[idx % len(_EXTENDED_DOMAINS)]
    station = f"STN-{rng.randint(100, 999)}"
    win = rng.choice([5, 15, 30, 60])
    flow = rng.choice([12.5, 18.0, 24.5, 32.0])

    if negative:
        if idx % 2 == 0:
            a, b = rng.randint(11, 24), rng.randint(12, 22)
            c, e = rng.randint(5, 12), rng.randint(10, 35)
            ans_num = a * b + c * a - e
            steps = _decomp_ledger(a, b, c, a, e)
            user_q = (
                f"Pure analytical calculation for {d['domain']}: compute ({a} * {b}) + ({c} * {a}) - {e} "
                'and output ONLY valid JSON: {"total": <integer>, "status": "ok"}. No external tools apply.'
            )
            think = f"[L1] Self-contained arithmetic; declared hardware tools are irrelevant so abstain from invoking any tool. [L2] {steps}. [L3] Emit JSON."
            exp = {"total": ans_num, "status": "ok"}
            return {
                "pillar": "tool_abstention",
                "messages": [
                    {"role": "system", "content": "You are an analytical assistant.\n" + _TOOLS_BLOCK},
                    {"role": "user", "content": user_q},
                    {"role": "assistant", "content": f"<think>{think}</think>\n{json.dumps(exp)}"},
                ],
                "expected": {"abstain": True, "answer": exp},
            }

        sc = _make_rich_scenario(rng, d)
        w_a = _witness_l3_lines(
            sc["options"], d, active_mode=d["mode_a"], req_bool=False,
            max_cost=sc["cost_max_a"], max_lat=sc["lat_max_a"], max_res=sc["res_max_a"],
        )
        think = (
            f"[L1] Declared station/valve tools are irrelevant to this static constraint task; abstain from tool invocation. "
            f"F1: {sc['inv_steps']}. [L2] mode={d['mode_a']}, units={sc['units_a']}, cost<={sc['cost_max_a']}, lat<={sc['lat_max_a']}, res<={sc['res_max_a']}. "
            f"[L3] {w_a}. Select {sc['ans_a']}."
        )
        exp = {"primary": sc["ans_a"], "preserved": sc["inv_val"]}
        return {
            "pillar": "tool_abstention",
            "messages": [
                {"role": "system", "content": "You are an analytical assistant.\n" + _TOOLS_BLOCK},
                {"role": "user", "content": sc["full_prompt"]},
                {"role": "assistant", "content": f"<think>{think}</think>\n{json.dumps(exp)}"},
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
                {"role": "assistant", "content": f"<think>Live station telemetry requested for {station} ({win}m); invoke fetch_telemetry.</think>\n{tc}"},
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
                {"role": "assistant", "content": f"<think>Hardware self-test requested on {dev} with deep=true; invoke run_diagnostic.</think>\n{tc}"},
            ],
            "expected": {"tool": "run_diagnostic", "args": {"device_id": dev, "deep": "true"}},
        }
    manifold = f"MNF-{rng.randint(1, 9)}"
    bad_tc = _format_tool_call("set_valve_flow", {"manifold": manifold, "liters_per_min": 999.0})
    good_tc = _format_tool_call("set_valve_flow", {"manifold": manifold, "liters_per_min": flow})
    return {
        "pillar": "tool_recovery",
        "messages": [
            {"role": "system", "content": "Adjust actuators carefully and recover from validation errors.\n" + _TOOLS_BLOCK},
            {"role": "user", "content": f"Set manifold {manifold} flow to safe limit (max 40 L/min, target {flow} L/min)."},
            {"role": "assistant", "content": f"<think>Invoke set_valve_flow on {manifold}.</think>\n{bad_tc}"},
            {"role": "user", "content": f"<tool_response>\n{{\"error\": \"liters_per_min=999.0 exceeds hardware cap 40.0; target={flow}\"}}\n</tool_response>"},
            {"role": "assistant", "content": f"<think>[AUDIT] Parameter 999.0 violated cap 40.0; [REBUILD] retry set_valve_flow on {manifold} with liters_per_min={flow}.</think>\n{good_tc}"},
        ],
        "expected": {"tool": "set_valve_flow", "args": {"manifold": manifold, "liters_per_min": str(flow)}},
    }


def _make_ifeval_item(rng: random.Random, idx: int) -> dict[str, Any]:
    d = _EXTENDED_DOMAINS[idx % len(_EXTENDED_DOMAINS)]
    v1, v2, v3 = rng.randint(12, 35), rng.randint(11, 19), rng.randint(10, 29)
    prod = v1 * v2
    diff = prod - v3
    if idx % 2 == 0:
        choice = rng.choice(["BETA", "GAMMA", "DELTA", "SIGMA"])
        obj = {"product": prod, "difference": diff, "choice": choice}
        think = f"[L1] Raw single-line JSON, no fences, exact keys product, difference, choice. [L2] {_decomp_mul(v1, v2)}; {prod}-{v3}={diff}."
        return {
            "pillar": "schema_json",
            "messages": [
                {"role": "system", "content": "Follow output format constraints strictly. No markdown fences unless requested."},
                {"role": "user", "content": (
                    f"Given x={v1}, y={v2}, z={v3}, and tag={choice}, compute product=x*y and difference=x*y-z. "
                    "Output ONLY single-line raw JSON with exact keys ['product', 'difference', 'choice'] and integer numeric values."
                )},
                {"role": "assistant", "content": f"<think>{think}</think>\n{json.dumps(obj)}"},
            ],
            "expected": obj,
        }
    lines = [f"LINE 1: DOMAIN={d['domain'].upper()}", f"LINE 2: SUM={v1 + v2}", "LINE 3: STATUS=VERIFIED"]
    think = f"[L1] Exactly 3 lines prefixed LINE 1..3, no extra text. [L2] {v1}+{v2}={v1+v2}."
    return {
        "pillar": "schema_lines",
        "messages": [
            {"role": "system", "content": "Follow every line prefix and line-count rule strictly."},
            {"role": "user", "content": (
                f"Output EXACTLY 3 lines after </think>:\n"
                f"LINE 1: DOMAIN=<{d['domain']} in uppercase>\n"
                f"LINE 2: SUM=<evaluated {v1} + {v2}>\n"
                "LINE 3: STATUS=VERIFIED\nNo extra lines or bullets."
            )},
            {"role": "assistant", "content": f"<think>{think}</think>\n" + "\n".join(lines)},
        ],
        "expected": {"lines": lines},
    }


def _make_vision_rpr_item(rng: random.Random, idx: int, split_tag: str) -> dict[str, Any]:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    if idx % 2 == 0:
        img = Image.new("RGB", (448, 320), (18, 22, 30))
        draw = ImageDraw.Draw(img)
        cats = rng.sample(["UHF", "LASER", "LFP", "EUV", "DWDM", "CRYO", "POD", "JIT"], 4)
        vals = rng.sample(range(18, 94), 4)
        cols = [c[1] for c in rng.sample(_PALETTE, 4)]
        draw.text((24, 16), f"Subsystem Load ({split_tag.upper()}-{idx})", fill=(235, 240, 248))
        draw.line([(44, 265), (416, 265)], fill=(120, 130, 150), width=2)
        for i, (cat, val, col) in enumerate(zip(cats, vals, cols)):
            x0 = 64 + i * 88
            x1 = x0 + 50
            y0 = 265 - int(val * 2.0)
            draw.rectangle([x0, y0, x1, 264], fill=col, outline=(240, 244, 255), width=1)
            draw.text((x0 + 14, y0 - 16), str(val), fill=(255, 255, 255))
            draw.text((x0 + 8, 274), cat, fill=(210, 218, 232))
        img_path = IMG_DIR / f"{split_tag}_bar_{idx:03d}.png"
        img.save(img_path)

        max_i = max(range(4), key=lambda k: vals[k])
        min_i = min(range(4), key=lambda k: vals[k])
        s01 = vals[0] + vals[1]
        s23 = vals[2] + vals[3]
        tot = s01 + s23
        spread = vals[max_i] - vals[min_i]
        exp = {"highest": cats[max_i], "total": tot, "spread": spread}
        q = (
            "Inspect the bar chart and return ONLY raw JSON with keys "
            '`{"highest": "<category>", "total": <sum_of_4_bars>, "spread": <max_minus_min>}`.'
        )
        think = (
            f"[L1] Read bars left-to-right: {cats[0]}={vals[0]}, {cats[1]}={vals[1]}, {cats[2]}={vals[2]}, {cats[3]}={vals[3]}. "
            f"[L2] Sum: {vals[0]}+{vals[1]}={s01}, {vals[2]}+{vals[3]}={s23}, {s01}+{s23}={tot}. "
            f"Max={cats[max_i]}({vals[max_i]}), min={cats[min_i]}({vals[min_i]}), spread={vals[max_i]}-{vals[min_i]}={spread}. [L3] Emit JSON."
        )
        return {
            "pillar": "vision_rpr",
            "images": [str(img_path)],
            "messages": [
                {"role": "system", "content": "Read visual values accurately, compute step-by-step in <think>, and output raw JSON."},
                {"role": "user", "content": q},
                {"role": "assistant", "content": f"<think>{think}</think>\n{json.dumps(exp)}"},
            ],
            "expected": exp,
        }

    img = Image.new("RGB", (384, 384), (246, 248, 252))
    draw = ImageDraw.Draw(img)
    draw.line([(192, 0), (192, 384)], fill=(180, 188, 204), width=2)
    draw.line([(0, 192), (384, 192)], fill=(180, 188, 204), width=2)
    quads = [("top-left", 24, 24), ("top-right", 216, 24), ("bottom-left", 24, 216), ("bottom-right", 216, 216)]
    picked = rng.sample(_PALETTE, 4)
    shapes = [rng.choice(["circle", "square", "triangle"]) for _ in range(4)]
    items = []
    for (qn, ox, oy), (cn, col), sh in zip(quads, picked, shapes):
        if sh == "circle":
            draw.ellipse([ox + 28, oy + 28, ox + 116, oy + 116], fill=col, outline=(25, 30, 40), width=3)
        elif sh == "square":
            draw.rectangle([ox + 28, oy + 28, ox + 116, oy + 116], fill=col, outline=(25, 30, 40), width=3)
        else:
            draw.polygon([(ox + 72, oy + 24), (ox + 24, oy + 120), (ox + 120, oy + 120)], fill=col, outline=(25, 30, 40))
        items.append((qn, cn, sh))
    img_path = IMG_DIR / f"{split_tag}_quad_{idx:03d}.png"
    img.save(img_path)
    tq, tc, ts = rng.choice(items)
    exp = {"quadrant": tq, "color": tc, "shape": ts}
    q = f'Identify the object in the **{tq}** quadrant and return ONLY raw JSON: `{{"quadrant": "{tq}", "color": "<color>", "shape": "<shape>"}}`.'
    scan = "; ".join(f"{qn}={cn} {sh}" for qn, cn, sh in items)
    think = f"[L1] Scan all 4 quadrants: {scan}. [L2] Target {tq} has color={tc}, shape={ts}. [L3] Emit JSON."
    return {
        "pillar": "vision_rpr",
        "images": [str(img_path)],
        "messages": [
            {"role": "system", "content": "Inspect spatial regions carefully in <think> and output raw JSON."},
            {"role": "user", "content": q},
            {"role": "assistant", "content": f"<think>{think}</think>\n{json.dumps(exp)}"},
        ],
        "expected": exp,
    }


def _load_and_validate_frontier_files() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load bespoke frontier-authored files, validate formatting, and split 92% train / 8% val."""
    train_frontier: list[dict[str, Any]] = []
    val_frontier: list[dict[str, Any]] = []
    for fname in FRONTIER_FILES:
        fpath = CORPUS_DIR / fname
        if not fpath.exists():
            print(f"[Warn] Bespoke frontier file {fpath} not present; skipping.", flush=True)
            continue
        items: list[dict[str, Any]] = []
        for idx, line in enumerate(fpath.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            msgs = obj.get("messages")
            assert isinstance(msgs, list) and len(msgs) >= 2, f"{fname}:{idx} invalid messages"
            has_asst = False
            for m in msgs:
                if m.get("role") == "assistant":
                    has_asst = True
                    c = str(m.get("content", ""))
                    assert "<think>" in c and "</think>" in c, f"{fname}:{idx} missing <think>...</think>"
                    post = c.split("</think>", 1)[1].strip()
                    assert len(post) > 0, f"{fname}:{idx} empty post-think answer"
                    assert "<tool_call" not in c.lower(), f"{fname}:{idx} unexpected <tool_call>"
            assert has_asst, f"{fname}:{idx} no assistant turn"
            obj.setdefault("images", [])
            obj.setdefault("pillar", fname.replace(".jsonl", ""))
            items.append(obj)
        n_val = min(4, max(1, len(items) // 12))
        val_frontier.extend(items[:n_val])
        train_frontier.extend(items[n_val:])
        print(f"[Frontier Loaded] {fname}: {len(items)} unique items ({len(items)-n_val} train, {n_val} val)", flush=True)
    return train_frontier, val_frontier


def _verify_all_generated_code_samples(rows: list[dict[str, Any]]) -> None:
    checked = 0
    for r in rows:
        ver = r.get("verifier") or {}
        if ver.get("kind") != "python_code":
            continue
        ans = r["messages"][-1]["content"]
        code = ans.split("```python", 1)[1].split("```", 1)[0].strip()
        proc = subprocess.run(
            [sys.executable, "-I", "-c", f"{code}\n\n{ver['test']}\n"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        assert proc.returncode == 0, f"Code sample failed self-test ({r.get('pillar')}): {proc.stderr}"
        checked += 1
    print(f"[Self-Test] Verified {checked} executable Python RPR/AI/Debug code tasks (100% pass).", flush=True)


def build_sft_split(
    seed: int,
    *,
    is_val: bool,
    extra_frontier: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = list(extra_frontier or [])
    split_tag = "val" if is_val else "train"

    if is_val:
        counts = {
            "debug": 16,
            "ai_weights": 14,
            "self_improve": 14,
            "single_stream": 18,
            "controller": 12,
            "tool_pos": 6,
            "tool_neg": 6,
            "ifeval": 4,
            "vision": 6,
        }
    else:
        counts = {
            "debug": 110,
            "ai_weights": 96,
            "self_improve": 96,
            "single_stream": 135,
            "controller": 88,
            "tool_pos": 36,
            "tool_neg": 32,
            "ifeval": 28,
            "vision": 44,
        }

    for i in range(counts["debug"]):
        rows.append(make_diagnostic_debug_item(rng, i))
    for i in range(counts["ai_weights"]):
        rows.append(make_ai_weights_mechanics_item(rng, i))
    for i in range(counts["self_improve"]):
        rows.append(make_self_improvement_meta_item(rng, i))
    for i in range(counts["single_stream"]):
        rows.append(_make_single_stream_rpr_item(rng, i))
    for i in range(counts["controller"]):
        rows.append(_make_controller_stage_item(rng, i))
    for i in range(counts["tool_pos"]):
        rows.append(_make_clean_tool_item(rng, i, negative=False))
    for i in range(counts["tool_neg"]):
        rows.append(_make_clean_tool_item(rng, i, negative=True))
    for i in range(counts["ifeval"]):
        rows.append(_make_ifeval_item(rng, i))
    for i in range(counts["vision"]):
        rows.append(_make_vision_rpr_item(rng, i, split_tag))

    rng.shuffle(rows)
    for idx, r in enumerate(rows, start=1):
        r["id"] = f"{split_tag}-{idx:04d}"
        r.setdefault("images", [])
    return rows


def _strip_prior_think(msgs: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for m in msgs:
        if m["role"] == "assistant" and "</think>" in m["content"]:
            out.append({"role": "assistant", "content": m["content"].split("</think>", 1)[1].strip()})
        else:
            out.append(m)
    return out


def build_rlvr_score_96(seed: int) -> list[dict[str, Any]]:
    """Build 96 shuffled verifiable RLVR + 2-turn SCoRe episodes spanning all reasoning & AI pillars."""
    rng = random.Random(seed)
    episodes: list[dict[str, Any]] = []

    # 1) 24 2-Turn SCoRe Single-Stream RPR episodes (18 premise switches + 6 distractors)
    for i in range(24):
        d = _EXTENDED_DOMAINS[i % len(_EXTENDED_DOMAINS)]
        sc = _make_rich_scenario(rng, d)
        is_flip = i < 18
        sys_m = "You are a careful analytical problem solver. Follow all constraints strictly.\n" + _ANSWER_FORMAT_NOTE
        upd = sc["premise_update"] if is_flip else sc["irrelevant_update"]
        exp_pri = sc["ans_b"] if is_flip else sc["ans_a"]
        ref_think = _single_stream_turn2_premise_think(sc) if is_flip else _single_stream_turn2_irrel_think(sc)
        episodes.append({
            "id": f"score-2turn-{i+1:02d}",
            "kind": "math_opt",
            "messages": [
                {"role": "system", "content": sys_m},
                {"role": "user", "content": sc["full_prompt"]},
                {"role": "assistant", "content": json.dumps({"primary": sc["ans_a"], "preserved": sc["inv_val"]})},
                {"role": "user", "content": f"New evidence arrived:\n{upd}\n\nProvide your updated answer.\n{_ANSWER_FORMAT_NOTE}"},
            ],
            "reference_answer": f"<think>{ref_think}</think>\n" + json.dumps({"primary": exp_pri, "preserved": sc["inv_val"]}),
            "verifier": {
                "primary": exp_pri,
                "preserved": sc["inv_val"],
                "anchor_primary": sc["ans_a"] if is_flip else None,
                "forbid_tool_call": True,
            },
        })

    # 2) 18 Diagnostic Bug-Hunt & Cognitive-Trap RLVR episodes
    for i in range(18):
        ex = make_diagnostic_debug_item(rng, i)
        msgs = _strip_prior_think(ex["messages"][:-1])
        if "verifier" in ex:
            episodes.append({
                "id": f"rlvr-debug-{i+1:02d}",
                "kind": "python_code",
                "messages": msgs,
                "reference_answer": ex["messages"][-1]["content"],
                "verifier": {**ex["verifier"], "forbid_tool_call": True},
            })
        else:
            episodes.append({
                "id": f"rlvr-trap-{i+1:02d}",
                "kind": "schema_json",
                "messages": msgs,
                "reference_answer": ex["messages"][-1]["content"],
                "verifier": {"expected": ex["expected"], "forbid_tool_call": True},
            })

    # 3) 20 AI Weight Mechanics & Self-Improvement RLVR episodes (code + math)
    for i in range(20):
        ex = make_ai_weights_mechanics_item(rng, i) if i < 10 else make_self_improvement_meta_item(rng, i - 10)
        msgs = _strip_prior_think(ex["messages"][:-1])
        if "verifier" in ex:
            episodes.append({
                "id": f"rlvr-ai-code-{i+1:02d}",
                "kind": "python_code",
                "messages": msgs,
                "reference_answer": ex["messages"][-1]["content"],
                "verifier": {**ex["verifier"], "forbid_tool_call": True},
            })
        else:
            episodes.append({
                "id": f"rlvr-ai-math-{i+1:02d}",
                "kind": "schema_json",
                "messages": msgs,
                "reference_answer": ex["messages"][-1]["content"],
                "verifier": {"expected": ex["expected"], "forbid_tool_call": True},
            })

    # 4) 12 1-Turn Multi-Hop Option + Ledger Math episodes
    for i in range(12):
        d = _EXTENDED_DOMAINS[i % len(_EXTENDED_DOMAINS)]
        sc = _make_rich_scenario(rng, d)
        t1_ref = "<think>" + _single_stream_turn1_think(sc) + "</think>\n" + json.dumps({"primary": sc["ans_a"], "preserved": sc["inv_val"]})
        episodes.append({
            "id": f"rlvr-math-{i+1:02d}",
            "kind": "math_opt",
            "messages": [
                {"role": "system", "content": "Solve concisely without repeating the prompt. Output evaluated JSON."},
                {"role": "user", "content": sc["full_prompt"]},
            ],
            "reference_answer": t1_ref,
            "verifier": {"primary": sc["ans_a"], "preserved": sc["inv_val"], "forbid_tool_call": True},
        })

    # 5) 10 RPR Auditor + 8 Tool Pos/Neg + 4 Strict IFEval episodes
    for i in range(10):
        ex = _make_controller_stage_item(rng, i * 4 + 3)
        episodes.append({
            "id": f"rlvr-audit-{i+1:02d}",
            "kind": "audit",
            "messages": ex["messages"][:-1],
            "reference_answer": ex["messages"][-1]["content"],
            "verifier": {"verdict": ex["expected"]["verdict"], "forbid_tool_call": True},
        })
    for i in range(8):
        neg = i >= 4
        ex = _make_clean_tool_item(rng, i, negative=neg)
        msgs = _strip_prior_think(ex["messages"][:-1])
        ver = (
            {"abstain": True, "forbid_tool_call": True, "expected": ex["expected"]["answer"]}
            if neg
            else {"abstain": False, "tool": ex["expected"]["tool"], "args": ex["expected"]["args"]}
        )
        episodes.append({
            "id": f"rlvr-tool-{i+1:02d}",
            "kind": "tool_neg" if neg else "tool_pos",
            "messages": msgs,
            "reference_answer": ex["messages"][-1]["content"],
            "verifier": ver,
        })
    for i in range(4):
        ex = _make_ifeval_item(rng, i)
        episodes.append({
            "id": f"rlvr-ifeval-{i+1:02d}",
            "kind": ex["pillar"],
            "messages": ex["messages"][:-1],
            "reference_answer": ex["messages"][-1]["content"],
            "verifier": {"expected": ex["expected"], "forbid_tool_call": True},
        })

    rng.shuffle(episodes)
    return episodes


def build_heldout_broad_42(seed: int) -> list[dict[str, Any]]:
    """Build 42-task unseen multi-domain gate across 7 pillars (6 tasks per pillar)."""
    rng = random.Random(seed)
    tasks: list[dict[str, Any]] = []

    # Pillar 1 (6): Standalone 2-Turn RPR Belief Revision (4 premise switches + 2 distractors)
    for i in range(6):
        d = _EXTENDED_DOMAINS[(i + 3) % len(_EXTENDED_DOMAINS)]
        sc = _make_rich_scenario(rng, d)
        is_flip = i < 4
        sys_m = "You are a careful analytical problem solver. Follow all constraints strictly.\n" + _ANSWER_FORMAT_NOTE
        upd = sc["premise_update"] if is_flip else sc["irrelevant_update"]
        exp_pri = sc["ans_b"] if is_flip else sc["ans_a"]
        tasks.append({
            "id": f"broad-rpr2t-{i+1:02d}",
            "pillar": "standalone_rpr_2turn",
            "kind": "math_opt",
            "images": [],
            "messages": [
                {"role": "system", "content": sys_m},
                {"role": "user", "content": sc["full_prompt"]},
                {"role": "assistant", "content": json.dumps({"primary": sc["ans_a"], "preserved": sc["inv_val"]})},
                {"role": "user", "content": f"New evidence arrived:\n{upd}\n\nProvide your updated answer.\n{_ANSWER_FORMAT_NOTE}"},
            ],
            "verifier": {"primary": exp_pri, "preserved": sc["inv_val"], "forbid_tool_call": True},
        })

    # Pillar 2 (6): Diagnostic Root-Cause Debugging & Cognitive Traps (4 code fixes + 2 math traps)
    for i, c_idx in enumerate([0, 1, 2, 4, 6, 7]):
        ex = make_diagnostic_debug_item(rng, c_idx)
        msgs = _strip_prior_think(ex["messages"][:-1])
        tasks.append({
            "id": f"broad-debug-{i+1:02d}",
            "pillar": "diagnostic_debugging",
            "kind": "python_code" if "verifier" in ex else "schema_json",
            "images": [],
            "messages": msgs,
            "verifier": {**ex["verifier"], "forbid_tool_call": True} if "verifier" in ex else {"expected": ex["expected"], "forbid_tool_call": True},
        })

    # Pillar 3 (6): AI Weights & Self-Improvement Science (3 weight mechanics + 3 self-improvement)
    for i, (fn_gen, c_idx) in enumerate([
        (make_ai_weights_mechanics_item, 0),
        (make_ai_weights_mechanics_item, 1),
        (make_ai_weights_mechanics_item, 4),
        (make_self_improvement_meta_item, 0),
        (make_self_improvement_meta_item, 1),
        (make_self_improvement_meta_item, 2),
    ]):
        ex = fn_gen(rng, c_idx)
        msgs = _strip_prior_think(ex["messages"][:-1])
        tasks.append({
            "id": f"broad-ai-{i+1:02d}",
            "pillar": "ai_weights_and_self_improve",
            "kind": "python_code" if "verifier" in ex else "schema_json",
            "images": [],
            "messages": msgs,
            "verifier": {**ex["verifier"], "forbid_tool_call": True} if "verifier" in ex else {"expected": ex["expected"], "forbid_tool_call": True},
        })

    # Pillar 4 (6): Unassisted Multi-Hop Math + Option Verification
    for i in range(6):
        d = _EXTENDED_DOMAINS[i % len(_EXTENDED_DOMAINS)]
        sc = _make_rich_scenario(rng, d)
        tasks.append({
            "id": f"broad-math-{i+1:02d}",
            "pillar": "multihop_math",
            "kind": "math_opt",
            "images": [],
            "messages": [
                {"role": "system", "content": "Solve concisely without repeating the prompt. Output evaluated JSON."},
                {"role": "user", "content": sc["full_prompt"]},
            ],
            "verifier": {"primary": sc["ans_a"], "preserved": sc["inv_val"], "forbid_tool_call": True},
        })

    # Pillar 5 (6): RPR Auditor + Strict IFEval Schemas (4 audits + 2 IFEval)
    for i in range(4):
        ex = _make_controller_stage_item(rng, i * 4 + 3)
        tasks.append({
            "id": f"broad-audit-{i+1:02d}",
            "pillar": "audit_and_ifeval",
            "kind": "audit",
            "images": [],
            "messages": ex["messages"][:-1],
            "verifier": {"verdict": ex["expected"]["verdict"], "forbid_tool_call": True},
        })
    for i in range(2):
        ex = _make_ifeval_item(rng, i)
        tasks.append({
            "id": f"broad-ifeval-{i+1:02d}",
            "pillar": "audit_and_ifeval",
            "kind": ex["pillar"],
            "images": [],
            "messages": ex["messages"][:-1],
            "verifier": {"expected": ex["expected"], "forbid_tool_call": True},
        })

    # Pillar 6 (6): Tool Calling & Abstention Calibration (3 positive + 3 negative abstention)
    for i in range(3):
        ex = _make_clean_tool_item(rng, i, negative=False)
        msgs = _strip_prior_think(ex["messages"][:-1])
        tasks.append({
            "id": f"broad-toolpos-{i+1:02d}",
            "pillar": "tool_calibration",
            "kind": "tool_pos",
            "images": [],
            "messages": msgs,
            "verifier": {"abstain": False, "tool": ex["expected"]["tool"], "args": ex["expected"]["args"]},
        })
    for i in range(3):
        ex = _make_clean_tool_item(rng, i, negative=True)
        tasks.append({
            "id": f"broad-toolneg-{i+1:02d}",
            "pillar": "tool_calibration",
            "kind": "tool_neg",
            "images": [],
            "messages": ex["messages"][:-1],
            "verifier": {"abstain": True, "forbid_tool_call": True, "expected": ex["expected"]["answer"]},
        })

    # Pillar 7 (6): Multimodal Vision RPR (3 bar charts + 3 quadrant spatial grids)
    for i in range(6):
        ex = _make_vision_rpr_item(rng, i, "eval")
        tasks.append({
            "id": f"broad-vision-{i+1:02d}",
            "pillar": "vision_multimodal",
            "kind": "schema_json",
            "images": ex["images"],
            "messages": ex["messages"][:-1],
            "verifier": {"expected": ex["expected"], "forbid_tool_call": True},
        })

    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble the 100%-frontier RPR v4 corpus.")
    parser.add_argument("--require-frontier-files", action="store_true", help="Assert all 3 bespoke frontier JSONL files exist.")
    args = parser.parse_args()

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    if args.require_frontier_files:
        for fname in FRONTIER_FILES:
            assert (CORPUS_DIR / fname).exists(), f"Missing required bespoke frontier file: {CORPUS_DIR / fname}"

    train_frontier, val_frontier = _load_and_validate_frontier_files()
    train_rows = build_sft_split(seed=20261126, is_val=False, extra_frontier=train_frontier)
    val_rows = build_sft_split(seed=20261127, is_val=True, extra_frontier=val_frontier)
    rlvr_rows = build_rlvr_score_96(seed=20261128)
    broad_42 = build_heldout_broad_42(seed=20261129)

    _verify_all_generated_code_samples(train_rows + val_rows)
    assert_decontaminated(train_rows + val_rows + rlvr_rows + broad_42)

    for fname, dataset in [
        ("train_rpr_v4.jsonl", train_rows),
        ("val_rpr_v4.jsonl", val_rows),
        ("rlvr_score_96.jsonl", rlvr_rows),
        ("heldout_broad_42.jsonl", broad_42),
    ]:
        (CORPUS_DIR / fname).write_text("\n".join(json.dumps(r) for r in dataset) + "\n", encoding="utf-8")

    n_turns = sum(sum(1 for m in r["messages"] if m["role"] == "assistant") for r in train_rows)
    print(
        f"[RPR v4 Corpus Built] {CORPUS_DIR}:\n"
        f"  train_rpr_v4 = {len(train_rows)} conversations ({n_turns} assistant turns)\n"
        f"  val_rpr_v4   = {len(val_rows)} conversations\n"
        f"  rlvr_score_96= {len(rlvr_rows)} shuffled RLVR/SCoRe episodes\n"
        f"  broad_42     = {len(broad_42)} unseen evaluation tasks across 7 pillars\n"
        f"  100% Python unit-tested & zero benchmark contamination verified!",
        flush=True,
    )


if __name__ == "__main__":
    main()
