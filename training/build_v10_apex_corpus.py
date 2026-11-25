"""Build the Master RPR v10 APEX Corpus (`3.18B` `40L` Pure 16-bit `bfloat16`, `256k` `LongRoPE2YaRN` Context):
1. Self-Discover + Hypothesis-Falsify-Verify Complex Logic & SAT (`Zebra/Einstein`, `Knights-and-Knaves`, `3-SAT`,
   `Quantified Modal/Set Logic`, `Backward-Induction Game Trees`, `Graph Coloring`).
2. Real-Time Web Research & Verifiable Inline Citations (`Search-R1` + `WebThinker` + `CiteCheck` `[S1]/[S2]`
   claim-level attribution, conflict/erratum resolution, and honest abstention when unverified).
3. Zero-Prior Unseen Problem Solving (`OOD` Formal Systems, Non-Mutating Functional Code, Exact Relative Tool Paths,
   `log_softmax` vs `softmax`, Elementwise Python Ops).
4. Advanced Recursive Self-Improvement (`Darwin Gödel Machine` Archive Selection, `Transformer^2` Prompt-Dispatched
   `SVF` Singular Mixing, `DAPO` + `Dr. GRPO` Unbiased RLVR, `AlphaEdit`/`MEMIT`/`TIES`, `LongRoPE2` Dual-Path RoPE).
5. Long-Context Multi-Needle Synthesis & Contradiction Auditing (`4k–7k` SFT multi-document `[DOC-XXXX]` training
   + `16k..64k` Multi-Hop Needle & Citation Held-Out Suite).

Outputs inside `training/data/rpr_v10_apex_corpus/`:
  - `heldout_v10_apex_36.jsonl`         : 36 strictly held-out frontier tasks across 4 pillars (9 each)
  - `heldout_long_256k_needles_12.jsonl`: 12 strictly held-out 16k/32k/48k/64k multi-hop needle + citation tasks
  - `train_v10_sft.jsonl`               : ~980 deduplicated, `python3 -I` unit-verified SFT conversations
  - `val_v10_joint.jsonl`               : 100 stratified validation conversations
  - `orpo_v10_pairs.jsonl`              : 240 contrastive `(messages, chosen, rejected)` preference pairs
  - `dapo_v10_hard_episodes.jsonl`      : 96 challenging verifiable episodes for `DAPO` Dynamic Sampling + `Dr. GRPO`
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from training.build_rpr_v5_rsi_corpus import (
    V5_CORPUS_DIR,
    _deduplicate_rows,
    _verify_executable_code_rows,
)
from training.build_training_data import assert_decontaminated
from training.build_v9_rsi_zero_prior_cline_corpus import (
    V9_CORPUS_DIR,
    _assert_zero_heldout_leakage,
)


V10_CORPUS_DIR = Path(__file__).resolve().parent / "data" / "rpr_v10_apex_corpus"


def _xml_call(fn_name: str, args: dict[str, Any]) -> str:
    body = []
    for k, v in args.items():
        if isinstance(v, bool):
            vs = "true" if v else "false"
        elif isinstance(v, (dict, list)):
            vs = json.dumps(v)
        else:
            vs = str(v)
        body.append(f"<parameter={k}>\n{vs}\n</parameter>")
    inner = "\n".join(body)
    return f"<tool_call>\n<function={fn_name}>\n{inner}\n</function>\n</tool_call>"


# ===========================================================================
# 1. Held-Out 36-Task Frontier Benchmark (`heldout_v10_apex_36.jsonl`)
#    Built FIRST with strictly disjoint names/symbols for 0-overlap guarantee!
# ===========================================================================


def build_heldout_v10_apex_36() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return `(heldout_36, heldout_code_ref_checks)` across 4 frontier pillars (9 tasks each)."""
    tasks: list[dict[str, Any]] = []
    code_checks: list[dict[str, Any]] = []

    # -----------------------------------------------------------------------
    # Pillar 1: `complex_logic_sat` (9 tasks)
    # -----------------------------------------------------------------------
    # 1.1 & 1.2: 4-Entity Constraint Grid (`Zebra / Einstein` unique assignment)
    # Bays 1..4 hold modules {Quasar, Nadir, Zenith, Vortex} with languages {Ada, Zig, Nim, Lua}.
    # Ground truth 1: Bay1=(Nadir, Nim), Bay2=(Quasar, Ada), Bay3=(Vortex, Zig), Bay4=(Zenith, Lua)
    tasks.append({
        "id": "v10-logic-01",
        "pillar": "complex_logic_sat",
        "kind": "json_values",
        "messages": [{
            "role": "user",
            "content": (
                "Four server bays numbered `1, 2, 3, 4` from left to right each host exactly one daemon "
                "(`Nadir`, `Quasar`, `Vortex`, `Zenith`) written in a distinct language (`Ada`, `Lua`, `Nim`, `Zig`):\n"
                "1. `Quasar` is immediately to the left of `Vortex`.\n"
                "2. The `Ada` daemon is in Bay `2`.\n"
                "3. `Zenith` runs `Lua` and is at one of the two end bays (`1` or `4`).\n"
                "4. `Nadir` is somewhere to the left of `Quasar`.\n"
                "5. The `Zig` daemon is immediately to the left of `Zenith`.\n"
                'Deduce the unique assignment and return ONLY JSON `{"bay_1_lang": "<str>", "vortex_bay": <int>, "zig_daemon": "<str>", "zenith_bay": <int>}`.'
            ),
        }],
        "verifier": {
            "expected": {"bay_1_lang": "Nim", "vortex_bay": 3, "zig_daemon": "Vortex", "zenith_bay": 4},
            "forbid_tool_call": True,
        },
    })

    # Ground truth 2: Racks 1..4: 1=(Beryl, 40), 2=(Onyx, 10), 3=(Topaz, 30), 4=(Jade, 20)
    tasks.append({
        "id": "v10-logic-02",
        "pillar": "complex_logic_sat",
        "kind": "json_values",
        "messages": [{
            "role": "user",
            "content": (
                "Four optical switches (`Beryl`, `Jade`, `Onyx`, `Topaz`) occupy slots `1, 2, 3, 4` and carry distinct bandwidths `{10, 20, 30, 40}` Gbps:\n"
                "1. `Topaz` is in slot `3`.\n"
                "2. The switch in slot `1` has bandwidth strictly greater than `Topaz`, and `Topaz` has bandwidth strictly greater than `Jade`.\n"
                "3. `Onyx` has `10` Gbps and is adjacent to `Beryl`.\n"
                "4. `Jade` is NOT in slot `2`.\n"
                'Return ONLY JSON `{"slot_1_switch": "<str>", "topaz_gbps": <int>, "jade_slot": <int>, "jade_gbps": <int>}`.'
            ),
        }],
        "verifier": {
            "expected": {"slot_1_switch": "Beryl", "topaz_gbps": 30, "jade_slot": 4, "jade_gbps": 20},
            "forbid_tool_call": True,
        },
    })

    # 1.3 & 1.4: Knights-and-Knaves Biconditional / Exact-Truth-Count Puzzles
    # P, Q, R in {Knight(True), Knave(False)}:
    # P says: "Q and R are of the same type." (P <-> (Q == R))
    # Q says: "P is a Knave." (Q <-> not P)
    # R says: "At most one of P, Q, R is a Knight." (R <-> (P+Q+R <= 1))
    # Check:
    # - P=T => Q==R. From Q <-> not P=F, Q=F, so R=F. Then P+Q+R = 1 <= 1, so R's statement is TRUE, contradicting R=F!
    # - P=F => Q!=R, and from Q <-> not F = T, Q=T! Since Q!=R, R=F! Now P+Q+R = 0+1+0 = 1 <= 1 -> wait: if P+Q+R=1, R's statement "P+Q+R <= 1" is TRUE, contradicting R=F!
    # Let's verify programmatically right here with itertools.product!
    sat_kn: list[tuple[bool, bool, bool]] = []
    for p_b, q_b, r_b in itertools.product([False, True], repeat=3):
        s_p = (q_b and r_b)
        s_q = (not p_b)
        s_r = (p_b == q_b)
        if (p_b == s_p) and (q_b == s_q) and (r_b == s_r):
            sat_kn.append((p_b, q_b, r_b))
    assert len(sat_kn) == 1 and sat_kn[0] == (False, True, False)
    tasks.append({
        "id": "v10-logic-03",
        "pillar": "complex_logic_sat",
        "kind": "json_values",
        "messages": [{
            "role": "user",
            "content": (
                "Three islanders (`Alden`, `Bryn`, `Corin`) are each either a Knight (always tells the truth) or a Knave (always lies):\n"
                "- `Alden` says: \"Both `Bryn` and `Corin` are Knights.\"\n"
                "- `Bryn` says: \"`Alden` is a Knave.\"\n"
                "- `Corin` says: \"`Alden` and `Bryn` are the same type (both Knights or both Knaves).\"\n"
                'Determine the unique valid assignment and return ONLY JSON `{"Alden": <bool>, "Bryn": <bool>, "Corin": <bool>, "knight_count": <int>}` (`true` = Knight, `false` = Knave).'
            ),
        }],
        "verifier": {
            "expected": {"Alden": False, "Bryn": True, "Corin": False, "knight_count": 1},
            "forbid_tool_call": True,
        },
    })

    # 1.4: 4-variable 3-CNF SAT unique model
    sat_4: list[tuple[bool, bool, bool, bool]] = []
    for x1, x2, x3, x4 in itertools.product([False, True], repeat=4):
        c1 = x1 or x2
        c2 = (not x1) or x3
        c3 = (not x2) or x3
        c4 = (not x3) or x4
        c5 = (not x1) or (not x4)
        c6 = x2 or (not x4)
        c7 = (not x2) or (not x1)
        # Wait let's test uniqueness:
        # c1(x1|x2), c2(~x1|x3), c3(~x2|x3) => x3 MUST be True!
        # c4(~x3|x4) => x4 MUST be True!
        # c5(~x1|~x4) => since x4=T, x1 MUST be False!
        # c1(x1|x2) => since x1=F, x2 MUST be True!
        if c1 and c2 and c3 and c4 and c5 and c6 and c7:
            sat_4.append((x1, x2, x3, x4))
    assert len(sat_4) == 1 and sat_4[0] == (False, True, True, True)
    tasks.append({
        "id": "v10-logic-04",
        "pillar": "complex_logic_sat",
        "kind": "json_values",
        "messages": [{
            "role": "user",
            "content": (
                "Find the unique Boolean assignment `(x1, x2, x3, x4)` satisfying all 7 clauses simultaneously:\n"
                "  C1: `(x1 OR x2)`\n"
                "  C2: `(NOT x1 OR x3)`\n"
                "  C3: `(NOT x2 OR x3)`\n"
                "  C4: `(NOT x3 OR x4)`\n"
                "  C5: `(NOT x1 OR NOT x4)`\n"
                "  C6: `(x2 OR NOT x4)`\n"
                "  C7: `(NOT x1 OR NOT x2)`\n"
                'Return ONLY JSON `{"x1": <bool>, "x2": <bool>, "x3": <bool>, "x4": <bool>, "true_count": <int>}`.'
            ),
        }],
        "verifier": {
            "expected": {"x1": False, "x2": True, "x3": True, "x4": True, "true_count": 3},
            "forbid_tool_call": True,
        },
    })

    # 1.5: Subtraction Game Backward Induction (`S = {1, 3, 4}`, starting stones `N = 18`, normal play: last move wins)
    win = [False] * 19
    for n_s in range(1, 19):
        win[n_s] = any(n_s - m >= 0 and not win[n_s - m] for m in (1, 3, 4))
    winning_moves_18 = [m for m in (1, 3, 4) if not win[18 - m]]
    p_positions = [k for k in range(1, 19) if not win[k]]
    assert win[18] is True and len(winning_moves_18) == 1
    tasks.append({
        "id": "v10-logic-05",
        "pillar": "complex_logic_sat",
        "kind": "json_values",
        "messages": [{
            "role": "user",
            "content": (
                "In a two-player normal-play subtraction game (the player who takes the last token wins; a player with no legal move loses), "
                "each turn a player must remove `1`, `3`, or `4` tokens from a single pile of `N = 18` tokens.\n"
                "Using backward induction (`P`-positions vs `N`-positions):\n"
                "1. Determine whether the first player has a forced win (`first_player_wins`).\n"
                "2. Find the unique winning opening move `opening_remove` in `{1, 3, 4}`.\n"
                "3. Count how many `n` in `1..18` (inclusive) are losing `P`-positions (`num_p_positions_1_to_18`).\n"
                'Return ONLY JSON `{"first_player_wins": <bool>, "opening_remove": <int>, "num_p_positions_1_to_18": <int>}`.'
            ),
        }],
        "verifier": {
            "expected": {
                "first_player_wins": True,
                "opening_remove": winning_moves_18[0],
                "num_p_positions_1_to_18": len(p_positions),
            },
            "forbid_tool_call": True,
        },
    })

    # 1.6: Quantified Relational Logic over Finite Domain `D = {1, 2, 3, 4}`
    dom = [1, 2, 3, 4]
    rel = lambda x, y: (x + y) % 3 == 0 or x == 2 * y
    q_forall_exists = all(any(rel(x, y) for y in dom) for x in dom)
    q_exists_forall = any(all(rel(x, y) for x in dom) for y in dom)
    pair_cnt = sum(1 for x in dom for y in dom if rel(x, y))
    tasks.append({
        "id": "v10-logic-06",
        "pillar": "complex_logic_sat",
        "kind": "json_values",
        "messages": [{
            "role": "user",
            "content": (
                "Let domain `D = {1, 2, 3, 4}` and binary relation `R(x, y)` hold iff `((x + y) % 3 == 0) or (x == 2 * y)`.\n"
                "Evaluate over `x, y in D`:\n"
                "1. `forall_x_exists_y`: `∀x ∈ D, ∃y ∈ D : R(x, y)`\n"
                "2. `exists_y_forall_x`: `∃y ∈ D, ∀x ∈ D : R(x, y)`\n"
                "3. `total_true_pairs`: exact count of ordered pairs `(x, y) ∈ D × D` satisfying `R(x, y)`.\n"
                'Return ONLY JSON `{"forall_x_exists_y": <bool>, "exists_y_forall_x": <bool>, "total_true_pairs": <int>}`.'
            ),
        }],
        "verifier": {
            "expected": {
                "forall_x_exists_y": q_forall_exists,
                "exists_y_forall_x": q_exists_forall,
                "total_true_pairs": pair_cnt,
            },
            "forbid_tool_call": True,
        },
    })

    # 1.7: Chromatic Number of Wheel Graph `W_6` (hub `H` connected to all 5 vertices of odd cycle `C_5`)
    tasks.append({
        "id": "v10-logic-07",
        "pillar": "complex_logic_sat",
        "kind": "json_values",
        "messages": [{
            "role": "user",
            "content": (
                "Consider the 6-vertex wheel graph `W_6` formed by a 5-vertex cycle `v1-v2-v3-v4-v5-v1` (`C_5`) plus a central hub `H` adjacent to all 5 cycle vertices.\n"
                "1. What is the chromatic number `chi_cycle` of `C_5` alone?\n"
                "2. What is the chromatic number `chi_wheel` of `W_6`?\n"
                "3. Is `W_6` 3-colorable (`is_3_colorable`)?\n"
                'Return ONLY JSON `{"chi_cycle": <int>, "chi_wheel": <int>, "is_3_colorable": <bool>}`.'
            ),
        }],
        "verifier": {
            "expected": {"chi_cycle": 3, "chi_wheel": 4, "is_3_colorable": False},
            "forbid_tool_call": True,
        },
    })

    # 1.8: Extensive-Form 2-Player Minimax Tree with Chance/Subgame Perfection
    # P1 chooses Left or Right:
    # - Left -> P2 chooses L1 (payoff P1=4, P2=3) or L2 (payoff P1=7, P2=1). P2 maximizes P2 -> picks L1 (3 > 1) -> P1 gets 4.
    # - Right -> P2 chooses R1 (payoff P1=6, P2=2) or R2 -> P1 chooses R2a (payoff P1=5, P2=5) or R2b (payoff P1=2, P2=8).
    #   At R2, P1 maximizes P1 -> picks R2a (5 > 2), giving P2=5.
    #   At Right, P2 compares R1 (P2=2) vs R2 (P2=5) -> picks R2 (5 > 2), giving P1=5!
    # At Root, P1 compares Left (P1=4) vs Right (P1=5) -> picks Right! Subgame-perfect outcome: (5, 5)!
    tasks.append({
        "id": "v10-logic-08",
        "pillar": "complex_logic_sat",
        "kind": "json_values",
        "messages": [{
            "role": "user",
            "content": (
                "Solve this perfect-information sequential game by backward induction (Subgame-Perfect Nash Equilibrium; each player strictly maximizes their own payoff in `(u1, u2)`):\n"
                "- Root (`Player 1`): chooses `Left` or `Right`.\n"
                "  - If `Left`, `Player 2` chooses `L1` -> `(4, 3)` or `L2` -> `(7, 1)`.\n"
                "  - If `Right`, `Player 2` chooses `R1` -> `(6, 2)` or `R2` (which hands play back to `Player 1`).\n"
                "    - At `R2`, `Player 1` chooses `R2a` -> `(5, 5)` or `R2b` -> `(2, 8)`.\n"
                'Return ONLY JSON `{"root_choice": "<Left|Right>", "p1_payoff": <int>, "p2_payoff": <int>}`.'
            ),
        }],
        "verifier": {
            "expected": {"root_choice": "Right", "p1_payoff": 5, "p2_payoff": 5},
            "forbid_tool_call": True,
        },
    })

    # 1.9: Pure Python DPLL 2-SAT / Small-CNF Solver (`solve_small_cnf_horn`)
    fn_sat = "find_lex_first_sat"
    ref_sat_code = (
        "def find_lex_first_sat(num_vars: int, clauses: list[list[int]]) -> list[bool] | None:\n"
        "    for bits in range(1 << num_vars):\n"
        "        assign = [bool((bits >> (num_vars - 1 - i)) & 1) for i in range(num_vars)]\n"
        "        ok = True\n"
        "        for cl in clauses:\n"
        "            if not any((assign[lit - 1] if lit > 0 else (not assign[-lit - 1])) for lit in cl):\n"
        "                ok = False\n"
        "                break\n"
        "        if ok:\n"
        "            return assign\n"
        "    return None"
    )
    test_sat = (
        "assert find_lex_first_sat(3, [[1, 2], [-1, 3], [-2, 3], [-3]]) is None\n"
        "assert find_lex_first_sat(3, [[1, 2], [-1, 3], [-2, -3]]) == [False, True, False]\n"
        "assert find_lex_first_sat(2, [[1], [2]]) == [True, True]\n"
    )
    tasks.append({
        "id": "v10-logic-09",
        "pillar": "complex_logic_sat",
        "kind": "python_code",
        "messages": [{
            "role": "user",
            "content": (
                f"Implement `{fn_sat}(num_vars: int, clauses: list[list[int]]) -> list[bool] | None` in Python (`num_vars <= 12`).\n"
                "Each literal `lit` in a clause is a 1-based signed integer (`+k` means `x_k` is `True`, `-k` means `x_k` is `False`).\n"
                "Return the lexicographically smallest satisfying `[x_1, ..., x_num_vars]` (`False < True`), or `None` if unsatisfiable."
            ),
        }],
        "verifier": {"kind": "python_code", "test": test_sat, "forbid_tool_call": True},
    })
    code_checks.append({"id": "chk_sat", "messages": [{"role": "assistant", "content": f"```python\n{ref_sat_code}\n```"}], "verifier": {"kind": "python_code", "test": test_sat}})

    # -----------------------------------------------------------------------
    # Pillar 2: `realtime_research_citations` (9 tasks)
    # -----------------------------------------------------------------------
    # 2.1 & 2.2: Live `realtime_web_research` tool invocation when asked for post-cutoff / external specs
    tasks.append({
        "id": "v10-res-01",
        "pillar": "realtime_research_citations",
        "kind": "tool_pos",
        "messages": [{
            "role": "user",
            "content": (
                "MCP tool registry: `realtime_web_research(query)` and `verify_citations_and_claims(draft_answer, sources_json)`.\n\n"
                "Look up the latest published benchmark numbers and methodology for `CiteCheck citation grounding evaluation` before answering. Do not guess from memory."
            ),
        }],
        "verifier": {
            "tool": "realtime_web_research",
            "args": {},
        },
    })

    tasks.append({
        "id": "v10-res-02",
        "pillar": "realtime_research_citations",
        "kind": "tool_pos",
        "messages": [{
            "role": "user",
            "content": (
                "Registered MCP functions: `realtime_web_research(query)` | `inspect_model_weights(layer_filter)`.\n\n"
                "Search the live web for `DAPO Decoupled Clip Dynamic Sampling Policy Optimization ByteDanceTsinghua` to verify its exact four stabilization techniques."
            ),
        }],
        "verifier": {
            "tool": "realtime_web_research",
            "args": {},
        },
    })

    # 2.3, 2.4, 2.5: Multi-Source Conflict & Erratum Synthesis with Mandatory `[S_i]` Inline Citations
    tasks.append({
        "id": "v10-res-03",
        "pillar": "realtime_research_citations",
        "kind": "citation_grounded",
        "messages": [{
            "role": "user",
            "content": (
                "Synthesize the verified latency and throughput for `Helios-KV v2.4` using ONLY the retrieved sources below. "
                "Prefer primary/newer official errata over stale secondary blog posts, include inline `[S#]` citations, and end with JSON `{\"p99_ms\": <float>, \"qps\": <int>}`:\n\n"
                "[S1] (TechPulse Blog, 2026-01-12): Early preview reported Helios-KV v2.4 p99 latency at 14.8 ms and 42,000 QPS.\n"
                "[S2] (Helios-KV v2.4 Architecture Spec, 2026-03-04): Sustained throughput under NVMe-RDMA is 58,400 QPS.\n"
                "[S3] (Official Helios-KV Security & Benchmark Erratum, 2026-05-19): Corrects the preview p99 latency measurement from 14.8 ms (timer bug in S1) to 19.2 ms; confirms sustained throughput remains 58,400 QPS."
            ),
        }],
        "verifier": {
            "required_citations": ["[S2]", "[S3]"],
            "forbidden_citations": ["[S4]", "[S5]"],
            "valid_citations": ["[S1]", "[S2]", "[S3]"],
            "expected": {"p99_ms": 19.2, "qps": 58400},
            "forbid_tool_call": True,
        },
    })

    tasks.append({
        "id": "v10-res-04",
        "pillar": "realtime_research_citations",
        "kind": "citation_grounded",
        "messages": [{
            "role": "user",
            "content": (
                "Answer using ONLY the sources below with inline `[S#]` citations and conclude with JSON `{\"effective_retention_pct\": <float>, \"sample_size\": <int>}`:\n\n"
                "[S1] (Preprint v1, 2025-11-02): Cohort study (`n = 410`) initially estimated 88.5% 12-month retention.\n"
                "[S2] (Peer-Reviewed Camera-Ready & Audit Log, 2026-04-15): After removing 30 duplicate enrollment records, final deduplicated sample size is `n = 380` with 84.0% 12-month retention.\n"
                "[S3] (Supplementary Appendix Table B, 2026-04-15): Confirms `319 / 380` (`84.0%`) retained participants across 6 clinical sites."
            ),
        }],
        "verifier": {
            "required_citations": ["[S2]", "[S3]"],
            "forbidden_citations": ["[S4]", "[S5]"],
            "valid_citations": ["[S1]", "[S2]", "[S3]"],
            "expected": {"effective_retention_pct": 84.0, "sample_size": 380},
            "forbid_tool_call": True,
        },
    })

    # 2.5: Honest Citation-Grounded Abstention when Requested Metric is Absent!
    tasks.append({
        "id": "v10-res-05",
        "pillar": "realtime_research_citations",
        "kind": "citation_grounded",
        "messages": [{
            "role": "user",
            "content": (
                "Using ONLY the retrieved sources `[S1]` and `[S2]`, report the `FP8` power consumption in Watts of the `Aether-X9` accelerator. "
                "If the retrieved sources do NOT contain `FP8` power consumption in Watts, do NOT guess: cite `[S1]` and `[S2]` to explain what they actually cover and output JSON `{\"status\": \"insufficient_evidence\", \"watts\": null}`:\n\n"
                "[S1] (Aether-X9 Datasheet Rev 1.1): Lists `192 GB` HBM3e capacity and `6.4 TB/s` memory bandwidth.\n"
                "[S2] (Aether-X9 MLPerf Interconnect Note): Reports `900 GB/s` bidirectional optical fabric link speed; thermal and wattage specs deferred to Rev 2.0."
            ),
        }],
        "verifier": {
            "required_citations": ["[S1]", "[S2]"],
            "forbidden_citations": ["[S3]", "[S4]"],
            "valid_citations": ["[S1]", "[S2]"],
            "expected": {"status": "insufficient_evidence", "watts": None},
            "forbid_tool_call": True,
        },
    })

    # 2.6 & 2.7: Claim-Level Citation Auditing (`CiteCheck` Precision/Recall & Unsupported Claim Detection)
    tasks.append({
        "id": "v10-res-06",
        "pillar": "realtime_research_citations",
        "kind": "json_values",
        "messages": [{
            "role": "user",
            "content": (
                "Audit the 3 claims in the draft below against sources `[S1]..[S3]`:\n"
                "- `[S1]`: `Kronos-DB` uses Raft consensus for metadata.\n"
                "- `[S2]`: `Kronos-DB` compresses cold blocks with `Zstandard` at ratio `3.4x`.\n"
                "- `[S3]`: `Kronos-DB` requires Linux kernel `>= 6.1` for `io_uring`.\n\n"
                "Draft:\n"
                "- Claim 1: \"Kronos-DB uses Raft consensus for metadata [S1].\"\n"
                "- Claim 2: \"Kronos-DB compresses cold blocks at ratio 5.2x [S2].\"\n"
                "- Claim 3: \"Kronos-DB requires Linux kernel >= 6.1 [S4].\"\n\n"
                "Classify whether each claim is strictly verified (`c1_ok`, `c2_ok`, `c3_ok`) and count `verified_count`.\n"
                'Return ONLY JSON `{"c1_ok": <bool>, "c2_ok": <bool>, "c3_ok": <bool>, "verified_count": <int>}`.'
            ),
        }],
        "verifier": {
            "expected": {"c1_ok": True, "c2_ok": False, "c3_ok": False, "verified_count": 1},
            "forbid_tool_call": True,
        },
    })

    tasks.append({
        "id": "v10-res-07",
        "pillar": "realtime_research_citations",
        "kind": "json_values",
        "messages": [{
            "role": "user",
            "content": (
                "In `ALCE` citation evaluation, a response makes `5` factual statements, `4` of which carry supported citations (`citation_recall = 4 / 5 = 0.80`). "
                "Across the response, there are `5` total `[S_i]` citation tags emitted, of which `4` genuinely contribute to entailing their sentence (`citation_precision = 4 / 5 = 0.80`).\n"
                "If a second draft instead makes `4` factual statements (all `4` fully entailed by their citations -> `recall = 1.0`), emits `5` total citation tags, and `1` tag is an irrelevant redundant citation so only `4` of `5` tags are necessary (`precision = 0.80`), compute harmonic mean `citation_f1 = round(2 * precision * recall / (precision + recall), 4)` for the second draft.\n"
                'Return ONLY JSON `{"recall": <float>, "precision": <float>, "citation_f1": <float>}`.'
            ),
        }],
        "verifier": {
            "expected": {"recall": 1.0, "precision": 0.8, "citation_f1": round(2 * 0.8 * 1.0 / 1.8, 4)},
            "forbid_tool_call": True,
        },
    })

    # 2.8: Multi-hop Cross-Source Synthesis (`[S1]` + `[S2]` + `[S3]`)
    tasks.append({
        "id": "v10-res-08",
        "pillar": "realtime_research_citations",
        "kind": "citation_grounded",
        "messages": [{
            "role": "user",
            "content": (
                "Compute the annual cluster energy cost in USD for `Orion-Cluster` by combining `[S1]`, `[S2]`, and `[S3]`. "
                "Cite all three sources `[S1]`, `[S2]`, `[S3]` inline in your explanation and finish with JSON `{\"total_kw\": <float>, \"annual_usd\": <float>}`:\n\n"
                "[S1] (Orion Hardware Inventory): Cluster comprises `25` racks drawing `12.0 kW` IT load per rack (`300.0 kW` IT load).\n"
                "[S2] (Facility Cooling Audit): Data center Power Usage Effectiveness (`PUE`) is `1.20` (`total_kw = it_kw * PUE`).\n"
                "[S3] (Utility Contract 2026): Contracted flat rate is `$0.10` per kWh over an `8,000`-hour operating year (`annual_usd = total_kw * 8000 * 0.10`)."
            ),
        }],
        "verifier": {
            "required_citations": ["[S1]", "[S2]", "[S3]"],
            "forbidden_citations": ["[S4]"],
            "valid_citations": ["[S1]", "[S2]", "[S3]"],
            "expected": {"total_kw": 360.0, "annual_usd": 288000.0},
            "forbid_tool_call": True,
        },
    })

    # 2.9: Python Inline Citation Verifier Implementation (`audit_bracket_citations`)
    fn_cite = "audit_bracket_citations"
    ref_cite_code = (
        "import re\n\n"
        "def audit_bracket_citations(answer_text: str, valid_ids: set[str]) -> dict[str, object]:\n"
        "    tags = re.findall(r'\\[(S\\d+)\\]', answer_text)\n"
        "    unknown = sorted({t for t in tags if t not in valid_ids})\n"
        "    used_valid = sorted({t for t in tags if t in valid_ids})\n"
        "    return {\n"
        "        'total_tags': len(tags),\n"
        "        'used_valid': used_valid,\n"
        "        'unknown_tags': unknown,\n"
        "        'is_clean': len(tags) > 0 and len(unknown) == 0,\n"
        "    }"
    )
    test_cite = (
        "res1 = audit_bracket_citations('Latency is 19.2 ms [S3] at 58k QPS [S2][S3].', {'S1', 'S2', 'S3'})\n"
        "assert res1 == {'total_tags': 3, 'used_valid': ['S2', 'S3'], 'unknown_tags': [], 'is_clean': True}\n"
        "res2 = audit_bracket_citations('Claim [S1] and [S9].', {'S1', 'S2'})\n"
        "assert res2['unknown_tags'] == ['S9'] and res2['is_clean'] is False\n"
        "res3 = audit_bracket_citations('Uncited text.', {'S1'})\n"
        "assert res3['is_clean'] is False\n"
    )
    tasks.append({
        "id": "v10-res-09",
        "pillar": "realtime_research_citations",
        "kind": "python_code",
        "messages": [{
            "role": "user",
            "content": (
                f"Implement `{fn_cite}(answer_text: str, valid_ids: set[str]) -> dict[str, object]` in Python.\n"
                "Extract every bracketed source tag matching `r\"\\[(S\\d+)\\]\"` (e.g. `\"[S2]\"` -> `\"S2\"`). Return a dict with exact keys:\n"
                "- `'total_tags'`: total occurrences found (`int`),\n"
                "- `'used_valid'`: sorted unique list of found IDs that belong to `valid_ids`,\n"
                "- `'unknown_tags'`: sorted unique list of found IDs NOT in `valid_ids`,\n"
                "- `'is_clean'`: `True` iff `total_tags > 0` and `len(unknown_tags) == 0`."
            ),
        }],
        "verifier": {"kind": "python_code", "test": test_cite, "forbid_tool_call": True},
    })
    code_checks.append({"id": "chk_cite", "messages": [{"role": "assistant", "content": f"```python\n{ref_cite_code}\n```"}], "verifier": {"kind": "python_code", "test": test_cite}})

    # -----------------------------------------------------------------------
    # Pillar 3: `unseen_ood_self_discover` (9 tasks)
    # -----------------------------------------------------------------------
    # 3.1: Non-mutating circular Z_5 cellular step (`step_z5_ring`)
    fn_ring = "step_z5_ring"
    ref_ring_code = (
        "def step_z5_ring(ring: list[int], rounds: int) -> list[int]:\n"
        "    cur = list(ring)\n"
        "    n = len(cur)\n"
        "    if n == 0:\n"
        "        return []\n"
        "    for _ in range(rounds):\n"
        "        cur = [(cur[(i - 1) % n] + 2 * cur[i] + 3 * cur[(i + 1) % n]) % 5 for i in range(n)]\n"
        "    return cur"
    )
    _r_tmp = [1, 2, 0, 4]
    for _ in range(2):
        _r_tmp = [(_r_tmp[(i - 1) % 4] + 2 * _r_tmp[i] + 3 * _r_tmp[(i + 1) % 4]) % 5 for i in range(4)]
    test_ring = (
        "orig = [1, 2, 0, 4]\n"
        "out = step_z5_ring(orig, 2)\n"
        "assert orig == [1, 2, 0, 4], 'Must not mutate caller list!'\n"
        "assert step_z5_ring([1, 2, 0, 4], 0) == [1, 2, 0, 4]\n"
        f"assert out == {_r_tmp!r}\n"
    )
    tasks.append({
        "id": "v10-ood-01",
        "pillar": "unseen_ood_self_discover",
        "kind": "python_code",
        "messages": [{
            "role": "user",
            "content": (
                f"Implement `{fn_ring}(ring: list[int], rounds: int) -> list[int]` in Python WITHOUT mutating `ring` in place.\n"
                "In each synchronous round on circular length-`n` `ring` (`[] -> []`), index `i` updates simultaneously to:\n"
                "  `next_ring[i] = (ring[(i - 1) % n] + 2 * ring[i] + 3 * ring[(i + 1) % n]) % 5`.\n"
                "Return a new list after `rounds >= 0` rounds (even when `rounds == 0`, return a fresh copy)."
            ),
        }],
        "verifier": {"kind": "python_code", "test": test_ring, "forbid_tool_call": True},
    })
    code_checks.append({"id": "chk_ring", "messages": [{"role": "assistant", "content": f"```python\n{ref_ring_code}\n```"}], "verifier": {"kind": "python_code", "test": test_ring}})

    # 3.2: Numerically stable `log_softmax` + Cross-Entropy loss in pure Python (`stable_log_probs_and_nll`)
    fn_lsm = "stable_log_probs_and_nll"
    ref_lsm_code = (
        "import math\n\n"
        "def stable_log_probs_and_nll(logits: list[float], target_idx: int) -> tuple[list[float], float]:\n"
        "    m = max(logits)\n"
        "    lse = m + math.log(sum(math.exp(x - m) for x in logits))\n"
        "    log_p = [x - lse for x in logits]\n"
        "    return log_p, -log_p[target_idx]"
    )
    test_lsm = (
        "lp, nll = stable_log_probs_and_nll([1000.0, 1000.0], 1)\n"
        "assert abs(lp[0] - (-0.69314718056)) < 1e-7 and abs(nll - 0.69314718056) < 1e-7\n"
        "lp2, nll2 = stable_log_probs_and_nll([-900.0, -902.0], 0)\n"
        "assert lp2[0] < 0.0 and abs(nll2 + lp2[0]) < 1e-12\n"
    )
    tasks.append({
        "id": "v10-ood-02",
        "pillar": "unseen_ood_self_discover",
        "kind": "python_code",
        "messages": [{
            "role": "user",
            "content": (
                f"Implement `{fn_lsm}(logits: list[float], target_idx: int) -> tuple[list[float], float]` in pure Python (`math` only).\n"
                "Compute numerically stable `log_probs` (`log_softmax`, where every entry is `<= 0.0`, NOT `softmax` probabilities!) and scalar negative log-likelihood `-log_probs[target_idx]` without overflow on `[1000.0, 1000.0]` or `[-900.0, -902.0]`."
            ),
        }],
        "verifier": {"kind": "python_code", "test": test_lsm, "forbid_tool_call": True},
    })
    code_checks.append({"id": "chk_lsm", "messages": [{"role": "assistant", "content": f"```python\n{ref_lsm_code}\n```"}], "verifier": {"kind": "python_code", "test": test_lsm}})

    # 3.3: Unseen Bicyclic Monoid `(a, b) * (c, d) = (a + c - min(b, c), b + d - min(b, c))`
    def _bicyclic(p1: tuple[int, int], p2: tuple[int, int]) -> tuple[int, int]:
        a, b = p1
        c, d = p2
        m = min(b, c)
        return (a + c - m, b + d - m)

    u_bc = _bicyclic((2, 5), (3, 1))
    v_bc = _bicyclic((3, 1), (2, 5))
    w_bc = _bicyclic(u_bc, (4, 2))
    tasks.append({
        "id": "v10-ood-03",
        "pillar": "unseen_ood_self_discover",
        "kind": "json_values",
        "messages": [{
            "role": "user",
            "content": (
                "Define the binary operation `⊗` on pairs of non-negative integers `N × N` by:\n"
                "  `(a, b) ⊗ (c, d) = (a + c - min(b, c), b + d - min(b, c))`\n"
                "Compute:\n"
                "1. `u = (2, 5) ⊗ (3, 1)` as `[u0, u1]`\n"
                "2. `v = (3, 1) ⊗ (2, 5)` as `[v0, v1]`\n"
                "3. `w = ((2, 5) ⊗ (3, 1)) ⊗ (4, 2)` as `[w0, w1]`\n"
                "4. `identity_pair`: the unique two-sided identity `[e0, e1]` such that `(a, b) ⊗ (e0, e1) = (a, b)` for all `(a, b)`.\n"
                'Return ONLY JSON `{"u": [<int>, <int>], "v": [<int>, <int>], "w": [<int>, <int>], "identity_pair": [<int>, <int>]}`.'
            ),
        }],
        "verifier": {
            "expected": {"u": list(u_bc), "v": list(v_bc), "w": list(w_bc), "identity_pair": [0, 0]},
            "forbid_tool_call": True,
        },
    })

    # 3.4: Exact Relative Workspace Path Tool Discovery (no `/root/cline/` prefix!)
    tasks.append({
        "id": "v10-ood-04",
        "pillar": "unseen_ood_self_discover",
        "kind": "tool_pos",
        "messages": [{
            "role": "user",
            "content": (
                "Workspace tools provided: `read_file(path)` | `grep_search(query, path)`.\n\n"
                "Inspect the proprietary scheduler formula defined in relative workspace file `lattice_runtime/quota_gate.py` so you can compute `eval_burst_cap(7)`. Use the exact relative path."
            ),
        }],
        "verifier": {
            "tool": "read_file",
            "args": {"path": "lattice_runtime/quota_gate.py"},
        },
    })

    # 3.5: Unseen Type-Lattice Join (`Bot < Int < Num < Any`, `Bot < Str < Any`, `Join(Int, Str) = Any`)
    fn_lat = "infer_lattice_join"
    ref_lat_code = (
        "def infer_lattice_join(t1: str, t2: str) -> str:\n"
        "    anc = {\n"
        "        'Bot': ['Bot', 'Int', 'Num', 'Str', 'Any'],\n"
        "        'Int': ['Int', 'Num', 'Any'],\n"
        "        'Num': ['Num', 'Any'],\n"
        "        'Str': ['Str', 'Any'],\n"
        "        'Any': ['Any'],\n"
        "    }\n"
        "    if t1 == 'Bot':\n"
        "        return t2\n"
        "    if t2 == 'Bot':\n"
        "        return t1\n"
        "    s2 = set(anc[t2])\n"
        "    for cand in anc[t1]:\n"
        "        if cand in s2:\n"
        "            return cand\n"
        "    return 'Any'"
    )
    test_lat = (
        "assert infer_lattice_join('Bot', 'Str') == 'Str'\n"
        "assert infer_lattice_join('Int', 'Num') == 'Num'\n"
        "assert infer_lattice_join('Int', 'Str') == 'Any'\n"
        "assert infer_lattice_join('Int', 'Int') == 'Int'\n"
    )
    tasks.append({
        "id": "v10-ood-05",
        "pillar": "unseen_ood_self_discover",
        "kind": "python_code",
        "messages": [{
            "role": "user",
            "content": (
                f"Implement `{fn_lat}(t1: str, t2: str) -> str` in Python for the 5-element type lattice `{{\"Bot\", \"Int\", \"Num\", \"Str\", \"Any\"}}` ordered by:\n"
                "  - `Bot <= T <= Any` for every type `T`,\n"
                "  - `Int <= Num <= Any`,\n"
                "  - `Str <= Any` (`Str` is incomparable with `Int` and `Num`, so their least upper bound is `Any`).\n"
                "Return the least upper bound (`join`) of `t1` and `t2`."
            ),
        }],
        "verifier": {"kind": "python_code", "test": test_lat, "forbid_tool_call": True},
    })
    code_checks.append({"id": "chk_lat", "messages": [{"role": "assistant", "content": f"```python\n{ref_lat_code}\n```"}], "verifier": {"kind": "python_code", "test": test_lat}})

    # 3.6: Interval Merging with Minimum Gap Coalescing (`coalesce_spans`)
    fn_sp = "coalesce_spans"
    ref_sp_code = (
        "def coalesce_spans(spans: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:\n"
        "    if not spans:\n"
        "        return []\n"
        "    ordered = sorted(spans, key=lambda x: (x[0], x[1]))\n"
        "    merged: list[tuple[int, int]] = [ordered[0]]\n"
        "    for s, e in ordered[1:]:\n"
        "        ps, pe = merged[-1]\n"
        "        if s <= pe + max_gap:\n"
        "            merged[-1] = (ps, max(pe, e))\n"
        "        else:\n"
        "            merged.append((s, e))\n"
        "    return merged"
    )
    test_sp = (
        "inp = [(10, 14), (1, 3), (4, 7)]\n"
        "assert coalesce_spans(inp, 1) == [(1, 7), (10, 14)]\n"
        "assert inp == [(10, 14), (1, 3), (4, 7)]\n"
        "assert coalesce_spans([(1, 10), (3, 5)], 0) == [(1, 10)]\n"
        "assert coalesce_spans([], 2) == []\n"
    )
    tasks.append({
        "id": "v10-ood-06",
        "pillar": "unseen_ood_self_discover",
        "kind": "python_code",
        "messages": [{
            "role": "user",
            "content": (
                f"Implement `{fn_sp}(spans: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]` in Python (`max_gap >= 0`) without mutating `spans`.\n"
                "Sort intervals `(start, end)` (`start <= end`), and merge an interval `(s, e)` into the previous `(ps, pe)` whenever `s <= pe + max_gap`."
            ),
        }],
        "verifier": {"kind": "python_code", "test": test_sp, "forbid_tool_call": True},
    })
    code_checks.append({"id": "chk_sp", "messages": [{"role": "assistant", "content": f"```python\n{ref_sp_code}\n```"}], "verifier": {"kind": "python_code", "test": test_sp}})

    # 3.7: Custom Stack Machine with `DUP`, `SWAP`, `SUB` (`top - second`), `JZ`
    tasks.append({
        "id": "v10-ood-07",
        "pillar": "unseen_ood_self_discover",
        "kind": "json_values",
        "messages": [{
            "role": "user",
            "content": (
                "Trace this custom stack VM (stack grows rightward; `TOP` is the rightmost element):\n"
                "- `PUSH k`: append `k`\n"
                "- `DUP`: append a copy of `TOP`\n"
                "- `SWAP`: swap the top two elements `[..., a, b] -> [..., b, a]`\n"
                "- `SUB`: pop `top` (`b`), pop `second` (`a`), and push `b - a` (NOTE: `top - second`, NOT `second - top`!)\n"
                "- `MUL`: pop top two and push their product\n"
                "Program: `PUSH 4 ; PUSH 9 ; SUB ; DUP ; PUSH 2 ; SWAP ; SUB ; MUL`\n"
                'Return ONLY JSON `{"final_top": <int>, "stack_depth": <int>}`.'
            ),
        }],
        "verifier": {
            # PUSH 4 -> [4]; PUSH 9 -> [4, 9]; SUB (top=9, second=4 -> 9-4=5) -> [5];
            # DUP -> [5, 5]; PUSH 2 -> [5, 5, 2]; SWAP -> [5, 2, 5];
            # SUB (top=5, second=2 -> 5-2=3) -> [5, 3]; MUL -> [15]!
            "expected": {"final_top": 15, "stack_depth": 1},
            "forbid_tool_call": True,
        },
    })

    # 3.8: Unseen Piecewise Recurrence (`f(n)`)
    f_vals = {0: 2, 1: 3}
    for nn_i in range(2, 9):
        if nn_i % 2 == 0:
            f_vals[nn_i] = 2 * f_vals[nn_i - 1] - f_vals[nn_i - 2] + 1
        else:
            f_vals[nn_i] = f_vals[nn_i - 1] + 2 * f_vals[nn_i - 2]
    tasks.append({
        "id": "v10-ood-08",
        "pillar": "unseen_ood_self_discover",
        "kind": "json_values",
        "messages": [{
            "role": "user",
            "content": (
                "Given the parity-switched sequence `g(0) = 2`, `g(1) = 3`, and for `n >= 2`:\n"
                "  - if `n` is even: `g(n) = 2 * g(n - 1) - g(n - 2) + 1`\n"
                "  - if `n` is odd:  `g(n) = g(n - 1) + 2 * g(n - 2)`\n"
                'Compute `g(4)`, `g(6)`, and `g(7)` and return ONLY JSON `{"g4": <int>, "g6": <int>, "g7": <int>}`.'
            ),
        }],
        "verifier": {
            "expected": {"g4": f_vals[4], "g6": f_vals[6], "g7": f_vals[7]},
            "forbid_tool_call": True,
        },
    })

    # 3.9: Discrete KL Divergence `KL(P || Q) = sum_i p_i * log(p_i / q_i)` with zero-support `p_i == 0` convention `0 * log(0) = 0`
    fn_kl = "discrete_kl_div"
    ref_kl_code = (
        "import math\n\n"
        "def discrete_kl_div(p: list[float], q: list[float]) -> float:\n"
        "    total = 0.0\n"
        "    for pi, qi in zip(p, q):\n"
        "        if pi > 0.0:\n"
        "            if qi <= 0.0:\n"
        "                return math.inf\n"
        "            total += pi * math.log(pi / qi)\n"
        "    return total"
    )
    test_kl = (
        "import math\n"
        "assert abs(discrete_kl_div([0.5, 0.5], [0.5, 0.5])) < 1e-12\n"
        "assert abs(discrete_kl_div([0.0, 1.0], [0.25, 0.75]) - (-math.log(0.75))) < 1e-9\n"
        "assert math.isinf(discrete_kl_div([0.5, 0.5], [1.0, 0.0]))\n"
    )
    tasks.append({
        "id": "v10-ood-09",
        "pillar": "unseen_ood_self_discover",
        "kind": "python_code",
        "messages": [{
            "role": "user",
            "content": (
                f"Implement `{fn_kl}(p: list[float], q: list[float]) -> float` in pure Python (`math` only) computing `KL(P || Q) = sum(p_i * ln(p_i / q_i))`:\n"
                "- Use `0 * ln(0 / q_i) = 0.0` whenever `p_i == 0.0`.\n"
                "- Return `math.inf` if `p_i > 0.0` and `q_i <= 0.0`."
            ),
        }],
        "verifier": {"kind": "python_code", "test": test_kl, "forbid_tool_call": True},
    })
    code_checks.append({"id": "chk_kl", "messages": [{"role": "assistant", "content": f"```python\n{ref_kl_code}\n```"}], "verifier": {"kind": "python_code", "test": test_kl}})

    # -----------------------------------------------------------------------
    # Pillar 4: `advanced_rsi_dgm_svf` (9 tasks)
    # -----------------------------------------------------------------------
    # 4.1: `Transformer^2` Prompt-Dispatched `SVF` Expert Mixing (`dispatch_svf_experts`)
    fn_svf = "dispatch_svf_experts"
    ref_svf_code = (
        "import numpy as np\n\n"
        "def dispatch_svf_experts(W: np.ndarray, expert_z: np.ndarray, mix_weights: np.ndarray) -> np.ndarray:\n"
        "    U, S, Vt = np.linalg.svd(W.astype(np.float64), full_matrices=False)\n"
        "    w_norm = mix_weights / np.sum(mix_weights)\n"
        "    z_eff = w_norm @ expert_z\n"
        "    s_mod = S * (1.0 + z_eff)\n"
        "    return ((U * s_mod[None, :]) @ Vt).astype(W.dtype)"
    )
    test_svf = (
        "import numpy as np\n"
        "W0 = np.diag([4.0, 2.0, 1.0])\n"
        "ez = np.array([[0.5, 0.0, 0.0], [0.0, 1.0, 0.0]])\n"
        "mw = np.array([2.0, 2.0])\n"
        "Wn = dispatch_svf_experts(W0, ez, mw)\n"
        "np.testing.assert_allclose(Wn, np.diag([5.0, 3.0, 1.0]), atol=1e-6)\n"
    )
    tasks.append({
        "id": "v10-rsi-01",
        "pillar": "advanced_rsi_dgm_svf",
        "kind": "python_code",
        "messages": [{
            "role": "user",
            "content": (
                f"Implement `{fn_svf}(W: np.ndarray, expert_z: np.ndarray, mix_weights: np.ndarray) -> np.ndarray` in NumPy (`Transformer^2` SVF adaptation):\n"
                "- `W`: `(d_out, d_in)`, `expert_z`: `(E, k)` where `k = min(d_out, d_in)`, `mix_weights`: positive 1D `(E,)`.\n"
                "- Normalize `alpha = mix_weights / sum(mix_weights)`, blend `z_eff = alpha @ expert_z`, decompose `W` by reduced SVD, multiply its singular spectrum elementwise by `(1 + z_eff)`, and reconstruct the adapted matrix in `W.dtype`."
            ),
        }],
        "verifier": {"kind": "python_code", "test": test_svf, "forbid_tool_call": True},
    })
    code_checks.append({"id": "chk_svf", "messages": [{"role": "assistant", "content": f"```python\n{ref_svf_code}\n```"}], "verifier": {"kind": "python_code", "test": test_svf}})

    # 4.2: `Darwin Gödel Machine` (`DGM`) Evolutionary Archive Parent Selection
    # Score `fitness_i + beta * (1 / sqrt(1 + child_count_i))` subject to `anchor_regression <= 0.0`!
    tasks.append({
        "id": "v10-rsi-02",
        "pillar": "advanced_rsi_dgm_svf",
        "kind": "json_values",
        "messages": [{
            "role": "user",
            "content": (
                "In a `Darwin Gödel Machine` (`DGM`) self-improvement archive, only non-regressing nodes (`anchor_nll_delta <= 0.0`) are eligible parents, and each eligible node `i` receives exploration-adjusted score:\n"
                "  `ucb_score(i) = eval_acc(i) + 0.10 / sqrt(1 + children(i))`\n"
                "Evaluate these 4 archive nodes:\n"
                "- `gen_0`: `eval_acc = 0.72`, `children = 8`, `anchor_nll_delta = 0.00`\n"
                "- `gen_1`: `eval_acc = 0.81`, `children = 3`, `anchor_nll_delta = -0.01`\n"
                "- `gen_2`: `eval_acc = 0.88`, `children = 0`, `anchor_nll_delta = +0.04` (regressed anchor!)\n"
                "- `gen_3`: `eval_acc = 0.79`, `children = 0`, `anchor_nll_delta = -0.02`\n"
                'Select the eligible parent maximizing `ucb_score` (rounded to 4 decimals) and return ONLY JSON `{"selected_parent": "<id>", "ucb_score": <float>, "eligible_count": <int>}`.'
            ),
        }],
        "verifier": {
            # gen_0: 0.72 + 0.10/3 = 0.7533; gen_1: 0.81 + 0.10/2 = 0.86; gen_2: ineligible (+0.04 > 0); gen_3: 0.79 + 0.10/1 = 0.89!
            "expected": {"selected_parent": "gen_3", "ucb_score": 0.89, "eligible_count": 3},
            "forbid_tool_call": True,
        },
    })

    # 4.3: `Dr. GRPO` vs Standard `GRPO` Unbiased Advantage & `DAPO` Dynamic Zero-Variance Filter
    tasks.append({
        "id": "v10-rsi-03",
        "pillar": "advanced_rsi_dgm_svf",
        "kind": "json_values",
        "messages": [{
            "role": "user",
            "content": (
                "Compare `Dr. GRPO` + `DAPO` against standard `GRPO` on two rollout groups of size `G = 4`:\n"
                "- Group A rewards: `[1.0, 1.0, 1.0, 1.0]`\n"
                "- Group B rewards: `[1.0, 0.2, 0.2, 0.2]`\n"
                "1. Does `DAPO` Dynamic Sampling skip Group A (`skip_group_a`) because `std(Group A) == 0`?\n"
                "2. Under `Dr. GRPO` (mean-centered WITHOUT dividing by group `std`), compute the exact unnormalized advantage `A_dr = R_1 - mean(Group B)` for the first rollout (`R_1 = 1.0`) in Group B.\n"
                "3. Under standard `GRPO` using population `std = sqrt(mean((R - mean)^2))`, compute `A_std = round((R_1 - mean(Group B)) / std(Group B), 4)`.\n"
                'Return ONLY JSON `{"skip_group_a": <bool>, "A_dr": <float>, "A_std": <float>}`.'
            ),
        }],
        "verifier": {
            # mean(B) = 0.4; R1 - mean = 0.6; var = (0.36 + 3*0.04)/4 = 0.48/4 = 0.12; std = sqrt(0.12) = 0.34641016; 0.6 / std = sqrt(3) = 1.7321
            "expected": {"skip_group_a": True, "A_dr": 0.6, "A_std": round(0.6 / math.sqrt(0.12), 4)},
            "forbid_tool_call": True,
        },
    })

    # 4.4: `LongRoPE2` Dual-Path Continuous Effective Position (`pivot = 32768`)
    tasks.append({
        "id": "v10-rsi-04",
        "pillar": "advanced_rsi_dgm_svf",
        "kind": "json_values",
        "messages": [{
            "role": "user",
            "content": (
                "A `LongRoPE2` dual-path rotary embedding uses `pivot = 32768` and band scale `s_band = 3.5` on low-frequency bands via:\n"
                "  `eff_pos(p) = min(p, 32768) + max(p - 32768, 0) / s_band`\n"
                "Compute:\n"
                "1. `eff_16384 = eff_pos(16384)` (short-context exactness check)\n"
                "2. `eff_32768 = eff_pos(32768)` (`C^0` boundary continuity)\n"
                "3. `eff_262144 = eff_pos(262144)` (`256k` max context effective RoPE position)\n"
                "4. `fits_128k_horizon = (eff_262144 < 128000)`\n"
                'Return ONLY JSON `{"eff_16384": <float>, "eff_32768": <float>, "eff_262144": <float>, "fits_128k_horizon": <bool>}`.'
            ),
        }],
        "verifier": {
            "expected": {
                "eff_16384": 16384.0,
                "eff_32768": 32768.0,
                "eff_262144": 32768.0 + (262144.0 - 32768.0) / 3.5,
                "fits_128k_horizon": True,
            },
            "forbid_tool_call": True,
        },
    })

    # 4.5: `AlphaEdit` Orthogonal Nullspace Weight Update (`alphaedit_nullspace_step`)
    fn_ae = "alphaedit_nullspace_step"
    ref_ae_code = (
        "import numpy as np\n\n"
        "def alphaedit_nullspace_step(W: np.ndarray, K_new: np.ndarray, V_target: np.ndarray, V_anchor: np.ndarray, reg: float = 1e-2) -> np.ndarray:\n"
        "    d_in = W.shape[1]\n"
        "    P_null = np.eye(d_in, dtype=np.float64) - (V_anchor.T @ V_anchor)\n"
        "    R = V_target.astype(np.float64) - (W.astype(np.float64) @ K_new.astype(np.float64))\n"
        "    KP = P_null @ K_new.astype(np.float64)\n"
        "    A = (K_new.astype(np.float64) @ KP.T) @ P_null + reg * np.eye(d_in, dtype=np.float64)\n"
        "    # Project right-hand side onto P_null so update lives strictly in anchor nullspace:\n"
        "    B = (R @ KP.T) @ P_null\n"
        "    delta = np.linalg.solve(A.T, B.T).T @ P_null\n"
        "    return (W.astype(np.float64) + delta).astype(W.dtype)"
    )
    # Let's make the prompt specify the exact clean formula `P = I - V_anchor.T @ V_anchor`, `K_p = P @ K_new`, `Delta = R @ K_p.T @ (K_p @ K_p.T + reg * I)^-1 @ P`!
    ref_ae_code = (
        "import numpy as np\n\n"
        "def alphaedit_nullspace_step(W: np.ndarray, K_new: np.ndarray, V_target: np.ndarray, V_anchor: np.ndarray, reg: float = 1e-2) -> np.ndarray:\n"
        "    d_in = W.shape[1]\n"
        "    P = np.eye(d_in, dtype=np.float64) - (V_anchor.T @ V_anchor)\n"
        "    R = V_target - (W @ K_new)\n"
        "    Kp = P @ K_new\n"
        "    M = (Kp @ Kp.T) + reg * np.eye(d_in, dtype=np.float64)\n"
        "    delta = np.linalg.solve(M.T, (R @ Kp.T).T).T @ P\n"
        "    return (W + delta).astype(W.dtype)"
    )
    test_ae = (
        "import numpy as np\n"
        "W = np.ones((2, 4))\n"
        "V_anc = np.eye(4)[:2, :]  # protects coords 0 and 1\n"
        "K_new = np.array([[1.0, 0.0], [0.5, -1.0], [1.0, 0.0], [0.0, 1.0]])\n"
        "V_tgt = np.array([[5.0, -2.0], [3.0, 4.0]])\n"
        "Wn = alphaedit_nullspace_step(W, K_new, V_tgt, V_anc, reg=1e-7)\n"
        "np.testing.assert_allclose(Wn @ V_anc.T, W @ V_anc.T, atol=1e-6)\n"
        "K_pure = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])\n"
        "Wn2 = alphaedit_nullspace_step(W, K_pure, V_tgt, V_anc, reg=1e-8)\n"
        "np.testing.assert_allclose(Wn2 @ K_pure, V_tgt, atol=1e-5)\n"
    )
    tasks.append({
        "id": "v10-rsi-05",
        "pillar": "advanced_rsi_dgm_svf",
        "kind": "python_code",
        "messages": [{
            "role": "user",
            "content": (
                f"Implement `{fn_ae}(W: np.ndarray, K_new: np.ndarray, V_target: np.ndarray, V_anchor: np.ndarray, reg: float = 1e-2) -> np.ndarray` in NumPy (`AlphaEdit` nullspace weight surgery):\n"
                "- `W`: `(d_out, d_in)`, `K_new`: `(d_in, m)`, `V_target`: `(d_out, m)`, `V_anchor`: `(r, d_in)` with orthonormal rows (`V_anchor @ V_anchor.T = I_r`).\n"
                "- Form projector `P = I - V_anchor.T @ V_anchor`, residual `R = V_target - W @ K_new`, projected keys `Kp = P @ K_new`, and `M = Kp @ Kp.T + reg * I`.\n"
                "- Compute `Delta = R @ Kp.T @ M^-1 @ P` via `np.linalg.solve` (so `Delta @ V_anchor.T == 0` identically!) and return `W + Delta` in `W.dtype`."
            ),
        }],
        "verifier": {"kind": "python_code", "test": test_ae, "forbid_tool_call": True},
    })
    code_checks.append({"id": "chk_ae", "messages": [{"role": "assistant", "content": f"```python\n{ref_ae_code}\n```"}], "verifier": {"kind": "python_code", "test": test_ae}})

    # 4.6: Live MCP `seal_outer_loop_self_rewrite` Tool Call when User Requests Verifiable Self-Modification
    tasks.append({
        "id": "v10-rsi-06",
        "pillar": "advanced_rsi_dgm_svf",
        "kind": "tool_pos",
        "messages": [{
            "role": "user",
            "content": (
                "RSI MCP functions: `seal_outer_loop_self_rewrite(task_prompt, target_solution, python_unit_test, steps)` | `inspect_model_weights(layer_filter)`.\n\n"
                "Run an outer-loop SEAL nullspace self-rewrite to internalize `def cube_minus_two(x): return x**3 - 2` verified against `assert cube_minus_two(3) == 25`."
            ),
        }],
        "verifier": {
            "tool": "seal_outer_loop_self_rewrite",
            "args": {},
        },
    })

    # 4.7: `Geiping et al.` Recurrent Depth Parameter Efficiency vs Unrolled Unique Parameters
    tasks.append({
        "id": "v10-rsi-07",
        "pillar": "advanced_rsi_dgm_svf",
        "kind": "json_values",
        "messages": [{
            "role": "user",
            "content": (
                "A shared-weight `LatentReasoningRecurrence` block has `d_model = 2048` and `d_latent = 2560` with `4` bias-free linear projections (`w_gate`, `w_up`, `w_anc`, `w_down`, each `2048 × 2560`), ignoring 1D norm vectors.\n"
                "1. Compute exact unique matrix parameters in millions rounded to 4 decimals (`unique_millions = round(4 * 2048 * 2560 / 1e6, 4)`).\n"
                "2. When looped for `T = 2` recurrent steps with zero-initialized `loop_scales = 0`, what is the unrolled effective matrix parameter equivalent (`effective_millions = round(2 * unique_millions, 4)`) and `step0_logit_drift`?\n"
                'Return ONLY JSON `{"unique_millions": <float>, "effective_millions": <float>, "step0_logit_drift": <float>}`.'
            ),
        }],
        "verifier": {
            "expected": {
                "unique_millions": round(4 * 2048 * 2560 / 1e6, 4),
                "effective_millions": round(2 * round(4 * 2048 * 2560 / 1e6, 4), 4),
                "step0_logit_drift": 0.0,
            },
            "forbid_tool_call": True,
        },
    })

    # 4.8: Non-Mutating `DARE` + Rescaling (`dare_drop_and_rescale`)
    fn_dare = "dare_drop_and_rescale"
    ref_dare_code = (
        "import numpy as np\n\n"
        "def dare_drop_and_rescale(delta_w: np.ndarray, keep_mask: np.ndarray, drop_prob: float) -> np.ndarray:\n"
        "    if drop_prob >= 1.0:\n"
        "        return np.zeros_like(delta_w)\n"
        "    scale = 1.0 / (1.0 - drop_prob)\n"
        "    return (delta_w.copy() * keep_mask * scale).astype(delta_w.dtype)"
    )
    test_dare = (
        "import numpy as np\n"
        "dw = np.array([[2.0, -4.0], [6.0, 8.0]])\n"
        "msk = np.array([[1.0, 0.0], [0.0, 1.0]])\n"
        "out = dare_drop_and_rescale(dw, msk, 0.5)\n"
        "np.testing.assert_allclose(dw, np.array([[2.0, -4.0], [6.0, 8.0]]))\n"
        "np.testing.assert_allclose(out, np.array([[4.0, 0.0], [0.0, 16.0]]))\n"
    )
    tasks.append({
        "id": "v10-rsi-08",
        "pillar": "advanced_rsi_dgm_svf",
        "kind": "python_code",
        "messages": [{
            "role": "user",
            "content": (
                f"Implement `{fn_dare}(delta_w: np.ndarray, keep_mask: np.ndarray, drop_prob: float) -> np.ndarray` in NumPy (`DARE` adapter sparsification) without mutating `delta_w`:\n"
                "- `keep_mask` has binary `{0, 1}` entries matching `delta_w.shape`, and `0.0 <= drop_prob <= 1.0`.\n"
                "- If `drop_prob >= 1.0`, return `np.zeros_like(delta_w)`; otherwise return `(delta_w * keep_mask) / (1.0 - drop_prob)` in `delta_w.dtype`."
            ),
        }],
        "verifier": {"kind": "python_code", "test": test_dare, "forbid_tool_call": True},
    })
    code_checks.append({"id": "chk_dare", "messages": [{"role": "assistant", "content": f"```python\n{ref_dare_code}\n```"}], "verifier": {"kind": "python_code", "test": test_dare}})

    # 4.9: Pure `bfloat16` (`16-bit`) vs `int8` Memory & Precision Verification
    tasks.append({
        "id": "v10-rsi-09",
        "pillar": "advanced_rsi_dgm_svf",
        "kind": "json_values",
        "messages": [{
            "role": "user",
            "content": (
                "A pure `16-bit` (`bfloat16`) language-vision model has `3,178,000,000` (`3.178B`) parameters (`2` bytes per parameter, `1` sign bit + `8` exponent bits + `7` explicit fraction bits).\n"
                "1. Compute exact weight footprint in decimal gigabytes `gb_decimal = round(3178000000 * 2 / 1e9, 4)`.\n"
                "2. Compute exact `bfloat16` ULP spacing on `[2.0, 4.0)` (`ulp_at_2 = 2^1 * 2^-7`).\n"
                "3. Compute the smallest float strictly greater than `2.0` representable in `bfloat16` (`next_after_2`).\n"
                'Return ONLY JSON `{"gb_decimal": <float>, "ulp_at_2": <float>, "next_after_2": <float>}`.'
            ),
        }],
        "verifier": {
            "expected": {"gb_decimal": 6.356, "ulp_at_2": 0.015625, "next_after_2": 2.015625},
            "forbid_tool_call": True,
        },
    })

    assert len(tasks) == 36, f"Expected 36 tasks, got {len(tasks)}"
    return tasks, code_checks


# ===========================================================================
# 2. Held-Out Ultra-Long-Context Multi-Hop Needle + Citation Suite (`16k..64k`)
# ===========================================================================


def _build_multi_needle_archive_prompt(
    rng: random.Random,
    *,
    target_tokens: int,
    project_name: str,
    subsystem_code: str,
    distractor_code: str,
    raw_base: int,
    stale_mult: int,
    true_mult: int,
    offset_add: int,
    doc_prefix: str = "DOC",
) -> tuple[str, list[str], str, int]:
    """Construct a realistic multi-document archive of `~target_tokens` tokens containing:
      - Hop 1 (`doc_h1` near 12% depth): identifies active subsystem `subsystem_code` for `project_name`.
      - Distractor (`doc_stale` near 35% depth): gives an outdated multiplier `stale_mult` (later revoked!).
      - Hop 2 (`doc_h2` near 58% depth): gives the raw baseline `raw_base` for `subsystem_code`.
      - Hop 3 / Superseding Erratum (`doc_h3` near 86% depth): revokes `doc_stale` and establishes authoritative
        multiplier `true_mult` and additive calibration `+ offset_add` (`final_metric = raw_base * true_mult + offset_add`).
    """
    # ~85 tokens per filler document -> choose num_docs to match `target_tokens`!
    num_docs = max(40, int(round(target_tokens / 82.0)))
    idx_h1 = max(2, int(num_docs * 0.12))
    idx_stale = max(idx_h1 + 3, int(num_docs * 0.36))
    idx_h2 = max(idx_stale + 3, int(num_docs * 0.59))
    idx_h3 = min(num_docs - 3, max(idx_h2 + 3, int(num_docs * 0.86)))

    domains = [
        "telemetry ring-buffer checkpointing",
        "RDMA congestion window pacing",
        "zero-copy page-cache eviction",
        "optical transceiver thermal telemetry",
        "quorum lease heartbeat scheduling",
        "columnar parquet bloom-filter indexing",
        "vectorized SIMD checksum verification",
        "distributed WAL segment compaction",
    ]
    modules = ["zephyr-io", "kestrel-rpc", "talon-fs", "merlin-sched", "corvus-net", "albatross-gc", "falcon-wal", "osprey-kv"]

    docs: list[str] = []
    id_h1 = f"{doc_prefix}-{idx_h1:04d}"
    id_stale = f"{doc_prefix}-{idx_stale:04d}"
    id_h2 = f"{doc_prefix}-{idx_h2:04d}"
    id_h3 = f"{doc_prefix}-{idx_h3:04d}"

    for d_i in range(1, num_docs + 1):
        did = f"[{doc_prefix}-{d_i:04d}]"
        if d_i == idx_h1:
            docs.append(
                f"{did} (Architecture Routing Registry — Official): For program `{project_name}`, the active production "
                f"compute kernel is codenamed `{subsystem_code}` (legacy prototype `{distractor_code}` was decommissioned "
                f"prior to cutover and must never be used for production capacity calculations)."
            )
        elif d_i == idx_stale:
            docs.append(
                f"{did} (Preliminary Draft Note — Early Bench): Provisional scaling factor for `{subsystem_code}` was "
                f"tentatively logged as `{stale_mult}x` with `+ 0` offset pending hardware burn-in verification."
            )
        elif d_i == idx_h2:
            docs.append(
                f"{did} (Verified Hardware Telemetry Log): Validated baseline unit throughput for active kernel "
                f"`{subsystem_code}` is `{raw_base}` units/sec (whereas retired prototype `{distractor_code}` measured `999` units/sec)."
            )
        elif d_i == idx_h3:
            docs.append(
                f"{did} (Mandatory Calibration Erratum — Supersedes {id_stale}): Formal audit REVOKES `{id_stale}` (`{stale_mult}x` was "
                f"invalidated). The authoritative calibration formula for `{subsystem_code}` is `effective_throughput = baseline * {true_mult} + {offset_add}`."
            )
        else:
            dom = domains[(d_i * 7) % len(domains)]
            mod = modules[(d_i * 5) % len(modules)]
            v_a = 100 + ((d_i * 37) % 850)
            v_b = 10 + ((d_i * 19) % 85)
            docs.append(
                f"{did} (Subsystem Maintenance Log — `{mod}`): Routine audit of {dom} across cluster partition `P-{d_i % 16:02d}` "
                f"recorded queue watermark `{v_a}` frames, tail jitter `{v_b}` microseconds, zero CRC framing faults, and nominal "
                f"voltage rails across redundant power shelves during continuous 24-hour soak testing."
            )

    final_val = raw_base * true_mult + offset_add
    header = (
        f"You are auditing a `{num_docs}`-document engineering archive (`[{doc_prefix}-0001]` .. `[{doc_prefix}-{num_docs:04d}]`).\n"
        f"Task: Determine the authoritative `effective_throughput` for program `{project_name}` by:\n"
        f"1. Finding the active production kernel codename for `{project_name}`.\n"
        f"2. Finding that kernel's verified baseline throughput.\n"
        f"3. Applying ONLY the latest authoritative calibration formula (ignoring decommissioned prototypes and any revoked draft notes!).\n"
        f"Cite the 3 authoritative document IDs inline (e.g. `[{doc_prefix}-XXXX]`) — do NOT cite the revoked draft note as authoritative — and end with JSON:\n"
        f'`{{"kernel": "<str>", "effective_throughput": <int>, "authoritative_docs": ["{doc_prefix}-XXXX", "{doc_prefix}-YYYY", "{doc_prefix}-ZZZZ"]}}`\n\n'
        "=== BEGIN ARCHIVE ===\n"
    )
    footer = (
        f"\n=== END ARCHIVE ===\n"
        f"Now synthesize the verified answer for `{project_name}` with inline `[{doc_prefix}-XXXX]` citations and the final JSON object."
    )
    return header + "\n".join(docs) + footer, [f"[{id_h1}]", f"[{id_h2}]", f"[{id_h3}]"], f"[{id_stale}]", final_val


def build_heldout_long_256k_needles_12() -> list[dict[str, Any]]:
    """12 multi-hop needle + erratum + citation tasks spanning `16k`, `32k`, `48k`, and `64k` tokens."""
    rng = random.Random(20269911)
    specs = [
        # (target_tokens, project_name, subsystem_code, distractor_code, raw_base, stale_mult, true_mult, offset_add)
        (16384, "Borealis-Grid", "KRN-Vesper-7", "KRN-Old-1", 42, 3, 5, 18),
        (16384, "Calypso-Mesh", "KRN-Solstice-4", "KRN-Beta-0", 35, 2, 6, 14),
        (16384, "Draco-Fabric", "KRN-Perihelion-9", "KRN-Draft-2", 28, 4, 7, 22),
        (32768, "Erebus-Vault", "KRN-Apogee-3", "KRN-Legacy-8", 54, 2, 4, 31),
        (32768, "Fornax-Stream", "KRN-Meridian-6", "KRN-Alpha-5", 63, 5, 3, 27),
        (32768, "Galatea-Core", "KRN-Ecliptic-8", "KRN-Mock-3", 47, 6, 5, 19),
        (49152, "Hyperion-Bus", "KRN-umbra-2", "KRN-Zero-9", 71, 3, 4, 16),
        (49152, "Iapetus-Node", "KRN-Zenithal-5", "KRN-Pre-4", 38, 7, 6, 25),
        (49152, "Janus-Ledger", "KRN-Parallax-1", "KRN-Temp-7", 52, 2, 5, 33),
        (65536, "Kraken-Array", "KRN-Syzygy-9", "KRN-Stub-6", 64, 3, 5, 29),
        (65536, "Lyra-Switch", "KRN-Azimuth-7", "KRN-Early-2", 45, 8, 4, 37),
        (65536, "Mimas-Router", "KRN-Equinox-3", "KRN-Test-1", 59, 2, 6, 21),
    ]
    rows: list[dict[str, Any]] = []
    for idx, (tok_t, proj, sub_c, dist_c, r_b, st_m, tr_m, off_a) in enumerate(specs, start=1):
        prompt, req_cites, stale_cite, final_v = _build_multi_needle_archive_prompt(
            rng,
            target_tokens=tok_t,
            project_name=proj,
            subsystem_code=sub_c,
            distractor_code=dist_c,
            raw_base=r_b,
            stale_mult=st_m,
            true_mult=tr_m,
            offset_add=off_a,
            doc_prefix="DOC",
        )
        auth_plain = [c.strip("[]") for c in req_cites]
        rows.append({
            "id": f"v10-long-{idx:02d}-{tok_t // 1024}k",
            "pillar": f"long_context_{tok_t // 1024}k",
            "target_context_tokens": tok_t,
            "kind": "citation_grounded",
            "messages": [{"role": "user", "content": prompt}],
            "verifier": {
                "required_citations": req_cites,
                "forbidden_citations": [],
                "expected": {
                    "kernel": sub_c,
                    "effective_throughput": final_v,
                    "authoritative_docs": auth_plain,
                },
                "forbid_tool_call": True,
            },
            "meta": {"revoked_doc": stale_cite, "stale_value": r_b * st_m},
        })
    return rows


# ===========================================================================
# 3. Training Generators (`Self-Discover Logic`, `Search-R1/CiteCheck Research`,
#    `Zero-Prior & Root-Cause Code`, `DGM/SVF/AlphaEdit RSI`, `Long-Context SFT`)
# ===========================================================================


def make_v10_self_discover_logic_item(rng: random.Random, idx: int) -> dict[str, Any]:
    """Teach `[SELF-DISCOVER] -> [PROPAGATE] -> [VERIFY]` on SAT, Knights, Constraint Grids, Quantifiers & Game Trees."""
    variant = idx % 6

    if variant == 0:
        # 3-variable 3-CNF unique satisfying assignment
        target = (rng.choice([False, True]), rng.choice([False, True]), rng.choice([False, True]))
        # Build 5 clauses satisfied ONLY by `target`!
        clauses_str: list[str] = []
        all_other = [t for t in itertools.product([False, True], repeat=3) if t != target]
        rng.shuffle(all_other)
        for wrong in all_other[:5]:
            # Clause falsified by `wrong` (hence satisfied by `target` since `target != wrong`!)
            lits = [f"p{j+1}" if not wrong[j] else f"NOT p{j+1}" for j in range(3)]
            clauses_str.append("(" + " OR ".join(lits) + ")")
        # Add 2 unit/binary clauses from remaining wrong assignments so uniqueness is guaranteed!
        for wrong in all_other[5:]:
            lits = [f"p{j+1}" if not wrong[j] else f"NOT p{j+1}" for j in range(3)]
            clauses_str.append("(" + " OR ".join(lits) + ")")
        n_true = sum(int(x) for x in target)
        c_block = "\n".join(f"  C{k+1}: `{cl}`" for k, cl in enumerate(clauses_str))
        prompt = (
            f"Find the unique Boolean assignment `(p1, p2, p3)` satisfying all {len(clauses_str)} clauses:\n{c_block}\n"
            'Return ONLY JSON `{"p1": <bool>, "p2": <bool>, "p3": <bool>, "num_true": <int>}`.'
        )
        think = (
            "[SELF-DISCOVER] Select modules: (1) Clause Elimination, (2) Full Assignment Verification.\n"
            f"[SOLVE] Each 3-literal maxterm rules out one falsifying triple; testing candidate `p = ({target[0]}, {target[1]}, {target[2]})` satisfies all {len(clauses_str)} clauses.\n"
            f"[VERIFY] Count of `True` literals is `{n_true}`; every other triple falsifies one of `C1..C{len(clauses_str)}`."
        )
        ans = json.dumps({"p1": target[0], "p2": target[1], "p3": target[2], "num_true": n_true})
        return {
            "id": f"v10_logic_{idx:04d}",
            "pillar": "complex_logic_sat",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}</think>\n{ans}"}],
            "verifier": {"expected": {"p1": target[0], "p2": target[1], "p3": target[2], "num_true": n_true}, "forbid_tool_call": True},
        }

    if variant == 1:
        # 3-Agent Knights & Knaves Biconditional Deduction
        names = rng.choice([("Dax", "Elin", "Faye"), ("Gael", "Hana", "Ivo"), ("Juno", "Kael", "Lira"), ("Milo", "Nia", "Orin")])
        n1, n2, n3 = names
        mode = (idx // 6) % 3
        if mode == 0:
            # n1: "n2 is a Knave", n2: "n1 and n3 are different types", n3: "n1 is a Knave"
            # Let's solve programmatically!
            sols = [
                (a, b, c)
                for a, b, c in itertools.product([False, True], repeat=3)
                if (a == (not b)) and (b == (a != c)) and (c == (not a))
            ]
            s1_txt = f"`{n2}` is a Knave."
            s2_txt = f"`{n1}` and `{n3}` are of different types."
            s3_txt = f"`{n1}` is a Knave."
        elif mode == 1:
            sols = [
                (a, b, c)
                for a, b, c in itertools.product([False, True], repeat=3)
                if (a == (b and c)) and (b == a) and (c == (not b))
            ]
            s1_txt = f"Both `{n2}` and `{n3}` are Knights."
            s2_txt = f"`{n1}` is a Knight."
            s3_txt = f"`{n2}` is a Knave."
        else:
            sols = [
                (a, b, c)
                for a, b, c in itertools.product([False, True], repeat=3)
                if (a == (not c)) and (b == (a and c)) and (c == b)
            ]
            s1_txt = f"`{n3}` is a Knave."
            s2_txt = f"Both `{n1}` and `{n3}` are Knights."
            s3_txt = f"`{n2}` is a Knight."
        assert len(sols) == 1
        a_v, b_v, c_v = sols[0]
        k_cnt = int(a_v) + int(b_v) + int(c_v)
        prompt = (
            f"Each of `{n1}`, `{n2}`, `{n3}` is either a Knight (always truthful) or a Knave (always lies):\n"
            f"- `{n1}` says: \"{s1_txt}\"\n"
            f"- `{n2}` says: \"{s2_txt}\"\n"
            f"- `{n3}` says: \"{s3_txt}\"\n"
            f'Return ONLY JSON `{{"{n1}": <bool>, "{n2}": <bool>, "{n3}": <bool>, "knights": <int>}}`.'
        )
        think = (
            f"[SELF-DISCOVER] Translate each claim into a biconditional `Speaker <-> Claim` and case-split `{n1} in {{True, False}}`.\n"
            f"[SOLVE] Testing `{n1}={a_v}` forces `{n2}={b_v}` and `{n3}={c_v}`; the opposite branch yields an immediate biconditional contradiction.\n"
            f"[VERIFY] Re-evaluating all 3 statements under `({a_v}, {b_v}, {c_v})` matches every speaker's truth value (`knights = {k_cnt}`)."
        )
        exp = {n1: a_v, n2: b_v, n3: c_v, "knights": k_cnt}
        return {
            "id": f"v10_logic_{idx:04d}",
            "pillar": "complex_logic_sat",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}</think>\n{json.dumps(exp)}"}],
            "verifier": {"expected": exp, "forbid_tool_call": True},
        }

    if variant == 2:
        # 4-Position Linear Ordering / Zebra Grid
        items = list(rng.choice([
            ["Cobalt", "Amber", "Crimson", "Teal"],
            ["Quartz", "Basalt", "Granite", "Shale"],
            ["Lyric", "Sonnet", "Ballad", "Haiku"],
        ]))
        rng.shuffle(items)
        p1, p2, p3, p4 = items  # positions 1, 2, 3, 4
        prompt = (
            f"Four artifacts (`{items[0]}`, `{items[1]}`, `{items[2]}`, `{items[3]}` — listed here in arbitrary order: `{sorted(items)}`) occupy pedestals `1, 2, 3, 4` from left to right:\n"
            f"1. `{p2}` is immediately to the left of `{p3}`.\n"
            f"2. `{p1}` is somewhere to the left of `{p2}`.\n"
            f"3. `{p4}` is at pedestal `4`.\n"
            f'Deduce all 4 positions and return ONLY JSON `{{"pos_1": "<str>", "pos_2": "<str>", "pos_3": "<str>", "pos_4": "<str>"}}`.'
        )
        # Wait: let's fix the prompt wording so `items` isn't leaked in order!
        alpha = sorted(items)
        prompt = (
            f"Four artifacts (`{alpha[0]}`, `{alpha[1]}`, `{alpha[2]}`, `{alpha[3]}`) occupy pedestals `1, 2, 3, 4` from left to right:\n"
            f"1. `{p4}` is on pedestal `4`.\n"
            f"2. `{p2}` is immediately to the left of `{p3}`.\n"
            f"3. `{p1}` is strictly to the left of `{p2}`.\n"
            f'Deduce the unique order and return ONLY JSON `{{"pos_1": "<str>", "pos_2": "<str>", "pos_3": "<str>", "pos_4": "<str>"}}`.'
        )
        think = (
            f"[SELF-DISCOVER] Propagate fixed slots first, then contiguous pairs `(x, x+1)`.\n"
            f"[SOLVE] Pedestal `4` is `{p4}`, leaving `{1, 2, 3}` for `{p1}, {p2}, {p3}`. Since `{p1} < pos({p2})` and `pos({p3}) = pos({p2}) + 1`, the only fit in `{1, 2, 3}` is `1={p1}, 2={p2}, 3={p3}`.\n"
            "[VERIFY] All 3 clues hold simultaneously."
        )
        exp = {"pos_1": p1, "pos_2": p2, "pos_3": p3, "pos_4": p4}
        return {
            "id": f"v10_logic_{idx:04d}",
            "pillar": "complex_logic_sat",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}</think>\n{json.dumps(exp)}"}],
            "verifier": {"expected": exp, "forbid_tool_call": True},
        }

    if variant == 3:
        # Subtraction Game Backward Induction (`moves = {1, 2, k}`, `N` pile)
        k_mv = rng.choice([4, 5])
        moves = (1, 2, k_mv)
        n_pile = rng.randint(11, 16)
        win = [False] * (n_pile + 1)
        for s in range(1, n_pile + 1):
            win[s] = any(s - m >= 0 and not win[s - m] for m in moves)
        p_cnt = sum(1 for s in range(1, n_pile + 1) if not win[s])
        best_m = min((m for m in moves if n_pile - m >= 0 and not win[n_pile - m]), default=-1)
        prompt = (
            f"In a two-player normal-play subtraction game (last move wins), allowed moves remove `1`, `2`, or `{k_mv}` chips from a pile of `N = {n_pile}` chips.\n"
            f"Compute: (1) `first_wins` (`bool`), (2) smallest winning move `min_winning_move` (`-1` if `first_wins` is `false`), and (3) `p_positions_count` among `1..{n_pile}`.\n"
            'Return ONLY JSON `{"first_wins": <bool>, "min_winning_move": <int>, "p_positions_count": <int>}`.'
        )
        p_list = [s for s in range(0, n_pile + 1) if not win[s]]
        think = (
            f"[SELF-DISCOVER] Backward induction from `0` (`P`-position): state `s` is `P` iff every `s - m` (`m in {{1, 2, {k_mv}}}`) is `N`.\n"
            f"[SOLVE] `P`-positions in `0..{n_pile}` are `{p_list}` -> `{p_cnt}` losing states in `1..{n_pile}`.\n"
            f"[VERIFY] At `N = {n_pile}`, `first_wins = {win[n_pile]}` and `min_winning_move = {best_m}`."
        )
        exp = {"first_wins": win[n_pile], "min_winning_move": best_m, "p_positions_count": p_cnt}
        return {
            "id": f"v10_logic_{idx:04d}",
            "pillar": "complex_logic_sat",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}</think>\n{json.dumps(exp)}"}],
            "verifier": {"expected": exp, "forbid_tool_call": True},
        }

    if variant == 4:
        # Quantified Set/Modular Logic Counterexample & Truth Evaluation
        mod_k = rng.choice([4, 5, 6])
        rem_r = rng.randint(1, mod_k - 1)
        dom = list(range(1, mod_k + 1))
        rel = lambda x, y: (x * y) % mod_k == rem_r
        fa_ex = all(any(rel(x, y) for y in dom) for x in dom)
        ex_fa = any(all(rel(x, y) for x in dom) for y in dom)
        tot = sum(1 for x in dom for y in dom if rel(x, y))
        prompt = (
            f"Over finite domain `D = {{1, ..., {mod_k}}}`, let `P(x, y)` hold iff `(x * y) % {mod_k} == {rem_r}`.\n"
            f"Evaluate `forall_x_exists_y` (`∀x ∈ D ∃y ∈ D, P(x, y)`), `exists_y_forall_x` (`∃y ∈ D ∀x ∈ D, P(x, y)`), and `satisfying_pairs` (`|(x, y) ∈ D×D : P(x, y)|`).\n"
            'Return ONLY JSON `{"forall_x_exists_y": <bool>, "exists_y_forall_x": <bool>, "satisfying_pairs": <int>}`.'
        )
        think = (
            f"[SELF-DISCOVER] Build row/column truth counts over `D = 1..{mod_k}` (`x = {mod_k}` gives `(x*y) % {mod_k} == 0 != {rem_r}`!).\n"
            f"[SOLVE] Row `x = {mod_k}` has `0` true entries so `forall_x_exists_y = {fa_ex}` and `exists_y_forall_x = {ex_fa}`; counting all pairs gives `{tot}`."
        )
        exp = {"forall_x_exists_y": fa_ex, "exists_y_forall_x": ex_fa, "satisfying_pairs": tot}
        return {
            "id": f"v10_logic_{idx:04d}",
            "pillar": "complex_logic_sat",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}</think>\n{json.dumps(exp)}"}],
            "verifier": {"expected": exp, "forbid_tool_call": True},
        }

    # variant == 5: Python Exact Constraint-Backtracking k-Colorability (`is_graph_k_colorable`)
    base_fn = rng.choice(["can_k_color_graph", "valid_vertex_k_coloring", "graph_admits_k_colors"])
    fn = f"{base_fn}_{idx % 11}" if idx >= 18 else base_fn
    prompt = (
        f"Implement `{fn}(num_nodes: int, edges: list[tuple[int, int]], k: int) -> bool` in Python (`0`-indexed vertices `0..num_nodes-1`, `num_nodes <= 12`).\n"
        "Return `True` iff the undirected graph admits a proper `k`-coloring where no edge connects two vertices of the same color (if a self-loop `(u, u)` exists, return `False`)."
    )
    think = (
        "[SELF-DISCOVER] Check self-loops upfront (`u == v -> False`), build adjacency sets, and run backtracking coloring over `0..num_nodes-1` with symmetry breaking (`node 0` fixed to color `0` when `k >= 1`)."
    )
    code = (
        f"def {fn}(num_nodes: int, edges: list[tuple[int, int]], k: int) -> bool:\n"
        "    if num_nodes == 0:\n"
        "        return True\n"
        "    if k <= 0:\n"
        "        return False\n"
        "    adj = [set() for _ in range(num_nodes)]\n"
        "    for u, v in edges:\n"
        "        if u == v:\n"
        "            return False\n"
        "        adj[u].add(v)\n"
        "        adj[v].add(u)\n"
        "    colors = [-1] * num_nodes\n\n"
        "    def backtrack(idx: int) -> bool:\n"
        "        if idx == num_nodes:\n"
        "            return True\n"
        "        used = {colors[nb] for nb in adj[idx] if colors[nb] != -1}\n"
        "        limit = 1 if idx == 0 else k\n"
        "        for c in range(limit):\n"
        "            if c not in used:\n"
        "                colors[idx] = c\n"
        "                if backtrack(idx + 1):\n"
        "                    return True\n"
        "                colors[idx] = -1\n"
        "        return False\n\n"
        "    return backtrack(0)"
    )
    test = (
        f"assert {fn}(4, [(0,1),(1,2),(2,3),(3,0)], 2) is True\n"
        f"assert {fn}(3, [(0,1),(1,2),(2,0)], 2) is False\n"
        f"assert {fn}(4, [(0,1),(0,2),(0,3),(1,2),(2,3),(1,3)], 3) is False\n"
        f"assert {fn}(4, [(0,1),(0,2),(0,3),(1,2),(2,3),(1,3)], 4) is True\n"
    )
    return {
        "id": f"v10_logic_{idx:04d}",
        "pillar": "complex_logic_sat",
        "images": [],
        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}</think>\n```python\n{code}\n```"}],
        "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
    }


def make_v10_realtime_research_citation_item(rng: random.Random, idx: int) -> dict[str, Any]:
    """Teach `Search-R1` + `WebThinker` + `CiteCheck`: multi-turn `realtime_web_research`, conflict resolution, & strict `[S_i]` citations."""
    variant = idx % 5

    if variant == 0:
        # Multi-turn WebThinker flow: Query 1 -> sees conflict between [S1] blog and [S2] release note -> Query 2 erratum -> Verified [S2][S3] synthesis!
        sys_names = ["NovaKV-3", "PulsarFS-5", "StratosRPC-2", "VanguardSQL-4", "AegisCache-8"]
        sname = sys_names[idx % len(sys_names)] + f".{idx % 7}"
        stale_v = rng.randint(110, 150)
        true_v = stale_v + rng.choice([24, 32, 45])
        bw_gb = rng.choice([48.5, 62.0, 74.5])
        u_msg = (
            "Available tools:\n"
            "- `realtime_web_research(query: str, max_results: int = 5)`\n"
            "- `verify_citations_and_claims(draft_answer: str, sources_json: str)`\n\n"
            f"Research the verified sustained write IOPS (in thousands) and replication bandwidth (`GB/s`) of `{sname}`, resolving any discrepancies between preview posts and official release errata. Cite `[S#]` inline."
        )
        t1_think = f"[SEARCH-PLAN] I have no prior verified release data for `{sname}`. Call `realtime_web_research` first rather than guessing."
        t1_call = _xml_call("realtime_web_research", {"query": f"{sname} sustained write IOPS replication bandwidth release errata", "max_results": 4})
        tool_resp_1 = (
            "<tool_response>\n"
            f"[S1] (DevDigest Preview, 2026-01): Early beta of {sname} hit {stale_v}k write IOPS.\n"
            f"[S2] ({sname} GA Specification, 2026-04): Sustained replication bandwidth is {bw_gb} GB/s; see GA Erratum #14 for corrected write IOPS.\n"
            "</tool_response>"
        )
        t2_think = f"[SEARCH-VERIFY] `[S1]` is an early beta (`{stale_v}k`) while `[S2]` references `GA Erratum #14` for corrected write IOPS. Issue a targeted follow-up search for `{sname} GA Erratum 14 write IOPS`."
        t2_call = _xml_call("realtime_web_research", {"query": f"{sname} GA Erratum 14 corrected sustained write IOPS", "max_results": 3})
        tool_resp_2 = (
            "<tool_response>\n"
            f"[S3] ({sname} Official GA Erratum #14, 2026-05): Supersedes beta preview ({stale_v}k); verified sustained write throughput is {true_v}k IOPS with {bw_gb} GB/s replication bandwidth.\n"
            "</tool_response>"
        )
        t3_think = (
            f"[CITE-CHECK] Primary official erratum `[S3]` (`2026-05`) supersedes beta `[S1]` (`{stale_v}k -> {true_v}k IOPS`) and corroborates `[S2]` (`{bw_gb} GB/s`). Every number will carry its exact `[S#]` tag."
        )
        final_ans = (
            f"For **{sname}**, verified sustained write performance is **{true_v}k IOPS** [S3] (superseding the earlier `{stale_v}k` beta preview figure [S1, S3]), "
            f"and sustained replication bandwidth is **{bw_gb} GB/s** [S2, S3].\n\n"
            + json.dumps({"write_kiops": true_v, "replication_gbps": bw_gb})
        )
        return {
            "id": f"v10_res_{idx:04d}",
            "pillar": "realtime_research_citations",
            "images": [],
            "messages": [
                {"role": "user", "content": u_msg},
                {"role": "assistant", "content": f"<think>\n{t1_think}</think>\n{t1_call}"},
                {"role": "tool", "content": tool_resp_1},
                {"role": "assistant", "content": f"<think>\n{t2_think}</think>\n{t2_call}"},
                {"role": "tool", "content": tool_resp_2},
                {"role": "assistant", "content": f"<think>\n{t3_think}</think>\n{final_ans}"},
            ],
        }

    if variant == 1:
        # Single-turn conflicting multi-source synthesis with `[S#]` citations + JSON
        drug = rng.choice(["Vexorin-B", "Caldren-7", "Lumivax-3", "Trelon-K"]) + f"-{idx % 9}"
        old_eff = round(rng.uniform(71.0, 79.0), 1)
        new_eff = round(old_eff - rng.choice([4.5, 6.0, 5.5]), 1)
        n_pat = rng.choice([620, 740, 880])
        prompt = (
            f"Synthesize the verified Phase-III efficacy (`efficacy_pct`) and adjudicated cohort size (`cohort_n`) for `{drug}` using ONLY `[S1]..[S3]`. "
            "Prioritize peer-reviewed/adjudicated data over unreviewed press releases, include inline `[S#]` citations, and end with JSON `{\"efficacy_pct\": <float>, \"cohort_n\": <int>}`:\n\n"
            f"[S1] (Sponsor Press Summary, 2025-10): Interim readout claimed {old_eff}% efficacy across 650 enrollees.\n"
            f"[S2] (Independent Adjudication Report, 2026-03): Final intention-to-treat cohort is `n = {n_pat}` patients.\n"
            f"[S3] (Peer-Reviewed Final Trial Table 2, 2026-04): Adjudicated primary endpoint efficacy is {new_eff}% across `n = {n_pat}` patients (replacing interim {old_eff}%)."
        )
        think = (
            f"[CITE-CHECK] `[S1]` is an interim press summary (`{old_eff}%`), whereas `[S2]` and `[S3]` are the later peer-reviewed/adjudicated records (`efficacy_pct = {new_eff}%` [S3], `cohort_n = {n_pat}` [S2, S3])."
        )
        ans = (
            f"Based on the adjudicated Phase-III analysis, `{drug}` achieved **{new_eff}%** primary endpoint efficacy [S3] across an intention-to-treat cohort of **n = {n_pat}** patients [S2, S3] (superseding the `{old_eff}%` interim press readout [S1, S3]).\n"
            + json.dumps({"efficacy_pct": new_eff, "cohort_n": n_pat})
        )
        return {
            "id": f"v10_res_{idx:04d}",
            "pillar": "realtime_research_citations",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}</think>\n{ans}"}],
            "verifier": {
                "kind": "citation_grounded",
                "required_citations": ["[S2]", "[S3]"],
                "valid_citations": ["[S1]", "[S2]", "[S3]"],
                "expected": {"efficacy_pct": new_eff, "cohort_n": n_pat},
                "forbid_tool_call": True,
            },
        }

    if variant == 2:
        # Honest Epistemic Abstention with Citations when Evidence is Missing
        chip = rng.choice(["Soliton-M4", "HyperCore-Z2", "Photonix-R8"]) + f"-{idx % 6}"
        mem_gb = rng.choice([128, 256, 384])
        prompt = (
            f"Using ONLY `[S1]` and `[S2]`, report the `INT4` sparse TOPS rating of `{chip}`. "
            "If unstated in the sources, do NOT fabricate a number: cite `[S1]` and `[S2]` for what is known and finish with JSON `{\"status\": \"insufficient_evidence\", \"int4_tops\": null}`:\n\n"
            f"[S1] ({chip} Brief): Equipped with {mem_gb} GB LPDDR6X unified memory.\n"
            f"[S2] ({chip} SDK Release Note): Adds BF16 kernel fusion; integer quantization (`INT8`/`INT4`) benchmarks have not yet been published."
        )
        think = (
            f"[CITE-CHECK] `[S1]` gives memory (`{mem_gb} GB`) and `[S2]` explicitly states `INT4` benchmarks are unpublished. Abstain cleanly with `[S1]`/`[S2]` citations and `null`."
        )
        ans = (
            f"The provided sources do not report an `INT4` sparse TOPS figure for `{chip}`: `[S1]` specifies `{mem_gb} GB` LPDDR6X memory [S1], and `[S2]` notes that `INT8`/`INT4` benchmarks have not yet been published [S2].\n"
            + json.dumps({"status": "insufficient_evidence", "int4_tops": None})
        )
        return {
            "id": f"v10_res_{idx:04d}",
            "pillar": "realtime_research_citations",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}</think>\n{ans}"}],
            "verifier": {
                "kind": "citation_grounded",
                "required_citations": ["[S1]", "[S2]"],
                "valid_citations": ["[S1]", "[S2]"],
                "expected": {"status": "insufficient_evidence", "int4_tops": None},
                "forbid_tool_call": True,
            },
        }

    if variant == 3:
        # Multi-source quantitative aggregation (`[S1]` + `[S2]` + `[S3]`)
        nodes = rng.choice([16, 24, 32])
        gpus_per = rng.choice([4, 8])
        tflops_per = rng.choice([250.0, 300.0, 400.0])
        eff = rng.choice([0.75, 0.80])
        tot_gpus = nodes * gpus_per
        sustained_tflops = round(tot_gpus * tflops_per * eff, 2)
        prompt = (
            "Combine `[S1]`, `[S2]`, and `[S3]` to compute `total_gpus` and `sustained_tflops` (`total_gpus * peak_tflops * scaling_eff`). "
            "Include inline `[S1]`, `[S2]`, `[S3]` citations and finish with JSON `{\"total_gpus\": <int>, \"sustained_tflops\": <float>}`:\n\n"
            f"[S1] (Cluster Topology): Deployment consists of `{nodes}` compute nodes with `{gpus_per}` GPUs per node.\n"
            f"[S2] (GPU Spec Sheet): Each GPU delivers `{tflops_per}` BF16 peak TFLOPS.\n"
            f"[S3] (Distributed Scaling Telemetry): Measured multi-node scaling efficiency at `{nodes}` nodes is `{eff}`."
        )
        think = (
            f"[CITE-CHECK] `total_gpus = {nodes} * {gpus_per} = {tot_gpus}` [S1]; `sustained_tflops = {tot_gpus} * {tflops_per} * {eff} = {sustained_tflops}` [S1, S2, S3]."
        )
        ans = (
            f"The deployment contains **{tot_gpus} GPUs** (`{nodes} × {gpus_per}`) [S1], each rated at **{tflops_per} BF16 TFLOPS** [S2], yielding **{sustained_tflops} sustained TFLOPS** at `{eff}` scaling efficiency [S3].\n"
            + json.dumps({"total_gpus": tot_gpus, "sustained_tflops": sustained_tflops})
        )
        return {
            "id": f"v10_res_{idx:04d}",
            "pillar": "realtime_research_citations",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}</think>\n{ans}"}],
            "verifier": {
                "kind": "citation_grounded",
                "required_citations": ["[S1]", "[S2]", "[S3]"],
                "valid_citations": ["[S1]", "[S2]", "[S3]"],
                "expected": {"total_gpus": tot_gpus, "sustained_tflops": sustained_tflops},
                "forbid_tool_call": True,
            },
        }

    # variant == 4: Multi-turn `realtime_web_research` + `verify_citations_and_claims` MCP pipeline!
    topic = rng.choice([
        ("LongRoPE2 Microsoft 128k context window perplexity", "LongRoPE2 preserves short-context accuracy via evolutionary search and mixed-context training [S1]."),
        ("Transformer-Squared Sakana Singular Value Fine-Tuning SVF", "Transformer^2 adapts singular value vectors z with far fewer parameters than LoRA [S1]."),
        ("Darwin Godel Machine Sakana open-ended archive", "Darwin Gödel Machine maintains an archive of self-modified coding agents validated empirically on benchmarks [S1]."),
    ])
    q_str, s_line = topic
    u_msg = (
        "Available tools:\n"
        "- `realtime_web_research(query: str, max_results: int = 5)`\n"
        "- `verify_citations_and_claims(draft_answer: str, sources_json: str)`\n\n"
        f"Research `{q_str}`, verify your cited summary with `verify_citations_and_claims`, and present the verified result."
    )
    c1 = _xml_call("realtime_web_research", {"query": q_str, "max_results": 3})
    src_obj = [{"id": "S1", "title": q_str, "snippet": s_line.replace(" [S1].", ".")}]
    tr1 = f"<tool_response>\n{json.dumps({'sources': src_obj})}\n</tool_response>"
    c2 = _xml_call("verify_citations_and_claims", {"draft_answer": s_line, "sources_json": json.dumps(src_obj)})
    tr2 = '<tool_response>\n{"is_verified": true, "citation_precision": 1.0, "uncited_claims": 0}\n</tool_response>'
    return {
        "id": f"v10_res_{idx:04d}",
        "pillar": "realtime_research_citations",
        "images": [],
        "messages": [
            {"role": "user", "content": u_msg},
            {"role": "assistant", "content": f"<think>\nQuery live literature via `realtime_web_research` first.</think>\n{c1}"},
            {"role": "tool", "content": tr1},
            {"role": "assistant", "content": f"<think>\nDraft cited claim `[S1]` and run `verify_citations_and_claims` to confirm zero unsupported spans.</think>\n{c2}"},
            {"role": "tool", "content": tr2},
            {"role": "assistant", "content": f"<think>\n`verify_citations_and_claims` passed (`precision = 1.0`). Output verified summary.</think>\n{s_line}"},
        ],
    }


def make_v10_zero_prior_and_root_cause_item(rng: random.Random, idx: int) -> dict[str, Any]:
    """Target every root-cause failure pattern: non-mutating copies, `log_softmax` vs `softmax`, elementwise `zip`, relative paths, & OOD systems!"""
    variant = idx % 6

    if variant == 0:
        # Non-mutating 1D Cellular / Stencil Automaton (cures caller-mutation bugs!)
        base_fn = rng.choice(["evolve_stencil_nonmutating", "advance_parity_tape", "step_modulo_automaton"])
        fn = f"{base_fn}_{idx % 13}" if idx >= 18 else base_fn
        mod_m = rng.choice([3, 4, 7])
        c_l, c_m, c_r = rng.randint(1, 3), rng.randint(1, 3), rng.randint(1, 3)
        prompt = (
            f"Implement `{fn}(tape: list[int], steps: int) -> list[int]` in Python WITHOUT mutating the input list `tape`.\n"
            f"For a 1D zero-padded boundary (`out_of_bounds = 0`), each step updates index `i` simultaneously to:\n"
            f"  `nxt[i] = ({c_l} * left + {c_m} * cur + {c_r} * right) % {mod_m}`.\n"
            "Even if `steps == 0`, return a fresh shallow copy `list(tape)`."
        )
        think = (
            "[INVARIANT] Start with `cur = list(tape)` so `tape` is NEVER mutated in place (even for `steps == 0`!).\n"
            f"[TRACE] For each step in `range(steps)`, compute a brand-new list comprehension using `left = cur[i-1] if i > 0 else 0` and `right = cur[i+1] if i + 1 < n else 0`."
        )
        code = (
            f"def {fn}(tape: list[int], steps: int) -> list[int]:\n"
            "    cur = list(tape)\n"
            "    n = len(cur)\n"
            "    for _ in range(steps):\n"
            "        cur = [\n"
            f"            ({c_l} * (cur[i - 1] if i > 0 else 0) + {c_m} * cur[i] + {c_r} * (cur[i + 1] if i + 1 < n else 0)) % {mod_m}\n"
            "            for i in range(n)\n"
            "        ]\n"
            "    return cur"
        )
        # Compute ground-truth in Python for unit test!
        t_in = [1, 2, 1]
        cur_t = list(t_in)
        for _ in range(2):
            cur_t = [(c_l * (cur_t[i - 1] if i > 0 else 0) + c_m * cur_t[i] + c_r * (cur_t[i + 1] if i + 1 < 3 else 0)) % mod_m for i in range(3)]
        test = (
            f"src = [1, 2, 1]\n"
            f"r0 = {fn}(src, 0)\nassert r0 == [1, 2, 1] and r0 is not src\n"
            f"r2 = {fn}(src, 2)\nassert src == [1, 2, 1], 'Mutated input!'\n"
            f"assert r2 == {cur_t!r}\n"
        )
        return {
            "id": f"v10_zp_{idx:04d}",
            "pillar": "unseen_ood_self_discover",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
        }

    if variant == 1:
        # Numerically stable `log_softmax` + Soft-Target Cross-Entropy using `zip` (cures `softmax` confusion & list-float TypeError!)
        base_fn = rng.choice(["safe_logprob_cross_entropy", "log_probs_and_soft_ce", "shifted_logsoftmax_loss"])
        fn = f"{base_fn}_{idx % 13}" if idx >= 18 else base_fn
        prompt = (
            f"Implement `{fn}(logits: list[float], target_probs: list[float]) -> tuple[list[float], float]` in pure Python (`math` only, NO NumPy).\n"
            "Return `(log_probs, ce_loss)` where `log_probs[i] = log(softmax(logits)[i])` (`<= 0.0`, computed via `x - m - log(sum(exp(x - m)))` without overflow) "
            "and `ce_loss = -sum(p * lp for p, lp in zip(target_probs, log_probs))`."
        )
        think = (
            "[INVARIANT] `log_softmax` subtracts `lse = m + math.log(sum(math.exp(x - m) for x in logits))`, NOT `exp(x - m) / sum_exp`!\n"
            "[PYTHON TYPES] `logits` and `target_probs` are Python `list[float]` (not NumPy arrays!), so iterate with `[x - lse for x in logits]` and `zip(target_probs, log_probs)`."
        )
        code = (
            "import math\n\n"
            f"def {fn}(logits: list[float], target_probs: list[float]) -> tuple[list[float], float]:\n"
            "    m = max(logits)\n"
            "    lse = m + math.log(sum(math.exp(x - m) for x in logits))\n"
            "    log_probs = [x - lse for x in logits]\n"
            "    ce_loss = -sum(p * lp for p, lp in zip(target_probs, log_probs) if p > 0.0)\n"
            "    return log_probs, ce_loss"
        )
        test = (
            f"lp, ce = {fn}([1100.0, 1100.0], [0.25, 0.75])\n"
            "assert lp[0] < 0.0 and abs(lp[0] - (-0.69314718056)) < 1e-7\n"
            "assert abs(ce - 0.69314718056) < 1e-7\n"
        )
        return {
            "id": f"v10_zp_{idx:04d}",
            "pillar": "unseen_ood_self_discover",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
        }

    if variant == 2:
        # Exact Relative Workspace Path in Tool Discovery (cures `/root/cline/` path prefixing!)
        pkg = rng.choice(["scheduler_core", "telemetry_bus", "router_engine", "quant_kernels", "consensus_raft"])
        mod = rng.choice(["burst_window.py", "epoch_budget.py", "frame_codec.py", "stride_table.py"])
        rel_path = f"{pkg}/{mod}"
        fn_q = rng.choice(["calc_epoch_floor", "derive_frame_slack", "compute_slot_quorum"])
        arg_n = rng.randint(6, 19)
        u_msg = (
            "Available tools:\n"
            "- `read_file(path: str)`\n"
            "- `grep_search(query: str, path: str)`\n\n"
            f"Read relative workspace file `{rel_path}` to inspect `{fn_q}({arg_n})`. Pass the exact relative path without prepending any `/root` or absolute directory."
        )
        t_think = f"[PATH-CHECK] `{rel_path}` is a workspace-relative path; pass `path=\"{rel_path}\"` verbatim without inventing `/root/cline/`."
        t_call = _xml_call("read_file", {"path": rel_path})
        coef = rng.randint(3, 7)
        bias = rng.randint(2, 11)
        val_out = coef * arg_n + bias
        tr = f"<tool_response>\ndef {fn_q}(n: int) -> int:\n    return {coef} * n + {bias}\n</tool_response>"
        t2_think = f"[SOLVE] From `{rel_path}`, `{fn_q}({arg_n}) = {coef} * {arg_n} + {bias} = {val_out}`."
        return {
            "id": f"v10_zp_{idx:04d}",
            "pillar": "unseen_ood_self_discover",
            "images": [],
            "messages": [
                {"role": "user", "content": u_msg},
                {"role": "assistant", "content": f"<think>\n{t_think}</think>\n{t_call}"},
                {"role": "tool", "content": tr},
                {"role": "assistant", "content": f"<think>\n{t2_think}</think>\n`{fn_q}({arg_n})` evaluates to `{val_out}` (`{coef} * {arg_n} + {bias}`)."},
            ],
        }

    if variant == 3:
        # Unseen Non-Associative Algebra with Explicit Left vs Right Association Check
        p_c, q_c, r_c = rng.randint(2, 4), rng.randint(1, 4), rng.randint(1, 5)
        x0, y0, z0 = rng.randint(2, 5), rng.randint(2, 5), rng.randint(2, 4)
        op = lambda a, b: p_c * a * b - q_c * a + r_c * b
        left_a = op(op(x0, y0), z0)
        right_a = op(x0, op(y0, z0))
        prompt = (
            f"Define the unfamiliar binary operator `◈` on integers by:\n"
            f"  `a ◈ b = {p_c}*a*b - {q_c}*a + {r_c}*b`\n"
            f"Compute step-by-step without assuming associativity:\n"
            f"1. `xy = {x0} ◈ {y0}`\n"
            f"2. `left_assoc = ({x0} ◈ {y0}) ◈ {z0}`\n"
            f"3. `right_assoc = {x0} ◈ ({y0} ◈ {z0})`\n"
            f"4. `is_equal = (left_assoc == right_assoc)`\n"
            'Return ONLY JSON `{"xy": <int>, "left_assoc": <int>, "right_assoc": <int>, "is_equal": <bool>}`.'
        )
        xy = op(x0, y0)
        yz = op(y0, z0)
        think = (
            f"[SELF-DISCOVER] Substitute exact positional arguments `(a, b)` twice:\n"
            f"1. `xy = {p_c}*{x0}*{y0} - {q_c}*{x0} + {r_c}*{y0} = {xy}`.\n"
            f"2. `left_assoc = ({xy}) ◈ {z0} = {p_c}*({xy})*{z0} - {q_c}*({xy}) + {r_c}*{z0} = {left_a}`.\n"
            f"3. `yz = {y0} ◈ {z0} = {yz}` -> `right_assoc = {x0} ◈ ({yz}) = {p_c}*{x0}*({yz}) - {q_c}*{x0} + {r_c}*({yz}) = {right_a}`."
        )
        exp = {"xy": xy, "left_assoc": left_a, "right_assoc": right_a, "is_equal": (left_a == right_a)}
        return {
            "id": f"v10_zp_{idx:04d}",
            "pillar": "unseen_ood_self_discover",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}</think>\n{json.dumps(exp)}"}],
            "verifier": {"expected": exp, "forbid_tool_call": True},
        }

    if variant == 4:
        # Non-mutating Matrix Row Normalization / Masking in NumPy
        base_fn = rng.choice(["normalize_active_rows", "unit_norm_nonzero_rows", "safe_l2_row_proj"])
        fn = f"{base_fn}_{idx % 13}" if idx >= 18 else base_fn
        prompt = (
            f"Implement `{fn}(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray` in NumPy WITHOUT mutating `mat` in place.\n"
            "For each row `i` of 2D float array `mat`, if its L2 norm `norm_i > eps`, divide row `i` by `norm_i`; otherwise leave row `i` as all zeros."
        )
        think = (
            "[INVARIANT] Copy `out = mat.astype(np.float64, copy=True)` first so `mat` is untouched; compute `norms = np.linalg.norm(out, axis=1, keepdims=True)` and use `np.where(norms > eps, out / np.maximum(norms, eps), 0.0)`."
        )
        code = (
            "import numpy as np\n\n"
            f"def {fn}(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:\n"
            "    arr = mat.astype(np.float64, copy=True)\n"
            "    norms = np.linalg.norm(arr, axis=1, keepdims=True)\n"
            "    res = np.where(norms > eps, arr / np.maximum(norms, eps), 0.0)\n"
            "    return res.astype(mat.dtype)"
        )
        test = (
            "import numpy as np\n"
            "X = np.array([[3.0, 4.0], [0.0, 0.0]])\n"
            f"Y = {fn}(X)\n"
            "np.testing.assert_allclose(X, np.array([[3.0, 4.0], [0.0, 0.0]]))\n"
            "np.testing.assert_allclose(Y, np.array([[0.6, 0.8], [0.0, 0.0]]), atol=1e-6)\n"
        )
        return {
            "id": f"v10_zp_{idx:04d}",
            "pillar": "unseen_ood_self_discover",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
        }

    # variant == 5: Custom Term-Rewriting Normal Form Termination Count
    # String over {'A', 'B'}: leftmost rule `BA -> AB` (bubble sort inversions!) or `AB -> B`
    s_len = rng.randint(5, 7)
    chars = [rng.choice(["A", "B"]) for _ in range(s_len)]
    s_init = "".join(chars)
    # Count inversions `(i < j with chars[i] == 'B' and chars[j] == 'A')` = exact rewrite steps of `BA -> AB`!
    inv_cnt = sum(1 for i in range(s_len) for j in range(i + 1, s_len) if chars[i] == "B" and chars[j] == "A")
    normal_form = "".join(sorted(chars))
    prompt = (
        f"In a string rewrite system with single rule `BA -> AB`, each step replaces one contiguous `\"BA\"` substring with `\"AB\"` until no `\"BA\"` remains.\n"
        f"Starting from `\"{s_init}\"`, determine the unique normal form string `normal_form` and exact number of rewrite steps `steps`.\n"
        'Return ONLY JSON `{"normal_form": "<str>", "steps": <int>}`.'
    )
    think = (
        f"[SELF-DISCOVER] Invariant: `BA -> AB` preserves counts of `A` and `B` and decreases the `(B_before_A)` inversion count by exactly `1` per step!\n"
        f"[SOLVE] Counting pairs `(i < j)` with `s[i]=='B'` and `s[j]=='A'` in `\"{s_init}\"` gives `steps = {inv_cnt}`, terminating at `\"{normal_form}\"`."
    )
    exp = {"normal_form": normal_form, "steps": inv_cnt}
    return {
        "id": f"v10_zp_{idx:04d}",
        "pillar": "unseen_ood_self_discover",
        "images": [],
        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}</think>\n{json.dumps(exp)}"}],
        "verifier": {"expected": exp, "forbid_tool_call": True},
    }


def make_v10_advanced_rsi_item(rng: random.Random, idx: int) -> dict[str, Any]:
    """Train deep `v10` RSI mastery: `Transformer^2` SVF, `Darwin Gödel Machine`, `Dr. GRPO + DAPO`, `AlphaEdit`, `TIES`, `LongRoPE2`, & MCP self-rewrite."""
    variant = idx % 7

    if variant == 0:
        # Transformer^2 Prompt-Dispatched SVF Singular Expert Blending
        base_fn = rng.choice(["blend_svf_singular_bank", "apply_transformer2_svf", "svf_expert_compose"])
        fn = f"{base_fn}_{idx % 11}" if idx >= 21 else base_fn
        prompt = (
            f"Implement `{fn}(W: np.ndarray, Z_bank: np.ndarray, router_logits: np.ndarray) -> np.ndarray` in NumPy (`Transformer^2` dynamic SVF):\n"
            "- `W`: `(d_out, d_in)`, `Z_bank`: `(E, k)` where `k = min(d_out, d_in)`, `router_logits`: 1D `(E,)`.\n"
            "- Compute numerically stable softmax routing weights `probs` over `router_logits`, form `z_mix = probs @ Z_bank`, factor `U, S, Vt = np.linalg.svd(W, full_matrices=False)`, scale `S_new = S * (1.0 + z_mix)`, and return `(U * S_new[None, :]) @ Vt` in `W.dtype` without mutating `W`."
        )
        think = (
            "[RSI-SVF] Subtract `np.max(router_logits)` for stable softmax `probs` (`(E,)`), blend `z_mix = probs @ Z_bank` (`(k,)`), and reconstruct `(U * (S * (1.0 + z_mix))[None, :]) @ Vt`."
        )
        code = (
            "import numpy as np\n\n"
            f"def {fn}(W: np.ndarray, Z_bank: np.ndarray, router_logits: np.ndarray) -> np.ndarray:\n"
            "    shifted = router_logits - np.max(router_logits)\n"
            "    exps = np.exp(shifted)\n"
            "    probs = exps / np.sum(exps)\n"
            "    z_mix = probs @ Z_bank\n"
            "    U, S, Vt = np.linalg.svd(W.astype(np.float64), full_matrices=False)\n"
            "    S_new = S * (1.0 + z_mix)\n"
            "    return ((U * S_new[None, :]) @ Vt).astype(W.dtype)"
        )
        test = (
            "import numpy as np\n"
            "W = np.diag([6.0, 2.0])\n"
            "Zb = np.array([[0.5, 0.0], [0.5, 1.0]])\n"
            f"Wn = {fn}(W, Zb, np.array([0.0, 0.0]))\n"
            "np.testing.assert_allclose(Wn, np.diag([9.0, 3.0]), atol=1e-6)\n"
        )
        return {
            "id": f"v10_rsi_{idx:04d}",
            "pillar": "advanced_rsi_dgm_svf",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
        }

    if variant == 1:
        # Gram / Orthonormal Nullspace Gradient Projection (`grad @ (I - V.T @ V)`)
        base_fn = rng.choice(["project_grad_to_nullspace", "nullspace_safe_delta", "orthogonal_anchor_filter"])
        fn = f"{base_fn}_{idx % 11}" if idx >= 21 else base_fn
        prompt = (
            f"Implement `{fn}(grad_w: np.ndarray, anchor_basis: np.ndarray) -> np.ndarray` in NumPy (`AlphaEdit` / SEAL nullspace protection):\n"
            "- `grad_w`: `(d_out, d_in)` proposed weight update, `anchor_basis`: `(r, d_in)` rows spanning protected input directions (not necessarily orthonormal!).\n"
            "- Compute thin SVD `U_a, S_a, Vt_a = np.linalg.svd(anchor_basis, full_matrices=False)`, keep rows `V_keep = Vt_a[S_a > 1e-7]`, and return `grad_w - (grad_w @ V_keep.T) @ V_keep` in `grad_w.dtype` without mutating `grad_w`."
        )
        think = (
            "[RSI-NULLSPACE] SVD of `anchor_basis` yields orthonormal row basis `V_keep = Vt_a[S_a > 1e-7]` (`(k, d_in)`), so `G_proj = G - (G @ V_keep.T) @ V_keep` satisfies `G_proj @ anchor_basis.T == 0` even when `anchor_basis` has redundant/unnormalized rows!"
        )
        code = (
            "import numpy as np\n\n"
            f"def {fn}(grad_w: np.ndarray, anchor_basis: np.ndarray) -> np.ndarray:\n"
            "    G = grad_w.astype(np.float64, copy=True)\n"
            "    if anchor_basis.size == 0:\n"
            "        return G.astype(grad_w.dtype)\n"
            "    _, S_a, Vt_a = np.linalg.svd(anchor_basis.astype(np.float64), full_matrices=False)\n"
            "    V_keep = Vt_a[S_a > 1e-7]\n"
            "    if V_keep.shape[0] == 0:\n"
            "        return G.astype(grad_w.dtype)\n"
            "    proj = G - (G @ V_keep.T) @ V_keep\n"
            "    return proj.astype(grad_w.dtype)"
        )
        test = (
            "import numpy as np\n"
            "G = np.array([[2.0, 3.0, 4.0], [-1.0, 5.0, 2.0]])\n"
            "A = np.array([[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])\n"
            f"Gp = {fn}(G, A)\n"
            "np.testing.assert_allclose(Gp @ A.T, np.zeros((2, 2)), atol=1e-7)\n"
            "np.testing.assert_allclose(Gp[:, 1:], G[:, 1:], atol=1e-7)\n"
        )
        return {
            "id": f"v10_rsi_{idx:04d}",
            "pillar": "advanced_rsi_dgm_svf",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
        }

    if variant == 2:
        # DAPO + Dr. GRPO Unbiased Advantage Computation in Python
        base_fn = rng.choice(["dr_grpo_group_advantages", "compute_dapo_drgrpo_adv", "unbiased_group_centered_adv"])
        fn = f"{base_fn}_{idx % 11}" if idx >= 21 else base_fn
        prompt = (
            f"Implement `{fn}(rewards: list[float], eps: float = 1e-8) -> tuple[bool, list[float]]` in pure Python (`DAPO` + `Dr. GRPO`):\n"
            "- Given group rewards `rewards` of length `G >= 2`, compute `mean_r = sum(rewards) / len(rewards)` and `adv = [r - mean_r for r in rewards]` (NO division by `std`!).\n"
            "- Return `(is_active_group, adv)` where `is_active_group` is `True` iff `max(rewards) - min(rewards) > eps` (`DAPO` dynamic non-zero variance filter)."
        )
        think = (
            "[RSI-RLVR] `Dr. GRPO` removes `1/std(R)` normalization so nearly-solved groups do not explode gradients, while `DAPO` skips zero-spread groups (`max(R) - min(R) <= eps`)."
        )
        code = (
            f"def {fn}(rewards: list[float], eps: float = 1e-8) -> tuple[bool, list[float]]:\n"
            "    mean_r = sum(rewards) / len(rewards)\n"
            "    adv = [r - mean_r for r in rewards]\n"
            "    active = (max(rewards) - min(rewards)) > eps\n"
            "    return active, adv"
        )
        test = (
            f"act0, adv0 = {fn}([1.0, 1.0, 1.0, 1.0])\n"
            "assert act0 is False and all(abs(x) < 1e-12 for x in adv0)\n"
            f"act1, adv1 = {fn}([1.0, 0.0, 0.0, 0.0])\n"
            "assert act1 is True and abs(adv1[0] - 0.75) < 1e-9 and abs(adv1[1] - (-0.25)) < 1e-9\n"
        )
        return {
            "id": f"v10_rsi_{idx:04d}",
            "pillar": "advanced_rsi_dgm_svf",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
        }

    if variant == 3:
        # Darwin Gödel Machine Archive Lineage Selector in Python
        base_fn = rng.choice(["select_dgm_parent_node", "pick_archive_ancestor", "dgm_ucb_lineage_choice"])
        fn = f"{base_fn}_{idx % 11}" if idx >= 21 else base_fn
        beta_v = rng.choice([0.1, 0.15, 0.2])
        prompt = (
            f"Implement `{fn}(archive: list[dict], beta: float = {beta_v}) -> str | None` in Python (`Darwin Gödel Machine` archive selector):\n"
            "Each dict in `archive` has keys `{'id': str, 'score': float, 'children': int, 'regressed': bool}`.\n"
            "Filter out any node with `node['regressed'] == True`; among remaining nodes, return the `'id'` maximizing `node['score'] + beta / ((1 + node['children']) ** 0.5)`, or `None` if no non-regressed node exists."
        )
        think = (
            "[RSI-DGM] Filter `[n for n in archive if not n['regressed']]`; if empty return `None`; else `max(..., key=lambda n: n['score'] + beta / ((1 + n['children']) ** 0.5))['id']`."
        )
        code = (
            f"def {fn}(archive: list[dict], beta: float = {beta_v}) -> str | None:\n"
            "    valid = [n for n in archive if not n.get('regressed', False)]\n"
            "    if not valid:\n"
            "        return None\n"
            "    best = max(valid, key=lambda n: float(n['score']) + beta / ((1.0 + int(n['children'])) ** 0.5))\n"
            "    return str(best['id'])"
        )
        test = (
            "nodes = [\n"
            "    {'id': 'a0', 'score': 0.80, 'children': 8, 'regressed': False},\n"
            "    {'id': 'a1', 'score': 0.95, 'children': 0, 'regressed': True},\n"
            "    {'id': 'a2', 'score': 0.78, 'children': 0, 'regressed': False},\n"
            "]\n"
            f"assert {fn}(nodes, beta=0.2) == 'a2'\n"
            f"assert {fn}([{{'id': 'x', 'score': 1.0, 'children': 0, 'regressed': True}}]) is None\n"
        )
        return {
            "id": f"v10_rsi_{idx:04d}",
            "pillar": "advanced_rsi_dgm_svf",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
        }

    if variant == 4:
        # TIES-Merging (Trim -> Elect Sign -> Disjoint Mean) without mutating inputs!
        base_fn = rng.choice(["ties_merge_task_deltas", "nonmutating_ties_consensus", "fuse_deltas_ties"])
        fn = f"{base_fn}_{idx % 11}" if idx >= 21 else base_fn
        prompt = (
            f"Implement `{fn}(deltas: list[np.ndarray], top_frac: float = 0.5) -> np.ndarray` in NumPy (`TIES-Merging`, non-mutating):\n"
            "1. For each delta `d` in `deltas`, flatten a copy and keep the `k = int(round(top_frac * d.size))` entries with largest `abs(d)` (zeroing the rest; if `k <= 0` return zeros).\n"
            "2. Compute elementwise elected sign `s_elect = np.sign(sum_of_trimmed_deltas)`.\n"
            "3. At each coordinate, average ONLY the trimmed entries whose `np.sign(entry) == s_elect` and `entry != 0` (`0.0` where count is `0`)."
        )
        think = (
            "[RSI-TIES] Work on `.astype(np.float64, copy=True)` flattened vectors, keep top-`k` indices via `np.argsort(np.abs(flat))[-k:]`, compute `s_elect = np.sign(np.sum(trimmed, axis=0))`, and divide `np.sum(aligned, axis=0)` by `np.maximum(count_aligned, 1)`."
        )
        code = (
            "import numpy as np\n\n"
            f"def {fn}(deltas: list[np.ndarray], top_frac: float = 0.5) -> np.ndarray:\n"
            "    shape = deltas[0].shape\n"
            "    dtype = deltas[0].dtype\n"
            "    n = deltas[0].size\n"
            "    k = int(round(top_frac * n))\n"
            "    if k <= 0:\n"
            "        return np.zeros(shape, dtype=dtype)\n"
            "    trimmed = []\n"
            "    for d in deltas:\n"
            "        flat = d.astype(np.float64).ravel().copy()\n"
            "        out = np.zeros_like(flat)\n"
            "        idx = np.argsort(np.abs(flat), kind='mergesort')[-k:]\n"
            "        out[idx] = flat[idx]\n"
            "        trimmed.append(out)\n"
            "    mat = np.stack(trimmed, axis=0)\n"
            "    s_elect = np.sign(np.sum(mat, axis=0))\n"
            "    mask = (mat != 0.0) & (np.sign(mat) == s_elect[None, :])\n"
            "    counts = np.sum(mask, axis=0)\n"
            "    merged = np.where(counts > 0, np.sum(np.where(mask, mat, 0.0), axis=0) / np.maximum(counts, 1), 0.0)\n"
            "    return merged.reshape(shape).astype(dtype)"
        )
        test = (
            "import numpy as np\n"
            "a = np.array([6.0, -4.0, 0.2, -0.1])\n"
            "b = np.array([2.0, 8.0, -0.3, 0.1])\n"
            f"m = {fn}([a, b], top_frac=0.5)\n"
            "np.testing.assert_allclose(m, np.array([4.0, 8.0, 0.0, 0.0]), atol=1e-6)\n"
            "np.testing.assert_allclose(a, np.array([6.0, -4.0, 0.2, -0.1]))\n"
        )
        return {
            "id": f"v10_rsi_{idx:04d}",
            "pillar": "advanced_rsi_dgm_svf",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
        }

    if variant == 5:
        # Multi-turn Live MCP Weight Inspection -> SEAL + Transformer^2 SVF + DGM Self-Rewrite!
        fn_u = f"poly_horner_{idx % 9}"
        c2, c1, c0 = rng.randint(2, 5), rng.randint(1, 6), rng.randint(1, 9)
        u_msg = (
            "Available tools:\n"
            "- `inspect_model_weights(layer_filter: str)`\n"
            "- `seal_outer_loop_self_rewrite(task_prompt: str, target_solution: str, python_unit_test: str, steps: int = 4)`\n\n"
            f"First inspect the live `40L` `latent_recur` and `attn_res` scales, then execute a verified outer-loop SEAL weight rewrite to internalize `def {fn_u}(x): return ({c2}*x - {c1})*x + {c0}`."
        )
        call1 = _xml_call("inspect_model_weights", {"layer_filter": "latent_recur"})
        resp1 = (
            "<tool_response>\n"
            '{"total_params": 3177950000, "dtypes": ["bfloat16"], "pure_16bit_bf16": true, '
            '"matched_tensors": [{"name": "language_model.model.latent_recur_35.loop_scales", "shape": [2, 2048], "dtype": "bfloat16"}]}\n'
            "</tool_response>"
        )
        val_at_2 = (c2 * 2 - c1) * 2 + c0
        call2 = _xml_call(
            "seal_outer_loop_self_rewrite",
            {
                "task_prompt": f"Implement `{fn_u}(x)` in Python.",
                "target_solution": f"```python\ndef {fn_u}(x: int) -> int:\n    return ({c2} * x - {c1}) * x + {c0}\n```",
                "python_unit_test": f"assert {fn_u}(2) == {val_at_2}",
                "steps": 4,
            },
        )
        resp2 = (
            "<tool_response>\n"
            '{"accepted": true, "post_unit_test_passed": true, "svf_dispatch_expert": "code_nullspace", '
            '"dgm_archive_node": "dgm_gen_004", "anchor_regression": -0.0014}\n'
            "</tool_response>"
        )
        return {
            "id": f"v10_rsi_{idx:04d}",
            "pillar": "advanced_rsi_dgm_svf",
            "images": [],
            "messages": [
                {"role": "user", "content": u_msg},
                {"role": "assistant", "content": f"<think>\nInspect `latent_recur` tensors in pure `16-bit` `bfloat16` first.</think>\n{call1}"},
                {"role": "tool", "content": resp1},
                {"role": "assistant", "content": f"<think>\n`latent_recur_35` confirmed in `bfloat16` (`3.18B`). Now run `seal_outer_loop_self_rewrite` with isolated unit test `assert {fn_u}(2) == {val_at_2}`.</think>\n{call2}"},
                {"role": "tool", "content": resp2},
                {"role": "assistant", "content": f"<think>\nRewrite accepted (`post_unit_test_passed = true`, `anchor_regression = -0.0014`, logged to `dgm_gen_004`).</think>\nLive `16-bit` (`bfloat16`) weights inspected (`3.18B`, `latent_recur_35.loop_scales` `[2, 2048]`) and outer-loop SEAL nullspace + SVF rewrite accepted (`{fn_u}(2) == {val_at_2}`, `anchor_regression = -0.0014`, DGM node `dgm_gen_004`)."},
            ],
        }

    # variant == 6: Honest Architecture Self-Knowledge (`What is in weights vs external runtime`)
    prompt = (
        "Explain accurately what is stored directly inside `LFM3-VL-3.2B-RPR-V10-APEX-BF16`'s pure `16-bit` (`bfloat16`) weights versus what executes in the external runtime/MCP harness."
    )
    think = (
        "[EPISTEMIC-HONESTY] Distinguish strictly between forward-pass `bfloat16` tensor parameters (`3.18B` across `40` hybrid `conv`/`full_attention` layers + `Engram` + `5x CrossLayerAttnRes` + `LatentReasoningRecurrence` + `LongRoPE2YaRN` + internalized Self-Discover/RPR/Citation/RSI priors) and external tools (`python3 -I`, `realtime_web_research`, `verify_citations_and_claims`, and `seal_outer_loop_self_rewrite` persistent overlay mutation)."
    )
    ans = (
        "### Inside Pure `16-bit` (`bfloat16`) Model Weights (`~3.18B` Parameters)\n"
        "- **40-Layer Hybrid Trunk + Vision Tower:** `14` `full_attention` layers (with `256k` `LongRoPE2YaRN` continuous dual-path RoPE and `MoBA` sparse block routing), `sliding_attention`, and gated `ShortConv` layers.\n"
        "- **Learned Architectural Sidecars:** `Engram` conditional 2-gram memory (`L6, L18`), `5` `CrossLayerAttnRes` depth-attention gates (`L26, L30, L32, L36, L38`), and 2-step shared-weight `LatentReasoningRecurrence` (`L35`).\n"
        "- **Internalized Reasoning & RSI Capabilities:** `Self-Discover` constraint deduction, `[S#]` citation-grounded synthesis, zero-prior hypothesis-testing, and weight-surgery derivations (`SVD/PiSSA`, `Transformer^2 SVF`, `AlphaEdit/MEMIT`, `TIES`, `DGM`, `Dr. GRPO`).\n\n"
        "### External Runtime / MCP Harness (`:8991` & `rpr-seal-rsi`)\n"
        "- **Live Web Retrieval & Citation Verification:** `realtime_web_research` fetches live post-cutoff web/arXiv/Wikipedia sources, and `verify_citations_and_claims` audits `[S#]` entailment.\n"
        "- **Isolated Code Execution & Multi-Pass RPR Controller:** `python3 -I` sandboxes and `rpr_pyramid_solve` orchestrate multi-branch verification.\n"
        "- **Persistent Outer-Loop Weight Mutation:** A standard forward pass leaves `.safetensors` immutable; persistent weight edits execute via `seal_outer_loop_self_rewrite` (`Transformer^2 SVF + Nullspace + DGM` archive gating)."
    )
    return {
        "id": f"v10_rsi_{idx:04d}",
        "pillar": "advanced_rsi_dgm_svf",
        "images": [],
        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}</think>\n{ans}"}],
    }


def make_v10_long_context_sft_item(rng: random.Random, idx: int) -> dict[str, Any]:
    """Generate `4k–6k` token OOD Scientific Observatory Multi-Needle + Erratum + `[REC-XXXX]` Citation SFT items!
    Uses a completely disjoint scientific domain, filler vocabulary, and JSON schema from `heldout_long_256k_needles_12`!
    """
    missions = ["Helios-Survey", "Cassini-Deep", "Kepler-Wide", "Vega-Array", "Sirius-Lens", "Altair-Probe", "Polaris-Scan", "Antares-Watch"]
    sensors = ["SNS-CryoBol", "SNS-FabryPerot", "SNS-GrismUV", "SNS-MicroCal", "SNS-Heterodyne", "SNS-Coronagraph", "SNS-Echelle", "SNS-Polarimeter"]
    retired = ["RET-Bench0", "RET-Bench1", "RET-Bench2", "RET-Bench3"]

    mission = f"{missions[idx % len(missions)]}-{idx + 10}"
    sensor_c = f"{sensors[(idx * 3) % len(sensors)]}-{(idx % 9) + 1}"
    ret_c = f"{retired[idx % len(retired)]}-{idx % 5}"
    raw_flux = rng.randint(25, 75)
    stale_gain = rng.choice([2, 3, 7])
    true_gain = rng.choice([4, 5, 6])
    dark_bias = rng.randint(11, 39)
    tok_target = rng.choice([1900, 2150, 2400])

    num_recs = max(24, int(round(tok_target / 78.0)))
    i_h1 = max(2, int(num_recs * 0.14))
    i_st = max(i_h1 + 3, int(num_recs * 0.38))
    i_h2 = max(i_st + 3, int(num_recs * 0.62))
    i_h3 = min(num_recs - 3, max(i_h2 + 3, int(num_recs * 0.87)))

    astro_targets = ["NGC-4151", "M87-Jet", "Cygnus-X1", "Tau-Ceti", "TRAPPIST-1e", "Vela-Pulsar", "3C-273", "Beta-Pictoris"]
    bands = ["near-infrared", "sub-millimeter", "soft-X-ray", "Lyman-alpha", "far-ultraviolet", "narrowband-OIII"]

    id_h1 = f"REC-{i_h1:04d}"
    id_st = f"REC-{i_st:04d}"
    id_h2 = f"REC-{i_h2:04d}"
    id_h3 = f"REC-{i_h3:04d}"

    lines: list[str] = []
    for r_i in range(1, num_recs + 1):
        rid = f"[REC-{r_i:04d}]"
        if r_i == i_h1:
            lines.append(
                f"{rid} (Flight Instrument Manifest): Campaign `{mission}` operates primary focal-plane detector "
                f"`{sensor_c}` (engineering spare `{ret_c}` remained stowed in ground storage)."
            )
        elif r_i == i_st:
            lines.append(
                f"{rid} (Pre-Launch Thermal Note): Uncalibrated benchtop gain for `{sensor_c}` was estimated at "
                f"`{stale_gain}x` with zero dark-current offset prior to vacuum cooldown."
            )
        elif r_i == i_h2:
            lines.append(
                f"{rid} (Orbital Photometry Ledger): Raw photon count rate measured on `{sensor_c}` equals "
                f"`{raw_flux}` mJy (stowed ground unit `{ret_c}` synthetic test signal was `888` mJy)."
            )
        elif r_i == i_h3:
            lines.append(
                f"{rid} (Post-Cooldown Calibration Bulletin — Supersedes {id_st}): Cryogenic calibration SUPERSEDES "
                f"`{id_st}` (`{stale_gain}x` discarded). Certified formula for `{sensor_c}` is `calibrated_flux = raw_flux * {true_gain} + {dark_bias}`."
            )
        else:
            tgt = astro_targets[(r_i * 7) % len(astro_targets)]
            bnd = bands[(r_i * 5) % len(bands)]
            exp_s = 120 + ((r_i * 29) % 600)
            seeing = round(0.4 + ((r_i * 13) % 9) * 0.1, 1)
            lines.append(
                f"{rid} (Nightly Slewing Log — `{tgt}`): Completed `{exp_s}`-second {bnd} integration under "
                f"`{seeing}`-arcsecond star-tracker stability; cryocooler cold-tip held steady at `4.2` Kelvin with nominal reaction-wheel momentum."
            )

    final_flux = raw_flux * true_gain + dark_bias
    prompt = (
        f"Review the `{num_recs}` observatory records (`[REC-0001]` .. `[REC-{num_recs:04d}]`) below.\n"
        f"Find the certified `calibrated_flux` for campaign `{mission}` by chaining the active flight detector, its orbital `raw_flux`, "
        f"and the superseding post-cooldown calibration bulletin (ignoring stowed spares and superseded pre-launch notes). "
        f"Include inline `[REC-XXXX]` citations and conclude with JSON `{{\"detector\": \"<str>\", \"calibrated_flux\": <int>, \"cited_records\": [\"REC-XXXX\", \"REC-YYYY\", \"REC-ZZZZ\"]}}`:\n\n"
        + "\n".join(lines)
    )
    c1, c2, c3 = f"[{id_h1}]", f"[{id_h2}]", f"[{id_h3}]"
    auth_plain = [id_h1, id_h2, id_h3]
    think = (
        f"[LONG-CTX NEEDLE SCAN]\n"
        f"1. Manifest {c1}: active flight detector for `{mission}` is `{sensor_c}` (`{ret_c}` stowed).\n"
        f"2. Photometry {c2}: `raw_flux = {raw_flux}` mJy.\n"
        f"3. Cryo Bulletin {c3}: supersedes `{id_st}` (`{stale_gain}x`) -> `calibrated_flux = {raw_flux} * {true_gain} + {dark_bias} = {final_flux}`."
    )
    ans = (
        f"Campaign `{mission}` uses active detector **`{sensor_c}`** {c1} with orbital raw flux **`{raw_flux}` mJy** {c2}. "
        f"Applying the superseding post-cooldown calibration (`{raw_flux} * {true_gain} + {dark_bias}`) yields **`{final_flux}` mJy** {c3}.\n"
        + json.dumps({"detector": sensor_c, "calibrated_flux": final_flux, "cited_records": auth_plain})
    )
    return {
        "id": f"v10_longsft_{idx:04d}",
        "pillar": "long_context_synthesis",
        "images": [],
        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}</think>\n{ans}"}],
    }


def make_v10_orpo_pair(rng: random.Random, idx: int) -> dict[str, Any]:
    """Contrastive ORPO pairs targeting citations, complex logic, non-mutation, `log_softmax`, relative paths, & RSI math."""
    kind = idx % 6
    if kind == 0:
        row = make_v10_realtime_research_citation_item(rng, 1 + 5 * (idx // 6))
        chosen = row["messages"][-1]["content"]
        rejected = (
            "<think>\nTake the first number mentioned in S1 without checking later errata.</think>\n"
            "According to the report [S9], the efficacy was higher in the press summary.\n"
            '{"efficacy_pct": 99.9, "cohort_n": 650}'
        )
        return {"id": f"v10_orpo_{idx:04d}", "messages": row["messages"][:-1], "chosen": chosen, "rejected": rejected}
    if kind == 1:
        row = make_v10_zero_prior_and_root_cause_item(rng, 0 + 6 * (idx // 6))
        chosen = row["messages"][-1]["content"]
        # Rejected mutates `tape` in place!
        rejected = chosen.replace("cur = list(tape)", "cur = tape").replace("cur = [\n", "cur[:] = [\n")
        return {"id": f"v10_orpo_{idx:04d}", "messages": row["messages"][:-1], "chosen": chosen, "rejected": rejected}
    if kind == 2:
        row = make_v10_zero_prior_and_root_cause_item(rng, 1 + 6 * (idx // 6))
        chosen = row["messages"][-1]["content"]
        # Rejected returns positive `softmax` probabilities instead of negative `log_softmax`!
        rejected = chosen.replace("log_probs = [x - lse for x in logits]", "log_probs = [math.exp(x - m) for x in logits]")
        return {"id": f"v10_orpo_{idx:04d}", "messages": row["messages"][:-1], "chosen": chosen, "rejected": rejected}
    if kind == 3:
        row = make_v10_zero_prior_and_root_cause_item(rng, 2 + 6 * (idx // 6))
        chosen = row["messages"][1]["content"]
        # Rejected hallucinates `/root/cline/` prefix!
        rejected = chosen.replace("<parameter=path>\n", "<parameter=path>\n/root/cline/")
        return {"id": f"v10_orpo_{idx:04d}", "messages": row["messages"][:1], "chosen": chosen, "rejected": rejected}
    if kind == 4:
        row = make_v10_self_discover_logic_item(rng, 1 + 6 * (idx // 6))
        chosen = row["messages"][-1]["content"]
        rejected = "<think>\nGuessing all islanders are Knights without checking biconditionals.</think>\n" + json.dumps({"knights": 3})
        return {"id": f"v10_orpo_{idx:04d}", "messages": row["messages"][:-1], "chosen": chosen, "rejected": rejected}
    row = make_v10_advanced_rsi_item(rng, 0 + 7 * (idx // 6))
    chosen = row["messages"][-1]["content"]
    rejected = chosen.replace("(U * S_new[None, :]) @ Vt", "U @ Vt[:, :1]")
    return {"id": f"v10_orpo_{idx:04d}", "messages": row["messages"][:-1], "chosen": chosen, "rejected": rejected}


def build_dapo_hard_episodes_96(rng: random.Random) -> list[dict[str, Any]]:
    """Build 96 genuinely challenging multi-domain RLVR episodes so `G=4` rollouts produce rich split-variance (`0 < pass < 4`) updates!"""
    episodes: list[dict[str, Any]] = []

    # 1. 28 Challenging Complex Logic / Multi-Clue Grid / Non-Associative / Game-Tree JSON Puzzles
    for i in range(28):
        if i % 2 == 0:
            # 4-variable CNF + weighted objective
            t_vec = (bool(i & 1), bool((i >> 1) & 1), True, bool((i >> 2) & 1))
            # Build 9 clauses uniquely pinning `t_vec`
            others = [v for v in itertools.product([False, True], repeat=4) if v != t_vec]
            rng.shuffle(others)
            cls_list = []
            for w in others:
                # Pick 3 indices where `w` differs or include all differing indices up to 3
                diffs = [j for j in range(4) if w[j] != t_vec[j]]
                extras = [j for j in range(4) if j not in diffs]
                chosen_idx = (diffs + extras)[:3]
                lits = [f"v{j+1}" if not w[j] else f"NOT v{j+1}" for j in chosen_idx]
                c_s = "(" + " OR ".join(lits) + ")"
                if c_s not in cls_list:
                    cls_list.append(c_s)
            # Verify uniqueness!
            sols = []
            for cand in itertools.product([False, True], repeat=4):
                ok = True
                for c_s in cls_list:
                    parts = [p.strip() for p in c_s.strip("()").split("OR")]
                    sat_c = any((not cand[int(p[-1]) - 1]) if p.startswith("NOT ") else cand[int(p[-1]) - 1] for p in parts)
                    if not sat_c:
                        ok = False
                        break
                if ok:
                    sols.append(cand)
            assert sols == [t_vec]
            w_score = 1 * int(t_vec[0]) + 2 * int(t_vec[1]) + 4 * int(t_vec[2]) + 8 * int(t_vec[3])
            c_txt = "\n".join(f"  - `{c}`" for c in cls_list)
            prompt = (
                f"Find the unique Boolean assignment `(v1, v2, v3, v4)` satisfying every clause:\n{c_txt}\n"
                f"Also compute `bitmask_score = 1*int(v1) + 2*int(v2) + 4*int(v3) + 8*int(v4)`.\n"
                'Return ONLY JSON `{"v1": <bool>, "v2": <bool>, "v3": <bool>, "v4": <bool>, "bitmask_score": <int>}`.'
            )
            exp = {"v1": t_vec[0], "v2": t_vec[1], "v3": t_vec[2], "v4": t_vec[3], "bitmask_score": w_score}
            ref = f"<think>\n[SELF-DISCOVER] Propagate forced clauses and verify uniqueness -> `{t_vec}`, `bitmask_score = {w_score}`.</think>\n{json.dumps(exp)}"
            episodes.append({
                "id": f"dapo10_logic_{i:03d}",
                "kind": "json_values",
                "messages": [{"role": "user", "content": prompt}],
                "verifier": {"expected": exp, "forbid_tool_call": True},
                "reference_answer": ref,
            })
        else:
            # Multi-step non-associative 3-stage nested evaluation `(a ⊕ b) ⊕ (b ⊕ c)`
            p, q, r = rng.randint(2, 4), rng.randint(2, 4), rng.randint(1, 5)
            a, b, c = rng.randint(2, 5), rng.randint(2, 5), rng.randint(2, 5)
            op = lambda x, y: p * x * y - q * x + r * y - 2
            ab = op(a, b)
            bc = op(b, c)
            total = op(ab, bc)
            prompt = (
                f"Given non-associative operator `x ⊕ y = {p}*x*y - {q}*x + {r}*y - 2`, evaluate:\n"
                f"1. `left_part = {a} ⊕ {b}`\n"
                f"2. `right_part = {b} ⊕ {c}`\n"
                f"3. `combined = left_part ⊕ right_part`\n"
                'Return ONLY JSON `{"left_part": <int>, "right_part": <int>, "combined": <int>}`.'
            )
            exp = {"left_part": ab, "right_part": bc, "combined": total}
            ref = f"<think>\n`left_part = {ab}`, `right_part = {bc}`, `combined = {p}*{ab}*{bc} - {q}*{ab} + {r}*{bc} - 2 = {total}`.</think>\n{json.dumps(exp)}"
            episodes.append({
                "id": f"dapo10_logic_{i:03d}",
                "kind": "json_values",
                "messages": [{"role": "user", "content": prompt}],
                "verifier": {"expected": exp, "forbid_tool_call": True},
                "reference_answer": ref,
            })

    # 2. 24 Citation-Grounded Multi-Source Conflict & Multi-Hop Synthesis Episodes (`kind = "citation_grounded"`)
    for i in range(24):
        src_row = make_v10_realtime_research_citation_item(rng, (1 if i % 2 == 0 else 3) + 5 * i)
        episodes.append({
            "id": f"dapo10_cite_{i:03d}",
            "kind": "citation_grounded",
            "messages": src_row["messages"][:-1],
            "verifier": src_row["verifier"],
            "reference_answer": src_row["messages"][-1]["content"],
        })

    # 3. 24 Challenging Non-Mutating / Numerical / Graph Python Code Episodes (`kind = "python_code"`)
    for i in range(24):
        if i % 3 == 0:
            src_row = make_v10_self_discover_logic_item(rng, 5 + 6 * i)
        elif i % 3 == 1:
            src_row = make_v10_zero_prior_and_root_cause_item(rng, 0 + 6 * i)
        else:
            src_row = make_v10_zero_prior_and_root_cause_item(rng, 1 + 6 * i)
        episodes.append({
            "id": f"dapo10_code_{i:03d}",
            "kind": "python_code",
            "messages": src_row["messages"][:-1],
            "verifier": src_row["verifier"],
            "reference_answer": src_row["messages"][-1]["content"],
        })

    # 4. 20 Advanced RSI Weight-Modification Code Episodes (`SVF`, `Nullspace`, `TIES`, `DGM`)
    for i in range(20):
        v_mod = [0, 1, 3, 4][i % 4]
        src_row = make_v10_advanced_rsi_item(rng, v_mod + 7 * i)
        episodes.append({
            "id": f"dapo10_rsi_{i:03d}",
            "kind": "python_code",
            "messages": src_row["messages"][:-1],
            "verifier": src_row["verifier"],
            "reference_answer": src_row["messages"][-1]["content"],
        })

    rng.shuffle(episodes)
    assert len(episodes) == 96
    return episodes


def build_v10_corpus() -> dict[str, Any]:
    V10_CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261010)

    # 1. Build strictly held-out `36`-task APEX suite + `12`-task `16k..64k` Multi-Needle Long-Context suite FIRST!
    apex36, apex36_code_checks = build_heldout_v10_apex_36()
    n_chk = _verify_executable_code_rows(apex36_code_checks)
    assert n_chk == len(apex36_code_checks), f"Held-out reference code check failed: {n_chk}/{len(apex36_code_checks)}"

    apex36_path = V10_CORPUS_DIR / "heldout_v10_apex_36.jsonl"
    apex36_path.write_text("\n".join(json.dumps(r) for r in apex36) + "\n", encoding="utf-8")

    long12 = build_heldout_long_256k_needles_12()
    long12_path = V10_CORPUS_DIR / "heldout_long_256k_needles_12.jsonl"
    long12_path.write_text("\n".join(json.dumps(r) for r in long12) + "\n", encoding="utf-8")

    # 2. Load stratified `v9` replay (preserving 100% of PIL Vision RPR + 2-turn RPR + `[EFFORT: FLASH]` + Hermes/Cline tools!)
    v9_sft = [json.loads(l) for l in (V9_CORPUS_DIR / "train_v9_sft.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    vis_rows = [r for r in v9_sft if r.get("images")]
    non_vis_v9 = [r for r in v9_sft if not r.get("images")]
    rng.shuffle(non_vis_v9)
    replay_v9 = vis_rows + non_vis_v9[:280]

    # 3. Generate new `v10` Frontier SFT traces across all 5 pillars!
    logic_items = [make_v10_self_discover_logic_item(rng, i) for i in range(156)]
    research_items = [make_v10_realtime_research_citation_item(rng, i) for i in range(150)]
    zp_root_items = [make_v10_zero_prior_and_root_cause_item(rng, i) for i in range(156)]
    rsi_items = [make_v10_advanced_rsi_item(rng, i) for i in range(154)]
    long_sft_items = [make_v10_long_context_sft_item(rng, i) for i in range(64)]

    # Verify 100% of executable Python code items in `python3 -I` BEFORE training!
    all_generated = logic_items + research_items + zp_root_items + rsi_items + long_sft_items
    code_rows = [r for r in all_generated if r.get("verifier", {}).get("kind") == "python_code"]
    n_verified = _verify_executable_code_rows(code_rows)
    assert n_verified == len(code_rows) and n_verified >= 200, f"Expected >=200 verified code items, got {n_verified}/{len(code_rows)}"

    rng.shuffle(all_generated)
    val_new = all_generated[:56]
    train_new = all_generated[56:]

    v9_val = [json.loads(l) for l in (V9_CORPUS_DIR / "val_v9_joint.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    val_joint = _deduplicate_rows(v9_val[:44] + val_new)
    train_sft = _deduplicate_rows(replay_v9 + train_new)
    rng.shuffle(train_sft)

    # 4. Build `240` Contrastive ORPO Pairs (`v9` replay + `v10` citation/logic/non-mutating/path/RSI pairs)
    v9_orpo = [json.loads(l) for l in (V9_CORPUS_DIR / "orpo_v9_pairs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rng.shuffle(v9_orpo)
    v10_orpo_new = [make_v10_orpo_pair(rng, i) for i in range(168)]
    orpo_all = v9_orpo[:72] + v10_orpo_new
    rng.shuffle(orpo_all)

    # 5. Build `96` Challenging `DAPO + Dr. GRPO` Verifiable Episodes (and verify every code reference solution!)
    dapo_96 = build_dapo_hard_episodes_96(rng)
    dapo_code_refs = [
        {"id": ep["id"], "messages": [{"role": "assistant", "content": ep["reference_answer"]}], "verifier": ep["verifier"]}
        for ep in dapo_96
        if ep.get("kind") == "python_code"
    ]
    n_dapo_ver = _verify_executable_code_rows(dapo_code_refs)
    assert n_dapo_ver == len(dapo_code_refs), f"DAPO code ref check failed: {n_dapo_ver}/{len(dapo_code_refs)}"

    # 6. Strict Decontamination & Zero-Leakage Check across ALL Held-Out Suites!
    assert_decontaminated(train_sft + val_joint + orpo_all + dapo_96 + apex36)
    _assert_zero_heldout_leakage(
        all_generated + v10_orpo_new + dapo_96,
        [
            V5_CORPUS_DIR / "heldout_v5_rsi_56.jsonl",
            V9_CORPUS_DIR / "heldout_zero_prior_24.jsonl",
            apex36_path,
            long12_path,
        ],
    )

    (V10_CORPUS_DIR / "train_v10_sft.jsonl").write_text("\n".join(json.dumps(r) for r in train_sft) + "\n", encoding="utf-8")
    (V10_CORPUS_DIR / "val_v10_joint.jsonl").write_text("\n".join(json.dumps(r) for r in val_joint) + "\n", encoding="utf-8")
    (V10_CORPUS_DIR / "orpo_v10_pairs.jsonl").write_text("\n".join(json.dumps(r) for r in orpo_all) + "\n", encoding="utf-8")
    (V10_CORPUS_DIR / "dapo_v10_hard_episodes.jsonl").write_text("\n".join(json.dumps(r) for r in dapo_96) + "\n", encoding="utf-8")

    summary = {
        "train_v10_sft": len(train_sft),
        "val_v10_joint": len(val_joint),
        "orpo_v10_pairs": len(orpo_all),
        "dapo_v10_hard_episodes": len(dapo_96),
        "heldout_v10_apex_36": len(apex36),
        "heldout_long_256k_needles_12": len(long12),
        "long_sft_4k_6k_items": len(long_sft_items),
        "python_unit_verified_count": n_verified + n_dapo_ver + n_chk,
        "vision_items_preserved": len(vis_rows),
    }
    (V10_CORPUS_DIR / "corpus_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("[v10 APEX Corpus Built & Decontaminated]", json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    build_v10_corpus()
