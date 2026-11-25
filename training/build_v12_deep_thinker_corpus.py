"""Master Zero-Leakage Corpus Builder for `RPR v12 DEEP-THINKER` (`LFM3-VL-3.2B-RPR-V12-DEEP-BF16`).

Combines 4 major 2025-2026 small-reasoning-model research breakthroughs:
  1. **Stanford Four Cognitive Habits (`Gandhi et al., 2025`)**: Every deep trace natively practices
     Subgoal Decomposition (`L0`), Backward Chaining & Dual-Perspective Derivation (`L1`), Concrete
     Hand-Traced Toy Execution (`L2`), and Explicit Mid-Thought Error Catching & Selective Backtracking (`L3`).
  2. **Adaptive Thinking Depth (`360..660` tokens on complex reasoning vs `90..160` tokens on simple tool calls)**:
     Replaces `v4..v11`'s short `~120`-token bullet templates so natural `<think>` scales deeply without hitting
     `max_tokens=1024` truncation or premature `</think>` entropy collapse.
  3. **Function-Space Plasticity Replay from `24L` `Base LFM3-VL` (`base_lfm3_teacher_replay.jsonl`)**:
     Injects verified native `LFM3` multi-paragraph `<think>` trajectories in token space with zero `24L/40L`
     weight-space mismatch.
  4. **Length-De-Biased Cognitive `ORPO` (`400` pairs) & Hard Long-Horizon `DAPO/Dr. GRPO` (`200` episodes)**.

Strictly enforces `assert_decontaminated` + `_assert_zero_heldout_leakage` across ALL held-out suites
(`56`, `zp24`, `apex36`, `long12`, `heldout_broad_42`, `heldout_eval_20`, `pilot`, `benchmark_100`) and executes
EVERY synthesized Python function in isolated `python3 -I`!
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

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from training.build_training_data import assert_decontaminated
from training.build_v9_rsi_zero_prior_cline_corpus import V9_CORPUS_DIR
from training.build_v10_apex_corpus import V10_CORPUS_DIR, _assert_zero_heldout_leakage, _xml_call
from training.build_v11_omega_corpus import V11_CORPUS_DIR, make_v11_true_long_kv_item
from training.verifiable_rewards import score_response

V12_CORPUS_DIR = ROOT_DIR / "training" / "data" / "rpr_v12_deep_corpus"

_OPENINGS = (
    "Let's unpack the exact requirements and work backward from the postconditions before writing a single line of output.",
    "Hold on — this specification has subtle boundary and shape invariants, so let's build a Reversible Pyramid plan step by step.",
    "Let's decompose the problem carefully: first lock down the foundational invariants, then derive two independent views, and finally hand-trace a concrete toy case.",
    "Before committing to an answer, let's list every constraint, check where a naive shortcut would fail, and verify on a small concrete example.",
    "Let's reason from first principles here rather than guessing a template: start with exact types and edge conditions, then test a concrete case.",
    "Let's map the dependency pyramid from foundation axioms (`L0`) up to adversarial self-verification (`L3`) so no edge case slips through.",
)

_BACKTRACK_CUES = (
    "Wait — let me pause and test where a rushed implementation would go wrong:",
    "Hold on, let me double-check that step against a tricky edge case before trusting it:",
    "Wait a second — let's try to break our own hypothesis on a degenerate or asymmetric input:",
    "Let's stop and audit that assumption carefully — does it still hold on boundary/rectangular inputs?",
)


def _open_line(rng: random.Random) -> str:
    return rng.choice(_OPENINGS)


def _bt_line(rng: random.Random) -> str:
    return rng.choice(_BACKTRACK_CUES)


# ===========================================================================
# Pillar 1 (`200` Deep Traces): Complex Logic, 3/4-SAT DPLL, Game Trees (`mex`/Minimax),
#                               Quantified Finite Models (`∀x ∃y`), & Multi-Agent Logic
# ===========================================================================


def make_v12_complex_logic_sat_item(rng: random.Random, idx: int) -> dict[str, Any]:
    variant = idx % 8

    if variant == 0:
        # 4-Variable / 5-Variable Exact CNF Model Counting + Backbone Identification (JSON)
        n_vars = 4
        var_names = ["p1", "p2", "p3", "p4"]
        # Plant at least 1 satisfying assignment so ~75% are SAT and ~25% can be UNSAT
        force_unsat = (idx % 5 == 4)
        clauses: list[list[int]] = []
        if force_unsat:
            clauses = [[1, 2], [1, -2], [-1, 3], [-1, -3], [2, 3, 4], [-2, -4]]
        else:
            planted = [rng.choice([False, True]) for _ in range(n_vars)]
            for _ in range(6):
                vs = sorted(rng.sample([1, 2, 3, 4], k=3))
                lits = [v if rng.random() < 0.5 else -v for v in vs]
                if not any((planted[abs(l) - 1] if l > 0 else not planted[abs(l) - 1]) for l in lits):
                    pick = rng.randrange(len(lits))
                    v_id = abs(lits[pick])
                    lits[pick] = v_id if planted[v_id - 1] else -v_id
                if lits not in clauses:
                    clauses.append(lits)

        models: list[tuple[bool, ...]] = []
        for bits in itertools.product([False, True], repeat=n_vars):
            ok = True
            for cl in clauses:
                if not any((bits[abs(l) - 1] if l > 0 else not bits[abs(l) - 1]) for l in cl):
                    ok = False
                    break
            if ok:
                models.append(bits)

        is_sat = len(models) > 0
        backbone_true = [
            var_names[j]
            for j in range(n_vars)
            if is_sat and all(m[j] is True for m in models)
        ]

        def _fmt_cl(cl: list[int]) -> str:
            return "(" + " OR ".join((f"{var_names[l-1]}" if l > 0 else f"NOT {var_names[-l-1]}") for l in cl) + ")"

        cl_lines = "\n".join(f"  C{k+1}: {_fmt_cl(cl)}" for k, cl in enumerate(clauses))
        prompt = (
            f"Consider the boolean CNF formula over `{var_names}` with {len(clauses)} clauses:\n"
            f"{cl_lines}\n\n"
            "Determine:\n"
            "1. `satisfiable` (`true` or `false`),\n"
            "2. `model_count` (exact number of truth assignments in `{False, True}^4` satisfying ALL clauses),\n"
            "3. `backbone_true` (sorted list of variable names that evaluate to `True` in EVERY satisfying model, or `[]` if unsatisfiable or none).\n"
            "Return ONLY JSON `{\"satisfiable\": ..., \"model_count\": ..., \"backbone_true\": [...]}`."
        )

        # Build genuine step-by-step branch-and-prune trace over p1=False vs p1=True!
        def _branch_desc(p1_val: bool, p2_val: bool) -> str:
            sub_m = []
            for p3_v in (False, True):
                for p4_v in (False, True):
                    b = (p1_val, p2_val, p3_v, p4_v)
                    failed_c = next(
                        (k + 1 for k, cl in enumerate(clauses) if not any((b[abs(l) - 1] if l > 0 else not b[abs(l) - 1]) for l in cl)),
                        None,
                    )
                    if failed_c is None:
                        sub_m.append(f"(p3={int(p3_v)},p4={int(p4_v)}):SAT")
                    else:
                        sub_m.append(f"(p3={int(p3_v)},p4={int(p4_v)}):fails C{failed_c}")
            return ", ".join(sub_m)

        think = (
            f"{_open_line(rng)}\n"
            "1. `[L0 Foundation & Postconditions]`: We have 4 boolean variables `(p1, p2, p3, p4)` (`2^4 = 16` total assignments) and "
            f"{len(clauses)} conjunctive clauses `C1..C{len(clauses)}`. A clause is satisfied iff at least ONE literal in it is `True`. "
            "If `model_count == 0`, `satisfiable` must be `false` and `backbone_true` must be `[]` (NOT vacuously all variables — notice the prompt explicitly specifies `[]` if unsatisfiable!).\n"
            f"2. `[L1 Polarity & Unit-Propagation Analysis]`: {_bt_line(rng)} In `NOT p_k`, setting `p_k = True` makes the literal `False`, whereas setting `p_k = False` makes `NOT p_k` `True`. "
            "A common mistake is misreading `NOT p2` as satisfied when `p2 = True`. Let's partition the 16 states cleanly into 4 quadrants `(p1, p2) in {(0,0), (0,1), (1,0), (1,1)}` and check every `(p3, p4)` pair against `C1..C"
            f"{len(clauses)}`:\n"
            f"3. `[L2 Exhaustive Quadrant Trace]`:\n"
            f"   - Quadrant `p1=0, p2=0`: {_branch_desc(False, False)}\n"
            f"   - Quadrant `p1=0, p2=1`: {_branch_desc(False, True)}\n"
            f"   - Quadrant `p1=1, p2=0`: {_branch_desc(True, False)}\n"
            f"   - Quadrant `p1=1, p2=1`: {_branch_desc(True, True)}\n"
            f"4. `[L3 Cross-Check & Backbone Audit]`: Collecting all `SAT` leaves across the 4 quadrants gives exactly `{len(models)}` satisfying assignment(s): "
            f"`{[tuple(int(x) for x in m) for m in models]}`.\n"
            "   Wait — let's double-check `backbone_true`: a variable belongs to `backbone_true` ONLY if `model_count > 0` AND its coordinate is `1` in *every single* satisfying tuple above. "
            f"Checking coordinates `0..3` across all `{len(models)}` models yields `backbone_true = {backbone_true}`. Every clause is verified on every surviving model."
        )
        ans = {"satisfiable": is_sat, "model_count": len(models), "backbone_true": backbone_true}
        return {
            "id": f"v12_logic_{idx:04d}",
            "pillar": "complex_logic_sat",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n{json.dumps(ans)}"}],
            "verifier": {"abstain": True, "forbid_tool_call": True, "expected": ans},
        }

    if variant == 1:
        # Standalone Python DPLL / Unit-Propagation CNF Solver (`python3 -I` verified)
        fn = f"count_cnf_models_dpll_{idx % 19}" if idx >= 16 else "count_cnf_models_dpll"
        prompt = (
            f"Implement `{fn}(num_vars: int, clauses: list[list[int]]) -> tuple[bool, int]` in Python:\n"
            "- Variables are `1..num_vars`; a positive integer `v` means `x_v` is `True`, and `-v` means `x_v` is `False`.\n"
            "- An empty clause `[]` inside `clauses` is immediately unsatisfiable; an empty clause list `[]` has `2**num_vars` models.\n"
            "- Use DPLL unit propagation + recursive branching (multiplying by `2**(unassigned free vars)` when all clauses are satisfied) to return `(is_satisfiable, exact_model_count)` without mutating `clauses`."
        )
        think = (
            f"{_open_line(rng)}\n"
            "1. `[L0 Specification & Boundary Invariants]`: Let's check the two degenerate cases first because naive recursion often mishandles free variables:\n"
            "   - If `clauses == []` (all clauses already satisfied) and `k` variables among `1..num_vars` remain unassigned, EVERY combination of those `k` free variables satisfies the empty conjunction, contributing `2 ** k` models!\n"
            "   - If ANY clause in `clauses` is empty (`len(c) == 0`), that disjunction has zero true literals, so this branch contributes `0` models immediately.\n"
            f"2. `[L1 Unit Propagation & Tautology Handling]`: {_bt_line(rng)} What if a clause contains both `v` and `-v` (a tautological clause like `[1, -1, 2]`)? Under ANY assignment to variable `1`, `[1, -1, 2]` is satisfied! If we preprocess each clause into a `frozenset(c)` and drop any clause where `any(-lit in s for lit in s)`, tautological clauses are simplified cleanly upfront.\n"
            "   - Next, when assigning `var = val` (`lit_true = var if val else -var`, `lit_false = -lit_true`):\n"
            "     * Every clause containing `lit_true` is satisfied and removed.\n"
            "     * Every clause containing `lit_false` loses `lit_false` (`c - {lit_false}`); if that produces `frozenset()`, conflict (`0` models)!\n"
            "   - During unit propagation, if a unit clause `{u}` appears, variable `abs(u)` is forced to `u > 0`. Wait — what if both `{u}` and `{-u}` appear simultaneously? Simplifying with `u` turns `{-u}` into `frozenset()`, immediately triggering conflict and returning `0`!\n"
            "3. `[L2 Concrete Toy Trace]`: Test on `num_vars = 3`, `clauses = [[1, 2], [-1, 2]]`:\n"
            "   - Neither clause is unit initially. Pick variable `1`:\n"
            "     * Branch `x_1 = True` (`lit_true = 1`): `[1, 2]` disappears; `[-1, 2]` shrinks to unit `[2]`. Unit propagation sets `x_2 = True`, leaving `clauses = []` with `x_3` unassigned (`1` free var -> `2^1 = 2` models: `(T,T,F), (T,T,T)`).\n"
            "     * Branch `x_1 = False` (`lit_true = -1`): `[-1, 2]` disappears; `[1, 2]` shrinks to unit `[2]`. Unit propagation sets `x_2 = True`, leaving `x_3` free (`2^1 = 2` models: `(F,T,F), (F,T,T)`).\n"
            "   - Total = `2 + 2 = 4` models (`x_2` must be `True`, `x_1` and `x_3` free -> `4` models). Exact match!\n"
            "4. `[L3 Non-Mutation Check]`: Converting input `clauses` into new `frozenset` objects leaves the caller's `list[list[int]]` completely untouched."
        )
        code = (
            f"def {fn}(num_vars: int, clauses: list[list[int]]) -> tuple[bool, int]:\n"
            "    init_cls: list[frozenset[int]] = []\n"
            "    for raw_c in clauses:\n"
            "        if len(raw_c) == 0:\n"
            "            return False, 0\n"
            "        s = frozenset(int(x) for x in raw_c)\n"
            "        if any(-lit in s for lit in s):\n"
            "            continue\n"
            "        init_cls.append(s)\n\n"
            "    def _assign(cls: list[frozenset[int]], lit_true: int) -> list[frozenset[int]] | None:\n"
            "        lit_false = -lit_true\n"
            "        nxt: list[frozenset[int]] = []\n"
            "        for c in cls:\n"
            "            if lit_true in c:\n"
            "                continue\n"
            "            if lit_false in c:\n"
            "                rem = c - {lit_false}\n"
            "                if not rem:\n"
            "                    return None\n"
            "                nxt.append(rem)\n"
            "            else:\n"
            "                nxt.append(c)\n"
            "        return nxt\n\n"
            "    def _dpll(cls: list[frozenset[int]], assigned_count: int) -> int:\n"
            "        cur = cls\n"
            "        cur_assigned = assigned_count\n"
            "        while True:\n"
            "            if not cur:\n"
            "                return 1 << (num_vars - cur_assigned)\n"
            "            unit_lit = next((next(iter(c)) for c in cur if len(c) == 1), None)\n"
            "            if unit_lit is None:\n"
            "                break\n"
            "            res = _assign(cur, unit_lit)\n"
            "            if res is None:\n"
            "                return 0\n"
            "            cur = res\n"
            "            cur_assigned += 1\n"
            "        shortest = min(cur, key=len)\n"
            "        pivot_var = abs(next(iter(shortest)))\n"
            "        total = 0\n"
            "        for chosen_lit in (pivot_var, -pivot_var):\n"
            "            sub = _assign(cur, chosen_lit)\n"
            "            if sub is not None:\n"
            "                total += _dpll(sub, cur_assigned + 1)\n"
            "        return total\n\n"
            "    cnt = _dpll(init_cls, 0)\n"
            "    return (cnt > 0), cnt"
        )
        test = (
            "orig = [[1, 2], [-1, 2]]\n"
            f"assert {fn}(3, orig) == (True, 4)\n"
            "assert orig == [[1, 2], [-1, 2]]\n"
            f"assert {fn}(2, [[1], [-1]]) == (False, 0)\n"
            f"assert {fn}(4, []) == (True, 16)\n"
            f"assert {fn}(3, [[1, -1], [2]]) == (True, 4)\n"
        )
        return {
            "id": f"v12_logic_{idx:04d}",
            "pillar": "complex_logic_sat",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
        }

    if variant == 2:
        # Impartial Subtraction / Split Game Sprague-Grundy (`mex`) Exact Analysis (JSON)
        moves = rng.choice([(1, 3, 4), (1, 2, 5), (2, 3, 5), (1, 4, 5)])
        n_pile = rng.randint(11, 16)
        g = [0] * (n_pile + 1)
        for s in range(1, n_pile + 1):
            reach = {g[s - m] for m in moves if s - m >= 0}
            mex = 0
            while mex in reach:
                mex += 1
            g[s] = mex
        p2_pile = rng.randint(4, 8)
        xor_g = g[n_pile] ^ g[p2_pile]
        win_moves_single = sorted([m for m in moves if n_pile - m >= 0 and g[n_pile - m] == 0])
        prompt = (
            f"Two players play an impartial normal-play token game (`last valid move wins`) with allowed single-pile removal set `M = {list(moves)}`:\n"
            f"- On a turn, a player chooses ONE pile and removes `m in {list(moves)}` tokens (`m <= pile_size`).\n"
            f"Compute:\n"
            f"1. `grundy_n` — the Sprague-Grundy value `G({n_pile})` of a single pile of `{n_pile}` tokens,\n"
            f"2. `winning_removals_single` — sorted list of removals `m in {list(moves)}` from a single pile of `{n_pile}` that leave a losing (`G == 0`) state (`[]` if `G({n_pile}) == 0`),\n"
            f"3. `two_pile_first_wins` — boolean (`true`/`false`) whether the first player has a winning strategy from the two-pile state `({n_pile}, {p2_pile})`.\n"
            "Return ONLY JSON `{\"grundy_n\": ..., \"winning_removals_single\": [...], \"two_pile_first_wins\": ...}`."
        )
        step_trace = ", ".join(f"G({k})={g[k]}" for k in range(n_pile + 1))
        think = (
            f"{_open_line(rng)}\n"
            f"1. `[L0 Normal-Play Sprague-Grundy Axioms]`: Under normal play (a player unable to move loses), `G(0) = 0`, and for `s >= 1`, `G(s) = mex({{G(s - m) : m in {list(moves)}, s - m >= 0}})`, where `mex(S)` is the smallest non-negative integer NOT in `S`.\n"
            f"2. `[L1 Multi-Pile Bouton/Grundy Theorem]`: For two independent piles `({n_pile}, {p2_pile})` where each turn modifies exactly one pile, the combined Grundy value is the bitwise XOR `G({n_pile}) ^ G({p2_pile})`. First player wins iff `G({n_pile}) ^ G({p2_pile}) != 0`.\n"
            f"   {_bt_line(rng)} Let's make sure we don't confuse `mex` with `max + 1`, and check `s < min(M)` carefully (for instance if `M = (2, 3, 5)`, `G(1)` has NO legal moves so `mex(empty) = 0`, NOT `1`!).\n"
            f"3. `[L2 Bottom-Up Table `G(0)..G({n_pile})`]`: Let's compute every state sequentially and double-check transitions:\n"
            f"   - `{step_trace}`.\n"
            f"   Let's explicitly verify `s = {n_pile}`: legal removals `m in {list(moves)}` reach states `{[n_pile - m for m in moves if n_pile - m >= 0]}` with Grundy values `{[g[n_pile - m] for m in moves if n_pile - m >= 0]}`. "
            f"The smallest non-negative integer absent from that set is `G({n_pile}) = {g[n_pile]}`.\n"
            f"4. `[L3 Winning Move & Two-Pile Audit]`: A removal `m` from `{n_pile}` is winning on a single pile iff `G({n_pile} - m) == 0`, giving `winning_removals_single = {win_moves_single}` "
            f"(sanity check: non-empty iff `G({n_pile}) > 0`, which holds since `{g[n_pile] > 0}` == `{len(win_moves_single) > 0}`). "
            f"For two piles `({n_pile}, {p2_pile})`, `G({n_pile}) ^ G({p2_pile}) = {g[n_pile]} ^ {g[p2_pile]} = {xor_g}`, so `two_pile_first_wins = {xor_g != 0}`."
        )
        ans = {
            "grundy_n": g[n_pile],
            "winning_removals_single": win_moves_single,
            "two_pile_first_wins": (xor_g != 0),
        }
        return {
            "id": f"v12_logic_{idx:04d}",
            "pillar": "complex_logic_sat",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n{json.dumps(ans)}"}],
            "verifier": {"abstain": True, "forbid_tool_call": True, "expected": ans},
        }

    if variant == 3:
        # Quantified First-Order Finite Model Checking (`∀x ∈ D ∃y ∈ D : R(x, y)`) + Counterexample Witness
        dom = [1, 2, 3, 4, 5]
        mod_p = rng.choice([5, 6, 7])
        target_r = rng.randint(0, mod_p - 1)
        # R(x, y) iff (x * y + x) % mod_p == target_r
        witness_map = {x: [y for y in dom if (x * y + x) % mod_p == target_r] for x in dom}
        forall_exists = all(len(witness_map[x]) > 0 for x in dom)
        exists_forall = any(all((x * y + x) % mod_p == target_r for x in dom) for y in dom)
        failing_x = sorted([x for x in dom if len(witness_map[x]) == 0])
        pair_count = sum(len(v) for v in witness_map.values())
        prompt = (
            f"Let finite domain `D = {dom}` and binary relation `R(x, y)` hold iff `(x * y + x) mod {mod_p} == {target_r}`.\n"
            "Evaluate over `x, y in D`:\n"
            "1. `forall_x_exists_y`: `∀x ∈ D, ∃y ∈ D : R(x, y)` (`true`/`false`),\n"
            "2. `exists_y_forall_x`: `∃y ∈ D, ∀x ∈ D : R(x, y)` (`true`/`false`),\n"
            "3. `true_pair_count`: total number of ordered pairs `(x, y) ∈ D × D` where `R(x, y)` holds,\n"
            "4. `counterexample_x`: sorted list of `x ∈ D` that have NO `y ∈ D` satisfying `R(x, y)` (`[]` if `forall_x_exists_y` is `true`).\n"
            "Return ONLY JSON `{\"forall_x_exists_y\": ..., \"exists_y_forall_x\": ..., \"true_pair_count\": ..., \"counterexample_x\": [...]}`."
        )
        rows_desc = "\n".join(f"   - `x={x}` (`x*(y+1) mod {mod_p} == {target_r}`): matching `y in D` -> `{witness_map[x]}`" for x in dom)
        think = (
            f"{_open_line(rng)}\n"
            f"1. `[L0 Quantifier Order & Algebraic Factoring]`: Notice `(x * y + x) = x * (y + 1) (mod {mod_p})`. "
            "Also keep the quantifier order strictly distinct:\n"
            "   - `∀x ∃y R(x, y)` means EVERY row `x` in the relation table is non-empty (`counterexample_x == []`).\n"
            "   - `∃y ∀x R(x, y)` is much stronger: a SINGLE fixed column `y` must appear in the witness list of *every* `x in D`!\n"
            f"2. `[L1 Trap Check]`: {_bt_line(rng)} Note `D = {dom}`, so `y` ranges over `1..5` (`y + 1` ranges over `2..6`), NOT `0..{mod_p - 1}`! "
            f"Even when `gcd(x, {mod_p}) == 1`, the unique modular solution for `y + 1 (mod {mod_p})` might correspond to `y = 0` or `y >= 6`, which is OUTSIDE `D = {dom}`! So we MUST evaluate every `x in {dom}` explicitly over `y in {dom}` rather than assuming modular invertibility implies a solution in `D`.\n"
            f"3. `[L2 Full `5x5` Relation Table Trace]`:\n"
            f"{rows_desc}\n"
            f"4. `[L3 Verification of All 4 Outputs]`:\n"
            f"   - Row witness counts: `{[len(witness_map[x]) for x in dom]}`, summing to `true_pair_count = {pair_count}`.\n"
            f"   - Rows with empty witness list: `counterexample_x = {failing_x}`, so `forall_x_exists_y = {forall_exists}`.\n"
            f"   - Common intersection across all rows `cap_{{x in D}} W(x)`: checking each `y in {dom}` across all 5 lists gives `exists_y_forall_x = {exists_forall}`."
        )
        ans = {
            "forall_x_exists_y": forall_exists,
            "exists_y_forall_x": exists_forall,
            "true_pair_count": pair_count,
            "counterexample_x": failing_x,
        }
        return {
            "id": f"v12_logic_{idx:04d}",
            "pillar": "complex_logic_sat",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n{json.dumps(ans)}"}],
            "verifier": {"abstain": True, "forbid_tool_call": True, "expected": ans},
        }

    if variant == 4:
        # 2-SAT Implication SCC Solver in Python (`python3 -I` verified)
        fn = f"solve_2sat_assignment_{idx % 17}" if idx >= 16 else "solve_2sat_assignment"
        prompt = (
            f"Implement `{fn}(num_vars: int, clauses: list[tuple[int, int]]) -> list[bool] | None` in Python:\n"
            "- Variables are `1..num_vars` (`+v` means `x_v`, `-v` means `not x_v`).\n"
            "- Build the `2 * num_vars` implication graph (`(a OR b)` adds `~a -> b` and `~b -> a`) and run Tarjan or Kosaraju SCCs.\n"
            "- Return `None` if any variable `v` and `-v` lie in the same SCC; otherwise return `[x_1, ..., x_num_vars]` (`list[bool]` of length `num_vars`) satisfying every clause."
        )
        think = (
            f"{_open_line(rng)}\n"
            "1. `[L0 2-SAT Implication & Topological Polarity Law]`: Each binary clause `(a OR b)` is logically equivalent to BOTH `(~a -> b)` and `(~b -> a)`. "
            "Wait — what about a unit clause `(a, a)`? `~a -> a` (twice), forcing any path from `~a` to reach `a`, so `a` MUST lie later than `~a` in topological order!\n"
            f"2. `[L1 Tarjan vs Kosaraju Component Ordering Trap]`: {_bt_line(rng)} This is the #1 bug in 2-SAT implementations:\n"
            "   - In **Kosaraju** (pass 1 finishing order + pass 2 on transpose graph), SCC IDs `comp[u]` are assigned in **topological order** (`u -> v` implies `comp[u] <= comp[v]`), so we set `x_i = (comp[pos_i] > comp[neg_i])`.\n"
            "   - In **Tarjan** (popping SCCs on DFS post-order), SCC IDs are assigned in **reverse topological order** (`u -> v` implies `tarjan_id[u] >= tarjan_id[v]`), so `x_i = (tarjan_id[pos_i] < tarjan_id[neg_i])`! Reversing `<` and `>` produces the exact negation of a valid model!\n"
            "3. `[L2 Concrete Toy Check on Forced Unit Clause `[(1, 1)]`]`: Let's test our sign rule on `num_vars = 1, clauses = [(1, 1)]` (`x_1` MUST be `True`):\n"
            "   - Implication edge: `-1 -> +1` (no edge `+1 -> -1`).\n"
            "   - Using iterative/recursive Kosaraju: topological order has `-1` (`comp = 0`) before `+1` (`comp = 1`).\n"
            "   - Rule `comp[+1] > comp[-1]` gives `1 > 0` -> `True`! Clauses satisfied! If we had written `<`, it would have returned `[False]`, failing `(1, 1)`!\n"
            "4. `[L3 Indexing & Edge Cases]`: Map literal `lit` (`v` in `1..n`) to node `2*(v-1)` for `+v` and `2*(v-1)+1` for `-v`, so `neg(node) = node ^ 1`. Handle `num_vars == 0` -> `[]` cleanly."
        )
        code = (
            f"def {fn}(num_vars: int, clauses: list[tuple[int, int]]) -> list[bool] | None:\n"
            "    if num_vars == 0:\n"
            "        return []\n"
            "    n_nodes = 2 * num_vars\n"
            "    adj: list[list[int]] = [[] for _ in range(n_nodes)]\n"
            "    radj: list[list[int]] = [[] for _ in range(n_nodes)]\n\n"
            "    def _node(lit: int) -> int:\n"
            "        v = abs(lit) - 1\n"
            "        return (2 * v) if lit > 0 else (2 * v + 1)\n\n"
            "    for a, b in clauses:\n"
            "        na, nb = _node(a), _node(b)\n"
            "        adj[na ^ 1].append(nb)\n"
            "        radj[nb].append(na ^ 1)\n"
            "        adj[nb ^ 1].append(na)\n"
            "        radj[na].append(nb ^ 1)\n\n"
            "    vis = [False] * n_nodes\n"
            "    order: list[int] = []\n"
            "    for s in range(n_nodes):\n"
            "        if vis[s]:\n"
            "            continue\n"
            "        stack: list[tuple[int, int]] = [(s, 0)]\n"
            "        vis[s] = True\n"
            "        while stack:\n"
            "            u, ei = stack[-1]\n"
            "            if ei < len(adj[u]):\n"
            "                stack[-1] = (u, ei + 1)\n"
            "                w = adj[u][ei]\n"
            "                if not vis[w]:\n"
            "                    vis[w] = True\n"
            "                    stack.append((w, 0))\n"
            "            else:\n"
            "                stack.pop()\n"
            "                order.append(u)\n\n"
            "    comp = [-1] * n_nodes\n"
            "    cid = 0\n"
            "    for s in reversed(order):\n"
            "        if comp[s] != -1:\n"
            "            continue\n"
            "        q = [s]\n"
            "        comp[s] = cid\n"
            "        while q:\n"
            "            u = q.pop()\n"
            "            for w in radj[u]:\n"
            "                if comp[w] == -1:\n"
            "                    comp[w] = cid\n"
            "                    q.append(w)\n"
            "        cid += 1\n\n"
            "    ans: list[bool] = []\n"
            "    for i in range(num_vars):\n"
            "        if comp[2 * i] == comp[2 * i + 1]:\n"
            "            return None\n"
            "        ans.append(comp[2 * i] > comp[2 * i + 1])\n"
            "    return ans"
        )
        test = (
            "cls = [(1, 2), (-1, 2), (-2, 3)]\n"
            f"m = {fn}(3, cls)\n"
            "assert m is not None and len(m) == 3\n"
            "assert all((m[abs(a)-1] if a > 0 else not m[abs(a)-1]) or (m[abs(b)-1] if b > 0 else not m[abs(b)-1]) for a, b in cls)\n"
            f"assert {fn}(1, [(1, 1), (-1, -1)]) is None\n"
        )
        return {
            "id": f"v12_logic_{idx:04d}",
            "pillar": "complex_logic_sat",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
        }

    if variant == 5:
        # Three-Role Truth-Teller / Liar / Alternator Deduction System (JSON)
        roles = ["Truth", "Liar", "XorSpy"]
        valid_worlds: list[tuple[str, str, str]] = []
        target_world = tuple(rng.sample(roles, k=3))
        # Construct 3 statements uniquely or cleanly constraining (A, B, C)
        # Truth: statement True; Liar: statement False; XorSpy: statement1 True and statement2 False!
        # Let's define crisp boolean rules for Knight (always true), Knave (always false), Spy (can be either, exactly 1 Knight, 1 Knave, 1 Spy):
        # Wait: make it 100% deterministic over all 6 permutations of ("Knight", "Knave", "Spy") or all 8 of ("Knight", "Knave")^3!
        # Let's use 4 agents P, Q, R, S in {"Knight", "Knave"}^4 (`16` worlds) so truth values of statements are 100% deterministic functions of the world!
        planted_4 = [rng.choice([True, False]) for _ in range(4)]  # True=Knight, False=Knave
        # Statements:
        # P says: "Exactly {k_true} of the four of us are Knights." (if P is Knight, k_true == sum; if Knave, pick wrong k)
        sum_true = sum(planted_4)
        p_claim_k = sum_true if planted_4[0] else ((sum_true + 1) % 5)
        # Q says: "P and R have the same type."
        q_stmt_true = (planted_4[0] == planted_4[2])
        if not planted_4[1]:
            # Flip Q's claim to "P and R have different types" so Q is consistent with planted_4!
            q_same = False
        else:
            q_same = True
        # Wait: let's evaluate ALL 16 worlds under fixed concrete claims!
        q_word = "the same type" if (planted_4[1] == (planted_4[0] == planted_4[2])) else "different types"
        r_word = "at least one of Q or S is a Knave" if (planted_4[2] == ((not planted_4[1]) or (not planted_4[3]))) else "both Q and S are Knights"
        s_word = "P is a Knight" if (planted_4[3] == planted_4[0]) else "P is a Knave"

        consistent: list[tuple[bool, bool, bool, bool]] = []
        for w in itertools.product([False, True], repeat=4):
            p, q, r, s = w
            c_p = (sum(w) == p_claim_k)
            c_q = (p == r) if q_word == "the same type" else (p != r)
            c_r = ((not q) or (not s)) if "Knave" in r_word else (q and s)
            c_s = p if s_word == "P is a Knight" else (not p)
            if (p == c_p) and (q == c_q) and (r == c_r) and (s == c_s):
                consistent.append(w)

        knights_unique = [
            nm
            for j, nm in enumerate(["P", "Q", "R", "S"])
            if consistent and all(w[j] is True for w in consistent)
        ]
        prompt = (
            "Four islanders `P, Q, R, S` are each either a `Knight` (every statement is `True`) or a `Knave` (every statement is `False`):\n"
            f"- `P` says: \"Exactly {p_claim_k} of the four of us are Knights.\"\n"
            f"- `Q` says: \"`P` and `R` are of {q_word}.\"\n"
            f"- `R` says: \"{r_word[0].upper() + r_word[1:]}.\"\n"
            f"- `S` says: \"`{s_word}`.\"\n\n"
            "Find the exact `consistent_world_count` across all `16` assignments in `{Knave, Knight}^4`, and `always_knights` (sorted list of names among `[\"P\", \"Q\", \"R\", \"S\"]` that are `Knight` in EVERY consistent world).\n"
            "Return ONLY JSON `{\"consistent_world_count\": ..., \"always_knights\": [...]}`."
        )
        p_case_true = (
            f"If `P = Knight` (`p = 1`), `P`'s statement requires `p + q + r + s == {p_claim_k}`, AND `S`'s statement (`{s_word}`) forces `S = {'Knight' if s_word == 'P is a Knight' else 'Knave'}`!"
        )
        p_case_false = (
            f"If `P = Knave` (`p = 0`), `P`'s statement must be `False` (`p + q + r + s != {p_claim_k}`), AND `S`'s statement (`{s_word}`) forces `S = {'Knave' if s_word == 'P is a Knight' else 'Knight'}`!"
        )
        world_strs = [str({nm: ("Knight" if w[j] else "Knave") for j, nm in enumerate(["P", "Q", "R", "S"])}) for w in consistent]
        think = (
            f"{_open_line(rng)}\n"
            "1. `[L0 Biconditional Formalization]`: An islander `X` (`x in {0, 1}` for `{Knave, Knight}`) is consistent iff `x == truth_value(claim_X)`. Let's write all four biconditionals explicitly:\n"
            f"   - `p <-> (p + q + r + s == {p_claim_k})`\n"
            f"   - `q <-> ({'p == r' if q_word == 'the same type' else 'p != r'})`\n"
            f"   - `r <-> ({'(not q) or (not s)' if 'Knave' in r_word else 'q and s'})`\n"
            f"   - `s <-> ({'p' if s_word == 'P is a Knight' else 'not p'})`\n"
            f"2. `[L1 Case Split on `P` via `S` Coupling]`: {_bt_line(rng)} Notice `s` is determined immediately by `p`! That collapses `16` states to just `8` `(p, q, r)` triples:\n"
            f"   - {p_case_true}\n"
            f"   - {p_case_false}\n"
            "3. `[L2 Exhaustive Verification of `(q, r)` in Both Branches]`: Checking all 4 `(q, r) in {(0,0), (0,1), (1,0), (1,1)}` under `p = 1` and under `p = 0` against all four biconditionals leaves the exact consistent world set:\n"
            f"   - Surviving worlds (`{len(consistent)}`): `{world_strs}`.\n"
            "4. `[L3 Re-Verify Every Islander's Statement on Each Surviving World]`: Re-evaluating `P, Q, R, S` on each surviving assignment confirms `x == claim_X` for all 4 agents, giving "
            f"`consistent_world_count = {len(consistent)}` and `always_knights = {knights_unique}`."
        )
        ans = {"consistent_world_count": len(consistent), "always_knights": knights_unique}
        return {
            "id": f"v12_logic_{idx:04d}",
            "pillar": "complex_logic_sat",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n{json.dumps(ans)}"}],
            "verifier": {"abstain": True, "forbid_tool_call": True, "expected": ans},
        }

    if variant == 6:
        # Exact Minimax with Alpha-Beta / Depth-3 Partizan Game Tree Value (JSON)
        leaves = [rng.randint(-9, 14) for _ in range(8)]
        # Max at root (depth 0, branching 2), Min at depth 1 (branching 2), Max at depth 2 (branching 2 -> 8 leaves)
        d2 = [max(leaves[2 * k], leaves[2 * k + 1]) for k in range(4)]
        d1 = [min(d2[0], d2[1]), min(d2[2], d2[3])]
        root_val = max(d1[0], d1[1])
        opt_branch = "L" if d1[0] >= d1[1] else "R"  # tie-break L
        # Force strict inequality so tie-break is unambiguous
        if d1[0] == d1[1]:
            leaves[0] += 3
            leaves[1] += 3
            d2 = [max(leaves[2 * k], leaves[2 * k + 1]) for k in range(4)]
            d1 = [min(d2[0], d2[1]), min(d2[2], d2[3])]
            root_val = max(d1[0], d1[1])
            opt_branch = "L" if d1[0] > d1[1] else "R"

        # Count leaves visited by left-to-right alpha-beta pruning!
        visited = 0

        def _ab(depth: int, node_idx: int, alpha: int, beta: int, is_max: bool) -> int:
            nonlocal visited
            if depth == 3:
                visited += 1
                return leaves[node_idx]
            if is_max:
                v = -10**9
                for ch in (0, 1):
                    v = max(v, _ab(depth + 1, 2 * node_idx + ch, alpha, beta, False))
                    alpha = max(alpha, v)
                    if alpha >= beta:
                        break
                return v
            v = 10**9
            for ch in (0, 1):
                v = min(v, _ab(depth + 1, 2 * node_idx + ch, alpha, beta, True))
                beta = min(beta, v)
                if alpha >= beta:
                    break
            return v

        ab_root = _ab(0, 0, -10**9, 10**9, True)
        assert ab_root == root_val
        prompt = (
            f"A complete binary zero-sum game tree of depth `3` (`8` terminal leaves left-to-right: `{leaves}`) has:\n"
            "- Depth `0` (root): `MAX` player chooses `'L'` or `'R'`,\n"
            "- Depth `1`: `MIN` player chooses `'L'` or `'R'`,\n"
            "- Depth `2`: `MAX` player chooses `'L'` or `'R'`.\n"
            "Compute:\n"
            "1. `minimax_value` at the root,\n"
            "2. `optimal_root_move` (`\"L\"` or `\"R\"`),\n"
            "3. `alphabeta_leaves_evaluated` under standard left-to-right alpha-beta pruning (`prune when alpha >= beta`).\n"
            "Return ONLY JSON `{\"minimax_value\": ..., \"optimal_root_move\": ..., \"alphabeta_leaves_evaluated\": ...}`."
        )
        think = (
            f"{_open_line(rng)}\n"
            "1. `[L0 Alternating Minimax Structure]`: Depth `2` has four `MAX` nodes over leaf pairs `(0,1), (2,3), (4,5), (6,7)`; Depth `1` has two `MIN` nodes; Depth `0` is `MAX`.\n"
            f"2. `[L1 Exact Backward Induction]`: Let's compute pure minimax first:\n"
            f"   - Depth-2 `MAX` nodes: `max({leaves[0]}, {leaves[1]}) = {d2[0]}`, `max({leaves[2]}, {leaves[3]}) = {d2[1]}`, `max({leaves[4]}, {leaves[5]}) = {d2[2]}`, `max({leaves[6]}, {leaves[7]}) = {d2[3]}`.\n"
            f"   - Depth-1 `MIN` nodes: `left_min = min({d2[0]}, {d2[1]}) = {d1[0]}`, `right_min = min({d2[2]}, {d2[3]}) = {d1[1]}`.\n"
            f"   - Root `MAX`: `max({d1[0]}, {d1[1]}) = {root_val}`, choosing `\"{opt_branch}\"`.\n"
            f"3. `[L2 Left-to-Right Alpha-Beta Pruning Trace]`: {_bt_line(rng)} Alpha-beta pruning at `alpha >= beta` depends on exact bound updates at EVERY ancestor — miscounting by 1 leaf happens if you prune inside a `MAX` child when `alpha < beta` or forget `MIN` passes `beta` down to its second `MAX` child!\n"
            f"   Let's trace leaf-by-leaf with `(alpha, beta)`:\n"
            f"   - Left subtree (`root alpha=-inf, beta=+inf` -> `L_min alpha=-inf, beta=+inf`):\n"
            f"     * `L_max_0`: evaluates `leaf[0]={leaves[0]}` (`alpha={leaves[0]}` < `+inf`), then `leaf[1]={leaves[1]}` -> returns `{d2[0]}`. (`2` leaves). Now `L_min` sets `beta = {d2[0]}`!\n"
            f"     * `L_max_1` (`alpha=-inf, beta={d2[0]}`): evaluates `leaf[2]={leaves[2]}` (`alpha={leaves[2]}`). Does `{leaves[2]} >= {d2[0]}`? `{'YES -> prunes leaf[3]!' if leaves[2] >= d2[0] else f'NO -> must also evaluate leaf[3]={leaves[3]}!'}`\n"
            f"     * `L_min` returns `{d1[0]}`, so Root updates `alpha = {d1[0]}`.\n"
            f"   - Right subtree (`R_min` receives `alpha={d1[0]}, beta=+inf`):\n"
            f"     * `R_max_0` (`alpha={d1[0]}, beta=+inf`): `beta=+inf` so neither `leaf[4]={leaves[4]}` nor `leaf[5]={leaves[5]}` can prune (`alpha >= +inf` is false) -> evaluates both, returning `{d2[2]}`. Now `R_min` updates `beta = min(+inf, {d2[2]}) = {d2[2]}`.\n"
            f"     * Check at `R_min`: is `alpha ({d1[0]}) >= beta ({d2[2]})`? "
            + (
                f"`YES ({d1[0]} >= {d2[2]})`! The entire `R_max_1` subtree (`leaf[6], leaf[7]`) is beta-pruned immediately!"
                if d1[0] >= d2[2]
                else f"`NO ({d1[0]} < {d2[2]})`, so `R_max_1` (`alpha={d1[0]}, beta={d2[2]}`) evaluates `leaf[6]={leaves[6]}`, sets `alpha=max({d1[0]}, {leaves[6]})={max(d1[0], leaves[6])}`, and `{'prunes leaf[7]!' if max(d1[0], leaves[6]) >= d2[2] else 'evaluates leaf[7] as well!'}`"
            )
            + f"\n4. `[L3 Final Tally]`: Counting every evaluated terminal node above gives `alphabeta_leaves_evaluated = {visited}`, and alpha-beta root value `{ab_root}` matches pure minimax `{root_val}`."
        )
        ans = {
            "minimax_value": root_val,
            "optimal_root_move": opt_branch,
            "alphabeta_leaves_evaluated": visited,
        }
        return {
            "id": f"v12_logic_{idx:04d}",
            "pillar": "complex_logic_sat",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n{json.dumps(ans)}"}],
            "verifier": {"abstain": True, "forbid_tool_call": True, "expected": ans},
        }

    # variant == 7: Proper Graph k-Coloring Chromatic Polynomial / Exact Valid Colorings on Small Constraint Graph (Python)
    fn = f"count_proper_k_colorings_{idx % 17}" if idx >= 16 else "count_proper_k_colorings"
    prompt = (
        f"Implement `{fn}(num_nodes: int, edges: list[tuple[int, int]], k: int) -> int` in Python:\n"
        "- Nodes are `0..num_nodes-1`; `k >= 0` colors `0..k-1` are available.\n"
        "- If any self-loop `(u, u)` exists in `edges` or (`num_nodes > 0` and `k == 0`), return `0`; if `num_nodes == 0`, return `1`.\n"
        "- Use backtracking with forward degree/saturation ordering (connected components multiply independently!) to return the exact number of proper `k`-colorings."
    )
    think = (
        f"{_open_line(rng)}\n"
        "1. `[L0 Edge Cases & Self-Loop Trap]`: Let's check degenerate inputs first:\n"
        "   - `num_nodes == 0`: empty product has `1` valid coloring even if `k == 0`.\n"
        "   - `num_nodes > 0` and `k <= 0`: `0` colorings.\n"
        "   - Any self-loop `u == v` in `edges` requires `color[u] != color[u]`, which is impossible -> immediately return `0`!\n"
        f"2. `[L1 Component Independence + Symmetry Breaking]`: {_bt_line(rng)} Disconnected components have completely independent color choices (an isolated vertex multiplies the count by `k`!). Moreover, in each non-trivial connected component of size `>= 1`, fixing the first vertex `v_0` of that component to color `0` and multiplying the component's count by `k` cuts work by `k`x immediately!\n"
        "3. `[L2 Concrete Toy Trace on Triangle `K_3` with `k = 3`]`: \n"
        "   - `num_nodes = 3`, `edges = [(0,1), (1,2), (0,2)]`, `k = 3`.\n"
        "   - Fix `color[0] = 0` (factor `k = 3`). Vertex `1` can take `{1, 2}` (`2` choices); for each, vertex `2` has `1` remaining color -> `3 * 2 * 1 = 6`, matching `k*(k-1)*(k-2) = 6`.\n"
        "   - Add an isolated vertex `3` (`num_nodes = 4`): component `{3}` contributes `* 3` -> `6 * 3 = 18`.\n"
        "4. `[L3 Duplicate-Edge & Non-Mutation Check]`: Build `adj: list[set[int]]` deduplicating undirected edges cleanly without mutating `edges`."
    )
    code = (
        f"def {fn}(num_nodes: int, edges: list[tuple[int, int]], k: int) -> int:\n"
        "    if num_nodes == 0:\n"
        "        return 1\n"
        "    if k <= 0:\n"
        "        return 0\n"
        "    adj: list[set[int]] = [set() for _ in range(num_nodes)]\n"
        "    for u, v in edges:\n"
        "        if u == v:\n"
        "            return 0\n"
        "        adj[u].add(v)\n"
        "        adj[v].add(u)\n\n"
        "    seen = [False] * num_nodes\n"
        "    total_ways = 1\n"
        "    for start in range(num_nodes):\n"
        "        if seen[start]:\n"
        "            continue\n"
        "        comp: list[int] = []\n"
        "        q = [start]\n"
        "        seen[start] = True\n"
        "        while q:\n"
            "            cur = q.pop()\n"
            "            comp.append(cur)\n"
            "            for nb in adj[cur]:\n"
            "                if not seen[nb]:\n"
            "                    seen[nb] = True\n"
            "                    q.append(nb)\n"
        "        if len(comp) == 1:\n"
        "            total_ways *= k\n"
        "            continue\n"
        "        comp.sort(key=lambda x: -len(adj[x]))\n"
        "        color = [-1] * num_nodes\n"
        "        color[comp[0]] = 0\n\n"
        "        def _dfs(pos: int) -> int:\n"
        "            if pos == len(comp):\n"
        "                return 1\n"
        "            u = comp[pos]\n"
        "            forbidden = {color[nb] for nb in adj[u] if color[nb] != -1}\n"
        "            ways = 0\n"
        "            for c in range(k):\n"
        "                if c not in forbidden:\n"
        "                    color[u] = c\n"
        "                    ways += _dfs(pos + 1)\n"
        "                    color[u] = -1\n"
        "            return ways\n\n"
        "        comp_ways = k * _dfs(1)\n"
        "        if comp_ways == 0:\n"
        "            return 0\n"
        "        total_ways *= comp_ways\n"
        "    return total_ways"
    )
    test = (
        f"assert {fn}(3, [(0, 1), (1, 2), (0, 2)], 3) == 6\n"
        f"assert {fn}(4, [(0, 1), (1, 2), (0, 2)], 3) == 18\n"
        f"assert {fn}(3, [(0, 1), (1, 1)], 4) == 0\n"
        f"assert {fn}(0, [], 0) == 1\n"
    )
    return {
        "id": f"v12_logic_{idx:04d}",
        "pillar": "complex_logic_sat",
        "images": [],
        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n```python\n{code}\n```"}],
        "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
    }


# ===========================================================================
# Pillar 2 (`220` Deep Traces): Weight Surgery, Nullspace/SVD/DoRA/Muon/TIES,
#                               `DGM` UCB Archive Evolution, & `DFT`/`Dr. GRPO` RSI
# ===========================================================================


def make_v12_weight_surgery_and_rsi_item(rng: random.Random, idx: int) -> dict[str, Any]:
    variant = idx % 8

    if variant == 0:
        # Right-Nullspace Safe Weight Update `W + dW @ (I - V_r^T V_r)` (`python3 -I` verified)
        fn = f"nullspace_guarded_weight_step_{idx % 19}" if idx >= 16 else "nullspace_guarded_weight_step"
        prompt = (
            f"Implement `{fn}(W: np.ndarray, dW: np.ndarray, X_anchor: np.ndarray, sv_tol: float = 1e-6) -> np.ndarray` in NumPy without mutating inputs:\n"
            "- `W` and `dW` have shape `(d_out, d_in)`; `X_anchor` has shape `(n_samples, d_in)` representing protected anchor activations.\n"
            "- Compute thin SVD of `X_anchor` in `float64` (`U, S, Vt = np.linalg.svd(..., full_matrices=False)`), keep rows `V_keep = Vt[S > sv_tol]` spanning the active input rowspace, and project `dW` onto its orthogonal complement: `dW_safe = dW - (dW @ V_keep.T) @ V_keep`.\n"
            "- Return `(W + dW_safe)` in `float64` so `(W_new @ X_anchor.T)` is preserved on all singular directions `> sv_tol`."
        )
        think = (
            f"{_open_line(rng)}\n"
            "1. `[L0 Linear-Algebraic Invariant]`: A linear layer computes `Y = X_anchor @ W.T` (or `W @ x` for column `x` of length `d_in`). For `W_new = W + dW_safe` to leave `W_new @ x == W @ x` unchanged for every `x` in the rowspace of `X_anchor`, `dW_safe` must annihilate `rowspace(X_anchor)` on the **right** (`dW_safe @ V_keep.T == 0`).\n"
            f"2. `[L1 Rectangular Shape & Left-vs-Right Projection Check]`: {_bt_line(rng)} Suppose `W` is `(5, 4)` (`d_out=5, d_in=4`) and `X_anchor` is `(2, 4)` (`n_samples=2, d_in=4`):\n"
            "   - `U, S, Vt = np.linalg.svd(X_anchor.astype(np.float64), full_matrices=False)` gives `Vt` of shape `(min(2, 4), 4) = (2, 4)`.\n"
            "   - Filtering `V_keep = Vt[S > sv_tol]` gives shape `(r, 4)` with orthonormal rows (`V_keep @ V_keep.T = I_r`).\n"
            "   - Wait — what if `X_anchor` is all zeros (`r = 0`, `V_keep` has shape `(0, 4)`)? Let's trace `(dW @ V_keep.T) @ V_keep`: `(5, 4) @ (4, 0) -> (5, 0)`, `@ (0, 4) -> (5, 4)` of zeros! Even better, checking `if V_keep.shape[0] == 0: return W64 + dW64` makes `r = 0` explicit and crystal clear.\n"
            "   - And notice why writing `(I - V_keep @ V_keep.T)` would be WRONG: `V_keep @ V_keep.T` is `(r, r)`, NOT `(d_in, d_in)`! The `(d_in, d_in)` projector is `V_keep.T @ V_keep`, and computing `(dW64 @ V_keep.T) @ V_keep` avoids forming `(d_in, d_in)` altogether (`O(r * d_out * d_in)`).\n"
            "3. `[L2 Toy Verification on `d_out=3, d_in=2` with `X_anchor = [[4.0, 0.0]]`]`: \n"
            "   - `S = [4.0] > 1e-6`, `V_keep = [[1.0, 0.0]]` (or `[[-1.0, 0.0]]`; note sign cancels in `V_keep.T @ V_keep`!).\n"
            "   - For `dW = [[7.0, 9.0], [2.0, 5.0], [-1.0, 3.0]]`, `(dW @ V_keep.T) @ V_keep` extracts column 0 `[[7, 0], [2, 0], [-1, 0]]`, so `dW_safe = [[0, 9], [0, 5], [0, 3]]`.\n"
            "   - Check anchor output: `(W + dW_safe) @ [4, 0]^T = W @ [4, 0]^T + [0, 0, 0]^T = W @ [4, 0]^T`! Exact zero drift!\n"
            "4. `[L3 Non-Mutation Check]`: Cast `W`, `dW`, `X_anchor` with `np.asarray(..., dtype=np.float64)` and construct new arrays without `+=`."
        )
        code = (
            "import numpy as np\n\n"
            f"def {fn}(W: np.ndarray, dW: np.ndarray, X_anchor: np.ndarray, sv_tol: float = 1e-6) -> np.ndarray:\n"
            "    w64 = np.array(W, dtype=np.float64, copy=True)\n"
            "    dw64 = np.array(dW, dtype=np.float64, copy=True)\n"
            "    x64 = np.asarray(X_anchor, dtype=np.float64)\n"
            "    if x64.size == 0:\n"
            "        return w64 + dw64\n"
            "    _, s_vals, vt = np.linalg.svd(x64, full_matrices=False)\n"
            "    v_keep = vt[s_vals > sv_tol]\n"
            "    if v_keep.shape[0] == 0:\n"
            "        return w64 + dw64\n"
            "    dw_safe = dw64 - (dw64 @ v_keep.T) @ v_keep\n"
            "    return w64 + dw_safe"
        )
        test = (
            "import numpy as np\n"
            "rng = np.random.default_rng(11)\n"
            "W = rng.standard_normal((5, 4))\n"
            "dW = rng.standard_normal((5, 4))\n"
            "X = rng.standard_normal((2, 4))\n"
            f"W_new = {fn}(W, dW, X)\n"
            "np.testing.assert_allclose(W_new @ X.T, W @ X.T, atol=1e-8)\n"
            f"np.testing.assert_allclose({fn}(W, dW, np.zeros((2, 4))), W + dW, atol=1e-9)\n"
        )
        return {
            "id": f"v12_rsi_{idx:04d}",
            "pillar": "rsi_and_weight_surgery",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
        }

    if variant == 1:
        # DoRA Weight-Decomposed Low-Rank Adaptation Forward Merge `m[:, None] * (V / max(||V||_row, eps))`
        fn = f"merge_dora_weight_{idx % 19}" if idx >= 16 else "merge_dora_weight"
        prompt = (
            f"Implement `{fn}(W0: np.ndarray, lora_B: np.ndarray, lora_A: np.ndarray, magnitude: np.ndarray, scale: float = 1.0, eps: float = 1e-8) -> np.ndarray` in NumPy without mutating inputs:\n"
            "- `W0` has shape `(d_out, d_in)`, `lora_B` has shape `(d_out, r)`, `lora_A` has shape `(r, d_in)`, and `magnitude` has shape `(d_out,)`.\n"
            "- Form directional matrix `V = W0 + scale * (lora_B @ lora_A)` in `float64`.\n"
            "- Normalize each output row `i` of `V` by `max(||V[i, :]||_2, eps)` and multiply row `i` by `magnitude[i]`, returning the `(d_out, d_in)` `float64` merged weight."
        )
        think = (
            f"{_open_line(rng)}\n"
            "1. `[L0 DoRA Decomposition Equation]`: In Liu et al. (DoRA, ICML 2024), each output channel (row `i in 0..d_out-1` of `W` in `y = W x`) has its own scalar magnitude `m_i` and unit row direction `V[i, :] / ||V[i, :]||_2`.\n"
            f"2. `[L1 Row-Norm Axis & Broadcasting Audit]`: {_bt_line(rng)} Let's test shapes on rectangular `d_out = 5, d_in = 3, r = 2` so we CANNOT mix up `axis=1` vs `axis=0`:\n"
            "   - `lora_B @ lora_A` is `(5, 2) @ (2, 3) -> (5, 3)`, matching `W0` `(5, 3)`.\n"
            "   - Row L2 norms require summing across `d_in` (`axis=1`): `row_norms = np.linalg.norm(V, ord=2, axis=1, keepdims=True)` -> shape `(5, 1)`!\n"
            "   - Wait — if someone wrote `axis=0`, `row_norms` would have shape `(1, 3)` and `magnitude[:, None]` `(5, 1)` would either broadcast wrong or divide columns instead of rows! Checking `(5, 3)` catches that immediately.\n"
            "   - Also note the specification says `max(||V[i, :]||_2, eps)` (clamping with `np.maximum(row_norms, eps)`), NOT `row_norms + eps` (which shifts unit norms by `1e-8` when `eps` is larger like `1e-3`!). Let's use `np.maximum(row_norms, eps)` so exact identity holds when `magnitude == ||W0||_row` and `scale == 0`.\n"
            "3. `[L2 Toy Hand Trace on `2x2` Matrix]`: Take `W0 = [[3.0, 4.0], [0.0, 0.0]]`, `scale = 0.0`, `magnitude = [10.0, 5.0]`, `eps = 1e-8`:\n"
            "   - `V = [[3.0, 4.0], [0.0, 0.0]]`.\n"
            "   - Row 0 norm `= 5.0` -> `10.0 * [3/5, 4/5] = [6.0, 8.0]` (row norm `= 10.0`).\n"
            "   - Row 1 norm `= 0.0 -> clamped to 1e-8` -> `5.0 * [0.0, 0.0] / 1e-8 = [0.0, 0.0]` (finite, zero `NaN`!).\n"
            "4. `[L3 Final Shape Check]`: `(magnitude64[:, None] / denom) * V` has shape `(5, 1) * (5, 3) -> (5, 3)` in `float64`."
        )
        code = (
            "import numpy as np\n\n"
            f"def {fn}(\n"
            "    W0: np.ndarray,\n"
            "    lora_B: np.ndarray,\n"
            "    lora_A: np.ndarray,\n"
            "    magnitude: np.ndarray,\n"
            "    scale: float = 1.0,\n"
            "    eps: float = 1e-8,\n"
            ") -> np.ndarray:\n"
            "    w64 = np.asarray(W0, dtype=np.float64)\n"
            "    b64 = np.asarray(lora_B, dtype=np.float64)\n"
            "    a64 = np.asarray(lora_A, dtype=np.float64)\n"
            "    m64 = np.asarray(magnitude, dtype=np.float64).reshape(-1, 1)\n"
            "    v_dir = w64 + float(scale) * (b64 @ a64)\n"
            "    denom = np.maximum(np.linalg.norm(v_dir, axis=1, keepdims=True), float(eps))\n"
            "    return (m64 / denom) * v_dir"
        )
        test = (
            "import numpy as np\n"
            "W0 = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])\n"
            "B = np.zeros((2, 2)); A = np.zeros((2, 3))\n"
            "m = np.array([10.0, 5.0])\n"
            f"out = {fn}(W0, B, A, m, scale=1.0, eps=1e-6)\n"
            "np.testing.assert_allclose(out, np.array([[6.0, 8.0, 0.0], [0.0, 0.0, 0.0]]), atol=1e-9)\n"
        )
        return {
            "id": f"v12_rsi_{idx:04d}",
            "pillar": "rsi_and_weight_surgery",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
        }

    if variant == 2:
        # Darwin Gödel Machine (`DGM`) UCB1 + Non-Regressing Lineage Archive Selector (JSON analytical RSI)
        c_exp = rng.choice([0.4, 0.5, 0.6])
        total_rollouts = rng.choice([16, 25, 36])  # exact integer sqrt(ln N) or clean log!
        ln_tot = math.log(total_rollouts)
        nodes = [
            {"id": "gen_01", "score": round(0.62 + 0.01 * (idx % 5), 2), "children": 4, "regressed": False},
            {"id": "gen_02", "score": round(0.84 + 0.01 * (idx % 3), 2), "children": 1, "regressed": True},
            {"id": "gen_03", "score": round(0.71 + 0.01 * (idx % 4), 2), "children": 1, "regressed": False},
            {"id": "gen_04", "score": round(0.76 + 0.01 * (idx % 4), 2), "children": 3, "regressed": False},
        ]

        def _ucb(nd: dict[str, Any]) -> float:
            return nd["score"] + c_exp * math.sqrt(ln_tot / (1.0 + nd["children"]))

        elig = [nd for nd in nodes if not nd["regressed"]]
        scored = [(nd["id"], round(_ucb(nd), 4)) for nd in elig]
        best_id, best_ucb = max(scored, key=lambda kv: (kv[1], kv[0]))
        prompt = (
            "In a Darwin Gödel Machine (`DGM`) self-improving agent archive, each candidate node has `{id, score, children, regressed}`:\n"
            f"```json\n{json.dumps(nodes)}\n```\n"
            "Selection rules:\n"
            "1. Exclude any node with `regressed == true` (hard anchor-safety gate).\n"
            f"2. Among eligible nodes, compute `ucb(node) = score + {c_exp} * sqrt(ln({total_rollouts}) / (1 + children))`.\n"
            "3. Select `selected_id` maximizing `ucb(node)` and report `selected_ucb` rounded to `4` decimal places (`eligible_count` = number of non-regressed nodes).\n"
            "Return ONLY JSON `{\"selected_id\": ..., \"selected_ucb\": ..., \"eligible_count\": ...}`."
        )
        ucb_lines = "\n".join(
            f"   - `{nd['id']}`: `score={nd['score']}`, `1 + children = {1 + nd['children']}` -> `{nd['score']} + {c_exp} * sqrt({ln_tot:.6f} / {1 + nd['children']}) = {_ucb(nd):.6f}` (rounded: `{round(_ucb(nd), 4)}`)"
            for nd in elig
        )
        think = (
            f"{_open_line(rng)}\n"
            "1. `[L0 Hard Anchor Gate Before UCB]`: Look closely at `gen_02` — it has the highest raw `score` (`"
            f"{nodes[1]['score']}`), which is a classic distractor trap! Because `gen_02.regressed == true`, the anchor-safety gate disqualifies `gen_02` BEFORE UCB scoring (`eligible_count = {len(elig)}`).\n"
            f"2. `[L1 Exact Logarithm & Denominator Check]`: {_bt_line(rng)} Note the formula uses natural logarithm `ln({total_rollouts}) = {ln_tot:.6f}` (NOT `log10` or `log2`) and denominator `(1 + children)` inside `sqrt(...)`, NOT `sqrt(ln N) / (1 + children)`! Let's check parentheses carefully: `sqrt(ln({total_rollouts}) / (1 + children))` divides inside the square root.\n"
            f"3. `[L2 Numerical Evaluation of All `{len(elig)}` Eligible Nodes]`:\n"
            f"{ucb_lines}\n"
            f"4. `[L3 Ranking & Rounding Verification]`: Comparing the exact values `{scored}` confirms `{best_id}` wins with `selected_ucb = {best_ucb}` (`eligible_count = {len(elig)}`)."
        )
        ans = {"selected_id": best_id, "selected_ucb": best_ucb, "eligible_count": len(elig)}
        return {
            "id": f"v12_rsi_{idx:04d}",
            "pillar": "advanced_rsi_dgm_svf",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n{json.dumps(ans)}"}],
            "verifier": {"abstain": True, "forbid_tool_call": True, "expected": ans},
        }

    if variant == 3:
        # TIES-Merging (`Trim, Elect Sign, Disjoint Merge`) Task-Vector Surgery (`python3 -I` verified)
        fn = f"ties_disjoint_merge_{idx % 19}" if idx >= 16 else "ties_disjoint_merge"
        prompt = (
            f"Implement `{fn}(W_base: np.ndarray, task_deltas: list[np.ndarray], trim_quantile: float = 0.5, scaling: float = 1.0) -> np.ndarray` in NumPy without mutating inputs:\n"
            "- If `task_deltas` is empty, return a `float64` copy of `W_base`.\n"
            "- **Step 1 (Trim):** For each delta `D_m` in `task_deltas` (cast to `float64`), compute scalar threshold `thr_m = np.quantile(np.abs(D_m), trim_quantile)` and zero out entries with `np.abs(D_m) < thr_m`.\n"
            "- **Step 2 (Elect Sign):** Sum the trimmed deltas `S_tot = sum_m D_trim_m` and set elected sign `E = np.sign(S_tot)` (`+1.0`, `-1.0`, or `0.0`).\n"
            "- **Step 3 (Disjoint Mean):** At each coordinate `(i, j)`, average ONLY the trimmed deltas `D_trim_m[i, j]` whose sign matches `E[i, j]` and is non-zero (`np.sign(D_trim_m) == E` and `E != 0`, `0.0` where count is `0`), then return `W_base + scaling * merged_delta` in `float64`."
        )
        think = (
            f"{_open_line(rng)}\n"
            "1. `[L0 TIES-Merging 3-Step Invariants (Yadav et al., NeurIPS 2023)]`: Standard averaging of task vectors suffers from magnitude dilution (averaging zeros in!) and sign interference (`+2` and `-1` partially canceling). Let's check all 3 steps carefully.\n"
            f"2. `[L1 Subtle Zero-Sign & Disjoint-Count Trap]`: {_bt_line(rng)} Suppose at coordinate `(0, 1)` two trimmed deltas are `+3.0` and `-3.0`:\n"
            "   - `S_tot[0, 1] = 0.0`, so `E[0, 1] = np.sign(0.0) = 0.0`.\n"
            "   - If we ONLY checked `np.sign(D_trim_m) == E` WITHOUT requiring `E != 0`, then a third delta whose trimmed entry was `0.0` would have `np.sign(0.0) == 0.0 == E` (`count = 1`, `0.0 / 1 = 0.0`), whereas if NO delta was zero, `count = 0` (`0.0 / 0` -> `NaN` if unguarded!).\n"
            "   - Guarding `agree_mask = (np.sign(stacked) == E[None, ...]) & (E[None, ...] != 0.0)` and dividing with `np.where(counts > 0, agreed_sum / np.maximum(counts, 1), 0.0)` guarantees zero `NaN`s for ALL shapes (`1D`, `2D`, `3D`)!\n"
            "3. `[L2 Concrete Toy Trace]`: Let `W_base = [10.0, 20.0]`, `D_1 = [4.0, -0.1]`, `D_2 = [2.0, -5.0]`, `D_3 = [-1.0, -3.0]`, `trim_quantile = 0.5`, `scaling = 1.0`:\n"
            "   - `|D_1| = [4.0, 0.1]`, median `= 2.05` -> `D_trim_1 = [4.0, 0.0]`.\n"
            "   - `|D_2| = [2.0, 5.0]`, median `= 3.5` -> `D_trim_2 = [0.0, -5.0]`.\n"
            "   - `|D_3| = [1.0, 3.0]`, median `= 2.0` -> `D_trim_3 = [0.0, -3.0]`.\n"
            "   - `S_tot = [4.0, -8.0]`, `E = [+1.0, -1.0]`.\n"
            "   - Coord 0: only `D_trim_1 = 4.0` has sign `+1` (`count = 1`) -> disjoint mean `4.0` (NOT `4.0 / 3`!).\n"
            "   - Coord 1: `D_trim_2 = -5.0` and `D_trim_3 = -3.0` have sign `-1` (`count = 2`) -> disjoint mean `-4.0`.\n"
            "   - Output: `[14.0, 16.0]`. Every step verified!\n"
            "4. `[L3 Non-Mutation & Arbitrary-Rank Tensor Check]`: Using `axis=0` over `stacked = np.stack(...)` works seamlessly for `1D`, `2D`, or higher-rank weight tensors."
        )
        code = (
            "import numpy as np\n\n"
            f"def {fn}(\n"
            "    W_base: np.ndarray,\n"
            "    task_deltas: list[np.ndarray],\n"
            "    trim_quantile: float = 0.5,\n"
            "    scaling: float = 1.0,\n"
            ") -> np.ndarray:\n"
            "    base64 = np.array(W_base, dtype=np.float64, copy=True)\n"
            "    if not task_deltas:\n"
            "        return base64\n"
            "    trimmed_list: list[np.ndarray] = []\n"
            "    for d in task_deltas:\n"
            "        d64 = np.asarray(d, dtype=np.float64)\n"
            "        thr = float(np.quantile(np.abs(d64), trim_quantile))\n"
            "        trimmed_list.append(np.where(np.abs(d64) >= thr, d64, 0.0))\n"
            "    stacked = np.stack(trimmed_list, axis=0)\n"
            "    elected = np.sign(np.sum(stacked, axis=0))\n"
            "    agree = (np.sign(stacked) == elected[None, ...]) & (elected[None, ...] != 0.0)\n"
            "    counts = np.sum(agree, axis=0)\n"
            "    agreed_sum = np.sum(np.where(agree, stacked, 0.0), axis=0)\n"
            "    merged = np.where(counts > 0, agreed_sum / np.maximum(counts, 1), 0.0)\n"
            "    return base64 + float(scaling) * merged"
        )
        test = (
            "import numpy as np\n"
            "wb = np.array([[10.0, 20.0]])\n"
            "ds = [np.array([[4.0, -0.1]]), np.array([[2.0, -5.0]]), np.array([[-1.0, -3.0]])]\n"
            f"res = {fn}(wb, ds, trim_quantile=0.5, scaling=1.0)\n"
            "np.testing.assert_allclose(res, np.array([[14.0, 16.0]]), atol=1e-9)\n"
            f"np.testing.assert_allclose({fn}(wb, []), wb, atol=1e-9)\n"
        )
        return {
            "id": f"v12_rsi_{idx:04d}",
            "pillar": "rsi_and_weight_surgery",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
        }

    if variant == 4:
        # Dynamic Fine-Tuning (`DFT`) & `Dr. GRPO` Unbiased Token-Normalized Policy Gradient Math (JSON)
        probs = [round(x, 2) for x in rng.sample([0.02, 0.05, 0.10, 0.25, 0.40, 0.50, 0.80], k=4)]
        dft_floor = 0.05
        dft_losses = [max(p, dft_floor) * (-math.log(p)) for p in probs]
        mean_dft = round(sum(dft_losses) / len(dft_losses), 4)
        std_ce = round(sum(-math.log(p) for p in probs) / len(probs), 4)
        rewards = [1.0, 1.0, 0.2, -0.2]
        mean_r = sum(rewards) / 4.0
        dr_adv_0 = round(rewards[0] - mean_r, 4)
        prompt = (
            f"Given a 4-token target trajectory with model token probabilities `p = {probs}` and Bounded Dynamic Fine-Tuning (`DFT`) floor `eps_floor = {dft_floor}`:\n"
            "- Standard SFT mean NLL is `mean_ce = (1/4) * sum_t (-ln(p_t))`.\n"
            "- Bounded DFT mean loss is `mean_dft = (1/4) * sum_t (max(p_t, eps_floor) * (-ln(p_t)))`.\n"
            f"- Separately, in a 4-rollout `Dr. GRPO` group with rewards `R = {rewards}` (unbiased group-mean baseline WITHOUT std division!), compute rollout-0 advantage `dr_grpo_adv0 = R[0] - mean(R)`.\n"
            "Report all three values rounded to `4` decimal places in JSON `{\"mean_ce\": ..., \"mean_dft\": ..., \"dr_grpo_adv0\": ...}`."
        )
        tok_breakdown = "\n".join(
            f"   - `p_{k}={p}`: `-ln({p}) = {-math.log(p):.6f}`, `w_{k} = max({p}, {dft_floor}) = {max(p, dft_floor):.2f}` -> `w * ce = {max(p, dft_floor)*(-math.log(p)):.6f}`"
            for k, p in enumerate(probs)
        )
        think = (
            f"{_open_line(rng)}\n"
            "1. `[L0 Equations & Distinction Between GRPO vs Dr. GRPO]`: Standard `GRPO` divides `(R_i - mean(R))` by `std(R)`, which biases optimization toward near-zero-variance easy/impossible prompts and penalizes token length. `Dr. GRPO` (Liu et al., 2025) omits `std(R)` division, so `dr_grpo_adv0 = R[0] - mean(R)` directly!\n"
            f"2. `[L1 Natural Log & Floor Check]`: {_bt_line(rng)} Ensure we use natural logarithm `ln(p_t)` (`math.log`), NOT base-2, and apply `max(p_t, {dft_floor})` token-by-token BEFORE multiplying `-ln(p_t)` (NOT `max(mean(p), {dft_floor}) * mean_ce`!).\n"
            f"3. `[L2 Per-Token Numerical Trace]`:\n"
            f"{tok_breakdown}\n"
            f"   - `sum(-ln p_t) = {sum(-math.log(p) for p in probs):.6f}` -> `mean_ce = {std_ce}`.\n"
            f"   - `sum(w_t * (-ln p_t)) = {sum(dft_losses):.6f}` -> `mean_dft = {mean_dft}`.\n"
            f"4. `[L3 Dr. GRPO Advantage Check]`: `mean({rewards}) = {sum(rewards)} / 4 = {mean_r:.4f}`, so `dr_grpo_adv0 = {rewards[0]} - {mean_r:.4f} = {dr_adv_0}`."
        )
        ans = {"mean_ce": std_ce, "mean_dft": mean_dft, "dr_grpo_adv0": dr_adv_0}
        return {
            "id": f"v12_rsi_{idx:04d}",
            "pillar": "advanced_rsi_dgm_svf",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n{json.dumps(ans)}"}],
            "verifier": {"abstain": True, "forbid_tool_call": True, "expected": ans},
        }

    if variant == 5:
        # Quintic Newton-Schulz Orthogonalization Step (`Muon` Optimizer Weight-Space Step in NumPy)
        fn = f"muon_newton_schulz_orth_{idx % 19}" if idx >= 16 else "muon_newton_schulz_orth"
        prompt = (
            f"Implement `{fn}(G: np.ndarray, steps: int = 5, eps: float = 1e-7) -> np.ndarray` in NumPy without mutating `G`:\n"
            "- Normalize 2D matrix `G` in `float64` by `||G||_F + eps`: `X = G / (np.linalg.norm(G, 'fro') + eps)`.\n"
            "- If `rows > cols`, transpose `X` first so `X @ X.T` has the smaller dimension `(min(m, n), min(m, n))`, run `steps` quintic Newton-Schulz iterations with coefficients `(a, b, c) = (3.4445, -4.7750, 2.0315)`:\n"
            "  `A = X @ X.T`; `X = a * X + (b * A + c * (A @ A)) @ X`,\n"
            "  and transpose back at the end so the returned `float64` array has exact shape `G.shape`."
        )
        think = (
            f"{_open_line(rng)}\n"
            "1. `[L0 Muon Quintic Polynomial & Rectangular Shape Law]`: For `X` of shape `(m, n)`:\n"
            "   - If `m <= n`, `A = X @ X.T` is `(m, m)`, `B = b * A + c * (A @ A)` is `(m, m)`, and `B @ X` is `(m, m) @ (m, n) -> (m, n)`.\n"
            "   - If `m > n` (`tall` matrix, e.g. `(5, 3)`), transposing `X = X.T` first makes `X` `(3, 5)`, so `A = X @ X.T` is `(3, 3)` instead of `(5, 5)` (faster AND numerically identical because odd polynomials commute with transpose: `p(X^T) = p(X)^T`!).\n"
            f"2. `[L1 Zero-Matrix & `steps == 0` Check]`: {_bt_line(rng)} What if `G` is all zeros `np.zeros((4, 5))`? Dividing by `0.0 + eps` yields `X = 0.0` (finite, zero `NaN`!), and every polynomial iteration keeps `X = 0.0`! And if we forgot to transpose back when `m > n`, a `(5, 4)` input would return `(4, 5)` and fail rectangular tests!\n"
            "3. `[L2 Toy Singular-Value Fixed-Point Check]`: For a single singular value `s0 = 0.5`, iteration 1 maps `s1 = 0.5 * (3.4445 - 4.7750 * 0.25 + 2.0315 * 0.0625) = 0.5 * (3.4445 - 1.19375 + 0.12696875) = 1.18886`, rapidly pulling all non-zero singular values into `[0.7, 1.2]` near semi-orthogonality.\n"
            "4. `[L3 Non-Mutation & Return Type]`: Work on `np.asarray(G, dtype=np.float64)` and return `X.T if tall else X`."
        )
        code = (
            "import numpy as np\n\n"
            f"def {fn}(G: np.ndarray, steps: int = 5, eps: float = 1e-7) -> np.ndarray:\n"
            "    g64 = np.asarray(G, dtype=np.float64)\n"
            "    tall = g64.shape[0] > g64.shape[1]\n"
            "    x = g64.T.copy() if tall else g64.copy()\n"
            "    x = x / (np.linalg.norm(x, ord='fro') + float(eps))\n"
            "    a, b, c = 3.4445, -4.7750, 2.0315\n"
            "    for _ in range(steps):\n"
            "        aa = x @ x.T\n"
            "        x = a * x + (b * aa + c * (aa @ aa)) @ x\n"
            "    return x.T if tall else x"
        )
        test = (
            "import numpy as np\n"
            "rng = np.random.default_rng(3)\n"
            "for shp in ((5, 3), (3, 5)):\n"
            "    G = rng.standard_normal(shp)\n"
            f"    Q = {fn}(G, steps=5)\n"
            "    assert Q.shape == shp and np.all(np.isfinite(Q))\n"
            "    s = np.linalg.svd(Q, compute_uv=False)\n"
            "    assert np.all((s > 0.65) & (s < 1.25))\n"
            f"assert np.allclose({fn}(np.zeros((4, 3))), 0.0)\n"
        )
        return {
            "id": f"v12_rsi_{idx:04d}",
            "pillar": "rsi_and_weight_surgery",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
        }

    if variant == 6:
        # Fisher-Information Diagonal Precision-Weighted Checkpoint Merging (`EWC` / Fisher Soup in NumPy)
        fn = f"fisher_precision_merge_{idx % 19}" if idx >= 16 else "fisher_precision_merge"
        prompt = (
            f"Implement `{fn}(theta_a: dict[str, np.ndarray], fisher_a: dict[str, np.ndarray], theta_b: dict[str, np.ndarray], fisher_b: dict[str, np.ndarray], prior_damping: float = 1e-6) -> dict[str, np.ndarray]` in Python/NumPy without mutating inputs:\n"
            "- For every parameter key `k` in `theta_a`, clamp negative Fisher noise via `fa = np.maximum(fisher_a[k], 0.0)` and `fb = np.maximum(fisher_b[k], 0.0)` in `float64`.\n"
            "- Compute the posterior MAP precision-weighted merge:\n"
            "  `theta_star[k] = ((fa + prior_damping) * theta_a[k] + (fb + prior_damping) * theta_b[k]) / (fa + fb + 2.0 * prior_damping)`\n"
            "  so coordinates where both Fisher diagonals are `0` cleanly fall back to the exact `0.5 * (theta_a[k] + theta_b[k])` midpoint!"
        )
        think = (
            f"{_open_line(rng)}\n"
            "1. `[L0 Bayesian Laplace / Fisher Merging Law (Matena & Raffel, NeurIPS 2022)]`: Under diagonal Gaussian posteriors `N(theta_a, diag(F_a)^-1)` and `N(theta_b, diag(F_b)^-1)` with isotropic prior precision `prior_damping`, the joint precision is `(fa + prior_damping) + (fb + prior_damping) = fa + fb + 2 * prior_damping`.\n"
            f"2. `[L1 Zero-Fisher Fallback & Dict Type Trap]`: {_bt_line(rng)} Two classic mistakes to avoid:\n"
            "   - First, `theta_a` and `theta_b` are Python `dict[str, np.ndarray]` — we MUST iterate `{k: ... for k in theta_a}` rather than applying arithmetic directly to `dict`.\n"
            "   - Second, why is `+ 2.0 * prior_damping` in the denominator (instead of `+ 1.0 * prior_damping`)? Check `fa = 0, fb = 0`: numerator is `prior_damping * (theta_a + theta_b)`, so dividing by `2.0 * prior_damping` cancels `prior_damping` to give `0.5 * (theta_a + theta_b)`! If the denominator had `1.0 * prior_damping`, `fa = fb = 0` would return `1.0 * (theta_a + theta_b)` (double the weights!!).\n"
            "3. `[L2 Toy Numerical Check]`: Take `theta_a = [2.0, 10.0]`, `fa = [3.0, 0.0]`, `theta_b = [6.0, 20.0]`, `fb = [1.0, 0.0]`, `prior_damping = 1e-8`:\n"
            "   - Coord 0: `(3 * 2 + 1 * 6) / 4 = 12 / 4 = 3.0` (pulled 3:1 toward `theta_a`).\n"
            "   - Coord 1: `(1e-8 * 10 + 1e-8 * 20) / (2e-8) = 15.0` (exact midpoint).\n"
            "4. `[L3 Non-Mutation Check]`: Build fresh `float64` arrays per key."
        )
        code = (
            "import numpy as np\n\n"
            f"def {fn}(\n"
            "    theta_a: dict[str, np.ndarray],\n"
            "    fisher_a: dict[str, np.ndarray],\n"
            "    theta_b: dict[str, np.ndarray],\n"
            "    fisher_b: dict[str, np.ndarray],\n"
            "    prior_damping: float = 1e-6,\n"
            ") -> dict[str, np.ndarray]:\n"
            "    damp = float(prior_damping)\n"
            "    merged: dict[str, np.ndarray] = {}\n"
            "    for k in theta_a:\n"
            "        ta = np.asarray(theta_a[k], dtype=np.float64)\n"
            "        tb = np.asarray(theta_b[k], dtype=np.float64)\n"
            "         wa = np.maximum(np.asarray(fisher_a[k], dtype=np.float64), 0.0) + damp\n"
            "        wb = np.maximum(np.asarray(fisher_b[k], dtype=np.float64), 0.0) + damp\n"
            "        merged[k] = (wa * ta + wb * tb) / (wa + wb)\n"
            "    return merged"
        ).replace("         wa =", "        wa =")
        test = (
            "import numpy as np\n"
            "ta = {'w': np.array([2.0, 10.0])}; fa = {'w': np.array([3.0, -0.5])}\n"
            "tb = {'w': np.array([6.0, 20.0])}; fb = {'w': np.array([1.0, 0.0])}\n"
            f"m = {fn}(ta, fa, tb, fb, prior_damping=1e-9)\n"
            "np.testing.assert_allclose(m['w'], np.array([3.0, 15.0]), atol=1e-6)\n"
        )
        return {
            "id": f"v12_rsi_{idx:04d}",
            "pillar": "rsi_and_weight_surgery",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
        }

    # variant == 7: Rank-k Truncated SVD Energy & Frobenius Approximation Error (`JSON` analytical SVF/LoRA math)
    s_vals = sorted([rng.randint(2, 12) for _ in range(4)], reverse=True)
    if s_vals[1] == s_vals[2]:
        s_vals[1] += 1
        s_vals.sort(reverse=True)
    tot_sq = sum(s * s for s in s_vals)
    k_keep = 2
    kept_sq = sum(s * s for s in s_vals[:k_keep])
    rem_sq = sum(s * s for s in s_vals[k_keep:])
    energy_ratio = round(kept_sq / tot_sq, 4)
    fro_err = round(math.sqrt(rem_sq), 4)
    spec_err = float(s_vals[k_keep])
    prompt = (
        f"A rectangular weight matrix `W` has exact non-zero singular values `S = {s_vals}`.\n"
        f"Suppose we approximate `W` by its optimal rank-`{k_keep}` truncated SVD `W_{k_keep}` (Eckart-Young-Mirsky theorem). Compute:\n"
        f"1. `energy_retained`: `||W_{k_keep}||_F^2 / ||W||_F^2` rounded to `4` decimal places,\n"
        f"2. `frobenius_error`: `||W - W_{k_keep}||_F` rounded to `4` decimal places,\n"
        f"3. `spectral_error`: operator 2-norm error `||W - W_{k_keep}||_2`.\n"
        "Return ONLY JSON `{\"energy_retained\": ..., \"frobenius_error\": ..., \"spectral_error\": ...}`."
    )
    think = (
        f"{_open_line(rng)}\n"
        f"1. `[L0 Eckart-Young-Mirsky Norm Identities]`: Because `U` and `V` are orthogonal, singular values govern BOTH Frobenius and spectral norms:\n"
        f"   - `||W||_F^2 = sum_i s_i^2`, and `W - W_{k_keep}` has residual singular values `S[{k_keep}:] = {s_vals[k_keep:]}`.\n"
        f"   - `||W - W_{k_keep}||_F = sqrt(sum_{{i >= {k_keep}}} s_i^2)`.\n"
        f"   - `||W - W_{k_keep}||_2 = s_{k_keep}` (the largest *discarded* singular value, `0`-indexed `S[{k_keep}] = {s_vals[k_keep]}`, NOT `sum` of discarded singular values!).\n"
        f"2. `[L1 Common Trap Audit]`: {_bt_line(rng)} People frequently compute `energy_retained` using unsquared `sum(s_i)` (nuclear norm!) instead of squared Frobenius energy `sum(s_i^2)`, or confuse `||W - W_k||_2` with `||W - W_k||_F`. Let's square every singular value explicitly.\n"
        f"3. `[L2 Exact Arithmetic]`: \n"
        f"   - Squares: `{[s*s for s in s_vals]}`.\n"
        f"   - Total squared Frobenius norm: `{' + '.join(str(s*s) for s in s_vals)} = {tot_sq}`.\n"
        f"   - Top-{k_keep} squared norm: `{s_vals[0]**2} + {s_vals[1]**2} = {kept_sq}` -> `energy_retained = {kept_sq} / {tot_sq} = {kept_sq / tot_sq:.6f}` -> `{energy_ratio}`.\n"
        f"   - Residual squared norm: `{s_vals[2]**2} + {s_vals[3]**2} = {rem_sq}` -> `frobenius_error = sqrt({rem_sq}) = {math.sqrt(rem_sq):.6f}` -> `{fro_err}`.\n"
        f"   - Largest discarded singular value (`0`-indexed `S[2]`): `spectral_error = {spec_err}`.\n"
        f"4. `[L3 Partition Sanity Check]`: `{kept_sq} + {rem_sq} == {tot_sq}` (`energy_retained + rem/tot == 1.0`). Verified!"
    )
    ans = {"energy_retained": energy_ratio, "frobenius_error": fro_err, "spectral_error": spec_err}
    return {
        "id": f"v12_rsi_{idx:04d}",
        "pillar": "advanced_rsi_dgm_svf",
        "images": [],
        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n{json.dumps(ans)}"}],
        "verifier": {"abstain": True, "forbid_tool_call": True, "expected": ans},
    }


# ===========================================================================
# Pillar 3 (`170` Deep Traces): Zero-Prior Unseen Formal Systems, Semirings,
#                               Bytecode Automata, & IEEE/bfloat16 Precision
# ===========================================================================


def make_v12_zero_prior_ood_item(rng: random.Random, idx: int) -> dict[str, Any]:
    variant = idx % 5

    if variant == 0:
        # Non-Commutative Skew-Modular Operator `a ⧓ b = (c1*a - b + k) mod M` Left- vs Right-Fold (JSON)
        m_mod = rng.choice([13, 17, 19])
        c1 = rng.choice([2, 3])
        k_off = rng.randint(1, 5)
        seq = [rng.randint(1, m_mod - 1) for _ in range(4)]

        def _op(a: int, b: int) -> int:
            return (c1 * a - b + k_off) % m_mod

        lf1 = _op(seq[0], seq[1])
        lf2 = _op(lf1, seq[2])
        left_fold = _op(lf2, seq[3])

        rf1 = _op(seq[2], seq[3])
        rf2 = _op(seq[1], rf1)
        right_fold = _op(seq[0], rf2)

        prompt = (
            f"Define the non-associative, non-commutative operator `a ⧓ b = ({c1} * a - b + {k_off}) mod {m_mod}` over `Z_{m_mod}` (`0 <= r < {m_mod}`).\n"
            f"For the 4-tuple `v = {seq}`, compute:\n"
            f"- `left_fold`: `(((v[0] ⧓ v[1]) ⧓ v[2]) ⧓ v[3])`,\n"
            f"- `right_fold`: `(v[0] ⧓ (v[1] ⧓ (v[2] ⧓ v[3])))`,\n"
            "- `associative_match`: boolean (`left_fold == right_fold`).\n"
            "Return ONLY JSON `{\"left_fold\": ..., \"right_fold\": ..., \"associative_match\": ...}`."
        )
        think = (
            f"{_open_line(rng)}\n"
            f"1. `[L0 Operator Law & Non-Commutativity]`: Notice `a ⧓ b = ({c1}*a - b + {k_off}) mod {m_mod}` multiplies ONLY the **left** operand `a` by `{c1}` and subtracts the **right** operand `b`! In Python/math modulo `{m_mod}`, negative intermediates wrap into `0..{m_mod - 1}` (e.g. `-2 mod {m_mod} = {(-2) % m_mod}`, NOT `-2`!).\n"
            f"2. `[L1 Right-Fold Argument Order Trap]`: {_bt_line(rng)} In `right_fold = v[0] ⧓ (v[1] ⧓ (v[2] ⧓ v[3]))`, when we evaluate `r1 = v[2] ⧓ v[3]` and move outward to `v[1] ⧓ r1`, `v[1]` is the **left** operand `a` and `r1` is the **right** operand `b`! Swapping them (`r1 ⧓ v[1]`) is the classic stack-evaluation error.\n"
            f"3. `[L2 Step-by-Step Arithmetic Trace]`:\n"
            f"   - **Left fold**:\n"
            f"     * `lf1 = {seq[0]} ⧓ {seq[1]} = ({c1}*{seq[0]} - {seq[1]} + {k_off}) mod {m_mod} = ({c1*seq[0] - seq[1] + k_off}) mod {m_mod} = {lf1}`.\n"
            f"     * `lf2 = {lf1} ⧓ {seq[2]} = ({c1}*{lf1} - {seq[2]} + {k_off}) mod {m_mod} = ({c1*lf1 - seq[2] + k_off}) mod {m_mod} = {lf2}`.\n"
            f"     * `left_fold = {lf2} ⧓ {seq[3]} = ({c1}*{lf2} - {seq[3]} + {k_off}) mod {m_mod} = ({c1*lf2 - seq[3] + k_off}) mod {m_mod} = {left_fold}`.\n"
            f"   - **Right fold**:\n"
            f"     * `rf1 = {seq[2]} ⧓ {seq[3]} = ({c1}*{seq[2]} - {seq[3]} + {k_off}) mod {m_mod} = ({c1*seq[2] - seq[3] + k_off}) mod {m_mod} = {rf1}`.\n"
            f"     * `rf2 = {seq[1]} ⧓ {rf1} = ({c1}*{seq[1]} - {rf1} + {k_off}) mod {m_mod} = ({c1*seq[1] - rf1 + k_off}) mod {m_mod} = {rf2}`.\n"
            f"     * `right_fold = {seq[0]} ⧓ {rf2} = ({c1}*{seq[0]} - {rf2} + {k_off}) mod {m_mod} = ({c1*seq[0] - rf2 + k_off}) mod {m_mod} = {right_fold}`.\n"
            f"4. `[L3 Final Check]`: Both residues lie in `0..{m_mod - 1}`, and `associative_match = {left_fold == right_fold}`."
        )
        ans = {"left_fold": left_fold, "right_fold": right_fold, "associative_match": (left_fold == right_fold)}
        return {
            "id": f"v12_zp_{idx:04d}",
            "pillar": "unseen_ood_self_discover",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n{json.dumps(ans)}"}],
            "verifier": {"abstain": True, "forbid_tool_call": True, "expected": ans},
        }

    if variant == 1:
        # Tropical `(min, +)` Matrix-Power k-Hop Shortest Walk (`python3 -I` verified)
        fn = f"tropical_minplus_power_{idx % 17}" if idx >= 15 else "tropical_minplus_power"
        prompt = (
            f"Implement `{fn}(adj_cost: list[list[float]], hops: int) -> list[list[float]]` in Python (`math.inf` represents absent edges):\n"
            "- Define tropical `(min, +)` matrix multiplication `(A ⊗ B)[i][j] = min_t (A[i][t] + B[t][j])`.\n"
            "- For `hops == 0`, return the `(n, n)` tropical identity `I_trop` (`0.0` on diagonal `i == j`, `math.inf` off-diagonal)!\n"
            "- Compute the exact `hops`-fold tropical product `adj_cost^(⊗ hops)` (`hops >= 0`) via binary exponentiation without mutating `adj_cost`."
        )
        think = (
            f"{_open_line(rng)}\n"
            "1. `[L0 Tropical Semiring `(R u {+inf}, min, +)` Identity Law]`: What is the multiplicative identity `I_trop` in `(min, +)` algebra?\n"
            "   - Additive operation is `min`, whose neutral element (zero of the semiring!) is `+inf` (`min(x, +inf) = x`).\n"
            "   - Multiplicative operation is `+`, whose neutral element (one of the semiring!) is `0.0` (`x + 0.0 = x`).\n"
            f"   - {_bt_line(rng)} If someone mistakenly initialized `I_trop` with `1.0` on the diagonal and `0.0` off-diagonal (standard ring identity!), `(A ⊗ I)[i][j] = min_t(A[i][t] + I[t][j])` would fail completely! Let's verify `(A ⊗ I_trop)[i][j]` with `I_trop[t][j] = 0.0 if t == j else +inf`: `min(A[i][j] + 0.0, +inf) = A[i][j]`. Exact identity!\n"
            "2. `[L1 `+inf + (+inf)` Arithmetic Check]`: In Python float arithmetic, `math.inf + x == math.inf` for any finite `x` or `+math.inf` (since edge weights are finite or `+math.inf`, never `-math.inf`, `NaN` cannot occur).\n"
            "3. `[L2 Toy Trace on `2x2` Graph with `hops = 2`]`: Let `A = [[inf, 3.0], [2.0, 1.0]]`:\n"
            "   - `hops = 0` -> `[[0.0, inf], [inf, 0.0]]`.\n"
            "   - `hops = 2` (`A ⊗ A`):\n"
            "     * `(0,0): min(inf+inf, 3+2) = 5.0`\n"
            "     * `(0,1): min(inf+3, 3+1) = 4.0`\n"
            "     * `(1,0): min(2+inf, 1+2) = 3.0`\n"
            "     * `(1,1): min(2+3, 1+1) = 2.0`.\n"
            "4. `[L3 Binary Exponentiation Loop]`: Initialize `res = I_trop`, `base = copy(adj_cost)`, and square `base = _mul(base, base)` while shifting `hops >>= 1` (`O(n^3 log(hops))`)."
        )
        code = (
            "import math\n\n"
            f"def {fn}(adj_cost: list[list[float]], hops: int) -> list[list[float]]:\n"
            "    n = len(adj_cost)\n"
            "    inf = math.inf\n\n"
            "    def _mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:\n"
            "        out = [[inf] * n for _ in range(n)]\n"
            "        for i in range(n):\n"
            "            row_a = a[i]\n"
            "            out_i = out[i]\n"
            "            for t in range(n):\n"
            "                ait = row_a[t]\n"
            "                if ait == inf:\n"
            "                    continue\n"
            "                row_b = b[t]\n"
            "                for j in range(n):\n"
            "                    cand = ait + row_b[j]\n"
            "                    if cand < out_i[j]:\n"
            "                        out_i[j] = cand\n"
            "        return out\n\n"
            "    res = [[0.0 if i == j else inf for j in range(n)] for i in range(n)]\n"
            "    base = [[float(x) for x in row] for row in adj_cost]\n"
            "    p = int(hops)\n"
            "    while p > 0:\n"
            "        if p & 1:\n"
            "            res = _mul(res, base)\n"
            "        base = _mul(base, base)\n"
            "        p >>= 1\n"
            "    return res"
        )
        test = (
            "import math\n"
            "A = [[math.inf, 3.0], [2.0, 1.0]]\n"
            f"assert {fn}(A, 0) == [[0.0, math.inf], [math.inf, 0.0]]\n"
            f"assert {fn}(A, 1) == A\n"
            f"assert {fn}(A, 2) == [[5.0, 4.0], [3.0, 2.0]]\n"
        )
        return {
            "id": f"v12_zp_{idx:04d}",
            "pillar": "zero_prior_formal",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
        }

    if variant == 2:
        # `bfloat16` Exact Round-to-Nearest-Even (`RNE`) & 1-ULP Spacing Calculation (JSON)
        exp_k = rng.choice([6, 7, 8, 9])  # [64..128], [128..256], [256..512], [512..1024]
        base_pow = 1 << exp_k
        ulp = base_pow / 128.0  # bfloat16 has 7 explicit mantissa bits!
        m_idx = rng.randint(1, 120)
        # Pick either exact halfway tie (odd vs even mantissa LSB!) or non-tie
        half_tie = (idx % 2 == 0)
        if half_tie:
            raw_val = base_pow + m_idx * ulp + 0.5 * ulp
            # RNE rounds to m_idx if m_idx is EVEN, else m_idx + 1!
            rounded_idx = m_idx if (m_idx % 2 == 0) else (m_idx + 1)
        else:
            raw_val = base_pow + m_idx * ulp + 0.75 * ulp
            rounded_idx = m_idx + 1
        bf16_val = base_pow + rounded_idx * ulp
        abs_err = abs(bf16_val - raw_val)
        prompt = (
            f"IEEE-like `bfloat16` has `1` sign bit, `8` exponent bits (bias `127`), and `7` explicit fraction/mantissa bits, using Round-to-Nearest-Even (`RNE`) on exact halfway ties.\n"
            f"For the positive real number `x = {raw_val:.6f}`, compute:\n"
            "1. `ulp_size`: exact 1-ULP grid spacing of `bfloat16` on the binade containing `x`,\n"
            "2. `bf16_rounded`: exact value obtained after `bfloat16` RNE rounding,\n"
            "3. `abs_rounding_error`: `|bf16_rounded - x|`.\n"
            "Return ONLY JSON `{\"ulp_size\": ..., \"bf16_rounded\": ..., \"abs_rounding_error\": ...}`."
        )
        think = (
            f"{_open_line(rng)}\n"
            f"1. `[L0 Binade & `bfloat16` ULP Formula]`: First locate `x = {raw_val:.6f}` between consecutive powers of two:\n"
            f"   - `2^{exp_k} = {base_pow}` and `2^{exp_k + 1} = {2 * base_pow}`, so `{base_pow} <= x < {2 * base_pow}`.\n"
            f"   - Because `bfloat16` has `p = 7` explicit mantissa bits (`128` equal subdivisions per binade), `ulp_size = 2^{exp_k} / 2^7 = {base_pow} / 128 = {ulp}`.\n"
            f"2. `[L1 Mantissa Index & RNE Halfway Tie Audit]`: {_bt_line(rng)} Let's express `(x - {base_pow}) / ulp_size` exactly so we see whether `x` sits on a halfway `.5` boundary:\n"
            f"   - `(x - {base_pow}) / {ulp} = ({raw_val - base_pow:.6f}) / {ulp} = {(raw_val - base_pow) / ulp:.4f}`.\n"
            + (
                f"   - AHA! The mantissa offset is `{m_idx}.5` — an EXACT halfway tie between `m_lo = {m_idx}` (LSB `{'0 = even' if m_idx % 2 == 0 else '1 = odd'}`) and `m_hi = {m_idx + 1}` (LSB `{'0 = even' if (m_idx + 1) % 2 == 0 else '1 = odd'}`)!\n"
                f"   - Standard schoolbook rounding (`floor(x + 0.5)`) would blindly round UP to `{m_idx + 1}`, which is WRONG when `{m_idx}` is even! Under IEEE Round-to-Nearest-Even (`RNE`), we MUST pick the neighbor with **even** mantissa index: `m_rne = {rounded_idx}`."
                if half_tie
                else f"   - The fractional part is `.75 > 0.5` (strictly closer to `m_hi = {m_idx + 1}` than `m_lo = {m_idx}`), so RNE rounds unambiguously to `m_rne = {rounded_idx}`."
            )
            + f"\n3. `[L2 Reconstruct Rounded Value & Error]`:\n"
            f"   - `bf16_rounded = {base_pow} + {rounded_idx} * {ulp} = {bf16_val}`.\n"
            f"   - `abs_rounding_error = |{bf16_val} - {raw_val}| = {abs_err}` (`<= 0.5 * ulp_size = {0.5 * ulp}`, confirming valid nearest rounding!).\n"
            "4. `[L3 Final Check]`: All three values are exact dyadic rationals."
        )
        ans = {"ulp_size": ulp, "bf16_rounded": bf16_val, "abs_rounding_error": abs_err}
        return {
            "id": f"v12_zp_{idx:04d}",
            "pillar": "zero_prior_formal",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n{json.dumps(ans)}"}],
            "verifier": {"abstain": True, "forbid_tool_call": True, "expected": ans},
        }

    if variant == 3:
        # Dual-Register / Stack Reversible Bytecode VM Trace (JSON)
        init_a = rng.randint(3, 9)
        init_b = rng.randint(2, 7)
        c_add = rng.randint(2, 5)
        c_xor = rng.randint(3, 6)
        # Program of 5 instructions on (R0, R1, acc_list)
        r0, r1 = init_a, init_b
        emitted: list[int] = []
        steps_log: list[str] = []
        prog = [
            ("ADD_MUL", c_add),     # R0 = R0 + c * R1
            ("EMIT_DIFF", 0),       # append R0 - R1
            ("ROT_XOR", c_xor),     # (R0, R1) = (R1 ^ c, R0)
            ("CLAMP_MOD", 11),      # R0 = R0 % 11; R1 = R1 % 11
            ("EMIT_DIFF", 0),       # append R0 - R1
        ]
        for st_i, (op_nm, arg_v) in enumerate(prog, start=1):
            if op_nm == "ADD_MUL":
                r0 = r0 + arg_v * r1
            elif op_nm == "EMIT_DIFF":
                emitted.append(r0 - r1)
            elif op_nm == "ROT_XOR":
                r0, r1 = (r1 ^ arg_v), r0
            elif op_nm == "CLAMP_MOD":
                r0, r1 = (r0 % arg_v), (r1 % arg_v)
            steps_log.append(f"   - Step {st_i} (`{op_nm} {arg_v}`): `R0={r0}, R1={r1}, out={emitted}`")

        prog_txt = "\n".join(f"  {i+1}. `{op} {av}`" for i, (op, av) in enumerate(prog))
        prompt = (
            "A custom two-register machine `(R0, R1)` with output tape `out = []` has instruction semantics:\n"
            "- `ADD_MUL k`: `R0 <- R0 + k * R1` (`R1` unchanged)\n"
            "- `EMIT_DIFF 0`: append `(R0 - R1)` to `out` (registers unchanged)\n"
            "- `ROT_XOR k`: **simultaneously** update `(R0, R1) <- (R1 ^ k, R0)` (`^` is bitwise XOR)\n"
            "- `CLAMP_MOD m`: update `R0 <- R0 mod m` and `R1 <- R1 mod m` (`0 <= r < m`)\n\n"
            f"Starting from `R0 = {init_a}, R1 = {init_b}, out = []`, execute:\n{prog_txt}\n\n"
            "Return ONLY JSON `{\"final_r0\": ..., \"final_r1\": ..., \"out\": [...]}`."
        )
        think = (
            f"{_open_line(rng)}\n"
            f"1. `[L0 Semantics & Simultaneous Assignment Invariant]`: Look carefully at `ROT_XOR k`: `(R0, R1) <- (R1 ^ k, R0)` is **simultaneous**! {_bt_line(rng)} If you updated `R0 = R1 ^ k` first and then set `R1 = R0`, `R1` would get `R1 ^ k` instead of the OLD `R0`! Also remember `^` is bitwise XOR (`{init_b} ^ {c_xor} = {init_b ^ c_xor}`), NOT exponentiation!\n"
            f"2. `[L1 Initial State]`: `R0 = {init_a}, R1 = {init_b}, out = []`.\n"
            f"3. `[L2 Instruction-by-Instruction Execution Trace]`:\n"
            + "\n".join(steps_log)
            + f"\n4. `[L3 Re-Check Bitwise XOR & Signed Difference]`: In Step 3, bitwise XOR `{init_b} ^ {c_xor}` in binary is `{bin(init_b)} ^ {bin(c_xor)} = {init_b ^ c_xor}`, and `R1` received the pre-Step-3 `R0 = {init_a + c_add * init_b}`. In Step 5, `R0 - R1 = {r0} - {r1} = {r0 - r1}`. Everything matches."
        )
        ans = {"final_r0": r0, "final_r1": r1, "out": emitted}
        return {
            "id": f"v12_zp_{idx:04d}",
            "pillar": "unseen_ood_self_discover",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n{json.dumps(ans)}"}],
            "verifier": {"abstain": True, "forbid_tool_call": True, "expected": ans},
        }

    # variant == 4: Parity-Switched Synchronous Ring Cellular Automaton (`python3 -I` verified)
    fn = f"step_parity_ring_automaton_{idx % 17}" if idx >= 15 else "step_parity_ring_automaton"
    prompt = (
        f"Implement `{fn}(ring: list[int], rounds: int, modulus: int = 9) -> list[int]` in Python without mutating `ring`:\n"
        "- If `not ring` or `rounds <= 0`, return a shallow `list(ring)` copy unchanged (do NOT apply `% modulus` when `rounds == 0`!).\n"
        "- In round `t` (`1`-indexed: `t = 1, 2, ..., rounds`), all `n = len(ring)` cells update **synchronously** from the round-`(t-1)` state `cur`:\n"
        "  * For odd `t` (`t % 2 == 1`): `nxt[i] = (cur[(i - 1) % n] + 2 * cur[i] + cur[(i + 1) % n]) % modulus`.\n"
        "  * For even `t` (`t % 2 == 0`): `nxt[i] = (cur[(i + 1) % n] - cur[(i - 1) % n]) % modulus`."
    )
    think = (
        f"{_open_line(rng)}\n"
        "1. `[L0 Three Classic Cellular-Automaton Traps]`: Let's check every invariant in the specification:\n"
        "   - **Trap 1 (`rounds == 0`):** Must return `list(ring)` untouched — even if an element in `ring` is `>= modulus` or negative, `0` rounds performs zero transitions!\n"
        "   - **Trap 2 (`1`-indexed vs `0`-indexed round parity):** `t` starts at `1` (`range(1, rounds + 1)`), so the FIRST round (`t = 1`) uses the **odd** rule! Using `for t in range(rounds)` with `t % 2 == 1` would invert odd and even rounds!\n"
        f"   - **Trap 3 (Synchronous vs In-Place Update):** {_bt_line(rng)} Updating `cur[i]` in-place inside `for i in range(n)` would feed the *new* `cur[i - 1]` into cell `i` (Gauss-Seidel) instead of synchronous Jacobi update! Always allocate a fresh `nxt = [0] * n` each round.\n"
        "2. `[L1 Singleton `n == 1` Wrap-Around Check]`: When `n == 1`, `(0 - 1) % 1 == 0` and `(0 + 1) % 1 == 0`:\n"
        "   - Round 1 (odd): `(cur[0] + 2*cur[0] + cur[0]) % m = (4 * cur[0]) % m`.\n"
        "   - Round 2 (even): `(cur[0] - cur[0]) % m = 0`.\n"
        "3. `[L2 Concrete Toy Trace on `ring = [1, 3, 2], rounds = 2, modulus = 7`]`: \n"
        "   - Round `t = 1` (odd): `i=0: (2 + 2 + 3)%7 = 0`; `i=1: (1 + 6 + 2)%7 = 2`; `i=2: (3 + 4 + 1)%7 = 1` -> `[0, 2, 1]`.\n"
        "   - Round `t = 2` (even, `right - left`): `i=0: (2 - 1)%7 = 1`; `i=1: (1 - 0)%7 = 1`; `i=2: (0 - 2)%7 = 5` -> `[1, 1, 5]`.\n"
        "4. `[L3 Non-Mutation Verification]`: `cur = list(ring)` leaves `ring` untouched."
    )
    code = (
        f"def {fn}(ring: list[int], rounds: int, modulus: int = 9) -> list[int]:\n"
        "    cur = list(ring)\n"
        "    n = len(cur)\n"
        "    if n == 0 or rounds <= 0:\n"
        "        return cur\n"
        "    for t in range(1, rounds + 1):\n"
        "        if t % 2 == 1:\n"
        "            cur = [(cur[(i - 1) % n] + 2 * cur[i] + cur[(i + 1) % n]) % modulus for i in range(n)]\n"
        "        else:\n"
        "            cur = [(cur[(i + 1) % n] - cur[(i - 1) % n]) % modulus for i in range(n)]\n"
        "    return cur"
    )
    test = (
        "orig = [1, 3, 2]\n"
        f"assert {fn}(orig, 0, modulus=7) == [1, 3, 2]\n"
        f"assert {fn}(orig, 2, modulus=7) == [1, 1, 5]\n"
        "assert orig == [1, 3, 2]\n"
    )
    return {
        "id": f"v12_zp_{idx:04d}",
        "pillar": "zero_prior_formal",
        "images": [],
        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n```python\n{code}\n```"}],
        "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
    }


# ===========================================================================
# Pillar 4 (`150` Deep Traces): Real-Time Research, Multi-Hop Provenance,
#                               Superseding Errata, Statistical Synthesis & Null Abstention
# ===========================================================================


def make_v12_research_and_citations_item(rng: random.Random, idx: int) -> dict[str, Any]:
    variant = idx % 3

    if variant == 0:
        # Multi-Source `[DOC-XXXX]` Provenance + Superseding Erratum + Inverse-Variance Weighted Synthesis
        sys_name = f"AuroraBus-{100 + idx}"
        m1 = round(rng.uniform(18.0, 32.0), 1)
        m2_bad = round(m1 + 6.5, 1)
        m4_true = round(m1 + rng.uniform(1.5, 4.5), 1)
        se1, se4 = 0.5, 1.0  # precisions w1 = 4.0, w4 = 1.0 -> exact clean decimal!
        w1, w4 = 1.0 / (se1**2), 1.0 / (se4**2)
        pooled = round((w1 * m1 + w4 * m4_true) / (w1 + w4), 4)
        pooled_se = round(math.sqrt(1.0 / (w1 + w4)), 4)

        prompt = (
            f"Synthesize the authoritative inverse-variance pooled latency (`ms`) for `{sys_name}` from the research dossier below:\n"
            f"[DOC-0101] `{sys_name}` Cohort-A verified telemetry: `mean = {m1} ms`, `SE = {se1} ms`.\n"
            f"[DOC-0102] `{sys_name}` Cohort-B preliminary uncalibrated draft: `mean = {m2_bad} ms`, `SE = 0.5 ms`.\n"
            f"[DOC-0103] `ZephyrLink-99` unrelated interconnect benchmark: `mean = 11.2 ms`, `SE = 0.25 ms`.\n"
            f"[DOC-0104] Official Erratum for `{sys_name}` Cohort-B (supersedes and revokes DOC-0102): recalibrated `mean = {m4_true} ms`, `SE = {se4} ms`.\n\n"
            "Instructions:\n"
            "- Exclude unrelated systems and revoked/preliminary drafts (do NOT bracket revoked or unrelated doc IDs in prose!).\n"
            "- Compute `pooled_mean = sum(w_i * mean_i) / sum(w_i)` and `pooled_se = sqrt(1 / sum(w_i))` with `w_i = 1 / SE_i^2` (`4` decimal places).\n"
            "- Write 2 concise sentences citing ONLY the authoritative `[DOC-XXXX]` tags inline, followed by JSON `{\"pooled_mean\": ..., \"pooled_se\": ..., \"authoritative_docs\": [...]}`."
        )
        think = (
            f"{_open_line(rng)}\n"
            f"1. `[L0 Source Provenance & Revocation Filter]`: Let's audit all four dossier records before doing any math:\n"
            f"   - `DOC-0101`: Verified `{sys_name}` Cohort-A (`mean={m1}, SE={se1}`) -> **KEEP**.\n"
            f"   - `DOC-0102`: Preliminary uncalibrated draft for Cohort-B, explicitly revoked by `DOC-0104` -> **EXCLUDE** (and remember: NEVER write `[DOC-0102]` in square brackets in the visible prose, or citation verifiers will flag a forbidden citation!).\n"
            f"   - `DOC-0103`: `ZephyrLink-99` (different system) -> **EXCLUDE**.\n"
            f"   - `DOC-0104`: Official recalibrated Erratum for `{sys_name}` Cohort-B (`mean={m4_true}, SE={se4}`) -> **KEEP**.\n"
            f"2. `[L1 Inverse-Variance Formula Audit]`: {_bt_line(rng)} Weight is `1 / SE^2` (inverse **variance**), NOT `1 / SE`! Let's check:\n"
            f"   - `w_101 = 1 / ({se1})^2 = 1 / 0.25 = 4.0`.\n"
            f"   - `w_104 = 1 / ({se4})^2 = 1 / 1.0 = 1.0`.\n"
            f"   - `sum_w = 4.0 + 1.0 = 5.0`.\n"
            f"3. `[L2 Exact Calculation]`: \n"
            f"   - `pooled_mean = (4.0 * {m1} + 1.0 * {m4_true}) / 5.0 = ({4.0 * m1:.2f} + {m4_true:.2f}) / 5.0 = {pooled}`.\n"
            f"   - `pooled_se = sqrt(1.0 / 5.0) = sqrt(0.2) = {math.sqrt(0.2):.6f}` -> rounded to 4 decimals: `{pooled_se}`.\n"
            "4. `[L3 Citation Synchronization Check]`: Authoritative list is `[\"DOC-0101\", \"DOC-0104\"]`; prose cites `[DOC-0101]` and `[DOC-0104]` and mentions neither `DOC-0102` nor `DOC-0103` in brackets."
        )
        ans_j = {"pooled_mean": pooled, "pooled_se": pooled_se, "authoritative_docs": ["DOC-0101", "DOC-0104"]}
        prose = (
            f"Based on verified `{sys_name}` Cohort-A telemetry [DOC-0101] and the recalibrated Cohort-B erratum [DOC-0104] "
            f"(excluding the superseded preliminary draft), the inverse-variance pooled latency is {pooled} ms with a pooled standard error of {pooled_se} ms."
        )
        return {
            "id": f"v12_res_{idx:04d}",
            "pillar": "realtime_research_citations",
            "kind": "citation_grounded",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n{prose}\n{json.dumps(ans_j)}"}],
            "verifier": {
                "required_citations": ["DOC-0101", "DOC-0104"],
                "forbidden_citations": ["DOC-0102", "DOC-0103"],
                "expected": ans_j,
                "forbid_tool_call": True,
            },
        }

    if variant == 1:
        # Calibrated Epistemic Abstention (`null` JSON when crucial conversion parameter is absent!)
        chem = f"Catalyst-VX{idx}"
        moles_given = round(rng.uniform(1.2, 4.8), 2)
        prompt = (
            f"Context excerpt:\n"
            f"- Trial 14 charged `{moles_given} mol` of `{chem}` at `340 K` into Reactor B and achieved `82.5%` selectivity.\n"
            f"- Molar mass and reactor headspace pressure of `{chem}` were redacted from the public appendix.\n\n"
            f"Compute the exact mass in grams (`mass_grams`) of `{chem}` charged into Reactor B. If the excerpt lacks a required physical constant to compute `mass_grams` without guessing, abstain by setting `mass_grams` to `null` and `missing_parameter` to `\"molar_mass\"`.\n"
            "Return ONLY JSON `{\"mass_grams\": ..., \"missing_parameter\": ...}`."
        )
        think = (
            f"{_open_line(rng)}\n"
            f"1. `[L0 Dimensional Dependency Check]`: Converting molar amount `n = {moles_given} mol` to `mass_grams` requires `m = n * M`, where `M` (`g/mol`) is the molar mass of `{chem}`.\n"
            f"2. `[L1 Hallucination Trap Audit]`: {_bt_line(rng)} `{chem}` is a synthetic redacted identifier, and the excerpt explicitly states its molar mass was redacted (`82.5%` selectivity and `340 K` temperature cannot substitute for `g/mol`!). Fabricating a placeholder like `100 g/mol` would violate epistemic calibration.\n"
            "3. `[L2 Schema Check]`: Set `\"mass_grams\": null` (JSON `null`, NOT string `\"null\"` or `0.0`!) and `\"missing_parameter\": \"molar_mass\"`.\n"
            "4. `[L3 Tool Abstention Check]`: No tools are needed or permitted; emit clean JSON directly."
        )
        ans = {"mass_grams": None, "missing_parameter": "molar_mass"}
        return {
            "id": f"v12_res_{idx:04d}",
            "pillar": "research_and_ablation",
            "kind": "tool_neg",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n{json.dumps(ans)}"}],
            "verifier": {"abstain": True, "forbid_tool_call": True, "expected": ans},
        }

    # variant == 2: Multi-Hop `[S#]` Relay Calculation + Forbidden Superseded `[S#]` Citation
    base_tp = rng.randint(120, 220)
    mult_s3 = rng.choice([1.25, 1.5])
    ovh_s4 = rng.randint(12, 28)
    net_tp = round(base_tp * mult_s3 - ovh_s4, 4)
    prompt = (
        "Review the four technical notes on ` Kronos-KV` pipeline throughput:\n"
        f"[S1] Production baseline shard rate: `{base_tp} req/s`.\n"
        "[S2] Early alpha note (DEPRECATED — superseded by [S3]): speculative multiplier `1.80x`.\n"
        f"[S3] Verified GA release note (supersedes [S2]): speculative multiplier `{mult_s3}x` over `[S1]`.\n"
        f"[S4] Checksum framing overhead subtracted after speculative scaling: `{ovh_s4} req/s`.\n\n"
        "Compute `net_req_per_s = baseline * verified_multiplier - framing_overhead`. Write 1-2 sentences citing ONLY verified active `[S#]` sources inline (never bracket `[S2]`), followed by JSON `{\"net_req_per_s\": ..., \"citations\": [...]}`."
    )
    think = (
        f"{_open_line(rng)}\n"
        f"1. `[L0 Provenance Graph]`: `[S1]` gives `baseline = {base_tp}`; `[S2]` (`1.80x`) is explicitly deprecated and superseded by `[S3]` (`{mult_s3}x`); `[S4]` gives post-scaling framing deduction `{ovh_s4}`.\n"
        f"2. `[L1 Order-of-Operations & Forbidden-Bracket Audit]`: {_bt_line(rng)} First check the wording in `[S4]`: subtracted **after** speculative scaling, meaning `({base_tp} * {mult_s3}) - {ovh_s4}`, NOT `({base_tp} - {ovh_s4}) * {mult_s3}`! Second, in prose we must NOT write `[S2]` in square brackets.\n"
        f"3. `[L2 Exact Arithmetic]`: `{base_tp} * {mult_s3} = {base_tp * mult_s3}`, minus `{ovh_s4}` = `{net_tp}`.\n"
        "4. `[L3 Final Check]`: Cite `[S1]`, `[S3]`, and `[S4]` in both prose and `\"citations\"`."
    )
    ans_j = {"net_req_per_s": net_tp, "citations": ["S1", "S3", "S4"]}
    prose = f"Combining the `{base_tp} req/s` baseline [S1] with the `{mult_s3}x` GA speculative multiplier [S3] and subtracting `{ovh_s4} req/s` framing overhead [S4] yields `{net_tp} req/s`."
    return {
        "id": f"v12_res_{idx:04d}",
        "pillar": "realtime_research_citations",
        "kind": "citation_grounded",
        "images": [],
        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n{prose}\n{json.dumps(ans_j)}"}],
        "verifier": {
            "required_citations": ["S1", "S3", "S4"],
            "forbidden_citations": ["S2"],
            "expected": ans_j,
            "forbid_tool_call": True,
        },
    }


# ===========================================================================
# Pillar 5 (`150` Deep Traces): Algorithmic Debugging Traps + 2-Turn RPR
#                               Selective Belief Revision (`preserving L0/L1!`)
# ===========================================================================


def make_v12_algo_and_2turn_rpr_item(rng: random.Random, idx: int) -> dict[str, Any]:
    variant = idx % 3

    if variant == 0:
        # 2-Turn RPR Selective Belief Revision (Preserving Independent Subtree!)
        units_a = rng.randint(20, 45)
        rate_a = rng.randint(12, 25)
        fixed_b = rng.randint(150, 320)
        bonus_b = rng.randint(30, 85)
        tax_old = 40
        tax_new = rng.randint(55, 95)

        branch_a = units_a * rate_a
        branch_b = fixed_b + bonus_b  # INDEPENDENT OF `units_a` and `tax`!
        pri_t1 = branch_a + branch_b - tax_old

        new_units_a = units_a + rng.randint(5, 14)
        new_branch_a = new_units_a * rate_a
        pri_t2 = new_branch_a + branch_b - tax_new

        t1_u = (
            f"Facility ledger clauses:\n"
            f"- `(A1)` Stream-Alpha revenue = `{units_a}` batches * `{rate_a}` credits/batch.\n"
            f"- `(B1)` Stream-Beta independent reserve = `{fixed_b}` base credits + `{bonus_b}` SLA credits.\n"
            f"- `(C1)` Net payout `primary = Stream_Alpha + Stream_Beta - {tax_old}`, and `preserved = Stream_Beta`.\n"
            "Return ONLY JSON `{\"primary\": ..., \"preserved\": ...}`."
        )
        t1_th = (
            f"{_open_line(rng)}\n"
            f"1. `[L0 Foundation Leaves]`: `A1` -> `{units_a} * {rate_a} = {branch_a}`; `B1` -> `{fixed_b} + {bonus_b} = {branch_b}` (independent subtree); `fee = {tax_old}`.\n"
            f"2. `[L1 Synthesis]`: `primary = {branch_a} + {branch_b} - {tax_old} = {pri_t1}`, `preserved = {branch_b}`."
        )
        t1_a = f"<think>\n{t1_th}\n</think>\n{json.dumps({'primary': pri_t1, 'preserved': branch_b})}"

        t2_u = (
            f"Belief-revision update: `(A1)` batches increased from `{units_a}` to `{new_units_a}` (same `{rate_a}` rate), and the `(C1)` deduction changed from `{tax_old}` to `{tax_new}`. `(B1)` is unchanged.\n"
            "Apply Reversible Pyramid selective revision: keep unaffected `(B1)` as `preserved`, recompute `primary`, and return ONLY JSON `{\"primary\": ..., \"preserved\": ...}`."
        )
        t2_th = (
            f"{_open_line(rng)}\n"
            "1. `[L0 Dependency DAG Audit (Reversible Pyramid Selective Invalidation)]`:\n"
            f"   - Mutated leaves: `A1.batches: {units_a} -> {new_units_a}` and `C1.deduction: {tax_old} -> {tax_new}`.\n"
            f"   - Unaffected foundation subtree: `B1` (`{fixed_b} + {bonus_b} = {branch_b}`) has ZERO dependency on `A1` or `C1.deduction` -> **PRESERVE `Stream_Beta = {branch_b}` without drift!**\n"
            f"2. `[L1 Selective Recomputation]`: {_bt_line(rng)} Don't compute a delta from `pri_t1` with the wrong sign on `{tax_new} - {tax_old}` (since deduction increased by `{tax_new - tax_old}`, net payout *decreases* by `{tax_new - tax_old}`!). Let's verify BOTH via direct recomputation AND delta check:\n"
            f"3. `[L2 Dual-Path Verification]`:\n"
            f"   - **Path A (Direct from L0/L1):** `new_Stream_Alpha = {new_units_a} * {rate_a} = {new_branch_a}`; `primary = {new_branch_a} + {branch_b} - {tax_new} = {pri_t2}`.\n"
            f"   - **Path B (Delta from Turn 1 `{pri_t1}`):** `{pri_t1} + ({new_units_a - units_a}) * {rate_a} - ({tax_new} - {tax_old}) = {pri_t1} + {(new_units_a - units_a)*rate_a} - {tax_new - tax_old} = {pri_t2}`.\n"
            f"4. `[L3 Consensus]`: Both paths give `primary = {pri_t2}` and `preserved = {branch_b}`."
        )
        t2_a = f"<think>\n{t2_th}\n</think>\n{json.dumps({'primary': pri_t2, 'preserved': branch_b})}"
        return {
            "id": f"v12_rpr2t_{idx:04d}",
            "pillar": "standalone_rpr_2turn",
            "train_last_turn_only": False,
            "images": [],
            "messages": [
                {"role": "user", "content": t1_u},
                {"role": "assistant", "content": t1_a},
                {"role": "user", "content": t2_u},
                {"role": "assistant", "content": t2_a},
            ],
            "verifier": {"forbid_tool_call": True, "primary": pri_t2, "preserved": branch_b, "anchor_primary": pri_t1},
        }

    if variant == 1:
        # Directed Cycle Detection (`3-Color WHITE/GRAY/BLACK DFS` — NEVER flagging diamond DAGs as cycles!)
        fn = f"has_directed_cycle_dfs_{idx % 19}" if idx >= 15 else "has_directed_cycle_dfs"
        prompt = (
            f"Implement `{fn}(graph: dict[str, list[str]]) -> bool` in Python:\n"
            "- `graph` maps node names to outgoing neighbor lists (neighbors may include sink nodes absent as top-level keys in `graph`).\n"
            "- Return `True` iff the directed graph contains at least one directed cycle (including self-loops `u -> u`), and `False` on all DAGs (such as diamond DAG `A->B, A->C, B->D, C->D`)."
        )
        think = (
            f"{_open_line(rng)}\n"
            "1. `[L0 Diamond-DAG False-Positive Trap]`: Why does a naive `visited: set[str]` fail on directed graphs?\n"
            f"   - {_bt_line(rng)} On the diamond DAG `A -> B -> D` and `A -> C -> D`, DFS explores `A -> B -> D`, finishes `D` and `B`, and then explores `A -> C -> D`. When `C` sees `D` already in `visited`, a 1-set `visited` check falsely reports a cycle! A directed cycle exists ONLY when we encounter a `GRAY` (`state == 1`) ancestor currently on the active recursion stack — NOT a `BLACK` (`state == 2`) completed node!\n"
            "2. `[L1 Implicit Sink Nodes]`: Notice a neighbor `v` in `graph[u]` might NOT appear as a key in `graph` (`graph.get(v, ())`). Using `graph[v]` directly would raise `KeyError` on sink nodes!\n"
            "3. `[L2 Toy Trace on Diamond vs Cycle]`: \n"
            "   - Diamond `{'A': ['B', 'C'], 'B': ['D'], 'C': ['D']}`: `A(1) -> B(1) -> D(1 -> 2), B(2) -> C(1) -> D(already 2, ignore!), C(2), A(2)` -> returns `False`.\n"
            "   - Self-loop `{'X': ['X']}`: `X` enters `state = 1`, checks neighbor `'X'` which has `state == 1` -> immediately returns `True`.\n"
            "4. `[L3 Iterative/Recursive Clean Implementation]`: Collect all start keys `list(graph.keys())` and run 3-color (`0=unvisited, 1=visiting, 2=done`) DFS."
        )
        code = (
            f"def {fn}(graph: dict[str, list[str]]) -> bool:\n"
            "    state: dict[str, int] = {}\n\n"
            "    def _dfs(u: str) -> bool:\n"
            "        state[u] = 1\n"
            "        for v in graph.get(u, ()):\n"
            "            sv = state.get(v, 0)\n"
            "            if sv == 1:\n"
            "                return True\n"
            "            if sv == 0 and _dfs(v):\n"
            "                return True\n"
            "        state[u] = 2\n"
            "        return False\n\n"
            "    for node in list(graph.keys()):\n"
            "        if state.get(node, 0) == 0 and _dfs(node):\n"
            "            return True\n"
            "    return False"
        )
        test = (
            f"assert {fn}({{'A': ['B', 'C'], 'B': ['D'], 'C': ['D']}}) is False\n"
            f"assert {fn}({{'A': ['B'], 'B': ['C'], 'C': ['A']}}) is True\n"
            f"assert {fn}({{'X': ['X']}}) is True\n"
            f"assert {fn}({{}}) is False\n"
        )
        return {
            "id": f"v12_algo_{idx:04d}",
            "pillar": "diagnostic_debugging",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
        }

    # variant == 2: Sliding-Window Monotonic Deque Maximum (`O(n)` without mutating `nums`)
    fn = f"sliding_window_maxima_{idx % 19}" if idx >= 15 else "sliding_window_maxima"
    prompt = (
        f"Implement `{fn}(nums: list[int], window_k: int) -> list[int]` in O(n) time using `collections.deque`:\n"
        "- If `not nums` or `window_k <= 0` or `window_k > len(nums)`, return `[]`.\n"
        "- Otherwise return `[max(nums[i : i + window_k]) for i in range(len(nums) - window_k + 1)]` in linear O(n) time without mutating `nums`."
    )
    think = (
        f"{_open_line(rng)}\n"
        "1. `[L0 Deque Invariants]`: Store **indices** (not values!) in `dq: deque[int]` maintaining two invariants at step `i`:\n"
        "   - **Range invariant:** `dq[0] >= i - window_k + 1` (if `dq[0] <= i - window_k`, `dq.popleft()`).\n"
        "   - **Monotonic decreasing values:** while `dq` and `nums[dq[-1]] <= nums[i]`, `dq.pop()`.\n"
        f"2. `[L1 Duplicate-Value & `window_k > len(nums)` Trap]`: {_bt_line(rng)} Suppose we stored values instead of indices, or used `<` instead of `<=`: if we stored values, expiring the front element when duplicates exist can pop the wrong occurrence! Storing indices `dq` and evicting `dq[0] <= i - window_k` handles duplicates unambiguously. Also check `window_k > len(nums)` -> returns `[]` cleanly.\n"
        "3. `[L2 Toy Trace on `nums = [4, 2, 5, 1, 3], window_k = 3`]`: \n"
        "   - `i=0 (4)`: `dq=[0]`\n"
        "   - `i=1 (2)`: `dq=[0, 1]`\n"
        "   - `i=2 (5)`: pop `1, 0`; `dq=[2]`; `i >= 2` -> append `nums[2]=5`.\n"
        "   - `i=3 (1)`: `dq=[2, 3]` -> append `nums[2]=5`.\n"
        "   - `i=4 (3)`: pop `3`; `dq=[2, 4]`; `2 > 4 - 3 = 1` (still in window `[2,3,4]`) -> append `nums[2]=5`. Output `[5, 5, 5]`.\n"
        "4. `[L3 Complexity]`: Each index is pushed once and popped at most once -> `O(n)`."
    )
    code = (
        "from collections import deque\n\n"
        f"def {fn}(nums: list[int], window_k: int) -> list[int]:\n"
        "    n = len(nums)\n"
        "    if n == 0 or window_k <= 0 or window_k > n:\n"
        "        return []\n"
        "    dq: deque[int] = deque()\n"
        "    out: list[int] = []\n"
        "    for i, val in enumerate(nums):\n"
        "        while dq and dq[0] <= i - window_k:\n"
        "            dq.popleft()\n"
        "        while dq and nums[dq[-1]] <= val:\n"
        "            dq.pop()\n"
        "        dq.append(i)\n"
        "        if i >= window_k - 1:\n"
        "            out.append(nums[dq[0]])\n"
        "    return out"
    )
    test = (
        f"assert {fn}([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]\n"
        f"assert {fn}([2, 2], 3) == []\n"
        f"assert {fn}([], 1) == []\n"
    )
    return {
        "id": f"v12_algo_{idx:04d}",
        "pillar": "diagnostic_debugging",
        "images": [],
        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n```python\n{code}\n```"}],
        "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
    }


# ===========================================================================
# Pillar 6: Length-De-Biased Cognitive `ORPO` Pairs (`400`)
#           Chosen = `380..560`-token verified 4-habit reasoning
#           Rejected = rushed `~90`-token template guess with a realistic bug!
# ===========================================================================


def build_v12_deep_vs_shallow_orpo_pairs(
    rng: random.Random,
    deep_items: list[dict[str, Any]],
    legacy_orpo: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []

    # 1. Convert 240 single-turn deep verified items into explicit Deep-Verified vs Rushed-Buggy ORPO pairs!
    candidates = [it for it in deep_items if len(it["messages"]) == 2]
    rng.shuffle(candidates)
    for idx, it in enumerate(candidates[:240]):
        u_msg = it["messages"][0]
        chosen_txt = it["messages"][1]["content"]
        ver = it.get("verifier") or {}
        if ver.get("kind") == "python_code":
            # Extract function signature and craft a realistic rushed 90-token buggy draft!
            m_def = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)", chosen_txt, re.DOTALL)
            fn_nm = m_def.group(1) if m_def else "solve_fn"
            if "svd(" in chosen_txt and "@ Vt" in chosen_txt:
                rej_code = f"import numpy as np\n\ndef {fn_nm}(*args, **kwargs):\n    W = np.asarray(args[0], dtype=np.float64)\n    U, S, Vt = np.linalg.svd(W, full_matrices=False)\n    return U * S[None, :]"
                rej_th = "Quickly apply SVD to W and scale U columns."
            elif "compute_uv=False" in chosen_txt:
                rej_code = f"import numpy as np\n\ndef {fn_nm}(*args, **kwargs):\n    _, s = np.linalg.svd(args[0], compute_uv=False)\n    return s"
                rej_th = "Unpack singular values from svd(compute_uv=False)."
            elif "dict[" in u_msg["content"]:
                rej_code = f"def {fn_nm}(a, b, *args, **kwargs):\n    return 0.5 * a + 0.5 * b"
                rej_th = "Average the two input state dictionaries directly."
            else:
                rej_code = f"def {fn_nm}(*args, **kwargs):\n    return None"
                rej_th = "Return default result immediately without checking edge cases."
            rejected_txt = f"<think>\n{rej_th}\n</think>\n```python\n{rej_code}\n```"
        else:
            exp = ver.get("expected") or {}
            wrong_exp = {}
            for k, v in exp.items():
                if isinstance(v, bool):
                    wrong_exp[k] = not v
                elif isinstance(v, int):
                    wrong_exp[k] = v + 2
                elif isinstance(v, float):
                    wrong_exp[k] = round(v * 1.15 + 0.5, 4)
                elif v is None:
                    wrong_exp[k] = 100.0
                else:
                    wrong_exp[k] = v
            rejected_txt = f"<think>\nEstimate the quantities quickly in one pass without checking counterexamples.\n</think>\n{json.dumps(wrong_exp)}"

        pairs.append(
            {
                "id": f"v12_deep_orpo_{idx:04d}",
                "messages": [u_msg],
                "chosen": chosen_txt,
                "rejected": rejected_txt,
            }
        )

    # 2. Keep 160 tool/path/SVD/citation precision ORPO pairs from v11
    for p in legacy_orpo[:160]:
        pairs.append(p)

    return pairs


def main() -> None:
    V12_CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261202)

    # 1. ALL held-out files on disk — strictly checked for 0 identifier & 0 10-gram leakage!
    heldout_paths = [
        ROOT_DIR / "training" / "data" / "rpr_v5_rsi_corpus" / "heldout_v5_rsi_56.jsonl",
        V9_CORPUS_DIR / "heldout_zero_prior_24.jsonl",
        V10_CORPUS_DIR / "heldout_v10_apex_36.jsonl",
        V10_CORPUS_DIR / "heldout_long_256k_needles_12.jsonl",
        ROOT_DIR / "training" / "data" / "rpr_v4_corpus" / "heldout_broad_42.jsonl",
        ROOT_DIR / "training" / "data" / "heldout_eval_20.jsonl",
    ]
    heldout_all: list[dict[str, Any]] = []
    for hp in heldout_paths:
        if hp.exists():
            heldout_all.extend(json.loads(l) for l in hp.read_text(encoding="utf-8").splitlines() if l.strip())

    # 2. Synthesize 890 deep 4-habit (`360..660` tok `<think>`) traces across all 5 reasoning pillars!
    deep_new: list[dict[str, Any]] = []
    for i in range(200):
        deep_new.append(make_v12_complex_logic_sat_item(rng, i))
    for i in range(220):
        deep_new.append(make_v12_weight_surgery_and_rsi_item(rng, i))
    for i in range(170):
        deep_new.append(make_v12_zero_prior_ood_item(rng, i))
    for i in range(150):
        deep_new.append(make_v12_research_and_citations_item(rng, i))
    for i in range(150):
        deep_new.append(make_v12_algo_and_2turn_rpr_item(rng, i))

    code_verified = 0
    think_word_counts: list[int] = []
    dapo_new: list[dict[str, Any]] = []

    for it in deep_new:
        ver = it.get("verifier")
        a_txt = it["messages"][-1]["content"]
        u_txt = "\n".join(str(m.get("content", "")) for m in it["messages"][:-1] if m.get("role") == "user")
        th_part = a_txt.split("</think>")[0].replace("<think>", "").strip()
        think_word_counts.append(len(th_part.split()))
        if isinstance(ver, dict):
            kind = (
                "python_code"
                if ver.get("kind") == "python_code"
                else ("citation_grounded" if "required_citations" in ver else ("multihop_math" if "primary" in ver else "tool_neg"))
            )
            sc = score_response(a_txt, prompt_text=u_txt, kind=kind, verifier=ver, max_think_tokens=3200)
            assert sc["correct"], f"Self-verification failed on {it['id']} ({kind}): {sc}"
            if kind == "python_code":
                code_verified += 1
            if len(dapo_new) < 90 and rng.random() < 0.16:
                dapo_new.append(
                    {
                        "id": f"dapo_{it['id']}",
                        "pillar": it["pillar"],
                        "kind": kind,
                        "messages": it["messages"][:-1],
                        "verifier": ver,
                        "reference_answer": a_txt,
                    }
                )

    # 3. Load:
    #    (A) `hybrid_long_think_v12.jsonl` (`DeepSeek-V4.1-Flash`, `DeepSeek-v4-Flash`, `DeepSeek-V4-Pro`, `s1K-1.1`, `OpenThoughts3`: `300..2,800` tok `<think>`)
    #    (B) `arex2_upgraded_bespoke.jsonl` (`AREX-2 27B` `python3 -I`-verified `300..2,600` tok `<think>` upgrades on bespoke RSI/RPR problems)
    #    (C) All `290` bespoke `agy_bespoke_*.jsonl` problems & verified solutions that stayed on disk!
    hybrid_hf_path = V12_CORPUS_DIR / "hybrid_long_think_v12.jsonl"
    hybrid_hf_rows: list[dict[str, Any]] = []
    if hybrid_hf_path.exists():
        hybrid_hf_rows = [json.loads(l) for l in hybrid_hf_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    arex_path = V12_CORPUS_DIR / "arex2_upgraded_bespoke.jsonl"
    arex_rows: list[dict[str, Any]] = []
    arex_upgraded_orig_ids: set[str] = set()
    if arex_path.exists():
        for ln in arex_path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                ar = json.loads(ln)
                arex_rows.append(ar)
                orig_id = str(ar.get("id", "")).replace("arex2_deep_", "", 1)
                arex_upgraded_orig_ids.add(orig_id)

    bespoke_rows: list[dict[str, Any]] = []
    for bp in sorted(V12_CORPUS_DIR.glob("agy_bespoke_*.jsonl")):
        for line in bp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)
                if isinstance(obj.get("messages"), list) and len(obj["messages"]) >= 2:
                    bespoke_rows.append(obj)

    bespoke_think_words: list[int] = []
    for hr in hybrid_hf_rows + arex_rows:
        a_h = str(hr["messages"][-1].get("content", ""))
        th_h = a_h.split("</think>")[0].replace("<think>", "").strip()
        hw = len(th_h.split())
        think_word_counts.append(hw)
        bespoke_think_words.append(hw)

    arex_by_orig = {str(ar.get("id", "")).replace("arex2_deep_", "", 1): ar for ar in arex_rows}
    extra_orpo_arex: list[dict[str, Any]] = []

    for br in bespoke_rows:
        a_b = str(br["messages"][-1].get("content", ""))
        assert "<think>" in a_b and "</think>" in a_b, f"Missing <think>...</think> in bespoke {br.get('id')}"
        ver_b = br.get("verifier")
        if isinstance(ver_b, dict) and ver_b:
            u_b = "\n".join(str(m.get("content", "")) for m in br["messages"][:-1] if m.get("role") == "user")
            k_b = (
                "python_code"
                if ver_b.get("kind") == "python_code"
                else ("tool_pos" if ver_b.get("tool") else ("citation_grounded" if "required_citations" in ver_b else ("multihop_math" if "primary" in ver_b else ("audit" if "verdict" in ver_b else "tool_neg"))))
            )
            sc_b = score_response(a_b, prompt_text=u_b, kind=k_b, verifier=ver_b, max_think_tokens=3200)
            assert sc_b["correct"], f"Bespoke verification failed on {br.get('id')}: {sc_b}"
            if k_b == "python_code":
                code_verified += 1
            # Feed ALL verifiable bespoke problems into Stage-3 DAPO/Dr. GRPO!
            if "</tool_call>" not in a_b:
                ref_ans_dapo = arex_by_orig[br["id"]]["messages"][-1]["content"] if br["id"] in arex_by_orig else a_b
                dapo_new.append(
                    {
                        "id": f"dapo_{br['id']}",
                        "pillar": br.get("pillar", "unseen_ood_self_discover"),
                        "kind": k_b,
                        "messages": br["messages"][:-1],
                        "verifier": ver_b,
                        "reference_answer": ref_ans_dapo,
                    }
                )
            # Pair AREX-2 deep `<think>` (`chosen`) directly against the short 50-word `<think>` (`rejected`) in ORPO!
            if br["id"] in arex_by_orig and len(br["messages"]) == 2:
                extra_orpo_arex.append(
                    {
                        "id": f"v12_arex_len_orpo_{br['id']}",
                        "messages": br["messages"][:-1],
                        "chosen": arex_by_orig[br["id"]]["messages"][-1]["content"],
                        "rejected": a_b,
                    }
                )

    # 4. Selectively retain ONLY essential Vision, XML Tool-Calling, Abstention, Audit (`CHECK A1:`), & 2-Turn RPR anchors
    #    from `v11` (DROPPING legacy short-`<think>` single-turn math/code/RSI rows that caused `124`-token `<think>` lock-in!)
    v11_sft = [json.loads(l) for l in (V11_CORPUS_DIR / "train_v11_sft.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    v11_val = [json.loads(l) for l in (V11_CORPUS_DIR / "val_v11_joint.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    v11_orpo = [json.loads(l) for l in (V11_CORPUS_DIR / "orpo_v11_pairs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    v11_dapo = [json.loads(l) for l in (V11_CORPUS_DIR / "dapo_v11_episodes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    essential_anchors: list[dict[str, Any]] = []
    for r in v11_sft:
        has_img = bool(r.get("images"))
        last_a = str(r["messages"][-1].get("content", ""))
        is_tool = "<tool_call>" in last_a
        is_audit = "CHECK A" in last_a and "SUMMARY:" in last_a
        is_multiturn = len(r["messages"]) >= 4
        if has_img or is_tool or is_audit or is_multiturn:
            essential_anchors.append(r)

    rng.shuffle(hybrid_hf_rows)
    rng.shuffle(arex_rows)
    rng.shuffle(bespoke_rows)
    rng.shuffle(deep_new)

    val_hf = hybrid_hf_rows[:36]
    train_hf = hybrid_hf_rows[36:]
    val_arex = arex_rows[:6]
    train_arex = arex_rows[6:]
    kept_short_bespoke = [r for r in bespoke_rows if r.get("id") not in arex_upgraded_orig_ids]
    val_deep = val_hf + val_arex + deep_new[:28]
    train_deep = train_hf + train_arex + deep_new[28:480]

    # Weight `300..2,800`-token `DeepSeek-V4.1-Flash` / `s1K` / `OpenThoughts3` (`2x`) & `AREX-2` verified bespoke (`3x`),
    # plus retain all remaining verified bespoke traces (`1x`) so ZERO bespoke problems are wasted!
    hf_rep2 = [{**r, "id": f"{r['id']}_hf2"} for r in train_hf]
    arex_rep2 = [{**r, "id": f"{r['id']}_ar2"} for r in train_arex]
    arex_rep3 = [{**r, "id": f"{r['id']}_ar3"} for r in train_arex]
    combined_raw = (
        train_hf + hf_rep2
        + train_arex + arex_rep2 + arex_rep3
        + deep_new[28:480]
        + kept_short_bespoke
        + essential_anchors[:220]
    )
    seen_sig: set[str] = set()
    merged_sft: list[dict[str, Any]] = []
    for r in combined_raw:
        rid = str(r.get("id", ""))
        if rid.endswith(("_hf2", "_ar2", "_ar3")):
            sig = rid
        else:
            sig = f"{rid}|" + json.dumps([m.get("content", "") for m in r["messages"][:-1]], sort_keys=True)
        if sig in seen_sig:
            continue
        seen_sig.add(sig)
        merged_sft.append(r)

    merged_val = val_deep + [r for r in v11_val if r.get("images") or "<tool_call>" in str(r["messages"][-1].get("content", ""))][:30]
    long_kv_rows = [make_v11_true_long_kv_item(rng, i) for i in range(80)]
    orpo_all = extra_orpo_arex + build_v12_deep_vs_shallow_orpo_pairs(rng, train_deep, v11_orpo)
    dapo_all = dapo_new + v11_dapo[:90]
    rng.shuffle(merged_sft)
    rng.shuffle(orpo_all)
    rng.shuffle(dapo_all)

    # 5. Scrub any accidental 10-gram or function-name collisions against ALL held-out benchmarks, then assert 0 leakage!
    eval_fn_s: set[str] = set()
    eval_10g_s: set[tuple[str, ...]] = set()
    for r_ev in heldout_all:
        u_ev = " ".join(str(m.get("content", "")) for m in r_ev.get("messages", []) if m.get("role") == "user")
        for fn_m in re.findall(r"(?:Implement|Write|def)\s+`?([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", u_ev):
            if len(fn_m) >= 6:
                eval_fn_s.add(fn_m)
        w_ev = u_ev.lower().split()
        for j_w in range(max(0, len(w_ev) - 10 + 1)):
            eval_10g_s.add(tuple(w_ev[j_w : j_w + 10]))

    def _clean_rows(rows_in: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out_r = []
        for r_l in rows_in:
            all_t = "\n".join(str(m.get("content", "")) for m in r_l.get("messages", []))
            if any(fn_m in all_t for fn_m in eval_fn_s):
                continue
            u_l = " ".join(str(m.get("content", "")) for m in r_l.get("messages", []) if m.get("role") == "user")
            w_l = u_l.lower().split()
            if any(tuple(w_l[j_w : j_w + 10]) in eval_10g_s for j_w in range(max(0, len(w_l) - 10 + 1))):
                continue
            out_r.append(r_l)
        return out_r

    merged_sft = _clean_rows(merged_sft)
    merged_val = _clean_rows(merged_val)
    long_kv_rows = _clean_rows(long_kv_rows)

    all_check = merged_sft + merged_val + long_kv_rows
    assert_decontaminated(all_check)
    _assert_zero_heldout_leakage(all_check, [hp for hp in heldout_paths if hp.exists()])

    def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

    _write_jsonl(V12_CORPUS_DIR / "train_v12_sft.jsonl", merged_sft)
    _write_jsonl(V12_CORPUS_DIR / "train_v12_long_kv.jsonl", long_kv_rows)
    _write_jsonl(V12_CORPUS_DIR / "val_v12_joint.jsonl", merged_val)
    _write_jsonl(V12_CORPUS_DIR / "orpo_v12_pairs.jsonl", orpo_all)
    _write_jsonl(V12_CORPUS_DIR / "dapo_v12_episodes.jsonl", dapo_all)

    mean_deep_words = round(sum(think_word_counts) / max(1, len(think_word_counts)), 1)
    bespoke_min_tok = round(min(bespoke_think_words) * 1.40) if bespoke_think_words else 0
    bespoke_mean_tok = round((sum(bespoke_think_words) / len(bespoke_think_words)) * 1.40, 1) if bespoke_think_words else 0.0
    bespoke_max_tok = round(max(bespoke_think_words) * 1.40) if bespoke_think_words else 0
    stats = {
        "train_v12_sft": len(merged_sft),
        "deepseek_v4_1_flash_and_hf_long_unique": len(hybrid_hf_rows),
        "arex2_27b_verified_bespoke_upgrades": len(arex_rows),
        "antigravity_bespoke_retained_all": len(bespoke_rows),
        "long_think_approx_tokens_min_mean_max": [bespoke_min_tok, bespoke_mean_tok, bespoke_max_tok],
        "deep_4habit_synthesized": len(deep_new),
        "essential_multimodal_tool_rpr_anchors": min(220, len(essential_anchors)),
        "mean_deep_think_words": mean_deep_words,
        "approx_mean_deep_think_tokens": round(mean_deep_words * 1.40, 1),
        "train_v12_long_kv_8k_26k": len(long_kv_rows),
        "val_v12_joint": len(merged_val),
        "orpo_v12_pairs": len(orpo_all),
        "dapo_v12_episodes": len(dapo_all),
        "python_functions_verified_isolated": code_verified,
        "heldout_tasks_checked_zero_leakage": len(heldout_all),
    }
    (V12_CORPUS_DIR / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"[v12 Deep-Thinker Corpus Built & Zero-Leakage Verified] {json.dumps(stats)}", flush=True)


if __name__ == "__main__":
    main()
