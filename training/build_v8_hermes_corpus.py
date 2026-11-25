"""Build the RPR v8 Corpus (`2.90B` `Engram` + `AttnRes` + `MoBA` + `MuonClip` + Multi-Turn `Python-TIR` + `FLASH/DEEP` Effort Dials + Hermes Agentic Tools + Adversarial Auditor Calibration).

Outputs inside `training/data/rpr_v8_hermes_corpus/`:
  1. `train_v8_sft.jsonl`          : ~960 high-density, deduplicated conversations (stratified `v7.1` 8-pillar anchor replay +
                                     multi-turn `python_exec` & traceback self-debug + `[EFFORT: FLASH]` vs `[EFFORT: DEEP]`
                                     controllable reasoning dials + Hermes `<tools>` calling + adversarial no-op auditor +
                                     `AttnRes` long-context option-table retrieval)
  2. `orpo_v8_pairs.jsonl`         : 160 explicit `(messages, chosen, rejected)` contrastive pairs (adversarial prototype-memo
                                     audit false-rebuilds, `FLASH` token-budget discipline vs rambling, natural math-spec code,
                                     and long-context Turn-1 option retrieval)
  3. `rlvr_v8_hermes_48.jsonl`     : 48 verifiable RL episodes (multi-turn traceback repair, Python/NumPy specs, adversarial
                                     audit, and 2-turn RPR)
  4. `val_v8_joint.jsonl`          : 80 stratified validation conversations covering all `v5/v7` + `v8` Hermes/Flash/Audit skills
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from training.build_rpr_v5_rsi_corpus import (
    V5_CORPUS_DIR,
    _deduplicate_rows,
    _verify_executable_code_rows,
)
from training.build_training_data import assert_decontaminated
from training.build_v6_repair_and_seal_corpus import V6_CORPUS_DIR, _as_eval_task
from training.build_v7_master_and_longctx_corpus import (
    V7_CORPUS_DIR,
    make_long_context_session_item,
    make_unassisted_direct_math_rpr_item,
    make_v7_math_spec_code_and_rsi_item,
    make_v7_orpo_contrastive_pair,
)
from training.collect_and_build_rpr_corpus import _strip_prior_think


V8_CORPUS_DIR = Path(__file__).resolve().parent / "data" / "rpr_v8_hermes_corpus"


# ---------------------------------------------------------------------------
# 1. Adversarial Irrelevant-Update RPR Auditor Calibration (`0.0%` False Rebuilds!)
# ---------------------------------------------------------------------------


def make_v8_adversarial_auditor_item(rng: random.Random, idx: int) -> dict[str, Any]:
    """Train Level-1 RPR Auditor never to mark `A2..A5` INVALID from prototype/canary numbers when `A1` stays VALID."""
    pairs = [
        ("SOLAR_ARRAY", "FUSION_TORUS", "grid_phaselock", 240, 40, 80, 996, 22, 7, 91, 24, 6, 107),
        ("QUIC_MULTIPLEX", "RDMA_ROCE", "zero_copy_dma", 180, 30, 60, 840, 19, 8, 68, 22, 9, 88),
        ("CRYO_BIOREACTOR", "PERFUSION_SKID", "inline_raman", 100, 25, 50, 340, 21, 10, 51, 29, 7, 84),
        ("FLASH_COLUMN", "MEMBRANE_CASCADE", "solvent_recycle", 120, 30, 60, 664, 29, 10, 65, 24, 6, 89),
    ]
    m_live, m_alt, feat, dem, cap_l, cap_a, bud, pen_l, pen_a, g_lat, ovh_l, ovh_a, g_res = pairs[idx % len(pairs)]
    dem_j = dem + rng.choice([0, 40, 80])
    bud_j = bud + rng.choice([0, 60, 120])
    is_noop = (idx % 3 != 0)

    rules = (
        f"System Profile:\n"
        f"- Target load: {dem_j} units. Budget ceiling: {bud_j}.\n"
        f"- Profile {m_live}: unit_cap={cap_l}, delay_pen={pen_l} (ceiling {g_lat}), res_ovh={ovh_l} (gross {g_res}), {feat} optional.\n"
        f"- Profile {m_alt}: unit_cap={cap_a}, delay_pen={pen_a} (ceiling {g_lat}), res_ovh={ovh_a} (gross {g_res}), {feat}=true mandatory.\n"
        f"Active live profile: {m_live}."
    )
    active_l1 = (
        f"Active Level-1 Assumptions:\n"
        f"- A1: active_mode={m_live}; boolean_feature={feat}=optional\n"
        f"- A2: workload demand is {dem_j}, and under {m_live} each unit handles {cap_l}\n"
        f"- A3: (required units) * unit_cost <= {bud_j}\n"
        f"- A4: base_lat + {pen_l} <= {g_lat}\n"
        f"- A5: per_unit_res <= {g_res} - {ovh_l}"
    )

    if is_noop:
        memo_kind = rng.choice(["QA prototype lab", "sandboxed canary VM", "offline staging simulation", "shadow bench-test"])
        ev = (
            f"Latest Evidence:\n"
            f"Engineering note: A {memo_kind} evaluated {m_alt} at {cap_a} per-unit throughput, {pen_a} penalty, "
            f"and {ovh_a} overhead across {rng.randint(2, 5)} test nodes; live production traffic remains on {m_live} with unchanged rules."
        )
        think = (
            f"[AGM-LOCK] Latest Evidence describes a non-production {memo_kind} on `{m_alt}` while explicitly keeping live production on `{m_live}`.\n"
            f"[RULE] Since `A1` (`active_mode={m_live}`) and global demand/budget are unchanged, `{m_alt}` prototype numbers (`{cap_a}`, `{pen_a}`, `{ovh_a}`) do NOT apply to live `{m_live}` — ALL of `A1..A5` remain `VALID`."
        )
        ans = (
            f"CHECK A1: VALID | Live production stays on {m_live}.\n"
            f"CHECK A2: VALID | Live demand {dem_j} and {m_live} capacity {cap_l} are unchanged ({memo_kind} does not affect live production).\n"
            f"CHECK A3: VALID | Live budget cap {bud_j} is unchanged.\n"
            f"CHECK A4: VALID | Live {m_live} penalty {pen_l} and cap {g_lat} are unchanged.\n"
            f"CHECK A5: VALID | Live {m_live} overhead {ovh_l} and limit {g_res} are unchanged.\n"
            f"SUMMARY: Non-switching {memo_kind} does not alter live {m_live} production; all assumptions A1..A5 remain VALID."
        )
        inv_exp: list[str] = []
    else:
        ev = (
            f"Latest Evidence:\n"
            f"Regime change: Live production cut over from {m_live} to {m_alt} effective immediately (demand {dem_j} and budget {bud_j} unchanged)."
        )
        think = (
            f"[AGM-SWITCH] Live production switched `{m_live} -> {m_alt}`, invalidating `A1` (mode & `{feat}=true`), `A2` (`{cap_l} -> {cap_a}`), "
            f"`A4` (`{pen_l} -> {pen_a}`), and `A5` (`{ovh_l} -> {ovh_a}`), while `A3` (`{bud_j}`) stays `VALID`."
        )
        ans = (
            f"CHECK A1: INVALID | Active production switched from {m_live} to {m_alt} ({feat}=true required).\n"
            f"CHECK A2: INVALID | Per-unit capacity changes from {cap_l} to {cap_a} under {m_alt}.\n"
            f"CHECK A3: VALID | Total budget cap {bud_j} is unchanged.\n"
            f"CHECK A4: INVALID | Active penalty changes from {pen_l} to {pen_a} under {m_alt}.\n"
            f"CHECK A5: INVALID | Active overhead changes from {ovh_l} to {ovh_a} under {m_alt}.\n"
            f"SUMMARY: Live regime switched from {m_live} to {m_alt}, invalidating A1, A2, A4, and A5."
        )
        inv_exp = ["A1", "A2", "A4", "A5"]

    sys_p = (
        "You are a strict logical consistency auditor for Level 1 Foundations.\n"
        "Output CHECK A1..A5: <VALID or INVALID> | <reason> followed by SUMMARY: <1-sentence summary>."
    )
    return {
        "pillar": "audit_and_ifeval",
        "kind": "audit",
        "images": [],
        "messages": [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": f"{rules}\n\n{active_l1}\n\n{ev}\n\nCheck each active assumption A1..A5 against Latest Evidence."},
            {"role": "assistant", "content": f"<think>{think}</think>\n{ans}"},
        ],
        "verifier": {"invalidated": inv_exp, "valid_ids": ["A1", "A2", "A3", "A4", "A5"]},
    }


# ---------------------------------------------------------------------------
# 2. Kimi K3.1 Pro Multi-Turn Python Execution & Traceback Self-Debug
# ---------------------------------------------------------------------------


def make_v8_multiturn_python_self_debug_item(rng: random.Random, idx: int) -> dict[str, Any]:
    """Multi-turn interactive Python traceback-repair & `python_exec` tool-integrated reasoning trajectories."""
    variant = idx % 6
    sys_dbg = (
        "You are a frontier AI & systems engineer. Reason carefully through invariants, tensor shapes, and edge cases in <think>, "
        "and output clean Python inside ```python ... ```."
    )

    if variant == 0:
        # Multi-turn traceback repair on TIES-Merging (fixing `np.linalg.norm(deltas)` bug -> clean stack + top-k + disjoint mean!)
        fn = rng.choice(["ties_merge_models", "ties_combine_deltas", "merge_task_vectors_ties"])
        kr = rng.choice([0.25, 0.5])
        u1 = (
            f"Implement `{fn}(W0: np.ndarray, deltas: list[np.ndarray], keep_ratio: float = {kr}, lam: float = 1.0) -> np.ndarray` in NumPy:\n"
            "1. Trim each delta in `deltas` to its top `ceil(keep_ratio * size)` entries by magnitude (ties broken by earlier flat index).\n"
            "2. Elect coordinate-wise sign `s = np.sign(sum(trimmed))`.\n"
            "3. Average ONLY trimmed entries matching non-zero `s` at each coordinate (`0.0` where `s == 0`), and return `W0 + lam * merged`."
        )
        bad_a1 = (
            "<think>[L1] Compute Frobenius norm across deltas and average.</think>\n"
            f"```python\nimport numpy as np\n\ndef {fn}(W0: np.ndarray, deltas: list[np.ndarray], keep_ratio: float = {kr}, lam: float = 1.0) -> np.ndarray:\n"
            "    if not deltas:\n        return W0.copy()\n"
            "    m = np.linalg.norm(deltas, ord='fro')\n"
            "    return W0 + lam * np.mean(deltas, axis=0)\n```"
        )
        u2 = (
            "Sandbox `python3 -I` test execution failed with:\n"
            "```\nValueError: Invalid norm order 'fro' for vectors / 3D stacked list of 2D arrays\n```\n"
            "Diagnose why the candidate violated all 3 TIES steps in `<think>`, then output the corrected ```python ... ``` implementation."
        )
        good_think = (
            "[INVALIDATE] Passing `deltas` (`list` of 2D arrays) to `np.linalg.norm(..., ord='fro')` treats it as a 3D array and raises `ValueError`, "
            "and `np.mean` completely skipped per-delta top-k trimming and sign election!\n"
            "[REBUILD] 1. If `not deltas or keep_ratio <= 0`: return `W0.copy()`. Set `k = max(1, math.ceil(keep_ratio * W0.size))`.\n"
            "2. Loop `for d in deltas`: `flat = d.reshape(-1)`, `idx = np.argsort(-np.abs(flat), kind='stable')[:k]`, copy `buf[idx] = flat[idx]`, reshape to `W0.shape`.\n"
            "3. `stk = np.stack(trimmed, axis=0)`, `elected = np.sign(np.sum(stk, axis=0))`, mask `agree = (np.sign(stk) == elected[None, ...]) & (elected[None, ...] != 0)`, "
            "and divide `np.sum(np.where(agree, stk, 0.0), axis=0)` by `np.maximum(np.sum(agree, axis=0), 1)`."
        )
        good_code = (
            "import math\nimport numpy as np\n\n"
            f"def {fn}(W0: np.ndarray, deltas: list[np.ndarray], keep_ratio: float = {kr}, lam: float = 1.0) -> np.ndarray:\n"
            "    if not deltas or keep_ratio <= 0.0:\n"
            "        return W0.copy()\n"
            "    k = max(1, math.ceil(keep_ratio * W0.size))\n"
            "    trimmed = []\n"
            "    for d in deltas:\n"
            "        flat = d.reshape(-1)\n"
            "        buf = np.zeros_like(flat, dtype=np.float64)\n"
            "        idx = np.argsort(-np.abs(flat), kind='stable')[:k]\n"
            "        buf[idx] = flat[idx]\n"
            "        trimmed.append(buf.reshape(W0.shape))\n"
            "    stk = np.stack(trimmed, axis=0)\n"
            "    elected = np.sign(np.sum(stk, axis=0))\n"
            "    agree = (np.sign(stk) == elected[None, ...]) & (elected[None, ...] != 0)\n"
            "    merged = np.sum(np.where(agree, stk, 0.0), axis=0) / np.maximum(np.sum(agree, axis=0), 1)\n"
            "    return W0 + lam * merged"
        )
        v_a, v_b = float(rng.randint(6, 9)), float(rng.randint(6, 9))
        test = (
            "import numpy as np\n"
            "W0 = np.zeros((2, 2))\n"
            f"d1 = np.array([[{v_a}, -4.0], [0.01, 0.02]])\n"
            f"d2 = np.array([[{v_b},  1.0], [-3.0, 0.01]])\n"
            f"res = {fn}(W0, [d1, d2], keep_ratio=0.5, lam=1.0)\n"
            f"assert np.allclose(res, np.array([[{(v_a+v_b)/2}, -4.0], [-3.0, 0.0]]))"
        )
        return {
            "pillar": "rsi_and_weight_surgery",
            "kind": "python_code",
            "train_last_turn_only": True,
            "images": [],
            "messages": [
                {"role": "system", "content": sys_dbg},
                {"role": "user", "content": u1},
                {"role": "assistant", "content": bad_a1},
                {"role": "user", "content": u2},
                {"role": "assistant", "content": f"<think>{good_think}</think>\n```python\n{good_code}\n```"},
            ],
            "verifier": {"kind": "python_code", "test": test},
        }

    if variant == 1:
        # Multi-turn traceback repair on Monotonic Prefix Deque (`stack.pop()` bug -> `dq.popleft()` + `dq.pop()`)
        fn = rng.choice(["min_subarray_ge_k", "shortest_sum_at_least", "min_contig_len_ge"])
        kv = rng.randint(6, 20)
        u1 = (
            f"Write `{fn}(nums: list[int], k: int) -> int` returning the shortest non-empty contiguous subarray length with sum >= `k` "
            "(or `-1` if none), where `nums` may contain negative integers."
        )
        bad_a1 = (
            "<think>[L1] Pop from end of stack when prefix difference >= k.</think>\n"
            f"```python\nfrom collections import deque\n\ndef {fn}(nums: list[int], k: int) -> int:\n"
            "    pref = [0]\n    for x in nums:\n        pref.append(pref[-1] + x)\n"
            "    best = -1\n    dq = deque()\n"
            "    for j, p in enumerate(pref):\n"
            "        while dq and p - pref[dq[-1]] >= k:\n            best = j - dq.pop()\n"
            "        dq.append(j)\n    return best\n```"
        )
        u2 = (
            "Sandbox `python3 -I` test execution failed with:\n"
            f"```\nAssertionError: {fn}([84, -37, 32, 40, 95], 167) returned 5, expected 3\n```\n"
            "Fix the monotonic deque invariants in `<think>` and output ONLY the corrected ```python ... ``` block."
        )
        good_think = (
            "[INVALIDATE] Two bugs in `dq`: (1) `best = j - dq.pop()` overwrites `best` without `min(...)` and skips pruning dominated larger prefix sums; "
            "(2) when `dq` maintains strictly increasing `pref[i]`, the smallest `pref` is at `dq[0]` (`popleft`), while any `pref[dq[-1]] >= p` at `i < j` is dominated by `j` (`dq.pop()`).\n"
            "[REBUILD] Initialize `best = len(nums) + 1`, pop valid candidates from front with `best = min(best, j - dq.popleft())`, prune back while `pref[dq[-1]] >= p`, and return `best if best <= len(nums) else -1`."
        )
        good_code = (
            "from collections import deque\n\n"
            f"def {fn}(nums: list[int], k: int) -> int:\n"
            "    pref = [0]\n"
            "    for x in nums:\n"
            "        pref.append(pref[-1] + x)\n"
            "    dq: deque[int] = deque()\n"
            "    best = len(nums) + 1\n"
            "    for j, p in enumerate(pref):\n"
            "        while dq and p - pref[dq[0]] >= k:\n"
            "            best = min(best, j - dq.popleft())\n"
            "        while dq and pref[dq[-1]] >= p:\n"
            "            dq.pop()\n"
            "        dq.append(j)\n"
            "    return best if best <= len(nums) else -1"
        )
        test = (
            f"assert {fn}([84, -37, 32, 40, 95], 167) == 3\n"
            f"assert {fn}([{kv-1}, -3, {kv}], {kv}) == 1\n"
            f"assert {fn}([1, -2, 1], 4) == -1"
        )
        return {
            "pillar": "diagnostic_debugging",
            "kind": "python_code",
            "train_last_turn_only": True,
            "images": [],
            "messages": [
                {"role": "system", "content": sys_dbg},
                {"role": "user", "content": u1},
                {"role": "assistant", "content": bad_a1},
                {"role": "user", "content": u2},
                {"role": "assistant", "content": f"<think>{good_think}</think>\n```python\n{good_code}\n```"},
            ],
            "verifier": {"kind": "python_code", "test": test},
        }

    if variant == 2:
        # Multi-turn traceback repair on SVD Null-Space (`[v for v in Vh if S[v] > tol]` TypeError -> `Vh[S > tol]`)
        fn = rng.choice(["svd_nullspace_step", "nullspace_weight_patch", "project_orth_complement"])
        d_o, d_i = rng.randint(4, 8), rng.randint(9, 14)
        u1 = (
            f"Implement `{fn}(W: np.ndarray, X_pres: np.ndarray, dW: np.ndarray, tol: float = 1e-8) -> np.ndarray` in NumPy "
            "that projects `dW` onto `P_perp = I - V_k.T @ V_k` where `V_k` are the right singular vector rows of `X_pres` with singular values `> tol`, returning `W + dW @ P_perp`."
        )
        bad_a1 = (
            "<think>[L1] Filter rows of Vh where S[v] > tol.</think>\n"
            f"```python\nimport numpy as np\n\ndef {fn}(W: np.ndarray, X_pres: np.ndarray, dW: np.ndarray, tol: float = 1e-8) -> np.ndarray:\n"
            "    _, S, Vh = np.linalg.svd(X_pres, full_matrices=False)\n"
            "    V_k = [v for v in Vh if S[v] > tol]\n"
            "    return W + dW @ (np.eye(W.shape[1]) - V_k.T @ V_k)\n```"
        )
        u2 = (
            "Sandbox `python3 -I` test execution failed with:\n"
            "```\nIndexError: arrays used as indices must be of integer (or boolean) type\n```\n"
            "Fix the NumPy SVD row-masking & empty-input handling in `<think>` and output ONLY the corrected ```python ... ``` block."
        )
        good_think = (
            "[INVALIDATE] Iterating `for v in Vh` yields 1D float row vectors `v`, so `S[v]` raises `IndexError` (and a Python `list` has no `.T` attribute!).\n"
            "[REBUILD] Handle `X_pres.size == 0` -> `return W + dW`. Otherwise unpack `_, S, Vh = np.linalg.svd(X_pres, full_matrices=False)` and boolean-index `V_k = Vh[S > tol]` (`[k, d_in]`). "
            "If `V_k.shape[0] == 0`, return `W + dW`; else `P_perp = np.eye(W.shape[1], dtype=W.dtype) - (V_k.T @ V_k)` and return `W + (dW @ P_perp)`."
        )
        good_code = (
            "import numpy as np\n\n"
            f"def {fn}(W: np.ndarray, X_pres: np.ndarray, dW: np.ndarray, tol: float = 1e-8) -> np.ndarray:\n"
            "    if X_pres.size == 0:\n"
            "        return W + dW\n"
            "    _, S, Vh = np.linalg.svd(X_pres, full_matrices=False)\n"
            "    V_k = Vh[S > tol]\n"
            "    if V_k.shape[0] == 0:\n"
            "        return W + dW\n"
            "    P_perp = np.eye(W.shape[1], dtype=W.dtype) - (V_k.T @ V_k)\n"
            "    return W + (dW @ P_perp)"
        )
        test = (
            "import numpy as np\n"
            f"rng = np.random.default_rng({idx + 301})\n"
            f"W, X, dW = rng.normal(size=({d_o}, {d_i})), rng.normal(size=(3, {d_i})), rng.normal(size=({d_o}, {d_i}))\n"
            f"Wn = {fn}(W, X, dW)\n"
            "assert np.allclose(X @ Wn.T, X @ W.T, atol=1e-9)"
        )
        return {
            "pillar": "rsi_and_weight_surgery",
            "kind": "python_code",
            "train_last_turn_only": True,
            "images": [],
            "messages": [
                {"role": "system", "content": sys_dbg},
                {"role": "user", "content": u1},
                {"role": "assistant", "content": bad_a1},
                {"role": "user", "content": u2},
                {"role": "assistant", "content": f"<think>{good_think}</think>\n```python\n{good_code}\n```"},
            ],
            "verifier": {"kind": "python_code", "test": test},
        }

    if variant == 3:
        # Multi-turn traceback repair on DoRA Row-Norm Decomposition (`(V / denom) * m[:, None]`)
        fn = rng.choice(["dora_linear_merge", "fuse_dora_row_mag", "recompose_dora_weights"])
        d_o, d_i, r_k = rng.randint(4, 8), rng.randint(7, 12), rng.randint(2, 4)
        u1 = (
            f"Implement `{fn}(W0: np.ndarray, B: np.ndarray, A: np.ndarray, m: np.ndarray, scale: float, eps: float = 1e-8) -> np.ndarray` in NumPy: "
            "compute `V = W0 + scale * (B @ A)`, normalize each row `i` by `max(||V[i]||_2, eps)`, and scale row `i` by `m[i]`."
        )
        bad_a1 = (
            "<think>[L1] Normalize rows with np.eye and multiply by m.</think>\n"
            f"```python\nimport numpy as np\n\ndef {fn}(W0: np.ndarray, B: np.ndarray, A: np.ndarray, m: np.ndarray, scale: float, eps: float = 1e-8) -> np.ndarray:\n"
            "    V = W0 + scale * (B @ A)\n"
            "    norms = np.linalg.norm(V, axis=1, keepdims=True)\n"
            "    return m * (V / (norms + eps * np.eye(V.shape[0])))\n```"
        )
        u2 = (
            "Sandbox `python3 -I` test execution failed with:\n"
            f"```\nValueError: operands could not be broadcast together with shapes ({d_o},{d_i}) ({d_o},{d_o})\n```\n"
            "Fix the broadcasting and `max(||V[i]||_2, eps)` formula in `<think>`, then output ONLY the ```python ... ``` block."
        )
        good_think = (
            f"[INVALIDATE] Adding `np.eye({d_o})` (`[{d_o}, {d_o}]`) to `norms` (`[{d_o}, 1]`) broadcasts to `[{d_o}, {d_o}]` != `V.shape` (`[{d_o}, {d_i}]`), "
            f"and 1D `m` (`[{d_o}]`) needs column broadcasting `m[:, None]` (`[{d_o}, 1]`).\n"
            "[REBUILD] Compute `denom = np.maximum(np.linalg.norm(V, axis=1, keepdims=True), eps)` (`[d_out, 1]`) and return `(V / denom) * m[:, None]`."
        )
        good_code = (
            "import numpy as np\n\n"
            f"def {fn}(W0: np.ndarray, B: np.ndarray, A: np.ndarray, m: np.ndarray, scale: float, eps: float = 1e-8) -> np.ndarray:\n"
            "    V = W0 + scale * (B @ A)\n"
            "    denom = np.maximum(np.linalg.norm(V, axis=1, keepdims=True), eps)\n"
            "    return (V / denom) * m[:, None]"
        )
        test = (
            "import numpy as np\n"
            f"rng = np.random.default_rng({idx + 402})\n"
            f"W0, B, A = rng.normal(size=({d_o}, {d_i})), rng.normal(size=({d_o}, {r_k})), rng.normal(size=({r_k}, {d_i}))\n"
            f"m = np.abs(rng.normal(loc=2.0, scale=0.3, size=({d_o},))) + 0.1\n"
            f"Wf = {fn}(W0, B, A, m, 0.5)\n"
            "assert np.allclose(np.linalg.norm(Wf, axis=1), m, atol=1e-6)"
        )
        return {
            "pillar": "rsi_and_weight_surgery",
            "kind": "python_code",
            "train_last_turn_only": True,
            "images": [],
            "messages": [
                {"role": "system", "content": sys_dbg},
                {"role": "user", "content": u1},
                {"role": "assistant", "content": bad_a1},
                {"role": "user", "content": u2},
                {"role": "assistant", "content": f"<think>{good_think}</think>\n```python\n{good_code}\n```"},
            ],
            "verifier": {"kind": "python_code", "test": test},
        }

    if variant == 4:
        # Kimi K2 / Muon Quintic Newton-Schulz Orthogonalization step in NumPy (RSI Weight-Optimizer Mastery!)
        fn = rng.choice(["muon_newton_schulz_orth", "quintic_orthogonalize_grad", "ns5_matrix_whiten"])
        r_d, c_d = rng.randint(6, 12), rng.randint(12, 20)
        if idx % 2 == 1:
            r_d, c_d = c_d, r_d
        prompt = (
            f"Implement `{fn}(G: np.ndarray, steps: int = 5, eps: float = 1e-7) -> np.ndarray` in NumPy (`import numpy as np`) "
            "computing Kimi/Muon's 5-step quintic Newton-Schulz matrix orthogonalization:\n"
            "- Normalize `X = G.astype(np.float64) / (np.linalg.norm(G) + eps)`; if `G.shape[0] > G.shape[1]`, transpose `X = X.T` first so `rows <= cols`.\n"
            "- Run `steps` iterations with `(a, b, c) = (3.4445, -4.7750, 2.0315)`: `A = X @ X.T`, `B = b * A + c * (A @ A)`, `X = a * X + B @ X`.\n"
            "- Transpose back if originally tall, and return `X`."
        )
        think = (
            "[L1] Quintic Newton-Schulz converges when `||X||_2 <= 1` and `X.shape[0] <= X.shape[1]` (`A = X @ X.T` is the smaller `[m, m]` Gram matrix).\n"
            "[L2] Normalize by Frobenius norm `+ eps`, transpose if `rows > cols`, apply `X = a * X + (b * A + c * (A @ A)) @ X` `steps` times, and restore orientation."
        )
        code = (
            "import numpy as np\n\n"
            f"def {fn}(G: np.ndarray, steps: int = 5, eps: float = 1e-7) -> np.ndarray:\n"
            "    a, b, c = 3.4445, -4.7750, 2.0315\n"
            "    tall = G.shape[0] > G.shape[1]\n"
            "    X = G.astype(np.float64)\n"
            "    X = X / (np.linalg.norm(X) + eps)\n"
            "    if tall:\n"
            "        X = X.T\n"
            "    for _ in range(steps):\n"
            "        A = X @ X.T\n"
            "        X = a * X + (b * A + c * (A @ A)) @ X\n"
            "    return X.T if tall else X"
        )
        test = (
            "import numpy as np\n"
            f"rng = np.random.default_rng({idx + 505})\n"
            f"G = rng.normal(size=({r_d}, {c_d}))\n"
            f"Q = {fn}(G, steps=5)\n"
            f"assert Q.shape == ({r_d}, {c_d})\n"
            "gram = (Q.T @ Q) if Q.shape[0] > Q.shape[1] else (Q @ Q.T)\n"
            "assert 0.55 <= float(np.mean(np.diag(gram))) <= 1.35"
        )
        return {
            "pillar": "rsi_and_weight_surgery",
            "kind": "python_code",
            "train_last_turn_only": True,
            "images": [],
            "messages": [
                {"role": "system", "content": sys_dbg},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"<think>{think}</think>\n```python\n{code}\n```"},
            ],
            "verifier": {"kind": "python_code", "test": test},
        }

    # variant == 5: Sandboxed `<function=python_exec>` multi-turn TIR verification
    base_m = make_unassisted_direct_math_rpr_item(rng, idx + 300)
    u1 = base_m["messages"][1]["content"]
    exp = base_m["expected"]
    # Extract ledger expression from Turn-1 think
    t1_raw = base_m["messages"][2]["content"]
    sys_tir = (
        "You are an analytical problem solver with access to `python_exec`.\n"
        "<tools>\n"
        '[{"name": "python_exec", "description": "Execute isolated Python 3 code and print exact results.", '
        '"parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}]\n'
        "</tools>\n"
        'Verify exact arithmetic via `<tool_call>` if helpful, then output JSON `{"primary": "OPT-X", "preserved": <integer>}`.'
    )
    py_call = (
        "<think>[FLASH] Delegate exact ledger arithmetic to `python_exec` before finalizing JSON.</think>\n"
        "<tool_call>\n<function=python_exec>\n<parameter=code>\n"
        f"print({exp['preserved']})\n"
        "</parameter>\n</function>\n</tool_call>"
    )
    # Better: extract actual `a * b + c * d - e` from `t1_raw`!
    import re as _re
    m_expr = _re.search(r"Compute `([^`]+)` by distributive", t1_raw)
    expr_str = m_expr.group(1) if m_expr else str(exp["preserved"])
    py_call = (
        f"<think>[FLASH] Compute exact ledger `{expr_str}` in `python_exec`.</think>\n"
        f"<tool_call>\n<function=python_exec>\n<parameter=code>\nprint({expr_str})\n</parameter>\n</function>\n</tool_call>"
    )
    tool_resp = f'<tool_response>\n{{"python_stdout": "{exp["preserved"]}"}}\n</tool_response>\nNow finish checking the active regime bounds and output final JSON.'
    final_msg = t1_raw
    return {
        "pillar": "multistep_tool_use",
        "kind": "math_opt",
        "images": [],
        "messages": [
            {"role": "system", "content": sys_tir},
            {"role": "user", "content": u1},
            {"role": "assistant", "content": py_call},
            {"role": "user", "content": tool_resp},
            {"role": "assistant", "content": final_msg},
        ],
        "expected": exp,
        "verifier": {"primary": exp["primary"], "preserved": exp["preserved"], "forbid_tool_call": True},
    }


# ---------------------------------------------------------------------------
# 3. DeepSeek-V4.1-Flash Controllable Effort Dials (`[EFFORT: FLASH]` vs `[EFFORT: DEEP]`)
#    + Native Hermes Agentic `<tools>` Calling (`terminal`, `read_file`, `edit_file`, `search_files`)
# ---------------------------------------------------------------------------


def make_v8_hermes_flash_and_agent_item(rng: random.Random, idx: int) -> dict[str, Any]:
    """Train `[EFFORT: FLASH]` ultra-compact (<40 tok) thinking + Hermes XML `<tool_call>` & `[EFFORT: DEEP]` verification."""
    kind = idx % 4

    if kind == 0:
        # Hermes Agentic Tool Calling under `[EFFORT: FLASH]` (`<tool_call><function=...><parameter=...>`)
        scenarios = [
            (
                "read_file",
                {"path": "src/router/dispatch.py", "offset": 1, "limit": 120},
                "Inspect `src/router/dispatch.py` lines 1-120 so we can check how fallback retries are handled.",
                "[FLASH] Call `read_file` on `src/router/dispatch.py` (`offset=1, limit=120`).",
            ),
            (
                "terminal",
                {"command": "pytest -q tests/test_cache.py"},
                "Run pytest quietly on `tests/test_cache.py` to verify our fix.",
                "[FLASH] Execute `pytest -q tests/test_cache.py` via `terminal`.",
            ),
            (
                "search_files",
                {"pattern": "scaled_dot_product_attention", "path": "src/"},
                "Search `src/` for all occurrences of `scaled_dot_product_attention`.",
                "[FLASH] Search `src/` for `scaled_dot_product_attention` using `search_files`.",
            ),
        ]
        fn_name, params, user_req, flash_think = scenarios[(idx // 4) % len(scenarios)]
        tools_json = json.dumps([
            {"name": "read_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, "required": ["path"]}},
            {"name": "terminal", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
            {"name": "search_files", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern"]}},
        ])
        param_xml = "\n".join(f"<parameter={k}>\n{v}\n</parameter>" for k, v in params.items())
        ans = f"<think>{flash_think}</think>\n<tool_call>\n<function={fn_name}>\n{param_xml}\n</function>\n</tool_call>"
        return {
            "pillar": "multistep_tool_use",
            "images": [],
            "messages": [
                {"role": "system", "content": f"You are Hermes, an autonomous local AI engineering agent. [EFFORT: FLASH]\n<tools>\n{tools_json}\n</tools>"},
                {"role": "user", "content": user_req},
                {"role": "assistant", "content": ans},
            ],
        }

    if kind == 1:
        # Hermes Multi-Turn Tool Result -> Concise Grounded Synthesis under `[EFFORT: FLASH]` (0 extra unnecessary tool calls!)
        port = rng.choice([8991, 8080, 5173, 3000])
        ms = round(rng.uniform(4.2, 18.7), 1)
        model_id = rng.choice(["lfm3-vl-2.9b-rpr-v8-hermes", "twil-8b-soup-hermes", "ncm-hermes-v6"])
        tools_json = json.dumps([
            {"name": "terminal", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
        ])
        u1 = f"Check the health JSON on `http://127.0.0.1:{port}/health` and report the active model and latency."
        a1 = f"<think>[FLASH] Fetch `/health` on port `{port}` via `terminal`.</think>\n<tool_call>\n<function=terminal>\n<parameter=command>\ncurl -s http://127.0.0.1:{port}/health\n</parameter>\n</function>\n</tool_call>"
        u2 = f'<tool_response>\n{{"status": "ok", "active_model": "{model_id}", "latency_ms": {ms}}}\n</tool_response>'
        a2 = (
            f"<think>[FLASH] Health endpoint returned `status=ok`, `active_model={model_id}`, `latency_ms={ms}`. No further tool calls needed.</think>\n"
            f"The server on `127.0.0.1:{port}` is healthy (`status: ok`): active model is `{model_id}` with `{ms} ms` latency."
        )
        return {
            "pillar": "multistep_tool_use",
            "images": [],
            "messages": [
                {"role": "system", "content": f"You are Hermes, an autonomous local AI engineering agent. [EFFORT: FLASH]\n<tools>\n{tools_json}\n</tools>"},
                {"role": "user", "content": u1},
                {"role": "assistant", "content": a1},
                {"role": "user", "content": u2},
                {"role": "assistant", "content": a2},
            ],
        }

    if kind == 2:
        # `[EFFORT: FLASH]` Fast Analytical Answer (`<= 35` think tokens!)
        x, y, z = rng.randint(12, 35), rng.randint(14, 29), rng.randint(8, 25)
        ans_v = x + y - z
        u = f"Compute `{x} + {y} - {z}` and return raw JSON `{{\"result\": <int>}}`."
        a = f"<think>[FLASH] `{x} + {y} = {x+y}`; `{x+y} - {z} = {ans_v}`.</think>\n{json.dumps({'result': ans_v})}"
        return {
            "pillar": "clause_witness_math",
            "images": [],
            "messages": [
                {"role": "system", "content": "Solve accurately and concisely. [EFFORT: FLASH]"},
                {"role": "user", "content": u},
                {"role": "assistant", "content": a},
            ],
        }

    # kind == 3: `[EFFORT: DEEP]` Full `[L1]->[L2]->[L3]` Verification on Natural Math-Spec Code / RSI
    ex = make_v7_math_spec_code_and_rsi_item(rng, idx)
    ex["messages"][0]["content"] = ex["messages"][0]["content"] + " [EFFORT: DEEP]"
    return ex


# ---------------------------------------------------------------------------
# 4. Contrastive `ORPO` Pairs for v8 (`Auditor`, `FLASH vs Ramble`, `Code`, `Long-Context`)
# ---------------------------------------------------------------------------


def make_v8_orpo_pair(rng: random.Random, idx: int) -> dict[str, Any]:
    slot = idx % 4
    if slot == 0:
        # Adversarial prototype memo: ALL `A1..A5: VALID` (chosen) vs over-cautious `A2..A5: INVALID` false rebuild (rejected)!
        aud = make_v8_adversarial_auditor_item(rng, idx * 3 + 1)  # `is_noop == True`
        chosen = aud["messages"][2]["content"]
        rejected = (
            "<think>[L1] Live regime stays unchanged so A1 is VALID, but prototype memo mentioned different capacity/penalty so mark A2, A4, A5 INVALID.</think>\n"
            "CHECK A1: VALID | Active mode unchanged.\n"
            "CHECK A2: INVALID | Prototype memo tested different throughput.\n"
            "CHECK A3: VALID | Budget unchanged.\n"
            "CHECK A4: INVALID | Prototype memo tested different penalty.\n"
            "CHECK A5: INVALID | Prototype memo tested different overhead.\n"
            "SUMMARY: Active regime remains unchanged; A1 is VALID while A2, A4, A5 are INVALID due to prototype evidence."
        )
        return {"id": f"orpo_v8-{idx:03d}", "messages": aud["messages"][:2], "chosen": chosen, "rejected": rejected}

    if slot == 1:
        # `[EFFORT: FLASH]` crisp tool call (chosen) vs 150-token overthinking ramble (rejected)!
        fl = make_v8_hermes_flash_and_agent_item(rng, idx * 4 + 0)
        chosen = fl["messages"][2]["content"]
        tool_tail = chosen.split("</think>\n", 1)[-1]
        rejected = (
            "<think>Let me think at length about every possible way we might inspect files in this workspace. First, we could consider reading the file or searching directories, and let us review all parameters carefully over several paragraphs before deciding what to call...</think>\n"
            + tool_tail
        )
        return {"id": f"orpo_v8-{idx:03d}", "messages": fl["messages"][:2], "chosen": chosen, "rejected": rejected}

    if slot == 2:
        # Long-context Turn-1 option retrieval & explicit 6-option numeric check (chosen) vs stale option / None (rejected)!
        lc = make_long_context_session_item(rng, idx * 3, target_total_tokens=2400, for_eval=False)
        msgs_p = _strip_prior_think(lc["messages"][:-1])
        chosen = lc["messages"][-1]["content"]
        bad_j = json.dumps({"primary": "OPT-A" if lc["expected"]["primary"] != "OPT-A" else "OPT-B", "preserved": lc["expected"]["preserved"]})
        rejected = f"<think>[AUDIT] Regime switched; guess option without checking Turn-1 table bounds.</think>\n{bad_j}"
        return {"id": f"orpo_v8-{idx:03d}", "messages": msgs_p, "chosen": chosen, "rejected": rejected}

    return make_v7_orpo_contrastive_pair(rng, idx)


def main() -> None:
    V8_CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    rng_tr = random.Random(20261801)
    rng_op = random.Random(20261802)
    rng_rl = random.Random(20261803)
    rng_va = random.Random(20261804)

    # Zero-contamination guard across ALL held-out benchmarks (`heldout_v5_rsi_56`, `final_test_24`, `long_context_eval_16`)
    forbid_prompts: set[str] = set()
    for p_file in [
        V5_CORPUS_DIR / "heldout_v5_rsi_56.jsonl",
        V7_CORPUS_DIR / "long_context_eval_16.jsonl",
    ]:
        if p_file.exists():
            for line in p_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    for m in json.loads(line).get("messages", []):
                        if m.get("role") == "user":
                            forbid_prompts.add(str(m.get("content", "")).strip())
    seal_ep_path = V6_CORPUS_DIR / "seal_svf_episodes.json"
    if seal_ep_path.exists():
        sd = json.loads(seal_ep_path.read_text(encoding="utf-8"))
        for t_ex in sd.get("final_test_24", []):
            for m in t_ex.get("messages", []):
                if m.get("role") == "user":
                    forbid_prompts.add(str(m.get("content", "")).strip())

    def _clean(row: dict[str, Any]) -> bool:
        return not any(
            m.get("role") == "user" and str(m.get("content", "")).strip() in forbid_prompts
            for m in row.get("messages", [])
        )

    # 1. Stratified 8-pillar replay from `train_v7_sft.jsonl` (`420` rows incl. PIL vision!) + `540` new `v8` rows!
    v7_all = [
        json.loads(l)
        for l in (V7_CORPUS_DIR / "train_v7_sft.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    vis_v7 = [r for r in v7_all if r.get("images") and _clean(r)]
    txt_v7 = [r for r in v7_all if not r.get("images") and _clean(r)]
    rng_tr.shuffle(txt_v7)
    train_rows: list[dict[str, Any]] = list(vis_v7[:36]) + list(txt_v7[:384])

    for i in range(144):
        it = make_v8_multiturn_python_self_debug_item(rng_tr, i)
        if _clean(it):
            train_rows.append(it)
    for i in range(144):
        it = make_v8_hermes_flash_and_agent_item(rng_tr, i)
        if _clean(it):
            train_rows.append(it)
    for i in range(84):
        it = make_v8_adversarial_auditor_item(rng_tr, i)
        if _clean(it):
            train_rows.append(it)
    for i in range(72):
        # Premise-switch weighted long-context sessions (`idx * 3` & `idx * 3 + 1` are premise switches!) for `AttnRes`!
        lc_it = make_long_context_session_item(rng_tr, i + 600, target_total_tokens=rng_tr.choice([2200, 3000]), for_eval=False)
        if _clean(lc_it):
            train_rows.append(lc_it)
    for i in range(90):
        ms_it = make_v7_math_spec_code_and_rsi_item(rng_tr, i + 400)
        if _clean(ms_it):
            train_rows.append(ms_it)

    train_rows = [r for r in _deduplicate_rows(train_rows) if _clean(r)]
    rng_tr.shuffle(train_rows)
    for idx, r in enumerate(train_rows, start=1):
        r["id"] = f"train_v8-{idx:04d}"
        r.setdefault("images", [])

    # 2. 160 Contrastive `ORPO` Pairs
    orpo_pairs = [p for i in range(160) for p in [make_v8_orpo_pair(rng_op, i)] if _clean(p)]

    # 3. 48 Multi-Domain Verifiable `RLVR` Episodes (Multi-turn Python self-debug + Natural Math-Spec + RPR)
    rlvr_48: list[dict[str, Any]] = []
    for i in range(18):
        t = _as_eval_task(make_v8_multiturn_python_self_debug_item(rng_rl, i), f"v8-rlvr-dbg-{i+1:02d}")
        if _clean(t):
            rlvr_48.append(t)
    for i in range(18):
        t = _as_eval_task(make_v7_math_spec_code_and_rsi_item(rng_rl, i + 100), f"v8-rlvr-spec-{i+1:02d}")
        if _clean(t):
            rlvr_48.append(t)
    for i in range(12):
        t = _as_eval_task(make_unassisted_direct_math_rpr_item(rng_rl, i + 100), f"v8-rlvr-rpr-{i+1:02d}")
        if _clean(t):
            rlvr_48.append(t)
    rng_rl.shuffle(rlvr_48)

    # 4. Joint Validation Set (`val_v8_joint.jsonl`: 48 from `val_v7_joint` + 32 new `v8` items)
    v7_val = [
        json.loads(l)
        for l in (V7_CORPUS_DIR / "val_v7_joint.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    val_rows: list[dict[str, Any]] = [r for r in v7_val[:48] if _clean(r)]
    for i in range(12):
        val_rows.append(make_v8_multiturn_python_self_debug_item(rng_va, i))
    for i in range(12):
        val_rows.append(make_v8_hermes_flash_and_agent_item(rng_va, i))
    for i in range(8):
        val_rows.append(make_v8_adversarial_auditor_item(rng_va, i))
    val_rows = [r for r in _deduplicate_rows(val_rows) if _clean(r)]
    for idx, r in enumerate(val_rows, start=1):
        r["id"] = f"val_v8-{idx:04d}"
        r.setdefault("images", [])

    # Verify ONLY final assistant turn of multi-turn self-debug rows (`_verify_executable_code_rows` checks `messages[-1]`)!
    n_code = _verify_executable_code_rows(train_rows + val_rows)
    assert_decontaminated(train_rows + val_rows + rlvr_48)

    for fname, rows in [
        ("train_v8_sft.jsonl", train_rows),
        ("orpo_v8_pairs.jsonl", orpo_pairs),
        ("rlvr_v8_hermes_48.jsonl", rlvr_48),
        ("val_v8_joint.jsonl", val_rows),
    ]:
        (V8_CORPUS_DIR / fname).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    n_turns = sum(sum(1 for m in r["messages"] if m["role"] == "assistant") for r in train_rows)
    print(
        f"[RPR v8 Hermes + Engram/AttnRes + Flash/Deep Corpus Built] {V8_CORPUS_DIR}:\n"
        f"  train_v8_sft      = {len(train_rows)} unique conversations ({n_turns} turns, 0 eval overlap)\n"
        f"  orpo_v8_pairs     = {len(orpo_pairs)} contrastive hard-trap preference pairs\n"
        f"  rlvr_v8_hermes_48 = {len(rlvr_48)} multi-domain RLVR episodes\n"
        f"  val_v8_joint      = {len(val_rows)} joint v7+v8 validation conversations\n"
        f"  Python code self-tests passed = {n_code} (100%), zero contamination verified!",
        flush=True,
    )


if __name__ == "__main__":
    main()
