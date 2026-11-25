"""Build the RPR v7 Master Corpus (`36L 2.65B` Block Expansion + Math-Spec Code Translation + `ORPO` Hard-Trap Contrastive Pairs + Option-Verified Mental Math + Long-Context Multi-Turn Sessions).

Outputs inside `training/data/rpr_v7_36l_corpus/`:
  1. `train_v7_sft.jsonl`          : ~1,400 deduplicated `1x` conversations (full 8-pillar `v5` replay + math-notation-to-NumPy/Python
                                     translation + double-bug & algorithmic repair + option-verified distributive mental math +
                                     multi-turn long-context RPR + SEAL `<weight_rewrite>`)
  2. `orpo_v7_pairs.jsonl`         : 180 explicit `(messages_prompt, chosen, rejected)` contrastive preference pairs on natural prompts
  3. `rlvr_v7_diverse_72.jsonl`    : 72 multi-domain verifiable RL episodes (Python `python3 -I`, unassisted math, 2-turn RPR, RSI gates)
  4. `val_v7_joint.jsonl`          : ~88 stratified validation conversations covering all `v5` + `v7` + long-context pillars
  5. `long_context_eval_16.jsonl`  : 16 unseen multi-turn long-context sessions at `4k`, `8k`, `16k`, and `24k` (`up to 39k` actual) tokens
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
    make_v5_clause_witness_rpr_item,
    make_v5_research_ablation_item,
)
from training.build_training_data import assert_decontaminated
from training.build_v6_repair_and_seal_corpus import (
    V6_CORPUS_DIR,
    _as_eval_task,
    make_v6_parameterized_code_item,
    make_v6_seal_svf_demo_item,
)
from training.collect_and_build_rpr_corpus import _strip_prior_think
from training.high_signal_pillars import make_diagnostic_debug_item


V7_CORPUS_DIR = Path(__file__).resolve().parent / "data" / "rpr_v7_36l_corpus"


def _format_all_options_check(
    opt_ids: list[str],
    opts: dict[str, dict[str, Any]],
    *,
    mode_req: str,
    b_feat: str,
    b_req: bool,
    cost_f: str,
    max_cost: int,
    lat_f: str,
    lim_lat: int,
    res_f: str,
    lim_res: int,
) -> str:
    """Deterministic, concise 1-clause check of EVERY option `OPT-A..OPT-F` with concrete numeric violations."""
    parts: list[str] = []
    for oid in opt_ids:
        o = opts[oid]
        if o["mode"] != mode_req:
            parts.append(f"{oid}: mode={o['mode']}!={mode_req}->FAIL")
        elif b_req and not o[b_feat]:
            parts.append(f"{oid}: {b_feat}=false!=true->FAIL")
        elif o[cost_f] > max_cost:
            parts.append(f"{oid}: {cost_f}={o[cost_f]}>{max_cost}->FAIL")
        elif o[lat_f] > lim_lat:
            parts.append(f"{oid}: {lat_f}={o[lat_f]}>{lim_lat}->FAIL")
        elif o[res_f] > lim_res:
            parts.append(f"{oid}: {res_f}={o[res_f]}>{lim_res}->FAIL")
        else:
            parts.append(f"{oid}: PASS")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# 1. Unassisted `direct` Distributive Mental-Math + Full Option Check + 2-Turn RPR
# ---------------------------------------------------------------------------


def make_unassisted_direct_math_rpr_item(rng: random.Random, idx: int) -> dict[str, Any]:
    """Teach exact distributive 2-digit mental multiplication + explicit `OPT-A..OPT-F` numeric verification."""
    regimes = [
        ("SOLAR_GRID", "HYDRO_PUMP", "sync_inverter", "array_kw", "panel_usd", "ramp_sec", "footprint_m2"),
        ("OPTICAL_BUS", "COAX_RING", "fec_reed", "lane_gbps", "port_usd", "jitter_ns", "thermal_w"),
        ("CRYO_ION", "NEUTRAL_ATOM", "magic_trap", "shot_khz", "cryo_usd", "dephasing_us", "rack_u"),
        ("ZEOLITE_BED", "MEMBRANE_PSA", "regen_valve", "flow_slpm", "skid_usd", "purge_sec", "vessel_kg"),
    ]
    mode_a, mode_b, b_feat, cap_lbl, cost_f, lat_f, res_f = regimes[idx % len(regimes)]

    # Exact 3-clause ledger `a * b + c * d - e` with distributive breakdown (identical RNG sequence for eval parity!)
    a, b = rng.randint(11, 24), rng.randint(12, 23)
    c, d = rng.randint(6, 15), rng.randint(11, 19)
    e = rng.randint(18, 49)
    ab_t, ab_u = a * ((b // 10) * 10), a * (b % 10)
    cd_t, cd_u = c * ((d // 10) * 10), c * (d % 10)
    ab = ab_t + ab_u
    cd = cd_t + cd_u
    pres_val = ab + cd - e

    cap_a = rng.choice([20, 25, 30, 40])
    mult_a = rng.choice([2, 3])
    cap_b = cap_a * mult_a
    units_b = rng.choice([2, 3])
    units_a = units_b * mult_a
    demand = cap_a * units_a

    max_cost_a = rng.randint(110, 175)
    budget_cap = units_a * max_cost_a
    max_cost_b = budget_cap // units_b

    pen_a, pen_b = rng.randint(18, 28), rng.randint(6, 13)
    gross_lat = rng.randint(52, 74)
    lim_lat_a, lim_lat_b = gross_lat - pen_a, gross_lat - pen_b

    ovh_a, ovh_b = rng.randint(20, 30), rng.randint(7, 14)
    gross_res = rng.randint(76, 98)
    lim_res_a, lim_res_b = gross_res - ovh_a, gross_res - ovh_b

    opt_ids = ["OPT-A", "OPT-B", "OPT-C", "OPT-D", "OPT-E", "OPT-F"]
    win_a, win_b = rng.sample(opt_ids, 2)
    rem = [o for o in opt_ids if o not in (win_a, win_b)]

    opts: dict[str, dict[str, Any]] = {}
    opts[win_a] = {"mode": mode_a, b_feat: False, cost_f: max_cost_a - 8, lat_f: lim_lat_a - 3, res_f: lim_res_a - 4}
    opts[win_b] = {"mode": mode_b, b_feat: True, cost_f: max_cost_b - 12, lat_f: lim_lat_b - 4, res_f: lim_res_b - 5}
    opts[rem[0]] = {"mode": mode_a, b_feat: True, cost_f: max_cost_a + 22, lat_f: lim_lat_a - 2, res_f: lim_res_a - 3}
    opts[rem[1]] = {"mode": mode_a, b_feat: False, cost_f: max_cost_a - 6, lat_f: lim_lat_a + 11, res_f: lim_res_a - 2}
    opts[rem[2]] = {"mode": mode_b, b_feat: False, cost_f: max_cost_b - 15, lat_f: lim_lat_b - 5, res_f: lim_res_b - 6}
    opts[rem[3]] = {"mode": mode_b, b_feat: True, cost_f: max_cost_b - 9, lat_f: lim_lat_b - 2, res_f: lim_res_b + 14}

    opt_lines = "\n".join(
        f"- {oid}: mode={o['mode']}, {b_feat}={str(o[b_feat]).lower()}, {cost_f}={o[cost_f]}, {lat_f}={o[lat_f]}, {res_f}={o[res_f]}"
        for oid in opt_ids
        for o in [opts[oid]]
    )

    chk_a = _format_all_options_check(
        opt_ids, opts, mode_req=mode_a, b_feat=b_feat, b_req=False,
        cost_f=cost_f, max_cost=max_cost_a, lat_f=lat_f, lim_lat=lim_lat_a, res_f=res_f, lim_res=lim_res_a,
    )
    chk_b = _format_all_options_check(
        opt_ids, opts, mode_req=mode_b, b_feat=b_feat, b_req=True,
        cost_f=cost_f, max_cost=max_cost_b, lat_f=lat_f, lim_lat=lim_lat_b, res_f=res_f, lim_res=lim_res_b,
    )

    u1 = (
        f"System Specification:\n"
        f"1. Active operating regime starts in `{mode_a}` (where `{b_feat}` is optional; `{cap_lbl} = {cap_a}` per unit; "
        f"latency penalty = `{pen_a}`; resource overhead = `{ovh_a}`).\n"
        f"   Alternate regime `{mode_b}` requires `{b_feat}=true`, `{cap_lbl} = {cap_b}` per unit, "
        f"latency penalty = `{pen_b}`, resource overhead = `{ovh_b}`.\n"
        f"2. Total demand is `{demand}` (`required_units = ceil(demand / per_unit)`). Budget cap: `required_units * {cost_f} <= {budget_cap}`.\n"
        f"3. Timing rule: `{lat_f} + active_penalty <= {gross_lat}`. Resource rule: `{res_f} <= {gross_res} - active_overhead`.\n"
        f"4. Invariant Calibration Ledger (`preserved`): `{a}` batches at `{b}` units plus `{c}` spares at `{d}` units minus `{e}` tare adjustment.\n\n"
        f"Candidate options:\n{opt_lines}\n\n"
        'Return ONLY valid JSON: `{"primary": "OPT-X", "preserved": <integer>}`.'
    )
    t1_think = (
        f"[LEDGER] Compute `{a} * {b} + {c} * {d} - {e}` by distributive split: "
        f"`{a}*{b} = {a}*{(b//10)*10} + {a}*{b%10} = {ab_t} + {ab_u} = {ab}`; "
        f"`{c}*{d} = {c}*{(d//10)*10} + {c}*{d%10} = {cd_t} + {cd_u} = {cd}`; "
        f"sum `{ab} + {cd} = {ab + cd}`; subtract `{ab + cd} - {e} = {pres_val}`.\n"
        f"[BOUNDS] Active `{mode_a}` (`{b_feat}` optional): `units = {demand} / {cap_a} = {units_a}`; "
        f"`{cost_f} <= {budget_cap} / {units_a} = {max_cost_a}`; `{lat_f} <= {gross_lat} - {pen_a} = {lim_lat_a}`; "
        f"`{res_f} <= {gross_res} - {ovh_a} = {lim_res_a}`.\n"
        f"[CHECK] {chk_a}.\n"
        f"[WINNER] Unique PASS is `{win_a}`, `preserved = {pres_val}`."
    )
    a1_json = json.dumps({"primary": win_a, "preserved": pres_val})

    is_switch = (idx % 3 != 2)
    if is_switch:
        u2 = (
            f"New evidence arrived:\n"
            f"Production telemetry switched the active operating regime from `{mode_a}` to `{mode_b}` effective immediately "
            f"(total demand `{demand}`, budget cap `{budget_cap}`, and calibration ledger `{a}*{b} + {c}*{d} - {e}` are unchanged).\n\n"
            'Provide your updated JSON `{"primary": "OPT-X", "preserved": <integer>}`.'
        )
        t2_think = (
            f"[AUDIT] Active regime switched `{mode_a} -> {mode_b}` (`{win_a}` is now INVALID). Preserve verified ledger `preserved = {pres_val}`.\n"
            f"[REBUILD] Under `{mode_b}` (`{b_feat}=true` required): `units = {demand} / {cap_b} = {units_b}`; "
            f"`{cost_f} <= {budget_cap} / {units_b} = {max_cost_b}`; `{lat_f} <= {gross_lat} - {pen_b} = {lim_lat_b}`; "
            f"`{res_f} <= {gross_res} - {ovh_b} = {lim_res_b}`.\n"
            f"[CHECK] {chk_b}.\n"
            f"[WINNER] Unique PASS is `{win_b}` (replaces invalidated `{win_a}`), `preserved = {pres_val}`."
        )
        final_exp = {"primary": win_b, "preserved": pres_val}
    else:
        u2 = (
            f"New evidence arrived:\n"
            f"An offline lab bench ran a separate diagnostic on `{mode_b}` (`999` synthetic load), "
            f"but live production stays in `{mode_a}` with zero rule or ledger changes.\n\n"
            'Provide your updated JSON `{"primary": "OPT-X", "preserved": <integer>}`.'
        )
        t2_think = (
            f"[AUDIT] Offline lab diagnostic does NOT switch live `{mode_a}` production. All `{mode_a}` bounds and ledger `{pres_val}` remain VALID.\n"
            f"[KEEP] Keep `{win_a}` and `preserved = {pres_val}`."
        )
        final_exp = {"primary": win_a, "preserved": pres_val}

    return {
        "pillar": "unassisted_direct_math_rpr",
        "images": [],
        "messages": [
            {"role": "system", "content": "You are a careful analytical problem solver. Follow all constraints strictly.\nEnd with valid JSON `{\"primary\": \"OPT-X\", \"preserved\": <integer>}`."},
            {"role": "user", "content": u1},
            {"role": "assistant", "content": f"<think>{t1_think}</think>\n{a1_json}"},
            {"role": "user", "content": u2},
            {"role": "assistant", "content": f"<think>{t2_think}</think>\n{json.dumps(final_exp)}"},
        ],
        "expected": final_exp,
    }


# ---------------------------------------------------------------------------
# 2. Natural Math-Notation-to-Python/NumPy & Multi-Gate RSI Generator
#    (ZERO spoon-fed NumPy code in user prompts; 100% unit-tested!)
# ---------------------------------------------------------------------------


def make_v7_math_spec_code_and_rsi_item(rng: random.Random, idx: int) -> dict[str, Any]:
    """Teach translation from mathematical / conceptual specs to exact Python/NumPy + full candidate gate math."""
    sys_code = rng.choice([
        "You are a frontier AI systems and Recursive Self-Improvement (RSI) engineer. Derive exact invariants in <think>, then output ONLY a ```python ... ``` block.",
        "Reason carefully through matrix shapes, numerical stability, and edge cases inside <think>, then output ONLY a ```python ... ``` code block.",
    ])
    pat = idx % 9

    if pat == 0:
        # Math-notation `Sigma_reg^-1 @ k_vec` -> `np.linalg.solve(Sigma_reg, k_vec)` (NEVER Python `^-1` XOR!)
        fn_name = rng.choice(["closed_form_rank1_edit", "kkt_linear_memory_patch", "cov_whitened_mlp_update", "rank1_associative_write"])
        reg = rng.choice([0.03, 0.05, 0.07, 0.10])
        d_out, d_hid = rng.randint(5, 9), rng.randint(9, 15)
        prompt = (
            f"Implement `{fn_name}(W_proj: np.ndarray, k_vec: np.ndarray, v_target: np.ndarray, Sigma_cov: np.ndarray, reg: float = {reg}) -> np.ndarray` "
            f"in NumPy so `W_new @ k_vec == v_target` (`W_proj` has shape `[{d_out}, {d_hid}]`, `k_vec` has shape `[{d_hid}]`, `v_target` has shape `[{d_out}]`) "
            f"while minimizing drift over key covariance `Sigma_cov` (`[{d_hid}, {d_hid}]`):\n"
            f"- Regularize `Sigma_reg = Sigma_cov + reg * np.eye(Sigma_cov.shape[0])`.\n"
            f"- Solve `u = Sigma_reg^-1 @ k_vec` and return the closed-form rank-1 update `W_proj + np.outer(v_target - W_proj @ k_vec, u) / (k_vec @ u)`."
        )
        think = (
            "[L1] Mathematical notation `u = Sigma_reg^-1 @ k_vec` means solving the linear system `Sigma_reg @ u = k_vec` via `u = np.linalg.solve(Sigma_reg, k_vec)` "
            "(NEVER write `Sigma_reg^-1` in Python because `^` is integer bitwise XOR!).\n"
            "[L2] Compute residual `res = v_target - (W_proj @ k_vec)` and scalar denominator `denom = float(np.dot(k_vec, u))`.\n"
            "[L3] Return `W_proj + np.outer(res, u) / denom`, which satisfies `W_new @ k_vec == v_target`."
        )
        code = (
            "import numpy as np\n\n"
            f"def {fn_name}(W_proj: np.ndarray, k_vec: np.ndarray, v_target: np.ndarray, Sigma_cov: np.ndarray, reg: float = {reg}) -> np.ndarray:\n"
            "    Sigma_reg = Sigma_cov + reg * np.eye(Sigma_cov.shape[0], dtype=Sigma_cov.dtype)\n"
            "    u = np.linalg.solve(Sigma_reg, k_vec)\n"
            "    denom = float(np.dot(k_vec, u))\n"
            "    return W_proj + np.outer(v_target - (W_proj @ k_vec), u) / denom"
        )
        test = (
            "import numpy as np\n"
            f"rng = np.random.default_rng({idx + 1101})\n"
            f"W = rng.normal(size=({d_out}, {d_hid}))\n"
            f"A = rng.normal(size=(20, {d_hid}))\n"
            "S = (A.T @ A) / 20.0\n"
            f"kv, vt = rng.normal(size=({d_hid},)), rng.normal(size=({d_out},))\n"
            f"Wn = {fn_name}(W, kv, vt, S)\n"
            "assert np.allclose(Wn @ kv, vt, atol=1e-9)"
        )
        return {
            "pillar": "rsi_and_weight_surgery",
            "images": [],
            "messages": [{"role": "system", "content": sys_code}, {"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>{think}</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test},
        }

    if pat == 1:
        # Natural math-spec SVD Null-Space Projection (`V_k = Vh[S > tol]`, NEVER `[v for v in Vh if S[v] > tol]`!)
        fn_name = rng.choice(["orthogonal_nullspace_edit", "anchor_preserving_weight_step", "svd_null_complement_update", "safe_subspace_weight_patch"])
        tol_e = rng.choice([6, 7, 8, 9])
        d_out, d_in = rng.randint(4, 9), rng.randint(9, 15)
        prompt = (
            f"Implement `{fn_name}(W: np.ndarray, X_keep: np.ndarray, Delta: np.ndarray, tol: float = 1e-{tol_e}) -> np.ndarray` in NumPy:\n"
            f"- `W` (`[d_out, d_in]`): linear layer weights (`y = X @ W.T`).\n"
            f"- `X_keep` (`[n_samples, d_in]`): anchor input activations that must incur zero output drift (`X_keep @ W_new.T == X_keep @ W.T`).\n"
            f"- `Delta` (`[d_out, d_in]`): raw weight update.\n"
            "Compute the right singular vectors `Vh` of `X_keep`, select the basis rows `V_k` whose singular values strictly exceed `tol`, "
            "project `Delta` onto the orthogonal complement `P_null = I - V_k.T @ V_k` via `Delta @ P_null`, and return `W + Delta_proj`."
        )
        think = (
            "[L1] `_, S, Vh = np.linalg.svd(X_keep, full_matrices=False)` gives singular values `S` (`[r]`) and right singular vectors `Vh` (`[r, d_in]`).\n"
            "[L2] Boolean mask indexing `V_k = Vh[S > tol]` (`[k, d_in]`) extracts the active row basis cleanly (never index `S` with a row vector `v`!). "
            "If `X_keep.shape[0] == 0` or `V_k.shape[0] == 0`, `P_null = I` so `W + Delta` is returned.\n"
            "[L3] Otherwise `P_null = np.eye(W.shape[1], dtype=W.dtype) - (V_k.T @ V_k)` satisfies `X_keep @ P_null == 0`; return `W + (Delta @ P_null)`."
        )
        code = (
            "import numpy as np\n\n"
            f"def {fn_name}(W: np.ndarray, X_keep: np.ndarray, Delta: np.ndarray, tol: float = 1e-{tol_e}) -> np.ndarray:\n"
            "    d_in = W.shape[1]\n"
            "    if X_keep.size == 0:\n"
            "        return W + Delta\n"
            "    _, S, Vh = np.linalg.svd(X_keep, full_matrices=False)\n"
            "    V_k = Vh[S > tol]\n"
            "    if V_k.shape[0] == 0:\n"
            "        return W + Delta\n"
            "    P_null = np.eye(d_in, dtype=W.dtype) - (V_k.T @ V_k)\n"
            "    return W + (Delta @ P_null)"
        )
        test = (
            "import numpy as np\n"
            f"rng = np.random.default_rng({idx + 1202})\n"
            f"W = rng.normal(size=({d_out}, {d_in}))\n"
            f"X = rng.normal(size=(3, {d_in}))\n"
            f"D = rng.normal(size=({d_out}, {d_in}))\n"
            f"Wn = {fn_name}(W, X, D)\n"
            "assert np.allclose(X @ Wn.T, X @ W.T, atol=1e-9)\n"
            "assert not np.allclose(Wn, W, atol=1e-4)\n"
            f"assert np.allclose({fn_name}(W, np.zeros((0, {d_in})), D), W + D)"
        )
        return {
            "pillar": "rsi_and_weight_surgery",
            "images": [],
            "messages": [{"role": "system", "content": sys_code}, {"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>{think}</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test},
        }

    if pat == 2:
        # Natural math-spec Row-Wise Norm Decomposition (`max(||V[i]||_2, eps)`)
        fn_name = rng.choice(["recompose_dora_matrix", "fuse_magnitude_direction_linear", "dora_row_norm_merge", "apply_row_magnitude_lora"])
        eps_e = rng.choice([7, 8, 9])
        d_out, d_in, r_dim = rng.randint(4, 8), rng.randint(7, 13), rng.randint(2, 4)
        prompt = (
            f"Implement `{fn_name}(W_base: np.ndarray, B: np.ndarray, A: np.ndarray, g: np.ndarray, scale: float, eps: float = 1e-{eps_e}) -> np.ndarray` in NumPy:\n"
            "- `W_base` (`[d_out, d_in]`), low-rank factors `B` (`[d_out, r]`) and `A` (`[r, d_in]`), scalar `scale = alpha / r`.\n"
            "- `g` (`[d_out]`): learned per-output-row magnitude vector (`y = x @ W.T`).\n"
            "Form directional matrix `V = W_base + scale * (B @ A)`, normalize each output row `i` by `max(||V[i]||_2, eps)`, and scale row `i` by `g[i]`."
        )
        think = (
            "[L1] `V = W_base + scale * (B @ A)` has shape `[d_out, d_in]`.\n"
            "[L2] Each row's Euclidean norm `||V[i]||_2` is `np.linalg.norm(V, axis=1, keepdims=True)` (`[d_out, 1]`), clamped below via `denom = np.maximum(norms, eps)`.\n"
            "[L3] Broadcast row scalars `g[:, None]` (`[d_out, 1]`) across columns: return `(V / denom) * g[:, None]`."
        )
        code = (
            "import numpy as np\n\n"
            f"def {fn_name}(W_base: np.ndarray, B: np.ndarray, A: np.ndarray, g: np.ndarray, scale: float, eps: float = 1e-{eps_e}) -> np.ndarray:\n"
            "    V = W_base + scale * (B @ A)\n"
            "    denom = np.maximum(np.linalg.norm(V, axis=1, keepdims=True), eps)\n"
            "    return (V / denom) * g[:, None]"
        )
        test = (
            "import numpy as np\n"
            f"rng = np.random.default_rng({idx + 1303})\n"
            f"W0 = rng.normal(size=({d_out}, {d_in}))\n"
            f"B, A = rng.normal(size=({d_out}, {r_dim})), rng.normal(size=({r_dim}, {d_in}))\n"
            f"g = np.abs(rng.normal(loc=1.8, scale=0.3, size=({d_out},))) + 0.1\n"
            f"Wm = {fn_name}(W0, B, A, g, 0.75)\n"
            f"assert Wm.shape == ({d_out}, {d_in}) and np.allclose(np.linalg.norm(Wm, axis=1), g, atol=1e-6)\n"
            f"Wz = {fn_name}(np.zeros((2, 3)), np.zeros((2, 1)), np.zeros((1, 3)), np.array([2.0, 3.0]), 1.0)\n"
            "assert np.allclose(Wz, 0.0)"
        )
        return {
            "pillar": "rsi_and_weight_surgery",
            "images": [],
            "messages": [{"role": "system", "content": sys_code}, {"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>{think}</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test},
        }

    if pat == 3:
        # Natural 3-step TIES-Merging spec (NO NumPy code in prompt!)
        fn_name = rng.choice(["ties_task_vector_merge", "elect_and_merge_deltas", "trim_elect_disjoint_merge", "combine_deltas_ties"])
        kr = rng.choice([0.25, 0.5, 0.75])
        prompt = (
            f"Implement `{fn_name}(W_init: np.ndarray, task_deltas: list[np.ndarray], density: float = {kr}, scaling: float = 1.0) -> np.ndarray` in NumPy:\n"
            "1. **Trim**: For each delta tensor in `task_deltas`, keep only the top `k = ceil(density * num_params)` entries by absolute magnitude (setting all remaining entries to `0.0`; break ties deterministically by earlier flat index).\n"
            "2. **Elect Sign**: Sum the trimmed deltas elementwise `S = sum(trimmed)` and set the elected sign to `np.sign(S)` (`+1`, `-1`, or `0`).\n"
            "3. **Disjoint Mean**: At every coordinate, average ONLY the trimmed delta values whose sign matches the non-zero elected sign (`0.0` where elected sign is `0`), and return `W_init + scaling * merged`."
        )
        think = (
            "[L1] Handle empty `task_deltas` (`return W_init.copy()`) and `density <= 0` (`return W_init.copy()`). Otherwise `k = max(1, math.ceil(density * W_init.size))`.\n"
            "[L2] For each `d` in `task_deltas`, flatten `flat = d.reshape(-1)`, pick `idx = np.argsort(-np.abs(flat), kind='stable')[:k]`, write `buf[idx] = flat[idx]` into a zero array, and reshape to `W_init.shape`.\n"
            "[L3] Stack `stk = np.stack(trimmed, axis=0)`, compute `elected = np.sign(np.sum(stk, axis=0))`, mask `match = (np.sign(stk) == elected[None, ...]) & (elected[None, ...] != 0)`, divide `np.sum(np.where(match, stk, 0.0), axis=0)` by `np.maximum(np.sum(match, axis=0), 1)`, and return `W_init + scaling * merged`."
        )
        code = (
            "import math\nimport numpy as np\n\n"
            f"def {fn_name}(W_init: np.ndarray, task_deltas: list[np.ndarray], density: float = {kr}, scaling: float = 1.0) -> np.ndarray:\n"
            "    if not task_deltas or density <= 0.0:\n"
            "        return W_init.copy()\n"
            "    k = max(1, math.ceil(density * W_init.size))\n"
            "    trimmed = []\n"
            "    for d in task_deltas:\n"
            "        flat = d.reshape(-1)\n"
            "        buf = np.zeros_like(flat, dtype=np.float64)\n"
            "        idx_top = np.argsort(-np.abs(flat), kind='stable')[:k]\n"
            "        buf[idx_top] = flat[idx_top]\n"
            "        trimmed.append(buf.reshape(W_init.shape))\n"
            "    stk = np.stack(trimmed, axis=0)\n"
            "    elected = np.sign(np.sum(stk, axis=0))\n"
            "    match = (np.sign(stk) == elected[None, ...]) & (elected[None, ...] != 0)\n"
            "    merged = np.sum(np.where(match, stk, 0.0), axis=0) / np.maximum(np.sum(match, axis=0), 1)\n"
            "    return W_init + scaling * merged"
        )
        v1, v2 = float(rng.randint(6, 9)), float(rng.randint(6, 9))
        test = (
            "import numpy as np\n"
            "W0 = np.ones((2, 2))\n"
            f"d1 = np.array([[{v1}, -5.0], [0.02, 0.01]])\n"
            f"d2 = np.array([[{v2},  1.0], [-3.0, 0.01]])\n"
            f"out = {fn_name}(W0, [d1, d2], density=0.5, scaling=1.0)\n"
            f"assert np.allclose(out, W0 + np.array([[{(v1+v2)/2}, -5.0], [-3.0, 0.0]]))"
        )
        return {
            "pillar": "rsi_and_weight_surgery",
            "images": [],
            "messages": [{"role": "system", "content": sys_code}, {"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>{think}</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test},
        }

    if pat == 4:
        # Upper-bound binary search bug repair (NEVER touch `hi = len(arr) - 1`; fix `mid = (lo + hi + 1) // 2`!)
        fn_name = rng.choice(["rightmost_at_most", "upper_feasible_index", "last_slot_within_cap", "max_valid_rank_idx"])
        arg_a, arg_b = rng.choice([("items", "cap"), ("scores", "bound"), ("weights", "budget")])
        lim = rng.randint(8, 40)
        prompt = (
            f"Fix the infinite-loop bug in `{fn_name}` below so it returns the largest index `i` in sorted `{arg_a}` with `{arg_a}[i] <= {arg_b}` (or `-1` if none). Output ONLY a ```python block:\n\n"
            "```python\n"
            f"def {fn_name}({arg_a}: list[int], {arg_b}: int) -> int:\n"
            f"    if not {arg_a} or {arg_a}[0] > {arg_b}:\n"
            "        return -1\n"
            f"    lo, hi = 0, len({arg_a}) - 1\n"
            "    while lo < hi:\n"
            "        mid = (lo + hi) // 2\n"
            f"        if {arg_a}[mid] <= {arg_b}:\n"
            "            lo = mid\n"
            "        else:\n"
            "            hi = mid - 1\n"
            "    return lo\n"
            "```"
        )
        think = (
            f"[L1] Keep `lo, hi = 0, len({arg_a}) - 1` (changing `hi` to `len({arg_a})` or `len({arg_a}) + 1` causes `IndexError` when all elements are `<= {arg_b}`!).\n"
            f"[L2] When `hi == lo + 1`, floor division `mid = (lo + hi) // 2` evaluates to `lo`; if `{arg_a}[lo] <= {arg_b}`, `lo = mid = lo` never advances -> infinite loop!\n"
            "[REBUILD] Use upper midpoint `mid = (lo + hi + 1) // 2` so `lo < mid <= hi` whenever `lo < hi`, guaranteeing termination."
        )
        code = (
            f"def {fn_name}({arg_a}: list[int], {arg_b}: int) -> int:\n"
            f"    if not {arg_a} or {arg_a}[0] > {arg_b}:\n"
            "        return -1\n"
            f"    lo, hi = 0, len({arg_a}) - 1\n"
            "    while lo < hi:\n"
            "        mid = (lo + hi + 1) // 2\n"
            f"        if {arg_a}[mid] <= {arg_b}:\n"
            "            lo = mid\n"
            "        else:\n"
            "            hi = mid - 1\n"
            "    return lo"
        )
        test = (
            f"assert {fn_name}([], {lim}) == -1\n"
            f"assert {fn_name}([{lim+1}], {lim}) == -1\n"
            f"assert {fn_name}([{lim-3}, {lim+4}], {lim}) == 0\n"
            f"assert {fn_name}([{lim-5}, {lim-1}], {lim}) == 1\n"
            f"assert {fn_name}([{lim-4}, {lim}, {lim}, {lim+2}], {lim}) == 2"
        )
        return {
            "pillar": "diagnostic_debugging",
            "images": [],
            "messages": [{"role": "system", "content": sys_code}, {"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>{think}</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test},
        }

    if pat == 5:
        # Numerically stable log_softmax across +1000 and -1000
        fn_name = rng.choice(["exact_log_softmax", "numerically_stable_log_probs", "log_softmax_shifted", "safe_log_categorical"])
        big = rng.choice([900.0, 1050.0, 1200.0])
        prompt = (
            f"Naive `math.log(math.exp(z_i) / sum(math.exp(z) for z in scores))` raises `OverflowError` on `scores = [{big}, {big+1.0}]` "
            f"and underflows to `ZeroDivisionError` on `scores = [{-big}, {-big+1.0}]`. "
            f"Write `{fn_name}(scores: list[float]) -> list[float]` that is algebraically exact and numerically stable across both extremes (returning `[]` on empty list). Output ONLY a ```python block."
        )
        think = (
            "[L1] Subtract `c = max(scores)`: `log(exp(z_i) / sum_j exp(z_j)) = (z_i - c) - log(sum_j exp(z_j - c)) = z_i - lse` where `lse = c + log(sum_j exp(z_j - c))`.\n"
            "[L2] Since every `z_j - c <= 0`, `math.exp(z_j - c)` lies in `(0, 1]` (zero overflow), and at `z_j == c` `math.exp(0) == 1.0` so the sum is `>= 1.0` (zero underflow).\n"
            "[L3] Return `[z - lse for z in scores]` (same length as `scores`)."
        )
        code = (
            "import math\n\n"
            f"def {fn_name}(scores: list[float]) -> list[float]:\n"
            "    if not scores:\n"
            "        return []\n"
            "    c = max(scores)\n"
            "    lse = c + math.log(sum(math.exp(z - c) for z in scores))\n"
            "    return [z - lse for z in scores]"
        )
        test = (
            f"assert {fn_name}([]) == []\n"
            f"a = {fn_name}([{big}, {big + 1.0}])\n"
            f"b = {fn_name}([{-big}, {-big + 1.0}])\n"
            "assert len(a) == 2 and abs(a[1] - a[0] - 1.0) < 1e-9\n"
            "assert abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9"
        )
        return {
            "pillar": "diagnostic_debugging",
            "images": [],
            "messages": [{"role": "system", "content": sys_code}, {"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>{think}</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test},
        }

    if pat == 6:
        # Shortest subarray sum >= target with negative integers (monotonic prefix deque)
        fn_name = rng.choice(["shortest_subarray_at_least", "min_window_len_ge_sum", "smallest_contiguous_ge_target", "min_len_subarray_ge"])
        kv = rng.randint(5, 22)
        prompt = (
            f"Implement `{fn_name}(arr: list[int], target: int) -> int` returning the minimum length of a non-empty contiguous subarray "
            f"with sum >= `target` (or `-1` if none exists), where `arr` MAY contain negative integers. "
            "Explain in `<think>` why two-pointer sliding window fails on negatives, then output ONLY a ```python block."
        )
        think = (
            "[L1] Two-pointer sliding window fails when `arr` has negative integers because extending or shrinking a window is non-monotonic (a negative element decreases prefix sums, so shrinking `left` can increase or decrease future sums unpredictably).\n"
            "[L2] Compute prefix sums `pref = [0, ...]` (`len(arr) + 1`) and maintain a `collections.deque` `dq` of indices with strictly increasing `pref[i]`:\n"
            "  1. Pop valid starts from the FRONT: `while dq and pj - pref[dq[0]] >= target: best = min(best, j - dq.popleft())`.\n"
            "  2. Prune dominated starts from the BACK: `while dq and pref[dq[-1]] >= pj: dq.pop()`.\n"
            "[L3] Append `j` and return `best if best <= len(arr) else -1`."
        )
        code = (
            "from collections import deque\n\n"
            f"def {fn_name}(arr: list[int], target: int) -> int:\n"
            "    pref = [0]\n"
            "    for x in arr:\n"
            "        pref.append(pref[-1] + x)\n"
            "    dq: deque[int] = deque()\n"
            "    best = len(arr) + 1\n"
            "    for j, pj in enumerate(pref):\n"
            "        while dq and pj - pref[dq[0]] >= target:\n"
            "            best = min(best, j - dq.popleft())\n"
            "        while dq and pref[dq[-1]] >= pj:\n"
            "            dq.pop()\n"
            "        dq.append(j)\n"
            "    return best if best <= len(arr) else -1"
        )
        test = (
            f"assert {fn_name}([2, -1, 2], 3) == 3\n"
            f"assert {fn_name}([84, -37, 32, 40, 95], 167) == 3\n"
            f"assert {fn_name}([{kv - 2}, -4, {kv}], {kv}) == 1\n"
            f"assert {fn_name}([1, -2, 1], 5) == -1"
        )
        return {
            "pillar": "diagnostic_debugging",
            "images": [],
            "messages": [{"role": "system", "content": sys_code}, {"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>{think}</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test},
        }

    if pat == 7:
        # BOTH bugs: late-binding loop lambda AND mutable default list!
        fn_name = rng.choice(["make_scale_closures", "build_step_Incrementers", "compile_multiplier_hooks", "register_bias_lambdas"]).lower()
        arg1, arg2, vname, op_sym = rng.choice([
            ("deltas", "call_counts", "d", "+"),
            ("factors", "seen_batches", "f", "*"),
            ("biases", "run_ledger", "b", "+"),
        ])
        prompt = (
            "Find and fix BOTH bugs in this Python helper. Output ONLY a ```python block:\n\n"
            "```python\n"
            f"def {fn_name}({arg1}: list[int], {arg2}: list[int] = []):\n"
            f"    callbacks = [lambda x: x {op_sym} {vname} for {vname} in {arg1}]\n"
            f"    {arg2}.append(len({arg1}))\n"
            f"    return callbacks, {arg2}\n"
            "```"
        )
        think = (
            f"[L1] Inspecting `{fn_name}` reveals TWO distinct Python bugs that MUST BOTH be fixed:\n"
            f"  - Bug 1 (late-binding closure): `lambda x: x {op_sym} {vname}` captures `{vname}` by reference so every lambda uses `{arg1}[-1]`. Fix with default binding `lambda x, {vname}={vname}: x {op_sym} {vname}`.\n"
            f"  - Bug 2 (mutable default argument): `{arg2}: list[int] = []` persists across independent calls when `{arg2}` is omitted! Fix with `{arg2}: list[int] | None = None` and `if {arg2} is None: {arg2} = []`.\n"
            f"[REBUILD] Apply BOTH fixes and return `(callbacks, {arg2})`."
        )
        code = (
            f"def {fn_name}({arg1}: list[int], {arg2}: list[int] | None = None):\n"
            f"    if {arg2} is None:\n"
            f"        {arg2} = []\n"
            f"    callbacks = [lambda x, {vname}={vname}: x {op_sym} {vname} for {vname} in {arg1}]\n"
            f"    {arg2}.append(len({arg1}))\n"
            f"    return callbacks, {arg2}"
        )
        v_first = 3
        v_sec = 7
        exp0 = (10 + v_first) if op_sym == "+" else (10 * v_first)
        exp1 = (10 + v_sec) if op_sym == "+" else (10 * v_sec)
        test = (
            f"cbs1, log1 = {fn_name}([{v_first}, {v_sec}])\n"
            f"assert [cb(10) for cb in cbs1] == [{exp0}, {exp1}]\n"
            "assert log1 == [2]\n"
            f"cbs2, log2 = {fn_name}([5])\n"
            "assert log2 == [1], f'Mutable default leaked across calls: {log2}'\n"
            "custom = [9]\n"
            f"_, log3 = {fn_name}([1, 2, 3], custom)\n"
            "assert log3 == [9, 3]"
        )
        return {
            "pillar": "diagnostic_debugging",
            "images": [],
            "messages": [{"role": "system", "content": sys_code}, {"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>{think}</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test},
        }

    # pat == 8: Safety-Gated Checkpoint Promotion (`CAND-A..CAND-D` explicit macro_avg arithmetic for EVERY passing candidate!)
    f_math = rng.randint(68, 76)
    f_tool = rng.randint(76, 83)
    f_code = rng.randint(70, 78)
    max_nll = round(rng.uniform(0.36, 0.43), 2)

    c_ids = ["CAND-A", "CAND-B", "CAND-C", "CAND-D"]
    rng.shuffle(c_ids)
    win_c, runner_c, fail_tool_c, fail_nll_c = c_ids

    # Runner-up passes all gates; winner passes all gates with strictly higher macro_avg!
    r_m, r_t, r_c = f_math + rng.randint(3, 8), f_tool + rng.randint(3, 7), f_code + rng.randint(4, 9)
    # Adjust r_c mod 3 if we want clean 2-decimal or exact float
    w_m, w_t, w_c = r_m + rng.randint(2, 5), r_t + rng.randint(1, 3), r_c + rng.randint(1, 4)

    cand_map = {
        win_c: {"math": w_m, "tool": w_t, "code": w_c, "nll": round(max_nll - 0.05, 2)},
        runner_c: {"math": r_m, "tool": r_t, "code": r_c, "nll": round(max_nll - 0.04, 2)},
        fail_tool_c: {"math": w_m + 4, "tool": f_tool - 5, "code": w_c + 2, "nll": round(max_nll - 0.03, 2)},
        fail_nll_c: {"math": w_m + 5, "tool": w_t + 3, "code": w_c + 4, "nll": round(max_nll + 0.09, 2)},
    }
    disp_order = ["CAND-A", "CAND-B", "CAND-C", "CAND-D"]
    rng.shuffle(disp_order)

    lines = "\n".join(
        f"- {cid}: math={cand_map[cid]['math']}%, tool={cand_map[cid]['tool']}%, code={cand_map[cid]['code']}%, val_nll={cand_map[cid]['nll']:.2f}"
        for cid in disp_order
    )
    sum_r = r_m + r_t + r_c
    avg_r = round(sum_r / 3.0, 2)
    sum_w = w_m + w_t + w_c
    avg_w = round(sum_w / 3.0, 2)

    prompt = (
        f"RSI Promotion Gate: Promote the candidate checkpoint maximizing `macro_avg = round((math + tool + code) / 3, 2)` "
        f"subject to ALL safety floors: `math >= {f_math}%`, `tool >= {f_tool}%`, `code >= {f_code}%`, and `val_nll <= {max_nll:.2f}`:\n"
        f"{lines}\n\n"
        'Return raw JSON: `{"promoted": "<CAND-ID>", "macro_avg": <number>}`.'
    )
    think = (
        f"[L1] Safety gates: `math >= {f_math}`, `tool >= {f_tool}`, `code >= {f_code}`, `val_nll <= {max_nll:.2f}`.\n"
        f"[L2] Gate check & exact `macro_avg` for EVERY passing candidate:\n"
        f"  - `{fail_tool_c}`: `tool={cand_map[fail_tool_c]['tool']} < {f_tool}` -> REJECT.\n"
        f"  - `{fail_nll_c}`: `val_nll={cand_map[fail_nll_c]['nll']:.2f} > {max_nll:.2f}` -> REJECT.\n"
        f"  - `{runner_c}`: PASSES all gates; sum `{r_m} + {r_t} + {r_c} = {sum_r}`, `macro_avg = {sum_r} / 3 = {avg_r}`.\n"
        f"  - `{win_c}`: PASSES all gates; sum `{w_m} + {w_t} + {w_c} = {sum_w}`, `macro_avg = {sum_w} / 3 = {avg_w}`.\n"
        f"[L3] Compare passing candidates directly: `{avg_w} > {avg_r}`, so promote `{win_c}` with `macro_avg = {avg_w}`."
    )
    exp_json = {"promoted": win_c, "macro_avg": avg_w}
    return {
        "pillar": "rsi_and_weight_surgery",
        "kind": "json_values",
        "images": [],
        "messages": [
            {"role": "system", "content": "You are a rigorous RSI evaluation gatekeeper. Check every safety gate and compute exact arithmetic for every passing candidate in <think>, then output raw JSON."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"<think>{think}</think>\n{json.dumps(exp_json)}"},
        ],
        "expected": exp_json,
        "verifier": exp_json,
    }


# ---------------------------------------------------------------------------
# 3. Multi-Turn Long-Context Session Generator (Training `2k-4k` & Unseen Eval `4k-24k`!)
# ---------------------------------------------------------------------------


def _build_realistic_distractor_log(rng: random.Random, turn_no: int, target_chars: int) -> str:
    """Synthesize dense, realistic multi-subsystem engineering telemetry & code logs (~target_chars) with zero active regime changes."""
    subsystems = [
        ("telemetry_ingestor", "kafka_lag_ms", "ring_buffer_fill_pct", "crc32_ok"),
        ("checkpoint_daemon", "nvme_flush_mb_s", "fsync_p99_ms", "atomic_rename_ok"),
        ("kv_eviction_pool", "rot_cache_occupancy", "page_faults", "slab_freelist"),
        ("sandbox_profiler", "offline_canary_ops", "shadow_grad_norm", "isolated_replay"),
    ]
    sub_name, m1, m2, flag = subsystems[turn_no % len(subsystems)]
    chunks = [
        f"### [Session Log Turn {turn_no} — Non-Production Subsystem `{sub_name}` Telemetry Digest]\n"
        f"Note: This digest records auxiliary background metrics from isolated `{sub_name}` shards ONLY; "
        "it does NOT modify the primary cluster operating regime, demand, budget, or Turn-1 calibration ledger."
    ]
    row_id = 0
    while sum(len(c) for c in chunks) < target_chars:
        row_id += 1
        v1 = round(rng.uniform(1.2, 48.9), 2)
        v2 = round(rng.uniform(12.0, 94.5), 1)
        h_hex = f"{rng.randint(0x100000, 0xFFFFFF):06x}"
        chunks.append(
            f"- `shard_{turn_no:02d}_{row_id:03d}` (`commit={h_hex}`): `{m1}={v1}`, `{m2}={v2}`, "
            f"`{flag}=true`, `aux_counter={rng.randint(100, 999)}`, `status=ISOLATED_AUX_OK`"
        )
    return "\n".join(chunks)


def make_long_context_session_item(
    rng: random.Random,
    idx: int,
    *,
    target_total_tokens: int = 2400,
    for_eval: bool = False,
) -> dict[str, Any]:
    """Create a multi-turn session where Turn 1 sets an invariant ledger + initial regime, intermediate turns inject
    thousands of tokens of dense subsystem logs & audit queries, and the final turn either switches the regime or adds
    an irrelevant sandbox note — testing true long-range retrieval across `4k..39k` tokens!
    """
    base_item = make_unassisted_direct_math_rpr_item(rng, idx)
    u1 = base_item["messages"][1]["content"]
    a1 = base_item["messages"][2]["content"]
    u_final = base_item["messages"][3]["content"]
    a_final = base_item["messages"][4]["content"]
    exp = base_item["expected"]
    init_exp = json.loads(a1.split("</think>")[-1].strip())

    rem_tokens = max(600, target_total_tokens - 650)
    n_mid_turns = 2 if target_total_tokens <= 3000 else (4 if target_total_tokens <= 9000 else (7 if target_total_tokens <= 17000 else 10))
    chars_per_turn = int((rem_tokens * 3.65) / max(1, n_mid_turns))

    msgs: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a long-context Revisable Pyramid Reasoning (RPR) engineering assistant. "
                "Track Turn-1 foundational invariants (`preserved` ledger) accurately across arbitrarily long multi-turn logs, "
                "ignore isolated/auxiliary subsystem noise, and rebuild downstream option selections ONLY when live production premises change."
            ),
        },
        {"role": "user", "content": u1},
        {"role": "assistant", "content": a1},
    ]

    for m_i in range(1, n_mid_turns + 1):
        log_blob = _build_realistic_distractor_log(rng, m_i, chars_per_turn)
        msgs.append({
            "role": "user",
            "content": (
                f"{log_blob}\n\n"
                f"Checkpoint query {m_i}: Did this `{m_i}` digest alter the live operating regime or Turn-1 `preserved` ledger? "
                'Reply briefly after `</think>` with current JSON `{"primary": "OPT-X", "preserved": <integer>}`.'
            ),
        })
        msgs.append({
            "role": "assistant",
            "content": (
                f"<think>[AUDIT] Digest {m_i} is isolated auxiliary telemetry; live regime `{init_exp['primary']}` and Turn-1 ledger `{init_exp['preserved']}` are unchanged.</think>\n"
                + json.dumps(init_exp)
            ),
        })

    msgs.append({"role": "user", "content": u_final})
    if for_eval:
        is_premise_switch = "switched the active operating regime" in u_final
        return {
            "id": f"longctx-{target_total_tokens//1024}k-{idx+1:02d}",
            "pillar": "long_context_rpr",
            "target_tokens": target_total_tokens,
            "category": "premise_change" if is_premise_switch else "irrelevant_update",
            "kind": "math_opt",
            "images": [],
            "messages": _strip_prior_think(msgs),
            "expected_initial": init_exp,
            "verifier": {"primary": exp["primary"], "preserved": exp["preserved"], "forbid_tool_call": True},
        }

    msgs.append({"role": "assistant", "content": a_final})
    return {
        "pillar": "long_context_session_rpr",
        "images": [],
        "messages": msgs,
        "expected": exp,
    }


# ---------------------------------------------------------------------------
# 4. Contrastive Preference (`ORPO`) Hard-Trap Pairs on Natural Prompts!
# ---------------------------------------------------------------------------


def make_v7_orpo_contrastive_pair(rng: random.Random, idx: int) -> dict[str, Any]:
    """Build explicit `(messages, chosen, rejected)` contrastive pairs on NATURAL math-notation/bug prompts."""
    kind = idx % 10

    if kind == 0:
        # Prompt CONTAINS `Sigma_reg^-1 @ k_vec`; chosen uses `np.linalg.solve`, rejected echoes `Sigma_reg^-1`!
        ex = make_v7_math_spec_code_and_rsi_item(rng, idx * 9 + 0)
        chosen = ex["messages"][2]["content"]
        rejected = chosen.replace("np.linalg.solve(Sigma_reg, k_vec)", "Sigma_reg^-1 @ k_vec")
        return {"id": f"orpo-{idx:03d}", "messages": ex["messages"][:2], "chosen": chosen, "rejected": rejected}

    if kind == 1:
        # Natural SVD prompt: `Vh[S > tol]` vs `[v for v in Vh if S[v] > tol]`!
        ex = make_v7_math_spec_code_and_rsi_item(rng, idx * 9 + 1)
        chosen = ex["messages"][2]["content"]
        rejected = chosen.replace("V_k = Vh[S > tol]", "V_k = [v for v in Vh if S[v] > tol]")
        return {"id": f"orpo-{idx:03d}", "messages": ex["messages"][:2], "chosen": chosen, "rejected": rejected}

    if kind == 2:
        # Natural DoRA prompt: `(V / denom) * g[:, None]` vs `np.eye` / `dtype=float` bug!
        ex = make_v7_math_spec_code_and_rsi_item(rng, idx * 9 + 2)
        chosen = ex["messages"][2]["content"]
        rejected = chosen.replace(
            "denom = np.maximum(np.linalg.norm(V, axis=1, keepdims=True), eps)\n    return (V / denom) * g[:, None]",
            "denom = np.linalg.norm(V, axis=0, keepdims=True, dtype=float)\n    return W_base + (V / denom) * g",
        )
        return {"id": f"orpo-{idx:03d}", "messages": ex["messages"][:2], "chosen": chosen, "rejected": rejected}

    if kind == 3:
        # Natural TIES prompt: per-delta `argsort(-abs(flat))[:k]` vs `np.linalg.norm(task_deltas, ord='fro')`!
        ex = make_v7_math_spec_code_and_rsi_item(rng, idx * 9 + 3)
        chosen = ex["messages"][2]["content"]
        rejected = (
            "<think>[L1] Compute Frobenius norm of task_deltas list directly.</think>\n"
            "```python\nimport numpy as np\n\ndef ties_broken(W_init, task_deltas, density=0.5, scaling=1.0):\n"
            "    m = np.linalg.norm(task_deltas, ord='fro')\n"
            "    return W_init + scaling * np.mean(task_deltas, axis=0)\n```"
        )
        return {"id": f"orpo-{idx:03d}", "messages": ex["messages"][:2], "chosen": chosen, "rejected": rejected}

    if kind == 4:
        # Binary search: `mid = (lo + hi + 1) // 2` vs `hi = len(arr) + 1` IndexError!
        ex = make_v7_math_spec_code_and_rsi_item(rng, idx * 9 + 4)
        chosen = ex["messages"][2]["content"]
        rejected = chosen.replace("mid = (lo + hi + 1) // 2", "mid = (lo + hi) // 2").replace("- 1\n    while", "+ 1\n    while")
        return {"id": f"orpo-{idx:03d}", "messages": ex["messages"][:2], "chosen": chosen, "rejected": rejected}

    if kind == 5:
        # Stable log_softmax: `[z - lse for z in scores]` vs filtering `if z > c`!
        ex = make_v7_math_spec_code_and_rsi_item(rng, idx * 9 + 5)
        chosen = ex["messages"][2]["content"]
        rejected = chosen.replace(
            "lse = c + math.log(sum(math.exp(z - c) for z in scores))\n    return [z - lse for z in scores]",
            "return [z - c for z in scores if z > c] + [c]",
        )
        return {"id": f"orpo-{idx:03d}", "messages": ex["messages"][:2], "chosen": chosen, "rejected": rejected}

    if kind == 6:
        # Monotonic deque: `dq.popleft()` + `dq.pop()` vs single `stack.pop()`!
        ex = make_v7_math_spec_code_and_rsi_item(rng, idx * 9 + 6)
        chosen = ex["messages"][2]["content"]
        rejected = chosen.replace("dq.popleft()", "dq.pop()").replace("while dq and pref[dq[-1]] >= pj:\n            dq.pop()\n        ", "")
        return {"id": f"orpo-{idx:03d}", "messages": ex["messages"][:2], "chosen": chosen, "rejected": rejected}

    if kind == 7:
        # Double-bug helper: fixing BOTH closure AND mutable default vs fixing only closure!
        ex = make_v7_math_spec_code_and_rsi_item(rng, idx * 9 + 7)
        chosen = ex["messages"][2]["content"]
        rejected = chosen.replace(": list[int] | None = None", ": list[int] = []").replace("    if ", "    if False and ")
        return {"id": f"orpo-{idx:03d}", "messages": ex["messages"][:2], "chosen": chosen, "rejected": rejected}

    if kind == 8:
        # Safety-gated `CAND-A..D` promotion: exact macro_avg comparison vs picking runner-up without computing both!
        ex = make_v7_math_spec_code_and_rsi_item(rng, idx * 9 + 8)
        chosen = ex["messages"][2]["content"]
        exp_d = ex["expected"]
        wrong_c = next(c for c in ("CAND-A", "CAND-B", "CAND-C", "CAND-D") if c != exp_d["promoted"])
        rejected = f'<think>[L1] Skip exact sum and guess `{wrong_c}`.</think>\n{json.dumps({"promoted": wrong_c, "macro_avg": round(exp_d["macro_avg"] - 1.5, 2)})}'
        return {"id": f"orpo-{idx:03d}", "messages": ex["messages"][:2], "chosen": chosen, "rejected": rejected}

    # kind == 9: Explicit `OPT-A..OPT-F` numeric check on Turn-2 Premise Revision vs shorthand `bound->FAIL` guess!
    ex = make_unassisted_direct_math_rpr_item(rng, idx * 3)
    msgs_prompt = _strip_prior_think(ex["messages"][:-1])
    chosen = ex["messages"][-1]["content"]
    init_ans = json.loads(ex["messages"][2]["content"].split("</think>")[-1].strip())
    bad_ans = {"primary": init_ans["primary"], "preserved": ex["expected"]["preserved"] - 17}
    rejected = f"<think>[AUDIT] OPT-A: bound->FAIL; OPT-B: bound->FAIL; keep old option without checking numbers.</think>\n{json.dumps(bad_ans)}"
    return {"id": f"orpo-{idx:03d}", "messages": msgs_prompt, "chosen": chosen, "rejected": rejected}


def main() -> None:
    V7_CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    rng_tr = random.Random(20261401)
    rng_op = random.Random(20261402)
    rng_rl = random.Random(20261403)
    rng_va = random.Random(20261404)
    rng_lc = random.Random(20261499)

    # 1. Build 16 unseen Long-Context Evaluation Sessions (`4k`, `8k`, `16k`, `24k` tokens — identical prompts for apples-to-apples!)
    long_eval_16: list[dict[str, Any]] = []
    for k_i, tgt_toks in enumerate((4096, 8192, 16384, 24576)):
        for s_j in range(4):
            raw_idx = k_i * 12 + (2 if s_j == 2 else s_j * 3)
            lc_item = make_long_context_session_item(rng_lc, raw_idx, target_total_tokens=tgt_toks, for_eval=True)
            lc_item["id"] = f"longctx-{tgt_toks//1024}k-{s_j+1:02d}"
            long_eval_16.append(lc_item)

    # Collect all held-out user prompts (`heldout_v5_rsi_56` + `final_test_24` + `long_context_eval_16`) for zero-overlap guard
    forbid_prompts: set[str] = set()
    for p_file in [V5_CORPUS_DIR / "heldout_v5_rsi_56.jsonl"]:
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
    for lc_ex in long_eval_16:
        for m in lc_ex.get("messages", []):
            if m.get("role") == "user":
                forbid_prompts.add(str(m.get("content", "")).strip())

    def _clean(row: dict[str, Any]) -> bool:
        return not any(
            m.get("role") == "user" and str(m.get("content", "")).strip() in forbid_prompts
            for m in row.get("messages", [])
        )

    # 2. Assemble `train_v7_sft.jsonl`: FULL 8-pillar `v5` replay + Natural Math-Spec Code/RSI + Option-Checked Math + Long-Context + SEAL!
    v5_rows = [
        json.loads(l)
        for l in (V5_CORPUS_DIR / "train_rpr_v5.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    train_rows: list[dict[str, Any]] = [r for r in v5_rows if _clean(r)]

    for i in range(252):  # 28 instances of each of the 9 natural math-spec / multi-bug / gate-promotion patterns!
        ms_it = make_v7_math_spec_code_and_rsi_item(rng_tr, i)
        if _clean(ms_it):
            train_rows.append(ms_it)
    for i in range(120):
        c_it = make_v6_parameterized_code_item(rng_tr, i)
        if _clean(c_it):
            train_rows.append(c_it)
    for i in range(36):
        d_it = make_diagnostic_debug_item(rng_tr, i)
        if _clean(d_it):
            train_rows.append(d_it)
    for i in range(120):
        m_it = make_unassisted_direct_math_rpr_item(rng_tr, i)
        if _clean(m_it):
            train_rows.append(m_it)
    for i in range(64):
        lc_tr = make_long_context_session_item(rng_tr, i + 200, target_total_tokens=rng_tr.choice([1800, 2600, 3400]), for_eval=False)
        if _clean(lc_tr):
            train_rows.append(lc_tr)
    for i in range(48):
        s_it = make_v6_seal_svf_demo_item(rng_tr, i)
        if _clean(s_it):
            train_rows.append(s_it)

    train_rows = [r for r in _deduplicate_rows(train_rows) if _clean(r)]
    rng_tr.shuffle(train_rows)
    for idx, r in enumerate(train_rows, start=1):
        r["id"] = f"train_v7-{idx:04d}"
        r.setdefault("images", [])

    # 3. Build 180 Contrastive `ORPO` Hard-Trap Pairs on Natural Prompts
    orpo_pairs: list[dict[str, Any]] = []
    for i in range(180):
        pair = make_v7_orpo_contrastive_pair(rng_op, i)
        if _clean(pair):
            orpo_pairs.append(pair)

    # 4. Build 72 Multi-Domain Verifiable `RLVR` Episodes (heavily covering natural math-spec code + RSI gates + 2-turn RPR)
    rlvr_72: list[dict[str, Any]] = []
    for i in range(36):
        ex = make_v7_math_spec_code_and_rsi_item(rng_rl, i)
        t = _as_eval_task(ex, f"v7-rlvr-mathspec-{i+1:02d}")
        if _clean(t):
            rlvr_72.append(t)
    for i in range(18):
        t = _as_eval_task(make_unassisted_direct_math_rpr_item(rng_rl, i), f"v7-rlvr-directmath-{i+1:02d}")
        if _clean(t):
            rlvr_72.append(t)
    for i in range(12):
        t = _as_eval_task(make_v5_clause_witness_rpr_item(rng_rl, i), f"v7-rlvr-rpr2t-{i+1:02d}")
        if _clean(t):
            rlvr_72.append(t)
    for i in range(6):
        t = _as_eval_task(make_v5_research_ablation_item(rng_rl, i), f"v7-rlvr-res-{i+1:02d}")
        if _clean(t):
            rlvr_72.append(t)
    rng_rl.shuffle(rlvr_72)

    # 5. Build Joint Validation Set (`val_v7_joint.jsonl`: 40 from `val_rpr_v5` + 48 from `v7` natural specs/math/long-context!)
    v5_val = [
        json.loads(l)
        for l in (V5_CORPUS_DIR / "val_rpr_v5.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    val_rows: list[dict[str, Any]] = [r for r in v5_val[:40] if _clean(r)]
    for i in range(18):
        val_rows.append(make_v7_math_spec_code_and_rsi_item(rng_va, i))
    for i in range(12):
        val_rows.append(make_unassisted_direct_math_rpr_item(rng_va, i))
    for i in range(8):
        val_rows.append(make_long_context_session_item(rng_va, i + 500, target_total_tokens=2200, for_eval=False))
    for i in range(6):
        val_rows.append(make_v6_seal_svf_demo_item(rng_va, i))
    val_rows = [r for r in _deduplicate_rows(val_rows) if _clean(r)]
    for idx, r in enumerate(val_rows, start=1):
        r["id"] = f"val_v7-{idx:04d}"
        r.setdefault("images", [])

    # Verify 100% of executable Python snippets & zero benchmark contamination!
    n_code = _verify_executable_code_rows(train_rows + val_rows)
    assert_decontaminated(train_rows + val_rows + rlvr_72 + long_eval_16)

    for fname, rows in [
        ("train_v7_sft.jsonl", train_rows),
        ("orpo_v7_pairs.jsonl", orpo_pairs),
        ("rlvr_v7_diverse_72.jsonl", rlvr_72),
        ("val_v7_joint.jsonl", val_rows),
        ("long_context_eval_16.jsonl", long_eval_16),
    ]:
        (V7_CORPUS_DIR / fname).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    n_turns = sum(sum(1 for m in r["messages"] if m["role"] == "assistant") for r in train_rows)
    print(
        f"[RPR v7.1 Master + Math-Spec + ORPO + Long-Context Corpus Built] {V7_CORPUS_DIR}:\n"
        f"  train_v7_sft        = {len(train_rows)} unique conversations ({n_turns} turns, 0 eval overlap)\n"
        f"  orpo_v7_pairs       = {len(orpo_pairs)} contrastive hard-trap preference pairs on natural prompts\n"
        f"  rlvr_v7_diverse_72  = {len(rlvr_72)} multi-domain RLVR episodes\n"
        f"  val_v7_joint        = {len(val_rows)} joint v5+v7+long-context validation conversations\n"
        f"  long_context_eval_16= {len(long_eval_16)} unseen 4k/8k/16k/24k multi-turn sessions\n"
        f"  Python code self-tests passed = {n_code} (100%), zero contamination verified!",
        flush=True,
    )


if __name__ == "__main__":
    main()
