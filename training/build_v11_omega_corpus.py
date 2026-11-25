"""Master Decontaminated Corpus Builder for `RPR v11 OMEGA` (`LFM3-VL-3.2B-RPR-V11-OMEGA-BF16`).

Generates `training/data/rpr_v11_omega_corpus/`:
  - `train_v11_sft.jsonl`: ~1,500+ deduplicated conversations (~2,000+ turns) with rich, free exploratory
    `<think>` Self-Discover & Counterexample-Testing traces across Weight-Mod/RSI, Research/Citations/OOD Discovery,
    Zero-Prior Formal & Complex Logic/SAT, Algorithmic Invariants/Debugging, Calibrated Tools, 2-Turn RPR, & Vision.
  - `train_v11_long_kv.jsonl`: 80 true `8k..26k`-token multi-needle + superseding-erratum + `[REC-XXXX]`/`[LOG-XXXX]`
    inline-citation archives trained via Zero-Swap Prefix-KV Cached Backpropagation!
  - `val_v11_joint.jsonl`: Stratified heldout-disjoint validation split.
  - `orpo_v11_pairs.jsonl`: 320 high-contrast preference pairs curing subtle Python/NumPy/SVD/Dict/Tool-Path/Binary-Search traps.
  - `dapo_v11_episodes.jsonl`: 140 hard multi-domain verifiable episodes for `DAPO + Dr. GRPO` with free `<think>` (`max_think=900`).

Strictly enforces `assert_decontaminated` + `_assert_zero_heldout_leakage` across ALL held-out suites (`56`, `zp24`, `apex36`, `long12`, `pilot`, `benchmark_100`) and executes EVERY synthesized Python function in isolated `python3 -I`!
"""

from __future__ import annotations

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
from training.verifiable_rewards import score_response


V11_CORPUS_DIR = ROOT_DIR / "training" / "data" / "rpr_v11_omega_corpus"


# ===========================================================================
# 1. Deep Weight-Modification, NumPy Geometry & Autonomous RSI Traces
# ===========================================================================


def make_v11_weight_mod_rsi_item(rng: random.Random, idx: int) -> dict[str, Any]:
    variant = idx % 10

    if variant == 0:
        # 1-return `compute_uv=False` vs 3-return `full_matrices=False` SVD Spectral Operator Norm / Nuclear Scaling
        fn = rng.choice(["clamp_operator_norm", "bound_matrix_spectral_radius", "rescale_above_sigma_cap"])
        if idx >= 30:
            fn = f"{fn}_{idx % 13}"
        cap_ex = rng.choice([2.5, 3.0, 4.0])
        prompt = (
            f"Implement `{fn}(M: np.ndarray, sigma_cap: float) -> np.ndarray` in NumPy without mutating `M`:\n"
            "- Compute the largest singular value `s_max` of 2D matrix `M` in `float64`.\n"
            "- If `s_max > sigma_cap` (with `sigma_cap > 0`), uniformly scale `M` by `sigma_cap / s_max` so its spectral norm equals `sigma_cap`; otherwise return a `float64` copy of `M` unchanged."
        )
        think = (
            "Let's check NumPy's `svd` API carefully before writing code so we don't hit an unpacking error:\n"
            "- `np.linalg.svd(A, full_matrices=False)` (`compute_uv=True` default) returns a **3-tuple** `(U, S, Vt)`.\n"
            "- `np.linalg.svd(A, compute_uv=False)` returns a **single 1D `ndarray`** `s_vals` of singular values sorted descending (`s_vals[0]` is `s_max`) — unpacking `_, s_vals = svd(..., compute_uv=False)` would crash with `ValueError`!\n"
            "- Edge check: if `M` is all zeros (`s_max == 0.0`) and `sigma_cap > 0`, `s_max > sigma_cap` is `False`, returning a clean `float64` copy without dividing by zero.\n"
            "- Test mentally on `diag([6.0, 2.0])` with `sigma_cap = 3.0`: `s_max = 6.0 > 3.0`, scale `0.5` -> `diag([3.0, 1.0])`, operator norm `3.0`."
        )
        code = (
            "import numpy as np\n\n"
            f"def {fn}(M: np.ndarray, sigma_cap: float) -> np.ndarray:\n"
            "    work = np.array(M, dtype=np.float64, copy=True)\n"
            "    s_vals = np.linalg.svd(work, compute_uv=False)\n"
            "    s_max = float(s_vals[0]) if s_vals.size > 0 else 0.0\n"
            "    if s_max > sigma_cap:\n"
            "        work = work * (sigma_cap / s_max)\n"
            "    return work"
        )
        test = (
            "import numpy as np\n"
            f"A = np.diag([10.0, 2.0])\nB = {fn}(A, {cap_ex})\n"
            "np.testing.assert_allclose(A, np.diag([10.0, 2.0]))\n"
            f"np.testing.assert_allclose(np.linalg.svd(B, compute_uv=False)[0], {cap_ex}, atol=1e-7)\n"
            f"C = np.diag([1.0, 0.5])\nnp.testing.assert_allclose({fn}(C, {cap_ex}), C, atol=1e-9)\n"
        )
        return {
            "id": f"v11_wmod_{idx:04d}",
            "pillar": "rsi_and_weight_surgery",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
        }

    if variant == 1:
        # Weight-Dict `dict[str, np.ndarray]` Safe Interpolation & Promotion Gate (NEVER `float * dict`!)
        fn = rng.choice(["safe_checkpoint_soup_gate", "guarded_state_dict_blend", "accept_or_revert_weights"])
        if idx >= 30:
            fn = f"{fn}_{idx % 13}"
        prompt = (
            f"Implement `{fn}(base_sd: dict[str, np.ndarray], cand_sd: dict[str, np.ndarray], base_acc: float, cand_acc: float, base_loss: float, cand_loss: float, loss_slack: float = 0.02, mix: float = 0.75) -> tuple[bool, dict[str, np.ndarray]]` in Python/NumPy:\n"
            "- Accept the candidate iff `cand_acc > base_acc` AND `cand_loss <= base_loss + loss_slack`.\n"
            "- If accepted, return `(True, blended)` where each key `k` maps to `((1.0 - mix) * base_sd[k] + mix * cand_sd[k]).astype(np.float64)`.\n"
            "- If rejected, return `(False, copy_of_base)` with fresh `float64` array copies so mutating the returned dict never mutates `base_sd`."
        )
        think = (
            "Let's trace both branches and check Python types carefully:\n"
            "1. `base_sd` and `cand_sd` are Python `dict[str, np.ndarray]` objects — writing `(1 - mix) * base_sd + mix * cand_sd` directly on `dict` raises `TypeError: unsupported operand type(s) for *: 'float' and 'dict'`! We MUST use a dictionary comprehension `{k: ... for k in base_sd}`.\n"
            "2. Check boundary inequalities:\n"
            "   - `cand_acc > base_acc` is strict: equal accuracy (`cand_acc == base_acc`) must reject (`False`).\n"
            "   - `cand_loss <= base_loss + loss_slack` is inclusive (`<=`).\n"
            "3. Non-mutation invariant: on rejection, `{k: np.array(v, dtype=np.float64, copy=True) for k, v in base_sd.items()}` ensures deep array copies rather than shallow `base_sd.copy()`."
        )
        code = (
            "import numpy as np\n\n"
            f"def {fn}(\n"
            "    base_sd: dict[str, np.ndarray],\n"
            "    cand_sd: dict[str, np.ndarray],\n"
            "    base_acc: float,\n"
            "    cand_acc: float,\n"
            "    base_loss: float,\n"
            "    cand_loss: float,\n"
            "    loss_slack: float = 0.02,\n"
            "    mix: float = 0.75,\n"
            ") -> tuple[bool, dict[str, np.ndarray]]:\n"
            "    if cand_acc > base_acc and cand_loss <= base_loss + loss_slack:\n"
            "        blended = {\n"
            "            k: ((1.0 - mix) * base_sd[k].astype(np.float64) + mix * cand_sd[k].astype(np.float64)).astype(np.float64)\n"
            "            for k in base_sd\n"
            "        }\n"
            "        return True, blended\n"
            "    reverted = {k: np.array(v, dtype=np.float64, copy=True) for k, v in base_sd.items()}\n"
            "    return False, reverted"
        )
        test = (
            "import numpy as np\n"
            "b = {'w': np.array([2.0, 4.0])}\n"
            "c = {'w': np.array([6.0, 8.0])}\n"
            f"ok1, out1 = {fn}(b, c, 0.70, 0.80, 0.50, 0.51, loss_slack=0.02, mix=0.5)\n"
            "assert ok1 is True\n"
            "np.testing.assert_allclose(out1['w'], np.array([4.0, 6.0]))\n"
            f"ok2, out2 = {fn}(b, c, 0.70, 0.80, 0.50, 0.55, loss_slack=0.02, mix=0.5)\n"
            "assert ok2 is False\n"
            "out2['w'][0] = 999.0\n"
            "np.testing.assert_allclose(b['w'], np.array([2.0, 4.0]))\n"
        )
        return {
            "id": f"v11_wmod_{idx:04d}",
            "pillar": "rsi_and_weight_surgery",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
        }

    if variant == 2:
        # Rectangular SVD Singular Scaling (`(U * (S * scales)[None, :]) @ Vt` — never omitting `@ Vt`!)
        fn = rng.choice(["modulate_singular_spectrum", "apply_svf_gain_vector", "svd_spectrum_transform"])
        if idx >= 30:
            fn = f"{fn}_{idx % 13}"
        prompt = (
            f"Implement `{fn}(W: np.ndarray, gains: np.ndarray) -> np.ndarray` in NumPy without mutating `W`:\n"
            "- Factor rectangular `W` of shape `(d_out, d_in)` via thin SVD `U, S, Vt = np.linalg.svd(W.astype(np.float64), full_matrices=False)` (`k = min(d_out, d_in)`).\n"
            "- Scale singular values elementwise by 1D `gains` of shape `(k,)` (`S_mod = S * gains`) and reconstruct `U @ diag(S_mod) @ Vt` in `float64`."
        )
        think = (
            "Let's trace every matrix shape on a non-square example `(d_out=5, d_in=3)` (`k = 3`) and check the identity law:\n"
            "- `U` has shape `(5, 3)`, `S` has shape `(3,)`, `Vt` has shape `(3, 3)`.\n"
            "- `S_mod = S * gains.astype(np.float64)` has shape `(3,)`.\n"
            "- `U * S_mod[None, :]` scales each column `j` of `U` by `S_mod[j]`, giving `(5, 3)`.\n"
            "- CRITICAL: we MUST right-multiply `@ Vt` (`(5, 3) @ (3, 3) -> (5, 3)`)! Omitting `@ Vt` would fail the universal identity check `fn(W, np.ones(k)) == W` whenever `Vt != I`!"
        )
        code = (
            "import numpy as np\n\n"
            f"def {fn}(W: np.ndarray, gains: np.ndarray) -> np.ndarray:\n"
            "    U, S, Vt = np.linalg.svd(W.astype(np.float64), full_matrices=False)\n"
            "    S_mod = S * gains.astype(np.float64)\n"
            "    return (U * S_mod[None, :]) @ Vt"
        )
        test = (
            "import numpy as np\n"
            "rng = np.random.default_rng(7)\n"
            "W = rng.standard_normal((5, 3))\n"
            f"np.testing.assert_allclose({fn}(W, np.ones(3)), W, atol=1e-9)\n"
            f"np.testing.assert_allclose({fn}(W, np.full(3, 2.5)), 2.5 * W, atol=1e-9)\n"
        )
        return {
            "id": f"v11_wmod_{idx:04d}",
            "pillar": "rsi_and_weight_surgery",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
        }

    if variant == 3:
        # Multi-Key Linear / Sylvester Weight Edit (`Delta @ M = R @ K.T` -> `np.linalg.solve(M.T, (R @ K.T).T).T`)
        fn = rng.choice(["ridge_key_value_weight_update", "closed_form_assoc_memory_edit", "damped_multi_key_patch"])
        if idx >= 30:
            fn = f"{fn}_{idx % 13}"
        prompt = (
            f"Implement `{fn}(W: np.ndarray, keys: np.ndarray, targets: np.ndarray, cov_prior: np.ndarray, lam: float = 0.05) -> np.ndarray` in NumPy without mutating `W`:\n"
            "- `W`: `(d_out, d_in)`, `keys`: `(d_in, m)`, `targets`: `(d_out, m)`, symmetric PSD `cov_prior`: `(d_in, d_in)`.\n"
            "- Compute residual `R = targets - W @ keys` (`(d_out, m)`) and normal matrix `M = cov_prior + keys @ keys.T + lam * I_{d_in}` (`(d_in, d_in)`).\n"
            "- Solve `Delta @ M = R @ keys.T` for `Delta` (`(d_out, d_in)`) using `np.linalg.solve` (no explicit `inv`) and return `(W + Delta).astype(np.float64)`."
        )
        think = (
            "Let's verify transpose shapes carefully so `np.linalg.solve` never raises a dimension mismatch:\n"
            "- `R = targets - W @ keys` has shape `(d_out, m)`.\n"
            "- `RHS = R @ keys.T` has shape `(d_out, m) @ (m, d_in) = (d_out, d_in)` (NOT `R.T @ keys.T`, which would fail when `d_out != m`!).\n"
            "- We want `Delta @ M = RHS`. Taking transposes gives `M.T @ Delta.T = RHS.T`, where `M.T` is `(d_in, d_in)` and `RHS.T` is `(d_in, d_out)`.\n"
            "- `np.linalg.solve(M.T, RHS.T)` returns `Delta.T` of shape `(d_in, d_out)`, so `.T` gives `Delta` of shape `(d_out, d_in)`!\n"
            "- Zero-residual check: when `targets == W @ keys`, `R == 0` -> `Delta == 0` -> returns `W`."
        )
        code = (
            "import numpy as np\n\n"
            f"def {fn}(W: np.ndarray, keys: np.ndarray, targets: np.ndarray, cov_prior: np.ndarray, lam: float = 0.05) -> np.ndarray:\n"
            "    W64 = W.astype(np.float64, copy=True)\n"
            "    K64 = keys.astype(np.float64)\n"
            "    V64 = targets.astype(np.float64)\n"
            "    C64 = cov_prior.astype(np.float64)\n"
            "    d_in = W64.shape[1]\n"
            "    R = V64 - (W64 @ K64)\n"
            "    rhs = R @ K64.T\n"
            "    M = C64 + (K64 @ K64.T) + lam * np.eye(d_in, dtype=np.float64)\n"
            "    delta = np.linalg.solve(M.T, rhs.T).T\n"
            "    return W64 + delta"
        )
        test = (
            "import numpy as np\n"
            "rng = np.random.default_rng(11)\n"
            "W = rng.standard_normal((5, 4))\n"
            "K = np.eye(4)[:, :2]\n"
            "V = np.ones((5, 2)) * 3.0\n"
            f"Wn = {fn}(W, K, V, np.zeros((4, 4)), lam=1e-6)\n"
            "np.testing.assert_allclose(Wn @ K, V, atol=1e-4)\n"
        )
        return {
            "id": f"v11_wmod_{idx:04d}",
            "pillar": "rsi_and_weight_surgery",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
        }

    if variant == 4:
        # Row-Nullspace Anchor-Preserving Weight Step (`(W_new - W) @ X_ref.T == 0`)
        fn = rng.choice(["anchor_orthogonal_weight_step", "nullspace_constrained_update", "preserve_activation_row_update"])
        if idx >= 30:
            fn = f"{fn}_{idx % 13}"
        prompt = (
            f"Implement `{fn}(W: np.ndarray, X_ref: np.ndarray, delta_raw: np.ndarray, tol: float = 1e-7) -> np.ndarray` in NumPy without mutating `W`:\n"
            "- `W`: `(d_out, d_in)`, `X_ref`: `(n_samples, d_in)` reference input activations (possibly rank-deficient with `n_samples < d_in`), `delta_raw`: `(d_out, d_in)`.\n"
            "- Project each row of `delta_raw` into the right nullspace of `X_ref` (retaining singular directions of `X_ref` with `sigma > tol`) and return `W + delta_proj` in `float64` so `(W_new @ X_ref.T)` equals `(W @ X_ref.T)` up to floating-point precision."
        )
        think = (
            "Let's derive the exact right-nullspace projector for `X_ref` (`(n_samples, d_in)`):\n"
            "1. Thin SVD `_, S, Vt = np.linalg.svd(X_ref.astype(np.float64), full_matrices=False)` gives orthonormal rows `V_act = Vt[S > tol]` of shape `(r, d_in)` spanning `rowspace(X_ref)`.\n"
            "2. The orthogonal projector onto `nullspace(X_ref)` is `P_perp = I_{d_in} - V_act.T @ V_act` (`(d_in, d_in)`).\n"
            "3. Projecting rows of `delta_raw`: `delta_proj = delta_raw @ P_perp = delta_raw - (delta_raw @ V_act.T) @ V_act` (`(d_out, d_in)`).\n"
            "4. Verify invariant: `delta_proj @ V_act.T = delta_raw @ V_act.T - (delta_raw @ V_act.T) @ (V_act @ V_act.T) = 0` since `V_act @ V_act.T = I_r`! Therefore `(W + delta_proj) @ X_ref.T == W @ X_ref.T` identically!"
        )
        code = (
            "import numpy as np\n\n"
            f"def {fn}(W: np.ndarray, X_ref: np.ndarray, delta_raw: np.ndarray, tol: float = 1e-7) -> np.ndarray:\n"
            "    W64 = W.astype(np.float64, copy=True)\n"
            "    D64 = delta_raw.astype(np.float64)\n"
            "    if X_ref.size == 0:\n"
            "        return W64 + D64\n"
            "    _, S, Vt = np.linalg.svd(X_ref.astype(np.float64), full_matrices=False)\n"
            "    V_act = Vt[S > tol]\n"
            "    if V_act.shape[0] > 0:\n"
            "        D64 = D64 - (D64 @ V_act.T) @ V_act\n"
            "    return W64 + D64"
        )
        test = (
            "import numpy as np\n"
            "rng = np.random.default_rng(19)\n"
            "W = rng.standard_normal((6, 5))\n"
            "X = rng.standard_normal((2, 5))\n"
            "D = rng.standard_normal((6, 5))\n"
            f"Wn = {fn}(W, X, D)\n"
            "np.testing.assert_allclose(Wn @ X.T, W @ X.T, atol=1e-9)\n"
            "assert np.linalg.norm(Wn - W) > 0.1\n"
        )
        return {
            "id": f"v11_wmod_{idx:04d}",
            "pillar": "rsi_and_weight_surgery",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
        }

    if variant == 5:
        # DoRA Magnitude-Direction Decoupled Forward / Weight Reconstruction
        fn = rng.choice(["dora_merge_effective_weight", "reconstruct_dora_linear", "compute_dora_merged_matrix"])
        if idx >= 30:
            fn = f"{fn}_{idx % 13}"
        prompt = (
            f"Implement `{fn}(W0: np.ndarray, lora_B: np.ndarray, lora_A: np.ndarray, mag: np.ndarray, scale: float = 1.0, eps: float = 1e-8) -> np.ndarray` in NumPy without mutating inputs:\n"
            "- `W0`: `(d_out, d_in)`, `lora_B`: `(d_out, r)`, `lora_A`: `(r, d_in)`, 1D row magnitude `mag`: `(d_out,)`.\n"
            "- Form directional matrix `V = W0 + scale * (lora_B @ lora_A)`, compute L2 norm of each row `row_norm = sqrt(sum_j V_{i,j}^2) + eps`, and return `(mag[:, None] / row_norm[:, None]) * V` in `float64`."
        )
        think = (
            "Let's trace DoRA's row-wise normalization carefully:\n"
            "- In linear layer `y = x @ W.T`, each output channel `i` corresponds to **row `i`** of `W` (`axis=1`, `keepdims=True` -> shape `(d_out, 1)`).\n"
            "- `V = W0 + scale * (lora_B @ lora_A)` has shape `(d_out, d_in)`.\n"
            "- `row_norm = np.linalg.norm(V, axis=1, keepdims=True) + eps` has shape `(d_out, 1)`.\n"
            "- Multiplying `(mag[:, None] / row_norm) * V` gives exact row norms `mag[i]` (up to `eps`)."
        )
        code = (
            "import numpy as np\n\n"
            f"def {fn}(W0: np.ndarray, lora_B: np.ndarray, lora_A: np.ndarray, mag: np.ndarray, scale: float = 1.0, eps: float = 1e-8) -> np.ndarray:\n"
            "    V = W0.astype(np.float64) + float(scale) * (lora_B.astype(np.float64) @ lora_A.astype(np.float64))\n"
            "    row_norm = np.linalg.norm(V, axis=1, keepdims=True) + eps\n"
            "    return (mag.astype(np.float64)[:, None] / row_norm) * V"
        )
        test = (
            "import numpy as np\n"
            "W0 = np.array([[3.0, 4.0, 0.0], [0.0, 6.0, 8.0]])\n"
            "B = np.zeros((2, 2))\nA = np.zeros((2, 3))\n"
            "m = np.array([10.0, 5.0])\n"
            f"Wm = {fn}(W0, B, A, m)\n"
            "np.testing.assert_allclose(np.linalg.norm(Wm, axis=1), m, atol=1e-6)\n"
        )
        return {
            "id": f"v11_wmod_{idx:04d}",
            "pillar": "rsi_and_weight_surgery",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
        }

    if variant == 6:
        # TIES-Merging (Trim Top-K% Magnitude -> Elect Majority Sign -> Disjoint Mean)
        fn = rng.choice(["ties_merge_task_vectors", "merge_deltas_ties", "ties_disjoint_consensus"])
        if idx >= 30:
            fn = f"{fn}_{idx % 13}"
        prompt = (
            f"Implement `{fn}(deltas: list[np.ndarray], keep_ratio: float = 0.5) -> np.ndarray` in NumPy:\n"
            "- Given `K >= 1` task delta arrays of identical shape:\n"
            "  1. **Trim**: in each flattened delta of size `N`, keep the `top_k = max(1, int(round(keep_ratio * N)))` largest-magnitude entries (breaking ties by keeping entries `>= threshold` or via `argsort` of `-abs(d)`), zeroing the rest.\n"
            "  2. **Elect Sign**: at each coordinate, compute `elected_sign = np.sign(sum_t trimmed_t)`.\n"
            "  3. **Disjoint Merge**: at each coordinate, average ONLY the trimmed entries whose `np.sign(val) == elected_sign` and `elected_sign != 0` (0.0 where count is 0)."
        )
        think = (
            "Let's trace all 3 steps of TIES-Merging carefully:\n"
            "1. Flatten each delta `d` (`N = d.size`), pick `k = max(1, min(N, int(round(keep_ratio * N))))` indices via `idx = np.argsort(-np.abs(flat), kind='stable')[:k]`, copy into a zero array, and reshape back.\n"
            "2. Stack `trimmed` into `(K, *shape)`. Sum across `axis=0` -> `elected = np.sign(np.sum(trimmed, axis=0))`.\n"
            "3. Match mask: `mask = (np.sign(trimmed) == elected[None, ...]) & (elected[None, ...] != 0)`.\n"
            "4. Count `cnt = np.sum(mask, axis=0)`; return `np.where(cnt > 0, np.sum(trimmed * mask, axis=0) / np.maximum(cnt, 1), 0.0)`."
        )
        code = (
            "import numpy as np\n\n"
            f"def {fn}(deltas: list[np.ndarray], keep_ratio: float = 0.5) -> np.ndarray:\n"
            "    shape = deltas[0].shape\n"
            "    n = deltas[0].size\n"
            "    k = max(1, min(n, int(round(keep_ratio * n))))\n"
            "    trimmed_list = []\n"
            "    for d in deltas:\n"
            "        flat = d.astype(np.float64).ravel()\n"
            "        out = np.zeros_like(flat)\n"
            "        top_idx = np.argsort(-np.abs(flat), kind='stable')[:k]\n"
            "        out[top_idx] = flat[top_idx]\n"
            "        trimmed_list.append(out.reshape(shape))\n"
            "    stacked = np.stack(trimmed_list, axis=0)\n"
            "    elected = np.sign(np.sum(stacked, axis=0))\n"
            "    mask = (np.sign(stacked) == elected[None, ...]) & (elected[None, ...] != 0)\n"
            "    cnt = np.sum(mask, axis=0)\n"
            "    summed = np.sum(stacked * mask, axis=0)\n"
            "    return np.where(cnt > 0, summed / np.maximum(cnt, 1), 0.0)"
        )
        test = (
            "import numpy as np\n"
            "d1 = np.array([4.0, -0.1, -6.0, 0.2])\n"
            "d2 = np.array([2.0, 0.05, 1.0, -5.0])\n"
            f"m = {fn}([d1, d2], keep_ratio=0.5)\n"
            "np.testing.assert_allclose(m, np.array([3.0, 0.0, -6.0, -5.0]), atol=1e-7)\n"
        )
        return {
            "id": f"v11_wmod_{idx:04d}",
            "pillar": "rsi_and_weight_surgery",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
        }

    if variant == 7:
        # Exact IEEE Floating-Point ULP Spacing (`bf16` vs `fp16` vs `fp32`) + Footprint Math
        exp_pow = rng.choice([2, 3, 4, 5, 6])
        val_x = float(2**exp_pow)
        ulp_fp16 = val_x * (2.0**-10)
        ulp_bf16 = val_x * (2.0**-7)
        ratio = int(round(ulp_bf16 / ulp_fp16))  # always 2^3 = 8!
        next_bf16 = val_x + ulp_bf16
        prompt = (
            f"Compare IEEE `float16` (`1` sign, `5` exponent, `10` explicit mantissa bits) and `bfloat16` (`1` sign, `8` exponent, `7` explicit mantissa bits) on the interval `[{val_x}, {2.0 * val_x})`:\n"
            f"1. Compute exact `ulp_fp16` at `{val_x}`.\n"
            f"2. Compute exact `ulp_bf16` at `{val_x}`.\n"
            "3. Compute the exact integer ratio `bf16_over_fp16 = int(ulp_bf16 / ulp_fp16)` and `next_bf16` (the smallest representable `bfloat16` strictly greater than "
            f"`{val_x}`).\n"
            'Return ONLY JSON `{"ulp_fp16": <float>, "ulp_bf16": <float>, "bf16_over_fp16": <int>, "next_bf16": <float>}`.'
        )
        think = (
            f"Let's carefully apply the floating-point ULP formula `ULP(x) = 2^(floor(log2(x)) - p)` where `p` is the number of explicit fraction bits (NEVER multiply by `2^+p`!):\n"
            f"- Here `x = {val_x} = 2^{exp_pow}`, so `floor(log2(x)) = {exp_pow}`.\n"
            f"1. For `float16` (`p = 10` fraction bits): `ulp_fp16 = 2^({exp_pow} - 10) = {val_x} / 1024 = {ulp_fp16}`.\n"
            f"2. For `bfloat16` (`p = 7` fraction bits): `ulp_bf16 = 2^({exp_pow} - 7) = {val_x} / 128 = {ulp_bf16}`.\n"
            f"3. Sanity check: `bfloat16` has 3 fewer mantissa bits than `float16`, so its ULP spacing is `2^3 = 8` times coarser (`{ulp_bf16} / {ulp_fp16} = {ratio}`).\n"
            f"4. Next representable `bfloat16` above `{val_x}` is `{val_x} + {ulp_bf16} = {next_bf16}`."
        )
        expected = {"ulp_fp16": ulp_fp16, "ulp_bf16": ulp_bf16, "bf16_over_fp16": ratio, "next_bf16": next_bf16}
        return {
            "id": f"v11_wmod_{idx:04d}",
            "pillar": "rsi_and_weight_surgery",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n{json.dumps(expected)}"}],
            "verifier": {"expected": expected, "forbid_tool_call": True},
        }

    if variant == 8:
        # Sparse Conditional Lookup Table Total Params vs Per-Token Active Params
        n_rows = rng.choice([16384, 24576, 49152])
        d_mod = rng.choice([1536, 2048, 2560])
        r_bot = rng.choice([192, 256, 384])
        tbl_p = n_rows * d_mod
        gate_p = 2 * r_bot * d_mod
        tot_p = tbl_p + gate_p
        act_p = d_mod + gate_p  # 1 row gather `d_mod` + 2 dense gate projections!
        prompt = (
            f"A conditional n-gram memory module has a `{n_rows} × {d_mod}` embedding table (fetching exactly `1` row of `{d_mod}` floats per token) and a dense bottleneck gate consisting of two bias-free linear projections (`{d_mod} -> {r_bot}` and `{r_bot} -> {d_mod}`, ignoring 1D norms).\n"
            "Compute:\n"
            "1. `table_params`: total parameters in the lookup table.\n"
            "2. `gate_params`: total parameters in the two gate matrices.\n"
            "3. `total_module_params`: `table_params + gate_params`.\n"
            "4. `active_params_per_token`: exact parameters read/multiplied for a single token (`1` fetched row + `gate_params`).\n"
            'Return ONLY JSON `{"table_params": <int>, "gate_params": <int>, "total_module_params": <int>, "active_params_per_token": <int>}`.'
        )
        think = (
            f"Let's multiply digit-by-digit carefully without dropping digits:\n"
            f"1. `table_params = {n_rows} * {d_mod} = {tbl_p}`.\n"
            f"2. `gate_params = 2 * {r_bot} * {d_mod} = {2 * r_bot} * {d_mod} = {gate_p}`.\n"
            f"3. `total_module_params = {tbl_p} + {gate_p} = {tot_p}`.\n"
            f"4. Per token, only `1` row (`{d_mod}` weights) is gathered from the table while both gate projections run densely: `active_params_per_token = {d_mod} + {gate_p} = {act_p}`."
        )
        expected = {
            "table_params": tbl_p,
            "gate_params": gate_p,
            "total_module_params": tot_p,
            "active_params_per_token": act_p,
        }
        return {
            "id": f"v11_wmod_{idx:04d}",
            "pillar": "rsi_and_weight_surgery",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n{json.dumps(expected)}"}],
            "verifier": {"expected": expected, "forbid_tool_call": True},
        }

    # variant == 9: Darwin Gödel Machine (`DGM`) UCB Lineage Selection + `Dr. GRPO` Advantage Math
    c_exp = 0.12
    s0, ch0 = round(0.68 + 0.01 * (idx % 5), 2), 7
    s1, ch1 = round(0.76 + 0.01 * (idx % 4), 2), 3
    s2_reg = round(0.89, 2)  # regressed anchor NLL!
    s3, ch3 = round(0.75 + 0.01 * (idx % 4), 2), 0
    u0 = round(s0 + c_exp / math.sqrt(1 + ch0), 4)
    u1 = round(s1 + c_exp / math.sqrt(1 + ch1), 4)
    u3 = round(s3 + c_exp / math.sqrt(1 + ch3), 4)
    cands = [("node_0", u0), ("node_1", u1), ("node_3", u3)]
    best_id, best_u = max(cands, key=lambda kv: kv[1])
    prompt = (
        f"In a `Darwin Gödel Machine` (`DGM`) self-improvement archive (`c = {c_exp}`), each lineage node has `(benchmark_score, children_spawned, anchor_nll_delta)`:\n"
        f"- `node_0`: `score = {s0}`, `children = {ch0}`, `anchor_nll_delta = 0.00`\n"
        f"- `node_1`: `score = {s1}`, `children = {ch1}`, `anchor_nll_delta = -0.01`\n"
        f"- `node_2`: `score = {s2_reg}`, `children = 0`, `anchor_nll_delta = +0.05` (violates `anchor_nll_delta <= 0.0`!)\n"
        f"- `node_3`: `score = {s3}`, `children = {ch3}`, `anchor_nll_delta = -0.02`\n"
        f"Filter to eligible non-regressed nodes (`anchor_nll_delta <= 0.0`) and select the parent maximizing `ucb = round(score + {c_exp} / sqrt(1 + children), 4)`.\n"
        'Return ONLY JSON `{"eligible_nodes": <int>, "chosen_parent": "<str>", "best_ucb": <float>}`.'
    )
    think = (
        f"Let's check denominator `sqrt(1 + children)` carefully (NOT `1 / children`, which would divide by zero at `children = 0`!):\n"
        f"1. Filter `anchor_nll_delta <= 0.0`: `node_0` (`0.00`), `node_1` (`-0.01`), and `node_3` (`-0.02`) pass (`eligible_nodes = 3`); `node_2` (`+0.05`) is excluded.\n"
        f"2. Compute exact `sqrt(1 + children)`:\n"
        f"   - `node_0`: `{s0} + {c_exp} / sqrt({1 + ch0}) = {u0}`.\n"
        f"   - `node_1`: `{s1} + {c_exp} / sqrt(4) = {s1} + {c_exp / 2:.4f} = {u1}`.\n"
        f"   - `node_3`: `{s3} + {c_exp} / sqrt(1) = {s3} + {c_exp} = {u3}`.\n"
        f"3. Comparing `{u0}`, `{u1}`, `{u3}` -> winner is `{best_id}` with `best_ucb = {best_u}`."
    )
    expected = {"eligible_nodes": 3, "chosen_parent": best_id, "best_ucb": best_u}
    return {
        "id": f"v11_wmod_{idx:04d}",
        "pillar": "rsi_and_weight_surgery",
        "images": [],
        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n{json.dumps(expected)}"}],
        "verifier": {"expected": expected, "forbid_tool_call": True},
    }


# ===========================================================================
# 2. Research Methodology, Citations, Tool Path Fidelity & OOD Discovery
# ===========================================================================


def make_v11_research_discovery_item(rng: random.Random, idx: int) -> dict[str, Any]:
    variant = idx % 8

    if variant == 0:
        # Unbiased `pass@k` vs Naive Plugin `1 - (1 - c/n)^k` (explicitly squaring/cubing `(1 - c/n)**k`!)
        n_tot = rng.choice([5, 6, 8, 10])
        c_ok = rng.randint(1, max(1, n_tot - 3))
        k_pick = rng.choice([2, 3])
        unb = round(1.0 - math.comb(n_tot - c_ok, k_pick) / math.comb(n_tot, k_pick), 4)
        fail_rate = (n_tot - c_ok) / n_tot
        plug = round(1.0 - (fail_rate**k_pick), 4)
        prompt = (
            f"An evaluation generates `n = {n_tot}` rollouts for an unseen problem and `c = {c_ok}` rollouts pass unit tests.\n"
            f"1. Compute Chen et al.'s unbiased estimator `unbiased_pass_at_k = round(1 - comb(n - c, k) / comb(n, k), 4)` for `k = {k_pick}`.\n"
            f"2. Compute the naive with-replacement plugin estimate `plugin_pass_at_k = round(1 - (1 - c / n)**k, 4)` for `k = {k_pick}`.\n"
            'Return ONLY JSON `{"unbiased_pass_at_k": <float>, "plugin_pass_at_k": <float>}`.'
        )
        think = (
            f"Let's compute both estimators step-by-step and make sure we raise `(1 - c/n)` to power `k = {k_pick}`!\n"
            f"1. Unbiased estimator (`n = {n_tot}, c = {c_ok}, n - c = {n_tot - c_ok}, k = {k_pick}`):\n"
            f"   - `comb({n_tot - c_ok}, {k_pick}) = {math.comb(n_tot - c_ok, k_pick)}`\n"
            f"   - `comb({n_tot}, {k_pick}) = {math.comb(n_tot, k_pick)}`\n"
            f"   - `1 - {math.comb(n_tot - c_ok, k_pick)} / {math.comb(n_tot, k_pick)} = {unb}`.\n"
            f"2. Plugin estimator:\n"
            f"   - Single-sample failure fraction `q = 1 - {c_ok}/{n_tot} = {n_tot - c_ok}/{n_tot} = {fail_rate:.6f}`.\n"
            f"   - Raise `q` to power `k = {k_pick}`: `q**{k_pick} = ({n_tot - c_ok}**{k_pick}) / ({n_tot}**{k_pick}) = {(n_tot - c_ok)**k_pick} / {n_tot**k_pick} = {fail_rate**k_pick:.6f}`.\n"
            f"   - `plugin_pass_at_k = round(1 - {fail_rate**k_pick:.6f}, 4) = {plug}`."
        )
        expected = {"unbiased_pass_at_k": unb, "plugin_pass_at_k": plug}
        return {
            "id": f"v11_res_{idx:04d}",
            "pillar": "research_and_ablation",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n{json.dumps(expected)}"}],
            "verifier": {"expected": expected, "forbid_tool_call": True},
        }

    if variant == 1:
        # 2x2 Factorial Main Effects, Additive Prediction, & Superadditive Interaction Synergy
        base_v = round(rng.uniform(52.0, 68.0), 1)
        da = round(rng.choice([3.5, 4.5, 5.5, 6.0]), 1)
        db = round(rng.choice([2.5, 3.0, 4.0, 5.0]), 1)
        syn = round(rng.choice([-3.5, -2.0, 2.5, 4.0]), 1)
        only_a = round(base_v + da, 1)
        only_b = round(base_v + db, 1)
        add_pred = round(base_v + da + db, 1)
        both_ab = round(add_pred + syn, 1)
        is_super = syn > 0.0
        prompt = (
            f"A `2 × 2` ablation measures accuracy across four runs:\n"
            f"- Baseline (`neither A nor B`): `{base_v}`\n"
            f"- `+Module_A` only: `{only_a}`\n"
            f"- `+Module_B` only: `{only_b}`\n"
            f"- `+Module_A + Module_B` combined: `{both_ab}`\n"
            "Compute `effect_A = only_A - base`, `additive_expected = base + effect_A + (only_B - base)`, `synergy = combined - additive_expected`, and boolean `is_superadditive` (`synergy > 0`).\n"
            'Return ONLY JSON `{"effect_A": <float>, "additive_expected": <float>, "synergy": <float>, "is_superadditive": <bool>}`.'
        )
        think = (
            f"Let's add the decimals carefully:\n"
            f"- `effect_A = {only_a} - {base_v} = {da}`.\n"
            f"- `effect_B = {only_b} - {base_v} = {db}`.\n"
            f"- `additive_expected = {base_v} + {da} + {db} = {round(base_v + da, 1)} + {db} = {add_pred}` (double-check: `{add_pred} - {base_v} = {round(da + db, 1)}`).\n"
            f"- `synergy = {both_ab} - {add_pred} = {syn}` -> `is_superadditive = {str(is_super).lower()}`."
        )
        expected = {"effect_A": da, "additive_expected": add_pred, "synergy": syn, "is_superadditive": is_super}
        return {
            "id": f"v11_res_{idx:04d}",
            "pillar": "research_and_ablation",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n{json.dumps(expected)}"}],
            "verifier": {"expected": expected, "forbid_tool_call": True},
        }

    if variant == 2:
        # Multi-Floor Candidate Gate Filtering (`>=` on 3 skill cols, `<=` on perplexity)
        f_math, f_code, f_tool, cap_ppl = 76, 79, 83, 1.62
        # Build 4 runs where exactly 2 pass all floors and 1 wins by mean
        r1 = (84, 86, 88, 1.54)  # PASS, mean = 86.0
        r2 = (91, 77, 90, 1.51)  # FAIL code 77 < 79
        r3 = (88, 89, 87, 1.68)  # FAIL ppl 1.68 > 1.62
        r4 = (80, 82, 84, 1.59)  # PASS, mean = 82.0
        prompt = (
            f"Select the winning checkpoint from `CKPT-A..CKPT-D` subject to mandatory floors `math >= {f_math}`, `code >= {f_code}`, `tools >= {f_tool}`, and `val_ppl <= {cap_ppl}`:\n"
            f"- `CKPT-A`: math={r1[0]}, code={r1[1]}, tools={r1[2]}, val_ppl={r1[3]}\n"
            f"- `CKPT-B`: math={r2[0]}, code={r2[1]}, tools={r2[2]}, val_ppl={r2[3]}\n"
            f"- `CKPT-C`: math={r3[0]}, code={r3[1]}, tools={r3[2]}, val_ppl={r3[3]}\n"
            f"- `CKPT-D`: math={r4[0]}, code={r4[1]}, tools={r4[2]}, val_ppl={r4[3]}\n"
            "Among checkpoints passing ALL 4 gates, pick the highest unweighted mean `(math + code + tools) / 3` rounded to 2 decimals.\n"
            'Return ONLY JSON `{"passing_count": <int>, "winner": "<str>", "winner_mean": <float>}`.'
        )
        think = (
            f"Check every single constraint on each checkpoint before averaging:\n"
            f"- `CKPT-A`: `84>={f_math}` (OK), `86>={f_code}` (OK), `88>={f_tool}` (OK), `1.54<={cap_ppl}` (OK) -> **PASS**, mean `(84+86+88)/3 = 258/3 = 86.0`.\n"
            f"- `CKPT-B`: `code=77 < {f_code}` -> **FAIL**.\n"
            f"- `CKPT-C`: `val_ppl=1.68 > {cap_ppl}` -> **FAIL**.\n"
            f"- `CKPT-D`: `80>={f_math}` (OK), `82>={f_code}` (OK), `84>={f_tool}` (OK), `1.59<={cap_ppl}` (OK) -> **PASS**, mean `(80+82+84)/3 = 246/3 = 82.0`.\n"
            "2 checkpoints pass (`CKPT-A`, `CKPT-D`), and `CKPT-A` wins with `86.0`."
        )
        expected = {"passing_count": 2, "winner": "CKPT-A", "winner_mean": 86.0}
        return {
            "id": f"v11_res_{idx:04d}",
            "pillar": "research_and_ablation",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n{json.dumps(expected)}"}],
            "verifier": {"expected": expected, "forbid_tool_call": True},
        }

    if variant == 3:
        # Strict Exact Tool Name & Exact User-Provided Relative File Path Discipline!
        tool_nm = rng.choice(["read_file", "inspect_source", "open_module", "load_workspace_file"])
        rel_dir = rng.choice(["core_engine", "opt_kernels", "symbolic_verifiers", "router_pkg", "kv_paging"])
        rel_fn = rng.choice([" dispatch_gate.py", "nullspace_svd.py", "rope_scaling.py", "archive_ucb.py"]).strip()
        rel_path = f"{rel_dir}/{rel_fn}"
        prompt = (
            f"Workspace tool schema: `{tool_nm}(path: str)`.\n\n"
            f"You have zero prior knowledge of `{rel_path}`. Read `{rel_path}` first to inspect its exact implementation before answering."
        )
        think = (
            f"I have no prior knowledge of `{rel_path}`, so I must call the exact tool `{tool_nm}` (never pluralize or rename it!) "
            f"with the exact relative path `{rel_path}` (never prepend `/cline/...` or `/workspace/...` prefixes!)."
        )
        call = _xml_call(tool_nm, {"path": rel_path})
        return {
            "id": f"v11_res_{idx:04d}",
            "pillar": "multistep_tool_use",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n{call}"}],
            "verifier": {"tool": tool_nm, "args": {"path": rel_path}},
        }

    if variant == 4:
        # Citation-Grounded Multi-Source Arithmetic + Erratum Superseding Preview (Solar Desalination Domain)
        strings = rng.choice([20, 30, 40])
        mw_per = rng.choice([1.5, 2.0])
        inv_eff = rng.choice([0.92, 0.95])
        days = 300
        tariff = 45.0
        net_mw = round(strings * mw_per * inv_eff, 2)
        annual_rev = round(net_mw * days * tariff, 2)
        prompt = (
            "Compute verified `net_ac_mw` and `seasonal_credit_eur` for `Helios-Basin` using ONLY authoritative sources, citing `[S1]`, `[S2]`, `[S3]` inline (excluding revoked `[S4]`) and ending with JSON `{\"net_ac_mw\": <float>, \"seasonal_credit_eur\": <float>}`:\n\n"
            f"[S1] (Array Commissioning Manifest): `{strings}` photovoltaic strings generating `{mw_per} MW` DC per string (`{strings * mw_per} MW` raw DC).\n"
            f"[S2] (2026 Grid-Tie Calibration Erratum): Supersedes `[S4]`; certified inverter conversion factor is `{inv_eff}` (`net_ac_mw = raw_dc_mw * factor`).\n"
            f"[S3] (Regional Dispatch Tariff): `{days}` sunny dispatch days at `€{tariff}/MW-day` (`seasonal_credit_eur = net_ac_mw * {days} * {tariff}`).\n"
            "[S4] (Revoked Brochure): Unverified brochure inverter factor `0.99` (revoked)."
        )
        think = (
            f"Let's verify provenance and decimal places carefully:\n"
            f"- Exclude revoked `[S4]` (`0.99`); cite `[S1]`, `[S2]`, `[S3]`.\n"
            f"- Raw DC output from `[S1]`: `{strings} * {mw_per} = {strings * mw_per} MW`.\n"
            f"- Net AC output from `[S2]`: `{strings * mw_per} * {inv_eff} = {net_mw} MW`.\n"
            f"- Seasonal credit from `[S3]`: `{net_mw} * {days} = {net_mw * days}`; `{net_mw * days} * {tariff} = {annual_rev}` EUR."
        )
        ans = (
            f"`Helios-Basin` produces `{strings * mw_per} MW` raw DC across `{strings}` photovoltaic strings [S1], yielding `{net_mw} MW` net AC under the certified `{inv_eff}` grid-tie erratum [S2]. "
            f"Across `{days}` dispatch days at `€{tariff}/MW-day`, seasonal credit totals `€{annual_rev}` [S3].\n"
            f'{json.dumps({"net_ac_mw": net_mw, "seasonal_credit_eur": annual_rev})}'
        )
        return {
            "id": f"v11_res_{idx:04d}",
            "pillar": "realtime_research_citations",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n{ans}"}],
            "verifier": {
                "required_citations": ["[S1]", "[S2]", "[S3]"],
                "forbidden_citations": ["[S4]"],
                "valid_citations": ["[S1]", "[S2]", "[S3]", "[S4]"],
                "expected": {"net_ac_mw": net_mw, "seasonal_credit_eur": annual_rev},
                "forbid_tool_call": True,
            },
        }

    if variant == 5:
        # Citation-Grounded Epistemic Abstention (MUST cite `[S1]` & `[S2]` inline to explain what they DO cover!)
        dev_nm = f"Chronos-NIC-{idx % 9 + 1}"
        prompt = (
            f"Using ONLY `[S1]` and `[S2]`, report the `PCIe Gen6` idle power in milliwatts for `{dev_nm}`. "
            "If missing from the sources, do NOT guess: cite `[S1]` and `[S2]` inline to explain what they cover and end with JSON `{\"status\": \"insufficient_evidence\", \"idle_mw\": null}`:\n\n"
            f"[S1] ({dev_nm} Brief): Lists `400 GbE` RDMA throughput and `16` hardware queues.\n"
            f"[S2] ({dev_nm} Firmware Release Note): Documents RoCEv2 congestion ECN marking; power telemetry deferred to Rev B."
        )
        think = (
            f"Check `[S1]` and `[S2]`: `[S1]` only covers `400 GbE` throughput and queue count, while `[S2]` covers RoCEv2 ECN and explicitly defers power specs. "
            "Important: the prompt instructs me to cite `[S1]` and `[S2]` inline in my explanation BEFORE outputting the abstention JSON!"
        )
        ans = (
            f"Neither retrieved source reports `PCIe Gen6` idle power in milliwatts: `[S1]` only specifies `400 GbE` RDMA throughput and queue count [S1], "
            f"while `[S2]` covers RoCEv2 ECN marking and notes that power telemetry is deferred to Rev B [S2].\n"
            '{"status": "insufficient_evidence", "idle_mw": null}'
        )
        return {
            "id": f"v11_res_{idx:04d}",
            "pillar": "realtime_research_citations",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n{ans}"}],
            "verifier": {
                "required_citations": ["[S1]", "[S2]"],
                "valid_citations": ["[S1]", "[S2]"],
                "expected": {"status": "insufficient_evidence", "idle_mw": None},
                "forbid_tool_call": True,
            },
        }

    if variant == 6:
        # Claim-Level Entailment + Source-ID Verification (`c_ok = True` ONLY if BOTH fact AND `[S_i]` tag are valid!)
        prompt = (
            "Verify each claim in the draft strictly against `[S1]..[S3]` (a claim is `true` ONLY if its factual value matches the cited source AND the `[S#]` tag exists and supports it):\n"
            "- `[S1]`: `boreas-kv` uses `LSM-tree` compaction with `64 MB` memtables.\n"
            "- `[S2]`: `boreas-kv` achieves `99.95%` availability across `3` AZs.\n"
            "- `[S3]`: `boreas-kv` encrypts WAL segments with `AES-256-GCM`.\n\n"
            "Draft:\n"
            "- Claim 1: \"boreas-kv uses LSM-tree compaction with 64 MB memtables [S1].\"\n"
            "- Claim 2: \"boreas-kv achieves 99.999% availability across 3 AZs [S2].\"\n"
            "- Claim 3: \"boreas-kv encrypts WAL segments with AES-256-GCM [S5].\"\n\n"
            'Return ONLY JSON `{"claim1_ok": <bool>, "claim2_ok": <bool>, "claim3_ok": <bool>, "verified_total": <int>}`.'
        )
        think = (
            "Audit both factual accuracy AND citation tag validity on every claim:\n"
            "- Claim 1: matches `[S1]` (`LSM-tree`, `64 MB`) and cites `[S1]` -> `claim1_ok = true`.\n"
            "- Claim 2: claims `99.999%` whereas `[S2]` says `99.95%` -> `claim2_ok = false`.\n"
            "- Claim 3: text matches `[S3]`, BUT it cites `[S5]` which does NOT exist (`valid = S1..S3`)! A hallucinated citation tag invalidates the claim -> `claim3_ok = false`.\n"
            "- `verified_total = 1`."
        )
        expected = {"claim1_ok": True, "claim2_ok": False, "claim3_ok": False, "verified_total": 1}
        return {
            "id": f"v11_res_{idx:04d}",
            "pillar": "realtime_research_citations",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n{json.dumps(expected)}"}],
            "verifier": {"expected": expected, "forbid_tool_call": True},
        }

    # variant == 7: Deduplicated Sorted Citation Extractor in Python (`sorted({t for t in tags ...})`)
    fn = rng.choice(["summarize_doc_citations", "check_inline_source_tags", "parse_verified_brackets"])
    if idx >= 24:
        fn = f"{fn}_{idx % 11}"
    prompt = (
        f"Implement `{fn}(text: str, allowed_tags: set[str]) -> dict[str, object]` in Python (`re`):\n"
        "- Find all occurrences matching `r\"\\[(DOC-\\d{4})\\]\"` (e.g. `\"[DOC-0012]\"` -> `\"DOC-0012\"`).\n"
        "- Return `{'occurrences': <int>, 'unique_allowed': <sorted unique list>, 'unique_invalid': <sorted unique list>, 'all_grounded': <bool>}` where `all_grounded` is `True` iff `occurrences > 0` and `len(unique_invalid) == 0`."
    )
    think = (
        "Watch out for duplicate citations in `text` (e.g. `\"[DOC-0002] ... [DOC-0002]\"`):\n"
        "- `occurrences` counts ALL regex matches (`len(found)`).\n"
        "- `unique_allowed` and `unique_invalid` MUST deduplicate via `set` comprehension `{t for t in found ...}` before `sorted(...)`, NOT `[t for t in found ...]`!"
    )
    code = (
        "import re\n\n"
        f"def {fn}(text: str, allowed_tags: set[str]) -> dict[str, object]:\n"
        "    found = re.findall(r'\\[(DOC-\\d{4})\\]', text)\n"
        "    uniq_ok = sorted({t for t in found if t in allowed_tags})\n"
        "    uniq_bad = sorted({t for t in found if t not in allowed_tags})\n"
        "    return {\n"
        "        'occurrences': len(found),\n"
        "        'unique_allowed': uniq_ok,\n"
        "        'unique_invalid': uniq_bad,\n"
        "        'all_grounded': len(found) > 0 and len(uniq_bad) == 0,\n"
        "    }"
    )
    test = (
        f"r = {fn}('Fact A [DOC-0002] and Fact B [DOC-0001][DOC-0002].', {{'DOC-0001', 'DOC-0002'}})\n"
        "assert r == {'occurrences': 3, 'unique_allowed': ['DOC-0001', 'DOC-0002'], 'unique_invalid': [], 'all_grounded': True}\n"
    )
    return {
        "id": f"v11_res_{idx:04d}",
        "pillar": "realtime_research_citations",
        "images": [],
        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n```python\n{code}\n```"}],
        "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
    }


# ===========================================================================
# 3. Algorithmic Invariants, Posets, Binary Search, Cycle DFS & Log-Domain Code
# ===========================================================================


def make_v11_algo_invariant_item(rng: random.Random, idx: int) -> dict[str, Any]:
    variant = idx % 8

    if variant == 0:
        # Lower-Bound (`first_index >= target`) Binary Search with 2-element `[10, 20]` termination proof!
        fn = rng.choice(["lower_bound_idx", "find_first_at_least", "bisect_first_ge"])
        if idx >= 24:
            fn = f"{fn}_{idx % 11}"
        prompt = (
            f"Implement `{fn}(sorted_vals: list[int], threshold: int) -> int` in `O(log n)` Python:\n"
            "- Given ascending `sorted_vals`, return the smallest index `i` with `sorted_vals[i] >= threshold`, or `-1` if no element is `>= threshold`."
        )
        think = (
            "Let's distinguish **lower-bound (`first >=`)** from upper-bound (`last <=`) using a 2-element test `[10, 20]`:\n"
            "- Search range `[lo, hi)` with `lo = 0, hi = len(sorted_vals)` (half-open, so `lo == len(sorted_vals)` cleanly detects no element `>= threshold`!).\n"
            "- Standard floor midpoint `mid = (lo + hi) // 2`:\n"
            "  - If `sorted_vals[mid] >= threshold`: `mid` is a valid candidate, so shrink right boundary `hi = mid`.\n"
            "  - Else (`sorted_vals[mid] < threshold`): `mid` is strictly too small, so advance `lo = mid + 1`.\n"
            "- Test on `[10, 20]`:\n"
            "  - `threshold = 10`: `lo=0, hi=2 -> mid=1 (20>=10) -> hi=1 -> mid=0 (10>=10) -> hi=0` -> stops with `lo=0` (exact!).\n"
            "  - `threshold = 25`: `lo=0, hi=2 -> mid=1 (20<25) -> lo=2` -> stops with `lo == len == 2 -> return -1` (exact!).\n"
            "  - Empty `[]`: `lo=0, hi=0 -> return -1`."
        )
        code = (
            f"def {fn}(sorted_vals: list[int], threshold: int) -> int:\n"
            "    lo, hi = 0, len(sorted_vals)\n"
            "    while lo < hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        if sorted_vals[mid] >= threshold:\n"
            "            hi = mid\n"
            "        else:\n"
            "            lo = mid + 1\n"
            "    return lo if lo < len(sorted_vals) else -1"
        )
        test = (
            f"assert {fn}([], 5) == -1\n"
            f"assert {fn}([10, 20], 10) == 0\n"
            f"assert {fn}([10, 20], 15) == 1\n"
            f"assert {fn}([10, 20], 25) == -1\n"
            f"assert {fn}([2, 4, 4, 4, 9], 4) == 1\n"
        )
        return {
            "id": f"v11_algo_{idx:04d}",
            "pillar": "diagnostic_debugging",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
        }

    if variant == 1:
        # 3-Color Directed Cycle Detection (`0=unvisited, 1=visiting, 2=done` — Diamond DAG == False!)
        fn = rng.choice(["detect_digraph_cycle", "has_directed_cycle_3color", "check_task_graph_cycle"])
        if idx >= 24:
            fn = f"{fn}_{idx % 11}"
        prompt = (
            f"Implement `{fn}(adj: dict[str, list[str]]) -> bool` in Python:\n"
            "- Return `True` iff the directed graph `adj` contains at least one directed cycle (including self-loops and nodes appearing only as neighbors).\n"
            "- Must return `False` on diamond DAGs like `{'A': ['B', 'C'], 'B': ['D'], 'C': ['D']}`."
        )
        think = (
            "Let's write clean, bug-free 3-color DFS (`color: dict[str, int]`):\n"
            "- `0` (absent): unvisited.\n"
            "- `1`: currently on the active recursion call stack -> encountering `color.get(v, 0) == 1` proves a back-edge cycle (`return True`).\n"
            "- `2`: completely finished without cycles -> encountering `color.get(v, 0) == 2` on a shared sink `D` in a diamond DAG immediately skips `D` (`return False`).\n"
            "- Syntax check: assign `color[u] = 1` on entry and `color[u] = 2` on exit (NEVER assign to `color.get(...)`!). Iterate over `list(adj.keys())`."
        )
        code = (
            f"def {fn}(adj: dict[str, list[str]]) -> bool:\n"
            "    color: dict[str, int] = {}\n\n"
            "    def dfs(u: str) -> bool:\n"
            "        c = color.get(u, 0)\n"
            "        if c == 1:\n"
            "            return True\n"
            "        if c == 2:\n"
            "            return False\n"
            "        color[u] = 1\n"
            "        for v in adj.get(u, ()):\n"
            "            if dfs(v):\n"
            "                return True\n"
            "        color[u] = 2\n"
            "        return False\n\n"
            "    return any(dfs(node) for node in list(adj.keys()))"
        )
        test = (
            f"assert {fn}({{'A': ['B', 'C'], 'B': ['D'], 'C': ['D']}}) is False\n"
            f"assert {fn}({{'A': ['B'], 'B': ['C'], 'C': ['A']}}) is True\n"
            f"assert {fn}({{'X': ['X']}}) is True\n"
            f"assert {fn}({{}}) is False\n"
        )
        return {
            "id": f"v11_algo_{idx:04d}",
            "pillar": "diagnostic_debugging",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
        }

    if variant == 2:
        # Numerically Stable Logit Binary Cross-Entropy (`max(z, 0) - z*y + log1p(exp(-|z|))`)
        fn = rng.choice(["mean_logit_bce", "numerically_stable_bce", "binary_logit_nll"])
        if idx >= 24:
            fn = f"{fn}_{idx % 11}"
        prompt = (
            f"Implement `{fn}(logits: list[float], targets: list[int]) -> float` in pure Python (`math` only):\n"
            "- Compute the mean binary cross-entropy loss `-mean(y * ln(sigmoid(z)) + (1 - y) * ln(1 - sigmoid(z)))` for `y in {0, 1}`.\n"
            "- Must not overflow or return `nan` on extreme logits like `+1000.0` or `-1000.0` (`[] -> 0.0`)."
        )
        think = (
            "Derive the exact overflow-free identity for `-y * log(sigmoid(z)) - (1 - y) * log(1 - sigmoid(z))`:\n"
            "- Since `sigmoid(z) = 1 / (1 + e^{-z})`, the per-item loss simplifies algebraically to `log(1 + e^z) - z * y`.\n"
            "- For `z > 0`, `log(1 + e^z) = z + log(1 + e^{-z})`; for `z <= 0`, `log(1 + e^z) = 0 + log(1 + e^z)`.\n"
            "- Combining both branches into one formula: `loss(z, y) = max(z, 0.0) - z * y + math.log1p(math.exp(-abs(z)))`!\n"
            "- Check extremes:\n"
            "  - `z = +1000.0, y = 1`: `1000 - 1000 + log1p(exp(-1000)) = 0.0` (exact!).\n"
            "  - `z = -1000.0, y = 1`: `0 - (-1000) + 0 = 1000.0` (finite, zero overflow!)."
        )
        code = (
            "import math\n\n"
            f"def {fn}(logits: list[float], targets: list[int]) -> float:\n"
            "    if not logits:\n"
            "        return 0.0\n"
            "    total = 0.0\n"
            "    for z, y in zip(logits, targets):\n"
            "        total += max(z, 0.0) - z * float(y) + math.log1p(math.exp(-abs(z)))\n"
            "    return total / len(logits)"
        )
        test = (
            "import math\n"
            f"assert abs({fn}([1000.0, -1000.0], [1, 0])) < 1e-12\n"
            f"assert abs({fn}([0.0], [1]) - math.log(2.0)) < 1e-12\n"
            f"assert abs({fn}([-1000.0], [1]) - 1000.0) < 1e-9\n"
        )
        return {
            "id": f"v11_algo_{idx:04d}",
            "pillar": "diagnostic_debugging",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
        }

    if variant == 3:
        # Numerically Stable `log_softmax` + NLL returning `(list[float], float)` (NEVER `list / float`!)
        fn = rng.choice(["log_softmax_with_ce", "compute_logprobs_and_loss", "stable_categorical_nll"])
        if idx >= 24:
            fn = f"{fn}_{idx % 11}"
        prompt = (
            f"Implement `{fn}(scores: list[float], gold_idx: int) -> tuple[list[float], float]` in pure Python (`math`):\n"
            "- Return `(log_probs, nll)` where `log_probs[i] = ln(exp(scores[i]) / sum_j exp(scores[j]))` and `nll = -log_probs[gold_idx]`, remaining finite for `scores` in `[-1e4, +1e4]`."
        )
        think = (
            "Let's make sure we work in **log-space** (`x - lse`), NOT probability space (`probs / s` on a Python `list` raises `TypeError` and `math.log(0.0)` crashes on `-1000.0`!):\n"
            "1. `m = max(scores)`.\n"
            "2. `lse = m + math.log(sum(math.exp(x - m) for x in scores))`.\n"
            "3. `log_probs = [x - lse for x in scores]`.\n"
            "4. `nll = -log_probs[gold_idx]`. Verify on `[1000.0, -1000.0], gold_idx=1`: `lse = 1000.0`, `log_probs = [0.0, -2000.0]`, `nll = 2000.0` (exact, zero underflow/overflow!)."
        )
        code = (
            "import math\n\n"
            f"def {fn}(scores: list[float], gold_idx: int) -> tuple[list[float], float]:\n"
            "    m = max(scores)\n"
            "    lse = m + math.log(sum(math.exp(x - m) for x in scores))\n"
            "    log_probs = [x - lse for x in scores]\n"
            "    return log_probs, -log_probs[gold_idx]"
        )
        test = (
            f"lp, nll = {fn}([1000.0, -1000.0], 1)\n"
            "assert abs(lp[0]) < 1e-12 and abs(lp[1] - (-2000.0)) < 1e-9 and abs(nll - 2000.0) < 1e-9\n"
        )
        return {
            "id": f"v11_algo_{idx:04d}",
            "pillar": "diagnostic_debugging",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
        }

    if variant == 4:
        # Finite Hasse Poset / Lattice Least-Upper-Bound (`join`) via Upper-Set Intersection!
        fn = rng.choice(["poset_least_upper_bound", "dag_lattice_lub", "compute_type_lub"])
        if idx >= 24:
            fn = f"{fn}_{idx % 11}"
        prompt = (
            f"Implement `{fn}(covers: dict[str, list[str]], a: str, b: str) -> str` in Python:\n"
            "- `covers[u]` lists the immediate strict supertypes `v` (`u < v`) in a finite bounded join-semilattice.\n"
            "- Return the unique **least upper bound (`join`)** of `a` and `b` — i.e., the common upper bound `u` reachable from both `a` and `b` (`0` or more steps) that has NO strict subtype `w < u` also in their common upper bounds."
        )
        think = (
            "Crucial trap to avoid: a partial order is NOT a total linear order! Two incomparable branches (like `'Int'` and `'Bool'` both below `'Top'`) cannot be compared by a single integer rank.\n"
            "- Instead, compute the reachable upper closure `anc(x)` via BFS/DFS from `x` (including `x` itself).\n"
            "- Common upper bounds: `common = anc(a) & anc(b)`.\n"
            "- An element `u in common` is the **least** upper bound iff no OTHER `w in common` (`w != u`) can reach `u` (`u not in (anc(w) - {w})` for all `w in common`, equivalently `{w for w in common if all(u not in (anc(z) - {z}) for z in common)}` — wait: `u` is minimal in `common` iff for every `w in common`, if `u in anc(w)` then `w == u`!).\n"
            "- Test on diamond `Bot -> {A, B} -> Mid -> Top`, `Bot -> C -> Top`: `anc(A)={A,Mid,Top}`, `anc(C)={C,Top}`, `common={Top}` -> returns `'Top'`! For `(A, B)`, `common={Mid, Top}`: `Top in anc(Mid)` so `Top` is not minimal, `Mid` is minimal -> returns `'Mid'`!"
        )
        code = (
            f"def {fn}(covers: dict[str, list[str]], a: str, b: str) -> str:\n"
            "    def upper_closure(start: str) -> set[str]:\n"
            "        seen = {start}\n"
            "        stack = [start]\n"
            "        while stack:\n"
            "            cur = stack.pop()\n"
            "            for nxt in covers.get(cur, ()):\n"
            "                if nxt not in seen:\n"
            "                    seen.add(nxt)\n"
            "                    stack.append(nxt)\n"
            "        return seen\n\n"
            "    common = upper_closure(a) & upper_closure(b)\n"
            "    closures = {x: upper_closure(x) for x in common}\n"
            "    for cand in common:\n"
            "        if all(cand not in closures[other] for other in common if other != cand):\n"
            "            return cand\n"
            "    raise ValueError('No unique LUB')"
        )
        test = (
            "cov = {'Null': ['I32', 'Text'], 'I32': ['Scalar'], 'Scalar': ['Object'], 'Text': ['Object'], 'Object': []}\n"
            f"assert {fn}(cov, 'I32', 'Scalar') == 'Scalar'\n"
            f"assert {fn}(cov, 'I32', 'Text') == 'Object'\n"
            f"assert {fn}(cov, 'Null', 'Text') == 'Text'\n"
        )
        return {
            "id": f"v11_algo_{idx:04d}",
            "pillar": "unseen_ood_self_discover",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
        }

    if variant == 5:
        # Prefix-Sum + Monotonic Deque Shortest Subarray Sum `>= target` (Negative Entries Allowed!)
        fn = rng.choice(["shortest_window_at_least", "min_subarray_ge_sum", "deque_shortest_sum_ge"])
        if idx >= 24:
            fn = f"{fn}_{idx % 11}"
        prompt = (
            f"Implement `{fn}(arr: list[int], target: int) -> int` in `O(n)` Python (`collections.deque`):\n"
            "- `arr` may contain **negative**, zero, and positive integers.\n"
            "- Return the minimum length `r - l >= 1` of a contiguous subarray with sum `>= target`, or `-1` if none exists."
        )
        think = (
            "Two-pointer sliding window fails when `arr` has negative numbers! Use LeetCode-862 monotonic increasing prefix-sum deque:\n"
            "1. Build `pref` of length `n + 1` (`pref[0] = 0`, `pref[i + 1] = pref[i] + arr[i]`).\n"
            "2. Maintain `dq = deque()` of indices with strictly increasing `pref[idx]`:\n"
            "   - Pop from **front** while `dq and pref[r] - pref[dq[0]] >= target`: `best = min(best, r - dq.popleft())`.\n"
            "   - Pop from **back** while `dq and pref[dq[-1]] >= pref[r]`: `dq.pop()`.\n"
            "   - Append `r`.\n"
            "3. Verify on `[3, -2, 4], target = 5`: `pref = [0, 3, 1, 5]`; at `r=3 (5)`, `5 - 0 >= 5 -> len 3`, then `5 - 1 = 4 < 5` -> returns `3`."
        )
        code = (
            "from collections import deque\n\n"
            f"def {fn}(arr: list[int], target: int) -> int:\n"
            "    n = len(arr)\n"
            "    pref = [0] * (n + 1)\n"
            "    for i, v in enumerate(arr):\n"
            "        pref[i + 1] = pref[i] + v\n"
            "    dq: deque[int] = deque()\n"
            "    best = n + 1\n"
            "    for r, pr in enumerate(pref):\n"
            "        while dq and pr - pref[dq[0]] >= target:\n"
            "            best = min(best, r - dq.popleft())\n"
            "        while dq and pref[dq[-1]] >= pr:\n"
            "            dq.pop()\n"
            "        dq.append(r)\n"
            "    return best if best <= n else -1"
        )
        test = (
            f"assert {fn}([2, -1, 2], 3) == 3\n"
            f"assert {fn}([84, -37, 32, 40, 95], 167) == 3\n"
            f"assert {fn}([1, -2, 1], 4) == -1\n"
        )
        return {
            "id": f"v11_algo_{idx:04d}",
            "pillar": "diagnostic_debugging",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
        }

    if variant == 6:
        # Non-Mutating Synchronous Ring Step (`cur = list(cells)` + `0`-round base case!)
        fn = rng.choice(["evolve_mod7_ring", "synchronous_stencil_rounds", "advance_cyclic_automaton"])
        if idx >= 24:
            fn = f"{fn}_{idx % 11}"
        prompt = (
            f"Implement `{fn}(cells: list[int], steps: int) -> list[int]` in Python WITHOUT mutating `cells`:\n"
            "- On each synchronous step (`steps >= 0`, `[] -> []`), every index `i` updates simultaneously to `(2 * cells[(i - 1) % n] + cells[i] + 4 * cells[(i + 1) % n]) % 7`.\n"
            "- Return a fresh `list[int]` after `steps` steps (`steps == 0` must return a fresh copy equal to `cells`)."
        )
        think = (
            "Two classic bugs to avoid:\n"
            "1. If `steps == 0`, initializing `out = [0] * n` and looping `range(0)` would return all zeros instead of `list(cells)`! Always initialize `cur = list(cells)` first!\n"
            "2. Inside the `for _ in range(steps):` loop, read from `cur` to build a NEW list `nxt`, then rebind `cur = nxt` at the end of the round (never keep reading from original `cells` on step 2, and never overwrite `cur[i]` mid-round!)."
        )
        code = (
            f"def {fn}(cells: list[int], steps: int) -> list[int]:\n"
            "    cur = list(cells)\n"
            "    n = len(cur)\n"
            "    if n == 0:\n"
            "        return []\n"
            "    for _ in range(steps):\n"
            "        cur = [(2 * cur[(i - 1) % n] + cur[i] + 4 * cur[(i + 1) % n]) % 7 for i in range(n)]\n"
            "    return cur"
        )
        c0 = [3, 1, 5, 2]
        c2 = list(c0)
        for _ in range(2):
            c2 = [(2 * c2[(i - 1) % 4] + c2[i] + 4 * c2[(i + 1) % 4]) % 7 for i in range(4)]
        test = (
            "inp = [3, 1, 5, 2]\n"
            f"r0 = {fn}(inp, 0)\n"
            "assert r0 == [3, 1, 5, 2] and r0 is not inp\n"
            f"r2 = {fn}(inp, 2)\n"
            f"assert inp == [3, 1, 5, 2] and r2 == {c2!r}\n"
        )
        return {
            "id": f"v11_algo_{idx:04d}",
            "pillar": "unseen_ood_self_discover",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
        }

    # variant == 7: Lexicographically First CNF SAT Solver (`list[bool] | None`)
    fn = rng.choice(["solve_lex_min_cnf", "first_satisfying_assignment", "backtrack_cnf_lex"])
    if idx >= 24:
        fn = f"{fn}_{idx % 11}"
    prompt = (
        f"Implement `{fn}(n_vars: int, cnf: list[list[int]]) -> list[bool] | None` in Python:\n"
        "- Variables are `1..n_vars` (`n_vars <= 14`). Literal `+k` holds when `assign[k - 1]` is `True`; `-k` holds when `assign[k - 1]` is `False`.\n"
        "- Search `False` before `True` at each variable `0..n_vars-1` so the first satisfying assignment found is lexicographically smallest (`False < True`), or return `None` if unsatisfiable."
    )
    think = (
        "Let's design clean recursive backtracking with early falsified-clause pruning:\n"
        "- Represent partial assignment `assign: list[bool | None] = [None] * n_vars`.\n"
        "- A clause `cl` is **already falsified** if EVERY literal `lit in cl` has been assigned (`assign[abs(lit)-1] is not None`) and evaluates to `False`.\n"
        "- At depth `idx == n_vars`, all clauses are satisfied -> return `[bool(x) for x in assign]`.\n"
        "- At depth `idx < n_vars`, try `val = False` first, then `val = True` (`False < True` guarantees lexicographic minimum!). Reset `assign[idx] = None` on backtrack!"
    )
    code = (
        f"def {fn}(n_vars: int, cnf: list[list[int]]) -> list[bool] | None:\n"
        "    assign: list[bool | None] = [None] * n_vars\n\n"
        "    def partial_conflict() -> bool:\n"
        "        for cl in cnf:\n"
        "            clause_sat = False\n"
        "            has_unassigned = False\n"
        "            for lit in cl:\n"
        "                v = assign[abs(lit) - 1]\n"
        "                if v is None:\n"
        "                    has_unassigned = True\n"
        "                elif (lit > 0 and v) or (lit < 0 and not v):\n"
        "                    clause_sat = True\n"
        "                    break\n"
        "            if not clause_sat and not has_unassigned:\n"
        "                return True\n"
        "        return False\n\n"
        "    def dfs(i: int) -> list[bool] | None:\n"
        "        if partial_conflict():\n"
        "            return None\n"
        "        if i == n_vars:\n"
        "            return [bool(x) for x in assign]\n"
        "        for val in (False, True):\n"
        "            assign[i] = val\n"
        "            res = dfs(i + 1)\n"
        "            if res is not None:\n"
        "                return res\n"
        "        assign[i] = None\n"
        "        return None\n\n"
        "    return dfs(0)"
    )
    test = (
        f"assert {fn}(3, [[1, 2], [-1, 3], [2]]) == [False, True, False]\n"
        f"assert {fn}(2, [[1], [-1]]) is None\n"
    )
    return {
        "id": f"v11_algo_{idx:04d}",
        "pillar": "complex_logic_sat",
        "images": [],
        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n```python\n{code}\n```"}],
        "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
    }


# ===========================================================================
# 4. Zero-Prior Formal Systems & Complex Logic/SAT Exploratory Reasoning
# ===========================================================================


def make_v11_formal_and_logic_item(rng: random.Random, idx: int) -> dict[str, Any]:
    variant = idx % 6

    if variant == 0:
        # Tape Register Interpreter (careful 2-pop / 1-push arity — binary opcodes remove BOTH operands before pushing 1 result!)
        a0, b0, c0 = rng.randint(3, 8), rng.randint(10, 16), rng.randint(2, 5)
        d_v = b0 - a0
        fin_top = d_v * (d_v - c0)
        prompt = (
            "Simulate this LIFO register tape starting empty `[]` (head is the rightmost slot):\n"
            "- `EMIT k`: append integer `k` at the right end\n"
            "- `CLONE`: duplicate the rightmost integer in place\n"
            "- `FLIP`: transpose the rightmost pair `[..., u, v] -> [..., v, u]`\n"
            "- `MINUS`: pop rightmost `v`, pop next `u`, and append `v - u` (reduces length by 1)\n"
            "- `TIMES`: pop rightmost pair and append their product (reduces length by 1)\n"
            f"Program: `EMIT {a0} ; EMIT {b0} ; MINUS ; CLONE ; EMIT {c0} ; FLIP ; MINUS ; TIMES`.\n"
            'Return ONLY JSON `{"head_value": <int>, "tape_len": <int>}`.'
        )
        think = (
            "Track tape contents explicitly — every binary opcode (`MINUS`, `TIMES`) pops **both** operands before appending 1 result (net length `-1`):\n"
            f"1. `EMIT {a0}` -> `[{a0}]` (len 1)\n"
            f"2. `EMIT {b0}` -> `[{a0}, {b0}]` (len 2)\n"
            f"3. `MINUS`: pop `v = {b0}`, pop `u = {a0}` (tape `[]`), append `{b0} - {a0} = {d_v}` -> `[{d_v}]` (len 1)\n"
            f"4. `CLONE`: `[{d_v}, {d_v}]` (len 2)\n"
            f"5. `EMIT {c0}` -> `[{d_v}, {d_v}, {c0}]` (len 3)\n"
            f"6. `FLIP` -> `[{d_v}, {c0}, {d_v}]` (len 3)\n"
            f"7. `MINUS`: pop `v = {d_v}`, pop `u = {c0}` (leaving `[{d_v}]`), append `{d_v} - {c0} = {d_v - c0}` -> `[{d_v}, {d_v - c0}]` (len 2)\n"
            f"8. `TIMES`: pop `{d_v - c0}` and `{d_v}` (leaving `[]`), append `{d_v} * {d_v - c0} = {fin_top}` -> `[{fin_top}]` (len 1)."
        )
        expected = {"head_value": fin_top, "tape_len": 1}
        return {
            "id": f"v11_flog_{idx:04d}",
            "pillar": "zero_prior_formal",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n{json.dumps(expected)}"}],
            "verifier": {"expected": expected, "forbid_tool_call": True},
        }

    if variant == 1:
        # Parity-Switched Sequence (computing EVERY intermediate step `n=2, 3, 4, 5, 6, 7` without skipping odd `n=3, 5`!)
        h0, h1 = rng.randint(1, 3), rng.randint(2, 4)
        seq = {0: h0, 1: h1}
        for n_i in range(2, 8):
            if n_i % 2 == 0:
                seq[n_i] = 2 * seq[n_i - 1] - seq[n_i - 2] + 2
            else:
                seq[n_i] = seq[n_i - 1] + 2 * seq[n_i - 2]
        prompt = (
            f"Given `h(0) = {h0}`, `h(1) = {h1}`, and for `n >= 2`:\n"
            "  - even `n`: `h(n) = 2 * h(n - 1) - h(n - 2) + 2`\n"
            "  - odd `n`:  `h(n) = h(n - 1) + 2 * h(n - 2)`\n"
            'Compute `h(4)`, `h(6)`, and `h(7)` and return ONLY JSON `{"h4": <int>, "h6": <int>, "h7": <int>}`.'
        )
        think = (
            "CRITICAL: `h(4)` depends on `h(3)` and `h(2)` — NEVER skip odd indices `n = 3` and `n = 5`!\n"
            f"- `n = 0`: `{h0}`, `n = 1`: `{h1}`\n"
            f"- `n = 2` (even): `2*{seq[1]} - {seq[0]} + 2 = {seq[2]}`\n"
            f"- `n = 3` (odd): `{seq[2]} + 2*{seq[1]} = {seq[3]}`\n"
            f"- `n = 4` (even): `2*{seq[3]} - {seq[2]} + 2 = {seq[4]}`\n"
            f"- `n = 5` (odd): `{seq[4]} + 2*{seq[3]} = {seq[5]}`\n"
            f"- `n = 6` (even): `2*{seq[5]} - {seq[4]} + 2 = {seq[6]}`\n"
            f"- `n = 7` (odd): `{seq[6]} + 2*{seq[5]} = {seq[7]}`."
        )
        expected = {"h4": seq[4], "h6": seq[6], "h7": seq[7]}
        return {
            "id": f"v11_flog_{idx:04d}",
            "pillar": "unseen_ood_self_discover",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n{json.dumps(expected)}"}],
            "verifier": {"expected": expected, "forbid_tool_call": True},
        }

    if variant == 2:
        # Graph Coloring: Odd/Even Cycle `C_m` + Wheel Graph `W_{m+1}` (Hub connected to ALL rim vertices adds +1 color!)
        m_rim = rng.choice([5, 6, 7, 8])
        chi_c = 2 if (m_rim % 2 == 0) else 3
        chi_w = chi_c + 1  # Hub is adjacent to EVERY rim vertex, so hub CANNOT reuse any rim color -> `chi_c + 1`!
        ok_3 = chi_w <= 3
        prompt = (
            f"Consider cycle graph `C_{m_rim}` (`{m_rim}` rim vertices) and wheel graph `W_{m_rim + 1}` formed by adding a central hub vertex connected to all `{m_rim}` vertices of `C_{m_rim}`.\n"
            "Compute `chi_rim` (chromatic number of `C_m`), `chi_wheel` (chromatic number of `W_{m+1}`), and boolean `wheel_3_colorable` (`chi_wheel <= 3`).\n"
            'Return ONLY JSON `{"chi_rim": <int>, "chi_wheel": <int>, "wheel_3_colorable": <bool>}`.'
        )
        think = (
            f"Let's reason from graph adjacency — don't guess!\n"
            f"1. Cycle `C_{m_rim}` has `{m_rim}` (`{'even' if m_rim % 2 == 0 else 'odd'}`) vertices, so `chi_rim = {chi_c}`.\n"
            f"2. In `W_{m_rim + 1}`, the central hub is adjacent to **every single rim vertex**! Therefore the hub's color CANNOT appear anywhere on the rim `C_{m_rim}`.\n"
            f"3. Since coloring the rim alone requires `{chi_c}` colors, adding a universal apex/hub vertex ALWAYS increases the chromatic number by `+1`: `chi_wheel = {chi_c} + 1 = {chi_w}`!\n"
            f"4. Thus `wheel_3_colorable = ({chi_w} <= 3) = {str(ok_3).lower()}`."
        )
        expected = {"chi_rim": chi_c, "chi_wheel": chi_w, "wheel_3_colorable": ok_3}
        return {
            "id": f"v11_flog_{idx:04d}",
            "pillar": "complex_logic_sat",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n{json.dumps(expected)}"}],
            "verifier": {"expected": expected, "forbid_tool_call": True},
        }

    if variant == 3:
        # Impartial Subtraction Game DP (`0..N` explicit P/N table — counting ALL `P`-positions in `1..N`!)
        moves = rng.choice([(1, 2, 4), (1, 3, 4), (1, 2, 5)])
        n_pile = rng.randint(14, 20)
        dp = {0: False}  # False = P-position (second player wins), True = N-position (first player wins)
        for k_p in range(1, n_pile + 1):
            dp[k_p] = any((k_p - m >= 0) and (not dp[k_p - m]) for m in moves)
        p_positions = [k_p for k_p in range(1, n_pile + 1) if not dp[k_p]]
        win_moves = [m for m in moves if n_pile - m >= 0 and not dp[n_pile - m]]
        open_mv = min(win_moves) if win_moves else 0
        prompt = (
            f"In a normal-play subtraction game starting with `N = {n_pile}` stones, a player may remove any `m in {list(moves)}` stones; the player with no legal move (`0` stones) loses.\n"
            f"Determine `p1_wins` at `N = {n_pile}`, `smallest_winning_move` (`0` if `p1_wins` is `false`), and `p_count_1_to_N` (number of losing `P`-positions in `1..{n_pile}`).\n"
            'Return ONLY JSON `{"p1_wins": <bool>, "smallest_winning_move": <int>, "p_count_1_to_N": <int>}`.'
        )
        tbl_str = ", ".join(f"{k}:{'N' if dp[k] else 'P'}" for k in range(0, n_pile + 1))
        think = (
            f"Build the exact backward-induction table from `0` up to `{n_pile}` (`0` is `P`; `k` is `N` iff some `k - m` in `{list(moves)}` is `P`):\n"
            f"- Table: `{tbl_str}`.\n"
            f"- `P`-positions in `1..{n_pile}`: `{p_positions}` -> count = `{len(p_positions)}`.\n"
            f"- At `N = {n_pile}`, state is `{'N' if dp[n_pile] else 'P'}` (`p1_wins = {str(dp[n_pile]).lower()}`), winning moves to `P` = `{win_moves}` -> `{open_mv}`."
        )
        expected = {"p1_wins": dp[n_pile], "smallest_winning_move": open_mv, "p_count_1_to_N": len(p_positions)}
        return {
            "id": f"v11_flog_{idx:04d}",
            "pillar": "complex_logic_sat",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n{json.dumps(expected)}"}],
            "verifier": {"expected": expected, "forbid_tool_call": True},
        }

    if variant == 4:
        # Rank-Deficient Linear System with Exact Linear Combination Invariant (`Q` unique even though `x` is NOT unique!)
        c1, c2 = rng.randint(1, 3), rng.randint(2, 4)
        rhs1 = rng.randint(10, 22)
        mult_eq = rng.choice([2, 3])
        q_mult = rng.choice([3, 4, 5])
        q_val = q_mult * rhs1
        prompt = (
            f"Real variables `(x, y, z)` satisfy the two equations:\n"
            f"  (E1) `{c1}*x + {c2}*y - z = {rhs1}`\n"
            f"  (E2) `{mult_eq * c1}*x + {mult_eq * c2}*y - {mult_eq}*z = {mult_eq * rhs1}`\n"
            f"Determine whether `x` is uniquely determined (`x_is_unique`), whether `Q = {q_mult * c1}*x + {q_mult * c2}*y - {q_mult}*z` is uniquely determined (`q_is_unique`), and `q_exact` (`null` if not unique).\n"
            'Return ONLY JSON `{"x_is_unique": <bool>, "q_is_unique": <bool>, "q_exact": <int|null>}`.'
        )
        think = (
            f"Check linear independence AND check whether `Q` lies in the rowspace!\n"
            f"1. Notice `(E2) = {mult_eq} * (E1)` (`({mult_eq * c1}, {mult_eq * c2}, {-mult_eq}) == {mult_eq} * ({c1}, {c2}, -1)`), so we have only **1 independent equation in 3 unknowns**! For instance `(x=0, y=0, z={-rhs1})` and `(x=1, y=0, z={c1 - rhs1})` both satisfy `(E1)` and `(E2)` with different `x` -> `x_is_unique = false`.\n"
            f"2. Now check `Q`: `({q_mult * c1}, {q_mult * c2}, {-q_mult}) = {q_mult} * ({c1}, {c2}, -1)`! For EVERY solution `(x, y, z)` of `(E1)`, `Q = {q_mult} * ({c1}*x + {c2}*y - z) = {q_mult} * {rhs1} = {q_val}`!\n"
            f"3. Even though individual variables are underdetermined, `Q` is an exact multiple of `(E1)` so `q_is_unique = true` and `q_exact = {q_val}`!"
        )
        expected = {"x_is_unique": False, "q_is_unique": True, "q_exact": q_val}
        return {
            "id": f"v11_flog_{idx:04d}",
            "pillar": "zero_prior_formal",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n{json.dumps(expected)}"}],
            "verifier": {"expected": expected, "forbid_tool_call": True},
        }

    # variant == 5: Knights & Knaves 3-Islander Biconditional Truth Table Enumeration
    # P says "Q is a Knave", Q says "P and R have the same type", R says "Q is a Knight"
    # Let's check all 8 `(p, q, r)` in `{False, True}^3`:
    sols = []
    for p_v in (False, True):
        for q_v in (False, True):
            for r_v in (False, True):
                ok_p = p_v == (not q_v)
                ok_q = q_v == (p_v == r_v)
                ok_r = r_v == q_v
                if ok_p and ok_q and ok_r:
                    sols.append((p_v, q_v, r_v))
    assert len(sols) == 1  # p=True -> q=False -> r=False -> p==r is False == q(False) -> unique (True, False, False)!
    p_ans, q_ans, r_ans = sols[0]
    prompt = (
        "On an island of Knights (always tell the truth) and Knaves (always lie), three residents `Paxon`, `Quill`, and `Riven` speak:\n"
        "- `Paxon`: \"Quill is a Knave.\"\n"
        "- `Quill`: \"Paxon and Riven are of the same type (both Knights or both Knaves).\"\n"
        "- `Riven`: \"Quill is a Knight.\"\n"
        'Return ONLY JSON `{"Paxon": <bool>, "Quill": <bool>, "Riven": <bool>, "knights_total": <int>}` (`true` = Knight, `false` = Knave).'
    )
    think = (
        "Represent each statement as an exact Boolean biconditional `speaker == claim` and test BOTH cases of `Quill` (`Q`):\n"
        "- Case 1 (`Q = true`): Riven's claim `Q == true` is true, so `R = true`. Paxon's claim `Q == false` is false, so `P = false`. Then `P == R` is `false == true` (`false`), contradicting `Q = true`!\n"
        "- Case 2 (`Q = false`): Riven's claim `Q == true` is false -> `R = false`. Paxon's claim `Q == false` is true -> `P = true`. Check Quill's claim `P == R`: `true == false` is `false`, which matches `Q = false` (Quill is a Knave lying)!\n"
        "- Unique assignment: `Paxon = true, Quill = false, Riven = false`, `knights_total = 1`."
    )
    expected = {"Paxon": p_ans, "Quill": q_ans, "Riven": r_ans, "knights_total": int(p_ans) + int(q_ans) + int(r_ans)}
    return {
        "id": f"v11_flog_{idx:04d}",
        "pillar": "complex_logic_sat",
        "images": [],
        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n{json.dumps(expected)}"}],
        "verifier": {"expected": expected, "forbid_tool_call": True},
    }


# ===========================================================================
# 5. True `8k..26k` Long-Context Multi-Needle + Erratum + `[LOG-XXXX]` SFT
# ===========================================================================


def make_v11_true_long_kv_item(rng: random.Random, idx: int) -> dict[str, Any]:
    """Generate true `8k..26k`-token multi-needle + superseding-erratum archives in a strictly disjoint
    Oceanographic / Seismic / Compiler Telemetry domain (`[LOG-XXXX]`) for Zero-Swap Prefix-KV SFT!
    """
    expeditions = ["Mariana-Trench", "Cascadia-Array", "Reykjanes-Ridge", "Sunda-Arc", "Kermadec-Deep", "Aleutian-Watch", "Azores-Buoy", "Lofoten-Acoustic"]
    probes = ["PRB-Hydrophone", "PRB-Magneto", "PRB-Tiltmeter", "PRB-CTD", "PRB-Doppler", "PRB-Gravimeter", "PRB-Broadband", "PRB-Piezometer"]
    obsolete = ["OBS-ProtoA", "OBS-ProtoB", "OBS-ProtoC", "OBS-ProtoD"]

    exp_name = f"{expeditions[idx % len(expeditions)]}-{100 + idx}"
    active_prb = f"{probes[(idx * 5) % len(probes)]}-{(idx % 7) + 2}"
    obs_prb = f"{obsolete[idx % len(obsolete)]}-{idx % 4}"
    base_mv = rng.randint(24, 68)
    revoked_mult = rng.choice([2, 3, 8])
    auth_mult = rng.choice([4, 5, 6, 7])
    cal_offset = rng.randint(12, 38)
    calibrated_val = base_mv * auth_mult + cal_offset

    # Span `8k, 14k, 20k, 26k` tokens!
    tok_target = [8192, 14336, 20480, 26624][idx % 4]
    num_logs = max(90, int(round(tok_target / 80.0)))
    i1 = max(3, int(num_logs * rng.uniform(0.08, 0.18)))
    ist = max(i1 + 5, int(num_logs * rng.uniform(0.32, 0.42)))
    i2 = max(ist + 5, int(num_logs * rng.uniform(0.54, 0.66)))
    i3 = min(num_logs - 3, max(i2 + 5, int(num_logs * rng.uniform(0.82, 0.92))))

    id1, id_st, id2, id3 = f"LOG-{i1:04d}", f"LOG-{ist:04d}", f"LOG-{i2:04d}", f"LOG-{i3:04d}"
    basins = ["Abyssal-Plain-N", "Transform-Fault-E", "Seamount-Flank-W", "Subduction-Toe-S", "Hydrothermal-Vent-4", "Continental-Slope-2"]
    metrics = ["salinity conductivity", "acoustic ping roundtrip", "benthic pressure drift", "battery bus ripple", "fiber-optic Bragg strain", "rubidium clock phase"]

    entries: list[str] = []
    for k in range(1, num_logs + 1):
        tag = f"[LOG-{k:04d}]"
        if k == i1:
            entries.append(
                f"{tag} (Official Sensor Deployment Manifest): For survey `{exp_name}`, the active seafloor instrument is `{active_prb}` "
                f"(retired testbed `{obs_prb}` was pulled before deployment and must never be used)."
            )
        elif k == ist:
            entries.append(
                f"{tag} (Early Deck Bench Note): Preliminary bench gain for `{active_prb}` was tentatively written as `{revoked_mult}x` pending seafloor pressure calibration."
            )
        elif k == i2:
            entries.append(
                f"{tag} (Verified Seafloor Baseline Telemetry): Validated raw signal for active instrument `{active_prb}` is `{base_mv}` mV "
                f"(whereas retired testbed `{obs_prb}` logged `888` mV)."
            )
        elif k == i3:
            entries.append(
                f"{tag} (Authoritative Pressure Calibration Erratum — Supersedes {id_st}): Formal calibration REVOKES `{id_st}` (`{revoked_mult}x` invalidated). "
                f"The authoritative calibrated reading for `{active_prb}` is `calibrated_mv = raw_mv * {auth_mult} + {cal_offset}`."
            )
        else:
            bs = basins[(k * 7) % len(basins)]
            mt = metrics[(k * 3) % len(metrics)]
            n1 = 200 + ((k * 41) % 700)
            n2 = 15 + ((k * 23) % 70)
            entries.append(
                f"{tag} (Routine Buoy Housekeeping — `{bs}`): Hourly check of {mt} on mooring `M-{k % 24:02d}` recorded `{n1}` packets, "
                f"`{n2}` ppm drift, nominal umbilical insulation resistance, and zero acoustic modem framing drops."
            )

    prompt = (
        f"You are auditing a `{num_logs}`-entry oceanographic archive (`[LOG-0001]` .. `[LOG-{num_logs:04d}]`).\n"
        f"Determine the authoritative `calibrated_mv` for survey `{exp_name}`:\n"
        f"1. Identify the active seafloor instrument for `{exp_name}`.\n"
        "2. Find its verified raw signal in mV.\n"
        "3. Apply ONLY the authoritative calibration erratum (excluding retired testbeds and revoked bench notes).\n"
        "Cite all 3 authoritative `[LOG-XXXX]` IDs inline in square brackets in your explanation — do NOT cite the revoked note as authoritative — and end with JSON:\n"
        '`{"instrument": "<str>", "calibrated_mv": <int>, "supporting_logs": ["LOG-XXXX", "LOG-YYYY", "LOG-ZZZZ"]}`\n\n'
        "=== BEGIN ARCHIVE ===\n"
        + "\n".join(entries)
        + f"\n=== END ARCHIVE ===\nSynthesize the verified answer for `{exp_name}` with inline `[LOG-XXXX]` citations and trailing JSON."
    )
    think = (
        f"Let's trace the 3 authoritative hops across `[LOG-0001]`..`[LOG-{num_logs:04d}]` and verify exact arithmetic + bracket tags:\n"
        f"1. `[{id1}]` designates `{active_prb}` as the active instrument for `{exp_name}` (retiring `{obs_prb}`).\n"
        f"2. `[{id2}]` gives validated `raw_mv = {base_mv}` mV for `{active_prb}`.\n"
        f"3. `[{id3}]` revokes `{id_st}` (`{revoked_mult}x`) and establishes `calibrated_mv = {base_mv} * {auth_mult} + {cal_offset} = {base_mv * auth_mult} + {cal_offset} = {calibrated_val}`.\n"
        f"Ensure ALL three tags `[{id1}]`, `[{id2}]`, and `[{id3}]` appear in square brackets in prose and in `supporting_logs`!"
    )
    ans_json = {"instrument": active_prb, "calibrated_mv": calibrated_val, "supporting_logs": [id1, id2, id3]}
    ans = (
        f"For survey `{exp_name}`, the active seafloor instrument is `{active_prb}` [{id1}], which recorded a verified baseline signal of `{base_mv}` mV [{id2}]. "
        f"Applying the authoritative pressure calibration erratum `calibrated_mv = {base_mv} * {auth_mult} + {cal_offset} = {calibrated_val}` (which supersedes `{id_st}`) yields `{calibrated_val}` mV [{id3}].\n"
        f"{json.dumps(ans_json)}"
    )
    return {
        "id": f"v11_longkv_{idx:04d}",
        "pillar": "long_context_synthesis",
        "target_tokens": tok_target,
        "images": [],
        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}\n</think>\n{ans}"}],
    }


# ===========================================================================
# 6. Contrastive `ORPO` Pairs (`320`) Targeting Exact Failure Modes
# ===========================================================================


def build_v11_orpo_pairs(rng: random.Random, existing_orpo: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = list(existing_orpo)
    for i in range(80):
        kind = i % 5
        if kind == 0:
            # 1-return `svd(..., compute_uv=False)` vs buggy 2-tuple unpacking
            pairs.append({
                "id": f"v11_orpo_svd_{i}",
                "messages": [{"role": "user", "content": f"Write `top_singular_val_{i}(A: np.ndarray) -> float` returning the largest singular value of 2D `A`."}],
                "chosen": f"<think>\n`np.linalg.svd(A, compute_uv=False)` returns a single 1D array `s`, NOT a tuple!</think>\n```python\nimport numpy as np\n\ndef top_singular_val_{i}(A: np.ndarray) -> float:\n    s = np.linalg.svd(A.astype(np.float64), compute_uv=False)\n    return float(s[0]) if s.size else 0.0\n```",
                "rejected": f"<think>\nCall svd with compute_uv=False.</think>\n```python\nimport numpy as np\n\ndef top_singular_val_{i}(A: np.ndarray) -> float:\n    _, s = np.linalg.svd(A.astype(np.float64), compute_uv=False)\n    return float(np.max(s))\n```",
            })
        elif kind == 1:
            # Dict weight interpolation via dict comprehension vs `float * dict` TypeError
            pairs.append({
                "id": f"v11_orpo_dict_{i}",
                "messages": [{"role": "user", "content": f"Write `lerp_weights_{i}(d0: dict[str, np.ndarray], d1: dict[str, np.ndarray], t: float) -> dict[str, np.ndarray]`."}],
                "chosen": f"<think>\nPython `dict` does not support `*` or `+`; iterate keys in a dict comprehension.</think>\n```python\nimport numpy as np\n\ndef lerp_weights_{i}(d0: dict[str, np.ndarray], d1: dict[str, np.ndarray], t: float) -> dict[str, np.ndarray]:\n    return {{k: (1.0 - t) * d0[k] + t * d1[k] for k in d0}}\n```",
                "rejected": f"<think>\nInterpolate dictionaries directly.</think>\n```python\nimport numpy as np\n\ndef lerp_weights_{i}(d0: dict[str, np.ndarray], d1: dict[str, np.ndarray], t: float) -> dict[str, np.ndarray]:\n    return (1.0 - t) * d0 + t * d1\n```",
            })
        elif kind == 2:
            # Exact tool name & relative path vs hallucinated `/cline/modules/` or `read_files`
            rel = f"sched_core/step_gate_{i}.py"
            pairs.append({
                "id": f"v11_orpo_path_{i}",
                "messages": [{"role": "user", "content": f"Tools: `read_file(path: str)`.\nInspect `{rel}` before answering."}],
                "chosen": f"<think>\nCall exact tool `read_file` with exact relative path `{rel}`.</think>\n{_xml_call('read_file', {'path': rel})}",
                "rejected": f"<think>\nRead file from workspace.</think>\n{_xml_call('read_files', {'path': '/cline/modules/' + rel})}",
            })
        elif kind == 3:
            # Lower-bound binary search `(lo + hi) // 2` with `hi = mid` vs buggy `(lo + hi + 1) // 2` with `lo = mid`
            pairs.append({
                "id": f"v11_orpo_bs_{i}",
                "messages": [{"role": "user", "content": f"Write `first_ge_{i}(a: list[int], t: int) -> int` returning smallest index with `a[idx] >= t` or `-1`."}],
                "chosen": f"<think>\nHalf-open `[0, len(a))`, floor mid `(lo + hi) // 2`, `hi = mid` when `a[mid] >= t` else `lo = mid + 1`.</think>\n```python\ndef first_ge_{i}(a: list[int], t: int) -> int:\n    lo, hi = 0, len(a)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if a[mid] >= t:\n            hi = mid\n        else:\n            lo = mid + 1\n    return lo if lo < len(a) else -1\n```",
                "rejected": f"<think>\nBinary search midpoint.</think>\n```python\ndef first_ge_{i}(a: list[int], t: int) -> int:\n    lo, hi = 0, len(a)\n    while lo < hi:\n        mid = (lo + hi + 1) // 2\n        if a[mid] >= t:\n            lo = mid\n        else:\n            hi = mid - 1\n    return lo if lo < hi else -1\n```",
            })
        else:
            # Rectangular SVD reconstruction MUST keep `@ Vt`!
            pairs.append({
                "id": f"v11_orpo_svf_{i}",
                "messages": [{"role": "user", "content": f"Write `scale_sv_{i}(W: np.ndarray, g: np.ndarray) -> np.ndarray` returning `U @ diag(S * g) @ Vt`."}],
                "chosen": f"<think>\nMultiply `(U * (S * g)[None, :]) @ Vt` so `g = ones` reconstructs `W` exactly.</think>\n```python\nimport numpy as np\n\ndef scale_sv_{i}(W: np.ndarray, g: np.ndarray) -> np.ndarray:\n    U, S, Vt = np.linalg.svd(W.astype(np.float64), full_matrices=False)\n    return (U * (S * g)[None, :]) @ Vt\n```",
                "rejected": f"<think>\nScale U by S * g.</think>\n```python\nimport numpy as np\n\ndef scale_sv_{i}(W: np.ndarray, g: np.ndarray) -> np.ndarray:\n    U, S, Vt = np.linalg.svd(W.astype(np.float64), full_matrices=False)\n    return (U @ np.diag(S)) * g\n```",
            })
    return pairs


def main() -> None:
    V11_CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261123)

    # 1. Load all held-out tasks SOLELY for strict zero-leakage decontamination verification!
    heldout_paths = [
        ROOT_DIR / "training" / "data" / "rpr_v5_rsi_corpus" / "heldout_v5_rsi_56.jsonl",
        V9_CORPUS_DIR / "heldout_zero_prior_24.jsonl",
        V10_CORPUS_DIR / "heldout_v10_apex_36.jsonl",
        V10_CORPUS_DIR / "heldout_long_256k_needles_12.jsonl",
    ]
    heldout_all: list[dict[str, Any]] = []
    for hp in heldout_paths:
        heldout_all.extend(json.loads(l) for l in hp.read_text(encoding="utf-8").splitlines() if l.strip())

    # 2. Load proven base corpus (`v10` SFT, ORPO, DAPO, Val)
    base_sft = [json.loads(l) for l in (V10_CORPUS_DIR / "train_v10_sft.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    base_val = [json.loads(l) for l in (V10_CORPUS_DIR / "val_v10_joint.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    base_orpo = [json.loads(l) for l in (V10_CORPUS_DIR / "orpo_v10_pairs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    base_dapo = [json.loads(l) for l in (V10_CORPUS_DIR / "dapo_v10_hard_episodes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    # 3. Synthesize 620 new deep exploratory v11 traces + unit-verify every single Python function in `python3 -I`!
    new_items: list[dict[str, Any]] = []
    for i in range(180):
        new_items.append(make_v11_weight_mod_rsi_item(rng, i))
    for i in range(160):
        new_items.append(make_v11_research_discovery_item(rng, i))
    for i in range(150):
        new_items.append(make_v11_algo_invariant_item(rng, i))
    for i in range(130):
        new_items.append(make_v11_formal_and_logic_item(rng, i))

    code_verified = 0
    dapo_extra: list[dict[str, Any]] = []
    for it in new_items:
        ver = it.get("verifier")
        if isinstance(ver, dict):
            a_txt = it["messages"][-1]["content"]
            u_txt = it["messages"][0]["content"]
            kind = "python_code" if ver.get("kind") == "python_code" else ("tool_pos" if ver.get("tool") else ("citation_grounded" if "required_citations" in ver else "json_values"))
            sc = score_response(a_txt, prompt_text=u_txt, kind=kind, verifier=ver, max_think_tokens=900)
            assert sc["correct"], f"Self-verification failed on {it['id']}: {sc}"
            if kind == "python_code":
                code_verified += 1
            if len(dapo_extra) < 64 and rng.random() < 0.16:
                dapo_extra.append({
                    "id": f"dapo_{it['id']}",
                    "pillar": it["pillar"],
                    "kind": kind,
                    "messages": it["messages"][:-1],
                    "verifier": ver,
                    "reference_answer": a_txt,
                })

    rng.shuffle(new_items)
    val_new = new_items[:45]
    train_new = new_items[45:]

    # Deduplicate by user prompt content while preserving multimodal & multi-turn diversity
    seen_u: set[str] = set()
    merged_sft: list[dict[str, Any]] = []
    for row in train_new + base_sft:
        sig = json.dumps([m.get("content", "") for m in row["messages"][:-1]], sort_keys=True)
        if sig in seen_u:
            continue
        seen_u.add(sig)
        merged_sft.append(row)

    merged_val = base_val + val_new
    long_kv_rows = [make_v11_true_long_kv_item(rng, i) for i in range(80)]
    orpo_all = build_v11_orpo_pairs(rng, base_orpo)
    dapo_all = base_dapo + dapo_extra
    rng.shuffle(merged_sft)
    rng.shuffle(orpo_all)
    rng.shuffle(dapo_all)

    # Verify all newly synthesized items strictly first, and scrub any legacy base rows with accidental 10-gram phrasing overlap
    _assert_zero_heldout_leakage(new_items + long_kv_rows, heldout_paths)
    eval_fn_s: set[str] = set()
    eval_10g_s: set[tuple[str, ...]] = set()
    import re as _re
    for r_ev in heldout_all:
        u_ev = " ".join(str(m.get("content", "")) for m in r_ev.get("messages", []) if m.get("role") == "user")
        for fn_m in _re.findall(r"(?:Implement|Write|def)\s+`?([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", u_ev):
            if len(fn_m) >= 6:
                eval_fn_s.add(fn_m)
        w_ev = u_ev.lower().split()
        for j_w in range(max(0, len(w_ev) - 10 + 1)):
            eval_10g_s.add(tuple(w_ev[j_w : j_w + 10]))

    def _clean_legacy(rows_in: list[dict[str, Any]]) -> list[dict[str, Any]]:
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

    merged_sft = _clean_legacy(merged_sft)
    merged_val = _clean_legacy(merged_val)

    # 4. Strict Zero-Contamination Audit against `pilot_problems`, `benchmark_100`, AND all 128 held-out tasks!
    all_train_check = merged_sft + merged_val + long_kv_rows
    assert_decontaminated(all_train_check)
    _assert_zero_heldout_leakage(all_train_check, heldout_paths)

    def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

    _write_jsonl(V11_CORPUS_DIR / "train_v11_sft.jsonl", merged_sft)
    _write_jsonl(V11_CORPUS_DIR / "train_v11_long_kv.jsonl", long_kv_rows)
    _write_jsonl(V11_CORPUS_DIR / "val_v11_joint.jsonl", merged_val)
    _write_jsonl(V11_CORPUS_DIR / "orpo_v11_pairs.jsonl", orpo_all)
    _write_jsonl(V11_CORPUS_DIR / "dapo_v11_episodes.jsonl", dapo_all)

    stats = {
        "train_v11_sft": len(merged_sft),
        "train_v11_long_kv_8k_26k": len(long_kv_rows),
        "val_v11_joint": len(merged_val),
        "orpo_v11_pairs": len(orpo_all),
        "dapo_v11_episodes": len(dapo_all),
        "python_functions_verified_isolated": code_verified,
        "heldout_tasks_checked_zero_leakage": len(heldout_all),
    }
    (V11_CORPUS_DIR / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"[v11 OMEGA Corpus Built & Zero-Leakage Verified] {json.dumps(stats)}", flush=True)


if __name__ == "__main__":
    main()
