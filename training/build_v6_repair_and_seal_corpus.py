"""Build the RPR v6 Corpus: Leftover-Limits Repair (`python_code` + Unassisted Math) + 3-Split SEAL/SVF Self-Weight-Rewriting RL Suite.

Generates inside `training/data/rpr_v6_seal_corpus/`:
  1. `train_v6_repair.jsonl` (~460 unique `1x` conversations targeting `python_code` SVD/axis/binary-search precision,
     unassisted mental-math witnesses, `<weight_rewrite>` SEAL+SVF demonstrations, and multimodal/tool/research anchors)
  2. `val_v6_repair.jsonl` (~76 validation conversations)
  3. `rlvr_v6_repair_64.jsonl` (64 shuffled on-policy RLVR episodes heavily weighted toward `python_code` + unassisted math/RPR)
  4. `seal_svf_episodes.json` (Strict 3-split `Diagnose != Reward != Final Test` episodes for live `theta_parent -> theta_copy`
     self-weight-rewriting RL and out-of-sample autonomous self-rewrite evaluation)
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from training.build_rpr_v5_rsi_corpus import (
    V5_CORPUS_DIR,
    _deduplicate_rows,
    _verify_executable_code_rows,
    make_v5_clause_witness_rpr_item,
    make_v5_multistep_tool_item,
    make_v5_research_ablation_item,
    make_v5_rsi_weight_surgery_item,
    make_v5_vision_item,
)
from training.build_training_data import assert_decontaminated
from training.collect_and_build_rpr_corpus import _strip_prior_think
from training.high_signal_pillars import make_diagnostic_debug_item


V6_CORPUS_DIR = Path(__file__).resolve().parent / "data" / "rpr_v6_seal_corpus"

SEAL_SYSTEM_PROMPT = (
    "You are an autonomous Recursive Self-Improvement (RSI) weight-rewriting policy (Hybrid SEAL + SVF + Nullspace-LoRA).\n"
    "Given diagnostic failure traces from a model copy on a dev batch, diagnose the root weight/reasoning deficit in <think>, "
    "then emit ONLY a valid JSON `<weight_rewrite>` object with exact keys:\n"
    "{\n"
    '  "diagnosis": "<concise root-cause summary>",\n'
    '  "action": "<rewrite | noop>",\n'
    '  "target_group": "<upper_full_attn_mlp | all_upper_blocks>",\n'
    '  "svf_gains": [<g_code>, <g_math>, <g_rpr>, <g_rsi>],\n'
    '  "lr_b": <float 0.0..0.00015>,\n'
    '  "lr_a": <float 0.0..0.00005>,\n'
    '  "steps": <int 0..6>,\n'
    '  "nullspace_alpha": <float 0.5..0.98>,\n'
    '  "wise_ft_gamma": <float 0.80..1.00>,\n'
    '  "self_edits": [{"prompt": "<synthetic training prompt>", "response": "<think>...</think>\\n<verified answer>"}]\n'
    "}"
)


# ---------------------------------------------------------------------------
# Parameterized Python/NumPy Coding Variants (Zero Memorization, 100% Unit-Tested)
# ---------------------------------------------------------------------------


def make_v6_parameterized_code_item(rng: random.Random, idx: int) -> dict[str, Any]:
    """Generate diverse, parameterized Python/NumPy & algorithmic debugging tasks (100% unit-tested, zero eval overlap)."""
    sys_code = rng.choice([
        "You are a frontier AI weight-surgery and Recursive Self-Improvement (RSI) engineer. Derive exact matrix invariants in <think>, then provide clean Python code inside ```python ... ```.",
        "Diagnose root causes and shape/loop invariants inside <think>, then output clean Python code inside ```python ... ```.",
    ])
    variant = idx % 10

    if variant == 0:
        fn_name = rng.choice(["project_update_nullspace", "nullspace_orthogonal_step", "alpha_edit_null_update", "zero_drift_lora_merge"])
        tol_e = rng.choice([5, 6, 7, 8, 9])
        scale = rng.choice([0.5, 1.0, 1.25, 1.5, 2.0])
        d_out, d_in = rng.randint(4, 10), rng.randint(9, 16)
        prompt = (
            f"Implement `{fn_name}(W, X_anchor, dW, sv_tol=1e-{tol_e})` in NumPy (`import numpy as np`) "
            f"(e.g. for a `[{d_out}, {d_in}]` layer withUpdate scale `{scale}` when `alpha={scale}` is `1.0` here: return `W + (dW @ P_perp)`):\n"
            "- `W` (`[d_out, d_in]`): pretrained linear layer weight matrix (`y = X @ W.T`).\n"
            "- `X_anchor` (`[N, d_in]`): preserved input activations whose outputs `X_anchor @ W_new.T` MUST equal `X_anchor @ W.T`.\n"
            "- `dW` (`[d_out, d_in]`): proposed weight delta.\n"
            "Unpack `_, s_vals, vh = np.linalg.svd(X_anchor, full_matrices=False)`, select `v_k = vh[s_vals > sv_tol]`, "
            "build `p_perp = np.eye(W.shape[1]) - v_k.T @ v_k`, and return `W + (dW @ p_perp)`."
        )
        think = (
            "[L1] In NumPy, `_, s_vals, vh = np.linalg.svd(X_anchor, full_matrices=False)` returns `(U, S, Vh)` where `vh` has shape `[k, d_in]`.\n"
            "[L2] Keep orthonormal basis rows `v_k = vh[s_vals > sv_tol]`. Then `p_perp = np.eye(d_in, dtype=W.dtype) - (v_k.T @ v_k)` satisfies `X_anchor @ p_perp == 0`.\n"
            "[L3] Handle empty `X_anchor` or `v_k.shape[0] == 0` (`p_perp = I`) and return `W + (dW @ p_perp)`."
        )
        code = (
            "import numpy as np\n\n"
            f"def {fn_name}(W: np.ndarray, X_anchor: np.ndarray, dW: np.ndarray, sv_tol: float = 1e-{tol_e}) -> np.ndarray:\n"
            "    d_in = W.shape[1]\n"
            "    if X_anchor.size == 0:\n"
            "        return W + dW\n"
            "    _, s_vals, vh = np.linalg.svd(X_anchor, full_matrices=False)\n"
            "    v_k = vh[s_vals > sv_tol]\n"
            "    p_perp = np.eye(d_in, dtype=W.dtype) - (v_k.T @ v_k) if v_k.shape[0] > 0 else np.eye(d_in, dtype=W.dtype)\n"
            "    return W + (dW @ p_perp)"
        )
        test = (
            "import numpy as np\n"
            f"rng = np.random.default_rng({idx + 101})\n"
            f"W = rng.normal(size=({d_out}, {d_in}))\n"
            f"X = rng.normal(size=(3, {d_in}))\n"
            f"dW = rng.normal(size=({d_out}, {d_in}))\n"
            f"W2 = {fn_name}(W, X, dW)\n"
            "assert np.allclose(X @ W2.T, X @ W.T, atol=1e-9)\n"
            "assert not np.allclose(W2, W, atol=1e-4)"
        )
        return {
            "pillar": "rsi_and_weight_surgery",
            "images": [],
            "messages": [
                {"role": "system", "content": sys_code},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"<think>{think}</think>\n```python\n{code}\n```"},
            ],
            "verifier": {"kind": "python_code", "test": test},
        }

    if variant == 1:
        fn_name = rng.choice(["merge_dora_weights", "apply_dora_row_fusion", "dora_directional_fuse", "compute_dora_merged_linear"])
        eps_e = rng.choice([6, 7, 8, 9])
        d_out, d_in, r_dim = rng.randint(4, 9), rng.randint(6, 13), rng.randint(2, 5)
        prompt = (
            f"Implement `{fn_name}(W0, B, A, m, scale, eps=1e-{eps_e})` in NumPy (`import numpy as np`) "
            f"(tested on `W0` of shape `[{d_out}, {d_in}]`, rank `r={r_dim}`):\n"
            "- `W0` (`[d_out, d_in]`), `B` (`[d_out, r]`), `A` (`[r, d_in]`), row-magnitude vector `m` (`[d_out]`), float `scale`.\n"
            "Compute `v_dir = W0 + scale * (B @ A)`, compute row L2 norms `row_norms = np.maximum(np.linalg.norm(v_dir, axis=1, keepdims=True), eps)` "
            "(note: `np.linalg.norm` takes `(x, ord=None, axis=1, keepdims=True)` with NO `dtype` argument!), and return `(m[:, None] / row_norms) * v_dir`."
        )
        think = (
            "[L1] `v_dir = W0 + scale * (B @ A)` has shape `[d_out, d_in]`, where row `i` is output channel `i`.\n"
            "[L2] Compute `row_norms = np.maximum(np.linalg.norm(v_dir, axis=1, keepdims=True), eps)` (`[d_out, 1]`, never pass `dtype` to `np.linalg.norm` and never add `W0` again).\n"
            "[L3] Return `(m[:, None] / row_norms) * v_dir` so each row `i` has L2 norm `|m[i]|`."
        )
        code = (
            "import numpy as np\n\n"
            f"def {fn_name}(W0: np.ndarray, B: np.ndarray, A: np.ndarray, m: np.ndarray, scale: float, eps: float = 1e-{eps_e}) -> np.ndarray:\n"
            "    v_dir = W0 + scale * (B @ A)\n"
            "    row_norms = np.maximum(np.linalg.norm(v_dir, axis=1, keepdims=True), eps)\n"
            "    return (m[:, None] / row_norms) * v_dir"
        )
        test = (
            "import numpy as np\n"
            f"rng = np.random.default_rng({idx + 202})\n"
            f"W0 = rng.normal(size=({d_out}, {d_in}))\n"
            f"B, A = rng.normal(size=({d_out}, {r_dim})), rng.normal(size=({r_dim}, {d_in}))\n"
            f"m = np.abs(rng.normal(loc=2.0, scale=0.4, size=({d_out},))) + 0.2\n"
            f"Wf = {fn_name}(W0, B, A, m, scale=1.5)\n"
            f"assert Wf.shape == ({d_out}, {d_in})\n"
            "assert np.allclose(np.linalg.norm(Wf, axis=1), m, atol=1e-6)"
        )
        return {
            "pillar": "rsi_and_weight_surgery",
            "images": [],
            "messages": [
                {"role": "system", "content": sys_code},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"<think>{think}</think>\n```python\n{code}\n```"},
            ],
            "verifier": {"kind": "python_code", "test": test},
        }

    if variant == 2:
        fn_name = rng.choice(["ties_combine_task_vectors", "ties_disjoint_merge", "merge_ties_task_deltas"])
        kr = rng.choice([0.25, 0.4, 0.5, 0.6, 0.75])
        lam_v = rng.choice([0.8, 1.0, 1.2])
        prompt = (
            f"Implement `{fn_name}(W_base, deltas, keep_ratio={kr}, lam={lam_v})` in NumPy (`import numpy as np`):\n"
            "1. **Trim**: For each `d` in `deltas`, flatten `flat = d.reshape(-1)`, find the top `k = math.ceil(keep_ratio * flat.size)` indices by `|flat|` "
            "via `idx_top = np.argsort(-np.abs(flat), kind='stable')[:k]`, copy signed `buf[idx_top] = flat[idx_top]` into a zero buffer, and reshape back to `W_base.shape`.\n"
            "2. **Elect Sign**: Stack `stk = np.stack(trimmed, axis=0)` and compute `elected = np.sign(np.sum(stk, axis=0))`.\n"
            "3. **Disjoint Mean**: Average ONLY trimmed entries matching non-zero `elected` sign at each coordinate, and return `W_base + lam * mean_d`."
        )
        think = (
            "[L1] Flatten `flat = d.reshape(-1)`, create `buf = np.zeros_like(flat, dtype=np.float64)`, copy signed `buf[idx_top] = flat[idx_top]`, and reshape to `W_base.shape`.\n"
            "[L2] Stack `stk = np.stack(trimmed, axis=0)` (`[M, ...]`) and elect sign `elected = np.sign(np.sum(stk, axis=0))`.\n"
            "[L3] Mask `agree = (np.sign(stk) == elected[None, ...]) & (elected[None, ...] != 0)` and divide `np.sum(np.where(agree, stk, 0.0), axis=0)` by `np.maximum(np.sum(agree, axis=0), 1)`."
        )
        code = (
            "import math\nimport numpy as np\n\n"
            f"def {fn_name}(W_base: np.ndarray, deltas: list[np.ndarray], keep_ratio: float = {kr}, lam: float = {lam_v}) -> np.ndarray:\n"
            "    if not deltas:\n"
            "        return W_base.copy()\n"
            "    k = max(1, math.ceil(keep_ratio * W_base.size)) if keep_ratio > 0 else 0\n"
            "    trimmed = []\n"
            "    for d in deltas:\n"
            "        flat = d.reshape(-1)\n"
            "        buf = np.zeros_like(flat, dtype=np.float64)\n"
            "        if k > 0:\n"
            "            idx_top = np.argsort(-np.abs(flat), kind='stable')[:k]\n"
            "            buf[idx_top] = flat[idx_top]\n"
            "        trimmed.append(buf.reshape(W_base.shape))\n"
            "    stk = np.stack(trimmed, axis=0)\n"
            "    elected = np.sign(np.sum(stk, axis=0))\n"
            "    agree = (np.sign(stk) == elected[None, ...]) & (elected[None, ...] != 0)\n"
            "    mean_d = np.sum(np.where(agree, stk, 0.0), axis=0) / np.maximum(np.sum(agree, axis=0), 1)\n"
            "    return W_base + lam * mean_d"
        )
        v_a = float(rng.randint(3, 8))
        v_b = float(rng.randint(2, 6))
        test = (
            "import numpy as np\n"
            "W0 = np.zeros((2, 2))\n"
            f"d1 = np.array([[{v_a}, -4.0], [0.01, 0.02]])\n"
            f"d2 = np.array([[{v_b},  1.0], [-6.0, 0.01]])\n"
            f"res = {fn_name}(W0, [d1, d2], keep_ratio=0.5, lam=1.0)\n"
            f"assert res.shape == (2, 2) and np.allclose(res, np.array([[{(v_a+v_b)/2}, -4.0], [-6.0, 0.0]]))"
        )
        return {
            "pillar": "rsi_and_weight_surgery",
            "images": [],
            "messages": [
                {"role": "system", "content": sys_code},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"<think>{think}</think>\n```python\n{code}\n```"},
            ],
            "verifier": {"kind": "python_code", "test": test},
        }

    if variant == 3:
        # ROME Rank-1 Causal MLP Update (`np.linalg.solve` — NEVER Python `^-1` bitwise XOR!)
        fn_name = rng.choice(["rome_rank1_update", "apply_rome_mlp_patch", "causal_rank1_weight_edit"])
        reg = rng.choice([0.02, 0.04, 0.08, 0.12])
        d_m, d_f = rng.randint(6, 10), rng.randint(10, 16)
        prompt = (
            f"Implement `{fn_name}(W2, k_star, v_star, C_cov, reg={reg})` in NumPy (`import numpy as np`) "
            f"for a `[{d_m}, {d_f}]` MLP down-projection `W2` so `W2_new @ k_star == v_star`:\n"
            "- Regularize `c_reg = C_cov + reg * np.eye(C_cov.shape[0])`.\n"
            "- Solve `u = np.linalg.solve(c_reg, k_star)` (never use `^-1` in Python!) and return `W2 + np.outer(v_star - W2 @ k_star, u) / np.dot(k_star, u)`."
        )
        think = (
            "[L1] Constrained KKT rank-1 update solves `c_reg @ u = k_star` via `u = np.linalg.solve(c_reg, k_star)` (in Python `^` is bitwise XOR, never matrix inverse!).\n"
            "[L2] Residual `r = v_star - (W2 @ k_star)` (`[d_model]`) and denominator `denom = float(np.dot(k_star, u))` give `W2 + np.outer(r, u) / denom`.\n"
            "[L3] Verify `(W2 + np.outer(r, u) / (k_star @ u)) @ k_star == v_star`."
        )
        code = (
            "import numpy as np\n\n"
            f"def {fn_name}(W2: np.ndarray, k_star: np.ndarray, v_star: np.ndarray, C_cov: np.ndarray, reg: float = {reg}) -> np.ndarray:\n"
            "    c_reg = C_cov + reg * np.eye(C_cov.shape[0], dtype=C_cov.dtype)\n"
            "    u = np.linalg.solve(c_reg, k_star)\n"
            "    denom = float(np.dot(k_star, u))\n"
            "    return W2 + np.outer(v_star - (W2 @ k_star), u) / denom"
        )
        test = (
            "import numpy as np\n"
            f"rng = np.random.default_rng({idx + 404})\n"
            f"W2 = rng.normal(size=({d_m}, {d_f}))\n"
            f"K = rng.normal(size=(24, {d_f}))\n"
            "C = (K.T @ K) / 24.0\n"
            f"ks, vs = rng.normal(size=({d_f},)), rng.normal(size=({d_m},))\n"
            f"Wn = {fn_name}(W2, ks, vs, C)\n"
            "assert np.allclose(Wn @ ks, vs, atol=1e-9)"
        )
        return {
            "pillar": "rsi_and_weight_surgery",
            "images": [],
            "messages": [
                {"role": "system", "content": sys_code},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"<think>{think}</think>\n```python\n{code}\n```"},
            ],
            "verifier": {"kind": "python_code", "test": test},
        }

    if variant == 4:
        # Upper-Mid Monotonic Binary Search Bug Fix
        fn_name = rng.choice(["last_index_at_most", "rightmost_valid_slot", "max_feasible_tier", "find_upper_budget_index"])
        lim = rng.randint(6, 35)
        prompt = (
            f"Diagnose and fix the infinite-loop bug in `{fn_name}(arr: list[int], limit: int) -> int` "
            f"(e.g. on `arr=[{lim-2}, {lim+3}], limit={lim}`), which must return the largest index `i` in sorted `arr` with `arr[i] <= limit` (or `-1` if none):\n"
            "```python\n"
            f"def {fn_name}(arr: list[int], limit: int) -> int:\n"
            "    if not arr or arr[0] > limit:\n"
            "        return -1\n"
            "    lo, hi = 0, len(arr) - 1\n"
            "    while lo < hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        if arr[mid] <= limit:\n"
            "            lo = mid\n"
            "        else:\n"
            "            hi = mid - 1\n"
            "    return lo\n"
            "```"
        )
        think = (
            f"[L1] On 2-element interval `lo = 0, hi = 1` (`arr=[{lim-2}, {lim+3}], limit={lim}`), floor midpoint `mid = (0 + 1) // 2 = 0`.\n"
            "[L2] Since `arr[0] <= limit`, `lo = mid = 0` leaves `[lo, hi] = [0, 1]` unchanged forever -> infinite loop!\n"
            "[REBUILD] Bias midpoint upward: `mid = (lo + hi + 1) // 2` so `mid > lo` whenever `lo < hi`, ensuring strict interval contraction every step."
        )
        code = (
            f"def {fn_name}(arr: list[int], limit: int) -> int:\n"
            "    if not arr or arr[0] > limit:\n"
            "        return -1\n"
            "    lo, hi = 0, len(arr) - 1\n"
            "    while lo < hi:\n"
            "        mid = (lo + hi + 1) // 2\n"
            "        if arr[mid] <= limit:\n"
            "            lo = mid\n"
            "        else:\n"
            "            hi = mid - 1\n"
            "    return lo"
        )
        test = (
            f"assert {fn_name}([], {lim}) == -1\n"
            f"assert {fn_name}([{lim+2}, {lim+5}], {lim}) == -1\n"
            f"assert {fn_name}([{lim-2}, {lim+3}], {lim}) == 0\n"
            f"assert {fn_name}([{lim-3}, {lim}, {lim}, {lim+4}], {lim}) == 2"
        )
        return {
            "pillar": "diagnostic_debugging",
            "images": [],
            "messages": [
                {"role": "system", "content": sys_code},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"<think>{think}</think>\n```python\n{code}\n```"},
            ],
            "verifier": {"kind": "python_code", "test": test},
        }

    if variant == 5:
        # Numerically Stable Log-Softmax (Shift by `m = max(logits)` — NEVER `exp_max / sum_exp`!)
        fn_name = rng.choice(["compute_stable_log_softmax", "safe_log_softmax_vec", "shift_invariant_log_probs"])
        big_v = rng.choice([800.0, 950.0, 1100.0, 1250.0])
        prompt = (
            f"Write `{fn_name}(logits: list[float]) -> list[float]` using `import math` that computes exact `log(exp(x_i) / sum_j exp(x_j))` "
            f"without `OverflowError` on `[{big_v}, {big_v + 1.0}]` or `ZeroDivisionError` on `[{-big_v}, {-big_v + 1.0}]` (returning `[]` on empty input)."
        )
        think = (
            "[L1] Identity: `log(exp(x_i) / sum_j exp(x_j)) = x_i - (m + log(sum_j exp(x_j - m)))` where `m = max(logits)`.\n"
            "[L2] Subtracting `m = max(logits)` guarantees every `x_j - m <= 0` (`exp(x_j - m) in (0, 1]`, no overflow) and at least one term equals `exp(0) = 1.0` (sum >= 1.0, no underflow).\n"
            "[L3] Compute `lse = m + math.log(sum(math.exp(x - m) for x in logits))` and return `[x - lse for x in logits]`."
        )
        code = (
            "import math\n\n"
            f"def {fn_name}(logits: list[float]) -> list[float]:\n"
            "    if not logits:\n"
            "        return []\n"
            "    m = max(logits)\n"
            "    lse = m + math.log(sum(math.exp(x - m) for x in logits))\n"
            "    return [x - lse for x in logits]"
        )
        test = (
            f"assert {fn_name}([]) == []\n"
            f"r1 = {fn_name}([{big_v}, {big_v}])\n"
            "assert abs(r1[0] - (-0.69314718)) < 1e-5\n"
            f"r2 = {fn_name}([{-big_v}, {-big_v}])\n"
            "assert abs(r2[1] - (-0.69314718)) < 1e-5"
        )
        return {
            "pillar": "diagnostic_debugging",
            "images": [],
            "messages": [
                {"role": "system", "content": sys_code},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"<think>{think}</think>\n```python\n{code}\n```"},
            ],
            "verifier": {"kind": "python_code", "test": test},
        }

    if variant == 6:
        # Monotonic Deque Shortest Subarray >= K with Negative Numbers
        fn_name = rng.choice(["shortest_window_sum_at_least", "min_subarray_ge_target", "monotonic_shortest_sum"])
        targ_k = rng.randint(4, 25)
        prompt = (
            f"Implement `{fn_name}(nums: list[int], k: int) -> int` in O(n) time using `from collections import deque` "
            f"(e.g. for `k={targ_k}` when `nums` contains negative integers): return the minimum length of a non-empty contiguous subarray "
            "with sum >= `k`, or `-1` if none exists."
        )
        think = (
            "[L1] Build prefix sums `pref` of length `len(nums) + 1` so subarray `nums[i:j]` has sum `pref[j] - pref[i]`.\n"
            "[L2] Maintain a monotonic increasing `dq` of indices into `pref`: (1) while `dq and pj - pref[dq[0]] >= k`, update `best = min(best, j - dq.popleft())`; "
            "(2) while `dq and pref[dq[-1]] >= pj`, call `dq.pop()` because `j > dq[-1]` and `pj <= pref[dq[-1]]` dominates `dq[-1]`.\n"
            "[L3] Append `j` and return `best if best <= len(nums) else -1`."
        )
        code = (
            "from collections import deque\n\n"
            f"def {fn_name}(nums: list[int], k: int) -> int:\n"
            "    pref = [0]\n"
            "    for x in nums:\n"
            "        pref.append(pref[-1] + x)\n"
            "    dq: deque[int] = deque()\n"
            "    best = len(nums) + 1\n"
            "    for j, pj in enumerate(pref):\n"
            "        while dq and pj - pref[dq[0]] >= k:\n"
            "            best = min(best, j - dq.popleft())\n"
            "        while dq and pref[dq[-1]] >= pj:\n"
            "            dq.pop()\n"
            "        dq.append(j)\n"
            "    return best if best <= len(nums) else -1"
        )
        test = (
            f"assert {fn_name}([{targ_k - 1}, -2, {targ_k}], {targ_k}) == 1\n"
            f"assert {fn_name}([2, -1, 2], 3) == 3\n"
            f"assert {fn_name}([1, -3, 1], 4) == -1"
        )
        return {
            "pillar": "diagnostic_debugging",
            "images": [],
            "messages": [
                {"role": "system", "content": sys_code},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"<think>{think}</think>\n```python\n{code}\n```"},
            ],
            "verifier": {"kind": "python_code", "test": test},
        }

    if variant == 7:
        # Late-Binding Closure + Mutable Default Argument Double Bug
        fn_name = rng.choice(["compile_shift_callbacks", "make_delta_functions", "create_bias_closures"])
        sample_off = rng.randint(3, 19)
        buggy = (
            f"def {fn_name}(shifts: list[int], audit_log: list[int] = []):\n"
            "    ops = [lambda x: x + s for s in shifts]\n"
            "    audit_log.append(len(shifts))\n"
            "    return ops, audit_log"
        )
        fixed = (
            f"def {fn_name}(shifts: list[int], audit_log: list[int] | None = None):\n"
            "    log = [] if audit_log is None else audit_log\n"
            "    ops = [lambda x, s=s: x + s for s in shifts]\n"
            "    log.append(len(shifts))\n"
            "    return ops, log"
        )
        prompt = (
            f"Fix BOTH bugs (mutable default argument + late-binding closure variable, e.g. when `shifts=[1, {sample_off}]`) "
            f"in `{fn_name}` and output ONLY the corrected ```python block:\n\n```python\n{buggy}\n```"
        )
        think = (
            "[L1] Bug 1: mutable default `audit_log: list[int] = []` leaks state across independent calls when omitted -> replace with `None` sentinel.\n"
            "[L2] Bug 2: `lambda x: x + s` captures loop variable `s` by late reference so all lambdas add `shifts[-1]` -> bind default `lambda x, s=s: x + s` (NOT `shifts[0]`, bind `s=s`!).\n"
            "[REBUILD] Instantiate `log = [] if audit_log is None else audit_log` and bind `s=s` per iteration."
        )
        test = (
            f"f1, l1 = {fn_name}([1, {sample_off}])\n"
            f"f2, l2 = {fn_name}([5])\n"
            f"assert [f(10) for f in f1] == [11, {10 + sample_off}]\n"
            "assert l1 == [2] and l2 == [1]"
        )
        return {
            "pillar": "diagnostic_debugging",
            "images": [],
            "messages": [
                {"role": "system", "content": sys_code},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"<think>{think}</think>\n```python\n{fixed}\n```"},
            ],
            "verifier": {"kind": "python_code", "test": test},
        }

    if variant == 8:
        # SVF Singular-Value Subspace Weight Rewrite Function
        fn_name = rng.choice(["svf_modulate_weight", "apply_svf_singular_gains", "sakana_svf_step"])
        r_rows, r_cols = rng.randint(5, 9), rng.randint(7, 12)
        prompt = (
            f"Implement `{fn_name}(W: np.ndarray, z_gains: np.ndarray) -> np.ndarray` in NumPy (`import numpy as np`) "
            f"(tested on `[{r_rows}, {r_cols}]` matrix `W`) for Singular Value Fine-Tuning (SVF):\n"
            "- Compute thin SVD `u, s, vh = np.linalg.svd(W, full_matrices=False)`.\n"
            "- Scale `s_new = s * (1.0 + z_gains)` and return `(u * s_new[None, :]) @ vh`."
        )
        think = (
            "[L1] Thin SVD gives `W == (u * s[None, :]) @ vh`.\n"
            "[L2] Modulating singular values by `s_new = s * (1.0 + z_gains)` preserves exact identity at `z_gains = 0`.\n"
            "[L3] Return `(u * s_new[None, :]) @ vh`."
        )
        code = (
            "import numpy as np\n\n"
            f"def {fn_name}(W: np.ndarray, z_gains: np.ndarray) -> np.ndarray:\n"
            "    u, s, vh = np.linalg.svd(W, full_matrices=False)\n"
            "    s_new = s * (1.0 + z_gains)\n"
            "    return (u * s_new[None, :]) @ vh"
        )
        test = (
            "import numpy as np\n"
            f"rng = np.random.default_rng({idx + 303})\n"
            f"W = rng.normal(size=({r_rows}, {r_cols}))\n"
            f"k = min({r_rows}, {r_cols})\n"
            f"assert np.allclose({fn_name}(W, np.zeros(k)), W, atol=1e-10)\n"
            f"assert np.allclose({fn_name}(W, np.ones(k)), 2.0 * W, atol=1e-10)"
        )
        return {
            "pillar": "rsi_and_weight_surgery",
            "images": [],
            "messages": [
                {"role": "system", "content": sys_code},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"<think>{think}</think>\n```python\n{code}\n```"},
            ],
            "verifier": {"kind": "python_code", "test": test},
        }

    # variant == 9: Parameterized Harmonic-Mean Round-Trip Rate Trap (distinct speeds `(v1, v2)` with integer harmonic mean!)
    v1, v2, hm = rng.choice([(30, 60, 40), (20, 30, 24), (60, 120, 80), (45, 90, 60), (24, 72, 36), (15, 30, 20)])
    d_leg = v1 * v2
    t1, t2 = d_leg // v1, d_leg // v2
    naive = (v1 + v2) // 2
    prompt = (
        f"A courier drone flies from Depot A to Depot B at `{v1} km/h` and returns along the exact same route from Depot B to Depot A at `{v2} km/h`. "
        f"An engineer claims the round-trip average speed is `({v1} + {v2}) / 2 = {naive} km/h`. "
        "Audit this claim in `<think>` and output raw JSON `{\"naive_mean_wrong\": <bool>, \"exact_avg_kmh\": <int>}`."
    )
    think = (
        f"[L1] Average speed equals `(total distance) / (total time)`, NOT the arithmetic mean of leg speeds, because equal distances take UNEQUAL times.\n"
        f"[L2] Pick convenient one-way distance `D = {v1} * {v2} = {d_leg} km`: outbound takes `{d_leg} / {v1} = {t1} h`, return takes `{d_leg} / {v2} = {t2} h`.\n"
        f"[AUDIT] Total distance = `2 * {d_leg} = {2 * d_leg} km` (NEVER `{d_leg} + ...`), total time = `{t1} + {t2} = {t1 + t2} h`.\n"
        f"[REBUILD] Exact harmonic mean speed = `{2 * d_leg} / {t1 + t2} = {hm} km/h` != `{naive} km/h`."
    )
    exp_hm = {"naive_mean_wrong": True, "exact_avg_kmh": hm}
    return {
        "pillar": "diagnostic_debugging",
        "images": [],
        "messages": [
            {"role": "system", "content": "Audit intuitive math traps from definitions in <think>, then output raw JSON only."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"<think>{think}</think>\n{json.dumps(exp_hm)}"},
        ],
        "expected": exp_hm,
    }


# ---------------------------------------------------------------------------
# Hybrid SEAL + SVF Self-Weight-Rewriting SFT Demonstrations
# ---------------------------------------------------------------------------


def make_v6_seal_svf_demo_item(rng: random.Random, idx: int) -> dict[str, Any]:
    """Generate verified `<weight_rewrite>` training demonstrations teaching autonomous weight rewriting."""
    scenario = idx % 4

    if scenario == 0:
        # Diagnose Python/NumPy SVD & Binary-Search Bugs -> Emit Code-Targeted SEAL+SVF Rewrite
        ex_code = make_v6_parameterized_code_item(rng, idx * 2)  # SVD or Binary Search
        u_prompt = ex_code["messages"][1]["content"]
        gold_resp = ex_code["messages"][2]["content"]
        diag_msg = (
            "Dev Diagnostic Batch (`2` tasks):\n"
            "- `diag_1`: Failed unit test (`AssertionError` / `ValueError: shapes not aligned` in NumPy SVD subspace projection; "
            "model unpacked `s, u, vt = np.linalg.svd(...)` instead of `u, s, vh`).\n"
            "- `diag_2`: `TimeoutExpired` on rightmost monotonic binary search (`mid = (lo + hi) // 2` stuck on 2-element interval `[0, 1]`).\n"
            "Current dev pass rate: `0/2` (`0.0%`). Anchor NLL (`Vision/Tools/Research`): `0.62`.\n"
            "Propose a surgical `<weight_rewrite>` plan for `theta_copy`."
        )
        plan = {
            "diagnosis": "NumPy SVD tuple-order unpacking error (U,S,Vh) and floor-midpoint infinite loop in upper-bound binary search.",
            "action": "rewrite",
            "target_group": "upper_full_attn_mlp",
            "svf_gains": [round(rng.uniform(0.35, 0.55), 2), 0.10, 0.10, 0.25],
            "lr_b": 7.5e-5,
            "lr_a": 2.2e-5,
            "steps": 3,
            "nullspace_alpha": 0.88,
            "wise_ft_gamma": 0.90,
            "self_edits": [
                {"prompt": u_prompt, "response": gold_resp},
            ],
        }
        think = (
            "[DIAGNOSE] Both dev failures stem from Python/NumPy algorithmic invariant bugs: unpacking `np.linalg.svd` in wrong order "
            "and floor-midpoint stagnation on `lo = mid`.\n"
            "[SUBSPACE] Boost `svf_gains[0]` (code precision) and `svf_gains[3]` (weight-surgery algebra) on `upper_full_attn_mlp` with "
            "`nullspace_alpha=0.88` and `wise_ft_gamma=0.90` so Vision/Tools/Research anchor NLL does not drift.\n"
            "[SELF-EDIT] Synthesize a verified NumPy SVD null-space self-edit with explicit `(U, S, Vh)` check for 3 LoRA+ steps."
        )
        return {
            "pillar": "seal_self_rewrite",
            "images": [],
            "messages": [
                {"role": "system", "content": SEAL_SYSTEM_PROMPT},
                {"role": "user", "content": diag_msg},
                {"role": "assistant", "content": f"<think>{think}</think>\n{json.dumps(plan)}"},
            ],
        }

    if scenario == 1:
        # Diagnose Unassisted Multi-Clause Arithmetic & 2-Turn RPR Slip -> Emit Math/RPR SEAL+SVF Rewrite
        ex_math = make_v5_clause_witness_rpr_item(rng, idx * 3 + 1)
        # Flatten 2-turn into concise Turn-2 belief-revision self-edit
        t2_prompt = ex_math["messages"][1]["content"] + "\n\nUpdate:\n" + ex_math["messages"][3]["content"]
        t2_resp = ex_math["messages"][4]["content"]
        diag_msg = (
            "Dev Diagnostic Batch (`2` tasks):\n"
            "- `diag_1`: Missed `preserved` integer by dropping 3rd subtraction clause in unassisted single-pass mode.\n"
            "- `diag_2`: Correct `<think>` option check on premise switch, but echoed Turn-1 option ID in final JSON.\n"
            "Current dev pass rate: `0/2` (`0.0%`). Anchor NLL: `0.62`.\n"
            "Propose a surgical `<weight_rewrite>` plan for `theta_copy`."
        )
        plan = {
            "diagnosis": "Multi-clause ledger omission and Turn-2 readout pointer slipback to invalidated Turn-1 option.",
            "action": "rewrite",
            "target_group": "upper_full_attn_mlp",
            "svf_gains": [0.05, round(rng.uniform(0.30, 0.50), 2), round(rng.uniform(0.35, 0.55), 2), 0.10],
            "lr_b": 7.0e-5,
            "lr_a": 2.0e-5,
            "steps": 3,
            "nullspace_alpha": 0.90,
            "wise_ft_gamma": 0.92,
            "self_edits": [
                {"prompt": t2_prompt, "response": t2_resp},
            ],
        }
        think = (
            "[DIAGNOSE] Failures show missing `[CLAUSES]` extraction witness and missing `[WINNER]` post-audit readout check.\n"
            "[SUBSPACE] Activate `svf_gains[1]` (clause math) and `svf_gains[2]` (RPR belief revision) on `upper_full_attn_mlp` "
            "with `nullspace_alpha=0.90` to keep anchor capabilities orthogonal.\n"
            "[SELF-EDIT] Provide a verified `[AUDIT] -> [PRESERVE] -> [REBUILD] -> [WINNER]` self-edit for 3 conservative LoRA+ steps."
        )
        return {
            "pillar": "seal_self_rewrite",
            "images": [],
            "messages": [
                {"role": "system", "content": SEAL_SYSTEM_PROMPT},
                {"role": "user", "content": diag_msg},
                {"role": "assistant", "content": f"<think>{think}</think>\n{json.dumps(plan)}"},
            ],
        }

    if scenario == 2:
        # Diagnose RSI Weight-Surgery / DoRA / TIES Precision Failure
        ex_dora = make_v6_parameterized_code_item(rng, idx * 6 + 1)
        diag_msg = (
            "Dev Diagnostic Batch (`2` tasks):\n"
            "- `diag_1`: `fuse_dora_linear` normalized columns instead of output neuron rows (`axis=1`) and added `W0` twice.\n"
            "- `diag_2`: `ties_merge_deltas` took unsigned magnitudes `abs(d)` instead of signed top-k values.\n"
            "Current dev pass rate: `0/2` (`0.0%`). Anchor NLL: `0.62`.\n"
            "Propose a surgical `<weight_rewrite>` plan for `theta_copy`."
        )
        plan = {
            "diagnosis": "Output-row normalization axis error in DoRA fusion and sign loss during TIES top-k trimming.",
            "action": "rewrite",
            "target_group": "upper_full_attn_mlp",
            "svf_gains": [0.40, 0.10, 0.05, 0.45],
            "lr_b": 8.0e-5,
            "lr_a": 2.4e-5,
            "steps": 4,
            "nullspace_alpha": 0.85,
            "wise_ft_gamma": 0.90,
            "self_edits": [
                {"prompt": ex_dora["messages"][1]["content"], "response": ex_dora["messages"][2]["content"]},
            ],
        }
        think = (
            "[DIAGNOSE] Both errors are linear-algebra shape/sign invariants in weight-merging implementations.\n"
            "[SUBSPACE] Elevate `svf_gains[0]` (code) and `svf_gains[3]` (weight surgery) with `nullspace_alpha=0.85`.\n"
            "[SELF-EDIT] Synthesize an exact row-normalized `(m[:, None] / row_norms) * V` DoRA implementation."
        )
        return {
            "pillar": "seal_self_rewrite",
            "images": [],
            "messages": [
                {"role": "system", "content": SEAL_SYSTEM_PROMPT},
                {"role": "user", "content": diag_msg},
                {"role": "assistant", "content": f"<think>{think}</think>\n{json.dumps(plan)}"},
            ],
        }

    # scenario == 3: Healthy Copy (`2/2 = 100%` pass!) -> Teach Calibrated `noop` Abstention from Over-Editing!
    diag_msg = (
        "Dev Diagnostic Batch (`2` tasks):\n"
        "- `diag_1`: PASS (`correct=True`, `think_tokens=118`, `echo=0.00`).\n"
        "- `diag_2`: PASS (`correct=True`, `think_tokens=94`, `echo=0.00`).\n"
        "Current dev pass rate: `2/2` (`100.0%`). Anchor NLL: `0.61`.\n"
        "Propose a surgical `<weight_rewrite>` plan for `theta_copy`."
    )
    plan = {
        "diagnosis": "All diagnostic tasks already pass with concise closed thinking and healthy anchor NLL; unnecessary weight updates risk drift.",
        "action": "noop",
        "target_group": "upper_full_attn_mlp",
        "svf_gains": [0.0, 0.0, 0.0, 0.0],
        "lr_b": 0.0,
        "lr_a": 0.0,
        "steps": 0,
        "nullspace_alpha": 1.0,
        "wise_ft_gamma": 1.0,
        "self_edits": [],
    }
    think = (
        "[DIAGNOSE] `theta_copy` already achieves `100.0%` (`2/2`) with zero formatting or echo defects.\n"
        "[SUBSPACE] By RSI non-regression invariant, applying non-zero gradient updates when error is zero can only increase KL/NLL drift.\n"
        "[SELF-EDIT] Emit `action='noop'` with zero `svf_gains` and `steps=0`."
    )
    return {
        "pillar": "seal_self_rewrite",
        "images": [],
        "messages": [
            {"role": "system", "content": SEAL_SYSTEM_PROMPT},
            {"role": "user", "content": diag_msg},
            {"role": "assistant", "content": f"<think>{think}</think>\n{json.dumps(plan)}"},
        ],
    }


# ---------------------------------------------------------------------------
# Strict 3-Split SEAL + SVF Self-Rewrite RL & Out-of-Sample Evaluation Builder
# ---------------------------------------------------------------------------


def _as_eval_task(raw_ex: dict[str, Any], tid: str) -> dict[str, Any]:
    msgs = _strip_prior_think(raw_ex["messages"][:-1])
    if "verifier" in raw_ex:
        return {
            "id": tid,
            "pillar": raw_ex["pillar"],
            "kind": "python_code",
            "images": [],
            "messages": msgs,
            "reference_answer": raw_ex["messages"][-1]["content"],
            "verifier": {**raw_ex["verifier"], "forbid_tool_call": True},
        }
    exp = raw_ex["expected"]
    if isinstance(exp, dict) and {"primary", "preserved"}.issubset(exp.keys()):
        return {
            "id": tid,
            "pillar": raw_ex["pillar"],
            "kind": "math_opt",
            "images": [],
            "messages": msgs,
            "reference_answer": raw_ex["messages"][-1]["content"],
            "verifier": {"primary": exp["primary"], "preserved": exp["preserved"], "forbid_tool_call": True},
        }
    return {
        "id": tid,
        "pillar": raw_ex["pillar"],
        "kind": "json_values",
        "images": [],
        "messages": msgs,
        "reference_answer": raw_ex["messages"][-1]["content"],
        "verifier": {"expected": exp, "forbid_tool_call": True},
    }


def build_three_split_seal_suite(seed: int) -> dict[str, Any]:
    """Build strictly disjoint `Diagnose` != `Reward` != `Final Test` suites for self-weight-rewriting RL."""
    rng = random.Random(seed)
    rl_episodes: list[dict[str, Any]] = []

    for ep_i in range(16):
        fam = ep_i % 4
        diag_tasks = []
        rew_tasks = []
        if fam == 0:
            # SVD / DoRA / TIES Weight-Surgery Python Code family
            for j in range(2):
                diag_tasks.append(_as_eval_task(make_v6_parameterized_code_item(rng, j), f"ep{ep_i}_diag_{j}"))
            for j in range(3):
                rew_tasks.append(_as_eval_task(make_v6_parameterized_code_item(rng, j), f"ep{ep_i}_rew_{j}"))
        elif fam == 1:
            # Algorithmic Binary-Search / Debug Python Code family
            for j in range(2):
                diag_tasks.append(_as_eval_task(make_v6_parameterized_code_item(rng, 3 + j * 2), f"ep{ep_i}_diag_{j}"))
            for j in range(3):
                rew_tasks.append(_as_eval_task(make_v6_parameterized_code_item(rng, 3 + j * 2), f"ep{ep_i}_rew_{j}"))
        elif fam == 2:
            # 2-Turn RPR + Clause-Witness Math family
            for j in range(2):
                diag_tasks.append(_as_eval_task(make_v5_clause_witness_rpr_item(rng, j * 3 + 1), f"ep{ep_i}_diag_{j}"))
            for j in range(3):
                rew_tasks.append(_as_eval_task(make_v5_clause_witness_rpr_item(rng, j), f"ep{ep_i}_rew_{j}"))
        else:
            # Mixed RSI Math + Research Statistical Rigor family
            diag_tasks.append(_as_eval_task(make_v5_rsi_weight_surgery_item(rng, 4 + (ep_i % 3)), f"ep{ep_i}_diag_0"))
            diag_tasks.append(_as_eval_task(make_v5_research_ablation_item(rng, ep_i), f"ep{ep_i}_diag_1"))
            rew_tasks.append(_as_eval_task(make_v5_rsi_weight_surgery_item(rng, 5), f"ep{ep_i}_rew_0"))
            rew_tasks.append(_as_eval_task(make_v5_research_ablation_item(rng, ep_i + 1), f"ep{ep_i}_rew_1"))
            rew_tasks.append(_as_eval_task(make_v6_parameterized_code_item(rng, 4), f"ep{ep_i}_rew_2"))

        rl_episodes.append({
            "episode_id": f"seal_ep_{ep_i+1:02d}",
            "family": ["weight_surgery_code", "algo_debug_code", "rpr_clause_math", "rsi_research_mix"][fam],
            "diagnose_tasks": diag_tasks,
            "reward_tasks": rew_tasks,
        })

    # Disjoint Dev-Diagnose batch (4 tasks) & Dev-Gate batch (8 tasks) for final autonomous self-rewrite test
    rng_test = random.Random(seed + 777)
    dev_diagnose_4 = [
        _as_eval_task(make_v6_parameterized_code_item(rng_test, 0), "final_dev_diag_01"),
        _as_eval_task(make_v6_parameterized_code_item(rng_test, 3), "final_dev_diag_02"),
        _as_eval_task(make_v5_clause_witness_rpr_item(rng_test, 1), "final_dev_diag_03"),
        _as_eval_task(make_v6_parameterized_code_item(rng_test, 2), "final_dev_diag_04"),
    ]
    dev_gate_8 = [
        _as_eval_task(make_v6_parameterized_code_item(rng_test, k), f"final_dev_gate_{k+1:02d}")
        for k in range(4)
    ] + [
        _as_eval_task(make_v5_clause_witness_rpr_item(rng_test, k), f"final_dev_gate_{k+5:02d}")
        for k in range(4)
    ]

    # Completely Untouched 24-Task Final Self-Rewrite Test Suite (6 per hard pillar!)
    rng_final = random.Random(seed + 9999)
    final_test_24: list[dict[str, Any]] = []
    for i in range(6):
        final_test_24.append(_as_eval_task(make_v6_parameterized_code_item(rng_final, i % 5), f"sr_test_wcode_{i+1:02d}"))
    for i in range(6):
        final_test_24.append(_as_eval_task(make_v6_parameterized_code_item(rng_final, 3 if i % 2 == 0 else i + 5), f"sr_test_debug_{i+1:02d}"))
    for i in range(6):
        final_test_24.append(_as_eval_task(make_v5_clause_witness_rpr_item(rng_final, i * 3 + 1), f"sr_test_rpr2t_{i+1:02d}"))
    for i in range(6):
        final_test_24.append(_as_eval_task(make_v5_clause_witness_rpr_item(rng_final, i * 3), f"sr_test_math_{i+1:02d}"))

    return {
        "rl_episodes": rl_episodes,
        "dev_diagnose_4": dev_diagnose_4,
        "dev_gate_8": dev_gate_8,
        "final_test_24": final_test_24,
    }


def main() -> None:
    V6_CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    rng_tr = random.Random(20261301)
    rng_va = random.Random(20261302)
    rng_rl = random.Random(20261303)

    seal_suite = build_three_split_seal_suite(seed=20261350)

    # Collect held-out user prompt fingerprints (`heldout_v5_rsi_56` + `final_test_24`) for strict zero-overlap guard
    forbid_user_prompts: set[str] = set()
    heldout_56_path = V5_CORPUS_DIR / "heldout_v5_rsi_56.jsonl"
    if heldout_56_path.exists():
        for line in heldout_56_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)
                for m in obj.get("messages", []):
                    if m.get("role") == "user":
                        forbid_user_prompts.add(str(m.get("content", "")).strip())
    for t_ex in seal_suite["final_test_24"]:
        for m in t_ex.get("messages", []):
            if m.get("role") == "user":
                forbid_user_prompts.add(str(m.get("content", "")).strip())

    def _is_eval_free(row: dict[str, Any]) -> bool:
        return not any(
            m.get("role") == "user" and str(m.get("content", "")).strip() in forbid_user_prompts
            for m in row.get("messages", [])
        )

    # Load stratified multi-pillar replay from `train_rpr_v5.jsonl` so NO v4/v5 capability (Audit, RPR-L1/2/3, IFEval, Frontier) can fade!
    v5_all = [
        json.loads(l)
        for l in (V5_CORPUS_DIR / "train_rpr_v5.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    v5_by_pil: dict[str, list[dict[str, Any]]] = {}
    for r in v5_all:
        if _is_eval_free(r):
            v5_by_pil.setdefault(str(r.get("pillar", "misc")), []).append(r)
    for plist in v5_by_pil.values():
        rng_tr.shuffle(plist)

    replay_quotas = {
        "rpr_controller_audit": 20,
        "rpr_controller_l1": 16,
        "rpr_controller_l2": 16,
        "rpr_controller_l3": 16,
        "schema_lines": 16,
        "schema_json": 16,
        "diagnostic_debug": 15,
        "diagnostic_trap_math": 16,
        "diagnostic_debug_2turn": 4,
        "clause_witness_and_debug": 20,
        "frontier_systems_and_debugging": 28,
        "frontier_math_and_logic_rpr": 28,
        "frontier_ai_and_self_improvement": 28,
        "ai_self_improvement": 16,
        "ai_weights_mechanics": 15,
        "ai_weights_code": 4,
        "ai_self_improvement_code": 4,
        "internalized_rpr_2turn": 24,
        "standalone_rpr_2turn": 44,
        "clause_witness_math": 32,
        "rsi_and_weight_surgery": 36,
        "research_and_ablation": 36,
        "multistep_tool_use": 40,
        "tool_positive": 12,
        "tool_recovery": 8,
        "vision_multimodal": 48,
    }
    train_rows: list[dict[str, Any]] = []
    for pil_k, quota in replay_quotas.items():
        train_rows.extend(v5_by_pil.get(pil_k, [])[:quota])

    # Add diverse parameterized v6 code/debug/RSI, clause-witness math, and `<weight_rewrite>` SEAL+SVF demonstrations
    for i in range(220):
        cand = make_v6_parameterized_code_item(rng_tr, i)
        if _is_eval_free(cand):
            train_rows.append(cand)
    for i in range(48):
        cand = make_v5_clause_witness_rpr_item(rng_tr, i)
        if _is_eval_free(cand):
            train_rows.append(cand)
    for i in range(52):
        cand = make_v6_seal_svf_demo_item(rng_tr, i)
        if _is_eval_free(cand):
            train_rows.append(cand)

    train_rows = [r for r in _deduplicate_rows(train_rows) if _is_eval_free(r)]
    rng_tr.shuffle(train_rows)
    for idx, r in enumerate(train_rows, start=1):
        r["id"] = f"train_v6-{idx:04d}"
        r.setdefault("images", [])

    val_rows: list[dict[str, Any]] = []
    for i in range(24):
        val_rows.append(make_v6_parameterized_code_item(rng_va, i))
    for i in range(12):
        val_rows.append(make_v5_clause_witness_rpr_item(rng_va, i))
    for i in range(10):
        val_rows.append(make_v6_seal_svf_demo_item(rng_va, i))
    for i in range(8):
        val_rows.append(make_v5_research_ablation_item(rng_va, i))
    for i in range(8):
        val_rows.append(make_v5_multistep_tool_item(rng_va, i))
    for i in range(10):
        val_rows.append(make_v5_vision_item(rng_va, i + 900, "val_v6"))
    val_rows.extend(v5_by_pil.get("rpr_controller_audit", [])[-4:])
    val_rows = [r for r in _deduplicate_rows(val_rows) if _is_eval_free(r)]
    for idx, r in enumerate(val_rows, start=1):
        r["id"] = f"val_v6-{idx:04d}"
        r.setdefault("images", [])

    rlvr_64: list[dict[str, Any]] = []
    for i in range(36):
        ex = make_v6_parameterized_code_item(rng_rl, i)
        t = _as_eval_task(ex, f"v6-rlvr-code-{i+1:02d}")
        if _is_eval_free(t):
            rlvr_64.append(t)
    for i in range(16):
        ex = make_v5_clause_witness_rpr_item(rng_rl, i)
        t = _as_eval_task(ex, f"v6-rlvr-math-{i+1:02d}")
        if _is_eval_free(t):
            rlvr_64.append(t)
    for i in range(12):
        ex = make_v5_rsi_weight_surgery_item(rng_rl, 4 + (i % 3)) if i < 6 else make_v5_research_ablation_item(rng_rl, i)
        t = _as_eval_task(ex, f"v6-rlvr-rsi-{i+1:02d}")
        if _is_eval_free(t):
            rlvr_64.append(t)
    rng_rl.shuffle(rlvr_64)

    n_code = _verify_executable_code_rows(train_rows + val_rows)
    all_check = (
        train_rows
        + val_rows
        + rlvr_64
        + seal_suite["dev_diagnose_4"]
        + seal_suite["dev_gate_8"]
        + seal_suite["final_test_24"]
    )
    assert_decontaminated(all_check)

    for fname, rows in [
        ("train_v6_repair.jsonl", train_rows),
        ("val_v6_repair.jsonl", val_rows),
        ("rlvr_v6_repair_64.jsonl", rlvr_64),
    ]:
        (V6_CORPUS_DIR / fname).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    (V6_CORPUS_DIR / "seal_svf_episodes.json").write_text(json.dumps(seal_suite, indent=2) + "\n", encoding="utf-8")

    n_turns = sum(sum(1 for m in r["messages"] if m["role"] == "assistant") for r in train_rows)
    print(
        f"[RPR v6.1 Balanced Anti-Overfit + SEAL/SVF Corpus Built] {V6_CORPUS_DIR}:\n"
        f"  train_v6_repair   = {len(train_rows)} unique conversations ({n_turns} turns, 0 eval overlap)\n"
        f"  val_v6_repair     = {len(val_rows)} conversations\n"
        f"  rlvr_v6_repair_64 = {len(rlvr_64)} shuffled code+math+RSI RLVR episodes\n"
        f"  seal_svf_episodes = {len(seal_suite['rl_episodes'])} 3-split RL episodes + 24 unseen self-rewrite test tasks\n"
        f"  Python code self-tests passed = {n_code} (100%), zero contamination verified!",
        flush=True,
    )


if __name__ == "__main__":
    main()
