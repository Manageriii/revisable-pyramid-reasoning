"""Build the Master RPR v9 Corpus (`2.91B` Pure 16-bit `bfloat16`):
Deep Model Weight Modification + Zero-Prior-Knowledge (`OOD`) First-Principles Discovery +
Algorithmic Base-Case Invariant Tracing + Frontier Research + Cline/Hermes Native & RPR/SEAL MCP Tools.

Outputs inside `training/data/rpr_v9_seal_corpus/`:
  1. `train_v9_sft.jsonl`          : ~920 deduplicated, execution-verified SFT conversations
  2. `orpo_v9_pairs.jsonl`         : 200 contrastive `(messages, chosen, rejected)` pairs targeting exact
                                     SVD row-slice geometry, TIES sign consensus, binary-search/DFS/deque
                                     invariants, zero-prior epistemic humility, and RPR distraction immunity
  3. `rlvr_v9_episodes_64.jsonl`   : 64 verifiable RL episodes (`python3 -I` unit tests + OOD formal systems + RPR)
  4. `val_v9_joint.jsonl`          : 96 stratified validation conversations
  5. `heldout_zero_prior_24.jsonl` : 24 strictly held-out Zero-Prior-Knowledge, Weight-Surgery & Research tasks
                                     (0 6-gram overlap with training; completely disjoint symbols & function names)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
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
from training.build_v8_hermes_corpus import V8_CORPUS_DIR


V9_CORPUS_DIR = Path(__file__).resolve().parent / "data" / "rpr_v9_seal_corpus"


# ===========================================================================
# 1. Held-Out 24-Task Zero-Prior-Knowledge (`OOD`) + Weight-Surgery Benchmark
#    (Built FIRST so `assert_decontaminated` enforces 0 overlap against it!)
# ===========================================================================


def build_heldout_zero_prior_24() -> list[dict[str, Any]]:
    """24 unseen tasks testing:
      - `zero_prior_formal` (8): invented algebras, custom stack/register VMs, novel rewrite systems
      - `weight_mod_rsi`    (8): unseen weight-surgery code & numerical precision derivations
      - `research_and_ood_discovery` (8): zero-prior tool discovery, epistemic abstention, research stats, & tricky algos
    """
    tasks: list[dict[str, Any]] = []

    # --- Pillar A: 8 Unseen Zero-Prior Formal Systems (completely invented rules!) ---
    for i, (a, b, p, q, c0) in enumerate([(3, 5, 2, 3, 4), (4, 2, 3, 2, 5)], start=1):
        # Define a brand-new non-associative operator `x ⋈ y = p*x*y - q*x + y + c0`
        ab = p * a * b - q * a + b + c0
        ba = p * b * a - q * b + a + c0
        lhs = p * ab * 2 - q * ab + 2 + c0
        tasks.append({
            "id": f"zp9-formal-0{i}",
            "pillar": "zero_prior_formal",
            "kind": "json_values",
            "messages": [{
                "role": "user",
                "content": (
                    f"Consider an unfamiliar algebraic magma `(Z, ⋈)` defined strictly by the rule:\n"
                    f"  `x ⋈ y = {p}*x*y - {q}*x + y + {c0}`\n"
                    f"Do NOT assume commutativity or associativity. Compute:\n"
                    f"1. `u = {a} ⋈ {b}`\n"
                    f"2. `v = {b} ⋈ {a}`\n"
                    f"3. `chained = ({a} ⋈ {b}) ⋈ 2`\n"
                    f"4. `is_commutative_pair = (u == v)`\n"
                    f'Return ONLY JSON `{{"u": <int>, "v": <int>, "chained": <int>, "is_commutative_pair": <bool>}}`.'
                ),
            }],
            "verifier": {"expected": {"u": ab, "v": ba, "chained": lhs, "is_commutative_pair": (ab == ba)}, "tol": 1e-6, "forbid_tool_call": True},
        })

    # Custom unseen 2-register bytecode VM (`R0`, `R1`)
    for idx_vm, (init0, init1, k_add) in enumerate([(7, 3, 5), (9, 4, 6)], start=3):
        r0, r1 = init0, init1
        # Trace: MIRROR (`R0, R1 = R0 + R1, R0 - R1`), FOLD(k_add) (`R0 = R0 * 2 - R1 + k_add`), CLAMP0 (`if R1 < 0: R1 = -R1`)
        r0, r1 = r0 + r1, r0 - r1
        r0 = r0 * 2 - r1 + k_add
        if r1 < 0:
            r1 = -r1
        r0, r1 = r0 + r1, r0 - r1
        tasks.append({
            "id": f"zp9-formal-0{idx_vm}",
            "pillar": "zero_prior_formal",
            "kind": "json_values",
            "messages": [{
                "role": "user",
                "content": (
                    f"A custom hardware micro-core has two integer registers `(R0, R1)` initialized to `({init0}, {init1})` and executes three custom opcodes:\n"
                    f"- `MIRROR`: simultaneously updates `(R0, R1) <- (R0 + R1, R0 - R1)` using pre-instruction values.\n"
                    f"- `SKEW c`: updates `R0 <- 2 * R0 - R1 + c` (leaving `R1` unchanged).\n"
                    f"- `RECT`: if `R1 < 0`, sets `R1 <- -R1`; otherwise leaves `R1` unchanged.\n"
                    f"Execute the 4-instruction program: `MIRROR ; SKEW {k_add} ; RECT ; MIRROR`.\n"
                    f'Return ONLY JSON `{{"R0": <int>, "R1": <int>}}`.'
                ),
            }],
            "verifier": {"expected": {"R0": r0, "R1": r1}, "tol": 1e-6, "forbid_tool_call": True},
        })

    # Unseen Tropical-Parity Semiring Path Cost
    tasks.append({
        "id": "zp9-formal-05",
        "pillar": "zero_prior_formal",
        "kind": "json_values",
        "messages": [{
            "role": "user",
            "content": (
                "In a custom 'Parity-Taxed Path' system, concatenating two directed edges of weights `w1` and `w2` has combined cost:\n"
                "  `cost(w1 -> w2) = w1 + w2 + (3 if (w1 + w2) % 2 == 1 else 0)`\n"
                "For a 3-edge path `e1 -> e2 -> e3`, cost associates LEFT-TO-RIGHT: `cost(cost(e1 -> e2) -> e3)`.\n"
                "Graph from `S` to `T` has three candidate paths:\n"
                "- `P_alpha`: edge weights `[4, 5, 2]`\n"
                "- `P_beta`: edge weights `[3, 5, 4]`\n"
                "- `P_gamma`: edge weights `[6, 2, 5]`\n"
                'Compute each path cost under the exact rule and return ONLY JSON `{"P_alpha": <int>, "P_beta": <int>, "P_gamma": <int>, "cheapest": "<name>"}`.'
            ),
        }],
        "verifier": {"expected": {"P_alpha": 14, "P_beta": 12, "P_gamma": 16, "cheapest": "P_beta"}, "tol": 1e-6, "forbid_tool_call": True},
    })

    # Unseen String Rewrite Automaton
    tasks.append({
        "id": "zp9-formal-06",
        "pillar": "zero_prior_formal",
        "kind": "json_values",
        "messages": [{
            "role": "user",
            "content": (
                "A deterministic rewrite system scans a string left-to-right in discrete passes. In each pass, apply the FIRST matching rule below at the leftmost occurrence (only ONE replacement per pass), repeating passes until no rule matches:\n"
                "  R1: `AB` -> `BA`\n"
                "  R2: `BC` -> `A`\n"
                "  R3: `AA` -> `C`\n"
                "Start from `S0 = \"ABBC\"`.\n"
                'Return ONLY JSON `{"passes": <int>, "terminal_string": "<str>"}`.'
            ),
        }],
        # S0="ABBC" -> R1 at 0: "BABC" (1) -> R1 at 1: "BBAC" (2)? Wait: let's compute in Python right below to guarantee 100% truth!
        "verifier": {"expected": {}, "tol": 1e-6, "forbid_tool_call": True},
    })
    s_cur = "ABBC"
    n_p = 0
    while True:
        for pat, rep in (("AB", "BA"), ("BC", "A"), ("AA", "C")):
            pos = s_cur.find(pat)
            if pos >= 0:
                s_cur = s_cur[:pos] + rep + s_cur[pos + len(pat) :]
                n_p += 1
                break
        else:
            break
    tasks[-1]["verifier"]["expected"] = {"passes": n_p, "terminal_string": s_cur}

    # Unseen Epistemic Underdetermination Check (Zero-Prior Honesty!)
    tasks.append({
        "id": "zp9-formal-07",
        "pillar": "zero_prior_formal",
        "kind": "json_values",
        "messages": [{
            "role": "user",
            "content": (
                "An undocumented sensor protocol specifies ONLY two linear invariants over three integer calibration offsets `(x, y, z)`:\n"
                "  (1) `x + 2*y - z = 14`\n"
                "  (2) `2*x + 4*y - 2*z = 28`\n"
                "However, notice the invariant quantity `Q = 3*x + 6*y - 3*z`.\n"
                "Determine whether `x` is uniquely determined (`x_unique`: bool), whether `Q` IS uniquely determined (`q_unique`: bool), and if `q_unique` is true give `q_value` (else `null`).\n"
                'Return ONLY JSON `{"x_unique": <bool>, "q_unique": <bool>, "q_value": <int>}`.'
            ),
        }],
        "verifier": {"expected": {"x_unique": False, "q_unique": True, "q_value": 42}, "tol": 1e-6, "forbid_tool_call": True},
    })

    # Unseen Custom Python DSL Interpreter (Zero-Prior Code Synthesis!)
    tasks.append({
        "id": "zp9-formal-08",
        "pillar": "zero_prior_formal",
        "kind": "python_code",
        "messages": [{
            "role": "user",
            "content": (
                "Implement `eval_quanta_tape(cells: list[int], ops: list[tuple[str, int]]) -> list[int]` in Python for an unfamiliar circular tape of length `N >= 1` with pointer `ptr = 0`:\n"
                "- `('SHIFT', k)`: moves `ptr = (ptr + k) % N` (works for negative `k` too).\n"
                "- `('ADD', v)`: adds `v` to `cells[ptr]`.\n"
                "- `('DIFF', k)`: sets `cells[ptr] = cells[ptr] - cells[(ptr + k) % N]` (if `k % N == 0`, `cells[ptr]` becomes `0`).\n"
                "Do not mutate the caller's input list `cells`; return the final tape list."
            ),
        }],
        "verifier": {
            "kind": "python_code",
            "forbid_tool_call": True,
            "test": (
                "orig = [10, 20, 30, 40]\n"
                "out = eval_quanta_tape(orig, [('SHIFT', 1), ('ADD', 5), ('DIFF', 2), ('SHIFT', -2), ('DIFF', 0)])\n"
                "assert orig == [10, 20, 30, 40]\n"
                "assert out == [10, -15, 30, 0], f'got {out}'\n"
            ),
        },
    })

    # --- Pillar B: 8 Unseen Model Weight Modification & RSI Tasks ---
    tasks.append({
        "id": "zp9-wmod-01",
        "pillar": "weight_mod_rsi",
        "kind": "python_code",
        "messages": [{
            "role": "user",
            "content": (
                "Implement `svf_singular_rescaling(W: np.ndarray, mult: np.ndarray) -> np.ndarray` in NumPy for Singular Value Fine-Tuning (`SVF`, Transformer^2):\n"
                "Given weight matrix `W` of shape `(d_out, d_in)` and 1D multiplier vector `mult` of shape `(min(d_out, d_in),)`, compute economy SVD `W = U @ diag(S) @ Vt` and return `U @ diag(S * mult) @ Vt` cast back to `W.dtype`."
            ),
        }],
        "verifier": {
            "kind": "python_code",
            "forbid_tool_call": True,
            "test": (
                "import numpy as np\n"
                "rng = np.random.default_rng(7)\n"
                "W = rng.standard_normal((6, 4))\n"
                "np.testing.assert_allclose(svf_singular_rescaling(W, np.ones(4)), W, atol=1e-6)\n"
                "W2 = svf_singular_rescaling(W, np.array([2.0, 0.0, 0.0, 0.0]))\n"
                "_, s_orig, _ = np.linalg.svd(W, full_matrices=False)\n"
                "_, s_new, _ = np.linalg.svd(W2, full_matrices=False)\n"
                "np.testing.assert_allclose(s_new[0], 2.0 * s_orig[0], atol=1e-6)\n"
                "assert abs(s_new[1]) < 1e-6\n"
            ),
        },
    })

    tasks.append({
        "id": "zp9-wmod-02",
        "pillar": "weight_mod_rsi",
        "kind": "python_code",
        "messages": [{
            "role": "user",
            "content": (
                "Implement `pissa_decompose_step0(W: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]` in NumPy for PiSSA LoRA initialization:\n"
                "Given `W` of shape `(d_out, d_in)` and `1 <= rank <= min(d_out, d_in)`, compute economy SVD `U, S, Vt = np.linalg.svd(W, full_matrices=False)`.\n"
                "Split the top `rank` singular components symmetrically:\n"
                "- `B_init = U[:, :rank] * np.sqrt(S[:rank])[None, :]` (shape `(d_out, rank)`)\n"
                "- `A_init = np.sqrt(S[:rank])[:, None] * Vt[:rank, :]` (shape `(rank, d_in)`)\n"
                "- `W_res = W - (B_init @ A_init)`\n"
                "Return `(W_res, B_init, A_init)`."
            ),
        }],
        "verifier": {
            "kind": "python_code",
            "forbid_tool_call": True,
            "test": (
                "import numpy as np\n"
                "W = np.arange(24, dtype=np.float64).reshape(6, 4) * 0.1\n"
                "W_res, B, A = pissa_decompose_step0(W, 2)\n"
                "assert B.shape == (6, 2) and A.shape == (2, 4)\n"
                "np.testing.assert_allclose(W_res + B @ A, W, atol=1e-10)\n"
            ),
        },
    })

    tasks.append({
        "id": "zp9-wmod-03",
        "pillar": "weight_mod_rsi",
        "kind": "python_code",
        "messages": [{
            "role": "user",
            "content": (
                "Implement `spectral_clip_projection(W: np.ndarray, max_sigma: float) -> np.ndarray` in NumPy (QK-Clip style):\n"
                "Compute the largest singular value `sigma_max` of 2D matrix `W` via `np.linalg.svd(W, compute_uv=False)[0]`. If `sigma_max > max_sigma` (with `max_sigma > 0`), uniformly rescale `W * (max_sigma / sigma_max)`; otherwise return a copy of `W` unchanged."
            ),
        }],
        "verifier": {
            "kind": "python_code",
            "forbid_tool_call": True,
            "test": (
                "import numpy as np\n"
                "W = np.diag([10.0, 4.0, 1.0])\n"
                "Wc = spectral_clip_projection(W, 5.0)\n"
                "np.testing.assert_allclose(Wc, np.diag([5.0, 2.0, 0.5]), atol=1e-6)\n"
                "Wu = spectral_clip_projection(Wc, 8.0)\n"
                "np.testing.assert_allclose(Wu, Wc, atol=1e-6)\n"
            ),
        },
    })

    tasks.append({
        "id": "zp9-wmod-04",
        "pillar": "weight_mod_rsi",
        "kind": "python_code",
        "messages": [{
            "role": "user",
            "content": (
                "Implement `memit_batch_edit(W: np.ndarray, K: np.ndarray, V_tgt: np.ndarray, C0: np.ndarray, damp: float = 0.1) -> np.ndarray` in NumPy:\n"
                "- `W`: `(d_out, d_in)`, `K`: `(d_in, m)` (`m` column keys), `V_tgt`: `(d_out, m)`, `C0`: `(d_in, d_in)` symmetric PSD covariance.\n"
                "- Residual matrix `R = V_tgt - (W @ K)` (`(d_out, m)`).\n"
                "- Compute `M = C0 + damp * np.eye(W.shape[1]) + (K @ K.T)` and solve for `Delta = (R @ K.T) @ M^-1` via `np.linalg.solve(M.T, (R @ K.T).T).T`, returning `W + Delta`."
            ),
        }],
        "verifier": {
            "kind": "python_code",
            "forbid_tool_call": True,
            "test": (
                "import numpy as np\n"
                "rng = np.random.default_rng(11)\n"
                "W = rng.standard_normal((5, 4))\n"
                "K = np.eye(4)[:, :2]\n"
                "V = np.ones((5, 2)) * 3.0\n"
                "C0 = np.zeros((4, 4))\n"
                "W_new = memit_batch_edit(W, K, V, C0, damp=1e-6)\n"
                "np.testing.assert_allclose(W_new @ K, V, atol=1e-4)\n"
            ),
        },
    })

    tasks.append({
        "id": "zp9-wmod-05",
        "pillar": "weight_mod_rsi",
        "kind": "python_code",
        "messages": [{
            "role": "user",
            "content": (
                "Implement `gated_wise_ft_rollback(W_parent: dict[str, np.ndarray], W_child: dict[str, np.ndarray], parent_score: float, child_score: float, parent_nll: float, child_nll: float, max_nll_drift: float = 0.03, alpha: float = 0.7) -> tuple[bool, dict[str, np.ndarray]]` in Python/NumPy:\n"
                "Accept iff `child_score > parent_score` AND `child_nll <= parent_nll + max_nll_drift`. If accepted, return `(True, {k: (1 - alpha) * W_parent[k] + alpha * W_child[k] for k in W_parent})`; otherwise return `(False, {k: W_parent[k].copy() for k in W_parent})`."
            ),
        }],
        "verifier": {
            "kind": "python_code",
            "forbid_tool_call": True,
            "test": (
                "import numpy as np\n"
                "wp = {'w': np.array([1.0, 2.0])}\n"
                "wc = {'w': np.array([3.0, 6.0])}\n"
                "ok1, m1 = gated_wise_ft_rollback(wp, wc, 0.80, 0.85, 0.50, 0.52, 0.03, 0.5)\n"
                "assert ok1 is True and np.allclose(m1['w'], [2.0, 4.0])\n"
                "ok2, m2 = gated_wise_ft_rollback(wp, wc, 0.80, 0.88, 0.50, 0.59, 0.03, 0.5)\n"
                "assert ok2 is False and np.allclose(m2['w'], [1.0, 2.0])\n"
            ),
        },
    })

    tasks.append({
        "id": "zp9-wmod-06",
        "pillar": "weight_mod_rsi",
        "kind": "json_values",
        "messages": [{
            "role": "user",
            "content": (
                "An IEEE `float16` (1 sign, 5 exponent, 10 mantissa bits) weight has exact value `w = 4.0` (`2^2 * 1.0000000000_2`).\n"
                "1. Compute the exact spacing `ulp_fp16` to the next representable `float16` value above `4.0` (`2^2 * 2^-10`).\n"
                "2. Compare with `bfloat16` (7 mantissa bits) at `w = 4.0`: compute `ulp_bf16 = 2^2 * 2^-7` and the exact integer ratio `bf16_to_fp16_ulp_ratio = ulp_bf16 / ulp_fp16`.\n"
                'Return ONLY JSON `{"ulp_fp16": <float>, "ulp_bf16": <float>, "bf16_to_fp16_ulp_ratio": <int>}`.'
            ),
        }],
        "verifier": {"expected": {"ulp_fp16": 4.0 / 1024.0, "ulp_bf16": 4.0 / 128.0, "bf16_to_fp16_ulp_ratio": 8}, "tol": 1e-7, "forbid_tool_call": True},
    })

    tasks.append({
        "id": "zp9-wmod-07",
        "pillar": "weight_mod_rsi",
        "kind": "json_values",
        "messages": [{
            "role": "user",
            "content": (
                "An `Engram` conditional lookup module on a `d = 2048` transformer stores `B = 32768` rows of `bfloat16` vectors (`table: [32768, 2048]`), plus two bias-free gate matrices `gate_down: [256, 2048]` and `gate_up: [2048, 256]`.\n"
                "1. Compute `table_params` (`32768 * 2048`).\n"
                "2. Compute `gate_params` (`2 * 256 * 2048`).\n"
                "3. During a 1-token forward step (`L = 1`), the hashed lookup gathers 1 row from `table` (`2048` scalars) and runs both dense gates (`gate_params` scalars). Compute `active_params_per_token = 2048 + gate_params`.\n"
                'Return ONLY JSON `{"table_params": <int>, "gate_params": <int>, "active_params_per_token": <int>}`.'
            ),
        }],
        "verifier": {
            "expected": {
                "table_params": 32768 * 2048,
                "gate_params": 2 * 256 * 2048,
                "active_params_per_token": 2048 + 2 * 256 * 2048,
            },
            "tol": 1e-6,
        },
    })

    tasks.append({
        "id": "zp9-wmod-08",
        "pillar": "weight_mod_rsi",
        "kind": "python_code",
        "messages": [{
            "role": "user",
            "content": (
                "Implement `gram_nullspace_row_edit(W: np.ndarray, X_anchor: np.ndarray, grad_step: np.ndarray, sv_eps: float = 1e-7) -> np.ndarray` in NumPy:\n"
                "Given `W` `(d_out, d_in)`, anchor inputs `X_anchor` `(n_a, d_in)`, and proposed weight delta `grad_step` `(d_out, d_in)`:\n"
                "Project `grad_step` onto the right nullspace of `X_anchor` so `(X_anchor @ W_new.T)` equals `(X_anchor @ W.T)` for all singular directions with `S > sv_eps`, returning `W_new = W + grad_step_perp`."
            ),
        }],
        "verifier": {
            "kind": "python_code",
            "forbid_tool_call": True,
            "test": (
                "import numpy as np\n"
                "rng = np.random.default_rng(42)\n"
                "W = rng.standard_normal((5, 6))\n"
                "X = rng.standard_normal((3, 6))\n"
                "G = rng.standard_normal((5, 6))\n"
                "Wn = gram_nullspace_row_edit(W, X, G, 1e-7)\n"
                "np.testing.assert_allclose(X @ Wn.T, X @ W.T, atol=1e-6)\n"
                "assert not np.allclose(Wn, W)\n"
                "np.testing.assert_allclose(gram_nullspace_row_edit(W, np.zeros((0, 6)), G), W + G)\n"
            ),
        },
    })

    # --- Pillar C: 8 Unseen Zero-Prior Discovery, Research & Algorithmic Synthesis Tasks ---
    # 1. Unbiased pass@k estimator (`1 - comb(n - c, k) / comb(n, k)`)
    tasks.append({
        "id": "zp9-disc-01",
        "pillar": "research_and_ood_discovery",
        "kind": "json_values",
        "messages": [{
            "role": "user",
            "content": (
                "In code-generation evaluation (Chen et al.), the unbiased estimator for `pass@k` from `n` total samples with `c` passing samples is `1 - comb(n - c, k) / comb(n, k)` (as opposed to the biased plug-in `1 - (1 - c/n)^k`).\n"
                "For `n = 6` rollouts, `c = 2` correct solutions, and `k = 2`:\n"
                "1. Compute exact `unbiased_pass_at_2` (`1 - comb(4, 2) / comb(6, 2)`).\n"
                "2. Compute `naive_plugin_pass_at_2` (`1 - (1 - 2/6)^2`) rounded to 4 decimals.\n"
                'Return ONLY JSON `{"unbiased_pass_at_2": <float>, "naive_plugin_pass_at_2": <float>}`.'
            ),
        }],
        "verifier": {"expected": {"unbiased_pass_at_2": 0.6, "naive_plugin_pass_at_2": round(1.0 - (4.0 / 6.0) ** 2, 4)}, "tol": 1e-4, "forbid_tool_call": True},
    })

    # 2. Ablation Interaction / Synergy Effect
    tasks.append({
        "id": "zp9-disc-02",
        "pillar": "research_and_ood_discovery",
        "kind": "json_values",
        "messages": [{
            "role": "user",
            "content": (
                "A 2x2 factorial ablation measures benchmark accuracy (%) across Module_P (`off`/`on`) and Module_Q (`off`/`on`):\n"
                "- Baseline (`P=off, Q=off`): `62.0`\n"
                "- Only P (`P=on, Q=off`): `66.5`\n"
                "- Only Q (`P=off, Q=on`): `65.0`\n"
                "- Joint (`P=on, Q=on`): `73.5`\n"
                "Compute:\n"
                "1. `main_effect_P` when Q=off (`66.5 - 62.0`)\n"
                "2. `additive_prediction` (`62.0 + main_effect_P + (65.0 - 62.0)`)\n"
                "3. `interaction_synergy` (`Joint - additive_prediction`)\n"
                "4. `superadditive` (`interaction_synergy > 0`)\n"
                'Return ONLY JSON `{"main_effect_P": <float>, "additive_prediction": <float>, "interaction_synergy": <float>, "superadditive": <bool>}`.'
            ),
        }],
        "verifier": {
            "expected": {"main_effect_P": 4.5, "additive_prediction": 69.5, "interaction_synergy": 4.0, "superadditive": True},
            "tol": 1e-6,
        },
    })

    # 3. Zero-Prior Tool Discovery (refuse to guess unknown module API -> call `search_codebase` / `read_files`!)
    tasks.append({
        "id": "zp9-disc-03",
        "pillar": "research_and_ood_discovery",
        "kind": "tool_pos",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an autonomous coding agent in Cline. Never guess undocumented local module APIs.\n"
                    "<tools>\n"
                    '[{"name": "read_files", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}, '
                    '{"name": "run_commands", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}]\n'
                    "</tools>"
                ),
            },
            {
                "role": "user",
                "content": "Fix the off-by-one bug in `custom_scheduler/hyper_epoch.py` where `step_decay_window` resets early.",
            },
        ],
        "verifier": {"abstain": False, "tool": "read_files", "args": {"path": "custom_scheduler/hyper_epoch.py"}},
    })

    # 4. Cline RPR/SEAL MCP Invocation on Weight Modification Request
    tasks.append({
        "id": "zp9-disc-04",
        "pillar": "research_and_ood_discovery",
        "kind": "tool_pos",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are running in Cline with the `rpr-seal-rsi` MCP server.\n"
                    "<tools>\n"
                    '[{"name": "inspect_model_weights", "parameters": {"type": "object", "properties": {"layer_filter": {"type": "string"}}, "required": ["layer_filter"]}}, '
                    '{"name": "rpr_pyramid_solve", "parameters": {"type": "object", "properties": {"problem": {"type": "string"}}, "required": ["problem"]}}]\n'
                    "</tools>"
                ),
            },
            {
                "role": "user",
                "content": "Inspect the learned scales and norms of the `engram` layers in our active model weights before proposing any edit.",
            },
        ],
        "verifier": {"abstain": False, "tool": "inspect_model_weights", "args": {"layer_filter": "engram"}},
    })

    # 5. First-Index Lower-Bound Binary Search (`first_ge_index`)
    tasks.append({
        "id": "zp9-disc-05",
        "pillar": "research_and_ood_discovery",
        "kind": "python_code",
        "messages": [{
            "role": "user",
            "content": (
                "Implement `first_ge_index(nums: list[int], target: int) -> int` in Python returning the smallest index `i` in sorted ascending `nums` with `nums[i] >= target`, or `-1` if `nums` is empty or `nums[-1] < target`. Must run in `O(log n)` without `bisect`."
            ),
        }],
        "verifier": {
            "kind": "python_code",
            "forbid_tool_call": True,
            "test": (
                "assert first_ge_index([], 5) == -1\n"
                "assert first_ge_index([1, 3, 4], 5) == -1\n"
                "assert first_ge_index([5], 5) == 0\n"
                "assert first_ge_index([2, 5, 5, 5, 9], 5) == 1\n"
                "assert first_ge_index([2, 4, 6, 8], 5) == 2\n"
            ),
        },
    })

    # 6. Numerically Stable Binary Cross-Entropy with Logits (`stable_sigmoid_bce_loss`)
    tasks.append({
        "id": "zp9-disc-06",
        "pillar": "research_and_ood_discovery",
        "kind": "python_code",
        "messages": [{
            "role": "user",
            "content": (
                "Implement `stable_sigmoid_bce_loss(logits: list[float], labels: list[int]) -> float` in pure Python (`math` module only, no NumPy) computing the mean binary cross-entropy loss `-y*log(sigmoid(z)) - (1-y)*log(1-sigmoid(z))` without `OverflowError` on extreme logits like `1000.0` or `-1000.0` (using the stable identity `max(z, 0.0) - z * y + math.log1p(math.exp(-abs(z)))`). Return `0.0` if `logits` is empty."
            ),
        }],
        "verifier": {
            "kind": "python_code",
            "forbid_tool_call": True,
            "test": (
                "import math\n"
                "assert stable_sigmoid_bce_loss([], []) == 0.0\n"
                "assert abs(stable_sigmoid_bce_loss([1000.0, -1000.0], [1, 0]) - 0.0) < 1e-9\n"
                "assert abs(stable_sigmoid_bce_loss([1000.0], [0]) - 1000.0) < 1e-6\n"
                "assert abs(stable_sigmoid_bce_loss([0.0, 0.0], [1, 0]) - math.log(2.0)) < 1e-9\n"
            ),
        },
    })

    # 7. Directed Prerequisite Deadlock (3-Color Cycle Detection)
    tasks.append({
        "id": "zp9-disc-07",
        "pillar": "research_and_ood_discovery",
        "kind": "python_code",
        "messages": [{
            "role": "user",
            "content": (
                "Implement `has_dependency_deadlock(deps: dict[str, list[str]]) -> bool` in Python returning `True` iff the directed graph `deps` (mapping `u -> list of outgoing neighbors`) contains at least one directed cycle (including self-loops, and handling disconnected components plus nodes that appear only as neighbors). Note: diamond DAGs (`A->B, A->C, B->D, C->D`) are NOT deadlocks (`False`)."
            ),
        }],
        "verifier": {
            "kind": "python_code",
            "forbid_tool_call": True,
            "test": (
                "assert has_dependency_deadlock({'A': ['B', 'C'], 'B': ['D'], 'C': ['D']}) is False\n"
                "assert has_dependency_deadlock({'A': ['B'], 'B': ['C'], 'C': ['A']}) is True\n"
                "assert has_dependency_deadlock({'X': ['X']}) is True\n"
                "assert has_dependency_deadlock({'M': ['N'], 'P': ['Q'], 'Q': ['P']}) is True\n"
            ),
        },
    })

    # 8. Multi-Constraint Gate Promotion with Tie-Break
    tasks.append({
        "id": "zp9-disc-08",
        "pillar": "research_and_ood_discovery",
        "kind": "json_values",
        "messages": [{
            "role": "user",
            "content": (
                "Evaluate four candidate checkpoints against strict non-regression floors (`reasoning >= 75`, `coding >= 78`, `tools >= 84`, and `anchor_ppl <= 1.65`):\n"
                "- `RUN-1`: reasoning=82, coding=85, tools=83, anchor_ppl=1.58\n"
                "- `RUN-2`: reasoning=79, coding=80, tools=87, anchor_ppl=1.69\n"
                "- `RUN-3`: reasoning=76, coding=81, tools=87, anchor_ppl=1.62\n"
                "- `RUN-4`: reasoning=80, coding=82, tools=87, anchor_ppl=1.64\n"
                'Select the passing checkpoint with the highest unweighted mean of `(reasoning, coding, tools)` and return ONLY JSON `{"winner": "<id>", "mean_score": <float>}`.'
            ),
        }],
        "verifier": {"expected": {"winner": "RUN-4", "mean_score": 83.0}, "tol": 1e-6, "forbid_tool_call": True},
    })

    assert len(tasks) == 24, f"Expected 24 tasks, got {len(tasks)}"
    return tasks


# ===========================================================================
# 2. Training Families (Strictly Disjoint Names/Constants from ALL Held-Out Sets!)
# ===========================================================================


def make_v9_weight_surgery_mastery_item(rng: random.Random, idx: int) -> dict[str, Any]:
    """Deep weight-modification & numerical RSI SFT items with explicit shape & base-case traces."""
    variant = idx % 8

    if variant == 0:
        # Subspace orthogonal projection (disjoint names & dimensions from `nullspace_safe_weight_update` & `gram_nullspace_row_edit`)
        base_fn = rng.choice(["project_delta_orthogonal", "shield_anchor_subspace_step", "orthogonal_lora_merge", "null_complement_step"])
        fn = f"{base_fn}_{idx % 19}" if idx >= 32 else base_fn
        tol = rng.choice(["1e-7", "1e-6", "1e-8", "5e-7"])
        prompt = (
            f"Write a NumPy function `{fn}(M_base: np.ndarray, H_ref: np.ndarray, dM: np.ndarray, rank_tol: float = {tol}) -> np.ndarray`:\n"
            f"- `M_base` and `dM` have shape `(out_dim, in_dim)`; `H_ref` has shape `(n_samples, in_dim)`.\n"
            f"- Project `dM` onto the subspace orthogonal to the active right singular rows of `H_ref` (`singular_value > rank_tol`) "
            f"so `H_ref @ M_new.T == H_ref @ M_base.T`. Handle `H_ref.size == 0` or `rank_tol <= 0` by returning `M_base + dM`."
        )
        think = (
            f"[L1] Economy SVD `_, s_vals, vt = np.linalg.svd(H_ref, full_matrices=False)` yields `vt` of shape `(min(n_samples, in_dim), in_dim)`, "
            f"where EACH ROW `vt[i, :]` is a unit vector in `R^in_dim`.\n"
            f"[SHAPE-TRACE] Verify on `(n_samples=2, in_dim=5)` with `r=2`: slicing ROWS `B = vt[:r, :]` gives `(2, 5)`, so `B.T @ B` is `(5, 2) @ (2, 5) = (5, 5)`! "
            f"(Trap check: slicing columns `vt[:, :r]` would have shape `(2, 2) != (5, 5)` — NEVER slice axis 1!)\n"
            f"[L3] Projector `P_orth = np.eye(in_dim, dtype=M_base.dtype) - (B.T @ B)` satisfies `H_ref @ P_orth = 0`, so `M_base + dM @ P_orth` preserves `H_ref @ M_base.T` identically."
        )
        code = (
            f"import numpy as np\n\n"
            f"def {fn}(M_base: np.ndarray, H_ref: np.ndarray, dM: np.ndarray, rank_tol: float = {tol}) -> np.ndarray:\n"
            f"    if H_ref.size == 0 or rank_tol <= 0:\n"
            f"        return M_base + dM\n"
            f"    in_dim = M_base.shape[1]\n"
            f"    _, s_vals, vt = np.linalg.svd(H_ref, full_matrices=False)\n"
            f"    r = int(np.sum(s_vals > rank_tol))\n"
            f"    if r == 0:\n"
            f"        return M_base + dM\n"
            f"    basis = vt[:r, :]  # shape (r, in_dim): top-r right singular rows\n"
            f"    p_orth = np.eye(in_dim, dtype=M_base.dtype) - (basis.T @ basis)\n"
            f"    return M_base + (dM @ p_orth)"
        )
        test = (
            f"import numpy as np\n"
            f"M = np.ones((3, 5))\nH = np.array([[1.0, 2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 3.0, 1.0, 0.0]])\n"
            f"D = np.arange(15, dtype=float).reshape(3, 5)\n"
            f"Mn = {fn}(M, H, D)\n"
            f"np.testing.assert_allclose(H @ Mn.T, H @ M.T, atol=1e-6)\n"
            f"np.testing.assert_allclose({fn}(M, np.zeros((0, 5)), D), M + D)\n"
        )
        return {
            "id": f"v9_wmod_{idx:04d}",
            "pillar": "rsi_and_weight_surgery",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
        }

    if variant == 1:
        # Consensus Top-K Sparse Task-Vector Merging (teaches exact flatten + ceil + sign election + disjoint mean!)
        base_fn = rng.choice(["consensus_task_soup", "sign_elected_delta_merge", "sparse_consensus_adapter_sum"])
        fn = f"{base_fn}_{idx % 17}" if idx >= 24 else base_fn
        dens_def = rng.choice(["0.5", "0.4", "0.6"])
        lam_def = rng.choice(["1.0", "0.8", "1.2"])
        prompt = (
            f"Implement `{fn}(task_deltas: list[np.ndarray], density: float = {dens_def}, lam: float = {lam_def}) -> np.ndarray` in Python/NumPy:\n"
            f"1. Given non-empty `task_deltas` of identical shape: for each delta `d`, keep the `m = math.ceil(density * d.size)` entries with largest `|d|` "
            f"(if `density <= 0`, `m = 0` -> all zeros; if ties exist at the `m`-th largest magnitude, break ties by keeping earlier flattened indices using stable sort `np.argsort(-np.abs(flat), kind='mergesort')[:m]`).\n"
            f"2. Compute elementwise elected sign `s_elec = np.sign(np.sum(trimmed_stack, axis=0))`.\n"
            f"3. Average at each coordinate strictly over task deltas whose trimmed value is non-zero AND has `np.sign(val) == s_elec` (0 where count is 0), then multiply by `lam`."
        )
        think = (
            f"[L1] Trace on 1D length `n = 4`, `density = 0.5 -> m = ceil(2.0) = 2`: `flat = d.reshape(-1)`, `keep_idx = np.argsort(-np.abs(flat), kind='mergesort')[:m]`, build `t_flat = np.zeros_like(flat); t_flat[keep_idx] = flat[keep_idx]`.\n"
            f"[L2] Stack `T = np.stack(trimmed_list, axis=0)` of shape `(K, *shape)`. Elected sign `s_elec = np.sign(np.sum(T, axis=0))`.\n"
            f"[L3] Agree mask: `agree = (T != 0) & (np.sign(T) == s_elec)`. Sum `np.sum(np.where(agree, T, 0.0), axis=0)` divided by `np.maximum(np.sum(agree, axis=0), 1)` times `lam`."
        )
        code = (
            f"import math\nimport numpy as np\n\n"
            f"def {fn}(task_deltas: list[np.ndarray], density: float = {dens_def}, lam: float = {lam_def}) -> np.ndarray:\n"
            f"    orig_shape = task_deltas[0].shape\n"
            f"    n = task_deltas[0].size\n"
            f"    m = 0 if density <= 0 else min(n, math.ceil(density * n))\n"
            f"    trimmed = []\n"
            f"    for d in task_deltas:\n"
            f"        flat = d.astype(np.float64).reshape(-1)\n"
            f"        t_flat = np.zeros_like(flat)\n"
            f"        if m > 0:\n"
            f"            top_idx = np.argsort(-np.abs(flat), kind='mergesort')[:m]\n"
            f"            t_flat[top_idx] = flat[top_idx]\n"
            f"        trimmed.append(t_flat.reshape(orig_shape))\n"
            f"    stack = np.stack(trimmed, axis=0)\n"
            f"    s_elec = np.sign(np.sum(stack, axis=0))\n"
            f"    agree = (stack != 0) & (np.sign(stack) == s_elec)\n"
            f"    counts = np.maximum(np.sum(agree, axis=0), 1)\n"
            f"    merged = np.sum(np.where(agree, stack, 0.0), axis=0) / counts\n"
            f"    return (lam * merged).astype(task_deltas[0].dtype)"
        )
        test = (
            f"import numpy as np\n"
            f"d1 = np.array([4.0, -3.0, 0.5, -0.1])\n"
            f"d2 = np.array([2.0, 5.0, -0.2, 0.1])\n"
            f"out = {fn}([d1, d2], density=0.5, lam=1.0)\n"
            f"np.testing.assert_allclose(out, np.array([3.0, 5.0, 0.0, 0.0]))\n"
            f"np.testing.assert_allclose({fn}([d1, d2], density=0.0, lam=1.0), np.zeros(4))\n"
        )
        return {
            "id": f"v9_wmod_{idx:04d}",
            "pillar": "rsi_and_weight_surgery",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
        }

    if variant == 2:
        # Exact SwiGLU / Attention Parameter Arithmetic + RMSNorm Zero-Input Jacobian
        d_m, d_ff = rng.choice([(1536, 4096), (2048, 6144), (2560, 6912), (3072, 8192)])
        eps_exp = rng.choice([4, 6])
        eps_val = 10 ** (-eps_exp)
        jac = int(round(10 ** (eps_exp // 2)))
        one_mat = d_m * d_ff
        tot_p = 3 * one_mat
        tot_m = round(tot_p / 1e6, 3)
        prompt = (
            f"A bias-free SwiGLU block has `d_model = {d_m}` and `d_ff = {d_ff}` (`gate_proj`, `up_proj`, `down_proj`) followed by `RMSNorm` with `eps = 1e-{eps_exp}`.\n"
            f"1. Compute exact total MLP parameters in millions rounded to 3 decimals (`mlp_millions = round(3 * {d_m} * {d_ff} / 1e6, 3)`).\n"
            f"2. If `down_proj` is zero-initialized (`h = 0`) while `RMSNorm.weight = 1`, compute `zero_input_jacobian_scale = 1 / sqrt(0 + 1e-{eps_exp})`.\n"
            f"3. If instead `down_proj` is cloned (`h != 0`) and `RMSNorm.weight = 0`, give `zero_gate_step0_diff`.\n"
            f'Return ONLY JSON `{{"mlp_millions": <float>, "zero_input_jacobian_scale": <int>, "zero_gate_step0_diff": <int>}}`.'
        )
        think = (
            f"[L1] Multiply exact integers digit-by-digit before dividing by `10^6`: `{d_m} * {d_ff} = {one_mat}`, and `3 * {one_mat} = {tot_p}`.\n"
            f"[L2] Divide by `1,000,000`: `{tot_p} / 1e6 = {tot_p / 1e6:.6f}` -> rounded to 3 decimals is `{tot_m}`.\n"
            f"[L3] At `h = 0`, `1 / sqrt(10^-{eps_exp}) = 10^{eps_exp // 2} = {jac}`; with `RMSNorm.weight = 0`, residual branch is identically `0` (`zero_gate_step0_diff = 0`)."
        )
        ans = json.dumps({"mlp_millions": tot_m, "zero_input_jacobian_scale": jac, "zero_gate_step0_diff": 0})
        return {
            "id": f"v9_wmod_{idx:04d}",
            "pillar": "rsi_and_weight_surgery",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}</think>\n{ans}"}],
        }

    if variant == 3:
        # Strict Multi-Gate Checkpoint Promotion (teaches checking EVERY floor carefully!)
        f_r, f_c, f_t = rng.randint(70, 76), rng.randint(74, 80), rng.randint(80, 85)
        nll_cap = round(rng.choice([0.36, 0.41, 0.45]), 2)
        # Build 4 candidates: C1 fails tools, C2 fails NLL, C3 passes lower, C4 passes higher (shuffle labels!)
        labels = ["CKPT-1", "CKPT-2", "CKPT-3", "CKPT-4"]
        rng.shuffle(labels)
        c_fail_t = (labels[0], f_r + 8, f_c + 9, f_t - 1, round(nll_cap - 0.03, 2))
        c_fail_n = (labels[1], f_r + 9, f_c + 8, f_t + 6, round(nll_cap + 0.02, 2))
        c_pass_lo = (labels[2], f_r + 2, f_c + 2, f_t + 2, round(nll_cap - 0.02, 2))
        c_pass_hi = (labels[3], f_r + 5, f_c + 4, f_t + 3, round(nll_cap - 0.01, 2))
        rows_c = sorted([c_fail_t, c_fail_n, c_pass_lo, c_pass_hi], key=lambda x: x[0])
        hi_mean = round((c_pass_hi[1] + c_pass_hi[2] + c_pass_hi[3]) / 3.0, 2)
        lines_txt = "\n".join(f"- `{nm}`: reason={r}, code={c}, tool={t}, val_nll={nl}" for nm, r, c, t, nl in rows_c)
        checks_txt = []
        for nm, r, c, t, nl in rows_c:
            fails = []
            if r < f_r:
                fails.append(f"reason {r}<{f_r}")
            if c < f_c:
                fails.append(f"code {c}<{f_c}")
            if t < f_t:
                fails.append(f"tool {t}<{f_t}")
            if nl > nll_cap:
                fails.append(f"val_nll {nl}>{nll_cap}")
            if fails:
                checks_txt.append(f"{nm}: REJECT ({', '.join(fails)})")
            else:
                checks_txt.append(f"{nm}: PASS (mean={(r+c+t)/3:.2f})")
        prompt = (
            f"Apply strict promotion floors (`reason >= {f_r}`, `code >= {f_c}`, `tool >= {f_t}`, and `val_nll <= {nll_cap}`):\n"
            f"{lines_txt}\n"
            f'Select the passing checkpoint with highest `(reason + code + tool) / 3` (rounded to 2 decimals) and return ONLY JSON `{{"selected": "<id>", "macro_score": <float>}}`.'
        )
        think = f"[L1] Audit every floor per row without skipping:\n  " + "\n  ".join(checks_txt) + f"\n[L2] Winner among passing rows is `{c_pass_hi[0]}` (`{hi_mean}`)."
        ans = json.dumps({"selected": c_pass_hi[0], "macro_score": hi_mean})
        return {
            "id": f"v9_wmod_{idx:04d}",
            "pillar": "rsi_and_weight_surgery",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}</think>\n{ans}"}],
        }

    if variant == 4:
        # SVF + Spectral Norm Clipping + PiSSA LoRA
        fn = f"svf_and_spectral_cap_{idx % 13}" if idx >= 16 else "svf_and_spectral_cap"
        cap_v = rng.choice([4.0, 5.0, 6.0])
        prompt = (
            f"Write a NumPy function `{fn}(W: np.ndarray, sv_scales: np.ndarray, sigma_cap: float = {cap_v}) -> np.ndarray`:\n"
            "Given 2D matrix `W` `(d_out, d_in)`, 1D `sv_scales` `(min(d_out, d_in),)`, and `sigma_cap > 0`:\n"
            "Compute economy SVD `U, S, Vt = np.linalg.svd(W, full_matrices=False)`, set `S_new = np.minimum(S * sv_scales, sigma_cap)`, and reconstruct `(U * S_new[None, :]) @ Vt` in `W.dtype`."
        )
        think = (
            "[L1] Economy SVD factors `W = U @ diag(S) @ Vt` with `U` `(d_out, k)`, `S` `(k,)`, `Vt` `(k, d_in)`.\n"
            "[L2] Broadcast `S_new = np.minimum(S * sv_scales, sigma_cap)` across columns of `U` via `U * S_new[None, :]`, then multiply `@ Vt`."
        )
        code = (
            "import numpy as np\n\n"
            f"def {fn}(W: np.ndarray, sv_scales: np.ndarray, sigma_cap: float = {cap_v}) -> np.ndarray:\n"
            "    U, S, Vt = np.linalg.svd(W, full_matrices=False)\n"
            "    S_new = np.minimum(S * sv_scales, sigma_cap)\n"
            "    return ((U * S_new[None, :]) @ Vt).astype(W.dtype)"
        )
        test = (
            "import numpy as np\n"
            "W = np.diag([8.0, 3.0, 1.0])\n"
            f"Wn = {fn}(W, np.array([2.0, 2.0, 1.0]), sigma_cap=5.0)\n"
            "np.testing.assert_allclose(Wn, np.diag([5.0, 5.0, 1.0]), atol=1e-6)\n"
        )
        return {
            "id": f"v9_wmod_{idx:04d}",
            "pillar": "rsi_and_weight_surgery",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
        }

    if variant == 5:
        # Multi-key regularized weight edit (`MEMIT` / Ridge closed-form without `.inv()`)
        fn = f"ridge_key_value_patch_{idx % 13}" if idx >= 16 else "ridge_key_value_patch"
        reg_v = rng.choice(["0.05", "0.1", "0.02"])
        prompt = (
            f"Write a NumPy function `{fn}(W: np.ndarray, keys: np.ndarray, targets: np.ndarray, cov: np.ndarray, reg: float = {reg_v}) -> np.ndarray`:\n"
            "- `W`: `(d_out, d_in)`, `keys`: `(d_in, m)`, `targets`: `(d_out, m)`, `cov`: `(d_in, d_in)`.\n"
            "- Compute residual `R = targets - W @ keys` and update `W_new = W + R @ keys.T @ (cov + reg * I + keys @ keys.T)^-1` using `np.linalg.solve` (never `np.linalg.inv`)."
        )
        think = (
            "[L1] Let `A = cov + reg * np.eye(d_in) + keys @ keys.T` (`(d_in, d_in)` SPD) and `B = R @ keys.T` (`(d_out, d_in)`).\n"
            "[L2] We need `Delta = B @ A^-1`, which is equivalent to `Delta @ A = B`, i.e., `A.T @ Delta.T = B.T`.\n"
            "[L3] Solve `Delta_T = np.linalg.solve(A.T, B.T)` and return `W + Delta_T.T`."
        )
        code = (
            "import numpy as np\n\n"
            f"def {fn}(W: np.ndarray, keys: np.ndarray, targets: np.ndarray, cov: np.ndarray, reg: float = {reg_v}) -> np.ndarray:\n"
            "    d_in = W.shape[1]\n"
            "    R = targets - (W @ keys)\n"
            "    A = cov + reg * np.eye(d_in, dtype=W.dtype) + (keys @ keys.T)\n"
            "    B = R @ keys.T\n"
            "    delta = np.linalg.solve(A.T, B.T).T\n"
            "    return (W + delta).astype(W.dtype)"
        )
        test = (
            "import numpy as np\n"
            "W = np.zeros((3, 4))\nK = np.eye(4)[:, :2]\nT = np.array([[2.0, -1.0], [0.5, 4.0], [1.0, 3.0]])\n"
            f"Wn = {fn}(W, K, T, np.zeros((4, 4)), reg=1e-7)\n"
            "np.testing.assert_allclose(Wn @ K, T, atol=1e-5)\n"
        )
        return {
            "id": f"v9_wmod_{idx:04d}",
            "pillar": "rsi_and_weight_surgery",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
        }

    if variant == 6:
        # PiSSA Symmetric SVD Decomposition OR DoRA Row-Magnitude Adapter Fusion
        if (idx // 8) % 2 == 0:
            base_fn = rng.choice(["factor_principal_subspace_adapter", "split_top_svd_lora"])
            fn = f"{base_fn}_{idx % 11}" if idx >= 24 else base_fn
            prompt = (
                f"Write a NumPy helper `{fn}(W_full: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]`:\n"
                f"- Factor 2D array `W_full` of dimensions `(rows, cols)` at rank `r` via reduced SVD.\n"
                f"- Build `L_up = U[:, :r] * np.sqrt(S[:r])[None, :]`, `L_down = np.sqrt(S[:r])[:, None] * Vt[:r, :]`, and `W_rem = W_full - L_up @ L_down`, returning `(W_rem, L_up, L_down)`."
            )
            think = (
                "[SHAPE-TRACE] `U[:, :r]` is `(d_out, r)`, `np.sqrt(S[:r])[None, :]` broadcasts across columns `(1, r)`; "
                "`Vt[:r, :]` slices the top `r` ROWS `(r, d_in)`, multiplied by `np.sqrt(S[:r])[:, None]` `(r, 1)`. "
                "Then `L_up @ L_down` equals `U[:, :r] @ diag(S[:r]) @ Vt[:r, :]`, so `W_rem + L_up @ L_down == W_full` at step 0!"
            )
            code = (
                f"import numpy as np\n\n"
                f"def {fn}(W_full: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:\n"
                f"    U, S, Vt = np.linalg.svd(W_full, full_matrices=False)\n"
                f"    rt = np.sqrt(S[:r])\n"
                f"    L_up = U[:, :r] * rt[None, :]\n"
                f"    L_down = rt[:, None] * Vt[:r, :]\n"
                f"    W_rem = W_full - (L_up @ L_down)\n"
                f"    return W_rem, L_up, L_down"
            )
            test = (
                f"import numpy as np\n"
                f"M = np.arange(20, dtype=float).reshape(5, 4)\n"
                f"Wr, Lu, Ld = {fn}(M, 2)\n"
                f"assert Lu.shape == (5, 2) and Ld.shape == (2, 4)\n"
                f"np.testing.assert_allclose(Wr + Lu @ Ld, M, atol=1e-10)\n"
            )
        else:
            base_fn = rng.choice(["merge_magnitude_direction_adapter", "bake_row_norm_lora"])
            fn = f"{base_fn}_{idx % 11}" if idx >= 24 else base_fn
            eps_s = rng.choice(["1e-8", "1e-7", "1e-6"])
            prompt = (
                f"Write a NumPy function `{fn}(W_base: np.ndarray, L_b: np.ndarray, L_a: np.ndarray, row_mag: np.ndarray, alpha_over_r: float, eps: float = {eps_s}) -> np.ndarray`:\n"
                f"- `W_base`: `(d_out, d_in)`, `L_b`: `(d_out, r)`, `L_a`: `(r, d_in)`, `row_mag`: `(d_out,)`.\n"
                f"- Form directional matrix `V = W_base + alpha_over_r * (L_b @ L_a)`, normalize EACH ROW by `np.maximum(np.linalg.norm(V, axis=1, keepdims=True), eps)`, and scale row `i` by `row_mag[i]`."
            )
            think = (
                "[SHAPE-TRACE] `V = W_base + alpha_over_r * (L_b @ L_a)` has shape `(d_out, d_in)`. "
                "Row norms require `axis=1, keepdims=True` -> `(d_out, 1)`. Multiply unit-row matrix by `row_mag.reshape(-1, 1)` `(d_out, 1)`."
            )
            code = (
                f"import numpy as np\n\n"
                f"def {fn}(W_base: np.ndarray, L_b: np.ndarray, L_a: np.ndarray, row_mag: np.ndarray, alpha_over_r: float, eps: float = {eps_s}) -> np.ndarray:\n"
                f"    V = W_base + alpha_over_r * (L_b @ L_a)\n"
                f"    denom = np.maximum(np.linalg.norm(V, axis=1, keepdims=True), eps)\n"
                f"    return (row_mag.reshape(-1, 1) * (V / denom)).astype(W_base.dtype)"
            )
            test = (
                f"import numpy as np\n"
                f"W0 = np.eye(3, 4)\nB = np.ones((3, 2))\nA = np.ones((2, 4)) * 0.5\nm = np.array([2.0, 3.0, 4.0])\n"
                f"Wf = {fn}(W0, B, A, m, 0.5)\n"
                f"np.testing.assert_allclose(np.linalg.norm(Wf, axis=1), m, atol=1e-6)\n"
            )
        return {
            "id": f"v9_wmod_{idx:04d}",
            "pillar": "rsi_and_weight_surgery",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
        }

    # variant == 7: Exact IEEE `bfloat16` / `float16` ULP & Round-to-Nearest-Even Stagnation Math
    exp_pow = rng.choice([0, 1, 2, 3])
    w_val = float(2 ** exp_pow) * rng.choice([1.0, 1.25, 1.5])
    ulp_bf16 = (2.0 ** exp_pow) * (2.0 ** -7)
    ulp_fp16 = (2.0 ** exp_pow) * (2.0 ** -10)
    half_ulp_bf16 = 0.5 * ulp_bf16
    prompt = (
        f"A weight parameter has exact binary value `w = {w_val}` (which lies in `[2^{exp_pow}, 2^{exp_pow + 1})`).\n"
        f"1. Compute its `bfloat16` (7 fraction bits) unit-in-last-place `ulp_bf16 = 2^{exp_pow} * 2^-7`.\n"
        f"2. Compute its `float16` (10 fraction bits) unit-in-last-place `ulp_fp16 = 2^{exp_pow} * 2^-10`.\n"
        f"3. Under pure `bfloat16` SGD without FP32 master weights, any update `|lr * g| < half_ulp_bf16` (`0.5 * ulp_bf16`) can round back to `w`; report `half_ulp_bf16`.\n"
        f'Return ONLY JSON `{{"ulp_bf16": <float>, "ulp_fp16": <float>, "half_ulp_bf16": <float>}}`.'
    )
    think = (
        f"[L1] Since `{w_val}` lies in `[2^{exp_pow}, 2^{exp_pow + 1})`, its unbiased exponent is `E = {exp_pow}`.\n"
        f"[L2] `ulp_bf16 = 2^{exp_pow} / 128 = {ulp_bf16}`, `ulp_fp16 = 2^{exp_pow} / 1024 = {ulp_fp16}`, and `half_ulp_bf16 = 0.5 * {ulp_bf16} = {half_ulp_bf16}`."
    )
    ans = json.dumps({"ulp_bf16": ulp_bf16, "ulp_fp16": ulp_fp16, "half_ulp_bf16": half_ulp_bf16})
    return {
        "id": f"v9_wmod_{idx:04d}",
        "pillar": "rsi_and_weight_surgery",
        "images": [],
        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}</think>\n{ans}"}],
    }


def make_v9_zero_prior_discovery_item(rng: random.Random, idx: int) -> dict[str, Any]:
    """Train Zero-Prior-Knowledge First-Principles Derivation + Cline/Hermes Environment Discovery."""
    variant = idx % 7

    if variant == 0:
        # Invented non-commutative operator `x ◈ y = c1*x - c2*y + x*y`
        c1, c2 = rng.randint(2, 4), rng.randint(1, 3)
        x0, y0, z0 = rng.randint(2, 5), rng.randint(2, 5), rng.randint(1, 3)
        xy = c1 * x0 - c2 * y0 + x0 * y0
        yz = c1 * y0 - c2 * z0 + y0 * z0
        left_assoc = c1 * xy - c2 * z0 + xy * z0
        right_assoc = c1 * x0 - c2 * yz + x0 * yz
        prompt = (
            f"Define a novel binary operation `◈` on integers by:\n"
            f"  `a ◈ b = {c1}*a - {c2}*b + a*b`\n"
            f"Without assuming any standard algebraic laws, compute `left_val = ({x0} ◈ {y0}) ◈ {z0}`, `right_val = {x0} ◈ ({y0} ◈ {z0})`, and `associative_here = (left_val == right_val)`.\n"
            f'Return ONLY JSON `{{"left_val": <int>, "right_val": <int>, "associative_here": <bool>}}`.'
        )
        think = (
            f"[AXIOMS] Use ONLY `a ◈ b = {c1}*a - {c2}*b + a*b`.\n"
            f"[PROBE] Step 1 (`left_val`): `{x0} ◈ {y0} = {c1}*{x0} - {c2}*{y0} + {x0}*{y0} = {c1*x0} - {c2*y0} + {x0*y0} = {xy}`. "
            f"Then `{xy} ◈ {z0} = {c1}*{xy} - {c2}*{z0} + {xy}*{z0} = {c1*xy} - {c2*z0} + {xy*z0} = {left_assoc}`.\n"
            f"[AUDIT] Step 2 (`right_val`): `{y0} ◈ {z0} = {c1}*{y0} - {c2}*{z0} + {y0}*{z0} = {yz}`. "
            f"Then `{x0} ◈ {yz} = {c1}*{x0} - {c2}*{yz} + {x0}*{yz} = {right_assoc}` -> `associative_here = {left_assoc == right_assoc}`."
        )
        ans = json.dumps({"left_val": left_assoc, "right_val": right_assoc, "associative_here": (left_assoc == right_assoc)})
        return {
            "id": f"v9_zp_{idx:04d}",
            "pillar": "zero_prior_discovery",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}</think>\n{ans}"}],
        }

    if variant == 1:
        # Multi-turn Cline Zero-Prior Codebase Discovery (`read_files` -> inspect unfamiliar DSL -> fix!)
        mod_path = rng.choice(["runtime/flux_gate.py", "engine/pulse_budget.py", "kernels/rank_router.py"])
        k_const = rng.randint(4, 9)
        tools_spec = json.dumps([
            {"name": "read_files", "parameters": {"type": "object", "properties": {"paths": {"type": "array", "items": {"type": "string"}}}, "required": ["paths"]}},
            {"name": "run_commands", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
        ])
        msgs = [
            {"role": "system", "content": f"You are an autonomous agent in Cline. When encountering unfamiliar local modules, read the source before answering.\n<tools>\n{tools_spec}\n</tools>"},
            {"role": "user", "content": f"What exact integer does `compute_stride_quota(12)` return in `{mod_path}`?"},
            {
                "role": "assistant",
                "content": (
                    f"<think>\n[ZERO-PRIOR] `{mod_path}` is an unfamiliar workspace file — never guess its formula. Read `{mod_path}` first.</think>\n"
                    f"<tool_call>\n<function=read_files>\n<parameter=paths>\n[\"{mod_path}\"]\n</parameter>\n</function>\n</tool_call>"
                ),
            },
            {
                "role": "tool",
                "content": f"def compute_stride_quota(n: int) -> int:\n    # Custom quota: floor(n / 3) * {k_const} + (n % 2)\n    return (n // 3) * {k_const} + (n % 2)\n",
            },
            {
                "role": "assistant",
                "content": (
                    f"<think>\n[VERIFY] From `{mod_path}`, `compute_stride_quota(12) = (12 // 3) * {k_const} + (12 % 2) = 4 * {k_const} + 0 = {4 * k_const}`.</think>\n"
                    f"`compute_stride_quota(12)` in `{mod_path}` returns **`{4 * k_const}`** (`(12 // 3) * {k_const} + (12 % 2) = 4 * {k_const} + 0`)."
                ),
            },
        ]
        return {"id": f"v9_zp_{idx:04d}", "pillar": "zero_prior_discovery", "images": [], "messages": msgs}

    if variant == 2:
        # Multi-turn Cline + `rpr-seal-rsi` MCP Self-Rewrite Tool Orchestration!
        tools_mcp = json.dumps([
            {"name": "inspect_model_weights", "parameters": {"type": "object", "properties": {"layer_filter": {"type": "string"}}, "required": ["layer_filter"]}},
            {
                "name": "seal_outer_loop_self_rewrite",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_layer": {"type": "integer"},
                        "svf_scale": {"type": "number"},
                        "nullspace_protect": {"type": "boolean"},
                        "verify_code": {"type": "string"},
                    },
                    "required": ["target_layer", "svf_scale", "nullspace_protect", "verify_code"],
                },
            },
        ])
        lyr = rng.choice([26, 30, 32, 34])
        msgs = [
            {"role": "system", "content": f"You are running in Cline with the `rpr-seal-rsi` MCP server. Always inspect weights first and use `nullspace_protect=true` when running outer-loop SEAL copy rewrites.\n<tools>\n{tools_mcp}\n</tools>"},
            {"role": "user", "content": f"Inspect layer `{lyr}` weights and run a gated outer-loop SEAL copy test with `svf_scale=1.04` and nullspace protection enabled."},
            {
                "role": "assistant",
                "content": (
                    f"<think>\n[STEP 1] Inspect layer `{lyr}` spectral/Frobenius statistics before proposing a gated SEAL copy edit.</think>\n"
                    f"<tool_call>\n<function=inspect_model_weights>\n<parameter=layer_filter>\nlayers.{lyr}.\n</parameter>\n</function>\n</tool_call>"
                ),
            },
            {
                "role": "tool",
                "content": json.dumps({"layer": lyr, "down_proj_shape": [2048, 5632], "fro_norm": 41.82, "dtype": "bfloat16"}),
            },
            {
                "role": "assistant",
                "content": (
                    f"<think>\n[STEP 2] Layer `{lyr}` is healthy (`bfloat16`, `fro_norm=41.82`). Invoke `seal_outer_loop_self_rewrite` on a sandboxed copy with `nullspace_protect=true` so base capabilities cannot regress.</think>\n"
                    f"<tool_call>\n<function=seal_outer_loop_self_rewrite>\n<parameter=target_layer>\n{lyr}\n</parameter>\n<parameter=svf_scale>\n1.04\n</parameter>\n<parameter=nullspace_protect>\ntrue\n</parameter>\n<parameter=verify_code>\nassert 2 + 2 == 4\n</parameter>\n</function>\n</tool_call>"
                ),
            },
        ]
        return {"id": f"v9_zp_{idx:04d}", "pillar": "zero_prior_discovery", "images": [], "messages": msgs}

    if variant == 3:
        # Epistemic Humility / Underdetermined System Detection
        k1 = rng.randint(3, 7)
        rhs = rng.randint(12, 30)
        prompt = (
            f"Given ONLY the single constraint `{k1}*a + {2*k1}*b = {k1*rhs}` over real numbers `(a, b)`:\n"
            f"1. Is `a` uniquely determined (`a_determined`: bool)?\n"
            f"2. Is the linear combination `L = a + 2*b` uniquely determined (`l_determined`: bool), and what is `l_value`?\n"
            f'Return ONLY JSON `{{"a_determined": <bool>, "l_determined": <bool>, "l_value": <int>}}`.'
        )
        think = (
            f"[AXIOMS] Divide `{k1}*a + {2*k1}*b = {k1*rhs}` by `{k1}` -> `a + 2*b = {rhs}`.\n"
            f"[AUDIT] One equation in two unknowns leaves 1 free degree of freedom (`a = {rhs} - 2*b` varies freely with `b`), so `a_determined = false`! "
            f"However, `L = a + 2*b` is invariant and equals `{rhs}` identically (`l_determined = true`)."
        )
        ans = json.dumps({"a_determined": False, "l_determined": True, "l_value": rhs})
        return {
            "id": f"v9_zp_{idx:04d}",
            "pillar": "zero_prior_discovery",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}</think>\n{ans}"}],
        }

    if variant == 4:
        # Cline `rpr_pyramid_solve` MCP Delegation on High-Complexity Changing-Premise Problem
        tools_rpr = json.dumps([
            {"name": "rpr_pyramid_solve", "parameters": {"type": "object", "properties": {"problem": {"type": "string"}, "new_evidence": {"type": "string"}}, "required": ["problem"]}},
        ])
        prob_s = f"Allocate {rng.choice([120, 180, 240])} MW across 6 candidate turbines under changing grid-sync constraints."
        ev_s = "Grid authority switched active profile to HVDC_SYNC and tightened ramp ceiling by 12 ms."
        return {
            "id": f"v9_zp_{idx:04d}",
            "pillar": "zero_prior_discovery",
            "images": [],
            "messages": [
                {"role": "system", "content": f"You are in Cline with `rpr_pyramid_solve` available for multi-pass ATMS belief revision.\n<tools>\n{tools_rpr}\n</tools>"},
                {"role": "user", "content": f"Use the RPR pyramid solver tool on:\nProblem: {prob_s}\nUpdate: {ev_s}"},
                {
                    "role": "assistant",
                    "content": (
                        "<think>\nInvoke `rpr_pyramid_solve` with both `problem` and `new_evidence` so Level-1 ATMS dependency auditing runs.</think>\n"
                        f"<tool_call>\n<function=rpr_pyramid_solve>\n<parameter=problem>\n{prob_s}\n</parameter>\n<parameter=new_evidence>\n{ev_s}\n</parameter>\n</function>\n</tool_call>"
                    ),
                },
            ],
        }

    if variant == 5:
        # Unfamiliar State-Machine / Rewrite Trace from Pure Axioms (disjoint symbols from heldout!)
        a0, b0 = rng.randint(3, 8), rng.randint(2, 6)
        add_c = rng.randint(2, 5)
        # Step 1: TWIST -> (a + 2*b, a - b)
        a1, b1 = a0 + 2 * b0, a0 - b0
        # Step 2: LIFT(add_c) -> a = a + add_c, b = abs(b)
        a2, b2 = a1 + add_c, abs(b1)
        # Step 3: TWIST -> (a + 2*b, a - b)
        a3, b3 = a2 + 2 * b2, a2 - b2
        prompt = (
            f"An unfamiliar automaton maintains integer state `(X, Y)` starting at `({a0}, {b0})` under two rules:\n"
            f"- `TWIST`: simultaneously updates `(X, Y) <- (X + 2*Y, X - Y)`.\n"
            f"- `LIFT k`: updates `(X, Y) <- (X + k, |Y|)`.\n"
            f"Trace `TWIST ; LIFT {add_c} ; TWIST` step-by-step and output ONLY JSON `{{\"X\": <int>, \"Y\": <int>}}`."
        )
        think = (
            f"[AXIOMS] Trace state table cleanly:\n"
            f"  - Start: `(X, Y) = ({a0}, {b0})`\n"
            f"  - After `TWIST`: `X = {a0} + 2*{b0} = {a1}`, `Y = {a0} - {b0} = {b1}`\n"
            f"  - After `LIFT {add_c}`: `X = {a1} + {add_c} = {a2}`, `Y = |{b1}| = {b2}`\n"
            f"  - After `TWIST`: `X = {a2} + 2*{b2} = {a3}`, `Y = {a2} - {b2} = {b3}`."
        )
        return {
            "id": f"v9_zp_{idx:04d}",
            "pillar": "zero_prior_discovery",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}</think>\n" + json.dumps({"X": a3, "Y": b3})}],
        }

    # variant == 6: Frontier Research Statistics (Unbiased pass@k + 2x2 Factorial Synergy)
    if (idx // 7) % 2 == 0:
        n_tot, c_ok, k_pick = rng.choice([(8, 3, 2), (7, 2, 2), (9, 3, 3), (10, 4, 2)])
        unbiased = round(1.0 - math.comb(n_tot - c_ok, k_pick) / math.comb(n_tot, k_pick), 4)
        plugin = round(1.0 - ((1.0 - c_ok / n_tot) ** k_pick), 4)
        prompt = (
            f"Given `n = {n_tot}` independent model rollouts on a benchmark problem with `c = {c_ok}` passing unit tests, compute at `k = {k_pick}`:\n"
            f"1. `unbiased` = `round(1 - comb(n - c, k) / comb(n, k), 4)`\n"
            f"2. `plugin` = `round(1 - (1 - c / n)**k, 4)`\n"
            f'Return ONLY JSON `{{"unbiased": <float>, "plugin": <float>}}`.'
        )
        think = (
            f"[L1] `comb({n_tot} - {c_ok}, {k_pick}) = comb({n_tot - c_ok}, {k_pick}) = {math.comb(n_tot - c_ok, k_pick)}`, "
            f"`comb({n_tot}, {k_pick}) = {math.comb(n_tot, k_pick)}`.\n"
            f"[L2] `unbiased = 1 - {math.comb(n_tot - c_ok, k_pick)}/{math.comb(n_tot, k_pick)} = {unbiased}`; "
            f"`plugin = 1 - ({n_tot - c_ok}/{n_tot})^{k_pick} = {plugin}`."
        )
        return {
            "id": f"v9_zp_{idx:04d}",
            "pillar": "research_and_ablation",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}</think>\n" + json.dumps({"unbiased": unbiased, "plugin": plugin})}],
        }
    base = float(rng.randint(55, 68))
    d1, d2 = float(rng.randint(2, 6)), float(rng.randint(2, 6))
    syn = float(rng.choice([-2.5, 3.0, 4.5]))
    joint = base + d1 + d2 + syn
    prompt = (
        f"In a 2x2 ablation study of Feature_U and Feature_V:\n"
        f"- neither: `{base}`, only U: `{base + d1}`, only V: `{base + d2}`, both U+V: `{joint}`.\n"
        f'Compute `additive_expected = {base} + ({base + d1} - {base}) + ({base + d2} - {base})`, `synergy_delta = both - additive_expected`, and `is_synergistic = (synergy_delta > 0)`. Return ONLY JSON `{{"additive_expected": <float>, "synergy_delta": <float>, "is_synergistic": <bool>}}`.'
    )
    think = f"[L1] `d_U = {d1}`, `d_V = {d2}` -> `additive_expected = {base + d1 + d2}`.\n[L2] `synergy_delta = {joint} - {base + d1 + d2} = {syn}` -> `is_synergistic = {syn > 0}`."
    return {
        "id": f"v9_zp_{idx:04d}",
        "pillar": "research_and_ablation",
        "images": [],
        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}</think>\n" + json.dumps({"additive_expected": base + d1 + d2, "synergy_delta": syn, "is_synergistic": (syn > 0)})}],
    }


def make_v9_algorithmic_invariant_item(rng: random.Random, idx: int) -> dict[str, Any]:
    """Cure 1-pass pointer/index/sign slips by teaching 2-element base-case invariant traces!"""
    variant = idx % 4

    if variant == 0:
        # Rightmost passing index (`nums[i] <= cap`) — cures `lo = mid + 1` vs `lo = mid` off-by-1!
        base_fn = rng.choice(["rightmost_at_most", "last_feasible_slot", "upper_feasible_idx"])
        fn = f"{base_fn}_{idx % 17}" if idx >= 12 else base_fn
        miss = rng.choice([-1, -1, -1])
        prompt = (
            f"Implement `{fn}(vals: list[int], cap: int) -> int` in Python returning the largest index `i` in sorted ascending `vals` with `vals[i] <= cap`, or `{miss}` if `vals` is empty or `vals[0] > cap`. Must run in `O(log n)` without `bisect`."
        )
        think = (
            "[INVARIANT] Maintain `vals[lo] <= cap` and `i_star in [lo, hi]`.\n"
            "[2-ELEMENT TRACE] Test `vals = [3, 10], cap = 5` (`lo=0, hi=1`):\n"
            "  - Use upper-mid `mid = (lo + hi + 1) // 2 = 1`.\n"
            "  - If `vals[mid] <= cap`: `mid` itself is valid, so set `lo = mid` (NEVER `mid + 1`, which would skip `mid`!).\n"
            "  - Else (`vals[1]=10 > 5`): `mid` is invalid, so set `hi = mid - 1 = 0`.\n"
            "  - Loop ends at `lo == hi == 0`, returning `0` (`vals[0] = 3 <= 5`) — exact!"
        )
        code = (
            f"def {fn}(vals: list[int], cap: int) -> int:\n"
            f"    if not vals or vals[0] > cap:\n"
            f"        return -1\n"
            f"    lo, hi = 0, len(vals) - 1\n"
            f"    while lo < hi:\n"
            f"        mid = (lo + hi + 1) // 2\n"
            f"        if vals[mid] <= cap:\n"
            f"            lo = mid\n"
            f"        else:\n"
            f"            hi = mid - 1\n"
            f"    return lo"
        )
        test = (
            f"assert {fn}([], 4) == -1\nassert {fn}([7, 9], 4) == -1\n"
            f"assert {fn}([3, 10], 5) == 0\nassert {fn}([2, 4, 4, 8], 4) == 2\nassert {fn}([1, 2, 3], 9) == 2\n"
        )
        return {
            "id": f"v9_algo_{idx:04d}",
            "pillar": "diagnostic_debugging",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
        }

    if variant == 1:
        # 3-Color Directed Cycle Detection — cures `state == 1` vs `state == 2` boolean inversion!
        base_fn = rng.choice(["detect_directed_loop", "graph_has_backedge", "contains_task_cycle"])
        fn = f"{base_fn}_{idx % 17}" if idx >= 12 else base_fn
        prompt = (
            f"Implement `{fn}(adj: dict[str, list[str]]) -> bool` in Python returning `True` iff directed graph `adj` contains a cycle (handling diamond DAGs as `False` and disconnected components / sink-only neighbor nodes cleanly)."
        )
        think = (
            "[INVARIANT] 3-state DFS coloring `color: dict[str, int]`:\n"
            "  - `0`: unvisited\n"
            "  - `1`: currently on active recursion stack -> seeing `color.get(v, 0) == 1` is a BACK-EDGE = CYCLE (`return True`!)\n"
            "  - `2`: completely explored cycle-free -> seeing `color.get(v, 0) == 2` is a safe cross/forward edge (`return False`!)\n"
            "[TRACE] Diamond `A->B, A->C, B->D, C->D`: `D` finishes (`color['D']=2`), then `C->D` sees `2 -> False` (no false positive!)."
        )
        code = (
            f"def {fn}(adj: dict[str, list[str]]) -> bool:\n"
            f"    color: dict[str, int] = {{}}\n\n"
            f"    def dfs(u: str) -> bool:\n"
            f"        st = color.get(u, 0)\n"
            f"        if st == 1:\n"
            f"            return True\n"
            f"        if st == 2:\n"
            f"            return False\n"
            f"        color[u] = 1\n"
            f"        for nxt in adj.get(u, ()):\n"
            f"            if dfs(nxt):\n"
            f"                return True\n"
            f"        color[u] = 2\n"
            f"        return False\n\n"
            f"    return any(dfs(node) for node in list(adj.keys()))"
        )
        test = (
            f"assert {fn}({{'A': ['B', 'C'], 'B': ['D'], 'C': ['D']}}) is False\n"
            f"assert {fn}({{'A': ['B'], 'B': ['A']}}) is True\n"
            f"assert {fn}({{'S': ['S']}}) is True\n"
        )
        return {
            "id": f"v9_algo_{idx:04d}",
            "pillar": "diagnostic_debugging",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
        }

    if variant == 2:
        # Numerically Stable Softmax / LogSumExp — cures algebraic sign loops!
        base_fn = rng.choice(["safe_boltzmann_probs", "numerically_stable_probs", "shifted_softmax_vec"])
        fn = f"{base_fn}_{idx % 17}" if idx >= 12 else base_fn
        hi_v = rng.choice([1100, 1200, 1400])
        prompt = (
            f"Implement `{fn}(logits: list[float]) -> list[float]` in pure Python (`math` module only) returning softmax probabilities without overflow or underflow on extreme inputs like `[{hi_v}.0, {hi_v - 1}.0]` or `[-{hi_v}.0, -{hi_v}.0]`. Return `[]` on empty input."
        )
        think = (
            "[L1] Identity: subtracting scalar `c = max(logits)` multiplies both numerator and denominator by `e^-c`, leaving probabilities strictly unchanged!\n"
            "[L2] Since `x - c <= 0` for every `x`, `math.exp(x - c)` lies in `(0, 1]` with `math.exp(c - c) == 1.0`, so `denom >= 1.0` — zero overflow, zero division by zero, no extra factors needed!"
        )
        code = (
            f"import math\n\n"
            f"def {fn}(logits: list[float]) -> list[float]:\n"
            f"    if not logits:\n"
            f"        return []\n"
            f"    c = max(logits)\n"
            f"    exps = [math.exp(x - c) for x in logits]\n"
            f"    total = sum(exps)\n"
            f"    return [e / total for e in exps]"
        )
        test = (
            f"assert {fn}([]) == []\n"
            f"p = {fn}([1200.0, 1200.0])\nassert abs(p[0] - 0.5) < 1e-9 and abs(p[1] - 0.5) < 1e-9\n"
            f"p2 = {fn}([-1200.0, -1201.0])\nassert abs(sum(p2) - 1.0) < 1e-9 and p2[0] > p2[1]\n"
        )
        return {
            "id": f"v9_algo_{idx:04d}",
            "pillar": "diagnostic_debugging",
            "images": [],
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}</think>\n```python\n{code}\n```"}],
            "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
        }

    # variant == 3: Shortest Subarray with Sum >= Target (with negative numbers!) — cures variable-name & pop-end bugs!
    base_fn = rng.choice(["shortest_window_at_least", "min_subarray_ge_target", "compact_sum_window"])
    fn = f"{base_fn}_{idx % 17}" if idx >= 12 else base_fn
    prompt = (
        f"Implement `{fn}(arr: list[int], target: int) -> int` in Python returning the length of the shortest non-empty contiguous subarray with sum `>= target` (where `arr` may contain negative integers!), or `-1` if none exists, in `O(n)` time."
    )
    think = (
        "[L1] Negative numbers break two-pointer monotonicity; use prefix sums `pref` of length `n + 1` (`pref[0] = 0`) + monotonic increasing `deque` `dq` of indices.\n"
        "[TRACE] For each `(j, cur_p)` in `enumerate(pref)`:\n"
        "  1. Pop from LEFT while `dq and cur_p - pref[dq[0]] >= target`: `best = min(best, j - dq.popleft())`.\n"
        "  2. Pop from RIGHT (`dq.pop()`) while `dq and cur_p <= pref[dq[-1]]` (dominated larger prefix at earlier index!).\n"
        "  3. `dq.append(j)`. (Check variable name `cur_p` matches `for j, cur_p in enumerate(pref)`!)."
    )
    code = (
        f"from collections import deque\n\n"
        f"def {fn}(arr: list[int], target: int) -> int:\n"
        f"    pref = [0]\n"
        f"    for v in arr:\n"
        f"        pref.append(pref[-1] + v)\n"
        f"    dq: deque[int] = deque()\n"
        f"    best = len(arr) + 1\n"
        f"    for j, cur_p in enumerate(pref):\n"
        f"        while dq and cur_p - pref[dq[0]] >= target:\n"
        f"            best = min(best, j - dq.popleft())\n"
        f"        while dq and cur_p <= pref[dq[-1]]:\n"
        f"            dq.pop()\n"
        f"        dq.append(j)\n"
        f"    return best if best <= len(arr) else -1"
    )
    test = (
        f"assert {fn}([2, -1, 2], 3) == 3\n"
        f"assert {fn}([1, -3, 4, 2], 5) == 2\n"
        f"assert {fn}([1, 2], 4) == -1\n"
    )
    return {
        "id": f"v9_algo_{idx:04d}",
        "pillar": "diagnostic_debugging",
        "images": [],
        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{think}</think>\n```python\n{code}\n```"}],
        "verifier": {"kind": "python_code", "test": test, "forbid_tool_call": True},
    }


def make_v9_orpo_pair(rng: random.Random, idx: int) -> dict[str, Any]:
    """Contrastive ORPO pairs targeting exact SVD row slicing, TIES sign consensus, binary search, 3-color DFS, & zero-prior discovery."""
    kind = idx % 5
    if kind == 0:
        row = make_v9_weight_surgery_mastery_item(rng, 0 + 8 * (idx // 5))
        chosen = row["messages"][-1]["content"]
        # Rejected commits the exact column-slice bug `vt[:, :r]`!
        rejected = chosen.replace("vt[:r, :]", "vt[:, r : r + 1]").replace("NEVER slice axis 1", "slice column r")
        return {"id": f"v9_orpo_{idx:04d}", "messages": row["messages"][:-1], "chosen": chosen, "rejected": rejected}
    if kind == 1:
        row = make_v9_algorithmic_invariant_item(rng, 0 + 4 * (idx // 5))
        chosen = row["messages"][-1]["content"]
        rejected = chosen.replace("lo = mid\n        else:\n            hi = mid - 1", "lo = mid + 1\n        else:\n            hi = mid")
        return {"id": f"v9_orpo_{idx:04d}", "messages": row["messages"][:-1], "chosen": chosen, "rejected": rejected}
    if kind == 2:
        row = make_v9_algorithmic_invariant_item(rng, 1 + 4 * (idx // 5))
        chosen = row["messages"][-1]["content"]
        rejected = chosen.replace("if st == 1:\n            return True\n        if st == 2:\n            return False", "if st == 1:\n            return False\n        if st == 2:\n            return True")
        return {"id": f"v9_orpo_{idx:04d}", "messages": row["messages"][:-1], "chosen": chosen, "rejected": rejected}
    if kind == 3:
        row = make_v9_algorithmic_invariant_item(rng, 2 + 4 * (idx // 5))
        chosen = row["messages"][-1]["content"]
        rejected = (
            "<think>\nSubtract c = max(logits)... wait multiply by exp(-c) -> exp(x - 2c)... wait no exp(x - c + c)... "
            "let me re-derive exp(x - c) * exp(-c) again and again...</think>\n```python\ndef broken(logits):\n    return logits\n```"
        )
        return {"id": f"v9_orpo_{idx:04d}", "messages": row["messages"][:-1], "chosen": chosen, "rejected": rejected}
    # kind == 4: Zero-prior humility vs hallucinating unknown file contents
    row = make_v9_zero_prior_discovery_item(rng, 1 + 7 * (idx // 5))
    return {
        "id": f"v9_orpo_{idx:04d}",
        "messages": row["messages"][:2],
        "chosen": row["messages"][2]["content"],
        "rejected": "<think>\nI can probably guess `compute_stride_quota(12)` returns `36` without reading the file.</think>\n`compute_stride_quota(12)` returns `36`.",
    }


def _assert_zero_heldout_leakage(generated_rows: list[dict[str, Any]], eval_paths: list[Path]) -> None:
    """Enforce zero target-function identifier overlap and zero 10-word user-prompt overlap with held-out benchmarks."""
    import re

    eval_fn_names: set[str] = set()
    eval_10grams: set[tuple[str, ...]] = set()
    for p in eval_paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            u_txt = " ".join(str(m.get("content", "")) for m in r.get("messages", []) if m.get("role") == "user")
            for fn in re.findall(r"(?:Implement|Write|def)\s+`?([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", u_txt):
                if len(fn) >= 6:
                    eval_fn_names.add(fn)
            words = u_txt.lower().split()
            for j in range(max(0, len(words) - 10 + 1)):
                eval_10grams.add(tuple(words[j : j + 10]))

    for row in generated_rows:
        txt = "\n".join(str(m.get("content", "")) for m in row.get("messages", []))
        for fn in eval_fn_names:
            assert fn not in txt, f"Held-out function name leakage detected: {fn} in {row.get('id')}"
        u_txt = " ".join(str(m.get("content", "")) for m in row.get("messages", []) if m.get("role") == "user")
        words = u_txt.lower().split()
        for j in range(max(0, len(words) - 10 + 1)):
            gram = tuple(words[j : j + 10])
            assert gram not in eval_10grams, f"Held-out 10-gram leakage in {row.get('id')}: {' '.join(gram)}"


def build_v9_corpus() -> dict[str, Any]:
    V9_CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261955)

    # 1. Build strictly held-out 24-task Zero-Prior + Weight-Mod benchmark FIRST
    zp24 = build_heldout_zero_prior_24()
    zp24_path = V9_CORPUS_DIR / "heldout_zero_prior_24.jsonl"
    zp24_path.write_text("\n".join(json.dumps(t) for t in zp24) + "\n", encoding="utf-8")

    # 2. Load stratified `v8` 8-pillar + Hermes + Vision + `[EFFORT: FLASH]` replay
    v8_sft = [json.loads(l) for l in (V8_CORPUS_DIR / "train_v8_sft.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    vis_rows = [r for r in v8_sft if r.get("images")]
    non_vis_v8 = [r for r in v8_sft if not r.get("images")]
    rng.shuffle(non_vis_v8)
    replay_v8 = vis_rows + non_vis_v8[:330]

    # 3. Generate new `v9` Weight Surgery, Zero-Prior Discovery, & Algorithmic Base-Case items
    wmod_items = [make_v9_weight_surgery_mastery_item(rng, i) for i in range(192)]
    zp_items = [make_v9_zero_prior_discovery_item(rng, i) for i in range(160)]
    algo_items = [make_v9_algorithmic_invariant_item(rng, i) for i in range(140)]

    # Verify 100% of executable Python code items in `python3 -I` BEFORE training!
    code_candidates = [r for r in (wmod_items + algo_items) if r.get("verifier", {}).get("kind") == "python_code"]
    n_verified = _verify_executable_code_rows(code_candidates)
    assert n_verified == len(code_candidates) and n_verified >= 200, f"Expected >=200 verified code items, got {n_verified}"

    all_new = wmod_items + zp_items + algo_items
    rng.shuffle(all_new)

    val_new = all_new[:52]
    train_new = all_new[52:]

    v8_val = [json.loads(l) for l in (V8_CORPUS_DIR / "val_v8_joint.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    val_joint = _deduplicate_rows(v8_val[:44] + val_new)
    train_sft = _deduplicate_rows(replay_v8 + train_new)
    rng.shuffle(train_sft)

    # 4. Build ORPO contrastive pairs (`v8` replay + `v9` weight-surgery/algo/zero-prior pairs)
    v8_orpo = [json.loads(l) for l in (V8_CORPUS_DIR / "orpo_v8_pairs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rng.shuffle(v8_orpo)
    v9_orpo_new = [make_v9_orpo_pair(rng, i) for i in range(130)]
    orpo_all = v8_orpo[:70] + v9_orpo_new
    rng.shuffle(orpo_all)

    # 5. Build 64 execution-verified RLVR episodes
    v8_rl = [json.loads(l) for l in (V8_CORPUS_DIR / "rlvr_v8_hermes_48.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rlvr_new: list[dict[str, Any]] = []
    for idx_r in range(44):
        src = (wmod_items[idx_r] if idx_r % 2 == 0 else algo_items[idx_r])
        tc_s = src.get("verifier", {}).get("test")
        if tc_s:
            rlvr_new.append({
                "id": f"rlvr9_{idx_r:03d}",
                "kind": "python_code",
                "messages": src["messages"][:-1],
                "verifier": {"kind": "python_code", "test": tc_s, "forbid_tool_call": True},
                "reference_answer": src["messages"][-1]["content"],
            })
    rlvr_64 = (rlvr_new + v8_rl[:28])[:64]
    rng.shuffle(rlvr_64)

    # 6. Strict decontamination against ALL held-out benchmarks (`pilot_10`, `benchmark_100`, `heldout_v5_rsi_56`, AND `heldout_zero_prior_24`!)
    assert_decontaminated(train_sft + val_joint + orpo_all + rlvr_64 + zp24)
    _assert_zero_heldout_leakage(
        wmod_items + zp_items + algo_items + v9_orpo_new + rlvr_new,
        [
            V5_CORPUS_DIR / "heldout_v5_rsi_56.jsonl",
            zp24_path,
        ],
    )

    (V9_CORPUS_DIR / "train_v9_sft.jsonl").write_text("\n".join(json.dumps(r) for r in train_sft) + "\n", encoding="utf-8")
    (V9_CORPUS_DIR / "val_v9_joint.jsonl").write_text("\n".join(json.dumps(r) for r in val_joint) + "\n", encoding="utf-8")
    (V9_CORPUS_DIR / "orpo_v9_pairs.jsonl").write_text("\n".join(json.dumps(r) for r in orpo_all) + "\n", encoding="utf-8")
    (V9_CORPUS_DIR / "rlvr_v9_episodes_64.jsonl").write_text("\n".join(json.dumps(r) for r in rlvr_64) + "\n", encoding="utf-8")

    summary = {
        "train_v9_sft": len(train_sft),
        "val_v9_joint": len(val_joint),
        "orpo_v9_pairs": len(orpo_all),
        "rlvr_v9_episodes": len(rlvr_64),
        "heldout_zero_prior_24": len(zp24),
        "python_unit_verified_count": n_verified,
        "vision_items_preserved": len(vis_rows),
    }
    (V9_CORPUS_DIR / "corpus_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("[v9 Corpus Built & Decontaminated]", json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    build_v9_corpus()
