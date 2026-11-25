"""Unit tests for RPR v12 Deep-Thinker architecture (`v8_arch.py`) and controller (`rpr_live_controller.py`)."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from rpr_live_controller import (
    _compare_two_python_fns_in_sandbox,
    build_ast_signature_smoke_harness,
    run_live_rpr_pyramid_turn,
)
from v8_arch import LatentReasoningRecurrence, ThoughtAnchorHeadTemp


def test_v12_latent_recurrence_entry_inject_zero_step0_and_grads() -> None:
    mx.random.seed(42)
    mod = LatentReasoningRecurrence(hidden_size=64, latent_ffn_dim=96, num_loops=4, entry_inject=True)
    mod.apply(lambda x: x.astype(mx.bfloat16) if mx.issubdtype(x.dtype, mx.floating) else x)
    h = mx.random.normal((1, 8, 64)).astype(mx.bfloat16)
    srcs = [mx.random.normal((1, 8, 64)).astype(mx.bfloat16) for _ in range(4)]

    mod.eval()
    out0 = mod(h, srcs)
    mx.eval(out0)
    diff0 = float(mx.max(mx.abs(out0.astype(mx.float32) - h.astype(mx.float32))).item())
    assert diff0 == 0.0, f"Expected exact 0.0 Step-0 identity, got {diff0}"

    mod.train()

    def _loss_fn(m: nn.Module) -> mx.array:
        return mx.mean(m(h, srcs).astype(mx.float32) ** 2)

    _, grads = nn.value_and_grad(mod, _loss_fn)(mod)
    mx.eval(grads)
    assert float(mx.max(mx.abs(grads["loop_scales"])).item()) > 0.0


def test_v12_thought_anchor_head_temp_zero_step0_identity() -> None:
    ta = ThoughtAnchorHeadTemp(num_heads=8)
    q = mx.random.normal((1, 8, 12, 32)).astype(mx.bfloat16)
    q_scaled = ta.scale_queries(q)
    mx.eval(q_scaled)
    diff = float(mx.max(mx.abs(q_scaled.astype(mx.float32) - q.astype(mx.float32))).item())
    assert diff == 0.0
    ta.log_temp = mx.array([0.2, -0.2, 0.0, 0.1, -0.1, 0.3, -0.3, 0.0], dtype=mx.bfloat16)
    q_mod = ta.scale_queries(q)
    mx.eval(q_mod)
    assert float(mx.max(mx.abs(q_mod.astype(mx.float32) - q.astype(mx.float32))).item()) > 0.01


def test_differential_python_sandbox_catches_missing_vt_bug() -> None:
    spec = "Implement `modulate_sv(W: np.ndarray, gains: np.ndarray) -> np.ndarray` computing U @ diag(S * gains) @ Vt."
    buggy_a = (
        "import numpy as np\n"
        "def modulate_sv(W: np.ndarray, gains: np.ndarray) -> np.ndarray:\n"
        "    U, S, Vt = np.linalg.svd(W.astype(np.float64), full_matrices=False)\n"
        "    return U * (S * gains)[None, :]\n"
    )
    exact_b = (
        "import numpy as np\n"
        "def modulate_sv(W: np.ndarray, gains: np.ndarray) -> np.ndarray:\n"
        "    U, S, Vt = np.linalg.svd(W.astype(np.float64), full_matrices=False)\n"
        "    return (U * (S * gains)[None, :]) @ Vt\n"
    )
    h = build_ast_signature_smoke_harness(exact_b, spec)
    assert h is not None
    b_ok, agree, diff_rep = _compare_two_python_fns_in_sandbox(buggy_a, exact_b, h)
    assert b_ok is True
    assert agree is False
    assert "DIFF_DETECTED" in diff_rep


def test_spurious_tool_leak_repaired_on_json_only_spec() -> None:
    calls = {"n": 0}

    def _mock_gen(msgs, max_toks, think):
        calls["n"] += 1
        if calls["n"] == 1:
            return "<think>Try tool.</think>\n<tool_call><function=calc><parameter=x>1</parameter></function></tool_call>"
        return '<think>Compute directly: 6 * 7 = 42.</think>\n{"ans": 42}'

    out = run_live_rpr_pyramid_turn(
        [{"role": "user", "content": "Evaluate 6 * 7 and return ONLY JSON `{\"ans\": ...}`."}],
        _mock_gen,
    )
    assert "<tool_call>" not in out["text"]
    assert "repaired_spurious_tool_leak" in out["audit_note"]
