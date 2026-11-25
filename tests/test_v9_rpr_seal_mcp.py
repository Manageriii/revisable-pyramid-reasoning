"""Unit tests for the `v9` Zero-Leakage Live RPR Controller & `rpr-seal-rsi` MCP Server (Pure CPU)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from mcp_rpr_seal_server import handle_rpc_message
from rpr_live_controller import run_isolated_python, run_live_rpr_pyramid_turn


ROOT_DIR = Path(__file__).resolve().parent.parent


def test_mcp_initialize_and_tools_list() -> None:
    init_resp = handle_rpc_message(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}}
    )
    assert init_resp is not None
    assert init_resp["result"]["serverInfo"]["name"] == "rpr-seal-rsi"

    tools_resp = handle_rpc_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    assert tools_resp is not None
    names = {t["name"] for t in tools_resp["result"]["tools"]}
    assert names == {
        "rpr_pyramid_solve",
        "rpr_belief_audit",
        "inspect_model_weights",
        "seal_outer_loop_self_rewrite",
        "realtime_web_research",
        "verify_citations_and_claims",
    }


def test_mcp_stdio_roundtrip_subprocess() -> None:
    req_lines = "\n".join(
        [
            json.dumps({"jsonrpc": "2.0", "id": 10, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}}),
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}),
            json.dumps({"jsonrpc": "2.0", "id": 11, "method": "tools/list", "params": {}}),
        ]
    ) + "\n"
    proc = subprocess.run(
        [sys.executable, str(ROOT_DIR / "mcp_rpr_seal_server.py")],
        input=req_lines,
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    lines = [json.loads(ln) for ln in proc.stdout.strip().splitlines() if ln.strip()]
    assert len(lines) == 2
    assert lines[0]["id"] == 10
    assert lines[1]["id"] == 11 and len(lines[1]["result"]["tools"]) == 6


def test_rpr_live_controller_passthrough_on_tool_call() -> None:
    def _mock_gen(msgs, max_toks, think):
        return "<think>Read file first.</think>\n<tool_call>\n<function=read_files>\n<parameter=path>\na.py\n</parameter>\n</function>\n</tool_call>"

    out = run_live_rpr_pyramid_turn([{"role": "user", "content": "Inspect a.py"}], _mock_gen)
    assert out["passes"] == 1
    assert out["repaired"] is False
    assert out["audit_note"] == "tool_call_passthrough"


def test_rpr_live_controller_repairs_broken_python_via_sandbox() -> None:
    calls = []

    def _mock_gen(msgs, max_toks, think):
        calls.append(len(msgs))
        if "[RPR LEVEL-1 ASSUMPTION AUDIT]" in str(msgs[-1].get("content", "")):
            return "<think>Fixed NameError: use parameter x.</think>\n```python\ndef add_two(x: int) -> int:\n    return x + 2\n```"
        return "<think>draft</think>\n```python\ndef add_two(x: int) -> int:\n    return unknown_var + 2\n```"

    out = run_live_rpr_pyramid_turn(
        [{"role": "user", "content": "Implement `add_two(x: int) -> int` returning `x + 2`."}],
        _mock_gen,
    )
    assert out["repaired"] is True
    assert "return x + 2" in out["text"]
    ok, _, _ = run_isolated_python("def add_two(x): return x + 2\nassert add_two(5) == 7")
    assert ok is True
