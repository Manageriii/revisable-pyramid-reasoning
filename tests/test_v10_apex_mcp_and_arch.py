"""Pure-CPU unit tests for `v10` APEX Citation Verifier, `Transformer^2` SVF Dispatch, `DGM` Archive, & Non-Mutation RPR."""

from __future__ import annotations

import json

from mcp_rpr_seal_server import handle_rpc_message
from rpr_live_controller import (
    find_inline_arithmetic_discrepancies,
    run_live_rpr_pyramid_turn,
)
from rpr_seal_runtime_ops import (
    dispatch_svf_gains_from_prompt,
    verify_citations_and_claims_op,
)
from training.verifiable_rewards import score_response


def test_verify_citations_and_claims_mcp_tool() -> None:
    sources = [
        {"id": "S1", "title": "Preview", "snippet": "Early preview measured 14.8 ms."},
        {"id": "S2", "title": "GA Spec", "snippet": "Sustained throughput is 58400 QPS."},
        {"id": "S3", "title": "Erratum", "snippet": "Corrected p99 latency is 19.2 ms at 58400 QPS."},
    ]
    good = "Verified p99 latency is 19.2 ms [S3] and throughput is 58400 QPS [S2][S3].\n{\"p99_ms\": 19.2, \"qps\": 58400}"
    res_good = verify_citations_and_claims_op(good, sources)
    assert res_good["is_verified"] is True
    assert res_good["citation_precision"] == 1.0
    assert res_good["citation_recall"] == 1.0

    bad = "Verified latency is 99.9 ms [S3] and throughput is 58400 QPS [S9]."
    res_bad = verify_citations_and_claims_op(bad, sources)
    assert res_bad["is_verified"] is False
    assert "S9" in res_bad["unknown_citations"]

    # Also test via MCP JSON-RPC `tools/call`
    rpc = handle_rpc_message(
        {
            "jsonrpc": "2.0",
            "id": 42,
            "method": "tools/call",
            "params": {
                "name": "verify_citations_and_claims",
                "arguments": {"draft_answer": good, "sources_json": json.dumps(sources)},
            },
        }
    )
    assert rpc is not None and rpc["result"]["isError"] is False
    parsed = json.loads(rpc["result"]["content"][0]["text"])
    assert parsed["is_verified"] is True


def test_transformer2_svf_prompt_dispatch() -> None:
    r_rsi = dispatch_svf_gains_from_prompt("Apply AlphaEdit nullspace SVD and SVF singular scaling to bfloat16 weights.")
    assert r_rsi["svf_dispatch_expert"] == "b0_advanced_rsi"

    r_logic = dispatch_svf_gains_from_prompt("Solve this Knights and Knaves Boolean SAT clause puzzle by backward induction.")
    assert r_logic["svf_dispatch_expert"] == "b1_complex_logic"

    r_cite = dispatch_svf_gains_from_prompt("Synthesize [S1] preprint and [S2] official erratum with inline citation tags.")
    assert r_cite["svf_dispatch_expert"] == "b2_research_citations"


def test_rpr_controller_catches_inplace_list_mutation() -> None:
    def _mock_gen(msgs, max_toks, think):
        last_u = str(msgs[-1].get("content", ""))
        if "[RPR LEVEL-1 ASSUMPTION AUDIT]" in last_u:
            assert "mutated input list" in last_u
            return "<think>Copy input first.</think>\n```python\ndef bump_list(vals: list[int]) -> list[int]:\n    out = list(vals)\n    out[0] += 1\n    return out\n```"
        return "<think>draft</think>\n```python\ndef bump_list(vals: list[int]) -> list[int]:\n    vals[0] += 1\n    return vals\n```"

    res = run_live_rpr_pyramid_turn(
        [{"role": "user", "content": "Implement `bump_list(vals: list[int]) -> list[int]` without mutating `vals`."}],
        _mock_gen,
    )
    assert res["repaired"] is True
    assert "out = list(vals)" in res["text"]


def test_rpr_controller_repairs_hallucinated_citations() -> None:
    def _mock_gen(msgs, max_toks, think):
        last_u = str(msgs[-1].get("content", ""))
        if "[CITE-CHECK AUDIT]" in last_u:
            return '<think>Use valid [S2].</think>\nLatency is 19.2 ms [S2].\n{"ms": 19.2}'
        return '<think>draft</think>\nLatency is 19.2 ms [S9].\n{"ms": 19.2}'

    prompt = "Using [S1] (14 ms preview) and [S2] (19.2 ms official erratum), cite inline [S#] and output JSON:\n[S1] 14 ms\n[S2] 19.2 ms"
    res = run_live_rpr_pyramid_turn([{"role": "user", "content": prompt}], _mock_gen)
    assert res["repaired"] is True
    assert "[S2]" in res["text"] and "[S9]" not in res["text"]


def test_citation_grounded_verifiable_reward() -> None:
    sc_ok = score_response(
        '<think>Check S2 and S3.</think>\nThroughput is 58400 QPS [S2] at 19.2 ms [S3].\n{"p99_ms": 19.2, "qps": 58400}',
        prompt_text="Synthesize sources.",
        kind="citation_grounded",
        verifier={
            "required_citations": ["[S2]", "[S3]"],
            "valid_citations": ["[S1]", "[S2]", "[S3]"],
            "expected": {"p99_ms": 19.2, "qps": 58400},
            "forbid_tool_call": True,
        },
    )
    assert sc_ok["correct"] is True

    sc_halluc = score_response(
        '<think>Check.</think>\nThroughput is 58400 QPS [S2][S8].\n{"p99_ms": 19.2, "qps": 58400}',
        prompt_text="Synthesize sources.",
        kind="citation_grounded",
        verifier={
            "required_citations": ["[S2]"],
            "valid_citations": ["[S1]", "[S2]", "[S3]"],
            "expected": {"p99_ms": 19.2, "qps": 58400},
            "forbid_tool_call": True,
        },
    )
    assert sc_halluc["correct"] is False


def test_rpr_inline_arithmetic_ast_verifier_and_archive_citecheck() -> None:
    slips = find_inline_arithmetic_discrepancies("Apply 54 * 4 + 31 = 217 and 360.0 * 8000 * 0.10 = 28,800.0 while 2^1 * 2^-7 = 0.015625.")
    assert any("247" in s for s in slips)
    assert any("288000" in s for s in slips)
    # Must never evaluate bitwise XOR `^` as arithmetic!
    assert not any("2^-7" in s for s in slips)

    # Archive `[DOC-XXXX]` CiteCheck formatting audit when draft emitted bare JSON without inline brackets
    def _mock_doc_gen(msgs, max_toks, think):
        last_u = str(msgs[-1].get("content", ""))
        if "[CITE-CHECK FORMAT AUDIT]" in last_u:
            return (
                '<think>Add inline [DOC-0001] and [DOC-0002].</think>\n'
                'Kernel KRN-A is active [DOC-0001] with throughput 120 [DOC-0002].\n'
                '{"kernel": "KRN-A", "effective_throughput": 120, "authoritative_docs": ["DOC-0001", "DOC-0002"]}'
            )
        return (
            '<think>[DOC-0001] KRN-A, [DOC-0002] 120.</think>\n'
            '{"kernel": "KRN-A", "effective_throughput": 120, "authoritative_docs": ["DOC-0001", "DOC-0002"]}'
        )

    arch_prompt = (
        "Audit `[DOC-0001]` .. `[DOC-0010]`. Cite authoritative IDs inline (e.g. `[DOC-XXXX]`) and end with JSON.\n"
        "=== BEGIN ARCHIVE ===\n[DOC-0001] KRN-A\n[DOC-0002] 120\n=== END ARCHIVE ===\nSynthesize inline citations and JSON."
    )
    res = run_live_rpr_pyramid_turn([{"role": "user", "content": arch_prompt}], _mock_doc_gen)
    assert res["repaired"] is True
    assert "[DOC-0001]" in res["text"] and "[DOC-0002]" in res["text"]

