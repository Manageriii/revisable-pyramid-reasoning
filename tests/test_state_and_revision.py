"""Deterministic unit tests for RPR state, dependency graph, revision, verifiers, and engine."""

from __future__ import annotations

import json
from pathlib import Path

from engine import (
    _verify_inline_equations,
    parse_level1_output,
    parse_level2_output,
    parse_level3_output,
    run_mode,
)
from evaluate import run_evaluation
from model import (
    GenerationStats,
    MockModelClient,
    ModelResponse,
    build_recovery_messages,
    extract_json_object,
    split_thinking,
    strip_tool_calls,
)
from revision import VerificationFindings, parse_audit_output, reconstruct
from state import Assumption, DependencyNode, PyramidState, VerifiedFact
from symbolic_critic import (
    extract_l2_bounds,
    parse_prompt_options,
    repair_l2_cross_bindings,
    verify_options_against_bounds,
)
from verifiers import (
    check_candidate_constraints,
    load_problems,
    parse_candidate_answer,
    safe_eval_arith,
    score_problem_run,
)


def test_split_thinking_and_json_extraction() -> None:
    raw = 'I think step by step.\n</think>\n```json\n{"primary": "OPT-B", "preserved": 60}\n```'
    thinking, content = split_thinking(raw)
    assert "step by step" in thinking
    assert '"OPT-B"' in content
    parsed = extract_json_object(raw)
    assert parsed == {"primary": "OPT-B", "preserved": 60}

    typo_raw = '{"assumptions": ["cpu only"], "facts": [{"id": "F1", "expr": "4 * 18 + 12", "value": 84, "depends_on": []}}]}'
    repaired = extract_json_object(typo_raw)
    assert repaired is not None and "assumptions" in repaired and repaired["facts"][0]["value"] == 84


def test_safe_eval_arith_allows_math_and_blocks_unsafe_nodes() -> None:
    assert safe_eval_arith("4 * 18 + 12") == 84.0
    assert safe_eval_arith("(120 / 30) * 180") == 720.0
    assert safe_eval_arith("-5 + 3") == -2.0
    assert safe_eval_arith("10 / 0") is None
    assert safe_eval_arith("__import__('os').system('echo x')") is None
    assert safe_eval_arith("True + 1") is None
    assert safe_eval_arith("x + 1") is None


def test_pyramid_state_has_no_oracle_constraints_spec() -> None:
    state = PyramidState(problem_id="t0", problem_text="Test")
    assert not hasattr(state, "constraints_spec")


def test_dependency_graph_transitive_taint_and_fact_preservation() -> None:
    state = PyramidState(problem_id="t1", problem_text="Test")
    state.assumptions = {
        "A1": Assumption(id="A1", statement="CPU only"),
        "A2": Assumption(id="A2", statement="4 shards x 18 GB + 12"),
    }
    state.nodes = {
        "L2-1": DependencyNode(id="L2-1", level=2, kind="strategy", statement="Quantize for CPU", depends_on=["A1"]),
        "L2-2": DependencyNode(id="L2-2", level=2, kind="subproblem", statement="Compute shard total", depends_on=["A2"]),
        "L3-1": DependencyNode(id="L3-1", level=3, kind="reasoning_step", statement="Pick OPT-D", depends_on=["L2-1"]),
    }
    state.verified_facts = {
        "F1": VerifiedFact(id="F1", statement="4 * 18 + 12 = 84 GB", depends_on=[], verified_by="ast_math", expr="4 * 18 + 12", value=84.0),
        "F2": VerifiedFact(id="F2", statement="Total storage is 84 GB", depends_on=["A2"]),
        "F3": VerifiedFact(id="F3", statement="CPU latency is 95 ms", depends_on=["A1"]),
        "F4": VerifiedFact(id="F4", statement="Derived from L2-1", depends_on=["L2-1"]),
    }

    tainted = state.compute_tainted_nodes({"A1"})
    assert tainted == {"L2-1", "L3-1"}

    surviving = state.surviving_facts({"A1"})
    assert [f.id for f in surviving] == ["F1", "F2"]


def test_reconstruct_discards_nodes_and_enforces_max_reconstructions() -> None:
    state = PyramidState(problem_id="t2", problem_text="Test", max_reconstructions=1)
    state.assumptions["A1"] = Assumption(id="A1", statement="CPU only")
    state.nodes["L2-1"] = DependencyNode(id="L2-1", level=2, kind="strategy", statement="CPU plan", depends_on=["A1"])
    state.verified_facts["F1"] = VerifiedFact(id="F1", statement="84 GB", depends_on=[], verified_by="ast_math", expr="4*18+12", value=84.0)
    state.candidate_answer = {"primary": "OPT-D", "preserved": 84}

    findings = VerificationFindings(
        foundation_invalid=True,
        invalidated_assumptions=["A1"],
        replacement_assumptions={"A1": "GPU available"},
        tainted_nodes=["L2-1"],
        preserved_facts=["F1"],
        trigger_reason="GPU became available",
    )
    state, rebuilt = reconstruct(state, findings, phase="midway")
    assert rebuilt is True
    assert state.assumptions["A1"].status == "invalidated"
    assert state.assumptions["A1_r1"].status == "active"
    assert state.nodes["L2-1"].status == "discarded"
    assert list(state.verified_facts.keys()) == ["F1"]
    assert state.candidate_answer is None
    assert len(state.revision_log) == 1

    state, rebuilt_again = reconstruct(state, findings, phase="midway_2")
    assert rebuilt_again is False
    assert state.reconstruction_count == 1


def test_close_think_recovery_on_truncated_thinking() -> None:
    problems = load_problems(Path(__file__).resolve().parent.parent / "data" / "pilot_problems.jsonl")
    ord1 = next(p for p in problems if p.id == "pilot-ord-01")

    class TruncatingMockModel:
        model_name = "trunc-mock"

        def __init__(self) -> None:
            self.calls = 0

        def generate(
            self,
            messages: list[dict[str, str]],
            *,
            max_tokens: int = 1024,
            temperature: float = 0.0,
            enable_thinking: bool = True,
        ) -> ModelResponse:
            self.calls += 1
            if enable_thinking:
                raw = "We compute 140/35 = 4 gateways, flash_kb <= 512, 14*25+40 = 390, so OPT-B."
                return ModelResponse(
                    raw_text=raw,
                    thinking_text=raw,
                    content_text="",
                    stats=GenerationStats(
                        prompt_tokens=80,
                        generation_tokens=max_tokens,
                        total_tokens=80 + max_tokens,
                        wall_clock_sec=0.01,
                        finish_reason="length",
                    ),
                )
            return ModelResponse(
                raw_text='{"primary": "OPT-B", "preserved": 390}',
                thinking_text="",
                content_text='{"primary": "OPT-B", "preserved": 390}',
                stats=GenerationStats(
                    prompt_tokens=95,
                    generation_tokens=12,
                    total_tokens=107,
                    wall_clock_sec=0.01,
                    finish_reason="stop",
                ),
            )

    res = run_mode("direct", ord1, TruncatingMockModel(), max_total_tokens=4096)
    assert res.final_answer == {"primary": "OPT-B", "preserved": 390}
    assert any(t["stage"].endswith("_close_think_recovery") for t in res.traces)


def test_build_recovery_messages_and_opt_x_placeholder() -> None:
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "Solve."}]
    rec = build_recovery_messages(msgs, "2 nodes * 340 = 680 -> OPT-B", "Emit JSON.")
    assert len(rec) == 2
    assert rec[-1]["role"] == "user"
    assert "2 nodes * 340 = 680 -> OPT-B" in rec[-1]["content"]
    assert all(m["role"] != "assistant" for m in rec)

    assert parse_candidate_answer('{"primary": "OPT-X", "preserved": null}') is None
    assert parse_candidate_answer("Choose OPT-X or OPT-B") == {"primary": "OPT-B", "preserved": None}
    assert parse_candidate_answer("Choose OPT-E or OPT-F") == {"primary": "OPT-F", "preserved": None}
    assert parse_candidate_answer("PRIMARY: OPT-E\nPRESERVED: 325") == {"primary": "OPT-E", "preserved": 325}


def test_line_parsers_strip_tool_calls_and_correct_mental_math() -> None:
    l1_raw = (
        "<think>Thinking...</think>\n"
        "A1: Active regime is CPU\n"
        "A2: Workload is 120 req/s\n"
        "F1 [NONE]: 14 * 18 + 9 * 12 - 35 = 315 | Ledger GB\n"
        "<tool_call>\n<function=reason>\n</tool_call>"
    )
    assumptions, facts = parse_level1_output(l1_raw)
    assert len(assumptions) == 2
    assert assumptions[0] == {"id": "A1", "statement": "Active regime is CPU"}
    assert len(facts) == 1
    assert safe_eval_arith(facts[0]["expr"]) == 325.0  # Fixes deliberate 315 slip
    multi_eq_raw = "F1 [NONE]: 14*18 + 9*12 - 35 = 252 + 108 - 35 = 325 | Ledger"
    _, multi_facts = parse_level1_output(multi_eq_raw)
    assert len(multi_facts) == 1
    assert multi_facts[0]["expr"] == "14*18 + 9*12 - 35"
    assert safe_eval_arith(multi_facts[0]["expr"]) == 325.0

    l3_raw = (
        "<think>Check options</think>\n"
        "L3-1: GPU -> 2 nodes, cost<=400, ram<=56, lat<=110\n"
        "L3-2: OPT-A cost>400; OPT-B/C CPU; OPT-D no fp16; OPT-E valid; OPT-F lat>110\n"
        '{"primary": "OPT-E", "preserved": 325}'
    )
    l3_steps, l3_cand = parse_level3_output(l3_raw)
    assert len(l3_steps) == 2
    assert l3_cand == {"primary": "OPT-E", "preserved": 325}
    assert (
        _verify_inline_equations("required_units = 120 / 60 = 4; max_cost = 800 / 2 = 350")
        == "required_units = 120 / 60 = 2; max_cost = 800 / 2 = 400"
    )

    l2_raw = (
        "<think>Plan</think>\n"
        "L2-1 [A1, A2]: Compute 4 nodes and max cost 200\n"
        "L2-2 [NONE]: Keep F1 ledger intact\n"
    )
    nodes = parse_level2_output(l2_raw)
    assert nodes[0] == {
        "id": "L2-1",
        "kind": "strategy",
        "statement": "Compute 4 nodes and max cost 200",
        "depends_on": ["A1", "A2"],
    }
    assert nodes[1]["depends_on"] == []

    audit_raw = (
        "<think>Checking</think>\n"
        "CHECK A1: INVALID | Active regime switched from CPU to GPU\n"
        "CHECK A2: VALID | Demand stays 120 req/s\n"
        "SUMMARY: Regime changed.\n"
        "<tool_call><function=noop></function></tool_call>"
    )
    assert "<tool_call>" not in strip_tool_calls(audit_raw)
    inv, _, reason = parse_audit_output(audit_raw, {"A1", "A2"})
    assert inv == ["A1"]
    assert "Regime changed" in reason


def test_all_four_modes_end_to_end_with_mock_model() -> None:
    problems = load_problems(Path(__file__).resolve().parent.parent / "data" / "pilot_problems.jsonl")
    pc1 = next(p for p in problems if p.id == "pilot-pc-01")
    irr1 = next(p for p in problems if p.id == "pilot-irr-01")

    def responder(messages: list[dict[str, str]]) -> str:
        joined = "\n".join(m["content"] for m in messages)
        if "strict logical consistency auditor" in joined:
            if "Dedicated GPU hardware is now online" in joined and "A1: Active mode is CPU" in joined:
                return (
                    "CHECK A1: INVALID | Active mode switched to GPU.\n"
                    "CHECK A2: VALID | Workload demand unchanged.\n"
                    "SUMMARY: Active mode flipped from CPU to GPU."
                )
            return "CHECK A1: VALID | Active mode unchanged.\nSUMMARY: All assumptions valid."
        if "Level 1 (Foundation)" in joined:
            is_irr = "160 req/s" in joined
            pres_expr = "9 * 14 + 6 * 11 - 22" if is_irr else "14 * 18 + 9 * 12 - 35"
            pres_val = 170 if is_irr else 325
            if "Dedicated GPU hardware is now online" in joined:
                return (
                    "A1: Active mode is GPU (60 req/s/node, tensor_fp16=true)\n"
                    "A2: Workload is 120 req/s\n"
                    f"F1 [NONE]: {pres_expr} = {pres_val} | Ledger"
                )
            return (
                "A1: Active mode is CPU (30 req/s/node)\n"
                f"A2: Workload is {'160 req/s' if is_irr else '120 req/s'}\n"
                f"F1 [NONE]: {pres_expr} = {pres_val} | Ledger"
            )
        if "Level 2 (Decomposition" in joined:
            if "Active mode is GPU" in joined:
                return "L2-1 [A1, A2]: 2 nodes, max cost 400\nL2-2 [NONE]: Keep F1"
            return "L2-1 [A1, A2]: 4 nodes, max cost 200\nL2-2 [NONE]: Keep F1"
        if "Level 3 (Detailed Reasoning" in joined:
            if "160 req/s" in joined:
                return json.dumps({"steps": ["Select OPT-E"], "primary": "OPT-E", "preserved": 170})
            if "Active mode is GPU" in joined:
                return json.dumps({"steps": ["Select OPT-E"], "primary": "OPT-E", "preserved": 325})
            return json.dumps({"steps": ["Select OPT-B"], "primary": "OPT-B", "preserved": 325})
        if "Dedicated GPU hardware is now online" in joined:
            return json.dumps({"primary": "OPT-E", "preserved": 325})
        if "160 req/s" in joined:
            return json.dumps({"primary": "OPT-E", "preserved": 170})
        return json.dumps({"primary": "OPT-B", "preserved": 325})

    mock = MockModelClient(responder=responder)
    for mode in ("direct", "recheck", "restart", "rpr"):
        res = run_mode(mode, pc1, mock, max_total_tokens=4096)  # type: ignore[arg-type]
        assert res.final_answer == {"primary": "OPT-E", "preserved": 325}
        if mode == "rpr":
            assert res.pyramid_state is not None
            assert len(res.pyramid_state.revision_log) == 1

    res_irr = run_mode("rpr", irr1, mock, max_total_tokens=4096)
    assert res_irr.final_answer == {"primary": "OPT-E", "preserved": 170}
    assert res_irr.pyramid_state is not None
    assert len(res_irr.pyramid_state.revision_log) == 0


def test_token_budget_stops_generation() -> None:
    problems = load_problems(Path(__file__).resolve().parent.parent / "data" / "pilot_problems.jsonl")
    pc1 = problems[0]
    mock = MockModelClient(responder=lambda _: '{"primary": "OPT-D", "preserved": 84}')
    res = run_mode("recheck", pc1, mock, max_total_tokens=50)
    assert res.budget.budget_exhausted is True


def test_pilot_and_benchmark_100_unique_solutions() -> None:
    root = Path(__file__).resolve().parent.parent / "data"
    pilot = load_problems(root / "pilot_problems.jsonl")
    assert len(pilot) == 20
    assert [p.category for p in pilot].count("premise_change") == 10
    assert [p.category for p in pilot].count("irrelevant_update") == 5
    assert [p.category for p in pilot].count("ordinary") == 5

    bench = load_problems(root / "benchmark_100.jsonl")
    assert len(bench) == 100
    cats = [p.category for p in bench]
    assert cats.count("ordinary") == 50
    assert cats.count("premise_change") == 35
    assert cats.count("irrelevant_update") == 15

    final_labels = {p.expected_final["primary"] for p in [*pilot, *bench]}
    assert {"OPT-E", "OPT-F"}.issubset(final_labels)

    for p in [*pilot, *bench]:
        for phase, spec, exp in [
            ("initial", p.initial_constraints_spec, p.expected_initial),
            ("final", p.final_constraints_spec, p.expected_final),
        ]:
            valid = [
                oid
                for oid in spec["options"]
                if not check_candidate_constraints({"primary": oid, "preserved": 0}, spec)
            ]
            assert valid == [exp["primary"]], f"{p.id} {phase} valid={valid} != [{exp['primary']}]"


def test_checkpoint_resume_skips_completed_runs(tmp_path: Path) -> None:
    problems = load_problems(Path(__file__).resolve().parent.parent / "data" / "pilot_problems.jsonl")
    ord1 = next(p for p in problems if p.id == "pilot-ord-01")
    mock1 = MockModelClient(responder=lambda _: '{"primary": "OPT-D", "preserved": 390}')
    run_evaluation([ord1], ["direct", "restart"], mock1, output_dir=tmp_path, token_budget=4096, resume=True)
    assert len(mock1.calls) == 2

    mock2 = MockModelClient(responder=lambda _: '{"primary": "OPT-D", "preserved": 390}')
    run_evaluation(
        [ord1],
        ["direct", "restart", "recheck"],
        mock2,
        output_dir=tmp_path,
        token_budget=4096,
        resume=True,
    )
    assert len(mock2.calls) == 2
    assert len((tmp_path / "traces.jsonl").read_text(encoding="utf-8").splitlines()) == 3
    summary_data = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary_data["config"]["completed_runs"] == 3


def test_symbolic_critic_zero_leakage_and_cross_binding_repair() -> None:
    from revision import verify_from_level_one

    problems = load_problems(Path(__file__).resolve().parent.parent / "data" / "pilot_problems.jsonl")
    ord1 = next(p for p in problems if p.id == "pilot-ord-01")
    opts = parse_prompt_options(ord1.initial_prompt)
    assert set(opts.keys()) == {"OPT-A", "OPT-B", "OPT-C", "OPT-D", "OPT-E", "OPT-F"}
    assert opts["OPT-D"]["cost_usd"] == 270
    assert opts["OPT-D"]["tensor_fp16"] is False

    st = PyramidState(problem_id=ord1.id, problem_text=ord1.initial_prompt)
    st.assumptions["A1"] = Assumption(id="A1", statement="active_mode=CPU; boolean_feature=tensor_fp16 optional")
    st.nodes = {
        "L2-1": DependencyNode(id="L2-1", level=2, kind="strategy", statement="required_units = 140 / 35 = 4"),
        "L2-2": DependencyNode(id="L2-2", level=2, kind="strategy", statement="max_per_unit_cost = 1200 / 140 = 8.57143"),
        "L2-3": DependencyNode(id="L2-3", level=2, kind="strategy", statement="required_mode = CPU; mandatory_feature = tensor_fp16 optional"),
        "L2-4": DependencyNode(id="L2-4", level=2, kind="strategy", statement="max_base_time_or_lat = 80 - 25 = 55"),
        "L2-5": DependencyNode(id="L2-5", level=2, kind="strategy", statement="max_per_unit_resource = 60 - 18 = 42"),
    }
    repairs = repair_l2_cross_bindings(st.nodes)
    assert len(repairs) == 1
    assert st.nodes["L2-2"].statement == "max_per_unit_cost = 1200 / 4 = 300"

    bounds = extract_l2_bounds(list(st.nodes.values()))
    passing, _ = verify_options_against_bounds(opts, bounds)
    assert passing == ["OPT-D"]

    st.nodes["L2-5"].statement = "max_per_unit_resource = 4"
    unsat_bounds = extract_l2_bounds(list(st.nodes.values()))
    unsat_passing, _ = verify_options_against_bounds(opts, unsat_bounds)
    assert unsat_passing == []
    findings, _ = verify_from_level_one(st, use_self_checks=True, run_model_audit=False)
    assert findings.foundation_invalid is True
    assert any("UNSAT_L2_BOUNDS" in v for v in findings.violated_constraints)


def test_training_rewards_and_decontamination() -> None:
    from training.build_training_data import assert_decontaminated
    from training.verifiable_rewards import compute_five_gram_echo, parse_xml_tool_calls, score_response

    prompt = "Active workload demand: 120 Mbps. Budget ceiling: required * usd <= 600. Pick OPT-A..OPT-F."
    good = '<think>Capacity=30, units=4, max cost=150. OPT-B valid.</think>\n{"primary": "OPT-B", "preserved": 195}'
    sc_good = score_response(
        good,
        prompt_text=prompt,
        kind="math_opt",
        verifier={"primary": "OPT-B", "preserved": 195, "forbid_tool_call": True},
    )
    assert sc_good["correct"] is True
    assert sc_good["closed_think"] is True
    assert sc_good["tool_leaked"] is False

    echo_leak = (
        "<think>Active workload demand: 120 Mbps. Budget ceiling: required * usd <= 600. Pick OPT-A..OPT-F.</think>\n"
        "<tool_call><function=fetch_telemetry><parameter=station_id>STN-1</parameter></function></tool_call>"
    )
    assert compute_five_gram_echo(prompt, echo_leak) > 0.3
    sc_bad = score_response(
        echo_leak,
        prompt_text=prompt,
        kind="tool_abstention",
        verifier={"abstain": True, "forbid_tool_call": True, "expected": {"total": 100}},
    )
    assert sc_bad["correct"] is False
    assert sc_bad["tool_leaked"] is True

    calls = parse_xml_tool_calls("<tool_call>\n<function=fetch_telemetry>\n<parameter=station_id>\nSTN-42\n</parameter>\n<parameter=window_min>\n15\n</parameter>\n</function>\n</tool_call>")
    assert calls == [{"name": "fetch_telemetry", "arguments": {"station_id": "STN-42", "window_min": "15"}}]

    self_reminder = "<think>Tools are listed in system prompt; never emit <tool_call>. 50 + 50 = 100.</think>\n{\"total\": 100}"
    sc_reminder = score_response(
        self_reminder,
        prompt_text=prompt,
        kind="tool_abstention",
        verifier={"abstain": True, "forbid_tool_call": True, "expected": {"total": 100}},
    )
    assert sc_reminder["tool_leaked"] is False
    assert sc_reminder["correct"] is True

    train_path = Path(__file__).resolve().parent.parent / "training" / "data" / "train_sft.jsonl"
    if train_path.exists():
        rows = [json.loads(l) for l in train_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert_decontaminated(rows)


def test_unsat_l2_bounds_does_not_clobber_midway_reconstruction_budget() -> None:
    st = PyramidState(problem_id="t3", problem_text="Test", evidence=["init"], max_reconstructions=2)
    st.assumptions["A1"] = Assumption(id="A1", statement="mode=DIESEL")
    st.nodes["L2-1"] = DependencyNode(id="L2-1", level=2, kind="strategy", statement="max_cost = 10")

    unsat_findings = VerificationFindings(
        foundation_invalid=True,
        invalidated_assumptions=[],
        replacement_assumptions={},
        tainted_nodes=[],
        preserved_facts=[],
        violated_constraints=["UNSAT_L2_BOUNDS: 0 options satisfy bounds"],
        trigger_reason="UNSAT_L2_BOUNDS: 0 options satisfy bounds",
    )
    st, rebuilt1 = reconstruct(st, unsat_findings, phase="phase1_d2")
    assert rebuilt1 is True
    assert st.reconstruction_count == 1
    assert st.max_reconstructions == 2

    # Simulating a midway update should still allow a second reconstruction.
    st.evidence.append("Mode switched to RAIL")
    midway_findings = VerificationFindings(
        foundation_invalid=True,
        invalidated_assumptions=["A1"],
        replacement_assumptions={"A1": "mode=RAIL"},
        tainted_nodes=[],
        preserved_facts=[],
        trigger_reason="Active mode changed to RAIL",
    )
    st, rebuilt2 = reconstruct(st, midway_findings, phase="midway_update")
    assert rebuilt2 is True
    assert st.reconstruction_count == 2

    # Conversely, repeating UNSAT_L2_BOUNDS with identical evidence & foundation should clamp budget.
    st_dup = PyramidState(problem_id="t4", problem_text="Test", evidence=["init"], max_reconstructions=3)
    st_dup.assumptions["A1"] = Assumption(id="A1", statement="mode=DIESEL")
    st_dup, r1 = reconstruct(st_dup, unsat_findings, phase="phase1_d2")
    assert r1 is True
    assert st_dup.max_reconstructions == 3
    st_dup, r2 = reconstruct(st_dup, unsat_findings, phase="phase1_d2_again")
    assert r2 is True
    assert st_dup.max_reconstructions == 2
    st_dup, r3 = reconstruct(st_dup, unsat_findings, phase="phase1_d2_blocked")
    assert r3 is False


def test_repair_l2_from_l1_assumptions_fixes_stale_bound_carryover() -> None:
    from symbolic_critic import repair_l2_from_l1_assumptions

    assumptions = [
        Assumption(id="A3", statement="total budget cap formula is (required arrays) * monthly_usd <= 1200"),
        Assumption(id="A4_new", statement="effective latency/time cap formula is base_read_ms + 5 <= 50"),
        Assumption(id="A5_new", statement="per-unit resource/capacity limit formula is 80 - 12 = 68"),
    ]
    nodes = {
        "L2-2": DependencyNode(id="L2-2", level=2, kind="strategy", statement="max_per_unit_cost = 900 / 3 = 300"),
        "L2-4": DependencyNode(id="L2-4", level=2, kind="strategy", statement="max_base_time_or_lat = 30 - 5 = 25"),
        "L2-5": DependencyNode(id="L2-5", level=2, kind="strategy", statement="max_per_unit_resource = 50 - 12 = 38"),
    }
    repairs = repair_l2_from_l1_assumptions(assumptions, nodes)
    assert len(repairs) == 3
    assert nodes["L2-2"].statement == "max_per_unit_cost = 1200 / 3 = 400"
    assert nodes["L2-4"].statement == "max_base_time_or_lat = 50 - 5 = 45"
    assert nodes["L2-5"].statement == "max_per_unit_resource = 80 - 12 = 68"


def test_rpr_v4_boolean_sync_satisfiability_guard_and_depth2_completion() -> None:
    from symbolic_critic import format_bounds_readable, repair_l2_from_l1_assumptions

    # 1. F1 prose prefix stripping in parse_level1_output
    _, f1_list = parse_level1_output(
        "<think>calc</think>\nA1: active_mode=IN_PERSON; boolean_feature=recording_rig optional\n"
        "F1 [NONE]: invariant ledger calculation: 7 * 18 + 8 * 12 - 19 = 126 + 96 - 19 = 203"
    )
    assert len(f1_list) == 1
    assert f1_list[0]["expr"] == "7 * 18 + 8 * 12 - 19"
    assert safe_eval_arith(f1_list[0]["expr"]) == 203.0

    # 2. Tuple parenthesis parsing in extract_l2_bounds + readable formatting
    tuple_node = DependencyNode(
        id="L2-3",
        level=2,
        kind="strategy",
        statement="required_mode = RAIL_EXPRESS; mandatory_feature = (cold_telemetry, True required)",
    )
    b_tuple = extract_l2_bounds([tuple_node])
    assert b_tuple["mandatory_feature"] == ("cold_telemetry", True)
    assert "(" not in format_bounds_readable(b_tuple)

    alias_node = DependencyNode(
        id="L2-3",
        level=2,
        kind="strategy",
        statement="active_mode = CPU; mandatory_feature = tensor_fp16 optional=true",
    )
    b_alias = extract_l2_bounds([alias_node])
    assert b_alias["required_mode"] == "CPU"
    assert b_alias["mandatory_feature"] == ("tensor_fp16", False)

    _, displaced_f1 = parse_level1_output(
        "<think>ok</think>\n"
        "A1: active_mode=CPU; boolean_feature=tensor_fp16 optional\n"
        "A5: 14 * 18 + 9 * 12 - 35 = 325\n"
        "F1 [NONE]: invariant ledger calculation"
    )
    assert safe_eval_arith(displaced_f1[0]["expr"]) == 325.0

    # 3. Boolean optional sync from A1 -> L2-3 + monotonic satisfiability rollback guard
    problems = load_problems(Path(__file__).resolve().parent.parent / "data" / "pilot_problems.jsonl")
    ord1 = next(p for p in problems if p.id == "pilot-ord-01")
    opts = parse_prompt_options(ord1.initial_prompt)

    a_good_bool = [
        Assumption(id="A1", statement="active_mode=CPU; boolean_feature=tensor_fp16 optional"),
        Assumption(id="A4", statement="base_lat_ms + 25 <= 80"),
        Assumption(id="A5", statement="footprint_gb <= 60 - 18"),
    ]
    nodes_bool = {
        "L2-1": DependencyNode(id="L2-1", level=2, kind="strategy", statement="required_units = 140 / 35 = 4"),
        "L2-2": DependencyNode(id="L2-2", level=2, kind="strategy", statement="max_per_unit_cost = 1200 / 4 = 300"),
        "L2-3": DependencyNode(id="L2-3", level=2, kind="strategy", statement="required_mode = CPU; mandatory_feature = tensor_fp16=true required"),
        "L2-4": DependencyNode(id="L2-4", level=2, kind="strategy", statement="max_base_time_or_lat = 80 - 25 = 55"),
        "L2-5": DependencyNode(id="L2-5", level=2, kind="strategy", statement="max_per_unit_resource = 60 - 18 = 42"),
    }
    reps = repair_l2_from_l1_assumptions(a_good_bool, nodes_bool, options=opts)
    assert len(reps) == 1
    assert "tensor_fp16 optional" in nodes_bool["L2-3"].statement
    passing_fixed, _ = verify_options_against_bounds(opts, extract_l2_bounds(list(nodes_bool.values())))
    assert passing_fixed == ["OPT-D"]

    # Now simulate L2 self-check having valid 60-18=42 while L1 had typo 35-18=17 (which would make 0 options pass):
    a_bad_l1 = [
        Assumption(id="A1", statement="active_mode=CPU; boolean_feature=tensor_fp16 optional"),
        Assumption(id="A5", statement="footprint_gb <= 35 - 18"),
    ]
    reps_guarded = repair_l2_from_l1_assumptions(a_bad_l1, nodes_bool, options=opts)
    assert reps_guarded == []
    assert "60 - 18 = 42" in nodes_bool["L2-5"].statement

    # 4. Self-contradictory INVALID audit line filtered on irrelevant update
    contradictory_audit = (
        "<think>check</think>\n"
        "CHECK A1: VALID | active_mode=IN_PERSON matches rule.\n"
        "CHECK A2: INVALID | Latest Evidence states main conference track has not switched from IN_PERSON, so capacity stays 60.\n"
        "SUMMARY: Active regime=IN_PERSON."
    )
    inv_ids, _, _ = parse_audit_output(contradictory_audit, {"A1", "A2"})
    assert inv_ids == []

    # 5. Depth-2 max_reconstructions exhaustion still advances to Level 3 (never returns None)
    def stubborn_unsat_responder(messages: list[dict[str, str]]) -> str:
        joined = "\n".join(m["content"] for m in messages)
        if "Level 1 (Foundation)" in joined:
            return (
                "A1: active_mode=CPU; boolean_feature=tensor_fp16=true\n"
                "A2: demand 140, handles 35\n"
                "F1 [NONE]: invariant ledger calculation: 14 * 25 + 6 * 13 - 38 = 390"
            )
        if "Level 2 (Decomposition" in joined:
            return (
                "L2-1 [A1, A2]: required_units = 140 / 35 = 4\n"
                "L2-2 [A1, A2]: max_per_unit_cost = 1200 / 4 = 300\n"
                "L2-3 [A1]: required_mode = CPU; mandatory_feature = tensor_fp16=true\n"
                "L2-4 [A1]: max_base_time_or_lat = 80 - 25 = 55\n"
                "L2-5 [A1]: max_per_unit_resource = 60 - 18 = 42"
            )
        return 'L3-1: bounds\nL3-2: OPT-D pass\n{"primary": "OPT-D", "preserved": 1}'

    res_stubborn = run_mode(
        "rpr",
        ord1,
        MockModelClient(responder=stubborn_unsat_responder),
        max_total_tokens=8192,
        max_reconstructions=1,
    )
    assert res_stubborn.final_answer == {"primary": "OPT-D", "preserved": 390}


def test_direct_tir_sandbox_calc_and_28l_upcycle_config() -> None:
    from training.upcycle_and_train_rpr_v5_bf16 import build_upcycled_28l_config_dict

    problems = load_problems(Path(__file__).resolve().parent.parent / "data" / "pilot_problems.jsonl")
    pc1 = next(p for p in problems if p.id == "pilot-pc-01")

    def tir_responder(messages: list[dict[str, str]]) -> str:
        last_u = messages[-1]["content"]
        if "<tool_response>" in last_u:
            assert '"result": 325' in last_u
            if any("Dedicated GPU hardware is now online" in m["content"] for m in messages):
                return '<think>GPU valid=OPT-E, calc=325.</think>\n{"primary": "OPT-E", "preserved": 325}'
            return '<think>CPU valid=OPT-B, calc=325.</think>\n{"primary": "OPT-B", "preserved": 325}'
        return (
            "<think>Call calc for ledger.</think>\n"
            "<tool_call>\n<function=calc>\n<parameter=expr>\n14 * 18 + 9 * 12 - 35\n</parameter>\n</function>\n</tool_call>"
        )

    res_tir = run_mode("direct_tir", pc1, MockModelClient(responder=tir_responder), max_total_tokens=8192)
    assert res_tir.initial_answer == {"primary": "OPT-B", "preserved": 325}
    assert res_tir.final_answer == {"primary": "OPT-E", "preserved": 325}
    assert res_tir.budget.model_calls == 4

    base_cfg_path = Path(__file__).resolve().parent.parent / "checkpoints" / "LFM3-VL-2B-RPR-BF16" / "config.json"
    if base_cfg_path.exists():
        cfg_24 = json.loads(base_cfg_path.read_text(encoding="utf-8"))
        cfg_28 = build_upcycled_28l_config_dict(cfg_24)
        assert cfg_28["text_config"]["num_hidden_layers"] == 28
        assert len(cfg_28["text_config"]["layer_types"]) == 28
        assert cfg_28["text_config"]["layer_types"].count("full_attention") == 8


def test_mtp_speculative_tri_hybrid_lossless_parity() -> None:
    import mlx.core as mx
    from mlx_vlm.models.lfm3.config import ModelConfig as Lfm3TextConfig
    from mlx_vlm.models.lfm3.language import LanguageModel
    from mtp_speculative import SequentialMTPHead, mtp_speculative_generate, standard_greedy_generate

    mx.random.seed(42)
    tcfg = Lfm3TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        conv_L_cache=7,
        sliding_window=8,  # Forces RotatingKVCache wrap-around during 24-token generation!
        layer_types=["conv", "sliding_attention", "conv", "full_attention"],
    )

    class _DummyVLM:
        def __init__(self, lm: LanguageModel):
            self.language_model = lm

    vlm = _DummyVLM(LanguageModel(tcfg))
    mtp = SequentialMTPHead(tcfg, mtp_intermediate_size=96, shortlist_ids=list(range(64)))
    mtp.warm_start_from_target_lm(vlm.language_model)
    mx.eval(vlm.language_model.parameters(), mtp.parameters())

    prompt = [3, 11, 19, 27, 35, 43]
    for rep_p in (1.0, 1.08):
        ref_ids, _ = standard_greedy_generate(vlm, prompt, max_tokens=24, eos_token_id=127, repetition_penalty=rep_p)
        for dk in (3, 4, 5):
            spec_ids, stats = mtp_speculative_generate(
                vlm, mtp, prompt, max_tokens=24, draft_k=dk, eos_token_id=127, repetition_penalty=rep_p
            )
            assert spec_ids == ref_ids, f"MTP mismatch (dk={dk}, rep={rep_p}):\n  spec={spec_ids}\n  ref ={ref_ids}"
            assert stats["tokens"] == 24


def test_v6_32l_config_and_seal_plan_parser() -> None:
    from training.run_v6_32l_mtp_seal_pipeline import (
        REWRITE_TARGET_LAYERS,
        build_upcycled_32l_config_from_28l,
        parse_weight_rewrite_plan,
    )

    cfg_28_path = Path(__file__).resolve().parent.parent / "checkpoints" / "LFM3-VL-2.2B-RPR-RSI-BF16" / "config.json"
    if cfg_28_path.exists():
        cfg_28 = json.loads(cfg_28_path.read_text(encoding="utf-8"))
        cfg_32 = build_upcycled_32l_config_from_28l(cfg_28)
        ltypes = cfg_32["text_config"]["layer_types"]
        assert cfg_32["text_config"]["num_hidden_layers"] == 32
        assert len(ltypes) == 32
        assert ltypes.count("full_attention") == 10
        for r_idx in REWRITE_TARGET_LAYERS:
            assert ltypes[r_idx] == "full_attention", f"Layer {r_idx} is {ltypes[r_idx]}, expected full_attention"

    raw_plan = (
        "<think>Diagnose code failure; apply SVF basis 0 and 1 self-edit in null space.</think>\n"
        '{"action": "rewrite", "svf_gains": [0.65, 0.15, 0.0, 0.1], "lr_b": 8e-5, "lr_a": 2e-5, '
        '"steps": 2, "nullspace_alpha": 0.90, "wise_ft_gamma": 0.92, '
        '"self_edits": [{"prompt": "Fix bug", "response": "<think>Check bounds.</think>\\n```python\\nx=1\\n```"}]}'
    )
    parsed, r_sch = parse_weight_rewrite_plan(raw_plan)
    assert parsed is not None
    assert parsed["action"] == "rewrite"
    assert parsed["steps"] == 2
    assert len(parsed["self_edits"]) == 1
    assert r_sch >= 0.80

    nested_bad = '<think>Done.</think>\n{"action": "rewrite", "svf_gains": [[0.5, "bad", null]], "lr_b": [1e-4], "steps": "3.0"}'
    p_bad, _ = parse_weight_rewrite_plan(nested_bad)
    assert p_bad is not None
    assert p_bad["svf_gains"] == [0.5, 0.0, 0.0, 0.0]
    assert p_bad["steps"] == 3


def test_v7_36l_config_sandbox_self_edits_and_audit_guard() -> None:
    from revision import parse_audit_output
    from training.run_v7_36l_zero_degrade_pipeline import (
        REWRITE_TARGET_LAYERS_36L,
        SEAL_POLICY_NEW_INDICES,
        UPCYCLED_36L_INDICES,
        build_upcycled_36l_config_from_28l,
        filter_verified_self_edits_sandboxed,
    )

    # 1. Global non-change SUMMARY & `from 31 to 31` no-op audit guard
    noop_audit = (
        "<think>Check DIESEL_CONVOY.</think>\n"
        "CHECK A1: VALID | unchanged\n"
        "CHECK A4: INVALID | active penalty=19 changes cap from 31 to 31\n"
        "SUMMARY: Active DIESEL_CONVOY regime preserves A1..A5, and Latest Evidence confirmed no mode switch."
    )
    inv, _, _ = parse_audit_output(noop_audit, {"A1", "A2", "A3", "A4", "A5"})
    assert inv == []

    # 2. 36L config (`12` full_attention layers = `2.0x` base, disjoint policy vs rewrite layers)
    cfg_28_path = Path(__file__).resolve().parent.parent / "checkpoints" / "LFM3-VL-2.2B-RPR-RSI-BF16" / "config.json"
    if cfg_28_path.exists():
        cfg_28 = json.loads(cfg_28_path.read_text(encoding="utf-8"))
        cfg_36 = build_upcycled_36l_config_from_28l(cfg_28)
        ltypes = cfg_36["text_config"]["layer_types"]
        assert cfg_36["text_config"]["num_hidden_layers"] == 36
        assert len(ltypes) == 36
        assert ltypes.count("full_attention") == 12
        assert set(SEAL_POLICY_NEW_INDICES).issubset(set(UPCYCLED_36L_INDICES))
        assert set(SEAL_POLICY_NEW_INDICES).isdisjoint(set(REWRITE_TARGET_LAYERS_36L))
        for r_idx in REWRITE_TARGET_LAYERS_36L:
            assert ltypes[r_idx] == "full_attention"

    # 3. Sandboxed `python3 -I` self-edit filter rejects broken Python and accepts valid Python/JSON self-edits
    cands = [
        {"prompt": "Good code", "response": "<think>Verify.</think>\n```python\ndef f(x):\n    return x + 1\nassert f(2) == 3\n```"},
        {"prompt": "Syntax bug", "response": "<think>Oops.</think>\n```python\ndef f(x):\n    return x[\n```"},
        {"prompt": "Runtime bug", "response": "<think>Oops.</think>\n```python\nraise ValueError('bad')\n```"},
        {"prompt": "Good JSON", "response": '<think>2+2=4.</think>\n{"ans": 4}'},
    ]
    kept = filter_verified_self_edits_sandboxed(cands)
    assert [x["prompt"] for x in kept] == ["Good code", "Good JSON"]


def test_rpr_controller_v6_f1_think_recovery_l2_empty_retry_and_unsat_guard() -> None:
    # 1. Bare integer `F1 [NONE]: 165` recovers multi-op ledger equation from `<think>`!
    raw_bare_f1 = (
        "<think>Ledger: 7 * 23 + 4 * 12 - 44 = 161 + 48 - 44 = 165.</think>\n"
        "A1: active_mode=STANDARD_HDD; boolean_feature=hw_encrypt=optional\n"
        "F1 [NONE]: 165 | invariant ledger"
    )
    assump, facts = parse_level1_output(raw_bare_f1)
    assert len(assump) == 1
    assert len(facts) == 1
    assert facts[0]["expr"] == "7 * 23 + 4 * 12 - 44"
    assert safe_eval_arith(facts[0]["expr"]) == 165.0

    # 2. Omitted `F1` line still recovers multi-op ledger equation from `<think>`!
    raw_omitted_f1 = (
        "<think>Active mode=STANDARD_HDD; ledger is 7 * 23 + 4 * 12 - 44 = 165.</think>\n"
        "A1: active_mode=STANDARD_HDD; boolean_feature=hw_encrypt=optional\n"
        "A2: workload demand is 240, each unit handles 40"
    )
    _, facts2 = parse_level1_output(raw_omitted_f1)
    assert len(facts2) == 1
    assert safe_eval_arith(facts2[0]["expr"]) == 165.0

    # 3. End-to-end RPR test:
    #    - Phase 1 omits `F1` completely -> `_advance_level_1` triggers `_level1_f1` extraction (`165`)
    #    - Phase 2 first Level 2 emits prose without `L2-*` -> `_level2_format_retry` recovers `L2-1..L2-5`
    #    - Residual `UNSAT_L2_BOUNDS` at depth 3 does NOT overwrite `unsat_l2_fallback` answer!
    problems = load_problems(Path(__file__).resolve().parent.parent / "data" / "pilot_problems.jsonl")
    pc1 = next(p for p in problems if p.id == "pilot-pc-01")
    l2_calls = {"p2": 0}

    def responder(messages: list[dict[str, str]]) -> str:
        sys_txt = messages[0]["content"]
        joined = "\n".join(m["content"] for m in messages)
        if "Extract ONLY the invariant ledger" in sys_txt:
            return "F1 [NONE]: 14 * 18 + 9 * 12 - 35 = 325 | invariant ledger"
        if "strict logical consistency auditor" in joined:
            if "Dedicated GPU hardware is now online" in joined:
                return "CHECK A1: INVALID | switched to GPU\nSUMMARY: Active mode switched to GPU."
            return "CHECK A1: VALID | unchanged\nSUMMARY: No change."
        if "Level 1 (Foundation)" in sys_txt:
            if "Dedicated GPU hardware is now online" in joined:
                return "A1: active_mode=GPU; boolean_feature=tensor_fp16=true\nA2: demand is 120, unit handles 60"
            return "A1: active_mode=CPU; boolean_feature=optional\nA2: demand is 120, unit handles 30"
        if "Level 2 (Decomposition" in sys_txt:
            if "Dedicated GPU hardware is now online" in joined:
                l2_calls["p2"] += 1
                return "Active regime=GPU (all valid)."  # Deliberately omits L2-1..L2-5 on first call!
            return "L2-1 [A1, A2]: required_units = 120 / 30 = 4"
        if "Output ONLY 5 tagged lines L2-1..L2-5" in sys_txt:
            # Deliberately return UNSAT cost=10 so depth-3 fallback solves from full Problem Rules
            return (
                "L2-1 [A1, A2]: required_units = 120 / 60 = 2\n"
                "L2-2 [A1, A2]: max_per_unit_cost = 20 / 2 = 10\n"
                "L2-3 [A1]: required_mode = GPU; mandatory_feature = tensor_fp16=true\n"
                "L2-4 [A1]: max_base_time_or_lat = 125 - 15 = 110\n"
                "L2-5 [A1]: max_per_unit_resource = 64 - 8 = 56"
            )
        if "Level 3 (Detailed Reasoning" in sys_txt:
            if "Dedicated GPU hardware is now online" in joined:
                return 'L3-1: GPU\nL3-2: OPT-E pass\n{"primary": "OPT-E", "preserved": 1}'
            return 'L3-1: CPU\nL3-2: OPT-B pass\n{"primary": "OPT-B", "preserved": 1}'
        return '{"primary": "OPT-B", "preserved": 325}'

    res = run_mode("rpr", pc1, MockModelClient(responder=responder), max_total_tokens=8192)
    assert res.final_answer == {"primary": "OPT-E", "preserved": 325}
    assert any(t.get("stage", "").endswith("_level1_f1") for t in res.traces)
    assert any(t.get("stage", "").endswith("_level2_format_retry") for t in res.traces)


def test_v8_engram_attnres_moba_muonclip_and_agm_audit() -> None:
    import mlx.core as mx
    import mlx.nn as nn

    from revision import parse_audit_output
    from v8_arch import (
        CrossLayerAttnRes,
        EngramConditionalMemory,
        Muon2D,
        moba_pinned_sparse_sdpa,
        zeropower_via_newtonschulz5,
    )

    # 1. AGM Mode-Lock Audit Guard eliminates prototype-memo false rebuilds (`bench-033` & `bench-067` patterns)
    proto_audit = (
        "<think>QA prototype tested NVME_FABRIC, live stays STANDARD_HDD.</think>\n"
        "CHECK A1: VALID | Active regime stays STANDARD_HDD\n"
        "CHECK A2: INVALID | prototype memo tested 80 kIOPS\n"
        "SUMMARY: Active regime=STANDARD_HDD remains unchanged; A1 is VALID while A2 is INVALID due to non-switching QA prototype lab evidence."
    )
    inv, _, _ = parse_audit_output(proto_audit, {"A1", "A2", "A3", "A4", "A5"})
    assert inv == []

    # Real regime switch still invalidates cleanly!
    real_switch = (
        "<think>Switched to NVME_FABRIC.</think>\n"
        "CHECK A1: INVALID | Active mode switched from STANDARD_HDD to NVME_FABRIC\n"
        "CHECK A2: INVALID | Capacity switched from 40 to 80\n"
        "SUMMARY: Active regime switched from STANDARD_HDD to NVME_FABRIC."
    )
    inv_real, _, _ = parse_audit_output(real_switch, {"A1", "A2", "A3"})
    assert inv_real == ["A1", "A2"]

    # 2. EngramConditionalMemory & CrossLayerAttnRes Step-0 exact zero-identity + nonzero trained response
    h0 = mx.random.normal((1, 6, 64)).astype(mx.float32)
    ids = mx.array([[10, 25, 42, 99, 105, 7]], dtype=mx.int32)
    prev = mx.array([[0, 10, 25, 42, 99, 105]], dtype=mx.int32)
    eng = EngramConditionalMemory(hidden_size=64, num_buckets=256, gate_rank=16)
    h_eng = eng(h0, ids, prev)
    mx.eval(h_eng)
    assert float(mx.max(mx.abs(h_eng - h0)).item()) == 0.0  # Exact Step-0 identity!

    eng.out_scale = mx.ones((64,), dtype=mx.bfloat16)
    # Streaming decode parity: step-by-step `L=1` == full `L=6`!
    h_full = eng(h0, ids, prev)
    step_outs = [eng(h0[:, j : j + 1, :], ids[:, j : j + 1], prev[:, j : j + 1]) for j in range(6)]
    h_stream = mx.concatenate(step_outs, axis=1)
    mx.eval(h_full, h_stream)
    assert float(mx.max(mx.abs(h_full - h_stream)).item()) < 1e-6

    ar = CrossLayerAttnRes(hidden_size=64, num_sources=4, bottleneck_dim=16)
    srcs = [mx.random.normal((1, 6, 64)).astype(mx.float32) for _ in range(4)]
    h_ar0 = ar(h0, srcs)
    mx.eval(h_ar0)
    assert float(mx.max(mx.abs(h_ar0 - h0)).item()) == 0.0  # Exact Step-0 identity!

    # 3. MoBA Pinned-Sink + Top-K Block Attention (`L_kv = 4608 > 4096`)
    q_dec = mx.random.normal((1, 16, 1, 32)).astype(mx.bfloat16)
    k_long = mx.random.normal((1, 4, 4608, 32)).astype(mx.bfloat16)
    v_long = mx.random.normal((1, 4, 4608, 32)).astype(mx.bfloat16)
    out_dec = moba_pinned_sparse_sdpa(q_dec, k_long, v_long, scale=32**-0.5)
    mx.eval(out_dec)
    assert out_dec.shape == (1, 16, 1, 32)
    assert not bool(mx.any(mx.isnan(out_dec)).item())

    # 4. Muon2D 5-Step Quintic Newton-Schulz Orthogonalization (`Q @ Q.T` approx `I`)
    g_mat = mx.random.normal((32, 64)).astype(mx.float32)
    q_orth = zeropower_via_newtonschulz5(g_mat, steps=5).astype(mx.float32)
    gram = q_orth @ q_orth.T
    mx.eval(gram)
    diag_mean = float(mx.mean(mx.diag(gram)).item())
    assert 0.55 <= diag_mean <= 1.35

    lin = nn.Linear(64, 32, bias=False)
    muon = Muon2D(learning_rate=1e-3)
    w_before = mx.array(lin.weight)
    muon.update(lin, {"weight": g_mat})
    mx.eval(lin.weight)
    assert float(mx.max(mx.abs(lin.weight - w_before)).item()) > 1e-5


