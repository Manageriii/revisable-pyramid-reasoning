"""State representation and Assumption Dependency Graph for Revisable Pyramid Reasoning (RPR)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import time
from typing import Any, Literal


AssumptionStatus = Literal["active", "invalidated", "replaced"]
NodeStatus = Literal["valid", "tainted", "discarded"]


@dataclass
class Assumption:
    """A Level-1 foundational assumption or constraint premise."""

    id: str
    statement: str
    source: Literal["given", "inferred"] = "given"
    status: AssumptionStatus = "active"
    invalidation_reason: str | None = None
    replaced_by: str | None = None


@dataclass
class DependencyNode:
    """A node at Level 2 (decomposition/strategy) or Level 3 (detailed reasoning)."""

    id: str
    level: int
    kind: Literal["subproblem", "strategy", "reasoning_step", "conclusion"]
    statement: str
    depends_on: list[str] = field(default_factory=list)
    status: NodeStatus = "valid"


@dataclass
class VerifiedFact:
    """A derived or given fact tracked alongside its assumption dependencies."""

    id: str
    statement: str
    depends_on: list[str] = field(default_factory=list)
    verified_by: Literal["given", "ast_math", "model_audit"] = "given"
    expr: str | None = None
    value: float | None = None


@dataclass
class RevisionRecord:
    """Audit entry explaining why a reconstruction or audit occurred."""

    step: int
    phase: str
    trigger_reason: str
    invalidated_assumptions: list[str]
    tainted_nodes: list[str]
    preserved_facts: list[str]
    rebuilt_from_level: int
    timestamp_sec: float = field(default_factory=time.time)


@dataclass
class PyramidState:
    """Mutable working state for a single problem across pyramid depths and revisions."""

    problem_id: str
    problem_text: str
    evidence: list[str] = field(default_factory=list)
    objective: str = ""
    core_constraints: list[str] = field(default_factory=list)
    assumptions: dict[str, Assumption] = field(default_factory=dict)
    nodes: dict[str, DependencyNode] = field(default_factory=dict)
    verified_facts: dict[str, VerifiedFact] = field(default_factory=dict)
    candidate_answer: dict[str, Any] | None = None
    current_depth: int = 0
    reconstruction_count: int = 0
    max_reconstructions: int = 2
    revision_log: list[RevisionRecord] = field(default_factory=list)
    seen_foundation_signatures: set[str] = field(default_factory=set)

    def active_assumptions(self) -> list[Assumption]:
        """Return currently active Level-1 assumptions in insertion order."""
        return [a for a in self.assumptions.values() if a.status == "active"]

    def invalidated_assumption_ids(self) -> list[str]:
        """Return IDs of assumptions marked invalidated."""
        return [aid for aid, a in self.assumptions.items() if a.status == "invalidated"]

    def active_nodes(self, level: int | None = None) -> list[DependencyNode]:
        """Return non-discarded dependency nodes, optionally filtered by level."""
        return [
            node
            for node in self.nodes.values()
            if node.status != "discarded" and (level is None or node.level == level)
        ]

    def compute_tainted_nodes(self, invalid_roots: set[str]) -> set[str]:
        """Compute transitive closure of nodes that depend on any ID in invalid_roots."""
        if not invalid_roots:
            return set()
        tainted: set[str] = set()
        frontier = set(invalid_roots)
        changed = True
        while changed:
            changed = False
            for node_id, node in self.nodes.items():
                if node.status == "discarded" or node_id in tainted:
                    continue
                if any(dep in frontier or dep in tainted for dep in node.depends_on):
                    tainted.add(node_id)
                    changed = True
        return tainted

    def surviving_facts(self, invalid_roots: set[str]) -> list[VerifiedFact]:
        """Return verified facts that do not transitively depend on invalid_roots."""
        tainted_nodes = self.compute_tainted_nodes(invalid_roots)
        bad_ids = set(invalid_roots) | tainted_nodes
        fact_map = self.verified_facts

        def _is_tainted_fact(fid: str, visiting: set[str]) -> bool:
            if fid in bad_ids:
                return True
            if fid in visiting or fid not in fact_map:
                return False
            visiting.add(fid)
            for dep in fact_map[fid].depends_on:
                if dep in bad_ids or _is_tainted_fact(dep, visiting):
                    bad_ids.add(fid)
                    return True
            return False

        surviving: list[VerifiedFact] = []
        for fact in self.verified_facts.values():
            if not _is_tainted_fact(fact.id, set()):
                surviving.append(fact)
        return surviving

    def foundation_signature(self) -> str:
        """Return a stable digest of active assumptions and evidence to detect revision loops."""
        payload = {
            "evidence": self.evidence,
            "active_assumptions": sorted(
                (a.id, a.statement) for a in self.active_assumptions()
            ),
            "invalidated": sorted(self.invalidated_assumption_ids()),
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    def format_level_summary(self, max_level: int = 3, *, include_invalidated: bool = True) -> str:
        """Format current pyramid state concisely for model prompts."""
        lines: list[str] = [f"Objective: {self.objective or 'Solve the problem accurately.'}"]
        if self.core_constraints:
            lines.append("Core constraints:")
            lines.extend(f"- {c}" for c in self.core_constraints)

        active_a = self.active_assumptions()
        if active_a:
            lines.append("Level 1 — Active Assumptions:")
            for a in active_a:
                lines.append(f"- {a.id} ({a.source}): {a.statement}")

        if include_invalidated:
            invalid_a = [a for a in self.assumptions.values() if a.status == "invalidated"]
            if invalid_a:
                lines.append("Invalidated Assumptions (DO NOT RELY ON):")
                for a in invalid_a:
                    lines.append(f"- {a.id}: {a.statement}")

        if self.verified_facts:
            lines.append("Preserved Verified Facts:")
            for f in self.verified_facts.values():
                deps = ", ".join(f.depends_on) if f.depends_on else "independent"
                verified_suffix = (
                    f" (verified: {f.expr} = {f.value:g})"
                    if f.value is not None and f.expr
                    else (f" (verified = {f.value:g})" if f.value is not None else "")
                )
                lines.append(f"- {f.id} [depends_on: {deps}]: {f.statement}{verified_suffix}")

        if max_level >= 2:
            l2_nodes = self.active_nodes(level=2)
            if l2_nodes:
                lines.append("Level 2 — Decomposition & Strategy:")
                for n in l2_nodes:
                    deps = ", ".join(n.depends_on) if n.depends_on else "none"
                    lines.append(f"- {n.id} [{n.status}, depends_on: {deps}]: {n.statement}")

        if max_level >= 3:
            l3_nodes = self.active_nodes(level=3)
            if l3_nodes:
                lines.append("Level 3 — Detailed Reasoning:")
                for n in l3_nodes:
                    deps = ", ".join(n.depends_on) if n.depends_on else "none"
                    lines.append(f"- {n.id} [{n.status}, depends_on: {deps}]: {n.statement}")

        if self.candidate_answer is not None:
            lines.append(f"Current candidate answer: {json.dumps(self.candidate_answer)}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize PyramidState for experiment logs."""
        return {
            "problem_id": self.problem_id,
            "objective": self.objective,
            "core_constraints": list(self.core_constraints),
            "evidence": list(self.evidence),
            "assumptions": {k: asdict(v) for k, v in self.assumptions.items()},
            "nodes": {k: asdict(v) for k, v in self.nodes.items()},
            "verified_facts": {k: asdict(v) for k, v in self.verified_facts.items()},
            "candidate_answer": self.candidate_answer,
            "current_depth": self.current_depth,
            "reconstruction_count": self.reconstruction_count,
            "revision_log": [asdict(r) for r in self.revision_log],
        }
