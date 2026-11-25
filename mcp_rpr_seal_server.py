#!/usr/bin/env python3
"""Stdio Model Context Protocol (MCP) Server: `rpr-seal-rsi` (`v10` APEX — for Cline & Hermes).

Exposes 6 live tools with zero dual-process Metal GPU contention on Apple Silicon:
  1. `rpr_pyramid_solve`           : Multi-pass RPR Pyramid + Self-Discover + `python3 -I` + CiteCheck controller
  2. `rpr_belief_audit`            : ATMS Level-1 Assumption & AGM Mode-Lock belief-revision auditor
  3. `inspect_model_weights`       : Pure 16-bit (`bfloat16`) `3.18B` `40L` tensor, ULP, `Engram`/`AttnRes`/`LatentRecur` inspection
  4. `seal_outer_loop_self_rewrite`: Outer-Loop `Hybrid SEAL + Transformer^2 SVF + Nullspace + DGM` weight self-rewrite
  5. `realtime_web_research`       : Live multi-source web + arXiv + Wikipedia retrieval returning `[S1]..[Sn]` evidence
  6. `verify_citations_and_claims` : `CiteCheck` / `ALCE` inline `[S#]` citation & numeric entailment verifier
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


SERVER_NAME = "rpr-seal-rsi"
SERVER_VERSION = "2.0.0"
LOCAL_MLX_BASE = os.environ.get("RPR_LOCAL_MLX_BASE", "http://127.0.0.1:8991")
DEFAULT_V10_DIR = ROOT_DIR / "checkpoints" / "LFM3-VL-3.2B-RPR-V10-APEX-BF16"
FALLBACK_V9_DIR = ROOT_DIR / "checkpoints" / "LFM3-VL-2.9B-RPR-V9-SEAL-BF16"
FALLBACK_V8_DIR = ROOT_DIR / "checkpoints" / "LFM3-VL-2.9B-RPR-V8-HERMES-BF16"


TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "name": "rpr_pyramid_solve",
        "description": (
            "Solve a complex logic/SAT, math, research, zero-prior formal system, or coding problem using the live "
            "Multi-Pass Reconstructible Pyramid Reasoning (RPR) controller (Self-Discover + Level-1 Assumptions/Facts -> "
            "Level-2 Dependency Audit -> Level-3 Solve + isolated `python3 -I` non-mutation/TIR & CiteCheck verification)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "problem": {"type": "string", "description": "Full problem statement or code/math/logic specification."},
                "new_evidence": {"type": "string", "description": "Optional mid-session update or premise change to audit via ATMS."},
                "max_tokens": {"type": "integer", "description": "Max generation tokens per pyramid pass (default 640)."},
            },
            "required": ["problem"],
        },
    },
    {
        "name": "rpr_belief_audit",
        "description": (
            "Run an explicit ATMS Level-1 Assumption & AGM Mode-Lock Audit over `assumptions` and `facts` given `new_evidence`, "
            "identifying which assumptions remain VALID vs INVALID without discarding untainted verified facts."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "assumptions": {"type": "array", "items": {"type": "string"}, "description": "Active assumptions A1..An."},
                "facts": {"type": "array", "items": {"type": "string"}, "description": "Verified facts F1..Fm."},
                "new_evidence": {"type": "string", "description": "New observation, constraint change, or candidate distractor."},
            },
            "required": ["assumptions", "new_evidence"],
        },
    },
    {
        "name": "inspect_model_weights",
        "description": (
            "Inspect the active pure 16-bit (`bfloat16`) `LFM3-VL-3.2B-RPR-V10-APEX-BF16` (`40L`, `3.18B`, `256k` context) model weights: "
            "exact tensor shapes, `dtype` verification, Frobenius norms, `bfloat16` ULP spacing, and learned `Engram` (`out_scale`), "
            "`CrossLayerAttnRes` (`res_scale`), & `LatentReasoningRecurrence` (`loop_scales`) gates."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "layer_filter": {
                    "type": "string",
                    "description": "Substring filter for tensor names (e.g. 'latent_recur', 'engram', 'attn_res', 'layers.36', 'layers.38', or 'all').",
                },
                "max_tensors": {"type": "integer", "description": "Maximum matched tensors to list (default 20)."},
            },
            "required": ["layer_filter"],
        },
    },
    {
        "name": "seal_outer_loop_self_rewrite",
        "description": (
            "Execute a GPU-serialized, non-destructive Outer-Loop `Hybrid SEAL + Transformer^2 Prompt-Dispatched SVF + Orthogonal Nullspace "
            "+ Darwin Gödel Machine (DGM)` (`theta_parent -> theta_copy`) weight rewrite on upper reasoning layers (`26, 30, 36, 38` on `40L`) "
            "with isolated `python3 -I` unit-test verification, anchor-NLL rollback gating, and DGM evolutionary archive logging."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "diagnostic_prompt": {"type": "string", "description": "Target coding/math/logic prompt used to verify improvement."},
                "task_prompt": {"type": "string", "description": "Alias for `diagnostic_prompt`."},
                "verify_code": {"type": "string", "description": "Python `assert ...` unit-test snippet executed in `python3 -I`."},
                "python_unit_test": {"type": "string", "description": "Alias for `verify_code`."},
                "self_edit_response": {"type": "string", "description": "Optional verified `<think>...</think> ```python ... ``` ` self-edit trace."},
                "target_solution": {"type": "string", "description": "Alias for `self_edit_response`."},
                "steps": {"type": "integer", "description": "Inner nullspace-projected AdamW steps in [0, 8] (default 3)."},
                "svf_scale": {"type": "number", "description": "Base Transformer^2 SVF singular dispatch gain in [-1.0, 1.0] (default 0.36)."},
                "wise_ft_gamma": {"type": "number", "description": "WiSE-FT interpolation factor in [0.0, 1.0] (default 0.50)."},
                "nullspace_alpha": {"type": "number", "description": "Orthogonal nullspace protection strength in [0.0, 1.0] (default 0.95)."},
                "apply_live_session": {"type": "boolean", "description": "Keep accepted weights active in the live `:8991` session (default false)."},
                "reset_to_base": {"type": "boolean", "description": "Restore pristine base `v10` weights in `:8991` immediately."},
            },
        },
    },
    {
        "name": "realtime_web_research",
        "description": (
            "Perform live real-time multi-source web, arXiv, and Wikipedia retrieval (`Search-R1` / `WebThinker`), "
            "returning structured `[S1]..[Sn]` sources with URLs, publication dates, snippets, and page excerpts ready for inline `[S#]` citations."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Targeted search query for live verification."},
                "max_results": {"type": "integer", "description": "Maximum `[S1]..[Sn]` sources to return (default 5, max 10)."},
                "fetch_top_pages": {"type": "integer", "description": "Number of top organic pages to fetch clean body excerpts from (default 2)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "verify_citations_and_claims",
        "description": (
            "Verify inline `[S#]` or `[DOC-XXXX]` citations and numeric/factual claim grounding in `draft_answer` against `sources_json` "
            "(`CiteCheck` / `ALCE` precision, recall, F1, unknown citation detection, and unverified numeric claim detection)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "draft_answer": {"type": "string", "description": "Draft response containing inline `[S1]`, `[S2]` citations."},
                "sources_json": {"type": "string", "description": "JSON list of sources (from `realtime_web_research`) or raw `[S1] ...` text."},
            },
            "required": ["draft_answer", "sources_json"],
        },
    },
]


def _post_json(path: str, payload: dict[str, Any], timeout: float = 120.0) -> dict[str, Any]:
    url = f"{LOCAL_MLX_BASE.rstrip('/')}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _read_safetensors_header_only(st_path: Path, flt: str, limit: int) -> list[dict[str, Any]]:
    """Zero-GPU pure-CPU safetensors header reader fallback when `:8991` is unloaded."""
    import struct

    if not st_path.exists():
        return []
    with st_path.open("rb") as fh:
        hdr_len = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(hdr_len).decode("utf-8"))
    out = []
    f_low = (flt or "").strip().lower()
    for k, meta in hdr.items():
        if k == "__metadata__":
            continue
        if f_low and f_low not in ("all", "*") and f_low not in k.lower():
            continue
        shape = meta.get("shape", [])
        n_el = 1
        for d in shape:
            n_el *= int(d)
        out.append({"name": k, "shape": shape, "params": n_el, "dtype": meta.get("dtype", "BF16")})
        if len(out) >= limit:
            break
    return out


def _active_checkpoint_dir() -> Path:
    for cand in (DEFAULT_V10_DIR, FALLBACK_V9_DIR, FALLBACK_V8_DIR):
        if (cand / "model.safetensors").exists():
            return cand
    return FALLBACK_V9_DIR


def _call_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    if name == "rpr_pyramid_solve":
        return _post_json(
            "/v1/rpr/solve",
            {
                "mode": "solve",
                "problem": str(args.get("problem") or ""),
                "new_evidence": str(args.get("new_evidence") or ""),
                "max_tokens": int(args.get("max_tokens") or 640),
            },
        )
    if name == "rpr_belief_audit":
        return _post_json(
            "/v1/rpr/solve",
            {
                "mode": "audit",
                "assumptions": args.get("assumptions") or [],
                "facts": args.get("facts") or [],
                "new_evidence": str(args.get("new_evidence") or ""),
            },
        )
    if name == "inspect_model_weights":
        flt = str(args.get("layer_filter") or "latent_recur")
        lim = int(args.get("max_tensors") or 20)
        try:
            return _post_json("/v1/weights/inspect", {"layer_filter": flt, "max_tensors": lim}, timeout=30.0)
        except Exception as exc:
            mdir = _active_checkpoint_dir()
            rows = _read_safetensors_header_only(mdir / "v8_arch_ext.safetensors", flt, lim)
            if len(rows) < lim:
                rows.extend(_read_safetensors_header_only(mdir / "model.safetensors", flt, lim - len(rows)))
            return {
                "mode": "offline_header_inspection",
                "note": f"Server fallback ({exc}); read safetensors headers directly with 0 GPU memory.",
                "model_dir": str(mdir),
                "layer_filter": flt,
                "tensors": rows,
            }
    if name == "seal_outer_loop_self_rewrite":
        return _post_json("/v1/seal/rewrite", dict(args), timeout=180.0)
    if name == "realtime_web_research":
        from rpr_seal_runtime_ops import realtime_web_research_op

        return realtime_web_research_op(
            str(args.get("query") or ""),
            max_results=int(args.get("max_results") or 5),
            fetch_top_pages=int(args.get("fetch_top_pages") if args.get("fetch_top_pages") is not None else 2),
        )
    if name == "verify_citations_and_claims":
        from rpr_seal_runtime_ops import verify_citations_and_claims_op

        return verify_citations_and_claims_op(
            str(args.get("draft_answer") or ""),
            args.get("sources_json") if "sources_json" in args else args.get("sources", ""),
        )
    raise ValueError(f"Unknown tool: {name}")


def handle_rpc_message(msg: dict[str, Any]) -> dict[str, Any] | None:
    method = msg.get("method", "")
    msg_id = msg.get("id")
    params = msg.get("params") or {}

    if msg_id is None:
        return None

    if method == "initialize":
        req_proto = str(params.get("protocolVersion") or "2024-11-05")
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": req_proto,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            },
        }
    if method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS_SCHEMA}}
    if method == "resources/list":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"resources": []}}
    if method == "prompts/list":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"prompts": []}}
    if method == "tools/call":
        t_name = str(params.get("name") or "")
        t_args = params.get("arguments") or {}
        try:
            res = _call_tool(t_name, t_args)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(res, indent=2)}],
                    "isError": False,
                },
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps({"error": str(exc)}, indent=2)}],
                    "isError": True,
                },
            }
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def main() -> None:
    """Serve MCP over stdio supporting both newline-delimited JSON-RPC (MCP 2024-11-05+) and Content-Length framing."""
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    while True:
        first_line = stdin.readline()
        if not first_line:
            break
        stripped = first_line.strip()
        if not stripped:
            continue

        use_content_length = False
        if stripped.lower().startswith(b"content-length:"):
            use_content_length = True
            length = int(stripped.split(b":", 1)[1].strip())
            while True:
                hdr_line = stdin.readline()
                if not hdr_line or hdr_line in (b"\r\n", b"\n"):
                    break
            raw_body = stdin.read(length)
        else:
            raw_body = stripped

        try:
            req_msg = json.loads(raw_body.decode("utf-8"))
        except Exception:
            continue

        resp_msg = handle_rpc_message(req_msg)
        if resp_msg is None:
            continue
        out_bytes = json.dumps(resp_msg).encode("utf-8")
        if use_content_length:
            stdout.write(f"Content-Length: {len(out_bytes)}\r\n\r\n".encode("ascii") + out_bytes)
        else:
            stdout.write(out_bytes + b"\n")
        stdout.flush()


if __name__ == "__main__":
    main()
