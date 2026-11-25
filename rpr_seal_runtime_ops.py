"""Live Runtime Operations for `:8991` (`serve_local_mlx.py`) and Cline/Hermes `rpr-seal-rsi` MCP Server (`v10` APEX):

1. `inspect_live_or_disk_weights`: Inspects pure `16-bit` (`bfloat16`) `3.18B` `40L` (`v10`) or `2.91B` `36L` (`v9`) tensors,
   Frobenius norms, `bfloat16` ULP spacing, and learned `Engram`, `CrossLayerAttnRes`, & `LatentReasoningRecurrence` scales.
2. `run_live_outer_loop_seal_rewrite`: Executes a GPU-serialized, non-destructive Outer-Loop
   `Hybrid SEAL + Transformer^2 Prompt-Dispatched SVF + Orthogonal Nullspace + Darwin Gödel Machine (DGM)`
   (`theta_parent -> theta_copy`) weight rewrite on auto-detected upper `full_attention` layers (`(26, 30, 36, 38)` on `40L`,
   `(26, 30, 32, 34)` on `36L`) with `python3 -I` unit-test verification, anchor-NLL non-regression gating, and DGM lineage logging.
3. `realtime_web_research_op` & `verify_citations_and_claims_op`: Live multi-source web/Wikipedia/arXiv retrieval returning
   structured `[S1]..[Sn]` evidence + `CiteCheck`/`ALCE` inline citation & numeric entailment verification.
4. `run_rpr_endpoint_op`: Multi-pass RPR Pyramid solver (`problem` + `new_evidence`) & ATMS belief-revision auditor.
"""

from __future__ import annotations

import gc
import html
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Callable
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from model import extract_json_object, split_thinking
from rpr_live_controller import (
    extract_last_python_block,
    run_isolated_python,
    run_live_rpr_pyramid_turn,
)
from verifiers import values_match


ROOT_DIR = Path(__file__).resolve().parent
REWRITE_LAYERS_36L = (26, 30, 32, 34)
REWRITE_LAYERS_40L = (26, 30, 36, 38)
REWRITE_LAYERS = REWRITE_LAYERS_40L
OVERLAY_DIR = ROOT_DIR / "checkpoints" / "seal_user_overlays"
DGM_ARCHIVE_PATH = OVERLAY_DIR / "dgm_archive.json"

SVF_EXPERT_NAMES = (
    "b0_advanced_rsi",
    "b1_complex_logic",
    "b2_research_citations",
    "b3_ood_code",
)

_ANCHOR_PROMPTS = [
    (
        [{"role": "user", "content": 'Compute `17 * 13 - 29` and output ONLY JSON `{"ans": 192}`.'}],
        '{"ans": 192}',
    ),
    (
        [{"role": "user", "content": "Write a 1-line Python function `sq_plus_one(x)` returning `x * x + 1`."}],
        "```python\ndef sq_plus_one(x):\n    return x * x + 1\n```",
    ),
]


def _detect_rewrite_layers(vlm: Any) -> tuple[int, ...]:
    n_layers = len(vlm.language_model.model.layers)
    return REWRITE_LAYERS_40L if n_layers >= 40 else REWRITE_LAYERS_36L


def _bf16_ulp_at(val: float) -> float:
    a = abs(float(val))
    if a <= 0.0 or not math.isfinite(a):
        return 0.0
    exp_e = math.floor(math.log2(a))
    return float((2.0**exp_e) * (2.0**-7))


def dispatch_svf_gains_from_prompt(prompt_text: str, base_scale: float = 0.36) -> dict[str, Any]:
    """`Transformer^2` prompt-dispatched singular expert router across `(b0_rsi, b1_logic, b2_research, b3_code)`."""
    low = (prompt_text or "").lower()
    kw_sets = [
        ("weight", "svd", "svf", "nullspace", "lora", "ties", "memit", "alphaedit", "dgm", "grpo", "dapo", "rope", "bfloat16", "safeten"),
        ("clause", "knight", "knave", "sat", "boolean", "forall", "exists", "chromatic", "subtraction game", "induction", "bay", "slot"),
        ("[s1]", "cite", "citation", "erratum", "source", "preprint", "retention", "qps", "cohort", "insufficient_evidence", "research"),
        ("def ", "implement", "python", "list[", "dict[", "ndarray", "without mutating", "log_softmax", "deque", "graph", "ring"),
    ]
    raw_logits = []
    for kws in kw_sets:
        hits = sum(1.0 for kw in kws if kw in low)
        raw_logits.append(0.25 + 0.85 * hits)
    m_l = max(raw_logits)
    exps = [math.exp(x - m_l) for x in raw_logits]
    s_e = sum(exps)
    probs = [e / s_e for e in exps]
    top_idx = max(range(len(probs)), key=lambda i: probs[i])
    gains = [round(max(-1.0, min(1.0, base_scale * 2.2 * p)), 4) for p in probs]
    return {
        "svf_dispatch_expert": SVF_EXPERT_NAMES[top_idx],
        "expert_probabilities": {SVF_EXPERT_NAMES[i]: round(probs[i], 4) for i in range(4)},
        "svf_gains": gains,
    }


def select_dgm_archive_parent(beta: float = 0.12) -> tuple[str | None, dict[str, Any]]:
    """Select the highest exploration-adjusted non-regressing ancestor in `dgm_archive.json` (`Darwin Gödel Machine`)."""
    if not DGM_ARCHIVE_PATH.exists():
        default_doc = {
            "model": "LFM3-VL-3.2B-RPR-V10-APEX-BF16",
            "rewrite_layers": list(REWRITE_LAYERS_40L),
            "svf_experts": list(SVF_EXPERT_NAMES),
            "nodes": [
                {
                    "id": "dgm_gen_000_root",
                    "parent_id": None,
                    "score": 0.85,
                    "anchor_regression": 0.0,
                    "regressed": False,
                    "children": 0,
                    "overlay_path": None,
                    "created_at": int(time.time()),
                }
            ],
        }
        return "dgm_gen_000_root", default_doc

    try:
        doc = json.loads(DGM_ARCHIVE_PATH.read_text(encoding="utf-8"))
    except Exception:
        doc = {"nodes": []}
    nodes = doc.get("nodes") or []
    valid = [n for n in nodes if not n.get("regressed", False)]
    if not valid:
        return None, doc
    best = max(valid, key=lambda n: float(n.get("score", 0.0)) + beta / math.sqrt(1.0 + int(n.get("children", 0))))
    return str(best.get("id")), doc


def record_dgm_archive_node(
    doc: dict[str, Any],
    *,
    parent_id: str | None,
    accepted: bool,
    after_passed: bool,
    anchor_regression: float,
    overlay_path: str | None,
    svf_expert: str,
    prompt_preview: str,
) -> str:
    """Append a verified generation node to `dgm_archive.json` and update parent child count."""
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    nodes = list(doc.get("nodes") or [])
    new_id = f"dgm_gen_{len(nodes):03d}"
    if accepted and parent_id is not None:
        for n in nodes:
            if n.get("id") == parent_id:
                n["children"] = int(n.get("children", 0)) + 1
    nodes.append(
        {
            "id": new_id,
            "parent_id": parent_id,
            "score": 1.0 if after_passed else 0.0,
            "anchor_regression": round(float(anchor_regression), 5),
            "regressed": bool(not accepted or anchor_regression > 0.025),
            "accepted": bool(accepted),
            "children": 0,
            "svf_dispatch_expert": svf_expert,
            "overlay_path": overlay_path,
            "task_preview": prompt_preview[:140],
            "created_at": int(time.time()),
        }
    )
    doc["nodes"] = nodes
    DGM_ARCHIVE_PATH.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return new_id


def inspect_live_or_disk_weights(
    vlm: Any | None,
    model_dir: str | Path,
    *,
    layer_filter: str = "engram",
    max_tensors: int = 24,
) -> dict[str, Any]:
    """Return exact shapes, dtypes, Frobenius norms, `bfloat16` ULP spacing, and learned `Engram`/`AttnRes`/`LatentRecur` scales."""
    mdir = Path(model_dir)
    flt = (layer_filter or "").strip().lower()

    cfg_meta: dict[str, Any] = {}
    cfg_p = mdir / "config.json"
    if cfg_p.exists():
        try:
            tcfg = json.loads(cfg_p.read_text(encoding="utf-8")).get("text_config", {})
            cfg_meta = {
                "num_hidden_layers": tcfg.get("num_hidden_layers"),
                "max_position_embeddings": tcfg.get("max_position_embeddings"),
                "full_attention_layers": sum(1 for x in tcfg.get("layer_types", []) if x == "full_attention"),
            }
        except Exception:
            cfg_meta = {}

    if vlm is not None:
        flat_params = list(tree_flatten(vlm.parameters()))
    else:
        flat_params = []
        for st_name in ("v8_arch_ext.safetensors", "model.safetensors"):
            st_p = mdir / st_name
            if st_p.exists():
                loaded = mx.load(str(st_p))
                flat_params.extend(list(loaded.items()))

    total_params = sum(int(v.size) for _, v in flat_params)
    dtypes_seen = sorted({str(v.dtype).replace("mlx.core.", "") for _, v in flat_params})

    matched: list[dict[str, Any]] = []
    scale_summary: dict[str, Any] = {}

    for name, arr in flat_params:
        if "out_scale" in name or "res_scale" in name or "loop_scales" in name:
            af = arr.astype(mx.float32)
            mx.eval(af)
            scale_summary[name] = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype).replace("mlx.core.", ""),
                "l2_norm": round(float(mx.sqrt(mx.sum(af * af)).item()), 5),
                "mean_abs": round(float(mx.mean(mx.abs(af)).item()), 5),
                "max_abs": round(float(mx.max(mx.abs(af)).item()), 5),
            }
        if flt and flt not in ("all", "*") and flt not in name.lower():
            continue
        if len(matched) >= max_tensors:
            continue
        af = arr.astype(mx.float32)
        frob = float(mx.sqrt(mx.sum(af * af)).item())
        max_a = float(mx.max(mx.abs(af)).item()) if arr.size > 0 else 0.0
        mean_a = float(mx.mean(mx.abs(af)).item()) if arr.size > 0 else 0.0
        matched.append(
            {
                "name": name,
                "shape": list(arr.shape),
                "params": int(arr.size),
                "dtype": str(arr.dtype).replace("mlx.core.", ""),
                "frobenius_norm": round(frob, 5),
                "mean_abs": round(mean_a, 6),
                "max_abs": round(max_a, 6),
                "bf16_ulp_at_max": _bf16_ulp_at(max_a),
            }
        )

    return {
        "model_dir": str(mdir),
        **cfg_meta,
        "total_parameters": total_params,
        "total_billions": round(total_params / 1e9, 4),
        "dtypes": dtypes_seen,
        "pure_16bit_bfloat16": ("uint32" not in dtypes_seen and "uint8" not in dtypes_seen),
        "layer_filter": layer_filter,
        "matched_count": len(matched),
        "tensors": matched,
        "extension_scales": scale_summary,
        "seal_bank_available": (mdir / "seal_nullspace_svf_bank.safetensors").exists(),
        "dgm_archive_available": DGM_ARCHIVE_PATH.exists(),
    }


def _compute_anchor_nll(vlm: Any, processor: Any) -> float:
    from mlx_vlm.prompt_utils import apply_chat_template

    tok = processor.tokenizer
    eos_id = int(tok.eos_token_id)
    losses: list[float] = []
    for msgs, target_ans in _ANCHOR_PROMPTS:
        p_s = apply_chat_template(processor, vlm.config, msgs, num_images=0, enable_thinking=False)
        p_ids = list(tok.encode(p_s, add_special_tokens=False))
        a_ids = list(tok.encode(target_ans, add_special_tokens=False)) + [eos_id]
        full = mx.array([p_ids + a_ids], dtype=mx.int32)
        logits = vlm.language_model(full[:, :-1]).logits.astype(mx.float32)
        s_i = max(0, len(p_ids) - 1)
        ce = nn.losses.cross_entropy(logits[:, s_i:, :], full[:, s_i + 1 :], reduction="mean")
        mx.eval(ce)
        losses.append(float(ce.item()))
    return round(sum(losses) / max(1, len(losses)), 5)


def _eval_diagnostic_probe(
    vlm: Any,
    processor: Any,
    diagnostic_prompt: str,
    verify_code: str,
    expected_json: dict[str, Any] | None,
) -> tuple[bool, str, str]:
    import mlx_vlm
    from mlx_vlm.prompt_utils import apply_chat_template

    msgs = [{"role": "user", "content": diagnostic_prompt}]
    p_s = apply_chat_template(processor, vlm.config, msgs, num_images=0, enable_thinking=True)
    res = mlx_vlm.generate(vlm, processor, p_s, max_tokens=448, temperature=0.0, repetition_penalty=1.03, verbose=False)
    raw = res.text or ""
    norm = ("<think>" + raw) if (p_s.endswith("<think>") and not raw.lstrip().startswith("<think>")) else raw

    if verify_code.strip():
        py_blk = extract_last_python_block(norm) or ""
        if not py_blk:
            return False, norm, "No Python block generated"
        ok, _, err = run_isolated_python(f"{py_blk}\n\n{verify_code}\n", timeout_sec=2.5)
        return ok, norm, err

    if expected_json:
        parsed = extract_json_object(norm)
        if isinstance(parsed, dict) and all(values_match(parsed.get(k), v) for k, v in expected_json.items()):
            return True, norm, ""
        return False, norm, f"JSON mismatch: got {parsed}"

    return ("</think>" in norm and len(norm.strip()) > 16), norm, ""


def run_live_outer_loop_seal_rewrite(
    vlm: Any,
    processor: Any,
    model_dir: str | Path,
    req: dict[str, Any],
    *,
    pristine_snap: dict[int, dict[str, mx.array]] | None = None,
) -> dict[str, Any]:
    """Execute a GPU-serialized Outer-Loop `Hybrid SEAL + Transformer^2 SVF + Orthogonal Nullspace + DGM`
    (`theta_parent -> theta_copy`) weight rewrite on auto-detected upper `full_attention` layers.
    """
    from mlx_vlm.prompt_utils import apply_chat_template

    mdir = Path(model_dir)
    bank_path = mdir / "seal_nullspace_svf_bank.safetensors"
    if not bank_path.exists():
        return {"status": "error", "error": f"SEAL Nullspace+SVF bank not found at {bank_path}"}

    target_layers = _detect_rewrite_layers(vlm)

    if req.get("reset_to_base") and pristine_snap is not None:
        for idx, d in pristine_snap.items():
            lyr = vlm.language_model.model.layers[idx]
            lyr.self_attn.o_proj.weight = mx.array(d["o_proj"])
            lyr.feed_forward.w3.weight = mx.array(d["w3"])
        mx.eval([vlm.language_model.model.layers[idx].self_attn.o_proj.weight for idx in pristine_snap])
        return {"status": "ok", "action": "reset_to_pristine_base", "layers_restored": list(target_layers)}

    t0 = time.perf_counter()
    bank = mx.load(str(bank_path))
    mx.eval(list(bank.values()))

    base_snap: dict[int, dict[str, mx.array]] = {}
    for idx in target_layers:
        lyr = vlm.language_model.model.layers[idx]
        base_snap[idx] = {
            "o_proj": mx.array(lyr.self_attn.o_proj.weight),
            "w3": mx.array(lyr.feed_forward.w3.weight),
        }
    mx.eval([arr for d in base_snap.values() for arr in d.values()])

    # Support both `diagnostic_prompt`/`verify_code`/`self_edit_response` and `task_prompt`/`python_unit_test`/`target_solution`
    diag_prompt = str(
        req.get("diagnostic_prompt")
        or req.get("task_prompt")
        or "Implement `svf_diag_scale(W: np.ndarray, g: np.ndarray) -> np.ndarray` returning `(U * (S * g)[None, :]) @ Vt` via economy SVD."
    )
    verify_code = str(
        req.get("verify_code")
        or req.get("python_unit_test")
        or "import numpy as np\nW = np.diag([4.0, 2.0])\nassert np.allclose(svf_diag_scale(W, np.array([1.5, 0.5])), np.diag([6.0, 1.0]))"
    )
    expected_json = req.get("expected_json") if isinstance(req.get("expected_json"), dict) else None

    base_scale = max(-1.0, min(1.0, float(req.get("svf_scale", 0.36))))
    dispatch_info = dispatch_svf_gains_from_prompt(diag_prompt, base_scale=base_scale)
    raw_gains = req.get("svf_gains")
    if isinstance(raw_gains, list) and len(raw_gains) == 4:
        svf_gains = [max(-1.0, min(1.0, float(x))) for x in raw_gains]
    else:
        svf_gains = dispatch_info["svf_gains"]

    gamma = max(0.0, min(1.0, float(req.get("wise_ft_gamma", 0.50))))
    null_alpha = max(0.0, min(1.0, float(req.get("nullspace_alpha", 0.95))))
    inner_steps = max(0, min(8, int(req.get("steps", 3))))
    lr_b = max(0.0, min(5e-5, float(req.get("lr_b", 1.2e-5))))
    max_nll_drift = float(req.get("max_nll_drift", 0.025))
    apply_live = bool(req.get("apply_live_session", False))

    parent_dgm_id, dgm_doc = select_dgm_archive_parent(beta=0.12)

    vlm.eval()
    before_nll = _compute_anchor_nll(vlm, processor)
    before_ok, before_out, before_err = _eval_diagnostic_probe(vlm, processor, diag_prompt, verify_code, expected_json)

    self_edit_resp = str(req.get("self_edit_response") or req.get("target_solution") or "").strip()
    verified_self_edit = False
    if self_edit_resp and verify_code.strip():
        cand_py = extract_last_python_block(self_edit_resp) or self_edit_resp
        ok_se, _, _ = run_isolated_python(f"{cand_py}\n\n{verify_code}\n", timeout_sec=2.5)
        verified_self_edit = ok_se

    seal_deltas: dict[tuple[int, str], mx.array] = {}
    if verified_self_edit and inner_steps > 0 and lr_b > 0.0:
        tok = processor.tokenizer
        eos_id = int(tok.eos_token_id)
        p_s = apply_chat_template(processor, vlm.config, [{"role": "user", "content": diag_prompt}], num_images=0, enable_thinking=True)
        ans_s = self_edit_resp[len("<think>") :] if (p_s.endswith("<think>") and self_edit_resp.startswith("<think>")) else self_edit_resp
        p_ids = list(tok.encode(p_s, add_special_tokens=False))
        a_ids = list(tok.encode(ans_s, add_special_tokens=False)) + [eos_id]
        full_ids = mx.array([(p_ids + a_ids)[:1536]], dtype=mx.int32)
        s_i = max(0, len(p_ids) - 1)

        vlm.freeze()
        for idx in target_layers:
            lyr = vlm.language_model.model.layers[idx]
            lyr.self_attn.o_proj.unfreeze()
            lyr.feed_forward.w3.unfreeze()

        inner_opt = optim.AdamW(learning_rate=lr_b, weight_decay=0.0)

        def _ce_fn(m: nn.Module, seq: mx.array) -> mx.array:
            lg = m.language_model(seq[:, :-1]).logits.astype(mx.float32)
            return nn.losses.cross_entropy(lg[:, s_i:, :], seq[:, s_i + 1 :], reduction="mean")

        vg = nn.value_and_grad(vlm, _ce_fn)
        for _ in range(inner_steps):
            lv, gr = vg(vlm, full_ids)
            gr, _ = optim.clip_grad_norm(gr, max_norm=1.0)
            inner_opt.update(vlm, gr)
            mx.eval(lv, vlm.trainable_parameters(), inner_opt.state)
        vlm.freeze()
        vlm.eval()

        for idx in target_layers:
            lyr = vlm.language_model.model.layers[idx]
            seal_deltas[(idx, "o_proj")] = lyr.self_attn.o_proj.weight.astype(mx.float32) - base_snap[idx]["o_proj"].astype(mx.float32)
            seal_deltas[(idx, "w3")] = lyr.feed_forward.w3.weight.astype(mx.float32) - base_snap[idx]["w3"].astype(mx.float32)

    delta_norms: dict[str, float] = {}
    overlay_tensors: dict[str, mx.array] = {}
    for idx in target_layers:
        lyr = vlm.language_model.model.layers[idx]
        for short_m, p_tag, mod_obj in [
            ("o_proj", "attn_o", lyr.self_attn.o_proj),
            ("w3", "ffn_in", lyr.feed_forward.w3),
        ]:
            w0 = base_snap[idx][short_m].astype(mx.float32)
            svf_sum = mx.zeros_like(w0)
            for b_i, g_val in enumerate(svf_gains):
                key_b = f"svf_b{b_i}.L{idx}.{short_m}"
                if abs(g_val) > 1e-6 and key_b in bank:
                    svf_sum = svf_sum + float(g_val) * bank[key_b].astype(mx.float32)
            d_seal = seal_deltas.get((idx, short_m))
            p_key = f"L{idx}.{p_tag}.P_perp"
            if d_seal is not None and p_key in bank:
                p_perp = bank[p_key].astype(mx.float32)
                d_proj = (1.0 - null_alpha) * d_seal + null_alpha * (d_seal @ p_perp)
                tot_delta = gamma * (svf_sum + d_proj)
            else:
                tot_delta = gamma * svf_sum
            w_new = (w0 + tot_delta).astype(mx.bfloat16)
            mod_obj.weight = w_new
            mx.eval(w_new)
            dn = float(mx.sqrt(mx.sum(tot_delta * tot_delta)).item())
            delta_norms[f"L{idx}.{short_m}"] = round(dn, 5)
            overlay_tensors[f"language_model.model.layers.{idx}.{'self_attn.o_proj' if short_m == 'o_proj' else 'feed_forward.w3'}.weight"] = w_new

    vlm.eval()
    after_nll = _compute_anchor_nll(vlm, processor)
    after_ok, after_out, after_err = _eval_diagnostic_probe(vlm, processor, diag_prompt, verify_code, expected_json)

    anchor_reg = round(after_nll - before_nll, 5)
    nll_ok = after_nll <= before_nll + max_nll_drift
    accepted = bool(after_ok and nll_ok)
    overlay_path_str: str | None = None

    if accepted:
        OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        ov_path = OVERLAY_DIR / f"seal_overlay_{ts}.safetensors"
        mx.save_safetensors(
            str(ov_path),
            overlay_tensors,
            metadata={"format": "mlx", "nullspace_alpha": str(null_alpha), "svf_expert": dispatch_info["svf_dispatch_expert"]},
        )
        overlay_path_str = str(ov_path)

    dgm_node_id = record_dgm_archive_node(
        dgm_doc,
        parent_id=parent_dgm_id,
        accepted=accepted,
        after_passed=after_ok,
        anchor_regression=anchor_reg,
        overlay_path=overlay_path_str,
        svf_expert=dispatch_info["svf_dispatch_expert"],
        prompt_preview=diag_prompt,
    )

    kept_live = bool(accepted and apply_live)
    if not kept_live:
        for idx, d in base_snap.items():
            lyr = vlm.language_model.model.layers[idx]
            lyr.self_attn.o_proj.weight = mx.array(d["o_proj"])
            lyr.feed_forward.w3.weight = mx.array(d["w3"])
        mx.eval([vlm.language_model.model.layers[idx].self_attn.o_proj.weight for idx in base_snap])

    gc.collect()
    mx.clear_cache()
    return {
        "status": "ok",
        "accepted_by_gate": accepted,
        "kept_in_live_session": kept_live,
        "target_layers": list(target_layers),
        "svf_dispatch_expert": dispatch_info["svf_dispatch_expert"],
        "svf_expert_probabilities": dispatch_info["expert_probabilities"],
        "svf_gains": svf_gains,
        "dgm_parent_node": parent_dgm_id,
        "dgm_archive_node": dgm_node_id,
        "saved_overlay_path": overlay_path_str,
        "verified_self_edit_used": verified_self_edit,
        "before": {"diagnostic_passed": before_ok, "anchor_nll": before_nll, "error": before_err[:240]},
        "after": {"diagnostic_passed": after_ok, "anchor_nll": after_nll, "error": after_err[:240]},
        "anchor_regression": anchor_reg,
        "nullspace_alpha": null_alpha,
        "wise_ft_gamma": gamma,
        "delta_frobenius_norms": delta_norms,
        "wall_clock_sec": round(time.perf_counter() - t0, 3),
    }


# ===========================================================================
# Live Real-Time Web Research & `CiteCheck` Inline Citation Grounding Verifier
# ===========================================================================


class _CleanTextHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg", "nav", "footer", "header"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg", "nav", "footer", "header"} and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            s = data.strip()
            if s:
                self.chunks.append(s)


def _html_to_clean_text(raw_html: str, max_chars: int = 900) -> str:
    parser = _CleanTextHTMLParser()
    try:
        parser.feed(raw_html)
    except Exception:
        pass
    joined = re.sub(r"\s+", " ", " ".join(parser.chunks)).strip()
    return joined[:max_chars]


def _http_get_text(url: str, timeout: float = 6.0) -> str:
    import ssl
    import subprocess

    ua = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) RPR-Research-Agent/10.0"
    cafile = "/etc/ssl/cert.pem" if Path("/etc/ssl/cert.pem").exists() else None
    try:
        import certifi  # type: ignore[import-not-found]

        cafile = certifi.where()
    except Exception:
        pass
    try:
        ctx = ssl.create_default_context(cafile=cafile) if cafile else ssl.create_default_context()
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": ua,
                "Accept": "text/html,application/json,application/xml;q=0.9,*/*;q=0.8",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            raw = resp.read(220_000)
            return raw.decode("utf-8", errors="replace")
    except Exception:
        # macOS native Keychain/SecureTransport fallback via system `/usr/bin/curl`
        proc = subprocess.run(
            ["/usr/bin/curl", "-fsSL", "--max-time", str(max(2, int(timeout))), "-A", ua, url],
            capture_output=True,
            timeout=timeout + 2.0,
            check=True,
        )
        return proc.stdout[:220_000].decode("utf-8", errors="replace")


def realtime_web_research_op(
    query: str,
    *,
    max_results: int = 5,
    fetch_top_pages: int = 2,
) -> dict[str, Any]:
    """Retrieve live multi-source web, Wikipedia, and arXiv results formatted as `[S1]..[Sn]` with URLs, dates, and snippets."""
    q_clean = (query or "").strip()
    if not q_clean:
        return {"status": "error", "error": "Empty research query", "sources": []}

    max_results = max(1, min(10, int(max_results)))
    fetch_top_pages = max(0, min(4, int(fetch_top_pages)))
    candidates: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    errors: list[str] = []

    # 1. DuckDuckGo HTML organic search
    try:
        ddg_url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": q_clean})
        ddg_html = _http_get_text(ddg_url, timeout=6.0)
        link_matches = re.findall(
            r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            ddg_html,
            flags=re.DOTALL | re.IGNORECASE,
        )
        snip_matches = re.findall(
            r'<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>|<div[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</div>',
            ddg_html,
            flags=re.DOTALL | re.IGNORECASE,
        )
        for idx_m, (href, raw_title) in enumerate(link_matches[:max_results]):
            real_url = html.unescape(href)
            if real_url.startswith("//"):
                real_url = "https:" + real_url
            if "uddg=" in real_url:
                parsed_q = urllib.parse.parse_qs(urllib.parse.urlparse(real_url).query)
                if parsed_q.get("uddg"):
                    real_url = urllib.parse.unquote(parsed_q["uddg"][0])
            if not real_url.startswith("http") or real_url in seen_urls:
                continue
            seen_urls.add(real_url)
            title_txt = re.sub(r"<[^>]+>", "", html.unescape(raw_title)).strip()
            raw_sn = ""
            if idx_m < len(snip_matches):
                raw_sn = snip_matches[idx_m][0] or snip_matches[idx_m][1]
            snip_txt = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(raw_sn))).strip()
            candidates.append(
                {
                    "title": title_txt or real_url,
                    "url": real_url,
                    "source_type": "web",
                    "published": None,
                    "snippet": snip_txt[:420],
                }
            )
    except Exception as exc:
        errors.append(f"ddg:{exc}")

    # 2. OpenAlex BM25 Academic Search (250M+ peer-reviewed & arXiv works with dates, citations, & abstracts!)
    if len(candidates) < max_results:
        try:
            oa_url = "https://api.openalex.org/works?" + urllib.parse.urlencode(
                {"search": q_clean, "per-page": min(3, max_results)}
            )
            oa_doc = json.loads(_http_get_text(oa_url, timeout=6.0))
            for wk in oa_doc.get("results", []):
                ttl = str(wk.get("display_name") or wk.get("title") or "").strip()
                u_str = str(wk.get("doi") or wk.get("id") or "").strip()
                if not ttl or not u_str or u_str in seen_urls:
                    continue
                seen_urls.add(u_str)
                inv = wk.get("abstract_inverted_index") or {}
                abs_words: list[tuple[int, str]] = []
                if isinstance(inv, dict):
                    for wd, positions in inv.items():
                        for pos in (positions or []):
                            abs_words.append((int(pos), str(wd)))
                abs_words.sort(key=lambda x: x[0])
                abs_txt = " ".join(w for _, w in abs_words[:85])
                cited = int(wk.get("cited_by_count") or 0)
                snip = f"{abs_txt} (Cited by {cited})".strip() if abs_txt else f"Academic work cited by {cited}."
                candidates.append(
                    {
                        "title": f"[OpenAlex] {ttl}",
                        "url": u_str,
                        "source_type": "academic",
                        "published": str(wk.get("publication_date") or "")[:10] or None,
                        "snippet": snip[:460],
                    }
                )
        except Exception as exc:
            errors.append(f"openalex:{exc}")

    # 3. arXiv Atom API with strict `AND`-joined keywords (`all:w1 AND all:w2`)
    if len(candidates) < max_results:
        try:
            q_terms = [re.sub(r"[^A-Za-z0-9_-]", "", w) for w in q_clean.split() if len(w) >= 3][:6]
            ax_q = " AND ".join(f"all:{w}" for w in q_terms) if q_terms else f"all:{q_clean}"
            ax_url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
                {"search_query": ax_q, "start": 0, "max_results": min(3, max_results)}
            )
            ax_xml = _http_get_text(ax_url, timeout=6.0)
            root = ET.fromstring(ax_xml)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            for entry in root.findall("atom:entry", ns):
                id_el = entry.find("atom:id", ns)
                title_el = entry.find("atom:title", ns)
                sum_el = entry.find("atom:summary", ns)
                pub_el = entry.find("atom:published", ns)
                u_str = (id_el.text or "").strip() if id_el is not None else ""
                if not u_str or u_str in seen_urls:
                    continue
                seen_urls.add(u_str)
                t_str = re.sub(r"\s+", " ", (title_el.text or "").strip()) if title_el is not None else u_str
                s_str = re.sub(r"\s+", " ", (sum_el.text or "").strip()) if sum_el is not None else ""
                p_str = (pub_el.text or "")[:10] if pub_el is not None else None
                candidates.append(
                    {
                        "title": f"[arXiv] {t_str}",
                        "url": u_str,
                        "source_type": "arxiv",
                        "published": p_str,
                        "snippet": s_str[:460],
                    }
                )
        except Exception as exc:
            errors.append(f"arxiv:{exc}")

    # 4. Wikipedia Search API enrichment
    if len(candidates) < max_results:
        try:
            wp_url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
                {"action": "query", "list": "search", "srsearch": q_clean, "srlimit": 2, "format": "json"}
            )
            wp_doc = json.loads(_http_get_text(wp_url, timeout=5.0))
            for hit in wp_doc.get("query", {}).get("search", []):
                ttl = str(hit.get("title") or "").strip()
                if not ttl:
                    continue
                page_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(ttl.replace(' ', '_'))}"
                if page_url in seen_urls:
                    continue
                seen_urls.add(page_url)
                sn = re.sub(r"<[^>]+>", "", html.unescape(str(hit.get("snippet") or ""))).strip()
                candidates.append(
                    {
                        "title": f"[Wikipedia] {ttl}",
                        "url": page_url,
                        "source_type": "wikipedia",
                        "published": str(hit.get("timestamp") or "")[:10] or None,
                        "snippet": sn[:400],
                    }
                )
        except Exception as exc:
            errors.append(f"wikipedia:{exc}")

    selected = candidates[:max_results]
    fetched = 0
    for item in selected:
        if fetched >= fetch_top_pages or item["source_type"] == "arxiv":
            item["page_excerpt"] = ""
            continue
        try:
            raw_pg = _http_get_text(item["url"], timeout=4.5)
            item["page_excerpt"] = _html_to_clean_text(raw_pg, max_chars=850)
            fetched += 1
        except Exception:
            item["page_excerpt"] = ""

    sources: list[dict[str, Any]] = []
    ctx_lines: list[str] = []
    for idx_s, it in enumerate(selected, start=1):
        sid = f"S{idx_s}"
        pub_note = f", {it['published']}" if it.get("published") else ""
        body_txt = it["snippet"]
        if it.get("page_excerpt") and len(it["page_excerpt"]) > len(body_txt):
            body_txt = f"{body_txt} | Excerpt: {it['page_excerpt'][:500]}"
        src_entry = {
            "id": sid,
            "tag": f"[{sid}]",
            "title": it["title"],
            "url": it["url"],
            "source_type": it["source_type"],
            "published": it["published"],
            "snippet": it["snippet"],
            "page_excerpt": it.get("page_excerpt", ""),
        }
        sources.append(src_entry)
        ctx_lines.append(f"[{sid}] ({it['title']}{pub_note} — {it['url']}): {body_txt}")

    return {
        "status": "ok" if sources else "no_results",
        "query": q_clean,
        "count": len(sources),
        "sources": sources,
        "formatted_context": "\n".join(ctx_lines),
        "fetch_notes": errors[:3],
    }


def _parse_sources_input(sources_in: Any) -> dict[str, str]:
    """Normalize `sources` (JSON string, list of dicts, or raw `[S1] ...` text) into `{'S1': full_text, ...}`."""
    src_map: dict[str, str] = {}
    if isinstance(sources_in, str):
        s_str = sources_in.strip()
        if s_str.startswith("[") or s_str.startswith("{"):
            try:
                parsed = json.loads(s_str)
                return _parse_sources_input(parsed)
            except Exception:
                pass
        for m in re.finditer(r"\[(S\d+|DOC-\d{4}|REC-\d{4})\]\s*(.*?)(?=\n\[(?:S\d+|DOC-\d{4}|REC-\d{4})\]|\Z)", s_str, flags=re.DOTALL):
            src_map[m.group(1)] = m.group(2).strip()
        return src_map
    if isinstance(sources_in, dict) and "sources" in sources_in:
        return _parse_sources_input(sources_in["sources"])
    if isinstance(sources_in, list):
        for idx, item in enumerate(sources_in, start=1):
            if isinstance(item, dict):
                sid = str(item.get("id") or f"S{idx}").strip("[]")
                txt = " ".join(str(item.get(k) or "") for k in ("title", "published", "snippet", "page_excerpt", "text", "content")).strip()
                src_map[sid] = txt
            elif isinstance(item, str):
                src_map[f"S{idx}"] = item
    return src_map


def verify_citations_and_claims_op(draft_answer: str, sources_in: Any) -> dict[str, Any]:
    """`CiteCheck` / `ALCE`-style inline citation & numeric entailment verifier."""
    src_map = _parse_sources_input(sources_in)
    valid_ids = set(src_map.keys())
    _, clean_body = split_thinking(draft_answer if "<think>" in draft_answer else f"<think></think>{draft_answer}")
    body = clean_body.strip() or (draft_answer or "").strip()
    # Strip trailing standalone JSON block when splitting prose claims
    prose_only = re.sub(r"\{[^{}]*\}\s*$", "", body).strip()

    all_tags = re.findall(r"\[(S\d+|DOC-\d{4}|REC-\d{4})\]", body)
    unknown_tags = sorted({t for t in all_tags if t not in valid_ids})
    used_valid = sorted({t for t in all_tags if t in valid_ids})

    raw_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", prose_only) if len(s.strip()) >= 12]
    uncited_sentences: list[str] = []
    numeric_mismatches: list[dict[str, Any]] = []
    supported_sentences = 0
    factual_sentences = 0

    for sent in raw_sents:
        if sent.startswith("#") or sent.startswith("```"):
            continue
        factual_sentences += 1
        sent_tags = re.findall(r"\[(S\d+|DOC-\d{4}|REC-\d{4})\]", sent)
        if not sent_tags:
            uncited_sentences.append(sent[:180])
            continue
        valid_sent_tags = [t for t in sent_tags if t in valid_ids]
        if not valid_sent_tags or any(t not in valid_ids for t in sent_tags):
            continue

        joined_ev = " ".join(src_map[t] for t in valid_sent_tags).replace(",", "")
        sent_no_tags = re.sub(r"\[(?:S\d+|DOC-\d{4}|REC-\d{4})\]", "", sent).replace(",", "")
        nums_in_sent = re.findall(r"\b\d+\.\d+\b|\b\d{2,}\b", sent_no_tags)
        ev_nums = [float(x) for x in re.findall(r"\b\d+(?:\.\d+)?\b", joined_ev)]

        mismatch_found = False
        for num_s in nums_in_sent:
            if num_s in joined_ev:
                continue
            val_f = float(num_s)
            # Check if `val_f` equals a simple 2-number product or sum from the cited sources!
            derived_ok = any(
                abs(val_f - (a * b)) <= 1e-2 or abs(val_f - (a + b)) <= 1e-2
                for a in ev_nums[:16]
                for b in ev_nums[:16]
            )
            if not derived_ok:
                mismatch_found = True
                numeric_mismatches.append({"sentence": sent[:160], "unverified_number": num_s, "cited": valid_sent_tags})
                break

        if not mismatch_found:
            supported_sentences += 1

    recall = round(supported_sentences / max(1, factual_sentences), 4) if factual_sentences > 0 else (1.0 if used_valid and not unknown_tags else 0.0)
    precision = round((len(all_tags) - sum(1 for t in all_tags if t not in valid_ids)) / max(1, len(all_tags)), 4) if all_tags else 0.0
    if numeric_mismatches and precision > 0.0:
        precision = round(max(0.0, precision - 0.25 * len(numeric_mismatches)), 4)
    f1 = round(2 * precision * recall / max(1e-9, precision + recall), 4) if (precision + recall) > 0 else 0.0

    is_verified = bool(len(all_tags) > 0 and len(unknown_tags) == 0 and len(numeric_mismatches) == 0 and len(uncited_sentences) == 0)
    return {
        "status": "ok",
        "is_verified": is_verified,
        "citation_precision": precision,
        "citation_recall": recall,
        "citation_f1": f1,
        "total_tags": len(all_tags),
        "used_valid_citations": used_valid,
        "unknown_citations": unknown_tags,
        "factual_sentences": factual_sentences,
        "supported_sentences": supported_sentences,
        "uncited_claims": len(uncited_sentences),
        "uncited_sentence_previews": uncited_sentences[:5],
        "numeric_mismatches": numeric_mismatches[:5],
    }


def run_rpr_endpoint_op(
    generate_fn: Callable[[list[dict[str, Any]], int, bool], str],
    req: dict[str, Any],
) -> dict[str, Any]:
    """Handle `/v1/rpr/solve` & `/v1/rpr/audit` requests using the active `v10` (`3.18B` `40L` `256k`) model."""
    op_mode = str(req.get("mode") or "solve").strip().lower()
    if op_mode == "audit":
        assumptions = req.get("assumptions") or []
        facts = req.get("facts") or []
        evidence = str(req.get("new_evidence") or "")
        a_block = "\n".join(f"- A{i+1}: {a}" for i, a in enumerate(assumptions)) or "- A1: Initial regime active"
        f_block = "\n".join(f"- F{i+1}: {f}" for i, f in enumerate(facts)) or "- F1: Invariant baseline"
        msgs = [
            {
                "role": "system",
                "content": (
                    "You are an ATMS Level-1 Belief-Revision Auditor. Check each assumption A_i against the New Evidence "
                    "(distinguishing live regime switches from non-binding hypotheticals/QA distractors!). "
                    "Output `CHECK A1: VALID | INVALID - <reason>` lines followed by `SUMMARY: VALID | INVALID` and which facts F_j remain preserved."
                ),
            },
            {
                "role": "user",
                "content": f"Assumptions:\n{a_block}\n\nVerified Facts:\n{f_block}\n\nNew Evidence:\n{evidence}",
            },
        ]
        raw = generate_fn(msgs, int(req.get("max_tokens") or 448), True)
        thinking, content = split_thinking(raw if "<think>" in raw else f"<think></think>{raw}")
        return {"status": "ok", "mode": "audit", "audit_report": content or raw, "reasoning": thinking}

    problem = str(req.get("problem") or "")
    new_ev = str(req.get("new_evidence") or "").strip()
    max_toks = int(req.get("max_tokens") or 640)
    if new_ev:
        user_content = (
            f"{problem}\n\n"
            f"[NEW EVIDENCE / BELIEF REVISION UPDATE]\n{new_ev}\n\n"
            "Audit which Level-1 assumptions changed vs which verified facts remain invariant, then derive the updated solution."
        )
    else:
        user_content = problem

    msgs = req.get("messages") if isinstance(req.get("messages"), list) else [{"role": "user", "content": user_content}]
    out = run_live_rpr_pyramid_turn(msgs, generate_fn, max_tokens=max_toks, has_images=False)
    raw = out["text"]
    thinking, content = split_thinking(raw if "<think>" in raw else f"<think></think>{raw}")
    return {
        "status": "ok",
        "mode": "solve",
        "answer": content or raw,
        "reasoning": thinking,
        "passes": out["passes"],
        "repaired": out["repaired"],
        "audit_note": out["audit_note"],
    }
