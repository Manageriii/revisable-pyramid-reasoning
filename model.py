"""Local MLX model wrapper and response parsing helpers for RPR."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import time
from typing import Any, Callable, Protocol


DEFAULT_MODEL_PATH = (
    "./models/LFM3-VL-2B-MLX-8bit"
    if Path("./models/LFM3-VL-2B-MLX-8bit").exists()
    else "LiquidAI/LFM3-VL-2B-MLX-8bit"
)

_THINK_CLOSE_RE = re.compile(r"</think>\s*", re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"^<think>\s*", re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_TOOL_CALL_RE = re.compile(r"<tool_call>.*?(?:</tool_call>|$)", re.DOTALL | re.IGNORECASE)


def strip_tool_calls(text: str) -> str:
    """Remove any `<tool_call>...</tool_call>` XML blocks emitted by chat-tuned models."""
    return _TOOL_CALL_RE.sub("", text).strip()


@dataclass
class GenerationStats:
    """Per-call token, latency, and memory telemetry."""

    prompt_tokens: int
    generation_tokens: int
    total_tokens: int
    wall_clock_sec: float
    peak_memory_gb: float = 0.0
    finish_reason: str = "stop"
    truncated_by_budget: bool = False


@dataclass
class ModelResponse:
    """Normalized generation result returned by any ModelClient."""

    raw_text: str
    thinking_text: str
    content_text: str
    stats: GenerationStats


class ModelClient(Protocol):
    """Protocol shared by real MLX models and deterministic test doubles."""

    model_name: str

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        enable_thinking: bool = True,
    ) -> ModelResponse:
        """Generate a completion for chat messages."""
        ...


def split_thinking(text: str) -> tuple[str, str]:
    """Separate `<think>...</think>` scratchpad content from the visible reply."""
    stripped = text.strip()
    if "</think>" in stripped.lower():
        parts = _THINK_CLOSE_RE.split(stripped, maxsplit=1)
        thinking = _THINK_OPEN_RE.sub("", parts[0]).strip()
        content = strip_tool_calls(parts[1]) if len(parts) > 1 else ""
        return thinking, content
    if stripped.lower().startswith("<think>"):
        return _THINK_OPEN_RE.sub("", stripped).strip(), ""
    return "", strip_tool_calls(stripped)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract the best matching JSON object from model output text."""
    _, content = split_thinking(text)
    candidates = [content, text] if content else [text]

    for candidate in candidates:
        if not candidate:
            continue
        fenced = _CODE_FENCE_RE.findall(candidate)
        search_pool = fenced + [candidate]
        for chunk in search_pool:
            chunk = chunk.strip()
            if not chunk:
                continue
            for candidate_chunk in (chunk, re.sub(r"\}\s*\}(?=\s*\])", "}", chunk)):
                try:
                    parsed = json.loads(candidate_chunk)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    pass

            for obj in _scan_balanced_json_dicts(chunk):
                return obj
    return None


def _scan_balanced_json_dicts(text: str) -> list[dict[str, Any]]:
    """Scan a string for balanced `{...}` JSON objects, preferring the last valid dict."""
    found: list[dict[str, Any]] = []
    idx = 0
    length = len(text)
    while idx < length:
        if text[idx] != "{":
            idx += 1
            continue
        depth = 0
        in_str = False
        escaped = False
        for end in range(idx, length):
            ch = text[end]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    snippet = text[idx : end + 1]
                    try:
                        parsed = json.loads(snippet)
                        if isinstance(parsed, dict):
                            found.append(parsed)
                    except json.JSONDecodeError:
                        pass
                    idx = end
                    break
        idx += 1
    found.reverse()
    return found


def _is_vlm_model_dir(model_path: str) -> bool:
    """Detect whether a local model directory uses a multimodal MLX-VLM architecture."""
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.exists():
        return "vl" in model_path.lower()
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    model_type = str(cfg.get("model_type", "")).lower()
    return "vision_config" in cfg or "_vl" in model_type or model_type == "qwen3_5"


class MLXModelClient:
    """Local Apple Silicon MLX inference client supporting both mlx-vlm and mlx-lm."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        *,
        adapter_path: str | None = None,
        repetition_penalty: float = 1.03,
        default_enable_thinking: bool = True,
    ) -> None:
        self.model_name = f"{model_path} + adapter:{Path(adapter_path).name}" if adapter_path else model_path
        self.adapter_path = adapter_path
        self.repetition_penalty = repetition_penalty
        self.default_enable_thinking = default_enable_thinking
        self._backend: str
        self._model: Any
        self._processor: Any

        if _is_vlm_model_dir(model_path):
            self._load_vlm(model_path, adapter_path=adapter_path)
        else:
            try:
                self._load_lm(model_path, adapter_path=adapter_path)
            except ValueError:
                self._load_vlm(model_path, adapter_path=adapter_path)

    def _load_vlm(self, model_path: str, *, adapter_path: str | None = None) -> None:
        import mlx_vlm

        self._model, self._processor = mlx_vlm.load(model_path)
        if (Path(model_path) / "v8_arch_ext.safetensors").exists():
            from v8_arch import attach_v8_extensions

            attach_v8_extensions(self._model, sidecar_dir=model_path, enable_moba=False)
        self._model.eval()
        if adapter_path:
            from mlx_vlm.trainer.peft import load_adapters

            self._model = load_adapters(self._model, adapter_path)
        self._backend = "mlx_vlm"

    def _load_lm(self, model_path: str, *, adapter_path: str | None = None) -> None:
        import mlx_lm

        if adapter_path:
            self._model, self._processor = mlx_lm.load(model_path, adapter_path=adapter_path)
        else:
            self._model, self._processor = mlx_lm.load(model_path)
        self._backend = "mlx_lm"

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        enable_thinking: bool | None = None,
    ) -> ModelResponse:
        use_thinking = (
            self.default_enable_thinking if enable_thinking is None else enable_thinking
        )
        if self._backend == "mlx_vlm":
            return self._generate_vlm(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                enable_thinking=use_thinking,
            )
        return self._generate_lm(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_thinking=use_thinking,
        )

    def _generate_vlm(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        enable_thinking: bool,
    ) -> ModelResponse:
        import mlx_vlm
        from mlx_vlm.prompt_utils import apply_chat_template

        prompt = apply_chat_template(
            self._processor,
            self._model.config,
            messages,
            num_images=0,
            enable_thinking=enable_thinking,
        )
        t0 = time.perf_counter()
        result = mlx_vlm.generate(
            self._model,
            self._processor,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            repetition_penalty=self.repetition_penalty,
            verbose=False,
        )
        dt = time.perf_counter() - t0
        raw_text = result.text or ""
        if enable_thinking and prompt.rstrip().endswith("<think>") and not raw_text.lstrip().lower().startswith("<think>"):
            raw_text = "<think>" + raw_text
        thinking, content = split_thinking(raw_text)
        prompt_tokens = int(getattr(result, "prompt_tokens", 0) or 0)
        generation_tokens = int(getattr(result, "generation_tokens", 0) or 0)
        total_tokens = int(
            getattr(result, "total_tokens", prompt_tokens + generation_tokens)
            or (prompt_tokens + generation_tokens)
        )
        peak_mem = float(getattr(result, "peak_memory", 0.0) or 0.0)
        finish_reason = str(getattr(result, "finish_reason", "stop") or "stop")
        return ModelResponse(
            raw_text=raw_text,
            thinking_text=thinking,
            content_text=content,
            stats=GenerationStats(
                prompt_tokens=prompt_tokens,
                generation_tokens=generation_tokens,
                total_tokens=total_tokens,
                wall_clock_sec=dt,
                peak_memory_gb=peak_mem,
                finish_reason=finish_reason,
            ),
        )

    def _generate_lm(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        enable_thinking: bool,
    ) -> ModelResponse:
        import mlx_lm
        from mlx_lm.sample_utils import make_sampler

        try:
            prompt = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            prompt = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        sampler = make_sampler(temp=temperature)
        t0 = time.perf_counter()
        chunks: list[str] = []
        last_resp = None
        for resp in mlx_lm.stream_generate(
            self._model,
            self._processor,
            prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        ):
            chunks.append(resp.text)
            last_resp = resp
        dt = time.perf_counter() - t0
        raw_text = "".join(chunks)
        thinking, content = split_thinking(raw_text)
        prompt_tokens = int(getattr(last_resp, "prompt_tokens", 0) or 0)
        generation_tokens = int(getattr(last_resp, "generation_tokens", 0) or 0)
        peak_mem = float(getattr(last_resp, "peak_memory", 0.0) or 0.0)
        finish_reason = str(getattr(last_resp, "finish_reason", "stop") or "stop")
        return ModelResponse(
            raw_text=raw_text,
            thinking_text=thinking,
            content_text=content,
            stats=GenerationStats(
                prompt_tokens=prompt_tokens,
                generation_tokens=generation_tokens,
                total_tokens=prompt_tokens + generation_tokens,
                wall_clock_sec=dt,
                peak_memory_gb=peak_mem,
                finish_reason=finish_reason,
            ),
        )


@dataclass
class MockModelClient:
    """Deterministic scripted model double for unit testing without GPU/weights."""

    responder: Callable[[list[dict[str, str]]], str]
    model_name: str = "mock-model"
    calls: list[list[dict[str, str]]] = field(default_factory=list)

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        enable_thinking: bool = False,
    ) -> ModelResponse:
        self.calls.append(messages)
        raw_text = self.responder(messages)
        thinking, content = split_thinking(raw_text)
        prompt_tokens = sum(max(1, len(m["content"].split())) for m in messages)
        gen_tokens = min(max_tokens, max(1, len(raw_text.split())))
        return ModelResponse(
            raw_text=raw_text,
            thinking_text=thinking,
            content_text=content,
            stats=GenerationStats(
                prompt_tokens=prompt_tokens,
                generation_tokens=gen_tokens,
                total_tokens=prompt_tokens + gen_tokens,
                wall_clock_sec=0.001,
                peak_memory_gb=0.0,
                finish_reason="stop",
            ),
        )


def needs_close_think_recovery(resp: ModelResponse, *, enable_thinking: bool) -> bool:
    """Return True when a thinking call hit length before emitting `</think>`."""
    return (
        enable_thinking
        and resp.stats.finish_reason == "length"
        and "</think>" not in resp.raw_text.lower()
    )


def build_recovery_messages(
    messages: list[dict[str, str]],
    thinking_body: str,
    instruction: str,
) -> list[dict[str, str]]:
    """Append truncated scratchpad reasoning into the trailing user turn.

    Avoids intermediate `<think>` tags in assistant history, which chat templates
    (such as LFM3-VL's Jinja template with `preserve_thinking=false`) strip out.
    """
    base_msgs = [dict(m) for m in messages]
    scratchpad_block = (
        f"--- Scratchpad reasoning completed so far ---\n{thinking_body}\n"
        f"--- End scratchpad ---\n\n{instruction}"
    )
    if base_msgs and base_msgs[-1].get("role") == "user":
        base_msgs[-1]["content"] = f"{base_msgs[-1]['content']}\n\n{scratchpad_block}"
        return base_msgs
    return [*base_msgs, {"role": "user", "content": scratchpad_block}]


def recover_close_think_json(
    model: ModelClient,
    messages: list[dict[str, str]],
    truncated_resp: ModelResponse,
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
    instruction: str = "Output ONLY a compact valid JSON object summarizing your conclusion above (use evaluated numbers, not arithmetic expressions).",
) -> tuple[ModelResponse, ModelResponse]:
    """Request only the final JSON object from a truncated `<think>` scratchpad.

    Returns `(merged_response, recovery_call_response)` so callers can record the
    recovery call's exact token/latency telemetry while passing merged output downstream.
    """
    thinking_body = truncated_resp.thinking_text or truncated_resp.raw_text
    recovery_messages = build_recovery_messages(messages, thinking_body, instruction)
    rec_resp = model.generate(
        recovery_messages,
        max_tokens=max_tokens,
        temperature=temperature,
        enable_thinking=False,
    )
    final_content = (rec_resp.content_text or rec_resp.raw_text).strip()
    merged = ModelResponse(
        raw_text=f"<think>{thinking_body}</think>\n{final_content}",
        thinking_text=thinking_body,
        content_text=final_content,
        stats=GenerationStats(
            prompt_tokens=truncated_resp.stats.prompt_tokens + rec_resp.stats.prompt_tokens,
            generation_tokens=(
                truncated_resp.stats.generation_tokens + rec_resp.stats.generation_tokens
            ),
            total_tokens=truncated_resp.stats.total_tokens + rec_resp.stats.total_tokens,
            wall_clock_sec=(
                truncated_resp.stats.wall_clock_sec + rec_resp.stats.wall_clock_sec
            ),
            peak_memory_gb=max(
                truncated_resp.stats.peak_memory_gb, rec_resp.stats.peak_memory_gb
            ),
            finish_reason=rec_resp.stats.finish_reason,
        ),
    )
    return merged, rec_resp
