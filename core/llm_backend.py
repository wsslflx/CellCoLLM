#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any

# Load .env from project root so subprocesses that import this module
# pick up OLLAMA_* settings without needing manual shell exports.
_env_path = Path(__file__).parents[1] / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

DEFAULT_BASE_URL = "https://dev.chat.cosy.bio/ollama"
# No default chat model — model selection is an open, to-be-decided question
# (see PIPELINE_REQUIREMENTS.md §5.6). Set CHAT_MODEL once chosen.
DEFAULT_NUM_CTX = 16000
DEFAULT_NUM_PREDICT = 4096

_model_max_context_cache: dict[str, int] = {}


def ollama_base_url() -> str:
    return os.getenv("BASE_URL", DEFAULT_BASE_URL).strip().rstrip("/")


def ollama_headers(require_api_key: bool = True) -> dict[str, str]:
    key = os.getenv("API_KEY", "").strip()
    if key:
        return {"Authorization": f"Bearer {key}"}
    if require_api_key:
        raise RuntimeError(
            "Missing API_KEY. Set API_KEY for authenticated access "
            "to the Ollama/Open WebUI backend."
        )
    return {}


def resolve_chat_model(model: str | None = None) -> str:
    if isinstance(model, str) and model.strip():
        return model.strip()
    env_model = os.getenv("CHAT_MODEL", "").strip()
    if env_model:
        return env_model
    raise RuntimeError(
        "No chat model resolved. Model selection is not yet fixed for this "
        "project (PIPELINE_REQUIREMENTS.md §5.6) — set CHAT_MODEL "
        "explicitly once one is chosen."
    )


def get_model_max_context(model: str) -> int:
    """
    Real max context length the server reports for `model` (e.g. `qwen3.context_length`
    in /api/show's model_info — key is namespaced per model family). Cached per model
    name. Never raises: falls back to DEFAULT_NUM_CTX and warns on any failure, so a
    lookup problem degrades to today's fixed behavior instead of breaking a run.
    """
    if model in _model_max_context_cache:
        return _model_max_context_cache[model]
    try:
        req = urllib.request.Request(
            f"{ollama_base_url()}/api/show",
            data=json.dumps({"model": model}).encode("utf-8"),
            headers={**ollama_headers(), "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            info = json.loads(resp.read()).get("model_info", {})
        max_context = next(v for k, v in info.items() if k.endswith(".context_length"))
        max_context = int(max_context)
    except Exception as exc:
        print(f"Warning: could not determine max context length for {model!r} ({exc}); "
              f"falling back to {DEFAULT_NUM_CTX}.")
        max_context = DEFAULT_NUM_CTX
    _model_max_context_cache[model] = max_context
    return max_context


def compute_num_ctx(
    model: str, estimated_prompt_tokens: int, num_predict: int = DEFAULT_NUM_PREDICT, margin: int = 512,
) -> tuple[int, bool]:
    """
    Size num_ctx to fit `estimated_prompt_tokens` + generation headroom, capped at the
    model's real max context. Never goes below DEFAULT_NUM_CTX (today's known-working
    default) — this only ever raises the window for larger prompts, never shrinks it.
    Returns (num_ctx, overflow_risk) — overflow_risk is True when even the model's own
    max isn't enough for the desired size.
    """
    model_max = get_model_max_context(model)
    desired = estimated_prompt_tokens + num_predict + margin
    num_ctx = min(model_max, max(desired, DEFAULT_NUM_CTX))
    return num_ctx, desired > model_max


def make_chat_llm(model: str | None, temperature: float, **kwargs: Any):
    from langchain_ollama import ChatOllama

    kwargs.setdefault("num_ctx", DEFAULT_NUM_CTX)
    kwargs.setdefault("num_predict", DEFAULT_NUM_PREDICT)
    resolved = resolve_chat_model(model)
    # qwen3.x burns its entire num_predict budget on hidden thinking tokens
    # unless reasoning is explicitly disabled (see PIPELINE_REQUIREMENTS.md §5.6).
    if "qwen3" in resolved.lower():
        kwargs.setdefault("reasoning", False)
    return ChatOllama(
        base_url=ollama_base_url(),
        model=resolved,
        temperature=temperature,
        client_kwargs={"headers": ollama_headers(), "timeout": 300, **kwargs.pop("client_kwargs", {})},
        **kwargs,
    )
