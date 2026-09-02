#!/usr/bin/env python3
"""
Shared structured-JSON-response parsing and retry logic for approaches that
force LLM output into a JSON schema (§5.5 PIPELINE_REQUIREMENTS.md: "structured
output enforcement... with a parse-failure retry policy"). Parameterized by
the caller's required keys so different approaches can share this without
sharing a schema.

approaches/naive/run_naive.py has its own local copy of this logic (kept
independent deliberately, so this module's introduction carries zero risk to
naive's already-verified behavior). New approaches should use this module.
"""
from __future__ import annotations

import json
import re
import time

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_structured_response(raw_text: str, required_keys: set[str]) -> dict:
    """Extract and validate the JSON object; raises ValueError if malformed or incomplete."""
    text = raw_text.strip()
    fence_match = _FENCE_RE.search(text)
    payload = fence_match.group(1).strip() if fence_match else text
    parsed = json.loads(payload)  # raises json.JSONDecodeError (a ValueError) on malformed JSON
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}")
    missing = required_keys - parsed.keys()
    if missing:
        raise ValueError(f"Response missing required keys: {sorted(missing)}")
    return parsed


def call_llm_with_retry(
    llm, prompt: str, required_keys: set[str], max_attempts: int = 3,
) -> tuple[dict, str, float, int, dict]:
    """Call the LLM, retrying on malformed/incomplete JSON up to max_attempts."""
    last_error = None
    for attempt in range(1, max_attempts + 1):
        t0 = time.time()
        response = llm.invoke([("user", prompt)])
        elapsed = time.time() - t0
        raw_text = response.content if hasattr(response, "content") else str(response)
        response_metadata = getattr(response, "response_metadata", {}) or {}
        try:
            parsed = parse_structured_response(raw_text, required_keys)
            return parsed, raw_text, elapsed, attempt - 1, response_metadata
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            print(f"  parse attempt {attempt}/{max_attempts} failed: {exc}")
    raise RuntimeError(f"Failed to get parseable JSON after {max_attempts} attempts: {last_error}") from last_error
