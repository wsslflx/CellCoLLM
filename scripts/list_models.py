#!/usr/bin/env python3
"""
List models available on the configured Ollama/Open WebUI backend, to help
pick a chat model for the naive pipeline that trades off performance vs. speed
(PIPELINE_REQUIREMENTS.md §5.6 — model choice is intentionally not fixed).

Shows, per model:
  - size on disk (proxy for download/load cost)
  - parameter count (rough proxy for capability)
  - quantization (affects both speed and quality)
  - whether it is currently loaded in VRAM (/api/ps), i.e. warm and fast to hit
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import httpx

from core.llm_backend import ollama_base_url, ollama_headers


def main() -> None:
    base = ollama_base_url()
    headers = ollama_headers(require_api_key=False)

    r_tags = httpx.get(f"{base}/api/tags", headers=headers, timeout=15)
    r_tags.raise_for_status()
    all_models = r_tags.json().get("models", [])

    loaded_names: set[str] = set()
    r_ps = httpx.get(f"{base}/api/ps", headers=headers, timeout=10)
    if r_ps.status_code == 200:
        loaded_names = {m.get("name", "") for m in r_ps.json().get("models", [])}
    else:
        print(f"(/api/ps not accessible: {r_ps.status_code} — VRAM/warm status unknown)\n")

    if not all_models:
        print("No models reported by /api/tags.")
        return

    rows = []
    for m in all_models:
        name = m.get("name", "?")
        size_gb = m.get("size", 0) / 1e9
        det = m.get("details", {})
        rows.append({
            "name": name,
            "size_gb": size_gb,
            "family": det.get("family", "?"),
            "params": det.get("parameter_size", "?"),
            "quant": det.get("quantization_level", "?"),
            "warm": "yes" if name in loaded_names else "",
        })

    rows.sort(key=lambda r: r["size_gb"])

    print(f"{'Model':<35} {'Size':>8}  {'Params':>8}  {'Quant':>8}  {'Family':<12}  Warm")
    print("-" * 90)
    for r in rows:
        print(f"{r['name']:<35} {r['size_gb']:>6.1f}G  {r['params']:>8}  {r['quant']:>8}  {r['family']:<12}  {r['warm']}")

    print(f"\n{len(rows)} models available at {base}")
    print("Smaller size / fewer params -> faster but weaker; 'warm' models avoid a cold-load delay on first call.")


if __name__ == "__main__":
    main()
