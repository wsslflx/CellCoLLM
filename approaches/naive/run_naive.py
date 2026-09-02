#!/usr/bin/env python3
"""
Naive approach, single gene: take one of the gene's cell-type sets (CL|UBERON
IDs only, no labels) from the binarised expression dataset, ask one LLM call
what they have in common, log everything to MLflow.

Two directions, same idea, different prompt:
  - positive: cell types where the gene IS reliably expressed
  - negative: cell types where the gene is essentially NEVER expressed

No matching, no contrasting, no scoring — this is deliberately the simplest
possible version, to (a) get a first real signal and (b) test whether the
model can reason from bare ontology IDs at all (PIPELINE_REQUIREMENTS.md
§5.6 open question). By default both directions run in one invocation and
are linked under one "gene parent" MLflow run so they're grouped/comparable.

The prompt (v2) forces structured JSON output — property/confidence/rationale/
abstained, plus a bounded "recognized_examples" audit of which IDs the model
claims to actually know, so a confident-but-wrong guess is auditable rather
than buried in prose (see approaches/naive/prompts/naive_id_only_*_v2.txt).

Usage:
    python approaches/naive/run_naive.py --gene ENSG00000001626 --gene-symbol CFTR
    python approaches/naive/run_naive.py --gene ENSG00000001626 --set negative
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

import mlflow

from core.data_loader import GeneExpressionDataset
from core.llm_backend import make_chat_llm, resolve_chat_model
from core.mlflow_utils import (
    RunContext,
    get_or_create_gene_parent_run,
    log_json_artifact,
    log_text_artifact,
    tracked_run,
    verify_blinding,
)

APPROACH = "naive"
PROMPTS_DIR = Path(__file__).parent / "prompts"
# v1: free-text response, no schema. v2: forced structured JSON (property/confidence/
# rationale/abstained + a bounded ID-recognition audit). Kept both so a supervisor demo
# can show the before/after of adding structure, not just the latest version.
PROMPT_VERSIONS = {
    "v1": {
        "structured": False,
        "files": {"positive": "naive_id_only_positive_v1", "negative": "naive_id_only_negative_v1"},
    },
    "v2": {
        "structured": True,
        "files": {"positive": "naive_id_only_positive_v2", "negative": "naive_id_only_negative_v2"},
    },
}
REQUIRED_RESPONSE_KEYS = {"n_ids_recognized", "recognized_examples", "property", "confidence", "rationale", "abstained"}
MAX_PARSE_ATTEMPTS = 3

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def build_prompt(prompt_file_stem: str, cl_ids: list[str]) -> str:
    template = (PROMPTS_DIR / f"{prompt_file_stem}.txt").read_text()
    cell_type_list = "\n".join(f"- {c}" for c in cl_ids)
    return template.format(cell_type_list=cell_type_list, n=len(cl_ids))


def parse_response(raw_text: str) -> dict:
    """Extract and validate the JSON object; raises ValueError if malformed or incomplete."""
    text = raw_text.strip()
    fence_match = _FENCE_RE.search(text)
    payload = fence_match.group(1).strip() if fence_match else text
    parsed = json.loads(payload)  # raises json.JSONDecodeError (a ValueError) on malformed JSON
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}")
    missing = REQUIRED_RESPONSE_KEYS - parsed.keys()
    if missing:
        raise ValueError(f"Response missing required keys: {sorted(missing)}")
    return parsed


def call_llm_with_retry(llm, prompt: str) -> tuple[dict, str, float, int]:
    """Call the LLM, retrying on malformed/incomplete JSON up to MAX_PARSE_ATTEMPTS (§5.5 retry policy)."""
    last_error = None
    for attempt in range(1, MAX_PARSE_ATTEMPTS + 1):
        t0 = time.time()
        response = llm.invoke([("user", prompt)])
        elapsed = time.time() - t0
        raw_text = response.content if hasattr(response, "content") else str(response)
        try:
            parsed = parse_response(raw_text)
            return parsed, raw_text, elapsed, attempt - 1
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            print(f"  parse attempt {attempt}/{MAX_PARSE_ATTEMPTS} failed: {exc}")
    raise RuntimeError(f"Failed to get parseable JSON after {MAX_PARSE_ATTEMPTS} attempts: {last_error}") from last_error


def run_direction(
    ds: GeneExpressionDataset, gene_id: str, gene_symbol: str, species: str,
    model: str, temperature: float, seed: int, input_set: str, parent_run_id: str,
    prompt_version: str,
) -> None:
    cell_type_ids = (
        ds.positive_cell_types(gene_id) if input_set == "positive" else ds.negative_cell_types(gene_id)
    )
    version_cfg = PROMPT_VERSIONS[prompt_version]
    prompt_file_stem = version_cfg["files"][input_set]
    structured = version_cfg["structured"]

    ctx = RunContext(
        approach=APPROACH,
        approach_version=prompt_version,
        gene_id=gene_id,
        gene_symbol=gene_symbol,
        species=species,
        model=model,
        prompt_mode="id_only",
        input_set=input_set,
        temperature=temperature,
        seed=seed,
        dataset_hash=ds.dataset_hash,
        prompt_version=prompt_file_stem,
    )

    with tracked_run(ctx, parent_run_id=parent_run_id) as run:
        print(f"\n[{input_set}] MLflow run: {run.info.run_id} (parent: {parent_run_id})")
        mlflow.log_metric("n_cell_types", len(cell_type_ids))
        log_json_artifact(cell_type_ids, f"{input_set}_cell_types.json")

        if not cell_type_ids:
            print(f"[{input_set}] Empty set for this gene — aborting before the LLM call.")
            mlflow.set_tag("status", "EARLY_EXIT_EMPTY_SET")
            return

        prompt = build_prompt(prompt_file_stem, cell_type_ids)
        blinding_ok, hits = verify_blinding(prompt, forbidden_terms=[gene_symbol] if gene_symbol != gene_id else [])
        mlflow.set_tag("blinding_verified", blinding_ok)
        if not blinding_ok:
            mlflow.set_tag("status", "FAILED_BLINDING")
            raise SystemExit(f"[{input_set}] Blinding check failed — prompt contains: {hits}")

        log_text_artifact(prompt, "prompt.txt")
        print(f"[{input_set}] Calling {model} ({prompt_version})...")

        if structured:
            llm = make_chat_llm(model=model, temperature=temperature, seed=seed, format="json")
            try:
                parsed, raw_text, elapsed, retries = call_llm_with_retry(llm, prompt)
            except RuntimeError as exc:
                log_text_artifact(str(exc), "parse_error.txt")
                mlflow.set_tag("status", "FAILED_PARSE")
                print(f"[{input_set}] {exc}")
                return

            log_text_artifact(raw_text, "response_raw.txt")
            log_json_artifact(parsed, "response_parsed.json")
            log_json_artifact(parsed.get("recognized_examples", []), "recognized_examples.json")

            mlflow.log_metrics({
                "latency_s": elapsed,
                "parse_retries": retries,
                "n_ids_recognized": parsed["n_ids_recognized"],
                "confidence": parsed["confidence"],
            })
            mlflow.set_tags({
                "status": "COMPLETED",
                "abstained": bool(parsed["abstained"]),
            })
            print(f"--- [{input_set}] Response ({elapsed:.1f}s, {retries} retries) ---")
            print(json.dumps(parsed, indent=2))
        else:
            llm = make_chat_llm(model=model, temperature=temperature, seed=seed)
            t0 = time.time()
            response = llm.invoke([("user", prompt)])
            elapsed = time.time() - t0
            response_text = response.content if hasattr(response, "content") else str(response)

            log_text_artifact(response_text, "response.txt")
            mlflow.log_metric("latency_s", elapsed)
            mlflow.set_tag("status", "COMPLETED")
            print(f"--- [{input_set}] Response ({elapsed:.1f}s) ---\n{response_text}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene", required=True, help="Canonical Ensembl gene ID, e.g. ENSG00000001626")
    parser.add_argument("--gene-symbol", default=None,
                         help="Human-readable symbol for MLflow tags/logging only — never sent to the LLM")
    parser.add_argument("--species", default="human")
    parser.add_argument("--model", default=None, help="Overrides CHAT_MODEL from .env")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default=None, help="Path to the binarised expression tsv (default: repo root)")
    parser.add_argument("--set", dest="input_set", choices=["positive", "negative", "both"], default="both")
    parser.add_argument("--prompt-version", choices=list(PROMPT_VERSIONS), default="v2")
    args = parser.parse_args()

    gene_id = args.gene
    gene_symbol = args.gene_symbol or gene_id
    model = resolve_chat_model(args.model)

    print("Loading dataset...")
    ds = GeneExpressionDataset.load(args.dataset) if args.dataset else GeneExpressionDataset.load()
    if not ds.has_gene(gene_id):
        raise SystemExit(f"Gene {gene_id!r} not present in dataset ({ds.path.name}). Aborting.")

    summary = ds.expression_summary(gene_id)
    print(f"Gene {gene_id} ({gene_symbol}): {summary}")

    parent_run_id = get_or_create_gene_parent_run(
        approach=APPROACH, gene_id=gene_id, gene_symbol=gene_symbol, species=args.species,
        dataset_hash=ds.dataset_hash, expression_summary=summary,
    )

    directions = ["positive", "negative"] if args.input_set == "both" else [args.input_set]
    for direction in directions:
        run_direction(
            ds, gene_id, gene_symbol, args.species, model, args.temperature, args.seed,
            input_set=direction, parent_run_id=parent_run_id, prompt_version=args.prompt_version,
        )


if __name__ == "__main__":
    main()
