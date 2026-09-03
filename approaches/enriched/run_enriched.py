#!/usr/bin/env python3
"""
Enriched approach, single gene: resolve each cell type's CL/UBERON IDs to
their real ontology labels (core.ontology_lookup, backed by a pinned local
cache — no network at run time) and hand the LLM correct labels instead of
bare IDs, then ask the same "what do these have in common" question as the
naive approach. v1 isolates one variable against naive: does the model
reason better given ground-truth cell-type/tissue names alone? v2 adds each
term's OBO definition and is_a hierarchy (PIPELINE_REQUIREMENTS.md §4).
The cell-type list is grouped by UBERON tissue (not one line per raw pair)
to stay within the model's context window on large gene sets.

Two directions, same as naive:
  - positive: cell types where the gene IS reliably expressed
  - negative: cell types where the gene is essentially NEVER expressed

Lives in its own MLflow experiment (CellCoLLM/enriched) — approaches/naive's
files are untouched, so naive v1/v2 remain runnable exactly as before.

Usage:
    python approaches/enriched/run_enriched.py --gene ENSG00000132763 --model qwen3:32b
    python approaches/enriched/run_enriched.py --gene ENSG00000132763 --set negative
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

import mlflow

from core.data_loader import GeneExpressionDataset, parse_pair
from core.llm_backend import DEFAULT_NUM_PREDICT, compute_num_ctx, make_chat_llm, resolve_chat_model
from core.mlflow_utils import (
    RunContext,
    get_or_create_gene_parent_run,
    log_json_artifact,
    log_text_artifact,
    tracked_run,
    verify_blinding,
)
from core.ontology_lookup import OntologyLookup
from core.structured_llm import call_llm_with_retry

APPROACH = "enriched"
PROMPTS_DIR = Path(__file__).parent / "prompts"
PROMPT_VERSIONS = {
    "v1": {
        "prompt_mode": "labels_only",
        "files": {"positive": "enriched_labels_only_positive_v1", "negative": "enriched_labels_only_negative_v1"},
    },
    "v2": {
        "prompt_mode": "glossary",
        "files": {"positive": "enriched_glossary_positive_v2", "negative": "enriched_glossary_negative_v2"},
    },
}
REQUIRED_RESPONSE_KEYS = {"property", "confidence", "rationale", "abstained"}
# Fraction of (num_ctx - num_predict) that counts as "saturated" — a prompt this close to
# the edge of its available budget either barely fits or (more likely, if it's a large
# gene set) was silently truncated by Ollama, which reports the truncated count either way.
CONTEXT_UTILIZATION_OVERFLOW_THRESHOLD = 0.98


def log_context_utilization(input_set: str, actual_prompt_tokens: int | None, num_ctx: int) -> None:
    """Ground-truth overflow check from the model's own reported prompt_eval_count —
    logged and printed for both successful and failed (parse-error) calls."""
    if actual_prompt_tokens is None:
        return
    context_budget = num_ctx - DEFAULT_NUM_PREDICT
    context_utilization = actual_prompt_tokens / context_budget
    context_overflow_confirmed = context_utilization >= CONTEXT_UTILIZATION_OVERFLOW_THRESHOLD
    mlflow.log_metric("context_utilization", context_utilization)
    mlflow.set_tag("context_overflow_confirmed", context_overflow_confirmed)
    if context_overflow_confirmed:
        print(f"[{input_set}] CONFIRMED OVERFLOW: actual prompt used {actual_prompt_tokens}/{context_budget} "
              f"of the available context ({context_utilization:.0%}) — this prompt was very likely truncated; "
              f"treat this response as unreliable.")


def build_grouped_list(cell_type_ids: list[str], lookup: OntologyLookup) -> tuple[str, int, list[str]]:
    """
    One line per unique UBERON tissue, listing every CL cell type observed
    there — replaces a flat one-line-per-pair list, which repeats each
    tissue's label/ID on every one of its (often many) cell-type lines. The
    dataset's UBERON universe (~54 IDs) is far smaller than CL's (~677+), so
    grouping by tissue is essentially always the better compression, for any
    gene — a fixed axis, not a per-run choice (keeps positive/negative
    formatting identical, PIPELINE_REQUIREMENTS.md §4.3.1).
    Returns (grouped_text, n_groups, unresolved_pairs).
    """
    groups: dict[str, list[str]] = {}
    unresolved_pairs = []
    for pair in cell_type_ids:
        cl_id, uberon_id = parse_pair(pair)
        groups.setdefault(uberon_id, []).append(cl_id)
        cl = lookup.resolve_cl(cl_id)
        uberon = lookup.resolve_uberon(uberon_id)
        if not (cl.resolved and uberon.resolved):
            unresolved_pairs.append(pair)

    lines = []
    for uberon_id in sorted(groups):
        uberon = lookup.resolve_uberon(uberon_id)
        uberon_text = uberon.label if uberon.resolved else f"(unresolved: {uberon_id})"
        cl_parts = []
        for cl_id in groups[uberon_id]:
            cl = lookup.resolve_cl(cl_id)
            cl_text = cl.label if cl.resolved else f"(unresolved: {cl_id})"
            cl_parts.append(f"{cl_text} (CL id: {cl_id})")
        lines.append(f"- {uberon_text} (UBERON id: {uberon_id}): " + ", ".join(cl_parts))

    return "\n".join(lines), len(groups), unresolved_pairs


_MARKER_TOKEN_RE = re.compile(r"\b[\w/]+-(?:positive|negative|high|low|bright|dim)\b", re.IGNORECASE)
_BRACKET_TAG_RE = re.compile(r"\s?\[[A-Z]{2,6}\]")
_EXAMPLES_RE = re.compile(r"^Examples?:", re.IGNORECASE)


def _strip_definition_noise(definition: str) -> tuple[str, int]:
    """
    Remove marker/phenotype code lists (flow-cytometry detail, not shared
    function/lineage/physiology), inline bracket cross-ref tags, and
    "Examples: ..." trailers from an OBO definition. Sentence-level removal,
    triggered by marker-token density (>=2 per sentence) rather than
    enumerating lead-in phrases, so a single incidental marker mention (e.g.
    "expresses the CD4 coreceptor") is correctly left alone. Returns
    (cleaned_text, chars_removed); falls back to the original text if
    stripping would leave nothing (e.g. every sentence was marker detail).
    """
    original_len = len(definition)
    text = _BRACKET_TAG_RE.sub("", definition)
    sentences = re.split(r"(?<=[.])\s+", text.strip())
    kept = [
        s for s in sentences
        if len(_MARKER_TOKEN_RE.findall(s)) < 2 and not _EXAMPLES_RE.match(s.strip())
    ]
    cleaned = " ".join(kept).strip()
    if not cleaned:
        return definition, 0
    return cleaned, original_len - len(cleaned)


def _glossary_block(
    ontology: str, term_id: str, lookup: OntologyLookup, include_definition: bool, include_hierarchy: bool,
    hierarchy_depth: int, strip_marker_noise: bool,
) -> tuple[str, bool, int]:
    """Format one glossary entry for a single unique CL or UBERON term; returns (block, unresolved, chars_stripped)."""
    term = lookup.resolve_cl(term_id) if ontology == "cl" else lookup.resolve_uberon(term_id)
    label = term.label if term.resolved else f"(unresolved: {term_id})"
    lines = [f"{term_id} - {label}"]
    chars_stripped = 0
    if include_definition:
        definition = term.definition if term.resolved and term.definition else "(no definition available)"
        if strip_marker_noise and term.resolved and term.definition:
            definition, chars_stripped = _strip_definition_noise(definition)
        lines.append(f"  Definition: {definition}")
    if include_hierarchy:
        ancestors = lookup.ancestors(ontology, term_id, hierarchy_depth) if term.resolved else []
        if ancestors:
            chain = " -> ".join(a.label if a.resolved else f"(unresolved: {a.id})" for a in ancestors)
            lines.append(f"  Broader categories: {chain}")
        else:
            lines.append(f"  Broader categories: (none found within depth {hierarchy_depth})")
    return "\n".join(lines), not term.resolved, chars_stripped


def build_glossary(
    cell_type_ids: list[str], lookup: OntologyLookup, include_definition: bool, include_hierarchy: bool,
    hierarchy_depth: int, strip_marker_noise: bool,
) -> tuple[str, dict]:
    """
    One glossary block per *unique* CL/UBERON term referenced in cell_type_ids
    (deduplicated — a set can repeat the same term across many rows), so the
    per-entry cell-type list below it can stay as compact as enriched v1's.
    """
    cl_ids: set[str] = set()
    uberon_ids: set[str] = set()
    for pair in cell_type_ids:
        cl_id, uberon_id = parse_pair(pair)
        cl_ids.add(cl_id)
        uberon_ids.add(uberon_id)

    n_unresolved = 0
    chars_stripped_total = 0
    cl_blocks = []
    for term_id in sorted(cl_ids):
        block, unresolved, chars_stripped = _glossary_block(
            "cl", term_id, lookup, include_definition, include_hierarchy, hierarchy_depth, strip_marker_noise,
        )
        cl_blocks.append(block)
        n_unresolved += unresolved
        chars_stripped_total += chars_stripped
    uberon_blocks = []
    for term_id in sorted(uberon_ids):
        block, unresolved, chars_stripped = _glossary_block(
            "uberon", term_id, lookup, include_definition, include_hierarchy, hierarchy_depth, strip_marker_noise,
        )
        uberon_blocks.append(block)
        n_unresolved += unresolved
        chars_stripped_total += chars_stripped

    glossary = (
        "Cell types (CL):\n" + "\n\n".join(cl_blocks) + "\n\nTissues (UBERON):\n" + "\n\n".join(uberon_blocks)
    )
    counts = {
        "n_unique_cl_terms": len(cl_ids),
        "n_unique_uberon_terms": len(uberon_ids),
        "n_glossary_unresolved": n_unresolved,
    }
    if strip_marker_noise:
        counts["definition_chars_stripped"] = chars_stripped_total
    return glossary, counts


def build_prompt(
    prompt_file_stem: str, cell_type_ids: list[str], lookup: OntologyLookup, prompt_mode: str,
    include_definition: bool = False, include_hierarchy: bool = False, hierarchy_depth: int = 0,
    strip_marker_noise: bool = False,
) -> tuple[str, list[str], int, dict, str | None]:
    template = (PROMPTS_DIR / f"{prompt_file_stem}.txt").read_text()
    grouped_list, n_groups, unresolved_pairs = build_grouped_list(cell_type_ids, lookup)

    glossary_counts: dict = {}
    glossary_text: str | None = None
    if prompt_mode == "glossary":
        glossary_text, glossary_counts = build_glossary(
            cell_type_ids, lookup, include_definition, include_hierarchy, hierarchy_depth, strip_marker_noise,
        )
        prompt = template.format(
            glossary=glossary_text, cell_type_list=grouped_list, n=len(cell_type_ids), n_tissues=n_groups,
        )
    else:
        prompt = template.format(cell_type_list=grouped_list, n=len(cell_type_ids), n_tissues=n_groups)
    return prompt, unresolved_pairs, n_groups, glossary_counts, glossary_text


def run_direction(
    ds: GeneExpressionDataset, lookup: OntologyLookup, gene_id: str, gene_symbol: str, species: str,
    model: str, temperature: float, seed: int, input_set: str, parent_run_id: str,
    prompt_version: str, include_definition: bool, include_hierarchy: bool, hierarchy_depth: int,
    strip_marker_noise: bool,
) -> None:
    cell_type_ids = (
        ds.positive_cell_types(gene_id) if input_set == "positive" else ds.negative_cell_types(gene_id)
    )
    version_cfg = PROMPT_VERSIONS[prompt_version]
    prompt_file_stem = version_cfg["files"][input_set]
    prompt_mode = version_cfg["prompt_mode"]
    # v1 has no glossary — these axes are structurally inert for it; log False/0 so
    # every enriched run (v1 or v2) is directly comparable in the MLflow UI.
    effective_include_definition = include_definition if prompt_mode == "glossary" else False
    effective_include_hierarchy = include_hierarchy if prompt_mode == "glossary" else False
    effective_hierarchy_depth = hierarchy_depth if prompt_mode == "glossary" else 0
    effective_strip_marker_noise = strip_marker_noise if prompt_mode == "glossary" else False

    ctx = RunContext(
        approach=APPROACH,
        approach_version=prompt_version,
        gene_id=gene_id,
        gene_symbol=gene_symbol,
        species=species,
        model=model,
        prompt_mode=prompt_mode,
        input_set=input_set,
        temperature=temperature,
        seed=seed,
        dataset_hash=ds.dataset_hash,
        prompt_version=prompt_file_stem,
        extra_params={
            "cl_data_version": lookup.provenance.get("cl_data_version"),
            "uberon_data_version": lookup.provenance.get("uberon_data_version"),
            "include_definition": effective_include_definition,
            "include_hierarchy": effective_include_hierarchy,
            "hierarchy_depth": effective_hierarchy_depth,
            "strip_marker_noise": effective_strip_marker_noise,
            "list_group_by": "uberon",
        },
    )

    with tracked_run(ctx, parent_run_id=parent_run_id) as run:
        print(f"\n[{input_set}] MLflow run: {run.info.run_id} (parent: {parent_run_id})")
        mlflow.log_metric("n_cell_types", len(cell_type_ids))
        log_json_artifact(cell_type_ids, f"{input_set}_cell_types.json")

        if not cell_type_ids:
            print(f"[{input_set}] Empty set for this gene — aborting before the LLM call.")
            mlflow.set_tag("status", "EARLY_EXIT_EMPTY_SET")
            return

        prompt, unresolved_pairs, n_groups, glossary_counts, glossary_text = build_prompt(
            prompt_file_stem, cell_type_ids, lookup, prompt_mode,
            effective_include_definition, effective_include_hierarchy, effective_hierarchy_depth,
            effective_strip_marker_noise,
        )
        mlflow.log_metric("n_unresolved", len(unresolved_pairs))
        mlflow.log_metric("n_list_groups", n_groups)
        log_json_artifact(unresolved_pairs, "unresolved_ids.json")
        if glossary_counts:
            mlflow.log_metrics(glossary_counts)
        if glossary_text is not None:
            log_text_artifact(glossary_text, "glossary.txt")

        blinding_ok, hits = verify_blinding(prompt, forbidden_terms=[gene_symbol] if gene_symbol != gene_id else [])
        mlflow.set_tag("blinding_verified", blinding_ok)
        if not blinding_ok:
            mlflow.set_tag("status", "FAILED_BLINDING")
            raise SystemExit(f"[{input_set}] Blinding check failed — prompt contains: {hits}")

        log_text_artifact(prompt, "prompt.txt")

        # Rough chars/4 estimate (no tokenizer dependency) — an early, pre-call heads-up
        # only; calibration against the model's real tokenizer showed this is unreliable
        # in both directions, so it no longer drives num_ctx sizing (see compute_num_ctx)
        # or counts as a confirmed overflow — that comes from the post-call ground-truth
        # check below, once actual_prompt_tokens (prompt_eval_count) is known.
        estimated_prompt_tokens = len(prompt) // 4
        num_ctx = compute_num_ctx(model)
        mlflow.log_metric("estimated_prompt_tokens", estimated_prompt_tokens)
        mlflow.log_param("num_ctx", num_ctx)
        estimated_overflow_risk = estimated_prompt_tokens + DEFAULT_NUM_PREDICT > num_ctx
        mlflow.set_tag("context_overflow_risk_estimated", estimated_overflow_risk)
        if estimated_overflow_risk:
            print(f"[{input_set}] Heads-up: estimated prompt size ~{estimated_prompt_tokens} tokens "
                  f"may exceed num_ctx={num_ctx} — this is an unreliable pre-call guess, "
                  f"the confirmed check runs after the LLM responds.")

        print(f"[{input_set}] Calling {model} ({prompt_version}, {len(unresolved_pairs)} unresolved of {len(cell_type_ids)})...")

        llm = make_chat_llm(model=model, temperature=temperature, seed=seed, format="json", num_ctx=num_ctx)
        try:
            parsed, raw_text, elapsed, retries, response_metadata = call_llm_with_retry(
                llm, prompt, REQUIRED_RESPONSE_KEYS,
            )
        except RuntimeError as exc:
            log_context_utilization(input_set, getattr(exc, "response_metadata", {}).get("prompt_eval_count"), num_ctx)
            log_text_artifact(str(exc), "parse_error.txt")
            mlflow.set_tag("status", "FAILED_PARSE")
            print(f"[{input_set}] {exc}")
            return

        actual_prompt_tokens = response_metadata.get("prompt_eval_count")
        if actual_prompt_tokens is not None:
            mlflow.log_metric("actual_prompt_tokens", actual_prompt_tokens)
        log_context_utilization(input_set, actual_prompt_tokens, num_ctx)

        log_text_artifact(raw_text, "response_raw.txt")
        log_json_artifact(parsed, "response_parsed.json")

        mlflow.log_metrics({
            "latency_s": elapsed,
            "parse_retries": retries,
            "confidence": parsed["confidence"],
        })
        mlflow.set_tags({
            "status": "COMPLETED",
            "abstained": bool(parsed["abstained"]),
        })
        print(f"--- [{input_set}] Response ({elapsed:.1f}s, {retries} retries) ---")
        print(json.dumps(parsed, indent=2))


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
    parser.add_argument("--prompt-version", choices=list(PROMPT_VERSIONS), default="v1")
    parser.add_argument("--include-definition", action=argparse.BooleanOptionalAction, default=True,
                         help="v2 only: include each glossary term's OBO definition")
    parser.add_argument("--include-hierarchy", action=argparse.BooleanOptionalAction, default=True,
                         help="v2 only: include each glossary term's is_a ancestor chain")
    parser.add_argument("--hierarchy-depth", type=int, default=2,
                         help="v2 only: is_a traversal depth for the ancestor chain (default 2)")
    parser.add_argument("--strip-marker-noise", action=argparse.BooleanOptionalAction, default=True,
                         help="v2 only: strip marker/phenotype code lists and cross-ref tags from definitions")
    args = parser.parse_args()

    if args.prompt_version != "v2" and (
        not args.include_definition or not args.include_hierarchy or args.hierarchy_depth != 2
        or not args.strip_marker_noise
    ):
        print(f"Note: --include-definition/--include-hierarchy/--hierarchy-depth/--strip-marker-noise are v2-only "
              f"and are ignored for --prompt-version {args.prompt_version}.")

    gene_id = args.gene
    gene_symbol = args.gene_symbol or gene_id
    model = resolve_chat_model(args.model)

    print("Loading dataset...")
    ds = GeneExpressionDataset.load(args.dataset) if args.dataset else GeneExpressionDataset.load()
    if not ds.has_gene(gene_id):
        raise SystemExit(f"Gene {gene_id!r} not present in dataset ({ds.path.name}). Aborting.")

    print("Loading ontology cache...")
    lookup = OntologyLookup()

    summary = ds.expression_summary(gene_id)
    print(f"Gene {gene_id} ({gene_symbol}): {summary}")

    parent_run_id = get_or_create_gene_parent_run(
        approach=APPROACH, gene_id=gene_id, gene_symbol=gene_symbol, species=args.species,
        dataset_hash=ds.dataset_hash, expression_summary=summary,
    )

    directions = ["positive", "negative"] if args.input_set == "both" else [args.input_set]
    for direction in directions:
        run_direction(
            ds, lookup, gene_id, gene_symbol, args.species, model, args.temperature, args.seed,
            input_set=direction, parent_run_id=parent_run_id, prompt_version=args.prompt_version,
            include_definition=args.include_definition, include_hierarchy=args.include_hierarchy,
            hierarchy_depth=args.hierarchy_depth, strip_marker_noise=args.strip_marker_noise,
        )


if __name__ == "__main__":
    main()
