#!/usr/bin/env python3
"""
Statistical approach, single gene: compute the positive-vs-negative contrast
with a Fisher's exact test over ontology features, then ask the LLM only to
NAME the biology behind the features that actually differ.

Why this exists: naive v1/v2/v3 and enriched v1/v2 all scored at chance on the
discrimination validation layer and converged on the same generic answer
("immune cells") regardless of gene — because each call sees ONE set in
isolation and reports its most salient feature, and this atlas is ~45% immune
on both sides. PIPELINE_REQUIREMENTS.md §5.2 predicted exactly that failure and
designates the split architecture those approaches use as the control arm, not
the default. Joint contrast is the specified fix but measures 68-72k tokens,
over qwen3:32b's window for every gene.

So: the statistics find WHICH features differ (and in which direction), and the
LLM reasons about WHY — over ~20 features with full definitions instead of
~2000 raw cell types. Measured at ~1k tokens vs ~39k for enriched v2.

Unlike every other approach this produces ONE run per gene, not a positive and
a negative direction — the contrast is computed internally, so there are no
separate directions to run. Runs are tagged input_set="contrast".

Usage:
    python approaches/statistical/run_statistical.py --gene ENSG00000149554 --model qwen3:32b
    python approaches/statistical/run_statistical.py --gene ENSG00000149554 --no-include-tissue
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

import mlflow

import re

from core.data_loader import GeneExpressionDataset
from core.enrichment import (
    DEFAULT_UBERON_RELATIONS,
    KIND_LINEAGE,
    KIND_PROCESS,
    KIND_TISSUE,
    compute_enrichment,
    compute_enrichment_corrected,
    pair_features,
    tissue_homogeneity,
)
from core.llm_backend import compute_num_ctx, make_chat_llm, resolve_chat_model
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

APPROACH = "statistical"
PROMPTS_DIR = Path(__file__).parent / "prompts"
BASELINES_PATH = Path(__file__).parents[2] / "data" / "feature_baselines.json"
PROMPT_VERSIONS = {
    # v1: Fisher's exact against a global null; lineage + tissue features only.
    "v1": {"prompt_mode": "enrichment_table", "file": "statistical_contrast_v1", "corrected": False},
    # v2: binomial against a two-factor baseline-corrected null; adds GO process
    # features and UBERON is_a/part_of rollup.
    "v2": {"prompt_mode": "enrichment_corrected", "file": "statistical_contrast_v2", "corrected": True},
}
REQUIRED_RESPONSE_KEYS = {"property", "confidence", "rationale", "abstained"}
# Above this fraction of all-or-nothing tissues, the enrichment is reporting study
# structure rather than gene biology and the run should not be read as a result.
BATCH_ARTIFACT_THRESHOLD = 0.2
# Above this token overlap with the feature labels shown, the "property" is a
# restatement of its own input rather than an inference (v1's observed failure).
RESTATEMENT_THRESHOLD = 0.6
KIND_HEADINGS = {
    KIND_PROCESS: "biological processes (what these cells do)",
    KIND_LINEAGE: "cell lineages (what these cells are)",
    KIND_TISSUE: "tissues (where these cells are)",
}
_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP = {"cell", "cells", "of", "the", "and", "in", "a", "an", "type", "types", "positive", "negative"}


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall((text or "").lower()) if w not in _STOP and len(w) > 2}


def property_overlap(property_text: str, labels: list[str]) -> float:
    """
    Fraction of the property's content words that also appear in the given labels.
    High overlap means the model restated its input rather than inferring beyond it
    — the concern that a statistics-first design reduces the LLM to a formatter.
    Deterministic, no model involved.
    """
    prop = _tokens(property_text)
    if not prop:
        return 0.0
    pool = set().union(*(_tokens(l) for l in labels)) if labels else set()
    return len(prop & pool) / len(prop)


def _resolve_any(lookup: OntologyLookup, kind: str, term_id: str):
    if kind == KIND_PROCESS:
        return lookup.resolve_go(term_id)
    if kind == KIND_TISSUE:
        return lookup.resolve_uberon(term_id)
    return lookup.resolve_cl(term_id)


def render_features(
    results: list, lookup: OntologyLookup, heading: str, include_definition: bool, corrected: bool,
) -> str:
    """v1 renders a flat list with odds ratios; v2 groups by feature kind and reports
    observed-vs-expected so the model can see specificity rather than raw frequency."""
    if not results:
        return f"{heading}\n  (none reached significance)"
    if not corrected:
        lines = [heading]
        for r in results:
            lines.append(f"  {r.label}  [{r.n_pos} expressing vs {r.n_neg} non-expressing, "
                         f"odds ratio {r.odds_ratio:.1f}, q={r.q_value:.0e}]")
            if include_definition:
                term = _resolve_any(lookup, r.kind, r.term_id)
                lines.append(f"      {(term.definition or '(no definition available)').strip()}")
        return "\n".join(lines)

    lines = []
    for kind in (KIND_PROCESS, KIND_LINEAGE, KIND_TISSUE):
        group = [r for r in results if r.kind == kind]
        if not group:
            continue
        lines.append(f"{heading} — {KIND_HEADINGS[kind]}:")
        for r in group:
            observed = r.n_pos / (r.n_pos + r.n_neg)
            lines.append(f"  {r.label}  [observed {observed:.0%} vs expected {r.expected_rate:.0%}, "
                         f"excess {r.excess:+.0%}, q={r.q_value:.0e}, n={r.n_pos + r.n_neg}]")
            if include_definition:
                term = _resolve_any(lookup, r.kind, r.term_id)
                lines.append(f"      {(term.definition or '(no definition available)').strip()}")
        lines.append("")
    return "\n".join(lines).rstrip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gene", required=True, help="Canonical Ensembl gene ID, e.g. ENSG00000149554")
    parser.add_argument("--gene-symbol", default=None,
                         help="Human-readable symbol for MLflow tags/logging only — never sent to the LLM")
    parser.add_argument("--species", default="human")
    parser.add_argument("--model", default=None, help="Overrides CHAT_MODEL from .env")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default=None, help="Path to the binarised expression tsv (default: repo root)")
    parser.add_argument("--prompt-version", choices=list(PROMPT_VERSIONS), default="v1")
    parser.add_argument("--hierarchy-depth", type=int, default=3,
                         help="How many is_a levels each cell type is rolled up to (default 3)")
    parser.add_argument("--min-feature-count", type=int, default=20,
                         help="Skip features observed fewer than this many times in total (underpowered)")
    parser.add_argument("--q-threshold", type=float, default=0.05, help="Benjamini-Hochberg cutoff")
    parser.add_argument("--top-n-enriched", type=int, default=8)
    parser.add_argument("--top-n-depleted", type=int, default=5)
    parser.add_argument("--include-tissue", action=argparse.BooleanOptionalAction, default=True,
                         help="Include UBERON tissue features — these carry most of the batch artifact, "
                              "so --no-include-tissue is the ablation for it")
    parser.add_argument("--include-definition", action=argparse.BooleanOptionalAction, default=True,
                         help="Show each feature's OBO definition so the LLM has biology to reason over")
    parser.add_argument("--go-depth", type=int, default=3,
                         help="v2: is_a depth over which capable_of GO processes are inherited (default 3). "
                              "Deeper = more coverage but generic processes spread across many cell types")
    parser.add_argument("--uberon-depth", type=int, default=3,
                         help="v2: depth for UBERON is_a/part_of tissue rollup (default 3)")
    parser.add_argument("--include-process", action=argparse.BooleanOptionalAction, default=True,
                         help="v2: include GO biological-process features. These are the only features that "
                              "yield properties rather than categories, but are also the most annotation-circular (§L3)")
    parser.add_argument("--min-excess", type=float, default=0.10,
                         help="v2: minimum |observed - expected| for a feature to be shown. With large n, "
                              "trivially small excesses reach significance; this is the effect-size floor")
    args = parser.parse_args()

    gene_id = args.gene
    gene_symbol = args.gene_symbol or gene_id
    model = resolve_chat_model(args.model)
    version_cfg = PROMPT_VERSIONS[args.prompt_version]
    corrected = version_cfg["corrected"]

    baselines, grand_mean, baselines_key = {}, 0.0, None
    if corrected:
        if not BASELINES_PATH.exists():
            raise SystemExit(
                f"{BASELINES_PATH} not found — v2 tests against per-feature baselines. "
                f"Run: python scripts/build_feature_baselines.py"
            )
        payload = json.loads(BASELINES_PATH.read_text())
        prov = payload["provenance"]
        cfg = prov["feature_config"]
        # Changing any feature-extraction parameter changes which features exist, so
        # stale baselines would be silently mismatched. Refuse rather than guess.
        mismatch = [
            f"{k}: baselines built with {cfg.get(k)!r}, this run uses {v!r}"
            for k, v in (("hierarchy_depth", args.hierarchy_depth), ("go_depth", args.go_depth),
                         ("uberon_depth", args.uberon_depth), ("include_tissue", args.include_tissue),
                         ("include_process", args.include_process))
            if cfg.get(k) != v
        ]
        if mismatch:
            raise SystemExit(
                "Feature configuration does not match the cached baselines:\n  " + "\n  ".join(mismatch)
                + f"\nEither use the cached settings or rebuild: python scripts/build_feature_baselines.py"
            )
        baselines = {k: v["baseline_rate"] for k, v in payload["baselines"].items()}
        grand_mean = prov["grand_mean_rate"]
        baselines_key = prov["baselines_key"]
        print(f"Loaded {len(baselines)} feature baselines (grand mean {grand_mean:.1%}, key {baselines_key})")

    print("Loading dataset...")
    ds = GeneExpressionDataset.load(args.dataset) if args.dataset else GeneExpressionDataset.load()
    if not ds.has_gene(gene_id):
        raise SystemExit(f"Gene {gene_id!r} not present in dataset ({ds.path.name}). Aborting.")

    print("Loading ontology cache...")
    lookup = OntologyLookup()
    summary = ds.expression_summary(gene_id)
    print(f"Gene {gene_id} ({gene_symbol}): {summary}")

    pos, neg = ds.positive_cell_types(gene_id), ds.negative_cell_types(gene_id)
    parent_run_id = get_or_create_gene_parent_run(
        approach=APPROACH, gene_id=gene_id, gene_symbol=gene_symbol, species=args.species,
        dataset_hash=ds.dataset_hash, expression_summary=summary,
    )

    ctx = RunContext(
        approach=APPROACH,
        approach_version=args.prompt_version,
        gene_id=gene_id,
        gene_symbol=gene_symbol,
        species=args.species,
        model=model,
        prompt_mode=version_cfg["prompt_mode"],
        input_set="contrast",  # one run per gene — the contrast is computed, not split
        temperature=args.temperature,
        seed=args.seed,
        dataset_hash=ds.dataset_hash,
        prompt_version=version_cfg["file"],
        extra_params={
            "cl_data_version": lookup.provenance.get("cl_data_version"),
            "uberon_data_version": lookup.provenance.get("uberon_data_version"),
            "hierarchy_depth": args.hierarchy_depth,
            "min_feature_count": args.min_feature_count,
            "q_threshold": args.q_threshold,
            "top_n_enriched": args.top_n_enriched,
            "top_n_depleted": args.top_n_depleted,
            "include_tissue": args.include_tissue,
            "include_definition": args.include_definition,
            **({
                "go_depth": args.go_depth,
                "uberon_depth": args.uberon_depth,
                "uberon_relations": ",".join(DEFAULT_UBERON_RELATIONS),
                "include_process": args.include_process,
                "min_excess": args.min_excess,
                "baselines_key": baselines_key,
                "grand_mean_rate": round(grand_mean, 6),
                "gene_rate": round(len(pos) / (len(pos) + len(neg)), 6) if (pos or neg) else 0.0,
            } if corrected else {}),
        },
    )

    with tracked_run(ctx, parent_run_id=parent_run_id) as run:
        print(f"\nMLflow run: {run.info.run_id} (parent: {parent_run_id})")
        mlflow.log_metrics({"n_positive_pairs": len(pos), "n_negative_pairs": len(neg)})

        if not pos or not neg:
            print("One of the sets is empty — no contrast to compute. Aborting before the LLM call.")
            mlflow.set_tag("status", "EARLY_EXIT_EMPTY_SET")
            return

        n_skipped = 0
        if corrected:
            print("Computing baseline-corrected enrichment (binomial vs two-factor null + BH)...")
            results, n_skipped = compute_enrichment_corrected(
                pos, neg, lookup, baselines=baselines, grand_mean=grand_mean,
                min_feature_count=args.min_feature_count, hierarchy_depth=args.hierarchy_depth,
                include_tissue=args.include_tissue, include_process=args.include_process,
                go_depth=args.go_depth, uberon_depth=args.uberon_depth,
                uberon_relations=DEFAULT_UBERON_RELATIONS,
            )
        else:
            print("Computing enrichment (Fisher's exact + BH)...")
            results = compute_enrichment(
                pos, neg, lookup, hierarchy_depth=args.hierarchy_depth,
                min_feature_count=args.min_feature_count, include_tissue=args.include_tissue,
            )
        log_json_artifact([r.as_dict() for r in results], "enrichment_full.json")

        significant = [r for r in results if r.q_value < args.q_threshold]
        if corrected:
            # Effect-size floor: with large n, trivially small excesses reach significance.
            significant = [r for r in significant if abs(r.excess) >= args.min_excess]
        enriched = sorted([r for r in significant if r.enriched],
                          key=lambda r: -abs(r.effect))[: args.top_n_enriched]
        depleted = sorted([r for r in significant if not r.enriched],
                          key=lambda r: -abs(r.effect))[: args.top_n_depleted]
        print(f"  {len(results)} features tested, {len(significant)} significant at q<{args.q_threshold}"
              f"{f' and |excess|>={args.min_excess}' if corrected else ''} "
              f"({len(enriched)} enriched / {len(depleted)} depleted shown)"
              f"{f', {n_skipped} skipped for missing baseline' if n_skipped else ''}")

        shown = enriched + depleted
        mlflow.log_metrics({
            "n_lineage_features_shown": sum(1 for r in shown if r.kind == KIND_LINEAGE),
            "n_process_features_shown": sum(1 for r in shown if r.kind == KIND_PROCESS),
            "n_tissue_features_shown": sum(1 for r in shown if r.kind == KIND_TISSUE),
            "n_features_skipped_no_baseline": n_skipped,
            "max_excess": max((abs(r.excess) for r in shown if r.excess is not None), default=0.0),
        })
        if corrected:
            # §L5: understudied cell types lack process annotation, and those are
            # disproportionately the interesting ones. Log how much of THIS gene's
            # data the process features could actually speak to.
            with_proc = sum(
                1 for p in pos + neg
                if any(k == KIND_PROCESS for k, _, _ in pair_features(
                    p, lookup, hierarchy_depth=args.hierarchy_depth, include_tissue=False,
                    include_process=True, go_depth=args.go_depth))
            )
            mlflow.log_metric("frac_rows_with_process_annotation", with_proc / max(len(pos) + len(neg), 1))

        homogeneity = tissue_homogeneity(pos, neg, lookup)
        log_json_artifact(homogeneity, "tissue_homogeneity.json")
        batch_risk = homogeneity["frac_allornothing_tissues"] > BATCH_ARTIFACT_THRESHOLD
        mlflow.set_tag("batch_artifact_warning", batch_risk)
        mlflow.log_metrics({
            "n_features_tested": len(results),
            "n_significant": len(significant),
            "n_enriched": len(enriched),
            "n_depleted": len(depleted),
            "n_allornothing_tissues": homogeneity["n_allornothing_tissues"],
            "frac_allornothing_tissues": homogeneity["frac_allornothing_tissues"],
        })
        if batch_risk:
            print(f"  WARNING: {homogeneity['n_allornothing_tissues']}/{homogeneity['n_tissues_considered']} tissues "
                  f"({homogeneity['frac_allornothing_tissues']:.0%}) are all-or-nothing positive/negative. "
                  f"Enrichment here likely reflects study/batch structure, not gene biology "
                  f"(PIPELINE_REQUIREMENTS.md §L9, §L11) — do not read this run as a biological result.")

        if not significant:
            print("No features reached significance — nothing to name. Aborting before the LLM call.")
            mlflow.set_tag("status", "EARLY_EXIT_NO_SIGNIFICANT_FEATURES")
            return

        template = (PROMPTS_DIR / f"{version_cfg['file']}.txt").read_text()
        enr_head = "ENRICHED where the gene is expressed" if corrected else \
                   "SIGNIFICANTLY ENRICHED where the gene is expressed:"
        dep_head = "DEPLETED where the gene is expressed" if corrected else \
                   "SIGNIFICANTLY DEPLETED where the gene is expressed:"
        prompt = template.format(
            enriched_block=render_features(enriched, lookup, enr_head, args.include_definition, corrected),
            depleted_block=render_features(depleted, lookup, dep_head, args.include_definition, corrected),
        )

        blinding_ok, hits = verify_blinding(prompt, forbidden_terms=[gene_symbol] if gene_symbol != gene_id else [])
        mlflow.set_tag("blinding_verified", blinding_ok)
        if not blinding_ok:
            mlflow.set_tag("status", "FAILED_BLINDING")
            raise SystemExit(f"Blinding check failed — prompt contains: {hits}")

        log_text_artifact(prompt, "prompt.txt")
        estimated = len(prompt) // 4
        mlflow.log_metric("estimated_prompt_tokens", estimated)
        print(f"Prompt is ~{estimated} tokens. Calling {model}...")

        num_ctx = compute_num_ctx(model)
        mlflow.log_param("num_ctx", num_ctx)
        llm = make_chat_llm(model=model, temperature=args.temperature, seed=args.seed,
                            format="json", num_ctx=num_ctx)
        try:
            parsed, raw_text, elapsed, retries, response_metadata = call_llm_with_retry(
                llm, prompt, REQUIRED_RESPONSE_KEYS)
        except RuntimeError as exc:
            log_text_artifact(str(exc), "parse_error.txt")
            mlflow.set_tag("status", "FAILED_PARSE")
            print(exc)
            return

        actual = response_metadata.get("prompt_eval_count")
        if actual is not None:
            mlflow.log_metric("actual_prompt_tokens", actual)
        log_text_artifact(raw_text, "response_raw.txt")
        log_json_artifact(parsed, "response_parsed.json")
        mlflow.log_metrics({"latency_s": elapsed, "parse_retries": retries, "confidence": parsed["confidence"]})

        # Novelty check (deterministic, no LLM): does the property say anything the
        # feature labels didn't already? Directly measures the concern that a
        # statistics-first design reduces the LLM to verbalising its own input.
        prop_text = parsed.get("property") or ""
        overlap_shown = property_overlap(prop_text, [r.label for r in shown])
        all_labels = [v.get("label") for section in ("cl", "uberon", "go")
                      for v in lookup._cache.get(section, {}).values() if v.get("label")]
        overlap_ontology = property_overlap(prop_text, all_labels)
        mlflow.log_metrics({
            "property_overlap_shown": overlap_shown,
            "property_overlap_ontology": overlap_ontology,
        })
        mlflow.set_tag("property_restates_input", overlap_shown >= RESTATEMENT_THRESHOLD)
        print(f"  property overlap with shown labels: {overlap_shown:.0%} "
              f"({'restatement' if overlap_shown >= RESTATEMENT_THRESHOLD else 'goes beyond input'})")

        mlflow.set_tags({"status": "COMPLETED", "abstained": bool(parsed["abstained"])})
        print(f"--- Response ({elapsed:.1f}s, {retries} retries) ---")
        print(json.dumps(parsed, indent=2))


if __name__ == "__main__":
    main()
