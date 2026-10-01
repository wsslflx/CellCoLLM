#!/usr/bin/env python3
"""
No-LLM baselines: classical GO over-representation analysis of a gene's positive
(or negative) cell-type set, following g:Profiler's g:GOSt method (v1/v2), and the
same evidence under the baseline-corrected two-factor null (v3/v4).

This is arm 1 of a two-arm experiment isolating what an LLM actually adds. Both
arms see the same evidence — GO biological processes attached to cell types via
CL's capable_of relation — but this arm uses only statistics. No model is involved
at any point.

Four versions, encoding statistic x unit of analysis:
  v1  g:Profiler-style (hypergeometric + g:SCS), distinct cell types
  v2  g:Profiler-style, dataset rows (CL x tissue) — violates the distinct-draws
      assumption, so its p-values are anti-conservative
  v3  corrected: binomial vs a two-factor null (gene rate x term baseline), cell types
  v4  corrected, dataset rows
v1/v2 run one query per direction (positive / negative); v3/v4 compute the contrast
internally and produce ONE run per gene tagged input_set="contrast".

READ THE README BEFORE INTERPRETING v1/v2. Classical over-representation assumes the
query is a small slice of the background. Here it often isn't — query/background runs
33%-82% across the test genes — which caps the maximum achievable fold enrichment at
N/n (as low as 1.22x) while making trivial folds highly significant.

Scoring: the deliverable is a ranked GO term table. Since the GO-prediction experiment,
each run also writes a ranked prediction list (`go_predictions.json`, plus an
effect-size-ranked `go_predictions_effect.json`) and scores it against the gene's own GO
annotation into the SAME MLflow run (`gopred_*` metrics). v1/v2 score the POSITIVE
direction only; the negative direction stays diagnostic. --no-score disables this.

--universe called uses the universe of the comparison ladder: annotated items the gene
is called for, biological-process annotations only. The default `annotated` is the
original behaviour (counts missing items, and non-BP capable_of targets such as
"bicarbonate transmembrane transporter activity") and is kept so earlier runs stay
reproducible.

Usage:
    python approaches/go_enrichment/run_go_enrichment.py --gene ENSG00000132763
    python approaches/go_enrichment/run_go_enrichment.py --gene ENSG00000132763 --prompt-version v3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

import mlflow
import numpy as np

from core.data_loader import GeneExpressionDataset, parse_pair
from core.go_enrichment import enrich
from core.go_evidence import (
    UNIT_CELL_TYPES, UNIT_ROWS, baselines_key, corrected_stats, rank_corrected, rank_gprofiler,
)
from core.go_experiment import GOShared
from core.go_scoring import log_scored_predictions, log_significant_only_score
from core.mlflow_utils import (
    RunContext,
    get_or_create_gene_parent_run,
    log_json_artifact,
    log_text_artifact,
    tracked_run,
)
from core.ontology_lookup import OntologyLookup
from core.run_identity import fingerprint

APPROACH = "go_enrichment"
PROMPT_VERSIONS = {
    "v1": {"unit": UNIT_CELL_TYPES, "statistics": "gprofiler"},
    "v2": {"unit": UNIT_ROWS, "statistics": "gprofiler"},
    "v3": {"unit": UNIT_CELL_TYPES, "statistics": "corrected"},
    "v4": {"unit": UNIT_ROWS, "statistics": "corrected"},
    "v5": {"unit": UNIT_CELL_TYPES, "statistics": "transfer"},
    "v6": {"unit": UNIT_ROWS, "statistics": "transfer"},
}
# Per statistic: MLflow prompt_mode and the label used in prompt_version.
STAT_MODE = {"gprofiler": ("go_overrepresentation", "gprofiler_gost"),
             "corrected": ("go_corrected", "corrected_two_factor"),
             "transfer": ("go_transfer", "corrected_plus_coannotation_transfer")}
# Above this query/background ratio, significance stops being informative because
# the background is mostly the query — see the README's dedicated section.
LOW_POWER_RATIO = 0.5
TSV_COLUMNS = ("go_id", "label", "term_size", "query_size", "intersection_size", "expected",
               "fold_enrichment", "precision", "recall", "p_value", "p_gscs",
               "p_bonferroni", "p_fdr_bh")


def build_annotation(ds: GeneExpressionDataset, lookup: OntologyLookup, unit: str, go_depth: int):
    """
    Legacy universe (--universe annotated). Returns (background_items, item_to_go, term_labels).

    Background is g:Profiler's default "only annotated" domain scope: items with
    at least one GO term. Un-annotated items can never be enriched, so including
    them would only dilute every test.
    """
    cl_processes: dict[str, set[str]] = {}
    labels: dict[str, str] = {}
    for cl_id in {parse_pair(p)[0] for p in ds.df.index}:
        terms = lookup.processes(cl_id, go_depth)
        if terms:
            cl_processes[cl_id] = {t.id for t in terms}
            for t in terms:
                labels[t.id] = t.label or t.id

    if unit == UNIT_CELL_TYPES:
        background = sorted(cl_processes)
        item_to_go = cl_processes
    else:  # rows: each CL|UBERON pair is its own item, inheriting its cell type's terms
        background = [p for p in ds.df.index if parse_pair(p)[0] in cl_processes]
        item_to_go = {p: cl_processes[parse_pair(p)[0]] for p in background}
    return background, item_to_go, labels


def to_items(pairs: list[str], unit: str) -> list[str]:
    return sorted({parse_pair(p)[0] for p in pairs}) if unit == UNIT_CELL_TYPES else list(pairs)


def _context(args, gene_id, gene_symbol, prompt_version, input_set, unit, statistics, extra, ds, lookup, shared):
    tags = {}
    if getattr(args, "gene_split", None):
        tags["gene_split"] = args.gene_split
    if shared is not None and shared.truth is not None:
        tags["stratum"] = shared.stratum(gene_id)
    tags["condition"] = f"{APPROACH}:{prompt_version}:{input_set}"
    tags["code_hash"] = fingerprint(APPROACH)   # content hash of the files that shape this run's output
    if getattr(args, "unfrozen_override", False):
        tags["unfrozen_override"] = True        # test-split run started without a matching freeze
    return RunContext(
        approach=APPROACH,
        approach_version=prompt_version,
        gene_id=gene_id,
        gene_symbol=gene_symbol,
        species=args.species,
        model="none",  # explicit marker: no LLM is involved in this arm
        prompt_mode=STAT_MODE[statistics][0],
        input_set=input_set,
        temperature=0.0,
        seed=args.seed,
        dataset_hash=ds.dataset_hash,
        prompt_version=f"{STAT_MODE[statistics][1]}_{unit}_{prompt_version}",
        extra_params={
            "cl_data_version": lookup.provenance.get("cl_data_version"),
            "unit": unit,
            "statistics": statistics,
            "go_depth": args.go_depth,
            "min_term_size": args.min_term_size,
            **extra,
        },
        extra_tags=tags,
    )


def run_direction(
    ds, lookup, gene_id, gene_symbol, input_set, parent_run_id, prompt_version, args, shared=None,
) -> None:
    """v1/v2: one g:Profiler-style query against the background."""
    cfg = PROMPT_VERSIONS[prompt_version]
    unit = cfg["unit"]
    if args.universe == "called":
        background, item_to_go = shared.builder.called_universe(gene_id, unit)
        labels = shared.builder.labels
    else:
        background, item_to_go, labels = build_annotation(ds, lookup, unit, args.go_depth)
    pairs = ds.positive_cell_types(gene_id) if input_set == "positive" else ds.negative_cell_types(gene_id)
    query = to_items(pairs, unit)

    ctx = _context(args, gene_id, gene_symbol, prompt_version, input_set, unit, "gprofiler", {
        "universe": args.universe,
        "background_scope": "annotated_only" if args.universe == "annotated" else "annotated_called_bp_only",
        "max_term_size": args.max_term_size,
        "n_simulations": args.n_simulations,
        "alpha": args.alpha,
        "correction_methods": "gscs,bonferroni,fdr_bh",
    }, ds, lookup, shared)

    with tracked_run(ctx, parent_run_id=parent_run_id) as run:
        print(f"\n[{input_set}] MLflow run: {run.info.run_id} (parent: {parent_run_id})")
        results, diag = enrich(
            query, background, item_to_go, labels,
            min_term_size=args.min_term_size, max_term_size=args.max_term_size,
            n_simulations=args.n_simulations, alpha=args.alpha, seed=args.seed,
        )

        ratio = diag["query_background_ratio"]
        ceiling = diag["max_fold_enrichment_possible"]
        print(f"[{input_set}] unit={unit}  query={diag['n_query']} / background={diag['n_background']} "
              f"({ratio:.0%})  terms tested={diag['n_terms_tested']}")
        print(f"[{input_set}] g:SCS threshold={diag['gscs_threshold']:.2e}  "
              f"max possible fold enrichment={ceiling:.2f}x")

        n_gscs = sum(1 for r in results if r.p_gscs < args.alpha)
        n_bonf = sum(1 for r in results if r.p_bonferroni < args.alpha)
        n_bh = sum(1 for r in results if r.p_fdr_bh < args.alpha)
        low_power = ratio > LOW_POWER_RATIO
        mlflow.log_metrics({
            "n_background": diag["n_background"],
            "n_query": diag["n_query"],
            "query_background_ratio": ratio,
            "max_fold_enrichment_possible": ceiling,
            "n_terms_total": diag["n_terms_total"],
            "n_terms_tested": diag["n_terms_tested"],
            "n_significant_gscs": n_gscs,
            "n_significant_bonferroni": n_bonf,
            "n_significant_fdr_bh": n_bh,
            "gscs_threshold": diag["gscs_threshold"],
        })
        mlflow.set_tag("low_power_warning", low_power)
        if low_power:
            print(f"[{input_set}] WARNING: the query is {ratio:.0%} of the background, so no term can "
                  f"exceed {ceiling:.2f}x enrichment and trivial folds will still reach significance. "
                  f"Read fold_enrichment against the {ceiling:.2f}x ceiling, not the p-value. "
                  f"See approaches/README.md (go_enrichment / background-query size problem).")

        log_json_artifact([r.as_dict() for r in results], "enrichment_results.json")
        # g:Profiler's own default (significant=TRUE) returns ONLY this table, not the full one above.
        # We keep the full table as what's actually scored (go_predictions.json, below) because a
        # significant-only cut would shrink or empty the ranked list for many genes (10/40 dev genes had
        # zero significant A1' terms) for reasons unrelated to prediction quality, and would unfairly
        # handicap the no-LLM rungs against go_llm, which always answers with up to 10 terms. This
        # artifact exists purely so "what would g:Profiler actually show you" is inspectable separately.
        log_json_artifact([r.as_dict() for r in results if r.p_gscs < args.alpha], "enrichment_results_significant_only.json")
        log_json_artifact(diag, "gscs_simulation.json")
        tsv = "\t".join(TSV_COLUMNS) + "\n" + "\n".join(
            "\t".join(str(getattr(r, c)) for c in TSV_COLUMNS) for r in results
        )
        log_text_artifact(tsv, "enrichment_results.tsv")

        if results:
            top = results[0]
            mlflow.log_metrics({
                "top_term_p_gscs": top.p_gscs,
                "top_term_p_value": top.p_value,
                "top_term_fold_enrichment": top.fold_enrichment,
                "top_term_precision": top.precision,
                "top_term_recall": top.recall,
            })
            print(f"[{input_set}] significant: g:SCS {n_gscs}, Bonferroni {n_bonf}, BH {n_bh} "
                  f"(of {diag['n_terms_tested']} tested)")
            print(f"[{input_set}] top terms by g:SCS p-value:")
            for r in results[:8]:
                flag = "" if r.p_gscs < args.alpha else "  (n.s.)"
                print(f"    {r.fold_enrichment:5.2f}x  p_gscs={r.p_gscs:9.2e}  "
                      f"{r.intersection_size:4d}/{r.term_size:<4d}  {r.label[:44]}{flag}")
            mlflow.set_tag("status", "COMPLETED")
        else:
            print(f"[{input_set}] no terms passed the size filter — nothing to report.")
            mlflow.set_tag("status", "EARLY_EXIT_NO_TERMS")

        # Score the POSITIVE direction against the gene's own GO annotation (same run).
        if args.score and input_set == "positive" and shared is not None and shared.truth.has(gene_id):
            cands = shared.builder.candidates
            rank_p = rank_gprofiler(results, cands, shared.builder.labels, "p")
            m = log_scored_predictions(
                rank_p, rank_gprofiler(results, cands, shared.builder.labels, "effect"),
                shared.truth, gene_id, shared.model)
            print(f"[{input_set}] scored: F1@3={m['f1_at_3']:.3f}  headroom@3={m.get('headroom_at_3', float('nan')):.3f}")
            sig_ids = {r.go_id for r in results if r.p_gscs < args.alpha}
            ms = log_significant_only_score(rank_p, sig_ids, shared.truth, gene_id, shared.model)
            print(f"[{input_set}] significant-only (p_gscs<{args.alpha}, {len(sig_ids)} terms): "
                  + (f"F1@3={ms['f1_at_3']:.3f}  headroom@3={ms.get('headroom_at_3', float('nan')):.3f}" if ms else "no significant terms"))


def run_corrected(ds, lookup, gene_id, gene_symbol, parent_run_id, prompt_version, args, shared) -> None:
    """v3/v4: the contrast computed internally — one run per gene, input_set='contrast'."""
    unit = PROMPT_VERSIONS[prompt_version]["unit"]
    b = shared.builder
    ev = b.evidence(gene_id, unit)
    ctx = _context(args, gene_id, gene_symbol, prompt_version, "contrast", unit, "corrected", {
        "universe": "called",
        "background_scope": "annotated_called_bp_only",
        "baselines_key": baselines_key(b),
        "grand_mean_rate": round(shared.baselines[unit]["grand_mean"], 6),
        "gene_rate": round(ev.gene_rate, 6),
        "n_candidates": len(b.candidates),
    }, ds, lookup, shared)

    with tracked_run(ctx, parent_run_id=parent_run_id) as run:
        print(f"\n[contrast] MLflow run: {run.info.run_id} (parent: {parent_run_id})")
        results = corrected_stats(ev, shared.baselines[unit], args.min_term_size, b.candidates)
        live = [r for r in results if r.tested]
        n_q05 = sum(1 for r in live if r.q_value < args.alpha)
        print(f"[contrast] unit={unit}  positive={ev.n_pos} / called={ev.n_called} ({ev.gene_rate:.0%})  "
              f"terms tested={len(live)}  significant (q<{args.alpha}): {n_q05}")
        mlflow.log_metrics({
            "n_pos": ev.n_pos, "n_called": ev.n_called, "gene_rate": ev.gene_rate,
            "n_terms_tested": len(live), "n_significant_q05": n_q05,
            "max_excess": max((r.excess for r in live), default=0.0),
        })
        log_json_artifact([r.as_dict() for r in results], "enrichment_results.json")
        # Same note as v1/v2: g:Profiler's significant=TRUE default would show only this subset.
        log_json_artifact([r.as_dict() for r in live if r.q_value < args.alpha], "enrichment_results_significant_only.json")
        top = sorted((r for r in live if r.excess > 0), key=lambda r: r.p_value)[:8]
        print("[contrast] top enriched terms by p:")
        for r in top:
            print(f"    observed {r.observed:4.0%} vs expected {r.expected:4.0%}  excess {r.excess:+5.0%}  "
                  f"q={r.q_value:8.2e}  {r.k_pos:4d}/{r.K:<4d}  {r.label[:44]}")
        mlflow.set_tag("status", "COMPLETED")
        if args.score and shared.truth.has(gene_id):
            rank_p = rank_corrected(results, "p")
            m = log_scored_predictions(rank_p, rank_corrected(results, "effect"),
                                       shared.truth, gene_id, shared.model)
            print(f"[contrast] scored: F1@3={m['f1_at_3']:.3f}  headroom@3={m.get('headroom_at_3', float('nan')):.3f}")
            sig_ids = {r.go_id for r in live if r.q_value < args.alpha}
            ms = log_significant_only_score(rank_p, sig_ids, shared.truth, gene_id, shared.model)
            print(f"[contrast] significant-only (q<{args.alpha}, {len(sig_ids)} terms): "
                  + (f"F1@3={ms['f1_at_3']:.3f}  headroom@3={ms.get('headroom_at_3', float('nan')):.3f}" if ms else "no significant terms"))


def run_transfer(ds, lookup, gene_id, gene_symbol, parent_run_id, prompt_version, args, shared) -> None:
    """
    v5/v6: corrected statistics (as v3/v4) turned into a ranking of GENE-LEVEL GO terms by co-annotation
    transfer (core/go_transfer.py). The no-LLM counterpart for the free-form comparison: unlike v1-v4 it
    can name terms outside the 52-term evidence vocabulary. The target gene is left out of the
    association table, so its own annotation cannot leak in.
    """
    unit = PROMPT_VERSIONS[prompt_version]["unit"]
    b = shared.builder
    ev = b.evidence(gene_id, unit)
    tr = shared.transfer(args.transfer_min_genes)
    ctx = _context(args, gene_id, gene_symbol, prompt_version, "contrast", unit, "transfer", {
        "universe": "called",
        "background_scope": "annotated_called_bp_only",
        "baselines_key": baselines_key(b),
        "grand_mean_rate": round(shared.baselines[unit]["grand_mean"], 6),
        "gene_rate": round(ev.gene_rate, 6),
        "n_candidates": len(b.candidates),
        "transfer_min_genes": args.transfer_min_genes,
        "transfer_vocab_size": len(tr.vocab),
        "transfer_top_n": args.transfer_top_n,
        "transfer_leave_one_out": True,
        "transfer_weight": "neg_log10_p_enriched_only",
    }, ds, lookup, shared)

    with tracked_run(ctx, parent_run_id=parent_run_id) as run:
        print(f"\n[contrast] MLflow run: {run.info.run_id} (parent: {parent_run_id})")
        results = corrected_stats(ev, shared.baselines[unit], args.min_term_size, b.candidates)
        weights = {r.go_id: float(-np.log10(max(r.p_value, 1e-300))) for r in results if r.tested and r.excess > 0}
        ranking = tr.rank(gene_id, weights, top_n=args.transfer_top_n)
        in_table = sum(1 for r in ranking[:10] if r["go_id"] in set(b.candidates))
        print(f"[contrast] unit={unit}  {len(weights)} enriched evidence terms -> {len(ranking)} gene-level terms ranked "
              f"(vocabulary {len(tr.vocab)}); {in_table}/10 of the top 10 are evidence terms")
        mlflow.log_metrics({"n_pos": ev.n_pos, "n_called": ev.n_called, "gene_rate": ev.gene_rate,
                            "n_weighted_terms": len(weights), "n_ranked": len(ranking),
                            "frac_in_table": in_table / max(min(10, len(ranking)), 1)})
        log_json_artifact([r.as_dict() for r in results], "enrichment_results.json")
        log_json_artifact(tr.explain(gene_id, weights), "transfer_explanation.json")
        print("[contrast] top predicted gene-level terms:", [r["label"] for r in ranking[:6]])
        mlflow.set_tag("status", "COMPLETED")
        if args.score and shared.truth.has(gene_id):
            # No candidate-vocabulary ceiling applies: the answer space is the whole vocabulary.
            m = log_scored_predictions(ranking, None, shared.truth, gene_id, None)
            print(f"[contrast] scored: F1@3={m['f1_at_3']:.3f}  F1@10={m['f1_at_10']:.3f}")


def run_gene(gene_id: str, gene_symbol: str, args, shared: GOShared | None = None,
             ds=None, lookup=None) -> None:
    """Run one gene for args.prompt_version. `shared` lets a batch runner reuse loaded state."""
    cfg = PROMPT_VERSIONS[args.prompt_version]
    uses_baselines = cfg["statistics"] in ("corrected", "transfer")
    need_builder = args.universe == "called" or uses_baselines or args.score
    if shared is None and need_builder:
        shared = GOShared.load(args.dataset, args.go_depth, args.min_term_size,
                               need_truth=args.score or cfg["statistics"] == "transfer", need_baselines=uses_baselines)
    if shared is not None:
        ds, lookup = shared.ds, shared.lookup
    if ds is None:
        print("Loading dataset...")
        ds = GeneExpressionDataset.load(args.dataset) if args.dataset else GeneExpressionDataset.load()
    if lookup is None:
        print("Loading ontology cache...")
        lookup = OntologyLookup()
    if "go" not in lookup._cache:
        raise SystemExit(
            "The ontology cache has no 'go' section — rebuild it with capable_of extraction:\n"
            "    python scripts/build_ontology_cache.py"
        )
    if not ds.has_gene(gene_id):
        raise SystemExit(f"Gene {gene_id!r} not present in dataset ({ds.path.name}). Aborting.")

    summary = ds.expression_summary(gene_id)
    print(f"Gene {gene_id} ({gene_symbol}): {summary}")
    parent_run_id = get_or_create_gene_parent_run(
        approach=APPROACH, gene_id=gene_id, gene_symbol=gene_symbol, species=args.species,
        dataset_hash=ds.dataset_hash, expression_summary=summary,
    )
    if cfg["statistics"] == "corrected":
        run_corrected(ds, lookup, gene_id, gene_symbol, parent_run_id, args.prompt_version, args, shared)
        return
    if cfg["statistics"] == "transfer":
        run_transfer(ds, lookup, gene_id, gene_symbol, parent_run_id, args.prompt_version, args, shared)
        return
    directions = ["positive", "negative"] if args.input_set == "both" else [args.input_set]
    for direction in directions:
        run_direction(ds, lookup, gene_id, gene_symbol, direction, parent_run_id,
                      args.prompt_version, args, shared)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gene", required=True, help="Canonical Ensembl gene ID")
    parser.add_argument("--gene-symbol", default=None, help="For MLflow tags only")
    parser.add_argument("--species", default="human")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--set", dest="input_set", choices=["positive", "negative", "both"], default="both",
                         help="v1/v2 only; v3/v4 compute the contrast internally")
    parser.add_argument("--prompt-version", choices=list(PROMPT_VERSIONS), default="v1",
                         help="v1/v2 = g:Profiler-style (cell types / rows); v3/v4 = corrected (cell types / rows); "
                              "v5/v6 = corrected + co-annotation transfer, can name terms outside the evidence vocabulary")
    parser.add_argument("--go-depth", type=int, default=3,
                         help="is_a depth over which capable_of GO terms are inherited (default 3)")
    parser.add_argument("--min-term-size", type=int, default=3,
                         help="Skip terms covering fewer background items than this. Default 3 because "
                              "56 of the 126 available GO terms cover exactly one cell type")
    parser.add_argument("--max-term-size", type=int, default=0, help="0 = unlimited (v1/v2)")
    parser.add_argument("--n-simulations", type=int, default=2000,
                         help="Random queries used to derive the g:SCS threshold (default 2000, "
                              "matching g:Profiler's original simulations)")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--universe", choices=["annotated", "called"], default="annotated",
                         help="v1/v2: 'annotated' = original behaviour; 'called' = the comparison ladder's universe "
                              "(annotated items the gene is called for, biological-process annotations only). "
                              "v3/v4 always use 'called'")
    parser.add_argument("--transfer-min-genes", type=int, default=30,
                         help="v5/v6: a gene-level term needs at least this many annotated genes to enter the vocabulary")
    parser.add_argument("--transfer-top-n", type=int, default=100, help="v5/v6: length of the ranked prediction list")
    parser.add_argument("--score", action=argparse.BooleanOptionalAction, default=True,
                         help="Score the ranked prediction against the gene's own GO annotation into the same run")
    parser.add_argument("--gene-split", choices=["dev", "test", "smoke"], default=None,
                         help="Tag the run with its experiment split (see scripts/select_go_genes.py)")
    return parser


def make_args(**overrides) -> argparse.Namespace:
    """Argument namespace with defaults, for in-process callers (scripts/run_go_experiment.py)."""
    ns = build_parser().parse_args(["--gene", overrides.pop("gene", "X")])
    ns.__dict__.update(overrides)
    return ns


def main() -> None:
    args = build_parser().parse_args()
    run_gene(args.gene, args.gene_symbol or args.gene, args)


if __name__ == "__main__":
    main()
