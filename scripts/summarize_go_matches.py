#!/usr/bin/env python3
"""
Overview of one condition's significant/predicted GO terms against the genes' own true
annotation: how many genes get nothing, and of what IS returned, how much is an exact match,
a generalisation (upward) or specialisation (downward) of the truth at what distance, and how
much is no match at all.

Run separately PER CONDITION on purpose (one go_enrichment version, or one go_llm
version x output_mode x evidence combination) — not a single pooled report — so that
re-running one arm after a change only means re-running this for that one condition; the
summaries for everything else stay untouched and valid. Each run of this script:

  1. Pulls COMPLETED runs for --condition / --gene-split from MLflow, restricted to the
     CURRENT code hash for that approach (core/run_identity.py) unless --allow-mixed — the
     same discipline as scripts/analyze_go_experiment.py, so a summary can never silently mix
     runs from different code.
  2. Builds the "significant" (or, for versions with no significance test, "predicted") term
     set per gene:
       go_enrichment v1/v2   p_gscs < alpha              (g:Profiler's own primary correction)
       go_enrichment v3/v4   q_value < alpha  (BH)
       go_enrichment v5/v6   no significance test exists (co-annotation transfer is a ranking,
                              not a hypothesis test) -> the full ranked go_predictions.json list
       go_llm (any version)  no significance test exists -> the full go_predictions.json list
     The caveat for the last two is recorded in the output, not silently assumed away.
  3. Classifies each term against the gene's DIRECT true annotations via core/go_match.py
     (exact / upward(d) / downward(d) / no_match), and bins the distances.
  4. Writes ONE JSON file: pretty-printed (human-readable), with a tidy per-gene table
     (notebook-ready, e.g. `pandas.DataFrame(data["per_gene"])`), the aggregate histograms,
     and the exact MLflow run_ids it was built from, so the summary is traceable and does not
     need to be re-derived by hand. A short table is also printed to the terminal.

Usage:
    python scripts/summarize_go_matches.py --condition go_enrichment:v3:contrast --gene-split dev
    python scripts/summarize_go_matches.py --condition go_llm:v3:freeform:true --gene-split dev
    python scripts/summarize_go_matches.py --condition go_enrichment:v5:contrast --gene-split dev --bin-edges 1,2,3,5,10
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import mlflow

from core.go_experiment import GOShared
from core.go_match import DOWNWARD, EXACT, NO_MATCH, UPWARD, bin_label, classify_terms
from core.mlflow_utils import _TRACKING_URI
from core.run_identity import fingerprint

OUT_DIR = Path(__file__).parents[1] / "data" / "go_experiment" / "match_summaries"
DEFAULT_BIN_EDGES = [1, 2, 3, 5, 10]
# Versions with no hypothesis test: there is nothing to call "significant" at a threshold, so the
# full ranked prediction list stands in for it (documented in the output as `significance_basis`).
NO_SIGNIFICANCE_VERSIONS = {"v5", "v6"}


def approach_of(condition: str) -> str:
    approach = condition.split(":", 1)[0]
    if approach not in ("go_enrichment", "go_llm"):
        raise SystemExit(f"Unrecognised condition {condition!r}: must start with 'go_enrichment:' or 'go_llm:'")
    return approach


def load_runs(condition: str, gene_split: str, allow_mixed: bool) -> tuple[list[dict], dict]:
    """Latest COMPLETED run per gene for this exact condition, restricted to the current code hash."""
    approach = approach_of(condition)
    mlflow.set_tracking_uri(_TRACKING_URI)
    exp = mlflow.get_experiment_by_name(f"CellCoLLM/{approach}")
    if exp is None:
        raise SystemExit(f"No experiment CellCoLLM/{approach} found.")
    df = mlflow.search_runs(
        [exp.experiment_id], max_results=50000,
        filter_string=f"tags.condition = '{condition}' and tags.gene_split = '{gene_split}' and tags.status = 'COMPLETED'",
        order_by=["start_time ASC"],
    )
    current = fingerprint(approach)
    latest: dict[str, dict] = {}
    excluded = {"stale_code": 0, "unfrozen": 0}
    for _, r in df.iterrows():
        h = r.get("tags.code_hash")
        if not allow_mixed and h != current:
            excluded["stale_code"] += 1
            continue
        if not allow_mixed and str(r.get("tags.unfrozen_override")) == "True" and gene_split == "test":
            excluded["unfrozen"] += 1
            continue
        latest[r["tags.gene_id"]] = {"run_id": r["run_id"], "gene_symbol": r.get("tags.gene_symbol", r["tags.gene_id"])}
    meta = {"approach": approach, "current_code_hash": current, "excluded": excluded, "n_runs_seen": len(df)}
    return list(latest.items()), meta


def significant_ids(approach: str, version: str, run_id: str, alpha: float) -> tuple[list[str], str]:
    """(go_ids, basis string describing how they were selected)."""
    def art(path):
        return json.load(open(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=path)))

    if approach == "go_enrichment" and version in ("v1", "v2"):
        rows = art("enrichment_results.json")
        return [r["go_id"] for r in rows if r.get("p_gscs", 1.0) < alpha], f"p_gscs < {alpha} (g:SCS)"
    if approach == "go_enrichment" and version in ("v3", "v4"):
        rows = art("enrichment_results.json")
        return [r["go_id"] for r in rows if r.get("tested") and r.get("q_value", 1.0) < alpha], f"q_value < {alpha} (BH)"
    # go_enrichment v5/v6, and every go_llm condition: no hypothesis test exists.
    preds = art("go_predictions.json")
    label = f"go_enrichment {version} (co-annotation transfer)" if approach == "go_enrichment" else f"go_llm {version}"
    return [r["go_id"] for r in preds], f"NO SIGNIFICANCE TEST for {label} — full ranked prediction list used instead"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", required=True, help="e.g. go_enrichment:v3:contrast or go_llm:v3:freeform:true")
    ap.add_argument("--gene-split", required=True, choices=["dev", "test", "smoke", "all"])
    ap.add_argument("--alpha", type=float, default=0.05, help="Significance threshold for v1-v4 (ignored otherwise)")
    ap.add_argument("--bin-edges", default=",".join(map(str, DEFAULT_BIN_EDGES)),
                    help="Comma-separated upper bounds for the (non-zero) distance bins, e.g. '1,2,3,5,10'")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--allow-mixed", action="store_true",
                    help="Include runs from any code version (plumbing checks only)")
    args = ap.parse_args()
    bin_edges = [int(x) for x in args.bin_edges.split(",")]
    version = args.condition.split(":")[1]

    print(f"Loading shared state (ontology, annotations)...")
    sh = GOShared.load(need_baselines=False)
    runs, meta = load_runs(args.condition, args.gene_split, args.allow_mixed)
    print(f"  {len(runs)} genes with a COMPLETED run under the current code "
          f"({meta['excluded']['stale_code']} excluded as stale, {meta['excluded']['unfrozen']} as unfrozen)")
    if not runs:
        raise SystemExit("Nothing to summarise. (--allow-mixed includes runs from any code version, for plumbing checks.)")

    per_gene = []
    up_hist: dict[str, int] = {}
    down_hist: dict[str, int] = {}
    n_exact = n_upward = n_downward = n_no_match = n_zero_sig = n_unscoreable = 0
    basis = None
    for gene_id, info in runs:
        if not sh.truth.has(gene_id):
            n_unscoreable += 1
            continue
        ids, basis = significant_ids(meta["approach"], version, info["run_id"], args.alpha)
        truth = sh.truth.annotations[gene_id]
        matches = classify_terms(sh.truth.go, ids, truth)
        row = {"gene_id": gene_id, "symbol": info["gene_symbol"], "run_id": info["run_id"],
               "n_significant": len(matches), "n_exact": 0, "n_upward": 0, "n_downward": 0, "n_no_match": 0}
        if not matches:
            n_zero_sig += 1
        for m in matches:
            if m.kind == EXACT:
                n_exact += 1; row["n_exact"] += 1
            elif m.kind == UPWARD:
                n_upward += 1; row["n_upward"] += 1
                b = bin_label(m.distance, bin_edges); up_hist[b] = up_hist.get(b, 0) + 1
            elif m.kind == DOWNWARD:
                n_downward += 1; row["n_downward"] += 1
                b = bin_label(m.distance, bin_edges); down_hist[b] = down_hist.get(b, 0) + 1
            else:
                n_no_match += 1; row["n_no_match"] += 1
        per_gene.append(row)

    n_genes = len(per_gene)
    n_terms = n_exact + n_upward + n_downward + n_no_match
    report = {
        "condition": args.condition, "approach": meta["approach"], "version": version,
        "gene_split": args.gene_split, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_hash": meta["current_code_hash"], "alpha": args.alpha, "bin_edges": bin_edges,
        "significance_basis": basis,
        "mlflow": {"tracking_uri": _TRACKING_URI, "n_runs_matched": len(runs),
                  "n_excluded_stale_code": meta["excluded"]["stale_code"],
                  "n_excluded_unfrozen": meta["excluded"]["unfrozen"],
                  "run_ids": {g: info["run_id"] for g, info in runs}},  # every matched gene -> its run_id, for traceability
        "genes": {"n_total": n_genes, "n_unscoreable_no_truth": n_unscoreable,
                 "n_zero_significant": n_zero_sig,
                 "frac_zero_significant": n_zero_sig / n_genes if n_genes else None},
        "terms": {"n_total": n_terms, "n_exact": n_exact, "n_upward_total": n_upward,
                 "n_downward_total": n_downward, "n_no_match": n_no_match,
                 "frac_exact": n_exact / n_terms if n_terms else None,
                 "frac_no_match": n_no_match / n_terms if n_terms else None,
                 "upward_bins": up_hist, "downward_bins": down_hist},
        "per_gene": per_gene,
    }

    safe = args.condition.replace(":", "_")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    out_path = out / f"match_summary__{safe}__{args.gene_split}.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=False))

    print(f"\n=== {args.condition}  ({args.gene_split}) ===")
    print(f"basis: {basis}")
    print(f"genes: {n_genes} scored, {n_zero_sig} with zero significant terms ({report['genes']['frac_zero_significant']:.0%})"
          if n_genes else "no scoreable genes")
    print(f"terms: {n_terms} total"
          f"  exact={n_exact} ({n_exact / n_terms:.0%})" if n_terms else "")
    if n_terms:
        print(f"       upward (generalisation)  ={n_upward:4d} ({n_upward / n_terms:.0%})  " +
              ", ".join(f"{k}:{v}" for k, v in sorted(up_hist.items())))
        print(f"       downward (specialisation)={n_downward:4d} ({n_downward / n_terms:.0%})  " +
              ", ".join(f"{k}:{v}" for k, v in sorted(down_hist.items())))
        print(f"       no match                 ={n_no_match:4d} ({n_no_match / n_terms:.0%})")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
