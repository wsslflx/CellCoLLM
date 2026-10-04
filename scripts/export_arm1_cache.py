#!/usr/bin/env python3
"""
One-off export: flatten everything arm 1 (go_enrichment v1) produced into a single table, so later analyses
read one parquet file in seconds instead of re-downloading ~31,500 MLflow artifacts (2-5 minutes per pass).

Writes to data/go_experiment/cache/:
  arm1_tested_terms__<split>.parquet   one row per (gene, TESTED term), significant or not:
      gene_id, symbol, go_id, label, term_size (K in the gene's called universe), query_size (n),
      n_background (N), intersection_size, expected, fold_enrichment, p_value, p_gscs, p_bonferroni, p_fdr_bh,
      gscs_threshold, p_floor (smallest p the test could produce for this K, n, N), reachable,
      significant (p_gscs < alpha), rank (among the gene's significant terms, by p_gscs then go_id),
      kind / distance / matched_term (core.go_match.classify_term against the gene's direct true terms),
      ic (predicted term), ic_true / ic_gap (IC of the matched true term; gap = ic_true - ic; match kinds only)
  arm1_cache_meta__<split>.json        condition, alpha, n genes, generated_at

Gene-level columns (breadth, gene family, ceiling, chance baseline, ...) already live in
arm1_chance_baseline__<split>.json (a superset of arm1_gene_properties__<split>.json); read both through
core/arm1_cache.py::load_arm1_cache().

Re-run this only when arm 1 itself is re-run. Usage:
    python scripts/export_arm1_cache.py --condition go_enrichment:v1:positive --gene-split all
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import mlflow
import pandas as pd
from scipy.stats import hypergeom

from core.arm1_cache import CACHE_DIR
from core.go_experiment import GOShared
from core.go_match import classify_term
from scripts.summarize_go_matches import load_runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", default="go_enrichment:v1:positive")
    ap.add_argument("--gene-split", default="all")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=None, help="Smoke-test on a subset")
    args = ap.parse_args()

    sh = GOShared.load(need_baselines=False)
    go = sh.truth.go
    # allow_mixed=True: same reasoning as scripts/analyze_arm1_gene_properties.py (deterministic artifacts only;
    # truth/IC recomputed here).
    runs, _ = load_runs(args.condition, args.gene_split, allow_mixed=True)
    runs = [(g, i) for g, i in runs if sh.truth.has(g)]
    if args.limit:
        runs = runs[: args.limit]
    print(f"{len(runs)} genes")

    rows = []
    for n_done, (gene_id, info) in enumerate(runs, 1):
        enr = json.load(open(mlflow.artifacts.download_artifacts(run_id=info["run_id"], artifact_path="enrichment_results.json")))
        diag = json.load(open(mlflow.artifacts.download_artifacts(run_id=info["run_id"], artifact_path="gscs_simulation.json")))
        thr, N, n = diag["gscs_threshold"], diag["n_background"], diag["n_query"]
        td = sh.truth.annotations[gene_id]
        sig = sorted((r for r in enr if r.get("p_gscs", 1.0) < args.alpha), key=lambda r: (r["p_gscs"], r["go_id"]))
        rank = {r["go_id"]: i for i, r in enumerate(sig, 1)}
        for r in enr:
            t, K = r["go_id"], r["term_size"]
            m = classify_term(go, t, td)
            p_floor = float(hypergeom.sf(min(K, n) - 1, N, K, n)) if n > 0 else 1.0
            true_t = t if m.kind == "exact" else m.matched_term
            ic, ic_true = go.ic(t), (go.ic(true_t) if true_t else None)
            rows.append({
                "gene_id": gene_id, "symbol": info["gene_symbol"], "go_id": t, "label": r.get("label", go.label(t)),
                "term_size": K, "query_size": n, "n_background": N, "intersection_size": r["intersection_size"],
                "expected": r["expected"], "fold_enrichment": r["fold_enrichment"], "p_value": r["p_value"],
                "p_gscs": r["p_gscs"], "p_bonferroni": r["p_bonferroni"], "p_fdr_bh": r["p_fdr_bh"],
                "gscs_threshold": thr, "p_floor": p_floor,
                # relative tolerance: when the query nearly fills the background, p_floor == threshold up to
                # float rounding (see scripts/analyze_arm1_term_profile.py)
                "reachable": p_floor <= thr * (1 + 1e-9),
                "significant": r.get("p_gscs", 1.0) < args.alpha, "rank": rank.get(t),
                "kind": m.kind, "distance": m.distance, "matched_term": true_t,
                "ic": ic, "ic_true": ic_true, "ic_gap": (ic_true - ic) if ic_true is not None else None,
            })
        if n_done % 2000 == 0:
            print(f"  {n_done}/{len(runs)} genes")

    df = pd.DataFrame(rows)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = CACHE_DIR / f"arm1_tested_terms__{args.gene_split}.parquet"
    df.to_parquet(out, index=False)
    (CACHE_DIR / f"arm1_cache_meta__{args.gene_split}.json").write_text(json.dumps({
        "condition": args.condition, "gene_split": args.gene_split, "alpha": args.alpha, "n_genes": len(runs),
        "n_rows": len(df), "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=2))

    sig = df[df.significant]
    print(f"\nWrote {out}  ({len(df):,} tested rows, {len(sig):,} significant)")
    print("significant by kind:", sig.kind.value_counts().to_dict())
    if not args.limit:
        want = {"no_match": 25143, "downward": 644, "upward": 795, "exact": 197}
        got = sig.kind.value_counts().to_dict()
        assert got == want, f"MISMATCH vs validated match summary: {got} != {want}"
        assert not (sig.reachable == False).any(), "a significant term was flagged unreachable"  # noqa: E712
        print("cross-check vs validated match summary: OK")


if __name__ == "__main__":
    main()
