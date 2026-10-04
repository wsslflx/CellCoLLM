#!/usr/bin/env python3
"""
Chance baseline for arm 1's matches: how many matches would a RANDOM pick from the 52 candidate GO terms
have produced, given each gene's own true terms?

Why: a "downward" match only needs the candidate to be ANY descendant of one of the gene's true terms.
A gene whose true terms are all very general (low IC) has many such descendants among the 52 candidates,
so almost any guess matches. This script quantifies that per gene:

  chance          fraction of the 52 candidates that would count as a match (exact/upward/downward)
  chance_<kind>   the same, split by match kind
  mean_true_ic    mean information content of the gene's direct true terms (low = general terms only)

Expected matches for a gene that produced n_significant predictions under random picking
= n_significant * chance (expectation of a draw without replacement). Compared against the observed
counts in notebooks/arm1_go_enrichment_results.ipynb (Plot 16).

Usage:  python scripts/analyze_arm1_chance_baseline.py
Reads:  data/go_experiment/match_summaries/arm1_gene_properties__all.json
Writes: data/go_experiment/match_summaries/arm1_chance_baseline__all.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd

from core.go_experiment import GOShared
from core.go_match import classify_term

DIR = Path(__file__).parents[1] / "data" / "go_experiment" / "match_summaries"


def main() -> None:
    sh = GOShared.load(need_baselines=False)
    go, cands = sh.truth.go, list(sh.builder.candidates)
    gp = pd.DataFrame(json.loads((DIR / "arm1_gene_properties__all.json").read_text())["rows"])
    rows = []
    for g in gp.gene_id:
        if not sh.truth.has(g):
            continue
        td = sh.truth.annotations[g]
        kinds = [classify_term(go, c, td).kind for c in cands]
        ics = [go.ic(t) for t in td]
        rows.append(dict(
            gene_id=g, chance=sum(k != "no_match" for k in kinds) / len(cands),
            chance_exact=kinds.count("exact") / len(cands), chance_upward=kinds.count("upward") / len(cands),
            chance_downward=kinds.count("downward") / len(cands),
            mean_true_ic=float(np.mean(ics)), max_true_ic=max(ics), min_true_ic=min(ics)))
    m = gp.merge(pd.DataFrame(rows), on="gene_id")
    m.to_json(DIR / "arm1_chance_baseline__all.json", orient="records")
    w = m[m.n_significant > 0]
    for k in ("exact", "upward", "downward"):
        print(f"{k:9s} observed {w['n_' + k].sum():5d}   expected by chance {(w.n_significant * w['chance_' + k]).sum():7.1f}")
    print(f"Wrote {DIR / 'arm1_chance_baseline__all.json'} ({len(m)} genes)")


if __name__ == "__main__":
    main()
