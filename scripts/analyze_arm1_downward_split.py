#!/usr/bin/env python3
"""
Split arm 1's significant predictions by WHICH relations they have to the gene's direct true terms.

core.go_match.classify_term gives each predicted term ONE label and checks upward first. A prediction can
however stand in both relations to a gene's true terms at once:
  down relation  some true term is an ANCESTOR of the prediction (the prediction is more specific than it)
  up relation    some true term is a DESCENDANT of the prediction (the prediction is more general than it)
A prediction with both relations is labelled "upward" by classify_term regardless of which is closer. This script
makes the split explicit so the "downward" matches can be looked at on their own:

  only_down                      down relation only            (this is exactly classify_term's "downward")
  both_down_closer               both relations, d_down <  d_up  (labelled upward by classify_term)
  both_up_closer_or_equal        both relations, d_down >= d_up  (labelled upward by classify_term)

For the down relation it records the true ancestor with the smallest distance (d_down) and its IC, so the IC
distance (ic_pred - ic_true_down, positive = prediction more specific) can be plotted.

Reads the arm 1 cache (scripts/export_arm1_cache.py). Writes
data/go_experiment/match_summaries/arm1_downward_split__<split>.json (one row per significant prediction that has a down relation, plus `all_true_terms`: every distinct direct true term with its IC).

Usage:  python scripts/analyze_arm1_downward_split.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from core.arm1_cache import SUMMARY_DIR, load_arm1_cache
from core.go_experiment import GOShared


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gene-split", default="all")
    args = ap.parse_args()

    sh = GOShared.load(need_baselines=False)
    go = sh.truth.go
    sig = load_arm1_cache(args.gene_split).sig

    rows, counts = [], Counter()
    for r in sig.itertuples():
        td = sh.truth.annotations[r.gene_id]
        pid = r.go_id
        if pid in td:
            counts["exact"] += 1
            continue
        d_up = min((go.ancestor_depths(t).get(pid) for t in td if go.ancestor_depths(t).get(pid) is not None), default=None)
        pred_depths = go.ancestor_depths(pid)
        down = [(pred_depths[t], t) for t in td if t in pred_depths]
        d_down, t_down = min(down) if down else (None, None)
        if d_down is None:
            counts["up_only" if d_up is not None else "no_match"] += 1
            continue
        group = ("only_down" if d_up is None else
                 "both_down_closer" if d_down < d_up else "both_up_closer_or_equal")
        counts[group] += 1
        rows.append({"gene_id": r.gene_id, "symbol": r.symbol, "go_id": pid, "label": r.label, "group": group,
                     "classify_term_kind": r.kind, "d_down": d_down, "d_up": d_up, "true_term": t_down,
                     "true_label": go.label(t_down), "ic_pred": go.ic(pid), "ic_true": go.ic(t_down),
                     "ic_distance": go.ic(pid) - go.ic(t_down), "n_true_terms_that_are_ancestors": len(down),
                     "all_ancestor_true_terms": sorted(t for _, t in down)})

    # checks against the already-validated classification
    assert counts["only_down"] == 644, counts
    assert all(x["classify_term_kind"] == "downward" for x in rows if x["group"] == "only_down")
    assert all(x["classify_term_kind"] == "upward" for x in rows if x["group"] != "only_down")
    # Reference for "are the true terms these matches attach to unusually general?": every distinct direct true
    # term in the gene annotations of the genes analysed, with its IC and how many of those genes carry it directly.
    genes = set(sig["gene_id"]) | set(load_arm1_cache(args.gene_split).genes["gene_id"])
    carriers = Counter(t for g in genes if sh.truth.has(g) for t in sh.truth.annotations[g])
    all_true_terms = [{"go_id": t, "label": go.label(t), "ic": go.ic(t), "n_genes_direct": n} for t, n in carriers.items()]

    out = SUMMARY_DIR / f"arm1_downward_split__{args.gene_split}.json"
    out.write_text(json.dumps({"counts": dict(counts), "rows": rows, "all_true_terms": all_true_terms}))
    print(dict(counts))
    print(f"Wrote {out} ({len(rows)} predictions with a down relation)")


if __name__ == "__main__":
    main()
