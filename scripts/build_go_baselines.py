#!/usr/bin/env python3
"""
Per-GO-term baseline positive-rates over EVERY gene, at both units of analysis.

Why: the corrected statistic (core/go_evidence.corrected_stats, used by go_enrichment
v3/v4 and go_llm v3/v4) tests each GO term against what would be expected for THIS
gene given how readily that term is called "expressed" in general:

    expected = gene_rate x baseline_rate(term) / grand_mean

Same estimator as scripts/build_feature_baselines.py — per gene, positive items among
called carriers of the term; then the mean across genes with >=100 contributing genes —
but restricted to GO terms and computed at BOTH units:
  cell_types  a cell type is positive if any of its rows is; called if any row is
  rows        each CL|UBERON pair once
over annotated items only (see core/go_evidence.py). `grand_mean` is the mean over genes
of the gene's positive rate in that same universe.

Writes data/go_baselines.json keyed on dataset hash, CL version, go depth and minimum
term size; the consumers refuse a mismatch rather than silently using stale baselines.

Usage:
    python scripts/build_go_baselines.py
    python scripts/build_go_baselines.py --go-depth 3 --min-term-size 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from core.data_loader import GeneExpressionDataset
from core.go_evidence import (
    DEFAULT_GO_DEPTH, DEFAULT_MIN_TERM_SIZE, GO_BASELINES_PATH, UNITS,
    GOEvidenceBuilder, baselines_key,
)
from core.go_ontology import GOOntology
from core.ontology_lookup import OntologyLookup

MIN_GENES_CONTRIBUTING = 100


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--go-depth", type=int, default=DEFAULT_GO_DEPTH)
    ap.add_argument("--min-term-size", type=int, default=DEFAULT_MIN_TERM_SIZE)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--out", default=str(GO_BASELINES_PATH))
    args = ap.parse_args()

    print("Loading dataset, ontologies...")
    ds = GeneExpressionDataset.load(args.dataset) if args.dataset else GeneExpressionDataset.load()
    lookup, go = OntologyLookup(), GOOntology.load()
    b = GOEvidenceBuilder(ds, lookup, go, args.go_depth, args.min_term_size)
    print(f"  {len(b.ann_cl)} annotated cell types, {len(b.ann_row_index)} annotated rows, "
          f"{len(b.candidates)} candidate GO terms (carried by >={args.min_term_size} cell types)")

    units = {}
    for unit in UNITS:
        M, pos, called = b.unit_arrays(unit)
        t0 = time.time()
        P, C = pos.astype(np.float32), called.astype(np.float32)
        pos_counts, called_counts = M.T @ P, M.T @ C          # terms x genes
        with np.errstate(invalid="ignore", divide="ignore"):
            rates = np.where(called_counts > 0, pos_counts / called_counts, np.nan)
            baseline = np.nanmean(rates, axis=1)
            gene_rate = np.where(C.sum(0) > 0, P.sum(0) / C.sum(0), np.nan)
        contributing = (called_counts > 0).sum(axis=1)
        grand = float(np.nanmean(gene_rate))
        terms, skipped = {}, 0
        for i, g in enumerate(b.candidates):
            if contributing[i] < MIN_GENES_CONTRIBUTING or not np.isfinite(baseline[i]):
                skipped += 1
                continue
            terms[g] = round(float(baseline[i]), 6)
        vals = list(terms.values())
        print(f"  [{unit}] {len(terms)} baselines ({skipped} skipped), grand mean {grand:.1%}, "
              f"spread {min(vals):.0%}..{max(vals):.0%}  ({time.time() - t0:.1f}s)")
        units[unit] = {"grand_mean": round(grand, 6), "terms": terms}

    payload = {
        "units": units,
        "provenance": {
            "baselines_key": baselines_key(b),
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset_hash": ds.dataset_hash,
            "cl_data_version": b.cl_data_version,
            "go_depth": args.go_depth,
            "min_term_size": args.min_term_size,
            "min_genes_contributing": MIN_GENES_CONTRIBUTING,
            "n_genes": len(b.genes),
            "n_candidates": len(b.candidates),
        },
    }
    Path(args.out).write_text(json.dumps(payload, sort_keys=True))
    print(f"Wrote {args.out}  (key {payload['provenance']['baselines_key']})")


if __name__ == "__main__":
    main()
