#!/usr/bin/env python3
"""
How well are the CELL TYPES in the dataset annotated with GO biological processes?

This is purely about cell-type annotation — no genes, no expression values are
involved. It answers: how many of the dataset's cell types carry GO process
terms (via CL's capable_of / capable_of_part_of relations), how many terms each
carries, and how large the distinct GO vocabulary is.

The direct-vs-inherited distinction matters a lot. CL asserts capable_of on
general terms and lets subtypes inherit it, so direct annotation looks far
sparser than the usable coverage. Both are reported, plus the curve across
inheritance depths.

Reads only data/ontology_cache.json (built by scripts/build_ontology_cache.py
with capable_of extraction). No network, no LLM.

Usage:
    python scripts/analyze_go_coverage.py
    python scripts/analyze_go_coverage.py --depth 3 --json out.json
"""
from __future__ import annotations

import argparse
import json
import statistics as stats
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from core.data_loader import GeneExpressionDataset, parse_pair
from core.ontology_lookup import OntologyLookup

DEPTH_CURVE = (0, 1, 2, 3, 4, 6, 8)


def describe(counts: list[int], label: str) -> dict:
    """Summary stats over per-cell-type GO term counts."""
    nonzero = [c for c in counts if c > 0]
    return {
        "scope": label,
        "n_cell_types": len(counts),
        "n_with_any_go": len(nonzero),
        "frac_with_any_go": len(nonzero) / len(counts) if counts else 0.0,
        "mean_all": stats.mean(counts) if counts else 0.0,
        "mean_annotated_only": stats.mean(nonzero) if nonzero else 0.0,
        "median_annotated_only": stats.median(nonzero) if nonzero else 0.0,
        "min_all": min(counts) if counts else 0,
        "min_annotated_only": min(nonzero) if nonzero else 0,
        "max": max(counts) if counts else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--depth", type=int, default=3,
                         help="is_a depth over which capable_of is inherited for the headline figures (default 3)")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--json", default=None, help="Optional path to write the full report as JSON")
    parser.add_argument("--top", type=int, default=10, help="How many extremes/most-common to list")
    args = parser.parse_args()

    ds = GeneExpressionDataset.load(args.dataset) if args.dataset else GeneExpressionDataset.load()
    lookup = OntologyLookup()
    if "go" not in lookup._cache:
        raise SystemExit(
            "The ontology cache has no 'go' section — rebuild it with capable_of extraction:\n"
            "    python scripts/build_ontology_cache.py"
        )

    rows = list(ds.df.index)
    cell_types = sorted({parse_pair(p)[0] for p in rows})
    row_cl = [parse_pair(p)[0] for p in rows]

    print(f"Dataset: {ds.path.name}")
    print(f"  rows (cell type x tissue pairs): {len(rows)}")
    print(f"  distinct cell types (CL terms):  {len(cell_types)}")
    print(f"  ontology: CL {lookup.provenance.get('cl_data_version')}, "
          f"{lookup.provenance.get('go_terms_extracted')} GO terms extracted")
    print()

    # --- headline: direct vs inherited at the chosen depth ---
    report = {"dataset": ds.path.name, "dataset_hash": ds.dataset_hash,
              "n_rows": len(rows), "n_cell_types": len(cell_types),
              "cl_data_version": lookup.provenance.get("cl_data_version"), "scopes": {}}

    for label, depth in (("direct only", 0), (f"inherited (is_a depth {args.depth})", args.depth)):
        per_type = {c: lookup.processes(c, depth) for c in cell_types}
        counts = [len(v) for v in per_type.values()]
        summary = describe(counts, label)
        uniq = {t.id for v in per_type.values() for t in v}
        summary["n_unique_go_terms"] = len(uniq)
        covered = {c for c, v in per_type.items() if v}
        summary["frac_rows_covered"] = sum(1 for c in row_cl if c in covered) / len(row_cl)
        report["scopes"][label] = summary

        print(f"=== {label} ===")
        print(f"  cell types with >=1 GO term : {summary['n_with_any_go']}/{summary['n_cell_types']} "
              f"({summary['frac_with_any_go']:.1%})")
        print(f"  data rows covered           : {summary['frac_rows_covered']:.1%}")
        print(f"  unique GO terms in use      : {summary['n_unique_go_terms']}")
        print(f"  GO terms per cell type      : mean {summary['mean_annotated_only']:.1f}, "
              f"median {summary['median_annotated_only']:.0f} (annotated cell types only)")
        print(f"                                mean {summary['mean_all']:.1f} across ALL cell types "
              f"(counting un-annotated as 0)")
        print(f"  floor / ceiling             : {summary['min_annotated_only']} / {summary['max']} "
              f"(floor among annotated; {summary['min_all']} counting un-annotated)")
        print()

    # --- depth curve: how coverage and dilution trade off ---
    print("=== coverage vs inheritance depth ===")
    print(f"  {'depth':>5s} {'cell types':>11s} {'rows':>7s} {'unique GO':>10s} {'median/type':>12s}  most-inherited term")
    curve = []
    for d in DEPTH_CURVE:
        per_type = {c: lookup.processes(c, d) for c in cell_types}
        covered = {c for c, v in per_type.items() if v}
        uniq = {t.id for v in per_type.values() for t in v}
        nonzero = sorted(len(v) for v in per_type.values() if v)
        freq = Counter(t.id for v in per_type.values() for t in v)
        top_id, top_n = freq.most_common(1)[0] if freq else (None, 0)
        top_label = lookup.resolve_go(top_id).label if top_id else "-"
        row = {"depth": d, "frac_cell_types": len(covered) / len(cell_types),
               "frac_rows": sum(1 for c in row_cl if c in covered) / len(row_cl),
               "n_unique_go": len(uniq), "median_per_annotated": nonzero[len(nonzero) // 2] if nonzero else 0,
               "most_common_term": top_label, "most_common_n_cell_types": top_n}
        curve.append(row)
        print(f"  {d:5d} {row['frac_cell_types']:10.0%} {row['frac_rows']:7.0%} {row['n_unique_go']:10d} "
              f"{row['median_per_annotated']:12d}  {str(top_label)[:34]} ({top_n})")
    report["depth_curve"] = curve
    print()

    # --- vocabulary and extremes at the chosen depth ---
    per_type = {c: lookup.processes(c, args.depth) for c in cell_types}
    freq = Counter(t.id for v in per_type.values() for t in v)
    print(f"=== most widely assigned GO terms (depth {args.depth}) ===")
    for go_id, n in freq.most_common(args.top):
        print(f"  {n:4d} cell types  {go_id}  {lookup.resolve_go(go_id).label}")
    print()
    print(f"=== GO terms assigned to exactly one cell type: "
          f"{sum(1 for n in freq.values() if n == 1)} of {len(freq)} ===")
    print()

    ranked = sorted(per_type.items(), key=lambda kv: -len(kv[1]))
    print(f"=== cell types with the MOST GO terms (depth {args.depth}) ===")
    for cl_id, terms in ranked[: args.top]:
        print(f"  {len(terms):3d}  {lookup.resolve_cl(cl_id).label or cl_id}")
    n_zero = sum(1 for v in per_type.values() if not v)
    print(f"\n=== cell types with NO GO terms: {n_zero} ({n_zero / len(cell_types):.0%}) ===")
    for cl_id, terms in ranked[-args.top:]:
        if not terms:
            print(f"    {lookup.resolve_cl(cl_id).label or cl_id}")

    report["most_common_terms"] = [
        {"go_id": go_id, "label": lookup.resolve_go(go_id).label, "n_cell_types": n}
        for go_id, n in freq.most_common(args.top)
    ]
    report["n_go_terms_on_single_cell_type"] = sum(1 for n in freq.values() if n == 1)

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
