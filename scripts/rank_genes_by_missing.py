#!/usr/bin/env python3
"""
Rank genes in the binarised expression dataset by number of missing
(insufficient-data) values across cell types, ascending — rank 1 is the gene
with the most complete 0/1 coverage. Useful for picking a "clean" first gene
to test an approach on, before dealing with sparse/missing-heavy genes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from core.data_loader import GeneExpressionDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, help="Path to the tsv (default: repo root)")
    parser.add_argument("--top", type=int, default=20, help="How many genes to print")
    parser.add_argument("--output", default=None, help="Optional path to write the full ranking as CSV")
    parser.add_argument(
        "--max-positive-ratio", type=float, default=None,
        help="Exclude near-ubiquitous genes: keep only genes with n_positive / n_total <= this fraction "
             "(e.g. 0.8). Without it, low-missing genes tend to be housekeeping genes expressed almost "
             "everywhere, which give a trivial positive-set (no real contrast, see PIPELINE_REQUIREMENTS.md §2.4).",
    )
    args = parser.parse_args()

    ds = GeneExpressionDataset.load(args.dataset) if args.dataset else GeneExpressionDataset.load()
    df = ds.df

    n_missing = df.isna().sum(axis=0)
    n_positive = (df == 1).sum(axis=0)
    n_negative = (df == 0).sum(axis=0)
    n_total = len(df)

    ranking = (
        n_missing.to_frame("n_missing")
        .assign(n_positive=n_positive, n_negative=n_negative, n_total=n_total,
                positive_ratio=n_positive / n_total)
        .sort_values("n_missing", kind="stable")
    )
    ranking.index.name = "gene_id"

    if args.max_positive_ratio is not None:
        before = len(ranking)
        ranking = ranking[ranking["positive_ratio"] <= args.max_positive_ratio]
        print(f"Filtered to positive_ratio <= {args.max_positive_ratio}: {len(ranking)}/{before} genes remain.\n")

    print(f"{ds.n_genes} genes, {ds.n_cell_type_pairs} cell-type pairs, dataset hash {ds.dataset_hash[:12]}\n")
    print(f"Top {args.top} genes by lowest missing-value count:\n")
    print(ranking.head(args.top).to_string())

    if args.output:
        out_path = Path(args.output)
        ranking.to_csv(out_path)
        print(f"\nFull ranking ({len(ranking)} genes) written to {out_path}")


if __name__ == "__main__":
    main()
