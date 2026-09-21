#!/usr/bin/env python3
"""
Choose and split the genes for the GO-prediction experiment (go_enrichment v1-v4 vs go_llm).

The RULES are fixed here and written into the output so the selection is reproducible and
cannot be tuned after seeing results; the THRESHOLDS below are defaults to confirm before
the real run. Selection uses truth-derived criteria (n_direct_terms, room, stratum), which
is disclosed: results generalise to the selected stratum, not to all genes.

Eligibility (all must hold):
  scoreable       >=1 non-IEA biological-process annotation (data/go_ceiling_genes.tsv)
  annotated       n_direct_terms >= --min-direct-terms. Excludes sparsely annotated genes
                  where one matching term gives F1 = 1.0 trivially
  signal can exist  the gene's positive rate at the cell-type unit lies in
                  [--rate-min, --rate-max]. Uses expression data only, never truth
  room            f1_top1 - f1_top1_random >= the --room-quantile quantile of the genes
                  that passed the checks above (the ceiling has to be worth reaching)

Strata:  carrying  = the gene's PROPAGATED annotation contains >=1 candidate GO term.
                     This is the primary stratum.
         not_carrying = does not; a small diagnostic stratum (--n-diag), always in `test`.

Split:  deterministic by sha256(salt|gene). `dev` is for prompt/top-N/ranking decisions
        only; `test` is touched once, after the pipeline is frozen. The three standard
        genes (MMACHC, TTI2, CHEK1) are `smoke`, never dev or test.

Needs data/go_ceiling_genes.tsv:  python scripts/analyze_go_ceiling.py --tsv data/go_ceiling_genes.tsv

Usage:
    python scripts/select_go_genes.py --n-dev 40 --n-test 120
    python scripts/select_go_genes.py --n-dev 40 --n-test 120 --out /tmp/genes.tsv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from core.go_evidence import UNIT_CELL_TYPES
from core.go_experiment import GOShared

CEILING_TSV = Path(__file__).parents[1] / "data" / "go_ceiling_genes.tsv"
DEFAULT_OUT = Path(__file__).parents[1] / "data" / "go_experiment" / "genes.tsv"
SMOKE_GENES = ["ENSG00000132763", "ENSG00000129696", "ENSG00000149554"]


def order_key(salt: str, gene: str) -> str:
    return hashlib.sha256(f"{salt}|{gene}".encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-dev", type=int, default=40, help="Development genes (carrying stratum)")
    ap.add_argument("--n-test", type=int, default=120, help="Test genes (carrying stratum)")
    ap.add_argument("--n-diag", type=int, default=30, help="Diagnostic not-carrying genes (all in the test split)")
    ap.add_argument("--min-direct-terms", type=int, default=3)
    ap.add_argument("--rate-min", type=float, default=0.05)
    ap.add_argument("--rate-max", type=float, default=0.60)
    ap.add_argument("--room-quantile", type=float, default=0.25)
    ap.add_argument("--salt", default="cellcollm-go-v1")
    ap.add_argument("--ceiling-tsv", default=str(CEILING_TSV))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    if not Path(args.ceiling_tsv).exists():
        raise SystemExit(f"{args.ceiling_tsv} not found. Run: python scripts/analyze_go_ceiling.py --tsv {args.ceiling_tsv}")
    ceil = {}
    with open(args.ceiling_tsv, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            ceil[row["ensembl_id"]] = row

    print("Loading dataset, ontology, annotations...")
    sh = GOShared.load(need_baselines=False)
    rates = sh.gene_rates(UNIT_CELL_TYPES)
    col = sh.builder.gene_col

    rows, reasons = {}, Counter()
    survivors = []
    for g in sh.builder.genes:
        info = {"gene": g, "symbol": sh.symbol(g), "split": "", "stratum": "", "n_direct_terms": "",
                "room_top1": "", "gene_rate": f"{rates[col[g]]:.4f}" if np.isfinite(rates[col[g]]) else ""}
        if g in SMOKE_GENES:
            info.update(split="smoke", stratum=sh.stratum(g))
        elif g not in ceil:
            info["split"] = "excluded:not_scoreable"
        else:
            c = ceil[g]
            n_direct, room = int(c["n_direct_terms"]), float(c["f1_top1"]) - float(c["f1_top1_random"])
            info.update(n_direct_terms=n_direct, room_top1=f"{room:.4f}")
            if n_direct < args.min_direct_terms:
                info["split"] = "excluded:sparse_annotation"
            elif not (np.isfinite(rates[col[g]]) and args.rate_min <= rates[col[g]] <= args.rate_max):
                info["split"] = "excluded:rate_out_of_range"
            else:
                survivors.append((g, room))
        rows[g] = info

    cutoff = float(np.quantile([r for _, r in survivors], args.room_quantile)) if survivors else 0.0
    eligible = []
    for g, room in survivors:
        if room < cutoff:
            rows[g]["split"] = "excluded:low_room"
        else:
            rows[g]["stratum"] = sh.stratum(g)
            eligible.append(g)

    carrying = sorted((g for g in eligible if rows[g]["stratum"] == "carrying"), key=lambda g: order_key(args.salt, g))
    diag = sorted((g for g in eligible if rows[g]["stratum"] == "not_carrying"), key=lambda g: order_key(args.salt, g))
    for i, g in enumerate(carrying):
        rows[g]["split"] = "dev" if i < args.n_dev else "test" if i < args.n_dev + args.n_test else "held"
    for i, g in enumerate(diag):
        rows[g]["split"] = "test" if i < args.n_diag else "held"

    for r in rows.values():
        reasons[r["split"]] += 1
    print(f"\nCriteria: n_direct_terms>={args.min_direct_terms}, positive rate in [{args.rate_min}, {args.rate_max}], "
          f"room>=q{args.room_quantile:.2f} of survivors (= {cutoff:.4f}), salt={args.salt!r}")
    print(f"eligible: {len(eligible)}  (carrying {len(carrying)}, not_carrying {len(diag)})")
    for k, v in sorted(reasons.items()):
        print(f"  {k:32s}{v:7d}")
    short = args.n_dev + args.n_test - len(carrying)
    if short > 0:
        print(f"\nWARNING: only {len(carrying)} carrying genes are eligible; --n-dev + --n-test asks for "
              f"{args.n_dev + args.n_test}. Loosen a criterion or lower the sizes.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["gene", "symbol", "split", "stratum", "n_direct_terms", "room_top1", "gene_rate"],
                           delimiter="\t")
        w.writeheader()
        w.writerows(rows.values())
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
