#!/usr/bin/env python3
"""
Oracle ceiling: how well could ANY method score, if it could only name terms from the
cell-type GO vocabulary?

The planned verification compares GO terms predicted from cell-type evidence against
the gene's own GO annotation. The cell-type vocabulary is small (126 terms) and sits
at cell-behaviour granularity, while genes are annotated to specific molecular
processes. Both sides are propagated up the GO DAG and weighted by information
content (IC), so a match on `biological_process` counts for nothing. This script asks,
per gene, how much headroom exists BEFORE any LLM or statistics are involved:

  reachable_ic_fraction   IC mass of the gene's propagated terms that the vocabulary's
                          ancestors can reach / the gene's total IC mass. A recall
                          ceiling for a predictor that names the whole vocabulary
                          (precision-blind).
  max_shared_ic           IC of the most specific term shared between the gene and the
                          vocabulary: whether an informative meeting point exists.
  f1_top1                 best IC-weighted F1 achievable by naming ONE vocabulary term
                          (with its ancestors). Includes precision.
  f1_top3                 same, greedy union of up to 3 terms.
  f1_top1_random          the FLOOR: expected F1 of naming one vocabulary term at random.
                          f1_top1 - f1_top1_random is the room a method has to show skill.

A gene whose ceiling is near zero cannot be scored by any method in this vocabulary and
should be dropped from a comparison rather than counted as a loss.

As a reference, the same ceilings are computed for random vocabularies of the same
size whose terms are IC-matched to the real one. The cell-type vocabulary only counts
as informative if it beats those.

No LLM, no network, no MLflow. Reads data/ontologies/raw/{go-basic.obo,
goa_human.gaf.gz, hgnc_complete_set.txt} and data/ontology_cache.json.

Usage:
    python scripts/analyze_go_ceiling.py
    python scripts/analyze_go_ceiling.py --include-iea --json out.json --tsv genes.tsv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from core.data_loader import parse_pair, read_gene_ids
from core.go_ontology import (
    BP_ROOT, HGNC_PATH, GOOntology, load_gene_annotations,
)
from core.go_scoring import CeilingModel
from core.ontology_lookup import OntologyLookup

DEFAULT_TEST_GENES = {"ENSG00000132763": "MMACHC", "ENSG00000129696": "TTI2",
                      "ENSG00000149554": "CHEK1", "ENSG00000012048": "BRCA2"}
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


def cell_type_vocabulary(go: GOOntology, lookup: OntologyLookup, dataset_path, depth: int,
                         min_term_size: int) -> tuple[dict[str, int], dict]:
    """GO term -> number of distinct annotated cell types carrying it (term size)."""
    from core.data_loader import GeneExpressionDataset
    ds = GeneExpressionDataset.load(dataset_path) if dataset_path else GeneExpressionDataset.load()
    cell_types = sorted({parse_pair(p)[0] for p in ds.df.index})
    size: Counter[str] = Counter()
    n_annotated = 0
    unresolved = set()
    for cl in cell_types:
        terms = {t.id for t in lookup.processes(cl, depth)}
        if not terms:
            continue
        n_annotated += 1
        for t in terms:
            pid = go.resolve(t)
            if pid and go.is_bp(pid):
                size[pid] += 1
            else:
                unresolved.add(t)
    info = {"n_cell_types": len(cell_types), "n_annotated_cell_types": n_annotated,
            "unresolved_or_non_bp": sorted(unresolved), "min_term_size": min_term_size}
    return dict(size), info


def random_ic_matched_vocab(go, ic_pool: dict[str, float], real: list[str], rng) -> list[str]:
    """One random term per real term, drawn from annotated BP terms of similar IC."""
    ids = np.array(list(ic_pool))
    ics = np.array([ic_pool[i] for i in ids])
    chosen: set[str] = set()
    for v in real:
        near = ids[np.abs(ics - go.ic(v)) <= 0.25]
        near = [t for t in near if t not in chosen and t != BP_ROOT]
        if near:
            chosen.add(str(rng.choice(near)))
    return sorted(chosen)


def summarize(values: list[float]) -> dict:
    a = np.array(values)
    d = {f"q{int(q * 100)}": float(np.quantile(a, q)) for q in QUANTILES}
    d.update({"mean": float(a.mean()), "frac_gt_0": float((a > 0).mean())})
    return d


def load_symbols() -> dict[str, str]:
    out = {}
    with open(HGNC_PATH, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("ensembl_gene_id"):
                out[row["ensembl_gene_id"]] = row["symbol"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--depth", type=int, default=3, help="is_a depth for cell-type capable_of inheritance")
    ap.add_argument("--min-term-size", type=int, default=3, help="'testable' vocabulary = terms on >= this many cell types")
    ap.add_argument("--include-iea", action="store_true", help="Include IEA-evidence gene annotations (excluded by default)")
    ap.add_argument("--null-runs", type=int, default=20, help="Random IC-matched vocabularies for the reference (0 = skip)")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--tsv", default=None, help="Per-gene table for later gene selection")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    go = GOOntology.load()
    excluded = frozenset({"ND"}) if args.include_iea else frozenset({"IEA", "ND"})
    ann, stats = load_gene_annotations(go, excluded_evidence=excluded)
    go.fit_information_content(ann)
    lookup = OntologyLookup()

    dataset_genes = read_gene_ids(args.dataset) if args.dataset else read_gene_ids()
    genes = [g for g in dataset_genes if g in ann]
    print(f"GO {go.data_version}; evidence excluded: {sorted(excluded)}")
    print(f"dataset genes: {len(dataset_genes)}; with >=1 propagated BP annotation: {len(genes)} "
          f"({len(genes) / len(dataset_genes):.1%}); {len(dataset_genes) - len(genes)} have none and are unscoreable")

    size, vinfo = cell_type_vocabulary(go, lookup, args.dataset, args.depth, args.min_term_size)
    vocabs = {"all": sorted(size), f"testable(size>={args.min_term_size})":
              sorted(t for t, n in size.items() if n >= args.min_term_size)}
    print(f"cell-type vocabulary (depth {args.depth}): {len(vocabs['all'])} GO terms; "
          f"{len(vocabs[list(vocabs)[1]])} testable; "
          f"{len(vinfo['unresolved_or_non_bp'])} unresolved/non-BP in go-basic")

    truth = {g: go.propagate(ann[g]) for g in genes}
    tmass = {g: sum(go.ic(t) for t in truth[g]) for g in genes}
    symbols = load_symbols()
    report = {"go_version": go.data_version, "excluded_evidence": sorted(excluded), "gaf_stats": stats,
              "n_dataset_genes": len(dataset_genes), "n_scoreable_genes": len(genes), "vocabularies": {}}
    per_gene_rows: dict[str, dict] = {}

    ic_pool = {t: go.ic(t) for t in go._g.nodes if go.is_bp(t) and t != BP_ROOT and go._counts.get(t, 0) >= 3}
    rng = np.random.default_rng(args.seed)

    for name, vocab in vocabs.items():
        model = CeilingModel(go, vocab)
        rows = {g: model.gene(truth[g], tmass[g]) for g in genes}
        if name == list(vocabs)[1]:
            per_gene_rows = rows
        entry = {"n_terms": len(vocab)}
        print(f"\n=== vocabulary: {name}  ({len(vocab)} terms) ===")
        print(f"  {'metric':24s}" + "".join(f"{f'q{int(q*100)}':>8s}" for q in QUANTILES) + f"{'mean':>8s}{'>0':>8s}")
        for m in ("reachable_ic_fraction", "max_shared_ic", "f1_top1", "f1_top1_random", "f1_top3"):
            s = summarize([r[m] for r in rows.values()])
            entry[m] = s
            print(f"  {m:24s}" + "".join(f"{s[f'q{int(q*100)}']:8.3f}" for q in QUANTILES)
                  + f"{s['mean']:8.3f}{s['frac_gt_0']:8.0%}")
        room = [r["f1_top1"] - r["f1_top1_random"] for r in rows.values()]
        entry["room_top1"] = summarize(room)
        print(f"  {'room (top1 - random)':24s}" + "".join(f"{entry['room_top1'][f'q{int(q*100)}']:8.3f}" for q in QUANTILES)
              + f"{entry['room_top1']['mean']:8.3f}")
        # An interpretable notion of 'informative meeting point': the most specific shared
        # term is carried by at most 5% (IC >= 3.0) / 1% (IC >= 4.6) of annotated genes.
        # 'reachable > 0' is useless as a criterion: every gene shares generic terms.
        for label, cut in (("<=5% of genes carry it (IC>=3.0)", -np.log(0.05)), ("<=1% of genes carry it (IC>=4.6)", -np.log(0.01))):
            frac = float(np.mean([r["max_shared_ic"] >= cut for r in rows.values()]))
            entry[f"frac_max_shared_ic_ge_{cut:.1f}"] = frac
            print(f"  share a term that {label}: {frac:.0%} of scoreable genes")
        # genes carrying a vocabulary term in their propagated set (true-path)
        vset = set(vocab)
        carrying = [g for g in genes if truth[g] & vset]
        entry["n_genes_carrying_a_vocab_term"] = len(carrying)
        print(f"  genes whose propagated annotation contains >=1 vocabulary term: {len(carrying)} "
              f"({len(carrying) / len(genes):.0%} of scoreable)")
        # The ceiling is not uniform: the vocabulary is concentrated in one region of the DAG,
        # so it should be far better for genes annotated in that region than for the rest.
        carry_set = set(carrying)
        for label, members in (("carry a vocabulary term", carry_set), ("do not", set(genes) - carry_set)):
            vals = [rows[g]["f1_top1"] for g in members]
            ics = [rows[g]["max_shared_ic"] for g in members]
            entry[f"median_f1_top1_{'carrying' if members is carry_set else 'not_carrying'}"] = float(np.median(vals))
            print(f"    genes that {label:24s} n={len(members):6d}  median f1_top1={np.median(vals):.3f}  "
                  f"median max_shared_ic={np.median(ics):.2f}")

        if args.null_runs:
            null = {m: [] for m in ("reachable_ic_fraction", "max_shared_ic", "f1_top1", "f1_top3")}
            for _ in range(args.null_runs):
                nm = CeilingModel(go, random_ic_matched_vocab(go, ic_pool, vocab, rng))
                r = [nm.gene(truth[g], tmass[g]) for g in genes]
                for m in null:
                    null[m].append(float(np.median([x[m] for x in r])))
            entry["null_median_of_medians"] = {m: float(np.mean(v)) for m, v in null.items()}
            entry["null_runs"] = args.null_runs
            print(f"  reference: random IC-matched vocabularies ({args.null_runs} draws), mean of per-draw medians:")
            print("    " + "   ".join(f"{m}={np.mean(v):.3f}" for m, v in null.items()))
        report["vocabularies"][name] = entry

    named = [g for g in DEFAULT_TEST_GENES if g in per_gene_rows]
    print(f"\n=== named genes (testable vocabulary) ===")
    print(f"  {'gene':9s}{'reach':>8s}{'max_ic':>8s}{'f1_top1':>9s}{'random':>8s}{'f1_top3':>9s}  #direct terms")
    for g in DEFAULT_TEST_GENES:
        if g in per_gene_rows:
            r = per_gene_rows[g]
            print(f"  {DEFAULT_TEST_GENES[g]:9s}{r['reachable_ic_fraction']:8.3f}{r['max_shared_ic']:8.2f}"
                  f"{r['f1_top1']:9.3f}{r['f1_top1_random']:8.3f}{r['f1_top3']:9.3f}  {len(ann[g])}")
        else:
            print(f"  {DEFAULT_TEST_GENES[g]:9s} not scoreable (no non-IEA BP annotation, or not in dataset)")
    top = sorted(per_gene_rows.items(), key=lambda kv: -kv[1]["f1_top3"])[:10]
    print("\n  highest f1_top3 ceilings — NOTE: dominated by genes with only 1-2 direct annotations, where a")
    print("  single matching term gives F1 = 1.0 trivially. Not a selection list on its own; see n_direct_terms:")
    for g, r in top:
        print(f"    {symbols.get(g, g):10s} f1_top3={r['f1_top3']:.3f}  f1_top1={r['f1_top1']:.3f}  "
              f"max_ic={r['max_shared_ic']:.2f}  n_direct_terms={len(ann[g])}")

    if args.tsv:
        with open(args.tsv, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["ensembl_id", "symbol", "n_direct_terms", "reachable_ic_fraction", "max_shared_ic", "f1_top1", "f1_top1_random", "f1_top3"])
            for g, r in per_gene_rows.items():
                w.writerow([g, symbols.get(g, ""), len(ann[g]), *(f"{r[k]:.6f}" for k in (
                    "reachable_ic_fraction", "max_shared_ic", "f1_top1", "f1_top1_random", "f1_top3"))])
        print(f"\nWrote {args.tsv}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
