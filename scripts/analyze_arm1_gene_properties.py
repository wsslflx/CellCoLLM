#!/usr/bin/env python3
"""
Deep-dive: what characterizes the genes/terms where arm 1 (go_enrichment v1) actually produces a real
match, versus the ones where it doesn't? Re-walks the same full-dataset run already summarized by
scripts/summarize_go_matches.py and adds the covariates needed for that question — expression breadth,
term specificity (information content), annotation richness, term size, significance rank, which
vocabulary terms drive the hits, tissue/batch homogeneity, HGNC gene-family, and the evidence strength of
the matched truth term.

Writes two files to data/go_experiment/match_summaries/, consumed by the notebook (scripts/build_arm1_notebook.py)
rather than recomputed there:

  arm1_gene_properties__<split>.json   one row per gene (breadth, annotation richness, ceiling,
                                        batch-homogeneity, gene family, match counts)
  arm1_term_properties__<split>.json   one row per SIGNIFICANT term (not just the matches -- every
                                        outcome including no_match, so "does rank predict correctness"
                                        is answerable), with term_size, IC, rank, and (for real matches)
                                        the matched truth term's GOA evidence codes and its IC gap
                                        (ic_true - ic_predicted, signed) vs. the predicted term

Both are checked against the already-validated match_summary__...json before being written, so a
divergence in this richer extraction is caught immediately rather than silently trusted.

Usage:
    python scripts/analyze_arm1_gene_properties.py --condition go_enrichment:v1:positive --gene-split all
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import mlflow

from core.enrichment import tissue_homogeneity
from core.go_experiment import GOShared
from core.go_match import classify_term
from core.go_ontology import HGNC_PATH, load_gene_annotation_evidence
from scripts.summarize_go_matches import load_runs

OUT_DIR = Path(__file__).parents[1] / "data" / "go_experiment" / "match_summaries"
CEILING_TSV = Path(__file__).parents[1] / "data" / "go_ceiling_genes.tsv"
MATCH_SUMMARY_TEMPLATE = str(OUT_DIR / "match_summary__{safe}__{split}.json")
# Evidence-code grouping used earlier in this project's truth-policy discussion (approaches/README.md):
# measured / directly observed vs. inferred from sequence similarity, phylogeny, or author statement.
EXPERIMENTAL_EVIDENCE = frozenset({"EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "HTP", "HDA", "HMP", "HGI", "HEP"})


def load_hgnc_groups() -> dict[str, str]:
    """Ensembl id -> first listed HGNC gene_group (a gene can belong to several; the first is a reasonable
    single label for a descriptive plot, not a complete family membership)."""
    import csv
    out = {}
    with open(HGNC_PATH, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            ens, grp = row.get("ensembl_gene_id"), row.get("gene_group")
            if ens and grp:
                out[ens] = grp.split("|")[0].strip()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", default="go_enrichment:v1:positive")
    ap.add_argument("--gene-split", default="all")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--limit", type=int, default=None, help="For smoke-testing on a subset before a full run")
    args = ap.parse_args()
    version = args.condition.split(":")[1]
    safe = args.condition.replace(":", "_")

    print("Loading shared state (ontology, annotations, dataset)...")
    sh = GOShared.load(need_baselines=False)
    ceiling = {}
    with open(CEILING_TSV, newline="") as f:
        import csv
        for row in csv.DictReader(f, delimiter="\t"):
            ceiling[row["ensembl_id"]] = row
    gene_groups = load_hgnc_groups()
    print("Loading per-(gene,term) GOA evidence codes (for the matched-term evidence check)...")
    evidence_map = load_gene_annotation_evidence(sh.truth.go)

    # allow_mixed=True is deliberate, not a shortcut: this script only READS enrichment_results.json (the
    # output of a deterministic statistical test on fixed input data) and recomputes truth/IC itself from
    # the current ontology -- it never trusts a logged metric that could differ under different code. The
    # run-identity fingerprint for go_enrichment also covers core/go_ontology.py (shared with go_llm's
    # fingerprint dependencies), so an unrelated, additive change there (e.g. adding
    # load_gene_annotation_evidence for this very script) shifts the fingerprint without touching v1's
    # actual enrichment math -- requiring an exact-fingerprint match here would wrongly discard the
    # already-validated full-dataset run every time any file in that shared dependency list changes.
    runs, meta = load_runs(args.condition, args.gene_split, allow_mixed=True)
    if args.limit:
        runs = runs[: args.limit]
    print(f"  {len(runs)} genes with a COMPLETED run under the current code")

    gene_rows, term_rows = [], []
    outcome_totals = Counter()
    for gene_id, info in runs:
        if not sh.truth.has(gene_id):
            continue
        truth_direct = sh.truth.annotations[gene_id]
        enr_path = mlflow.artifacts.download_artifacts(run_id=info["run_id"], artifact_path="enrichment_results.json")
        enrichment = {r["go_id"]: r for r in json.load(open(enr_path))}
        # same significance rule as scripts/summarize_go_matches.py's significant_ids() for v1/v2 (g:SCS);
        # computed from the artifact already loaded above instead of a second, redundant download.
        sig_ids = [go_id for go_id, r in enrichment.items() if r.get("p_gscs", 1.0) < args.alpha]
        # rank by p_gscs ascending among THIS gene's significant terms, matching the ranking already
        # used to build go_predictions.json (ties broken by go_id for determinism)
        ranked = sorted(sig_ids, key=lambda g: (enrichment[g]["p_gscs"], g))

        n_exact = n_up = n_down = n_no = 0
        for rank, go_id in enumerate(ranked, start=1):
            m = classify_term(sh.truth.go, go_id, truth_direct)
            outcome_totals[m.kind] += 1
            row = {
                "gene_id": gene_id, "symbol": info["gene_symbol"], "go_id": m.go_id, "label": m.label,
                "kind": m.kind, "distance": m.distance, "rank": rank,
                "p_gscs": enrichment[go_id]["p_gscs"], "term_size": enrichment[go_id]["term_size"],
                "ic": sh.truth.go.ic(m.go_id),
            }
            if m.kind in ("exact", "upward", "downward"):
                # IC gap vs. the SPECIFIC true term that achieved the match's minimum distance
                # (m.matched_term; for "exact" the matched true term is the predicted term itself).
                # Signed as ic_true - ic_predicted: positive means the model under-shot specificity
                # (predicted a broader/cheaper term than the real answer -- expected for "upward"),
                # negative means it over-shot (predicted a narrower term than the real answer --
                # expected for "downward"). IC = -ln(fraction of genes carrying the term), so a
                # MORE SPECIFIC term has a HIGHER IC.
                true_for_ic = m.go_id if m.kind == "exact" else m.matched_term
                row["matched_term"] = true_for_ic
                row["ic_true"] = sh.truth.go.ic(true_for_ic)
                row["ic_gap"] = row["ic_true"] - row["ic"]
            if m.kind != "no_match":
                # which of the gene's OWN true terms did this prediction relate to, and how well-evidenced
                # is THAT specific (gene, true-term) pair? Upward: the true term is an ancestor-chain
                # start; downward: classify_term already found a true term among the prediction's own
                # ancestors. Re-derive which true term(s) by the same rule classify_term used.
                if m.kind == "exact":
                    matched_true = {m.go_id}
                elif m.kind == "upward":
                    matched_true = {t for t in truth_direct if sh.truth.go.ancestor_depths(t).get(m.go_id) is not None}
                else:  # downward
                    depths = sh.truth.go.ancestor_depths(m.go_id)
                    matched_true = {t for t in truth_direct if t in depths}
                codes = set()
                for t in matched_true:
                    codes |= evidence_map.get((gene_id, t), frozenset())
                row["matched_evidence_codes"] = sorted(codes)
                row["matched_evidence_experimental"] = any(c in EXPERIMENTAL_EVIDENCE for c in codes)
            if m.kind == "exact":
                n_exact += 1
            elif m.kind == "upward":
                n_up += 1
            elif m.kind == "downward":
                n_down += 1
            else:
                n_no += 1
            term_rows.append(row)

        pos = sh.ds.positive_cell_types(gene_id)
        neg = sh.ds.negative_cell_types(gene_id)
        summ = sh.ds.expression_summary(gene_id)
        th = tissue_homogeneity(pos, neg, sh.lookup)
        c = ceiling.get(gene_id, {})
        gene_rows.append({
            "gene_id": gene_id, "symbol": info["gene_symbol"],
            "n_pos": summ["n_positive"], "n_total": summ["n_total"],
            "breadth": summ["n_positive"] / summ["n_total"] if summ["n_total"] else None,
            "n_direct_terms": int(c["n_direct_terms"]) if c else None,
            "f1_top1": float(c["f1_top1"]) if c else None,
            "n_exact": n_exact, "n_upward": n_up, "n_downward": n_down, "n_no_match": n_no,
            "n_significant": len(ranked),
            "frac_allornothing_tissues": th["frac_allornothing_tissues"],
            "n_tissues_considered": th["n_tissues_considered"],
            "gene_group": gene_groups.get(gene_id),
        })

    n_genes = len(gene_rows)
    print(f"\nProcessed {n_genes} genes, {sum(outcome_totals.values())} significant terms.")

    # --- regression check against the already-validated match summary -----------------------------
    existing_path = Path(MATCH_SUMMARY_TEMPLATE.format(safe=safe, split=args.gene_split))
    if args.limit:
        print("  --limit set: skipping the full-dataset regression cross-check (expected to not match).")
    elif existing_path.exists():
        existing = json.loads(existing_path.read_text())
        want = {"exact": existing["terms"]["n_exact"], "upward": existing["terms"]["n_upward_total"],
               "downward": existing["terms"]["n_downward_total"], "no_match": existing["terms"]["n_no_match"]}
        for k, v in want.items():
            assert outcome_totals[k] == v, f"MISMATCH vs existing match summary: {k} {outcome_totals[k]} != {v}"
        assert sum(outcome_totals.values()) == existing["terms"]["n_total"]
        print(f"  cross-check vs {existing_path.name}: OK ({dict(outcome_totals)})")
    else:
        print(f"  WARNING: {existing_path} not found -- skipping the regression cross-check")

    # --- GNAT1 grounding spot-check (same gene used in the earlier manual check) ------------------
    gnat1 = next((r for r in term_rows if r["symbol"] == "GNAT1" and r["label"] == "visual perception"), None)
    if gnat1:
        print(f"  GNAT1/visual perception: kind={gnat1['kind']} evidence={gnat1.get('matched_evidence_codes')} "
              f"(expect kind=exact, codes including IMP and/or TAS)")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (out / f"arm1_gene_properties__{args.gene_split}.json").write_text(
        json.dumps({"condition": args.condition, "gene_split": args.gene_split, "generated_at": stamp,
                   "rows": gene_rows}, indent=2))
    (out / f"arm1_term_properties__{args.gene_split}.json").write_text(
        json.dumps({"condition": args.condition, "gene_split": args.gene_split, "generated_at": stamp,
                   "rows": term_rows}, indent=2))
    print(f"\nWrote {out}/arm1_gene_properties__{args.gene_split}.json ({len(gene_rows)} genes)")
    print(f"Wrote {out}/arm1_term_properties__{args.gene_split}.json ({len(term_rows)} terms)")


if __name__ == "__main__":
    main()
