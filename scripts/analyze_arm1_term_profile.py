#!/usr/bin/env python3
"""
Term-level profile of arm 1 (go_enrichment v1): what do the frequently assigned GO terms look like?

The other arm-1 analyses are per GENE. This one takes the 52 candidate GO terms as the unit and asks, per term:

  frequency in the cell types   how many annotated cell types (and CL|UBERON rows) carry the term
  precision vs. base rate       of the genes the term was assigned to, how often it is a real match, versus
                                the share of ALL genes for which this term would be a match
  tissue purity                 how concentrated the term's carrier rows are in one tissue
  reachability                  for how many genes the term can reach significance AT ALL, given its size K,
                                the gene's query size n and background N (the smallest p-value the hypergeometric
                                test can produce is when all min(K, n) possible overlaps are hit)
  gene profile                  breadth and HGNC gene families of the genes the term was assigned to

and, over term PAIRS, how redundant the terms are (shared assigned genes / shared carrier cell types / GO branch).

IMPORTANT on K: `term_size` in enrichment_results.json is the number of carriers inside THAT GENE's called
universe, so it differs from gene to gene. Everything here that talks about "how often the term occurs among the
cell types" uses the gene-independent carrier count from GOEvidenceBuilder.M_ct instead.

Writes to data/go_experiment/match_summaries/:
  arm1_term_profile__<split>.json   one row per candidate term
  arm1_term_pairs__<split>.json     52x52 matrices over a shared term order

Usage:
    python scripts/analyze_arm1_term_profile.py --condition go_enrichment:v1:positive --gene-split all
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import mlflow
import numpy as np
from scipy.stats import hypergeom, spearmanr

from core.data_loader import parse_pair
from core.go_experiment import GOShared
from core.go_match import EXACT, classify_term
from scripts.analyze_arm1_gene_properties import load_hgnc_groups
from scripts.summarize_go_matches import load_runs

OUT_DIR = Path(__file__).parents[1] / "data" / "go_experiment" / "match_summaries"
EXPECTED_TOTALS = {"exact": 197, "upward": 795, "downward": 644, "no_match": 25143}  # match_summary, full run
N_SIGNIFICANT_TERMS = 26779


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", default="go_enrichment:v1:positive")
    ap.add_argument("--gene-split", default="all")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--limit", type=int, default=None, help="Smoke-test on a subset (checks are skipped)")
    args = ap.parse_args()

    print("Loading shared state...")
    sh = GOShared.load(need_baselines=False)
    go, b = sh.truth.go, sh.builder
    terms = list(b.candidates)
    tidx = {t: j for j, t in enumerate(terms)}
    label = {t: go.label(t) for t in terms}

    gene_props = {r["gene_id"]: r for r in json.loads((OUT_DIR / f"arm1_gene_properties__{args.gene_split}.json").read_text())["rows"]}
    gene_groups = load_hgnc_groups()

    # allow_mixed=True: same justification as scripts/analyze_arm1_gene_properties.py -- only deterministic
    # artifacts are read and truth/IC are recomputed fresh, so an unrelated shift of the code fingerprint
    # must not discard the validated full-dataset run.
    runs, _ = load_runs(args.condition, args.gene_split, allow_mixed=True)
    if args.limit:
        runs = runs[: args.limit]
    runs = [(g, i) for g, i in runs if sh.truth.has(g)]
    print(f"  {len(runs)} genes")

    # ---- cell-type side: how often does each term occur among the original cell types? ---------------
    n_ct, n_rows = len(b.ann_cl), len(b.ann_row_names)
    K_ct, K_row = b.M_ct.sum(0).astype(int), b.M_row.sum(0).astype(int)
    row_tissue = [parse_pair(p)[1] for p in b.ann_row_names]
    tissue_label = {u: (sh.lookup.resolve_uberon(u).label or u) for u in set(row_tissue)}
    all_rows_tissue_share = {u: c / n_rows for u, c in Counter(row_tissue).items()}

    purity = {}
    for t, j in tidx.items():
        carriers = np.flatnonzero(b.M_row[:, j] > 0)
        cnt = Counter(row_tissue[i] for i in carriers)
        tot = sum(cnt.values())
        top_u, top_n = cnt.most_common(1)[0] if cnt else (None, 0)
        ent = -sum((c / tot) * math.log(c / tot) for c in cnt.values()) if tot else 0.0
        purity[t] = {
            "n_tissues": len(cnt), "top_tissue": tissue_label.get(top_u), "top_tissue_share": top_n / tot if tot else None,
            "top_tissue_share_of_all_rows": all_rows_tissue_share.get(top_u),
            "entropy_normalized": ent / math.log(len(cnt)) if len(cnt) > 1 else 0.0,
        }

    # ---- base match rate: for how many genes would each term count as a match? ------------------------
    base_match = {t: Counter() for t in terms}
    for gene_id, _ in runs:
        td = sh.truth.annotations[gene_id]
        for t in terms:
            base_match[t][classify_term(go, t, td).kind] += 1

    # ---- one pass over the runs: assignments, outcomes, reachability -----------------------------------
    assigned_genes = {t: set() for t in terms}
    kinds = {t: Counter() for t in terms}
    reach = {t: Counter() for t in terms}  # tested / reachable
    violations = []
    for n_done, (gene_id, info) in enumerate(runs, 1):
        enr = json.load(open(mlflow.artifacts.download_artifacts(run_id=info["run_id"], artifact_path="enrichment_results.json")))
        diag = json.load(open(mlflow.artifacts.download_artifacts(run_id=info["run_id"], artifact_path="gscs_simulation.json")))
        thr, N, n = diag["gscs_threshold"], diag["n_background"], diag["n_query"]
        td = sh.truth.annotations[gene_id]
        for r in enr:
            t = r["go_id"]
            if t not in tidx:
                continue
            K = r["term_size"]
            p_floor = float(hypergeom.sf(min(K, n) - 1, N, K, n)) if n > 0 else 1.0
            # Significance is p_raw * alpha/thr < alpha, i.e. p_raw < thr. When the query nearly fills the
            # background and the term is fully contained in it, p_floor == thr exactly and float rounding in
            # p_gscs decides the comparison, so compare with a relative tolerance instead of strictly.
            ok = p_floor <= thr * (1 + 1e-9)
            reach[t]["tested"] += 1
            reach[t]["reachable"] += int(ok)
            if r.get("p_gscs", 1.0) < args.alpha:
                assigned_genes[t].add(gene_id)
                kinds[t][classify_term(go, t, td).kind] += 1
                if not ok:
                    violations.append((gene_id, t, K, n, N, p_floor, thr))
        if n_done % 2000 == 0:
            print(f"  {n_done}/{len(runs)} genes")

    # ---- checks -----------------------------------------------------------------------------------------
    totals = Counter()
    for t in terms:
        totals.update(kinds[t])
    n_sig = sum(totals.values())
    print(f"\nsignificant (gene,term) rows: {n_sig}   outcome totals: {dict(totals)}")
    if violations:
        print(f"  !! {len(violations)} assigned terms have p_floor >= threshold, e.g. {violations[:3]}")
        raise SystemExit("reachability invariant violated: the floor formula or N is wrong")
    if not args.limit:
        assert n_sig == N_SIGNIFICANT_TERMS, f"{n_sig} != {N_SIGNIFICANT_TERMS}"
        for k, v in EXPECTED_TOTALS.items():
            assert totals[k] == v, f"MISMATCH {k}: {totals[k]} != {v}"
        chance_path = OUT_DIR / f"arm1_chance_baseline__{args.gene_split}.json"
        if chance_path.exists():
            chance = json.loads(chance_path.read_text())
            lhs = sum(sum(base_match[t][k] for k in ("exact", "upward", "downward")) for t in terms)
            rhs = sum(r["chance"] * len(terms) for r in chance)
            assert abs(lhs - rhs) < 1e-6 * max(lhs, 1) + 1e-6, f"base-rate cross-check failed: {lhs} vs {rhs}"
            print(f"  cross-check vs chance baseline: OK ({lhs} gene-term matches)")
    motility = next((t for t in terms if label[t] == "cell motility"), None)
    if motility:
        assert K_ct[tidx[motility]] == int((b.M_ct[:, tidx[motility]] > 0).sum())

    # ---- gene profile -----------------------------------------------------------------------------------
    run_genes = [g for g, _ in runs]
    all_breadth = np.array([gene_props[g]["breadth"] for g in run_genes if gene_props.get(g, {}).get("breadth") is not None])
    fam_all = Counter(gene_groups.get(g) for g in run_genes)

    rows = []
    for t in terms:
        j = tidx[t]
        ag = assigned_genes[t]
        n_assigned = len(ag)
        bm = base_match[t]
        base_rate = sum(bm[k] for k in ("exact", "upward", "downward")) / len(runs)
        matches = kinds[t]["exact"] + kinds[t]["upward"] + kinds[t]["downward"]
        precision = matches / n_assigned if n_assigned else None
        br = np.array([gene_props[g]["breadth"] for g in ag if gene_props.get(g, {}).get("breadth") is not None])
        fam = Counter(gene_groups.get(g) for g in ag)
        top_fams = []
        for f, c in [x for x in fam.most_common(5) if x[0] is not None][:3]:
            top_fams.append({"family": f, "share_among_assigned": c / n_assigned,
                             "share_among_all": fam_all[f] / len(run_genes),
                             "enrichment": (c / n_assigned) / (fam_all[f] / len(run_genes))})
        rows.append({
            "go_id": t, "label": label[t], "ic": go.ic(t),
            "K_ct": int(K_ct[j]), "frac_ct": K_ct[j] / n_ct, "K_row": int(K_row[j]), "frac_row": K_row[j] / n_rows,
            "assigned": n_assigned, "n_exact": kinds[t]["exact"], "n_upward": kinds[t]["upward"],
            "n_downward": kinds[t]["downward"], "n_no_match": kinds[t]["no_match"],
            "precision": precision, "base_match_rate": base_rate,
            "base_exact": bm["exact"] / len(runs), "base_upward": bm["upward"] / len(runs), "base_downward": bm["downward"] / len(runs),
            "lift": (precision / base_rate) if (precision is not None and base_rate > 0) else None,
            **purity[t],
            "n_genes_tested": reach[t]["tested"], "n_reachable": reach[t]["reachable"],
            "frac_reachable": reach[t]["reachable"] / reach[t]["tested"] if reach[t]["tested"] else None,
            "mean_breadth_assigned": float(br.mean()) if len(br) else None,
            "median_breadth_assigned": float(np.median(br)) if len(br) else None,
            "top_families": top_fams,
        })

    a = np.array([r["assigned"] for r in rows]); k = np.array([r["K_ct"] for r in rows])
    rho_all = spearmanr(a, k).statistic
    rho_assigned = spearmanr(a[a > 0], k[a > 0]).statistic
    prec_ok = [(r["precision"], r["K_ct"]) for r in rows if r["assigned"] >= 100]
    rho_prec = spearmanr([p for p, _ in prec_ok], [kk for _, kk in prec_ok]).statistic
    summary = {"n_annotated_cell_types": n_ct, "n_annotated_rows": n_rows, "n_genes": len(runs),
               "spearman_assigned_vs_K_ct_all_terms": float(rho_all),
               "spearman_assigned_vs_K_ct_assigned_terms": float(rho_assigned),
               "spearman_precision_vs_K_ct_terms_assigned_ge_100": float(rho_prec),
               "median_breadth_all_genes": float(np.median(all_breadth))}
    print(f"\nSpearman(assigned, K_ct): all 52 terms {rho_all:.3f}; assigned terms only {rho_assigned:.3f}")
    print(f"Spearman(precision, K_ct), terms with >=100 assignments: {rho_prec:.3f}")

    # ---- pairs --------------------------------------------------------------------------------------------
    n_t = len(terms)
    ct_sets = [set(np.flatnonzero(b.M_ct[:, j] > 0)) for j in range(n_t)]
    J_genes, J_ct, related = np.zeros((n_t, n_t)), np.zeros((n_t, n_t)), np.zeros((n_t, n_t), dtype=int)
    for i, ti in enumerate(terms):
        di = go.ancestor_depths(ti)
        for j, tj in enumerate(terms):
            J_genes[i, j] = jaccard(assigned_genes[ti], assigned_genes[tj])
            J_ct[i, j] = jaccard(ct_sets[i], ct_sets[j])
            related[i, j] = int(i != j and (di.get(tj) is not None or go.ancestor_depths(tj).get(ti) is not None))

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    base = {"condition": args.condition, "gene_split": args.gene_split, "generated_at": stamp}
    (out / f"arm1_term_profile__{args.gene_split}.json").write_text(json.dumps({**base, "summary": summary, "rows": rows}, indent=2))
    (out / f"arm1_term_pairs__{args.gene_split}.json").write_text(json.dumps({
        **base, "terms": terms, "labels": [label[t] for t in terms], "jaccard_assigned_genes": J_genes.tolist(),
        "jaccard_carrier_cell_types": J_ct.tolist(), "ontology_related": related.tolist()}))
    print(f"Wrote {out}/arm1_term_profile__{args.gene_split}.json ({len(rows)} terms) and arm1_term_pairs__{args.gene_split}.json")


if __name__ == "__main__":
    main()
