#!/usr/bin/env python3
"""
Derived tables for the arm-2 (go_llm) results notebook (notebooks/arm2_go_llm_results.ipynb). Everything that is heavy
or needs the GO ontology is computed here; the notebook only loads and plots.

Reads   data/go_experiment/cache/arm2_*.parquet (scripts/export_arm2_cache.py) and arm1_tested_terms__all.parquet
Writes  data/go_experiment/match_summaries/
          arm2_gene_table__<cond>__<split>.parquet   one row per finished gene (see below)
          arm2_match_details__<cond>__<split>.parquet one row per real-match prediction (relations, true term)
          arm2_results__<cond>__<split>.json          invalid-answer classification, name forms, cost, checks

Gene table: arm-2 scores, valid/invalid/duplicate counts, outcome counts, "match@k" (any real match among the top k
predictions, k = 1, 3, 10), gene properties (arm-1 analysis), the paired arm-1 result for the same gene, and the
SHUFFLED-PREDICTION BASELINE: the same gene's predicted terms scored against the true terms of 5 other genes with
similar expression breadth (|difference| <= 0.05) and averaged. Where the observed matches are well above this, the
answers are specific to the gene; where they are not, the model is saying things that fit many genes. It is the
no-LLM stand-in for a "mismatched evidence" control, not a replacement for running one.

Invalid answers are classified for DIAGNOSIS only. Nothing here changes how any run was scored: scoring uses exact GO
names and exact synonyms, nothing else.

Usage:  python scripts/analyze_arm2_results.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from core.arm1_cache import load_arm1_cache
from core.arm2_cache import DEFAULT_CONDITION, load_arm2_cache, safe_name
from core.go_experiment import GOShared
from core.go_match import classify_term
from core.go_ontology import _SYNONYM_RE, BP_NAMESPACE, GOOntology
from scripts.analyze_arm2_coverage import load_universe

SUMMARY_DIR = ROOT / "data" / "go_experiment" / "match_summaries"
KINDS = ["exact", "upward", "downward", "no_match"]
N_DONORS, DONOR_BREADTH_TOL = 5, 0.05
ARM1_FULL_RUN = {"genes": 15754, "genes_with_real_match": 1220}


# ---- invalid-answer classification (diagnosis only) ------------------------------------------------------------
def build_indexes(go: GOOntology):
    """Name indexes over ALL GO namespaces. Built here rather than in core/go_ontology.py on purpose."""
    names_all: dict[str, set] = defaultdict(set)        # name or EXACT synonym -> ids, any namespace
    bp_loose: dict[str, set] = defaultdict(set)         # RELATED/BROAD/NARROW synonyms of BP terms -> ids
    bp_names: dict[str, str] = {}                       # BP official name (normalised) -> id
    for node, d in go._g.nodes(data=True):
        ns = d.get("namespace")
        if d.get("name"):
            names_all[go._norm(d["name"])].add((node, ns))
            if ns == BP_NAMESPACE:
                bp_names[go._norm(d["name"])] = node
        for syn in d.get("synonym", []) or []:
            m = _SYNONYM_RE.match(syn)
            if not m:
                continue
            if m.group(2) == "EXACT":
                names_all[go._norm(m.group(1))].add((node, ns))
            elif ns == BP_NAMESPACE:
                bp_loose[go._norm(m.group(1))].add(node)
    token_index: dict[str, list[str]] = defaultdict(list)
    for n in bp_names:
        for t in set(n.split()):
            token_index[t].append(n)
    return names_all, bp_loose, bp_names, token_index


def classify_invalid(go: GOOntology, text: str, idx) -> tuple[str, str | None]:
    names_all, bp_loose, bp_names, token_index = idx
    n = go._norm(text)
    hits = names_all.get(n, set())
    bp_hits = {i for i, ns in hits if ns == BP_NAMESPACE}
    if len(bp_hits) > 1:
        return "name matches several biological-process terms (ambiguous)", None
    other = {ns for _, ns in hits if ns != BP_NAMESPACE}
    if other:
        return ("a real GO term, but a molecular function" if "molecular_function" in other else
                "a real GO term, but a cellular component"), None
    if n in bp_loose:
        return "a related / broad / narrow synonym of a process term (near miss, not an exact name)", go.label(next(iter(bp_loose[n])))
    # reworded: share almost all words with an existing process name (token overlap, diagnostic only)
    toks = set(n.split())
    cand = Counter()
    for t in toks:
        if len(token_index.get(t, ())) <= 300:
            cand.update(token_index[t])
    best, best_j = None, 0.0
    for name, _ in cand.most_common(60):
        nt = set(name.split())
        j = len(toks & nt) / len(toks | nt)
        if j > best_j:
            best, best_j = name, j
    if best is not None and best_j >= 0.75:
        return "reworded: very close to an existing process name", go.label(bp_names[best])
    if "activity" in n.split() or "binding" in n.split():
        return "function-type wording (activity / binding), not a process name", None
    return "not a GO name", None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", default=DEFAULT_CONDITION)
    ap.add_argument("--gene-split", default="all")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--out-dir", default=str(SUMMARY_DIR))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    c2 = load_arm2_cache(args.condition, args.gene_split, args.cache_dir)
    c1 = load_arm1_cache(args.gene_split)
    sh = GOShared.load(need_baselines=False)
    go = sh.truth.go
    done, preds = c2.done.copy(), c2.predictions
    print(f"{len(done)} finished genes, {len(preds)} valid predictions")

    # ---- gene table ------------------------------------------------------------------------------------------------
    g = done.set_index("gene_id")[["symbol", "gopred_f1_at_1", "gopred_f1_at_3", "gopred_f1_at_10", "latency_s", "actual_prompt_tokens",
                                   "n_predictions_raw", "n_valid", "n_invalid", "n_outside_table", "frac_in_table", "parse_retries"]].copy()
    g = g.rename(columns={"gopred_f1_at_1": "f1_at_1", "gopred_f1_at_3": "f1_at_3", "gopred_f1_at_10": "f1_at_10"})
    g["n_duplicate"] = (g["n_predictions_raw"] - g["n_valid"] - g["n_invalid"]).clip(lower=0)
    cnt = preds.groupby(["gene_id", "kind"]).size().unstack(fill_value=0).reindex(columns=KINDS, fill_value=0)
    for k in KINDS:
        g[f"n_{k}"] = cnt[k].reindex(g.index).fillna(0).astype(int)
    real = preds["kind"] != "no_match"
    for k in (1, 3, 10):
        top = preds["rank"] <= k
        g[f"match_at_{k}"] = (real & top).groupby(preds["gene_id"]).any().reindex(g.index, fill_value=False).astype(int)
        g[f"exact_at_{k}"] = ((preds["kind"] == "exact") & top).groupby(preds["gene_id"]).any().reindex(g.index, fill_value=False).astype(int)
    g["any_match"] = (g["n_exact"] + g["n_upward"] + g["n_downward"] > 0).astype(int)
    g["any_exact"] = (g["n_exact"] > 0).astype(int)

    u = load_universe().set_index("gene_id")
    prop_cols = ["breadth", "n_direct_terms", "f1_top1", "mean_true_ic", "frac_allornothing_tissues", "gene_group", "chance"]
    g = g.join(u[[c for c in prop_cols if c in u]], how="left")

    # paired arm 1: significant terms ranked by p; a gene without any significant term has no prediction
    sig1 = c1.sig
    g["arm1_n_sig"] = sig1.groupby("gene_id").size().reindex(g.index).fillna(0).astype(int)
    for k in (1, 3, 10):
        top = sig1["rank"] <= k
        g[f"arm1_match_at_{k}"] = ((sig1["kind"] != "no_match") & top).groupby(sig1["gene_id"]).any().reindex(g.index, fill_value=False).astype(int)
        g[f"arm1_exact_at_{k}"] = ((sig1["kind"] == "exact") & top).groupby(sig1["gene_id"]).any().reindex(g.index, fill_value=False).astype(int)
    g["arm1_in_run"] = g.index.isin(set(c1.terms["gene_id"]))

    # ---- shuffled-prediction baseline --------------------------------------------------------------------------------
    pool = g[g["breadth"].notna()].sort_values("breadth")
    pool_b, pool_ids = pool["breadth"].to_numpy(), pool.index.to_numpy()
    by_gene = {gid: grp for gid, grp in preds.groupby("gene_id")}
    shuf = defaultdict(list)
    for gid in g.index:
        row = g.loc[gid]
        if gid not in by_gene or pd.isna(row["breadth"]):
            for key in ("shuf_exact", "shuf_upward", "shuf_downward", "shuf_no_match", "shuf_any_match", "shuf_any_exact", "shuf_match_at_3"):
                shuf[key].append(np.nan)
            continue
        lo_i, hi_i = np.searchsorted(pool_b, row["breadth"] - DONOR_BREADTH_TOL, side="left"), np.searchsorted(pool_b, row["breadth"] + DONOR_BREADTH_TOL, side="right")
        cands = pool_ids[lo_i:hi_i]
        cands = cands[cands != gid]
        donors = rng.choice(cands, N_DONORS, replace=len(cands) < N_DONORS)
        assert gid not in donors, "a gene was used as its own donor"
        pg = by_gene[gid].sort_values("rank")
        acc = Counter(); any_m = any_e = m3 = 0
        for d in donors:
            td = sh.truth.annotations[d]
            kinds = [classify_term(go, pid, td).kind for pid in pg["go_id"]]
            acc.update(kinds)
            any_m += any(k != "no_match" for k in kinds); any_e += any(k == "exact" for k in kinds)
            m3 += any(k != "no_match" for k in kinds[:3])
        for k in KINDS:
            shuf[f"shuf_{k}"].append(acc[k] / N_DONORS)
        shuf["shuf_any_match"].append(any_m / N_DONORS); shuf["shuf_any_exact"].append(any_e / N_DONORS); shuf["shuf_match_at_3"].append(m3 / N_DONORS)
    for k, v in shuf.items():
        g[k] = v

    # ---- match details (relations to the true terms) ---------------------------------------------------------------------
    det = []
    for r in preds[preds["kind"].isin(["upward", "downward"])].itertuples():
        td = sh.truth.annotations[r.gene_id]
        depths = go.ancestor_depths(r.go_id)
        also_down = any(t in depths for t in td) if r.kind == "upward" else None   # an upward label can also be a descendant of another true term
        det.append({"gene_id": r.gene_id, "go_id": r.go_id, "label": r.label, "kind": r.kind, "distance": r.distance,
                    "true_term": r.matched_term, "true_label": go.label(r.matched_term), "ic": r.ic, "ic_true": r.ic_true,
                    "also_more_specific_than_other_true_term": also_down})
    details = pd.DataFrame(det)

    # ---- invalid answers, name forms ----------------------------------------------------------------------------------------
    results: dict = {"condition": args.condition, "gene_split": args.gene_split, "n_finished_genes": len(done),
                     "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "n_valid_predictions": int(len(preds))}
    if c2.invalid is not None and len(c2.invalid):
        idx = build_indexes(go)
        distinct = {t: classify_invalid(go, t, idx) for t in c2.invalid["term"].unique()}
        inv = c2.invalid.assign(klass=c2.invalid["term"].map(lambda t: distinct[t][0]), nearest=c2.invalid["term"].map(lambda t: distinct[t][1]))
        results["invalid"] = {
            "n_answers": int(len(inv)), "n_distinct": int(inv["term"].nunique()),
            "by_class_answers": inv["klass"].value_counts().to_dict(),
            "by_class_distinct": inv.drop_duplicates("term")["klass"].value_counts().to_dict(),
            "top_strings": [{"term": t, "n": int(n), "class": distinct[t][0], "nearest_go_name": distinct[t][1]}
                            for t, n in inv["term"].value_counts().head(25).items()]}
    else:
        results["invalid"] = None
    w = preds["as_written"].fillna("").str.strip()
    differs = w.str.lower() != preds["label"].str.lower()
    is_id = w.str.upper().str.startswith("GO:")
    results["name_forms"] = {"n_valid_predictions": int(len(preds)), "n_official_name": int((~differs).sum()),
                             "n_other_form": int((differs & ~is_id).sum()), "n_written_as_id": int(is_id.sum()),
                             "top_other_forms": [{"written": a, "resolved_to": b, "n": int(n)} for (a, b), n in
                                                 preds[differs & ~is_id].groupby(["as_written", "label"]).size().sort_values(ascending=False).head(15).items()]}
    lat = done["latency_s"].dropna()
    results["cost"] = {"latency_median_s": float(lat.median()), "latency_p90_s": float(lat.quantile(0.9)), "latency_mean_s": float(lat.mean()),
                       "prompt_tokens_median": float(done["actual_prompt_tokens"].median()),
                       "n_universe": int(len(u)), "n_remaining": int(len(u) - len(done)),
                       "remaining_hours_at_median": float((len(u) - len(done)) * lat.median() / 3600)}

    # ---- checks ----------------------------------------------------------------------------------------------------------------
    assert int(g["n_valid"].sum()) == len(preds), "valid-prediction count differs from the exported predictions table"
    assert int(sum(g[f"n_{k}"].sum() for k in KINDS)) == len(preds), "outcome totals differ from the predictions table"
    assert abs(g["f1_at_3"].mean() - done["gopred_f1_at_3"].mean()) < 1e-12
    in_arm1 = g[g["arm1_in_run"]]
    results["checks"] = {"valid_predictions_match_export": True, "outcomes_sum_to_predictions": True,
                         "arm1_genes_in_pairing": int(len(in_arm1)), "arm1_any_match_genes_all_genes_known_full_run": ARM1_FULL_RUN}
    full1 = (c1.sig.groupby("gene_id")["kind"].apply(lambda s: (s != "no_match").any()))
    assert int(full1.sum()) == ARM1_FULL_RUN["genes_with_real_match"], "arm-1 side does not reproduce its known full-run result"

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    name = f"{safe_name(args.condition)}__{args.gene_split}"
    g.reset_index().to_parquet(out / f"arm2_gene_table__{name}.parquet", index=False)
    details.to_parquet(out / f"arm2_match_details__{name}.parquet", index=False)
    (out / f"arm2_results__{name}.json").write_text(json.dumps(results, indent=2, default=float))

    print(f"gene table {g.shape}; match details {details.shape}")
    print(f"observed any-match {g['any_match'].mean():.3f}  vs shuffled {g['shuf_any_match'].mean():.3f}   |   exact {g['any_exact'].mean():.3f} vs {g['shuf_any_exact'].mean():.3f}")
    inv = results["invalid"]
    print("invalid answers by class:", inv["by_class_answers"] if inv else "(no invalid table in the cache; re-export on the server)")
    print(f"name forms: {results['name_forms']['n_official_name']} official, {results['name_forms']['n_other_form']} other form, {results['name_forms']['n_written_as_id']} as ID")
    print(f"cost: median {results['cost']['latency_median_s']:.1f}s per gene; {results['cost']['n_remaining']} genes remaining -> {results['cost']['remaining_hours_at_median']:.0f} h at the median")


if __name__ == "__main__":
    main()
