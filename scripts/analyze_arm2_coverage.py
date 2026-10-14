#!/usr/bin/env python3
"""
Which genes has arm 2 (go_llm) already covered, which are missing, and can the finished part be trusted to
represent the whole?

Arm 2 runs the genes one after the other in dataset order, so at any time only the genes up to some point have
results. If the finished genes differ systematically from the missing ones (more broadly expressed, better
annotated, easier ...), any average over the finished genes is biased as an estimate for the full dataset.
This script measures that, in four steps:

  1 COVERAGE        done / failed / missing counts, split by the reason a gene is missing, and the share done by
                    position in the run order
  2 WHO IS MISSING  the share done within each bin of the gene properties (expression breadth, number of known true
                    terms, oracle ceiling, specificity of the true terms, tissue/batch exposure, gene family,
                    whether arm 1 got a match), and the done-vs-missing difference of each property
                    (standardised mean difference d: |d| below 0.1 is a negligible difference, above 0.5 a large one)
  3 BIAS CHECK      arm 1 has results for EVERY gene, so it works as an outside reference: how does arm 1 perform on
                    the genes arm 2 has finished, compared with the ones still missing? A clear gap means the finished
                    part is easier (or harder) than the rest, independent of anything arm 2 did
  4 CORRECTION      arm 2's headline scores on the finished genes ("raw") against the same scores re-weighted so the
                    finished genes match the full population in breadth and annotation richness ("re-weighted"),
                    with bootstrap intervals. Re-weighting only fixes differences in the properties used; it cannot
                    repair a difference that is not captured by them. Strata with too few finished genes are flagged.

"Universe" = the scoreable genes (>=1 non-IEA biological-process annotation, from data/go_ceiling_genes.tsv).
Properties come from the arm-1 analysis (arm1_chance_baseline__all.json), which exists for 15,754 of them; the others
count in the overall coverage but not in the property bins.

Reads   data/go_experiment/cache/arm2_*.parquet   (scripts/export_arm2_cache.py, copied from the server)
Writes  data/go_experiment/match_summaries/arm2_coverage__<cond>__<split>.json
        data/go_experiment/match_summaries/arm2_missing_genes__<cond>__<split>.csv   (every missing gene and why)

Usage:  python scripts/analyze_arm2_coverage.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from core.arm2_cache import DEFAULT_CONDITION, load_arm2_cache, safe_name
from core.data_loader import read_gene_ids

SUMMARY_DIR = ROOT / "data" / "go_experiment" / "match_summaries"
CEILING_TSV = ROOT / "data" / "go_ceiling_genes.tsv"
NUMERIC_PROPS = {  # property -> plain-language name
    "breadth": "expression breadth (share of cell types expressing the gene)",
    "n_direct_terms": "number of known true GO terms",
    "f1_top1": "oracle ceiling (best possible score with this vocabulary)",
    "mean_true_ic": "specificity of the true terms (mean IC)",
    "frac_allornothing_tissues": "tissue batch-signature exposure",
    "n_significant": "arm 1: number of significant terms",
}
MIN_DONE_FOR_REWEIGHT = 200
MIN_PER_STRATUM = 20
RNG = np.random.default_rng(0)


def load_universe() -> pd.DataFrame:
    ceil = pd.read_csv(CEILING_TSV, sep="\t").rename(columns={"ensembl_id": "gene_id"})
    props = pd.DataFrame(json.loads((SUMMARY_DIR / "arm1_chance_baseline__all.json").read_text()))
    u = ceil[["gene_id", "n_direct_terms", "f1_top1"]].merge(
        props.drop(columns=[c for c in ("n_direct_terms", "f1_top1") if c in props]), on="gene_id", how="left")
    u["position"] = u["gene_id"].map({g: i for i, g in enumerate(read_gene_ids())})
    m = u[["n_exact", "n_upward", "n_downward"]].sum(axis=1, min_count=1)
    u["arm1_any_match"] = np.where(u["n_exact"].isna(), np.nan, (m > 0).astype(float))
    u["arm1_match_terms"] = m
    return u


def smd(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or len(b) < 2:
        return None
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / sd) if sd > 0 else 0.0


def boot_diff(a: np.ndarray, b: np.ndarray, n: int = 2000) -> list[float] | None:
    if len(a) < 5 or len(b) < 5:
        return None
    da = RNG.choice(a, (n, len(a))).mean(axis=1) - RNG.choice(b, (n, len(b))).mean(axis=1)
    return [float(np.percentile(da, 2.5)), float(np.percentile(da, 97.5))]


def binned_coverage(u: pd.DataFrame, col: str, q: int = 5) -> list[dict]:
    d = u[u[col].notna()]
    if d[col].nunique() < 2:
        return []
    bins = pd.qcut(d[col], q, duplicates="drop")
    out = []
    for iv, g in d.groupby(bins, observed=True):
        out.append({"bin": f"{iv.left:.3g} to {iv.right:.3g}", "n_universe": len(g), "n_done": int(g["done"].sum()),
                    "pct_done": float(g["done"].mean())})
    return out


def weighted_metrics(df: pd.DataFrame, w: np.ndarray) -> dict:
    """Mean per-gene scores and prediction-level shares, weighted per gene (w=1 gives the raw numbers)."""
    out = {}
    for m in ("f1_at_1", "f1_at_3", "f1_at_10"):
        out[m] = float(np.average(df[m], weights=w))
    out["gene_any_match"] = float(np.average(df["any_match"], weights=w))
    out["gene_any_exact"] = float(np.average(df["any_exact"], weights=w))
    tot = np.sum(w * df["n_pred"])
    for k in ("exact", "upward", "downward", "no_match"):
        out[f"share_{k}"] = float(np.sum(w * df[f"n_{k}"]) / tot) if tot else None
    out["share_in_evidence_table"] = float(np.sum(w * df["n_in_table"]) / tot) if tot else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", default=DEFAULT_CONDITION)
    ap.add_argument("--gene-split", default="all")
    ap.add_argument("--cache-dir", default=None, help="Where the arm2_*.parquet files are (default data/go_experiment/cache)")
    ap.add_argument("--out-dir", default=str(SUMMARY_DIR))
    ap.add_argument("--n-boot", type=int, default=300)
    args = ap.parse_args()

    cache = load_arm2_cache(args.condition, args.gene_split, args.cache_dir)
    u = load_universe()
    runs = cache.runs.set_index("gene_id")
    done_ids = set(cache.done["gene_id"])
    u["done"] = u["gene_id"].isin(done_ids)
    N, n_done = len(u), int(u["done"].sum())

    # ---- 1 coverage ---------------------------------------------------------------------------------------------
    def reason(g: str) -> str:
        if g in done_ids:
            return "done"
        if g not in runs.index:
            return "no run yet"
        r = runs.loc[g]
        if r["status"] == "COMPLETED":
            return "completed under older code"
        return {"FAILED_BLINDING": "skipped: symbol appears in the prompt (blinding)", "FAILED_PARSE": "failed: answer could not be parsed",
                "UNFINISHED": "run started but not finished"}.get(r["status"], f"status {r['status']}")
    u["reason"] = u["gene_id"].map(reason)
    u["position_decile"] = pd.qcut(u["position"].rank(method="first"), 10, labels=False)
    coverage = {"n_universe": N, "n_done": n_done, "pct_done": n_done / N, "reasons": u["reason"].value_counts().to_dict(),
                "by_position_decile": [{"decile": int(d) + 1, "n_universe": len(g), "n_done": int(g["done"].sum()), "pct_done": float(g["done"].mean())}
                                       for d, g in u.groupby("position_decile")],
                "n_with_properties": int(u["breadth"].notna().sum()), "n_without_properties": int(u["breadth"].isna().sum())}

    # ---- 2 who is missing -----------------------------------------------------------------------------------------
    by_prop, rep = {}, []
    for col, name in NUMERIC_PROPS.items():
        if col not in u:
            continue
        by_prop[col] = {"name": name, "bins": binned_coverage(u, col)}
        d, m = u.loc[u["done"] & u[col].notna(), col].to_numpy(float), u.loc[~u["done"] & u[col].notna(), col].to_numpy(float)
        ks = ks_2samp(d, m) if len(d) > 1 and len(m) > 1 else None
        rep.append({"property": col, "name": name, "mean_done": float(d.mean()) if len(d) else None,
                    "mean_missing": float(m.mean()) if len(m) else None, "median_done": float(np.median(d)) if len(d) else None,
                    "median_missing": float(np.median(m)) if len(m) else None, "smd": smd(d, m),
                    "ks_statistic": float(ks.statistic) if ks else None, "ks_p": float(ks.pvalue) if ks else None})
    fam = u["gene_group"].fillna("(no gene family)")
    top = fam.value_counts().drop("(no gene family)", errors="ignore").head(10).index
    u["family"] = np.where(fam.isin(top), fam, np.where(fam == "(no gene family)", fam, "(other families)"))
    by_prop["family"] = {"name": "HGNC gene family", "bins": [
        {"bin": f, "n_universe": len(g), "n_done": int(g["done"].sum()), "pct_done": float(g["done"].mean())}
        for f, g in u[u["breadth"].notna()].groupby("family")]}
    by_prop["arm1_any_match"] = {"name": "did arm 1 get a real match", "bins": [
        {"bin": "yes" if v == 1 else "no", "n_universe": len(g), "n_done": int(g["done"].sum()), "pct_done": float(g["done"].mean())}
        for v, g in u[u["arm1_any_match"].notna()].groupby("arm1_any_match")]}

    # ---- 3 bias check against arm 1 (known for every gene) ----------------------------------------------------------
    ref = {}
    has = u[u["breadth"].notna()]
    for col, name in [("arm1_any_match", "arm 1: share of genes with at least one real match"),
                      ("f1_top1", "oracle ceiling (best possible score)"), ("chance", "chance baseline match rate"),
                      ("breadth", "expression breadth")]:
        d, m = has.loc[has["done"], col].to_numpy(float), has.loc[~has["done"], col].to_numpy(float)
        ref[col] = {"name": name, "done": float(d.mean()) if len(d) else None, "missing": float(m.mean()) if len(m) else None,
                    "difference": float(d.mean() - m.mean()) if len(d) and len(m) else None, "ci95": boot_diff(d, m)}
    dm, mm = has.loc[has["done"]], has.loc[~has["done"]]
    for key, nm, a, b in [("arm1_match_per_term", "arm 1: share of its significant terms that match", dm, mm)]:
        ra = a["arm1_match_terms"].sum() / max(a["n_significant"].sum(), 1)
        rb = b["arm1_match_terms"].sum() / max(b["n_significant"].sum(), 1)
        ref[key] = {"name": nm, "done": float(ra), "missing": float(rb), "difference": float(ra - rb), "ci95": None}

    # ---- 4 raw vs re-weighted arm-2 scores --------------------------------------------------------------------------
    rew = {"status": "skipped", "reason": f"fewer than {MIN_DONE_FOR_REWEIGHT} finished genes with properties"}
    p = cache.predictions
    if len(dm) >= MIN_DONE_FOR_REWEIGHT:
        g = cache.done.set_index("gene_id")
        cnt = p.groupby(["gene_id", "kind"]).size().unstack(fill_value=0).reindex(columns=["exact", "upward", "downward", "no_match"], fill_value=0)
        sc = pd.DataFrame({"f1_at_1": g["gopred_f1_at_1"], "f1_at_3": g["gopred_f1_at_3"], "f1_at_10": g["gopred_f1_at_10"]})
        for k in cnt:
            sc[f"n_{k}"] = cnt[k]
        sc = sc.fillna({f"n_{k}": 0 for k in cnt}).dropna(subset=["f1_at_3"])
        sc["n_pred"] = sc[[f"n_{k}" for k in cnt]].sum(axis=1)
        sc["n_in_table"] = p[p["in_table"]].groupby("gene_id").size().reindex(sc.index).fillna(0)
        sc["any_match"] = ((sc["n_exact"] + sc["n_upward"] + sc["n_downward"]) > 0).astype(float)
        sc["any_exact"] = (sc["n_exact"] > 0).astype(float)
        sc = sc.join(has.set_index("gene_id")[["breadth", "n_direct_terms"]], how="inner")
        edges_b = np.unique(np.quantile(has["breadth"], np.linspace(0, 1, 6)))
        edges_a = np.unique(np.quantile(has["n_direct_terms"], np.linspace(0, 1, 4)))
        def stratum(df):
            b = np.clip(np.searchsorted(edges_b, df["breadth"], side="right") - 1, 0, len(edges_b) - 2)
            a = np.clip(np.searchsorted(edges_a, df["n_direct_terms"], side="right") - 1, 0, len(edges_a) - 2)
            return pd.Series(b * 10 + a, index=df.index)
        has = has.assign(stratum=stratum(has).to_numpy())
        sc["stratum"] = stratum(sc)
        n_pop, n_fin = has["stratum"].value_counts(), sc["stratum"].value_counts()
        w = sc["stratum"].map(n_pop / n_fin).to_numpy(float)
        thin = {int(s): {"n_universe": int(n_pop[s]), "n_done": int(n_fin.get(s, 0))} for s in n_pop.index if n_fin.get(s, 0) < MIN_PER_STRATUM}
        uncovered = float(sum(n_pop[s] for s in n_pop.index if n_fin.get(s, 0) == 0) / n_pop.sum())
        raw, adj = weighted_metrics(sc, np.ones(len(sc))), weighted_metrics(sc, w)
        boots_r, boots_w = [], []
        idx_by_s = {s: np.flatnonzero(sc["stratum"].to_numpy() == s) for s in n_fin.index}
        for _ in range(args.n_boot):
            pick = np.concatenate([RNG.choice(ix, len(ix)) for ix in idx_by_s.values()])
            sub = sc.iloc[pick]
            boots_r.append(weighted_metrics(sub, np.ones(len(sub))))
            boots_w.append(weighted_metrics(sub, sub["stratum"].map(n_pop / n_fin).to_numpy(float)))
        ci = lambda bs, k: [float(np.percentile([b[k] for b in bs if b[k] is not None], 2.5)), float(np.percentile([b[k] for b in bs if b[k] is not None], 97.5))]
        rew = {"status": "ok", "n_done_used": len(sc), "n_strata": int(len(n_pop)), "kish_effective_n": float(w.sum() ** 2 / (w ** 2).sum()),
               "max_weight": float(w.max()), "median_weight": float(np.median(w)), "uncovered_universe_share": uncovered,
               "thin_strata_under_min": thin, "min_per_stratum": MIN_PER_STRATUM,
               "metrics": {k: {"raw": raw[k], "reweighted": adj[k], "raw_ci95": ci(boots_r, k), "reweighted_ci95": ci(boots_w, k)} for k in raw},
               "strata_definition": "5 quantile bins of expression breadth x 3 quantile bins of number of known true terms (computed on all genes with properties)"}

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    name = f"{safe_name(args.condition)}__{args.gene_split}"
    (out / f"arm2_coverage__{name}.json").write_text(json.dumps({
        "condition": args.condition, "gene_split": args.gene_split, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cache_meta": cache.meta, "coverage": coverage, "by_property": by_prop, "representativeness": rep,
        "arm1_reference": ref, "reweighting": rew}, indent=2, default=float))
    u.loc[~u["done"], ["gene_id", "symbol", "position", "reason", "breadth", "n_direct_terms", "f1_top1"]].sort_values("position").to_csv(
        out / f"arm2_missing_genes__{name}.csv", index=False)

    print(f"coverage: {n_done} of {N} scoreable genes done ({n_done / N:.1%}); reasons: {coverage['reasons']}")
    print("done vs missing, standardised difference per property (|d| < 0.1 negligible, > 0.5 large):")
    for r in rep:
        print(f"  {r['property']:28s} done {r['mean_done']!s:.8}  missing {r['mean_missing']!s:.8}  d = {r['smd']}")
    print("arm 1 on the same genes (done vs missing):", {k: (round(v['done'], 4), round(v['missing'], 4)) for k, v in ref.items() if v["done"] is not None})
    print("re-weighting:", rew["status"] if rew["status"] != "ok" else {k: (round(v['raw'], 4), round(v['reweighted'], 4)) for k, v in rew["metrics"].items() if k.startswith(("f1_at_3", "gene_any", "share_exact"))})


if __name__ == "__main__":
    main()
