#!/usr/bin/env python3
"""
Flatten arm 2's (go_llm) MLflow runs into two small files that can be copied to another machine, so the analysis
needs neither MLflow nor the artifacts. Run it where the MLflow database lives (the server), then copy the
two parquet files back.

Writes to data/go_experiment/cache/ (condition "go_llm:v3:freeform:true" -> go_llm_v3_freeform_true):
  arm2_runs__<cond>__<split>.parquet         one row per gene that has ANY run, including unfinished and failed ones:
      gene_id, symbol, status (COMPLETED | FAILED_PARSE | FAILED_BLINDING | UNFINISHED), run_id, code_hash,
      current_code (run belongs to the code that is current right now), model, start_time, scores (gopred_* incl.
      f1_at_1/3/5/10), latency_s, token counts, n_valid / n_invalid / frac_in_table, the model's hypothesis text
  arm2_predictions__<cond>__<split>.parquet  one row per (COMPLETED gene, predicted term):
      rank, go_id, label, as_written, confidence, in_table (is the term one of the evidence terms),
      kind / distance / matched_term (core.go_match against the gene's direct true terms), ic, ic_true, ic_gap
  arm2_invalid__<cond>__<split>.parquet      one row per (COMPLETED gene, answer that did not resolve to a GO term):
      gene_id, symbol, term (the text exactly as the model wrote it)
  arm2_cache_meta__<cond>__<split>.json

Which genes are MISSING is not decided here (this file only knows genes that have a run); the coverage analysis
(scripts/analyze_arm2_coverage.py) compares against the full list of scoreable genes.

Only COMPLETED runs of the current code count as done, as in scripts/run_go_experiment.py; a COMPLETED run of older
code is listed with current_code = False. A gene with several runs uses its latest completed one.

Built-in check: the F1@3 recomputed from the exported predictions must equal the F1@3 MLflow stored for the run.

Usage (on the server):
    python scripts/export_arm2_cache.py --condition go_llm:v3:freeform:true --gene-split all
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import mlflow
import pandas as pd

from core.arm2_cache import CACHE_DIR, safe_name
from core.go_experiment import GOShared
from core.go_match import classify_term
from core.go_scoring import score_ranking
from core.mlflow_utils import _TRACKING_URI
from core.run_identity import fingerprint

METRICS = ["gopred_f1_at_1", "gopred_f1_at_3", "gopred_f1_at_5", "gopred_f1_at_10", "gopred_f1_mean_1_10",
           "gopred_n_predictions", "gopred_truth_ic_mass", "latency_s", "actual_prompt_tokens", "context_utilization",
           "n_predictions_raw", "n_valid", "n_invalid", "n_outside_table", "invalid_rate", "frac_in_table", "parse_retries"]


def artifact_json(run_id: str, artifact_uri: str, name: str):
    """Read one JSON artifact. Local artifact roots are read in place (fast); anything else goes through MLflow."""
    parsed = urlparse(artifact_uri)
    if parsed.scheme in ("", "file"):
        path = Path(unquote(parsed.path)) / name
        if path.exists():
            return json.loads(path.read_text())
        return None
    try:
        return json.load(open(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=name)))
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", default="go_llm:v3:freeform:true")
    ap.add_argument("--gene-split", default="all")
    ap.add_argument("--allow-mixed", action="store_true", help="Count completed runs of ANY code version as done")
    ap.add_argument("--limit", type=int, default=None, help="Smoke-test on the first N genes with a run")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", default=str(CACHE_DIR))
    args = ap.parse_args()
    if not args.condition.startswith("go_llm:"):
        raise SystemExit("this exporter is for go_llm conditions; arm 1 has scripts/export_arm1_cache.py")

    mlflow.set_tracking_uri(_TRACKING_URI)
    exp = mlflow.get_experiment_by_name("CellCoLLM/go_llm")
    if exp is None:
        raise SystemExit("No experiment CellCoLLM/go_llm in this MLflow database.")
    df = mlflow.search_runs(
        [exp.experiment_id], max_results=100000, order_by=["start_time ASC"],
        filter_string=f"tags.condition = '{args.condition}' and tags.gene_split = '{args.gene_split}'")
    print(f"{len(df)} runs for {args.condition} / {args.gene_split}")
    current = fingerprint("go_llm")

    def status_of(r) -> str:
        s = r.get("tags.status")
        return s if isinstance(s, str) and s else "UNFINISHED"  # killed mid-gene: no status tag was ever set

    # one row per gene: the latest COMPLETED run (current code first), else the latest run of any kind
    chosen: dict[str, pd.Series] = {}
    for _, r in df.iterrows():
        g = r["tags.gene_id"]
        rank = (status_of(r) == "COMPLETED", r.get("tags.code_hash") == current)
        if g not in chosen or rank >= chosen[g]["_rank"]:
            r = r.copy(); r["_rank"] = rank
            chosen[g] = r
    genes = list(chosen)[: args.limit] if args.limit else list(chosen)

    sh = GOShared.load(need_baselines=False)
    go, cands = sh.truth.go, set(sh.builder.candidates)

    def load(g: str):
        r = chosen[g]
        if status_of(r) != "COMPLETED":
            return g, None, None, None
        return g, artifact_json(r["run_id"], r["artifact_uri"], "go_predictions.json"), \
            artifact_json(r["run_id"], r["artifact_uri"], "response_parsed.json"), \
            artifact_json(r["run_id"], r["artifact_uri"], "invalid_predictions.json")  # only exists if a run had invalid answers

    with ThreadPoolExecutor(args.workers) as pool:
        loaded = list(pool.map(load, genes))

    run_rows, pred_rows, invalid_rows, mismatches, checked = [], [], [], [], 0
    for g, preds, parsed, invalid in loaded:
        r = chosen[g]
        status = status_of(r)
        row = {"gene_id": g, "symbol": r.get("tags.gene_symbol"), "status": status, "run_id": r["run_id"],
               "code_hash": r.get("tags.code_hash"), "current_code": r.get("tags.code_hash") == current,
               "model": r.get("params.model"), "start_time": r["start_time"],
               "hypothesis": (parsed or {}).get("hypothesis") if isinstance(parsed, dict) else None}
        for m in METRICS:
            row[m] = r.get(f"metrics.{m}")
        run_rows.append(row)
        if status == "COMPLETED" and isinstance(invalid, list):
            invalid_rows += [{"gene_id": g, "symbol": row["symbol"], "term": str(t)} for t in invalid]
        if status != "COMPLETED" or preds is None or not sh.truth.has(g):
            continue
        td = sh.truth.annotations[g]
        conf = {}  # model's confidence per written name, from the parsed answer
        for p in (parsed or {}).get("predictions", []) if isinstance(parsed, dict) else []:
            if isinstance(p, dict) and p.get("name"):
                conf[str(p["name"]).strip()] = p.get("confidence")
        for p in preds:
            m = classify_term(go, p["go_id"], td)
            true_t = p["go_id"] if m.kind == "exact" else m.matched_term
            ic = go.ic(p["go_id"])
            ic_true = go.ic(true_t) if true_t else None
            pred_rows.append({
                "gene_id": g, "symbol": row["symbol"], "rank": p["rank"], "go_id": p["go_id"], "label": p["label"],
                "as_written": p.get("as_written"), "confidence": conf.get(str(p.get("as_written", "")).strip()),
                "in_table": p["go_id"] in cands, "kind": m.kind, "distance": m.distance, "matched_term": true_t,
                "ic": ic, "ic_true": ic_true, "ic_gap": (ic_true - ic) if ic_true is not None else None})
        # fidelity check: F1@3 from the exported list must equal what MLflow stored for the run
        stored = row["gopred_f1_at_3"]
        if stored is not None and stored == stored and checked < 300:
            checked += 1
            again = score_ranking(sh.truth, [p["go_id"] for p in preds], g, None)["f1_at_3"]
            if abs(again - stored) > 1e-9:
                mismatches.append((g, stored, again))

    runs, preds_df = pd.DataFrame(run_rows), pd.DataFrame(pred_rows)
    invalid_df = pd.DataFrame(invalid_rows, columns=["gene_id", "symbol", "term"])
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    cond, split = safe_name(args.condition), args.gene_split
    runs.to_parquet(out / f"arm2_runs__{cond}__{split}.parquet", index=False)
    preds_df.to_parquet(out / f"arm2_predictions__{cond}__{split}.parquet", index=False)
    invalid_df.to_parquet(out / f"arm2_invalid__{cond}__{split}.parquet", index=False)
    meta = {"condition": args.condition, "gene_split": split, "code_hash_current": current, "n_genes_with_run": len(runs),
            "status_counts": runs["status"].value_counts().to_dict() if len(runs) else {},
            "n_completed_current_code": int(((runs["status"] == "COMPLETED") & runs["current_code"]).sum()) if len(runs) else 0,
            "models": sorted(set(runs["model"].dropna())) if len(runs) else [], "n_predictions": len(preds_df), "n_invalid_answers": len(invalid_df),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (out / f"arm2_cache_meta__{cond}__{split}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"F1@3 fidelity check: {checked} genes compared, {len(mismatches)} mismatches")
    if mismatches:
        print("  e.g.", mismatches[:3])
        raise SystemExit("exported predictions do not reproduce the stored F1@3")
    if len(meta["models"]) > 1:
        print(f"WARNING: runs were made with more than one model: {meta['models']}")
    if meta["n_completed_current_code"] < meta["status_counts"].get("COMPLETED", 0):
        print("NOTE: some COMPLETED runs belong to older code and are marked current_code=False")
    print(f"Wrote arm2_runs, arm2_predictions and arm2_invalid ({len(invalid_df)} invalid answers) for {cond}__{split} to {out}")


if __name__ == "__main__":
    main()
