#!/usr/bin/env python3
"""
Run the GO-prediction experiment: every rung of the ladder on every selected gene.

In-process on purpose. Each of the thousands of runs needs the dataset, the GO ontology,
the GOA annotations and the evidence matrices; a subprocess per run would pay that (~20s)
every time. Everything is loaded once and shared. Resumable: a (condition, gene) already
COMPLETED in MLflow is skipped, so an interrupted run is simply re-launched.

Per gene:
  baselines stage (no LLM, seconds)
    go_enrichment v1, v2   g:Profiler-style, cell types / rows   (positive direction, --universe called)
    go_enrichment v3, v4   corrected, cell types / rows
    go_enrichment v5, v6   corrected + co-annotation transfer (can name terms outside the vocabulary)
  constrained stage (LLM)
    go_llm v1-v4  evidence=true   +  v1-v4 evidence=mismatched  +  v3 evidence=none      = 9 runs
  freeform stage (LLM) — the PRIMARY comparison
    go_llm v1-v4  evidence=true   +  v3, v4 evidence=mismatched  +  v3 evidence=none     = 7 runs
The mismatched/none controls for the no-LLM rungs are computed deterministically at
analysis time (scripts/analyze_go_experiment.py), not as runs.

Genes come from data/go_experiment/genes.tsv (scripts/select_go_genes.py). Run the
development split first, freeze the prompts, then the test split.

Usage:
    python scripts/run_go_experiment.py --split dev --stage baselines
    python scripts/run_go_experiment.py --split dev --stage constrained --limit 10
    python scripts/run_go_experiment.py --split test --stage all
    python scripts/run_go_experiment.py --split smoke --stage all --dry-run
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[1] / "approaches" / "go_enrichment"))
sys.path.insert(0, str(Path(__file__).parents[1] / "approaches" / "go_llm"))

import mlflow

import run_go_enrichment as A1
import run_go_llm as LLM
from core.go_experiment import GOShared
from core.mlflow_utils import _TRACKING_URI

GENES_FILE = Path(__file__).parents[1] / "data" / "go_experiment" / "genes.tsv"
MAX_CONSECUTIVE_FAILURES = 5


def conditions_for(stage: str) -> list[dict]:
    """The full condition list for a stage (see module docstring)."""
    out = []
    if stage in ("baselines", "all"):
        out += [{"kind": "go_enrichment", "version": v} for v in ("v1", "v2", "v3", "v4", "v5", "v6")]
    if stage in ("constrained", "all"):
        out += [{"kind": "go_llm", "version": v, "mode": "constrained", "evidence": "true"} for v in ("v1", "v2", "v3", "v4")]
        out += [{"kind": "go_llm", "version": v, "mode": "constrained", "evidence": "mismatched"} for v in ("v1", "v2", "v3", "v4")]
        out += [{"kind": "go_llm", "version": "v3", "mode": "constrained", "evidence": "none"}]
    if stage in ("freeform", "all"):
        out += [{"kind": "go_llm", "version": v, "mode": "freeform", "evidence": "true"} for v in ("v1", "v2", "v3", "v4")]
        out += [{"kind": "go_llm", "version": v, "mode": "freeform", "evidence": "mismatched"} for v in ("v3", "v4")]
        out += [{"kind": "go_llm", "version": "v3", "mode": "freeform", "evidence": "none"}]
    return out


def condition_key(c: dict) -> str:
    if c["kind"] == "go_enrichment":
        return f"go_enrichment:{c['version']}:{'contrast' if c['version'] in ('v3', 'v4', 'v5', 'v6') else 'positive'}"
    return f"go_llm:{c['version']}:{c['mode']}:{c['evidence']}"


def completed(split: str) -> set[tuple[str, str]]:
    """(condition, gene_id) already COMPLETED for this split, one query per experiment."""
    mlflow.set_tracking_uri(_TRACKING_URI)
    done: set[tuple[str, str]] = set()
    for name in ("CellCoLLM/go_enrichment", "CellCoLLM/go_llm"):
        exp = mlflow.get_experiment_by_name(name)
        if exp is None:
            continue
        df = mlflow.search_runs([exp.experiment_id], max_results=50000,
                                filter_string=f"tags.status = 'COMPLETED' and tags.gene_split = '{split}'")
        if len(df):
            done |= set(zip(df["tags.condition"], df["tags.gene_id"]))
    return done


def read_genes(path: Path, split: str, stratum: str | None) -> list[dict]:
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t") if r["split"] == split]
    if stratum:
        rows = [r for r in rows if r["stratum"] == stratum]
    return rows


def server_reachable() -> bool:
    import httpx
    from core.llm_backend import ollama_base_url, ollama_headers
    try:
        return httpx.get(ollama_base_url().rstrip("/") + "/api/tags", headers=ollama_headers(), timeout=20).status_code == 200
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=["dev", "test", "smoke"])
    ap.add_argument("--stage", default="all", choices=["baselines", "constrained", "freeform", "all"])
    ap.add_argument("--genes-file", default=str(GENES_FILE))
    ap.add_argument("--stratum", default=None, choices=["carrying", "not_carrying"], help="Restrict to one stratum")
    ap.add_argument("--limit", type=int, default=None, help="At most this many genes")
    ap.add_argument("--model", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="List what would run, run nothing")
    args = ap.parse_args()

    if args.split == "smoke":
        genes = [{"gene": g, "symbol": s, "stratum": ""} for g, s in
                 [("ENSG00000132763", "MMACHC"), ("ENSG00000129696", "TTI2"), ("ENSG00000149554", "CHEK1")]]
    else:
        if not Path(args.genes_file).exists():
            raise SystemExit(f"{args.genes_file} not found. Run: python scripts/select_go_genes.py")
        genes = read_genes(Path(args.genes_file), args.split, args.stratum)
    if args.limit:
        genes = genes[: args.limit]
    conds = conditions_for(args.stage)
    needs_llm = any(c["kind"] == "go_llm" for c in conds)
    print(f"split={args.split} stage={args.stage}: {len(genes)} genes x {len(conds)} conditions "
          f"= {len(genes) * len(conds)} runs")
    if args.dry_run:
        for c in conds:
            print("  ", condition_key(c))
        return
    if needs_llm and not server_reachable():
        raise SystemExit("The LLM server is unreachable (GET /api/tags failed). Baseline stage does not need it: "
                         "--stage baselines. Otherwise retry when the server is back; nothing has been run.")

    print("Loading shared state (dataset, ontology, annotations, baselines)...")
    shared = GOShared.load()
    done = completed(args.split)
    print(f"  {len(done)} (condition, gene) pairs already COMPLETED for split={args.split}; these are skipped")

    t0, n_run, n_skip, n_fail, consecutive = time.time(), 0, 0, 0, 0
    total = len(genes) * len(conds)
    for gi, gene in enumerate(genes, 1):
        gid = gene["gene"]
        sym = shared.symbol(gid)
        if not shared.ds.has_gene(gid) or not shared.truth.has(gid):
            print(f"[{gi}/{len(genes)}] {gid}: not in dataset or not scoreable — skipped")
            continue
        for c in conds:
            key = condition_key(c)
            if (key, gid) in done:
                n_skip += 1
                continue
            try:
                if c["kind"] == "go_enrichment":
                    a = A1.make_args(gene=gid, prompt_version=c["version"], universe="called",
                                     input_set="positive", gene_split=args.split, seed=args.seed)
                    A1.run_gene(gid, sym, a, shared)
                else:
                    a = LLM.make_args(gene=gid, prompt_version=c["version"], output_mode=c["mode"],
                                      evidence=c["evidence"], gene_split=args.split, seed=args.seed, model=args.model)
                    LLM.run_gene(gid, sym, a, shared)
                n_run += 1
                consecutive = 0
            except Exception as exc:  # a failed condition must not stop the batch; SystemExit (blinding) still does
                n_fail += 1
                consecutive += 1
                print(f"  FAILED {key} {gid}: {type(exc).__name__}: {exc}")
                if consecutive >= MAX_CONSECUTIVE_FAILURES:
                    raise SystemExit(f"{MAX_CONSECUTIVE_FAILURES} consecutive failures — aborting (server down?). "
                                     f"Re-run the same command to resume.")
        elapsed = time.time() - t0
        finished = n_run + n_skip + n_fail
        eta = elapsed / max(n_run, 1) * (total - finished) if n_run else float("nan")
        print(f"[{gi}/{len(genes)}] {sym}: run {n_run}, skipped {n_skip}, failed {n_fail}  "
              f"elapsed {elapsed / 60:.1f} min, ETA ~{eta / 60:.0f} min")
    print(f"\nDone: {n_run} run, {n_skip} skipped, {n_fail} failed.")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
