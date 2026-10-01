#!/usr/bin/env python3
"""
Measure the LLM's own run-to-run variance on an UNCHANGED prompt — the noise floor against which
any prompt-tuning change must be judged. Without this, a prompt-to-prompt difference in aggregate
score cannot be told apart from the model simply not being perfectly deterministic.

Runs the SAME condition (prompt, evidence, everything) on the SAME genes multiple times (temperature
0, same seed every time — so any difference is backend/sampling noise, not a deliberate variation),
tagged `gene_split="noise_<condition>"` so these throwaway repeats never collide with real dev/test/
smoke runs or their resumability bookkeeping, and are trivially filterable out of any real analysis.

Reports, per metric: the spread ACROSS REPEATS of the gene-level score (within-gene noise) and,
more importantly, the spread across repeats of the AGGREGATE (mean-over-genes) score — because the
aggregate is what a prompt-tuning decision is actually based on. A prompt change smaller than this
aggregate spread is not a safe basis for a decision.

Usage:
    python scripts/measure_llm_noise.py --condition go_llm:v3:freeform:true --split dev --limit 20 --repeats 3
"""
from __future__ import annotations

import argparse
import json
import statistics as stats
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
from core.run_identity import fingerprint
from scripts.run_go_experiment import parse_condition, read_genes

OUT_DIR = Path(__file__).parents[1] / "data" / "go_experiment" / "noise_checks"
METRICS = ("gopred_f1_at_1", "gopred_f1_at_3", "gopred_f1_at_5", "gopred_f1_at_10", "frac_in_table", "n_invalid")


def run_once(c: dict, gene_id: str, sym: str, shared: GOShared, repeat: int, split_tag: str) -> dict | None:
    if c["kind"] == "go_enrichment":
        a = A1.make_args(gene=gene_id, prompt_version=c["version"], universe="called",
                         input_set="positive", gene_split=split_tag, seed=42)
        A1.run_gene(gene_id, sym, a, shared)
        condition = f"go_enrichment:{c['version']}:{'contrast' if c['version'] in ('v3', 'v4', 'v5', 'v6') else 'positive'}"
    else:
        # cache_bust forces a genuinely fresh LLM computation per repeat — without it, Ollama's
        # exact-prompt cache makes repeats 2+ deterministic echoes of repeat 1, not independent draws
        # (confirmed empirically: 20/20 test genes gave bit-identical repeat2==repeat3 without this).
        a = LLM.make_args(gene=gene_id, prompt_version=c["version"], output_mode=c["mode"],
                          evidence=c["evidence"], gene_split=split_tag, seed=42,
                          cache_bust=f"{split_tag}-{gene_id}")
        run_id = LLM.run_gene(gene_id, sym, a, shared)
        if run_id is None:
            return None
        condition = f"go_llm:{c['version']}:{c['mode']}:{c['evidence']}"
    mlflow.set_tracking_uri(_TRACKING_URI)
    df = mlflow.search_runs(
        [mlflow.get_experiment_by_name(f"CellCoLLM/{c['kind']}").experiment_id], max_results=1,
        filter_string=f"tags.condition = '{condition}' and tags.gene_id = '{gene_id}' and tags.gene_split = '{split_tag}'",
        order_by=["start_time DESC"])
    if not len(df):
        return None
    row = df.iloc[0]
    return {m: row.get(f"metrics.{m}") for m in METRICS}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", required=True, type=parse_condition)
    ap.add_argument("--genes-file", default=str(Path(__file__).parents[1] / "data" / "go_experiment" / "genes.tsv"))
    ap.add_argument("--split", default="dev", choices=["dev", "test", "smoke"])
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    cond_str = f"{args.condition['kind']}:{args.condition.get('version')}" + (
        f":{args.condition.get('mode')}:{args.condition.get('evidence')}" if args.condition["kind"] == "go_llm" else "")
    # include the code hash so a tuning round never collides with (or silently reuses) a previous
    # round's gene_split tags -- each prompt/code version gets its own clearly-separate noise-check runs.
    code_tag = fingerprint(args.condition["kind"])
    safe = f"{cond_str.replace(':', '_')}_{code_tag}"

    if args.split == "smoke":
        genes = [{"gene": g, "symbol": s} for g, s in
                 [("ENSG00000132763", "MMACHC"), ("ENSG00000129696", "TTI2"), ("ENSG00000149554", "CHEK1")]]
    else:
        genes = read_genes(Path(args.genes_file), args.split, None)
    genes = genes[: args.limit]
    print(f"Measuring noise floor for {cond_str} over {len(genes)} genes x {args.repeats} repeats "
          f"(same prompt, temperature 0, seed 42, every time)")

    print("Loading shared state...")
    needs_llm = args.condition["kind"] == "go_llm"
    if needs_llm:
        import httpx
        from core.llm_backend import ollama_base_url, ollama_headers
        if httpx.get(ollama_base_url().rstrip("/") + "/api/tags", headers=ollama_headers(), timeout=20).status_code != 200:
            raise SystemExit("LLM server unreachable.")
    shared = GOShared.load()

    per_repeat: list[dict[str, list[float]]] = [{m: [] for m in METRICS} for _ in range(args.repeats)]
    for gene in genes:
        gid, sym = gene["gene"], gene.get("symbol") or shared.symbol(gene["gene"])
        if not shared.ds.has_gene(gid) or not shared.truth.has(gid):
            continue
        for r in range(args.repeats):
            split_tag = f"noise_{safe}_r{r + 1}"
            # The LLM server has dropped connections mid-call a couple of times during this project
            # (httpcore.RemoteProtocolError). A single transient drop shouldn't void a ~10-minute batch,
            # so retry a few times with a short backoff before giving up on this one gene/repeat.
            row = None
            for attempt in range(3):
                try:
                    row = run_once(args.condition, gid, sym, shared, r, split_tag)
                    break
                except Exception as exc:
                    print(f"  {sym} repeat {r + 1} attempt {attempt + 1}/3 failed: {type(exc).__name__}: {exc}")
                    if attempt < 2:
                        time.sleep(5)
            if row is None:
                print(f"  {sym} repeat {r + 1}: FAILED / no score after retries")
                continue
            for m in METRICS:
                if row[m] is not None:
                    per_repeat[r][m].append(row[m])
        print(f"  {sym}: done")

    print(f"\n=== Noise floor: {cond_str}, {len(genes)} genes, {args.repeats} repeats ===")
    print(f"{'metric':20s}" + "".join(f"repeat {i + 1:>8d}" for i in range(args.repeats)) + f"{'spread':>10s}")
    report = {"condition": cond_str, "n_genes": len(genes), "repeats": args.repeats, "metrics": {}}
    for m in METRICS:
        means = [stats.mean(per_repeat[r][m]) if per_repeat[r][m] else float("nan") for r in range(args.repeats)]
        spread = max(means) - min(means) if all(x == x for x in means) else float("nan")  # nan-safe
        print(f"{m:20s}" + "".join(f"{x:14.4f}" for x in means) + f"{spread:10.4f}")
        report["metrics"][m] = {"per_repeat_mean": means, "aggregate_spread": spread}
    print("\n'spread' = max-min of the AGGREGATE (mean-over-genes) across repeats — the number any")
    print("prompt-tuning change has to beat before it's distinguishable from noise.")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    out_path = out / f"noise_{safe}_{args.split}.json"
    out_path.write_text(json.dumps(report, indent=2, default=float))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
