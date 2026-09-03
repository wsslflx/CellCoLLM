#!/usr/bin/env python3
"""
Batch-run the fixed set of standard test genes through one or more
approach/version combinations, instead of invoking each run_*.py by hand.

Each --run value is "approach:version" (e.g. "naive:v1", "enriched:v2").
The version string is forwarded as-is to the underlying script's
--prompt-version — this script has no knowledge of which versions exist, so
future versions (v3, ...) work without any change here as long as they're
registered in the approach's own PROMPT_VERSIONS.

Every combo is run for every gene with --set both and otherwise-default
options (model from CHAT_MODEL/.env, etc.) — no other flags are exposed here
by design; add passthrough later if a batch run actually needs to vary them.

Optionally, pass --score-go-match --judge-model <model> to also run the
GO-match validation layer (validation/score_go_match.py) against every
positive/negative run this script just produced — the resulting run_ids are
looked up from MLflow (by approach/approach_version/gene_id/input_set,
most recent), not parsed from subprocess output.

Usage:
    python scripts/run_test_genes.py --run naive:v1 --run enriched:v2
    python scripts/run_test_genes.py --run enriched:v1 --gene ENSG00000001626
    python scripts/run_test_genes.py --run enriched:v2 --score-go-match --judge-model llama3.3:70b
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]

DEFAULT_TEST_GENES = ["ENSG00000132763", "ENSG00000129696", "ENSG00000149554"]

APPROACH_SCRIPTS = {
    "naive": REPO_ROOT / "approaches" / "naive" / "run_naive.py",
    "enriched": REPO_ROOT / "approaches" / "enriched" / "run_enriched.py",
}
SCORE_GO_MATCH_SCRIPT = REPO_ROOT / "validation" / "score_go_match.py"


def parse_run(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError(f"--run must be 'approach:version', got {value!r}")
    approach, version = value.split(":", 1)
    if approach not in APPROACH_SCRIPTS:
        raise argparse.ArgumentTypeError(
            f"Unknown approach {approach!r} in --run {value!r}. Known approaches: {sorted(APPROACH_SCRIPTS)}"
        )
    return approach, version


def find_recent_run_ids(approach: str, approach_version: str, gene_id: str) -> dict[str, str]:
    """Look up the most recent positive/negative run_ids for this combo, via
    MLflow tags — not parsed from subprocess output. Returns {input_set: run_id}."""
    import mlflow

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    exp = mlflow.get_experiment_by_name(f"CellCoLLM/{approach}")
    if exp is None:
        return {}
    found = {}
    for input_set in ("positive", "negative"):
        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=(
                f"tags.approach_version = '{approach_version}' and tags.gene_id = '{gene_id}' "
                f"and tags.input_set = '{input_set}' and tags.status = 'COMPLETED'"
            ),
            max_results=1, order_by=["start_time DESC"],
        )
        if len(runs) > 0:
            found[input_set] = runs.iloc[0]["run_id"]
    return found


def run_go_match_scoring(run_id: str, judge_model: str, label: str) -> bool:
    print(f"\n{'-' * 80}\nscoring {label}  run_id={run_id}\n{'-' * 80}", flush=True)
    proc = subprocess.run(
        [sys.executable, str(SCORE_GO_MATCH_SCRIPT), "--run-id", run_id, "--judge-model", judge_model],
        cwd=REPO_ROOT,
    )
    return proc.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", dest="runs", action="append", type=parse_run, required=True,
                         help="'approach:version', e.g. --run naive:v1 --run enriched:v2 (repeatable)")
    parser.add_argument("--gene", dest="genes", action="append", default=None,
                         help=f"Override the default test genes (repeatable). Default: {DEFAULT_TEST_GENES}")
    parser.add_argument("--score-go-match", action="store_true",
                         help="Also run validation/score_go_match.py against every run this script produces")
    parser.add_argument("--judge-model", default=None,
                         help="Required if --score-go-match is set; must differ from each combo's generator model")
    args = parser.parse_args()
    if args.score_go_match and not args.judge_model:
        parser.error("--score-go-match requires --judge-model")

    genes = args.genes or DEFAULT_TEST_GENES
    results = []
    score_results = []

    for approach, version in args.runs:
        script = APPROACH_SCRIPTS[approach]
        for gene in genes:
            print(f"\n{'=' * 80}\n{approach}:{version}  gene={gene}\n{'=' * 80}", flush=True)
            t0 = time.time()
            proc = subprocess.run(
                [sys.executable, str(script), "--gene", gene, "--set", "both", "--prompt-version", version],
                cwd=REPO_ROOT,
            )
            elapsed = time.time() - t0
            ok = proc.returncode == 0
            results.append((approach, version, gene, ok, elapsed))

            if ok and args.score_go_match:
                run_ids = find_recent_run_ids(approach, version, gene)
                for input_set, run_id in run_ids.items():
                    label = f"{approach}:{version}  {gene}  {input_set}"
                    score_results.append((label, run_go_match_scoring(run_id, args.judge_model, label)))

    print(f"\n{'=' * 80}\nSummary\n{'=' * 80}")
    for approach, version, gene, ok, elapsed in results:
        status = "OK" if ok else "FAILED"
        print(f"  [{status:6}] {approach}:{version}  {gene}  ({elapsed:.1f}s)")
    if score_results:
        print("\nGO-match scoring:")
        for label, ok in score_results:
            print(f"  [{'OK' if ok else 'FAILED':6}] {label}")

    if any(not ok for *_, ok, _ in results) or any(not ok for _, ok in score_results):
        sys.exit(1)


if __name__ == "__main__":
    main()
