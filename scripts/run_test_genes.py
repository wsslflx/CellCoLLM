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

Optionally run a validation layer against every positive/negative run this
script just produced (run_ids are looked up from MLflow by
approach/approach_version/gene_id/input_set, not parsed from subprocess output):

  --score-discrimination  the PRIMARY layer (validation/score_discrimination.py):
                          coverage/leakage/J against the dataset's own ground
                          truth, with the grounding ablation. Works on any gene.
  --score-go-match        the SECONDARY layer (validation/score_go_match.py):
                          benchmark-only GO comparison, meaningful only for
                          well-annotated genes. See its caveats before relying on it.

Either requires --judge-model, which must differ from the generator model.

Usage:
    python scripts/run_test_genes.py --run naive:v1 --run enriched:v2
    python scripts/run_test_genes.py --run enriched:v1 --gene ENSG00000001626
    python scripts/run_test_genes.py --run enriched:v2 --score-discrimination --judge-model llama3.3:70b
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
    "statistical": REPO_ROOT / "approaches" / "statistical" / "run_statistical.py",
    "go_enrichment": REPO_ROOT / "approaches" / "go_enrichment" / "run_go_enrichment.py",
    "go_llm": REPO_ROOT / "approaches" / "go_llm" / "run_go_llm.py",
}
# Approaches that produce a ranked GO term list rather than a property statement. They score
# themselves against the gene's own GO annotation inline (gopred_* metrics, same run), so the
# judge-based validation layers below do not apply.
UNSCORABLE_APPROACHES = {"go_enrichment", "go_llm"}
# naive/enriched produce a positive and a negative run per gene; statistical computes
# the contrast internally and produces a single run tagged input_set="contrast".
INPUT_SETS = ("positive", "negative", "contrast")
SCORE_GO_MATCH_SCRIPT = REPO_ROOT / "validation" / "score_go_match.py"
SCORE_DISCRIMINATION_SCRIPT = REPO_ROOT / "validation" / "score_discrimination.py"


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
    for input_set in INPUT_SETS:
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


def run_scorer(script: Path, run_id: str, judge_model: str, label: str, kind: str) -> bool:
    print(f"\n{'-' * 80}\n{kind} scoring {label}  run_id={run_id}\n{'-' * 80}", flush=True)
    proc = subprocess.run(
        [sys.executable, str(script), "--run-id", run_id, "--judge-model", judge_model],
        cwd=REPO_ROOT,
    )
    return proc.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", dest="runs", action="append", type=parse_run, required=True,
                         help="'approach:version', e.g. --run naive:v1 --run enriched:v2 (repeatable)")
    parser.add_argument("--gene", dest="genes", action="append", default=None,
                         help=f"Override the default test genes (repeatable). Default: {DEFAULT_TEST_GENES}")
    parser.add_argument("--score-discrimination", action="store_true",
                         help="Run the primary coverage/leakage validation layer on every run produced")
    parser.add_argument("--score-go-match", action="store_true",
                         help="Also run the secondary GO-match layer (benchmark-only; well-annotated genes)")
    parser.add_argument("--judge-model", default=None,
                         help="Required for either scorer; must differ from each combo's generator model")
    args = parser.parse_args()
    if (args.score_go_match or args.score_discrimination) and not args.judge_model:
        parser.error("--score-discrimination/--score-go-match require --judge-model")

    genes = args.genes or DEFAULT_TEST_GENES
    results = []
    score_results = []

    for approach, version in args.runs:
        script = APPROACH_SCRIPTS[approach]
        for gene in genes:
            print(f"\n{'=' * 80}\n{approach}:{version}  gene={gene}\n{'=' * 80}", flush=True)
            t0 = time.time()
            cmd = [sys.executable, str(script), "--gene", gene, "--prompt-version", version]
            # statistical/go_llm have no directions; go_enrichment v3/v4 compute the contrast internally
            # (--set is accepted and ignored there), so only naive/enriched/go_enrichment get it.
            if approach in ("naive", "enriched", "go_enrichment"):
                cmd += ["--set", "both"]
            proc = subprocess.run(cmd, cwd=REPO_ROOT)
            elapsed = time.time() - t0
            ok = proc.returncode == 0
            results.append((approach, version, gene, ok, elapsed))

            if approach in UNSCORABLE_APPROACHES and (args.score_discrimination or args.score_go_match):
                print(f"  (no judge-layer scoring for {approach} — it scores itself against the gene's "
                      f"own GO annotation inline)", flush=True)
            elif ok and (args.score_discrimination or args.score_go_match):
                run_ids = find_recent_run_ids(approach, version, gene)
                for input_set, run_id in run_ids.items():
                    label = f"{approach}:{version}  {gene}  {input_set}"
                    if args.score_discrimination:
                        score_results.append((f"[disc]    {label}", run_scorer(
                            SCORE_DISCRIMINATION_SCRIPT, run_id, args.judge_model, label, "discrimination")))
                    if args.score_go_match:
                        score_results.append((f"[gomatch] {label}", run_scorer(
                            SCORE_GO_MATCH_SCRIPT, run_id, args.judge_model, label, "GO-match")))

    print(f"\n{'=' * 80}\nSummary\n{'=' * 80}")
    for approach, version, gene, ok, elapsed in results:
        status = "OK" if ok else "FAILED"
        print(f"  [{status:6}] {approach}:{version}  {gene}  ({elapsed:.1f}s)")
    if score_results:
        print("\nValidation scoring:")
        for label, ok in score_results:
            print(f"  [{'OK' if ok else 'FAILED':6}] {label}")

    if any(not ok for *_, ok, _ in results) or any(not ok for _, ok in score_results):
        sys.exit(1)


if __name__ == "__main__":
    main()
