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

Usage:
    python scripts/run_test_genes.py --run naive:v1 --run enriched:v2
    python scripts/run_test_genes.py --run enriched:v1 --gene ENSG00000001626
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]

DEFAULT_TEST_GENES = ["ENSG00000012048", "ENSG00000129696", "ENSG00000149554"]

APPROACH_SCRIPTS = {
    "naive": REPO_ROOT / "approaches" / "naive" / "run_naive.py",
    "enriched": REPO_ROOT / "approaches" / "enriched" / "run_enriched.py",
}


def parse_run(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError(f"--run must be 'approach:version', got {value!r}")
    approach, version = value.split(":", 1)
    if approach not in APPROACH_SCRIPTS:
        raise argparse.ArgumentTypeError(
            f"Unknown approach {approach!r} in --run {value!r}. Known approaches: {sorted(APPROACH_SCRIPTS)}"
        )
    return approach, version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", dest="runs", action="append", type=parse_run, required=True,
                         help="'approach:version', e.g. --run naive:v1 --run enriched:v2 (repeatable)")
    parser.add_argument("--gene", dest="genes", action="append", default=None,
                         help=f"Override the default test genes (repeatable). Default: {DEFAULT_TEST_GENES}")
    args = parser.parse_args()

    genes = args.genes or DEFAULT_TEST_GENES
    results = []

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
            results.append((approach, version, gene, proc.returncode == 0, elapsed))

    print(f"\n{'=' * 80}\nSummary\n{'=' * 80}")
    for approach, version, gene, ok, elapsed in results:
        status = "OK" if ok else "FAILED"
        print(f"  [{status:6}] {approach}:{version}  {gene}  ({elapsed:.1f}s)")

    if any(not ok for *_, ok, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
