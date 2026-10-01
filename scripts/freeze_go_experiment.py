#!/usr/bin/env python3
"""
Freeze the GO-prediction pipeline before the test split is run.

All prompt wording, top-N, ranking and threshold choices are made on the dev split. When they are final, this
writes approaches/go_experiment_freeze.json — a manifest of

  * the code fingerprint of each approach (content hash of the prompts, scoring, evidence and run code),
  * the sha256 of every file behind those fingerprints (so a later mismatch names the file that changed),
  * the data the results depend on: dataset hash, baselines key, CL and GO versions, GOA file hash,
  * the sha256 of the gene list (so the test genes cannot change after the fact),
  * the git HEAD and whether the tree was dirty.

The manifest is meant to be COMMITTED. scripts/run_go_experiment.py --split test refuses to run unless the
current code, data and gene list still match it, and scripts/analyze_go_experiment.py --eval-split test
reports the same check. Changing anything after freezing therefore requires a deliberate re-freeze, which
shows up in the git history.

Usage:
    python scripts/freeze_go_experiment.py --note "prompts v1 frozen after dev round 4"
    python scripts/freeze_go_experiment.py --check        # is the current state still the frozen one?
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from core.go_experiment import GOShared
from core.run_identity import FREEZE_PATH, fingerprints, verify_freeze, write_freeze

GENES_FILE = Path(__file__).parents[1] / "data" / "go_experiment" / "genes.tsv"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genes-file", default=str(GENES_FILE))
    ap.add_argument("--note", default="", help="Free text stored in the manifest (what was decided, on which dev round)")
    ap.add_argument("--check", action="store_true", help="Only compare the current state with the existing freeze")
    args = ap.parse_args()

    print("Loading shared state...")
    shared = GOShared.load()
    if args.check:
        problems = verify_freeze(shared, args.genes_file)
        if problems:
            print("NOT MATCHING the freeze:\n  - " + "\n  - ".join(problems))
            sys.exit(1)
        print(f"OK — the current code, data and gene list match {FREEZE_PATH.name}")
        return
    if not Path(args.genes_file).exists():
        raise SystemExit(f"{args.genes_file} not found. Run: python scripts/select_go_genes.py")
    if FREEZE_PATH.exists():
        old = json.loads(FREEZE_PATH.read_text())
        print(f"Replacing the freeze from {old['frozen_at']} ({old.get('note') or 'no note'}).")
    m = write_freeze(shared, args.genes_file, args.note)
    print(f"Wrote {FREEZE_PATH}")
    for a, h in fingerprints().items():
        print(f"  {a:14s} {h}")
    print(f"  git HEAD {m['git']['head'][:8]}, tree {'DIRTY' if m['git']['dirty'] else 'clean'}"
          f"{' — commit first, so the freeze can be tied to a commit' if m['git']['dirty'] else ''}")
    print("Commit approaches/go_experiment_freeze.json together with the code it describes.")


if __name__ == "__main__":
    main()
