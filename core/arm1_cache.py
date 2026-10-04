"""
Fast access to arm 1's (go_enrichment v1) results for analysis work.

The tables are produced once by scripts/export_arm1_cache.py (and the analyze_arm1_* scripts); loading them
takes seconds, versus minutes for re-walking MLflow artifacts. Typical use in a notebook:

    from core.arm1_cache import load_arm1_cache
    c = load_arm1_cache()
    c.terms    # one row per (gene, tested term): p-values, K/n/N, reachability, match kind, IC gap, ...
    c.sig      # c.terms restricted to significant rows
    c.genes    # one row per gene: breadth, gene_group, ceiling, chance baseline, mean_true_ic, ...
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[1]
CACHE_DIR = ROOT / "data" / "go_experiment" / "cache"
SUMMARY_DIR = ROOT / "data" / "go_experiment" / "match_summaries"


@dataclass
class Arm1Cache:
    terms: pd.DataFrame
    genes: pd.DataFrame
    meta: dict

    @property
    def sig(self) -> pd.DataFrame:
        return self.terms[self.terms["significant"]]


def load_arm1_cache(split: str = "all") -> Arm1Cache:
    path = CACHE_DIR / f"arm1_tested_terms__{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run: python scripts/export_arm1_cache.py --gene-split {split}")
    genes = pd.DataFrame(json.loads((SUMMARY_DIR / f"arm1_chance_baseline__{split}.json").read_text()))
    meta = json.loads((CACHE_DIR / f"arm1_cache_meta__{split}.json").read_text())
    return Arm1Cache(pd.read_parquet(path), genes, meta)
