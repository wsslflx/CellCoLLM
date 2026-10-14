"""
Fast access to arm 2's (go_llm) results for analysis work, without MLflow.

The tables come from scripts/export_arm2_cache.py (run where the MLflow database is, then copied here):

    from core.arm2_cache import load_arm2_cache
    c = load_arm2_cache()          # default condition go_llm:v3:freeform:true, split all
    c.runs         one row per gene that has ANY run (status, scores, latency, hypothesis, ...)
    c.done         the genes that are finished under the current code (status COMPLETED, current_code)
    c.predictions  one row per (finished gene, predicted term) with match kind and IC gap
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[1]
CACHE_DIR = ROOT / "data" / "go_experiment" / "cache"
DEFAULT_CONDITION = "go_llm:v3:freeform:true"


def safe_name(condition: str) -> str:
    return condition.replace(":", "_")


@dataclass
class Arm2Cache:
    runs: pd.DataFrame
    predictions: pd.DataFrame
    meta: dict

    @property
    def done(self) -> pd.DataFrame:
        return self.runs[(self.runs["status"] == "COMPLETED") & self.runs["current_code"]]


def load_arm2_cache(condition: str = DEFAULT_CONDITION, split: str = "all", cache_dir: Path | None = None) -> Arm2Cache:
    d = Path(cache_dir) if cache_dir else CACHE_DIR
    name = f"{safe_name(condition)}__{split}"
    runs_path = d / f"arm2_runs__{name}.parquet"
    if not runs_path.exists():
        raise FileNotFoundError(f"{runs_path} missing -- run scripts/export_arm2_cache.py on the machine with the MLflow database "
                                "and copy the arm2_*.parquet files here")
    return Arm2Cache(pd.read_parquet(runs_path), pd.read_parquet(d / f"arm2_predictions__{name}.parquet"),
                     json.loads((d / f"arm2_cache_meta__{name}.json").read_text()))
