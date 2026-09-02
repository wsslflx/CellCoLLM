#!/usr/bin/env python3
"""
Shared MLflow logging schema for CellCoLLM approaches.

Every approach (naive today, full pipeline later) must go through
`tracked_run(...)` so runs stay comparable across approaches — the fixed
core fields below are what makes a later "did the expensive version beat
the naive baseline, on the same gene?" query possible at all. Anything
approach-specific (coverage, leakage, cross-species similarity, ...) is
logged by the caller inside the `with` block using plain mlflow calls.

See PIPELINE_REQUIREMENTS.md §7.6 (experiment tracking) and §7.3 (the
artifact schema this mirrors — config_hash, provenance, etc.).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import mlflow

_PROJECT_ROOT = Path(__file__).parents[1]
# MLflow 3.x deprecated the plain filesystem store; sqlite is the documented
# lightweight replacement (PIPELINE_REQUIREMENTS.md §7.6).
_TRACKING_URI = f"sqlite:///{_PROJECT_ROOT / 'mlflow.db'}"


def _experiment_name(approach: str) -> str:
    return f"CellCoLLM/{approach}"


def git_commit_sha() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_PROJECT_ROOT, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_PROJECT_ROOT, capture_output=True, text=True, timeout=5,
        ).stdout.strip())
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        return "unknown"


def compute_config_hash(**fields: Any) -> str:
    """Stable short hash over identifying config fields (order-independent)."""
    payload = json.dumps(fields, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def verify_blinding(text: str, forbidden_terms: list[str]) -> tuple[bool, list[str]]:
    """Case-insensitive scan for terms that must never reach an LLM prompt (gene symbol, aliases, ...)."""
    lowered = text.lower()
    hits = [t for t in forbidden_terms if t and t.lower() in lowered]
    return (len(hits) == 0, hits)


@dataclass
class RunContext:
    """Identifying fields every approach must supply. Extend via `extra_params`/`extra_tags`, never by skipping these."""
    approach: str
    approach_version: str
    gene_id: str
    gene_symbol: str
    species: str
    model: str
    prompt_mode: str
    input_set: str  # which cell-type set this run reasons over, e.g. "positive" / "negative"
    mode: str = "naive_exploration"
    temperature: float = 0.0
    seed: int | None = None
    dataset_hash: str | None = None
    prompt_version: str | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)
    extra_tags: dict[str, Any] = field(default_factory=dict)


def get_or_create_gene_parent_run(
    approach: str, gene_id: str, gene_symbol: str, species: str,
    dataset_hash: str | None = None, expression_summary: dict | None = None,
) -> str:
    """
    Return the run_id of a grouping "gene parent" run under experiment
    `CellCoLLM/{approach}` for this gene — reused across separate script
    invocations (e.g. a positive-set run today, a negative-set run later)
    so both land under the same parent and are grouped/comparable in the UI.
    """
    mlflow.set_tracking_uri(_TRACKING_URI)
    exp_name = _experiment_name(approach)
    mlflow.set_experiment(exp_name)
    experiment = mlflow.get_experiment_by_name(exp_name)

    existing = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.role = 'gene_parent' and tags.gene_id = '{gene_id}'",
        max_results=1,
    )
    if len(existing) > 0:
        return existing.iloc[0]["run_id"]

    with mlflow.start_run(run_name=f"{approach}_{gene_symbol}_gene") as run:
        mlflow.log_params({"gene_id": gene_id, "species": species, "dataset_hash": dataset_hash})
        if expression_summary:
            mlflow.log_metrics(expression_summary)
        mlflow.set_tags({
            "role": "gene_parent",
            "gene_id": gene_id,
            "gene_symbol": gene_symbol,
            "approach": approach,
            "git_commit": git_commit_sha(),
        })
        return run.info.run_id


@contextmanager
def tracked_run(ctx: RunContext, parent_run_id: str | None = None) -> Iterator[Any]:
    """
    Start an MLflow run under experiment `CellCoLLM/{ctx.approach}`, logging the
    fixed core schema, then yield the active run for approach-specific logging
    (mlflow.log_metric / log_artifact / etc.) inside the `with` block.

    If `parent_run_id` is given (see `get_or_create_gene_parent_run`), the run
    is linked as its child via the `mlflow.parentRunId` tag — this works even
    across separate process invocations, unlike `nested=True`, which requires
    the parent run to still be active in the same process.
    """
    mlflow.set_tracking_uri(_TRACKING_URI)
    mlflow.set_experiment(_experiment_name(ctx.approach))

    config_hash = compute_config_hash(
        gene_id=ctx.gene_id, species=ctx.species, approach=ctx.approach,
        approach_version=ctx.approach_version, model=ctx.model,
        temperature=ctx.temperature, seed=ctx.seed, prompt_mode=ctx.prompt_mode,
        input_set=ctx.input_set, prompt_version=ctx.prompt_version, dataset_hash=ctx.dataset_hash,
        extra_params=sorted(ctx.extra_params.items()),
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"{ctx.approach}_{ctx.input_set}_{ctx.gene_symbol}_{timestamp}"

    tags = {
        "approach": ctx.approach,
        "approach_version": ctx.approach_version,
        "gene_id": ctx.gene_id,
        "gene_symbol": ctx.gene_symbol,
        "prompt_mode": ctx.prompt_mode,
        "input_set": ctx.input_set,
        "mode": ctx.mode,
        "git_commit": git_commit_sha(),
        "config_hash": config_hash,
        **ctx.extra_tags,
    }
    if parent_run_id:
        tags["mlflow.parentRunId"] = parent_run_id

    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        mlflow.log_params({
            "gene_id": ctx.gene_id,
            "species": ctx.species,
            "model": ctx.model,
            "temperature": ctx.temperature,
            "seed": ctx.seed,
            "dataset_hash": ctx.dataset_hash,
            "prompt_version": ctx.prompt_version,
            **ctx.extra_params,
        })
        yield run


def log_text_artifact(text: str, filename: str, artifact_path: str | None = None) -> None:
    """Log a string as a text-file artifact on the active run (prompt, response, ...)."""
    mlflow.log_text(text, artifact_file=f"{artifact_path}/{filename}" if artifact_path else filename)


def log_json_artifact(obj: Any, filename: str, artifact_path: str | None = None) -> None:
    mlflow.log_dict(obj, f"{artifact_path}/{filename}" if artifact_path else filename)
