#!/usr/bin/env python3
"""
Run identity for the GO-prediction experiment: which exact code, data and gene list produced a run.

Why this exists. The git tag on a run is `commit` or `commit-dirty`. Once the working tree is dirty, two
different prompt versions carry the same tag, so "did all runs come from the same code?" is unanswerable,
and a resumable runner keyed only on (condition, gene) would silently keep results produced by an OLD
prompt. So every run is tagged with `code_hash`: a hash of the CONTENT of the files that shape its
output, per approach —

    go_enrichment   run script + statistics/evidence/scoring/transfer/ontology code
    go_llm          run script + ALL prompt templates + evidence/scoring/ontology code + LLM plumbing

Consumers:
  scripts/run_go_experiment.py   skips a condition only if COMPLETED under the CURRENT hash
  scripts/analyze_go_experiment.py  uses only runs from the current hash (refuses to mix)
  scripts/freeze_go_experiment.py   writes the freeze manifest; the test split will not run unless the
                                    code, the gene list and the data all still match it

Editing a README, a docstring-free comment elsewhere, or an unrelated approach does not change a hash.
Editing a prompt template, a scoring function or the evidence code does.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).parents[1]
FREEZE_PATH = ROOT / "approaches" / "go_experiment_freeze.json"

# Paths (globs allowed) relative to the repo root. Anything that can change a run's output belongs here.
SPECS: dict[str, list[str]] = {
    "go_enrichment": [
        "approaches/go_enrichment/run_go_enrichment.py",
        "core/go_enrichment.py", "core/go_evidence.py", "core/go_scoring.py",
        "core/go_ontology.py", "core/go_transfer.py", "core/go_experiment.py",
    ],
    "go_llm": [
        "approaches/go_llm/run_go_llm.py", "approaches/go_llm/prompts/*.txt",
        "core/go_evidence.py", "core/go_scoring.py", "core/go_ontology.py", "core/go_experiment.py",
        "core/structured_llm.py", "core/llm_backend.py",
    ],
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_hashes(approach: str) -> dict[str, str]:
    """{relative path: sha256} for every file in the approach's spec."""
    out: dict[str, str] = {}
    for pattern in SPECS[approach]:
        for p in sorted(ROOT.glob(pattern)):
            out[str(p.relative_to(ROOT))] = _sha256(p)
    return out


@lru_cache(maxsize=None)
def fingerprint(approach: str) -> str:
    """12-hex code hash of the approach's files. Cached: files do not change within a process."""
    return hashlib.sha256(json.dumps(file_hashes(approach), sort_keys=True).encode()).hexdigest()[:12]


def fingerprints() -> dict[str, str]:
    return {a: fingerprint(a) for a in SPECS}


def git_state() -> dict:
    def run(*cmd):
        return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
    try:
        return {"head": run("git", "rev-parse", "HEAD"), "dirty": bool(run("git", "status", "--porcelain"))}
    except Exception:
        return {"head": "unknown", "dirty": None}


@lru_cache(maxsize=None)
def _cached_sha(path: str) -> str:
    return _sha256(Path(path))


def data_state(shared, genes_file: str | Path | None) -> dict:
    """Everything OTHER than code that determines results: data versions and the gene list."""
    from core.go_evidence import baselines_key
    from core.go_ontology import GAF_PATH
    return {
        "dataset_hash": shared.ds.dataset_hash,
        "baselines_key": baselines_key(shared.builder),
        "cl_data_version": shared.lookup.provenance.get("cl_data_version"),
        "go_data_version": shared.truth.go.data_version if shared.truth else None,
        "gaf_sha256": _cached_sha(str(GAF_PATH)),
        "genes_file_sha256": _cached_sha(str(genes_file)) if genes_file and Path(genes_file).exists() else None,
    }


def write_freeze(shared, genes_file: str | Path, note: str = "", path: Path = FREEZE_PATH) -> dict:
    manifest = {
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": note,
        "git": git_state(),
        "fingerprints": fingerprints(),
        "files": {a: file_hashes(a) for a in SPECS},
        "data": data_state(shared, genes_file),
    }
    Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def verify_freeze(shared, genes_file: str | Path | None, path: Path = FREEZE_PATH) -> list[str]:
    """Problems, human-readable; an empty list means everything matches the freeze."""
    path = Path(path)
    if not path.exists():
        return [f"no freeze manifest at {path} — run: python scripts/freeze_go_experiment.py"]
    m = json.loads(path.read_text())
    problems: list[str] = []
    for approach in SPECS:
        if fingerprint(approach) != m["fingerprints"].get(approach):
            now, then = file_hashes(approach), m["files"].get(approach, {})
            changed = sorted({f for f in set(now) | set(then) if now.get(f) != then.get(f)})
            problems.append(f"{approach} code changed since the freeze ({m['fingerprints'].get(approach)} -> "
                            f"{fingerprint(approach)}): {', '.join(changed)}")
    now_data = data_state(shared, genes_file)
    for k, v in m["data"].items():
        if now_data.get(k) != v:
            problems.append(f"{k} changed since the freeze ({str(v)[:16]} -> {str(now_data.get(k))[:16]})")
    return problems
