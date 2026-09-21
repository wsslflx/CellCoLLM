#!/usr/bin/env python3
"""
Per-feature baseline positive-rates, computed over EVERY gene in the dataset.

Why this exists: Fisher's exact test (statistical v1) assumes a feature is
distributed between a gene's positive and negative sets in proportion to their
sizes. That is measurably false — features carry systematic detection
differences. Across 40 random genes, tissues spanned a ~40-point baseline
positive-rate spread (spinal cord 79% ... blood 39%) and cell types ~28 points
(mast cell 78% ... mature T cell 50%), plausibly reflecting sequencing depth
and RNA content. So "esophagus is enriched" is true for nearly every gene and
is not evidence about any particular one.

This script establishes what each feature scores for an *average* gene, so
statistical v2 can test against a realistic null instead of a false one.
Deterministic, no LLM, no network.

Usage:
    python scripts/build_feature_baselines.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np

from core.data_loader import GeneExpressionDataset
from core.enrichment import (
    DEFAULT_GO_DEPTH,
    DEFAULT_HIERARCHY_DEPTH,
    DEFAULT_UBERON_DEPTH,
    DEFAULT_UBERON_RELATIONS,
    pair_features,
)
from core.ontology_lookup import OntologyLookup

BASELINES_PATH = Path(__file__).parents[1] / "data" / "feature_baselines.json"
# A feature needs to appear in enough genes for its mean to mean anything.
MIN_GENES_CONTRIBUTING = 100


def baselines_key(dataset_hash: str, provenance: dict, cfg: dict) -> str:
    """
    Identity of the feature universe these baselines describe. Changing the
    dataset, the ontology release, or ANY feature-extraction parameter changes
    which features exist and what they mean — so the runner must refuse stale
    baselines rather than silently mixing them with a different configuration.
    """
    payload = json.dumps({
        "dataset_hash": dataset_hash,
        "cl_data_version": provenance.get("cl_data_version"),
        "uberon_data_version": provenance.get("uberon_data_version"),
        **cfg,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    cfg = {
        "hierarchy_depth": DEFAULT_HIERARCHY_DEPTH,
        "go_depth": DEFAULT_GO_DEPTH,
        "uberon_depth": DEFAULT_UBERON_DEPTH,
        "uberon_relations": list(DEFAULT_UBERON_RELATIONS),
        "include_tissue": True,
        "include_process": True,
    }
    print("Loading dataset and ontology cache...")
    ds = GeneExpressionDataset.load()
    lookup = OntologyLookup()

    print("Expanding features for every row...")
    t0 = time.time()
    rows = list(ds.df.index)
    row_features = [
        pair_features(
            p, lookup,
            hierarchy_depth=cfg["hierarchy_depth"], include_tissue=cfg["include_tissue"],
            include_process=cfg["include_process"], go_depth=cfg["go_depth"],
            uberon_depth=cfg["uberon_depth"], uberon_relations=DEFAULT_UBERON_RELATIONS,
        )
        for p in rows
    ]
    feature_index: dict = {}
    for feats in row_features:
        for kind, term_id, label in feats:
            feature_index.setdefault((kind, term_id), label)
    feat_list = sorted(feature_index)
    feat_pos = {f: i for i, f in enumerate(feat_list)}
    by_kind: dict = defaultdict(int)
    for kind, _ in feat_list:
        by_kind[kind] += 1
    print(f"  {len(feat_list)} distinct features in {time.time() - t0:.1f}s  ({dict(by_kind)})")

    # rows x features indicator matrix; one matmul per gene-block gives per-feature counts
    M = np.zeros((len(rows), len(feat_list)), dtype=np.float32)
    for r, feats in enumerate(row_features):
        for kind, term_id, _ in feats:
            M[r, feat_pos[(kind, term_id)]] = 1.0

    print(f"Computing baselines over all {len(ds.df.columns)} genes...")
    t0 = time.time()
    values = ds.df.to_numpy(dtype=np.float32)  # rows x genes, with NaN for "insufficient data"
    positive = np.nan_to_num(values == 1.0).astype(np.float32)
    called = np.nan_to_num(np.isfinite(values)).astype(np.float32)

    pos_counts = M.T @ positive   # features x genes
    called_counts = M.T @ called  # features x genes

    with np.errstate(invalid="ignore", divide="ignore"):
        rates = np.where(called_counts > 0, pos_counts / called_counts, np.nan)
    baseline = np.nanmean(rates, axis=1)
    n_contributing = np.sum(called_counts > 0, axis=1)

    gene_called = called.sum(axis=0)
    gene_pos = positive.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        gene_rates = np.where(gene_called > 0, gene_pos / gene_called, np.nan)
    grand_mean = float(np.nanmean(gene_rates))
    print(f"  done in {time.time() - t0:.1f}s. Grand mean positive rate across genes: {grand_mean:.1%}")

    out_baselines, skipped = {}, 0
    for i, (kind, term_id) in enumerate(feat_list):
        if n_contributing[i] < MIN_GENES_CONTRIBUTING or not np.isfinite(baseline[i]):
            skipped += 1
            continue
        out_baselines[f"{kind}:{term_id}"] = {
            "label": feature_index[(kind, term_id)],
            "kind": kind,
            "baseline_rate": round(float(baseline[i]), 6),
            "n_genes_contributing": int(n_contributing[i]),
        }
    print(f"  {len(out_baselines)} baselines retained, {skipped} skipped (<{MIN_GENES_CONTRIBUTING} genes)")

    for kind in (by_kind or {}):
        vals = [v["baseline_rate"] for v in out_baselines.values() if v["kind"] == kind]
        if vals:
            print(f"    {kind:8s} spread: {min(vals):.0%} .. {max(vals):.0%}  ({len(vals)} features)")

    payload = {
        "provenance": {
            "built_at": datetime.now(timezone.utc).isoformat(),
            "dataset_hash": ds.dataset_hash,
            "n_genes": int(len(ds.df.columns)),
            "grand_mean_rate": grand_mean,
            "min_genes_contributing": MIN_GENES_CONTRIBUTING,
            "cl_data_version": lookup.provenance.get("cl_data_version"),
            "uberon_data_version": lookup.provenance.get("uberon_data_version"),
            "feature_config": cfg,
            "baselines_key": baselines_key(ds.dataset_hash, lookup.provenance, cfg),
        },
        "baselines": out_baselines,
    }
    BASELINES_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINES_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\nWrote {BASELINES_PATH} ({BASELINES_PATH.stat().st_size / 1e3:.1f} KB)")
    print(f"  baselines_key = {payload['provenance']['baselines_key']}")


if __name__ == "__main__":
    main()
