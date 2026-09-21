#!/usr/bin/env python3
"""
GO-term evidence tables for one gene, at either unit of analysis — the single source
used by go_enrichment v3/v4 (no LLM) and go_llm (LLM), so the rungs of the comparison
ladder cannot drift apart.

Two units, mirroring go_enrichment v1/v2:
  cell_types  each annotated CL term once. A cell type is POSITIVE if >=1 of its rows
              is 1, CALLED if >=1 row is non-missing. This makes the positive set
              identical to go_enrichment v1's query.
  rows        each annotated CL|UBERON pair once (value 1 = positive, NaN = not called).

Universe: annotated items only (cell types / rows with >=1 GO term), and among those
only items the gene is CALLED for. `gene_rate` and the baselines' `grand_mean` are both
defined over that universe, so the two-factor null is internally consistent. This
differs from `statistical` v2, which used all rows.

Candidate vocabulary: GO biological-process terms carried by >=`min_term_size` distinct
annotated cell types, fixed for BOTH units so ceilings, floors and rankings are
comparable across every rung.

The corrected statistic mirrors core/enrichment.compute_enrichment_corrected:
    expected = gene_rate x feature baseline / grand mean
    binomial test of k_pos successes in K carriers against `expected`.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, false_discovery_control

from core.data_loader import GeneExpressionDataset, parse_pair
from core.go_ontology import GOOntology
from core.ontology_lookup import OntologyLookup

UNIT_CELL_TYPES = "cell_types"
UNIT_ROWS = "rows"
UNITS = (UNIT_CELL_TYPES, UNIT_ROWS)
DEFAULT_GO_DEPTH = 3
DEFAULT_MIN_TERM_SIZE = 3
GO_BASELINES_PATH = Path(__file__).parents[1] / "data" / "go_baselines.json"


@dataclass
class TermCount:
    go_id: str
    label: str
    K: int        # carriers among the items the gene is called for
    k_pos: int    # of those, how many are positive


@dataclass
class GOEvidence:
    gene_id: str
    unit: str
    n_pos: int
    n_called: int
    terms: list[TermCount]

    @property
    def gene_rate(self) -> float:
        return self.n_pos / self.n_called if self.n_called else 0.0


@dataclass
class CorrectedTerm:
    go_id: str
    label: str
    K: int
    k_pos: int
    observed: float
    expected: float
    excess: float
    p_value: float
    q_value: float
    tested: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


class GOEvidenceBuilder:
    """Gene-independent structures built once; evidence for any gene is then a couple of matmuls."""

    def __init__(self, ds: GeneExpressionDataset, lookup: OntologyLookup, go: GOOntology,
                 go_depth: int = DEFAULT_GO_DEPTH, min_term_size: int = DEFAULT_MIN_TERM_SIZE):
        self.go_depth, self.min_term_size = go_depth, min_term_size
        self.dataset_hash = ds.dataset_hash
        self.cl_data_version = lookup.provenance.get("cl_data_version")
        rows = list(ds.df.index)
        cl_of_row = [parse_pair(p)[0] for p in rows]

        cl_go: dict[str, frozenset[str]] = {}
        for cl in sorted(set(cl_of_row)):
            terms = set()
            for t in lookup.processes(cl, go_depth):
                pid = go.resolve(t.id)
                if pid and go.is_bp(pid):
                    terms.add(pid)
            if terms:
                cl_go[cl] = frozenset(terms)
        self.cl_go = cl_go
        self.ann_cl = sorted(cl_go)
        counts = Counter(t for ts in cl_go.values() for t in ts)
        self.candidates = sorted(t for t, n in counts.items() if n >= min_term_size)
        self.labels = {t: go.label(t) for t in self.candidates}
        cidx = {t: j for j, t in enumerate(self.candidates)}
        C = len(self.candidates)

        cl_pos = {cl: i for i, cl in enumerate(self.ann_cl)}
        self.M_ct = np.zeros((len(self.ann_cl), C), dtype=np.float32)
        for cl, i in cl_pos.items():
            for t in cl_go[cl]:
                if t in cidx:
                    self.M_ct[i, cidx[t]] = 1.0

        ann_rows = [i for i, cl in enumerate(cl_of_row) if cl in cl_go]
        self.ann_row_index = ann_rows
        self.ann_row_names = [rows[i] for i in ann_rows]
        self.ann_row_cl = [cl_of_row[i] for i in ann_rows]
        self.M_row = self.M_ct[[cl_pos[cl_of_row[i]] for i in ann_rows]]

        values = ds.df.to_numpy(dtype=np.float32)[ann_rows, :]        # annotated rows x genes
        self.genes = list(ds.df.columns)
        self.gene_col = {g: j for j, g in enumerate(self.genes)}
        self.row_pos = (values == 1.0)
        self.row_called = np.isfinite(values)
        ct_of_row = np.array([cl_pos[cl_of_row[i]] for i in ann_rows])
        self.ct_pos = np.zeros((len(self.ann_cl), len(self.genes)), dtype=bool)
        self.ct_called = np.zeros_like(self.ct_pos)
        for i in range(len(self.ann_cl)):
            sel = ct_of_row == i
            self.ct_pos[i] = self.row_pos[sel].any(axis=0)
            self.ct_called[i] = self.row_called[sel].any(axis=0)

    # ---- per-unit matrices (used by the baseline builder too) --------------------
    def unit_arrays(self, unit: str):
        """(term-incidence M, positive matrix, called matrix), items x {terms, genes}."""
        if unit == UNIT_CELL_TYPES:
            return self.M_ct, self.ct_pos, self.ct_called
        if unit == UNIT_ROWS:
            return self.M_row, self.row_pos, self.row_called
        raise ValueError(f"unit must be one of {UNITS}, got {unit!r}")

    def called_universe(self, gene_id: str, unit: str) -> tuple[list[str], dict[str, set[str]]]:
        """
        (items, item -> GO ids) for annotated items THIS gene is called for, with
        biological-process annotations only. This is the universe the whole ladder
        uses; go_enrichment's legacy default (--universe annotated) also counts
        missing items and non-BP capable_of targets, so it differs slightly.
        """
        j = self.gene_col[gene_id]
        if unit == UNIT_CELL_TYPES:
            items = [cl for i, cl in enumerate(self.ann_cl) if self.ct_called[i, j]]
            return items, {cl: set(self.cl_go[cl]) for cl in items}
        keep = [i for i in range(len(self.ann_row_names)) if self.row_called[i, j]]
        items = [self.ann_row_names[i] for i in keep]
        return items, {self.ann_row_names[i]: set(self.cl_go[self.ann_row_cl[i]]) for i in keep}

    def evidence(self, gene_id: str, unit: str) -> GOEvidence:
        if gene_id not in self.gene_col:
            raise KeyError(f"Gene {gene_id!r} not in dataset")
        M, pos, called = self.unit_arrays(unit)
        j = self.gene_col[gene_id]
        p, c = pos[:, j].astype(np.float32), called[:, j].astype(np.float32)
        K, k = c @ M, p @ M
        terms = [TermCount(t, self.labels[t], int(K[i]), int(k[i]))
                 for i, t in enumerate(self.candidates) if K[i] > 0]
        return GOEvidence(gene_id, unit, int(p.sum()), int(c.sum()), terms)


# ---- baselines -------------------------------------------------------------------
def baselines_key(builder: GOEvidenceBuilder) -> str:
    import hashlib
    payload = json.dumps({"dataset_hash": builder.dataset_hash, "cl": builder.cl_data_version,
                          "go_depth": builder.go_depth, "min_term_size": builder.min_term_size,
                          "n_candidates": len(builder.candidates)}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_go_baselines(builder: GOEvidenceBuilder, path: str | Path = GO_BASELINES_PATH) -> dict:
    """{unit: {"grand_mean": float, "terms": {go_id: baseline_rate}}}; refuses stale baselines."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: python scripts/build_go_baselines.py")
    payload = json.loads(path.read_text())
    want, have = baselines_key(builder), payload["provenance"]["baselines_key"]
    if want != have:
        raise SystemExit(
            f"GO baselines are stale (built with key {have}, this configuration is {want}: "
            f"dataset, CL version, --go-depth or --min-term-size changed). "
            f"Rebuild: python scripts/build_go_baselines.py"
        )
    return payload["units"]


# ---- corrected statistics ---------------------------------------------------------
def corrected_stats(ev: GOEvidence, unit_baselines: dict, min_term_size: int = DEFAULT_MIN_TERM_SIZE,
                    candidates: list[str] | None = None) -> list[CorrectedTerm]:
    """
    Binomial test against the two-factor null for every candidate term. Terms with fewer
    than `min_term_size` carriers for THIS gene (missing data can shrink K), or with no
    baseline, are kept but flagged tested=False so rankings still cover the whole
    candidate set.
    """
    grand, base = unit_baselines["grand_mean"], unit_baselines["terms"]
    rate = ev.gene_rate
    tested: list[CorrectedTerm] = []
    ps: list[float] = []
    seen = set()
    for t in ev.terms:
        seen.add(t.go_id)
        b = base.get(t.go_id)
        if t.K < min_term_size or b is None or not grand:
            tested.append(CorrectedTerm(t.go_id, t.label, t.K, t.k_pos, t.k_pos / t.K if t.K else 0.0,
                                        float("nan"), float("nan"), 1.0, 1.0, tested=False))
            continue
        expected = min(0.999, max(0.001, rate * b / grand))
        p = binomtest(t.k_pos, t.K, expected, alternative="two-sided").pvalue
        obs = t.k_pos / t.K
        tested.append(CorrectedTerm(t.go_id, t.label, t.K, t.k_pos, obs, expected, obs - expected, float(p), 1.0))
        ps.append(float(p))
    live = [x for x in tested if x.tested]
    if live:
        qs = false_discovery_control(np.array([x.p_value for x in live]), method="bh")
        for x, q in zip(live, qs):
            x.q_value = float(q)
    for go_id in (candidates or []):
        if go_id not in seen:
            tested.append(CorrectedTerm(go_id, "", 0, 0, 0.0, float("nan"), float("nan"), 1.0, 1.0, tested=False))
    return tested


# ---- rankings (pre-registered; see the experiment protocol in approaches/README.md) ----
def _finish(rows: list[dict]) -> list[dict]:
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


def rank_corrected(results: list[CorrectedTerm], by: str = "p") -> list[dict]:
    """
    by="p" (primary): enriched terms (excess > 0) ordered by p ascending, ties by larger excess
    then go_id; then the remaining tested terms by excess descending; then untested terms.
    by="effect": all tested terms by excess descending, ties by p.
    """
    live = [r for r in results if r.tested]
    dead = sorted((r for r in results if not r.tested), key=lambda r: r.go_id)
    if by == "effect":
        ordered = sorted(live, key=lambda r: (-r.excess, r.p_value, r.go_id))
    else:
        enriched = sorted((r for r in live if r.excess > 0), key=lambda r: (r.p_value, -r.excess, r.go_id))
        rest = sorted((r for r in live if not r.excess > 0), key=lambda r: (-r.excess, r.p_value, r.go_id))
        ordered = enriched + rest
    ordered += dead
    return _finish([{"go_id": r.go_id, "label": r.label,
                     "score": (-np.log10(max(r.p_value, 1e-300)) if r.tested and r.excess > 0 else 0.0)}
                    for r in ordered])


def rank_gprofiler(results, candidates: list[str], labels: dict[str, str], by: str = "p") -> list[dict]:
    """
    Same idea for go_enrichment's hypergeometric GOResult rows, restricted to `candidates`:
    by="p": enriched terms (fold > 1) by raw p ascending, ties by fold then go_id; then the rest
    by fold descending; then candidates that were not tested at all.
    by="effect": by fold enrichment descending, ties by p.
    """
    cand = set(candidates)
    live = [r for r in results if r.go_id in cand]
    tested_ids = {r.go_id for r in live}
    dead = sorted(cand - tested_ids)
    if by == "effect":
        ordered = sorted(live, key=lambda r: (-r.fold_enrichment, r.p_value, r.go_id))
    else:
        enriched = sorted((r for r in live if r.fold_enrichment > 1), key=lambda r: (r.p_value, -r.fold_enrichment, r.go_id))
        rest = sorted((r for r in live if not r.fold_enrichment > 1), key=lambda r: (-r.fold_enrichment, r.p_value, r.go_id))
        ordered = enriched + rest
    rows = [{"go_id": r.go_id, "label": r.label,
             "score": (-np.log10(max(r.p_value, 1e-300)) if r.fold_enrichment > 1 else 0.0)} for r in ordered]
    rows += [{"go_id": g, "label": labels.get(g, g), "score": 0.0} for g in dead]
    return _finish(rows)
