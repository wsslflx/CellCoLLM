#!/usr/bin/env python3
"""
Classical GO over-representation analysis, following g:Profiler's g:GOSt method.

This is the no-LLM scientific baseline arm: a query set of items is tested
against a background for over-representation of GO terms, using Fisher's
one-tailed test (cumulative hypergeometric) with g:SCS multiple-testing
correction, plus Bonferroni and Benjamini-Hochberg as g:Profiler also offers.

Method reference — https://biit.cs.ut.ee/gprofiler/page/docs and
Raudvere et al. NAR 2019 47(W1):W191-W198 (doi:10.1093/nar/gkz369).

IMPORTANT FIDELITY NOTE: g:Profiler annotates *genes* with GO terms; here the
annotated items are *cell types* (via CL capable_of). Their web API therefore
cannot be called — this reimplements the method against a different annotation
universe. It is "g:Profiler's method", not g:Profiler.

Distinct from core/enrichment.py, which contrasts a positive set against a
negative set. This module does query-versus-background over-representation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import false_discovery_control, hypergeom

DEFAULT_MIN_TERM_SIZE = 3
DEFAULT_MAX_TERM_SIZE = 0  # 0 = unlimited
DEFAULT_N_SIMULATIONS = 2000  # matches the simulation count in g:Profiler's original paper
DEFAULT_ALPHA = 0.05


@dataclass
class GOResult:
    """One row of the result table, mirroring g:Profiler's reported columns."""
    go_id: str
    label: str
    term_size: int          # background items annotated with this term
    query_size: int
    intersection_size: int
    expected: float         # query_size * term_size / n_background
    fold_enrichment: float  # intersection / expected
    precision: float        # intersection / query_size
    recall: float           # intersection / term_size
    p_value: float          # cumulative hypergeometric (Fisher one-tailed)
    p_gscs: float
    p_bonferroni: float
    p_fdr_bh: float

    def as_dict(self) -> dict:
        return asdict(self)


def build_annotation_matrix(
    items: list[str], item_to_go: dict[str, set[str]],
) -> tuple[np.ndarray, list[str]]:
    """
    items x terms binary indicator matrix over the background.
    Returns (matrix, term_ids) with term_ids in stable sorted order.
    """
    term_ids = sorted({go for it in items for go in item_to_go.get(it, ())})
    index = {go: j for j, go in enumerate(term_ids)}
    matrix = np.zeros((len(items), len(term_ids)), dtype=np.float32)
    for i, it in enumerate(items):
        for go in item_to_go.get(it, ()):
            matrix[i, index[go]] = 1.0
    return matrix, term_ids


def hypergeometric_pvalues(
    term_sizes: np.ndarray, intersections: np.ndarray, n_background: int, n_query: int,
) -> np.ndarray:
    """
    Fisher's one-tailed test == cumulative hypergeometric probability, vectorised.
    P(X >= k) = sf(k-1) with X ~ Hypergeom(N=n_background, K=term_size, n=n_query).
    """
    return hypergeom.sf(intersections - 1, n_background, term_sizes, n_query)


def gscs_threshold(
    matrix: np.ndarray, query_size: int, n_simulations: int = DEFAULT_N_SIMULATIONS,
    alpha: float = DEFAULT_ALPHA, seed: int = 0,
) -> float:
    """
    g:SCS (Set Counts and Sizes) threshold for a given query size.

    Per g:Profiler's documentation, the threshold is the upper `alpha` quantile of
    the *minimum* p-value obtained from randomly generated queries of that size.
    Random queries are drawn as actual item sets rather than sampling each term's
    overlap independently — sampling real sets is what preserves the term-overlap
    dependency that g:SCS exists to model (a cell type annotated with two terms
    correlates them).

    Returns the raw p-value that corresponds to an experiment-wide error rate of
    `alpha`; callers rescale observed p-values by alpha / threshold.
    """
    n_items, n_terms = matrix.shape
    if n_terms == 0 or query_size <= 0 or query_size > n_items:
        return alpha
    rng = np.random.default_rng(seed)
    term_sizes = matrix.sum(axis=0)

    min_p = np.empty(n_simulations, dtype=np.float64)
    block = 200  # keep the simulated indicator block modest in memory
    done = 0
    while done < n_simulations:
        this = min(block, n_simulations - done)
        picks = np.empty((this, n_items), dtype=np.float32)
        for r in range(this):
            row = np.zeros(n_items, dtype=np.float32)
            row[rng.choice(n_items, size=query_size, replace=False)] = 1.0
            picks[r] = row
        overlaps = picks @ matrix  # (this x n_terms) intersection counts
        pvals = hypergeom.sf(overlaps - 1, n_items, term_sizes[None, :], query_size)
        min_p[done:done + this] = np.nanmin(pvals, axis=1)
        done += this

    threshold = float(np.quantile(min_p, alpha))
    # Guard against a degenerate all-ones distribution (possible when the query is
    # nearly the whole background, so every random draw looks like the background).
    return threshold if threshold > 0 else float(np.finfo(np.float64).tiny)


def enrich(
    query_items: list[str], background_items: list[str], item_to_go: dict[str, set[str]],
    term_labels: dict[str, str] | None = None, *,
    min_term_size: int = DEFAULT_MIN_TERM_SIZE, max_term_size: int = DEFAULT_MAX_TERM_SIZE,
    n_simulations: int = DEFAULT_N_SIMULATIONS, alpha: float = DEFAULT_ALPHA, seed: int = 0,
) -> tuple[list[GOResult], dict]:
    """
    Over-representation of GO terms in `query_items` against `background_items`.

    Returns (results sorted by g:SCS p-value ascending, diagnostics dict). The
    diagnostics carry the g:SCS threshold and the structural fold-enrichment
    ceiling — see the README: when the query is a large fraction of the
    background, max achievable fold enrichment is N/n regardless of biology.
    """
    background = list(dict.fromkeys(background_items))  # dedupe, preserve order
    n_background = len(background)
    query_set = set(query_items)
    query_in_bg = [it for it in background if it in query_set]
    n_query = len(query_in_bg)

    matrix, term_ids = build_annotation_matrix(background, item_to_go)
    diagnostics = {
        "n_background": n_background,
        "n_query": n_query,
        "query_background_ratio": n_query / n_background if n_background else 0.0,
        # k <= term_size and expected = n*K/N, so fold <= N/n no matter the biology.
        "max_fold_enrichment_possible": n_background / n_query if n_query else 0.0,
        "n_terms_total": len(term_ids),
        "n_simulations": n_simulations,
        "alpha": alpha,
        "seed": seed,
    }
    if not term_ids or not n_query:
        diagnostics.update({"n_terms_tested": 0, "gscs_threshold": alpha})
        return [], diagnostics

    term_sizes = matrix.sum(axis=0)
    keep = term_sizes >= min_term_size
    if max_term_size:
        keep &= term_sizes <= max_term_size
    kept_idx = np.flatnonzero(keep)
    diagnostics["n_terms_tested"] = int(kept_idx.size)
    if kept_idx.size == 0:
        diagnostics["gscs_threshold"] = alpha
        return [], diagnostics

    matrix_k = matrix[:, kept_idx]
    sizes_k = term_sizes[kept_idx]
    query_mask = np.array([1.0 if it in query_set else 0.0 for it in background], dtype=np.float32)
    intersections = query_mask @ matrix_k

    p_raw = hypergeometric_pvalues(sizes_k, intersections, n_background, n_query)
    threshold = gscs_threshold(matrix_k, n_query, n_simulations=n_simulations, alpha=alpha, seed=seed)
    diagnostics["gscs_threshold"] = threshold

    p_gscs = np.minimum(1.0, p_raw * (alpha / threshold))
    p_bonf = np.minimum(1.0, p_raw * kept_idx.size)
    p_bh = false_discovery_control(p_raw, method="bh")

    labels = term_labels or {}
    results = []
    for j, idx in enumerate(kept_idx):
        go_id = term_ids[idx]
        K, k = float(sizes_k[j]), float(intersections[j])
        expected = n_query * K / n_background
        results.append(GOResult(
            go_id=go_id, label=labels.get(go_id, go_id),
            term_size=int(K), query_size=n_query, intersection_size=int(k),
            # Full precision deliberately — rounding here broke the fold <= N/n
            # invariant that the README derives, and this is a scientific artifact.
            expected=float(expected),
            fold_enrichment=float(k / expected) if expected else 0.0,
            precision=float(k / n_query) if n_query else 0.0,
            recall=float(k / K) if K else 0.0,
            p_value=float(p_raw[j]), p_gscs=float(p_gscs[j]),
            p_bonferroni=float(p_bonf[j]), p_fdr_bh=float(p_bh[j]),
        ))
    results.sort(key=lambda r: (r.p_gscs, r.p_value))
    return results, diagnostics
