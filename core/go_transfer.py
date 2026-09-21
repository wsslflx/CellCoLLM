#!/usr/bin/env python3
"""
Co-annotation transfer: a no-LLM baseline that can name GO terms OUTSIDE the cell-type
vocabulary.

The cell-level GO terms in the evidence (52 candidates: "phagocytosis", "cell motility", ...)
and the gene-level terms a gene is annotated with are different vocabularies, so the plain
statistical baselines (go_enrichment v1-v4) can only ever answer with the 52. An LLM is not
so confined, which confounds any free-form comparison with vocabulary size.

This baseline removes the confound by learning the cell-level -> gene-level mapping from data:
across ALL other genes in GOA, which gene-level terms tend to be annotated to genes that carry a
given cell-level term? For each enriched evidence term c and each gene-level term T:

    s(c, T) = -log10 P(X >= J)    X ~ Hypergeom(M genes, n_T genes with T, n_c genes with c),
                                  J = genes carrying BOTH   (over-representation of T among c-genes)

and a term is scored by evidence-weighted association:  score(T) = sum_c  w_c * s(c, T),
where w_c = -log10 p of the evidence term in the corrected statistics (enriched terms only).

LEAVE-ONE-OUT: the target gene's own annotation is removed from every count before scoring —
otherwise its true terms would leak into the association table and inflate the baseline. The
remaining ~16k genes' annotations are legitimate training signal (non-IEA, the same evidence
policy as the ground truth); the LLM arms have no such table.

If the LLM cannot beat this, its world knowledge adds nothing over a lookup table built from
the very annotation resource it is scored against.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.stats import hypergeom

from core.go_ontology import BP_ROOT
from core.go_scoring import GOTruth

DEFAULT_MIN_GENES = 30
MAX_ASSOC = 300.0   # cap on -log10 p so a single perfect co-occurrence cannot dominate


class GOTransfer:
    def __init__(self, truth: GOTruth, candidates: list[str], min_genes: int = DEFAULT_MIN_GENES):
        go = truth.go
        self.truth, self.candidates, self.min_genes = truth, list(candidates), min_genes
        self.genes = sorted(g for g in truth.annotations if truth.annotations[g])
        self.gene_idx = {g: i for i, g in enumerate(self.genes)}
        self.vocab = sorted(t for t, n in go._counts.items()
                            if n >= min_genes and t != BP_ROOT and go.is_bp(t) and go.ic(t) > 0)
        cidx = {t: j for j, t in enumerate(self.candidates)}
        vidx = {t: j for j, t in enumerate(self.vocab)}
        rc, cc, rv, cv = [], [], [], []
        for i, g in enumerate(self.genes):
            for t in go.propagate(truth.annotations[g]):
                if t in cidx:
                    rc.append(i); cc.append(cidx[t])
                if t in vidx:
                    rv.append(i); cv.append(vidx[t])
        G = len(self.genes)
        self.C = sparse.csr_matrix((np.ones(len(rc), dtype=np.float32), (rc, cc)), shape=(G, len(self.candidates)))
        self.T = sparse.csr_matrix((np.ones(len(rv), dtype=np.float32), (rv, cv)), shape=(G, len(self.vocab)))
        self.J = (self.C.T @ self.T).toarray()            # candidates x vocab: genes carrying both
        self.n_c = np.asarray(self.C.sum(axis=0)).ravel()
        self.n_T = np.asarray(self.T.sum(axis=0)).ravel()

    def association(self, gene_id: str | None, rows: np.ndarray) -> np.ndarray:
        """s(c, T) for the candidate rows `rows` (indices), leaving `gene_id` out if it is in the table."""
        J, n_c, n_T, M = self.J[rows], self.n_c[rows], self.n_T, len(self.genes)
        gi = self.gene_idx.get(gene_id) if gene_id else None
        if gi is not None:
            cg = self.C[gi].toarray().ravel()[rows]
            tg = self.T[gi].toarray().ravel()
            J = J - np.outer(cg, tg)
            n_c = n_c - cg
            n_T = n_T - tg
            M -= 1
        with np.errstate(divide="ignore"):
            s = -np.log10(np.maximum(hypergeom.sf(J - 1, M, n_T[None, :], n_c[:, None]), 10 ** -MAX_ASSOC))
        return np.where(J > (n_c[:, None] * n_T[None, :] / M), s, 0.0)   # over-representation only

    def rank(self, gene_id: str | None, weights: dict[str, float], top_n: int = 100, leave_out: bool = True) -> list[dict]:
        """Ranked gene-level terms for evidence weights {candidate go_id: w}. `leave_out=False` exists only to show the leak."""
        cidx = {t: j for j, t in enumerate(self.candidates)}
        use = [(cidx[t], w) for t, w in weights.items() if t in cidx and w > 0]
        if not use:
            return []
        rows = np.array([r for r, _ in use])
        w = np.array([x for _, x in use])
        s = self.association(gene_id if leave_out else None, rows)
        score = w @ s
        go = self.truth.go
        order = sorted(range(len(self.vocab)), key=lambda j: (-score[j], self.vocab[j]))[:top_n]
        return [{"go_id": self.vocab[j], "label": go.label(self.vocab[j]), "rank": i + 1, "score": float(score[j])}
                for i, j in enumerate(order) if score[j] > 0]

    def explain(self, gene_id: str | None, weights: dict[str, float], top_c: int = 5, top_t: int = 5) -> list[dict]:
        """For the strongest evidence terms, their most associated gene-level terms (audit trail for a run)."""
        cidx = {t: j for j, t in enumerate(self.candidates)}
        go = self.truth.go
        out = []
        for t, wt in sorted(weights.items(), key=lambda kv: -kv[1])[:top_c]:
            if t not in cidx:
                continue
            s = self.association(gene_id, np.array([cidx[t]]))[0]
            top = np.argsort(-s)[:top_t]
            out.append({"evidence_term": go.label(t), "weight": float(wt),
                        "top_gene_level_terms": [{"label": go.label(self.vocab[j]), "assoc": float(s[j])} for j in top if s[j] > 0]})
        return out
