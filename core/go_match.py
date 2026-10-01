#!/usr/bin/env python3
"""
Classify a predicted GO term against a gene's DIRECT true annotations, by graph distance.

Three mutually exclusive outcomes per predicted term:
  exact          the term IS one of the gene's direct annotations (distance 0)
  upward(d)      the term is a strict ANCESTOR of some true term, reached in d edges going up
                 from that true term (the prediction GENERALISES the truth — the common case for
                 cell-level evidence terms, e.g. "phagocytosis, engulfment" (true) -> "phagocytosis"
                 (predicted) is upward, d=1)
  downward(d)    the term is a strict DESCENDANT of some true term, reached in d edges going up
                 FROM the prediction to that true term (the prediction SPECIALISES the truth —
                 plausible but unannotated)
  no_match       neither: a lateral relation through a shared ancestor, or no informative relation
                 at all

A term cannot be both upward and downward of the SAME gene (that would require a cycle in the
DAG), but against a set of several true terms it is classified by whichever relation is found;
ties (reachable both ways via different true terms) do not occur for a single predicted term
against a single DAG, so each term gets exactly one label.

This is a simple, auditable alternative to the IC-weighted F1/headroom scoring in
core/go_scoring.py — a raw edge count rather than an information-content-weighted overlap. It is
intentionally coarser (every edge counts the same, even though the DAG is uneven — see
approaches/README.md) and exists for the match/mismatch breakdown in
scripts/summarize_go_matches.py, not as a replacement for the primary score.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.go_ontology import GOOntology

EXACT, UPWARD, DOWNWARD, NO_MATCH = "exact", "upward", "downward", "no_match"


@dataclass
class TermMatch:
    go_id: str
    label: str
    kind: str          # one of EXACT, UPWARD, DOWNWARD, NO_MATCH
    distance: int | None  # edges; None for NO_MATCH


def classify_term(go: GOOntology, predicted_id: str, truth_direct: set[str]) -> TermMatch:
    """Classify one predicted term against the gene's direct (unpropagated) true annotations."""
    pid = go.resolve(predicted_id) or predicted_id
    label = go.label(pid)
    if pid in truth_direct:
        return TermMatch(pid, label, EXACT, 0)

    # upward: predicted term found while climbing from a true term
    up = None
    for t in truth_direct:
        d = go.ancestor_depths(t).get(pid)
        if d is not None and (up is None or d < up):
            up = d
    if up is not None:
        return TermMatch(pid, label, UPWARD, up)

    # downward: a true term found while climbing from the predicted term
    pred_depths = go.ancestor_depths(pid)
    down = None
    for t in truth_direct:
        d = pred_depths.get(t)
        if d is not None and (down is None or d < down):
            down = d
    if down is not None:
        return TermMatch(pid, label, DOWNWARD, down)

    return TermMatch(pid, label, NO_MATCH, None)


def classify_terms(go: GOOntology, predicted_ids: list[str], truth_direct: set[str]) -> list[TermMatch]:
    return [classify_term(go, p, truth_direct) for p in dict.fromkeys(predicted_ids)]  # dedupe, keep order


def bin_label(d: int, edges: list[int]) -> str:
    """edges=[1,2,3,5,10] -> bins 'd=1','d=2','d=3','3<d<=5','5<d<=10','d>10'."""
    prev = 0
    for e in edges:
        if d <= e:
            return f"d={d}" if e - prev <= 1 else f"{prev}<d<={e}"
        prev = e
    return f"d>{edges[-1]}"
