#!/usr/bin/env python3
"""
Deterministic scoring of predicted GO terms against a gene's own GO annotation.

No LLM and no judge: predictions and truth are both propagated up the GO DAG
(is_a + part_of) and compared as information-content-weighted sets (CAFA protocol).
Root-level matches carry IC 0 and therefore score nothing.

Definitions (all on PROPAGATED sets, weights = IC from human GOA):
    truth       propagate(direct non-IEA biological_process annotations of the gene)
    prediction  propagate(top-k predicted terms)
    precision   IC mass of (prediction & truth) / IC mass of prediction
    recall      IC mass of (prediction & truth) / IC mass of truth
    F1@k        harmonic mean of the two

    ceiling@k   best F1 reachable by naming <=k terms from the CANDIDATE vocabulary
                (greedy union — a lower bound on the true optimum)
    floor@k     mean F1 of k candidate terms drawn at random
    headroom@k  (F1@k - floor@k) / (ceiling@k - floor@k)

Headroom is only meaningful when the prediction is confined to the candidate
vocabulary (constrained mode). For free-form output the ceiling is not the
candidate-vocabulary ceiling, so headroom could exceed 1 — report raw F1@k there.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from core.go_ontology import (
    BP_ROOT, DEFAULT_EXCLUDED_EVIDENCE, GOOntology, load_gene_annotations,
)

DEFAULT_KS = (1, 3, 5, 10)
DEFAULT_FLOOR_DRAWS = 500


def f1_from_mass(overlap: float, pred_mass: float, truth_mass: float) -> float:
    if overlap <= 0 or pred_mass <= 0 or truth_mass <= 0:
        return 0.0
    p, r = overlap / pred_mass, overlap / truth_mass
    return 2 * p * r / (p + r)


class CeilingModel:
    """Vectorised per-gene ceilings/floors for one candidate vocabulary."""

    def __init__(self, go: GOOntology, vocab: list[str]):
        self.go, self.vocab = go, vocab
        anc = [go.ancestors(v) for v in vocab]
        self.terms = sorted(set().union(*anc)) if anc else []       # only terms the vocabulary can reach
        self.col = {t: j for j, t in enumerate(self.terms)}
        self.ic = np.array([go.ic(t) for t in self.terms])
        self.A = np.zeros((len(vocab), len(self.terms)))            # vocabulary term -> its propagated ancestors
        for i, a in enumerate(anc):
            for t in a:
                self.A[i, self.col[t]] = 1.0
        self.pred_mass = self.A @ self.ic                           # IC mass of each candidate prediction

    def truth_weights(self, truth: set[str]) -> np.ndarray:
        """IC-weighted indicator of the truth restricted to the terms this vocabulary can reach."""
        inter = np.zeros(len(self.terms))
        for t in truth:
            j = self.col.get(t)
            if j is not None:
                inter[j] = 1.0
        return inter * self.ic

    def gene(self, truth: set[str], truth_mass: float) -> dict:
        w = self.truth_weights(truth)
        reach = float(w.sum())
        out = {"reachable_ic_fraction": reach / truth_mass if truth_mass else 0.0,
               "max_shared_ic": float(self.ic[w > 0].max()) if reach > 0 else 0.0}
        overlaps = self.A @ w                                        # IC mass of (prediction ∩ truth), per term
        f_single = np.array([f1_from_mass(o, m, truth_mass) for o, m in zip(overlaps, self.pred_mass)])
        out["f1_top1"] = float(f_single.max()) if len(f_single) else 0.0
        # Floor: expected F1 of naming one vocabulary term AT RANDOM. Exact, no sampling.
        out["f1_top1_random"] = float(f_single.mean()) if len(f_single) else 0.0
        out["f1_top3"] = self.greedy_ceiling(w, truth_mass, k=3)
        return out

    def greedy_ceiling(self, w: np.ndarray, truth_mass: float, k: int) -> float:
        chosen = np.zeros(len(self.terms))
        best = 0.0
        for _ in range(k):
            cand_f = []
            for i in range(len(self.vocab)):
                u = np.maximum(chosen, self.A[i])
                cand_f.append(f1_from_mass(float((u * w).sum()), float((u * self.ic).sum()), truth_mass))
            i = int(np.argmax(cand_f))
            if cand_f[i] <= best + 1e-12:
                break
            best, chosen = cand_f[i], np.maximum(chosen, self.A[i])
        return best

    def floor_at_k(self, truth: set[str], truth_mass: float, k: int,
                   n_draws: int = DEFAULT_FLOOR_DRAWS, seed: str = "") -> float:
        """Mean F1 of k random vocabulary terms. Exact for k=1; Monte-Carlo (seeded) otherwise."""
        w = self.truth_weights(truth)
        if k == 1:
            ov = self.A @ w
            return float(np.mean([f1_from_mass(o, m, truth_mass) for o, m in zip(ov, self.pred_mass)]))
        rng = np.random.default_rng(int(hashlib.sha256(f"floor{seed}".encode()).hexdigest()[:8], 16))
        f = []
        for _ in range(n_draws):
            idx = rng.choice(len(self.vocab), size=min(k, len(self.vocab)), replace=False)
            u = self.A[idx].max(axis=0)
            f.append(f1_from_mass(float((u * w).sum()), float((u * self.ic).sum()), truth_mass))
        return float(np.mean(f))


@dataclass
class GOTruth:
    """Ontology + non-IEA gene annotations + IC, loaded once and shared across every run."""
    go: GOOntology
    annotations: dict[str, set[str]]           # gene -> DIRECT BP terms
    excluded_evidence: frozenset
    gaf_stats: dict

    @classmethod
    def load(cls, excluded_evidence=DEFAULT_EXCLUDED_EVIDENCE) -> "GOTruth":
        go = GOOntology.load()
        ann, stats = load_gene_annotations(go, excluded_evidence=frozenset(excluded_evidence))
        go.fit_information_content(ann)
        return cls(go, ann, frozenset(excluded_evidence), stats)

    def has(self, gene_id: str) -> bool:
        return bool(self.annotations.get(gene_id))

    def truth_set(self, gene_id: str) -> set[str]:
        return self.go.propagate(self.annotations.get(gene_id, ()))

    def mass(self, terms) -> float:
        return sum(self.go.ic(t) for t in terms)

    def f1_at_k(self, ranked: list[str], gene_id: str, k: int) -> float:
        truth = self.truth_set(gene_id)
        pred = self.go.propagate(ranked[:k])
        return f1_from_mass(self.mass(pred & truth), self.mass(pred), self.mass(truth))


def score_ranking(
    truth: GOTruth, ranked_go_ids: list[str], gene_id: str, model: CeilingModel | None,
    ks: tuple[int, ...] = DEFAULT_KS, floor_draws: int = DEFAULT_FLOOR_DRAWS,
) -> dict:
    """
    F1@k for the ranked list; if `model` (the candidate vocabulary) is given, also
    ceiling@{1,3}, floor@{1,3} and headroom@{1,3}. Values that are undefined (no room
    between floor and ceiling) are simply omitted so they are never logged as 0.
    """
    ranked = list(dict.fromkeys(ranked_go_ids))  # dedupe, keep order
    t_set = truth.truth_set(gene_id)
    t_mass = truth.mass(t_set)
    out: dict = {"n_predictions": len(ranked), "truth_ic_mass": t_mass}
    f1_by_k = {}
    for k in range(1, max(max(ks), 10) + 1):
        pred = truth.go.propagate(ranked[:k])
        f1_by_k[k] = f1_from_mass(truth.mass(pred & t_set), truth.mass(pred), t_mass)
    for k in ks:
        out[f"f1_at_{k}"] = f1_by_k[k]
    out["f1_mean_1_10"] = float(np.mean([f1_by_k[k] for k in range(1, 11)])) if ranked else 0.0
    if model is not None and t_mass > 0:
        w = model.truth_weights(t_set)
        for k in (1, 3):
            ceil = model.greedy_ceiling(w, t_mass, k)
            floor = model.floor_at_k(t_set, t_mass, k, floor_draws, seed=f"{gene_id}:{k}")
            out[f"ceiling_at_{k}"], out[f"floor_at_{k}"] = ceil, floor
            if ceil - floor > 1e-9:
                out[f"headroom_at_{k}"] = (out[f"f1_at_{k}"] - floor) / (ceil - floor)
    return out


def fit_constant_prior(truth: GOTruth, dev_genes: list[str], vocab: list[str], k: int = 3) -> list[str]:
    """
    Greedy best FIXED k-term list: the terms that, named for every gene regardless of
    evidence, maximise mean F1 over `dev_genes`. Fit on development genes only so the
    test genes never inform the baseline they are compared against.
    """
    model = CeilingModel(truth.go, vocab)
    genes = [g for g in dev_genes if truth.has(g)]
    W = np.array([model.truth_weights(truth.truth_set(g)) for g in genes])
    tm = np.array([truth.mass(truth.truth_set(g)) for g in genes])
    chosen_idx: list[int] = []
    union = np.zeros(len(model.terms))
    best_score = -1.0
    for _ in range(k):
        cand_scores = []
        for i in range(len(vocab)):
            u = np.maximum(union, model.A[i])
            pm = float((u * model.ic).sum())
            ov = W @ u
            f = np.where((ov > 0) & (tm > 0) & (pm > 0),
                         2 * (ov / max(pm, 1e-12)) * (ov / np.maximum(tm, 1e-12)) /
                         np.maximum(ov / max(pm, 1e-12) + ov / np.maximum(tm, 1e-12), 1e-12), 0.0)
            cand_scores.append(float(f.mean()))
        i = int(np.argmax(cand_scores))
        if cand_scores[i] <= best_score + 1e-12:
            break
        best_score = cand_scores[i]
        chosen_idx.append(i)
        union = np.maximum(union, model.A[i])
    return [vocab[i] for i in chosen_idx]


def log_scored_predictions(
    rank_p: list[dict], rank_effect: list[dict] | None, truth: GOTruth, gene_id: str,
    model: CeilingModel | None,
) -> dict:
    """
    Log the ranked prediction list(s) as artifacts and the scores as `gopred_*` metrics
    into the ACTIVE MLflow run, so scoring lives in the same run as the prediction.
    Returns the primary metrics dict.
    """
    import mlflow
    from core.mlflow_utils import log_json_artifact

    log_json_artifact(rank_p, "go_predictions.json")
    metrics = score_ranking(truth, [r["go_id"] for r in rank_p], gene_id, model)
    mlflow.log_metrics({f"gopred_{k}": float(v) for k, v in metrics.items()})
    if rank_effect is not None:
        log_json_artifact(rank_effect, "go_predictions_effect.json")
        eff = score_ranking(truth, [r["go_id"] for r in rank_effect], gene_id, model)
        mlflow.log_metrics({f"gopred_effect_{k}": float(v) for k, v in eff.items()
                            if k.startswith(("f1_at_", "headroom_at_"))})
    return metrics


def log_significant_only_score(
    rank_p: list[dict], significant_ids: set[str], truth: GOTruth, gene_id: str, model: CeilingModel | None,
) -> dict | None:
    """
    SECONDARY metric for arm 1 (go_enrichment v1-v4 only, where a real significance test exists):
    F1@k/headroom computed on the SAME ranking as the primary score, but restricted to terms that
    actually cleared the significance threshold (g:SCS or BH, at --alpha). Logged as `gopred_sigonly_*`
    ALONGSIDE the primary `gopred_*` metrics (which stay unfiltered) -- see approaches/README.md for why
    the primary score is not filtered: it would shrink or empty the ranked list for many genes and
    unfairly handicap this arm against go_llm, which always answers with up to 10 terms. This exists so
    "how good is arm 1 when it's actually confident" is visible as a distinct, comparable number.

    Returns None (and logs nothing) if there are zero significant terms for this gene -- a real and
    informative outcome (see scripts/summarize_go_matches.py for how often that happens), but not one
    that should be mixed into an F1 average as an uninformative 0.
    """
    import mlflow

    filtered = [r for r in rank_p if r["go_id"] in significant_ids]
    mlflow.log_metric("gopred_sigonly_n_significant", len(filtered))
    if not filtered:
        return None
    metrics = score_ranking(truth, [r["go_id"] for r in filtered], gene_id, model)
    mlflow.log_metrics({f"gopred_sigonly_{k}": float(v) for k, v in metrics.items()})
    return metrics
