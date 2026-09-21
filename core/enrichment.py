#!/usr/bin/env python3
"""
Ontology-feature enrichment: which cell-type / tissue features differ between
the set where a gene IS expressed and the set where it is NOT.

This is the "statistics find the contrast, the LLM names it" split — see
approaches/statistical/. Each CL|UBERON pair is expanded into a set of
features (its own CL term, its is_a ancestors, its UBERON tissue), each
feature gets a Fisher's exact test on positive-vs-negative counts, and
p-values are Benjamini-Hochberg corrected.

Deliberately LLM-free and deterministic: every number here is a count you can
check by hand. PIPELINE_REQUIREMENTS.md §6.4 earmarks enrichment for the
verification layer too, hence living in core/ rather than inside one approach.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass

from scipy.stats import binomtest, false_discovery_control, fisher_exact

from core.data_loader import parse_pair
from core.ontology_lookup import OntologyLookup

DEFAULT_HIERARCHY_DEPTH = 3
DEFAULT_GO_DEPTH = 3
DEFAULT_UBERON_DEPTH = 3
DEFAULT_UBERON_RELATIONS = ("is_a", "part_of")
DEFAULT_MIN_FEATURE_COUNT = 20
DEFAULT_MIN_EXCESS = 0.10

# Feature kinds. "lineage" and "tissue" are taxonomy/anatomy — categories. "process"
# comes from CL's capable_of links to GO biological processes and is the only kind
# that yields *properties* (phagocytosis, cytokine production). Process features are
# also the most vulnerable to annotation circularity (§L3): a cell type is annotated
# capable_of phagocytosis partly because of the genes it expresses. Kind is logged so
# any result resting on process features is identifiable.
KIND_LINEAGE, KIND_PROCESS, KIND_TISSUE = "lineage", "process", "tissue"
# A tissue whose cell types are >95% or <5% positive is behaving all-or-nothing,
# which is study/batch structure rather than biology (PIPELINE_REQUIREMENTS.md
# §L9 study bias, §L11 dropout/thresholding fragility).
ALL_OR_NOTHING_HI = 0.95
ALL_OR_NOTHING_LO = 0.05
DEFAULT_MIN_CELL_TYPES_PER_TISSUE = 15


@dataclass
class FeatureResult:
    kind: str  # KIND_LINEAGE | KIND_PROCESS | KIND_TISSUE
    term_id: str
    label: str
    n_pos: int
    n_neg: int
    odds_ratio: float  # >1 enriched where expressed, <1 depleted (v1, global null)
    p_value: float
    q_value: float
    # v2 only — baseline-corrected null. expected_rate is what this gene should
    # achieve on this feature given both its own propensity and the feature's
    # detection baseline; excess is the signed effect size over that expectation.
    baseline_rate: float | None = None
    expected_rate: float | None = None
    excess: float | None = None

    @property
    def enriched(self) -> bool:
        return self.excess > 0 if self.excess is not None else self.odds_ratio > 1.0

    @property
    def effect(self) -> float:
        """Signed effect size — excess where available, else log-ish from the odds ratio."""
        return self.excess if self.excess is not None else (self.odds_ratio - 1.0)

    def as_dict(self) -> dict:
        return asdict(self)


def pair_features(
    pair: str, lookup: OntologyLookup, hierarchy_depth: int = DEFAULT_HIERARCHY_DEPTH,
    include_tissue: bool = True, include_process: bool = False,
    go_depth: int = DEFAULT_GO_DEPTH, uberon_depth: int = DEFAULT_UBERON_DEPTH,
    uberon_relations: tuple[str, ...] = DEFAULT_UBERON_RELATIONS,
) -> set[tuple[str, str, str]]:
    """
    Expand one CL|UBERON pair into (kind, term_id, label) features:
      - lineage: the CL term itself plus its is_a ancestors
      - process: GO biological processes the cell type is capable_of (inherited)
      - tissue:  the UBERON term plus its is_a/part_of ancestors
    Tissue rollup matters because without it a gene's signal fragments across
    lung/nose/trachea instead of aggregating at "respiratory system".
    """
    cl_id, uberon_id = parse_pair(pair)
    out: set[tuple[str, str, str]] = set()

    cl = lookup.resolve_cl(cl_id)
    if cl.resolved:
        out.add((KIND_LINEAGE, cl_id, cl.label))
    for anc in lookup.ancestors("cl", cl_id, hierarchy_depth):
        if anc.resolved:
            out.add((KIND_LINEAGE, anc.id, anc.label))

    if include_process:
        for go in lookup.processes(cl_id, go_depth):
            if go.resolved:
                out.add((KIND_PROCESS, go.id, go.label))

    if include_tissue:
        ub = lookup.resolve_uberon(uberon_id)
        if ub.resolved:
            out.add((KIND_TISSUE, uberon_id, ub.label))
        for anc in lookup.ancestors("uberon", uberon_id, uberon_depth, uberon_relations):
            if anc.resolved:
                out.add((KIND_TISSUE, anc.id, anc.label))
    return out


def count_features(
    pos_pairs: list[str], neg_pairs: list[str], lookup: OntologyLookup, **feature_kwargs,
) -> tuple[dict, dict]:
    """Per-feature (positive, negative) counts. Shared by both scoring methods."""
    cache: dict[str, set] = {}

    def feats(pair: str) -> set:
        if pair not in cache:
            cache[pair] = pair_features(pair, lookup, **feature_kwargs)
        return cache[pair]

    n_pos_by_feat: dict = defaultdict(int)
    n_neg_by_feat: dict = defaultdict(int)
    for p in pos_pairs:
        for f in feats(p):
            n_pos_by_feat[f] += 1
    for p in neg_pairs:
        for f in feats(p):
            n_neg_by_feat[f] += 1
    return n_pos_by_feat, n_neg_by_feat


def _apply_bh(tested: list[FeatureResult], p_values: list[float]) -> list[FeatureResult]:
    if not tested:
        return []
    for result, q in zip(tested, false_discovery_control(p_values, method="bh")):
        result.q_value = float(q)
    return sorted(tested, key=lambda r: r.q_value)


def compute_enrichment(
    pos_pairs: list[str], neg_pairs: list[str], lookup: OntologyLookup, *,
    hierarchy_depth: int = DEFAULT_HIERARCHY_DEPTH,
    min_feature_count: int = DEFAULT_MIN_FEATURE_COUNT,
    include_tissue: bool = True,
) -> list[FeatureResult]:
    """
    v1 scoring: Fisher's exact against a GLOBAL null (feature distributed in
    proportion to set sizes). Kept for `statistical` v1 reproducibility — but note
    that null is measurably wrong: features carry baseline detection differences
    (tissues span ~40 points, cell types ~28), so this reports real over-
    representation that is not specific to the gene. v2 corrects it.
    """
    n_pos_by_feat, n_neg_by_feat = count_features(
        # v1's original feature universe: lineage + exact tissue only. No process
        # features and no UBERON rollup, so v1 runs stay comparable to those logged
        # before v2 existed.
        pos_pairs, neg_pairs, lookup, hierarchy_depth=hierarchy_depth, include_tissue=include_tissue,
        include_process=False, uberon_depth=0,
    )
    P, N = len(pos_pairs), len(neg_pairs)
    tested, p_values = [], []
    for feat in set(n_pos_by_feat) | set(n_neg_by_feat):
        a, b = n_pos_by_feat.get(feat, 0), n_neg_by_feat.get(feat, 0)
        if a + b < min_feature_count:  # underpowered, don't spend a test on it
            continue
        odds_ratio, p_value = fisher_exact([[a, P - a], [b, N - b]])
        kind, term_id, label = feat
        tested.append(FeatureResult(kind, term_id, label, a, b, float(odds_ratio), float(p_value), 1.0))
        p_values.append(p_value)
    return _apply_bh(tested, p_values)


def compute_enrichment_corrected(
    pos_pairs: list[str], neg_pairs: list[str], lookup: OntologyLookup, *,
    baselines: dict, grand_mean: float,
    min_feature_count: int = DEFAULT_MIN_FEATURE_COUNT,
    **feature_kwargs,
) -> tuple[list[FeatureResult], int]:
    """
    v2 scoring: binomial test against a TWO-FACTOR null.

        expected = gene's overall rate x feature's baseline rate / grand mean

    Both factors are required. Correcting only for the tissue baseline made
    MMACHC/esophagus look like pure artifact (+6% over a 71% tissue baseline);
    including MMACHC's own low propensity (20% of cell types) revealed it as a
    +51% excess over expectation. Conversely TTI2/eye deflated from +44% to +17%
    once TTI2's 80% overall expression was accounted for.

    Returns (results, n_skipped_no_baseline).
    """
    n_pos_by_feat, n_neg_by_feat = count_features(pos_pairs, neg_pairs, lookup, **feature_kwargs)
    P, N = len(pos_pairs), len(neg_pairs)
    gene_rate = P / (P + N) if (P + N) else 0.0

    tested, p_values, skipped = [], [], 0
    for feat in set(n_pos_by_feat) | set(n_neg_by_feat):
        a, b = n_pos_by_feat.get(feat, 0), n_neg_by_feat.get(feat, 0)
        n = a + b
        if n < min_feature_count:
            continue
        kind, term_id, label = feat
        baseline = baselines.get(f"{kind}:{term_id}")
        if baseline is None:
            skipped += 1
            continue
        expected = min(0.999, max(0.001, gene_rate * baseline / grand_mean)) if grand_mean else gene_rate
        p_value = binomtest(a, n, expected, alternative="two-sided").pvalue
        observed = a / n
        # odds_ratio kept for continuity with v1's artifact schema, but `excess` is
        # the effect size v2 ranks and filters on.
        odds_ratio = (observed / (1 - observed)) / (expected / (1 - expected)) if 0 < observed < 1 else (
            float("inf") if observed >= 1 else 0.0)
        tested.append(FeatureResult(
            kind, term_id, label, a, b, float(odds_ratio), float(p_value), 1.0,
            baseline_rate=float(baseline), expected_rate=float(expected), excess=float(observed - expected),
        ))
        p_values.append(p_value)
    return _apply_bh(tested, p_values), skipped


def tissue_homogeneity(
    pos_pairs: list[str], neg_pairs: list[str], lookup: OntologyLookup,
    min_cell_types: int = DEFAULT_MIN_CELL_TYPES_PER_TISSUE,
) -> dict:
    """
    Batch/study-artifact diagnostic. Real expression varies within a tissue; a
    tissue whose cell types are ~all positive or ~all negative usually reflects
    which study profiled it and at what depth, not gene biology. A high fraction
    means any enrichment computed here is largely study structure (§L9, §L11).
    """
    counts: dict = defaultdict(lambda: [0, 0])
    for p in pos_pairs:
        counts[parse_pair(p)[1]][0] += 1
    for p in neg_pairs:
        counts[parse_pair(p)[1]][1] += 1

    considered, flagged = [], []
    for uberon_id, (a, b) in counts.items():
        if a + b < min_cell_types:
            continue
        rate = a / (a + b)
        label = lookup.resolve_uberon(uberon_id).label or uberon_id
        considered.append(uberon_id)
        if rate > ALL_OR_NOTHING_HI or rate < ALL_OR_NOTHING_LO:
            flagged.append({"uberon_id": uberon_id, "label": label,
                            "positive_rate": round(rate, 4), "n_cell_types": a + b})
    flagged.sort(key=lambda d: -d["n_cell_types"])
    return {
        "n_tissues_considered": len(considered),
        "n_allornothing_tissues": len(flagged),
        "frac_allornothing_tissues": len(flagged) / len(considered) if considered else 0.0,
        "flagged_tissues": flagged,
    }
