#!/usr/bin/env python3
"""
The Gene Ontology DAG plus human gene -> GO annotations, for scoring GO predictions.

Two things this module exists to make deterministic:

1. Propagation. Curators annotate a gene to the most specific term the evidence
   supports, and GO obeys the true-path rule: an annotation to a term implies
   annotation to every ancestor along is_a / part_of. Going UP the DAG is therefore
   deterministic; going down is not. Predictions and ground truth are both compared
   on their propagated sets (the CAFA protocol).

2. Information content. Once sets are propagated, every set contains the roots, so
   plain overlap is inflated by matches on `biological_process`. IC = -ln p(term),
   with p taken from human GOA frequency after propagation, gives roots IC 0 and
   specific terms high IC, so a match is worth what it tells you.

Sources (all under data/ontologies/raw/, pinned in `provenance`):
  go-basic.obo        the DAG (is_a, part_of used for propagation)
  goa_human.gaf.gz    gene annotations with evidence codes (direct only)
  hgnc_complete_set   Ensembl gene id <-> UniProt accession, since the dataset is
                      keyed by Ensembl id and the GAF by UniProt accession

Only the biological_process aspect is used. NOT-qualified annotations are dropped.
IEA and ND are excluded by default: IEA is frequently derived from the same sources
that annotate cell types, so counting it would inflate agreement.
"""
from __future__ import annotations

import gzip
import math
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import obonet

RAW_DIR = Path(__file__).parents[1] / "data" / "ontologies" / "raw"
GO_OBO_PATH = RAW_DIR / "go-basic.obo"
GAF_PATH = RAW_DIR / "goa_human.gaf.gz"
HGNC_PATH = RAW_DIR / "hgnc_complete_set.txt"

BP_ROOT = "GO:0008150"
BP_NAMESPACE = "biological_process"
PROPAGATION_RELATIONS = ("is_a", "part_of")
DEFAULT_EXCLUDED_EVIDENCE = frozenset({"IEA", "ND"})
_SYNONYM_RE = re.compile(r'^"(.*)"\s+(EXACT|BROAD|NARROW|RELATED)\b')


class GOOntology:
    """GO DAG with deterministic upward propagation and annotation-derived IC."""

    def __init__(self, graph, data_version: str | None = None):
        self._g = graph
        self.data_version = data_version
        # alt_id -> primary id, so annotations written against merged terms still resolve
        self._alt: dict[str, str] = {
            alt: node for node, d in graph.nodes(data=True) for alt in d.get("alt_id", [])
        }
        self._counts: Counter[str] | None = None
        self._n_annotated_genes = 0
        self._name_index: dict[str, str] | None = None
        self.n_ambiguous_names = 0

    @classmethod
    def load(cls, obo_path: str | Path = GO_OBO_PATH) -> "GOOntology":
        graph = obonet.read_obo(str(obo_path))
        return cls(graph, graph.graph.get("data-version"))

    # ---- structure ---------------------------------------------------------------
    def resolve(self, go_id: str) -> str | None:
        """Primary id for go_id (following alt_id), or None if the term is unknown."""
        if go_id in self._g:
            return go_id
        return self._alt.get(go_id)

    def label(self, go_id: str) -> str:
        pid = self.resolve(go_id)
        return self._g.nodes[pid].get("name", go_id) if pid else go_id

    def is_bp(self, go_id: str) -> bool:
        pid = self.resolve(go_id)
        return bool(pid) and self._g.nodes[pid].get("namespace") == BP_NAMESPACE

    @lru_cache(maxsize=None)
    def _ancestors(self, go_id: str, relations: tuple[str, ...]) -> frozenset[str]:
        out: set[str] = {go_id}
        stack = [go_id]
        while stack:
            cur = stack.pop()
            for _, parent, rel in self._g.out_edges(cur, keys=True):
                if rel in relations and parent not in out:
                    out.add(parent)
                    stack.append(parent)
        return frozenset(out)

    @lru_cache(maxsize=None)
    def _ancestor_depths(self, go_id: str, relations: tuple[str, ...]) -> dict[str, int]:
        """BFS (not DFS) so the depth recorded for each ancestor is the SHORTEST path, since a DAG can
        reach the same ancestor via paths of different length. go_id itself is depth 0."""
        depth = {go_id: 0}
        frontier = [go_id]
        while frontier:
            nxt = []
            for cur in frontier:
                for _, parent, rel in self._g.out_edges(cur, keys=True):
                    if rel in relations and parent not in depth:
                        depth[parent] = depth[cur] + 1
                        nxt.append(parent)
            frontier = nxt
        return depth

    def ancestor_depths(self, go_id: str, relations: tuple[str, ...] = PROPAGATION_RELATIONS) -> dict[str, int]:
        """{ancestor_id: shortest number of is_a/part_of edges from go_id up to it}; go_id itself -> 0.
        Used to classify a predicted term against a true term as exact / generalization / specialization:
        see core/go_match.py."""
        pid = self.resolve(go_id)
        return dict(self._ancestor_depths(pid, tuple(relations))) if pid else {}

    def ancestors(self, go_id: str, relations: tuple[str, ...] = PROPAGATION_RELATIONS) -> set[str]:
        """All ancestors of go_id along `relations`, INCLUDING go_id itself."""
        pid = self.resolve(go_id)
        return set(self._ancestors(pid, tuple(relations))) if pid else set()

    def propagate(self, go_ids, relations: tuple[str, ...] = PROPAGATION_RELATIONS) -> set[str]:
        out: set[str] = set()
        for go_id in go_ids:
            out |= self.ancestors(go_id, relations)
        return out

    # ---- names -> ids (for free-form LLM output) -----------------------------------
    @staticmethod
    def _norm(text: str) -> str:
        """Lowercase and collapse punctuation/whitespace, applied to BOTH the index and the query, so
        "G-protein coupled" and "G protein-coupled" match. Punctuation only: no stemming, no fuzzy
        matching, because a near-miss name is a different term ("visual signal transduction" is not
        "ABA signal transduction")."""
        return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

    def build_name_index(self) -> None:
        """
        Lowercased biological_process names and EXACT synonyms -> GO id. A name that
        maps to more than one term is dropped rather than guessed at, and counted in
        `n_ambiguous_names`. Only EXACT synonyms are used: BROAD/NARROW/RELATED ones
        would let a near-miss score as a hit.
        """
        seen: dict[str, set[str]] = defaultdict(set)
        for node, d in self._g.nodes(data=True):
            if d.get("namespace") != BP_NAMESPACE:
                continue
            if d.get("name"):
                seen[self._norm(d["name"])].add(node)
            for syn in d.get("synonym", []) or []:
                m = _SYNONYM_RE.match(syn)
                if m and m.group(2) == "EXACT":
                    seen[self._norm(m.group(1))].add(node)
        self._name_index = {k: next(iter(v)) for k, v in seen.items() if len(v) == 1}
        self.n_ambiguous_names = sum(1 for v in seen.values() if len(v) > 1)

    def resolve_name(self, text: str) -> str | None:
        """GO id for an exact (case/whitespace-insensitive) name or EXACT synonym, else None."""
        if self._name_index is None:
            self.build_name_index()
        return self._name_index.get(self._norm(text))

    # ---- information content -----------------------------------------------------
    def fit_information_content(self, gene_to_terms: dict[str, set[str]]) -> None:
        """
        p(term) = fraction of annotated genes whose PROPAGATED set contains the term.
        Genes with no annotation are excluded from the denominator, so the root has
        p = 1 and therefore IC exactly 0.
        """
        counts: Counter[str] = Counter()
        n = 0
        for terms in gene_to_terms.values():
            prop = self.propagate(terms)
            if not prop:
                continue
            n += 1
            counts.update(prop)
        self._counts, self._n_annotated_genes = counts, n

    def ic(self, go_id: str) -> float:
        """-ln p(term). Terms never annotated to any gene get the maximum observed IC."""
        if self._counts is None:
            raise RuntimeError("call fit_information_content() first")
        pid = self.resolve(go_id) or go_id
        c = self._counts.get(pid, 0)
        if c == 0:
            return math.log(self._n_annotated_genes)
        return 0.0 - math.log(c / self._n_annotated_genes)  # 0.0 - x avoids printing -0.0 at the root

    def resnik(self, a_ids, b_ids) -> float:
        """Resnik similarity of two term sets: max IC over the ancestors they share."""
        shared = self.propagate(a_ids) & self.propagate(b_ids)
        return max((self.ic(t) for t in shared), default=0.0)


def load_ensembl_to_uniprot(hgnc_path: str | Path = HGNC_PATH) -> dict[str, set[str]]:
    """Ensembl gene id -> UniProt accessions, from the HGNC complete set."""
    import csv
    out: dict[str, set[str]] = defaultdict(set)
    with open(hgnc_path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            ens, uni = row.get("ensembl_gene_id"), row.get("uniprot_ids")
            if ens and uni:
                out[ens].update(u.strip() for u in uni.split("|") if u.strip())
    return dict(out)


def load_gene_annotations(
    ontology: GOOntology, gaf_path: str | Path = GAF_PATH, hgnc_path: str | Path = HGNC_PATH,
    excluded_evidence: frozenset[str] | set[str] = DEFAULT_EXCLUDED_EVIDENCE,
) -> tuple[dict[str, set[str]], dict]:
    """
    Ensembl gene id -> DIRECT (unpropagated) biological_process GO ids.

    Returns (annotations, stats). `stats` records how many GAF rows each filter
    removed, so the effect of the evidence policy is visible rather than assumed.
    """
    ens_to_uni = load_ensembl_to_uniprot(hgnc_path)
    uni_to_ens: dict[str, set[str]] = defaultdict(set)
    for ens, unis in ens_to_uni.items():
        for u in unis:
            uni_to_ens[u].add(ens)

    stats = Counter()
    ann: dict[str, set[str]] = defaultdict(set)
    with gzip.open(gaf_path, "rt") as f:
        for line in f:
            if line.startswith("!"):
                continue
            c = line.rstrip("\n").split("\t")
            stats["rows"] += 1
            if c[8] != "P":
                stats["not_bp_aspect"] += 1
                continue
            if "NOT" in c[3]:
                stats["not_qualified"] += 1
                continue
            if c[6] in excluded_evidence:
                stats[f"excluded_evidence_{c[6]}"] += 1
                continue
            genes = uni_to_ens.get(c[1])
            if not genes:
                stats["uniprot_not_in_hgnc"] += 1
                continue
            pid = ontology.resolve(c[4])
            if not pid or not ontology.is_bp(pid):
                stats["term_unresolved_or_not_bp"] += 1
                continue
            for g in genes:
                ann[g].add(pid)
            stats["kept"] += 1
    return dict(ann), dict(stats)


def load_gene_annotation_evidence(
    ontology: GOOntology, gaf_path: str | Path = GAF_PATH, hgnc_path: str | Path = HGNC_PATH,
    excluded_evidence: frozenset[str] | set[str] = DEFAULT_EXCLUDED_EVIDENCE,
) -> dict[tuple[str, str], frozenset[str]]:
    """
    {(gene_id, go_id): evidence codes seen for that exact pair in GOA}.

    `load_gene_annotations` builds the per-gene SET of true terms but discards which evidence code
    supports which specific term, so there is no way afterwards to ask "was THIS matched term backed by
    experimental evidence (IDA/IMP/...) or an inferred one (IBA/ISS/TAS/...)?". This is a sibling, not a
    replacement: same GAF pass, same filters (aspect/qualifier/evidence/resolvability), purely additive —
    `load_gene_annotations` and its callers are unchanged.
    """
    ens_to_uni = load_ensembl_to_uniprot(hgnc_path)
    uni_to_ens: dict[str, set[str]] = defaultdict(set)
    for ens, unis in ens_to_uni.items():
        for u in unis:
            uni_to_ens[u].add(ens)

    evidence: dict[tuple[str, str], set[str]] = defaultdict(set)
    with gzip.open(gaf_path, "rt") as f:
        for line in f:
            if line.startswith("!"):
                continue
            c = line.rstrip("\n").split("\t")
            if c[8] != "P" or "NOT" in c[3] or c[6] in excluded_evidence:
                continue
            genes = uni_to_ens.get(c[1])
            if not genes:
                continue
            pid = ontology.resolve(c[4])
            if not pid or not ontology.is_bp(pid):
                continue
            for g in genes:
                evidence[(g, pid)].add(c[6])
    return {k: frozenset(v) for k, v in evidence.items()}
