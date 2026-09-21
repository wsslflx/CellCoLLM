#!/usr/bin/env python3
"""
Everything the GO-prediction experiment loads once and shares across runs.

Loading the dataset (~10s), the GO ontology + GOA annotations (~7s) and building the
evidence matrices is far too slow to repeat per run when the experiment is thousands of
runs, so run scripts take an optional pre-built `GOShared`; run standalone they build
their own.
"""
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, field

import numpy as np

from core.data_loader import GeneExpressionDataset
from core.go_evidence import (
    DEFAULT_GO_DEPTH, DEFAULT_MIN_TERM_SIZE, GOEvidenceBuilder, load_go_baselines,
)
from core.go_ontology import HGNC_PATH, GOOntology
from core.go_scoring import CeilingModel, GOTruth
from core.go_transfer import DEFAULT_MIN_GENES, GOTransfer
from core.ontology_lookup import OntologyLookup


@dataclass
class GOShared:
    ds: GeneExpressionDataset
    lookup: OntologyLookup
    builder: GOEvidenceBuilder
    truth: GOTruth | None
    model: CeilingModel | None
    baselines: dict | None
    _symbols: dict[str, str] = field(default_factory=dict)
    _aliases: dict[str, list[str]] = field(default_factory=dict)
    _transfer: dict = field(default_factory=dict)

    @classmethod
    def load(cls, dataset: str | None = None, go_depth: int = DEFAULT_GO_DEPTH,
             min_term_size: int = DEFAULT_MIN_TERM_SIZE, need_truth: bool = True,
             need_baselines: bool = True) -> "GOShared":
        ds = GeneExpressionDataset.load(dataset) if dataset else GeneExpressionDataset.load()
        lookup = OntologyLookup()
        truth = GOTruth.load() if need_truth else None
        go = truth.go if truth else GOOntology.load()
        builder = GOEvidenceBuilder(ds, lookup, go, go_depth, min_term_size)
        model = CeilingModel(go, builder.candidates) if truth else None
        baselines = load_go_baselines(builder) if need_baselines else None
        return cls(ds, lookup, builder, truth, model, baselines)

    def _load_hgnc(self) -> None:
        with open(HGNC_PATH, newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                ens = row.get("ensembl_gene_id")
                if ens:
                    self._symbols[ens] = row["symbol"]
                    names = (row.get("alias_symbol") or "").split("|") + (row.get("prev_symbol") or "").split("|")
                    self._aliases[ens] = [n.strip() for n in names if n.strip()]

    def symbol(self, gene_id: str) -> str:
        if not self._symbols:
            self._load_hgnc()
        return self._symbols.get(gene_id, gene_id)

    def aliases(self, gene_id: str) -> list[str]:
        if not self._symbols:
            self._load_hgnc()
        return self._aliases.get(gene_id, [])

    def stratum(self, gene_id: str) -> str:
        """'carrying' if the gene's propagated annotation contains >=1 candidate term, else 'not_carrying'."""
        if not self.truth or not self.truth.has(gene_id):
            return "unscoreable"
        return "carrying" if self.truth.truth_set(gene_id) & set(self.builder.candidates) else "not_carrying"

    def gene_rates(self, unit: str) -> np.ndarray:
        """Positive rate of every gene over the items it is called for, at this unit (cached)."""
        cache = self.builder.__dict__.setdefault("_rate_cache", {})
        if unit not in cache:
            _, pos, called = self.builder.unit_arrays(unit)
            with np.errstate(invalid="ignore", divide="ignore"):
                cache[unit] = np.where(called.sum(0) > 0, pos.sum(0) / called.sum(0), np.nan)
        return cache[unit]

    def choose_donor(self, gene_id: str, unit: str, seed: int, tolerance: float = 0.05) -> str:
        """
        Deterministic donor for the mismatched-evidence control: a DIFFERENT gene whose overall
        positive rate is within `tolerance` of the target's, so the control keeps the input
        distribution (breadth) comparable and differs only in WHICH cell types are positive.
        """
        b = self.builder
        rates = self.gene_rates(unit)
        target = rates[b.gene_col[gene_id]]
        order = sorted((g for g in b.genes if g != gene_id),
                       key=lambda g: hashlib.sha256(f"{seed}|{gene_id}|{g}".encode()).hexdigest())
        for g in order:
            r = rates[b.gene_col[g]]
            if np.isfinite(r) and abs(r - target) <= tolerance:
                return g
        raise RuntimeError(f"No donor gene within ±{tolerance} of {target:.2f} for {gene_id}")

    def transfer(self, min_genes: int = DEFAULT_MIN_GENES) -> GOTransfer:
        """Co-annotation transfer table (built once per min_genes; see core/go_transfer.py)."""
        if min_genes not in self._transfer:
            self._transfer[min_genes] = GOTransfer(self.truth, self.builder.candidates, min_genes)
        return self._transfer[min_genes]
