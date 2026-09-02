#!/usr/bin/env python3
"""
Loader for the binarised gene-expression dataset used as the Step 2/3 stand-in
(PIPELINE_REQUIREMENTS.md) for early approaches: rows are CL:xxx|UBERON:xxx
(cell type, tissue) pairs, columns are Ensembl gene IDs, values are 1
(expressed), 0 (not expressed), or missing (insufficient data — NOT a 0).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_DATASET_PATH = Path(__file__).parents[1] / "binarised_gene_expression_human.tsv"


def parse_pair(pair: str) -> tuple[str, str]:
    """Split a 'CL:xxxxxxx|UBERON:xxxxxxx' row key into (cl_id, uberon_id)."""
    cl_id, uberon_id = pair.split("|")
    return cl_id, uberon_id


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class GeneExpressionDataset:
    path: Path
    df: pd.DataFrame
    dataset_hash: str

    @classmethod
    def load(cls, path: str | Path = DEFAULT_DATASET_PATH) -> "GeneExpressionDataset":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        df = pd.read_csv(path, sep="\t", index_col=0)
        return cls(path=path, df=df, dataset_hash=_sha256_file(path))

    @property
    def n_cell_type_pairs(self) -> int:
        return len(self.df)

    @property
    def n_genes(self) -> int:
        return len(self.df.columns)

    def has_gene(self, gene_id: str) -> bool:
        return gene_id in self.df.columns

    def _require_gene(self, gene_id: str) -> "pd.Series":
        if gene_id not in self.df.columns:
            raise KeyError(f"Gene {gene_id!r} not present in dataset ({self.path.name})")
        return self.df[gene_id]

    def positive_cell_types(self, gene_id: str) -> list[str]:
        """CL|UBERON pair keys where the gene is reliably expressed (value == 1)."""
        col = self._require_gene(gene_id)
        return col.index[col == 1].tolist()

    def negative_cell_types(self, gene_id: str) -> list[str]:
        """CL|UBERON pair keys where the gene is almost never expressed (value == 0)."""
        col = self._require_gene(gene_id)
        return col.index[col == 0].tolist()

    def expression_summary(self, gene_id: str) -> dict:
        col = self._require_gene(gene_id)
        return {
            "n_positive": int((col == 1).sum()),
            "n_negative": int((col == 0).sum()),
            "n_missing": int(col.isna().sum()),
            "n_total": int(len(col)),
        }
