#!/usr/bin/env python3
"""
GO (Gene Ontology) biological-process annotations for genes, via MyGene.info,
with a persistent lazy cache — unlike core/ontology_lookup.py's fully
upfront-built cache (CL/UBERON scope was fully known from the dataset's row
index), the set of genes that will ever need GO terms here (scored genes plus
whichever genes get drawn as random distractors) can't be enumerated in
advance. So: fetch live on first use, write the raw result into the cache
file immediately, never re-fetch a gene once it's cached — pinned/reproducible
from that point on, without a big upfront build job.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

MYGENE_BASE_URL = "https://mygene.info/v3"
GO_CACHE_PATH = Path(__file__).parents[1] / "data" / "go_cache.json"


def fetch_go_terms_live(ensembl_id: str, timeout: int = 10) -> list[dict]:
    """
    One live GET to MyGene.info for a gene's GO:BP (biological process)
    annotations. Returns a list of {"go_id", "term", "evidence"} — [] on any
    failure (gene not found, network error, unexpected response shape), so a
    single bad lookup doesn't crash the caller.
    """
    url = f"{MYGENE_BASE_URL}/gene/{ensembl_id}?fields=go"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read())
        bp_entries = data.get("go", {}).get("BP", [])
        if isinstance(bp_entries, dict):  # MyGene.info returns a bare dict, not a list, for a single annotation
            bp_entries = [bp_entries]
        terms = []
        seen_go_ids = set()
        for entry in bp_entries:
            go_id = entry.get("id")
            if not go_id or go_id in seen_go_ids:
                continue
            seen_go_ids.add(go_id)
            terms.append({"go_id": go_id, "term": entry.get("term"), "evidence": entry.get("evidence")})
        return terms
    except Exception:
        return []


class GOLookup:
    def __init__(self, cache_path: str | Path = GO_CACHE_PATH):
        self.cache_path = Path(cache_path)
        if self.cache_path.exists():
            self._cache: dict = json.loads(self.cache_path.read_text())
        else:
            self._cache = {}

    def _save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache, indent=2, sort_keys=True))

    def get_bp_terms(self, gene_id: str, non_iea_only: bool = True) -> list[dict]:
        """Cache hit returns cached (filtered) terms; cache miss fetches live,
        persists the raw fetched result immediately, then returns filtered terms."""
        if gene_id not in self._cache:
            self._cache[gene_id] = {"bp_terms": fetch_go_terms_live(gene_id)}
            self._save()
        terms = self._cache[gene_id]["bp_terms"]
        if non_iea_only:
            terms = [t for t in terms if t.get("evidence") != "IEA"]
        return terms
