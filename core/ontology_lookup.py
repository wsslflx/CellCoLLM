#!/usr/bin/env python3
"""
Runtime lookup for CL/UBERON labels, backed by the small pre-built cache in
data/ontology_cache.json (see scripts/build_ontology_cache.py). Deliberately
dumb and fast — never hits the network or parses the full OBO files here;
rebuilding the cache is a separate, explicit step.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CACHE_PATH = Path(__file__).parents[1] / "data" / "ontology_cache.json"


@dataclass
class ResolvedTerm:
    id: str
    label: str | None
    definition: str | None
    status: str  # "OK" / "OBSOLETE" / "NOT_FOUND"
    parents: list[str] = field(default_factory=list)  # direct is_a parent ids, same ontology

    @property
    def resolved(self) -> bool:
        return self.status == "OK" and self.label is not None


class OntologyLookup:
    def __init__(self, cache_path: str | Path = DEFAULT_CACHE_PATH):
        cache_path = Path(cache_path)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Ontology cache not found at {cache_path}. Run scripts/build_ontology_cache.py first."
            )
        self._cache: dict[str, Any] = json.loads(cache_path.read_text())

    @property
    def provenance(self) -> dict:
        return self._cache["provenance"]

    def _resolve(self, ontology: str, term_id: str) -> ResolvedTerm:
        entry = self._cache[ontology].get(term_id)
        if entry is None:
            return ResolvedTerm(id=term_id, label=None, definition=None, status="NOT_FOUND")
        return ResolvedTerm(
            id=term_id, label=entry["label"], definition=entry["definition"], status=entry["status"],
            parents=entry.get("parents", []),
        )

    def resolve_cl(self, cl_id: str) -> ResolvedTerm:
        return self._resolve("cl", cl_id)

    def resolve_uberon(self, uberon_id: str) -> ResolvedTerm:
        return self._resolve("uberon", uberon_id)

    def ancestors(self, ontology: str, term_id: str, depth: int) -> list[ResolvedTerm]:
        """
        BFS over cached is_a `parents`, depth-limited, deduped by id. Terms
        with multiple parents contribute all of them (no single path is
        picked, PIPELINE_REQUIREMENTS.md §4.4); a chain that reaches a term
        missing from the cache (beyond the build-time hierarchy depth, or
        NOT_FOUND) simply stops there rather than erroring.
        """
        visited = {term_id}
        result: list[ResolvedTerm] = []
        frontier = [term_id]
        for _ in range(depth):
            next_frontier: list[str] = []
            for tid in frontier:
                entry = self._cache[ontology].get(tid)
                if entry is None:
                    continue
                for parent_id in entry.get("parents", []):
                    if parent_id in visited:
                        continue
                    visited.add(parent_id)
                    result.append(self._resolve(ontology, parent_id))
                    next_frontier.append(parent_id)
            if not next_frontier:
                break
            frontier = next_frontier
        return result

    def ancestors_cl(self, cl_id: str, depth: int) -> list[ResolvedTerm]:
        return self.ancestors("cl", cl_id, depth)

    def ancestors_uberon(self, uberon_id: str, depth: int) -> list[ResolvedTerm]:
        return self.ancestors("uberon", uberon_id, depth)
