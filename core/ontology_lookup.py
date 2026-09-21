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
    part_of: list[str] = field(default_factory=list)  # anatomical containment (UBERON)
    capable_of: list[str] = field(default_factory=list)  # GO biological processes (CL)

    @property
    def resolved(self) -> bool:
        return self.status == "OK" and self.label is not None

    def related(self, relations: tuple[str, ...]) -> list[str]:
        """Direct targets across the requested relation names."""
        by_name = {"is_a": self.parents, "part_of": self.part_of, "capable_of": self.capable_of}
        out: list[str] = []
        for rel in relations:
            out.extend(by_name.get(rel, []))
        return out


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
            parents=entry.get("parents", []), part_of=entry.get("part_of", []),
            capable_of=entry.get("capable_of", []),
        )

    def resolve_cl(self, cl_id: str) -> ResolvedTerm:
        return self._resolve("cl", cl_id)

    def resolve_uberon(self, uberon_id: str) -> ResolvedTerm:
        return self._resolve("uberon", uberon_id)

    def resolve_go(self, go_id: str) -> ResolvedTerm:
        """GO biological-process terms, harvested from cl.obo during cache build."""
        return self._resolve("go", go_id)

    def processes(self, cl_id: str, depth: int) -> list[ResolvedTerm]:
        """
        GO biological processes a cell type is capable_of, including those inherited
        from its is_a ancestors up to `depth`. Direct annotation is sparse (~13% of
        dataset cell types); inheritance raises coverage substantially (~53% at
        depth 3), because capable_of is asserted on general terms and inherited by
        subtypes. Depth is a real trade-off — deeper means more coverage but generic
        processes spread across many cell types (PIPELINE_REQUIREMENTS.md §4.3.2).
        """
        go_ids: list[str] = list(self._resolve("cl", cl_id).capable_of)
        for ancestor in self.ancestors("cl", cl_id, depth):
            go_ids.extend(ancestor.capable_of)
        seen, out = set(), []
        for go_id in go_ids:
            if go_id in seen:
                continue
            seen.add(go_id)
            out.append(self.resolve_go(go_id))
        return out

    def ancestors(
        self, ontology: str, term_id: str, depth: int, relations: tuple[str, ...] = ("is_a",),
    ) -> list[ResolvedTerm]:
        """
        BFS over the cached relation graph, depth-limited, deduped by id. Terms
        with multiple parents contribute all of them (no single path is
        picked, PIPELINE_REQUIREMENTS.md §4.4); a chain that reaches a term
        missing from the cache (beyond the build-time hierarchy depth, or
        NOT_FOUND) simply stops there rather than erroring.

        `relations` selects which edges to walk — ("is_a",) for taxonomy, and
        ("is_a", "part_of") for UBERON where anatomical containment (lung ->
        respiratory system) matters as much as subtype relationships.
        """
        key = {"is_a": "parents", "part_of": "part_of", "capable_of": "capable_of"}
        visited = {term_id}
        result: list[ResolvedTerm] = []
        frontier = [term_id]
        for _ in range(depth):
            next_frontier: list[str] = []
            for tid in frontier:
                entry = self._cache[ontology].get(tid)
                if entry is None:
                    continue
                for rel in relations:
                    for parent_id in entry.get(key.get(rel, rel), []):
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
