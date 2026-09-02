#!/usr/bin/env python3
"""
Build a small local cache of CL/UBERON labels + definitions, scoped to only
the IDs actually used in binarised_gene_expression_human.tsv (~731 of them),
extracted from the official pinned OBO Foundry ontology releases.

Run this explicitly and only when the cache needs (re)building. Runtime code
(core/ontology_lookup.py) reads only the resulting small cache file — never
the network, never the full ~17MB/12MB OBO files — so an approach run stays
fast and doesn't depend on an external service being up.

Usage:
    python scripts/build_ontology_cache.py
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlretrieve

sys.path.insert(0, str(Path(__file__).parents[1]))

import obonet

from core.data_loader import GeneExpressionDataset, parse_pair

# obonet leaves `def` as the raw OBO value: '"text..." [xref, xref]' — strip to just the quoted text.
_DEF_RE = re.compile(r'^"(.*)"\s*(?:\[.*\])?\s*$', re.DOTALL)


def _clean_definition(raw_def: str | None) -> str | None:
    if not raw_def:
        return None
    match = _DEF_RE.match(raw_def.strip())
    return match.group(1) if match else raw_def.strip()

RAW_DIR = Path(__file__).parents[1] / "data" / "ontologies" / "raw"
CACHE_PATH = Path(__file__).parents[1] / "data" / "ontology_cache.json"

SOURCES = {
    "cl": "https://purl.obolibrary.org/obo/cl.obo",
    "uberon": "https://purl.obolibrary.org/obo/uberon/basic.obo",
}

# Upper bound on is_a traversal depth cached at build time. Runtime code
# (core/ontology_lookup.py) can request any depth up to this without a
# rebuild; going deeper later requires re-running this script with a larger
# value here. Kept well above today's runtime default (2) for headroom.
CACHE_HIERARCHY_DEPTH = 4


def download(name: str, url: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / f"{name}.obo"
    if not dest.exists():
        print(f"Downloading {name} from {url} ...")
        urlretrieve(url, dest)
        print(f"  -> {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"Using cached download: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def _is_a_parents(graph, term_id: str) -> list[str]:
    """Direct is_a parent IDs (obonet models OBO relations as edges keyed by relation type)."""
    if term_id not in graph.nodes:
        return []
    return [parent for _, parent, key in graph.out_edges(term_id, keys=True) if key == "is_a"]


def extract_terms(obo_path: Path, wanted_ids: set[str], max_depth: int = CACHE_HIERARCHY_DEPTH) -> tuple[dict, str]:
    """
    Resolve `wanted_ids` plus every is_a ancestor reachable within `max_depth`
    hops (BFS, all parents on multiple-inheritance terms — no single path is
    picked, see PIPELINE_REQUIREMENTS.md §4.4). Every resolved term (original
    or ancestor) records its own direct `parents` so core/ontology_lookup.py
    can walk the chain at read time without re-parsing the OBO file.
    """
    print(f"Parsing {obo_path.name}...")
    graph = obonet.read_obo(obo_path)
    data_version = graph.graph.get("data-version", "unknown")

    all_ids = set(wanted_ids)
    frontier = set(wanted_ids)
    for _ in range(max_depth):
        next_frontier = {p for tid in frontier for p in _is_a_parents(graph, tid)} - all_ids
        if not next_frontier:
            break
        all_ids |= next_frontier
        frontier = next_frontier
    print(f"  {len(wanted_ids)} requested ids -> {len(all_ids)} including is_a ancestors up to depth {max_depth}")

    resolved = {}
    for term_id in sorted(all_ids):
        if term_id not in graph.nodes:
            resolved[term_id] = {"label": None, "definition": None, "status": "NOT_FOUND", "parents": []}
            continue
        node = graph.nodes[term_id]
        is_obsolete = str(node.get("is_obsolete", "false")).strip().lower() == "true"
        resolved[term_id] = {
            "label": node.get("name"),
            "definition": _clean_definition(node.get("def")),
            "status": "OBSOLETE" if is_obsolete else "OK",
            "parents": _is_a_parents(graph, term_id),
        }
    return resolved, data_version


def main() -> None:
    print("Loading dataset to determine which CL/UBERON IDs are needed...")
    ds = GeneExpressionDataset.load()
    cl_ids: set[str] = set()
    uberon_ids: set[str] = set()
    for pair in ds.df.index:
        cl_id, uberon_id = parse_pair(pair)
        cl_ids.add(cl_id)
        uberon_ids.add(uberon_id)
    print(f"Need {len(cl_ids)} CL ids and {len(uberon_ids)} UBERON ids.")

    cl_path = download("cl", SOURCES["cl"])
    uberon_path = download("uberon", SOURCES["uberon"])

    cl_resolved, cl_version = extract_terms(cl_path, cl_ids)
    uberon_resolved, uberon_version = extract_terms(uberon_path, uberon_ids)

    for label, resolved in [("CL", cl_resolved), ("UBERON", uberon_resolved)]:
        n_ok = sum(1 for v in resolved.values() if v["status"] == "OK")
        n_obsolete = sum(1 for v in resolved.values() if v["status"] == "OBSOLETE")
        n_missing = sum(1 for v in resolved.values() if v["status"] == "NOT_FOUND")
        print(f"{label}: {n_ok}/{len(resolved)} resolved (dataset ids + is_a ancestors), "
              f"{n_obsolete} obsolete, {n_missing} not found")

    cache = {
        "provenance": {
            "cl_source": SOURCES["cl"],
            "cl_data_version": cl_version,
            "uberon_source": SOURCES["uberon"],
            "uberon_data_version": uberon_version,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "dataset_hash": ds.dataset_hash,
            "hierarchy_cache_depth": CACHE_HIERARCHY_DEPTH,
        },
        "cl": cl_resolved,
        "uberon": uberon_resolved,
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))
    print(f"\nWrote {CACHE_PATH} ({CACHE_PATH.stat().st_size / 1e3:.1f} KB)")


if __name__ == "__main__":
    main()
