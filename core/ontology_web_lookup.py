#!/usr/bin/env python3
"""
Live CL/UBERON term lookup via EBI's public Ontology Lookup Service (OLS4).

UNLIKE core/ontology_lookup.py (the pinned local cache used by `enriched`),
this hits the network on every call and is NOT reproducible in the usual
sense — OLS4's content can change over time. It exists only for the
exploratory `naive` v3 agentic tool-use test: the point there is to see what
the model does when it can genuinely look something up itself, not to
provide a stable, versioned data source. Callers must log whatever this
returns verbatim per run, since it cannot be assumed stable on rerun.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

OLS4_BASE_URL = "https://www.ebi.ac.uk/ols4/api"


def fetch_term_live(ontology: str, obo_id: str, timeout: int = 10) -> dict:
    """
    One live GET to EBI OLS4 for a single CL or UBERON term.
    Returns {"id", "label", "definition", "found"} — found=False (not an
    exception) on a 404, timeout, or unexpected response shape, so one bad
    lookup doesn't crash the run.
    """
    url = f"{OLS4_BASE_URL}/ontologies/{ontology}/terms?obo_id={obo_id}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read())
        terms = data.get("_embedded", {}).get("terms", [])
        if not terms:
            return {"id": obo_id, "label": None, "definition": None, "found": False}
        term = terms[0]
        description = term.get("description") or []
        return {
            "id": obo_id,
            "label": term.get("label"),
            "definition": description[0] if description else None,
            "found": True,
        }
    except Exception:
        return {"id": obo_id, "label": None, "definition": None, "found": False}
