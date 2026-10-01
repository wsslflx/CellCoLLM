#!/usr/bin/env python3
"""
Arm 2: give an LLM the same cell-type GO evidence the no-LLM baselines see, and ask it to
predict GO biological-process terms for an undisclosed gene. Scored deterministically
(no judge) against the gene's own GO annotation, into the SAME MLflow run.

Four versions, encoding statistic x unit of analysis (mirroring go_enrichment v1-v4):
  v1  raw counts table, distinct cell types        (2a: can the LLM do the statistics itself?)
  v2  raw counts table, dataset rows
  v3  corrected table, distinct cell types         (2b: the same table go_enrichment v3 ranks)
  v4  corrected table, dataset rows
So "go_llm v3 minus go_enrichment v3" isolates what the LLM adds on identical statistics.

Two orthogonal flags:
  --output-mode constrained  choose and rank up to N GO ids copied from the table (primary)
  --output-mode freeform     name up to N GO biological-process terms; resolved to ids
                             deterministically by exact name / EXACT synonym (secondary)
  --evidence true            the gene's own table
  --evidence mismatched      ANOTHER gene's table (control: is the evidence used at all?)
  --evidence none            no table (control: the LLM's prior)

The prompts ask the model to INTERPRET the evidence, not read it back: a `hypothesis` (the shared
program, the molecular machinery it implies) comes first, then the predicted terms. In free-form
mode the answer is meant to describe what the gene product DOES, which is a different level from the
cell-behaviour terms in the evidence; `frac_in_table` measures how much of the answer is merely
evidence terms re-listed. Constrained mode, which can only re-rank evidence terms, is kept as a
control (how well can an LLM re-weight the statistics?). Free-form is the primary comparison.

No abstention key: a ranked-list metric needs a ranking, and an empty answer would score
below the random floor for reasons unrelated to biology.

Scoring lives in core/go_scoring.py. Headroom (vs candidate-vocabulary floor and ceiling)
is logged only for constrained mode; free-form output is not confined to the candidate
vocabulary so its headroom could exceed 1 — raw F1@k is what is compared there.

Usage:
    python approaches/go_llm/run_go_llm.py --gene ENSG00000149554 --prompt-version v3
    python approaches/go_llm/run_go_llm.py --gene ENSG00000149554 --prompt-version v3 --output-mode freeform
    python approaches/go_llm/run_go_llm.py --gene ENSG00000149554 --prompt-version v3 --evidence mismatched
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

import mlflow

from core.go_evidence import (
    UNIT_CELL_TYPES, UNIT_ROWS, GOEvidence, baselines_key, corrected_stats,
)
from core.go_experiment import GOShared
from core.go_scoring import log_scored_predictions
from core.llm_backend import compute_num_ctx, make_chat_llm, resolve_chat_model
from core.mlflow_utils import (
    RunContext,
    get_or_create_gene_parent_run,
    log_json_artifact,
    log_text_artifact,
    tracked_run,
    verify_blinding,
)
from core.run_identity import fingerprint
from core.structured_llm import call_llm_with_retry

APPROACH = "go_llm"
PROMPTS_DIR = Path(__file__).parent / "prompts"
PROMPT_VERSIONS = {
    "v1": {"unit": UNIT_CELL_TYPES, "statistics": "raw"},
    "v2": {"unit": UNIT_ROWS, "statistics": "raw"},
    "v3": {"unit": UNIT_CELL_TYPES, "statistics": "corrected"},
    "v4": {"unit": UNIT_ROWS, "statistics": "corrected"},
}
REQUIRED_RESPONSE_KEYS = {"hypothesis", "predictions"}
UNIT_TEXT = {
    UNIT_CELL_TYPES: ("each item is a distinct cell type (a Cell Ontology term), counted once however many tissues it was sampled in",
                      "cell types"),
    UNIT_ROWS: ("each item is one cell type x tissue sample from a single-cell atlas", "cell-type x tissue samples"),
}
_GO_ID_RE = re.compile(r"(?:GO:)?\s*(\d{7})")
_LLM_CACHE: dict = {}


# ---- evidence ------------------------------------------------------------------
def go_definition(shared: GOShared, go_id: str) -> str:
    raw = shared.truth.go._g.nodes[go_id].get("def", "") if shared.truth else ""
    m = re.match(r'^"(.*)"\s*\[', raw or "")
    return (m.group(1) if m else raw or "").strip()


def render_table(shared: GOShared, ev: GOEvidence | None, statistics: str, unit: str,
                 include_definition: bool, results=None):
    """
    The evidence. Raw statistics -> one counts table (sorted by GO id, so order carries no hint).
    Corrected statistics -> (enriched block, depleted block), each strongest-first by p. Splitting
    by sign means the top of the enriched block is exactly the no-LLM ranking (go_enrichment v3/v4)
    and depleted terms can never be mistaken for enriched ones by reading down a list.
    `ev=None` renders the no-evidence control (candidate list only, no counts).
    """
    b = shared.builder

    def line(text: str, go_id: str) -> str:
        return text + (f"\n    definition: {go_definition(shared, go_id) or '(none)'}" if include_definition else "")

    if ev is None:
        return "\n".join(f"{g} | {b.labels[g]}" for g in b.candidates)
    if statistics == "raw":
        return "\n".join(line(f"{t.go_id} | {t.label} | carried by {t.K} | gene expressed in {t.k_pos} of them", t.go_id)
                         for t in sorted(ev.terms, key=lambda t: t.go_id))

    def row(r) -> str:
        return line(f"{r.go_id} | {r.label} | expressed in {r.k_pos}/{r.K} ({r.observed:.0%}) vs expected "
                    f"{r.expected:.0%} | excess {r.excess:+.0%} | q={r.q_value:.1e}", r.go_id)
    live = [r for r in results if r.tested]
    up = sorted((r for r in live if r.excess >= 0), key=lambda r: (r.p_value, -r.excess, r.go_id))
    down = sorted((r for r in live if r.excess < 0), key=lambda r: (r.p_value, r.excess, r.go_id))
    return ("\n".join(row(r) for r in up) or "(none)"), ("\n".join(row(r) for r in down) or "(none)")


def build_prompt(shared: GOShared, cfg: dict, output_mode: str, evidence: str, ev: GOEvidence | None,
                 results, include_definition: bool, cache_bust: str | None = None) -> str:
    """
    `cache_bust`, if given, prepends an inert metadata-looking line so the byte-for-byte prompt prefix
    differs per call. DIAGNOSTIC USE ONLY (scripts/measure_llm_noise.py) — never set in a real
    experiment run. Ollama caches the KV-state of a prompt's processing and replays an EXACT repeated
    prefix deterministically (confirmed empirically: repeating the identical prompt gave bit-identical
    output on 20/20 test genes, with a matching latency drop). That makes "run the same prompt N times"
    measure one real draw plus N-1 guaranteed cache echoes, not N independent samples. Breaking the
    prefix forces a genuine fresh computation each time so repeat-run variance can be measured honestly.
    """
    unit, statistics = cfg["unit"], cfg["statistics"]
    unit_desc, unit_word = UNIT_TEXT[unit]
    template = (PROMPTS_DIR / f"go_llm_{statistics}_{output_mode}_v1.txt").read_text()
    if cache_bust:
        template = f"[internal request id, not part of the evidence: {cache_bust}]\n\n" + template
    if evidence == "none":
        summary = "No expression evidence is available for this gene."
        listing = ("Candidate GO terms (no expression counts are available):\n"
                   + render_table(shared, None, statistics, unit, include_definition)
                   if output_mode == "constrained" else "(no evidence available)")
        parts = {"table": listing} if statistics == "raw" else {"enriched_table": listing, "depleted_table": "(none)"}
    else:
        summary = f"The gene is reliably expressed in {ev.n_pos} of {ev.n_called} {unit_word} ({ev.gene_rate:.0%})."
        rendered = render_table(shared, ev, statistics, unit, include_definition, results)
        parts = {"table": rendered} if statistics == "raw" else {"enriched_table": rendered[0], "depleted_table": rendered[1]}
    return template.format(unit_description=unit_desc, gene_summary=summary, **parts)


# ---- output handling --------------------------------------------------------------
def _entries(parsed: dict, key: str) -> list[tuple[str, float | None]]:
    out = []
    for p in parsed.get("predictions") or []:
        if isinstance(p, str):
            out.append((p, None))
        elif isinstance(p, dict):
            val = p.get(key) or p.get("go_id") or p.get("name") or p.get("term") or p.get("id")
            conf = p.get("confidence")
            if val:
                out.append((str(val), float(conf) if isinstance(conf, (int, float)) else None))
    return out


def resolve_predictions(shared: GOShared, parsed: dict, output_mode: str, max_predictions: int):
    """(ranked rows, diagnostics). Constrained keeps only candidate ids; free-form resolves names."""
    cand, go = set(shared.builder.candidates), shared.truth.go
    entries = _entries(parsed, "go_id" if output_mode == "constrained" else "name")[:max_predictions]
    rows, invalid, seen = [], [], set()
    for text, conf in entries:
        if output_mode == "constrained":
            m = _GO_ID_RE.search(text)
            gid = f"GO:{m.group(1)}" if m else None
            gid = gid if gid in cand else None
        else:
            m = _GO_ID_RE.fullmatch(text.strip()) if text.strip().upper().startswith("GO:") else None
            gid = go.resolve(f"GO:{m.group(1)}") if m else go.resolve_name(text)
        if gid is None:
            invalid.append(text)
        elif gid not in seen:
            seen.add(gid)
            rows.append({"go_id": gid, "label": go.label(gid), "rank": len(rows) + 1,
                         "score": conf if conf is not None else 0.0, "as_written": text})
    in_table = sum(1 for r in rows if r["go_id"] in cand)
    diag = {"n_predictions_raw": len(entries), "n_valid": len(rows), "n_invalid": len(invalid),
            "invalid_terms": invalid, "n_in_table": in_table, "n_outside_table": len(rows) - in_table,
            # Share of the answer that merely re-lists evidence terms. ~1.0 means the model read the
            # statistics back; lower means it named processes the evidence does not contain.
            "frac_in_table": in_table / len(rows) if rows else float("nan")}
    return rows, diag


def get_llm(model: str, temperature: float, seed: int):
    key = (model, temperature, seed)
    if key not in _LLM_CACHE:
        _LLM_CACHE[key] = make_chat_llm(model=model, temperature=temperature, seed=seed,
                                        format="json", num_ctx=compute_num_ctx(model))
    return _LLM_CACHE[key]


# ---- one run ----------------------------------------------------------------------------
def run_gene(gene_id: str, gene_symbol: str, args, shared: GOShared) -> str | None:
    """Returns the MLflow run id (or None if the LLM output could not be parsed)."""
    cfg = PROMPT_VERSIONS[args.prompt_version]
    unit, statistics = cfg["unit"], cfg["statistics"]
    model = resolve_chat_model(args.model)
    if shared.truth is None:
        raise SystemExit("go_llm needs GOShared with truth loaded (scoring is inline).")
    if statistics == "corrected" and shared.baselines is None:
        raise SystemExit("corrected versions need the GO baselines: python scripts/build_go_baselines.py")

    # Which gene's evidence does the prompt show?
    donor = None
    if args.evidence == "true":
        shown_gene = gene_id
    elif args.evidence == "mismatched":
        donor = shared.choose_donor(gene_id, unit, args.seed, args.donor_tolerance)
        shown_gene = donor
    else:
        shown_gene = None
    ev = shared.builder.evidence(shown_gene, unit) if shown_gene else None
    results = (corrected_stats(ev, shared.baselines[unit], args.min_term_size, shared.builder.candidates)
               if (ev is not None and statistics == "corrected") else None)

    parent_run_id = get_or_create_gene_parent_run(
        approach=APPROACH, gene_id=gene_id, gene_symbol=gene_symbol, species=args.species,
        dataset_hash=shared.ds.dataset_hash, expression_summary=shared.ds.expression_summary(gene_id))

    tags = {"condition": f"{APPROACH}:{args.prompt_version}:{args.output_mode}:{args.evidence}",
            "output_mode": args.output_mode, "evidence": args.evidence, "stratum": shared.stratum(gene_id),
            "code_hash": fingerprint(APPROACH)}   # content hash of prompts + code that shape this run
    if args.gene_split:
        tags["gene_split"] = args.gene_split
    if getattr(args, "unfrozen_override", False):
        tags["unfrozen_override"] = True        # test-split run started without a matching freeze
    ctx = RunContext(
        approach=APPROACH, approach_version=args.prompt_version, gene_id=gene_id, gene_symbol=gene_symbol,
        species=args.species, model=model, prompt_mode=f"go_{statistics}_{args.output_mode}",
        input_set="contrast", temperature=args.temperature, seed=args.seed,
        dataset_hash=shared.ds.dataset_hash, prompt_version=f"go_llm_{statistics}_{args.output_mode}_v1",
        extra_params={
            "unit": unit, "statistics": statistics, "output_mode": args.output_mode, "evidence": args.evidence,
            "donor_gene_id": donor or "", "donor_tolerance": args.donor_tolerance,
            "go_depth": args.go_depth, "min_term_size": args.min_term_size,
            "n_candidates": len(shared.builder.candidates), "max_predictions": args.max_predictions,
            "evidence_excluded": ",".join(sorted(shared.truth.excluded_evidence)),
            "go_data_version": shared.truth.go.data_version, "include_definition": args.include_definition,
            "cl_data_version": shared.lookup.provenance.get("cl_data_version"),
            **({"baselines_key": baselines_key(shared.builder)} if statistics == "corrected" else {}),
        },
        extra_tags=tags,
    )

    with tracked_run(ctx, parent_run_id=parent_run_id) as run:
        print(f"\n[{ctx.approach_version}:{args.output_mode}:{args.evidence}] {gene_symbol}  run {run.info.run_id}")
        prompt = build_prompt(shared, cfg, args.output_mode, args.evidence, ev, results, args.include_definition,
                             cache_bust=getattr(args, "cache_bust", None))

        ok, hits = verify_blinding(prompt, [gene_symbol] if gene_symbol != gene_id else [])
        mlflow.set_tag("blinding_verified", ok)
        if not ok:
            mlflow.set_tag("status", "FAILED_BLINDING")
            raise SystemExit(f"Blinding check failed — prompt contains: {hits}")
        alias_hits = [a for a in shared.aliases(gene_id)
                      if len(a) >= 3 and re.search(rf"(?<![A-Za-z0-9]){re.escape(a)}(?![A-Za-z0-9])", prompt, re.I)]
        mlflow.set_tag("blinding_alias_hits", ",".join(alias_hits))

        log_text_artifact(prompt, "prompt.txt")
        if ev is not None:
            log_json_artifact({"shown_gene": shown_gene, "n_pos": ev.n_pos, "n_called": ev.n_called,
                               "terms": [t.__dict__ for t in ev.terms]}, "evidence_table.json")
        num_ctx = compute_num_ctx(model)
        mlflow.log_param("num_ctx", num_ctx)
        llm = get_llm(model, args.temperature, args.seed)
        try:
            parsed, raw_text, elapsed, retries, meta = call_llm_with_retry(llm, prompt, REQUIRED_RESPONSE_KEYS)
        except RuntimeError as exc:
            log_text_artifact(str(exc), "parse_error.txt")
            mlflow.set_tag("status", "FAILED_PARSE")
            print(exc)
            return None

        actual = meta.get("prompt_eval_count")
        if actual is not None:
            mlflow.log_metrics({"actual_prompt_tokens": actual, "context_utilization": actual / num_ctx})
        mlflow.log_metrics({"latency_s": elapsed, "parse_retries": retries})
        log_text_artifact(raw_text, "response_raw.txt")
        log_json_artifact(parsed, "response_parsed.json")

        rows, diag = resolve_predictions(shared, parsed, args.output_mode, args.max_predictions)
        mlflow.log_metrics({"n_predictions_raw": diag["n_predictions_raw"], "n_valid": diag["n_valid"],
                            "n_invalid": diag["n_invalid"], "n_outside_table": diag["n_outside_table"],
                            "invalid_rate": diag["n_invalid"] / max(diag["n_predictions_raw"], 1)})
        if diag["n_valid"]:
            mlflow.log_metric("frac_in_table", diag["frac_in_table"])
        if diag["invalid_terms"]:
            log_json_artifact(diag["invalid_terms"], "invalid_predictions.json")
        mlflow.set_tag("no_valid_predictions", diag["n_valid"] == 0)

        # Headroom is only defined against the candidate vocabulary, so constrained mode only.
        metrics = log_scored_predictions(rows, None, shared.truth, gene_id,
                                         shared.model if args.output_mode == "constrained" else None)
        mlflow.set_tag("status", "COMPLETED")
        print(f"  {diag['n_valid']} valid / {diag['n_invalid']} invalid   F1@3={metrics['f1_at_3']:.3f}"
              + (f"   headroom@3={metrics['headroom_at_3']:.3f}" if "headroom_at_3" in metrics else "")
              + f"   ({elapsed:.1f}s)")
        for r in rows[:5]:
            print(f"    {r['rank']}. {r['label']}")
        return run.info.run_id


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gene", required=True, help="Canonical Ensembl gene ID")
    ap.add_argument("--gene-symbol", default=None, help="For MLflow tags and the blinding check only — never sent to the LLM")
    ap.add_argument("--species", default="human")
    ap.add_argument("--model", default=None, help="Overrides CHAT_MODEL from .env")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--prompt-version", choices=list(PROMPT_VERSIONS), default="v3",
                    help="v1/v2 raw counts; v3/v4 corrected table; odd = cell types, even = rows")
    ap.add_argument("--output-mode", choices=["constrained", "freeform"], default="constrained")
    ap.add_argument("--evidence", choices=["true", "mismatched", "none"], default="true")
    ap.add_argument("--donor-tolerance", type=float, default=0.05,
                    help="mismatched: donor's overall positive rate must be within this of the target's")
    ap.add_argument("--max-predictions", type=int, default=10)
    ap.add_argument("--go-depth", type=int, default=3)
    ap.add_argument("--min-term-size", type=int, default=3)
    ap.add_argument("--include-definition", action=argparse.BooleanOptionalAction, default=False,
                    help="Ablation: show each GO term's definition. Off by default for information parity with the baselines")
    ap.add_argument("--gene-split", choices=["dev", "test", "smoke"], default=None)
    ap.add_argument("--cache-bust", default=None,
                    help="DIAGNOSTIC ONLY (scripts/measure_llm_noise.py): breaks Ollama's exact-prompt "
                         "cache so repeats are genuinely independent calls. Never use in a real run.")
    return ap


def make_args(**overrides) -> argparse.Namespace:
    ns = build_parser().parse_args(["--gene", overrides.pop("gene", "X")])
    ns.__dict__.update(overrides)
    return ns


def main() -> None:
    args = build_parser().parse_args()
    shared = GOShared.load(args.dataset, args.go_depth, args.min_term_size,
                           need_truth=True, need_baselines=PROMPT_VERSIONS[args.prompt_version]["statistics"] == "corrected")
    if not shared.ds.has_gene(args.gene):
        raise SystemExit(f"Gene {args.gene!r} not present in dataset. Aborting.")
    run_gene(args.gene, args.gene_symbol or args.gene, args, shared)


if __name__ == "__main__":
    main()
