#!/usr/bin/env python3
"""
Validation layer: rate a completed naive/enriched run's inferred property
against the gene's actual known GO (biological process) annotation — a fast
numeric estimate (PIPELINE_REQUIREMENTS.md §6.3's "external accuracy" check),
not the full per-cell-type coverage/leakage machinery (§6, deferred).

Design (see plan discussion): the judge never sees the gene name/symbol, only
the property text and a shuffled, unlabeled list combining the gene's real GO
terms with terms from randomly drawn distractor genes (background/specificity
control, §6.2.4) — one single structured-JSON call, scored by the harness,
not the LLM. Judge model must differ from the scored run's generator model
(§6.2.1's independence requirement, enforced not just advised).

Usage:
    python validation/score_go_match.py --run-id <mlflow_run_id> --judge-model llama3.3:70b
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import mlflow

from core.data_loader import read_gene_ids
from core.go_lookup import GOLookup
from core.llm_backend import make_chat_llm
from core.mlflow_utils import verify_blinding
from core.structured_llm import call_llm_with_retry

# Results are logged into the SAME MLflow run being scored (mlflow.start_run(run_id=...)
# reopens it), not a separate "validation" experiment — everything about one generation
# run lives in one place. Every new metric/tag/param uses a go_match_ prefix, and
# artifacts go under a go_match/ subfolder, so nothing collides with the generation
# run's own logged data (its own "status", "latency_s", "prompt.txt", etc.).
PROMPT_PATH = Path(__file__).parent / "prompts" / "go_match_score.txt"
REQUIRED_RESPONSE_KEYS = {"judgments"}
VERDICT_WEIGHTS = {"match": 1.0, "partial": 0.5, "no_match": 0.0}
DEFAULT_TOP_N_GO_TERMS = 10
DEFAULT_N_DISTRACTOR_GENES = 3
MAX_DISTRACTOR_REDRAWS = 10


def get_source_run_info(run_id: str) -> dict:
    run = mlflow.get_run(run_id)
    tags = run.data.tags
    params = run.data.params
    status = tags.get("status")
    abstained = tags.get("abstained") == "True"

    property_text = None
    try:
        path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="response_parsed.json")
        property_text = json.loads(Path(path).read_text()).get("property")
    except Exception:
        pass
    if property_text is None:
        try:
            path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="response.txt")
            property_text = Path(path).read_text().strip()
        except Exception:
            property_text = None

    return {
        "gene_id": params.get("gene_id"),
        "gene_symbol": tags.get("gene_symbol"),
        "input_set": tags.get("input_set"),
        "approach": tags.get("approach"),
        "approach_version": tags.get("approach_version"),
        "generator_model": params.get("model"),
        "status": status,
        "abstained": abstained,
        "property_text": property_text,
    }


def sample_distractor_terms(
    lookup: GOLookup, all_gene_ids: list[str], exclude_gene_id: str, n_distractors: int,
    top_n: int, rng: random.Random,
) -> tuple[list[str], list[dict], int]:
    """Returns (distractor_gene_ids_used, distractor_terms, n_redraws)."""
    candidates = [g for g in all_gene_ids if g != exclude_gene_id]
    rng.shuffle(candidates)
    used_ids, terms, redraws = [], [], 0
    for candidate in candidates:
        if len(used_ids) >= n_distractors:
            break
        candidate_terms = lookup.get_bp_terms(candidate)[:top_n]
        if not candidate_terms:
            redraws += 1
            if redraws > MAX_DISTRACTOR_REDRAWS:
                break
            continue
        used_ids.append(candidate)
        terms.extend(candidate_terms)
    return used_ids, terms, redraws


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True, help="MLflow run_id of the naive/enriched run to score")
    parser.add_argument("--judge-model", required=True,
                         help="Must differ from the scored run's generator model (no fallback/default)")
    parser.add_argument("--top-n-go-terms", type=int, default=DEFAULT_TOP_N_GO_TERMS)
    parser.add_argument("--n-distractor-genes", type=int, default=DEFAULT_N_DISTRACTOR_GENES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default=None, help="Path to the binarised expression tsv (default: repo root)")
    args = parser.parse_args()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    info = get_source_run_info(args.run_id)
    if not info["gene_id"]:
        raise SystemExit(f"Could not read gene_id from run {args.run_id!r}. Aborting.")

    print(f"Scoring run {args.run_id} ({info['approach']}:{info['approach_version']}, "
          f"gene={info['gene_id']}, set={info['input_set']})")

    if info["status"] != "COMPLETED" or info["abstained"]:
        print(f"Nothing to score (status={info['status']!r}, abstained={info['abstained']}) — aborting before any call.")
        return
    if info["property_text"] is None:
        raise SystemExit(f"Could not extract a property/response text from run {args.run_id!r}. Aborting.")
    if args.judge_model == info["generator_model"]:
        raise SystemExit(
            f"--judge-model ({args.judge_model!r}) must differ from the scored run's generator model "
            f"({info['generator_model']!r}) — independence between generator and scorer is a hard requirement here."
        )
    existing_judge = mlflow.get_run(args.run_id).data.params.get("go_match_judge_model")
    if existing_judge is not None:
        raise SystemExit(
            f"Run {args.run_id!r} was already scored (go_match_judge_model={existing_judge!r}). "
            f"MLflow params are immutable once logged, so this run can't be rescored with different "
            f"settings in place — results live in the same run as the generation data, by design, "
            f"so there's no separate validation run to just delete and redo."
        )

    gene_id = info["gene_id"]
    gene_symbol = info["gene_symbol"] or gene_id
    rng = random.Random(args.seed)

    all_gene_ids = read_gene_ids(args.dataset) if args.dataset else read_gene_ids()

    lookup = GOLookup()
    real_terms = lookup.get_bp_terms(gene_id)[: args.top_n_go_terms]
    if not real_terms:
        print(f"No non-IEA BP GO terms found for {gene_id} — nothing to score against (thin annotation). Aborting.")
        return

    distractor_gene_ids, distractor_terms, n_redraws = sample_distractor_terms(
        lookup, all_gene_ids, gene_id, args.n_distractor_genes, args.top_n_go_terms, rng,
    )
    print(f"Real GO terms: {len(real_terms)}. Distractor genes: {distractor_gene_ids} "
          f"({len(distractor_terms)} terms, {n_redraws} redraw(s)).")

    entries = [{"is_real": True, **t} for t in real_terms] + [{"is_real": False, **t} for t in distractor_terms]
    rng.shuffle(entries)
    index_map = {i: e for i, e in enumerate(entries)}
    term_list_text = "\n".join(f"{i}. {e['term']} ({e['go_id']})" for i, e in index_map.items())

    template = PROMPT_PATH.read_text()
    prompt = template.format(property_text=info["property_text"], term_list=term_list_text)

    blinding_ok, hits = verify_blinding(prompt, forbidden_terms=[gene_symbol] if gene_symbol != gene_id else [])
    if not blinding_ok:
        raise SystemExit(f"Blinding check failed on the validation prompt — contains: {hits}")

    with mlflow.start_run(run_id=args.run_id):
        print(f"Attaching go_match results to existing run: {args.run_id}")
        mlflow.log_params({
            "go_match_judge_model": args.judge_model,
            "go_match_go_source": "mygene.info(lazy-cached)",
            "go_match_top_n_go_terms": args.top_n_go_terms,
            "go_match_n_distractor_genes": args.n_distractor_genes,
            "go_match_distractor_gene_ids": distractor_gene_ids,
            "go_match_seed": args.seed,
        })
        mlflow.set_tag("go_match_blinding_verified", blinding_ok)
        mlflow.log_text(prompt, artifact_file="go_match/prompt.txt")

        llm = make_chat_llm(model=args.judge_model, temperature=0.0, seed=args.seed, format="json")
        try:
            parsed, raw_text, elapsed, retries, _ = call_llm_with_retry(llm, prompt, REQUIRED_RESPONSE_KEYS)
        except RuntimeError as exc:
            mlflow.log_text(str(exc), artifact_file="go_match/parse_error.txt")
            mlflow.set_tag("go_match_status", "FAILED_PARSE")
            print(exc)
            return

        mlflow.log_text(raw_text, artifact_file="go_match/response_raw.txt")

        verdicts = {j["index"]: j["verdict"] for j in parsed["judgments"] if "index" in j and "verdict" in j}
        real_scores, background_scores = [], []
        full_judgments = []
        for i, entry in index_map.items():
            verdict = verdicts.get(i)
            weight = VERDICT_WEIGHTS.get(verdict)
            full_judgments.append({"index": i, **entry, "verdict": verdict})
            if weight is None:
                continue
            (real_scores if entry["is_real"] else background_scores).append(weight)
        mlflow.log_dict(full_judgments, "go_match/judgments_full.json")

        n_real_terms, n_background_terms = len(real_scores), len(background_scores)
        real_score = sum(1 for s in real_scores if s == 1.0) / n_real_terms if n_real_terms else 0.0
        background_score = sum(1 for s in background_scores if s == 1.0) / n_background_terms if n_background_terms else 0.0
        real_score_weighted = sum(real_scores) / n_real_terms if n_real_terms else 0.0
        background_score_weighted = sum(background_scores) / n_background_terms if n_background_terms else 0.0

        mlflow.log_metrics({
            "go_match_real_score": real_score,
            "go_match_background_score": background_score,
            "go_match_real_score_weighted": real_score_weighted,
            "go_match_background_score_weighted": background_score_weighted,
            "go_match_n_real_terms": n_real_terms,
            "go_match_n_distractor_terms_total": n_background_terms,
            "go_match_n_redraws": n_redraws,
            "go_match_latency_s": elapsed,
            "go_match_parse_retries": retries,
        })
        mlflow.set_tag("go_match_status", "COMPLETED")

        print(f"real_score:       {sum(1 for s in real_scores if s == 1.0)}/{n_real_terms} "
              f"(weighted {real_score_weighted:.2f})")
        print(f"background_score: {sum(1 for s in background_scores if s == 1.0)}/{n_background_terms} "
              f"(weighted {background_score_weighted:.2f})")


if __name__ == "__main__":
    main()
