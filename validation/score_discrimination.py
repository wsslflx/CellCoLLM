#!/usr/bin/env python3
"""
Primary validation layer: does a run's inferred property actually SEPARATE the
cell types that express the gene from the ones that don't?

Unlike the GO-match layer (validation/score_go_match.py, now a demoted secondary
signal), this judges the property against the thing it is actually a claim about
— cell types — using the dataset itself as ground truth. No GO annotation needed,
so it works on every gene, including unannotated ones.

  coverage      = P(property applies | positive set)
  leakage       = P(property applies | negative set)
  discrimination (Youden's J) = coverage - leakage    range -1..+1, 0 = chance

The negative set is a matched control by construction (same gene, same atlas), so
chance and ceiling are defined a priori and a vague property scores ~0 automatically
(coverage ~ leakage) — the §6.2.4 specificity check comes free. The judge never sees
the gene, nor which set any cell type came from (§6.2.1/§6.2.2).

GROUNDING ABLATION: every scoring run makes two calls over the identical sample —
one showing only cell-type/tissue labels, one additionally showing OBO definitions
and is_a ancestors. `disc_grounding_effect` = grounded_j - ungrounded_j quantifies
how much the judge relies on supplied data versus its own recall. If it is ~0, the
judgement is recall-driven and you know that, quantitatively.

Usage:
    python validation/score_discrimination.py --run-id <mlflow_run_id> --judge-model llama3.3:70b
    python validation/score_discrimination.py --run-id <id> --judge-model llama3.3:70b \\
        --property-override "these are cells" --calibration-label floor
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import mlflow

from core.data_loader import GeneExpressionDataset, parse_pair
from core.llm_backend import compute_num_ctx, make_chat_llm
from core.mlflow_utils import verify_blinding
from core.ontology_lookup import OntologyLookup
from core.structured_llm import call_llm_with_retry

PROMPTS_DIR = Path(__file__).parent / "prompts"
REQUIRED_RESPONSE_KEYS = {"judgments"}
CALIBRATION_EXPERIMENT = "CellCoLLM/validation_calibration"

# "applies" is the strict numerator; "partial" counts half in the *_weighted variants.
# "insufficient_information" is excluded from BOTH numerator and denominator (§6.3) —
# "cannot tell" must never be silently folded into "does not apply".
STRICT_WEIGHTS = {"applies": 1.0, "partial": 0.0, "does_not_apply": 0.0}
WEIGHTED_WEIGHTS = {"applies": 1.0, "partial": 0.5, "does_not_apply": 0.0}

DEFAULT_K_PER_SET = 40
DEFAULT_HIERARCHY_DEPTH = 2


def get_source_run_info(run_id: str) -> dict:
    run = mlflow.get_run(run_id)
    tags, params = run.data.tags, run.data.params
    property_text, property_source = None, None
    try:
        path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="response_parsed.json")
        property_text = json.loads(Path(path).read_text()).get("property")
        property_source = "response_parsed.json"
    except Exception:
        pass
    raw_text = None
    if property_text is None:
        try:
            path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="response.txt")
            raw_text = Path(path).read_text().strip()
            property_source = "response.txt(needs_extraction)"
        except Exception:
            pass
    return {
        "gene_id": params.get("gene_id"),
        "gene_symbol": tags.get("gene_symbol"),
        "input_set": tags.get("input_set"),
        "approach": tags.get("approach"),
        "approach_version": tags.get("approach_version"),
        "generator_model": params.get("model"),
        "status": tags.get("status"),
        "abstained": tags.get("abstained") == "True",
        "property_text": property_text,
        "raw_text": raw_text,
        "property_source": property_source,
    }


def extract_property_from_free_text(raw_text: str, judge_model: str, seed: int) -> tuple[str | None, str]:
    """naive v1 has no structured output — its response is an essay that opens with
    boilerplate about ID notation. Scoring that verbatim guarantees a meaningless 0,
    so pull out the actual biological claim with one extra call (logged for audit)."""
    template = (PROMPTS_DIR / "extract_property.txt").read_text()
    prompt = template.format(response_text=raw_text[:12000])
    llm = make_chat_llm(model=judge_model, temperature=0.0, seed=seed, format="json")
    parsed, raw, _, _, _ = call_llm_with_retry(llm, prompt, {"property"})
    return parsed.get("property"), raw


def sample_balanced(
    ds: GeneExpressionDataset, gene_id: str, k: int, rng: random.Random,
) -> tuple[list[dict], dict]:
    """k positive + k negative CL|UBERON pairs, shuffled together, set labels retained
    only in the harness (never shown to the judge). Pairs, not CL terms: 59-74% of CL
    terms occur in BOTH sets (same cell type, different tissue), so a CL-only sample
    would carry ambiguous labels."""
    pos, neg = ds.positive_cell_types(gene_id), ds.negative_cell_types(gene_id)
    pos_cl, neg_cl = {parse_pair(p)[0] for p in pos}, {parse_pair(p)[0] for p in neg}
    overlap_cl = pos_cl & neg_cl

    pos_s = rng.sample(pos, min(k, len(pos)))
    neg_s = rng.sample(neg, min(k, len(neg)))
    entries = [{"pair": p, "is_positive": True} for p in pos_s] + [{"pair": p, "is_positive": False} for p in neg_s]
    rng.shuffle(entries)

    sampled_cl = {parse_pair(e["pair"])[0] for e in entries}
    n_ambiguous = sum(1 for e in entries if parse_pair(e["pair"])[0] in overlap_cl)
    diagnostics = {
        "disc_n_positive_sampled": len(pos_s),
        "disc_n_negative_sampled": len(neg_s),
        "disc_n_distinct_cl_in_sample": len(sampled_cl),
        # How much of the sample is unresolvable by cell-type identity alone — sets the
        # realistic ceiling for J. Measured at 59-74% dataset-wide.
        "disc_sample_cl_overlap_rate": n_ambiguous / len(entries) if entries else 0.0,
    }
    return entries, diagnostics


def render_entries(entries: list[dict], lookup: OntologyLookup, grounded: bool, depth: int) -> str:
    lines = []
    for i, e in enumerate(entries):
        cl_id, ub_id = parse_pair(e["pair"])
        cl, ub = lookup.resolve_cl(cl_id), lookup.resolve_uberon(ub_id)
        cl_text = cl.label if cl.resolved else f"(unresolved: {cl_id})"
        ub_text = ub.label if ub.resolved else f"(unresolved: {ub_id})"
        if not grounded:
            lines.append(f"{i}. {cl_text} (in {ub_text})")
            continue
        block = [f"{i}. {cl_text} (in {ub_text})"]
        definition = cl.definition if cl.resolved and cl.definition else "(no definition available)"
        block.append(f"     Cell type definition: {definition}")
        ancestors = lookup.ancestors("cl", cl_id, depth) if cl.resolved else []
        if ancestors:
            chain = " -> ".join(a.label if a.resolved else f"(unresolved: {a.id})" for a in ancestors)
            block.append(f"     Broader categories: {chain}")
        ub_def = ub.definition if ub.resolved and ub.definition else "(no definition available)"
        block.append(f"     Tissue definition: {ub_def}")
        lines.append("\n".join(block))
    return "\n".join(lines)


def compute_metrics(entries: list[dict], verdicts: dict[int, str], label: str) -> tuple[dict, list[dict]]:
    """coverage/leakage/J under strict and partial-weighted scoring. Missing indices are
    treated as insufficient_information and counted explicitly (score_go_match dropped
    them silently)."""
    buckets = {(True, "strict"): [], (False, "strict"): [], (True, "weighted"): [], (False, "weighted"): []}
    n_insufficient = n_missing = 0
    detail = []
    for i, e in enumerate(entries):
        verdict = verdicts.get(i)
        if verdict is None:
            n_missing += 1
            verdict = "insufficient_information"
        detail.append({"index": i, "condition": label, "pair": e["pair"], "is_positive": e["is_positive"],
                       "verdict": verdict})
        if verdict == "insufficient_information" or verdict not in STRICT_WEIGHTS:
            n_insufficient += 1
            continue
        buckets[(e["is_positive"], "strict")].append(STRICT_WEIGHTS[verdict])
        buckets[(e["is_positive"], "weighted")].append(WEIGHTED_WEIGHTS[verdict])

    def rate(vals): return sum(vals) / len(vals) if vals else 0.0
    coverage, leakage = rate(buckets[(True, "strict")]), rate(buckets[(False, "strict")])
    cov_w, leak_w = rate(buckets[(True, "weighted")]), rate(buckets[(False, "weighted")])
    metrics = {
        f"disc_{label}_coverage": coverage,
        f"disc_{label}_leakage": leakage,
        f"disc_{label}_j": coverage - leakage,
        f"disc_{label}_coverage_weighted": cov_w,
        f"disc_{label}_leakage_weighted": leak_w,
        f"disc_{label}_j_weighted": cov_w - leak_w,
        f"disc_{label}_insufficient_rate": n_insufficient / len(entries) if entries else 0.0,
        f"disc_{label}_n_missing_verdicts": n_missing,
        f"disc_{label}_n_scored_positive": len(buckets[(True, "strict")]),
        f"disc_{label}_n_scored_negative": len(buckets[(False, "strict")]),
    }
    return metrics, detail


def score_condition(
    entries: list[dict], lookup: OntologyLookup, property_text: str, grounded: bool, depth: int,
    judge_model: str, seed: int, forbidden_terms: list[str],
) -> tuple[dict, list[dict], str, str]:
    label = "grounded" if grounded else "ungrounded"
    template = (PROMPTS_DIR / f"discrimination_{label}.txt").read_text()
    prompt = template.format(property_text=property_text,
                             cell_type_list=render_entries(entries, lookup, grounded, depth))

    ok, hits = verify_blinding(prompt, forbidden_terms=forbidden_terms)
    if not ok:
        raise SystemExit(f"[{label}] Blinding check failed — prompt contains: {hits}")

    print(f"[{label}] judging {len(entries)} cell types with {judge_model}...")
    llm = make_chat_llm(model=judge_model, temperature=0.0, seed=seed, format="json",
                        num_ctx=compute_num_ctx(judge_model))
    parsed, raw_text, _, _, _ = call_llm_with_retry(llm, prompt, REQUIRED_RESPONSE_KEYS)
    verdicts = {j["index"]: j["verdict"] for j in parsed["judgments"] if "index" in j and "verdict" in j}
    metrics, detail = compute_metrics(entries, verdicts, label)
    return metrics, detail, prompt, raw_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True, help="MLflow run_id of the naive/enriched run to score")
    parser.add_argument("--judge-model", required=True,
                         help="Must differ from the scored run's generator model (no fallback/default)")
    parser.add_argument("--k-per-set", type=int, default=DEFAULT_K_PER_SET)
    parser.add_argument("--hierarchy-depth", type=int, default=DEFAULT_HIERARCHY_DEPTH)
    parser.add_argument("--seed", type=int, default=0, help="XOR'd with a run_id-derived seed")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--no-ablation", action="store_true",
                         help="Run only the grounded condition (skips the recall-dependence measurement)")
    parser.add_argument("--property-override", default=None,
                         help="Score this property instead of the run's own (calibration floor/ceiling)")
    parser.add_argument("--calibration-label", default=None,
                         help="Required with --property-override; logs to the calibration experiment instead")
    args = parser.parse_args()
    if bool(args.property_override) != bool(args.calibration_label):
        parser.error("--property-override and --calibration-label must be used together")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    info = get_source_run_info(args.run_id)
    if not info["gene_id"]:
        raise SystemExit(f"Could not read gene_id from run {args.run_id!r}. Aborting.")
    is_calibration = args.calibration_label is not None

    print(f"Run {args.run_id} ({info['approach']}:{info['approach_version']}, "
          f"gene={info['gene_id']}, set={info['input_set']})")

    if args.judge_model == info["generator_model"]:
        raise SystemExit(
            f"--judge-model ({args.judge_model!r}) must differ from the scored run's generator model "
            f"({info['generator_model']!r}) — generator/scorer independence is a hard requirement (§6.2.1)."
        )
    if not is_calibration:
        if info["status"] != "COMPLETED" or info["abstained"]:
            print(f"Nothing to score (status={info['status']!r}, abstained={info['abstained']}) — aborting.")
            return
        already = mlflow.get_run(args.run_id).data.params.get("disc_judge_model")
        if already is not None:
            raise SystemExit(
                f"Run {args.run_id!r} was already scored (disc_judge_model={already!r}). MLflow params are "
                f"immutable once logged, so it cannot be rescored in place. Use --property-override with "
                f"--calibration-label to score a different property against this run's sample instead."
            )

    # run_id-derived seed: different sample per run, reproducible for that run. The fixed
    # seed=42 in score_go_match.py is exactly what collapsed its distractors to 4 genes.
    seed = int(hashlib.sha256(args.run_id.encode()).hexdigest()[:8], 16) ^ args.seed
    rng = random.Random(seed)

    extraction_raw = None
    property_text = args.property_override
    if property_text is None:
        property_text = info["property_text"]
        if property_text is None and info["raw_text"]:
            print("Unstructured response (no response_parsed.json) — extracting the property...")
            property_text, extraction_raw = extract_property_from_free_text(
                info["raw_text"], args.judge_model, seed)
        if property_text is None:
            raise SystemExit(f"Could not obtain a property from run {args.run_id!r}. Aborting.")
    print(f"Property: {property_text[:160]}")

    ds = GeneExpressionDataset.load(args.dataset) if args.dataset else GeneExpressionDataset.load()
    lookup = OntologyLookup()
    entries, diagnostics = sample_balanced(ds, info["gene_id"], args.k_per_set, rng)
    print(f"Sampled {diagnostics['disc_n_positive_sampled']} positive + "
          f"{diagnostics['disc_n_negative_sampled']} negative pairs "
          f"(CL-overlap rate {diagnostics['disc_sample_cl_overlap_rate']:.0%} — the irreducible ceiling)")

    gene_symbol = info["gene_symbol"] or info["gene_id"]
    forbidden = [gene_symbol] if gene_symbol != info["gene_id"] else []

    conditions = [True] if args.no_ablation else [False, True]
    all_metrics, all_detail, artifacts = dict(diagnostics), [], {}
    t0 = time.time()
    for grounded in conditions:
        metrics, detail, prompt, raw = score_condition(
            entries, lookup, property_text, grounded, args.hierarchy_depth,
            args.judge_model, seed, forbidden)
        label = "grounded" if grounded else "ungrounded"
        all_metrics.update(metrics)
        all_detail.extend(detail)
        artifacts[f"disc/prompt_{label}.txt"] = prompt
        artifacts[f"disc/response_{label}_raw.txt"] = raw
    all_metrics["disc_latency_s"] = time.time() - t0
    if not args.no_ablation:
        all_metrics["disc_grounding_effect"] = all_metrics["disc_grounded_j"] - all_metrics["disc_ungrounded_j"]

    params = {
        "disc_judge_model": args.judge_model,
        "disc_k_per_set": args.k_per_set,
        "disc_hierarchy_depth": args.hierarchy_depth,
        "disc_seed": seed,
        "disc_property_source": "override" if args.property_override else info["property_source"],
        "disc_cl_data_version": lookup.provenance.get("cl_data_version"),
        "disc_uberon_data_version": lookup.provenance.get("uberon_data_version"),
    }

    if is_calibration:
        mlflow.set_experiment(CALIBRATION_EXPERIMENT)
        ctx = mlflow.start_run(run_name=f"calib_{args.calibration_label}_{info['gene_id']}")
    else:
        ctx = mlflow.start_run(run_id=args.run_id)
    with ctx as run:
        print(f"Logging to run {run.info.run_id}"
              f"{' (' + CALIBRATION_EXPERIMENT + ')' if is_calibration else ' (co-located with generation run)'}")
        mlflow.log_params(params)
        mlflow.log_metrics(all_metrics)
        mlflow.set_tag("disc_status", "COMPLETED")
        if is_calibration:
            mlflow.set_tags({"calibration_label": args.calibration_label, "sampled_from_run_id": args.run_id,
                             "gene_id": info["gene_id"], "role": "validation_calibration"})
            mlflow.log_param("disc_property_text", property_text[:480])
        for name, text in artifacts.items():
            mlflow.log_text(text, artifact_file=name)
        mlflow.log_dict(all_detail, "disc/judgments_full.json")
        mlflow.log_dict(entries, "disc/sample.json")
        if extraction_raw is not None:
            mlflow.log_text(extraction_raw, artifact_file="disc/extracted_property.txt")

    print()
    for label in (["grounded"] if args.no_ablation else ["ungrounded", "grounded"]):
        m = all_metrics
        print(f"  {label:11s} coverage={m[f'disc_{label}_coverage']:.2f}  "
              f"leakage={m[f'disc_{label}_leakage']:.2f}  "
              f"J={m[f'disc_{label}_j']:+.3f}  "
              f"(insufficient {m[f'disc_{label}_insufficient_rate']:.0%})")
    if not args.no_ablation:
        print(f"  grounding_effect (grounded_J - ungrounded_J) = {all_metrics['disc_grounding_effect']:+.3f}")
        print("    ~0 means the ontology data changed nothing -> the judge is running on recall.")


if __name__ == "__main__":
    main()
