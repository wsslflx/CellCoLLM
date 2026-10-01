#!/usr/bin/env python3
"""
Analyse the GO-prediction experiment from MLflow: the pre-registered paired contrasts,
the controls, and the signal-existence gates.

TWO FAMILIES, analysed and Holm-corrected separately:

PRIMARY — free-form. The LLM is asked to INTERPRET the evidence and name the GO terms of the gene
itself, so it can leave the 52-term evidence vocabulary. Endpoint: raw F1@3 (`gopred_f1_at_3`);
headroom is not defined here because the ceiling is not the candidate-vocabulary ceiling. The no-LLM
counterpart that can also leave the vocabulary is go_enrichment v5/v6 (co-annotation transfer).
    F1a/b  go_llm v3/v4 (true)  minus  go_enrichment v5/v6          does the LLM add to the same evidence + transfer?
    F2a/b  go_llm v3/v4 (true)  minus  constant prior               beats fixed terms named for every gene?
    F3a/b  go_enrichment v5/v6  minus  constant prior               does the transfer baseline beat the prior?
    F4     go_llm v3 true       minus  go_llm v3 mismatched         is the evidence used?
    F5a/b  go_llm v3/v4         minus  go_llm v1/v2                 do statistics help the LLM?

CONTROL — constrained. The LLM can only re-rank the 52 evidence terms ("how well can it re-weight the
statistics?"). Endpoint: `gopred_headroom_at_3` = (F1@3 - floor@3) / (ceiling@3 - floor@3).
    C1a/b  go_llm v3/v4 (true)   minus  go_enrichment v3/v4        the LLM on the SAME statistics
    C2a/b  go_enrichment v3/v4   minus  go_enrichment v1/v2        value of the corrected statistics
    C3a/b  go_llm v1/v2 (true)   minus  constant prior              raw counts vs no evidence at all
    C4     go_llm v3 true        minus  go_llm v3 mismatched        is the evidence used?
    C5a/b  go_llm v3/v4          minus  go_llm v1/v2                do statistics help the LLM?

Controls computed here, deterministically (no runs needed): constant priors (fit on the dev split only),
and the no-LLM rungs given a donor gene's evidence (scored against the target).

Signal-existence gate, per family: a rung passes if it beats BOTH the constant prior and its own
mismatched-evidence control with a bootstrap CI excluding 0. If no rung passes the family is reported
INCONCLUSIVE — not as "the LLM adds nothing".

Also reported: `frac_in_table` per free-form condition — the share of predicted terms that merely
re-list evidence terms (~1 = the model read the statistics back; low = it named processes the evidence
does not contain).

Usage:
    python scripts/analyze_go_experiment.py --eval-split test
    python scripts/analyze_go_experiment.py --eval-split dev --prior-split dev   # plumbing checks only
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[1] / "approaches" / "go_enrichment"))

import mlflow

import run_go_enrichment as A1
from core.go_enrichment import enrich
from core.go_evidence import corrected_stats, rank_corrected, rank_gprofiler
from core.go_experiment import GOShared
from core.go_ontology import BP_ROOT
from core.go_scoring import CeilingModel, fit_constant_prior, score_ranking
from core.mlflow_utils import _TRACKING_URI
from core.run_identity import FREEZE_PATH, fingerprints, verify_freeze

warnings.filterwarnings("ignore", message="Mean of empty slice")  # conditions with no valid predictions give all-NaN columns
GENES_FILE = Path(__file__).parents[1] / "data" / "go_experiment" / "genes.tsv"
OUT_DIR = Path(__file__).parents[1] / "data" / "go_experiment"
UNIT_OF = {"v1": "cell_types", "v2": "rows", "v3": "cell_types", "v4": "rows", "v5": "cell_types", "v6": "rows"}
STAT_OF = {"v1": "gprofiler", "v2": "gprofiler", "v3": "corrected", "v4": "corrected"}
PRIMARY = "gopred_headroom_at_3"
N_BOOT = 10000

PRIMARY_FF = "gopred_f1_at_3"
CONTRASTS = [   # CONTROL family (constrained, headroom@3)
    ("C1a  LLM adds to same stats (cell types)", "go_llm:v3:constrained:true", "go_enrichment:v3:contrast"),
    ("C1b  LLM adds to same stats (rows)", "go_llm:v4:constrained:true", "go_enrichment:v4:contrast"),
    ("C2a  corrected vs g:Profiler (cell types)", "go_enrichment:v3:contrast", "go_enrichment:v1:positive"),
    ("C2b  corrected vs g:Profiler (rows)", "go_enrichment:v4:contrast", "go_enrichment:v2:positive"),
    ("C3a  LLM raw counts vs constant prior (cell types)", "go_llm:v1:constrained:true", "constant_prior"),
    ("C3b  LLM raw counts vs constant prior (rows)", "go_llm:v2:constrained:true", "constant_prior"),
    ("C4   evidence is used (v3 true vs mismatched)", "go_llm:v3:constrained:true", "go_llm:v3:constrained:mismatched"),
    ("C5a  statistics help the LLM (cell types)", "go_llm:v3:constrained:true", "go_llm:v1:constrained:true"),
    ("C5b  statistics help the LLM (rows)", "go_llm:v4:constrained:true", "go_llm:v2:constrained:true"),
]
GATE_RUNGS = {  # rung -> its mismatched-evidence control
    "go_enrichment:v3:contrast": "go_enrichment:v3:mismatched",
    "go_enrichment:v4:contrast": "go_enrichment:v4:mismatched",
    "go_llm:v3:constrained:true": "go_llm:v3:constrained:mismatched",
    "go_llm:v4:constrained:true": "go_llm:v4:constrained:mismatched",
}
CONTRASTS_FF = [   # PRIMARY family (free-form, raw F1@3)
    ("F1a  LLM adds to evidence + transfer (cell types)", "go_llm:v3:freeform:true", "go_enrichment:v5:contrast"),
    ("F1b  LLM adds to evidence + transfer (rows)", "go_llm:v4:freeform:true", "go_enrichment:v6:contrast"),
    ("F2a  LLM vs constant prior (cell types)", "go_llm:v3:freeform:true", "constant_prior_ff"),
    ("F2b  LLM vs constant prior (rows)", "go_llm:v4:freeform:true", "constant_prior_ff"),
    ("F3a  transfer baseline vs constant prior (cell types)", "go_enrichment:v5:contrast", "constant_prior_ff"),
    ("F3b  transfer baseline vs constant prior (rows)", "go_enrichment:v6:contrast", "constant_prior_ff"),
    ("F4   evidence is used (v3 true vs mismatched)", "go_llm:v3:freeform:true", "go_llm:v3:freeform:mismatched"),
    ("F5a  statistics help the LLM (cell types)", "go_llm:v3:freeform:true", "go_llm:v1:freeform:true"),
    ("F5b  statistics help the LLM (rows)", "go_llm:v4:freeform:true", "go_llm:v2:freeform:true"),
]
GATE_RUNGS_FF = {
    "go_enrichment:v5:contrast": "go_enrichment:v5:mismatched",
    "go_enrichment:v6:contrast": "go_enrichment:v6:mismatched",
    "go_llm:v3:freeform:true": "go_llm:v3:freeform:mismatched",
    "go_llm:v4:freeform:true": "go_llm:v4:freeform:mismatched",
}


# ---- statistics (pure functions; tested without MLflow) ----------------------------------
def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Paired bootstrap over genes of mean(a - b): estimate, 95% CI, two-sided p."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"n": len(d), "mean_diff": float(d.mean()), "ci_lo": float(np.quantile(boots, 0.025)),
            "ci_hi": float(np.quantile(boots, 0.975)), "p": float(min(1.0, max(p, 1.0 / (n_boot + 1))))}


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    m, adj, running = len(ps), [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


def evaluate_gate(table: dict[str, dict[str, float]], baselines_of: dict[str, str] = GATE_RUNGS,
                  prior_key: str = "constant_prior") -> dict:
    """
    table[condition] = {gene: headroom@3}. A rung passes if it beats the constant prior AND its own
    mismatched control with a CI excluding 0. Returns per-rung detail and an overall verdict.
    """
    detail = {}
    for rung, control in baselines_of.items():
        if rung not in table:
            continue
        r = {}
        for name, ref in (("vs_prior", prior_key), ("vs_mismatched", control)):
            if ref not in table:
                r[name] = None
                continue
            genes = sorted(set(table[rung]) & set(table[ref]))
            r[name] = paired_bootstrap([table[rung][g] for g in genes], [table[ref][g] for g in genes]) if len(genes) > 2 else None
        r["passes"] = bool(r["vs_prior"] and r["vs_mismatched"] and r["vs_prior"]["ci_lo"] > 0 and r["vs_mismatched"]["ci_lo"] > 0)
        detail[rung] = r
    passing = [k for k, v in detail.items() if v["passes"]]
    return {"verdict": "SIGNAL DETECTED" if passing else "INCONCLUSIVE", "passing_rungs": passing, "detail": detail}


# ---- data access -----------------------------------------------------------------------------
def load_runs(split: str) -> "list[dict]":
    mlflow.set_tracking_uri(_TRACKING_URI)
    rows = []
    for name in ("CellCoLLM/go_enrichment", "CellCoLLM/go_llm"):
        exp = mlflow.get_experiment_by_name(name)
        if exp is None:
            continue
        df = mlflow.search_runs([exp.experiment_id], max_results=100000, order_by=["start_time ASC"],
                                filter_string=f"tags.status = 'COMPLETED' and tags.gene_split = '{split}'")
        for _, r in df.iterrows():
            rows.append({"condition": r["tags.condition"], "gene": r["tags.gene_id"], "stratum": r.get("tags.stratum"),
                         "model": r.get("params.model"), "commit": r.get("tags.git_commit"),
                         "code_hash": r.get("tags.code_hash"), "unfrozen": str(r.get("tags.unfrozen_override")) == "True",
                         **{k[len("metrics."):]: v for k, v in r.items() if k.startswith("metrics.")}})
    return rows  # later runs overwrite earlier ones for the same (condition, gene) downstream


def no_llm_ranking(sh: GOShared, shown_gene: str, statistics: str, unit: str) -> list[str]:
    """Ranked GO ids for the no-LLM rung given `shown_gene`'s evidence (used for mismatched controls)."""
    b = sh.builder
    if statistics == "corrected":
        ev = b.evidence(shown_gene, unit)
        return [r["go_id"] for r in rank_corrected(corrected_stats(ev, sh.baselines[unit], 3, b.candidates), "p")]
    bg, i2g = b.called_universe(shown_gene, unit)
    query = A1.to_items(sh.ds.positive_cell_types(shown_gene), unit)
    res, _ = enrich(query, bg, i2g, {}, min_term_size=3, n_simulations=20)  # ranking uses raw p; g:SCS is irrelevant here
    return [r["go_id"] for r in rank_gprofiler(res, b.candidates, b.labels, "p")]


def broad_vocab(sh: GOShared, min_genes: int = 100) -> list[str]:
    go = sh.truth.go
    return sorted(t for t, n in go._counts.items()
                  if n >= min_genes and t != BP_ROOT and go.is_bp(t) and go.ic(t) > 0)


def read_split_genes(path: Path, split: str) -> list[str]:
    with open(path, newline="") as f:
        return [r["gene"] for r in csv.DictReader(f, delimiter="\t") if r["split"] == split]


def transfer_ranking(sh: GOShared, target: str, shown_gene: str, unit: str) -> list[str]:
    """go_enrichment v5/v6 ranking for `target`, built from `shown_gene`'s evidence (mismatched control)."""
    b = sh.builder
    cs = corrected_stats(b.evidence(shown_gene, unit), sh.baselines[unit], 3, b.candidates)
    w = {r.go_id: float(-np.log10(max(r.p_value, 1e-300))) for r in cs if r.tested and r.excess > 0}
    return [r["go_id"] for r in sh.transfer().rank(target, w, top_n=100)]


def render_contrasts(table: dict, contrasts: list, seed: int, unit_label: str) -> tuple[list[str], list]:
    lines = ["| contrast | n | mean diff | 95% CI | p (raw) | p (Holm) |", "|---|---|---|---|---|---|"]
    results = []
    for name, a, b in contrasts:
        if a not in table or b not in table:
            results.append((name, None))
            continue
        genes = sorted(set(table[a]) & set(table[b]))
        results.append((name, paired_bootstrap([table[a][g] for g in genes], [table[b][g] for g in genes], seed=seed)
                        if len(genes) > 2 else None))
    adj = iter(holm([r["p"] for _, r in results if r]))
    for name, r in results:
        if r is None:
            lines.append(f"| {name} | – | not available | | | |")
        else:
            lines.append(f"| {name} | {r['n']} | {r['mean_diff']:+.3f} | [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] | {r['p']:.4f} | {next(adj):.4f} |")
    return lines, results


def render_gate(gate: dict) -> list[str]:
    lines = [f"**{gate['verdict']}**" + (f" — passing rungs: {', '.join(f'`{r}`' for r in gate['passing_rungs'])}" if gate["passing_rungs"] else
             ". No rung beats both the constant prior and its mismatched-evidence control with a CI excluding 0. "
             "This is INCONCLUSIVE, not evidence that the LLM adds nothing: the comparison cannot discriminate."), "",
             "| rung | vs constant prior | vs mismatched control | passes |", "|---|---|---|---|"]
    for rung, d in gate["detail"].items():
        cells = [f"{x['mean_diff']:+.3f} [{x['ci_lo']:+.3f}, {x['ci_hi']:+.3f}]" if x else "n/a" for x in (d["vs_prior"], d["vs_mismatched"])]
        lines.append(f"| `{rung}` | {cells[0]} | {cells[1]} | {'yes' if d['passes'] else 'no'} |")
    return lines


def means_table(table: dict, keep, seed: int) -> list[str]:
    lines = ["| condition | n genes | mean | 95% CI |", "|---|---|---|---|"]
    for c in sorted(table, key=lambda c: (c.startswith("go_llm"), c)):
        vals = np.array(list(table[c].values()))
        if len(vals) < 2 or not keep(c):
            continue
        bs = paired_bootstrap(vals, np.zeros(len(vals)), seed=seed)
        lines.append(f"| `{c}` | {len(vals)} | {fmt(vals.mean())} | [{fmt(bs['ci_lo'])}, {fmt(bs['ci_hi'])}] |")
    return lines


# ---- report --------------------------------------------------------------------------------------
def fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-split", default="test", choices=["dev", "test", "smoke"])
    ap.add_argument("--prior-split", default="dev", choices=["dev", "test", "smoke"],
                    help="Split the constant prior is FIT on. Must not be the eval split for a real analysis")
    ap.add_argument("--stratum", default="carrying", help="'carrying' (primary), 'not_carrying', or 'all'")
    ap.add_argument("--genes-file", default=str(GENES_FILE))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-mixed", action="store_true",
                    help="Use runs from ANY code version (plumbing checks only — a real analysis must not mix them)")
    args = ap.parse_args()

    if args.eval_split == args.prior_split:
        print("WARNING: the constant prior is fit on the evaluation split — this leaks and is only valid for plumbing checks.\n")

    print("Loading shared state...")
    sh = GOShared.load()
    runs = load_runs(args.eval_split)

    # --- run identity: use ONE code version. Test split -> the FROZEN one; otherwise the current one. ---
    import json as _json
    if args.eval_split == "test" and FREEZE_PATH.exists():
        reference, ref_label = _json.loads(FREEZE_PATH.read_text())["fingerprints"], "the FROZEN pipeline"
    else:
        reference, ref_label = fingerprints(), "the current code"
    identity_notes = []
    if args.eval_split == "test":
        problems = verify_freeze(sh, args.genes_file)
        identity_notes.append("**Freeze check: OK** — code, data and gene list match the freeze manifest." if not problems else
                              "**Freeze check FAILED:** " + "; ".join(problems))
    if not args.allow_mixed:
        kept, dropped = [], {}
        for r in runs:
            approach = r["condition"].split(":")[0]
            why = None
            if r["code_hash"] != reference.get(approach):
                why = f"{approach}: produced by code {r['code_hash'] or 'with no hash'} (reference {reference.get(approach)})"
            elif r["unfrozen"] and args.eval_split == "test":
                why = f"{approach}: started without a matching freeze (unfrozen_override)"
            (dropped.__setitem__(why, dropped.get(why, 0) + 1) if why else kept.append(r))
        runs = kept
        identity_notes.append(f"Runs are restricted to {ref_label}: " + ", ".join(f"{a} `{h}`" for a, h in reference.items()) + ".")
        for why, n in sorted(dropped.items()):
            identity_notes.append(f"Excluded {n} runs — {why}.")
        if not runs:
            raise SystemExit(f"No {args.eval_split}-split runs were produced by {ref_label}. " + " ".join(identity_notes)
                             + " (--allow-mixed uses runs from any code version, for plumbing checks only.)")
    else:
        identity_notes.append("**--allow-mixed: runs from any code version are included. Not a valid analysis.**")
    latest: dict[tuple[str, str], dict] = {}
    for r in runs:
        latest[(r["condition"], r["gene"])] = r
    if args.stratum != "all":
        latest = {k: v for k, v in latest.items() if v["stratum"] == args.stratum}
    genes_all = sorted({g for _, g in latest})
    print(f"{len(latest)} completed runs, {len(genes_all)} genes ({args.eval_split} split, stratum={args.stratum})")

    by_cond: dict[str, dict[str, dict]] = {}
    for (c, g), r in latest.items():
        by_cond.setdefault(c, {})[g] = r
    table = {c: {g: r[PRIMARY] for g, r in d.items() if PRIMARY in r and np.isfinite(r[PRIMARY])} for c, d in by_cond.items()}

    # --- deterministic controls ---
    prior_genes = read_split_genes(Path(args.genes_file), args.prior_split) if Path(args.genes_file).exists() else []
    prior_genes = [g for g in prior_genes if sh.truth.has(g)]
    if not prior_genes:
        raise SystemExit(f"No {args.prior_split} genes found in {args.genes_file} to fit the constant prior on.")
    go = sh.truth.go
    prior = fit_constant_prior(sh.truth, prior_genes, sh.builder.candidates, 3)
    vocab = broad_vocab(sh)
    prior_ff = fit_constant_prior(sh.truth, prior_genes, vocab, 3)
    print(f"constant prior, 52-term vocabulary (fit on {len(prior_genes)} {args.prior_split} genes): {[go.label(p) for p in prior]}")
    print(f"constant prior, broad vocabulary ({len(vocab)} terms):                                {[go.label(p) for p in prior_ff]}")
    evalg = [g for g in genes_all if sh.truth.has(g)]

    table["constant_prior"] = {}
    for g in evalg:
        s_ = score_ranking(sh.truth, prior, g, sh.model)
        if "headroom_at_3" in s_:
            table["constant_prior"][g] = s_["headroom_at_3"]
    for v in ("v1", "v2", "v3", "v4"):
        tab = {}
        for g in evalg:
            donor = sh.choose_donor(g, UNIT_OF[v], 42, 0.05)
            s_ = score_ranking(sh.truth, no_llm_ranking(sh, donor, STAT_OF[v], UNIT_OF[v]), g, sh.model)
            if "headroom_at_3" in s_:
                tab[g] = s_["headroom_at_3"]
        table[f"go_enrichment:{v}:mismatched"] = tab

    # free-form family: raw F1@3 for every condition that can leave the vocabulary or is compared with one
    ff_keys = [c for c in by_cond if "freeform" in c or c.split(":")[:2] in (["go_enrichment", "v5"], ["go_enrichment", "v6"])]
    table_ff = {c: {g: r[PRIMARY_FF] for g, r in by_cond[c].items() if PRIMARY_FF in r and np.isfinite(r[PRIMARY_FF])} for c in ff_keys}
    table_ff["constant_prior_ff"] = {g: sh.truth.f1_at_k(prior_ff, g, 3) for g in evalg}
    for v in ("v5", "v6"):
        table_ff[f"go_enrichment:{v}:mismatched"] = {
            g: sh.truth.f1_at_k(transfer_ranking(sh, g, sh.choose_donor(g, UNIT_OF[v], 42, 0.05), UNIT_OF[v]), g, 3) for g in evalg}
    rng = np.random.default_rng(args.seed)
    rnd = float(np.mean([sh.truth.f1_at_k(list(rng.choice(vocab, 3, replace=False)), g, 3) for g in evalg for _ in range(20)]))

    # --- report ---
    lines = [f"# GO-prediction experiment — {args.eval_split} split, stratum `{args.stratum}`",
             f"_generated {datetime.now().isoformat(timespec='seconds')}; tracking store `{_TRACKING_URI}`_", ""]
    models = sorted({r["model"] for r in latest.values() if r.get("model") and r["model"] != "none"})
    lines += [f"Generator model(s): {', '.join(models) or 'n/a'}.", "", "## Run identity", ""] + [f"- {n}" for n in identity_notes] + [""]

    # PRIMARY: free-form
    lines += ["# PRIMARY — free-form (the LLM interprets the evidence and names the gene's own GO terms)", "",
              f"Endpoint: raw `{PRIMARY_FF}`. Constant prior (broad vocabulary, fit on {len(prior_genes)} `{args.prior_split}` genes): "
              + ", ".join(f"*{go.label(p)}*" for p in prior_ff) + f". Random 3 broad terms: F1@3 = {rnd:.3f}.",
              "The LLM may name terms outside the 52-term evidence vocabulary; the no-LLM rungs go_enrichment v5/v6 can too "
              "(co-annotation transfer), which is what keeps the comparison from being a vocabulary-size contest.", "",
              "## Mean F1@3", ""]
    lines += means_table(table_ff, lambda c: True, args.seed)
    lines += ["", "## Pre-registered contrasts — free-form (paired bootstrap over genes, Holm-corrected within this family)", ""]
    ff_lines, ff_results = render_contrasts(table_ff, CONTRASTS_FF, args.seed, "F1@3")
    lines += ff_lines
    gate_ff = evaluate_gate(table_ff, GATE_RUNGS_FF, prior_key="constant_prior_ff")
    lines += ["", "## Signal-existence gate — free-form", ""] + render_gate(gate_ff)
    lines += ["", "## Does the model interpret, or read the table back? (`frac_in_table`, `n_outside_table`)", "",
              "| condition | n | frac_in_table | outside table (mean) | invalid predictions (mean) |", "|---|---|---|---|---|"]
    for c in sorted(ff_keys):
        d = by_cond[c]
        lines.append(f"| `{c}` | {len(d)} | {fmt(np.nanmean([r.get('frac_in_table', np.nan) for r in d.values()]), 2)} | "
                     f"{fmt(np.nanmean([r.get('n_outside_table', np.nan) for r in d.values()]), 1)} | "
                     f"{fmt(np.nanmean([r.get('n_invalid', np.nan) for r in d.values()]), 1)} |")

    # CONTROL: constrained
    lines += ["", "# CONTROL — constrained (the LLM can only re-rank the 52 evidence terms)", "",
              f"Endpoint: `{PRIMARY}` (0 = a random 3-term prediction, 1 = best possible from the candidate vocabulary). "
              f"Constant prior (52-term vocabulary): " + ", ".join(f"*{go.label(p)}*" for p in prior) + ".", "",
              "## Mean headroom@3", ""]
    lines += means_table(table, lambda c: "freeform" not in c, args.seed)
    lines += ["", "## Pre-registered contrasts — constrained (Holm-corrected within this family)", ""]
    c_lines, c_results = render_contrasts(table, CONTRASTS, args.seed, "headroom@3")
    lines += c_lines
    gate = evaluate_gate(table)
    lines += ["", "## Signal-existence gate — constrained", ""] + render_gate(gate)
    lines += ["", "## Output validity (constrained mode)", "", "| condition | mean invalid rate | mean valid predictions |", "|---|---|---|"]
    for c in sorted(by_cond):
        if c.startswith("go_llm") and "constrained" in c:
            d = by_cond[c]
            lines.append(f"| `{c}` | {fmt(np.mean([r.get('invalid_rate', np.nan) for r in d.values()]))} | {fmt(np.mean([r.get('n_valid', np.nan) for r in d.values()]), 1)} |")

    text = "\n".join(lines) + "\n"
    print("\n" + text)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (out / f"analysis_{args.eval_split}_{stamp}.md").write_text(text)
    (out / f"analysis_{args.eval_split}_{stamp}.json").write_text(json.dumps(
        {"gate_freeform": gate_ff, "gate_constrained": gate,
         "contrasts_freeform": [{"name": n, **(r or {})} for n, r in ff_results],
         "contrasts_constrained": [{"name": n, **(r or {})} for n, r in c_results],
         "table_freeform": table_ff, "table_constrained": table}, indent=2, default=float))
    print(f"Wrote {out}/analysis_{args.eval_split}_{stamp}.md/.json")


if __name__ == "__main__":
    main()
