#!/usr/bin/env python3
"""
How fast is an LLM on this server RIGHT NOW, and how long would the whole experiment take with it?

For each model it does two things, with the same client settings the pipeline uses (JSON mode, num_ctx,
reasoning handling from core/llm_backend.py):

  1. PING   a tiny prompt. If this is slow, the server is queueing or overloaded, regardless of the model.
  2. REAL   the exact arm-2 prompt for a few genes (smoke genes first, then dev genes). Reports per call:
              time to first token   (waiting in the server queue + reading the prompt)
              generation time       (producing the answer)
              tokens/s, answer length, whether it parsed as JSON, how many predicted names resolve to a GO term

It prints an estimate per run (median and 90th percentile call) and for the complete dataset. Calls are
sequential (one at a time), one call per gene and condition -- that is how scripts/run_go_experiment.py runs.
Retries on malformed JSON, and server slow-downs later, would add to the estimate. It does not use MLflow.

Examples:
    python scripts/benchmark_llm_model.py                          # the model set as CHAT_MODEL in .env
    python scripts/benchmark_llm_model.py --model gpt-oss:120b
    python scripts/benchmark_llm_model.py --model gpt-oss:120b --model qwen3.5:122b --n-genes 3 --timeout 400
    python scripts/benchmark_llm_model.py --model gpt-oss:20b --ping-only          # 30 seconds, no data loading
    python scripts/benchmark_llm_model.py --model gpt-oss:120b --n-conditions 8    # the estimate for 8 conditions

Stops cleanly on Ctrl-C and still prints what it has. A call that fails (timeout, dropped connection) is counted
and listed, not hidden.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as stats
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "approaches" / "go_llm"))

from core.llm_backend import compute_num_ctx, make_chat_llm, resolve_chat_model
from core.structured_llm import parse_structured_response

OUT_DIR = ROOT / "data" / "go_experiment" / "model_benchmarks"
PING_PROMPT = 'Reply with only this JSON object: {"ok": true}'
# gene counts to extrapolate to (counted from data/go_experiment/genes.tsv)
# eligible  = passes all four selection rules of scripts/select_go_genes.py (the pool dev/test are drawn from)
# scoreable = has >=1 non-IEA biological-process annotation, so a prediction can be scored at all
SCOPES = {"test split (180 genes)": 180, "dev + test (220 genes)": 220,
          "all eligible genes (2,710)": 2710, "all scoreable genes (15,944)": 15944,
          "every gene in the dataset (18,908)": 18908}


def timed_stream(llm, prompt: str) -> dict:
    """One streamed call. Returns timings and the raw answer, or the error."""
    t0, first, first_text, text, reasoning, meta = time.time(), None, None, [], 0, {}
    try:
        for chunk in llm.stream([("user", prompt)]):
            now = time.time() - t0
            first = now if first is None else first
            piece = chunk.content or ""
            if piece and first_text is None:
                first_text = now
            text.append(piece)
            reasoning += len((chunk.additional_kwargs or {}).get("reasoning_content") or "")
            meta.update({k: v for k, v in (chunk.response_metadata or {}).items() if v is not None})
    except Exception as exc:  # timeout, dropped connection, gateway error
        return {"ok": False, "seconds": time.time() - t0, "error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    total = time.time() - t0
    gen_tok, gen_s = meta.get("eval_count"), (meta.get("eval_duration") or 0) / 1e9
    return {"ok": True, "seconds": total, "first_token_s": first, "first_answer_s": first_text,
            "generation_s": max(0.0, total - (first or 0)), "gen_tokens": gen_tok,
            "tokens_per_s": (gen_tok / gen_s) if gen_tok and gen_s else None,
            "prompt_tokens": meta.get("prompt_eval_count"), "load_s": (meta.get("load_duration") or 0) / 1e9,
            "answer_chars": len("".join(text)), "reasoning_chars": reasoning, "raw": "".join(text)}


def real_prompts(n: int, condition: str):
    """(symbol, prompt, shared, LLM module) for n genes, built exactly as run_go_llm builds them."""
    import run_go_llm as LLM
    from core.go_experiment import GOShared
    from scripts.run_go_experiment import parse_condition
    c = parse_condition(condition)
    shared = GOShared.load(need_baselines=True)
    cfg = LLM.PROMPT_VERSIONS[c["version"]]
    rows = list(csv.DictReader(open(ROOT / "data" / "go_experiment" / "genes.tsv"), delimiter="\t"))
    genes = [r for r in rows if r["split"] == "smoke"] + [r for r in rows if r["split"] == "dev"]
    out = []
    for r in genes[:n]:
        ev = shared.builder.evidence(r["gene"], cfg["unit"])
        res = (LLM.corrected_stats(ev, shared.baselines[cfg["unit"]], 3, shared.builder.candidates)
               if cfg["statistics"] == "corrected" else None)
        out.append((r["symbol"], LLM.build_prompt(shared, cfg, c["mode"], c["evidence"], ev, res, False)))
    return out, shared, LLM, c


def pct(values: list[float], q: float) -> float:
    return sorted(values)[min(len(values) - 1, int(q * len(values)))]


def hours(n_genes: int, seconds: float, n_conditions: int) -> str:
    h = n_genes * seconds * n_conditions / 3600
    return f"{h / 24:6.1f} days" if h >= 48 else f"{h:6.1f} h   "


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", default=None,
                    help="Repeat the option to compare several models. Default: CHAT_MODEL from .env")
    ap.add_argument("--condition", default="go_llm:v3:freeform:true")
    ap.add_argument("--n-genes", type=int, default=3, help="Real prompts to time per model (default 3)")
    ap.add_argument("--timeout", type=float, default=300, help="Seconds without a response before a call is abandoned")
    ap.add_argument("--ping-only", action="store_true", help="Only the tiny-prompt check (no data loading)")
    ap.add_argument("--n-conditions", type=int, default=1, help="Multiply the dataset estimate by this many conditions")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.model = [resolve_chat_model(m) for m in (args.model or [None])]  # no --model -> CHAT_MODEL from .env
    print("models:", ", ".join(args.model), flush=True)

    prompts = shared = LLM = c = None
    if not args.ping_only:
        print("loading dataset and building the real prompts ...", flush=True)
        prompts, shared, LLM, c = real_prompts(args.n_genes, args.condition)

    report = []
    try:
        for model in args.model:
            print(f"\n=== {model} ===", flush=True)
            llm = make_chat_llm(model=model, temperature=args.temperature, seed=args.seed, format="json",
                                num_ctx=compute_num_ctx(model), client_kwargs={"timeout": args.timeout})
            ping = timed_stream(llm, PING_PROMPT)
            print("ping  : " + (f"{ping['seconds']:.1f}s  (first token after {ping['first_token_s']:.1f}s)"
                                if ping["ok"] else f"FAILED after {ping['seconds']:.0f}s  {ping['error']}"), flush=True)
            calls = []
            for symbol, prompt in (prompts or []):
                r = timed_stream(llm, prompt)
                r["gene"] = symbol
                if r["ok"]:
                    try:
                        rows, diag = LLM.resolve_predictions(shared, parse_structured_response(r.pop("raw"), LLM.REQUIRED_RESPONSE_KEYS), c["mode"], 10)
                        r.update(parsed=True, n_predictions=diag["n_predictions_raw"], n_valid=diag["n_valid"])
                    except Exception as exc:
                        r.update(parsed=False, parse_error=f"{type(exc).__name__}: {str(exc)[:100]}")
                    tps = f"{r['tokens_per_s']:.0f} tok/s" if r["tokens_per_s"] else "? tok/s"
                    print(f"real  : {symbol}: {r['seconds']:.1f}s = {r['first_token_s']:.1f}s to first token + {r['generation_s']:.1f}s generating "
                          f"({r['gen_tokens']} tok, {tps}, {r['answer_chars']} chars answer, {r['reasoning_chars']} chars reasoning)  "
                          + (f"valid names {r['n_valid']}/{r['n_predictions']}" if r.get("parsed") else f"NOT PARSED: {r['parse_error']}"), flush=True)
                else:
                    print(f"real  : {symbol}: FAILED after {r['seconds']:.0f}s  {r['error']}", flush=True)
                calls.append(r)
            report.append({"model": model, "ping": ping, "calls": calls})
    except KeyboardInterrupt:
        print("\ninterrupted -- summarising what finished", flush=True)

    # ---- summary -------------------------------------------------------------------------------------------
    print("\n" + "=" * 78)
    for entry in report:
        ok = [r for r in entry["calls"] if r["ok"] and r.get("parsed")]
        # the first call of a model may include loading it into memory; keep it out of the estimate if others exist
        warm = [r for r in ok if r["load_s"] < 1.0] or ok
        print(f"{entry['model']}")
        ping = entry["ping"]
        print("  tiny-prompt ping : " + (f"{ping['seconds']:.1f}s" if ping["ok"] else "FAILED (server did not answer in time)"))
        print(f"  real calls       : {len(ok)} usable of {len(entry['calls'])}"
              + (f"  (failed: {sum(1 for r in entry['calls'] if not r['ok'])}, unparsable: {sum(1 for r in entry['calls'] if r['ok'] and not r.get('parsed'))})" if len(ok) != len(entry["calls"]) else ""))
        if not warm:
            print("  no usable real call, so no estimate for this model" if entry["calls"] else "  (ping only, no estimate)")
            continue
        sec = [r["seconds"] for r in warm]
        med, p90 = stats.median(sec), pct(sec, 0.9)
        wait = stats.median([r["first_token_s"] for r in warm])
        print(f"  per run          : median {med:.1f}s, 90th percentile {p90:.1f}s   "
              f"(of the median: ~{wait:.1f}s waiting for the first token, ~{med - wait:.1f}s generating)")
        print(f"  estimate, complete dataset ({args.n_conditions} condition{'s' if args.n_conditions != 1 else ''}, sequential):")
        for name, n in SCOPES.items():
            print(f"      {name:30s} {hours(n, med, args.n_conditions)}  (pessimistic: {hours(n, p90, args.n_conditions).strip()})")
    print("=" * 78)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"latency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                               "condition": args.condition, "timeout": args.timeout, "results": report}, indent=2, default=str))
    print("saved", out)


if __name__ == "__main__":
    main()
