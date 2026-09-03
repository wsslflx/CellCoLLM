# Approaches

This directory holds the incremental, single-variable-at-a-time experiments that test
whether an LLM can infer a gene's function from the pattern of cell types it is (and
isn't) expressed in — see `PIPELINE_REQUIREMENTS.md` for the full pipeline design these
approaches are early, deliberately narrow slices of.

Each approach isolates one question and answers it before moving to the next:

| Approach | Version | Question it isolates |
|---|---|---|
| `naive` | v1 | Can the model say anything from bare `CL:xxx\|UBERON:xxx` IDs alone, free-text? |
| `naive` | v2 | Same, but forced into structured JSON with a self-reported ID-recognition audit |
| `enriched` | v1 | Does removing the ID-recognition problem (real labels, not bare IDs) change the answer? |
| `enriched` | v2 | Does adding each term's official definition + ontology lineage change the answer further? |

All four share the same underlying task ("what do these cell types have in common"),
the same two directions per gene, and the same MLflow logging discipline — so results
are directly comparable across approaches/versions for a given gene.

## Shared design across every approach/version

- **Two directions per gene, one call each:**
  - **positive** — cell types where the gene IS reliably expressed → infer what property would explain expression.
  - **negative** — cell types where the gene is essentially NEVER expressed → infer what property would explain absence.
- **Blinding.** The gene's symbol/name is never sent to the LLM (only its Ensembl ID, used purely for bookkeeping); every assembled prompt is scanned by `verify_blinding()` before the call, and a run aborts (`status=FAILED_BLINDING`) if it ever finds the forbidden term.
- **MLflow.** Every run goes through `core/mlflow_utils.tracked_run()`, which logs a fixed core schema (gene, species, model, temperature, seed, dataset hash, prompt version, git commit, a `config_hash` of the identifying fields) into experiment `CellCoLLM/{approach}`. The positive and negative runs for one gene are linked under a shared "gene parent" run (`get_or_create_gene_parent_run`) so they're grouped in the UI; `enriched` and `naive` are separate experiments (not nested together) but both carry the same `gene_id` tag, so cross-approach comparison is a `tags.gene_id` search across experiments, not run-nesting.
- **Dataset.** `binarised_gene_expression_human.tsv` — for a given gene, `0` = essentially never expressed in that cell type/tissue, `1` = reliably expressed, blank = insufficient data. Rows are keyed by `CL:xxxxxxx|UBERON:xxxxxxx` pairs (a specific cell type observed in a specific tissue).
- **CLI shape.** Both `run_naive.py` and `run_enriched.py` share the same base flags: `--gene` (required, Ensembl ID), `--gene-symbol` (logging only, never sent to the LLM), `--species`, `--model` (falls back to `CHAT_MODEL` in `.env`), `--temperature`, `--seed`, `--dataset`, `--set {positive,negative,both}`, `--prompt-version`.

---

## `naive` (`approaches/naive/run_naive.py`)

The simplest possible version: hand the LLM the gene's cell-type set as bare
`CL:xxxxxxx|UBERON:xxxxxxx` identifiers — no labels, no lookups, nothing but the raw
IDs — and ask what they have in common. This tests whether the model can reason from
ontology IDs it has to recognize/recall from its own training, which is an open
question this approach exists specifically to answer (and, per findings so far,
frequently answers *no* to — see v2 below).

### v1 — free text, no schema

```bash
python approaches/naive/run_naive.py --gene ENSG00000001626 --gene-symbol CFTR --prompt-version v1
```

- Prompt (`prompts/naive_id_only_{positive,negative}_v1.txt`): a short instruction plus
  the raw ID list, asking for the most specific shared biological property, with an
  explicit permission to say "I don't recognize enough of these" instead of guessing.
- The LLM's response is logged verbatim as free text (`response.txt`) — no parsing, no
  retry, no forced structure. `latency_s` is the only metric beyond the shared schema.
- Purpose: a first, completely unconstrained read on whether the model volunteers
  honest uncertainty or confidently guesses — before any structure is imposed.

### v2 — forced JSON + ID-recognition audit

```bash
python approaches/naive/run_naive.py --gene ENSG00000001626 --gene-symbol CFTR --prompt-version v2
```

- Prompt (`prompts/naive_id_only_{positive,negative}_v2.txt`) adds:
  1. An explicit **audit step**: pick up to 10 entries where the model is confident it
     knows *both* the CL and UBERON meaning, reporting each half's recognition and
     confidence independently (`cl_id`/`cl_meaning`/`cl_certain`,
     `uberon_id`/`uberon_meaning`/`uberon_certain`) — so a model that knows a cell type
     but not its tissue (or vice versa) doesn't get to claim the whole pair.
  2. A forced JSON schema: `n_ids_recognized`, `recognized_examples`, `property`,
     `confidence`, `rationale`, `abstained` — every response is `json.loads`-validated
     against `REQUIRED_RESPONSE_KEYS`, with up to `MAX_PARSE_ATTEMPTS=3` retries on
     malformed/incomplete JSON (`llm.invoke` called with `format="json"`).
  3. Explicit "bad vs. good property" examples and an abstention framing ("abstaining is
     correct when the evidence doesn't support one — it is not a failure").
- Logged: `n_ids_recognized`, `confidence`, `parse_retries`, `latency_s` as metrics;
  `recognized_examples.json`, `response_parsed.json`, `response_raw.txt`,
  `prompt.txt` as artifacts; `abstained` as a tag.
- **What this found**: the audit reliably exposes hallucinated ID recall — e.g. a model
  confidently calling `CL:0000066` "fibroblast" or "erythrocyte" in different runs when
  it's actually "epithelial cell." This result is what directly motivated `enriched`.

---

## `enriched` (`approaches/enriched/run_enriched.py`)

Removes the ID-recall confound `naive` v2 exposed: instead of asking the LLM to recall
what an ID means, the harness resolves it deterministically via a **pinned local
ontology cache** (`core/ontology_lookup.py`, built by `scripts/build_ontology_cache.py`
from the official CL/Uberon OBO releases, scoped to only the ~700–900 IDs actually used
in the dataset — no network calls at run time, no full-ontology bloat). `enriched` lives
in its own MLflow experiment (`CellCoLLM/enriched`) and never touches `approaches/naive/`.

Both versions render the cell-type set as **one line per tissue, listing every cell
type observed there** (not one line per raw pair) — the dataset has a small, fixed
UBERON universe (~54 tissues) versus a much larger CL universe (~700+ cell types), so
grouping by tissue removes a lot of repeated "located in: X" text for genes with large
sets, without losing any information. This is logged as `list_group_by="uberon"` and
applies identically to positive and negative (uniform structure, no formatting
asymmetry between the two sets).

### v1 — resolved labels only

```bash
python approaches/enriched/run_enriched.py --gene ENSG00000132763 --model qwen3:32b --prompt-version v1
```

- Prompt (`prompts/enriched_labels_only_{positive,negative}_v1.txt`): the grouped
  tissue/cell-type list, using each term's real resolved label instead of its bare ID.
  An entry that couldn't be resolved (`NOT_FOUND`/`OBSOLETE` in the cache) renders as
  `(unresolved: CL:xxxxxxx)` rather than being silently dropped or guessed at.
- Same forced-JSON schema as naive v2, minus the ID-recognition audit (moot — labels
  are given, not recalled): `property`, `confidence`, `rationale`, `abstained`.
- Logged: `n_cell_types`, `n_unresolved`, `n_list_groups` metrics;
  `{positive,negative}_cell_types.json`, `unresolved_ids.json`, `prompt.txt`,
  `response_raw.txt`, `response_parsed.json` artifacts.
- Isolates: does the model reason better given *correct* cell-type/tissue names alone,
  with zero other context (no definitions, no hierarchy)?

### v2 — glossary (definitions + is_a hierarchy)

```bash
python approaches/enriched/run_enriched.py --gene ENSG00000132763 --model qwen3:32b --prompt-version v2
```

- Adds a **glossary section**: one block per *unique* CL/UBERON term referenced in the
  set (deduplicated — many rows share the same term), each with its OBO definition and
  its `is_a` ancestor chain, printed once and referenced by the compact list below it
  (rather than repeating full definitions on every one of a gene's — sometimes
  thousands of — rows, which would blow past the context window on its own).
- Three independently toggleable, MLflow-logged ablation axes (all default on/`2`):
  - `--include-definition` / `--no-include-definition` — include each term's OBO definition.
  - `--include-hierarchy` / `--no-include-hierarchy` — include the `is_a` ancestor chain.
  - `--hierarchy-depth N` (default `2`) — how many `is_a` hops to include; ancestors are
    collected via BFS over *all* parents (CL/Uberon permit multiple inheritance — no
    single path is picked), deduplicated by ID.
  - `--strip-marker-noise` / `--no-strip-marker-noise` (default on) — removes
    flow-cytometry marker/phenotype sentences (e.g. "Markers include F4/80-positive,
    CD68-positive..."), inline bracket cross-ref tags (`[ZFA]`), and "Examples: ..."
    trailers from definitions — technical noise not relevant to "what do these cell
    types share." Sentence-level removal triggered by marker-token density (≥2 per
    sentence), so an incidental single mention (e.g. "expresses the CD4 coreceptor")
    is correctly left alone. Logs `definition_chars_stripped`.
- Ancestor lookups are served entirely from the pre-built cache (`OntologyLookup.ancestors()`),
  which itself was extended with a `parents` field and a build-time hierarchy cap
  (`CACHE_HIERARCHY_DEPTH=4` in `scripts/build_ontology_cache.py`) — no live OBO parsing
  at prompt-build time.
- **Context-budget tracking** (this approach can produce very large prompts — genes with
  large expression sets and rich definitions have driven prompts well past naive context
  assumptions during development):
  - `num_ctx` is always sized to the target model's *real* max context length
    (`core/llm_backend.get_model_max_context()`, queried live from `/api/show` and
    cached per model — e.g. 40,960 for `qwen3:32b`), not a fixed guess.
  - `estimated_prompt_tokens` (a cheap `chars // 4` heuristic) is logged pre-call as an
    early heads-up (`context_overflow_risk_estimated` tag) — but calibration against the
    model's real tokenizer showed this estimate is unreliable in both directions, so it
    no longer drives sizing decisions.
  - The trustworthy signal is post-call: `actual_prompt_tokens` (the model's own reported
    `prompt_eval_count`), `context_utilization` (fraction of the available budget used),
    and `context_overflow_confirmed` (`True` above 98% utilization — Ollama silently
    truncates oversized prompts to fit, so a prompt sitting right at the edge of its
    budget is a reliable truncation signal even though the reported count itself never
    reads as "over the limit"). This check runs on both successful and failed
    (`FAILED_PARSE`) calls, since a truncated prompt is a plausible cause of malformed
    JSON output.
- Isolates: given correct labels *and* correct definitions/lineage, does the model's
  answer change or improve further — and, ablation-wise, which of those two axes (or
  the hierarchy depth) actually drives any change?

---

## Batch-running test genes

```bash
python scripts/run_test_genes.py --run naive:v1 --run enriched:v2
```

Runs every `approach:version` combo (repeatable `--run approach:version`) across the 3
standard test genes (hardcoded default; override with repeatable `--gene`), with fixed
defaults otherwise (`--set both`, model from `.env`). See
`scripts/run_test_genes.py` — it has no built-in knowledge of which versions exist, so
future versions (`v3`, ...) work automatically once registered in the approach's own
`PROMPT_VERSIONS`.
