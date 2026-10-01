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
| `naive` | v3 | Can the model *look up* IDs itself via a live ontology tool instead of recalling them? |
| `enriched` | v1 | Does removing the ID-recognition problem (real labels, not bare IDs) change the answer? |
| `enriched` | v2 | Does adding each term's official definition + ontology lineage change the answer further? |
| `statistical` | v1 | If a statistical test finds the contrast, can the LLM name the biology behind it? |
| `statistical` | v2 | Same, but testing against a realistic null and with *properties* (GO processes) as features, not just categories |
| `go_enrichment` | v1 / v2 | **No LLM.** Classical GO over-representation (g:Profiler's method) as the scientific baseline the LLM arms must beat |
| `go_enrichment` | v3 / v4 | **No LLM.** The same evidence under the baseline-corrected two-factor null — the no-LLM counterpart of `go_llm` v3/v4 |
| `go_enrichment` | v5 / v6 | **No LLM.** v3/v4 plus **co-annotation transfer**: a baseline that can name gene-level GO terms *outside* the evidence vocabulary, the no-LLM counterpart of free-form `go_llm` |
| `go_llm` | v1-v4 | Given the same cell-type GO evidence, can an **LLM interpret it** and name the gene's own GO terms — and what does it add over the statistics and the transfer baseline? Scored deterministically. See the experiment protocol |

All of them share the same underlying question, response schema, and MLflow logging
discipline, so results are directly comparable per gene. They differ in one structural
way worth knowing up front:

- **`naive` and `enriched` run two directions per gene** (positive and negative), each
  a separate LLM call that sees only one set.
- **`statistical` computes the contrast itself** and produces **one run per gene**,
  tagged `input_set=contrast`. There are no directions to run.
- **`go_enrichment` v1/v2** run one g:Profiler-style query per direction (only the
  positive one is scored); **v3/v4** compute the contrast internally, one `contrast` run
  per gene. **`go_llm`** also produces one `contrast` run per gene *per condition*
  (output mode × evidence). Both are described in the GO-prediction experiment protocol
  below, and neither is judged by the LLM-judge validation layers — they score themselves
  deterministically against the gene's own GO annotation.

## Shared design across every approach/version

- **Directions (`naive` and `enriched` only):** two per gene, one call each.
  - **positive** — cell types where the gene IS reliably expressed → infer what property explains expression.
  - **negative** — cell types where the gene is essentially NEVER expressed → **predict, by exclusion, what the expressing cells would look like**. Both directions therefore describe the *same* target (the expressing cells) by different routes, which makes them a consistency check rather than two independent measurements.
  - `statistical` has no directions — it sees both sets at once and emits one `contrast` run.
  - `go_enrichment` v1/v2 have both directions (positive scored, negative diagnostic); v3/v4 and `go_llm` are single `contrast` runs.
- **Blinding.** The gene's symbol/name is never sent to the LLM (only its Ensembl ID, used purely for bookkeeping); every assembled prompt is scanned by `verify_blinding()` before the call, and a run aborts (`status=FAILED_BLINDING`) if it ever finds the forbidden term.
- **MLflow.** Every run goes through `core/mlflow_utils.tracked_run()`, which logs a fixed core schema (gene, species, model, temperature, seed, dataset hash, prompt version, git commit, a `config_hash` of the identifying fields) into experiment `CellCoLLM/{approach}`. The positive and negative runs for one gene are linked under a shared "gene parent" run (`get_or_create_gene_parent_run`) so they're grouped in the UI; `enriched` and `naive` are separate experiments (not nested together) but both carry the same `gene_id` tag, so cross-approach comparison is a `tags.gene_id` search across experiments, not run-nesting. The GO-prediction experiment (`go_enrichment`, `go_llm`) adds three tags — `condition` (`approach:version[:mode:evidence]`, the join key for `scripts/analyze_go_experiment.py`), `gene_split` (`dev`/`test`/`smoke`) and `stratum` (`carrying`/`not_carrying`) — and logs its scores as `gopred_*` metrics **in the same run** as the prediction. `CELLCO_MLFLOW_URI` redirects any run to a scratch store; unset, runs go to `mlflow.db`.
- **Dataset.** `binarised_gene_expression_human.tsv` — for a given gene, `0` = essentially never expressed in that cell type/tissue, `1` = reliably expressed, blank = insufficient data. Rows are keyed by `CL:xxxxxxx|UBERON:xxxxxxx` pairs (a specific cell type observed in a specific tissue).
- **CLI shape.** All runners share the same base flags: `--gene` (required, Ensembl ID), `--gene-symbol` (logging only, never sent to the LLM), `--species`, `--model` (falls back to `CHAT_MODEL` in `.env`), `--temperature`, `--seed`, `--dataset`, `--prompt-version`. `run_naive.py` and `run_enriched.py` additionally take `--set {positive,negative,both}`; `run_statistical.py` does not (no directions). `run_go_enrichment.py` takes `--set` (used by v1/v2 only, ignored by v3/v4) and has no `--model`; `run_go_llm.py` takes `--model` but no `--set`, and adds `--output-mode` and `--evidence`.

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

## `statistical` (`approaches/statistical/run_statistical.py`)

### Why it exists

`naive` v1/v2/v3 and `enriched` v1/v2 all scored **at chance** on the discrimination
validation layer, and all converged on the same generic answer — "immune cells" — for
three unrelated genes (a B12-metabolism enzyme, a DNA-damage scaffold, a checkpoint
kinase). The cause is architectural, not a prompt problem:

- Each call sees **one set in isolation**, so it reports whatever is most salient in
  that set. This atlas is ~45% immune/haematopoietic on **both** sides, so "immune
  cells" is both the most salient feature and completely non-discriminating. The
  leakage numbers confirm it — e.g. CHEK1 `enriched:v2` scored coverage 0.46 against
  leakage 0.44.
- `PIPELINE_REQUIREMENTS.md` §5.2 predicted this exactly, and designates the split
  architecture those approaches use as the **control arm**, not the default.
- The specified fix — joint contrast, both sets in one prompt — measures **68–72k
  tokens**, over `qwen3:32b`'s 40,960 window for every gene.

The published answer to this shape of problem is a division of labour: a statistical
test finds *what* differs, the LLM names *why*.
[LangLasso](https://arxiv.org/html/2601.10458v1) tested precisely this and found
precomputed summary statistics beat both raw sampling ("missed discriminative
features") and full data ("token limitations prevented systematic evaluation").
[GPTCelltype](https://www.nature.com/articles/s41592-024-02235-4) reaches its best
accuracy from the *top ten* differential genes, not the full expression matrix.

### Method

```bash
python approaches/statistical/run_statistical.py --gene ENSG00000149554 --model qwen3:32b
```

1. **Roll up.** Every `CL|UBERON` pair is expanded into features: its own CL term, its
   `is_a` ancestors up to `--hierarchy-depth`, and its UBERON tissue
   (`core/enrichment.pair_features`, reusing `OntologyLookup.ancestors`).
2. **Test.** One Fisher's exact test per feature on positive-vs-negative counts.
   Features seen fewer than `--min-feature-count` times in total are skipped as
   underpowered.
3. **Correct.** Benjamini–Hochberg via `scipy.stats.false_discovery_control`.
4. **Render.** The top `--top-n-enriched` / `--top-n-depleted` significant features are
   shown with counts, odds ratio, q-value **and their full OBO definition**.

Measured on CHEK1: **1,403 tokens** versus 38,731 for `enriched:v2` — a ~28× reduction,
138 features tested, 50 significant at q<0.05.

### The division of labour, stated honestly

The statistics decide **which** features are worth showing; the definitions are what the
LLM actually reasons over. That distinction matters — handed only feature names and
counts, the model would be verbalising a table rather than inferring anything.

Where it does and doesn't add value:

- If the enriched features share a clean single ancestor ("these are all epithelial
  cells"), the LLM adds little — and per §L3 that answer is also the most
  annotation-circular, since CL terms were assigned partly *from* expression data.
- The LLM earns its place when the enriched features span unrelated ontology branches
  and the shared property is **not** an ontology term. Naming what ciliated cells, club
  cells and ionocytes have in common requires knowing what they do.

**What it cannot express:** only properties reducible to ontology features. "Cells under
high replicative demand" is not a CL term and will never surface, however true it is.

### Parameters

| Flag | Default | Trade-off |
|---|---|---|
| `--hierarchy-depth` | 3 | Deeper = more general ancestors, more shared features, less specificity |
| `--min-feature-count` | 20 | Lower = more features tested, more multiple-testing burden |
| `--q-threshold` | 0.05 | BH significance cutoff |
| `--top-n-enriched` / `--top-n-depleted` | 8 / 5 | How many features reach the prompt |
| `--include-tissue` / `--no-include-tissue` | on | **The batch-artifact ablation** — see below |
| `--include-definition` | on | Off reduces the LLM to verbalising the table |

### Batch-artifact warning — read this before interpreting any result

`core/enrichment.tissue_homogeneity()` flags tissues whose cell types are >95% or <5%
positive. Real expression varies within a tissue; all-or-nothing tissues are the
signature of study/batch structure — which dataset profiled that tissue, at what depth
(§L9 study bias, §L11 dropout/thresholding fragility).

Measured on the current dataset:

| Gene | all-or-nothing tissues | examples |
|---|---|---|
| TTI2 | **12 of 40 (30%)** | eye 100% positive, kidney 100%, musculature 100%, adipose 100% |
| CHEK1 | 6 of 40 (15%) | adrenal gland 100% positive, lamina propria 0% |

Logged as `n_allornothing_tissues` / `frac_allornothing_tissues`, with the tag
`batch_artifact_warning=True` and a terminal warning above 20%. **A warned run must not
be read as a biological result** — its enrichment is largely reporting which study
contributed which tissue. Running with `--no-include-tissue` removes UBERON features
entirely; on CHEK1 that drops significant features from 50 to 29, i.e. roughly 40% of the
apparent signal was tissue-level.

### What it logs

- **Metrics:** `n_features_tested`, `n_significant`, `n_enriched`, `n_depleted`,
  `n_positive_pairs`, `n_negative_pairs`, `n_allornothing_tissues`,
  `frac_allornothing_tissues`, `estimated_prompt_tokens`, `actual_prompt_tokens`,
  `latency_s`, `parse_retries`, `confidence`.
- **Artifacts:** `prompt.txt`, `response_raw.txt`, `response_parsed.json`,
  **`enrichment_full.json`** (every tested feature with counts/OR/p/q — the complete
  auditable evidence, not just what reached the prompt), `tissue_homogeneity.json`.

### Known result (CHEK1, v1)

> *"hormone-regulated immune and developmental processes in secretory and barrier tissues"*

Wrong for a checkpoint kinase — but a **faithful reading of the evidence supplied**
(endocrine gland OR 27.8 q=6e-18, innate lymphoid cell OR 10.0, adrenal gland, embryo).
Scored J = −0.222 on the discrimination layer, no better than `enriched:v2`'s +0.026
(and within that layer's ±0.164 sampling noise, so not a reliable difference).

The conclusion to draw: **the reasoning step worked; the evidence was the problem.** The
enrichment is real and highly significant, but on this dataset it is substantially
reporting batch structure. Two things it does get right that the LLM approaches did not —
it reports `leukocyte` as **depleted** for MMACHC (OR 0.46) and CHEK1 (OR 0.73), where
the LLM approaches answered "immune surveillance" and earned J = −0.257; and every number
is traceable to a count you can check without a model.

### v2 — baseline-corrected null + property features

```bash
python scripts/build_ontology_cache.py        # once: adds capable_of -> GO, UBERON part_of
python scripts/build_feature_baselines.py     # once: per-feature baselines over all 18,908 genes
python approaches/statistical/run_statistical.py --gene ENSG00000149554 --prompt-version v2
```

v2 fixes three problems in v1, each measured rather than assumed.

#### 1. v1's null hypothesis is wrong

Fisher's exact test assumes a feature is distributed between the positive and negative
sets in proportion to their sizes — i.e. that it has no intrinsic tendency to be called
"expressed". Measured over **all 18,908 genes**, that is false for essentially every
feature:

| feature kind | baseline positive-rate spread |
|---|---|
| lineage (cell types) | **28% … 93%** |
| process (GO) | 34% … 87% |
| tissue (UBERON) | 43% … 87% |

Grand mean across genes: **57.4%**. So mast cells and spinal-cord samples are called
"expressed" far more often than T cells and blood samples *for any gene* — plausibly
sequencing depth and RNA content. v1 reports "esophagus enriched" correctly, but that
is true for nearly every gene and is therefore not evidence about *this* gene.

**v2 tests against a two-factor null instead:**

```
expected = gene's overall rate  ×  feature's baseline rate  /  grand mean rate
```

then a binomial test of observed against expected, with `excess = observed − expected`
as the effect size. Both factors are required — correcting only for the feature
baseline is actively misleading:

| | v1 (raw) | 1-factor | **v2 (2-factor)** |
|---|---|---|---|
| MMACHC / esophagus | "strongly enriched" | +6% → looks like artifact | **+53%, real** |
| TTI2 / eye | "100% positive!" | +44% → looks dramatic | **+15%, modest** |

MMACHC expresses in only 20% of cell types, so 77% in esophagus is remarkable; TTI2
expresses in 80% everywhere, so 100% in eye is barely above expectation. The gene's own
propensity is what separates them.

Baselines live in `data/feature_baselines.json`, keyed on dataset hash + ontology
versions + **every feature-extraction parameter**. Changing `--go-depth` and friends
changes which features exist, so the runner **refuses to run** on a key mismatch rather
than silently mixing configurations.

#### 2. v1's output is categories, not properties

`is_a` ancestors are taxonomy by construction, which is why v1's answer read as a
restatement of its own input (enriched: *hematopoietic precursor cell, innate lymphoid
cell, NK cell* → answer: *"Hematopoietic and innate immune progenitor and effector
cells"*).

§4.1 specifies joining CL + Uberon + **GO**; v1 used only `is_a`. v2 adds
`capable_of` / `capable_of_part_of` links to GO biological processes — *phagocytosis*,
*cytokine production*, *stem cell division*. These are properties.

Coverage at `--go-depth 3`: **53% of cell types, 58% of rows, 126 distinct processes**
(direct annotation is only 13% — inheritance down the `is_a` hierarchy does the work).
All 126 GO names come from `cl.obo` itself, so no `go.obo` download is needed.

#### 3. v1 fragments the tissue signal

v1 tested tissues only at their exact level (54 features), so a respiratory gene's
signal splits across lung/nose/trachea and may miss significance on each. v2 rolls up
via **both `is_a` and `part_of`**, so `lung` now reaches `respiratory system` and
`lower respiratory tract`. Feature universe: **1,207 total — 860 lineage, 126 process,
221 tissue**.

#### Parameters added in v2

| Flag | Default | Trade-off |
|---|---|---|
| `--go-depth` | 3 | Deeper = more coverage (13% direct → 53% at depth 3 → 58% plateau) but generic processes spread across many cell types |
| `--uberon-depth` | 3 | Tissue rollup depth |
| `--include-process` / `--no-include-process` | on | Process features are the only source of *properties* — and the most annotation-circular (§L3) |
| `--min-excess` | 0.10 | Effect-size floor. With large n, trivially small excesses reach significance; q-value alone is not enough |

#### Novelty check — measuring whether the LLM adds anything

A statistics-first design risks reducing the LLM to verbalising a table. v2 measures
this directly and deterministically: `property_overlap_shown` is the fraction of the
property's content words that also appear in the feature labels it was shown, and
`property_restates_input` tags runs above 60%.

On CHEK1, v2 scored **11% overlap** ("goes beyond input"), against v1's near-verbatim
concatenation. That does not prove the inference is *correct* — only that it isn't a
readout.

#### Caveats, stated plainly

- **Annotation circularity (§L3)** is sharpest for process features: a cell type is
  annotated `capable_of` phagocytosis partly *because* of the genes it expresses.
  `n_process_features_shown` is logged so any result resting on them is identifiable.
  This is deliberately **not** mentioned in the prompt — that would bias the model.
- **Coverage bias (§L5)**: 38% of rows have no process annotation even with
  inheritance, and understudied cell types are disproportionately affected — exactly
  the genes the method is most needed for. Logged as
  `frac_rows_with_process_annotation`.
- **Bounded vocabulary**: features can only express what the ontologies name. A
  property like "cells under high replicative demand" is not a GO or CL term and will
  never surface as a *feature* — though the LLM is explicitly invited to propose one
  beyond the supplied labels.
- **Nested features are not independent** — both `is_a` and GO are hierarchies, so BH
  correction is applied to correlated tests. Unavoidable here; not silently ignored.
- **Baseline correction reduces but does not eliminate the batch concern.** The
  all-or-nothing tissue diagnostic from v1 is retained.

#### Known result (CHEK1, v2)

> *"Involvement in innate immune response and hematopoietic development in specific anatomical regions"*

295 features tested, 100 significant at q<0.05 with |excess| ≥ 0.10. Property overlap
11% — a genuine inference rather than a restatement, but still not "cell-cycle
checkpoint control", and it scored J = −0.214 on the discrimination layer (within that
layer's ±0.164 sampling noise of v1's −0.222, so not a reliable difference).

The honest reading remains what v1 established: the statistics and the reasoning step
both work; the evidence they rest on is compromised by dataset-level artifacts.

---

## `go_enrichment` (`approaches/go_enrichment/run_go_enrichment.py`) — no-LLM baseline

### Purpose

This is arm 1 of a two-arm experiment isolating **what an LLM actually contributes**.
Both arms see identical evidence — GO biological processes attached to cell types via
CL's `capable_of` relation — but this arm uses only established statistics and **no model
at any point**. It is the baseline any LLM approach has to justify itself against.

Method: Fisher's one-tailed test (cumulative hypergeometric) with **g:SCS** multiple
testing correction, plus Bonferroni and Benjamini–Hochberg, following g:Profiler's
g:GOSt (v1/v2). v3/v4 keep the no-LLM property but swap the statistic for the
baseline-corrected two-factor test used by `statistical` v2, restricted to GO terms — so
that `go_llm` v3/v4 has a counterpart on identical statistics.

**References**

- Kolberg L, Raudvere U, Kuzmin I, Adler P, Vilo J, Peterson H. *g:Profiler —
  interoperable web service for functional enrichment analysis and gene identifier
  mapping (2023 update).* Nucleic Acids Research 51(W1):W207–W212.
  https://doi.org/10.1093/nar/gkad347
- Raudvere U, Kolberg L, Kuzmin I, Arak T, Adler P, Peterson H, Vilo J. *g:Profiler: a
  web server for functional enrichment analysis and conversions of gene lists (2019
  update).* Nucleic Acids Research 47(W1):W191–W198.
  https://doi.org/10.1093/nar/gkz369
- Original publication introducing g:SCS (2007):
  https://pubmed.ncbi.nlm.nih.gov/17478515/
- Method documentation — the source for the g:SCS description and domain-scope options
  implemented here: https://biit.cs.ut.ee/gprofiler/page/docs

### ⚠ The background / query size problem — read this before interpreting any result

**a) Why over-representation assumes query ≪ background.** The test asks whether a term
appears more often in the query than a random draw from the background would give. The
expected intersection is `n × K / N` (query size × term size / background size). That
question only has room to be answered when the query is a small slice of the background.

**b) The structural ceiling.** The intersection `k` can never exceed the term size `K`,
so:

```
maximum achievable fold-enrichment  =  K / (n × K / N)  =  N / n  =  1 / (query/background ratio)
```

The query/background ratio therefore caps how enriched *anything* can look, before any
biology is involved.

**c) Measured ceilings for this dataset** (v1, background N = 359 annotated cell types at
`--go-depth 3`):

| gene | direction | query n | n/N | max possible fold |
|---|---|---|---|---|
| MMACHC | positive | 118 | 33% | **3.04×** |
| TTI2 | negative | 143 | 40% | 2.51× |
| CHEK1 | positive | 216 | 60% | 1.66× |
| CHEK1 | negative | 231 | 64% | 1.55× |
| TTI2 | positive | 285 | 79% | **1.26×** |
| MMACHC | negative | 295 | 82% | **1.22×** |

For TTI2-positive, **no term can exceed 1.26× enrichment** — not because the biology is
weak, but because the query is 79% of the background.

**d) The double bind: the ceiling collapses exactly where significance gets easy.** Large
queries also have high power, so trivial folds reach significance. Worked example with
`secretion by cell` (K = 84):

| query | expected | observed | fold | p-value |
|---|---|---|---|---|
| TTI2 positive (n=285) | 66.7 | 80 | **1.20×** | **6.3×10⁻⁶** |
| MMACHC positive (n=118) | 27.6 | 33 | **1.20×** | 0.098 |

The *same* 1.20× fold is highly significant for the large query and not significant for
the small one.

This is not hypothetical — it is what the runs produce. TTI2-positive's top result is
`T cell mediated immunity` at **1.22× fold**, which is 97% of its 1.26× ceiling, with
p_gscs = 9.4×10⁻³. Statistically significant, biologically saturated, and essentially
meaningless.

**e) How to read results from this arm.** Check `query_background_ratio` first. Above
~0.5, treat significance as uninformative and read `fold_enrichment` against
`max_fold_enrichment_possible` instead: a 1.22× result against a 1.26× ceiling is
saturated, not strong. Both numbers are logged per run, and `low_power_warning` is tagged
and printed above 0.5. MMACHC-positive (33%, 3.04× headroom) is the only case in the
current test set with genuine power.

**f) Why this is not fixable within a faithful implementation.** The ratio is a property
of the data — a gene expressed in most cell types has a positive set that *is* most of
the background. Alternatives, none of which are g:Profiler:

- Contrast positive against negative directly instead of against the background — this is
  what `statistical` v1/v2 do, and part of why they exist.
- Restrict to genes with narrow expression, so the positive set is a small slice.
- Use the negative set as the background. A deviation worth noting; not implemented.

This arm is kept deliberately faithful so it functions as the established-method
baseline. The limitation is documented rather than engineered around.

### Fidelity caveats

**1. g:Profiler annotates genes; this annotates cell types.** Their API takes gene lists
and cannot be called here, so the *method* is reimplemented against a cell-type→GO
annotation universe. Describe it as "g:Profiler's method", never as g:Profiler.

**2. g:SCS's documented ordering property does not hold in this regime — verified.**
g:Profiler documents g:SCS as *more conservative than BH, less strict than Bonferroni*.
Measured here, it is frequently **less** conservative than both:

| gene / direction | significant: g:SCS | Bonferroni | BH |
|---|---|---|---|
| MMACHC positive | 1 | 1 | 1 |
| CHEK1 positive | 3 | 2 | 3 |
| CHEK1 negative | 2 | 1 | 2 |
| **TTI2 positive** | **2** | **0** | **0** |
| TTI2 negative | 4 | 2 | 4 |

The cause is arithmetic, not an implementation error. g:SCS's correction factor is
`alpha / threshold`, where the threshold is the 5th percentile of the minimum p-value over
random queries. In g:Profiler's regime — thousands of GO terms, small queries — that
minimum is tiny and the factor is enormous. Here there are only **52 testable terms** and
the query can be 79% of the background, so random queries rarely achieve a small p; the
threshold comes out around 1.2×10⁻², giving a correction factor of just **4.3×** against
Bonferroni's **52×**. The implementation was verified by recomputing the threshold
brute-force independently (exact match) and confirming seed reproducibility, so what this
shows is that **g:SCS applied outside its design regime loses its conservativeness
guarantee**. All three corrected p-values are reported so this is visible per term.

**3. v2's row unit breaks the hypergeometric's distinct-draws assumption.** The same cell
type appearing across many tissues is not an independent draw, so v2's p-values are
anti-conservative. Visible in its output: MMACHC-positive v2 returns `histamine
secretion`, `prostaglandin production`, `peripheral tolerance` and more all at exactly
`8/16` — these are all mast-cell terms, inflated together by 8 mast-cell *rows*. v2 exists
for comparison with v1, not as a defensible primary analysis.

**4. g:Profiler's real defaults, checked against the official docs and the current
`gprofiler2` client** (`gost()`, the live tool's own interface — not the retired
`gProfileR` package):

| Parameter | g:Profiler default | Here |
|---|---|---|
| `correction_method` | `g_SCS` | ✅ g:SCS, primary |
| `user_threshold` (α) | `0.05` | ✅ `--alpha 0.05` |
| `domain_scope` | `annotated` ("only annotated genes") | ✅ `--universe annotated` |
| `ordered_query` | `FALSE` | ✅ query is an unordered set |
| `measure_underrepresentation` | `FALSE` (over-representation only) | ✅ one-sided in v1/v2 |
| `significant` | `TRUE` — **only significant terms are returned** | ⚠️ see below |
| term-size filter | **none** — `min_set_size`/`max_set_size` do not exist in `gost()`; they were a `gProfileR`-only parameter, removed from the current client | ⚠️ see below |

**`significant=TRUE` is deliberately not mirrored for scoring, and here is why.**
g:Profiler's default only affects what is *displayed* — the statistics underneath are
identical either way. In this pipeline, the ranked term list isn't just a display: it is
`go_predictions.json`, the exact object the IC-weighted F1/headroom scoring reads. Cutting
it to significant-only would make it empty or near-empty for a large share of genes for
reasons that have nothing to do with prediction quality — measured on 40 dev genes,
**10 of 40 genes have zero significant A1' (v3) terms**, median 3. Worse, `go_llm` always
answers with up to 10 terms regardless of significance, so forcing the no-LLM rungs down
to a significant-only list would not test "is the LLM better", it would test "does the LLM
get to answer with more candidates" — confounding the C1/C2 contrasts. So the full ranked
table stays what gets scored *primarily*. What g:Profiler would actually show by default is
logged alongside it as `enrichment_results_significant_only.json`, for inspection — and,
for v1–v4 only (the versions with a real hypothesis test), **also scored** as a SECONDARY
metric: `gopred_sigonly_f1_at_k` / `gopred_sigonly_headroom_at_k`, computed on the same
ranking restricted to terms with `p_gscs < alpha` (v1/v2) or `q_value < alpha` (v3/v4),
alpha defaulting to g:Profiler's own `0.05`. `gopred_sigonly_n_significant` is logged even
when it's 0, and scoring is skipped rather than recorded as a misleading F1=0 in that case
(`core/go_scoring.log_significant_only_score`). This is informational — "how good is arm 1
when it's actually confident" — and is **not** part of the pre-registered contrasts, for the
same list-length-fairness reason above. On MSR1 it already shows the expected direction:
full-list F1@3 = 0.022 (headroom −0.10) vs significant-only F1@3 = 0.044 (headroom +0.04).

**The `--min-term-size 3` filter has no g:Profiler default to approximate — there isn't
one in the current client.** It exists only because our annotation universe is far
sparser than genome-wide gene annotation: of the 124 candidate GO terms, **55 are carried
by exactly 1 cell type and 17 more by exactly 2** (measured on the real cache). A term
with one or two carriers has no meaningful "is this count surprising" question to ask —
testing it anyway only adds a wasted hypothesis to the multiple-testing correction and
occasionally a spurious "100% enriched" from one data point. `--min-term-size 3` removes
these 72 terms before testing, leaving the 52 "candidate" terms used throughout. This is
our own necessary addition, not a reproduction of anything g:Profiler does.

### Versions

| version | statistic | unit | query/background for MMACHC positive |
|---|---|---|---|
| `v1` | g:Profiler-style (hypergeometric + g:SCS) | distinct cell types — each CL term once, matching g:Profiler's semantics | 118 / 359 = 33% |
| `v2` | g:Profiler-style | dataset rows (CL×tissue pairs) — preserves tissue weighting | 230 / 1470 = 16% |
| `v3` | corrected (binomial vs two-factor null) | distinct cell types | — (one contrast run, `input_set=contrast`) |
| `v4` | corrected | dataset rows | — |
| `v5` | corrected + co-annotation transfer | distinct cell types | — (one contrast run) |
| `v6` | corrected + co-annotation transfer | dataset rows | — |

v5/v6 are described under "Co-annotation transfer" below. v3/v4 compute the positive-vs-rest contrast internally, exactly as `statistical` v2 does,
so they produce ONE run per gene rather than a positive and a negative direction. They
exist as the **no-LLM counterpart of `go_llm` v3/v4**: same table, same statistic, no
model. See the experiment protocol below.

v2's ratio is *better* (16% vs 33%, ceiling 6.39× vs 3.04×) because rows inflate the
background faster than the query — but that improvement is purchased with the
non-independence in caveat 3, so it is not a free win.

### Parameters

| Flag | Default | Notes |
|---|---|---|
| `--prompt-version` | v1 | `v1`/`v2` g:Profiler-style (cell types / rows); `v3`/`v4` corrected (cell types / rows) |
| `--set` | both | v1/v2 only: `positive`, `negative` or `both`; ignored by v3/v4 |
| `--universe` | annotated | `annotated` = original behaviour (reproduces earlier runs); `called` = the comparison ladder's universe (annotated items the gene is called for, biological-process annotations only). v3/v4 always use `called` |
| `--score` / `--no-score` | on | Score the ranked prediction against the gene's own GO annotation, into the same run |
| `--gene-split` | none | Tag the run `dev`/`test`/`smoke` for the experiment |
| `--transfer-min-genes` | 30 | v5/v6: a gene-level GO term needs this many annotated genes to enter the transfer vocabulary (2,738 terms at 30) |
| `--transfer-top-n` | 100 | v5/v6: length of the ranked prediction list |
| `--go-depth` | 3 | Inheritance depth for `capable_of` |
| `--min-term-size` | 3 | **55 of the 124 candidate GO terms are carried by exactly 1 cell type, 17 more by exactly 2** — our own addition (see Fidelity caveat 4), not a g:Profiler default |
| `--max-term-size` | 0 | 0 = unlimited |
| `--n-simulations` | 2000 | Random queries for the g:SCS threshold, matching g:Profiler's original simulation count |
| `--alpha` | 0.05 | matches g:Profiler's `user_threshold` default |
| `--seed` | 42 | g:SCS threshold is reproducible per seed |

### What it logs

Experiment `CellCoLLM/go_enrichment`, with `model="none"` as the explicit marker that no
model was involved, and `prompt_mode="go_overrepresentation"` (v1/v2) or `"go_corrected"`
(v3/v4).

- **Params**: `unit`, `statistics` (`gprofiler`/`corrected`), `universe`, `go_depth`,
  `min_term_size`; v1/v2 add `max_term_size`, `n_simulations`, `alpha`,
  `correction_methods`; v3/v4 add `baselines_key`, `grand_mean_rate`, `gene_rate`,
  `n_candidates`.
- **Metrics, v1/v2**: `n_background`, `n_query`, `query_background_ratio`,
  `max_fold_enrichment_possible`, `n_terms_total`, `n_terms_tested`,
  `n_significant_gscs` / `_bonferroni` / `_fdr_bh`, `gscs_threshold`, and
  `top_term_*` (p_gscs, p_value, fold_enrichment, precision, recall).
- **Metrics, v3/v4**: `n_pos`, `n_called`, `gene_rate`, `n_terms_tested`,
  `n_significant_q05`, `max_excess`.
- **Scoring metrics** (all versions, `--score`): `gopred_f1_at_{1,3,5,10}`,
  `gopred_f1_mean_1_10`, `gopred_ceiling_at_{1,3}`, `gopred_floor_at_{1,3}`,
  `gopred_headroom_at_{1,3}`, `gopred_n_predictions`, and `gopred_effect_*` for the
  effect-size ranking (see the experiment protocol).
- **Tags**: `low_power_warning` (v1/v2), `condition`, `gene_split`, `stratum`.
- **Artifacts**: `enrichment_results.json` (v1/v2 also `.tsv` with g:Profiler-shaped
  columns, full precision, and `gscs_simulation.json` — query size, threshold,
  simulation count, seed, so the correction is auditable), `go_predictions.json` (the
  ranking that is scored: enriched terms by p ascending, then the rest) and
  `go_predictions_effect.json` (the same terms ranked by effect size instead, because
  ranking by p is dominated by term size).

### Scoring (added for the GO-prediction experiment)

The judge-based layers (`validation/score_discrimination.py`, `score_go_match.py`) still do
not apply — there is no property statement — and `scripts/run_test_genes.py` still skips
them for this approach. Instead every run now scores its own ranked prediction list against
the gene's own GO annotation, deterministically, into the **same MLflow run** (`gopred_*`
metrics; see the experiment protocol for the definitions). v1/v2 score the *positive*
direction only; the negative direction stays diagnostic. `--no-score` disables this.

`--universe called` selects the universe used by the comparison ladder: annotated items the
gene is called for, biological-process annotations only. The default `annotated` is the
original behaviour and is kept so earlier runs reproduce (it also counts missing items and
non-BP `capable_of` targets — e.g. *pancreatic ductal cell*'s only annotation is the
molecular function "bicarbonate transmembrane transporter activity", which made it count as
annotated: 359 cell types against the ladder's 358).

### Co-annotation transfer (v5/v6) — a no-LLM baseline that can leave the vocabulary

The evidence terms (52 cell-level GO terms such as *phagocytosis*, *cell motility*) and the terms
a gene is annotated with are different vocabularies. v1–v4 can only ever answer with the 52, but
an LLM can name anything, which would turn any free-form comparison into a vocabulary-size contest.
v5/v6 remove that confound by *learning* the cell-level → gene-level mapping from GOA:

```
s(c, T)   = -log10 P(X >= J),  X ~ Hypergeom(M genes, n_T genes with T, n_c genes with c),  J = genes with both
score(T)  = sum over enriched evidence terms c of  w_c * s(c, T),    w_c = -log10 p_c from the corrected statistics
```

Sanity check on the learned mapping: *phagocytosis* → endocytosis, import into cell;
*visual perception* → sensory perception of light stimulus; *spermatogenesis* → reproductive
process, gamete generation.

**Leave-one-out.** The target gene's own annotation is removed from every count before scoring;
otherwise its true terms would leak into the association table. Verified against a brute-force
recomputation that physically removes the gene: maximum difference 0.00 over 2,738 terms. The other
~16k genes' annotations (non-IEA, the same policy as the ground truth) are legitimate training signal.
The LLM arms have no such table, so this is a deliberately strong baseline: if the LLM cannot beat
it, its world knowledge adds nothing over a lookup built from the resource it is scored against.

Logs `transfer_explanation.json` (for the strongest evidence terms, their most associated
gene-level terms), `frac_in_table` (share of the top 10 that are evidence terms, ~0.15–0.2 in dev),
`gopred_f1_at_k`. No headroom: the answer space is the whole vocabulary, so there is no
candidate-vocabulary ceiling.

### Observed result worth recording (MMACHC, v1 positive — the only high-power case)

Only one term reaches significance: `T cell mediated immunity`, 1.83× (21/35). But
`bile acid biosynthetic process` and `detoxification` both sit at exactly **3.04×** — the
ceiling — at 4/4 cell types, failing significance purely on term size. For a cobalamin
metabolism gene those are the biologically plausible hits, and the method cannot promote
them because only 4 cell types in the entire background carry the annotation. That is the
coverage limitation (§L5) and the term-size floor interacting, not a negative result about
the gene.

## `go_llm` (`approaches/go_llm/run_go_llm.py`) — arm 2

Arm 2 of the two-arm experiment: the LLM is given the **same** cell-type GO evidence the
no-LLM baselines see and asked to **interpret** it — what do the expressing cell types have in
common, what machinery would a gene need to be expressed in exactly this pattern — and then name
the GO biological-process terms of the undisclosed gene itself. Predictions are scored
**deterministically** against the gene's own GO annotation — there is no judge model, so the
judge≠generator guard and judge noise do not exist here.

### Versions and flags

| version | table shown | unit |
|---|---|---|
| `v1` | raw counts (carriers `K`, expressed-in `k`) | distinct cell types |
| `v2` | raw counts | dataset rows |
| `v3` | corrected: observed vs expected, excess, q — an **enriched** block then a **depleted** block, each strongest-first by p | distinct cell types |
| `v4` | corrected | dataset rows |

Odd versions are the cell-type unit and even the row unit, in both `go_llm` and
`go_enrichment`. **`go_llm` v3 minus `go_enrichment` v3 isolates what the LLM adds on
identical statistics.**

| Flag | Values | Meaning |
|---|---|---|
| `--output-mode` | `freeform` (**primary**), `constrained` (control) | freeform: name up to 10 GO BP terms for the gene itself; resolved to ids by exact name / EXACT synonym after punctuation normalisation (`GOOntology.resolve_name`), unresolved counted, never fuzzy-matched. constrained: pick and rank up to 10 GO ids copied from the evidence — the LLM can only *re-rank* the statistics, so this is the control "how well can it re-weight them?" |
| `--evidence` | `true`, `mismatched`, `none` | `mismatched` shows a **donor** gene's table (control: is the evidence used at all?); `none` shows no table (control: the LLM's prior) |
| `--include-definition` | off | ablation; off for information parity with the baselines |

The donor is chosen deterministically from a hash of `(seed, target)`, never the target,
with an overall positive rate within ±0.05 of the target's — so the control preserves how
broad the gene is and differs only in *which* cell types are positive.

There is **no abstention key**: a ranked-list metric needs a ranking, and an empty answer
would score below the random floor for reasons unrelated to biology.

Prompts are 1.5–3.1k tokens (median 2.3k; measured `prompt_eval_count`, at most ~8% of the context
window; a character-count estimate under-reads by ~30%, so trust the logged
`actual_prompt_tokens`). Measured latency: median 6.8 s, max 13 s per call. Nothing in a prompt
carries gene identity; `verify_blinding` checks the symbol and HGNC aliases are logged as a tag.

### Why the prompt asks for interpretation

The first version asked the model to *choose the terms from the table that describe the gene*. On
real output that turned the model into a reader of the statistics: on 8 dev genes 88–91% of its
free-form answer was table terms re-listed (`frac_in_table`), its rationale restated the table
("statistically significant and biologically coherent"), and it ranked *depleted* terms first
because the table was sorted by two-sided p. Constrained mode cannot infer at all by construction:
its only allowed answers are the evidence.

The templates were therefore rewritten **in place** (same four versions, same files):

- the evidence is *clues, not answers*: the prompt says the answer terms describe what the gene
  product **does**, a different level from the cell-behaviour terms shown;
- a **`hypothesis` field comes first** — the shared program or cell state, the machinery it
  implies (receptors, transporters, enzymes, regulators), what the under-expression rules out —
  and only then the predicted terms, so the answer follows from a stated interpretation;
- evidence is split into an **enriched block and a depleted block** (each strongest-first), so
  copying the top of the list cannot select depleted terms, and the top of the enriched block is
  exactly the no-LLM ranking (`go_enrichment` v3/v4);
- free-form asks for *process* terms only ("…activity" terms are molecular functions and are
  rejected) and for the exact GO wording, or a more general term whose name the model is sure of.

Effect on the same 8 dev genes (real model, `qwen3:32b`; `frac_in_table` = share of the answer that is
just evidence terms):

| condition | `frac_in_table` before → after | F1@3 before → after |
|---|---|---|
| v3 free-form | 0.91 → 0.67 | 0.052 → 0.048 |
| v1 free-form | 0.88 → 0.82 | 0.060 → 0.081 |

**With n = 8 the F1 changes are within noise and are not evidence of improvement.** What did change
is behaviour: hypotheses are now genuine interpretations (for CCL8: immune cells engaged in
motility, phagocytosis and antigen handling → "a receptor, signalling molecule or cytoskeletal
regulator") and 33% of v3's answers are terms the evidence does not contain. It is still wrong in
instructive ways: for PAX4 (a pancreatic transcription factor) it inferred "secretory machinery" and
predicted vesicle terms, none of which resolved. About one prediction per run fails to resolve —
molecular-function wording, or near-miss names such as "synaptic transmission" for "chemical synaptic
transmission". Only punctuation is normalised ("G-protein coupled" = "G protein-coupled"); fuzzy
matching was rejected because it mapped "visual signal transduction" to "ABA signal transduction".
Prompt wording was tuned on dev genes only, for two rounds.

```bash
python approaches/go_llm/run_go_llm.py --gene ENSG00000149554 --prompt-version v3
python approaches/go_llm/run_go_llm.py --gene ENSG00000149554 --prompt-version v3 --output-mode freeform
python approaches/go_llm/run_go_llm.py --gene ENSG00000149554 --prompt-version v3 --evidence mismatched
```

---

## The GO-prediction experiment — protocol

### Question and design

Does an LLM add anything to the classical statistics, on identical evidence? The evidence
is GO biological-process terms attached to cell types (CL `capable_of`, inherited down
`is_a`); the ground truth is the gene's own GO annotation (GOA, **non-IEA** evidence only —
IEA is often derived from the same sources that annotate cell types). The gene is never an
input. Both sides are **propagated up the GO DAG (`is_a` + `part_of`)** and compared as
**information-content-weighted** sets (CAFA protocol), so a match on `biological_process`
scores nothing.

**The ladder** (each rung differs from its neighbour in one thing):

| rung | LLM | statistics | version |
|---|---|---|---|
| A1 | no | g:Profiler-style | `go_enrichment` v1 (cell types) / v2 (rows) |
| **A1'** | no | corrected two-factor | `go_enrichment` v3 / v4 |
| **A2** | no | corrected + **co-annotation transfer** (can leave the vocabulary) | `go_enrichment` v5 / v6 |
| 2a | yes | none (raw counts) | `go_llm` v1 / v2 |
| 2b | yes | corrected (A1''s table) | `go_llm` v3 / v4 |

Two comparison families, because the two output modes answer different questions:

- **Primary — free-form.** The LLM interprets the evidence and names the gene's own GO terms; it
  can leave the 52-term evidence vocabulary. The no-LLM rung that can *also* leave it is **A2**
  (`go_enrichment` v5/v6); without it the comparison would be a vocabulary-size contest. Endpoint:
  raw `gopred_f1_at_3`.
- **Control — constrained.** The LLM can only re-rank the 52 evidence terms. Endpoint:
  `gopred_headroom_at_3`. This measures how well an LLM can re-weight statistics, not whether it
  can infer.
Candidate vocabulary: the **52** GO terms carried by ≥3 distinct annotated cell types
(124 in total; 358 of 677 cell types annotated), fixed for **both** units so ceilings,
floors and rankings are comparable across every rung. `core/go_evidence.py` is the single
source of the tables; the evidence module was checked to reproduce `go_enrichment`'s
counts exactly (0 mismatches across 4 genes × 2 units under `--universe called`).

### Scoring

```
truth       propagate(direct non-IEA BP annotations of the gene)
prediction  propagate(top-k predicted terms)
precision   IC mass of (prediction ∩ truth) / IC mass of prediction
recall      IC mass of (prediction ∩ truth) / IC mass of truth
F1@k        harmonic mean
ceiling@k   best F1 reachable by naming <=k candidate terms (greedy union)
floor@k     mean F1 of k random candidate terms
headroom@k  (F1@k - floor@k) / (ceiling@k - floor@k)
```

Worked example (TTI2, cell-type vocabulary): the gene's two direct annotations propagate to
28 terms with a total IC mass of 77.68 — dominated by *positive regulation of DNA damage
checkpoint* (IC 8.12). Only 9 of the 52 candidates share anything with it, and every shared
term is generic (*regulation of biological process*, *positive regulation of cellular
process*, …). Best candidate: *positive regulation of neutrophil chemotaxis*, overlap 7.32,
precision 7.32/106.11 = 0.069, recall 7.32/77.68 = 0.094, **F1 = 0.080 = the ceiling**; the
random floor is 0.006. Headroom is only defined for constrained mode: free-form output is
not confined to the candidate vocabulary, so its "headroom" could exceed 1 — free-form is
reported as raw F1@k against baselines.

### What the data says about the ceiling (from `scripts/analyze_go_ceiling.py`)

15,944 of 18,908 dataset genes are scoreable. Over them the median ceiling (best single
candidate) is **0.125** against a random floor of **0.016**; only **33%** of genes carry a
candidate term after propagation. The three standard genes (MMACHC, TTI2, CHEK1) have almost
no headroom (ceilings 0.02–0.08) and are **smoke-test genes only**. The best *constant*
predictor — the same fixed terms for every gene, no evidence — already reaches about 16% of
the way from floor to ceiling, so every rung must beat that before it means anything.

### Controls

| control | tests | computed |
|---|---|---|
| random | floor | analytically, headroom 0 by construction |
| constant prior | "was the evidence used at all?" | fit on **dev** genes only, applied to test genes, at analysis time. One over the 52-term vocabulary (constrained family) and one over 1,094 broad GO terms (free-form family) |
| mismatched evidence (A1, A1', A2, 2a, 2b) | evidence is used, not just the prior | donor gene, as above; no-LLM rungs at analysis time, LLM rungs as runs |
| no evidence (LLM) | the LLM's own prior | empty table |

### Genes, split, freezing

`scripts/select_go_genes.py` selects and splits (rules and defaults are in its docstring and
are written into the output). Eligibility: scoreable; ≥3 direct annotations (drops the
sparsely annotated genes where one matching term scores F1 = 1.0 trivially); overall
positive rate in [0.05, 0.60] (uses expression only, never truth); ceiling-minus-floor above
a quantile of the survivors. Primary stratum = the gene carries a candidate term. Split is
deterministic by hash. **All prompt wording, top-N, ranking and threshold choices are made on
the dev split only; then the pipeline is frozen and the test split is run once.** How that is
enforced is under "Run identity and freezing" below.

Selection uses truth-derived criteria (carrying, room). That is disclosed: results generalise
to the selected stratum, not to all genes. Three genes are not enough for an inference — the
per-gene noise is large; dozens to hundreds of genes are needed.

### Run identity and freezing

The git tag on a run is `commit` or `commit-dirty`. With a dirty tree, two different prompt versions
carry the same tag, and a resumable runner keyed on (condition, gene) alone would silently keep
results produced by an *old* prompt. So identity is by **content**:

- Every `go_enrichment` and `go_llm` run carries a **`code_hash` tag**: a hash of the files that shape
  its output — `go_llm`: its run script, all four prompt templates, the evidence/scoring/ontology code and
  the LLM plumbing; `go_enrichment`: its run script and the statistics/evidence/scoring/transfer/ontology
  code (`core/run_identity.py` lists them). Changing a prompt or a scoring function changes the hash;
  changing a README, or the *other* approach's code, does not. Restoring an edit restores the hash.
- **`scripts/run_go_experiment.py` skips a condition only if it was completed under the current hash.**
  Runs from other code (or from before hashes existed) are reported as stale and their conditions run
  again, so an interrupted run resumes, and a changed prompt is never mistaken for finished work.
- **`scripts/analyze_go_experiment.py` uses runs from exactly one code version** — the current one for
  dev, the *frozen* one for test — lists what it excluded, and refuses to run if nothing matches.
  `--allow-mixed` exists for plumbing checks and stamps the report as not a valid analysis.
- **`scripts/freeze_go_experiment.py`** writes `approaches/go_experiment_freeze.json` when dev work is
  finished: the code fingerprints (plus the sha256 of every file behind them, so a mismatch names the
  file), the data the results depend on (dataset hash, baselines key, CL and GO versions, the GOA file
  hash), the sha256 of the gene list, and the git HEAD and dirty state. **Commit the manifest with the
  code it describes.** `python scripts/freeze_go_experiment.py --check` compares the current state with it.
- **The test split will not run unless code, data and gene list all still match the freeze**
  (`--allow-unfrozen` overrides it and tags every run `unfrozen_override`, which the analysis excludes).
  Changing anything after freezing therefore needs a deliberate re-freeze that shows up in git history.

`data/` is git-ignored. Its derived files (`ontology_cache.json`, `feature_baselines.json`,
`go_baselines.json`, `go_ceiling_genes.tsv`, `go_ceiling_report.json`) are deterministic outputs of the
raw files in `data/ontologies/raw/` and rebuild with the four commands in the Scripts table; they came
back byte-identical after they were lost once.

### Pre-registered contrasts

Test split, primary stratum; paired bootstrap over genes (10,000 resamples), **Holm-corrected
within each family**.

**Primary family — free-form, raw `gopred_f1_at_3`:**

| | contrast | question |
|---|---|---|
| F1a/b | `go_llm` v3/v4 (true) − `go_enrichment` v5/v6 | does the LLM add to the same evidence + a data-driven mapping? |
| F2a/b | `go_llm` v3/v4 − constant prior | does it beat fixed terms named for every gene? |
| F3a/b | `go_enrichment` v5/v6 − constant prior | does the transfer baseline beat the prior at all? |
| F4 | `go_llm` v3 true − mismatched | is the evidence used? |
| F5a/b | `go_llm` v3/v4 − v1/v2 | do statistics help the LLM? |

**Control family — constrained, `gopred_headroom_at_3`:**

| | contrast | question |
|---|---|---|
| C1a/b | `go_llm` v3/v4 − `go_enrichment` v3/v4 | does the LLM add to the *same* statistics when it can only re-rank? |
| C2a/b | `go_enrichment` v3/v4 − v1/v2 | value of the corrected statistics |
| C3a/b | `go_llm` v1/v2 − constant prior | raw counts vs no evidence at all |
| C4 | `go_llm` v3 true − mismatched | is the evidence used? |
| C5a/b | `go_llm` v3/v4 − v1/v2 | do statistics help the LLM? |

The analysis also reports `frac_in_table` per free-form condition, so "the LLM beat the baseline" can
be checked against "the LLM only re-listed the table".

**Signal-existence gate (one per family):** a rung passes only if it beats *both* the constant prior *and* its
own mismatched-evidence control with a bootstrap CI excluding 0. If no rung passes, the
analysis reports **INCONCLUSIVE — not "the LLM adds nothing"**: a null caused by the atlas
artifacts (immune-dominated, batch-driven tissues) is indistinguishable from a null caused
by the LLM, and the gate exists to stop the former being read as the latter.

### Running it

```bash
python scripts/analyze_go_ceiling.py --tsv data/go_ceiling_genes.tsv   # once: per-gene ceilings
python scripts/build_go_baselines.py                                   # once: per-GO-term baselines, both units
python scripts/select_go_genes.py --n-dev 40 --n-test 120              # writes data/go_experiment/genes.tsv
python scripts/run_go_experiment.py --split dev --stage baselines      # no LLM, seconds per gene
python scripts/run_go_experiment.py --split dev --stage freeform --limit 10   # the primary family
#   ... tune on dev only, then freeze (commit the manifest together with the code) ...
python scripts/freeze_go_experiment.py --note "what was decided, after which dev round"
python scripts/run_go_experiment.py --split test --stage all           # refuses unless frozen; resumable
python scripts/analyze_go_experiment.py --eval-split test
```

Per gene: 6 no-LLM runs (v1–v6) and 16 LLM runs (free-form 7, constrained 9). At the measured
median of 6.8 s per call that is roughly 2 minutes per gene, so about 3–4 hours for 100 test genes.
`--stage` allows free-form first. The runner is
in-process (one load of the dataset, ontology and annotations), resumable (a condition
already COMPLETED for a gene is skipped), and aborts after 5 consecutive failures so a dead
server is not silently ground through. `CELLCO_MLFLOW_URI` redirects runs to a scratch store.

### One thing already observed

On the six dev genes examined, the corrected statistic and g:Profiler's hypergeometric produce nearly
identical top-3 sets (order differs; F1@3 scores the union), so C2 is expected to be close
to zero: ranking by p-value is dominated by term size and sample size, which both statistics
share, and the baseline correction moves the ranking little. Whether that holds at scale is
what the test split answers.

### Caveats that stay (all arms affected equally, so relative comparisons hold)

- Ground truth is incomplete; absolute scores stay low.
- Cell-type GO and gene GO share literature provenance, so this measures **rediscovery of
  annotated biology**, not novel discovery.
- Atlas artifacts apply equally to every arm; the gate catches a null caused by them.
- The constrained comparison understates the LLM's real edge (naming terms outside the
  vocabulary); free-form is where that would show, with the vocabulary confound stated.
- The transfer baseline (A2) is trained on other genes' GOA annotations, which the LLM arms are
  not given explicitly — a deliberately strong baseline. Its leave-one-out removes only the target
  gene, so annotation of *closely related* genes (paralogs) still informs it.
- A free-form answer that leaves the evidence vocabulary can still be scored only against terms
  the resolver can match; unresolved terms (`n_invalid`) are dropped, not penalised, so they lower
  the effective list length rather than the score directly.

### Decisions taken and limitations accepted for now

These were considered and deliberately left as they are; they are recorded so the results are read with
them in mind.

- **Truth policy: non-IEA.** The ground truth is the gene's GOA biological-process annotations with IEA
  and ND excluded. That still includes curated but *non-experimental* evidence: of the 130,285 rows kept,
  **47.5% are experimental** (IDA 28.4%, IMP 16.7%, IGI 1.4%, IEP 0.7%, …) and **52.5% are not** (IBA
  19.8%, ISS 14.8%, TAS 9.0%, NAS 8.3%, IC 0.6%). Some of the latter are inferred from sequence or
  phylogeny rather than measured, so the truth is broader — and partly more circular with respect to
  cell-type annotation — than an experimental-only truth would be. No experimental-only sensitivity run
  has been made; if a result depends on this choice, that is the first check to run.
- **Arm 1 coverage.** Only 358 of 677 cell types carry any GO term (53%), so part of a gene's expression
  signal never reaches the evidence; the annotated share of each gene's positives is not logged. Genes
  whose positives are mostly unannotated cell types simply have little evidence to work with.
- **Arm 2 hypothesis text.** The model's `hypothesis` is logged (`response_parsed.json`) but not scored or
  measured for restating its input; only `frac_in_table` (how much of the *answer* re-lists evidence
  terms) is computed.

---

## Batch-running test genes

```bash
python scripts/run_test_genes.py --run naive:v1 --run enriched:v2 --run statistical:v1
python scripts/run_test_genes.py --run go_enrichment:v3 --run go_llm:v3
```

Runs every `approach:version` combo (repeatable `--run approach:version`) across the 3
standard test genes (hardcoded default; override with repeatable `--gene`), with fixed
defaults otherwise (model from `.env`). One subprocess per run. `--set both` is passed to
`naive`, `enriched` and `go_enrichment` only; `statistical` and `go_llm` have no directions
and yield one `contrast` run per gene. Add `--score-discrimination --judge-model <model>` to
score every run produced with the judge-based layer; the lookup handles all three `input_set`
values. `go_enrichment` and `go_llm` are **skipped** by the judge layers (they produce a
ranked GO list, not a property statement, and score themselves inline). This script has no
built-in knowledge of which versions exist, so future versions work automatically once
registered in the approach's own `PROMPT_VERSIONS`.

That is the right tool for smoke tests on the three standard genes. For the GO-prediction
experiment itself — thousands of runs — use `scripts/run_go_experiment.py` instead, which
runs in-process and is resumable (see the experiment protocol). The three standard genes
have almost no GO headroom (ceilings 0.02–0.08) and are not suitable for drawing conclusions.

---

## Scripts and data files

| Path | What it is | Needs |
|---|---|---|
| `scripts/build_ontology_cache.py` | Extracts CL / UBERON / GO terms (incl. `capable_of` → GO, UBERON `part_of`) into `data/ontology_cache.json` | `data/ontologies/raw/{cl,uberon}.obo` |
| `scripts/build_feature_baselines.py` | Per-feature baselines over all genes for `statistical` v2 → `data/feature_baselines.json` | ontology cache |
| `scripts/analyze_go_coverage.py` | How many **cell types** carry GO terms, terms per cell type, vocabulary size (no genes involved) | ontology cache |
| `scripts/analyze_go_ceiling.py` | Per-gene oracle ceiling and random floor for the cell-type GO vocabulary → `data/go_ceiling_genes.tsv`, `data/go_ceiling_report.json`. `--null-runs 0` takes ~30 s; the default 20 draws take ~9 min | `data/ontologies/raw/{go-basic.obo,goa_human.gaf.gz,hgnc_complete_set.txt}` |
| `scripts/build_go_baselines.py` | Per-GO-term baselines at both units → `data/go_baselines.json` (refused if stale) | dataset, ontology cache, GO |
| `scripts/select_go_genes.py` | Eligibility, strata and dev/test split → `data/go_experiment/genes.tsv` | `go_ceiling_genes.tsv` |
| `scripts/run_go_experiment.py` | Runs every rung × gene in-process, resumable | genes file; LLM server for the LLM stages |
| `scripts/analyze_go_experiment.py` | Contrasts, controls, gate → `data/go_experiment/analysis_*.md/.json` | MLflow runs |
| `scripts/summarize_go_matches.py` | Per-condition match/mismatch overview (below) → `data/go_experiment/match_summaries/*.json` | MLflow runs for that one condition |
| `scripts/freeze_go_experiment.py` | Writes / checks the freeze manifest `approaches/go_experiment_freeze.json` (code fingerprints, data versions, gene-list hash) | genes file |
| `scripts/run_test_genes.py` | Subprocess-per-run smoke runner for the 3 standard genes | — |

### Match/mismatch overview (`scripts/summarize_go_matches.py`)

A second, cheaper view of a condition's output than the IC-weighted F1/headroom score: for every
significant (or, where no significance test exists, predicted) term, is it an **exact** match to
the gene's own direct GO annotation, a **generalisation** (reached by climbing up from a true term
— the common case, e.g. "phagocytosis, engulfment" (true) → "phagocytosis" (predicted), 1 edge),
a **specialisation** (climbing up from the prediction reaches a true term — plausible but
unannotated), or **no match** at all (a lateral relation through a shared ancestor, or nothing
informative in common). Generalisation and specialisation are each binned by edge distance
(`core/go_match.py`; default bins `d=1, d=2, d=3, 3<d<=5, 5<d<=10, d>10`).

This is a coarser, simpler measure than the primary score: a raw edge count, with every edge
weighted the same even though the DAG is uneven (see Fidelity caveat 4 above) — it exists for an
interpretable breakdown and notebook charts, not to replace IC-weighted F1/headroom.

**Run once per condition, by design.** `--condition go_enrichment:v3:contrast` or
`--condition go_llm:v3:freeform:true` (one approach:version[:mode:evidence] per invocation), writing
its own `match_summary__<condition>__<split>.json`. Re-running one arm after a change only means
re-running this for that one condition; every other condition's file is untouched (verified: file
mtimes for unrelated conditions do not change). Like the main analysis, it is restricted to the
CURRENT code hash for that approach (`--allow-mixed` for plumbing checks only), and it is
deterministic — re-running an unchanged condition reproduces byte-identical bin counts.

**Significance has no meaning for every condition**, and the output says so rather than
pretending otherwise: `go_enrichment` v1/v2 use g:SCS (`p_gscs < alpha`), v3/v4 use BH
(`q_value < alpha`), but v5/v6 (co-annotation transfer) and every `go_llm` condition are rankings
with no hypothesis test at all — for those, the full ranked `go_predictions.json` list is used as
the "significant" set, and `significance_basis` in the output states this explicitly.

The output is **one JSON file**: pretty-printed (so it reads directly), with a tidy `per_gene`
table (`pandas.DataFrame(data["per_gene"])` loads straight in — gene, n significant, n exact/
upward/downward/no-match — for notebook charts), the aggregate histograms, and the exact
`mlflow.run_ids` it was built from, so a summary is traceable without re-deriving it by hand.

```bash
python scripts/summarize_go_matches.py --condition go_enrichment:v3:contrast --gene-split test
python scripts/summarize_go_matches.py --condition go_llm:v3:freeform:true --gene-split test
```

`data/ontologies/raw/` holds the downloaded ontologies and annotations: `cl.obo`,
`uberon.obo`, `go-basic.obo` (release 2026-07-26), `goa_human.gaf.gz` and
`hgnc_complete_set.txt` (the Ensembl ↔ UniProt map that joins the dataset to the GAF).
Core modules for the GO work: `core/go_ontology.py` (DAG, propagation, IC, name index),
`core/go_evidence.py` (evidence tables, corrected statistics, rankings),
`core/go_scoring.py` (deterministic scoring), `core/go_match.py` (exact/generalisation/
specialisation classification by edge distance), `core/run_identity.py` (code fingerprints, freeze), `core/go_transfer.py` (co-annotation transfer,
leave-one-out), `core/go_experiment.py` (shared loader, donor choice), `core/go_enrichment.py`
(g:Profiler-style test).
