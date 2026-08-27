# Cell-Type Contrast Pipeline — Requirements Specification

**Status:** Draft v0.1
**Purpose:** Requirements document for a pipeline that infers the biological context of poorly characterised genes by contrasting the cell types that express them against matched cell types that do not, and validating hypotheses by independent replication across species.
**Audience:** Human collaborators and coding agents (e.g. Claude Code) implementing the pipeline.

---

## 0. Overview

### 0.1 Core idea

Genes are often poorly characterised. Cell types, by contrast, are richly described — anatomically, functionally, morphologically, and developmentally — in curated ontologies. The pipeline exploits this asymmetry.

For a query gene:

1. Identify cell types in which the gene is reproducibly expressed (**positive set**).
2. Identify **matched** cell types in which it is weakly expressed or absent (**negative set**).
3. Ask an LLM to propose, in open-ended natural language, the most specific biological property that is **enriched in the positive set and depleted in the negative set**.
4. Independently score that hypothesis for **coverage**, **leakage**, and **specificity**.
5. Repeat the whole procedure independently in a second species using the gene's ortholog.
6. Compare the two independently derived hypotheses. Convergence is treated as evidence of a conserved functional context.

This is a form of enrichment analysis in which **cell types are the entities** and **natural language is the annotation space**.

### 0.2 Two-stage architecture

```
                    Human input (file + CLI flags)
                                 |
              +------------------+------------------+
              |                                     |
      STAGE 1 (species Y)                   STAGE 1 (species Z)
      -- runs in isolation --               -- runs in isolation --
              |                                     |
        Step 2 Atlas lookup                   Step 2 Atlas lookup
        Step 3 Pos/neg sets [UNCERTAIN]       Step 3 Pos/neg sets [UNCERTAIN]
        Step 4 Ontology text                  Step 4 Ontology text
        Step 4b Preprocessing (optional)      Step 4b Preprocessing (optional)
        Step 5 LLM contrast                   Step 5 LLM contrast
        Step 6 Scoring                        Step 6 Scoring
        Step 7 Artifact                       Step 7 Artifact
              |                                     |
              +------------------+------------------+
                                 |
                     STAGE 2 — Step 8
                Cross-species comparison + null calibration
```

**Hard architectural constraint:** the two Stage 1 branches must not exchange data, in either direction, at any point before Step 8. This is not a stylistic preference — the entire evidential value of cross-species convergence depends on it. Enforce structurally (separate working directories, no shared mutable state, an input manifest per branch), not merely by convention.

### 0.3 Execution modes

Mode is set by a CLI flag and changes what "success" means. Pipeline mechanics are otherwise identical.

| | `discovery` | `benchmark` | `negative_control` |
|---|---|---|---|
| Gene chosen | Poorly characterised | Well characterised | Any / synthetic |
| Ground truth | None exists | Exists, held internally, hidden from LLM | Known to be absent |
| Question asked | "What does this gene do?" | "Does the pipeline recover what we already know?" | "Does the pipeline correctly find nothing?" |
| Success criterion | Internal evidence: coverage, leakage, specificity, cross-species convergence | External: semantic similarity of hypothesis to known function | **Abstention / low confidence** |
| Output is | A testable hypothesis | A performance metric | A calibration metric |
| Failure means | Possibly no signal (uninformative, not necessarily wrong) | Pipeline broken or miscalibrated | Pipeline produces false confidence |

`negative_control` inputs include:
- Broadly expressed housekeeping genes (no discriminating property can exist)
- Randomly assembled cell-type sets (no biological coherence)
- **Contamination gradients** — 25 % / 50 % / 75 % mismatched cell types swapped into the positive set

The contamination gradient generalises the Hu et al. "50/50 mix" design. Real-vs-random is the easy test; **partial contamination is the diagnostic one**, because real positive/negative sets are never perfectly clean either. Confidence should degrade smoothly and proportionally with contamination — a method that only separates clean from pure-noise extremes is much less useful than headline real-vs-random numbers suggest.

### 0.4 Blinding

In `benchmark` mode the pipeline knows the answer and must prevent the LLM from knowing it. Otherwise the benchmark measures memorisation, not reasoning.

Blinding requirements:
- Never pass gene symbol, aliases, or accession IDs to any LLM call — refer only to "the query gene".
- Strip gene names, aliases, and known disease associations from retrieved ontology text (Step 4).
- Strip the query gene from any marker-gene list in a cell-type description (circular evidence).
- Verify blinding automatically and **fail the run** on a hit, rather than proceeding.

Blinding should also be applied in `discovery` mode, for consistency: if benchmark results were obtained under blinding, discovery results must be produced under the same conditions or the two are not comparable.

---

## Step 1 — Human Input

### 1.1 What it should do

Accept two distinct kinds of input and resolve them into a fully specified, unambiguous task that the rest of the pipeline can execute without further human judgement:

1. **A structured input file** (JSON/YAML/CSV) describing **what** to analyse — gene(s), species, per-gene overrides.
2. **CLI flags** describing **how** to run — mode, thresholds, model choice, ontology/atlas versions, output paths.

The separation matters: the file is reusable across runs (e.g. the same 100-gene benchmark set executed under three different thresholds) without editing the file.

### 1.2 Interface sketch

```bash
python run_pipeline.py \
  --input genes.json \
  --mode benchmark \
  --species human,mouse \
  --expression-threshold 0.25 \
  --confidence-cutoff 0.7 \
  --atlas-version tabula-sapiens-v1 \
  --cl-version 2025-03-15 \
  --output results/run_042/
```

```json
{
  "queries": [
    {"gene": "CFTR",  "species": ["human", "mouse"]},
    {"gene": "SCN9A", "species": ["human", "mouse"]}
  ]
}
```

### 1.3 What it must resolve

- **Gene identity** — one canonical, stable identifier, not the user's typed string.
- **Species scope** — explicit species pair or set; never silently inferred.
- **Mode** — `discovery | benchmark | negative_control`.
- **All parameters governing downstream steps** — captured once, here, so they are reproducible and never silently re-decided mid-pipeline.

### 1.4 What is important

1. **Gene identity resolution must be unambiguous before anything else runs.** Genes are referenced across overlapping systems — HGNC symbol, Ensembl ID, Entrez/NCBI Gene ID, UniProt accession. Symbols are not stable identifiers: they get renamed, reused, and are ambiguous across species. Resolve to a canonical ID and carry the **ID**, not the symbol, through the pipeline.
   - Human CFTR = `ENSG00000001626`; mouse *Cftr* = `ENSMUSG00000041301`. The ID prefix also disambiguates species.
2. **Explicit species scope**, with a stated default if omitted.
3. **Reproducibility of configuration.** Ontology and atlas releases are versioned and change over time. Pin exact versions, or benchmark results become non-reproducible as underlying databases update.
4. **Mode is a first-class parameter**, not an afterthought — it constrains what data may flow downstream (see §0.4).
5. **Precedence order is explicit and logged**: flags override file, file overrides defaults.
6. **Every run writes its fully resolved config** (merged file + flags + defaults) to the output directory, so the run is reproducible from that artifact alone.
7. **Batch is the default assumption.** Benchmark and negative-control modes require many genes per run; a single-gene query is simply a list of length one.
8. **Fail-fast validation of the entire file before any expensive work begins** — do not discover a malformed gene ID at query 87 of 100 after an hour of compute.

### 1.5 Edge cases

**Gene resolution**
- Ambiguous or deprecated gene symbol (renamed, withdrawn, or matching multiple genes/species)
- Paralog confusion — close paralogs may be returned instead of, or alongside, the intended gene
- Non-protein-coding genes (lncRNAs etc.) — decide upfront whether in scope or explicitly excluded

**Availability**
- Gene has **no annotated ortholog** in the second species → fail gracefully with a clear reason; must not silently produce a low-confidence result that resembles a real negative finding
- Gene **not present in either atlas** → a *different* failure mode from "no ortholog"; must be distinguished
- 1:many or many:many orthology — meaningfully harder than clean 1:1; record orthology type and confidence

**Input handling**
- Flag and file disagree on a parameter → precedence rule applied and logged, never silently resolved
- Malformed/unparseable file, or valid file with invalid content
- Duplicate entries in the input file
- Partial failure mid-batch → **recommended: skip, log, continue**, with a summary of failures at the end
- Mode flag incompatible with file content (e.g. `--mode benchmark` on genes with no ground truth)
- Resuming an interrupted batch without recomputing completed queries
- User supplies a **custom positive/negative cell-type set** manually, bypassing Step 3 — support as an override for expert-driven runs and debugging

### 1.6 Additional categories to specify

- **Input validation and error reporting** — what is returned on failure, and at which stage (fail fast vs. best-effort with warnings)
- **Species availability registry** — maintained list of species with both (a) an atlas and (b) ontology coverage; validate at input rather than discovering mid-pipeline
- **Config versioning / provenance** — log atlas version, CL version, ortholog DB version per run
- **Blinding mechanism** — formal mechanism to withhold gene identity from downstream LLM calls while tracking ground truth internally

### 1.7 Initial implementation ideas

- Use a gene-normalisation service (e.g. **MyGene.info**) to resolve arbitrary input (symbol, alias, accession) to a canonical ID across species in one call.
- Use **Ensembl Compara** or **OrthoDB** for ortholog resolution; return orthology type (1:1 vs 1:many) and confidence as part of the resolved input.
- Represent the fully resolved input as a single structured config object (JSON schema) consumed by every downstream step — this becomes the run's audit trail.

---

## Step 2 — Atlas Lookup (Expression Data Retrieval)

### 2.1 What it should do

For each resolved gene + species pair, query the appropriate single-cell atlas and return a **per-cell-type expression summary** covering **every annotated cell type in that atlas** — not only those that express the gene. Negatives matter as much as positives, so **this step must not filter**. It produces the full quantitative readout that Step 3 thresholds.

Runs **per species, in isolation**. The independence requirement begins here, not at the LLM step.

### 2.2 Output shape

Per gene, per species: a table of `cell_type_id → expression statistics`, plus provenance metadata (atlas, version, cells/donors per cell type).

**Header / provenance block:**

```json
{
  "query": {
    "gene_symbol": "CFTR",
    "ensembl_id": "ENSG00000001626",
    "species": "homo_sapiens"
  },
  "source": {
    "atlas": "Tabula Sapiens",
    "atlas_version": "v1.0",
    "accessed_via": "cellxgene Census",
    "census_version": "2025-01-30",
    "normalization": "log1p(CP10K)",
    "aggregation": "pseudobulk mean per cell_type"
  },
  "gene_status": "PRESENT_IN_REFERENCE",
  "n_cell_types_returned": 174
}
```

**Where these fields come from:**

| Field | Origin |
|---|---|
| `ensembl_id` | Step 1 gene resolution (this is the actual atlas query key) |
| `atlas`, `atlas_version` | Step 1 — CLI flag or per-species default |
| `census_version` | Step 2 runtime — read from the Census API at query time |
| `accessed_via` | Step 2 — which client performed the query |
| `normalization`, `aggregation` | Step 2 — what this step actually did, recorded so it is not a hidden assumption |

**Per-cell-type table (excerpt; illustrative values):**

| cl_id | cell_type_label | tissue | mean_expr | pct_expressing | n_cells | n_donors | flags |
|---|---|---|---|---|---|---|---|
| CL:0019001 | tracheobronchial serous cell | lung | 4.82 | 0.91 | 1204 | 8 | — |
| CL:0002325 | pancreatic ductal cell | pancreas | 3.94 | 0.86 | 2871 | 6 | — |
| CL:0009043 | intestinal crypt stem cell | small intestine | 3.11 | 0.74 | 940 | 5 | — |
| CL:0002633 | respiratory basal cell | lung | 2.05 | 0.48 | 3412 | 9 | INTERMEDIATE |
| CL:0000082 | lung epithelial cell | lung | 1.88 | 0.41 | 5190 | 11 | COARSE_ANNOTATION, PARENT_OF_LEAF |
| CL:0002062 | pulmonary alveolar type 1 cell | lung | 0.09 | 0.03 | 4455 | 10 | — |
| CL:0000182 | hepatocyte | liver | 0.02 | 0.01 | 8003 | 7 | — |
| CL:0002071 | enterocyte of colon | large intestine | 0.21 | 0.06 | 3344 | 6 | — |
| CL:0000236 | B cell | multiple | 0.01 | 0.00 | 12908 | 14 | — |
| CL:0005019 | pancreatic epsilon cell | pancreas | 1.97 | 0.44 | 31 | 1 | LOW_SUPPORT, SINGLE_DONOR |

**Summary block appended by the step:**

```json
{
  "distribution_summary": {
    "n_cell_types_above_zero": 61,
    "n_cell_types_pct_expressing_gt_0.5": 3,
    "max_mean_expr": 4.82,
    "median_mean_expr": 0.07,
    "expression_breadth": 0.017
  },
  "early_exit_check": {
    "gene_absent": false,
    "all_zero": false,
    "ubiquitous": false,
    "verdict": "PROCEED"
  },
  "excluded_from_downstream": [
    {"cl_id": "CL:0005019", "reason": "n_cells < 50; single donor"}
  ]
}
```

Note that obviously irrelevant cell types (B cells, erythroblasts) are deliberately retained: Step 3 needs the full landscape to select matched negatives, and the full distribution defines what "high" expression means.

### 2.3 What is important

1. **Return statistics, not a binary call.** This step reports *how much* and *in how many cells*. It does not decide "expressed / not expressed" — that belongs to Step 3, where the threshold is explicit, tunable and logged. Merging the two makes threshold sensitivity analysis impossible.
2. **Two distinct measures, both required.** Mean/median expression and *fraction of cells expressing* capture different things. Moderate expression in 95 % of cells differs biologically from very high expression in 5 % (subpopulation, doublet, or contamination artifact).
3. **Cell types must be returned as Cell Ontology IDs**, not free-text labels. Unmapped cell types are an explicit condition to handle — never silently dropped or fuzzy string-matched.
4. **Sample size and support tracked per cell type.** 12 cells from one donor is not comparable evidence to 8000 cells from 15 donors.
5. **Normalisation explicit and consistent.** Raw counts are not comparable across cell types (sequencing depth differs). State it, keep it consistent within a species, log it — and note it may differ **between** atlases, which is a direct threat to cross-species comparability.
6. **Provenance recorded** — atlas version, access date, preprocessing. Atlases are re-released and re-annotated.

### 2.4 Edge cases

**Data availability**
- Gene absent from the atlas reference annotation entirely (common for lncRNAs, newly annotated genes, or genes under a different name in the atlas's reference)
- Gene present but zero everywhere — biologically real, or technical dropout; indistinguishable from this data alone
- Gene expressed in essentially **all** cell types (housekeeping) → no contrast possible; **detect here, flag, and short-circuit** rather than producing a meaningless hypothesis downstream

**Cell-type annotation**
- Atlas labels with no CL mapping, or mapped only to a very general parent ("epithelial cell") too coarse to inform
- The same conceptual cell type under multiple labels/IDs within one atlas (annotation inconsistency across tissues or submitting labs)
- **Hierarchically nested annotations** — both "T cell" and "CD8+ T cell" present as separate entries; parent and child are not independent observations and would double-count
- Cell types present in one species' atlas with no counterpart in the other (does not break Step 2, but flag for Stage 2 interpretation)

**Statistical / technical**
- Very low cell counts → unreliable estimate
- Single-donor cell types → cannot separate cell-type biology from individual idiosyncrasy
- **Dropout** — single-cell data has high rates of false zeros; low readings are systematically less trustworthy than high ones. This matters disproportionately because the negative set is *defined by absence*
- Batch effects between tissues/datasets within one atlas
- Multiple atlases available for the same species with disagreeing values → which wins, or is agreement required?

**Scope decisions to make now**
- Are developmental/non-adult cell types in scope? (Expression shifts substantially across development)
- Are disease-state cell types in scope, or healthy tissue only? (Expression in tumour cells is a different question from normal function)
- Sex-specific or tissue-restricted cell types present in one species but not the other

### 2.5 Additional categories to specify

- **Caching layer** — keyed on (gene, species, atlas version); cache invalidation tied to version. Atlas queries repeat constantly across a benchmark batch.
- **Minimum support thresholds** — explicit, configurable rules for excluding low-cell/low-donor cell types, applied and logged consistently.
- **Hierarchy handling policy** — a stated rule (leaf-level only, or collapse to a fixed CL depth) so parent/child double-counting is handled uniformly.
- **Early-exit conditions** — formal criteria (`gene_absent`, `all_zero`, `ubiquitous`) that terminate cleanly with an informative status.
- **Cross-atlas comparability audit** — document where the two atlases differ in normalisation, annotation granularity and tissue coverage. These differences confound the cross-species convergence signal and must be recorded, not invisible.

### 2.6 Initial implementation ideas

- **cellxgene Census** as the primary interface — programmatic access across many datasets with CL-standardised annotations, handling expression pull and ontology mapping together.
- Pull **pseudobulk summaries per cell type** rather than working at single-cell resolution — the output is inherently cell-type-level, and aggregating early reduces both dropout noise and data volume.
- Consider **multi-donor support** as a default minimum-support rule; donor reproducibility is the closest thing to a built-in replication check at this stage.
- Store output as a versioned artifact per (gene, species, atlas version) so Step 3 threshold sweeps rerun cheaply without re-querying.

---

## Step 3 — Positive / Negative Set Construction

**This is the step where a bad decision quietly poisons everything downstream, and it is the least standardised part of the pipeline.**

> **STATUS: UNCERTAIN.** Unlike most other steps, there is no clear best-practice mechanism here yet — §3.3's four matching strategies are candidates, not a settled choice, and §L1/§L12 note that different strategies produce different hypotheses with no ground truth to adjudicate between them. Treat everything in this section as a starting hypothesis about *how* to do matching, not a specified procedure. This is the step most likely to change shape as we learn more, and the one most worth revisiting before investing in a fixed implementation.

### 3.1 What it should do

Take Step 2's full table and produce two disjoint lists of CL IDs, plus a third bucket:

- **Positive set** — cell types where the gene is reliably expressed
- **Negative set** — cell types where the gene is absent/low **and which are otherwise comparable to the positive set**
- **Excluded** — intermediate expression, insufficient support, or structurally problematic, each with a logged reason

Output is IDs and rationale only. No ontology text yet.

### 3.2 What is important

1. **Two separate decisions, not one.** *Thresholding* (which cell types express the gene) and *matching* (which non-expressing cell types are fair comparators) are logically distinct. The first is mechanical; the second is where scientific validity lives. Implement them separately.

2. **The matching criterion determines what the LLM can possibly find.** If the positive set is all lung/pancreas/gut epithelium and the negative set is B cells and erythroblasts, the LLM will correctly answer *"the positive cells are epithelial"* — true, trivial, and unrelated to the gene. **Bad negatives produce true-but-useless answers, which are harder to detect than false ones.**

3. **Negatives must break collinearity, not merely be "similar."** See §L1 — cell-type properties are heavily confounded with one another. If every positive is *secretory* **and** *endoderm-derived*, the LLM can only distinguish those if the negative set contains endoderm-derived non-secretory cells. **Design requirement: negatives should be chosen to break collinearity between candidate explanatory properties.** This is stronger and more specific than "ontology sibling" or "same tissue."

4. **An explicit intermediate zone is required.** Cell types near the threshold (e.g. respiratory basal cell at 0.48) go to *excluded*, never force-assigned. Forcing borderline cases negative creates leakage; forcing them positive dilutes signal.

5. **Set sizes need floors and ceilings.** Too few positives (n = 1, 2) makes any shared property unfalsifiable — one cell type always has properties. Too many overflows LLM context and dilutes contrast. Both bounds configurable and enforced, with clean early exit if the floor cannot be met.

6. **Balance between sets.** A 3-vs-80 contrast is a different task from 3-vs-3. Negative sampling strategy (how many, chosen how) must be explicit.

7. **Determinism.** Any sampling must be seeded and logged. Otherwise re-running the same query yields different hypotheses, and the cross-species convergence score partly measures sampling noise.

### 3.3 Matching strategies

Support more than one; treat the choice as an experimental variable.

| Strategy | Mechanism | Strengths | Weaknesses |
|---|---|---|---|
| **Ontology-sibling** | Walk up the CL `is_a` hierarchy from each positive; take non-expressing siblings | Guarantees structural comparability; uses data already at hand | Brittle where CL hierarchy depth is uneven |
| **Tissue/organ matching** | Negatives from the same tissue as positives | Simple; controls anatomical context | May yield same-tissue negatives with very different function |
| **Transcriptome-similarity** | Nearest non-expressing cell types by overall expression profile, excluding the query gene | Most rigorous "similar in everything except this gene" | Extra dependency; harder to justify simply |
| **Stratified random** | Sample across the full landscape | Useful **baseline** — should produce vaguer properties, confirming matching does something | Not a serious primary strategy |

**Agreement across matching strategies is an additional robustness signal**, parallel to cross-species convergence.

The **transcriptome-similarity** strategy is a candidate application for repurposing a pretrained transcriptome–text embedding space (e.g. CellWhisperer's), which would find well-matched negatives without building the embedding from scratch.

### 3.4 Edge cases

**Threshold-related**
- No cell type clears the positive threshold → early exit; distinguish "not expressed anywhere measurable" from "threshold too strict"
- Only 1–2 positives clear → below minimum viable set size
- Positives cluster in a single tissue → the property found may be anatomical rather than functional; flag in output
- Bimodal/continuous expression with no natural gap → threshold choice is arbitrary; **sensitivity analysis is essential specifically here**

**Matching-related**
- No valid matched negatives exist (gene expressed in all members of the sibling group/tissue)
- Positive set spans several unrelated tissues → "matched" is ill-defined; may require per-positive matching rather than set-level
- CL hierarchy yields a sibling group that is too small, or a parent that is uselessly general
- Chosen negatives share a confound with each other (all one tissue, all one donor cohort) that the LLM will latch onto instead of the real signal

**Structural (inherited from Step 2 flags)**
- Parent/child overlap — must never appear on opposite sides of the contrast, and must not double-count on the same side
- `LOW_SUPPORT` / `SINGLE_DONOR` → exclude by default; make configurable
- A cell type appearing in multiple tissues with divergent expression — one entity or two?

**Mode-specific**
- `benchmark`: nothing in set construction may use ground-truth function. Selecting negatives *because* they lack the known function is leakage upstream of the LLM.
- `negative_control`: this is where contamination is injected, via a **deliberate, parameterised** mechanism (25/50/75 % mismatched swap-in), not ad hoc.

### 3.5 Additional categories to specify

- **Sensitivity analysis as a first-class output** — run the step at several thresholds, record how membership changes, report stability.
- **Confound audit** — automated check for systematic differences between the sets other than the gene: tissue distribution, cell count, donor overlap, annotation granularity. Surface as warnings attached to the sets.
- **Set-construction provenance** — strategy, parameters, seed, exclusions with reasons; carried forward so a surprising hypothesis can be traced back to a selection artifact.
- **Symmetry / reversed-contrast check** — construct the reversed contrast (negatives treated as positives) and confirm the pipeline does not produce an equally confident property. If it does, matching is not isolating the gene's signal.
- **Human override hook** — accept a curated positive/negative set from Step 1's file.
- **Collinearity diagnostic** — for a retained hypothesis, check which *other* properties the positive set shares at similar coverage (see §L1.3).

---

## Step 4 — Cell Ontology Description Retrieval

### 4.1 What it should do

Convert bare CL identifiers into the text the LLM reasons over. Because CL links out rather than storing everything natively, this requires **joining at least three ontologies**:

- **CL** — identity, definition, `is_a` hierarchy, `develops_from`, markers
- **Uberon** — anatomical location (`part_of`)
- **GO** — biological processes and cellular components (`capable_of`)

Output: one uniformly structured description block per cell type, plus a record of what was and was not retrievable.

### 4.2 Reference: what a CL term contains

| Field | Content | Example (AT2 cell) |
|---|---|---|
| **Identifier** | `CL:nnnnnnn`, stable join key | `CL:0002063` |
| **Textual definition** | Curated human-readable sentence | Located in alveolar epithelium; columnar morphology; produces and secretes surfactant |
| **`is_a`** | Parent/child subsumption; **implies inheritance** — children inherit parent properties | chondrocyte `is_a` mesenchyme cell |
| **`develops_from`** | Developmental lineage; **no** inheritance implication | hepatocyte `develops_from` mesenchymal cell |
| **`part_of`** | Anatomical location, via an Uberon term | alveolus (`UBERON:...`) |
| **`capable_of`** | Function, via GO biological process terms | surfactant homeostasis (`GO:...`) |
| **Markers** | Cell-surface markers, chiefly via `has_plasma_membrane_part`, with high/low-expression variants; only markers necessary to define the type | CD-marker sets, esp. immune cells |
| **Neuron-specific relations** | Which tracts/nerves a neuron's projections fasciculate with | Betz cell ↔ corticospinal tract |

### 4.3 What is important

1. **Uniform structure across both sets.** If positive descriptions are rich and negative ones sparse, the LLM may latch onto **description richness itself** as the discriminating signal — a pure artifact. Field ordering and formatting identical on both sides; asymmetry in available detail measured and flagged, never ignored.

2. **Relation traversal depth is a real, sweepable parameter.** `is_a` implies inheritance, so decide whether to include only directly-asserted relations or also inherited ancestor properties. Too little and shared properties stay invisible; too much and every cell type dissolves into the same generic ancestor text ("epithelial cell", "native cell"). Configurable, not hardcoded.

3. **Resolve linked IDs to labels.** `GO:0070254` means nothing to an LLM reasoning in natural language; "mucus secretion" does. Substitute the label, retain the ID alongside for traceability and Step 6 symbolic scoring.

4. **Blinding is enforced here, not just in the prompt.** This is the last point where gene-identifying information can leak. CL definitions and linked GO terms sometimes name genes or gene products, and disease-associated cell types can give the answer away. Apply the filter to the **description corpus itself**.

5. **Version pinning** for CL, Uberon and GO — all three independently versioned and frequently updated. Cross-species comparability requires both branches use the **same** releases.

6. **Description assembly format is a design choice with consequences.** Structured triples (`is_a: X; part_of: Y; capable_of: Z`) are precise and machine-checkable. Prose paragraphs read more naturally to an LLM — and GeneTEA's finding that coherent *descriptions* outperform fragmented annotations argues for prose. Support both; treat format as an ablation variable.

### 4.4 Edge cases

**Missing or thin data**
- CL term with no textual definition, or only a stub
- No `capable_of` links at all — **functional annotation is far less complete than structural annotation across CL**; many cell types have location and lineage but no process links
- No Uberon `part_of` link (common for circulating/non-tissue-resident cell types)
- Very general term with almost no distinguishing content
- **Systematic asymmetry**: positives well-annotated (because well-studied) and negatives sparse, or vice versa. Annotation richness correlates with study attention, which correlates with the very biology being inferred — a genuine confound, not just noise

**Ontology structure**
- Deprecated/obsolete CL terms with `replaced_by` pointers to follow
- Terms with multiple parents (CL permits multiple inheritance) — which ancestry path is traversed?
- Cross-references to obsolete GO/Uberon terms
- **Species-specific coverage differences** — human cell types are generally better annotated than mouse, and far better than non-model species; a structural threat to cross-species comparability
- Circular or unexpectedly deep hierarchies during ancestor traversal

**Content risks**
- Definition text naming the query gene or its aliases → blinding failure
- Definition text naming a disease strongly associated with the gene (e.g. a cell type defined partly by cystic fibrosis pathology gives away CFTR)
- Marker lists in `has_plasma_membrane_part` containing the query gene itself → **circular evidence; must be stripped**
- Text length varying wildly across cell types, creating implicit weighting in the prompt

**Practical**
- Ontology file size and load time (full CL + Uberon + GO is large) → caching required
- API rate limits if using live services rather than local files

### 4.5 Additional categories to specify

- **Annotation-richness audit** — quantify completeness per cell type (fields populated, text length, GO link count); report the positive-vs-negative distribution as an explicit confound warning.
- **Blinding verification** — automated post-retrieval scan of assembled blocks for gene symbol, aliases, Ensembl ID and associated disease terms; **fail the run** on a hit in benchmark mode.
- **Field-inclusion configuration** — which relation types are included (`is_a`, `part_of`, `capable_of`, markers, `develops_from`, neuron-specific) as a config parameter. This enables ablation of *which description axes* drive the LLM's conclusions — a genuinely interesting result in itself.
- **Fallback policy for sparse terms** — climb to parent and inherit, mark as low-information, or exclude entirely. If excluded, set composition changes, so it must feed back to Step 3.
- **Description caching** — keyed on (CL ID, ontology versions, field config); common cell types recur across hundreds of queries in a batch.
- **Local pinned OBO/OWL files over live API** — better for version control, reproducibility, speed, and removes a network dependency from the hot path.

---

## Step 4b — Optional Preprocessing (pre-LLM-Contrast)

**Status: optional / not yet specified in detail.** An optional transformation stage between the assembled description blocks (Step 4 output) and the contrast call (Step 5), for whatever cleanup or compression turns out to be needed once real description blocks are seen — not a required part of the pipeline as currently understood.

### 4b.1 Motivation

Step 4's output is raw, uniformly structured description text per cell type. Depending on what real ontology text looks like at scale, it may be useful to transform it before it reaches Step 5's contrast prompt — e.g.:

- **Length normalisation / compression** — collapsing verbose or repetitive descriptions so text length does not implicitly weight the prompt (§4.4 already flags text-length variance as a risk at the retrieval stage; this would be a place to actively correct it rather than only flag it).
- **De-duplication of near-identical parent/child text** introduced by relation-traversal depth (§4.3.2), if inheritance produces heavily overlapping descriptions across sibling cell types.
- **Noise/boilerplate stripping** beyond what blinding removes (§4.3.4) — generic ontology phrasing that carries no discriminating signal for any gene.
- **Format conversion** if the structured-triples-vs-prose ablation (§4.3.6, Appendix C item 1) settles on a transformation rather than a retrieval-time choice.

### 4b.2 What is important

1. **Optional and skippable.** The pipeline must run correctly with this step as a no-op passthrough; nothing downstream may assume it ran.
2. **If used, it is part of the tracked configuration**, not a silent transformation — logged the same way as any other Step 4/5 parameter (prompt version, field-inclusion config), since it changes what the LLM actually sees.
3. **Must not leak or re-introduce blinding failures.** Any preprocessing that summarises or rewrites text (e.g. via an LLM call) reopens the blinding-verification requirement (§4.3.4) — the scan must run on the *preprocessed* text, not just the raw retrieval.
4. **Must preserve the uniform-structure requirement (§4.3.1) across positive and negative sets** — asymmetric preprocessing (e.g. compressing one side more than the other) would reintroduce exactly the artifact §4.3.1 exists to prevent.

### 4b.3 Open questions

- Is this a rule-based transformation or itself an LLM call? An LLM-based preprocessing step adds another place for hallucination and another blinding checkpoint.
- Should it be ablated like other format choices (§5.5 prompt ablations), i.e. is "preprocessed vs. raw" itself a planned experiment rather than a settled pipeline default?
- Interaction with caching (§4.5 description caching, in Step 4) — does the cache key need to include the preprocessing config?

This section will be filled in once there is a concrete need observed from real description blocks, rather than speculated in advance.

---

## Step 5 — LLM Contrast

The reasoning core, and the step with the most ways to fail silently: a confident, fluent, plausible wrong answer looks identical to a right one at this stage.

### 5.1 What it should do

Take the positive and negative description blocks and produce a **candidate hypothesis**: the most specific biological property enriched among positives and depleted among negatives, in open-ended natural language.

Structured, machine-parseable output:

```json
{
  "property": "epithelial cells specialized in active transmembrane
               ion and fluid secretion across exocrine/mucosal surfaces",
  "confidence": 0.85,
  "rationale": "positive set shares secretion-linked GO terms (mucus,
                bicarbonate, fluid transport); negative set is epithelial
                but specializes in absorption, gas exchange, or metabolism",
  "abstained": false
}
```

**This step proposes only. It does not verify.** Verification is Step 6, deliberately separated so the model does not grade its own homework.

### 5.2 Prompt architecture: joint vs. split

**Default: joint contrast — both sets in one prompt, one call.**

The target is a property that *separates* the sets. In a split architecture (summarise positives alone → summarise negatives alone → compare summaries), each summarisation call has no idea what it is being contrasted against, so it compresses along whatever axis is most salient in isolation — which may not be the discriminating axis. If both sets summarise to "epithelial cells of the respiratory and digestive tracts," the contrast signal has been destroyed before comparison, even though a real discriminator (secretory vs. absorptive) exists in the data.

**Split is still worth building as a control arm**, because it is harder to game: the positive-summarisation call sees only cell types, with no indication a contrast is coming, making shortcut gene-recognition less likely to drive output. It also yields interpretable intermediates.

**Specified experiment:** run both on the benchmark set.
- Split recovers substantially *less* known biology → the contrast framing is doing real work.
- Joint recovers substantially *more* on well-known genes but converges with split on obscure ones → joint is benefiting from memorisation on the famous ones. This is otherwise unmeasurable.

Note the contrast with Stage 2: within a species, positives and negatives are two halves of one question, not independent evidence sources. Cross-species independence is a different thing entirely and *is* strictly enforced.

### 5.3 What is important

1. **Abstention must be a first-class output.** This is the single most important design point in the step. Hu et al. found GPT-4 gave zero confidence and refused to name 87 % of fully random gene sets, while other models (GPT-3.5, Gemini Pro, Mixtral, Llama2-70b) were falsely confident. **The capability varies enormously by model and cannot be assumed.** The entire validity of `negative_control` mode depends on it. Design the prompt so abstention is explicitly sanctioned with a fixed response form (Hu et al. returned a fixed string rather than a name for zero-confidence cases), and treat **abstention rate on controls as a primary model-selection criterion**.
2. **Specificity must be actively demanded.** The failure mode is not wrong answers but *vague-but-true* ones — "these are epithelial cells" is unfalsifiable and useless. Ask for the **most specific** property that still covers the positive set. Output must be specific enough for Step 6 to test per-cell-type; a property that cannot be checked per-cell-type is malformed output.
3. **Blinding enforced upstream, asserted here.** The prompt never names the gene — only "the query gene."
4. **Determinism and reproducibility.** Log temperature, seed, model version and full prompt text. Model versions change underneath you.
5. **Self-consistency as a cheap robustness signal.** Run the same contrast *n* times; measure agreement. If the same input yields three different properties, the hypothesis is noise regardless of per-run confidence.
6. **Confidence calibration is testable.** Hu et al. found higher self-reported confidence predicted higher similarity to truth — for one model on a different task. Whether it holds here is empirical. If it does, confidence becomes a useful filter before the expensive cross-species step.
7. **Symmetry of presentation.** Format both blocks identically; vary set order across runs, or at least test for order sensitivity. "The first-listed group is always positive" is a learnable artifact.

### 5.4 Edge cases

**Output quality**
- Property too vague to test per-cell-type ("involved in cellular processes")
- **Compound properties** joined by "and" — one hypothesis or several? Affects Step 6 scoring
- Model restates the input rather than abstracting ("these cells are found in lung, pancreas, intestine") — a description, not a property
- Output not parseable into the required schema
- Property true of positives but rationale cites the wrong cell types

**Model behaviour**
- **False confidence on negative controls** (the Hu et al. failure mode) — must be measured per model, never assumed
- **Model recognises the gene despite blinding**, from a distinctive cell-type combination. CFTR's positive set is effectively a fingerprint to anyone who knows the biology — and the model does
- Refusal or hedging for reasons unrelated to the biology
- Context length exceeded on large sets or verbose descriptions
- Model latches onto a set-construction artifact (all positives one tissue, all negatives sparsely annotated) rather than biology

**Input-related**
- One side much richer in annotation than the other
- Very small sets (3 vs 3) → any property trivially findable
- Very large sets → contrast dilutes, context overflows
- Positive set genuinely heterogeneous with no shared property → correct answer is abstention, hard to distinguish from model failure

**Mode-specific**
- `benchmark`: memorisation may drive output rather than reasoning; inflates apparent performance in a way that can be bounded but not eliminated (see §L2)
- `negative_control`: success is *low* confidence; metrics must be inverted, not reused

### 5.5 Additional categories to specify

- **Prompt versioning** — prompts are code. Version them, store exact text per run, treat changes as invalidating prior results.
- **Model-agnostic interface** — do not couple to one provider. Hu et al.'s central finding was that models differ enormously on exactly the capability the pipeline depends on. Given existing Ollama/llama3.3 infrastructure, **whether a local open model can abstain reliably is a genuine open question for this setup specifically** and must be tested, not assumed.
- **Structured output enforcement** — schema-constrained generation or JSON mode, with a parse-failure retry policy and a retry cap.
- **Prompt ablations as planned experiments** — chain-of-thought vs. direct; structured triples vs. prose; with/without explicit specificity instruction; positive-first vs. negative-first. These are results about what the method needs, not tuning.
- **Cost and latency budgeting** — genes × thresholds × matching strategies × self-consistency runs × species multiplies fast. Estimate early; it constrains feasible sweeping.
- **Reversed-contrast control at this step** — feed sets swapped; confirm no equally confident property emerges. Cheap, catches a whole class of set-construction problems.

### 5.6 LLM backend configuration

Reused from the existing SMTB2025/DendoLLM infrastructure rather than built new, so all pipeline LLM calls (Step 5 contrast, Step 6 scoring, Step 8 comparison) share one client layer.

**Access:** Ollama-compatible endpoint behind an Open WebUI proxy (`dev.chat.cosy.bio`), reached via `langchain_ollama` (`ChatOllama` / `OllamaEmbeddings`). Auth is a bearer JWT from the Open WebUI account, not an Ollama-native key.

**Configuration surface (env-driven, `.env` at project root, mirroring SMTB2025's `core/llm_backend.py`):**

| Variable | Purpose | Default |
|---|---|---|
| `OLLAMA_BASE_URL` | Endpoint base URL | `https://dev.chat.cosy.bio/ollama` |
| `OLLAMA_API_KEY` | Bearer token for the proxy | none — required, raises if missing |
| `OLLAMA_CHAT_MODEL` | Chat/generation model | none fixed — **model selection is an open question, not pinned in this document** |
| `OLLAMA_EMBED_MODEL` | Embedding model (needed for Step 3 transcriptome-similarity matching and Step 8 embedding-based comparison) | none — required, raises if missing |
| `EMBED_BACKEND` | `ollama` (default) or `openai`, for swapping the embedding provider independent of chat | `ollama` |

**Design points carried over from SMTB2025 and applicable here:**
- All model/endpoint config lives in one module (`llm_backend.py`-equivalent); nothing downstream hardcodes a model name — satisfies the model-agnostic-interface requirement above (§5.5) at the config level, not just the interface level.
- `num_ctx` / `num_predict` capped explicitly per call (VRAM and proxy-timeout pressure); some model families (e.g. `qwen3.x`) require `reasoning=False` to stop consuming the entire output budget on hidden thinking tokens before emitting JSON — a class of failure worth checking for whichever model is eventually chosen, since it looks like a parse failure but is actually a budget problem.
- Structured JSON output (§5.5) is enforced at the prompt/parsing layer, not by the backend itself — the backend module has no opinion on output schema.

**What is explicitly NOT decided here:** which chat model to use for generation (Step 5), scoring (Step 6), or comparison (Step 8). Appendix C item 4 already flags this as open (local open model vs. frontier model, and whether abstention reliability differs by step). Different steps may reasonably use different models — e.g. a cheaper/faster model for the high-volume per-cell-type scoring in Step 6 versus a stronger model for the one-shot contrast in Step 5 — and the config surface above supports that (override per call), but no such split is specified yet.

---

---

## Step 6 — Hypothesis Scoring (Coverage & Leakage)

The verification layer — what separates the pipeline from "an LLM said something plausible." Step 5's output is a *claim*; a claim must be tested against the evidence independently of whoever made it.

### 6.1 What it should do

Test the proposed property against **every cell type in both sets individually**, producing:

- **Coverage** — fraction of positive-set cell types the property genuinely applies to (does the hypothesis explain the positives?)
- **Leakage** — fraction of negative-set cell types the property also applies to (does it wrongly explain the negatives?)
- **Specificity** — how narrow the property is
- **Verdict** — retain/discard against configured thresholds

Output: per-cell-type judgments with reasoning, aggregate scores, and a decision, all traceable to individual calls.

### 6.2 What is important

1. **Independence from generation is the entire point.** The scoring call must not see Step 5's rationale, must not know which set a cell type came from, and ideally uses a separate model instance or a different model. If the scorer knows "this one is a positive," it will find a way to agree. Blinded, per-cell-type, one at a time.
2. **Per-cell-type, never set-level.** Ask "does property P apply to cell type C?" for each C independently. Set-level questions invite holistic hand-waving; per-cell-type questions are falsifiable and reveal *which* members drove a score.
3. **Coverage and leakage are asymmetric in importance.** High leakage is fatal — it means the property does not discriminate, which is the one thing it claims to do. Moderate coverage failure is more forgivable: a property explaining 4 of 6 positives may be a real finding about a subset. Threshold them separately; do not collapse to a single F-score by default.
4. **Specificity needs its own metric.** "These are cells" achieves perfect coverage and zero leakage against a well-chosen negative set and is worthless. Hu et al. handled the analogous problem with a Jaccard index between the query gene set and all genes associated with the proposed name. **Analog here: how many cell types in the whole atlas (not just the two sets) does the property apply to?** 3 of 174 is specific; 120 of 174 is not, regardless of coverage/leakage.
5. **The scorer is itself an LLM and therefore fallible.** This is not formal verification — it is a second opinion under better conditions (blinded, atomised, no stake in the answer). Where possible supplement with **non-LLM symbolic checks**: if the property maps to a GO term, do the positive cell types' actual `capable_of` links contain that term or its descendants? That check depends on no model's judgement.
6. **Graded judgments beat binary ones.** Yes/no discards information. Use a 0–1 applicability score or three levels (clearly / partially / not) — mirroring the contamination-gradient logic, where the interesting behaviour is in the middle.

### 6.3 Edge cases

**Property-related**
- Compound properties ("secretory *and* ciliated") — score jointly, or decompose and score each conjunct? Decomposition is more informative but changes what is tested
- Property so vague every cell type passes (coverage 100 %, leakage 100 %) — should be caught by specificity, but needs an explicit rule
- Property so specific only one positive passes
- Property references something absent from the CL descriptions entirely (model invoked outside knowledge) — hallucination or legitimate inference? Both occur and are hard to distinguish
- Property restates set membership rather than making a biological claim

**Scoring mechanics**
- Scorer and generator agree suspiciously often → check that independence is actually enforced (same model, context leaking through)
- Scorer disagrees with itself on repeat calls → needs its own self-consistency measurement
- A sparsely annotated cell type cannot be judged either way → **"insufficient information" must be a distinct outcome from "does not apply"**, excluded from numerator and denominator rather than counted negative
- Ties/borderline cases clustering at the threshold

**Structural**
- Small sets make scores coarse — with 3 positives, coverage moves in 33 % increments and confidence intervals are enormous. Report uncertainty; do not present 0.67 as precise
- The excluded/intermediate cell types from Step 3 — arguably **should** be scored as an extra signal: if the property applies to intermediate-expression cell types, that is supporting evidence; if not, a warning
- Full-atlas specificity scoring is expensive — 174 calls per hypothesis × every gene in a batch

**Mode-specific**
- `benchmark`: **scoring is not the same as ground-truth comparison.** A hypothesis can score perfectly on coverage/leakage and still be a different property from the known function. Compute and report both separately — internal validity and external accuracy are different questions
- `negative_control`: if Step 5 abstained, there is nothing to score and the run terminates successfully. If Step 5 **did not** abstain on a control, scoring becomes the diagnostic: does the spurious property at least fail coverage/leakage, or does it pass? **The second case — the scoring layer fooled by the same artifacts that fooled the generator — is the most important failure to detect in the entire pipeline.**

### 6.4 Additional categories to specify

- **Scorer-model selection and validation** — hand-label a set of (property, cell type) pairs and measure scorer accuracy against human judgement before trusting it. An unvalidated scorer is just a second guess.
- **Threshold configuration and justification** — retain/discard cutoffs for coverage, leakage and specificity as config parameters, swept in sensitivity analysis, justified against benchmark performance rather than intuition.
- **Symbolic verification layer** — where the property maps to GO/Uberon terms, check cell types' actual annotations directly. Report symbolic and LLM-based scores separately; agreement between them is itself a quality signal.
- **GO enrichment as a standard verification method — to discuss further.** Alongside the per-cell-type LLM/symbolic checks above, run a classical GO-term enrichment analysis (e.g. hypergeometric/Fisher test, the standard bioinformatics tool for exactly this question) over the positive set versus the negative/background set, independent of the LLM entirely. This gives a non-LLM, statistically established second line of evidence for whether the proposed property (or its closest mapped GO term) is actually enriched, using the same well-understood machinery the field already trusts for gene-set enrichment — analogous to what Step 8 proposes for cross-species comparison (Resnik/Lin GO semantic similarity, §8.5) but applied here within a single species/branch. Open questions to resolve before specifying this fully: whether enrichment is computed over GO terms attached to the *cell types* (via `capable_of`, per §4.2) or over the marker genes historically used to define those cell types (which reopens the annotation-circularity concern, §L3); how it interacts with the existing symbolic verification layer above (complementary evidence vs. redundant); and what background set is appropriate (whole atlas vs. matched negative set, mirroring the specificity-metric background choice above).
- **Background / null distribution** — score the property against randomly sampled atlas cell types to establish chance coverage. Coverage of 0.8 means something very different if random cell types score 0.1 versus 0.7. This turns raw scores into something closer to a significance measure.
- **Full audit trail** — every per-cell-type judgment with the scorer's reasoning, retained. A surprising hypothesis surviving to Stage 2 must be reconstructable.
- **Cost control for specificity scoring** — the most expensive operation in the pipeline. Consider stratified atlas sampling rather than exhaustive scoring, with the scheme specified and seeded.

---

## Step 7 — Stage 1 Output Artifact & Handoff

Small step, but it defines the contract between stages — and because Stage 2's validity rests on branch independence, this is where independence is made **auditable** rather than merely assumed.

### 7.1 What it should do

Serialise a single-species run into one self-contained, versioned artifact that:

1. **Carries the hypothesis forward** — a minimal payload for Stage 2
2. **Records full provenance** — every parameter, version, intermediate and decision
3. **Certifies independence** — evidence the branch never accessed the other species' data
4. **Terminates cleanly on failure** — early exits and abstentions are legitimate outcomes with their own schema, not error states

### 7.2 What is important

1. **Separate payload from provenance.** Stage 2 must consume a deliberately minimal object. If it can see the full artifact it can see the gene symbol, cell types and rationale — any of which could leak and let Stage 2 "agree" for reasons other than genuine convergence. **Enforce structurally**: two files, or a payload sub-object that is the only thing Stage 2 may read.
2. **Independence certification is a real requirement.** Back the claim with something checkable: a hash of the branch's complete input set, a log of every data source accessed, and an assertion that no cross-species artifact path appears. If branches run in one process, guard at code level (separate working directories, no shared mutable state).
3. **Negative and null results are first-class artifacts.** Abstention, early exit and discarded hypothesis all produce a complete artifact with a clear status — in `negative_control` mode these *are* the successful outcomes, and in `discovery` mode "no signal" must be distinguishable from "crashed."
4. **Everything Stage 2 needs to weight the comparison must be in the payload** — confidence, coverage, leakage, specificity, set sizes. Convergence between two strong hypotheses means more than between two shaky ones.
5. **Machine-readable status taxonomy**, not free text: `RETAINED`, `DISCARDED_LOW_COVERAGE`, `DISCARDED_HIGH_LEAKAGE`, `DISCARDED_LOW_SPECIFICITY`, `ABSTAINED`, `EARLY_EXIT_GENE_ABSENT`, `EARLY_EXIT_UBIQUITOUS`, `EARLY_EXIT_INSUFFICIENT_POSITIVES`, `FAILED_*`.
6. **Content-addressable identity** — a hash over the full resolved configuration, so identical runs are detectable, caches are safe, and two artifacts can be checked for comparability at all.

### 7.3 Artifact schema sketch

```json
{
  "artifact_id": "sha256:...",
  "schema_version": "1.0",
  "status": "RETAINED",

  "payload": {
    "property": "epithelial cells specialized in active transmembrane
                 ion and fluid secretion across exocrine/mucosal surfaces",
    "confidence": 0.85,
    "coverage": 1.00,
    "leakage": 0.00,
    "specificity": 0.04,
    "n_positive": 3,
    "n_negative": 3
  },

  "provenance": {
    "gene": {"symbol": "CFTR", "ensembl_id": "ENSG00000001626"},
    "species": "homo_sapiens",
    "mode": "benchmark",
    "config_hash": "sha256:...",
    "versions": {
      "atlas": "tabula-sapiens-v1", "census": "2025-01-30",
      "cl": "2025-03-15", "uberon": "2025-02-10", "go": "2025-03-01",
      "model": "...", "prompt": "contrast-v3", "pipeline": "0.4.1"
    },
    "parameters": {
      "expression_threshold": 0.25,
      "matching_strategy": "ontology_sibling",
      "seed": 42
    },
    "sets": {
      "positive": ["CL:0019001", "CL:0002325", "CL:0009043"],
      "negative": ["CL:0002062", "CL:0000182", "CL:0002071"]
    },
    "intermediates": {
      "step5_rationale": "...",
      "step6_per_celltype_judgments": [],
      "excluded_cell_types": []
    },
    "robustness": {
      "self_consistency_runs": 3,
      "self_consistency_agreement": 0.91,
      "threshold_sweep": [],
      "reversed_contrast_confidence": 0.12
    }
  },

  "independence": {
    "input_manifest_hash": "sha256:...",
    "accessed_sources": ["tabula-sapiens-v1", "cl-2025-03-15", "..."],
    "cross_species_access": false
  },

  "benchmark_only": {
    "ground_truth": "chloride channel; epithelial ion/fluid transport",
    "blinding_verified": true,
    "blinding_scan_hits": []
  }
}
```

`benchmark_only` **must reside in a section Stage 2 structurally cannot read**, or blinding fails at the last possible moment.

### 7.4 Edge cases

- **Partial completion** — Steps 1–4 succeeded, Step 5 failed. Capture what completed so batch failures are diagnosable without a rerun
- **Multiple surviving hypotheses** (n-best or multi-strategy runs) → payload becomes a list; Stage 2 needs a defined policy (best-only, all-pairs, consensus)
- **Robustness sweeps produced divergent hypotheses across thresholds** — which is *the* payload? Needs a stated rule (e.g. modal hypothesis across sweeps, with divergence recorded as a stability score)
- **Schema drift** across pipeline versions → hence `schema_version`, plus a rule on whether artifacts from different versions may be compared
- **One species `RETAINED`, the other `ABSTAINED`** → not a comparison; needs its own outcome category. Non-replication is informative but is not disagreement
- **Branches used different ontology versions** → detect and either block or flag loudly; comparability is undermined
- **Size** — per-cell-type judgments and sweep results can be large; decide whether intermediates live inline or as referenced side files

### 7.5 Additional categories to specify

- **Artifact validation on write and on read** — schema-validate before Stage 2 consumes anything, so malformed artifacts fail loudly rather than yielding a spurious comparison
- **Batch manifest** — run-level index over all artifacts (gene × species × status); this is what gets analysed for benchmark and control results
- **Retention / immutability policy** — artifacts are the experimental record; append-only, never overwritten in place
- **Comparability precondition check** — a function taking two artifacts and returning whether they are eligible for Stage 2 at all (same schema version, ontology releases, prompt version, mode)
- **Human-readable rendering** — a small report generator alongside the JSON; raw JSON is a poor debugging surface and these will be read constantly during development

### 7.6 Experiment tracking (MLflow)

The JSON artifact (§7.3) remains the source of truth — it is what independence certification, Stage 2 comparison and re-analysis all read. MLflow sits **on top of** it as an index and comparison UI; it does not replace the schema. Motivation: the parameter space (genes × species × thresholds × matching strategies × self-consistency runs × models, §L14) makes "which run had which config and how did its scores compare" a constant question during development and benchmarking, and grepping timestamped JSON directories (the pattern used in the SMTB2025 predecessor project) does not scale to it.

**Run structure — nested, mirroring the two-stage architecture (§0.2):**
- One MLflow **parent run** per pipeline invocation (one gene, one query).
- One **child run per Stage 1 branch** (per species) — logs everything in §7.3's `provenance` block.
- One **child run for Stage 2** (comparison) once both branches complete — logs §8's similarity scores, null percentile, relationship classification.

**Logged per Stage 1 run:**

| MLflow field | Source |
|---|---|
| `params`: `gene_id`, `species`, `mode`, `expression_threshold`, `matching_strategy`, `seed`, `atlas_version`, `cl_version`, `chat_model`, `prompt_version` | §7.3 `provenance.parameters` / `provenance.versions` |
| `metrics`: `confidence`, `coverage`, `leakage`, `specificity`, `n_positive`, `n_negative`, `self_consistency_agreement` | §7.3 `payload` / `provenance.robustness` |
| `tags`: `status` (`RETAINED` / `ABSTAINED` / `EARLY_EXIT_*` / `FAILED_*`, §7.2.5), `config_hash` | §7.3 `status`, `provenance.config_hash` |
| `artifacts`: the full Step 7 JSON, the human-readable render (§7.5) | Step 7 output |

**Logged per Stage 2 run:** `params` for comparator model/version and null-generation config; `metrics` for similarity score(s), z-score, empirical p-value, confidence-weighted verdict score; `tags` for relationship class (`IDENTICAL` / `ONE_SUBSUMES_OTHER` / `OVERLAPPING` / `COMPLEMENTARY` / `UNRELATED` / `CONTRADICTORY`, §8.5); the comparison report as an artifact.

**What this buys, concretely:**
- Parallel-coordinates and table views across runs for the threshold/matching-strategy/model sweeps that §3.5, §5.5 and §6.4 already call for as planned experiments — without a bespoke analysis script per sweep.
- `config_hash` as a tag makes duplicate-run detection and caching checks (§7.2.6) queryable directly, rather than requiring a separate lookup.
- Batch-level views (§8.5) — e.g. convergence rate across a benchmark set — become an MLflow search/filter over child-run tags and metrics instead of a custom aggregation pass.

**Explicitly out of scope for MLflow:** independence certification (§7.2.2) and blinding verification (§0.4) are correctness properties of the artifact itself and must hold regardless of whether tracking is attached; MLflow is a read-side convenience, and its absence or failure must never be able to affect what an artifact contains or how Stage 2 behaves.

**Deployment:** self-hosted tracking server (local file or SQLite backend store to start, matching the project's existing no-managed-service pattern); no new external dependency beyond the `mlflow` package itself.

---

## Step 8 — Cross-Species Comparison (Stage 2)

The final step, carrying the proposal's main validity claim. Everything before this produced hypotheses; this decides whether they constitute evidence.

### 8.1 What it should do

Take two (or more) Stage 1 artifacts for orthologous genes in different species, compare their payload hypotheses, and produce a convergence verdict: do independently derived hypotheses agree, and is that agreement meaningful rather than coincidental?

Output: similarity score(s), a convergence classification, and — critically — a **null-calibrated significance estimate**.

### 8.2 What is important

1. **Similarity must be semantic, not lexical.** The branches never saw each other and will not use the same words for the same concept. Hu et al. faced exactly this and used semantic similarity — a 0–1 score of closeness in meaning regardless of shared wording — to compare LLM-generated names against curated GO names. That is the direct template.

2. **A raw similarity score is meaningless without a null distribution.** *The single most important requirement in this step.* See §8.3.

3. **Comparison must be blind to gene identity and to both branches' provenance.** If an LLM comparator can see it is looking at CFTR twice, it will find agreement. The comparator sees two property strings and nothing else.

4. **Order symmetry.** `compare(A, B)` must equal `compare(B, A)`. With an LLM comparator this is not automatic — test it, and average over both orderings if it fails.

5. **Convergence is graded, not binary.** Two hypotheses can be identical; one may *subsume* the other (human "ion transport" vs. mouse "chloride secretion" — a specialisation, arguably stronger evidence than a tie); they may overlap; or contradict. Classify the **relationship**, not just the distance.

6. **Non-replication is a result, not a failure.** One retains and the other abstains, or they disagree — this may mean functional divergence, poor atlas coverage in one species, or an artifact. All three are worth distinguishing; none should be silently dropped.

7. **Confidence-weighted aggregation.** Incorporate payload confidence, coverage, leakage and specificity into the final verdict — not just the similarity score.

### 8.3 Null-calibrated significance — detailed specification

**The problem.** Suppose CFTR yields:

```
Human:  "epithelial cells specialized in active transmembrane
         ion and fluid secretion across exocrine/mucosal surfaces"
Mouse:  "epithelial cells mediating secretory ion/fluid transport
         in exocrine and mucosal glands"
Similarity: 0.91
```

0.91 looks excellent — but there is no basis for that judgement yet. The question is not "is 0.91 high on a 0–1 scale" but "is 0.91 high *for this kind of comparison*." Both strings are written by the same model family, in the same output format, constrained to the same genre (biological property statements), drawing on the same vocabulary space — and both begin with "epithelial cells" because most annotated cell types are epithelial. **That shared structure produces similarity before any biological agreement exists.**

**Building the null.** Compare **non-orthologous pairs**: human gene A's hypothesis against mouse gene B's hypothesis, where A and B are unrelated.

```
human CFTR   vs mouse Myh7    → 0.71
human CFTR   vs mouse Alb     → 0.68
human SCN9A  vs mouse Cftr    → 0.64
human INS    vs mouse Rho     → 0.73
human ALB    vs mouse Scn9a   → 0.59
... (n = 1000 mismatched pairs)

Null distribution:
  mean      0.68
  std dev   0.07
  95th pct  0.79
  99th pct  0.84
  max       0.89
```

**The crucial finding in this illustration: unrelated genes score 0.68 on average, not 0.1.** The "shared genre" effect alone accounts for most of the scale.

**Interpreting the observation against it:**

```
CFTR observed:  0.91
z-score:        (0.91 − 0.68) / 0.07 = 3.29
Percentile:     > 99.9th
Empirical p:    0/1000 null pairs ≥ 0.91  →  p < 0.001
```

**Why this changes decisions, not just presentation.** A second gene at similarity **0.74** reads as "pretty good agreement" raw. Against this null it is the ~72nd percentile, z ≈ 0.86, p ≈ 0.28 — indistinguishable from two unrelated genes. Without the null it would have been called convergence and carried forward as a spurious hypothesis.

The practical payoff: a **defensible** threshold rather than one picked by inspection. Write "we required similarity exceeding the 99th percentile of a null distribution of 1000 non-orthologous pairs (0.84)" rather than "we required similarity > 0.8."

**Design requirements for the null:**

- **Match the null to the comparison** — same model, prompt version, comparator, species pair and mode. A null built from a different prompt version does not calibrate current results.
- **Watch for a confounded null** — randomly sampled mismatched pairs will occasionally be functionally related (two ion channels, two secreted proteins), inflating the null mean and making the test conservative. Either accept that (the safe direction) or filter for functional dissimilarity — and state which.
- **Null per condition** — each threshold / matching-strategy condition needs its own null, since hypothesis *style* may differ systematically between conditions.
- **A second null worth having** — same-gene comparison with **shuffled cell-type sets** on both sides. This calibrates a different question: not "do unrelated genes agree by chance" but "does the pipeline manufacture agreement from noise." Cheap, and answers a different failure mode.
- **Null generation is a first-class pipeline mode**, run alongside every batch — not an analysis afterthought. Without it, no convergence number is interpretable.

### 8.4 Edge cases

**Comparison mechanics**
- One `RETAINED`, other `ABSTAINED` → not a comparison; distinct outcome category
- Both abstained → success in `negative_control`; no signal in `discovery`
- One or both early-exited → non-comparable, and the reason matters for interpretation
- Hypotheses agree but are both vague → high similarity, low information; payload specificity must gate this
- Different granularity (general vs. specific) — convergence or partial convergence?
- **Complementary rather than similar** hypotheses (different aspects of the same biology: "secretory" vs. "apical membrane localized") — semantically distant, biologically consistent. A hard case, likely under-detected by pure similarity scoring

**Ortholog-related**
- 1:many or many:many orthology → which paralog is *the* comparison, and does comparing several inflate false convergence via multiple testing?
- Ortholog exists but has genuinely diverged in function (neofunctionalisation) → correct answer is non-convergence, and the pipeline cannot distinguish that from methodological failure
- Ortholog assignment itself is wrong

**Cross-species confounds — the important category**
- **Atlas asymmetry** — human atlases are richer in tissues and cell types than mouse; the branches were not looking at comparable landscapes even when both succeeded
- **Annotation asymmetry** — human CL terms are better annotated, so human hypotheses may be systematically more specific
- **Non-independent inputs** — see §L-COUPLING below
- **The comparator LLM's own priors** — it knows human and mouse biology are similar and may be primed toward agreement

**Scaling**
- More than two species → pairwise (agreement matrix, more informative) or single consensus (simpler, hides which pairs drive it)?
- Multiple testing across a benchmark batch of many genes

### 8.5 Additional categories to specify

- **Comparator validation** — hand-label hypothesis pairs (clearly same / related / unrelated) and measure the comparator against human judgement before trusting it. Same requirement as the Step 6 scorer.
- **Multiple comparison methods reported side by side** — embedding cosine similarity; LLM-judged relationship classification; and where properties map to GO terms, **symbolic GO semantic similarity (Resnik/Lin)**, a well-established non-LLM measure and the strongest evidence when available. Agreement across methods is itself a robustness signal, and mirrors GeneTEA's cross-species ortholog embedding check.
- **Relationship taxonomy** rather than a bare score: `IDENTICAL`, `ONE_SUBSUMES_OTHER`, `OVERLAPPING`, `COMPLEMENTARY`, `UNRELATED`, `CONTRADICTORY`.
- **Confound audit in output** — atlas coverage asymmetry, annotation-richness asymmetry, set-size differences, attached to every convergence verdict.
- **Final report schema** — gene, per-species payloads, similarity under each method, null percentile, relationship class, confound flags, overall verdict with supporting evidence.
- **Batch-level analysis mode** — across a benchmark set, convergence rate for known genes vs. the mismatched-pair null is the headline result. Must be a supported output, not assembled manually.

---

## Limitations

Separated by whether they are **structural** (cannot be engineered away; must be stated), **boundable** (measure and report), or **practical**.

### §L-COUPLING — Cross-species branches are only partially independent

**This qualifies the pipeline's central evidential claim and must be stated explicitly in any write-up.**

Stage 2's logic is: two branches ran in isolation, they agreed, therefore the answer is real rather than an artifact of either branch. That inference is valid only if the branches can agree **only by both being right**. They can also agree by **both drawing on the same source** — and they partly do.

**Where the coupling lives.** Step 4 joins CL + Uberon + GO. **Uberon and GO are species-neutral by design.** There is no "human GO" and "mouse GO" — there is one GO. Same for Uberon: "respiratory tract" is one term used for both species. So the *functional and anatomical vocabulary* — precisely the material the LLM reasons over — is shared infrastructure between branches. CL is partially shared too: it is a single cross-species ontology, many terms are used verbatim for both species, and species-specific terms are often defined by analogy to, or harmonised against, a human counterpart.

```
HUMAN BRANCH                          MOUSE BRANCH
  CL:0019001 tracheobronchial          CL:0002325 pancreatic
    serous cell                          ductal cell
      |                                      |
  part_of  -> UBERON:0001004  <-- SAME --> UBERON:0001004
  capable_of -> GO:0070254    <-- SAME --> GO:0070254
                (mucus secretion)
```

**Concrete failure scenario.** Suppose GO curators once linked both airway secretory cells and pancreatic ductal cells to `GO:0006811 (ion transport)` and, for historical reasons, did not link alveolar type 1 cells or hepatocytes. Then:

- Human branch reads its `capable_of` links, sees "ion transport" on positives only, proposes *ion transport*.
- Mouse branch reads its `capable_of` links — **the same GO terms** — sees the same pattern, proposes *ion transport*.
- Stage 2: similarity 0.94. Convergent. Retained as conserved biology.

Nothing was independently confirmed. Both branches rediscovered a **single curation decision, made once, by one set of curators, in one database**. The convergence measured GO's internal consistency, not biological conservation. **This failure produces exactly the signature being looked for** — high similarity, high confidence, clean coverage/leakage in both branches. It does not look like a bug.

**What does stay independent.** The coupling is in the *annotation vocabulary*, not in the *evidence that selects which annotations appear*:
- **Expression data** — Tabula Sapiens and Tabula Muris are separate experiments, animals/donors, labs, sequencing runs. Which cell types land in the positive set is independently determined.
- **Cell-type composition** — the atlases cover different tissues at different depths.
- **Which ontology terms get invoked** — even sharing one GO, the branches pull different subsets, because different cell types are in play.

**Mitigations:**

1. **Quantify the overlap.** Per converged gene, compute how many GO/Uberon terms appear in *both* branches' description sets. Convergence built on 90 % shared terms is much weaker than one built on 20 %. Cheap, and directly answers the objection.
2. **Add a species-specific evidence axis.** Bring in something genuinely unshared — species-specific phenotype ontologies (HP for human, MP for mouse) or species-specific literature annotation. Convergence surviving removal of shared vocabulary is much stronger.
3. **Ablate the shared layer.** Run with GO links stripped, using only CL definitions and Uberon. If convergence holds, GO was not doing the work; if it collapses, the signal is mostly GO-mediated — worth knowing either way.
4. **Extend species distance.** Human/mouse is the worst case, being the most harmonised. A more distant species (zebrafish, fly) with independently developed annotation practice makes convergence harder to achieve by shared curation and more meaningful when achieved.
5. **Check the null for the same effect.** The mismatched-pair null partially absorbs this — if shared vocabulary inflates similarity generally, unrelated pairs also score high, raising the floor. Whether the coupling appears as a shifted null mean or as gene-specific inflation is empirically testable.

**Honest framing for the write-up:** convergence shows that **the same property is derivable from two independently measured expression datasets, using a partially shared annotation vocabulary**. The expression evidence is independent; the descriptive language is not. This still rules out atlas-specific artifacts, cell-type-composition quirks, and single-dataset noise. It does not rule out ontology-level curation artifacts. Saying so directly is stronger than having a reviewer say it.

---

### Structural limitations

#### §L1 — Expression location ≠ function (confounding among cell-type properties)

The method rests on "genes are expressed where they are needed." That is a heuristic, not a law.

**Note on framing:** the limitation is *not* that cell types are described only by location. CL descriptions carry function (`capable_of` → GO), morphology, lineage and markers — a genuinely rich functional characterisation. The problem is the **inferential step itself**:

```
gene G is expressed in cell types that do X
        |   <-- this arrow is the assumption
gene G is involved in X
```

Enriching the description of X — from "in the lung" to "performs regulated mucus secretion via apical vesicle fusion" — makes the *conclusion sharper*. It does not make the arrow more valid. You get a more specific claim resting on the same assumption.

**Why the arrow fails even with perfect functional descriptions.** A cell type that performs mucus secretion is *also* derived from a particular lineage, running a particular transcription-factor program, in a particular chromatin state, exposed to particular signalling, shaped by a particular selection history. **Every one of those correlates with "performs mucus secretion" and each is a sufficient explanation for why G is on there.**

Three ways to get a confident, convergent, wrong answer:

- **Co-regulation.** G sits in a locus controlled by an enhancer that also drives genuine secretory genes. G is expressed in every secretory cell type, faithfully, across species — because it shares regulatory real estate, not function. The expression pattern is real and conserved; the functional inference is wrong.
- **Lineage inheritance.** G's expression tracks a developmental lineage that happens to produce secretory cells. "Secretory" and "endoderm-derived" are near-collinear in the positive set, and matched negatives probably do not break the tie.
- **Housekeeping-with-a-bias.** G does something general (e.g. membrane trafficking) that secretory cells simply need more of. Enrichment is quantitative, driven by cellular demand; the specific functional label over-reads a general role.

In all three, richer descriptions make the method *more* confident, because the positives really do share the described property. **Description quality is not the failure point.**

**Actionable consequences:**

1. **§3.2.3 design requirement** — negatives must be chosen to break collinearity between candidate explanatory properties, not merely to be "similar."
2. **Collinearity diagnostic** — for a retained hypothesis, check which *other* properties the positive set shares at comparable coverage. If "endoderm-derived" scores as well as "secretory," **report both**: the method genuinely cannot distinguish them, and saying so is more honest than reporting whichever the LLM named first.

#### §L2 — LLM memorisation cannot be excluded, and it inflates the benchmark

The gene symbol is blinded, but **the positive cell-type set is a fingerprint**. Any competent model seeing airway secretory + pancreatic ductal + intestinal crypt cells contrasted against alveolar and hepatic cells will recognise the CFTR pattern from training data.

The benchmark uses well-characterised genes precisely because ground truth exists — **which is the same reason those genes are heavily represented in training data.** The genes that can be validated are the genes most likely to be memorised; the genes actually of interest are those where memorisation cannot help. **Benchmark performance therefore systematically overestimates discovery performance.**

Bounding strategies (mitigate, do not eliminate):
- Test on genes characterised *after* a model's training cutoff
- Compare open vs. closed models with different training corpora
- Check whether performance correlates with literature volume per gene
- Run the split-architecture control arm (§5.2), which is harder to game

#### §L3 — Annotation circularity

CL/GO annotations for a cell type were often assigned partly *because* of the genes it expresses. If a cell type is annotated `capable_of` mucus secretion on evidence including CFTR-family expression, then inferring CFTR's function from that annotation closes a loop. This is the same concern flagged for CellWhisperer's AI-curated descriptions, but here it is baked into decades of human curation.

Partial mitigation: strip the query gene from marker lists (§4.4), and prefer properties supported by non-expression evidence where distinguishable.

#### §L4 — Cannot distinguish "no signal" from "no data"

An abstention or non-convergence has at least four causes:
1. The gene has no cell-type-specific function
2. The atlas does not cover the relevant tissue
3. Annotation is too sparse to support inference
4. The method failed

The pipeline reports one outcome for all four. Any interpretation of negative results must acknowledge this.

#### §L5 — Input quality degrades exactly where the method is most needed

Target genes are understudied — and are often expressed in understudied cell types, in understudied tissues, with thin CL/GO annotation. **The method may work well on genes it is not needed for and poorly on the ones it is.** This is not a small effect and should be measured directly (e.g. performance stratified by annotation richness of the positive set).

#### §L6 — Output is a hypothesis, not a finding

The honest ceiling is "prioritised, evidence-scored candidates for someone else to test experimentally." No result from this pipeline is a biological finding.

#### §L7 — No effect size, mechanism, or direction

"Associated with secretory epithelium" does not say whether the gene is required, regulatory, incidental, or downstream. It is a **context**, not a function — the proposal's own "biological context" language is appropriately careful and the write-up should stay that careful.

#### §L8 — Cell-type resolution only

Everything is aggregated to cell-type level. Function depending on cell state, developmental stage, tissue microenvironment, or condition (stressed vs. resting) is invisible.

---

### Boundable limitations

#### §L9 — Study bias throughout

Which tissues appear in atlases, which cell types get fine-grained annotation, which GO terms exist — all track research attention, which tracks disease relevance and historical funding. Immune and neural cell types are richly described; most others are not. Every recovered property is filtered through this. **Measure and report; do not claim to have corrected for it.**

#### §L10 — Human/mouse is the easiest and least convincing species pair

They are the best-annotated and most harmonised pair, maximising **both** the coupling problem (§L-COUPLING) and the memorisation problem (§L2). More distant species would be stronger evidence but have far worse atlas and ontology coverage. **There is a direct trade-off between evidential strength and data availability, with no current sweet spot.**

#### §L11 — Dropout and thresholding fragility

Single-cell data has systematic false zeros. **The negative set is defined by absence, and absence is the least reliable measurement in the assay.** Threshold sweeps measure sensitivity to this directly, which helps — but the underlying uncertainty does not go away.

#### §L12 — Everything downstream inherits Step 3's choices

The negative set determines what property is findable. Different matching strategies produce different hypotheses for the same gene, and there is no ground truth for "correct matching." **The output is a family of answers, not an answer** — report it that way.

#### §L13 — LLM-scores-LLM is a second opinion, not verification

The Step 6 scorer and Step 8 comparator are models with **correlated blind spots** to the generator. Symbolic GO-based checks help where properties map cleanly to ontology terms — but open-vocabulary output frequently will not.

---

### Practical limitations

#### §L14 — Cost scales multiplicatively

```
genes × species × thresholds × matching strategies × self-consistency runs
      × per-cell-type scoring × full-atlas specificity × null distribution
```

A modest benchmark becomes very large very quickly. This constrains how much sensitivity analysis is feasible, **which in turn weakens the robustness claims the design plans to make**. Budget early.

---

### The two limitations most likely to be raised by reviewers

**§L2 (memorisation confounding the benchmark)** undermines the main validation claim.
**§L1 (expression location is not function)** undermines the premise.

Neither is fatal, but **both need explicit treatment in the proposal — with a concrete mitigation experiment attached to each, even a partial one — rather than a mention in a closing limitations paragraph.**

---

## Appendix A — Key prior art and what is reused from each

| Paper | Relation to this pipeline | Reusable components |
|---|---|---|
| **Hu et al., Nature Methods (2025)** — LLM evaluation for gene set function | Closest precedent for the LLM reasoning core, but entities are genes in a set, not cell types selected by one gene | (a) Semantic similarity for scoring free-text against ground truth; (b) three-tier real / 50-50 / random control design; (c) Jaccard-style specificity metric; (d) finding that **only GPT-4 reliably abstained on random sets** — abstention capability is model-dependent and must be tested |
| **Schaefer, Peneder & Bock, Nature Biotechnology (2025)** — CellWhisperer | Contrastive transcriptome–text embedding (CLIP-like), not reasoning over a contrast; retrieval/chat rather than hypothesis generation | (a) Zero-shot cell-type prediction benchmark as an evaluation template; (b) embeddings potentially repurposable for matched-negative selection (§3.3); (c) **cautionary case on AI-curated text as model input** (§L3); (d) already integrated with cellxgene |
| **Boyle et al., Genome Biology (2025)** — GeneTEA | Free-text de novo annotation space via classical NLP (bag-of-words / TF-IDF), not an LLM; cannot generate novel synthesising hypotheses | (a) Motivation argument that fixed vocabularies cause redundancy and poor interpretability; (b) design choice to use coherent *descriptions* rather than fragmented annotations (§4.3.6); (c) cross-species ortholog nearest-neighbour embedding check as a lightweight complement to Step 8 |

---

## Appendix B — Data sources

| Resource | Role | Notes |
|---|---|---|
| **cellxgene Census** | Primary expression interface | Aggregates many datasets; CL-standardised annotations; handles expression + ontology mapping together |
| **Tabula Sapiens** | Human atlas | ~475 cell types, 24 organs, ~481k cells post-QC, expert-annotated against an ontology |
| **Tabula Muris** | Mouse atlas | ~100k cells, 20 organs; designed to be comparable to the human counterpart |
| **Mouse Cell Atlas (scMCA)** | Alternative mouse atlas | Different sequencing method (Microwell-seq), >400k cells, >40 tissues |
| **Human Cell Atlas (HCA)** | Broader human resource | Coordinated collection rather than a single dataset |
| **Cell Ontology (CL)** | Cell-type descriptions | OBO Foundry; stable IDs, definitions, hierarchy, markers |
| **Uberon** | Anatomy | Species-neutral — see §L-COUPLING |
| **GO** | Function / process / component | Species-neutral — see §L-COUPLING |
| **Ensembl Compara / OrthoDB** | Ortholog mapping | Assumed as given input; returns orthology type and confidence |
| **MyGene.info** | Gene ID normalisation | Symbol/alias/accession → canonical ID |
| **HP / MP** | Species-specific phenotype ontologies | Candidate independent evidence axis (§L-COUPLING mitigation 2) |

---

## Appendix C — Open design questions

1. Description assembly format: structured triples vs. prose paragraphs — decide by ablation (§4.3.6).
2. Relation traversal depth in CL — configurable, to be swept (§4.3.2).
3. Which matching strategy is primary; whether cross-strategy agreement becomes a formal robustness criterion (§3.3).
4. Whether a local open model (llama3.3:70b via Ollama) can abstain reliably enough for `negative_control` mode, or whether a frontier model is required for the generation step (§5.5).
5. Payload policy when threshold sweeps yield divergent hypotheses (§7.4).
6. Pairwise vs. consensus comparison when extending beyond two species (§8.4).
7. Whether to filter null pairs for functional dissimilarity, accepting a conservative test otherwise (§8.3).
8. Full-atlas vs. stratified-sample specificity scoring, given cost (§6.4).
