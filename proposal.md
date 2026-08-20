Project Hub: [[DendoLLM]]

# New Proposal (2026-08-10)

TL;DR
- 

## Cross-species natural-language enrichment of gene expression across cell types

### Enhanced proposal text

Many genes remain incompletely characterised, even though nearly every gene has a measurable expression pattern across cell types. Cell types, by contrast, are richly described in natural language: we know where they occur, what they look like, what they do, and how they relate to other cells. We propose to exploit this asymmetry to infer the biological context of poorly characterised genes. 
For each gene, we will identify the cell types in which it is reproducibly expressed and, crucially, contrast these gene-positive cell types with matched cell types in which the gene is weakly expressed or absent. 
Using existing Cell Ontology definitions, atlas annotations, anatomical location, morphology, and known physiology, a large language model will propose the most specific biological property that is enriched among the gene-positive cell types and depleted among the comparison cell types. 
The LLM will therefore generate candidate natural-language annotations rather than being restricted to a predefined vocabulary; an independent scoring step will then quantify positive-set coverage, negative-set leakage, and significance. 
In this sense, the approach is a form of enrichment analysis in which cell types are the entities and natural language is the annotation space. The result is a testable hypothesis about the functional context in which the gene is used.

A second key feature is that the analysis will be performed independently in multiple species. For orthologous genes, each species will be analysed using its own single-cell atlas and cell-type annotations, without exposing one species' result to another. 
The resulting hypotheses will be compared only after inference. 
A biological property that repeatedly emerges from different species, despite different datasets and cell-type compositions, provides evidence for a conserved functional context and is less likely to reflect a peculiarity of one atlas. 
We will benchmark the method using well-characterised genes while keeping gene identity hidden from the model, assess robustness across expression thresholds and matched background sets, and use carefully matched random cell-type sets and broadly expressed genes to test whether the method correctly returns no specific property when the data do not support one.

### One-sentence version

For each gene, compare the biological descriptions of cell types that express it with matched cell types that do not, infer the natural-language property that best separates the two groups, and retain hypotheses that recur independently across species.

## Prior art

The papers below are ranked by conceptual overlap with the proposed method. None of the papers identified in the search appears to combine all of the following in one workflow: a single gene as the query; gene-positive versus gene-negative cell-type contrast; unrestricted natural-language cell-type annotations; independent testing of the proposed descriptor; and recurrence across multiple species as an internal validation criterion.

### Top 3 most relevant

1. Hu et al., Nature Methods (2025), “Evaluation of large language models for discovery of gene set function.”
Similar: Uses an LLM to infer a common biological function from an input set without relying only on fixed enrichment vocabularies, and evaluates random gene sets as a negative control.
Difference: The entities are genes in a gene set. Here, one gene selects cell types, and gene-positive and gene-negative cell types are contrasted to infer the gene's biological context.

2. Schaefer, Peneder & Bock, Nature Biotechnology (2025), “Multimodal learning enables chat-based exploration of single-cell data” (CellWhisperer).
Similar: Directly connects transcriptomic profiles with textual annotations and enables natural-language questions about cells and genes.
Difference: It is a multimodal transcriptome-text model for search, annotation, and chat; it does not formulate a positive-versus-negative cell-type enrichment problem for an individual gene.

3. Boyle et al., Genome Biology (2025), “Natural language processing of gene descriptions for overrepresentation analysis with GeneTEA.”
Similar: Uses free-text biological descriptions to create a de novo annotation space for overrepresentation analysis, addressing limitations of fixed ontologies.
Difference: The free text describes genes and the input is a gene list. Here, cell types are the enriched entities and their descriptions are used to interpret one gene.

### Other relevant papers

Tan et al., Nature Communications (2026), “An embedding-based framework enables statistical testing of gene-set function hypotheses inferred by large language models.”
Important for method design because it statistically tests de novo LLM-derived functional hypotheses using embeddings. It still operates on gene sets rather than cell-type sets selected by a single gene.

Wang et al., Nature Methods (2025), “GeneAgent: self-verification language agent for gene-set analysis using domain databases.”
Uses an LLM agent plus biological databases to improve gene-set analysis and reduce hallucination through self-verification. It focuses on gene-set functional annotation rather than cell-type contrast for individual genes.

Hou & Ji, Nature Methods (2024), “Assessing GPT-4 for cell type annotation in single-cell RNA-seq analysis.”
Shows that LLMs can infer cell identities from marker-gene sets across tissues and species. The direction is genes to cell-type label, whereas the proposed method goes from one gene to its expressing cell types to a discriminating biological property.

Levine et al., ICML/PMLR (2024), “Cell2Sentence: Teaching Large Language Models the Language of Biology.”
Represents single-cell expression profiles as language-like sequences so LLMs can perform single-cell tasks. It does not perform natural-language enrichment over gene-positive versus gene-negative cell types.

Li et al., arXiv (2025), “A Brain Cell Type Resource Created by Large Language Models and a Multi-Agent AI System for Collaborative Community Annotation” (BRAINCELL-AID).
Combines free-text descriptions, ontology labels, RAG, and LLMs to annotate brain cell-type marker gene sets and cell clusters. It does not infer an individual gene's context from its distribution across cell types.

Tarashansky et al., eLife (2021), “Mapping single-cell atlases throughout Metazoa unravels cell type evolution” (SAMap).
Relevant to the multi-species component because it maps homologous cell types and shared expression programs across distant species. It provides cross-species cell-type mapping rather than natural-language gene-context inference.

Karlsson et al., Science Advances (2021), “A single-cell type transcriptomics map of human tissues.”
Demonstrates genome-wide classification of genes by cell-type-specific expression and provides the type of atlas-level gene-to-cell-type distribution the proposed method would exploit. It does not infer free-text shared properties or use explicit positive-versus-negative comparison.

Clarke et al., Communications Biology (2024), “Rummagene: massive mining of gene sets from supporting materials of biomedical research publications.”
Uses large collections of published gene sets and literature text for enrichment and gene-function prediction, showing the value of text beyond standard ontologies. It does not use cell-type descriptions selected by one gene.

## Suggested novelty statement

Prior work has shown that LLMs can summarize gene-set function, that free text can augment or replace fixed enrichment vocabularies, and that transcriptomes can be aligned with natural-language descriptions. Our approach combines these ideas in a different direction: it treats the cell types associated with an individual gene as the entities of an enrichment analysis, explicitly contrasts gene-positive with gene-negative cell types, and asks for the biological property that best separates them. The analysis is then repeated independently across species so that recurrent hypotheses can be prioritized as conserved functional contexts.

Use “to our knowledge” rather than claiming that no prior method exists unless a formal systematic literature review is performed.

## References

Hu M, Alkhairy S, Lee I, et al. Evaluation of large language models for discovery of gene set function. Nature Methods. 2025;22:82–91. doi.org/10.1038/s41592-024-02525-x

Schaefer M, Peneder P, Bock C. Multimodal learning enables chat-based exploration of single-cell data. Nature Biotechnology. 2025. doi.org/10.1038/s41587-025-02857-9

Boyle IA, Aquib NA, Kocak M, et al. Natural language processing of gene descriptions for overrepresentation analysis with GeneTEA. Genome Biology. 2025;26:376. doi.org/10.1186/s13059-025-03844-8

Tan Y, Wang L-J, Liang T, et al. An embedding-based framework enables statistical testing of gene-set function hypotheses inferred by large language models. Nature Communications. Published 30 July 2026. doi.org/10.1038/s41467-026-75972-z

Wang Z, Jin Q, Wei C-H, et al. GeneAgent: self-verification language agent for gene-set analysis using domain databases. Nature Methods. 2025;22:1677–1685. doi.org/10.1038/s41592-025-02748-6

Hou W, Ji Z. Assessing GPT-4 for cell type annotation in single-cell RNA-seq analysis. Nature Methods. 2024;21:1462–1465. doi.org/10.1038/s41592-024-02235-4

Levine D, Rizvi SA, Lévy S, et al. Cell2Sentence: Teaching Large Language Models the Language of Biology. Proceedings of the 41st International Conference on Machine Learning. PMLR 235:27299–27325 (2024). proceedings.mlr.press/v235/levine24a.html

Li R, Chen W, Li Z, et al. A Brain Cell Type Resource Created by Large Language Models and a Multi-Agent AI System for Collaborative Community Annotation (BRAINCELL-AID). arXiv:2510.17064 (2025). arxiv.org/abs/2510.17064

Tarashansky AJ, Musser JM, Khariton M, et al. Mapping single-cell atlases throughout Metazoa unravels cell type evolution. eLife. 2021;10:e66747. doi.org/10.7554/eLife.66747

Karlsson M, Zhang C, Méar L, et al. A single-cell type transcriptomics map of human tissues. Science Advances. 2021;7:eabh2169. doi.org/10.1126/sciadv.abh2169

Clarke DJB, Marino GB, Deng EZ, Xie Z, Evangelista JE, Ma’ayan A. Rummagene: massive mining of gene sets from supporting materials of biomedical research publications. Communications Biology. 2024. doi.org/10.1038/s42003-024-06177-7


