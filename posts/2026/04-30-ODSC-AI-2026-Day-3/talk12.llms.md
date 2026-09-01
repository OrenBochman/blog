## GraphRAG in Healthcare: Enhancing Clinical Reasoning with Knowledge Graphs, GNNs, and Agents

- Giuseppe Futia
- CSI Piemonte
- [slides](https://docs.google.com/presentation/d/1ipi1sVaTisg-L-vLalUmf03cUZvA3V1x/edit)

> **NOTE:**
>
> - **Main topic:** The talk explains how graph technologies can support healthcare applications by combining:
>
>   - **Knowledge graphs** for structured medical knowledge representation.
>   - **Large language models (LLMs)** for extraction, annotation, and reasoning.
>   - **Graph neural networks (GNNs)** for graph-aware embeddings and disambiguation.
>   - **Agents** that can query both public medical knowledge and private patient data.
>
> - **Healthcare data challenges:**
>
>   - Medical data comes from heterogeneous sources: electronic health records, lab results, diagnoses, medications, clinical notes, reports, publications, and ontologies.
>   - Patient data is sensitive, so the system should avoid sending it to external LLM services.
>   - The speaker argues for keeping patient data inside local or legacy infrastructure and accessing it virtually when needed.
>
> - **Proposed architecture:**
>
>   - Use local/open LLMs rather than remote API-based models.
>   - Store public medical knowledge in a graph database such as Neo4j.
>   - Keep private clinical data in legacy databases and materialize it only at query time.
>   - Use graph-based components for ontology integration, information extraction, enrichment, and patient-data access.
>
> - **Medical ontologies as semantic infrastructure:**
>
>   - The talk highlights resources such as **UMLS** — Unified Medical Language System — **ICD-10**, and **HPO** — Human Phenotype Ontology.
>   - UMLS acts as a bridge across medical vocabularies.
>   - ICD-10 provides hierarchical disease classifications.
>   - HPO connects phenotypic abnormalities, symptoms, diseases, and sometimes frequency information.
>   - These ontologies help normalize ambiguous medical terms, such as distinguishing a virus, disease, symptom, or clinical finding.
>
> - **Entity recognition and disambiguation:**
>
>   - Clinical narratives contain ambiguous terms and synonyms.
>   - The system first identifies candidate entities, then disambiguates them using ontology context.
>   - Example: “Zika” may refer to multiple related medical entities.
>   - Another example distinguishes “shortness of breath,” “dyspnea,” and “tachypnea,” showing that lexical similarity alone is not enough.
>
> - **Ontology mapping workflow:**
>
>   - Candidate selection is performed using vector similarity over embeddings stored in Neo4j.
>   - Candidate disambiguation is then performed with an LLM, using contextual information from ontology structure, definitions, synonyms, and hierarchy.
>   - The speaker emphasizes that LLM quality depends heavily on the quality and relevance of the context provided.
>
> - **Role of graph neural networks:**
>
>   - GNNs are introduced as a way to improve embeddings by incorporating neighborhood structure.
>
>   - The speaker explains message passing through three steps:
>
>     - message,
>     - aggregate,
>     - update.
>
>   - Instead of representing a node only by its text, a GNN represents it using information from neighboring nodes and relationships.
>
> - **Why GNNs help:**
>
>   - Pure textual embeddings can miss correct ontology matches when terms are lexically different.
>   - GNNs can use relational structure to recover semantically correct candidates.
>   - In the validation example, Qwen embeddings failed to place the correct entity in the top five in 34 out of 368 cases; GNN re-ranking rescued about half of those cases.
>   - Example: “cervicalgia” should map to “neck pain”; text-only embeddings ranked it 19th, while the GNN-enhanced representation moved it to first place.
>
> - **G-Retriever model:**
>
>   - The talk introduces a GNN-plus-LLM approach based on G-Retriever.
>   - It extracts a relevant subgraph using a Prize-Collecting Steiner Tree-style method.
>   - The subgraph is encoded by a GNN and passed to the LLM as graph-derived “soft tokens.”
>   - This gives the LLM graph-structured context rather than only textual context.
>
> - **Graph agent use case:**
>
>   - A graph-based agent can use several tools:
>
>     - query public medical knowledge in the graph,
>     - query private patient data virtually,
>     - combine both to answer clinical questions.
>
>   - Example questions include retrieving a patient’s follow-up plan or identifying possible diseases based on HPO symptom coverage.
>
>   - The key advantage is that answers are grounded in explicit ontology structure rather than relying only on the LLM’s internal knowledge.
>
> - **Core message:**
>
>   - Graphs provide a structured, interpretable, and privacy-preserving foundation for healthcare AI.
>   - LLMs are useful, but they need well-selected, semantically organized context.
>   - GNNs improve retrieval and disambiguation by exploiting graph topology.
>   - Agents can unify these components into systems that reason across public medical knowledge and private patient data without unnecessarily moving sensitive data.
>
> - **Closing material:**
>
>   - The speaker briefly promotes a related book and a knowledge graph training program.
>   - The ODSC host closes the event and encourages attendees to revisit sessions on demand and provide feedback.

### Reflection

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2026,
  author = {Bochman, Oren},
  title = {GraphRAG in {Healthcare}},
  date = {2026-04-28},
  url = {https://orenbochman.github.io/posts/2026/04-30-ODSC-AI-2026-Day-3/talk12.html},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2026. “GraphRAG in Healthcare.” April 28. <https://orenbochman.github.io/posts/2026/04-30-ODSC-AI-2026-Day-3/talk12.html>.
