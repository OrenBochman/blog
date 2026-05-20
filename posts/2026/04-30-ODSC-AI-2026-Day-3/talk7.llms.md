## The AI Agent Memory Landscape

Play

![Vimeo](https://f.vimeocdn.com/p/images/crawler_logo.png)

- William Lyon
  - Neo4j
  - papers:
    - [Memory in the Age of AI Agents](https://arxiv.org/abs/2512.13564)
    - [Three Memory Types](https://graphstuff.com/three-memory-types)
    - [slides](https://graphstuff.com/odsceast26)
    - [GitHub](https://github.com/neo4j-labs/create-context-graph)
    - [Website](https://create-context-graph.dev)

> **NOTE:**
>
> - **Topic:** AI agent memory, presented through Neo4j’s work on graph-based memory for production agents.
>
> - **Core analogy: flying a plane**
>
>   - A passenger suddenly asked to fly a plane may have instruments, manuals, and a goal, but lacks experience, route-specific knowledge, institutional procedures, and prior debriefs.
>   - The speaker argues that many agents are in the same position: they receive a role and goal, plus tools and retrieval, but lack accumulated operational memory.
>
> - **Main problem**
>
>   - Production agents need more than Retrieval-Augmented Generation (RAG), tools, and prompts.
>   - They need access to the organizational context humans use when making decisions: policies, precedents, customer history, prior decisions, procedures, and tacit/institutional knowledge.
>
> - **Context graph**
>
>   - A “context graph” is presented as a structured representation of the information needed to make decisions inside an organization.
>   - It connects entities, events, decisions, policies, risk factors, documents, people, accounts, transactions, and reasoning traces.
>   - The key point is that before evaluating whether an agent made the right decision, we need to know what information a human would have used to make that decision.
>
> - **Memory taxonomy**
>
>   - The talk distinguishes between:
>
>     - **Short-term memory:** current interaction, messages, session state.
>     - **Long-term memory:** extracted facts, entities, preferences, relationships, and durable knowledge.
>     - **Reasoning memory:** decisions, plans, traces, justifications, and prior problem-solving paths.
>
>   - The speaker also references literature that divides memory into:
>
>     - **Token-level memory:** external memory accessible to application developers.
>     - **Parametric memory:** knowledge stored in model weights.
>     - **Latent memory:** internal model representations.
>
>   - The practical focus is token-level memory because it is what engineers can build around when using Large Language Model (LLM) APIs.
>
> - **RAG versus agent memory**
>
>   - RAG mainly retrieves relevant chunks from documents, often using embeddings and vector search.
>   - Agent memory also retrieves, but adds a learning component: it constructs memory from conversations, tool calls, decisions, and prior interactions.
>   - The speaker treats RAG and memory as overlapping, not sharply separated.
>
> - **Limitations of flat memory**
>
>   - Simple chat history, files, or vector stores miss explicit relationships between remembered facts.
>   - Graph memory is presented as better suited for representing relationships, provenance, decisions, entities, policies, and evolving context.
>
> - **Financial services demo**
>
>   - The example context graph includes customers, accounts, transactions, approvals, risk factors, policies, and prior decisions.
>   - An agent evaluates a customer request for a \$25,000 credit limit increase.
>   - The agent uses tools to fetch customer data, policies, precedents, fraud signals, and graph analytics.
>   - Graph algorithms such as node similarity and community detection are exposed as tools the agent can call.
>   - The agent can also generate Cypher queries directly against Neo4j.
>
> - **Decision trace**
>
>   - The agent does not merely answer; it records why it made the decision.
>   - Prior similar requests, risk factors, policy constraints, and supporting evidence are written back into the graph.
>   - This makes future agents able to reuse the decision context.
>
> - **Neo4j Agent Memory**
>
>   - The speaker introduces **Neo4j Agent Memory**, an open-source Python package and hosted service.
>   - It provides abstractions for short-term, long-term, and reasoning memory.
>   - The Python package aims to integrate with many Python agent frameworks.
>   - The hosted service is intended to support use outside Python as well.
>
> - **Memory construction pipeline**
>
>   - New messages can trigger background entity extraction, entity resolution, and enrichment.
>   - Large Language Models are useful for extraction and resolution, especially with an ontology, but they are slow and expensive if used alone.
>   - The system therefore supports a pipeline approach, combining tools such as spaCy with LLM-based enrichment.
>
> - **Importance of domain ontologies**
>
>   - The speaker emphasizes that knowledge graph quality depends heavily on the ontology used to extract structured data from unstructured text.
>   - A pharmaceutical research setting would need entities such as papers, genes, proteins, drugs, and diseases.
>   - The default model mentioned is based on a POLE-style ontology: person, organization, location, event, object, with extensions.
>
> - **Create Context Graph CLI**
>
>   - The talk demos a command-line tool called `create context graph`.
>   - It scaffolds a full-stack agent memory application.
>   - The user can choose demo data or connect real systems, select a domain ontology, choose an agent framework such as Pydantic, connect to Neo4j, enable entity extraction and preference detection, and configure model providers or embedding models.
>   - It can also generate a Model Context Protocol (MCP) server for exposing memory to other agent environments.
>
> - **Healthcare demo**
>
>   - A generated healthcare example includes patients, doctors, providers, facilities, treatments, and treatment-plan decisions.
>   - The agent queries the context graph to retrieve recent treatment decision traces and explain the reasoning behind them.
>
> - **Real data connectors**
>
>   - The system can ingest data from sources such as GitHub, Claude Code session history, and Google Workspace.
>   - The goal is to unify project context: code changes, documents, requirements, decisions, tool calls, and discussions.
>   - A Claude Code example shows messages, files, tool calls, and decisions represented as nodes and relationships in a graph.
>
> - **Multi-agent memory**
>
>   - The speaker describes a financial-services multi-agent setup with specialized agents such as know-your-customer, anti-money-laundering, compliance, and credit agents.
>   - When one agent discovers something important, such as a sanctioned individual or suspicious customer, it writes that to shared memory.
>   - Other agents can immediately use that information.
>
> - **Cross-framework compatibility**
>
>   - A key challenge is making the same memory layer usable by agents written in different frameworks and languages.
>   - The speaker mentions a compliance kit / Technology Compatibility Kit (TCK) for validating memory implementations.
>   - Example agents include Pydantic, Vercel AI SDK with TypeScript, Go, LangGraph, C#, and R.
>   - The point is to enforce a shared memory shape, API, and behavioral specification.
>
> - **Main takeaway**
>
>   - Effective production agents need structured, shared, queryable, and evolving memory.
>   - A graph-based memory layer can connect short-term interactions, long-term knowledge, reasoning traces, tools, policies, and organizational context.
>   - This makes agents more auditable, more consistent, and better able to act with the kind of institutional knowledge humans rely on.

### reflections

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2026,
  author = {Bochman, Oren},
  title = {The {AI} {Agent} {Memory} {Landscape}},
  date = {2026-04-28},
  url = {https://orenbochman.github.io/posts/2026/04-30-ODSC-AI-2026-Day-3/talk7.html},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2026. “The AI Agent Memory Landscape.” April 28. <https://orenbochman.github.io/posts/2026/04-30-ODSC-AI-2026-Day-3/talk7.html>.
