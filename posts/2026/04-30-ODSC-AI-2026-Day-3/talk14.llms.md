## Agentic LLMs in Practice

- Naman Goyal
  - [website](https://namangoyal.com/)
  - [LinkedIn](https://www.linkedin.com/in/goyal-naman/)
  - [slides](https://drive.google.com/file/d/1qWMLECJSSEjaSTprg6ZwGTtidP18ffGn)
  - [colab](https://github.com/thenamangoyal/llm-engineering-workshop)
  - [Google DeepMind](https://www.deepmind.com/)

> **NOTE:**
>
> - [nworkshopb1](https://github.com/thenamangoyal/llm-engineering-workshop/blob/main/Agentic_LLMs_Workshop.ipynb)
>   - Module 1 — Function calling, end to end. A function-calling agent loop on SQLite + a mock weather API, with strict Pydantic-validated tool arguments.
>   - Module 2 — Reference architectures. A router-worker state machine with Pydantic contracts at every node, compared head-to-head with a free-form ReAct loop.
>   - Module 3 — Surviving production. A retry-storm demo on a deliberately flaky upstream, with a playground cell where you tune the retry policy yourself and watch the bars move.
>   - Module 4 — Observability. An OpenTelemetry-style traced agent run, rendered as a Gantt chart you generate from your own spans.
> - [pydantic](https://pydantic.dev/) - a data validation and settings management library for Python, based on type annotations. It provides a way to define data models with type hints and validates the data against those models, making it easier to work with structured data in Python applications.
> - [sqlite](https://www.sqlite.org/) - a C-language library that implements a small, fast, self-contained, high-reliability, full-featured, SQL database engine.
> - [MCP](https://modelcontextprotocol.org/) - a standard for connecting language models to external data sources and tools, enabling them to access and manipulate information beyond their training data. MCP defines a protocol for communication between language models and external services, allowing for more dynamic and interactive applications.
> - [A2A](https://a2a.dev/) - a framework for building agentic applications that can interact with each other and with external services using the Model Context Protocol (MCP). A2A provides tools and libraries for creating, managing, and orchestrating agentic applications in a scalable and efficient way.
> - [LangGraph](https://langchain-ai.github.io/langgraph/)
> - [Plan-and-Execute pattern](https://blog.langchain.dev/planning-agents/) (LangChain blog)
> - [Tenacity](https://tenacity.readthedocs.io/)
> - [Function-calling guide (OpenAI)](https://platform.openai.com/docs/guides/function-calling)
> - [ReAct (Yao et al., 2022)](https://arxiv.org/abs/2210.03629)
> - [OpenTelemetry](https://opentelemetry.io/)

### Reflection

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2026,
  author = {Bochman, Oren},
  title = {Agentic {LLMs} in {Practice}},
  date = {2026-04-28},
  url = {https://orenbochman.github.io/posts/2026/04-30-ODSC-AI-2026-Day-3/talk14.html},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2026. “Agentic LLMs in Practice.” April 28. <https://orenbochman.github.io/posts/2026/04-30-ODSC-AI-2026-Day-3/talk14.html>.
