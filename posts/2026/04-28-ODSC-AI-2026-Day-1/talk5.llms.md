## From Intelligent to Agentic Applications: Using Model Context Protocol to Support Agentic Behaviors in Your Application

Play

![Vimeo](https://f.vimeocdn.com/p/images/crawler_logo.png)

- Kyle Stratis
  - [Homepage](https://kylestratis.com/) - Very interesting!!
  - [LinkedIn](https://www.linkedin.com/in/kylestratis/)
  - Book
    - [AI Agents with MCP](https://www.oreilly.com/library/view/ai-agents-with/9798341639546//)
- [slides](https://docs.google.com/presentation/d/1y7kfXkQIda5YFL6f_EieHqWqL4cC-BYD)

> **NOTE:**
>
> - **Why the book was written**
>   - Kyle began writing the book after struggling to build his first MCP client in early 2025.
>   - At the time, MCP documentation was heavily focused on servers, while guidance for building clients was sparse.
>   - His goal is to provide a practical and architectural reference for MCP, especially in Python.
>   - The book covers MCP servers, clients, security, transports, implementation details, and deployment patterns.
> - **What MCP does**
>   - MCP decouples tools, prompts, and data from the agent application itself.
>   - This prevents developers from having to rebuild the same tools separately for each agent or framework.
>   - MCP allows service providers to expose agent-ready interfaces in a standardized way.
>   - In Kyle’s framing, MCP is especially useful for distributing tools and capabilities across teams, products, and enterprise users.
> - **Agent-building advice**
>   - Standard software engineering principles still matter: code quality, maintainability, and avoiding unnecessary duplication.
>   - The newer challenge with agents is their broader attack surface.
>   - Because agents operate over natural language and may access many systems, they are more vulnerable to misuse or prompt-driven failures.
>   - Kyle highlights Simon Willison’s “lethal trifecta”: combining access to private data, external communication, and code execution can create serious data-exfiltration risks.
>   - A safer architecture separates these capabilities, for example through separate agents, coordinators, or sub-agents.
> - **Evaluation and observability**
>   - Agent evaluation should happen both during development and in production.
>   - Kyle recommends development-time evaluation tools such as Promptfoo for testing tool choice, expected behavior, and likely user scenarios.
>   - He also stresses the need for tracing and observability to understand what users actually ask and how the agent responds in real conditions.
> - **Python command-line integration**
>   - The workshop’s Python code interacts with the command line using standard Python patterns.
>   - The `if __name__ == "__main__"` construct allows the script to be run directly from the command line.
>   - Basic functions such as `input()` and `print()` are sufficient for simple command-line interaction.
>   - For richer command-line interfaces, Kyle mentions tools such as Click, Typer, and Argparse, with a preference for purpose-built packages over raw Argparse for more polished user interfaces.
> - **Skills versus MCP**
>   - Kyle sees skills as useful, especially for personal or small-team workflows.
>   - Skills are essentially selectively invoked prompts, and they work well for tasks such as code review, editing, drafting, note expansion, and flashcard generation.
>   - Their limitations include weaker dependency management, lack of straightforward version pinning, and model-specific behavior.
>   - MCP is broader: it can expose prompts, tools, resources, data, and more advanced interaction patterns.
> - **When to use a skill**
>   - Use a skill when the task is primarily prompt-based.
>   - Skills are appropriate for lightweight personal automation or small-team workflows.
>   - They are a good first step because many useful agent behaviors can be achieved with a well-written prompt.
>   - Example: using Claude Code and personal notes to generate language-learning flashcards.
> - **When to use an MCP server**
>   - Use an MCP server when the agent needs stronger distribution, authentication, versioning, or integration with external systems.
>   - MCP is better when exposing tools, data, or API-like capabilities to many users.
>   - It is also preferable for advanced interaction patterns, such as elicitation, where the server needs to request user input to complete its work.
>   - Kyle’s rule of thumb: if the agent interaction resembles calling or exposing an external API, an MCP server is usually the better abstraction.
> - **Main takeaway**
>   - MCP is presented as an infrastructure layer for scalable, reusable, and secure agent capabilities.
>   - Skills are lightweight prompt-level extensions; MCP servers are more formal, distributable, and suitable for tool- and data-rich agent ecosystems.
>   - Building effective agents requires not only tool access, but also careful security architecture, evaluation, and runtime observability.

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2026,
  author = {Bochman, Oren},
  title = {From {Intelligent} to {Agentic} {Applications}},
  date = {2026-04-28},
  url = {https://orenbochman.github.io/posts/2026/04-28-ODSC-AI-2026-Day-1/talk5.html},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2026. “From Intelligent to Agentic Applications.” April 28. <https://orenbochman.github.io/posts/2026/04-28-ODSC-AI-2026-Day-1/talk5.html>.
