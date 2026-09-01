## Book Author: AI Agents with MCP

- Kyle Stratis
  - [Homepage](https://kylestratis.com/) Very interesting!!
  - [LinkedIn](https://www.linkedin.com/in/kylestratis/)
  - Book
    - [AI Agents with MCP](https://www.oreilly.com/library/view/ai-agents-with/9798341639546//)
  - [slides](https://docs.google.com/presentation/d/1y7kfXkQIda5YFL6f_EieHqWqL4cC-BYD)

> **NOTE:**
>
> - **Event context**
>
>   - The transcript is from a virtual book signing with Kyle Stratis, founder/principal consultant of Stratus Data Labs and VP of Engineering at the SQL Institute.
>   - The discussion centers on his book ***AI Agents with MCP***, currently in early access on O’Reilly and available for pre-order.
>
> - **Motivation for the book**
>
>   - The book grew out of Kyle’s difficulty building his first **Model Context Protocol (MCP)** client in early 2025.
>   - At the time, most MCP documentation focused on servers, while client-side implementation guidance was sparse.
>   - His goal is to provide an authoritative, practical reference for building with MCP in Python, covering clients, servers, security, transports, architecture, and deployment.
>
> - **What MCP solves**
>
>   - MCP decouples tools, prompts, data, and resources from the agent application itself.
>   - Without MCP, tools are often tightly coupled to a specific agent framework or application.
>   - With MCP, service providers can expose agent-ready capabilities once, and application developers can reuse them across agents and use cases.
>   - This makes MCP especially useful for enterprises and platform teams that need standardized, distributable agent interfaces.
>
> - **Advice for building AI agents**
>
>   - Standard software engineering principles still apply: clean architecture, code quality, and avoiding unnecessary duplication.
>   - Agent systems introduce a broader security attack surface because models can interpret arbitrary natural-language inputs.
>   - Kyle highlights [Simon Willison’s “lethal trifecta”: privileged data access, external-world access, and code execution](https://simonwillison.net/2025/Aug/9/bay-area-ai/). Combining all three in one agent can create serious data-exfiltration risks.
>   - A safer architecture separates these capabilities across agents, sub-agents, or coordinated components.
>
> - **Evaluation and observability**
>
>   - Agent evaluation should happen both during development and in production.
>   - Tools like Promptfoo can help test tool choice, prompt behavior, and expected user scenarios.
>   - Observability and tracing are necessary to understand what users actually ask, how the agent responds, and where it fails.
>
> - **Python CLI discussion**
>
>   - The workshop code uses the standard Python pattern `if __name__ == "__main__":` to run code directly from the command line.
>   - Basic CLI interaction can be handled with `input()` and `print()`.
>   - For richer command-line interfaces, Kyle mentions libraries such as `Click`, `Typer`, and `Argparse`, with a preference for purpose-built CLI packages over bare Argparse.
>
> - **Skills versus MCP**
>
>   - Skills are described as selectively retrievable prompts that models can use for specific tasks.
>   - They are useful for personal workflows, small-team automation, code review, editing, drafting, note expansion, and lightweight repeated tasks.
>   - Their limitations include weak versioning, difficulty pinning dependencies, inconsistent model usage, and often being tied to a specific model ecosystem.
>   - MCP can also expose prompts, but it additionally supports tools, resources, data access, sampling, authentication, and richer interaction patterns.
>
> - **When to use a skill**
>
>   - Use a skill when the task is mostly prompt-based, lightweight, personal, or limited to a small team.
>   - Example: using Claude to generate language-learning flashcards from notes.
>   - Skills are lower-overhead and easier to create than a full MCP server.
>
> - **When to use an MCP server**
>
>   - Use an MCP server when the capability resembles an external API, needs authentication, must be distributed broadly, requires versioning, or exposes tools/data rather than just a prompt.
>   - MCP is more appropriate for enterprise distribution, service-provider integrations, and advanced interaction patterns such as elicitation.
>
> - **Overall takeaway**
>
>   - MCP is best understood as infrastructure for distributing reusable, agent-ready capabilities.
>   - Skills are excellent for lightweight prompt augmentation; MCP is better for robust, scalable, authenticated, and reusable agent integrations.
>   - The practical challenge in modern agent development is not only making agents useful, but making them secure, observable, evaluable, and maintainable.

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2026,
  author = {Bochman, Oren},
  title = {Kyle {Stratis}},
  date = {2026-04-28},
  url = {https://orenbochman.github.io/posts/2026/04-28-ODSC-AI-2026-Day-1/talk12.html},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2026. “Kyle Stratis.” April 28. <https://orenbochman.github.io/posts/2026/04-28-ODSC-AI-2026-Day-1/talk12.html>.
