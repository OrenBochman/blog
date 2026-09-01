## Building FSI Agents with Claude

- [Mikaela Grace](https://www.linkedin.com/in/mikaelagrace/)
- [April Guo](https://www.linkedin.com/in/aprilguo/)
- Anthropic
- [slides](https://docs.google.com/presentation/d/1OOwvHJSE7sGkuxwhAIRTUVKIHXOfWqUL/edit?usp=drive_link&ouid=114139178201838113621&rtpof=true&sd=true)

> **NOTE:**
>
> - The talk is led by Mikaela Grace and April Guo from Anthropic’s Applied AI team, with a focus on building Claude-based agents for financial-services Industry workflows.
>
> - The agenda has three parts:
>
>   1.  best practices for agent design,
>   2.  financial-services examples using `Claude Cowork`, `Claude Code`, `Office agents`, `skills`, and `Model Context Protocol` integrations,
>   3.  hands-on practice building either a no-code `Cowork` agent or a Python evaluation workflow.
>
> - The first key distinction is between **workflows** and **agents**:
>
>   - workflows follow a predefined, deterministic path of prompts, tools, routing, and chained steps;
>   - agents dynamically decide their own process, tool use, and iteration path.
>
> - Workflows are best when the task is repeatable, predictable, low-variance, and does not require much model judgment.
>
> - Agents are better for open-ended, multi-step tasks where the path to completion depends on user input, changing context, or information discovered during execution.
>
> - The basic agent loop is described as: model receives a goal, uses tools, observes the environment, reasons about the result, and iterates until it reaches a success or stopping condition.
>
> - Good agent design depends less on elaborate prompting and more on giving the model:
>
>   - clear tools,
>   - useful context,
>   - strong tool descriptions,
>   - explicit success criteria,
>   - observability into what it did.
>
> - Tool descriptions are critical because they are the agent’s interface to its environment. They should explain purpose, expected use cases, edge cases, input/output formats, and examples.
>
> - The speakers recommend designing tools at the right abstraction level. Too many low-level tools force the model to reason excessively; a smaller set of higher-level, task-oriented tools often improves performance and reduces token usage.
>
> - Progressive disclosure is a recurring design principle:
>
>   - do not dump all possible context into the prompt;
>   - let the agent retrieve logs, files, references, or data only when needed;
>   - use pagination, filters, and targeted retrieval to prevent context flooding.
>
> - Context engineering is treated as a major production concern, even with large context windows. The goal is to tune the context for the current task rather than maximize the amount of text provided.
>
> - Long-running agents need memory. The agent should be instructed on what to remember, how to write structured notes, and how to reuse those notes across later runs.
>
> - Compaction is recommended when the context grows too large: raw tool outputs can be summarized, stale details removed, and only the relevant trajectory preserved.
>
> - Evaluation is central to agent development. The speakers recommend building a small evaluation suite early, then iterating against it as the agent changes.
>
> - Evaluations should grade outcomes rather than exact paths. A stronger model may solve the task differently, so path-based grading can falsely mark good behavior as failure.
>
> - Graders can be code-based for objective checks, or use a large language model as judge for harder-to-formalize qualities such as style, synthesis quality, or judgment.
>
> - Manual transcript review remains important. When an agent fails, the developer should inspect what the model saw, what tools it called, why it made decisions, and where the trajectory diverged.
>
> - The financial-services section introduces Claude Cowork as a domain workflow surface equipped with skills, plugins, connectors, and Office integrations.
>
> - A **skill** is described as a reusable package of domain instructions, usually centered on a `skill.md` file. It can include natural-language procedures, standard operating procedures, examples, code snippets, reference documents, and evaluation criteria.
>
> - Example finance skills include discounted cash flow modeling, comparable company analysis, three-statement modeling, investment-banking documentation, equity research, due diligence memos, and earnings updates.
>
> - Model Context Protocol integrations let Claude connect to internal and external data systems through a standardized interface. Examples mentioned include S&P Global / Capital IQ-style data, earnings-call transcripts, FactSet estimates, ownership data, and screening tools.
>
> - Claude in Excel is presented as a way to bring the agent into the analyst’s existing workspace. Claude can inspect workbooks, understand formulas, update files, write auditable formulas, and use connected data sources.
>
> - Claude is also described as available in PowerPoint and Word, so generated analyses can flow into common finance deliverables rather than remaining in a chat interface.
>
> - Plugins are framed as a distribution mechanism. A plugin can bundle skills, connector instructions, subagents, hooks, and guardrails, then be shared through an internal marketplace or scheduled to run periodically.
>
> - The speakers treat the “Model Context Protocol versus command-line interface” debate as a false dichotomy:
>
>   - command-line interface access can be simple and efficient for coding or local automation;
>   - Model Context Protocol is often better for enterprise settings needing role-based access control, bounded permissions, and governed integrations.
>
> - The main demo shows a Cowork workflow for updating an Apple financial model after the latest earnings release.
>
> - The demo workflow includes:
>
>   - reading an existing Apple financial model,
>   - invoking a custom earnings-update skill,
>   - retrieving new financial data through connectors,
>   - producing beat/miss analysis,
>   - rolling the model forward,
>   - generating a research note,
>   - creating an updated Excel workbook.
>
> - A key auditability feature is that generated financial numbers in the workbook include citations to the source data used by Claude.
>
> - After Cowork generates the workbook, the analyst can open it in Excel and use Claude in Excel to audit the sheet, fix formula errors, correct references, improve formatting, or extend the model.
>
> - Cowork is positioned as better for document synthesis, non-code deliverables, data-connected workflows, and managing reusable skills through a user interface.
>
> - Claude Code is positioned as better for development-heavy tasks, remote code execution, and software-engineering workflows.
>
> - Skills are expected to change during development. The speakers recommend iterating on skills while building the agent, splitting skills when necessary, and moving instructions out of the system prompt into skills when that improves structure.
>
> - In production, skills should be updated more conservatively, usually in response to observed failures or changed business requirements.
>
> - Hallucinations are described less as an intrinsic mystery and more as a systems-design symptom: the agent often lacks the tool, information, permission, or environment capability needed to complete the task correctly.
>
> - For choosing between a tool, a skill, and a subagent:
>
>   - use a tool for concrete external actions or data retrieval;
>   - use a skill for reusable procedural knowledge;
>   - use a subagent for complex delegated work with its own context and objective.
>
> - Cost control is discussed through model choice, effort level, tool design, and context management. Higher effort should be reserved for harder tasks; simpler tasks can use lower effort settings.
>
> - Subagent context sharing depends on the harness. The parent agent or orchestration layer should explicitly decide what context the subagent receives, rather than passing everything by default.
>
> - The hands-on portion suggests two exercises:
>
>   - use a skill-maker workflow to build a brand-guidelines or Excel-related skill and schedule it in Cowork;
>   - use a Colab notebook to build evaluation suites for agents.
>
> - The speakers recommend verifier patterns for difficult tasks: give the agent explicit criteria or a verifier tool/skill, have it check its own output, and let it iterate until the criteria are satisfied.
>
> - Claude Code’s “route loop” or similar self-verification pattern is described as a way to make Claude repeatedly test its work against a predefined checklist or design specification.
>
> - If Claude does not invoke a newly uploaded skill, the likely issue is the skill instructions. The `skill.md` should state clearly when the skill should be used and include example prompts that trigger it.
>
> - The closing guidance on context management is pragmatic: start with built-in compaction and memory mechanisms, then build custom context management only when evaluations show a clear need.

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2026,
  author = {Bochman, Oren},
  title = {Building {FSI} {Agents} with {Claude}},
  date = {2026-04-28},
  url = {https://orenbochman.github.io/posts/2026/04-28-ODSC-AI-2026-Day-1/talk13.html},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2026. “Building FSI Agents with Claude.” April 28. <https://orenbochman.github.io/posts/2026/04-28-ODSC-AI-2026-Day-1/talk13.html>.
