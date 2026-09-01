## Your MCP Server is Bad (and You Should Feel Bad)

- [slides](https://drive.google.com/file/d/1XkVpRrt5ZKWdvv3tAUXPjfuKtaiqHQmU/view)
- [Jeremiah Lowin](https://www.linkedin.com/in/jlowin/)

### Notes

- One key takeaway is that search tool can keep the MCP costs in check by removing the need for giving the llm a full list of MCP servers.

## The Spectrum of Agentic Coding: From Vibe Coding to High-quality Software Engineering

- Y.K. Sugi
  - [LinkedIn](https://www.linkedin.com/in/ykdojo/)
- Eventual
- [slides](https://ykdojo.github.io/claude-code-tips/content/spectrum-slides.html)

> **NOTE:**
>
> - **Talk topic:** YK presents “the spectrum of agentic coding,” based on roughly three years of work with AI coding agents, including an early ChatGPT-plugin tool called **Kaguya** that could edit local files and run commands.
>
> - **Core thesis:** “Vibe coding” and traditional software engineering are not opposites. They sit on a continuum: higher speed usually means lower assurance, while higher quality requires more structure, review, testing, and human understanding.
>
> - **Level 1 — Vibe coding**
>
>   - Let the AI generate and modify code quickly.
>   - Useful for prototypes, one-off scripts, experiments, and exploratory work.
>   - Weakness: low maintainability, minimal testing, little version-control discipline, and shallow understanding of the generated code.
>
> - **Level 2 — Agentic coding with discipline**
>
>   - Adds basic engineering hygiene: Git, GitHub, file-level understanding, basic security precautions, and some tests.
>   - The developer may not understand every line, but should understand what each file is for and how files relate.
>
> - **Level 3 — Agentic software engineering**
>
>   - Adds stronger software-engineering practices: verified tests, pre-commit hooks, continuous integration, manual testing, end-to-end tests, regression checks, and AI-assisted pull-request review.
>   - The human develops functional-level understanding: what each function or class does, even if not every line is manually inspected.
>
> - **Level 4 — High-quality software engineering**
>
>   - AI-generated output should be indistinguishable from strong human-written production code.
>   - Requires line-by-line understanding, self-reflection loops, and interactive AI code review.
>   - The developer should ask the AI why specific lines exist, what trade-offs were considered, and whether alternatives were evaluated.
>
> - **Interactive AI review workflow**
>
>   - For a large pull request, the speaker suggests asking the AI to:
>
>     - order files by review priority,
>     - summarize each file,
>     - identify what to pay attention to,
>     - answer follow-up questions interactively.
>
>   - This is contrasted with weaker “one-shot” AI review, where the AI gives a single broad review without dialogue.
>
> - **Advanced uses of AI in engineering**
>
>   - AI can replace or augment web research when grounded in current sources.
>   - AI can control browsers or command-line tools to test applications.
>   - AI can accelerate learning by explaining generated code and suggesting improvements, not merely producing output.
>
> - **Main warning about “slop”**
>
>   - Low-quality AI code is not mainly a tool problem; it is often a user-process problem.
>   - More generated code or more tokens do not automatically imply worse quality. Quality depends on how the AI is used, reviewed, tested, and constrained.
>
> - **Dynamic Vibe Adjustment**
>
>   - The speaker adds a fifth idea: start with vibe coding, then increase rigor only when the problem demands it.
>   - When something fails, move deeper into the code, organize the codebase, or isolate the problem in a simplified environment.
>
> - **Example from game development**
>
>   - The speaker tried to implement a Unity feature where enemies behind logs become partially transparent.
>   - Vibe coding failed in the full project.
>   - The solution was to ask Claude Code to copy the project, strip away irrelevant features, build a minimal environment, solve the feature there, and then transfer it back.
>
> - **“Vibe iceberg” metaphor**
>
>   - Surface-level vibe coding is fine until it breaks.
>   - Below the surface are deeper practices: code understanding, refactoring, isolation, testing, review, and debugging.
>
> - **Use in design thinking and business contexts**
>
>   - AI is useful during design thinking for fast prototyping and market/user research.
>   - The speaker emphasizes that product ideation is not only about problem and solution fit, but also about understanding current market context.
>
> - **Managing expectations in teams**
>
>   - Teams should explicitly define how much understanding is expected:
>
>     - vibe-level,
>     - file-level,
>     - function-level,
>     - line-by-line.
>
>   - They should also define expected review quality and change those norms over time as needed.
>
> - **Practical takeaway:** Master all levels of agentic coding and choose the right level for the task. Use vibe coding for speed when risk is low; use disciplined, test-heavy, interactive review workflows when correctness, maintainability, or safety matters.

## Using Personal AI in 2026(OpenClaw)

- [Brian Turcotte]()
- [Brendan O\`Leary]()
- Kilo
- [Slides](https://docs.google.com/presentation/d/1Y37ueidWMzyYxevtidcssIRqvBJPY3M4ZW-3VKc4bmo/)

> **NOTE:**
>
> - **Core idea:** Personal AI is not just a chat interface like ChatGPT, Claude, or Gemini. It is an always-on agent that can run in the background, access tools, remember context, and take actions on the user’s behalf.
>
> - **Main distinction:**
>
>   - Chat AI is like a smart colleague you call when needed.
>   - Personal AI is like a junior employee with its own desk, accounts, tools, memory, and recurring tasks.
>
> - **Demo platform:** The session uses **OpenClaw**, an open-source personal AI agent harness, through Kilo’s managed hosted layer, **Kilo Claw**.
>
> - **Why this is now practical:** Large language models have improved enough, and the surrounding open-source agent infrastructure has matured enough, to support agents that can perform real digital work.
>
> - **OpenClaw mental model:**
>
>   - The agent runtime executes tasks.
>   - Tools and integrations give access to external systems.
>   - Markdown files provide persistent memory and operating rules.
>
> - **Important OpenClaw files:**
>
>   - `soul.md`: the agent’s behavioral core, tone, personality, and hard constraints.
>   - `memory.md`: long-term facts, preferences, and decisions.
>   - Daily memory logs: task history, decisions, and ongoing work.
>   - `user.md`: information about the user, such as preferences, timezone, and briefing style.
>   - `agent.md`: the operating contract: goals, priorities, quality bar, and escalation rules.
>
> - **Key warning:** The agent is “the state of its files,” not the emotional history of the conversation. If an instruction is not written into the right persistent file, the agent may not retain it.
>
> - **Skills and tools:**
>
>   - Skills are discrete instruction sets that teach the agent specific capabilities.
>   - Examples include web search, PDF reading, spreadsheet creation, charting, API interaction, and platform-specific workflows.
>   - Workflows may combine many skills.
>   - ClawHub is presented as the skills marketplace for OpenClaw.
>
> - **Heartbeat / cron jobs:**
>
>   - Scheduled tasks make the agent proactive.
>   - Examples include daily briefings, weekly planning, market monitoring, email triage, and meeting preparation.
>
> - **Guardrails recommended before automation:**
>
>   - Never allow the agent to send, publish, post, or externally communicate without explicit approval.
>   - Require the agent to write important preferences and rules into persistent files.
>   - Give the agent its own dedicated accounts rather than access to the user’s personal accounts.
>   - Use restricted bot accounts for email, GitHub, project management, and other services.
>   - Store credentials in a password manager rather than pasting them into chat.
>
> - **Operational safety examples:**
>
>   - Bot email addresses can follow a pattern such as `name.bot@domain`.
>   - Bot GitHub accounts may create issues and pull requests but should not approve or merge them.
>   - Bot accounts should be easy to revoke during offboarding.
>
> - **Setup demo:**
>
>   - Create a Kilo Claw bot.
>   - Choose an avatar and personality.
>   - Optionally connect Telegram, Discord, or Slack as chat channels.
>   - Connect platforms such as GitHub, Google, 1Password, or other services.
>   - Edit `agent.md` directly with guardrails such as anti-looping, approval requirements, no deletion, date verification, and concise communication.
>
> - **Anti-looping rule:** If the agent attempts the same action twice with the same result, it should stop and report the issue rather than wasting tokens or credits.
>
> - **Starter skills mentioned:**
>
>   - Summarization.
>   - Documentation reading.
>   - Quarto Markdown / markdown navigation.
>   - Skill-security skills that help the agent evaluate whether other skills are safe to install.
>
> - **Starter automation: daily briefing**
>
>   - Runs every morning.
>   - Pulls from trusted blogs, X accounts, calendar, email, weather, and memory.
>   - Summarizes relevant news.
>   - Suggests two or three tasks the agent could complete without user involvement.
>
> - **Meeting-prep use case:**
>
>   - The agent reads calendar attendees, attached documents, and customer relationship management context.
>   - It sends a briefing shortly before the meeting so the user enters prepared.
>
> - **Personal customer relationship management workflow:**
>
>   - The agent scans email and calendar interactions.
>   - It tracks names, emails, companies, roles, and last interaction dates.
>   - It filters out marketing, automated notifications, cold outreach, and newsletters.
>   - It flags people who have not been contacted for a specified period, such as 14 days.
>
> - **Cost-management advice:**
>
>   - First get the workflow working, then optimize cost.
>   - Use cheaper models for routine retrieval or preprocessing.
>   - Use stronger models for summarization, debugging, or complex reasoning.
>   - Track usage with commands such as `/usage`.
>   - Configure daily usage summaries to monitor unexpected costs.
>
> - **Developer use cases:**
>
>   - Monitor GitHub repositories.
>   - Watch for continuous integration failures.
>   - Triage incoming issues.
>   - Draft issue responses.
>   - Summarize pull requests.
>   - Generate fixes, run tests, and submit pull requests for review.
>
> - **Agent as tool bridge:**
>
>   - The agent can connect email, Slack, Calendar, GitHub, Linear, Notion, and other systems.
>   - It can turn an email into a Linear task, link it to a GitHub issue, or draft a Slack message.
>   - The broader idea is to abstract away app-specific interfaces and let the agent coordinate across systems.
>
> - **Advanced example: smart dashboard**
>
>   - Brian describes a custom dashboard connected to Kilo Claw through a lightweight Supabase database.
>   - The agent writes generated artifacts, content drafts, and code outputs into the database.
>   - The dashboard lets him track multiple concurrent agent tasks.
>
> - **Newsletter automation example:**
>
>   - The agent pulls from trusted X accounts, blogs, Slack history, email, calendar, company blog posts, YouTube videos, and event calendars.
>   - It uses an HTML template stored in a connected GitHub repo.
>   - It generates a full newsletter draft that can be reviewed and sent.
>
> - **Final takeaway:** Personal AI in 2026 means gradually offloading repetitive administrative and digital work—email triage, calendar prep, meeting prep, content drafting, newsletter creation, issue triage, and cross-tool coordination—while keeping human review for creative, strategic, and high-stakes decisions.

## From Model Security to Mission Security: Why AI Fails in 2026

- Parthasarathi Chakraborty
  - [LinkedIn](https://www.linkedin.com/in/infosec-exec-partha/)
- Varun Agaskar
  - [LinkedIn](https://www.linkedin.com/in/varunagaskar/)
- Broadridge
- [slides](https://docs.google.com/presentation/d/1h51VhRvP8fopEag8XHH-Id4RsOSW6oQo)

> **NOTE:**
>
> - The speakers argue that “Why AI Fails in 2026” should really be understood as “how to prevent AI initiatives from failing,” especially in enterprise security and agentic artificial intelligence systems.
>
> - **Central thesis:**
>
>   - The main risk is no longer only **model security**: bias, poisoning, hallucination, bad answers. as we now have a bigger issue is **mission security**: what happens when an AI agent can take real actions across GitHub, Jira, cloud storage, infrastructure, tickets, and enterprise systems.
>
> - **Shift in AI usage:**
>
>   - Earlier AI was mostly an answering tool: ask a question, get advice.
>   - Now organizations want AI agents to execute work: modify infrastructure, manage tickets, analyze vulnerabilities, create pull requests, and interact with other agents.
>   - This raises the risk from “bad output” to “bad action.”
>
> - **Example risks discussed:**
>
>   - A prompt injection in a Jira ticket could cause an agent to clone or expose the wrong repository.
>   - An agent crawling Confluence or SharePoint might misread an old instruction such as “open port 22 to 0.0.0.0/0” and apply it in production.
>   - A vulnerability-management agent with broad Jira permissions might close thousands of tickets, including real unresolved vulnerabilities, because it overgeneralizes “duplicate” or “redundant.”
>
> - **Model security vs. mission security:**
>
>   - **Model security** focuses on whether the model behaves correctly.
>   - **Mission security** focuses on the complete workflow: inputs, context, interpretation, permissions, actions, observability, and recovery.
>   - In agentic systems, the model is only one component of a larger operational system.
>
> - **The “core four” pillars of AI mission security:**
>
>   - **Identity:** Treat AI agents as machine identities, not anonymous tools or shared API keys.
>   - **Action governance:** Define what the agent may do, when it may do it, and when approval is required.
>   - **Context control:** Restrict what the agent can read; context should follow a need-to-know principle.
>   - **Execution monitoring:** Track what the agent actually does, not just what it was designed to do.
>
> - **The five implementation principles implied by the Jira example:**
>
>   - Give each AI agent a unique identity with short-lived credentials.
>   - Use least privilege rather than senior-engineer or shared-admin access.
>   - Put high-impact actions behind approval workflows.
>   - Enforce policies at runtime, not only at design or annual review time.
>   - Maintain observability and auditability: who acted, what action occurred, and why it occurred.
>
> - **Recommended architecture:**
>
>   - Start with user input.
>   - Pass through the AI/model layer.
>   - Route actions through a policy engine.
>   - Apply identity, context limits, action controls, monitoring, and auditability across the full path.
>
> - **Security concern for vulnerability management:**
>
>   - AI may let attackers discover and chain vulnerabilities much faster.
>   - Low or medium vulnerabilities may become critical when combined.
>   - Vulnerability prioritization may need to shift from isolated severity scores to exploit chains and systemic exposure.
>
> - **AI is not always the right tool:**
>
>   - The speakers warn against applying AI to everything.
>   - Deterministic problems may be better solved with classic automation, robotic process automation, scripts, or workflow tools.
>   - AI is more appropriate for ambiguous, behavioral, or unknown-pattern problems.
>
> - **Operational recommendations:**
>
>   - Use least privilege.
>   - Avoid long-lived credentials.
>   - Rotate tokens and credentials.
>   - Prevent agents from crawling unnecessary data.
>   - Add checks and balances before actions.
>   - Monitor continuously.
>   - Keep a kill switch so bad actions can be stopped, not merely detected.
>
> - **Closing framing:**
>
>   - AI security should follow the lifecycle: identify, protect, detect, respond, and recover.
>   - The goal is not merely securing the model, but securing the entire AI-enabled mission.
>
> - **Q&A note on observability and open source tools:**
>
>   - The speakers say they use a hybrid of open source and commercial tools.
>   - Open source tools are more acceptable for pilots, testing, or red teaming.
>   - For production security logging and monitoring, they prefer mature, supported, industry-specific tooling because remediation speed matters when vulnerabilities appear.

### Reflections

- security talks rarely have high entropy content - once you’ve heard a talk about security best practices you are not likely to hear anything new or surprising.
- agents live in a tension between high risk and low effort. It is hard to dedicate an environment with minimal risk and friction for an agent to operate in for small insignificant tasks, yet without autonomy and higher levels of prvilage we are unlikely to get the full benefits of agents.

## Enhancing Context Engineering with Agentic Integration into Vector Database Queries

- [Scott Askinosie](https://www.linkedin.com/in/scott-askinosie/)
- Contextual AI
- [slides](https://docs.google.com/presentation/d/1CLqqynGftaVJv6569hintj7-r4_CK9Qh)

> **NOTE:**
>
> - **Topic and goal**
>
>   - The workshop explains how to build an agentic retrieval system around a vector database.
>   - The arc is: vector databases → filtered vector search → retrieval-augmented generation/context engineering → agents → orchestration → evaluation and monitoring.
>
> - **Why vector search matters**
>
>   - Traditional search evolved from keyword matching to BM25 and then to AI-generated search answers.
>   - Keyword search fails when literal terms do not capture user intent, or when ads and search-engine optimization distort results.
>   - Embeddings solve part of this by turning text, images, or product metadata into high-dimensional vectors that can be compared semantically.
>
> - **Core vector database concepts**
>
>   - A vector is described simply as a list of numbers.
>   - Embedding models map semantically related objects close together in vector space.
>   - Text and images can occupy the same semantic space: for example, the word “banana” and an image of a banana may be close.
>   - The embedding model used to create the database must also be used for querying; mixing models can produce bad retrieval because different models encode meaning differently.
>   - Model choice matters: some models are better for legal, multilingual, or domain-specific retrieval.
>
> - **Hands-on vector database notebook**
>
>   - The workshop uses **[Qdrant](https://qdrant.tech/)** as the vector database because it exposes the mechanics more manually than some higher-level systems.
>   - Environment setup includes Docker, Qdrant, an OpenAI API key, and notebook dependencies.
>   - The dataset is an H&M e-commerce product dataset from Hugging Face.
>   - The full dataset has about 105,000 products, but the workshop uses a smaller subset of 2,500 products for speed.
>   - Product metadata is converted into text descriptions, embedded into 1,536-dimensional vectors, and inserted into Qdrant.
>   - UUIDs are used so repeated inserts update existing objects rather than duplicating them.
>   - Batch insertion is discussed as a function of network speed and local setup.
>
> - **Vector search, filtering, and RAG**
>
>   - Basic vector search retrieves semantically similar products.
>   - Metadata filtering is necessary for strict constraints such as color, gender, category, or department.
>   - Vector similarity alone may not reliably respect categorical constraints like “blue” or “women’s.”
>   - Filtered search works more like SQL: conditions such as “must have color group name white” are applied alongside vector search.
>   - The workshop then demonstrates retrieval-augmented generation, where retrieved product context is passed to a generative model.
>   - Manual filtering is powerful but awkward for users, because real users do not want to fill in long structured forms.
>
> - **Agents as an abstraction layer**
>
>   - An agent is framed as a large language model with tools, memory, and the ability to reason-act-observe.
>   - A basic agent can be created mostly through system instructions.
>   - Statelessness is demonstrated: without memory, the model does not remember earlier turns.
>   - Memory is added by passing prior conversation state or structured context back into the model.
>   - Long-term agent memory is presented as an active research area, with references to methods such as Zettelkasten-like memory and newer autonomous memory approaches.
>
> - **Dynamic filtering agent**
>
>   - The agent is given tools to inspect the vector database schema and construct filters dynamically.
>   - Instead of requiring the user to specify filters manually, the agent maps natural language into structured query filters.
>   - Example: “Show me white tops for women” becomes filters over gender, color group, and product type.
>   - Negative constraints such as “clothing, but not black” are also discussed.
>   - The speaker notes that prompts often need iteration when the agent becomes too explicit or chooses suboptimal filters.
>
> - **Query decomposition**
>
>   - Query decomposition is presented as a central failure point in retrieval systems.
>
>   - A user query may contain several distinct requests, constraints, or intents.
>
>   - If the whole query is embedded once, the vector search may retrieve results from the “middle” of several semantic clusters and answer none of the subqueries well.
>
>   - The better pattern is:
>
>     - decompose the user query into subqueries;
>     - retrieve separately for each subquery;
>     - pass the retrieved results plus the original query to a generative model;
>     - synthesize a unified answer.
>
>   - This is positioned as a major improvement over naive retrieval-augmented generation.
>
> - **Agent orchestration**
>
>   - A single general-purpose model struggles when user intent changes mid-conversation.
>
>   - Example: a customer complains about overcharging, a torn garment, and a wrong address in one message.
>
>   - These are different intents: billing, support, and account/profile update.
>
>   - An orchestrator can classify intent, split the request, route parts to specialized sub-agents, and compose a unified response.
>
>   - The workshop uses agents such as:
>
>     - intent classifier;
>     - orchestrator;
>     - billing agent;
>     - product agent;
>     - support agent;
>     - unknown/clarification path.
>
> - **Slot filling and clarification**
>
>   - The orchestrator checks whether it has enough information to route the task.
>   - “Slots” are required pieces of information, such as account number, product type, billing address, or specific product issue.
>   - If information is missing, the agent asks targeted follow-up questions.
>   - This improves both system reliability and user experience, because users often begin with vague requests like “I need help.”
>
> - **Architecture over code**
>
>   - The speaker emphasizes that modern LLMs understand agentic orchestration patterns well.
>   - The critical work is specifying the architecture: departments, intents, sub-agents, schemas, tools, and routing logic.
>   - LLMs can help generate tool functions and agent instructions, but the system still needs careful testing.
>   - The distinction between “vibe coding” and more disciplined AI-assisted engineering is discussed: the key difference is whether the developer gives precise architecture, constraints, and testable specifications.
>
> - **Failure points in production**
>
>   - Retrieval may fail if the vector database changes, filters break, or the wrong collection is queried.
>   - Orchestration may fail if intent classification is wrong or if required slots are missing.
>   - Sub-agents may produce irrelevant, incomplete, or hallucinated answers.
>   - Generative models may ignore retrieved context or take too much creative license.
>   - The speaker argues that each node in the pipeline needs visibility and evaluation.
>
> - **Evaluation and monitoring**
>
>   - Retrieval evaluation metrics discussed include:
>
>     - mean reciprocal rank;
>     - normalized discounted cumulative gain;
>     - precision at k;
>     - whether expected products appear in the top results.
>
>   - Generative evaluation checks whether the model uses the provided context, answers the question, and avoids hallucination.
>
>   - Model comparison can also support cost optimization: cheaper models may perform as well as larger models for specific tasks.
>
>   - Real-time observability is recommended so developers can see exactly where failures occur.
>
> - **OPIC / Comet monitoring**
>
>   - OPIC is introduced as an open-source tool for monitoring agentic and retrieval systems.
>   - It can track retrieval metrics, context precision, generative relevance, and hallucination-like behavior.
>   - It can use an LLM judge to score outputs against the input and context.
>   - The judge should be customized with specific evaluation instructions.
>   - Sampling can be done on every request or on a subset of requests, depending on cost and need.
>
> - **Final takeaway**
>
>   - The workshop’s main message is that robust context engineering requires more than a vector database.
>   - A production-grade system needs semantic retrieval, metadata filtering, query decomposition, intent orchestration, slot filling, specialized agents, and continuous evaluation.
>   - The strongest pattern is not “one big agent,” but a structured system of smaller agents with clear roles, tools, routing, and observability.

## Implementing a Self-service Data Platform

- Andrew Jones
- [Slides](https://docs.google.com/presentation/d/1rHKRY2LozyAQHIGtlPLBZG3MqXtQA0LQ/)

> **NOTE:**
>
> - The talk explains how to build a **self-service data platform** that removes the data engineering team as an organizational bottleneck.
>
> - The speaker argues that data platforms can influence not only engineers, but the whole organization’s ability to create, manage, govern, and monetize data and artificial intelligence.
>
> - A common failure mode is that data engineers become trapped in low-value support work:
>
>   - answering why pipelines failed,
>   - explaining why numbers changed,
>   - approving data access,
>   - adding fields or integrations,
>   - handling urgent requests from other teams.
>
> - This bottleneck slows down the use of data and artificial intelligence, and creates hidden opportunity costs when teams abandon valuable ideas because the data work is blocked.
>
> - When teams cannot get what they need from central data engineering, they often create **shadow data engineering** or **shadow IT**:
>
>   - their own pipelines,
>   - their own analytics tools,
>   - their own unofficial data teams.
>
> - Simply hiring enough data engineers to handle every request is usually unrealistic, so the better model is to learn from **platform engineering** in software development.
>
> - The speaker compares data engineering bottlenecks to older DevOps bottlenecks:
>
>   - centralized infrastructure teams once controlled deployments, observability, cloud provisioning, and production troubleshooting;
>   - software teams had to raise tickets and wait;
>   - platform engineering changed this by giving developers self-service capabilities.
>
> - The proposed equivalent for data is a platform that abstracts away:
>
>   - pipeline creation,
>   - data management,
>   - governance,
>   - regulatory compliance,
>   - observability,
>   - data sharing.
>
> - The goal is not to eliminate data engineers, but to shift them from ticket handling to building reusable platform capabilities once for the whole organization.
>
> - The speaker describes a first failed or partially successful attempt called **Data Platform Gateway**:
>
>   - producers submitted Avro schemas;
>   - the platform validated incoming data;
>   - tables were provisioned in the data warehouse;
>   - data could then be consumed by data scientists and engineers.
>
> - That first attempt had three main problems:
>
>   - it forced software engineers to learn Avro, which did not fit their Ruby-heavy stack;
>   - it did not meet teams where they already worked;
>   - the data platform remained a central handoff point, so teams still did not fully own their data.
>
> - As a result, the system had some adoption, but did not remove the need for tickets or central data engineering intervention.
>
> - The more successful implementation used **data contracts**.
>
> - A data contract is described as a human- and machine-readable document that captures the context needed to automate data platform capabilities.
>
> - In the successful architecture:
>
>   - the data contract lived in the producing team’s Git repository;
>   - it sat alongside the code that generated the data;
>   - it used tooling and languages familiar to the team;
>   - libraries made it easy to write data to the warehouse;
>   - the same contract provisioned the required warehouse tables.
>
> - This design worked better because it made the platform an enabler rather than a destination or gatekeeper.
>
> - Producing teams retained ownership and autonomy over their data:
>
>   - changing schemas,
>   - managing access,
>   - publishing data products,
>   - maintaining their part of the warehouse.
>
> - The speaker reports that this model eventually supported more than 200 data contracts in production without human data-engineering intervention for deployment and management.
>
> - Data contracts then became the basis for adding further platform capabilities:
>
>   - schema-based table provisioning,
>   - data quality monitoring,
>   - service-level objectives such as timeliness,
>   - observability alerts to owners,
>   - data quality rules,
>   - automated anonymization of sensitive fields.
>
> - The central insight is that once enough context is captured in a contract, many data-platform operations can be automated.
>
> - The team’s role shifted from being a data engineering team to being a **data platform team** that enables creation, management, sharing, governance, and reuse of data products.
>
> - In the question-and-answer section, the speaker clarifies the governance model:
>
>   - central experts or governance bodies still define policies;
>   - data owners remain responsible for their own data because they understand it best;
>   - the platform automates policy implementation.
>
> - The governance example is anonymization:
>
>   - data owners classify sensitive fields;
>   - central teams define policy;
>   - the platform applies anonymization automatically.
>
> - The final message is that effective self-service data platforms combine ownership, automation, and governance by embedding policy and operational capabilities into the platform rather than relying on manual tickets.

## Efficient Language Models via Quantization

- [Tim Dettmers](https://www.linkedin.com/in/timdettmers/)
- CMU
- Ai2
- [slides](https://docs.google.com/presentation/d/1FbFWadAHww42Os_MmqrTC24holhjIvGa)
- Papers:
  - [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)
  - [QLoRA: 4-bit Finetuning](https://arxiv.org/abs/2305.14314)

> **NOTE:**
>
> - The talk explains **quantization for large language models**, focusing on two practical problems:
>
>   - quantized inference, especially the role of outliers;
>   - low-precision fine-tuning, especially **QLoRA**.
>
> - The motivation is that language models became too large for many researchers to use or fine-tune on ordinary hardware. Quantization is presented as a way to reduce memory cost while preserving model quality and generation speed.
>
> - For inference, a good quantization method must satisfy three constraints:
>
>   - reduce memory footprint;
>   - preserve model performance;
>   - maintain fast generation.
>
> - Demmers argues that most transformer cost comes from matrix multiplication:
>
>   - attention and feed-forward layers dominate memory and compute;
>   - therefore, efficient large-model deployment mostly means approximating matrix multiplication without destroying quality.
>
> - Several approximation strategies are contrasted:
>
>   - **low-rank projection** saves compute but often loses too much quality;
>   - **sparsification** can preserve quality but is hard to exploit efficiently on current hardware;
>   - **quantization** is more practical because it reduces precision while keeping dense matrix operations hardware-friendly.
>
> - Basic quantization is introduced as mapping high-precision values, such as 16-bit values, into lower-precision bins, such as 8-bit or 4-bit integers. Proper rescaling is essential; otherwise many bins go unused and information capacity is wasted.
>
> - The central obstacle is **outliers**:
>
>   - neural networks contain rare large activation values;
>   - unlike many statistical settings, these outliers cannot simply be discarded;
>   - if they are included naively in the quantization range, they force most quantization bins away from the bulk of the distribution, causing large error.
>
> - The speaker describes discovering a scale-dependent structure in language model outliers:
>
>   - smaller models show relatively random outlier patterns;
>   - larger models develop stable outlier dimensions;
>   - at sufficient scale, outliers appear reliably in the same dimensions across layers and inputs.
>
> - This led to the **LLM.int8** approach:
>
>   - run data through the model to identify outlier dimensions;
>   - handle the tiny outlier subspace in 16-bit precision;
>   - handle the remaining approximately 99.9% of computation in 8-bit precision;
>   - add the results back together to recover near-16-bit model quality with much lower memory use.
>
> - The broader lesson is that successful quantization depends on exploiting structure, not merely lowering precision uniformly.
>
> - The talk then shifts to **performance density per bit**:
>
>   - the right question is not simply “how low can precision go?”;
>   - it is “which precision maximizes model quality for a fixed memory budget?”
>   - empirically, around 4-bit precision often gives the best tradeoff; below that, for example at 3-bit, quality tends to collapse.
>
> - For fine-tuning, the speaker explains why naive 4-bit training fails:
>
>   - gradients may still carry useful information;
>   - but direct 4-bit weight updates are too coarse and unstable;
>   - with only 16 representable values, small learning updates cannot be represented well.
>
> - **QLoRA**, or Quantized Low-Rank Adaptation, solves this by:
>
>   - keeping the base model frozen in 4-bit precision;
>   - adding small 16-bit trainable adapter matrices;
>   - backpropagating through the quantized model but updating only the high-precision adapters.
>
> - This makes fine-tuning dramatically cheaper:
>
>   - the speaker claims roughly 17× efficiency improvement compared with regular fine-tuning;
>   - large-model fine-tuning can move from multi-server data-center hardware to a single consumer GPU in some cases.
>
> - The speaker connects this to the **[Bits and Bytes](https://github.com/TimDettmers/bitsandbytes)** library:
>
>   - it implements these quantization methods as open-source software;
>   - the speaker emphasizes that publishing papers is insufficient unless the methods are usable by ordinary researchers.
>
> - In the question section, the speaker clarifies mixed-precision decomposition:
>
>   - outlier dimensions are detected once before deployment;
>   - those dimensions are computed in 16-bit;
>   - the rest of the matrix multiplication is computed in 8-bit;
>   - the outputs are recombined.
>
> - On choosing between **LoRA** and **QLoRA**:
>
>   - QLoRA is more memory efficient;
>   - the speaker says its quality is generally comparable to LoRA when tuned correctly;
>   - future versions of Bits and Bytes are expected to make k-bit QLoRA more stable, making QLoRA the default practical choice.

### Reflection

If one could just quantize all the weights to 1 bit then everyone would. This is something that people have been talking about for a long time. However it doesn’t just work and Demmers explains a bit about why. Intuitively, numeric analysis teaches us how approximating matrix multiplications leads to compounding errors when the matrix is near singular. Also outliers play a role because they can dominate the output of a layer, so if they are quantized too coarsely, the model’s behavior can change drastically.

## Building Evaluation Systems for AI Coding Agents at Scale

- Building Evaluation Systems for AI Coding Agents at Scale by
- [Karen Zhou](https://www.linkedin.com/in/karenzhou0/),
- Anthropic

> **NOTE:**
>
> - **Talk topic**
>
>   - Karen from Anthropic’s Cloud Code team discusses how to evaluate whether AI coding assistants are genuinely helping developers.
>   - The central question is not merely whether an agent passes a benchmark, but whether it improves real developer productivity and avoids subtle failures.
>
> - **What makes a good evaluation**
>
>   - A useful evaluation should be:
>
>     - **Reliable**: repeated runs should produce stable results.
>     - **Sensitive**: scores should change when the model or harness improves or regresses.
>     - **Representative**: tasks should match real user workflows, not only benchmark distributions.
>     - **Fast enough**: slow evaluations reduce iteration speed.
>     - **Interpretable**: failures should explain what went wrong, not only report pass/fail.
>     - **Affordable**: expensive evaluations are often dropped when budgets tighten.
>
> - **Evaluation design principles**
>
>   - Avoid tasks that are either too hard or too saturated, because both stop distinguishing model quality.
>   - Prefer deterministic, programmatic verification when possible.
>   - Use large language model graders only when outputs are genuinely open-ended.
>   - Be explicit about paths, formats, success criteria, and assumptions, because agents do not infer context the way humans do.
>   - Treat benchmark results as measuring the combination of **model + harness**, not the model alone.
>
> - **Limits of public benchmarks**
>
>   - Public coding benchmarks usually measure whether the final output is correct.
>   - They often miss how the agent behaved while reaching the answer.
>   - A model can pass tests while still producing a frustrating user experience.
>
> - **Behavioral failures benchmarks often miss**
>
>   - **Laziness**
>
>     - The agent stops early, truncates work, or says it will leave the rest to the user.
>     - This can sometimes be detected through token usage and task-completion signals.
>     - However, stopping early is not always wrong, for example when safety requires refusal.
>
>   - **Instruction-following failures**
>
>     - The agent ignores explicit constraints from the user prompt or project files.
>     - These failures can be subtle because the final output may look plausible unless the full transcript is checked.
>
>   - **Over-engineering**
>
>     - The agent adds unnecessary abstractions, performs broad refactors, or optimizes prematurely.
>     - The result may be technically correct but misaligned with the user’s actual request.
>
> - **Why real-world evaluation is harder**
>
>   - Real codebases evolve and contain inconsistent instructions.
>   - Context windows shift between sessions.
>   - Agents operate over multiple turns, not just single prompts.
>   - Errors can compound over long horizons.
>   - Frontier models may saturate existing benchmarks, making them less useful for real-world differentiation.
>
> - **Human-aligned grading**
>
>   - Automated graders are only useful if they stay aligned with human judgment.
>
>   - Common grader failure modes include:
>
>     - Ambiguous rubrics.
>     - Lack of surrounding code context.
>     - Bias toward familiar coding styles.
>     - Preference for outputs resembling the grader model’s own style.
>
>   - The proposed remedy is a curated golden set of human-labeled examples, refreshed over time.
>
>   - Multiple independent graders and blind human spot checks help detect drift.
>
>   - Rubrics should be tracked per behavioral dimension, not only as one overall score.
>
> - **Multi-agent evaluation challenges**
>
>   - Single-agent evaluation is comparatively simple: one model, one prompt, one output.
>
>   - Multi-agent systems introduce harder questions:
>
>     - Which agent caused a failure?
>     - Was the problem in execution, delegation, or coordination?
>     - Did agents duplicate work, contradict one another, or fail to share context?
>     - Can the system be replayed deterministically?
>
>   - Evaluation must consider both end-to-end task success and per-agent behavior within each agent’s role.
>
> - **Closing the loop**
>
>   - Evaluation should feed back into model training, harness design, and product improvement.
>   - Otherwise, it is only measurement, not an improvement system.
>   - Anthropic’s described loop uses internally generated evaluation data, replay, grading, human feedback datasets, and harness/model updates.
>   - The speaker explicitly notes that user conversation data is not used in this loop.
>
> - **Future direction**
>
>   - Coding agents may become more self-evaluating and self-improving.
>   - Self-evaluation is promising but risky because models are often poor at recognizing what they do not know.
>   - Long-horizon evaluation remains an open research problem, especially across hours, days, sessions, and multiple agents.
>   - Mature evaluation infrastructure may become a trust signal for whether coding agents are safe and useful in production.
>
> - **Main takeaways**
>
>   - Not all evaluations are high-signal.
>   - Public benchmarks are useful but incomplete.
>   - Behavioral evaluations are needed for laziness, instruction following, and over-engineering.
>   - Automated graders must be calibrated against human judgment.
>   - Multi-agent systems require new evaluation methods.
>   - The most valuable evaluation systems form a feedback flywheel into training and harness improvement.

## Driving Data Quality with Data Contracts

- Andrew Jones
- Book: Driving Data Quality with Data Contracts

> **NOTE:**
>
> - The session is an interview with Andrew about his book, *Driving Data Quality with Data Contracts*.
> - The book is framed not only as a technical guide to data contracts, but as a broader guide to improving organizational data quality, governance, and collaboration between data producers and consumers.
> - Andrew says the book grew out of internal work at his company, followed by blog posts that attracted attention from practitioners facing similar data-quality problems.
> - A data contract is described as a human- and machine-readable document that explains a dataset well enough for people and tools to use it reliably.
> - The central analogy is to an Application Programming Interface contract: it gives producers and consumers a shared, version-controlled understanding of what is being provided and how it may change.
> - The main value of data contracts is confidence: consumers can build data products, analytics, and machine-learning systems without fearing silent upstream schema or quality changes.
> - Two common implementation mistakes are highlighted:
>   - Treating “contract” as enforcement and control, which creates friction with producers.
>   - Starting with tools and automation before building organizational buy-in.
> - Andrew emphasizes that data contracts are primarily a cultural and operational change, supported by technology rather than solved by technology alone.
> - Data contracts improve the producer–consumer relationship by forcing explicit negotiation over requirements such as timeliness, structure, ownership, and change management.
> - In Andrew’s example from a payments company, unreliable internal data was blocking machine-learning and product use cases; introducing data contracts improved reliability enough to support revenue-generating product features.
> - Data contracts fit naturally with modern architectures such as lakehouses and streaming platforms because they are not tied to a specific technology stack.
> - They are especially important for data mesh, where federated governance and self-service platforms require machine-readable context about datasets.
> - Automation becomes essential at scale because data engineering teams otherwise become bottlenecks; contracts provide the metadata and rules automation needs to act correctly.
> - For tooling, Andrew recommends looking at the Open Data Contract Standard as a useful starting point, while noting that standards and vendor tooling are still immature.
> - Smaller teams and startups can start by defining a simple contract format, working closely with data producers, and solving one concrete pain point before scaling.
> - Andrew argues that startups may even have an advantage because teams are closer together and producers can become active owners of the solution.
> - Looking ahead, he expects data contracts to become more important as artificial intelligence and production data products require higher reliability.
> - He also hopes standards mature so that contracts can plug into catalogs, quality tools, and platforms without custom glue code or vendor lock-in.
> - The final lesson from the book is to challenge old assumptions about data work, especially the belief that data quality can only be fixed downstream.
> - Andrew’s core claim is that reliable data must be improved at the source, by treating data with the same seriousness, ownership, documentation, and change management as software APIs.

## The Ralph Wiggum Phenomenon Evolving Agentic Coding

- [Ray Myers](linkedin.com/in/cadrlife)
- OpenHands
- [slides](https://drive.google.com/file/d/1uKEFfzcDzScShC1gdUyvddPqHrLutpMI/)

> **NOTE:**
>
> - Awesome talk
> - The talk introduces the “Ralph Wiggum Phenomenon,” a simple orchestration pattern for running coding agents repeatedly on small, well-defined tasks.
> - Ralph Wiggum Phenomenon (meme) - A reference to the character Ralph Wiggum from “The Simpsons,” used to illustrate evolving agentic coding concepts.
> - Ralph is a technique, In its purest form Ralph is a bash loop!
>
> ``` bash
> for i in {1..10}; do
>   agent -f RALPH.md --auto-approve"
> done
> ```
>
> run to completion headless, repeatedly.
>
> - what is RALPH.md? It is a prompt that contains the following:
>   - Choose the most important task from PLAN.md
>   - Do only that task. verify it
>   - Mark it complete in PLAN.md
>
> ### Risks:
>
> - cost
>
> - rate limit
>
> - sandboxing (docker etc.)
>
> - **Core idea:** Ralph is not a product, prompt, skill file, or agent. It is a technique: repeatedly run an autonomous coding agent against a reusable `ralph.md` instruction file that tells it to pick the next task, complete it, verify it, update the plan, and exit.
>
> - **Basic workflow:**
>
>   - Maintain a `plan.md` file with unfinished tasks.
>   - Use a reusable `ralph.md` prompt that says: choose the next important task, do only that task, verify it, mark it complete, then stop.
>   - Run the agent headlessly in a loop, ideally with a bounded number of iterations.
>   - Observe failures and refine the process over time.
>
> - **Why the pattern matters:** Ralph forces better context discipline. Instead of letting an agent accumulate a long, degraded conversation history, each run starts fresh and retrieves only the context needed for one task.
>
> - **Risks and guardrails:** Running agents with auto-approval can create token costs and operational risks. The speaker recommends loop limits, sandboxing, Docker or isolated environments, and careful guardrails before using high-autonomy workflows.
>
> - **Main engineering lesson:** Large tasks must be decomposed into small, intelligible tasks. Since the user is not present to correct the agent interactively, the workflow depends heavily on automated tests and machine-checkable feedback.
>
> - **Shift in AI coding workflow:** The speaker frames Ralph as a move beyond chatbot-style coding and ordinary coding-agent command-line interfaces. The user increasingly interacts with the task list and development process rather than directly steering every agent step.
>
> - **Relation to spec-driven development:** Ralph can complement spec-driven development, but it is not the same thing. Spec-driven development focuses on capturing intent; Ralph focuses on executing that intent through a repeatable single-task development loop.
>
> - **Critique: generating too much code:** The speaker agrees that producing large volumes of code is not itself success. Ralph should be used to build what matters well, not merely to maximize velocity or code volume.
>
> - **Single-agent versus multi-agent:** The speaker argues that one reliable agent is often enough. Before adding more agents, it is usually better to diagnose why a single-agent loop fails and improve the task structure, tests, or process.
>
> - **Importance of testability:** Ralph works best when the codebase has strong automated tests. Compiler-like systems are especially suitable because they resemble pure functions: well-defined inputs, well-defined outputs, and few side effects.
>
> - **The “octopus problem”:** Code that touches databases, external services, mutable state, and multiple side effects is much harder for agents to test and modify safely. Systems become more Ralph-friendly when side effects are isolated and test boundaries are clear.
>
> - **More mature `ralph.md` pattern:** A more developed Ralph file may instruct the agent to:
>
>   - choose one unfinished task from `plan.md`;
>   - read any progress file for that task;
>   - inspect pending Git changes from previous attempts;
>   - either continue or reset prior work;
>   - implement only the selected task and related tests;
>   - verify the result;
>   - commit if complete;
>   - otherwise update the progress file and exit.
>
> - **Handling repeated failures:** The speaker suggests explicitly checking for unfinished changes, deciding whether to continue or revert, and possibly using branches per task, especially in multi-agent setups.
>
> - **Task management:** External systems like Linear or Jira can feed the workflow, but the speaker prefers keeping granular task state in version control, often as Markdown or JSON, so task progress and code changes stay consistent with each branch.
>
> - **Model choice:** Higher-capability models tend to work better for autonomous workflows. Self-hosted models may work, but they are generally harder to make reliable for high-autonomy agent loops.
>
> - **Takeaway:** Ralph is a simple but powerful pattern for disciplined autonomous coding: fresh context, one task at a time, strong automated feedback, bounded autonomy, and continuous process improvement.

## Solo to Production: End-to-End Ownership with AI

------------------------------------------------------------------------

- [Solo to Production-End-to-End Ownership with AI]()
- [Brian Turcotte]()
- [twitter](https://twitter.com/coldopn)
- Kilo
- [Slides](https://docs.google.com/presentation/d/1rwLtrTzd9RXMcyab-wOw0qAG3BCRm4dbcbcQvj-MVqU)
- [GitHub](https://github.com/bturcotte520)

> **NOTE:**
>
> - The transcript is a developer-focused talk by Brian, a developer relations engineer at Kilo Code, about using AI agents for end-to-end software ownership.
>
> - The central claim is that modern AI coding systems shift the bottleneck from “can you code?” to “do you know what you want to build?”
>
> - “End-to-end ownership” is defined as one person owning the full product lifecycle:
>
>   - idea
>   - architecture and planning
>   - coding
>   - review
>   - deployment
>   - monitoring
>   - maintenance
>
> - Brian argues that AI agents now make this model realistic even for people who are technical but not formally trained as full-stack engineers.
>
> - Kilo Code is presented as an agentic engineering platform spanning:
>
>   - Visual Studio Code extension
>   - command-line interface
>   - cloud agents
>   - code review automation
>   - deployment workflows
>   - shared project context across interfaces
>
> - The live demo uses an open-source browser-based synthesizer as the starting point.
>
> - The first demo phase happens in the IDE:
>
>   - Brian asks the agent to add “musical typing”
>   - the app gains keyboard-based note playing
>   - MIDI-style interaction is added
>   - recording and quantization are added
>
> - The second phase happens in the CLI:
>
>   - Brian asks the agent to add a drum machine or beat programmer
>   - the agent writes components, fixes errors, and updates the app
>   - the result is a working sequencer with kick, snare, and other drum controls
>
> - The demo emphasizes that agents can handle not just code generation but also terminal commands, commits, pushes, and project operations.
>
> - Kilo’s automated code review is shown as part of the workflow:
>
>   - users can configure review strictness
>   - choose repositories
>   - set focus areas such as security, performance, bugs, style, tests, and documentation
>   - add custom instructions for the review agent
>
> - GitHub integration is presented as the connective tissue that lets agents work across local IDE, CLI, cloud execution, pull requests, and reviews.
>
> - The third phase uses a cloud agent:
>
>   - Brian asks it to redesign the synth interface
>   - the instruction is to make the app feel vintage, bubbly, and visually richer
>   - the cloud agent works remotely and prepares changes for a pull request
>
> - A major theme is model agnosticism:
>
>   - different models should be used for different tasks
>   - stronger reasoning models are useful for planning and architecture
>   - cheaper or lighter models are often sufficient for small implementation or UI tasks
>   - avoiding dependence on a single model provider is framed as strategically important
>
> - Brian argues that the model landscape is becoming more diverse rather than consolidating around one dominant model.
>
> - The talk also introduces a cloud-based personal agent setup, described as having its own accounts and integrations:
>
>   - Google account
>   - password manager
>   - GitHub
>   - search tools
>   - Slack
>   - Discord
>   - Telegram interface
>
> - Brian gives an example of using such an agent to help draft a weekly newsletter by gathering company and external information and producing an HTML preview.
>
> - The final extension of the workflow is automated maintenance:
>
>   - an agent can monitor and maintain the open-source project
>   - the project can receive recurring updates and care without constant manual intervention
>
> - Kilo is described as open source, with its backend source available, and users can inspect, fork, customize, or contribute to the platform.
>
> - The talk mentions custom agent modes, such as:
>
>   - coding
>   - asking
>   - reviewing
>   - planning
>   - debugging
>
> - The conclusion is that AI-assisted end-to-end ownership is not a future possibility but already practical today.
>
> - The main takeaway is that individuals can now build, review, ship, and maintain real software much faster and with broader scope than was previously feasible alone

## Build, Test, and Debug dbt Faster with Claude

- [Bruno Souza de Lima](linkedin.com/in/brunoszdl)
- phData
- [Slides](https://docs.google.com/presentation/d/1pBbI9DvPllY9zdVqq8avPEupU16kBNTspB8kmCRmKnk/edit?slide=id.g3d30caa7339_0_0#slide=id.g3d30caa7339_0_0)
- [workshop repo](github.com/bruno-szdl/dbt-claude-workshop)

> **NOTE:**
>
> - **Workshop topic:** Bruno, a lead engineer at Peach Data and active dbt community member, presents a workshop on using Claude to build, test, document, and debug dbt projects faster.
>
> - **Core argument:** AI can generate dbt-style SQL quickly, but speed alone is not enough. The real value comes from making Claude context-aware so it follows project structure, team conventions, current dbt practices, and business logic.
>
> - **dbt overview for beginners:**
>
>   - dbt helps transform raw operational data into clean, modeled, decision-ready data.
>   - It replaces scattered stored procedures and notebooks with a structured SQL-first workflow.
>   - It brings software engineering practices into analytics engineering, including modularity, testing, documentation, lineage, and continuous integration/continuous deployment.
>
> - **Demo project structure:**
>
>   - The project uses a simple e-commerce dataset with customers, orders, and payments.
>   - Raw data is transformed through staging models into mart models.
>   - The staging layer standardizes and renames fields.
>   - The mart layer contains business-facing models such as customers and orders.
>   - DuckDB is used locally to avoid requiring external warehouse setup.
>
> - **Why Claude needs project context:**
>
>   - Claude is described as a strong drafter but a poor project member by default.
>   - Without instructions, it may generate valid-looking SQL that ignores project conventions.
>   - Claude does not automatically understand the dbt directed acyclic graph, business assumptions, naming conventions, or preferred documentation style.
>
> - **Three mechanisms for improving Claude’s dbt output:**
>
>   - **`claude.md`:** Stores general project instructions that should be included in every Claude interaction, such as project layout, database configuration, and rules like not modifying seed files.
>   - **Model Context Protocol (MCP):** Gives Claude tools for interacting with dbt, such as running `dbt build` and retrieving up-to-date dbt documentation.
>   - **Agent skills:** Specialized markdown instructions that teach Claude how to perform specific tasks, such as writing dbt models, adding meaningful tests, or producing documentation in the project’s preferred style.
>
> - **Skills are emphasized as especially important:**
>
>   - Skills encode team-specific best practices.
>   - They should be used for targeted tasks rather than general background context.
>   - Example skills include a project style guide, meaningful dbt tests, and documentation quality rules.
>   - The presenter notes that skill descriptions matter because Claude decides whether to invoke a skill based partly on its description.
>
> - **Recommended workflow with Claude:**
>
>   - Define one clear task at a time.
>   - Give enough project context.
>   - Use MCP tools and skills deliberately.
>   - Review every output.
>   - Run dbt tests and builds to validate changes.
>   - Iterate through small, testable steps rather than asking Claude to do many things at once.
>
> - **Hands-on model-building example:**
>
>   - Claude is asked to create an intermediate model combining order and payment information.
>   - The presenter reviews whether Claude follows the project style guide.
>   - When Claude does not clearly use the intended skill, the presenter reruns the task while explicitly requesting the style guide skill.
>   - Claude then produces a better-structured model with documentation and tests.
>
> - **Hands-on refactoring example:**
>
>   - Claude enhances the existing customers mart model to use the new intermediate order-payment model.
>   - The presenter checks whether joins, naming conventions, and documentation follow project rules.
>   - Claude initially places documentation in YAML rather than the intended markdown file, showing why human review remains necessary.
>   - The presenter then asks Claude to refactor the documentation using the documentation quality skill.
>
> - **Where Claude is useful:**
>
>   - First drafts of dbt models.
>   - Generic and skill-guided test suggestions.
>   - Documentation scaffolding.
>   - Project exploration.
>   - Running validation commands through tools.
>
> - **Where caution is needed:**
>
>   - Business logic assumptions.
>   - Company-specific metric definitions.
>   - Glossary and status semantics.
>   - Project conventions.
>   - Lineage impact analysis.
>   - Any generated code that has not been tested and reviewed.
>
> - **Debugging example:**
>
>   - The presenter introduces a failing uniqueness test on `order_id`.
>   - The failure occurs because the orders model has multiple rows per order due to split payments.
>   - Instead of asking Claude to immediately fix the issue, the presenter first asks Claude to explain the error.
>   - After reviewing Claude’s diagnosis, the presenter asks for a proposed fix.
>   - Claude’s first suggested fix is not ideal, so the presenter steers it toward using the intermediate model created earlier.
>   - The final corrected model passes the dbt test.
>
> - **Main lesson:** Use AI for speed, not authority. Claude can accelerate dbt development substantially, but the human developer remains responsible for task framing, context, review, business logic, and validation.
>
> - **Practical takeaway:** A reproducible AI-assisted dbt workflow should combine small prompts, explicit project context, dbt-aware tools, well-written skills, continuous review, and frequent validation with dbt commands.

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2026,
  author = {Bochman, Oren},
  title = {ODSC {AI} 2026 - {Day} 2},
  date = {2026-04-29},
  url = {https://orenbochman.github.io/posts/2026/04-29-ODSC-AI-2026-Day-2/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2026. “ODSC AI 2026 - Day 2.” April 29. <https://orenbochman.github.io/posts/2026/04-29-ODSC-AI-2026-Day-2/>.
