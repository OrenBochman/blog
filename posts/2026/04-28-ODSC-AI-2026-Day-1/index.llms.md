## Tensor Logic The Language of AI

Play

![Vimeo](https://f.vimeocdn.com/p/images/crawler_logo.png)

Pedro Domingos

[Wikipedia](https://en.wikipedia.org/wiki/Pedro_Domingos)

[website](https://homes.cs.washington.edu/~pedrod/)

[The Master Algorithm](https://www.hachettebookgroup.com/titles/pedro-domingos/the-master-algorithm/9780465061921/?lens=basic-books)

[Markov Logic - An Interface Layer for Artificial Intelligence](https://link.springer.com/book/10.1007/978-3-031-01549-6)

University of Washington

[Tensor Logic](https://tensor-logic.org/)

[slides](https://homes.cs.washington.edu/~pedrod/tls.pdf)

TODO:

read the a paper

implement it as a blockly playground to understand.

many talk about tensor logic with chat GPT.

> **NOTE:**
>
> - **Central thesis**
>   - AI still lacks a proper native language.
>   - Existing candidates each solved only part of the problem:
>     - **Lisp / Prolog:** good for symbolic reasoning, weak for learning.
>     - **Graphical models / Bayesian networks:** useful for statistical AI, but inference did not scale well enough.
>     - **Markov logic networks:** elegant neuro-symbolic/probabilistic logic, but computationally expensive.
>     - **Python + NumPy / PyTorch / TensorFlow / JAX:** practical and scalable, but not intrinsically built for automated reasoning or symbolic knowledge.
> - **What an AI language should provide**
>   - Hide non-AI implementation details.
>   - Incorporate domain knowledge naturally.
>   - Support automatic reasoning.
>   - Support automatic learning.
>   - Produce transparent models.
>   - Be reliable.
>   - Scale efficiently, especially on modern hardware.
> - **Domingos’s proposal: TensorLogic**
>   - TensorLogic is presented as a unified language for AI.
>   - It combines:
>     - **Logic programming**, the mathematical basis of symbolic AI.
>     - **Tensor algebra**, the mathematical basis of deep learning.
>   - A TensorLogic program is essentially a **set of tensor equations**.
> - **Logic programming refresher**
>   - Logic programs consist of **facts** and **rules**.
>   - Example facts: `parent(Bob, Chris)` or `ancestor(Alice, Bob)`.
>   - Rules define relations, such as ancestry through parenthood.
>   - Domingos emphasizes the database interpretation:
>     - A rule corresponds to **joins** followed by **projection**.
>     - This connects logic programming directly to relational query execution.
>   - Inference can be done by:
>     - **Forward chaining:** repeatedly apply rules until no new facts appear.
>     - **Backward chaining:** start from a query and recursively prove its subgoals.
> - **Tensor algebra refresher**
>   - Tensors generalize scalars, vectors, and matrices.
>   - They are defined by type and shape.
>   - Core operations include tensor sum, tensor product, elementwise product, and contraction.
>   - Domingos highlights **Einstein summation**, or **Einsum**, as the key operation:
>     - Repeated indices imply summation.
>     - Matrix multiplication, tensor contraction, and many neural-network computations can be expressed compactly this way.
> - **How TensorLogic unifies the two**
>   - A TensorLogic equation generalizes a Datalog rule numerically.
>   - Logical joins become tensor products or contractions.
>   - Logical projection becomes summation over indices.
>   - This makes symbolic rules and neural computations instances of the same formalism.
> - **Neural networks in TensorLogic**
>   - Domingos argues that many standard architectures can be written compactly as tensor equations:
>     - Multilayer perceptrons.
>     - Recurrent neural networks.
>     - Convolutional neural networks.
>     - Transformers.
>   - He claims a transformer can be represented in roughly a dozen equations, with attention layers expressed through matrix multiplications for queries, keys, and values.
> - **Inference in TensorLogic**
>   - TensorLogic supports both major logic-programming styles:
>     - **Forward chaining:** execute tensor equations sequentially, computing values whose inputs are available.
>     - **Backward chaining:** treat equations like functions and recursively evaluate only what is needed for a query.
>   - The choice depends on the application.
> - **Learning in TensorLogic**
>   - Since the language has one central construct, the tensor equation, learning reduces to differentiating tensor equations.
>   - Domingos argues that gradients of TensorLogic programs are themselves TensorLogic programs.
>   - This enables gradient descent over both neural and symbolic structures.
>   - He describes this as **backpropagation through structure**, a generalization of **backpropagation through time** for recurrent neural networks.
>   - The idea is that different examples may instantiate different computation structures, but the derivatives can still be accumulated over shared parameters.
> - **Symbolic AI in TensorLogic**
>   - Because TensorLogic generalizes logic programming, existing Prolog- or Datalog-like programs can be imported naturally.
>   - The goal is not merely to call symbolic code from neural code, but to represent both within one common language.
>   - Domingos also suggests TensorLogic can express graphical models, kernel machines, and reasoning in embedding spaces.
> - **Main claimed advantages**
>   - One language for symbolic reasoning and statistical learning.
>   - Reasoning and learning “out of the box.”
>   - GPU-friendly computation through tensor operations.
>   - More transparent models than ordinary deep-learning code.
>   - Potentially simpler debugging and automatic generation of AI models.
> - **Closing message**
>   - TensorLogic is presented as a candidate for “one language for all of AI.”
>   - Domingos frames it as combining the scalability and learnability of deep learning with the transparency, reliability, and reasoning capacity of symbolic AI.

### Reflection

- A first look at the paper helped me see a few new patterns
- Using sparse attention + Forward/Backward chaining to systematically reduce ambiguity.
- Using sparse attention + Forward/Backward chaining to create hierarchical discourse level summary *state*.

## Practical Foundations for Organization-Wide AI Adoption

Play

![Vimeo](https://f.vimeocdn.com/p/images/crawler_logo.png)

- Ivan Lourenço Gomes
  - [LinkedIn](https://www.linkedin.com/in/ivan-louren%C3%A7o-gomes-07694956/)
  - Daweb Schools

> **NOTE:**
>
> - **Audience:** The talk is aimed at business owners, managers, team leaders, and employees who want to push AI adoption forward in their workplace.
>
> - **Central argument:** Buying individual AI subscriptions is not enough. Real organizational value comes from shared processes, reusable knowledge, testing, and team-wide adoption.
>
> - **Current problem in AI adoption:**
>
>   - Many companies report using AI, but most remain stuck in experimentation or pilot projects.
>   - A common reason is that businesses have not redesigned workflows around AI.
>   - In many cases, companies have not even implemented the basic foundations needed for useful AI adoption.
>
> - **Three layers of AI adoption:**
>
>   - **AI starter pack:** Basic shared AI workflows that every organization should implement first.
>   - **No-code agents:** More advanced automated workflows using tools like Zapier, Make, n8n, Salesforce, or HubSpot.
>   - **Custom AI tools:** Internal applications built with developer support to solve business-specific problems.
>
> - **Layer 1: The AI starter pack**
>
>   - The starter pack consists of:
>     - A **shared knowledge base** to ground AI responses in accurate company information.
>     - **Reusable instructions** so teams can standardize prompts, tone, formats, and procedures.
>     - **Systematic testing** to identify and fix weak or incorrect AI outputs.
>   - The speaker emphasizes that AI work should be collaborative, not isolated to individual employees.
>
> - **Hotel customer-service example:**
>
>   - A hotel customer-service team can use AI to answer guest emails faster.
>   - Without a shared knowledge base, employees must manually add details such as parking prices, pool availability, or booking policies every time.
>   - A shared knowledge base can include documents on booking policies, hotel information, room types, facilities, and location-specific details.
>   - A reusable AI assistant, such as a Google Gemini Gem or equivalent ChatGPT tool, can combine those documents with predefined instructions for tone and structure.
>   - The result is faster, more accurate, and more consistent customer communication.
>
> - **Importance of testing:**
>
>   - Teams should test AI assistants with real or simulated cases before deployment.
>   - Errors should be logged in a simple worksheet.
>   - Some failures indicate missing knowledge-base content; others require better instructions.
>   - The process should be iterative: build the knowledge base, write instructions, test, refine, and repeat.
>
> - **Other starter-pack tools:**
>
>   - Meeting transcription and summaries, such as MeetGeek.
>   - AI features in Google Workspace.
>   - GitHub Copilot for developers.
>   - Gamma for presentations.
>   - NotebookLM for policy, handbook, or training-material use cases.
>   - These tools are useful only when combined with shared knowledge, instructions, and testing.
>
> - **Layer 2: No-code AI agents**
>
>   - No-code agents are a next step once teams already use basic AI workflows effectively.
>
>   - The distinction between Gems/custom assistants and agents is:
>
>     - A Gem is **reactive**: it waits for the user to ask.
>     - An agent is **proactive**: it can monitor triggers, watch folders, react to schedules, and initiate workflows.
>     - A Gem mainly produces outputs in chat.
>     - An agent can orchestrate actions across tools, such as creating documents, scheduling meetings, or triggering other agents.
>
> - **Consulting workflow example:**
>
>   - A new client schedules a meeting through a form.
>   - The consultant researches the client, prepares a meeting brief, conducts the call, summarizes notes, and prepares next steps.
>   - The speaker argues that the human conversation should remain human because it builds trust.
>   - Other tasks—research, briefing, summarization, proposal preparation, and scheduling—can be delegated to AI agents.
>
> - **Customer Intelligence Team example:**
>
>   - A **research agent** investigates potential customers.
>   - A **briefing agent** prepares meeting briefs from the research.
>   - A **follow-up agent** summarizes meetings, drafts proposals, schedules calls, and defines next steps.
>   - The speaker recommends splitting agents into small, specialized tasks rather than building one large agent for everything.
>
> - **Layer 3: Custom internal AI tools**
>
>   - Custom tools require more IT effort but can deliver high business value.
>   - The speaker recommends building a basic internal web app infrastructure using Firebase, authentication, user accounts, permissions, and APIs.
>   - Once the foundation exists, teams can quickly add AI-powered tools for document processing, translation, summarization, image recognition, audio/video processing, and data workflows.
>
> - **Custom-tool examples from a client project:**
>
>   - A multilingual content tool for a website operating in 24 languages.
>
>     - It used the DeepL API and glossaries to improve consistency.
>     - It reduced expensive and manually intensive translation workflows.
>
>   - An invoice-processing tool that extracts product codes, quantities, and prices, then compares them with purchase orders.
>
>     - It saved around 10 hours of work per week.
>
>   - A knowledge-base CMS feeding a website chatbot.
>
>     - It reduced load on human specialists by answering common customer questions.
>
>   - An image archive that classified more than 10,000 product images using Gemini.
>
>     - It allowed the team to search images using predefined taxonomies.
>
>   - Google Chat bots for internal workflows.
>
> - **Main lesson from the custom-tool section:**
>
>   - Success came not from the technology alone, but from interviewing the people doing the work and building tools around their real pain points.
>   - AI adoption should solve specific operational problems, not exist as “AI for the sake of AI.”
>
> - **AI effort-benefit curve:**
>
>   - AI can eventually make individuals and teams dramatically more productive.
>
>   - However, the easy productivity gains come only after significant upfront effort:
>
>     - interviewing teams,
>     - gathering data,
>     - creating knowledge bases,
>     - writing instructions,
>     - testing,
>     - and building reusable infrastructure.
>
>   - Many organizations get stuck before reaching the high-benefit stage.
>
> - **Final takeaway:**
>
>   - Start with simple, shared, testable AI workflows.
>   - Move to no-code agents only after the basics are working.
>   - Invest in custom internal tools when the organization has clear, repeated, high-value problems.
>   - Sustainable AI adoption requires intention, discipline, iteration, and team-wide participation.

### Reflections

- The talk provides a practical roadmap for organizations struggling to move beyond AI experimentation.
- The emphasis on shared knowledge, instructions, and testing is crucial for scaling AI across teams.
- The distinction between reactive Gems and proactive agents is a useful framework for thinking about AI workflows.

## vLLM with the Transformers Modelling Backend

Play

![Vimeo](https://f.vimeocdn.com/p/images/crawler_logo.png)

- Harry Mellor
  - vLLM with the Transformers Modelling Backend
  - [LinkedIn](https://www.linkedin.com/in/harrymellor/)
  - HuggingFace
  - [tutorial](https://github.com/hmellor/vllm-transformers-modelling-backend-tutorial)

> **NOTE:**
>
> - Tools:
>   - Transformers
>   - torchtitan
>   - Axolotl
>   - TRL
>   - Unsloth
>
> 1.  [Transformers](https://github.com/hmellor/vllm-transformers-modelling-backend-tutorial/blob/main/notebooks/lesson_1_transformers.ipynb) - covers `transformes serve`, `from_pretrained` and `generate_batch`
> 2.  [vLLM](https://github.com/hmellor/vllm-transformers-modelling-backend-tutorial/blob/main/notebooks/lesson_2_vllm.ipynb) - covers `llm serve` and the `LLM` class.
> 3.  [transformers backend](https://github.com/hmellor/vllm-transformers-modelling-backend-tutorial/blob/main/notebooks/lesson_3_transformers_backend.ipynb) how vLLm can run a transformers model implementation without reimplementing it from scratch.
>     - c.f. [Building a compatible model backend for inference](https://huggingface.co/docs/transformers/en/community_integrations/transformers_as_backend)
> 4.  [Bring your own transformers model](https://github.com/hmellor/vllm-transformers-modelling-backend-tutorial/blob/main/notebooks/lesson_4_bring_your_model.ipynb) to vLLM
>     - When making a model compatible with the Transformers backend, watch out for:
>       - **Missing** kwargs at any level\*\* — The most common issue. If OlmoeModel.forward accepted \*\*kwargs but OlmoeDecoderLayer.forward didn’t, attention_instances would be silently dropped.
>       - **Custom attention** not using ALL_ATTENTION_FUNCTIONS — Models that compute attention inline can’t be dispatched to vLLM’s kernels. The model must use the standard dispatch pattern.
>       - **Incorrect TP plans** — Misspecifying “colwise” vs “rowwise” will produce wrong results silently. Remember: projections that increase dimension (Q, K, V, gate, up) are typically “colwise”, and projections that decrease dimension (O, down) are “rowwise”.
>       - **Non-standard attention mask handling** — Models that manipulate attention weights directly (e.g., adding positional bias to attention scores after softmax) may not be expressible through the standard attention interface.

## The Verifier–Compiler Loop: Turning Human Preferences into Production Agent Judgment

Play

![Vimeo](https://f.vimeocdn.com/p/images/crawler_logo.png)

- Ruslan Belkin
  - [LinkedIn](https://www.linkedin.com/in/rbelkin/)
  - Inflection AI
  - [slides](https://docs.google.com/presentation/d/1Cz5h0j3K5yyYH_BcuXorl-c1e9ymcQ28/edit?pli=1&slide=id.p1#slide=id.p1)

> **NOTE:**
>
> - Ruslan Belkin assumes that everything he says people can do - is doable and actually contributes to success. I.e. if you do it like i say you will succeed. This seems like a demand that is rooted in fantasy. Can we record everything that happen when the ginie is overwhelming us with nonsense and we are trying just to … it?
> - I cover this in agentic patterns as this is similar to some of my own thinking.

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

## The Operational Transformation of Data Architecture

Play

![Vimeo](https://f.vimeocdn.com/p/images/crawler_logo.png)

- Andy Petrella
  - [LI](https://www.linkedin.com/in/andypetrella/)
  - [GitHub](https://github.com/andypetrella)
  - Book:
    - [Fundamentals of Data Observability](https://www.oreilly.com/library/view/fundamentals-of-data/9781098133283/)
  - NextLab SRL

> **NOTE:**
>
> - [DataBricks](https://databricks.com/) - a unified data analytics platform that provides tools for data engineering, data science, and machine learning.
> - [dc43](https://dc43.com/) - a data catalog and data governance platform built on top of the Model Context Protocol (MCP) to help organizations manage and govern their data assets in the age of AI.
>   - Governance layer
>   - Services layer
>   - Integration layer
>   - Execution layer
> - [collibra](https://www.collibra.com/) - a data intelligence company that provides a cross-organizational data governance platform.
> - [apache spark](https://spark.apache.org/) - an open-source unified analytics engine for large-scale data processing, with built-in modules for streaming, SQL, machine learning and graph processing.
> - [databricks DLT](https://databricks.com/product/delta-lake) - a storage layer that brings reliability to data lakes. It provides ACID transactions, scalable metadata handling, and unifies streaming and batch data processing.

## Harness Engineering: Practical Patterns for Agent-First Software Development

Play

![Vimeo](https://f.vimeocdn.com/p/images/crawler_logo.png)

- Ryan Lopopolo
  - [LinkedIn](https://www.linkedin.com/in/ryanlopopolo/)
  - [GitHub](https://github.com/lopopolo)
  - [slides]()

> **NOTE:**
>
> - **Core thesis:** Modern coding agents can produce substantial software end-to-end, but their effectiveness depends less on raw model capability and more on the surrounding engineering harness: context, feedback loops, tests, documentation, review workflows, and operational constraints.
>
> - **Speaker’s experiment:**
>
>   - Over roughly six months in 2025, the speaker’s team used coding agents to build an internal productivity agent.
>   - The product reached beta with about 200 internal users.
>   - The team aimed to have agents produce essentially all code, with minimal direct human editing.
>   - Human engineers shifted from writing code to designing, staffing, guiding, and validating the “code factory.”
>
> - **Why traditional software workflows change:**
>
>   - If code generation becomes cheap, teams no longer need to optimize primarily around human coding throughput.
>   - Human attention becomes the scarce resource.
>   - Agent context and attention are also scarce: broader tasks dilute agent performance.
>   - Work should therefore be decomposed into tightly scoped agent sessions.
>
> - **Main bottlenecks:**
>
>   - **Human attention:** humans should not repeatedly provide the same synchronous feedback.
>   - **Model context:** agents need the right information at the right time, not every possible instruction upfront.
>   - **Model attention:** the more unrelated context an agent sees, the less reliable it becomes.
>
> - **Harness engineering:**
>
>   - Harness engineering means designing systems that deliver the right context to agents at the right stage of work.
>   - Code review comments are treated as evidence of missing context: if a human repeatedly comments on an issue, that requirement should be encoded into the harness.
>   - The goal is to move feedback earlier in the process so the agent avoids predictable mistakes before review.
>
> - **Context as the central mechanism:**
>
>   - Agents receive context from prompts, `agents.md`, repository structure, documentation, tests, linters, tool outputs, and reviews.
>   - Non-functional requirements should be written down explicitly: reliability, performance, typography, architecture, testing expectations, security, and code style.
>   - Since agents start “fresh” on every task, they do not accumulate tacit team knowledge the way humans do through onboarding and repeated review.
>
> - **The codebase is part of the prompt:**
>
>   - Agents inspect nearby files and imitate local patterns.
>   - Homogeneous repository structure helps agents generalize from one file or module to another.
>   - Good existing code improves future agent output because it becomes useful in-context evidence.
>
> - **Use `agents.md` as a map, not a manual:**
>
>   - A short `agents.md` with pointers to deeper documentation worked better than a huge file containing everything.
>   - The speaker contrasts a roughly 300-line map-like file with a 3,000-line overloaded file.
>   - The agent should be told where to look depending on task type, such as frontend architecture, numerical analysis, reliability, or performance.
>
> - **Front-of-process techniques:**
>
>   - Provide compact, discoverable documentation for agent personas and engineering expectations.
>   - Maintain documents such as:
>     - how to write reliable production code,
>     - how to write performant TypeScript,
>     - how to structure frontend architecture,
>     - how to use internal libraries,
>     - how to validate user journeys.
>   - Encode common operational lessons, such as requiring network calls to have timeouts and retries.
>
> - **Middle-of-process techniques:**
>
>   - Use fast tests, builds, and linters so agents can hill-climb toward correct solutions.
>   - Linters should check not only syntax but also repository structure, package boundaries, configuration consistency, and architectural rules.
>   - High-quality linter failure messages should explain the remediation in human-readable prose.
>   - These tool outputs become just-in-time prompts for the agent.
>
> - **Repository architecture for agents:**
>
>   - The team used many small package boundaries in a monorepo to help agents reason locally.
>   - The speaker mentions around 500 local NPM packages.
>   - The architecture was intentionally over-partitioned, even without deploying true microservices.
>   - This allowed agents and humans to restrict the relevant context for a change.
>
> - **Static constraints and schema discipline:**
>
>   - The team used tools such as [Zod](https://github.com/colinhacks/zod) and [Pydantic](https://pydantic-docs.helpmanual.io/) to validate data at boundaries.
>   - They tried to eliminate unknown, untyped, or overly loose internal data structures.
>   - This reduced the agent tendency to generate redundant validation or dead code deep inside the system.
>
> - **End-of-process techniques:**
>
>   - Treat agents like teammates rather than tools that must be watched continuously.
>   - Require a proof of work from agents:
>     - tests run,
>     - quality assurance plan,
>     - logs inspected,
>     - screenshots or videos,
>     - evidence that the ticket requirements were met.
>   - Agents were taught to attach media and validation artifacts to pull requests.
>
> - **Review agents:**
>
>   - The team distilled patterns from hundreds of human-reviewed pull requests.
>   - Reviewer agents were created to catch common mistakes before the human reviewer.
>   - Review personas included reliability, security, performance, frontend architecture, modularity, and quality assurance.
>   - These reviewers used the same guardrail documents given to implementation agents.
>
> - **Important reviewer-agent design choice:**
>
>   - Review agents must be biased toward merging, not endlessly blocking.
>   - Without that bias, reviewer agents can continuously “heckle” implementation agents and prevent convergence.
>   - The review job was framed as identifying sufficiently important issues, such as P2-and-above concerns.
>
> - **Shift-left feedback loop:**
>
>   - The speaker’s recurring theme is to move review feedback, production feedback, and operational feedback earlier into the agent workflow.
>   - Repeated human interventions should become documentation, skills, tests, or linters.
>   - Every human correction is a candidate for automation or prompt injection.
>
> - **Team productivity effect:**
>
>   - The team moved from roughly 3.5 pull requests per day to 5–10 pull requests per day.
>   - The limiting factor became how much parallel agent work the team could schedule, not how much code humans could personally write.
>   - Each new engineer improved the shared harness by adding their own view of “what good looks like.”
>
> - **Handling production failures:**
>
>   - The team initially lacked normal production observability because humans were not directly doing the work.
>   - They used agents to build observability tooling as code.
>   - Dashboards and alerts were defined in JSON and YAML.
>   - Agents could inspect metrics, logs, and dashboards, then propose missing instrumentation or alerting.
>   - The speaker notes that they still used humans for release branches and smoke testing; continuous deployment was not fully automated.
>
> - **Agent personas in practice:**
>
>   - Personas were coarse-grained and horizontal across the software development lifecycle.
>
>   - Examples:
>
>     - reliability reviewer,
>     - security reviewer,
>     - performance reviewer,
>     - frontend architecture reviewer,
>     - package layering reviewer,
>     - quality assurance reviewer.
>
>   - Each persona was given a small set of relevant documents and asked to evaluate a diff from that role.
>
> - **Key takeaway:**
>
>   - Code is becoming cheap, but attention, context, and validation remain expensive.
>   - The highest-leverage work is to build harnesses that encode team judgment into documents, tests, linters, review agents, and fast feedback loops.
>   - The role of the human engineer shifts from producing code to designing systems that let agents produce acceptable code reliably.

## Real-Time Event-Time Consistent Analytics Pipelines using Kafka, Flink, and Apache Pinot

Play

![Vimeo](https://f.vimeocdn.com/p/images/crawler_logo.png)

- Deep Patel
  - [LinkedIn](https://www.linkedin.com/in/deeppatel710/)
  - Robinhood

> **NOTE:**
>
> - **Talk topic:** Deep, a senior data engineer at Robinhood, explains how Robinhood built **Catbox Data**, an internal real-time analytics platform using **Kafka, Flink, and Apache Pinot**.
>
> - **Motivation: batch analytics was too slow**
>
>   - Robinhood relied heavily on Looker and Presto-based dashboards.
>   - Dashboards were slow, sometimes taking many seconds or even minutes.
>   - There was no standardized materialization of reusable analytic cubes.
>   - Batch data freshness was often daily, hourly, or at best around 30 minutes.
>   - Product, analytics, and fraud-investigation teams needed fresher data, especially after launches or alerts.
>
> - **Platform goals**
>
>   - Provide low-latency ingestion.
>   - Support sub-second or near-sub-second analytical queries.
>   - Preserve **event-time consistency**, not merely processing-time consistency.
>   - Combine fresh streaming data with reconciled historical data.
>   - Keep a SQL-based interface for data scientists and analysts.
>   - Scale horizontally as usage and data volume increased.
>
> - **Architecture overview**
>
>   - Product and service events are emitted into **[Kafka](https://kafka.apache.org/)**.
>   - A central **[Flink](https://flink.apache.org/)** application consumes Kafka streams.
>   - Flink performs stateful processing and enrichment where needed.
>   - Processed events are written back to Kafka.
>   - **[Apache Pinot](https://pinot.apache.org/)** ingests the processed Kafka stream in real time.
>   - In parallel, historical and reconciled data is processed from the *[data lake](https://en.wikipedia.org/wiki/Data_lake)* using **[Spark](https://spark.apache.org/)** and ingested into Pinot as offline data.
>
> - **Why Flink**
>
>   - Flink was chosen for true streaming, event-time processing, state management, and exactly-once-style semantics.
>   - Existing Robinhood infrastructure already used Flink for fraud and feature engineering.
>   - Event time mattered because financial analytics should reflect when a trade occurred, not when the pipeline happened to process it.
>
> - **Why Apache Pinot**
>
>   - Pinot is a distributed columnar Online Analytical Processing database designed for high-throughput, low-latency analytics.
>   - It supports real-time ingestion from Kafka.
>   - It supports *upserts*, useful for changing entities such as order status or user profile attributes.
>   - It can consume change data capture streams, for example through [Debezium](https://debezium.io/).
>   - It offers *retention policies*, *partial upserts*, *indexing*, and high query throughput.
>   - *Star-tree* and other indexes were important for fast analytical queries.
>
> - **Hybrid table strategy**
>
>   - Robinhood used Pinot’s hybrid table model to combine real-time and offline data.
>   - A table has both an online/real-time side and an offline/batch side with the same schema.
>   - Users query a single logical table without needing to know which part is online or offline.
>   - Pinot determines the boundary between offline and real-time data and merges the results.
>   - This enabled Lambda-style architecture: fast approximate/fresh data first, then corrected/reconciled data later.
>   - The same mechanism also helped bootstrap historical data before starting real-time ingestion.
>
> - **Join and denormalization strategy**
>
>   - The team preferred to denormalize data at the producer level whenever possible.
>   - Flink joins were used sparingly because stateful joins can become difficult to operate and debug, especially with late-arriving data.
>   - Some enrichment was done in Flink, for example attaching customer tags to transactions.
>   - Pinot’s query-time joins provided another option when producer-side denormalization was not feasible.
>
> - **Results**
>
>   - Dashboard P95 latency improved from about **24 seconds** to about **800 milliseconds**.
>   - Some dashboards loaded in **50–100 milliseconds** after migration.
>   - Data freshness improved from daily/hourly/30-minute batch updates to about **five seconds**.
>   - Concurrent usage grew from roughly **50 users** to around **500 users**.
>   - Robinhood also moved from Looker to Superset for dashboarding.
>
> - **Lessons learned**
>
>   - Real-time analytics should be treated as a platform, not only as infrastructure for fraud or machine learning features.
>   - Event-time consistency is critical in financial analytics.
>   - Hybrid real-time/offline tables are a practical way to balance freshness with correctness.
>   - Denormalization should happen as early as possible, ideally at the producer, while stateful stream joins should be minimized.
>   - Index design in Pinot is central to achieving low-latency query performance.
>
> - **Q&A points**
>
>   - Pinot was not considered a good fit for unstructured data in Robinhood’s experience; they used document stores for that.
>   - Robinhood’s data engineering team was described as lean, with around 20 data engineers.
>   - The company uses AI agents heavily, including tools such as Claude Code and Codex, and encourages even non-technical employees to write code.

### Reflection

- This is one of the three most interesting talks of the conference for me along with Tensor Logic and RL talks.
- This talk is more on the level of a *pydata* session.
- I recently started to implement a pattern of incorporating an analytical DB (OLAP) in my agentic projects.
  - This led me to get tight with [DuckDB](https://duckdb.org/).
- I like many others was also looking for tooling efficient search and RAG that are required by many of the agentic applications and harness engineering patterns.
- I have lots of experience with Lucene as I used it to implement NLP boosted search for Wikipedia sized corpus.
- I got interested in implementing dual search tool i.e. [BM25F](https://en.wikipedia.org/wiki/Okapi_BM25) type ranking for [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) and RAG via vectors using lucene and this is started.
- This led me to and heard about [Apache Pinot](https://pinot.apache.org/) and since Pinot is built on top of Lucene this naturally peaked my curiosity.
- I am looking at using an OLAP for:
  1.  Tracking diverse project metrics.
  2.  Collecting proof of work artifacts.
  3.  Handling project traces from open telemetry and logs.
  4.  Having multiple replay buffers for RL sub-agents
  5.  Anomaly detection - for early detection of failure modes by the project and the agents

## Using Model Context Protocol with Python: A Getting Started Guide for Data Scientists

Play

![Vimeo](https://f.vimeocdn.com/p/images/crawler_logo.png)

- Ryan Day
  - [LinkedIn](https://www.linkedin.com/posts/ryanday1_apis-ai-data-share-7326761077726150656-QfHO/)
  - [CSBS](https://csbs.com/)
  - No slides
  - [FastAPI](https://fastapi.tiangolo.com/) FastAPI framework, high performance, easy to learn, fast to code, ready for production
  - [Tips](https://tips.handsonapibook.com)

> **NOTE:**
>
> - **Session topic:** “Using Model Context Protocol with Python: A Getting Started Guide for Data Scientists,” presented by Ryan Day.
> - **Central question:** how to give AI agents real agency:
>   - Provide them with more **context**, such as documentation, policies, API definitions, and domain knowledge.
>   - Give them **actions**, so they can call APIs, run commands, create tickets, book resources, or modify systems rather than merely suggest text.
> - **Why APIs matter:** the speaker argues that Representational State Transfer APIs, especially REST APIs, are now even more important because they expose machine-readable actions that agents can invoke.
> - **Four ways agents can interact with systems:**
>   - **Raw API access:** the agent receives an OpenAPI/Swagger specification, infers which endpoint to call, and executes a request, often using `curl`.
>   - **Agent skills:** markdown instructions guide the agent on how to use an API, when to use endpoints, and how to combine them.
>   - **Command-line interfaces:** agents use CLI tools, inspect `--help`, infer commands and parameters, and call APIs indirectly through the CLI.
>   - **Model Context Protocol (MCP):** a protocol designed specifically for agent-tool interaction, tool discovery, and structured access to actions, resources, and prompts.
> - **Raw API demo:**
>   - GitHub Copilot in Visual Studio Code was given an OpenAPI specification for a fantasy football API.
>   - The agent inspected the API, selected endpoints, asked for permission, ran `curl`, and retrieved details about [Patrick Mahomes](https://www.nfl.com/players/patrick-mahomes/).
>   - It also queried another endpoint to return player counts.
> - **Agent skills demo:**
>   - A markdown skill file described how to use an air travel API.
>   - The speaker asked the agent to retrieve 10 flights from American Airlines.
>   - The agent used the skill instructions and returned raw JSON from the API.
> - **CLI demo:**
>   - The speaker built an `air travel` command-line tool using `Typer`.
>   - The agent checked API health with the CLI and then inspected available commands with `--help`.
>   - The main advantage is that agents are already strong at using shell commands and can infer usage from help text.
> - **Model Context Protocol (MCP):**
>   - MCP is presented as a standard protocol for connecting agents to tools, data, prompts, files, databases, and APIs.
>   - MCP servers can expose three major things:
>     - **Tools:** actions the agent can invoke.
>     - **Resources:** static or semi-static content the agent can read.
>     - **Prompts:** reusable prompt templates or task guidance.
> - **How MCP differs from REST APIs:**
>   - MCP uses JSON Remote Procedure Call rather than simple stateless HTTP request-response patterns.
>   - MCP supports bidirectional, streaming client-server communication.
>   - MCP includes runtime tool discovery, so an agent can ask a server what capabilities are available.
>   - The protocol is agent-oriented rather than human/API-client-oriented.
> - **MCP ecosystem and standardization:**
>   - The speaker says MCP was initially proposed by Anthropic and later moved toward broader standardization through the Linux Foundation’s Agentic AI Foundation.
>   - The goal is for MCP to become a common standard across large language model providers, tool vendors, and agent platforms.
> - **Security warning:**
>   - Users should verify that MCP servers are trustworthy.
>   - The speaker warns that malicious or misleading MCP servers can impersonate legitimate providers.
>   - Authentication, especially OAuth, is becoming more important for deployed MCP servers.
> - **Using MCP in Visual Studio Code:**
>   - MCP servers can be added through the VS Code command palette.
>   - Servers may run over standard input/output or HTTP.
>   - Once connected, GitHub Copilot can discover the server’s tools and decide which to invoke.
>   - The agent still asks the user for permission before executing tool calls.
> - **MCP demo:**
>   - The speaker used an MCP server wrapping the Sports World Central football API.
>   - The user asked, “How many teams are in Sports World Central?”
>   - The agent discovered the available MCP tool, selected the count tool, requested permission, executed it, and returned that there were 20 teams.
>   - The speaker emphasized validating backend logs to confirm that the agent actually used the intended MCP server.
> - **Building MCP servers with Python:**
>   - The speaker recommends FastMCP as a Python framework for creating MCP servers.
>   - FastMCP handles tool publishing, server setup, JSON-RPC communication, and client discovery.
>   - A tool can be created by defining a normal Python function and decorating it with an MCP tool decorator.
>   - The implementation can call an underlying REST API using asynchronous HTTPX code.
>   - Existing software development kits can often be reused inside MCP servers.
> - **Response formatting for agents:**
>   - Instead of returning raw JSON, an MCP server can format API responses into more readable strings for the agent.
>   - This can make tool outputs easier for the agent to interpret and use.
> - **Hosting MCP servers:**
>   - The speaker hosted his MCP examples on Prefect Horizon, formerly FastMCP Cloud.
>   - Prefect Horizon provides hosting, authentication, HTTP endpoints, and a test client for chatting with the server.
>   - The speaker also mentions MCP registries, such as GitHub’s MCP registry.
> - **Example MCP projects:**
>   - A football API MCP server for Sports World Central.
>   - An air travel MCP server that can retrieve airline flight information, including United Airlines flights.
> - **Q&A takeaway:**
>   - These methods are not limited to Visual Studio Code.
>   - The speaker has seen similar workflows in Cursor, GitHub Desktop, Claude Code, ChatGPT-like interfaces, and enterprise agent environments.
>   - The specific availability depends on the platform and its guardrails.
> - **Overall conclusion:** MCP is positioned as a more agent-native alternative to raw APIs, markdown skills, and CLIs because it combines structured tool discovery, bidirectional communication, authentication, and reusable server-side abstractions for tools, resources, and prompts.

### Reflection

- This talk was a very practical introduction to the Model Context Protocol (MCP) and how it can be used to give AI agents real agency through structured tool access.

## Deploying Multimodal AI at the Edge: Engineering Patterns for Real-World Performance

Play

![Vimeo](https://f.vimeocdn.com/p/images/crawler_logo.png)

- Achyut Sarma Boggaram
  - [LinkedIn](https://www.linkedin.com/in/achyutsarma//odsc_2026_multi_modal_edge_ml/)
  - Torc Robotics
  - Notebooks
    - [Profiling](https://github.com/achbogga/odsc_2026_multi_modal_edge_ml/blob/main/01_profiling_bottlenecks.ipynb)
    - [Export](https://github.com/achbogga/odsc_2026_multi_modal_edge_ml/blob/main/02_export_and_validate.ipynb)
    - [Observability](https://github.com/achbogga/odsc_2026_multi_modal_edge_ml/blob/main/03_observability_canary_rollback.ipynb)

> **NOTE:**
>
> - **Topic:** The talk is a workshop on real-world engineering deployment patterns for ml models on edge systems, such as robots, autonomous vehicles, and other devices that may run without reliable internet.
>
> - **Central problem:** Offline model performance often fails to translate into production performance because **real-world deployment introduces constraints** that are absent from validation notebooks.
>
> - **Major deployment failure modes:**
>
>   - Network failures, including internal vehicle or robot communication failures between sensors, compute units, and controllers.
>   - Scaling failures when a system moves from a small test bench to thousands or millions of deployed devices.
>   - Domain shift, where validation data fails to represent night, weather, snow, unusual environments, or other real-world conditions.
>   - Latency, memory, sensor, preprocessing, and modality failures that are not captured by ordinary offline accuracy metrics.
>
> - **Edge deployment mindset:**
>
>   - Production systems must optimize for reliability, predictability, latency budgets, memory constraints, and graceful failure.
>   - Accuracy alone is not sufficient; the model must behave acceptably under hardware limits, sensor failures, and distributional surprises.
>
> - **Multimodal model design:**
>
>   - The talk discusses early, middle, late, and hybrid fusion architectures.
>   - Early fusion can give the model richer joint representations, but may be more expensive or fragile.
>   - Late fusion can be more modular and resilient to modality failure, but may lose useful cross-modal interactions.
>   - Hybrid fusion is presented as common in production because it balances representational power, latency, and robustness.
>
> - **Profiling and benchmarking:**
>
>   - The speaker emphasizes measuring inference properly rather than trusting naive timing code.
>   - For GPU timing, `torch.cuda.synchronize()` is essential because GPU execution is asynchronous.
>   - Benchmarks need sufficiently large sample sizes; otherwise latency measurements are too noisy to trust.
>   - Models should be evaluated with `torch.eval()` and deterministic settings where possible.
>
> - **Latency analysis:**
>
>   - Production latency is not just model forward-pass time.
>   - End-to-end latency includes preprocessing, model inference, postprocessing, memory movement, framework overhead, buffers, and sometimes network or sensor delays.
>   - The talk distinguishes mean latency, P50, P90, P95, P99, and max latency.
>   - P99 latency is especially important because it captures the tail behavior that affects real users or deployed devices.
>
> - **Profiler usage:**
>
>   - PyTorch Profiler is introduced as a way to identify bottleneck operations.
>   - Profiling tells the engineer whether optimization should focus on the model architecture, preprocessing, postprocessing, memory transfer, or a specific operator.
>
> - **Model export decision:**
>
>   - Exporting a model is not always necessary or desirable.
>   - If PyTorch on the target hardware already satisfies latency and memory budgets, exporting may add risk without benefit.
>   - Export is justified when deployment hardware, runtime requirements, or performance budgets demand it.
>
> - **ONNX export workflow:**
>
>   - The talk walks through exporting a PyTorch model to Open Neural Network Exchange (ONNX).
>   - Before export, the model should pass feasibility checks: clean forward pass, expected input/output shapes, no problematic non-tensor signatures, and no unsupported custom operations.
>   - After export, the model should be checked structurally and visually; `Netron` is mentioned as a useful graph visualization tool.
>
> - **Export risks:**
>
>   - Dynamic control flow can confuse exported graph representations.
>   - Unsupported custom `CUDA` or C++ operations may require plugins.
>   - Data-dependent shapes can break or complicate export.
>   - `ONNX opset` compatibility must be checked against the model and runtime.
>   - Precision drift can occur when converting between floating-point or quantized formats, such as FP32 to INT8.
>
> - **Correctness validation after export:**
>
>   - Exported models must be validated against the original model.
>   - The speaker recommends parity checks at output and layer levels.
>   - Validation should use domain-appropriate tolerances, such as maximum absolute difference and maximum relative difference.
>   - Classification, regression, radar, LiDAR, and other outputs may need different tolerance thresholds.
>
> - **Optimization levers:**
>
>   - Quantization.
>   - Post-training quantization.
>   - Quantization-aware training.
>   - Model pruning.
>   - Architecture simplification.
>   - Backbone replacement.
>   - Preprocessing optimization.
>   - Fusion-design changes.
>
> - **Graceful degradation:**
>
>   - Real-world systems should keep functioning when a modality or sensor fails.
>   - A robot or vehicle should not catastrophically fail because one camera, LiDAR stream, or preprocessing component is unavailable.
>   - Resilience must be built into the model architecture and the production controller.
>
> - **Observability and rollout:**
>
>   - After deployment, teams must monitor latency, failures, traffic, model health, and output behavior.
>   - The workshop introduces dashboards using tools such as Prometheus and Grafana.
>   - Observability is used not just for inspection, but for automated rollout decisions.
>
> - **Service-level objectives and canary deployment:**
>
>   - The speaker explains Service-Level Objectives (SLOs) as explicit thresholds for acceptable production behavior.
>   - A canary rollout gradually shifts traffic from model version V1 to V2.
>   - The rollout may move through stages such as 0%, 10%, 25%, and eventually 100%.
>   - If latency or failure metrics breach the SLO, the rollout controller should automatically roll back.
>
> - **Main takeaway:** Deploying machine-learning models on edge systems is an engineering discipline, not just a modeling exercise. A production-ready model must satisfy accuracy, latency, memory, exportability, observability, rollback, and resilience requirements under real-world operating conditions.

## Architectural Patterns for Building and Governing Production-Grade Multi-Agent Systems

Play

![Vimeo](https://f.vimeocdn.com/p/images/crawler_logo.png)

- [Dr. Ali Arsanjani](https://www.linkedin.com/in/ali-arsanjani/)
- [blogpost](https://dr-arsanjani.medium.com/the-anatomy-of-agentic-ai-0ae7d243d13c)

> **NOTE:**
>
> - **Speaker and topic**
>
>   - Dr. Ali Arsanjani of Google Cloud presents architectural patterns for building, governing, and scaling production-grade multi-agent AI systems.
>   - The talk focuses on agent maturity, orchestration, governance, self-improvement, security, cost control, and operational resilience.
>
> - **Basic agent architecture**
>
>   - Modern agents use a large language model as the reasoning core.
>   - Agents sense or retrieve information from structured data, unstructured data, digital business systems, and sometimes physical devices.
>   - Access to internal tools and resources is often handled through the **Model Context Protocol (MCP)**.
>   - Agents receive broad goals rather than rigid robotic process automation-style workflows.
>   - They reason, use memory, plan steps, call tools, and may coordinate with other agents through **agent-to-agent (A2A)** protocols.
>
> - **Agent maturity levels**
>
>   - Early systems use static function calls tied to a single large language model.
>   - More advanced systems dynamically choose tools at runtime.
>   - Single-agent systems may use reasoning-action-observation-reflection loops for iterative self-correction.
>   - Multi-agent systems introduce specialized sub-agents coordinated by a root agent.
>   - Higher maturity systems use a meta-agent to enforce policies, resolve conflicts, adjust plans, and govern behavior.
>   - The most advanced systems are self-improving agent ecosystems with multi-agent learning and swarm-like feedback.
>
> - **Security and isolation**
>
>   - Open-source multi-agent frameworks can reduce integration overhead, but should not be deployed naively.
>   - Production deployments need zero-trust infrastructure, sandboxing, identity controls, environment isolation, and defense across pre-action, in-action, and post-action phases.
>   - Agents should be isolated by design, for example with Docker sandboxes and restricted host file-system access.
>
> - **Skills as modular agent behavior**
>
>   - Instead of loading huge prompts with many system instructions, agents can use modular “skills” defined in markdown files.
>   - Skills allow agents to load only the procedural capabilities needed for a task.
>   - This reduces prompt bloat and supports more maintainable agent behavior.
>
> - **Frameworks and deployment**
>
>   - The speaker highlights Google’s Agent Development Toolkit 2.0 as a strong framework for graph-based workflows, shared memory, and multi-agent development.
>   - Other frameworks mentioned include LangGraph, LangChain, and CrewAI.
>   - Agents may be deployed on Gemini Enterprise Agent Engine, Cloud Run, or Google Kubernetes Engine.
>   - Enterprise-facing agents can be surfaced through Gemini Enterprise apps for broader organizational access.
>
> - **Memory architecture**
>
>   - Agents need both short-term and long-term memory.
>   - Session memory supports current interactions.
>   - A longer-term memory bank extracts salient topics from previous sessions and restores them later for personalization and continuity.
>
> - **Hybrid planner-scorer architecture**
>
>   - Self-improving systems benefit from separating generation from evaluation.
>   - A planner generates candidate solutions, optimized for creativity and breadth.
>   - A scorer evaluates those solutions using a quality rubric.
>   - The scoring rubric acts as a contract between agents and defines what “good” means for the domain.
>
> - **Custom evaluation metrics**
>
>   - Generic benchmarks are insufficient for domain-specific work such as legal contracts, loan servicing, or regulated workflows.
>   - Teams should define programmatic quality metrics tied to business, customer, and regulatory criteria.
>   - A golden dataset and scoring function should be developed with domain experts.
>   - These metrics function similarly to reward functions for continuous agent improvement.
>
> - **Preference-controlled synthetic data**
>
>   - When human-labeled examples are scarce, teams can generate preference pairs.
>   - One output is produced under conditions likely to yield a good answer, another under weaker conditions.
>   - These outputs can be labeled as chosen versus rejected and used to train scorers or preference models.
>
> - **Advanced tuning**
>
>   - The talk mentions supervised fine-tuning, parameter-efficient fine-tuning, Low-Rank Adaptation, and Direct Preference Optimization.
>   - Preference-based tuning helps align model behavior with desired outputs rather than merely imitating examples.
>
> - **Co-evolved agent training**
>
>   - A static scorer may become obsolete as the planner improves.
>   - The planner and scorer should improve together.
>   - Planner outputs train a better scorer, and scorer feedback trains a better planner.
>   - This creates a virtuous cycle that can compound system performance.
>
> - **Adversarial testing and red teaming**
>
>   - A dedicated red-team agent can probe the main system for jailbreaks, biases, edge cases, and failures.
>   - This should be proactive rather than only reactive.
>   - For critical tasks, adversarial testing may run on every task execution.
>   - For lower-risk tasks, it can run periodically through scheduled sampling.
>   - The resulting adversarial examples should feed back into preference datasets and co-evolution pipelines.
>
> - **Tokenomics and cost management**
>
>   - Self-improving systems can create runaway token costs if not controlled.
>   - A system-level monitor should track token use across agents.
>   - Agents should have iteration limits, budgets, and automatic pause or scale-down mechanisms.
>   - Cost control must be balanced against measurable return on investment.
>
> - **Business value measurement**
>
>   - Agent systems should be evaluated not only by token cost or technical metrics but by business outcomes.
>   - Relevant metrics include resolution rate, cost per incident, customer satisfaction, and first-call deflection.
>   - Dashboards should link operational agent behavior to business key performance indicators.
>   - Teams should start with one or two core metrics and expand gradually.
>
> - **Robustness and fault tolerance**
>
>   - Production systems need resilience against service failures, crashes, timeouts, and unreliable agents.
>
>   - Five robustness patterns are highlighted:
>
>     - **Adaptive retry:** measure successful retries after initial failures.
>     - **Watchdog timeout:** track timeout violations per hour.
>     - **Auto-healing:** log successful restarts after crashes.
>     - **Trust decay:** track rolling failure rates per agent.
>     - **Fallback model:** compare fallback-model accuracy against primary-model accuracy.
>
> - **Overall message**
>
>   - Production-grade multi-agent systems require more than orchestration.
>   - They need governance, modular skills, secure deployment, memory, custom evaluation, adversarial testing, cost controls, business-value tracking, and fault-tolerance mechanisms.
>   - The central architectural shift is from isolated agents toward governed, measurable, self-improving agent ecosystems.

## Book Author: AI Agents with MCP

Play

![Vimeo](https://f.vimeocdn.com/p/images/crawler_logo.png)

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

## Building FSI Agents with Claude

Play

![Vimeo](https://f.vimeocdn.com/p/images/crawler_logo.png)

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

## Meaningful Data Visualization in the Age of AI

Play

![Vimeo](https://f.vimeocdn.com/p/images/crawler_logo.png)

- Janet Six
  - [LinkedIn](https://www.linkedin.com/in/janetmsix)
  - Tom Sawyer Software
  - [slides](https://docs.google.com/presentation/d/1-LaKE9ksL89OPvXA6f98r4EYDLbbquDi/edit?slide=id.p1#slide=id.p1)

> **NOTE:**
>
> - **Topic:** Making data visualization more meaningful in the age of artificial intelligence.
>
> - **Speaker:** Dr. Janet Six, senior product manager at Tom Sawyer Software, with a background in graph visualization and artificial intelligence.
>
> - **Core argument:** Artificial intelligence can make visualization faster and more abundant, but speed does not guarantee usefulness. A visualization is meaningful only when it helps people understand, decide, validate, or act.
>
> - **Opening analogy:** The speaker uses a hand-movement exercise to show that some problems become easier when reframed. Data visualization has the same issue: instead of asking “what chart should we make?”, teams should ask “what task, decision, or system understanding must this visualization support?”
>
> - **Visualization types mentioned:** Tables, pie charts, line graphs, bar graphs, timelines, matrices, network graphs, hierarchical views, and combinations of these.
>
> - **Main warning about AI:** AI acts like a megaphone: it amplifies speed and volume, but it does not automatically improve quality, relevance, or decision value.
>
> - **Systems engineering frame:** The speaker argues that AI-enabled visualization should be approached through systems engineering because modern AI systems involve interacting tools, people, goals, workflows, requirements, regulations, business processes, and changing stakeholder needs.
>
> - **Definition of a system:** A system is described as a set of interacting elements organized to achieve one or more stated purposes. The speaker emphasizes that both parts are difficult:
>
>   - identifying all interacting elements, including people, software, hardware, processes, and behaviors;
>   - stating the system’s purposes concretely enough to guide design and evaluation.
>
> - **Silo problem:** Different departments may hold conflicting or incomplete requirements. For example, business teams may want to adopt a tool, while operations or compliance teams may later discover it is unsupported or non-compliant. Systems engineering helps detect these conflicts earlier, when changes are cheaper.
>
> - **Supply-chain example:** A logistics visualization can show delivery routes as a graph or matrix, including attributes such as hazardous cargo, refrigeration needs, or heavy goods. But the visualization is only useful if tied to concrete objectives, such as optimizing flow or ensuring backup suppliers within a defined response time.
>
> - **Requirements hierarchy:** High-level business needs must be decomposed into measurable requirements, then into technical solutions. Example: “maintain a resilient supply chain” becomes “have a backup supply chain ready within two hours,” which can then guide AI-tool selection and system design.
>
> - **Risk, cost, and performance trade-offs:** The speaker explains that systems often require balancing competing objectives. Reducing risk and increasing performance usually raises cost; holding cost fixed may require accepting more time or lower performance.
>
> - **Assessment and traceability:** Teams should repeatedly assess whether implementation choices trace back to the original requirements. The speaker stresses the importance of linking stakeholder goals to design decisions, tool configurations, outputs, and validation criteria.
>
> - **Validation versus verification:**
>
>   - **Validation:** checking whether the overall system was built correctly and satisfies the intended requirements.
>   - **Verification:** checking whether individual outputs or ongoing results are acceptable, especially in systems that produce new results continuously, such as generative artificial intelligence systems.
>
> - **V-model:** The speaker relates this process to the systems engineering V-model: requirements and design flow downward, development happens at the bottom, and testing, verification, and validation occur on the upward side.
>
> - **Systems of systems:** Modern AI systems may contain multiple agents or tools interacting with one another. Meaningful visualization may need to show both high-level parent systems and lower-level sub-agent behavior.
>
> - **Hierarchical visualization:** Different audiences need different levels of abstraction:
>
>   - executives may need high-level system status;
>   - engineers may need lower-level agent, workflow, and traceability views.
>
> - **Time and actors:** Useful visualizations should often include temporal structure and actor relationships, not just static snapshots. This supports event analysis, process understanding, and version comparison.
>
> - **Agentic explainability example:** In a fraud-detection scenario, a visualization might show:
>
>   - the suspicious transaction or finding;
>   - related merchants, accounts, and transactions;
>   - which agents contributed to the finding;
>   - confidence or certainty levels;
>   - where in the workflow the result was produced.
>
> - **Versioning:** The speaker notes that visualizations can also show how systems, graphs, or results change over time, either retrospectively or prospectively.
>
> - **Final recommendation:** Put visualization at the beginning of system design, not merely at the end. Use it to clarify requirements, expose conflicts, support optimization, communicate with stakeholders, and keep human experts in the loop.
>
> - **Takeaway:** Meaningful visualization is not just presentation. It is a systems-engineering instrument for understanding complex AI-enabled systems, making trade-offs explicit, tracing requirements to outputs, and supporting better human judgment.

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2026,
  author = {Bochman, Oren},
  title = {ODSC {AI} 2026},
  date = {2026-04-28},
  url = {https://orenbochman.github.io/posts/2026/04-28-ODSC-AI-2026-Day-1/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2026. “ODSC AI 2026.” April 28. <https://orenbochman.github.io/posts/2026/04-28-ODSC-AI-2026-Day-1/>.
