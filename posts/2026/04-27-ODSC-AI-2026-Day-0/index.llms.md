These are workshops and sessions that took place before the start of the main track.

[![](t1-hero.png)](t1-hero.png)

## Engineering the Harness: A Practical Workshop on Context Engineering for Generative AI

# An error occurred.

Unable to execute JavaScript.

I couldn’t find a recording of this workshop from the conference but here is a recording of a similar workshop that Rajiv Shah has provided on YouTube.

- Rajiv Shah
  - [linkedin](https://www.linkedin.com/in/rajistics/)
  - [website](https://rajivshah.com/)
  - [blog](https://rajivshah.com/blog/harness-engineering.html)
  - [slides](https://docs.google.com/presentation/d/1onY5e2wFiiExIPm1X16bJ23F3Zv9fwph)
  - [workshop](https://github.com/rajshah4/harness-engineering)

> **NOTE:**
>
> - In this deep dive into harness engineering, Rajiv Shah explores why coding harnesses—the systems surrounding AI models—are essential for the performance, flexibility, and control of coding agents. While models have evolved from fine-tuning to context engineering, harnesses are now the crucial layer for managing agentic workflows.
> - Key Takeaways:
>   - **Why Coding Harnesses Matters?**: Harnesses allow for the same base model to achieve drastically different results. A well-engineered harness can optimize performance, manage costs, and provide necessary control over agentic loops.
> - **Five Levers of Harness Engineering**:
>   1.  **Model**: We all know how to swap the model.
>   2.  **Retrieval**: Beyond simple RAG, agents should utilize *tools* like [grep](https://en.wikipedia.org/wiki/Grep), [BM25](https://en.wikipedia.org/wiki/Okapi_BM25), and [semantic search](https://en.wikipedia.org/wiki/Semantic_search) to interact with codebases effectively (6:52).
>   3.  **Memory**: Effective memory management includes three layers:
>
>   - the active context window (managing token limits),
>   - working state (using files/markdown for plans), and
>   - durable memory (using `agents.md` files or specialized skills).
>     - **Skills**: Externalizing expertise into reusable processes (“skills”) can significantly reduce the need for excessive orchestration code (17:02).
>
>   4.  **Loops, Tools, and Feedback**: Harnesses should facilitate efficient loops (e.g., planning, test-driven development, and verification) while providing sandboxed environments to handle security and friction (21:29).
>   5.  **Architecture**:
> - **Orchestration**: While multi-agent systems are popular, the video cautions that they introduce significant coordination costs. Single-agent setups with reflective critique often outperform complex swarms (27:37).
> - **Long-Term Outlook**: While some technical details (like specific compaction strategies) may be commoditized, the focus on domain-specific tools, security posture, and the ability to define reusable skills will remain critical for building high-performing, reliable coding agents (31:08).

### Reflection

- Wow what a polished talk.
- Rajiv Shah references the other harness engineering talk by Ryan Lopopolo from OpenAI, covered in

## Introduction to the Math Behind Transformers and LLMs

- David Hoyle
  - [LinkedIn](https://linkedin.com/in/davidchoyle?originalSubdomain=uk)
  - [slides](https://docs.google.com/presentation/d/1w9VVtjgXcCSlTK2UNa4BQ6FUJDXIRhE2/edit?usp=sharing&ouid=107678147228131583519&rtpof=true&sd=true)
  - [dunnhumby](https://www.dunnhumby.com/)

> **NOTE:**
>
> - **Purpose of the talk**
>   - David Hall explains the mathematics behind transformers and large language models in an accessible way.
>   - The goal is not to train a full large language model, build chat bots, or teach prompting, but to reduce fear of the mathematical structure behind these systems.
>   - The central modeling task is **next-word prediction**: given a partial sequence of words, predict the most likely next word.
> - **Next-word prediction as the core idea**
>   - A language model begins with a context, such as “The cat sat on the…”
>   - It predicts the next word, appends that word to the sequence, and then repeats the process.
>   - This recursive procedure can generate longer completions, poems, answers, or dialogue.
>   - Systems such as ChatGPT and Claude can be understood, at a high level, as sophisticated next-token prediction systems.
> - **Naive Markov model baseline**
>   - Hoyle introduces a first-order Markov model using the phrase: “The more you know, the more you realize you don’t know.”
>   - A Markov model predicts the next word using only the immediately preceding word.
>   - Its transition probabilities can be represented as a matrix.
>   - The model exposes two major problems:
>     - It ignores most of the previous context.
>     - It becomes impractical with large vocabularies because the transition matrix grows explosively.
>   - It can also generate nonsensical sentences, illustrating why probabilistic language generation needs better structure.
> - **Why embeddings are needed**
>   - One-hot representations are too large and sparse for realistic vocabularies.
>   - Large language models instead map tokens into lower-dimensional **embedding vectors**.
>   - These vectors numerically encode semantic relationships between words.
>   - Word2vec examples show that embeddings support meaningful operations, such as similarity comparison and analogy-like vector arithmetic.
>   - Hoyle demonstrates cosine similarity and examples such as comparing related words or testing vector analogies.
> - **Main intuition of transformers**
>   - Transformers map tokens to embedding vectors, then transform those vectors so they become **context-aware**.
>   - A word’s final vector should not merely represent the word itself; it should also encode relevant information from surrounding or preceding tokens.
>   - Once context-aware vectors are obtained, a relatively standard classifier can predict the next token.
> - **Attention mechanism**
>   - Attention constructs a new vector for each token by taking a weighted combination of other token vectors.
>   - The weights indicate how much each token should “pay attention” to other tokens.
>   - Instead of averaging the original embeddings directly, the model first applies learned linear transformations to produce value vectors.
> - **Self-attention**
>   - Self-attention computes attention weights from the tokens themselves.
>   - Tokens are transformed into **query** and **key** vectors.
>   - The similarity between a query and a key, usually via an inner product, determines how much attention one token pays to another.
>   - A softmax function converts these similarity scores into normalized attention weights.
>   - Learned matrices produce the query, key, and value vectors.
> - **Masking**
>   - For next-token prediction, the model must not look ahead at future tokens.
>   - Masking prevents later tokens from influencing the representation of earlier positions.
>   - This is done by forcing unwanted attention scores to effectively become zero after softmax.
>   - Decoder-style language models use masking so that each position only attends to previous tokens.
> - **Multi-headed attention**
>   - A single attention mechanism may not capture all relevant relationships between tokens.
>   - Multi-headed attention uses several attention heads in parallel.
>   - Different heads can specialize in different kinds of relationships or different parts of the embedding space.
>   - Their outputs are combined, often by concatenation, to form a richer context-aware representation.
> - **Transformer block structure**
>   - A transformer block contains:
>     - Multi-headed self-attention.
>     - A neural network layer for nonlinear mixing of information.
>     - Normalization layers for stable training.
>     - Residual connections to preserve and stabilize information flow.
>   - These blocks transform input embeddings into context-aware embeddings suitable for prediction.
> - **Predicting the next word**
>   - The model takes the final context-aware embedding vector in the sequence.
>   - A softmax classifier maps that vector to a probability distribution over the vocabulary.
>   - The next token can be chosen as the most probable word or sampled from the distribution for more variation.
>   - The process repeats by appending the selected token and predicting again.
> - **Training objective**
>   - The model learns its parameters from large text corpora.
>   - Training minimizes **[cross-entropy loss](https://en.wikipedia.org/wiki/Cross_entropy)**, which compares predicted token probabilities to the observed next token.
>   - Since the correct observed word has probability one and all alternatives have probability zero, this becomes equivalent to maximizing the likelihood of the training data.
>   - Large models need very large datasets because they contain many parameters.
> - **Positional information**
>   - Basic self-attention alone does not know word order.
>   - If tokens are rearranged, attention based only on token identities may not distinguish the new order properly.
>   - Transformers therefore add positional information to embeddings.
>   - Positional encodings allow the model to represent not only which words occur, but where they occur in the sequence.
> - **Types of transformer models**
>   - **Decoder-only models**
>     - Map context-aware vectors to predicted next tokens.
>     - Used in systems such as ChatGPT and Claude.
>     - Best suited for generative language modeling.
>   - **Encoder-only models**
>     - Map full token sequences into context-aware vectors without next-token generation.
>     - Useful for representation tasks.
>   - **Encoder-decoder models**
>     - Use an encoder to represent an input sequence and a decoder to generate an output sequence.
>     - Useful for tasks such as machine translation.
> - **Code example**
>   - Hoyle briefly shows how transformer operations can be implemented in PyTorch.
>   - Key operations include matrix multiplication for query-key similarity, scaling by the square root of the vector dimension, masking, softmax, and multiplying attention weights by value vectors.
>   - The point is that once high-level tensor operations are available, the mathematical structure of a transformer can be expressed compactly in code.
> - **Q&A**
>   - A [Markov model](https://en.wikipedia.org/wiki/Markov_model) cannot generate words outside its fixed vocabulary.
>   - Transformer models also predict only from a fixed vocabulary, but modern vocabularies are large enough to cover most practical cases.
>   - Infrequent or domain-specific words can be handled better through domain-specific fine-tuning.
>   - Older or smaller embedding datasets may produce weaker similarity results than modern embeddings.
>   - The main structural difference between encoder and decoder blocks is masking: decoders mask future tokens, while encoders generally do not.
> - **Overall takeaway**
>   - A large language model can be understood as a system that:
>     - Converts tokens into vectors.
>     - Uses attention to make those vectors context-aware.
>     - Uses a classifier to predict the next token.
>     - Repeats this process to generate text.
>   - The mathematics is built from familiar components: vectors, matrices, inner products, softmax, probability, and loss minimization.

## Introduction to Machine Learning: From Theory to Application

- Michael Galarnyk
  - [Website](https://mgalarnyk.github.io/)
  - Georgia Institute of Technology
  - [GitHub](https://github.com/mGalarnyk/MLModelsScikit-Learn/tree/main/powerpoints)

> **NOTE:**
>
> - **Topic: Introduction to machine learning with [scikit-learn](https://scikit-learn.org/)**
>
>   - The talk introduces machine learning from both theory and practical application.
>   - Galarnyk uses a GitHub repository with slides and notebooks so participants can follow along.
>   - The focus is on basic-to-intermediate machine learning concepts, especially as they apply to Python and [scikit-learn](https://scikit-learn.org/).
>
> - **Prerequisites and setup**
>
>   - Participants are expected to know basic Python: strings, numbers, conditionals, loops, lists, tuples, and dictionaries.
>   - Some familiarity with *linear regression* is helpful.
>   - Galarnyk recommends using Anaconda or Google Colab to manage Python environments and dependencies.
>   - Core libraries discussed include `NumPy`, `Pandas`, `Matplotlib`, and `scikit-learn`.
>
> - **Why fundamentals still matter in the age of large language models**
>
>   - Even though large language models can generate code, users still need enough understanding to debug mistakes.
>   - Models are better at generating text and code than at generating precise diagrams or visual explanations.
>   - A model may confidently provide incorrect output, so users need enough theory to evaluate whether the result is valid.
>   - Popular libraries such as `scikit-learn` are easier for A.I. tools to assist with because they have abundant examples, documentation, and community usage.
>
> - **Basic machine learning concepts**
>
>   - Machine learning is presented as giving computers the ability to learn from data without being explicitly programmed.
>   - The talk distinguishes between:
>     - **Features**: the input variables used to make predictions.
>     - **Target**: the value or class the model tries to predict.
>     - **Regression**: predicting continuous values, such as home prices.
>     - **Classification**: predicting categories, such as flower species.
>   - The Iris dataset is used as an example of a small classification dataset.
>
> - **Working with notebooks and debugging**
>
>   - Galarnyk emphasizes running [Jupyter notebooks](https://jupyter.org/) sequentially from top to bottom.
>   - A common error discussed is using a variable before it has been defined.
>   - Users are encouraged to inspect variable types, array shapes, and intermediate outputs.
>   - Errors are divided into:
>     - **Coding errors**, such as missing variables or malformed data.
>     - **Understanding errors**, such as applying a regression model to a classification problem.
>
> - **Data preparation**
>
>   - The talk covers loading data into [Pandas](https://pandas.pydata.org/) and identifying feature matrices and target vectors.
>   - Missing values are discussed, including simple removal as a practical shortcut.
>   - Galarnyk notes that missingness itself can sometimes be predictive and may be converted into a feature.
>
> - **Linear regression**
>
>   - Linear regression is introduced through the slope-intercept form, y = mx + b.
>   - Galarnyk explains the role of an intercept and how adding parameters can improve model fit.
>   - R^2 is introduced as a basic performance metric, where higher values generally indicate better fit.
>   - Visualizing the regression line is emphasized as important for communicating results to stakeholders.
>
> - **Train-test split and overfitting**
>
>   - Galarnyk explains why data should be split into training and testing sets.
>   - The model learns from the training set and is evaluated on unseen test data.
>   - Testing on the same data used for training rewards overly complex models that memorize the dataset.
>   - The common 75/25 train-test split is discussed, though modern systems may use much larger training proportions.
>   - `random_state` is used to make random splits reproducible.
>
> - **Decision trees**
>
>   - Decision trees are introduced as interpretable models that make predictions by asking a sequence of questions.
>   - For a housing-price example, the tree may split mainly on square footage if that feature is most predictive.
>   - Galarnyk stresses that giving a model many features does not guarantee it will use all of them.
>   - Tree depth is discussed as a hyperparameter controlling how many questions the tree may ask.
>
> - **Model interpretability**
>
>   - Interpretability is especially important in high-stakes domains such as healthcare and finance.
>   - Users need to understand not only whether a model predicts correctly, but how it reaches its predictions.
>   - Understanding model logic helps identify likely error patterns and evaluate whether the model is using appropriate signals.
>
> - **Hyperparameter tuning**
>
>   - Hyperparameters are settings chosen before or during training, such as maximum tree depth.
>   - Galarnyk demonstrates trying multiple values and comparing performance.
>   - The goal is not to maximize training performance but to find settings that generalize well to validation or test data.
>   - Repeatedly tuning on the same test set can leak test-set knowledge into the model-selection process.
>
> - **Bias-variance tradeoff**
>
>   - High-bias models, such as simple linear regression, may *underfit* by imposing too simple a structure.
>   - High-variance models, such as overly deep trees, may *overfit* by memorizing the training data.
>   - Traditional machine learning often seeks a middle ground between underfitting and overfitting.
>   - Galarnyk notes that modern deep learning complicates the classical picture because very large models can sometimes improve again at massive scale.
>
> - **Random forests**
>
>   - Random forests are introduced as ensembles of decision trees.
>   - They reduce overfitting by combining many trees trained on varied samples and feature subsets.
>   - Galarnyk explains bagging and the use of random feature subsets to prevent every tree from relying on the same dominant feature.
>   - Random forests are described as combining many “specialists” into a stronger aggregate predictor.
>
> - **Local versus cloud computation**
>
>   - Running models locally can reduce latency and simplify data transfer to the machine or GPU.
>   - Local execution may be limited by compute, memory, and hardware constraints.
>   - Cloud tools such as Google Colab reduce setup friction but depend on external services and policies.
>   - Galarnyk briefly discusses parallel and distributed computing as ways to speed up model training and inference.
>
> - **Privacy and personally identifiable information**
>
>   - A participant asks about protecting personally identifiable information when using cloud or artificial intelligence tools.
>   - Galarnyk frames this as an active and unresolved issue.
>   - One mitigation strategy mentioned is limiting tool access so models can query only the minimum information needed.
>
> - **Using artificial intelligence tools effectively**
>
>   - Galarnyk recommends giving models specific context, code, errors, and goals rather than vague requests.
>   - A better prompt explains what was attempted, what failed, and what the desired outcome is.
>   - Screenshots and exact error messages can help models debug code.
>   - Users should still understand the code well enough to judge whether the model’s fix is correct.
>
> - **Recommended next steps**
>
>   - Continue practicing with notebooks and small datasets.
>   - Learn how to diagnose errors, inspect data, and evaluate models.
>   - Read reliable books and documentation rather than relying only on large language models.
>   - Suggested resources include hands-on machine learning books, by[Christopher Bishop](https://www.microsoft.com/en-us/research/people/cmbishop/), and [Sebastian Raschka](https://sebastianraschka.com/).
>
> - **Main takeaway**
>
>   - The talk argues that machine learning fundamentals remain essential, even when artificial intelligence tools can generate code.
>   - Effective users need to understand the data, the model, the evaluation procedure, and the limits of automation.
>   - The practical goal is not just to run a model, but to know whether it is appropriate, interpretable, reliable, and useful.

## A Practical Introduction to Agentic AI

- Sudip Shrestha, Lead AI Engineer at ASI Government
  - [linkedin](https://www.linkedin.com/in/sudipshrestha/)
  - [website](https://www.asigovernment.com/)
  - [slides](https://docs.google.com/presentation/d/1cDMH5PURp5C3IHHQ40MkEW__qPwcVseu/edit?slide=id.p1#slide=id.p1)
  - [ASI Government](https://www.linkedin.com/company/asi-government/)

> **NOTE:**
>
> - **Main objective:** Build a working agentic artificial intelligence application from scratch in Google Colab, ending with a simple Gradio demo app.
> - **Core distinction: chatbot vs. agent**
>   - A chatbot mainly responds to prompts.
>   - An agent can decide, act, call tools, loop through steps, and move a workflow forward without constant human prompting.
>   - The underlying large language model may be the same; the difference is the workflow, tools, state, and control logic wrapped around it.
> - **Why agentic AI is becoming practical now**
>   - Tool calling has become more reliable.
>   - Frameworks such as LangGraph, CrewAI, AutoGen, and OpenAI’s SDK have matured.
>   - There is growing demand for applications that do more than produce text: they execute tasks, revise outputs, and integrate with systems.
> - **Key agentic AI capabilities discussed**
>   - Tool use through Python functions or APIs.
>   - Task decomposition into steps.
>   - Self-correction through critique and revision.
>   - Conditional routing based on intermediate results.
>   - Human-in-the-loop review when autonomy is risky.
>   - Observability and tracing for debugging.
> - **Important limitations and risks**
>   - Agents can become expensive or slow if they loop too much.
>   - Loops need maximum iteration caps.
>   - Results are nondeterministic, so evaluation must be systematic rather than based on one successful run.
>   - Tool descriptions and docstrings matter because the model uses them to decide which tool to call.
>   - Security requires least-privilege access, authentication, and careful API control.
>   - Debugging is harder because the model, tools, prompts, and graph logic all interact.
> - **LangGraph concepts introduced**
>   - **State:** shared memory passed through the graph.
>   - **Nodes:** Python functions that read and update the state.
>   - **Edges:** fixed transitions between nodes.
>   - **Conditional edges:** decision functions that choose the next step at runtime.
>   - **Loop caps:** safeguards that prevent infinite or costly cycles.
> - **Most important technical idea**
>   - Conditional edges turn a fixed pipeline into an agentic workflow.
>   - Example: a `should_revise` function checks a score; if the score is high enough, the workflow stops, otherwise it routes back to revision.
> - **Hands-on build structure**
>   - **Part 1:** Start with a plain large language model call, then add tool calling.
>   - **Part 2:** Build a LangGraph stateful self-correcting agent.
>   - **Part 3:** Wrap the system in a Gradio interface and store results in SQLite.
> - **Tool-calling example**
>   - Shrestha demonstrates simple Python tools such as `add` and `multiply`.
>   - These functions are wrapped so the model can decide when to call them.
>   - The docstring acts like prompt engineering inside code: it tells the model when and how to use the tool.
> - **Self-correcting LinkedIn post agent**
>   - The agent takes a topic and drafts a LinkedIn post.
>   - An evaluator scores the post out of 10.
>   - If the score is below the threshold, the agent revises the draft.
>   - If the score passes the threshold, the workflow ends.
>   - The process stores the topic, score, revision count, and feedback.
> - **Structured output**
>   - [Pydantic](https://pydantic-docs.helpmanual.io/) is used to force model outputs into a predictable schema.
>   - This makes evaluation and routing more reliable because the graph can depend on fields such as `score` and `feedback`.
> - **Application layer**
>   - [Gradio](https://gradio.app/) is used for a lightweight demo interface.
>   - [SQLite](https://www.sqlite.org/) stores generated posts and metadata.
>   - Shrestha notes that Gradio and Streamlit are useful for demos, while production apps may need frameworks such as [Next.js](https://nextjs.org/).
> - **Production considerations**
>   - Choose models according to task complexity, cost, and modality.
>   - Use stronger models only when the task justifies the cost.
>   - Keep evaluation criteria specific; vague rubrics lead to unreliable outputs.
>   - Multiple specialized agents can collaborate, but each should have a narrow role.
> - **Final takeaway**
>   - Agentic AI is not just “a better chatbot.”
>   - It is a workflow architecture where a language model uses tools, maintains state, evaluates its own work, and conditionally decides what to do next.

## Introduction to the Agent2Agent (A2A) Protocol

- Holt Skinner
  - [LinkedIn](https://www.linkedin.com/in/holt/)
  - [Videos](https://goo.gle/holt-videos)
  - [Slides](https://drive.google.com/file/d/1YsoXP-kPvVFE4UczYyVvXopLHHL7DMMR/view)
  - Google Cloud

> **NOTE:**
>
> - **Main idea**
>   - Modern AI applications are moving from simple chatbots toward agentic systems that reason, plan, call tools, and manage multi-step workflows.
>   - The A2A protocol provides a standardized way for independent agents to discover, communicate, delegate, and collaborate, regardless of the framework or model used to build them.
> - **Why A2A is needed**
>   - Agent frameworks such as LangGraph, LangChain, Google Agent Development Kit, Microsoft Agent Framework, CrewAI, and others are not automatically interoperable.
>   - A2A acts as a shared communication layer so agents built by different teams, companies, or frameworks can still cooperate.
>   - The protocol keeps agents **opaque**: implementation details do not need to be exposed, only the external communication contract.
> - **Governance and ecosystem**
>   - Google introduced A2A in 2025 and later donated it to the Linux Foundation.
>   - IBM’s Agent Communication Protocol was merged into A2A.
>   - The protocol is open source, community-governed, and maintained through a technical steering committee.
>   - The speaker emphasizes using the official documentation site, because unofficial or misleading A2A sites exist.
> - **A2A versus Model Context Protocol**
>   - **Model Context Protocol (MCP)** connects agents to tools and APIs.
>   - **Agent-to-Agent (A2A)** connects agents to peer agents.
>   - MCP is usually used for deterministic tool execution; A2A is used for dynamic, non-deterministic collaboration between agentic systems.
>   - In sophisticated systems, both protocols may appear together: agents coordinate via A2A while using MCP to perform their own tool calls.
> - **Why not treat agents as tools**
>   - Tools usually expose a narrow, schema-defined function.
>   - Agents are open-ended problem solvers that can handle ambiguity, multi-step reasoning, and delegation.
>   - Treating an agent merely as a tool can reduce its expressive and operational capacity.
> - **Related protocols built around A2A**
>   - **[Agent Payments Protocol](https://a2a-protocol.org/latest/):** secure payment authorization between agents.
>   - **[Universal Commerce Protocol](https://developers.google.com/merchant/ucp):** standardized AI shopping workflows from discovery to checkout.
>   - **[Agent2UI](https://a2ui.org/):** lets agents render structured user interfaces such as buttons and forms, rather than only text.
> - **How A2A works**
>   - Each A2A agent publishes an **agent card**, a JSON description similar in spirit to `robots.txt` or an OpenAPI specification.
>   - The agent card is available at `.well-known/agent-card.json`.
>   - It describes the agent’s name, capabilities, skills, supported protocols, endpoint, and communication methods.
>   - Agents can communicate using standard web technologies such as HTTP, JSON-RPC, gRPC, and REST-style JSON.
> - **Interaction modes**
>   - **Synchronous:** for quick request-response interactions.
>   - **Asynchronous:** creates a task ID, allowing the client to poll for completion.
>   - **Streaming:** sends partial updates or artifacts as work progresses.
>   - **Push notifications:** uses callbacks or webhooks when task completion time is unknown.
> - **Core protocol objects**
>   - Messages contain roles and parts, including text, files, multimodal data, or structured data.
>   - Tasks track work status, such as submitted, working, completed, or input required.
>   - Artifacts contain returned outputs from completed tasks.
>   - Skills describe what the remote agent is capable of doing.
> - **Available tooling**
>   - The A2A ecosystem includes software development kits for Python, TypeScript/Node.js, Java, Go, C#/.NET, and Rust.
>   - A technology compatibility kit helps developers test whether their agents correctly implement the protocol.
> - **Hands-on system built in the walkthrough**
>   - The speaker builds a healthcare concierge multi-agent system.
>   - The system includes several agents implemented with different frameworks but connected through A2A:
>     - An insurance policy coverage agent.
>     - A health research agent.
>     - A healthcare provider lookup agent.
>     - A top-level healthcare concierge or manager agent.
> - **Insurance coverage agent**
>   - Uses a health insurance policy document as input.
>   - Answers questions such as the cost of mental health therapy.
>   - Uses a system prompt instructing the model to answer from the provided document and say “I don’t know” when the answer is unavailable.
>   - Is wrapped as an A2A server with an agent card describing its role and skill.
> - **Health research agent**
>   - Built with Google Agent Development Kit.
>   - Uses Google Search as a research tool.
>   - Demonstrates that some frameworks can generate A2A-compatible servers and agent cards with relatively little boilerplate.
>   - Returns general mental-health access guidance, such as crisis help, provider selection, cost management, and treatment options.
> - **Healthcare provider agent**
>   - Uses a fake provider database for the demo.
>   - Exposes a provider-search function through MCP.
>   - Connects that MCP tool to a LangChain/LangGraph agent.
>   - Wraps the resulting agent as an A2A-compatible service.
>   - Demonstrates that the same A2A client can communicate with agents implemented in different frameworks.
> - **Sequential and manager-agent architectures**
>   - The speaker distinguishes between different multi-agent designs:
>     - Sequential workflows, where output from one agent feeds another.
>     - Manager or concierge workflows, where one agent decides which specialist agents to call.
>   - The healthcare concierge agent uses the descriptions and skills in each agent card to decide which sub-agent should handle each part of a user query.
> - **Full concierge demo**
>   - The user query asks how to get mental health therapy in Austin, which providers are nearby, and what insurance covers.
>   - The concierge agent delegates to provider and insurance agents, and attempts to use the research agent.
>   - Some model calls fail because of throttling or resource exhaustion, but the top-level agent still produces a useful answer from the successful sub-agent outputs.
>   - This is presented as an example of graceful degradation in a multi-agent system.
> - **Operational lessons**
>   - Building production-grade agent systems is harder than the hype suggests.
>   - The speaker notes that many organizations are interested in agents, but relatively few have deployed them in production.
>   - Even a relatively simple multi-agent demo requires substantial integration work, careful descriptions, local servers, ports, clients, cards, and error handling.
> - **Error handling and auditability**
>   - A2A includes protocol-level mechanisms for reporting errors.
>   - Whether those errors are exposed usefully depends on the framework’s implementation.
>   - The speaker notes that not all frameworks necessarily implement every A2A error pathway correctly.
> - **Prompt-injection mitigation**
>   - Different frameworks provide different protections.
>   - The speaker mentions Google Cloud Model Armor as one available option in the Gemini Enterprise Agent Platform ecosystem.
>   - Other defenses include data loss prevention, sensitive-data detection, classification checks, virus scanning, careful prompt design, and explicit mitigation workflows.
> - **Overall takeaway**
>   - A2A is positioned as the interoperability layer for multi-agent systems.
>   - It does not replace MCP, agent frameworks, models, storage, or deployment infrastructure.
>   - Its role is to let independently built agents discover each other, describe their capabilities, exchange tasks, return artifacts, and collaborate across framework boundaries.

## Building Responsible AI Agents with Open Source

- Olivia Buzek
  - [linkedin](https://www.linkedin.com/in/olivia-buzek/)
  - [slides](https://obuzek.github.io/ai-agents-workshop/slides/#/title)
  - IBM

> **NOTE:**
>
> - The talk explains how to build **responsible AI agents**, especially when agents operate on sensitive or high-stakes data.
>   - An agent is framed as a loop of **prompt + tools + data + reasoning + repetition**.
>   - The core pattern is the **ReAct loop**: observe, plan, act, then repeat.
> - Buzek argues that custom agents are still worth building when:
>   - The workflow is repeated often.
>   - The data is sensitive.
>   - The workflow must be shared across a team.
>   - Reliability, efficiency, and auditability matter.
> - A major theme is that agents should **augment human expertise**, not replace it.
>   - Buzek warns against systems that automate routine decisions while leaving only rare exceptions to humans, because this can erode judgment.
>   - The agent should preserve human agency, critical thinking, and domain expertise.
> - The main case study is a simulated **electronic health record (EHR)** inbox.
>   - Doctors receive many patient messages, lab updates, and administrative requests.
>   - The goal is to reduce cognitive load without letting the agent practice medicine.
>   - The agent should summarize, extract, classify, organize, and surface information.
>   - It should not draft medical advice or replace clinical judgment.
> - The interface deliberately avoids a general-purpose chat box.
>   - The speaker criticizes “sparkle button” AI features that simply open a broad chat interface.
>   - Instead, the agent is embedded into a familiar EHR-style user interface.
>   - Patient concerns are extracted and shown as structured items linked back to messages and records.
> - [Lab 1](https://obuzek.github.io/ai-agents-workshop/slides/#/section-lab-1) builds a basic clinical inbox agent.
>   - It uses LangGraph and a simple ReAct-style agent loop.
>   - Tools allow the agent to retrieve patient records and messages.
>   - The agent produces structured “patient concern” outputs.
>   - Buzek emphasizes **structured outputs** over asking the model to “return JSON,” because constrained decoding is more reliable and easier to integrate with deterministic software.
> - The first implementation works but exposes serious risks.
>   - The agent may pull an entire patient record into context.
>   - That means protected health information and personally identifiable information can enter model traces, logs, or tool contexts.
>   - There are no strong access controls.
>   - Generated concerns are unstable: rerunning the agent may produce a different concern list.
>   - There are no hallucination, completeness, or task-boundary checks.
>   - A careless memory system could mix information across patients.
> - [Lab 2](https://obuzek.github.io/ai-agents-workshop/slides/#/section-lab-2) adds observability with Langfuse.
>   - Observability is needed because traditional monitoring does not show whether the agent made good reasoning or tool-use decisions.
>   - Langfuse traces show prompts, tool calls, latency, costs, and model behavior.
>   - The traces reveal a key problem: observability itself can leak protected health information.
> - Buzek introduces masking for sensitive data.
>   - Microsoft Presidio is used for named-entity-based masking of personally identifiable information and protected health information.
>   - Masking helps, but Buzek stresses that it is not sufficient by itself.
>   - Even redacted traces may need strict access control.
>   - Masked traces may be useful for synthetic data generation, evaluation datasets, and debugging.
> - [Lab 3](https://obuzek.github.io/ai-agents-workshop/slides/#/section-lab-3) improves safety and grounding.
>   - The agent is changed so it no longer retrieves the whole patient record by default.
>   - Instead, it gets narrower tools: demographics, medications, conditions, labs, and messages.
>   - This makes the agent search for relevant evidence rather than dumping all data into context.
>   - Retrieval-augmented generation is discussed as an alternative, but it still risks injecting sensitive data.
> - The system adds hallucination and task checks.
>   - Claims made by the agent are extracted and checked for grounding in the record.
>   - A critic loop evaluates whether generated concerns are supported and on task.
>   - Failed outputs can be sent back for revision.
>   - [IBM Granite Guardian](https://www.ibm.com/granite/docs/models/guardian) is shown as a smaller, local model option for groundedness and harm checks.
> - Buzek distinguishes between different *evaluation roles*.
>   - A large language model as judge can check claims, but it costs tokens and requests.
>   - A smaller local model may be cheaper and more independent.
>   - Model choice matters greatly because different models behave differently inside the same agentic workflow .
> - [Lab 4](https://obuzek.github.io/ai-agents-workshop/slides/#/section-lab-4) addresses persistence and access control.
>   - Agent outputs are treated as derived protected health information.
>   - Therefore, they should inherit the same security policy as the underlying patient data.
>   - Buzek recommends structured storage, such as Postgres with row-level security.
>   - Access should be denied by default and granted only according to patient, provider, and concern-level permissions.
> - The final system introduces more stable concern management.
>   - Instead of regenerating a fresh concern list every time, the agent checks prior concerns.
>   - It can revise, discard, or create concerns based on new information.
>   - This makes the agent’s outputs more useful as part of an ongoing clinical workflow.
> - The overall lesson is that production agents are not just prompts and tools.
>   - They require observability, evaluation, access control, structured outputs, persistence, grounding, and human-centered design.
>   - In sensitive domains, the central engineering problem is not “how to make an agent,” but how to make one that is constrained, auditable, privacy-preserving, and genuinely useful.

> **NOTE:**
>
> **ReAct loop**: a common agentic pattern where the agent iteratively observes, plans, acts, and repeats.
>
> - Observe: the agent observes the current state, including inputs and tool outputs.
> - Plan: the agent formulates a plan based on its observations.
> - Act: the agent executes actions according to its plan.
> - Repeat: the agent iterates through the loop, continuously updating its understanding and actions.

> **NOTE:**
>
> **Guardian Pattern**: a design pattern for responsible AI agents where a smaller, local model (the “guardian”) evaluates the outputs of a larger, more powerful model to check for 1. grounding 2. relevance 3. safety before allowing those outputs to affect downstream processes.

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2026,
  author = {Bochman, Oren},
  title = {ODSC {AI} 2026},
  date = {2026-04-27},
  url = {https://orenbochman.github.io/posts/2026/04-27-ODSC-AI-2026-Day-0/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2026. “ODSC AI 2026.” April 27. <https://orenbochman.github.io/posts/2026/04-27-ODSC-AI-2026-Day-0/>.
