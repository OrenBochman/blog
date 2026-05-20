## The Changing Shape of AI Systems - From Monolithic Training to Continuous Adaptation

Play

![Vimeo](https://f.vimeocdn.com/p/images/crawler_logo.png)

- Sudip Roy
  - Adaption
  - adaptive data: creating niche datasets for underserved domains and languages;
  - adaptable intelligence: enabling systems to evolve from user feedback;
  - adaptable interfaces: moving beyond rigid chat boxes toward task-specific interfaces.

> **NOTE:**
>
> - The Speaker argues that AI systems have historically been built as static artifacts: train once, ship, freeze, and optimize mainly around inference cost and latency.[^1]
> - Traditional machine learning systems were often application-specific, smaller, and easier to retrain continuously, sometimes daily or weekly.
> - Foundation models changed this architecture:
>   - one large model serves many downstream tasks;
>   - training cost is concentrated into a single expensive pretraining run;
>   - the deployed model is usually frozen for months;
>   - most post-deployment engineering focuses on inference optimization.
> - The deployed “unit” of AI has grown over time:
>   - from a single stateless model;
>   - to compound systems with retrieval, databases, verification, and guardrails;
>   - to agentic systems that call tools, interact with environments, and receive feedback.
> - The speaker’s central criticism is that even agentic systems usually do not truly learn from deployment-time failures. A customer-support agent may fail today, log the failure, and still fail the same way tomorrow.
> - This creates a growing inefficiency:
>   - agentic interactions require many model calls, tool calls, and retries;
>   - failures generate useful feedback;
>   - current serving stacks are not designed to feed that feedback back into the model or system dynamically.
> - The main open question is: how can a deployed model improve over time?
> - Several candidate mechanisms are mentioned:
>   - fine-tuning;
>   - reinforcement learning from human feedback (RLHF);
>   - reinforcement learning;
>   - online learning;
>   - continual learning;
>   - memory systems.
> - The first major design question is where learning should live:
>   - Non-parametric memory stores changing information outside the model, such as retrieval systems or databases. It is cheap and dynamic but not true model learning.
>   - Parametric memory updates the model weights themselves. It is durable and low-latency at inference time, but expensive and slow to modify.
>   - The speaker suggests a hybrid: stable knowledge belongs more naturally in parameters, while ephemeral knowledge belongs outside the model.
> - The second major design question is compute allocation:
>   - Increasingly, much of the runtime cost is outside the model itself.
>   - Retrieval, search, verification, tool execution, and environment simulation can consume 30–50% or more of application time.
>   - Therefore, optimization should target the full “model plus harness,” not only the neural model.
> - The third major design question is governance:
>   - Static models can be versioned, red-teamed, and evaluated before deployment.
>   - Continuously adapting models complicate versioning, monitoring, reproducibility, rollback, and safety guarantees.
>   - A key concern is preventing uncontrolled behavioral drift.
> - A future adaptive support agent should:
>   - learn from yesterday’s failures;
>   - improve cheaply and frequently;
>   - make changes reversibly;
>   - provide an auditable contract about how far it is allowed to drift.
> - The speaker summarizes three architectural shifts:
>   - workloads are moving from static inference to agentic interaction;
>   - learning is moving from fixed memory toward hybrid parametric and non-parametric memory;
>   - compute is moving from the model alone to the model plus surrounding harness.
> - The broader thesis is that AI systems are evolving from static artifacts into living systems that continuously adapt as the world changes.
> - In Q&A, the speaker is asked “whether hybrid learning implies neuro-symbolic architecture”. He does not commit to that framing, but suggests that enterprise ontologies or structured representations may form part of the stable world model, while more transient facts remain in non-parametric memory.

### Reflection

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2026,
  author = {Bochman, Oren},
  title = {The {Changing} {Shape} of {AI} {Systems}},
  date = {2026-04-28},
  url = {https://orenbochman.github.io/posts/2026/04-30-ODSC-AI-2026-Day-3/talk13.html},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2026. “The Changing Shape of AI Systems.” April 28. <https://orenbochman.github.io/posts/2026/04-30-ODSC-AI-2026-Day-3/talk13.html>.

[^1]: looks like he needs to hear about RL
