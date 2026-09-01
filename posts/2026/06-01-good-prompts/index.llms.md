## Overview

- The reality is that LLM were not very good at following instructions. OpenAI and other vendors have ignored the harm this is causing and frequently making the problem worse by making models appear more certain and authoritative than they really are.
- Even today when they have improved, we still get plenty of issues with them not following instructions, hallucinating, or just not doing what we want. This is harder to see since when a session starts the LLM is more likely to behave and only later does it become more likely to go off the rails. Some of the innovations that have led to the improvements are:
  - Reinforcement learning from human feedback (RLHF) - which also makes the models overconfident and less likely to admit when they are wrong.
  - Chain of thought - agents have a budget of tokens to think before they create the final answer. This was one of the common techniques originally used in prompt engineering.
  - Tool use - giving the model access to tools (e.g. search, code execution, calculator, etc.) for task that Models are not good at but there are tools that are better at this.
  - Coding - agents may be weak at counting and statistics but they are good at writing code that can do this.
  - Steering - giving the model instructions after the prompt to steer it to do what we want if a chain of thought seems to be going astray.
- In the end LLMs are black box models meaning that none can ever know what is causing them to act out as they do.
- Unfortunately there is lots of (mis)information out there on prompting. I think the term “prompt engineering” is a nasty kind of misnomer to draw attention prompting it is an art and that it makes us work hard to mitigate the issues that LLM vendors have not been addressing in their quest to become solvent before they destroy the world…
- All we really had for a long time is an apocrypha of advice on how to write prompts and interact with the LLM to get the best results.
- Today we can try and do better by taking over the harness and using programmatic mitigation to many of the challenges of working with LLMs.
- Recently I viewed an extensive presentation in an OSDS conference by Sheamus McGovern [Prompt Design & Engineering Course](https://docs.google.com/presentation/d/1g_WHYFRsuiy-Fz87GCY805Sl837Lt_GB-q1xXsRCnHc/edit?slide=id.g346a7c1fee0_0_41#slide=id.g346a7c1fee0_0_41). The long presentation gave me a time out to rethink the topic. I had taken a course online on Andrew Ng’s platform and most of what that has suggested is still relevant. And now we are in an age where LLMs are now being used in agentic forms for ever more complex workflows.

And I came across a page I had in an old project called “good prompts” I decided to share it here in updated form. What is becoming clear is that as we want to get LLM to work for us in more complex and predictable ways we need to figure out how to work out many of the kinks the LLM vendors are not addressing. Also it is become increasingly expensive to work with LLMs in projects that create real value. This has led the industry to save thier prompts in version control systems and treat them more like small programs that need QA and checks to ensure they do not break or regress as the LLMs vendors downgrade or use more radical cost cutting measures to make their models more profitable.

> **NOTE:**
>
> [![good-prompts in a nutshell](../../../images/in_the_nut_shell_gemeni.png)](../../../images/in_the_nut_shell_gemeni.png "good-prompts in a nutshell")
>
> good-prompts in a nutshell
>
> - How do we mitigate the issues of working with LLMs and get them to do what we want?
> - To some extent we can write prompts.
> - For tasks thar LLMs are unsuited we should outsource to tools.
> - For more structured output we should use prompt chaining.
> - For complex workflows we should use orchestration tools for the prompt chaining.

This is a minimal checklist for writing good prompts and evaluating them. Some of this material comes from a talk called [the prompting playbooks](https://www.youtube.com/watch?v=G2B0YWuJUgI).

> **NOTE:**
>
> 0.  Start by keeping your prompt in \*.prompt.md files and in version control. This lets you keep track of mitigation and improvements to the prompt over time.
> 1.  start with a minimal prompt
>     - role
>     - data/documents (few-shot examples, relevant documents, RAG).
>     - task
>     - output shape
> 2.  run evaluations and look at failure modes
>     - for failures aim for a fix that generalizes beyond the specific test case
> 3.  add to the prompt only to address failures
>     - prefer structure (field, tool, criterion) over exhortation (NEVER, ALWAYS, CRITICALLY)
>     - avoid long bans lists
> 4.  When structure handling is needed (classify-then-act, plan-then-execute) move to prompt-chaining, a system with multiple model calls rather than numbered steps in a single prompt.
>     - via programmatic prompt chaining (Python/Pydantic/baml) for linear chains of thought and tool use.
>
>     - via orchestration tools [LangSmith](https://smith.langchain.com/), [LangFlow](https://www.langflow.org/), [LangChain](https://www.langchain.com/), [LangGraph](https://www.langchain.com/langgraph), [Instructor](https://python.useinstructor.com/) [Temporal](https://temporal.io/) etc. for more complex structures and branching.
>
>     - note that we can also think about RL here i.e. learning a policy to from experience to improve the prompt structure to avoid failures, reduce tokens and latency.
> 5.  Testing dimensions (what goes in a testing dashboard)
>     - check: LLM as a Judge v.s Deterministic checkers
>     - Model used (models change over long term and short term and different task need different model capabilities so think: fast, thinking, coding, multimodal, etc.)
>     - number of runs/violations
>     - pass/fail counts
>     - tokens,
>     - latency
> 6.  Asking for user input
>     - when to ask for user input
>     - how to ask for user input (what format, how to use it in the prompt)
> 7.  Referencing tool use in the body
>     - \#tool:
>     - \#tool:browser
>     - \#tool:vscode/askQuestion
>     - \${input:variableName}
>     - \${input:variableName:placeholder}
> 8.  use built in variables
> 9.  converting prompts to `tmx` form to work on models with different limited context windows
> 10. checking output compliance using [pydantic](https://pydantic-docs.helpmanual.io/) or [baml](https://baml.readthedocs.io/) schemas.
> 11. checking using another model. LLM as a judge.

{{\< import \_code_ollama_local.qmd \>}}

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2026,
  author = {Bochman, Oren},
  title = {All the {Good} {Prompts}},
  date = {2026-06-03},
  url = {https://orenbochman.github.io/posts/2026/06-01-good-prompts/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2026. “All the Good Prompts.” June 3. <https://orenbochman.github.io/posts/2026/06-01-good-prompts/>.
