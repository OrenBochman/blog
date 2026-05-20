## Towards Trustworthy LLMs: Understanding Limits, Advancing Capabilities, Ensuring Safety

Play

![Vimeo](https://f.vimeocdn.com/p/images/crawler_logo.png)

- Nouha Dziri
  - [LinkedIn](https://www.linkedin.com/in/nouha-dziri-3587427b//)
  - Cohere Labs
  - [slides](https://drive.google.com/file/d/1ORh6y6yqX8ZhNB6t5krQEys7mro3UcBU/view?usp=sharing)

> **NOTE:**
>
> - The Keynote focuses on building **trustworthy large language models** by understanding and improving reasoning, reliability, and safety.
> - **Core problem**
>   - Modern large language models are increasingly capable and autonomous, but their intelligence remains **jagged**.
>   - They can solve very difficult tasks, such as Olympiad-style problems, while still failing on simple but unfamiliar tasks, such as large-digit arithmetic without tools.
>   - Dziri argues that this unreliability is largely due to weak **out-of-distribution generalization**.
> - **Definition of reasoning**
>   - Reasoning is framed as drawing conclusions efficiently by composing learned concepts.
>   - Two key properties are emphasized:
>     - **Extrapolation**: generalizing beyond the training distribution.
>     - **Efficiency**: solving problems with less data, smaller models, and more structural understanding rather than brute-force memorization.
> - **Understanding model reasoning**
>   - Dziri describes research that represents a model’s chain of thought as a **computational graph**.
>   - The finding is that transformers often collapse multi-step reasoning into **subgraph matching**.
>   - This suggests that many successes are linked to whether relevant computational fragments were already present in training data.
>   - Models can appear to solve complex tasks, but may actually be reusing familiar patterns rather than discovering genuinely novel solutions.
> - **Limits of current reasoning**
>   - Large language models can generalize somewhat, but not at the level of human reasoning.
>   - They operate on a spectrum between pattern matching and genuine novelty.
>   - Pattern matching is still treated as a form of reasoning, but it is not sufficient for robust creativity or deep extrapolation.
> - **Reinforcement learning and reasoning**
>   - The talk discusses **reinforcement learning**, especially **Group Relative Policy Optimization (GRPO)**, in the context of models like DeepSeek-R1.
>   - Reinforcement learning can improve performance beyond supervised fine-tuning on tasks similar to training data.
>   - However, its gains decrease as task novelty increases.
>   - Sparse rewards, such as giving only pass/fail feedback at the end of a long solution, are inadequate for discovering difficult new reasoning strategies.
> - **Problem with sparse rewards**
>   - A model may get 80% of a reasoning process correct but receive zero reward if the final answer is wrong.
>   - Conversely, a model may reach the right answer through poor reasoning and receive full reward.
>   - This can reinforce bad trajectories and fail to teach the model where its reasoning succeeded or failed.
> - **Dense reward proposal**
>   - Dziri argues for **dense rewards**, where intermediate reasoning steps are evaluated and rewarded.
>   - In coding tasks, this can be approximated using unit tests that check individual functions or features.
>   - This gives the model partial credit for partial progress, creating a richer learning signal.
> - **Delta dataset and experiments**
>   - Dziri introduces a dataset called **Delta**, designed to contain tasks unlikely to have appeared in training data.
>   - Example task families include:
>     - A puzzle game involving factories that sort robots.
>     - **BounceSim**, a two-dimensional bouncing-ball simulation task used as a proxy for geometry-aware reasoning.
>   - With sparse rewards, models failed because almost all training rollouts received zero reward.
>   - With dense-reward warm-up, models learned useful subskills, rising from zero to around 80%.
>   - After switching back to binary reward, the model eventually converged to full solutions, described as a “grokking moment.”
> - **Conclusion on reinforcement learning**
>   - Reinforcement learning can both sharpen existing skills and help models discover new ones, depending on the setup.
>   - Success depends on the reward design, task hardness, data mixture, rollout infrastructure, and training recipe.
>   - Dziri emphasizes that experimental setup can strongly affect whether reinforcement learning appears powerful or ineffective.
> - **Efficiency remains unsolved**
>   - Despite progress in extrapolation, the field still relies heavily on large models, massive datasets, expensive compute, and costly inference.
>   - Dziri argues for “smarter scaling” rather than continued brute-force scaling.
> - **Safety and security**
>   - Dziri notes that the same out-of-distribution weakness affects safety.
>   - Models can refuse obvious harmful prompts but comply when the same request is phrased adversarially or unusually.
>   - This suggests that safety behavior is often shallow pattern recognition rather than deep understanding.
> - **Jailbreaking and adversarial training**
>   - Dziri describes adversarial jailbreak methods that increased attack success rates on frontier models.
>   - Adversarial data can be used to train safer models and reduce attack success on benchmarks.
>   - However, new attacks continue to emerge, creating an ongoing attack-defense race.
> - **Safety as a continuous process**
>   - Safety cannot be treated as a final fine-tuning step before release.
>   - It must be integrated across:
>     - Pre-training.
>     - Post-training.
>     - Inference-time monitoring.
>     - Ongoing stress testing and defenses.
> - **Agentic AI**
>   - The talk ends by noting that future systems will increasingly plan, act, and adapt autonomously.
>   - Some reasoning failures can be mitigated by agents using tools, retrieval, verification, and interaction with the environment.
>   - Dziri says her current work focuses on these agentic systems.
> - **Q&A**
>   - In response to a question about rewards, Dziri explains that traditional reinforcement learning gives reward after the model response, usually as correct or incorrect.
>   - Dziri argues that dense reward is more like a teacher giving detailed feedback on where a student succeeded or failed.
>   - In response to a question about analogical thinking, Dziri says partial rewards could potentially be combined with natural-language feedback, hints, or analogical explanations to improve generalization.

### Reflection

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2026,
  author = {Bochman, Oren},
  title = {Towards {Trustworthy} {LLMs}},
  date = {2026-04-28},
  url = {https://orenbochman.github.io/posts/2026/04-30-ODSC-AI-2026-Day-3/talk1.html},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2026. “Towards Trustworthy LLMs.” April 28. <https://orenbochman.github.io/posts/2026/04-30-ODSC-AI-2026-Day-3/talk1.html>.
