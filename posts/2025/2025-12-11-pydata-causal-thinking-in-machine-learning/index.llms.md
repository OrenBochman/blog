# An error occurred.

Unable to execute JavaScript.

> **TIP:**
>
> Most ML models excel at prediction, answering questions like “Who will buy our product?” or “Which customers are likely to churn?”. But when it comes to making actionable decisions, prediction alone can be misleading. Correlation does not imply causation, and business decisions require understanding causal relationships to drive the right outcomes.
>
> In this talk, we will explore how causal machine learning, specifically uplift modeling, can bridge the gap between prediction and decision making. Using a real-world use case, we will showcase how uplift modeling helps identify who will respond positively to interventions while avoiding those who they might deter.

Predictive ML models are used everywhere for data-driven decision making across industries. However, accurate forecasts don’t always translate to optimal actions.

We will begin by exploring the fundamental challenges of deriving actions from model predictions, especially when determining the right audience to target. After that, we will dive into some fundamental concepts of causal inference and how it differs from traditional ML. We will then introduce uplift modeling and cover some key concepts, e.g., treatment effects, counterfactuals, meta-learning approaches, etc. We will see how these elements work together to create causal ML models.

Finally, we will put theory into practice by building a sample uplift model in Python. We’ll walk through each step using real-world intervention data (publicly available), demonstrating how this approach can dramatically improve decision-making and ensure that the interventions target the right audience for the right reasons.

> **TIP:**
>
> - Attendees will learn when to use causal thinking vs predictive modeling and how to implement uplift models using Python.
> - They will also understand how to apply these techniques across different domains, such as marketing, healthcare, and other relevant fields.

> **TIP:**

> **TIP:**

- [talk repo](https://github.com/hugobowne/AI-for-SWEs)
- [slide deck](https://github.com/hugobowne/AI-for-SWEs)

## Outline

- Introduction and motivation \[1 min\]
- From correlation to causation \[4 min\]
  - Correlation vs Causation
  - When do we need a causal angle
- Core causal concepts \[4 min\]
  - Treatment effects
  - Counterfactuals
  - Intervention problem
- Uplift modeling concepts \[5 min\]
  - Four types of individual responses to a treatment
  - Meta learning approach
  - T-Learner and S-Learner comparison
- Hands-on case study \[10 min\]
  - Problem explanation and formulation
  - Predictive model output
  - Causal uplift model in Python
  - Compare targeting strategies and intervention impact
- Evaluation \[4 min\]
  - Why accuracy or F1 scores don’t work for uplift
  - Uplift curves
  - Qini coefficient
  - Explainability
- Practical Considerations \[2 min\]
  - A/B testing treatment effects
- Cross-domain applications

[![](slide01.png)](slide01.png)

[![](slide02.png)](slide02.png)

[![](slide03.png)](slide03.png)

[![](slide04.png)](slide04.png)

[![](slide05.png)](slide05.png)

[![](slide06.png)](slide06.png)

[![](slide07.png)](slide07.png)

[![](slide08.png)](slide08.png)

[![](slide09.png)](slide09.png)

[![](slide10.png)](slide10.png)

[![](slide11.png)](slide11.png)

[![](slide12.png)](slide12.png)

[![](slide13.png)](slide13.png)

[![](slide14.png)](slide14.png)

[![](slide15.png)](slide15.png)

[![](slide16.png)](slide16.png)

[![](slide17.png)](slide17.png)

[![](slide18.png)](slide18.png)

[![](slide19.png)](slide19.png)

[![](slide20.png)](slide20.png)

[![](slide21.png)](slide21.png)

[![](slide22.png)](slide22.png)

[![](slide23.png)](slide23.png)

[![](slide24.png)](slide24.png)

[![](slide25.png)](slide25.png)

[![](slide26.png)](slide26.png)

[![](slide27.png)](slide27.png)

[![](slide28.png)](slide28.png)

[![](slide29.png)](slide29.png)

[![](slide30.png)](slide30.png)

[![](slide31.png)](slide31.png)

[![](slide32.png)](slide32.png)

[![](slide33.png)](slide33.png)

[![](slide34.png)](slide34.png)

[![](slide35.png)](slide35.png)

[![](slide36.png)](slide36.png)

[![](slide37.png)](slide37.png)

[![](slide38.png)](slide38.png)

## Reflections

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2025,
  author = {Bochman, Oren},
  title = {Beyond {Just} {Prediction}},
  date = {2025-12-12},
  url = {https://orenbochman.github.io/posts/2025/2025-12-11-pydata-causal-thinking-in-machine-learning/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2025. “Beyond Just Prediction.” December 12. <https://orenbochman.github.io/posts/2025/2025-12-11-pydata-causal-thinking-in-machine-learning/>.
