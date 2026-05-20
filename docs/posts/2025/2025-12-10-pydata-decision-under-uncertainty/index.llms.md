[![pydata global](pydata_logo.png)](pydata_logo.png "pydata global")

pydata global

# An error occurred.

Unable to execute JavaScript.

Making decisions under uncertainty with applications in Bayesian optimization (PoI/EI) and Bayesian experimental design

> **TIP:**
>
> We often must make decisions under uncertainty—should you carry an umbrella if there’s a 30 % chance of rain? Bayesian decision theory provides a principled, probabilistic framework to answer such questions by combining beliefs (probabilities), utilities (what matters to us), and actions to maximize expected gain.

This talk:

- Introduces key decision‑theoretic concepts in intuitive terms.
- Uses a toy umbrella example to ground ideas in relatable context.
- Demonstrates applications in Bayesian optimization (PoI/EI) and Bayesian experimental design.
- Is hands‑on—with Python code and practical tools—so participants leave ready to apply these ideas to real‑world problems.

> **TIP:**
>
> - This talk bridges everyday decision-making (umbrella example) with advanced techniques like
> - Bayesian optimization and
> - Experimental design, and equips attendees with conceptual clarity and immediate code they can adapt to their data-driven workflows.

> **TIP:**
>
> Primarily data scientists, ML practitioners, and statisticians who:
>
> - Have applied Bayesian models but want a broader decision-theory perspective.
> - Want actionable insight into uncertainty-aware decision frameworks.
> - Seek practical demos in Python.

> **IMPORTANT:**
>
> - [GPyTorch](https://gpytorch.ai/) for Gaussian processes
> - [OptBayesExpt](https://github.com/usnistgov/optbayesexpt) for Bayesian experimental design

[workshop repo](github.com/KrisNguen135/Talks)

> **TIP:**

## Outline

This video provides a hands-on guide to Bayesian decision theory, explaining how to make decisions under uncertainty by combining beliefs (probabilities), utilities (what matters), and actions to maximize expected gain.

**Understanding Uncertainty**: Nguyen illustrates how daily scenarios, like deciding whether to carry an umbrella, or technical challenges, like tuning machine learning model parameters, involve decision-making under uncertainty. This is challenging due to unpredictable future events and varying costs/utilities in different outcomes.

[![](slide01.png)](slide01.png "Figure 1: Decision Under Uncertainty - A Guide to Bayesian Decision Theory")

Figure 1: Decision Under Uncertainty - A Guide to Bayesian Decision Theory

[![](slide02.png)](slide02.png "Figure 2: Who I am")

Figure 2: Who I am

[![](slide03.png)](slide03.png "Figure 3: Hyperparameters tuning overview")

Figure 3: Hyperparameters tuning overview

### Motivation & Core Concepts (5 min)

Quantifying Decisions: Bayesian decision theory involves quantifying uncertainties and costs. The “umbrella example” is used to demonstrate how to create a probability and associated cost table, assigning probabilities to scenarios (e.g., rain or no rain) and costs to actions (e.g., bringing an umbrella or not).

- Frame real-world decision problems: rain or shine, clinical trials, A/B testing.
- Introduce Bayesian decision theory: beliefs \times utilities \to action via expected utility maximization.

[![](slide04.png)](slide04.png "Figure 4: What makes these problems hard")

Figure 4: What makes these problems hard

- **Uncertainty**: we don’t know the future (e.g., ? ).
  - will it rain
  - how good each hyperparameter combination will be?
  - how each participant will respond to a question?

- **Utility/cost**: we value each scenario differently
  - rain is more troublesome
  - hyperparameter yields good accuracy are better
  - “informative” answers is better

## Toy Example: Should I Bring an Umbrella? (8 min)

**Optimizing Expected Utility/Cost**: The core principle is to choose actions that lead to the lowest expected cost or highest expected utility. This involves calculating a weighted average of costs or utilities based on their probabilities. For instance, in the umbrella example, bringing an umbrella is the optimal action if it results in a lower expected cost.

[![](slide05.png)](slide05.png "Figure 5: Should I Bring an Umbrella")

Figure 5: Should I Bring an Umbrella

- Define: Probability p of rain; utility/loss matrix

| Action      | Rain         | No Rain            |
|-------------|--------------|--------------------|
| Umbrella    | –1 (weight)  | –1 (inconvenience) |
| No Umbrella | –10 (soaked) | 0                  |

- Derive expected utility:

`EU_umbrella = -1` `EU_no_umbrella = -10p`

So bring umbrella if p \> 0.1

[![](slide06.png)](slide06.png "Figure 6: Quantifying the specifics")

Figure 6: Quantifying the specifics

[![](slide07.png)](slide07.png "Figure 7: Coming up with a good decision")

Figure 7: Coming up with a good decision

[![](slide08.png)](slide08.png "Figure 8: Interactive Demo")

Figure 8: Interactive Demo

- Interactive Python demo: explore how p and utility values shift the decision point.

### Bayesian Optimization: PoI & EI (8 min)

- Introduce Gaussian-process-based optimization and the need to trade off exploration vs. exploitation.
- Define Probability of Improvement (PoI) and Expected Improvement (EI)
- Show how they’re derived from decision theory: choosing the next point to maximize expected gain.

**Personalization of Bayesian Decision Theory**: The theory is highly personalizable, as the optimal decision depends on an individual’s beliefs about the world and their personal preferences regarding costs and utilities. This is demonstrated by showing how varying the probability of rain or the perceived cost of getting wet can alter the optimal decision.

[![](slide09.png)](slide09.png "Figure 9: The personalization of Bayesian decision theory")

Figure 9: The personalization of Bayesian decision theory

[![](slide10.png)](slide10.png "Figure 10: The personalization of Bayesian decision theory")

Figure 10: The personalization of Bayesian decision theory

[![](slide11.png)](slide11.png "Figure 11: Bayesian Optimization")

Figure 11: Bayesian Optimization

**Applications in Bayesian Optimization**: The theory is applied to Bayesian optimization, which aims to find the optimal inputs for “black-box” functions that are expensive to evaluate. Examples include hyperparameter tuning in machine learning and scientific discovery, where minimizing the number of evaluations is crucial.

- Python demo using GPyTorch: fit GP, compute PoI/EI acquisition functions, visualize decision boundary—why one chooses a high-uncertainty point vs. one near known good values.

[![](slide13.png)](slide13.png "Figure 12: Decision-theoretic optimization")

Figure 12: Decision-theoretic optimization

[![](slide14.png)](slide14.png "Figure 13: Decision-theoretic optimization")

Figure 13: Decision-theoretic optimization

### Bayesian Experimental Design (BED): Minimizing Uncertainty (8 min)

**Applications in Experimental Design**: Bayesian decision theory can also be used in experimental design to efficiently learn about unknown functions or quantities. By strategically choosing experiments, one can minimize uncertainty (e.g., using Shannon entropy as a cost function) and gather information as quickly as possible.

- Motivation: cost-sensitive data collection (labeling, surveys, medical tests).
- Define an information-based utility (e.g., expected reduction in entropy).
- Show how decision theory prescribes choosing the next experiment to maximize this expected utility.

[![](slide15.png)](slide15.png "Figure 14: Experimental Design")

Figure 14: Experimental Design

[![](slide20.png)](slide20.png "Figure 15: Decision-theoretic learning")

Figure 15: Decision-theoretic learning

- Python demo using [OptBayesExpt](https://github.com/usnistgov/optbayesexpt).

[![](slide21.png)](slide21.png "Figure 16: Decision-theoretic learning")

Figure 16: Decision-theoretic learning

[![](slide22.png)](slide22.png "Figure 17: Decision-theoretic learning")

Figure 17: Decision-theoretic learning

- Thanks for the free e-book !

### Summary & Takeaways (1 min)

- Reiterate the decision-theoretic arc: belief → utility → action.
- Emphasize the unifying framework across umbrella example, optimization, and experimental design.
- Share resources & practical tips: GPyTorch / scikit-optimize, OptBayesExpt

[![](slide23.png)](slide23.png "Figure 18: Takeaway")

Figure 18: Takeaway

[![](slide24.png)](slide24.png "Figure 19: What can go wrong")

Figure 19: What can go wrong

[![](slide25.png)](slide25.png "Figure 20: More resources")

Figure 20: More resources

## Reflection

- TODO
  - replace the static slides with shinylive demos based on the interactive notebooks in the talk repo. This will make the content more engaging and allow readers to experiment with the concepts in real time.
  - cover previous talks by Quan Nguyen
  - summary of the Bayesian optimization book or integration into Bayesian optimization as a continuation of the Bayesian stats serialization in my notes.

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2025,
  author = {Bochman, Oren},
  title = {Decisions {Under} {Uncertainty:} {A} {Hands‑On} {Guide} to
    {Bayesian} {Decision} {Theory}},
  date = {2025-12-10},
  url = {https://orenbochman.github.io/posts/2025/2025-12-10-pydata-decision-under-uncertainty/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2025. “Decisions Under Uncertainty: A Hands‑On Guide to Bayesian Decision Theory.” December 10. <https://orenbochman.github.io/posts/2025/2025-12-10-pydata-decision-under-uncertainty/>.
