This is a follow up to the previous post in the same spirit of the post on Mixture problems. I’ve recently implemnted some switching models and was delighted to see how the model quickly outperformed the established model. However, I was surprised to find that while kalman filtering is ubiqutios, switching models are far from being so. This is a note at what is available and what are the challenges in smoothing for switching models.

The Kalman filter and the DLA are two popular algorithms for baysian smoothing and filtering in state-space models, which are a class of probabilistic models that describe the evolution of a system over time.

A third noteworthy algorithm in this space geared towards target tracking is the Interacting Multiple Model (IMM) algorithm, which is designed to handle systems that can switch between different modes of operation. The IMM algorithm combines multiple models, each representing a different mode of the system, and uses a probabilistic approach to determine which model is most likely to be active at any given time.

Thus the IMM efficently performs smoothing and filtering for switching models, which are a type of state-space model that can switch between different modes of operation. The IMM algorithm is particularly useful for tracking targets that can exhibit different behaviors, such as a vehicle that can switch between different driving modes (e.g., cruising, accelerating, braking) or an aircraft that can switch between different flight modes (e.g., climbing, descending, cruising).

However the variant used in targer tracking is primarily designed for filtering. That is it can quickly figure the most likely driving mode at each time step and quickly adapt to changes so as to provide a good estimate of the state of the system at that time step.

It turns out that smoothing for switching models is much more difficult. Recall that smoothing is the problem of estimating the state of the system at a given time step, given all the observations up to that time step. In contrast to filtering, which only uses the observations up to the current time step, smoothing can use future observations to improve the estimate of the state at a given time step. For a regular state-space model, the Kalman smoother needs to recover a single state at each time step. But for a switching model, the bayesian smoother needs to store a likelyhood for each possible mode at each time step. This might not sound so bad but consider that these likelihoods can be path dependent. This means that in the worst case one must consider all possible permutation of modes across all time steps, which has combinatorial complexity and quickly becomes intractable as the number of time steps and modes increases.

Even if we make a simplifying assumption that the likelihoods are not path dependent, perhaps even just Markovian, we still have a highly bifurcating tree of possible modes gooing backwards in time that we need to inform out future estimates. And even if the states are sparse[^1] and sticky[^2] we still have a large number of possible mode sequences to consider.

Going forward we should be able to use the markov assumption to aggregate the likelihoods of the different modes at each time step. But I don’t see how we could avoid the combinatorial complexity of considering many histories going backwards in time… Well to be honest there *are a few ideas but I am skeptical about their viability*.

\<!– using viterbi to identify what modes transition are unlikely. using a data structures that allows to efficently store a filtering and smoothing prior so that going back and forth becomes a conjugate update….

Now it turns out that this is not a serious problem for target tracking because the emphasis is on filtering and the number of modes is typically small. But in other applications, such as econometrics or neuroscience, where the number of modes can be large and the emphasis is on smoothing, this can be a significant challenge.

## Approaches to smoothing for switching models

# An error occurred.

Unable to execute JavaScript.

Tatiana Kirsanova - On Bayesian Filtering for Markov Regime Switching Models [slides](https://econrsa.org/download/36029/?tmstv=1722250275)

In these cases, researchers have began to develop approches that reduce the complexity of smoothing for switching models to keep the model tractable. The main idea is to use a markov state model which does away with the path dependence by assuming that the transition probabilities between modes are Markovian, which allows for more efficient computation of the likelihoods and makes smoothing more tractable.

This types of models are disscussed in ([Frühwirth-Schnatter 2006](#ref-fruhwirth2006finite)) which is a great introduction to the topic.

Tatiana Kirsanova, in the lined video covers the connection and diffrences between the engineering and econometrics approaches. She discusses some ideas her groups has developed to make smoothing for switching models more tractable. c.f. “Filtering and Smoothing in State-Space Models with Multiple Regimes” (2026), with Nigar Hashimzade, Oleg Kirsanov and Junior Maih, Journal of Business and Economic Statistics, forthcoming….

Another promising direction based on dynamic programming approach (allowing for nonlinearity, non-Gaussianity and degeneracy in the observation and/or state-transition equations) is presented in [Bellman filtering and smoothing for state-space models](https://arxiv.org/abs/2008.11477) c.f.([Lange 2024](#ref-Lange_2024)).

The main approaches to smoothing for switching models are:

The trick used in the IMM for filtering is to combine the likelihoods of the different modes at each time step, which allows it to efficiently compute the most likely mode at each time step without having to consider all possible mode sequences.

This idea can be extended to smoothing by using a similar approach to combine the likelihoods of the different modes at each time step, but it requires additional assumptions about the structure of the model and the transition probabilities between modes. For example, one could assume that the transition probabilities between modes are Markovian, which allows for efficient computation of the likelihoods and makes smoothing more tractable.

1.  **Approximate inference methods**: These methods use approximations to the true posterior distribution to reduce the computational complexity of smoothing for switching models. Examples include variational inference, which approximates the posterior distribution with a simpler distribution, and particle filtering, which uses a set of particles to represent the posterior distribution and updates them over time.

2.  **Model simplification**: This approach involves simplifying the model by reducing the number of modes or by imposing constraints on the model parameters. For example, one could assume that the transition probabilities between modes are sparse, which can reduce the number of possible mode sequences and make smoothing more tractable.

## Conclusion

Smoothing for switching models is a challenging problem due to the combinatorial complexity of considering all possible mode sequences. However, by using approximate inference methods or simplifying the model, it is possible to make smoothing for switching models more tractable and applicable to a wider range of problems.

Frühwirth-Schnatter, Sylvia. 2006. *Finite Mixture and Markov Switching Models*. Springer Science & Business Media.

Lange, Rutger-Jan. 2024. “Bellman Filtering and Smoothing for State–Space Models.” *Journal of Econometrics* 238 (2): 105632. <https://doi.org/10.1016/j.jeconom.2023.105632>.

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2026,
  author = {Bochman, Oren},
  title = {Smoothing Problems},
  date = {2026-03-07},
  url = {https://orenbochman.github.io/posts/2026/03-08-smoothing-problems/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2026. “Smoothing Problems.” March 7. <https://orenbochman.github.io/posts/2026/03-08-smoothing-problems/>.

[^1]: i.e. only a few states transition are possible from each states

[^2]: other state transitions probabilities are domminated by self transition probabilities
