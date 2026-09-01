![](../../../images/lit-review-cover.jpg)

cover

> “Probabilistic models can capture the structure of extremely complex problems, but as the structure of the model becomes richer, probabilistic inference becomes harder..” Mandelbrot highlights the inadequacy of models ignoring extreme price movements, emphasizing the need for a framework that can accommodate them. — ([Griffiths and Yuille 2008](#ref-griffiths2008primer)) introduction.

I think that this work may be used as a starting point for a framework for building a cognitive substrate for RL agents that can learn multiple tasks in parallel.

My thinking is along the lines that the RL agents can learn at a number of levels.

Policies, Value functions, Options, Reward models,

To generalize though beyond tabular methods presents a number of challenges that might be addressed by the probabilistic models. Rules, and Grammars may be modeled as trees, etc and could greatly benefit from both priors and the ability to assemble into hierarchies.

> **NOTE:**
>
> [![Probabilistic inference in a nutshell](../../../images/in_the_nut_shell_gemeni.png)](../../../images/in_the_nut_shell_gemeni.png "Probabilistic inference in a nutshell")
>
> Probabilistic inference in a nutshell
>
> This tutorial covers the basics of probabilistic inference and how it can be used to model human cognition.

### Abstract

> Research in computer science, engineering, mathematics and statistics has produced a variety of tools that are useful in developing probabilistic models of human cognition. We provide an introduction to the principles of probabilistic inference that are used in the papers appearing in this special issue. We lay out the basic principles that underlie probabilistic models in detail, and then briefly survey some of the tools that can be used in applying these models to human cognition
>
> — ([Griffiths and Yuille 2008](#ref-griffiths2008primer))

## Glossary

This paper uses lots of big terms so let’s break them down so we can understand them better

Inductive Inference  
The process of reasoning from specific observations or examples to more general conclusions or theories.

Bayesian Inference  
A method of statistical inference in which Bayes’ theorem is used to update the probability for a hypothesis as more evidence or information becomes available.

Prior Probability (Prior)  
The initial probability assigned to a hypothesis before any data is observed, reflecting prior knowledge or beliefs.

Likelihood  
The probability of observing the given data under a specific hypothesis and background theory.

Posterior Probability (Posterior)  
The updated probability of a hypothesis after considering the observed data, calculated using Bayes’ theorem (proportional to the product of the prior and the likelihood).

Intuitive Theory  
A system of related concepts, causal laws, structural constraints, or explanatory principles that guide inductive inference in a particular domain of knowledge.

Structured Probabilistic Model  
A formal representation of an intuitive theory that defines a probability distribution over possible observables, often based on a graph structure capturing relationships.

Taxonomy  
A hierarchical system of classification, often represented as a tree structure, used to organize categories and concepts.

Size Principle  
The tendency in Bayesian learning for smaller, more specific hypotheses to be increasingly preferred over larger, more general hypotheses as more randomly sampled examples are observed.

Causal Graphical Model  
A probabilistic model that represents causal relationships between variables using a graph where nodes represent variables and edges represent direct causal influences.

Causal Grammar  
A formal system, inspired by linguistic grammars, that generates causal graphical models based on an ontology and a set of causal laws.

Hierarchical Bayesian Model  
A Bayesian model with multiple levels of parameters, where the priors for lower-level parameters are themselves governed by higher-level parameters, allowing for learning at different levels of abstraction.

Probabilistic Model  
A framework that uses probability theory and statistics to represent and reason about uncertainty in a system or process. In cognitive science, these models aim to explain human thought and behavior in terms of probabilities.

Bayesian Inference  
A method of statistical inference in which Bayes’ rule is used to update the probability for a hypothesis as more evidence or information becomes available. It involves combining a prior belief with the likelihood of the data to obtain a posterior belief.

Prior Probability (Prior)  
The initial probability assigned to a hypothesis before any data is observed. It represents pre-existing beliefs or knowledge.

Likelihood  
The probability of observing the data given a particular hypothesis. It quantifies how well a hypothesis explains the observed evidence.

Posterior Probability (Posterior)  
The updated probability of a hypothesis after considering the observed data, calculated using Bayes’ rule. It represents the revised belief in light of the evidence.

Marginalization  
A procedure in probability theory used to find the probability distribution of a subset of variables by summing (or integrating) over the remaining variables.

Hypothesis Space  
The set of all possible hypotheses that an agent considers as potential explanations for observed data.

Likelihood Ratio  
The ratio of the likelihood of the data under one hypothesis to the likelihood of the data under another hypothesis. It indicates the relative support provided by the data for the two hypotheses.

Prior Odds  
The ratio of the prior probabilities of two hypotheses. It represents the initial relative belief in the two hypotheses.

Bayesian Occam’s Razor  
The principle in Bayesian model selection that favors simpler models over more complex ones unless the added complexity is strongly supported by the data, effectively penalizing unwarranted complexity.

Graphical Model  
A probabilistic model that uses a graph to represent the dependencies between random variables. Nodes in the graph represent variables, and edges represent statistical relationships.

Directed Graphical Model (Bayes Net)  
A graphical model in which the edges between nodes are directed, representing conditional dependencies. The absence of a direct edge implies conditional independence.

Markov Condition  
A property of directed graphical models stating that each variable is conditionally independent of its non-descendants given its parents in the graph.

Undirected Graphical Model (Markov Random Field)  
A graphical model in which the edges between nodes are undirected, representing a symmetric relationship between connected variables.

Causal Graphical Model  
A type of graphical model that explicitly aims to represent causal relationships between variables, allowing for reasoning about the effects of interventions.

Generative Model  
A probabilistic model that describes a process for how the observed data could have been generated from underlying variables and parameters.

Latent Variable  
An unobserved or hidden variable that is assumed to influence the observed data in a probabilistic model.

Mixture Model  
A probabilistic model that assumes the observed data comes from a mixture of several different probability distributions.

Expectation-Maximization (EM) Algorithm  
An iterative algorithm used to find maximum likelihood or maximum a posteriori (MAP) estimates of parameters in probabilistic models with latent variables.

Markov Chain Monte Carlo (MCMC)  
A class of algorithms for obtaining samples from a probability distribution by constructing a Markov chain that has the desired distribution as its stationary distribution.

Stationary Distribution  
A probability distribution for a Markov chain that remains unchanged over time once the chain reaches it.

Conjugate Prior  
A prior probability distribution that, when combined with the likelihood function, results in a posterior distribution that is in the same family as the prior. This simplifies Bayesian calculations.

## Outline

- Introduces undirected graphical models, also known as Markov Random Fields (MRFs).
  - Explains the use of potential functions to represent dependencies between variables.
  - Notes their applications in fields like computer vision and artificial neural networks.
- Uses of Graphical Models
  - Discusses different perspectives on graphical models in artificial intelligence and statistics.
  - Highlights the use of Bayes nets for knowledge representation, probabilistic reasoning, and causal modeling (Box 2).
  - Discusses the application of graphical models for understanding generative processes and incorporating latent variables.
- Algorithms for Inference
  - Presents the challenges of inferring latent variables and learning probability distributions in models with latent variables.
  - Introduces the Expectation-Maximization (EM) algorithm for maximum likelihood or MAP estimation.
  - Explains the iterative process of EM, involving expectation (E-step) and maximization (M-step) computations.
- Conclusion
  - Emphasizes the potential of probabilistic models for developing a rational account of human cognition.
  - Underscores the importance of tools like graphical models, EM, and MCMC for addressing the challenges of probabilistic modeling.
  - Highlights the need for continued interdisciplinary collaboration to develop models that capture the complexities of human cognition.

## Reflections

## The paper

paper

Griffiths, Thomas, and Alan Yuille. 2008. “A Primer on Probabilistic Inference.” *The Probabilistic Mind: Prospects for Bayesian Cognitive Science*, 33–57.

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2015,
  author = {Bochman, Oren},
  title = {🧠 {Technical} {Introduction:} {A} Primer on Probabilistic
    Inference},
  date = {2015-03-20},
  url = {https://orenbochman.github.io/reviews/2006/griffiths-primer-on-probabilistic-inference/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2015. “🧠 Technical Introduction: A Primer on Probabilistic Inference .” March 20. <https://orenbochman.github.io/reviews/2006/griffiths-primer-on-probabilistic-inference/>.
