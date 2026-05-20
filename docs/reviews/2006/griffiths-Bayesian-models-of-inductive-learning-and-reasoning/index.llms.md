![](../../../images/lit-review-cover.jpg)

cover

> “The ideal market completely disregards those spikes—but a realistic model cannot.” Mandelbrot highlights the inadequacy of models ignoring extreme price movements, emphasizing the need for a framework that can accommodate them.

> **NOTE:**
>
> [![Inductive Learning in a nutshell](../../../images/in_the_nut_shell_coach_retouched.jpg)](../../../images/in_the_nut_shell_coach_retouched.jpg "Inductive Learning in a nutshell")
>
> Inductive Learning in a nutshell
>
> Inductive learning can be used a framework for learning to reason about the world

### Abstract

> Inductive inference allows humans to make powerful generalizations from sparse data when learning about word meanings, unobserved properties, causal relationships, and many other aspects of the world. Traditional accounts of induction emphasize either the power of statistical learning, or the importance of strong constraints from structured domain knowledge, intuitive theories or schemas. We argue that both components are necessary to explain the nature, use and acquisition of human knowledge, and we introduce a theory-based Bayesian framework for modeling inductive learning and reasoning as statistical inferences over structured knowledge representations.
>
> — ([Tenenbaum et al. 2006](#ref-tenenbaum2006theory))

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

## Outline

- Introduction
  - Describes the importance of inductive inference in human cognition, highlighting the challenge of generalizing from sparse data.
  - Mentions that traditional accounts of induction emphasize either statistical learning or domain-specific knowledge.
  - Argues for a theory-based Bayesian framework for modeling inductive learning and reasoning as statistical inferences over structured knowledge representations.
- Theory-based Bayesian Models
  - Introduces the core idea of theory-based Bayesian models, where prior knowledge shapes the hypothesis space and probability distributions.
  - Notes the distinction between domain-general statistical mechanisms and domain-specific knowledge representations.
  - Presents Bayes’ rule as a framework for combining prior knowledge (P(h\|T)) and likelihood (P(x\|h,T)) to calculate posterior probabilities (P(h\|x,T)).
  - Emphasizes the role of domain theory in generating hypothesis spaces, prior probabilities, and likelihoods, forming a probabilistic version of intuitive theories.
- Learning Names for Things
  - Describes the classic category learning task as an abstraction of word learning for object kinds.
  - Presents limitations of traditional statistical models for word learning that assume simple notions of categories and label-category mappings.
  - Introduces a Bayesian model for word learning that utilizes a tree-structured taxonomy of objects as the hypothesis space of word meanings.
  - Discusses how the model explains children’s generalization patterns, showing that generalization follows a gradient according to taxonomic distance, which sharpens with more examples.
  - Mentions the “size principle” as a general principle of Bayesian learning, leading to the preference for smaller, more specific hypotheses with increasing evidence.
- Reasoning about Hidden Properties
  - Introduces the task of property induction, where learners must generalize a novel property observed in one or more categories to other categories.
  - Presents a theory-based Bayesian model for property induction in the domain of biological species, assuming a tree-structured taxonomy and a mutation process for property generation.
  - Highlights the model’s ability to account for generalizations of “blank” biological properties, surpassing models based on generic knowledge representations.
  - Discusses the need for different priors to account for generalizations of different kinds of predicates, such as anatomical, behavioral, and disease properties.
  - Presents examples of Bayesian models using different structured graphs and stochastic processes to capture the specific inductive biases for different property types.
- Learning Theories to Support Property Induction
  - Addresses the problem of learning the taxonomic tree structure from raw data (species-property pairs) using Bayesian inference.
  - Explains how the best tree structure maximizes the likelihood of the observed properties, reflecting the smooth variation of features over the tree.
  - Discusses the role of a “taxonomic principle” as an abstract domain principle that guides the learning process.
  - Notes that the Bayesian framework allows for learning abstract domain principles themselves, choosing the best structural form (e.g., tree, linear order, clusters) based on the trade-off between complexity and fit to the data.
  - Emphasizes the ability to incorporate explicit instructions at any level of abstraction within the hierarchical Bayesian framework, which can lead to dramatic changes in inferences.
- Causal Learning and Reasoning
  - Highlights the central role of causal cognition in intuitive theories, suggesting that causality is often seen as defining a theory.
  - Presents the idea of causal graphical models as a particular kind of structured probabilistic model used to make inferences about observable events.
  - Discusses the learning of causal models, contrasting bottom-up statistical cues with top-down approaches relying on abstract domain knowledge about causal mechanisms.
  - Introduces the concept of “causal grammars” as probabilistic models that generate causal graphical models based on an ontology and a set of causal laws.
  - Provides examples of theory-based Bayesian causal induction, showing how abstract knowledge about risk factors, diseases, and symptoms can guide the learning of causal networks in a medical domain, and how domain principles such as the “activation law” can explain causal inferences in the “blicket detector” paradigm.
- Learning Abstract Causal Theories
  - Acknowledges the open question of how abstract causal principles or “framework theories” are learned.
  - Mentions the infinite relational model (IRM) as an example of a Bayesian model that can infer the number of classes, class membership, and relationships between classes from data.
  - Notes that while IRM can learn some simple framework knowledge, more powerful methods for learning in probabilistic logical systems are needed to account for richer causal theories.
- Conclusion
  - Reiterates the potential of the theory-based Bayesian framework for understanding human cognition, emphasizing the integration of both sophisticated inference processes and knowledge representations.
  - Acknowledges the limitations of current models and the need to address algorithmic and psychological plausibility issues.
  - Concludes that probabilistic inference over hierarchies of increasingly abstract and flexibly structured representations is a crucial idea for explaining inductive learning and reasoning.

## Reflection on the paper and beyond.

This seems to be one of the way to capture the issues of aggregation rules for signals in the transitions from simple to complex signaling systems.

In reality these are two different things

1.  learning a model from some state space.
2.  aggregating symbols using a rule like a grammar.

The two problems are related but only tangentially. This is one reason that transitioning from the simple Lewis signaling model to the more complex ones is so difficult.

## The paper

paper

Tenenbaum, Joshua B, Thomas L Griffiths, and Charles Kemp. 2006. “Theory-Based Bayesian Models of Inductive Learning and Reasoning.” *Trends in Cognitive Sciences* 10 (7): 309–18.

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2022,
  author = {Bochman, Oren},
  title = {🧠 {Theory-Based} {Bayesian} {Models} of {Inductive}
    {Learning} and {Reasoning}},
  date = {2022-03-10},
  url = {https://orenbochman.github.io/reviews/2006/griffiths-Bayesian-models-of-inductive-learning-and-reasoning/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2022. “🧠 Theory-Based Bayesian Models of Inductive Learning and Reasoning.” March 10. <https://orenbochman.github.io/reviews/2006/griffiths-Bayesian-models-of-inductive-learning-and-reasoning/>.
