This paper was mentioned in Geoffrey Hinton’s [Coursera course](https://www.coursera.org/learn/neural-networks-deep-learning/) as a way to simplify neural networks.

The main takeaway is that of modeling the loss using a mixture of Gaussians to cluster the weights and penalize the complexity of the model.

> **WARNING:**
>
> add categories
>
> add a citation for the course.
>
> It would be worth-while to look through what he says about this paper if only to ensure the main result is made to stand out from some of the others.
>
> Do a from scratch implementation of the paper
>
> Do a Python implementation of the paper.
>
> Why isn’t this technique easier to implement in practice?

## TL;DR

The primary aim of the paper ([Nowlan and Hinton 1992](#ref-nowlan2018simplifying)) is reducing the complexity of neural networks by employing a mixture of Gaussian priors to the weights, creating a “soft” weight-sharing mechanism. Instead of simply penalizing large weights (as in L2 regularization), this method clusters the weights, allowing some to stay close to zero and others to remain non-zero, depending on their usefulness. Soft weight sharing along with weight decay, improving generalization and making the model more interpretable.

## Abstract

> One way of simplifying neural networks so they generalize better is to add an extra term to the error function that will penalize complexity. Simple versions of this approach include penalizing the sum of the squares of the weights or penalizing the number of nonzero weights. We propose a more complicated penalty term in which the distribution of weight values is modeled as a mixture of multiple Gaussians. A set of weights is *simple* if the weights have high probability density under the mixture model. This can be achieved by clustering the weights into subsets with the weights in each cluster having very similar values. Since we do not know the appropriate means or variances of the clusters in advance, we allow the parameters of the mixture model to adapt at the same time as the network learns. Simulations on two different problems demonstrate that this complexity term is more effective than previous complexity terms
>
> – ([Nowlan and Hinton 1992](#ref-nowlan2018simplifying))

> **WARNING:**
>
> This notion of clustering weights is odd to say the least as these are just numbers in a data structure. Viewed as a method to reduce the effective number of parameters in the model, it makes some convoluted sense. What this idea seems to boil down to is that we are prioritizing neural net architectures with some abstract symmetry in the weights and thus a lower capacity and thus less prone to overfitting.
>
> - We shall shall soon see that the authors have attempted to motivate this idea in at least two ways:
>   1.  **Weight decay** - the penalty is a function of the weights themselves based on ([Plaut et al. 1986](#ref-plaut1986experiments))
>   2.  **A Bayesian perspective** - is a negative log density of the weights under a Gaussian prior.
> - it might also help if we learned that mixture models are often used to do clustering in unsupervised learning.
>
> A few quandaries then arise:
>
> 1.  How can we figure for different layers having weights, gradients and learning rates being more correlated then between layers.
> 2.  That there may be other structure so that the weights are not independent of each other.
>     1.  In classifiers the are continuous approximation of logic gates.
>     2.  In regression settings their values approximate continuous variables ?
> 3.  In many networks most of the weights are in the last layer, so we can use a different penalty for the last layer.
> 4.  Is there a way to impose an abstract symmetry on the weights of a neural network such that is commensurate with the problem?
> 5.  Can we impose multiple such symmetries on the network to give it other advantages?
>     - Invariance to certain transformations,
>     - using it for initialization,
>     - making the model more interpretable,
>     - Once we learn these mixture distribution of weights, can we use its parameters in, batch normalization, layer norm and with other regularization techniques like dropout?

## The problem:

This main problem in this paper is that of supervised ML

> How to train a model so it will generalize well on unseen data?

In deep learning this problem is exacerbated by the fact that neural networks require fitting lots of parameters while the data for training is limited. This naturally leads to overfitting - memorizing the data and noise rather than learning the underlying data generating process.

## Related work

How have others tried to solve this problem?

- **Weight Sharing**: Reducing effective numbers of *free* parameters in a neural network by using a prior distribution over the weights. This is equivalent to adding a penalty term to the loss function. c.f. ([Lang et al. 1990](#ref-lang1990time))
- **Early Stopping**: Does not restricts the parameters instead detecting the point when training and test scores diverge and stopping the training early. In reality weights are saved after each epoch and once we are certain that a divergence in say accuracy we restore the earlier weights c.f. ([Morgan and Bourlard 1989](#ref-morgan1989generalization)), ([Weigend et al. 1990](#ref-weigend1990predicting)).
- **Pruning**: Removing weight from the network ([Mozer and Smolensky 1989](#ref-mozer1989using)),
  1.  keep track of the importance of each unit and drop the least important ones - could work well in RL where we keep track of the importance of each state/action/features we might also care more about prioritizing certain states and discarding others. c.f. ([Mozer and Smolensky 1989](#ref-mozer1989using))
  2.  Use second order gradient information to estimate network sensitivity to weight changes and prune based on that. c.f. ([LeCun et al. 1990](#ref-lecun-90b))
- **Penalty Term**: Adding a term in the loss penelizing for the network’s complexity. c.f. ([Mozer and Smolensky 1989](#ref-mozer1989using)) ([LeCun et al. 1990](#ref-lecun-90b))[^1].
  1.  Complexity can be approximated using sum of the squares of the weights.
  2.  Differentiating the sum of the squares of the weights leads to weight decay. ([Plaut et al. 1986](#ref-plaut1986experiments))

Just a few ideas from this paper: ([Nowlan and Hinton 1992](#ref-nowlan2018simplifying))

\text{cost} = \text{data-misfit} + \lambda \times \text{complexity} \qquad

penalties:

\sum\_{i} w_i^2 \qquad \text{(L2 penalty)}

the authors provide two ways to think about this penalty:

1.  **Weight decay** - the penalty is a function of the weights themselves based on ([Plaut et al. 1986](#ref-plaut1986experiments))
2.  **a Bayesian perspective** - is a negative log density of the weights under a Gaussian prior.

The authors point out that the problem with L2 penalty term is that it prefers two weak interactions over a strong one.

\left(\frac{w}{2}\right)^2+\left(\frac{w}{2}\right)^2 \< w^2 + 0 \qquad

we want to keep larger weights and drop the smaller ones

p(w) = \pi_n \frac{1}{2 \pi\sigma_n}e^{-w^2/2\sigma^2_n} + \pi_b \frac{1}{2 \pi\sigma_b}e^{-w^2/2\sigma^2_b} \qquad

This is equivalent to the L2 penalty. ([MacKay 1991](#ref-mackay1991bayesian)) showed that we can do better with:

p(w) =\sum_i \lambda_i \sum\_{j} w_j^2 \qquad \text{(L2 penalty)}

where

- i is the layer and
- j is the weight in that layer

Different layers has a different penalty, which is equivalent to a Gaussian prior with different variances for each layer.[^2]

\sum\_{i} \delta(w{\_i\ne 0}) \qquad \text{(L0 penalty)}

p(w) = \sum\_{i} \delta(w_i \ne 0) + \lambda \sum\_{i} w_i^2

## The paper

paper

## Resources

- Article [using weight constraints to reduce generalization](https://machinelearningmastery.com/introduction-to-weight-constraints-to-reduce-generalization-error-in-deep-learning/)
- The paper is available at https://www.cs.utoronto.ca/~hinton/absps/sunspots.pdf

## An after thought

Can we use a Bayesian RL to tune the hyper-parameters of model and dataset. We can perhaps create an RL alg that controls the many aspects of training of a model. It can explore/exploit different setups on subsets of the data. Find variants that converge faster and are more robust by adding constraints at different levels. It can identify problems in the datasets (possible bad labels etc) . Ensambles, mixtures of experts, different regularization strategies. Different Learning rates and schedules globaly or per layer.

Lang, Kevin J, Alex H Waibel, and Geoffrey E Hinton. 1990. “A Time-Delay Neural Network Architecture for Isolated Word Recognition.” *Neural Networks* 3 (1): 23–43.

LeCun, Yann, J. S. Denker, S. Solla, R. E. Howard, and L. D. Jackel. 1990. “Optimal Brain Damage.” In *Advances in Neural Information Processing Systems (NIPS 1989)*, edited by David Touretzky, vol. 2. Morgan Kaufman. <https://proceedings.neurips.cc/paper_files/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf>.

MacKay, David JC. 1991. “Bayesian Modeling and Neural Networks.” *PhD Thesis, Dept. Of Computation and Neural Systems, CalTech*. <https://www.inference.org.uk/mackay/thesis.pdf>.

Morgan, Nelson, and Hervé Bourlard. 1989. “Generalization and Parameter Estimation in Feedforward Nets: Some Experiments.” *Advances in Neural Information Processing Systems* 2.

Mozer, Michael C, and Paul Smolensky. 1989. “Using Relevance to Reduce Network Size Automatically.” *Connection Science* 1 (1): 3–16.

Nowlan, Steven J., and Geoffrey E. Hinton. 1992. “Simplifying Neural Networks by Soft Weight-Sharing.” *Neural Computation* 4 (4): 473–93. <https://doi.org/10.1162/neco.1992.4.4.473>.

Plaut, D. C., S. J. Nowlan, and G. E. Hinton. 1986. *Experiments on Learning by Back-Propagation*. CMU-CS-86-126. Carnegie–Mellon University. <https://ni.cmu.edu/~plaut/papers/pdf/PlautNowlanHinton86TR.backprop.pdf>.

Weigend, Andreas S, Bernardo A Huberman, and David E Rumelhart. 1990. “Predicting the Future: A Connectionist Approach.” *International Journal of Neural Systems* 1 (03): 193–209.

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2022,
  author = {Bochman, Oren},
  title = {Simplifying {Neural} {Networks} by {Soft} {Weight} {Sharing}},
  date = {2022-06-22},
  url = {https://orenbochman.github.io/reviews/1991/simplifing-NN-by-soft-weight-sharing/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2022. “Simplifying Neural Networks by Soft Weight Sharing.” June 22. <https://orenbochman.github.io/reviews/1991/simplifing-NN-by-soft-weight-sharing/>.

[^1]: in the paper this citation is ambiguous, but I think this is the correct one - based on the abstract

[^2]: note: this answers the first of my question above.
