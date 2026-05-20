# Statistical Methods for Machine Learning

Larry Wasserman’s course

review

A review of Larry Wasserman’s course on Statistical Methods for Machine Learning, covering key concepts and lectures on various topics in machine learning and statistics.

Published

Tuesday, June 10, 2025

Keywords

course review

## Review

lesson notes

# An error occurred.

Unable to execute JavaScript.

1.  Statistics versus ML
2.  Concentration
3.  Probability
4.  Basic Statistics
5.  Minimaxity
6.  Regression
7.  Classification

**Theorem 1 (Hoeffding’s inequality)** If Z_1, Z_2, \ldots, Z_n are iid with mean \mu and P(a \le Z_i \le b) = 1, then for any \epsilon \> 0, P(\|Z_n − \mu\| \> \epsilon) \le 2e^{-2n\epsilon^2/(b-a)^2}, where Z_n = \frac{1}{n}\sum\_{i=1}^n Z_i.

**Definition 1 (VC Dimension)** Let A be a class of sets. If F is a finite set, let s(A, F) be the number of subset of F ‘picked out’ by A.

Define the growth function: s_n(A) = \sup\_{\|F\|=n} s(A, F).

Note that s_n(A) \le 2^n.

The VC dimension of a class of set A is VC(A) = \sup\\n : s_n(A) = 2^n\\. \qquad \tag{1}

If the VC dimension is finite, then there is a phase transition in the growth function from exponential to polynomial:

**Theorem 2 (Sauer’s Theorem)** Suppose that A has finite VC dimension d. Then, \forall n \ge d, s(A, n) \le( \frac{en}{d})^d. \qquad

Given data Z_1, \ldots, Z_n \sim P. The empirical measure Pn is Pn(A) = \frac{1}{n}\sum\_{i=1}^n I(Z_i \in A).

**Theorem 3 (Vapnik and Chervonenkis)** Let A be a class of sets. For any t \> \sqrt{2/n}, P(\sup\_{A \in A} \|P_n(A) − P(A)\| \> t) \le 4 s(A, 2n) e^{-nt^2/8} (4) and hence, with probability at least 1 − \delta, \sup\_{A \in A} \|P_n(A) − P(A)\| \le \sqrt{8/n \log( 4 s(A, 2n)/\delta)} (5). Hence, if A has finite VC dimension d then \sup\_{A \in A} \|P_n(A) − P(A)\| \le \sqrt{8/n (\log( 4/\delta)+ d \log(n e/d))} \qquad

**Theorem 4** Bernstein’s inequality is a more refined inequality than Hoeffding’s inequality. It is especially useful when the variance of Y is small.

Suppose that Y_1, \ldots, Y_n are iid with mean \mu, \text{Var}(Y_i) \le \sigma^2 and \|Y_i\| \le M. Then P(\|Y - \mu\| \> \epsilon) \le 2 \exp\\- n \epsilon^2 / (2\sigma^2 + 2M \epsilon / 3)\\ \qquad

It follows that P(\|Y - \mu\| \> t \sqrt{n} \epsilon + \epsilon \sigma^2 (1 - c)) \le e^{-t} for small enough \epsilon and c.

## Density Estimation

![](slide01.png)

Slide caption

# An error occurred.

Unable to execute JavaScript.

Figure 1: Lecture 02: Function Spaces

## Nonparametric Regression

https://www.stat.cmu.edu/~larry/=sml/nonpar2019.pdf

# An error occurred.

Unable to execute JavaScript.

Lecture 03: Concentration of Measure

# An error occurred.

Unable to execute JavaScript.

Lecture 04: Concentration of Measure

## Concentration of Measure

# An error occurred.

Unable to execute JavaScript.

Lecture 05: Concentration of Measure

## Linear Regression

Linear Regression

## Nonparametric Regression

Nonparametric Regression

## Nonparametric Regression - Ryan Tibshirani and Larry Wasserman

Nonparametric Regression

# An error occurred.

Unable to execute JavaScript.

Lecture 06: Linear Regression and Nonparametric Regression

## A Closer Look at Sparse Regression Ryan Tibshirani

sparsity

# An error occurred.

Unable to execute JavaScript.

Lecture 07: Nonparametric Regression

## A Closer Look at Sparse Regression Ryan Tibshirani

sparsity

# An error occurred.

Unable to execute JavaScript.

Lecture 08: Trend filtering

## A Closer Look at Sparse Regression Ryan Tibshirani

# An error occurred.

Unable to execute JavaScript.

Lecture 09: Linear Classification

Random Forests

# An error occurred.

Unable to execute JavaScript.

Lecture 10: Nonparametric Classification

Minimax Theory

# An error occurred.

Unable to execute JavaScript.

Lecture 11: Minimax Theory

Minimax Theory

# An error occurred.

Unable to execute JavaScript.

Lecture 12: Minimax Theory

![](https://www.stat.cmu.edu/~larry/=sml/nonparbayes.png)

Nonparametric Bayesian Inference

# An error occurred.

Unable to execute JavaScript.

Lecture 13: Nonparametric Bayes

# An error occurred.

Unable to execute JavaScript.

Lecture 14: Boosting

Clustering

# An error occurred.

Unable to execute JavaScript.

Lecture 15: Clustering

Clustering

# An error occurred.

Unable to execute JavaScript.

Lecture 16: Clustering

Clustering

# An error occurred.

Unable to execute JavaScript.

Lecture 17: Clustering

Graphical Models

# An error occurred.

Unable to execute JavaScript.

Lecture 18: Graphical Models

Directed Graphical Models

# An error occurred.

Unable to execute JavaScript.

Lecture 19: Graphical Models

Causal Inference

# An error occurred.

Unable to execute JavaScript.

Lecture 20: Graphical Models

![Conformal Prediction](https://www.stat.cmu.edu/~larry/=sml/Conformal.png)

# An error occurred.

Unable to execute JavaScript.

Lecture 21: Graphical Models

Dimension Reduction

# An error occurred.

Unable to execute JavaScript.

Lecture 22: Dimension Reduction

Dimension Reduction

# An error occurred.

Unable to execute JavaScript.

Lecture 23: Dimension Reduction & Random Matrix Theory

# An error occurred.

Unable to execute JavaScript.

Lecture 24: Random Matrix Theory & Differential Privacy
