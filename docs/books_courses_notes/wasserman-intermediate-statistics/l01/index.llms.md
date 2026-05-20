# CMU - 36-705 - Intermediate Statistics - Lesson 01

Larry Wasserman’s course

review

Material from Larry Wasserman’s course on Intermediate Statistics at CMU. This course covers a wide range of topics in statistical inference, including estimation, hypothesis testing, confidence intervals, and asymptotic theory. The lectures are available online and the course uses several textbooks as references.

Published

Tuesday, June 10, 2025

Keywords

course review

# An error occurred.

Unable to execute JavaScript.

- The lecture finishes a probability review and then begins probability inequalities.

- Change of variables is reviewed for functions of multiple random variables:

  - For a transformed variable (Z=g(X,Y)), start from the cumulative distribution function: F_Z(z)=P(g(X,Y)\le z)
  - Then integrate the joint density over the region satisfying the constraint.
  - Finally, differentiate the cumulative distribution function to obtain the density.

- Independence is defined formally:

  - 24. and (Y) are independent if P(X\in A, Y\in B)=P(X\in A)P(Y\in B) for all events (A,B).
  - Equivalently, the joint density factors: p(x,y)=p_X(x)p_Y(y)
  - Mutual independence generalizes this factorization to (n) random variables.

- Independent and identically distributed data are introduced as the default assumption for the class:

  - “Independent” means observations do not affect one another.
  - “Identically distributed” means each observation comes from the same marginal distribution.
  - The lecturer stresses that this is an assumption, not an automatic fact; time-series data are given as a common non-independent case.

- Several standard distributions are briefly reviewed:

  - [Normal distribution:](https://en.wikipedia.org/wiki/Normal_distribution) a family indexed by mean (\mu) and variance (\sigma^2).
  - [Multivariate normal distribution:](https://en.wikipedia.org/wiki/Multivariate_normal_distribution) a distribution over random vectors, with mean vector (\mu) and covariance matrix (\Sigma).
  - [Chi-square distribution:](https://en.wikipedia.org/wiki/Chi-squared_distribution) obtained as a sum of squared independent standard normal variables.
  - [Bernoulli distribution:](https://en.wikipedia.org/wiki/Bernoulli_distribution) a single coin-flip-like binary random variable.
  - [Binomial distribution:](https://en.wikipedia.org/wiki/Binomial_distribution) the sum of independent Bernoulli trials.
  - [Poisson distribution:](https://en.wikipedia.org/wiki/Poisson_distribution),
  - [Uniform distribution:](https://en.wikipedia.org/wiki/Uniform_distribution_(continuous)),
  - [Multinomial distribution:](https://en.wikipedia.org/wiki/Multinomial_distribution), and
  - [Exponential distribution:](https://en.wikipedia.org/wiki/Exponential_distribution) are mentioned as further reference material.

- The distinction between *random variables* and *parameters* is emphasized:

  - The random variable is the object being probabilistically described.
  - Parameters such as (\mu), (\sigma^2), (n), or (\theta) index a family of distributions and are treated as fixed, though often unknown.

- Sampling distributions are introduced:

  - A statistic is any function of the data, such as the sample mean, sample variance, median, maximum, or minimum.
  - Since the data are random, statistics computed from them are also random variables.
  - The distribution of such a statistic is its sampling distribution.

- For the sample mean \bar X:

  - If X_1,\dots,X_n are independent and identically distributed with mean \mu and variance \sigma^2, then: E\[\bar X\]=\mu,\qquad Var(\bar X)=\frac{\sigma^2}{n}
  - Thus the sample mean has the same center as the population but becomes more concentrated as (n) increases.

- The sample variance is discussed:

  - The usual denominator (n-1) gives an unbiased estimator of the population variance.
  - The lecturer notes that using (n) versus (n-1) should not matter much when (n) is large.

- In the special normal case:

  - If the original observations are normally distributed, then (X) is also normally distributed.
  - The scaled sample variance has a chi-square distribution.
  - The sample mean and sample variance are independent under normality.

- The lecture then shifts to probability inequalities:

  - These inequalities bound probabilities such as P(X\>t).
  - They are important in statistics and machine learning, especially for bounding error probabilities and proving convergence results.
  - They serve as a foundation for later topics such as [Vapnik–Chervonenkis theory](https://en.wikipedia.org/wiki/Vapnik%E2%80%93Chervonenkis_dimension) and [convergence theory](https://en.wikipedia.org/wiki/Convergence_of_random_variables).

- Gaussian tail bounds are introduced:

  - For a standard normal random variable, the probability of being far from zero decreases very quickly.
  - The key intuition is that Gaussian tails decay exponentially fast.
  - The lecturer proves the one-sided bound using elementary properties of the normal density, then extends it by symmetry.

- Markov’s inequality is introduced:

  - For a nonnegative random variable X with finite mean, P(X\ge t)\le \frac{E\[X\]}{t}
  - It is described as weak because it decays only like (1/t).
  - Its importance is that it is a basic tool for proving sharper inequalities.

- [Chebyshev’s inequality](https://en.wikipedia.org/wiki/Chebyshev%27s_inequality) is derived from Markov’s inequality:

  - Apply Markov’s inequality to ((X-E\[X\])^2).
  - This yields a bound involving the variance: P(\|X-E\[X\]\|\ge t)\le \frac{Var(X)}{t^2}
  - The improvement comes from using extra information: the existence of variance.

- [Hoeffding’s inequality](https://en.wikipedia.org/wiki/Hoeffding%27s_inequality) is stated as a major goal:

  - It applies to independent bounded random variables.
  - The boundedness assumption implies very thin tails.
  - The result gives an exponential bound for deviations of the sample mean from its expectation.
  - The lecturer stresses its importance in statistics and machine learning.

- The final part introduces the proof strategy for Hoeffding’s inequality:

  - Use moment generating functions because bounded random variables have all moments.
  - Use a universal bound on the moment generating function based on boundedness.
  - Apply the Chernoff method: exponentiate the event, apply Markov’s inequality, and optimize over an auxiliary positive parameter.

- The lecture ends by saying that the next class will continue the Hoeffding proof, using the Chernoff trick together with bounds on moment generating functions.

Lecture Notes 1
