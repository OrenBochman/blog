Convolutions arise in many areas of mathematics, physics, and engineering, particularly in signal processing and image analysis. At its core, a convolution is a mathematical operation that combines two functions to produce a third function that expresses how the shape of one is modified by the other.

## Definition and initial intuition

Here is the mathematical definition of the convolution of two functions f and g:

(f \* g)(t) = \int\_{-\infty}^{\infty} f(\tau) g(t - \tau) d\tau

It’s a shame that when we meet this formula as students we usually don’t have a good ~~intuitive~~ understanding of what this is suppose to mean.

This integral essentially slides one function over another, multiplying and integrating at each point to produce a new function. The result of the convolution operation can be interpreted in various ways depending on the context, such as blurring an image, filtering a signal, or combining probability distributions.

## In probability and statistics

Convolutions can be used to uniquely define a distribution in terms of its characteristic function characteristic function or its moment generating function moment generating function. While use of moment generating functions is more common in introductory probability courses, characteristic functions have some advantages, particularly in theoretical work and are more both more fundamental and guaranteed to exist for all probability distributions, something unfortunately not true of moment generating functions as some distributions do not have the requisite moments.

One area It is particularly useful is probability theory and statistics, where convolutions are used to determine the distribution of the sum of two independent random variables. If X and Y are independent random variables with probability density functions (PDFs) f_X(x) and f_Y(y), respectively, then the PDF of the sum Z = X + Y is given by the convolution of f_X and f_Y :

## Next lets explore convolutions further using some videos

# An error occurred.

Unable to execute JavaScript.

This is a very powerful

0:00 - Where do convolutions show up? 2:07 - Add two random variables 6:28 - A simple example 7:25 - Moving averages 8:32 - Image processing 13:42 - Measuring runtime 14:40 - Polynomial multiplication 18:10 - Speeding up with FFTs 21:22 - Concluding thoughts

# An error occurred.

Unable to execute JavaScript.

# An error occurred.

Unable to execute JavaScript.

# An error occurred.

Unable to execute JavaScript.

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2026,
  author = {Bochman, Oren},
  title = {Convoluted {Intuitions}},
  date = {2026-01-03},
  url = {https://orenbochman.github.io/posts/2026/01-03-convolutions-and-fft/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2026. “Convoluted Intuitions.” January 3. <https://orenbochman.github.io/posts/2026/01-03-convolutions-and-fft/>.
