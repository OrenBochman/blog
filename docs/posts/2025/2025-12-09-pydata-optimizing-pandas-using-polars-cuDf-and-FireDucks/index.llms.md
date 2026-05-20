[![pydata global](pydata_logo.png)](pydata_logo.png "pydata global")

pydata global

> **TIP:**
>
> In general, a Data Scientist spends significant efforts in transforming the raw data into a more digestible format before training an AI model or creating visualisations. Traditional tools such as pandas have long been the linchpin in this process, offering powerful capabilities but not without limitations. With numerous possible ways to write the same thing in pandas, often a user ends up selecting the uneconomical, inefficient ones, leading to large computational　costs　with the growth in data size. We introduce a couple of frequently occurring　intricate performance issues in pandas, and what we have learnt in solving the same using popular high-performance pandas alternatives: Polars, FireDucks and cuDF. The talk intends to highlight one of the best practices (breaking out of the loops) that one should follow while dealing with large-scale data analysis, while demonstrating the key advantages of the high-performance pandas alternatives based on different scenarios.

> **TIP:**
>
> - How the choice and execution order of API calls in writing an data-related application impacts its performance.
> - How to stop thinking the loop-based approach and design the algorithms using DataFrame APIs.
> - How the internal query optimizers in libraries like Polars, FireDucks etc, can be useful to bring SQL-like optimizations at python-level.
> - Whether to pay a large migration cost for optimizing an existing pandas-based application or to go smart with some minor modifications and save more operational cost.

> **TIP:**
>
> - Basic Python and PyTorch
> - Some familiarity with neural networks (e.g., feedforward, softmax)
> - No need for prior experience in building models from scratch

## Tools and Frameworks:

We will introduce you to certain modern frameworks in the workshop but the emphasis be on first principles and using vanilla Python and LLM calls to build AI-powered systems.

[workshop repo](https://github.com/hugobowne/AI-for-SWEs)

> **TIP:**

[repo](https://github.com/qsourav/pydata-global-2025)

## Outline

[![Title](slide01.png)](slide01.png "Title")

Title

[![Quick Introduction!](slide02.png)](slide02.png "Quick Introduction!")

Quick Introduction!

[![Who is this talk for?](slide03.png)](slide03.png "Who is this talk for?")

Who is this talk for?

[![Overview of the Application](slide04.png)](slide04.png "Overview of the Application")

Overview of the Application

[![Pandas](slide05.png)](slide05.png "Pandas")

Pandas

[![Exploring High-performance Pandas Alternatives](slide06.png)](slide06.png "Exploring High-performance Pandas Alternatives")

Exploring High-performance Pandas Alternatives

[![Comparison among Chosen Libraries](slide07.png)](slide07.png "Comparison among Chosen Libraries")

Comparison among Chosen Libraries

[![Bottleneck Analysis](slide08.png)](slide08.png "Bottleneck Analysis")

Bottleneck Analysis

[![Bottleneck Analysis](slide09.png)](slide09.png "Bottleneck Analysis")

Bottleneck Analysis

[![Exploring Type-1 Bottlenecks (Loop-based implementation)](slide10.png)](slide10.png "Exploring Type-1 Bottlenecks (Loop-based implementation)")

Exploring Type-1 Bottlenecks (Loop-based implementation)

[![Query 01 Problem Statement](slide11.png)](slide11.png "Query 01 Problem Statement")

Query 01 Problem Statement

- Fill missing values of “Description” column using the most frequent description of the specific “StockCode”.

[![Query 01: Implementation using iterrows](slide12.png)](slide12.png "Query 01: Implementation using iterrows")

Query 01: Implementation using iterrows

[![Query 01: Implementation using vectorized APIs](slide13.png)](slide13.png "Query 01: Implementation using vectorized APIs")

Query 01: Implementation using vectorized APIs

[![Query 02: Problem Statement](slide14.png)](slide14.png "Query 02: Problem Statement")

Query 02: Problem Statement

Find the number of transactions a user performed within the N days (e.g., 90) of the current transaction

[![Query 02: implementation using row-wise apply](slide15.png)](slide15.png "Query 02: implementation using row-wise apply")

Query 02: implementation using row-wise apply

[![Query 02: implementation using merge+filter](slide16.png)](slide16.png "Query 02: implementation using merge+filter")

Query 02: implementation using merge+filter

[![Query 03: Problem Statement](slide17.png)](slide17.png "Query 03: Problem Statement")

Query 03: Problem Statement

- Calculate total sales per Invoice for each Customer

[![Query 03: apply-based vs vectorized implementation](slide18.png)](slide18.png "Query 03: apply-based vs vectorized implementation")

Query 03: apply-based vs vectorized implementation

[![Exploring Type-2 Bottlenecks (Vectorized implementation without optimized data flow)](slide19.png)](slide19.png "Exploring Type-2 Bottlenecks (Vectorized implementation without optimized data flow)")

Exploring Type-2 Bottlenecks (Vectorized implementation without optimized data flow)

[![Query 04: Problem Statement](slide20.png)](slide20.png "Query 04: Problem Statement")

Query 04: Problem Statement

[![Query 04: Vectorized implementation (unoptimized data flow)](slide21.png)](slide21.png "Query 04: Vectorized implementation (unoptimized data flow)")

Query 04: Vectorized implementation (unoptimized data flow)

[![Query 04: Vectorized implementation (optimized data flow)](slide22.png)](slide22.png "Query 04: Vectorized implementation (optimized data flow)")

Query 04: Vectorized implementation (optimized data flow)

[![Exploring Type-3 Bottlenecks (Vectorized implementation with optimized data flow)](slide23.png)](slide23.png "Exploring Type-3 Bottlenecks (Vectorized implementation with optimized data flow)")

Exploring Type-3 Bottlenecks (Vectorized implementation with optimized data flow)

[![Query 05: Problem Statement](slide24.png)](slide24.png "Query 05: Problem Statement")

Query 05: Problem Statement

[![Query 05: Vectorized implementation (optimized data flow)](slide25.png)](slide25.png "Query 05: Vectorized implementation (optimized data flow)")

Query 05: Vectorized implementation (optimized data flow)

[![Learning Summary](slide26.png)](slide26.png "Learning Summary")

Learning Summary

[![Learning \#1: Breaking out of the loop](slide27.png)](slide27.png "Learning #1: Breaking out of the loop")

Learning \#1: Breaking out of the loop

[![Learning \#2: Single-node processing might be enough](slide28.png)](slide28.png "Learning #2: Single-node processing might be enough")

Learning \#2: Single-node processing might be enough

[![Learning \#3: FireDucks might be the one you are looking for!](slide29.png)](slide29.png "Learning #3: FireDucks might be the one you are looking for!")

Learning \#3: FireDucks might be the one you are looking for!

[![Thank you](slide30.png)](slide30.png "Thank you")

Thank you

https://github.com/qsourav/PyData-Global-2025

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2025,
  author = {Bochman, Oren},
  title = {Lessons Learnt in Optimizing a Large-Scale Pandas Application
    Using {Polars,} {FireDucks} and {cuDF:} {Go} {Smart} and {Save}
    {More!}},
  date = {2025-12-09},
  url = {https://orenbochman.github.io/posts/2025/2025-12-09-pydata-optimizing-pandas-using-polars-cuDf-and-FireDucks/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2025. “Lessons Learnt in Optimizing a Large-Scale Pandas Application Using Polars, FireDucks and cuDF: Go Smart and Save More!” December 9. <https://orenbochman.github.io/posts/2025/2025-12-09-pydata-optimizing-pandas-using-polars-cuDf-and-FireDucks/>.
