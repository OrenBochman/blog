# An error occurred.

Unable to execute JavaScript.

> **TIP:**
>
> Scientific Python Ecosystem offers a wide variety of numerical packages, such as `NumPy`, `CuPy`, or `JAX`. One of the domains that also captures a lot of attention in the community is sparse computing.
>
> In this talk, we will present the current landscape of sparse computing in the Python ecosystem and our efforts to revive/expand it.
>
> Our main contributions to the Python ecosystem cover:
>
> 1.  making a novel `Finch` sparse tensor compiler and Galley scheduler available for the community,
> 2.  standardizing various aspects of sparse computing.
>
> We will show how to use the `Finch` compiler with the PyData/Sparse package and how it outperforms well-established alternatives for multiple kernels, such as `[MTTKRP](http://tensor-compiler.org/docs/data_analytics.html)` or `[SDDMM](http://tensor-compiler.org/docs/machine_learning.html)`.
>
> Real-world use-cases will show you how, step-by-step, Python practitioners can migrate their code to an Array API compatible version and benefit from tensor operator fusion and autoscheduling capabilities offered by the `Finch` compiler.
>
> Apart from the existing Julia implementation, the number of sparse backends offered by PyData/Sparse will grow in the future to provide a Python-native alternatives for `scipy.sparse` and `Numba` solutions. One of them that is currently under development is finch-tensor-lite, a pure Python rewrite of `Finch.jl` compiler, meant to make the solution lightweight by dropping Julia runtime dependency while providing the majority of features.

In this talk we’re going to understand the current landscape of sparse computing in the Python ecosystem first. Then a high-level overview of the Finch technology and compiler’s architecture will be presented together with other solutions vital for the project: Array API Standard and binsparse format.

Next, we’re going to present a selected set of benchmarks - also focusing on real world use-cases: how Finch impacts users’ experience when writing sparse programs in Python. Last but not least a showcase of the current development will be shown - pure Python rewrite of Finch compiler.

> **TIP:**
>
> - Mateusz Sokół
>   - is a Software Engineer at Quansight, working on multitude of open source projects in the Scientific Python Ecosystem.
>   - His GitHub profile here: https://github.com/mtsokol
> - Willow Marie Ahrens
>   - is assistant professor in the School of Computer Science at Georgia Tech. She is inspired to make programming high-performance computers more productive, efficient, and accessible. Her research focuses on using compilers to accelerate productive programming languages with state-of-the-art data structures, algorithms, and architectures, bridging the gap between program flexibility and performance. She’s the author of the Finch sparse tensor programming language. Finch supports general programs on general tensor formats, such as sparse, run-length-encoded, banded, or otherwise structured tensors. Please reach out if you are interested in doing research at Georgia Tech!
>   - [talk repo](https://github.com/pydata/sparse) https://sparse.pydata.org/en/stable/
>   - [slide deck](slides.pdf)

## Outline

- **High-level user model:**

  - Domain users should be able to write NumPy-like array code.
  - They should specify whether inputs are dense or sparse, but not manually decide loop order, fusion strategy, or format-specific traversal.
  - Finch’s auto-scheduler is meant to replace the human performance engineer in that optimization step.

- **Project architecture:**

  - The Python-facing layer uses the Array Application Programming Interface standard so code can be written against a common array interface.
  - Finch lowers code through several stages: Python Array API → Finch Logic → Finch Notation → Finch Assembly → backend code such as Numba or C.
  - The speakers emphasize that each stage has interpreters/tests to validate correctness.

- **Compilation example:**

  - The example multiplies two tensors where one input is lower triangular.
  - Finch represents the triangular mask in its internal logic.
  - During lowering, it generates loops that only traverse the lower triangular part.
  - Algebraic simplification then removes useless work, such as multiplying by zero or keeping empty loops.

- **Use-case workflow:**

  - Start with NumPy-like numerical code.
  - Rewrite it to be Array API-compatible.
  - Isolate the computationally expensive loop.
  - Add a sparse compile decorator and wrap inputs in appropriate tensor formats.
  - Finch then Just-in-Time compiles the sparse kernel.

- **Examples shown:**

  - **Hyperlink-Induced Topic Search (HITS):** A graph link-analysis algorithm was rewritten from GraphBLAS-style code into Array API-compatible code, then the power-iteration kernel was compiled.
  - **Canonical Polyadic (CP) decomposition:** A masked tensor decomposition routine from `tensortools` was adapted with relatively small code changes and compiled with Finch.

- **Benchmark result:**

  - For Canonical Polyadic decomposition at low density, around 3%, the speakers report more than a sevenfold speedup on their test infrastructure.

- **Ecosystem direction:**

  - `finch-tensor` is available as a Python package, currently using JuliaCall to access the Julia implementation.
  - The project is also working on a pure Python compiler path using Numba as a backend.
  - The 2026 goal is to push “Finch Light” toward an initial community release.

- **Standardization angle:**

  - The speakers also discuss `binsparse`, a binary sparse storage/interchange format intended to standardize sparse array exchange across formats and platforms.
  - The goal is broader PyData/Sparse integration, including Finch as a selectable backend.

- **Talk:**

  - *PyData/Sparse and Finch*, PyData Global 2025.

- **Main thesis:**

  - Array programming is highly productive, but sparse arrays are still hard to use efficiently in Python.

  - The speakers argue that sparse tensor compilers can unify sparse computation by translating high-level array code into efficient kernels for many formats and backends.

[![Title](slide01.png)](slide01.png "Title")

Title

- **Problem being addressed:**

  - Sparsity appears in scientific computing, social networks, machine learning, and image processing.
  - Dense arrays have a simple relationship between mathematical indexing and memory layout.
  - Sparse arrays break that simplicity because formats such as Compressed Sparse Column store only nonzero values, plus index and pointer arrays.
  - Efficient sparse algorithms must avoid multiplying by zero and must traverse sparse data structures correctly.

[![Array Programming is Productive](slide02.png)](slide02.png "Array Programming is Productive")

Array Programming is Productive

- **Current ecosystem issue:**

  - Sparse support in the PyData ecosystem is fragmented.
  - SciPy, PyTorch, and PyData/Sparse each support different dimensions, formats, operations, and backends.
  - The desired target is a central sparse array library that is n-dimensional, performant, supports central processing units and graphics processing units, and has broad linear algebra and graph algorithm coverage.

[![Sparsity is everywhere](slide03.png)](slide03.png "Sparsity is everywhere")

Sparsity is everywhere

- **Why compilers matter:**

  - The implementation burden grows combinatorially across operations, structures, data formats, and hardware architectures.
  - A sparse tensor compiler can take a high-level sparse program plus format descriptions and generate specialized code automatically.
  - This lets different interfaces and sparse formats map into a shared intermediate representation rather than requiring hand-written kernels for every case.

[![Sparsity is Challenging](slide04.png)](slide04.png "Sparsity is Challenging")

Sparsity is Challenging

[![Fragmented sparse support today](slide05.png)](slide05.png "Fragmented sparse support today")

Fragmented sparse support today

[![Implementation Burden Grows Exponentially](slide06.png)](slide06.png "Implementation Burden Grows Exponentially")

Implementation Burden Grows Exponentially

[![Sparse Tensor Compilers Can Help](slide07.png)](slide07.png "Sparse Tensor Compilers Can Help")

Sparse Tensor Compilers Can Help

[![High-Level Sparse Array Programming in Python](slide08.png)](slide08.png "High-Level Sparse Array Programming in Python")

High-Level Sparse Array Programming in Python

[![Agenda](slide09.png)](slide09.png "Agenda")

Agenda

[![Project Overview](slide10.png)](slide10.png "Project Overview")

Project Overview

[![Array API standard](slide11.png)](slide11.png "Array API standard")

Array API standard

[![Finch compiler & Galley](slide12.png)](slide12.png "Finch compiler & Galley")

Finch compiler & Galley

[![Finch Architecture](slide13.png)](slide13.png "Finch Architecture")

Finch Architecture

[![Compilation Example:](slide14.png)](slide14.png "Compilation Example:")

Compilation Example:

[![Lowering example](slide15.png)](slide15.png "Lowering example")

Lowering example

[![Lowering example](slide16.png)](slide16.png "Lowering example")

Lowering example

[![Lowering example](slide17.png)](slide17.png "Lowering example")

Lowering example

[![Lowering example](slide18.png)](slide18.png "Lowering example")

Lowering example

[![Lowering example](slide19.png)](slide19.png "Lowering example")

Lowering example

[![Lowering example](slide20.png)](slide20.png "Lowering example")

Lowering example

[![Lowering example](slide21.png)](slide21.png "Lowering example")

Lowering example

[![Lowering example](slide22.png)](slide22.png "Lowering example")

Lowering example

[![Lowering example](slide23.png)](slide23.png "Lowering example")

Lowering example

[![Lowering example](slide24.png)](slide24.png "Lowering example")

Lowering example

[![Lowering example](slide25.png)](slide25.png "Lowering example")

Lowering example

[![Lowering example](slide26.png)](slide26.png "Lowering example")

Lowering example

[![Lowering example](slide27.png)](slide27.png "Lowering example")

Lowering example

[![Lowering example](slide28.png)](slide28.png "Lowering example")

Lowering example

[![Lowering example](slide29.png)](slide29.png "Lowering example")

Lowering example

[![Lowering example](slide30.png)](slide30.png "Lowering example")

Lowering example

[![Lowering example](slide31.png)](slide31.png "Lowering example")

Lowering example

[![Lowering example](slide32.png)](slide32.png "Lowering example")

Lowering example

[![Use cases & benchmarks](slide33.png)](slide33.png "Use cases & benchmarks")

Use cases & benchmarks

[![Use case workflow](slide34.png)](slide34.png "Use case workflow")

Use case workflow

[HITS](https://en.wikipedia.org/wiki/HITS_algorithm) ::: {.sl-text}

[![Use case: HITS algorithm](slide35.png)](slide35.png "Use case: HITS algorithm")

Use case: HITS algorithm

::::

[![HITS algorithm: Array API conversion](slide36.png)](slide36.png "HITS algorithm: Array API conversion")

HITS algorithm: Array API conversion

[![HITS algorithm: complied with Finch](slide37.png)](slide37.png "HITS algorithm: complied with Finch")

HITS algorithm: complied with Finch

[![Use case: CP Decomposition](slide38.png)](slide38.png "Use case: CP Decomposition")

Use case: CP Decomposition

[![CP Decomposition: accelerated computation](slide39.png)](slide39.png "CP Decomposition: accelerated computation")

CP Decomposition: accelerated computation

[![Try it out yourself!](slide40.png)](slide40.png "Try it out yourself!")

Try it out yourself!

[![Ecosystem Overview](slide41.png)](slide41.png "Ecosystem Overview")

Ecosystem Overview

[![Ecosystem Integration](slide42.png)](slide42.png "Ecosystem Integration")

Ecosystem Integration

[![binsparse: A binary sparse storage format](slide43.png)](slide43.png "binsparse: A binary sparse storage format")

binsparse: A binary sparse storage format

[![Questions](slide44.png)](slide44.png "Questions")

Questions

- **Q&A point:**

  - A question focused on understanding the lowering process.
  - The answer emphasized that the lowering design is intended to be extensible across triangular, run-length encoded, and general sparse matrix structures, making the system more future-proof.

[![Q&A](slide45.png)](slide45.png "Q&A")

Q&A

## Reflections

WoW

1.  a good sparse library for python calls for implementing many Stats and ML algorithm using this backend.
2.  Also streaming algorithms can benefit from this.
3.  The tensor logic needs `einsum` - is this op supported ?

> **TIP:**
>
> For situations where more complex operations are needed, Finch supports an ([**einsum?**](#ref-einsum)) syntax on sparse and structured tensors.
>
> ``` julia
> julia> @einsum E[i] += A[i, j] * B[i, j]
> julia> @einsum F[i, k] <<max>>= A[i, j] + B[j, k]
> ```

## Resources

Repos and Docs:

- [binsparse specification](https://github.com/GraphBLAS/binsparse-specification) A cross-platform binary storage format for sparse data, particularly sparse matrices.

- [finch-tensor](https://github.com/finch-tensor)

- [Program seamlessly with sparse and structured tensors](https://finch-tensor.org/)

- Papers:

  - [Finch: Sparse and Structured Tensor Programming with Control Flow](https://dl.acm.org/doi/10.1145/3720473)

  - [Galley: Modern Query Optimization for Sparse Tensor Programs](https://doi.org/10.48550/arXiv.2408.14706)

  - [Looplets: A Language for Structured Coiteration](https://doi.org/10.1145/3579990.3580020)

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2025,
  author = {Bochman, Oren},
  title = {Sparse \& {Finch}},
  date = {2025-12-12},
  url = {https://orenbochman.github.io/posts/2025/2025-12-11-pydata-sparse-and-finch/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2025. “Sparse & Finch.” December 12. <https://orenbochman.github.io/posts/2025/2025-12-11-pydata-sparse-and-finch/>.
