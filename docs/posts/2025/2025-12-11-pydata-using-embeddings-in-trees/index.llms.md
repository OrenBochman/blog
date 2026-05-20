[![pydata global](pydata_logo.png)](pydata_logo.png "pydata global")

pydata global

> **TIP:**
>
> Text embeddings are a powerful tool for encoding the essence of unstructured text data into a structured, dense, multidimensional vector representation. Due to their inner structure, tree based models such as decision trees, gradient boosted decision trees and random forests struggle to effectively use text embeddings features. This is due to the fact that trees can use only one feature every time they split, so the number of used embedding dimensions is limited to the tree depth.
>
> Other models, such as linear models for example, can use text embeddings more effectively because they are able to use all of the embedding dimensions simultaneously.
>
> In this presentation we will present a novel approach to transform text embedding features into a format that tree-based models can effectively use. The proposed approach combines the strengths of non-tree based models with predictive power of tree based models to create a more effective feature representation for tree-based models.

The presentation is aimed at Data Science and Machine Learning practitioners who are already familiar with tree-based models and want to learn how to effectively incorporate text embeddings features to boost the performances of their models.

The methodology showcased in the presentation is available in the [`sklearo`](https://github.com/ClaudioSalvatoreArcidiacono/sklearo) open source package.

> **TIP:**
>
> - fundamental machine learning concepts such as overfitting, cross-validation, and feature engineering is recommended but not required.

> **IMPORTANT:**
>
> - [sklearo](https://github.com/ClaudioSalvatoreArcidiacono/sklearo) A Python package featuring scikit-learn like transformers for feature preprocessing, compatible with all kind of dataframes thanks to narwhals.
> - [felimination](https://github.com/ClaudioSalvatoreArcidiacono/felimination) Utility class to perform recursive feature elimination with cross validation and permutation importance as importance metric.

> **TIP:**
>
> - Claudio Salvatore Arcidiacono
>   - Claudio Salvatore Arcidiacono is a Senior Data Scientist at Mollie. I have been working in the fintech sector over the past 7 years,
>   - He has lots of experience in classical machine learning problems, mainly in binary classification problems. He loves to contribute to data science open source packages like feature engine, scikit-learn and narwhals. He maintains a couple of packages himself (`[felimination](https://github.com/ClaudioSalvatoreArcidiacono/felimination)` and `[sklearo](https://github.com/ClaudioSalvatoreArcidiacono/sklearo)`). In his free time he is a coffee scientist, using a data driven approach to dial in the perfect cup of espresso.
>   - [talk repo](https://github.com/hugobowne/AI-for-SWEs)
>   - [slide deck](https://github.com/hugobowne/AI-for-SWEs)

## Outline

- 5 minutes Overview of text embeddings, how tree-based models are built, and the challenges they face with text embeddings compared to linear models.
- 5 minutes Explanation of how can we leverage non-tree based models to transform text embeddings into a format that tree based models can effectively use.
- 5 minutes Explanation on cross-fitting, a technique used to avoid target leakage when generating features using the target variable.
- 5 minutes Code examples of how this technique can be used in practice using the sklearo open source library.
- 5 minutes Performance comparison of tree based models using text embeddings as-is vs using the transformed features.

[![](slide01.png)](slide01.png)

[![](slide02.png)](slide02.png)

[![](slide03.png)](slide03.png)

[![](slide04.png)](slide04.png)

[![](slide05.png)](slide05.png)

[![](slide06.png)](slide06.png)

[![](slide07.png)](slide07.png)

[![](slide08.png)](slide08.png)

[![](slide09.png)](slide09.png)

[![](slide10.png)](slide10.png)

[![](slide11.png)](slide11.png)

[![](slide12.png)](slide12.png)

[![](slide13.png)](slide13.png)

[![](slide14.png)](slide14.png)

[![](slide15.png)](slide15.png)

[![](slide16.png)](slide16.png)

[![](slide17.png)](slide17.png)

[![](slide18.png)](slide18.png)

[![](slide19.png)](slide19.png)

[![](slide20.png)](slide20.png)

[![](slide21.png)](slide21.png)

[![](slide22.png)](slide22.png)

[![](slide23.png)](slide23.png)

[![](slide24.png)](slide24.png)

[![](slide25.png)](slide25.png)

[![](slide26.png)](slide26.png)

## Reflections

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2025,
  author = {Bochman, Oren},
  title = {How to {Effectively} Use Text Embeddings in Tree Based
    Models},
  date = {2025-12-12},
  url = {https://orenbochman.github.io/posts/2025/2025-12-11-pydata-using-embeddings-in-trees/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2025. “How to Effectively Use Text Embeddings in Tree Based Models.” December 12. <https://orenbochman.github.io/posts/2025/2025-12-11-pydata-using-embeddings-in-trees/>.
