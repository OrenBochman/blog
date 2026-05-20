[![pydata global](pydata_logo.png)](pydata_logo.png "pydata global")

pydata global

> **TIP:**
>
> In many regulated industries—finance, healthcare, insurance—logistic regression remains the model of choice for its interpretability and regulatory acceptability.
>
> Yet capturing non-linear effects and interactions often requires variable binning, and naive approaches (equal-width or quantile cuts) can either wash out signal or invite overfitting.
>
> In this 30-minute session, data scientists and risk analysts with a working knowledge of logistic regression and Python will learn to:
>
> - Diagnose the weaknesses of basic binning strategies.
> - Select and apply optimal-binning algorithms for different use cases.
> - Assess bin stability and guard against model overfit.
>
> All code, data samples, and a notebook will be available on GitHub.

Despite the rise of complex “black-box” models, regulated environments still demand transparency. Properly binned variables not only improve model fit but also yield coefficients that the business and auditors can interpret.

However, determining cut-points that preserve true signal while avoiding data-snooping bias is non-trivial.

> **TIP:**
>
> - Understand the basic idea behind binning (the what)
> - To know in which contexts variable binning makes sense (the when and why).
> - Choose among popular optimal-binning techniques (e.g., ChiMerge, MDLP, decision-tree-based) based on data size, feature type, and operational constraints (the how).

> **TIP:**
>
> - Data scientists and risk analysts who use logistic regression in regulated settings and need a reproducible, explainable feature-engineering pipeline.
> - Prerequisites: Basic Python (pandas, scikit-learn) and logistic-regression familiarity
> - Materials: GitHub repo with notebook, data samples, will be shared during the talk

> **IMPORTANT:**
>
> - [OptBinning](https://gnpalencia.org/optbinning/)

> **TIP:**

## Outline

[![Optimal Binning in Logistic Regression](slide01.png)](slide01.png "Optimal Binning in Logistic Regression")

Optimal Binning in Logistic Regression

[![Agenda](slide02.png)](slide02.png "Agenda")

Agenda

[![Who am I](slide03.png)](slide03.png "Who am I")

Who am I

[![Modeling under uncertainty](slide05.png)](slide05.png "Modeling under uncertainty")

Modeling under uncertainty

[![From model risks to modeling choices](slide06.png)](slide06.png "From model risks to modeling choices")

From model risks to modeling choices

[![Logistic regression recap](slide07.png)](slide07.png "Logistic regression recap")

Logistic regression recap

[![what is binning](slide08.png)](slide08.png "what is binning")

what is binning

[![WoE and IV](slide09.png)](slide09.png "WoE and IV")

WoE and IV

Weight of Evidence (WoE) and Information Value (IV) are two key concepts in variable binning for logistic regression.

WoE_j = \ln\left(\frac{Good_j / Total\\ Good}{Bad_j / Total\\ Bad }\right) \tag{1}

IV = \sum_j \left(\frac{Good_j}{Total\\ Good} - \frac{Bad_j}{Total\\ Bad}\right) \times WoE_j \tag{2}

[![IV as a feature selection metric](slide10.png)](slide10.png "IV as a feature selection metric")

IV as a feature selection metric

[![When log-odds are not linear](slide11.png)](slide11.png "When log-odds are not linear")

When log-odds are not linear

[![What is binning?](slide12.png)](slide12.png "What is binning?")

What is binning?

[![Model A vs Model B: What is wrong here?](slide13.png)](slide13.png "Model A vs Model B: What is wrong here?")

Model A vs Model B: What is wrong here?

[![Investigating like Sherlock Holmes](slide14.png)](slide14.png "Investigating like Sherlock Holmes")

Investigating like Sherlock Holmes

[![Case study dataset](slide15.png)](slide15.png "Case study dataset")

Case study dataset

[![Feature Overview](slide16.png)](slide16.png "Feature Overview")

Feature Overview

### Four Binning Strategies

[![Age vs CHD risk – Decile (Quantile) Binning](slide17.png)](slide17.png "Age vs CHD risk – Decile (Quantile) Binning")

Age vs CHD risk – Decile (Quantile) Binning

[![Age vs CHD risk – Equal-Width Binning](slide18.png)](slide18.png "Age vs CHD risk – Equal-Width Binning")

Age vs CHD risk – Equal-Width Binning

[![Age vs CHD risk – Tree-Based Binning](slide19.png)](slide19.png "Age vs CHD risk – Tree-Based Binning")

Age vs CHD risk – Tree-Based Binning

[![Age vs CHD risk – Optimized Binning](slide20.png)](slide20.png "Age vs CHD risk – Optimized Binning")

Age vs CHD risk – Optimized Binning

[![Four modeling approaches we will compare](slide21.png)](slide21.png "Four modeling approaches we will compare")

Four modeling approaches we will compare

[![AUC & ROC comparison](slide22.png)](slide22.png "AUC & ROC comparison")

AUC & ROC comparison

[![How Boosting Algorithms Handle Binning 1](slide23.png)](slide23.png "How Boosting Algorithms Handle Binning 1")

How Boosting Algorithms Handle Binning 1

[![How Boosting Algorithms Handle Binning 2](slide24.png)](slide24.png "How Boosting Algorithms Handle Binning 2")

How Boosting Algorithms Handle Binning 2

[![Optimal binning as an optimisation problem](slide25.png)](slide25.png "Optimal binning as an optimisation problem")

Optimal binning as an optimisation problem

[![MDLP: Entropy-based Binning](slide26.png)](slide26.png "MDLP: Entropy-based Binning")

MDLP: Entropy-based Binning

[![Mathematical programming-based optimal binning](slide27.png)](slide27.png "Mathematical programming-based optimal binning")

Mathematical programming-based optimal binning

[![Stochastic optimal binning](slide28.png)](slide28.png "Stochastic optimal binning")

Stochastic optimal binning

[![What “good” looks like](slide29.png)](slide29.png "What “good” looks like")

What “good” looks like

[![Conclusion & how to explore further](slide30.png)](slide30.png "Conclusion & how to explore further")

Conclusion & how to explore further

[![OptBinning library](slide31.png)](slide31.png "OptBinning library")

OptBinning library

- OptBinning is a Python library for optimal binning and scorecard modelling.
- Created and maintained by Guillermo Navas-Palencia.
- Implements mathematical programming formulations for:
  - Binary, continuous and multiclass targets.
  - Monotonicity, minimum size, and other business constraints
- Documentation: gnpalencia.org/optbinningGitHub
- repository: github.com/guillermo-navas-palencia/optbinning

[![Question](slide32.png)](slide32.png "Question")

Question

[![Thanks](slide33.png)](slide33.png "Thanks")

Thanks

### Reflection

We looked at what we mean by binning in Logistic Regression, why and when to use it, and how to choose an optimal binning technique based on data and operational constraints.

We also saw how to implement these techniques in Python using the OptBinning library.

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2025,
  author = {Bochman, Oren},
  title = {Optimal {Variable} {Binning} in {Logistic} {Regression}},
  date = {2025-12-10},
  url = {https://orenbochman.github.io/posts/2025/2025-12-10-pydata-optimal-binning-in-logistic-regression/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2025. “Optimal Variable Binning in Logistic Regression.” December 10. <https://orenbochman.github.io/posts/2025/2025-12-10-pydata-optimal-binning-in-logistic-regression/>.
