# An error occurred.

Unable to execute JavaScript.

> **TIP:**
>
> In Survival analysis tackles one of the oldest and most universal questions in data science: Can we learn from the past when something will happen in the future? I will introduce you to the core concepts of survival analysis, visualize time-to-event datasets with python and R, and introduce pertinent probability distributions. Classical analysis methods for fitting such datasets - some developed long before the age of modern computing - will be confronted to machine-learning approaches. Along the way, surprising paradoxes and counterintuitive results will reveal why survival analysis is not merely a blend of regression and classification, but an important prediction problem of its own.

Since at least 1693, when the first actuarial tables were used for calculating insurance premiums, survival (or “time-to-event”) analysis has been relevant for many disciplines. Whether predicting when a mechanical component will fail, when a patient will recover, or when a customer will return a product, survival analysis has applications in nearly every domain - from engineering and medicine to finance and e-commerce. Despite its broad applicability and deep statistical foundations, survival analysis remains underappreciated in modern data science.

I therefore want to give the audience, who does not need to have heard of survival analysis before, an impression about what survival analysis is about, what one needs to be careful with, and which analytical and computational tools to use to get to reliable predictions. In a step-by-step constructive approach, I will slowly guide the audience from the simplest flavor of the fully observed time-to-event-problem to the more intricate versions that include censoring and truncation, in which managing one’s own ignorance becomes the most important and challenging aspect. Numerous code examples in python and R will make the talk hands-on, and allow listeners to replicate the numerical experiments and visualizations. At the same time, I will constantly recur to lucid everyday-examples (what age should the house that you buy have so you avoid problems? how long can you use your winter tires on your car? why is milk often still good after the best-before date?) - and thereby hopefully convince the audience: Survival analysis is almost always everywhere.

> **TIP:**
>
> - After the talk, the audience will be able to recognize the time-to-event problem in their own domain, and
> - Use the appropriate tools in python and R to analyze and model it.

> **TIP:**
>
> - Data scientists and risk analysts who use logistic regression in regulated settings and need a reproducible, explainable feature-engineering pipeline.
> - Prerequisites: Basic Python (pandas, scikit-learn) and logistic-regression familiarity
> - Materials: GitHub repo with notebook, data samples, will be shared during the talk

[workshop repo](https://github.com/zgcharaf/Pydata-Global-Talk-December-2025)

> **TIP:**

## Outline

- Motivation: The oldest problem in data science? \[1 min\]
- Introduction: Prediction problems that are in fact survival problems? \[3 min\]
- The simple case: Fully observed datasets. Visualization of the cumulative failure distribution. \[3 min\]
- The Weibull distribution as the working horse of survival analysis: How to model early failures, constant risks and wear-outs. \[4 min\]
- Why reporting another case of illness can be good news. \[2 min\]
- Censoring: What can we learn from not having observed anything yet? \[2 min\]
- The Kaplan-Meier estimator and the maximum-likelihood principle. \[5 min\]
- Machine Learning approaches to the survival problem. \[3 min\]
- Outlook: Which degree of individualized survival forecasts can we expect in the future? \[2 min\]

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

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2025,
  author = {Bochman, Oren},
  title = {Reviving {Survival} {Analysis:} {Timeless,} {Yet}
    {Overlooked?}},
  date = {2025-12-10},
  url = {https://orenbochman.github.io/posts/2025/2025-12-10-pydata-survial/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2025. “Reviving Survival Analysis: Timeless, Yet Overlooked?” December 10. <https://orenbochman.github.io/posts/2025/2025-12-10-pydata-survial/>.
