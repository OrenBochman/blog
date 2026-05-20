# An error occurred.

Unable to execute JavaScript.

> **TIP:**
>
> Building ML models for financial fraud detection sounds straightforward, until you have to evaluate, validate, and deploy them in real-world pipelines. This talk walks through the practical stack, metrics, and mindsets needed to build fraud detection systems with modern Python. We’ll cover key challenges like concept drift, extreme class imbalance, false-positive overload, and why the usual ML workflows fall short. Along the way, we’ll explore a real-world architecture using classical ML, deep learning, and GNNs, plus the validation techniques and production patterns that make or break fraud systems. If you’re tired of toy problems and want patterns that survive real money and real latency, this talk’s for you.

This talk distills a production‑tested path for real‑time financial fraud detection in Python (inc. choosing the right objective, validating in time, and shipping with guardrails).

Core idea:

Optimize the business decision (alerts under cost/latency constraints), not just the ML score.

> **TIP:**
>
> - A copy‑and‑adapt roadmap for deploying financial fraud detection services with Python.
>
> - A latency‑aware model selection heuristic.
>
> - A minimal deployment pattern (service, thresholds, monitoring) that scales from pilot to production.

> **TIP:**
>
> - Basic Python and DataFrames,
> - ML classification basics,
> - HTTP/JSON.

> **TIP:**

[workshop repo](https://github.com/hugobowne/AI-for-SWEs)

> **TIP:**

## Outline

1.  Problem framing: Adversaries, label delay, extreme imbalance, and why “accuracy” lies.

2.  Metrics that matter: Precision and recall, AUC‑PR vs ROC, cost‑weighted utility, calibration for decisions.

3.  Validation done right: Temporal splits, rolling/blocked CV with gap, prequential test‑then‑train, leak and drift traps.

4.  Modeling under latency budgets: Where XGBoost shines, when to add tabular DL, injecting graph signals without blowing latency (simple handcrafted graph stats + GNNs).

5.  From notebook to service: Small, testable core, FastAPI endpoint, thresholds and shadow mode, alert quotas, analyst feedback loops.

6.  Operations & monitoring: Drift indicators, calibration checks, label‑delay dashboards, canaries/rollbacks.

7.  Wrap‑up/Q&A: Failure modes and a 1‑page runbook.

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

[![](slide27.png)](slide27.png)

[![](slide28.png)](slide28.png)

[![](slide29.png)](slide29.png)

[![](slide30.png)](slide30.png)

[![](slide01.png)](slide01.png)

## Reflections

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2025,
  author = {Bochman, Oren},
  title = {Realtime {Financial} {Fraud} {Detection} with {Modern}
    {Python}},
  date = {2025-12-10},
  url = {https://orenbochman.github.io/posts/2025/2025-12-10-pydata-real-time-financial-fraud-detection/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2025. “Realtime Financial Fraud Detection with Modern Python.” December 10. <https://orenbochman.github.io/posts/2025/2025-12-10-pydata-real-time-financial-fraud-detection/>.
