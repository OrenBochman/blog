## Beyond the black box - Interpretability of LLMs in Finance

- Hariom Tatsat
  - [Website](https://htatsat.com/)
  - [LinkedIn](https://linkedin.com/in/hariomtatsat)
  - book:
  - paper: [Beyond the Black Box: Interpretability of LLMs in Finance](https://arxiv.org/pdf/2505.24650)
- Barclays

> **NOTE:**
>
> - **Talk topic:** “Beyond the Black Box: Interpretability of LLMs in Finance,” presented by Hariom from Barclays’ Quantitative AI team.
>
> - **Central claim:** Finance is a high-stakes domain, so large language models need more than prompt engineering, guardrails, chain-of-thought prompting, and external evaluation. The speaker argues for looking inside model internals.
>
> - **Why interpretability matters in finance:**
>
>   - Many AI pilots fail to reach production because organizations lack confidence in model behavior.
>   - Leaders often attribute this lack of confidence to poor explainability.
>   - Existing enterprise tools mostly inspect models externally rather than analyzing internal representations.
>   - The speaker connects this to the broader lesson of the 2008 financial crisis: poorly understood models can create systemic risk.
>
> - **Types of interpretability discussed:**
>
>   - Feature attribution: estimating how much each input contributes to an output.
>   - Behavioral interpretability: testing how outputs change when inputs are perturbed.
>   - Simple surrogate models: decision trees, linear probes, and similar approximations.
>   - Visual explanations.
>   - **Mechanistic interpretability:** studying internal model structures directly, described as “neuroscience” or “MRI” for artificial intelligence.
>
> - **Mechanistic interpretability motivation:**
>
>   - Large language model neurons are often **polysemantic**, meaning one neuron may encode several unrelated concepts.
>   - This makes direct inspection difficult.
>   - Sparse autoencoders are presented as a way to decompose mixed internal activations into more interpretable features.
>
> - **Sparse autoencoders:**
>
>   - A sparse autoencoder is attached to an internal layer of a model.
>   - It acts like a microscope on the residual stream.
>   - It separates blended model concepts into features that can sometimes be assigned human-readable labels.
>   - These labels matter because raw numerical activations are much less useful for risk, compliance, and audit conversations.
>
> - **Use case 1: sentiment feature for credit risk**
>
>   - The team looked for internal features associated with credit-risk concepts.
>   - They used Neuronpedia and sparse-autoencoder features to identify model activations related to phrases such as “credit risk.”
>   - They then used feature steering: artificially increasing activation of the credit-risk feature.
>   - When the model was asked to score financial sentiment, steering made its reasoning focus more on credit-related cues such as lower credit score and secured financing.
>   - Across many sentences, steered outputs were closer to human annotations than unsteered outputs.
>
> - **Use case 2: “Warren Buffett AI” for trading signals**
>
>   - The hypothesis was that LLM internals may contain useful financial abstractions learned from internet-scale training data.
>   - The team tested whether internal features activated by financial news headlines could predict whether prices went up or down.
>   - They used around ten years of financial headlines, a Gemma model, sparse autoencoder features, and a classifier.
>   - Around 200 features were extracted.
>   - Important features included named entities, financial terms, and stock ticker symbols.
>   - The speaker framed this as early-stage but promising evidence that internal model representations may contain trading-relevant signal.
>
> - **Use case 3: hallucination police**
>
>   - The team proposed monitoring whether finance-specific internal features activate when a finance chatbot answers finance questions.
>   - If the relevant features do not activate above a threshold, the system treats the answer as insufficiently grounded.
>   - In that case, it can trigger prompt enhancement, citation retrieval, or additional grounding.
>   - The goal is not merely to detect hallucination from the output, but to use internal model behavior as an early warning signal.
>
> - **Practical workflow for hallucination control:**
>
>   - Identify finance-related sparse-autoencoder features.
>   - Monitor their activation during financial queries.
>   - Set a calibrated activation threshold.
>   - If activation is high, allow the answer.
>   - If activation is low, enrich the prompt or require grounded citations before producing the final response.
>
> - **Limitations:**
>
>   - The field is still early-stage.
>   - These methods are currently more feasible for open-weight models than closed commercial models.
>   - Sparse autoencoders may inspect only a limited number of layers.
>   - Broader circuit-tracing methods are needed to understand multi-layer model behavior.
>   - Thresholds and feature selection require calibration and judgment.
>
> - **Final takeaway:**
>
>   - The speaker argues that mechanistic interpretability is underrated in finance.
>   - Better internal understanding could increase trust among regulators, validators, risk teams, and business stakeholders.
>   - The broader ambition is to make AI systems safer and more production-ready in high-stakes domains by understanding not just what they output, but why their internal representations support those outputs.

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2026,
  author = {Bochman, Oren},
  title = {Beyond the Black Box -},
  date = {2026-04-28},
  url = {https://orenbochman.github.io/posts/2026/04-30-ODSC-AI-2026-Day-3/talk4.html},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2026. “Beyond the Black Box -.” April 28. <https://orenbochman.github.io/posts/2026/04-30-ODSC-AI-2026-Day-3/talk4.html>.
