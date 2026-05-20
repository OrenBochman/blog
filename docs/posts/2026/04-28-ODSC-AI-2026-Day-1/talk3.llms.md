## vLLM with the Transformers Modelling Backend

Play

![Vimeo](https://f.vimeocdn.com/p/images/crawler_logo.png)

- Harry Mellor
  - vLLM with the Transformers Modelling Backend
  - [LinkedIn](https://www.linkedin.com/in/harrymellor/)
  - HuggingFace
  - [tutorial](https://github.com/hmellor/vllm-transformers-modelling-backend-tutorial)

> **NOTE:**
>
> - Tools:
>   - Transformers
>   - torchtitan
>   - Axolotl
>   - TRL
>   - Unsloth
>
> 1.  [Transformers](https://github.com/hmellor/vllm-transformers-modelling-backend-tutorial/blob/main/notebooks/lesson_1_transformers.ipynb) - covers `transformes serve`, `from_pretrained` and `generate_batch`
> 2.  [vLLM](https://github.com/hmellor/vllm-transformers-modelling-backend-tutorial/blob/main/notebooks/lesson_2_vllm.ipynb) - covers `llm serve` and the `LLM` class.
> 3.  [transformers backend](https://github.com/hmellor/vllm-transformers-modelling-backend-tutorial/blob/main/notebooks/lesson_3_transformers_backend.ipynb) how vLLm can run a transformers model implementation without reimplementing it from scratch.
>     - c.f. [Building a compatible model backend for inference](https://huggingface.co/docs/transformers/en/community_integrations/transformers_as_backend)
> 4.  [Bring your own transformers model](https://github.com/hmellor/vllm-transformers-modelling-backend-tutorial/blob/main/notebooks/lesson_4_bring_your_model.ipynb) to vLLM
>     - When making a model compatible with the Transformers backend, watch out for:
>       - **Missing** kwargs at any level\*\* — The most common issue. If OlmoeModel.forward accepted \*\*kwargs but OlmoeDecoderLayer.forward didn’t, attention_instances would be silently dropped.
>       - **Custom attention** not using ALL_ATTENTION_FUNCTIONS — Models that compute attention inline can’t be dispatched to vLLM’s kernels. The model must use the standard dispatch pattern.
>       - **Incorrect TP plans** — Misspecifying “colwise” vs “rowwise” will produce wrong results silently. Remember: projections that increase dimension (Q, K, V, gate, up) are typically “colwise”, and projections that decrease dimension (O, down) are “rowwise”.
>       - **Non-standard attention mask handling** — Models that manipulate attention weights directly (e.g., adding positional bias to attention scores after softmax) may not be expressible through the standard attention interface.

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2026,
  author = {Bochman, Oren},
  title = {vLLM with the {Transformers} {Modelling} {Backend}},
  date = {2026-04-28},
  url = {https://orenbochman.github.io/posts/2026/04-28-ODSC-AI-2026-Day-1/talk3.html},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2026. “vLLM with the Transformers Modelling Backend.” April 28. <https://orenbochman.github.io/posts/2026/04-28-ODSC-AI-2026-Day-1/talk3.html>.
