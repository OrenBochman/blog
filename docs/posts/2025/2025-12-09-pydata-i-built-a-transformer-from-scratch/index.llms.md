[![pydata global](pydata_logo.png)](pydata_logo.png "pydata global")

pydata global

# An error occurred.

Unable to execute JavaScript.

> **TIP:**
>
> Want to understand how transformers actually work without wading through 10,000 lines of framework code or drowning in tensor shapes?
>
> This talk walks you through building a transformer model from scratch — no pre-trained shortcuts, no black-box abstractions — just clean PyTorch code and good old-fashioned curiosity.
>
> You’ll walk away with a clearer mental model of how attention, encoders, decoders, and masking really work.

Transformers power modern large language models, but their inner workings are often buried under complex libraries and unreadable abstractions. In this talk, we’ll peel back the layers and build the original Transformer architecture (Vaswani et al., 2017) step by step in PyTorch, from input embeddings to attention masks to the full encoder-decoder stack.

This talk is designed for attendees with a basic understanding of deep learning and [PyTorch](https://pytorch.org/) who want to go beyond surface-level blog posts and get a hands-on, conceptual grasp of what happens under the hood. You’ll see how each part of the transformer connects back to the equations in the original paper, how to debug common implementation pitfalls, and how to avoid getting lost in tensor dimension hell.

> **TIP:**
>
> - 🔍 A walkthrough of key components: attention, positional encoding, encoder/decoder stack
> - 🧠 Visual explanations of attention masks, shapes, and residuals
> - ⚠️ Common bugs and debugging strategies (like handling shape mismatches and masking errors)
> - ✅ Real-world implementation tips and tricks that demystify the architecture
>
> By the end of the talk, attendees will:
>
> - Understand the full forward pass of a transformer
> - Know how each component connects to the original paper
> - Feel more confident reading or writing custom model architectures

> **TIP:**
>
> - Basic Python and PyTorch
> - Some familiarity with neural networks (e.g., feedforward, softmax)
> - No need for prior experience in building models from scratch

## Tools and Frameworks:

We will introduce you to certain modern frameworks in the workshop but the emphasis be on first principles and using vanilla Python and LLM calls to build AI-powered systems.

[workshop repo](https://huggingface.co/datasets/bird-of-paradise/transformer-from-scratch-tutorial)

> **TIP:**

## Outline

[![Title](slide01.png)](slide01.png "Title")

Title

[![Who I am & What I Do](slide02.png)](slide02.png "Who I am & What I Do")

Who I am & What I Do

[![Agenda](slide03.png)](slide03.png "Agenda")

Agenda

[![Transformer Architecture](slide04.png)](slide04.png "Transformer Architecture")

Transformer Architecture

[![Transformer Architecture - Key Modules](slide05.png)](slide05.png "Transformer Architecture - Key Modules")

Transformer Architecture - Key Modules

[![Positional Encoding and Embeddings](slide06.png)](slide06.png "Positional Encoding and Embeddings")

Positional Encoding and Embeddings

[![Scaled Dot-Product Attention](slide07.png)](slide07.png "Scaled Dot-Product Attention")

Scaled Dot-Product Attention

[![Multi-Head Attention](slide08.png)](slide08.png "Multi-Head Attention")

Multi-Head Attention

[![Attention Masks: Causal vs Padding Masks](slide09.png)](slide09.png "Attention Masks: Causal vs Padding Masks")

Attention Masks: Causal vs Padding Masks

[![Feed Forward Networks](slide10.png)](slide10.png "Feed Forward Networks")

Feed Forward Networks

[![Forward Pass - Decoder Only](slide11.png)](slide11.png "Forward Pass - Decoder Only")

Forward Pass - Decoder Only

[![Debugging Tips](slide12.png)](slide12.png "Debugging Tips")

Debugging Tips

[![Next Steps - Continue Learning](slide13.png)](slide13.png "Next Steps - Continue Learning")

Next Steps - Continue Learning

[![Thank You & Let’s Connect](slide14.png)](slide14.png "Thank You & Let’s Connect")

Thank You & Let’s Connect

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2025,
  author = {Bochman, Oren},
  title = {I {Built} a {Transformer} from {Scratch} {So} {You} {Don’t}
    {Have} {To}},
  date = {2025-12-09},
  url = {https://orenbochman.github.io/posts/2025/2025-12-09-pydata-i-built-a-transformer-from-scratch/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2025. “I Built a Transformer from Scratch So You Don’t Have To.” December 9. <https://orenbochman.github.io/posts/2025/2025-12-09-pydata-i-built-a-transformer-from-scratch/>.
