# An error occurred.

Unable to execute JavaScript.

Session Video

# Efficient Long-Text Understanding with Short-Text Models

## Paper

- [Efficient Long-Text Understanding with Short-Text Models](https://arxiv.org/abs/2208.00748)

## Abstract:

Transformer-based pretrained language models (LMs) are ubiquitous across natural language understanding, but cannot be applied to long sequences such as stories, scientific articles and long documents, due to their quadratic complexity. While a myriad of efficient transformer variants have been proposed, they are typically based on custom implementations that require expensive pre-training from scratch. In this work, we propose SLED: SLiding-Encoder and Decoder, a simple approach for processing long sequences that re-uses and leverages battle-tested short-text pretrained LMs. We find that SLED is competitive with specialized models that are up to 50x larger and require a dedicated and expensive pre-training step.

## Speaker

- Maor Ivgi
  - PhD candidate in Tel Aviv university,
  - Maor is an NLP researcher and entrepreneur. He has vast experience in implementing state-of-the-art deep learning models for real-world use cases. He received his masters in Computer Science at Tel-Aviv University advised by Prof. Jonathan Berant, focusing on NLP models’ Robustness. As a Ph.D. candidate at Prof. Berant’s lab, his research is focused on long-range reasoning in large language models.

## Slides

[![Efficient Long-Text Understanding with Short-Text Models](session2/Screenshot%20from%202023-03-08%2018-03-48.png)](session2/Screenshot%20from%202023-03-08%2018-03-48.png "Efficient Long-Text Understanding with Short-Text Models")

Efficient Long-Text Understanding with Short-Text Models

[![NLP Papers](session2/Screenshot%20from%202023-03-08%2018-04-11.png)](session2/Screenshot%20from%202023-03-08%2018-04-11.png "NLP Papers")

NLP Papers

- NLP seems to have reached new level of maturity for use in Industry
  - c.f. Attention is all you need
  - c.f. BERT pre-training of deep bidirectional transformers for language understanding

[![Model Timeline](session2/Screenshot%20from%202023-03-08%2018-04-33.png)](session2/Screenshot%20from%202023-03-08%2018-04-33.png "Model Timeline")

Model Timeline

[![Q&A challenges](session2/Screenshot%20from%202023-03-08%2018-08-53.png)](session2/Screenshot%20from%202023-03-08%2018-08-53.png "Q&A challenges")

Q&A challenges

[![Transformers - Good on short text NLU](session2/Screenshot%20from%202023-03-08%2018-09-02.png)](session2/Screenshot%20from%202023-03-08%2018-09-02.png "Transformers - Good on short text NLU")

Transformers - Good on short text NLU

[![Long Text NLU Fail](session2/Screenshot%20from%202023-03-08%2018-09-05.png)](session2/Screenshot%20from%202023-03-08%2018-09-05.png "Long Text NLU Fail")

Long Text NLU Fail

[![Transformers Quadratic dependency limits](session2/Screenshot%20from%202023-03-08%2018-09-16.png)](session2/Screenshot%20from%202023-03-08%2018-09-16.png "Transformers Quadratic dependency limits")

Transformers Quadratic dependency limits

[![Transformers Attention complexity](session2/Screenshot%20from%202023-03-08%2018-12-44.png)](session2/Screenshot%20from%202023-03-08%2018-12-44.png "Transformers Attention complexity")

Transformers Attention complexity

- Transformers have issues with long texts:
  - self attention is O(n^2)
  - cross attention is O(nk) ![Novel Transformer Architecture Papers](session2/Screenshot%20from%202023-03-08%2018-13-05.png)
- Efficient LLM papers are:
- Hard to understand,
- Hard to generalize (due to platform specific engineering tricks)
- Expensive to reproduce
- Inference run into Memory is an issue
- Training is often on beginning of document so does not see the end
- Self Attention is has a limited window size.

[![SLED - Locality](session2/Screenshot%20from%202023-03-08%2018-16-04.png)](session2/Screenshot%20from%202023-03-08%2018-16-04.png "SLED - Locality")

SLED - Locality

[![SLED - Properties](session2/Screenshot%20from%202023-03-08%2018-17-52.png)](session2/Screenshot%20from%202023-03-08%2018-17-52.png "SLED - Properties")

SLED - Properties

- SLED’s Approach
  - Assume locality of information: “In an encoder-decoder architecture, the encoder can effectively contextualize input tokens with local context only, leaving long range dependency to be handled by the decoder.”
  - Split text into short fixed length overlapping chunks of text (short contexts).
  - Prepend the `prefix/prompt` to each chunk
  - The decoder will need to put it all together.

[![SLED Properties](session2/Screenshot%20from%202023-03-08%2018-18-43.png)](session2/Screenshot%20from%202023-03-08%2018-18-43.png "SLED Properties")

SLED Properties

[![Model Size effect](session2/Screenshot%20from%202023-03-08%2018-20-47.png)](session2/Screenshot%20from%202023-03-08%2018-20-47.png "Model Size effect")

Model Size effect

[![SLED Performance Boost](session2/Screenshot%20from%202023-03-08%2018-23-55.png)](session2/Screenshot%20from%202023-03-08%2018-23-55.png "SLED Performance Boost")

SLED Performance Boost

[![SLED is Competitive with short text models](session2/Screenshot%20from%202023-03-08%2018-24-34.png)](session2/Screenshot%20from%202023-03-08%2018-24-34.png "SLED is Competitive with short text models")

SLED is Competitive with short text models

[![Analysis](session2/Screenshot%20from%202023-03-08%2018-25-03.png)](session2/Screenshot%20from%202023-03-08%2018-25-03.png "Analysis")

Analysis

- this is a great slide!
- it summarizes lots of info
- SLED’s Analysis
  - Contextual encoding is crucial
  - Cheating is not enough
  - The is real benefit in fusion ![Finding a Needle in a Haystack](session2/Screenshot%20from%202023-03-08%2018-25-30.png)

[![Finding a Needle perfectly](session2/Screenshot%20from%202023-03-08%2018-27-07.png)](session2/Screenshot%20from%202023-03-08%2018-27-07.png "Finding a Needle perfectly")

Finding a Needle perfectly

[![Fusing Information Pieces](session2/Screenshot%20from%202023-03-08%2018-29-19.png)](session2/Screenshot%20from%202023-03-08%2018-29-19.png "Fusing Information Pieces")

Fusing Information Pieces

- what is Cheating?

[![Cheating is not enough](session2/Screenshot%20from%202023-03-08%2018-31-20.png)](session2/Screenshot%20from%202023-03-08%2018-31-20.png "Cheating is not enough")

Cheating is not enough

- Quantifying SLED’s benefits using relative improvement.

\text{Relative Improvement} = \frac{Score(SLED)-Score(Bart)}{Score(Bart)}

[![Gains Formula from longer inputs Gains](session2/Screenshot%20from%202023-03-08%2018-32-48.png)](session2/Screenshot%20from%202023-03-08%2018-32-48.png "Gains Formula from longer inputs Gains")

Gains Formula from longer inputs Gains

[![Chart of longer inputs Gains](session2/Screenshot%20from%202023-03-08%2018-33-45.png)](session2/Screenshot%20from%202023-03-08%2018-33-45.png "Chart of longer inputs Gains")

Chart of longer inputs Gains

[![Limitations & Future Work](session2/Screenshot%20from%202023-03-08%2018-43-21.png)](session2/Screenshot%20from%202023-03-08%2018-43-21.png "Limitations & Future Work")

Limitations & Future Work

- Limits & Future Work
  - Long outputs are still a constraint
  - No explicit global contextualization
  - No explicit global positional information
  - No applicable for decoder-only architecture
  - (Corrective) pre-training is expected to help

[![Takeways](session2/Screenshot%20from%202023-03-08%2018-43-51.png)](session2/Screenshot%20from%202023-03-08%2018-43-51.png "Takeways")

Takeways

- Takeaways
  - Individual pieces of information are localized
  - Fusioin in decoder works
  - SLED does well on long range tasks.

[![Questions](session2/Screenshot%20from%202023-03-08%2018-44-07.png)](session2/Screenshot%20from%202023-03-08%2018-44-07.png "Questions")

Questions

- Main points

They point out that the encoder can usually do a adequate job of understanding the input by looking at local context. Mostly a window with a few surrounding sentences. It uses this to create encode the input into a compact representation we call the state. The decoder will then be leverage the compression with “adaquate” encodings to efficently retrieve results from much longer contexts during inference on different tasks.

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2015,
  author = {Bochman, Oren},
  title = {Efficient {Long-Text} {Understanding} with {Short-Text}
    {Models}},
  date = {2015-11-01},
  url = {https://orenbochman.github.io/posts/2023/01-11-nlp-il-meetup-intuit/talk2.html},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2015. “Efficient Long-Text Understanding with Short-Text Models.” November 1. <https://orenbochman.github.io/posts/2023/01-11-nlp-il-meetup-intuit/talk2.html>.
