# An error occurred.

Unable to execute JavaScript.

Session Video

## SCROLLS: Standardized CompaRison Over Long Language Sequences

## Paper

[Standardized CompaRison Over Long Language Sequences SCROLLS](https://arxiv.org/abs/2201.03533)

## Abstract

NLP benchmarks have largely focused on short texts, such as sentences and paragraphs, even though long texts comprise a considerable amount of natural language in the wild. We introduce SCROLLS, a suite of tasks that require reasoning over long texts. We examine existing long-text datasets, and handpick ones where the text is naturally long, while prioritizing tasks that involve synthesizing information across the input. SCROLLS contains summarization, question answering, and natural language inference tasks, covering multiple domains, including literature, science, business, and entertainment. Initial baselines, including Longformer Encoder-Decoder, indicate that there is ample room for improvement on SCROLLS. We make all datasets available in a unified text-to-text format and host a live leaderboard to facilitate research on model architecture and pertaining methods.

## Speaker

- [Uri Shaham](https://www.linkedin.com/in/uri-shaham/) [Uri_Shaham](https://twitter.com/Uri_Shaham) [Page](https://urisha.github.io/) - PhD candidate in Tel Aviv university,
- Uri is a Ph.D. student at the Tel Aviv University NLP lab, working with Omer Levy. His research focuses on conditional language generation, involving model architectures, inference algorithms, and evaluation benchmarks.

## Slides

[![SCROLLS](session1/Screenshot%20from%202023-03-08%2017-24-23.png)](session1/Screenshot%20from%202023-03-08%2017-24-23.png "SCROLLS")

SCROLLS

[![SOTA in NLU](session1/Screenshot%20from%202023-03-08%2017-25-15.png)](session1/Screenshot%20from%202023-03-08%2017-25-15.png "SOTA in NLU")

SOTA in NLU

[![Problem - Transformers](session1/Screenshot%20from%202023-03-08%2017-25-54.png)](session1/Screenshot%20from%202023-03-08%2017-25-54.png "Problem - Transformers")

Problem - Transformers

[![Problem - Solutions](session1/Screenshot%20from%202023-03-08%2017-27-03.png)](session1/Screenshot%20from%202023-03-08%2017-27-03.png "Problem - Solutions")

Problem - Solutions

[![Evaluation on long texts](session1/Screenshot%20from%202023-03-08%2017-28-04.png)](session1/Screenshot%20from%202023-03-08%2017-28-04.png "Evaluation on long texts")

Evaluation on long texts

[![Can we do better?](session1/Screenshot%20from%202023-03-08%2017-29-24.png)](session1/Screenshot%20from%202023-03-08%2017-29-24.png "Can we do better?")

Can we do better?

- Preplexity of next token prediction
- Urvashi Khandelwal, He He, Peng Qi, and Dan Jurafsky. 2018. Sharp nearby, fuzzy far away: How neural language models use context. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 284–294, Melbourne, Australia. Association for Computational Linguistics.
- Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. Generalization through Memorization: Nearest Neighbor Language Models. In International Conference on Learning Representations (ICLR), 2020b
- Simeng Sun, Kalpesh Krishna, Andrew MattarellaMicke, and Mohit Iyyer. 2021. Do long-range language models actually use long-range context? In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 807–822, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.
- Ofir Press, Noah A. Smith, and Mike Lewis. 2021a. Shortformer: Better language modeling using shorter inputs. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 5493–5505, Online. Association for Computational Linguistics.

[![SCROLLS](session1/Screenshot%20from%202023-03-08%2017-29-36.png)](session1/Screenshot%20from%202023-03-08%2017-29-36.png "SCROLLS")

SCROLLS

[![Building SCROLS 1](session1/Screenshot%20from%202023-03-08%2017-30-54.png)](session1/Screenshot%20from%202023-03-08%2017-30-54.png "Building SCROLS 1")

Building SCROLS 1

[![Building SCROLS 2](session1/Screenshot%20from%202023-03-08%2017-31-24.png)](session1/Screenshot%20from%202023-03-08%2017-31-24.png "Building SCROLS 2")

Building SCROLS 2

[![Desiderata](session1/Screenshot%20from%202023-03-08%2017-31-32.png)](session1/Screenshot%20from%202023-03-08%2017-31-32.png "Desiderata")

Desiderata

[![Tasks](session1/Screenshot%20from%202023-03-08%2017-31-55.png)](session1/Screenshot%20from%202023-03-08%2017-31-55.png "Tasks")

Tasks

[![Example Q&A](session1/Screenshot%20from%202023-03-08%2017-32-52.png)](session1/Screenshot%20from%202023-03-08%2017-32-52.png "Example Q&A")

Example Q&A

[![delete](session1/Screenshot%20from%202023-03-08%2017-33-56.png)](session1/Screenshot%20from%202023-03-08%2017-33-56.png "delete")

delete

[![Examples require long-range reasoning](session1/Screenshot%20from%202023-03-08%2017-34-36.png)](session1/Screenshot%20from%202023-03-08%2017-34-36.png "Examples require long-range reasoning")

Examples require long-range reasoning

[![Does Context improve performance](session1/Screenshot%20from%202023-03-08%2017-34-38.png)](session1/Screenshot%20from%202023-03-08%2017-34-38.png "Does Context improve performance")

Does Context improve performance

[![Processing the entire input helps](session1/Screenshot%20from%202023-03-08%2017-35-07.png)](session1/Screenshot%20from%202023-03-08%2017-35-07.png "Processing the entire input helps")

Processing the entire input helps

[![Analysis](session1/Screenshot%20from%202023-03-08%2017-35-34.png)](session1/Screenshot%20from%202023-03-08%2017-35-34.png "Analysis")

Analysis

[![Does More context improve performance](session1/Screenshot%20from%202023-03-08%2017-37-36.png)](session1/Screenshot%20from%202023-03-08%2017-37-36.png "Does More context improve performance")

Does More context improve performance

[![Language understanding is crucial](session1/Screenshot%20from%202023-03-08%2017-46-58.png)](session1/Screenshot%20from%202023-03-08%2017-46-58.png "Language understanding is crucial")

Language understanding is crucial

[![Is more context all you need?](session1/Screenshot%20from%202023-03-08%2017-49-54.png)](session1/Screenshot%20from%202023-03-08%2017-49-54.png "Is more context all you need?")

Is more context all you need?

[![Is more context all you need?](session1/Screenshot%20from%202023-03-08%2017-50-52.png)](session1/Screenshot%20from%202023-03-08%2017-50-52.png "Is more context all you need?")

Is more context all you need?

[![How far is SCROLLS from being solved](session1/Screenshot%20from%202023-03-08%2017-50-59.png)](session1/Screenshot%20from%202023-03-08%2017-50-59.png "How far is SCROLLS from being solved")

How far is SCROLLS from being solved

[![Big room for improvement?](session1/Screenshot%20from%202023-03-08%2017-51-23.png)](session1/Screenshot%20from%202023-03-08%2017-51-23.png "Big room for improvement?")

Big room for improvement?

[![Leaderboard](session1/Screenshot%20from%202023-03-08%2017-51-44.png)](session1/Screenshot%20from%202023-03-08%2017-51-44.png "Leaderboard")

Leaderboard

[![Conclusions](session1/Screenshot%20from%202023-03-08%2017-52-37.png)](session1/Screenshot%20from%202023-03-08%2017-52-37.png "Conclusions")

Conclusions

[![Leaderboard](session1/Screenshot%20from%202023-03-08%2017-55-47.png)](session1/Screenshot%20from%202023-03-08%2017-55-47.png "Leaderboard")

Leaderboard

## Notes

- Few comments about this talk.
- Met with a company that worked on patents and had lots of issues with long range.
- Most of the points raised were ‘straw men’ so there is not much surprise.

# Efficient Long-Text Understanding with Short-Text Models

## Paper

[Efficient Long-Text Understanding with Short-Text Models](https://arxiv.org/abs/2208.00748)

## Abstract:

Transformer-based pretrained language models (LMs) are ubiquitous across natural language understanding, but cannot be applied to long sequences such as stories, scientific articles and long documents, due to their quadratic complexity. While a myriad of efficient transformer variants have been proposed, they are typically based on custom implementations that require expensive pretraining from scratch. In this work, we propose SLED: SLiding-Encoder and Decoder, a simple approach for processing long sequences that re-uses and leverages battle-tested short-text pretrained LMs. We find that SLED is competitive with specialized models that are up to 50x larger and require a dedicated and expensive pretraining step.

## Speaker

- \[Maor Ivgi\]
  - PhD candidate in Tel Aviv university
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
  - Not applicable for decoder-only architecture
  - (Corrective) pre-training is expected to help

[![Takeaways](session2/Screenshot%20from%202023-03-08%2018-43-51.png)](session2/Screenshot%20from%202023-03-08%2018-43-51.png "Takeaways")

Takeaways

- Takeaways
  - Individual pieces of information are localized
  - Fusion in decoder works
  - SLED does well on long range tasks.

[![Questions](session2/Screenshot%20from%202023-03-08%2018-44-07.png)](session2/Screenshot%20from%202023-03-08%2018-44-07.png "Questions")

Questions

- Main points They point out that the encoder can usually do a adequate job of understanding the input by looking at local context. Mostly a window with a few surrounding sentences. It uses this to create encode the input into a compact representation we call the state. The decoder will then be leverage the compression with “adequate” encodings to efficiently retrieve results from much longer contexts during inference on different tasks.

## An Overview of Modern Speech Recognition

### Abstract

Automatic speech recognition has been impacted by advances in related fields like image processing and natural language processing in recent years. One notable achievement in these areas has been the use of self-supervised learning to improve performance in computer vision and NLP tasks. This led to the development of the first self-supervised language model for speech representations, which has demonstrated impressive results in various NLP tasks. In this talk, we will review the key principles of automatic speech recognition and discuss the current progress, research, and challenges in the field

### Speaker

- Gal Hever
  - Algorithm Developer, Vision Map
  - MSc in Data Science, with over a decade of accumulated expertise in Machine Learning & Data Analytics from 8200, academy, and industry. Deploying algorithms to production by applying data-driven Machine Learning & AI solutions end to end, starting from research to development and testing.

### Slides

[![Overview](session3/Screenshot%20from%202023-03-09%2012-06-33.png)](session3/Screenshot%20from%202023-03-09%2012-06-33.png "Overview")

Overview

[![Conversational AI](session3/Screenshot%20from%202023-03-09%2012-26-06.png)](session3/Screenshot%20from%202023-03-09%2012-26-06.png "Conversational AI")

Conversational AI

[![ASR](session3/Screenshot%20from%202023-03-09%2012-26-15.png)](session3/Screenshot%20from%202023-03-09%2012-26-15.png "ASR")

ASR

[![ASR input challanges](session3/Screenshot%20from%202023-03-09%2012-26-34.png)](session3/Screenshot%20from%202023-03-09%2012-26-34.png "ASR input challanges")

ASR input challanges

[![Signal & Noise](session3/Screenshot%20from%202023-03-09%2012-26-36.png)](session3/Screenshot%20from%202023-03-09%2012-26-36.png "Signal & Noise")

Signal & Noise

[![Ideal System](session3/Screenshot%20from%202023-03-09%2012-26-40.png)](session3/Screenshot%20from%202023-03-09%2012-26-40.png "Ideal System")

Ideal System

[![ASR Task](session3/Screenshot%20from%202023-03-09%2012-27-02.png)](session3/Screenshot%20from%202023-03-09%2012-27-02.png "ASR Task")

ASR Task

[![slide009](session3/Screenshot%20from%202023-03-09%2012-28-01.png)](session3/Screenshot%20from%202023-03-09%2012-28-01.png "slide009")

slide009

[![slide010](session3/Screenshot%20from%202023-03-09%2012-38-13.png)](session3/Screenshot%20from%202023-03-09%2012-38-13.png "slide010")

slide010

[![slide011](session3/Screenshot%20from%202023-03-09%2012-38-26.png)](session3/Screenshot%20from%202023-03-09%2012-38-26.png "slide011")

slide011

[![WER Metric](session3/Screenshot%20from%202023-03-09%2012-38-54.png)](session3/Screenshot%20from%202023-03-09%2012-38-54.png "WER Metric")

WER Metric

[![ASR History](session3/Screenshot%20from%202023-03-09%2012-39-17.png)](session3/Screenshot%20from%202023-03-09%2012-39-17.png "ASR History")

ASR History

[![ASR Time Line](session3/Screenshot%20from%202023-03-09%2012-42-19.png)](session3/Screenshot%20from%202023-03-09%2012-42-19.png "ASR Time Line")

ASR Time Line

[![Augumentations](session3/Screenshot%20from%202023-03-09%2012-43-37.png)](session3/Screenshot%20from%202023-03-09%2012-43-37.png "Augumentations")

Augumentations

[![WER we are 21](session3/Screenshot%20from%202023-03-09%2012-44-16.png)](session3/Screenshot%20from%202023-03-09%2012-44-16.png "WER we are 21")

WER we are 21

[![WER we are 2](session3/Screenshot%20from%202023-03-09%2012-44-37.png)](session3/Screenshot%20from%202023-03-09%2012-44-37.png "WER we are 2")

WER we are 2

[![ASR challanges](session3/Screenshot%20from%202023-03-09%2012-45-14.png)](session3/Screenshot%20from%202023-03-09%2012-45-14.png "ASR challanges")

ASR challanges

[![diversity challange](session3/Screenshot%20from%202023-03-09%2012-46-00.png)](session3/Screenshot%20from%202023-03-09%2012-46-00.png "diversity challange")

diversity challange

[![language is dynamic](session3/Screenshot%20from%202023-03-09%2012-47-26.png)](session3/Screenshot%20from%202023-03-09%2012-47-26.png "language is dynamic")

language is dynamic

[![whar’s next](session3/Screenshot%20from%202023-03-09%2012-47-57.png)](session3/Screenshot%20from%202023-03-09%2012-47-57.png "whar’s next")

whar’s next

[![covid understanding challenges](session3/Screenshot%20from%202023-03-09%2012-49-03.png)](session3/Screenshot%20from%202023-03-09%2012-49-03.png "covid understanding challenges")

covid understanding challenges

[![Non verbal communication 1](session3/Screenshot%20from%202023-03-09%2012-49-21.png)](session3/Screenshot%20from%202023-03-09%2012-49-21.png "Non verbal communication 1")

Non verbal communication 1

[![Non verbal communication 2](session3/Screenshot%20from%202023-03-09%2012-49-57.png)](session3/Screenshot%20from%202023-03-09%2012-49-57.png "Non verbal communication 2")

Non verbal communication 2

[![DataNights Cohort](session3/Screenshot%20from%202023-03-09%2012-51-10.png)](session3/Screenshot%20from%202023-03-09%2012-51-10.png "DataNights Cohort")

DataNights Cohort

[![QR for ASR Course](session3/Screenshot%20from%202023-03-09%2012-51-53.png)](session3/Screenshot%20from%202023-03-09%2012-51-53.png "QR for ASR Course")

QR for ASR Course

[![Questions](session3/Screenshot%20from%202023-03-09%2012-52-15.png)](session3/Screenshot%20from%202023-03-09%2012-52-15.png "Questions")

Questions

### Reflections

- I’ve read a couple of books on the subject, but this shows more up to date results.
- Show me the papers?
- The Data Nights course should be worth taking

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2014,
  author = {Bochman, Oren},
  title = {SCROLLS - {Standardized} {CompaRison} {Over} {Long}
    {Language} {Sequences}},
  date = {2014-11-01},
  url = {https://orenbochman.github.io/posts/2023/01-11-nlp-il-meetup-intuit/talk1.html},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2014. “SCROLLS - Standardized CompaRison Over Long Language Sequences.” November 1. <https://orenbochman.github.io/posts/2023/01-11-nlp-il-meetup-intuit/talk1.html>.
