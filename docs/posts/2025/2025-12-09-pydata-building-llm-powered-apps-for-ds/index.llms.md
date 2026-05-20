[![pydata global](pydata_logo.png)](pydata_logo.png "pydata global")

pydata global

# An error occurred.

Unable to execute JavaScript.

> **TIP:**
>
> This workshop is designed to equip software engineers with the skills to build and iterate on generative AI-powered applications. Participants will explore key components of the AI software development lifecycle through first principles thinking, including prompt engineering, monitoring, evaluations, and handling non-determinism. The session focuses on using LLMs to build applications, such as querying PDFs, while providing insights into the engineering challenges unique to AI systems. By the end of the workshop, participants will know how to build a PDF-querying app, but all techniques learned will be generalizable for building a variety of generative AI applications.
>
> If you’re a data scientist, machine learning practitioner, or AI enthusiast, this workshop can also be valuable for learning about the software engineering aspects of AI applications, such as lifecycle management, iterative development, and monitoring, which are critical for production-level AI systems.

> **TIP:**
>
> - How to integrate AI models and APIs into a practical application.
> - Techniques to manage non-determinism and optimize outputs through prompt engineering.
> - How to monitor, log, and evaluate AI systems to ensure reliability.
> - The importance of handling structured outputs and using function calling in AI models.
> - The software engineering side of building AI systems, including iterative development, debugging, and performance monitoring.
> - Practical experience in building an app to query PDFs using multimodal models.

## What is Unique About This Session:

This workshop bridges the gap between software engineering and generative AI development. While most AI workshops focus solely on model usage or tuning, this session emphasizes the entire AI software lifecycle — from prompt engineering to monitoring and tracing. Participants will learn how to manage non-determinism and create production-ready AI applications, giving them the knowledge to tackle the software engineering challenges of AI-powered apps. The hands-on approach ensures that attendees walk away with practical skills and a functional app.

> **TIP:**
>
> - Basic programming knowledge in Python.
> - Familiarity with REST APIs.
> - Experience working with Jupyter Notebooks or similar environments (preferred but not required).
> - No prior experience with AI or machine learning is required.
> - Most importantly, a sense of curiosity and a desire to learn!
> - If you have a background in data science, ML, or AI, this workshop will help you understand the software engineering side of building AI applications.

## Tools and Frameworks:

We will introduce you to certain modern frameworks in the workshop but the emphasis be on first principles and using vanilla Python and LLM calls to build AI-powered systems.

[workshop repo](https://github.com/hugobowne/AI-for-SWEs)

> **TIP:**

------------------------------------------------------------------------

[![slide 01 - About the Workshop](slide01.png)](slide01.png "slide 01 - About the Workshop")

slide 01 - About the Workshop

[![Slide 02 - Session Flow](slide02.png)](slide02.png "Slide 02 - Session Flow")

Slide 02 - Session Flow

[![Slide 03 - Chat GPT](slide03-chatgpt.png)](slide03-chatgpt.png "Slide 03 - Chat GPT")

Slide 03 - Chat GPT

[![Slide 04 - Chat with Claude](slide04-claude.png)](slide04-claude.png "Slide 04 - Chat with Claude")

Slide 04 - Chat with Claude

[![Slide 05 - Session Flow](slide05.png)](slide05.png "Slide 05 - Session Flow")

Slide 05 - Session Flow

[![Slide 06 - Action](slide06.png)](slide06.png "Slide 06 - Action")

Slide 06 - Action

[![Slide 07 - What can an LLM do?](slide07.png)](slide07.png "Slide 07 - What can an LLM do?")

Slide 07 - What can an LLM do?

[![Slide 08 - Goals](slide08.png)](slide08.png "Slide 08 - Goals")

Slide 08 - Goals

[![Slide 09 - Agmented LLM (anthropic)](slide09.png)](slide09.png "Slide 09 - Agmented LLM (anthropic)")

Slide 09 - Agmented LLM (anthropic)

c.f. [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)

[![Slide 10 - AI POC](slide10.png)](slide10.png "Slide 10 - AI POC")

Slide 10 - AI POC

[![Slide 11 - 5 line Rag](slide11.png)](slide11.png "Slide 11 - 5 line Rag")

Slide 11 - 5 line Rag

[![Slide 12 - output](slide12.png)](slide12.png "Slide 12 - output")

Slide 12 - output

[![Slide 13 - How To Improve](slide13.png)](slide13.png "Slide 13 - How To Improve")

Slide 13 - How To Improve

[![Slide 14 - Show me the prompt](slide14.png)](slide14.png "Slide 14 - Show me the prompt")

Slide 14 - Show me the prompt

------------------------------------------------------------------------

> **TIP:**
>
>     - can I use this as a free resource? yes!
>     - can I use this to edit my repo from my IPAD? yes!

------------------------------------------------------------------------

## First Demo

1.  Don’t follow along in real time, just focus on the concepts.
2.  Follow the README.md in the repo.
3.  The first notebook is about using the code above to build a simple RAG system that queries different LLM (Claude, ChatGPT and Gemini). against some PDF document~~s~~.

------------------------------------------------------------------------

[![Slide 15](slide15.png)](slide15.png "Slide 15")

Slide 15

[![Slide 16](slide16.png)](slide16.png "Slide 16")

Slide 16

[![Slide 17](slide17.png)](slide17.png "Slide 17")

Slide 17

[![Slide 18](slide18.png)](slide18.png "Slide 18")

Slide 18

[![Slide 19](slide19.png)](slide19.png "Slide 19")

Slide 19

------------------------------------------------------------------------

> **TIP:**
>
> has a free tier.

> **TIP:**
>
> has a free tier.

------------------------------------------------------------------------

## Second Demo :

`3-vannila-python-query.py`

1.  
2.  Rebuild the front end in Gradio
3.  Add monitoring and logging (observability)

[![Slide 20](slide20.png)](slide20.png "Slide 20")

Slide 20

[![Slide 21](slide21.png)](slide21.png "Slide 21")

Slide 21

[![Slide 22](slide22.png)](slide22.png "Slide 22")

Slide 22

[![Slide 23](slide23.png)](slide23.png "Slide 23")

Slide 23

[![Slide 24](slide24.png)](slide24.png "Slide 24")

Slide 24

llms are

[![Slide 25 - Multimodel Session Ad](slide25.png)](slide25.png "Slide 25 - Multimodel Session Ad")

Slide 25 - Multimodel Session Ad

------------------------------------------------------------------------

[![Slide 30](slide30.png)](slide30.png "Slide 30")

Slide 30

[![Slide 31](slide31.png)](slide31.png "Slide 31")

Slide 31

[![Slide 32](slide32.png)](slide32.png "Slide 32")

Slide 32

[![Slide 33](slide33.png)](slide33.png "Slide 33")

Slide 33

[![Slide 34](slide34.png)](slide34.png "Slide 34")

Slide 34

[![Slide 35](slide35.png)](slide35.png "Slide 35")

Slide 35

[![Slide 36](slide36.png)](slide36.png "Slide 36")

Slide 36

Key: Align LLM outputs to your application needs

[![Slide 37](slide37.png)](slide37.png "Slide 37")

Slide 37

recommends use a spread sheet (slide comes from another workshop/talk)

------------------------------------------------------------------------

## Demo 3: UnStructured to Emailed Report (Two-Stage Pipeline)

- [Two-Stage AI Pipeline: From Unstructured Text to Personalized Email](https://github.com/hugobowne/genai-first-principles/blob/main/notebooks/NB-2.ipynb)
  1.  Setup (keys, imports, client)
  2.  Load LinkedIn data (via txt file)
  3.  Stage 1: Summarize LinkedIn posts \to JSON
      - Minimal Baseline
      - With Schema Definition, JSON Mode, and Error Handling
  4.  Stage 2: Structured Data \to Personalized Recruiter Email - Email Variation 1: Minimal (Baseline) - Email Variation 2: With Guardrails and Personalization Requirements - Complete Two-Stage Pipeline
  5.  LLM Judge (we don’t need a code check but a judgement call)

------------------------------------------------------------------------

[![Slide 38](slide38.png)](slide38.png "Slide 38")

Slide 38

[![Slide 39](slide39.png)](slide39.png "Slide 39")

Slide 39

- jumps back to … anthropic slide
  - do we need memory?
  - do we need tool use?
  - creating two command line tools like send email etc can cover most of our needs.

[![Slide 40](slide40.png)](slide40.png "Slide 40")

Slide 40

[![Slide 41 routing](slide41.png)](slide41.png "Slide 41 routing")

Slide 41 routing

[![Slide 42](slide42.png)](slide42.png "Slide 42")

Slide 42

[![Slide 43](slide43.png)](slide43.png "Slide 43")

Slide 43

[![Slide 44](slide44.png)](slide44.png "Slide 44")

Slide 44

[![Slide 45](slide45.png)](slide45.png "Slide 45")

Slide 45

## Demo 4: Function Calling

- [Function Calling with LLM APIs](https://github.com/hugobowne/genai-first-principles/blob/main/notebooks/NB-3.ipynb)
  - OpenAI Function Calling
  - Gemini Function Calling (not as clever yet, think about non-determinism)
  - **Enriching Data** with Search
    - there are lots of cool tools we can use here ([pinecone](https://www.pinecone.io/), [weaviate](https://weaviate.io/), [chromadb](https://www.chromadb.com/), etc)

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2025,
  author = {Bochman, Oren},
  title = {Building {LLM-Powered} {Applications} for {Data} {Scientists}
    and {Software} {Engineers}},
  date = {2025-12-09},
  url = {https://orenbochman.github.io/posts/2025/2025-12-09-pydata-building-llm-powered-apps-for-ds/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2025. “Building LLM-Powered Applications for Data Scientists and Software Engineers.” December 9. <https://orenbochman.github.io/posts/2025/2025-12-09-pydata-building-llm-powered-apps-for-ds/>.
