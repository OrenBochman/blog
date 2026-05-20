[![pydata global](pydata_logo.png)](pydata_logo.png "pydata global")

pydata global

> **TIP:**
>
> Discover how Insee (French National Statistics Institute) transitioned from fastText to a PyTorch-based model for text classification by developing and open-sourcing the torchTextClassifiers python package. This presentation will cover the creation, deployment, and practical applications of torchTextClassifiers in modernizing automatic coding systems, benefiting Insee and other European National Statistical Institutes (NSIs).

> **TIP:**
>
> - The rationale for moving from fastText to a PyTorch-based model​ in production
> - Packaging a PyTorch-based model architecture and open-source collaboration
> - Key features and architecture of torchTextClassifiers ​
> - Deployment strategies within a public administration (MLOps, cloud native tools, security)
> - Lessons learned and best practices for similar transitions​

Insee, France’s National Institute of Statistics and Economic Studies, has long relied on fastText for automatic coding tasks. Recognizing the need to modernize and future-proof this critical functionality, we developed torchTextClassifiers — an open-source Python package that enables easy training and deployment of a PyTorch-based model for text classification, paving the way for further innovation in this domain.

This session will delve into the motivations behind replacing the archived fastText package, the design and implementation of torchTextClassifiers , and its integration into Insee’s production environment. We’ll discuss the challenges faced during this transition, including model compatibility, performance optimization, and user adoption.​

> **TIP:**
>
> ### Cédric Couralet
>
> Cédric Couralet, Data Scientist at Insee, is an open-source enthusiast, with expertise in software architecture and secure system design.
>
> ### Meilame Tayebjee
>
> As a Data Scientist at the Innovation Lab of the French National Institute of Statistics and Economic Studies (Insee), I focus on the deployment of machine learning models, the enhancement of MLOps best practices, and the development of torchTextClassifiers, a PyTorch package designed to streamline the training of deep learning models for text classification.
>
> I am also pursuing a PhD in Computer Science jointly at CREST and Inria, where my research centers on foundational Transformer-based models for the analysis of healthcare pathways.

## Outline:

- Introduction to hallucinations in LLMs
- Common causes behind hallucinated outputs
- Impact on production applications
- Techniques for detecting and evaluating hallucinations
- Strategies to reduce hallucinations
- Best practices for building trustworthy AI products
- Key takeaways

------------------------------------------------------------------------

[slidedeck](https://inseefrlab.github.io/codif-ape-prez/slides/pydata-global/#/title-slide)

[![Slide 01 - What is this talk](slide01.png)](slide01.png "Slide 01 - What is this talk")

Slide 01 - What is this talk

This is a classifier for documents in terms of economic activity (NAF codes) based on their textual description.

[![Slide 02 - Use case 1](slide02.png)](slide02.png "Slide 02 - Use case 1")

Slide 02 - Use case 1

[![Slide 03 - Use case 2](slide03.png)](slide03.png "Slide 03 - Use case 2")

Slide 03 - Use case 2

[![Slide 04 - Ideal MLOps pipeline](slide04.png)](slide04.png "Slide 04 - Ideal MLOps pipeline")

Slide 04 - Ideal MLOps pipeline

[![Slide 05 - v.s. ours](slide05.png)](slide05.png "Slide 05 - v.s. ours")

Slide 05 - v.s. ours

[![Slide 06 - MLFLow](slide06.png)](slide06.png "Slide 06 - MLFLow")

Slide 06 - MLFLow

[![Slide 07 - The wrapper (code)](slide07.png)](slide07.png "Slide 07 - The wrapper (code)")

Slide 07 - The wrapper (code)

[![Slide 08 - Api serving](slide08.png)](slide08.png "Slide 08 - Api serving")

Slide 08 - Api serving

[![Slide 09 - Code](slide09.png)](slide09.png "Slide 09 - Code")

Slide 09 - Code

[![Slide 10 - Api serving](slide10.png)](slide10.png "Slide 10 - Api serving")

Slide 10 - Api serving

[![Slide 11 - Beyond fastText](slide11.png)](slide11.png "Slide 11 - Beyond fastText")

Slide 11 - Beyond fastText

[![Slide 12 - Why pytorch](slide12.png)](slide12.png "Slide 12 - Why pytorch")

Slide 12 - Why pytorch

[![Slide 13 - Architecture motivation](slide13.png)](slide13.png "Slide 13 - Architecture motivation")

Slide 13 - Architecture motivation

[![Slide 15 - torchTextClassifiers](slide15.png)](slide15.png "Slide 15 - torchTextClassifiers")

Slide 15 - torchTextClassifiers

https://github.com/InseeFrLab/torchTextClassifiers

[![Slide 16 - Targets](slide16.png)](slide16.png "Slide 16 - Targets")

Slide 16 - Targets

[![Slide 17 - Positioning](slide17.png)](slide17.png "Slide 17 - Positioning")

Slide 17 - Positioning

[![Slide 18 - components](slide18.png)](slide18.png "Slide 18 - components")

Slide 18 - components

[Demo](https://inseefrlab.github.io/torchTextClassifiers/notebooks/example.html)

[![Slide 19 - MLOps](slide19.png)](slide19.png "Slide 19 - MLOps")

Slide 19 - MLOps

[![Slide 20 - Roadmap](slide20.png)](slide20.png "Slide 20 - Roadmap")

Slide 20 - Roadmap

[![Slide 21 - Thank you](slide21.png)](slide21.png "Slide 21 - Thank you")

Slide 21 - Thank you

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2025,
  author = {Bochman, Oren},
  title = {torchTextClassifiers : {Modernizing} {Text} Classification
    for {French} {National} {Statistics}},
  date = {2025-12-09},
  url = {https://orenbochman.github.io/posts/2025/2025-12-09-pydata-torchTextClassifiers/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2025. “torchTextClassifiers : Modernizing Text Classification for French National Statistics.” December 9. <https://orenbochman.github.io/posts/2025/2025-12-09-pydata-torchTextClassifiers/>.
