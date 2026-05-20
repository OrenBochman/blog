# An error occurred.

Unable to execute JavaScript.

## Responsible AI Mitigations and Tracker

- Speakers:
  - [Besmira Nushi](https://besmiranushi.com/#publications), Principal Researcher at Microsoft
  - [Rahee Ghosh Peshawaria](https://www.linkedin.com/in/rahee-ghosh-peshawaria/), Senior Technical Program Manager at Microsoft
- Abstract:
  - Responsible AI Toolbox is an open-source effort at Microsoft to accelerate and operationalize Responsible AI via a set of interoperable tools, libraries, and customizable dashboards. Recently, we released two new tools as part of the toolbox:
    - [Responsible AI Mitigations](https://github.com/microsoft/responsible-ai-toolbox/tree/main/responsibleai/mitigations): Python library for implementing and exploring mitigations for Responsible AI.
    - [Responsible AI Tracker](https://github.com/microsoft/responsible-ai-toolbox/tree/main/responsibleai/tracker): JupyterLab extension for tracking, managing, and comparing Responsible AI mitigations and experiments.
  - Both tools contribute to supporting a systematic and targeted process of model improvement by identifying, diagnosing, mitigating, and comparing failure modes.

During the presentation we will delve into the new tooling additions and will illustrate their functionalities through a hands-on demonstration. In particular, we will show how insights extracted through Responsible AI Dashboard through the identification and diagnosis stage can then be used to ideate and implement different mitigation techniques via the Responsible AI Mitigations library. Finally, we will also demonstrate how different mitigation techniques can be compared and validated through Responsible AI Tracker directly on Jupyter Lab.

![](Screenshot%20from%202023-03-08%2019-11-39.png) ![](Screenshot%20from%202023-03-08%2019-11-42.png) ![](Screenshot%20from%202023-03-08%2019-13-00.png) ![](Screenshot%20from%202023-03-08%2019-14-27.png) ![](Screenshot%20from%202023-03-08%2019-14-32.png) ![](Screenshot%20from%202023-03-08%2019-16-13.png) ![](Screenshot%20from%202023-03-08%2019-17-28.png) ![](Screenshot%20from%202023-03-08%2019-18-22.png) ![](Screenshot%20from%202023-03-08%2019-18-46.png)

## Demo

[![](Screenshot%20from%202023-03-08%2019-21-51.png)](Screenshot%20from%202023-03-08%2019-21-51.png)

## Useful links

- [Blog](https://www.microsoft.com/en-us/research/articles/responsible-ai-mitigations-and-tracker-new-open-source-tools-for-guiding-mitigations-in-responsible-ai-2/)
- [Responsible AI Toolbox](https://github.com/microsoft/responsible-ai-toolbox)
- [Responsible AI Mitigations Repo](https://github.com/microsoft/responsible-ai-toolbox-mitigations)
- [Responsible AI Tracker Repo](https://github.com/microsoft/responsible-ai-toolbox-tracker)
- [Gender bias identification Repo](https://github.com/microsoft/responsible-ai-toolbox-genbit)
- [AI Show](https://www.youtube.com/playlist?list=PLlrxD0HtieHi0mwteKBOfEeOYf0LJU4O1)

The Workshop raised some questions.

1.  The imbalanced data that I am working with need to be aggregated - how do I fix class imbalance without impacting the aggregations?

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2023,
  author = {Bochman, Oren},
  title = {Responsible {AI} {Mitigations} and {Tracker}},
  date = {2023-03-08},
  url = {https://orenbochman.github.io/posts/2023/03-08-responsible-AI/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2023. “Responsible AI Mitigations and Tracker.” March 8. <https://orenbochman.github.io/posts/2023/03-08-responsible-AI/>.
