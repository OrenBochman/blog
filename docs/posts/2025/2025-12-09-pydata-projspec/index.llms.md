# An error occurred.

Unable to execute JavaScript.

> **TIP:**
>
> Most code and related workflows take place in “projects”, directories with descriptive metadata. There are so many types of these around these days, it is hard to know what is contained where. projspec solves this for the majority of the python-data ecosystem, so that you can introspect your projects, act on them, and search across all your projects, local or remote.

Daily workflows in pydata usually occur in the context of projects - a directory tree of stuff, with special metadata files describing those contents. Many metadata specifications are in use for each of the many tools that operate on projects, storing information in small yaml, toml or json files, or in the pyproject.toml file for python-specific projects. This model encompasses not only the majority of the environment management tools and task runners in pydata (uv, pixi, poetry, etc) but other essential tools (e.g., git), definitions (e.g., hugging-face dataset), deployment (briefcase, helm, wheel) and workflow-specific metadata (e.g., pyscript).

The range of possible metadata is bewildering! Most projects show how to invoke their functionality in README files, with the first step downloading some specific tool. In some way, all this flexibility has taken us backwards. There is no easy way to tell what type a project is and what definitions it contains without reading the supporting documentation and browsing specific files, or even downloading the whole thing and running a specific tool against it.

projspec aspires to be a layer over the most common pydata related project types. It provides introspection of project type and contents from the metadata definitions, and this can be done on remote project directories too. For each project type, we infer a set of “contents” (things that are defined in the project and inherently part of it) and “artifacts” (things the project can make or do, usually by calling a subprocess). A project can be multiple types at once: a project designed to be executed with pixi, for instance, still likely contains git information and may also have dataset declarations, things that pixi is not concerned with. Projects may also contain sub-projects of the same or different type, e.g., a conda recipe alongside a code library.

Projspec, due to be released in time for this talk, will provide a handy API to work with projects of many types, including introspection and effecting actions. It will have a way to index many projects locally or remotely, to allow for querying with complex criteria, to find the project that matches your needs - contains certain datasets, depends on specific library/versions or is capable of creating particular output types. We will demonstrate all of this!

> **TIP:**

- [talk repo](https://github.com/)
- [slide deck](https://github.com/hugobowne/AI-for-SWEs)

## Outline

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

## Reflections

My main impression is that projspec might be a very useful tool for people training models that work with code.

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2025,
  author = {Bochman, Oren},
  title = {Projspec: What’s This Project Anyway?},
  date = {2025-12-09},
  url = {https://orenbochman.github.io/posts/2025/2025-12-09-pydata-projspec/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2025. “Projspec: What’s This Project Anyway?” December 9. <https://orenbochman.github.io/posts/2025/2025-12-09-pydata-projspec/>.
