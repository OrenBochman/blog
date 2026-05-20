## Meaningful Data Visualization in the Age of AI

Play

![Vimeo](https://f.vimeocdn.com/p/images/crawler_logo.png)

- Janet Six
  - [LinkedIn](https://www.linkedin.com/in/janetmsix)
  - Tom Sawyer Software
  - [slides](https://docs.google.com/presentation/d/1-LaKE9ksL89OPvXA6f98r4EYDLbbquDi/edit?slide=id.p1#slide=id.p1)

> **NOTE:**
>
> - **Topic:** Making data visualization more meaningful in the age of artificial intelligence.
>
> - **Speaker:** Dr. Janet Six, senior product manager at Tom Sawyer Software, with a background in graph visualization and artificial intelligence.
>
> - **Core argument:** Artificial intelligence can make visualization faster and more abundant, but speed does not guarantee usefulness. A visualization is meaningful only when it helps people understand, decide, validate, or act.
>
> - **Opening analogy:** The speaker uses a hand-movement exercise to show that some problems become easier when reframed. Data visualization has the same issue: instead of asking “what chart should we make?”, teams should ask “what task, decision, or system understanding must this visualization support?”
>
> - **Visualization types mentioned:** Tables, pie charts, line graphs, bar graphs, timelines, matrices, network graphs, hierarchical views, and combinations of these.
>
> - **Main warning about AI:** AI acts like a megaphone: it amplifies speed and volume, but it does not automatically improve quality, relevance, or decision value.
>
> - **Systems engineering frame:** The speaker argues that AI-enabled visualization should be approached through systems engineering because modern AI systems involve interacting tools, people, goals, workflows, requirements, regulations, business processes, and changing stakeholder needs.
>
> - **Definition of a system:** A system is described as a set of interacting elements organized to achieve one or more stated purposes. The speaker emphasizes that both parts are difficult:
>
>   - identifying all interacting elements, including people, software, hardware, processes, and behaviors;
>   - stating the system’s purposes concretely enough to guide design and evaluation.
>
> - **Silo problem:** Different departments may hold conflicting or incomplete requirements. For example, business teams may want to adopt a tool, while operations or compliance teams may later discover it is unsupported or non-compliant. Systems engineering helps detect these conflicts earlier, when changes are cheaper.
>
> - **Supply-chain example:** A logistics visualization can show delivery routes as a graph or matrix, including attributes such as hazardous cargo, refrigeration needs, or heavy goods. But the visualization is only useful if tied to concrete objectives, such as optimizing flow or ensuring backup suppliers within a defined response time.
>
> - **Requirements hierarchy:** High-level business needs must be decomposed into measurable requirements, then into technical solutions. Example: “maintain a resilient supply chain” becomes “have a backup supply chain ready within two hours,” which can then guide AI-tool selection and system design.
>
> - **Risk, cost, and performance trade-offs:** The speaker explains that systems often require balancing competing objectives. Reducing risk and increasing performance usually raises cost; holding cost fixed may require accepting more time or lower performance.
>
> - **Assessment and traceability:** Teams should repeatedly assess whether implementation choices trace back to the original requirements. The speaker stresses the importance of linking stakeholder goals to design decisions, tool configurations, outputs, and validation criteria.
>
> - **Validation versus verification:**
>
>   - **Validation:** checking whether the overall system was built correctly and satisfies the intended requirements.
>   - **Verification:** checking whether individual outputs or ongoing results are acceptable, especially in systems that produce new results continuously, such as generative artificial intelligence systems.
>
> - **V-model:** The speaker relates this process to the systems engineering V-model: requirements and design flow downward, development happens at the bottom, and testing, verification, and validation occur on the upward side.
>
> - **Systems of systems:** Modern AI systems may contain multiple agents or tools interacting with one another. Meaningful visualization may need to show both high-level parent systems and lower-level sub-agent behavior.
>
> - **Hierarchical visualization:** Different audiences need different levels of abstraction:
>
>   - executives may need high-level system status;
>   - engineers may need lower-level agent, workflow, and traceability views.
>
> - **Time and actors:** Useful visualizations should often include temporal structure and actor relationships, not just static snapshots. This supports event analysis, process understanding, and version comparison.
>
> - **Agentic explainability example:** In a fraud-detection scenario, a visualization might show:
>
>   - the suspicious transaction or finding;
>   - related merchants, accounts, and transactions;
>   - which agents contributed to the finding;
>   - confidence or certainty levels;
>   - where in the workflow the result was produced.
>
> - **Versioning:** The speaker notes that visualizations can also show how systems, graphs, or results change over time, either retrospectively or prospectively.
>
> - **Final recommendation:** Put visualization at the beginning of system design, not merely at the end. Use it to clarify requirements, expose conflicts, support optimization, communicate with stakeholders, and keep human experts in the loop.
>
> - **Takeaway:** Meaningful visualization is not just presentation. It is a systems-engineering instrument for understanding complex AI-enabled systems, making trade-offs explicit, tracing requirements to outputs, and supporting better human judgment.

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2026,
  author = {Bochman, Oren},
  title = {Meaningful {Data} {Visualization}},
  date = {2026-04-28},
  url = {https://orenbochman.github.io/posts/2026/04-28-ODSC-AI-2026-Day-1/talk14.html},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2026. “Meaningful Data Visualization.” April 28. <https://orenbochman.github.io/posts/2026/04-28-ODSC-AI-2026-Day-1/talk14.html>.
