## Real-Time Event-Time Consistent Analytics Pipelines using Kafka, Flink, and Apache Pinot

- Deep Patel
  - [LinkedIn](https://www.linkedin.com/in/deeppatel710/)
  - Robinhood

> **NOTE:**
>
> - **Talk topic:** Deep, a senior data engineer at Robinhood, explains how Robinhood built **Catbox Data**, an internal real-time analytics platform using **Kafka, Flink, and Apache Pinot**.
>
> - **Motivation: batch analytics was too slow**
>
>   - Robinhood relied heavily on Looker and Presto-based dashboards.
>   - Dashboards were slow, sometimes taking many seconds or even minutes.
>   - There was no standardized materialization of reusable analytic cubes.
>   - Batch data freshness was often daily, hourly, or at best around 30 minutes.
>   - Product, analytics, and fraud-investigation teams needed fresher data, especially after launches or alerts.
>
> - **Platform goals**
>
>   - Provide low-latency ingestion.
>   - Support sub-second or near-sub-second analytical queries.
>   - Preserve **event-time consistency**, not merely processing-time consistency.
>   - Combine fresh streaming data with reconciled historical data.
>   - Keep a SQL-based interface for data scientists and analysts.
>   - Scale horizontally as usage and data volume increased.
>
> - **Architecture overview**
>
>   - Product and service events are emitted into **[Kafka](https://kafka.apache.org/)**.
>   - A central **[Flink](https://flink.apache.org/)** application consumes Kafka streams.
>   - Flink performs stateful processing and enrichment where needed.
>   - Processed events are written back to Kafka.
>   - **[Apache Pinot](https://pinot.apache.org/)** ingests the processed Kafka stream in real time.
>   - In parallel, historical and reconciled data is processed from the *[data lake](https://en.wikipedia.org/wiki/Data_lake)* using **[Spark](https://spark.apache.org/)** and ingested into Pinot as offline data.
>
> - **Why Flink**
>
>   - Flink was chosen for true streaming, event-time processing, state management, and exactly-once-style semantics.
>   - Existing Robinhood infrastructure already used Flink for fraud and feature engineering.
>   - Event time mattered because financial analytics should reflect when a trade occurred, not when the pipeline happened to process it.
>
> - **Why Apache Pinot**
>
>   - Pinot is a distributed columnar Online Analytical Processing database designed for high-throughput, low-latency analytics.
>   - It supports real-time ingestion from Kafka.
>   - It supports *upserts*, useful for changing entities such as order status or user profile attributes.
>   - It can consume change data capture streams, for example through [Debezium](https://debezium.io/).
>   - It offers *retention policies*, *partial upserts*, *indexing*, and high query throughput.
>   - *Star-tree* and other indexes were important for fast analytical queries.
>
> - **Hybrid table strategy**
>
>   - Robinhood used Pinot’s hybrid table model to combine real-time and offline data.
>   - A table has both an online/real-time side and an offline/batch side with the same schema.
>   - Users query a single logical table without needing to know which part is online or offline.
>   - Pinot determines the boundary between offline and real-time data and merges the results.
>   - This enabled Lambda-style architecture: fast approximate/fresh data first, then corrected/reconciled data later.
>   - The same mechanism also helped bootstrap historical data before starting real-time ingestion.
>
> - **Join and denormalization strategy**
>
>   - The team preferred to denormalize data at the producer level whenever possible.
>   - Flink joins were used sparingly because stateful joins can become difficult to operate and debug, especially with late-arriving data.
>   - Some enrichment was done in Flink, for example attaching customer tags to transactions.
>   - Pinot’s query-time joins provided another option when producer-side denormalization was not feasible.
>
> - **Results**
>
>   - Dashboard P95 latency improved from about **24 seconds** to about **800 milliseconds**.
>   - Some dashboards loaded in **50–100 milliseconds** after migration.
>   - Data freshness improved from daily/hourly/30-minute batch updates to about **five seconds**.
>   - Concurrent usage grew from roughly **50 users** to around **500 users**.
>   - Robinhood also moved from Looker to Superset for dashboarding.
>
> - **Lessons learned**
>
>   - Real-time analytics should be treated as a platform, not only as infrastructure for fraud or machine learning features.
>   - Event-time consistency is critical in financial analytics.
>   - Hybrid real-time/offline tables are a practical way to balance freshness with correctness.
>   - Denormalization should happen as early as possible, ideally at the producer, while stateful stream joins should be minimized.
>   - Index design in Pinot is central to achieving low-latency query performance.
>
> - **Q&A points**
>
>   - Pinot was not considered a good fit for unstructured data in Robinhood’s experience; they used document stores for that.
>   - Robinhood’s data engineering team was described as lean, with around 20 data engineers.
>   - The company uses AI agents heavily, including tools such as Claude Code and Codex, and encourages even non-technical employees to write code.

### Reflection

- This is one of the three most interesting talks of the conference for me along with Tensor Logic and RL talks.
- This talk is more on the level of a *pydata* session.
- I recently started to implement a pattern of incorporating an analytical DB (OLAP) in my agentic projects.
  - This led me to get tight with [DuckDB](https://duckdb.org/).
- I like many others was also looking for tooling efficient search and RAG that are required by many of the agentic applications and harness engineering patterns.
- I have lots of experience with Lucene as I used it to implement NLP boosted search for Wikipedia sized corpus.
- I got interested in implementing dual search tool i.e. [BM25F](https://en.wikipedia.org/wiki/Okapi_BM25) type ranking for [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) and RAG via vectors using lucene and this is started.
- This led me to and heard about [Apache Pinot](https://pinot.apache.org/) and since Pinot is built on top of Lucene this naturally peaked my curiosity.
- I am looking at using an OLAP for:
  1.  Tracking diverse project metrics.
  2.  Collecting proof of work artifacts.
  3.  Handling project traces from open telemetry and logs.
  4.  Having multiple replay buffers for RL sub-agents
  5.  Anomaly detection - for early detection of failure modes by the project and the agents

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2026,
  author = {Bochman, Oren},
  title = {Real-Time {Event-Time} {Consistent} {Analytics} {Pipelines}},
  date = {2026-04-28},
  url = {https://orenbochman.github.io/posts/2026/04-28-ODSC-AI-2026-Day-1/talk8.html},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2026. “Real-Time Event-Time Consistent Analytics Pipelines.” April 28. <https://orenbochman.github.io/posts/2026/04-28-ODSC-AI-2026-Day-1/talk8.html>.
