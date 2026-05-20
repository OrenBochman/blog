# An error occurred.

Unable to execute JavaScript.

> **TIP:**
>
> Get a firsthand look at how we built a lightweight feature store to accelerate electricity grid forecasting. We’ll cover our decision process, design choices, and implementation using Polars and Google Cloud Storage. Expect lessons learned, real-world bumps, and a clear view of the costs, trade-offs and benefits of our solution.

In this talk, we’ll share how we built a lightweight, production-ready feature store to support electricity grid forecasting. You’ll hear a firsthand account of our journey—from identifying the need to accelerating model prototyping through feature standardization and flexibility.

We’ll start with a high-level overview of our decision-making process: why we chose to build rather than buy, and the trade-offs we considered. Then, we’ll dive into the architecture of our custom feature store, detailing how we leveraged Polars for fast processing and Google Cloud Storage as a scalable backend.

Expect an honest look at the challenges we faced, the benefits we gained, and the costs we encountered along the way. Whether you’re considering building your own feature store or just curious about scaling ML for time series problems, this session will offer practical insights and real-world lessons.

> **TIP:**

## Outline

Building a Lightweight Feature Store For Electricity Grid Forecasts with `Polars`

[![Building a Lightweight Feature Store](slide01.png)](slide01.png "Building a Lightweight Feature Store")

Building a Lightweight Feature Store

Who’s talking?

- Hey, I’m Robin!
- Joined Electricity Maps in 2022 as a data engineer. Originally in the Data Platform team, now in the Grid Forecasts team.
- My mission: making sure our models get all the features they need to produce forecasts and distributing forecasts to customers.

[![Who's Talking](slide02.png)](slide02.png "Who’s Talking")

Who’s Talking

Today’s agenda:

- Forecasting at Electricity Maps
- Why we needed a feature store
- Breaking down our feature store
- A few learnings

[![Slide caption](slide03.png)](slide03.png "Slide caption")

Slide caption

Forecasting at Electricity Maps

[![Forecasting at Electricity Maps](slide04.png)](slide04.png "Forecasting at Electricity Maps")

Forecasting at Electricity Maps

Electricity Maps centralises, standardizes and forecasts global electricity data in real-time

[![Electricity Maps centralises, standardizes and forecasts global electricity data in real-time](slide05.png)](slide05.png "Electricity Maps centralises, standardizes and forecasts global electricity data in real-time")

Electricity Maps centralises, standardizes and forecasts global electricity data in real-time

### Our forecasts

Taking a look at the **problem setting** ✏️

- Entities to model:
  - 190+ consumption zones
  - 250+ interconnectors
- All available production modes:
  - Wind
  - Solar
  - Nuclear
  - Gas
  - Coal
  - And more (12 in total)
- Let’s not forget the **interconnections**
- Let’s provide **day-ahead prices**
- And also **load**
- And **carbon intensity**
- And finally, **renewable energy %**

[![Our forecasts](slide06.png)](slide06.png "Our forecasts")

Our forecasts

------------------------------------------------------------------------

### Our forecasts

[![Slide 07](slide07.png)](slide07.png "Slide 07")

Slide 07

------------------------------------------------------------------------

The features behind our forecasts: Stateful time series features

- Target Time: 2025-12-03 13:00:00

- Reference Time: 2025-12-01 13:00:00

- Target Time: 2025-12-03 13:00:00

- Reference Time: 2025-12-02 13:00:00

[![The features behind our forecasts: Stateful time series features](slide08.png)](slide08.png "The features behind our forecasts: Stateful time series features")

The features behind our forecasts: Stateful time series features

------------------------------------------------------------------------

## Why we need a feature store

[![Why we need a feature store](slide09.png)](slide09.png "Why we need a feature store")

Why we need a feature store

------------------------------------------------------------------------

## Before the feature store I

- pay every time they run a forecast
- wait every time they want to run a forecast
- doesn’t scale with the number of features.
- no shared features across users.

[![Costs and Speed](slide12.png)](slide12.png "Costs and Speed")

Costs and Speed

## Before the feature store II

- combos of features are tightly coupled
- combining features from different scales is very hard
- combinatorial class number explosion …

[![Combining features](slide13.png)](slide13.png "Combining features")

Combining features

------------------------------------------------------------------------

## Before the feature store III

- 🐢 Experimentation is slow: building a new combination of features + computing ~days.
- 🔥 Cloud costs are exploding the more we scale our forecast systems to train on longer horizons and on longer training sets (yearly).
- 😡 A lot of frustration with memory errors due to memory spikes and large parquet files that can’t load (30GB for a single zone).

[![Reasons for the feature store](slide14.png)](slide14.png "Reasons")

Reasons

------------------------------------------------------------------------

## Powering the Feature Store: GCS and Polars

[![Behind our Feature Store: GCS and Polars](slide16.png)](slide16.png "Behind our Feature Store: GCS and Polars")

Behind our Feature Store: GCS and Polars

------------------------------------------------------------------------

## General Overview

[![General Overview](slide17.png)](slide17.png "General Overview")

General Overview

## Lazy Queries

[![Slide 18](slide18.png)](slide18.png "Slide 18")

Slide 18

## The Querying process

[![the Querying process](slide19.png)](slide19.png "the Querying process")

the Querying process

## Lazy read query

[![Slide 20](slide20.png)](slide20.png "Slide 20")

Slide 20

## Aligning time dimensions

[![Aligning time dimensions](slide21.png)](slide21.png "Aligning time dimensions")

Aligning time dimensions

## How to mix several features together?

- 🌧️ WeatherData:
  - Update frequency 6 hours
  - Resolution 1 hour
  - Lookahead 36 hours
- 🔋 GridData:
  - Update frequency hourly
  - Resolution 30min
  - Lookahead 24 hours
- 🧠 What the model needs:
  - Available data for every lead time
  - Resolution 30min
  - Lookahead 96 hours

[![How to mix several features together](slide22.png)](slide22.png "How to mix several features together?")

How to mix several features together?

## Solving the right problem at the right time

Feature Generation Pipeline:

- **Feature Inputs**: Source resolution and maximum lead time.
- **Feature Generators**: Resolution and maximum lead time set by Feature Generator, refresh frequency based on source.
- **Feature Store Storage**: Resolution and maximum lead time set by Feature Generator, Entity.
- **Feature Store Client**: Resolution, maximum lead time set by user.

### Examples:

- 🌧️ **WeatherData**:
  - Update frequency: 6 hours
  - Resolution: 1 hour
  - Lookahead: 120 hours
- 🔋 **GridData**:
  - Update frequency: Hourly
  - Resolution: 15 minutes
  - Lookahead: 48 hours
- 🧠 **What the model needs**:
  - Available data for every lead time
  - Resolution: 30 minutes
  - Lookahead: 96 hours

[![Solving the right problem at the right time](slide23.png)](slide23.png "Solving the right problem at the right time")

Solving the right problem at the right time

## Solving the right problem at the right time II

[![Solving the right problem at the right time II](slide24.png)](slide24.png "Solving the right problem at the right time II")

Solving the right problem at the right time II

## Binding everything together

[![Binding everything together](slide25.png)](slide25.png "Binding everything together")

Binding everything together

## Binding everything together II

[![Binding everything together II](slide26.png)](slide26.png "Binding everything together II")

Binding everything together II

## Binding everything together III

[![Binding everything together III](slide27.png)](slide27.png "Binding everything together III")

Binding everything together III

Slide 28

[![Slide 28](slide28.png)](slide28.png "Slide 28")

Slide 28

Slide 29

[![Slide 29](slide29.png)](slide29.png "Slide 29")

Slide 29

Slide 30

[![Slide 30](slide30.png)](slide30.png "Slide 30")

Slide 30

Slide 31

[![Slide 31](slide31.png)](slide31.png "Slide 31")

Slide 31

Slide 32

[![Slide 32](slide32.png)](slide32.png "Slide 32")

Slide 32

Slide 33

[![Slide 33](slide33.png)](slide33.png "Slide 33")

Slide 33

Slide 34

[![Slide 34](slide34.png)](slide34.png "Slide 34")

Slide 34

Slide 35

[![Slide 35](slide35.png)](slide35.png "Slide 35")

Slide 35

THank You + Contact

[![Slide 36](slide36.png)](slide36.png "Slide 36")

Slide 36

## Reflections

- People often say that data science is 90% data cleaning and 10% modeling.

- This war story is a great example of this. The only time we heard about data science was when we talked about the feature aggregation. P.s. this should be a one liner.

- However for the one liner to actually work they had to iterate and do some data engineering work. The data engineering is trivial from a data science perspective however it takes a lot of time and effort to get it right. And data scientist are not dev ops. So there are often a few iterations.

- Another well known secret is even when it comes together the client etc will change thier api, data scheme or requirements and you have to iterate again.

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2025,
  author = {Bochman, Oren},
  title = {Building a {Lightweight} {Feature} {Store} for {Electricity}
    {Grid} {Forecasts} with {Polars}},
  date = {2025-12-12},
  url = {https://orenbochman.github.io/posts/2025/2025-12-11-pydata-lightweight-feat-store/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2025. “Building a Lightweight Feature Store for Electricity Grid Forecasts with Polars.” December 12. <https://orenbochman.github.io/posts/2025/2025-12-11-pydata-lightweight-feat-store/>.
