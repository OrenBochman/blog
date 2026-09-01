## A Moot Meetup

This week I went to a meetup - that was a waste of time. I had nothing positive to write about it and I lost the blog post I initially wrote about it. Which went something like this:

I went to the venue via commuting via bus and a long walk in rather inclement weather. The venue was disappointing - the members of the host company looked like they were ancient or in a deep depression. The spread was cold pizza and beer.

No one introduced the speaker. But it seems that the speaker was a consultant who was on good terms with the host company. The speaker was pitching a QA product for agents and she was a good speaker - engaging and controlling the room well. But whenever people asked questions one of the C-level people would smugly answer instead of letting the speaker answer.

Like many such meetups there was a lot of talk about agents in production. Although it sounds fascinating, besides the speaker pitching QA product for agent there was nothing new. I don’t expect all meetups to be great.\
Many times mediocre people want to talk about their mediocre projects and it can put your own work in perspective.

## The talks

My main issue was that the talk show cased how the lessons were about fixing the kind of errors that a junior developer wouldn’t make more than once.

1.  running a delete sql instead of a select query
2.  getting the data format wrong
3.  not converting a destination city to its international code.
4.  not handling processing children companies when the tinker symbol gives no data on the parent company in question

I asked if besides doing QA for problems we all know about fixing using unit test if there was any insights about dealing with the kind of problems that are both endemic to agents, AI, ML-Models, Large Language Models that we don’t already know how to handle.

The C-Level person smugly answered that “You don’t understand - These are not the same old problems that all devs know how to solve since they are now under the preview of AI and agents”. And that we shouldn’t be coding anymore just “editing specs”. He didn’t comment when the speaker pointed out that Large Language Models responses tend to swing from Savant to Idiot.

The second speaker run out of time before he could get to the meat of his talk - some project the company had developed for using multiple personas to drive decisions.

## Some almost random thoughts on what I wanted to hear.

So I kept thinking about this what are the agentic and ML issues that we want to deal with in our agentic systems.

1.  Handling helucinations - this is perhaps the most important issue to deal with.
2.  Can users bully the agents
3.  When the agents forgets what you said or its prompt or bias overrides what you already negotiated.
4.  When that vendor changes the model and your agents no longer perform as expected.
5.  Agents runs a long long time and eats up your compute budget.
6.  Reasoning agents - the reasoning may have very little to do with their actual output of the model.
7.  Language models lack the ability to learn from experience.
8.  Agent as a judge - this is perhaps the most interesting use of Agents - very much a capability that is hard to get without a language model.
9.  Find when agent that don’t respect parts of the prompt.
10. Agents don’t acknowledge when they don’t know something - as [RLHF](https://arxiv.org/abs/2009.01325) drives them to answer with greater confidence.
11. Models tend to amplify biases.

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2026,
  author = {Bochman, Oren},
  title = {Rise of the Agents},
  date = {2026-01-14},
  url = {https://orenbochman.github.io/posts/2026/01-14-rise-of-the-agents/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2026. “Rise of the Agents.” January 14. <https://orenbochman.github.io/posts/2026/01-14-rise-of-the-agents/>.
