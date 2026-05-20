## TL;DR

([Huttegger et al. 2010](#ref-huttegger2010evolutionary))

## Abstract

Abstract Transfer of information between senders and receivers, of one kind or another, is essential to all life. David Lewis introduced a game theoretic model of the simplest case, where one sender and one receiver have pure common interest. How hard or easy is it for evolution to achieve information transfer in Lewis signaling?. The answers involve surprising subtleties. We discuss some if these in terms of evolutionary dynamics in both finite and infinite populations, with and without mutation.

## This paper

I’ve not looked too deeply into this paper as it needs a deep dive into evolutionary dynamics which I’ve yet to study in depth. My interests are RL and complex signaling systems. Some of the results in evolutionary dynamics can be directly applied to reinforcement learning.

The paper looks at moran processes and mutations. This is interesting in the open ended setting of language evolution.

> Because this is a partnership game, average payoff is a **Lyapunov function** [^1] for the system \[in fact, it is even a potential function; see ([Hofbauer and Sigmund 1998](#ref-hofbauer1998evolutionary)) \].

The look at Lyapunov functions and the replicator dynamics. This is interesting in the context stability of equilibria in signaling games.

> raises the same questions in the context of evolution in finite populations with fixed population size (via the Moran process with and without mutation)

## The Paper

paper

Hofbauer, Josef, and Karl Sigmund. 1998. *Evolutionary Games and Population Dynamics*. Cambridge university press.

Huttegger, Simon M, Brian Skyrms, Rory Smead, and Kevin JS Zollman. 2010. “Evolutionary Dynamics of Lewis Signaling Games: Signaling Systems Vs. Partial Pooling.” *Synthese* 172: 177–91.

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2024,
  author = {Bochman, Oren},
  title = {Evolutionary Dynamics of {Lewis} Signaling Games: Signaling
    Systems Vs. Partial Pooling},
  date = {2024-10-08},
  url = {https://orenbochman.github.io/reviews/2009/HutteggerSkryms2009/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2024. “Evolutionary Dynamics of Lewis Signaling Games: Signaling Systems Vs. Partial Pooling.” October 8. <https://orenbochman.github.io/reviews/2009/HutteggerSkryms2009/>.

[^1]: **Lyapunov functions** are scalar functions that may be used to prove the stability of an equilibrium of an ODE
