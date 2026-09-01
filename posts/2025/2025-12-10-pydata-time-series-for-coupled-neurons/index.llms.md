# An error occurred.

Unable to execute JavaScript.

> **TIP:**
>
> The complex nervous system provides a repertoire of evolutionary properties like neuron spiking, bursting, and chaos that are yet to be fully understood.
>
> One approach is to tackle these time-dependent properties using the technique of “dynamical systems”, such as ordinary differential equations. Since the popular work by [Hodgkin and Huxley](https://en.wikipedia.org/wiki/Hodgkin%E2%80%93Huxley_model), many dynamical systems models of neurons have been proposed, of which [FitzHugh–Nagumo](https://en.wikipedia.org/wiki/FitzHugh%E2%80%93Nagumo_model) and [Morris–Lecar](https://en.wikipedia.org/wiki/Morris%E2%80%93Lecar_model) models draw special attention.
>
> The nervous system is made of a network of neurons, possessing a complex structural and functional topology. This topology is a function of different parameters, among which the coupling strength plays a major role. Our focus would be to systematically study the effect of various coupling strategies on the firing patterns exhibited by a collection of neurons.
>
> In this workshop, my goal is to popularize a reduced-order model of neuron dynamics known as the “denatured Morris–Lecar” system and to teach how Python can be efficiently used to perform research on time series analysis of coupled neurons.

This is a tutorial on hands-on time series analysis of coupled neuron models. We will build mathematical models of coupled neurons, and then utilize tools from nonlinear dynamics to analyze simulated time series. We will discuss various empirically informed coupling strategies and statistically efficient time series measures. This workshop is 100% Jupyter notebook and will have room to openly brainstorm ideas to extend and improve the studies. Let’s unravel some complex dynamics together!

The pipeline of this tutorial will be the following:

1.  Start by building a coupled neuron system based on different coupling strategies,
2.  Simulate the system and generate time series data,
3.  Perform time series analysis by computing various metrics from the nonlinear dynamics literature,
4.  Finally, discuss what these metrics tell us about the temporal behavior of neurons.

Coupling strategies we are going to look at:

1.  Gap junction coupling
2.  Chemical coupling
3.  A hybrid coupling influenced by a superconductor model in physics
4.  Electromagnetic coupling
5.  Coupling, which is not pairwise but higher-order (A bit of background on graph theory is recommended)
6.  A random coupling strategy

The audience would find this interesting because it would be a hands-on introduction to how the mechanisms of neurons can be explored using different tools from the nonlinear dynamics literature. Mathematically modelling the dynamics of neurons has attracted several researchers in recent years because of the popularity of artificial intelligence. This field of neuron dynamics is booming, and delivering this workshop would be timely. I would also ensure to leave some room for brainstorming further ideas with the audience and how this study could be potentially extended and improved, thus an interactive session.

The goal is to attract applied mathematicians, computer scientists, data scientists, engineers, and statisticians alike and provide them with a battery of tools to add to their knowledge base. The audience would then be able to apply these tools in domains other than neurodynamics, for example, climate, finance, or social science. The only technical background I would expect from the audience is familiarity with matplotlib, numpy and pandas, and some basic statistics (regression, correlation coefficient), linear algebra (matrix operations), and graphs (as in networks). After the tutorial, the audience will leave with a newly built insight into the mathematical modeling of neuron dynamics.

> **TIP:**
>
> We will implement the following methodologies/algorithms for time series analysis of coupled neuron models:
>
> 1.  [Hurst exponent](https://en.wikipedia.org/wiki/Hurst_exponent): measuring persistence of time series,
> 2.  [Sample entropy](https://en.wikipedia.org/wiki/Sample_entropy): measuring the complexity of time series,
> 3.  [0–1 test](https://talus.maths.usyd.edu.au/u/gottwald/preprints/testforchaos_MPI.pdf): measuring chaos,
> 4.  [Kuramoto order-parameter](https://en.wikipedia.org/wiki/Kuramoto_model): measuring synchrony between the neurons.

> **TIP:**
>
> - Basic Python and PyTorch
> - Some familiarity with neural networks (e.g., feedforward, softmax)
> - No need for prior experience in building models from scratch

> **TIP:**
>
> This tutorial is 100% Python. And I will be utilizing Jupyter Notebooks to deliver the workshop. Packages that need to be downloaded beforehand are:
>
> 1.  matplotlib for plotting,
> 2.  numpy and scipy for scientific computations,
> 3.  nolds for nonlinear measure for dynamical systems,
> 4.  pandas for data handling.

[workshop repo](https://github.com/indrag49/PyData-Global-Tutorial-2025)

> **TIP:**

## Outline

0–15 mins: Introduction to neurons as dynamical systems and why we care about their behavior over time. We will talk about a single neuron’s behavior and the selection of a mathematical model. We will also talk about the bursting phenomenon in neurons.

15-30 mins: We will then mathematically model a coupled system of neurons. We will cover the topic of “small networks” of neurons and what they teach us about the bigger picture: a complex, connected nervous system.

30-45 mins: Next, we will introduce various empirically informed coupling mechanisms. We will talk about how these couplings incorporate different firing patterns in the coupled neurons, ranging from regular behavior to chaotic firing.

45-75 mins: Finally, I will introduce time series analysis of neuron data. We will then implement the algorithms mentioned above to realize different dynamical properties of the neurons.

75-90 mins: Open the room to QA and brainstorm further ideas to improve/extend the analysis of neuron-time series data.

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

[![](slide21.png)](slide21.png)

[![](slide22.png)](slide22.png)

[![](slide23.png)](slide23.png)

[![](slide24.png)](slide24.png)

[![](slide25.png)](slide25.png)

[![](slide26.png)](slide26.png)

[![](slide27.png)](slide27.png)

[![](slide28.png)](slide28.png)

[![](slide29.png)](slide29.png)

[![](slide30.png)](slide30.png)

> **TIP:**
>
> article - [Taming the Chaos of Computational Experiments](https://www.siam.org/publications/siam-news/articles/taming-the-chaos-of-computational-experiments/)
>
> main points:
>
> 1.  Use Version Control (Preferably Git)
> 2.  Separate Experiments From Figures
> 3.  Create Human-digestible Logs
> 4.  Log Details of Individual Runs
> 5.  Use Scripts for Automation
> 6.  Separate and Organize Your Project Components
> 7.  Share Your Code, Data, and Experimental Logs
>
> (listed in roughly increasing order of difficulty)

[![](slide32.png)](slide32.png)

[![](slide33.png)](slide33.png)

[![](slide34.png)](slide34.png)

[![](slide35.png)](slide35.png)

[![](slide36.png)](slide36.png)

[![](slide37.png)](slide37.png)

## Reflection

This is a very exciting talk.

First of all it is the first time I see work on simulation of (coupled) neurons, a topic that get mentioned all too ofthen in popular science on chaos, dynamical systems, and complex systems, but rarely in practical terms.

I have taken a few courses on time series which might be relevant but this is both a new problem and a new set of tools for me.

Near the end of the talk we discuss networks of neurons, which is another topic altogether and it appears that there should be more exploration in this area - perhaps about criticality, percolation, and emergence.

The talk is well structured, with a clear outline and flow. The speaker clearly explains the motivation behind studying coupled neurons and the relevance of time series analysis in this context. The use of Python and Jupyter Notebooks makes the workshop accessible and practical.

------------------------------------------------------------------------

Some ideas

1.  using dlm as a black box to model neuron time series data
2.  using a kalman filter to denoise neuron time series data or to predict future states based on laplace transform methods.
3.  Are there control applications here?
    1.  (can a neuron drive coupled neurons into excitatory or inhibitory states using optimal control methods?)
    2.  can these neurons be used to as model for information themselves. Like a more realistic RNN model. This might lead to new insights about catastrophical forgetting and loss of plasticity in ANNs.
    3.  can they work as a classifier, regressor, rl agent?
    4.  in a network are there nodes that act as pain pleasure centers that can be targeted for RL rewards and punishments? This might suggest new architectures for modeling reinforcement learning agents - or at least a new way of implementing the reward/punishment mechanisms in say the OAK architecture recently described by Rich Sutton.
    5.  In a node are there structually placed neurons we might destroy to prevent siezure regimes? A kind of electroshock therapy for epilepsy?

    \
4.  also the neurons might have several regimes (both individual and collectively) this suggest the use switching mixture markov models.
5.  I would think a good challenge would be to build poincare maps of the neuron dynamics and use them to classify the different regimes of the neurons. This would then be useful to setup the switching markov model.
6.  question: Can one add a new chaotic components to the already versatile DLM that is inspired by long term dependencies. It could include some form of fractional differencing or fractional integration.
7.  can we convert the shiny app to a shinylive app so it runs just in the browser and does not need a server?
8.  popular video [You’ve (Likely) Been Playing The Game of Life Wrong](https://youtu.be/HBluLfX2F_k?t=1036)

# An error occurred.

Unable to execute JavaScript.

You’ve (Likely) Been Playing The Game of Life Wrong

> **TIP:**
>
> add citations to the various resources mentioned in the talk.
>
> embed the shiny app or at least link to it.
>
> link to the github repo with the notebooks.
>
> quick review [Taming the Chaos of Computational Experiments](https://www.siam.org/publications/siam-news/articles/taming-the-chaos-of-computational-experiments/)

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2025,
  author = {Bochman, Oren},
  title = {Time Series Analysis for Coupled Neurons.},
  date = {2025-12-10},
  url = {https://orenbochman.github.io/posts/2025/2025-12-10-pydata-time-series-for-coupled-neurons/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2025. “Time Series Analysis for Coupled Neurons.” December 10. <https://orenbochman.github.io/posts/2025/2025-12-10-pydata-time-series-for-coupled-neurons/>.
