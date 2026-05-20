> **NOTE:**
>
> [![Externalities in a nutshell](../../../images/in_a_nut_shell.png)](../../../images/in_a_nut_shell.png "Externalities in a nutshell")
>
> Externalities in a nutshell
>
> **Thomas Schelling’s 1973 article explores binary choices where one person’s decision affects others, known as externalities.** Using examples like hockey helmets and daylight saving, the paper examines situations where individual self-interest may not lead to the best collective outcome. **Schelling analyzes various configurations of these choices, including the Prisoner’s Dilemma and scenarios with multiple equilibria.** He discusses how factors like knowledge, observation, and the formation of coalitions influence these decisions. **The analysis extends to consider cases with more than two choices and situations where individuals have different payoffs.** Ultimately, the article provides a framework for understanding the complexities of interdependent decision-making and the challenges of achieving efficient social outcomes when externalities are present.

![](hero-ice-hockey.png)

hockey helmets

When I took [Scott E. Page](https://www.scottepage.com/) course on [model thinking](https://www.coursera.org/learn/model-thinking), I felt like I might gain more from reading the original papers that introduced the models, rather than just reading about them in the textbook. THis led me to books and later this paper by Thomas Schelling. In Schelling ([1973](#ref-schelling1973hockey)), the author explores the phenomenon of individuals making binary choices that are influenced by the choices of others. He uses examples such as wearing hockey helmets, carrying concealed weapons, and adhering to daylight saving time to illustrate the complexities that arise from varying externality configurations in large populations. The paper explores the dynamics of these choices, highlighting the significance of spatial arrangement and the influence of knowledge and observation on individual decisions.

I don’t mean to suggest that it is trivial, not easy, but that he is able to take a very simple model that then turns out to be very general.

One of my personal issues is complexity, and I think that reading classics like this imparts the power of simple models.

------------------------------------------------------------------------

> **TIP:**
>
> I enjoy reading Schelling’s work, because he likes to tackle interesting issues at a level where very simple analysis tools can be brought to bear and then he brings out reams of conclusions and implications of the analysis. This often lets him leverege a simple model to help reach a eureka moment.
>
> And to his credit - Schelling never lets things slide after the eureka moment, he follows through with the implications and consequences of the model, and thus we get a very rich analysis from a simple model. This paper had two versions - one 48 pages and another 83 pages. This is a quality of “doing the work” is often missing. Some researcher seem much more brilliant thant Schelling for example Oppenheimer, who was often satisfied with the quick wins and ended up with mistakes and bad conclusions.

![](hero-concealed-weapons.png)

concealed weapons

Here is a lighthearted Deep Dive into the paper:

## Glossary

This paper uses lots of big terms so let’s break them down so we can understand them better

Binary Choice  
A decision context offering exactly two mutually exclusive options.

Externality  
An effect of a decision or action on a third party who did not have a choice in the matter.

Internality  
The direct effect of an individual’s choice on their own well-being or payoff.

Payoff  
The benefit, reward, or consequence associated with a particular choice in a given situation.

Dominant Strategy  
A course of action that yields the best outcome for a decision-maker regardless of what other participants in the situation choose.

Dominated Strategy  
A course of action that yields a worse outcome for a decision-maker compared to another strategy, no matter what other participants choose.

Equilibrium  
A stable state in a system where opposing forces are balanced, and there is no inherent tendency for change. In game theory, it often refers to a situation where each actor is making the best choice given the choices of others.

Inefficient Equilibrium  
An equilibrium outcome where all individuals are acting rationally in their own self-interest, but the resulting overall outcome is worse for everyone compared to a different possible outcome.

Collective Maximum  
The configuration of choices across all individuals that results in the highest possible total or average payoff for the group as a whole.

Coalition  
A group of individuals who cooperate or coordinate their actions to achieve a shared goal.

Viable Coalition  
A coalition of sufficient size that its members benefit from cooperating on a particular choice, even if others do not join.

Free Rider  
An individual who benefits from the actions or resources of others without contributing themselves.

MPD (Multiperson Prisoner’s Dilemma)  
An extension of the Prisoner’s Dilemma to a group larger than two, characterized by a dominant strategy for each individual that leads to a collectively suboptimal outcome.

MIRV (Marginal Individual Right Value)  
A curve representing the value of choosing the “Right” option for individuals with differing internal preferences for “Left,” as the number choosing “Right” increases. Used in analyzing situations with identical externalities but different internalities.

IRC (Induced Right Choice)  
A curve showing, for any proportion of the population choosing “Right,” the proportion of the population for whom a “Right” choice would be preferred. Used in analyzing situations with identical internalities but different externalities.

Compatibility  
A situation where uniform choices among individuals lead to better outcomes for everyone compared to mixed choices.

Complementarity  
A situation where a mix of different choices among individuals leads to a better overall outcome compared to everyone making the same choice.

Dual Equilibria  
A situation where there are two stable equilibrium points, often at opposing extremes of the possible choices.

## Outline

- Introduction
  - Describes the phenomenon of individuals making binary choices that are affected by the choices of others.
  - Mentions examples such as wearing hockey helmets, carrying concealed weapons, and adhering to daylight saving time.
  - Discusses the complexities arising from varying externality configurations in large populations.
  - Highlights the simplified scenario of identical payoffs and influences for all individuals.
  - Introduces the concept of a “receiving” and “transmitting strength” for individuals in a system of externalities.
  - Explains the influence of spatial arrangement on individual choices.
- Knowledge and Observation
  - Discusses the impact of knowledge and observation on binary choices with externalities.
  - Presents examples illustrating the varying visibility of choices and their consequences.
  - Notes the importance of monitoring individual choices or aggregates for discipline and enforcement.
- Prisoner’s Dilemma
  - Introduces the classic prisoner’s dilemma scenario and its characteristics.
  - Extends the definition of prisoner’s dilemma to multi-person scenarios (MPD).
  - Defines the key parameters of MPD, including the minimum viable coalition size (k) and the influence of externalities.
  - Presents graphical representations (Figure 2) of different MPD scenarios.
  - Explains the concept of collective maximum and its relation to individual choices.
  - Connects the graphical representations to real-world examples like rationing schemes.
- The Significant Parameters
  - Discusses the significance of parameters like k, k/n, and nk in varying situations.
  - Highlights the impact of population size (n) on externalities and coalition viability.
  - Explains the concept of increasing and diminishing differentials between dominant and dominated choices.
  - Explores the possibility of collective maximum occurring with a mix of choices.
- Coalitions
  - Defines coalitions as groups making collective decisions in binary choice scenarios.
  - Discusses the existence of multiple coalitions with varying levels of formality.
  - Explains the strategic dynamics between two coalitions in a simplified game (Figure 4).
  - Introduces the concept of reaction curves and their impact on collective outcomes (Figure 5).
- Some Different Configurations
  - Explores scenarios where choice curves intersect, with implications for equilibria and collective maxima (Figures 6-9).
  - Discusses the case of “the commons” as an example of contingent externalities (Figure 10).
  - Analyzes situations with dual equilibria and their organizational challenges (Figure 11).
  - Presents MPD as a truncated dual equilibrium case (Figure 12).
- Curvatures
  - Briefly examines scenarios with non-linear choice curves, including U-shaped curves for compatibility and inverted U-shaped curves for complementarity (Figures 13-14).
  - Discusses the case of “sufficiency” with two intersections and its practical interpretations (Figure 15).
- Graduated Preferences
  - Relaxes the assumption of identical payoffs and explores scenarios with graduated preferences.
  - Analyzes cases with identical externalities and differing internalities (Figure 16).
  - Discusses scenarios with identical internalities and differing externalities (Figure 17).
  - Introduces the MIRV (Marginal Individual Right Value) curve and its significance.
  - Presents the IRC (Induced Right Choice) curve as a cumulative distribution of crossover points (Figure 18).
- More Than Two Choices
  - Briefly discusses the generalizability of the binary choice analysis to scenarios with three or more choices.
  - Highlights the differences in applying the analysis to symmetrical and asymmetrical cases.
  - Presents an example of a trinary choice with varying externality configurations (Figure 19).
- A Schematic Summary
  - Presents a schematic overview of different binary-choice payoff configurations using straight lines (Figure 20).
  - Classifies the scenarios based on equilibrium characteristics, universal preference, and collective maxima.
- Equilibria, Universal Preference, Uniformity, and Collective Maxima
  - Provides a concise classification scheme for binary choice situations with straight-line payoff curves.
  - Differentiates scenarios based on equilibrium characteristics, collective maximum positions, and preference structures.
  - Emphasizes the need to consider order, timing, reversibility of choices, information, and other factors not captured by the payoff curves alone.

## Reflections

# An error occurred.

Unable to execute JavaScript.

Figure 1: Dr. Milton Friedman explains how seat belt laws and helmet laws for adults only make sense only when citizens are government property.

While [Thomas Schelling](https://en.wikipedia.org/wiki/Thomas_Schelling) argues that regulation for helmets in hockey, might be not only necessary but also desirable.

[Milton Friedman](https://en.wikipedia.org/wiki/Milton_Friedman) says that according to Stewart Mills regulation should only be used when an individual’s actions cause harm to others. And thus that when helmet use is regulated this is a violation of personal freedom, with the implication that the regulation places social costs above individual freedom. He suggests this is a slippery slope where legislators asserts oversight over other personal choices on the grounds of social costs.

This results in a tension between when an externality should be regulated and when it should not.

![](hero-daylight-saving.png)

daylight saving

## The paper

paper

Schelling, Thomas C. 1973. “Hockey Helmets, Concealed Weapons, and Daylight Saving: A Study of Binary Choices with Externalities.” *Journal of Conflict Resolution* 17 (3): 381–428.

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2025,
  author = {Bochman, Oren},
  title = {🏒 {Hockey} {Helmets,} {Concealed} {Weapons,} and {Daylight}
    {Saving}},
  date = {2025-04-01},
  url = {https://orenbochman.github.io/reviews/1973/schelling-externalities/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2025. “🏒 Hockey Helmets, Concealed Weapons, and Daylight Saving.” April 1. <https://orenbochman.github.io/reviews/1973/schelling-externalities/>.
