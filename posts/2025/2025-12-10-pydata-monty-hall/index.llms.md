[![pydata global](pydata_logo.png)](pydata_logo.png "pydata global")

pydata global

# An error occurred.

Unable to execute JavaScript.

[Eyal Kazin](https://www.linkedin.com/in/eyal-kazin/) - Lessons in Decision Making from the Monty Hall Problem

# An error occurred.

Unable to execute JavaScript.

Monty Hall Problem - Numberphile with Lisa Goldberg

# An error occurred.

Unable to execute JavaScript.

Monty Hall Problem (best explanation) - Numberphile

> **TIP:**
>
> Switch or stay, what do you say? And more importantly, why?
>
> The Monty Hall Problem is a well-known brain teaser from which we can learn important lessons in decision making that are useful in general and in particular for data scientists.
>
> If you are not familiar with this problem, prepare to be perplexed 🤯. If you are, I hope to shine light on aspects that you might not have considered 💡.
>
> I introduce the problem and solve with three types of intuitions: Common, Bayesian and Causal. I summarise with a discussion on lessons learnt for better data decision making.

Imagine you’re a contestant on a game show. Three doors stand before you: behind one is a prize car, behind the other two are goats. You choose a door, and the host—who knows what’s behind each—reveals a goat behind one of the doors you didn’t pick. Now you’re asked: “Do you want to switch your choice or stay?”

This is the essence of the Monty Hall Problem, a classic puzzle that famously baffles our intuitions about probability. While it may seem like just a fun brain teaser, it offers profound lessons for decision-making under uncertainty.

In this talk, we’ll break down the Monty Hall Problem, explore its counterintuitive nature, and uncover what it teaches us about probabilistic reasoning and critical thinking. Together, we’ll navigate multiple perspectives.

Key Topics:

- The Monty Hall Problem: Origins, setup, and why it confuses even experts
- Misconceptions and cognitive biases: Why our gut reactions often lead us astray
- Bayesian thinking: The power of belief updating in uncertain scenarios
- Information theory: How the host’s actions reveal hidden information
- Causal reasoning: A fresh lens for understanding the game’s dynamics
- Real-world takeaways: Applying these lessons to practical decision-making

> **TIP:**
>
> - A clear understanding of the Monty Hall Problem and its solution
> - Insights into the pitfalls of intuitive probability judgments
> - Strategies for approaching complex decisions and probabilistic reasoning

> **TIP:**
>
> - Basic Python and PyTorch
> - Some familiarity with neural networks (e.g., feedforward, softmax)
> - No need for prior experience in building models from scratch

This session is for data scientists, analysts, and decision-makers at all experience levels. No advanced math is required—just curiosity and a willingness to rethink what you know about probability.

Join me to discover how a seemingly trivial game show puzzle can sharpen your decision-making skills and elevate your approach to statistics, data science, and beyond.

> **TIP:**

This talk has many slides and while Kazin is a Phd level speaker the martial is accessible at a high school level. My main interest in covering this problem is that I am taking a course on probability and I wanted to use this opportunity to review some of the classical problems in depth. Also I must admit that Kazin does present a some new ideas about the problem that I never considered before!

- [slides](https://docs.google.com/presentation/d/1oXKJ5Yk9vU1b1Y7c2Yk3F2e8yX1Zx8t3/edit?usp=sharing&ouid=115234987654321234567&rtpof=true&sd=true)

- [lecture blogpost](https://medium.com/data-science/lessons-in-decision-making-from-the-monty-hall-problem-a6032f4b1032) - Warning this is a paywalled article

## Outline

Kazin discusses the [Monty Hall problem](https://en.wikipedia.org/wiki/Monty_Hall_problem), a classic probability puzzle, and extracts several lessons applicable to data-driven decision-making.

The problem’s premise: choosing one of three doors, with a prize behind one and goats behind the others. After an initial choice, the host (Monty) opens a door with a goat, and the participant is given the option to switch their choice.

Here are the key takeaways from the video:

- **Counterintuitive Probabilities**: The problem highlights that assessing probabilities can be counterintuitive. While intuition (System 1 thinking) might suggest a 50/50 chance after a door is opened, a deeper, slower analysis (System 2 thinking), like the Bayesian approach, reveals that switching doors significantly increases the probability of winning.
- **Bayesian vs. Frequentist Approaches**: Kazin contrasts the frequentist approach, which sees the two remaining doors as having equal probability, with the Bayesian approach, which incorporates prior knowledge and updates beliefs based on the host’s action. The Bayesian method, which suggests always switching, is shown to be correct.
- **Value of Information**: The speaker explains how information theory, specifically “surprisal,” can quantify the amount of information the host conveys by opening a goat door, making the “N-door problem” (with many doors) more intuitive than the three-door problem. The larger the number of doors, the more intuitive it becomes to switch.
- **Lessons for Data Science**:
  - **Clarity in Ambiguity**: Data analysts often need to infer information not explicitly stated by data providers, similar to the implicit rule that Monty opens a goat door.
  - **Updating Beliefs**: New information should lead to updated beliefs, a core tenet of Bayesian statistics.
  - **Subjectivity in Decisions**: Data analysis involves subjective decisions, such as choosing between mean or median, which need justification.
  - **Simulations Aren’t Always Necessary**: For analytically solvable problems like Monty Hall, simulations might be overkill.
  - **Effective Visualizations**: A well-designed visual can effectively explain complex concepts and persuade.
  - **Multiple Solutions**: Problems can be solved in various ways.
  - **Embrace Ignorance and Humility**: Even experts can be wrong, emphasizing the importance of humility in data science.\
- **Real-World Applications**: Kazin explores potential real-world analogies for the Monty Hall problem, distinguishing between qualitative and quantitative analogies, and cautions against oversimplification. He offers adaptive tutoring as a possible valid quantitative analogy.

## Reflections

The Monty Hall problem is usually introduced to students together with conditional probability and Bayes theorem.

But even the legendry Bayesian Physicist [David J. C. MacKay](https://www.inference.org.uk/itprnn/mackay.html) in his famous book [Information Theory, Inference, and Learning Algorithms](https://www.inference.org.uk/itprnn/book.pdf)” devotes a chapter to this problem though he does disguise it as a new problem.

In retrospect, I can see that this is a good exposition of the problem and may revise my own notes to cover classics like:

## More Classic Probability Puzzles

- [The Birthday Problem](https://en.wikipedia.org/wiki/Birthday_problem) (Richard von Mises, 1939) - How many people do you need in a room before there’s a 50% chance two share a birthday?
- [Monty Hall](https://en.wikipedia.org/wiki/Monty_Hall_problem) (1975) - Is there a principled strategy to win the game show?
- [The Card Matching problem](https://en.wikipedia.org/wiki/Matching_problem) (Montmort, 1959) - What is the probability that at least one card matching its position when a deck of size N is shuffled? cards, labelled 1 to N, is randomly arranged
- [Gambler’s Ruin](https://en.wikipedia.org/wiki/Gambler%27s_ruin) (Huygens, 1657) - What is the probability that a gambler with finite wealth will go broke against an opponent with infinite wealth?
- [Buffon’s Needle](https://en.wikipedia.org/wiki/Buffon%27s_needle) (Buffon, 1777) - Estimating \pi using a needle and a floor with parallel lines.
- [The secretary problem](https://en.wikipedia.org/wiki/Secretary_problem) (Merrill M. Flood, 1949) - How to maximize the chances of selecting the best candidate when interviewing sequentially.
- [The Two Envelopes Problem](https://en.wikipedia.org/wiki/Two_envelopes_problem) (Martin Gardner, 1953) - A paradox involving two envelopes with money and the decision to switch or stay.
- [The St. Petersburg Paradox](https://en.wikipedia.org/wiki/St._Petersburg_paradox) (Daniel Bernoulli, 1738) - A game with an infinite expected value that challenges traditional notions of utility and risk.
- [Stable matching problem](https://en.wikipedia.org/wiki/Stable_marriage_problem) (Gale and Shapley, 1962) - How to find a stable matching between two sets of elements given preferences.
- [The Two Children Problem](https://en.wikipedia.org/wiki/Boy_or_girl_paradox) (Martin Gardner, 1959)
  - Mr. Jones has two children. The older child is a girl. What is the probability that both children are girls?
  - Mr. Smith has two children. At least one of them is a boy. What is the probability that both children are boys?
- [The Three Prisoners Problem](https://en.wikipedia.org/wiki/Three_prisoners_problem) (Martin Gardner, 1959) - A paradox involving three prisoners and the probability of being pardoned after one is revealed to be executed.
- \[Sleeping Beauty problem)(https://en.wikipedia.org/wiki/Sleeping_Beauty_problem) (Adam Elga, 2000) - A philosophical puzzle about self-locating belief and probability.
- [Carter catastrophe](https://en.wikipedia.org/wiki/Doomsday_argument) (Carter, 1983) - Predict the total number of humans who will ever live.
- [Two envelopes problem](https://en.wikipedia.org/wiki/Two_envelopes_problem) (Martin Gardner, 1953) - A paradox involving two envelopes with money and the decision to switch or stay.

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2025,
  author = {Bochman, Oren},
  title = {Lessons in {Decision} {Making} from the {Monty} {Hall}
    {Problem}},
  date = {2025-12-10},
  url = {https://orenbochman.github.io/posts/2025/2025-12-10-pydata-monty-hall/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2025. “Lessons in Decision Making from the Monty Hall Problem.” December 10. <https://orenbochman.github.io/posts/2025/2025-12-10-pydata-monty-hall/>.
