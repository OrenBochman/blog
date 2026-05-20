RL is full of big words like “off-policy learning” and “importance sampling”. If big words scare you, you are not alone. But most of the ideas in RL are simple and intuitive. So before we dive deep into mathematics and proofs, let’s try to understand these concepts in plain language.

## What is Off-Policy Learning ?

> **TIP:**
>
> off-policy learning  
> Is like trying to learn to play basketball by watching a game of football.

Let’s call the sum of all decisions we learn **a policy**. Now let’s try and make this definition more general. We want to learn a policy that is different from the policy that generated the data. Here the data is the actions chosen by the players in the game we watched.

To avoid confusion, we call the policy that generated the data the **behavior policy**, because that is how the players behaved in the game.

And the policy we want to learn, we call the **target policy** because that is the policy we want to target.

Off-policy learning  
Is a Reinforcement Learning technique where the agent learns from a behavior policy that is different from the target policy.

This is useful in reinforcement learning when the agent needs to learning from historical data or from a different agent.

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2025,
  author = {Bochman, Oren},
  title = {Off-Policy {Learning}},
  date = {2025-01-04},
  url = {https://orenbochman.github.io/posts/2025/2025-01-04-off-policy-learning/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2025. “Off-Policy Learning.” January 4. <https://orenbochman.github.io/posts/2025/2025-01-04-off-policy-learning/>.
