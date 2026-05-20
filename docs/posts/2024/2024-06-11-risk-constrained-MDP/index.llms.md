> **CAUTION:**
>
> get a copy of the paper
>
> look for talk in the paper - not found
>
> add a citation ([Borkar and Jain 2010](#ref-Borkar2010RiskconstrainedMD))
>
> review this paper
>
> further work - perhaps [this paper](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=Km1V8WwAAAAJ&citation_for_view=Km1V8WwAAAAJ:mB3voiENLucC)

### Summary of “Risk-Constrained Markov Decision Processes”

by [Vivek Borkar](https://en.wikipedia.org/wiki/Vivek_Borkar) and [Rahul Jain](https://scholar.google.com/citations?user=NIj18UQAAAAJ&hl=en&oi=sra)

**Abstract:** The paper introduces a new framework for constrained Markov decision processes (MDPs) with risk-type constraints, specifically using Conditional Value-at-Risk (CVaR) as the risk metric. It proposes an offline iterative algorithm to find the optimal risk-constrained control policy and sketches a stochastic approximation-based learning variant, proving its convergence to the optimal policy.

**Key Concepts:**

1.  **Constrained Markov Decision Processes (CMDPs):**

    - CMDPs extend MDPs to include constraints, which can be challenging due to the complexity of handling multiple stages and constraints that have different forms.

    - Traditional methods often fail when constraints involve conditional expectations or probabilities.

2.  **Risk Measures:**

    - CVaR is used instead of the traditional Value-at-Risk (VaR) because it is a coherent risk measure and convex, making it suitable for optimization.

    - CVaR measures the expected loss given that a loss exceeds a certain value, thus addressing the shortcomings of VaR.

3.  **Problem Formulation:**

    - The objective is to maximize the expected total reward over a finite time horizon while ensuring that the CVaR of the total cost remains bounded.

    - This is particularly relevant for decision-making problems where risk management is crucial, such as finance and reinsurance.

4.  **Algorithmic Solution:**

    - An offline iterative algorithm is proposed to solve the risk-constrained MDP (rMDP) problem.

    - The algorithm operates in multiple time scales, adjusting dual variables and iterating until convergence.

    - A proof of convergence for the proposed algorithm under certain conditions is provided.

5.  **Online Learning Algorithm:**

    - An online learning variant of the algorithm uses stochastic approximation to find the optimal control policy in a sample-based manner.

    - The convergence of the online algorithm is also established, ensuring it behaves similarly to the offline algorithm.

6.  **Applications and Relevance:**

    - The framework is useful in areas where decisions must be made under uncertainty with potential catastrophic risks, such as power systems with renewable energy integration and financial risk management.

    - The paper aims to stimulate further research in risk-constrained MDPs, offering a new approach to handling risk in dynamic decision-making problems. The two algorithms are:

## Offline Iterative Algorithm (iRMDP)

\begin{algorithm} \caption{iRMDP: Offline Iterative Algorithm}\begin{algorithmic} \PROCEDURE{oRMDP}{\$\lambda^0, \beta^0\$} \STATE Initialize \$\lambda^0, \$\beta^0\$ \FOR{\$m = 1, 2, \ldots\$ until convergence} \FOR{\$n = 1, 2, \ldots\$ until convergence} \FOR{\$t = T, \ldots, 0\$} \STATE \$J\_{t}^{n,m}(x,y) = \max\_{u} \left( r(x, u) + \int \int p(dx' \mid x, u) J\_{t+1}^{n,m}(x', y + c(x, u) + s) \phi(s) ds \right)\$ \STATE \$u\_{t}^{n,m}(z) \in \arg\max\_{u} \left( r(x, u) + \int \int p(dx' \mid x, u) J\_{t+1}^{n,m}(x', y + c(x, u) + s) \phi(s) ds \right)\$ \STATE \$V\_{t}^{n,m}(z) = \int p(dz' \mid z, u\_{t}^{n,m}(z)) V\_{t+1}^{n,m}(z')\$ \STATE \$Q\_{t}^{m}(z) = \frac{c(z) V\_{t}^{n,m}(z)}{\alpha} + \int p(dz' \mid z, u\_{t}^{n,m}(z)) Q\_{t+1}^{m}(z')\$ \ENDFOR \STATE \$\beta\_{n+1}^{m} = \beta\_{n}^{m} - \gamma\_{n} (\alpha - V\_{0}^{n,m}(z_0))\$ \ENDFOR \STATE \$\lambda\_{m+1} = \left( \lambda\_{m} - \eta\_{m} (C\_{\alpha} - Q\_{0}^{m}(z_0)) \right)^+\$ \ENDFOR \PROCEDURE \end{algorithmic} \end{algorithm}

## Online Learning Algorithm (oRMDP)

\begin{algorithm} \caption{oRMDP - Online Learning Algorithm}\begin{algorithmic} \PROCEDURE{oRMDP}{\$\lambda^0, \beta^0\$} \STATE Initialize \$\lambda^0, \$\beta^0\$ \FOR{\$k = 1, 2, \ldots\$} \FOR{\$t = T, T-1, \ldots, 0\$} \STATE \$J\_{t}^k(x,y,u) = J\_{t}^{k-1}(x, y, u) + a_k I\\X_t^k = x, Y_t^k = y, u_t^k = u\\ \left( r(x, u) + \max\_{u'} J\_{t+1}^k(X\_{t+1}^k, Y\_{t+1}^k, u') - J\_{t}^{k-1}(x, y, u) \right)\$ \STATE \$u_t^k = v\_{t+1}^k = \arg\max J\_{t}^k(X_t^k, Y_t^k, \cdot)\$ \STATE \$V\_{t}^k(z) = V\_{t}^{k-1}(z) + a_k I\\Z_t^k = z\\ (V\_{t+1}^k(Z\_{t+1}^k) - V\_{t}^{k-1}(z))\$ \STATE \$Q\_{t}^k(z) = Q\_{t}^{k-1}(z) + a_k I\\Z_t^k = z\\ \left( V\_{t}^k(z) c(z) + Q\_{t+1}^k(Z\_{t+1}^k) - Q\_{t}^{k-1}(z) \right)\$ \ENDFOR \STATE \$\beta^k = \beta^{k-1} - \gamma_k (\alpha - V\_{0}^k(z_0))\$ \STATE \$\lambda^k = \left( \lambda^{k-1} - \eta_k (C\_{\alpha} - Q\_{0}^k(z_0)) \right)^+\$ \ENDFOR \ENDPROCEDURE \end{algorithmic} \end{algorithm}

### Conclusion:

The paper contributes to the field of stochastic optimization by addressing the gap in handling risk constraints in MDPs. It presents a robust algorithmic solution and lays the groundwork for future research in risk-constrained decision-making processes.

Borkar, V., and R. Jain. 2010. “Risk-Constrained Markov Decision Processes.” *49th IEEE Conference on Decision and Control (CDC)*, 2664–69.

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2024,
  author = {Bochman, Oren},
  title = {Risk-Constrained {Markov} Decision Processes},
  date = {2024-06-11},
  url = {https://orenbochman.github.io/posts/2024/2024-06-11-risk-constrained-MDP/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2024. “Risk-Constrained Markov Decision Processes.” June 11. <https://orenbochman.github.io/posts/2024/2024-06-11-risk-constrained-MDP/>.
