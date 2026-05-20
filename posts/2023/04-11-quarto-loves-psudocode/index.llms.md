# Quick Sort

\begin{algorithm} \caption{Quicksort} \begin{algorithmic} \Procedure{Quicksort}{\$A, p, r\$} \If{\$p \< r\$} \State \$q = \$ \Call{Partition}{\$A, p, r\$} \State \Call{Quicksort}{\$A, p, q - 1\$} \State \Call{Quicksort}{\$A, q + 1, r\$} \EndIf \EndProcedure \Procedure{Partition}{\$A, p, r\$} \State \$x = A\[r\]\$ \State \$i = p - 1\$ \For{\$j = p\$ \To \$r - 1\$} \If{\$A\[j\] \< x\$} \State \$i = i + 1\$ \State exchange \$A\[i\]\$ with \$A\[j\]\$ \EndIf \State exchange \$A\[i\]\$ with \$A\[r\]\$ \EndFor \EndProcedure \end{algorithmic} \end{algorithm}

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2023,
  author = {Bochman, Oren},
  title = {Quarto Loves Pseudocode},
  date = {2023-04-11},
  url = {https://orenbochman.github.io/posts/2023/04-11-quarto-loves-psudocode/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2023. “Quarto Loves Pseudocode.” April 11. <https://orenbochman.github.io/posts/2023/04-11-quarto-loves-psudocode/>.
