## brace expansion

- the bash shell supports brace expansion.
- the idea is that the string before and after are concatenated with element in the braces

``` bash
#| label: bash brace expansion
echo "you won "{two,three,four}" points, "
```

``` bash
#| label: bash brace expansion 2
echo "you won "{1..10}" points, "
```

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2020,
  author = {Bochman, Oren},
  title = {Brace Expansion},
  date = {2020-06-12},
  url = {https://orenbochman.github.io/posts/2020/06-12-bash-tricks/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2020. “Brace Expansion.” June 12. <https://orenbochman.github.io/posts/2020/06-12-bash-tricks/>.
