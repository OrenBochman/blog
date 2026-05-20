[![google colab](colab.jpg)](colab.jpg "google colab")

google colab

So the trick is

- to use `--NotebookApp.allow_origin` and `--no-browser`
- and get the token from the command line when connecting to Google colab.

&nbsp;

    jupyter notebook --NotebookApp.allow_origin='https://colab.research.google.com' \
      --port=9090 --no-browser

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2020,
  author = {Bochman, Oren},
  title = {How to Avoid Cross Site Scripting {(XSS)} Errors},
  date = {2020-02-20},
  url = {https://orenbochman.github.io/posts/2020/02-20-avoid-cross-site-scriptin-errors-with-a-Jupyter-local-runtime/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2020. “How to Avoid Cross Site Scripting (XSS) Errors.” February 20. <https://orenbochman.github.io/posts/2020/02-20-avoid-cross-site-scriptin-errors-with-a-Jupyter-local-runtime/>.
