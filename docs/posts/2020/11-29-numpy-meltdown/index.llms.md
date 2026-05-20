> **NOTE:**
>
> Just a rant at `numpy` and `scipy` breaking when I really needed them.

## Numpy meltdown

Sad to report but `numpy` (and `scipy`) installation started to fail on macos due to ending of support of the native Accelerate library provided by Apple.

`numpy` now depends on `lapack/blas` which is not provided by Apple and needs to be installed separately.

Of course having upgraded to macos 11 has not made it any easier to get things working smoothly either.

``` bash
brew install numpy --with-openblas
```

no longer works either since the option was removed.

the main point is how shoddy python really is - everything most people do depends on numpy but numpy is a totally different project and have no qualms about their project breaking on a major platform and provide no fall back and no support.

And why does one need numpy in the first place - lack of support in python for numeric processing and essential data structures. And being so slow that one requires it be done by a library written in a lower level language.

Just saying these should never break - since they do the very basic processes of working with python are broken.

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2020,
  author = {Bochman, Oren},
  title = {Numpy Meltdown},
  date = {2020-11-29},
  url = {https://orenbochman.github.io/posts/2020/11-29-numpy-meltdown/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2020. “Numpy Meltdown.” November 29. <https://orenbochman.github.io/posts/2020/11-29-numpy-meltdown/>.
