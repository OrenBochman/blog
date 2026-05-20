BRAT is a rapid annotation tool. Its old no long maintained - the original developers seem to have moved on. Requests for RTL support have been largely ignored. Some advantages:

1.  Small code base (Python backend via cgi !? and JQuery front end)
2.  Fairly clean
3.  based on stav projecter
4.  nice data model
5.  live embedding http://brat.nlplab.org/embed.html
6.  many annotation types.

looking into modernising the UI to work with RTL.

Get brat in a docker container.

- https://hub.docker.com/r/cassj/brat/

## plan A: Recreate the svg element using d3.js

## plan B

Breaking up the element into smaller web-components in polymer. Organize things better using using modern javascript

## Brat and RTL support

# references

- https://github.com/nlplab/brat
- https://github.com/UniversalDependencies/docs/issues/52
- https://github.com/nlplab/brat/issues/774
- https://github.com/spyysalo/conllu.js/issues/11
- https://github.com/nlplab/brat/pull/1150

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2018,
  author = {Bochman, Oren},
  title = {Text Annotation with {BRAT}},
  date = {2018-01-16},
  url = {https://orenbochman.github.io/posts/2018/2018-01-16-BRAT/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2018. “Text Annotation with BRAT.” January 16. <https://orenbochman.github.io/posts/2018/2018-01-16-BRAT/>.
