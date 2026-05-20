## TLDR

I want Wikipedia style inline citations in my blog but without the massive headaches of templates. Being able to push a citation into wikidata and then cite with Qid or a DOI would be useful too.

## Requirements

Terminology:

1.  counter: a global variable which is incremented to generate the number of a citation
2.  inline citation which looks like: \[\[^(\[1\])\]\] or [(auth_title_year)]()
    1.  a link to a reference
    2.  the doietc of a citation.
    3.  a pop up on hover on the link with the citation
3.  reference list containing:
    1.  configuration for styling.
    2.  optional access to an external bib.tex file.
    3.  the full detailed text of the reference with:
        1.  an icon for the citation type
        2.  an icon for pdf
        3.  an icon for open access etc.
        4.  a link [^]() or a sequence ^ ^([a]() [b]() [c]()) of links back up each citation for the reference using a letters.
4.  label:
    1.  generated token in the using the template: `${auth}_${title}_${year}` where:
        - auth is last name, of the first author
        - title is first non stop word
        - the label is the same for repeated use
5.  id:
    1.  generated token in the using the template: `${auth}_${title}_${year}_${counter}`
    2.  the id is unique per location on the page. Ideally, the reference list would be generated using a few simple template from a data structure. The citation could hold the full citation or if the citations is already used before - just using the label This is an anonymous citation reference ^([\[1\]](#cite_note-1)) This is an named citation reference `<sup id="cite_ref-kag_5-0" class="reference"> <a href="#cite_note-kag-5">[5]</a> </sup>` These should be replaced with a markdown notation but that is out of the current scope of the project. Primarily as I am using Jekyll which forces to use of kramdown and I am unlikely to replace it. However specifying a bibtex file in the front matter might be of some interest too.

\$\] which renders to: `<ref-element cid="cite_note-kag-5">`

The reflist

# Resources

https://citationstyles.org/ https://peerj.com/articles/cs-214/?td=bl https://citation.js.org/ https://github.com/ORCID/bibtexParseJs https://github.com/ORCID

links: inline citation links: should be numbered should have a unique anchor to reference list entry should display popup of citation on hover. reference list should link back to

initial data structure eithe a list of key values or a dictionary `[ {id: citations_objext }]` each citation has an id and a label when we add a citation to list a more powerful data structure: list of objects, , citations, `[ { label: { ids : [ id1, id2, id3, ...], reference : object } } ]` with the following logic for adding look for label in list, if not found append id and reference. if found append id only. citations: • citeseerx has a list of references for a publications • sematic scholar has classification of publications by impotance.

multiple lists: wikipedia allows the uses of multiple lists on a page the football article have notes and citation. It may be of interest to add an authority list that is more of an inspiration and not to inline these items in the text. - Notes - end notes - Citations - inlined citation references - Further Reading - uncited references in wikipedia this is often just a list of internal links another matter - it is possible to use these components for notes

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2021,
  author = {Bochman, Oren},
  title = {What Is in a Citation?},
  date = {2021-08-29},
  url = {https://orenbochman.github.io/posts/2021/2021-09-22-what-is-in-a-citation/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2021. “What Is in a Citation?” August 29. <https://orenbochman.github.io/posts/2021/2021-09-22-what-is-in-a-citation/>.
