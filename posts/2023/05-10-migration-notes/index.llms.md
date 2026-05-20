I was able to stand on the shoulders of giants ([Rapp 2022](#ref-rapp2022ultimate)) ([Navarro 2022](#ref-navarro2022)), ([Hill 2022](#ref-hill2022)), ([Kaye 2022](#ref-kaye2022)) when I migrated this blog.

## Markdown

- Quarto’s markdown isn’t my favorite markdown implementation.
- It is based on [pandoc spec](https://pandoc.org/MANUAL.html#pandocs-markdown)

## The devil is in the details

There are lots of details that should be in the guide that are scattered all over the quarto site.

I decided that all posts should have the following fields in their front matter:

1.  title
2.  subtitle
3.  description
4.  date
5.  categories
6.  image
7.  image-description

## Virtual Environments

- are documented [here](https://quarto.org/docs/projects/virtual-environments.html#rstudio)
- ideal one can have one virtual environment for the whole site

## Lightbox Galleries

so far I used this only in the [this page](../2023-12-20-autogluon/index.llms.md)

the light box plugin was integrated into Quarto in the version 4.1 which I migrated to. I have been using light box to make notes of talks and so on. So in for this blog adding light boxes is a breeze.

All that’s really needed is to change setting in the frontmatter:

``` javascript
lightbox: true
```

which I did for all posts by adding the setting to the `_metadata.yaml` in the posts directory. And now all images default to opening within their own lightbox when clicked upon.

to disable the feature say, on a logo for example just add `.no-lightbox` css style to the image like this:

``` markdown
![caption](filename.png){.no-lightbox}
```

if you want to be able to scroll through a series of images we need to decorate each images as follows:

``` markdown
![caption](filename.png){group="my-gallery"}
```

An added bonus is that it is possible to zoom into these light-boxed images

## Extras

- the about page is based on [postcards package](https://cran.r-project.org/web/packages/postcards/readme/README.html)
- icons for navigation come from [bootstrap](https://icons.getbootstrap.com/?q=archive%3E)
- cover images are from [pexels](www.pexels.com)

### Open issues:

- can I readily integrate books and presentation into this blog ?
  - can I drop them in or do I need to build them in another repo
  - then deploy
  - then link!?
- how about embedding repls
- how about embedding shiny live apps

https://github.com/shafayetShafee

### Embedding PDF

- [plugin repo](https://github.com/jmgirard/embedpdf?tab=readme-ov-file)
- [documentation](https://jmgirard.github.io/embedpdf/example.html)

installation

``` bash
quarto add jmgirard/embedpdf
```

```
{{< pdf dummy.pdf >}}
{{< pdf dummy.pdf width=100% height=800 >}}
{{< pdf dummy.pdf border=1 >}}
{{< pdf dummy.pdf class=myclass >}}
```

Hill, Alison. 2022. “We Don’t Talk about Quarto.” April 4. <https://www.apreshill.com/blog/2022-04-we-dont-talk-about-quarto/>.

Kaye, Ella. 2022. “Welcome to My Quarto Website!” December 11. <https://ellakaye.co.uk/posts/2022-12-11_welcome-quarto>.

Navarro, Danielle. 2022. “Porting a Distill Blog to Quarto.” April 20. <https://blog.djnavarro.net/posts/2022-04-20_porting-to-quarto>.

Rapp, Albert. 2022. “The Ultimate Guide to Starting a Quarto Blog.” June 24. <https://albert-rapp.de/posts/13_quarto_blog_writing_guide/13_quarto_blog_writing_guide.html>.

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2024,
  author = {Bochman, Oren},
  title = {The {Great} {Migration}},
  date = {2024-01-30},
  url = {https://orenbochman.github.io/posts/2023/05-10-migration-notes/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2024. “The Great Migration.” January 30. <https://orenbochman.github.io/posts/2023/05-10-migration-notes/>.
