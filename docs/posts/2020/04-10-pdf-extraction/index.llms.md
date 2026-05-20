# PDF extraction hacks

``` python
#| label: pdf-2-png
#| fig-cap: "convert pdf to png"
from pdf2image import convert_from_path
pdf_path='in.pdf'

# Store Pdf with convert_from_path function
images = convert_from_path(pdf_path)
for i in range(len(images)):
    # Save pages as images in the pdf
    images[i].save('page'+ str(i) +'.png', 'PNG')
```

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2022,
  author = {Bochman, Oren},
  title = {PDF Extraction Hacks},
  date = {2022-04-10},
  url = {https://orenbochman.github.io/posts/2020/04-10-pdf-extraction/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2022. “PDF Extraction Hacks.” April 10. <https://orenbochman.github.io/posts/2020/04-10-pdf-extraction/>.
