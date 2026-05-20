## Abstract

> While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.
>
> — ([Dosovitskiy et al. 2020](#ref-DBLP:journals/corr/abs-2010-11929))

## See also

- [Paper](https://arxiv.org/abs/2010.11929)
- [Code - Vision Transformer and MLP-Mixer Architectures](https://github.com/google-research/vision_transformer)
- [ICLR - Video & Slides](https://iclr.cc/virtual/2021/oral/3458)
- [Blog post](https://research.google/pubs/an-image-is-worth-16x16-words-transformers-for-image-recognition-at-scale/)
- Third-party reviews:
  - [Review by Yannic Kilcher](https://www.youtube.com/@YannicKilcher)
  - [Sahil Khose](https://www.youtube.com/watch?v=aD-D8-D-ZyY)
  - [AI Coffee Break with Letitia](https://www.youtube.com/watch?v=DVoHvmww2lQ)
  - [Manish Chablani — Review](https://medium.com/@ManishChablani/vision-transformer-vit-an-image-is-worth-16x16-words-transformers-for-image-recognition-at-a4bd5c6f17a7)

Dosovitskiy, Alexey, Lucas Beyer, Alexander Kolesnikov, et al. 2020. “An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale.” *CoRR* abs/2010.11929. <https://arxiv.org/abs/2010.11929>.

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2021,
  author = {Bochman, Oren},
  title = {ViT -\/-\/- {An} {Image} Is Worth 16x16 Words: {Transformers}
    for {Image} {Recognition} at Scale},
  date = {2021-10-22},
  url = {https://orenbochman.github.io/reviews/2020/ViT/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2021. “ViT --- An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale.” October 22. <https://orenbochman.github.io/reviews/2020/ViT/>.
