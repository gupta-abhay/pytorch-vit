# Vision Transformers

Implementation of [Vision Transformer](https://openreview.net/forum?id=YicbFdNTTy) in PyTorch, a new model to achieve SOTA in vision classification with using transformer style encoders. Associated [blog](https://abhaygupta.dev/blog/vision-transformer) article.

> Credits to Phil Wang for the gif
![ViT](./static/vit.gif)

## Features

- [x] ViT
- [x] ViT with convolutional patches
- [x] ViT with convolutional stems
  - [x] Early Convolutional Stem
  - [x] Scaled ReLU Stem
- [X] GAP Pooling

## Citations

```BibTeX
@article{dosovitskiy2020image,
  title={An image is worth 16x16 words: Transformers for image recognition at scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}
```

```BibTeX
@article{xiao2021early,
  title={Early convolutions help transformers see better},
  author={Xiao, Tete and Singh, Mannat and Mintun, Eric and Darrell, Trevor and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv preprint arXiv:2106.14881},
  year={2021}
}
```

```BibTeX
@article{wang2021scaled,
  title={Scaled ReLU Matters for Training Vision Transformers},
  author={Wang, Pichao and Wang, Xue and Luo, Hao and Zhou, Jingkai and Zhou, Zhipeng and Wang, Fan and Li, Hao and Jin, Rong},
  journal={arXiv preprint arXiv:2109.03810},
  year={2021}
}
```

```BibTeX
@article{zhai2021scaling,
  title={Scaling vision transformers},
  author={Zhai, Xiaohua and Kolesnikov, Alexander and Houlsby, Neil and Beyer, Lucas},
  journal={arXiv preprint arXiv:2106.04560},
  year={2021}
}
```
