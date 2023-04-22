# Steam DINO: match Steam Banners with FB's DINO

This repository contains Python code to retrieve Steam games with similar store banners, using [Facebook's DINO][fb-dino-blog].

Image similarity is assessed by the cosine similarity between image features encoded by DINO.

![Similar vertical banners][wiki-cover]

## Model

DINO is a method to train self-supervised models, especially well-suited for Vision Transformers (ViT).
Model checkpoints were pre-trained on ImageNet-1k (1.28M images with 1000 classes) **with no label**.

In this repository, image features are extracted:
- following [different strategies][issue-feature-extraction],
- and based on different models (`ViT-S/16`, `ViT-S/8`, `ViT-B/16`, `ViT-B/8`).

## Data

Data is identical to the one used in [`steam-CLIP`][banner-repository-CLIP].

It consists of **vertical** Steam banners (300x450 resolution), available for 29982 out of 48792 games, i.e. 61.4% of games.

### Pre-processing

Images are resized to 224x224 resolution and available in an archive (703 MB) [as a release][github-input-data] in this repository.

However, DINO has its own [pre-processing pipeline][dino-pre-process],  as in [`eval_linear.py`][dino-linear] and [`eval_knn.py`][dino-knn]:
- resize to 256 resolution,
- center-crop at 224 resolution,
- normalize intensity.

```python
preprocess = pth_transforms.Compose(
    [
        pth_transforms.Resize(
            256, interpolation=pth_transforms.InterpolationMode.BICUBIC
        ),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)
```

Therefore, it would have been better:
- either to use 256 resolution for the input,
- or to use 224 resolution (as I did) but without resizing-then-center-cropping when calling DINO.

This is the case for [`eval_copy_detection.py`][dino-copy-detection]:

```python
transform = pth_transforms.Compose([ 
    pth_transforms.Resize((args.imsize, args.imsize), interpolation=3), 
    pth_transforms.ToTensor(), 
    pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), 
])
```

This is also the case for [`eval_image_retrieval.py`][dino-image-retrieval]:

```python
transform = pth_transforms.Compose([ 
    pth_transforms.ToTensor(), 
    pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), 
])
```

## Usage

Run [`match_steam_banners_with_DINO.ipynb`][match_steam_banners_with_DINO-notebook].
[![Open In Colab][colab-badge]][match_steam_banners_with_DINO-notebook]

## Results

Results were obtained in different settings identified by a suffix, e.g. `ComplexB8`, where:
-   `Simple` stands for the simple feature extraction, similar to the code in [`eval_knn.py`][dino-knn],
-   `Complex` stands for the complex feature extraction, similar to the code in [`eval_linear.py`][dino-linear],
-   `B8` for ViT-B/8: the `Base` architecture with patch resolution `8`.

If we look for trucks in banners similar to *Euro Truck Simulator 2*'s banner, results are:
- similar for `Simple` and `Complex`,
- more satisfactory with `B16` and `S8`, compared to `B8` or `S16`,
- slightly more satisfacfory with `B16` compared to `S8`.

Qualitatively, I would rank the strategies, starting with the most satisfactory one:
1) `SimpleB16`
2) `ComplexB16`
3) `ComplexS8`
4) `SimpleS8`
5) `SimpleB8`
6) `ComplexB8`
7) `ComplexS16`
8) `SimpleS16`

The ranking is compatible with the performance [observed in the paper for k-NN task][github-issue-knn]:
> `ViT-S/16` < `ViT-B/16` < `ViT-B/8` < `ViT-S/8`

with the exception that `B/16` seems to be the best performing model.

The fact that the `B/8` model seems under-performing is not surprising, as its hyperparameters [could have been further optimized][github-issue-b8].

[github-issue-knn]: <https://github.com/facebookresearch/dino/issues/13#issuecomment-857469740>
[github-issue-b8]: <https://github.com/facebookresearch/dino/issues/13#issuecomment-832617172>

## Perspectives

Other strategies for the creation of the image embedding would include:
- the concatenation of features extracted [at multiple scales][dino-multi-scale],
- the concatenation of the [CLS] token with [GeM pooled patch tokens][dino-gem-pooling], as for [copy detection][dino-copy-detection].

## References

-   Facebook's DINO:
    - [Official blog post][fb-dino-blog]
    - [Official Github repository][fb-dino-code]
    - [Caron, Mathilde, et al. *Emerging Properties in Self-Supervised Vision Transformers*. arXiv 2021.][fb-dino-paper] 
-   My usage of OpenAI's CLIP:
    - [`steam-CLIP`][banner-repository-CLIP]: retrieve games with similar banners, using OpenAI's CLIP (resolution 224),
    - [`steam-image-search`][natural-language-search]: retrieve games using natural language queries,
    - [`heroku-flask-api`][my-flask-API]: serve the matching results through an API built with Flask on Heroku.

<!-- Definitions -->

[wiki-cover]: <https://github.com/woctezuma/steam-DINO/wiki/img/illustration.jpg>
[match_steam_banners_with_DINO-notebook]: <https://colab.research.google.com/github/woctezuma/steam-DINO/blob/main/match_steam_banners_with_DINO.ipynb>

[issue-feature-extraction]: <https://github.com/facebookresearch/dino/issues/72>

[fb-dino-blog]: <https://ai.facebook.com/blog/dino-paws-computer-vision-with-self-supervised-transformers-and-10x-more-efficient-training>
[fb-dino-code]: <https://github.com/facebookresearch/dino>
[fb-dino-paper]: <https://arxiv.org/abs/2104.14294>

[banner-repository-CLIP]: <https://github.com/woctezuma/steam-CLIP>
[natural-language-search]: <https://github.com/woctezuma/steam-image-search>
[my-flask-API]: <https://github.com/woctezuma/heroku-flask-api>

[github-input-data]: <https://github.com/woctezuma/steam-DINO/releases/tag/input>
[dino-pre-process]: <https://github.com/woctezuma/match-steam-banners/blob/0c752609cac64448d874340abbaeb6d337f3e8ba/dino_utils.py#L165-L179>
[dino-linear]: <https://github.com/facebookresearch/dino/blob/main/eval_linear.py>
[dino-multi-scale]: <https://github.com/facebookresearch/dino/blob/ba9edd18db78a99193005ef991e04d63984b25a8/utils.py#L795-L809>
[dino-gem-pooling]: <https://github.com/facebookresearch/dino/blob/ba9edd18db78a99193005ef991e04d63984b25a8/eval_copy_detection.py#L166-L175>
[dino-copy-detection]: <https://github.com/facebookresearch/dino/blob/main/eval_copy_detection.py>
[dino-image-retrieval]: <https://github.com/facebookresearch/dino/blob/ba9edd18db78a99193005ef991e04d63984b25a8/eval_image_retrieval.py#L106-L109>
[dino-knn]: <https://github.com/facebookresearch/dino/blob/main/eval_knn.py>

[colab-badge]: <https://colab.research.google.com/assets/colab-badge.svg>
