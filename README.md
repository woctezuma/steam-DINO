# Steam DINO: match Steam Banners with Facebook's DINO

This repository contains Python code to retrieve Steam games with similar store banners, using [Facebook's DINO][fb-dino-blog].

Image similarity is assessed by the cosine similarity between image features encoded by DINO.

![Similar vertical banners][wiki-cover]

## Requirements

-   Install the latest version of [Python 3.X][download-python].
-   Install the required packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data

Data is identical to the one used in [`steam-CLIP`][banner-repository-CLIP].

It consists of **vertical** Steam banners (300x450 resolution), available for 29982 out of 48792 games, i.e. 61.4% of games.

## Usage

Run [`match_steam_banners_with_DINO.ipynb`][match_steam_banners_with_DINO-notebook].
[![Open In Colab][colab-badge]][match_steam_banners_with_DINO-notebook]

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

[fb-dino-blog]: <https://ai.facebook.com/blog/dino-paws-computer-vision-with-self-supervised-transformers-and-10x-more-efficient-training>
[fb-dino-code]: <https://github.com/facebookresearch/dino>
[fb-dino-paper]: <https://arxiv.org/abs/2104.14294>

[banner-repository-CLIP]: <https://github.com/woctezuma/steam-CLIP>
[natural-language-search]: <https://github.com/woctezuma/steam-image-search>
[my-flask-API]: <https://github.com/woctezuma/heroku-flask-api>

[download-python]: <https://www.python.org/downloads/>
[colab-badge]: <https://colab.research.google.com/assets/colab-badge.svg>
