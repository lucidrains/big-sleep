<img src="./samples/artificial_intelligence.png" width="384px"></img>

*artificial intelligence*

<img src="./samples/cosmic_love_and_attention.png" width="384px"></img>

*cosmic love and attention*

<img src="./samples/fire_in_the_sky.png" width="384px"></img>

*fire in the sky*

<img src="./samples/a_pyramid_made_of_ice.png" width="384px"></img>

*a pyramid made of ice*

<img src="./samples/a_lonely_house_in_the_woods.png" width="384px"></img>

*a lonely house in the woods*

<img src="./samples/marriage_in_the_mountains.png" width="384px"></img>

*marriage in the mountains*

## Big Sleep

A simple command line tool for text to image generation, using OpenAI's <a href="https://github.com/openai/CLIP">CLIP</a> and a <a href="https://arxiv.org/abs/1809.11096">BigGAN</a>.

<a href="https://twitter.com/advadnoun">Ryan Murdock</a> has done it again, combining CLIP and the generator from a BigGAN! This repository wraps up his work so it is easily accessible to anyone who owns a GPU.

Original notebook [![Open In Colab][colab-badge]][colab-notebook]

[colab-notebook]: <https://colab.research.google.com/drive/1NCceX2mbiKOSlAd_o7IU7nA9UskKN5WR?usp=sharing>
[colab-badge]: <https://colab.research.google.com/assets/colab-badge.svg>

## Install

```bash
$ pip install big-sleep
```

## Usage

```bash
$ imagine_big 'a pyramid made of ice'
```

That's it. Imagined images will be saved to whereever the command is invoked

You can set the number of latents to optimize over by setting `--num-latents`. Default is at `32`

## Advanced

You can invoke this in code with

```python
from big_sleep import Imagine

train = Imagine(
    text = 'fire in the sky',
    num_latents = 32,
    lr = 5e-2,
    save_every = 25,
    save_progress = True
)

train()
```

To save the trail of images during training, you just have to supply the `--save-progress` flag

```bash
$ imagine_big 'an owl flying over the night sky' --save-progress --save-every 100
```

## Alternatives

<a href="https://github.com/lucidrains/deep-daze">Deep Daze</a> - CLIP and a deep SIREN network

## Citations

```bibtex
@misc{unpublished2021clip,
    title  = {CLIP: Connecting Text and Images},
    author = {Alec Radford, Ilya Sutskever, Jong Wook Kim, Gretchen Krueger, Sandhini Agarwal},
    year   = {2021}
}
```

```bibtex
@misc{brock2019large,
    title   = {Large Scale GAN Training for High Fidelity Natural Image Synthesis}, 
    author  = {Andrew Brock and Jeff Donahue and Karen Simonyan},
    year    = {2019},
    eprint  = {1809.11096},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```
