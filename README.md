## Big Sleep

A simple command line tool for text to image generation, using OpenAI's CLIP and a BigGAN.

<a href="https://twitter.com/advadnoun">Ryan Murdock</a> has done it again, combining CLIP and the generator from a BigGAN. This repository wraps up his work so it is easily accessible to anyone who owns a GPU.

Original notebook [![Open In Colab][colab-badge]][colab-notebook]

[colab-notebook]: <https://colab.research.google.com/drive/1NCceX2mbiKOSlAd_o7IU7nA9UskKN5WR?usp=sharing>
[colab-badge]: <https://colab.research.google.com/assets/colab-badge.svg>

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
