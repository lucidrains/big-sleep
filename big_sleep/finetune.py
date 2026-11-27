"""Teach BigGAN-deep one new class (a person) from a folder of photos.

Stage 1 learns the new class vector plus one latent per photo with the generator frozen.
Stage 2 (optional) also fine-tunes the generator weights around those latents (pivotal tuning).
The result is a model folder for `dream --model <folder>`, where `<me>` in a prompt means this person.
"""
import random
from pathlib import Path

import fire
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
import torchvision.transforms as T
from torchvision.utils import save_image
from PIL import Image, ImageOps
from tqdm import trange

import big_sleep.big_sleep as bs
from big_sleep.big_sleep import PERSON_FILE, create_clip_img_transform
from big_sleep.biggan import BigGAN, WEIGHTS_NAME, CONFIG_NAME
from big_sleep.clip import load

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def load_photos(folder, image_size):
    paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    assert len(paths) > 0, f'no images ({", ".join(sorted(IMAGE_EXTENSIONS))}) found in {folder}'
    # phone photos store their rotation in EXIF
    pils = [ImageOps.exif_transpose(Image.open(p)).convert('RGB') for p in paths]
    to_tensor = T.Compose([T.Resize(image_size), T.CenterCrop(image_size), T.ToTensor()])
    return pils, torch.stack([to_tensor(im) for im in pils])


def add_class(biggan):
    # new class starts at the mean ImageNet class embedding
    old = biggan.embeddings.weight.data # (z_dim, num_classes)
    biggan.embeddings = nn.Linear(old.shape[1] + 1, old.shape[0], bias = False).to(old)
    biggan.embeddings.weight.data = torch.cat((old, old.mean(dim = 1, keepdim = True)), dim = 1)
    biggan.config.num_classes += 1
    return biggan.config.num_classes - 1


def finetune(
    images,
    out,
    image_size = 256,
    tune_generator = True,
    class_steps = 1000,
    generator_steps = 1000,
    latent_lr = .05,
    class_lr = .01,
    generator_lr = 3e-5,
    overwrite = False
):
    out = Path(out)
    assert overwrite or not (out / WEIGHTS_NAME).exists(), f'{out} already holds a model, pass --overwrite to replace it'

    pils, photos = load_photos(images, image_size)
    photos = photos.cuda()
    num_photos = len(pils)
    print(f'fine-tuning on {num_photos} photos at {image_size}px')

    biggan = BigGAN.from_pretrained(f'biggan-deep-{image_size}').cuda().eval()
    biggan.requires_grad_(False)
    class_index = add_class(biggan)

    clip, normalize = load('ViT-B/32', jit = False)
    clip.requires_grad_(False)

    clip_transform = create_clip_img_transform(224)
    with torch.no_grad():
        photo_embeds = F.normalize(clip.encode_image(torch.stack([clip_transform(im) for im in pils]).cuda()).float(), dim = -1)
    person_embed = F.normalize(photo_embeds.mean(dim = 0, keepdim = True), dim = -1)

    def embed(img):
        return F.normalize(clip.encode_image(normalize(F.interpolate(img, (224, 224), mode = 'bilinear', align_corners = False))).float(), dim = -1)

    # this fork's generator renders one image from one latent row per layer
    num_latents = len(biggan.config.layers) + 1
    classes = torch.zeros(num_latents, biggan.config.num_classes, device = 'cuda')
    classes[:, class_index] = 1.
    pivots = nn.Parameter(torch.randn(num_photos, num_latents, biggan.config.z_dim, device = 'cuda'))

    def generate(z):
        return (biggan(z, classes, 1) + 1) / 2

    def run(steps, param_groups, desc):
        optimizer = Adam(param_groups)
        pbar = trange(steps, desc = desc)
        for _ in pbar:
            if bs.terminate:
                break
            i = random.randrange(num_photos)
            recon = generate(pivots[i])
            recon_loss = F.mse_loss(recon, photos[i:i + 1]) + (1 - (embed(recon) * photo_embeds[i]).sum())
            # a random latent must also look like the person, since dream optimises the latent freely
            identity_loss = 1 - (embed(generate(torch.randn_like(pivots[i]))) * person_embed).sum()
            loss = recon_loss + identity_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description(f'{desc} recon {recon_loss.item():.3f} identity {identity_loss.item():.3f}')

    def preview(name):
        with torch.no_grad():
            shown = range(min(4, num_photos))
            rows = [photos[i] for i in shown] + [generate(pivots[i])[0] for i in shown]
            rows += [generate(torch.randn_like(pivots[0]))[0] for _ in shown]
        save_image(torch.stack(rows), out / name, nrow = len(shown))

    out.mkdir(parents = True, exist_ok = True)

    biggan.embeddings.weight.requires_grad_(True) # one-hot input, so only the new column gets gradients
    run(class_steps, [{'params': [pivots], 'lr': latent_lr}, {'params': [biggan.embeddings.weight], 'lr': class_lr}], 'class')
    preview('preview_class.png')

    if tune_generator and not bs.terminate:
        pivots.requires_grad_(False)
        biggan.requires_grad_(True)
        run(generator_steps, [{'params': biggan.parameters(), 'lr': generator_lr}], 'generator')
        preview('preview_generator.png')

    if bs.terminate:
        print('interrupted, saving the model as trained so far')

    torch.save(biggan.state_dict(), out / WEIGHTS_NAME)
    (out / CONFIG_NAME).write_text(biggan.config.to_json_string(), encoding = 'utf-8')
    torch.save({'class_index': class_index, 'clip_embedding': person_embed.half().cpu()}, out / PERSON_FILE)
    print(f'saved to {out}, previews: rows are photos, reconstructions, random samples')
    print(f'try: dream "<me> as an astronaut on mars" --model {out} --image_size {image_size}')


def main():
    fire.Fire(finetune)
