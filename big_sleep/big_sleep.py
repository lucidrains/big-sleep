import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

import torchvision
from torchvision.utils import save_image

import os
import sys
import subprocess
import signal
from pathlib import Path
from tqdm import trange
from collections import namedtuple

from big_sleep.biggan import BigGAN
from big_sleep.clip import load, tokenize, normalize_image

from einops import rearrange

assert torch.cuda.is_available(), 'CUDA must be available in order to use Deep Daze'

# graceful keyboard interrupt

terminate = False

def signal_handling(signum,frame):
    global terminate
    terminate = True

signal.signal(signal.SIGINT,signal_handling)

# helpers

def exists(val):
    return val is not None

def open_folder(path):
    if os.path.isfile(path):
        path = os.path.dirname(path)

    if not os.path.isdir(path):
        return

    cmd_list = None
    if sys.platform == 'darwin':
        cmd_list = ['open', '--', path]
    elif sys.platform == 'linux2' or sys.platform == 'linux':
        cmd_list = ['xdg-open', path]
    elif sys.platform in ['win32', 'win64']:
        cmd_list = ['explorer', path.replace('/','\\')]
    if cmd_list == None:
        return

    try:
        subprocess.check_call(cmd_list)
    except subprocess.CalledProcessError:
        pass
    except OSError:
        pass

# load clip

perceptor, preprocess = load()

# load biggan

class Latents(torch.nn.Module):
    def __init__(self, num_latents = 32):
        super().__init__()
        self.normu = torch.nn.Parameter(torch.zeros(num_latents, 128).normal_(std = 1))
        self.cls = torch.nn.Parameter(torch.zeros(num_latents, 1000).normal_(mean = -3.9, std = .3))
        self.register_buffer('thresh_lat', torch.tensor(1))

    def forward(self):
        return self.normu, torch.sigmoid(self.cls)

class Model(nn.Module):
    def __init__(self, image_size):
        super().__init__()
        assert image_size in (128, 256, 512), 'image size must be one of 128, 256, or 512'
        self.biggan = BigGAN.from_pretrained(f'biggan-deep-{image_size}')
        self.init_latents()

    def init_latents(self):
        self.latents = Latents()

    def forward(self):
        self.biggan.eval()
        out = self.biggan(*self.latents(), 1)
        return (out + 1) / 2

# load siren

class BigSleep(nn.Module):
    def __init__(
        self,
        num_cutouts = 128,
        loss_coef = 100,
        image_size = 512,
        bilinear = False
    ):
        super().__init__()
        self.loss_coef = loss_coef
        self.image_size = image_size
        self.num_cutouts = num_cutouts

        self.interpolation_settings = {'mode': 'bilinear', 'align_corners': False} if bilinear else {'mode': 'nearest'}

        self.model = Model(
            image_size = image_size
        )

    def reset(self):
        self.model.init_latents()

    def forward(self, text, return_loss = True):
        width, num_cutouts = self.image_size, self.num_cutouts

        out = self.model()

        if not return_loss:
            return out

        pieces = []
        for ch in range(num_cutouts):
            size = int(width * torch.zeros(1,).normal_(mean=.8, std=.3).clip(.5, .95))
            offsetx = torch.randint(0, width - size, ())
            offsety = torch.randint(0, width - size, ())
            apper = out[:, :, offsetx:offsetx + size, offsety:offsety + size]
            apper = F.interpolate(apper, (224, 224), **self.interpolation_settings)
            pieces.append(apper)

        into = torch.cat(pieces)
        into = normalize_image(into)

        image_embed = perceptor.encode_image(into)
        text_embed = perceptor.encode_text(text)

        latents, soft_one_hot_classes = self.model.latents()
        num_latents = latents.shape[0]
        latent_thres = self.model.latents.thresh_lat

        lat_loss =  torch.abs(1 - torch.std(latents, dim=1)).mean() + \
                    torch.abs(torch.mean(latents)).mean() + \
                    4 * torch.max(torch.square(latents).mean(), latent_thres)

        for array in latents:
            mean = torch.mean(array)
            diffs = array - mean
            var = torch.mean(torch.pow(diffs, 2.0))
            std = torch.pow(var, 0.5)
            zscores = diffs / std
            skews = torch.mean(torch.pow(zscores, 3.0))
            kurtoses = torch.mean(torch.pow(zscores, 4.0)) - 3.0

        lat_loss = lat_loss + torch.abs(kurtoses) / num_latents + torch.abs(skews) / num_latents
        cls_loss = ((50 * torch.topk(soft_one_hot_classes, largest = False, dim = 1, k = 999)[0]) ** 2).mean()

        sim_loss = -self.loss_coef * torch.cosine_similarity(text_embed, image_embed, dim = -1).mean()
        return (lat_loss, cls_loss, sim_loss)

class Imagine(nn.Module):
    def __init__(
        self,
        text,
        *,
        lr = .07,
        image_size = 512,
        gradient_accumulate_every = 1,
        save_every = 50,
        epochs = 20,
        iterations = 1050,
        save_progress = False,
        bilinear = False,
        open_folder = True,
        seed = None
    ):
        super().__init__()

        if exists(seed):
            assert not bilinear, 'the deterministic (seeded) operation does not work with interpolation, yet (ask pytorch)'
            torch.set_deterministic(True)
            torch.manual_seed(seed)

        self.epochs = epochs
        self.iterations = iterations

        model = BigSleep(
            image_size = image_size,
            bilinear = bilinear
        ).cuda()

        self.model = model

        self.lr = lr
        self.optimizer = Adam(model.model.latents.parameters(), lr)
        self.gradient_accumulate_every = gradient_accumulate_every
        self.save_every = save_every

        self.save_progress = save_progress
        self.open_folder = open_folder

        self.set_text(text)

    def set_text(self, text):
        self.text = text
        textpath = self.text.replace(' ','_')

        self.textpath = textpath
        self.filename = Path(f'./{textpath}.png')
        self.encoded_text = tokenize(text).cuda()

    def reset(self):
        self.model.reset()
        self.optimizer = Adam(self.model.model.latents.parameters(), self.lr)

    def train_step(self, epoch, i):
        total_loss = 0

        for _ in range(self.gradient_accumulate_every):
            losses = self.model(self.encoded_text)
            loss = sum(losses) / self.gradient_accumulate_every
            total_loss += loss
            loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        if (i + 1) % self.save_every == 0:
            with torch.no_grad():
                best = torch.topk(losses[2], k = 1, largest = False)[1]
                image = self.model.model()[best].cpu()
                save_image(image, str(self.filename))
                print(f'image updated at "./{str(self.filename)}"')

                if self.save_progress:
                    total_iterations = epoch * self.iterations + i
                    num = total_iterations // self.save_every
                    save_image(image, Path(f'./{self.textpath}.{num}.png'))

        return total_loss

    def forward(self):
        print(f'Imagining "{self.text}" from the depths of my weights...')

        if self.open_folder:
            open_folder('./')
            self.open_folder = False

        for epoch in trange(self.epochs, desc = 'epochs'):
            pbar = trange(self.iterations, desc='iteration')
            for i in pbar:
                loss = self.train_step(epoch, i)
                pbar.set_description(f'loss: {loss.item():.2f}')

                if terminate:
                    print('detecting keyboard interrupt, gracefully exiting')
                    return
