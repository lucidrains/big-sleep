import fire
import random as rnd
from big_sleep import Imagine
from pathlib import Path

def train(
    text,
    lr = .07,
    image_size = 512,
    gradient_accumulate_every = 1,
    epochs = 20,
    iterations = 1050,
    save_every = 50,
    overwrite = False,
    save_progress = False,
    save_date_time = False,
    bilinear = False,
    open_folder = True,
    seed = 0,
    random = False,
    torch_deterministic = False,
    max_classes = None,
    class_temperature = 2.,
    save_best = False
):
    print('Starting up...')

    if random:
        seed = rnd.randint(0, 1e6)

    imagine = Imagine(
        text,
        lr = lr,
        image_size = image_size,
        gradient_accumulate_every = gradient_accumulate_every,
        epochs = epochs,
        iterations = iterations,
        save_every = save_every,
        save_progress = save_progress,
        bilinear = bilinear,
        seed = seed,
        torch_deterministic = torch_deterministic,
        open_folder = open_folder,
        max_classes = max_classes,
        class_temperature = class_temperature,
        save_date_time = save_date_time,
        save_best = save_best
    )

    if not overwrite and imagine.filename.exists():
        answer = input('Imagined image already exists, do you want to overwrite? (y/n) ').lower()
        if answer not in ('yes', 'y'):
            exit()

    imagine()

def main():
    fire.Fire(train)
