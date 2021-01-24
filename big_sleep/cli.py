import fire
from big_sleep import Imagine
from pathlib import Path

def train(
    text,
    lr = .07,
    image_size = 512,
    gradient_accumulate_every = 4,
    epochs = 20,
    iterations = 1050,
    save_every = 50,
    overwrite = False,
    save_progress = False,
    bilinear = False,
    open_folder = True,
    seed = 0,
    torch_deterministic = False
):
    print('Starting up...')

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
        open_folder = open_folder
    )

    if not overwrite and imagine.filename.exists():
        answer = input('Imagined image already exists, do you want to overwrite? (y/n) ').lower()
        if answer not in ('yes', 'y'):
            exit()

    imagine()

def main():
    fire.Fire(train)
