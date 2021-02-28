import time
import shutil
import torch
from big_sleep import Imagine

terminate = False

def signal_handling(signum,frame):
    global terminate
    terminate = True

num_attempts = 4
for attempt in range(num_attempts):
    dream = Imagine(
        text = "an armchair in the form of pikachu\\an armchair imitating pikachu\\abstract",
        text_min = "blur\\zoom",
        lr = 7e-2,
        image_size = 512,
        gradient_accumulate_every = 1,
        save_every = 50,
        epochs = 5,
        iterations = 50,
        save_progress = False,
        bilinear = False,
        open_folder = False,
        seed = None,
        torch_deterministic = False,
        max_classes = 20,
        class_temperature = 2.,
        save_date_time = False,
        save_best = True,
        experimental_resample = True,
        ema_decay = 0.99
    )
    dream()
    shutil.copy(dream.textpath + ".best.png", f"{attempt}.png")
    try:
        time.sleep(2)
        del dream
        time.sleep(2)
        torch.cuda.empty_cache()
    except Exception:
        torch.cuda.empty_cache()