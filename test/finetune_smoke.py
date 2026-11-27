# End-to-end check: fine-tune on two fake photos for a few steps, then dream with <me>.
# Run from the repo root: python test/finetune_smoke.py
import tempfile
from pathlib import Path

import torch
from PIL import Image

from big_sleep import Imagine
from big_sleep.biggan import BigGAN
from big_sleep.finetune import finetune

with tempfile.TemporaryDirectory() as tmp:
    photos, model = Path(tmp) / 'photos', Path(tmp) / 'model'
    photos.mkdir()
    for i in range(2):
        Image.new('RGB', (300, 400), (200, 50 * i, 50)).save(photos / f'{i}.jpg')

    finetune(str(photos), str(model), class_steps = 3, generator_steps = 3)

    biggan = BigGAN.from_pretrained(str(model))
    assert biggan.config.num_classes == 1001
    assert biggan.embeddings.weight.shape == (128, 1001)
    assert (model / 'preview_generator.png').exists()

    dream = Imagine(text = '<me> on mars', model_path = str(model), image_size = 256,
                    epochs = 1, iterations = 2, save_every = 1, open_folder = False, num_cutouts = 8)
    assert dream.model.model.force_class == 1000
    _, classes = dream.model.model.get_latents()
    assert torch.all(classes[:, 1000] == 1)
    assert '<' not in str(dream.filename)
    dream()
    assert dream.filename.exists()
    dream.filename.unlink()

    dream.set_text('a lonely house')
    assert dream.model.model.force_class is None

print('finetune smoke test passed')
