# Exponential Moving Average (from https://gist.github.com/crowsonkb/76b94d5238272722290734bf4725d204)
"""Exponential moving average for PyTorch. Adapted from
https://www.zijianhu.com/post/pytorch/ema/ by crowsonkb
"""
from copy import deepcopy

import torch
from torch import nn


class EMA(nn.Module):
    def __init__(self, model, decay):
        super().__init__()
        self.model = model
        self.decay = decay
        self.register_buffer('accum', torch.tensor(1.))
        self._biased = deepcopy(self.model)
        self.average = deepcopy(self.model)
        for param in self._biased.parameters():
            param.detach_().zero_()
        for param in self.average.parameters():
            param.detach_().zero_()
        self.update()

    @torch.no_grad()
    def update(self):
        assert self.training, 'Update should only be called during training'

        self.accum *= self.decay

        model_params = dict(self.model.named_parameters())
        biased_params = dict(self._biased.named_parameters())
        average_params = dict(self.average.named_parameters())
        assert model_params.keys() == biased_params.keys() == average_params.keys(), f'Model parameter keys incompatible with EMA stored parameter keys'

        for name, param in model_params.items():
            biased_params[name].mul_(self.decay)
            biased_params[name].add_((1 - self.decay) * param)
            average_params[name].copy_(biased_params[name])
            average_params[name].div_(1 - self.accum)

        model_buffers = dict(self.model.named_buffers())
        biased_buffers = dict(self._biased.named_buffers())
        average_buffers = dict(self.average.named_buffers())
        assert model_buffers.keys() == biased_buffers.keys() == average_buffers.keys()

        for name, buffer in model_buffers.items():
            biased_buffers[name].copy_(buffer)
            average_buffers[name].copy_(buffer)

    def forward(self, *args, **kwargs):
        if self.training:
            return self.model(*args, **kwargs)
        return self.average(*args, **kwargs)
