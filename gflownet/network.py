from collections.abc import Iterable

import numpy as np
import torch
from torch import nn
from einops import rearrange


def check_shape(cur_shape):
    if isinstance(cur_shape, Iterable):
        return tuple(cur_shape)
    elif isinstance(cur_shape, int):
        return tuple([cur_shape,])
    else:
        raise NotImplementedError(f"Type {type(cur_shape)} not support")

class IdentityOne:
    def __call__(self, t, y):
        del t
        return torch.ones_like(y)

class TimeConder(nn.Module):
    def __init__(self, channel, out_dim, num_layers):
        super().__init__()
        self.register_buffer(
            "timestep_coeff", torch.linspace(start=0.1, end=100, steps=channel)[None]
        )
        self.timestep_phase = nn.Parameter(torch.randn(channel)[None])
        self.layers = nn.Sequential(
            nn.Linear(2 * channel, channel),
            *[
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(channel, channel),
                )
                for _ in range(num_layers - 1)
            ],
            nn.GELU(),
            nn.Linear(channel, out_dim)
        )

        self.layers[-1].weight.data.fill_(0.0)
        self.layers[-1].bias.data.fill_(0.01)

    def forward(self, t):
        sin_cond = torch.sin((self.timestep_coeff * t.float()) + self.timestep_phase)
        cos_cond = torch.cos((self.timestep_coeff * t.float()) + self.timestep_phase)
        cond = rearrange([sin_cond, cos_cond], "d b w -> b (d w)")
        return self.layers(cond)


class FourierMLP(nn.Module):
    def __init__(self, in_shape, out_shape, num_layers=2, channels=128,
                 zero_init=True, res=False):
        super().__init__()
        self.in_shape = check_shape(in_shape) # 2 -> (2,)
        self.out_shape = check_shape(out_shape)

        self.register_buffer(
            "timestep_coeff", torch.linspace(start=0.1, end=100, steps=channels)[None]
        )
        self.timestep_phase = nn.Parameter(torch.randn(channels)[None])
        self.input_embed = nn.Linear(int(np.prod(in_shape)), channels)
        self.timestep_embed = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.layers = nn.Sequential(
            nn.GELU(),
            *[
                nn.Sequential(nn.Linear(channels, channels), nn.GELU())
                for _ in range(num_layers)
            ],
        )
        self.final_layer = nn.Linear(channels, int(np.prod(self.out_shape)))
        if zero_init:
            self.final_layer.weight.data.fill_(0.0)
            self.final_layer.bias.data.fill_(0.0)

        self.residual = res

    # cond: (1,) or (1, 1) or (bs, 1); inputs: (bs, d)
    # output: (bs, d_out)
    def forward(self, cond, inputs):
        cond = cond.view(-1, 1).expand((inputs.shape[0], 1))
        sin_embed_cond = torch.sin(
            # (1, channels) * (bs, 1) + (1, channels)
            (self.timestep_coeff * cond.float()) + self.timestep_phase
        )
        cos_embed_cond = torch.cos(
            (self.timestep_coeff * cond.float()) + self.timestep_phase
        )
        embed_cond = self.timestep_embed(
            rearrange([sin_embed_cond, cos_embed_cond], "d b w -> b (d w)")
        ) # (bs, 2* channels) -> (bs, channels)
        embed_ins = self.input_embed(inputs.view(inputs.shape[0], -1)) # (bs, d) -> (bs, channels)

        input = embed_ins + embed_cond
        out = self.layers(input)
        if self.residual:
            out = out + input
        out = self.final_layer(out) # (bs, channels) -> (bs, d)
        return out.view(-1, *self.out_shape)