"""Minimal encoder utilities used by FoMo-Meta pretraining scripts.

This module currently provides the small subset of the original PFNS encoder
helpers that FoMo-Meta uses locally:

- `Linear`
- `Normalize`
- `get_normalized_uniform_encoder()`
"""

import math

import torch
import torch.nn as nn


class Normalize(nn.Module):
    """Apply a fixed affine normalization: `(x - mean) / std`."""

    def __init__(self, mean: float, std: float) -> None:
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class Linear(nn.Linear):
    """Linear layer with optional NaN-to-zero replacement before projection."""

    def __init__(
        self,
        num_features: int,
        emsize: int,
        replace_nan_by_zero: bool = False,
    ) -> None:
        super().__init__(num_features, emsize)
        self.num_features = num_features
        self.emsize = emsize
        self.replace_nan_by_zero = replace_nan_by_zero

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.replace_nan_by_zero:
            x = torch.nan_to_num(x, nan=0.0)
        return super().forward(x)

    def __setstate__(self, state) -> None:
        super().__setstate__(state)
        self.__dict__.setdefault('replace_nan_by_zero', True)


def get_normalized_uniform_encoder(encoder_creator):
    """Wrap an encoder for inputs sampled uniformly from `[0, 1]`.

    The wrapper normalizes the input to zero mean and unit variance before it is
    passed to `encoder_creator`.
    """

    return lambda in_dim, out_dim: nn.Sequential(
        Normalize(0.5, math.sqrt(1 / 12)),
        encoder_creator(in_dim, out_dim),
    )


__all__ = ['Linear', 'Normalize', 'get_normalized_uniform_encoder']
