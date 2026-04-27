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




class MLPEncoder(nn.Module):
    """Two-layer encoder: Linear -> GELU -> Linear.
    Output dimension is always `emsize` (Transformer embedding size).
    """
    def __init__(
        self,
        num_features: int,
        emsize: int,
        hidden_dim: int | None = None,
        replace_nan_by_zero: bool = False,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else emsize
        self.num_features = num_features
        self.emsize = emsize
        self.replace_nan_by_zero = replace_nan_by_zero

        self.net = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.GELU(approximate='none'),
            nn.Linear(hidden_dim, emsize),  # <-- always emsize
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.replace_nan_by_zero:
            x = torch.nan_to_num(x, nan=0.0)
        return self.net(x)

    def __setstate__(self, state) -> None:
        super().__setstate__(state)
        self.__dict__.setdefault("replace_nan_by_zero", True)
        


class MLPSeqEncoder(nn.Module):
    """Encoder that projects along both the feature (last) and sequence (first) dimensions.

    Forward pass for input of shape ``(seq_len, batch, num_features)``:

        1. Linear(num_features → hidden_feat)  [last dim]
        2. GELU
        3. Linear(seq_len → n_tokens)           [first dim, via transpose]
        4. GELU
        5. Linear(hidden_feat → emsize)         [last dim]

    Output shape: ``(n_tokens, batch, emsize)``.
    """

    def __init__(
        self,
        num_features: int,
        emsize: int,
        seq_len: int,
        n_tokens: int,
        hidden_feat: int | None = None,
        replace_nan_by_zero: bool = False,
    ) -> None:
        super().__init__()
        hidden_feat = hidden_feat if hidden_feat is not None else emsize
        self.replace_nan_by_zero = replace_nan_by_zero

        self.feat_proj = nn.Linear(num_features, hidden_feat)
        self.act1 = nn.GELU(approximate='none')
        self.seq_proj = nn.Linear(seq_len, n_tokens)
        self.act2 = nn.GELU(approximate='none')
        self.out_proj = nn.Linear(hidden_feat, emsize)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, batch, num_features)
        if self.replace_nan_by_zero:
            x = torch.nan_to_num(x, nan=0.0)
        x = self.act1(self.feat_proj(x))           # (seq_len, batch, hidden_feat)
        x = x.permute(1, 2, 0)                     # (batch, hidden_feat, seq_len)
        x = self.act2(self.seq_proj(x))             # (batch, hidden_feat, n_tokens)
        x = x.permute(2, 0, 1)                     # (n_tokens, batch, hidden_feat)
        x = self.out_proj(x)                        # (n_tokens, batch, emsize)
        return x


def get_normalized_uniform_encoder(encoder_creator):
    """Wrap an encoder for inputs sampled uniformly from `[0, 1]`.

    The wrapper normalizes the input to zero mean and unit variance before it is
    passed to `encoder_creator`.
    """

    return lambda in_dim, out_dim: nn.Sequential(
        Normalize(0.5, math.sqrt(1 / 12)),
        encoder_creator(in_dim, out_dim),
    )


def get_normalized_uniform_seq_encoder(encoder_creator):
    """Wrap an ``MLPSeqEncoder`` (or any encoder) for inputs sampled uniformly from ``[0, 1]``.
    """
    return lambda  num_features, emsize, seq_len, n_tokens: nn.Sequential(
        Normalize(0.5, math.sqrt(1 / 12)),
        encoder_creator(num_features, emsize, seq_len, n_tokens),
    )


__all__ = [
    "Linear",
    "Normalize",
    "MLPEncoder",
    "MLPSeqEncoder",
    "get_normalized_uniform_encoder",
    "get_normalized_uniform_seq_encoder",
]