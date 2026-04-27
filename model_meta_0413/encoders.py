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
        



import torch
import torch.nn as nn


class MLPSeqEncoder(nn.Module):
    """Mask-aware encoder with the same constructor as the original MLPSeqEncoder.
    Input:
        x: (seq_len, batch, num_features)
    Output:
        (n_tokens, batch, emsize)
    Compared with the original version:
    - keeps the same __init__ signature for compatibility
    - does NOT use Linear(seq_len -> n_tokens)
    - supports variable effective sequence length through masking
    - ignores padded rows
    - does not use positional embeddings
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
        self.num_features = num_features
        self.emsize = emsize
        self.seq_len = seq_len          # kept for API compatibility / optional validation
        self.n_tokens = n_tokens
        self.hidden_feat = hidden_feat
        self.replace_nan_by_zero = replace_nan_by_zero

        # Row-wise feature encoder
        self.feat_proj = nn.Linear(num_features, hidden_feat)
        self.act1 = nn.GELU(approximate="none")

        # Learned token queries: one query per output token
        self.token_queries = nn.Parameter(
            torch.randn(n_tokens, 1, hidden_feat) * 0.02
        )

        # Query-to-set attention
        # batch_first=False expects (L, B, E)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_feat,
            num_heads=4,
            batch_first=False,
        )

        self.out_proj = nn.Linear(hidden_feat, emsize)

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:
                Tensor of shape (seq_len, batch, num_features)
            valid_mask:
                Optional bool mask of shape (seq_len, batch).
                True means valid row, False means padding.
                If None:
                - when replace_nan_by_zero=True, rows containing NaN are treated as invalid
                - otherwise rows that are exactly all-zero are treated as invalid
        Returns:
            Tensor of shape (n_tokens, batch, emsize)
        """
        if x.ndim != 3:
            raise ValueError(
                f"x must have shape (seq_len, batch, num_features), got {tuple(x.shape)}"
            )
        T, B, F = x.shape
        if F != self.num_features:
            raise ValueError(
                f"Expected num_features={self.num_features}, but got input with last dim {F}"
            )
        # Optional compatibility check:
        # allow T <= configured seq_len, but reject larger inputs if desired
        if T > self.seq_len:
            raise ValueError(
                f"Input seq_len={T} exceeds configured seq_len={self.seq_len}"
            )
        x_in = x
        if valid_mask is None:
            if self.replace_nan_by_zero:
                nan_mask = torch.isnan(x_in).any(dim=-1)   # (T, B)
                x_in = torch.nan_to_num(x_in, nan=0.0)
                valid_mask = ~nan_mask
            else:
                # Safe only if padding rows are exactly zero
                valid_mask = (x_in.abs().sum(dim=-1) > 0)
        if valid_mask.shape != (T, B):
            raise ValueError(
                f"valid_mask must have shape {(T, B)}, got {tuple(valid_mask.shape)}"
            )
        # 1) Row-wise feature projection
        h = self.act1(self.feat_proj(x_in))   # (T, B, hidden_feat)
        # 2) Expand learned output queries across batch
        q = self.token_queries.expand(-1, B, -1)   # (n_tokens, B, hidden_feat)
        # 3) Build padding mask for attention
        # PyTorch convention: True = ignore position
        key_padding_mask = ~valid_mask.transpose(0, 1)   # (B, T)
        # Handle rare case where all rows are invalid for one example
        all_invalid = key_padding_mask.all(dim=1)   # (B,)
        if all_invalid.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_invalid, 0] = False
            h = h.clone()
            h[0, all_invalid] = 0.0
        # 4) Learned query pooling over valid rows
        z, _ = self.attn(
            query=q,                    # (n_tokens, B, hidden_feat)
            key=h,                      # (T, B, hidden_feat)
            value=h,                    # (T, B, hidden_feat)
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )                               # (n_tokens, B, hidden_feat)
        # 5) Output projection
        out = self.out_proj(z)          # (n_tokens, B, emsize)
        return out



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