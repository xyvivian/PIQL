"""
Positional encoding modules for the FoMo transformer.

All encoders follow the protocol:
    __init__(d_model, max_len=...)
    forward(x: Tensor[seq_len, bs, d_model]) -> Tensor[seq_len, bs, d_model]

Classes
-------
NoPositionalEncoding          – identity (no positional information added)
PositionalEncoding            – sinusoidal fixed encoding (Vaswani et al. 2017)
LearnedPositionalEncoding     – learned embedding per position
PairedScrambledPositionalEncodings – learned pairs, randomly permuted each forward pass
"""

import math

import torch
from torch import nn


class NoPositionalEncoding(nn.Module):
    """Pass-through: returns the input unchanged."""

    def __init__(self, d_model, max_len=None):
        super().__init__()

    def forward(self, x):
        return x


class PositionalEncoding(nn.Module):
    """Classic sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq_len, bs, d_model)
        return self.pe[:x.size(0), :] + x


class LearnedPositionalEncoding(nn.Module):
    """Learned positional embedding, one vector per position."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.max_seq_len = max_len
        self.positional_embeddings = nn.Parameter(torch.empty(max_len, d_model))
        nn.init.normal_(self.positional_embeddings, mean=0, std=d_model ** -0.5)

    def forward(self, x):
        seq_len, bs, d_model = x.shape
        assert seq_len <= len(self.positional_embeddings), \
            f'seq_len ({seq_len}) exceeds max_len ({self.max_seq_len}).'
        pos_emb = self.positional_embeddings[:seq_len]
        return pos_emb.unsqueeze(1).expand(seq_len, bs, d_model) + x


class PairedScrambledPositionalEncodings(LearnedPositionalEncoding):
    """
    Learned positional encodings stored as pairs; pairs are randomly permuted
    each forward pass so the model cannot rely on absolute order within a pair.
    """

    def forward(self, x):
        seq_len, bs, d_model = x.shape
        assert seq_len <= len(self.positional_embeddings), \
            f'seq_len ({seq_len}) exceeds max_len ({self.max_seq_len}).'
        assert len(self.positional_embeddings) % 2 == 0, \
            'max_len must be even for PairedScrambledPositionalEncodings.'

        paired_embs = self.positional_embeddings.view(
            len(self.positional_embeddings) // 2, 2, -1
        )
        pos_emb = (
            paired_embs[torch.randperm(len(paired_embs))]
            .view(*self.positional_embeddings.shape)[:seq_len]
        )
        return pos_emb.unsqueeze(1).expand(seq_len, bs, d_model) + x
