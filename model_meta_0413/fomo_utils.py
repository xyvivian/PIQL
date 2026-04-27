"""
Minimal utilities shared by the FoMo-Meta transformer modules.
"""
import torch
import torch.nn as nn


class SeqBN(nn.Module):
    """Sequence Batch Normalization – applies BN independently across the sequence dimension."""

    def __init__(self, d_model: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.d_model == x.shape[-1]
        flat_x = x.view(-1, self.d_model)
        flat_x = self.bn(flat_x)
        return flat_x.view(*x.shape)


def bool_mask_to_att_mask(mask: torch.Tensor) -> torch.Tensor:
    """Convert a boolean attention mask to a float mask for nn.MultiheadAttention.
    True  → 0.0   (position can be attended to)
    False → -inf  (position is masked out)
    """
    return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
