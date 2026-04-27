"""
FoMo Transformer encoder layer implementation.

Contains:
  - RouterMultiHeadAttention  – two-stage (compress → recover) attention via learned router tokens
  - TransformerEncoderLayer   – standard pre/post-norm encoder layer with optional router attention

Ported from pfns/fomo_layer.py; CustomizedMHA dependency removed (all usages were commented out).
"""
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import (
    Dropout,
    LayerNorm,
    Linear,
    Module,
    MultiheadAttention,
)
from torch.nn.modules.transformer import _get_activation_fn  # re-exported for callers
from torch.utils.checkpoint import checkpoint

from einops import repeat


# ---------------------------------------------------------------------------
# Router-based Multi-Head Attention
# ---------------------------------------------------------------------------

class RouterMultiHeadAttention(Module):
    """Two-stage attention that compresses the context through a set of learnable router tokens.

    Stages:
      1. Compress: router tokens attend to input  → compressed representation.
      2. Recover:  input attends to router tokens → full-length output.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float, batch_first: bool,
                 device=None, dtype=None, d_ff: int = None, num_R: int = 50,
                 dropout_rate: float = 0.2, **kwargs):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        d_ff = d_ff or 2 * d_model
        self.batch_first = batch_first

        # Stage 1: router tokens Q, input K/V → compressed
        self.att_compressor = MultiheadAttention(d_model, nhead, dropout=dropout,
                                                 batch_first=batch_first, **factory_kwargs)
        # Stage 2: input Q, router K/V → recovered
        self.att_recover = MultiheadAttention(d_model, nhead, dropout=dropout,
                                              batch_first=batch_first, **factory_kwargs)

        if batch_first:
            self.router = nn.Parameter(torch.randn(1, num_R, d_model))
        else:
            self.router = nn.Parameter(torch.randn(num_R, 1, d_model))

        self.dropout = nn.Dropout(dropout_rate)
        self.ln1 = nn.LayerNorm(d_model)

    def forward(self, query: Tensor, key: Tensor, value: Tensor,
                average_attn_weights: bool, skip_att: bool = False):
        """Mirrors the signature of nn.MultiheadAttention for drop-in use.

        Args:
            query / key / value: same tensor (self-attention context).
            average_attn_weights: passed through to underlying MHA.
            skip_att: if True, bypass attention and return query unchanged.
        """
        if skip_att:
            return query, None

        if self.batch_first:
            batch = query.shape[0]
            batch_router = repeat(self.router,
                                  'b_ph factor d -> (repeat b_ph) factor d',
                                  repeat=batch)
        else:
            batch = query.shape[1]
            batch_router = repeat(self.router,
                                  'factor b_ph d -> factor (repeat b_ph) d',
                                  repeat=batch)

        router, router_att_weight = self.att_compressor(
            batch_router, key, value, average_attn_weights=average_attn_weights)
        recovered_rep, recover_att_weight = self.att_recover(
            query, router, router, average_attn_weights=average_attn_weights)

        rep = query + self.dropout(recovered_rep)
        rep = self.ln1(rep)
        return rep, {'router_att': router_att_weight, 'recover_att': recover_att_weight}


# ---------------------------------------------------------------------------
# Transformer Encoder Layer
# ---------------------------------------------------------------------------

class TransformerEncoderLayer(Module):
    r"""Pre- or post-norm Transformer encoder layer with optional router attention.

    Supports three masking modes (selected via the type of ``src_mask``):

    * ``tuple``  – global-attention setup (three separate masks).
    * ``int``    – efficient eval masking; the int is the ``single_eval_pos``.
    * ``Tensor`` – standard attention mask.

    Args:
        d_model:                  model embedding dimension.
        nhead:                    number of attention heads.
        dim_feedforward:          inner dimension of the FFN (default 2048).
        dropout:                  dropout rate.
        activation:               activation function name or callable.
        pre_norm:                 if True use Pre-LayerNorm, else Post-LayerNorm.
        recompute_attn:           use gradient checkpointing for attention.
        save_trainingset_representations: cache context representations for eval.
        model_para_dict:          dict with keys ``num_R`` and ``last_layer_no_R``.
        is_final_layer:           flag passed from TransformerEncoderDiffInit.
    """

    __constants__ = ['batch_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "relu",
                 layer_norm_eps: float = 1e-5, batch_first: bool = False,
                 pre_norm: bool = False, device=None, dtype=None,
                 recompute_attn: bool = False,
                 save_trainingset_representations: bool = False,
                 model_para_dict: dict = None,
                 is_final_layer=None) -> None:
        self.src_right_att = None
        self.src_left_att = None
        self.num_R = model_para_dict['num_R']
        self.last_layer_no_R = model_para_dict['last_layer_no_R']
        self.is_final_layer = is_final_layer
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        if self.num_R is not None:
            print('using vanilla + router')
            print(f'last_layer_no_R={self.last_layer_no_R}, is_final_layer={self.is_final_layer}')
            self.router_att = RouterMultiHeadAttention(
                d_model, nhead, dropout=dropout, batch_first=batch_first,
                num_R=self.num_R, **factory_kwargs)
            self.self_attn = MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=batch_first,
                **factory_kwargs)
        else:
            print('using vanilla MHA')
            self.self_attn = MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=batch_first,
                **factory_kwargs)

        # Feed-forward network
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.pre_norm = pre_norm
        self.recompute_attn = recompute_attn
        self.save_trainingset_representations = save_trainingset_representations
        self.saved_src_to_attend_to = None

        self.activation = _get_activation_fn(activation)

        # Hooks for inspection / debugging
        self.before_ffn = None
        self.final_rep = None
        
        # Auxiliary cross-attention block
        self.aux_cross_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            **factory_kwargs,
        )

        # Separate norms for the auxiliary block
        self.aux_q_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.aux_mem_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.aux_out_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

        self.aux_dropout = Dropout(dropout)
        self.aux_att = None

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)
        self.__dict__.setdefault('save_trainingset_representations', False)

    # ------------------------------------------------------------------
    def forward(self, 
                src: Tensor, 
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                aux_memory=None,
                aux_key_padding_mask=None) -> Tensor:
        """Forward pass.

        Args:
            src:                 input sequence ``(T, B, d_model)``.
            src_mask:            attention mask – ``tuple``, ``int``, or ``Tensor``.
            src_key_padding_mask: per-batch key padding mask.
        """
        
        if self.save_trainingset_representations:
            assert isinstance(src_mask, int) and not self.training, \
                "save_trainingset_representations requires eval mode and an int src_mask"
                
        src_ = self.norm1(src) if self.pre_norm else src

        # ---- Global-attention mode (tuple of three masks) ----
        if isinstance(src_mask, tuple):
            assert not self.self_attn.batch_first
            assert src_key_padding_mask is None

            global_src_mask, trainset_src_mask, valset_src_mask = src_mask
            num_global = global_src_mask.shape[0]
            num_train = trainset_src_mask.shape[0]

            global_tok = src_[:num_global]
            train_tok = src_[num_global:num_global + num_train]
            global_and_train = src_[:num_global + num_train]
            eval_tok = src_[num_global + num_train:]

            attn = partial(checkpoint, self.self_attn) if self.recompute_attn else self.self_attn
            global_tok2 = attn(global_tok, global_and_train, global_and_train,
                               None, True, global_src_mask)[0]
            train_tok2 = attn(train_tok, global_tok, global_tok,
                              None, True, trainset_src_mask)[0]
            eval_tok2 = attn(eval_tok, src_, src_, None, True, valset_src_mask)[0]
            src2 = torch.cat([global_tok2, train_tok2, eval_tok2], dim=0)

        # ---- Efficient eval masking (int = single_eval_pos) ----
        elif isinstance(src_mask, int):
            assert src_key_padding_mask is None
            single_eval_position = src_mask
            src_to_attend_to = src_[:single_eval_position]

            if self.save_trainingset_representations:
                if single_eval_position == src_.shape[0] or single_eval_position is None:
                    self.saved_src_to_attend_to = src_to_attend_to
                elif single_eval_position == 0:
                    if self.saved_src_to_attend_to is None:
                        raise ValueError(
                            "Call with full src first to cache trainingset representations.")
                    src_to_attend_to = self.saved_src_to_attend_to
                else:
                    raise ValueError(
                        "save_trainingset_representations only supports "
                        "single_eval_position == 0 or == src.shape[0]")

            # Context (train) tokens: self-attention
            if self.num_R is not None:
                src_left, self.src_left_att = self.router_att(
                    src_[:single_eval_position],
                    src_[:single_eval_position],
                    src_[:single_eval_position],
                    average_attn_weights=False,
                    skip_att=self.last_layer_no_R and self.is_final_layer,
                )
            else:
                src_left, self.src_left_att = self.self_attn(
                    src_[:single_eval_position],
                    src_[:single_eval_position],
                    src_[:single_eval_position],
                    average_attn_weights=False,
                )

            # Query tokens: cross-attention into context
            src_right, self.src_right_att = self.self_attn(
                src_[single_eval_position:], src_to_attend_to, src_to_attend_to,
                average_attn_weights=False,
            )
            src2 = torch.cat([src_left, src_right], dim=0)

        # ---- Standard attention mask ----
        else:
            if self.recompute_attn:
                src2 = checkpoint(self.self_attn, src_, src_, src_,
                                  src_key_padding_mask, True, src_mask)[0]
            else:
                src2 = self.self_attn(src_, src_, src_,
                                      attn_mask=src_mask,
                                      key_padding_mask=src_key_padding_mask)[0]

        src = src + self.dropout1(src2)
        if not self.pre_norm:
            src = self.norm1(src)
            
        # ------------------------------------------------------------
        # Auxiliary cross-attention block
        # src:        (T, B, d_model)
        # aux_memory: (M, B, d_model)
        # ------------------------------------------------------------
        if aux_memory is not None:
            if self.pre_norm:
                # Pre-norm style:
                # normalize queries and memory before attention
                q = self.aux_q_norm(src)
                kv = self.aux_mem_norm(aux_memory)
                aux_out, self.aux_att = self.aux_cross_attn(
                    q, kv, kv,
                    key_padding_mask=aux_key_padding_mask,
                    average_attn_weights=False,
                )
                src = src + self.aux_dropout(aux_out)
            else:
                # Post-norm style:
                # attend first, then residual add, then normalize output
                aux_out, self.aux_att = self.aux_cross_attn(
                    src, aux_memory, aux_memory,
                    key_padding_mask=aux_key_padding_mask,
                    average_attn_weights=False,
                )
                src = src + self.aux_dropout(aux_out)
                src = self.aux_out_norm(src)
        src_ = self.norm2(src) if self.pre_norm else src
        self.before_ffn = src_
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src_))))
        src = src + self.dropout2(src2)
        if not self.pre_norm:
            src = self.norm2(src)
            
        self.final_rep = src
        return src
