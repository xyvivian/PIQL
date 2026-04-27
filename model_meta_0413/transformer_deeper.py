"""
FoMo Transformer model.

Contains:
  - TransformerModel            – the main sequence-to-sequence transformer for FoMo
  - TransformerEncoderDiffInit  – stack of encoder layers with independent weight initialisation

Sibling modules (transformer_layer, fomo_utils) are loaded from the same directory via a
sys.path insertion so this file stays importable even when the directory name contains hyphens.
"""
# Ensure sibling modules in FoMo-Meta are importable regardless of how this file is loaded.
import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, TransformerEncoder

from transformer_layer_deeper import TransformerEncoderLayer, _get_activation_fn
from fomo_utils import SeqBN, bool_mask_to_att_mask


# ---------------------------------------------------------------------------
# TransformerModel
# ---------------------------------------------------------------------------

class TransformerModel(nn.Module):
    """Prior-fitted transformer for FoMo anomaly detection.

    Accepts context (train) tokens and query (test) tokens in a single sequence
    and produces per-token output logits via learnable decoder heads.

    Key design choices
    ------------------
    * Efficient eval masking: query tokens cross-attend to context only (no
      leakage), controlled by passing an ``int`` src_mask equal to
      ``single_eval_pos``.
    * Multiple decoder heads: ``decoder_dict`` maps head names to output
      projections; ``decoder_once_dict`` provides "global" output tokens
      appended to the sequence.
    * Optional global attention tokens: prepended learnable embeddings that
      every token can attend to, enabling O(n) complexity variants.
    """

    def __init__(self, encoder, ninp: int, nhead: int, nhid: int, nlayers: int,
                 dropout: float = 0.0, style_encoder=None, y_encoder=None, internal_feature_encoder=None,
                 input_to_internal_encoder=None,
                 pos_encoder=None, decoder_dict: dict = None,
                 input_normalization: bool = False, init_method=None,
                 pre_norm: bool = False, activation: str = 'gelu',
                 recompute_attn: bool = False, num_global_att_tokens: int = 0,
                 full_attention: bool = False, all_layers_same_init: bool = False,
                 efficient_eval_masking: bool = True, decoder_once_dict: dict = None,
                 return_all_outputs: bool = False,
                 save_trainingset_representations: bool = False,
                 model_para_dict: dict = None):
        super().__init__()
        self.model_type = 'Transformer'
        self.model_para_dict = model_para_dict

        # Build encoder layers
        def _make_layer(is_final_layer):
            return TransformerEncoderLayer(
                ninp, nhead, nhid, dropout,
                activation=activation,
                pre_norm=pre_norm,
                recompute_attn=recompute_attn,
                save_trainingset_representations=save_trainingset_representations,
                model_para_dict=model_para_dict,
                is_final_layer=is_final_layer,
            )

        if all_layers_same_init:
            self.transformer_encoder = TransformerEncoder(_make_layer(is_final_layer=None), nlayers)
        else:
            self.transformer_encoder = TransformerEncoderDiffInit(_make_layer, nlayers)

        self.ninp = ninp
        self.encoder = encoder
        self.y_encoder = y_encoder
        self.pos_encoder = pos_encoder
        self.return_all_outputs = return_all_outputs
        self.internal_feature_encoder = internal_feature_encoder
        self.input_to_internal_encoder = input_to_internal_encoder

        # Build decoder head(s)
        def _make_decoder_dict(desc: dict):
            if not desc:
                return None
            heads = {}
            for key, (model_cls, n_out) in desc.items():
                if model_cls is None:
                    heads[key] = nn.Sequential(
                        nn.Linear(ninp, nhid),
                        nn.GELU(),
                        nn.Linear(nhid, n_out),
                    )
                else:
                    heads[key] = model_cls(ninp, nhid, n_out)
                print(f'Initialized decoder "{key}" with {desc[key]}, nout={n_out}')
            return nn.ModuleDict(heads)

        self.decoder_dict = _make_decoder_dict(decoder_dict)
        self.decoder_dict_once = _make_decoder_dict(decoder_once_dict)
        self.decoder_dict_once_embeddings = (
            nn.Parameter(torch.randn(len(self.decoder_dict_once), 1, ninp))
            if self.decoder_dict_once is not None else None
        )

        self.input_ln = SeqBN(ninp) if input_normalization else None
        self.style_encoder = style_encoder
        self.init_method = init_method

        if num_global_att_tokens:
            assert not full_attention, "Cannot combine global_att_tokens with full_attention"
        self.global_att_embeddings = (
            nn.Embedding(num_global_att_tokens, ninp) if num_global_att_tokens else None
        )
        self.full_attention = full_attention
        self.efficient_eval_masking = efficient_eval_masking
        self.nhid = nhid

        self.init_weights()

    # ------------------------------------------------------------------
    # Backward-compat state restoration
    # ------------------------------------------------------------------
    def __setstate__(self, state):
        super().__setstate__(state)
        self.__dict__.setdefault('efficient_eval_masking', False)
        if not hasattr(self, 'decoder_dict_once'):
            self.__dict__.setdefault('decoder_dict_once', None)
        if hasattr(self, 'decoder') and not hasattr(self, 'decoder_dict'):
            self.add_module('decoder_dict', nn.ModuleDict({'standard': self.decoder}))
        self.__dict__.setdefault('return_all_outputs', False)

        def _fix_gelu(module):
            if isinstance(module, nn.GELU):
                module.__dict__.setdefault('approximate', 'none')

        self.apply(_fix_gelu)

    # ------------------------------------------------------------------
    # Attention mask factories
    # ------------------------------------------------------------------
    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        return bool_mask_to_att_mask(mask)

    @staticmethod
    def generate_D_q_matrix(sz: int, query_size: int) -> Tensor:
        train_size = sz - query_size
        mask = torch.zeros(sz, sz) == 0
        mask[:, train_size:].zero_()
        mask |= torch.eye(sz) == 1
        return bool_mask_to_att_mask(mask)

    @staticmethod
    def generate_global_att_query_matrix(num_global: int, seq_len: int,
                                         num_query: int) -> Tensor:
        train_size = seq_len + num_global - num_query
        sz = seq_len + num_global
        mask = torch.zeros(num_query, sz) == 0
        mask[:, train_size:].zero_()
        mask[:, train_size:] |= torch.eye(num_query) == 1
        return bool_mask_to_att_mask(mask)

    @staticmethod
    def generate_global_att_trainset_matrix(num_global: int, seq_len: int,
                                             num_query: int) -> Tensor:
        trainset_size = seq_len - num_query
        mask = torch.zeros(trainset_size, num_global) == 0
        return bool_mask_to_att_mask(mask)

    @staticmethod
    def generate_global_att_globaltokens_matrix(num_global: int, seq_len: int,
                                                 num_query: int) -> Tensor:
        mask = torch.zeros(num_global, num_global + seq_len - num_query) == 0
        return bool_mask_to_att_mask(mask)

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------
    def init_weights(self):
        num_R = self.model_para_dict['num_R']

        def _init_router(layer):
            for group in [layer.router_att.att_compressor, layer.router_att.att_recover]:
                attns = group if isinstance(group, nn.ModuleList) else [group]
                for attn in attns:
                    nn.init.zeros_(attn.out_proj.weight)
                    nn.init.zeros_(attn.out_proj.bias)

        if self.init_method is not None:
            self.apply(self.init_method)

        for layer in self.transformer_encoder.layers:
            print(
                '+' * 20,
                '[CAUTIOUS] Zero-initialising attention output projections. '
                'Make sure this is intentional.',
                '+' * 20,
            )
            nn.init.zeros_(layer.linear2.weight)
            nn.init.zeros_(layer.linear2.bias)
            attns = layer.self_attn if isinstance(layer.self_attn, nn.ModuleList) else [layer.self_attn]
            for attn in attns:
                nn.init.zeros_(attn.out_proj.weight)
                nn.init.zeros_(attn.out_proj.bias)
            if num_R is not None:
                _init_router(layer)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, *args, **kwargs):
        """Flexible forward supporting three call signatures:

        1. ``model(train_x, train_y, test_x, src_mask=None, style=None, only_return_standard_out=True)``
        2. ``model((x, y), src_mask=None, single_eval_pos=None, only_return_standard_out=True)``
        3. ``model((style, x, y), src_mask=None, single_eval_pos=None, only_return_standard_out=True)``
        """
        if len(args) == 3:
            allowed = {'src_mask', 'style', 'only_return_standard_out'}
            assert set(kwargs) <= allowed, \
                f"Unexpected kwargs: {set(kwargs) - allowed}"
            x = args[0]
            if args[2] is not None:
                x = torch.cat((x, args[2]), dim=0)
                #padd x with zeros if x's first dimension < 5000
                if x.shape[0] < 5000:
                    pad_len = 5000 - x.shape[0]
                    pad = torch.zeros(
                        (pad_len, x.shape[1], x.shape[2]),
                        dtype=x.dtype,
                        device=x.device,
                    )
                    x = torch.cat((x, pad), dim=0)
            style = kwargs.pop('style', None)
            return self._forward((style, x, args[1]), single_eval_pos=len(args[0]), **kwargs)
        elif len(args) == 1 and isinstance(args[0], tuple):
            # allowed = {'src_mask', 'single_eval_pos', 'only_return_standard_out','alpha'}
            # assert set(kwargs) <= allowed, \
            #     f"Unexpected kwargs: {set(kwargs) - allowed}"
            return self._forward(*args, **kwargs)
        else:
            raise ValueError(f"Unexpected call signature: {len(args)} positional args")

    def _forward(self, 
                 src, 
                 src_mask=None, 
                 single_eval_pos=None,
                 only_return_standard_out: bool = True,
                 alpha=torch.Tensor([0.0])):
        assert isinstance(src, tuple), 'src must be a tuple: (x, y) or (style, x, y)'

        if len(src) == 2:
            src = (None,) + src
            
        if len(src) == 3:
            style_src, x_src, y_src = src
            global_internals = None
        elif len(src) == 4:
            style_src, x_src, y_src, global_internals = src
        
        if single_eval_pos is None:
            single_eval_pos = x_src.shape[0]
        
        
        x_to_global = self.input_to_internal_encoder(x_src)
        
        # Feature embedding
        x_src = self.encoder(x_src)
    
        if global_internals is not None:
            x_global_train = (1-alpha) * x_to_global + alpha * global_internals
        else:
            x_global_train = x_to_global
            
        global_internals = self.internal_feature_encoder(x_global_train)
        global_internal_pos = global_internals.shape[0]

        # Append "decode-once" placeholder tokens
        if self.decoder_dict_once is not None:
            x_src = torch.cat(
                [x_src, self.decoder_dict_once_embeddings.repeat(1, x_src.shape[1], 1)],
                dim=0,
            )

        # Label embedding
        if y_src is not None:
            if len(y_src.shape) < len(x_src.shape):
                y_src = y_src.unsqueeze(-1)
            y_src = self.y_encoder(y_src)

        # Style embedding
        if self.style_encoder:
            assert style_src is not None
            style_src = self.style_encoder(style_src).unsqueeze(0)
        else:
            style_src = torch.tensor([], device=x_src.device)

        # Global attention token embeddings
        global_src = (
            torch.tensor([], device=x_src.device)
            if self.global_att_embeddings is None
            else self.global_att_embeddings.weight.unsqueeze(1).repeat(1, x_src.shape[1], 1)
        )

        # Build attention mask
        if src_mask is None:
            if self.global_att_embeddings is None:
                full_len = len(x_src) + len(style_src)
                if self.full_attention:
                    src_mask = bool_mask_to_att_mask(
                        torch.ones((full_len, full_len), dtype=torch.bool)
                    ).to(x_src.device)
                elif self.efficient_eval_masking:
                    src_mask = single_eval_pos + len(style_src)
                else:
                    src_mask = self.generate_D_q_matrix(
                        full_len, len(x_src) - single_eval_pos
                    ).to(x_src.device)
            else:
                args_ = (
                    self.global_att_embeddings.num_embeddings,
                    len(x_src) + len(style_src),
                    len(x_src) + len(style_src) - single_eval_pos,
                )
                src_mask = (
                    self.generate_global_att_globaltokens_matrix(*args_).to(x_src.device),
                    self.generate_global_att_trainset_matrix(*args_).to(x_src.device),
                    self.generate_global_att_query_matrix(*args_).to(x_src.device),
                )

        # Combine context tokens with label embeddings
        train_x = x_src[:single_eval_pos]
        if y_src is not None:
            train_x = train_x + y_src[:single_eval_pos]
            
        src = torch.cat([global_src, style_src, train_x, x_src[single_eval_pos:]], dim=0)
        

        if self.input_ln is not None:
            src = self.input_ln(src)
        if self.pos_encoder is not None:
            src = self.pos_encoder(src)

        output = self.transformer_encoder(src, src_mask,aux_memory = global_internals)

        # Determine output slice boundaries
        num_prefix = len(style_src) + (
            self.global_att_embeddings.num_embeddings if self.global_att_embeddings else 0
        )
        out_start = num_prefix if self.return_all_outputs else single_eval_pos + num_prefix + global_internal_pos
        out_end = (
            -len(self.decoder_dict_once_embeddings)
            if self.decoder_dict_once is not None else None
        )

        # Decode "once" outputs from tail tokens
        output_once = (
            {k: v(output[-(i + 1)]) for i, (k, v) in enumerate(self.decoder_dict_once.items())}
            if self.decoder_dict_once is not None else {}
        )

        # Decode standard outputs
        output = (
            {k: v(output[out_start:out_end]) for k, v in self.decoder_dict.items()}
            if self.decoder_dict is not None else {}
        )

        if only_return_standard_out:
            return output['standard']
        if output_once:
            return output, output_once
        return output

    # ------------------------------------------------------------------
    # Utility: initialise from a smaller model
    # ------------------------------------------------------------------
    @torch.no_grad()
    def init_from_small_model(self, small_model):
        assert (isinstance(self.decoder, nn.Linear)
                and isinstance(self.encoder, (nn.Linear, nn.Sequential))
                and isinstance(self.y_encoder, (nn.Linear, nn.Sequential)))

        def _copy_encoder(my_enc, small_enc):
            my_lin = my_enc if isinstance(my_enc, nn.Linear) else my_enc[-1]
            sm_lin = small_enc if isinstance(small_enc, nn.Linear) else small_enc[-1]
            d = sm_lin.out_features
            my_lin.weight.zero_()
            my_lin.bias.zero_()
            my_lin.weight[:d] = sm_lin.weight
            my_lin.bias[:d] = sm_lin.bias

        _copy_encoder(self.encoder, small_model.encoder)
        _copy_encoder(self.y_encoder, small_model.y_encoder)

        d_sm = small_model.decoder.in_features
        self.decoder.weight[:, :d_sm] = small_model.decoder.weight
        self.decoder.bias = small_model.decoder.bias

        for my_l, sm_l in zip(self.transformer_encoder.layers,
                               small_model.transformer_encoder.layers):
            d_hid = sm_l.linear1.out_features
            d_in = my_l.linear1.in_features

            my_w = my_l.self_attn.in_proj_weight
            sm_w = sm_l.self_attn.in_proj_weight
            my_w.view(3, d_in, d_in)[:, :d_hid, :d_hid] = sm_w.view(3, d_hid, d_hid)
            my_l.self_attn.in_proj_bias.view(3, d_in)[:, :d_hid] = \
                sm_l.self_attn.in_proj_bias.view(3, d_hid)

            my_l.self_attn.out_proj.weight[:d_hid, :d_hid] = sm_l.self_attn.out_proj.weight
            my_l.self_attn.out_proj.bias[:d_hid] = sm_l.self_attn.out_proj.bias

            my_l.linear1.weight[:d_hid, :d_in] = sm_l.linear1.weight
            my_l.linear1.bias[:d_hid] = sm_l.linear1.bias
            my_l.linear2.weight[:d_in, :d_hid] = sm_l.linear2.weight
            my_l.linear2.bias[:d_in] = sm_l.linear2.bias

            scale = math.sqrt(d_hid / d_in)
            my_l.norm1.weight[:d_hid] = scale * sm_l.norm1.weight
            my_l.norm2.weight[:d_hid] = scale * sm_l.norm2.weight
            my_l.norm1.bias[:d_hid] = sm_l.norm1.bias
            my_l.norm2.bias[:d_hid] = sm_l.norm2.bias


# ---------------------------------------------------------------------------
# TransformerEncoderDiffInit
# ---------------------------------------------------------------------------

class TransformerEncoderDiffInit(Module):
    """Stack of N encoder layers, each independently initialised.

    Args:
        encoder_layer_creator: callable ``(is_final_layer) -> TransformerEncoderLayer``
        num_layers:            number of layers.
        norm:                  optional final layer normalisation.
    """

    __constants__ = ['norm']

    def __init__(self, encoder_layer_creator, num_layers: int, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([
            encoder_layer_creator(is_final_layer=(i == num_layers - 1))
            for i in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                aux_memory=None,
                aux_key_padding_mask=None) -> Tensor:
        output = src
        for i,mod in enumerate(self.layers):
            if i == 0:
                output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            else:                
                output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask,
                         aux_memory=aux_memory)
        if self.norm is not None:
            output = self.norm(output)
        return output
