import gc
import os
import pickle
import random
import sys
import time
from typing import Callable
import math
import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

# Add the FoMo-Meta root to sys.path so subdirectories are importable by their
# short names (e.g. `data_prior`, `model`), matching the pattern used in
# FoMo-Meta/model/transformer.py.
# _fomo_meta_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if _fomo_meta_root not in sys.path:
#     sys.path.insert(0, _fomo_meta_root)

from data_prior.GMM import generate_linear_transform, transform_samples
from dataset_loader.batch import PriorDataLoader
from dataset_loader.dataset import EpochDataset
from model_meta_0413.transformer_program_encoder_performer_mix import TransformerModel
from trainer import utils
from trainer.utils import (
    get_cosine_schedule_with_warmup,
    get_cosine_schedule_with_warmup_min_lr,
    get_openai_lr,
)

from model_meta_0413 import positional_encodings
import os
# os.environ["HF_HUB_OFFLINE"] = "1"

# Add trainer_embedder root to sys.path for loading the program encoder modules.
# _trainer_embedder_root = os.path.join(_fomo_meta_root, "trainer_embedder")
# if _trainer_embedder_root not in sys.path:
#     sys.path.insert(0, _trainer_embedder_root)

from trainer_embedder.lightning_allpriors_contrastive_train import (
    LitMultiPriorContrastive,
    ProgramTransformerEncoder,
    ProgramVectorizer,
    build_bootstrap_vocabs,
)


def load_pickle(file_path):
    with open(file_path, 'rb') as handle:
        inst = pickle.load(handle)
    return inst


def build_program_encoder_model(
    *,
    min_dim: int,
    max_dim: int,
    min_num_cluster: int,
    max_num_cluster: int,
    max_mean: int,
    max_var: int,
    inflate_scale: float,
    num_prototypes: int,
    num_cls: int,
    max_tokens: int | None,
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_feedforward: int,
    dropout: float,
    lr: float,
    easy_margin: float,
    hard_margin: float,
    hard_weight: float,
    hard_mining_weight: float,
    proxy_weight: float,
    proxy_temp: float,
    warmup_ratio: float,
    min_lr_ratio: float,
) -> LitMultiPriorContrastive:
    class _BootstrapArgs:
        def __init__(self):
            self.min_dim = min_dim
            self.max_dim = max_dim
            self.min_num_cluster = min_num_cluster
            self.max_num_cluster = max_num_cluster
            self.max_mean = max_mean
            self.max_var = max_var
            self.inflate_scale = inflate_scale
            self.num_prototypes = num_prototypes

    bootstrap_args = _BootstrapArgs()

    symbol_vocab, field_vocab, family_vocab, entity_type_vocab = build_bootstrap_vocabs(
        num_bootstrap=96,
        num_cls=num_cls,
        max_tokens=max_tokens,
        args=bootstrap_args,
    )

    vectorizer = ProgramVectorizer(
        symbol_vocab=symbol_vocab,
        field_vocab=field_vocab,
        family_vocab=family_vocab,
        entity_type_vocab=entity_type_vocab,
        d_model=d_model,
        max_dim=max(256, max_dim + 64),
        max_entity_id=256,
        max_index_pos=max(256, max_dim + 64),
    )

    encoder = ProgramTransformerEncoder(
        vectorizer=vectorizer,
        num_cls=num_cls,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )

    return LitMultiPriorContrastive(
        encoder=encoder,
        num_cls=num_cls,
        d_model=d_model,
        max_tokens=max_tokens,
        lr=lr,
        easy_margin=easy_margin,
        hard_margin=hard_margin,
        hard_weight=hard_weight,
        hard_mining_weight=hard_mining_weight,
        proxy_weight=proxy_weight,
        proxy_temp=proxy_temp,
        warmup_ratio=warmup_ratio,
        min_lr_ratio=min_lr_ratio,
    )


def load_program_encoder_weights(model: LitMultiPriorContrastive, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    result = model.load_state_dict(state_dict, strict=False)
    if getattr(result, "missing_keys", None):
        print("[program encoder ckpt] missing_keys:", result.missing_keys)
    if getattr(result, "unexpected_keys", None):
        print("[program encoder ckpt] unexpected_keys:", result.unexpected_keys)




def make_model_od(criterion, encoder_generator,
                  emsize=200, nhid=200, nlayers=6, nhead=2, dropout=0.0, seq_len=10,
                  input_normalization=False,
                  y_encoder_generator=None, pos_encoder_generator=None, decoder_dict={}, extra_prior_kwargs_dict={},
                  initializer=None,
                  efficient_eval_masking=True, num_global_att_tokens=0, **model_extra_args):
    style_encoder = None
    pos_encoder = (pos_encoder_generator or positional_encodings.NoPositionalEncoding)(emsize, seq_len * 2)
    if isinstance(criterion, nn.GaussianNLLLoss):
        n_out = 2
    elif isinstance(criterion, nn.CrossEntropyLoss):
        n_out = criterion.weight.shape[0]
    else:
        n_out = 1

    decoder_dict = decoder_dict if decoder_dict else {'standard': (None, n_out)}

    decoder_once_dict = {}

    encoder = encoder_generator(extra_prior_kwargs_dict['num_features'], emsize)
    _internal_enc_gen = internal_encoder_generator if internal_encoder_generator is not None else encoder_generator
    internal_feature_encoder = _internal_enc_gen(extra_prior_kwargs_dict['num_features'], emsize)
    input_to_internal_encoder = input_to_internal_encoder_generator(
                100,
                256,
                seq_len+100,
                10,
    )
    model = TransformerModel(encoder=encoder
                             , nhead=nhead
                             , ninp=emsize
                             , nhid=nhid
                             , nlayers=nlayers
                             , dropout=dropout
                             , style_encoder=style_encoder
                             , y_encoder=y_encoder_generator(1, emsize)
                             , internal_feature_encoder=internal_feature_encoder
                             , input_to_internal_encoder = input_to_internal_encoder
                             , input_normalization=input_normalization
                             , pos_encoder=pos_encoder
                             , decoder_dict=decoder_dict
                             , init_method=initializer
                             , efficient_eval_masking=efficient_eval_masking
                             , decoder_once_dict=decoder_once_dict
                             , num_global_att_tokens=num_global_att_tokens
                             , **model_extra_args
                             )
    model.criterion = criterion
    print(model)
    return model


class MetricRecorder:
    def __init__(self, seq_len, steps_per_epoch, verbose):
        self.seq_len = seq_len
        self.steps_per_epoch = steps_per_epoch
        self.verbose = verbose

        self.total_loss = 0.0
        self.total_positional_losses = torch.zeros(self.seq_len)
        self.total_positional_losses_recorded = torch.zeros(self.seq_len)
        self.nan_steps = 0.0
        self.ignore_steps = 0.0
        self.epoch_start_time = 0.0
        self.total_step_time = 0.0
        self.density_tree_loss = 0.0
        self.scm_prob_loss =0.0
        self.scm_contextual_loss = 0.0
        self.copula_loss = 0.0
        self.gmm_loss = 0.0
        self.gmm_step_count = 0
        self.copula_step_count = 0
        self.scm_prob_step_count = 0
        self.scm_contextual_step_count = 0
        self.density_tree_step_count = 0
        self.density_tree_step_count = 0
        self.disturb_copula_loss = 0
        self.disturb_copula_step_count = 0
        self.perturb_copula_loss =0 
        self.perturb_copula_step_count = 0
        self.model_names_list = ['gmm', 'disturbcorpula','perturbcorpula', 'prob', 'contextual'] #, 'density']
        

    def reset(self):
        self.total_loss = 0.0
        self.nan_steps = 0.0
        self.ignore_steps = 0.0
        self.epoch_start_time = 0.0
        self.total_step_time = 0.0
        self.density_tree_loss = 0.0
        self.scm_prob_loss =0.0
        self.scm_contextual_loss = 0.0
        self.copula_loss = 0.0
        self.gmm_loss = 0.0
        self.gmm_step_count = 0
        self.copula_step_count = 0
        self.scm_prob_step_count = 0
        self.scm_contextual_step_count = 0
        self.density_tree_step_count = 0
        self.disturb_copula_loss = 0
        self.disturb_copula_step_count = 0
        self.perturb_copula_loss =0 
        self.perturb_copula_step_count = 0
        
        
        

    def update(self, loss, losses, single_eval_pos, targets, nan_share, step_time, model_names,grad_vectors=None, cos_sim=None):
        if not torch.isnan(loss):
            self.total_loss += loss.cpu().detach().item()
            if model_names is not None and losses is not None:
                mean_loss = losses.mean(0)
                for i,name in enumerate(model_names):
                    l = mean_loss[i].cpu().detach().item()
                    if name == 'gmm':
                        self.gmm_loss += l
                        self.gmm_step_count += 1
                    elif name == 'disturbcorpula':
                        self.disturb_copula_loss += l
                        self.disturb_copula_step_count += 1
                    elif name == 'perturbcorpula':
                        self.perturb_copula_loss += l
                        self.perturb_copula_step_count += 1
                    elif name == 'contextual':
                        self.scm_contextual_loss += l
                        self.scm_contextual_step_count += 1
                    elif name == 'prob':
                        self.scm_prob_loss += l
                        self.scm_prob_step_count += 1
                    elif name == 'density':
                        self.density_tree_loss += l
                        self.density_tree_step_count += 1     
            self.nan_steps += nan_share.cpu().item()
            self.ignore_steps += (targets == -100).float().mean().cpu().item()
        self.total_step_time += step_time

   


    def fetch_and_print(self, epoch=None, lr=None):
        avg_loss = self.total_loss / self.steps_per_epoch
        avg_gmm_loss = self.gmm_loss / self.gmm_step_count if self.gmm_step_count != 0 else 0
        avg_copula_loss = self.copula_loss / self.copula_step_count if self.copula_step_count != 0 else 0
        avg_contextual_loss = self.scm_contextual_loss / self.scm_contextual_step_count if self.scm_contextual_step_count != 0 else 0
        avg_prob_loss = self.scm_prob_loss / self.scm_prob_step_count if self.scm_prob_step_count != 0 else 0
        avg_density_loss = self.density_tree_loss / self.density_tree_step_count if self.density_tree_step_count != 0 else 0
        avg_disturbcopula_loss = self.disturb_copula_loss / self.disturb_copula_step_count if self.disturb_copula_step_count != 0 else 0
        avg_perturbcopula_loss = self.perturb_copula_loss / self.perturb_copula_step_count if self.perturb_copula_step_count != 0 else 0

        nan_share = self.nan_steps / self.steps_per_epoch
        ignore_share = self.ignore_steps / self.steps_per_epoch
        total_time = time.time() - self.epoch_start_time
             
        if self.verbose:
            print('-' * 89)
            print(
                f' nan share {nan_share:5.2f} ignore share (for classification tasks) {ignore_share:5.4f}'
                f' | end of epoch {epoch:3d} | time: {total_time:5.2f}s | (approx) step time: {self.total_step_time:5.2f}s | '
                f'(approx) data time: {total_time - self.total_step_time:5.2f}s | mean loss {avg_loss:5.2f} | lr {lr}'
            )
            print(f" Avg losses: GMM={avg_gmm_loss:.4f}, Copula={avg_copula_loss:.4f}, "
                f"Contextual={avg_contextual_loss:.4f}, Prob={avg_prob_loss:.4f}, Density={avg_density_loss:.4f}")
            print('-' * 89)
            
        return {
            'avg_loss': avg_loss,
            'nan_share': nan_share,
            'ignore_share': ignore_share,
            'total_time': total_time,
            'avg_gmm_loss': avg_gmm_loss,
            'avg_copula_loss': avg_copula_loss,
            'avg_contextual_loss': avg_contextual_loss,
            'avg_prob_loss': avg_prob_loss,
            'avg_density_loss': avg_density_loss,
            'avg_disturbcopula_loss': avg_disturbcopula_loss,
            'avg_perturbcopula_loss': avg_perturbcopula_loss
        }


class ZeroShotOD(pl.LightningModule):
    def __init__(self, 
                 cfg, 
                 priordataloader_class_or_get_batch: PriorDataLoader | Callable, criterion,
                 encoder_generator, 
                 internal_encoder_generator=None,
                 input_to_internal_encoder_generator = None,
                 dropout=0.0,
                 weight_decay=0.0,
                 input_normalization=False,
                 y_encoder_generator=None,
                 pos_encoder_generator=None, 
                 decoder_dict={},
                 extra_prior_kwargs_dict={},
                 train_extra_dict=None, 
                 resume_from_ckpt=False,  # added here
                 scheduler=get_cosine_schedule_with_warmup_min_lr, #get_cosine_schedule_with_warmup,
                 load_weights_from_this_state_dict=None, 
                 validation_period=10, 
                 single_eval_pos_gen=None,
                 gpu_device='cuda:0',
                 aggregate_k_gradients=1, 
                 verbose=False, 
                 style_encoder_generator=None, 
                 epoch_callback=None,
                 step_callback=None,
                 continue_model=None,
                 initializer=None, 
                 initialize_with_model=None, 
                 train_mixed_precision=False, 
                 efficient_eval_masking=True,
                 border_decoder=None
                 , num_global_att_tokens=0,
                 T0 = 0, 
                 progress_bar=False,
                 **model_extra_args):
        super(ZeroShotOD, self).__init__()

        train_cfg = cfg.train
        prior_gmm_cfg = cfg.prior.mixture.gmm
        # train hyperparameters
        seq_len = train_cfg.seq_len
        self.batch_size = train_cfg.batch_size
        epochs = train_cfg.epochs
        self.steps_per_epoch = train_cfg.steps_per_epoch
        emsize = train_cfg.emsize
        nhead = train_cfg.nhead
        nhid = train_cfg.nhid
        nlayers = train_cfg.nlayer
        self.reuse_data_every_n = train_cfg.reuse_data_every_n
        num_device = train_cfg.num_device
        self.num_device = num_device
        self.steps_per_epoch = train_cfg.steps_per_epoch
        lr = train_cfg.lr  #/ num_device

        # Program encoder (text -> embedding) settings.
        self.program_encoder_ckpt = getattr(
            train_cfg,
            'program_encoder_ckpt',
            '/workspace/PIQL/trainer_embedder/ckpt/epoch-epoch=13.ckpt',
        )
        self.program_encoder_min_dim = int(getattr(train_cfg, 'program_encoder_min_dim', 8))
        self.program_encoder_max_dim = int(getattr(train_cfg, 'program_encoder_max_dim', 40))
        default_cluster = int(getattr(train_cfg, 'program_encoder_num_cluster', 3))
        self.program_encoder_min_num_cluster = int(
            getattr(train_cfg, 'program_encoder_min_num_cluster', default_cluster)
        )
        self.program_encoder_max_num_cluster = int(
            getattr(train_cfg, 'program_encoder_max_num_cluster', default_cluster)
        )
        self.program_encoder_max_mean = int(getattr(train_cfg, 'program_encoder_max_mean', 6))
        self.program_encoder_max_var = int(getattr(train_cfg, 'program_encoder_max_var', 6))
        self.program_encoder_inflate_scale = float(getattr(train_cfg, 'program_encoder_inflate_scale', 5.0))
        self.program_encoder_num_prototypes = int(getattr(train_cfg, 'program_encoder_num_prototypes', 16))
        self.program_encoder_num_cls = int(getattr(train_cfg, 'program_encoder_num_cls', 10))
        self.program_encoder_max_tokens = getattr(train_cfg, 'program_encoder_max_tokens', 2048)
        self.program_encoder_d_model = int(getattr(train_cfg, 'program_encoder_d_model', 256))
        self.program_encoder_nhead = int(getattr(train_cfg, 'program_encoder_nhead', 8))
        self.program_encoder_num_layers = int(getattr(train_cfg, 'program_encoder_num_layers', 4))
        self.program_encoder_dim_feedforward = int(getattr(train_cfg, 'program_encoder_dim_feedforward', 256))
        self.program_encoder_dropout = float(getattr(train_cfg, 'program_encoder_dropout', 0.1))
        self.program_encoder_lr = float(getattr(train_cfg, 'program_encoder_lr', 1e-4))
        self.program_encoder_easy_margin = float(getattr(train_cfg, 'program_encoder_easy_margin', 0.8))
        self.program_encoder_hard_margin = float(getattr(train_cfg, 'program_encoder_hard_margin', 0.4))
        self.program_encoder_hard_weight = float(getattr(train_cfg, 'program_encoder_hard_weight', 1.8))
        self.program_encoder_hard_mining_weight = float(getattr(train_cfg, 'program_encoder_hard_mining_weight', 0.5))
        self.program_encoder_proxy_weight = float(getattr(train_cfg, 'program_encoder_proxy_weight', 0.5))
        self.program_encoder_proxy_temp = float(getattr(train_cfg, 'program_encoder_proxy_temp', 0.07))
        self.program_encoder_warmup_ratio = float(getattr(train_cfg, 'program_encoder_warmup_ratio', 0.1))
        self.program_encoder_min_lr_ratio = float(getattr(train_cfg, 'program_encoder_min_lr_ratio', 5e-6))
        self.program_encoder = self._build_program_encoder_module()
        self._program_encoder_loaded = True
        self._program_encoder_weights_loaded = False

        # prior hyperparameters
        self.max_feature_dim = prior_gmm_cfg.max_feature_dim
        self.max_model_dim = prior_gmm_cfg.max_model_dim
        self.max_num_cluster = prior_gmm_cfg.max_num_cluster
        self.inflate_full = prior_gmm_cfg.inflate_full
        #self.model_names_list = ['gmm'] 

        # specifics for generate-one-train-one
        self.gen_one_train_one = False if train_extra_dict is None else True
        self.prior_train_data_gen = None if train_extra_dict is None else train_extra_dict['prior_train_data_gen']

        self.criterion = criterion

        self.apply_linear_transform = train_cfg.apply_linear_transform
        self.dataloader_para = extra_prior_kwargs_dict['pt_dataloader']

        self.base_data_path = f'{prior_gmm_cfg.data_dir}/num_feat_{self.max_feature_dim}'
        print('train data loader')
        if not self.apply_linear_transform and not self.gen_one_train_one:
            train_data_path = f'{self.base_data_path}/train'  # if provided, by default `epoch0` will be loaded
        else:
            train_data_path = None

        choices = 'global'
        self.train_dataset = EpochDataset(batch_size=self.batch_size, 
                                          seq_len=seq_len,
                                          steps_per_epoch=self.steps_per_epoch,
                                          hyperparameters=extra_prior_kwargs_dict['hyperparameters'],
                                          reuse_data_every_n=self.reuse_data_every_n, max_model_dim=self.max_model_dim,
                                          max_num_cluster=self.max_num_cluster,
                                          get_batch_method=priordataloader_class_or_get_batch,
                                          rank=0, num_device=num_device,  # rank is not yet set in __init__
                                          training=True, single_eval_pos_gen=single_eval_pos_gen,
                                          data_path=train_data_path,
                                          is_source_numpy=False if self.gen_one_train_one else True,
                                          choices=choices)

        self.train_dl = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                   collate_fn=self.train_dataset.prior_batch_collate_fn,
                                   **self.dataloader_para)
        self.val_dataset = None
        self.val_dl = None

        print(f'Style definition of first 3 examples: {None}')
        style_encoder = None
        pos_encoder = (pos_encoder_generator or positional_encodings.NoPositionalEncoding)(emsize, seq_len * 2)
        if isinstance(self.criterion, nn.GaussianNLLLoss):
            self.n_out = 2
        elif isinstance(self.criterion, nn.CrossEntropyLoss):
            self.n_out = self.criterion.weight.shape[0]
        else:
            self.n_out = 1

        # initialize model
        if continue_model:
            raise NotImplementedError
        else:
            decoder_dict = decoder_dict if decoder_dict else {'standard': (None, self.n_out)}

            decoder_once_dict = {}

            encoder = encoder_generator(extra_prior_kwargs_dict['num_features'], emsize)
            
            _internal_enc_gen = internal_encoder_generator if internal_encoder_generator is not None else encoder_generator
            internal_feature_encoder = _internal_enc_gen(extra_prior_kwargs_dict['num_features'], emsize)
            input_to_internal_encoder = input_to_internal_encoder_generator(
                100,
                256,
                seq_len+100,
                10,
            )
            self.model = TransformerModel(encoder=encoder
                                          , nhead=nhead
                                          , ninp=emsize
                                          , nhid=nhid
                                          , nlayers=nlayers
                                          , dropout=dropout
                                          , style_encoder=style_encoder
                                          , y_encoder=y_encoder_generator(1, emsize)
                                          , internal_feature_encoder=internal_feature_encoder
                                          , input_to_internal_encoder = input_to_internal_encoder
                                          , input_normalization=input_normalization
                                          , pos_encoder=pos_encoder
                                          , decoder_dict=decoder_dict
                                          , init_method=initializer
                                          , efficient_eval_masking=efficient_eval_masking
                                          , decoder_once_dict=decoder_once_dict
                                          , num_global_att_tokens=num_global_att_tokens
                                          , **model_extra_args
                                          )
            print(self.model)
        self.model.criterion = self.criterion

        print(
            f"Using a Transformer with {sum(p.numel() for p in self.model.parameters()) / 1000 / 1000:.{2}f} M parameters")

        try:
            for (k, v), (k2, v2) in zip(self.model.state_dict().items(), initialize_with_model.state_dict().items()):
                print(k, ((v - v2) / v).abs().mean(), v.shape)
        except Exception:
            pass

        # define parameters for optimizer & scheduler
        self.lr = lr
        self.scheduler_fn = scheduler
        self.warmup_epochs =  0 #epochs // 5  #epochs // 10 # warmup epochs, usually 1/10 of total epochs
        self.weight_decay = weight_decay
        self.epochs = epochs
        
        utils.check_compatibility(self.train_dl)
  
        self.train_recorder = MetricRecorder(seq_len=seq_len, steps_per_epoch=self.steps_per_epoch, verbose=verbose)
        self.train_losses = []
        self.val_losses = []


    def _build_program_encoder_module(self):
        program_encoder = build_program_encoder_model(
            min_dim=self.program_encoder_min_dim,
            max_dim=self.program_encoder_max_dim,
            min_num_cluster=self.program_encoder_min_num_cluster,
            max_num_cluster=self.program_encoder_max_num_cluster,
            max_mean=self.program_encoder_max_mean,
            max_var=self.program_encoder_max_var,
            inflate_scale=self.program_encoder_inflate_scale,
            num_prototypes=self.program_encoder_num_prototypes,
            num_cls=self.program_encoder_num_cls,
            max_tokens=self.program_encoder_max_tokens,
            d_model=self.program_encoder_d_model,
            nhead=self.program_encoder_nhead,
            num_layers=self.program_encoder_num_layers,
            dim_feedforward=self.program_encoder_dim_feedforward,
            dropout=self.program_encoder_dropout,
            lr=self.program_encoder_lr,
            easy_margin=self.program_encoder_easy_margin,
            hard_margin=self.program_encoder_hard_margin,
            hard_weight=self.program_encoder_hard_weight,
            hard_mining_weight=self.program_encoder_hard_mining_weight,
            proxy_weight=self.program_encoder_proxy_weight,
            proxy_temp=self.program_encoder_proxy_temp,
            warmup_ratio=self.program_encoder_warmup_ratio,
            min_lr_ratio=self.program_encoder_min_lr_ratio,
        )
        program_encoder.eval()
        for p in program_encoder.parameters():
            p.requires_grad = False
        return program_encoder


    def _ensure_program_encoder_loaded(self):
        if self.program_encoder is None:
            self.program_encoder = self._build_program_encoder_module()
            self._program_encoder_loaded = True

        if self._program_encoder_weights_loaded:
            self.program_encoder.eval().to(self.device)
            return

        if not self.program_encoder_ckpt or not os.path.exists(self.program_encoder_ckpt):
            raise FileNotFoundError(
                f"Program encoder checkpoint not found: {self.program_encoder_ckpt}"
            )

        load_program_encoder_weights(self.program_encoder, self.program_encoder_ckpt)
        self.program_encoder.eval().to(self.device)
        self._program_encoder_loaded = True
        self._program_encoder_weights_loaded = True
        print(f"Loaded program encoder checkpoint from: {self.program_encoder_ckpt}")


    def _encode_internal_xs(self, internal_xs):
        if internal_xs is None:
            return None
        if torch.is_tensor(internal_xs):
            return internal_xs.to(self.device)
        if not isinstance(internal_xs, (list, tuple)):
            raise TypeError(
                f"Expected internal_xs to be list/tuple/tensor, got {type(internal_xs)}"
            )

        self._ensure_program_encoder_loaded()
        embeds = []
        with torch.inference_mode():
            for text in internal_xs:
                if not isinstance(text, str):
                    text = str(text)
                emb = self.program_encoder._encode_text(text)
                emb = emb.reshape(-1).detach()
                embeds.append(emb)
        return torch.stack(embeds, dim=0).to(self.device)

    def get_alpha(self):
        p = self.current_epoch / self.epochs
        p1 = 0.1   # warmup (pure teacher)
        p2 = 0.4   # end of transition

        if p < p1:
            return 1.0
        elif p < p2:
            # cosine decay
            x = (p - p1) / (p2 - p1)
            return 0.5 * (1 + math.cos(math.pi * x))
        else:
            return 0.0


    def configure_optimizers(self):
        # learning rate
        if self.lr is None:
            self.lr = get_openai_lr(self.model)
            print(f"Using OpenAI max lr of {self.lr}.")
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = self.scheduler_fn(optimizer, self.warmup_epochs,
                                      self.epochs if self.epochs is not None else 100)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',  # Monitor a validation metric
                'interval': 'epoch',  # How often to step (options: 'epoch', 'step')
                'frequency': 1,  # How many epochs/steps between each step
            }
        }

    def on_fit_start(self) -> None:
        print('on_fit_start---setting ranks...')
        # Ensure program encoder is initialized once before any forward pass.
        self._ensure_program_encoder_loaded()
        self.train_dataset.set_rank(rank=self.global_rank)
        if self.val_dataset is not None:
            self.val_dataset.set_rank(rank=self.global_rank)
        if self.trainer.ckpt_path:
            print(f"Resuming training from checkpoint: {self.trainer.ckpt_path}")
        else:
            print("Training from scratch.")



    def train_dataloader(self):
        # generate data on the fly (if with "generate-one-train-one" paradigm)
        self.train_dataset.free_data()  # free data to allow space for data generation
        data_dict = self.generate_new_data_for_train()

        self.train_dataset.set_epoch_and_data(epoch=self.current_epoch, data_dict=data_dict)

        if data_dict is None:  # using the existing data (reuse 1 epoch of data)
            return self.train_dl
        else:  # load new data
            del data_dict
            gc.collect()
            torch.cuda.empty_cache()
            # set the epoch such that it will load different data
            self.train_dl = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                       collate_fn=self.train_dataset.prior_batch_collate_fn,
                                       **self.dataloader_para)#,num_workers=4)
            print( f'num batches per epoch={len(self.train_dl)}')
            return self.train_dl


    def generate_new_data_for_train(self):
        self.prior_train_data_gen.gen1tr1_epoch_id = self.current_epoch
        self.prior_train_data_gen.device = f"cuda:{self.global_rank}"
        
        # set the generation epoch to avoid repetitive generation
        if self.apply_linear_transform:
            # generate less on the fly, and use LT to fill up to `steps_per_epoch * batch_size`
            inliners_list, LA_list, sub_dims_list, model_names = self.prior_train_data_gen.generate_one_epoch_then_train_one(
                every_n_dim=1,  # generate `max_feature_dim * max_num_cluster`
                save_data=False)
            inliners_list, LA_list = self.increase_datasets_via_LT(inliners_list=inliners_list, LA_list=LA_list,
                                                                   sub_dims_list=sub_dims_list, transform_all=False,
                                                                   is_source_numpy=False)
        else:  # generate `steps_per_epoch * batch_size` all together on the fly
            inliners_list, LA_list, _ , model_names = self.prior_train_data_gen.generate_one_epoch_then_train_one(
                every_n_dim=None,  # generate `steps_per_epoch * batch_size`
                save_data=False)
            
        assert len(LA_list) >= int(self.steps_per_epoch * self.batch_size /self.num_device ), \
            print(
                f'number of training instances={len(LA_list)} should be >= {self.steps_per_epoch * self.batch_size}')

        return {'in': inliners_list, 'la': LA_list, 'model_names':model_names}



    def increase_datasets_via_LT(self, inliners_list, LA_list, sub_dims_list, transform_all, is_source_numpy=False):
        LT_in_list, LT_LA_list = [], []
        # needs to be at least (`steps_per_epoch * batch_size` - len(inliners_list))
        data_size = len(inliners_list)
        indices = list(range(data_size))  # Create a list of numbers from 0 to n-1
        random.shuffle(indices)  # Shuffle the list to get a random order
        if transform_all:
            transform_times = 1
        else:
            transform_times = (self.steps_per_epoch * self.batch_size + data_size) // data_size - 1

        for i in indices:
            for _ in range(transform_times):
                if sub_dims_list is None:
                    inliners, LA, sub_dims = inliners_list[i], LA_list[i], None
                    A, b = generate_linear_transform(dim=inliners.shape[-1],
                                                     device=None if is_source_numpy else inliners.device,
                                                     A_scale=1, b_scale=1,)
                else:
                    inliners, LA, sub_dims = inliners_list[i], LA_list[i], sub_dims_list[i]
                    A, b = generate_linear_transform(dim=len(sub_dims), device=None if is_source_numpy else inliners.device,
                                                     A_scale=1, b_scale=1,)

                inliners = transform_samples(samples=inliners, sub_dims=sub_dims, A=A, b=b,
                                             is_source_numpy=is_source_numpy)
                LA = transform_samples(samples=LA, sub_dims=sub_dims, A=A, b=b,
                                       is_source_numpy=is_source_numpy)

                LT_in_list.append(inliners)
                LT_LA_list.append(LA)

        if transform_all:
            return LT_in_list, LT_LA_list
        else:
            inliners_list = inliners_list + LT_in_list
            LA_list = LA_list + LT_LA_list
            return inliners_list, LA_list


    def val_dataloader(self):
        # DataLoader for validation
        return self.val_dl
    
    
    def test_dataloader(self):
        return self.test_dl
    

    def forward(self, full_data):
        internal_embeds = self._encode_internal_xs(full_data.internal_xs)
        internal_embeds = internal_embeds.reshape(
            internal_embeds.shape[0],
            self.program_encoder_num_cls,
            self.program_encoder_d_model,
        ).mean(dim=1)
        data = (full_data.style.to(self.device) if full_data.style is not None else None, 
                full_data.x.to(self.device),
                full_data.y.to(self.device) if full_data.y is not None else None,
            internal_embeds.to(self.device))
        model_names = full_data.model_names if full_data.model_names is not None else None
        # added here (do not include y in the feature)
        targets = full_data.target_y.to(self.device)
        single_eval_pos = full_data.single_eval_pos
        try:
            alpha = self.get_alpha()
            # If style is set to None, it should not be transferred to device
            out = self.model(
                tuple(e.to(self.device) if torch.is_tensor(e) else e for e in data),
                single_eval_pos=single_eval_pos,
                only_return_standard_out=False,
                alpha=alpha,
            )

            # def _contains_nan(obj):
            #     if torch.is_tensor(obj):
            #         return torch.isnan(obj).any().item()
            #     if isinstance(obj, dict):
            #         return any(_contains_nan(v) for v in obj.values())
            #     if isinstance(obj, (list, tuple)):
            #         return any(_contains_nan(v) for v in obj)
            #     return False

            # if _contains_nan(out):
            #     print('output has nan here!!')
            #     raise SystemExit(0)

            # this handling is for training old models only, this can be deleted soon(ish)
            # to only support models that return a tuple of dicts
            out, output_once = out if isinstance(out, tuple) else (out, None)
            output = out['standard'] if isinstance(out, dict) else out

            if single_eval_pos is not None:
                targets = targets[single_eval_pos:]

            if len(targets.shape) == len(output.shape):
                # this implies the prior uses a trailing 1 dimesnion
                # below we assume this not to be the case
                targets = targets.squeeze(-1)
            assert targets.shape == output.shape[:-1], f"Target shape {targets.shape} " \
                                                       f"does not match output shape {output.shape}"      
            assert not torch.isinf(output).any(), "Inf in outputs"
            assert output.shape[-1] == 2, "Each output must have 2 logits (classes)"
            assert targets.min() >= 0 and targets.max() < 2, "Target out of range"
            
            if isinstance(self.criterion, nn.GaussianNLLLoss):
                assert output.shape[-1] == 2, \
                    'need to write a little bit of code to handle multiple regression targets at once'

                mean_pred = output[..., 0]
                var_pred = output[..., 1].abs()
                losses = self.criterion(mean_pred.flatten(), targets.flatten(), var=var_pred.flatten())
            elif isinstance(self.criterion, (nn.MSELoss, nn.BCEWithLogitsLoss)):
                targets[torch.isnan(targets)] = -100
                losses = self.criterion(output.flatten(), targets.flatten())
            elif isinstance(self.criterion, nn.CrossEntropyLoss):
                #targets[torch.isnan(targets)] = -100
                # print(f"{targets.min()=},{targets.max()=}")
                # print(f"{output.min()=},{output.max()=}")
                losses = self.criterion(output.reshape(-1, self.n_out), targets.long().flatten())
            else:
                losses = self.criterion(output, targets)
            #print('losses shape before view', losses.shape)
            losses = losses.view(-1, output.shape[1]) 
            loss, nan_share = utils.torch_nanmean(losses.mean(0), return_nanshare=True)

        except Exception as e:
            print("Invalid step encountered, skipping...")
            print(e)
            raise e
        # print(loss)
        return loss, losses, single_eval_pos, targets, nan_share, model_names

    def training_step(self, batch, batch_idx):
        step_start = time.time()
        loss, losses, single_eval_pos, targets, nan_share, model_names = self.forward(full_data=batch)
        step_time = time.time() - step_start
        self.train_recorder.update(loss=loss,
                                   losses=losses,
                                   single_eval_pos=single_eval_pos, 
                                   targets=targets,
                                   nan_share=nan_share, 
                                   step_time=step_time, 
                                   model_names = model_names, 
                                   grad_vectors = None,
                                   cos_sim=None)
        return loss


    def on_train_epoch_start(self):
        self.train_recorder.epoch_start_time = time.time()
            
     
      
    def on_train_epoch_end(self) -> None:
        current_epoch = self.current_epoch
        print(f"Current epoch: {current_epoch}")
        lr = self.lr_schedulers().get_last_lr()[0]
 
        train_metric = self.train_recorder.fetch_and_print(epoch=self.current_epoch, lr=lr)
        
        self.log('train_loss', train_metric['avg_loss'], sync_dist=True)
        self.log('train_time', train_metric['total_time'], sync_dist=True)
        
        # Log learning rate
        self.log('lr',lr, sync_dist=True)

        # Log additional losses
        self.log('train_gmm_loss', train_metric['avg_gmm_loss'], sync_dist=True)
        self.log('train_disturb_copula_loss', train_metric['avg_disturbcopula_loss'], sync_dist=True)
        self.log('train_perturb_copula_loss', train_metric['avg_perturbcopula_loss'], sync_dist=True)
        self.log('train_contextual_loss', train_metric['avg_contextual_loss'], sync_dist=True)
        self.log('train_prob_loss', train_metric['avg_prob_loss'], sync_dist=True)
        self.log('train_density_loss', train_metric['avg_density_loss'], sync_dist=True)
        
        # Record average loss
        self.train_losses.append(train_metric['avg_loss'])

        # Reset metrics and clean up
        self.train_recorder.reset()
        gc.collect()
        torch.cuda.empty_cache()            

    
    def on_save_checkpoint(self, checkpoint):
        # Save the lists of train and val losses
        checkpoint['train_losses'] = self.train_losses
        checkpoint['val_losses'] = self.val_losses

    def on_load_checkpoint(self, checkpoint):
        state_dict = checkpoint.get('state_dict', {})
        has_program_encoder_weights = any(k.startswith('program_encoder.') for k in state_dict.keys())
        if has_program_encoder_weights:
            self._program_encoder_loaded = True
            self._program_encoder_weights_loaded = True

        # Load the lists of train and val losses
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        print('-' * 20)
        print(f'getting the train losses of length {len(self.train_losses)} & the val losses of length '
              f'{len(self.val_losses)} from the latest ckpt')
        train_losses_len = len(self.train_losses)
        val_losses_len = len(self.val_losses)
        if train_losses_len > val_losses_len:  # training collapsed after train epoch & before val epoch
            self.train_losses = self.train_losses[:val_losses_len]
        elif val_losses_len > train_losses_len:
            raise Exception  # then sth. is wrong
        print('-' * 20)
