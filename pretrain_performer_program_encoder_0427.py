"""Entry point for FoMo-Meta internal pretraining.

This script stays self-contained within the FoMo-Meta package by inserting the
package root into `sys.path` and importing sibling modules by their short names.
"""

import os
import random
import sys
import time
from datetime import datetime
from typing import Any, Callable, Optional

os.environ['NCCL_TIMEOUT'] = '3600'
os.environ['NCCL_BLOCKING_WAIT'] = '1'
os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'

_fomo_meta_root = os.path.dirname(os.path.abspath(__file__))
if _fomo_meta_root not in sys.path:
    sys.path.insert(0, _fomo_meta_root)

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from torch import nn

#from data_prior.generator_embed import PriorTrainDataGenerator
#from data_prior.generator_embed_0413_gmm_predefined import PriorTrainDataGenerator
from data_prior.generator_program_encoder import PriorTrainDataGenerator
from model_meta_0413 import encoders
from trainer.trainer_performer_program_encoder_0427 import ZeroShotOD

class single_eval_pos_generator:
    """Generate the first evaluation position used by the training batches."""

    def __init__(self, mode: str, num_test_x: int, seq_len: int, num_R: Optional[int]):
        self.mode = mode
        self.num_test_x = num_test_x
        self.seq_len = seq_len
        self.single_eval_pos: Optional[int] = None
        self.num_R = 0 if num_R is None else num_R

        if self.mode == 'constant':
            assert self.num_test_x > self.num_R, \
                print(f'we are in constant mode, please make sure '
                      f'seq-len{self.seq_len}-num_test_x{self.num_test_x}+num_R{self.num_R}<seq-len{self.seq_len}')

    def generate(self, seed: Optional[int] = None) -> int:
        if self.mode == 'constant':
            self.single_eval_pos = self.seq_len - self.num_test_x + self.num_R
        else:
            if seed is None:
                self.single_eval_pos = random.choices(range(0, self.seq_len - self.num_R))[0] + self.num_R
            else:
                rng = random.Random(seed)
                self.single_eval_pos = rng.choices(range(0, self.seq_len - self.num_R))[0] + self.num_R
        # single_eval_pos >= num_R
        return self.single_eval_pos


def make_pl_model(
    cfg: DictConfig,
    get_batch_function: Callable[..., Any],
    seq_len: int,
    num_features: int,
    hps: dict,
    generator_mode: str = 'constant',
    num_class: int = 2,
    num_R: Optional[int] = None,
    model_para_dict: Optional[dict] = None,
    train_extra_dict: Optional[dict] = None,
    resume_from_ckpt: bool = False,
) -> ZeroShotOD:
    """Build the Lightning module used for internal FoMo-Meta training."""
    criterion = nn.CrossEntropyLoss(weight=torch.ones(size=(num_class,)) / num_class, reduction='none',
                                    ignore_index=hps['ignore_index'])

    single_eval_pos_gen = single_eval_pos_generator(mode=generator_mode, num_test_x=hps['num_test_x'], seq_len=seq_len,
                                                    num_R=num_R)
    print('T0 value is', cfg.train.T0)
    pl_model = ZeroShotOD(cfg=cfg,
                          priordataloader_class_or_get_batch=get_batch_function, criterion=criterion,
                          encoder_generator=encoders.get_normalized_uniform_encoder(encoders.Linear),
                          input_to_internal_encoder_generator = encoders.get_normalized_uniform_seq_encoder(
                              lambda in_dim, out_dim, seq_len, n_tokens: encoders.MLPSeqEncoder(
                                  num_features=in_dim,
                                  emsize=out_dim,
                                  seq_len=seq_len,
                                  n_tokens=n_tokens,
                              )
                          ),
                          internal_encoder_generator = encoders.get_normalized_uniform_encoder(encoders.Linear),
                          y_encoder_generator=encoders.Linear,
                          extra_prior_kwargs_dict={'num_features': num_features,
                                                   'hyperparameters': hps,
                                                   'pt_dataloader': {'num_workers': 0, 'pin_memory': True},
                                                   'num_R': num_R},
                          single_eval_pos_gen=single_eval_pos_gen,
                          progress_bar=True,
                          train_extra_dict=train_extra_dict,
                          resume_from_ckpt=resume_from_ckpt,
                          T0=cfg.train.T0,
                          model_para_dict=model_para_dict,
                          )
    return pl_model


def set_seed(seed: int = 42) -> None:
    # https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


@hydra.main(version_base='1.3', config_path='configs', config_name='config')
def main(cfg: DictConfig) -> None:
    prior_train_data_gen = PriorTrainDataGenerator(cfg=cfg)
    train_cfg = cfg.train
    seq_len = train_cfg.seq_len
    hyperparameters = train_cfg.hyperparameters
    batch_size = train_cfg.batch_size
    epochs = train_cfg.epochs
    steps_per_epoch = train_cfg.steps_per_epoch
    lr = train_cfg.lr
    emsize = train_cfg.emsize
    nhead = train_cfg.nhead
    nhid = train_cfg.nhid
    nlayer = train_cfg.nlayer
    num_R = train_cfg.num_R
    reuse_data_every_n = train_cfg.reuse_data_every_n
    gen_one_train_one = train_cfg.gen_one_train_one
    resume_from_ckpt = train_cfg.resume_from_ckpt
    apply_linear_transform = train_cfg.apply_linear_transform
    seed = train_cfg.seed
    num_device = train_cfg.num_device
    T0 = train_cfg.T0
    max_feature_dim = cfg.prior.mixture.max_feature_dim
    set_seed(seed=seed)
    generator_mode = hyperparameters['mode']

    current_time = datetime.now().strftime('%Y%m%d')
    config_details = f'meta_performer_program_encoder_gmm_only_0427_context{seq_len}.feat{max_feature_dim}.R{num_R}.LT{apply_linear_transform}.gen1tr1{gen_one_train_one}.reuse{reuse_data_every_n}.E{epochs}.step{steps_per_epoch}.bs{batch_size}.lr{lr}.emb{emsize}.hdim{nhid}.nhead{nhead}.nlayer{nlayer}.ndevice{num_device}.T0{T0}_{current_time}'
    if train_cfg.last_layer_no_R:
        config_details = f'last_layer_no_R{train_cfg.last_layer_no_R}.{config_details}'

    if train_cfg.extra_heading != '':
        config_details = f'{train_cfg.extra_heading}.{config_details}'

    save_path = f'{train_cfg.model_dir}/{config_details}/seed{seed}'

    os.makedirs(save_path, exist_ok=True)
    start_time = time.time()
    zero_shot_od_pl_model = make_pl_model(cfg=cfg,
                                          get_batch_function=prior_train_data_gen.get_batch_all_models,
                                          seq_len=seq_len, 
                                          num_features=max_feature_dim,
                                          num_class=2,
                                          generator_mode=generator_mode, hps=hyperparameters,
                                          num_R=num_R,
                                          model_para_dict={'num_R': num_R,
                                                           'last_layer_no_R': train_cfg.last_layer_no_R},
                                          train_extra_dict=None if not gen_one_train_one else {
                                                     'prior_train_data_gen': prior_train_data_gen},
                                          resume_from_ckpt=resume_from_ckpt)

    logger = WandbLogger(project='Metafeature', name=config_details) if train_cfg.logging else None

    ckpt_callbacks = []
    train_ckpt_callback = ModelCheckpoint(
        monitor='train_loss',
        mode='min',
        save_top_k=3,
        dirpath=save_path,
        filename='min-trainloss-{epoch:02d}-{train_loss:.2f}',
        verbose=True,
        save_last=True
    )
    ckpt_callbacks.append(train_ckpt_callback)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path,
        filename="{epoch:02d}-{train_loss:.2f}",
        save_top_k=-1,
        every_n_epochs=50,
    )
    ckpt_callbacks.append(checkpoint_callback)   

    print(f'setting the max epochs to {epochs}')
    trainer = pl.Trainer(
        strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=ckpt_callbacks,
        logger=logger,
        max_epochs=epochs,
        enable_progress_bar=True,
        limit_val_batches=None if train_cfg.use_validation else 0,
        check_val_every_n_epoch=1,
        devices=num_device,
        reload_dataloaders_every_n_epochs=1,
        gradient_clip_val=1.0,
        num_sanity_val_steps=1,  
        use_distributed_sampler=False
    )
    
    trainer.fit(zero_shot_od_pl_model)#,ckpt_path='/ocean/projects/cis250290p/xding/FoMo-Meta_0413/ckpt/meta_program_encoder_gmm_only_context5000.feat100.R500.LTFalse.gen1tr1True.reuse100.E1500.step1000.bs2.lr0.0001.emb256.hdim512.nhead4.nlayer2.ndevice4.T00_20260421/seed0/epoch=149-train_loss=0.01.ckpt')

    train_time = time.time() - start_time
    print('total training time: {}'.format(train_time / 60))


if __name__ == "__main__":
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    main()
    
    
    
#CUDA_VISIBLE_DEVICES=0,1,2,3 python pretrain_performer_encoder.py train.apply_linear_transform=False train.gen_one_train_one=True  train.seed=0 train.num_R=500 train.epochs=1500