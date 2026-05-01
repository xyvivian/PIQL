import argparse
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset

from trainer_embedder.gmm_test import make_NdMclusterGMM_predefined
from trainer_embedder.gmm_trainer_old import (
    ProgramTransformerEncoder,
    ProgramVectorizer,
    build_vocab,
    tokenize_program,
)


@dataclass
class GMMParams:
    means: torch.Tensor
    diag_values: torch.Tensor
    sub_dims: torch.Tensor
    inflated_covariances: torch.Tensor  # shape: (K, |sub_dims|)
    inv_sub_covariances: torch.Tensor   # shape: (K, |sub_dims|)


def sample_base_gmm_params(
    *,
    dim: int,
    num_cluster: int,
    max_mean: int,
    max_var: int,
    inflate_scale: float,
    device: torch.device,
) -> GMMParams:
    # means = rand * randint(-max_mean, max_mean)
    means = torch.rand(num_cluster, dim, device=device) * torch.randint(
        low=-max_mean,
        high=max_mean + 1,
        size=(num_cluster, dim),
        device=device,
    )

    # diag_values = rand * randint(1, max_var)
    diag_values = torch.rand(num_cluster, dim, device=device) * torch.randint(
        low=1,
        high=max_var + 1,
        size=(num_cluster, dim),
        device=device,
    )
    diag_values[diag_values == 0] = max_var / 2

    # sub_dims
    n = np.random.randint(1, dim + 1)
    sub_dims = torch.sort(torch.randperm(dim, device=device)[:n]).values

    covariances = torch.diag_embed(diag_values)

    inflated_covariances = []
    inv_sub_covariances = []

    for cov in covariances:
        cov_copy = cov.clone()
        sub_cov = cov_copy[sub_dims, :][:, sub_dims]
        inv_sub_covariances.append(torch.diagonal(torch.linalg.inv(sub_cov), dim1=-2, dim2=-1))
        cov_copy[sub_dims[:, None], sub_dims] *= inflate_scale
        inflated_covariances.append(torch.diagonal(cov_copy, dim1=-2, dim2=-1)[sub_dims])

    inflated_covariances = torch.stack(inflated_covariances)
    inv_sub_covariances = torch.stack(inv_sub_covariances)

    return GMMParams(
        means=means,
        diag_values=diag_values,
        sub_dims=sub_dims,
        inflated_covariances=inflated_covariances,
        inv_sub_covariances=inv_sub_covariances,
    )


def _compute_subspace_terms(
    diag_values: torch.Tensor,
    sub_dims: torch.Tensor,
    inflate_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    covariances = torch.diag_embed(diag_values)
    inflated_covariances = []
    inv_sub_covariances = []
    for cov in covariances:
        cov_copy = cov.clone()
        sub_cov = cov_copy[sub_dims, :][:, sub_dims]
        inv_sub_covariances.append(torch.diagonal(torch.linalg.inv(sub_cov), dim1=-2, dim2=-1))
        cov_copy[sub_dims[:, None], sub_dims] *= inflate_scale
        inflated_covariances.append(torch.diagonal(cov_copy, dim1=-2, dim2=-1)[sub_dims])
    return torch.stack(inflated_covariances), torch.stack(inv_sub_covariances)


def make_positive_from_base(
    base: GMMParams,
    *,
    inflate_scale: float,
    mean_noise_std: float,
    diag_noise_std: float,
    inf_noise_std: float,
) -> GMMParams:
    means = base.means.clone()
    diag_values = base.diag_values.clone()
    sub_dims = base.sub_dims.clone()
    inflated_covariances = base.inflated_covariances.clone()
    inv_sub_covariances = base.inv_sub_covariances.clone()

    mode = random.choice(["means", "diag", "inflated", "sub_dims"])

    if mode == "means":
        means = means + mean_noise_std * torch.randn_like(means)

    elif mode == "diag":
        scale = 1.0 + diag_noise_std * torch.randn_like(diag_values)
        diag_values = torch.clamp(diag_values * scale, min=1e-4)
        inflated_covariances, inv_sub_covariances = _compute_subspace_terms(
            diag_values=diag_values,
            sub_dims=sub_dims,
            inflate_scale=inflate_scale,
        )

    elif mode == "inflated":
        scale = 1.0 + inf_noise_std * torch.randn_like(inflated_covariances)
        inflated_covariances = torch.clamp(inflated_covariances * scale, min=1e-4)
        # Keep inverse sub-cov from base in this mode (small local perturbation)

    else:  # sub_dims
        dim = diag_values.shape[1]
        n = len(sub_dims)
        if n < dim:
            chosen = set(sub_dims.detach().cpu().tolist())
            outside = [i for i in range(dim) if i not in chosen]
            if outside:
                replace_idx = np.random.randint(0, n)
                sub_dims[replace_idx] = outside[np.random.randint(0, len(outside))]
                sub_dims = torch.sort(sub_dims).values
                inflated_covariances, inv_sub_covariances = _compute_subspace_terms(
                    diag_values=diag_values,
                    sub_dims=sub_dims,
                    inflate_scale=inflate_scale,
                )

    return GMMParams(
        means=means,
        diag_values=diag_values,
        sub_dims=sub_dims,
        inflated_covariances=inflated_covariances,
        inv_sub_covariances=inv_sub_covariances,
    )


def params_to_description(params: GMMParams, *, device: torch.device) -> str:
    num_cluster = params.means.shape[0]
    gmm = make_NdMclusterGMM_predefined(
        means=params.means,
        diag_values=params.diag_values,
        num_cluster=num_cluster,
        embeds=None,
        device=device,
        sub_dim=params.sub_dims,
        inflated_covariances=params.inflated_covariances,
        inv_sub_covariances=params.inv_sub_covariances,
    )
    return gmm.describe_gmm_model()


class GMMTripletDataset(Dataset):
    def __init__(
        self,
        length: int,
        *,
        min_dim: int,
        max_dim: int,
        min_num_cluster: int,
        max_num_cluster: int,
        max_mean: int,
        max_var: int,
        inflate_scale: float,
        generation_device: torch.device,
        mean_noise_std: float,
        diag_noise_std: float,
        inf_noise_std: float,
        num_negatives: int,
    ):
        self.length = length
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.min_num_cluster = min_num_cluster
        self.max_num_cluster = max_num_cluster
        self.max_mean = max_mean
        self.max_var = max_var
        self.inflate_scale = inflate_scale
        self.generation_device = generation_device
        self.mean_noise_std = mean_noise_std
        self.diag_noise_std = diag_noise_std
        self.inf_noise_std = inf_noise_std
        self.num_negatives = num_negatives

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        _ = idx
        dim = int(np.random.randint(self.min_dim, self.max_dim + 1))
        num_cluster = int(np.random.randint(self.min_num_cluster, self.max_num_cluster + 1))

        anchor = sample_base_gmm_params(
            dim=dim,
            num_cluster=num_cluster,
            max_mean=self.max_mean,
            max_var=self.max_var,
            inflate_scale=self.inflate_scale,
            device=self.generation_device,
        )
        positive = make_positive_from_base(
            anchor,
            inflate_scale=self.inflate_scale,
            mean_noise_std=self.mean_noise_std,
            diag_noise_std=self.diag_noise_std,
            inf_noise_std=self.inf_noise_std,
        )

        negatives = []
        for _ in range(self.num_negatives):
            # Each negative can come from an arbitrary parameter regime,
            # including different dimensionality and number of clusters.
            neg_dim = int(np.random.randint(self.min_dim, self.max_dim + 1))
            neg_num_cluster = int(np.random.randint(self.min_num_cluster, self.max_num_cluster + 1))
            # Prefer negatives that differ structurally from anchor (dim and/or clusters).
            if self.min_dim < self.max_dim or self.min_num_cluster < self.max_num_cluster:
                for _retry in range(4):
                    if neg_dim != dim or neg_num_cluster != num_cluster:
                        break
                    neg_dim = int(np.random.randint(self.min_dim, self.max_dim + 1))
                    neg_num_cluster = int(np.random.randint(self.min_num_cluster, self.max_num_cluster + 1))

            neg_max_mean = int(np.random.randint(1, max(2, self.max_mean * 3) + 1))
            neg_max_var = int(np.random.randint(1, max(2, self.max_var * 3) + 1))
            neg_inflate_scale = float(np.random.uniform(1.5, max(2.0, self.inflate_scale * 3.0)))

            negative = sample_base_gmm_params(
                dim=neg_dim,
                num_cluster=neg_num_cluster,
                max_mean=neg_max_mean,
                max_var=neg_max_var,
                inflate_scale=neg_inflate_scale,
                device=self.generation_device,
            )
            negatives.append(negative)

        a_txt = params_to_description(anchor, device=self.generation_device)
        p_txt = params_to_description(positive, device=self.generation_device)
        n_txts = [params_to_description(neg, device=self.generation_device) for neg in negatives]
        return a_txt, p_txt, n_txts


class LitGMMMetricLearner(pl.LightningModule):
    def __init__(
        self,
        encoder: ProgramTransformerEncoder,
        *,
        num_cls: int,
        max_tokens: int | None = None,
        lr: float = 1e-3,
        margin: float = 1.0,
        warmup_ratio: float = 0.1,
        min_lr_ratio: float = 0.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_cls = num_cls
        self.max_tokens = max_tokens
        self.lr = lr
        self.warmup_ratio = warmup_ratio
        self.min_lr_ratio = min_lr_ratio
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    def _encode_text(self, text: str) -> torch.Tensor:
        tokens = tokenize_program(text, num_cls=self.num_cls, max_tokens=self.max_tokens)
        return self.encoder(tokens)

    def training_step(self, batch, batch_idx):
        del batch_idx
        batch_size = len(batch)
        sample_losses = []
        pos_dists = []
        neg_dists = []

        for a_txt, p_txt, n_txts in batch:
            a = self._encode_text(a_txt)
            p = self._encode_text(p_txt)
            neg_embs = torch.stack([self._encode_text(n_txt) for n_txt in n_txts], dim=0)

            a_rep = a.unsqueeze(0).expand(neg_embs.size(0), -1)
            p_rep = p.unsqueeze(0).expand(neg_embs.size(0), -1)

            sample_losses.append(self.loss_fn(a_rep, p_rep, neg_embs))
            pos_dists.append(torch.norm(a - p, dim=-1))
            neg_dists.append(torch.norm(a_rep - neg_embs, dim=-1).mean())

        loss = torch.stack(sample_losses).mean()

        with torch.no_grad():
            pos_dist = torch.stack(pos_dists).mean()
            neg_dist = torch.stack(neg_dists).mean()

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("train/pos_dist", pos_dist, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("train/neg_dist", neg_dist, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        total_steps = int(getattr(self.trainer, "estimated_stepping_batches", 0))
        if total_steps <= 0:
            total_steps = 1000

        warmup_steps = int(total_steps * self.warmup_ratio)
        warmup_steps = min(max(warmup_steps, 1), max(total_steps - 1, 1))

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step + 1) / float(warmup_steps)

            progress = float(current_step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def build_bootstrap_vocabs(
    num_bootstrap: int,
    num_cls: int,
    min_dim: int,
    max_dim: int,
    min_num_cluster: int,
    max_num_cluster: int,
    max_tokens: int | None = None,
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, int]]:
    texts: List[str] = []
    cpu = torch.device("cpu")
    for _ in range(num_bootstrap):
        dim = int(np.random.randint(min_dim, max_dim + 1))
        num_cluster = int(np.random.randint(min_num_cluster, max_num_cluster + 1))
        p = sample_base_gmm_params(
            dim=dim,
            num_cluster=num_cluster,
            max_mean=6,
            max_var=6,
            inflate_scale=5.0,
            device=cpu,
        )
        texts.append(params_to_description(p, device=cpu))

    tokenized = [tokenize_program(t, num_cls=num_cls, max_tokens=max_tokens) for t in texts]
    symbol_vocab = build_vocab(tokenized, "name")
    field_vocab = build_vocab(tokenized, "field")
    family_vocab = build_vocab(tokenized, "family")
    entity_type_vocab = build_vocab(tokenized, "entity_type")
    return symbol_vocab, field_vocab, family_vocab, entity_type_vocab


def parse_args():
    parser = argparse.ArgumentParser(description="Small PyTorch Lightning trainer for GMM text embedder.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps_per_epoch", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--min_lr_ratio", type=float, default=0.0)
    parser.add_argument("--margin", type=float, default=1.0)

    parser.add_argument("--min_dim", type=int, default=2)
    parser.add_argument("--max_dim", type=int, default=100)
    parser.add_argument("--min_num_cluster", type=int, default=1)
    parser.add_argument("--max_num_cluster", type=int, default=5)
    parser.add_argument("--max_mean", type=int, default=6)
    parser.add_argument("--max_var", type=int, default=6)
    parser.add_argument("--inflate_scale", type=float, default=5.0)

    parser.add_argument("--num_cls", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--mean_noise_std", type=float, default=0.1)
    parser.add_argument("--diag_noise_std", type=float, default=0.1)
    parser.add_argument("--inf_noise_std", type=float, default=0.1)
    parser.add_argument("--num_negatives", type=int, default=8)

    parser.add_argument("--seed", type=int, default=42)

    # logging / checkpointing
    parser.add_argument("--logging", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="FoMo-GMM-Embedder")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="./ckpt/gmm_triplet")

    # checkpoint load/resume
    parser.add_argument("--resume_from_ckpt", type=str, default="", help="Path to checkpoint for full trainer resume")
    parser.add_argument("--auto_resume", action="store_true", help="If set, resume from <save_path>/last.ckpt when it exists")
    parser.add_argument("--load_weights_from_ckpt", type=str, default="", help="Path to checkpoint to load model weights only")
    parser.add_argument("--load_weights_strict", action="store_true", help="Use strict=True when loading weights-only checkpoint")

    return parser.parse_args()


def _validate_ckpt_path(ckpt_path: str, arg_name: str) -> str:
    ckpt_path = ckpt_path.strip()
    if not ckpt_path:
        return ""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"{arg_name} does not exist: {ckpt_path}")
    return ckpt_path


def _maybe_resume_ckpt_path(args, save_path: str) -> Optional[str]:
    explicit = _validate_ckpt_path(args.resume_from_ckpt, "--resume_from_ckpt")
    if explicit:
        return explicit

    if args.auto_resume:
        candidate = os.path.join(save_path, "last.ckpt")
        if os.path.isfile(candidate):
            print(f"[ckpt] Auto-resume enabled. Using checkpoint: {candidate}")
            return candidate
        print(f"[ckpt] Auto-resume enabled, but no checkpoint found at: {candidate}")

    return None


def _maybe_load_weights_only(model: pl.LightningModule, args) -> None:
    ckpt_path = _validate_ckpt_path(args.load_weights_from_ckpt, "--load_weights_from_ckpt")
    if not ckpt_path:
        return

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    load_result = model.load_state_dict(state_dict, strict=args.load_weights_strict)
    print(f"[ckpt] Loaded weights from: {ckpt_path}")
    if hasattr(load_result, "missing_keys") and load_result.missing_keys:
        print(f"[ckpt] missing_keys: {load_result.missing_keys}")
    if hasattr(load_result, "unexpected_keys") and load_result.unexpected_keys:
        print(f"[ckpt] unexpected_keys: {load_result.unexpected_keys}")


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_run_name = (
        f"gmm_triplet.d{args.min_dim}-{args.max_dim}.k{args.min_num_cluster}-{args.max_num_cluster}.cls{args.num_cls}."
        f"E{args.epochs}.step{args.steps_per_epoch}.bs{args.batch_size}.lr{args.lr}."
        f"nneg{args.num_negatives}.{current_time}"
    )
    run_name = args.wandb_run_name if args.wandb_run_name else default_run_name

    save_path = os.path.join(args.save_dir, run_name, f"seed{args.seed}")
    os.makedirs(save_path, exist_ok=True)

    symbol_vocab, field_vocab, family_vocab, entity_type_vocab = build_bootstrap_vocabs(
        num_bootstrap=64,
        num_cls=args.num_cls,
        min_dim=args.min_dim,
        max_dim=args.max_dim,
        min_num_cluster=args.min_num_cluster,
        max_num_cluster=args.max_num_cluster,
        max_tokens=args.max_tokens,
    )

    vectorizer = ProgramVectorizer(
        symbol_vocab=symbol_vocab,
        field_vocab=field_vocab,
        family_vocab=family_vocab,
        entity_type_vocab=entity_type_vocab,
        d_model=args.d_model,
        max_dim=max(256, args.max_dim + 32),
        max_entity_id=max(32, args.max_num_cluster + 8),
        max_index_pos=max(256, args.max_dim + 32),
    )

    encoder = ProgramTransformerEncoder(
        vectorizer=vectorizer,
        num_cls=args.num_cls,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )

    model = LitGMMMetricLearner(
        encoder=encoder,
        num_cls=args.num_cls,
        max_tokens=args.max_tokens,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        min_lr_ratio=args.min_lr_ratio,
        margin=args.margin,
    )

    resume_ckpt_path = _maybe_resume_ckpt_path(args, save_path=save_path)
    if resume_ckpt_path is None:
        _maybe_load_weights_only(model, args)
    elif args.load_weights_from_ckpt.strip():
        print("[ckpt] --resume_from_ckpt/--auto_resume provided; ignoring --load_weights_from_ckpt")

    train_ds = GMMTripletDataset(
        length=args.steps_per_epoch * args.batch_size,
        min_dim=args.min_dim,
        max_dim=args.max_dim,
        min_num_cluster=args.min_num_cluster,
        max_num_cluster=args.max_num_cluster,
        max_mean=args.max_mean,
        max_var=args.max_var,
        inflate_scale=args.inflate_scale,
        generation_device=torch.device("cpu"),
        mean_noise_std=args.mean_noise_std,
        diag_noise_std=args.diag_noise_std,
        inf_noise_std=args.inf_noise_std,
        num_negatives=args.num_negatives,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda x: x,
    )

    logger = WandbLogger(project=args.wandb_project, name=run_name) if args.logging else None
    if logger is not None:
        logger.log_hyperparams(vars(args))
        logger.watch(model, log="all", log_freq=50)

    ckpt_callbacks = [
        ModelCheckpoint(
            monitor="train/loss_epoch",
            mode="min",
            save_top_k=3,
            dirpath=save_path,
            filename="min-trainloss-{epoch:02d}-{step:06d}",
            save_last=True,
            verbose=True,
        ),
        ModelCheckpoint(
            dirpath=save_path,
            filename="epoch-{epoch:02d}",
            save_top_k=-1,
            every_n_epochs=1,
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        limit_train_batches=args.steps_per_epoch,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=ckpt_callbacks,
    )
    trainer.fit(model, train_dataloaders=train_loader, ckpt_path=resume_ckpt_path)


if __name__ == "__main__":
    main()
