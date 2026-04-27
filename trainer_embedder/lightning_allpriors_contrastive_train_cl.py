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
from torch.distributions import Beta, Exponential, Normal, StudentT
from torch.utils.data import DataLoader, Dataset

from copula_test import (
    CorpulaGenerator,
    make_copula_predefined,
)
from gmm_test import make_NdMclusterGMM_predefined
from gmm_trainer_cls import (
    ProgramTransformerEncoder,
    ProgramVectorizer,
    build_vocab,
    tokenize_program,
)
from scm_test import (
    StructuralCausalModel,
    _detect_activation_name,
    describe_scm_model,
    make_scm_predefined,
)


FAMILY_TO_ID = {"gmm": 0, "copula": 1, "scm": 2}
GMM_OUTLIERS = ["inflated_cov"]
COPULA_OUTLIERS = ["disturb_covariance", "perturb_u_values"]
SCM_OUTLIERS = ["contextual", "prob"]


@dataclass
class GMMParams:
    means: torch.Tensor
    diag_values: torch.Tensor
    sub_dims: torch.Tensor
    inflated_covariances: torch.Tensor  # (K, |sub_dims|)
    inv_sub_covariances: torch.Tensor   # (K, |sub_dims|)


@dataclass
class CopulaParams:
    num_dims: int
    anomaly_type: str
    chol_base: torch.Tensor
    marginals: List[dict]
    perturb_u_prototypes: Optional[List[List[int]]] = None
    disturb_cov_prototypes: Optional[List[List[int]]] = None
    num_prototypes: int = 16


@dataclass
class SCMParams:
    num_layers: int
    hidden_size: int
    chosen_nodes: List[int]
    activation_names: List[str]
    layer_weights: List[torch.Tensor]
    layer_masks: List[torch.Tensor]
    outlier_type: str
    contextual_perturb_prob: float = 0.2
    contextual_num_mask_types: int = 16
    prototype_masks: Optional[List[torch.Tensor]] = None
    prototype_node_ids: Optional[List[dict]] = None
    prob_num_mask_types: int = 16
    high_noise: float = 5.0
    high_noise_prob: float = 0.2
    batch_factor: float = 2.0
    prob_prototype_scales: Optional[List[torch.Tensor]] = None
    prob_prototype_node_ids: Optional[List[dict]] = None


@dataclass
class PriorExample:
    family: str
    outlier_type: str
    dim: int
    payload: object


def _compute_subspace_terms(diag_values: torch.Tensor, sub_dims: torch.Tensor, inflate_scale: float):
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


def sample_base_gmm_params(dim: int, num_cluster: int, max_mean: int, max_var: int, inflate_scale: float, device: torch.device) -> GMMParams:
    means = torch.rand(num_cluster, dim, device=device) * torch.randint(
        low=-max_mean, high=max_mean + 1, size=(num_cluster, dim), device=device
    )
    diag_values = torch.rand(num_cluster, dim, device=device) * torch.randint(
        low=1, high=max_var + 1, size=(num_cluster, dim), device=device
    )
    diag_values[diag_values == 0] = max_var / 2
    n = np.random.randint(1, dim + 1)
    sub_dims = torch.sort(torch.randperm(dim, device=device)[:n]).values

    inflated_covariances, inv_sub_covariances = _compute_subspace_terms(diag_values, sub_dims, inflate_scale)
    return GMMParams(
        means=means,
        diag_values=diag_values,
        sub_dims=sub_dims,
        inflated_covariances=inflated_covariances,
        inv_sub_covariances=inv_sub_covariances,
    )


def perturb_gmm(base: GMMParams, inflate_scale: float, strength: float) -> GMMParams:
    means = base.means.clone()
    diag_values = base.diag_values.clone()
    sub_dims = base.sub_dims.clone()
    inflated_covariances = base.inflated_covariances.clone()
    inv_sub_covariances = base.inv_sub_covariances.clone()

    mode = random.choice(["means", "diag", "sub_dims", "inflated"])
    if mode == "means":
        means = means + strength * torch.randn_like(means)
    elif mode == "diag":
        diag_values = torch.clamp(diag_values * (1.0 + strength * torch.randn_like(diag_values)), min=1e-4)
        inflated_covariances, inv_sub_covariances = _compute_subspace_terms(diag_values, sub_dims, inflate_scale)
    elif mode == "sub_dims":
        d = diag_values.shape[1]
        if len(sub_dims) < d:
            chosen = set(sub_dims.detach().cpu().tolist())
            outside = [i for i in range(d) if i not in chosen]
            if outside:
                replace_idx = np.random.randint(0, len(sub_dims))
                sub_dims[replace_idx] = outside[np.random.randint(0, len(outside))]
                sub_dims = torch.sort(sub_dims).values
                inflated_covariances, inv_sub_covariances = _compute_subspace_terms(diag_values, sub_dims, inflate_scale)
    else:
        inflated_covariances = torch.clamp(
            inflated_covariances * (1.0 + strength * torch.randn_like(inflated_covariances)), min=1e-4
        )

    return GMMParams(
        means=means,
        diag_values=diag_values,
        sub_dims=sub_dims,
        inflated_covariances=inflated_covariances,
        inv_sub_covariances=inv_sub_covariances,
    )


def gmm_to_text(params: GMMParams, device: torch.device) -> str:
    gmm = make_NdMclusterGMM_predefined(
        means=params.means,
        diag_values=params.diag_values,
        num_cluster=params.means.shape[0],
        embeds=None,
        device=device,
        sub_dim=params.sub_dims,
        inflated_covariances=params.inflated_covariances,
        inv_sub_covariances=params.inv_sub_covariances,
    )
    return gmm.describe_gmm_model()


def _spec_to_predefined(spec: dict) -> dict:
    if spec.get("kind") == "interp":
        lo = float(spec.get("lo", -1.0))
        hi = float(spec.get("hi", 1.0))
        grid = int(spec.get("u", torch.tensor([])).numel()) if "u" in spec else 2000
        return {"DIST": "interp", "PARAMS": [lo, hi, grid]}

    dist = spec.get("dist", None)
    if isinstance(dist, Normal):
        return {"DIST": "normal", "PARAMS": [float(dist.loc), float(dist.scale)]}
    if isinstance(dist, Beta):
        return {"DIST": "beta", "PARAMS": [float(dist.concentration1), float(dist.concentration0)]}
    if isinstance(dist, Exponential):
        return {"DIST": "exponential", "PARAMS": [float(dist.rate)]}
    if isinstance(dist, StudentT):
        return {"DIST": "studentt", "PARAMS": [float(dist.df), float(dist.loc), float(dist.scale)]}
    return {"DIST": "normal", "PARAMS": [0.0, 1.0]}


def _extract_copula_params(model: CorpulaGenerator) -> CopulaParams:
    marginals = [_spec_to_predefined(s) for s in model.specs]
    perturb_u_prototypes = None
    disturb_cov_prototypes = None

    if model.anomaly_types == "perturb_u_values" and model.perturb_u_subset_masks is not None:
        perturb_u_prototypes = [
            mask.nonzero(as_tuple=True)[0].detach().cpu().tolist()
            for mask in model.perturb_u_subset_masks
        ]
    if model.anomaly_types == "disturb_covariance" and model.disturb_cov_block_starts is not None:
        disturb_cov_prototypes = []
        for s, e in zip(model.disturb_cov_block_starts, model.disturb_cov_block_ends):
            disturb_cov_prototypes.append(list(range(int(s.item()), int(e.item()))))

    return CopulaParams(
        num_dims=int(model.num_dims),
        anomaly_type=str(model.anomaly_types),
        chol_base=model.chol_base.detach().clone(),
        marginals=marginals,
        perturb_u_prototypes=perturb_u_prototypes,
        disturb_cov_prototypes=disturb_cov_prototypes,
        num_prototypes=int(model.num_prototypes),
    )


def sample_base_copula_params(dim: int, outlier_type: str, num_prototypes: int, device: torch.device) -> CopulaParams:
    model = CorpulaGenerator(
        num_dims=dim,
        device=str(device),
        anomaly_types=outlier_type,
        num_prototypes=num_prototypes,
    )
    return _extract_copula_params(model)


def perturb_copula(base: CopulaParams, strength: float) -> CopulaParams:
    new_marginals = [dict(m) for m in base.marginals]

    # Slightly perturb one marginal parameter
    if len(new_marginals) > 0:
        i = np.random.randint(0, len(new_marginals))
        dist = new_marginals[i]["DIST"].lower()
        p = list(new_marginals[i]["PARAMS"])
        if dist == "normal" and len(p) >= 2:
            p[0] = float(p[0]) + float(np.random.randn() * strength)
            p[1] = max(1e-3, float(p[1]) * (1.0 + float(np.random.randn() * strength * 0.5)))
        elif dist == "beta" and len(p) >= 2:
            p[0] = max(0.1, float(p[0]) * (1.0 + float(np.random.randn() * strength * 0.5)))
            p[1] = max(0.1, float(p[1]) * (1.0 + float(np.random.randn() * strength * 0.5)))
        elif dist == "exponential" and len(p) >= 1:
            p[0] = max(1e-3, float(p[0]) * (1.0 + float(np.random.randn() * strength * 0.5)))
        elif dist == "studentt" and len(p) >= 1:
            p[0] = max(2.0, float(p[0]) + float(np.random.randn() * strength * 2.0))
        new_marginals[i]["PARAMS"] = p

    perturb_u_prototypes = None
    disturb_cov_prototypes = None
    if base.anomaly_type == "perturb_u_values" and base.perturb_u_prototypes is not None:
        perturb_u_prototypes = [list(x) for x in base.perturb_u_prototypes]
        if len(perturb_u_prototypes) > 0:
            p = np.random.randint(0, len(perturb_u_prototypes))
            dims = set(perturb_u_prototypes[p])
            if np.random.rand() < 0.5 and len(dims) > 1:
                dims.remove(random.choice(list(dims)))
            else:
                dims.add(np.random.randint(0, base.num_dims))
            perturb_u_prototypes[p] = sorted(list(dims))
    elif base.anomaly_type == "disturb_covariance" and base.disturb_cov_prototypes is not None:
        disturb_cov_prototypes = [list(x) for x in base.disturb_cov_prototypes]
        if len(disturb_cov_prototypes) > 0:
            p = np.random.randint(0, len(disturb_cov_prototypes))
            block = disturb_cov_prototypes[p]
            if len(block) > 0:
                shift = np.random.choice([-1, 1])
                new_block = [min(base.num_dims - 1, max(0, b + shift)) for b in block]
                new_block = sorted(set(new_block))
                if len(new_block) > 0:
                    s, e = min(new_block), max(new_block)
                    disturb_cov_prototypes[p] = list(range(s, e + 1))

    return CopulaParams(
        num_dims=base.num_dims,
        anomaly_type=base.anomaly_type,
        chol_base=base.chol_base.clone(),
        marginals=new_marginals,
        perturb_u_prototypes=perturb_u_prototypes if perturb_u_prototypes is not None else base.perturb_u_prototypes,
        disturb_cov_prototypes=disturb_cov_prototypes if disturb_cov_prototypes is not None else base.disturb_cov_prototypes,
        num_prototypes=base.num_prototypes,
    )


def copula_to_text(params: CopulaParams, device: torch.device) -> str:
    model = make_copula_predefined(
        num_dims=params.num_dims,
        chol_base=params.chol_base,
        marginals=params.marginals,
        device=str(device),
        anomaly_type=params.anomaly_type,
        perturb_u_prototypes=params.perturb_u_prototypes,
        disturb_cov_prototypes=params.disturb_cov_prototypes,
        num_prototypes=params.num_prototypes,
    )
    return model.describe_model_specs()


def _extract_scm_params(model: StructuralCausalModel) -> SCMParams:
    activation_names = [_detect_activation_name(fn) for fn in model.activations]
    layer_weights = [layer.weight.detach().clone() for layer in model.mlp.layers]
    layer_masks = [layer.mask.detach().clone() for layer in model.mlp.layers]

    return SCMParams(
        num_layers=int(model.l),
        hidden_size=int(model.h),
        chosen_nodes=[int(x) for x in model.chosen_nodes],
        activation_names=activation_names,
        layer_weights=layer_weights,
        layer_masks=layer_masks,
        outlier_type=str(model.outlier_type),
        contextual_perturb_prob=float(getattr(model, "contextual_perturb_prob", 0.2)),
        contextual_num_mask_types=int(getattr(model, "contextual_num_mask_types", 16)),
        prototype_masks=[x.detach().clone() for x in model.prototype_masks] if model.prototype_masks is not None else None,
        prototype_node_ids=getattr(model, "prototype_node_ids", None),
        prob_num_mask_types=int(getattr(model, "prob_num_mask_types", 16)),
        high_noise=float(getattr(model, "last_prob_outlier_summary", {}).get("high_noise", 5.0)) if getattr(model, "last_prob_outlier_summary", None) else 5.0,
        high_noise_prob=float(getattr(model, "last_prob_outlier_summary", {}).get("high_noise_prob", 0.2)) if getattr(model, "last_prob_outlier_summary", None) else 0.2,
        batch_factor=float(getattr(model, "last_prob_outlier_summary", {}).get("batch_factor", 2.0)) if getattr(model, "last_prob_outlier_summary", None) else 2.0,
        prob_prototype_scales=[x.detach().clone() for x in model.prob_prototype_scales] if model.prob_prototype_scales is not None else None,
        prob_prototype_node_ids=getattr(model, "prob_prototype_node_ids", None),
    )


def sample_base_scm_params(dim: int, outlier_type: str, device: torch.device) -> SCMParams:
    max_num_layer = 5
    min_num_layer = max(int(np.sqrt(dim)) - 3, 2)
    min_hidden_size = max(int(math.floor(dim / min_num_layer)) + 2, 2)
    max_hidden_size = min(min_hidden_size + 7, 40)

    model = StructuralCausalModel(
        num_features=dim,
        min_num_layer=min_num_layer,
        max_num_layer=max_num_layer,
        min_hidden_size=min_hidden_size,
        max_hidden_size=max_hidden_size,
        device=str(device),
        drop_weight_prob=0.6,
        outlier_type=outlier_type,
    )
    if outlier_type == "prob":
        _ = model.sample_prob_outliers(num_samples=16, high_noise=5.0, high_noise_prob=0.2)
    return _extract_scm_params(model)


def perturb_scm(base: SCMParams, strength: float) -> SCMParams:
    layer_weights = [w.clone() for w in base.layer_weights]
    layer_masks = [m.clone() for m in base.layer_masks]

    # small structural perturbation on one layer weight matrix
    l = np.random.randint(0, len(layer_weights))
    layer_weights[l] = layer_weights[l] + strength * torch.randn_like(layer_weights[l])

    return SCMParams(
        num_layers=base.num_layers,
        hidden_size=base.hidden_size,
        chosen_nodes=list(base.chosen_nodes),
        activation_names=list(base.activation_names),
        layer_weights=layer_weights,
        layer_masks=layer_masks,
        outlier_type=base.outlier_type,
        contextual_perturb_prob=base.contextual_perturb_prob,
        contextual_num_mask_types=base.contextual_num_mask_types,
        prototype_masks=[x.clone() for x in base.prototype_masks] if base.prototype_masks is not None else None,
        prototype_node_ids=base.prototype_node_ids,
        prob_num_mask_types=base.prob_num_mask_types,
        high_noise=base.high_noise,
        high_noise_prob=base.high_noise_prob,
        batch_factor=base.batch_factor,
        prob_prototype_scales=[x.clone() for x in base.prob_prototype_scales] if base.prob_prototype_scales is not None else None,
        prob_prototype_node_ids=base.prob_prototype_node_ids,
    )


def scm_to_text(params: SCMParams, device: torch.device) -> str:
    model = make_scm_predefined(
        device=str(device),
        num_layers=params.num_layers,
        hidden_size=params.hidden_size,
        chosen_nodes=params.chosen_nodes,
        activation_names=params.activation_names,
        layer_weights=params.layer_weights,
        layer_masks=params.layer_masks,
        outlier_type=params.outlier_type,
        contextual_perturb_prob=params.contextual_perturb_prob,
        contextual_num_mask_types=params.contextual_num_mask_types,
        prototype_masks=params.prototype_masks,
        prototype_node_ids=params.prototype_node_ids,
        prob_num_mask_types=params.prob_num_mask_types,
        high_noise=params.high_noise,
        high_noise_prob=params.high_noise_prob,
        batch_factor=params.batch_factor,
        prob_prototype_scales=params.prob_prototype_scales,
        prob_prototype_node_ids=params.prob_prototype_node_ids,
    )
    return describe_scm_model(model)


def sample_prior_example(family: str, outlier_type: str, dim: int, device: torch.device, args) -> PriorExample:
    if family == "gmm":
        num_cluster = int(np.random.randint(args.min_num_cluster, args.max_num_cluster + 1))
        payload = sample_base_gmm_params(
            dim=dim,
            num_cluster=num_cluster,
            max_mean=args.max_mean,
            max_var=args.max_var,
            inflate_scale=args.inflate_scale,
            device=device,
        )
    elif family == "copula":
        payload = sample_base_copula_params(
            dim=dim,
            outlier_type=outlier_type,
            num_prototypes=args.num_prototypes,
            device=device,
        )
    elif family == "scm":
        payload = sample_base_scm_params(
            dim=dim,
            outlier_type=outlier_type,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported family: {family}")

    return PriorExample(family=family, outlier_type=outlier_type, dim=dim, payload=payload)


def perturb_prior_example(ex: PriorExample, strength: float, args) -> PriorExample:
    if ex.family == "gmm":
        payload = perturb_gmm(ex.payload, inflate_scale=args.inflate_scale, strength=strength)
    elif ex.family == "copula":
        payload = perturb_copula(ex.payload, strength=strength)
    elif ex.family == "scm":
        payload = perturb_scm(ex.payload, strength=strength)
    else:
        raise ValueError(f"Unsupported family: {ex.family}")
    return PriorExample(family=ex.family, outlier_type=ex.outlier_type, dim=ex.dim, payload=payload)


def prior_to_text(ex: PriorExample, device: torch.device) -> str:
    if ex.family == "gmm":
        return gmm_to_text(ex.payload, device=device)
    if ex.family == "copula":
        return copula_to_text(ex.payload, device=device)
    if ex.family == "scm":
        return scm_to_text(ex.payload, device=device)
    raise ValueError(f"Unsupported family: {ex.family}")


def random_outlier_type(family: str) -> str:
    if family == "gmm":
        return random.choice(GMM_OUTLIERS)
    if family == "copula":
        return random.choice(COPULA_OUTLIERS)
    if family == "scm":
        return random.choice(SCM_OUTLIERS)
    raise ValueError(f"Unsupported family: {family}")


class MultiPriorContrastiveDataset(Dataset):
    def __init__(self, length: int, args, generation_device: torch.device):
        self.length = length
        self.args = args
        self.generation_device = generation_device
        self.current_hard_negative_ratio = float(args.hard_negative_ratio)
        self.current_hard_strength = float(args.hard_strength)

    def set_curriculum(self, hard_negative_ratio: Optional[float] = None, hard_strength: Optional[float] = None):
        if hard_negative_ratio is not None:
            self.current_hard_negative_ratio = float(min(max(hard_negative_ratio, 0.0), 1.0))
        if hard_strength is not None:
            self.current_hard_strength = float(max(hard_strength, 1e-6))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        del idx
        family = random.choice(["gmm", "copula", "scm"])
        outlier_type = random_outlier_type(family)
        dim = int(np.random.randint(self.args.min_dim, self.args.max_dim + 1))

        anchor = sample_prior_example(family, outlier_type, dim, self.generation_device, self.args)
        positive = perturb_prior_example(anchor, strength=self.args.pos_strength, args=self.args)

        negatives = []
        negative_is_hard = []
        for _ in range(self.args.num_negatives):
            if random.random() < self.current_hard_negative_ratio:
                # hard negative: same family/outlier/dim but stronger perturbation
                neg = perturb_prior_example(anchor, strength=self.current_hard_strength, args=self.args)
                negative_is_hard.append(1)
            else:
                # easy negative: usually different family and potentially different dimension/outlier
                neg_family = random.choice([f for f in ["gmm", "copula", "scm"] if f != family])
                neg_outlier = random_outlier_type(neg_family)
                neg_dim = int(np.random.randint(self.args.min_dim, self.args.max_dim + 1))
                neg = sample_prior_example(neg_family, neg_outlier, neg_dim, self.generation_device, self.args)
                negative_is_hard.append(0)
            negatives.append(neg)

        a_txt = prior_to_text(anchor, device=self.generation_device)
        p_txt = prior_to_text(positive, device=self.generation_device)
        n_txts = [prior_to_text(n, device=self.generation_device) for n in negatives]
        return a_txt, p_txt, n_txts, negative_is_hard, FAMILY_TO_ID[family]


class LitMultiPriorContrastive(pl.LightningModule):
    def __init__(
        self,
        encoder: ProgramTransformerEncoder,
        *,
        num_cls: int,
        d_model: int,
        max_tokens: Optional[int],
        lr: float,
        easy_margin: float,
        hard_margin: float,
        hard_weight: float,
        hard_mining_weight: float,
        proxy_weight: float,
        proxy_temp: float,
        warmup_ratio: float,
        min_lr_ratio: float,
        use_curriculum: bool,
        curriculum_warmup_epochs: int,
        curriculum_hard_ratio_start: float,
        curriculum_hard_ratio_end: float,
        curriculum_hard_strength_start: float,
        curriculum_hard_strength_end: float,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_cls = num_cls
        self.max_tokens = max_tokens
        self.lr = lr
        self.easy_margin = easy_margin
        self.hard_margin = hard_margin
        self.hard_weight = hard_weight
        self.hard_mining_weight = hard_mining_weight
        self.proxy_weight = proxy_weight
        self.proxy_temp = proxy_temp
        self.warmup_ratio = warmup_ratio
        self.min_lr_ratio = min_lr_ratio
        self.use_curriculum = use_curriculum
        self.curriculum_warmup_epochs = max(int(curriculum_warmup_epochs), 0)
        self.curriculum_hard_ratio_start = float(curriculum_hard_ratio_start)
        self.curriculum_hard_ratio_end = float(curriculum_hard_ratio_end)
        self.curriculum_hard_strength_start = float(curriculum_hard_strength_start)
        self.curriculum_hard_strength_end = float(curriculum_hard_strength_end)

        self.emb_dim = int(num_cls * d_model)
        self.family_proxies = nn.Parameter(torch.randn(len(FAMILY_TO_ID), self.emb_dim))

    def _curriculum_value(self, start: float, end: float) -> float:
        if not self.use_curriculum:
            return float(end)

        max_epochs = int(getattr(self.trainer, "max_epochs", 1))
        epoch = int(getattr(self.trainer, "current_epoch", 0))

        if max_epochs <= 1:
            return float(end)

        if epoch < self.curriculum_warmup_epochs:
            progress = 0.0
        else:
            denom = max(max_epochs - self.curriculum_warmup_epochs - 1, 1)
            progress = float(epoch - self.curriculum_warmup_epochs) / float(denom)
            progress = min(max(progress, 0.0), 1.0)
        return float(start + (end - start) * progress)

    def on_train_epoch_start(self) -> None:
        ratio = self._curriculum_value(self.curriculum_hard_ratio_start, self.curriculum_hard_ratio_end)
        strength = self._curriculum_value(self.curriculum_hard_strength_start, self.curriculum_hard_strength_end)

        ratio = float(min(max(ratio, 0.0), 1.0))
        strength = float(max(strength, 1e-6))

        train_dl = getattr(self.trainer, "train_dataloader", None)
        if train_dl is not None:
            dls = train_dl if isinstance(train_dl, (list, tuple)) else [train_dl]
            for dl in dls:
                ds = getattr(dl, "dataset", None)
                if ds is not None and hasattr(ds, "set_curriculum"):
                    ds.set_curriculum(hard_negative_ratio=ratio, hard_strength=strength)

        self.log("curriculum/hard_negative_ratio", ratio, on_step=False, on_epoch=True, prog_bar=False)
        self.log("curriculum/hard_strength", strength, on_step=False, on_epoch=True, prog_bar=False)

    def _encode_text(self, text: str) -> torch.Tensor:
        tokens = tokenize_program(text, num_cls=self.num_cls, max_tokens=self.max_tokens)
        return self.encoder(tokens)

    def training_step(self, batch, batch_idx):
        del batch_idx
        batch_size = len(batch)

        triplet_losses = []
        anchor_embs = []
        family_targets = []
        pos_dists = []
        neg_dists = []

        for a_txt, p_txt, n_txts, neg_is_hard, family_id in batch:
            a = self._encode_text(a_txt)
            p = self._encode_text(p_txt)
            n = torch.stack([self._encode_text(t) for t in n_txts], dim=0)
            hard_mask = torch.tensor(neg_is_hard, device=a.device, dtype=torch.bool)

            d_pos = torch.norm(a - p, dim=-1)
            d_neg = torch.norm(a.unsqueeze(0) - n, dim=-1)

            margins = torch.where(
                hard_mask,
                torch.full_like(d_neg, self.hard_margin),
                torch.full_like(d_neg, self.easy_margin),
            )
            weights = torch.where(
                hard_mask,
                torch.full_like(d_neg, self.hard_weight),
                torch.ones_like(d_neg),
            )

            tri = torch.relu(d_pos - d_neg + margins)
            tri_loss = (tri * weights).mean()

            hardest_neg = d_neg.min()
            hard_mining = torch.relu(d_pos - hardest_neg + self.hard_margin)
            sample_loss = tri_loss + self.hard_mining_weight * hard_mining

            triplet_losses.append(sample_loss)
            anchor_embs.append(a)
            family_targets.append(int(family_id))
            pos_dists.append(d_pos)
            neg_dists.append(d_neg.mean())

        triplet_loss = torch.stack(triplet_losses).mean()
        anchor_embs = torch.stack(anchor_embs, dim=0)
        family_targets = torch.tensor(family_targets, device=anchor_embs.device, dtype=torch.long)

        # anchor-proxy family classification loss
        an = nn.functional.normalize(anchor_embs, dim=-1)
        pn = nn.functional.normalize(self.family_proxies, dim=-1)
        logits = an @ pn.transpose(0, 1)
        logits = logits / max(1e-6, self.proxy_temp)
        proxy_loss = nn.functional.cross_entropy(logits, family_targets)

        loss = triplet_loss + self.proxy_weight * proxy_loss

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("train/triplet", triplet_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("train/proxy", proxy_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("train/pos_dist", torch.stack(pos_dists).mean(), on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train/neg_dist", torch.stack(neg_dists).mean(), on_step=True, on_epoch=True, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        total_steps = int(getattr(self.trainer, "estimated_stepping_batches", 0))
        if total_steps <= 0:
            total_steps = 1000
        warmup_steps = int(total_steps * self.warmup_ratio)
        warmup_steps = min(max(warmup_steps, 1), max(total_steps - 1, 1))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
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


def build_bootstrap_vocabs(num_bootstrap: int, num_cls: int, max_tokens: Optional[int], args) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, int]]:
    texts = []
    cpu = torch.device("cpu")
    for _ in range(num_bootstrap):
        family = random.choice(["gmm", "copula", "scm"])
        outlier = random_outlier_type(family)
        dim = int(np.random.randint(args.min_dim, args.max_dim + 1))
        ex = sample_prior_example(family, outlier, dim, cpu, args)
        texts.append(prior_to_text(ex, device=cpu))

    tokenized = [tokenize_program(t, num_cls=num_cls, max_tokens=max_tokens) for t in texts]
    symbol_vocab = build_vocab(tokenized, "name")
    field_vocab = build_vocab(tokenized, "field")
    family_vocab = build_vocab(tokenized, "family")
    entity_type_vocab = build_vocab(tokenized, "entity_type")
    return symbol_vocab, field_vocab, family_vocab, entity_type_vocab


def parse_args():
    parser = argparse.ArgumentParser(description="Lightning trainer for multi-prior contrastive embedding (GMM + COPULA + SCM).")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps_per_epoch", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--min_lr_ratio", type=float, default=5e-6)

    parser.add_argument("--min_dim", type=int, default=2)
    parser.add_argument("--max_dim", type=int, default=100)
    parser.add_argument("--min_num_cluster", type=int, default=1)
    parser.add_argument("--max_num_cluster", type=int, default=5)
    parser.add_argument("--max_mean", type=int, default=6)
    parser.add_argument("--max_var", type=int, default=6)
    parser.add_argument("--inflate_scale", type=float, default=5.0)
    parser.add_argument("--num_prototypes", type=int, default=16)

    parser.add_argument("--num_negatives", type=int, default=8)
    parser.add_argument("--hard_negative_ratio", type=float, default=0.5)
    parser.add_argument("--pos_strength", type=float, default=0.08)
    parser.add_argument("--hard_strength", type=float, default=0.25)
    parser.add_argument("--use_curriculum", action="store_true")
    parser.add_argument("--curriculum_warmup_epochs", type=int, default=3)
    parser.add_argument("--curriculum_hard_ratio_start", type=float, default=0.1)
    parser.add_argument("--curriculum_hard_ratio_end", type=float, default=-1.0)
    parser.add_argument("--curriculum_hard_strength_start", type=float, default=-1.0)
    parser.add_argument("--curriculum_hard_strength_end", type=float, default=-1.0)

    parser.add_argument("--easy_margin", type=float, default=0.8)
    parser.add_argument("--hard_margin", type=float, default=0.4)
    parser.add_argument("--hard_weight", type=float, default=1.8)
    parser.add_argument("--hard_mining_weight", type=float, default=0.5)
    parser.add_argument("--proxy_weight", type=float, default=0.5)
    parser.add_argument("--proxy_temp", type=float, default=0.07)

    parser.add_argument("--num_cls", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="FoMo-AllPriors-Contrastive")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="./ckpt/allpriors_contrastive")
    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    if args.curriculum_hard_ratio_end < 0:
        args.curriculum_hard_ratio_end = args.hard_negative_ratio
    if args.curriculum_hard_strength_end < 0:
        args.curriculum_hard_strength_end = args.hard_strength
    if args.curriculum_hard_strength_start < 0:
        # Easier early negatives (larger perturbation), then move toward harder negatives.
        args.curriculum_hard_strength_start = max(args.hard_strength * 1.8, args.hard_strength + 0.15)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.wandb_run_name if args.wandb_run_name else (
        f"allpriors_contrastive.cls{args.num_cls}.E{args.epochs}.step{args.steps_per_epoch}."
        f"bs{args.batch_size}.lr{args.lr}.nneg{args.num_negatives}.{current_time}"
    )

    save_path = os.path.join(args.save_dir, run_name, f"seed{args.seed}")
    os.makedirs(save_path, exist_ok=True)

    symbol_vocab, field_vocab, family_vocab, entity_type_vocab = build_bootstrap_vocabs(
        num_bootstrap=96,
        num_cls=args.num_cls,
        max_tokens=args.max_tokens,
        args=args,
    )

    vectorizer = ProgramVectorizer(
        symbol_vocab=symbol_vocab,
        field_vocab=field_vocab,
        family_vocab=family_vocab,
        entity_type_vocab=entity_type_vocab,
        d_model=args.d_model,
        max_dim=max(256, args.max_dim + 64),
        max_entity_id=256,
        max_index_pos=max(256, args.max_dim + 64),
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

    model = LitMultiPriorContrastive(
        encoder=encoder,
        num_cls=args.num_cls,
        d_model=args.d_model,
        max_tokens=args.max_tokens,
        lr=args.lr,
        easy_margin=args.easy_margin,
        hard_margin=args.hard_margin,
        hard_weight=args.hard_weight,
        hard_mining_weight=args.hard_mining_weight,
        proxy_weight=args.proxy_weight,
        proxy_temp=args.proxy_temp,
        warmup_ratio=args.warmup_ratio,
        min_lr_ratio=args.min_lr_ratio,
        use_curriculum=args.use_curriculum,
        curriculum_warmup_epochs=args.curriculum_warmup_epochs,
        curriculum_hard_ratio_start=args.curriculum_hard_ratio_start,
        curriculum_hard_ratio_end=args.curriculum_hard_ratio_end,
        curriculum_hard_strength_start=args.curriculum_hard_strength_start,
        curriculum_hard_strength_end=args.curriculum_hard_strength_end,
    )

    train_ds = MultiPriorContrastiveDataset(
        length=args.steps_per_epoch * args.batch_size,
        args=args,
        generation_device=torch.device("cpu"),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda x: x,
    )

    logger = WandbLogger(project=args.wandb_project, name=run_name) if args.logging else None

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
    trainer.fit(model, train_dataloaders=train_loader)


if __name__ == "__main__":
    main()
