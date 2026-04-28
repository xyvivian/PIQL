# UPDATED SCM prob and SCM contextual!!

import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score
import time
import math
import random
from trainer_embedder.embedder import describe_prior_program
#from embedder import describe_prior_program

def lognormal_discrete(mu,
                       sigma,
                       minval:int,
                       maxval:int):
    # sample from lognormal distribtuion, making it discrete
    # input: mu, sigma, minval (int), maxval(int)
    # return: a integer value
    val = int(np.round(np.random.lognormal(mu, sigma)))
    return int(np.clip(val, minval, maxval))


def sample_layers_and_nodes(min_num_layer =2,
                            max_num_layer =5,
                            min_hidden_size = 3,
                            max_hidden_size = 8):
    #return: a randomly sampled hidden layer and number of layers
    l = lognormal_discrete(mu=0.7, sigma=0.4, minval=min_num_layer, maxval=max_num_layer)  # num layers
    h = lognormal_discrete(mu=1.2, sigma=0.5, minval=min_hidden_size, maxval=max_hidden_size)  # hidden size
    return l, h

@torch.no_grad()
def sample_noise_distribution(device='cpu'):
    mu = (torch.rand(1, device=device) - 0.5).item()
    sigma = (torch.rand(1, device=device) * (0.5 - 0.05) + 0.05).item()
    def noise_func(n):
        return torch.exp(mu + sigma * torch.randn(n, device=device))
    noise_func.mu = mu
    noise_func.sigma = sigma
    return noise_func

@torch.no_grad()
def sample_activation(device='cpu'):
    # sample activation functions (PyTorch version)
    activations = [
        ("tanh", torch.tanh),
        ("leaky_relu", lambda x: torch.where(x > 0, x, 0.01 * x)),
        ("elu", lambda x: torch.where(x > 0, x, torch.exp(x) - 1)),
        ("identity", lambda x: x),
    ]
    idx = torch.randint(0, len(activations), (1,), device=device).item()
    return activations[idx]


@torch.no_grad()
def random_noise_scales_per_sample(num_samples, 
                                   layer_sizes, 
                                   high_noise=5.0, 
                                   high_noise_prob=0.2, 
                                   device='cpu'):
    """
    Generate random noise scales for each sample and node, for each layer.
    Returns: List of tensors, each shape (num_samples, layer_size)
    """
    noise_scales = [
        torch.where(
            torch.rand(num_samples, n, device=device) < high_noise_prob,
            torch.full((num_samples, n), high_noise, device=device),
            torch.ones(num_samples, n, device=device)
        )
        for n in layer_sizes
    ]
    return noise_scales


@torch.no_grad()
def create_weight_mask(
    num_mask_types,
    layers, 
    chosen_nodes, 
    perturb_prob=0.5,
    device='cpu'
):
    """
    Generate only a limited number of mask prototypes, then assign each sample to one prototype.
    This avoids generating a fully independent mask per sample.
    """
    num_prototypes = max(1, int(num_mask_types))
    prototype_masks = []
    node_layer_size = layers[0].weight.shape[0]  # assumes all layers same size
    chosen_nodes = [int(x) for x in chosen_nodes]

    # Map chosen global ids -> per-layer local ids
    chosen_nodes_by_layer = {}
    for g in chosen_nodes:
        l = g // node_layer_size
        j = g % node_layer_size
        chosen_nodes_by_layer.setdefault(l, []).append(j)

    # Sample which chosen nodes are perturbed (subset, not all)
    active_nodes_by_layer = {}
    if len(chosen_nodes) > 0:
        for l in range(len(layers)):
            local_nodes = sorted(set(chosen_nodes_by_layer.get(l, [])))
            if len(local_nodes) == 0:
                continue
            node_active = (torch.rand(num_prototypes, len(local_nodes), device=device) < perturb_prob)
            active_nodes_by_layer[l] = (local_nodes, node_active)

        # Ensure each prototype has at least one active chosen node globally
        if len(active_nodes_by_layer) > 0:
            for p in range(num_prototypes):
                has_any = False
                for _, (_, node_active) in active_nodes_by_layer.items():
                    if bool(node_active[p].any().item()):
                        has_any = True
                        break
                if not has_any:
                    rand_global = chosen_nodes[torch.randint(0, len(chosen_nodes), (1,), device=device).item()]
                    l = rand_global // node_layer_size
                    j = rand_global % node_layer_size
                    if l in active_nodes_by_layer:
                        local_nodes, node_active = active_nodes_by_layer[l]
                        if j in local_nodes:
                            col = local_nodes.index(j)
                            node_active[p, col] = True

    for l, layer in enumerate(layers):
        weight = layer.weight  # (out_features, in_features)
        base_keep_mask = (layer.mask > 0) if hasattr(layer, 'mask') else torch.ones_like(weight, dtype=torch.bool)
        perturbable = (torch.abs(weight) > 0.5) & base_keep_mask  # (out, in)
        mask = torch.ones(num_prototypes, *weight.shape, device=device)

        # Apply perturbations only on active chosen nodes for this layer.
        local_nodes, node_active = active_nodes_by_layer.get(l, ([], None))
        if node_active is None or len(local_nodes) == 0:
            prototype_masks.append(mask)
            continue

        for c, node_idx in enumerate(local_nodes):
            eligible_parents = perturbable[node_idx, :].nonzero(as_tuple=True)[0]
            if len(eligible_parents) == 0:
                continue

            active_proto_ids = node_active[:, c].nonzero(as_tuple=True)[0]
            if len(active_proto_ids) == 0:
                continue

            # For each active prototype, perturb one eligible parent edge.
            rand_parent_idx = torch.randint(0, len(eligible_parents), (len(active_proto_ids),), device=device)
            parent_idx = eligible_parents[rand_parent_idx]
            random_flip = (torch.rand(len(active_proto_ids), device=device) < 0.5)
            mask[active_proto_ids, node_idx, parent_idx] = torch.where(
                random_flip,
                torch.tensor(-1.0, device=device),
                torch.tensor(0.0, device=device),
            )

        prototype_masks.append(mask)
    return num_prototypes, prototype_masks


@torch.no_grad()
def assign_weight_masks_for_samples(num_samples, prototype_masks, num_prototypes, device='cpu'):
    """Assign a prototype mask id for each sample and gather batched weight masks."""
    mask_type_ids = torch.randint(0, int(num_prototypes), (num_samples,), device=device)
    weight_masks = [pm[mask_type_ids] for pm in prototype_masks]
    return weight_masks, mask_type_ids


@torch.no_grad()
def create_noise_scale_prototypes(
    num_mask_types,
    layer_sizes,
    chosen_nodes,
    node_layer_size,
    high_noise=5.0,
    high_noise_prob=0.2,
    device='cpu'
):
    """
    Build a limited bank of noise-scale prototypes (like contextual masks), then
    assign samples to one prototype id.
    """
    num_prototypes = max(1, int(num_mask_types))
    prototype_scales = []
    chosen_nodes = [int(x) for x in chosen_nodes]

    for l, n in enumerate(layer_sizes):
        is_high = (torch.rand(num_prototypes, n, device=device) < high_noise_prob)
        scales = torch.where(
            is_high,
            torch.full((num_prototypes, n), float(high_noise), device=device),
            torch.ones((num_prototypes, n), device=device),
        )
        prototype_scales.append(scales)

    # Ensure each prototype perturbs at least one selected node
    if len(chosen_nodes) > 0:
        for p in range(num_prototypes):
            has_selected_high = False
            for g in chosen_nodes:
                l = g // int(node_layer_size)
                j = g % int(node_layer_size)
                if l < len(prototype_scales) and j < prototype_scales[l].shape[1]:
                    if float(prototype_scales[l][p, j]) == float(high_noise):
                        has_selected_high = True
                        break
            if not has_selected_high:
                g = int(chosen_nodes[torch.randint(0, len(chosen_nodes), (1,), device=device).item()])
                l = g // int(node_layer_size)
                j = g % int(node_layer_size)
                if l < len(prototype_scales) and j < prototype_scales[l].shape[1]:
                    prototype_scales[l][p, j] = float(high_noise)

    # Record selected feature ids (0..num_features-1) receiving high noise per prototype
    selected_sorted = sorted(chosen_nodes)
    feature_of_global = {g: i for i, g in enumerate(selected_sorted)}
    prototype_feature_ids = []
    for p in range(num_prototypes):
        high_noise_feature_ids = []
        for g in selected_sorted:
            l = g // int(node_layer_size)
            j = g % int(node_layer_size)
            if l < len(prototype_scales) and j < prototype_scales[l].shape[1]:
                if float(prototype_scales[l][p, j]) == float(high_noise):
                    high_noise_feature_ids.append(int(feature_of_global[g]))
        prototype_feature_ids.append({
            'prototype_id': int(p),
            'high_noise_nodes': sorted(high_noise_feature_ids),
        })

    return num_prototypes, prototype_scales, prototype_feature_ids


@torch.no_grad()
def assign_noise_scales_for_samples(num_samples, prototype_scales, num_prototypes, device='cpu'):
    """Assign a noise-scale prototype id for each sample and gather batched scales."""
    noise_type_ids = torch.randint(0, int(num_prototypes), (num_samples,), device=device)
    noise_scales = [ps[noise_type_ids] for ps in prototype_scales]
    return noise_scales, noise_type_ids



class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, min_abs=0.35, device='cpu'):
        super().__init__(in_features, out_features, False)
        # Sample weights from N(0, 1)
        with torch.no_grad():
            w = torch.normal(mean=0., std=1., size=self.weight.shape, device=device)
            self.weight.copy_(w) #_clipped)
        self.mask = nn.Parameter(torch.ones_like(self.weight), requires_grad=False)


    def set_random_mask(self, keep_prob=0.7):
        with torch.no_grad():
            self.mask[:] = (torch.rand_like(self.mask) < keep_prob).float()


    def forward(self, input, weight_mask=None):
        masked_weight = self.weight * self.mask
        if weight_mask is None:
            return nn.functional.linear(input, masked_weight, None)
        batch = input.size(0)
        masked_weight = masked_weight.unsqueeze(0)  # (1, out_features, in_features)
        weight = masked_weight.expand(batch, -1, -1) * weight_mask  # (batch, out_features, in_features)
        out = torch.bmm(input.unsqueeze(1), weight.transpose(1,2)).squeeze(1)
        return out




class SCM_MLP(nn.Module):
    def __init__(self, hidden_dim, num_layers, activations,device='cuda'):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(MaskedLinear(hidden_dim, hidden_dim,device=device))
        assert len(activations) == len(self.layers)
        self.activations = activations
        
        # Per-node noise distributions for each layer and neuron
        self.noise_funcs = [
            [sample_noise_distribution(device) for _ in range(hidden_dim)]  # per node
            for _ in range(num_layers)
        ]


    def set_masks(self, keep_prob=0.7):
        for layer in self.layers:
            layer.set_random_mask(keep_prob)
            
            
    def forward(self, 
                x):
        activations = []
        out = x
        batch_size = x.shape[0]
        for idx, layer in enumerate(self.layers):
            out = layer(out)
            # Generate per-node noise for the whole batch
            noises = torch.stack([
                self.noise_funcs[idx][j](batch_size)  # shape (batch,)
                for j in range(out.shape[1])
            ], dim=1)  # shape (batch, nodes)
            out = out + noises
            out = self.activations[idx](out)
            activations.append(out)
        return torch.cat(activations, dim=1)


    def forward_with_weight_masks(self,
                                x,
                                noise_std=0.1, 
                                weight_masks=None):
        """
        x: (batch, in_features)
        noise_scales: list of (batch, layer_size) tensors or None
        weight_masks: list of (batch, layer_size, in_features) tensors or None
        """
        activations = []
        out = x
        batch_size = x.shape[0]
        for idx, layer in enumerate(self.layers):
            mask = weight_masks[idx] if weight_masks is not None else None
            out = layer(out, weight_mask=mask) if mask is not None else layer(out)
            # Generate per-node noise for the whole batch
            noises = torch.stack([
                self.noise_funcs[idx][j](batch_size)  # shape (batch,)
                for j in range(out.shape[1])
            ], dim=1)  # shape (batch, nodes)
            out = out + noises
            out = self.activations[idx](out)
            activations.append(out)
        return torch.cat(activations, dim=1)
        
    
    
    def forward_with_noise_scales(self,
                                x,
                                noise_scales=None,
                                return_noises=False):
        activations = []
        out = x
        batch_size = x.shape[0]
        all_noises = []
        all_scales = []
        for idx, layer in enumerate(self.layers):
            out = layer(out)
            noises = torch.stack([
                self.noise_funcs[idx][j](batch_size)
                for j in range(out.shape[1])
            ], dim=1)  # (batch, nodes)
            scales = noise_scales[idx] if noise_scales is not None else torch.ones_like(noises)
            noises = noises * scales
            out = out + noises
            out = self.activations[idx](out)
            activations.append(out)
            if return_noises:
                all_noises.append(noises)
                all_scales.append(scales)
        if return_noises:
            return torch.cat(activations, dim=1), all_noises, all_scales
        else:
            return torch.cat(activations, dim=1)

    
class StructuralCausalModel:
    def __init__(self,
                num_features: int = 3,
                min_num_layer: int = 3,
                max_num_layer: int = 5,
                min_hidden_size: int = 8,
                max_hidden_size: int = 8,
                device = 'cpu',
                outlier_type = 'contextual',
                drop_weight_prob: float = 0.7,
                contextual_perturb_prob: float = 0.2,
                contextual_num_mask_types: int = 16,
                prob_num_mask_types: int = 16,
                ):
        self.device = device
        self.l, self.h = sample_layers_and_nodes(min_num_layer,max_num_layer,min_hidden_size, max_hidden_size)
        while self.l * self.h < num_features:
            self.l, self.h = sample_layers_and_nodes(min_num_layer,max_num_layer,min_hidden_size, max_hidden_size)
        self.activations = [sample_activation(device)[1] for _ in range(self.l)]
        self.mlp = SCM_MLP(self.h, self.l, activations=self.activations, device=device)
        self.mlp = self.mlp.to(device) 
        self.mlp.set_masks(keep_prob=drop_weight_prob) 
        self.num_features = num_features
        self.chosen_nodes = np.random.choice(self.l * self.h, self.num_features, replace=False)
        self.outlier_type = outlier_type
        self.contextual_perturb_prob = contextual_perturb_prob
        self.contextual_num_mask_types = contextual_num_mask_types
        self.prob_num_mask_types = prob_num_mask_types
        self.weight_mask_abs_threshold = 0.5
        self.last_contextual_mask_summary = None
        self.last_contextual_weight_masks = None
        self.last_prob_outlier_summary = None

        # Contextual mask cache (initialized once for contextual outlier mode)
        self.weight_masks = None
        self.mask_type_ids = None
        self.num_prototypes = 0
        self.prototype_masks = None
        self.prototype_node_ids = None
        self.contextual_mask_bank = None
        self._contextual_bank_perturb_prob = None

        # Prob outlier prototype bank (initialized for prob mode)
        self.prob_noise_scales = None
        self.prob_noise_type_ids = None
        self.prob_num_prototypes = 0
        self.prob_prototype_scales = None
        self.prob_prototype_node_ids = None
        self._prob_bank_signature = None

        if self.outlier_type == 'contextual':
            num_prototypes, prototype_masks = create_weight_mask(
                self.contextual_num_mask_types,
                self.mlp.layers,
                chosen_nodes=self.chosen_nodes,
                perturb_prob=self.contextual_perturb_prob,
                device=self.device,
            )
            self.weight_masks = None
            self.mask_type_ids = None
            self.num_prototypes = int(num_prototypes)
            self.prototype_masks = prototype_masks
            self.prototype_node_ids = self._build_prototype_node_ids()
        elif self.outlier_type == 'prob':
            self._refresh_prob_noise_bank(high_noise=5.0, high_noise_prob=0.2)
  

    @torch.no_grad()
    def _refresh_contextual_mask_bank(self, perturb_prob):
        """Refresh cached prototype masks if perturb_prob changes."""
        num_prototypes, prototype_masks = create_weight_mask(
            self.contextual_num_mask_types,
            self.mlp.layers,
            chosen_nodes=self.chosen_nodes,
            perturb_prob=perturb_prob,
            device=self.device,
        )
        self.weight_masks = None
        self.mask_type_ids = None
        self.num_prototypes = int(num_prototypes)
        self.prototype_masks = prototype_masks
        self.prototype_node_ids = self._build_prototype_node_ids()

    @torch.no_grad()
    def _refresh_prob_noise_bank(self, high_noise, high_noise_prob):
        """Refresh cached prototype noise scales for prob outlier mode."""
        layer_sizes = [layer.out_features for layer in self.mlp.layers]
        num_prototypes, prototype_scales, prototype_feature_ids = create_noise_scale_prototypes(
            num_mask_types=self.prob_num_mask_types,
            layer_sizes=layer_sizes,
            chosen_nodes=self.chosen_nodes,
            node_layer_size=self.h,
            high_noise=high_noise,
            high_noise_prob=high_noise_prob,
            device=self.device,
        )
        self.prob_noise_scales = None
        self.prob_noise_type_ids = None
        self.prob_num_prototypes = int(num_prototypes)
        self.prob_prototype_scales = prototype_scales
        self.prob_prototype_node_ids = prototype_feature_ids
        self._prob_bank_signature = (float(high_noise), float(high_noise_prob))

    @torch.no_grad()
    def _build_prototype_node_ids(self):
        """Return flipped/dropped selected feature ids for each prototype type."""
        if self.prototype_masks is None:
            return []

        selected_sorted = sorted(int(x) for x in self.chosen_nodes)
        feature_of_global = {g: i for i, g in enumerate(selected_sorted)}

        out = []
        for p in range(int(self.num_prototypes)):
            flipped_nodes = set()
            dropped_nodes = set()
            for g in selected_sorted:
                l = g // self.h
                node = g % self.h
                if l >= len(self.prototype_masks):
                    continue
                row = self.prototype_masks[l][p, node, :]
                has_flip = bool((row == -1.0).any().item())
                has_drop = bool((row == 0.0).any().item())

                feat_id = int(feature_of_global[g])
                # Make categories exclusive: dropped takes precedence over flipped
                if has_drop:
                    dropped_nodes.add(feat_id)
                elif has_flip:
                    flipped_nodes.add(feat_id)

            out.append({
                'prototype_id': int(p),
                'flipped_nodes': sorted(flipped_nodes),
                'dropped_nodes': sorted(dropped_nodes),
            })
        return out

    @torch.no_grad()
    def _build_prob_prototype_node_ids(self, high_noise=5.0):
        """Return high-noise selected feature ids for each prob prototype type."""
        if self.prob_prototype_scales is None:
            return []

        selected_sorted = sorted(int(x) for x in self.chosen_nodes)
        feature_of_global = {g: i for i, g in enumerate(selected_sorted)}

        out = []
        for p in range(int(self.prob_num_prototypes)):
            high_noise_nodes = set()
            for g in selected_sorted:
                l = g // self.h
                node = g % self.h
                if l >= len(self.prob_prototype_scales):
                    continue
                if float(self.prob_prototype_scales[l][p, node]) == float(high_noise):
                    high_noise_nodes.add(int(feature_of_global[g]))
            out.append({
                'prototype_id': int(p),
                'high_noise_nodes': sorted(high_noise_nodes),
            })
        return out
 

    @torch.no_grad()
    def _assign_contextual_weight_masks(self, num_samples):
        """Assign per-sample contextual masks from the cached prototype mask bank."""
        if self.prototype_masks is None or self.num_prototypes <= 0:
            raise RuntimeError("Contextual prototype masks are not initialized.")
        weight_masks, mask_type_ids = assign_weight_masks_for_samples(
            num_samples=num_samples,
            prototype_masks=self.prototype_masks,
            num_prototypes=self.num_prototypes,
            device=self.device,
        )
        self.weight_masks = weight_masks
        self.mask_type_ids = mask_type_ids
        return weight_masks, mask_type_ids

    @torch.no_grad()
    def _assign_prob_noise_scales(self, num_samples):
        """Assign per-sample prob noise scales from the cached prototype bank."""
        if self.prob_prototype_scales is None or self.prob_num_prototypes <= 0:
            raise RuntimeError("Prob noise-scale prototypes are not initialized.")
        noise_scales, noise_type_ids = assign_noise_scales_for_samples(
            num_samples=num_samples,
            prototype_scales=self.prob_prototype_scales,
            num_prototypes=self.prob_num_prototypes,
            device=self.device,
        )
        self.prob_noise_scales = noise_scales
        self.prob_noise_type_ids = noise_type_ids
        return noise_scales, noise_type_ids

    @torch.no_grad()
    def print_contextual_prototype_node_ids(self):
        """Print flipped/dropped node ids for each contextual prototype type."""
        if self.prototype_node_ids is None:
            self.prototype_node_ids = self._build_prototype_node_ids()
        for item in self.prototype_node_ids:
            print(
                f"[PROTOTYPE:{item['prototype_id']}] "
                f"[FLIPPED_NODES:{item['flipped_nodes']}] "
                f"[DROPPED_NODES:{item['dropped_nodes']}]"
            )
    

    @torch.no_grad()
    def sample_inliers(self, num_samples):
        # Generate random input (assume standard normal)
        x = torch.ones((num_samples, self.h), device=self.device)
        acts = self.mlp(x)  # shape: (num_samples, total_nodes)
        # Return only the selected nodes for each sample
        return acts[:, self.chosen_nodes]

    @torch.no_grad()
    def sample_prob_outliers(self, 
                            num_samples, 
                            high_noise=5.0, 
                            high_noise_prob=0.2, 
                            batch_factor=2):
        if (self.prob_prototype_scales is None) or (self._prob_bank_signature != (float(high_noise), float(high_noise_prob))):
            self._refresh_prob_noise_bank(high_noise=high_noise, high_noise_prob=high_noise_prob)

        self.last_prob_outlier_summary = {
            'high_noise': float(high_noise),
            'high_noise_prob': float(high_noise_prob),
            'batch_factor': float(batch_factor),
            'num_prototypes': int(self.prob_num_prototypes),
        }
        collected = []
        while len(collected) < num_samples:
            batch_size = max(int((num_samples - len(collected)) * batch_factor), 10000)
            x = torch.ones((batch_size, self.h), device=self.device)
            noise_scales, _ = self._assign_prob_noise_scales(batch_size)
            activations, all_noises, all_noise_scales = self.mlp.forward_with_noise_scales(
                x, noise_scales=noise_scales, return_noises=True
            )
            batch_mask = torch.ones(batch_size, dtype=torch.bool, device=x.device)
            for idx, (noises, scales) in enumerate(zip(all_noises, all_noise_scales)):
                high_noise_mask = (scales == high_noise)
                if high_noise_mask.any():
                    # For each node in this layer, get its mean and std
                    means = torch.tensor(
                        [float(getattr(self.mlp.noise_funcs[idx][j], 'mu', 0.0)) for j in range(noises.shape[1])],
                        device=x.device
                    )
                    stds = torch.tensor(
                        [float(getattr(self.mlp.noise_funcs[idx][j], 'sigma', 1.0)) for j in range(noises.shape[1])],
                        device=x.device
                    )
                    thresholds = means +  stds  # shape: (nodes,)
                    # Broadcast to batch shape
                    thresholds = thresholds.unsqueeze(0).expand_as(noises)
                    means = means.unsqueeze(0).expand_as(noises)
                    # Check (for high noise) if |noise - mean| >= threshold
                    valid = ((noises - means).abs() >= thresholds) | (~high_noise_mask)
                    valid = valid.all(dim=1)
                    batch_mask &= valid
            valid_idx = batch_mask.nonzero(as_tuple=True)[0]
            if len(valid_idx) > 0:
                acts_valid = activations[valid_idx][:, self.chosen_nodes]
                collected.append(acts_valid)
            total = sum(x.shape[0] for x in collected)
            if total >= num_samples:
                collected = torch.cat(collected)[:num_samples]
                break
        return collected
    
    
    @torch.no_grad()
    def sample_contextual_outliers(self, num_samples, perturb_prob=None):
        if perturb_prob is None:
            perturb_prob = self.contextual_perturb_prob
        if (self.prototype_masks is None) or (self._contextual_bank_perturb_prob != float(perturb_prob)):
            self._refresh_contextual_mask_bank(perturb_prob)
        x = torch.ones((num_samples, self.h), device=self.device)
        weight_masks, mask_type_ids = self._assign_contextual_weight_masks(num_samples)
        acts = self.mlp.forward_with_weight_masks(x, weight_masks=weight_masks)
        return acts[:, self.chosen_nodes]


    @torch.no_grad()
    def draw_batched_data(self, 
                          num_inliers, 
                          num_local_anomalies):
        raw_inliers = self.sample_inliers(num_inliers)

        if self.outlier_type == 'prob':
            raw_local_anomalies = self.sample_prob_outliers(num_samples=num_local_anomalies)
        elif self.outlier_type == 'contextual':
            raw_local_anomalies = self.sample_contextual_outliers(num_samples=num_local_anomalies)
        return raw_inliers, raw_local_anomalies
    
    
    
    
def make_probSCM(max_feature_dim: int,
                 min_num_layer: int,
                 max_num_layer: int,
                 min_hidden_size: int,
                 max_hidden_size: int,
                 alpha: float,
                 beta: float,
                 device):
    return StructuralCausalModel(num_features = max_feature_dim,
                                 min_num_layer=min_num_layer,
                                 max_num_layer = max_num_layer,
                                 min_hidden_size = min_hidden_size,
                                 max_hidden_size = max_hidden_size,
                                 device = device,
                                 outlier_type = 'prob',
                                 drop_weight_prob = 0.6)


def make_contextualSCM(max_feature_dim: int,
                 min_num_layer: int,
                 max_num_layer: int,
                 min_hidden_size: int,
                 max_hidden_size: int,
                 alpha: float,
                 beta: float,
                 device):
    return StructuralCausalModel(num_features = max_feature_dim,
                                 min_num_layer=min_num_layer,
                                 max_num_layer = max_num_layer,
                                 min_hidden_size = min_hidden_size,
                                 max_hidden_size = max_hidden_size,
                                 device = device,
                                 outlier_type = 'contextual',
                                 drop_weight_prob = 0.6)


def _activation_from_name(name: str):
    name = str(name).strip().lower()
    if name == "tanh":
        return torch.tanh
    if name == "leaky_relu":
        return lambda x: torch.where(x > 0, x, 0.01 * x)
    if name == "elu":
        return lambda x: torch.where(x > 0, x, torch.exp(x) - 1)
    if name == "identity":
        return lambda x: x
    raise ValueError(f"Unsupported activation name: {name}")


@torch.no_grad()
def make_scm_predefined(
    *,
    device,
    num_layers,
    hidden_size,
    chosen_nodes,
    activation_names,
    layer_weights,
    layer_masks,
    outlier_type='contextual',
    contextual_perturb_prob=0.2,
    contextual_num_mask_types=16,
    prototype_masks=None,
    prototype_node_ids=None,
    prob_num_mask_types=16,
    high_noise=5.0,
    high_noise_prob=0.2,
    batch_factor=2.0,
    prob_prototype_scales=None,
    prob_prototype_node_ids=None,
):
    """Build SCM from predefined structural/outlier parameters (similar to predefined loaders in other tests)."""
    num_layers = int(num_layers)
    hidden_size = int(hidden_size)
    chosen_nodes = [int(x) for x in chosen_nodes]
    num_features = len(chosen_nodes)

    if len(activation_names) != num_layers:
        raise ValueError(f"activation_names length ({len(activation_names)}) must equal num_layers ({num_layers})")
    if len(layer_weights) != num_layers or len(layer_masks) != num_layers:
        raise ValueError("layer_weights/layer_masks length must equal num_layers")

    scm = StructuralCausalModel(
        num_features=num_features,
        min_num_layer=num_layers,
        max_num_layer=num_layers,
        min_hidden_size=hidden_size,
        max_hidden_size=hidden_size,
        device=device,
        outlier_type=outlier_type,
        drop_weight_prob=1.0,
        contextual_perturb_prob=float(contextual_perturb_prob),
        contextual_num_mask_types=int(contextual_num_mask_types),
        prob_num_mask_types=int(prob_num_mask_types),
    )

    # Overwrite core structural parameters
    scm.chosen_nodes = np.array(chosen_nodes, dtype=np.int64)
    scm.num_features = int(num_features)
    scm.activations = [_activation_from_name(n) for n in activation_names]
    scm.mlp.activations = scm.activations

    for l in range(num_layers):
        w = torch.as_tensor(layer_weights[l], device=device, dtype=torch.float32)
        m = torch.as_tensor(layer_masks[l], device=device, dtype=torch.float32)
        if w.shape != (hidden_size, hidden_size):
            raise ValueError(f"layer_weights[{l}] must have shape ({hidden_size}, {hidden_size}), got {tuple(w.shape)}")
        if m.shape != (hidden_size, hidden_size):
            raise ValueError(f"layer_masks[{l}] must have shape ({hidden_size}, {hidden_size}), got {tuple(m.shape)}")
        scm.mlp.layers[l].weight.copy_(w)
        scm.mlp.layers[l].mask.copy_(m)

    # Overwrite outlier prototypes/state so description + sampling follow predefined params
    if outlier_type == 'contextual':
        scm.contextual_perturb_prob = float(contextual_perturb_prob)
        if prototype_masks is not None:
            pm = [torch.as_tensor(x, device=device, dtype=torch.float32) for x in prototype_masks]
            if len(pm) != num_layers:
                raise ValueError("prototype_masks must have one tensor per layer")
            p = int(pm[0].shape[0])
            for l in range(num_layers):
                if pm[l].shape != (p, hidden_size, hidden_size):
                    raise ValueError(
                        f"prototype_masks[{l}] must have shape ({p}, {hidden_size}, {hidden_size}), got {tuple(pm[l].shape)}"
                    )
            scm.num_prototypes = p
            scm.prototype_masks = pm
            scm.prototype_node_ids = prototype_node_ids if prototype_node_ids is not None else scm._build_prototype_node_ids()
            scm._contextual_bank_perturb_prob = float(contextual_perturb_prob)
    elif outlier_type == 'prob':
        if prob_prototype_scales is not None:
            ps = [torch.as_tensor(x, device=device, dtype=torch.float32) for x in prob_prototype_scales]
            if len(ps) != num_layers:
                raise ValueError("prob_prototype_scales must have one tensor per layer")
            p = int(ps[0].shape[0])
            for l in range(num_layers):
                if ps[l].shape != (p, hidden_size):
                    raise ValueError(
                        f"prob_prototype_scales[{l}] must have shape ({p}, {hidden_size}), got {tuple(ps[l].shape)}"
                    )
            scm.prob_num_prototypes = p
            scm.prob_prototype_scales = ps
            if prob_prototype_node_ids is None:
                scm.prob_prototype_node_ids = scm._build_prob_prototype_node_ids(high_noise=float(high_noise))
            else:
                scm.prob_prototype_node_ids = prob_prototype_node_ids
            scm._prob_bank_signature = (float(high_noise), float(high_noise_prob))
            scm.last_prob_outlier_summary = {
                'high_noise': float(high_noise),
                'high_noise_prob': float(high_noise_prob),
                'batch_factor': float(batch_factor),
                'num_prototypes': int(p),
            }

    return scm


def _detect_activation_name(fn):
    """Best-effort name for sampled activation function."""
    x = torch.tensor([-1.0, 0.0, 1.0])
    y = fn(x)
    if torch.allclose(y, x, atol=1e-6):
        return "identity"
    if torch.allclose(y, torch.tanh(x), atol=1e-6):
        return "tanh"
    # Distinguish leaky_relu vs elu using x=-1
    y_neg1 = float(fn(torch.tensor([-1.0]))[0])
    if abs(y_neg1 + 0.01) < 1e-4:
        return "leaky_relu"
    if abs(y_neg1 - (math.e**(-1) - 1)) < 1e-3:
        return "elu"
    return "unknown"


def describe_scm_model(scm, coeff_thresh=1e-8, decimals=3, max_parents_len=24, max_coeffs_len=24):
    """Serialize SCM in the same schema family as GMM/COPULA via `describe_prior_program`."""
    h = int(scm.h)
    selected_sorted = sorted(int(g) for g in scm.chosen_nodes)
    var_of_global = {g: i for i, g in enumerate(selected_sorted)}

    entities = []
    for g in selected_sorted:
        var_id = int(var_of_global[g])
        layer = g // h
        node = g % h
        eq_name = _detect_activation_name(scm.activations[layer])

        parent_vars = []
        parent_coeffs = []
        if layer > 0:
            W = scm.mlp.layers[layer].weight.detach()
            M = scm.mlp.layers[layer].mask.detach() if hasattr(scm.mlp.layers[layer], 'mask') else torch.ones_like(W)
            eff = (W * M)[node]  # (h,) from previous layer nodes
            for j in range(h):
                if abs(float(eff[j])) <= coeff_thresh:
                    continue
                parent_global = (layer - 1) * h + j
                if parent_global in var_of_global:
                    parent_vars.append(int(var_of_global[parent_global]))
                    parent_coeffs.append(float(eff[j]))

        entities.append(
            {
                "tag": "ENTITY",
                "fields": {
                    "TYPE": "VAR",
                    "ID": var_id,
                    "LAYER": int(layer),
                    "EQ": eq_name,
                    "PARENTS": torch.tensor(parent_vars, dtype=torch.float32) if len(parent_vars) > 0 else "none",
                    "COEFFS": torch.tensor(parent_coeffs, dtype=torch.float32) if len(parent_coeffs) > 0 else "none",
                },
            }
        )

    blocks = []
    outlier_type = getattr(scm, 'outlier_type', 'unknown')
    if outlier_type == 'contextual':
        pp = float(getattr(scm, 'contextual_perturb_prob', 0.2))
        th = float(getattr(scm, 'weight_mask_abs_threshold', 0.5))
        proto_nodes = getattr(scm, 'prototype_node_ids', None)
        if proto_nodes is None and hasattr(scm, '_build_prototype_node_ids'):
            proto_nodes = scm._build_prototype_node_ids()
        if proto_nodes is None:
            proto_nodes = []

        blocks.append(
            {
                "tag": "OUTLIER",
                "fields": {
                    "TYPE": "weight_mask",
                    "PERTURB_PROB": pp,
                    "WEIGHT_ABS_THRESH": th,
                    "N_PROTO": int(len(proto_nodes)),
                },
            }
        )
        for item in proto_nodes:
            blocks.append(
                {
                    "tag": "ENTITY",
                    "fields": {
                        "TYPE": "OUTLIER_PROTOTYPE",
                        "ID": int(item['prototype_id']),
                        "FLIPPED_NODES": torch.tensor(item['flipped_nodes'], dtype=torch.float32),
                        "DROPPED_NODES": torch.tensor(item['dropped_nodes'], dtype=torch.float32),
                    },
                }
            )
    elif outlier_type == 'prob':
        summary = getattr(scm, 'last_prob_outlier_summary', None)
        if summary is None:
            high_noise = 5.0
            high_noise_prob = 0.2
            batch_factor = 2.0
            n_proto = int(getattr(scm, 'prob_num_prototypes', 0))
        else:
            high_noise = float(summary.get('high_noise', 5.0))
            high_noise_prob = float(summary.get('high_noise_prob', 0.2))
            batch_factor = float(summary.get('batch_factor', 2.0))
            n_proto = int(summary.get('num_prototypes', 0))

        proto_nodes = getattr(scm, 'prob_prototype_node_ids', None)
        if proto_nodes is None:
            proto_nodes = []

        blocks.append(
            {
                "tag": "OUTLIER",
                "fields": {
                    "TYPE": "noise_scale",
                    "HIGH_NOISE": high_noise,
                    "HIGH_NOISE_PROB": high_noise_prob,
                    "BATCH_FACTOR": batch_factor,
                    "N_PROTO": n_proto,
                },
            }
        )
        for item in proto_nodes:
            blocks.append(
                {
                    "tag": "ENTITY",
                    "fields": {
                        "TYPE": "OUTLIER_PROTOTYPE",
                        "ID": int(item['prototype_id']),
                        "HIGH_NOISE_NODES": torch.tensor(item['high_noise_nodes'], dtype=torch.float32),
                    },
                }
            )
    else:
        blocks.append({"tag": "OUTLIER", "fields": {"TYPE": str(outlier_type), "PARAMS": "unknown"}})

    return describe_prior_program(
        family="SCM",
        global_fields={
            "DIM": int(len(selected_sorted)),
        },
        entities=entities,
        blocks=blocks,
        decimals=decimals,
        max_len_map={
            "PARENTS": max_parents_len,
            "COEFFS": max_coeffs_len,
            "FLIPPED_NODES": 24,
            "DROPPED_NODES": 24,
            "HIGH_NOISE_NODES": 24,
        },
    )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_layers = 3
    hidden_size = 6
    num_features = 10

    # predefined structural parameters (key params visible in describe_scm_model)
    chosen_nodes = [0, 1, 2, 4, 5, 7, 9, 12, 14, 16]
    activation_names = ["tanh", "leaky_relu", "identity"]
    layer_weights = [
        torch.randn(hidden_size, hidden_size, device=device),
        torch.randn(hidden_size, hidden_size, device=device),
        torch.randn(hidden_size, hidden_size, device=device),
    ]
    layer_masks = [
        (torch.rand(hidden_size, hidden_size, device=device) > 0.35).to(torch.float32),
        (torch.rand(hidden_size, hidden_size, device=device) > 0.35).to(torch.float32),
        (torch.rand(hidden_size, hidden_size, device=device) > 0.35).to(torch.float32),
    ]

    # Build predefined contextual prototypes
    _dummy_layers = []
    for w, m in zip(layer_weights, layer_masks):
        layer_obj = type("_LayerObj", (), {})()
        layer_obj.weight = w
        layer_obj.mask = m
        _dummy_layers.append(layer_obj)

    num_proto_ctx, prototype_masks_ctx = create_weight_mask(
        num_mask_types=16,
        layers=_dummy_layers,
        chosen_nodes=chosen_nodes,
        perturb_prob=0.2,
        device=device,
    )

    scm_contextual = make_scm_predefined(
        device=device,
        num_layers=num_layers,
        hidden_size=hidden_size,
        chosen_nodes=chosen_nodes,
        activation_names=activation_names,
        layer_weights=layer_weights,
        layer_masks=layer_masks,
        outlier_type='contextual',
        contextual_perturb_prob=0.2,
        contextual_num_mask_types=int(num_proto_ctx),
        prototype_masks=prototype_masks_ctx,
    )

    print("\n===== PREDEFINED SCM CONTEXTUAL =====")
    description_ctx = describe_scm_model(scm_contextual)
    print(description_ctx)
    inliers_ctx, outliers_ctx = scm_contextual.draw_batched_data(1000, 1000)
    print(outliers_ctx[:5, :5])

    # Build predefined prob prototypes
    high_noise = 5.0
    high_noise_prob = 0.2
    batch_factor = 2.0
    num_proto_prob, prototype_scales_prob, prototype_nodes_prob = create_noise_scale_prototypes(
        num_mask_types=16,
        layer_sizes=[hidden_size] * num_layers,
        chosen_nodes=chosen_nodes,
        node_layer_size=hidden_size,
        high_noise=high_noise,
        high_noise_prob=high_noise_prob,
        device=device,
    )

    scm_prob = make_scm_predefined(
        device=device,
        num_layers=num_layers,
        hidden_size=hidden_size,
        chosen_nodes=chosen_nodes,
        activation_names=activation_names,
        layer_weights=layer_weights,
        layer_masks=layer_masks,
        outlier_type='prob',
        prob_num_mask_types=int(num_proto_prob),
        high_noise=high_noise,
        high_noise_prob=high_noise_prob,
        batch_factor=batch_factor,
        prob_prototype_scales=prototype_scales_prob,
        prob_prototype_node_ids=prototype_nodes_prob,
    )

    print("\n===== PREDEFINED SCM PROB =====")
    description_prob = describe_scm_model(scm_prob)
    print(description_prob)
    inliers_prob, outliers_prob = scm_prob.draw_batched_data(1000, 1000)