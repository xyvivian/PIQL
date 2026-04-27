import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score
import time
import math
import random


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
def random_noise_scales_per_sample(num_samples, layer_sizes, high_noise=5.0, high_noise_prob=0.2, device='cpu'):
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
    num_samples, layers, chosen_nodes, perturb_prob=0.5, device='cpu'
):
    """
    Vectorized version: No per-sample loop.
    For each sample and layer, ensures that at least one parent of a chosen node is perturbed.
    """
    masks = []
    node_layer_size = layers[0].weight.shape[0]  # assumes all layers same size

    for l, layer in enumerate(layers):
        weight = layer.weight  # (out_features, in_features)
        perturbable = (torch.abs(weight) > 0.5)  # (out, in)
        mask = torch.ones(num_samples, *weight.shape, device=device)

        perturb_mask = (torch.rand(num_samples, *weight.shape, device=device) < perturb_prob) & perturbable.unsqueeze(0)
        flip_mask = torch.rand(num_samples, *weight.shape, device=device) < 0.5

        mask[perturb_mask & flip_mask] = -1.0
        mask[perturb_mask & (~flip_mask)] = 0.0

        # Find chosen nodes for this layer (global to local index)
        chosen_nodes_this_layer = [idx for idx in chosen_nodes if (idx // node_layer_size) == l]
        if len(chosen_nodes_this_layer) == 0:
            masks.append(mask)
            continue

        for cidx in chosen_nodes_this_layer:
            node_idx = cidx % node_layer_size  # local node index for this layer

            # For all samples: find if any parent is perturbed (and perturbable) for this node
            perturbed = ((mask[:, node_idx, :] != 1.0) & perturbable[node_idx, :])  # (num_samples, in_features)
            any_perturbed = perturbed.any(dim=1)  # (num_samples,)

            need_perturb = (~any_perturbed)  # (num_samples,)
            num_to_fix = need_perturb.sum().item()
            if num_to_fix == 0:
                continue

            # For those samples, pick a random eligible parent and force a perturbation
            eligible_parents = perturbable[node_idx, :].nonzero(as_tuple=True)[0]
            if len(eligible_parents) == 0:
                continue

            # Pick a random parent for each sample needing fix
            rand_idx = torch.randint(0, len(eligible_parents), (num_to_fix,), device=device)
            parent_idx = eligible_parents[rand_idx]  # (num_to_fix,)
            sample_idx = need_perturb.nonzero(as_tuple=True)[0]  # (num_to_fix,)

            # Randomly decide flip or zero
            random_flip = (torch.rand(num_to_fix, device=device) < 0.5)
            mask[sample_idx, node_idx, parent_idx] = torch.where(random_flip, -1.0, 0.0)

        masks.append(mask)
    return masks



class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, min_abs=0.35, device='cpu'):
        super().__init__(in_features, out_features, False)
        # Sample weights from N(0, 1)
        with torch.no_grad():
            w = torch.normal(mean=0., std=1., size=self.weight.shape, device=device)
            # abs_w = torch.clamp(torch.abs(w), min=torch.tensor(min_abs,device=device))
            # w_clipped = torch.sign(w) * abs_w
            self.weight.copy_(w) #_clipped)
        self.mask = nn.Parameter(torch.ones_like(self.weight), requires_grad=False)


    def set_random_mask(self, keep_prob=0.7):
        with torch.no_grad():
            self.mask[:] = (torch.rand_like(self.mask) < keep_prob).float()

    def forward(self, input, weight_mask=None):
        # input: (batch, in_features)
        # self.weight: (out_features, in_features)
        # weight_mask: (batch, out_features, in_features) or None
        masked_weight = self.weight * self.mask
        if weight_mask is None:
            return nn.functional.linear(input, masked_weight, None)
        # Use per-sample masked weights (batched matmul)
        # weight_mask shape: (batch, out_features, in_features)
        # input shape: (batch, in_features)
        # Expand masked_weight for batch: (1, out_features, in_features)
        batch = input.size(0)
        masked_weight = masked_weight.unsqueeze(0)  # (1, out_features, in_features)
        # Broadcast for batch
        weight = masked_weight.expand(batch, -1, -1) * weight_mask  # (batch, out_features, in_features)
        # Batched matmul: input (batch, in_features) × weight.transpose(-2, -1) (batch, in_features, out_features)
        # Result: (batch, out_features)
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
                ):
        self.device = device
        self.l, self.h = sample_layers_and_nodes(min_num_layer,max_num_layer,min_hidden_size, max_hidden_size)
        while self.l * self.h < num_features:
            #print('here???',self.l, self.h, num_features,min_num_layer,max_num_layer,min_hidden_size, max_hidden_size)
            self.l, self.h = sample_layers_and_nodes(min_num_layer,max_num_layer,min_hidden_size, max_hidden_size)
        self.activations = [sample_activation(device)[1] for _ in range(self.l)]
        self.mlp = SCM_MLP(self.h, self.l, activations=self.activations, device=device)
        self.mlp = self.mlp.to(device) 
        self.mlp.set_masks(keep_prob=drop_weight_prob) 
        self.num_features = num_features
        self.chosen_nodes = np.random.choice(self.l * self.h, self.num_features, replace=False)
        self.outlier_type = outlier_type
        self.prior_description = describe_scm_model(self)


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
        collected = []
        while len(collected) < num_samples:
            batch_size = max(int((num_samples - len(collected)) * batch_factor), 10000)
            x = torch.ones((batch_size, self.h), device=self.device)
            layer_sizes = [layer.out_features for layer in self.mlp.layers]
            noise_scales = random_noise_scales_per_sample(
                batch_size, layer_sizes, 
                high_noise=high_noise, 
                high_noise_prob=high_noise_prob, 
                device=self.device
            )
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

    # @torch.no_grad()
    # def sample_contextual_outliers(self, num_samples):
    #     x = torch.ones((num_samples, self.h), device=self.device)
    #     weight_masks = create_weight_mask(
    #         num_samples, self.l, self.h, self.h, self.chosen_nodes, device=self.device,perturb_prob=0.2
    #     )
    #     print(weight_masks)
    #     print(self.chosen_nodes)
    #     acts = self.mlp.forward_with_weight_masks(x, weight_masks=weight_masks)
    #     return acts[:, self.chosen_nodes]
    
    
    @torch.no_grad()
    def sample_contextual_outliers(self, num_samples, perturb_prob=0.2):
        x = torch.ones((num_samples, self.h), device=self.device)
        #start = time.time()
        weight_masks = create_weight_mask(
            num_samples, self.mlp.layers, chosen_nodes = self.chosen_nodes, perturb_prob=perturb_prob, device=self.device
        )
        #print('draw weights', time.time()-start)
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


def describe_scm_model(scm, coeff_thresh=1e-8, decimals=3):
    """Create a compact SCM descriptor string with structure + outlier settings."""
    h = int(scm.h)
    selected = [int(g) for g in scm.chosen_nodes]
    selected_sorted = sorted(selected)
    var_of_global = {g: i for i, g in enumerate(selected_sorted)}

    lines = [f"[FAMILY:SCM] [DIM:{len(selected_sorted)}]"]

    # Outlier description block
    outlier_type = getattr(scm, 'outlier_type', 'unknown')
    lines.append(f"[OUTLIER_TYPE:{outlier_type}]")
    if outlier_type == 'contextual':
        lines.append("[OUTLIER_PARAMS:perturb_prob=0.2(weight-mask perturbation)]")
    elif outlier_type == 'prob':
        lines.append("[OUTLIER_PARAMS:high_noise=5.0,high_noise_prob=0.2(noise-scale inflation)]")
    else:
        lines.append("[OUTLIER_PARAMS:unknown]")

    for g in selected_sorted:
        var_id = var_of_global[g]
        layer = g // h
        node = g % h
        eq_name = _detect_activation_name(scm.activations[layer])

        parent_vars = []
        parent_coeffs = []

        # Layer 0 depends on input x, not prior latent nodes in acts
        if layer > 0:
            W = scm.mlp.layers[layer].weight.detach()
            M = scm.mlp.layers[layer].mask.detach() if hasattr(scm.mlp.layers[layer], 'mask') else torch.ones_like(W)
            eff = (W * M)[node]  # (h,) from previous layer nodes
            for j in range(h):
                if abs(float(eff[j])) <= coeff_thresh:
                    continue
                parent_global = (layer - 1) * h + j
                if parent_global in var_of_global:
                    parent_vars.append(var_of_global[parent_global])
                    parent_coeffs.append(float(eff[j]))

        if parent_vars:
            p_txt = ",".join(map(str, parent_vars))
            c_txt = ",".join(f"{c:.{decimals}f}" for c in parent_coeffs)
        else:
            p_txt = "none"
            c_txt = "none"

        lines.append(
            f"[VAR:{var_id}] [EQ:{eq_name}] [PARENTS:{p_txt}] [COEFFS:{c_txt}]"
        )
    return "\n".join(lines)
