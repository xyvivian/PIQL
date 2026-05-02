import math
import math
import os
import pickle
import random
import sys
import time
from copy import deepcopy

# Add the FoMo-Meta root directory to sys.path.
_fomo_meta_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _fomo_meta_root not in sys.path:
    sys.path.insert(0, _fomo_meta_root)

import hydra
import numpy as np
from omegaconf import DictConfig
import torch
from tqdm import tqdm

from trainer_embedder.gmm_test import make_NdMclusterGMM
from trainer_embedder.scm_test import make_contextualSCM, make_probSCM, describe_scm_model
from trainer_embedder.copula_test import make_disturb_corpula, make_perturb_corpula, describe_model_specs as describe_copula_model
from data_prior.feature_transform import FeatureTransform
from dataset_loader.batch import Batch
import joblib

def load_pickle(file_path):
    with open(file_path, 'rb') as handle:
        inst = pickle.load(handle)
    return inst

from torch import Tensor

#===============Logics for Embedder ================================#
#TODO: change the base code if embedder is changed
def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]



class PriorTrainDataGenerator:  # generate synthetic data for training
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.train_cfg = cfg.train
        
        self.test_cfg = cfg.test
        
        self.seq_len = self.train_cfg.seq_len
        self.hyperparameters = self.train_cfg.hyperparameters
        self.device = self.train_cfg.device
        self.batch_size = self.train_cfg.batch_size
        self.epochs = self.train_cfg.epochs
        self.steps_per_epoch = self.train_cfg.steps_per_epoch
        self.reuse_data_every_n = self.train_cfg.reuse_data_every_n
        self.gen_one_train_one = self.train_cfg.gen_one_train_one
        self.apply_linear_transform = self.train_cfg.apply_linear_transform
        
        self.prior_probscm_cfg = cfg.prior.mixture.scm_prob
        self.prior_contextual_cfg = cfg.prior.mixture.scm_contextual
        self.prior_density = cfg.prior.mixture.density
        self.prior_gmm_cfg = cfg.prior.mixture.gmm
        self.prior_corpula_cfg = cfg.prior.mixture.corpula
        self.max_feature_dim = self.prior_gmm_cfg.max_feature_dim
        self.gen1tr1_epoch_id = 0
        
        # feature transform:
        self.FT = FeatureTransform(cfg=cfg)
        self.num_workers = None
        self.update_model_parameters()

    def set_num_workers(self, num_workers):
        self.num_workers = num_workers

    
    def generate_from_mixture(self, model,return_params=True): 
        num_inliers = self.seq_len
        num_LA = self.seq_len 
        inliers, LA =  model.draw_batched_data(num_inliers, num_LA) #return_params=return_params)
        return inliers, LA
    
    
    def update_model_parameters(self):
        model_choices = []
        # Copula Model
        corpula_params = dict(generate_fn=self.generate_from_mixture,
                              max_feature_dim =self.prior_corpula_cfg.max_feature_dim)
        model_choices.append(("disturbcorpula", make_disturb_corpula, corpula_params))
        model_choices.append(("perturbcorpula", make_perturb_corpula, corpula_params))
        # GMM Model
        gmm_params = dict(
                max_num_cluster=self.prior_gmm_cfg.max_num_cluster,
                max_model_dim=self.prior_gmm_cfg.max_model_dim, 
                diversity=self.prior_gmm_cfg.diversity,
                max_mean=self.prior_gmm_cfg.max_mean,
                max_var=self.prior_gmm_cfg.max_var, 
                inflate_full=self.prior_gmm_cfg.inflate_full,
                percentile=self.prior_gmm_cfg.percentile,
                generate_fn=self.generate_from_mixture)
        model_choices.append(("gmm", make_NdMclusterGMM, gmm_params))
        self.model_choices = model_choices
        #SCM model
        prob_params = dict(
                max_feature_dim=self.prior_probscm_cfg.max_feature_dim,
                min_num_layer=self.prior_probscm_cfg.min_num_layer,
                max_num_layer=self.prior_probscm_cfg.max_num_layer,
                min_hidden_size=self.prior_probscm_cfg.min_hidden_size,
                max_hidden_size=self.prior_probscm_cfg.max_hidden_size,
                alpha=self.prior_probscm_cfg.alpha,
                beta=self.prior_probscm_cfg.beta,
                generate_fn=self.generate_from_mixture
        )
        model_choices.append(("prob", make_probSCM, prob_params))
        self.model_choices = model_choices 

        contextual_params = dict(
                max_feature_dim=self.prior_probscm_cfg.max_feature_dim,
                min_num_layer=self.prior_probscm_cfg.min_num_layer,
                max_num_layer=self.prior_probscm_cfg.max_num_layer,
                min_hidden_size=self.prior_probscm_cfg.min_hidden_size,
                max_hidden_size=self.prior_probscm_cfg.max_hidden_size,
                alpha=self.prior_probscm_cfg.alpha,
                beta=self.prior_probscm_cfg.beta,
                generate_fn=self.generate_from_mixture
        )
        model_choices.append(("contextual", make_contextualSCM, contextual_params))
        self.model_choices = model_choices             

    @staticmethod
    def process_one_dataset(epoch_id,
                            step, 
                            device,
                            every_n_dim,
                            model_choices,
                            model_weights = None,
                            return_params=True): 
        #torch.cuda.empty_cache()
        if model_weights is None:
            model_index, model_entry = random.choice(list(enumerate(model_choices)))
        else:
            model_entry = next((name, constructor, params) for name, constructor, params in model_choices if name == model_weights)
            
        model_name, model_constructor, params = model_entry
        #print(model_name)
        # Set random seed for reproducibility
        seed = epoch_id + step + os.getpid()
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        params['device'] = device
        

        # You can have model-specific logic below
        if model_name == "gmm":
            #print('generating with gmm')
            max_model_dim = params['max_model_dim']
            max_num_cluster = params['max_num_cluster']
            if every_n_dim is None:
                # if every_n_dim is None, then it is generating to self.steps_per_epoch * self.batch_size
                dim = np.random.randint(low=2, high=max_model_dim + 1)  # draw from [2, max_model_dim]
                num_cluster = np.random.randint(low=2, high=max_num_cluster + 1)  # draw from [2, max_num_cluster]
            else:
                dim = (step // max_num_cluster + 1) * every_n_dim
                if dim == 1:  # for the case every_n_dim=1, dim ranges from [1, max_model_dim],
                    # we reset this case to dim=max_model_dim
                    dim = max_model_dim
            num_cluster = np.random.randint(low=2, high=params["max_num_cluster"] + 1)
            max_mean = np.random.randint(low=2, high=params["max_mean"] + 1)
            max_var = np.random.randint(low=2, high=params["max_var"] + 1)
            model = model_constructor(
                dim=dim,
                num_cluster=num_cluster,
                weights=torch.tensor([1 / num_cluster] * num_cluster, device=device),
                max_mean=max_mean,
                max_var=max_var,
                inflate_full=params["inflate_full"],
                sub_dims=None,
                percentile=params["percentile"],
                delta=0.05,
                device=device
            )
            prior_description = model.describe_gmm_model()         
            inliers, LA = params["generate_fn"](model=model)
            embeddings = prior_description
            
            sub_dims = model.sub_dims
            del model
            torch.cuda.empty_cache()
            return inliers, LA, sub_dims, (model_name, embeddings)
        elif model_name == "contextual" or model_name == 'prob':
            # Sample random contextual model hyperparameters
            feature_dim = np.random.randint(low=2, high=params["max_feature_dim"] + 1)
            min_num_layer =max(int(np.sqrt(feature_dim))-3,2)
            min_hidden_size = max(int(math.floor(feature_dim / params['min_num_layer'])) + 2 ,2)
            max_hidden_size = min(params['min_hidden_size']+ 7, params['max_hidden_size'])
            model = model_constructor(feature_dim,
                                      min_num_layer,
                                      params['max_num_layer'],
                                      min_hidden_size,
                                      max_hidden_size,
                                      params['alpha'],
                                      params['beta'], 
                                      device= device)
            inliers, LA = params["generate_fn"](model=model)
            prior_description = describe_scm_model(model)
            embeddings = prior_description
            
            del model
            return inliers, LA, None, (model_name, embeddings)
        elif model_name == "disturbcorpula":
            feature_dim = np.random.randint(low=2, high=params["max_feature_dim"] + 1)
            dim = feature_dim
            model = model_constructor(device= device,dim= dim)
            inliers, LA = params["generate_fn"](model=model)
            prior_description = describe_copula_model(model)
            embeddings = prior_description
            del model
            return inliers, LA, None, (model_name, embeddings)
        
        elif model_name == "perturbcorpula":
            feature_dim = np.random.randint(low=2, high=params["max_feature_dim"] + 1)
            dim = feature_dim
            model = model_constructor(device= device,
                                      dim= dim)
            inliers, LA = params["generate_fn"](model=model)
            prior_description = describe_copula_model(model)
            embeddings = prior_description
            del model
            return inliers, LA, None, (model_name, embeddings)
        
        else:
            raise ValueError(f"Unknown model name: {model_name}")
          
    
    def generate_batches(self, 
                         epoch, 
                         process_function=None, 
                         every_n_dim=10,
                         model_weights = None,
                         total_tasks = None,):
        # if every_n_dim is None, then it is generating to self.steps_per_epoch * self.batch_size
        if process_function is None:
            process_function = self.process_one_dataset  
        if total_tasks is None:
            if every_n_dim is None:
                total_tasks = int(self.steps_per_epoch * (self.batch_size) / self.cfg.train.num_device)
                #print(f'generating {self.steps_per_epoch * self.batch_size} datasets')
            else:
                total_tasks = (self.max_model_dim // every_n_dim) * self.max_num_cluster
                print(self.max_num_cluster)
                print(self.max_model_dim, every_n_dim) 
                print(f'generating models with dim from 2 to {self.max_model_dim} with an interval of {every_n_dim},'
                    f' each dim has {self.max_num_cluster} model(s) with num-of-clusters '
                    f'from 1 to {self.max_num_cluster}')

        print(f'using GPU to generate fast')
        inliners_list, LA_list, sub_dims_list = [], [], []
        model_name_list = []
        for step in tqdm(range(total_tasks)):
            if model_weights is not None:
                model_name = model_weights[step]
                inliners, LA, sub_dims, model_name = process_function(
                    epoch_id=epoch*total_tasks,
                    step=step,
                    device = self.device,
                    every_n_dim =every_n_dim,
                    model_choices = self.model_choices,
                    model_weights = model_name,
                    )
            else:
                inliners, LA, sub_dims, model_name = process_function(
                    epoch_id=epoch*total_tasks,
                    step=step,
                    device = self.device,
                    every_n_dim =every_n_dim,
                    model_choices = self.model_choices,
                    )
            inliners_list.append(inliners)
            LA_list.append(LA)
            sub_dims_list.append(sub_dims)
            model_name_list.append(model_name)

        return inliners_list, LA_list, sub_dims_list,model_name_list 

    def generate_one_epoch(self, 
                           epoch,
                           every_n_dim, 
                           model_weights=None,
                           total_tasks = None):
        # Generate all batches for the epoch at once using parallelization
        inliners, LA, sub_dims, model_names = self.generate_batches(epoch=epoch,
                                                       process_function=self.process_one_dataset,
                                                       every_n_dim=every_n_dim,
                                                       model_weights = model_weights,
                                                       total_tasks = total_tasks)

        return inliners, LA, sub_dims,model_names



    def generate_one_epoch_then_train_one(self,
                                          every_n_dim, 
                                          save_data,
                                          model_weights = None,
                                          total_tasks = None,):
        s = time.time()
        print('current gen1tr1 epoch_id: {}'.format(self.gen1tr1_epoch_id))
        inliners, LA, sub_dims, model_names = self.generate_one_epoch(epoch=self.gen1tr1_epoch_id, 
                                                                      every_n_dim=every_n_dim,
                                                         model_weights=model_weights,
                                                         total_tasks=total_tasks)

        self.gen1tr1_epoch_id += 1
        print('generation time: {} min'.format((time.time() - s) / 60))
        return inliners, LA, sub_dims, model_names


        

    def get_batch_all_models(self, 
                             list_of_data, 
                             seq_len=100, 
                             hyperparameters=None,
                             **kwargs):
        #print(len(list_of_data))
        # this will be part of the collate_fn to help prepare batched data
        xs = []
        internal_xs = []
        ys = []
        model_names = []
        is_train = kwargs['training']
        internal_choice = kwargs['internal_choice']
        single_eval_pos = kwargs['single_eval_pos'] if is_train else seq_len - 1
        # print('seq_len:', seq_len)
        # print('single_eval_pos:', single_eval_pos)
        num_inliners = single_eval_pos
        num_test_x = seq_len - single_eval_pos
        ignore_index = hyperparameters['ignore_index']
        # print('ignore_index:', ignore_index)
        
        
        def make_x_y_with_stored_data_internal_100(train_test_in, test_la, weight_variable=None):
            train_test_in = train_test_in[torch.randperm(train_test_in.shape[0])]
            test_la = test_la[torch.randperm(test_la.shape[0])]

            inliners = train_test_in[:num_inliners]

            test_inliner = train_test_in[num_inliners:]
            test_la = test_la[:num_test_x]

            test_x = torch.cat([test_inliner, test_la], dim=0)
            test_y = torch.tensor([0] * num_test_x + [1] * num_test_x)

            sample_indices = torch.randperm(2 * num_test_x)[:num_test_x]

            test_x = test_x[sample_indices]
            test_y = test_y[sample_indices]

            x = torch.cat([inliners, test_x], dim=0)  # (num_inliners+num_test_x, dim)
            internal_features = self.compute_100_internal_features(x)
            
            y = torch.cat([torch.tensor([ignore_index] * num_inliners), test_y], dim=0)

            feature_dim = x.shape[-1]
            if feature_dim < self.max_feature_dim:
                x = self.FT.feature_padding_torch(x=x, num_feature=feature_dim)
                num_feature= internal_features.shape[1]
                internal_features = torch.cat([internal_features, torch.zeros(internal_features.shape[0], self.max_feature_dim - num_feature, device=x.device)], dim=-1)
           
            x = torch.cat([internal_features, x], dim=0)
            y = torch.cat([torch.tensor([0]*internal_features.shape[0]),y],dim=0)
            return x, y, internal_features.shape[0]   
        
        for data in list_of_data:
            inliners = data['in'][:self.seq_len, :]
            la = data['la'][:self.seq_len, :]
            model_name = data['model_name']
            x, y, prepend_dim = make_x_y_with_stored_data_internal_100(train_test_in=inliners, 
                                                                 test_la=la,
                                                                 weight_variable=model_name)
            xs.append(x)
            ys.append(y)
            internal_xs.append(model_name[1])
            model_names.append(model_name[0])

        xs = torch.stack(xs, dim=0)  #.to(torch.float)  # (bs, seq_len, dim)
        ys = torch.stack(ys, dim=0)  #.to(torch.float)  # (bs, seq_len)
        return Batch(x=xs.transpose(0, 1), y=None, internal_xs=internal_xs, target_y=ys.transpose(0, 1),model_names = model_names, single_eval_pos=single_eval_pos + prepend_dim) #+prepend_dim)



    def compute_100_internal_features(self, data):
        """
        Compute per-dimension summary statistics on GPU using PyTorch.

        Returns:
            stats: torch.Tensor of shape (100, dim)
            rows = 9 original stats + 91 additional stats
        """
        if not data.is_cuda:
            tensor = data.cuda()
        else:
            tensor = data

        eps = 1e-8

        # Handle finite values for nan/inf fractions
        finite_mask = torch.isfinite(tensor)
        safe_tensor = torch.where(finite_mask, tensor, torch.zeros_like(tensor))

        n = tensor.shape[0]
        finite_count = finite_mask.sum(dim=0).clamp_min(1)

        mean = safe_tensor.sum(dim=0) / finite_count
        centered = torch.where(finite_mask, tensor - mean, torch.zeros_like(tensor))

        variance = (centered ** 2).sum(dim=0) / finite_count
        std = torch.sqrt(variance + eps)

        skewness = (centered ** 3).sum(dim=0) / finite_count / ((std + eps) ** 3)
        kurtosis_val = (centered ** 4).sum(dim=0) / finite_count / ((std + eps) ** 4) - 3

        # Replace non-finite values before quantiles
        quant_tensor = torch.where(finite_mask, tensor, mean.unsqueeze(0))

        # Original quantiles
        base_q_levels = torch.tensor(
            [0.10, 0.25, 0.50, 0.75, 0.90],
            device=tensor.device,
            dtype=tensor.dtype,
        )
        q10, q25, q50, q75, q90 = torch.quantile(
            quant_tensor, base_q_levels, dim=0
        )

        original_stats = [
            mean, variance, skewness, kurtosis_val,
            q10, q25, q50, q75, q90
        ]

        # ------------------------------------------------------------------
        # 1. More quantiles: 39 stats
        # ------------------------------------------------------------------
        extra_q_levels = torch.tensor(
            [
                0.01, 0.025, 0.05, 0.075,
                0.125, 0.15, 0.175, 0.20, 0.225,
                0.275, 0.30, 0.325, 0.35, 0.375,
                0.40, 0.425, 0.45, 0.475,
                0.525, 0.55, 0.575, 0.60, 0.625,
                0.65, 0.675, 0.70, 0.725,
                0.775, 0.80, 0.825, 0.85, 0.875,
                0.925, 0.95, 0.975, 0.99,
                0.995, 0.999, 0.001,
            ],
            device=tensor.device,
            dtype=tensor.dtype,
        )
        extra_quantiles = torch.quantile(quant_tensor, extra_q_levels, dim=0)

        q01 = torch.quantile(quant_tensor, 0.01, dim=0)
        q05 = torch.quantile(quant_tensor, 0.05, dim=0)
        q95 = torch.quantile(quant_tensor, 0.95, dim=0)
        q99 = torch.quantile(quant_tensor, 0.99, dim=0)

        # ------------------------------------------------------------------
        # 2. Robust spread / interval widths: 15 stats
        # ------------------------------------------------------------------
        x_min = torch.min(quant_tensor, dim=0).values
        x_max = torch.max(quant_tensor, dim=0).values

        iqr = q75 - q25
        range_90 = q95 - q05
        range_80 = q90 - q10
        range_50 = q75 - q25
        lower_tail_25 = q25 - q10
        upper_tail_25 = q90 - q75
        lower_tail_10 = q10 - q01
        upper_tail_10 = q99 - q90
        median_abs_dev = torch.median(torch.abs(quant_tensor - q50), dim=0).values
        mean_abs_dev = torch.mean(torch.abs(quant_tensor - mean), dim=0)
        std_over_mean_abs = std / (torch.abs(mean) + eps)
        iqr_over_std = iqr / (std + eps)
        range_over_std = (x_max - x_min) / (std + eps)
        q90_q10_ratio = q90 / (torch.abs(q10) + eps)
        q75_q25_ratio = q75 / (torch.abs(q25) + eps)

        robust_stats = [
            iqr,
            range_90,
            range_80,
            range_50,
            lower_tail_25,
            upper_tail_25,
            lower_tail_10,
            upper_tail_10,
            median_abs_dev,
            mean_abs_dev,
            std_over_mean_abs,
            iqr_over_std,
            range_over_std,
            q90_q10_ratio,
            q75_q25_ratio,
        ]

        # ------------------------------------------------------------------
        # 3. Min/max/extreme stats: 8 stats
        # ------------------------------------------------------------------
        abs_tensor = torch.abs(quant_tensor)
        min_stat = x_min
        max_stat = x_max
        range_stat = x_max - x_min
        abs_min = torch.min(abs_tensor, dim=0).values
        abs_max = torch.max(abs_tensor, dim=0).values
        mean_abs = torch.mean(abs_tensor, dim=0)
        median_abs = torch.median(abs_tensor, dim=0).values
        rms = torch.sqrt(torch.mean(quant_tensor ** 2, dim=0) + eps)
        extreme_stats = [
            min_stat,
            max_stat,
            range_stat,
            abs_min,
            abs_max,
            mean_abs,
            median_abs,
            rms,
        ]
        # ------------------------------------------------------------------
        # 4. Higher moments: 8 stats
        # ------------------------------------------------------------------
        z = centered / (std + eps)
        moment_5 = torch.mean(z ** 5, dim=0)
        moment_6 = torch.mean(z ** 6, dim=0)
        raw_moment_2 = torch.mean(quant_tensor ** 2, dim=0)
        raw_moment_3 = torch.mean(quant_tensor ** 3, dim=0)
        raw_moment_4 = torch.mean(quant_tensor ** 4, dim=0)
        central_moment_2 = torch.mean(centered ** 2, dim=0)
        central_moment_3 = torch.mean(centered ** 3, dim=0)
        central_moment_4 = torch.mean(centered ** 4, dim=0)
        clip_value = 100.0
        higher_moment_stats = [
            torch.clamp(moment_5, min=-clip_value, max=clip_value),
            torch.clamp(moment_6, min=-clip_value, max=clip_value),
            torch.clamp(raw_moment_2, min=-clip_value, max=clip_value),
            torch.clamp(raw_moment_3, min=-clip_value, max=clip_value),
            torch.clamp(raw_moment_4, min=-clip_value, max=clip_value),
            torch.clamp(central_moment_2, min=-clip_value, max=clip_value),
            torch.clamp(central_moment_3, min=-clip_value, max=clip_value),
            torch.clamp(central_moment_4, min=-clip_value, max=clip_value),
        ]
        # ------------------------------------------------------------------
        # 5. Tail / outlier proportions: 12 stats
        # ------------------------------------------------------------------
        frac_below_q01 = torch.mean((quant_tensor < q01).float(), dim=0)
        frac_below_q05 = torch.mean((quant_tensor < q05).float(), dim=0)
        frac_below_q10 = torch.mean((quant_tensor < q10).float(), dim=0)
        frac_above_q90 = torch.mean((quant_tensor > q90).float(), dim=0)
        frac_above_q95 = torch.mean((quant_tensor > q95).float(), dim=0)
        frac_above_q99 = torch.mean((quant_tensor > q99).float(), dim=0)
        frac_abs_gt_1std = torch.mean((torch.abs(centered) > std).float(), dim=0)
        frac_abs_gt_2std = torch.mean((torch.abs(centered) > 2 * std).float(), dim=0)
        frac_abs_gt_3std = torch.mean((torch.abs(centered) > 3 * std).float(), dim=0)
        low_iqr_bound = q25 - 1.5 * iqr
        high_iqr_bound = q75 + 1.5 * iqr
        frac_iqr_outlier_low = torch.mean((quant_tensor < low_iqr_bound).float(), dim=0)
        frac_iqr_outlier_high = torch.mean((quant_tensor > high_iqr_bound).float(), dim=0)
        frac_iqr_outlier_total = frac_iqr_outlier_low + frac_iqr_outlier_high
        tail_stats = [
            frac_below_q01,
            frac_below_q05,
            frac_below_q10,
            frac_above_q90,
            frac_above_q95,
            frac_above_q99,
            frac_abs_gt_1std,
            frac_abs_gt_2std,
            frac_abs_gt_3std,
            frac_iqr_outlier_low,
            frac_iqr_outlier_high,
            frac_iqr_outlier_total,
        ]
        # ------------------------------------------------------------------
        # 6. Sign / zero / finite behavior: 6 stats
        # ------------------------------------------------------------------
        frac_positive = torch.mean((tensor > 0).float(), dim=0)
        frac_negative = torch.mean((tensor < 0).float(), dim=0)
        frac_zero = torch.mean((tensor == 0).float(), dim=0)
        frac_nan = torch.mean(torch.isnan(tensor).float(), dim=0)
        frac_inf = torch.mean(torch.isinf(tensor).float(), dim=0)
        frac_finite = torch.mean(torch.isfinite(tensor).float(), dim=0)
        sign_finite_stats = [
            frac_positive,
            frac_negative,
            frac_zero,
            frac_nan,
            frac_inf,
            frac_finite,
        ]
        # ------------------------------------------------------------------
        # 7. Shape asymmetry stats: 3 stats
        # ------------------------------------------------------------------
        bowley_skew = (q75 + q25 - 2 * q50) / (q75 - q25 + eps)
        tail_asymmetry = (q90 + q10 - 2 * q50) / (q90 - q10 + eps)
        mean_median_gap = (mean - q50) / (std + eps)

        asymmetry_stats = [
            bowley_skew,
            tail_asymmetry,
            mean_median_gap,
        ]
        stats = torch.cat(
            [
                torch.stack(original_stats, dim=0),
                extra_quantiles,
                torch.stack(robust_stats, dim=0),
                torch.stack(extreme_stats, dim=0),
                torch.stack(higher_moment_stats, dim=0),
                torch.stack(tail_stats, dim=0),
                torch.stack(sign_finite_stats, dim=0),
                torch.stack(asymmetry_stats, dim=0),
            ],
            dim=0,
        )
        return stats