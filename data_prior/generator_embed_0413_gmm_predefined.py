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

from data_prior.gmm_embed_0413 import make_NdMclusterGMM_predefined
from data_prior.feature_transform import FeatureTransform
from dataset_loader.batch import Batch
import joblib

def load_pickle(file_path):
    with open(file_path, 'rb') as handle:
        inst = pickle.load(handle)
    return inst


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
        model_choices.append(("gmm", make_NdMclusterGMM_predefined, gmm_params))
        self.model_choices = model_choices
         

    @staticmethod
    def process_one_dataset(epoch_id,
                            step, 
                            device,
                            every_n_dim,
                            model_choices,
                            embeds,
                            mean,
                            variance,
                            cluster,
                            model_weights = None,
                            return_params=True): 
        #torch.cuda.empty_cache()
        if model_weights is None:
            model_index, model_entry = random.choice(list(enumerate(model_choices)))
        else:
            model_entry = next((name, constructor, params) for name, constructor, params in model_choices if name == model_weights)
            
        model_name, model_constructor, params = model_entry
        seed = epoch_id + step + os.getpid()
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        params['device'] = device
        
        # You can have model-specific logic below
        if model_name == "gmm":
            #random grab a numpy file from 
            model = model_constructor(
                means = mean,
                diag_values = variance,
                embeds = embeds,
                device= device,
                num_cluster = cluster,
            )
            
            embeddings = torch.from_numpy(model.embeds)          
            inliers, LA = params["generate_fn"](model=model)
            
            #print(model_name)      
            #print(inliers.shape, LA.shape, embeddings.shape)
            
            sub_dims = model.sub_dims
            del model
            torch.cuda.empty_cache()
            return inliers, LA, sub_dims, (model_name,embeddings)
          
    
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
        
        device_str = self.device
        gpu_id = int(device_str.split(":")[1])
        
        #randomly select a file from the prior-embed directory
        root_file = '/ocean/projects/cis250290p/xding/prior_embedding/prior_embed'
        random_file = random.choice(os.listdir(root_file))
        print(f'random file: {random_file} selected')
        # mean/variance arrays are stored as dtype=object in the source .npz,
        # so loading requires allow_pickle=True.
        data_info = np.load(os.path.join(root_file, random_file), allow_pickle=True)
        embeds = data_info['embeds'][gpu_id * 500 : (gpu_id+1) * 500]
        mean_embeds = data_info['mean_embeds'][gpu_id * 500 : (gpu_id+1) * 500]
        variance_embeds = data_info['variance_embeds'][gpu_id * 500 : (gpu_id+1) * 500]
        cluster_embeds = data_info['cluster_embeds'][gpu_id * 500 : (gpu_id+1) * 500]

        print(f'using GPU to generate fast')
        inliners_list, LA_list, sub_dims_list = [], [], []
        model_name_list = []
        for step in tqdm(range(total_tasks)):
            inliners, LA, sub_dims, model_name = process_function(
                epoch_id=epoch*total_tasks,
                step=step,
                device = self.device,
                every_n_dim =every_n_dim,
                model_choices = self.model_choices,
                embeds = embeds[step],
                mean = mean_embeds[step],
                variance = variance_embeds[step],
                cluster = cluster_embeds[step]
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
        ys = []
        internal_xs = []
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
        
        
        def make_x_y_with_stored_data_global(train_test_in, test_la, weight_variable=None):
            global_internal = weight_variable[1]
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
            y = torch.cat([torch.tensor([ignore_index] * num_inliners), test_y], dim=0)

            feature_dim = x.shape[-1]
            if feature_dim < self.max_feature_dim:
                x = self.FT.feature_padding_torch(x=x, num_feature=feature_dim)
            
            internal_features = global_internal.to(x.device) 
            return x, y, internal_features
        

        for data in list_of_data:
            inliners = data['in'][:self.seq_len, :]
            la = data['la'][:self.seq_len, :]
            model_name = data['model_name']
            x, y, embeds = make_x_y_with_stored_data_global(train_test_in=inliners, 
                                                                 test_la=la,
                                                                 weight_variable=model_name)
            xs.append(x)
            ys.append(y)
            internal_xs.append(embeds)
            model_names.append(model_name[0])

        xs = torch.stack(xs, dim=0)  #.to(torch.float)  # (bs, seq_len, dim)
        ys = torch.stack(ys, dim=0)  #.to(torch.float)  # (bs, seq_len)
        internal_xs = torch.stack(internal_xs, dim= 0)
        return Batch(x=xs.transpose(0, 1),
                     y=None,
                     internal_xs= internal_xs.transpose(0,1),
                     target_y=ys.transpose(0, 1),
                     model_names = model_names, 
                     single_eval_pos=single_eval_pos)