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

from data_prior.gmm_embed import make_NdMclusterGMM
from data_prior.scm_embed import make_contextualSCM, make_probSCM
from data_prior.copula_embed import make_disturb_corpula, make_perturb_corpula
from data_prior.feature_transform import FeatureTransform
from dataset_loader.batch import Batch
import joblib

def load_pickle(file_path):
    with open(file_path, 'rb') as handle:
        inst = pickle.load(handle)
    return inst

from torch import Tensor
from sentence_transformers import SentenceTransformer

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
        self.embed_model  = SentenceTransformer("/home/xding2/FoMo-Meta/bge-m3")
        # a pre-fitted pca model to dimension reduction
        self.pca = joblib.load("/home/xding2/FoMo-Meta/data_prior/pca_100.joblib")


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
                            pca,
                            embed_model,
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
            prior_description = model.prior_description
            #embed_model = embed_model.to(device)
            embeddings = embed_model.encode(prior_description, show_progress_bar=False)#,device=device)
            #reshape the embeddings if it is only one dimensional 
            if embeddings.ndim == 1:
                embeddings = embeddings[None, :]  # (d,) -> (1, d)
            embeddings = pca.transform(embeddings)
            embeddings = torch.from_numpy(embeddings)             
            inliers, LA = params["generate_fn"](model=model)
            
            # print(model_name)      
            # print(inliers.shape, LA.shape, embeddings.shape)
            
            sub_dims = model.sub_dims
            del model
            torch.cuda.empty_cache()
            return inliers, LA, sub_dims, (model_name,embeddings)
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
            prior_description = model.prior_description
            #embed_model = embed_model.to(device)
            embeddings = embed_model.encode(prior_description, show_progress_bar=False)#, device= device)
            #reshape the embeddings if it is only one dimensional 
            if embeddings.ndim == 1:
                embeddings = embeddings[None, :]  # (d,) -> (1, d)
            embeddings = pca.transform(embeddings)
            embeddings = torch.from_numpy(embeddings)
            
            # print(model_name)      
            # print(inliers.shape, LA.shape, embeddings.shape)
            
            del model
            return inliers, LA, None, (model_name,embeddings)
        elif model_name == "disturbcorpula":
            feature_dim = np.random.randint(low=2, high=params["max_feature_dim"] + 1)
            dim = feature_dim
            model = model_constructor(device= device,dim= dim)
            prior_description = model.prior_description
            #embed_model = embed_model.to(device)
            embeddings = embed_model.encode(prior_description, show_progress_bar=False)#, device=device)
            #reshape the embeddings if it is only one dimensional 
            if embeddings.ndim == 1:
                embeddings = embeddings[None, :]  # (d,) -> (1, d)
            embeddings = pca.transform(embeddings)
            embeddings = torch.from_numpy(embeddings)  
            inliers, LA = params["generate_fn"](model=model)
            
            del model
            return inliers, LA, None, (model_name,embeddings)
        
        elif model_name == "perturbcorpula":
            feature_dim = np.random.randint(low=2, high=params["max_feature_dim"] + 1)
            dim = feature_dim
            model = model_constructor(device= device,
                                      dim= dim)
            prior_description = model.prior_description
            #embed_model = embed_model.to(device)
            embeddings = embed_model.encode(prior_description, show_progress_bar=False) #, device=device)
            #reshape the embeddings if it is only one dimensional 
            if embeddings.ndim == 1:
                embeddings = embeddings[None, :]  # (d,) -> (1, d)
            embeddings = pca.transform(embeddings)
            embeddings = torch.from_numpy(embeddings)  
            inliers, LA = params["generate_fn"](model=model)
            
            del model
            return inliers, LA, None, (model_name,embeddings)
        
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
                    pca = self.pca,
                    embed_model = self.embed_model
                    )
            else:
                inliners, LA, sub_dims, model_name = process_function(
                    epoch_id=epoch*total_tasks,
                    step=step,
                    device = self.device,
                    every_n_dim =every_n_dim,
                    model_choices = self.model_choices,
                    pca = self.pca,
                    embed_model = self.embed_model
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
            x = torch.cat([internal_features, x], dim=0)
            y = torch.cat([torch.tensor([0]*internal_features.shape[0]),y],dim=0) 
            return x, y, internal_features.shape[0]
        

        for data in list_of_data:
            inliners = data['in'][:self.seq_len, :]
            la = data['la'][:self.seq_len, :]
            model_name = data['model_name']
            x, y, prepend_dim = make_x_y_with_stored_data_global(train_test_in=inliners, 
                                                                 test_la=la,
                                                                 weight_variable=model_name)
            xs.append(x)
            ys.append(y)
            model_names.append(model_name[0])

        xs = torch.stack(xs, dim=0)  #.to(torch.float)  # (bs, seq_len, dim)
        ys = torch.stack(ys, dim=0)  #.to(torch.float)  # (bs, seq_len)
        return Batch(x=xs.transpose(0, 1), y=None, target_y=ys.transpose(0, 1),model_names = model_names, single_eval_pos=single_eval_pos+prepend_dim)