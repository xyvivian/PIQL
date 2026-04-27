from abc import ABC
from data_prior.GMM import make_NdMclusterGMM, generate_linear_transform, transform_samples
from torch.utils.data import Dataset, IterableDataset, DataLoader
import pickle
import torch
import pytorch_lightning as pl
import random
import gc
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset, IterableDataset, DataLoader
import os
import torch
import numpy as np
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler
#from batch import Batch
import json

def load_pickle(file_path):
    with open(file_path, 'rb') as handle:
        inst = pickle.load(handle)
    return inst


def torch_nanmean(x, axis=0, return_nanshare=False):
    num = torch.where(torch.isnan(x), torch.full_like(x, 0), torch.full_like(x, 1)).sum(axis=axis)
    value = torch.where(torch.isnan(x), torch.full_like(x, 0), x).sum(axis=axis)
    if return_nanshare:
        return value / num, 1. - num / x.shape[axis]
    return value / num

def torch_nanstd(x, axis=0):
    num = torch.where(torch.isnan(x), torch.full_like(x, 0), torch.full_like(x, 1)).sum(axis=axis)
    value = torch.where(torch.isnan(x), torch.full_like(x, 0), x).sum(axis=axis)
    mean = value / num
    mean_broadcast = torch.repeat_interleave(mean.unsqueeze(axis), x.shape[axis], dim=axis)
    return torch.sqrt(torch.nansum(torch.square(mean_broadcast - x), axis=axis) / (num - 1))


def normalize_data(data, normalize_positions=-1, return_scaling=False):
    if normalize_positions > 0:
        mean = torch_nanmean(data[:normalize_positions], axis=0)
        std = torch_nanstd(data[:normalize_positions], axis=0) + .000001
    else:
        mean = torch_nanmean(data, axis=0)
        std = torch_nanstd(data, axis=0) + .000001
    data = (data - mean) / std
    data = torch.clip(data, min=-100, max=100)

    if return_scaling:
        return data, (mean, std)
    return data


def feature_subsampling(x: torch.Tensor, num_feature: int, max_feature_dim: int) -> torch.Tensor:
    # Randomly choose feature indices without replacement
    idx = torch.randperm(num_feature)[:max_feature_dim]
    # Sort to keep feature order
    idx, _ = torch.sort(idx)
    return x[:, idx]


def feature_scale(x: torch.Tensor, 
                  num_feature: int, 
                  max_feature_dim: int,
                  rescale_with_sqrt: bool = False) -> torch.Tensor:
    scale = num_feature / max_feature_dim
    if rescale_with_sqrt:
        scale = scale ** 0.5
    return x / scale


def pfn_transform(eval_xs,
                  max_feature_dim):
    """
    :param eval_xs: the inputs
    :param preprocess_transform: str, 'none', 'power', 'quantile', 'robust'
    :param eval_position: train-x <--eval_position---> test_x
    :param normalize_with_test: when perform (x-mean)/std, whether to include test_x
    :param rescale_with_sqrt: when rescale the features, whether to use "* sqrt(num-max-feat/num-used-feat)"
    """
    num_feature = eval_xs.shape[-1]
    if num_feature > max_feature_dim:
        eval_xs = feature_subsampling(x=eval_xs, num_feature=num_feature,max_feature_dim = max_feature_dim)
    eval_xs = normalize_data(eval_xs, normalize_positions=-1)
    # Rescale X
    eval_xs = feature_scale(x=eval_xs, num_feature=eval_xs.shape[-1], rescale_with_sqrt=False, max_feature_dim=max_feature_dim)
    if eval_xs.shape[-1] < max_feature_dim:
        eval_xs = torch.cat([eval_xs,
                            torch.zeros((eval_xs.shape[0], max_feature_dim - num_feature), device=eval_xs.device)],
                             dim=-1)
    return eval_xs



def sample(df: pd.DataFrame, 
           category_column: str, 
           anomaly_ratio: float, 
           random_state: int = 42, 
           norm_rank: int=1,
           max_normal_sample=5000,
           max_anomaly_sample=1000) -> pd.DataFrame:
    """
    Sample a dataset into 'normal' and 'anomaly' subsets based on a categorical column.

    - The most frequent category is treated as the normal class.
    - Anomalies are randomly sampled from all other classes according to the specified ratio.

    Parameters:
    - df: Input DataFrame containing the data.
    - category_column: Name of the categorical column used to define classes.
    - anomaly_ratio: Fraction of normal samples to include as anomalies (e.g., 0.1 for 10%).
    - random_state: Optional seed for reproducibility.
    - norm_rank: choose the top-norm_rank frequent class as norm samples


    Returns:
    - A DataFrame consisting of all normal samples and a random subset of anomalies.
    """
    # Skip if the DataFrame has more than 100 columns
    if df.shape[1] > 100:
        return None
    
    # Skip if the DataFrame has any non-numeric columns (excluding the category column)
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    non_numeric_cols = [col for col in non_numeric_cols if col != category_column]
    if len(non_numeric_cols) > 0:
        return None
    
    # Identify the normal class as the most frequent category
    counts = df[category_column].value_counts()
    # normal_class = counts.idxmax() # use the top 1
    norm_rank = min(len(counts), norm_rank)
    normal_class = counts.index[norm_rank - 1]

    # Split normals and potential anomalies
    normal_df = df[df[category_column] == normal_class].copy()
    n_norm = min(len(normal_df),max_normal_sample)
    normal_df = normal_df.sample(n=n_norm, random_state=random_state).copy()
    normal_df['anomaly_label'] = [0] * len(normal_df)
    other_df = df[df[category_column] != normal_class]

    # Determine number of anomalies to sample
    n_norm = len(normal_df)
    if anomaly_ratio >= 1.0:
        raise ValueError("anomaly_ratio must be less than 1")
    n_anom = int(anomaly_ratio * n_norm / (1.0 - anomaly_ratio))
    n_anom = min(n_anom, len(other_df), max_anomaly_sample)

    # Sample anomalies
    anomaly_df = other_df.sample(n=n_anom, random_state=random_state)
    anomaly_df['anomaly_label'] = [1] * len(anomaly_df)

    # Combine and shuffle
    combined = pd.concat([normal_df, anomaly_df]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    combined = combined.drop(category_column, axis=1, errors='ignore')
    return combined.astype(float).to_numpy()



def load_pickle(file_path):
    with open(file_path, 'rb') as handle:
        inst = pickle.load(handle)
    return inst


class EpochDataset(Dataset):
    def __init__(self, batch_size, seq_len, steps_per_epoch, hyperparameters, reuse_data_every_n, max_model_dim,
                 max_num_cluster, get_batch_method, rank, num_device, training, single_eval_pos_gen, data_path,
                 is_source_numpy, choices):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.steps_per_epoch = steps_per_epoch
        self.hyperparameters = hyperparameters
        self.internal_choice = choices
        self.reuse_data_every_n = reuse_data_every_n
        self.max_model_dim = max_model_dim
        self.max_num_cluster = max_num_cluster
        self.current_epoch = 0

        self.get_batch_method = get_batch_method
        self.rank = rank
        self.num_device = num_device
        self.training = training
        self.data_path = data_path
        self.single_eval_pos_gen = single_eval_pos_gen
        self.is_source_numpy = is_source_numpy

        self.in_data = None if data_path is None else load_pickle(file_path=f'{data_path}/epoch0/in.pickle')
        self.la_data = None if data_path is None else load_pickle(file_path=f'{data_path}/epoch0/la.pickle')
        self.model_name = None
        self.cached_single_eval_pos = None
        self.cached_single_eval_pos_epoch = None

    def set_epoch_and_data(self, epoch, data_dict=None):
        """
        Update the dataset to use data from the specified epoch.
        """
        self.current_epoch = epoch
        print('setting current epoch to:', epoch)

        if data_dict is not None:  # reuse saved data
            print('new data loaded...')
            self.in_data = data_dict['in']
            print('in data shape:', len(self.in_data))
            print(self.in_data[0].shape)
            self.la_data = data_dict['la']
            print('la data shape:', len(self.la_data))
            print(self.la_data[0].shape)
            self.model_names = data_dict['model_names']
            print('model names length:', len(self.model_names))

    def free_data(self):
        self.in_data = None
        self.la_data = None
        self.model_names = None
        gc.collect()
        torch.cuda.empty_cache()

    def set_training_mode(self, training):
        self.training = training

    def set_rank(self, rank):
        self.rank = rank
        print(f'rank is successfully set to {rank} out of {self.num_device} devices')

    def __len__(self):
        return int(self.steps_per_epoch * self.batch_size / self.num_device)

    def __getitem__(self, idx):
        if self.is_source_numpy:  # train/validation (stored in numpy)
            return {'in': torch.from_numpy(self.in_data[idx]).to(torch.float),
                    'la': torch.from_numpy(self.la_data[idx]).to(torch.float),
                    'model_name': None}
        else:  # train (generate on the fly)
            return {'in': self.in_data[idx], 'la': self.la_data[idx], 'model_name': self.model_names[idx], 'idx':idx}

    def prior_batch_collate_fn(self, batch_list):
        random_seed = triple_seed(base_seed=42,
                                  epoch=self.current_epoch,
                                  idx=batch_list[0]['idx'], 
                                  rank=self.rank)
        single_eval_pos = self.single_eval_pos_gen.generate(random_seed) 
        batch = self.get_batch_method(list_of_data=batch_list, seq_len=self.seq_len,
                                      hyperparameters=self.hyperparameters,
                                      training=self.training,
                                      single_eval_pos=single_eval_pos,
                                      internal_choice=self.internal_choice,
                                      )
        return batch



#===========For random seeding ==============
import random, hashlib

def triple_seed(base_seed: int, epoch: int, idx: int, rank: int, bits: int = 64) -> int:
    """
    Deterministically map (base_seed, epoch, idx, rank) -> integer seed.
    Uses SHA-256 so it's stable across processes and Python versions.
    """
    msg = f"{base_seed}:{epoch}:{idx}:{rank}".encode()
    h = hashlib.sha256(msg).digest()
    # take first `bits` bits from the hash as an integer seed
    return int.from_bytes(h[:bits // 8], "big")
