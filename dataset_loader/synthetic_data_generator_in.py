import numpy as np
import pandas as pd
import random
import os
from math import ceil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from adbench.myutils import Utils


SYNTHETIC_DATA_PATH = '/home/xding2/PIQL/synbench/gaussian_in/'

# currently, data generator only supports for generating the binary classification datasets
class DataGenerator():
    def __init__(self, 
                 seed: int = 42,
                 dataset: str = None):
        '''
        :param seed: seed for reproducible results
        :param dataset: specific the dataset name
        :param test_size: testing set size
        :param generate_duplicates: whether to generate duplicated samples when sample size is too small
        :param n_samples_threshold: threshold for generating the above duplicates, if generate_duplicates is False, then datasets with sample size smaller than n_samples_threshold will be dropped
        '''

        self.seed = seed
        self.dataset = dataset
        # dataset list
        self.dataset_list_gmm = [os.path.splitext(_)[0] for _ in os.listdir(SYNTHETIC_DATA_PATH)
                                if os.path.splitext(_)[1] == '.npz'] 
        # myutils function
        self.utils = Utils()


    def generator(self, 
                  X=None, 
                  y=None, 
                  scale=False,
                  la=None,
                  at_least_one_labeled=False,
                  noise_type=None, 
                  duplicate_times: int = 2,
                  max_size=10000, 
                  validation=False,): 
        # set seed for reproducible results
        self.utils.set_seed(self.seed)
        if self.dataset in self.dataset_list_gmm:
            data = np.load(os.path.join(SYNTHETIC_DATA_PATH, self.dataset + '.npz'), allow_pickle=True)
        else:
            raise NotImplementedError

        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
        mean_train = data['mean_train']
        mean_test = data['mean_test']
        variance_train = data['variance_train']
        variance_test = data['variance_test']
        global_mean =  data['global_mean']
        global_variance = data['global_variance']
        global_anomaly_mean = data['global_anomaly_mean']
        global_anomaly_variance= data['global_anomaly_variance']
        gmm_text = data['gmm_text']

        # standard scaling
        if scale:
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            col_mean = np.nanmean(X_train, axis=0)
            inds = np.where(np.isnan(col_mean))
            col_mean[inds] = 0

            inds = np.where(np.isnan(X_train))
            X_train[inds] = np.take(col_mean, inds[1])

            col_mean = np.nanmean(X_test, axis=0)
            inds = np.where(np.isnan(col_mean))
            col_mean[inds] = 0

            inds = np.where(np.isnan(X_test))
            X_test[inds] = np.take(col_mean, inds[1])

        return {'X_train': X_train, 
                'y_train': y_train, 
                'X_test': X_test, 
                'y_test': y_test,
                'mean_train': mean_train,
                'variance_train': variance_train,
                'mean_test': mean_test,
                'variance_test': variance_test,
                'global_mean': global_mean,
                'global_variance':global_variance,
                'global_anomaly_mean':global_anomaly_mean,
                'global_anomaly_variance':global_anomaly_variance,
                'gmm_text': gmm_text}