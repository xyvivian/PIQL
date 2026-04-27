import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from scipy.stats import chi2
import time


def _fmt_vec(x, decimals=3, max_len=8):
    vals = x.detach().to('cpu').flatten().tolist()
    shown = vals[:max_len]
    s = ",".join(f"{v:.{decimals}f}" for v in shown)
    if len(vals) > max_len:
        s += ",..."
    return s


def describe_gmm_model(model, decimals=3, max_mean_len=10, max_cov_len=10):
    """Return a compact text descriptor for a GaussianMixtureModel."""
    lines = []
    lines.append(f"[FAMILY:GMM] [DIM:{int(model.d)}] [N_COMP:{int(model.num_cluster)}]")

    for k in range(int(model.num_cluster)):
        w = float(model.weights[k])
        mean = model.means[k]
        cov = model.covariances[k]

        mean_txt = _fmt_vec(mean, decimals=decimals, max_len=max_mean_len)
        cov_diag_txt = _fmt_vec(torch.diagonal(cov, dim1=-2, dim2=-1), decimals=decimals, max_len=max_cov_len)

        lines.append(
            f"[COMP:{k}] [W:{w:.{decimals}f}] [MEAN:{mean_txt}] [COV:{cov_diag_txt}]"
        )

    # Optional outlier/inflation info from model fields
    sub_dims_txt = _fmt_vec(model.sub_dims.to(torch.float32), decimals=0, max_len=12)
    lines.append(
        f"[OUTLIER:inflated_cov] [SUB_DIMS:{sub_dims_txt}] "
        f"[THR_MINUS:{float(model.threshold_minus_delta):.{decimals}f}] "
        f"[THR_PLUS:{float(model.threshold_plus_delta):.{decimals}f}]"
    )
    return "\n".join(lines)

class GaussianMixtureModel:
    def __init__(self, means: torch.Tensor, covariances: torch.Tensor, weights: torch.Tensor,
                 percentile=0.80, delta=0.05, inflate_scale=5.0, inflate_full=False, sub_dims=None, device='cpu'):
        self.device = device
        self.means = means
        self.covariances = covariances
        self.weights = weights

        self.num_cluster = len(means)

        d = self.means[0].shape[0]
        self.d = d
        n = d if inflate_full else np.random.randint(1, d + 1)  # Generate random integer between 1 and d, inclusive
        if inflate_full and sub_dims is not None:
            print('we are inflating all the dimensions, however, sub_dims is provided')
            raise Exception
        self.sub_dims = torch.sort(torch.randperm(d)[:n]).values.to(self.device) if sub_dims is None else sub_dims.to(
            self.device)

        self.threshold_plus_delta = chi2.ppf(percentile + delta, df=len(self.sub_dims))
        self.threshold_minus_delta = chi2.ppf(percentile - delta, df=len(self.sub_dims))

        self.inflated_covariances = []
        self.inv_sub_covariances = []

        for cov in self.covariances:
            cov_copy = cov.clone()
            sub_cov = cov_copy[self.sub_dims, :][:, self.sub_dims]  # the sub_cov defines a new (smaller) Gaussian
            # n x n ---> sub_dims x n ---> sub_dims x sub_dims
            self.inv_sub_covariances.append(torch.linalg.inv(sub_cov))
            cov_copy[self.sub_dims[:, None], self.sub_dims] *= inflate_scale
            self.inflated_covariances.append(cov_copy)

        self.inflated_covariances = torch.stack(self.inflated_covariances)
        self.inv_sub_covariances = torch.stack(self.inv_sub_covariances)

        self.GMM4sample = [MultivariateNormal(self.means[cluster_id], self.covariances[cluster_id])
                           for cluster_id in range(len(self.weights))]
        
        self.ret_means = self.means.contiguous()
        self.ret_variances = torch.diagonal(self.covariances, dim1=-2, dim2=-1).contiguous()


        self.GMM4inf = [MultivariateNormal(self.means[cluster_id], self.inflated_covariances[cluster_id])
                        for cluster_id in range(len(self.weights))]
        
        self.inf_ret_means = self.means.contiguous()
        self.inf_ret_variances = torch.diagonal(self.inflated_covariances, dim1=-2, dim2=-1).contiguous()
        
        
        self.prior_description = describe_gmm_model(self)
        

    def draw_samples(self, num_samples, return_params=True):
        """
        Draws samples from the d-dimensional Gaussian Mixture Model.

        Parameters:
        num_samples (int): Number of samples to draw.
        return_params (bool): If True, also return per-sample means/covariances
                              and component ids.

        Returns:
        torch.Tensor: samples drawn from the GMM.
        If return_params is True, returns:
            samples (num_samples, d),
            sample_means (num_samples, d),
            sample_covariances (num_samples, d),  # diagonal variances only
            component_choices (num_samples,)
        """
        samples = torch.zeros(num_samples, self.d, device=self.device)
        sample_means = torch.zeros(num_samples, self.d, device=self.device) if return_params else None
        sample_covariances = torch.zeros(num_samples, self.d, device=self.device) if return_params else None
        component_choices = torch.multinomial(self.weights, num_samples, replacement=True)
        for cluster_id in range(self.num_cluster):
            mask = (component_choices == cluster_id)
            num_cluster_samples = mask.sum().item()
            if num_cluster_samples > 0:
                sample = self.GMM4sample[cluster_id].sample((num_cluster_samples,))
                samples[mask] = sample
                if return_params:
                    sample_means[mask] = self.means[cluster_id]
                    sample_covariances[mask] = torch.diagonal(self.covariances[cluster_id], dim1=-2, dim2=-1)
        if return_params:
            return samples, sample_means, sample_covariances, component_choices
        return samples

    def draw_inflated_samples(self, num_samples, return_params=True):
        """
        Draws samples from the d-dimensional inflated Gaussian Mixture Model.

        Parameters:
        num_samples (int): Number of samples to draw.

        Returns:
        torch.Tensor: samples drawn from the inflated GMM.
        """
        samples = torch.zeros(num_samples, self.d, device=self.device)
        component_choices = torch.multinomial(self.weights, num_samples, replacement=True)
        sample_means = torch.zeros(num_samples, self.d, device=self.device) if return_params else None
        sample_covariances = torch.zeros(num_samples, self.d, device=self.device) if return_params else None
        for cluster_id in range(self.num_cluster):
            mask = (component_choices == cluster_id)
            num_cluster_samples = mask.sum().item()
            if num_cluster_samples > 0:
                sample = self.GMM4inf[cluster_id].sample((num_cluster_samples,))  # .type_as(self.weights)
                samples[mask] = sample
                if return_params:
                    sample_means[mask] = self.means[cluster_id]
                    sample_covariances[mask] = torch.diagonal(self.covariances[cluster_id], dim1=-2, dim2=-1)
        if return_params:
            return samples, sample_means, sample_covariances, component_choices
        return samples


    def mahalanobis_distance(self, sample, mean, inv_covariance):
        """
        Computes the Mahalanobis distance of a sample from a given mean and inverse covariance matrix.

        Parameters:
        sample (torch.Tensor): Sample point. (d, )
        mean (torch.Tensor): Mean vector.  (d, )
        inv_covariance (torch.Tensor): Inverse covariance matrix.  (sub-dims, )

        Returns:
        float: Mahalanobis distance of the sample from the mean.
        """
        delta = sample[self.sub_dims] - mean[self.sub_dims]
        return torch.sqrt((delta @ inv_covariance @ delta).sum())

    def batched_squared_mahalanobis_distance(self, X, mean, inv_cov):
        delta = X[:, self.sub_dims] - mean[self.sub_dims]
        return torch.diag(delta @ inv_cov @ delta.T)



    def draw_inliers(self, num_samples, return_params=True):
        batch_size = max(num_samples * 2, 1000)
        samples = []
        sample_means = [] if return_params else None
        sample_covariances = [] if return_params else None
        total_samples_needed = num_samples
        while total_samples_needed > 0:
            raw_samples, raw_means, raw_covariances, _ = self.draw_samples(batch_size, return_params=True)
            batch_distances = self.get_squared_batched_dist(raw_samples)
            min_squared_distances = torch.min(batch_distances, dim=1).values
            inliner_mask = min_squared_distances < self.threshold_minus_delta
            selected_samples = raw_samples[inliner_mask]
            num_selected = selected_samples.shape[0]
            if num_selected > 0:
                if num_selected >= total_samples_needed:
                    samples.append(selected_samples[:total_samples_needed])
                    if return_params:
                        sample_means.append(raw_means[inliner_mask][:total_samples_needed])
                        sample_covariances.append(raw_covariances[inliner_mask][:total_samples_needed])
                    total_samples_needed = 0
                else:
                    samples.append(selected_samples)
                    if return_params:
                        sample_means.append(raw_means[inliner_mask])
                        sample_covariances.append(raw_covariances[inliner_mask])
                    total_samples_needed -= num_selected
        samples = torch.cat(samples)
        if return_params:
            sample_means = torch.cat(sample_means)
            sample_covariances = torch.cat(sample_covariances)
            return samples, sample_means, sample_covariances
        return samples

    def draw_local_anomalies(self, num_samples, return_params=True):
        batch_size = max(num_samples * 2, 1000)
        samples = []
        sample_means = [] if return_params else None
        sample_covariances = [] if return_params else None
        total_samples_needed = num_samples
        while total_samples_needed > 0:
            raw_samples, raw_means, raw_covariances, _ = self.draw_inflated_samples(batch_size, return_params=True)
            batch_distances = self.get_squared_batched_dist(raw_samples)
            min_squared_distances = torch.min(batch_distances, dim=1).values
            anomaly_mask = min_squared_distances > self.threshold_plus_delta
            selected_samples = raw_samples[anomaly_mask]
            num_selected = selected_samples.shape[0]
            if num_selected > 0:
                if num_selected >= total_samples_needed:
                    samples.append(selected_samples[:total_samples_needed])
                    if return_params:
                        sample_means.append(raw_means[anomaly_mask][:total_samples_needed])
                        sample_covariances.append(raw_covariances[anomaly_mask][:total_samples_needed])
                    total_samples_needed = 0
                else:
                    samples.append(selected_samples)
                    if return_params:
                        sample_means.append(raw_means[anomaly_mask])
                        sample_covariances.append(raw_covariances[anomaly_mask])
                    total_samples_needed -= num_selected
        samples = torch.cat(samples)
        if return_params:
            sample_means = torch.cat(sample_means)
            sample_covariances = torch.cat(sample_covariances)
            return samples, sample_means, sample_covariances
        return samples

    def assert_inliers(self, samples):
        for sample in samples:
            distances = [self.mahalanobis_distance(sample, mean, inv_cov) for mean, inv_cov in
                         zip(self.means, self.inv_sub_covariances)]
            assert min(distances) ** 2 < self.threshold_minus_delta

    def assert_local_anomalies(self, samples):
        for sample in samples:
            distances = [self.mahalanobis_distance(sample, mean, inv_cov) for mean, inv_cov in
                         zip(self.means, self.inv_sub_covariances)]
            assert min(distances) ** 2 > self.threshold_plus_delta

    def get_squared_batched_dist(self, raw_samples):
        batch_dist = []
        for mean, inv_cov in zip(self.means, self.inv_sub_covariances):
            distances = self.batched_squared_mahalanobis_distance(X=raw_samples, mean=mean, inv_cov=inv_cov)
            batch_dist.append(distances)
        return torch.stack(batch_dist, dim=1)  # (#samples, num_cluster)

    def draw_batched_data(self, num_inliers, num_local_anomalies, return_params=True):
        raw_inliers, ret_means, ret_covariances,_ = self.draw_samples(num_samples=int(num_inliers * 2), return_params=return_params)
        raw_local_anomalies,ret_anomaly_means, ret_anomaly_covariances,_ = self.draw_inflated_samples(num_samples=int(num_local_anomalies * 2),return_params=True)

        inliers_squared_dist = self.get_squared_batched_dist(raw_samples=raw_inliers)
        local_anomalies_squared_dist = self.get_squared_batched_dist(raw_samples=raw_local_anomalies)

        min_inliers_squared_dist = torch.min(inliers_squared_dist, dim=1).values
        min_local_anomalies_squared_dist = torch.min(local_anomalies_squared_dist, dim=1).values

        inliers_mask = min_inliers_squared_dist < self.threshold_minus_delta  # (#raw-inliers, )
        local_anomalies_mask = min_local_anomalies_squared_dist > self.threshold_plus_delta  # (#raw-la, )

        inliers = raw_inliers[inliers_mask][:num_inliers]
        local_anomalies = raw_local_anomalies[local_anomalies_mask][:num_local_anomalies]
        
        ret_means = ret_means[inliers_mask][:num_inliers]
        ret_covariances= ret_covariances[inliers_mask][:num_inliers]
        
        ret_anomaly_means = ret_anomaly_means[local_anomalies_mask][:num_local_anomalies]
        ret_anomaly_covariances = ret_anomaly_covariances[local_anomalies_mask][:num_local_anomalies]

        def add_extra(existing_samples, existing_means, existing_covariances, target_num_samples, draw_func):
            if existing_samples.shape[0] < target_num_samples:
                extra_samples, extra_means, extra_covariances = draw_func(num_samples=target_num_samples - existing_samples.shape[0], return_params=True)
                existing_samples = torch.concat([existing_samples, extra_samples], dim=0)
                existing_means = torch.concat([existing_means, extra_means], dim=0)
                existing_covariances = torch.concat([existing_covariances, extra_covariances], dim=0)
            return existing_samples, existing_means, existing_covariances

        inliers, ret_means, ret_covariances = add_extra(existing_samples=inliers, existing_means=ret_means, existing_covariances=ret_covariances, target_num_samples=num_inliers, draw_func=self.draw_inliers)
        local_anomalies, ret_anomaly_means, ret_anomaly_covariances = add_extra(existing_samples=local_anomalies, existing_means=ret_anomaly_means, existing_covariances=ret_anomaly_covariances, target_num_samples=num_local_anomalies,
                                    draw_func=self.draw_local_anomalies)
        return inliers, local_anomalies


def make_NdMclusterGMM(dim: int, num_cluster: int, weights: torch.Tensor, max_mean: int, max_var: int,
                       inflate_full: bool, device, sub_dims=None, percentile=0.80, delta=0.05):
    # Generate means between -max_mean and max_mean
    means = torch.rand(num_cluster, dim, device=device) * \
            torch.randint(low=-max_mean, high=max_mean+1, size=(num_cluster, dim, ), device=device)

    # Generate diagonal covariance matrices with positive entries between 1 and max_var
    diag_values = torch.rand(num_cluster, dim, device=device) * \
                  torch.randint(low=1, high=max_var+1, size=(num_cluster, dim, ), device=device)
    diag_values[diag_values == 0] = max_var / 2

    # Create batch of diagonal covariance matrices
    covariances = torch.diag_embed(diag_values)  # Shape: (num_cluster, dim, dim)

    N_d_M_cluster_gaussian = GaussianMixtureModel(
        means=means,
        covariances=covariances,
        weights=weights,
        inflate_full=inflate_full,
        sub_dims=sub_dims,
        percentile=percentile,
        delta=delta,
        device=device
    )
    return N_d_M_cluster_gaussian


def generate_constrained_eigenvals(d):
    # Generate uniformly distributed values in the range (-0.8, -0.2)
    low_range = np.random.uniform(-1.0, -0.1, size=d)

    # Generate uniformly distributed values in the range (0.2, 0.8)
    high_range = np.random.uniform(0.1, 1.0, size=d)

    # Randomly choose between the two ranges for each element
    choice = np.random.choice([0, 1], size=d)
    vector = np.where(choice == 0, low_range, high_range)

    return vector


def generate_full_rank_matrix(dim, device, scale=1):
    # Generate a random orthogonal matrix using QR decomposition
    A = np.random.rand(dim, dim)
    Q, _ = np.linalg.qr(A)

    eigenvals = generate_constrained_eigenvals(d=dim)
    eigenvals = np.diag(eigenvals)

    full_rank_matrix = Q @ eigenvals @ Q.T
    assert np.linalg.matrix_rank(full_rank_matrix) == dim
    if device is None:  # source is numpy
        return full_rank_matrix
    else:
        return torch.from_numpy(full_rank_matrix).to(dtype=torch.float, device=device)


def generate_linear_transform(dim, device, A_scale=1, b_scale=1):
    A = generate_full_rank_matrix(dim=dim, device=device, scale=A_scale)
    b = np.random.rand(dim) * np.random.randint(low=-b_scale, high=b_scale + 1, size=dim)  # [low, high)

    if device is not None:  # source is torch, transfer from numpy to torch
        b = torch.from_numpy(b).to(dtype=torch.float, device=device)
    return A, b


def transform_means(means, sub_dims, A, b):
    trans = []
    for mean in means:
        new_mean = mean.clone()
        new_mean[sub_dims] = A @ new_mean[sub_dims] + b
        trans.append(new_mean)
    return torch.stack(trans)


def transform_covs(covs, sub_dims, A):
    trans = []
    for cov in covs:
        new_cov = cov.clone()
        new_cov[sub_dims[:, None], sub_dims] = A @ new_cov[sub_dims[:, None], sub_dims] @ A.T
        trans.append(new_cov)
    return torch.stack(trans)


def transform_samples(samples, sub_dims, A, b, is_source_numpy=False):
    if is_source_numpy:
        new_samples = samples.copy()
    else:
        new_samples = samples.clone()

    if sub_dims is None:
        new_samples = new_samples @ A.T + b
    else:
        new_samples[:, sub_dims] = new_samples[:, sub_dims] @ A.T + b

    return new_samples

