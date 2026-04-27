"""Feature preprocessing utilities for FoMo-Meta.

This module keeps the public API from the previous implementation but reduces
duplication and adds clearer documentation.

Main pipeline used by `pfn_inference_transform*`:
1) optional feature subsampling (if input dim > max allowed dim)
2) z-score normalization using PFN utility (`normalize_data`)
3) optional per-column sklearn transform (`power` / `quantile` / `robust`)
4) feature scaling based on used-feature ratio
5) right padding to the model input dimension
"""

from __future__ import annotations

from typing import Optional
import warnings

import numpy as np
import torch
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler


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



class FeatureTransform:
    """Feature-level transforms used before FoMo model inference/training."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.prior_gmm_cfg = cfg.prior.mixture.gmm
        self.max_feature_dim = self.prior_gmm_cfg.max_feature_dim

    # ------------------------------------------------------------------
    # Basic transforms
    # ------------------------------------------------------------------
    def feature_scale(self, x, num_feature: int, rescale_with_sqrt: bool = False):
        """Scale by used-feature ratio to keep magnitude consistent across dims."""
        scale = (num_feature / self.max_feature_dim) ** (0.5 if rescale_with_sqrt else 1.0)
        return x / scale


    def feature_padding(
        self,
        x: np.ndarray,
        num_feature: int,
        internal_features: Optional[np.ndarray] = None,
        max_num_internal_features: Optional[int] = None,
    ) -> np.ndarray:
        """Pad numpy feature matrix to expected model width.

        If `internal_features` is provided, output width becomes:
        `max_feature_dim + max_num_internal_features`.
        """
        num_internal_features = internal_features.shape[1] if internal_features is not None else 0
        if max_num_internal_features is None:
            max_num_internal_features = num_internal_features

        # Avoid scaling internal features; scale only base features.
        if internal_features is not None:
            x = x[:, num_internal_features:]

        x = self.feature_scale(x=x, num_feature=num_feature)

        if internal_features is not None:
            target_dim = self.max_feature_dim + max_num_internal_features
            pad_dim = max(0, target_dim - (x.shape[1] + num_internal_features))
            x = np.concatenate([internal_features, x, np.zeros((x.shape[0], pad_dim))], axis=-1)
        else:
            pad_dim = max(0, self.max_feature_dim - num_feature)
            x = np.concatenate([x, np.zeros((x.shape[0], pad_dim))], axis=-1)
        return x


    def feature_padding_torch(
        self,
        x: torch.Tensor,
        num_feature: int,
        internal_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Torch version of `feature_padding`."""
        num_internal_features = internal_features.shape[1] if internal_features is not None else 0

        if internal_features is not None:
            x = x[:, num_internal_features:]

        x = self.feature_scale(x=x, num_feature=num_feature)

        if internal_features is not None:
            pad_dim = max(0, self.max_feature_dim + num_internal_features - num_feature)
            x = torch.cat(
                [internal_features, x, torch.zeros(x.shape[0], pad_dim, device=x.device, dtype=x.dtype)],
                dim=-1,
            )
        else:
            pad_dim = max(0, self.max_feature_dim - num_feature)
            x = torch.cat([x, torch.zeros(x.shape[0], pad_dim, device=x.device, dtype=x.dtype)], dim=-1)
        return x


    def feature_subsampling(self, x: np.ndarray, num_feature: int) -> np.ndarray:
        """Randomly choose `max_feature_dim` columns without replacement."""
        indices = sorted(np.random.choice(num_feature, self.max_feature_dim, replace=False))
        return x[:, indices]


    def feature_sparse_projection(self, x: np.ndarray, num_feature: int) -> np.ndarray:
        """Project high-dimensional features using sparse random projections."""
        scale = np.sqrt(3 / self.max_feature_dim)
        projection = np.array(
            [self.generate_1d_projection(num_feature=num_feature, scale=scale)
             for _ in range(self.max_feature_dim)]
        )
        return x @ projection.T


    @staticmethod
    def generate_1d_projection(num_feature: int, scale: float = 1.0) -> np.ndarray:
        """Generate one sparse random projection vector with {0, +scale, -scale}."""
        num_zeros = int(num_feature * 2 / 3)
        num_ones = int(num_feature / 6)
        num_neg_ones = num_feature - num_zeros - num_ones

        array = np.array([0] * num_zeros + [1] * num_ones + [-1] * num_neg_ones) * scale
        np.random.shuffle(array)
        return array


    # ------------------------------------------------------------------
    # PFN-style inference preprocessing
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_eval_xs(eval_xs: np.ndarray) -> None:
        if eval_xs.ndim != 2:
            raise ValueError(
                f"Transforms only allow input shape (#sampled, #feat), got {eval_xs.shape}"
            )

    @staticmethod
    def _build_preprocessor(preprocess_transform: str):
        if preprocess_transform in ('none', None):
            return None
        if preprocess_transform in ('power', 'power_all'):
            return PowerTransformer(standardize=True)
        if preprocess_transform in ('quantile', 'quantile_all'):
            return QuantileTransformer(output_distribution='normal')
        if preprocess_transform in ('robust', 'robust_all'):
            return RobustScaler(unit_variance=True)
        raise ValueError(f"Unknown preprocess_transform='{preprocess_transform}'")

    @staticmethod
    def _apply_columnwise_transform(
        eval_xs: np.ndarray,
        eval_position: int,
        preprocessor,
        preprocess_transform: str,
    ) -> np.ndarray:
        if preprocessor is None:
            return eval_xs

        print(f'feature preprocessing transform with {preprocess_transform}')
        warnings.simplefilter('error')
        try:
            for col in range(eval_xs.shape[1]):
                try:
                    preprocessor.fit(eval_xs[0:eval_position, col:col + 1])
                    eval_xs[:, col:col + 1] = preprocessor.transform(eval_xs[:, col:col + 1])
                except Exception:
                    # Keep legacy behavior: skip problematic columns silently.
                    pass
        finally:
            warnings.simplefilter('default')
        return eval_xs


    def _base_inference_transform(
        self,
        eval_xs: np.ndarray,
        preprocess_transform: str,
        eval_position: int,
        normalize_with_test: bool,
    ) -> np.ndarray:
        """Shared preprocessing before final scaling/padding."""
        self._validate_eval_xs(eval_xs)

        num_feature = eval_xs.shape[-1]
        if num_feature > self.max_feature_dim:
            eval_xs = self.feature_subsampling(x=eval_xs, num_feature=num_feature)

        # z-score normalization with PFN utility
        eval_xs_t = torch.from_numpy(eval_xs)
        eval_xs_t = normalize_data(
            eval_xs_t,
            normalize_positions=-1 if normalize_with_test else eval_position,
        )
        eval_xs = eval_xs_t.cpu().numpy()

        preprocessor = self._build_preprocessor(preprocess_transform)
        eval_xs = self._apply_columnwise_transform(
            eval_xs=eval_xs,
            eval_position=eval_position,
            preprocessor=preprocessor,
            preprocess_transform=preprocess_transform,
        )
        return eval_xs


    def pfn_inference_transform(
        self,
        eval_xs: np.ndarray,
        preprocess_transform: str,
        eval_position: int,
        normalize_with_test: bool = False,
        rescale_with_sqrt: bool = False,
    ) -> np.ndarray:
        """PFN-style preprocessing without internal summary features."""
        eval_xs = self._base_inference_transform(
            eval_xs=eval_xs,
            preprocess_transform=preprocess_transform,
            eval_position=eval_position,
            normalize_with_test=normalize_with_test,
        )

        eval_xs = self.feature_scale(
            x=eval_xs,
            num_feature=eval_xs.shape[-1],
            rescale_with_sqrt=rescale_with_sqrt,
        )

        if eval_xs.shape[-1] < self.max_feature_dim:
            pad_dim = self.max_feature_dim - eval_xs.shape[-1]
            eval_xs = np.concatenate([eval_xs, np.zeros((eval_xs.shape[0], pad_dim))], axis=-1)
        return eval_xs