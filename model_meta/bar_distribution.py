"""
Piecewise-continuous bar distributions for use as training criteria.

Ported from pfns/bar_distribution.py (originally from the
"Transformers Can Do Bayesian Inference" paper, aka PFNs).

Classes
-------
BarDistribution              – piecewise-uniform distribution over a finite range
FullSupportBarDistribution   – extends BarDistribution with half-normal tails so
                               the support covers (-inf, inf)

Utility functions
-----------------
get_bucket_limits    – compute bucket borders from data or a fixed range
get_custom_bar_dist  – reparameterise an existing BarDistribution's borders
"""

import torch
from torch import nn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_printed_already: set = set()


def _print_once(*msgs: str) -> None:
    """Print a message at most once per process lifetime."""
    msg = ' '.join([repr(m) for m in msgs])
    if msg not in _printed_already:
        print(msg)
        _printed_already.add(msg)


# ---------------------------------------------------------------------------
# BarDistribution
# ---------------------------------------------------------------------------

class BarDistribution(nn.Module):
    """
    Piecewise-uniform (bar) distribution over a closed interval.

    Each bar has uniform density; the model predicts unnormalised logits over
    bars and the log-likelihood is the log of the bar density at the target.

    Parameters
    ----------
    borders : Tensor[num_bars + 1]
        Sorted border positions; must start at the minimum and end at the
        maximum of the support.
    smoothing : float
        Label-smoothing coefficient (applied only during training).
    ignore_nan_targets : bool
        If True, positions with NaN targets contribute zero loss.
    """

    def __init__(self, borders: torch.Tensor, smoothing: float = 0.0,
                 ignore_nan_targets: bool = True):
        super().__init__()
        assert len(borders.shape) == 1
        self.register_buffer('borders', borders)
        self.register_buffer('smoothing', torch.tensor(smoothing))
        self.register_buffer('bucket_widths', borders[1:] - borders[:-1])

        full_width = self.bucket_widths.sum()
        assert (1 - (full_width / (borders[-1] - borders[0]))).abs() < 1e-2, (
            f'Bucket widths do not sum to the full range: '
            f'{full_width} vs {borders[-1] - borders[0]}'
        )
        assert (self.bucket_widths >= 0.0).all(), "borders must be sorted (non-decreasing)."

        self.num_bars = len(borders) - 1
        self.ignore_nan_targets = ignore_nan_targets
        self.to(borders.device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def __setstate__(self, state):
        super().__setstate__(state)
        self.__dict__.setdefault('append_mean_pred', False)

    def map_to_bucket_idx(self, y: torch.Tensor) -> torch.Tensor:
        """Map target values to their bucket indices."""
        target_sample = torch.searchsorted(self.borders, y) - 1
        target_sample[y == self.borders[0]] = 0
        target_sample[y == self.borders[-1]] = self.num_bars - 1
        return target_sample

    def ignore_init(self, y: torch.Tensor) -> torch.Tensor:
        """
        Return a boolean mask of NaN positions and replace NaNs with a
        valid default so downstream ops don't crash.
        """
        ignore_loss_mask = torch.isnan(y)
        if ignore_loss_mask.any():
            if not self.ignore_nan_targets:
                raise ValueError(f'Found NaN in target {y}')
            _print_once("A loss was ignored because there was nan target.")
        y[ignore_loss_mask] = self.borders[0]
        return ignore_loss_mask

    def compute_scaled_log_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """log p(y) for piecewise-uniform density: log softmax – log bucket_width."""
        bucket_log_probs = torch.log_softmax(logits, -1)
        return bucket_log_probs - torch.log(self.bucket_widths)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def forward(self, logits: torch.Tensor, y: torch.Tensor,
                mean_prediction_logits: torch.Tensor = None) -> torch.Tensor:
        """
        Negative log-density loss.

        Parameters
        ----------
        logits : Tensor[T, B, num_bars]
        y : Tensor[T, B]  (or broadcastable)
        mean_prediction_logits : optional Tensor for nonmyopic BO mean loss

        Returns
        -------
        loss : Tensor[T, B]
        """
        y = y.clone().view(*logits.shape[:-1])
        ignore_loss_mask = self.ignore_init(y)
        target_sample = self.map_to_bucket_idx(y)

        assert (target_sample >= 0).all() and (target_sample < self.num_bars).all(), (
            f'y {y} not in support borders (min, max) = {self.borders[[0, -1]]}'
        )
        assert logits.shape[-1] == self.num_bars

        scaled_bucket_log_probs = self.compute_scaled_log_probs(logits)
        nll_loss = -scaled_bucket_log_probs.gather(
            -1, target_sample[..., None]
        ).squeeze(-1)  # T x B

        if mean_prediction_logits is not None:
            if not self.training:
                print('Calculating loss incl mean prediction loss for nonmyopic BO.')
            scaled_mean_log_probs = self.compute_scaled_log_probs(mean_prediction_logits)
            nll_loss = torch.cat((nll_loss, self.mean_loss(logits, scaled_mean_log_probs)), 0)

        smooth_loss = -scaled_bucket_log_probs.mean(dim=-1)
        smoothing = self.smoothing if self.training else 0.0
        loss = (1.0 - smoothing) * nll_loss + smoothing * smooth_loss
        loss[ignore_loss_mask] = 0.0
        return loss

    def mean_loss(self, logits: torch.Tensor,
                  scaled_mean_logits: torch.Tensor) -> torch.Tensor:
        assert (len(logits.shape) == 3) and (len(scaled_mean_logits.shape) == 2)
        means = self.mean(logits).detach()           # T x B
        target_mean = self.map_to_bucket_idx(means).clamp_(0, self.num_bars - 1)
        return -scaled_mean_logits.gather(1, target_mean.T).mean(1).unsqueeze(0)

    # ------------------------------------------------------------------
    # Distribution statistics
    # ------------------------------------------------------------------

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        bucket_means = self.borders[:-1] + self.bucket_widths / 2
        p = torch.softmax(logits, -1)
        return p @ bucket_means

    def median(self, logits: torch.Tensor) -> torch.Tensor:
        return self.icdf(logits, 0.5)

    def icdf(self, logits: torch.Tensor, left_prob: float) -> torch.Tensor:
        """Quantile function: position with *left_prob* probability mass to the left."""
        probs = logits.softmax(-1)
        cumprobs = torch.cumsum(probs, -1)
        idx = torch.searchsorted(
            cumprobs,
            left_prob * torch.ones(*cumprobs.shape[:-1], 1, device=logits.device),
        ).squeeze(-1).clamp(0, cumprobs.shape[-1] - 1)
        cumprobs = torch.cat(
            [torch.zeros(*cumprobs.shape[:-1], 1, device=logits.device), cumprobs], -1
        )
        rest_prob = left_prob - cumprobs.gather(-1, idx[..., None]).squeeze(-1)
        left_border = self.borders[idx]
        right_border = self.borders[idx + 1]
        return left_border + (right_border - left_border) * rest_prob / probs.gather(
            -1, idx[..., None]
        ).squeeze(-1)

    def quantile(self, logits: torch.Tensor, center_prob: float = 0.682):
        side_probs = (1.0 - center_prob) / 2
        return torch.stack(
            (self.icdf(logits, side_probs), self.icdf(logits, 1.0 - side_probs)), -1
        )

    def mode(self, logits: torch.Tensor) -> torch.Tensor:
        density = logits.softmax(-1) / self.bucket_widths
        mode_inds = density.argmax(-1)
        bucket_means = self.borders[:-1] + self.bucket_widths / 2
        return bucket_means[mode_inds]

    def mean_of_square(self, logits: torch.Tensor) -> torch.Tensor:
        """E[x²] under the bar distribution."""
        left_borders = self.borders[:-1]
        right_borders = self.borders[1:]
        bucket_mean_of_square = (
            left_borders.square() + right_borders.square() + left_borders * right_borders
        ) / 3.0
        p = torch.softmax(logits, -1)
        return p @ bucket_mean_of_square

    def variance(self, logits: torch.Tensor) -> torch.Tensor:
        return self.mean_of_square(logits) - self.mean(logits).square()

    # ------------------------------------------------------------------
    # Acquisition functions (Bayesian optimisation)
    # ------------------------------------------------------------------

    def ucb(self, logits: torch.Tensor, best_f, rest_prob: float = (1 - 0.682) / 2,
            maximize: bool = True) -> torch.Tensor:
        """Upper Confidence Bound acquisition."""
        if maximize:
            rest_prob = 1.0 - rest_prob
        return self.icdf(logits, rest_prob)

    def ei(self, logits: torch.Tensor, best_f, maximize: bool = True) -> torch.Tensor:
        """Expected Improvement acquisition."""
        bucket_diffs = self.borders[1:] - self.borders[:-1]
        assert maximize
        if not torch.is_tensor(best_f) or not len(best_f.shape):
            best_f = torch.full(logits[..., 0].shape, best_f, device=logits.device)

        best_f_expanded = best_f[..., None].repeat(*[1] * len(best_f.shape), logits.shape[-1])
        clamped_best_f = best_f_expanded.clamp(self.borders[:-1], self.borders[1:])
        bucket_contributions = (
            (self.borders[1:] ** 2 - clamped_best_f ** 2) / 2
            - best_f_expanded * (self.borders[1:] - clamped_best_f)
        ) / bucket_diffs

        p = torch.softmax(logits, -1)
        return torch.einsum("...b,...b->...", p, bucket_contributions)

    def pi(self, logits: torch.Tensor, best_f, maximize: bool = True) -> torch.Tensor:
        """Probability of Improvement acquisition."""
        assert maximize is True
        if not torch.is_tensor(best_f) or not len(best_f.shape):
            best_f = torch.full(logits[..., 0].shape, best_f, device=logits.device)
        p = torch.softmax(logits, -1)
        border_widths = self.borders[1:] - self.borders[:-1]
        factor = 1.0 - ((best_f[..., None] - self.borders[:-1]) / border_widths).clamp(0.0, 1.0)
        return (p * factor).sum(-1)


# ---------------------------------------------------------------------------
# FullSupportBarDistribution
# ---------------------------------------------------------------------------

class FullSupportBarDistribution(BarDistribution):
    """
    BarDistribution extended with half-normal tails on both ends, giving
    support over the full real line (-inf, inf).
    """

    @staticmethod
    def halfnormal_with_p_weight_before(range_max, p: float = 0.5):
        """Half-normal with CDF(range_max) = p."""
        s = range_max / torch.distributions.HalfNormal(torch.tensor(1.0)).icdf(
            torch.tensor(p)
        )
        return torch.distributions.HalfNormal(s)

    def forward(self, logits: torch.Tensor, y: torch.Tensor,
                mean_prediction_logits: torch.Tensor = None) -> torch.Tensor:
        assert self.num_bars > 1
        y = y.clone().view(len(y), -1)
        ignore_loss_mask = self.ignore_init(y)
        target_sample = self.map_to_bucket_idx(y)
        target_sample.clamp_(0, self.num_bars - 1)

        assert logits.shape[-1] == self.num_bars

        scaled_bucket_log_probs = self.compute_scaled_log_probs(logits)
        assert len(scaled_bucket_log_probs) == len(target_sample)

        log_probs = scaled_bucket_log_probs.gather(
            -1, target_sample.unsqueeze(-1)
        ).squeeze(-1)

        side_normals = (
            self.halfnormal_with_p_weight_before(self.bucket_widths[0]),
            self.halfnormal_with_p_weight_before(self.bucket_widths[-1]),
        )

        # Left tail
        mask_left = target_sample == 0
        log_probs[mask_left] += (
            side_normals[0].log_prob(
                (self.borders[1] - y[mask_left]).clamp(min=1e-7)
            )
            + torch.log(self.bucket_widths[0])
        )
        # Right tail
        mask_right = target_sample == self.num_bars - 1
        log_probs[mask_right] += (
            side_normals[1].log_prob(
                (y[mask_right] - self.borders[-2]).clamp(min=1e-7)
            )
            + torch.log(self.bucket_widths[-1])
        )

        nll_loss = -log_probs

        if mean_prediction_logits is not None:
            assert not ignore_loss_mask.any(), \
                "Ignoring examples is not implemented with mean_prediction_logits."
            if not self.training:
                print('Calculating loss incl mean prediction loss for nonmyopic BO.')
            scaled_mean_log_probs = self.compute_scaled_log_probs(mean_prediction_logits)
            nll_loss = torch.cat(
                (nll_loss, self.mean_loss(logits, scaled_mean_log_probs)), 0
            )

        if self.smoothing:
            smooth_loss = -scaled_bucket_log_probs.mean(dim=-1)
            smoothing = self.smoothing if self.training else 0.0
            nll_loss = (1.0 - smoothing) * nll_loss + smoothing * smooth_loss

        if ignore_loss_mask.any():
            nll_loss[ignore_loss_mask] = 0.0

        return nll_loss

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        bucket_means = self.borders[:-1] + self.bucket_widths / 2
        p = torch.softmax(logits, -1)
        side_normals = (
            self.halfnormal_with_p_weight_before(self.bucket_widths[0]),
            self.halfnormal_with_p_weight_before(self.bucket_widths[-1]),
        )
        bucket_means[0] = -side_normals[0].mean + self.borders[1]
        bucket_means[-1] = side_normals[1].mean + self.borders[-2]
        return p @ bucket_means.to(logits.device)

    def mean_of_square(self, logits: torch.Tensor) -> torch.Tensor:
        left_borders = self.borders[:-1]
        right_borders = self.borders[1:]
        bucket_mean_of_square = (
            left_borders.square() + right_borders.square() + left_borders * right_borders
        ) / 3.0
        side_normals = (
            self.halfnormal_with_p_weight_before(self.bucket_widths[0]),
            self.halfnormal_with_p_weight_before(self.bucket_widths[-1]),
        )
        bucket_mean_of_square[0] = side_normals[0].variance + (
            -side_normals[0].mean + self.borders[1]
        ).square()
        bucket_mean_of_square[-1] = side_normals[1].variance + (
            side_normals[1].variance + self.borders[-2]
        ).square()
        p = torch.softmax(logits, -1)
        return p @ bucket_mean_of_square

    def pi(self, logits: torch.Tensor, best_f, maximize: bool = True) -> torch.Tensor:
        assert maximize is True
        if not torch.is_tensor(best_f) or not len(best_f.shape):
            best_f = torch.full(logits[..., 0].shape, best_f, device=logits.device)
        assert best_f.shape == logits[..., 0].shape

        p = torch.softmax(logits, -1)
        border_widths = self.borders[1:] - self.borders[:-1]
        factor = 1.0 - (
            (best_f[..., None] - self.borders[:-1]) / border_widths
        ).clamp(0.0, 1.0)

        side_normals = (
            self.halfnormal_with_p_weight_before(self.bucket_widths[0]),
            self.halfnormal_with_p_weight_before(self.bucket_widths[-1]),
        )
        pos_left = -(best_f - self.borders[1]).clamp(max=0.0)
        pos_right = (best_f - self.borders[-2]).clamp(min=0.0)

        factor[..., 0] = 0.0
        factor[..., 0][pos_left > 0.0] = side_normals[0].cdf(pos_left[pos_left > 0.0])
        factor[..., -1] = 1.0
        factor[..., -1][pos_right > 0.0] = 1.0 - side_normals[1].cdf(pos_right[pos_right > 0.0])
        return (p * factor).sum(-1)

    def ei_for_halfnormal(self, scale, best_f, maximize: bool = True) -> torch.Tensor:
        """EI for a HalfNormal(scale) distribution with mean 0."""
        assert maximize
        mean = torch.tensor(0.0)
        u = (mean - best_f) / scale
        normal = torch.distributions.Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))
        return 2 * scale * (updf + u * ucdf)

    def ei(self, logits: torch.Tensor, best_f, maximize: bool = True) -> torch.Tensor:
        if torch.isnan(logits).any():
            raise ValueError(f"logits contains NaNs: {logits}")
        bucket_diffs = self.borders[1:] - self.borders[:-1]
        assert maximize
        if not torch.is_tensor(best_f) or not len(best_f.shape):
            best_f = torch.full(logits[..., 0].shape, best_f, device=logits.device)
        assert best_f.shape == logits[..., 0].shape

        best_f_expanded = best_f[..., None].repeat(*[1] * len(best_f.shape), logits.shape[-1])
        clamped_best_f = best_f_expanded.clamp(self.borders[:-1], self.borders[1:])
        bucket_contributions = (
            (self.borders[1:] ** 2 - clamped_best_f ** 2) / 2
            - best_f_expanded * (self.borders[1:] - clamped_best_f)
        ) / bucket_diffs

        side_normals = (
            self.halfnormal_with_p_weight_before(self.bucket_widths[0]),
            self.halfnormal_with_p_weight_before(self.bucket_widths[-1]),
        )
        pos_left = -(best_f - self.borders[1]).clamp(max=0.0)
        pos_right = (best_f - self.borders[-2]).clamp(min=0.0)

        bucket_contributions[..., -1] = self.ei_for_halfnormal(
            side_normals[1].scale, pos_right
        )
        bucket_contributions[..., 0] = self.ei_for_halfnormal(
            side_normals[0].scale, torch.zeros_like(pos_left)
        ) - self.ei_for_halfnormal(side_normals[0].scale, pos_left)

        p = torch.softmax(logits, -1)
        return torch.einsum("...b,...b->...", p, bucket_contributions)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_bucket_limits(num_outputs: int, full_range: tuple = None,
                      ys: torch.Tensor = None, verbose: bool = False) -> torch.Tensor:
    """
    Compute bucket border positions.

    Exactly one of *full_range* or *ys* must be provided.

    Parameters
    ----------
    num_outputs : int  – number of bars (= number of buckets)
    full_range  : (min, max) tuple – use equally-spaced borders
    ys          : Tensor – use empirical quantiles of the data
    verbose     : bool
    """
    assert (ys is None) != (full_range is None), \
        'Exactly one of full_range or ys must be passed.'

    if ys is not None:
        ys = ys.flatten()
        ys = ys[~torch.isnan(ys)]
        if len(ys) % num_outputs:
            ys = ys[:-(len(ys) % num_outputs)]
        ys_per_bucket = len(ys) // num_outputs
        if full_range is None:
            full_range = (ys.min(), ys.max())
        else:
            assert full_range[0] <= ys.min() and full_range[1] >= ys.max()
            full_range = torch.tensor(full_range)
        ys_sorted, _ = ys.sort(0)
        bucket_limits = (
            ys_sorted[ys_per_bucket - 1::ys_per_bucket][:-1]
            + ys_sorted[ys_per_bucket::ys_per_bucket]
        ) / 2
        if verbose:
            print(f'Using {len(ys)} y evals to estimate {num_outputs} buckets.')
            print(full_range)
        bucket_limits = torch.cat(
            [full_range[0].unsqueeze(0), bucket_limits, full_range[1].unsqueeze(0)], 0
        )
    else:
        class_width = (full_range[1] - full_range[0]) / num_outputs
        bucket_limits = torch.cat(
            [
                full_range[0] + torch.arange(num_outputs).float() * class_width,
                torch.tensor(full_range[1]).unsqueeze(0),
            ],
            0,
        )

    assert len(bucket_limits) - 1 == num_outputs
    assert full_range[0] == bucket_limits[0]
    assert full_range[-1] == bucket_limits[-1]
    return bucket_limits


def get_custom_bar_dist(borders: torch.Tensor,
                        criterion: BarDistribution) -> BarDistribution:
    """
    Return a new BarDistribution whose borders are a softplus-reparameterisation
    of *borders* scaled to match the width structure of *criterion*.
    """
    borders_ = torch.nn.functional.softplus(borders) + 0.001
    borders_ = torch.cumsum(
        torch.cat([criterion.borders[0:1], criterion.bucket_widths]) * borders_, 0
    )
    return criterion.__class__(borders=borders_, handle_nans=criterion.handle_nans)
