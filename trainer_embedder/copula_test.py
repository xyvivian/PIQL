import random, time, math, numpy as np
from contextlib import contextmanager

import torch
from torch.distributions import Normal, Beta, Exponential, StudentT
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics  import roc_auc_score
import scipy
from scipy.stats import beta as scipy_beta
import torch, math, multiprocessing as mp, numpy as np
import scipy.stats as sps
from torch.distributions import Normal
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import betainc, betaincinv
#from trainer_embedder.embedder import describe_prior_program
from embedder import describe_prior_program

import time
import torch

# Make sure to sync for accurate GPU timings
def tic():
    torch.cuda.synchronize()
    return time.time()

def toc(start, label=""):
    torch.cuda.synchronize()
    end = time.time()
    print(f"{label}: {end - start:.6f} seconds")

# ───────────────────────── helper: icdf() calculation on torch ──────────────────────────
def _neg_log1p(u):
    # stable for u≈0
    return (-torch.log1p(-u)).clamp_min(0.0)


def exp_icdf(u, rate):
    """
    u     : (...,)  in (0,1)
    rate  : broadcastable with u   (λ > 0)
    """
    return _neg_log1p(u) / rate


def beta_icdf(p, a, b, n_grid: int = 1000):
    p = torch.as_tensor(p, dtype=torch.float32)
    a = torch.as_tensor(a, dtype=torch.float32)
    b = torch.as_tensor(b, dtype=torch.float32)
    if a.size() == torch.Size([]):
      a = a.expand(1)
    if b.size() == torch.Size([]):
      b = b.expand(1)
    x = torch.linspace(0.0, 1.0, n_grid + 1, dtype=torch.float32, device=p.device)
    mid = (x[:-1] + x[1:]) / 2.0
    mid = mid.view(-1, 1)  # shape [n_grid, 1]
    dx = 1.0 / n_grid

    log_norm = torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
    pdf_mid = torch.exp((a - 1) * torch.log(mid) +
                        (b - 1) * torch.log1p(-mid) -
                        log_norm)  # [n_grid, D]
    cdf = torch.cumsum(pdf_mid, dim=0) * dx  # [n_grid, D]
    cdf = torch.cat([torch.zeros(1, cdf.shape[1], device=p.device), cdf], dim=0)  # [n_grid+1, D]
    # Expand p to match shape [N, D]
    if p.ndim == 1:
        p = p.unsqueeze(1)
    if a.ndim == 1:
        a = a.unsqueeze(0)
        b = b.unsqueeze(0)
    if p.shape[1] != a.shape[1]:
        p = p.expand(-1, a.shape[1])
    # Searchsorted per-dimension
    N, D = p.shape
    idx = torch.empty((N, D), dtype=torch.long, device=p.device)
    for d in range(D):
        idx[:, d] = torch.searchsorted(cdf[:, d], p[:, d], right=False).clamp(1, n_grid)
    # Gather x and y for interpolation
    x0 = x[idx - 1]
    x1 = x[idx]
    y0 = torch.empty_like(p)
    y1 = torch.empty_like(p)
    for d in range(D):
        y0[:, d] = cdf[idx[:, d] - 1, d]
        y1[:, d] = cdf[idx[:, d], d]

    t = (p - y0) / (y1 - y0 + 1e-12)
    return torch.clamp(x0 + t * (x1 - x0), 0., 1.)


def student_t_icdf(u, df):
    u = torch.as_tensor(u, dtype=torch.float32)
    df = torch.as_tensor(df, dtype=torch.float32)

    p = 2.0 * torch.minimum(u, 1 - u)
    p = p.to(df.device)
    x = beta_icdf(p, 0.5 * df, torch.tensor(0.5,device = p.device))
    t = torch.sqrt(df * (1.0 / x - 1.0))
    return torch.where(u > 0.5, t, -t)


def normal_cdf(x):  
    return 0.5*(1+torch.special.erf(x/math.sqrt(2)))


def normal_ppf(u):  
    return Normal(0,1).icdf(u)


# ─────────────── low-level helpers ───────────────
def rand_corr_batch(batch, d, identity=False,device='cuda'):
    """(B,d,d) stack of correlation matrices on device/torch.float32"""
    if identity:
        eye = torch.eye(d, device=device, dtype=torch.float32)
        return eye.expand(batch, -1, -1).clone()

    A   = torch.rand(batch, d, d, device=device, dtype=torch.float32)
    psd = A @ A.transpose(-1, -2)
    diag= torch.diagonal(psd, dim1=-2, dim2=-1)
    norm= torch.sqrt(torch.clamp(diag, 1e-12))
    C   = psd / (norm.unsqueeze(-1)*norm.unsqueeze(-2))
    C.diagonal(dim1=-2, dim2=-1).fill_(1.)
    C += torch.eye(d, device=device, dtype=torch.float32).unsqueeze(0)*1e-6
    return C


def mvn_sample(chol,device):       # chol: (B,d,d) or (d,d)
    if chol.dim() == 2: chol = chol.unsqueeze(0)
    B,d = chol.shape[:2]
    z   = torch.randn(B, d, 1, device=device, dtype=torch.float32)
    return (chol @ z).squeeze(-1)           # (B,d)


# ─────────────── mixture helpers ───────────────
class GaussianMix:
    def __init__(self, comps, device='cpu'):
        w, mu, sigma = zip(*comps)
        self.device = device
        self.w = torch.tensor(w, device=device, dtype=torch.float32)
        self.mu = torch.tensor(mu, device=device, dtype=torch.float32)
        self.sigma = torch.tensor(sigma, device=device, dtype=torch.float32)
        self.w /= self.w.sum()

    def cdf(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        z = (x.unsqueeze(-1) - self.mu) / (self.sigma * math.sqrt(2))
        return (self.w * (0.5 * (1 + torch.erf(z)))).sum(-1)

    def ppf_bounds(self):
        lo = (self.mu - 5 * self.sigma).min().item()
        hi = (self.mu + 5 * self.sigma).max().item()
        return lo, hi

    def ppf(self, u, tol=1e-5, max_iter=100):
        """
        Numerical inverse CDF using bisection.
        u: tensor of shape (...)
        Returns: tensor of same shape
        """
        u = torch.as_tensor(u, device=self.device, dtype=torch.float32)
        low, high = self.ppf_bounds()
        low = torch.full_like(u, low)
        high = torch.full_like(u, high)

        for _ in range(max_iter):
            mid = 0.5 * (low + high)
            cdf_mid = self.cdf(mid)
            low = torch.where(cdf_mid < u, mid, low)
            high = torch.where(cdf_mid >= u, mid, high)
            if torch.max(high - low) < tol:
                break
        return 0.5 * (low + high)


class BetaMix:
    def __init__(self, comps,device='cuda'):
        self.comps = comps
        self.w = torch.tensor([c[0] for c in comps], device=device, dtype=torch.float32)
        self.a = torch.tensor([c[1] for c in comps], device=device, dtype=torch.float32)
        self.b = torch.tensor([c[2] for c in comps], device=device, dtype=torch.float32)
        self.loc = torch.tensor([c[3] for c in comps], device=device, dtype=torch.float32)
        self.sc = torch.tensor([c[4] for c in comps], device=device, dtype=torch.float32)
        self.w /= self.w.sum()
        self.device = device

    def cdf(self, x: torch.Tensor):
        # Fallback to CPU-based computation
        x_cpu = x.detach().cpu().numpy()
        cdf_vals = np.zeros_like(x_cpu)
        for w, a, b, loc, scale in zip(self.w, self.a, self.b, self.loc, self.sc):
            dist = scipy_beta(a=a.item(), b=b.item(), loc=loc.item(), scale=scale.item())
            cdf_vals += w.item() * dist.cdf(x_cpu)
        return torch.tensor(cdf_vals, device=self.device, dtype=torch.float32)

    def ppf_bounds(self):
        lo = (self.loc).min().item()
        hi = (self.loc + self.sc).max().item()
        return lo, hi

# ─────────────── marginal catalogue ───────────────
def rand_def(device='cuda',
             PPF_GRID = 1_000):
    cat = random.choice(["normal","beta","beta_mix","expo","gauss_mix","beta_mix","student"])
    if cat=="normal":
        return dict(kind="torch", dist=Normal(torch.tensor(random.uniform(-1,1)),
                                              torch.tensor(random.uniform(1.5,2))))
    if cat=="beta":
        return dict(kind="torch", dist=Beta(torch.tensor(random.uniform(1,5)),
                                            torch.tensor(random.uniform(1,5))))
    if cat=="expo":
        return dict(kind="torch", dist=Exponential(torch.tensor(random.uniform(1,2))))
    if cat=="student":
        return dict(kind="torch", dist=StudentT(torch.tensor(random.randint(3,10))))
    if cat=="gauss_mix":
        comps = [(random.uniform(0.3,0.7),
                  random.uniform(-3,3),
                  random.uniform(0.5,1.5)) for _ in range(random.randint(1,3))]
        mix   = GaussianMix(comps)
    else:
        comps = [(random.uniform(0.3,0.7),
                  random.uniform(1,5),
                  random.uniform(1,5),
                  -5.0, 10.0) for _ in range(random.randint(1,3))]
        mix   = BetaMix(comps,device=device)
    # build interpolation grids
    u_grid = torch.linspace(0.001,0.999,PPF_GRID, device=device, dtype=torch.float32)
    lo,hi  = mix.ppf_bounds()
    x_grid = torch.linspace(lo,hi,PPF_GRID, device=device, dtype=torch.float32)
    #cdf_vals = mix.cdf(x_grid)
    return dict(kind="interp", u=u_grid, x=x_grid, lo=lo, hi=hi)


class CorpulaGenerator:
    def __init__(self, 
                 num_dims,
                 device="cuda",
                 ppf_grid=2_000,
                 anomaly_types = None,
                 num_prototypes: int = 16):
        self.num_dims = num_dims
        self.device = device
        self.dtype = torch.float32
        self.ppf_grid = ppf_grid
        self.num_prototypes = max(1, int(num_prototypes))
        self.chol_base = torch.linalg.cholesky(
            rand_corr_batch(1, num_dims, device=device)[0]
        )
        self.specs = [rand_def(device=device, PPF_GRID=ppf_grid) for _ in range(num_dims)]
        self.anomaly_types = anomaly_types

        # Prototype-based subset selection for perturb_u_values outliers
        self.perturb_u_subset_masks = None   # (P, D) bool
        self.perturb_u_subset_sizes = None   # (P,) long
        # Prototype-based contiguous blocks for disturb_covariance outliers
        self.disturb_cov_block_starts = None  # (P,) long
        self.disturb_cov_block_sizes = None   # (P,) long
        self.disturb_cov_block_ends = None    # (P,) long (exclusive)
        if self.anomaly_types == "perturb_u_values":
            self._init_perturb_u_prototypes()
        elif self.anomaly_types == "disturb_covariance":
            self._init_disturb_cov_prototypes()

    def _init_perturb_u_prototypes(self):
        min_k = max(1, math.ceil(0.02 * self.num_dims))
        max_k = max(min_k, math.floor(0.2 * self.num_dims))

        masks = []
        sizes = []
        for _ in range(self.num_prototypes):
            k = int(torch.randint(min_k, max_k + 1, (1,), device=self.device).item())
            idx = torch.randperm(self.num_dims, device=self.device)[:k]
            m = torch.zeros(self.num_dims, dtype=torch.bool, device=self.device)
            m[idx] = True
            masks.append(m)
            sizes.append(k)

        self.perturb_u_subset_masks = torch.stack(masks, dim=0)   # (P, D)
        self.perturb_u_subset_sizes = torch.tensor(sizes, device=self.device, dtype=torch.long)

    def _ensure_perturb_u_prototypes(self):
        if self.perturb_u_subset_masks is None:
            self._init_perturb_u_prototypes()

    def _init_disturb_cov_prototypes(self):
        d = self.num_dims
        lowerbound = int(1 + d // 3)
        upperbound = min(int(1 + 2 * d // 3), d + 1)
        if lowerbound == upperbound:
            upperbound += 1

        k = torch.randint(lowerbound, upperbound, (self.num_prototypes,), device=self.device)  # (P,)
        max_start = d - k
        i0 = (torch.rand(self.num_prototypes, device=self.device) * (max_start + 1)).floor().long()  # (P,)
        i1 = i0 + k

        self.disturb_cov_block_starts = i0
        self.disturb_cov_block_sizes = k
        self.disturb_cov_block_ends = i1

    def _ensure_disturb_cov_prototypes(self):
        if self.disturb_cov_block_starts is None:
            self._init_disturb_cov_prototypes()


    def sample_inliers(self, num_inliers):
        """Sample clean multivariate normal data & transform to marginals"""
        z = torch.randn(num_inliers, self.num_dims, device=self.device, dtype=self.dtype)
        samples = (z @ self.chol_base.T)
        return samples

    def sample_outliers(self,
                        num_outliers,
                        method="perturb_u_values",
                        strength=0.2):
        """Generate outliers using perturb-u or disturbed covariance"""
        if method == "perturb_u_values":
            # 1. Sample from base MVN
            z = torch.randn(num_outliers, self.num_dims, device=self.device, dtype=self.dtype)
            samples = (z @ self.chol_base.T)

            # 2. Transform to uniform
            U = normal_cdf(samples)

            # Use pre-built prototype subsets and randomly assign each sample to one prototype
            self._ensure_perturb_u_prototypes()
            proto_ids = torch.randint(0, self.num_prototypes, (num_outliers,), device=self.device)
            mask = self.perturb_u_subset_masks[proto_ids]  # (num_outliers, num_dims)

            push0 = torch.rand_like(U) < 0.5
            z_mask, o_mask = mask & push0, mask & ~push0
            noise = torch.empty_like(U)
            noise[z_mask] = strength * torch.rand(z_mask.sum(), device=self.device) * 0.5
            noise[o_mask] = 1.0 - strength * torch.rand(o_mask.sum(), device=self.device) * 0.5
            U[mask] = noise[mask]

            samples = Normal(0, 1).icdf(U)
            return samples

        elif method == "disturb_covariance":
            d, device = self.num_dims, self.device
            base_L    = self.chol_base                       
            # Assign each outlier to one pre-built contiguous block prototype
            self._ensure_disturb_cov_prototypes()
            proto_ids = torch.randint(0, self.num_prototypes, (num_outliers,), device=device)
            k = self.disturb_cov_block_sizes[proto_ids]     # (B,)
            k_max  = int(k.max())

            # Prototype start/end indices per sample
            i0 = self.disturb_cov_block_starts[proto_ids]  # (B,)

            i1 = i0 + k                                # (B,)  exclusive end index
            L_rand_full = torch.linalg.cholesky(
                rand_corr_batch(num_outliers, k_max, identity=True, device=device)          # (B,k_max,k_max)
            )
            rows  = torch.arange(k_max, device=device).view(1, k_max, 1)     # (1,k_max,1)
            cols  = torch.arange(k_max, device=device).view(1, 1, k_max)     # (1,1,k_max)
            keep  = (rows < k.view(-1, 1, 1)) & (cols < k.view(-1, 1, 1))    # (B,k_max,k_max)

            L_rand_pad = torch.zeros(num_outliers, d, d, device=device)      # (B,d,d)
            L_rand_pad[:, :k_max, :k_max][keep] = L_rand_full[keep]
            L_mix = base_L.expand(num_outliers, -1, -1).clone()

            row = torch.arange(d, device=device).view(1, d)        # (1,d)
            mask_rows = (row >= i0.view(-1, 1)) & (row < i1.view(-1, 1))  # (B,d)
            col = torch.arange(d, device=device).view(1, 1, d)     # (1,1,d)
            mask = mask_rows.unsqueeze(-1) & (col < row.unsqueeze(-1) + 1)

            L_mix = torch.where(mask,
                                (1- strength) * L_mix + strength *L_rand_pad,
                                L_mix)
            L_mix.diagonal(dim1=-2, dim2=-1).clamp_(min=1e-6)
            return mvn_sample(L_mix, device=device)  
        else:
            raise ValueError(f"Unsupported outlier injection method: {method}")

    def _transform(self, samples):
        """
        Fully GPU-based transform.
        • Normal, Beta, Exp, StudentT → vectorized GPU code
        • Mixture (interp) → fast interpolation
        """
        U = normal_cdf(samples)
        eps = 1e-6
        U = torch.clamp(U, eps, 1-eps) #make sure U does not approch infty
        N, D = U.shape
        X = torch.empty_like(U)
        # Group columns by distribution type
        interp_cols, normal_cols, beta_cols, exp_cols, student_cols = [], [], [], [], []
        for d, spec in enumerate(self.specs):
            if spec["kind"] == "interp":
                interp_cols.append(d)
            elif spec["kind"] == "torch":
                dist = spec["dist"]
                if isinstance(dist, Normal):
                    normal_cols.append(d)
                elif isinstance(dist, Beta):
                    beta_cols.append(d)
                elif isinstance(dist, Exponential):
                    exp_cols.append(d)
                elif isinstance(dist, StudentT):
                    student_cols.append(d)
        # ---------- Interp mixture sampling ----------
        for d in interp_cols:
            u = U[:, d]
            spec = self.specs[d]
            grid_min = spec["u"][0]
            grid_max = spec["u"][-1]
            n_bins = len(spec["u"]) - 1
            bin_width = (grid_max - grid_min) / n_bins
            # Clamp u to grid range
            u_clamped = torch.clamp(u, grid_min, grid_max - 1e-6)
            # Fast approximate bin index (assumes linear CDF grid)
            idx = ((u_clamped - grid_min) / bin_width).long().clamp(0, n_bins - 1)
            idx = torch.clamp(idx, 1, len(spec["u"]) - 1)
            u_lo, u_hi = spec["u"][idx - 1], spec["u"][idx]
            x_lo, x_hi = spec["x"][idx - 1], spec["x"][idx]
            X[:, d] = x_lo + (u - u_lo) * (x_hi - x_lo) / (u_hi - u_lo)
        # ---------- Normal ----------
        if normal_cols:
            loc = torch.tensor([self.specs[d]["dist"].loc for d in normal_cols],
                            device=self.device).view(1, -1)
            scale = torch.tensor([self.specs[d]["dist"].scale for d in normal_cols],
                                device=self.device).view(1, -1)
            dist = Normal(loc, scale)
            X[:, normal_cols] = dist.icdf(U[:, normal_cols])
        # ---------- Beta ----------
        if beta_cols:
            a = torch.tensor([self.specs[d]["dist"].concentration1 for d in beta_cols],
                            device=self.device).view(1, -1)
            b = torch.tensor([self.specs[d]["dist"].concentration0 for d in beta_cols],
                            device=self.device).view(1, -1)
            X[:, beta_cols] = beta_icdf(U[:, beta_cols], a, b)
        # ---------- Exponential ----------
        if exp_cols:
            rate = torch.tensor([self.specs[d]["dist"].rate for d in exp_cols],
                                device=self.device).view(1, -1)
            X[:, exp_cols] = exp_icdf(U[:, exp_cols], rate)
        # ---------- StudentT ----------
        if student_cols:
            df = torch.tensor([self.specs[d]["dist"].df for d in student_cols],
                            device=self.device).view(1, -1)
            X[:, student_cols] = student_t_icdf(U[:, student_cols], df)
        return X



    @torch.no_grad()
    def draw_batched_data(self,
                          num_inliers,
                          num_local_anomalies):
        if self.anomaly_types is not None:
            METHOD = self.anomaly_types
        else:
            METHOD = random.choice(["disturb_covariance", "perturb_u_values"])  # or add "perturb_u_values"
        STRENGTH = random.uniform(0.2,0.4) if METHOD == 'perturb_u_values' else random.uniform(0.97,0.99)
        inliers = self.sample_inliers(num_inliers)

        outliers =  self.sample_outliers(num_local_anomalies, method=METHOD, strength=STRENGTH)
        combined = torch.cat([inliers, outliers], dim=0)

        X_combined = self._transform(combined)
        X_inliers = X_combined[:num_inliers]
        X_outliers = X_combined[num_inliers:]

        return X_inliers, X_outliers

    def describe_model_specs(self, decimals=3, max_chol_len=24, max_dims_len=24):
        """Return a descriptor in the same schema style used by GMM: global fields + ENTITY/BLOCK sections."""
        chol_text, n_chol = condense_chol(self.chol_base, decimals=decimals, max_vals=max_chol_len)

        entities = [
            {
                "tag": "ENTITY",
                "fields": {
                    "TYPE": "COPULA_BASE",
                    "ID": 0,
                    "COPULA_PARAM": "random_corr_cholesky",
                    "CHOL_N": int(n_chol),
                    "CHOL": chol_text,
                },
            }
        ]

        blocks = []
        anomaly_type = getattr(self, "anomaly_types", None)
        if anomaly_type is None:
            blocks.append(
                {
                    "tag": "OUTLIER",
                    "fields": {
                        "TYPE": "random_choice",
                        "OPTIONS": "disturb_covariance,perturb_u_values",
                    },
                }
            )
        elif anomaly_type == "perturb_u_values":
            n_proto = int(getattr(self, "num_prototypes", 0))
            subset_sizes = getattr(self, "perturb_u_subset_sizes", None)
            subset_masks = getattr(self, "perturb_u_subset_masks", None)

            fields = {
                "TYPE": "perturb_u_values",
                "N_PROTO": n_proto,
                "STRENGTH_RANGE": "0.2~0.4",
            }
            if subset_sizes is not None and len(subset_sizes) > 0:
                fields["SUBSET_K_MIN"] = int(subset_sizes.min().item())
                fields["SUBSET_K_MAX"] = int(subset_sizes.max().item())
            blocks.append({"tag": "OUTLIER", "fields": fields})

            if subset_masks is not None and subset_masks.numel() > 0:
                for p in range(int(subset_masks.shape[0])):
                    dims = subset_masks[p].nonzero(as_tuple=True)[0].to(torch.float32)
                    blocks.append(
                        {
                            "tag": "ENTITY",
                            "fields": {
                                "TYPE": "OUTLIER_PROTOTYPE",
                                "ID": int(p),
                                "PERTURBED_DIMS": dims,
                            },
                        }
                    )
        elif anomaly_type == "disturb_covariance":
            n_proto = int(getattr(self, "num_prototypes", 0))
            block_sizes = getattr(self, "disturb_cov_block_sizes", None)
            block_starts = getattr(self, "disturb_cov_block_starts", None)
            block_ends = getattr(self, "disturb_cov_block_ends", None)

            fields = {
                "TYPE": "disturb_covariance",
                "N_PROTO": n_proto,
                "STRENGTH_RANGE": "0.97~0.99",
            }
            if block_sizes is not None and len(block_sizes) > 0:
                fields["BLOCK_K_MIN"] = int(block_sizes.min().item())
                fields["BLOCK_K_MAX"] = int(block_sizes.max().item())
            blocks.append({"tag": "OUTLIER", "fields": fields})

            if block_starts is not None and block_ends is not None:
                for p in range(int(len(block_starts))):
                    s = int(block_starts[p].item())
                    e = int(block_ends[p].item())
                    dims = torch.arange(s, e, device=self.device, dtype=torch.float32)
                    blocks.append(
                        {
                            "tag": "ENTITY",
                            "fields": {
                                "TYPE": "OUTLIER_PROTOTYPE",
                                "ID": int(p),
                                "DISTURBED_DIMS": dims,
                            },
                        }
                    )
        else:
            blocks.append({"tag": "OUTLIER", "fields": {"TYPE": str(anomaly_type), "PARAMS": "custom"}})

        for i, spec in enumerate(self.specs):
            dist_name = "unknown"
            params = "none"

            if spec.get("kind") == "interp":
                lo = float(spec.get("lo", float("nan")))
                hi = float(spec.get("hi", float("nan")))
                grid = int(spec.get("u", torch.tensor([])).numel())
                dist_name = "interp"
                params = f"lo={lo:.6g},hi={hi:.6g},u_grid={grid}"
            else:
                dist = spec.get("dist", None)
                if dist is not None:
                    if isinstance(dist, Normal):
                        dist_name = "normal"
                        params = f"{float(dist.loc):.6g},{float(dist.scale):.6g}"
                    elif isinstance(dist, Beta):
                        dist_name = "beta"
                        params = f"{float(dist.concentration1):.6g},{float(dist.concentration0):.6g}"
                    elif isinstance(dist, Exponential):
                        dist_name = "exponential"
                        params = f"{float(dist.rate):.6g}"
                    elif isinstance(dist, StudentT):
                        dist_name = "studentt"
                        params = f"{float(dist.df):.6g},{float(dist.loc):.6g},{float(dist.scale):.6g}"
                    else:
                        dist_name = dist.__class__.__name__.lower()
                        params = "unparsed"

            entities.append(
                {
                    "tag": "ENTITY",
                    "fields": {
                        "TYPE": "MARGINAL",
                        "ID": int(i),
                        "DIST": dist_name,
                        "PARAMS": params,
                    },
                }
            )

        return describe_prior_program(
            family="COPULA",
            global_fields={
                "DIM": int(self.num_dims),
            },
            entities=entities,
            blocks=blocks,
            decimals=decimals,
            max_len_map={
                "CHOL": max_chol_len,
                "PERTURBED_DIMS": max_dims_len,
                "DISTURBED_DIMS": max_dims_len,
            },
        )


def make_corpula(device, 
                 max_feature_dim=100,
                 min_feature_dim = 2,
                 dim = None,
                 num_prototypes: int = 16):
    if dim is not None:
        num_features = dim
    else:
        num_features = np.random.randint(min_feature_dim, max_feature_dim)
    return CorpulaGenerator(num_dims=num_features, device=device, num_prototypes=num_prototypes)


def make_disturb_corpula(device,
                          max_feature_dim=100,
                          min_feature_dim = 2,
                          dim = None,
                          num_prototypes: int = 16):
    if dim is not None:
        num_features = dim
    else:
        num_features = np.random.randint(min_feature_dim, max_feature_dim)
    return CorpulaGenerator(num_dims=num_features, device=device,
                            anomaly_types="disturb_covariance",
                            num_prototypes=num_prototypes) 
    
    
def make_perturb_corpula(device,
                          max_feature_dim=100,
                          min_feature_dim = 2,
                          dim = None,
                          num_prototypes: int = 16):
    if dim is not None:
        num_features = dim
    else:
        num_features = np.random.randint(min_feature_dim, max_feature_dim)
    return CorpulaGenerator(num_dims=num_features, device=device,
                            anomaly_types="perturb_u_values",
                            num_prototypes=num_prototypes)


def _build_marginal_spec_from_predefined(marginal_def, device='cpu', ppf_grid=2000):
    """
    marginal_def format:
      {"DIST": "normal", "PARAMS": [mu, sigma]}
      {"DIST": "beta", "PARAMS": [a, b]}
      {"DIST": "exponential", "PARAMS": [rate]}
      {"DIST": "studentt", "PARAMS": [df, loc, scale]}  # loc/scale optional
      {"DIST": "interp", "PARAMS": [lo, hi, u_grid]}     # u_grid optional
    """
    dist_name = str(marginal_def["DIST"]).strip().lower()
    params = marginal_def.get("PARAMS", [])

    if dist_name == "normal":
        mu, sigma = float(params[0]), float(params[1])
        return {"kind": "torch", "dist": Normal(torch.tensor(mu), torch.tensor(sigma))}

    if dist_name == "beta":
        a, b = float(params[0]), float(params[1])
        return {"kind": "torch", "dist": Beta(torch.tensor(a), torch.tensor(b))}

    if dist_name == "exponential":
        rate = float(params[0])
        return {"kind": "torch", "dist": Exponential(torch.tensor(rate))}

    if dist_name == "studentt":
        df = float(params[0])
        loc = float(params[1]) if len(params) > 1 else 0.0
        scale = float(params[2]) if len(params) > 2 else 1.0
        return {"kind": "torch", "dist": StudentT(torch.tensor(df), loc=torch.tensor(loc), scale=torch.tensor(scale))}

    if dist_name == "interp":
        lo = float(params[0])
        hi = float(params[1])
        grid = int(params[2]) if len(params) > 2 else int(ppf_grid)
        u_grid = torch.linspace(0.001, 0.999, grid, device=device, dtype=torch.float32)
        x_grid = torch.linspace(lo, hi, grid, device=device, dtype=torch.float32)
        return {"kind": "interp", "u": u_grid, "x": x_grid, "lo": lo, "hi": hi}

    raise ValueError(f"Unsupported predefined marginal DIST: {dist_name}")


def make_copula_predefined(
    *,
    num_dims,
    chol_base,
    marginals,
    device='cpu',
    anomaly_type=None,
    perturb_u_prototypes=None,
    disturb_cov_prototypes=None,
    num_prototypes=None,
    ppf_grid=2000,
):
    """
    Build a COPULA generator from predefined parameters, similar to make_NdMclusterGMM_predefined.
    """
    if num_prototypes is None:
        n_from_perturb = len(perturb_u_prototypes) if perturb_u_prototypes is not None else 0
        n_from_disturb = len(disturb_cov_prototypes) if disturb_cov_prototypes is not None else 0
        num_prototypes = max(1, n_from_perturb, n_from_disturb, 16)

    model = CorpulaGenerator(
        num_dims=int(num_dims),
        device=device,
        ppf_grid=int(ppf_grid),
        anomaly_types=anomaly_type,
        num_prototypes=int(num_prototypes),
    )

    # predefined copula base
    chol_base_t = torch.as_tensor(chol_base, device=device, dtype=torch.float32)
    if chol_base_t.shape != (num_dims, num_dims):
        raise ValueError(f"chol_base must have shape ({num_dims}, {num_dims}), got {tuple(chol_base_t.shape)}")
    model.chol_base = chol_base_t

    # predefined marginals
    if len(marginals) != int(num_dims):
        raise ValueError(f"Expected {num_dims} marginals, got {len(marginals)}")
    model.specs = [
        _build_marginal_spec_from_predefined(m, device=device, ppf_grid=ppf_grid)
        for m in marginals
    ]

    # predefined perturb_u prototypes
    if perturb_u_prototypes is not None:
        masks = []
        sizes = []
        for dims in perturb_u_prototypes:
            m = torch.zeros(num_dims, dtype=torch.bool, device=device)
            idx = torch.as_tensor(dims, device=device, dtype=torch.long)
            m[idx] = True
            masks.append(m)
            sizes.append(int(idx.numel()))
        model.perturb_u_subset_masks = torch.stack(masks, dim=0)
        model.perturb_u_subset_sizes = torch.tensor(sizes, device=device, dtype=torch.long)
        model.num_prototypes = int(model.perturb_u_subset_masks.shape[0])

    # predefined disturb_cov prototypes (expected contiguous ranges as dimension lists)
    if disturb_cov_prototypes is not None:
        starts, ends, sizes = [], [], []
        for dims in disturb_cov_prototypes:
            dims_sorted = sorted(int(x) for x in dims)
            if len(dims_sorted) == 0:
                raise ValueError("Each disturb_cov prototype must contain at least one dimension")
            s, e = dims_sorted[0], dims_sorted[-1] + 1
            if dims_sorted != list(range(s, e)):
                raise ValueError(
                    f"disturb_cov prototype dims must be contiguous for current implementation, got {dims_sorted}"
                )
            starts.append(s)
            ends.append(e)
            sizes.append(e - s)
        model.disturb_cov_block_starts = torch.tensor(starts, device=device, dtype=torch.long)
        model.disturb_cov_block_ends = torch.tensor(ends, device=device, dtype=torch.long)
        model.disturb_cov_block_sizes = torch.tensor(sizes, device=device, dtype=torch.long)
        model.num_prototypes = int(model.disturb_cov_block_starts.shape[0])

    return model



def condense_chol(chol_base, decimals=3, max_vals=24):
    # move to cpu and take lower-triangle (including diagonal)
    L = chol_base.detach().to("cpu", torch.float32)
    tri = L[torch.tril_indices(L.shape[0], L.shape[1], offset=0).unbind(0)]

    # round + truncate
    tri = torch.round(tri * (10 ** decimals)) / (10 ** decimals)
    vals = tri.tolist()
    shown = vals[:max_vals]
    s = ",".join(f"{v:.{decimals}f}" for v in shown)
    if len(vals) > max_vals:
        s += ",..."
    return s, len(vals)



def describe_model_specs(model, decimals=3, max_chol_len=24, max_dims_len=24):
    """Backward-compatible wrapper. Prefer calling `model.describe_model_specs(...)`."""
    return model.describe_model_specs(
        decimals=decimals,
        max_chol_len=max_chol_len,
        max_dims_len=max_dims_len,
    )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_dims = 10

    # predefined copula parameter (valid lower-triangular Cholesky)
    chol_base = torch.linalg.cholesky(rand_corr_batch(1, num_dims, device=device)[0])

    # predefined marginals (same schema as description)
    marginals = [
        {"DIST": "studentt", "PARAMS": [8, 0, 1]},
        {"DIST": "beta", "PARAMS": [1.24619, 3.05254]},
        {"DIST": "normal", "PARAMS": [0.782339, 1.61374]},
        {"DIST": "exponential", "PARAMS": [1.89512]},
        {"DIST": "normal", "PARAMS": [-0.888, 1.76016]},
        {"DIST": "interp", "PARAMS": [-5.36984, 6.44931, 2000]},
        {"DIST": "exponential", "PARAMS": [1.44607]},
        {"DIST": "normal", "PARAMS": [-0.048419, 1.77265]},
        {"DIST": "studentt", "PARAMS": [4, 0, 1]},
        {"DIST": "normal", "PARAMS": [-0.949632, 1.68195]},
    ]

    # predefined contiguous prototype blocks for disturb_covariance
    disturb_cov_prototypes = [
        [3, 4, 5, 6, 7],
        [3, 4, 5, 6, 7, 8],
        [4, 5, 6, 7, 8, 9],
        [3, 4, 5, 6, 7, 8],
        [4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5],
        [5, 6, 7, 8, 9],
        [2, 3, 4, 5, 6, 7],
        [2, 3, 4, 5],
        [4, 5, 6, 7, 8],
        [3, 4, 5, 6, 7, 8],
        [4, 5, 6, 7, 8, 9],
        [6, 7, 8, 9],
        [2, 3, 4, 5, 6, 7],
        [2, 3, 4, 5, 6, 7],
        [2, 3, 4, 5, 6],
    ]

    model = make_copula_predefined(
        num_dims=num_dims,
        chol_base=chol_base,
        marginals=marginals,
        device=device,
        anomaly_type="disturb_covariance",
        disturb_cov_prototypes=disturb_cov_prototypes,
    )

    print("\n===== PREDEFINED DISTURB_COVARIANCE =====")
    descriptor_text = model.describe_model_specs()
    print(descriptor_text)
    inliers, outliers = model.draw_batched_data(1000, 1000)

    # predefined subset prototypes for perturb_u_values
    perturb_u_prototypes = [
        [0, 1, 2],
        [1, 3, 5],
        [2, 4, 6, 8],
        [0, 4, 7],
        [3, 6, 9],
        [1, 2, 8],
        [0, 5, 9],
        [2, 3, 4],
        [5, 6, 7],
        [7, 8, 9],
        [0, 2, 4, 6],
        [1, 3, 7, 9],
        [0, 8],
        [4, 5],
        [2, 9],
        [1, 6],
    ]

    model_perturb = make_copula_predefined(
        num_dims=num_dims,
        chol_base=chol_base,
        marginals=marginals,
        device=device,
        anomaly_type="perturb_u_values",
        perturb_u_prototypes=perturb_u_prototypes,
    )

    print("\n===== PREDEFINED PERTURB_U_VALUES =====")
    descriptor_text_perturb = model_perturb.describe_model_specs()
    print(descriptor_text_perturb)
    inliers_p, outliers_p = model_perturb.draw_batched_data(1000, 1000)