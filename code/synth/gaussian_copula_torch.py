"""GPU (torch) port of the Gaussian copula synthesizer.

Same algorithm as gaussian_copula_numpy.GaussianCopulaNumpy (rank -> probit
for numeric columns, frequency-interval -> probit for categorical columns,
joint correlation matrix with a nearest-PD eigenvalue-clipping fix,
Cholesky sampling, then inverse transforms), but every numeric op (rank
transform, correlation estimate, eigendecomposition, Cholesky, sampling,
and quantile interpolation) runs as a torch tensor op on `device`. This is
the synthesizer called the most times in a full sweep, so keeping it on
GPU matters more than for TVAE/CTGAN where the per-call cost is already
dominated by SGD epochs.

Validated against the numpy reference in code/tests/test_copula_parity.py
(same fitted correlation structure -> statistically indistinguishable
samples; RNG differs between numpy and torch so bit-exact equality is not
expected, only Monte-Carlo-level agreement).
"""
import numpy as np
import pandas as pd
import torch

EPS = 1e-6


def _average_rank(x):
    """scipy.stats.rankdata(method='average') equivalent, torch tensor in,
    float64 tensor out, same device as x."""
    n = x.shape[0]
    sorted_val, sorted_idx = torch.sort(x)
    unique_vals, inverse, counts = torch.unique_consecutive(
        sorted_val, return_inverse=True, return_counts=True
    )
    cum = torch.cumsum(counts, 0)
    group_start = (cum - counts).to(torch.float64)
    avg_rank_per_group = group_start + (counts.to(torch.float64) + 1) / 2.0
    sorted_ranks = avg_rank_per_group[inverse]
    ranks = torch.empty(n, dtype=torch.float64, device=x.device)
    ranks[sorted_idx] = sorted_ranks
    return ranks


def _normal_cdf(x):
    return 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))


def _normal_ppf(u):
    # torch.special.ndtri = inverse standard normal CDF (probit), matches scipy.special.ndtri
    return torch.special.ndtri(u)


def _nearest_pd_corr(corr, eps=1e-6):
    vals, vecs = torch.linalg.eigh(corr)
    vals = torch.clamp(vals, min=eps)
    corr_pd = (vecs * vals) @ vecs.T
    d = torch.sqrt(torch.diagonal(corr_pd).clamp(min=1e-12))
    corr_pd = corr_pd / torch.outer(d, d)
    corr_pd = (corr_pd + corr_pd.T) / 2.0
    corr_pd.fill_diagonal_(1.0)
    return corr_pd


def _interp_1d(u, levels, sorted_vals):
    """torch equivalent of np.interp(u, levels, sorted_vals): levels must be
    sorted ascending; values outside [levels[0], levels[-1]] are clamped to
    the boundary y-value (no extrapolation), matching numpy's default."""
    u = torch.clamp(u, levels[0], levels[-1])
    idx = torch.searchsorted(levels, u)
    idx = torch.clamp(idx, 1, levels.shape[0] - 1)
    x0, x1 = levels[idx - 1], levels[idx]
    y0, y1 = sorted_vals[idx - 1], sorted_vals[idx]
    denom = (x1 - x0).clamp(min=1e-12)
    frac = (u - x0) / denom
    return y0 + frac * (y1 - y0)


def _categorical_latent(cat_ids, K, device):
    """cat_ids: int64 tensor [n] of category indices in [0,K).
    Returns (u, bounds) where u is the within-frequency-interval uniform
    position for each row and bounds is the length-(K+1) cumulative
    frequency boundary tensor."""
    n = cat_ids.shape[0]
    counts = torch.bincount(cat_ids, minlength=K).to(torch.float64)
    freqs = counts / counts.sum()
    bounds = torch.zeros(K + 1, dtype=torch.float64, device=device)
    bounds[1:] = torch.cumsum(freqs, 0)
    bounds[-1] = 1.0

    sorted_idx = torch.argsort(cat_ids, stable=True)
    sorted_cat = cat_ids[sorted_idx]
    starts = torch.cumsum(counts, 0) - counts
    positions = torch.arange(n, device=device, dtype=torch.float64) - starts[sorted_cat]
    cnt_per_row_sorted = counts[sorted_cat]
    pos_frac_sorted = (positions + 0.5) / cnt_per_row_sorted

    pos_frac = torch.empty(n, dtype=torch.float64, device=device)
    pos_frac[sorted_idx] = pos_frac_sorted

    lo = bounds[cat_ids]
    hi = bounds[cat_ids + 1]
    u = lo + pos_frac * (hi - lo)
    return u, bounds


class GaussianCopulaTorch:
    def __init__(self, numeric_cols, categorical_cols, seed=0, device=None):
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.seed = seed
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gen = torch.Generator(device=self.device.type if self.device.type != "cpu" else "cpu")
        self.gen.manual_seed(seed)
        self.num_sorted = {}
        self.num_levels = {}
        self.cat_categories = {}
        self.cat_bounds = {}
        self.corr = None
        self.chol = None

    def reseed(self, seed):
        self.gen.manual_seed(seed)

    def fit(self, df):
        dev = self.device
        latent_cols = []
        for c in self.numeric_cols:
            x = torch.as_tensor(df[c].to_numpy(dtype=np.float64), device=dev)
            sorted_vals, _ = torch.sort(x)
            n = x.shape[0]
            levels = (torch.arange(n, device=dev, dtype=torch.float64) + 0.5) / n
            self.num_sorted[c] = sorted_vals
            self.num_levels[c] = levels
            ranks = _average_rank(x)
            u = torch.clamp((ranks - 0.5) / n, EPS, 1 - EPS)
            latent_cols.append(_normal_ppf(u))
        for c in self.categorical_cols:
            vals = df[c].astype(str).to_numpy()
            vc = df[c].astype(str).value_counts()
            cats = vc.index.tolist()
            cat_idx = {cat: i for i, cat in enumerate(cats)}
            code = np.array([cat_idx[v] for v in vals], dtype=np.int64)
            cat_ids = torch.as_tensor(code, device=dev)
            u, bounds = _categorical_latent(cat_ids, len(cats), dev)
            u = torch.clamp(u, EPS, 1 - EPS)
            self.cat_categories[c] = cats
            self.cat_bounds[c] = bounds
            latent_cols.append(_normal_ppf(u))
        if not latent_cols:
            raise ValueError("GaussianCopula needs at least one column")
        Z = torch.stack(latent_cols, dim=1)  # [n, d]
        Z_centered = Z - Z.mean(dim=0, keepdim=True)
        cov = (Z_centered.T @ Z_centered) / (Z.shape[0] - 1)
        std = torch.sqrt(torch.diagonal(cov).clamp(min=1e-12))
        corr = cov / torch.outer(std, std)
        corr = torch.nan_to_num(corr, nan=0.0)
        corr.fill_diagonal_(1.0)
        self.corr = _nearest_pd_corr(corr)
        self.chol = torch.linalg.cholesky(self.corr)
        return self

    def sample(self, k):
        dev = self.device
        d = self.chol.shape[0]
        z0 = torch.randn((k, d), generator=self.gen, device=dev, dtype=torch.float64)
        z = z0 @ self.chol.T
        col_i = 0
        out = {}
        for c in self.numeric_cols:
            u = torch.clamp(_normal_cdf(z[:, col_i]), EPS, 1 - EPS)
            vals = _interp_1d(u, self.num_levels[c], self.num_sorted[c])
            out[c] = vals.cpu().numpy()
            col_i += 1
        for c in self.categorical_cols:
            u = torch.clamp(_normal_cdf(z[:, col_i]), EPS, 1 - EPS)
            bounds = self.cat_bounds[c]
            idx = torch.searchsorted(bounds, u, right=True) - 1
            idx = torch.clamp(idx, 0, len(self.cat_categories[c]) - 1)
            idx_np = idx.cpu().numpy()
            cats = np.array(self.cat_categories[c])
            out[c] = cats[idx_np]
            col_i += 1
        return pd.DataFrame(out)
