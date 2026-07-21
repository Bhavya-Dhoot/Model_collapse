"""Reference (numpy/scipy-only, CPU) Gaussian copula synthesizer.

Numeric columns: empirical-CDF rank transform -> inverse normal (probit).
Categorical columns: ordinal-encoded by frequency, then each category is
given a sub-interval of [0,1] proportional to its empirical frequency; a
row's uniform value is its position within its category's interval
(ties broken by stable row order), then probit-transformed the same way
as numeric columns.

A joint Gaussian correlation matrix is fit over all transformed columns
(nearest-PD eigenvalue-clipping fix if the raw estimate isn't PD), and new
rows are sampled via a Cholesky factor of that correlation matrix, then
each column is inverted back: numeric via linear interpolation on the
empirical quantile function, categorical via looking up which frequency
interval the probit-normal-CDF value falls into.

This is the reference "oracle" implementation used to validate the torch
GPU port (gaussian_copula_torch.py) -- see code/tests/test_copula_parity.py.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

EPS = 1e-6


def _nearest_pd_corr(corr, eps=1e-6):
    vals, vecs = np.linalg.eigh(corr)
    vals = np.clip(vals, eps, None)
    corr_pd = (vecs * vals) @ vecs.T
    d = np.sqrt(np.diag(corr_pd))
    d[d == 0] = 1.0
    corr_pd = corr_pd / np.outer(d, d)
    corr_pd = (corr_pd + corr_pd.T) / 2.0
    np.fill_diagonal(corr_pd, 1.0)
    return corr_pd


class GaussianCopulaNumpy:
    def __init__(self, numeric_cols, categorical_cols, seed=0):
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.num_sorted = {}   # col -> sorted training values (for quantile inversion)
        self.cat_categories = {}  # col -> list of categories, freq order desc
        self.cat_bounds = {}   # col -> cumulative boundary array, len(cats)+1
        self.corr = None
        self.chol = None
        self.n_fit = None

    def reseed(self, seed):
        self.rng = np.random.default_rng(seed)

    def _numeric_latent(self, df):
        cols = []
        for c in self.numeric_cols:
            x = df[c].to_numpy(dtype=float)
            self.num_sorted[c] = np.sort(x)
            ranks = rankdata(x, method="average")
            u = (ranks - 0.5) / len(x)
            u = np.clip(u, EPS, 1 - EPS)
            cols.append(norm.ppf(u))
        return cols

    def _categorical_latent(self, df):
        cols = []
        for c in self.categorical_cols:
            vals = df[c].astype(str).to_numpy()
            vc = df[c].astype(str).value_counts()
            cats = vc.index.tolist()
            freqs = vc.to_numpy(dtype=float) / vc.sum()
            bounds = np.concatenate([[0.0], np.cumsum(freqs)])
            bounds[-1] = 1.0
            self.cat_categories[c] = cats
            self.cat_bounds[c] = bounds
            cat_idx = {cat: i for i, cat in enumerate(cats)}
            code = np.array([cat_idx[v] for v in vals])
            n = len(vals)
            u = np.empty(n, dtype=float)
            # within-category position: stable order by original row index
            for k in range(len(cats)):
                rows = np.where(code == k)[0]
                cnt = len(rows)
                if cnt == 0:
                    continue
                pos = (np.arange(cnt) + 0.5) / cnt
                u[rows] = bounds[k] + pos * (bounds[k + 1] - bounds[k])
            u = np.clip(u, EPS, 1 - EPS)
            cols.append(norm.ppf(u))
        return cols

    def fit(self, df):
        self.n_fit = len(df)
        latent_cols = self._numeric_latent(df) + self._categorical_latent(df)
        if not latent_cols:
            raise ValueError("GaussianCopula needs at least one column")
        Z = np.column_stack(latent_cols)
        corr = np.corrcoef(Z, rowvar=False)
        corr = np.atleast_2d(corr)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)
        self.corr = _nearest_pd_corr(corr)
        self.chol = np.linalg.cholesky(self.corr)
        return self

    def sample(self, k):
        d = self.chol.shape[0]
        z = self.rng.standard_normal((k, d)) @ self.chol.T
        col_i = 0
        out = {}
        for c in self.numeric_cols:
            u = norm.cdf(z[:, col_i])
            u = np.clip(u, EPS, 1 - EPS)
            sorted_vals = self.num_sorted[c]
            n = len(sorted_vals)
            levels = (np.arange(n) + 0.5) / n
            out[c] = np.interp(u, levels, sorted_vals)
            col_i += 1
        for c in self.categorical_cols:
            u = norm.cdf(z[:, col_i])
            u = np.clip(u, EPS, 1 - EPS)
            bounds = self.cat_bounds[c]
            cats = np.array(self.cat_categories[c])
            idx = np.searchsorted(bounds, u, side="right") - 1
            idx = np.clip(idx, 0, len(cats) - 1)
            out[c] = cats[idx]
            col_i += 1
        return pd.DataFrame(out)
