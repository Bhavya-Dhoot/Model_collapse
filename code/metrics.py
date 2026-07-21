"""
Metrics for comparing a synthetic sample against a real holdout H.

Every function has signature f(synth_df, holdout_df, numeric_cols,
categorical_cols, target_col=None, **kw) -> float, and returns np.nan
(never raises) on degenerate inputs (constant columns, empty categories,
single-class targets, etc). Callers should count NaNs to track how often
degeneracy occurs.

corr_frob uses Spearman correlation (rank-based) computed over numeric
columns only (categoricals are not included) so it is well-defined for
mixed-type frames without needing an arbitrary ordinal encoding choice for
categoricals; see docstring on corr_frob.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score

# Device used by the GPU-accelerated metrics (w1, coverage_knn). TSTR/C2ST
# stay on CPU -- sklearn's HistGradientBoostingClassifier has no GPU path,
# it just uses its own internal (OpenMP) threading.
_METRIC_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# "fast" (default): c2st_auc uses a single stratified 70/30 split instead of
# 5-fold CV, and c2st_auc/coverage_knn subsample to at most _FAST_MAX_ROWS
# per side. This is what dominates per-generation wallclock (profiled: 5-fold
# CV HistGradientBoosting over synth+holdout, with holdout often 4-5x larger
# than the synth sample, costs ~5x a single fit on the full concatenated set).
# "full": no subsampling, full 5-fold CV -- use for spot-checking that the
# fast metrics track the expensive ones on a handful of configs.
_METRICS_MODE = "fast"
_FAST_MAX_ROWS = 4000


def set_metric_device(device):
    global _METRIC_DEVICE
    _METRIC_DEVICE = device


def set_metrics_mode(mode):
    global _METRICS_MODE
    if mode not in ("fast", "full"):
        raise ValueError(f"metrics mode must be 'fast' or 'full', got {mode}")
    _METRICS_MODE = mode


def _make_hgb(**kw):
    """HistGradientBoostingClassifier with early stopping forced on
    regardless of dataset size (sklearn's default early_stopping='auto'
    only kicks in above 10k rows, so our ~2k-11k row fits were running the
    full max_iter every time) -- this is the other half of the C2ST/TSTR
    speedup, on top of fewer folds and subsampling."""
    return HistGradientBoostingClassifier(
        max_iter=100, early_stopping=True, n_iter_no_change=10,
        validation_fraction=0.1, random_state=0, **kw
    )


def _subsample(df, max_rows, seed=0):
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=seed)


def _safe_numeric(df, numeric_cols):
    if not numeric_cols:
        return None
    return df[numeric_cols].to_numpy(dtype=float)


def _w1_1d_torch(u, v, device):
    """1-D Wasserstein-1 distance between two empirical samples, ported
    from scipy.stats._cdf_distance (the same combined-support CDF-difference
    algorithm scipy uses), so it matches scipy.stats.wasserstein_distance to
    floating-point precision -- see code/tests/test_metrics_gpu.py for the
    oracle comparison. u, v: 1-D torch tensors (possibly different lengths).
    """
    u = u.to(device=device, dtype=torch.float64)
    v = v.to(device=device, dtype=torch.float64)
    u_sorted, _ = torch.sort(u)
    v_sorted, _ = torch.sort(v)
    all_values, _ = torch.sort(torch.cat([u, v]))
    deltas = all_values[1:] - all_values[:-1]
    u_cdf_idx = torch.searchsorted(u_sorted, all_values[:-1], right=True)
    v_cdf_idx = torch.searchsorted(v_sorted, all_values[:-1], right=True)
    u_cdf = u_cdf_idx.to(torch.float64) / u.shape[0]
    v_cdf = v_cdf_idx.to(torch.float64) / v.shape[0]
    return torch.sum(torch.abs(u_cdf - v_cdf) * deltas)


def mean_shift(synth, holdout, numeric_cols, categorical_cols=None, **kw):
    if not numeric_cols:
        return np.nan
    vals = []
    for c in numeric_cols:
        h = holdout[c].to_numpy(dtype=float)
        s = synth[c].to_numpy(dtype=float)
        std_h = np.nanstd(h)
        if std_h == 0 or not np.isfinite(std_h) or len(s) == 0:
            continue
        vals.append(abs(np.nanmean(s) - np.nanmean(h)) / std_h)
    return float(np.mean(vals)) if vals else np.nan


def var_ratio(synth, holdout, numeric_cols, categorical_cols=None, **kw):
    if not numeric_cols:
        return np.nan
    vals = []
    for c in numeric_cols:
        h = holdout[c].to_numpy(dtype=float)
        s = synth[c].to_numpy(dtype=float)
        var_h = np.nanvar(h)
        if var_h == 0 or not np.isfinite(var_h) or len(s) == 0:
            continue
        vals.append(np.nanvar(s) / var_h)
    return float(np.mean(vals)) if vals else np.nan


def w1(synth, holdout, numeric_cols, categorical_cols=None, device=None, **kw):
    """Mean over numeric cols of W1(synth, holdout), each column
    standardized by holdout's std. Computed on GPU via a torch port of
    scipy's combined-support CDF-difference algorithm (see
    _w1_1d_torch above); validated against scipy.stats.wasserstein_distance
    in code/tests/test_metrics_gpu.py."""
    if not numeric_cols:
        return np.nan
    dev = device or _METRIC_DEVICE
    vals = []
    for c in numeric_cols:
        h = holdout[c].to_numpy(dtype=float)
        s = synth[c].to_numpy(dtype=float)
        std_h = np.nanstd(h)
        if std_h == 0 or not np.isfinite(std_h) or len(s) == 0:
            continue
        u = torch.from_numpy(s / std_h)
        v = torch.from_numpy(h / std_h)
        vals.append(_w1_1d_torch(u, v, dev).item())
    return float(np.mean(vals)) if vals else np.nan


def tv_cat(synth, holdout, numeric_cols=None, categorical_cols=None, **kw):
    if not categorical_cols:
        return np.nan
    vals = []
    for c in categorical_cols:
        h = holdout[c].astype(str)
        s = synth[c].astype(str)
        if len(s) == 0:
            continue
        cats = pd.Index(pd.unique(pd.concat([h, s])))
        h_freq = h.value_counts(normalize=True).reindex(cats, fill_value=0.0)
        s_freq = s.value_counts(normalize=True).reindex(cats, fill_value=0.0)
        tv = 0.5 * np.abs(h_freq.to_numpy() - s_freq.to_numpy()).sum()
        vals.append(tv)
    return float(np.mean(vals)) if vals else np.nan


def corr_frob(synth, holdout, numeric_cols, categorical_cols=None, **kw):
    """Spearman correlation matrix Frobenius distance, numeric columns only.
    Categorical columns are excluded (an ordinal encoding of unordered
    categories would inject an arbitrary rank structure into the
    correlation estimate, biasing this metric); see module docstring.

    Uses pandas .corr(method='spearman') rather than
    scipy.stats.spearmanr directly: scipy's spearmanr returns a bare
    scalar (not a 1x1/2x2 matrix) when given exactly two columns, which
    would silently miscompute the Frobenius norm below for any
    2-numeric-column dataset -- pandas always returns a proper k x k
    matrix regardless of k.
    """
    cols = [c for c in numeric_cols if synth[c].nunique() > 1 and holdout[c].nunique() > 1]
    if len(cols) < 2:
        return np.nan
    try:
        corr_s = synth[cols].astype(float).corr(method="spearman").to_numpy()
        corr_h = holdout[cols].astype(float).corr(method="spearman").to_numpy()
    except Exception:
        return np.nan
    if np.any(~np.isfinite(corr_s)) or np.any(~np.isfinite(corr_h)):
        corr_s = np.nan_to_num(corr_s)
        corr_h = np.nan_to_num(corr_h)
    denom = np.linalg.norm(corr_h, ord="fro")
    if denom == 0:
        return np.nan
    return float(np.linalg.norm(corr_s - corr_h, ord="fro") / denom)


def cat_support(synth, holdout, numeric_cols=None, categorical_cols=None, **kw):
    """Fraction of (column, category) pairs present in H that still appear
    in the synthetic sample -- detects mode/category dropping."""
    if not categorical_cols:
        return np.nan
    present = 0
    total = 0
    for c in categorical_cols:
        h_cats = set(holdout[c].astype(str).unique())
        s_cats = set(synth[c].astype(str).unique())
        total += len(h_cats)
        present += len(h_cats & s_cats)
    if total == 0:
        return np.nan
    return float(present / total)


def _chunked_min_dist(query, ref, device, chunk_size=2048, exclude_self=False):
    """For each row in `query`, returns (min_dist_to_ref, argmin_index),
    computed in chunks via torch.cdist on `device` so we never materialize
    a full n_query x n_ref matrix for large n. If exclude_self, assumes
    query is ref and masks the diagonal (self-distance) to +inf."""
    q = torch.as_tensor(query, device=device, dtype=torch.float32)
    r = torch.as_tensor(ref, device=device, dtype=torch.float32)
    n_q = q.shape[0]
    min_d = torch.empty(n_q, device=device, dtype=torch.float32)
    min_idx = torch.empty(n_q, device=device, dtype=torch.long)
    for start in range(0, n_q, chunk_size):
        end = min(start + chunk_size, n_q)
        d = torch.cdist(q[start:end], r)  # [chunk, n_ref]
        if exclude_self:
            rows = torch.arange(end - start, device=device)
            d[rows, torch.arange(start, end, device=device)] = float("inf")
        vals, idx = torch.min(d, dim=1)
        min_d[start:end] = vals
        min_idx[start:end] = idx
    return min_d.cpu().numpy(), min_idx.cpu().numpy()


def _chunked_kth_dist(ref, k, device, chunk_size=1024):
    """For each row in ref, distance to its k-th nearest neighbor within
    ref itself (excluding itself), via chunked torch.cdist + topk on GPU."""
    r = torch.as_tensor(ref, device=device, dtype=torch.float32)
    n = r.shape[0]
    out = torch.empty(n, device=device, dtype=torch.float32)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        d = torch.cdist(r[start:end], r)  # [chunk, n]
        rows = torch.arange(end - start, device=device)
        d[rows, torch.arange(start, end, device=device)] = float("inf")  # exclude self
        kth, _ = torch.kthvalue(d, k, dim=1)
        out[start:end] = kth
    return out.cpu().numpy()


def coverage_knn(synth, holdout, numeric_cols, categorical_cols=None, k=5, device=None, **kw):
    """Simple coverage metric (Naeem et al. 2020 style, simplified):
    standardize numeric cols by holdout mean/std, build k-NN radii around
    each synth point (distance to its k-th synth neighbor), and report the
    fraction of holdout points that fall within at least one such ball.
    This rewards synthetic samples that spread out enough to "cover" the
    real manifold; a collapsed (low-variance) synth sample covers little.
    Distances computed via chunked torch.cdist on GPU (falls back to CPU
    tensors if no CUDA device), not sklearn NearestNeighbors.
    """
    if not numeric_cols:
        return np.nan
    dev = device or _METRIC_DEVICE
    if _METRICS_MODE == "fast":
        holdout = _subsample(holdout, _FAST_MAX_ROWS, seed=0)
        synth = _subsample(synth, _FAST_MAX_ROWS, seed=0)
    h = holdout[numeric_cols].to_numpy(dtype=float)
    s = synth[numeric_cols].to_numpy(dtype=float)
    mu = np.nanmean(h, axis=0)
    sd = np.nanstd(h, axis=0)
    sd[sd == 0] = 1.0
    h_z = (h - mu) / sd
    s_z = (s - mu) / sd
    n_s = s_z.shape[0]
    if n_s <= k or h_z.shape[0] == 0:
        return np.nan
    try:
        radii = _chunked_kth_dist(s_z, k, dev)  # [n_s], distance to k-th nearest synth neighbor
        dist_h_to_s, idx_h_to_s = _chunked_min_dist(h_z, s_z, dev)
        nearest_radius = radii[idx_h_to_s]
        covered = dist_h_to_s <= nearest_radius
        return float(np.mean(covered))
    except Exception:
        return np.nan


def tstr_auc(synth, holdout, numeric_cols, categorical_cols, target_col, **kw):
    """Train on Synthetic, Test on Real (holdout). HistGradientBoosting
    classifier, ROC-AUC. Returns np.nan if synth has a single target class
    or other degeneracy."""
    return _train_eval_auc(synth, holdout, numeric_cols, categorical_cols, target_col)


def trtr_auc(real_train, holdout, numeric_cols, categorical_cols, target_col, **kw):
    """Train on Real, Test on Real (holdout) -- the ceiling reference."""
    return _train_eval_auc(real_train, holdout, numeric_cols, categorical_cols, target_col)


def _train_eval_auc(train_df, holdout, numeric_cols, categorical_cols, target_col):
    try:
        y_train = train_df[target_col]
        y_test = holdout[target_col]
        if y_train.nunique() < 2:
            return np.nan
        feat_cols = numeric_cols + categorical_cols
        X_train = train_df[feat_cols].copy()
        X_test = holdout[feat_cols].copy()
        for c in categorical_cols:
            X_train[c] = X_train[c].astype(str)
            X_test[c] = X_test[c].astype(str)
        cat_idx = [feat_cols.index(c) for c in categorical_cols]
        clf = _make_hgb(categorical_features=cat_idx if cat_idx else None)
        # align target encoding: map to binary 0/1 using train's classes,
        # unseen holdout labels (shouldn't happen, same domain) -> nan-safe
        classes = sorted(y_train.unique().tolist())
        if len(classes) != 2:
            # binarize: treat the majority class in holdout as positive vs rest
            # (rare edge case for the specified binary-target datasets)
            pass
        y_train_bin = (y_train == classes[-1]).astype(int)
        y_test_bin = (y_test == classes[-1]).astype(int)
        if y_test_bin.nunique() < 2:
            return np.nan
        clf.fit(X_train, y_train_bin)
        proba = clf.predict_proba(X_test)[:, 1]
        return float(roc_auc_score(y_test_bin, proba))
    except Exception:
        return np.nan


def c2st_auc(synth, holdout, numeric_cols, categorical_cols, **kw):
    """CV ROC-AUC of a classifier discriminating synth (label 0) vs
    holdout (label 1). 0.5 = indistinguishable, 1.0 = fully separable
    (collapsed / detectable synth).

    In "fast" mode (the default, see set_metrics_mode): both sides are
    subsampled to a common, BALANCED size (min(len(synth), len(holdout),
    _FAST_MAX_ROWS) each) before fitting -- holdout is often 4-5x larger
    than the synth sample, which both slows the fit (more rows) and biases
    the AUC with a class-imbalance artifact rather than measuring a clean
    two-sample statistic, so equalizing sizes is a correctness fix as much
    as a speed one. Uses 3-fold CV with max_iter=100 + early_stopping=True
    (measured ~15x faster than uncapped 5-fold on a 2x2000-row problem;
    kept as CV rather than a single split so the estimate isn't a
    coin-flip on which rows landed in the one test split). "full" mode
    restores 5-fold CV with no subsampling, for spot-checking fast tracks
    full (see code/tests/test_metrics_gpu.py or --metrics full runs).
    """
    try:
        feat_cols = numeric_cols + categorical_cols
        if not feat_cols:
            return np.nan
        if _METRICS_MODE == "fast":
            cap = min(len(synth), len(holdout), _FAST_MAX_ROWS)
            synth = _subsample(synth, cap, seed=0)
            holdout = _subsample(holdout, cap, seed=0)
        s = synth[feat_cols].copy()
        h = holdout[feat_cols].copy()
        for c in categorical_cols:
            s[c] = s[c].astype(str)
            h[c] = h[c].astype(str)
        X = pd.concat([s, h], axis=0, ignore_index=True)
        y = np.concatenate([np.zeros(len(s)), np.ones(len(h))])
        if len(np.unique(y)) < 2 or len(y) < 10:
            return np.nan
        cat_idx = [feat_cols.index(c) for c in categorical_cols]
        clf = _make_hgb(categorical_features=cat_idx if cat_idx else None)
        n_splits = 3 if _METRICS_MODE == "fast" else 5
        min_class = min((y == 0).sum(), (y == 1).sum())
        if min_class < n_splits:
            n_splits = max(2, min_class)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
        return float(np.mean(scores))
    except Exception:
        return np.nan


METRIC_FUNCS = {
    "mean_shift": mean_shift,
    "var_ratio": var_ratio,
    "w1": w1,
    "tv_cat": tv_cat,
    "corr_frob": corr_frob,
    "cat_support": cat_support,
    "coverage": coverage_knn,
}


def compute_all_metrics(synth, holdout, numeric_cols, categorical_cols, target_col):
    """Compute the distributional metrics (excludes tstr_auc/trtr_auc which
    need special handling by the caller). Returns dict metric_name -> float,
    plus 'n_nan_metrics' count of metrics that returned nan.

    c2st_auc is computed separately (not via METRIC_FUNCS) because it
    deliberately includes target_col in the discriminating feature set --
    unlike the feature-distribution metrics above (tv_cat, cat_support,
    corr_frob), c2st asks "can anything tell synth from real apart", and
    excluding the label would weaken that test.
    """
    out = {}
    n_nan = 0
    for name, fn in METRIC_FUNCS.items():
        try:
            val = fn(synth, holdout, numeric_cols, categorical_cols, target_col=target_col)
        except Exception:
            val = np.nan
        if val is None or (isinstance(val, float) and np.isnan(val)):
            n_nan += 1
        out[name] = val
    out["c2st_auc"] = c2st_auc(synth, holdout, numeric_cols, categorical_cols + [target_col])
    if out["c2st_auc"] is None or np.isnan(out["c2st_auc"]):
        n_nan += 1
    out["tstr_auc"] = tstr_auc(synth, holdout, numeric_cols, categorical_cols, target_col)
    if out["tstr_auc"] is None or np.isnan(out["tstr_auc"]):
        n_nan += 1
    out["n_nan_metrics"] = n_nan
    return out
