"""Verifies the torch/GPU Gaussian copula (gaussian_copula_torch) agrees
with the numpy/scipy reference (gaussian_copula_numpy) at the level of
distributional statistics, not bit-for-bit samples (numpy's and torch's
RNGs differ, so exact sample equality is neither expected nor meaningful).

Fits both on the same real data, draws a large sample from each, and
checks that per-column means/stds and the fitted correlation matrices
agree to a loose Monte-Carlo tolerance. Run directly with `python
test_copula_parity.py` (no pytest dependency required).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synth.gaussian_copula_numpy import GaussianCopulaNumpy
from synth.gaussian_copula_torch import GaussianCopulaTorch


def _make_test_df(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(5, 2, n)
    x2 = 0.6 * x1 + rng.normal(0, 1, n)
    x3 = rng.exponential(2, n) * np.sign(rng.normal(size=n))  # has ties near 0
    x3[rng.random(n) < 0.1] = 0.0  # force some exact ties
    cat_a = rng.choice(["a", "b", "c"], size=n, p=[0.6, 0.3, 0.1])
    cat_b = rng.choice(["x", "y"], size=n, p=[0.8, 0.2])
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "cat_a": cat_a, "cat_b": cat_b})


def test_w1_and_correlation_parity():
    df = _make_test_df()
    numeric_cols = ["x1", "x2", "x3"]
    categorical_cols = ["cat_a", "cat_b"]

    np_model = GaussianCopulaNumpy(numeric_cols, categorical_cols, seed=1).fit(df)
    torch_model = GaussianCopulaTorch(numeric_cols, categorical_cols, seed=1).fit(df)

    k = 20000
    s_np = np_model.sample(k)
    s_torch = torch_model.sample(k)

    print("device used by torch copula:", torch_model.device)

    for c in numeric_cols:
        m_np, m_torch = s_np[c].mean(), s_torch[c].mean()
        sd_np, sd_torch = s_np[c].std(), s_torch[c].std()
        real_sd = df[c].std()
        print(f"[{c}] mean np={m_np:.4f} torch={m_torch:.4f}  std np={sd_np:.4f} torch={sd_torch:.4f}")
        assert abs(m_np - m_torch) < 0.1 * real_sd, f"mean mismatch on {c}"
        assert abs(sd_np - sd_torch) < 0.1 * real_sd, f"std mismatch on {c}"

    for c in categorical_cols:
        freq_np = s_np[c].value_counts(normalize=True).sort_index()
        freq_torch = s_torch[c].value_counts(normalize=True).sort_index()
        freq_np, freq_torch = freq_np.align(freq_torch, fill_value=0.0)
        max_diff = (freq_np - freq_torch).abs().max()
        print(f"[{c}] max category-frequency diff = {max_diff:.4f}")
        assert max_diff < 0.03, f"categorical frequency mismatch on {c}"

    corr_np = np_model.corr
    corr_torch = torch_model.corr.cpu().numpy()
    fro_diff = np.linalg.norm(corr_np - corr_torch, ord="fro") / np.linalg.norm(corr_np, ord="fro")
    print(f"correlation matrix relative Frobenius diff = {fro_diff:.4f}")
    assert fro_diff < 0.05

    print("PASS: torch copula matches numpy reference within Monte-Carlo tolerance")


if __name__ == "__main__":
    test_w1_and_correlation_parity()
