"""Validates the GPU (torch) Wasserstein-1 implementation against
scipy.stats.wasserstein_distance (the oracle), on random test cases with
equal and unequal sample sizes, plus a sanity check for coverage_knn.
Run directly with `python test_metrics_gpu.py` (no pytest dependency).
"""
import os
import sys

import numpy as np
import torch
from scipy.stats import wasserstein_distance

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics import _w1_1d_torch, coverage_knn
import pandas as pd


def test_w1_matches_scipy_oracle():
    rng = np.random.default_rng(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_err = 0.0
    for trial in range(20):
        n_u = rng.integers(50, 3000)
        n_v = rng.integers(50, 3000)
        u = rng.normal(rng.uniform(-3, 3), rng.uniform(0.5, 3), n_u)
        v = rng.normal(rng.uniform(-3, 3), rng.uniform(0.5, 3), n_v)
        expected = wasserstein_distance(u, v)
        got = _w1_1d_torch(torch.from_numpy(u), torch.from_numpy(v), device).item()
        err = abs(expected - got)
        max_err = max(max_err, err)
        assert err < 1e-6, f"trial {trial}: scipy={expected} torch={got} err={err}"
    print(f"PASS: w1 torch vs scipy oracle, max abs error over 20 trials = {max_err:.2e}")


def test_coverage_knn_sanity():
    rng = np.random.default_rng(0)
    n = 500
    holdout = pd.DataFrame({"x": rng.normal(0, 1, n), "y": rng.normal(0, 1, n)})
    synth_good = pd.DataFrame({"x": rng.normal(0, 1, n), "y": rng.normal(0, 1, n)})
    synth_collapsed = pd.DataFrame({"x": rng.normal(0, 0.01, n), "y": rng.normal(0, 0.01, n)})
    cov_good = coverage_knn(synth_good, holdout, ["x", "y"], k=5)
    cov_collapsed = coverage_knn(synth_collapsed, holdout, ["x", "y"], k=5)
    print(f"coverage: good={cov_good:.3f} collapsed={cov_collapsed:.3f}")
    assert cov_good > cov_collapsed, "collapsed synth should cover holdout less than well-matched synth"
    print("PASS: coverage_knn sanity check")


if __name__ == "__main__":
    test_w1_matches_scipy_oracle()
    test_coverage_knn_sanity()
