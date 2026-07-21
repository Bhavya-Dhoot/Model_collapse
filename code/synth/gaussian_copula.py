"""Dispatcher for the Gaussian copula synthesizer: picks the torch/GPU
implementation by default (fastest, and this synthesizer is invoked most
often in a full sweep), or the numpy/scipy CPU reference implementation
when impl='numpy' -- kept around as the correctness oracle, see
code/tests/test_copula_parity.py.
"""
from .gaussian_copula_numpy import GaussianCopulaNumpy
from .gaussian_copula_torch import GaussianCopulaTorch


class GaussianCopula:
    def __init__(self, numeric_cols, categorical_cols, seed=0, device=None, impl="torch", **kw):
        self.impl_name = impl
        if impl == "numpy":
            self._impl = GaussianCopulaNumpy(numeric_cols, categorical_cols, seed=seed)
        elif impl == "torch":
            self._impl = GaussianCopulaTorch(numeric_cols, categorical_cols, seed=seed, device=device)
        else:
            raise ValueError(f"unknown copula impl {impl}")

    def fit(self, df):
        self._impl.fit(df)
        return self

    def sample(self, k):
        return self._impl.sample(k)

    def reseed(self, seed):
        self._impl.reseed(seed)
