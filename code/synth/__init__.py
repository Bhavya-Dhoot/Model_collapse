"""Factory for the three synthesizers, all sharing a common
fit(df) / sample(k) interface."""
from .gaussian_copula import GaussianCopula
from .tvae import TVAE
from .ctgan import CTGAN

SYNTH_NAMES = ["gaussian_copula", "tvae", "ctgan"]


def get_synth(name, numeric_cols, categorical_cols, seed=0, device=None,
              epochs_vae=120, epochs_gan=200, copula_impl="torch"):
    if name == "gaussian_copula":
        return GaussianCopula(numeric_cols, categorical_cols, seed=seed, device=device,
                               impl=copula_impl)
    elif name == "tvae":
        return TVAE(numeric_cols, categorical_cols, seed=seed, device=device, epochs=epochs_vae)
    elif name == "ctgan":
        return CTGAN(numeric_cols, categorical_cols, seed=seed, device=device, epochs=epochs_gan)
    else:
        raise ValueError(f"unknown synthesizer {name}")
