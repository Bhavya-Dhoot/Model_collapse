"""TVAE-lite: a small variational autoencoder for tabular data.

Numeric columns are standard-scaled, categorical columns one-hot encoded
(TabularTransformer in base.py). Symmetric MLP encoder/decoder (128-128),
latent dim 16, Gaussian likelihood on numeric outputs + softmax
cross-entropy per categorical block, KL term weighted by beta.

Sampling draws z ~ N(0,I), decodes, and samples numeric values from the
decoder's predicted Gaussian (mean AND sigma) rather than just returning
the mean -- outputting the mean only would itself be a variance-collapsing
shortcut that confounds a study about model collapse, so we deliberately
keep the sampling stochastic.

Categorical blocks are sampled from the decoder's predicted softmax
distribution (torch.multinomial), NOT argmax. An earlier version of this
module used argmax and produced catastrophic mode dropping on adult
(cat_support ~0.19 at generation 0, i.e. already broken before the
self-consuming loop even starts) -- argmax always returns the single most
likely category for a given z, so even a well-trained decoder whose
per-z distribution is only mildly peaked will collapse dozens of
real categories down to whichever few are ever anyone's mode, especially
for high-cardinality columns like native-country (40+ levels). Sampling
from the softmax is the generative-model-correct way to decode a
categorical head and is what makes the numeric side's "sample with sigma,
not just the mean" comment (above) apply consistently to the categorical
side too.

Runs fully on the given torch device (default: resolved by gpu_utils,
GPU unless --device cpu). The whole encoded training matrix is moved to
the device once per fit() call; minibatches are drawn via torch.randperm
on-device so there is no per-batch host<->device transfer. Trains under
torch.autocast + GradScaler (AMP) since a VAE's single-pass loss/backward
is well-behaved under mixed precision (unlike the WGAN-GP critic in
ctgan.py, which needs a double-backward for the gradient penalty).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import TabularTransformer

LATENT_DIM = 16
HIDDEN = 128


class _Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim=LATENT_DIM, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logvar(h)


class _Decoder(nn.Module):
    def __init__(self, latent_dim, n_numeric, cat_dims, hidden=HIDDEN):
        super().__init__()
        self.n_numeric = n_numeric
        self.cat_dims = cat_dims
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        # numeric head predicts both mean and log-sigma per numeric column
        self.numeric_head = nn.Linear(hidden, 2 * n_numeric) if n_numeric > 0 else None
        self.cat_heads = nn.ModuleList([nn.Linear(hidden, dim) for dim in cat_dims])

    def forward(self, z):
        h = self.net(z)
        num_mu, num_logsigma = None, None
        if self.numeric_head is not None:
            out = self.numeric_head(h)
            num_mu, num_logsigma = out[:, :self.n_numeric], out[:, self.n_numeric:]
            num_logsigma = torch.clamp(num_logsigma, -5.0, 3.0)
        cat_logits = [head(h) for head in self.cat_heads]
        return num_mu, num_logsigma, cat_logits


class TVAE:
    def __init__(self, numeric_cols, categorical_cols, seed=0, device=None,
                 epochs=120, batch_size=256, beta=1.0, lr=1e-3, use_amp=True, verbose=False):
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.seed = seed
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.batch_size = batch_size
        self.beta = beta
        self.lr = lr
        self.use_amp = use_amp and self.device.type == "cuda"
        self.verbose = verbose
        self.transformer = TabularTransformer(numeric_cols, categorical_cols, numeric_mode="standard")
        self.encoder = None
        self.decoder = None
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.loss_history = []  # (epoch, recon, kl, elbo) per epoch if verbose

    def reseed(self, seed):
        self.gen.manual_seed(seed)

    def fit(self, df):
        torch.manual_seed(self.seed)
        self.transformer.fit(df)
        X = self.transformer.transform(df)  # numpy [n, input_dim]
        n_numeric = len(self.numeric_cols)
        cat_dims = self.transformer.cat_dims
        cat_slices = self.transformer.cat_slices()

        X_t = torch.from_numpy(X).to(self.device)  # moved to GPU once

        input_dim = X.shape[1]
        self.encoder = _Encoder(input_dim).to(self.device)
        self.decoder = _Decoder(LATENT_DIM, n_numeric, cat_dims).to(self.device)
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        opt = torch.optim.Adam(params, lr=self.lr)
        scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        n = X_t.shape[0]
        bs = min(self.batch_size, n)
        steps_per_epoch = max(1, n // bs)

        log_every = max(1, self.epochs // 20)
        for epoch in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            epoch_recon, epoch_kl, n_steps = 0.0, 0.0, 0
            for step in range(steps_per_epoch):
                idx = perm[step * bs:(step + 1) * bs]
                batch = X_t[idx]
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    mu, logvar = self.encoder(batch)
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    z = mu + eps * std
                    num_mu, num_logsigma, cat_logits = self.decoder(z)

                    recon = 0.0
                    if n_numeric > 0:
                        target_num = batch[:, :n_numeric]
                        sigma = torch.exp(num_logsigma)
                        # Gaussian NLL (up to additive constant)
                        recon = recon + (
                            num_logsigma + 0.5 * ((target_num - num_mu) / sigma) ** 2
                        ).sum(dim=1).mean()
                    for (start, end), logits in zip(cat_slices, cat_logits):
                        target_cat = batch[:, start:end].argmax(dim=1)
                        recon = recon + F.cross_entropy(logits, target_cat)

                    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
                    loss = recon + self.beta * kl

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                epoch_recon += recon.item()
                epoch_kl += kl.item()
                n_steps += 1
            avg_recon, avg_kl = epoch_recon / n_steps, epoch_kl / n_steps
            self.loss_history.append((epoch, avg_recon, avg_kl, avg_recon + self.beta * avg_kl))
            if self.verbose and (epoch % log_every == 0 or epoch == self.epochs - 1):
                print(f"  [tvae] epoch {epoch:4d}/{self.epochs}: recon={avg_recon:.4f} "
                      f"kl={avg_kl:.4f} elbo={avg_recon + self.beta * avg_kl:.4f}")
        return self

    def sample(self, k):
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            z = torch.randn((k, LATENT_DIM), generator=self.gen, device=self.device)
            num_mu, num_logsigma, cat_logits = self.decoder(z)
            numeric_dict = {}
            if num_mu is not None:
                sigma = torch.exp(num_logsigma)
                eps = torch.randn(num_mu.shape, generator=self.gen, device=self.device)
                sampled_num = num_mu + eps * sigma  # NOT mean-only: see module docstring
                numeric_dict = self.transformer.inverse_transform_numeric(
                    sampled_num.detach().cpu().numpy().astype(np.float64)
                )
            categorical_dict = {}
            for c, logits in zip(self.categorical_cols, cat_logits):
                probs = torch.softmax(logits, dim=1)
                idx = torch.multinomial(probs, num_samples=1, generator=self.gen).squeeze(1)
                cats = np.array(self.transformer.cat_categories[c])
                categorical_dict[c] = cats[idx.detach().cpu().numpy()]
        self.encoder.train()
        self.decoder.train()
        return self.transformer.assemble_df(numeric_dict, categorical_dict, k)
