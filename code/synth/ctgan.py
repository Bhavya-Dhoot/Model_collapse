"""CTGAN-lite: a small WGAN-GP for tabular data.

Numeric columns min-max scaled to [-1,1] (so the generator's tanh output
head is meaningful), categorical columns one-hot encoded. Generator MLP
128-128 with BatchNorm+ReLU, outputs tanh for numeric block and
Gumbel-Softmax per categorical block. Critic MLP 128-128, pac=1 (no
sample-packing by default; pac>1 concatenates that many samples' features
before scoring, exposed as a param but not the default). Gradient penalty
weight 10, n_critic critic steps per generator step (1 or 2, param).
Adam(2e-4, betas=(0.5, 0.9)), batch 256, ~200 epochs (param).

Runs fully on the given torch device; the encoded training matrix is
moved to GPU once and minibatches are drawn via on-device torch.randperm.
Deliberately NOT run under torch.autocast/AMP: WGAN-GP needs a
double-backward through the critic (grad of the interpolated critic
output, then grad of the gradient-penalty term w.r.t. critic params), and
mixed-precision double-backward is fragile (grad-scaler + create_graph
interactions can silently produce NaNs) -- per orchestrator instruction,
kept in fp32 rather than risk destabilizing the one part of the harness
that's already numerically the trickiest to get right.

Logs (print) if any categorical column loses support in the generated
sample relative to what was seen during training.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import TabularTransformer

HIDDEN = 128
NOISE_DIM = 128


class _Generator(nn.Module):
    def __init__(self, noise_dim, n_numeric, cat_dims, hidden=HIDDEN):
        super().__init__()
        self.n_numeric = n_numeric
        self.cat_dims = cat_dims
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
        )
        self.numeric_head = nn.Linear(hidden, n_numeric) if n_numeric > 0 else None
        self.cat_heads = nn.ModuleList([nn.Linear(hidden, dim) for dim in cat_dims])

    def forward(self, z, tau=0.5, hard=False):
        h = self.net(z)
        num_out = torch.tanh(self.numeric_head(h)) if self.numeric_head is not None else None
        cat_outs = [F.gumbel_softmax(head(h), tau=tau, hard=hard) for head in self.cat_heads]
        return num_out, cat_outs


class _Critic(nn.Module):
    def __init__(self, input_dim, pac=1, hidden=HIDDEN):
        super().__init__()
        self.pac = pac
        self.net = nn.Sequential(
            nn.Linear(input_dim * pac, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        n = x.shape[0]
        pac = self.pac
        if pac > 1:
            usable = (n // pac) * pac
            x = x[:usable].reshape(n // pac, -1)
        return self.net(x)


class CTGAN:
    def __init__(self, numeric_cols, categorical_cols, seed=0, device=None,
                 epochs=200, batch_size=256, n_critic=2, gp_weight=10.0,
                 pac=1, lr=2e-4, tau=0.5):
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.seed = seed
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_critic = n_critic
        self.gp_weight = gp_weight
        self.pac = pac
        self.lr = lr
        self.tau = tau
        self.transformer = TabularTransformer(numeric_cols, categorical_cols, numeric_mode="minmax")
        self.generator = None
        self.critic = None
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self._train_cat_support = {}
        self._last_lost_support = None

    def reseed(self, seed):
        self.gen.manual_seed(seed)

    def _gradient_penalty(self, real, fake):
        n = min(real.shape[0], fake.shape[0])
        pac = self.pac
        n = (n // pac) * pac
        if n == 0:
            return torch.tensor(0.0, device=self.device)
        real, fake = real[:n], fake[:n]
        alpha = torch.rand((n // pac, 1, 1), device=self.device)
        alpha = alpha.expand(-1, pac, real.shape[1]).reshape(n, -1)
        interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        d_interp = self.critic(interpolates)
        grad_outputs = torch.ones_like(d_interp)
        gradients = torch.autograd.grad(
            outputs=d_interp, inputs=interpolates, grad_outputs=grad_outputs,
            create_graph=True, retain_graph=True, only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.shape[0], -1)
        grad_norm = gradients.norm(2, dim=1)
        return ((grad_norm - 1) ** 2).mean()

    def fit(self, df):
        torch.manual_seed(self.seed)
        self.transformer.fit(df)
        X = self.transformer.transform(df)
        n_numeric = len(self.numeric_cols)
        cat_dims = self.transformer.cat_dims
        cat_slices = self.transformer.cat_slices()
        self._train_cat_support = {
            c: set(np.where(X[:, s:e].sum(axis=0) > 0)[0].tolist())
            for c, (s, e) in zip(self.categorical_cols, cat_slices)
        }

        X_t = torch.from_numpy(X).to(self.device)
        input_dim = X.shape[1]

        self.generator = _Generator(NOISE_DIM, n_numeric, cat_dims).to(self.device)
        self.critic = _Critic(input_dim, pac=self.pac).to(self.device)
        g_opt = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.9))
        c_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr, betas=(0.5, 0.9))

        n = X_t.shape[0]
        bs = min(self.batch_size, n)
        steps_per_epoch = max(1, n // bs)

        for epoch in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for step in range(steps_per_epoch):
                idx = perm[step * bs:(step + 1) * bs]
                real_batch = X_t[idx]
                bs_actual = real_batch.shape[0]

                # ---- critic steps (fp32, no autocast: WGAN-GP double-backward) ----
                for _ in range(self.n_critic):
                    z = torch.randn((bs_actual, NOISE_DIM), device=self.device)
                    with torch.no_grad():
                        num_out, cat_outs = self.generator(z, tau=self.tau)
                        fake_batch = self._assemble_fake(num_out, cat_outs, n_numeric)
                    c_opt.zero_grad(set_to_none=True)
                    d_real = self.critic(real_batch).mean()
                    d_fake = self.critic(fake_batch).mean()
                    gp = self._gradient_penalty(real_batch, fake_batch)
                    c_loss = d_fake - d_real + self.gp_weight * gp
                    c_loss.backward()
                    c_opt.step()

                # ---- generator step ----
                z = torch.randn((bs_actual, NOISE_DIM), device=self.device)
                g_opt.zero_grad(set_to_none=True)
                num_out, cat_outs = self.generator(z, tau=self.tau)
                fake_batch = self._assemble_fake(num_out, cat_outs, n_numeric)
                g_loss = -self.critic(fake_batch).mean()
                g_loss.backward()
                g_opt.step()
        return self

    def _assemble_fake(self, num_out, cat_outs, n_numeric):
        parts = []
        if num_out is not None:
            parts.append(num_out)
        parts.extend(cat_outs)
        return torch.cat(parts, dim=1) if parts else torch.zeros((0, 0), device=self.device)

    def sample(self, k):
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn((k, NOISE_DIM), generator=self.gen, device=self.device)
            num_out, cat_outs = self.generator(z, tau=self.tau, hard=True)
            numeric_dict = {}
            if num_out is not None:
                numeric_dict = self.transformer.inverse_transform_numeric(
                    num_out.detach().cpu().numpy().astype(np.float64)
                )
            cat_logits_np = [c.detach().cpu().numpy() for c in cat_outs]
            categorical_dict = self.transformer.inverse_transform_categorical_argmax(cat_logits_np)
        self.generator.train()

        total_lost = 0
        for c, logits in zip(self.categorical_cols, cat_logits_np):
            sampled_support = set(np.argmax(logits, axis=1).tolist())
            train_support = self._train_cat_support.get(c, set())
            lost = train_support - sampled_support
            if lost:
                cats = self.transformer.cat_categories[c]
                lost_names = [cats[i] for i in lost if i < len(cats)]
                print(f"[ctgan] WARNING: column '{c}' lost support for categories {lost_names} "
                      f"({len(lost)}/{len(train_support)} categories missing from sample)")
                total_lost += len(lost)
        self._last_lost_support = total_lost

        return self.transformer.assemble_df(numeric_dict, categorical_dict, k)
