"""
Build the IEEE single-column verification figure from plot_data.npz produced by
verify_theory.py: theory curves (lines) vs simulation (markers), for the
covariance-ratio E[Sigma_t]/Sigma and mean-drift-variance-ratio Var[mu_t]/Sigma,
averaged over the 5 coordinates, for n in {200,2000} and alpha in
{0,0.01,0.05,0.2,1.0}.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1.1,
    "lines.markersize": 3.5,
})

# Colorblind-safe (Okabe-Ito)
COLORS = {
    0.0: "#000000",
    0.01: "#0072B2",
    0.05: "#D55E00",
    0.2: "#009E73",
    1.0: "#CC79A7",
}

data = np.load("H:/Model-paper/code/plot_data.npz")
n_values = [200, 2000]
alpha_values = [0.0, 0.01, 0.05, 0.2, 1.0]

fig, axes = plt.subplots(2, 2, figsize=(3.4, 3.0), sharex="col")

for col, n in enumerate(n_values):
    ax_top = axes[0, col]
    ax_bot = axes[1, col]
    for alpha in alpha_values:
        t = data[f"{n}_{alpha}__t"]
        S_th = data[f"{n}_{alpha}__Sigma_theory"]
        S_sim = data[f"{n}_{alpha}__Sigma_sim"]
        V_th = data[f"{n}_{alpha}__varmu_theory"]
        V_sim = data[f"{n}_{alpha}__varmu_sim"]
        c = COLORS[alpha]
        marker_idx = np.arange(0, len(t), 20)

        ax_top.plot(t, S_th, "-", color=c, label=fr"$\alpha={alpha}$")
        ax_top.plot(t[marker_idx], S_sim[marker_idx], "o", color=c, mfc="none", mew=0.7)

        ax_bot.plot(t, V_th, "-", color=c)
        ax_bot.plot(t[marker_idx], V_sim[marker_idx], "o", color=c, mfc="none", mew=0.7)

    ax_top.set_title(fr"$n={n}$", fontsize=8)
    ax_bot.set_xlabel("generation $t$")
    ax_top.grid(alpha=0.25, linewidth=0.4)
    ax_bot.grid(alpha=0.25, linewidth=0.4)

axes[0, 0].set_ylabel(r"$E[\Sigma_t]/\Sigma$")
axes[1, 0].set_ylabel(r"$\mathrm{Var}[\mu_t]/\Sigma$")
axes[0, 1].legend(loc="upper right", frameon=False, handlelength=1.6)

fig.tight_layout(pad=0.4)
fig.savefig("H:/Model-paper/paper/figs/fig_theory_verify.pdf")
print("Saved paper/figs/fig_theory_verify.pdf")
