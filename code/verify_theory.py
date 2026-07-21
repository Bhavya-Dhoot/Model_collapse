"""
Monte-Carlo verification of the Gaussian real-data-anchoring recursion derived in
sec_theory.tex (Propositions 1-3). Simulates the exact self-consuming loop:

    at generation t, fit a Gaussian (biased/MLE, divide-by-n) to a training set of
    n rows = alpha*n fresh real rows ~ N(mu, Sigma)  +  (1-alpha)*n rows sampled
    from the previous fitted model G_{t-1} = N(mu_{t-1}, Sigma_{t-1}).

Checks, SEPARATELY for E[Sigma_t] and Var[mu_t] (these have very different
Monte-Carlo precision and must never be collapsed into a single "max rel err"):
    (a) the full transient recursion (Propositions 1-2), at checkpoints t=20..200
    (b) the non-zero fixed point Sigma_infinity, Var[mu_infinity] (Proposition 3),
        run out to a horizon T_fp(alpha) long enough that (1-alpha)^T_fp is
        negligible -- comparing simulation to the ASYMPTOTE before the transient
        has actually mixed is a measurement artifact, not evidence about the
        algebra, so this script sizes the horizon per alpha explicitly.

Every discrepancy is reported as a z-score = (sim - theory) / MC_stderr, where
MC_stderr is estimated by a batch-means (group) split of the replicates -- this
requires no distributional assumption (in particular it does NOT assume Sigma_t
is asymptotically Gaussian, which it is not: at alpha=0 it is a product of t iid
chi-square factors and becomes heavy-tailed as t grows, so the coefficient of
variation of the MEAN estimator itself is derived and reported below).

GPU (torch, float64) batched across replicates; falls back to CPU with a warning.
"""
import torch
import numpy as np
import math
import json

torch.set_default_dtype(torch.float64)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    print("WARNING: CUDA not available, falling back to CPU. This will be slow.")
    device = torch.device("cpu")

print(f"Using device: {device}")

# ---------------------------------------------------------------------------
d = 5
R = 2000                 # replicates (batched dimension)
N_GROUPS = 20             # batch-means groups for MC stderr
seed = 0
Z_FAIL = 4.0              # |z| above this, consistently, => algebra bug

torch.manual_seed(seed)
np.random.seed(seed)

mu_true = torch.arange(1, d + 1, dtype=torch.float64, device=device) * 1.0
sigma2_true = torch.tensor([0.5, 1.0, 2.0, 3.0, 5.0], dtype=torch.float64, device=device)
Sigma_true = torch.diag(sigma2_true)
L_true = torch.linalg.cholesky(Sigma_true)

n_values = [200, 2000]
alpha_values = [0.0, 0.01, 0.05, 0.2, 1.0]
T_TRAJ = 200              # horizon for the transient-recursion check (uniform)
TRAJ_CHECKPOINTS = list(range(20, T_TRAJ + 1, 20))


def fixed_point_horizon(alpha):
    """Generations needed so (1-alpha)^T_fp <= ~1e-6 (Prop. 3 rate rho=1-alpha),
    floored at T_TRAJ and capped for tractability."""
    if alpha <= 0:
        return None
    T_needed = math.ceil(14 / alpha)  # (1-alpha)^T ~ exp(-alpha*T); alpha*T=14 -> ~8e-7
    return max(T_TRAJ, min(T_needed, 1500))


# ---------------------------------------------------------------------------
# Closed-form theory
# ---------------------------------------------------------------------------
def theory_trajectory(alpha, n, sigma2, T):
    """Exact transient recursion (Props. 1-2), iterated generation by generation."""
    c = (n - 1) / n
    Et, vt = sigma2, 0.0
    Es, vs = [Et], [vt]
    for _ in range(T):
        E_new = alpha * c * sigma2 + (1 - alpha) * c * Et + alpha * (1 - alpha) * vt
        v_new = (alpha / n) * sigma2 + ((1 - alpha) / n) * Et + (1 - alpha) ** 2 * vt
        Et, vt = E_new, v_new
        Es.append(Et)
        vs.append(vt)
    return np.array(Es), np.array(vs)


def theory_fixed_point(alpha, n, sigma2):
    if alpha == 0:
        return 0.0, sigma2
    D = 1 - alpha + alpha * n * (2 - alpha)
    return (1 - 1 / D) * sigma2, (1 / D) * sigma2


def cv_alpha0(n, t):
    """Analytic coefficient of variation of the Sigma_t MEAN estimator at alpha=0:
    Sigma_t = Sigma_0 * prod_{k=1}^t R_k, R_k iid chi2_{n-1}/n.
    E[R]=(n-1)/n, E[R^2]=(n^2-1)/n^2 (chi2_{n-1} second moment). CV^2 of product
    = ((n+1)/(n-1))^t - 1. Included so the reported z-scores can be sanity-checked
    against a closed-form heavy-tail prediction, not just the empirical stderr."""
    return math.sqrt(((n + 1) / (n - 1)) ** t - 1)


# ---------------------------------------------------------------------------
# Batched GPU simulation
# ---------------------------------------------------------------------------
def simulate(alpha, n, T_run, R, device):
    n_r = int(round(alpha * n))
    n_s = n - n_r
    group_size = R // N_GROUPS

    mu_t = mu_true.unsqueeze(0).repeat(R, 1).clone()
    Sigma_t = Sigma_true.unsqueeze(0).repeat(R, 1, 1).clone()

    diag_idx = torch.arange(d)
    mu_hist = torch.empty((T_run + 1, R, d), dtype=torch.float64, device=device)
    Sigma_diag_hist = torch.empty((T_run + 1, R, d), dtype=torch.float64, device=device)
    mu_hist[0] = mu_t
    Sigma_diag_hist[0] = Sigma_t[:, diag_idx, diag_idx]

    for t in range(1, T_run + 1):
        parts = []
        if n_r > 0:
            z_real = torch.randn(R, n_r, d, device=device)
            parts.append(mu_true.view(1, 1, d) + torch.einsum('ij,rnj->rni', L_true, z_real))
        if n_s > 0:
            L_prev = torch.linalg.cholesky(Sigma_t)
            z_syn = torch.randn(R, n_s, d, device=device)
            parts.append(mu_t.unsqueeze(1) + torch.einsum('rij,rnj->rni', L_prev, z_syn))
        X = torch.cat(parts, dim=1)
        mu_new = X.mean(dim=1)
        Xc = X - mu_new.unsqueeze(1)
        Sigma_new = torch.einsum('rni,rnj->rij', Xc, Xc) / n

        mu_t, Sigma_t = mu_new, Sigma_new
        mu_hist[t] = mu_t
        Sigma_diag_hist[t] = Sigma_t[:, diag_idx, diag_idx]

    return Sigma_diag_hist.cpu().numpy(), mu_hist.cpu().numpy()


def batch_means_stat_and_stderr(samples_2d, stat_fn):
    """samples_2d: (R,) array. stat_fn: e.g. np.mean or (lambda x: x.var(ddof=1)).
    Returns (overall_stat, stderr) via a G-group batch-means split."""
    R = samples_2d.shape[0]
    g = R // N_GROUPS
    group_stats = np.array([stat_fn(samples_2d[i * g:(i + 1) * g]) for i in range(N_GROUPS)])
    overall = stat_fn(samples_2d)
    stderr = group_stats.std(ddof=1) / math.sqrt(N_GROUPS)
    return overall, stderr


# ---------------------------------------------------------------------------
results = []
all_z_traj_S, all_z_traj_V, all_z_fp_S, all_z_fp_V = [], [], [], []

for n in n_values:
    for alpha in alpha_values:
        T_fp = fixed_point_horizon(alpha)
        T_run = max(T_TRAJ, T_fp) if T_fp is not None else T_TRAJ
        print(f"\n=== n={n}, alpha={alpha}  (T_run={T_run}"
              f"{f', T_fp={T_fp}' if T_fp else ''}) ===")

        Sigma_diag_hist, mu_hist = simulate(alpha, n, T_run, R, device)

        row = {"n": n, "alpha": alpha, "T_run": T_run}
        z_traj_S, z_traj_V = [], []

        for k in range(d):
            sigma2_k = sigma2_true[k].item()
            Es_th, vs_th = theory_trajectory(alpha, n, sigma2_k, T_run)
            for tcp in TRAJ_CHECKPOINTS:
                sim_S, se_S = batch_means_stat_and_stderr(Sigma_diag_hist[tcp, :, k], np.mean)
                z_S = (sim_S - Es_th[tcp]) / se_S
                z_traj_S.append(z_S)

                sim_V, se_V = batch_means_stat_and_stderr(
                    mu_hist[tcp, :, k], lambda x: x.var(ddof=1))
                z_V = (sim_V - vs_th[tcp]) / se_V
                z_traj_V.append(z_V)

        traj_max_z_S = max(abs(z) for z in z_traj_S)
        traj_max_z_V = max(abs(z) for z in z_traj_V)
        all_z_traj_S += z_traj_S
        all_z_traj_V += z_traj_V
        print(f"  trajectory (t=20..{T_TRAJ}): max|z| Sigma={traj_max_z_S:.2f}  "
              f"max|z| Var[mu]={traj_max_z_V:.2f}")
        row["traj_max_absz_Sigma"] = float(traj_max_z_S)
        row["traj_max_absz_varmu"] = float(traj_max_z_V)

        if alpha > 0:
            z_fp_S, z_fp_V = [], []
            for k in range(d):
                sigma2_k = sigma2_true[k].item()
                Sig_inf_th, V_inf_th = theory_fixed_point(alpha, n, sigma2_k)
                sim_S, se_S = batch_means_stat_and_stderr(Sigma_diag_hist[T_fp, :, k], np.mean)
                z_S = (sim_S - Sig_inf_th) / se_S
                z_fp_S.append(z_S)
                sim_V, se_V = batch_means_stat_and_stderr(
                    mu_hist[T_fp, :, k], lambda x: x.var(ddof=1))
                z_V = (sim_V - V_inf_th) / se_V
                z_fp_V.append(z_V)
            fp_max_z_S = max(abs(z) for z in z_fp_S)
            fp_max_z_V = max(abs(z) for z in z_fp_V)
            all_z_fp_S += z_fp_S
            all_z_fp_V += z_fp_V
            print(f"  fixed point (t={T_fp}, (1-a)^t={ (1-alpha)**T_fp:.2e}): "
                  f"max|z| Sigma_inf={fp_max_z_S:.2f}  max|z| Var[mu_inf]={fp_max_z_V:.2f}")
            row["fp_T"] = T_fp
            row["fp_max_absz_Sigma"] = float(fp_max_z_S)
            row["fp_max_absz_varmu"] = float(fp_max_z_V)
        else:
            row["fp_T"] = None
            row["fp_max_absz_Sigma"] = None
            row["fp_max_absz_varmu"] = None
            cv = cv_alpha0(n, T_TRAJ)
            print(f"  (alpha=0: no anchored fixed point; analytic CV of Sigma_{T_TRAJ} "
                  f"mean estimator = {cv:.2f}, i.e. heavy right tail, large finite-R noise expected)")
            row["analytic_cv_Sigma_T"] = float(cv)

        results.append(row)

overall_max = {
    "traj_max_absz_Sigma": max(abs(z) for z in all_z_traj_S),
    "traj_max_absz_varmu": max(abs(z) for z in all_z_traj_V),
    "fp_max_absz_Sigma": max(abs(z) for z in all_z_fp_S) if all_z_fp_S else None,
    "fp_max_absz_varmu": max(abs(z) for z in all_z_fp_V) if all_z_fp_V else None,
}
verdict = "PASS" if all(v is None or v < Z_FAIL for v in overall_max.values()) else "FAIL"

print("\n" + "=" * 70)
print("OVERALL MAX |z-score| PER QUANTITY:")
for k, v in overall_max.items():
    print(f"  {k}: {v}")
print(f"VERDICT (threshold |z|<{Z_FAIL}): {verdict}")
print("=" * 70)

with open("H:/Model-paper/code/verify_results.json", "w") as f:
    json.dump({"overall_max_abs_z": overall_max, "verdict": verdict,
                "z_fail_threshold": Z_FAIL, "results": results}, f, indent=2)

# ---------------------------------------------------------------------------
# Figure data: trajectories for n in {200,2000}, all alphas, normalized by sigma2
# (re-simulate at T_TRAJ only, cheap, for a clean plotting set independent of the
# variable-length T_run arrays above)
# ---------------------------------------------------------------------------
plot_data = {}
for n in n_values:
    for alpha in alpha_values:
        Sigma_diag_hist, mu_hist = simulate(alpha, n, T_TRAJ, R, device)
        Es_avg = np.zeros(T_TRAJ + 1)
        vs_avg = np.zeros(T_TRAJ + 1)
        for k in range(d):
            Es_k, vs_k = theory_trajectory(alpha, n, sigma2_true[k].item(), T_TRAJ)
            Es_avg += Es_k / sigma2_true[k].item()
            vs_avg += vs_k / sigma2_true[k].item()
        Es_avg /= d
        vs_avg /= d
        sim_S_ratio = (Sigma_diag_hist / sigma2_true.cpu().numpy()[None, None, :]).mean(axis=(1, 2))
        sim_V_ratio = (mu_hist.var(axis=1, ddof=1) / sigma2_true.cpu().numpy()[None, :]).mean(axis=1)
        plot_data[f"{n}_{alpha}__Sigma_theory"] = Es_avg
        plot_data[f"{n}_{alpha}__Sigma_sim"] = sim_S_ratio
        plot_data[f"{n}_{alpha}__varmu_theory"] = vs_avg
        plot_data[f"{n}_{alpha}__varmu_sim"] = sim_V_ratio
        plot_data[f"{n}_{alpha}__t"] = np.arange(T_TRAJ + 1)

np.savez("H:/Model-paper/code/plot_data.npz", **plot_data)
print("\nSaved verify_results.json and plot_data.npz")
