import numpy as np
from scipy.special import digamma

# Scalar alpha=0 loop with UNBIASED (divide by n-1) covariance estimator.
# Sigma_t = Sigma_{t-1} * (chi2_{n-1}/(n-1))  [since MLE mean-centering costs 1 dof]
# E[Sigma_t] = Sigma_{t-1} exactly (unbiased) -> flat in expectation.
# But log(Sigma_t) = log(Sigma_{t-1}) + log(chi2_{n-1}/(n-1)), and
# E[log(chi2_k/k)] = log(2) + digamma(k/2) - log(k)  < 0  strictly (Jensen).

def drift_exact(k):
    return np.log(2) + digamma(k/2) - np.log(k)

for n in [50, 200, 2000]:
    k = n - 1
    d_exact = drift_exact(k)
    d_leading = -1.0/k
    print(f"n={n}: exact E[log ratio]={d_exact:.6e}, leading-order -1/(n-1)={d_leading:.6e}, ratio={d_exact/d_leading:.4f}")

# Monte Carlo check: simulate the scalar unbiased recursion, T generations, many reps,
# compare mean(log Sigma_t) drift-per-step to the exact digamma prediction, and confirm
# E[Sigma_t] stays ~flat while median/geometric-mean collapses.
rng = np.random.default_rng(0)
n = 200
k = n-1
T = 500
R = 20000
Sigma0 = 1.0
logS = np.zeros(R)
Sigma_lin = np.ones(R)*Sigma0
mean_traj = []
logmean_traj = []
for t in range(T):
    chi2 = rng.chisquare(k, size=R)
    ratio = chi2/k
    logS = logS + np.log(ratio)
    Sigma_lin = Sigma_lin*ratio
    if t % 100 == 99:
        mean_traj.append(Sigma_lin.mean())
        logmean_traj.append(logS.mean())

print("\nE[Sigma_t] over time (should stay ~1.0, unbiased):", mean_traj)
print("mean(log Sigma_t) over time (should be ~ t*drift, i.e. linearly decreasing):", logmean_traj)
print("expected mean(log Sigma_t) at checkpoints:", [ (t+1)*100*drift_exact(k) for t in range(len(logmean_traj))])
