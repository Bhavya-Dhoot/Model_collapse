"""
Analyze the two strengthening runs:
  A) fit a power law alpha* ~ n^(-beta) with a bootstrap CI on beta, to replace
     the loose "1.76x vs 4x" statement with a measured exponent + interval.
  B) test whether fresh anchoring beats fixed on categorical support with enough
     seeds to move the claim from "suggestive" to a real significance statement.
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

from analyze import AXES, alpha_star, degradation

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def load(pattern_files):
    frames = []
    for f in pattern_files:
        if os.path.exists(f):
            try:
                frames.append(pd.read_csv(f))
            except Exception:
                pass
    for d in glob.glob(os.path.join(RESULTS, "*_shards_*", "*.csv")):
        frames.append(pd.read_csv(d))
    d = pd.concat(frames, ignore_index=True)
    keys = ["dataset", "synth", "regime", "alpha", "seed", "generation", "n"]
    d = d.drop_duplicates(subset=[k for k in keys if k in d.columns], keep="last")
    if "sampled_with_replacement" in d.columns:
        swr = d["sampled_with_replacement"].astype(str).str.lower()
        d = d[swr != "true"]
    return d


# ---------- A) scaling power law -------------------------------------------
def scaling(d):
    s = d[(d.synth == "gaussian_copula") & (d.dataset == "adult")
          & (d.regime == "fixed")]
    ns = sorted(s["n"].unique())
    print(f"\n=== SCALING: n = {ns} ===")
    CARRY = ["w1", "tv_cat", "corr_frob", "cat_support", "c2st_auc"]
    # common absolute tolerance per axis, anchored at the smallest n
    ref_n = ns[0]
    eps = {}
    for ax in CARRY:
        dg = degradation(s[s["n"] == ref_n], ax)
        eps[ax] = 0.25 * dg.max() if dg is not None and not dg.dropna().empty else np.nan

    rows = {}
    for ax in CARRY:
        if not np.isfinite(eps[ax]):
            continue
        vals = []
        for n in ns:
            dg = degradation(s[s["n"] == n], ax)
            vals.append(alpha_star(dg, eps[ax]) if dg is not None else np.nan)
        rows[ax] = vals
    tbl = pd.DataFrame(rows, index=ns)
    print(tbl.round(3).to_string())

    # pooled log-log fit over axes with strictly positive, finite alpha*
    xs, ys = [], []
    for ax in rows:
        for n, v in zip(ns, rows[ax]):
            if np.isfinite(v) and v > 0:
                xs.append(np.log(n))
                ys.append(np.log(v))
    xs, ys = np.array(xs), np.array(ys)
    if len(xs) < 3:
        print("too few positive points for a slope")
        return
    beta_hat = -np.polyfit(xs, ys, 1)[0]      # alpha* ~ n^(-beta)
    # bootstrap CI
    rng = np.random.default_rng(0)
    betas = []
    for _ in range(2000):
        idx = rng.integers(0, len(xs), len(xs))
        betas.append(-np.polyfit(xs[idx], ys[idx], 1)[0])
    lo, hi = np.percentile(betas, [2.5, 97.5])
    print(f"\nfitted exponent beta = {beta_hat:.3f}  (95% CI [{lo:.3f}, {hi:.3f}])")
    print(f"theory predicts beta = 1.0 (alpha* ~ 1/n)")
    print(f"=> theory {'EXCLUDED' if hi < 1.0 else 'not excluded'} by the CI; "
          f"observed decline is {'sub-linear' if beta_hat < 1 else 'linear+'} in n")


# ---------- B) fresh vs fixed significance ---------------------------------
def regime(d):
    s = d[(d.synth == "gaussian_copula") & (d.dataset == "adult")
          & (d.n == 2000) & (d.generation == 12)]
    print("\n=== FRESH vs FIXED: cat_support at gen 12, per-seed paired test ===")
    print(f"{'alpha':>6}{'n_seed':>7}{'fixed':>8}{'fresh':>8}{'diff':>8}"
          f"{'t':>7}{'p':>8}")
    for a in sorted(s["alpha"].unique()):
        fx = s[(s.alpha == a) & (s.regime == "fixed")].set_index("seed")["cat_support"]
        fr = s[(s.alpha == a) & (s.regime == "fresh")].set_index("seed")["cat_support"]
        common = fx.index.intersection(fr.index)
        if len(common) < 3:
            continue
        fx, fr = fx.loc[common], fr.loc[common]
        diff = (fr - fx)
        t, p = stats.ttest_rel(fr, fx)
        print(f"{a:>6}{len(common):>7}{fx.mean():>8.3f}{fr.mean():>8.3f}"
              f"{diff.mean():>+8.3f}{t:>7.2f}{p:>8.4f}")


if __name__ == "__main__":
    d = load([os.path.join(RESULTS, f) for f in
              ("strengthen_regime.csv", "strengthen_scaling.csv",
               "main.csv")])
    scaling(d)
    regime(d)
