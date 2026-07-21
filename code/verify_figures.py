"""Independently recomputes, straight from the raw shard CSVs, the specific
numbers each of the four figures (fig_traj, fig_axes, fig_nsweep, fig_regime)
claims, and prints them as a table for a by-eye cross-check against the
plots and against results/analysis_thresholds.csv / code/analyze.py.

This file does NOT import make_figures.py's plotting code -- only its
load_data/degradation/alpha_star helpers, which mirror code/analyze.py
exactly -- so the numbers below are a genuine independent recomputation of
what ends up on the page, not a re-display of the same call path.
"""
import numpy as np
import pandas as pd

from make_figures import (AXES, BETA_FIT, BETA_FIT_CI, REF_N, alpha_star, common_tolerance_alpha_star,
                           degradation, load_data, load_main_csv)


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    d = load_data()
    print(f"[verify] loaded {len(d)} rows after cleanup/dedup")

    copula = d[d["synth"] == "gaussian_copula"]

    # (a) adult n=2000 fixed, tstr_auc at generation 0 and 12 for alpha=0 and alpha=1
    section("(a) fig_traj: adult n=2000 fixed -- tstr_auc at gen 0 / gen 12, alpha=0 / alpha=1")
    sub = copula[(copula["dataset"] == "adult") & (copula["n"] == 2000) & (copula["regime"] == "fixed")]
    rows = []
    for a in (0.0, 1.0):
        for gen in (0, 12):
            g = sub[(sub["alpha"] == a) & (sub["generation"] == gen)]
            rows.append({"alpha": a, "generation": gen, "tstr_auc_mean": g["tstr_auc"].mean(),
                         "n_seeds": g["seed"].nunique()})
    a_tab = pd.DataFrame(rows)
    print(a_tab.to_string(index=False))
    trtr = sub["trtr_auc"].mean()
    print(f"trtr_auc ceiling (adult, n=2000, fixed): {trtr:.4f}")
    print("Expect: alpha=0 drops from gen0 to gen12; alpha=1 stays ~flat and close to trtr_auc.")

    # (b) terminal var_ratio and corr_frob at alpha=0, all three datasets, n=2000 fixed
    section("(b) fig_axes/fig_traj context: terminal (gen=12) var_ratio & corr_frob at alpha=0, n=2000 fixed")
    rows = []
    for ds in ("adult", "bank-marketing", "credit-g"):
        g = copula[(copula["dataset"] == ds) & (copula["n"] == 2000) & (copula["regime"] == "fixed") &
                   (copula["alpha"] == 0.0) & (copula["generation"] == 12)]
        rows.append({"dataset": ds, "var_ratio": g["var_ratio"].mean(),
                     "corr_frob": g["corr_frob"].mean(), "n_seeds": g["seed"].nunique()})
    b_tab = pd.DataFrame(rows)
    print(b_tab.to_string(index=False))
    print("Expect: var_ratio stays near 1.0 (does NOT collapse) even at alpha=0; corr_frob moves further from 0.")

    # (c) per-axis alpha* for adult n=2000 fixed (eps = 25% of worst observed degradation,
    #     single terminal generation -- identical definition to code/analyze.py)
    section("(c) fig_axes: per-axis alpha* for adult, gaussian_copula, n=2000, fixed (eps=25% of worst)")
    sub2000 = copula[(copula["dataset"] == "adult") & (copula["n"] == 2000) & (copula["regime"] == "fixed")]
    rows = []
    for axis in AXES:
        deg = degradation(sub2000, axis)
        if deg is None:
            continue
        scale = deg.max()
        astar = alpha_star(deg, 0.25 * scale) if np.isfinite(scale) and scale > 0 else np.nan
        rows.append({"axis": axis, "worst_degradation": scale, "alpha_star": astar})
    c_tab = pd.DataFrame(rows)
    print(c_tab.to_string(index=False))
    print("Compare directly to results/analysis_thresholds.csv / code/analyze.py's own printed table "
          "(dataset=adult, n=2000, fixed, eps_frac_of_worst=0.25) -- should match to rounding.")

    # ---- everything below uses results/main.csv (7 n-values; 15 seeds at
    # n=2000), NOT the frozen shards glob above -- fig_nsweep/fig_regime read
    # main.csv, so their independent check must too.
    d_main = load_main_csv()
    print(f"\n[verify] loaded {len(d_main)} rows from results/main.csv after cleanup/dedup")
    copula_main = d_main[d_main["synth"] == "gaussian_copula"]

    # (c-extra) fig_nsweep: per-axis alpha* across n in {250,500,1000,2000,4000,8000,16000},
    # using a COMMON absolute tolerance per axis (25% of the n=250 worst
    # degradation, applied unchanged at every n) -- NOT analyze.py's
    # per-config-relative tolerance, which silently tightens the absolute bar
    # as n grows and can make alpha* rise with n as a pure definitional
    # artifact (this is what produced the earlier "n=8000 gets worse"
    # anomaly). Mirrors code/analyze_nscaling.py exactly.
    section("(c-extra) fig_nsweep: per-axis alpha* vs n (7 n-values), COMMON absolute tolerance")
    per_axis, eps_abs, ns = common_tolerance_alpha_star(copula_main)
    rows = []
    for axis in AXES:
        row = {"axis": axis, "eps_abs": eps_abs.get(axis, np.nan)}
        for n_val in ns:
            row[n_val] = per_axis.get(axis, {}).get(n_val, np.nan)
        rows.append(row)
    c2_tab = pd.DataFrame(rows).set_index("axis")
    print(c2_tab.round(4).to_string())
    print("\n(0.0 is a real value here: already within tolerance at alpha=0, not missing data.)")

    mono = []
    for axis in AXES:
        vals = [per_axis.get(axis, {}).get(n_val) for n_val in ns]
        if all(v is not None and np.isfinite(v) for v in vals):
            mono.append((axis, "falls" if all(np.diff(vals) <= 1e-9) else "not monotone"))
    n_falls = sum(1 for _, m in mono if m == "falls")
    print(f"\nmonotonicity (strict, all 7 points): {n_falls}/{len(mono)} axes fall monotonically "
          f"({', '.join(a for a, m in mono if m == 'not monotone')} do not; corr_frob is flat, not "
          f"falling, by design)")

    # (a) pooled power-law fit: alpha* ~ n^-beta, log-log OLS pooled over
    # axis/n pairs with alpha*>0 (log undefined at 0). The fit EXCLUDES
    # var_ratio and tstr_auc: Section res-variance establishes these are not
    # copula collapse axes (var_ratio is noise), so including them pollutes a
    # scaling fit. Pooling only the five carried collapse axes reproduces the
    # plotted beta=0.26 [0.07,0.54]; pooling all seven wrongly yields ~0.21.
    FIT_AXES = ["w1", "tv_cat", "corr_frob", "cat_support", "c2st_auc"]
    section("(a) pooled power-law fit alpha* ~ n^-beta over 5 collapse axes (cf. plotted beta=0.26)")
    pts = [(np.log(n_val), np.log(per_axis[axis][n_val]))
           for axis in FIT_AXES for n_val in ns
           if n_val in per_axis.get(axis, {}) and np.isfinite(per_axis[axis][n_val]) and per_axis[axis][n_val] > 0]
    xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
    slope, _ = np.polyfit(xs, ys, 1)
    beta_hat = -slope
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(5000):
        resampled_axes = rng.choice(FIT_AXES, size=len(FIT_AXES), replace=True)
        xs2, ys2 = [], []
        for axis in resampled_axes:
            for n_val in ns:
                v = per_axis.get(axis, {}).get(n_val)
                if v is not None and np.isfinite(v) and v > 0:
                    xs2.append(np.log(n_val)); ys2.append(np.log(v))
        if len(set(xs2)) < 2:
            continue
        s, _ = np.polyfit(xs2, ys2, 1)
        boots.append(-s)
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"independent re-fit: beta={beta_hat:.3f} over {len(pts)} (axis, n) points; "
          f"axis-resampling bootstrap 95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"plotted in fig_nsweep.pdf (externally supplied): beta={BETA_FIT} "
          f"CI [{BETA_FIT_CI[0]}, {BETA_FIT_CI[1]}]")
    print("Expect: both CIs sit well below 1.0 and exclude the theoretical beta=1 (1/n) law; "
          "exact values need not match (different bootstrap procedures) but should be the same ballpark.")

    # (b) fig_regime: adult n=2000 cat_support at gen=12, fixed vs fresh, 15 seeds/regime
    section("(b) fig_regime: adult n=2000 cat_support at gen=12, fixed vs fresh (15 seeds/regime)")
    try:
        from scipy import stats
        have_scipy = True
    except ImportError:
        have_scipy = False
    sub_regime = copula_main[(copula_main["dataset"] == "adult") & (copula_main["n"] == 2000) &
                              (copula_main["generation"] == 12)]
    rows = []
    for a in (0.0, 0.01, 0.05, 0.10, 0.20):
        fx = sub_regime[(sub_regime["alpha"] == a) & (sub_regime["regime"] == "fixed")]["cat_support"]
        fr = sub_regime[(sub_regime["alpha"] == a) & (sub_regime["regime"] == "fresh")]["cat_support"]
        row = {"alpha": a, "cat_support_fixed": fx.mean(), "cat_support_fresh": fr.mean(),
               "fresh_minus_fixed": fr.mean() - fx.mean(), "n_fixed": len(fx), "n_fresh": len(fr)}
        if have_scipy and len(fx) > 1 and len(fr) > 1:
            _, row["p_value"] = stats.ttest_ind(fx, fr, equal_var=False)
        rows.append(row)
    d_tab = pd.DataFrame(rows)
    print(d_tab.to_string(index=False))
    print("Expect: fresh_minus_fixed ~ 0 and NOT statistically significant (large p) at alpha in {0, 0.01} "
          "-- NO reversal, the old 3-seed dip was noise; fresh_minus_fixed > 0 and significant (small p) "
          "at alpha >= 0.05.")


if __name__ == "__main__":
    main()
