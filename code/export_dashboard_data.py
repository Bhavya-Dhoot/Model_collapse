"""Export every number the demo dashboard shows into one JSON file.

Reuses make_figures.py's loaders and estimators, so the dashboard and the paper
are computed by the same code path -- nothing here is hand-entered.
"""
import json
import os

import numpy as np
import pandas as pd

from make_figures import (AXES, BETA_FIT, BETA_FIT_CI, alpha_star,
                          common_tolerance_alpha_star, degradation, load_main_csv)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "demo", "data.json")
FIT_AXES = ["w1", "tv_cat", "corr_frob", "cat_support", "c2st_auc"]
DATASETS = ["adult", "credit-g", "bank-marketing"]

# The paper reports each experiment on a specific grid, and main.csv now holds
# every seed ever run (15 at low alpha from the fresh-vs-fixed strengthening
# run, 8 from the n-sweep, 3 from the original primary grid). Averaging over
# all of them silently changes the published numbers, so each panel is pinned
# to the grid its section of the paper actually used.
PRIMARY_SEEDS = [0, 1, 2]  # Results VI-A..VI-D: "the mean over three seeds"


def jnum(x):
    """JSON-safe float (NaN -> None), rounded to a sane number of digits."""
    if x is None:
        return None
    x = float(x)
    return None if not np.isfinite(x) else round(x, 6)


def traj_block(d, dataset, n=2000, regime="fixed", synth="gaussian_copula"):
    """mean and standard error of every axis, per (alpha, generation)."""
    sub = d[(d.dataset == dataset) & (d.n == n) & (d.regime == regime) & (d.synth == synth)
            & (d.seed.isin(PRIMARY_SEEDS))]
    out = {}
    for a, ga in sub.groupby("alpha"):
        rows = []
        for gen, gg in ga.groupby("generation"):
            rec = {"g": int(gen), "seeds": int(gg.seed.nunique())}
            for ax in AXES:
                v = gg[ax].dropna()
                rec[ax] = jnum(v.mean()) if len(v) else None
                rec[ax + "_se"] = jnum(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
            rows.append(rec)
        out[f"{a:g}"] = rows
    return out


def main():
    d = load_main_csv()
    copula = d[d.synth == "gaussian_copula"]
    primary = copula[copula.seed.isin(PRIMARY_SEEDS)]  # paper's Results VI-A..VI-D grid
    T = int(d.generation.max())
    data = {"meta": {"rows_total": int(len(d)), "terminal_generation": T,
                     "seeds_max": int(d.seed.nunique()),
                     "primary_seeds": len(PRIMARY_SEEDS),
                     "alphas": sorted(float(a) for a in d.alpha.unique()),
                     "ns": sorted(int(x) for x in d.n.unique())}}

    # ---- 1. trajectories (copula, n=2000, fixed): the alpha slider ----------
    data["traj"] = {ds: traj_block(d, ds) for ds in DATASETS}
    data["ceiling"] = {}
    for ds in DATASETS:
        sub = primary[(primary.dataset == ds) & (primary.n == 2000) & (primary.regime == "fixed")]
        data["ceiling"][ds] = jnum(sub["trtr_auc"].mean())

    # ---- 2. per-axis alpha* (paper Table I(a)) ------------------------------
    rows = []
    for ds, n in [("adult", 500), ("adult", 2000), ("bank-marketing", 2000), ("credit-g", 2000)]:
        sub = primary[(primary.dataset == ds) & (primary.n == n) & (primary.regime == "fixed")]
        rec = {"dataset": ds, "n": n}
        vals, vals_no_var = {}, {}
        for ax in AXES:
            deg = degradation(sub, ax)
            if deg is None:
                rec[ax] = None
                continue
            scale = deg.max()
            a = alpha_star(deg, 0.25 * scale) if np.isfinite(scale) and scale > 0 else np.nan
            rec[ax] = jnum(a)
            if np.isfinite(a):
                vals[ax] = a
                if ax != "var_ratio":
                    vals_no_var[ax] = a
        # paper Table I's "spread" column spans every axis; the headline 2.7x
        # claim excludes var_ratio, which is not a copula collapse axis
        rec["spread"] = jnum(max(vals.values()) - min(vals.values())) if vals else None
        if vals_no_var:
            lo, hi = min(vals_no_var.values()), max(vals_no_var.values())
            rec["spread_excl_var"] = jnum(hi - lo)
            rec["ratio_excl_var"] = jnum(hi / lo) if lo > 0 else None
            rec["binding_axis"] = max(vals_no_var, key=vals_no_var.get)
        rows.append(rec)
    data["alpha_star_table"] = rows

    # degradation curves behind the alpha* estimate (adult n=2000): axis vs alpha
    sub = primary[(primary.dataset == "adult") & (primary.n == 2000) & (primary.regime == "fixed")]
    curves = {}
    for ax in AXES:
        deg = degradation(sub, ax)
        if deg is None:
            continue
        scale = float(deg.max())
        curves[ax] = {"points": [{"alpha": jnum(a), "deg": jnum(v)} for a, v in deg.items()],
                      "tol": jnum(0.25 * scale),
                      "alpha_star": jnum(alpha_star(deg, 0.25 * scale) if scale > 0 else np.nan)}
    data["degradation_curves"] = curves

    # ---- 3. n-scaling: alpha* vs n, common absolute tolerance ---------------
    per_axis, eps_abs, ns = common_tolerance_alpha_star(copula)
    data["nscaling"] = {
        "ns": [int(x) for x in ns],
        "beta": BETA_FIT, "beta_ci": list(BETA_FIT_CI),
        "fit_axes": FIT_AXES,
        "axes": {ax: {"eps_abs": jnum(eps_abs.get(ax)),
                      "alpha_star": [jnum(per_axis.get(ax, {}).get(n)) for n in ns]}
                 for ax in AXES},
    }

    # ---- 4. fresh vs fixed anchoring on categorical support -----------------
    try:
        from scipy import stats
        have_scipy = True
    except ImportError:
        have_scipy = False
    reg = copula[(copula.dataset == "adult") & (copula.n == 2000) & (copula.generation == T)]
    rows = []
    for a in sorted(reg.alpha.unique()):
        # paired over seeds, as the paper reports: the same seed under both regimes
        fx = reg[(reg.alpha == a) & (reg.regime == "fixed")][["seed", "cat_support"]].set_index("seed")
        fr = reg[(reg.alpha == a) & (reg.regime == "fresh")][["seed", "cat_support"]].set_index("seed")
        j = fx.join(fr, lsuffix="_fx", rsuffix="_fr").dropna()
        if len(j) < 2:
            continue
        x, y = j["cat_support_fx"], j["cat_support_fr"]
        rec = {"alpha": jnum(a), "fixed": jnum(x.mean()), "fresh": jnum(y.mean()),
               "diff": jnum(y.mean() - x.mean()), "seeds": int(len(j)),
               "fixed_se": jnum(x.std(ddof=1) / np.sqrt(len(x))),
               "fresh_se": jnum(y.std(ddof=1) / np.sqrt(len(y)))}
        if have_scipy:
            rec["p"] = jnum(stats.ttest_rel(y, x).pvalue)
        rows.append(rec)
    data["regime"] = rows

    # ---- 5. cross-architecture (adult, fixed, n=2000) -----------------------
    arch = []
    for synth in ["gaussian_copula", "tvae", "ctgan"]:
        sub = d[(d.synth == synth) & (d.dataset == "adult") & (d.n == 2000) & (d.regime == "fixed")]
        if sub.empty:
            continue
        Ts = int(sub.generation.max())
        rec = {"synth": synth, "terminal_generation": Ts}
        for a in (0.0, 1.0):
            g0 = sub[(sub.alpha == a) & (sub.generation == 0)]
            gT = sub[(sub.alpha == a) & (sub.generation == Ts)]
            key = "a0" if a == 0.0 else "a1"
            rec[key] = {ax: {"start": jnum(g0[ax].mean()), "end": jnum(gT[ax].mean())}
                        for ax in ["cat_support", "var_ratio", "tstr_auc"]}
        arch.append(rec)
    data["architectures"] = arch

    # ---- 6. generation-0 fidelity gate -------------------------------------
    gate = []
    for synth in ["gaussian_copula", "tvae", "ctgan"]:
        sub = d[(d.synth == synth) & (d.dataset == "adult") & (d.n == 2000) & (d.generation == 0)]
        if sub.empty:
            continue
        gate.append({"synth": synth, "cat_support": jnum(sub.cat_support.mean()),
                     "tv_cat": jnum(sub.tv_cat.mean()), "tstr_auc": jnum(sub.tstr_auc.mean()),
                     "sec_per_gen": jnum(sub.wallclock_sec.mean())})
    data["gen0_gate"] = gate

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    print("wrote", os.path.abspath(OUT), os.path.getsize(OUT), "bytes")
    print("terminal generation:", T, "| datasets:", DATASETS)
    print("beta:", BETA_FIT, BETA_FIT_CI)
    for r in data["alpha_star_table"]:
        print(" alpha*", r["dataset"], r["n"], {k: r[k] for k in ("w1", "corr_frob", "cat_support", "c2st_auc", "spread")})


if __name__ == "__main__":
    main()
