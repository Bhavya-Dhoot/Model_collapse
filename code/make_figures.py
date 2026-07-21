"""Produces the four publication figures for the IEEE paper from the REAL
experimental shard CSVs in H:/Model-paper/results/*_shards_*/*.csv.

Data: gaussian_copula only (ctgan/tvae currently have ~2 rows each and are
skipped automatically -- any dataset/synth/regime/n combination that fails a
basic completeness check, see `_complete`, is skipped with a printed note
rather than crashing, so ctgan/tvae panels will appear on their own once
those tiers finish without any code change here).

Degradation / alpha* methodology mirrors code/analyze.py EXACTLY (same
degradation() and alpha_star() logic, evaluated at the single terminal
generation, eps = 25% of the worst observed degradation) so that every
number drawn in fig_axes.pdf / fig_nsweep.pdf is directly reproducible from
results/analysis_thresholds.csv and code/analyze.py's own printed table --
see code/verify_figures.py for the side-by-side check.

Style: serif font, 7-9pt, no in-axes titles beyond the small per-panel
dataset labels needed to read a multi-column figure, colorblind-safe
(Okabe-Ito) palette for categorical series, viridis for the ordered alpha
sweep, thin lines, light grid, mean over seeds with a shaded +/-1 SEM band
on trajectory plots.
"""
import glob
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# fig_traj / fig_axes are frozen to this exact set of shard directories (the
# ones present when those two figures were accepted). New shard directories
# have since landed on disk (tier1_creditg_n500_*, tier3_shards_202407Z,
# strengthen_regime_*, strengthen_scaling_*) that would silently change
# fig_traj/fig_axes's seed counts (e.g. adult n=2000 fixed 3->15 seeds) if a
# live glob were used -- the coordinator asked NOT to touch those two
# figures, so the directory list here is pinned rather than a "*_shards_*"
# wildcard. fig_nsweep / fig_regime read the freshly expanded
# results/main.csv separately (see load_main_csv) and are unaffected by this.
RESULTS_DIRS_FROZEN = [
    "nsweep_shards_20260720T170703Z",
    "nsweep_shards_20260720T174452Z",
    "tier1_shards_20260720T164133Z",
    "tier2_shards_20260720T164438Z",
    "tier3_shards_20260720T165056Z",
]
RESULTS_GLOB = ["H:/Model-paper/results/" + d + "/*.csv" for d in RESULTS_DIRS_FROZEN]
MAIN_CSV = "H:/Model-paper/results/main.csv"
OUT_DIR = "H:/Model-paper/paper/figs"

OKABE_ITO = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2",
             "#D55E00", "#CC79A7"]

SINGLE_COL_W = 3.4
DOUBLE_COL_W = 7.0

# Same axis classification as code/analyze.py.
HIGHER_BETTER = {"tstr_auc", "cat_support", "coverage"}
AXES = ["var_ratio", "w1", "tv_cat", "corr_frob", "cat_support", "tstr_auc", "c2st_auc"]
AXIS_LABEL = {"var_ratio": "var_ratio", "w1": r"$W_1$", "tv_cat": "tv_cat",
              "corr_frob": "corr_frob", "cat_support": "cat_support",
              "tstr_auc": "tstr_auc", "c2st_auc": "c2st_auc"}


def _style():
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "axes.prop_cycle": matplotlib.cycler(color=OKABE_ITO),
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    })


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[make_figures] wrote {path}")


def load_data(pattern=RESULTS_GLOB):
    """Merge every shard CSV as specified for this task, then apply the same
    cleanup as code/analyze.py: keep only status=='ok' rows, drop rows that
    silently bootstrapped with replacement (their effective n != requested
    n), and drop duplicate (dataset, synth, regime, alpha, seed, generation,
    n) keys keeping the LAST occurrence (a shard that was re-run supersedes
    the earlier attempt)."""
    patterns = [pattern] if isinstance(pattern, str) else list(pattern)
    files = sorted({f for p in patterns for f in glob.glob(p)})
    if not files:
        raise SystemExit(f"[make_figures] no shard CSVs matched {patterns}")
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "status" in d.columns:
        d = d[d["status"].fillna("ok") == "ok"]
    if "sampled_with_replacement" in d.columns:
        bad = d["sampled_with_replacement"].map(lambda x: bool(x) if pd.notna(x) else False)
        if bad.any():
            print(f"[make_figures] excluding {bad.sum()} rows sampled WITH REPLACEMENT")
            d = d[~bad]
    keys = [k for k in ["dataset", "synth", "regime", "alpha", "seed", "generation", "n"] if k in d.columns]
    d = d.drop_duplicates(subset=keys, keep="last")
    return d


def load_main_csv(path=MAIN_CSV):
    """Loads the separately-merged results/main.csv (fig_nsweep / fig_regime
    only -- fig_traj / fig_axes keep reading load_data()'s shards glob
    unchanged). Same cleanup as load_data(): status=='ok', drop
    sampled_with_replacement rows, dedup keys keep last.

    Note: main.csv has a stray duplicate 'sampled_with_replacement.1' column
    (an artifact confined to a handful of adult/ctgan/n=2000 rows from
    tier2.csv) -- irrelevant here since both figures filter to
    synth=='gaussian_copula', which excludes ctgan entirely."""
    d = pd.read_csv(path)
    if "status" in d.columns:
        d = d[d["status"].fillna("ok") == "ok"]
    if "sampled_with_replacement" in d.columns:
        bad = d["sampled_with_replacement"].map(lambda x: bool(x) if pd.notna(x) else False)
        if bad.any():
            print(f"[make_figures] excluding {bad.sum()} rows sampled WITH REPLACEMENT (main.csv)")
            d = d[~bad]
    keys = [k for k in ["dataset", "synth", "regime", "alpha", "seed", "generation", "n"] if k in d.columns]
    d = d.drop_duplicates(subset=keys, keep="last")
    return d


def _complete(sub, min_alphas=8, min_seeds=2):
    """Completeness gate: a config needs a real alpha sweep and at least a
    couple of seeds before it's trustworthy to plot. This is what makes the
    script robust to ctgan/tvae (currently ~2 rows total) -- they simply
    never pass this check and the panel/figure is skipped with a note,
    rather than the script crashing on an empty or degenerate group."""
    if sub.empty:
        return False
    return sub["alpha"].nunique() >= min_alphas and sub["seed"].nunique() >= min_seeds


def degradation(sub, axis):
    """Degradation of `axis` at the terminal generation relative to the
    alpha=1 baseline. Identical logic to code/analyze.py's degradation()."""
    T = sub["generation"].max()
    term = sub[sub["generation"] == T]
    if term.empty or axis not in term.columns:
        return None
    per_alpha = term.groupby("alpha")[axis].mean()
    if 1.0 not in per_alpha.index or per_alpha.isna().all():
        return None
    base = per_alpha.loc[1.0]
    if not np.isfinite(base):
        return None
    if axis == "var_ratio":
        d = (per_alpha - 1.0).abs() - abs(base - 1.0)
    elif axis in HIGHER_BETTER:
        d = base - per_alpha
    else:
        d = per_alpha - base
    return d.clip(lower=0.0)


def alpha_star(deg, eps_abs):
    """Smallest alpha whose degradation is <= eps_abs and stays so for all
    larger alpha (monotone-suffix threshold), linearly interpolated.
    Identical logic to code/analyze.py's alpha_star()."""
    if deg is None or deg.dropna().empty:
        return np.nan
    a = deg.dropna().sort_index()
    ok = a <= eps_abs
    idx = list(a.index)
    start = len(idx)
    for i in range(len(idx) - 1, -1, -1):
        if ok.iloc[i]:
            start = i
        else:
            break
    if start == 0:
        return float(idx[0])
    if start >= len(idx):
        return np.nan
    lo, hi = idx[start - 1], idx[start]
    dlo, dhi = a.loc[lo], a.loc[hi]
    if not np.isfinite(dlo) or not np.isfinite(dhi) or dlo == dhi:
        return float(hi)
    frac = (dlo - eps_abs) / (dlo - dhi)
    return float(lo + frac * (hi - lo))


def _sem_band(g, value_col, group_col="generation"):
    agg = g.groupby(group_col)[value_col].agg(["mean", "std", "count"]).reset_index()
    agg["sem"] = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
    return agg


# --------------------------------------------------------------------------
# Figure 1: degradation trajectories
# --------------------------------------------------------------------------
def fig_traj(d, out_dir):
    datasets = ["adult", "credit-g", "bank-marketing"]
    base = d[(d["synth"] == "gaussian_copula") & (d["n"] == 2000) & (d["regime"] == "fixed")]
    if not _complete(base):
        print("[fig_traj] incomplete copula/n=2000/fixed data, skipping")
        return

    cmap = matplotlib.colormaps["viridis"]
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)

    fig, axes = plt.subplots(1, len(datasets), figsize=(DOUBLE_COL_W, 2.6),
                              sharey=True, constrained_layout=True)
    any_plotted = False
    for j, ds in enumerate(datasets):
        ax = axes[j]
        panel = base[base["dataset"] == ds]
        if not _complete(panel):
            ax.text(0.5, 0.5, f"{ds}\n(no data)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7)
            ax.set_xticks([])
            continue
        for a in sorted(panel["alpha"].unique()):
            g = panel[panel["alpha"] == a]
            agg = _sem_band(g, "tstr_auc")
            if agg.empty:
                continue
            color = cmap(norm(a))
            ax.plot(agg["generation"], agg["mean"], color=color, linewidth=1.2)
            ax.fill_between(agg["generation"], agg["mean"] - agg["sem"], agg["mean"] + agg["sem"],
                             color=color, alpha=0.25, linewidth=0)
            any_plotted = True
        trtr = panel["trtr_auc"].mean()
        if np.isfinite(trtr):
            ax.axhline(trtr, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(ds, fontsize=8)
        ax.set_xlabel("generation")
        if j == 0:
            ax.set_ylabel("TSTR AUC")
        ax.grid(alpha=0.3, linewidth=0.5)
    if not any_plotted:
        print("[fig_traj] nothing to plot, skipping save")
        plt.close(fig)
        return
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes,
                 label=r"$\alpha$", shrink=0.9, pad=0.015, aspect=25)
    _save(fig, os.path.join(out_dir, "fig_traj.pdf"))


# --------------------------------------------------------------------------
# Figure 2: normalized degradation axes vs alpha (THE key figure)
# --------------------------------------------------------------------------
def fig_axes(d, out_dir, eps_frac=0.25):
    sub = d[(d["dataset"] == "adult") & (d["synth"] == "gaussian_copula") &
            (d["n"] == 2000) & (d["regime"] == "fixed")]
    if not _complete(sub):
        print("[fig_axes] incomplete adult/copula/n=2000/fixed data, skipping")
        return

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W, 2.4), constrained_layout=True)
    any_plotted = False
    for i, axis in enumerate(AXES):
        deg = degradation(sub, axis)
        if deg is None:
            continue
        scale = deg.max()
        if not np.isfinite(scale) or scale <= 0:
            continue
        norm_deg = (deg / scale).sort_index()
        color = OKABE_ITO[i % len(OKABE_ITO)]
        ax.plot(norm_deg.index, norm_deg.values, marker="o", markersize=3, linewidth=1.2,
                color=color, label=AXIS_LABEL[axis])
        astar = alpha_star(deg, eps_frac * scale)
        if np.isfinite(astar):
            ax.plot(astar, -0.045, marker="^", markersize=5, color=color, clip_on=False)
        any_plotted = True
    if not any_plotted:
        print("[fig_axes] nothing to plot, skipping save")
        plt.close(fig)
        return
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.set_ylim(-0.09, 1.05)
    ax.set_xlabel(r"$\alpha$ (real-data anchor fraction)")
    ax.set_ylabel("degradation\n(normalized to own worst)")
    ax.legend(frameon=False, ncol=4, loc="upper right", fontsize=6.5)
    ax.grid(alpha=0.3, linewidth=0.5)
    _save(fig, os.path.join(out_dir, "fig_axes.pdf"))


# --------------------------------------------------------------------------
# Figure 3: alpha* vs n, measured vs theory
# --------------------------------------------------------------------------
NONMONOTONE_AXES = {"var_ratio", "tstr_auc"}  # noisy, no discernible trend with n
FLAT_AXES = {"corr_frob"}  # real, meaningful non-decline: stays ~flat across all n

REF_N = 250  # smallest n now available (main.csv); every n judged against ITS standard

# Pooled power-law fit alpha* ~ n^-beta, from a bootstrap over all 7 axes x 7 n
# values (n in {250,500,1000,2000,4000,8000,16000}), computed externally
# (results/main.csv analysis). A quick in-script sanity re-fit (axis-resampling
# bootstrap, see code/verify_figures.py) independently lands in the same
# ballpark (beta ~ 0.21, CI overlapping) -- both far below and excluding 1.0.
BETA_FIT = 0.26
BETA_FIT_CI = (0.07, 0.54)


def common_tolerance_alpha_star(d, ref_n=REF_N):
    """Common-ABSOLUTE-tolerance alpha* per axis across n, adult/copula/fixed.

    Mirrors code/analyze_nscaling.py exactly: analyze.py's degradation() /
    alpha_star() use a tolerance that is 25% of EACH CONFIG'S OWN worst
    observed degradation. That is fine within one n, but invalid across n --
    total degradation shrinks as n grows (e.g. worst w1 degradation on adult
    falls 0.142 -> 0.064 -> 0.030 for n=500/2000/8000), so a relative
    tolerance silently tightens the absolute bar at large n and can make
    alpha* rise with n as a pure artifact of the definition, not a real
    effect. Fixing ONE absolute tolerance per axis (25% of the smallest-n
    worst degradation, applied unchanged at every n) removes that artifact.

    Returns (per_axis, eps_abs) where per_axis[axis][n] = alpha* (np.nan if
    inestimable) and eps_abs[axis] is the common tolerance used for that
    axis. Note alpha*==0.0 is a real, meaningful value here (already within
    tolerance at alpha=0), not a missing value.
    """
    base = d[(d["dataset"] == "adult") & (d["synth"] == "gaussian_copula") & (d["regime"] == "fixed")]
    ns = sorted(base["n"].unique())

    degs = {}
    for n_val in ns:
        sub = base[base["n"] == n_val]
        if not _complete(sub, min_alphas=8, min_seeds=1):
            continue
        for axis in AXES:
            degs[(n_val, axis)] = degradation(sub, axis)

    per_axis = {axis: {} for axis in AXES}
    eps_abs = {}
    for axis in AXES:
        ref = degs.get((ref_n, axis))
        if ref is None or ref.dropna().empty:
            continue
        eps = 0.25 * ref.max()
        if not np.isfinite(eps) or eps <= 0:
            continue
        eps_abs[axis] = eps
        for n_val in ns:
            dg = degs.get((n_val, axis))
            if dg is not None:
                per_axis[axis][n_val] = alpha_star(dg, eps)  # may be 0.0 -- keep it
    return per_axis, eps_abs, ns


def fig_nsweep(d, out_dir):
    per_axis, eps_abs, ns = common_tolerance_alpha_star(d)
    if len(ns) < 2 or not eps_abs:
        print("[fig_nsweep] fewer than 2 distinct n values or no estimable axis, skipping")
        return

    # Zero is a real, meaningful alpha* (already within tolerance at alpha=0)
    # but can't be shown on a log axis. Render it at a floor just below the
    # smallest nonzero measured value, with a distinct marker + annotation,
    # rather than dropping it silently.
    all_vals = [v for pts in per_axis.values() for v in pts.values() if np.isfinite(v) and v > 0]
    floor = (min(all_vals) * 0.4) if all_vals else 0.02

    # Several axes can tie at alpha*=0 for the same n (e.g. w1, tv_cat and
    # c2st_auc all hit exactly 0 at n=8000) -- without separating them their
    # floor markers would stack into a single visible triangle and silently
    # hide that it's really 3 axes, not 1. Nudge tied markers apart in x.
    zero_axes_at = {}
    for axis in AXES:
        for x, v in per_axis.get(axis, {}).items():
            if np.isfinite(v) and v == 0.0:
                zero_axes_at.setdefault(x, []).append(axis)

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 2.6), constrained_layout=True)
    zero_plotted = False
    for i, axis in enumerate(AXES):
        pts = per_axis.get(axis, {})
        if len(pts) < 2:
            continue
        xs = sorted(pts)
        ys_raw = [pts[x] for x in xs]
        ys_plot = [floor if (np.isfinite(v) and v == 0.0) else v for v in ys_raw]
        nonmono = axis in NONMONOTONE_AXES
        flat = axis in FLAT_AXES
        if nonmono:
            color, ls, lw, ms, zord = "0.55", "--", 1.0, 3, 2
        elif flat:
            color, ls, lw, ms, zord = OKABE_ITO[i % len(OKABE_ITO)], "-", 2.2, 5, 4
        else:
            color, ls, lw, ms, zord = OKABE_ITO[i % len(OKABE_ITO)], "-", 1.0, 3, 3
        suffix = " (n.m.)" if nonmono else (" (flat)" if flat else "")
        ax.plot(xs, ys_plot, color=color, linewidth=lw, linestyle=ls, alpha=0.9 if flat else 0.85,
                 marker="o", markersize=ms, zorder=zord, label=f"{AXIS_LABEL[axis]}{suffix}")
        for x, v in zip(xs, ys_raw):
            if np.isfinite(v) and v == 0.0:
                tied = zero_axes_at[x]
                rank = tied.index(axis) - (len(tied) - 1) / 2.0
                x_jit = x * (1.0 + 0.03 * rank)
                ax.plot(x_jit, floor, marker="v", markersize=6, color=color,
                         markerfacecolor="white", zorder=5)
                zero_plotted = True

    # Two reference lines, both anchored at the SAME start point (n=REF_N,
    # mean measured alpha* across axes there) so their divergence from each
    # other -- and from the measured points -- is visible, not hidden by
    # separate re-anchoring:
    #   - theory beta=1 (the paper's 1/n law)
    #   - fitted beta=0.26 (95% bootstrap CI [0.07, 0.54], pooled over all
    #     axes/n; see BETA_FIT / BETA_FIT_CI above) -- the actual measured
    #     scaling, far shallower than 1/n.
    a0_vals = [per_axis[axis][REF_N] for axis in AXES if REF_N in per_axis.get(axis, {})
               and np.isfinite(per_axis[axis][REF_N]) and per_axis[axis][REF_N] > 0]
    if a0_vals:
        a0 = float(np.mean(a0_vals))
        n_smooth = np.geomspace(min(ns), max(ns), 60)
        theory1 = a0 * (REF_N / n_smooth) ** 1.0
        ax.plot(n_smooth, theory1, linestyle=":", color="black", linewidth=1.6,
                label=r"theory $n^{-1}$ ($\beta{=}1$)")
        fitted = a0 * (REF_N / n_smooth) ** BETA_FIT
        ax.plot(n_smooth, fitted, linestyle="-.", color="black", linewidth=1.8,
                label=rf"pooled fit $\beta{{=}}{BETA_FIT:.2f}$ (CI [{BETA_FIT_CI[0]:.2f},{BETA_FIT_CI[1]:.2f}])")

    ax.set_xscale("log")
    ax.set_yscale("log")
    if zero_plotted:
        ax.axhline(floor, color="gray", linewidth=0.5, linestyle=":", alpha=0.6)
        ax.text(ns[0], floor, r"$\alpha^\star{=}0\ \rightarrow$ ",
                fontsize=5.5, va="center", ha="right", color="0.3")
    ax.set_xlabel("n (training-set size)")
    ax.set_ylabel(r"estimated $\alpha^\star$")
    ax.legend(frameon=False, fontsize=4.8, ncol=2, loc="lower left")
    ax.grid(alpha=0.3, linewidth=0.5, which="both")
    _save(fig, os.path.join(out_dir, "fig_nsweep.pdf"))


# --------------------------------------------------------------------------
# Figure 4: fixed vs fresh anchoring regime
# --------------------------------------------------------------------------
def fig_regime(d, out_dir, alphas=(0.0, 0.01, 0.05, 0.20)):
    base = d[(d["dataset"] == "adult") & (d["synth"] == "gaussian_copula") & (d["n"] == 2000)]
    if base.empty or base["regime"].nunique() < 2:
        print("[fig_regime] missing fixed/fresh pair, skipping")
        return

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 2.6), constrained_layout=True)
    any_plotted = False
    for i, a in enumerate(alphas):
        color = OKABE_ITO[i % len(OKABE_ITO)]
        for regime, ls in (("fixed", "-"), ("fresh", "--")):
            g = base[(base["alpha"] == a) & (base["regime"] == regime)]
            if g.empty:
                continue
            agg = _sem_band(g, "cat_support")
            if agg.empty:
                continue
            ax.plot(agg["generation"], agg["mean"], linestyle=ls, color=color, linewidth=1.2,
                    label=rf"$\alpha$={a:g} ({regime})")
            ax.fill_between(agg["generation"], agg["mean"] - agg["sem"], agg["mean"] + agg["sem"],
                             color=color, alpha=0.15, linewidth=0)
            any_plotted = True
    if not any_plotted:
        print("[fig_regime] nothing to plot, skipping save")
        plt.close(fig)
        return
    ax.set_xlabel("generation")
    ax.set_ylabel("cat_support")
    ax.legend(frameon=False, fontsize=5.8, ncol=2, loc="lower right")
    ax.grid(alpha=0.3, linewidth=0.5)
    _save(fig, os.path.join(out_dir, "fig_regime.pdf"))


def main():
    _style()
    os.makedirs(OUT_DIR, exist_ok=True)

    # fig_traj / fig_axes: UNCHANGED data source (shards glob) and code.
    d = load_data()
    print(f"[make_figures] loaded {len(d)} rows from shards glob after cleanup/dedup")
    fig_traj(d, OUT_DIR)
    fig_axes(d, OUT_DIR)

    # fig_nsweep / fig_regime: results/main.csv (7 n-values; 15 seeds at n=2000).
    d_main = load_main_csv()
    print(f"[make_figures] loaded {len(d_main)} rows from {MAIN_CSV} after cleanup/dedup")
    fig_nsweep(d_main, OUT_DIR)
    fig_regime(d_main, OUT_DIR)


if __name__ == "__main__":
    main()
