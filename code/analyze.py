"""
Estimate anchoring thresholds alpha* from the raw sweep results.

This is the paper's core inference step, so it is deliberately explicit about
every judgement call it makes. Run:  python code/analyze.py
Outputs: results/analysis_thresholds.csv, results/analysis_summary.txt
"""
import glob
import os

import numpy as np
import pandas as pd

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

# Metrics where a LARGER value is better (utility-like) vs where SMALLER is
# better (error-like). c2st_auc is error-like: 0.5 = indistinguishable (good),
# 1.0 = trivially separable (bad). var_ratio is special-cased below because its
# ideal is 1.0 from either side.
HIGHER_BETTER = {"tstr_auc", "cat_support", "coverage"}
LOWER_BETTER = {"w1", "tv_cat", "corr_frob", "mean_shift", "c2st_auc"}
AXES = ["var_ratio", "w1", "tv_cat", "corr_frob", "cat_support", "tstr_auc", "c2st_auc"]


def load_all():
    """Merge every shard produced by any tier. Append-only by construction, so
    if a config was re-run we keep the LAST occurrence (later runs supersede)."""
    files = sorted(glob.glob(os.path.join(RESULTS, "*_shards_*", "*.csv")))
    files += sorted(glob.glob(os.path.join(RESULTS, "*.csv")))
    frames = []
    for f in files:
        if os.path.basename(f).startswith("analysis"):
            continue
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if "generation" in d.columns and "alpha" in d.columns:
            d["_src"] = os.path.basename(f)
            frames.append(d)
    if not frames:
        raise SystemExit("no results found yet")
    d = pd.concat(frames, ignore_index=True)
    if "status" in d.columns:                       # drop recorded failures
        d = d[d["status"].fillna("ok") == "ok"]
    keys = ["dataset", "synth", "regime", "alpha", "seed", "generation", "n"]
    keys = [k for k in keys if k in d.columns]
    d = d.drop_duplicates(subset=keys, keep="last")
    # Exclude configs that silently bootstrapped the real pool -- these have a
    # smaller EFFECTIVE n than requested, so plotting them against n is wrong.
    if "sampled_with_replacement" in d.columns:
        bad = d["sampled_with_replacement"].fillna(False).astype(bool)
        if bad.any():
            print(f"[warn] excluding {bad.sum()} rows sampled WITH REPLACEMENT "
                  f"(effective n < requested n)")
            d = d[~bad]
    return d


def degradation(sub, axis):
    """Degradation of `axis` at the terminal generation, relative to the alpha=1
    baseline, in units where 0 = as good as retraining on real data every round.

    Using alpha=1 (not the theoretical ideal) as the reference is deliberate: it
    subtracts off the synthesizer's own irreducible finite-sample error, so what
    remains is attributable to self-consumption rather than to the model being
    imperfect in the first place.
    """
    T = sub["generation"].max()
    term = sub[sub["generation"] == T]
    if term.empty:
        return None
    per_alpha = term.groupby("alpha")[axis].mean()
    if 1.0 not in per_alpha.index or per_alpha.isna().all():
        return None
    base = per_alpha.loc[1.0]
    if not np.isfinite(base):
        return None

    if axis == "var_ratio":
        # ideal is 1.0; degradation = extra distance from 1 beyond the baseline's
        d = (per_alpha - 1.0).abs() - abs(base - 1.0)
    elif axis in HIGHER_BETTER:
        d = base - per_alpha                       # dropping below baseline is bad
    else:
        d = per_alpha - base                       # rising above baseline is bad
    return d.clip(lower=0.0)                       # doing better than baseline is not degradation


def alpha_star(deg, eps_abs):
    """Smallest alpha whose degradation is <= eps_abs AND stays <= it for all
    larger alpha. The monotonicity requirement matters: a single noisy alpha
    dipping under the tolerance is not a threshold, and taking a bare min would
    report that noise as the answer. Linearly interpolates between grid points."""
    if deg is None or deg.dropna().empty:
        return np.nan
    a = deg.dropna().sort_index()
    ok = a <= eps_abs
    # walk from the top; the threshold is where the all-ok suffix begins
    idx = list(a.index)
    start = len(idx)
    for i in range(len(idx) - 1, -1, -1):
        if ok.iloc[i]:
            start = i
        else:
            break
    if start == 0:
        return float(idx[0])                       # even alpha=0 is within tolerance
    if start >= len(idx):
        return np.nan                              # never within tolerance
    lo, hi = idx[start - 1], idx[start]            # crossing lies in (lo, hi]
    dlo, dhi = a.loc[lo], a.loc[hi]
    if not np.isfinite(dlo) or not np.isfinite(dhi) or dlo == dhi:
        return float(hi)
    frac = (dlo - eps_abs) / (dlo - dhi)
    return float(lo + frac * (hi - lo))


def theory_alpha_star(eps_rel, n):
    """Exact quadratic root from Prop. 3 (not the 1/(2 n eps) approximation)."""
    disc = (2 * n - 1) ** 2 - 4 * n * (1 / eps_rel - 1)
    if disc < 0:
        return np.nan
    return ((2 * n - 1) - np.sqrt(disc)) / (2 * n)


def main():
    d = load_all()
    print(f"loaded {len(d)} rows from {d['_src'].nunique()} files")
    gcols = [c for c in ["dataset", "synth", "regime", "n"] if c in d.columns]
    print(d.groupby(gcols).size().to_string())

    rows = []
    for keys, sub in d.groupby(gcols):
        if sub["alpha"].nunique() < 3:
            continue
        for axis in AXES:
            if axis not in sub.columns or sub[axis].isna().all():
                continue
            deg = degradation(sub, axis)
            if deg is None:
                continue
            scale = deg.max()
            for eps in (0.05, 0.10, 0.25):
                # tolerance expressed as a fraction of the WORST observed
                # degradation for this axis, so axes on different units are
                # comparable; absolute-unit thresholds are reported too.
                rows.append(dict(zip(gcols, keys if isinstance(keys, tuple) else (keys,))) | {
                    "axis": axis,
                    "eps_frac_of_worst": eps,
                    "eps_abs": eps * scale if np.isfinite(scale) else np.nan,
                    "worst_degradation": scale,
                    "alpha_star": alpha_star(deg, eps * scale) if np.isfinite(scale) else np.nan,
                    "T": int(sub["generation"].max()),
                    "n_seeds": int(sub["seed"].nunique()) if "seed" in sub else np.nan,
                })
    out = pd.DataFrame(rows)
    if out.empty:
        print("\n[no thresholds estimable yet] Every group either has <3 alpha "
              "values or lacks a completed alpha=1.0 baseline at its terminal "
              "generation. This is expected while the sweep is still running.")
        return
    out.to_csv(os.path.join(RESULTS, "analysis_thresholds.csv"), index=False)

    lines = ["=== alpha* by axis (eps = 25% of worst observed degradation) ==="]
    sel = out[out["eps_frac_of_worst"] == 0.25]
    if not sel.empty:
        piv = sel.pivot_table(index=[c for c in gcols], columns="axis",
                              values="alpha_star")
        lines.append(piv.round(3).to_string())
        lines.append("")
        lines.append("=== DO THE AXES AGREE? (spread of alpha* across axes) ===")
        lines.append("If this spread is small, claim C6 (axes impose different")
        lines.append("thresholds) is NOT supported and must be cut from the paper.")
        spread = piv.max(axis=1) - piv.min(axis=1)
        lines.append(spread.round(3).to_string())
    txt = "\n".join(lines)
    with open(os.path.join(RESULTS, "analysis_summary.txt"), "w") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
