"""
Correct test of the alpha* ~ 1/n scaling law.

analyze.py sets the tolerance to a FRACTION OF EACH CONFIG'S OWN worst observed
degradation. That is fine for comparing axes within a fixed n, but it is invalid
for comparing across n: total degradation shrinks as n grows, so a relative
tolerance silently tightens the absolute bar at large n and can make alpha* rise
with n even when the underlying scaling is real.

Here we fix ONE absolute tolerance per axis, common to every n (anchored to the
smallest n so every config is judged against the same standard), and re-estimate.
"""
import glob
import os

import numpy as np
import pandas as pd

from analyze import AXES, alpha_star, degradation, load_all

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
FRAC = 0.25          # tolerance as a fraction of the REFERENCE-n worst degradation
REF_N = 500          # reference: judge every n against the smallest n's standard


def main():
    d = load_all()
    d = d[(d.synth == "gaussian_copula") & (d.dataset == "adult")
          & (d.regime == "fixed")]
    ns = sorted(d["n"].unique())
    print(f"adult / copula / fixed, n = {ns}")

    degs = {}
    for n in ns:
        sub = d[d["n"] == n]
        for axis in AXES:
            if axis in sub.columns and not sub[axis].isna().all():
                degs[(n, axis)] = degradation(sub, axis)

    rows = []
    for axis in AXES:
        ref = degs.get((REF_N, axis))
        if ref is None or ref.dropna().empty:
            continue
        eps_abs = FRAC * ref.max()               # ONE bar, used at every n
        if not np.isfinite(eps_abs) or eps_abs <= 0:
            continue
        rec = {"axis": axis, "eps_abs": eps_abs}
        for n in ns:
            dg = degs.get((n, axis))
            rec[n] = alpha_star(dg, eps_abs) if dg is not None else np.nan
            rec[f"worst_{n}"] = dg.max() if dg is not None else np.nan
        rows.append(rec)

    t = pd.DataFrame(rows).set_index("axis")
    print("\n=== alpha* with a COMMON absolute tolerance (eps fixed per axis) ===")
    print(t[["eps_abs"] + ns].round(3).to_string())

    print("\n=== worst degradation actually observed (why the relative bar misleads) ===")
    print(t[[f"worst_{n}" for n in ns]].round(4).to_string())

    print("\n=== scaling check: alpha*(n) / alpha*(n_ref), vs theory 1/n ===")
    for n in ns:
        if n == REF_N:
            continue
        ratio = (t[REF_N] / t[n]).replace([np.inf, -np.inf], np.nan)
        theory = n / REF_N
        got = ratio.dropna()
        print(f"  n={REF_N}->{n}: theory predicts {theory:.1f}x reduction; "
              f"measured {got.median():.2f}x (median over {len(got)} axes, "
              f"range {got.min():.2f}-{got.max():.2f})")

    mono = []
    for axis in t.index:
        vals = [t.loc[axis, n] for n in ns]
        if all(np.isfinite(v) for v in vals):
            mono.append((axis, "falls" if all(np.diff(vals) <= 1e-9) else "NOT monotone"))
    print("\n=== monotonicity per axis ===")
    for a, m in mono:
        print(f"  {a:14s} {m}")

    t.to_csv(os.path.join(RESULTS, "analysis_nscaling.csv"))
    print(f"\nwrote {os.path.join(RESULTS, 'analysis_nscaling.csv')}")


if __name__ == "__main__":
    main()
