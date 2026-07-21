# Theory notes (plain language companion to `sec_theory.tex`)

## The setting in one paragraph

Every generation, you refit a Gaussian model on a training set that is a mix of
`alpha` fraction real rows (drawn fresh from the true distribution `P` every
time) and `1-alpha` fraction rows sampled from last generation's fitted model.
`alpha=0` is the classic "eat your own output forever" collapse loop. `alpha=1`
is "always retrain on fresh real data" (no recursion at all). We want to know,
for `0 < alpha < 1`, whether the loop settles down near `P` or drifts away.

We do this exactly for the Gaussian case (which is also exactly what a
Gaussian-copula tabular synthesizer does internally), then give a more general,
assumption-based argument for non-Gaussian synthesizers (CTGAN/TVAE-style),
being explicit about where that argument can break.

## Key results, in words

**1. The mean never gets systematically biased, at any alpha.** Averaging real
data and synthetic data, generation after generation, keeps the *expected* mean
exactly at the truth. What changes is the *spread* around it.

**2. At alpha=0 (no anchoring), two things shrink together.**
- The covariance decays geometrically: each generation multiplies the expected
  covariance by exactly `(n-1)/n`, purely from re-estimating a covariance
  matrix off of a finite `n`-row sample every time.
- The mean does a "random walk" whose variance grows like `t/n` for the first
  few tens/hundreds of generations — this is the textbook random-walk
  intuition. But because the covariance is shrinking at the same time (the
  step sizes of the walk are themselves shrinking), the walk's variance does
  **not** grow forever: it saturates at exactly `Sigma`, the original true
  covariance. In other words, without anchoring, the model doesn't wander off
  to infinity — it collapses to a point that sits a "typical distance"
  `sqrt(Sigma)` away from the truth, and then stays there. This refined
  "linear-then-saturating" picture is what the algebra actually gives (we
  expected a naive "linear forever" story going in, and the exact math forced
  a correction — that is one of the more interesting findings here).

- **A subtlety about "biased" vs "unbiased" covariance estimators.** If you
  use the standard *unbiased* sample covariance (divide by `n-1`), something
  strange happens: the *expected* covariance never shrinks at all — it stays
  exactly flat, generation after generation, forever unbiased. And yet, if you
  actually run the loop, the covariance still collapses to zero almost every
  time you try it. This is a real, well-known statistical phenomenon (Jensen's
  inequality on the log-scale of a multiplicative random process): being
  "unbiased" describes the average over infinitely many parallel universes,
  not what happens to any single realized model as it iterates. This is a
  useful thing to say when people push back with "but sample covariance is
  unbiased, so it can't be collapsing" — it can, and does.

**3. With any real-anchoring alpha > 0, both quantities stop shrinking to zero
and settle at a finite, non-zero fixed point.** We get exact closed-form
expressions for that plateau:

```
Sigma_infinity  =  (1 - 1/D) * Sigma
Var[mu_infinity] =  (1/D) * Sigma
   where D(alpha, n) = 1 - alpha + alpha * n * (2 - alpha)
```

A clean and slightly surprising exact identity falls out of the algebra:
`Sigma_infinity + Var[mu_infinity] = Sigma` — the "budget" of one true
covariance's worth of variance gets split between covariance-shrinkage and
mean-drift, and the split point is set by `alpha` and `n`.

The loop reaches this plateau geometrically, and the rate is exactly
`(1 - alpha)` per generation — i.e., the real-data fraction alone sets how
fast the process forgets its starting point / settles down. A useful rule of
thumb: after roughly `3/alpha` to `10/alpha` generations you're essentially at
the plateau.

## The headline practical result: the anchoring threshold `alpha*`

Define the tolerance: you're willing to accept losing at most a fraction
`epsilon` of the true variance in steady state. Solving for the smallest
`alpha` that satisfies this gives an exact quadratic, and — in the common
regime where `n * epsilon` is not tiny — a very clean approximation:

```
alpha*(epsilon, n)  ≈  (1 - epsilon) / (2 * n * epsilon)
```

**The counterintuitive part:** `alpha*` gets *smaller* as `n` grows. A big
dataset needs only a tiny sliver of real data mixed in to stay safe; a small
dataset needs a much bigger real fraction for the same tolerance. Concretely
(from the exact formula):

| n | epsilon=0.01 | epsilon=0.05 | epsilon=0.10 |
|---|---|---|---|
| 1,000 | 5.1% | 0.96% | 0.45% |
| 10,000 | 0.50% | 0.095% | 0.045% |
| 100,000 | 0.050% | 0.0095% | 0.0045% |

This matters directly for practice: healthcare and finance tabular datasets
are often small-`n` (hundreds to low thousands of rows) — exactly the regime
where this analysis says you need the *largest* real-anchor fraction, which is
also usually where real data is scarcest and most expensive. Large-`n`
consumer/web-scale tabular data is comparatively "safe" with only a trace of
real anchoring.

## Beyond Gaussian: how far does this travel, honestly

The Gaussian result generalizes to a soft, assumption-based statement: *if*
your synthesizer's one-step fitting procedure is a Lipschitz-stable
contraction (in Wasserstein-2 distance) with modulus `L`, and has finite-sample
fitting error of size `delta_n`, then anchoring with `alpha > 1 - 1/L` keeps
the loop stable, with a steady-state error of about `delta_n / (1 - (1-alpha)L)`.

This is explicitly **not** a theorem about CTGAN or TVAE. It is a template that
requires assumptions those models plausibly violate:
- Non-convex GAN/VAE training gives no guaranteed, run-independent Lipschitz
  constant `L`.
- Mode dropping is a discrete, catastrophic failure, not something a single
  smooth contraction number can describe.
- For categorical columns, once a support value is fully "forgotten" it's an
  **absorbing state**, not a shrinking-variance state: no amount of continuous
  contraction can bring it back. Critically, it *can* come back if you keep
  drawing **fresh** real anchors each generation (they'll reintroduce the
  missing category), but it will **never** come back if your "anchor" is a
  **fixed, static** real dataset that itself never contained that category in
  the first place. This fixed-vs-fresh distinction is the one place where
  "just add some real data" is not automatically a fix.

## Falsifiable predictions for the experiments section

1. **P1**: at alpha=0, `E[Sigma_t]/Sigma = ((n-1)/n)^t` exactly.
2. **P2**: at alpha>0, the variance-ratio plateaus at `1 - 1/D(alpha,n)`,
   `D = 1 - alpha + alpha n (2-alpha)`, reached to within `(1-alpha)^t` of the
   plateau.
3. **P3**: `alpha* ≈ (1-epsilon)/(2 n epsilon)`, i.e. scales as `1/(n*epsilon)`.
4. **P4**: a categorical level dropped at alpha=0 recovers within a few
   generations under fresh alpha>0 anchoring, never recovers under alpha>0
   anchoring from a fixed anchor set lacking that level.
5. **P5**: once converged (`t >> 1/alpha`), the plateau value is flat in `t`
   and independent of the starting point.

## Numerical verification — how we checked our own algebra, and what we found

All closed forms were (a) re-derived symbolically with `sympy`
(`code/fixed_point_solve.py`, `code/fixed_point_solve2.py`, `code/rate_solve.py`)
independent of the by-hand derivation, and (b) checked against a GPU-batched
Monte Carlo simulation of the *exact* loop (`code/verify_theory.py`, d=5,
n in {200, 2000}, T=200 generations, R=2000 replicates, float64).

**Important methodological point, found while verifying:** you cannot report
a single "max relative error" across `E[Sigma_t]` and `Var[mu_t]` — the two
quantities have very different Monte Carlo precision (`Var[mu_t]` is a
*variance* estimator, intrinsically noisier at fixed R than the *mean*
estimator `E[Sigma_t]`), so a large discrepancy in one looks identical to a
small discrepancy in the other unless you track them separately, in units of
each quantity's own standard error (a z-score). We also found that at small
`alpha` (e.g. 0.01), the fixed point is approached at rate `(1-alpha)` per
generation, so `T=200` generations is not long enough to have converged
(`0.99^200 ≈ 0.13`, i.e. 13% of the initial gap still remains) — comparing an
unconverged simulation against the asymptotic closed form would look like an
algebra bug when it is actually just an insufficient time horizon. We fixed
this by giving the fixed-point check an alpha-dependent horizon
(`T_fp ≈ 14/alpha`, capped at 1500 generations) long enough that
`(1-alpha)^{T_fp}` is negligible, while keeping the transient-recursion check
at a uniform `T=200` (that check doesn't require convergence — it's checking
the exact recursion at any given `t`, not the asymptote).

**Result:** see the run log / `code/verify_results.json` for the final
per-quantity max |z-score| and PASS/FAIL verdict — reported verbatim in the
main deliverable summary of this task, since it is copied into the paper as
the verification claim and should not be restated from memory here.

Figure: `paper/figs/fig_theory_verify.pdf` — theory curves (lines) vs.
simulation (markers), for the (normalized, averaged over the 5 coordinates)
covariance-ratio and mean-drift-variance-ratio trajectories, all (n, alpha)
combinations, IEEE single-column size.
