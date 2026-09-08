/* global window */
(function (root) {
'use strict';
/* Anchored self-consuming loop: closed forms from the paper, plus a live
   Monte-Carlo of the same loop so the two can be compared on screen.

   Scalar (one-eigenvalue) case, which is all the theory needs: fitting is
   affine-equivariant, so the full covariance reassembles along each eigenvalue.
   True distribution N(0, 1). */

const SIGMA = 1;

/* ---- closed forms (paper, Section IV) ---------------------------------- */

/** D(alpha, n) = 1 - alpha + alpha*n*(2 - alpha) */
function D(alpha, n) {
  return 1 - alpha + alpha * n * (2 - alpha);
}

/** Fixed point: Sigma_inf = (1 - 1/D)*Sigma, V_inf = Sigma/D. Sums to Sigma. */
function fixedPoint(alpha, n) {
  const d = D(alpha, n);
  return { sigma: (1 - 1 / d) * SIGMA, v: SIGMA / d, D: d, rate: 1 - alpha };
}

/** Exact deterministic recursion for (E[Sigma_t], V_t). */
function recursion(alpha, n, T) {
  const c = (n - 1) / n;
  let s = SIGMA, v = 0;
  const out = [{ t: 0, sigma: s, v: v, sum: s + v }];
  for (let t = 1; t <= T; t++) {
    const sPrev = s, vPrev = v;
    s = alpha * c * SIGMA + (1 - alpha) * c * sPrev + alpha * (1 - alpha) * vPrev;
    v = (alpha / n) * SIGMA + ((1 - alpha) / n) * sPrev + (1 - alpha) ** 2 * vPrev;
    out.push({ t, sigma: s, v: v, sum: s + v });
  }
  return out;
}

/** Anchoring threshold: exact quadratic root, and its leading-order form. */
function alphaStarExact(eps, n) {
  const disc = (2 * n - 1) ** 2 - 4 * n * (1 / eps - 1);
  if (disc < 0) return NaN;
  return ((2 * n - 1) - Math.sqrt(disc)) / (2 * n);
}
function alphaStarApprox(eps, n) {
  return (1 - eps) / (2 * n * eps);
}

/* ---- Monte-Carlo of the actual loop ------------------------------------ */

/** mulberry32: small seeded PRNG, so every run on stage is reproducible. */
function rng(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box-Muller, cached second draw. */
function normalSampler(rand) {
  let spare = null;
  return function () {
    if (spare !== null) { const s = spare; spare = null; return s; }
    let u, v, s;
    do { u = rand() * 2 - 1; v = rand() * 2 - 1; s = u * u + v * v; } while (s >= 1 || s === 0);
    const f = Math.sqrt((-2 * Math.log(s)) / s);
    spare = v * f;
    return u * f;
  };
}

/**
 * Run the loop for real: each generation fits (mean, biased variance) on
 * n rows, of which round(alpha*n) are freshly drawn from the true N(0,1)
 * and the rest are sampled from the previous generation's own fit.
 * Averages E[sigma_t] and Var[mu_t] across `reps` independent chains.
 */
function monteCarlo({ alpha, n, T, reps = 200, seed = 7 }) {
  const rand = rng(seed);
  const gauss = normalSampler(rand);
  const nr = Math.round(alpha * n);
  const ns = n - nr;

  let mu = new Float64Array(reps);      // each chain's current fitted mean
  let sig = new Float64Array(reps).fill(SIGMA); // and fitted variance
  const series = [{ t: 0, sigma: SIGMA, v: 0, sum: SIGMA }];

  for (let t = 1; t <= T; t++) {
    let sumSig = 0, sumMu = 0, sumMu2 = 0;
    for (let r = 0; r < reps; r++) {
      const sd = Math.sqrt(Math.max(sig[r], 0));
      // pool: nr real draws + ns synthetic draws from the previous fit
      let m = 0;
      const buf = new Float64Array(n);
      for (let i = 0; i < nr; i++) { const x = gauss(); buf[i] = x; m += x; }
      for (let i = 0; i < ns; i++) { const x = mu[r] + sd * gauss(); buf[nr + i] = x; m += x; }
      m /= n;
      let q = 0;
      for (let i = 0; i < n; i++) { const dd = buf[i] - m; q += dd * dd; }
      mu[r] = m;
      sig[r] = q / n;                    // biased (divide-by-n) estimate
      sumSig += sig[r]; sumMu += m; sumMu2 += m * m;
    }
    const eSig = sumSig / reps;
    const vMu = Math.max(sumMu2 / reps - (sumMu / reps) ** 2, 0);
    series.push({ t, sigma: eSig, v: vMu, sum: eSig + vMu });
  }
  return series;
}

root.AT = root.AT || {};
root.AT.sim = { SIGMA, D, fixedPoint, recursion, alphaStarExact, alphaStarApprox, monteCarlo };
}(window));
