# Claims reconciled against results (updated after Tier 1 + n-sweep landed)

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| C1 | Fixed point Σ∞=(1-1/D)Σ, V∞=Σ/D | **VERIFIED** | Orchestrator's two independent from-scratch sims (numpy CPU + torch GPU) |
| C2 | Conservation identity Σ∞+V∞=Σ | **VERIFIED** | Same; sum = 0.999–1.001 |
| C3 | Convergence rate 1−α | **SUPPORTED** | α=0.01 needed ~1500 gens to converge, consistent with rate 1−α |
| C4 | α* scales as 1/n | **QUANTIFIED (15-seed, 7 n-values)** | Power-law fit β=0.26, 95% CI [0.07,0.54]: excludes both 0 (falls with n) and 1 (not 1/n). Heterogeneous: w1/tv/c2st reach α*=0 by n=8000, corr_frob stays ~0.35 flat. |
| C5 | Collapse occurs at α=0 | **CONFIRMED** | Adult TSTR 0.674→0.389 (ceiling 0.893, α=1 control 0.670); bank corr 0.153→0.487 |
| C6 | Axes have different thresholds | **STRONGLY CONFIRMED** | Spread 0.44–0.59. Adult n=2000: var 0.154 vs corr 0.626 |
| C7 | Fresh anchoring repairs support, fixed does not | **CONFIRMED SIGNIFICANT (15 seeds)** | Fresh>fixed on support for α∈[0.02,0.35], all p<0.005 (peak +0.107 at α=0.10, p<1e-4). NO reversal at α≤0.01 (p≈0.45 — the 3-seed reversal was noise, now corrected in text). No effect on corr (p=0.59/0.32/0.98) — absorbing-state asymmetry holds. |
| C8 | Generalizes across architectures | **CONFIRMED (qualitative)** | All 3 lose support at α=0, recover at α=1: CTGAN 0.845→0.464 vs →0.820; TVAE 0.907→0.706 vs →0.918. Variance fails as signal in all 3 (copula flat, TVAE flat, CTGAN inflates to 1.60). Quantitative fixed-point is copula-only — stated as such in limitations. §res-arch + Table tab:arch |

## NEW FINDING NOT IN THE ORIGINAL PLAN (now a headline)

**Variance does not collapse for rank-based copulas.** var_ratio at α=0 ends at
0.959 (Adult), 0.925 (credit-g), 1.043 (bank-marketing) — flat, within noise of
the α=1 controls. The rank/CDF transform re-normalizes marginals every generation
by construction, immunizing the variance axis and displacing the damage into
dependence, support, and utility.

This **contradicted the drafted abstract and intro**, which led with variance
contraction as the empirical story. Both were rewritten to lead with the negative
result. The theory is not invalidated — it is exact for the Gaussian fitting
operator it describes — but its empirical reach is now explicitly bounded in the
text (§Results B, §Discussion limitations).

Also noted: credit-g shows cat_support = 1.000 at every α and generation; its
categoricals are low-cardinality, so the support failure mode cannot manifest
there. Support claims rest on Adult (40-level native-country).

## Remaining before submission
- [ ] Fold in CTGAN/TVAE results when tiers finish → resolve C8
- [ ] Regenerate figures from final main.csv
- [ ] Re-run analyze.py on the merged file; update any number that shifts
- [ ] Confirm no `sampled_with_replacement=True` rows in final analysis
