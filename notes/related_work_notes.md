# Related Work Notes

Honest notes on the ~10 most load-bearing prior papers: what each actually shows, and — critically —
what it does *not* cover with respect to (a) **tabular** data specifically and (b) **anchoring fractions**
(the real-data mixing ratio as a first-class, tunable, per-domain quantity). These are the papers our
"Real-Data Anchoring Thresholds for Preventing Model Collapse in Tabular Generative Models" contribution
must be positioned against, not ignore.

---

### 1. Shumailov et al., "AI Models Collapse When Trained on Recursively Generated Data" (Nature, 2024; preprint arXiv:2305.17493)
Shows, across VAEs, Gaussian mixture models, and language models, that **fully replacing** real data with
model-generated data over successive generations causes irreversible loss of distributional tails ("early"
and "late" collapse). This is the foundational demonstration of the phenomenon and the source of the name.
**Gap**: the training regime studied is *pure replacement* (0% real data retained after generation 0), not
a tunable real/synthetic mixture. No tabular data, no notion of an anchoring fraction, no threshold below
which collapse is avoidable — the paper's message is "collapse happens," not "here is the safe operating
region."

### 2. Alemohammad et al., "Self-Consuming Generative Models Go MAD" (arXiv:2307.01850)
Formalizes three families of self-consuming ("autophagous") loops for image generative models, distinguishing
whether real data is fixed, fresh, or absent at each generation, and whether synthetic samples are biased
toward quality over diversity. Demonstrates model autophagy disorder (MAD) empirically on GANs/diffusion
models for images. **Gap**: image domain only; the "fixed/fresh real data" loop types are a qualitative
taxonomy, not a quantitative fraction/threshold result, and nothing here transfers a mixing-ratio analysis
to mixed continuous/categorical tabular schemas.

### 3. Bertrand et al., "On the Stability of Iterative Retraining of Generative Models on their own Data" (ICLR 2024, arXiv:2310.00429)
**This is one of the papers our contribution must be argued against directly.** Bertrand et al. give a
rigorous stability analysis of iterative retraining on mixtures of real and synthetic data and derive a
"stability radius": provided the fraction of real data injected at each retraining step exceeds a
(model-dependent) threshold, the retraining dynamics remain stable near the true data distribution;
below it, they diverge. This is conceptually the closest existing result to an "anchoring threshold."
**Gap**: the analysis is generic/theoretical (kernel density estimators and general parametric families),
validated mainly on images; it does not instantiate the threshold for tabular generative architectures
(CTGAN, TVAE, TabDDPM, TabSyn, GReaT) with their mixed discrete/continuous, highly non-Gaussian marginals,
nor does it give practitioners a measured, dataset-conditional fraction to use. Our contribution is to make
this threshold concrete, measured, and architecture-specific for tabular models rather than asymptotic and
generic.

### 4. Gerstgrasser et al., "Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data" (COLM 2024, arXiv:2404.01413)
**Also directly load-bearing for positioning.** Shows that if successive generations of synthetic data are
*accumulated alongside* the original real data (rather than replacing it), collapse is avoided even as the
synthetic share of the corpus grows — tested on language model pretraining and simple regression/GMM tasks.
This is the paper that established "accumulate, don't replace" as a mitigation strategy. **Gap**: the
accumulate-vs-replace dichotomy is binary in spirit (either you keep all past real data or you don't); it
does not ask "what is the minimum real-data *fraction* per retraining round" for tabular architectures, nor
does it study tabular column-wise fidelity/marginal collapse (e.g., rare categorical levels, tail behavior
of numeric columns), which is the concrete failure mode tabular anchoring papers must address.

### 5. Dohmatob et al., "Strong Model Collapse" (ICLR 2025, arXiv:2410.04840) and "A Tale of Tails: Model Collapse as a Change of Scaling Laws" (ICML 2024, arXiv:2402.07043)
Give a precise theoretical (bias-variance / scaling-law) account of collapse in linear regression and
simplified neural network settings, showing that even a **1% synthetic contamination** can eventually flatten
or reverse scaling-law gains, and that collapse manifests as a change in the exponent/shape of the scaling
curve rather than merely a level shift. **Gap**: linear/random-projection theory, not tabular generative
models; the "1% is enough" finding is about training-corpus contamination for predictive models trained *on*
mixed data, not about the fraction of real vs. synthetic data used to train a *generator* in an iterative
tabular synthesis pipeline. It motivates why small contamination matters but gives no tabular-specific
threshold.

### 6. Seddik et al., "How Bad is Training on Synthetic Data? A Statistical Analysis of Language Model Collapse" (arXiv:2404.05090)
Provides a statistical analysis showing collapse is unavoidable under pure synthetic training, and — notably —
derives an **estimate of a maximal synthetic-data fraction** below which collapse can be avoided, for language
models. **Gap**: this is the language-model analogue of an "anchoring threshold," but it is derived under
LM-specific statistical assumptions (token distributions / n-gram or transformer loss landscapes) and gives
no guidance for mixed-type tabular feature spaces, categorical cardinality effects, or the kind of
utility/fidelity/privacy metrics (TSTR, alpha-precision, C2ST) practitioners use to evaluate tabular synthesizers.

### 7. Ferbach et al., "Self-Consuming Generative Models with Curated Data Provably Optimize Human Preferences" (arXiv:2407.09499)
**Also must be cited as prior art on mixing real data in the loop.** Proves that when synthetic data is
curated (filtered by a reward/preference model) before being fed back into retraining, and a **positive
fraction of real data is retained at each step**, the retraining loop is stable and provably converges toward
higher human-preference regions rather than collapsing. **Gap**: curation here means reward-based selection
of "good" synthetic samples (human-preference alignment for e.g. text-to-image models), not a raw real-data
anchoring ratio chosen for statistical safety; no tabular instantiation; the "positive fraction of real data"
condition is a stability *sufficient condition*, not a measured, task-specific threshold curve.

### 8. Kazdan et al., "Collapse or Thrive? Perils and Promises of Synthetic Data in a Self-Generating World" (arXiv:2410.16713)
Studies (via Gaussian estimation, kernel density estimation, and LM fine-tuning) how the *value* of adding
synthetic data depends on the absolute amount of real data available — confirming replacement causes collapse
while accumulation is stable, but showing the benefit of synthetic data is conditional on real-data volume.
**Gap**: again general statistical toy tasks plus language fine-tuning; no tabular generative architectures,
no anchoring-fraction threshold curve as a function of dataset size/dimensionality/column type mix.

### 9. Xu et al., "Modeling Tabular Data using Conditional GAN" (NeurIPS 2019) — CTGAN/TVAE
The canonical tabular generative modeling paper: introduces mode-specific normalization and a
conditional-generator training-by-sampling scheme to handle mixed continuous/categorical, multimodal,
imbalanced tabular columns. Establishes CTGAN and TVAE as baselines nearly every subsequent tabular
synthesis paper compares against. **Gap**: entirely single-generation training on real data; there is no
recursive/self-consuming retraining setting considered at all, let alone an anchoring fraction.

### 10. Kotelnikov et al., "TabDDPM: Modelling Tabular Data with Diffusion Models" (ICML 2023) and Zhang et al., "Mixed-Type Tabular Data Synthesis with Score-based Diffusion in Latent Space" (TabSyn, ICLR 2024)
State-of-the-art diffusion-based tabular generators, handling mixed numerical/categorical types via
multinomial + Gaussian diffusion (TabDDPM) or a learned latent space plus score-based diffusion (TabSyn).
Both report strong fidelity/utility numbers evaluated on real training data only. **Gap**: neither paper
studies what happens when these models are iteratively retrained on their own (or each other's) synthetic
output, nor mixtures thereof — the collapse question is simply out of scope, which is precisely the space
our anchoring-threshold study occupies.

---

## Honest summary of what is and is not already "taken"

Papers that **already study mixing real and synthetic data explicitly** and must be engaged with rather than
ignored: **Gerstgrasser et al. (accumulate vs. replace)**, **Bertrand et al. (stability radius under a real-data
fraction)**, **Ferbach et al. (stability under curated data + a positive real-data fraction)**, and
**Kazdan et al. (value of synthetic data conditional on real-data volume)**. Seddik et al. also derive a
maximal-synthetic-fraction estimate, but for language models. None of these five instantiate their
threshold/fraction results on tabular generative architectures (CTGAN, TVAE, CTAB-GAN(+), TabDDPM, TabSyn,
GReaT), on mixed discrete/continuous schemas, or against the standard tabular fidelity/utility/privacy
evaluation stack (TSTR, C2ST, alpha-precision/beta-recall, SDMetrics-style column-wise tests). That
combination — a measured, architecture- and dataset-conditional real-data anchoring fraction for tabular
generative models, validated with tabular-specific fidelity/utility metrics — is the gap this paper claims to
fill.
