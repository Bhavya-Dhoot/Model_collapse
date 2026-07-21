# Real-Data Anchoring Thresholds for Preventing Model Collapse in Tabular Generative Models

Code, experiments, and findings. (Paper is withheld pending conference presentation.)

## Layout

```
code/      experiment harness (synthesizers, metrics, driver, theory verification)
results/   tidy CSV of every (dataset, synthesizer, regime, alpha, seed, generation)
notes/     working notes: verified-citation summaries, theory derivation notes, findings
```

## Reproducing

```bash
python code/run_experiment.py --out results/main.csv      # full sweep
python code/verify_theory.py                              # Monte-Carlo check of the closed forms
python code/make_figures.py                               # regenerates the figures
```

Every reported number comes from `results/main.csv`. Nothing is hand-entered.
See `notes/claims_to_verify.md` for the status of every finding.
