#!/usr/bin/env bash
# Two definitive runs to firm up the two compute-fixable weak spots.
# Sequential (B then A), each idempotent via skip-logic, append-only.
set -e
cd "$(dirname "$0")"

echo "=== RUN B: fresh-vs-fixed decisiveness (15 seeds, low-alpha region) ==="
python run_experiment.py \
  --datasets adult --synths gaussian_copula \
  --alphas 0.0 0.01 0.02 0.05 0.10 0.20 0.35 \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 \
  --generations 12 --n 2000 --regimes fixed fresh \
  --metrics fast --device cuda --workers 6 \
  --out ../results/strengthen_regime.csv --allow-failures

echo "=== RUN A: scaling law (7 n-values 250..16000, 8 seeds, full alpha) ==="
python run_experiment.py \
  --datasets adult --synths gaussian_copula \
  --alphas 0.0 0.01 0.02 0.05 0.10 0.20 0.35 0.50 0.75 1.0 \
  --seeds 0 1 2 3 4 5 6 7 \
  --generations 12 --n 250 500 1000 2000 4000 8000 16000 --regimes fixed \
  --metrics fast --device cuda --workers 6 \
  --out ../results/strengthen_scaling.csv --allow-failures

echo "=== STRENGTHEN RUNS COMPLETE ==="
touch ../results/STRENGTHEN_DONE
