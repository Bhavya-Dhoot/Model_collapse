# Interactive demo

A presentation-grade companion to the paper. Every chart is recomputed from
`results/main.csv` at page load — nothing on the page is a stored image.

## Showing it

**Use `dashboard.html`.** Double-click it. It is fully self-contained: the CSS,
the data and every script are inlined, so it needs no server, no install and no
network. That is the version to put on a projector.

`index.html` is the same page in unbundled form (separate `styles.css`,
`data.json`, `lib/`, `panels/`). Browsers block `fetch` over `file://`, so it
needs a local server:

```bash
python -m http.server 8000 --directory demo
# then open http://localhost:8000/index.html
```

## What is on the page

| # | Section | What it shows |
|---|---------|---------------|
| 01 | Mechanism | The anchored loop as an animated SVG. The α slider changes the real/synthetic composition of a fixed n-row budget. |
| 02 | Theory, live | A real Monte-Carlo of the Gaussian loop, run in the browser, overlaid on the closed form. The conservation identity Σ∞ + Var[μ∞] = Σ is displayed as it holds. |
| 03 | Collapse | Measured degradation over 12 generations on Adult, credit-g and bank-marketing. |
| 04 | Negative result | The variance ratio does not contract for a rank-based copula, while the other axes rot. |
| 05 | Thresholds | Per-axis α★ and the 2.7× spread between the strictest and loosest axis. |
| 06 | Scaling | α★ against n, the measured exponent against the theoretical 1/n law, plus a budget calculator. |
| 07 | Fresh vs fixed | Categorical support recovery, paired over 15 seeds. |
| 08 | Architectures | Copula / TVAE / CTGAN, and where the theory stops applying. |
| 09 | Provenance | The grid, the seed counts and the commands that regenerate all of it. |

## Rebuilding after new results

```bash
python code/export_dashboard_data.py   # results/main.csv -> demo/data.json
python code/build_demo.py              # inlines everything -> demo/dashboard.html
```

`export_dashboard_data.py` imports the estimators from `code/make_figures.py`,
so the demo and the paper's figures are produced by the same code path.

## Which seeds each panel uses

`results/main.csv` accumulates every seed ever run, so averaging over all of it
would silently change the published numbers. Each panel is therefore pinned to
the grid its section of the paper used:

- **Trajectories, per-axis α★, degradation curves** — the 3-seed primary grid
  (paper, Results VI-A to VI-D). This reproduces Table I exactly for
  Adult n=2000, bank-marketing n=2000 and credit-g n=2000.
- **Fresh vs fixed** — all 15 seeds, paired t-test, as stated in Results VI-E.
- **n-sweep** — 8 seeds, one absolute tolerance per axis anchored at n=250.

## Known gap

The paper's Table I row for **Adult at n=500** does not reproduce from the
released `results/main.csv`, nor from the saved `results/analysis_thresholds.csv`
(which gives 0.291 / 0.409 / 0.632 / 0.500 against the printed
0.429 / 0.610 / 0.788 / 0.576). That row appears to come from a shard set that
predates the current merge. The demo therefore shows the three rows that do
reproduce. Worth correcting in the camera-ready.
