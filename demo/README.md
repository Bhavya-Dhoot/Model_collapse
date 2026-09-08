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

## Presentation mode

Click **Start presentation** (or press <kbd>P</kbd>). The page then runs itself:
it scrolls the argument in order and drives each panel's own controls, so the
findings demonstrate themselves without anyone touching the laptop.

| Key | Action |
|-----|--------|
| `P` | start / pause |
| `Space` | pause / resume |
| `←` `→` | previous / next section |
| `Esc` | exit |

Scrolling the wheel or touching the screen hands control straight back to you.
Appending `#present` to the URL starts the tour on load, for a kiosk.

Motion notes, since this is a live demo and not a toy:

- **One scroll authority.** A single `requestAnimationFrame` loop owns scroll
  while presenting, and the CSS `scroll-behavior: smooth` is switched off for
  the duration so the two never fight. No external motion library is used --
  the page has to run with no network.
- **A watchdog behind every animation.** Browsers stop firing animation frames
  in a backgrounded tab or on a sleeping display. Without a guard the tour
  would wedge mid-talk, so if frames stop arriving the step lands instantly and
  the tour carries on.
- **`prefers-reduced-motion` is a designed branch, not a kill switch.** The
  tour still advances and still performs; it jumps between sections and steps
  controls discretely instead of easing them.

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

## Why the charts are hand-drawn SVG and not a chart library

The bklit UI registry (`@bklit/*`) was evaluated for these charts. It is good, and
its registry is reachable — but it is React source built on `@visx` and `motion`,
distributed through shadcn, so it needs a bundler. That is incompatible with both
things this demo has to be:

- **`dashboard.html` must open by double-click with no network.** ES modules are
  blocked over `file://`, so a bundled React app cannot be the single offline file
  that goes on the projector.
- **The published copy runs under a strict CSP** that allows scripts only from a
  short CDN allowlist. bklit components are installed source, not a CDN package,
  so they cannot be loaded there either.

Adopting bklit therefore means adding React, Tailwind, a build step and a `dist/`
directory, and giving up the double-click file. The charts here are instead drawn
by `lib/chart.js` (about 160 lines of scale/axis/path helpers) with the shared
hover readout in `lib/hover.js`, which keeps the whole thing dependency-free.

If a hosted React version is ever wanted, bklit is the right choice for it — it
would live alongside this, not replace it.

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
  (paper, Results VI-A to VI-D). This reproduces every row of Table I exactly.
- **Fresh vs fixed** — all 15 seeds, paired t-test, as stated in Results VI-E.
- **n-sweep** — 8 seeds, one absolute tolerance per axis anchored at n=250.

## A correction this demo caught

Building the demo surfaced that the paper's Table I row for **Adult at n=500**
did not reproduce from the released results. Recomputing from
`results/main.csv` on the paper's own three-seed primary grid -- and
cross-checking against the saved `results/analysis_thresholds.csv` -- gave
0.291 / 0.409 / 0.632 / 0.500 (spread 0.480) against the printed
0.429 / 0.610 / 0.788 / 0.576 (spread 0.442). No seed subset, tolerance or
budget reproduced the printed row; it came from a shard set predating the
current merge. **The paper has been corrected**, and all four rows of Table I
now reproduce exactly from the released data. The other three rows were
correct as printed.
