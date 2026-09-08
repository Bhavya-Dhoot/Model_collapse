/* global window */
(function (root) {
'use strict';

const LABELS = {
  var_ratio: 'Variance ratio', w1: 'Wasserstein-1 (numeric)', tv_cat: 'Total variation (categorical)',
  corr_frob: 'Correlation structure', cat_support: 'Categorical support',
  tstr_auc: 'Downstream utility (TSTR)', c2st_auc: 'Detectability (C2ST)',
};
const SHORT = {
  var_ratio: 'Var ratio', w1: 'W1', tv_cat: 'TV (cat)', corr_frob: 'Corr',
  cat_support: 'Cat support', tstr_auc: 'TSTR AUC', c2st_auc: 'C2ST AUC',
};
const AXES = ['var_ratio', 'w1', 'tv_cat', 'corr_frob', 'cat_support', 'tstr_auc', 'c2st_auc'];
const NON_VAR = AXES.filter((a) => a !== 'var_ratio');
const PALETTE = {
  var_ratio: '#4e79a7', w1: '#f28e2b', tv_cat: '#e15759', corr_frob: '#59a14f',
  cat_support: '#b07aa1', tstr_auc: '#edc949', c2st_auc: '#76b7b2',
};

const CHART_W = 700, CHART_H = 360, CHART_PAD = { t: 16, r: 16, b: 40, l: 48 };

function computeScales(C, D) {
  const curves = D.degradation_curves;
  const maxY = Math.max(...AXES.flatMap((a) => curves[a].points.map((p) => p.deg))) * 1.08 || 1;
  const x = C.linear(0, 1, CHART_PAD.l, CHART_W - CHART_PAD.r);
  const y = C.linear(0, maxY, CHART_H - CHART_PAD.b, CHART_PAD.t);
  return { x, y };
}

function degradationHoverSpec(C, D, x, y) {
  const curves = D.degradation_curves;
  const xs = curves[AXES[0]].points.map((p) => p.alpha);
  const series = AXES.map((a) => ({
    label: LABELS[a], color: PALETTE[a],
    values: curves[a].points.map((p) => (p.deg === null || p.deg === undefined ? null : p.deg)),
  }));
  return {
    x, y, xs, series,
    pad: CHART_PAD, width: CHART_W, height: CHART_H,
    formatX: (v) => C.fmt(v, 2),
    formatY: (v) => C.fmt(v, 3),
    ariaLabel: 'Degradation vs alpha, all axes',
    title: (i) => 'α = ' + C.fmt(xs[i], 2),
  };
}

function renderChart(C, D, sel, x, y) {
  const width = CHART_W, height = CHART_H, pad = CHART_PAD;
  const curves = D.degradation_curves;
  const f = C.frame({
    width, height, pad, x, y, xLabel: 'α', yLabel: 'Degradation',
    xFormat: (v) => C.fmt(v, 2), yFormat: (v) => C.fmt(v, 2),
  });
  let g = f.grid;
  for (const a of AXES) {
    const isSel = a === sel;
    const pts = curves[a].points.map((p) => ({ x: p.alpha, y: p.deg }));
    g += C.line(pts, x, y, {
      stroke: PALETTE[a], 'stroke-width': isSel ? 2.5 : 1.5, opacity: isSel ? 1 : 0.25,
    });
    g += C.dots(pts, x, y, isSel ? 3.5 : 2.5, { fill: PALETTE[a], opacity: isSel ? 1 : 0.25 });
  }
  const sc = curves[sel];
  g += C.el('line', { x1: pad.l, x2: width - pad.r, y1: y(sc.tol), y2: y(sc.tol), class: 'ref-line' });
  if (sc.alpha_star !== null && sc.alpha_star !== undefined) {
    const ax = x(sc.alpha_star);
    g += C.el('line', { x1: ax, x2: ax, y1: pad.t, y2: height - pad.b, class: 'ref-line' });
    g += C.el('text', { x: ax + 6, y: pad.t + 14, class: 'annot' }, `α★ = ${C.fmt(sc.alpha_star, 3)}`);
  }
  const svg = f.open + g + f.close;
  const legend = C.legend(AXES.map((a) => ({ color: PALETTE[a], label: LABELS[a] })));
  return svg + legend;
}

function renderControls(sel) {
  return `<div class="controls"><div class="ctl"><label>Axis</label><div class="seg" data-role="axis-seg">${
    AXES.map((a) => `<button type="button" data-axis="${a}" aria-pressed="${a === sel}">${LABELS[a]}</button>`).join('')
  }</div></div></div>`;
}

function renderTable(C, D) {
  const rows = D.alpha_star_table;
  const heads = ['Dataset', 'n', ...AXES.map((a) => SHORT[a]), 'Spread'];
  const body = rows.map((r) => {
    const vals = NON_VAR.map((a) => r[a]);
    const max = Math.max(...vals), min = Math.min(...vals);
    const cells = AXES.map((a) => {
      let cls = a === 'var_ratio' ? 'num muted' : 'num';
      if (a !== 'var_ratio') {
        if (r[a] === max) cls += ' hi';
        else if (r[a] === min) cls += ' lo';
      }
      return `<td class="${cls}">${C.fmt(r[a], 3)}</td>`;
    }).join('');
    return `<tr><td>${r.dataset}</td><td class="num">${r.n}</td>${cells}<td class="num">${C.fmt(r.spread, 3)}</td></tr>`;
  }).join('');
  return `<div style="overflow-x:auto"><table><thead><tr>${
    heads.map((h) => `<th>${h}</th>`).join('')
  }</tr></thead><tbody>${body}</tbody></table></div>`;
}

function statRow(C, D) {
  const row = D.alpha_star_table.find((r) => r.dataset === 'adult' && r.n === 2000);
  if (!row) return '';
  const strict = row.binding_axis;
  let loose = NON_VAR[0];
  for (const a of NON_VAR) if (row[a] < row[loose]) loose = a;
  return `<div class="stat-row">
    <div class="stat"><div class="v">${LABELS[strict]}</div><div class="k">Strictest axis · α★ = ${C.fmt(row[strict], 3)}</div></div>
    <div class="stat"><div class="v">${LABELS[loose]}</div><div class="k">Loosest axis · α★ = ${C.fmt(row[loose], 3)}</div></div>
    <div class="stat"><div class="v">${C.fmt(row.ratio_excl_var, 1)}×</div><div class="k">Ratio (loosest / strictest, var_ratio excluded)</div></div>
  </div>`;
}

function render(el, D) {
  const C = root.AT.chart;
  const hover = root.AT.hover;
  let sel = 'corr_frob';
  const scales = computeScales(C, D);

  const html = `
    <div class="card">
      <h3>Where each axis crosses its tolerance</h3>
      <p class="sub">Degradation vs. α for every axis at Adult, n = 2000, copula, fixed anchoring.</p>
      ${renderControls(sel)}
      <div data-role="chart-wrap">${renderChart(C, D, sel, scales.x, scales.y)}</div>
      <p class="sub">Degradation is measured at the terminal generation relative to the α = 1 baseline and clipped at
        zero; α★ is the smallest α whose degradation is within tolerance and stays within it for every larger α,
        linearly interpolated between the two bracketing grid points, so an isolated noisy dip is not counted as a
        threshold.</p>
    </div>
    <div class="card">
      <h3>Per-axis thresholds across datasets</h3>
      <p class="sub">α★ for each axis, by dataset and sample size.</p>
      ${statRow(C, D)}
      ${renderTable(C, D)}
      <p class="sub">var_ratio is listed for completeness only; it is not a copula collapse axis and is excluded
        from the headline spread.</p>
    </div>
    <div class="note">Validating a synthetic pipeline against a single aggregate score can pass while the rare
      categories that made the dataset worth synthesizing are already gone; a safety specification has to be
      stated per axis.</div>
  `;
  el.innerHTML = html;

  const wrap = el.querySelector('[data-role="chart-wrap"]');
  const seg = el.querySelector('[data-role="axis-seg"]');

  function attachHover() {
    if (hover) hover.attach(wrap, degradationHoverSpec(C, D, scales.x, scales.y));
  }
  attachHover();

  const rerender = hover ? hover.throttle(doRerender) : doRerender;
  function doRerender(btn) {
    wrap.innerHTML = renderChart(C, D, sel, scales.x, scales.y);
    attachHover();
    seg.querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', String(b === btn)));
  }

  seg.addEventListener('click', (ev) => {
    const btn = ev.target.closest('button[data-axis]');
    if (!btn) return;
    sel = btn.getAttribute('data-axis');
    rerender(btn);
  });
}

root.AT = root.AT || {};
root.AT.panels = root.AT.panels || {};
root.AT.panels.axes = { render: render };
}(window));
