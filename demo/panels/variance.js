/* global window */
(function (root) {
'use strict';

function render(el, D) {
  const C = root.AT.chart;
  const datasets = ['adult', 'credit-g', 'bank-marketing'];

  // ---- domain for the three variance-ratio charts, widened if needed ----
  let yLo = 0.6, yHi = 1.45;
  datasets.forEach((ds) => {
    ['0', '1'].forEach((a) => {
      (D.traj[ds][a] || []).forEach((p) => {
        if (isFinite(p.var_ratio)) { yLo = Math.min(yLo, p.var_ratio); yHi = Math.max(yHi, p.var_ratio); }
      });
    });
  });

  const hoverInfo = {};

  function varRatioChart(ds) {
    const s0 = D.traj[ds]['0'], s1 = D.traj[ds]['1'];
    const gMax = Math.max(...s0.map((p) => p.g));
    const x = C.linear(0, gMax, 40, 330 - 12);
    const y = C.linear(yLo, yHi, 210 - 30, 14);
    const f = C.frame({
      width: 330, height: 230, pad: { t: 14, r: 12, b: 30, l: 40 },
      x, y, xLabel: 'generation', yFormat: (v) => C.fmt(v, 2),
    });
    const p0 = s0.map((p) => ({ x: p.g, y: p.var_ratio, hi: p.var_ratio + p.var_ratio_se, lo: p.var_ratio - p.var_ratio_se }));
    const p1 = s1.map((p) => ({ x: p.g, y: p.var_ratio }));
    let body = f.grid;
    body += C.el('line', { x1: f.pad.l, x2: 330 - f.pad.r, y1: y(1.0).toFixed(2), y2: y(1.0).toFixed(2), class: 'ref-line' });
    body += C.el('text', { x: 330 - f.pad.r, y: y(1.0) - 4, class: 'annot-sm', 'text-anchor': 'end' }, 'faithful = 1.0');
    body += C.band(p0, x, y, { fill: 'var(--danger)', opacity: 0.15 });
    body += C.line(p0, x, y, { stroke: 'var(--danger)', 'stroke-width': 2.4 });
    body += C.line(p1, x, y, { stroke: 'var(--ok)', 'stroke-width': 2, 'stroke-dasharray': '6 4' });
    body += C.el('text', { x: f.pad.l, y: 12, class: 'annot' }, ds);
    hoverInfo['varRatio_' + ds] = {
      x, y, pad: f.pad, width: f.width, height: f.height,
      xs: s0.map((p) => p.g),
    };
    return f.open + body + f.close;
  }

  function axisChart(field, label, invertGood, key) {
    const s0 = D.traj.adult['0'], s1 = D.traj.adult['1'];
    const gMax = Math.max(...s0.map((p) => p.g));
    const vals = s0.map((p) => p[field]).concat(s1.map((p) => p[field]));
    const lo = Math.min(...vals), hi = Math.max(...vals);
    const pad = (hi - lo) * 0.12 || 0.05;
    const x = C.linear(0, gMax, 46, 340 - 14);
    const y = C.linear(lo - pad, hi + pad, 250 - 34, 16);
    const f = C.frame({
      width: 340, height: 250, pad: { t: 16, r: 14, b: 34, l: 46 },
      x, y, xLabel: 'generation', yLabel: label, yFormat: (v) => C.fmt(v, 2),
    });
    const p0 = s0.map((p) => ({ x: p.g, y: p[field], hi: p[field] + (p[field + '_se'] || 0), lo: p[field] - (p[field + '_se'] || 0) }));
    const p1 = s1.map((p) => ({ x: p.g, y: p[field] }));
    let body = f.grid;
    body += C.band(p0, x, y, { fill: 'var(--danger)', opacity: 0.12 });
    body += C.line(p0, x, y, { stroke: 'var(--danger)', 'stroke-width': 2.4 });
    body += C.line(p1, x, y, { stroke: 'var(--ok)', 'stroke-width': 2, 'stroke-dasharray': '6 4' });
    hoverInfo[key] = {
      x, y, pad: f.pad, width: f.width, height: f.height,
      xs: s0.map((p) => p.g),
    };
    return f.open + body + f.close;
  }

  const adult0 = D.traj.adult['0'];
  const start = adult0[0];
  const end = adult0[adult0.length - 1];

  function stat(label, s, e, color) {
    const delta = e - s;
    const sign = delta > 0 ? '+' : '';
    return `<div class="stat"><div class="v" style="color:${color}">${C.fmt(s, 3)} → ${C.fmt(e, 3)}</div>` +
      `<div class="k">${label} (${sign}${C.fmt(delta, 3)})</div></div>`;
  }

  const html = `
    <div class="card grid-3">
      <div data-hover="varRatio_adult">
        <h3 style="margin-bottom:8px">Adult</h3>
        ${varRatioChart('adult')}
      </div>
      <div data-hover="varRatio_credit-g">
        <h3 style="margin-bottom:8px">Credit-G</h3>
        ${varRatioChart('credit-g')}
      </div>
      <div data-hover="varRatio_bank-marketing">
        <h3 style="margin-bottom:8px">Bank Marketing</h3>
        ${varRatioChart('bank-marketing')}
      </div>
    </div>
    <div style="margin-top:-4px">
      ${C.legend([
        { color: 'var(--danger)', label: 'α = 0 (pure self-consumption)' },
        { color: 'var(--ok)', label: 'α = 1 (real-data control)', dashed: true },
      ])}
    </div>
    <div class="card grid-2" style="margin-top:16px">
      <div data-hover="axis_corr_frob">
        <h3>Correlation-matrix error</h3>
        <p class="sub">Adult, α = 0 vs α = 1 (dashed). Rising is bad.</p>
        ${axisChart('corr_frob', 'corr_frob', false, 'axis_corr_frob')}
      </div>
      <div data-hover="axis_cat_support">
        <h3>Categorical support retained</h3>
        <p class="sub">Adult, α = 0 vs α = 1 (dashed). Falling is bad.</p>
        ${axisChart('cat_support', 'cat_support', false, 'axis_cat_support')}
      </div>
    </div>
    <div class="stat-row">
      ${stat('Variance ratio (α=0, gen 0 → ' + end.g + ')', start.var_ratio, end.var_ratio, 'var(--ok)')}
      ${stat('Correlation error (α=0)', start.corr_frob, end.corr_frob, 'var(--danger)')}
      ${stat('Categorical support (α=0)', start.cat_support, end.cat_support, 'var(--danger)')}
      ${stat('Utility, TSTR AUC (α=0)', start.tstr_auc, end.tstr_auc, 'var(--danger)')}
    </div>
    <div class="note">
      A monitor that watches marginal variance for collapse in a tabular copula pipeline will report health
      throughout this run, while dependence structure and rare-category coverage quietly decay underneath it.
      This happens because the rank/CDF transform rebuilds each marginal from its own generation's training
      set every round, so spread is re-normalized by construction and the damage is displaced into
      correlation structure, categorical support and downstream utility instead.
    </div>
  `;

  el.innerHTML = html;

  const AH = root.AT.hover;
  if (AH) {
    datasets.forEach((ds) => {
      const c = hoverInfo['varRatio_' + ds];
      if (!c) return;
      AH.attach(el.querySelector(`[data-hover="varRatio_${ds}"]`), {
        x: c.x, y: c.y, xs: c.xs, pad: c.pad, width: c.width, height: c.height,
        series: [
          { label: 'α = 0', color: 'var(--danger)', values: D.traj[ds]['0'].map((p) => p.var_ratio) },
          { label: 'α = 1', color: 'var(--ok)', values: D.traj[ds]['1'].map((p) => p.var_ratio) },
        ],
        formatX: (v) => 'generation ' + v,
        formatY: (v) => C.fmt(v, 3),
        ariaLabel: ds + ' variance ratio by generation',
        title: (i) => 'generation ' + c.xs[i],
      });
    });

    const corr = hoverInfo.axis_corr_frob;
    if (corr) {
      AH.attach(el.querySelector('[data-hover="axis_corr_frob"]'), {
        x: corr.x, y: corr.y, xs: corr.xs, pad: corr.pad, width: corr.width, height: corr.height,
        series: [
          { label: 'α = 0', color: 'var(--danger)', values: D.traj.adult['0'].map((p) => p.corr_frob) },
          { label: 'α = 1', color: 'var(--ok)', values: D.traj.adult['1'].map((p) => p.corr_frob) },
        ],
        formatX: (v) => 'generation ' + v,
        formatY: (v) => C.fmt(v, 3),
        ariaLabel: 'Adult correlation-matrix error by generation',
        title: (i) => 'generation ' + corr.xs[i],
      });
    }

    const supp = hoverInfo.axis_cat_support;
    if (supp) {
      AH.attach(el.querySelector('[data-hover="axis_cat_support"]'), {
        x: supp.x, y: supp.y, xs: supp.xs, pad: supp.pad, width: supp.width, height: supp.height,
        series: [
          { label: 'α = 0', color: 'var(--danger)', values: D.traj.adult['0'].map((p) => p.cat_support) },
          { label: 'α = 1', color: 'var(--ok)', values: D.traj.adult['1'].map((p) => p.cat_support) },
        ],
        formatX: (v) => 'generation ' + v,
        formatY: (v) => C.fmt(v, 3),
        ariaLabel: 'Adult categorical support retained by generation',
        title: (i) => 'generation ' + supp.xs[i],
      });
    }
  }
}

root.AT = root.AT || {};
root.AT.panels = root.AT.panels || {};
root.AT.panels.variance = { render: render };
}(window));
