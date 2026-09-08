/* global window */
(function (root) {
'use strict';

const NAME = { gaussian_copula: 'Gaussian copula', tvae: 'TVAE-lite', ctgan: 'CTGAN-lite' };
const THRESH = 0.05;

function renderSupport(C, D) {
  const width = 700, height = 300, pad = { t: 16, r: 90, b: 40, l: 130 };
  const rows = D.architectures;
  const vals = rows.flatMap((r) => [r.a0.cat_support.start, r.a0.cat_support.end, r.a1.cat_support.start, r.a1.cat_support.end]);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const padAmt = (hi - lo) * 0.12 || 0.05;
  const x = C.linear(lo - padAmt, hi + padAmt, pad.l, width - pad.r);
  const y = C.linear(0, rows.length - 1, pad.t, height - pad.b);
  const f = C.frame({
    width, height, pad, x, y, xLabel: 'Categorical support', yTicks: [],
    xFormat: (v) => C.fmt(v, 2), yGrid: false,
  });
  let g = f.grid;
  const rowH = (height - pad.b - pad.t) / (rows.length - 1 || 1);
  const yOff = { a0: -0.16, a1: 0.16 };
  rows.forEach((r, i) => {
    const yc = y(i);
    g += C.el('text', { x: pad.l - 10, y: yc - 10, class: 'tick tick-y' }, NAME[r.synth]);
    g += C.el('text', { x: pad.l - 10, y: yc + 6, class: 'annot-sm' }, `gen 0→${r.terminal_generation}`);
    [['a0', 'var(--danger)'], ['a1', 'var(--ok)']].forEach(([key, color]) => {
      const s = r[key].cat_support;
      const yy = y(i + yOff[key]);
      const yFix = () => yy;
      g += C.line([{ x: s.start, y: yy }, { x: s.end, y: yy }], x, yFix, { stroke: color, 'stroke-width': 2.5 });
      g += C.dots([{ x: s.start, y: yy }], x, () => yy, 4, { fill: color, opacity: 0.45 });
      g += C.dots([{ x: s.end, y: yy }], x, () => yy, 4, { fill: color });
      const label = `${C.fmt(s.start, 3)} → ${C.fmt(s.end, 3)}`;
      const rightEnd = x(s.end) >= x(s.start);
      g += C.el('text', {
        x: x(s.end) + (rightEnd ? 8 : -8), y: yy + 3.5, class: 'annot-sm',
        'text-anchor': rightEnd ? 'start' : 'end',
      }, label);
    });
  });
  const svg = f.open + g + f.close;
  const legend = C.legend([
    { color: 'var(--danger)', label: 'α = 0 (self-consumption)' },
    { color: 'var(--ok)', label: 'α = 1 (real-data control)' },
  ]);
  return svg + legend;
}

function renderVariance(C, D) {
  const width = 700, height = 280, pad = { t: 16, r: 16, b: 40, l: 52 };
  const rows = D.architectures;
  const vals = rows.flatMap((r) => [r.a0.var_ratio.start, r.a0.var_ratio.end, 1]);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const padAmt = (hi - lo) * 0.15 || 0.1;
  const x = C.linear(0, rows.length - 1, pad.l, width - pad.r);
  const y = C.linear(Math.min(0, lo - padAmt), hi + padAmt, height - pad.b, pad.t);
  const f = C.frame({
    width, height, pad, x, y, yLabel: 'Variance ratio (α = 0)', xTicks: [],
    yFormat: (v) => C.fmt(v, 2), xGrid: false,
  });
  let g = f.grid;
  g += C.el('line', { x1: pad.l, x2: width - pad.r, y1: y(1), y2: y(1), class: 'ref-line' });
  g += C.el('text', { x: width - pad.r - 2, y: y(1) - 5, class: 'annot-sm', 'text-anchor': 'end' }, 'ratio = 1 (no shift)');
  const barW = 26;
  rows.forEach((r, i) => {
    const cx = x(i);
    const vr = r.a0.var_ratio;
    const delta = vr.end - vr.start;
    [['start', vr.start, 'var(--ink-3)'], ['end', vr.end, delta < -THRESH ? 'var(--ok)' : (delta > THRESH ? 'var(--danger)' : 'var(--ink-3)')]].forEach(([k, v, color], j) => {
      const bx = cx - barW - 4 + j * (barW + 8);
      const yv = y(v), y0 = y(0);
      g += C.el('rect', {
        x: bx, y: Math.min(yv, y0), width: barW, height: Math.abs(y0 - yv), fill: color, opacity: k === 'start' ? 0.5 : 0.9, rx: 3,
      });
      g += C.el('text', { x: bx + barW / 2, y: Math.min(yv, y0) - 6, class: 'annot-sm', 'text-anchor': 'middle' }, C.fmt(v, 3));
    });
    g += C.el('text', { x: cx, y: height - pad.b + 30, class: 'tick tick-x' }, NAME[r.synth]);
  });
  const svg = f.open + g + f.close;
  const pills = rows.map((r) => {
    const vr = r.a0.var_ratio;
    const delta = vr.end - vr.start;
    const word = Math.abs(delta) < THRESH ? 'flat' : (delta < 0 ? 'contracts' : 'inflates');
    const pillClass = word === 'inflates' ? 'bad' : (word === 'contracts' ? 'ok' : '');
    return `<span class="pill${pillClass ? ' ' + pillClass : ''}">${NAME[r.synth]}: ${word}</span>`;
  }).join(' ');
  return svg + `<div class="stat-row">${pills}</div>`;
}

function renderGate(C, D) {
  const rows = D.gen0_gate;
  const heads = ['Synthesizer', 'Categorical support', 'Categorical TV', 'TSTR AUC', 'Sec / gen'];
  const body = rows.map((r) => `<tr>
    <td>${NAME[r.synth]}</td>
    <td class="num">${C.fmt(r.cat_support, 3)}</td>
    <td class="num">${C.fmt(r.tv_cat, 3)}</td>
    <td class="num">${C.fmt(r.tstr_auc, 3)}</td>
    <td class="num">${C.fmt(r.sec_per_gen, 1)}</td>
  </tr>`).join('');
  const table = `<table><thead><tr>${heads.map((h) => `<th>${h}</th>`).join('')}</tr></thead><tbody>${body}</tbody></table>`;
  const note = `<p class="sub">This gate exists because a synthesizer that is already degraded before any self-consumption cannot be used to study the loop: its generation-0 fidelity must be checked first. All three synthesizers qualify.</p>`;
  return `<div>${table}</div><div>${note}</div>`;
}

function render(el, D) {
  const C = root.AT.chart;
  const html = `
    <div class="card">
      <h3>Categorical support: lost under self-consumption, restored by anchoring</h3>
      <p class="sub">Adult, n = 2000, fixed anchoring. Terminal generation differs by architecture: the copula runs to generation 12, the two neural synthesizers to generation 8.</p>
      ${renderSupport(C, D)}
    </div>
    <div class="card">
      <h3>Variance is an unreliable detector in all three, but differently</h3>
      <p class="sub">Start → end variance ratio at α = 0, generation 0 to terminal generation. A move smaller than ${C.fmt(THRESH, 2)} is called flat.</p>
      ${renderVariance(C, D)}
    </div>
    <div class="card grid-2">
      <h3>Generation-0 fidelity gate</h3>
      ${renderGate(C, D)}
    </div>
    <div class="note warn">
      <strong>Scope limit.</strong> The exact closed-form theory covers Gaussian maximum-likelihood fitting only. For TVAE and CTGAN the quantitative theory is an analogy, not a derivation; only the two qualitative findings carry over — support collapse under self-consumption with recovery under anchoring, and the unreliability of variance as a detector.
    </div>
  `;
  el.innerHTML = html;
}

root.AT = root.AT || {};
root.AT.panels = root.AT.panels || {};
root.AT.panels.arch = { render: render };
}(window));
