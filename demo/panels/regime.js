/* global window */
(function (root) {
'use strict';

function render(el, D) {
  var C = root.AT.chart;
  var rows = D.regime;
  var gen = D.meta.terminal_generation;

  /* ---------- Card 1: grouped bars over alpha ---------- */
  var w1 = 700, h1 = 360, pad1 = { t: 46, r: 20, b: 46, l: 54 };
  var n = rows.length;
  var slot = (w1 - pad1.l - pad1.r) / n;
  var barW = slot * 0.62 / 2 - 4;

  var lo = Math.min.apply(null, rows.map(function (r) { return Math.min(r.fixed - r.fixed_se, r.fresh - r.fresh_se); }));
  var hi = Math.max.apply(null, rows.map(function (r) { return Math.max(r.fixed + r.fixed_se, r.fresh + r.fresh_se); }));
  var pad = (hi - lo) * 0.18;
  var y0 = lo - pad, y1 = hi + pad;
  var yScale = C.linear(y0, y1, h1 - pad1.b, pad1.t);
  var base = h1 - pad1.b;

  var g1 = '';
  C.ticks(y0, y1, 5).forEach(function (t) {
    if (t < y0 || t > y1) return;
    var py = yScale(t);
    g1 += C.el('line', { x1: pad1.l, x2: w1 - pad1.r, y1: py, y2: py, class: 'grid' });
    g1 += C.el('text', { x: pad1.l - 8, y: py + 3.5, class: 'tick tick-y' }, C.fmt(t, 2));
  });
  g1 += C.el('line', { x1: pad1.l, x2: w1 - pad1.r, y1: base, y2: base, class: 'axis' });

  function bar(cx, v, se, color, title) {
    var x = cx - barW / 2;
    var yTop = yScale(v);
    var s = '';
    s += C.el('rect', { x: x, y: yTop, width: barW, height: base - yTop, fill: color }, C.el('title', {}, title));
    var seTop = yScale(v + se), seBot = yScale(v - se);
    s += C.el('line', { x1: cx, x2: cx, y1: seTop, y2: seBot, stroke: 'var(--ink)', 'stroke-width': 1 });
    s += C.el('line', { x1: cx - 3, x2: cx + 3, y1: seTop, y2: seTop, stroke: 'var(--ink)', 'stroke-width': 1 });
    s += C.el('line', { x1: cx - 3, x2: cx + 3, y1: seBot, y2: seBot, stroke: 'var(--ink)', 'stroke-width': 1 });
    return s;
  }

  rows.forEach(function (r, i) {
    var cx = pad1.l + slot * (i + 0.5);
    g1 += bar(cx - barW / 2 - 3, r.fixed, r.fixed_se, 'var(--ink-3)', 'fixed, α = ' + r.alpha + ': ' + C.fmt(r.fixed, 3));
    g1 += bar(cx + barW / 2 + 3, r.fresh, r.fresh_se, 'var(--accent)', 'fresh, α = ' + r.alpha + ': ' + C.fmt(r.fresh, 3));
    g1 += C.el('text', { x: cx, y: h1 - pad1.b + 18, class: 'tick tick-x' }, String(r.alpha));
    var sig = r.p < 0.05;
    var label = r.p < 0.001 ? 'p<0.001' : 'p=' + C.fmt(r.p, 3);
    g1 += C.el('text', {
      x: cx, y: 18, class: 'annot-sm', fill: sig ? 'var(--ok)' : 'var(--ink-3)', 'text-anchor': 'middle',
    }, label);
  });

  var card1 = '<div class="card">' +
    '<h3>Categorical support at generation ' + gen + '</h3>' +
    '<p class="sub">Fixed vs. fresh anchoring, by anchor fraction &alpha;. Bars show mean support retained across ' + rows[0].seeds + ' seeds; whiskers are &plusmn;1 SE.</p>' +
    '<svg viewBox="0 0 ' + w1 + ' ' + h1 + '" class="chart" preserveAspectRatio="xMidYMid meet" role="img">' +
    C.el('text', { x: pad1.l, y: 30, class: 'axis-label', 'text-anchor': 'start' }, 'Categorical support retained (axis truncated)') +
    g1 + '</svg>' +
    C.legend([{ label: 'fixed anchor', color: 'var(--ink-3)' }, { label: 'fresh anchor', color: 'var(--accent)' }]) +
    '</div>';

  /* ---------- Card 2a: diff vs alpha ---------- */
  var maxAlpha = Math.max.apply(null, rows.map(function (r) { return r.alpha; }));
  var diffs = rows.map(function (r) { return r.diff; });
  var dLo = Math.min.apply(null, diffs.concat([0])), dHi = Math.max.apply(null, diffs.concat([0]));
  var dPad = (dHi - dLo) * 0.2 || 0.01;
  var xS = C.linear(0, maxAlpha, 56, 500);
  var yS = C.linear(dLo - dPad, dHi + dPad, 260, 16);
  var fr = C.frame({
    width: 520, height: 300, pad: { t: 16, r: 20, b: 40, l: 56 },
    x: xS, y: yS,
    xTicks: rows.map(function (r) { return r.alpha; }),
    xFormat: function (v) { return String(v); },
    yFormat: function (v) { return C.fmt(v, 2); },
    xLabel: 'α (anchor fraction)', yLabel: 'Δ support (fresh − fixed)',
  });
  var pts = rows.map(function (r) { return { x: r.alpha, y: r.diff }; });
  var g2 = fr.grid;
  g2 += C.el('line', { x1: fr.pad.l, x2: fr.width - fr.pad.r, y1: yS(0), y2: yS(0), class: 'ref-line' });
  g2 += C.line(pts, xS, yS, { stroke: 'var(--ink-3)', 'stroke-width': 1.5 });
  var peak = rows.reduce(function (a, b) { return b.diff > a.diff ? b : a; });
  rows.forEach(function (r) {
    var sig = r.p < 0.05;
    g2 += C.el('circle', {
      cx: xS(r.alpha).toFixed(2), cy: yS(r.diff).toFixed(2), r: 4.5,
      fill: sig ? 'var(--ok)' : 'var(--paper)', stroke: sig ? 'var(--ok)' : 'var(--ink-3)', 'stroke-width': 1.5,
    });
  });
  g2 += C.el('text', {
    x: xS(peak.alpha), y: yS(peak.diff) - 10, class: 'annot', 'text-anchor': 'middle',
  }, '+' + C.fmt(peak.diff, 3) + ' at α=' + peak.alpha);

  var card2a = '<div class="card"><h3>The gap, and when it is real</h3>' +
    '<p class="sub">Filled markers: paired t-test over seeds gives p&lt;0.05. Hollow: not distinguishable from zero.</p>' +
    '<div id="regime-diff-chart">' + fr.open + g2 + fr.close + '</div></div>';

  /* ---------- Card 2b: table ---------- */
  var trs = rows.map(function (r) {
    var sig = r.p < 0.05;
    var pTxt = r.p < 0.001 ? '<0.001' : C.fmt(r.p, 3);
    return '<tr><td>' + String(r.alpha) + '</td>' +
      '<td class="num">' + C.fmt(r.fixed, 3) + '</td>' +
      '<td class="num">' + C.fmt(r.fresh, 3) + '</td>' +
      '<td class="num' + (sig ? ' lo' : '') + '">' + C.fmt(r.diff, 3) + '</td>' +
      '<td class="num">' + pTxt + '</td></tr>';
  }).join('');
  var card2b = '<div class="card"><h3>Support by anchor regime</h3>' +
    '<p class="sub">Difference in green where the paired t-test over seeds is significant at p&lt;0.05.</p>' +
    '<table><thead><tr><th>α</th><th>fixed</th><th>fresh</th><th>difference</th><th>p</th></tr></thead>' +
    '<tbody>' + trs + '</tbody></table></div>';

  var nSig = rows.filter(function (r) { return r.p < 0.05; }).length;
  var zeroRow = rows.filter(function (r) { return r.alpha === 0; })[0];
  var zeroTxt = zeroRow ? (zeroRow.p < 0.05 ? 'significant (p=' + C.fmt(zeroRow.p, 3) + ')' : 'not significant (p=' + C.fmt(zeroRow.p, 3) + ')') : '—';

  var stats = '<div class="stat-row">' +
    '<div class="stat"><div class="v">+' + C.fmt(peak.diff, 3) + '</div><div class="k">peak improvement, at α=' + peak.alpha + '</div></div>' +
    '<div class="stat"><div class="v">' + nSig + ' / ' + n + '</div><div class="k">α values with p&lt;0.05</div></div>' +
    '<div class="stat"><div class="v">' + zeroTxt + '</div><div class="k">result at α=0</div></div>' +
    '</div>';

  var note = '<p class="note">Numeric drift is a contraction toward the truth that any real data point can pull back; ' +
    'a categorical level that has vanished from the anchor can only reappear in an anchor row that still contains it, ' +
    'so re-sampling the same anchor rows cannot restore a category they never held. Where the anchor is too small to ' +
    'sample the tail (the smallest α values), fixed and fresh anchoring are statistically indistinguishable, and the ' +
    'two regimes do not separate on correlation error at any α tested.</p>';

  el.innerHTML = card1 +
    '<div class="grid-2">' + card2a + card2b + '</div>' +
    stats + note;

  // hover readout on the diff chart; the p-value has no position on this y
  // axis, so it goes in the tooltip heading instead of a second, misleading dot
  if (root.AT.hover) {
    root.AT.hover.attach(el.querySelector('#regime-diff-chart'), {
      x: xS, y: yS,
      xs: rows.map(function (r) { return r.alpha; }),
      series: [
        { label: 'fresh − fixed', color: 'var(--accent)', values: diffs },
      ],
      pad: fr.pad, width: fr.width, height: fr.height,
      formatX: function (v) { return String(v); },
      formatY: function (v) { return C.fmt(v, 3); },
      ariaLabel: 'Fresh minus fixed support difference, by anchor fraction alpha',
      title: function (i) {
        var r = rows[i];
        var pTxt = r.p < 0.001 ? '<0.001' : C.fmt(r.p, 3);
        return 'α = ' + r.alpha + ' · p = ' + pTxt;
      },
    });
  }
}

root.AT = root.AT || {};
root.AT.panels = root.AT.panels || {};
root.AT.panels.regime = { render: render };
}(window));
