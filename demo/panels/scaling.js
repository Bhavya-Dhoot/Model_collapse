/* global window */
(function (root) {
'use strict';

var LABELS = {
  var_ratio: 'Variance ratio', w1: 'Wasserstein-1', tv_cat: 'Total variation',
  corr_frob: 'Correlation structure', cat_support: 'Categorical support',
  tstr_auc: 'Utility (TSTR)', c2st_auc: 'Detectability (C2ST)'
};
var COLORS = {
  w1: '#7c3aed', tv_cat: '#0891b2', corr_frob: '#c2410c',
  cat_support: '#15803d', c2st_auc: '#be185d'
};
var EXCLUDED = ['var_ratio', 'tstr_auc'];

function isNum(v) { return v !== null && v !== undefined && isFinite(v); }

function render(el, D) {
  var C = root.AT.chart;
  var S = root.AT.sim;
  var nd = D.nscaling;
  var ns = nd.ns;
  var n0 = ns[0];
  var state = { eps: 0.01, ni: 0 };

  function scalingChart() {
    var allVals = [];
    nd.fit_axes.concat(EXCLUDED).forEach(function (ax) {
      nd.axes[ax].alpha_star.forEach(function (v) { if (isNum(v)) allVals.push(v); });
    });
    var yMax = Math.max.apply(null, allVals) * 1.15;

    var x = C.log(ns[0], ns[ns.length - 1], 52, 700 - 16);
    var y = C.linear(0, yMax, 380 - 40, 16);
    var f = C.frame({
      width: 700, height: 380, pad: { t: 16, r: 16, b: 40, l: 52 },
      x: x, y: y, xLabel: 'training budget n (log scale)', yLabel: 'α★',
      xFormat: function (v) { return C.fmt(v, 0); }, yFormat: function (v) { return C.fmt(v, 2); },
    });
    var body = f.grid;

    var anchorVals = nd.fit_axes.map(function (ax) { return nd.axes[ax].alpha_star[0]; }).filter(isNum);
    var anchor = anchorVals.reduce(function (a, b) { return a + b; }, 0) / anchorVals.length;

    function sample(fn) {
      var pts = [];
      var l0 = Math.log10(ns[0]), l1 = Math.log10(ns[ns.length - 1]);
      for (var i = 0; i <= 40; i++) {
        var n = Math.pow(10, l0 + (l1 - l0) * i / 40);
        pts.push({ x: n, y: fn(n) });
      }
      return pts;
    }
    var measured = sample(function (n) { return anchor * Math.pow(n / n0, -nd.beta); });
    var theory = sample(function (n) { return Math.min(yMax, anchor * Math.pow(n / n0, -1)); });
    var ciLo = sample(function (n) { return anchor * Math.pow(n / n0, -nd.beta_ci[1]); });
    var ciHi = sample(function (n) { return anchor * Math.pow(n / n0, -nd.beta_ci[0]); });
    var ribbon = ciLo.map(function (p, i) { return { x: p.x, lo: p.y, hi: ciHi[i].y }; });

    body += C.band(ribbon, x, y, { fill: 'var(--accent)', opacity: 0.12 });

    EXCLUDED.forEach(function (ax) {
      var pts = ns.map(function (n, i) { return { x: n, y: nd.axes[ax].alpha_star[i] }; });
      body += C.line(pts, x, y, { stroke: 'var(--ink-3)', 'stroke-width': 1.25, 'stroke-dasharray': '2 3' });
    });

    nd.fit_axes.forEach(function (ax) {
      var pts = ns.map(function (n, i) { return { x: n, y: nd.axes[ax].alpha_star[i] }; });
      body += C.line(pts, x, y, { stroke: COLORS[ax], 'stroke-width': 2 });
      body += C.dots(pts, x, y, 3, { fill: COLORS[ax] });
    });

    body += C.line(theory, x, y, { stroke: 'var(--danger)', 'stroke-width': 2, 'stroke-dasharray': '6 4' });
    body += C.line(measured, x, y, { stroke: 'var(--accent)', 'stroke-width': 3 });

    var legendItems = nd.fit_axes.map(function (ax) { return { color: COLORS[ax], label: LABELS[ax] }; })
      .concat(EXCLUDED.map(function (ax) { return { color: 'var(--ink-3)', label: LABELS[ax] + ' (excluded from fit)', dashed: true }; }))
      .concat([
        { color: 'var(--accent)', label: 'measured: n^−' + C.fmt(nd.beta, 2) },
        { color: 'var(--danger)', label: 'theory: n^−1', dashed: true },
      ]);

    return '<div class="card">' +
      '<h3>α★ against training budget</h3>' +
      '<p class="sub">Anchoring threshold required at each training budget, per axis, against the fitted and theoretical power laws (shared anchor at n=' + C.fmt(n0, 0) + ').</p>' +
      f.open + body + f.close + C.legend(legendItems) + '</div>';
  }

  function statsCard() {
    var ci0 = nd.beta_ci[0], ci1 = nd.beta_ci[1];
    return '<div class="card">' +
      '<h3>Measured exponent</h3>' +
      '<p class="sub">Bootstrap fit of α★ ∝ n^−β, pooled across the fit axes.</p>' +
      '<div class="stat-row">' +
      '<div class="stat"><div class="v">' + C.fmt(nd.beta, 2) + '</div><div class="k">β (point estimate)</div></div>' +
      '<div class="stat"><div class="v">[' + C.fmt(ci0, 2) + ', ' + C.fmt(ci1, 2) + ']</div><div class="k">95% bootstrap interval</div></div>' +
      '</div>' +
      '<p style="margin-top:12px">The interval excludes 0, so α★ genuinely falls as the training budget grows. It also excludes 1 by a wide margin, so this is not a 1/n law: the pooled exponent (β=' + C.fmt(nd.beta, 2) + ') is roughly a fifth of what a 1/n decay would require.</p>' +
      '</div>';
  }

  function calcCard() {
    var n = ns[state.ni];
    var eps = state.eps;
    var exact = S.alphaStarExact(eps, n);
    var approx = S.alphaStarApprox(eps, n);
    var rows = isNum(exact) ? Math.ceil(exact * n) : NaN;
    return '<div class="card">' +
      '<h3>Budget calculator</h3>' +
      '<p class="sub">IDEALISED Gaussian bound: the closed-form α★ for a mean-shift tolerance ε at budget n, under the model’s Gaussian assumptions.</p>' +
      '<div class="controls">' +
      '<div class="ctl"><label>Tolerance ε = ' + C.fmt(eps, 3) + '</label>' +
      '<input type="range" data-role="eps" min="0.005" max="0.10" step="0.001" value="' + eps + '"/></div>' +
      '<div class="ctl"><label>Budget n</label><div class="seg" data-role="ns">' +
      ns.map(function (v) { return '<button data-n="' + v + '" aria-pressed="' + (v === n) + '">' + C.fmt(v, 0) + '</button>'; }).join('') +
      '</div></div>' +
      '</div>' +
      '<div class="stat-row">' +
      '<div class="stat"><div class="v">' + C.fmt(exact, 3) + '</div><div class="k">α★ (exact)</div></div>' +
      '<div class="stat"><div class="v">' + C.fmt(rows, 0) + '</div><div class="k">real rows required (⌈α★·n⌉)</div></div>' +
      '<div class="stat"><div class="v">' + C.fmt(approx, 3) + '</div><div class="k">α★ (leading-order approx.)</div></div>' +
      '</div>' +
      '<p class="sub" style="margin-top:10px">The empirical requirement is larger and falls more slowly than this bound — treat it as a floor, not a recommendation.</p>' +
      '</div>';
  }

  function panelHtml() {
    var w1 = nd.axes.w1.alpha_star, cf = nd.axes.corr_frob.alpha_star;
    return scalingChart() +
      '<div class="grid-2">' + statsCard() + calcCard() + '</div>' +
      '<div class="note">Scale makes nearly every axis cheaper to protect — Wasserstein-1 falls from ' +
      C.fmt(w1[0], 2) + ' to ' + C.fmt(w1[w1.length - 1], 2) + ' — except dependence structure, which stays ' +
      'roughly flat across the whole range (correlation structure: ' + C.fmt(cf[0], 2) + ' → ' + C.fmt(cf[cf.length - 1], 2) +
      '). That is why the pooled exponent falls far short of 1.</div>';
  }

  function wire() {
    var slider = el.querySelector('[data-role=eps]');
    if (slider) {
      slider.addEventListener('input', function () {
        state.eps = +slider.value;
        draw();
      });
    }
    var segBtns = el.querySelectorAll('[data-role=ns] button');
    for (var i = 0; i < segBtns.length; i++) {
      segBtns[i].addEventListener('click', function (ev) {
        state.ni = ns.indexOf(+ev.currentTarget.getAttribute('data-n'));
        draw();
      });
    }
  }

  function draw() {
    el.innerHTML = panelHtml();
    wire();
  }

  draw();
}

root.AT = root.AT || {};
root.AT.panels = root.AT.panels || {};
root.AT.panels.scaling = { render: render };
}(window));
