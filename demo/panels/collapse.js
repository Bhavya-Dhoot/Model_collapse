/* global window */
(function (root) {
  'use strict';

  function render(el, D) {
    var C = root.AT.chart;
    var datasets = Object.keys(D.traj);
    // open on alpha = 0: this section is about collapse, so it must show the
    // collapsing trajectory, not the flat alpha = 1 control
    var state = { ds: datasets.indexOf('adult') >= 0 ? 'adult' : datasets[0], alpha: '0' };

    function alphas(ds) {
      return Object.keys(D.traj[ds]).sort(function (a, b) { return +a - +b; });
    }
    if (alphas(state.ds).indexOf(state.alpha) < 0) state.alpha = alphas(state.ds).slice(-1)[0];

    var hoverInfo = {};

    function chart1(ds, hi) {
      var as = alphas(ds);
      var G = D.meta.terminal_generation;
      var series = as.map(function (a) { return { a: a, pts: D.traj[ds][a] }; });
      var ys = [];
      series.forEach(function (s) { s.pts.forEach(function (p) { if (isFinite(p.tstr_auc)) ys.push(p.tstr_auc); }); });
      ys.push(D.ceiling[ds]);
      var lo = Math.min.apply(null, ys), hiY = Math.max.apply(null, ys);
      var pad = (hiY - lo) * 0.08 || 0.02;
      var x = C.linear(0, G, 60, 700 - 16);
      var y = C.linear(lo - pad, hiY + pad, 330 - 40, 16);
      var f = C.frame({
        width: 700, height: 330, pad: { t: 16, r: 16, b: 40, l: 60 },
        x: x, y: y, xLabel: 'generation', yLabel: 'TSTR AUC',
        xTicks: [0, 2, 4, 6, 8, 10, 12], yFormat: function (v) { return C.fmt(v, 2); },
      });
      var body = f.grid;
      body += C.el('line', {
        x1: f.pad.l, x2: f.width - f.pad.r, y1: y(D.ceiling[ds]), y2: y(D.ceiling[ds]), class: 'ref-line',
      });
      body += C.el('text', {
        x: f.width - f.pad.r - 4, y: y(D.ceiling[ds]) - 6, class: 'annot-sm', 'text-anchor': 'end',
      }, 'train-on-real ceiling');

      series.forEach(function (s) {
        var pts = s.pts.map(function (p) { return { x: p.g, y: p.tstr_auc }; });
        var isHi = s.a === hi;
        if (isHi) {
          var band = s.pts.map(function (p) {
            return { x: p.g, lo: p.tstr_auc - p.tstr_auc_se, hi: p.tstr_auc + p.tstr_auc_se };
          });
          body += C.band(band, x, y, { fill: C.viridis(+s.a), opacity: 0.16 });
        }
        body += C.line(pts, x, y, {
          stroke: C.viridis(+s.a),
          'stroke-width': isHi ? 2.6 : 1.4,
          opacity: isHi ? 1 : 0.28,
        });
      });
      hoverInfo.chart1 = {
        x: x, y: y, pad: f.pad, width: f.width, height: f.height,
        xs: D.traj[ds][hi].map(function (p) { return p.g; }),
      };
      return f.open + body + f.close;
    }

    function smallChart(ds, hi, field, label, key) {
      var as = alphas(ds);
      var G = D.meta.terminal_generation;
      var series = as.map(function (a) { return { a: a, pts: D.traj[ds][a] }; });
      var ys = [];
      series.forEach(function (s) { s.pts.forEach(function (p) { if (isFinite(p[field])) ys.push(p[field]); }); });
      var lo = Math.min.apply(null, ys), hiY = Math.max.apply(null, ys);
      var pad = (hiY - lo) * 0.08 || 0.02;
      var x = C.linear(0, G, 48, 340 - 12);
      var y = C.linear(lo - pad, hiY + pad, 240 - 34, 14);
      var f = C.frame({
        width: 340, height: 240, pad: { t: 14, r: 12, b: 34, l: 48 },
        x: x, y: y, xLabel: 'generation', yLabel: label,
        xTicks: [0, 4, 8, 12], yFormat: function (v) { return C.fmt(v, 2); },
      });
      var body = f.grid;
      series.forEach(function (s) {
        var pts = s.pts.map(function (p) { return { x: p.g, y: p[field] }; });
        var isHi = s.a === hi;
        body += C.line(pts, x, y, {
          stroke: C.viridis(+s.a),
          'stroke-width': isHi ? 2.6 : 1.4,
          opacity: isHi ? 1 : 0.28,
        });
      });
      hoverInfo[key] = {
        x: x, y: y, pad: f.pad, width: f.width, height: f.height,
        xs: D.traj[ds][hi].map(function (p) { return p.g; }),
      };
      return f.open + body + f.close;
    }

    function statRow(ds, a) {
      var pts = D.traj[ds][a];
      var g0 = pts[0].tstr_auc, gT = pts[pts.length - 1].tstr_auc;
      var delta = gT - g0;
      var color = delta < 0 ? 'var(--danger)' : 'var(--ok)';
      return '<div class="stat-row">' +
        '<div class="stat"><div class="v">' + C.fmt(g0, 3) + '</div><div class="k">generation 0</div></div>' +
        '<div class="stat"><div class="v">' + C.fmt(gT, 3) + '</div><div class="k">generation ' + D.meta.terminal_generation + '</div></div>' +
        '<div class="stat"><div class="v" style="color:' + color + '">' + (delta >= 0 ? '+' : '') + C.fmt(delta, 3) + '</div><div class="k">change</div></div>' +
        '</div>';
    }

    function html() {
      var ds = state.ds, a = state.alpha;
      var as = alphas(ds);
      var segs = datasets.map(function (d) {
        return '<button data-ds="' + d + '" aria-pressed="' + (d === ds) + '">' + d + '</button>';
      }).join('');
      return '' +
        '<div class="card">' +
        '<h3>Collapse happens; anchoring stops it</h3>' +
        '<p class="sub">Test-set AUC of a model trained on the generation-g synthetic data (TSTR), tracked across ' + D.meta.terminal_generation + ' generations of self-consuming regeneration, at each real-data anchoring fraction &alpha;.</p>' +
        '<div class="controls">' +
        '<div class="ctl"><label>dataset</label><div class="seg" data-role="seg">' + segs + '</div></div>' +
        '<div class="ctl"><label>highlight &alpha;</label><input type="range" data-role="alpha" min="0" max="' + (as.length - 1) + '" step="1" value="' + as.indexOf(a) + '"><div class="val">' + a + '</div></div>' +
        '</div>' +
        '<div data-role="chart1">' + chart1(ds, a) + '</div>' +
        '<div class="colorbar"><span>&alpha; = 0</span><span class="ramp"></span><span>&alpha; = 1</span></div>' +
        statRow(ds, a) +
        '</div>' +
        '<div class="card">' +
        '<h3>Damage is not confined to utility</h3>' +
        '<div class="grid-2">' +
        '<div><div class="sub">Correlation-matrix error &mdash; lower is better</div><div data-role="chart2">' + smallChart(ds, a, 'corr_frob', 'corr Frob. err.', 'chart2') + '</div></div>' +
        '<div><div class="sub">Categorical support retained &mdash; higher is better</div><div data-role="chart3">' + smallChart(ds, a, 'cat_support', 'support retained', 'chart3') + '</div></div>' +
        '</div>' +
        '<p class="sub">With no real data (&alpha; = 0) the model degrades away from the train-on-real ceiling every generation; at &alpha; = 1 the curve stays flat at the ceiling. What is plotted here is self-consumption of a model\'s own synthetic output, not ordinary evaluation-set drift.</p>' +
        '</div>';
    }

    function attachHovers() {
      var AH = root.AT.hover;
      if (!AH) return;
      var ds = state.ds, a = state.alpha;
      var hiColor = C.viridis(+a);

      var c1 = hoverInfo.chart1;
      if (c1) {
        var ceilingVal = D.ceiling[ds];
        AH.attach(el.querySelector('[data-role="chart1"]'), {
          x: c1.x, y: c1.y, xs: c1.xs, pad: c1.pad, width: c1.width, height: c1.height,
          series: [
            { label: 'α = ' + a, color: hiColor, values: D.traj[ds][a].map(function (p) { return p.tstr_auc; }) },
            { label: 'ceiling', color: 'var(--ink-3)', values: c1.xs.map(function () { return ceilingVal; }) },
          ],
          formatX: function (v) { return 'generation ' + v; },
          formatY: function (v) { return C.fmt(v, 3); },
          ariaLabel: 'Downstream utility by generation',
          title: function (i) { return 'generation ' + c1.xs[i]; },
        });
      }

      var c2 = hoverInfo.chart2;
      if (c2) {
        AH.attach(el.querySelector('[data-role="chart2"]'), {
          x: c2.x, y: c2.y, xs: c2.xs, pad: c2.pad, width: c2.width, height: c2.height,
          series: [{ label: 'correlation error', color: hiColor, values: D.traj[ds][a].map(function (p) { return p.corr_frob; }) }],
          formatX: function (v) { return 'generation ' + v; },
          formatY: function (v) { return C.fmt(v, 3); },
          ariaLabel: 'Correlation-matrix error by generation',
          title: function (i) { return 'generation ' + c2.xs[i]; },
        });
      }

      var c3 = hoverInfo.chart3;
      if (c3) {
        AH.attach(el.querySelector('[data-role="chart3"]'), {
          x: c3.x, y: c3.y, xs: c3.xs, pad: c3.pad, width: c3.width, height: c3.height,
          series: [{ label: 'support retained', color: hiColor, values: D.traj[ds][a].map(function (p) { return p.cat_support; }) }],
          formatX: function (v) { return 'generation ' + v; },
          formatY: function (v) { return C.fmt(v, 3); },
          ariaLabel: 'Categorical support retained by generation',
          title: function (i) { return 'generation ' + c3.xs[i]; },
        });
      }
    }

    function paint() {
      el.innerHTML = html();
      wire();
      attachHovers();
    }

    var paintThrottled = root.AT.hover ? root.AT.hover.throttle(paint) : paint;

    function wire() {
      var segRoot = el.querySelector('[data-role="seg"]');
      if (segRoot) {
        segRoot.querySelectorAll('button').forEach(function (btn) {
          btn.addEventListener('click', function () {
            state.ds = btn.getAttribute('data-ds');
            var as = alphas(state.ds);
            if (as.indexOf(state.alpha) < 0) state.alpha = as.slice(-1)[0];
            paintThrottled();
          });
        });
      }
      var slider = el.querySelector('[data-role="alpha"]');
      if (slider) {
        slider.addEventListener('input', function () {
          var as = alphas(state.ds);
          state.alpha = as[+slider.value];
          paintThrottled();
        });
      }
    }

    paint();
  }

  root.AT = root.AT || {};
  root.AT.panels = root.AT.panels || {};
  root.AT.panels.collapse = { render: render };
}(window));
