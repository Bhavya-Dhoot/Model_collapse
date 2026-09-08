/* Theory panel: runs a real Monte-Carlo of the anchored loop in the browser
   and overlays the closed form, so the two can be compared live. Nothing here
   is precomputed -- every point is simulated when the controls change. */
(function (root) {
  'use strict';

  var ALPHAS = [0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1];
  var NS = [200, 500, 1000, 2000, 5000];
  var W = 700, H = 330;

  function render(el, D) {
    var C = root.AT.chart, S = root.AT.sim;

    var st = { ai: 5, n: 2000, T: 50, reps: 150, shown: 0, timer: null };

    el.innerHTML =
      '<div class="card">' +
        '<div class="controls">' +
          '<div class="ctl"><label for="th-a">Anchor fraction α</label>' +
            '<input type="range" id="th-a" min="0" max="' + (ALPHAS.length - 1) + '" step="1" value="' + st.ai + '">' +
            '<span class="val" id="th-av"></span></div>' +
          '<div class="ctl"><label>Training budget n</label><div class="seg" id="th-n">' +
            NS.map(function (n) {
              return '<button type="button" data-n="' + n + '" aria-pressed="' + (n === st.n) + '">' + n + '</button>';
            }).join('') + '</div></div>' +
          '<div class="ctl"><label>&nbsp;</label><button class="btn primary" id="th-run">Re-run simulation</button></div>' +
        '</div>' +
        '<div id="th-chart"></div>' +
        '<div id="th-legend"></div>' +
      '</div>' +
      '<div class="grid-2" style="margin-top:16px">' +
        '<div class="card"><h3>Conservation of the budget</h3>' +
          '<p class="sub">Variance the model loses does not vanish: it reappears as drift of the fitted mean. ' +
          'The two always sum to the true variance.</p>' +
          '<div id="th-cons"></div><div id="th-cons-stats" class="stat-row"></div></div>' +
        '<div class="card"><h3>Closed form at these settings</h3>' +
          '<div id="th-eq"></div><div id="th-stats" class="stat-row"></div>' +
          '<div class="note" style="margin-top:14px" id="th-note"></div></div>' +
      '</div>';

    var $ = function (id) { return el.querySelector(id); };

    function alpha() { return ALPHAS[st.ai]; }

    function compute() {
      var a = alpha();
      st.theory = S.recursion(a, st.n, st.T);
      st.mc = S.monteCarlo({ alpha: a, n: st.n, T: st.T, reps: st.reps, seed: 7 });
      st.fp = S.fixedPoint(a, st.n);
      st.shown = 0;
    }

    function play() {
      if (st.timer) clearInterval(st.timer);
      st.timer = setInterval(function () {
        st.shown++;
        if (st.shown >= st.T) { clearInterval(st.timer); st.timer = null; st.shown = st.T; }
        draw();
      }, 42);
    }

    function draw() {
      var a = alpha(), fp = st.fp;
      var k = st.shown;

      // ---- main chart: E[Sigma_t] against generation --------------------
      var lo = Math.min(0, fp.sigma - 0.05);
      for (var i = 0; i <= k; i++) lo = Math.min(lo, st.mc[i].sigma, st.theory[i].sigma);
      var x = C.linear(0, st.T, 60, W - 130);
      var y = C.linear(Math.max(0, lo - 0.04), 1.06, H - 44, 18);
      var f = C.frame({
        width: W, height: H, pad: { t: 18, r: 130, b: 44, l: 60 }, x: x, y: y,
        xLabel: 'generation t', yLabel: 'E[Σ t]  (true variance = 1)',
        xFormat: function (v) { return String(v); },
        yFormat: function (v) { return v.toFixed(2); },
      });

      var theoryPts = st.theory.slice(0, k + 1).map(function (p) { return { x: p.t, y: p.sigma }; });
      var mcPts = st.mc.slice(0, k + 1).map(function (p) { return { x: p.t, y: p.sigma }; });

      var s = f.open + f.grid;
      // fixed-point asymptote
      if (isFinite(fp.sigma) && fp.sigma >= y.domain[0] && fp.sigma <= y.domain[1]) {
        s += C.el('line', { x1: 60, x2: W - 130, y1: y(fp.sigma), y2: y(fp.sigma), class: 'ref-line' });
        s += C.el('text', { x: W - 126, y: y(fp.sigma) + 4, class: 'annot-sm', fill: 'var(--ink-2)' },
          'Σ∞ = ' + fp.sigma.toFixed(4));
      }
      s += C.line(theoryPts, x, y, { stroke: 'var(--accent)', 'stroke-width': 2.4 });
      s += C.dots(mcPts, x, y, 2.9, { fill: 'none', stroke: 'var(--danger)', 'stroke-width': 1.5 });
      // running head marker
      if (k > 0) {
        s += C.el('circle', { cx: x(k), cy: y(st.mc[k].sigma), r: 4.5, fill: 'var(--danger)' });
      }
      s += C.el('text', { x: W - 126, y: 26, class: 'annot' }, 'α = ' + a);
      s += C.el('text', { x: W - 126, y: 44, class: 'annot-sm' }, 'n = ' + st.n);
      s += C.el('text', { x: W - 126, y: 62, class: 'annot-sm' }, 'generation ' + k + ' / ' + st.T);
      s += f.close;
      $('#th-chart').innerHTML = s;

      // hover readout: the whole point of this panel is comparing the simulated
      // value against the closed form, so let the audience read both exactly
      if (root.AT.hover) {
        var gens = st.theory.slice(0, k + 1).map(function (p) { return p.t; });
        root.AT.hover.attach($('#th-chart'), {
          x: x, y: y, xs: gens,
          series: [
            { label: 'closed form', color: 'var(--accent)', values: gens.map(function (_, i) { return st.theory[i].sigma; }) },
            { label: 'simulated', color: 'var(--danger)', values: gens.map(function (_, i) { return st.mc[i].sigma; }) }
          ],
          pad: f.pad, width: f.width, height: f.height,
          formatX: function (v) { return String(v); },
          formatY: function (v) { return v.toFixed(5); },
          ariaLabel: 'Retained variance by generation, simulation against closed form',
          title: function (i) { return 'generation ' + gens[i]; }
        });
      }

      $('#th-legend').innerHTML = C.legend([
        { label: 'closed form (Propositions 1–3)', color: 'var(--accent)' },
        { label: 'Monte-Carlo, ' + st.reps + ' independent chains', color: 'var(--danger)' },
        { label: 'fixed point Σ∞', color: 'var(--ink-3)', dashed: true },
      ]);

      // ---- conservation chart: Sigma_t and V_t stacked to 1 -------------
      var cw = 340, ch = 210;
      var x2 = C.linear(0, st.T, 46, cw - 12);
      var y2 = C.linear(0, 1.08, ch - 40, 14);
      var f2 = C.frame({
        width: cw, height: ch, pad: { t: 14, r: 12, b: 40, l: 46 }, x: x2, y: y2,
        xLabel: 'generation t', yFormat: function (v) { return v.toFixed(1); },
      });
      var sig = st.theory.slice(0, k + 1).map(function (p) { return { x: p.t, lo: 0, hi: p.sigma }; });
      var vv = st.theory.slice(0, k + 1).map(function (p) { return { x: p.t, lo: p.sigma, hi: p.sigma + p.v }; });
      var s2 = f2.open + f2.grid;
      s2 += C.band(sig, x2, y2, { fill: 'var(--accent)', opacity: 0.55 });
      s2 += C.band(vv, x2, y2, { fill: 'var(--danger)', opacity: 0.5 });
      s2 += C.el('line', { x1: 46, x2: cw - 12, y1: y2(1), y2: y2(1), class: 'ref-line' });
      s2 += C.el('text', { x: cw - 14, y: y2(1) - 6, class: 'annot-sm', 'text-anchor': 'end' }, 'Σ = 1');
      s2 += f2.close;
      $('#th-cons').innerHTML = s2 + C.legend([
        { label: 'Σ t  (variance retained)', color: 'var(--accent)' },
        { label: 'Var[μ t]  (mean drift)', color: 'var(--danger)' },
      ]);

      var sum = fp.sigma + fp.v;
      $('#th-cons-stats').innerHTML = stat('Σ∞', fp.sigma.toFixed(5)) +
        stat('Var[μ∞]', fp.v.toFixed(5)) +
        stat('sum', sum.toFixed(6), Math.abs(sum - 1) < 1e-9 ? 'var(--ok)' : 'var(--danger)');

      // ---- closed forms + agreement -------------------------------------
      $('#th-eq').innerHTML =
        '<div class="eq">D(α,n) = 1 − α + αn(2 − α) = ' + fp.D.toFixed(3) + '</div>' +
        '<div class="eq">Σ∞ = (1 − 1/D)·Σ = ' + fp.sigma.toFixed(5) + '</div>' +
        '<div class="eq">Var[μ∞] = Σ/D = ' + fp.v.toFixed(5) + '</div>' +
        '<div class="eq">convergence rate ρ = 1 − α = ' + fp.rate.toFixed(3) + '</div>';

      var mcEnd = st.mc[k].sigma, thEnd = st.theory[k].sigma;
      var relErr = thEnd !== 0 ? Math.abs(mcEnd - thEnd) / Math.abs(thEnd) * 100 : 0;
      $('#th-stats').innerHTML =
        stat('simulated', mcEnd.toFixed(4)) +
        stat('closed form', thEnd.toFixed(4)) +
        stat('disagreement', relErr.toFixed(2) + '%', relErr < 2 ? 'var(--ok)' : 'var(--warn)');

      var half = a > 0 ? Math.log(0.5) / Math.log(1 - a) : Infinity;
      $('#th-note').innerHTML = a === 0
        ? '<strong>α = 0.</strong> With no real data the recursion is E[Σ t] = c·E[Σ t−1] with c = (n−1)/n, so ' +
          'variance decays geometrically to zero. There is no fixed point to converge to.'
        : '<strong>Convergence is geometric at rate 1 − α.</strong> At α = ' + a + ' the distance to the fixed ' +
          'point halves every ' + half.toFixed(1) + ' generations, so a short run at small α can look like it ' +
          'falsifies the formula when it has simply not converged yet.';
    }

    function stat(k, v, color) {
      return '<div class="stat"><div class="v"' + (color ? ' style="color:' + color + '"' : '') + '>' +
        v + '</div><div class="k">' + k + '</div></div>';
    }

    function refresh() {
      $('#th-av').textContent = String(alpha());
      compute();
      draw();
      play();
    }

    // dragging the slider re-runs a Monte-Carlo, so coalesce to one per frame
    var onAlpha = root.AT.hover
      ? root.AT.hover.throttle(function () { refresh(); })
      : refresh;
    $('#th-a').addEventListener('input', function (e) {
      st.ai = +e.target.value;
      $('#th-a').setAttribute('aria-valuetext', 'alpha ' + alpha());
      onAlpha();
    });
    $('#th-n').addEventListener('click', function (e) {
      var b = e.target.closest('button[data-n]');
      if (!b) return;
      st.n = +b.dataset.n;
      Array.prototype.forEach.call(el.querySelectorAll('#th-n button'), function (x) {
        x.setAttribute('aria-pressed', String(+x.dataset.n === st.n));
      });
      refresh();
    });
    $('#th-run').addEventListener('click', refresh);
    window.addEventListener('at:theme', draw);

    refresh();
  }

  root.AT = root.AT || {};
  root.AT.panels = root.AT.panels || {};
  root.AT.panels.theory = { render: render };
}(window));
