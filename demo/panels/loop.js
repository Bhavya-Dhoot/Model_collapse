/* global window */
(function (root) {
'use strict';

function nodeRect(x, y, w, h, label, sub) {
  var cx = x + w / 2;
  var out = '<rect x="' + x + '" y="' + y + '" width="' + w + '" height="' + h +
    '" rx="10" fill="var(--paper-2)" stroke="var(--rule)" stroke-width="1"/>';
  out += '<text x="' + cx + '" y="' + (y + (sub ? h / 2 - 4 : h / 2 + 4)) +
    '" text-anchor="middle" fill="var(--ink)" font-weight="650" font-size="13">' + label + '</text>';
  if (sub) {
    out += '<text x="' + cx + '" y="' + (y + h / 2 + 12) +
      '" text-anchor="middle" fill="var(--ink-3)" font-size="10.5">' + sub + '</text>';
  }
  return out;
}

function render(el, D) {
  var C = root.AT.chart;
  var alphas = D.meta.alphas;
  var ns = D.meta.ns;
  var state = { ai: alphas.indexOf(0.05) >= 0 ? alphas.indexOf(0.05) : 0, n: ns.indexOf(2000) >= 0 ? 2000 : ns[0] };

  function counts() {
    var alpha = alphas[state.ai];
    var n = state.n;
    var real = Math.ceil(alpha * n);
    var synth = n - real;
    return { alpha: alpha, n: n, real: real, synth: synth };
  }

  function svg(c) {
    var alpha = c.alpha;
    var realW = 1 + alpha * 6;
    var synthW = 1 + (1 - alpha) * 6;
    var realColor = alpha === 1 ? 'var(--ok)' : 'var(--ink-2)';
    var synthColor = alpha === 0 ? 'var(--danger)' : 'var(--ink-2)';
    var barX = 285, barY = 172, barW = 130, barH = 14;
    var realBarW = Math.max(0, barW * alpha);

    var s = '<svg viewBox="0 0 640 330" class="chart" preserveAspectRatio="xMidYMid meet" role="img">';
    // markerUnits="userSpaceOnUse" keeps the head a fixed size: the arrow
    // stroke-widths encode alpha, and a scaling head would swamp the diagram
    s += '<defs><marker id="loop-arrow" viewBox="0 0 10 10" refX="8" refY="5" ' +
      'markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto-start-reverse">' +
      '<path d="M0,0 L10,5 L0,10 Z" fill="context-stroke"/></marker></defs>';

    // real inflow: REAL POOL -> TRAINING SET
    s += '<path d="M150,55 L280,150" fill="none" stroke="' + realColor + '" stroke-width="' + realW.toFixed(2) +
      '" marker-end="url(#loop-arrow)"/>';
    // fit: TRAINING SET -> MODEL
    s += '<path d="M420,170 L470,170" fill="none" stroke="var(--ink-3)" stroke-width="2" marker-end="url(#loop-arrow)"/>';
    // synthetic feedback: MODEL -> TRAINING SET, curved over the top
    var fbPath = 'M540,140 C540,40 340,40 340,140';
    s += '<path id="loop-fb" d="' + fbPath + '" fill="none" stroke="' + synthColor + '" stroke-width="' + synthW.toFixed(2) +
      '" marker-end="url(#loop-arrow)"/>';
    // scored: MODEL -> HELD-OUT
    s += '<path d="M540,200 L540,250" fill="none" stroke="var(--ink-3)" stroke-width="2" marker-end="url(#loop-arrow)"/>';

    // traveling dot along the feedback path
    s += '<circle r="3.5" fill="' + synthColor + '"><animateMotion dur="3s" repeatCount="indefinite" path="' + fbPath + '"/></circle>';

    s += nodeRect(20, 20, 140, 60, 'REAL POOL');
    s += nodeRect(280, 140, 140, 60, 'TRAINING SET', C.fmt(c.n, 0) + ' rows');
    s += nodeRect(470, 140, 140, 60, 'MODEL Gₜ');
    s += nodeRect(470, 250, 140, 55, 'HELD-OUT REAL DATA', 'never trained on');

    // label the arrows
    s += '<text x="150" y="98" fill="var(--ink-3)" font-size="10.5" font-family="var(--mono)">α·n real</text>';
    s += '<text x="380" y="35" fill="var(--ink-3)" font-size="10.5" font-family="var(--mono)">(1−α)·n synthetic</text>';
    s += '<text x="428" y="163" fill="var(--ink-3)" font-size="10">fit</text>';
    s += '<text x="548" y="228" fill="var(--ink-3)" font-size="10">scored</text>';

    // stacked split bar inside training set node
    s += '<rect x="' + barX + '" y="' + barY + '" width="' + barW + '" height="' + barH +
      '" rx="3" fill="var(--ink-3)" fill-opacity="0.35"/>';
    if (realBarW > 0) {
      s += '<rect x="' + barX + '" y="' + barY + '" width="' + realBarW.toFixed(2) + '" height="' + barH +
        '" rx="3" fill="var(--accent)"/>';
    }
    s += '<text x="' + barX + '" y="' + (barY + barH + 12) + '" fill="var(--ink-3)" font-size="9.5" font-family="var(--mono)">' +
      C.fmt(c.real, 0) + ' real</text>';
    s += '<text x="' + (barX + barW) + '" y="' + (barY + barH + 12) + '" text-anchor="end" fill="var(--ink-3)" font-size="9.5" font-family="var(--mono)">' +
      C.fmt(c.synth, 0) + ' synthetic</text>';

    s += '</svg>';
    return s;
  }

  function pill(c) {
    if (c.alpha === 0) return '<span class="pill bad">pure self-consumption</span>';
    if (c.alpha === 1) return '<span class="pill ok">ordinary retraining on real data</span>';
    return '<span class="pill">anchored</span>';
  }

  function html() {
    var c = counts();
    var left = '<div class="card">' +
      '<h3>Anchor fraction α</h3>' +
      '<p class="sub">Raising α buys protection by displacing synthetic rows, never by enlarging the training set.</p>' +
      '<div class="controls">' +
      '<div class="ctl"><label>Budget n</label><div class="seg" data-role="ns">' +
      ns.map(function (v) { return '<button data-n="' + v + '" aria-pressed="' + (v === c.n) + '">' + C.fmt(v, 0) + '</button>'; }).join('') +
      '</div></div>' +
      '<div class="ctl"><label>α = ' + C.fmt(c.alpha, 2) + '</label>' +
      '<input type="range" data-role="alpha" min="0" max="' + (alphas.length - 1) + '" step="1" value="' + state.ai + '"/></div>' +
      '</div>' +
      '<p class="sub">α = ' + C.fmt(c.alpha, 2) + ' → ' + C.fmt(c.real, 0) + ' real + ' + C.fmt(c.synth, 0) +
      ' synthetic rows of a ' + C.fmt(c.n, 0) + '-row budget</p>' +
      pill(c) +
      '<div class="stat-row">' +
      '<div class="stat"><div class="v">' + C.fmt(c.n, 0) + '</div><div class="k">budget n</div></div>' +
      '<div class="stat"><div class="v">' + C.fmt(c.real, 0) + '</div><div class="k">real rows (⌈α·n⌉)</div></div>' +
      '<div class="stat"><div class="v">' + C.fmt(c.synth, 0) + '</div><div class="k">synthetic rows (n−⌈α·n⌉)</div></div>' +
      '</div>' +
      '</div>';
    var right = '<div class="card">' + svg(c) + '</div>';
    return '<div class="split">' + left + right + '</div>';
  }

  function wire() {
    var slider = el.querySelector('[data-role=alpha]');
    slider.addEventListener('input', function () {
      state.ai = +slider.value;
      draw();
    });
    var segBtns = el.querySelectorAll('[data-role=ns] button');
    for (var i = 0; i < segBtns.length; i++) {
      segBtns[i].addEventListener('click', function (ev) {
        state.n = +ev.currentTarget.getAttribute('data-n');
        draw();
      });
    }
  }

  function draw() {
    el.innerHTML = html();
    wire();
  }

  draw();
}

root.AT = root.AT || {};
root.AT.panels = root.AT.panels || {};
root.AT.panels.loop = { render: render };
}(window));
