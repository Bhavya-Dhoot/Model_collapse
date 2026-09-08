/* Boot: load the dataset (inlined by the build, fetched in dev), render every
   panel, wire the theme toggle and scroll-spy. */
(function () {
  'use strict';

  function boot(DATA) {
    window.AT.DATA = DATA;

    renderKpis(DATA);

    var panels = [
      ['panel-loop', 'loop'], ['panel-theory', 'theory'], ['panel-collapse', 'collapse'],
      ['panel-variance', 'variance'], ['panel-axes', 'axes'], ['panel-scaling', 'scaling'],
      ['panel-regime', 'regime'], ['panel-arch', 'arch'], ['panel-repro', 'repro'],
    ];
    panels.forEach(function (p) {
      var root = document.getElementById(p[0]);
      var mod = window.AT.panels && window.AT.panels[p[1]];
      if (!root) return;
      if (!mod || typeof mod.render !== 'function') {
        root.innerHTML = '<div class="note warn">Panel <span class="mono">' + p[1] + '</span> not loaded.</div>';
        return;
      }
      try {
        root.innerHTML = '';
        mod.render(root, DATA);
      } catch (err) {
        root.innerHTML = '<div class="note warn"><strong>Panel ' + p[1] + ' failed:</strong> ' +
          String(err && err.message || err) + '</div>';
        if (window.console) console.error('[panel ' + p[1] + ']', err);
      }
    });

    wireChrome();
  }

  function renderKpis(D) {
    var f = window.AT.chart.fmt;
    var adult = (D.alpha_star_table || []).filter(function (r) {
      return r.dataset === 'adult' && r.n === 2000;
    })[0] || {};
    var adultTraj = D.traj && D.traj.adult;
    var a0 = adultTraj && adultTraj['0'];
    var drop = a0 ? a0[0].tstr_auc + ' → ' + a0[a0.length - 1].tstr_auc : '—';
    if (a0) drop = f(a0[0].tstr_auc, 3) + ' → ' + f(a0[a0.length - 1].tstr_auc, 3);

    var items = [
      { v: drop, k: 'Adult utility (AUC) over ' + D.meta.terminal_generation +
          ' generations with no real data', cls: 'bad' },
      { v: f(adult.ratio_excl_var, 1) + '×', k: 'spread between the strictest and loosest per-axis threshold', cls: 'accent' },
      { v: 'n<sup>−' + D.nscaling.beta + '</sup>', k: 'measured scaling of α★ with dataset size, against a predicted n<sup>−1</sup>', cls: 'accent' },
      { v: 'flat', k: 'variance ratio under full self-consumption — the classic collapse signal never fires', cls: 'good' },
    ];
    document.getElementById('kpis').innerHTML = items.map(function (i) {
      return '<div class="kpi ' + i.cls + '"><div class="v">' + i.v + '</div><div class="k">' + i.k + '</div></div>';
    }).join('');
  }

  function wireChrome() {
    var btn = document.getElementById('theme-toggle');
    var root = document.documentElement;
    var saved = null;
    try { saved = localStorage.getItem('at-theme'); } catch (e) { /* private mode */ }
    if (saved) root.setAttribute('data-theme', saved);
    btn.addEventListener('click', function () {
      var next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem('at-theme', next); } catch (e) { /* ignore */ }
      window.dispatchEvent(new CustomEvent('at:theme', { detail: next }));
    });

    var links = Array.prototype.slice.call(document.querySelectorAll('.nav a.jump'));
    var secs = links.map(function (a) { return document.querySelector(a.getAttribute('href')); });
    if (!('IntersectionObserver' in window)) return;
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (!e.isIntersecting) return;
        var i = secs.indexOf(e.target);
        links.forEach(function (l, j) { l.classList.toggle('active', i === j); });
      });
    }, { rootMargin: '-60px 0px -70% 0px' });
    secs.forEach(function (s) { if (s) io.observe(s); });
  }

  function start() {
    if (window.AT && window.AT.DATA_INLINE) { boot(window.AT.DATA_INLINE); return; }
    fetch('data.json')
      .then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      })
      .then(boot)
      .catch(function (err) {
        document.querySelector('main').insertAdjacentHTML('afterbegin',
          '<div class="note warn" style="margin:24px 0"><strong>Could not load data.json.</strong> ' +
          String(err.message) + '. Open <span class="mono">dashboard.html</span> instead — it has the ' +
          'data embedded and needs no server.</div>');
      });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();
}());
