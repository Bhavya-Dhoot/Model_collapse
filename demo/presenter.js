/* Presentation mode.
 *
 * One scroll authority: a single requestAnimationFrame loop owns the page's
 * scroll position while presenting, and the CSS `scroll-behavior: smooth`
 * is switched off for the duration so the two never fight.
 *
 * Every motion here is motivated: the tour scrolls to direct the eye through
 * the argument in order, and each section "performs" by driving its own
 * controls, so the finding demonstrates itself without anyone touching the
 * laptop. Nothing decorative animates.
 *
 * prefers-reduced-motion is a designed branch, not a kill switch: the tour
 * still advances and still performs, but it jumps between sections and steps
 * controls discretely instead of easing them.
 */
(function (root) {
  'use strict';

  var REDUCED = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ---------------- the running order ---------------- */
  /* dwell = ms spent on the section after the scroll settles and after any
     performance finishes. Sections carrying more numbers get longer. */
  var SCRIPT = [
    { id: 'top', label: 'The question', dwell: 5200 },
    { id: 'mechanism', label: 'Mechanism', dwell: 1200, perform: performMechanism },
    { id: 'theory', label: 'Theory, live', dwell: 1500, perform: performTheory },
    { id: 'collapse', label: 'Collapse', dwell: 1200, perform: performCollapse },
    { id: 'variance', label: 'Negative result', dwell: 8000 },
    { id: 'axes', label: 'Thresholds', dwell: 1200, perform: performAxes },
    { id: 'scaling', label: 'Scaling', dwell: 1400, perform: performScaling },
    { id: 'regime', label: 'Fresh vs fixed', dwell: 8000 },
    { id: 'arch', label: 'Architectures', dwell: 8000 },
    { id: 'reproduce', label: 'Provenance', dwell: 6000 }
  ];

  var st = { on: false, playing: false, i: 0, token: 0, raf: null };

  /* ---------------- cancellable primitives ---------------- */

  function tokenAlive(t) { return t === st.token && st.playing; }

  function sleep(ms, t) {
    return new Promise(function (res) {
      var id = setTimeout(function () { res(tokenAlive(t)); }, ms);
      sleep._ids.push(id);
    });
  }
  sleep._ids = [];
  function clearSleeps() { sleep._ids.forEach(clearTimeout); sleep._ids = []; }

  function easeInOutCubic(p) {
    return p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2;
  }

  /** The single scroll authority. Resolves true if it completed uninterrupted. */
  function scrollToY(targetY, duration, t) {
    if (st.raf) { cancelAnimationFrame(st.raf); st.raf = null; }
    var maxY = document.documentElement.scrollHeight - window.innerHeight;
    targetY = Math.max(0, Math.min(targetY, maxY));
    if (REDUCED || duration <= 0) { window.scrollTo(0, targetY); return Promise.resolve(true); }

    var startY = window.scrollY;
    var delta = targetY - startY;
    if (Math.abs(delta) < 2) return Promise.resolve(true);
    var t0 = null;
    return new Promise(function (res) {
      var settled = false, guard;
      function finish(ok, snap) {
        if (settled) return;
        settled = true;
        clearTimeout(guard);
        if (st.raf) { cancelAnimationFrame(st.raf); st.raf = null; }
        if (snap) window.scrollTo(0, targetY);
        res(ok);
      }
      // Watchdog: browsers stop firing rAF in a backgrounded tab or on a
      // sleeping display. Without this the tour would wedge mid-talk, so if
      // frames stop arriving we land the scroll instantly and carry on.
      guard = setTimeout(function () { finish(tokenAlive(t), true); }, duration + 600);
      sleep._ids.push(guard);
      function step(now) {
        if (!tokenAlive(t)) { finish(false, false); return; }
        if (t0 === null) t0 = now;
        var p = Math.min((now - t0) / duration, 1);
        window.scrollTo(0, startY + delta * easeInOutCubic(p));
        if (p < 1) { st.raf = requestAnimationFrame(step); } else { finish(true, false); }
      }
      st.raf = requestAnimationFrame(step);
    });
  }

  /** Drive a range input from its current value to `to`, firing input events. */
  function sweepRange(input, to, duration, t) {
    if (!input) return Promise.resolve(true);
    var from = +input.value;
    if (from === to) return Promise.resolve(true);
    if (REDUCED) { input.value = to; input.dispatchEvent(new Event('input', { bubbles: true })); return Promise.resolve(true); }
    var t0 = null;
    return new Promise(function (res) {
      var settled = false, guard;
      function finish(ok, snap) {
        if (settled) return;
        settled = true;
        clearTimeout(guard);
        if (snap && +input.value !== to) {
          input.value = to;
          input.dispatchEvent(new Event('input', { bubbles: true }));
        }
        res(ok);
      }
      guard = setTimeout(function () { finish(tokenAlive(t), true); }, duration + 600);
      sleep._ids.push(guard);
      function step(now) {
        if (!tokenAlive(t)) { finish(false, false); return; }
        if (t0 === null) t0 = now;
        var p = Math.min((now - t0) / duration, 1);
        var v = Math.round(from + (to - from) * easeInOutCubic(p));
        if (+input.value !== v) {
          input.value = v;
          input.dispatchEvent(new Event('input', { bubbles: true }));
        }
        if (p < 1) requestAnimationFrame(step); else finish(true, false);
      }
      requestAnimationFrame(step);
    });
  }

  function clickSeq(nodes, gap, t) {
    var i = 0;
    return (function next() {
      if (i >= nodes.length || !tokenAlive(t)) return Promise.resolve(tokenAlive(t));
      nodes[i++].click();
      return sleep(gap, t).then(function (ok) { return ok ? next() : false; });
    }());
  }

  var q = function (sel) { return document.querySelector(sel); };
  var qa = function (sel) { return Array.prototype.slice.call(document.querySelectorAll(sel)); };

  /* ---------------- per-section performances ---------------- */

  function performMechanism(t) {
    var s = q('#panel-loop input[type=range]');
    if (!s) return Promise.resolve(true);
    // sweep the whole composition range, then rest at a realistic anchor
    return sweepRange(s, +s.min, 900, t)
      .then(function (ok) { return ok ? sweepRange(s, +s.max, 3400, t) : false; })
      .then(function (ok) { return ok ? sleep(900, t) : false; })
      .then(function (ok) { return ok ? sweepRange(s, 3, 1500, t) : false; });
  }

  function performTheory(t) {
    var s = q('#panel-theory input[type=range]');
    if (!s) return Promise.resolve(true);
    // a small anchor converges slowly; a larger one snaps to the fixed point
    return sweepRange(s, 3, 700, t)
      .then(function (ok) { return ok ? sleep(4200, t) : false; })
      .then(function (ok) { return ok ? sweepRange(s, 6, 900, t) : false; })
      .then(function (ok) { return ok ? sleep(4200, t) : false; });
  }

  function performCollapse(t) {
    var segs = qa('#panel-collapse .seg button');
    var s = q('#panel-collapse input[type=range]');
    return sweepRange(s, +s.min, 800, t)
      .then(function (ok) { return ok ? sleep(2600, t) : false; })
      .then(function (ok) { return ok ? sweepRange(s, +s.max, 2600, t) : false; })
      .then(function (ok) { return ok ? sleep(1200, t) : false; })
      .then(function (ok) { return ok && segs.length ? clickSeq(segs.slice(1), 2600, t) : ok; })
      .then(function (ok) { return ok && segs.length ? clickSeq([segs[0]], 1400, t) : ok; });
  }

  function performAxes(t) {
    var segs = qa('#panel-axes .seg button');
    return segs.length ? clickSeq(segs, 1500, t) : Promise.resolve(true);
  }

  function performScaling(t) {
    var s = q('#panel-scaling input[type=range]');
    var segs = qa('#panel-scaling .seg button');
    return sweepRange(s, +s.min, 1600, t)
      .then(function (ok) { return ok ? sleep(1500, t) : false; })
      .then(function (ok) { return ok && segs.length ? clickSeq(segs, 900, t) : ok; });
  }

  /* ---------------- the tour ---------------- */

  function targetFor(id) {
    if (id === 'top') return 0;
    var e = document.getElementById(id);
    if (!e) return 0;
    var navH = 58;
    return e.getBoundingClientRect().top + window.scrollY - navH - 14;
  }

  function runFrom(i) {
    st.i = Math.max(0, Math.min(i, SCRIPT.length - 1));
    var t = ++st.token;
    st.playing = true;
    paint();

    (function stepSection() {
      if (!tokenAlive(t)) return;
      var s = SCRIPT[st.i];
      paint();
      scrollToY(targetFor(s.id), st.i === 0 ? 700 : 1100, t)
        .then(function (ok) { return ok ? sleep(650, t) : false; })
        .then(function (ok) { return ok && s.perform ? s.perform(t) : ok; })
        .then(function (ok) { return ok ? sleep(s.dwell, t) : false; })
        .then(function (ok) {
          if (!ok || !tokenAlive(t)) return;
          if (st.i >= SCRIPT.length - 1) { st.i = 0; } else { st.i++; }
          stepSection();
        });
    }());
  }

  function play() { if (!st.on) enter(); runFrom(st.i); }

  function pause() {
    st.playing = false;
    st.token++;
    clearSleeps();
    if (st.raf) { cancelAnimationFrame(st.raf); st.raf = null; }
    paint();
  }

  function go(delta) {
    var i = st.i + delta;
    if (i < 0) i = SCRIPT.length - 1;
    if (i > SCRIPT.length - 1) i = 0;
    if (st.playing) runFrom(i);
    else { st.i = i; st.token++; scrollToY(targetFor(SCRIPT[i].id), 600, st.token); paint(); }
  }

  function enter() {
    st.on = true;
    document.documentElement.classList.add('presenting');
    // take sole ownership of scrolling
    document.documentElement.style.scrollBehavior = 'auto';
    q('#presenter-bar').hidden = false;
    paint();
  }

  function exit() {
    pause();
    st.on = false;
    document.documentElement.classList.remove('presenting');
    document.documentElement.style.scrollBehavior = '';
    q('#presenter-bar').hidden = true;
  }

  function paint() {
    var bar = q('#presenter-bar');
    if (!bar) return;
    q('#pr-label').textContent = SCRIPT[st.i].label;
    q('#pr-count').textContent = (st.i + 1) + ' / ' + SCRIPT.length;
    q('#pr-play').innerHTML = st.playing ? ICON_PAUSE : ICON_PLAY;
    q('#pr-play').setAttribute('aria-label', st.playing ? 'Pause' : 'Play');
    qa('#pr-dots .dot').forEach(function (d, i) {
      d.classList.toggle('on', i === st.i);
      d.classList.toggle('done', i < st.i);
    });
    q('#pr-progress').style.transform = 'scaleX(' + ((st.i + 1) / SCRIPT.length) + ')';
  }

  var ICON_PLAY = '<svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5.5v13l11-6.5z"/></svg>';
  var ICON_PAUSE = '<svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><rect x="7" y="5.5" width="3.4" height="13" rx="1"/><rect x="13.6" y="5.5" width="3.4" height="13" rx="1"/></svg>';

  function build() {
    var dots = SCRIPT.map(function (s, i) {
      return '<button class="dot" data-i="' + i + '" title="' + s.label + '" aria-label="Go to ' + s.label + '"></button>';
    }).join('');

    document.body.insertAdjacentHTML('beforeend',
      '<div id="presenter-bar" hidden role="group" aria-label="Presentation controls">' +
        '<div id="pr-progress-track"><div id="pr-progress"></div></div>' +
        '<div class="pr-inner">' +
          '<button class="pr-btn" id="pr-prev" aria-label="Previous section">' +
            '<svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><path d="M15 5.5v13L6 12z"/></svg></button>' +
          '<button class="pr-btn primary" id="pr-play" aria-label="Play"></button>' +
          '<button class="pr-btn" id="pr-next" aria-label="Next section">' +
            '<svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><path d="M9 5.5v13l9-6.5z"/></svg></button>' +
          '<div class="pr-meta"><span id="pr-label"></span><span id="pr-count"></span></div>' +
          '<div id="pr-dots">' + dots + '</div>' +
          '<button class="pr-btn" id="pr-exit" aria-label="Exit presentation">' +
            '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><path d="M6 6l12 12M18 6L6 18"/></svg></button>' +
        '</div>' +
      '</div>');

    q('#pr-play').addEventListener('click', function () { st.playing ? pause() : play(); });
    q('#pr-prev').addEventListener('click', function () { go(-1); });
    q('#pr-next').addEventListener('click', function () { go(1); });
    q('#pr-exit').addEventListener('click', exit);
    q('#pr-dots').addEventListener('click', function (e) {
      var d = e.target.closest('.dot');
      if (!d) return;
      var i = +d.dataset.i;
      if (st.playing) runFrom(i); else { st.i = i; st.token++; scrollToY(targetFor(SCRIPT[i].id), 600, st.token); paint(); }
    });

    var startBtn = q('#start-presentation');
    if (startBtn) startBtn.addEventListener('click', function () { st.i = 0; play(); });

    // A deliberate scroll by the presenter takes control back.
    ['wheel', 'touchstart'].forEach(function (ev) {
      window.addEventListener(ev, function () { if (st.playing) pause(); }, { passive: true });
    });

    document.addEventListener('keydown', function (e) {
      if (/^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) return;
      if (e.key === 'p' || e.key === 'P') { e.preventDefault(); st.playing ? pause() : play(); return; }
      if (!st.on) return;
      if (e.key === ' ') { e.preventDefault(); st.playing ? pause() : play(); }
      else if (e.key === 'ArrowRight' || e.key === 'PageDown') { e.preventDefault(); go(1); }
      else if (e.key === 'ArrowLeft' || e.key === 'PageUp') { e.preventDefault(); go(-1); }
      else if (e.key === 'Escape') { e.preventDefault(); exit(); }
    });

    if (/(^|[?&#])present/.test(location.hash + location.search)) { st.i = 0; play(); }
  }

  root.AT = root.AT || {};
  root.AT.presenter = { init: build, play: play, pause: pause, exit: exit };
}(window));
