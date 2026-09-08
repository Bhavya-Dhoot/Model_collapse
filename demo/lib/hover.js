/* Shared hover readout for the SVG charts: crosshair, per-series focus dots and
   a value tooltip.
 *
 * Why this exists: a static chart makes a professor squint at a line to guess a
 * value. The readout is the answer to "what is it exactly, at generation 7" --
 * so the motion here is motivated by a question the audience actually asks.
 *
 * Pointer and keyboard both drive it (the overlay is focusable, arrows step),
 * and it is inert under prefers-reduced-motion only in the sense that nothing
 * eases -- the readout itself still works, because it carries information.
 */
(function (root) {
  'use strict';

  var REDUCED = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function nearestIndex(xs, v) {
    var lo = 0, hi = xs.length - 1;
    if (v <= xs[lo]) return lo;
    if (v >= xs[hi]) return hi;
    while (hi - lo > 1) {
      var mid = (lo + hi) >> 1;
      if (xs[mid] < v) lo = mid; else hi = mid;
    }
    return (v - xs[lo] <= xs[hi] - v) ? lo : hi;
  }

  /**
   * @param {Element} container element wrapping exactly one <svg> chart
   * @param {Object} spec
   *   spec.x, spec.y        scales (from AT.chart)
   *   spec.xs               sorted array of x domain values
   *   spec.series           [{label, color, values:[y|null], dashed}]
   *   spec.pad, width, height
   *   spec.formatX(v), formatY(v)
   *   spec.title(i)         optional heading for the tooltip
   */
  function attach(container, spec) {
    if (!container) return;
    var svg = container.querySelector('svg');
    if (!svg || !spec || !spec.xs || !spec.xs.length) return;

    // idempotent: attaching twice to a surviving svg must not stack layers
    var old = container.querySelectorAll('.hv-layer, .chart-tip, svg rect[role="application"]');
    Array.prototype.forEach.call(old, function (n) { n.remove(); });

    container.classList.add('chart-wrap');

    var NS = 'http://www.w3.org/2000/svg';
    var pad = spec.pad, W = spec.width, H = spec.height;
    var plotL = pad.l, plotR = W - pad.r, plotT = pad.t, plotB = H - pad.b;

    var g = document.createElementNS(NS, 'g');
    g.setAttribute('class', 'hv-layer');
    g.setAttribute('opacity', '0');

    var cross = document.createElementNS(NS, 'line');
    cross.setAttribute('class', 'hv-cross');
    cross.setAttribute('y1', plotT);
    cross.setAttribute('y2', plotB);
    g.appendChild(cross);

    var dots = spec.series.map(function (s) {
      var c = document.createElementNS(NS, 'circle');
      c.setAttribute('r', '4.2');
      c.setAttribute('fill', s.color);
      c.setAttribute('stroke', 'var(--paper)');
      c.setAttribute('stroke-width', '1.6');
      g.appendChild(c);
      return c;
    });
    svg.appendChild(g);

    var tip = document.createElement('div');
    tip.className = 'chart-tip';
    tip.hidden = true;
    container.appendChild(tip);

    // transparent capture surface, focusable so the readout is keyboard-driven too
    var hit = document.createElementNS(NS, 'rect');
    hit.setAttribute('x', plotL);
    hit.setAttribute('y', plotT);
    hit.setAttribute('width', Math.max(0, plotR - plotL));
    hit.setAttribute('height', Math.max(0, plotB - plotT));
    hit.setAttribute('fill', 'transparent');
    hit.setAttribute('tabindex', '0');
    hit.setAttribute('role', 'application');
    hit.setAttribute('aria-label', (spec.ariaLabel || 'Chart') + '. Use arrow keys to read values.');
    hit.style.cursor = 'crosshair';
    hit.style.outline = 'none';
    svg.appendChild(hit);

    var idx = -1;

    function show(i) {
      if (i < 0 || i >= spec.xs.length) return;
      idx = i;
      var px = spec.x(spec.xs[i]);
      cross.setAttribute('x1', px);
      cross.setAttribute('x2', px);

      var rows = '';
      spec.series.forEach(function (s, k) {
        var v = s.values[i];
        var ok = v !== null && v !== undefined && isFinite(v);
        dots[k].setAttribute('opacity', ok ? '1' : '0');
        if (ok) {
          dots[k].setAttribute('cx', px);
          dots[k].setAttribute('cy', spec.y(v));
        }
        rows += '<div class="tip-row"><span class="tip-key">' +
          '<i style="background:' + s.color + '"></i>' + s.label + '</span>' +
          '<span class="tip-val">' + (ok ? spec.formatY(v) : '—') + '</span></div>';
      });

      tip.innerHTML = '<div class="tip-head">' +
        (spec.title ? spec.title(i) : spec.formatX(spec.xs[i])) + '</div>' + rows;
      tip.hidden = false;
      g.setAttribute('opacity', '1');

      // keep the tooltip inside the container
      var cw = container.clientWidth || W;
      var relX = (px / W) * cw;
      tip.style.left = '0px';
      var tw = tip.offsetWidth;
      var left = relX + 14;
      if (left + tw > cw - 4) left = relX - tw - 14;
      tip.style.left = Math.max(4, left) + 'px';
      tip.style.top = Math.max(4, (plotT / H) * (container.clientHeight || H) + 4) + 'px';
    }

    function hide() {
      idx = -1;
      g.setAttribute('opacity', '0');
      tip.hidden = true;
    }

    function fromEvent(e) {
      var r = svg.getBoundingClientRect();
      if (!r.width) return;
      var vx = ((e.clientX - r.left) / r.width) * W;   // client px -> viewBox units
      show(nearestIndex(spec.xs, spec.x.invert(vx)));
    }

    var onMove = throttle(fromEvent);
    hit.addEventListener('pointermove', function (e) { onMove(e); });
    hit.addEventListener('pointerleave', hide);
    hit.addEventListener('pointerdown', fromEvent);
    hit.addEventListener('focus', function () { show(idx < 0 ? 0 : idx); });
    hit.addEventListener('blur', hide);
    hit.addEventListener('keydown', function (e) {
      if (e.key === 'ArrowRight') { e.preventDefault(); show(Math.min(idx + 1, spec.xs.length - 1)); }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); show(Math.max(idx - 1, 0)); }
      else if (e.key === 'Home') { e.preventDefault(); show(0); }
      else if (e.key === 'End') { e.preventDefault(); show(spec.xs.length - 1); }
      else if (e.key === 'Escape') { hide(); hit.blur(); }
    });

    if (REDUCED) g.setAttribute('data-reduced', '1');
  }

  /** Coalesce a handler to one call per animation frame, so dragging a slider
   *  repaints once per frame instead of once per input event.
   *  A timer backs the frame up: browsers stop serving frames to a hidden or
   *  backgrounded tab, and without it a queued repaint would never land. */
  function throttle(fn) {
    var pending = false, lastArgs = null, timer = null;
    function run() {
      if (!pending) return;
      pending = false;
      clearTimeout(timer);
      timer = null;
      fn.apply(null, lastArgs);
    }
    return function () {
      lastArgs = arguments;
      if (pending) return;
      pending = true;
      requestAnimationFrame(run);
      timer = setTimeout(run, 120);
    };
  }

  root.AT = root.AT || {};
  root.AT.hover = { attach: attach, throttle: throttle };
}(window));
