/* global window */
(function (root) {
'use strict';
/* Minimal SVG chart primitives. No external library: the dashboard has to
   render from a local file with no network. Everything returns SVG strings. */

const NS = 'http://www.w3.org/2000/svg';
const fmt = (x, d = 3) => (x === null || x === undefined || !isFinite(x) ? '—' : (+x).toFixed(d));

function el(tag, attrs = {}, children = '') {
  const a = Object.entries(attrs)
    .filter(([, v]) => v !== null && v !== undefined && v !== false)
    .map(([k, v]) => `${k}="${String(v).replace(/"/g, '&quot;')}"`)
    .join(' ');
  return `<${tag}${a ? ' ' + a : ''}>${children}</${tag}>`;
}

function linear(d0, d1, r0, r1) {
  const span = d1 - d0 || 1;
  const f = (v) => r0 + ((v - d0) / span) * (r1 - r0);
  f.invert = (p) => d0 + ((p - r0) / (r1 - r0)) * span;
  f.domain = [d0, d1];
  f.range = [r0, r1];
  return f;
}

function log(d0, d1, r0, r1) {
  const l0 = Math.log10(d0), l1 = Math.log10(d1);
  const f = (v) => r0 + ((Math.log10(v) - l0) / (l1 - l0 || 1)) * (r1 - r0);
  f.invert = (p) => 10 ** (l0 + ((p - r0) / (r1 - r0)) * (l1 - l0));
  f.domain = [d0, d1];
  f.range = [r0, r1];
  f.isLog = true;
  return f;
}

function ticks(d0, d1, count = 5) {
  const span = d1 - d0;
  if (span <= 0) return [d0];
  const raw = span / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  const step = (norm >= 7.5 ? 10 : norm >= 3.5 ? 5 : norm >= 1.5 ? 2 : 1) * mag;
  const out = [];
  for (let v = Math.ceil(d0 / step) * step; v <= d1 + step * 1e-9; v += step) {
    out.push(Math.abs(v) < step * 1e-9 ? 0 : +v.toPrecision(12));
  }
  return out;
}

/**
 * Frame: axes, gridlines, labels. Returns {svgOpen, svgClose, x, y, w, h, pad}
 * so panels can draw series between open and close.
 */
function frame(opts) {
  const {
    width = 640, height = 300, pad = { t: 16, r: 16, b: 40, l: 52 },
    x, y, xLabel = '', yLabel = '', xTicks, yTicks,
    xFormat = (v) => String(v), yFormat = (v) => String(v),
    xGrid = true, yGrid = true, className = '',
  } = opts;

  const xs = xTicks || (x.isLog ? logTicks(x) : ticks(x.domain[0], x.domain[1], 6));
  const ys = yTicks || (y.isLog ? logTicks(y) : ticks(y.domain[0], y.domain[1], 5));
  let g = '';

  for (const t of ys) {
    if (t < Math.min(...y.domain) || t > Math.max(...y.domain)) continue;
    const py = y(t);
    if (yGrid) g += el('line', { x1: pad.l, x2: width - pad.r, y1: py, y2: py, class: 'grid' });
    g += el('text', { x: pad.l - 8, y: py + 3.5, class: 'tick tick-y' }, yFormat(t));
  }
  for (const t of xs) {
    if (t < Math.min(...x.domain) || t > Math.max(...x.domain)) continue;
    const px = x(t);
    if (xGrid) g += el('line', { x1: px, x2: px, y1: pad.t, y2: height - pad.b, class: 'grid' });
    g += el('text', { x: px, y: height - pad.b + 16, class: 'tick tick-x' }, xFormat(t));
  }
  g += el('line', { x1: pad.l, x2: width - pad.r, y1: height - pad.b, y2: height - pad.b, class: 'axis' });
  g += el('line', { x1: pad.l, x2: pad.l, y1: pad.t, y2: height - pad.b, class: 'axis' });
  if (xLabel) g += el('text', { x: (pad.l + width - pad.r) / 2, y: height - 4, class: 'axis-label' }, xLabel);
  if (yLabel) {
    g += el('text', {
      x: 0, y: 0, class: 'axis-label',
      transform: `translate(12,${(pad.t + height - pad.b) / 2}) rotate(-90)`,
    }, yLabel);
  }
  return {
    open: `<svg viewBox="0 0 ${width} ${height}" class="chart ${className}" preserveAspectRatio="xMidYMid meet" role="img">`,
    grid: g, close: '</svg>', width, height, pad,
  };
}

function logTicks(scale) {
  const [d0, d1] = scale.domain;
  const out = [];
  for (let e = Math.floor(Math.log10(d0)); e <= Math.ceil(Math.log10(d1)); e++) {
    for (const m of [1, 2, 5]) {
      const v = m * 10 ** e;
      if (v >= d0 * 0.999 && v <= d1 * 1.001) out.push(v);
    }
  }
  return out;
}

/** Path string through points; nulls break the line. */
function path(pts, x, y) {
  let d = '', pen = false;
  for (const p of pts) {
    if (p === null || p.y === null || p.y === undefined || !isFinite(p.y)) { pen = false; continue; }
    d += `${pen ? 'L' : 'M'}${x(p.x).toFixed(2)},${y(p.y).toFixed(2)}`;
    pen = true;
  }
  return d;
}

function line(pts, x, y, attrs = {}) {
  return el('path', { d: path(pts, x, y), fill: 'none', ...attrs });
}

/** Shaded band between lo and hi (e.g. +/- 1 standard error). */
function band(pts, x, y, attrs = {}) {
  const top = pts.filter((p) => p && isFinite(p.hi));
  if (!top.length) return '';
  let d = '';
  top.forEach((p, i) => { d += `${i ? 'L' : 'M'}${x(p.x).toFixed(2)},${y(p.hi).toFixed(2)}`; });
  for (let i = top.length - 1; i >= 0; i--) {
    d += `L${x(top[i].x).toFixed(2)},${y(top[i].lo).toFixed(2)}`;
  }
  return el('path', { d: d + 'Z', stroke: 'none', ...attrs });
}

function dots(pts, x, y, r = 3, attrs = {}) {
  return pts.filter((p) => p && isFinite(p.y))
    .map((p) => el('circle', { cx: x(p.x).toFixed(2), cy: y(p.y).toFixed(2), r, ...attrs }))
    .join('');
}

/** Perceptually ordered ramp for alpha in [0,1]: deep violet -> yellow. */
function viridis(t) {
  const stops = [
    [0.0, 68, 1, 84], [0.25, 59, 82, 139], [0.5, 33, 145, 140],
    [0.75, 94, 201, 98], [1.0, 253, 231, 37],
  ];
  t = Math.max(0, Math.min(1, t));
  for (let i = 0; i < stops.length - 1; i++) {
    const [a, ar, ag, ab] = stops[i], [b, br, bg, bb] = stops[i + 1];
    if (t >= a && t <= b) {
      const k = (t - a) / (b - a);
      return `rgb(${Math.round(ar + k * (br - ar))},${Math.round(ag + k * (bg - ag))},${Math.round(ab + k * (bb - ab))})`;
    }
  }
  return 'rgb(253,231,37)';
}

function legend(items, opts = {}) {
  const { dash = false } = opts;
  return `<div class="legend">${items.map((i) =>
    `<span class="legend-item"><span class="swatch${i.dashed || dash ? ' dashed' : ''}" style="--c:${i.color}"></span>${i.label}</span>`
  ).join('')}</div>`;
}

root.AT = root.AT || {};
root.AT.chart = { fmt, el, linear, log, ticks, frame, logTicks, path, line, band, dots, viridis, legend };
}(window));
