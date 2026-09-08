import { motion, useReducedMotion } from 'motion/react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { Card, Legend, Pill, Stat } from '@/components/Shell'
import { D as DATA, fmt } from '@/lib/data'

/* An animated, running prototype of the anchored self-consuming loop.
 *
 * Every animation here is motivated: the particles show what the composition
 * of a training set actually is at a given alpha, and the fidelity bar is
 * driven by the paper's own recursion -- not an eased decoration. Turning the
 * slider changes both, so the mechanism explains itself.
 *
 * Reduced motion is a designed branch: particles stop travelling and the
 * generation still advances, so the numbers and the story still land. */

const SIGMA = 1

/** D(alpha, n) = 1 - alpha + alpha*n*(2 - alpha) -- paper, Proposition 3. */
const Dcoef = (alpha: number, n: number) => 1 - alpha + alpha * n * (2 - alpha)

/** One exact step of E[Sigma_t] / Var[mu_t]. */
function step(sigma: number, v: number, alpha: number, n: number) {
  const c = (n - 1) / n
  return {
    sigma: alpha * c * SIGMA + (1 - alpha) * c * sigma + alpha * (1 - alpha) * v,
    v: (alpha / n) * SIGMA + ((1 - alpha) / n) * sigma + (1 - alpha) ** 2 * v,
  }
}

const PARTICLES = 14

export function LoopAnimation() {
  const reduced = useReducedMotion()
  const alphas = DATA.meta.alphas
  const [ai, setAi] = useState(() => Math.max(0, alphas.indexOf(0.05)))
  const alpha = alphas[ai]
  const ns = DATA.meta.ns
  const [N, setN] = useState(() => (ns.includes(2000) ? 2000 : ns[0]))
  const [gen, setGen] = useState(0)
  const [state, setState] = useState({ sigma: SIGMA, v: 0 })
  const [running, setRunning] = useState(true)
  const timer = useRef<number | null>(null)

  // advance the loop one generation per tick, using the exact recursion
  useEffect(() => {
    if (!running) return
    timer.current = window.setInterval(() => {
      setGen((g) => g + 1)
      setState((s) => step(s.sigma, s.v, alpha, N))
    }, 900)
    return () => { if (timer.current) window.clearInterval(timer.current) }
  }, [running, alpha, N])

  // changing alpha restarts the loop: the trajectory it produces is different
  useEffect(() => { setGen(0); setState({ sigma: SIGMA, v: 0 }) }, [alpha, N])

  const fixed = useMemo(() => {
    const d = Dcoef(alpha, N)
    return { sigma: (1 - 1 / d) * SIGMA, v: SIGMA / d }
  }, [alpha, N])

  const real = Math.ceil(alpha * N)
  const synth = N - real
  const nReal = Math.round(alpha * PARTICLES)

  const tone = alpha === 0 ? 'bad' : alpha === 1 ? 'good' : 'neutral'
  const label = alpha === 0 ? 'pure self-consumption' : alpha === 1 ? 'retraining on real data' : 'anchored'

  return (
    <div className="grid gap-4 lg:grid-cols-[320px_1fr]">
      <Card title="Anchor fraction α" sub="Raising α buys protection by displacing synthetic rows — never by enlarging the training set.">
        <div className="mb-4 flex flex-col gap-1.5">
          <span className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">Training budget n</span>
          <div role="group" aria-label="Training budget" className="inline-flex flex-wrap overflow-hidden rounded-lg border border-border">
            {ns.map((v, i) => (
              <button key={v} type="button" aria-pressed={v === N} onClick={() => setN(v)}
                className={`px-2.5 py-1.5 text-[12.5px] font-medium transition-colors ${i > 0 ? 'border-l border-border' : ''} ${
                  v === N ? 'bg-foreground text-background' : 'text-muted-foreground hover:text-foreground'}`}>
                {v.toLocaleString()}
              </button>
            ))}
          </div>
          <p className="text-[11.5px] leading-snug text-muted-foreground">
            Smaller budgets collapse visibly faster — the paper&rsquo;s central claim.
          </p>
        </div>

        <label className="flex flex-col gap-1.5">
          <span className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">α = {alpha}</span>
          <input
            type="range" min={0} max={alphas.length - 1} step={1} value={ai}
            onChange={(e) => setAi(Number(e.target.value))}
            aria-label="Anchor fraction alpha"
            aria-valuetext={`alpha ${alpha}, ${real} real of ${N} rows`}
            className="w-full accent-[var(--chart-1)]"
          />
        </label>

        <div className="mt-3"><Pill tone={tone as 'good' | 'bad' | 'neutral'}>{label}</Pill></div>

        <div className="mt-4 flex flex-wrap gap-5">
          <Stat value={real.toLocaleString()} label="real rows" tone="accent" />
          <Stat value={synth.toLocaleString()} label="synthetic rows" />
          <Stat value={gen} label="generation" />
        </div>

        <div className="mt-5">
          <div className="flex items-baseline justify-between text-[12px] text-muted-foreground">
            <span>variance retained</span>
            <span className="font-mono tabular-nums text-foreground">{fmt(state.sigma, 4)}</span>
          </div>
          <div className="mt-1.5 h-2 overflow-hidden rounded-full bg-muted">
            <motion.div
              className="h-full rounded-full"
              style={{ background: alpha === 0 ? 'var(--status-bad)' : 'var(--chart-1)' }}
              animate={{ width: `${Math.max(0, Math.min(1, state.sigma)) * 100}%` }}
              transition={reduced ? { duration: 0 } : { type: 'spring', stiffness: 120, damping: 20 }}
            />
          </div>
          <p className="mt-2 text-[12px] leading-relaxed text-muted-foreground">
            {alpha === 0
              ? 'With no real data the recursion has no fixed point: variance decays geometrically toward zero.'
              : <>Converging to the closed-form fixed point Σ∞ = <span className="font-mono text-foreground">{fmt(fixed.sigma, 4)}</span>, at rate 1 − α.</>}
          </p>
        </div>

        <button
          type="button"
          onClick={() => setRunning((r) => !r)}
          className="mt-4 rounded-lg border border-border px-3 py-1.5 text-[13px] font-medium transition-colors hover:bg-muted focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--chart-3)]"
        >
          {running ? 'Pause loop' : 'Run loop'}
        </button>
      </Card>

      <Card>
        <svg viewBox="0 0 640 330" className="w-full" role="img"
          aria-label={`Anchored self-consuming loop at alpha ${alpha}: ${real} real and ${synth} synthetic rows of a ${N} row budget.`}>
          <defs>
            <marker id="loop-head" viewBox="0 0 10 10" refX="8" refY="5" markerUnits="userSpaceOnUse"
              markerWidth="11" markerHeight="11" orient="auto-start-reverse">
              <path d="M0,0 L10,5 L0,10 Z" fill="currentColor" />
            </marker>
            <path id="feed-real" d="M150,58 L286,152" />
            <path id="feed-synth" d="M540,140 C540,42 342,42 342,140" />
          </defs>

          {/* structure */}
          <g className="text-muted-foreground" stroke="currentColor" fill="none" strokeWidth={1.6}>
            <use href="#feed-real" markerEnd="url(#loop-head)" />
            <use href="#feed-synth" markerEnd="url(#loop-head)" />
            <path d="M424,170 L468,170" markerEnd="url(#loop-head)" />
            <path d="M540,202 L540,248" markerEnd="url(#loop-head)" />
          </g>

          {[
            { x: 20, y: 24, w: 130, h: 62, t: 'Real pool', s: 'known-real rows' },
            { x: 286, y: 140, w: 138, h: 62, t: 'Training set', s: `${N.toLocaleString()} rows` },
            { x: 468, y: 140, w: 130, h: 62, t: 'Model Gₜ', s: `generation ${gen}` },
            { x: 468, y: 248, w: 130, h: 56, t: 'Held-out real', s: 'never trained on' },
          ].map((b) => (
            <g key={b.t}>
              <rect x={b.x} y={b.y} width={b.w} height={b.h} rx={11}
                className="fill-card stroke-border" strokeWidth={1} />
              <text x={b.x + b.w / 2} y={b.y + b.h / 2 - 3} textAnchor="middle"
                className="fill-foreground" fontSize="13" fontWeight="650">{b.t}</text>
              <text x={b.x + b.w / 2} y={b.y + b.h / 2 + 14} textAnchor="middle"
                className="fill-muted-foreground" fontSize="10.5">{b.s}</text>
            </g>
          ))}

          <text x="152" y="112" className="fill-muted-foreground" fontSize="10.5" fontFamily="ui-monospace, monospace">α·n real</text>
          <text x="380" y="36" className="fill-muted-foreground" fontSize="10.5" fontFamily="ui-monospace, monospace">(1−α)·n synthetic</text>

          {/* composition bar inside the training set */}
          <rect x={298} y={186} width={114} height={9} rx={4.5} className="fill-muted" />
          <motion.rect
            y={186} height={9} rx={4.5} x={298} fill="var(--chart-1)"
            animate={{ width: 114 * alpha }}
            transition={reduced ? { duration: 0 } : { type: 'spring', stiffness: 160, damping: 22 }}
          />

          {/* particles: each one is a row entering the next training set */}
          {!reduced && Array.from({ length: PARTICLES }, (_, i) => {
            const isReal = i < nReal
            return (
              <circle key={`${alpha}-${i}`} r={3.4}
                fill={isReal ? 'var(--chart-1)' : 'var(--muted-foreground)'}
                opacity={isReal ? 1 : 0.55}>
                <animateMotion dur={`${isReal ? 2.6 : 3.4}s`} repeatCount="indefinite"
                  begin={`${(i / PARTICLES) * (isReal ? 2.6 : 3.4)}s`}>
                  <mpath href={isReal ? '#feed-real' : '#feed-synth'} />
                </animateMotion>
              </circle>
            )
          })}
        </svg>

        <Legend items={[
          { label: 'real rows entering the budget', color: 'var(--chart-1)' },
          { label: 'the model’s own output fed back', color: 'var(--muted-foreground)' },
        ]} />
      </Card>
    </div>
  )
}
