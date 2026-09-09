import { type ReactNode, useEffect, useState } from 'react'

export function Card({
  title, sub, children, className = '',
}: { title?: string; sub?: string; children: ReactNode; className?: string }) {
  return (
    <section className={`rounded-xl border border-border bg-card p-5 ${className}`}>
      {title && <h3 className="text-[15px] font-semibold tracking-tight">{title}</h3>}
      {sub && <p className="mt-1 max-w-prose text-[13px] leading-relaxed text-muted-foreground">{sub}</p>}
      <div className={title || sub ? 'mt-4' : ''}>{children}</div>
    </section>
  )
}

export function Stat({ value, label, tone = 'default' }: { value: ReactNode; label: string; tone?: 'default' | 'good' | 'bad' | 'accent' }) {
  const color =
    tone === 'good' ? 'text-[var(--status-good)]'
      : tone === 'bad' ? 'text-[var(--status-bad)]'
        : tone === 'accent' ? 'text-[var(--chart-3)]'
          : ''
  return (
    <div className="min-w-28">
      <div className={`text-2xl font-semibold tracking-tight tabular-nums ${color}`}>{value}</div>
      <div className="mt-0.5 text-xs leading-snug text-muted-foreground">{label}</div>
    </div>
  )
}

/** A state chip. Semantic, not drawn from the categorical ramp. */
export function Pill({ tone = 'neutral', children }: { tone?: 'neutral' | 'good' | 'bad'; children: ReactNode }) {
  const cls = tone === 'good'
    ? 'bg-[color-mix(in_srgb,var(--status-good)_14%,transparent)] text-[var(--status-good)]'
    : tone === 'bad'
      ? 'bg-[color-mix(in_srgb,var(--status-bad)_14%,transparent)] text-[var(--status-bad)]'
      : 'bg-muted text-muted-foreground'
  return <span className={`inline-block rounded-full px-2.5 py-0.5 font-mono text-[11px] font-semibold ${cls}`}>{children}</span>
}

export function Segmented<T extends string | number>({
  options, value, onChange, label,
}: { options: { value: T; label: string }[]; value: T; onChange: (v: T) => void; label: string }) {
  return (
    <div className="flex flex-col gap-1.5">
      <span className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">{label}</span>
      <div role="group" aria-label={label} className="inline-flex overflow-hidden rounded-lg border border-border bg-muted/40">
        {options.map((o, i) => (
          <button
            key={String(o.value)}
            type="button"
            aria-pressed={o.value === value}
            onClick={() => onChange(o.value)}
            className={`px-3 py-1.5 text-[13px] font-medium transition-colors focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-[var(--chart-3)] ${
              i > 0 ? 'border-l border-border' : ''
            } ${o.value === value ? 'bg-foreground text-background' : 'text-muted-foreground hover:text-foreground'}`}
          >
            {o.label}
          </button>
        ))}
      </div>
    </div>
  )
}

export function Legend({ items }: { items: { label: string; color: string }[] }) {
  return (
    <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1.5">
      {items.map((i) => (
        <span key={i.label} className="inline-flex items-center gap-2 text-[12.5px] text-muted-foreground">
          <span aria-hidden className="h-[3px] w-4 rounded-full" style={{ background: i.color }} />
          {i.label}
        </span>
      ))}
    </div>
  )
}

export function Section({
  id, eyebrow, title, lede, children,
}: { id: string; eyebrow: string; title: string; lede?: string; children: ReactNode }) {
  return (
    <section id={id} className="scroll-mt-20 border-t border-border/60 py-14 first:border-t-0">
      <header className="mb-6">
        <span className="font-mono text-[11px] font-semibold uppercase tracking-[0.12em] text-[var(--chart-3)]">{eyebrow}</span>
        <h2 className="mt-2 text-balance text-2xl font-semibold tracking-tight sm:text-[28px]">{title}</h2>
        {lede && <p className="mt-2 max-w-prose text-[16px] leading-relaxed text-muted-foreground">{lede}</p>}
      </header>
      {children}
    </section>
  )
}

export function ThemeToggle() {
  const [dark, setDark] = useState(() =>
    typeof window !== 'undefined'
    && (localStorage.getItem('at-theme') === 'dark'
      || (!localStorage.getItem('at-theme') && window.matchMedia('(prefers-color-scheme: dark)').matches)))

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
    try { localStorage.setItem('at-theme', dark ? 'dark' : 'light') } catch { /* private mode */ }
  }, [dark])

  return (
    <button
      type="button"
      onClick={() => setDark((d) => !d)}
      aria-label={dark ? 'Switch to light theme' : 'Switch to dark theme'}
      className="grid size-8 place-items-center rounded-lg border border-border text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--chart-3)]"
    >
      {dark ? '☾' : '☀'}
    </button>
  )
}

/* ---------------------------------------------------------------------------
   ChartFrame

   bklit's bar charts ship a category axis (BarXAxis for vertical bars,
   BarYAxis for horizontal ones) but no numeric value axis: values are carried
   by gridlines and the tooltip. For a research audience that is not enough, so
   this supplies the missing axis.

   It also owns sizing. bklit's chart root sets `aspect-ratio` on itself and is
   `overflow-visible`, so wrapping it in a fixed-height box does not constrain
   it -- the chart simply spills over whatever follows. Here the chart owns its
   height via aspectRatio and the tick overlay is absolutely positioned on the
   same box, so the two can never disagree.

   The label positions replicate bklit's own value scale exactly: domain
   [0, max * 1.1] mapped across the plot area, which sits inside a 40px top and
   bottom margin.
   --------------------------------------------------------------------------- */

const CHART_MARGIN_PX = 40

/** Rounded tick values across [0, max], excluding 0 (the baseline). */
export function niceTicks(max: number, count = 4): number[] {
  if (!Number.isFinite(max) || max <= 0) return []
  const raw = max / count
  const mag = 10 ** Math.floor(Math.log10(raw))
  const norm = raw / mag
  const stepSize = (norm >= 7.5 ? 10 : norm >= 3.5 ? 5 : norm >= 1.5 ? 2 : 1) * mag
  const out: number[] = []
  for (let v = stepSize; v <= max + stepSize * 1e-9; v += stepSize) {
    out.push(Number(v.toPrecision(12)))
  }
  return out
}

export function ChartFrame({
  max, ticks, unit, children, decimals = 2,
}: {
  max: number
  ticks?: number[]
  unit?: string
  decimals?: number
  children: ReactNode
}) {
  const domainMax = max * 1.1
  const values = ticks ?? niceTicks(max)
  return (
    <div className="relative">
      {children}
      <div aria-hidden className="pointer-events-none absolute inset-0">
        {values.map((v) => (
          <span
            key={v}
            className="absolute left-0 w-8 -translate-y-1/2 text-right font-mono text-[10.5px] tabular-nums text-muted-foreground"
            style={{ top: `calc(${CHART_MARGIN_PX}px + ${1 - v / domainMax} * (100% - ${CHART_MARGIN_PX * 2}px))` }}
          >
            {v.toFixed(decimals)}
          </span>
        ))}
      </div>
      {unit && (
        <span className="pointer-events-none absolute left-0 top-1.5 font-mono text-[10.5px] uppercase tracking-wider text-muted-foreground">
          {unit}
        </span>
      )}
    </div>
  )
}
