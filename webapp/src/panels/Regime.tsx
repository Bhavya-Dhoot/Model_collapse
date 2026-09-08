import { BarChart } from '@/components/charts/bar-chart'
import { Bar } from '@/components/charts/bar'
import { BarXAxis } from '@/components/charts/bar-x-axis'
import { BarYAxis } from '@/components/charts/bar-y-axis'
import { Grid } from '@/components/charts/grid'
import { ChartTooltip } from '@/components/charts/tooltip'
import { Card, Legend, Pill, Stat } from '@/components/Shell'
import { D, fmt } from '@/lib/data'

/* Story: numeric drift is a contraction that any amount of real data pulls
   back, so fixed vs fresh anchoring should agree there. A lost category only
   comes back through an anchor row that still contains it, so fixed and fresh
   anchoring separate on categorical support specifically -- and nowhere else. */

const pFmt = (p: number | undefined) => (p === undefined ? '—' : p < 0.001 ? '<0.001' : fmt(p, 3))

export function Regime() {
  const rows = D.regime
  const chartData = rows.map((r) => ({
    name: String(r.alpha),
    fixed: r.fixed,
    fresh: r.fresh,
  }))

  const peak = rows.reduce((best, r) => (best === null || r.diff > best.diff ? r : best), null as (typeof rows)[number] | null)
  const significant = rows.filter((r) => r.p !== undefined && r.p < 0.05)
  const atZero = rows.find((r) => r.alpha === 0)

  return (
    <div className="grid gap-4">
      <Card
        title="Fixed vs. fresh anchoring, by real-data fraction"
        sub="Fixed anchoring reuses one real-data sample across every generation; fresh anchoring draws a new one each generation. Numeric drift is a contraction that either regime pulls back equally -- the two only diverge on categorical support, where a category can only return through an anchor row that happens to contain it."
      >
        <div className="h-[400px]">
          <BarChart data={chartData} xDataKey="name" barGap={0.3}>
            <Grid horizontal numTicksRows={5} />
            <Bar dataKey="fixed" fill="var(--chart-3)" />
            <Bar dataKey="fresh" fill="var(--chart-1)" />
            <BarXAxis showAllLabels />
            <BarYAxis />
            <ChartTooltip />
          </BarChart>
        </div>
        <Legend items={[
          { label: 'fixed anchoring', color: 'var(--chart-3)' },
          { label: 'fresh anchoring', color: 'var(--chart-1)' },
        ]}
        />

        <div className="mt-5 flex flex-wrap gap-6">
          <Stat
            value={peak ? fmt(peak.diff, 3) : '—'}
            label={peak ? `peak improvement — α = ${peak.alpha}` : 'peak improvement'}
            tone="good"
          />
          <Stat value={`${significant.length} / ${rows.length}`} label="α values significant (p < 0.05)" tone="accent" />
          <Stat
            value={atZero ? fmt(atZero.diff, 3) : '—'}
            label="difference at α = 0"
            tone={atZero ? (atZero.diff > 0 ? 'good' : 'default') : 'default'}
          />
        </div>

        <div className="mt-5 overflow-x-auto">
          <table className="w-full text-[13.5px]">
            <thead>
              <tr className="border-b border-border text-muted-foreground">
                <th scope="col" className="py-1.5 pr-3 text-left font-medium">α</th>
                <th scope="col" className="py-1.5 pr-3 text-right font-medium">fixed</th>
                <th scope="col" className="py-1.5 pr-3 text-right font-medium">fresh</th>
                <th scope="col" className="py-1.5 pr-3 text-right font-medium">difference</th>
                <th scope="col" className="py-1.5 pr-3 text-right font-medium">p</th>
                <th scope="col" className="py-1.5 text-right font-medium">result</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.alpha} className="border-b border-border">
                  <td className="py-1.5 pr-3 text-left">{r.alpha}</td>
                  <td className="py-1.5 pr-3 text-right tabular-nums">{fmt(r.fixed, 3)}</td>
                  <td className="py-1.5 pr-3 text-right tabular-nums">{fmt(r.fresh, 3)}</td>
                  <td className="py-1.5 pr-3 text-right tabular-nums">{fmt(r.diff, 3)}</td>
                  <td className="py-1.5 pr-3 text-right tabular-nums">{pFmt(r.p)}</td>
                  <td className="py-1.5 text-right">
                    {r.p !== undefined && r.p < 0.05
                      ? <Pill tone="good">significant</Pill>
                      : <Pill>not significant</Pill>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <p className="mt-4 border-l-2 border-[var(--chart-3)] bg-muted/40 py-2.5 pl-3 text-[13.5px] leading-relaxed text-muted-foreground">
          The anchoring regime is a modeling choice, not a free parameter to tune away collapse -- it only moves
          the categorical-support axis, and only where the table above shows a significant difference.
        </p>
      </Card>
    </div>
  )
}
