import { BarChart } from '@/components/charts/bar-chart'
import { Bar } from '@/components/charts/bar'
import { BarXAxis } from '@/components/charts/bar-x-axis'
import { Grid } from '@/components/charts/grid'
import { ChartTooltip } from '@/components/charts/tooltip'
import { Card, ChartFrame, Legend, Stat } from '@/components/Shell'
import { AXES, AXIS_COLOR, AXIS_LABEL, D, fmt, type Axis } from '@/lib/data'

/* Per-axis anchoring thresholds. This is a comparison of magnitudes across
   seven named categories, so a bar chart is the right form -- the axes have no
   order, and the question is "which one binds", not "how does it trend". */

const COLLAPSE_AXES = AXES.filter((a) => a !== 'var_ratio' && a !== 'tstr_auc')

export function Thresholds() {
  const adult = D.alpha_star_table.find((r) => r.dataset === 'adult' && r.n === 2000)!

  const perAxis = AXES.map((a) => ({
    axis: a,
    name: AXIS_LABEL[a],
    alphaStar: adult[a] as number | null,
  })).filter((r) => r.alphaStar !== null)
    .sort((a, b) => (b.alphaStar as number) - (a.alphaStar as number))

  const carried = perAxis.filter((r) => COLLAPSE_AXES.includes(r.axis as never))
  const strictest = carried[0]
  const loosest = carried[carried.length - 1]

  // One row per dataset/budget, one bar series per axis.
  const byDataset = D.alpha_star_table.map((r) => {
    const row: Record<string, unknown> = { name: `${r.dataset} n=${r.n}` }
    for (const a of COLLAPSE_AXES) row[a] = r[a]
    return row
  })

  const maxPerAxis = Math.max(...perAxis.map((r) => r.alphaStar as number))
  const maxByDataset = Math.max(...byDataset.flatMap((r) =>
    COLLAPSE_AXES.map((a) => r[a]).filter((v): v is number => typeof v === 'number' && Number.isFinite(v))))

  return (
    <div className="grid gap-4">
      <Card
        title="Where each axis crosses its tolerance"
        sub={`Adult at n=2000, Gaussian copula, fixed anchoring. α★ is the smallest real-data fraction that keeps an axis within tolerance and stays within it for every larger α, so an isolated noisy dip is not counted as a threshold.`}
      >
        <ChartFrame max={maxPerAxis} unit="α★" decimals={2}>
          <BarChart data={perAxis} xDataKey="name" aspectRatio="16 / 6" barGap={0.3}>
            <Grid horizontal numTicksRows={4} />
            <Bar dataKey="alphaStar" fill="var(--chart-1)" />
            <BarXAxis maxLabels={7} showAllLabels />
            <ChartTooltip />
          </BarChart>
        </ChartFrame>
        <Legend items={[{ label: 'α★ — required real-data fraction', color: 'var(--chart-1)' }]} />

        <div className="mt-5 flex flex-wrap gap-6">
          <Stat value={fmt(strictest?.alphaStar, 3)} label={`strictest axis — ${strictest?.name.toLowerCase()}`} tone="bad" />
          <Stat value={fmt(loosest?.alphaStar, 3)} label={`loosest axis — ${loosest?.name.toLowerCase()}`} tone="good" />
          <Stat value={`${fmt(adult.ratio_excl_var, 1)}×`} label="spread between them" tone="accent" />
        </div>

        <p className="mt-4 border-l-2 border-[var(--chart-3)] bg-muted/40 py-2.5 pl-3 text-[13.5px] leading-relaxed text-muted-foreground">
          Validating a synthetic pipeline against one aggregate score can pass while the rare categories
          that made the dataset worth synthesizing are already gone. A safety specification has to be
          stated per axis. <span className="text-foreground">Variance is shown for completeness only</span> —
          it is not a copula collapse axis, so it is excluded from the spread above.
        </p>
      </Card>

      <Card
        title="The disagreement is not specific to one dataset"
        sub="Each group is a dataset and training budget; each bar an axis. The binding axis changes between datasets, which is the point — there is no single number to ship."
      >
        <ChartFrame max={maxByDataset} unit="α★" decimals={2}>
          <BarChart data={byDataset} xDataKey="name" aspectRatio="16 / 7" barGap={0.25}>
            <Grid horizontal numTicksRows={4} />
            {COLLAPSE_AXES.map((a) => (
              <Bar key={a} dataKey={a} fill={AXIS_COLOR[a as Axis]} />
            ))}
            <BarXAxis showAllLabels />
            <ChartTooltip />
          </BarChart>
        </ChartFrame>
        <Legend items={COLLAPSE_AXES.map((a) => ({ label: AXIS_LABEL[a as Axis], color: AXIS_COLOR[a as Axis] }))} />
      </Card>
    </div>
  )
}

