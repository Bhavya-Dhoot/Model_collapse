import { BarChart } from '@/components/charts/bar-chart'
import { Bar } from '@/components/charts/bar'
import { BarXAxis } from '@/components/charts/bar-x-axis'
import { BarYAxis } from '@/components/charts/bar-y-axis'
import { Grid } from '@/components/charts/grid'
import { ChartTooltip } from '@/components/charts/tooltip'
import { Card, Legend, Stat } from '@/components/Shell'
import { AXIS_COLOR, AXIS_LABEL, D, fmt, type Axis } from '@/lib/data'

/* How anchoring cost falls as the real-data budget n grows. The theoretical
   closed form predicts alpha_star ~ 1/n (exponent 1); the measured pooled
   exponent is far below that, so a bar chart per axis across the n-grid shows
   the shortfall directly instead of implying a smooth trend a line would. */

export function Scaling() {
  const { ns, beta, beta_ci, fit_axes, axes } = D.nscaling

  const byN = ns.map((n, i) => {
    const row: Record<string, unknown> = { name: String(n) }
    for (const a of fit_axes) {
      const v = axes[a]?.alpha_star[i]
      if (v !== null && v !== undefined) row[a] = v
    }
    return row
  })

  const corrFrob = axes.corr_frob
  const w1 = axes.w1

  return (
    <div className="grid gap-4">
      <Card
        title="α★ falls as the real-data budget grows"
        sub="One row per training budget n, one bar per axis fit to the scaling law. Missing bars mean the axis's threshold was undefined at that budget (not zero)."
      >
        <div className="h-[400px]">
          <BarChart data={byN} xDataKey="name" barGap={0.25}>
            <Grid horizontal numTicksRows={5} />
            {fit_axes.map((a) => (
              <Bar key={a} dataKey={a} fill={AXIS_COLOR[a as Axis]} />
            ))}
            <BarXAxis showAllLabels />
            <BarYAxis />
            <ChartTooltip />
          </BarChart>
        </div>
        <Legend items={fit_axes.map((a) => ({ label: AXIS_LABEL[a as Axis], color: AXIS_COLOR[a as Axis] }))} />

        <div className="mt-5 flex flex-wrap gap-6">
          <Stat value={fmt(beta, 2)} label={`measured exponent β — 95% CI [${fmt(beta_ci[0], 2)}, ${fmt(beta_ci[1], 2)}]`} tone="accent" />
          <Stat value="1.00" label="theoretical exponent (closed form, 1/n)" tone="bad" />
        </div>

        <p className="mt-4 text-[13.5px] leading-relaxed text-muted-foreground">
          The bootstrap interval excludes 0 — α★ genuinely falls as n grows — but it also excludes 1 by
          a wide margin, so the scaling is not a 1/n law: more data helps, just far less than the theory promises.
        </p>

        <p className="mt-3 border-l-2 border-[var(--chart-3)] bg-muted/40 py-2.5 pl-3 text-[13.5px] leading-relaxed text-muted-foreground">
          What the theory asks for: at the smallest budget, correlation structure needs α★ = {fmt(corrFrob?.alpha_star[0])}
          {' '}and Wasserstein-1 needs α★ = {fmt(w1?.alpha_star[0])}; at the largest budget those fall to
          {' '}{fmt(corrFrob?.alpha_star[corrFrob.alpha_star.length - 1])} and {fmt(w1?.alpha_star[w1.alpha_star.length - 1])}
          {' '}respectively. Scale makes nearly every axis cheaper to protect except dependence structure
          (correlation), which stays roughly flat — that shortfall is why the pooled exponent falls short of 1.
          Variance ratio and TSTR utility are excluded from the fit because they are non-monotone in n.
        </p>
      </Card>
    </div>
  )
}
