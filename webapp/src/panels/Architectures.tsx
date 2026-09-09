import { BarChart } from '@/components/charts/bar-chart'
import { Bar } from '@/components/charts/bar'
import { BarXAxis } from '@/components/charts/bar-x-axis'
import { Grid } from '@/components/charts/grid'
import { ChartTooltip } from '@/components/charts/tooltip'
import { Card, ChartFrame, Legend, Pill, niceTicks } from '@/components/Shell'
import { D, SYNTH_LABEL, fmt } from '@/lib/data'

/* The exact closed-form theory covers Gaussian maximum-likelihood fitting
   only. TVAE and CTGAN are neural synthesizers -- the theory is an analogy
   there, not a derivation -- so this panel checks whether the two qualitative
   findings (anchoring restores categorical support; anchoring stabilizes
   variance) survive the change of architecture, using bar charts across the
   three synthesizers. */

const FLAT_THRESHOLD = 0.05

export function Architectures() {
  const rows = D.architectures
  const gate = D.gen0_gate

  const supportData = rows.map((r) => ({
    name: SYNTH_LABEL[r.synth] ?? r.synth,
    selfConsuming: r.a0.cat_support.end,
    anchored: r.a1.cat_support.end,
  }))

  const varRatioData = rows.map((r) => ({
    name: SYNTH_LABEL[r.synth] ?? r.synth,
    varRatio: r.a0.var_ratio.end,
  }))

  const supportMax = Math.max(...supportData.flatMap((r) => [r.selfConsuming, r.anchored])
    .filter((v): v is number => v !== null && v !== undefined && Number.isFinite(v)))

  const varRatioMax = Math.max(...varRatioData.map((r) => r.varRatio)
    .filter((v): v is number => v !== null && v !== undefined && Number.isFinite(v)))
  const varRatioTicks = Array.from(new Set([...niceTicks(varRatioMax, 4), 1])).sort((a, b) => a - b)

  const varRatioMoves = rows.map((r) => {
    const delta = r.a0.var_ratio.end - r.a0.var_ratio.start
    const tone: 'flat' | 'contracts' | 'inflates' =
      Math.abs(delta) < FLAT_THRESHOLD ? 'flat' : delta < 0 ? 'contracts' : 'inflates'
    return { synth: r.synth, label: SYNTH_LABEL[r.synth] ?? r.synth, tone }
  })

  const copula = rows.find((r) => r.synth === 'gaussian_copula')
  const neural = rows.find((r) => r.synth !== 'gaussian_copula')

  return (
    <div className="grid gap-4">
      <Card
        title="Categorical support"
        sub="End-of-run categorical support at α=0 (self-consuming) versus anchored, for each architecture."
      >
        <ChartFrame max={supportMax} unit="support" decimals={2}>
          <BarChart data={supportData} xDataKey="name" aspectRatio="16 / 6" barGap={0.3}>
            <Grid horizontal numTicksRows={4} />
            <Bar dataKey="selfConsuming" fill="var(--status-bad)" />
            <Bar dataKey="anchored" fill="var(--status-good)" />
            <BarXAxis showAllLabels />
            <ChartTooltip />
          </BarChart>
        </ChartFrame>
        <Legend items={[
          { label: 'Self-consuming (α=0)', color: 'var(--status-bad)' },
          { label: 'Anchored (α=1)', color: 'var(--status-good)' },
        ]}
        />
      </Card>

      <Card
        title="Variance ratio at α=0"
        sub="Self-consuming end-of-run variance ratio per architecture, with how it moved from its start value."
      >
        <ChartFrame max={varRatioMax} ticks={varRatioTicks} unit="variance ratio" decimals={2}>
          <BarChart data={varRatioData} xDataKey="name" aspectRatio="16 / 6" barGap={0.3}>
            <Grid horizontal numTicksRows={4} />
            <Bar dataKey="varRatio" fill="var(--chart-1)" />
            <BarXAxis showAllLabels />
            <ChartTooltip />
          </BarChart>
        </ChartFrame>
        <div className="mt-3 flex flex-wrap gap-3">
          {varRatioMoves.map((m) => (
            <span key={m.synth} className="inline-flex items-center gap-2 text-[12.5px] text-muted-foreground">
              {m.label}
              <Pill tone={m.tone === 'contracts' ? 'good' : m.tone === 'inflates' ? 'bad' : 'neutral'}>{m.tone}</Pill>
            </span>
          ))}
        </div>
      </Card>

      <Card title="Generation-0 gate" sub="Baseline quality of each synthesizer's first generation, before any collapse.">
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-left text-[13px]">
            <thead>
              <tr className="border-b border-border text-muted-foreground">
                <th scope="col" className="py-2 pr-4 font-medium">Synthesizer</th>
                <th scope="col" className="py-2 pr-4 font-medium">Categorical support</th>
                <th scope="col" className="py-2 pr-4 font-medium">Categorical TV</th>
                <th scope="col" className="py-2 pr-4 font-medium">TSTR AUC</th>
                <th scope="col" className="py-2 pr-4 font-medium">Seconds / generation</th>
              </tr>
            </thead>
            <tbody>
              {gate.map((g) => (
                <tr key={g.synth} className="border-b border-border/60">
                  <td className="py-2 pr-4">{SYNTH_LABEL[g.synth] ?? g.synth}</td>
                  <td className="py-2 pr-4 tabular-nums">{fmt(g.cat_support, 3)}</td>
                  <td className="py-2 pr-4 tabular-nums">{fmt(g.tv_cat, 3)}</td>
                  <td className="py-2 pr-4 tabular-nums">{fmt(g.tstr_auc, 3)}</td>
                  <td className="py-2 pr-4 tabular-nums">{g.sec_per_gen.toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <p className="border-l-2 border-border bg-muted/40 py-2.5 pl-3 text-[13.5px] leading-relaxed text-muted-foreground">
        The closed-form theory is exact for Gaussian maximum-likelihood fitting only; for TVAE and CTGAN it is an
        analogy, not a derivation, and only the two qualitative findings above — anchoring restores categorical
        support, and anchoring changes the variance-ratio drift — carry over. The terminal generation also differs
        by architecture: the copula collapses by generation {fmt(copula?.terminal_generation, 0)}, while the
        neural synthesizers reach their terminal generation by {fmt(neural?.terminal_generation, 0)}.
      </p>
    </div>
  )
}
