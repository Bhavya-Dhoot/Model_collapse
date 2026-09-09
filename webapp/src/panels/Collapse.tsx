import { useState } from 'react'
import { BarChart } from '@/components/charts/bar-chart'
import { Bar } from '@/components/charts/bar'
import { BarXAxis } from '@/components/charts/bar-x-axis'
import { Grid } from '@/components/charts/grid'
import { ChartTooltip } from '@/components/charts/tooltip'
import { Card, ChartFrame, Legend, Segmented, Stat } from '@/components/Shell'
import { D, DATASETS, alphaKeys, fmt } from '@/lib/data'

/* Story: with no real data (small alpha) the model degrades every generation;
   with alpha = 1 (all real data every generation) it stays flat, because the
   pipeline is then just re-fitting real data, not evaluating drift. Comparing
   a chosen alpha against the alpha=1 control isolates that. */

function seriesFor(dataset: string, alphaKey: string, metric: 'tstr_auc' | 'cat_support') {
  const points = D.traj[dataset]?.[alphaKey] ?? []
  const byGen = new Map(points.map((p) => [p.g, p]))
  const terminal = D.meta.terminal_generation
  const rows: { g: number; value: number | null }[] = []
  for (let g = 0; g <= terminal; g++) {
    rows.push({ g, value: byGen.get(g)?.[metric] ?? null })
  }
  return rows
}

export function Collapse() {
  const [dataset, setDataset] = useState<(typeof DATASETS)[number]>('adult')
  const candidateAlphas = alphaKeys(dataset).filter((a) => a !== '1')
  const [alpha, setAlpha] = useState(candidateAlphas[0] ?? '0')
  const selectedAlpha = candidateAlphas.includes(alpha) ? alpha : (candidateAlphas[0] ?? '0')

  const selectedAuc = seriesFor(dataset, selectedAlpha, 'tstr_auc')
  const controlAuc = seriesFor(dataset, '1', 'tstr_auc')
  const aucData = selectedAuc.map((r, i) => ({
    name: String(r.g),
    selected: r.value,
    control: controlAuc[i]?.value ?? null,
  }))

  const selectedSupport = seriesFor(dataset, selectedAlpha, 'cat_support')
  const controlSupport = seriesFor(dataset, '1', 'cat_support')
  const supportData = selectedSupport.map((r, i) => ({
    name: String(r.g),
    selected: r.value,
    control: controlSupport[i]?.value ?? null,
  }))

  const aucMax = Math.max(
    ...aucData.flatMap((r) => [r.selected, r.control]).filter((v): v is number => v !== null && Number.isFinite(v)),
  )
  const supportMax = Math.max(
    ...supportData.flatMap((r) => [r.selected, r.control]).filter((v): v is number => v !== null && Number.isFinite(v)),
  )

  const gen0 = selectedAuc[0]?.value ?? null
  const terminalValue = selectedAuc[selectedAuc.length - 1]?.value ?? null
  const changed = gen0 !== null && terminalValue !== null ? terminalValue - gen0 : null
  const ceiling = D.ceiling[dataset]

  return (
    <div className="grid gap-4">
      <Card
        title="Self-consumption drives the drop, not the metric"
        sub="Regenerating from a model's own synthetic output, generation after generation, degrades utility -- unless every generation is refit on real data (alpha = 1), which holds it flat. That contrast is the evidence that we are measuring self-consumption, not evaluation noise."
      >
        <div className="flex flex-wrap gap-6">
          <Segmented
            label="Dataset"
            value={dataset}
            onChange={(v) => setDataset(v)}
            options={DATASETS.map((d) => ({ value: d, label: d }))}
          />
          <Segmented
            label="alpha"
            value={selectedAlpha}
            onChange={(v) => setAlpha(v)}
            options={candidateAlphas.map((a) => ({ value: a, label: `α = ${a}` }))}
          />
        </div>

        <div className="mt-5">
          <ChartFrame max={aucMax} unit="AUC" decimals={2}>
            <BarChart data={aucData} xDataKey="name" aspectRatio="16 / 6" barGap={0.3}>
              <Grid horizontal numTicksRows={4} />
              <Bar dataKey="selected" fill="var(--chart-1)" />
              <Bar dataKey="control" fill="var(--chart-2)" />
              <BarXAxis maxLabels={8} />
              <ChartTooltip />
            </BarChart>
          </ChartFrame>
        </div>
        <Legend
          items={[
            { label: `α = ${selectedAlpha}`, color: 'var(--chart-1)' },
            { label: 'α = 1 (real-data control)', color: 'var(--chart-2)' },
          ]}
        />

        <div className="mt-5 flex flex-wrap gap-6">
          <Stat value={fmt(gen0, 3)} label="generation 0 utility" />
          <Stat value={fmt(terminalValue, 3)} label={`generation ${D.meta.terminal_generation} utility`} />
          <Stat
            value={changed === null ? '—' : `${changed >= 0 ? '+' : ''}${changed.toFixed(3)}`}
            label="change over run"
            tone={changed === null ? 'default' : changed < 0 ? 'bad' : 'good'}
          />
          <Stat value={fmt(ceiling, 3)} label={`train-on-real ceiling — ${dataset}`} tone="accent" />
        </div>
      </Card>

      <Card
        title="Categorical support retained"
        sub="The same contrast for how much of the original categorical support survives regeneration. A collapse that erases rare categories can still look fine on an aggregate score, so this axis is tracked on its own."
      >
        <ChartFrame max={supportMax} unit="support" decimals={2}>
          <BarChart data={supportData} xDataKey="name" aspectRatio="16 / 6" barGap={0.3}>
            <Grid horizontal numTicksRows={4} />
            <Bar dataKey="selected" fill="var(--chart-1)" />
            <Bar dataKey="control" fill="var(--chart-2)" />
            <BarXAxis maxLabels={8} />
            <ChartTooltip />
          </BarChart>
        </ChartFrame>
        <Legend
          items={[
            { label: `α = ${selectedAlpha}`, color: 'var(--chart-1)' },
            { label: 'α = 1 (real-data control)', color: 'var(--chart-2)' },
          ]}
        />
      </Card>
    </div>
  )
}
