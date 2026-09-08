import { motion, useReducedMotion } from 'motion/react'
import type { ReactNode } from 'react'
import { Section, ThemeToggle } from '@/components/Shell'
import { LoopAnimation } from '@/panels/LoopAnimation'
import { Thresholds } from '@/panels/Thresholds'
import { Collapse } from '@/panels/Collapse'
import { Regime } from '@/panels/Regime'
import { Scaling } from '@/panels/Scaling'
import { Architectures } from '@/panels/Architectures'
import { D } from '@/lib/data'

/* One scroll authority: the browser's own scrolling. No smooth-scroll library,
   no scroll-linked scrubbing. Sections rise once as they enter view, from a
   visible resting state -- never parked invisible waiting on an observer. */
function Reveal({ children }: { children: ReactNode }) {
  const reduced = useReducedMotion()
  if (reduced) return <>{children}</>
  return (
    <motion.div
      initial={{ opacity: 0.45, y: 14 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.15 }}
      transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
    >
      {children}
    </motion.div>
  )
}

const NAV = [
  ['mechanism', 'Mechanism'],
  ['collapse', 'Collapse'],
  ['axes', 'Thresholds'],
  ['scaling', 'Scaling'],
  ['regime', 'Fresh vs fixed'],
  ['arch', 'Architectures'],
] as const

export default function App() {
  return (
    <div className="min-h-dvh bg-background text-foreground">
      <a href="#mechanism" className="sr-only focus:not-sr-only focus:absolute focus:left-3 focus:top-3 focus:z-[100] focus:rounded-lg focus:bg-foreground focus:px-3 focus:py-2 focus:text-background">
        Skip to content
      </a>

      <nav className="sticky top-0 z-50 flex items-center gap-1 border-b border-border bg-background/85 px-6 py-2.5 backdrop-blur">
        <span className="mr-auto text-[15px] font-semibold tracking-tight">Anchoring&nbsp;Thresholds</span>
        {NAV.map(([id, label]) => (
          <a key={id} href={`#${id}`}
            className="rounded-md px-2.5 py-1.5 text-[13px] font-medium text-muted-foreground transition-colors hover:bg-muted hover:text-foreground">
            {label}
          </a>
        ))}
        <ThemeToggle />
      </nav>

      <main className="mx-auto max-w-6xl px-6 pb-24">
        <header className="py-14">
          <h1 className="max-w-[19ch] text-balance text-4xl font-semibold tracking-tight sm:text-5xl">
            How much real data keeps a generative model from eating itself?
          </h1>
          <p className="mt-5 max-w-[62ch] text-lg leading-relaxed text-muted-foreground">
            Tabular synthesizers are increasingly retrained on data an earlier synthesizer produced.
            That closed loop degrades the model. Keeping a fraction α of genuinely real rows in every
            training set arrests it — and this page runs that loop live.
          </p>
          <p className="mt-6 text-[13.5px] text-muted-foreground">
            Bhavya Dhoot, Yashee Hinger, Karthik G. M. · Vellore Institute of Technology ·
            every figure recomputed from {D.meta.rows_total.toLocaleString()} released result rows
          </p>
        </header>

        <Section id="mechanism" eyebrow="The mechanism" title="The anchored self-consuming loop"
          lede="The budget is fixed at n rows; α decides its composition, not its size. The fidelity bar is driven by the paper's own recursion, so what you see is the real trajectory.">
          <Reveal><LoopAnimation /></Reveal>
        </Section>

        <Section id="collapse" eyebrow="What really happens" title="Collapse is real, and anchoring arrests it"
          lede="Twelve generations on three OpenML benchmarks. The α=1 control stays flat, so what is measured is self-consumption and not evaluation drift.">
          <Reveal><Collapse /></Reveal>
        </Section>

        <Section id="axes" eyebrow="Thresholds" title="Every axis has its own threshold"
          lede="α★ is the smallest real-data fraction that keeps an axis within tolerance and stays there for every larger α.">
          <Reveal><Thresholds /></Reveal>
        </Section>

        <Section id="scaling" eyebrow="Scaling" title="Bigger datasets need less anchoring — far more slowly than theory promises"
          lede="The closed form predicts α★ ∝ 1/n. Measured across seven budgets the exponent is 0.26, with an interval that excludes 1 by a wide margin.">
          <Reveal><Scaling /></Reveal>
        </Section>

        <Section id="regime" eyebrow="Not all real data is equal" title="A lost category only comes back in a row that contains it"
          lede="Numeric drift is a contraction any real data pulls back. Losing a rare category is closer to an absorbing state — so it matters whether anchor rows are re-used or freshly drawn.">
          <Reveal><Regime /></Reveal>
        </Section>

        <Section id="arch" eyebrow="Does it generalize?" title="Two findings survive the change of architecture"
          lede="The exact theory covers Gaussian fitting only. Qualitatively, all three synthesizers lose categorical support under self-consumption and recover it under anchoring.">
          <Reveal><Architectures /></Reveal>
        </Section>
      </main>

      <footer className="border-t border-border px-6 py-8 text-center text-[13px] text-muted-foreground">
        Companion to <em>Real-Data Anchoring Thresholds for Preventing Model Collapse in Tabular Generative Models</em>.
        Charts built with bklit UI; data from <span className="font-mono">results/main.csv</span>.
      </footer>
    </div>
  )
}
