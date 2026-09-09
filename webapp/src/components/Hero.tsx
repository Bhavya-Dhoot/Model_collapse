import { useReducedMotion } from 'motion/react'
import { D } from '@/lib/data'

/* The one ambient layer on the page.
 *
 * Motion budget: this is the single moving background, and there is no canvas
 * and no scroll-linked scrubbing anywhere else -- the rest of the page's motion
 * is entrance-once plus the mechanism, which is data-driven. The clip sits
 * behind a scrim heavy enough that the heading keeps full contrast; it decorates
 * the title, it never sits under body copy.
 *
 * It must not cost the first paint: preload is metadata only, and the poster
 * colour is painted by CSS so there is something correct on screen immediately.
 * Under prefers-reduced-motion the video is not mounted at all -- the gradient
 * alone carries the hero, which is a designed state rather than a fallback. */

export function Hero() {
  const reduced = useReducedMotion()

  return (
    <header className="relative isolate -mx-6 overflow-hidden px-6 pb-16 pt-20 sm:pb-20 sm:pt-28">
      {/* ambient layer */}
      <div className="absolute inset-0 -z-10 bg-[#0a0a0b]">
        {!reduced && (
          <video
            className="size-full object-cover opacity-55"
            src="/media/loop-hero.mp4"
            autoPlay
            muted
            loop
            playsInline
            preload="metadata"
            aria-hidden="true"
            tabIndex={-1}
          />
        )}
        {/* scrim: keeps the heading at full contrast over a moving image */}
        <div className="absolute inset-0 bg-gradient-to-r from-[#0a0a0b] via-[#0a0a0b]/85 to-[#0a0a0b]/45" />
        <div className="absolute inset-x-0 bottom-0 h-32 bg-gradient-to-t from-background to-transparent" />
      </div>

      <p className="font-mono text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--chart-1)]">
        Model collapse · tabular generative models
      </p>
      <h1 className="mt-4 max-w-[19ch] text-balance text-4xl font-semibold tracking-tight text-white sm:text-5xl lg:text-6xl">
        How much real data keeps a generative model from eating itself?
      </h1>
      <p className="mt-6 max-w-[62ch] text-lg leading-relaxed text-white/70">
        Tabular synthesizers are increasingly retrained on data an earlier synthesizer produced.
        That closed loop degrades the model. Keeping a fraction α of genuinely real rows in every
        training set arrests it — and this page runs that loop live.
      </p>

      <div className="mt-8 flex flex-wrap items-center gap-x-6 gap-y-2 text-[13.5px] text-white/55">
        <span>Bhavya Dhoot · Yashee Hinger · Karthik G. M.</span>
        <span className="hidden sm:inline text-white/25">|</span>
        <span>Vellore Institute of Technology</span>
        <span className="hidden sm:inline text-white/25">|</span>
        <span>{D.meta.rows_total.toLocaleString()} result rows, recomputed on load</span>
      </div>

      <a
        href="#mechanism"
        className="mt-9 inline-flex items-center gap-2 rounded-lg bg-white px-4 py-2.5 text-[14px] font-semibold text-[#0a0a0b] transition-opacity hover:opacity-90 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
      >
        See the loop run
        <span aria-hidden>↓</span>
      </a>
    </header>
  )
}
