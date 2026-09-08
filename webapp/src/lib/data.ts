import raw from '../data.json'

/* The shape of demo/data.json, produced by code/export_dashboard_data.py from
   results/main.csv using the same estimators as the paper's figure scripts. */

export interface TrajPoint {
  g: number
  seeds: number
  var_ratio: number | null
  w1: number | null
  tv_cat: number | null
  corr_frob: number | null
  cat_support: number | null
  tstr_auc: number | null
  c2st_auc: number | null
  [k: string]: number | null
}

export interface AlphaStarRow {
  dataset: string
  n: number
  var_ratio: number | null
  w1: number | null
  tv_cat: number | null
  corr_frob: number | null
  cat_support: number | null
  tstr_auc: number | null
  c2st_auc: number | null
  spread: number | null
  spread_excl_var: number | null
  ratio_excl_var: number | null
  binding_axis: string
  [k: string]: string | number | null
}

export interface RegimeRow {
  alpha: number
  fixed: number
  fresh: number
  diff: number
  seeds: number
  fixed_se: number
  fresh_se: number
  p?: number
}

export interface ArchRow {
  synth: string
  terminal_generation: number
  a0: Record<string, { start: number; end: number }>
  a1: Record<string, { start: number; end: number }>
}

export interface Dataset {
  meta: {
    rows_total: number
    terminal_generation: number
    seeds_max: number
    primary_seeds: number
    alphas: number[]
    ns: number[]
  }
  traj: Record<string, Record<string, TrajPoint[]>>
  ceiling: Record<string, number>
  alpha_star_table: AlphaStarRow[]
  degradation_curves: Record<string, { points: { alpha: number; deg: number }[]; tol: number; alpha_star: number | null }>
  nscaling: {
    ns: number[]
    beta: number
    beta_ci: [number, number]
    fit_axes: string[]
    axes: Record<string, { eps_abs: number | null; alpha_star: (number | null)[] }>
  }
  regime: RegimeRow[]
  architectures: ArchRow[]
  gen0_gate: { synth: string; cat_support: number; tv_cat: number; tstr_auc: number; sec_per_gen: number }[]
}

export const D = raw as unknown as Dataset

export const AXES = [
  'var_ratio', 'w1', 'tv_cat', 'corr_frob', 'cat_support', 'tstr_auc', 'c2st_auc',
] as const
export type Axis = (typeof AXES)[number]

export const AXIS_LABEL: Record<Axis, string> = {
  var_ratio: 'Variance ratio',
  w1: 'Wasserstein-1',
  tv_cat: 'Total variation',
  corr_frob: 'Correlation structure',
  cat_support: 'Categorical support',
  tstr_auc: 'Utility (TSTR)',
  c2st_auc: 'Detectability (C2ST)',
}

export const AXIS_SHORT: Record<Axis, string> = {
  var_ratio: 'variance',
  w1: 'W₁',
  tv_cat: 'TV',
  corr_frob: 'correlation',
  cat_support: 'support',
  tstr_auc: 'utility',
  c2st_auc: 'C2ST',
}

export const SYNTH_LABEL: Record<string, string> = {
  gaussian_copula: 'Gaussian copula',
  tvae: 'TVAE-lite',
  ctgan: 'CTGAN-lite',
}

export const DATASETS = ['adult', 'credit-g', 'bank-marketing'] as const

/** Fixed hue order, never cycled: an axis keeps its colour across every chart. */
export const AXIS_COLOR: Record<Axis, string> = {
  corr_frob: 'var(--chart-1)',
  cat_support: 'var(--chart-2)',
  w1: 'var(--chart-3)',
  c2st_auc: 'var(--chart-4)',
  tv_cat: 'var(--chart-5)',
  tstr_auc: 'var(--muted-foreground)',
  var_ratio: 'var(--muted-foreground)',
}

export const fmt = (v: number | null | undefined, d = 3) =>
  v === null || v === undefined || !Number.isFinite(v) ? '—' : v.toFixed(d)

export const pct = (v: number, d = 1) => `${(v * 100).toFixed(d)}%`

/** Alpha keys present in the trajectory block, numerically sorted. */
export function alphaKeys(dataset: string): string[] {
  return Object.keys(D.traj[dataset] ?? {}).sort((a, b) => Number(a) - Number(b))
}
