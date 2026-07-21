"""Self-consuming (recursive) training-loop experiment driver.

Generation 0: train G_0 on n real rows sampled from D_real.
Generation t>=1: build an n-row training set = alpha*n real rows (anchor)
+ (1-alpha)*n rows sampled from G_{t-1}; train G_t from scratch on it.
Anchoring regime: 'fixed' reuses the same anchor subset every generation;
'fresh' draws a new alpha*n real rows each generation.
Every generation, G_t's n-row sample is scored against the (fixed,
disjoint) real holdout H.

Determinism: every stochastic draw (real-row subsampling, model init /
minibatch order, evaluation sampling) is derived from a CRC32 hash of a
descriptive string tuple (see combine_seed) rather than Python's hash()
(which is randomized per-process), so re-running the same config tuple
reproduces the same numbers.

Parallelism (--workers > 1): the unit of work is a "chain group" --
one (dataset, synth, seed) tuple, covering every requested alpha for it
sequentially in a single process (see process_chain_group). This is the
right granularity because generation 0 does not depend on alpha (it's
fit once and reused across every alpha in the group, same as the
single-process path), so splitting work any finer than "one group" would
mean recomputing gen0 redundantly in different processes. Groups
themselves are fully independent and run in a ProcessPoolExecutor, each
writing to its own shard CSV (simpler and safer than locking a shared
file), merged into --out when all groups finish.

Skip logic: rows already present in the target file (--out, or a shard)
are not recomputed, so a crashed or interrupted run resumes where it
left off.

Failure handling: if a synthesizer's fit/sample/eval raises, that config
is recorded as an explicit FAILED row (status='failed', error=<message>)
rather than silently vanishing from the CSV -- and the whole run exits
nonzero unless --allow-failures is passed, so a failure can't get lost in
a long sweep's output.
"""
import argparse
import csv
import glob
import json
import os
import platform
import sys
import time
import traceback
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch

import metrics
from data import load_dataset
from data import MAX_ROWS as DATA_DEFAULT_MAX_ROWS
from gpu_utils import (device_info, free_memory, peak_memory_mb,
                        reset_peak_memory, resolve_device)
from synth import SYNTH_NAMES, get_synth

CSV_FIELDS = [
    "dataset", "synth", "regime", "alpha", "seed", "generation",
    "n", "wallclock_sec", "peak_mem_mb", "gen0_shared",
    "mean_shift", "var_ratio", "w1", "tv_cat", "corr_frob", "cat_support",
    "coverage", "c2st_auc", "tstr_auc", "trtr_auc", "n_nan_metrics",
    "ctgan_lost_support", "status", "error", "run_ts",
    "sampled_with_replacement", "effective_pool_size",
]

ID_COLS = ["dataset", "synth", "regime", "alpha", "seed", "n", "generation"]


def combine_seed(*parts):
    """Deterministic (process-independent) derived seed, unlike Python's
    hash() which is salted per-process for strings."""
    s = "|".join(str(p) for p in parts)
    return zlib.crc32(s.encode()) % (2 ** 31 - 1)


def load_existing_keys(path):
    """Key tuple must include `n`: gen 0 (and therefore every generation
    downstream) is a genuinely different model per n, so two different n
    values sharing one output file must not be mistaken for duplicates of
    each other on (dataset,synth,regime,alpha,seed,generation) alone."""
    keys = set()
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            for _, r in df.iterrows():
                keys.add((str(r["dataset"]), str(r["synth"]), str(r["regime"]),
                           float(r["alpha"]), int(r["seed"]), int(r["n"]), int(r["generation"])))
        except Exception as e:
            print(f"[run_experiment] warning: could not read existing {path} ({e}); starting fresh")
    return keys


def append_row(path, row):
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def merge_shards(shard_files, out_path):
    """Concatenate shard CSVs (+ any pre-existing --out) into one file,
    de-duplicated on the identifying config tuple. Ties are broken by
    `run_ts` (the newest run wins, not whichever frame happened to be
    concatenated last) so re-running a config never silently reverts to
    stale numbers depending on shard iteration order. Written atomically
    (temp file + os.replace) so a crash mid-write can never leave out_path
    truncated or half-written -- results/ is append-only, and "corrupt the
    only copy while merging" would defeat that."""
    frames = []
    if os.path.exists(out_path):
        frames.append(pd.read_csv(out_path))
    for f in shard_files:
        if os.path.exists(f) and os.path.getsize(f) > 0:
            frames.append(pd.read_csv(f))
    if not frames:
        return 0
    merged = pd.concat(frames, ignore_index=True)
    if "run_ts" in merged.columns:
        merged = merged.sort_values("run_ts", na_position="first")
    merged = merged.drop_duplicates(subset=ID_COLS, keep="last")
    tmp_path = out_path + f".tmp{os.getpid()}"
    merged.to_csv(tmp_path, index=False)
    os.replace(tmp_path, out_path)
    return len(merged)


def get_anchor(df_pool, regime, alpha, n, dataset, seed, gen, anchor_cache):
    n_anchor = int(round(alpha * n))
    if n_anchor == 0:
        return df_pool.iloc[0:0].reset_index(drop=True)
    replace = n_anchor > len(df_pool)
    if regime == "fixed":
        key = (dataset, alpha, seed)
        if key not in anchor_cache:
            s = combine_seed(dataset, "fixed_anchor", alpha, seed)
            anchor_cache[key] = df_pool.sample(
                n=n_anchor, random_state=s, replace=replace
            ).reset_index(drop=True)
        return anchor_cache[key]
    else:  # fresh
        s = combine_seed(dataset, "fresh_anchor", alpha, seed, gen)
        return df_pool.sample(n=n_anchor, random_state=s, replace=replace).reset_index(drop=True)


def run_generation_eval(model, holdout, numeric_cols, categorical_cols, target_col, n, eval_seed):
    if hasattr(model, "reseed"):
        model.reseed(eval_seed)
    sample = model.sample(n)
    met = metrics.compute_all_metrics(sample, holdout, numeric_cols, categorical_cols, target_col)
    return sample, met


def build_row(dataset, synth_name, regime, alpha, seed, gen, n, wallclock, peak_mem,
              gen0_shared, met, trtr, ctgan_lost, run_ts, sampled_with_replacement, effective_pool_size):
    row = {
        "dataset": dataset, "synth": synth_name, "regime": regime, "alpha": alpha,
        "seed": seed, "generation": gen, "n": n, "wallclock_sec": round(wallclock, 4),
        "peak_mem_mb": round(peak_mem, 2), "gen0_shared": gen0_shared,
        "trtr_auc": trtr, "ctgan_lost_support": ctgan_lost,
        "status": "ok", "error": "", "run_ts": run_ts,
        "sampled_with_replacement": sampled_with_replacement,
        "effective_pool_size": effective_pool_size,
    }
    for k in ["mean_shift", "var_ratio", "w1", "tv_cat", "corr_frob", "cat_support",
              "coverage", "c2st_auc", "tstr_auc", "n_nan_metrics"]:
        row[k] = met.get(k, np.nan)
    return row


def build_failed_row(dataset, synth_name, regime, alpha, seed, gen, n, exc, run_ts):
    """A synthesizer's fit/sample/eval raised for this config. We record
    this as an explicit row (status='failed', error=<exception>) rather
    than silently skipping it -- a config that just vanishes from the CSV
    is indistinguishable from one nobody has run yet, which is exactly
    the failure mode that makes a long sweep untrustworthy."""
    row = {k: np.nan for k in CSV_FIELDS}
    row.update({
        "dataset": dataset, "synth": synth_name, "regime": regime, "alpha": alpha,
        "seed": seed, "generation": gen, "n": n, "gen0_shared": False,
        "status": "failed", "error": f"{type(exc).__name__}: {exc}", "run_ts": run_ts,
    })
    return row


def process_chain_group(dataset, synth_name, seed, n, alphas, regimes, generations,
                         epochs_vae, epochs_gan, device_arg, copula_impl, metrics_mode,
                         out_path, run_ts, resume_from=None):
    """Process every (regime, alpha) combination for one
    (dataset, synth, seed, n) group. n is part of the group key (not swept
    inside it) because generation 0's training-set SIZE is n -- a
    different n means a genuinely different gen-0 model, so there is
    nothing to share across n values the way there is across alpha/regime
    (neither of which affects gen 0 at all). Self-contained (resolves its
    own device + metrics config), so this is exactly what a worker process
    calls under --workers > 1, and what the default --workers 1 path calls
    once per group in a plain loop.

    `out_path` is where this call's own rows get appended (a per-run shard
    under --workers > 1, so distinct runs never reuse/overwrite each
    other's files). `resume_from` is an optional second file (typically
    the canonical --out, already merged from earlier runs' shards) whose
    keys ALSO count as "already done" -- without this, a freshly-named
    (and therefore empty) shard directory would make every re-run recompute
    everything it had already computed in a prior run, since the new shard
    has no history of its own to skip against.
    Returns (n_rows_written, n_failures)."""
    device = resolve_device(device_arg)
    metrics.set_metric_device(device)
    metrics.set_metrics_mode(metrics_mode)

    # the real-data pool must have at least n rows to draw n distinct rows
    # from; data.py's own default cap (20000) covers every n up to that,
    # only a larger n (e.g. the n-sweep's 32000) needs a bigger pool.
    d = load_dataset(dataset, max_rows=max(n, DATA_DEFAULT_MAX_ROWS))
    df_real, holdout = d["df_real"], d["df_holdout"]
    numeric_cols, categorical_cols = d["numeric_cols"], d["categorical_cols"]
    target_col = d["target_col"]
    synth_cat_cols = categorical_cols + [target_col]

    # pool_size is the number of DISTINCT real rows actually available to
    # draw from -- if it's smaller than the row count a draw asks for
    # (n at gen 0, or the anchor size alpha*n at gen>=1), that draw is a
    # bootstrap resample (sampled with replacement), not n/n_anchor
    # independent observations. Recorded explicitly per row below rather
    # than left implicit, per the methodological bug this fixes: a config
    # whose real "effective n" is smaller than its nominal n must be
    # visible in the raw results, not something a reader has to infer.
    pool_size = len(df_real)
    if pool_size < n:
        print(f"[run_experiment] WARNING: {dataset} real pool has only {pool_size} rows "
              f"but n={n} -- gen-0 (and any anchor larger than {pool_size}) will sample "
              f"WITH REPLACEMENT; effective sample size is {pool_size}, not {n}.")

    existing_keys = load_existing_keys(out_path)
    if resume_from and resume_from != out_path:
        existing_keys |= load_existing_keys(resume_from)
    n_written = 0
    failure_count = 0

    trtr_seed = combine_seed(dataset, seed, n, "trtr_real_sample")
    real_n = df_real.sample(n=min(n, len(df_real)), random_state=trtr_seed).reset_index(drop=True)
    trtr = metrics.trtr_auc(real_n, holdout, numeric_cols, categorical_cols, target_col)

    # ---- generation 0: fit once, shared across every (regime, alpha) in this group ----
    model_seed = combine_seed(dataset, synth_name, seed, n, "gen0_model")
    realsample_seed = combine_seed(dataset, seed, n, "gen0_realsample")
    train_df0 = df_real.sample(
        n=min(n, len(df_real)), random_state=realsample_seed, replace=n > len(df_real)
    ).reset_index(drop=True)

    gen0_failed, gen0_error = False, None
    model0 = wallclock0 = peak0 = met0 = ctgan_lost0 = None
    try:
        reset_peak_memory(device)
        t0 = time.time()
        model0 = get_synth(synth_name, numeric_cols, synth_cat_cols, seed=model_seed, device=device,
                            epochs_vae=epochs_vae, epochs_gan=epochs_gan, copula_impl=copula_impl)
        model0.fit(train_df0)
        eval_seed = combine_seed(dataset, synth_name, seed, n, "gen0_eval")
        _, met0 = run_generation_eval(model0, holdout, numeric_cols, categorical_cols,
                                       target_col, n, eval_seed)
        wallclock0 = time.time() - t0
        peak0 = peak_memory_mb(device)
        ctgan_lost0 = getattr(model0, "_last_lost_support", None)
        print(f"[gen0] {dataset}/{synth_name} n={n} seed={seed}: {wallclock0:.2f}s, "
              f"var_ratio={met0.get('var_ratio'):.3f}, w1={met0.get('w1'):.3f}")
    except Exception as e:
        print(f"[gen0] {dataset}/{synth_name} n={n} seed={seed}: FAILED -- {type(e).__name__}: {e}")
        traceback.print_exc()
        gen0_failed, gen0_error = True, e
        failure_count += 1

    anchor_cache = {}

    for regime in regimes:
        for alpha in alphas:
            cfg0 = (dataset, synth_name, regime, alpha, seed, n, 0)
            if cfg0 not in existing_keys:
                if gen0_failed:
                    row = build_failed_row(dataset, synth_name, regime, alpha, seed, 0, n, gen0_error, run_ts)
                else:
                    row = build_row(dataset, synth_name, regime, alpha, seed, 0, n,
                                     wallclock0, peak0, True, met0, trtr, ctgan_lost0, run_ts,
                                     n > pool_size, pool_size)
                append_row(out_path, row)
                existing_keys.add(cfg0)
                n_written += 1

            if gen0_failed:
                continue  # no valid model to build generation 1..T on

            all_gens_done = all(
                (dataset, synth_name, regime, alpha, seed, n, g) in existing_keys
                for g in range(0, generations + 1)
            )
            if all_gens_done:
                continue

            prev_model, prev_owned = model0, False  # gen0 model is shared -- never free it here

            for gen in range(1, generations + 1):
                cfg = (dataset, synth_name, regime, alpha, seed, n, gen)
                try:
                    anchor_df = get_anchor(df_real, regime, alpha, n, dataset, seed, gen, anchor_cache)
                    n_synth = n - len(anchor_df)
                    if n_synth > 0:
                        sample_seed = combine_seed(dataset, synth_name, seed, n, alpha, regime, gen, "prev_sample")
                        if hasattr(prev_model, "reseed"):
                            prev_model.reseed(sample_seed)
                        synth_part = prev_model.sample(n_synth)
                        train_df = (pd.concat([anchor_df, synth_part], ignore_index=True)
                                    if len(anchor_df) else synth_part)
                    else:
                        train_df = anchor_df

                    reset_peak_memory(device)
                    t0 = time.time()
                    model_seed_t = combine_seed(dataset, synth_name, seed, n, alpha, regime, gen, "model")
                    model = get_synth(synth_name, numeric_cols, synth_cat_cols, seed=model_seed_t, device=device,
                                       epochs_vae=epochs_vae, epochs_gan=epochs_gan, copula_impl=copula_impl)
                    model.fit(train_df)
                    eval_seed = combine_seed(dataset, synth_name, seed, n, alpha, regime, gen, "eval")
                    _, met = run_generation_eval(model, holdout, numeric_cols, categorical_cols,
                                                  target_col, n, eval_seed)
                    wallclock = time.time() - t0
                    peak = peak_memory_mb(device)
                    ctgan_lost = getattr(model, "_last_lost_support", None)
                except Exception as e:
                    print(f"[gen{gen}] {dataset}/{synth_name} n={n} alpha={alpha} regime={regime} "
                          f"seed={seed}: FAILED -- {type(e).__name__}: {e}")
                    traceback.print_exc()
                    if cfg not in existing_keys:
                        row = build_failed_row(dataset, synth_name, regime, alpha, seed, gen, n, e, run_ts)
                        append_row(out_path, row)
                        existing_keys.add(cfg)
                        n_written += 1
                    failure_count += 1
                    if prev_owned:
                        free_memory(device, prev_model)
                    break  # can't continue this alpha branch without a valid model

                if cfg not in existing_keys:
                    n_anchor_t = int(round(alpha * n))  # matches get_anchor's own computation
                    row = build_row(dataset, synth_name, regime, alpha, seed, gen, n,
                                     wallclock, peak, False, met, trtr, ctgan_lost, run_ts,
                                     n_anchor_t > pool_size, pool_size)
                    append_row(out_path, row)
                    existing_keys.add(cfg)
                    n_written += 1
                print(f"[gen{gen}] {dataset}/{synth_name} n={n} alpha={alpha} regime={regime} seed={seed}: "
                      f"{wallclock:.2f}s, var_ratio={met.get('var_ratio'):.3f}, w1={met.get('w1'):.3f}")

                if prev_owned:
                    free_memory(device, prev_model)
                prev_model, prev_owned = model, True

            if prev_owned:
                free_memory(device, prev_model)

    if not gen0_failed:
        free_memory(device, model0)

    return n_written, failure_count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--synths", nargs="+", required=True, choices=SYNTH_NAMES)
    p.add_argument("--alphas", nargs="+", type=float, required=True)
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--generations", type=int, required=True, help="T: runs generations 0..T")
    p.add_argument("--n", nargs="+", type=int, required=True,
                    help="one or more training-set sizes (per generation). Each distinct n is "
                         "its own chain-group (gen 0 depends on n, so it can't be shared across "
                         "n values the way it's shared across alpha/regime).")
    p.add_argument("--regimes", nargs="+", choices=["fixed", "fresh"], required=True,
                    help="one or more anchoring regimes, swept inside each chain-group "
                         "(sharing gen 0, which doesn't depend on regime either).")
    p.add_argument("--epochs-vae", type=int, default=120)
    p.add_argument("--epochs-gan", type=int, default=200)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--limit-config", type=int, default=None,
                    help="cap the number of (dataset,synth,seed) chain-groups processed, for pilots")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--copula-impl", type=str, default="torch", choices=["torch", "numpy"])
    p.add_argument("--metrics", type=str, default="fast", choices=["fast", "full"],
                    help="fast (default): capped/single-split C2ST + capped coverage, for the full sweep. "
                         "full: 5-fold CV, no subsampling -- for spot-checking fast tracks full.")
    p.add_argument("--allow-failures", action="store_true",
                    help="record a FAILED row and continue instead of exiting nonzero when a "
                         "synthesizer's fit/sample/eval raises")
    p.add_argument("--workers", type=int, default=1,
                    help="process-pool workers, each handling one whole (dataset,synth,seed) "
                         "chain-group (all alphas) independently. 1 (default): sequential, in "
                         "this process, writing directly to --out. >1: shards --out into "
                         "per-group files (merged into --out when all groups finish) and runs "
                         "groups across a ProcessPoolExecutor -- see module docstring.")
    args = p.parse_args()

    device = resolve_device(args.device)  # fail fast here, before spawning any workers
    print(f"[run_experiment] device = {device}, metrics = {args.metrics}, workers = {args.workers}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
        },
        "device": device_info(device),
    }
    manifest_path = os.path.splitext(args.out)[0] + "_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[run_experiment] wrote manifest to {manifest_path}")

    groups = []
    for dataset in args.datasets:
        for synth_name in args.synths:
            for seed in args.seeds:
                for n in args.n:
                    groups.append((dataset, synth_name, seed, n))
    if args.limit_config is not None:
        groups = groups[:args.limit_config]
    n_rows_max = len(groups) * len(args.regimes) * len(args.alphas) * (args.generations + 1)
    print(f"[run_experiment] {len(groups)} (dataset,synth,seed,n) chain-groups x {len(args.regimes)} regimes x "
          f"{len(args.alphas)} alphas x {args.generations + 1} generations = {n_rows_max} rows total (before skip)")

    t_start = time.time()
    # One run_ts per invocation: stamped onto every row (tie-breaker for
    # merge_shards so a re-run's rows deterministically win over a stale
    # duplicate) and folded into the shard directory name so a re-run never
    # reuses -- and therefore never risks overwriting -- a previous run's
    # shard files (results/ is append-only; see module docstring).
    run_ts = datetime.now(timezone.utc).isoformat()
    run_ts_safe = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if args.workers <= 1:
        total_written, total_failures = 0, 0
        for (dataset, synth_name, seed, n) in groups:
            n_written, n_fail = process_chain_group(
                dataset, synth_name, seed, n, args.alphas, args.regimes, args.generations,
                args.epochs_vae, args.epochs_gan, args.device, args.copula_impl, args.metrics,
                args.out, run_ts,
            )
            total_written += n_written
            total_failures += n_fail
    else:
        shard_dir = os.path.splitext(args.out)[0] + f"_shards_{run_ts_safe}"
        os.makedirs(shard_dir, exist_ok=True)
        threads_per_worker = 2  # ~2/worker keeps N workers from oversubscribing the box
        import mp_worker

        total_written, total_failures = 0, 0
        shard_paths = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {}
            for (dataset, synth_name, seed, n) in groups:
                shard_path = os.path.join(shard_dir, f"shard__{dataset}__{synth_name}__seed{seed}__n{n}.csv")
                shard_paths.append(shard_path)
                kwargs = dict(
                    threads_per_worker=threads_per_worker, dataset=dataset, synth_name=synth_name,
                    seed=seed, n=n, alphas=args.alphas, regimes=args.regimes, generations=args.generations,
                    epochs_vae=args.epochs_vae, epochs_gan=args.epochs_gan,
                    device_arg=args.device, copula_impl=args.copula_impl, metrics_mode=args.metrics,
                    out_path=shard_path, run_ts=run_ts, resume_from=args.out,
                )
                fut = ex.submit(mp_worker.run_chain_group_worker, kwargs)
                futures[fut] = (dataset, synth_name, seed, n)
            for fut in as_completed(futures):
                dataset, synth_name, seed, n = futures[fut]
                try:
                    n_written, n_fail = fut.result()
                except Exception as e:
                    print(f"[run_experiment] worker process CRASHED for {dataset}/{synth_name}/seed{seed}/n{n}: {e}")
                    n_written, n_fail = 0, 1
                total_written += n_written
                total_failures += n_fail
                print(f"[run_experiment] group done: {dataset}/{synth_name}/seed{seed}/n{n} "
                      f"({n_written} rows written, {n_fail} failures)")

        n_merged = merge_shards(shard_paths, args.out)
        print(f"[run_experiment] merged {len(shard_paths)} shard(s) -> {args.out} ({n_merged} total rows)")

    elapsed = time.time() - t_start
    print(f"[run_experiment] done in {elapsed:.1f}s. {total_failures} failure(s), "
          f"{total_written} row(s) written this run.")
    if total_failures > 0 and not args.allow_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
