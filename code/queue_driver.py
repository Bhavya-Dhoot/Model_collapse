"""Persistent, self-advancing driver for the remaining overnight sweep
queue. Designed to run unattended for hours: all state goes to
results/queue_driver.log, no interactive prompts, safe to kill and
re-run (every step is idempotent via run_experiment.py's own
skip-if-present logic), and it never deletes or truncates existing
results (results/ is append-only).

Queue, strict priority order:
  1. n-sweep        -> results/nsweep.csv
  2. credit-g n=500 -> results/tier1_creditg_n500.csv
  3. Tier 2 (ctgan)  -> results/tier2.csv
  4. Tier 3 (tvae)   -> results/tier3.csv

For each step: if a run_experiment.py process already targets that
step's --out (a leftover from an earlier crashed/duplicate attempt),
this driver waits for it instead of launching a second one -- that
exact "--allow-failures kept a crashed run going and it cascaded into
duplicate launches" failure mode is what wedged the GPU earlier.
Otherwise it launches the step directly via subprocess.Popen (a real,
directly-owned child, not another shell wrapper), and after the process
exits -- however it exits -- scans for and kills any orphaned
multiprocessing-fork children whose command line embeds
"parent_pid=<that pid>" (Windows does not reap a pool's spawned children
just because the pool parent died, so OS-level parent/child lookups are
not reliable once the parent is gone; the literal parent_pid in each
child's own command line is durable). Retries a failed step once; two
failures in a row and the step is logged as failed and skipped rather
than blocking everything behind it.

At the end (whether every step succeeded or not): merges every
available results file into results/main.csv (write to a temp path,
then os.replace -- main.csv is never truncated in place), runs
make_figures.py against it, and writes results/COMPLETION_REPORT.txt.
Then stops.
"""
import glob
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

import pandas as pd
import psutil

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(CODE_DIR), "results")
LOG_PATH = os.path.join(RESULTS_DIR, "queue_driver.log")
REPORT_PATH = os.path.join(RESULTS_DIR, "COMPLETION_REPORT.txt")
MAIN_CSV = os.path.join(RESULTS_DIR, "main.csv")
PYTHON = sys.executable

ID_COLS = ["dataset", "synth", "regime", "alpha", "seed", "n", "generation"]

QUEUE = [
    {
        "name": "nsweep",
        "out": os.path.join(RESULTS_DIR, "nsweep.csv"),
        "args": ["--datasets", "adult", "--synths", "gaussian_copula",
                 "--alphas", "0.0", "0.01", "0.02", "0.05", "0.10", "0.20", "0.35", "0.50", "0.75", "1.0",
                 "--seeds", "0", "1", "2", "--generations", "12", "--n", "500", "8000",
                 "--regimes", "fixed", "--metrics", "fast", "--workers", "2", "--allow-failures"],
    },
    {
        "name": "creditg_n500",
        "out": os.path.join(RESULTS_DIR, "tier1_creditg_n500.csv"),
        "args": ["--datasets", "credit-g", "--synths", "gaussian_copula",
                 "--alphas", "0.0", "0.01", "0.02", "0.05", "0.10", "0.20", "0.35", "0.50", "0.75", "1.0",
                 "--seeds", "0", "1", "2", "--generations", "12", "--n", "500",
                 "--regimes", "fixed", "fresh", "--metrics", "fast", "--workers", "2", "--allow-failures"],
    },
    {
        "name": "tier2_ctgan",
        "out": os.path.join(RESULTS_DIR, "tier2.csv"),
        "args": ["--datasets", "adult", "--synths", "ctgan",
                 "--alphas", "0.0", "0.05", "0.15", "0.5", "1.0",
                 "--seeds", "0", "1", "--generations", "8", "--n", "2000",
                 "--regimes", "fixed", "fresh", "--metrics", "fast", "--epochs-gan", "200",
                 "--workers", "1", "--allow-failures"],
    },
    {
        "name": "tier3_tvae",
        "out": os.path.join(RESULTS_DIR, "tier3.csv"),
        "args": ["--datasets", "adult", "--synths", "tvae",
                 "--alphas", "0.0", "0.05", "0.15", "0.5", "1.0",
                 "--seeds", "0", "1", "--generations", "8", "--n", "2000",
                 "--regimes", "fixed", "fresh", "--metrics", "fast", "--epochs-vae", "120",
                 "--workers", "2", "--allow-failures"],
    },
]


def log(msg):
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def find_existing_run(out_path):
    """Is some run_experiment.py process already targeting this exact
    --out file? (leftover from an earlier crashed/duplicate attempt)."""
    needle = out_path.replace("\\", "/")
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = " ".join(p.info["cmdline"] or []).replace("\\", "/")
            if "run_experiment.py" in cmdline and needle in cmdline:
                return p.info["pid"]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def wait_for_pid(pid, poll_s=15):
    while psutil.pid_exists(pid):
        time.sleep(poll_s)


def reap_orphans_of(pid):
    """multiprocessing-fork workers embed 'parent_pid=<pid>' literally in
    their own command line -- durable even after the real parent is gone,
    unlike OS ppid lookups."""
    needle = f"parent_pid={pid}"
    reaped = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = " ".join(p.info["cmdline"] or [])
            if needle in cmdline:
                p.kill()
                reaped.append(p.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if reaped:
        log(f"  reaped {len(reaped)} orphaned child(ren) of dead PID {pid}: {reaped}")
    return reaped


def run_step(step, max_attempts=2):
    name, out_path = step["name"], step["out"]

    existing_pid = find_existing_run(out_path)
    if existing_pid:
        log(f"[{name}] a run_experiment.py process (PID {existing_pid}) already targets {out_path} "
            f"-- waiting for it instead of launching a duplicate")
        wait_for_pid(existing_pid)
        reap_orphans_of(existing_pid)
        log(f"[{name}] pre-existing process finished; proceeding with normal skip-if-present launch "
            f"to pick up anything it didn't finish")

    for attempt in range(1, max_attempts + 1):
        log(f"[{name}] attempt {attempt}/{max_attempts}: launching")
        step_log = os.path.join(RESULTS_DIR, f"{name}_attempt{attempt}.log")
        with open(step_log, "w") as lf:
            proc = subprocess.Popen(
                [PYTHON, "run_experiment.py"] + step["args"] + ["--out", out_path],
                cwd=CODE_DIR, stdout=lf, stderr=subprocess.STDOUT,
            )
            pid = proc.pid
            log(f"[{name}] PID {pid}, logging to {step_log}")
            proc.wait()
            exit_code = proc.returncode
        reap_orphans_of(pid)
        log(f"[{name}] attempt {attempt} exited with code {exit_code}")
        if exit_code == 0:
            log(f"[{name}] SUCCEEDED")
            return True
        log(f"[{name}] attempt {attempt} FAILED (exit {exit_code}) -- see {step_log}")

    log(f"[{name}] FAILED after {max_attempts} attempts -- SKIPPING, moving to next queue item")
    return False


def merge_all_results():
    """Concatenate every available results source into results/main.csv,
    deduplicated on the identifying config tuple (run_ts breaks ties,
    newest wins). Written atomically (temp file + os.replace)."""
    sources = []
    tier1_csv = os.path.join(RESULTS_DIR, "tier1.csv")
    if os.path.exists(tier1_csv):
        sources.append(("tier1.csv (merged)", tier1_csv))
    else:
        shard_dirs = sorted(glob.glob(os.path.join(RESULTS_DIR, "tier1_shards_*")))
        for sd in shard_dirs:
            for shard in glob.glob(os.path.join(sd, "*.csv")):
                sources.append((f"tier1 shard {os.path.basename(shard)}", shard))

    for fname in ["nsweep.csv", "tier1_creditg_n500.csv", "tier2.csv", "tier3.csv"]:
        p = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            sources.append((fname, p))

    frames = []
    for label, path in sources:
        try:
            df = pd.read_csv(path)
            if not df.empty:
                frames.append(df)
        except Exception as e:
            log(f"  WARNING: could not read {label} ({path}): {e}")

    if not frames:
        log("merge_all_results: no source data found at all")
        return None, sources

    merged = pd.concat(frames, ignore_index=True)
    if "run_ts" in merged.columns:
        merged = merged.sort_values("run_ts", na_position="first")
    merged = merged.drop_duplicates(subset=ID_COLS, keep="last")

    tmp_path = MAIN_CSV + f".tmp{os.getpid()}"
    merged.to_csv(tmp_path, index=False)
    os.replace(tmp_path, MAIN_CSV)
    return merged, sources


def write_completion_report(merged, sources, step_results, fatal_error=None):
    lines = []
    lines.append(f"COMPLETION REPORT -- {datetime.now(timezone.utc).isoformat()}")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Queue step results:")
    for name, ok in step_results.items():
        lines.append(f"  {name}: {'OK' if ok else 'FAILED (skipped after retries)'}")
    lines.append("")

    if fatal_error:
        lines.append("FATAL / UNRECOVERABLE ERROR:")
        lines.append(fatal_error)
        lines.append("")

    if merged is None:
        lines.append("No merged data available -- main.csv was not produced.")
        with open(REPORT_PATH, "w") as f:
            f.write("\n".join(lines))
        return

    lines.append(f"Sources merged into main.csv ({len(sources)} files):")
    for label, path in sources:
        lines.append(f"  {label}: {path}")
    lines.append("")
    lines.append(f"TOTAL ROWS in main.csv: {len(merged)}")
    lines.append("")

    lines.append("Per (dataset, synth, n, regime): row count and distinct alpha count:")
    if all(c in merged.columns for c in ["dataset", "synth", "n", "regime", "alpha"]):
        grp = merged.groupby(["dataset", "synth", "n", "regime"]).agg(
            rows=("alpha", "size"), n_alphas=("alpha", "nunique")
        ).reset_index()
        lines.append(grp.to_string(index=False))
    lines.append("")

    if "status" in merged.columns:
        failed = merged[merged["status"] == "failed"]
        lines.append(f"Rows with status=failed: {len(failed)}")
        if len(failed):
            cols = [c for c in ["dataset", "synth", "regime", "alpha", "seed", "n", "generation", "error"]
                    if c in failed.columns]
            lines.append(failed[cols].to_string(index=False))
    lines.append("")

    if "sampled_with_replacement" in merged.columns:
        any_replacement = merged["sampled_with_replacement"].fillna(False).astype(bool).any()
        lines.append(f"sampled_with_replacement is False everywhere: {not any_replacement}")
        if any_replacement:
            rep_rows = merged[merged["sampled_with_replacement"].fillna(False).astype(bool)]
            cols = [c for c in ["dataset", "synth", "n", "alpha", "seed", "generation", "effective_pool_size"]
                    if c in rep_rows.columns]
            summary = rep_rows[cols].drop_duplicates(subset=["dataset", "n"]) if "dataset" in rep_rows.columns else rep_rows
            lines.append("Datasets/n with replacement sampling detected (first occurrence each):")
            lines.append(summary.to_string(index=False))
    else:
        lines.append("sampled_with_replacement column not present in some source rows (older pilot data).")
    lines.append("")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))
    log(f"wrote {REPORT_PATH}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log("=== queue_driver starting ===")
    step_results = {}
    try:
        for step in QUEUE:
            step_results[step["name"]] = run_step(step)

        log("all queue steps attempted; merging results")
        merged, sources = merge_all_results()

        log("running make_figures.py")
        try:
            fig_log = os.path.join(RESULTS_DIR, "make_figures.log")
            with open(fig_log, "w") as lf:
                subprocess.run(
                    [PYTHON, "make_figures.py", "--csv", MAIN_CSV, "--nscale-csv", MAIN_CSV],
                    cwd=CODE_DIR, stdout=lf, stderr=subprocess.STDOUT, timeout=600,
                )
        except Exception as e:
            log(f"make_figures.py raised: {e}")

        write_completion_report(merged, sources, step_results)
        log("=== queue_driver DONE ===")
    except Exception as e:
        tb = traceback.format_exc()
        log(f"UNRECOVERABLE ERROR in queue_driver: {e}\n{tb}")
        try:
            merged, sources = merge_all_results()
        except Exception:
            merged, sources = None, []
        write_completion_report(merged, sources, step_results, fatal_error=f"{e}\n{tb}")


if __name__ == "__main__":
    main()
