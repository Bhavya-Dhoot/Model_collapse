"""
Overnight watchdog for the sweep. Logs progress, reaps orphaned worker
processes, and flags stalls. Read-only with respect to results/ -- it never
deletes or rewrites experiment data, it only kills dead-parent processes.

    python code/watchdog.py            # runs until the queue is done or killed

ponytail: deliberately a single polling loop, not a supervisor framework. The
only thing it needs to survive is a laptop GPU wedging overnight.
"""
import glob
import os
import subprocess
import time

import pandas as pd

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
LOG = os.path.join(RESULTS, "watchdog.log")
POLL_S = 300           # 5 min
STALL_LIMIT = 4        # 4 polls with no new rows == 20 min == stalled


def rows():
    n = 0
    for f in glob.glob(os.path.join(RESULTS, "*_shards_*", "*.csv")):
        try:
            n += len(pd.read_csv(f))
        except Exception:
            pass
    return n


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def python_procs():
    """[(pid, parent_pid_or_None, cmdline)] for running python processes."""
    ps = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "ForEach-Object { \"$($_.ProcessId)|$($_.CommandLine)\" }"
    )
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return []
    procs = []
    for line in out.splitlines():
        if "|" not in line:
            continue
        pid, _, cmd = line.partition("|")
        try:
            pid = int(pid.strip())
        except ValueError:
            continue
        parent = None
        if "parent_pid=" in cmd:
            try:
                parent = int(cmd.split("parent_pid=")[1].split(",")[0].split(")")[0])
            except Exception:
                parent = None
        procs.append((pid, parent, cmd))
    return procs


def reap_orphans():
    """Kill spawned workers whose parent process no longer exists. This is the
    exact failure that wedged the GPU earlier: on Windows, killing a pool parent
    does not reliably reap its children, and the survivors hold VRAM forever."""
    procs = python_procs()
    live = {p for p, _, _ in procs}
    killed = []
    for pid, parent, _cmd in procs:
        if parent is not None and parent not in live:
            try:
                subprocess.run(["powershell", "-NoProfile", "-Command",
                                f"Stop-Process -Id {pid} -Force"],
                               capture_output=True, timeout=30)
                killed.append(pid)
            except Exception:
                pass
    if killed:
        log(f"REAPED orphaned workers (dead parent): {killed}")
    return killed


def gpu():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30).stdout.strip()
        return out
    except Exception:
        return "n/a"


def main():
    log("watchdog start")
    last, stalls = rows(), 0
    while True:
        time.sleep(POLL_S)
        reap_orphans()
        cur = rows()
        n_run = sum(1 for _, _, c in python_procs() if "run_experiment" in c)
        log(f"rows={cur} (+{cur - last}) run_experiment_procs={n_run} gpu=[{gpu()}]")
        if cur == last:
            stalls += 1
            if stalls >= STALL_LIMIT:
                log(f"*** STALL: no new rows in {STALL_LIMIT * POLL_S // 60} min. "
                    f"procs={n_run}. Orphans reaped above if any. ***")
                stalls = 0
        else:
            stalls = 0
        if cur == last and n_run == 0:
            log("no run_experiment processes and no growth -- queue appears DONE")
            break
        last = cur
    log("watchdog exit")


if __name__ == "__main__":
    main()
