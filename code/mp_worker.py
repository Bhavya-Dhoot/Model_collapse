"""Process-pool worker entry point for run_experiment.py's --workers > 1.

This module's job is narrow but load-bearing: set OMP_NUM_THREADS /
MKL_NUM_THREADS / OPENBLAS_NUM_THREADS / NUMEXPR_NUM_THREADS BEFORE numpy,
scipy, sklearn, or torch are ever imported in the child process. Those
libraries read thread-count env vars once, at their own import time --
setting them after the fact (e.g. inside the submitted function, after
`import run_experiment` has already pulled in numpy) does nothing.

On Windows (spawn is the only start method), each ProcessPoolExecutor
worker re-executes this module's top level from scratch to resolve the
target function, so keeping this file free of any (transitive) numpy
import at module scope -- and doing the heavy `import run_experiment`
lazily, inside the function body -- is what actually prevents e.g. 8
workers x numpy's-default-all-cores from oversubscribing the machine and
running slower than serial (this was flagged explicitly: ~2 threads per
worker is the target when running --workers N).
"""
import os


def _configure_threads(threads_per_worker):
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(threads_per_worker)


def run_chain_group_worker(kwargs):
    """kwargs must contain 'threads_per_worker' plus everything
    process_chain_group(**kwargs) needs. Runs in its own process; imports
    the heavy libraries only after threads are capped."""
    threads_per_worker = kwargs.pop("threads_per_worker")
    _configure_threads(threads_per_worker)
    import run_experiment  # deferred: numpy/torch/sklearn import happens here, after capping
    return run_experiment.process_chain_group(**kwargs)
