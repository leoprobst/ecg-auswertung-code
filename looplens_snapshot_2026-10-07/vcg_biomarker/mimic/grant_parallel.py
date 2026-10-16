"""Order-preserving process map under the live resource grant (audit R15).

Several stages computed independent blocks one after another on a single
core -- the gap models' hyperparameter grid (2.15 h), the repeatability
blocks and the reference-range blocks -- while the rest of the machine idled.
BLAS stays at one thread per process on purpose: its summation order depends
on the thread count, and a result that changed with the machine's load would
break the project's bitwise reproducibility. Parallelism therefore comes from
processes, never from BLAS threads (audit R11).

Guarantees:

* results come back in input order, so concatenation and tie-breaking are
  unchanged;
* each worker runs with one BLAS thread, exactly like the job process;
* unmanaged calls (tests, the single-stage CLI) stay serial unless
  ``VCG_GRANT_PARALLEL=1``; ``0`` disables processes everywhere.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Sequence

PARALLEL_ENV = "VCG_GRANT_PARALLEL"


def grant_workers(task_count: int) -> int:
    """Worker processes for ``task_count`` independent tasks."""
    from .resource_grants import GRANT_ENV, granted_workers, worker_ceiling

    setting = os.environ.get(PARALLEL_ENV, "").strip()
    if setting == "0" or task_count <= 1:
        return 1
    if not os.environ.get(GRANT_ENV):
        if setting != "1":
            return 1
        return max(1, min(int(task_count), (os.cpu_count() or 2) - 1))
    return max(1, min(int(task_count), granted_workers(worker_ceiling(1))))


def grant_parallel_map(
    function: Callable[..., Any],
    argument_list: Sequence[tuple],
    *,
    parallel: bool = True,
) -> list:
    """``[function(*arguments) for arguments in argument_list]``, maybe parallel."""
    workers = grant_workers(len(argument_list)) if parallel else 1
    if workers <= 1:
        return [function(*arguments) for arguments in argument_list]
    from joblib import Parallel, delayed, parallel_config

    # loky would otherwise give each worker cpu_count // n_jobs BLAS threads.
    with parallel_config(backend="loky", inner_max_num_threads=1):
        return list(
            Parallel(n_jobs=workers)(
                delayed(function)(*arguments) for arguments in argument_list
            )
        )
