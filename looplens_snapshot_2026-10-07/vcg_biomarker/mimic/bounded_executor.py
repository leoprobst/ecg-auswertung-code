"""Ordered executor mapping with bounded producer and future memory."""
from __future__ import annotations

from collections import deque
from itertools import islice
import os
from concurrent.futures import Executor, Future
from typing import Callable, Iterable, Iterator, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def adaptive_ordered_map(executor, function, values, *, max_workers,
                         batch_size=64, resident_workers=0, capacity_provider=None):
    """Resample admission between bounded chunks with an idle owned workload.

    Only active task count changes; a process pool's resident memory is not
    reclaimed by this adapter. Closing cancels pending tasks in the inner map.
    """
    if type(max_workers) is not int or max_workers < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError("Positive integer worker and batch limits required")
    from .resources import recommend_dynamic_workers
    provider = capacity_provider or (lambda: recommend_dynamic_workers(
        max_workers, sample_interval=.05, current_workers=resident_workers,
        reserve_gb=float(os.environ.get("VCG_JOB_OS_RESERVE_GB", "6"))))
    source = iter(values)
    while batch := list(islice(source, batch_size)):
        limit = provider()
        if type(limit) is not int or limit < 1:
            raise ValueError("Invalid live task admission")
        stream = bounded_ordered_map(executor, function, batch,
                                     max_pending=min(max_workers, limit))
        try:
            yield from stream
        finally:
            stream.close()


def pipelined_ordered_map(executor, function, values, *, capacity_provider,
                          lookahead_factor=4, resample_seconds=2.0,
                          clock=None):
    """Input-ordered results without a batch barrier (audit R13/E8).

    At most ``capacity_provider()`` tasks are running at once, re-read every
    ``resample_seconds``; finished results wait in input order behind a slow
    one, bounded to ``lookahead_factor`` times the admission, so a slow
    record never idles the pool and the output order stays that of the
    input. Closing the iterator cancels tasks that have not started.
    """
    import time as _time
    from concurrent.futures import FIRST_COMPLETED, wait

    if type(lookahead_factor) is not int or lookahead_factor < 1:
        raise ValueError("lookahead_factor must be a positive integer")
    now = clock or _time.monotonic
    source = iter(values)
    pending: deque[Future] = deque()
    exhausted = False
    limit = 1
    sampled_at = None
    try:
        while True:
            if sampled_at is None or now() - sampled_at >= resample_seconds:
                limit = capacity_provider()
                if type(limit) is not int or limit < 1:
                    raise ValueError("Invalid live task admission")
                sampled_at = now()
            running = sum(1 for future in pending if not future.done())
            while (
                not exhausted
                and running < limit
                and len(pending) < limit * lookahead_factor
            ):
                try:
                    value = next(source)
                except StopIteration:
                    exhausted = True
                    break
                pending.append(executor.submit(function, value))
                running += 1
            if not pending:
                return
            while pending and pending[0].done():
                yield pending.popleft().result()
            if pending and not pending[0].done():
                wait(
                    [future for future in pending if not future.done()],
                    timeout=resample_seconds,
                    return_when=FIRST_COMPLETED,
                )
    finally:
        for future in pending:
            future.cancel()


def bounded_ordered_map(
    executor: Executor, function: Callable[[T], R], values: Iterable[T], *, max_pending: int
) -> Iterator[R]:
    """Keep at most ``max_pending`` submitted, unconsumed results.

    Unlike pre-3.14 Executor.map, this does not eagerly consume all input.
    Ordering and exception delivery follow input order, independent of timing.
    Closing the iterator cancels work that has not started; running tasks still
    belong to the caller's executor and must be joined by its owner.
    """
    if isinstance(max_pending, bool) or not isinstance(max_pending, int) or max_pending < 1:
        raise ValueError("max_pending must be a positive integer")
    source = iter(values)
    pending: deque[Future[R]] = deque()
    try:
        for _ in range(max_pending):
            try:
                value = next(source)
            except StopIteration:
                break
            pending.append(executor.submit(function, value))
        while pending:
            yield pending.popleft().result()
            try:
                value = next(source)
            except StopIteration:
                continue
            pending.append(executor.submit(function, value))
    finally:
        for future in pending:
            future.cancel()
