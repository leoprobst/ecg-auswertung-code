from concurrent.futures import Future, ThreadPoolExecutor

import pytest

from vcg_biomarker.mimic.bounded_executor import bounded_ordered_map
from vcg_biomarker.mimic.bounded_executor import adaptive_ordered_map


def test_adaptive_admission_changes_between_chunks():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock
    import time
    lock = Lock()
    counts = {"active": 0, "peak": 0}
    observed = []
    limits = iter((1, 3, 2))
    def capacity():
        if counts["peak"]:
            observed.append(counts["peak"])
        counts["peak"] = 0
        return next(limits)
    def work(value):
        with lock:
            counts["active"] += 1
            counts["peak"] = max(counts["peak"], counts["active"])
        try:
            time.sleep(.01)
            return value * 2
        finally:
            with lock:
                counts["active"] -= 1
    with ThreadPoolExecutor(max_workers=5) as executor:
        actual = list(adaptive_ordered_map(executor, work, range(18), max_workers=5,
                      batch_size=6, capacity_provider=capacity))
    observed.append(counts["peak"])
    assert observed == [1, 3, 2]
    assert actual == [value * 2 for value in range(18)]


def test_bounded_submission_does_not_consume_all_input():
    produced = []
    def values():
        for value in range(100):
            produced.append(value)
            yield value
    with ThreadPoolExecutor(max_workers=2) as pool:
        stream = bounded_ordered_map(pool, lambda x: x * 2, values(), max_pending=4)
        assert next(stream) == 0
        assert produced == list(range(4))
        assert next(stream) == 2
        assert produced == list(range(5))
        assert list(stream) == list(range(4, 200, 2))


def test_serial_parallel_order_and_values_identical():
    values = list(range(123))
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(bounded_ordered_map(pool, lambda x: x*x, values, max_pending=8)) == [x*x for x in values]


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_bound_rejected(limit):
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(ValueError, match="positive integer"):
            list(bounded_ordered_map(pool, abs, [], max_pending=limit))


def test_worker_exception_propagates_without_consuming_remainder():
    produced = []
    def values():
        for value in range(100):
            produced.append(value)
            yield value
    def fail(value):
        if value == 0:
            raise ValueError("synthetic worker failure")
        return value
    with ThreadPoolExecutor(max_workers=2) as pool:
        with pytest.raises(ValueError, match="synthetic worker failure"):
            list(bounded_ordered_map(pool, fail, values(), max_pending=4))
    assert len(produced) == 4


def test_close_cancels_queued_futures():
    class ControlledExecutor:
        def __init__(self):
            self.futures = []
        def submit(self, function, value):
            future = Future()
            if not self.futures:
                future.set_result(function(value))
            self.futures.append(future)
            return future
    executor = ControlledExecutor()
    stream = bounded_ordered_map(executor, abs, range(20), max_pending=3)
    assert next(stream) == 0
    stream.close()
    assert all(f.cancelled() for f in executor.futures[1:])


def test_empty_source_returns_no_results():
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert list(bounded_ordered_map(pool, abs, [], max_pending=2)) == []
