"""Marker evaluation in worker processes (audit E11).

The subject-cluster bootstrap spends most of its time in numpy calls that
keep the GIL (``Generator.multinomial``, fancy indexing). Measured on a
5,000-row endpoint: eight threads ran 1.75x faster than one, four threads
no faster than eight. Worker processes remove that ceiling.

The inputs of one marker chunk (train, validation, endpoint universe and
the two quality arrays) are placed once into a shared-memory block:
numeric numpy columns as raw arrays, every other column, the indexes and
the constants as one pickle. A worker attaches once per chunk and rebuilds
frames with the same columns, order, dtypes and index; each marker then
runs the unchanged ``_evaluate_single_marker``. Seeds depend only on the
marker index, so results equal the thread path.
"""
from __future__ import annotations

import os
import pickle
import secrets
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import get_context, shared_memory
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

_ALIGNMENT = 64
_HEADER_LENGTH_BYTES = 8
_SHARED_KINDS = frozenset("biuf")
PROCESS_ENV = "VCG_VALIDATION_MARKER_PROCESSES"


def process_mode_requested() -> bool:
    """Managed jobs use processes; ``VCG_VALIDATION_MARKER_PROCESSES`` overrides.

    Unmanaged calls (tests, the single-stage CLI) keep threads unless the
    variable is "1"; "0" disables processes everywhere.
    """
    from .resource_grants import GRANT_ENV

    setting = os.environ.get(PROCESS_ENV, "").strip()
    if setting == "0":
        return False
    if setting == "1":
        return True
    return bool(os.environ.get(GRANT_ENV))


def _aligned(offset: int) -> int:
    return (offset + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT


def _is_shareable(values: Any) -> bool:
    dtype = getattr(values, "dtype", None)
    return isinstance(dtype, np.dtype) and dtype.kind in _SHARED_KINDS


def shared_bytes_needed(
    frames: dict[str, Optional[pd.DataFrame]],
    arrays: dict[str, Optional[np.ndarray]],
) -> int:
    """Bytes of the numeric part, for admission before allocating."""
    total = 0
    for frame in frames.values():
        if frame is None:
            continue
        for position in range(frame.shape[1]):
            column = frame.iloc[:, position]
            if _is_shareable(column):
                total += _aligned(column.dtype.itemsize * len(column))
    for array in arrays.values():
        if array is not None and _is_shareable(array):
            total += _aligned(array.dtype.itemsize * len(array))
    return total


class SharedChunk:
    """Owner of one chunk's inputs in shared memory (main process)."""

    def __init__(
        self,
        *,
        frames: dict[str, Optional[pd.DataFrame]],
        arrays: dict[str, Optional[np.ndarray]],
        constants: dict[str, Any],
    ) -> None:
        self.token = secrets.token_hex(8)
        placements: list[tuple[int, np.ndarray]] = []
        cursor = 0

        def place(values: np.ndarray) -> tuple[int, str, int]:
            nonlocal cursor
            contiguous = np.ascontiguousarray(values)
            offset = cursor
            placements.append((offset, contiguous))
            cursor = _aligned(cursor + contiguous.nbytes)
            return offset, contiguous.dtype.str, len(contiguous)

        frame_specs: dict[str, Optional[dict]] = {}
        for key, frame in frames.items():
            if frame is None:
                frame_specs[key] = None
                continue
            if not frame.columns.is_unique:
                raise ValueError("Shared marker frames need unique columns")
            shared: dict[Any, tuple[int, str, int]] = {}
            rest_columns = []
            for position, name in enumerate(frame.columns):
                column = frame.iloc[:, position]
                if _is_shareable(column):
                    shared[name] = place(column.to_numpy(copy=False))
                else:
                    rest_columns.append(name)
            frame_specs[key] = {
                "index": frame.index,
                "columns": frame.columns,
                "shared": shared,
                "rest": frame.loc[:, rest_columns],
            }
        array_specs: dict[str, Any] = {}
        for key, array in arrays.items():
            if array is None:
                array_specs[key] = None
            elif _is_shareable(array):
                array_specs[key] = ("shared", place(np.asarray(array)))
            else:
                array_specs[key] = ("pickled", np.asarray(array))
        header = pickle.dumps(
            {
                "token": self.token,
                "frames": frame_specs,
                "arrays": array_specs,
                "constants": constants,
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        data_start = _aligned(_HEADER_LENGTH_BYTES + len(header))
        size = max(1, data_start + cursor)
        self._memory = shared_memory.SharedMemory(create=True, size=size)
        try:
            buffer = self._memory.buf
            buffer[:_HEADER_LENGTH_BYTES] = len(header).to_bytes(
                _HEADER_LENGTH_BYTES, "little"
            )
            buffer[_HEADER_LENGTH_BYTES:_HEADER_LENGTH_BYTES + len(header)] = header
            # Offsets in the header are relative to data_start.
            for offset, values in placements:
                start = data_start + offset
                target = np.ndarray(
                    values.shape, dtype=values.dtype, buffer=buffer, offset=start
                )
                target[...] = values
                del target
        except BaseException:
            self.close()
            raise
        self.name = self._memory.name
        self.data_start = data_start
        self.size = size

    def close(self) -> None:
        memory = getattr(self, "_memory", None)
        if memory is None:
            return
        self._memory = None
        memory.close()
        try:
            memory.unlink()
        except FileNotFoundError:
            pass


# Worker side ----------------------------------------------------------------

_ATTACHED: dict[str, Any] = {}


def _initialize_worker() -> None:
    # The job's main process pins BLAS to one thread; workers match it so
    # that every reduction runs with the same thread count as the thread path.
    from threadpoolctl import threadpool_limits

    _ATTACHED["thread_limits"] = threadpool_limits(limits=1)


def _release_attached() -> None:
    _ATTACHED.pop("payload", None)
    memory = _ATTACHED.pop("memory", None)
    _ATTACHED.pop("token", None)
    if memory is not None:
        try:
            memory.close()
        except BufferError:
            # A view is still referenced; the mapping ends with the process.
            pass


def _attach(name: str, token: str) -> dict[str, Any]:
    if _ATTACHED.get("token") == token:
        return _ATTACHED["payload"]
    _release_attached()
    memory = shared_memory.SharedMemory(name=name)
    buffer = memory.buf
    length = int.from_bytes(buffer[:_HEADER_LENGTH_BYTES], "little")
    header = pickle.loads(
        buffer[_HEADER_LENGTH_BYTES:_HEADER_LENGTH_BYTES + length]
    )
    if header["token"] != token:
        memory.close()
        raise RuntimeError("Shared marker chunk token mismatch")
    data_start = _aligned(_HEADER_LENGTH_BYTES + length)

    def view(spec: tuple[int, str, int]) -> np.ndarray:
        offset, dtype, count = spec
        return np.ndarray(
            (count,), dtype=np.dtype(dtype), buffer=buffer,
            offset=data_start + offset,
        )

    frames: dict[str, Optional[pd.DataFrame]] = {}
    for key, spec in header["frames"].items():
        if spec is None:
            frames[key] = None
            continue
        rest = spec["rest"]
        data = {
            name: (
                view(spec["shared"][name])
                if name in spec["shared"]
                else rest[name]
            )
            for name in spec["columns"]
        }
        frames[key] = pd.DataFrame(
            data, index=spec["index"], columns=spec["columns"], copy=False
        )
    arrays: dict[str, Optional[np.ndarray]] = {}
    for key, spec in header["arrays"].items():
        if spec is None:
            arrays[key] = None
        elif spec[0] == "shared":
            arrays[key] = view(spec[1])
        else:
            arrays[key] = spec[1]
    payload = {
        "frames": frames,
        "arrays": arrays,
        "constants": header["constants"],
    }
    _ATTACHED.update(token=token, memory=memory, payload=payload)
    return payload


def _evaluate_marker_task(
    name: str,
    token: str,
    feature_index: int,
    feature_id: str,
    panel_id: str,
):
    from .validation import _evaluate_single_marker

    payload = _attach(name, token)
    frames, arrays = payload["frames"], payload["arrays"]
    constants = payload["constants"]
    return _evaluate_single_marker(
        endpoint_id=constants["endpoint_id"],
        feature_index=feature_index,
        feature_id=feature_id,
        panel_id=panel_id,
        train=frames["train"],
        validation=frames["validation"],
        train_quality=arrays["train_quality"],
        validation_quality=arrays["validation_quality"],
        endpoint_universe=frames["endpoint_universe"],
        specification=constants["specification"],
        bootstrap_iterations=constants["bootstrap_iterations"],
    )


# Pool -------------------------------------------------------------------------


class MarkerProcessPool:
    """Spawn-safe pool; processes start only up to the highest admission."""

    def __init__(self, max_workers: int) -> None:
        self.max_workers = max(1, int(max_workers))
        self._executor = ProcessPoolExecutor(
            max_workers=self.max_workers,
            mp_context=get_context("spawn"),
            initializer=_initialize_worker,
        )

    def evaluate(
        self,
        chunk: SharedChunk,
        tasks: list[tuple[int, str, str]],
        admitted: Callable[[int], int],
    ) -> list:
        results: list = [None] * len(tasks)
        pending = deque(enumerate(tasks))
        in_flight: dict = {}
        try:
            while pending or in_flight:
                allowed = max(1, min(self.max_workers, int(admitted(len(in_flight)))))
                while pending and len(in_flight) < allowed:
                    position, (feature_index, feature_id, panel_id) = pending.popleft()
                    future = self._executor.submit(
                        _evaluate_marker_task,
                        chunk.name,
                        chunk.token,
                        feature_index,
                        feature_id,
                        panel_id,
                    )
                    in_flight[future] = position
                done, _ = wait(
                    tuple(in_flight), timeout=1.0, return_when=FIRST_COMPLETED
                )
                for future in done:
                    results[in_flight.pop(future)] = future.result()
        except BaseException:
            for future in in_flight:
                future.cancel()
            raise
        return results

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)


def live_admission(worker_count: int) -> Callable[[int], int]:
    """Grant and measured memory decide, re-sampled at most every 2 s."""
    state = {"at": 0.0, "value": 1}

    def admitted(in_flight: int) -> int:
        now = time.monotonic()
        if now - state["at"] >= 2.0:
            from .resources import recommend_dynamic_workers

            state["value"] = recommend_dynamic_workers(
                worker_count,
                sample_interval=0.05,
                current_workers=in_flight,
                reserve_gb=float(os.environ.get("VCG_JOB_OS_RESERVE_GB", "2")),
            )
            state["at"] = now
        return state["value"]

    return admitted
