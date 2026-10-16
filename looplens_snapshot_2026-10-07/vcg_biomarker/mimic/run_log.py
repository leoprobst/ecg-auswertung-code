"""A record of what a long run actually did, in numbers.

A month-long study wrote 31,414 events and 31,133 of them -- 99.1 % -- were
free-text progress lines that overwrite each other and carry no measurements.
``supervisor.log`` was zero bytes. The ``logging`` module was not used anywhere
in this package. When a single query then spent 25.9 hours producing 3,357
rows, the only way to find out what it had been doing was to compare file
modification times.

This module is the answer to that. It appends structured records to
``<run_root>/run.jsonl`` -- one JSON object per line, opened in append mode, so
the file survives the process restarts that a paused-and-resumed study goes
through, and so a crash leaves everything written up to that point.

Four record kinds, each chosen because its absence cost real time:

* ``stage`` -- start, end, duration, rows in and out, and **why** a stage ran
  or was skipped. A run once walked back over eleven stages and nothing said
  which of them recomputed or on what grounds.
* ``query`` -- label, duration, row count, the DuckDB limit in force, resident
  memory before and after. The 25.9-hour query would have been visible after
  five minutes.
* ``sample`` -- resident memory and available memory every 30 seconds while a
  stage runs. A curve shows what memory was climbing towards; a single number
  at the moment of failure does not.
* ``error`` -- the full traceback with the stage and the query label attached.

**Logging must never be able to end a run.** Every write is wrapped: a full
disk, a locked file or a serialisation failure degrades to silence, not to a
lost month of computation. That is also why this writes plain lines rather
than using ``logging`` handlers, whose failure modes are harder to contain.
"""

from __future__ import annotations

import json
import functools
import logging
import os
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional


RUN_LOG_VERSION = "1.4.0"
RUN_LOG_FILENAME = "run.jsonl"
SAMPLE_INTERVAL_SECONDS = 30.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _process_tree_rss_gb() -> Optional[float]:
    """Resident memory of this process and its children.

    The children matter: extraction and validation do their work in a pool, so
    the parent's own figure would understate the run by most of its footprint.
    """

    try:
        import psutil
    except ImportError:
        return None
    try:
        process = psutil.Process(os.getpid())
        total = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                total += child.memory_info().rss
            except Exception:
                continue
        return round(total / 1024**3, 3)
    except Exception:
        return None


def _available_gb() -> Optional[float]:
    try:
        import psutil

        return round(psutil.virtual_memory().available / 1024**3, 3)
    except Exception:
        return None


@dataclass
class RunLog:
    """Append-only structured record for one run."""

    path: Path
    stage: Optional[str] = None
    _sampler: Optional[threading.Thread] = field(default=None, repr=False)
    _stop: Optional[threading.Event] = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    enabled: bool = True
    run_id: str = ""
    job_id: str = ""
    attempt_id: str = ""
    feature_runtime_contract: dict[str, Any] = field(default_factory=dict)
    _phases: dict[str, dict[str, float]] = field(default_factory=dict, repr=False)
    _last_resources: dict = field(default_factory=dict, repr=False)
    _queries: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)
    _query_sequence: int = field(default=0, repr=False)
    write_failures: int = 0

    def resource_sample(self) -> dict[str, Any]:
        """Process-tree CPU/I/O deltas; RSS includes children, not only parent."""
        try:
            import psutil
        except ImportError as exc:
            return {"resource_status": "unavailable", "resource_error": str(exc)}
        try:
            parent = psutil.Process()
            entries = {}
            rss = 0
            for process in [parent, *parent.children(recursive=True)]:
                try:
                    cpu = process.cpu_times()
                    io = process.io_counters()
                    entries[(process.pid, process.create_time())] = (
                        cpu.user + cpu.system, io.read_bytes, io.write_bytes)
                    rss += process.memory_info().rss
                except psutil.Error:
                    continue
            now = time.monotonic()
            previous = self._last_resources
            dt = now - previous.get("at", now)
            old = previous.get("entries", {})
            deltas = [sum(max(0, values[i] - old[key][i]) for key, values in entries.items() if key in old)
                      for i in range(3)]
            self._last_resources = {"at": now, "entries": entries}
            return {"rss_gb": round(rss / 1024**3, 3),
                    "available_gb": _available_gb(), "process_count": len(entries),
                    "tree_cpu_core_percent": 100 * deltas[0] / dt if dt > 0 else None,
                    "tree_read_bytes_per_second": deltas[1] / dt if dt > 0 else None,
                    "tree_write_bytes_per_second": deltas[2] / dt if dt > 0 else None,
                    "sample_interval_seconds": dt,
                    "cpu_scope": "surviving_process_deltas_not_system_cpu"}
        except (OSError, psutil.Error) as exc:
            return {"resource_status": "unavailable", "resource_error": str(exc)}

    def phase_totals(self) -> dict:
        with self._lock:
            return {name: dict(value) for name, value in self._phases.items()}

    def active_queries(self) -> list[dict[str, Any]]:
        """Elapsed wall time only: an active query has no known completion fraction."""
        now = time.monotonic()
        with self._lock:
            return [{"query_id": key, "label": value["label"],
                     "stage": value["stage"],
                     "elapsed_seconds": max(0., now - value["began"])}
                    for key, value in self._queries.items()]

    @classmethod
    def for_run(cls, run_root: Path | str) -> "RunLog":
        root = Path(run_root)
        try:
            root.mkdir(parents=True, exist_ok=True)
        except Exception:
            return cls(path=root / RUN_LOG_FILENAME, enabled=False)
        return cls(path=root / RUN_LOG_FILENAME)

    # ------------------------------------------------------------- writing
    def write(self, kind: str, **fields: Any) -> None:
        """Append one record.  Never raises."""

        if not self.enabled:
            return
        record = {
            "at": _now(),
            "kind": str(kind),
            "log_version": RUN_LOG_VERSION,
            "pid": os.getpid(),
            "run_id": self.run_id,
            "job_id": self.job_id,
            "attempt_id": self.attempt_id,
        }
        if self.stage is not None:
            record.setdefault("stage", self.stage)
        record.update(fields)
        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
        except Exception:
            return
        try:
            with self._lock:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
        except Exception:
            # A full disk or a locked file must not end a month-long run.
            self.write_failures += 1
            if self.write_failures == 1:
                logging.getLogger(__name__).warning("Run telemetry write failed: %s", self.path, exc_info=True)
            return

    # -------------------------------------------------------------- stages
    @contextmanager
    def stage_scope(
        self,
        stage: str,
        *,
        decision: str = "computed",
        reason: str = "",
        rows_in: Optional[int] = None,
    ) -> Iterator["RunLog"]:
        """Bracket one pipeline stage.

        ``decision``/``reason`` are the point of this: "skipped, reuse key
        identical" and "recomputed, source_code_hash differs" are the two
        sentences that explain a walk-back, and neither was recorded anywhere.
        """

        previous = self.stage
        self.stage = str(stage)
        self._phases.clear()
        self._last_resources.clear()
        self.resource_sample()
        began = time.monotonic()
        self.write(
            "stage",
            event="begin",
            decision=str(decision),
            reason=str(reason),
            rows_in=rows_in,
            rss_gb=_process_tree_rss_gb(),
            available_gb=_available_gb(),
        )
        self.start_sampling()
        try:
            yield self
        except BaseException as exc:
            self.exception(exc)
            self.write(
                "stage",
                event="failed",
                duration_seconds=round(time.monotonic() - began, 3),
                rss_gb=_process_tree_rss_gb(),
                available_gb=_available_gb(),
            )
            raise
        else:
            self.write(
                "stage",
                event="end",
                phases=self.phase_totals(),
                duration_seconds=round(time.monotonic() - began, 3),
                rss_gb=_process_tree_rss_gb(),
                available_gb=_available_gb(),
            )
        finally:
            self.stop_sampling()
            self.stage = previous

    # ------------------------------------------------------------- queries
    @contextmanager
    def query_scope(
        self,
        label: str,
        *,
        memory_limit_gb: Optional[float] = None,
        detail: str = "",
    ) -> Iterator[dict]:
        """Bracket one database query.

        Yields a small dict the caller may set ``rows`` on, so the record
        carries how much came back rather than only how long it took.
        """

        began = time.monotonic()
        before = _process_tree_rss_gb()
        outcome: dict[str, Any] = {"rows": None}
        with self._lock:
            self._query_sequence += 1
            query_id = f"{os.getpid()}:{self._query_sequence}"
            self._queries[query_id] = {"label": str(label), "began": began, "stage": self.stage}
        self.write("query", label=str(label), event="begin", query_id=query_id,
                   memory_limit_gb=memory_limit_gb, detail=str(detail), rss_before_gb=before)
        try:
            yield outcome
        except BaseException as exc:
            self.write(
                "query",
                label=str(label),
                event="failed",
                query_id=query_id,
                duration_seconds=round(time.monotonic() - began, 3),
                memory_limit_gb=memory_limit_gb,
                detail=str(detail),
                rss_before_gb=before,
                rss_after_gb=_process_tree_rss_gb(),
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        finally:
            with self._lock:
                self._queries.pop(query_id, None)
        self.write(
            "query",
            label=str(label),
            event="end",
            query_id=query_id,
            duration_seconds=round(time.monotonic() - began, 3),
            rows=outcome.get("rows"),
            memory_limit_gb=memory_limit_gb,
            detail=str(detail),
            rss_before_gb=before,
            rss_after_gb=_process_tree_rss_gb(),
            available_gb=_available_gb(),
        )

    # -------------------------------------------------------------- errors
    def exception(self, exc: BaseException, *, label: str = "") -> None:
        self.write(
            "error",
            label=str(label),
            error=f"{type(exc).__name__}: {exc}",
            traceback="".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ),
        )

    # ------------------------------------------------------------ sampling
    def start_sampling(
        self, interval_seconds: float = SAMPLE_INTERVAL_SECONDS
    ) -> None:
        """Record memory every ``interval_seconds`` until stopped.

        A daemon thread: if the run dies, this must not keep the interpreter
        alive.
        """

        if not self.enabled or self._sampler is not None:
            return
        stop = threading.Event()

        def loop() -> None:
            while not stop.wait(interval_seconds):
                self.write(
                    "sample",
                    **self.resource_sample(),
                    phases=self.phase_totals(),
                    active_queries=self.active_queries(),
                )

        thread = threading.Thread(
            target=loop, name="run-log-sampler", daemon=True
        )
        self._stop = stop
        self._sampler = thread
        try:
            thread.start()
        except Exception:
            self._sampler = None
            self._stop = None

    def stop_sampling(self) -> None:
        if self._stop is not None:
            self._stop.set()
        thread = self._sampler
        self._sampler = None
        self._stop = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)


# ------------------------------------------------------- the active log

# One run per worker process, so a module-level handle is enough and is what
# lets ``StageManifest.matches`` report *why* it reused an artifact without
# threading a log through twenty call sites. A ContextVar would be tidier but
# does not follow into the worker threads the stages start, which is exactly
# where the interesting decisions happen.
_ACTIVE: Optional["RunLog"] = None
_ACTIVE_LOCK = threading.Lock()


def set_active_log(log: Optional["RunLog"]) -> None:
    global _ACTIVE
    with _ACTIVE_LOCK:
        _ACTIVE = log


def active_log() -> Optional["RunLog"]:
    return _ACTIVE


def profiled_phase(name: str):
    """Aggregate hot-call timings; do not write millions of per-marker events."""
    def decorate(function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            log = active_log()
            if log is None:
                return function(*args, **kwargs)
            began, cpu = time.perf_counter(), time.thread_time()
            with log._lock:
                counts = log._phases.setdefault(name, {"calls": 0, "failures": 0,
                    "active": 0, "wall_seconds_sum": 0, "thread_cpu_seconds_sum": 0})
                counts["active"] += 1
            failed = False
            try:
                return function(*args, **kwargs)
            except BaseException:
                failed = True
                raise
            finally:
                with log._lock:
                    counts["calls"] += 1
                    counts["active"] -= 1
                    counts["failures"] += int(failed)
                    counts["wall_seconds_sum"] += time.perf_counter() - began
                    counts["thread_cpu_seconds_sum"] += time.thread_time() - cpu
        return wrapped
    return decorate


@contextmanager
def active(log: Optional["RunLog"]) -> Iterator[Optional["RunLog"]]:
    previous = active_log()
    set_active_log(log)
    try:
        yield log
    finally:
        set_active_log(previous)


@contextmanager
def profile_query(
    label: str,
    *,
    memory_limit_gb: Optional[float] = None,
    detail: str = "",
) -> Iterator[dict[str, Any]]:
    """Profile a production query when a guided run log is active.

    The no-log path is intentionally a near-zero-cost context manager so
    single-stage CLI calls and unit tests do not need a synthetic logger. The
    memory limit is inherited from the admitted stage when callers do not
    provide one explicitly.
    """

    log = active_log()
    if memory_limit_gb is None:
        try:
            memory_limit_gb = float(os.environ["VCG_JOB_MEMORY_GB"])
        except (KeyError, TypeError, ValueError):
            memory_limit_gb = None
    if log is None:
        yield {"rows": None}
        return
    with log.query_scope(
        label,
        memory_limit_gb=memory_limit_gb,
        detail=detail,
    ) as outcome:
        yield outcome


def note_reuse(
    stage: str, *, reused: bool, reason: str, detail: str = ""
) -> None:
    """Record one reuse decision, if anybody is listening.

    A no-op without an active log, so the twenty call sites stay honest in
    tests and in single-stage runs that never opened one.
    """

    log = active_log()
    if log is None:
        return
    log.write(
        "reuse",
        # A pilot calls the generic extract-features/validate service, whose
        # artifact manifest carries that generic name. The run-level stage is
        # nevertheless extract-features-pilot-1000 (or 10000), and collapsing
        # the reuse record back to the service name created a phantom stage
        # with no duration and made pilot throughput impossible to audit.
        decided_stage=str(log.stage or stage),
        artifact_stage=str(stage),
        reused=bool(reused),
        reason=str(reason),
        detail=str(detail),
    )


# ------------------------------------------------------------------ reading


def read_records(path: Path | str) -> list[dict]:
    """Every readable record.  A truncated final line is skipped, not fatal.

    A crash mid-write is exactly when this file matters most, so a partial
    last line must not make the whole log unreadable.
    """

    records: list[dict] = []
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return records
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def summarize(records: list[dict]) -> dict[str, Any]:
    """The view that was rebuilt by hand five times in one day.

    Per stage: how long it ran, whether it reused or recomputed and *why*,
    how much memory it held, and how many rows its queries returned. The
    reuse reason is the part that was missing when eleven finished stages
    were walked back and nobody could say what had moved.
    """

    stages: dict[str, dict[str, Any]] = {}
    # When each stage last started. A stage that has not begun in this pass
    # gets a timestamp beyond any record, so its earlier errors are history.
    last_begin: dict[str, str] = {}
    for record in records:
        if record.get("kind") == "stage" and record.get("event") == "begin":
            last_begin[str(record.get("stage", ""))] = str(record.get("at", ""))

    def _absorb_peak(entry: dict[str, Any], value: Any) -> None:
        if value is None:
            return
        current = entry["peak_rss_gb"]
        entry["peak_rss_gb"] = (
            float(value) if current is None else max(float(current), float(value))
        )

    def _absorb_low(entry: dict[str, Any], value: Any) -> None:
        if value is None:
            return
        current = entry["lowest_available_gb"]
        entry["lowest_available_gb"] = (
            float(value) if current is None else min(float(current), float(value))
        )

    def entry_for(name: str) -> dict[str, Any]:
        return stages.setdefault(
            name,
            {
                "stage": name,
                "decision": None,
                "reason": None,
                "duration_seconds": None,
                "peak_rss_gb": None,
                "rows": None,
                "peak_rss_gb": None,
                "lowest_available_gb": None,
                "state": "running",
                "passes": 0,
                "position": None,
            },
        )

    for record in records:
        kind = record.get("kind")
        name = str(record.get("stage", ""))
        if kind == "stage":
            entry = entry_for(name)
            event = record.get("event")
            if event == "begin":
                # Only one stage runs at a time here, so a stage still marked
                # running when another starts did not finish -- it was killed,
                # and a killed process cannot write its own end record. Saying
                # "running" for it would be reading the gap as progress.
                for other in stages.values():
                    if other is not entry and other.get("state") == "running":
                        other["state"] = "abgebrochen"
                # A new pass wipes the previous one. The file spans every
                # attempt a run ever made -- a stage recomputed after a fix,
                # a stage tried by hand before the run resumed -- and merging
                # them showed a stage as "failed" hours after it had been
                # repaired, and as "done, 171 min" while it was running.
                # Records are chronological, so the last begin wins.
                passes = int(entry.get("passes") or 0) + 1
                entry.update(
                    decision=record.get("decision"),
                    reason=record.get("reason"),
                    duration_seconds=None,
                    peak_rss_gb=None,
                    lowest_available_gb=None,
                    rows=None,
                    state="running",
                    passes=passes,
                )
            # The envelope carries its own memory reading. Without this a
            # stage shorter than the sampling interval -- therapy-response
            # now finishes in six seconds -- reports no memory at all.
            _absorb_peak(entry, record.get("rss_gb"))
            _absorb_low(entry, record.get("available_gb"))
            if event == "skipped":
                entry["decision"] = record.get("decision") or "skipped"
                entry["reason"] = record.get("reason")
                entry["state"] = "skipped"
            elif event in ("end", "failed"):
                entry["duration_seconds"] = record.get("duration_seconds")
                entry["state"] = "done" if event == "end" else "failed"
            if record.get("position") is not None:
                entry["position"] = int(record["position"])
        elif kind == "reuse":
            # Key on the *stage that was running*, never on the manifest's own
            # name. A manifest may call itself something else -- the
            # ``ingest-events`` stage checks a manifest named
            # ``ingest-clinical-events`` -- and keying on that invented two
            # stage rows that no run ever had, which then showed up as
            # "abgebrochen" once abandoned stages were detected.
            #
            # The manifest name is still worth showing: it says which artifact
            # was examined.
            decided = str(record.get("decided_stage", "") or "")
            entry = entry_for(name or decided)
            entry["decision"] = (
                "wiederverwendet" if record.get("reused") else "neu gerechnet"
            )
            reason = str(record.get("reason") or "")
            if decided and decided != name:
                reason = f"{reason}; Artefakt {decided}" if reason else (
                    f"Artefakt {decided}"
                )
            entry["reason"] = reason
        elif kind == "sample" and name:
            _absorb_peak(entry_for(name), record.get("rss_gb"))
            _absorb_low(entry_for(name), record.get("available_gb"))
        elif kind == "query" and name:
            _absorb_peak(entry_for(name), record.get("rss_after_gb"))
            _absorb_low(entry_for(name), record.get("available_gb"))
        if (
            kind in ("query", "stage")
            and name
            and record.get("rows") is not None
        ):
            entry = entry_for(name)
            entry["rows"] = int(entry["rows"] or 0) + int(record["rows"])

    queries = [
        item
        for item in records
        if item.get("kind") == "query" and item.get("duration_seconds")
    ]
    queries.sort(
        key=lambda item: float(item.get("duration_seconds") or 0.0),
        reverse=True,
    )
    # Aggregated from the per-stage figures, which are already scoped to each
    # stage's latest pass. Taking them from every record instead reported the
    # 14.4 GB peak of a hand-run abandoned hours earlier as the run's memory
    # high-water mark, while every line above it said otherwise.
    samples = [
        entry["peak_rss_gb"]
        for entry in stages.values()
        if entry.get("peak_rss_gb") is not None
    ]
    available = [
        entry["lowest_available_gb"]
        for entry in stages.values()
        if entry.get("lowest_available_gb") is not None
    ]
    # Measurements taken outside any stage envelope belong to no pass and can
    # therefore not be superseded by one. Dropping them would lose the sampler
    # of a run that opened no stage at all.
    samples += [
        float(item["rss_gb"])
        for item in records
        if not str(item.get("stage", ""))
        and item.get("rss_gb") is not None
    ]
    available += [
        float(item["available_gb"])
        for item in records
        if not str(item.get("stage", ""))
        and item.get("available_gb") is not None
    ]
    # Pipeline order, not the order names first appeared. A stage run by hand
    # before the study resumed was logged first and therefore sat above step
    # one, which reads as the run having started in the wrong place. Stages
    # the run has not reached carry no position and keep their first-seen
    # order, after the ones that do.
    ordered_stages = sorted(
        stages.values(),
        key=lambda entry: (
            entry.get("position") is None,
            entry.get("position") or 0,
        ),
    )
    return {
        "stages": ordered_stages,
        "slowest_queries": queries[:10],
        "peak_rss_gb": max(samples) if samples else None,
        "lowest_available_gb": min(available) if available else None,
        # An error belongs to the pass that is current only if it happened
        # after that stage's latest begin. The file keeps every attempt --
        # including stages tried by hand before a run resumed -- and a binder
        # error from the previous evening read as a live failure of a stage
        # the run had not yet reached.
        "errors": [
            item
            for item in records
            if item.get("kind") == "error"
            and str(item.get("at", ""))
            >= last_begin.get(str(item.get("stage", "")), "")
        ],
        "earlier_errors": [
            item
            for item in records
            if item.get("kind") == "error"
            and str(item.get("at", ""))
            < last_begin.get(str(item.get("stage", "")), "")
        ],
        "discarded": [
            item
            for item in records
            if item.get("kind") == "workflow"
            and item.get("event") == "completed_list_discarded"
        ],
        "record_count": len(records),
    }


def _short_error(text: object, limit: int = 96) -> str:
    """The first line of an error, abbreviated visibly.

    A hard slice cut ``subject_id`` to ``subject_i`` with nothing to show it
    had been cut, which reads as broken output rather than as an excerpt.
    DuckDB errors also carry a candidate-binding list and a source line below
    the message; the first line is the part worth putting in a table.
    """

    first = str(text or "").strip().splitlines()
    if not first:
        return ""
    line = first[0].strip()
    if len(line) <= limit:
        return line
    return line[: limit - 1].rstrip() + "…"


def _gb(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{float(value):.1f} GB"
    try:
        return f"{float(str(value)):.1f} GB"
    except ValueError:
        return "-"


def render_summary(summary: Mapping[str, Any]) -> str:
    """Plain text for a terminal."""

    lines: list[str] = []
    lines.append(
        f"{'Stufe':32s} {'Dauer':>9s} {'Spitze':>8s} {'Zeilen':>10s}  "
        f"{'Zustand':11s}  Entscheidung"
    )
    for entry in summary.get("stages", []):
        duration = entry.get("duration_seconds")
        shown = f"{float(duration) / 60:.1f} min" if duration else "-"
        peak = entry.get("peak_rss_gb")
        peak_shown = f"{float(peak):.1f} GB" if peak is not None else "-"
        rows = entry.get("rows")
        rows_shown = f"{int(rows):,}" if rows is not None else "-"
        reason = entry.get("reason") or ""
        decision = entry.get("decision") or ""
        suffix = f" ({reason})" if reason else ""
        passes = int(entry.get("passes") or 0)
        if passes > 1:
            # Otherwise a stage that was repaired and recomputed looks
            # identical to one that ran once.
            suffix += f" [{passes}. Durchlauf]"
        lines.append(
            f"{str(entry.get('stage', '')):32s} {shown:>9s} {peak_shown:>8s} "
            f"{rows_shown:>10s}  {str(entry.get('state', '')):11s}  "
            f"{decision}{suffix}"
        )
    peak = summary.get("peak_rss_gb")
    low = summary.get("lowest_available_gb")
    lines.append("")
    lines.append(
        f"Speicherspitze: {_gb(peak)}   "
        f"niedrigster freier Speicher: {_gb(low)}"
    )
    for item in summary.get("discarded") or []:
        lines.append("")
        lines.append(
            f"Erledigte Stufen verworfen ({item.get('discarded_stage_count', '?')}): "
            f"{item.get('reason', '')}"
        )
    slowest = summary.get("slowest_queries") or []
    if slowest:
        lines.append("")
        lines.append("Langsamste Abfragen:")
        for item in slowest[:5]:
            seconds = float(item.get("duration_seconds") or 0.0)
            lines.append(
                f"  {seconds / 60:8.1f} min  {item.get('label', '')}"
                f"  ({item.get('rows', '?')} Zeilen)"
            )
    errors = summary.get("errors") or []
    if errors:
        lines.append("")
        lines.append(f"Fehler im laufenden Anlauf: {len(errors)}")
        for item in errors[:3]:
            lines.append(
                f"  {item.get('stage', '')}: {_short_error(item.get('error'))}"
            )
        lines.append("  (vollstaendig mit --errors)")
    earlier = summary.get("earlier_errors") or []
    if earlier:
        lines.append("")
        lines.append(
            "Aus einem frueheren Anlauf, vor dem letzten Start "
            "dieser Stufe: "
            f"{len(earlier)}"
        )
        for item in earlier[:3]:
            lines.append(
                f"  {str(item.get('at', ''))[:19]} "
                f"{item.get('stage', '')}: "
                f"{_short_error(item.get('error'))}"
            )
    return "\n".join(lines)


__all__ = [
    "RUN_LOG_FILENAME",
    "RUN_LOG_VERSION",
    "SAMPLE_INTERVAL_SECONDS",
    "RunLog",
    "active",
    "active_log",
    "note_reuse",
    "profile_query",
    "read_records",
    "render_summary",
    "set_active_log",
    "summarize",
]
