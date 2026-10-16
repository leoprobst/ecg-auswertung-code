from __future__ import annotations

import os
import platform
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

import numpy as np

from .config import MimicProjectConfig
from .storage import atomic_write_json


BENCHMARK_VERSION = "1.1.0"
MEMORY_SAFETY_POLICY_VERSION = "1.2.0"
_THREAD_LIMITS_APPLIED = False
_THREADPOOL_LIMITER = None


@dataclass(frozen=True)
class HardwareInventory:
    physical_cores: int
    logical_cores: int
    total_memory_gb: float
    available_memory_gb: float
    cpu_name: str
    platform: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MemoryStatus:
    total_gb: float
    available_gb: float
    used_percent: float
    swap_used_gb: float
    #: Remaining commit (RAM + page file) on Windows; ``None`` elsewhere.
    commit_available_gb: Optional[float] = None
    #: Physically available RAM before the commit limit was applied.
    physical_available_gb: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


def commit_available_gb() -> Optional[float]:
    """Remaining Windows commit charge (``ullAvailPageFile``), else ``None``.

    Allocations fail at the commit limit even while physical RAM is free. On
    2026-09-23 the machine had 9.1 GB physically available but only 6.0 GB of
    commit left (fixed 7.4 GB page file), and every guard looked only at the
    physical figure.
    """
    if os.name != "nt":
        return None
    try:
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return float(status.ullAvailPageFile) / 1024**3
    except Exception:
        return None


def available_memory_gb() -> Optional[float]:
    """The smaller of physically available RAM and remaining commit."""
    status = detect_memory_status()
    return None if status is None else status.available_gb


@dataclass(frozen=True)
class StageMemorySafetyPolicy:
    policy_version: str
    stage: str
    workload_class: str
    requested_memory_gb: float
    os_reserve_gb: float
    critical_available_gb: float
    recovery_available_gb: float
    duckdb_memory_limit_gb: float
    non_duckdb_headroom_gb: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class WorkerBenchmark:
    workers: int
    tasks: int
    elapsed_seconds: float
    tasks_per_second: float
    peak_parent_rss_gb: float
    peak_process_tree_rss_gb: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ResourceBenchmark:
    benchmark_version: str
    created_at: str
    hardware: HardwareInventory
    worker_results: tuple[WorkerBenchmark, ...]
    io_write_mb_s: float
    io_read_mb_s: float
    recommended_workers: int
    global_memory_budget_gb: float
    blas_threads_per_worker: int = 1

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["hardware"] = self.hardware.to_dict()
        payload["worker_results"] = [
            item.to_dict() for item in self.worker_results
        ]
        return payload


def detect_hardware() -> HardwareInventory:
    try:
        import psutil

        physical = psutil.cpu_count(logical=False) or os.cpu_count() or 1
        logical = psutil.cpu_count(logical=True) or physical
        memory = psutil.virtual_memory()
        total_gb = memory.total / 1024**3
        available_gb = memory.available / 1024**3
    except ImportError:
        logical = os.cpu_count() or 1
        physical = max(1, logical // 2)
        total_gb = 8.0
        available_gb = 4.0
    return HardwareInventory(
        physical_cores=int(physical),
        logical_cores=int(logical),
        total_memory_gb=float(total_gb),
        available_memory_gb=float(available_gb),
        cpu_name=platform.processor() or platform.machine(),
        platform=platform.platform(),
    )


def detect_memory_status() -> Optional[MemoryStatus]:
    """Return live system memory instead of the startup-time inventory."""
    try:
        import psutil
    except ImportError:
        return None
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    physical = float(memory.available / 1024**3)
    commit = commit_available_gb()
    return MemoryStatus(
        total_gb=float(memory.total / 1024**3),
        available_gb=physical if commit is None else min(physical, commit),
        used_percent=float(memory.percent),
        swap_used_gb=float(swap.used / 1024**3),
        commit_available_gb=commit,
        physical_available_gb=physical,
    )


def cpu_token_capacity(hardware: HardwareInventory) -> int:
    """Total CPU tokens the local machine can hand out to batch/background
    work at once, reserving a small headroom for the OS and the GUI's own
    event loop.

    Based on logical (hyperthreaded) core count, not just physical cores:
    this budgets multi-PROCESS parallelism, where each worker process
    pins its own BLAS/OMP/MKL thread count to 1 (apply_worker_thread_limits()
    below), so hyperthread sibling threads still provide real additional
    throughput instead of contending with a worker's own multi-threaded
    math library. Previously capped at physical_cores - 1, which left
    the second hardware thread of every core entirely unused -- half the
    machine's advertised capacity -- for no corresponding safety benefit.
    Shared by resource_profile_for_config() (jobs.py) and
    LocalJobSupervisor (supervisor.py); both MUST use this same function
    so a job's requested cpu_tokens never exceeds what the supervisor is
    willing to admit.
    """
    return max(1, hardware.logical_cores - 1)


def current_cpu_percent() -> Optional[float]:
    """Live, system-wide CPU utilization since the previous call.

    Non-blocking (uses psutil's "since last call" mode) so it is cheap
    enough to poll on a GUI timer. The first call in a process always
    returns 0.0 per psutil's own contract -- callers that need an
    accurate first reading should discard it.
    """
    try:
        import psutil
    except ImportError:
        return None
    return float(psutil.cpu_percent(interval=None))


def recommend_dynamic_workers(
    max_workers: int,
    *,
    sample_interval: float = 0.15,
    current_workers: Optional[int] = None,
    reserve_gb: Optional[float] = None,
) -> int:
    """Suggest a worker count for the NEXT micro-batch of a running job,
    from a fresh CPU/RAM sample. Managed root jobs may use a renewed
    supervisor grant above their initial allocation; unmanaged jobs retain
    max_workers as their ceiling. Child processes cannot spend a root grant.
    """
    try:
        import psutil
    except ImportError:
        return max(1, int(max_workers))
    from .resource_grants import worker_ceiling, granted_workers
    max_workers = granted_workers(
        worker_ceiling(max_workers), resident_workers=current_workers,
    )
    def own_cpu_seconds():
        parent = psutil.Process()
        values = {}
        for process in [parent, *parent.children(recursive=True)]:
            try:
                cpu = process.cpu_times()
                values[(process.pid, process.create_time())] = cpu.user + cpu.system
            except (psutil.Error, OSError, AttributeError):
                continue
        return values

    logical_cores = psutil.cpu_count(logical=True) or 1
    before_at = time.monotonic()
    try:
        before = own_cpu_seconds()
    except (psutil.Error, OSError, AttributeError):
        before = {}
    load_percent = psutil.cpu_percent(interval=sample_interval)
    try:
        after = own_cpu_seconds()
    except (psutil.Error, OSError, AttributeError):
        after = {}
    elapsed = max(time.monotonic() - before_at, 1e-6)
    own_cores = sum(max(0.0, value - before[key]) for key, value in after.items()
                    if key in before) / elapsed
    busy_cores = load_percent / 100.0 * logical_cores
    foreign_cores = max(0.0, busy_cores - min(busy_cores, own_cores))
    free_cores = max(1.0, logical_cores - 1 - foreign_cores)
    target = int(round(min(float(max_workers), free_cores)))
    if current_workers is not None:
        # Pool is idle at the call site: existing resident workers are already
        # included in system RAM usage. Budget only additions, or release workers
        # when reserve is consumed. Use the largest observed child plus headroom.
        try:
            children = psutil.Process().children(recursive=True)
            peak = max((child.memory_info().rss / 1024**3 for child in children), default=0.)
            # Measured worker size when there are worker processes (R3: the
            # old 2 GB floor overstated an extraction worker ~7.5-fold);
            # thread-based stages keep the conservative 2 GB share.
            per_worker = max(0.25, peak * 1.2) if peak > 0 else 2.0
            available = float(psutil.virtual_memory().available) / 1024**3
            commit = commit_available_gb()
            if commit is not None:
                available = min(available, commit)
            import math
            reserve = float(os.environ.get("VCG_JOB_OS_RESERVE_GB", "2.0")) if reserve_gb is None else float(reserve_gb)
            memory_target = int(current_workers) + math.floor((available - reserve) / per_worker)
            target = min(target, memory_target)
        except (psutil.Error, OSError, ValueError):
            # Failed telemetry never authorizes expansion.
            target = min(target, 1)
    return max(1, min(target, int(max_workers)))


def critical_available_memory_gb(*, reserve_gb: float) -> float:
    """Hard floor that triggers an atomic pause after sustained pressure."""
    return max(1.5, float(reserve_gb))


def stage_memory_safety_policy(
    stage: str,
    *,
    reserve_gb: float,
    requested_memory_gb: float,
) -> StageMemorySafetyPolicy:
    """Return an explicit RAM policy for one managed workload.

    The job allocation is a process-tree budget, not a DuckDB budget.  DuckDB
    therefore receives only the stage-appropriate part of the allocation;
    waveform extraction and statistical analysis retain headroom for Python,
    Arrow and worker processes. The configured system-RAM reserve is literal;
    workload-specific headroom is accounted for inside the job allocation.
    """

    normalized = str(stage or "unknown").strip().casefold()
    workload_class = _memory_workload_class(normalized)
    requested = max(1.0, float(requested_memory_gb))
    reserve = max(0.0, float(reserve_gb))
    critical = critical_available_memory_gb(reserve_gb=reserve)
    recovery = max(
        critical + 0.75,
        critical_available_memory_gb(reserve_gb=reserve) + 1.0,
    )

    minimum_headroom = {
        "waveform": 2.0,
        "analysis": 1.5,
        "join": 1.5,
        "orchestration": 2.0,
        "streaming": 1.0,
        "light": 1.0,
        "balanced": 1.0,
    }[workload_class]
    headroom_fraction = {
        "waveform": 0.25,
        "analysis": 0.20,
        "join": 0.20,
        "orchestration": 0.25,
        "streaming": 0.15,
        "light": 0.15,
        "balanced": 0.20,
    }[workload_class]
    desired_headroom = max(minimum_headroom, requested * headroom_fraction)
    duckdb_budget = max(1.0, requested - desired_headroom)
    duckdb_cap = {
        "streaming": 3.0,
        "light": 3.0,
        "join": 6.0,
        "analysis": 8.0,
    }.get(workload_class)
    # The two evidence stages build and independently attest the complete
    # clinical context relation. Unlike the marker-blocked repeatability
    # self-join, this wide relation cannot be decomposed without weakening the
    # exact-equivalence check. Give only these stages the measured larger
    # ceiling; all other joins retain the conservative 6 GB cap.
    if normalized in {"build-evidence", "attest-event-evidence"}:
        duckdb_cap = 10.0
    if duckdb_cap is not None:
        duckdb_budget = min(duckdb_budget, duckdb_cap)
    # DuckDB's SET memory_limit path uses integral GB values.  Record the same
    # effective value here instead of advertising unusable fractional budget.
    duckdb_budget = float(max(1, int(duckdb_budget)))
    return StageMemorySafetyPolicy(
        policy_version=MEMORY_SAFETY_POLICY_VERSION,
        stage=normalized or "unknown",
        workload_class=workload_class,
        requested_memory_gb=requested,
        os_reserve_gb=reserve,
        critical_available_gb=float(critical),
        recovery_available_gb=float(recovery),
        duckdb_memory_limit_gb=duckdb_budget,
        non_duckdb_headroom_gb=max(0.0, requested - duckdb_budget),
    )


@contextmanager
def stage_worker_memory_scope(
    stage: str,
    *,
    default_allocation_gb: float = 2.0,
    default_reserve_gb: float = 6.0,
) -> Iterator[StageMemorySafetyPolicy]:
    """Apply a stage-local DuckDB cap inside an already admitted worker."""

    allocation = float(
        os.environ.get(
            "VCG_JOB_MEMORY_ALLOCATION_GB",
            os.environ.get(
                "VCG_JOB_MEMORY_GB",
                str(max(1.0, float(default_allocation_gb))),
            ),
        )
    )
    reserve = float(
        os.environ.get(
            "VCG_JOB_OS_RESERVE_GB",
            str(max(0.0, float(default_reserve_gb))),
        )
    )
    policy = stage_memory_safety_policy(
        stage,
        reserve_gb=reserve,
        requested_memory_gb=allocation,
    )
    previous_limit = os.environ.get("VCG_JOB_MEMORY_GB")
    previous_stage = os.environ.get("VCG_JOB_ACTIVE_STAGE")
    os.environ["VCG_JOB_MEMORY_GB"] = str(policy.duckdb_memory_limit_gb)
    os.environ["VCG_JOB_ACTIVE_STAGE"] = policy.stage
    try:
        yield policy
    finally:
        if previous_limit is None:
            os.environ.pop("VCG_JOB_MEMORY_GB", None)
        else:
            os.environ["VCG_JOB_MEMORY_GB"] = previous_limit
        if previous_stage is None:
            os.environ.pop("VCG_JOB_ACTIVE_STAGE", None)
        else:
            os.environ["VCG_JOB_ACTIVE_STAGE"] = previous_stage


def _memory_workload_class(stage: str) -> str:
    if stage.startswith("extract-features") or stage in {
        "validate-delineation",
        "physiological-gap-markers",
    }:
        return "waveform"
    if stage.startswith("validate") or stage in {
        "discover-phenome",
        "refresh-phenome-promotions",
        "group-vcg",
        "therapy-response",
    }:
        return "analysis"
    if stage in {
        "build-evidence",
        "attest-event-evidence",
        "build-index",
        # Measured, not assumed. ``repeatability`` fell through to "balanced"
        # from a time when it had never run, and was handed a 12 GB DuckDB
        # budget with 4 GB of headroom -- while performing the widest read in
        # the project, a self-join over two times 1,753 marker columns. On
        # 2026-08-29 it twice drove available memory to 3.13 GB against its own
        # 3.30 GB floor and was paused, losing 4.6 hours the second time.
        #
        # It is a join, and the join budget (6 GB) is enough: with marker
        # blocking the stage runs at about 11 minutes per block and its memory
        # settles wherever the DuckDB budget lets it, rather than needing it.
        # A smaller budget lowers total usage without slowing the work, because
        # DuckDB otherwise simply fills the buffer it is given.
        "repeatability",
    }:
        return "join"
    if stage == "study-run":
        return "orchestration"
    if stage in {
        "ingest-events",
        "ingest",
        "parse-reports",
        "parse-notes",
        "build-labels",
        "inventory",
    }:
        return "streaming"
    if stage in {
        "report",
        "report-build",
        "audit",
        "audit-biomarkers",
        "attest-labels",
        "freeze",
    }:
        return "light"
    return "balanced"


def launch_available_memory_gb(
    *,
    reserve_gb: float,
    requested_memory_gb: float,
) -> float:
    """RAM required before launching a process, including startup headroom."""
    startup_headroom = min(
        4.0,
        max(1.0, float(requested_memory_gb) * 0.20),
    )
    return float(reserve_gb) + startup_headroom


def global_memory_budget_gb(
    hardware: HardwareInventory,
    *,
    reserve_gb: float = 6.0,
) -> float:
    return max(
        2.0,
        min(
            hardware.total_memory_gb * 0.80,
            hardware.total_memory_gb - reserve_gb,
        ),
    )


def benchmark_resources(
    config: MimicProjectConfig,
    *,
    candidates: Iterable[int] = (1, 2, 4, 6, 8, 10, 12, 14),
    tasks_per_worker: int = 3,
    progress: Optional[Callable[[str], None]] = None,
) -> ResourceBenchmark:
    hardware = detect_hardware()
    candidate_values = tuple(
        sorted(
            {
                max(1, min(int(value), hardware.logical_cores))
                for value in candidates
                if int(value) > 0
            }
        )
    )
    worker_results: list[WorkerBenchmark] = []
    parent_process = None
    try:
        import psutil

        parent_process = psutil.Process()
    except ImportError:
        pass

    for workers in candidate_values:
        tasks = max(workers, workers * max(1, int(tasks_per_worker)))
        if progress:
            progress(f"Benchmark {workers} Worker: {tasks} synthetische ECG-Aufgaben")
        started = time.perf_counter()
        peak_parent_rss = _rss_gb(parent_process)
        peak_tree_rss = _rss_tree_gb(parent_process)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for _ in executor.map(_benchmark_worker, range(tasks), chunksize=1):
                peak_parent_rss = max(
                    peak_parent_rss,
                    _rss_gb(parent_process),
                )
                peak_tree_rss = max(
                    peak_tree_rss,
                    _rss_tree_gb(parent_process),
                )
        elapsed = max(time.perf_counter() - started, 1e-9)
        worker_results.append(
            WorkerBenchmark(
                workers=workers,
                tasks=tasks,
                elapsed_seconds=elapsed,
                tasks_per_second=tasks / elapsed,
                peak_parent_rss_gb=peak_parent_rss,
                peak_process_tree_rss_gb=peak_tree_rss,
            )
        )

    write_speed, read_speed = _benchmark_io(config.jobs_root)
    best_rate = max(item.tasks_per_second for item in worker_results)
    near_best = [
        item
        for item in worker_results
        if item.tasks_per_second >= best_rate * 0.95
        and item.workers <= cpu_token_capacity(hardware)
    ]
    if near_best:
        recommended = max(near_best, key=lambda item: item.workers).workers
    else:
        recommended = max(worker_results, key=lambda item: item.tasks_per_second).workers
    result = ResourceBenchmark(
        benchmark_version=BENCHMARK_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(),
        hardware=hardware,
        worker_results=tuple(worker_results),
        io_write_mb_s=write_speed,
        io_read_mb_s=read_speed,
        recommended_workers=recommended,
        global_memory_budget_gb=global_memory_budget_gb(
            hardware,
            reserve_gb=config.os_memory_reserve_gb,
        ),
    )
    output = config.jobs_root / "hardware_benchmark.json"
    atomic_write_json(output, result.to_dict())
    return result


def apply_worker_thread_limits() -> None:
    global _THREAD_LIMITS_APPLIED, _THREADPOOL_LIMITER
    if _THREAD_LIMITS_APPLIED:
        return
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = "1"
    try:
        from threadpoolctl import threadpool_limits

        _THREADPOOL_LIMITER = threadpool_limits(limits=1)
    except ImportError:
        pass
    _THREAD_LIMITS_APPLIED = True


def _benchmark_worker(seed: int) -> float:
    apply_worker_thread_limits()
    rng = np.random.default_rng(20260725 + int(seed))
    signal = rng.normal(size=(5000, 12))
    signal -= np.mean(signal, axis=0, keepdims=True)
    spectrum = np.fft.rfft(signal, axis=0)
    covariance = np.cov(signal, rowvar=False)
    singular = np.linalg.svd(covariance, compute_uv=False)
    return float(np.sum(np.abs(spectrum[:100])) + np.sum(singular))


def _benchmark_io(root: Path) -> tuple[float, float]:
    root.mkdir(parents=True, exist_ok=True)
    payload = np.arange(4 * 1024 * 1024, dtype=np.float32)
    size_mb = payload.nbytes / 1024**2
    path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="vcg-io-benchmark-",
            suffix=".bin",
            dir=root,
            delete=False,
        ) as handle:
            path = Path(handle.name)
            started = time.perf_counter()
            handle.write(payload.tobytes())
            handle.flush()
            os.fsync(handle.fileno())
        write_elapsed = max(time.perf_counter() - started, 1e-9)
        started = time.perf_counter()
        with path.open("rb") as handle:
            while handle.read(8 * 1024 * 1024):
                pass
        read_elapsed = max(time.perf_counter() - started, 1e-9)
        return size_mb / write_elapsed, size_mb / read_elapsed
    finally:
        if path is not None and path.exists():
            path.unlink()


def _rss_gb(process) -> float:
    if process is None:
        return 0.0
    try:
        return process.memory_info().rss / 1024**3
    except Exception:
        return 0.0


def _rss_tree_gb(process) -> float:
    if process is None:
        return 0.0
    try:
        processes = [process, *process.children(recursive=True)]
        return sum(
            child.memory_info().rss
            for child in processes
            if child.is_running()
        ) / 1024**3
    except Exception:
        return _rss_gb(process)
