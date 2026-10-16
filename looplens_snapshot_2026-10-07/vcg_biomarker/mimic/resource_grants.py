"""Supervisor-owned, expiring worker grants with drain acknowledgement.

Only the root job process may expand. Child stages cannot independently spend
the same grant. A revoked grant stays charged until an idle boundary acknowledges
it; this prevents admission of another job while old workers are still resident.
"""
from __future__ import annotations

import json
import math
import os
import statistics
from collections import deque
from pathlib import Path
import time
from typing import Optional

from .storage import atomic_write_json

GRANT_ENV = "VCG_RESOURCE_GRANT_FILE"
SAMPLE_SECONDS = 2.0
GRANT_TTL_SECONDS = 8.0
FOREIGN_HISTORY_SAMPLES = 5
WORKER_MEMORY_HISTORY_SAMPLES = 30
WORKER_MEMORY_SAFETY = 1.2
MINIMUM_WORKER_MEMORY_GB = 0.25


def grant_status_text(root: Path, job_id: str, owner_pid: int | None) -> str:
    """Read-only UI status; permission is not an assertion of active CPU work."""
    grant = read_grant(ResourceGrantCoordinator(root, 1, 2).path(job_id))
    if not grant or grant.get("owner_pid") != owner_pid:
        return "Worker-Budget: kein aktueller Messnachweis"
    reasons = {
        "cpu_or_ram_budget": "CPU-/RAM-Budget",
        "shared_or_queued_workload": "weitere Jobs vorgemerkt/aktiv",
        "telemetry_not_ready": "Messwerte noch unvollständig",
        "ram_pressure_drain": "RAM-Druck: Worker laufen geordnet aus",
    }
    reason = reasons.get(grant.get("limiting_reason"), "Limitierungsgrund unbekannt")
    return f"Worker-Budget {grant['workers']}/{grant['ceiling']} · {reason}"


def read_grant(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            return {}
        age = time.time() - float(value["sampled_at"])
        if not 0 <= age <= GRANT_TTL_SECONDS:
            return {}
        if type(value.get("workers")) is not int or value["workers"] < 1:
            return {}
        if (type(value.get("ceiling")) is not int or value["ceiling"] < value["workers"]
                or type(value.get("generation")) is not int or value["generation"] < 1
                or type(value.get("owner_pid")) is not int or value["owner_pid"] < 1):
            return {}
        return value
    except (OSError, ValueError, TypeError, KeyError):
        return {}


_GRANT_INVALID_SINCE: list = [None]


def _note_grant_validity(valid: bool) -> None:
    if valid:
        _GRANT_INVALID_SINCE[0] = None
    elif _GRANT_INVALID_SINCE[0] is None:
        _GRANT_INVALID_SINCE[0] = time.monotonic()


def invalid_grant_seconds() -> Optional[float]:
    """Seconds this managed job has been without a valid own grant (O3).

    ``None`` when the job is unmanaged or its grant is valid. Without a live
    supervisor a grant expires after 8 s and the job silently continues on
    one worker; the progress messages report that after 60 s.
    """
    if not os.environ.get(GRANT_ENV) or _GRANT_INVALID_SINCE[0] is None:
        return None
    return time.monotonic() - _GRANT_INVALID_SINCE[0]


def worker_ceiling(fallback: int) -> int:
    path = os.environ.get(GRANT_ENV)
    if not path:
        return max(1, int(fallback))
    grant = read_grant(Path(path))
    _note_grant_validity(grant.get("owner_pid") == os.getpid())
    if grant.get("owner_pid") != os.getpid():
        return 1
    return max(1, int(grant.get("ceiling", 1)))


def granted_workers(fallback: int, *, resident_workers: int | None = None) -> int:
    path = os.environ.get(GRANT_ENV)
    if not path:
        return max(1, int(fallback))
    target = Path(path)
    grant = read_grant(target)
    _note_grant_validity(grant.get("owner_pid") == os.getpid())
    if grant.get("owner_pid") != os.getpid():
        return 1
    if resident_workers is not None:
        atomic_write_json(target.with_suffix(".ack.json"), {
            "owner_pid": os.getpid(), "generation": grant["generation"],
            "resident_workers": max(0, int(resident_workers)), "acknowledged_at": time.time(),
        })
    return min(max(1, int(fallback)), grant["workers"])


# A job that uses less than this share of its grant, measured, reports only
# what it uses (plus one) as its demand; the rest goes to the other jobs.
FAIR_SHARE_UNDERUSE_FRACTION = 0.6


def fair_cpu_shares(demands: dict[str, float], room: int) -> dict[str, int]:
    """Max-min fair division of ``room`` workers (audit E8).

    Water-filling: every job gets an equal share; a job that demands less
    keeps only its demand and the remainder is divided among the others.
    Every job keeps at least one worker.
    """
    remaining = {key: float(value) for key, value in demands.items()}
    shares: dict[str, int] = {}
    left = float(max(len(remaining), room))
    while remaining:
        equal = left / len(remaining)
        satisfied = {key: value for key, value in remaining.items() if value <= equal}
        if not satisfied:
            for key in sorted(remaining):
                shares[key] = max(1, int(math.floor(equal)))
            break
        for key, value in satisfied.items():
            shares[key] = max(1, int(math.ceil(value)))
            left -= value
            del remaining[key]
    return shares


class ResourceGrantCoordinator:
    """Single writer under the supervisor's existing single-owner lifecycle."""

    def __init__(self, root: Path, capacity: int, reserve_gb: float):
        self.root, self.capacity, self.reserve_gb = root, capacity, reserve_gb
        self.last_sample = 0.0
        self.grants: dict[str, dict] = {}
        self.stable: dict[str, int] = {}
        self.previous_cpu: dict[tuple[int, float], float] = {}
        self.previous_at = 0.0
        # Foreign load of the last five samples (10 s); the median decides.
        self.foreign_history: deque[float] = deque(maxlen=FOREIGN_HISTORY_SAMPLES)
        # Largest private memory of one worker process per job, last minute.
        self.worker_memory_history: dict[str, deque[float]] = {}

    def path(self, job_id: str) -> Path:
        # Job IDs are external input; use a deterministic filename, never a path.
        from .storage import content_hash
        return self.root / (content_hash({"job_id": job_id}) + ".json")

    def charged_workers(self, record) -> int:
        base = int(record.spec.resource_profile.cpu_tokens)
        grant = self.grants.get(record.spec.job_id)
        if not grant:
            # Expiry revokes execution permission, not already resident workers.
            # Recover reservations before admitting work after a supervisor restart.
            try:
                saved = json.loads(self.path(record.spec.job_id).read_text(encoding="utf-8"))
                if (isinstance(saved, dict) and saved.get("owner_pid") == record.worker_pid
                        and saved.get("job_id") == record.spec.job_id
                        and all(type(saved.get(key)) is int and saved[key] >= 1
                                for key in ("workers", "generation", "unacknowledged_workers"))):
                    grant = saved
                    self.grants[record.spec.job_id] = saved
            except (OSError, ValueError, TypeError):
                pass
        if not grant:
            return base
        try:
            ack = json.loads(self.path(record.spec.job_id).with_suffix(".ack.json").read_text(encoding="utf-8"))
            if (ack["owner_pid"] == record.worker_pid and ack["generation"] == grant["generation"]
                    and type(ack["resident_workers"]) is int and ack["resident_workers"] >= 0):
                return max(base, grant["workers"], ack["resident_workers"])
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return max(base, grant["workers"], grant["unacknowledged_workers"])

    def sample(self, running, *, queued: bool, available_gb: float | None) -> None:
        now = time.time()
        if now - self.last_sample < SAMPLE_SECONDS:
            return
        self.last_sample = now
        import psutil
        samples = {}
        keys_by_job: dict[str, list] = {}
        telemetry_ok = available_gb is not None
        for record in running:
            if not record.worker_pid:
                telemetry_ok = False
                continue
            child_memory: list[float] = []
            try:
                parent = psutil.Process(record.worker_pid)
                for process in [parent, *parent.children(recursive=True)]:
                    cpu = process.cpu_times()
                    key = (process.pid, process.create_time())
                    samples[key] = cpu.user + cpu.system
                    keys_by_job.setdefault(record.spec.job_id, []).append(key)
                    if process.pid != parent.pid:
                        try:
                            memory = process.memory_info()
                            child_memory.append(
                                float(getattr(memory, "private", memory.rss)) / 1024**3
                            )
                        except psutil.Error:
                            pass
            except (psutil.Error, TypeError):
                telemetry_ok = False
            if child_memory:
                history = self.worker_memory_history.setdefault(
                    record.spec.job_id,
                    deque(maxlen=WORKER_MEMORY_HISTORY_SAMPLES),
                )
                history.append(max(child_memory))
            else:
                # A stage without worker processes: an earlier stage's worker
                # size must not keep deciding this one's memory share.
                self.worker_memory_history.pop(record.spec.job_id, None)
        elapsed = now - self.previous_at
        own_cores = 0.0
        own_cores_by_job: dict[str, float] = {}

        def interval_seconds(key, value) -> float:
            if key in self.previous_cpu:
                return max(0.0, value - self.previous_cpu[key])
            if key[1] >= self.previous_at - 1.0:
                # Started after the previous sample: all of its CPU time
                # belongs to this interval and to this job. Leaving it out
                # (audit E3) booked every newly spawned worker's import as
                # foreign load and cut the grant exactly when the pool grew.
                return max(0.0, value)
            return 0.0

        if self.previous_at and elapsed > 0:
            own_seconds = 0.0
            for key, value in samples.items():
                own_seconds += interval_seconds(key, value)
            own_cores = own_seconds / elapsed
            for job_id, keys in keys_by_job.items():
                own_cores_by_job[job_id] = sum(
                    interval_seconds(key, samples[key]) for key in keys
                ) / elapsed
        total_busy = psutil.cpu_percent(interval=None) * (psutil.cpu_count() or 1) / 100.0
        own_measured = bool(self.previous_at)
        telemetry_ok &= own_measured
        self.previous_cpu, self.previous_at = samples, now
        # The first sample has no own-CPU interval yet; booking the whole
        # machine load as foreign would pull the 10-s median down for five
        # samples and shrink every grant right after a start.
        if own_measured:
            self.foreign_history.append(max(0.0, total_busy - own_cores))
        # One 2-s sample decided the grant before; a single spike of foreign
        # load removed workers at once. The median of 10 s ignores spikes, and
        # the workers run below normal priority, so a real foreground load
        # takes the CPU immediately anyway.
        foreign_cores = (
            statistics.median(self.foreign_history) if self.foreign_history else 0.0
        )
        cpu_room = max(1, math.floor(self.capacity - foreign_cores))
        # Audit E8: several standard jobs share the budget max-min fairly
        # instead of all sitting at their admission tokens. A queued job
        # still makes the running ones hold at their base so it can start.
        standard = [
            item for item in running
            if not item.spec.resource_profile.interactive
        ]
        sharing = len(standard) > 1 and not queued
        shares: dict[str, int] = {}
        if sharing:
            demands: dict[str, float] = {}
            for item in standard:
                job_id = item.spec.job_id
                base_workers = max(1, int(item.spec.resource_profile.cpu_tokens))
                granted_now = int(self.grants.get(job_id, {}).get("workers", base_workers))
                used = own_cores_by_job.get(job_id)
                if (
                    used is not None
                    and used < FAIR_SHARE_UNDERUSE_FRACTION * granted_now
                ):
                    demands[job_id] = max(float(base_workers), math.ceil(used) + 1.0)
                else:
                    demands[job_id] = float(self.capacity)
            shares = fair_cpu_shares(demands, cpu_room)
        for record in running:
            job_id = record.spec.job_id
            base = max(1, int(record.spec.resource_profile.cpu_tokens))
            self.charged_workers(record)  # Restore outstanding reservations first.
            old = self.grants.get(job_id, {})
            current = int(old.get("workers", base))
            # Expansion is initially restricted to a single root workload. Parallel
            # stages/jobs retain their already-admitted joint allocation.
            expandable = len(running) == 1 and not queued and not record.spec.resource_profile.interactive
            spare_ram = max(0.0, (available_gb or 0.0) - self.reserve_gb)
            per_worker = self.memory_per_worker_gb(
                job_id,
                configured=float(record.spec.resource_profile.memory_gb) / base,
            )
            if expandable:
                target = min(cpu_room, base + math.floor(spare_ram / per_worker))
                reason = "cpu_or_ram_budget"
            elif sharing and job_id in shares:
                ram_share = spare_ram / len(standard)
                target = max(
                    1,
                    min(shares[job_id], base + math.floor(ram_share / per_worker)),
                )
                reason = "fair_share_between_jobs"
            else:
                target, reason = base, "shared_or_queued_workload"
            if not telemetry_ok:
                target, reason = min(base, current), "telemetry_not_ready"
            if available_gb is not None and available_gb < self.reserve_gb:
                target, reason = max(1, current - 1), "ram_pressure_drain"
            if target > current:
                self.stable[job_id] = self.stable.get(job_id, 0) + 1
                target = min(target, current + 1) if self.stable[job_id] >= 3 else current
                if target > current:
                    self.stable[job_id] = 0
            else:
                self.stable[job_id] = 0
            target = max(1, min(target, self.capacity))
            generation = int(old.get("generation", 0)) + int(target != current or not old)
            outstanding = self.charged_workers(record)
            grant = {
                "version": "1.0.0", "job_id": job_id, "owner_pid": record.worker_pid,
                "sampled_at": now, "generation": generation, "workers": target,
                "ceiling": (
                    self.capacity
                    if expandable or (sharing and job_id in shares)
                    else base
                ),
                "unacknowledged_workers": max(outstanding, current),
                "limiting_reason": reason, "ram_reserve_gb": self.reserve_gb,
                "available_gb": available_gb, "cpu_foreign_cores": foreign_cores,
                "cpu_own_cores": own_cores,
                "cpu_job_cores": own_cores_by_job.get(job_id),
                "memory_per_worker_gb": per_worker,
                "pressure_started_at": (old.get("pressure_started_at") or now)
                if available_gb is not None and available_gb < self.reserve_gb else None,
            }
            self.grants[job_id] = grant
            atomic_write_json(self.path(job_id), grant)

    def memory_per_worker_gb(self, job_id: str, *, configured: float) -> float:
        """Measured private memory of one worker, else the profile's share.

        The fixed floor of 2 GB overstated an extraction worker by a factor of
        ~7.5 (measured 0.26-0.27 GB private on A131, 2026-09-23); once the CPU
        bookkeeping was repaired, that floor would have become the limit.
        Stages without worker processes keep the configured share.
        """
        history = self.worker_memory_history.get(job_id)
        if history:
            return max(MINIMUM_WORKER_MEMORY_GB, max(history) * WORKER_MEMORY_SAFETY)
        return max(MINIMUM_WORKER_MEMORY_GB, float(configured))

    def drain_in_progress(self, record, available_gb: float) -> bool:
        """Bounded grace for live workers; severe pressure still permits a pause."""
        grant = self.grants.get(record.spec.job_id, {})
        started = grant.get("pressure_started_at")
        return bool(
            available_gb >= 1.0 and started is not None
            and 0 <= time.time() - started < 30.0
            and self.charged_workers(record) > grant.get("workers", self.capacity)
        )
