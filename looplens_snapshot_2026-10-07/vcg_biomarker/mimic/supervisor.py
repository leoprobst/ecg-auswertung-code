from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from .config import MimicProjectConfig
from .jobs import JobStore
from .models import JobRecord, JobState
from .runtime_snapshot import create_runtime_snapshot, runtime_environment
from .resource_grants import ResourceGrantCoordinator, GRANT_ENV
from .resources import (
    MemoryStatus,
    cpu_token_capacity,
    detect_hardware,
    detect_memory_status,
    global_memory_budget_gb,
    launch_available_memory_gb,
    stage_memory_safety_policy,
)


SUPERVISOR_POLL_SECONDS = 1.0
SUPERVISOR_STALE_SECONDS = 15
MEMORY_PRESSURE_SAMPLES = 3
MEMORY_READY_SAMPLES = 3
# Exit code of a job worker that paused itself (mimic_job_worker returns 75).
WORKER_PAUSE_EXIT_CODE = 75

# Opening words of an automatic memory pause.  The resume pass matches on
# this, which is how it tells its own pause from a person's: a pause someone
# asked for must never be undone by the machine.
MEMORY_PAUSE_MARKER = "Automatische Pause wegen kritischem Arbeitsspeicher"

# Ten minutes between routine memory lines: a month of running stays
# under 5,000 lines, and a pause still has a visible run-up.
TRACE_INTERVAL_SECONDS = 600.0
MEMORY_WAIT_NOTICE_SECONDS = 60.0


class LocalJobSupervisor:
    def __init__(
        self,
        config: MimicProjectConfig,
        *,
        memory_probe: Callable[[], Optional[MemoryStatus]] = detect_memory_status,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.config = config
        self.store = JobStore.for_project(config)
        self.hardware = detect_hardware()
        self.cpu_capacity = cpu_token_capacity(self.hardware)
        self.interactive_cpu_capacity = min(
            2,
            max(1, self.hardware.logical_cores - self.cpu_capacity),
        )
        self.memory_capacity_gb = global_memory_budget_gb(
            self.hardware,
            reserve_gb=config.os_memory_reserve_gb,
        )
        self._children: dict[str, subprocess.Popen] = {}
        self._memory_probe = memory_probe
        self._clock = clock
        self._low_memory_samples: dict[str, int] = {}
        self._memory_ready_samples: dict[str, int] = {}
        self._memory_resume_samples: dict[str, int] = {}
        self._memory_wait_notices: dict[str, float] = {}
        self._last_trace_at: float = 0.0
        self._resource_grants = ResourceGrantCoordinator(
            config.jobs_root / "resource_grants", self.cpu_capacity, config.os_memory_reserve_gb,
        )

    def _trace_periodic(self, running, memory_status) -> None:
        """A memory curve instead of a single endpoint.

        Ten minutes apart: enough to see what the memory ran towards before a
        pause, few enough that a month of running stays a small file.
        """

        now = self._clock()
        if now - self._last_trace_at < TRACE_INTERVAL_SECONDS:
            return
        self._last_trace_at = now
        available = (
            f"{memory_status.available_gb:.2f} GB frei"
            if memory_status is not None
            else "Speicher unbekannt"
        )
        names = ", ".join(item.spec.stage for item in running) or "keine"
        self._trace(f"{available}; laufend: {names}")

    def _trace(self, message: str) -> None:
        """One line to stdout, which ``ensure_supervisor`` captures.

        ``jobs/supervisor.log`` existed since July and was zero bytes: the
        file was opened for the child's output and the child never wrote any.
        An empty log is worse than none, because it answers "is this being
        recorded?" with a yes. The supervisor decides when a month-long run
        pauses and when it resumes, so those are the lines it owes.

        Wrapped: a closed pipe must not take down the process that keeps the
        study alive.
        """

        try:
            print(
                f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z] {message}",
                flush=True,
            )
        except Exception:
            return

    def run_forever(self) -> None:
        self._trace(
            f"Supervisor gestartet: {self.cpu_capacity} CPU-Token, "
            f"{self.memory_capacity_gb:.1f} GB Budget, "
            f"Reserve {self.config.os_memory_reserve_gb:.1f} GB"
        )
        self._cleanup_historical_orphans()
        self.store.recover_stale_jobs()
        reason = "unbekannt"
        try:
            while True:
                self.run_once()
                time.sleep(SUPERVISOR_POLL_SECONDS)
        except KeyboardInterrupt:
            reason = "Unterbrechung (Strg+C oder Signal)"
            raise
        except BaseException as exc:
            reason = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            # Audit O3: without a live supervisor the grants expire after 8 s
            # and every job continues on one worker, silently. The departure
            # and its cause belong in the log.
            self._trace(
                f"Supervisor beendet ({reason}); laufende Jobs fallen nach "
                "Ablauf der Zuteilung auf einen Worker zurueck"
            )

    def run_once(self) -> None:
        self.store.supervisor_heartbeat(self.config.project_file)
        self._reap_children()
        self.store.recover_stale_jobs()
        running = self.store.list(
            states=(
                JobState.RUNNING,
                JobState.PAUSING,
                JobState.CANCELLING,
            )
        )
        memory_status = self._memory_probe()
        self._resource_grants.sample(
            running, queued=bool(self.store.list(states=(JobState.QUEUED,))),
            available_gb=memory_status.available_gb if memory_status is not None else None,
        )
        self._trace_periodic(running, memory_status)
        if self._guard_running_jobs_for_memory(running, memory_status):
            return
        self._resume_memory_paused_jobs(memory_status)
        used_standard_cpu = sum(
            self._resource_grants.charged_workers(item)
            for item in running
            if not item.spec.resource_profile.interactive
        )
        used_interactive_cpu = sum(
            self._resource_grants.charged_workers(item)
            for item in running
            if item.spec.resource_profile.interactive
        )
        used_memory = sum(item.spec.resource_profile.memory_gb for item in running)
        writer_namespaces = {
            item.spec.resource_profile.writer_namespace
            for item in running
            if item.spec.resource_profile.writer_namespace
        }
        for record in self.store.list(states=(JobState.QUEUED,)):
            if record.spec.job_id in self._children:
                continue
            profile = record.spec.resource_profile
            writer = profile.writer_namespace
            if writer and writer in writer_namespaces:
                continue
            profile_cpu_capacity = (
                self.interactive_cpu_capacity
                if profile.interactive
                else self.cpu_capacity
            )
            if (
                profile.cpu_tokens > profile_cpu_capacity
                or profile.memory_gb > self.memory_capacity_gb
            ):
                self.store.transition(
                    record.spec.job_id,
                    JobState.BLOCKED,
                    message=(
                        "Resource request exceeds the local supervisor budget"
                    ),
                    error_message=(
                        f"requested {profile.cpu_tokens} CPU tokens and "
                        f"{profile.memory_gb:g} GB RAM; available "
                        f"{profile_cpu_capacity} CPU tokens and "
                        f"{self.memory_capacity_gb:.1f} GB RAM"
                    ),
                )
                continue
            cpu_fits = (
                used_interactive_cpu + profile.cpu_tokens
                <= self.interactive_cpu_capacity
                if profile.interactive
                else used_standard_cpu + profile.cpu_tokens
                <= self.cpu_capacity
            )
            fits = (
                cpu_fits
                and used_standard_cpu + used_interactive_cpu + profile.cpu_tokens <= self.cpu_capacity
                and used_memory + profile.memory_gb <= self.memory_capacity_gb
            )
            if not fits:
                continue
            if not self._live_memory_fits(record, memory_status):
                continue
            if not self._launch(record):
                continue
            self._clear_memory_wait(record)
            if profile.interactive:
                used_interactive_cpu += profile.cpu_tokens
            else:
                used_standard_cpu += profile.cpu_tokens
            used_memory += profile.memory_gb
            if memory_status is not None:
                startup_headroom = (
                    launch_available_memory_gb(
                        reserve_gb=0.0,
                        requested_memory_gb=profile.memory_gb,
                    )
                )
                memory_status = MemoryStatus(
                    total_gb=memory_status.total_gb,
                    available_gb=max(
                        0.0,
                        memory_status.available_gb - startup_headroom,
                    ),
                    used_percent=memory_status.used_percent,
                    swap_used_gb=memory_status.swap_used_gb,
                )
            if writer:
                writer_namespaces.add(writer)

    def _guard_running_jobs_for_memory(
        self,
        running: list[JobRecord],
        memory_status: Optional[MemoryStatus],
    ) -> bool:
        if memory_status is None:
            self._low_memory_samples.clear()
            return False
        # A cooperative pause must drain and release memory before another
        # job is selected. Repeated polls are not evidence of a completed pause.
        if any(
            record.state == JobState.PAUSING
            and str(record.error_message or "").startswith(MEMORY_PAUSE_MARKER)
            for record in running
        ):
            self._low_memory_samples.clear()
            return True
        candidates = [
            record
            for record in running
            if record.state == JobState.RUNNING
            and not record.spec.resource_profile.interactive
        ]
        policies = {
            record.spec.job_id: stage_memory_safety_policy(
                self._active_runtime_stage(record),
                reserve_gb=self.config.os_memory_reserve_gb,
                requested_memory_gb=(
                    record.spec.resource_profile.memory_gb
                ),
            )
            for record in candidates
        }
        threatened = [
            record
            for record in candidates
            if memory_status.available_gb
            <= policies[record.spec.job_id].critical_available_gb
        ]
        threatened_ids = {record.spec.job_id for record in threatened}
        for job_id in tuple(self._low_memory_samples):
            if job_id not in threatened_ids:
                self._low_memory_samples.pop(job_id, None)
        if not threatened:
            return False
        for record in threatened:
            self._low_memory_samples[record.spec.job_id] = (
                self._low_memory_samples.get(record.spec.job_id, 0) + 1
            )
        eligible = [
            record
            for record in threatened
            if self._low_memory_samples.get(record.spec.job_id, 0)
            >= MEMORY_PRESSURE_SAMPLES
            and not self._resource_grants.drain_in_progress(
                record, memory_status.available_gb,
            )
        ]
        if not eligible:
            return True

        candidate = min(
            eligible,
            key=lambda record: (
                record.spec.priority,
                -record.spec.resource_profile.memory_gb,
                record.spec.job_id,
            ),
        )
        policy = policies[candidate.spec.job_id]
        samples = self._low_memory_samples.get(candidate.spec.job_id, 0)
        message = (
            f"{MEMORY_PAUSE_MARKER}: "
            f"{memory_status.available_gb:.2f} GB verfügbar, "
            "stufenspezifische Sicherheitsgrenze "
            f"{policy.critical_available_gb:.2f} GB "
            f"({policy.stage}/{policy.workload_class})"
        )
        self._trace(f"{candidate.spec.job_id}: {message} (Stichproben {samples})")
        self.store.request_pause(
            candidate.spec.job_id,
            message=message,
            diagnostic=message,
        )
        self.store.append_event(
            candidate.spec.job_id,
            "memory_pressure_pause_requested",
            message,
            {
                "available_memory_gb": round(memory_status.available_gb, 3),
                "total_memory_gb": round(memory_status.total_gb, 3),
                "used_memory_percent": round(memory_status.used_percent, 3),
                "swap_used_gb": round(memory_status.swap_used_gb, 3),
                "critical_memory_gb": round(
                    policy.critical_available_gb,
                    3,
                ),
                "recovery_memory_gb": round(
                    policy.recovery_available_gb,
                    3,
                ),
                "configured_os_reserve_gb": float(
                    self.config.os_memory_reserve_gb
                ),
                "active_stage": policy.stage,
                "workload_class": policy.workload_class,
                "memory_policy_version": policy.policy_version,
                "requested_memory_gb": policy.requested_memory_gb,
                "duckdb_memory_limit_gb": (
                    policy.duckdb_memory_limit_gb
                ),
                "non_duckdb_headroom_gb": (
                    policy.non_duckdb_headroom_gb
                ),
                "samples": int(samples),
            },
        )
        self._low_memory_samples.clear()
        return True

    def _resume_memory_paused_jobs(
        self,
        memory_status: Optional[MemoryStatus],
    ) -> None:
        """Restart what this supervisor paused, once the memory came back.

        Pausing under memory pressure is right; staying paused afterwards is
        not. The pause was never resumed automatically, so a month-long run
        stopped at the first pressure spike and waited for somebody to notice
        -- on 2026-08-26 that happened five times before ten in the morning,
        and the machine had 16 GB free while the run sat idle.

        Two things keep this safe. Only pauses carrying MEMORY_PAUSE_MARKER
        are touched, so a pause a person asked for stays; and the recovery
        threshold is the stage's own ``recovery_available_gb``, which sits
        above the critical level that triggered the pause, so the run does
        not resume straight back into the pressure that stopped it. The same
        consecutive-sample count as the launch path guards against flapping.
        """

        if memory_status is None:
            self._memory_resume_samples.clear()
            return
        paused = self.store.list(states=(JobState.PAUSED,))
        live_ids = set()
        for record in paused:
            if not str(record.error_message or "").startswith(
                MEMORY_PAUSE_MARKER
            ):
                continue
            policy = stage_memory_safety_policy(
                self._active_runtime_stage(record),
                reserve_gb=self.config.os_memory_reserve_gb,
                requested_memory_gb=record.spec.resource_profile.memory_gb,
            )
            if memory_status.available_gb < policy.recovery_available_gb:
                self._memory_resume_samples.pop(record.spec.job_id, None)
                continue
            live_ids.add(record.spec.job_id)
            samples = self._memory_resume_samples.get(record.spec.job_id, 0) + 1
            self._memory_resume_samples[record.spec.job_id] = samples
            if samples < MEMORY_READY_SAMPLES:
                continue
            message = (
                "Arbeitsspeicher erholt: "
                f"{memory_status.available_gb:.2f} GB verfuegbar, "
                f"Erholungsgrenze {policy.recovery_available_gb:.2f} GB "
                f"({policy.stage}) -- Job wird fortgesetzt"
            )
            self._trace(f"{record.spec.job_id}: {message}")
            try:
                self.store.request_resume(record.spec.job_id)
            except Exception:
                # A job someone cancelled or finished meanwhile is not an
                # error worth stopping the supervisor for.
                self._memory_resume_samples.pop(record.spec.job_id, None)
                continue
            self.store.append_event(
                record.spec.job_id,
                "memory_pressure_resume_requested",
                message,
                {
                    "available_memory_gb": round(
                        memory_status.available_gb, 3
                    ),
                    "recovery_memory_gb": round(
                        policy.recovery_available_gb, 3
                    ),
                    "active_stage": policy.stage,
                    "samples": int(samples),
                },
            )
            # Never admit multiple resumptions from the same RAM sample.
            self._memory_resume_samples.clear()
            return
        for job_id in tuple(self._memory_resume_samples):
            if job_id not in live_ids:
                self._memory_resume_samples.pop(job_id, None)

    def _active_runtime_stage(self, record: JobRecord) -> str:
        if record.spec.stage != "study-run":
            return record.spec.stage
        runtime = self.store.latest_runtime_stage(record.spec.job_id)
        if runtime is None:
            return record.spec.stage
        stage = str(runtime.get("stage", "")).strip()
        return stage or record.spec.stage

    def _live_memory_fits(
        self,
        record: JobRecord,
        memory_status: Optional[MemoryStatus],
    ) -> bool:
        if memory_status is None:
            return True
        policy = stage_memory_safety_policy(
            self._active_runtime_stage(record),
            reserve_gb=self.config.os_memory_reserve_gb,
            requested_memory_gb=record.spec.resource_profile.memory_gb,
        )
        required_gb = max(
            policy.recovery_available_gb,
            launch_available_memory_gb(
            reserve_gb=self.config.os_memory_reserve_gb,
            requested_memory_gb=record.spec.resource_profile.memory_gb,
            ),
        )
        if memory_status.available_gb >= required_gb:
            ready_samples = (
                self._memory_ready_samples.get(record.spec.job_id, 0) + 1
            )
            self._memory_ready_samples[record.spec.job_id] = ready_samples
            if ready_samples >= MEMORY_READY_SAMPLES:
                return True
            self.store.heartbeat(
                record.spec.job_id,
                progress_message=(
                    "Bestätige stabile Arbeitsspeicherreserve "
                    f"({ready_samples}/{MEMORY_READY_SAMPLES})"
                ),
            )
            return False
        self._memory_ready_samples[record.spec.job_id] = 0
        now = self._clock()
        last_notice = self._memory_wait_notices.get(record.spec.job_id)
        if (
            last_notice is None
            or now - last_notice >= MEMORY_WAIT_NOTICE_SECONDS
        ):
            message = (
                "Wartet auf freien Arbeitsspeicher: "
                f"{memory_status.available_gb:.2f} GB verfügbar, "
                f"{required_gb:.2f} GB zum sicheren Start erforderlich "
                f"({policy.stage}/{policy.workload_class})"
            )
            self.store.heartbeat(
                record.spec.job_id,
                progress_message=message,
            )
            self.store.append_event(
                record.spec.job_id,
                "memory_pressure_wait",
                message,
                {
                    "available_memory_gb": round(
                        memory_status.available_gb,
                        3,
                    ),
                    "required_memory_gb": round(required_gb, 3),
                    "configured_os_reserve_gb": float(
                        self.config.os_memory_reserve_gb
                    ),
                    "active_stage": policy.stage,
                    "workload_class": policy.workload_class,
                    "memory_policy_version": policy.policy_version,
                    "critical_memory_gb": round(
                        policy.critical_available_gb,
                        3,
                    ),
                    "recovery_memory_gb": round(
                        policy.recovery_available_gb,
                        3,
                    ),
                    "duckdb_memory_limit_gb": (
                        policy.duckdb_memory_limit_gb
                    ),
                    "non_duckdb_headroom_gb": (
                        policy.non_duckdb_headroom_gb
                    ),
                },
            )
            self._memory_wait_notices[record.spec.job_id] = now
        return False

    def _clear_memory_wait(self, record: JobRecord) -> None:
        self._memory_ready_samples.pop(record.spec.job_id, None)
        if self._memory_wait_notices.pop(record.spec.job_id, None) is None:
            return
        self.store.append_event(
            record.spec.job_id,
            "memory_pressure_wait_cleared",
            "Arbeitsspeicherreserve wiederhergestellt; Job wird gestartet",
        )

    def _launch(self, record: JobRecord) -> bool:
        reserved = self.store.reserve_for_launch(
            record.spec.job_id,
            supervisor_pid=os.getpid(),
        )
        if reserved is None:
            return False
        command = [
            sys.executable,
            "-m",
            "vcg_biomarker.mimic_job_worker",
            "--project",
            str(self.config.project_file),
            "--job-id",
            record.spec.job_id,
        ]
        kwargs: dict = {
            "cwd": str(Path(__file__).resolve().parents[2]),
            "env": runtime_environment(),
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
        kwargs["env"][GRANT_ENV] = str(self._resource_grants.path(record.spec.job_id))
        if os.name == "nt":
            # Below-normal priority (audit R6): Windows then gives any
            # foreground program the CPU within milliseconds, so the grant can
            # aim at the whole machine instead of keeping a safety margin.
            # Pool workers inherit the class of a below-normal parent.
            kwargs["creationflags"] = (
                subprocess.CREATE_NO_WINDOW
                | subprocess.BELOW_NORMAL_PRIORITY_CLASS
            )
        try:
            process = subprocess.Popen(command, **kwargs)
        except Exception as exc:
            self.store.transition(
                record.spec.job_id,
                JobState.FAILED,
                message=f"Worker launch failed: {exc}",
                error_message=f"{type(exc).__name__}: {exc}",
            )
            return False
        self._children[record.spec.job_id] = process
        _lower_io_priority(process.pid)
        self.store.attach_worker(record.spec.job_id, process.pid)
        self.store.append_event(
            record.spec.job_id,
            "worker_launched",
            f"Worker process {process.pid} launched",
            {"pid": process.pid},
        )
        return True

    def _reap_children(self) -> None:
        for job_id, process in list(self._children.items()):
            exit_code = process.poll()
            if exit_code is None:
                continue
            self._children.pop(job_id, None)
            terminated = _terminate_process_pool_children(process.pid)
            try:
                record = self.store.get(job_id)
            except KeyError:
                continue
            if terminated:
                self.store.append_event(
                    job_id,
                    "orphan_cleanup",
                    (
                        "Terminated process-pool children left after the "
                        f"worker exited: {terminated}"
                    ),
                    {
                        "worker_pid": int(process.pid),
                        "terminated_pids": list(terminated),
                    },
                )
            if (
                exit_code == WORKER_PAUSE_EXIT_CODE
                and record.state == JobState.QUEUED
            ):
                # The worker paused itself (it set PAUSED before exiting with
                # 75) and a resume re-queued the job before this reap ran.
                # Treating that orderly exit as a crash put the queued job back
                # to PAUSED, where nothing resumed it again (A147, 27.-30.09.).
                self.store.append_event(
                    job_id,
                    "worker_paused_exit_after_resume",
                    "Paused worker exited after the job was already queued "
                    "for resume",
                    {"worker_pid": int(process.pid), "exit_code": int(exit_code)},
                )
                continue
            if record.state in (
                JobState.QUEUED,
                JobState.RUNNING,
                JobState.PAUSING,
                JobState.RESUMING,
            ):
                self.store.transition(
                    job_id,
                    JobState.PAUSED,
                    message=(
                        "Worker exited unexpectedly with code "
                        f"{exit_code}; recovered as paused"
                    ),
                    error_message=f"worker exit code {exit_code}",
                )
            elif record.state == JobState.CANCELLING:
                self.store.transition(
                    job_id,
                    JobState.CANCELLED,
                    message=(
                        "Cancellation completed after worker exit "
                        f"(code {exit_code})"
                    ),
                )

    def _cleanup_historical_orphans(self) -> None:
        """Clean process-pool children left by a killed job worker.

        Worker PIDs come from this project's immutable job events. Matching is
        additionally restricted to Python's explicit ``spawn_main`` command
        line, so unrelated local Python processes are never selected.
        """
        for record in self.store.list(limit=10_000):
            worker_pids = {
                int(event["payload"]["pid"])
                for event in self.store.events(record.spec.job_id)
                if event["event_type"] in {"worker_launched", "worker_attached"}
                and int(event.get("payload", {}).get("pid") or 0) > 0
            }
            for worker_pid in sorted(worker_pids):
                if _pid_alive(worker_pid):
                    continue
                terminated = _terminate_process_pool_children(worker_pid)
                if not terminated:
                    continue
                self.store.append_event(
                    record.spec.job_id,
                    "orphan_cleanup",
                    (
                        "Recovered orphaned process-pool children for dead "
                        f"worker {worker_pid}: {terminated}"
                    ),
                    {
                        "worker_pid": worker_pid,
                        "terminated_pids": list(terminated),
                    },
                )


def ensure_supervisor(config: MimicProjectConfig) -> Optional[int]:
    store = JobStore.for_project(config)
    status = store.supervisor_status()
    pid = int(status.get("pid") or 0)
    heartbeat = status.get("heartbeat_at")
    if pid and _pid_alive(pid) and _heartbeat_recent(heartbeat):
        return pid
    runtime_root = create_runtime_snapshot(
        Path(__file__).resolve().parents[2], config.jobs_root / "runtime_snapshots"
    )
    command = [
        sys.executable,
        "-m",
        "vcg_biomarker.mimic_supervisor",
        "--project",
        str(config.project_file),
    ]
    log_path = config.jobs_root / "supervisor.log"
    log_handle = log_path.open("ab")
    kwargs: dict = {
        "cwd": str(runtime_root),
        "env": runtime_environment(),
        "stdin": subprocess.DEVNULL,
        "stdout": log_handle,
        "stderr": subprocess.STDOUT,
        "close_fds": True,
    }
    if os.name == "nt":
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NO_WINDOW
        )
    else:
        kwargs["start_new_session"] = True
    try:
        process = subprocess.Popen(command, **kwargs)
    finally:
        log_handle.close()
    return process.pid


def _lower_io_priority(pid: int) -> None:
    """Best effort: a job's disk traffic must not stall the desktop."""
    if os.name != "nt":
        return
    try:
        import psutil

        psutil.Process(int(pid)).ionice(psutil.IOPRIO_LOW)
    except Exception:
        return


def _heartbeat_recent(value: Optional[str]) -> bool:
    if not value:
        return False
    from datetime import datetime, timezone

    try:
        heartbeat = datetime.fromisoformat(value)
    except ValueError:
        return False
    if heartbeat.tzinfo is None:
        heartbeat = heartbeat.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - heartbeat).total_seconds() < SUPERVISOR_STALE_SECONDS


def _pid_alive(pid: int) -> bool:
    try:
        import psutil

        return psutil.pid_exists(pid)
    except ImportError:
        if os.name == "nt":
            return bool(pid)
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True


_SPAWN_PARENT_PATTERN = re.compile(r"spawn_main\(parent_pid=(\d+),")


def _is_process_pool_child_of(
    command_line: list[str] | tuple[str, ...],
    parent_pid: int,
) -> bool:
    text = " ".join(str(item) for item in command_line)
    match = _SPAWN_PARENT_PATTERN.search(text)
    return bool(match and int(match.group(1)) == int(parent_pid))


def _terminate_process_pool_children(parent_pid: int) -> tuple[int, ...]:
    if int(parent_pid) <= 0:
        return ()
    try:
        import psutil
    except ImportError:
        return ()

    matches = []
    for process in psutil.process_iter(("pid", "cmdline")):
        try:
            command_line = process.info.get("cmdline") or ()
            if _is_process_pool_child_of(command_line, parent_pid):
                matches.append(process)
        except (psutil.AccessDenied, psutil.NoSuchProcess, ValueError):
            continue
    if not matches:
        return ()

    selected_pids = tuple(sorted(int(process.pid) for process in matches))
    for process in matches:
        try:
            process.terminate()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    _, survivors = psutil.wait_procs(matches, timeout=3.0)
    for process in survivors:
        try:
            process.kill()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    psutil.wait_procs(survivors, timeout=2.0)
    return selected_pids
