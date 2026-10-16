from __future__ import annotations

import builtins
import json
import os
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional

from .config import (
    OPERATIONAL_CONFIG_FIELDS,
    MimicProjectConfig,
    operational_config_payload,
)
from .models import JobRecord, JobSpec, JobState, ResourceProfile
from .storage import canonical_json, content_hash
from .presets import STUDY_PRESET_ID, resolve_preset_id


JOB_SCHEMA_VERSION = "1.1.0"
JOB_PROJECT_CONFIG_KEY = "__job_project_config"
JOB_PROJECT_CONFIG_HASH_KEY = "__job_project_config_hash"
_ETA_SUFFIX = re.compile(r"\bETA=[^,;]*$")
_MESSAGE_FRACTION = re.compile(r"(\d[\d,.]*)\s*/\s*(\d[\d,.]*)")


def _localized_count(text: str) -> int:
    """'640,110' or '640.110' -> 640110."""
    digits = re.sub(r"[^\d]", "", text)
    return int(digits) if digits else -1
_RESUME_ETA_MIN_SECONDS = 30.0
_RESUME_ETA_MIN_RECORDS = 250

# Optional data roots. Going from "unset" to "set" while a run is pinned to
# the old snapshot means the run is provably missing an entire evidence
# layer, and its empty outputs would read like assessed absence.
OPTIONAL_SOURCE_CONFIG_FIELDS: tuple[str, ...] = ("mimic_note_root",)


def project_config_drift(
    snapshot: Mapping[str, Any],
    live: Mapping[str, Any],
) -> dict[str, list[str]]:
    """Classify how a live project config differs from a job's pinned snapshot.

    Pinning the scientific configuration per job is deliberate. This function
    does not undo that; it only makes the divergence visible, and separates
    the benign case (resource knobs) from the case where a data source became
    available after the run was pinned.
    """

    operational: list[str] = []
    scientific: list[str] = []
    source_added: list[str] = []
    for key in sorted(set(snapshot) | set(live)):
        before = snapshot.get(key)
        after = live.get(key)
        if before == after:
            continue
        if key in OPERATIONAL_CONFIG_FIELDS:
            operational.append(key)
        elif (
            key in OPTIONAL_SOURCE_CONFIG_FIELDS
            and before in (None, "")
            and after not in (None, "")
        ):
            source_added.append(key)
        else:
            scientific.append(key)
    return {
        "operational": operational,
        "scientific": scientific,
        "source_added": source_added,
    }


def validate_job_run_manifest_root(
    record: JobRecord,
    config: MimicProjectConfig,
    requested_root: str | Path,
) -> Path:
    """Validate the only optional filesystem target accepted by `jobs delete`.

    The function deliberately accepts neither arbitrary run output nor a
    recursive tree.  It is limited to a study-run directory containing exactly
    one guided-workflow manifest whose run identity matches the job.
    """

    if record.spec.stage != "study-run":
        raise ValueError("Only study-run jobs can own a guided run manifest root")
    run_id = str(record.spec.arguments.get("run_id", "")).strip()
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("Job has no safe, single-component run_id")
    runs_root = config.runs_root.resolve()
    expected_root = (runs_root / run_id).resolve()
    requested_path = Path(requested_root)
    if requested_path.is_symlink():
        raise ValueError("Run manifest root must not be a symbolic link")
    root = requested_path.resolve()
    if root != expected_root or root.parent != runs_root:
        raise ValueError(
            f"Run root must exactly match this job's root: {expected_root}"
        )
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"Run manifest root is not a regular directory: {root}")
    entries = tuple(root.iterdir())
    if len(entries) != 1 or entries[0].name != "guided_workflow.manifest.json":
        raise ValueError(
            "Run root deletion is allowed only when the directory contains "
            "guided_workflow.manifest.json and nothing else"
        )
    manifest_path = entries[0]
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("Guided workflow manifest is not a regular file")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    study_spec = dict(payload.get("study_spec") or {})
    if str(study_spec.get("run_id", "")) != run_id:
        raise ValueError("Guided workflow manifest belongs to a different run")
    job_preset = str(record.spec.arguments.get("preset_id", ""))
    manifest_preset = str(study_spec.get("preset_id", ""))
    if job_preset and manifest_preset != job_preset:
        raise ValueError("Guided workflow manifest preset does not match the job")
    return root


def remove_confirmed_job_run_manifest_root(
    record: JobRecord,
    config: MimicProjectConfig,
    requested_root: str | Path,
) -> Path:
    """Remove one already-confirmed manifest file and its now-empty directory."""

    root = validate_job_run_manifest_root(record, config, requested_root)
    manifest_path = root / "guided_workflow.manifest.json"
    manifest_path.unlink()
    root.rmdir()
    return root


# Control flow, not errors: derived from BaseException (like KeyboardInterrupt)
# so that no ``except Exception`` in a stage can swallow them. As
# RuntimeErrors, a pause raised from a progress callback inside the broad-axis
# sweep was caught by its per-axis ``except Exception`` and recorded the axis
# as "failed" -- the stage then finished with that axis silently missing.
# Handlers that mean these name them explicitly (worker, evidence).
class JobPauseRequested(BaseException):
    pass


class JobCancellationRequested(BaseException):
    pass


_TERMINAL_STATES = {
    JobState.CANCELLED,
    JobState.FAILED,
    JobState.COMPLETED,
    JobState.BLOCKED,
}

_DELETABLE_STATES = {
    JobState.PAUSED,
    JobState.CANCELLED,
    JobState.FAILED,
    JobState.COMPLETED,
    JobState.BLOCKED,
}

_ALLOWED_TRANSITIONS: dict[JobState, set[JobState]] = {
    JobState.QUEUED: {
        JobState.RUNNING,
        JobState.PAUSED,
        JobState.CANCELLING,
        JobState.CANCELLED,
        JobState.BLOCKED,
    },
    JobState.RUNNING: {
        JobState.PAUSING,
        JobState.PAUSED,
        JobState.CANCELLING,
        JobState.CANCELLED,
        JobState.FAILED,
        JobState.COMPLETED,
        JobState.BLOCKED,
    },
    JobState.PAUSING: {
        JobState.PAUSED,
        JobState.CANCELLING,
        JobState.CANCELLED,
        JobState.FAILED,
    },
    JobState.PAUSED: {
        JobState.RESUMING,
        JobState.CANCELLING,
        JobState.CANCELLED,
    },
    JobState.RESUMING: {
        JobState.QUEUED,
        JobState.RUNNING,
        JobState.PAUSED,
        JobState.CANCELLING,
        JobState.CANCELLED,
        JobState.FAILED,
    },
    JobState.CANCELLING: {
        JobState.CANCELLED,
        JobState.FAILED,
    },
    JobState.CANCELLED: {JobState.QUEUED},
    JobState.FAILED: {JobState.QUEUED},
    JobState.COMPLETED: set(),
    JobState.BLOCKED: {JobState.QUEUED, JobState.CANCELLED},
}


class JobStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @classmethod
    def for_project(cls, config: MimicProjectConfig) -> "JobStore":
        return cls(config.jobs_database_path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            str(self.path),
            timeout=30.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS vcg_job_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS vcg_jobs (
                    job_id TEXT PRIMARY KEY,
                    spec_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    progress_current INTEGER NOT NULL DEFAULT 0,
                    progress_total INTEGER NOT NULL DEFAULT 0,
                    progress_message TEXT NOT NULL DEFAULT '',
                    worker_pid INTEGER,
                    heartbeat_at TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    error_message TEXT NOT NULL DEFAULT '',
                    artifact_hash TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_vcg_jobs_state_priority
                    ON vcg_jobs(state, priority DESC, created_at);
                CREATE TABLE IF NOT EXISTS vcg_job_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(job_id) REFERENCES vcg_jobs(job_id)
                );
                CREATE INDEX IF NOT EXISTS idx_vcg_job_events_job
                    ON vcg_job_events(job_id, event_id);
                CREATE TABLE IF NOT EXISTS vcg_supervisor (
                    supervisor_id INTEGER PRIMARY KEY CHECK(supervisor_id = 1),
                    pid INTEGER,
                    heartbeat_at TEXT,
                    started_at TEXT,
                    project_file TEXT
                );
                """
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO vcg_job_metadata(key, value)
                VALUES ('schema_version', ?)
                """,
                [JOB_SCHEMA_VERSION],
            )

    def enqueue(self, spec: JobSpec) -> JobRecord:
        self._require_study_runtime(spec)
        project_config = dict(
            spec.arguments.get(JOB_PROJECT_CONFIG_KEY, {})
        )
        project_config_hash = str(
            spec.arguments.get(JOB_PROJECT_CONFIG_HASH_KEY, "")
        )
        if bool(project_config) != bool(project_config_hash):
            raise ValueError(
                "Job project configuration and its hash must be supplied "
                "together"
            )
        if (
            project_config
            and content_hash(project_config) != project_config_hash
        ):
            raise ValueError("Job project configuration hash is invalid")
        now = _utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """
                    INSERT INTO vcg_jobs(
                        job_id, spec_json, state, priority, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        spec.job_id,
                        canonical_json(spec.to_dict()),
                        JobState.QUEUED.value,
                        spec.priority,
                        spec.created_at,
                        now,
                    ],
                )
                self._insert_event(
                    connection,
                    spec.job_id,
                    "enqueued",
                    f"Stage {spec.stage!r} queued",
                    {
                        "resource_profile": (
                            spec.resource_profile.to_dict()
                        ),
                        "project_config_hash": (
                            project_config_hash
                        ),
                    },
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return self.get(spec.job_id)

    def create(
        self,
        *,
        stage: str,
        project_file: str | Path,
        arguments: Optional[Mapping[str, Any]] = None,
        resource_profile: Optional[ResourceProfile] = None,
        priority: int = 50,
        requested_by: str = "local_user",
        job_id: Optional[str] = None,
    ) -> JobRecord:
        identifier = job_id or (
            f"{stage}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-"
            f"{uuid.uuid4().hex[:8]}"
        )
        resolved_arguments = dict(arguments or {})
        if (
            JOB_PROJECT_CONFIG_KEY in resolved_arguments
            or JOB_PROJECT_CONFIG_HASH_KEY in resolved_arguments
        ):
            raise ValueError(
                "Reserved job project-snapshot arguments cannot be supplied "
                "by a caller"
            )
        resolved_profile = resource_profile
        project_config: dict[str, Any] = {}
        project_config_hash = ""
        try:
            config = MimicProjectConfig.load(project_file)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            if resolved_profile is None:
                resolved_profile = default_resource_profile(stage)
        else:
            project_config = config.to_dict()
            project_config_hash = content_hash(project_config)
            if resolved_profile is None:
                resolved_profile = resource_profile_for_config(
                    stage,
                    config,
                    resolved_arguments,
                )
        if project_config:
            resolved_arguments[JOB_PROJECT_CONFIG_KEY] = project_config
            resolved_arguments[
                JOB_PROJECT_CONFIG_HASH_KEY
            ] = project_config_hash
        return self.enqueue(
            JobSpec(
                job_id=identifier,
                stage=stage,
                project_file=str(Path(project_file).resolve()),
                arguments=resolved_arguments,
                resource_profile=resolved_profile,
                priority=priority,
                requested_by=requested_by,
            )
        )

    def get(self, job_id: str) -> JobRecord:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM vcg_jobs WHERE job_id = ?",
                [job_id],
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown job: {job_id}")
        return _row_to_record(row)

    def list(
        self,
        *,
        states: Optional[Iterable[JobState | str]] = None,
        limit: int = 500,
    ) -> tuple[JobRecord, ...]:
        query = "SELECT * FROM vcg_jobs"
        parameters: list[Any] = []
        if states:
            values = [
                item.value if isinstance(item, JobState) else str(item)
                for item in states
            ]
            query += f" WHERE state IN ({','.join('?' for _ in values)})"
            parameters.extend(values)
        query += " ORDER BY priority DESC, created_at DESC LIMIT ?"
        parameters.append(max(1, int(limit)))
        with self.connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return tuple(_row_to_record(row) for row in rows)

    def events(
        self,
        job_id: str,
        *,
        after_event_id: int = 0,
    ) -> builtins.list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT event_id, created_at, event_type, message, payload_json
                FROM vcg_job_events
                WHERE job_id = ? AND event_id > ?
                ORDER BY event_id
                """,
                [job_id, int(after_event_id)],
            ).fetchall()
        return [
            {
                "event_id": int(row["event_id"]),
                "created_at": row["created_at"],
                "event_type": row["event_type"],
                "message": row["message"],
                "payload": json.loads(row["payload_json"]),
            }
            for row in rows
        ]

    def latest_runtime_stage(self, job_id: str) -> Optional[dict[str, Any]]:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT event_id, created_at, message, payload_json
                FROM vcg_job_events
                WHERE job_id = ? AND event_type = 'runtime_stage'
                ORDER BY event_id DESC
                LIMIT 1
                """,
                [job_id],
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(row["payload_json"])
        payload.update(
            {
                "event_id": int(row["event_id"]),
                "created_at": row["created_at"],
                "message": row["message"],
            }
        )
        return payload

    def transition(
        self,
        job_id: str,
        target: JobState,
        *,
        message: str = "",
        worker_pid: Optional[int] = None,
        error_message: Optional[str] = None,
        artifact_hash: Optional[str] = None,
    ) -> JobRecord:
        now = _utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT state FROM vcg_jobs WHERE job_id = ?",
                    [job_id],
                ).fetchone()
                if row is None:
                    raise KeyError(f"Unknown job: {job_id}")
                current = JobState(row["state"])
                if target != current and target not in _ALLOWED_TRANSITIONS[current]:
                    raise ValueError(
                        f"Invalid job transition {current.value} -> {target.value}"
                    )
                assignments = ["state = ?", "updated_at = ?"]
                parameters: list[Any] = [target.value, now]
                if (
                    worker_pid is not None
                    or target in _TERMINAL_STATES
                    or target == JobState.PAUSED
                ):
                    assignments.append("worker_pid = ?")
                    parameters.append(worker_pid)
                if target == JobState.RUNNING and current != JobState.RUNNING:
                    assignments.extend(["started_at = COALESCE(started_at, ?)", "heartbeat_at = ?"])
                    parameters.extend([now, now])
                if target in _TERMINAL_STATES:
                    assignments.append("finished_at = ?")
                    parameters.append(now)
                elif target in (JobState.QUEUED, JobState.RESUMING):
                    assignments.append("finished_at = NULL")
                if error_message is not None:
                    assignments.append("error_message = ?")
                    parameters.append(error_message)
                if artifact_hash is not None:
                    assignments.append("artifact_hash = ?")
                    parameters.append(artifact_hash)
                parameters.append(job_id)
                connection.execute(
                    f"UPDATE vcg_jobs SET {', '.join(assignments)} WHERE job_id = ?",
                    parameters,
                )
                self._insert_event(
                    connection,
                    job_id,
                    "state",
                    message or f"{current.value} -> {target.value}",
                    {"from": current.value, "to": target.value},
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return self.get(job_id)

    def request_pause(
        self,
        job_id: str,
        *,
        message: str = "Pause requested",
        diagnostic: Optional[str] = None,
    ) -> JobRecord:
        record = self.get(job_id)
        if record.state == JobState.QUEUED:
            return self.transition(
                job_id,
                JobState.PAUSED,
                message="Paused before start",
                error_message=diagnostic,
            )
        # Asking a job that is already stopping to stop is not an error. The
        # UI polls on an interval, so a second click -- or a click that lands
        # just after the worker picked the request up -- is ordinary, and
        # answering it with an exception makes a no-op look like a failure.
        if record.state in {JobState.PAUSING, JobState.PAUSED}:
            return record
        if record.state != JobState.RUNNING:
            raise ValueError(
                f"Nur laufende Aufgaben koennen pausiert werden; diese ist "
                f"'{record.state.value}'."
            )
        return self.transition(
            job_id,
            JobState.PAUSING,
            message=message,
            error_message=diagnostic,
        )

    def request_resume(self, job_id: str) -> JobRecord:
        record = self.get(job_id)
        # Already on its way back: nothing to do, and nothing wrong either.
        if record.state in {JobState.RESUMING, JobState.QUEUED, JobState.RUNNING}:
            return record
        if record.state != JobState.PAUSED:
            raise ValueError(
                f"Nur pausierte Aufgaben koennen fortgesetzt werden; diese ist "
                f"'{record.state.value}'."
            )
        self._bind_legacy_project_snapshot(job_id)
        self._rebind_operational_project_config(job_id)
        self.transition(job_id, JobState.RESUMING, message="Resume requested")
        return self.transition(
            job_id,
            JobState.QUEUED,
            message="Queued for resume",
            error_message="",
        )

    def request_cancel(self, job_id: str) -> JobRecord:
        record = self.get(job_id)
        if record.state in _TERMINAL_STATES:
            return record
        if record.state in (JobState.QUEUED, JobState.PAUSED, JobState.RESUMING):
            return self.transition(job_id, JobState.CANCELLED, message="Cancelled")
        return self.transition(job_id, JobState.CANCELLING, message="Cancellation requested")

    def retry(self, job_id: str) -> JobRecord:
        record = self.get(job_id)
        if record.state not in (JobState.FAILED, JobState.CANCELLED, JobState.BLOCKED):
            raise ValueError(f"Job {job_id} cannot be retried from {record.state.value}")
        self._bind_legacy_project_snapshot(job_id)
        self._rebind_operational_project_config(job_id)
        return self.transition(
            job_id,
            JobState.QUEUED,
            message="Retry queued",
            error_message="",
        )

    def _require_study_runtime(self, spec: JobSpec) -> None:
        if spec.stage == "study-run":
            from .runtime_preflight import require_isolated_study_supervisor

            require_isolated_study_supervisor(
                self.supervisor_status(), self.path.parent / "runtime_snapshots"
            )

    def _bind_legacy_project_snapshot(self, job_id: str) -> JobRecord:
        self._require_study_runtime(self.get(job_id).spec)
        record = self.get(job_id)
        arguments = dict(record.spec.arguments)
        recorded_config = dict(
            arguments.get(JOB_PROJECT_CONFIG_KEY, {})
        )
        recorded_hash = str(
            arguments.get(JOB_PROJECT_CONFIG_HASH_KEY, "")
        )
        if recorded_config:
            if (
                content_hash(recorded_config) != recorded_hash
            ):
                raise RuntimeError(
                    f"Job {job_id} has a corrupted project snapshot"
                )
            return record
        config = MimicProjectConfig.load(record.spec.project_file)
        project_config = config.to_dict()
        project_config_hash = content_hash(project_config)
        spec_payload = record.spec.to_dict()
        arguments[JOB_PROJECT_CONFIG_KEY] = project_config
        arguments[JOB_PROJECT_CONFIG_HASH_KEY] = project_config_hash
        spec_payload["arguments"] = arguments
        now = _utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """
                    UPDATE vcg_jobs
                    SET spec_json = ?, updated_at = ?
                    WHERE job_id = ?
                    """,
                    [
                        canonical_json(spec_payload),
                        now,
                        job_id,
                    ],
                )
                self._insert_event(
                    connection,
                    job_id,
                    "project_snapshot_bound",
                    "Legacy job bound to the current project configuration",
                    {"project_config_hash": project_config_hash},
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return self.get(job_id)

    def _rebind_operational_project_config(self, job_id: str) -> JobRecord:
        """Refresh resource knobs before a paused or failed job continues.

        Scientific settings remain pinned to the original snapshot.  Updating
        the persisted JobSpec is important because the worker applies the
        resource profile before it reconstructs the runtime configuration.
        """

        record = self.get(job_id)
        rebindable_states = {
            JobState.PAUSED,
            JobState.FAILED,
            JobState.CANCELLED,
            JobState.BLOCKED,
        }
        if record.state not in rebindable_states:
            raise ValueError(
                "Operational settings may only be rebound while a job is "
                "paused or awaiting retry"
            )
        arguments = dict(record.spec.arguments)
        snapshot = dict(arguments.get(JOB_PROJECT_CONFIG_KEY, {}))
        snapshot_hash = str(arguments.get(JOB_PROJECT_CONFIG_HASH_KEY, ""))
        if not snapshot or content_hash(snapshot) != snapshot_hash:
            raise RuntimeError(f"Job {job_id} has a corrupted project snapshot")

        live_config = MimicProjectConfig.load(record.spec.project_file)
        live_payload = live_config.to_dict()
        before_operational = operational_config_payload(snapshot)
        after_operational = operational_config_payload(live_payload)
        rebound_snapshot = dict(snapshot)
        rebound_snapshot.update(after_operational)
        rebound_config = MimicProjectConfig.from_dict(rebound_snapshot)
        stage_arguments = {
            key: value
            for key, value in arguments.items()
            if key not in (JOB_PROJECT_CONFIG_KEY, JOB_PROJECT_CONFIG_HASH_KEY)
        }
        rebound_profile = resource_profile_for_config(
            record.spec.stage,
            rebound_config,
            stage_arguments,
        )
        rebound_hash = content_hash(rebound_snapshot)
        arguments[JOB_PROJECT_CONFIG_KEY] = rebound_snapshot
        arguments[JOB_PROJECT_CONFIG_HASH_KEY] = rebound_hash
        rebound_spec = replace(
            record.spec,
            arguments=arguments,
            resource_profile=rebound_profile,
        )
        now = _utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = connection.execute(
                    """
                    UPDATE vcg_jobs
                    SET spec_json = ?, updated_at = ?
                    WHERE job_id = ? AND state = ?
                    """,
                    [
                        canonical_json(rebound_spec.to_dict()),
                        now,
                        job_id,
                        record.state.value,
                    ],
                )
                if cursor.rowcount != 1:
                    raise RuntimeError(
                        f"Job {job_id} changed state during resource rebind"
                    )
                self._insert_event(
                    connection,
                    job_id,
                    "operational_config_rebound",
                    "Job rebound to current operational resource settings",
                    {
                        "before": before_operational,
                        "after": after_operational,
                        "resource_profile_before": (
                            record.spec.resource_profile.to_dict()
                        ),
                        "resource_profile_after": rebound_profile.to_dict(),
                        "scientific_snapshot_preserved": True,
                    },
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return self.get(job_id)

    def delete(self, job_id: str) -> JobRecord:
        """Atomically remove a non-active job and all of its event rows."""

        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM vcg_jobs WHERE job_id = ?",
                    [job_id],
                ).fetchone()
                if row is None:
                    raise KeyError(f"Unknown job: {job_id}")
                record = _row_to_record(row)
                if record.state not in _DELETABLE_STATES:
                    raise ValueError(
                        f"Job {job_id} cannot be deleted from "
                        f"{record.state.value}; pause or cancel it first"
                    )
                connection.execute(
                    "DELETE FROM vcg_job_events WHERE job_id = ?",
                    [job_id],
                )
                cursor = connection.execute(
                    "DELETE FROM vcg_jobs WHERE job_id = ? AND state = ?",
                    [job_id, record.state.value],
                )
                if cursor.rowcount != 1:
                    raise RuntimeError(
                        f"Job {job_id} changed state during deletion"
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return record

    def claim(self, job_id: str, worker_pid: int) -> JobRecord:
        return self.transition(
            job_id,
            JobState.RUNNING,
            message=f"Claimed by worker {worker_pid}",
            worker_pid=worker_pid,
        )

    def reserve_for_launch(
        self,
        job_id: str,
        *,
        supervisor_pid: int,
    ) -> Optional[JobRecord]:
        """Atomically reserve a queued job before creating its worker process."""
        now = _utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                updated = connection.execute(
                    """
                    UPDATE vcg_jobs
                    SET state = ?,
                        worker_pid = ?,
                        heartbeat_at = ?,
                        started_at = COALESCE(started_at, ?),
                        updated_at = ?
                    WHERE job_id = ? AND state = ?
                    """,
                    [
                        JobState.RUNNING.value,
                        int(supervisor_pid),
                        now,
                        now,
                        now,
                        job_id,
                        JobState.QUEUED.value,
                    ],
                )
                if updated.rowcount != 1:
                    connection.rollback()
                    return None
                self._insert_event(
                    connection,
                    job_id,
                    "reserved",
                    f"Reserved for launch by supervisor {supervisor_pid}",
                    {"supervisor_pid": int(supervisor_pid)},
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return self.get(job_id)

    def attach_worker(self, job_id: str, worker_pid: int) -> None:
        now = _utc_now()
        with self.connect() as connection:
            updated = connection.execute(
                """
                UPDATE vcg_jobs
                SET worker_pid = ?, heartbeat_at = ?, updated_at = ?
                WHERE job_id = ? AND state = ?
                """,
                [
                    int(worker_pid),
                    now,
                    now,
                    job_id,
                    JobState.RUNNING.value,
                ],
            )
            if updated.rowcount == 1:
                self._insert_event(
                    connection,
                    job_id,
                    "worker_attached",
                    f"Worker process {worker_pid} attached",
                    {"pid": int(worker_pid)},
                )

    def heartbeat(
        self,
        job_id: str,
        *,
        progress_current: Optional[int] = None,
        progress_total: Optional[int] = None,
        progress_message: Optional[str] = None,
    ) -> None:
        assignments = ["heartbeat_at = ?", "updated_at = ?"]
        now = _utc_now()
        values: list[Any] = [now, now]
        if progress_current is not None:
            assignments.append("progress_current = ?")
            values.append(max(0, int(progress_current)))
        if progress_total is not None:
            assignments.append("progress_total = ?")
            values.append(max(0, int(progress_total)))
        if progress_message is not None:
            assignments.append("progress_message = ?")
            values.append(str(progress_message))
        values.append(job_id)
        with self.connect() as connection:
            connection.execute(
                f"UPDATE vcg_jobs SET {', '.join(assignments)} WHERE job_id = ?",
                values,
            )

    def set_priority(self, job_id: str, priority: int) -> JobRecord:
        value = int(priority)
        if not 0 <= value <= 100:
            raise ValueError("priority must be between 0 and 100")
        record = self.get(job_id)
        if record.state in _TERMINAL_STATES:
            raise ValueError(
                f"Priority cannot change after {record.state.value}"
            )
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE vcg_jobs
                SET priority = ?, updated_at = ?
                WHERE job_id = ?
                """,
                [value, _utc_now(), job_id],
            )
            self._insert_event(
                connection,
                job_id,
                "priority",
                f"Priority changed to {value}",
                {"priority": value},
            )
        return self.get(job_id)

    def append_event(
        self,
        job_id: str,
        event_type: str,
        message: str,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> None:
        with self.connect() as connection:
            self._insert_event(
                connection,
                job_id,
                event_type,
                message,
                payload or {},
            )

    def control_state(self, job_id: str) -> JobState:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT state FROM vcg_jobs WHERE job_id = ?",
                [job_id],
            ).fetchone()
        if row is None:
            raise KeyError(job_id)
        return JobState(row["state"])

    def supervisor_heartbeat(self, project_file: str | Path) -> None:
        now = _utc_now()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO vcg_supervisor(
                    supervisor_id, pid, heartbeat_at, started_at, project_file
                ) VALUES (
                    1, ?, ?, COALESCE(
                        (SELECT started_at FROM vcg_supervisor WHERE supervisor_id=1),
                        ?
                    ), ?
                )
                """,
                [os.getpid(), now, now, str(Path(project_file).resolve())],
            )

    def supervisor_status(self) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM vcg_supervisor WHERE supervisor_id=1"
            ).fetchone()
        return dict(row) if row is not None else {}

    def recover_stale_jobs(self, *, stale_after_seconds: int = 90) -> tuple[str, ...]:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=stale_after_seconds)
        recovered: list[str] = []
        for record in self.list(states=(JobState.RUNNING, JobState.PAUSING, JobState.CANCELLING)):
            heartbeat = _parse_time(record.heartbeat_at)
            worker_alive = bool(
                record.worker_pid and _pid_is_alive(record.worker_pid)
            )
            if worker_alive:
                continue
            if record.worker_pid is None and heartbeat is not None and heartbeat >= cutoff:
                continue
            if record.state == JobState.CANCELLING:
                self.transition(record.spec.job_id, JobState.CANCELLED, message="Recovered after worker exit")
            elif record.state == JobState.PAUSING:
                self.transition(record.spec.job_id, JobState.PAUSED, message="Recovered paused job")
            else:
                self.transition(
                    record.spec.job_id,
                    JobState.PAUSED,
                    message=(
                        "Worker process disappeared; recovered as paused. "
                        "Atomic outputs remain resumable."
                    ),
                    error_message="worker process is no longer alive",
                )
            recovered.append(record.spec.job_id)
        return tuple(recovered)

    @staticmethod
    def _insert_event(
        connection: sqlite3.Connection,
        job_id: str,
        event_type: str,
        message: str,
        payload: Mapping[str, Any],
    ) -> None:
        connection.execute(
            """
            INSERT INTO vcg_job_events(
                job_id, created_at, event_type, message, payload_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [job_id, _utc_now(), event_type, message, canonical_json(dict(payload))],
        )


class JobControl:
    def __init__(
        self,
        store: JobStore,
        job_id: str,
        *,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.store = store
        self.job_id = job_id
        self._clock = clock
        record = store.get(job_id)
        self._session_baseline = max(0, int(record.progress_current))
        self._session_started = float(clock())
        self._resumed = bool(
            self._session_baseline > 0
            and (
                record.progress_total <= 0
                or self._session_baseline < record.progress_total
            )
        )

    def checkpoint(
        self,
        message: str = "",
        *,
        current: Optional[int] = None,
        total: Optional[int] = None,
    ) -> None:
        progress_message = self._resume_aware_message(
            message,
            current=current,
            total=total,
        )
        self.store.heartbeat(
            self.job_id,
            progress_current=current,
            progress_total=total,
            progress_message=progress_message or None,
        )
        if progress_message:
            self.store.append_event(
                self.job_id,
                "progress",
                progress_message,
            )
        state = self.store.control_state(self.job_id)
        if state == JobState.PAUSING:
            raise JobPauseRequested("Pause requested at an atomic checkpoint")
        if state == JobState.CANCELLING:
            raise JobCancellationRequested("Cancellation requested")

    def set_runtime_stage(
        self,
        stage: str,
        *,
        state: str = "running",
        memory_policy: Optional[Mapping[str, Any]] = None,
    ) -> None:
        normalized = str(stage).strip()
        payload: dict[str, Any] = {
            "stage": normalized,
            "state": str(state),
        }
        if memory_policy is not None:
            payload["memory_policy"] = dict(memory_policy)
        self.store.append_event(
            self.job_id,
            "runtime_stage",
            f"Runtime stage {normalized}: {state}",
            payload,
        )

    def _resume_aware_message(
        self,
        message: str,
        *,
        current: Optional[int],
        total: Optional[int],
    ) -> str:
        if (
            not message
            or not self._resumed
            or current is None
            or total is None
            or _ETA_SUFFIX.search(message) is None
        ):
            return message
        # The rewrite is only valid when the checkpoint counts the same thing
        # the message counts. A guided study checkpoints weighted workflow
        # units (43,573/488,650 in A131) while the message is the stage's own
        # "Features 168,575/640,110; ... ETA=...". Rewriting that ETA from
        # workflow units replaced the extraction's correct ~79 h with a
        # whole-run forecast of ~370 h under the stage's label.
        stated = _MESSAGE_FRACTION.search(message)
        if stated is not None and (
            _localized_count(stated.group(1)) != int(current)
            or _localized_count(stated.group(2)) != int(total)
        ):
            return message
        done = max(0, int(current))
        target = max(0, int(total))
        if done >= target:
            return _ETA_SUFFIX.sub("ETA=0m 00s", message)
        elapsed = max(0.0, float(self._clock()) - self._session_started)
        completed_this_session = max(0, done - self._session_baseline)
        if (
            elapsed < _RESUME_ETA_MIN_SECONDS
            or completed_this_session < _RESUME_ETA_MIN_RECORDS
        ):
            return _ETA_SUFFIX.sub(
                "ETA=wird neu kalibriert",
                message,
            )
        rate = completed_this_session / elapsed
        if rate <= 0:
            return _ETA_SUFFIX.sub(
                "ETA=wird neu kalibriert",
                message,
            )
        return _ETA_SUFFIX.sub(
            f"ETA={_format_job_duration((target - done) / rate)}",
            message,
        )


def _format_job_duration(seconds: float) -> str:
    remaining = max(0, int(seconds))
    hours, remainder = divmod(remaining, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m {seconds:02d}s"


def default_resource_profile(stage: str) -> ResourceProfile:
    if stage == "study-run":
        return ResourceProfile(
            cpu_tokens=6,
            memory_gb=14.0,
            io_weight=3,
            writer_namespace="guided-study",
        )
    if stage == "extract-features":
        return ResourceProfile(
            cpu_tokens=6,
            memory_gb=12.0,
            io_weight=3,
            writer_namespace="features",
        )
    if stage in (
        "validate",
        "discover-phenome",
        "refresh-phenome-promotions",
        "group-vcg",
    ):
        return ResourceProfile(
            cpu_tokens=2,
            memory_gb=6.0,
            io_weight=1,
            writer_namespace="runs",
            interactive=True,
        )
    if stage == "screen-broad-axes":
        # Holds one split's marker matrix plus its ranks in memory; the tests
        # themselves are cheap once those exist.
        return ResourceProfile(
            cpu_tokens=6,
            memory_gb=14.0,
            io_weight=3,
            writer_namespace="runs",
        )
    if stage == "deep-combination-search":
        # Reine Rechenlast auf einem Merkmalsrahmen, der in den Speicher passt:
        # viele Kerne, wenig Arbeitsspeicher.
        return ResourceProfile(
            cpu_tokens=8,
            memory_gb=10.0,
            io_weight=1,
            writer_namespace="runs",
        )
    if stage in ("external-validation", "population-cohorts"):
        # Feature extraction over external recordings. CODE-15/SaMi-Trop now
        # use spawned record workers too, so reserving the one-CPU fallback
        # while starting six processes would oversubscribe the supervisor.
        return ResourceProfile(
            cpu_tokens=6,
            memory_gb=8.0,
            io_weight=3,
            writer_namespace="external",
        )
    if stage == "repeatability":
        # A self-join over the master index plus a trajectory pass over every
        # subject: DuckDB-bound, not single-threaded.
        return ResourceProfile(
            cpu_tokens=4,
            memory_gb=10.0,
            io_weight=2,
            writer_namespace="runs",
        )
    if stage == "attest-labels":
        return ResourceProfile(
            cpu_tokens=1,
            memory_gb=2.0,
            io_weight=1,
            writer_namespace="label-attestation",
            interactive=True,
        )
    if stage == "ingest-events":
        return ResourceProfile(
            cpu_tokens=2,
            memory_gb=3.0,
            io_weight=3,
            writer_namespace="clinical-data",
        )
    if stage == "attest-event-evidence":
        return ResourceProfile(
            cpu_tokens=2,
            memory_gb=8.0,
            io_weight=3,
            writer_namespace="evidence-migration",
        )
    if stage in (
        "ingest",
        "build-index",
        "build-evidence",
        "build-labels",
    ):
        return ResourceProfile(
            cpu_tokens=2,
            memory_gb=8.0,
            io_weight=3,
            writer_namespace=stage,
        )
    if stage in (
        "report",
        "report-build",
        "audit",
        "audit-biomarkers",
        "validate-delineation",
    ):
        return ResourceProfile(
            cpu_tokens=2 if stage == "validate-delineation" else 1,
            memory_gb=4.0 if stage == "validate-delineation" else 3.0,
            io_weight=2 if stage == "validate-delineation" else 1,
            writer_namespace=(
                "biomarker-audit"
                if stage == "audit-biomarkers"
                else "delineation-benchmark"
                if stage == "validate-delineation"
                else "reports"
            ),
            interactive=True,
        )
    return ResourceProfile(cpu_tokens=1, memory_gb=2.0, io_weight=1)


def resource_profile_for_config(
    stage: str,
    config: MimicProjectConfig,
    arguments: Optional[Mapping[str, Any]] = None,
) -> ResourceProfile:
    from .resources import (
        cpu_token_capacity,
        detect_hardware,
        global_memory_budget_gb,
    )

    args = dict(arguments or {})
    hardware = detect_hardware()
    cpu_capacity = cpu_token_capacity(hardware)
    memory_capacity = global_memory_budget_gb(
        hardware,
        reserve_gb=config.os_memory_reserve_gb,
    )
    if stage == "study-run":
        run_id = str(args.get("run_id", "mimic-development-guided"))
        preset_id = resolve_preset_id(
            str(args.get("preset_id", STUDY_PRESET_ID))
        )
        workers = max(1, min(int(config.workers), cpu_capacity))
        if preset_id == "exhaustive_research":
            # Exhaustive changes scientific breadth, not the user's resource
            # ceiling.  Respect the live operational settings so a resumed job
            # cannot silently jump back to all logical cores and 24 GB RAM.
            workers = max(1, min(int(config.workers), cpu_capacity))
            memory_gb = min(
                memory_capacity,
                max(1.0, float(config.duckdb_memory_limit_gb)),
            )
        else:
            memory_gb = min(memory_capacity, max(10.0, workers * 2.0))
        return ResourceProfile(
            cpu_tokens=workers,
            memory_gb=memory_gb,
            io_weight=3,
            writer_namespace=f"guided-study:{run_id}",
        )
    if stage == "extract-features":
        workers = max(1, min(int(config.workers), cpu_capacity))
        panels = "_".join(
            str(item)
            for item in args.get(
                "panel_ids",
                ("primary_core", "sentinel"),
            )
        )
        if args.get("locked_test_only"):
            scope = "locked-test"
        elif args.get("max_records") not in (None, ""):
            scope = f"smoke-{int(args['max_records'])}"
        else:
            scope = "development"
        return ResourceProfile(
            cpu_tokens=workers,
            memory_gb=min(memory_capacity, max(8.0, workers * 2.0)),
            io_weight=3,
            writer_namespace=f"features:{panels}:{scope}",
        )
    if stage == "benchmark":
        candidates = tuple(
            int(item)
            for item in args.get("candidates", (1, 2, 4, 6, 7))
        )
        workers = min(
            cpu_capacity,
            max((item for item in candidates if item > 0), default=1),
        )
        return ResourceProfile(
            cpu_tokens=max(1, workers),
            memory_gb=min(memory_capacity, max(4.0, workers * 1.25)),
            io_weight=2,
            writer_namespace="resource-benchmark",
        )
    if stage in (
        "external-validation",
        "population-cohorts",
        "deep-combination-search",
    ):
        workers = max(
            1,
            min(
                int(args.get("workers", config.workers)),
                cpu_capacity,
            ),
        )
        run_id = str(args.get("run_id", "external"))
        return ResourceProfile(
            cpu_tokens=workers,
            memory_gb=min(
                memory_capacity,
                10.0
                if stage == "deep-combination-search"
                else max(8.0, workers * 1.25),
            ),
            io_weight=(
                1 if stage == "deep-combination-search" else 3
            ),
            writer_namespace=(
                f"run:{run_id}"
                if stage == "deep-combination-search"
                else f"external:{run_id}"
            ),
        )
    if stage == "validate-delineation":
        requested = max(1, min(2, cpu_capacity))
        return ResourceProfile(
            cpu_tokens=requested,
            memory_gb=min(memory_capacity, 5.0),
            io_weight=2,
            writer_namespace="delineation-benchmark",
            interactive=True,
        )
    if stage in ("validate", "discover-phenome"):
        run_id = str(args.get("run_id", "mimic-development"))
        interactive_preview = bool(args.get("interactive_preview", False))
        cpu_tokens = (
            min(
                2,
                max(1, hardware.logical_cores - cpu_capacity),
            )
            if interactive_preview
            else max(1, min(int(config.workers), cpu_capacity))
        )
        return ResourceProfile(
            cpu_tokens=cpu_tokens,
            memory_gb=min(
                memory_capacity,
                max(6.0, cpu_tokens * 1.5),
            ),
            io_weight=1,
            writer_namespace=f"run:{run_id}",
            interactive=interactive_preview,
        )
    if stage in (
        "group-vcg",
        "refresh-phenome-promotions",
        "freeze",
        "report",
        "report-build",
        "audit",
    ):
        run_id = str(args.get("run_id", "mimic-development"))
        interactive_cpu_capacity = min(
            2,
            max(1, hardware.logical_cores - cpu_capacity),
        )
        interactive = stage in (
            "validate",
            "discover-phenome",
            "refresh-phenome-promotions",
            "group-vcg",
            "report",
            "report-build",
            "audit",
        )
        if stage == "group-vcg":
            endpoint_id = str(args.get("endpoint_id", "endpoint"))
            split = str(args.get("split", "validation"))
            writer_namespace = (
                f"group-vcg:{run_id}:{endpoint_id}:{split}"
            )
        else:
            writer_namespace = f"run:{run_id}"
        return ResourceProfile(
            cpu_tokens=interactive_cpu_capacity,
            memory_gb=min(
                memory_capacity,
                6.0 if stage in ("validate", "discover-phenome") else 3.0,
            ),
            io_weight=1,
            writer_namespace=writer_namespace,
            interactive=interactive,
        )
    if stage == "attest-labels":
        return ResourceProfile(
            cpu_tokens=1,
            memory_gb=min(memory_capacity, 2.0),
            io_weight=1,
            writer_namespace="label-attestation",
            interactive=True,
        )
    if stage == "ingest-events":
        return ResourceProfile(
            cpu_tokens=max(1, min(int(config.workers), cpu_capacity)),
            memory_gb=min(memory_capacity, 3.0),
            io_weight=3,
            writer_namespace="clinical-data",
        )
    if stage == "attest-event-evidence":
        return ResourceProfile(
            cpu_tokens=max(1, min(int(config.workers), cpu_capacity)),
            memory_gb=min(memory_capacity, 8.0),
            io_weight=3,
            writer_namespace="evidence-migration",
        )
    if stage in ("therapy-response", "physiological-gap-markers"):
        # Both read the whole feature glob -- 1,527 partitions of 1,889
        # columns.  Neither was listed above, so both fell through to the
        # 1 CPU / 2 GB default.  One therapy-response query then spent 25.9
        # hours producing 3,357 rows before the machine ran out of memory.
        #
        # Two corrections to what this comment first claimed, both measured
        # afterwards and both worth keeping visible:
        #
        # 1. **This branch does not run for the guided study.**  A guided run
        #    is one job carrying a single ``study-run`` profile; the per-stage
        #    profiles here are never consulted for it.  Raising the number
        #    changed nothing, and saying otherwise sent me looking in the
        #    wrong place twice.  The cap that actually applies per stage comes
        #    from ``resources.stage_worker_memory_scope``.  This branch still
        #    matters for a stage started on its own.
        #
        # 2. **The read convention *was* the defect.**  This said it was not,
        #    on the grounds that other stages read the same glob and survive.
        #    Measured on the real query at full size, with an 8 GB DuckDB
        #    limit: ``union_by_name=true`` took 109 s and held 19.36 GB
        #    resident, leaving 0.22 GB free; without it, the identical 3,357
        #    rows came back in 9.5 s at 2.11 GB.  Over 11 GB of that sits
        #    outside DuckDB's accounting, which is why an 8 GB limit did not
        #    contain it.  See ``storage.feature_parquet_clause``.  After the
        #    fix this stage finishes in six seconds at 0.27 GB.
        #
        # 10 GB rather than a worker-scaled figure: validate reads the same
        # glob at 10.5 GB and has run for days, so that is the level this
        # machine is known to sustain.
        return ResourceProfile(
            cpu_tokens=max(1, min(int(config.workers), cpu_capacity)),
            memory_gb=min(memory_capacity, 10.0),
            io_weight=3,
            writer_namespace="clinical-data",
        )
    if stage in (
        "ingest",
        "build-index",
        "build-evidence",
        "build-labels",
        "parse-reports",
        "parse-notes",
    ):
        # These are one-shot DuckDB-internal-parallelism batch stages, same
        # shape as extract-features -- scale with the user's configured
        # worker count instead of a hardware-independent fixed cap.
        return ResourceProfile(
            cpu_tokens=max(1, min(int(config.workers), cpu_capacity)),
            memory_gb=min(memory_capacity, 8.0),
            io_weight=3,
            writer_namespace="clinical-data",
        )
    return default_resource_profile(stage)


def _row_to_record(row: sqlite3.Row) -> JobRecord:
    payload = json.loads(row["spec_json"])
    profile_payload = payload.pop("resource_profile", {})
    payload["arguments"] = dict(payload.get("arguments", {}))
    payload["priority"] = int(row["priority"])
    spec = JobSpec(
        **payload,
        resource_profile=ResourceProfile(**profile_payload),
    )
    return JobRecord(
        spec=spec,
        state=JobState(row["state"]),
        progress_current=int(row["progress_current"]),
        progress_total=int(row["progress_total"]),
        progress_message=row["progress_message"],
        worker_pid=row["worker_pid"],
        heartbeat_at=row["heartbeat_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        error_message=row["error_message"],
        artifact_hash=row["artifact_hash"],
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        import psutil

        return psutil.pid_exists(pid)
    except ImportError:
        if os.name == "nt":
            return True
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
