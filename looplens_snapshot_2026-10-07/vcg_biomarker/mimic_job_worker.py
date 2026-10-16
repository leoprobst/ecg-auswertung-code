from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import traceback
import uuid
from multiprocessing import freeze_support

from .mimic.config import load_or_discover_config
from .mimic.jobs import (
    JOB_PROJECT_CONFIG_HASH_KEY,
    JOB_PROJECT_CONFIG_KEY,
    JobCancellationRequested,
    JobControl,
    JobPauseRequested,
    JobStore,
    project_config_drift,
)
from .mimic.models import JobState
from .mimic.resources import apply_worker_thread_limits
from .mimic.resources import stage_memory_safety_policy
from .mimic.services import run_service_stage
from .mimic.storage import atomic_write_json, content_hash
from .mimic.run_log import RunLog, active as active_run_log


@contextmanager
def _standalone_job_telemetry(config, spec):
    """Single-stage jobs need the same query/resource visibility as guided runs."""
    if spec.stage == "study-run":
        # The guided workflow owns its stage envelopes and sampler.
        yield None
        return
    root = config.jobs_root / "telemetry" / content_hash({"job_id": spec.job_id})[:20]
    log = RunLog.for_run(root)
    log.job_id = spec.job_id
    log.attempt_id = uuid.uuid4().hex
    with active_run_log(log), log.stage_scope(spec.stage):
        yield log


def main() -> int:
    parser = argparse.ArgumentParser(prog="vcg-mimic-job-worker")
    parser.add_argument("--project", required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    live_config = load_or_discover_config(args.project)
    store = JobStore.for_project(live_config)
    record = store.get(args.job_id)
    memory_policy = _apply_job_resource_environment(
        record.spec.resource_profile,
        stage=record.spec.stage,
        os_reserve_gb=live_config.os_memory_reserve_gb,
    )
    apply_worker_thread_limits()
    if record.state == JobState.PAUSING:
        store.transition(
            args.job_id,
            JobState.PAUSED,
            message="Paused before worker execution",
        )
        return 75
    if record.state == JobState.CANCELLING:
        store.transition(
            args.job_id,
            JobState.CANCELLED,
            message="Cancelled before worker execution",
        )
        return 76
    store.claim(args.job_id, os.getpid())
    store.append_event(
        args.job_id,
        "memory_policy_applied",
        (
            f"RAM policy {memory_policy.policy_version}: "
            f"{memory_policy.stage}/{memory_policy.workload_class}, "
            f"DuckDB {memory_policy.duckdb_memory_limit_gb:g} GB of "
            f"{memory_policy.requested_memory_gb:g} GB allocation"
        ),
        memory_policy.to_dict(),
    )
    control = JobControl(store, args.job_id)
    if record.spec.stage == "study-run":
        control.set_runtime_stage(
            record.spec.stage,
            state="worker_started",
            memory_policy=memory_policy.to_dict(),
        )
    try:
        config = _runtime_config_for_job(
            live_config,
            record.spec,
        )
        stage_arguments = dict(record.spec.arguments)
        stage_arguments.pop(JOB_PROJECT_CONFIG_KEY, None)
        stage_arguments.pop(JOB_PROJECT_CONFIG_HASH_KEY, None)
        stage_arguments["_execution_id"] = record.spec.job_id
        with _standalone_job_telemetry(config, record.spec):
            result = run_service_stage(
                config,
                record.spec.stage,
                stage_arguments,
                progress=lambda message: None,
                job_control=control,
            )
        artifact_hash = _result_hash(result)
        _finalize_progress(store, args.job_id)
        store.transition(
            args.job_id,
            JobState.COMPLETED,
            message="Job completed",
            artifact_hash=artifact_hash,
        )
        return 0
    except JobPauseRequested as exc:
        _sync_guided_manifest_status(
            live_config,
            record.spec,
            status="paused",
            diagnostic=str(exc),
        )
        store.transition(
            args.job_id,
            JobState.PAUSED,
            message=str(exc),
        )
        return 75
    except JobCancellationRequested as exc:
        _sync_guided_manifest_status(
            live_config,
            record.spec,
            status="cancelled",
            diagnostic=str(exc),
        )
        store.transition(
            args.job_id,
            JobState.CANCELLED,
            message=str(exc),
        )
        return 76
    except Exception as exc:
        _sync_guided_manifest_status(
            live_config,
            record.spec,
            status="failed",
            diagnostic=f"{type(exc).__name__}: {exc}",
        )
        store.append_event(
            args.job_id,
            "traceback",
            traceback.format_exc(),
        )
        store.transition(
            args.job_id,
            JobState.FAILED,
            message=f"{type(exc).__name__}: {exc}",
            error_message=f"{type(exc).__name__}: {exc}",
        )
        return 1


def _sync_guided_manifest_status(
    config,
    spec,
    *,
    status: str,
    diagnostic: str,
) -> None:
    """Keep the run manifest consistent with its supervising study job."""

    if spec.stage != "study-run":
        return
    run_id = str(spec.arguments.get("run_id", "")).strip()
    if not run_id:
        return
    path = Path(config.runs_root) / run_id / "guided_workflow.manifest.json"
    if not path.is_file():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(payload, dict):
        return
    payload["status"] = str(status)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    payload["terminal_diagnostic"] = str(diagnostic)
    atomic_write_json(path, payload)


def _result_hash(result) -> str:
    if hasattr(result, "to_dict"):
        return content_hash(result.to_dict())
    if hasattr(result, "__dict__"):
        return content_hash(result.__dict__)
    return content_hash(str(result))


def _finalize_progress(store: JobStore, job_id: str) -> None:
    record = store.get(job_id)
    if record.progress_total <= 0:
        store.heartbeat(
            job_id,
            progress_message="Job completed",
        )
        return
    store.heartbeat(
        job_id,
        progress_current=record.progress_total,
        progress_total=record.progress_total,
        progress_message="Job completed",
    )


def _apply_job_resource_environment(
    profile,
    *,
    stage: str | None = None,
    os_reserve_gb: float = 6.0,
):
    os.environ["VCG_JOB_CPU_TOKENS"] = str(max(1, int(profile.cpu_tokens)))
    allocation = max(1.0, float(profile.memory_gb))
    os.environ["VCG_JOB_MEMORY_ALLOCATION_GB"] = str(allocation)
    os.environ["VCG_JOB_OS_RESERVE_GB"] = str(max(0.0, os_reserve_gb))
    policy = stage_memory_safety_policy(
        stage or "balanced",
        reserve_gb=os_reserve_gb,
        requested_memory_gb=allocation,
    )
    os.environ["VCG_JOB_MEMORY_GB"] = str(
        allocation if stage is None else policy.duckdb_memory_limit_gb
    )
    return policy


def _runtime_config_for_job(live_config, spec):
    project_config = dict(
        spec.arguments.get(JOB_PROJECT_CONFIG_KEY, {})
    )
    project_config_hash = str(
        spec.arguments.get(JOB_PROJECT_CONFIG_HASH_KEY, "")
    )
    if not project_config:
        return live_config
    snapshot_payload = project_config
    if content_hash(snapshot_payload) != project_config_hash:
        raise RuntimeError(
            f"Job {spec.job_id} project configuration hash is invalid"
        )
    snapshot = type(live_config).from_dict(snapshot_payload)
    if snapshot.work_root.resolve() != live_config.work_root.resolve():
        raise RuntimeError(
            f"Job {spec.job_id} project snapshot points to another work root"
        )
    # Pinning the scientific configuration per job is intentional. What must
    # not happen silently is a data source appearing after the pin: the job
    # would keep producing empty evidence for that layer while the project
    # config claims the source is available.
    drift = project_config_drift(snapshot_payload, live_config.to_dict())
    if drift["source_added"]:
        raise RuntimeError(
            f"Job {spec.job_id} ist auf eine Konfiguration ohne "
            f"{', '.join(drift['source_added'])} festgelegt, das Projekt "
            "konfiguriert diese Quelle inzwischen aber. Der Lauf wuerde diese "
            "Evidenzebene stillschweigend leer lassen. Lauf als historisch "
            "bewahren und mit neuer run_id neu starten."
        )
    if drift["scientific"]:
        print(
            f"[warn] Job {spec.job_id}: Projektkonfiguration weicht vom "
            f"festgelegten Job-Snapshot ab ({', '.join(drift['scientific'])}); "
            "der Lauf verwendet weiterhin den Snapshot.",
            flush=True,
        )
    if spec.stage == "benchmark":
        return live_config

    # CPU and memory are operational allocations. Scientific configuration
    # hashes remain bound to the immutable job snapshot.
    snapshot.to_dict = lambda: dict(snapshot_payload)
    snapshot.workers = max(
        1,
        int(spec.resource_profile.cpu_tokens),
    )
    snapshot.duckdb_memory_limit_gb = max(
        1,
        int(math.ceil(float(spec.resource_profile.memory_gb))),
    )
    return snapshot


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())
