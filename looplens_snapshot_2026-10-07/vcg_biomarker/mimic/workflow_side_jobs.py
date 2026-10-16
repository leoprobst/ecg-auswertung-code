"""Independent long stages as supervised side jobs (audit E8).

The guided workflow runs its stages one after another. ``population-cohorts``
depends only on ``inventory`` but ran last, for about 19 hours, while the
four-to-seven-day feature extraction could have shared the machine with it.
It is therefore queued as a standalone job as soon as ``inventory`` is done;
the supervisor runs it beside the study under its fair-share grant.

When the workflow reaches the stage in its own order it waits for the side
job and then calls the stage as usual. The stage's own reuse check finds the
finished artifact and returns in seconds. A failed, cancelled or blocked side
job is simply recomputed in line, so the side job can only save time, never
change a result or lose one.

Only runs under the supervisor (with a job control) use side jobs; direct
calls keep the strictly sequential order.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

SIDE_STAGES: tuple[str, ...] = ("population-cohorts",)
SIDE_STAGE_PREREQUISITES: Mapping[str, tuple[str, ...]] = {
    "population-cohorts": ("inventory",),
}
SIDE_JOB_REGISTRY_NAME = "side_jobs.json"
SIDE_JOB_POLL_SECONDS = 15.0
_FINISHED_STATES = frozenset({"completed", "failed", "cancelled", "blocked"})


def _registry_path(config, run_id: str) -> Path:
    return Path(config.runs_root) / str(run_id) / SIDE_JOB_REGISTRY_NAME


def _read_registry(path: Path) -> dict[str, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(key): str(value) for key, value in dict(payload).items()}


def launch_side_jobs(
    config,
    *,
    run_id: str,
    stage_ids: Iterable[str],
    completed: Iterable[str],
    job_control,
    arguments_for: Callable[[str], Mapping[str, Any]],
    progress: Optional[Callable[[str], None]] = None,
) -> dict[str, str]:
    """Queue every side stage whose prerequisites are done; idempotent."""
    if job_control is None or not hasattr(job_control, "store"):
        return {}
    from .storage import atomic_write_json

    stages = set(stage_ids)
    done = set(completed)
    path = _registry_path(config, run_id)
    registry = _read_registry(path)
    store = job_control.store
    project_file = store.get(job_control.job_id).spec.project_file
    changed = False
    for stage in SIDE_STAGES:
        if stage not in stages or stage in done or stage in registry:
            continue
        if not set(SIDE_STAGE_PREREQUISITES.get(stage, ())) <= done:
            continue
        record = store.create(
            stage=stage,
            project_file=project_file,
            arguments=dict(arguments_for(stage)),
            requested_by=f"study-run:{job_control.job_id}",
        )
        registry[stage] = record.spec.job_id
        changed = True
        if progress is not None:
            progress(f"{stage} laeuft als eigener Job parallel ({record.spec.job_id})")
    if changed:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, registry)
    return registry


def wait_for_side_job(
    config,
    stage: str,
    *,
    run_id: str,
    job_control,
    progress: Optional[Callable[[str], None]] = None,
    poll_seconds: float = SIDE_JOB_POLL_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> Optional[str]:
    """Block until the side job of ``stage`` has finished; its final state.

    The study job's own pause and cancel requests are honoured while waiting
    (``job_control.checkpoint`` raises them).
    """
    if job_control is None or not hasattr(job_control, "store"):
        return None
    job_id = _read_registry(_registry_path(config, run_id)).get(stage)
    if job_id is None:
        return None
    store = job_control.store
    while True:
        try:
            state = str(store.get(job_id).state.value)
        except KeyError:
            return None
        if state in _FINISHED_STATES:
            return state
        message = f"Wartet auf den parallelen Job {stage} ({state})"
        if progress is not None:
            progress(message)
        job_control.checkpoint(message)
        sleep(poll_seconds)
