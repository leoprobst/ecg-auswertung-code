"""Scientifically neutral progress estimates for the desktop job view.

The analysis worker intentionally records raw checkpoints.  A validation
message such as ``29/34 endpoints`` is *not* 85 % of the computational work:
one endpoint may take seconds and another hours.  Treating every embedded
counter as a linear work counter caused the A017 pilot to sit at the same
displayed value for many hours and made its ETA unusable.

This module keeps that presentation concern out of the analysis identity.  It
uses measured durations from the run's Go/No-Go artifact and falls back to the
backend's declared stage priors.  Only record-count stages use their reported
fraction; elapsed time alone never creates statistical work progress. An
exceeded prior becomes unknown remaining work, not an artificial 95 % plateau.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from ..mimic.live_progress import (
    OPEN_ENDED_STAGES,
    VALIDATE_WEIGHT_IS_A_LOWER_BOUND,
    stage_weight,
    UNIFORM_PROGRESS_STAGES,
)


PROGRESS_ESTIMATOR_VERSION = "1.8.0"

# These counters describe a uniform number of ECG records or input rows.  The
# validation counters describe differently expensive endpoints/tables and are
# therefore deliberately absent.
_RELIABLE_LINEAR_FRACTION_STAGES = UNIFORM_PROGRESS_STAGES


@dataclass(frozen=True)
class CalibratedJobProgress:
    fraction: float
    remaining_seconds: Optional[float]
    stages_completed: int
    stages_total: int
    current_stage: str = ""
    estimate_is_lower_bound: bool = False
    basis: str = ""
    remaining_low_seconds: Optional[float] = None
    remaining_high_seconds: Optional[float] = None
    estimate_has_unmodelled_work: bool = False
    estimate_status: str = "planning_estimate"
    current_stage_fraction_known: bool = False
    current_stage_elapsed_seconds: float = 0.0
    current_stage_fraction: Optional[float] = None
    current_stage_message: str = ""
    current_stage_remaining_seconds: Optional[float] = None


def _read_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _study_stages(record: Any) -> list[str]:
    arguments = dict(getattr(getattr(record, "spec", None), "arguments", {}) or {})
    study_spec = arguments.get("study_spec")
    if not isinstance(study_spec, Mapping):
        return []
    return [str(value) for value in (study_spec.get("stage_ids") or ())]


def development_record_count(master_root: Path) -> Optional[int]:
    """Actual unlocked inventory; never extrapolate from a hardcoded cohort."""
    rows = _read_mapping(Path(master_root) / "ecg_master_index_summary.json").get("splits")
    if not isinstance(rows, list):
        return None
    counts = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("split") not in {"train", "validation"}:
            continue
        value = row.get("ecg_count")
        if type(value) is not int or value < 0 or row["split"] in counts:
            return None
        counts[row["split"]] = value
    return sum(counts.values()) if len(counts) == 2 else None


def _elapsed_job_seconds(record: Any) -> float:
    started_at = getattr(record, "started_at", None)
    if not started_at:
        return 0.0
    try:
        started = datetime.fromisoformat(str(started_at))
        now = datetime.now(started.tzinfo or timezone.utc)
        return max(0.0, (now - started).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _calibration(go_no_go: Mapping[str, Any]) -> dict[str, float]:
    runtime = go_no_go.get("runtime_and_memory")
    if not isinstance(runtime, Mapping):
        return {}
    if "stage_nonreuse_reported_attempt_seconds" in runtime:
        from ..mimic.runtime_calibration import nonreuse_attempt_medians
        return nonreuse_attempt_medians(runtime)
    # Total effort includes failed/restarted attempts and is not one future
    # execution. Use successful historical attempts only, still as unmatched
    # priors (cache state and scientific configuration may differ).
    attempts = runtime.get("stage_nonreuse_reported_attempt_seconds",
                           runtime.get("stage_successful_attempt_seconds"))
    if isinstance(attempts, Mapping):
        attempt_priors: dict[str, float] = {}
        for stage, values in attempts.items():
            if not isinstance(values, list):
                continue
            finite = []
            for value in values:
                if isinstance(value, bool):
                    continue
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(number) and number >= 0:
                    finite.append(number)
            if finite:
                attempt_priors[str(stage)] = float(np.median(finite))
        return attempt_priors
    raw = runtime.get("stage_duration_seconds")
    if not isinstance(raw, Mapping):
        return {}
    result: dict[str, float] = {}
    for stage, value in raw.items():
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(seconds) and seconds >= 0.0:
            result[str(stage)] = seconds
    return result


def _feature_full_seconds(
    go_no_go: Mapping[str, Any],
    expected_development_records: Optional[int],
    target_feature_runtime_contract: object = None,
) -> Optional[float]:
    """Scale a nonreuse pilot only with a matching prospective workload."""

    if not expected_development_records or expected_development_records <= 0:
        return None
    details = go_no_go.get("details")
    performance = details.get("performance") if isinstance(details, Mapping) else None
    pilots = performance.get("pilots") if isinstance(performance, Mapping) else None
    pilot = pilots.get("1000") if isinstance(pilots, Mapping) else None
    if not isinstance(pilot, Mapping):
        return None
    from ..mimic.runtime_calibration import feature_timing_contracts_match
    reference_contract = pilot.get("feature_runtime_contract")
    if not isinstance(reference_contract, Mapping):
        return None
    reused_records = reference_contract.get("observed_reused_records")
    if type(reused_records) is not int or reused_records != 0:
        return None
    if (pilot.get("duration_basis") != "median_successful_explicit_nonreuse_attempts"
            or not feature_timing_contracts_match(reference_contract, target_feature_runtime_contract)):
        return None
    try:
        rate = float(str(pilot.get("feature_seconds_per_record")))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(rate) or rate <= 0.0:
        return None
    return rate * int(expected_development_records)


def _stage_priors(
    stages: list[str],
    go_no_go: Mapping[str, Any],
    *,
    expected_development_records: Optional[int],
    target_feature_runtime_contract: object = None,
) -> dict[str, float]:
    priors = {stage: max(1.0, stage_weight(stage) * 3600.0) for stage in stages}
    priors.update({
        stage: max(1.0, seconds)
        for stage, seconds in _calibration(go_no_go).items()
        if stage in priors
    })
    feature_seconds = _feature_full_seconds(go_no_go, expected_development_records, target_feature_runtime_contract)
    if feature_seconds is not None and "extract-features" in priors:
        priors["extract-features"] = feature_seconds
    return priors


def _live_stage_state(live: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    entries = live.get("stages")
    if not isinstance(entries, list):
        return result
    for entry in entries:
        if isinstance(entry, Mapping) and entry.get("stage"):
            result[str(entry["stage"])] = entry
    return result


def calibrated_job_progress(
    record: Any,
    run_root: Optional[Path],
    *,
    expected_development_records: Optional[int] = None,
) -> Optional[CalibratedJobProgress]:
    """Reconstruct useful progress without trusting heterogeneous counters."""

    stages = _study_stages(record)
    if not stages:
        return None
    root = Path(run_root) if run_root is not None else None
    live = _read_mapping(root / "live_progress.json") if root else {}
    job_id = str(getattr(getattr(record, "spec", None), "job_id", ""))
    if live.get("job_id") and job_id and live["job_id"] != job_id:
        live = {}  # Never carry progress from a previous job into a new one.
    live_entries = live.get('stages')
    if isinstance(live_entries, list):
        live_ids = [entry.get('stage') for entry in live_entries if isinstance(entry, Mapping)]
        if (bool(live_ids) and len(live_ids) == len(live_entries)
                and len(live_ids) == live.get('stages_total')
                and all(isinstance(stage, str) and stage in stages for stage in live_ids)
                and len(set(live_ids)) == len(live_ids)
                ):
            # The full run omits pilot gates contained in the original job
            # spec. Its live stage list is the executed attempt's plan.
            stages = live_ids
    state_value = getattr(record, "state", "")
    state = str(getattr(state_value, "value", state_value)).lower()
    if state == "completed":
        return CalibratedJobProgress(
            fraction=1.0,
            remaining_seconds=0.0,
            stages_completed=len(stages),
            stages_total=len(stages),
            basis="abgeschlossen",
            estimate_status="completed",
        )
    go_no_go = (
        _read_mapping(root / "go_no_go" / "go_no_go.json") if root else {}
    )
    inherited_calibration = (
        _read_mapping(root / "runtime_calibration.json") if root else {}
    )
    # Merge per stage: the pilot report contains no upstream measurements and
    # must not discard valid inherited preparation/cache timings.
    timing_source = {**inherited_calibration, **go_no_go}
    timing_source["runtime_and_memory"] = {"stage_duration_seconds": {
        **_calibration(inherited_calibration), **_calibration(go_no_go)}}
    priors = _stage_priors(
        stages,
        timing_source,
        expected_development_records=expected_development_records,
        target_feature_runtime_contract=live.get("feature_runtime_contract"),
    )
    stage_state = _live_stage_state(live)
    completed = {
        stage
        for stage, payload in stage_state.items()
        if stage in stages and str(payload.get("state", "")).lower() == "done"
    }
    current = str(live.get("current_stage") or "")
    if current not in stages:
        current = ""

    # Before the new worker has written its first state file, an old completed
    # pilot snapshot may still be present in the same scientific run folder.
    # Ignore it when the job itself is not complete and the live file claims
    # all stages are done.
    if not current and len(completed) >= len(stages):
        completed.clear()

    total_seconds = sum(priors.values()) or 1.0
    completed_seconds = sum(priors[stage] for stage in stages if stage in completed)
    current_fraction = 0.0
    current_remaining = 0.0
    current_elapsed = 0.0
    if current and current not in completed:
        current_prior = max(1.0, priors[current])
        entry = stage_state.get(current, {})
        try:
            current_elapsed = max(0.0, float(entry.get("elapsed_seconds") or 0.0))
        except (TypeError, ValueError):
            current_elapsed = 0.0
        # A snapshot is only rewritten at checkpoints. Advance its stage
        # clock up to the worker heartbeat, never beyond it: a dead worker
        # must not acquire apparent progress merely because the GUI refreshes.
        if state in {"running", "pausing"}:
            try:
                updated = datetime.fromisoformat(str(live.get("updated_at")))
                heartbeat = datetime.fromisoformat(str(getattr(record, "heartbeat_at", None)))
                current_elapsed += max(0.0, (heartbeat - updated).total_seconds())
            except (TypeError, ValueError, OverflowError):
                pass

        if current in _RELIABLE_LINEAR_FRACTION_STAGES:
            try:
                reported = float(live.get("current_stage_fraction") or 0.0)
            except (TypeError, ValueError):
                reported = 0.0
            reported = min(1.0, max(0.0, reported))
            if reported > 0.0 and current_elapsed > 0.0:
                observed_total = current_elapsed / reported
                expected_total = observed_total
                current_fraction = min(0.99, reported)
                current_remaining = max(0.0, expected_total - current_elapsed)
                # Reused records count as completed work, not as compute
                # throughput. The new worker reports the exact cache count.
                message = str(live.get("current_stage_message") or "")
                counts = re.search(r"Features ([\d,]+)/([\d,]+).*Reused=([\d,]+)", message)
                if counts:
                    done, total, reused = (int(value.replace(",", "")) for value in counts.groups())
                    if 0 <= reused < done <= total:
                        current_remaining = current_elapsed * (total - done) / (done - reused)
                    else:
                        current_remaining = 0.0  # no fresh throughput: unknown
            else:
                current_fraction = 0.0
                current_remaining = max(0.0, current_prior - current_elapsed)
        else:
            # Wall time is not completed statistical work. Show the last
            # observed work fraction, and explicitly expire an exceeded prior.
            current_fraction = 0.0
            current_remaining = max(0.0, current_prior - current_elapsed)
        completed_seconds += priors[current] * current_fraction

    pending_seconds = sum(
        priors[stage]
        for stage in stages
        if stage not in completed
        and stage != current
        and stage not in OPEN_ENDED_STAGES
    )
    remaining = current_remaining + pending_seconds
    prior_expired = bool(current and current not in completed and current_remaining <= 0)
    lower_bound = prior_expired or (
        VALIDATE_WEIGHT_IS_A_LOWER_BOUND
        and "validate" in stages
        and "validate" not in completed
    ) or any(
        stage in OPEN_ENDED_STAGES and stage not in completed for stage in stages
    )
    display_message = str(live.get("current_stage_message") or "") if current else ""
    if current == "extract-features" and not display_message.startswith("Features "):
        # Worker-resizing notices briefly replace the live text, but the job
        # record retains the most recent record checkpoint. Never substitute
        # its counter for the live fraction or silently call it current.
        checkpoint_message = str(getattr(record, "progress_message", "") or "")
        checkpoint = re.search(r"Features ([\d,]+)/([\d,]+)", checkpoint_message)
        if checkpoint and current_fraction > 0:
            done, total = (int(value.replace(",", "")) for value in checkpoint.groups())
            if total > 0 and abs(done / total - current_fraction) <= 0.005:
                display_message += f"; letzter Feature-Checkpoint: {done:,}/{total:,}"
    return CalibratedJobProgress(
        fraction=min(1.0, max(0.0, completed_seconds / total_seconds)),
        remaining_seconds=remaining if remaining > 0.0 else None,
        stages_completed=len(completed),
        stages_total=len(stages),
        current_stage=current,
        current_stage_fraction_known=bool(
            current in _RELIABLE_LINEAR_FRACTION_STAGES and current_fraction > 0
        ),
        current_stage_elapsed_seconds=current_elapsed,
        current_stage_fraction=current_fraction if current in _RELIABLE_LINEAR_FRACTION_STAGES and current_fraction > 0 else None,
        current_stage_message=display_message,
        current_stage_remaining_seconds=(current_remaining if current and current_fraction > 0 and current_remaining > 0 else None),
        estimate_is_lower_bound=False,
        estimate_has_unmodelled_work=lower_bound,
        estimate_status="partial_planning_estimate" if lower_bound else "planning_estimate",
        basis=(
            "Zeitvorwert überschritten; Restarbeit unbekannt" if prior_expired else
            "Historische Stufenzeiten als unsichere Planungswerte"
            if _calibration(timing_source)
            else "Stufen-Laufzeitprofil"
        ),
        # A single historical execution cannot estimate a prediction interval.
        remaining_low_seconds=None,
        remaining_high_seconds=None,
    )


__all__ = [
    "CalibratedJobProgress",
    "PROGRESS_ESTIMATOR_VERSION",
    "calibrated_job_progress",
]
