from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from .config import MimicProjectConfig
from .jobs import JobControl, JobStore
from .models import JobRecord, StudyRunSpec
from .presets import (
    PRODUCTION_PREVIEW_EXTERNAL_RECORDS,
    PRODUCTION_PREVIEW_PRESET_ID,
    PRODUCTION_PREVIEW_RECORDS,
    STUDY_PRESET_ID,
    get_analysis_preset,
    resolve_preset_id,
)
from .resources import stage_worker_memory_scope
from .run_lineage import GENERATED_RUN_ID, resolve_current_analysis
from .run_log import RunLog, active as active_run_log
from .storage import (
    StageManifest,
    atomic_write_json,
    content_hash,
    operational_config_hash,
    scientific_config_hash,
    source_tree_hash,
)
from .live_progress import (
    STAGE_WEIGHTS,
    LiveProgressReporter,
    RunProgress,
    parse_fraction as parse_fraction,
)
from .user_guidance import stage_descriptor
from .workflow_graph import ordered_stages, dependency_graph


GUIDED_WORKFLOW_VERSION = "1.27.0"
GUIDED_PREPARATION_STAGES = (
    "inventory",
    "genetic-feasibility",
    "benchmark",
    "ingest",
    "ingest-events",
    "build-index",
    "parse-reports",
    "parse-notes",
    "build-evidence",
    "attest-event-evidence",
    "build-labels",
    "attest-labels",
    "validate-delineation",
    "audit-biomarkers",
)
GUIDED_PILOT_1K_STAGES = (
    "extract-features-pilot-1000",
    "validate-pilot-1000",
)
GUIDED_PILOT_10K_STAGES = (
    "extract-features-pilot-10000",
    "validate-pilot-10000",
)
GUIDED_PILOT_FINAL_STAGES = ("pilot-go-no-go",)
GUIDED_FULL_ANALYSIS_STAGES = (
    "extract-features",
    # Before every screen: the screens need to know which markers are one
    # quantity under several names, and the answer is computed from the
    # markers themselves. The DAG (workflow_graph) orders it after
    # therapy-response and physiological-gap-markers. There is no data flow
    # either way -- neither of the two reads the redundancy groups and the
    # redundancy stage reads neither of their outputs -- so the order does not
    # change any result (audit O2).
    "marker-redundancy",
    "therapy-response",
    "physiological-gap-markers",
    "repeatability",
    "validate",
)
# Compatibility export for callers that only need the canonical full path.
GUIDED_BASE_STAGES = GUIDED_PREPARATION_STAGES + GUIDED_FULL_ANALYSIS_STAGES
# Laeuft nach allem Wichtigen: offene Laufzeit, jederzeit abbrechbar,
# und nichts weiter unten haengt an seinem Ergebnis.
# The four paired and time-to-event designs. They run after the screens
# because they read the same feature partitions and are cheap by comparison --
# each reads a handful of markers over a cohort the endpoints already define,
# rather than the whole catalog. None of them feeds anything below.
GUIDED_TARGETED_DESIGNS = (
    "within-subject-change",
    "voltage-mass-discordance",
    "time-to-event",
    "occlusion-yield",
    # The fixed thesis versions (THESEN_FASSUNGEN_2026-09-24.md). They read
    # the features, the labels, the evidence and this run's reference ranges;
    # nothing below reads their result.
    "thesis-tests",
)
GUIDED_FINAL_STAGES = (
    # Both read ``screening_results.parquet`` and nothing else, so they run
    # before the open-ended screens: the control panel says whether the run
    # just finished is worth reading further, which is exactly the question to
    # answer before spending hours on the broad axes.
    "control-panel",
    # Whether a marker was measurable, and whether that depends on the
    # diagnosis. Reads five columns over the cohort once; it belongs beside
    # the control panel because both ask what a result is worth before
    # anything expensive reads it.
    "segment-availability",
    # A second, independent derivation of the same labels, from the published
    # Ext-ICD table. It reads the finished labels and nothing else, so it costs
    # a minute and belongs beside the control panel: both answer "is what this
    # run measured what it says it measured" before anything expensive starts.
    "ext-icd-crosscheck",
    # The reference ranges. They read the finished features and the clinical
    # context and nothing else, so they cost one pass and belong here rather
    # than at the end: the reporting set downstream is written against them.
    "reference-ranges",
    # The transform table has to exist before the reporting set can require it.
    "transform-agreement",
    # CODE-15 supplies primary-care reference distributions and SaMi-Trop
    # supplies Chagas mortality follow-up. They are analysed separately from
    # MIMIC and never used to tune MIMIC thresholds.
    "population-cohorts",
    "two-stage-selection",
    "screen-broad-axes",
    *GUIDED_TARGETED_DESIGNS,
    # Last of the analysis stages: it reads the reference ranges, the
    # repeatability coefficients, the transform table, the screen and the
    # operating points, and assembles what a report must contain.
    "reporting-guideline",
    "report",
    "audit",
)

# One weight table for the whole project. There used to be a second, private
# copy here; the two drifted apart -- this one never learned about
# external-validation, screen-broad-axes or the deep combination search, so
# every one of them counted as the default. A progress display and an ETA that
# disagree about what the run consists of are worse than none.
_STAGE_WEIGHTS = STAGE_WEIGHTS
_DEFAULT_STAGE_WEIGHT = 1.0
# Integer scale for job_control.checkpoint()'s current/total (stored as
# SQLite INTEGER columns) -- fine enough to show smooth sub-stage
# progress movement without needing float support in the jobs store.
_PROGRESS_SCALE = 1000


def _stage_weight(stage: str) -> float:
    return _STAGE_WEIGHTS.get(stage, _DEFAULT_STAGE_WEIGHT)


_FRACTION_RE = re.compile(r"([\d,.]+)\s*/\s*([\d,.]+)")


def _stage_internal_fraction(message: str) -> float:
    # Many stage-internal progress messages embed a "current/total" count
    # (e.g. "Entlassungsbriefe 50.000/200.000") -- used here only to
    # interpolate *within* the current stage's weight slice, never to
    # replace the overall stage-index-based progress (that conflation was
    # the bug: a batch-progress message like "50,000/200,000" was
    # previously fed straight into the SAME current/total the outer
    # per-stage checkpoint used, so the displayed percentage/ETA reflected
    # whichever message arrived last -- current batch or overall stage --
    # with no way to tell which).
    match = _FRACTION_RE.search(message)
    if not match:
        return 0.0
    try:
        current = float(match.group(1).replace(".", "").replace(",", ""))
        total = float(match.group(2).replace(".", "").replace(",", ""))
    except ValueError:
        return 0.0
    if total <= 0:
        return 0.0
    return max(0.0, min(1.0, current / total))


def _stage_fraction_high_water(previous: float, message: str) -> float:
    """Keep overall progress monotonic across differently sized subtasks."""

    return max(
        max(0.0, min(1.0, float(previous))),
        _stage_internal_fraction(message),
    )


@dataclass(frozen=True)
class GuidedWorkflowResult:
    study_id: str
    run_id: str
    preset_id: str
    manifest_path: Path
    completed_stages: tuple[str, ...]
    workflow_hash: str
    development_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["manifest_path"] = str(self.manifest_path)
        return data


def guided_study_spec(
    *,
    preset_id: str,
    run_id: str,
    study_id: Optional[str] = None,
    include_external_validation: bool = False,
    stop_after_pilots: bool = False,
) -> StudyRunSpec:
    # Resolve first: every branch below compares the identifier directly, so
    # a retired name would otherwise build a quietly different stage list --
    # 25 stages instead of 27 -- and the spec would record the retired name as
    # if it were the configuration that ran.
    preset_id = resolve_preset_id(preset_id)
    preset = get_analysis_preset(preset_id)
    stages = list(GUIDED_PREPARATION_STAGES)
    if preset_id == PRODUCTION_PREVIEW_PRESET_ID:
        # No pilots: the bounded extraction is the cohort. Everything the
        # study runs after its pilots runs here, in the study's order.
        stages.extend(GUIDED_FULL_ANALYSIS_STAGES)
        stages.extend(GUIDED_FINAL_STAGES)
        stages.insert(stages.index("reporting-guideline"), "score-discovery")
        return StudyRunSpec(
            study_id=study_id or run_id,
            display_name_de=preset.display_name_de,
            preset_id=preset_id,
            run_id=run_id,
            stage_ids=ordered_stages(stages),
            include_phenome_scan=False,
            development_only=True,
        )
    stages.extend(GUIDED_PILOT_1K_STAGES)
    if preset_id == "exhaustive_research":
        stages.extend(GUIDED_PILOT_10K_STAGES)
    if stop_after_pilots:
        stages.extend(GUIDED_PILOT_FINAL_STAGES)
        return StudyRunSpec(
            study_id=study_id or run_id,
            display_name_de=f"{preset.display_name_de} - Piloten/Go-No-Go",
            preset_id=preset_id,
            run_id=run_id,
            stage_ids=ordered_stages(stages),
            include_phenome_scan=False,
            development_only=True,
        )
    if preset_id != "fast_preview":
        stages.extend(GUIDED_FULL_ANALYSIS_STAGES)
    # This flag is reserved for an explicit post-freeze replay.  The normal
    # discovery workflow always passes False: external data may only be touched
    # after directions, scaling, thresholds and coefficients have been frozen.
    # The external-validation service independently enforces that contract.
    if include_external_validation and preset_id != "fast_preview":
        stages.append("external-validation")
    if preset.policy.include_broad_phenome_scan:
        stages.append("discover-phenome")
    stages.extend(GUIDED_FINAL_STAGES)
    if preset_id != "fast_preview":
        # Finite follow-up only. Existing serialized job specifications retain
        # their original stage list; the open-ended deep search stays opt-in.
        stages.insert(stages.index("reporting-guideline"), "score-discovery")
    return StudyRunSpec(
        study_id=study_id or run_id,
        display_name_de=preset.display_name_de,
        preset_id=preset_id,
        run_id=run_id,
        stage_ids=ordered_stages(stages),
        include_phenome_scan=preset.policy.include_broad_phenome_scan,
        development_only=True,
    )


def enqueue_guided_study(
    config: MimicProjectConfig,
    *,
    preset_id: str,
    run_id: str,
    priority: int = 60,
    requested_by: str = "gui",
    stop_after_pilots: bool = True,
) -> JobRecord:
    spec = guided_study_spec(
        preset_id=preset_id,
        run_id=run_id,
        # External evaluation is intentionally a post-freeze operation.  A
        # configured collection alone is not authority to select directions
        # or thresholds from a still-mutable development run.
        include_external_validation=False,
        stop_after_pilots=stop_after_pilots,
    )
    arguments = {
        "study_spec": spec.to_dict(),
        "preset_id": preset_id,
        "run_id": run_id,
        "include_locked_test": False,
        "stop_after_pilots": stop_after_pilots,
    }
    return JobStore.for_project(config).create(
        stage="study-run",
        project_file=config.save(),
        arguments=arguments,
        priority=priority,
        requested_by=requested_by,
    )


def _assert_generated_run_is_current(
    config: MimicProjectConfig,
    run_id: str,
) -> Optional[str]:
    """Reject a queued generated run after its scientific identity changed.

    The GUI hides jobs from older generations, but a detached supervisor may
    already have dequeued one.  This backend guard closes that race before the
    stale run directory or any upstream artifact can be touched.  Hand-written
    legacy/development identifiers remain available for bounded previews;
    the caller rejects them for full development runs.
    """

    if GENERATED_RUN_ID.fullmatch(str(run_id)) is None:
        return None
    current, fingerprint = resolve_current_analysis(config)
    if str(current.run_id) != str(run_id):
        raise RuntimeError(
            "Queued analysis run is no longer compatible with the current "
            f"scientific source/configuration: requested={run_id}, "
            f"current={current.run_id}. Enqueue the current run instead."
        )
    return str(fingerprint.fingerprint)


def _assert_run_preflight_gates(
    config: MimicProjectConfig,
    run_id: str,
    run_fingerprint: str,
) -> None:
    """Require the audited source-bound gates before a generated run starts."""

    run_root = config.runs_root / str(run_id)
    requirements = (
        (
            "software_preflight_gate.json",
            lambda value: (
                value.get("status") == "passed"
                and int(value.get("unexpected_skips", -1)) == 0
                and int(value.get("uncaptured_warnings", -1)) == 0
            ),
        ),
        (
            "gui_review_manifest.json",
            lambda value: value.get("status") == "passed",
        ),
    )
    failures: list[str] = []
    for filename, passed in requirements:
        path = run_root / filename
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            failures.append(f"{filename}: missing or unreadable")
            continue
        if not isinstance(payload, dict):
            failures.append(f"{filename}: invalid payload")
            continue
        identity_matches = (
            str(payload.get("run_id", "")) == str(run_id)
            and str(payload.get("run_fingerprint", ""))
            == str(run_fingerprint)
        )
        if not identity_matches or not passed(payload):
            failures.append(f"{filename}: failed or stale")
    if failures:
        raise RuntimeError(
            "Generated analysis run refused before preparation: "
            + "; ".join(failures)
        )


def _workflow_backend_hash(package_root: Path) -> str:
    """GUI and test edits cannot alter analysis resume identity."""
    return content_hash({
        "mimic": source_tree_hash(package_root / "mimic"),
        "core": source_tree_hash(package_root / "core"),
    })


def run_guided_study(
    config: MimicProjectConfig,
    arguments: Mapping[str, Any],
    *,
    progress: Optional[Callable[[str], None]] = None,
    job_control: Optional[JobControl] = None,
) -> GuidedWorkflowResult:
    if bool(arguments.get("include_locked_test", False)):
        raise PermissionError("Guided workflows never read the locked test")
    preset_id = resolve_preset_id(
        str(arguments.get("preset_id", STUDY_PRESET_ID))
    )
    run_id = str(arguments.get("run_id", "mimic-development-guided"))
    run_fingerprint = _assert_generated_run_is_current(config, run_id)
    full_development = (
        not bool(arguments.get("stop_after_pilots", False))
        and preset_id not in ("fast_preview", PRODUCTION_PREVIEW_PRESET_ID)
    )
    if preset_id == PRODUCTION_PREVIEW_PRESET_ID and run_fingerprint is not None:
        raise RuntimeError(
            "Production-path preview refused: it must not run under a generated "
            "development run identity; use a named preview run id."
        )
    if full_development and run_fingerprint is None:
        raise RuntimeError(
            "Full development run refused: a current generated run identity is required; "
            "legacy or manually named runs cannot bypass scientific release."
        )
    if run_fingerprint is not None:
        _assert_run_preflight_gates(config, run_id, run_fingerprint)
        if full_development:
            from .scientific_release import scientific_release_valid
            if not scientific_release_valid(config.runs_root / run_id, run_fingerprint):
                raise RuntimeError("Full development run refused: source-bound scientific release missing or stale; a technical pilot GO is insufficient.")
            from .capacity_planning import require_capacity_at_start
            require_capacity_at_start(config.runs_root / run_id, config.work_root, run_fingerprint)
    spec = guided_study_spec(
        preset_id=preset_id,
        run_id=run_id,
        study_id=str(
            dict(arguments.get("study_spec", {})).get("study_id", run_id)
        ),
        include_external_validation=False,
        stop_after_pilots=bool(arguments.get("stop_after_pilots", False)),
    )
    preset = get_analysis_preset(preset_id)
    run_root = config.runs_root / run_id
    recorded_stages = dict(arguments.get("study_spec", {})).get("stage_ids")
    if recorded_stages:
        # A saved job is an immutable execution plan, including its ordering.
        # New plans place bounded score discovery before aggregate reports;
        # resuming an older job must not silently reorder its completed work.
        spec = replace(spec, stage_ids=tuple(recorded_stages))
    # A reviewed delta is a distinct QC design, never fabricated 1k/10k output.
    # Apply identically at launch/resume, including a GUI-supplied stage list.
    # All other stages retain their recorded ordering and identities.
    if full_development:
        from .scientific_release import reviewed_delta_acceptance_valid
        if reviewed_delta_acceptance_valid(config.runs_root / run_id, run_fingerprint):
            pilot_stages = set(GUIDED_PILOT_1K_STAGES + GUIDED_PILOT_10K_STAGES + GUIDED_PILOT_FINAL_STAGES)
            spec = replace(spec, stage_ids=tuple(s for s in spec.stage_ids if s not in pilot_stages))
    run_root.mkdir(parents=True, exist_ok=True)
    # Survives the process. Of 31,414 events the last run produced, 31,133
    # were free-text progress lines that overwrite each other and vanish with
    # the worker; there was no record of how long a query ran, how much it
    # read, or why a stage recomputed. This is that record.
    log = RunLog.for_run(run_root)
    log.run_id = run_id
    log.job_id = str(getattr(job_control, "job_id", ""))
    log.attempt_id = uuid.uuid4().hex
    manifest_path = run_root / "guided_workflow.manifest.json"
    project_config_hash = scientific_config_hash(config)
    project_operational_config_hash = operational_config_hash(config)
    workflow_hash = content_hash(
        {
            "workflow_version": GUIDED_WORKFLOW_VERSION,
            "study_spec": spec.to_dict(),
            "scientific_project_config_hash": project_config_hash,
            # GUI edits must not invalidate a running analysis workflow.
            # Scientific stages retain their own stricter artifact identities.
            "implementation_hash": _workflow_backend_hash(Path(__file__).resolve().parents[1]),
        }
    )
    completed: list[str] = []
    manifest = _load_manifest(manifest_path)
    previous_hash = str(manifest.get("workflow_hash", ""))
    previous_completed = [
        str(item) for item in manifest.get("completed_stages", ())
    ]
    contract_changed = bool(
        previous_hash and previous_hash != workflow_hash and previous_completed
    )
    if contract_changed:
        # This used to raise, and it cost a run.  The concern is real -- stage
        # artifacts from two different versions of the code must never be
        # mixed -- but refusing the whole workflow is the wrong instrument for
        # it, for two reasons.
        #
        # The hash covers the entire package: every module, the interface, the
        # test suite.  Editing a test file, or a label in a dialog, or a stage
        # that has not run yet, changes it.  On 2026-08-20 a run that had
        # spent four hours in attest-event-evidence was restarted by the
        # supervisor after a memory dip and refused on the way back in,
        # because unrelated modules had been edited while it worked.
        #
        # And it is redundant.  Every stage already carries its own reuse key
        # over its own inputs, configuration and implementation, and those
        # keys have been hardened repeatedly.  They are precise where this one
        # is coarse: a change to the criteria module cannot affect an ingest
        # artifact, and the ingest stage's own key knows that.
        #
        # So the completed list is discarded rather than trusted, and every
        # stage is offered to its own reuse check.  Nothing is mixed: a stage
        # whose key still matches skips in seconds, and one whose key moved
        # recomputes.  Discarding a resume aid costs time; refusing a run
        # costs the run.
        completed = []
        log.write(
            "workflow",
            event="completed_list_discarded",
            run_id=run_id,
            reason=(
                "Vertragshash des Quellstands weicht ab; jede Stufe wird "
                "einzeln gegen ihren eigenen Wiederverwendungsschluessel "
                "geprueft"
            ),
            previous_workflow_hash=previous_hash,
            workflow_hash=workflow_hash,
            discarded_stage_count=len(previous_completed),
            discarded_stages=list(previous_completed),
        )
        if progress is not None:
            progress(
                "Quellstand hat sich seit den erledigten Stufen geaendert; "
                "jede Stufe wird einzeln geprueft statt den Lauf abzubrechen"
            )
    elif previous_hash == workflow_hash:
        completed = [
            str(item)
            for item in previous_completed
            if str(item) in spec.stage_ids
        ]
    _write_workflow_manifest(
        manifest_path,
        spec,
        workflow_hash,
        completed,
        status="running",
        project_config_hash=project_config_hash,
        operational_config_hash=project_operational_config_hash,
    )

    # Terminal status board. Writes a machine-readable mirror next to the run
    # manifest so a detached run can be inspected from a fresh console.
    live = LiveProgressReporter(
        RunProgress(run_id=run_id, stages=list(spec.stage_ids), completed=set(completed),
                    job_id=log.job_id, attempt_id=log.attempt_id),
        state_path=run_root / "live_progress.json",
    )

    # Imported lazily to keep services -> workflow free of an import cycle.
    from .services import run_service_stage

    # Own job_control checkpointing exclusively for the whole guided run,
    # in weighted-overall-progress units (see _STAGE_WEIGHTS) -- ``progress``
    # here is the *raw*, unmanaged callback (services.py passes it through
    # unwrapped for the "study-run" stage specifically) so stage-internal
    # progress messages (e.g. "current batch" counts from a long stage
    # like extract-features) never reach job_control directly and
    # overwrite this stage-index-based tracking with unrelated numbers.
    total_weight = sum(_stage_weight(stage) for stage in spec.stage_ids) or 1.0

    log.write(
        "workflow",
        event="begin",
        run_id=run_id,
        preset_id=preset_id,
        workflow_hash=workflow_hash,
        stage_count=len(spec.stage_ids),
        already_complete=len(completed),
        contract_changed=contract_changed,
    )

    from .workflow_side_jobs import SIDE_STAGES, launch_side_jobs, wait_for_side_job

    def _launch_ready_side_jobs() -> None:
        # Audit E8: long independent stages run beside the study as their
        # own supervised jobs; see ``workflow_side_jobs``.
        launch_side_jobs(
            config,
            run_id=run_id,
            stage_ids=spec.stage_ids,
            completed=completed,
            job_control=job_control,
            arguments_for=lambda side_stage: _stage_arguments(
                config, side_stage, preset_id=preset_id, run_id=run_id,
            ),
            progress=progress,
        )

    _launch_ready_side_jobs()
    for index, stage in enumerate(spec.stage_ids, start=1):
        descriptor = stage_descriptor(stage)
        completed_weight = sum(_stage_weight(item) for item in completed)
        spent = _pilot_purpose_already_served(config, stage, preset_id=preset_id)
        if spent is not None:
            # Measured 2026-08-28: validate-pilot-1000 costs 23.9 hours and
            # validate-pilot-10000 another 1.9. Their reuse key covers
            # mimic/storage.py, so any edit there -- a docstring included --
            # buys those 26 hours again. That is the provenance guard working
            # correctly; what is wrong is re-proving something already
            # superseded.
            #
            # These stages exist to show the pipeline runs and scales *before*
            # days are committed to the full extraction. The project says so
            # itself: "proves the analysis path produces readable tables",
            # "a last look at the analysis path at scale". Once that
            # extraction has completed on the full cohort, the question they
            # answer has been answered by the thing they were guarding.
            #
            # Nothing downstream reads their artifacts -- the report and audit
            # stages take the main run id, not the pilot roots -- so this
            # changes no result. It is recorded rather than silent.
            log.write(
                "stage",
                stage=stage,
                event="skipped",
                decision="skipped",
                reason=spent,
                position=index,
            )
            if progress is not None:
                progress(
                    f"Schritt {index}/{len(spec.stage_ids)}: "
                    f"{descriptor.display_name_de} entfaellt -- {spent}"
                )
            if stage not in completed:
                completed.append(stage)
            live.finish_stage(stage)
            continue
        if stage in completed:
            log.write(
                "stage",
                stage=stage,
                event="skipped",
                decision="skipped",
                reason="im Laufmanifest als abgeschlossen, Vertragshash unveraendert",
                position=index,
            )
            if progress is not None:
                progress(
                    f"Schritt {index}/{len(spec.stage_ids)}: "
                    f"{descriptor.display_name_de} bereits sicher abgeschlossen"
                )
            continue
        message = (
            f"Schritt {index}/{len(spec.stage_ids)}: "
            f"{descriptor.display_name_de}"
        )
        if progress is not None:
            progress(message)
        stage_weight = _stage_weight(stage)
        live.begin_stage(stage)
        if job_control is not None:
            job_control.checkpoint(
                message,
                current=int(round(completed_weight * _PROGRESS_SCALE)),
                total=int(round(total_weight * _PROGRESS_SCALE)),
            )
        stage_fraction_high_water = 0.0

        def _stage_progress(
            inner_message: str,
            *,
            _completed_weight: float = completed_weight,
            _current_stage_weight: float = stage_weight,
        ) -> None:
            nonlocal stage_fraction_high_water
            inner_message = _with_grant_warning(inner_message)
            if stage == "extract-features":
                live.progress.feature_runtime_contract = dict(log.feature_runtime_contract)
            if progress is not None:
                progress(inner_message)
            fraction = _stage_fraction_high_water(
                stage_fraction_high_water,
                inner_message,
            )
            from .live_progress import UNIFORM_PROGRESS_STAGES
            if stage not in UNIFORM_PROGRESS_STAGES:
                fraction = 0.0  # Endpoint/table counts are not equal-cost work.
            stage_fraction_high_water = fraction
            live.update(fraction, inner_message)
            if job_control is None:
                return
            overall = _completed_weight + fraction * _current_stage_weight
            job_control.checkpoint(
                inner_message,
                current=int(round(overall * _PROGRESS_SCALE)),
                total=int(round(total_weight * _PROGRESS_SCALE)),
            )

        stage_arguments = _stage_arguments(
            config,
            stage,
            preset_id=preset_id,
            run_id=run_id,
        )
        if stage in {
            "extract-features-pilot-1000",
            "validate-pilot-1000",
            "extract-features-pilot-10000",
            "validate-pilot-10000",
            # The pilots freeze the hypotheses, but the same guard must also
            # cover the later complete development extraction. Otherwise a
            # formula edit between Go/No-Go and the full run could silently
            # redefine the marker while keeping the inspected run_id.
            "extract-features",
        }:
            from .marker_freeze import assert_marker_freeze_current

            assert_marker_freeze_current(config, run_id=run_id)
        if stage in SIDE_STAGES:
            side_state = wait_for_side_job(
                config, stage, run_id=run_id, job_control=job_control,
                progress=progress,
            )
            if side_state is not None:
                log.write(
                    "stage",
                    stage=stage,
                    event="policy",
                    position=index,
                    side_job_state=side_state,
                    reason=(
                        "parallel job finished; the stage call reuses its artifact"
                        if side_state == "completed"
                        else "parallel job did not complete; computed in line"
                    ),
                )
        service_stage = _service_stage_id(stage)
        with stage_worker_memory_scope(
            service_stage,
            default_allocation_gb=config.duckdb_memory_limit_gb,
            default_reserve_gb=config.os_memory_reserve_gb,
        ) as memory_policy, active_run_log(log), log.stage_scope(
            stage,
            # The stage's own reuse check runs inside; it reports its verdict
            # as a separate "reuse" record via StageManifest.matches, so a
            # two-second stage here is visibly a reuse and not a mystery.
            decision="offered",
            reason=f"Schritt {index}/{len(spec.stage_ids)}",
        ):
            log.write(
                "stage",
                event="policy",
                position=index,
                service_stage=service_stage,
                memory_policy=memory_policy.to_dict(),
            )
            set_runtime_stage = getattr(
                job_control,
                "set_runtime_stage",
                None,
            )
            if callable(set_runtime_stage):
                set_runtime_stage(
                    service_stage,
                    state="running",
                    memory_policy=memory_policy.to_dict(),
                )
            result = run_service_stage(
                config,
                service_stage,
                stage_arguments,
                progress=_stage_progress,
                # job_control stays None for the inner stage call: this
                # function owns all job_control.checkpoint() calls for the
                # whole guided run via _stage_progress above, so the inner
                # stage's own progress(message) calls (still forwarded to the
                # caller's raw progress for display/logging) don't ALSO
                # trigger a second, differently-scaled checkpoint.
                job_control=None,
            )
            # What the stage produced, from what it already returns. The
            # column existed and was always empty, which promises a number the
            # log never had.
            counts = dict(getattr(result, "row_counts", {}) or {})
            if counts:
                log.write(
                    "stage",
                    service_stage=service_stage,
                    event="rows",
                    position=index,
                    rows=int(sum(int(value) for value in counts.values())),
                    row_counts=counts,
                )
            if callable(set_runtime_stage):
                set_runtime_stage(
                    service_stage,
                    state="completed",
                    memory_policy=memory_policy.to_dict(),
                )
        if stage == "audit-biomarkers":
            # The service normally performs this freeze itself. Keep the
            # workflow-level invariant as well so alternative service
            # adapters cannot accidentally reach a pilot without it.
            from .marker_freeze import freeze_marker_inventory

            freeze_marker_inventory(
                config,
                run_id=run_id,
                finalize_inputs=True,
            )
        _assert_pilot_gate(stage, result)
        if stage not in completed:
            completed.append(stage)
        live.finish_stage(stage)
        _launch_ready_side_jobs()
        _write_workflow_manifest(
            manifest_path,
            spec,
            workflow_hash,
            completed,
            status="running",
            project_config_hash=project_config_hash,
            operational_config_hash=project_operational_config_hash,
        )

    _write_workflow_manifest(
        manifest_path,
        spec,
        workflow_hash,
        completed,
        status="complete",
        project_config_hash=project_config_hash,
        operational_config_hash=project_operational_config_hash,
    )
    log.write(
        "workflow",
        event="end",
        run_id=run_id,
        completed_stages=len(completed),
        stage_count=len(spec.stage_ids),
    )
    live.finish_run()
    return GuidedWorkflowResult(
        study_id=spec.study_id,
        run_id=run_id,
        preset_id=preset_id,
        manifest_path=manifest_path,
        completed_stages=tuple(completed),
        workflow_hash=workflow_hash,
    )


_GRANT_WARNING_STATE = {"last": 0.0}
GRANT_WARNING_AFTER_SECONDS = 60.0
GRANT_WARNING_INTERVAL_SECONDS = 600.0


def _with_grant_warning(message: str) -> str:
    """Prefix a progress message when the grant has been invalid for > 60 s.

    Audit O3: without a live supervisor the job keeps running on one worker
    and nothing says so. Repeated at most every ten minutes.
    """
    import time as _time

    from .resource_grants import invalid_grant_seconds

    invalid = invalid_grant_seconds()
    if invalid is None or invalid < GRANT_WARNING_AFTER_SECONDS:
        return message
    now = _time.monotonic()
    if now - _GRANT_WARNING_STATE["last"] < GRANT_WARNING_INTERVAL_SECONDS:
        return message
    _GRANT_WARNING_STATE["last"] = now
    return (
        f"Warnung: seit {invalid / 60.0:.0f} min keine gueltige Ressourcen-"
        "Zuteilung -- laeuft der Supervisor? Rechnet mit einem Worker. | "
        + message
    )


def _stage_arguments(
    config: MimicProjectConfig,
    stage: str,
    *,
    preset_id: str,
    run_id: str,
) -> dict[str, Any]:
    # Same reason as in guided_study_spec: the branches below compare the
    # identifier directly, so a retired name must become canonical first.
    preset_id = resolve_preset_id(preset_id)
    preset = get_analysis_preset(preset_id)
    policy = preset.policy
    pilot_size = _pilot_size(stage)
    scaling_pilot = pilot_size is not None and pilot_size >= 10_000
    # Both pilots exercise the exact frozen marker inventory.  The previous
    # 10k shortcut retained only Core+Sentinel and silently omitted 166/180
    # fixed-plane 2D markers because those deliberately live in Discovery
    # Reserve.  The 10k pilot still disables bootstrap/search expansion below;
    # broad feature coverage is pipeline QC, not outcome-based marker choice.
    stage_panel_ids = policy_panel_ids(preset_id)
    preview = preset_id == "fast_preview" or pilot_size is not None
    # The production-path preview runs every stage with production settings,
    # but its cohort is a capped, endpoint-enriched extraction. The screen's
    # own guard refuses such artifacts without the explicit preview flag,
    # because prevalence-dependent metrics and probability models are not
    # interpretable on them. Only that flag is set; nothing else is reduced.
    capped_cohort = preview or preset_id == PRODUCTION_PREVIEW_PRESET_ID
    max_records = pilot_size if pilot_size is not None else (
        1_000 if preset_id == "fast_preview"
        else PRODUCTION_PREVIEW_RECORDS if preset_id == PRODUCTION_PREVIEW_PRESET_ID
        else None
    )
    effective_run_id = (
        run_id
        if pilot_size is None or preset_id == "fast_preview"
        else f"{run_id}-pilot-{pilot_size}"
    )
    common = {"force": False}
    if stage == "marker-redundancy":
        return {
            **common,
            "run_id": effective_run_id,
            "panel_ids": list(stage_panel_ids),
            "feature_glob": _feature_glob(
                config,
                stage_panel_ids,
                max_records=max_records,
            ),
        }
    if stage == "repeatability":
        return {
            **common,
            "run_id": effective_run_id,
            "panel_ids": list(stage_panel_ids),
            # Passed explicitly, like every sibling stage. Left implicit, this
            # stage fell back to its own default of ``<panel>/**/*.parquet``,
            # which matched the pilot partitions and a four-column failure
            # report alongside the 2,563 real ones.
            "feature_glob": _feature_glob(
                config,
                stage_panel_ids,
                max_records=max_records,
            ),
            "max_markers": 60 if preview else None,
        }
    if stage == "parse-notes":
        # A guided run must not hard-fail just because MIMIC-IV-Note (a
        # separately credentialed download) is absent. The stage then writes
        # a non-complete manifest plus a coverage artifact, and the
        # note-dependent association domains are recorded as not assessed --
        # so "absent" never reads as "screened, nothing found".
        return {
            **common,
            "allow_missing_note_sources": True,
        }
    if stage == "inventory":
        return {**common, "count_rows": False}
    if stage == "benchmark":
        benchmark_path = config.jobs_root / "hardware_benchmark.json"
        return {
            **common,
            # Up to the machine's capacity (audit E12): stopping at 7 made the
            # recommendation unable to exceed 7 on 8 cores / 16 threads.
            # ``benchmark_resources`` clamps every value to the logical cores.
            "candidates": (1, 2, 4, 6, 8, 10, 12, 14, 15),
            "tasks_per_worker": 3,
            "apply_recommendation": not benchmark_path.is_file(),
        }
    if stage == "audit-biomarkers":
        return {
            **common,
            "run_id": run_id,
            "output_root": str(config.work_root / "audits" / "biomarkers"),
            "static_only": False,
        }
    if stage == "pilot-go-no-go":
        return {
            **common,
            "run_id": run_id,
            "preset_id": preset_id,
        }
    if stage == "validate-delineation":
        max_records = (
            3
            if preset_id == "fast_preview"
            else None
        )
        return {
            **common,
            "benchmark_database_root": str(
                config.delineation_benchmark_root
            ),
            "output_path": str(
                config.resources_root
                / "delineation"
                / "benchmark_suite.json"
            ),
            # All six cohorts are attempted; only the release core can block.
            # INCART and STAFF III are separate PhysioNet downloads, and an
            # absent optional cohort used to abort a month-long run at this QC
            # step. Absent cohorts are recorded as unassessed, never imputed.
            "dataset_ids": (
                "ludb",
                "qtdb",
                "nstdb",
                "afdb",
                "incart",
                "staffiii",
            ),
            "required_dataset_ids": ("ludb", "qtdb", "nstdb"),
            "max_records": max_records,
            "max_seconds_per_record": 120,
            "strict": preset_id != "fast_preview",
        }
    if _service_stage_id(stage) == "extract-features":
        return {
            **common,
            "panel_ids": stage_panel_ids,
            "include_locked_test": False,
            "max_records": max_records,
            "partition_size": 250,
            "micro_batch_size": config.feature_micro_batch_size,
        }
    if stage == "physiological-gap-markers":
        return {
            **common,
            "run_id": effective_run_id,
            "panel_ids": ("primary_core",),
            "feature_glob": _feature_glob(
                config,
                stage_panel_ids,
                max_records=max_records,
            ),
        }
    if _service_stage_id(stage) == "validate":
        feature_glob = _feature_glob(
            config,
            stage_panel_ids,
            max_records=max_records,
        )
        return {
            **common,
            "run_id": effective_run_id,
            "panel_ids": stage_panel_ids,
            "feature_glob": feature_glob,
            "bootstrap_iterations": (
                0
                if scaling_pilot
                else min(200, policy.bootstrap_iterations)
                if pilot_size is not None
                else policy.bootstrap_iterations
            ),
            "include_predefined_scores": True,
            "include_ptb_frozen_scores": pilot_size is None,
            "include_exploratory_scores": True,
            "include_all_ecg_secondary": not preview,
            "grouped_score_folds": (
                min(3, policy.grouped_folds)
                if pilot_size is not None
                else policy.grouped_folds
            ),
            "grouped_score_repeats": (
                1 if pilot_size is not None else policy.grouped_cv_repeats
            ),
            "grouped_score_representatives_per_family": (
                1
                if scaling_pilot
                else min(2, policy.score_representatives_per_family)
                if pilot_size is not None
                else policy.score_representatives_per_family
            ),
            "grouped_score_selected_pairs": (
                1
                if scaling_pilot
                else min(4, policy.score_validation_pairs)
                if pilot_size is not None
                else policy.score_validation_pairs
            ),
            "grouped_score_selected_triples": (
                0
                if scaling_pilot
                else min(2, policy.score_validation_triples)
                if pilot_size is not None
                else policy.score_validation_triples
            ),
            "grouped_score_triple_seed_pairs": (
                0
                if scaling_pilot
                else min(10, policy.score_triple_seed_pairs)
                if pilot_size is not None
                else policy.score_triple_seed_pairs
            ),
            "maximum_component_abs_correlation": (
                policy.maximum_component_abs_correlation
            ),
            "minimum_score_incremental_auc": (
                policy.minimum_score_incremental_auc
            ),
            "maximum_criterion_single_candidates": (
                1
                if scaling_pilot
                else 2
                if pilot_size is not None
                else 10
                if preset_id == "exhaustive_research"
                else 6
            ),
            "maximum_criterion_combination_candidates": (
                0
                if scaling_pilot
                else 1
                if pilot_size is not None
                else 10
                if preset_id == "exhaustive_research"
                else 3
            ),
            "preview": capped_cohort,
        }
    if stage == "discover-phenome":
        return {
            **common,
            "run_id": run_id,
            "panel_ids": policy_panel_ids(preset_id),
            "feature_glob": _feature_glob(
                config,
                policy_panel_ids(preset_id),
                max_records=None,
            ),
            "bootstrap_iterations": policy.bootstrap_iterations,
            "future_horizons_days": (30, 90, 365),
            "max_future_phenotypes": None,
        }
    if stage == "therapy-response":
        # Falls through to the bare default otherwise, and the dispatcher then
        # analyses only Core plus Sentinel -- 320 of 1,753 markers.
        return {
            **common,
            "run_id": run_id,
            "panel_ids": policy_panel_ids(preset_id),
            "feature_glob": _feature_glob(
                config,
                policy_panel_ids(preset_id),
                max_records=max_records,
            ),
            "bootstrap_iterations": (
                0 if scaling_pilot else min(500, policy.bootstrap_iterations)
            ),
        }
    if stage == "external-validation":
        # Two defaults bite here: the output lands under a generic "external"
        # directory instead of this run, and only Core is transported -- 80 of
        # 1,753 markers, on the only cohorts that can speak to generalisation.
        return {
            **common,
            "run_id": run_id,
            "panel_ids": policy_panel_ids(preset_id),
            "bootstrap_iterations": (
                0 if preview else min(500, policy.bootstrap_iterations)
            ),
            "max_records_per_source": 2_000 if preview else None,
        }
    if stage == "population-cohorts":
        return {
            **common,
            "run_id": run_id,
            # The clinically prespecified 320-marker set is suitable for
            # reference intervals. Broad discovery markers stay in MIMIC's
            # discovery arm until promoted, avoiding 1,600 additional
            # external hypotheses and several terabytes of redundant work.
            "panel_ids": ("primary_core", "sentinel"),
            "max_records_per_cohort": (
                1_000 if preview
                else PRODUCTION_PREVIEW_EXTERNAL_RECORDS
                if preset_id == PRODUCTION_PREVIEW_PRESET_ID
                else None
            ),
            "minimum_signal_quality": 0.0,
            "workers": config.workers,
            "verify_source_sha256": not preview,
        }
    if stage == "screen-broad-axes":
        return {
            **common,
            "run_id": run_id,
            "panel_ids": policy_panel_ids(preset_id),
            "feature_glob": _feature_glob(
                config,
                policy_panel_ids(preset_id),
                max_records=max_records,
            ),
            "max_markers": 60 if preview else None,
        }
    if stage == "score-discovery":
        return {**common, "run_id": run_id, "panel_ids": policy_panel_ids(preset_id),
                "feature_glob": _feature_glob(config, policy_panel_ids(preset_id), max_records=max_records),
                "preview": capped_cohort, "max_endpoints": 6}
    if stage == "deep-combination-search":
        # Without this the stage falls through to the bare default and gets
        # the wrong run id (writing into a foreign run directory), the wrong
        # panels (missing every Discovery Reserve marker, which is what the
        # search exists for) and a feature glob that may match nothing at all.
        return {
            **common,
            "run_id": run_id,
            "panel_ids": policy_panel_ids(preset_id),
            "feature_glob": _feature_glob(
                config,
                policy_panel_ids(preset_id),
                max_records=max_records,
            ),
            "max_depth": 12 if preset_id == "exhaustive_research" else 6,
            "beam_width": 8 if preset_id == "exhaustive_research" else 4,
            # A preview must not spend an open-ended stage on a preview.
            "seconds_per_endpoint": 300.0 if preview else None,
        }
    if stage in {"report", "audit"}:
        return {
            **common,
            "run_id": run_id,
            "feature_glob": _feature_glob(
                config,
                policy_panel_ids(preset_id),
                max_records=max_records,
            ),
            "minimum_bootstrap_iterations": (
                0 if preview else policy.bootstrap_iterations
            ),
            # A bounded cohort fails the completeness checks by construction
            # (capped features, no phenome scan); the preview writes the
            # audit and reads it instead of ending on it.
            "strict": not capped_cohort,
            "deep_feature_verification": not preview,
        }
    # Every remaining stage -- the control panel and the reading stages after
    # it -- works on this run's own results. Without the identifier each fell
    # back to the service default ``mimic-development-guided``, which is the
    # name manual runs used to carry; under a generated identifier the control
    # panel then looked for screening results that were written elsewhere.
    # Found by the production-path preview, four hours after ``validate``.
    # The reading stages that take features get this run's feature glob as
    # well: their service default is the complete development extraction,
    # which a bounded preview does not have.
    return {
        **common,
        "run_id": run_id,
        "panel_ids": stage_panel_ids,
        "feature_glob": _feature_glob(config, stage_panel_ids, max_records=max_records),
    }


def _service_stage_id(stage: str) -> str:
    if stage.startswith("extract-features-pilot-"):
        return "extract-features"
    if stage.startswith("validate-pilot-"):
        return "validate"
    return stage


def _pilot_size(stage: str) -> Optional[int]:
    for prefix in ("extract-features-pilot-", "validate-pilot-"):
        if stage.startswith(prefix):
            value = stage.removeprefix(prefix)
            if value.isdigit() and int(value) > 0:
                return int(value)
    return None


def _pilot_purpose_already_served(
    config: MimicProjectConfig,
    stage: str,
    *,
    preset_id: str = STUDY_PRESET_ID,
) -> Optional[str]:
    """Why a pilot stage need not run again, or None if it must.

    The relation is one-directional and simple: a pilot de-risks the full
    extraction. If that extraction has finished on the full cohort, the risk
    it was checking for did not materialise, and running the check again
    tells nobody anything new.

    Deliberately strict about the evidence. The full manifest must say
    ``complete`` and every output it recorded must still be on disk -- an
    interrupted or half-deleted extraction is not proof of anything, and must
    send the pilots back to work.
    """

    if _pilot_size(stage) is None:
        return None
    # The panel key has to come from the run's own preset. Reading it from the
    # study preset would point a preview run at a feature set it never writes,
    # and a preview has no full extraction to be superseded by anyway.
    panel_key = "_".join(policy_panel_ids(preset_id))
    manifest_path = (
        Path(config.features_root) / panel_key / "development"
        / "features.manifest.json"
    )
    if not manifest_path.is_file():
        return None
    try:
        manifest = StageManifest.load(manifest_path)
    except (OSError, ValueError, KeyError, TypeError):
        # A manifest that cannot be read is not evidence of anything; the
        # pilots then run. Deliberately not a bare ``except``: the first
        # version had one and it swallowed a NameError, which silently turned
        # the whole rule off while every check below still looked right.
        return None
    if manifest.status != "complete":
        return None
    # A completed extraction answers the pilot's question only for the exact
    # feature implementation that is about to run.  In particular, a 3.0
    # full store must never retire the 3.1 pilots after marker formulas or
    # extraction semantics changed.
    from .feature_extraction import FEATURE_STAGE_VERSION

    if manifest.stage_version != FEATURE_STAGE_VERSION:
        return None
    if not all(Path(item).exists() for item in manifest.output_paths):
        return None
    rows = int(dict(manifest.row_counts or {}).get("successful_ecgs", 0))
    return (
        "Vollextraktion bereits abgeschlossen "
        f"({rows:,} ECGs) -- der Pilot sollte genau sie absichern"
    )


def _assert_pilot_gate(stage: str, result: object) -> None:
    size = _pilot_size(stage)
    if size is None:
        return
    status = str(getattr(result, "status", ""))
    if status != "complete":
        raise RuntimeError(
            f"Pilot {size:,} wurde nicht vollstaendig abgeschlossen."
        )
    row_counts = dict(getattr(result, "row_counts", {}) or {})
    if stage.startswith("extract-features-pilot-"):
        selected = int(row_counts.get("selected_ecgs", 0))
        successful = int(row_counts.get("successful_ecgs", 0))
        if selected != size:
            raise RuntimeError(
                f"Pilot-Gate erwartete {size:,} ECGs, erhielt {selected:,}."
            )
        if successful < max(1, int(round(0.80 * selected))):
            raise RuntimeError(
                "Pilot-Gate gestoppt: weniger als 80 % der ECGs konnten "
                "technisch ausgewertet werden."
            )
    elif sum(int(value) for value in row_counts.values()) <= 0:
        raise RuntimeError(
            f"Pilot-Gate {size:,} erzeugte keine auswertbaren Vorschautabellen."
        )


def policy_panel_ids(preset_id: str) -> tuple[str, ...]:
    return tuple(get_analysis_preset(preset_id).panel_ids)


def guided_service_stage_ids() -> tuple[str, ...]:
    stages: list[str] = []
    for preset_id in (
        "fast_preview",
        "recommended_clinical",
        "exhaustive_research",
    ):
        spec = guided_study_spec(
            preset_id=preset_id,
            run_id="contract-audit",
            include_external_validation=True,
        )
        stages.extend(_service_stage_id(stage) for stage in spec.stage_ids)
    return tuple(dict.fromkeys(stages))


def _feature_glob(
    config: MimicProjectConfig,
    panel_ids: tuple[str, ...],
    *,
    max_records: Optional[int],
) -> str:
    scope = (
        f"smoke-development-{int(max_records)}"
        if max_records is not None
        else "development"
    )
    return str(
        config.features_root
        / "_".join(panel_ids)
        / scope
        / "split=*"
        / "part-*.parquet"
    )


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        return dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}


def _write_workflow_manifest(
    path: Path,
    spec: StudyRunSpec,
    workflow_hash: str,
    completed: list[str],
    *,
    status: str,
    project_config_hash: str = "",
    operational_config_hash: str = "",
) -> None:
    atomic_write_json(
        path,
        {
            "schema_version": GUIDED_WORKFLOW_VERSION,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "development_only": True,
            "locked_test_read": False,
            "study_spec": spec.to_dict(),
            "workflow_hash": workflow_hash,
            # Recorded separately from workflow_hash so an artifact reader can
            # tell *which* project configuration produced these stages without
            # having to reconstruct the composite hash.
            "project_config_hash": project_config_hash,
            "operational_config_hash": operational_config_hash,
            "completed_stages": list(completed),
            "declared_stage_dependencies": dependency_graph(spec.stage_ids),
            "remaining_stages": [
                stage for stage in spec.stage_ids if stage not in completed
            ],
        },
    )


__all__ = [
    "GUIDED_WORKFLOW_VERSION",
    "GuidedWorkflowResult",
    "enqueue_guided_study",
    "guided_study_spec",
    "guided_service_stage_ids",
    "policy_panel_ids",
    "run_guided_study",
]
