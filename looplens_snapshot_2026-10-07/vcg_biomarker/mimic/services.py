from __future__ import annotations

from datetime import datetime, timezone
import json
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from ..core.delineation.benchmark_suite import (
    DELINEATION_RELEASE_POLICY_VERSION,
    DELINEATION_SUITE_VERSION,
    delineation_implementation_hash,
    inventory_delineation_benchmarks,
    resolve_benchmark_root,
    run_delineation_benchmark_suite,
)
from .acceptance import (
    AcceptanceExpectations,
    run_development_acceptance_audit,
)
from .biomarker_audit import run_biomarker_audit
from .clinical_utility import assess_clinical_utility_from_files
from .config import MimicProjectConfig
from .evidence import build_clinical_evidence
from .event_ingest import ingest_clinical_event_tables
from .event_equivalence import (
    confirm_or_attest_clinical_event_evidence,
)
from .feature_extraction import (
    FeatureExtractionOptions,
    build_machine_report_labels,
    extract_mimic_features,
)
from .group_vcg import GroupVCGOptions, run_group_vcg_analysis
from .ingest import ingest_standardized_tables
from .inventory import build_inventory
from .jobs import JobControl
from .labels import (
    attest_development_label_artifacts,
    build_clinical_labels,
)
from .locking import freeze_validation_run
from .master_index import build_ecg_master_index
from .models import ReportSpec, ReportType
from .note_labels import build_note_labels
from .panels import STUDY_PANEL_IDS, default_feature_panels
from .physiological_gap_models import run_physiological_gap_discovery
from .ptb_transfer import DEFAULT_PTB_TRANSFER_RUN_ID
from .report_builder import build_configured_report
from .reports import generate_run_reports
from .resources import benchmark_resources
from .snapshots import create_artifact_snapshot
from .storage import atomic_write_json, content_hash, file_sha256
from .validation import DevelopmentValidationOptions, run_development_validation
from .feature_extraction import feature_implementation_hash
from .fiducial_limitation import build_sensitivity_map


ProgressCallback = Callable[[str], None]

DELINEATION_CACHE_VERSION = "1.0.0"


def _directory_stat_fingerprint(root: Path) -> dict[str, Any]:
    """Cheap, deterministic identity for an immutable benchmark release."""

    entries: list[dict[str, int | str]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        stat = path.stat()
        entries.append(
            {
                "path": str(path.relative_to(root)).replace("\\", "/"),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return {
        "root": str(root.resolve()),
        "file_count": len(entries),
        "content_stat_hash": content_hash(entries),
    }


def _delineation_cache_request(
    database_root: Path,
    *,
    dataset_ids: tuple[str, ...],
    required_dataset_ids: tuple[str, ...],
    additional_roots: tuple[str, ...],
    max_records: Optional[int],
    max_seconds_per_record: int,
    strict: bool,
) -> dict[str, Any]:
    roots: dict[str, Any] = {}
    for dataset_id in dataset_ids:
        resolved = resolve_benchmark_root(
            database_root,
            dataset_id,
            additional_roots=additional_roots,
        )
        roots[dataset_id] = (
            {"status": "missing"}
            if resolved is None
            else _directory_stat_fingerprint(resolved)
        )
    return {
        "cache_version": DELINEATION_CACHE_VERSION,
        "implementation_hash": delineation_implementation_hash(),
        "suite_version": DELINEATION_SUITE_VERSION,
        "release_policy_version": DELINEATION_RELEASE_POLICY_VERSION,
        "dataset_ids": list(dataset_ids),
        "required_dataset_ids": list(required_dataset_ids),
        "max_records": max_records,
        "max_seconds_per_record": int(max_seconds_per_record),
        "strict": bool(strict),
        "datasets": roots,
    }


def _load_cached_delineation_suite(
    output_path: Path,
    request: dict[str, Any],
) -> Optional[dict[str, Any]]:
    cache_path = output_path.with_suffix(output_path.suffix + ".cache.json")
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    if (
        cache.get("request_hash") != content_hash(request)
        or cache.get("output_sha256") != file_sha256(output_path)
        or payload.get("status") != "complete"
        or payload.get("implementation_hash")
        != request["implementation_hash"]
        or payload.get("suite_version") != DELINEATION_SUITE_VERSION
        or payload.get("release_policy_version")
        != DELINEATION_RELEASE_POLICY_VERSION
    ):
        return None
    return payload


def _write_delineation_cache_attestation(
    output_path: Path,
    request: dict[str, Any],
) -> Path:
    cache_path = output_path.with_suffix(output_path.suffix + ".cache.json")
    atomic_write_json(
        cache_path,
        {
            "cache_version": DELINEATION_CACHE_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "request_hash": content_hash(request),
            "request": request,
            "output_path": str(output_path.resolve()),
            "output_sha256": file_sha256(output_path),
        },
    )
    return cache_path

MIMIC_BACKEND_CONTRACT_VERSION = "1.9.0"
MANAGED_SERVICE_STAGE_IDS = frozenset(
    {
        "study-run",
        "inventory",
        "genetic-feasibility",
        "ingest",
        "ingest-events",
        "build-index",
        "build-evidence",
        "attest-event-evidence",
        "build-labels",
        "attest-labels",
        "parse-reports",
        "parse-notes",
        "extract-features",
        "therapy-response",
        "physiological-gap-markers",
        "marker-redundancy",
        "control-panel",
        "segment-availability",
        "reference-ranges",
        "transform-agreement",
        "reporting-guideline",
        "ext-icd-crosscheck",
        "two-stage-selection",
        "within-subject-change",
        "voltage-mass-discordance",
        "time-to-event",
        "occlusion-yield",
        "thesis-tests",
        "repeatability",
        "external-validation",
        "population-cohorts",
        "screen-broad-axes",
        "deep-combination-search",
        "score-discovery",
        "validate",
        "group-vcg",
        "freeze",
        "report",
        "report-build",
        "benchmark",
        "validate-delineation",
        "audit-biomarkers",
        "pilot-go-no-go",
        "assess-clinical-utility",
        "snapshot",
        "audit",
        "discover-phenome",
        "refresh-phenome-promotions",
    }
)


def run_service_stage(
    config: MimicProjectConfig,
    stage: str,
    arguments: Optional[Mapping[str, Any]] = None,
    *,
    progress: Optional[ProgressCallback] = None,
    job_control: Optional[JobControl] = None,
) -> Any:
    args = dict(arguments or {})
    callback = _managed_progress(progress, job_control)
    if stage == "genetic-feasibility":
        from .gwas_feasibility import run_gwas_feasibility

        return run_gwas_feasibility(
            config,
            force=bool(args.get("force", False)),
            progress=callback,
        )
    if stage == "assess-clinical-utility":
        callback("Clinical-utility metadata validation started")
        outputs = assess_clinical_utility_from_files(
            Path(args["dossier_path"]),
            Path(args["evidence_registry_path"]),
            Path(args["output_root"]),
        )
        callback("Clinical-utility metadata assessment complete")
        return outputs
    if stage == "study-run":
        from .workflow import run_guided_study

        # Passes the *raw* progress (not the regex-inferring `callback`
        # above): run_guided_study() owns job_control checkpointing for
        # the whole guided run itself, in weighted overall-progress units
        # spanning every stage -- if it were also wrapped through
        # _managed_progress here, that function's per-message regex
        # "X/Y" parsing would intermittently overwrite the weighted
        # current/total with whatever unrelated count last appeared in a
        # stage-internal message (e.g. a batch counter), which is
        # exactly the bug this was built to fix.
        return run_guided_study(
            config,
            args,
            progress=progress,
            job_control=job_control,
        )
    if stage == "inventory":
        return build_inventory(
            config,
            count_rows=bool(args.get("count_rows", False)),
            progress=callback,
        )
    if stage == "ingest":
        return ingest_standardized_tables(
            config,
            force=bool(args.get("force", False)),
            progress=callback,
        )
    if stage == "ingest-events":
        return ingest_clinical_event_tables(
            config,
            force=bool(args.get("force", False)),
            progress=callback,
        )
    if stage == "build-index":
        return build_ecg_master_index(
            config,
            force=bool(args.get("force", False)),
            progress=callback,
        )
    if stage == "build-evidence":
        return build_clinical_evidence(
            config,
            force=bool(args.get("force", False)),
            progress=callback,
        )
    if stage == "attest-event-evidence":
        # The shadow rebuild runs only on request (force) or when the
        # evidence is not directly event-backed (audit E9).
        return confirm_or_attest_clinical_event_evidence(
            config,
            full_equivalence=bool(args.get("force", False)),
            progress=callback,
        )
    if stage == "build-labels":
        return build_clinical_labels(
            config,
            force=bool(args.get("force", False)),
            progress=callback,
        )
    if stage == "attest-labels":
        return attest_development_label_artifacts(
            config,
            force=bool(args.get("force", False)),
            progress=callback,
        )
    if stage == "parse-reports":
        return build_machine_report_labels(
            config,
            force=bool(args.get("force", False)),
            progress=callback,
        )
    if stage == "parse-notes":
        # Running this stage without a note root produces zero rows that read
        # downstream like "screened, nothing found". Refuse instead, unless
        # the caller explicitly accepts an unassessed note layer.
        if config.mimic_note_root is None and not bool(
            args.get("allow_missing_note_sources", False)
        ):
            raise RuntimeError(
                "Stufe 'parse-notes' wurde angefordert, aber 'mimic_note_root' "
                "ist in dieser Konfiguration nicht gesetzt. Der Lauf würde "
                "leere Note-Evidenz erzeugen, die von 'geprüft, kein Befund' "
                "nicht zu unterscheiden ist. Entweder 'mimic_note_root' "
                "konfigurieren oder die Stufe bewusst mit "
                "allow_missing_note_sources=True überspringen."
            )
        return build_note_labels(
            config,
            force=bool(args.get("force", False)),
            progress=callback,
        )
    if stage == "extract-features":
        return extract_mimic_features(
            config,
            FeatureExtractionOptions(
                panel_ids=_tuple_argument(
                    args.get("panel_ids", ("primary_core", "sentinel"))
                ),
                include_locked_test=bool(args.get("include_locked_test", False)),
                locked_test_only=bool(args.get("locked_test_only", False)),
                freeze_run_id=_optional_string(args.get("freeze_run_id")),
                max_records=_optional_int(args.get("max_records")),
                partition_size=int(args.get("partition_size", 250)),
                force=bool(args.get("force", False)),
                execution_id=_optional_string(args.get("_execution_id")),
                micro_batch_size=int(
                    args.get(
                        "micro_batch_size",
                        config.feature_micro_batch_size,
                    )
                ),
            ),
            progress=callback,
        )
    if stage == "physiological-gap-markers":
        gap_panel_ids = _tuple_argument(
            args.get("panel_ids", ("primary_core",))
        )
        gap_feature_ids = _optional_tuple_argument(args.get("feature_ids"))
        if gap_feature_ids is None:
            panel_lookup = {
                panel.panel_id: panel for panel in default_feature_panels()
            }
            unknown_gap_panels = sorted(set(gap_panel_ids) - set(panel_lookup))
            if unknown_gap_panels:
                raise ValueError(
                    f"Unknown feature panels: {unknown_gap_panels}"
                )
            gap_feature_ids = tuple(
                dict.fromkeys(
                    feature_id
                    for panel_id in gap_panel_ids
                    for feature_id in panel_lookup[panel_id].feature_ids
                )
            )
        gap_feature_glob = _optional_string(args.get("feature_glob"))
        if gap_feature_glob is None:
            gap_feature_glob = str(
                config.features_root
                / "_".join(gap_panel_ids)
                / "development"
                / "split=*"
                / "part-*.parquet"
            )
        return run_physiological_gap_discovery(
            config,
            feature_glob=gap_feature_glob,
            feature_ids=gap_feature_ids,
            output_root=(
                config.runs_root
                / str(args.get("run_id", "mimic-development"))
                / "physiological_gap"
            ),
            random_seed=int(args.get("random_seed", config.random_seed)),
            progress=callback,
        )
    if stage == "therapy-response":
        from .therapy_response import run_therapy_response_analysis

        return run_therapy_response_analysis(
            config,
            run_id=str(args.get("run_id", "mimic-development")),
            panel_ids=_tuple_argument(
                args.get("panel_ids", ("primary_core", "sentinel"))
            ),
            feature_glob=_optional_string(args.get("feature_glob")),
            iterations=int(args.get("bootstrap_iterations", 2000)),
            seed=int(args.get("random_seed", config.random_seed)),
            force=bool(args.get("force", False)),
            progress=callback,
        )
    if stage == "external-validation":
        from .external_validation import (
            load_frozen_external_contract,
            run_external_validation,
        )

        if config.external_ecg_root is None:
            raise RuntimeError(
                "Stufe 'external-validation' wurde angefordert, aber "
                "'external_ecg_root' ist nicht konfiguriert. Externe Kohorten "
                "sind die einzige Evidenzquelle in diesem Projekt, die zur "
                "Generalisierbarkeit ueber ein Krankenhaus hinaus sprechen "
                "kann; ohne sie gibt es hier nichts zu berichten."
            )
        development_run_id = _optional_string(
            args.get("development_run_id", args.get("freeze_run_id"))
        )
        if development_run_id is None:
            raise RuntimeError(
                "Stufe 'external-validation' benoetigt development_run_id "
                "eines unveraenderlichen Freeze; Richtungen und Schwellen "
                "duerfen extern niemals neu gewaehlt werden."
            )
        frozen_contract = load_frozen_external_contract(
            config.runs_root / development_run_id
        )
        requested = _tuple_argument(
            args.get("panel_ids", ("primary_core",))
        )
        marker_ids = list(
            dict.fromkeys(
                item.model.feature_id
                for item in frozen_contract.markers
                if item.panel_id in requested
            )
        )
        if not marker_ids:
            raise RuntimeError(
                "Der Freeze enthaelt keine extern pruefbaren Marker in den "
                f"angeforderten Panels: {requested}"
            )
        limit = _optional_int(args.get("max_markers"))
        if limit is not None:
            marker_ids = marker_ids[:limit]
        return run_external_validation(
            config.external_ecg_root,
            identity_manifest=config.external_identity_manifest,
            identity_manifest_sha256=config.external_identity_manifest_sha256,
            frozen_contract=frozen_contract,
            marker_ids=tuple(marker_ids),
            output_root=(
                config.external_validation_root
                / str(args.get("run_id", "external"))
            ),
            max_records_per_source=_optional_int(
                args.get("max_records_per_source")
            ),
            powerline_hz=(
                float(args["powerline_hz"])
                if args.get("powerline_hz") is not None
                else None
            ),
            minimum_signal_quality=float(
                args.get("minimum_signal_quality", 0.0)
            ),
            workers=int(args.get("workers", config.workers)),
            bootstrap_iterations=int(args.get("bootstrap_iterations", 0)),
            progress=callback,
        )
    if stage == "population-cohorts":
        from .population_cohorts import run_population_cohort_analysis

        return run_population_cohort_analysis(
            config,
            run_id=str(args.get("run_id", "population-cohorts")),
            panel_ids=_tuple_argument(
                args.get("panel_ids", ("primary_core", "sentinel"))
            ),
            additional_marker_ids=_tuple_argument(
                args.get("additional_marker_ids", ())
            ),
            max_records_per_cohort=_optional_int(
                args.get("max_records_per_cohort")
            ),
            minimum_signal_quality=float(
                args.get("minimum_signal_quality", 0.0)
            ),
            workers=int(args.get("workers", config.workers)),
            verify_source_sha256=bool(args.get("verify_source_sha256", True)),
            force=bool(args.get("force", False)),
            progress=callback,
        )
    if stage == "marker-redundancy":
        from .marker_redundancy import run_marker_redundancy

        return run_marker_redundancy(
            config,
            run_id=str(args.get("run_id", "mimic-development-guided")),
            panel_ids=_tuple_argument(
                args.get(
                    "panel_ids",
                    STUDY_PANEL_IDS,
                )
            ),
            feature_glob=_optional_string(args.get("feature_glob")),
            force=bool(args.get("force", False)),
            progress=callback,
        )
    if stage == "control-panel":
        from .control_panel import run_control_panel

        return run_control_panel(
            config,
            run_id=str(args.get("run_id", "mimic-development-guided")),
            force=bool(args.get("force", False)),
            progress=callback,
        )
    if stage == "segment-availability":
        from .segment_availability import run_segment_availability

        return run_segment_availability(
            config,
            run_id=str(args.get("run_id", "mimic-development-guided")),
            panel_ids=_tuple_argument(
                args.get(
                    "panel_ids",
                    STUDY_PANEL_IDS,
                )
            ),
            feature_glob=_optional_string(args.get("feature_glob")),
            force=bool(args.get("force", False)),
            progress=callback,
        )
    if stage == "reference-ranges":
        from .reference_ranges import run_reference_ranges

        return run_reference_ranges(
            config,
            run_id=str(args.get("run_id", "mimic-development-guided")),
            panel_ids=_tuple_argument(
                args.get(
                    "panel_ids",
                    STUDY_PANEL_IDS,
                )
            ),
            feature_glob=_optional_string(args.get("feature_glob")),
            force=bool(args.get("force", False)),
            progress=callback,
        )
    if stage == "transform-agreement":
        from .transform_agreement import run_transform_agreement

        return run_transform_agreement(
            config,
            run_id=str(args.get("run_id", "mimic-development-guided")),
            panel_ids=_tuple_argument(
                args.get(
                    "panel_ids",
                    STUDY_PANEL_IDS,
                )
            ),
            feature_glob=_optional_string(args.get("feature_glob")),
            force=bool(args.get("force", False)),
            progress=callback,
        )
    if stage == "reporting-guideline":
        from .reporting_guideline import run_reporting_guideline

        return run_reporting_guideline(
            config,
            run_id=str(args.get("run_id", "mimic-development-guided")),
            force=bool(args.get("force", False)),
            progress=callback,
        )
    if stage == "ext-icd-crosscheck":
        from .ext_icd import run_ext_icd_crosscheck

        return run_ext_icd_crosscheck(
            config,
            run_id=str(args.get("run_id", "mimic-development-guided")),
            force=bool(args.get("force", False)),
            progress=callback,
        )
    if stage == "two-stage-selection":
        from .two_stage_selection import run_two_stage_selection

        return run_two_stage_selection(
            config,
            run_id=str(args.get("run_id", "mimic-development-guided")),
            force=bool(args.get("force", False)),
            progress=callback,
        )
    if stage == "within-subject-change":
        from .within_subject import run_within_subject_change

        return run_within_subject_change(
            config,
            run_id=str(args.get("run_id", "mimic-development-guided")),
            panel_ids=_tuple_argument(
                args.get(
                    "panel_ids",
                    STUDY_PANEL_IDS,
                )
            ),
            feature_glob=_optional_string(args.get("feature_glob")),
            force=bool(args.get("force", False)),
            preview=bool(args.get("preview", args.get("interactive_preview", False))),
            progress=callback,
        )
    if stage == "voltage-mass-discordance":
        from .voltage_mass import run_voltage_mass_discordance

        return run_voltage_mass_discordance(
            config,
            run_id=str(args.get("run_id", "mimic-development-guided")),
            panel_ids=_tuple_argument(
                args.get(
                    "panel_ids",
                    STUDY_PANEL_IDS,
                )
            ),
            feature_glob=_optional_string(args.get("feature_glob")),
            force=bool(args.get("force", False)),
            progress=callback,
        )
    if stage == "time-to-event":
        from .survival import run_time_to_event

        return run_time_to_event(
            config,
            run_id=str(args.get("run_id", "mimic-development-guided")),
            panel_ids=_tuple_argument(
                args.get(
                    "panel_ids",
                    STUDY_PANEL_IDS,
                )
            ),
            feature_glob=_optional_string(args.get("feature_glob")),
            force=bool(args.get("force", False)),
            progress=callback,
        )
    if stage == "occlusion-yield":
        from .occlusion import run_occlusion_yield

        return run_occlusion_yield(
            config,
            run_id=str(args.get("run_id", "mimic-development-guided")),
            panel_ids=_tuple_argument(
                args.get(
                    "panel_ids",
                    STUDY_PANEL_IDS,
                )
            ),
            feature_glob=_optional_string(args.get("feature_glob")),
            force=bool(args.get("force", False)),
            progress=callback,
        )
    if stage == "thesis-tests":
        from .thesis_tests import run_thesis_tests

        return run_thesis_tests(
            config,
            run_id=str(args.get("run_id", "mimic-development-guided")),
            panel_ids=_tuple_argument(args.get("panel_ids", STUDY_PANEL_IDS)),
            feature_glob=_optional_string(args.get("feature_glob")),
            force=bool(args.get("force", False)),
            progress=callback,
        )
    if stage == "repeatability":
        from .repeatability_stage import run_repeatability_analysis

        return run_repeatability_analysis(
            config,
            run_id=str(args.get("run_id", "mimic-development-guided")),
            feature_glob=_optional_string(args.get("feature_glob")),
            panel_ids=_tuple_argument(
                args.get("panel_ids", ("primary_core", "sentinel"))
            ),
            max_markers=_optional_int(args.get("max_markers")),
            force=bool(args.get("force", False)),
            progress=callback,
        )
    if stage == "screen-broad-axes":
        from .broad_axis_stage import run_broad_axis_screen

        return run_broad_axis_screen(
            config,
            run_id=str(args.get("run_id", "mimic-development-guided")),
            panel_ids=_tuple_argument(
                args.get("panel_ids", ("primary_core", "sentinel"))
            ),
            feature_glob=_optional_string(args.get("feature_glob")),
            axes=_optional_tuple_argument(args.get("axes")),
            max_markers=_optional_int(args.get("max_markers")),
            force=bool(args.get("force", False)),
            progress=callback,
        )
    if stage == "score-discovery":
        from .score_stage import run_score_discovery

        return run_score_discovery(config, run_id=str(args.get("run_id", "mimic-development-guided")),
            panel_ids=_tuple_argument(args.get("panel_ids", STUDY_PANEL_IDS)),
            feature_glob=_optional_string(args.get("feature_glob")),
            endpoint_ids=_optional_tuple_argument(args.get("endpoint_ids")),
            max_endpoints=int(args.get("max_endpoints", 6)), preview=bool(args.get("preview", False)),
            max_patients=_optional_int(args.get("max_patients", 50000)),
            force=bool(args.get("force", False)), progress=callback)
    if stage == "deep-combination-search":
        from .combination_stage import run_deep_combination_search

        return run_deep_combination_search(
            config,
            run_id=str(args.get("run_id", "mimic-development-guided")),
            endpoint_ids=_optional_tuple_argument(args.get("endpoint_ids")),
            panel_ids=_tuple_argument(
                args.get("panel_ids", ("primary_core", "sentinel"))
            ),
            feature_glob=_optional_string(args.get("feature_glob")),
            max_depth=_optional_int(args.get("max_depth", 12)),
            patience=_optional_int(args.get("patience", 2)),
            minimum_increment=float(args.get("minimum_increment", 0.002)),
            beam_width=int(args.get("beam_width", 8)),
            seconds_per_endpoint=(
                float(args["seconds_per_endpoint"])
                if args.get("seconds_per_endpoint") is not None
                else None
            ),
            run_until_stopped=bool(args.get("run_until_stopped", False)),
            max_rows_per_endpoint=_optional_int(
                args.get("max_rows_per_endpoint")
            ),
            force=bool(args.get("force", False)),
            progress=callback,
        )
    if stage == "validate":
        return run_development_validation(
            config,
            DevelopmentValidationOptions(
                run_id=str(args.get("run_id", "mimic-development")),
                endpoint_ids=_optional_tuple_argument(args.get("endpoint_ids")),
                panel_ids=_tuple_argument(
                    args.get("panel_ids", ("primary_core", "sentinel"))
                ),
                feature_glob=_optional_string(args.get("feature_glob")),
                bootstrap_iterations=_optional_int(
                    args.get("bootstrap_iterations")
                ),
                include_predefined_scores=bool(
                    args.get("include_predefined_scores", True)
                ),
                include_ptb_frozen_scores=bool(
                    args.get("include_ptb_frozen_scores", True)
                ),
                ptb_transfer_run_id=str(
                    args.get(
                        "ptb_transfer_run_id",
                        DEFAULT_PTB_TRANSFER_RUN_ID,
                    )
                ),
                include_exploratory_scores=bool(
                    args.get("include_exploratory_scores", True)
                ),
                include_all_ecg_secondary=bool(
                    args.get("include_all_ecg_secondary", True)
                ),
                grouped_score_folds=int(
                    args.get("grouped_score_folds", 5)
                ),
                grouped_score_repeats=int(
                    args.get("grouped_score_repeats", 1)
                ),
                grouped_score_representatives_per_family=int(
                    args.get("grouped_score_representatives_per_family", 2)
                ),
                grouped_score_selected_pairs=int(
                    args.get("grouped_score_selected_pairs", 1)
                ),
                grouped_score_selected_triples=int(
                    args.get("grouped_score_selected_triples", 1)
                ),
                grouped_score_triple_seed_pairs=int(
                    args.get("grouped_score_triple_seed_pairs", 25)
                ),
                maximum_component_abs_correlation=float(
                    args.get(
                        "maximum_component_abs_correlation",
                        0.90,
                    )
                ),
                minimum_score_incremental_auc=float(
                    args.get("minimum_score_incremental_auc", 0.005)
                ),
                maximum_criterion_single_candidates=int(
                    args.get("maximum_criterion_single_candidates", 6)
                ),
                maximum_criterion_combination_candidates=int(
                    args.get("maximum_criterion_combination_candidates", 3)
                ),
                preview=bool(
                    args.get(
                        "preview",
                        args.get("interactive_preview", False),
                    )
                ),
                force=bool(args.get("force", False)),
            ),
            progress=callback,
        )
    if stage == "group-vcg":
        return run_group_vcg_analysis(
            config,
            GroupVCGOptions(
                run_id=str(args.get("run_id", "mimic-development")),
                endpoint_id=str(args.get("endpoint_id", "")),
                split=str(args.get("split", "validation")),
                panel_ids=_tuple_argument(
                    args.get("panel_ids", ("primary_core", "sentinel"))
                ),
                feature_glob=_optional_string(args.get("feature_glob")),
                max_per_class=int(args.get("max_per_class", 40)),
                minimum_per_class=int(args.get("minimum_per_class", 10)),
                candidate_multiplier=int(
                    args.get("candidate_multiplier", 4)
                ),
                quality_minimum=float(args.get("quality_minimum", 0.75)),
                bootstrap_iterations=int(
                    args.get("bootstrap_iterations", 500)
                ),
                force=bool(args.get("force", False)),
                execution_id=_optional_string(args.get("_execution_id")),
            ),
            progress=callback,
        )
    if stage == "freeze":
        return freeze_validation_run(
            config,
            str(args.get("run_id", "mimic-development")),
        )
    if stage == "report":
        return generate_run_reports(
            config,
            str(args.get("run_id", "mimic-development")),
            feature_glob=_optional_string(args.get("feature_glob")),
            force=bool(args.get("force", False)),
            progress=callback,
        )
    if stage == "report-build":
        run_id = str(args.get("run_id", "mimic-development"))
        return build_configured_report(
            config,
            ReportSpec(
                report_id=str(
                    args.get("report_id", f"{run_id}-study-report")
                ),
                report_type=ReportType(
                    str(args.get("report_type", ReportType.STUDY.value))
                ),
                run_id=run_id,
                language=str(args.get("language", "de")),
                formats=_tuple_argument(
                    args.get(
                        "formats",
                        ("pdf", "html", "xlsx", "parquet"),
                    )
                ),
                endpoint_ids=_tuple_argument(args.get("endpoint_ids", ())),
                feature_ids=_tuple_argument(args.get("feature_ids", ())),
                panel_ids=_tuple_argument(
                    args.get("panel_ids", ("primary_core", "sentinel"))
                ),
                sections=_tuple_argument(
                    args.get(
                        "sections",
                        (
                            "summary",
                            "methods",
                            "cohort",
                            "performance",
                            "baselines",
                            "stability",
                            "limitations",
                            "provenance",
                        ),
                    )
                ),
                include_case_ids=bool(args.get("include_case_ids", False)),
                include_case_appendix=bool(
                    args.get("include_case_appendix", False)
                ),
            ),
            force=bool(args.get("force", False)),
            progress=callback,
        )
    if stage == "benchmark":
        result = benchmark_resources(
            config,
            candidates=_int_tuple_argument(
                args.get("candidates", (1, 2, 4, 6, 8, 10, 12, 14))
            ),
            tasks_per_worker=int(args.get("tasks_per_worker", 3)),
            progress=callback,
        )
        if bool(args.get("apply_recommendation", False)):
            config.workers = result.recommended_workers
            config.save()
        return result
    if stage == "validate-delineation":
        database_root = Path(
            args.get(
                "benchmark_database_root",
                config.delineation_benchmark_root,
            )
        )
        output_path = Path(
            args.get(
                "output_path",
                config.resources_root
                / "delineation"
                / "benchmark_suite.json",
            )
        )
        dataset_ids = _tuple_argument(
            args.get(
                "dataset_ids",
                ("ludb", "qtdb", "nstdb", "afdb", "incart", "staffiii"),
            )
        )
        required_dataset_ids = _tuple_argument(
            args.get(
                "required_dataset_ids",
                ("ludb", "qtdb", "nstdb"),
            )
        )
        additional_roots = (
            (str(config.external_ecg_root),)
            if config.external_ecg_root is not None
            else ()
        )
        max_records = _optional_int(args.get("max_records"))
        max_seconds_per_record = int(
            args.get("max_seconds_per_record", 120)
        )
        strict = bool(args.get("strict", False))
        inventory = inventory_delineation_benchmarks(database_root)
        cache_request = _delineation_cache_request(
            database_root,
            dataset_ids=dataset_ids,
            required_dataset_ids=required_dataset_ids,
            additional_roots=additional_roots,
            max_records=max_records,
            max_seconds_per_record=max_seconds_per_record,
            strict=strict,
        )
        result = _load_cached_delineation_suite(output_path, cache_request)
        if result is None:
            from .storage import _own_grant_workers

            # Records of QTDB and STAFF III in parallel (audit E7). The count
            # does not change any result and is not part of the cache key.
            record_workers = max(
                1,
                int(_own_grant_workers() or config.workers),
            )
            result = run_delineation_benchmark_suite(
                database_root,
                output_path,
                dataset_ids=dataset_ids,
                max_records=max_records,
                max_seconds_per_record=max_seconds_per_record,
                strict=strict,
                additional_roots=additional_roots,
                required_dataset_ids=required_dataset_ids,
                progress=callback,
                record_workers=record_workers,
            )
            _write_delineation_cache_attestation(output_path, cache_request)
        else:
            callback(
                "Delineationspruefung unveraendert und hashverifiziert "
                "wiederverwendet"
            )
        result["inventory"] = inventory.to_dict(orient="records")
        result["output_path"] = str(output_path.resolve())
        # A declared limitation is only honest if it reaches the markers.
        # The map is measured here, by the same stage and the same delineator
        # that just produced the gate, so it can never describe a different
        # implementation than the one certified.
        gate = result.get("release_gate") or {}
        landmarks = tuple(
            str(item) for item in (gate.get("affected_landmarks") or [])
        )
        if landmarks:
            callback(
                "Messe Marker-Reliabilitaet unter den deklarierten "
                f"Limitationen: {', '.join(landmarks)}"
            )
            map_path = build_sensitivity_map(
                benchmark_database_root=database_root,
                resources_root=config.resources_root,
                landmarks=landmarks,
                feature_implementation_hash=feature_implementation_hash(),
                progress=callback,
            )
            result["fiducial_sensitivity_map"] = (
                str(map_path) if map_path is not None else None
            )
        return result
    if stage == "audit-biomarkers":
        result = run_biomarker_audit(
            Path(
                args.get(
                    "output_root",
                    config.work_root / "audits" / "biomarkers",
                )
            ),
            dynamic=not bool(args.get("static_only", False)),
        )
        audit_run_id = _optional_string(args.get("run_id"))
        if audit_run_id is not None:
            from .marker_freeze import freeze_marker_inventory

            freeze_marker_inventory(
                config,
                run_id=audit_run_id,
                finalize_inputs=True,
            )
        return result
    if stage == "pilot-go-no-go":
        from .pilot_go_no_go import build_pilot_go_no_go_report

        return build_pilot_go_no_go_report(
            config,
            run_id=str(args.get("run_id", "mimic-development-guided")),
            preset_id=str(args.get("preset_id", "exhaustive_research")),
        )
    if stage == "snapshot":
        return create_artifact_snapshot(
            config,
            artifact_kind=str(args.get("artifact_kind", "features")),
            source_glob=str(args["source_glob"]),
            snapshot_id=_optional_string(args.get("snapshot_id")),
            expected_files=_optional_int(args.get("expected_files")),
            require_complete=bool(args.get("require_complete", False)),
            notes=_tuple_argument(args.get("notes", ())),
        )
    if stage == "audit":
        return run_development_acceptance_audit(
            config,
            run_id=str(args.get("run_id", "mimic-development")),
            ptb_run_id=str(
                args.get(
                    "ptb_run_id",
                    DEFAULT_PTB_TRANSFER_RUN_ID,
                )
            ),
            expectations=AcceptanceExpectations(
                ecg_count=int(args.get("expected_ecg_count", 800_035)),
                subject_count=int(
                    args.get("expected_subject_count", 161_352)
                ),
                echo_study_count=int(
                    args.get("expected_echo_study_count", 206_488)
                ),
                # None: taken from the extraction manifest (selected_ecgs).
                development_feature_rows=_optional_int(
                    args.get("expected_development_feature_rows")
                ),
                label_definition_count=int(
                    args.get("expected_label_definition_count", 63)
                ),
                minimum_bootstrap_iterations=int(
                    args.get("minimum_bootstrap_iterations", 1_000)
                ),
                group_vcg_endpoints=_tuple_argument(
                    args.get(
                        "group_vcg_endpoints",
                        (
                            "acute_mi_high_specificity",
                            "echo_lvh_strict",
                            "reduced_lvef",
                            "right_heart_strain_proxy",
                            "echo_lae",
                            "hyperkalemia_near_ecg",
                        ),
                    )
                ),
            ),
            require_freeze=bool(args.get("require_freeze", False)),
            strict=bool(args.get("strict", False)),
            deep_feature_verification=bool(
                args.get("deep_feature_verification", True)
            ),
            progress=callback,
        )
    if stage in ("evaluate-test", "ptb-evaluate-test"):
        raise PermissionError(
            f"{stage} cannot be queued as a background development job. "
            "Locked-test access requires an explicit foreground release action."
        )
    if stage == "discover-phenome":
        from .phenome import PhenomeDiscoveryOptions, run_phenome_discovery

        return run_phenome_discovery(
            config,
            PhenomeDiscoveryOptions.from_mapping(args),
            progress=callback,
        )
    if stage == "refresh-phenome-promotions":
        from .phenome_promotions import refresh_phenome_promotions

        return refresh_phenome_promotions(
            config,
            str(args.get("run_id", "mimic-development")),
            force=bool(args.get("force", False)),
            progress=callback,
        )
    raise ValueError(f"Unsupported managed stage: {stage}")


def _managed_progress(
    progress: Optional[ProgressCallback],
    job_control: Optional[JobControl],
) -> ProgressCallback:
    def callback(message: str) -> None:
        if progress is not None:
            progress(message)
        if job_control is None:
            return
        current, total = _parse_progress(message)
        job_control.checkpoint(
            message,
            current=current,
            total=total,
        )

    return callback


def _parse_progress(message: str) -> tuple[Optional[int], Optional[int]]:
    match = re.search(r"([\d,.]+)\s*/\s*([\d,.]+)", message)
    if not match:
        return None, None
    return _localized_int(match.group(1)), _localized_int(match.group(2))


def _localized_int(value: str) -> int:
    return int(value.replace(".", "").replace(",", ""))


def _tuple_argument(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    return tuple(str(item) for item in value)


def _optional_tuple_argument(value: Any) -> Optional[tuple[str, ...]]:
    if value in (None, ""):
        return None
    return _tuple_argument(value)


def _int_tuple_argument(value: Any) -> tuple[int, ...]:
    if isinstance(value, str):
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    return tuple(int(item) for item in value)


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    return int(value)


def _optional_string(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    return str(value)
