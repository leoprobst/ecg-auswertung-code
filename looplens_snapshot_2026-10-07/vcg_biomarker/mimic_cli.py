from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from datetime import datetime, timezone
from multiprocessing import freeze_support
from pathlib import Path
from typing import Optional

from .core.external_loader import ExternalECGMeta
from .core.delineation.benchmark_suite import (
    inventory_delineation_benchmarks,
    run_delineation_benchmark_suite,
)
from .mimic.acceptance import (
    AcceptanceExpectations,
    run_development_acceptance_audit,
)
from .mimic.biomarker_audit import run_biomarker_audit
from .mimic.clinical_utility import assess_clinical_utility_from_files
from .mimic.config import MimicProjectConfig, load_or_discover_config
from .mimic.ecg_lab import ECGCaseLibrary
from .mimic.event_ingest import ingest_clinical_event_tables
from .mimic.event_equivalence import (
    confirm_or_attest_clinical_event_evidence,
)
from .mimic.evidence import build_clinical_evidence
from .mimic.feature_extraction import (
    FeatureExtractionOptions,
    build_machine_report_labels,
    extract_mimic_features,
)
from .mimic.group_vcg import GroupVCGOptions, run_group_vcg_analysis
from .mimic.ingest import ingest_standardized_tables
from .mimic.inventory import build_inventory
from .mimic.jobs import (
    JOB_PROJECT_CONFIG_HASH_KEY,
    JOB_PROJECT_CONFIG_KEY,
    JobStore,
    remove_confirmed_job_run_manifest_root,
    validate_job_run_manifest_root,
)
from .mimic.labels import (
    attest_development_label_artifacts,
    build_clinical_labels,
)
from .mimic.locking import (
    LockedEvaluationOptions,
    evaluate_locked_test,
    freeze_validation_run,
    invalidate_test_lock,
)
from .mimic.master_index import build_ecg_master_index
from .mimic.models import ReportSpec, ReportType
from .mimic.note_labels import build_note_labels
from .mimic.phenome import PhenomeDiscoveryOptions, run_phenome_discovery
from .mimic.phenome_promotions import refresh_phenome_promotions
from .mimic.pilot_archive import archive_partial_feature_pilot
from .mimic.ptb_transfer import (
    DEFAULT_PTB_TRANSFER_RUN_ID,
    PTBRevalidationOptions,
    evaluate_ptb_locked_test,
    freeze_ptb_revalidation,
    run_ptb_revalidation,
)
from .mimic.report_builder import build_configured_report
from .mimic.reports import generate_run_reports
from .mimic.resources import benchmark_resources
from .mimic.snapshots import create_artifact_snapshot
from .mimic.supervisor import ensure_supervisor
from .mimic.panels import STUDY_PANEL_IDS
from .mimic.validation import (
    DevelopmentValidationOptions,
    run_development_validation,
)


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        command = args.command
        if command == "assess-clinical-utility":
            outputs = assess_clinical_utility_from_files(
                args.dossier,
                args.evidence_registry,
                args.output,
            )
            _print(
                json.dumps(
                    {
                        key: str(path.resolve())
                        for key, path in outputs.items()
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        config = _load_config(args)
        if command == "jobs":
            return _handle_jobs(config, args)
        if command == "preflight":
            from .mimic.preflight import render_report, run_preflight

            report = run_preflight(
                config,
                stages=args.stage,
                cohort_rows=args.cohort_rows,
            )
            _print(render_report(report))
            return 0 if all(item.ok for item in report.findings) else 1
        if command == "run-log":
            from .mimic.run_log import (
                RUN_LOG_FILENAME,
                read_records,
                render_summary,
                summarize,
            )

            path = config.runs_root / str(args.run_id) / RUN_LOG_FILENAME
            if not path.is_file():
                _print(
                    f"Kein Laufprotokoll unter {path}. "
                    "Laeufe vor Version 1.0.0 des Protokolls haben keines."
                )
                return 1
            records = read_records(path)
            if args.errors:
                failures = [
                    item for item in records if item.get("kind") == "error"
                ]
                if not failures:
                    _print("Keine Fehler protokolliert.")
                    return 0
                for item in failures:
                    _print(
                        f"--- {item.get('at', '')}  Stufe "
                        f"{item.get('stage', '?')}  {item.get('label', '')}"
                    )
                    _print(str(item.get("traceback") or item.get("error", "")))
                return 0
            summary = summarize(records)
            if args.json:
                _print(json.dumps(summary, indent=2, sort_keys=True, default=str))
            else:
                _print(render_summary(summary))
            return 0
        if command == "progress":
            import time as _time

            from .mimic.live_progress import read_live_state, render_live_board

            state_path = (
                config.runs_root / str(args.run_id) / "live_progress.json"
            )
            interval = args.watch
            # Redraw in place so the board stays readable. Appending a new
            # block every refresh scrolls the previous one away, which makes a
            # long run impossible to follow.
            in_place = (
                interval is not None
                and not args.no_clear
                and sys.stdout.isatty()
            )
            drawn = 0
            try:
                while True:
                    # Measure every pass: the window can be resized while the
                    # board is running. A board wider than the terminal wraps,
                    # and then the cursor is moved up by the logical line count
                    # while the screen advanced by the physical one -- the top
                    # of the previous board stays behind and reads as a
                    # duplicated header.
                    columns = int(args.width) or shutil.get_terminal_size(
                        (100, 30)
                    ).columns
                    board = render_live_board(
                        read_live_state(state_path),
                        run_id=str(args.run_id),
                        state_path=state_path,
                        width=max(60, columns - 1),
                    )
                    if in_place and drawn:
                        # Move up over the previous board and clear downward.
                        sys.stdout.write(f"{chr(27)}[{drawn}A{chr(27)}[J")
                    sys.stdout.write(board + chr(10))
                    sys.stdout.flush()
                    drawn = board.count(chr(10)) + 1
                    if interval is None:
                        return 0
                    _time.sleep(max(1.0, float(interval)))
            except KeyboardInterrupt:
                sys.stdout.write(chr(10))
                return 0
        if command == "deep-combinations":
            from .mimic.combination_stage import (
                STOP_FILE_NAME,
                run_deep_combination_search,
            )

            run_root = (
                config.runs_root / str(args.run_id) / "deep_combinations"
            )
            if args.status:
                from .mimic.combination_stage import (
                    describe_live_state,
                    read_live_state,
                )

                print(describe_live_state(read_live_state(run_root)))
                return 0
            if args.stop:
                from .mimic.combination_stage import request_stop

                request_stop(run_root, mode=args.stop_mode, requested_by="cli")
                explanation = {
                    "after_sweep": (
                        "Every remaining endpoint is still searched, but only "
                        "to the shallowest depth already reached, so none is "
                        "left without combinations."
                    ),
                    "after_endpoint": (
                        "The endpoint being searched is finished; later ones "
                        "are not started."
                    ),
                    "immediate": (
                        "The search ends after the current depth level. The "
                        "sweep stays partial: later endpoints have no "
                        "combinations at all."
                    ),
                }[args.stop_mode]
                print(f"Stop requested ({args.stop_mode}). {explanation}")
                print("Every depth level already written stays valid.")
                return 0
            # A stop file left over from a previous run would end this one
            # before it starts.
            stale_stop = run_root / STOP_FILE_NAME
            if stale_stop.exists():
                stale_stop.unlink()
            manifest = run_deep_combination_search(
                config,
                run_id=str(args.run_id),
                endpoint_ids=(
                    tuple(_csv_tuple(args.endpoint_ids))
                    if args.endpoint_ids
                    else None
                ),
                panel_ids=tuple(_csv_tuple(args.panel_ids)),
                feature_glob=args.feature_glob,
                max_depth=args.max_depth,
                patience=args.patience,
                minimum_increment=args.minimum_increment,
                beam_width=args.beam_width,
                seconds_per_endpoint=args.seconds_per_endpoint,
                run_until_stopped=bool(args.until_stopped),
                force=bool(args.force),
                progress=_progress,
            )
            print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
            return 0
        if command == "benchmark":
            benchmark = benchmark_resources(
                config,
                candidates=_int_csv_tuple(args.candidates),
                tasks_per_worker=args.tasks_per_worker,
                progress=_progress,
            )
            if args.apply_recommendation:
                config.workers = benchmark.recommended_workers
                config.resource_mode = "auto"
                config.save()
            _print(
                json.dumps(
                    benchmark.to_dict(),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if command == "adopt-features":
            from .mimic.feature_adoption import (
                adopt_stored_partitions,
                verify_stored_partitions,
            )

            options = FeatureExtractionOptions(
                panel_ids=tuple(args.panels),
                max_records=args.max_records,
            )
            runner = (
                verify_stored_partitions
                if args.check_only
                else adopt_stored_partitions
            )
            report = runner(
                config,
                options,
                sample_size=args.sample_size,
                seed=args.seed,
                progress=_print,
            )
            _print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if report.get("equivalent") else 2
        if command == "audit-biomarkers":
            output_root = (
                args.output
                if args.output is not None
                else config.work_root / "audits" / "biomarkers"
            )
            result = run_biomarker_audit(
                output_root,
                dynamic=not args.static_only,
            )
            _print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
            if result.default_panel_blockers or result.calculator_errors:
                return 2
            return 0
        if command == "validate-delineation":
            output_path = (
                args.output
                if args.output is not None
                else config.resources_root
                / "delineation"
                / "benchmark_suite.json"
            )
            dataset_root = (
                args.ludb_root
                if args.ludb_root is not None
                else args.benchmark_database_root
                if args.benchmark_database_root is not None
                else config.delineation_benchmark_root
            )
            dataset_ids = (
                ("ludb",)
                if args.ludb_root is not None
                else _csv_tuple(args.datasets)
            )
            inventory = inventory_delineation_benchmarks(dataset_root)
            result = run_delineation_benchmark_suite(
                dataset_root,
                output_path,
                dataset_ids=dataset_ids,
                max_records=args.max_records,
                max_seconds_per_record=args.max_seconds_per_record,
                strict=args.strict,
                progress=_progress,
            )
            _print(
                json.dumps(
                    {
                        "output_path": str(Path(output_path).resolve()),
                        "status": result["status"],
                        "datasets_completed": result["datasets_completed"],
                        "datasets_missing": result["datasets_missing"],
                        "inventory": inventory.to_dict(orient="records"),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if command == "snapshot":
            snapshot = create_artifact_snapshot(
                config,
                artifact_kind=args.artifact_kind,
                source_glob=args.source_glob,
                snapshot_id=args.snapshot_id,
                expected_files=args.expected_files,
                require_complete=args.require_complete,
                notes=_optional_csv_tuple(args.notes) or (),
            )
            _print(json.dumps(snapshot.to_dict(), indent=2, sort_keys=True))
            return 0
        if command == "discover-phenome":
            _show_manifest(
                run_phenome_discovery(
                    config,
                    PhenomeDiscoveryOptions(
                        run_id=args.run_id,
                        feature_glob=args.feature_glob,
                        panel_ids=_csv_tuple(args.panels),
                        train_fdr_alpha=args.train_fdr_alpha,
                        validation_fdr_alpha=args.validation_fdr_alpha,
                        min_class_inference=args.min_class_inference,
                        min_class_formal=args.min_class_formal,
                        bootstrap_iterations=args.bootstrap_iterations,
                        max_phenotypes=args.max_phenotypes,
                        max_future_phenotypes=(
                            args.max_future_phenotypes
                        ),
                        future_horizons_days=_int_csv_tuple(
                            args.future_horizons
                        ),
                        force=args.force,
                    ),
                    progress=_progress,
                )
            )
            return 0
        if command == "refresh-phenome-promotions":
            _show_manifest(
                refresh_phenome_promotions(
                    config,
                    args.run_id,
                    force=args.force,
                    progress=_progress,
                )
            )
            return 0
        if command == "ecg":
            return _handle_ecg(config, args)
        if command == "inventory":
            inventory = build_inventory(
                config,
                count_rows=args.count_rows,
                progress=_progress,
            )
            _print(
                f"Inventory complete: {len(inventory.items)} tables, "
                f"{len(inventory.errors)} errors, "
                f"{len(inventory.warnings)} warnings"
            )
            if inventory.errors:
                for item in inventory.errors:
                    _print(f"ERROR {item.table_id}: {item.diagnostic}")
                return 2
            return 0
        if command == "ingest":
            _show_manifest(
                ingest_standardized_tables(
                    config,
                    force=args.force,
                    progress=_progress,
                )
            )
            return 0
        if command == "ingest-events":
            _show_manifest(
                ingest_clinical_event_tables(
                    config,
                    force=args.force,
                    progress=_progress,
                )
            )
            return 0
        if command == "build-index":
            _show_manifest(
                build_ecg_master_index(
                    config,
                    force=args.force,
                    progress=_progress,
                )
            )
            return 0
        if command == "build-evidence":
            _show_manifest(
                build_clinical_evidence(
                    config,
                    force=args.force,
                    progress=_progress,
                )
            )
            return 0
        if command == "attest-event-evidence":
            # --force runs the full shadow rebuild (release check, E9).
            attestation = confirm_or_attest_clinical_event_evidence(
                config,
                full_equivalence=args.force,
                progress=_progress,
            )
            _print(json.dumps(attestation, indent=2, sort_keys=True))
            return 0
        if command == "build-labels":
            _show_manifest(
                build_clinical_labels(
                    config,
                    force=args.force,
                    progress=_progress,
                )
            )
            return 0
        if command == "attest-labels":
            attestation = attest_development_label_artifacts(
                config,
                force=args.force,
                progress=_progress,
            )
            _print(json.dumps(attestation, indent=2, sort_keys=True))
            return 0
        if command == "parse-reports":
            _show_manifest(
                build_machine_report_labels(
                    config,
                    force=args.force,
                    progress=_progress,
                )
            )
            return 0
        if command == "parse-notes":
            _show_manifest(
                build_note_labels(
                    config,
                    force=args.force,
                    progress=_progress,
                )
            )
            return 0
        if command == "extract-features":
            _show_manifest(
                extract_mimic_features(
                    config,
                    FeatureExtractionOptions(
                        panel_ids=_csv_tuple(args.panels),
                        include_locked_test=args.include_locked_test,
                        locked_test_only=bool(
                            getattr(
                                args,
                                "locked_test_for_frozen_run",
                                None,
                            )
                        ),
                        freeze_run_id=getattr(
                            args,
                            "locked_test_for_frozen_run",
                            None,
                        ),
                        max_records=args.max_records,
                        partition_size=args.partition_size,
                        micro_batch_size=getattr(
                            args,
                            "micro_batch_size",
                            config.feature_micro_batch_size,
                        ),
                        force=args.force,
                    ),
                    progress=_progress,
                )
            )
            return 0
        if command == "validate":
            _show_manifest(
                run_development_validation(
                    config,
                    DevelopmentValidationOptions(
                        run_id=args.run_id,
                        endpoint_ids=_optional_csv_tuple(args.endpoints),
                        panel_ids=_csv_tuple(args.panels),
                        feature_glob=args.feature_glob,
                        bootstrap_iterations=args.bootstrap_iterations,
                        include_predefined_scores=not args.no_predefined_scores,
                        include_ptb_frozen_scores=(
                            not args.no_ptb_frozen_scores
                        ),
                        ptb_transfer_run_id=args.ptb_transfer_run_id,
                        include_exploratory_scores=(
                            not args.no_exploratory_scores
                        ),
                        include_all_ecg_secondary=(
                            not args.no_all_ecg_secondary
                        ),
                        grouped_score_folds=args.grouped_score_folds,
                        grouped_score_repeats=args.grouped_score_repeats,
                        grouped_score_representatives_per_family=(
                            args.grouped_score_representatives_per_family
                        ),
                        grouped_score_selected_pairs=(
                            args.grouped_score_selected_pairs
                        ),
                        grouped_score_selected_triples=(
                            args.grouped_score_selected_triples
                        ),
                        grouped_score_triple_seed_pairs=(
                            args.grouped_score_triple_seed_pairs
                        ),
                        maximum_component_abs_correlation=(
                            args.maximum_component_abs_correlation
                        ),
                        minimum_score_incremental_auc=(
                            args.minimum_score_incremental_auc
                        ),
                        maximum_criterion_single_candidates=(
                            args.maximum_criterion_single_candidates
                        ),
                        maximum_criterion_combination_candidates=(
                            args.maximum_criterion_combination_candidates
                        ),
                        preview=args.preview,
                        force=args.force,
                    ),
                    progress=_progress,
                )
            )
            return 0
        if command == "group-vcg":
            _show_manifest(
                run_group_vcg_analysis(
                    config,
                    GroupVCGOptions(
                        run_id=args.run_id,
                        endpoint_id=args.endpoint_id,
                        split=args.split,
                        panel_ids=_csv_tuple(args.panels),
                        feature_glob=args.feature_glob,
                        max_per_class=args.max_per_class,
                        minimum_per_class=args.minimum_per_class,
                        candidate_multiplier=args.candidate_multiplier,
                        quality_minimum=args.quality_minimum,
                        bootstrap_iterations=args.bootstrap_iterations,
                        force=args.force,
                    ),
                    progress=_progress,
                )
            )
            return 0
        if command == "ptb-revalidate":
            _show_manifest(
                run_ptb_revalidation(
                    config,
                    ptb_root=args.ptb_root,
                    feature_cache=args.feature_cache,
                    options=PTBRevalidationOptions(
                        run_id=args.run_id,
                        bootstrap_iterations=args.bootstrap_iterations,
                        random_seed=config.random_seed,
                        force=args.force,
                    ),
                    progress=_progress,
                )
            )
            return 0
        if command == "ptb-freeze":
            lock = freeze_ptb_revalidation(
                config,
                run_id=args.run_id,
            )
            _print(
                f"PTB run {args.run_id!r} frozen. "
                f"Freeze hash: {lock.freeze_hash}"
            )
            return 0
        if command == "ptb-evaluate-test":
            _show_manifest(
                evaluate_ptb_locked_test(
                    config,
                    run_id=args.run_id,
                    bootstrap_iterations=args.bootstrap_iterations,
                    progress=_progress,
                )
            )
            return 0
        if command == "freeze":
            lock = freeze_validation_run(config, args.run_id)
            _print(
                f"Run {args.run_id!r} frozen. Freeze hash: {lock.freeze_hash}"
            )
            return 0
        if command == "evaluate-test":
            _show_manifest(
                evaluate_locked_test(
                    config,
                    LockedEvaluationOptions(
                        run_id=args.run_id,
                        feature_glob=args.feature_glob,
                    ),
                    progress=_progress,
                )
            )
            return 0
        if command == "invalidate-lock":
            lock = invalidate_test_lock(
                config,
                args.run_id,
                args.reason,
            )
            _print(
                f"Run {args.run_id!r} invalidated at {lock.invalidated_at}: "
                f"{lock.invalidation_reason}"
            )
            return 0
        if command == "report":
            if args.report_action == "build":
                _show_manifest(
                    build_configured_report(
                        config,
                        _report_spec_from_args(args),
                        force=args.force,
                        progress=_progress,
                    )
                )
                return 0
            _show_manifest(
                generate_run_reports(
                    config,
                    args.run_id,
                    feature_glob=args.feature_glob,
                    force=args.force,
                    progress=_progress,
                )
            )
            return 0
        if command == "audit":
            payload = run_development_acceptance_audit(
                config,
                run_id=args.run_id,
                ptb_run_id=args.ptb_run_id,
                expectations=AcceptanceExpectations(
                    ecg_count=args.expected_ecg_count,
                    subject_count=args.expected_subject_count,
                    echo_study_count=args.expected_echo_study_count,
                    development_feature_rows=(
                        args.expected_development_feature_rows
                    ),
                    label_definition_count=(
                        args.expected_label_definition_count
                    ),
                    minimum_bootstrap_iterations=(
                        args.minimum_bootstrap_iterations
                    ),
                    group_vcg_endpoints=(
                        _optional_csv_tuple(args.group_vcg_endpoints) or ()
                    ),
                ),
                require_freeze=args.require_freeze,
                strict=args.strict,
                deep_feature_verification=(
                    not args.skip_deep_feature_verification
                ),
                progress=_progress,
            )
            _print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if command == "run-all":
            return _run_all(config, args)
        parser.error(f"Unsupported command: {command}")
    except KeyboardInterrupt:
        _print("Cancelled by user. Completed atomic partitions remain resumable.")
        return 130
    except Exception as exc:
        _print(f"FAILED: {type(exc).__name__}: {exc}")
        if getattr(args, "debug", False):
            traceback.print_exc()
        return 1
    return 0


def _run_all(config: MimicProjectConfig, args: argparse.Namespace) -> int:
    inventory = build_inventory(
        config,
        count_rows=args.count_rows,
        progress=_progress,
    )
    if inventory.errors:
        for item in inventory.errors:
            _print(f"ERROR {item.table_id}: {item.diagnostic}")
        raise RuntimeError("Inventory has blocking schema/path errors")
    ingest_standardized_tables(
        config,
        force=args.force,
        progress=_progress,
    )
    ingest_clinical_event_tables(
        config,
        force=args.force,
        progress=_progress,
    )
    build_ecg_master_index(
        config,
        force=args.force,
        progress=_progress,
    )
    build_machine_report_labels(
        config,
        force=args.force,
        progress=_progress,
    )
    build_note_labels(
        config,
        force=args.force,
        progress=_progress,
    )
    build_clinical_evidence(
        config,
        force=args.force,
        progress=_progress,
    )
    build_clinical_labels(
        config,
        force=args.force,
        progress=_progress,
    )
    attest_development_label_artifacts(
        config,
        force=args.force,
        progress=_progress,
    )
    if args.ptb_root and args.ptb_feature_cache:
        run_ptb_revalidation(
            config,
            ptb_root=args.ptb_root,
            feature_cache=args.ptb_feature_cache,
            options=PTBRevalidationOptions(
                run_id=args.ptb_transfer_run_id,
                bootstrap_iterations=(
                    args.bootstrap_iterations
                    if args.bootstrap_iterations is not None
                    else 1000
                ),
                random_seed=config.random_seed,
                force=args.force,
            ),
            progress=_progress,
        )
        freeze_ptb_revalidation(
            config,
            run_id=args.ptb_transfer_run_id,
        )
    panels = _csv_tuple(args.panels)
    # run-all is deliberately development-only. Any test-waveform access
    # requires a separate, explicit release action after the freeze.
    include_locked = False
    extract_mimic_features(
        config,
        FeatureExtractionOptions(
            panel_ids=panels,
            include_locked_test=include_locked,
            max_records=args.max_records,
            partition_size=args.partition_size,
            micro_batch_size=getattr(
                args,
                "micro_batch_size",
                config.feature_micro_batch_size,
            ),
            force=args.force,
        ),
        progress=_progress,
    )
    if args.max_records is not None:
        scope = (
            f"smoke-{'all' if include_locked else 'development'}"
            f"-{args.max_records}"
        )
    else:
        scope = "all" if include_locked else "development"
    validation_feature_glob = str(
        config.features_root
        / "_".join(panels)
        / scope
        / "split=*"
        / "part-*.parquet"
    )
    validation_defaults = DevelopmentValidationOptions()
    run_development_validation(
        config,
        DevelopmentValidationOptions(
            run_id=args.run_id,
            endpoint_ids=_optional_csv_tuple(args.endpoints),
            panel_ids=panels,
            feature_glob=validation_feature_glob,
            bootstrap_iterations=args.bootstrap_iterations,
            include_predefined_scores=not args.no_predefined_scores,
            include_ptb_frozen_scores=not args.no_ptb_frozen_scores,
            ptb_transfer_run_id=args.ptb_transfer_run_id,
            include_exploratory_scores=not args.no_exploratory_scores,
            include_all_ecg_secondary=not args.no_all_ecg_secondary,
            grouped_score_folds=getattr(
                args,
                "grouped_score_folds",
                validation_defaults.grouped_score_folds,
            ),
            grouped_score_repeats=getattr(
                args,
                "grouped_score_repeats",
                validation_defaults.grouped_score_repeats,
            ),
            grouped_score_representatives_per_family=(
                getattr(
                    args,
                    "grouped_score_representatives_per_family",
                    validation_defaults.grouped_score_representatives_per_family,
                )
            ),
            grouped_score_selected_pairs=getattr(
                args,
                "grouped_score_selected_pairs",
                validation_defaults.grouped_score_selected_pairs,
            ),
            grouped_score_selected_triples=getattr(
                args,
                "grouped_score_selected_triples",
                validation_defaults.grouped_score_selected_triples,
            ),
            grouped_score_triple_seed_pairs=(
                getattr(
                    args,
                    "grouped_score_triple_seed_pairs",
                    validation_defaults.grouped_score_triple_seed_pairs,
                )
            ),
            maximum_component_abs_correlation=(
                getattr(
                    args,
                    "maximum_component_abs_correlation",
                    validation_defaults.maximum_component_abs_correlation,
                )
            ),
            minimum_score_incremental_auc=getattr(
                args,
                "minimum_score_incremental_auc",
                validation_defaults.minimum_score_incremental_auc,
            ),
            maximum_criterion_single_candidates=(
                getattr(
                    args,
                    "maximum_criterion_single_candidates",
                    validation_defaults.maximum_criterion_single_candidates,
                )
            ),
            maximum_criterion_combination_candidates=(
                getattr(
                    args,
                    "maximum_criterion_combination_candidates",
                    validation_defaults.maximum_criterion_combination_candidates,
                )
            ),
            preview=args.max_records is not None,
            force=args.force,
        ),
        progress=_progress,
    )
    if args.max_records is None:
        run_phenome_discovery(
            config,
            PhenomeDiscoveryOptions(
                run_id=args.run_id,
                feature_glob=validation_feature_glob,
                panel_ids=panels,
                bootstrap_iterations=(
                    args.bootstrap_iterations
                    if args.bootstrap_iterations is not None
                    else 1000
                ),
                force=args.force,
            ),
            progress=_progress,
        )
    generate_run_reports(
        config,
        args.run_id,
        feature_glob=validation_feature_glob,
        force=args.force,
        progress=_progress,
    )
    _print(
        "Development pipeline complete. Locked labels and waveforms remain sealed. "
        "Run freeze explicitly after scientific review; evaluate-test is never "
        "called by run-all."
    )
    return 0


def _load_config(args: argparse.Namespace) -> MimicProjectConfig:
    project = getattr(args, "project", None)
    database_root = getattr(args, "database_root", None)
    work_root = getattr(args, "work_root", None)
    persist_config = False
    if project:
        config = load_or_discover_config(project)
        project_path = Path(project)
        expected_file = (
            project_path / "project.json"
            if project_path.is_dir()
            else project_path
        )
        persist_config = not expected_file.is_file()
    elif database_root or work_root:
        config = MimicProjectConfig.discover(
            database_root or r"E:\Database",
            work_root or r"E:\VCG\mimic_validation",
        )
        persist_config = True
    else:
        config = load_or_discover_config()
    if getattr(args, "workers", None) is not None:
        config.workers = int(args.workers)
        persist_config = True
    if getattr(args, "memory_gb", None) is not None:
        config.duckdb_memory_limit_gb = int(args.memory_gb)
        persist_config = True
    config.ensure_directories()
    if persist_config or not config.project_file.is_file():
        config.save()
    return config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m vcg_biomarker.mimic_cli",
        description=(
            "Research-only VCG Feature Scanner workflow for clinical MIMIC "
            "reference validation."
        ),
    )
    parser.add_argument("--project", help="Project JSON file or work directory")
    parser.add_argument("--database-root", type=Path)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--memory-gb", type=int)
    parser.add_argument("--debug", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    clinical_utility = subparsers.add_parser(
        "assess-clinical-utility",
        help=(
            "Assess a v2 clinical-utility dossier from verified aggregate "
            "evidence metadata; referenced artifacts and locked data are "
            "never opened"
        ),
    )
    clinical_utility.add_argument(
        "--dossier",
        type=Path,
        required=True,
        help="ClinicalUtilityDossier v2 JSON metadata",
    )
    clinical_utility.add_argument(
        "--evidence-registry",
        type=Path,
        required=True,
        help="Tamper-evident EvidenceRegistry v2 JSON metadata",
    )
    clinical_utility.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Named output directory published atomically",
    )

    jobs = subparsers.add_parser(
        "jobs",
        help="Persistent background job control",
    )
    job_commands = jobs.add_subparsers(dest="jobs_command", required=True)
    jobs_list = job_commands.add_parser("list")
    jobs_list.add_argument(
        "--states",
        help="Comma-separated job states",
    )
    jobs_list.add_argument("--limit", type=int, default=100)
    jobs_list.add_argument("--json", action="store_true")
    jobs_start = job_commands.add_parser("start")
    jobs_start.add_argument(
        "stage",
        choices=(
            "inventory",
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
            "physiological-gap-markers",
            "validate",
            "group-vcg",
            "freeze",
            "discover-phenome",
            "refresh-phenome-promotions",
            "report",
            "report-build",
            "benchmark",
            "validate-delineation",
            "snapshot",
            "audit",
            "audit-biomarkers",
            "assess-clinical-utility",
        ),
    )
    jobs_start.add_argument(
        "--arguments-json",
        default="{}",
        help="JSON object passed to the shared stage service",
    )
    jobs_start.add_argument("--priority", type=int, default=50)
    jobs_start.add_argument("--job-id")
    jobs_start.add_argument("--no-supervisor", action="store_true")
    for action in ("pause", "resume", "cancel", "retry", "events"):
        command = job_commands.add_parser(action)
        command.add_argument("job_id")
    jobs_delete = job_commands.add_parser(
        "delete",
        help="Permanently delete a paused or terminal job and its event log",
    )
    jobs_delete.add_argument("job_id")
    jobs_delete.add_argument(
        "--confirm-job-id",
        required=True,
        help="Must exactly repeat JOB_ID",
    )
    jobs_delete.add_argument(
        "--delete-run-manifest-root",
        help=(
            "Optional exact study-run directory to remove. It must contain "
            "only this job's guided_workflow.manifest.json."
        ),
    )
    jobs_delete.add_argument(
        "--confirm-delete-run-manifest-root",
        action="store_true",
        help="Explicitly confirm removal of the validated one-file run root",
    )
    archive_pilot = job_commands.add_parser(
        "archive-pilot",
        help=(
            "Preserve a paused partial development extraction as a historical "
            "pilot and close the job"
        ),
    )
    archive_pilot.add_argument("job_id")
    archive_pilot.add_argument(
        "--reason",
        default=(
            "Outdated implementation and artifact hashes; retained only as a "
            "partial hypothesis-generating pilot before a clean rerun."
        ),
    )

    preflight_cmd = subparsers.add_parser(
        "preflight",
        help=(
            "Prueft die noch ausstehenden Stufen gegen die vorhandenen "
            "Artefakte: fehlende Spalten, breite Materialisierungen."
        ),
    )
    preflight_cmd.add_argument(
        "--stage",
        action="append",
        default=None,
        help="Nur diese Stufe pruefen. Mehrfach angebbar.",
    )
    preflight_cmd.add_argument(
        "--cohort-rows",
        type=int,
        default=None,
        help=(
            "Zeilenzahl der Entwicklungskohorte, fuer die Abschaetzung des "
            "Speicherbedarfs breiter Materialisierungen."
        ),
    )

    run_log_cmd = subparsers.add_parser(
        "run-log",
        help=(
            "Fasst run.jsonl eines Laufs zusammen: Stufen mit Dauer, "
            "Speicherspitze und Wiederverwendungsgrund, langsamste Abfragen, "
            "Fehler."
        ),
    )
    run_log_cmd.add_argument("--run-id", required=True)
    run_log_cmd.add_argument(
        "--json",
        action="store_true",
        help="Die Zusammenfassung als JSON ausgeben statt als Tabelle.",
    )
    run_log_cmd.add_argument(
        "--errors",
        action="store_true",
        help=(
            "Nur die Fehler ausgeben, mit vollstaendigem Traceback. Fuer den "
            "Fall, dass ein Lauf abgebrochen ist."
        ),
    )

    live = subparsers.add_parser(
        "progress",
        help="Show the live status of a run: current stage and overall percent.",
    )
    live.add_argument("--run-id", required=True)
    live.add_argument(
        "--watch",
        nargs="?",
        type=float,
        const=10.0,
        default=None,
        help=(
            "Laufend aktualisieren statt einmal ausgeben. Ohne Zahl alle 10 "
            "Sekunden; mit Zahl in diesem Abstand."
        ),
    )
    live.add_argument(
        "--width",
        type=int,
        default=0,
        help=(
            "Breite der Anzeige in Zeichen. Standard: die Breite des "
            "Terminals."
        ),
    )
    live.add_argument(
        "--no-clear",
        action="store_true",
        help=(
            "Jede Aktualisierung als neuen Block anhaengen, statt die "
            "bestehenden Zeilen zu ersetzen. Fuer Log-Dateien."
        ),
    )

    deep = subparsers.add_parser(
        "deep-combinations",
        help=(
            "Grow marker combinations adaptively until extra depth stops "
            "buying held-out discrimination."
        ),
    )
    deep.add_argument("--run-id", default="mimic-development-guided")
    deep.add_argument("--endpoint-ids", default=None)
    deep.add_argument("--panel-ids", default="primary_core,sentinel")
    deep.add_argument("--feature-glob", default=None)
    deep.add_argument(
        "--max-depth",
        type=int,
        default=12,
        help="Hard depth ceiling. Ignored with --until-stopped.",
    )
    deep.add_argument(
        "--patience",
        type=int,
        default=2,
        help="Stop after this many depth levels without a held-out gain.",
    )
    deep.add_argument("--minimum-increment", type=float, default=0.002)
    deep.add_argument("--beam-width", type=int, default=8)
    deep.add_argument("--seconds-per-endpoint", type=float, default=None)
    deep.add_argument(
        "--until-stopped",
        action="store_true",
        help=(
            "Remove the depth ceiling and the automatic stopping rule; run "
            "until the stop file appears. Results are written after every "
            "depth level, so stopping is always safe."
        ),
    )
    deep.add_argument(
        "--stop",
        action="store_true",
        help="Ask a running deep search to stop.",
    )
    deep.add_argument(
        "--stop-mode",
        choices=["after_sweep", "after_endpoint", "immediate"],
        default="after_sweep",
        help=(
            "after_sweep (default): finish every remaining endpoint at the "
            "shallowest depth already reached, so none is left without any "
            "combination. after_endpoint: finish the endpoint being searched. "
            "immediate: stop after the current depth level, leaving the sweep "
            "partial."
        ),
    )
    deep.add_argument(
        "--status",
        action="store_true",
        help="Print the live depth and the estimate for the next level.",
    )
    deep.add_argument("--force", action="store_true")

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--candidates", default="1,2,4,6,8,10,12,14")
    benchmark.add_argument("--tasks-per-worker", type=int, default=3)
    benchmark.add_argument("--apply-recommendation", action="store_true")

    delineation_benchmark = subparsers.add_parser(
        "validate-delineation",
        help=(
            "Run task-appropriate LUDB, QTDB, Noise Stress, AFDB, INCART "
            "and STAFF III delineation audits"
        ),
    )
    delineation_benchmark.add_argument(
        "--ludb-root",
        type=Path,
        help="Legacy shortcut: run only LUDB from this local root",
    )
    delineation_benchmark.add_argument(
        "--benchmark-database-root",
        type=Path,
        help="Root containing the local open PhysioNet benchmark datasets",
    )
    delineation_benchmark.add_argument(
        "--datasets",
        default="ludb,qtdb,nstdb,afdb,incart,staffiii",
        help="Comma-separated benchmark dataset IDs",
    )
    delineation_benchmark.add_argument("--output", type=Path)
    delineation_benchmark.add_argument("--max-records", type=int)
    delineation_benchmark.add_argument(
        "--max-seconds-per-record", type=int, default=120
    )
    delineation_benchmark.add_argument("--strict", action="store_true")

    adopt_features = subparsers.add_parser(
        "adopt-features",
        help=(
            "Prove that an existing extraction still matches the current "
            "marker code by recomputing a sample, and adopt it if it does"
        ),
    )
    adopt_features.add_argument(
        "--panels",
        nargs="+",
        default=list(STUDY_PANEL_IDS),
    )
    adopt_features.add_argument("--max-records", type=int)
    adopt_features.add_argument(
        "--sample-size",
        type=int,
        default=250,
        help="Recordings to recompute and compare, marker for marker",
    )
    adopt_features.add_argument(
        "--seed",
        default="feature-adoption-v1",
        help="Names the sample; the same seed selects the same recordings",
    )
    adopt_features.add_argument(
        "--check-only",
        action="store_true",
        help="Compare and report, but write no attestation",
    )

    biomarker_audit = subparsers.add_parser(
        "audit-biomarkers",
        help=(
            "Audit every biomarker contract without reading MIMIC locked_test "
            "or PTB fold 10"
        ),
    )
    biomarker_audit.add_argument("--output", type=Path)
    biomarker_audit.add_argument(
        "--static-only",
        action="store_true",
        help="Skip deterministic synthetic signal checks",
    )

    snapshot = subparsers.add_parser("snapshot")
    snapshot.add_argument("--artifact-kind", default="features")
    snapshot.add_argument("--source-glob", required=True)
    snapshot.add_argument("--snapshot-id")
    snapshot.add_argument("--expected-files", type=int)
    snapshot.add_argument("--require-complete", action="store_true")
    snapshot.add_argument("--notes")

    phenome = subparsers.add_parser("discover-phenome")
    phenome.add_argument("--run-id", default="phenome-discovery")
    phenome.add_argument("--feature-glob")
    phenome.add_argument("--panels", default="primary_core,sentinel")
    phenome.add_argument("--train-fdr-alpha", type=float, default=0.01)
    phenome.add_argument("--validation-fdr-alpha", type=float, default=0.05)
    phenome.add_argument("--min-class-inference", type=int, default=30)
    phenome.add_argument("--min-class-formal", type=int, default=100)
    phenome.add_argument("--bootstrap-iterations", type=int, default=1000)
    phenome.add_argument("--max-phenotypes", type=int)
    phenome.add_argument(
        "--max-future-phenotypes",
        type=int,
        default=100,
        help=(
            "Power-ranked prospective phenotypes (default 100); "
            "0 scans every eligible phenotype"
        ),
    )
    phenome.add_argument(
        "--future-horizons",
        default="30,90,365",
        help=(
            "Comma-separated censored follow-up horizons in days; "
            "locked_test is never scanned"
        ),
    )
    phenome.add_argument("--force", action="store_true")

    promotion_refresh = subparsers.add_parser(
        "refresh-phenome-promotions",
        help=(
            "Rebuild draft PheWAS promotion candidates from immutable "
            "replicated development hits under the current feature-role policy"
        ),
    )
    promotion_refresh.add_argument(
        "--run-id",
        default="mimic-development",
    )
    promotion_refresh.add_argument("--force", action="store_true")

    ecg = subparsers.add_parser("ecg")
    ecg_commands = ecg.add_subparsers(dest="ecg_command", required=True)
    ecg_import = ecg_commands.add_parser("import")
    ecg_import.add_argument("source", type=Path)
    ecg_import.add_argument("--sampling-rate", type=float)
    ecg_import.add_argument("--unit")
    ecg_import.add_argument("--age", type=int)
    ecg_import.add_argument("--sex")
    ecg_import.add_argument("--pseudonym")
    ecg_import.add_argument("--lead-mapping-json", default="{}")
    ecg_import.add_argument("--retain-source-path", action="store_true")
    ecg_import.add_argument("--copy-original", action="store_true")
    ecg_list = ecg_commands.add_parser("list")
    ecg_list.add_argument("--json", action="store_true")
    ecg_analyze = ecg_commands.add_parser("analyze")
    ecg_analyze.add_argument("case_id")
    ecg_analyze.add_argument("--panels", default="primary_core")
    ecg_analyze.add_argument(
        "--vcg-method",
        choices=("kors", "inverse_dower", "pca"),
        default="kors",
    )
    ecg_analyze.add_argument("--powerline-hz", type=float, default=60.0)
    ecg_annotate = ecg_commands.add_parser("annotate")
    ecg_annotate.add_argument("case_id")
    ecg_annotate.add_argument("--fiducials-json", required=True)
    ecg_annotate.add_argument("--notes", default="")
    ecg_annotate.add_argument("--parent-revision-id", default="")

    inventory = subparsers.add_parser("inventory")
    inventory.add_argument("--count-rows", action="store_true")

    for name in (
        "ingest",
        "ingest-events",
        "build-index",
        "build-evidence",
        "attest-event-evidence",
        "build-labels",
        "attest-labels",
        "parse-reports",
        "parse-notes",
    ):
        command = subparsers.add_parser(name)
        command.add_argument("--force", action="store_true")

    features = subparsers.add_parser("extract-features")
    _add_feature_arguments(features)

    validate = subparsers.add_parser("validate")
    _add_validation_arguments(validate, include_panels=True)
    validate.add_argument("--force", action="store_true")

    group_vcg = subparsers.add_parser(
        "group-vcg",
        help=(
            "Build development-only positive/negative group VCG loops from "
            "an immutable feature snapshot"
        ),
    )
    group_vcg.add_argument("--run-id", default="mimic-development")
    group_vcg.add_argument("--endpoint-id", required=True)
    group_vcg.add_argument(
        "--split",
        choices=("train", "validation"),
        default="validation",
    )
    group_vcg.add_argument("--panels", default="primary_core,sentinel")
    group_vcg.add_argument("--feature-glob")
    group_vcg.add_argument("--max-per-class", type=int, default=40)
    group_vcg.add_argument("--minimum-per-class", type=int, default=10)
    group_vcg.add_argument("--candidate-multiplier", type=int, default=4)
    group_vcg.add_argument("--quality-minimum", type=float, default=0.75)
    group_vcg.add_argument("--bootstrap-iterations", type=int, default=500)
    group_vcg.add_argument("--force", action="store_true")

    ptb_revalidate = subparsers.add_parser("ptb-revalidate")
    ptb_revalidate.add_argument("--ptb-root", type=Path, required=True)
    ptb_revalidate.add_argument("--feature-cache", type=Path, required=True)
    ptb_revalidate.add_argument(
        "--run-id",
        default=DEFAULT_PTB_TRANSFER_RUN_ID,
    )
    ptb_revalidate.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=1000,
    )
    ptb_revalidate.add_argument("--force", action="store_true")

    ptb_freeze = subparsers.add_parser("ptb-freeze")
    ptb_freeze.add_argument(
        "--run-id",
        default=DEFAULT_PTB_TRANSFER_RUN_ID,
    )

    ptb_evaluate = subparsers.add_parser("ptb-evaluate-test")
    ptb_evaluate.add_argument(
        "--run-id",
        default=DEFAULT_PTB_TRANSFER_RUN_ID,
    )
    ptb_evaluate.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=2000,
    )

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--run-id", required=True)

    evaluate = subparsers.add_parser("evaluate-test")
    evaluate.add_argument("--run-id", required=True)
    evaluate.add_argument("--feature-glob")

    invalidate = subparsers.add_parser("invalidate-lock")
    invalidate.add_argument("--run-id", required=True)
    invalidate.add_argument("--reason", required=True)

    report = subparsers.add_parser("report")
    report.add_argument(
        "report_action",
        nargs="?",
        choices=("canonical", "build"),
        default="canonical",
    )
    report.add_argument("--run-id", required=True)
    report.add_argument("--feature-glob")
    report.add_argument("--report-id")
    report.add_argument(
        "--report-type",
        choices=tuple(item.value for item in ReportType),
        default=ReportType.STUDY.value,
    )
    report.add_argument("--language", choices=("de", "en"), default="de")
    report.add_argument(
        "--formats",
        default="pdf,html,xlsx,parquet",
    )
    report.add_argument("--endpoints")
    report.add_argument(
        "--report-panels",
        default="primary_core,sentinel",
    )
    report.add_argument("--sections")
    report.add_argument("--include-case-ids", action="store_true")
    report.add_argument("--include-case-appendix", action="store_true")
    report.add_argument("--force", action="store_true")

    audit = subparsers.add_parser(
        "audit",
        help=(
            "Verify development acceptance without reading MIMIC locked "
            "outcomes or PTB fold 10"
        ),
    )
    audit.add_argument("--run-id", required=True)
    audit.add_argument(
        "--ptb-run-id",
        default=DEFAULT_PTB_TRANSFER_RUN_ID,
    )
    audit.add_argument("--expected-ecg-count", type=int, default=800_035)
    audit.add_argument(
        "--expected-subject-count",
        type=int,
        default=161_352,
    )
    audit.add_argument(
        "--expected-echo-study-count",
        type=int,
        default=206_488,
    )
    audit.add_argument(
        "--expected-development-feature-rows",
        type=int,
        default=None,
        help="Default: selected ECG count from the extraction manifest",
    )
    audit.add_argument(
        "--expected-label-definition-count",
        type=int,
        default=63,
    )
    audit.add_argument(
        "--minimum-bootstrap-iterations",
        type=int,
        default=1_000,
    )
    audit.add_argument(
        "--group-vcg-endpoints",
        default=(
            "acute_mi_high_specificity,echo_lvh_strict,reduced_lvef,"
            "right_heart_strain_proxy,echo_lae,hyperkalemia_near_ecg"
        ),
    )
    audit.add_argument("--require-freeze", action="store_true")
    audit.add_argument("--strict", action="store_true")
    audit.add_argument(
        "--skip-deep-feature-verification",
        action="store_true",
        help="Use recorded integrity metadata for diagnostic previews only",
    )

    run_all = subparsers.add_parser("run-all")
    _add_feature_arguments(run_all)
    _add_validation_arguments(run_all)
    run_all.add_argument("--count-rows", action="store_true")
    run_all.add_argument(
        "--development-only",
        action="store_true",
        help=(
            "Deprecated compatibility flag. run-all is always development-only "
            "and never extracts locked-test waveforms."
        ),
    )
    run_all.add_argument("--ptb-root", type=Path)
    run_all.add_argument("--ptb-feature-cache", type=Path)
    return parser


def _add_feature_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--panels",
        default="primary_core,sentinel",
        help="Comma-separated panel IDs",
    )
    parser.add_argument(
        "--locked-test-for-frozen-run",
        metavar="RUN_ID",
        help=(
            "Explicit post-freeze action: extract only locked-test waveforms "
            "with the implementation and panel bound to RUN_ID."
        ),
    )
    parser.add_argument(
        "--include-locked-test",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--partition-size", type=int, default=250)
    parser.add_argument("--micro-batch-size", type=int, default=25)
    parser.add_argument("--force", action="store_true")


def _add_validation_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_panels: bool = False,
) -> None:
    if include_panels:
        parser.add_argument(
            "--panels",
            default="primary_core,sentinel",
            help="Comma-separated panel IDs matching the feature files",
        )
    parser.add_argument("--run-id", default="mimic-development")
    parser.add_argument("--endpoints", help="Comma-separated endpoint IDs")
    parser.add_argument("--feature-glob")
    parser.add_argument("--bootstrap-iterations", type=int)
    parser.add_argument("--grouped-score-folds", type=int, default=5)
    parser.add_argument("--grouped-score-repeats", type=int, default=1)
    parser.add_argument(
        "--grouped-score-representatives-per-family",
        type=int,
        default=2,
    )
    parser.add_argument("--grouped-score-selected-pairs", type=int, default=1)
    parser.add_argument("--grouped-score-selected-triples", type=int, default=1)
    parser.add_argument("--grouped-score-triple-seed-pairs", type=int, default=25)
    parser.add_argument(
        "--maximum-component-abs-correlation",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--minimum-score-incremental-auc",
        type=float,
        default=0.005,
    )
    parser.add_argument(
        "--maximum-criterion-single-candidates", type=int, default=6
    )
    parser.add_argument(
        "--maximum-criterion-combination-candidates", type=int, default=3
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help=(
            "Analyze an immutable smoke/partial snapshot descriptively. "
            "Preview results are clearly marked and cannot be frozen."
        ),
    )
    parser.add_argument("--no-predefined-scores", action="store_true")
    parser.add_argument("--no-ptb-frozen-scores", action="store_true")
    parser.add_argument(
        "--ptb-transfer-run-id",
        default=DEFAULT_PTB_TRANSFER_RUN_ID,
    )
    parser.add_argument("--no-exploratory-scores", action="store_true")
    parser.add_argument(
        "--no-all-ecg-secondary",
        action="store_true",
        help="Skip the secondary repeated-ECG patient-clustered analysis",
    )


def _handle_jobs(
    config: MimicProjectConfig,
    args: argparse.Namespace,
) -> int:
    store = JobStore.for_project(config)
    action = args.jobs_command
    if action == "list":
        states = _optional_csv_tuple(args.states)
        records = store.list(states=states, limit=args.limit)
        if args.json:
            print(
                json.dumps(
                    [_public_job_record(record) for record in records],
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
            )
            return 0
        if not records:
            _print("Keine Jobs vorhanden.")
            return 0
        _print(
            f"{'JOB-ID':40} {'STATUS':12} {'FORTSCHRITT':14} "
            f"{'PRIO':4} STUFE"
        )
        for record in records:
            progress = (
                f"{record.progress_current}/{record.progress_total}"
                if record.progress_total
                else "-"
            )
            _print(
                f"{record.spec.job_id[:40]:40} "
                f"{record.state.value:12} {progress:14} "
                f"{record.spec.priority:4d} {record.spec.stage}"
            )
        return 0
    if action == "start":
        arguments = _json_object(args.arguments_json, "--arguments-json")
        record = store.create(
            stage=args.stage,
            project_file=config.project_file,
            arguments=arguments,
            priority=args.priority,
            job_id=args.job_id,
        )
        supervisor_pid = None
        if not args.no_supervisor:
            supervisor_pid = ensure_supervisor(config)
        _print(
            f"Job {record.spec.job_id} queued; supervisor PID "
            f"{supervisor_pid or 'not started'}"
        )
        return 0
    if action == "archive-pilot":
        archived = archive_partial_feature_pilot(
            config,
            args.job_id,
            reason=args.reason,
        )
        _print(json.dumps(archived.to_dict(), indent=2, sort_keys=True))
        return 0
    if action == "delete":
        if args.confirm_job_id != args.job_id:
            raise ValueError("--confirm-job-id must exactly repeat the job id")
        record = store.get(args.job_id)
        run_root = None
        if args.delete_run_manifest_root:
            if not args.confirm_delete_run_manifest_root:
                raise ValueError(
                    "--delete-run-manifest-root additionally requires "
                    "--confirm-delete-run-manifest-root"
                )
            run_root = validate_job_run_manifest_root(
                record,
                config,
                args.delete_run_manifest_root,
            )
        elif args.confirm_delete_run_manifest_root:
            raise ValueError(
                "--confirm-delete-run-manifest-root requires "
                "--delete-run-manifest-root"
            )
        deleted = store.delete(args.job_id)
        removed_root = None
        if run_root is not None:
            removed_root = remove_confirmed_job_run_manifest_root(
                deleted,
                config,
                run_root,
            )
        _print(
            json.dumps(
                {
                    "deleted_job_id": deleted.spec.job_id,
                    "deleted_state": deleted.state.value,
                    "deleted_run_manifest_root": (
                        str(removed_root) if removed_root is not None else None
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if action == "pause":
        record = store.request_pause(args.job_id)
    elif action == "resume":
        record = store.request_resume(args.job_id)
        ensure_supervisor(config)
    elif action == "cancel":
        record = store.request_cancel(args.job_id)
    elif action == "retry":
        record = store.retry(args.job_id)
        ensure_supervisor(config)
    elif action == "events":
        print(
            json.dumps(
                store.events(args.job_id),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    else:
        raise ValueError(f"Unsupported jobs action: {action}")
    _print(f"{record.spec.job_id}: {record.state.value}")
    return 0


def _public_job_record(record) -> dict:
    payload = record.to_dict()
    spec = payload.get("spec") or {}
    arguments = dict(spec.get("arguments") or {})
    snapshot_hash = str(
        arguments.pop(JOB_PROJECT_CONFIG_HASH_KEY, "")
    )
    arguments.pop(JOB_PROJECT_CONFIG_KEY, None)
    spec["arguments"] = arguments
    spec["project_config_hash"] = snapshot_hash
    return payload


def _handle_ecg(
    config: MimicProjectConfig,
    args: argparse.Namespace,
) -> int:
    library = ECGCaseLibrary(config)
    action = args.ecg_command
    if action == "import":
        mapping = _json_object(args.lead_mapping_json, "--lead-mapping-json")
        case = library.import_case(
            args.source,
            metadata=ExternalECGMeta(
                age=args.age,
                sex=args.sex,
                sampling_rate=args.sampling_rate,
                lead_mapping={
                    str(key): str(value) for key, value in mapping.items()
                },
                default_unit=args.unit,
            ),
            pseudonym=args.pseudonym,
            retain_source_path=args.retain_source_path,
            copy_original=args.copy_original,
        )
        print(json.dumps(case.to_dict(), indent=2, sort_keys=True, default=str))
        return 0
    if action == "list":
        cases = library.list_cases()
        if args.json:
            print(
                json.dumps(
                    [case.to_dict() for case in cases],
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
            )
        else:
            for case in cases:
                _print(
                    f"{case.case_id} | {case.pseudonym} | "
                    f"{case.source_format} | {case.sampling_rate_hz:g} Hz"
                )
        return 0
    if action == "analyze":
        result = library.analyze_case(
            args.case_id,
            panel_ids=_csv_tuple(args.panels),
            vcg_method=args.vcg_method,
            powerline_hz=args.powerline_hz,
        )
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0
    if action == "annotate":
        fiducials = _json_object(args.fiducials_json, "--fiducials-json")
        revision = library.add_annotation(
            args.case_id,
            fiducials={
                str(key): int(value) for key, value in fiducials.items()
            },
            notes=args.notes,
            parent_revision_id=args.parent_revision_id,
        )
        print(
            json.dumps(
                revision.to_dict(),
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
        return 0
    raise ValueError(f"Unsupported ECG action: {action}")


def _report_spec_from_args(args: argparse.Namespace) -> ReportSpec:
    sections = _optional_csv_tuple(args.sections) or (
        "summary",
        "methods",
        "cohort",
        "performance",
        "baselines",
        "stability",
        "limitations",
        "provenance",
    )
    report_id = args.report_id or (
        f"{args.run_id}-{args.report_type}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    return ReportSpec(
        report_id=report_id,
        report_type=ReportType(args.report_type),
        run_id=args.run_id,
        language=args.language,
        formats=_csv_tuple(args.formats),
        endpoint_ids=_optional_csv_tuple(args.endpoints) or (),
        panel_ids=_csv_tuple(args.report_panels),
        sections=sections,
        include_case_ids=args.include_case_ids,
        include_case_appendix=args.include_case_appendix,
    )


def _json_object(value: str, option_name: str) -> dict:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{option_name} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{option_name} must contain a JSON object")
    return payload


def _csv_tuple(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise ValueError("At least one comma-separated value is required")
    return items


def _optional_csv_tuple(value: Optional[str]) -> Optional[tuple[str, ...]]:
    return _csv_tuple(value) if value else None


def _int_csv_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in _csv_tuple(value))


def _progress(message: str) -> None:
    _print(message)


def _print(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def _show_manifest(manifest) -> None:
    _print(
        f"{manifest.stage} complete; rows={manifest.row_counts}; "
        f"outputs={len(manifest.output_paths)}"
    )


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())
