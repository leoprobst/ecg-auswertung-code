from __future__ import annotations

import os
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def test_atlas_revision_changes_when_later_stage_manifest_arrives(tmp_path: Path) -> None:
    from vcg_biomarker.gui.widgets.mimic_project_widget import _artifact_revision

    validation = tmp_path / "validate.manifest.json"
    discovery = tmp_path / "phenome.manifest.json"
    broad = tmp_path / "broad_axes.manifest.json"
    external = tmp_path / "population_cohorts.manifest.json"
    paths = (validation, discovery, broad, external)
    before = _artifact_revision(paths)
    validation.write_text("complete", encoding="utf-8")
    after_validation = _artifact_revision(paths)
    discovery.write_text("complete", encoding="utf-8")
    after_discovery = _artifact_revision(paths)
    broad.write_text("complete", encoding="utf-8")
    after_broad = _artifact_revision(paths)
    external.write_text("complete", encoding="utf-8")
    assert len({before, after_validation, after_discovery, after_broad,
                _artifact_revision(paths)}) == 5


def test_vcg_camera_contains_extreme_points_without_changing_them():
    from vcg_biomarker.gui.widgets.mimic_project_widget import _vcg_camera_bounds
    points = np.array([[-12., 0., 0.], [3., 2., 1.], [np.nan, 0., 0.]])
    before = points.copy()
    center, distance = _vcg_camera_bounds(points, 800, 400)
    radius = np.max(np.linalg.norm(points[:2] - center, axis=1))
    angle = np.arctan(np.tan(np.deg2rad(30.)) * .5)
    assert distance * np.sin(angle) > radius
    np.testing.assert_array_equal(points, before)


def test_group_vcg_keeps_pointwise_uncertainty_in_time_domain() -> None:
    from PyQt6.QtWidgets import QApplication
    import pyqtgraph as pg
    from vcg_biomarker.gui.widgets.mimic_project_widget import MimicProjectWidget

    app = QApplication.instance() or QApplication([])
    assert app is not None
    projection = pg.PlotWidget()
    delta_plot = pg.PlotWidget()
    view = SimpleNamespace(
        _group_vcg_gl=None,
        group_vcg_projection_plots={"XY": (projection, "x", "y")},
        group_vcg_delta_plot=delta_plot,
        _clear_group_vcg_view=lambda: None,
        _clear_group_vcg_projections=lambda: (projection.clear(), delta_plot.clear()),
    )
    summary = pd.DataFrame([
        {"label_state": state, "sample_index": index, "time_ms": float(index * 2),
         "median_x_mV": offset + index * .1, "median_y_mV": offset + index * .2,
         "median_z_mV": 0.0}
        for state, offset in (("negative", 0.0), ("positive", 1.0))
        for index in range(3)
    ])
    difference = pd.DataFrame({
        "time_ms": [0.0, 2.0, 4.0], "delta_magnitude_mV": [1.0, 1.1, 1.2],
        "delta_magnitude_ci_low_mV": [.8, .9, 1.0],
        "delta_magnitude_ci_high_mV": [1.2, 1.3, 1.4],
    })
    try:
        MimicProjectWidget._render_group_vcg(view, summary, difference)
        assert len(projection.plotItem.listDataItems()) == 9
        assert len(delta_plot.plotItem.listDataItems()) == 3
    finally:
        projection.close()
        delta_plot.close()


def test_continuous_research_score_is_not_labelled_auc():
    from vcg_biomarker.gui.widgets.research_results_widget import research_rows
    payload = {"version": "1.1.0", "clinical_claim_allowed": False,
        "task_registry": [], "results": [{"endpoint_id": "potassium",
        "models": {"vcg": {"evaluation": {"metrics": {"rmse": 0.4, "n_patients": 72}}}}}]}
    row = research_rows(payload)[0]
    assert row["metric"].startswith("RMSE")
    assert row["value"] == 0.4
    assert row["ci"] is None


def test_unknown_stage_progress_is_not_displayed_as_zero_percent() -> None:
    from vcg_biomarker.gui.run_progress import CalibratedJobProgress
    from vcg_biomarker.gui.widgets import mimic_project_widget as widget
    from vcg_biomarker.mimic.models import JobState

    record = SimpleNamespace(state=JobState.RUNNING)
    estimate = CalibratedJobProgress(
        fraction=0.0, remaining_seconds=None, stages_completed=9,
        stages_total=19, current_stage="attest-event-evidence",
        current_stage_elapsed_seconds=120, basis="Restarbeit unbekannt",
    )
    label = widget._job_progress_text(record, estimate)
    assert "9/19" in label and "unbekannt" in label
    assert "%" not in label
    tooltip = widget._job_cell_tooltip(record, 2, estimate)
    assert "attest-event-evidence" in tooltip
    assert "Heartbeat" in tooltip and "kein Zeitanteil" in tooltip
    assert "unbekannt" in widget._job_eta_text(record, estimate)


def test_live_feature_progress_shows_exact_stage_count_and_scoped_eta() -> None:
    from vcg_biomarker.gui.run_progress import CalibratedJobProgress
    from vcg_biomarker.gui.widgets import mimic_project_widget as widget
    from vcg_biomarker.mimic.models import JobState

    record = SimpleNamespace(state=JobState.RUNNING)
    estimate = CalibratedJobProgress(
        fraction=0.0355, remaining_seconds=1_696_000,
        stages_completed=14, stages_total=37, current_stage='extract-features',
        current_stage_fraction_known=True, current_stage_fraction=0.0603,
        current_stage_message='Features 38,600/640,110; OK=38,250, Reused=6,725',
        current_stage_remaining_seconds=99 * 3600,
        estimate_has_unmodelled_work=True,
    )
    progress = widget._job_progress_text(record, estimate)
    assert '38,600/640,110' in progress and '6.03 %' in progress
    assert '3.55 %' in progress and 'zeitgewichtete Schätzung' in progress
    eta = widget._job_eta_text(record, estimate)
    assert 'Aktuelle Stufe' in eta and 'Gesamt: Planungswert' in eta


def test_full_run_uses_executed_live_stages_not_unused_pilot_gates(tmp_path: Path) -> None:
    from vcg_biomarker.gui.run_progress import calibrated_job_progress
    from vcg_biomarker.mimic.models import JobState

    (tmp_path / 'live_progress.json').write_text(json.dumps({
        'job_id': 'job-1', 'stages_total': 2, 'current_stage': 'extract-features',
        'current_stage_fraction': 0.25,
        'stages': [{'stage': 'inventory', 'state': 'done'},
                   {'stage': 'extract-features', 'state': 'running', 'elapsed_seconds': 100}],
    }), encoding='utf-8')
    record = SimpleNamespace(state=JobState.RUNNING, heartbeat_at=None,
        spec=SimpleNamespace(job_id='job-1', arguments={'study_spec': {'stage_ids': [
            'inventory', 'extract-features-pilot-1000', 'extract-features']}}))
    result = calibrated_job_progress(record, tmp_path)
    assert result.stages_total == 2 and result.stages_completed == 1
    assert result.current_stage == 'extract-features'
    assert result.current_stage_remaining_seconds == pytest.approx(300)


def test_measured_stage_progress_and_completion_keep_percent() -> None:
    from vcg_biomarker.gui.run_progress import CalibratedJobProgress
    from vcg_biomarker.gui.widgets import mimic_project_widget as widget
    from vcg_biomarker.mimic.models import JobState

    estimate = CalibratedJobProgress(
        fraction=0.5, remaining_seconds=120, stages_completed=1,
        stages_total=3, current_stage="extract-features",
        current_stage_fraction_known=True,
    )
    assert "50.0 %" in widget._job_progress_text(
        SimpleNamespace(state=JobState.RUNNING), estimate,
    )
    complete = CalibratedJobProgress(
        fraction=1.0, remaining_seconds=0, stages_completed=3, stages_total=3,
    )
    assert "100.0 %" in widget._job_progress_text(
        SimpleNamespace(state=JobState.COMPLETED), complete,
    )


def test_gui_backend_contract_and_superseded_failures() -> None:
    from vcg_biomarker.gui.widgets import mimic_project_widget as widget_module
    from vcg_biomarker.mimic.models import JobState

    synchronized, message = widget_module._gui_backend_contract_status()
    assert synchronized
    assert "synchron" in message
    spec = SimpleNamespace(
        stage="report",
        arguments={"run_id": "development"},
        resource_profile=SimpleNamespace(writer_namespace="reports"),
        project_file="project.json",
        created_at="2026-08-01T10:00:00+00:00",
    )
    failed = SimpleNamespace(spec=spec, state=JobState.FAILED)
    key = widget_module._job_supersession_key(failed)
    assert widget_module._job_display_group(failed, {}) == "attention"
    assert widget_module._job_display_group(
        failed,
        {key: "2026-08-01T11:00:00+00:00"},
    ) == "history"


def test_case_result_presentation_makes_projection_abstention_explicit() -> None:
    from vcg_biomarker.gui.widgets import mimic_project_widget as widget_module

    item = {
        "research_pattern_status": "not_evaluable",
        "not_evaluable_reason": "P-loop projection is not evaluable",
        "analysis_restriction": {
            "status": "p-contract",
            "version": "1.0.0",
            "abstention_reason": "P-loop projection is not evaluable",
        },
    }
    assert (
        widget_module._case_pattern_status_text(item)
        == "nicht beurteilbar – P-Loop-Restriktion"
    )
    tooltip = widget_module._analysis_restriction_tooltip(item)
    assert "Nicht beurteilbar" in tooltip
    assert "p-contract / 1.0.0" in tooltip


def test_gwas_gui_never_turns_feasibility_into_a_causal_claim() -> None:
    from vcg_biomarker.gui.widgets import mimic_project_widget as widget_module

    message, ready = widget_module._gwas_context_presentation(
        {
            "source_inventory_complete": True,
            "n_summary_files": 2,
            "available_ancestry_arms": ["EUR", "multi-ancestry"],
            "mendelian_randomization_status": "not_estimable",
            "polygenic_score_status": "not_estimable",
            "causal_claims_allowed": False,
        }
    )
    assert ready
    assert "nur als genetischer Architekturkontext" in message
    assert "nicht schätzbar" in message
    assert "Kausal-" in message

    missing, ready = widget_module._gwas_context_presentation({})
    assert not ready
    assert "separater Kontextarm" in missing


def test_population_forest_coordinates_are_log_scaled_and_clipped() -> None:
    from vcg_biomarker.gui.widgets import mimic_project_widget as widget_module

    valid, x, left, right, clipped_low, clipped_high = (
        widget_module._population_forest_coordinates(
            np.asarray([1.0, 600_000.0, 0.05, np.nan, -1.0]),
            np.asarray([0.5, 100_000.0, 0.01, 0.2, 0.2]),
            np.asarray([2.0, 900_000.0, 0.2, 2.0, 2.0]),
        )
    )
    assert valid.tolist() == [True, True, True, False, False]
    assert x.tolist() == pytest.approx([0.0, 3.0, -3.0])
    assert left.tolist() == pytest.approx([1.0, 0.0, 0.0])
    assert right.tolist() == pytest.approx(
        [1.0, 0.0, float(np.log2(0.2) - np.log2(0.125))]
    )
    assert clipped_low.tolist() == [False, False, True]
    assert clipped_high.tolist() == [False, True, False]
    assert np.all(np.isfinite(x))
    assert np.all(left >= 0.0)
    assert np.all(right >= 0.0)


def test_late_case_result_does_not_load_arrays_or_other_result_views():
    from vcg_biomarker.gui.widgets.mimic_project_widget import MimicProjectWidget
    # No config or result widgets: reaching either after rejection would fail.
    view = SimpleNamespace(expert_lab=SimpleNamespace(set_analysis=lambda payload: False))
    MimicProjectWidget._render_case_results(view, {'case_id': 'previous-case'})


def test_lab_evidence_link_opens_population_results_without_case_claim():
    from vcg_biomarker.gui.widgets.mimic_project_widget import MimicProjectWidget
    calls = []
    tabs = SimpleNamespace(
        count=lambda: 4,
        tabText=lambda index: ('Projekt', 'Analyse', 'Ergebnisse', 'ECG-Labor')[index],
        setCurrentIndex=lambda index: calls.append(('tab', index)),
    )
    view = SimpleNamespace(workspace_tabs=tabs, refresh_results=lambda: calls.append('refresh'))
    MimicProjectWidget._show_results_from_lab(view)
    assert calls == [('tab', 2), 'refresh']


def test_frozen_job_action_handlers_are_read_only(monkeypatch):
    from vcg_biomarker.gui.widgets import mimic_project_widget as module
    calls = []
    record = SimpleNamespace(spec=SimpleNamespace(arguments={'run_id': 'frozen'}))
    store = SimpleNamespace(get=lambda job: record, set_priority=lambda *a: calls.append('priority'))
    view = SimpleNamespace(_selected_job_id='job', job_store=store,
        _current_run_id=lambda: 'current', priority_spin=SimpleNamespace(value=lambda: 2),
        refresh_jobs=lambda: calls.append('refresh'))
    monkeypatch.setattr(module.QMessageBox, 'information', lambda *a: calls.append('notice'))
    monkeypatch.setattr(module, 'ensure_supervisor', lambda *a: calls.append('supervisor'))
    module.MimicProjectWidget._apply_job_action(view, lambda *a: calls.append('mutation'), start_supervisor=True)
    module.MimicProjectWidget.update_selected_priority(view)
    assert calls == ['notice']


def test_current_job_action_still_reaches_store(monkeypatch):
    from vcg_biomarker.gui.widgets import mimic_project_widget as module
    calls = []
    record = SimpleNamespace(spec=SimpleNamespace(arguments={'run_id': 'current'}))
    view = SimpleNamespace(_selected_job_id='job', job_store=SimpleNamespace(get=lambda job: record),
        _current_run_id=lambda: 'current', refresh_jobs=lambda: calls.append('refresh'), config='config')
    monkeypatch.setattr(module, 'ensure_supervisor', lambda config: calls.append('supervisor'))
    module.MimicProjectWidget._apply_job_action(view, lambda job: calls.append(job), start_supervisor=True)
    assert calls == ['job', 'supervisor', 'refresh']


def test_monitor_includes_frozen_active_runs_without_reenabling_history():
    from vcg_biomarker.gui.widgets import mimic_project_widget as module
    def record(job, run, state, stage='study-run'):
        return SimpleNamespace(spec=SimpleNamespace(job_id=job, arguments={'run_id': run}, stage=stage), state=state)
    current = record('new', 'current', module.JobState.COMPLETED)
    old = record('old', 'frozen', module.JobState.COMPLETED)
    running = record('active', 'frozen', module.JobState.RUNNING)
    paused = record('paused', 'frozen', module.JobState.PAUSED)
    technical = record('technical', 'frozen', module.JobState.RUNNING, 'inventory')
    result = module._jobs_for_monitor([current, old, running], [running, paused, technical], 'current')
    assert [r.spec.job_id for r in result] == ['new', 'active', 'paused']
    assert not module._job_visible_for_current_run(running, 'current')
    # A job omitted from the 250-entry history still appears from the active query.
    assert module._jobs_for_monitor([], [running], 'current') == [running]


def test_gui_hides_jobs_from_incompatible_scientific_runs() -> None:
    from vcg_biomarker.gui.widgets import mimic_project_widget as widget_module

    def record(arguments):
        return SimpleNamespace(spec=SimpleNamespace(arguments=arguments))

    current = "A010-2026-08-31-current"
    assert widget_module._job_visible_for_current_run(record({}), current)
    assert widget_module._job_visible_for_current_run(
        record({"run_id": current}), current
    )
    assert not widget_module._job_visible_for_current_run(
        record({"run_id": "A001-2026-08-20-legacy"}), current
    )
    assert not widget_module._job_visible_for_current_run(
        record({"study_spec": {"run_id": "pilot-legacy"}}), current
    )
    assert widget_module._result_run_visible_for_current(
        SimpleNamespace(run_id=current, validation_compatible=True), current
    )
    assert not widget_module._result_run_visible_for_current(
        SimpleNamespace(run_id=current, validation_compatible=False), current
    )
    assert not widget_module._result_run_visible_for_current(
        SimpleNamespace(run_id="A001-legacy", validation_compatible=True),
        current,
    )


def test_a017_calibration_does_not_treat_endpoint_count_as_work(
    tmp_path: Path,
) -> None:
    from vcg_biomarker.gui.run_progress import calibrated_job_progress
    from vcg_biomarker.mimic.models import JobState

    run_root = tmp_path / "A017"
    (run_root / "go_no_go").mkdir(parents=True)
    (run_root / "go_no_go" / "go_no_go.json").write_text(
        json.dumps(
            {
                "runtime_and_memory": {
                    "stage_duration_seconds": {"validate-pilot-1000": 100.0}
                }
            }
        ),
        encoding="utf-8",
    )
    (run_root / "live_progress.json").write_text(
        json.dumps(
            {
                "current_stage": "validate-pilot-1000",
                # A heterogeneous 29/34 endpoint counter must be ignored.
                "current_stage_fraction": 29 / 34,
                "stages": [
                    {
                        "stage": "validate-pilot-1000",
                        "state": "running",
                        "elapsed_seconds": 20.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    record = SimpleNamespace(
        state=JobState.RUNNING,
        started_at="2026-09-05T00:00:00+00:00",
        spec=SimpleNamespace(
            arguments={
                "study_spec": {"stage_ids": ["validate-pilot-1000"]}
            }
        ),
    )
    estimate = calibrated_job_progress(record, run_root)
    assert estimate is not None
    assert estimate.fraction == pytest.approx(0.0)
    assert estimate.remaining_seconds == pytest.approx(80.0)


def test_a017_feature_rate_scales_the_full_extraction_eta(
    tmp_path: Path,
) -> None:
    from vcg_biomarker.gui.run_progress import calibrated_job_progress
    from vcg_biomarker.mimic.models import JobState

    run_root = tmp_path / "A017"
    (run_root / "go_no_go").mkdir(parents=True)
    (run_root / "go_no_go" / "go_no_go.json").write_text(
        json.dumps(
            {
                "details": {
                    "performance": {
                        "pilots": {
                            "1000": {"feature_seconds_per_record": 0.4}
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (run_root / "live_progress.json").write_text(
        json.dumps(
            {
                "current_stage": "extract-features",
                "current_stage_fraction": 0.25,
                "stages": [
                    {
                        "stage": "extract-features",
                        "state": "running",
                        "elapsed_seconds": 100.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    record = SimpleNamespace(
        state=JobState.RUNNING,
        started_at="2026-09-05T00:00:00+00:00",
        spec=SimpleNamespace(
            arguments={"study_spec": {"stage_ids": ["extract-features"]}}
        ),
    )
    estimate = calibrated_job_progress(
        record,
        run_root,
        expected_development_records=1_000,
    )
    assert estimate is not None
    assert estimate.fraction == pytest.approx(0.25)
    assert estimate.remaining_seconds == pytest.approx(300.0)


def test_worker_resize_keeps_last_exact_feature_checkpoint_visible(tmp_path: Path) -> None:
    from vcg_biomarker.gui.run_progress import calibrated_job_progress
    from vcg_biomarker.gui.widgets.mimic_project_widget import _job_progress_text
    from vcg_biomarker.mimic.models import JobState

    (tmp_path / 'live_progress.json').write_text(json.dumps({
        'current_stage': 'extract-features',
        'current_stage_fraction': .25,
        'current_stage_message': 'Worker angepasst: 8 -> 10',
        'stages': [{'stage': 'extract-features', 'state': 'running',
                    'elapsed_seconds': 100}],
    }), encoding='utf-8')
    record = SimpleNamespace(
        state=JobState.RUNNING,
        progress_message='Features 251/1,000; OK=251, Fehler=0, Reused=0',
        spec=SimpleNamespace(arguments={'study_spec': {'stage_ids': ['extract-features']}}),
    )
    estimate = calibrated_job_progress(record, tmp_path)
    assert estimate is not None
    label = _job_progress_text(record, estimate)
    assert '25.00 %' in label
    assert 'letzter Feature-Checkpoint: 251/1,000' in label
    assert 'Gesamt:' in label


def test_only_a_complete_go_no_go_opens_the_full_run(tmp_path: Path) -> None:
    from vcg_biomarker.gui.widgets import mimic_project_widget as widget_module

    target = tmp_path / "go_no_go" / "go_no_go.json"
    target.parent.mkdir(parents=True)
    target.write_text(
        json.dumps(
            {
                "decision": "GO_FOR_FROZEN_FULL_RUN",
                "run_id": tmp_path.name,
                "details": {
                    f"pilot_{size}": {"features": {"row_counts": {"implausible_markers": 0}}}
                    for size in (1000, 10000)
                },
                "full_run_started": False,
                "gates": {"software": True, "locked_test_unread": True},
            }
        ),
        encoding="utf-8",
    )
    # Technical completion alone cannot authorize a full scientific run.
    assert not widget_module._full_run_authorized(tmp_path)
    target.write_text(
        json.dumps(
            {
                "decision": "GO_FOR_FROZEN_FULL_RUN",
                "full_run_started": False,
                "gates": {"software": True, "locked_test_unread": False},
            }
        ),
        encoding="utf-8",
    )
    assert not widget_module._full_run_authorized(tmp_path)


def test_progress_calibration_excludes_other_stages_and_tracks_heartbeat(tmp_path):
    from vcg_biomarker.gui.run_progress import calibrated_job_progress
    from vcg_biomarker.mimic.models import JobState

    (tmp_path / "runtime_calibration.json").write_text(json.dumps({
        "runtime_and_memory": {"stage_duration_seconds": {
            "validate-pilot-1000": 100, "unrelated": 10000,
        }},
    }), encoding="utf-8")
    (tmp_path / "live_progress.json").write_text(json.dumps({
        "updated_at": "2026-09-05T00:00:00+00:00",
        "current_stage": "validate-pilot-1000",
        "stages": [{"stage": "validate-pilot-1000", "state": "running",
                    "elapsed_seconds": 20}],
    }), encoding="utf-8")
    record = SimpleNamespace(
        state=JobState.RUNNING, heartbeat_at="2026-09-05T00:00:10+00:00",
        spec=SimpleNamespace(arguments={"study_spec": {
            "stage_ids": ["validate-pilot-1000"],
        }}),
    )
    result = calibrated_job_progress(record, tmp_path)
    assert result.fraction == pytest.approx(0.0)
    assert result.remaining_seconds == pytest.approx(70)
    assert result.current_stage_elapsed_seconds == pytest.approx(30)
    assert not result.current_stage_fraction_known
    # Repainting without a new heartbeat does not simulate more work.
    assert calibrated_job_progress(record, tmp_path) == result
    record.state = JobState.PAUSED
    assert calibrated_job_progress(record, tmp_path).fraction == pytest.approx(0.0)


def test_run_local_result_paths_never_search_sibling_runs(
    tmp_path: Path,
) -> None:
    from vcg_biomarker.gui.widgets import mimic_project_widget as widget_module

    current = tmp_path / "current"
    historical = tmp_path / "historical"
    historical_phenome = (
        historical / "phenome" / "broad_discovery_ranked.parquet"
    )
    historical_phenome.parent.mkdir(parents=True)
    historical_phenome.write_bytes(b"historical")
    historical_transfer = historical / "transfer_matrix.parquet"
    historical_transfer.write_bytes(b"historical")

    assert widget_module._run_local_phenome_path(current) == (
        current / "phenome" / "broad_discovery_ranked.parquet"
    )
    assert widget_module._run_local_transfer_path(current) == (
        current / "ptb_transfer_matrix.parquet"
    )

    current_legacy_phenome = (
        current / "phenome" / "phenome_replicated_hits.parquet"
    )
    current_legacy_phenome.parent.mkdir(parents=True)
    current_legacy_phenome.write_bytes(b"current")
    current_legacy_transfer = current / "transfer_matrix.parquet"
    current_legacy_transfer.write_bytes(b"current")
    assert widget_module._run_local_phenome_path(current) == current_legacy_phenome
    assert widget_module._run_local_transfer_path(current) == (
        current_legacy_transfer
    )


def test_feature_failure_report_requires_current_run_provenance(
    tmp_path: Path,
) -> None:
    from vcg_biomarker.gui.widgets.mimic_project_widget import MimicProjectWidget

    runs_root = tmp_path / "runs"
    features_root = tmp_path / "features"
    old_root = features_root / "old-panel" / "development"
    old_root.mkdir(parents=True)
    (old_root / "features.manifest.json").write_text(
        json.dumps({"status": "complete"}),
        encoding="utf-8",
    )
    (old_root / "feature_failure_report.parquet").write_bytes(b"old")
    current_run = runs_root / "current"
    current_run.mkdir(parents=True)
    stub = SimpleNamespace(
        config=SimpleNamespace(
            runs_root=runs_root,
            features_root=features_root,
        )
    )
    unavailable = MimicProjectWidget._feature_failure_path(stub, "current")
    assert unavailable == features_root / "__not_available__.parquet"

    current_root = features_root / "current-panel" / "development"
    current_root.mkdir(parents=True)
    current_report = current_root / "feature_failure_report.parquet"
    current_report.write_bytes(b"current")
    (current_run / "development_provenance.json").write_text(
        json.dumps({"feature_integrity": {"root": str(current_root)}}),
        encoding="utf-8",
    )
    assert MimicProjectWidget._feature_failure_path(stub, "current") == (
        current_report.resolve()
    )


def test_case_results_from_historical_run_are_stripped() -> None:
    from vcg_biomarker.gui.widgets.mimic_project_widget import MimicProjectWidget

    stub = SimpleNamespace(_current_run_id=lambda: "current")
    payload = {
        "feature_values": {"spatial_qrs_t_angle_deg": 123.0},
        "classical_criteria": {"qrs": {"positive": True}},
        "research_findings": [{"endpoint_id": "old-finding"}],
        "score_releases": [{"endpoint_id": "old-score"}],
        "single_marker_patterns": [{"feature_id": "old-marker"}],
        "research_diagnosis_summary": {
            "probable_research_diagnoses": [{"endpoint_id": "old"}]
        },
        "score_release_inventory": {"source_run_id": "historical"},
    }
    safe = MimicProjectWidget._case_payload_for_current_run(stub, payload)
    assert safe["research_findings"] == []
    assert safe["score_releases"] == []
    assert safe["single_marker_patterns"] == []
    assert safe["criterion_inventory"]["status"] == "historical_run_hidden"
    assert safe["feature_values"] == payload["feature_values"]
    assert safe["classical_criteria"] == payload["classical_criteria"]
    assert payload["research_findings"]


def test_preview_validation_is_not_reported_as_formal_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vcg_biomarker.gui.widgets import mimic_project_widget as widget_module

    run_root = tmp_path / "run"
    run_root.mkdir()
    monkeypatch.setattr(
        widget_module,
        "_validation_complete",
        lambda path: path == run_root,
    )
    (run_root / "validate.manifest.json").write_text(
        json.dumps({"status": "complete"}),
        encoding="ascii",
    )
    provenance_path = run_root / "development_provenance.json"
    provenance_path.write_text(
        json.dumps(
            {
                "feature_scope": "development_preview",
                "preview": True,
                "analysis_completeness": "partial_preview",
                "locked_test_labels_read": False,
            }
        ),
        encoding="ascii",
    )
    assert not widget_module._formal_development_validation_complete(
        run_root
    )
    assert widget_module._validation_presentation(run_root) == (
        "partielle Vorschau (nicht einfrierbar)",
        False,
    )
    preview_payload = json.loads(provenance_path.read_text(encoding="ascii"))
    preview_payload["sampling_design"] = "endpoint_enriched_pilot"
    provenance_path.write_text(json.dumps(preview_payload), encoding="ascii")
    assert widget_module._validation_presentation(run_root) == (
        "endpunktangereicherter Pilot "
        "(keine Prävalenzschätzung, nicht einfrierbar)",
        False,
    )
    preview_payload.update(
        {
            "locked_test_labels_read": False,
            "prevalence_metrics_interpretable": False,
            "code_hash": "validation-current",
            "feature_integrity": {
                "source_code_hash": "feature-current"
            },
        }
    )
    provenance_path.write_text(json.dumps(preview_payload), encoding="ascii")
    (run_root / "validate.manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "stage_version": widget_module.VALIDATION_STAGE_VERSION,
                "notes": [
                    "freeze_eligible=False",
                    "prevalence_dependent_metrics=disabled_for_preview",
                ],
            }
        ),
        encoding="ascii",
    )
    assert widget_module._pilot_validation_compatible(
        run_root,
        expected_feature_code_hash="feature-current",
        expected_validation_code_hash="validation-current",
    )

    provenance_path.write_text(
        json.dumps(
            {
                "feature_scope": "development_only",
                "preview": False,
                "analysis_completeness": "complete_snapshot",
                "locked_test_labels_read": False,
            }
        ),
        encoding="ascii",
    )
    assert widget_module._formal_development_validation_complete(run_root)
    assert not widget_module._formal_development_validation_complete(
        run_root,
        expected_feature_code_hash="feature-current",
        expected_validation_code_hash="validation-current",
    )
    assert widget_module._validation_presentation(
        run_root,
        expected_feature_code_hash="feature-current",
        expected_validation_code_hash="validation-current",
    ) == (
        "historischer Entwicklungslauf "
        "(Code oder Feature-Snapshot veraltet)",
        False,
    )
    provenance_path.write_text(
        json.dumps(
            {
                "feature_scope": "development_only",
                "preview": False,
                "analysis_completeness": "complete_snapshot",
                "locked_test_labels_read": False,
                "code_hash": "validation-current",
                "feature_integrity": {
                    "source_code_hash": "feature-current",
                },
            }
        ),
        encoding="ascii",
    )
    assert widget_module._formal_development_validation_complete(
        run_root,
        expected_feature_code_hash="feature-current",
        expected_validation_code_hash="validation-current",
    )
    assert widget_module._validation_presentation(
        run_root,
        expected_feature_code_hash="feature-current",
        expected_validation_code_hash="validation-current",
    ) == ("vollst\u00e4ndig", True)
    assert widget_module._validation_presentation(run_root) == (
        "vollst\u00e4ndig",
        True,
    )


def test_feature_readiness_requires_current_stage_and_code_hash(
    tmp_path: Path,
) -> None:
    from vcg_biomarker.gui.widgets import mimic_project_widget as widget_module

    root = tmp_path / "features" / "development"
    root.mkdir(parents=True)
    manifest = root / "features.manifest.json"
    partition_index = root / "partition_index.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "complete",
                "stage_version": widget_module.FEATURE_STAGE_VERSION,
            }
        ),
        encoding="ascii",
    )
    partition_index.write_text(
        json.dumps(
            {
                "source_code_hash": "feature-old",
                "artifact_hash": "artifact",
                "row_count": 10,
            }
        ),
        encoding="ascii",
    )
    assert not widget_module._feature_manifest_complete(
        manifest,
        expected_code_hash="feature-current",
    )

    partition_index.write_text(
        json.dumps(
            {
                "source_code_hash": "feature-current",
                "artifact_hash": "artifact",
                "row_count": 10,
            }
        ),
        encoding="ascii",
    )
    assert widget_module._feature_manifest_complete(
        manifest,
        expected_code_hash="feature-current",
    )

    manifest.write_text(
        json.dumps(
            {
                "status": "complete",
                "stage_version": "legacy-stage",
            }
        ),
        encoding="ascii",
    )
    assert not widget_module._feature_manifest_complete(
        manifest,
        expected_code_hash="feature-current",
    )


def test_biomarker_audit_readiness_requires_current_clean_hash(
    tmp_path: Path,
) -> None:
    from vcg_biomarker.gui.widgets import mimic_project_widget as widget_module

    path = tmp_path / "biomarker_audit_summary.json"
    payload = {
        "audit_version": widget_module.BIOMARKER_AUDIT_VERSION,
        "implementation_hash": "audit-current",
        "calculator_errors": 0,
        "default_panel_blockers": 0,
    }
    path.write_text(json.dumps(payload), encoding="ascii")
    assert widget_module._biomarker_audit_current(
        path,
        expected_implementation_hash="audit-current",
    )

    payload["implementation_hash"] = "audit-old"
    path.write_text(json.dumps(payload), encoding="ascii")
    assert not widget_module._biomarker_audit_current(
        path,
        expected_implementation_hash="audit-current",
    )

    payload["implementation_hash"] = "audit-current"
    payload["calculator_errors"] = 1
    path.write_text(json.dumps(payload), encoding="ascii")
    assert not widget_module._biomarker_audit_current(
        path,
        expected_implementation_hash="audit-current",
    )


def test_delineation_readiness_requires_current_three_dataset_core(
    tmp_path: Path,
) -> None:
    from vcg_biomarker.gui.widgets import mimic_project_widget as widget_module

    path = tmp_path / "benchmark_suite.json"
    payload = {
        "suite_version": widget_module.DELINEATION_SUITE_VERSION,
        "implementation_hash": "delineation-current",
        "datasets_completed": ["ludb", "qtdb", "nstdb"],
        "datasets_missing": ["afdb", "incart", "staffiii"],
        "release_core_complete": True,
        "locked_test_read": False,
    }
    path.write_text(json.dumps(payload), encoding="ascii")
    assert widget_module._delineation_benchmark_current(
        path,
        expected_implementation_hash="delineation-current",
    )

    payload["datasets_completed"] = ["ludb", "qtdb"]
    path.write_text(json.dumps(payload), encoding="ascii")
    assert not widget_module._delineation_benchmark_current(
        path,
        expected_implementation_hash="delineation-current",
    )

def test_required_group_vcg_readiness_requires_every_output(
    tmp_path: Path,
) -> None:
    from vcg_biomarker.gui.widgets import mimic_project_widget as widget_module

    run_root = tmp_path / "run"
    for endpoint_id in widget_module.DEFAULT_GROUP_VCG_ENDPOINTS:
        root = (
            run_root
            / "group_vcg"
            / f"endpoint={endpoint_id}"
            / "split=validation"
        )
        root.mkdir(parents=True)
        output = root / "group_vcg_summary.parquet"
        output.write_text("test", encoding="ascii")
        (root / "group_vcg.manifest.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "output_paths": [str(output.resolve())],
                    "notes": [
                        "scope=development_only",
                        "locked_test_read=False",
                    ],
                }
            ),
            encoding="ascii",
        )
    assert widget_module._required_group_vcg_complete(run_root)

    missing = (
        run_root
        / "group_vcg"
        / f"endpoint={widget_module.DEFAULT_GROUP_VCG_ENDPOINTS[0]}"
        / "split=validation"
        / "group_vcg_summary.parquet"
    )
    missing.unlink()
    assert not widget_module._required_group_vcg_complete(run_root)


def test_mimic_widget_reports_project_readiness_and_current_review_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("PyQt6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication, QLabel, QPushButton, QToolBar

    from vcg_biomarker.gui.widgets import mimic_project_widget as widget_module
    from vcg_biomarker.mimic.config import MimicProjectConfig
    from vcg_biomarker.mimic.jobs import JobStore

    database_root = tmp_path / "database"
    mimic_iv = database_root / "mimic-iv-3.1"
    mimic_ed = database_root / "mimic-iv-ed-2.2"
    mimic_ecg = database_root / "mimic-iv-ecg"
    for directory in (mimic_iv, mimic_ed, mimic_ecg):
        directory.mkdir(parents=True)
    echo_measurements = database_root / "structured-measurement.csv"
    echo_studies = database_root / "echo-study-list.csv"
    echo_records = database_root / "echo-record-list.csv"
    for path in (echo_measurements, echo_studies, echo_records):
        path.write_text("", encoding="ascii")

    config = MimicProjectConfig(
        mimic_iv_root=mimic_iv,
        mimic_ed_root=mimic_ed,
        mimic_ecg_root=mimic_ecg,
        echo_structured_path=echo_measurements,
        echo_study_list_path=echo_studies,
        echo_record_list_path=echo_records,
        work_root=tmp_path / "work",
    )
    config.ensure_directories()
    JobStore.for_project(config).create(
        stage="inventory",
        project_file=config.save(),
        job_id="reconnect-test",
    )
    pd.DataFrame(
        {
            "source": ["mimic_iv"],
            "table_id": ["patients"],
            "status": ["ok"],
            "row_count": [1],
            "diagnostic": [""],
        }
    ).to_parquet(config.work_root / "data_inventory.parquet", index=False)

    current_run, _ = widget_module.resolve_current_analysis(config)
    report_root = config.runs_root / current_run.run_id / "reports"
    report_root.mkdir(parents=True)
    pd.DataFrame(
        {
            "subject_id": [1],
            "study_id": [11],
            "split": ["validation"],
            "ecg_time": ["2026-01-01"],
            "endpoint_id": ["reduced_lvef"],
            "clinical_label_state": ["positive"],
            "label_confidence": ["high"],
            "label_time_precision": ["timestamp"],
            "label_evidence_ids": ["echo:7"],
            "label_evidence": ["LVEF 35%"],
            "vcg_feature_id": ["spatial_qrs_t_angle_deg"],
            "vcg_value": [132.0],
            "threshold_distance": [0.8],
            "signal_quality_score": [0.95],
            "discordance_category": [
                "clinical_positive_classical_negative_vcg_positive"
            ],
            "review_status": ["pending"],
            "review_priority": [1.0],
        }
    ).to_parquet(report_root / "review_candidates.parquet", index=False)
    transfer_path = report_root.parent / "ptb_transfer_matrix.parquet"
    pd.DataFrame(
        {
            "transfer_type": [
                "shared_single_marker",
                "ptb_frozen_score",
            ],
            "ptb_validation_rank_biserial": [0.3, 0.4],
            "mimic_validation_rank_biserial": [0.2, -0.1],
            "train_direction_agreement": [True, False],
        }
    ).to_parquet(transfer_path, index=False)

    monkeypatch.setattr(
        widget_module,
        "load_or_discover_config",
        lambda: config,
    )
    app = QApplication.instance() or QApplication([])
    assert app is not None
    widget = widget_module.MimicProjectWidget()
    try:
        assert widget.results_mode_combo.currentData() == "simple"
        assert not widget.results_atlas_tabs.isTabVisible(1)
        assert widget.atlas_run_combo.count() == widget.results_run_combo.count()
        widget.results_mode_combo.setCurrentIndex(1)
        assert widget.results_atlas_tabs.isTabVisible(1)
        widget.results_mode_combo.setCurrentIndex(0)
        assert not widget.results_atlas_tabs.isTabVisible(1)
        viewport = widget.ecg_plot.parentWidget().parentWidget().parentWidget()
        assert viewport.maximumHeight() == 480
        # The run name is no longer typed. It is derived from what
        # determines the result, so the same configuration continues its
        # analysis instead of forking a new one on every restart.
        assert not hasattr(widget, "run_id")
        assert widget.analysis_name_label.text()
        assert widget.analysis_notes.text() == ""
        # Review-detail rendering is tested independently of the result-run
        # compatibility gate. The synthetic fixture does not build a complete
        # validation provenance bundle, so select its current run explicitly.
        widget.report_run_combo.addItem("synthetic current", current_run.run_id)
        widget.report_run_combo.setCurrentIndex(
            widget.report_run_combo.count() - 1
        )
        widget.refresh_review()
        widget.review_table.selectRow(0)
        widget._show_review_details()
        widget._draw_transfer(transfer_path)
        details = widget.review_details.toPlainText()

        assert widget.workspace_tabs.count() == 6
        assert [
            widget.workspace_tabs.tabText(index)
            for index in range(widget.workspace_tabs.count())
        ] == [
            "Projekt",
            "Analyse",
            "Ergebnisse",
            "ECG-Labor",
            "Berichte",
            "Simulation",
        ]
        assert widget.project_tabs.count() == 2
        assert widget.project_tabs.tabText(1) == "Datengrundlage"
        # No purpose dropdown: the study has one configuration, and the
        # only remaining question -- how long to keep refining -- is
        # answered while the run is going, not before it starts.
        assert not hasattr(widget, "preset_combo")
        assert widget._preset_id() == "exhaustive_research"
        assert "auszuwählen" in widget.study_scope_label.text()
        assert widget.study_action_button.text() == "Analyse starten"
        assert widget.pause_button.isHidden()
        assert widget.resume_button.isHidden()
        assert widget.retry_button.isHidden()
        assert widget.cancel_button.isHidden()
        assert not widget.technical_tools_contents.isVisible()
        assert widget.stage_combo.parentWidget() is widget.technical_tools_contents
        assert widget.project_status.text().startswith("Projekt bereit")
        assert "Backend synchron" in widget.backend_sync_status.text()
        assert widget.result_view_combo.count() == widget.result_tabs.count()
        assert widget.result_tabs.tabBar().isHidden()
        assert widget.cohort_tabs.tabText(
            widget.cohort_tabs.count() - 1
        ) == "Laufender Teilstand"
        assert widget.foundation_overview.rowCount() == 7
        assert widget.cohort_tabs.tabText(1) == "Klinische Referenzen"
        assert widget.live_marker_combo.toolTip().strip()
        assert widget.live_feature_plot.toolTip().strip()
        assert widget.transfer_plot.toolTip().strip()
        assert len(widget.transfer_plot.listDataItems()) == 2
        assert widget.future_phenotype_limit_combo.toolTip().strip()
        assert (
            widget.inference_filter.currentData()
            == "formal_inference"
        )
        assert widget.inference_filter.toolTip().strip()
        screening_where, screening_parameters = (
            widget._screening_result_filter()
        )
        assert "inference_status = ?" in screening_where
        assert "feature_id NOT IN" in screening_where
        assert "formal_inference" in screening_parameters
        assert "fiducial_confidence" in screening_parameters
        assert widget.minimumSizeHint().width() <= 1366
        assert widget.minimumSizeHint().height() <= 768
        assert hasattr(widget, "population_diagnostic_table")
        assert hasattr(widget, "population_agreement_table")
        assert "Externe Populationen" in {
            widget.guideline_tabs.tabText(index)
            for index in range(widget.guideline_tabs.count())
        }
        assert {
            "CODE-15-Befundlabels",
            "Direct-8-Lead-Agreement",
        }.issubset(
            {
                widget.population_tabs.tabText(index)
                for index in range(widget.population_tabs.count())
            }
        )
        assert all(
            widget.result_view_combo.itemText(index).strip()
            for index in range(widget.result_view_combo.count())
        )
        assert widget.jobs_table.rowCount() == 1
        assert widget.jobs_table.item(0, 0).text() == "Datenquellen prüfen"
        assert (
            widget.jobs_table.item(0, 0).data(Qt.ItemDataRole.UserRole)
            == "reconnect-test"
        )
        assert "clinical_label_state: positive" in details
        assert "label_evidence_ids: echo:7" in details
        assert "vcg_feature_id: spatial_qrs_t_angle_deg" in details
        buttons = widget.findChildren(QPushButton)
        assert buttons
        assert all(button.toolTip().strip() for button in buttons)
        action_text = " ".join(button.text() for button in buttons).lower()
        assert "evaluate-test" not in action_text
        assert "locked test auswerten" not in action_text
        assert "ptb fold 10" not in action_text
        visible_text = " ".join(
            [label.text() for label in widget.findChildren(QLabel)]
            + [
                widget.workspace_tabs.tabText(index)
                for index in range(widget.workspace_tabs.count())
            ]
        )
        assert not any(
            marker in visible_text
            for marker in ("\u00c3", "\u00c2", "\ufffd")
        )
        assert not widget.include_case_ids.isChecked()
        assert not widget.include_case_appendix.isChecked()
        assert widget.report_endpoint_combo.count() > 50
        assert widget.report_marker_combo.count() > 100
        assert widget.report_endpoint_combo.toolTip().strip()
        assert widget.report_marker_combo.toolTip().strip()
        assert widget.report_endpoint_combo.isHidden()
        assert widget.report_marker_combo.isHidden()
        widget.report_type_combo.setCurrentIndex(
            widget.report_type_combo.findData("endpoint")
        )
        assert not widget.report_endpoint_combo.isHidden()
        assert widget.report_marker_combo.isHidden()
        widget.report_type_combo.setCurrentIndex(
            widget.report_type_combo.findData("marker")
        )
        assert not widget.report_marker_combo.isHidden()
        assert widget.report_endpoint_combo.isHidden()
        result_tabs = [
            widget.result_tabs.tabText(index)
            for index in range(widget.result_tabs.count())
        ]
        assert {
            "ROC & PR",
            "Scoreverteilungen",
            "Betriebspunkte",
            "Gruppen-VCG",
            "Sensitivität & Subgruppen",
            "Adjustierung & Nutzen",
        } <= set(result_tabs)
        assert widget.group_vcg_build_button.toolTip().strip()
        assert widget.group_vcg_audit_table.toolTip().strip()
        assert widget.result_perspective_tabs.count() == 4
        assert [
            widget.result_perspective_tabs.tabText(index)
            for index in range(widget.result_perspective_tabs.count())
        ] == [
            "Pathologien",
            "Biomarker",
            "Kriterienkandidaten",
            "Neue Zusammenhänge",
        ]
        assert widget.discovery_scope_combo.toolTip().strip()
        assert widget.discovery_cardiac_combo.findData(
            "non_cardiovascular"
        ) >= 0
        assert widget.discovery_status_combo.findData("replicated") >= 0
        widget.result_perspective_tabs.setCurrentIndex(3)
        assert widget.result_sort_combo.findData(
            "association_strength"
        ) >= 0
        widget.discovery_scope_combo.setCurrentIndex(
            widget.discovery_scope_combo.findData("pathology_hypothesis")
        )
        widget.discovery_cardiac_combo.setCurrentIndex(
            widget.discovery_cardiac_combo.findData("non_cardiovascular")
        )
        discovery_query = widget._primary_result_query()
        assert discovery_query.candidate_types == (
            "pathology_hypothesis",
        )
        assert discovery_query.baseline_ids == (
            "non_cardiovascular",
        )
        assert widget.case_findings.toolTip().strip()
        widget.case_findings.sortItems(0, Qt.SortOrder.AscendingOrder)
        # This isolated result-table fixture has no selected lab case. Keep
        # the production guard against late results from another case intact.
        monkeypatch.setattr(widget.expert_lab, 'set_analysis', lambda _payload: True)
        widget._render_case_results({"research_findings": [
            {"endpoint_id": "legacy", "sensitivity": .99, "specificity": .95},
            {"endpoint_id": "explicit", "sensitivity": .99,
             "primary_operating_point": {"sensitivity_validation": .60, "specificity_validation": .90}},
        ]})
        assert widget.case_findings.isSortingEnabled()
        assert widget.case_findings.item(0, 0).text() == 'explicit'
        assert "99" not in widget.case_findings.item(1, 8).text()
        assert "Validierungsherkunft" in widget.case_findings.item(1, 8).toolTip()
        assert widget.case_findings.item(0, 8).text() == widget_module._format_percentage(.60)
        assert widget.case_findings.item(0, 9).text() == widget_module._format_percentage(.90)
        assert 'Expliziter Wert' in widget.case_findings.item(0, 8).toolTip()
        assert "erreichte" in widget.case_findings.horizontalHeaderItem(9).text()
        assert "95-%-Spez.-Ziel" in widget.case_scores.horizontalHeaderItem(7).text()
        widget.results_run_combo.clear()
        widget.results_run_combo.addItem("Pilot", current_run.run_id)
        monkeypatch.setattr(
            widget.result_repository,
            "available_runs",
            lambda **_kwargs: (
                SimpleNamespace(
                    run_id=current_run.run_id,
                    complete_compatible=False,
                    validation_compatible=True,
                ),
            ),
        )
        pilot_query = widget._primary_result_query()
        assert not pilot_query.complete_runs_only
        assert not pilot_query.include_historical

        widget.results_run_combo.clear()
        widget.results_run_combo.addItem("Historisch", "A001-legacy")
        historical_query = widget._primary_result_query()
        assert historical_query.run_id == ""
        assert not historical_query.include_historical

        widget.report_run_combo.addItem("Ohne Review", "no-review")
        widget.report_run_combo.setCurrentIndex(
            widget.report_run_combo.findData("no-review")
        )
        widget.refresh_review()
        assert widget._active_review_path is None
        assert "keine lokale" in widget.review_details.toPlainText()
    finally:
        widget.close()

    from vcg_biomarker.gui.main_window import MainWindow

    window = MainWindow()
    try:
        assert window.tab_widget.isTabVisible(0)
        assert window.tab_widget.tabBar().isHidden()
        assert window.menuBar().isHidden()
        assert all(
            not window.tab_widget.isTabVisible(index)
            for index in range(1, window.tab_widget.count())
        )
        assert all(
            toolbar.isHidden()
            for toolbar in window.findChildren(QToolBar)
        )
        file_menu = window.menuBar().actions()[0].menu()
        visible_file_actions = {
            action.text().replace("&", "")
            for action in file_menu.actions()
            if action.isVisible() and not action.isSeparator()
        }
        assert visible_file_actions == {"Exit"}
        assert "workers" not in window._status_label.text().casefold()
        assert window._status_label.text().strip()
    finally:
        window.close()


def test_paged_table_displays_clinical_names_instead_of_internal_ids(
    tmp_path: Path,
) -> None:
    pytest.importorskip("PyQt6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication

    from vcg_biomarker.gui.widgets.paged_table import ParquetPageModel

    path = tmp_path / "friendly.parquet"
    pd.DataFrame(
        {
            "endpoint_id": ["qtc_gt_480"],
            "feature_id": ["spatial_qrs_t_angle_deg"],
        }
    ).to_parquet(path, index=False)
    app = QApplication.instance() or QApplication([])
    model = ParquetPageModel()
    model.set_source(path)
    try:
        endpoint = model.index(0, 0)
        marker = model.index(0, 1)
        assert "QT" in str(model.data(endpoint))
        assert model.data(endpoint) != "qtc_gt_480"
        assert "Interne ID: qtc_gt_480" in str(
            model.data(endpoint, Qt.ItemDataRole.ToolTipRole)
        )
        assert model.data(marker) != "spatial_qrs_t_angle_deg"
    finally:
        model.deleteLater()
        app.processEvents()


def test_paged_parquet_table_diagnoses_zero_column_artifact(
    tmp_path: Path,
) -> None:
    pytest.importorskip("PyQt6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PyQt6.QtWidgets import QApplication

    from vcg_biomarker.gui.widgets.paged_table import ParquetPageModel

    path = tmp_path / "empty.parquet"
    pd.DataFrame().to_parquet(path, index=False)
    app = QApplication.instance() or QApplication([])
    model = ParquetPageModel()
    model.set_source(path, where_sql="endpoint_id = ?", parameters=("x",))
    try:
        assert model.last_error
        assert model.rowCount() == 1
        assert model.frame.iloc[0]["status"] == "nicht verfügbar"
    finally:
        model.deleteLater()
        app.processEvents()


def test_guideline_comparison_pairs_classical_criterion_with_new_marker() -> (
    None
):
    from vcg_biomarker.gui.widgets import mimic_project_widget as widget_module

    payload = {
        "research_findings": [
            {
                "endpoint_id": "echo_lvh_strict",
                "endpoint_name_de": "Echo-LVH (strikt)",
            },
        ],
        "score_releases": [
            {
                "endpoint_id": "echo_lvh_strict",
                "release_id": "lvh-score-v1",
                "stage": "development",
                "research_score": 1.23,
                "research_pattern_status": (
                    "above_train_95_specificity_threshold"
                ),
                "evidence_status": "development_validation_stable",
                "primary_operating_point": {
                    "distance_to_threshold": 0.42,
                },
                "validation_metrics": {
                    "roc_auc": 0.81,
                    "roc_auc_ci_low": 0.76,
                    "roc_auc_ci_high": 0.86,
                    "sensitivity_at_95_specificity": 0.31,
                },
                "classical_comparators": [
                    {
                        "display_name": "Cornell-Kriterium",
                        "criterion_id": "cornell_lvh",
                        "evaluable": True,
                        "positive": False,
                        "value": 1.9,
                    },
                    {
                        "display_name": "Sokolow-Lyon",
                        "criterion_id": "sokolow_lyon_lvh",
                        "evaluable": False,
                        "positive": None,
                        "value": None,
                    },
                ],
            },
            {
                "endpoint_id": "acute_mi_high_specificity",
                "release_id": "",
                "stage": "development",
                "research_score": None,
                "research_pattern_status": "not_evaluable",
                "evidence_status": "development_evidence_unavailable",
                "primary_operating_point": {},
                "validation_metrics": {},
                "classical_comparators": [],
            },
        ],
    }

    rows = widget_module._build_guideline_comparison_rows(payload)

    assert len(rows) == 3
    lvh_rows = {row[1]: row for row in rows if row[0] == "Echo-LVH (strikt)"}
    cornell_row = lvh_rows["Cornell-Kriterium"]
    assert cornell_row[2] == "negativ"
    assert cornell_row[3] == "1.9"
    assert cornell_row[4] == "lvh-score-v1 · development"
    assert cornell_row[5] == "1.23"
    assert cornell_row[6] == "0.42"
    assert cornell_row[7] == "above_train_95_specificity_threshold"
    assert cornell_row[8] == "0.81 [0.76; 0.86]"
    assert cornell_row[9] == "0.31"
    assert cornell_row[10] == "development_validation_stable"

    sokolow_row = lvh_rows["Sokolow-Lyon"]
    assert sokolow_row[2] == "nicht auswertbar"

    no_criterion_row = next(
        row for row in rows if row[0] == "acute_mi_high_specificity"
    )
    assert no_criterion_row[1] == "kein anerkanntes klassisches Kriterium"
    assert no_criterion_row[2] == "nicht auswertbar"

    # Without a comparison_lookup entry, the new DB-wide verdict column is
    # a plain "-", never a fabricated claim.
    assert cornell_row[11] == "-"


def test_guideline_comparison_attaches_db_wide_verdict_when_available() -> (
    None
):
    from vcg_biomarker.gui.widgets import mimic_project_widget as widget_module

    payload = {
        "research_findings": [
            {
                "endpoint_id": "echo_lvh_strict",
                "endpoint_name_de": "Echo-LVH (strikt)",
            },
        ],
        "score_releases": [
            {
                "endpoint_id": "echo_lvh_strict",
                "release_id": "lvh-score-v1",
                "stage": "development",
                "research_score": 1.23,
                "research_pattern_status": (
                    "above_train_95_specificity_threshold"
                ),
                "evidence_status": "development_validation_stable",
                "primary_operating_point": {"distance_to_threshold": 0.42},
                "validation_metrics": {
                    "roc_auc": 0.81,
                    "roc_auc_ci_low": 0.76,
                    "roc_auc_ci_high": 0.86,
                    "sensitivity_at_95_specificity": 0.31,
                },
                "classical_comparators": [
                    {
                        "display_name": "Cornell-Kriterium",
                        "criterion_id": "cornell_lvh",
                        "evaluable": True,
                        "positive": False,
                        "value": 1.9,
                    },
                ],
            },
        ],
    }
    lookup = {
        ("echo_lvh_strict", "classical_cornell_lvh"): {
            "comparison_type": "paired_continuous_marker_discrimination",
            "comparison_verdict_de": "neuer Marker signifikant überlegen",
            "delta_auc": 0.09,
            "delta_auc_ci_low": 0.03,
            "delta_auc_ci_high": 0.15,
        }
    }
    rows = widget_module._build_guideline_comparison_rows(
        payload,
        comparison_lookup=lookup,
    )
    assert len(rows) == 1
    assert rows[0][11] == (
        "neuer Marker signifikant überlegen (ΔAUC 0.09 [0.03; 0.15])"
    )


def test_deep_search_status_line_reports_position_depth_and_estimate() -> None:
    """The open-ended stage must be observable while it runs."""

    from vcg_biomarker.mimic.combination_stage import describe_live_state

    text = describe_live_state(
        {
            "endpoint_id": "reduced_lvef_strict",
            "endpoint_position": 7,
            "endpoints_total": 63,
            "current_depth": 5,
            "estimated_next_level_seconds": 42.0,
        }
    )
    # Everything the run needs to be steerable from a glance.
    assert "7/63" in text
    assert "Tiefe 5" in text
    assert "42" in text
    assert "reduced_lvef_strict" in text
    assert "kein Live-Status" in describe_live_state(None)


def test_the_widget_binds_only_to_current_run_deep_search(tmp_path) -> None:
    """A newer historical search must never receive current GUI controls.

    Written against the method rather than a live widget: booting Qt and the
    real project configuration would make this test depend on the machine
    rather than on the logic being checked.
    """

    from types import SimpleNamespace

    from vcg_biomarker.gui.widgets.mimic_project_widget import MimicProjectWidget
    from vcg_biomarker.mimic.combination_stage import LIVE_FILE_NAME

    current_run_id = "current-run"
    stub = SimpleNamespace(
        config=SimpleNamespace(runs_root=tmp_path),
        _current_run_id=lambda: current_run_id,
    )
    assert MimicProjectWidget._deep_search_live(stub) == (None, None)

    for name, depth, age in (("old-run", 99, 0.0), (current_run_id, 9, 100.0)):
        root = tmp_path / name / "deep_combinations"
        root.mkdir(parents=True)
        (root / LIVE_FILE_NAME).write_text(
            json.dumps({"run_id": name, "current_depth": depth}),
            encoding="utf-8",
        )
        if age:
            stamp = (root / LIVE_FILE_NAME).stat().st_mtime - age
            os.utime(root / LIVE_FILE_NAME, (stamp, stamp))

    root, state = MimicProjectWidget._deep_search_live(stub)
    assert state["run_id"] == current_run_id
    assert state["current_depth"] == 9
    assert root.parent.name == current_run_id


def test_deep_search_rejects_mismatched_internal_run_id(tmp_path) -> None:
    from types import SimpleNamespace

    from vcg_biomarker.gui.widgets.mimic_project_widget import MimicProjectWidget
    from vcg_biomarker.mimic.combination_stage import LIVE_FILE_NAME

    current_run_id = "current-run"
    root = tmp_path / current_run_id / "deep_combinations"
    root.mkdir(parents=True)
    (root / LIVE_FILE_NAME).write_text(
        json.dumps({"run_id": "historical-run", "current_depth": 17}),
        encoding="utf-8",
    )
    stub = SimpleNamespace(
        config=SimpleNamespace(runs_root=tmp_path),
        _current_run_id=lambda: current_run_id,
    )
    assert MimicProjectWidget._deep_search_live(stub) == (None, None)


def test_a_corrupt_live_file_is_ignored_rather_than_shown(tmp_path) -> None:
    from types import SimpleNamespace

    from vcg_biomarker.gui.widgets.mimic_project_widget import MimicProjectWidget
    from vcg_biomarker.mimic.combination_stage import LIVE_FILE_NAME

    current_run_id = "current-run"
    root = tmp_path / current_run_id / "deep_combinations"
    root.mkdir(parents=True)
    (root / LIVE_FILE_NAME).write_text("{not json", encoding="utf-8")
    stub = SimpleNamespace(
        config=SimpleNamespace(runs_root=tmp_path),
        _current_run_id=lambda: current_run_id,
    )
    assert MimicProjectWidget._deep_search_live(stub) == (None, None)


def test_asking_a_stopping_job_to_stop_again_is_not_an_error(tmp_path) -> None:
    """The GUI polls on an interval, so a stale button is ordinary.

    Between the poll that enabled the pause button and the click that lands,
    the job can already have moved on -- a stage finishing is enough, and a
    second click while the worker picks the request up certainly is. Answering
    that with an exception made a no-op look like a failure, which is what the
    user saw when pausing the run.
    """

    from vcg_biomarker.mimic.jobs import JobStore
    from vcg_biomarker.mimic.models import JobState

    project = tmp_path / "project.json"
    project.write_text("{}", encoding="utf-8")
    store = JobStore(tmp_path / "jobs.sqlite")
    record = store.create(
        stage="inventory",
        project_file=project,
        arguments={},
        requested_by="test",
    )
    job_id = record.spec.job_id

    # Queued pauses immediately.
    assert store.request_pause(job_id).state == JobState.PAUSED
    # Asking again changes nothing and raises nothing.
    assert store.request_pause(job_id).state == JobState.PAUSED

    # Resuming something already on its way back is equally harmless. The
    # guard sits before the project rebind, so it answers without doing that
    # work for a job that never left.
    store.transition(job_id, JobState.RESUMING, message="resuming")
    assert store.request_resume(job_id).state == JobState.RESUMING


def test_a_job_that_cannot_pause_says_so_in_the_users_language(tmp_path) -> None:
    from vcg_biomarker.mimic.jobs import JobStore
    from vcg_biomarker.mimic.models import JobState

    project = tmp_path / "project.json"
    project.write_text("{}", encoding="utf-8")
    store = JobStore(tmp_path / "jobs.sqlite")
    record = store.create(
        stage="inventory",
        project_file=project,
        arguments={},
        requested_by="test",
    )
    store.transition(record.spec.job_id, JobState.RUNNING, message="running")
    store.transition(record.spec.job_id, JobState.COMPLETED, message="done")
    with pytest.raises(ValueError, match="Nur laufende Aufgaben"):
        store.request_pause(record.spec.job_id)
