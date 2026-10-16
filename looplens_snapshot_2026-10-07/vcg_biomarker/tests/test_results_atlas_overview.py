from __future__ import annotations

import pandas as pd
import numpy as np
import json
from types import SimpleNamespace

from vcg_biomarker.gui.results_atlas.charts import auc_matrix_rgba
from vcg_biomarker.gui.results_atlas.availability_view import (
    AtlasAvailabilityWidget, AvailabilitySnapshot, load_availability,
)
from vcg_biomarker.gui.results_atlas.export import save_plot, save_table
from vcg_biomarker.gui.results_atlas.discovery_view import (
    AtlasDiscoveryWidget, DiscoverySnapshot, hypothesis_scatter_frame,
    load_discovery_snapshot,
)
from vcg_biomarker.gui.results_atlas.marker_view import AtlasMarkerWidget, marker_rows
from vcg_biomarker.gui.results_atlas.marker_context import (
    AtlasMarkerContextWidget, MarkerContext, load_marker_context,
)
from vcg_biomarker.gui.results_atlas.model_quality import (
    AtlasModelQualityWidget, ModelQualitySnapshot, load_model_quality,
)
from vcg_biomarker.gui.results_atlas.paired_view import (
    AtlasPairedWidget, load_paired_snapshot, paired_effect, paired_summary,
    paired_outcome_summary,
)
from vcg_biomarker.gui.results_atlas.rule_audit import (
    AtlasRuleAuditWidget, load_rule_audit,
)
from vcg_biomarker.gui.results_atlas.broad_axis_view import (
    AtlasBroadAxisWidget, load_broad_axis_snapshot,
)
from vcg_biomarker.mimic.broad_axis_stage import BROAD_AXIS_STAGE_VERSION
from vcg_biomarker.mimic.control_panel import CONTROL_PANEL_VERSION
from vcg_biomarker.mimic.reference_ranges import REFERENCE_RANGE_STAGE_VERSION
from vcg_biomarker.mimic.repeatability_stage import REPEATABILITY_STAGE_VERSION
from vcg_biomarker.mimic.transform_agreement import TRANSFORM_AGREEMENT_VERSION
from vcg_biomarker.gui.results_atlas.transport_view import AtlasTransportWidget, load_transport_snapshot
from vcg_biomarker.gui.results_atlas.prognosis_view import AtlasPrognosisWidget, load_prognosis
from vcg_biomarker.gui.results_atlas.response_view import load_response
from vcg_biomarker.gui.results_atlas.curve_view import load_marker_curves
from vcg_biomarker.gui.results_atlas.score_view import load_score_snapshot
from vcg_biomarker.mimic.score_research import VERSION as SCORE_VERSION
from vcg_biomarker.mimic.survival import SURVIVAL_VERSION
from vcg_biomarker.mimic.therapy_response import THERAPY_RESPONSE_VERSION
from vcg_biomarker.mimic.segment_availability import SEGMENT_AVAILABILITY_VERSION
from vcg_biomarker.gui.results_atlas.map_view import (
    AtlasMapWidget, load_baseline_frame, load_map_frame,
)
from vcg_biomarker.gui.results_atlas.overview import (
    ResultsAtlasWidget, _atlas_state_text, _limits_text, _paired_comparison_text,
    cards_from_page, endpoint_evidence_frame, load_atlas_page, load_atlas_scope,
    load_atlas_quality, marker_gate_rows,
    load_gate_patterns,
)
from vcg_biomarker.gui.results_atlas.row_scope import validation_output_assurance
from vcg_biomarker.mimic.result_query import ResultPage
from vcg_biomarker.mimic.storage import StageManifest, capture_output_integrity, content_hash


def test_marker_gate_rows_fail_closed() -> None:
    assert "kein Gate" in marker_gate_rows(pd.Series({"failed_gates": None}))[0][1]
    rows = marker_gate_rows(pd.Series({"failed_gates": "adjustment|calibration"}))
    assert len(rows) == 2
    assert all(status == "Nicht belegt" for _, status in rows)
    assert {label for label, _ in rows} == {"Adjustierte Bestätigung", "Kalibration"}


def test_marker_dossier_plots_only_real_ci_and_sorts_auc_numerically(monkeypatch) -> None:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    assert app is not None
    widget = AtlasMarkerWidget()
    try:
        widget._request = ("run", "area")
        page = ResultPage(
            run_id="run", perspective="biomarkers", artifact_path=None,
            rows=pd.DataFrame([
                {"endpoint_id": "low", "endpoint_name_de": "Niedrig",
                 "feature_id": "area", "roc_auc": .21, "roc_auc_ci_low": .10,
                 "roc_auc_ci_high": .30, "n_positive": 30, "n_negative": 100},
                {"endpoint_id": "high", "endpoint_name_de": "Hoch",
                 "feature_id": "area", "roc_auc": .91, "roc_auc_ci_low": np.nan,
                 "roc_auc_ci_high": np.nan, "n_positive": 30, "n_negative": 100},
            ]), total_count=2, offset=0, limit=200, snapshot_hash="synthetic",
            analysis_completeness="complete_snapshot", status="ok",
        )
        context = MarkerContext("area", pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), ())
        widget._loaded(0, (page, context))
        assert widget.performance_display.isTabEnabled(0)
        assert "1 von 2" in widget.note.text()
        assert len(widget.plot.listDataItems()) == 2
        widget.table.sortItems(1, Qt.SortOrder.AscendingOrder)
        assert widget.table.item(0, 0).text() == "Niedrig"
        exported = []
        monkeypatch.setattr(
            "vcg_biomarker.gui.results_atlas.marker_view.choose_table_export",
            lambda _parent, frame, metadata, **_kwargs: exported.append((frame, metadata)),
        )
        widget.search.setText("Niedrig")
        widget._export_table()
        assert exported[0][0].endpoint_id.tolist() == ["low"]
        assert exported[0][1]["feature_id"] == "area"
    finally:
        widget.close()


def test_validation_output_assurance_accepts_current_manifest_without_claiming_hash(tmp_path) -> None:
    output = tmp_path / "result.json"
    output.write_bytes(b"example")
    manifest_only = StageManifest.complete(
        stage="validate:current_run", stage_version="synthetic",
        input_hash="input", config_hash="config", output_paths=(str(output),),
    )
    assert validation_output_assurance(manifest_only, output, "current_run") == "manifest_only"
    assert validation_output_assurance(manifest_only, output, "another_run") is None
    hashed = StageManifest.complete(
        stage="validate:current_run", stage_version="synthetic",
        input_hash="input", config_hash="config", output_paths=(str(output),),
        output_integrity=capture_output_integrity((output,)),
    )
    assert validation_output_assurance(hashed, output, "current_run") == "sha256"
    output.write_bytes(b"changed")
    assert validation_output_assurance(hashed, output, "current_run") is None


def test_paired_comparison_keeps_fixed_rule_and_auc_separate(tmp_path) -> None:
    artifact = tmp_path / "pathology_marker_explorer.parquet"
    pd.DataFrame({"endpoint_id": ["echo"]}).to_parquet(artifact)
    comparison = tmp_path / "vcg_vs_classical_comparison.parquet"
    pd.DataFrame([
        {"endpoint_id": "echo", "evaluation_split": "validation",
         "comparison_type": "paired_fixed_rule_operating_point",
         "vcg_feature_id": "area", "baseline_feature_id": "classical_cornell",
         "inference_status": "formal_inference", "paired_n": 150,
         "sensitivity_delta_vs_fixed_rule": .08,
         "sensitivity_delta_ci_low": .02, "sensitivity_delta_ci_high": .15,
         "baseline_validation_specificity": .94,
         "vcg_specificity_at_specificity_match": .92,
         "delta_auc": .9, "delta_auc_ci_low": .8, "delta_auc_ci_high": 1.0},
        {"endpoint_id": "echo", "evaluation_split": "validation",
         "comparison_type": "paired_continuous_marker_discrimination",
         "vcg_feature_id": "area", "baseline_feature_id": "qrs_duration",
         "inference_status": "descriptive_only", "paired_n": 130,
         "delta_auc": .04, "delta_auc_ci_low": np.nan,
         "delta_auc_ci_high": np.nan},
    ]).to_parquet(comparison)
    StageManifest.complete(
        stage="validate", stage_version="synthetic", input_hash="synthetic",
        config_hash="synthetic", output_paths=(str(artifact), str(comparison)),
        output_integrity=capture_output_integrity((artifact, comparison)),
    ).save(tmp_path / "validate.manifest.json")
    page = ResultPage(
        run_id="run", perspective="pathologies", artifact_path=artifact,
        rows=pd.DataFrame(), total_count=0, offset=0, limit=1,
        snapshot_hash="synthetic", analysis_completeness="complete_snapshot",
        status="ok",
    )
    snapshot = load_paired_snapshot(page, "echo")
    assert snapshot.status == "ok" and snapshot.total_count == 2
    assert snapshot.outcome_counts["n"].sum() == 2
    assert "1 positives Δ-CI" in paired_outcome_summary(snapshot)
    assert "1 ohne formales CI" in paired_outcome_summary(snapshot)
    assert "nicht Gleichwertigkeit" in paired_outcome_summary(snapshot)
    fixed = snapshot.rows.loc[snapshot.rows["comparison_type"].eq(
        "paired_fixed_rule_operating_point")].iloc[0]
    continuous = snapshot.rows.loc[snapshot.rows["comparison_type"].eq(
        "paired_continuous_marker_discrimination")].iloc[0]
    assert paired_effect(fixed) == ("Sensitivitäts-Δ", .08, .02, .15)
    assert paired_effect(continuous)[0] == "AUC-Δ"
    assert "2 gepaarte Vergleiche" in paired_summary(snapshot, "echo", "area")
    assert "1 mit formaler Schätzung" in paired_summary(snapshot, "echo", "area")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    assert app is not None
    widget = AtlasPairedWidget()
    try:
        widget.set_snapshot(snapshot)
        assert "Sensitivitäts-Δ" in widget.fixed.table.item(0, 3).text()
        assert "CI nicht berechnet" in widget.continuous.table.item(0, 3).text()
        assert widget.fixed.display.isTabEnabled(0)
        assert not widget.continuous.display.isTabEnabled(0)
        assert widget.continuous.display.currentIndex() == 1
        assert "Validation-Spezifitäten" in widget.fixed.note.text()
    finally:
        widget.close()
    StageManifest.complete(
        stage="validate", stage_version="synthetic", input_hash="synthetic",
        config_hash="synthetic", output_paths=(str(artifact), str(comparison)),
    ).save(tmp_path / "validate.manifest.json")
    assert "kein Ausgabe-SHA-256" in load_paired_snapshot(page, "echo").note
    StageManifest.complete(
        stage="validate", stage_version="synthetic", input_hash="synthetic",
        config_hash="synthetic", output_paths=(str(artifact), str(comparison)),
        output_integrity=capture_output_integrity((artifact, comparison)),
    ).save(tmp_path / "validate.manifest.json")
    comparison.write_bytes(b"changed")
    assert load_paired_snapshot(page, "echo").status == "missing"


def test_simple_card_pairs_are_counted_beyond_detail_row_limit(tmp_path) -> None:
    artifact = tmp_path / "pathology_marker_explorer.parquet"
    comparison = tmp_path / "vcg_vs_classical_comparison.parquet"
    pd.DataFrame({"endpoint_id": ["echo"]}).to_parquet(artifact)
    rows = [
        {"endpoint_id": "echo", "evaluation_split": "validation",
         "comparison_type": "paired_continuous_marker_discrimination",
         "vcg_feature_id": f"other_{index}", "baseline_feature_id": f"b{index:03d}",
         "inference_status": "descriptive_only", "paired_n": 80,
         "delta_auc": .01, "delta_auc_ci_low": np.nan, "delta_auc_ci_high": np.nan,
         "sensitivity_delta_vs_fixed_rule": np.nan,
         "sensitivity_delta_ci_low": np.nan, "sensitivity_delta_ci_high": np.nan}
        for index in range(105)
    ]
    rows.append({**rows[0], "vcg_feature_id": "special", "baseline_feature_id": "zzz",
                 "inference_status": "formal_inference", "delta_auc": .08,
                 "delta_auc_ci_low": .02, "delta_auc_ci_high": .14})
    pd.DataFrame(rows).to_parquet(comparison)
    StageManifest.complete(
        stage="validate", stage_version="synthetic", input_hash="synthetic",
        config_hash="synthetic", output_paths=(str(artifact), str(comparison)),
        output_integrity=capture_output_integrity((artifact, comparison)),
    ).save(tmp_path / "validate.manifest.json")
    page = ResultPage(
        run_id="run", perspective="pathologies", artifact_path=artifact,
        rows=pd.DataFrame(), total_count=0, offset=0, limit=1,
        snapshot_hash="synthetic", analysis_completeness="complete_snapshot", status="ok",
    )
    snapshot = load_paired_snapshot(page, "", focus_pairs=(("echo", "special"),))
    assert snapshot.total_count == 106
    assert len(snapshot.rows) == 100
    assert not snapshot.rows["vcg_feature_id"].eq("special").any()
    assert snapshot.focus_counts[("echo", "special")] == (1, 1)
    assert "1 mit formaler Schätzung" in paired_summary(snapshot, "echo", "special")


def test_rule_audit_separates_clinical_rule_and_device_report(tmp_path) -> None:
    artifact = tmp_path / "pathology_marker_explorer.parquet"
    classical = tmp_path / "classical_baseline_results.parquet"
    machine = tmp_path / "machine_report_baseline_results.parquet"
    registry = tmp_path / "baseline_registry.parquet"
    pd.DataFrame({"endpoint_id": ["echo"]}).to_parquet(artifact)
    pd.DataFrame([{
        "endpoint_id": "echo", "baseline_id": "classical_cornell",
        "evaluation_split": "validation", "n_positive": 40, "n_negative": 100,
        "sensitivity": .7, "sensitivity_ci_low": .6, "sensitivity_ci_high": .8,
        "specificity": .9, "specificity_ci_low": .85, "specificity_ci_high": .95,
        "inference_status": "formal_inference",
    }]).to_parquet(classical)
    pd.DataFrame([{
        "endpoint_id": "echo", "baseline_id": "machine_report_lvh",
        "evaluation_split": "validation", "n_positive": 38, "n_negative": 90,
        "sensitivity": .5, "specificity": .8,
        "inference_status": "descriptive_only", "uncertain_count": 4,
        "not_mentioned_count": 22, "unavailable_count": 7,
    }]).to_parquet(machine)
    pd.DataFrame([{
        "baseline_id": "classical_cornell", "display_name": "Cornell-Spannung",
        "source_title": "Leitlinienquelle", "source_url": "https://example.org/rule",
        "source_version": "2009", "formula": "R(aVL) + S(V3)",
        "exclusions": ["Schrittmacher"],
    }]).to_parquet(registry)
    StageManifest.complete(
        stage="validate", stage_version="synthetic", input_hash="synthetic",
        config_hash="synthetic", output_paths=tuple(map(str, (artifact, classical, machine, registry))),
        output_integrity=capture_output_integrity((artifact, classical, machine, registry)),
    ).save(tmp_path / "validate.manifest.json")
    controls_root = tmp_path / "controls"
    controls_root.mkdir()
    control_path = controls_root / "control_panel.parquet"
    verdict_path = controls_root / "control_panel_verdict.json"
    pd.DataFrame([{
        "control_id": "hyperkalemia_peaked_t", "kind": "positive",
        "evaluation_split": "validation", "status": "recovered", "found": True,
        "expected_direction": 1, "observed_effect": .71, "target_kind": "binary",
        "mechanism": "synthetische bekannte Richtung",
    }]).to_parquet(control_path)
    verdict_path.write_text(json.dumps({
        "run_id": "run", "verdict_split": "validation",
        "n_positive_recovered": 1, "n_positive_tested": 1,
        "n_negative_false_positive": 0, "n_negative_tested": 0,
    }), encoding="utf-8")
    StageManifest.complete(
        stage="control-panel", stage_version=CONTROL_PANEL_VERSION,
        input_hash="synthetic", config_hash="synthetic",
        output_paths=(str(control_path), str(verdict_path)),
    ).save(controls_root / "control_panel.manifest.json")
    page = ResultPage(
        run_id="run", perspective="pathologies", artifact_path=artifact,
        rows=pd.DataFrame(), total_count=0, offset=0, limit=1,
        snapshot_hash="synthetic", analysis_completeness="complete_snapshot",
        status="ok",
    )
    snapshot = load_rule_audit(page)
    assert snapshot.status == "ok"
    assert len(snapshot.classical) == len(snapshot.machine) == 1
    assert len(snapshot.controls) == 1
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    assert app is not None
    widget = AtlasRuleAuditWidget()
    try:
        widget.set_snapshot(snapshot)
        assert widget.rules_table.item(0, 0).text() == "Cornell-Spannung"
        assert "Leitlinienquelle" in widget.rules_table.item(0, 0).toolTip()
        assert [widget.rule_display.tabText(index) for index in range(2)] == ["Diagramm", "Zahlen"]
        widget.rule_search.setText("nicht vorhanden")
        assert widget.rules_table.rowCount() == 0
        widget.rule_search.setText("cornell")
        assert widget.rules_table.rowCount() == 1
        assert "0.700 (0.600–0.800)" == widget.rules_table.item(0, 1).text()
        assert widget.machine_table.item(0, 4).text() == "4"
        assert widget.machine_table.item(0, 5).text() == "22"
        assert "Hyperkaliämie" in widget.control_table.item(0, 0).text()
        assert "1 von 1" in widget.control_note.text()
        assert "kein unabhängiger Goldstandard" in widget.machine_note.text()
        assert widget.definitions_table.rowCount() == 1
        assert widget.definitions_table.item(0, 1).text() == "R(aVL) + S(V3)"
        assert "keine vergleichbare Referenzzahl" in widget.definitions_table.item(0, 3).text()
        assert "2009" in widget.definitions_table.item(0, 2).toolTip()
    finally:
        widget.close()
    StageManifest.complete(
        stage="validate", stage_version="synthetic", input_hash="synthetic",
        config_hash="synthetic",
        output_paths=tuple(map(str, (artifact, classical, machine, registry))),
    ).save(tmp_path / "validate.manifest.json")
    assert "Ausgabe-SHA-256 fehlen" in load_rule_audit(page).note
    StageManifest.complete(
        stage="validate", stage_version="synthetic", input_hash="synthetic",
        config_hash="synthetic",
        output_paths=tuple(map(str, (artifact, classical, machine, registry))),
        output_integrity=capture_output_integrity((artifact, classical, machine, registry)),
    ).save(tmp_path / "validate.manifest.json")
    machine.write_bytes(b"changed")
    assert load_rule_audit(page).status == "partial"
    verdict_path.write_text(json.dumps({
        "run_id": "different-run", "verdict_split": "validation",
    }), encoding="utf-8")
    assert load_rule_audit(page).controls.empty


def test_discovery_effect_scales_and_pending_reset() -> None:
    rows = pd.DataFrame([
        {"discovery_class": "pathology_hypothesis", "target_kind": "retrospective_proxy",
         "target_id": "echo", "feature_id": "area", "association_effect": .4,
         "validation_q_value": .001, "adjustment_robust": True},
        {"discovery_class": "non_pathology_association", "target_kind": "binary",
         "target_id": "drug", "feature_id": "area", "association_effect": .7,
         "validation_q_value": .01, "adjustment_robust": False},
        {"discovery_class": "non_pathology_association", "target_kind": "continuous",
         "target_id": "potassium", "feature_id": "area", "association_effect": -.3,
         "validation_q_value": .02, "adjustment_robust": False},
        {"discovery_class": "future_outcome_hypothesis", "target_kind": "event",
         "target_id": "death", "feature_id": "area", "association_effect": .5,
         "validation_q_value": .0001, "adjustment_robust": False},
    ])
    binary = hypothesis_scatter_frame(rows, "context_binary")
    assert len(binary) == 1 and np.isclose(binary.iloc[0]["effect_x"], .2)
    assert np.isclose(hypothesis_scatter_frame(rows, "context_continuous").iloc[0]["effect_x"], -.3)
    assert np.isclose(hypothesis_scatter_frame(rows, "future_outcome").iloc[0]["minus_log10_q_capped"], 4.)
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    assert app is not None
    widget = AtlasDiscoveryWidget()
    try:
        widget.set_snapshot(DiscoverySnapshot("ok", "", rows, 4, pd.DataFrame(), "run", "hash"))
        assert widget.group.count() == 4
        assert not widget._hypothesis_points.empty
        widget.set_snapshot(DiscoverySnapshot("pending", "wartet", pd.DataFrame(), 0,
                                              pd.DataFrame(), "run", "hash"))
        assert widget._hypothesis_points.empty
        assert widget.hypothesis_plot.isHidden()
    finally:
        widget.close()


def test_full_broad_axis_density_uses_only_verified_validation_family(tmp_path) -> None:
    runs_root = tmp_path / "runs"
    root = runs_root / "R1" / "broad_axes"
    root.mkdir(parents=True)
    screening = root / "broad_axis_screening.parquet"
    summary = root / "broad_axis_summary.json"
    rows = [
        {"axis": "laboratory", "temporal_relation": "before_ecg",
         "target_kind": "continuous", "evaluation_split": "validation",
         "effect": -.3, "q_value": .02, "p_value": .005,
         "target_id": "K", "feature_id": "area"},
        {"axis": "medication", "temporal_relation": "after_ecg",
         "target_kind": "binary", "evaluation_split": "validation",
         "effect": .8, "q_value": .01, "p_value": .002,
         "target_id": "drug", "feature_id": "area"},
        {"axis": "medication", "temporal_relation": "after_ecg",
         "target_kind": "binary", "evaluation_split": "train",
         "effect": .9, "q_value": .001, "p_value": 1e-10,
         "target_id": "drug", "feature_id": "area"},
    ]
    rows.extend({**rows[0], "p_value": .01 + index / 1000,
                 "feature_id": f"area_{index}"} for index in range(30))
    pd.DataFrame(rows).to_parquet(screening)
    summary.write_text(json.dumps({
        "run_id": "R1", "locked_test_read": False,
        "fdr_family_definition": "per_axis_not_pooled_across_axes",
    }), encoding="utf-8")
    StageManifest.complete(
        stage="screen-broad-axes", stage_version=BROAD_AXIS_STAGE_VERSION,
        input_hash="input", config_hash="config",
        output_paths=(str(screening), str(summary)),
        output_integrity=capture_output_integrity((screening, summary)),
    ).save(root / "broad_axes.manifest.json")
    repository = SimpleNamespace(config=SimpleNamespace(runs_root=runs_root))
    snapshot = load_broad_axis_snapshot(repository, "R1")
    assert snapshot.status == "ok"
    assert snapshot.families["tested_rows"].sum() == 32
    assert snapshot.bins["n"].sum() == 32
    assert set(snapshot.qq["axis"]) == {"laboratory"}
    assert snapshot.qq["n_tests"].eq(31).all()
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    assert app is not None
    widget = AtlasBroadAxisWidget()
    try:
        widget.set_snapshot(snapshot)
        assert widget.family.count() == 2
        assert not widget._selected_bins.empty
        assert not widget._selected_qq.empty
        assert "keine Patienten" in widget.detail.text()
        assert "über binäre und kontinuierliche Ziele gemeinsam" in widget.detail.text()
        assert "q = 0,05" in widget.detail.text()
    finally:
        widget.close()
    screening.write_bytes(b"changed")
    assert load_broad_axis_snapshot(repository, "R1").status == "incompatible"


def test_marker_context_labels_hospital_ranges_and_transform_limits(tmp_path) -> None:
    artifact = tmp_path / "pathology_marker_explorer.parquet"
    pd.DataFrame({"feature_id": ["area"]}).to_parquet(artifact)
    ranges_root = tmp_path / "reference_ranges"
    transform_root = tmp_path / "guideline"
    repeat_root = tmp_path / "repeatability"
    for root in (ranges_root, transform_root, repeat_root):
        root.mkdir()
    ranges = ranges_root / "marker_reference_ranges.parquet"
    transforms = transform_root / "transform_agreement.parquet"
    repeat = repeat_root / "marker_repeatability.parquet"
    pd.DataFrame([{
        "marker_id": "area", "age_band": "50–64", "sex_band": "female",
        "rate_band": "60–99", "subject_count": 300, "observed_count": 280,
        "estimable": True, "percentile_2_5": 12., "percentile_50": 42.,
        "percentile_97_5": 93.,
    }]).to_parquet(ranges)
    pd.DataFrame([{
        "marker_id": "area", "primary_transform": "kors",
        "comparison_transform": "inverse_dower", "paired_count": 300,
        "bias": 5., "bias_ci_low": 3., "bias_ci_high": 7.,
        "limit_of_agreement_low": -11., "limit_of_agreement_high": 21.,
        "spearman_rho": .8, "agreement_claim_scope": "descriptive_only",
    }]).to_parquet(transforms)
    pd.DataFrame([{
        "marker_id": "area", "status": "evaluable", "n_pairs": 150,
        "icc": .83, "repeatability_coefficient": 6.,
        "smallest_detectable_change": 8.,
        "estimand": "upper_bound_on_measurement_error",
    }]).to_parquet(repeat)
    for root, name, stage, version, path, hashed in (
        (ranges_root, "reference_ranges", "reference-ranges", REFERENCE_RANGE_STAGE_VERSION, ranges, False),
        (transform_root, "transform_agreement", "transform-agreement", TRANSFORM_AGREEMENT_VERSION, transforms, False),
        (repeat_root, "repeatability", "repeatability", REPEATABILITY_STAGE_VERSION, repeat, True),
    ):
        StageManifest.complete(
            stage=stage, stage_version=version, input_hash="synthetic",
            config_hash="synthetic", output_paths=(str(path),),
            output_integrity=capture_output_integrity((path,)) if hashed else (),
        ).save(root / f"{name}.manifest.json")
    page = ResultPage(
        run_id="run", perspective="biomarkers", artifact_path=artifact,
        rows=pd.DataFrame(), total_count=0, offset=0, limit=1,
        snapshot_hash="synthetic", analysis_completeness="complete_snapshot",
        status="ok",
    )
    context = load_marker_context(page, "area")
    assert (len(context.ranges), len(context.transforms), len(context.repeatability)) == (1, 1, 1)
    assert "ohne Dateihash" in context.source_notes[0]
    assert "hashgeprüft" in context.source_notes[2]
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    assert app is not None
    widget = AtlasMarkerContextWidget()
    try:
        widget.set_context(context)
        assert "keine gesunde" in widget.ranges_note.text()
        assert widget.ranges_table.item(0, 5).text() == "42.00"
        assert "95-%-Differenzbereich" in widget.transforms_note.text()
        assert widget.repeat_table.item(0, 2).text() == "0.83"
    finally:
        widget.close()
    repeat.write_bytes(b"changed")
    assert load_marker_context(page, "area").repeatability.empty


def _page(rows: list[dict], *, status: str = "ok") -> ResultPage:
    return ResultPage(
        run_id="development-test",
        perspective="pathologies",
        artifact_path=None,
        rows=pd.DataFrame(rows),
        total_count=len(rows),
        offset=0,
        limit=200,
        snapshot_hash="synthetic",
        analysis_completeness="complete_snapshot",
        status=status,
        message_de="Keine Daten" if status != "ok" else "",
    )


def test_prognosis_requires_verified_completed_stage(tmp_path) -> None:
    artifact = tmp_path / "pathology_marker_explorer.parquet"
    pd.DataFrame({"endpoint_id": ["echo"]}).to_parquet(artifact)
    root = tmp_path / "survival"
    root.mkdir()
    fits = root / "time_to_event.parquet"
    curves = root / "time_to_event_curves.parquet"
    pd.DataFrame([{
        "feature_id": "area", "population": "first_ecg_per_patient",
        "status": "estimable", "n_patients": 500, "n_events": 42,
        "hazard_ratio_per_sd": 1.4, "hr_lower": 1.1, "hr_upper": 1.8,
        "q_value_global": .02,
    }]).to_parquet(fits)
    pd.DataFrame([{
        "feature_id": "area", "group": 1, "group_label": "Terzil 1 von 3",
        "n": 170, "n_events": 8, "time_days": float(day),
        "cumulative_incidence": .05, "lower_bound": .02, "upper_bound": .08,
    } for day in range(250)]).to_parquet(curves)
    page = ResultPage(
        run_id="test", perspective="pathologies", artifact_path=artifact,
        rows=pd.DataFrame(), total_count=0, offset=0, limit=1,
        snapshot_hash="synthetic", analysis_completeness="complete_snapshot", status="ok",
    )
    assert load_prognosis(page).status == "pending"
    StageManifest.complete(
        stage="time-to-event", stage_version=SURVIVAL_VERSION,
        input_hash="synthetic", config_hash="synthetic",
        output_paths=(str(fits), str(curves)),
        output_integrity=capture_output_integrity((fits, curves)),
    ).save(root / "time_to_event.manifest.json")
    snapshot = load_prognosis(page)
    assert snapshot.status == "ok"
    assert len(snapshot.fits) == 1
    assert len(snapshot.curves) <= 121
    assert snapshot.curves.time_days.iloc[0] == 0
    assert snapshot.curves.time_days.iloc[-1] == 249
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    assert app is not None
    widget = AtlasPrognosisWidget()
    widget.set_snapshot(snapshot)
    assert widget.plot.listDataItems()
    widget.close()
    curves.write_bytes(b"changed")
    assert load_prognosis(page).status == "incompatible"


def test_response_keeps_observed_change_separate_from_causal_claim(tmp_path) -> None:
    artifact = tmp_path / "pathology_marker_explorer.parquet"
    pd.DataFrame({"endpoint_id": ["tachycardia"]}).to_parquet(artifact)
    root = tmp_path / "therapy_response"
    root.mkdir()
    result = root / "therapy_response_results.parquet"
    row = {
        "scope": "medication_class", "medication_class": "beta_blocker",
        "status": "estimable", "effect": -.1, "ci_low": -.2, "ci_high": -.02,
        "n_pairs": 60, "n_subjects": 54, "rate_before": .3, "rate_after": .2,
        "run_id": "test",
        "claim_scope": "physiological_response_association_not_causal_effect",
    }
    pd.DataFrame([row]).to_parquet(result)
    page = ResultPage(
        run_id="test", perspective="pathologies", artifact_path=artifact,
        rows=pd.DataFrame(), total_count=0, offset=0, limit=1,
        snapshot_hash="synthetic", analysis_completeness="complete_snapshot", status="ok",
    )
    assert load_response(page).status == "pending"
    StageManifest.complete(
        stage="therapy-response", stage_version=THERAPY_RESPONSE_VERSION,
        input_hash="synthetic", config_hash="synthetic",
        output_paths=(str(result),),
        output_integrity=capture_output_integrity((result,)),
    ).save(root / "therapy_response.manifest.json")
    snapshot = load_response(page)
    assert snapshot.status == "ok"
    assert "kein kausaler Nutzen" in snapshot.note
    pd.DataFrame([{**row, "claim_scope": "causal_treatment_benefit"}]).to_parquet(result)
    assert load_response(page).status == "incompatible"


def test_marker_curves_are_validation_scoped_and_hash_verified(tmp_path) -> None:
    artifact = tmp_path / "pathology_marker_explorer.parquet"
    pd.DataFrame({"endpoint_id": ["echo"]}).to_parquet(artifact)
    curves = tmp_path / "performance_curve_points.parquet"
    bins = tmp_path / "score_distribution_bins.parquet"
    operating = tmp_path / "operating_point_results.parquet"
    curve = {
        "endpoint_id": "echo", "feature_id": "area", "evaluation_split": "validation",
        "point_kind": "continuous_marker_curve", "curve_type": "roc", "point_order": 0,
        "x": 0., "y": 0., "n_positive": 20, "n_negative": 40,
        "roc_auc": .8, "roc_auc_ci_low": .7, "roc_auc_ci_high": .9,
        "average_precision": .65, "average_precision_ci_low": .5,
        "average_precision_ci_high": .75,
    }
    pd.DataFrame([curve, {**curve, "evaluation_split": "train"},
                  {**curve, "feature_id": "other"}]).to_parquet(curves)
    bin_row = {
        "endpoint_id": "echo", "feature_id": "area", "evaluation_split": "validation",
        "label": 1, "bin_left": 0., "bin_right": .25,
        "fraction_within_class": .2, "class_total": 20,
        "display_transform": "direction_aligned_train_zscore_clipped_to_5sd",
    }
    pd.DataFrame([bin_row]).to_parquet(bins)
    point = {
        "endpoint_id": "echo", "feature_id": "area", "evaluation_split": "validation",
        "operating_point": "sensitivity_at_95_specificity", "threshold_source": "train",
        "target_attainable": True, "sensitivity": .6, "specificity": .91,
        "sensitivity_ci_low": .4, "sensitivity_ci_high": .8,
        "specificity_ci_low": .84, "specificity_ci_high": .96,
        "ppv": .7, "npv": .9, "inference_status": "formal_inference",
    }
    pd.DataFrame([point, {**point, "evaluation_split": "train"}]).to_parquet(operating)
    StageManifest.complete(
        stage="validate", stage_version="synthetic", input_hash="synthetic",
        config_hash="synthetic", output_paths=(str(artifact), str(curves), str(bins), str(operating)),
        output_integrity=capture_output_integrity((artifact, curves, bins, operating)),
    ).save(tmp_path / "validate.manifest.json")
    page = ResultPage(
        run_id="test", perspective="pathologies", artifact_path=artifact,
        rows=pd.DataFrame(), total_count=0, offset=0, limit=1,
        snapshot_hash="synthetic", analysis_completeness="complete_snapshot", status="ok",
    )
    snapshot = load_marker_curves(page, "echo", "area")
    assert snapshot.status == "ok"
    assert len(snapshot.curves) == len(snapshot.distributions) == 1
    assert len(snapshot.operating_points) == 1
    assert snapshot.operating_points.iloc[0].specificity == .91
    StageManifest.complete(
        stage="validate", stage_version="synthetic", input_hash="synthetic",
        config_hash="synthetic", output_paths=(str(artifact), str(curves), str(bins), str(operating)),
    ).save(tmp_path / "validate.manifest.json")
    assert "kein Ausgabe-SHA-256" in load_marker_curves(page, "echo", "area").note
    StageManifest.complete(
        stage="validate", stage_version="synthetic", input_hash="synthetic",
        config_hash="synthetic", output_paths=(str(artifact), str(curves), str(bins), str(operating)),
        output_integrity=capture_output_integrity((artifact, curves, bins, operating)),
    ).save(tmp_path / "validate.manifest.json")
    curves.write_bytes(b"changed")
    assert load_marker_curves(page, "echo", "area").status == "incompatible"


def test_score_atlas_uses_sealed_research_report_without_clinical_promotion(tmp_path) -> None:
    inputs = {"version": SCORE_VERSION, "sampling_design": "complete_clinical_phenome_union",
              "max_patients": 50000, "config_hash": "synthetic"}
    digest = content_hash(inputs)
    root = tmp_path / "run" / "score_discovery" / digest[:20]
    root.mkdir(parents=True)
    inputs_path = root / "inputs.json"
    inputs_path.write_text(json.dumps(inputs), encoding="utf-8")
    report_path = root / "score_research_report.json"
    report = {
        "version": SCORE_VERSION, "input_hash": digest,
        "clinical_claim_allowed": False, "locked_test_read": False,
        "results": [{
            "endpoint_id": "echo", "models": {"ecg": {
                "score_spec": {"components": ["qrs_duration_ms"], "coefficients": [.2],
                               "formula": "expit(intercept + ... )"},
                "evaluation": {"metrics": {"status": "estimable", "auc": .7,
                                           "auc_ci95": [.6, .8], "brier": .2,
                                           "n_patients": 100, "n_events": 20,
                                           "n_predicted": 95, "n_eligible": 100}},
                "point_score": {"metrics": {"auc": .68, "brier": .21}},
                "folds": [{"components": ["qrs_duration_ms"]},
                          {"components": ["other"]}],
            }},
            "paired_comparisons": {"vcg_vs_ecg": {
                "status": "research_comparison", "delta_auc_second_minus_first": .04,
                "delta_auc_ci95": [-.01, .09], "n_common": 90, "n_patients": 90,
            }},
        }],
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    StageManifest.complete(
        stage="score-discovery", stage_version=SCORE_VERSION,
        input_hash=digest, config_hash="synthetic",
        output_paths=(str(report_path), str(inputs_path)),
        output_integrity=capture_output_integrity((report_path, inputs_path)),
    ).save(root / "score_discovery.manifest.json")

    class Repository:
        config = SimpleNamespace(runs_root=tmp_path)

        @staticmethod
        def available_runs():
            return [SimpleNamespace(run_id="run", complete_compatible=True)]

    snapshot = load_score_snapshot(Repository(), "run")
    assert snapshot.status == "ok"
    assert snapshot.models.iloc[0].auc == .7
    assert snapshot.models.iloc[0].point_auc == .68
    assert snapshot.models.iloc[0].component_fold_counts == {"qrs_duration_ms": 1}
    assert snapshot.models.iloc[0].max_patients == 50000
    assert snapshot.comparisons.iloc[0].delta_auc == .04
    report["clinical_claim_allowed"] = True
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert load_score_snapshot(Repository(), "run").status == "incompatible"


def test_atlas_does_not_present_same_ecg_or_classical_rule_as_new() -> None:
    headline, cards = cards_from_page(_page([
        {
            "endpoint_id": "qtc", "feature_id": "qt", "roc_auc": 1.0,
            "reference_independence": "same_ecg_technical_comparison",
            "candidate_type": "vcg_single", "selection_eligible": True,
        },
        {
            "endpoint_id": "lvh", "feature_id": "sokolow", "roc_auc": .8,
            "candidate_type": "classical_rule", "selection_eligible": True,
        },
    ]))
    assert cards == []
    assert "kein auswahlfähiger neuer Markerbefund" in headline


def test_atlas_preserves_missing_ci_and_research_claim() -> None:
    headline, cards = cards_from_page(_page([{
        "endpoint_id": "reduced_lvef", "endpoint_name_de": "Reduzierte LVEF",
        "feature_id": "area", "feature_name_de": "Räumlicher Flächenwinkel",
        "roc_auc": .79, "roc_auc_ci_low": None, "roc_auc_ci_high": None,
        "n_positive": 24, "n_negative": 150,
        "candidate_type": "vcg_single", "selection_eligible": True,
        "reference_kind": "timed_quantitative",
        "timing_evidence_note": "no_clock_alignment_confirmed",
        "clock_alignment_confirmed_n": 0,
        "claim_ceiling": "independent_reference_comparison",
        "evidence_stage": "exploratory",
    }]))
    assert "Noch kein belastbar" in headline
    assert len(cards) == 1
    assert "CI nicht berechnet" in cards[0].body
    assert "24 Fälle" in cards[0].body
    assert "Forschungshypothese" in cards[0].detail
    assert "nicht unabhängig abgeglichen" in cards[0].detail
    assert "nicht fair vergleichbar" in cards[0].comparison
    assert "Kalibration" in cards[0].open_questions
    assert "besserer klinischer Befund ist nicht belegt" in cards[0].takeaway
    assert cards[0].evidence_label == "Forschungshypothese"


def test_historical_marker_never_becomes_a_simple_mode_finding() -> None:
    _, cards = cards_from_page(_page([{
        "endpoint_id": "echo", "feature_id": "old", "roc_auc": .98,
        "selection_eligible": True, "candidate_type": "vcg_single",
        "evidence_stage": "clinically_validated",
        "historical_pre_audit_result": True,
    }]))
    assert not cards


def test_atlas_incompatible_result_has_no_cards() -> None:
    headline, cards = cards_from_page(_page([], status="incompatible_historical_run"))
    assert headline == "Keine Daten"
    assert not cards


def test_simple_result_state_distinguishes_pending_empty_and_research() -> None:
    pending = _page([], status="artifact_not_available")
    incompatible = _page([], status="incompatible_historical_run")
    empty = _page([], status="empty")
    screened = _page([{"endpoint_id": "echo", "feature_id": "area",
                       "selection_eligible": False, "roc_auc": .75}])
    candidate = _page([{"endpoint_id": "echo", "feature_id": "area",
                        "selection_eligible": True, "roc_auc": .75,
                        "evidence_stage": "exploratory"}])
    assert "Noch nicht berechnet" in _atlas_state_text(pending, [])
    assert "Nicht verwendbar" in _atlas_state_text(incompatible, [])
    assert "keine Ergebniszeilen" in _atlas_state_text(empty, [])
    assert "kein auswahlfähiger" in _atlas_state_text(screened, [])
    assert "Forschungshinweise" in _atlas_state_text(candidate, cards_from_page(candidate)[1])


def test_endpoint_evidence_omits_unmeasured_and_self_comparison() -> None:
    page = _page([
        {"endpoint_id": "e1", "feature_id": "valid", "roc_auc": .73,
         "candidate_type": "vcg_single", "selection_eligible": True},
        {"endpoint_id": "e1", "feature_id": "missing", "roc_auc": None,
         "candidate_type": "vcg_single", "selection_eligible": True},
        {"endpoint_id": "e1", "feature_id": "self", "roc_auc": 1.,
         "reference_independence": "same_ecg_technical_comparison",
         "candidate_type": "vcg_single", "selection_eligible": True},
        {"endpoint_id": "e2", "feature_id": "other", "roc_auc": .8,
         "candidate_type": "vcg_single", "selection_eligible": True},
    ])
    frame = endpoint_evidence_frame(page, "e1")
    assert frame["feature_id"].tolist() == ["valid"]


def test_atlas_queries_all_rows_but_returns_bounded_eligible_sample(tmp_path) -> None:
    artifact = tmp_path / "pathology_marker_explorer.parquet"
    rows = []
    for index in range(240):
        rows.append({
            "endpoint_id": "e1", "endpoint_name_de": "Endpunkt eins",
            "feature_id": f"self_{index}", "feature_name_de": "Technischer Vergleich",
            "evidence_stage": "exploratory", "reference_kind": "same_ecg",
            "claim_ceiling": "agreement_only",
            "reference_independence": "same_ecg_technical_comparison",
            "candidate_type": "vcg_single", "selection_eligible": True,
            "roc_auc": 1., "evaluation_split": "validation",
        })
    rows.append({
        "endpoint_id": "e2", "endpoint_name_de": "Endpunkt zwei",
        "feature_id": "new", "feature_name_de": "Neuer Marker",
        "evidence_stage": "exploratory", "reference_kind": "timed_quantitative",
        "claim_ceiling": "independent_reference_comparison",
        "reference_independence": "reference_quality_requires_review",
        "candidate_type": "vcg_single", "selection_eligible": True,
        "roc_auc": .74, "evaluation_split": "validation",
        "historical_pre_audit_result": False,
    })
    rows.append({
        "endpoint_id": "e3", "endpoint_name_de": "Historischer Treffer",
        "feature_id": "old", "feature_name_de": "Alte Formel",
        "evidence_stage": "guideline_candidate", "reference_kind": "timed_quantitative",
        "claim_ceiling": "independent_reference_comparison",
        "reference_independence": "independent",
        "candidate_type": "vcg_single", "selection_eligible": True,
        "roc_auc": .99, "evaluation_split": "validation",
        "historical_pre_audit_result": True,
    })
    pd.DataFrame(rows).to_parquet(artifact)

    class Repository:
        def query(self, _query):
            return ResultPage(
                run_id="run", perspective="pathologies", artifact_path=artifact,
                rows=pd.DataFrame(rows[:1]), total_count=len(rows), offset=0,
                limit=1, snapshot_hash="synthetic",
                analysis_completeness="complete_snapshot", status="ok",
            )

    page = load_atlas_page(Repository(), "run", "")
    assert page.total_count == 1
    assert page.rows["feature_id"].tolist() == ["new"]
    assert load_atlas_scope(page) == {
        "evaluated_endpoints": 2, "eligible_endpoints": 1, "eligible_pairs": 1,
    }


def test_atlas_overview_keeps_distinct_questions_when_one_has_many_markers(tmp_path) -> None:
    artifact = tmp_path / "pathology_marker_explorer.parquet"
    common = {
        "evidence_stage": "exploratory", "reference_kind": "timed_quantitative",
        "claim_ceiling": "independent_reference_comparison",
        "reference_independence": "independent", "candidate_type": "vcg_single",
        "selection_eligible": True, "evaluation_split": "validation",
        "historical_pre_audit_result": False,
    }
    rows = [
        {**common, "endpoint_id": "frequent", "feature_id": f"m{i:03d}",
         "roc_auc": .9 - i * .0001}
        for i in range(250)
    ] + [{**common, "endpoint_id": "other", "feature_id": "other_marker",
          "roc_auc": .7}]
    pd.DataFrame(rows).to_parquet(artifact)

    class Repository:
        def query(self, _query):
            return ResultPage(
                run_id="run", perspective="pathologies", artifact_path=artifact,
                rows=pd.DataFrame(rows[:1]), total_count=len(rows), offset=0,
                limit=1, snapshot_hash="synthetic",
                analysis_completeness="complete_snapshot", status="ok",
            )

    page = load_atlas_page(Repository(), "run", "")
    assert page.total_count == 251
    assert set(page.rows.endpoint_id) == {"frequent", "other"}
    assert len(page.rows.loc[page.rows.endpoint_id.eq("frequent")]) == 5
    detail = load_atlas_page(Repository(), "run", "frequent")
    assert detail.total_count == 250
    assert len(detail.rows) == 200
    assert detail.rows.endpoint_id.eq("frequent").all()


def test_auc_heatmap_missing_is_transparent_not_chance() -> None:
    rgba = auc_matrix_rgba(np.array([[np.nan, .5, .8]]))
    assert rgba.shape == (1, 3, 4)
    assert rgba[0, 0, 3] == 0
    assert rgba[0, 1, 3] == 255
    assert rgba[0, 2, 3] == 255
    assert not np.array_equal(rgba[0, 1, :3], rgba[0, 2, :3])


def test_map_aggregates_eligible_validation_and_keeps_domain(tmp_path) -> None:
    artifact = tmp_path / "pathology_marker_explorer.parquet"
    frame = pd.DataFrame([
        {"endpoint_id": "lvef", "endpoint_name_de": "Reduzierte LVEF",
         "pathology_group": "Herzstruktur", "conceptual_family": "QRS–T",
         "feature_id": "a", "roc_auc": .72, "selection_eligible": True,
         "candidate_type": "vcg_single", "reference_independence": "independent",
         "claim_ceiling": "independent_reference_comparison", "evaluation_split": "validation"},
        {"endpoint_id": "lvef", "endpoint_name_de": "Reduzierte LVEF",
         "pathology_group": "Herzstruktur", "conceptual_family": "QRS–T",
         "feature_id": "b", "roc_auc": .79, "selection_eligible": True,
         "candidate_type": "vcg_single", "reference_independence": "independent",
         "claim_ceiling": "independent_reference_comparison", "evaluation_split": "validation"},
        {"endpoint_id": "lvef", "endpoint_name_de": "Reduzierte LVEF",
         "pathology_group": "Herzstruktur", "conceptual_family": "QRS–T",
         "feature_id": "invalid", "roc_auc": 1.4, "selection_eligible": True,
         "candidate_type": "vcg_single", "reference_independence": "independent",
         "claim_ceiling": "independent_reference_comparison", "evaluation_split": "validation"},
        {"endpoint_id": "lvef", "endpoint_name_de": "Reduzierte LVEF",
         "pathology_group": "Herzstruktur", "conceptual_family": "QRS–T",
         "feature_id": "self", "roc_auc": 1., "selection_eligible": True,
         "candidate_type": "vcg_single",
         "reference_independence": "same_ecg_technical_comparison",
         "claim_ceiling": "agreement_only", "evaluation_split": "validation"},
        {"endpoint_id": "lvef", "endpoint_name_de": "Reduzierte LVEF",
         "pathology_group": "Herzstruktur", "conceptual_family": "QRS–T",
         "feature_id": "old", "roc_auc": .98, "selection_eligible": True,
         "candidate_type": "vcg_single", "reference_independence": "independent",
         "claim_ceiling": "independent_reference_comparison",
         "evaluation_split": "validation", "historical_pre_audit_result": True},
    ])
    frame.to_parquet(artifact)

    class Repository:
        def query(self, _query):
            return ResultPage(
                run_id="run", perspective="pathologies", artifact_path=artifact,
                rows=frame.head(1), total_count=len(frame), offset=0,
                limit=1, snapshot_hash="synthetic",
                analysis_completeness="complete_snapshot", status="ok",
            )

    result = load_map_frame(Repository(), "run")
    assert len(result) == 1
    assert result.iloc[0]["pathology_group"] == "Herzstruktur"
    assert result.iloc[0]["candidate_count"] == 2
    assert result.iloc[0]["navigation_auc"] == .79


def test_marker_map_has_scale_and_preserves_unmeasured_cells() -> None:
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    assert app is not None
    widget = AtlasMapWidget()
    try:
        widget.set_frame(pd.DataFrame([
            {"endpoint_id": "a", "endpoint_name_de": "Ziel A",
             "pathology_group": "Herzstruktur", "conceptual_family": "qrs_t_coupling",
             "navigation_auc": .75},
            {"endpoint_id": "b", "endpoint_name_de": "Ziel B",
             "pathology_group": "Herzstruktur", "conceptual_family": "atrial_activation",
             "navigation_auc": .65},
        ]))
        assert widget.legend_gradient.pixmap() is not None
        assert widget._grid.shape == (2, 2)
        assert np.isnan(widget._grid).sum() == 2
    finally:
        widget.close()


def test_gate_counts_overlap_and_do_not_form_false_funnel(tmp_path) -> None:
    artifact = tmp_path / "pathology_marker_explorer.parquet"
    pd.DataFrame([
        {"evidence_stage": "exploratory", "candidate_type": "vcg_single",
         "reference_independence": "independent", "evaluation_split": "validation",
         "failed_gates": "adjustment|external_validation"},
        {"evidence_stage": "exploratory", "candidate_type": "vcg_single",
         "reference_independence": "independent", "evaluation_split": "validation",
         "failed_gates": "adjustment"},
        {"evidence_stage": "exploratory", "candidate_type": "classical_rule",
         "reference_independence": "independent", "evaluation_split": "validation",
         "failed_gates": "adjustment"},
        {"evidence_stage": "guideline_candidate", "candidate_type": "vcg_single",
         "reference_independence": "independent", "evaluation_split": "validation",
         "failed_gates": "external_validation", "historical_pre_audit_result": True},
    ]).to_parquet(artifact)
    page = ResultPage(
        run_id="run", perspective="pathologies", artifact_path=artifact,
        rows=pd.DataFrame(), total_count=3, offset=0, limit=1,
        snapshot_hash="synthetic", analysis_completeness="complete_snapshot",
        status="ok",
    )
    stages, gates = load_atlas_quality(page)
    assert stages == {"exploratory": 2}
    assert dict(gates) == {"adjustment": 2, "external_validation": 1}
    patterns = load_gate_patterns(page)
    assert int(patterns.iloc[0].total_pairs) == 2
    assert dict(zip(patterns.pattern, patterns.n)) == {
        "adjustment": 1, "adjustment|external_validation": 1,
    }


def test_classical_rules_remain_separate_from_new_marker_map(tmp_path) -> None:
    artifact = tmp_path / "pathology_marker_explorer.parquet"
    frame = pd.DataFrame([
        {"endpoint_id": "lvef", "endpoint_name_de": "Reduzierte LVEF",
         "feature_id": "sokolow", "feature_name_de": "Sokolow-Lyon",
         "candidate_type": "classical_rule", "roc_auc": .66,
         "evaluation_split": "validation", "reference_kind": "timed_quantitative"},
        {"endpoint_id": "lvef", "endpoint_name_de": "Reduzierte LVEF",
         "feature_id": "direct_area", "feature_name_de": "Direkter Flächenwinkel",
         "candidate_type": "vcg_single", "roc_auc": .77,
         "evaluation_split": "validation", "reference_kind": "timed_quantitative"},
    ])
    frame.to_parquet(artifact)

    class Repository:
        def query(self, _query):
            return ResultPage(
                run_id="run", perspective="pathologies", artifact_path=artifact,
                rows=frame.head(1), total_count=2, offset=0, limit=1,
                snapshot_hash="synthetic", analysis_completeness="complete_snapshot",
                status="ok",
            )

    classical = load_baseline_frame(Repository(), "run")
    assert classical["feature_id"].tolist() == ["sokolow"]
    assert classical.attrs["total_count"] == 1


def test_atlas_exports_plot_and_table_with_provenance(tmp_path) -> None:
    import hashlib
    import pyqtgraph as pg
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    plot = pg.PlotWidget()
    try:
        plot.plot([.6, .8], [1, 2])
        metadata = {"run_id": "synthetic", "split": "validation"}
        png = tmp_path / "plot.png"
        svg = tmp_path / "plot.svg"
        csv = tmp_path / "values.csv"
        png_sidecar = save_plot(plot, png, metadata)
        svg_sidecar = save_plot(plot, svg, metadata)
        csv_sidecar = save_table(pd.DataFrame({"auc": [.6, .8]}), csv, metadata)
        assert png.read_bytes().startswith(b"\x89PNG")
        assert "<svg" in svg.read_text(encoding="utf-8")
        assert "auc" in csv.read_text(encoding="utf-8-sig")
        for exported, sidecar in ((png, png_sidecar), (svg, svg_sidecar),
                                  (csv, csv_sidecar)):
            provenance = json.loads(sidecar.read_text(encoding="utf-8"))
            assert provenance["run_id"] == "synthetic"
            assert provenance["_export"]["filename"] == exported.name
            assert provenance["_export"]["sha256"] == hashlib.sha256(
                exported.read_bytes()
            ).hexdigest()
    finally:
        plot.close()
        app.processEvents()


def test_broad_scan_waits_for_complete_compatible_manifest(tmp_path) -> None:
    from types import SimpleNamespace

    class Repository:
        def available_runs(self):
            return (SimpleNamespace(
                run_id="run", complete_compatible=True,
                phenome_status="running", phenome_compatible=True,
            ),)

        def query(self, _query):
            raise AssertionError("unfinished broad scan must not be queried")

    snapshot = load_discovery_snapshot(Repository(), "run")
    assert snapshot.status == "pending"
    assert snapshot.rows.empty


def test_broad_scan_domain_chart_counts_only_adjusted_replication(tmp_path) -> None:
    from types import SimpleNamespace

    artifact = tmp_path / "new_associations_explorer.parquet"
    frame = pd.DataFrame([
        {"discovery_domain": "renal", "adjustment_robust": True,
         "unadjusted_replicated": True, "target_name": "Ziel 1"},
        {"discovery_domain": "renal", "adjustment_robust": False,
         "unadjusted_replicated": True, "target_name": "Ziel 2"},
        {"discovery_domain": "pulmonary", "adjustment_robust": True,
         "unadjusted_replicated": False, "target_name": "Ziel 3"},
    ])
    frame.to_parquet(artifact)

    class Repository:
        def available_runs(self):
            return (SimpleNamespace(
                run_id="run", complete_compatible=True,
                phenome_status="complete", phenome_compatible=True,
            ),)

        def query(self, _query):
            return ResultPage(
                run_id="run", perspective="discovery", artifact_path=artifact,
                rows=frame, total_count=3, offset=0, limit=75,
                snapshot_hash="synthetic", analysis_completeness="complete_snapshot",
                status="ok",
            )

    snapshot = load_discovery_snapshot(Repository(), "run")
    assert snapshot.status == "ok"
    assert snapshot.domains[["domain", "hypothesis_count"]].values.tolist() == [["renal", 1]]


def test_marker_dossier_keeps_clinical_endpoints_separate() -> None:
    page = ResultPage(
        run_id="run", perspective="biomarkers", artifact_path=None,
        rows=pd.DataFrame([
            {"endpoint_id": "echo_lvef", "feature_id": "area", "roc_auc": .78,
             "reference_independence": "independent"},
            {"endpoint_id": "ecg_angle", "feature_id": "area", "roc_auc": 1.,
             "reference_independence": "same_ecg_technical_comparison"},
            {"endpoint_id": "potassium", "feature_id": "area", "roc_auc": None,
             "reference_independence": "independent"},
            {"endpoint_id": "historical", "feature_id": "area", "roc_auc": .99,
             "reference_independence": "independent", "historical_pre_audit_result": True},
        ]),
        total_count=4, offset=0, limit=200, snapshot_hash="synthetic",
        analysis_completeness="complete_snapshot", status="ok",
    )
    result = marker_rows(page)
    assert result["endpoint_id"].tolist() == ["echo_lvef"]


def test_simple_mode_paired_comparison_requires_baseline_and_ci() -> None:
    rows = pd.DataFrame([{
        "feature_id": "vcg_area", "feature_name_de": "Flächenwinkel",
        "baseline_feature_id": "classical_qrs_duration",
        "delta_auc": .04, "delta_auc_ci_low": -.01, "delta_auc_ci_high": .09,
    }])
    assert "nicht belegt" in _paired_comparison_text(rows)
    assert "nicht fair vergleichbar" in _paired_comparison_text(
        rows.drop(columns="delta_auc_ci_low")
    )
    rows.loc[0, "delta_auc_ci_low"] = .01
    assert "aus der Markerübersicht allein nicht belegt" in _paired_comparison_text(rows)


def test_simple_mode_missing_gates_are_not_drawn_as_success() -> None:
    rows = pd.DataFrame([{
        "failed_gates": "calibration|net_benefit|external_validation",
    }])
    text = _limits_text(rows)
    assert "Kalibration" in text and "Entscheidungsnutzen" in text
    assert "externe Bestätigung" in text
    assert "nicht beurteilbar" in _limits_text(pd.DataFrame())


def test_model_quality_uses_only_manifested_validation_endpoint(tmp_path) -> None:
    artifact = tmp_path / "pathology_marker_explorer.parquet"
    pd.DataFrame({"endpoint_id": ["echo"]}).to_parquet(artifact)
    calibration = tmp_path / "calibration_curve_points.parquet"
    pd.DataFrame([
        {"endpoint_id": endpoint, "model_id": "m", "evaluation_split": split,
         "mean_predicted": .3, "observed_rate": .2,
         "observed_ci_low": .1, "observed_ci_high": .4, "n": 30, "events": 6}
        for endpoint, split in (("echo", "validation"), ("echo", "train"),
                                ("potassium", "validation"))
    ]).to_parquet(calibration)
    decision = tmp_path / "decision_curve_results.parquet"
    pd.DataFrame([{
        "endpoint_id": "echo", "model_id": "m", "evaluation_split": "validation",
        "threshold_probability": .2, "net_benefit": .04,
        "treat_all_net_benefit": .01, "treat_none_net_benefit": 0.,
        "evidence_scope": "internal_development_descriptive",
        "clinical_utility_evidence_eligible": False,
    }]).to_parquet(decision)
    metrics = tmp_path / "adjusted_model_results.parquet"
    pd.DataFrame([{
        "endpoint_id": endpoint, "model_id": "clinical_ecg_vcg",
        "evaluation_split": split, "n_subjects": 120,
        "roc_auc": .78, "brier_score": .16,
        "calibration_intercept": -.05, "calibration_slope": .92,
        "inference_status": "formal_inference",
    } for endpoint, split in (("echo", "validation"), ("echo", "train"),
                              ("potassium", "validation"))]).to_parquet(metrics)
    StageManifest.complete(
        stage="validate", stage_version="synthetic", input_hash="synthetic",
        config_hash="synthetic", output_paths=(str(artifact), str(calibration), str(metrics)),
    ).save(tmp_path / "validate.manifest.json")
    page = ResultPage(
        run_id="run", perspective="pathologies", artifact_path=artifact,
        rows=pd.DataFrame(), total_count=0, offset=0, limit=1,
        snapshot_hash="synthetic", analysis_completeness="complete_snapshot",
        status="ok",
    )
    snapshot = load_model_quality(page, "echo")
    assert snapshot.status == "ok"
    assert len(snapshot.calibration) == 1
    assert len(snapshot.metrics) == 1
    assert snapshot.metrics.iloc[0].brier_score == .16
    assert "Ausgabe-SHA-256 fehlt" in snapshot.message
    assert snapshot.decision.empty  # A stray, unmanifested curve is not evidence.


def test_availability_counts_and_subgroups_stay_endpoint_marker_scoped(tmp_path) -> None:
    artifact = tmp_path / "pathology_marker_explorer.parquet"
    pd.DataFrame({"endpoint_id": ["echo"]}).to_parquet(artifact)
    screening = tmp_path / "screening_results.parquet"
    pd.DataFrame([
        {"endpoint_id": endpoint, "feature_id": marker,
         "analysis_type": "subgroup_stability", "evaluation_split": split,
         "subgroup_variable": "age", "subgroup_level": "<65",
         "subgroup_time_scope": "available_at_ecg", "roc_auc": .7,
         "n_positive": 10, "n_negative": 30, "inference_status": "descriptive_only"}
        for endpoint, marker, split in (
            ("echo", "area", "validation"), ("echo", "area", "train"),
            ("echo", "other", "validation"), ("potassium", "area", "validation"),
        )
    ]).to_parquet(screening)
    StageManifest.complete(
        stage="validate", stage_version="synthetic", input_hash="synthetic",
        config_hash="synthetic", output_paths=(str(artifact), str(screening)),
        output_integrity=capture_output_integrity((artifact, screening)),
    ).save(tmp_path / "validate.manifest.json")
    page = ResultPage(
        run_id="run", perspective="pathologies", artifact_path=artifact,
        rows=pd.DataFrame([{
            "endpoint_id": "echo", "feature_id": "area",
            "n_reference_eligible": 100, "n_marker_raw_observed": 80,
            "n_marker_observed": 70, "n_analysis_restriction_excluded": 10,
        }]), total_count=1, offset=0, limit=200, snapshot_hash="synthetic",
        analysis_completeness="complete_snapshot", status="ok",
    )
    snapshot = load_availability(page, "echo")
    assert (snapshot.reference_eligible, snapshot.raw_observed,
            snapshot.analysis_observed) == (100, 80, 70)
    assert len(snapshot.subgroups) == 1
    assert snapshot.missing_reasons is not None
    assert snapshot.missing_reasons["count"].tolist() == [70, 10, 20]
    assert snapshot.missing_reasons["share_of_eligible_pct"].tolist() == [70.0, 10.0, 20.0]
    assert snapshot.subgroups["roc_auc_ci_low"].isna().all()
    other = load_availability(page, "echo", "missing-marker")
    assert other.marker_id == "" and other.reference_eligible is None
    page.rows.loc[0, "n_marker_observed"] = 90
    invalid = load_availability(page, "echo")
    assert invalid.reference_eligible is None
    assert invalid.missing_reasons is not None and invalid.missing_reasons.empty


def test_segment_availability_is_endpoint_scoped_and_not_called_validation_only(tmp_path) -> None:
    artifact = tmp_path / "pathology_marker_explorer.parquet"
    pd.DataFrame({"endpoint_id": ["echo"]}).to_parquet(artifact)
    root = tmp_path / "availability"
    root.mkdir()
    table = root / "segment_availability.parquet"
    row = {
        "endpoint_id": "echo", "segment": "p", "sentinel_marker": "p_area",
        "n_positive": 120, "n_negative": 300,
        "available_positive_pct": 40., "available_negative_pct": 70.,
        "gap_points": -30., "differential": True,
    }
    pd.DataFrame([row, {**row, "endpoint_id": "potassium"}]).to_parquet(table)
    StageManifest.complete(
        stage="segment-availability", stage_version=SEGMENT_AVAILABILITY_VERSION,
        input_hash="synthetic", config_hash="synthetic", output_paths=(str(table),),
    ).save(root / "segment_availability.manifest.json")
    page = ResultPage(
        run_id="test", perspective="pathologies", artifact_path=artifact,
        rows=pd.DataFrame([{"endpoint_id": "echo", "feature_id": "area",
                            "n_reference_eligible": 100, "n_marker_raw_observed": 80,
                            "n_marker_observed": 70,
                            "n_analysis_restriction_excluded": 10}]),
        total_count=1, offset=0, limit=1, snapshot_hash="synthetic",
        analysis_completeness="complete_snapshot", status="ok",
    )
    snapshot = load_availability(page, "echo")
    assert len(snapshot.segment_rows) == 1
    assert "Train+Validation" in snapshot.segment_note
    assert "ohne Dateihash" in snapshot.segment_note
    table.write_bytes(b"changed")
    assert load_availability(page, "echo").segment_rows.empty


def test_detail_marker_selection_updates_availability_context(monkeypatch) -> None:
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    assert app is not None
    widget = ResultsAtlasWidget(repository=None)
    try:
        page = _page([
            {"endpoint_id": "echo", "feature_id": "first", "roc_auc": .8,
             "candidate_type": "vcg_single", "selection_eligible": True},
            {"endpoint_id": "echo", "feature_id": "second", "roc_auc": .7,
             "candidate_type": "vcg_single", "selection_eligible": True},
        ])
        widget._last_page = page
        widget._last_request = ("development-test", "echo", 1)
        widget._show_endpoint_evidence(page)
        assert not widget.endpoint_display.isTabEnabled(0)
        assert widget.endpoint_display.currentIndex() == 1
        starts = []
        monkeypatch.setattr(widget, "_start_availability_load", lambda: starts.append(True))
        widget._select_marker_row(1, 0)
        assert widget._selected_marker_id == "second"
        assert widget._selected_endpoint_id == "echo"
        assert widget.availability_view._snapshot.marker_id == "second"
        assert "second" in widget.availability_view.note.text()
        assert starts == [True]
    finally:
        widget.close()


def test_detail_navigation_distinguishes_pending_empty_and_available() -> None:
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    assert app is not None
    widget = ResultsAtlasWidget(repository=None)
    try:
        index = widget.detail_tabs.indexOf(widget.prognosis_view)
        item = widget._detail_nav_items[index]
        widget._set_nav_status(widget.prognosis_view, "pending")
        assert "ausstehend" in item.text(0)
        widget._set_nav_status(widget.prognosis_view, "empty")
        assert "ohne Werte" in item.text(0)
        widget._set_nav_status(widget.prognosis_view, "ok")
        assert "Ergebnisse" in item.text(0)
    finally:
        widget.close()


def test_spatial_link_carries_the_selected_clinical_endpoint() -> None:
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    assert app is not None
    widget = ResultsAtlasWidget(repository=None)
    try:
        selected = []
        widget.spatial_requested.connect(selected.append)
        widget._selected_endpoint_id = "echo_lvef_reduced"
        widget.spatial_button.click()
        assert selected == ["echo_lvef_reduced"]
    finally:
        widget.close()


def test_detail_navigation_groups_all_atlas_views_and_tracks_programmatic_links() -> None:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    assert app is not None
    widget = ResultsAtlasWidget(repository=None)
    try:
        indices = {
            widget.detail_nav.topLevelItem(group).child(child).data(
                0, Qt.ItemDataRole.UserRole
            )
            for group in range(widget.detail_nav.topLevelItemCount())
            for child in range(widget.detail_nav.topLevelItem(group).childCount())
        }
        assert indices == set(range(widget.detail_tabs.count()))
        widget.set_detailed(True)
        assert widget._map_worker is None
        widget.detail_tabs.setCurrentIndex(widget.marker_tab_index)
        assert widget._map_worker is None
        assert widget.detail_nav.currentItem().data(
            0, Qt.ItemDataRole.UserRole
        ) == widget.marker_tab_index
        widget._navigate_detail(widget._detail_nav_items[0])
        assert widget.detail_tabs.currentIndex() == 0
        widget.set_detailed(False)
        assert widget.detail_shell.isHidden()
    finally:
        widget.close()


def test_detail_back_forward_tracks_views_without_mixing_runs(monkeypatch) -> None:
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    assert app is not None
    widget = ResultsAtlasWidget(repository=None)
    try:
        widget.detail_tabs.setCurrentIndex(1)
        widget.detail_tabs.setCurrentIndex(2)
        assert widget.history_back_button.isEnabled()
        widget.history_back_button.click()
        assert widget.detail_tabs.currentIndex() == 1
        assert widget.history_forward_button.isEnabled()
        widget.history_forward_button.click()
        assert widget.detail_tabs.currentIndex() == 2
        widget._last_request = ("old_run", "", 1)
        monkeypatch.setattr(widget, "_start_load", lambda *_args: None)
        widget.refresh("new_run")
        assert not widget.history_back_button.isEnabled()
        assert not widget.history_forward_button.isEnabled()
    finally:
        widget.close()


def test_atlas_detail_fits_common_viewports_without_horizontal_scroll() -> None:
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    assert app is not None
    widget = ResultsAtlasWidget(repository=None)
    try:
        widget.set_detailed(True)
        for width, height in ((1366, 768), (1440, 900), (1920, 1080)):
            widget.resize(width, height)
            widget.show()
            app.processEvents()
            assert widget.width() == width
            assert widget.height() == height
            assert widget.detail_scroll.horizontalScrollBar().maximum() == 0
            assert widget.detail_scroll.viewport().width() > 500
        widget.detail_tabs.setCurrentWidget(widget.baseline_view)
        app.processEvents()
        assert widget.detail_scroll.verticalScrollBar().maximum() >= 0
    finally:
        widget.close()


def test_simple_answer_cards_scroll_inside_common_laptop_viewport() -> None:
    from PyQt6.QtWidgets import QApplication, QLabel

    app = QApplication.instance() or QApplication([])
    assert app is not None
    widget = ResultsAtlasWidget(repository=None)
    try:
        for column in range(3):
            card = QLabel("Forschungshinweis mit Einschränkungen. " * 35)
            card.setWordWrap(True)
            widget.cards_layout.addWidget(card, 0, column)
        widget.set_detailed(False)
        widget.resize(1366, 768)
        widget.show()
        app.processEvents()
        assert widget.size().width() == 1366
        assert widget.size().height() == 768
        assert widget.simple_scroll.horizontalScrollBar().maximum() == 0
        assert widget.simple_scroll.verticalScrollBar().maximum() > 0
        widget.set_detailed(True)
        assert widget.simple_scroll.isHidden()
    finally:
        widget.close()


def test_atlas_signature_design_keeps_empty_state_honest_in_both_themes() -> None:
    from PyQt6.QtWidgets import QApplication
    from vcg_biomarker.gui.theme import generate_stylesheet

    app = QApplication.instance() or QApplication([])
    assert app is not None
    widget = ResultsAtlasWidget(repository=None)
    try:
        assert widget.hero.objectName() == "resultsAtlasHero"
        assert widget.empty_state.objectName() == "resultsAtlasEmpty"
        for dark in (False, True):
            widget.setStyleSheet(generate_stylesheet(dark))
            widget.resize(1366, 768)
            widget.show()
            app.processEvents()
            assert widget.width() == 1366
            assert widget.simple_scroll.horizontalScrollBar().maximum() == 0
            assert widget.empty_state.isVisible()
        widget.set_detailed(True)
        assert widget.empty_state.isHidden()
        widget.set_detailed(False)
        widget.refresh("")
        assert "weder ein negativer befund" in widget.empty_message.text().lower()
        assert widget.empty_state.isVisible()
    finally:
        widget.close()


def test_simple_atlas_labels_other_research_without_promoting_it() -> None:
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    assert app is not None
    widget = ResultsAtlasWidget(repository=None)
    try:
        payload = SimpleNamespace(
            scores=SimpleNamespace(status="ok", models=pd.DataFrame([
                {"auc": .72, "endpoint_id": "echo"},
            ])),
            prognosis=SimpleNamespace(status="ok", fits=pd.DataFrame([
                {"status": "estimable"},
            ])),
            response=SimpleNamespace(status="ok", rows=pd.DataFrame([
                {"scope": "medication_class", "status": "estimable_point_only"},
            ])),
        )
        widget._show_research_summary(payload)
        assert widget.research_host.property("has_data") is True
        assert "weder neue klinische" in widget.research_summary.text().lower()
        requested = []
        widget.detail_requested.connect(requested.append)
        widget.score_summary_button.click()
        assert requested == [widget.detail_tabs.indexOf(widget.score_view)]
        widget.set_detailed(True)
        assert widget.research_host.isHidden()
    finally:
        widget.close()


def test_availability_widget_only_draws_subgroup_ci_when_real() -> None:
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    assert app is not None
    widget = AtlasAvailabilityWidget()
    try:
        frame = pd.DataFrame([
            {"subgroup_variable": "age", "subgroup_level": "<65",
             "roc_auc": .7, "roc_auc_ci_low": .6, "roc_auc_ci_high": .8,
             "n_positive": 20, "n_negative": 30, "inference_status": "formal_inference"},
            {"subgroup_variable": "age", "subgroup_level": ">=65",
             "roc_auc": .65, "roc_auc_ci_low": np.nan, "roc_auc_ci_high": np.nan,
             "n_positive": 25, "n_negative": 30, "inference_status": "descriptive_only"},
        ])
        widget.set_snapshot(AvailabilitySnapshot(
            "echo", "area", 100, 80, 70, 10, frame, "Test",
            pd.DataFrame([{"category": "roh nicht messbar", "count": 20,
                           "interpretation": "nicht aufgelöst"}]),
        ))
        assert not widget.subgroup_plot.isHidden()
        assert widget.subgroup_table.item(0, 3).text() == "0.600–0.800"
        assert widget.subgroup_table.item(1, 3).text() == "nicht berechnet"
        assert "1 von 2" in widget.subgroup_note.text()
        assert widget.reason_table.item(0, 1).text() == "20"
    finally:
        widget.close()


def test_legacy_navigator_does_not_build_gui_cache_or_read_unsealed_artifacts(tmp_path) -> None:
    from vcg_biomarker.gui.widgets.mimic_project_widget import MimicProjectWidget

    artifact = tmp_path / "pathology_marker_explorer.parquet"
    pd.DataFrame({"endpoint_id": ["echo"]}).to_parquet(artifact)
    load_paths = MimicProjectWidget._ensure_result_explorer_artifacts
    assert load_paths(None, tmp_path) == {}
    assert not (tmp_path / "ui_cache").exists()
    StageManifest.complete(
        stage="validate", stage_version="synthetic", input_hash="synthetic",
        config_hash="synthetic", output_paths=(str(artifact),),
    ).save(tmp_path / "validate.manifest.json")
    assert load_paths(None, tmp_path) == {
        "pathology_marker": artifact,
    }
    assert not (tmp_path / "ui_cache").exists()


def test_model_quality_chart_has_matching_numeric_table_and_provenance() -> None:
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    widget = AtlasModelQualityWidget()
    try:
        snapshot = ModelQualitySnapshot(
            "ok", "Interne Validation", pd.DataFrame([{
                "model_id": "m", "calibration_bin": 1, "n": 30, "events": 6,
                "mean_predicted": .25, "observed_rate": .2,
                "observed_ci_low": .1, "observed_ci_high": .4,
            }]), pd.DataFrame([{
                "model_id": "m", "threshold_probability": .2,
                "net_benefit": .04, "treat_all_net_benefit": .01,
                "treat_none_net_benefit": 0.,
                "evidence_scope": "internal_development_descriptive",
            }]), "echo", "run", "snapshot",
            metrics=pd.DataFrame([{
                "model_id": "m", "n_subjects": 120, "roc_auc": .8,
                "brier_score": .16, "calibration_intercept": -.05,
                "calibration_slope": .92, "inference_status": "formal_inference",
            }]),
        )
        widget.set_snapshot(snapshot)
        assert widget.points_table.rowCount() == 1
        assert widget.metrics_table.rowCount() == 1
        assert widget.metrics_table.item(0, 3).text() == "0.160"
        assert widget._metadata()["snapshot_hash"] == "snapshot"
        widget.tabs.setCurrentIndex(1)
        assert widget.points_table.rowCount() == 1
        assert "net_benefit" in [widget.points_table.horizontalHeaderItem(i).text()
                                 for i in range(widget.points_table.columnCount())]
    finally:
        widget.close()
        app.processEvents()


def test_invalid_calibration_points_do_not_leave_a_false_reference_only_chart() -> None:
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    assert app is not None
    widget = AtlasModelQualityWidget()
    try:
        widget.set_snapshot(ModelQualitySnapshot(
            "ok", "Interne Validation", pd.DataFrame([{
                "model_id": "m", "mean_predicted": np.nan,
                "observed_rate": .3, "observed_ci_low": np.nan,
                "observed_ci_high": np.nan,
            }]), pd.DataFrame(), "echo",
        ))
        assert not widget.calibration_plot.listDataItems()
        assert "Keine gültigen" in widget.calibration_plot.plotItem.titleLabel.text
    finally:
        widget.close()


def test_atlas_shutdown_rejects_new_result_queries() -> None:
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    widget = ResultsAtlasWidget(repository=None)
    try:
        assert widget.request_shutdown()
        widget.refresh("run", "endpoint", revision=1)
        widget.set_detailed(True)
        assert widget._workers == []
        assert widget._map_worker is None
        assert widget.marker_view.request_shutdown()
    finally:
        widget.close()
        app.processEvents()


def test_transport_keeps_external_cohorts_separate_and_manifest_bound(tmp_path) -> None:
    import json
    from types import SimpleNamespace

    root = tmp_path / "external" / "run" / "population_cohorts"
    root.mkdir(parents=True)
    coverage = root / "coverage.parquet"
    mortality = root / "mortality_associations.parquet"
    pd.DataFrame([
        {"cohort": "CODE-15", "marker_id": "a", "availability": .8},
        {"cohort": "SaMi-Trop", "marker_id": "a", "availability": .5},
    ]).to_parquet(coverage)
    pd.DataFrame([
        {"cohort": "CODE-15", "marker_id": "a", "status": "estimable",
         "hazard_ratio_per_sd": 1.2, "ci_low": 1.1, "ci_high": 1.3,
         "n": 100, "events": 12, "outcome": "mortality", "q_value_cohort": .04},
        {"cohort": "SaMi-Trop", "marker_id": "a", "status": "estimable",
         "hazard_ratio_per_sd": 1.5, "ci_low": 1.2, "ci_high": 1.9,
         "n": 70, "events": 15, "outcome": "mortality", "q_value_cohort": .03},
        {"cohort": "CODE-15", "marker_id": "b", "status": "non_estimable",
         "hazard_ratio_per_sd": None, "ci_low": None, "ci_high": None,
         "n": 40, "events": 2, "outcome": "mortality", "q_value_cohort": None},
    ]).to_parquet(mortality)
    manifest = {
        "status": "complete", "run_id": "run", "feature_implementation_hash": "feature-hash",
        "locked_test_read": False, "input_hash": "external-snapshot",
        "output_paths": [str(coverage.resolve()), str(mortality.resolve())],
        "output_integrity": list(capture_output_integrity((coverage, mortality))),
    }
    (root / "population_cohorts.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    class Repository:
        config = SimpleNamespace(external_validation_root=tmp_path / "external")
        expected_feature_code_hash = "feature-hash"

        def available_runs(self):
            return (SimpleNamespace(run_id="run", complete_compatible=True),)

    snapshot = load_transport_snapshot(Repository(), "run")
    assert snapshot.status == "ok"
    assert snapshot.coverage["cohort"].tolist() == ["CODE-15", "SaMi-Trop"]
    assert snapshot.mortality["cohort"].tolist() == ["CODE-15", "CODE-15", "SaMi-Trop"]
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    assert app is not None
    widget = AtlasTransportWidget()
    widget.set_snapshot(snapshot)
    assert widget.mortality_table.rowCount() == 2
    assert widget.mortality_table.item(1, 4).text() == "nicht schätzbar"
    widget.cohort.setCurrentIndex(1)
    assert widget.mortality_table.rowCount() == 1
    assert widget.mortality_table.item(0, 0).text() == widget.cohort.currentData()
    widget.close()
    pd.DataFrame([
        {"cohort": "CODE-15", "marker_id": f"marker_{index:03d}", "status": "estimable",
         "hazard_ratio_per_sd": 1.2, "ci_low": 1.1, "ci_high": 1.3,
         "n": 100, "events": 12, "outcome": "mortality", "q_value_cohort": .01 + index / 10000}
        for index in range(105)
    ] + [{
        "cohort": "SaMi-Trop", "marker_id": "one", "status": "estimable",
        "hazard_ratio_per_sd": 1.5, "ci_low": 1.2, "ci_high": 1.9,
        "n": 70, "events": 15, "outcome": "mortality", "q_value_cohort": .4,
    }]).to_parquet(mortality)
    manifest["output_integrity"] = list(capture_output_integrity((coverage, mortality)))
    (root / "population_cohorts.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    large = load_transport_snapshot(Repository(), "run")
    assert large.status == "ok"
    assert large.mortality.cohort.value_counts().to_dict() == {"CODE-15": 100, "SaMi-Trop": 1}
    manifest["run_id"] = "other"
    (root / "population_cohorts.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert load_transport_snapshot(Repository(), "run").status == "incompatible"
