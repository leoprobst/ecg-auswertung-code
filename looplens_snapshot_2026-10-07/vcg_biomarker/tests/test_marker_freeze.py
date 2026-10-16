from __future__ import annotations

import json

import pandas as pd
import pytest

from vcg_biomarker.mimic.config import MimicProjectConfig
from vcg_biomarker.mimic.marker_freeze import (
    assert_marker_freeze_current,
    freeze_marker_inventory,
    marker_freeze_payload,
)


def test_researched_backend_families_are_completely_exported_in_freeze():
    """Registration alone must not leave research implementations off-panel."""
    from vcg_biomarker.core.biomarkers import BiomarkerRegistry
    from vcg_biomarker.core.temporal_research import ANALYSES

    sources = {
        "propagation_sequence_v1", "loop_parametrisation_v1",
        "thesis_derived_v1", "ischemia_thesis_v1", "decision_thesis_v1",
        "conduction_uniformity_v1", "pathology_mechanism_round_20260913",
        "qrs_t_coupling_extension_v1",
    }
    payload = marker_freeze_payload()
    frozen = {row["marker_id"]: row for row in payload["markers"]}
    definitions = BiomarkerRegistry.definitions()
    for source in sources:
        expected = {item.name for item in definitions if item.source == source}
        assert expected, f"Research family disappeared: {source}"
        assert expected <= frozen.keys(), f"Registered but not frozen: {expected - frozen.keys()}"
        for name in expected:
            row = frozen[name]
            assert row["formula"] and row["unit"] and row["references"]
            assert row["panel_ids"]
            assert row["initial_evidence_tier"] == "exploratory"
    assert set(payload["temporal_research"]["analyses"]) == set(ANALYSES)
    assert not payload["temporal_research"]["clinical_claim_allowed"]


def test_additional_research_families_are_frozen_with_explicit_reference_status():
    from vcg_biomarker.core.biomarkers import BiomarkerRegistry

    sources = {
        "literature_2026_v1", "literature_2026b_v1",
        "morphology_variability_v1", "circular_projection_v1",
        "rotation_invariant_v1", "segment_free_v1_1", "ndat_short_record_1.1.1",
    }
    frozen = {row["marker_id"]: row for row in marker_freeze_payload()["markers"]}
    for source in sources:
        definitions = [d for d in BiomarkerRegistry.definitions() if d.source == source]
        assert definitions, source
        for definition in definitions:
            row = frozen[definition.name]
            assert row["formula"] and row["unit"] and row["panel_ids"]
            expected = ("references_declared_not_clinical_validation" if definition.references
                        else "no_reference_declared")
            assert row["reference_status"] == expected
            assert row["references"] == sorted(set(definition.references))
            assert row["initial_evidence_tier"] == "exploratory"


def test_fixed_plane_inventory_covers_every_segment_and_anatomical_plane():
    payload = marker_freeze_payload()
    projected = [row for row in payload["markers"] if row["fixed_plane_projection"] is not None]
    assert len(projected) == 180
    for segment in ("p", "qrs", "qrst", "st", "t"):
        for plane, axes in (("frontal_xy", ["X", "Y"]), ("horizontal_xz", ["X", "Z"]), ("sagittal_yz", ["Y", "Z"])):
            rows = [row for row in projected if row["fixed_plane_projection"]["segment"] == segment
                    and row["fixed_plane_projection"]["plane"] == plane]
            assert len(rows) == 12
            assert all(row["fixed_plane_projection"]["axes"] == axes for row in rows)
            assert all(not row["fixed_plane_projection"]["clinical_threshold_validated"] for row in rows)
            assert all(row["formula"] and row["unit"] and row["panel_ids"] for row in rows)


def test_nonanatomical_and_nonprojection_markers_are_not_assigned_a_plane():
    from vcg_biomarker.mimic.marker_freeze import projection_metadata
    for marker in ("svd_component_1", "qrs_area", "qrs_loop_best_fit_area", "svg_azimuth"):
        assert projection_metadata(marker) is None


def test_projection_hypotheses_cover_all_markers_without_posthoc_plane_selection():
    from vcg_biomarker.mimic.labels import _label_rules
    payload = marker_freeze_payload()
    rows = payload["projection_hypotheses"]
    assert len(rows) == 396
    projected = {row["marker_id"] for row in payload["markers"] if row["fixed_plane_projection"]}
    assert {row["marker_id"] for row in rows} == projected
    assert {row["endpoint_id"] for row in rows} <= set(_label_rules())
    assert not pd.DataFrame(rows).duplicated(["marker_id", "endpoint_id"]).any()
    for endpoint, group in pd.DataFrame(rows).groupby("endpoint_id"):
        assert set(group.plane) == {"frontal_xy", "horizontal_xz", "sagittal_yz"}, endpoint
        assert group.groupby("plane").size().nunique() == 1
    assert all(row["expected_direction"] == "two_sided_no_prespecified_monotonic_direction" for row in rows)
    assert not any(row["clinical_threshold_validated"] or row["monitoring_lead_recommendation_allowed"] for row in rows)


def _config(tmp_path):
    return MimicProjectConfig(
        mimic_iv_root=tmp_path / "mimic",
        mimic_ed_root=tmp_path / "ed",
        mimic_ecg_root=tmp_path / "ecg",
        echo_structured_path=tmp_path / "echo.csv",
        echo_study_list_path=tmp_path / "echo-study.csv",
        echo_record_list_path=tmp_path / "echo-record.csv",
        work_root=tmp_path / "work",
        workers=1,
        duckdb_memory_limit_gb=2,
    )


def test_marker_freeze_is_machine_readable_and_self_consistent(tmp_path):
    config = _config(tmp_path)
    result = freeze_marker_inventory(
        config,
        run_id="R001",
        finalize_inputs=True,
    )
    checked = assert_marker_freeze_current(config, run_id="R001")
    freeze = json.loads(
        (config.runs_root / "R001" / "marker_freeze.json").read_text("utf-8")
    )
    assert checked["freeze_hash"] == result["freeze_hash"] == freeze["freeze_hash"]
    marker = next(
        item
        for item in freeze["payload"]["markers"]
        if item["marker_id"] == "qrs_t_integral_angle_direct_8lead_deg"
    )
    assert marker["unit"] == "degrees"
    assert marker["segment_dependencies"] == ["qrs", "t"]
    assert marker["initial_evidence_tier"] == "exploratory"
    assert marker["references"] == [
        "https://doi.org/10.1038/s41598-026-37361-w"
    ]


def test_marker_preflight_snapshot_finalizes_then_checks_upstream_inputs(
    tmp_path,
):
    config = _config(tmp_path)
    config.ensure_directories()
    inventory = config.work_root / "data_inventory.json"
    inventory.write_text('{"inventory":"v1"}', encoding="utf-8")
    first = freeze_marker_inventory(config, run_id="R001")
    snapshot_path = config.runs_root / "R001" / "run_preflight_snapshot.json"
    first_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))

    second = freeze_marker_inventory(
        config,
        run_id="R001",
        finalize_inputs=True,
    )
    second_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert second["snapshot_hash"] != first["snapshot_hash"]
    assert second_snapshot["created_at"] == first_snapshot["created_at"]
    assert second_snapshot["payload"]["input_snapshot_phase"] == "prepared"

    inventory.write_text('{"inventory":"v2"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="upstream inputs changed"):
        assert_marker_freeze_current(config, run_id="R001")
    with pytest.raises(RuntimeError, match="Use a new run_id"):
        freeze_marker_inventory(config, run_id="R001")


def test_unfinalized_preflight_cannot_be_used_by_pilot(tmp_path):
    config = _config(tmp_path)
    freeze_marker_inventory(config, run_id="R001")
    with pytest.raises(RuntimeError, match="has not been finalized"):
        assert_marker_freeze_current(config, run_id="R001")


def test_derived_biomarker_audit_does_not_invalidate_prepared_inputs(tmp_path):
    config = _config(tmp_path)
    audit = (
        config.work_root
        / "audits"
        / "biomarkers"
        / "biomarker_audit_summary.json"
    )
    audit.parent.mkdir(parents=True)
    audit.write_text('{"generated_at":"first"}', encoding="utf-8")
    freeze_marker_inventory(config, run_id="R001", finalize_inputs=True)

    audit.write_text('{"generated_at":"retry"}', encoding="utf-8")

    assert_marker_freeze_current(config, run_id="R001")


def test_tampered_snapshot_is_refused(tmp_path):
    config = _config(tmp_path)
    freeze_marker_inventory(config, run_id="R001")
    snapshot_path = config.runs_root / "R001" / "run_preflight_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["payload"]["random_seed"] = -1
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    with pytest.raises(RuntimeError, match="snapshot and marker freeze differ"):
        assert_marker_freeze_current(config, run_id="R001")


def test_changed_marker_freeze_is_refused_under_same_run_id(tmp_path, monkeypatch):
    config = _config(tmp_path)
    freeze_marker_inventory(config, run_id="R001")
    original = marker_freeze_payload()
    changed = {**original, "panel_version": "changed-after-pilot"}
    monkeypatch.setattr(
        "vcg_biomarker.mimic.marker_freeze.marker_freeze_payload",
        lambda: changed,
    )
    with pytest.raises(RuntimeError, match="Use a new run_id"):
        assert_marker_freeze_current(config, run_id="R001")
