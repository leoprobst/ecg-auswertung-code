import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.core.temporal_research import (
    GRID, VERSION, arc_clock, direction_time_occupancy, serial_arc_clock,
    rr_history_t_response, pvc_sinus_transition, record_descriptor,
)
from vcg_biomarker.mimic.temporal_response import temporal_response_table


def descriptor(power=1):
    return {"source": "mimic", "subject_id": "1", "transform": "kors", "version": VERSION,
            "signal_path": "record_morphology", "qrs": {"arc_clock": (GRID ** power).tolist(), "duration_ms": 100}}


def test_serial_distance_known_curve_and_no_scalar_panel_alias():
    from vcg_biomarker.mimic.feature_roles import feature_profile, ROLE_NON_SCALAR
    profile = feature_profile("context_temporal_research_json")
    assert profile.role == ROLE_NON_SCALAR
    assert not profile.discovery_eligible
    assert not profile.combination_eligible
    result = serial_arc_clock(descriptor(), descriptor(2))
    assert result["absolute_distance"] == pytest.approx(1 / 6, abs=0.001)
    assert result["qrs_duration_delta_ms"] == 0
    assert not result["clinical_claim_allowed"]


@pytest.mark.parametrize("key", ["source", "subject_id", "transform", "version", "signal_path"])
def test_serial_identity_and_version_are_mandatory(key):
    a, b = descriptor(), descriptor()
    b[key] = "different"
    assert serial_arc_clock(a, b)["status"] == "non_estimable"
    del a[key]
    assert serial_arc_clock(a, b)["status"] == "non_estimable"


def test_arc_invariant_to_constant_scalar_and_translation_not_anisotropy():
    t = np.linspace(0, 1, 100)
    points = np.column_stack([t, t ** 2, t ** 3])
    np.testing.assert_allclose(arc_clock(points), arc_clock(4 * points + 20))
    assert not np.allclose(arc_clock(points), arc_clock(points * [8, 1, 1]))
    points[5] = np.nan
    with pytest.raises(ValueError):
        arc_clock(points)


def test_direction_counts_use_valid_denominator_without_imputing_noise():
    points = np.tile([1., 1., 1.], (12, 1))
    assert direction_time_occupancy(points, noise_rms_mv=None)["status"] == "non_estimable"
    points[0] = 0
    result = direction_time_occupancy(points, noise_rms_mv=0.01)
    assert result["valid_counts"] == [3, 4, 4]
    assert result["total_counts"] == [4, 4, 4]
    assert result["fractions"][0][-1] == 1
    assert result["valid_fraction"] == pytest.approx(11 / 12)


def sequence():
    rng = np.random.default_rng(123)
    times = np.cumsum(rng.uniform(0.8, 1.2, 90))
    return [{"r_time_s": float(t), "beat_type": "sinus", "qt_ms": float(400 + rng.normal(0, 10)),
             "t_half_fraction": float(0.4 + rng.normal(0, .01)), "qrs_clock": GRID.tolist(),
             "qrs_area_direction": [1., 0., 0.]} for t in times]


def test_rr_requires_long_record_and_annotations_and_nondegenerate_design():
    beats = sequence()
    assert rr_history_t_response(beats)["status"] == "descriptive"
    assert rr_history_t_response(beats[:10])["status"] == "non_estimable"
    for beat in beats:
        beat["beat_type"] = "unknown"
    assert rr_history_t_response(beats)["status"] == "non_estimable"


def test_rr_does_not_sort_invalid_temporal_order():
    beats = sequence()
    beats[10]["r_time_s"] = beats[9]["r_time_s"]
    assert rr_history_t_response(beats)["reason"] == "invalid_beat_time_order"


def test_pvc_transition_requires_explicit_labels_and_tracks_coupling():
    beats = sequence()[:5]
    assert pvc_sinus_transition(beats)["status"] == "non_estimable"
    beats[2]["beat_type"] = "pvc"
    result = pvc_sinus_transition(beats)
    assert result["n_pairs"] == 1
    assert result["pairs"][0]["coupling_interval_ms"] > 0
    assert result["pairs"][0]["area_direction_difference_deg"] == 0
    beats[3]["beat_type"] = "unknown"
    assert pvc_sinus_transition(beats)["status"] == "non_estimable"


def test_record_adapter_produces_serializable_payload_and_explicit_abstention():
    point = lambda sample: SimpleNamespace(sample=sample, confidence=1.)
    beat = SimpleNamespace(r_peak=point(15), r_onset=point(5), r_offset=point(30),
                           t_onset=point(40), t_offset=point(65), segment_confidence=lambda _: 1.)
    delineation = SimpleNamespace(beats=[beat], representative_beat=beat)
    t = np.linspace(0, 2, 80)
    xyz = np.column_stack([np.sin(t), np.cos(t), t])
    result = record_descriptor(delineation, xyz, 500, source="mimic", subject_id=1, transform="kors")
    json.dumps(result, allow_nan=False)
    assert len(result["qrs"]["arc_clock"]) == 33
    assert result["rhythm_context"]["pattern"] == "unknown"
    phase_points = np.asarray(result["qrs"]["phase_xyz_mv"])
    assert phase_points.shape == (33, 3)
    np.testing.assert_allclose(phase_points[0] + phase_points[-1], 0., atol=1e-14)
    assert result["rr_history_t_response"]["status"] == "non_estimable"
    xyz[10] = np.nan
    result = record_descriptor(delineation, xyz, 500, source="mimic", subject_id=1, transform="kors")
    assert result["qrs"]["status"] == "non_estimable"


def pair():
    return {"subject_id": 1, "split_before": "train", "split_after": "train",
            "hours_before_to_exposure": .5, "hours_exposure_to_after": 1.,
            "temporal_before": json.dumps(descriptor()), "temporal_after": json.dumps(descriptor(2))}


def test_production_pair_adapter_keeps_missing_denominators_and_excludes_locked():
    valid = pair()
    missing = {**valid, "temporal_before": None}
    locked = {**valid, "split_after": "locked_test"}
    result = temporal_response_table(pd.DataFrame([valid, missing, locked]))
    assert len(result) == 3
    assert result.status.tolist() == ["descriptive", "non_estimable", "non_estimable"]
    assert not result.clinical_claim_allowed.any()
    assert "status" in temporal_response_table(pd.DataFrame()).columns


@pytest.mark.parametrize("delta", [0, -1, 3, np.nan])
def test_pair_time_window_fail_closed(delta):
    row = pair()
    row["hours_before_to_exposure"] = delta
    result = temporal_response_table(pd.DataFrame([row]))
    assert result.iloc[0].status == "non_estimable"


def test_malformed_payload_is_not_a_record_failure():
    row = pair()
    row["temporal_before"] = "{broken"
    assert temporal_response_table(pd.DataFrame([row])).iloc[0].status == "non_estimable"


def test_invalid_stored_clock_abstains():
    b = copy.deepcopy(descriptor())
    b["qrs"]["arc_clock"][10] = -1
    assert serial_arc_clock(descriptor(), b)["status"] == "non_estimable"


def test_dialysis_projection_is_patient_and_split_isolated():
    import duckdb
    from vcg_biomarker.mimic.temporal_response import dialysis_temporal_response
    features = pd.DataFrame({"subject_id": [1, 1, 2], "study_id": [10, 11, 11],
        "split": ["train", "train", "locked_test"], "extraction_status": ["ok"] * 3,
        "context_temporal_research_json": [json.dumps(descriptor()), json.dumps(descriptor(2)), "do_not_parse"]})
    pairs = pd.DataFrame({"subject_id": [1], "first_study_id": [10], "second_study_id": [11],
                         "split": ["train"], "hours_between": [8.], "n_sessions": [1]})
    with duckdb.connect() as connection:
        connection.register("fixture_features", features)
        result = dialysis_temporal_response(connection, "fixture_features", pairs, descriptor_available=True)
        assert result.iloc[0].status == "descriptive"
        missing = dialysis_temporal_response(connection, "fixture_features", pairs, descriptor_available=False)
        assert len(missing) == 1 and missing.iloc[0].status == "non_estimable"


def test_freeze_registers_analysis_contract_without_inflating_scalar_panel():
    from vcg_biomarker.mimic.marker_freeze import marker_freeze_payload
    freeze = marker_freeze_payload()
    assert set(freeze["temporal_research"]["analyses"]) == {
        "serial_arc_clock", "direction_time_occupancy", "rr_history_t_response",
        "pvc_sinus_transition", "serial_similarity_decomposition",
    }
    assert not freeze["temporal_research"]["clinical_claim_allowed"]
    assert all(m["marker_id"] != "context_temporal_research_json" for m in freeze["markers"])
