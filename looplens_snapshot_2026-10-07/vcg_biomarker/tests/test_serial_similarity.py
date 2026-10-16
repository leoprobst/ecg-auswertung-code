import json

import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.core.temporal_research import GRID, VERSION, serial_similarity_decomposition
from vcg_biomarker.mimic.temporal_response import temporal_response_table


def trajectory():
    return np.random.default_rng(89).normal(size=(len(GRID), 3))


def test_known_gain_and_proper_rotation():
    x = trajectory()
    r = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    result = serial_similarity_decomposition(x, 2.5*x @ r)
    assert result["gain_after_over_before"] == pytest.approx(2.5)
    assert result["rotation_deg"] == pytest.approx(90)
    assert result["relative_residual"] < 1e-14
    np.testing.assert_allclose(result["rotation_matrix"], r, atol=1e-14)


def test_reflection_is_not_absorbed_as_rotation():
    x = trajectory()
    result = serial_similarity_decomposition(x, x * [-1., 1., 1.])
    assert np.linalg.det(result["rotation_matrix"]) == pytest.approx(1)
    assert result["relative_residual"] > .1


@pytest.mark.parametrize("bad", [None, np.zeros((33, 3)), np.ones((33, 3)),
                                  np.full((33, 3), np.nan), np.ones((32, 3))])
def test_missing_or_degenerate_trajectory_abstains(bad):
    assert serial_similarity_decomposition(trajectory(), bad)["status"] == "non_estimable"


def test_existing_production_pair_adapter_exports_decomposition_and_blocks_locked_test():
    x = trajectory()
    def descriptor(points):
        return json.dumps({"source": "mimic", "subject_id": "1", "transform": "kors",
                           "version": VERSION, "signal_path": "record_morphology",
                           "qrs": {"arc_clock": GRID.tolist(), "duration_ms": 100,
                                   "phase_xyz_mv": points.tolist()}})
    row = {"subject_id": 1, "split_before": "train", "split_after": "train",
           "hours_before_to_exposure": 1., "hours_exposure_to_after": 1.,
           "temporal_before": descriptor(x), "temporal_after": descriptor(2*x)}
    output = temporal_response_table(pd.DataFrame([row]))
    result = json.loads(output.iloc[0].details_json)["similarity_decomposition"]
    assert result["gain_after_over_before"] == pytest.approx(2)
    assert not result["clinical_claim_allowed"]
    import duckdb
    from vcg_biomarker.mimic.temporal_response import dialysis_temporal_response
    features = pd.DataFrame({"subject_id": [1, 1], "study_id": [10, 11],
                             "split": ["train", "train"], "extraction_status": ["ok", "ok"],
                             "context_temporal_research_json": [descriptor(x), descriptor(2*x)]})
    pairs = pd.DataFrame([{"subject_id": 1, "first_study_id": 10, "second_study_id": 11,
                           "split": "train", "hours_between": 4., "n_sessions": 1}])
    with duckdb.connect() as connection:
        connection.register("synthetic_features", features)
        dialysis = dialysis_temporal_response(connection, "synthetic_features", pairs,
                                              descriptor_available=True)
    details = json.loads(dialysis.iloc[0].details_json)
    assert details["rhythm_context"]["comparison"] == "unknown"
    assert not details["rhythm_context"]["clinical_rhythm_stability_verified"]
    assert details["similarity_decomposition"]["gain_after_over_before"] == pytest.approx(2)
    assert "not_proven_post_completion" in details["interpretation"]
    assert details["intervention_context"]["count_semantics"] == "procedure_rows_not_independent_sessions"
    assert not details["intervention_context"]["session_identity_verified"]
    assert not details["intervention_context"]["treatment_completion_verified"]
    row.update(split_before="test", split_after="test")
    assert temporal_response_table(pd.DataFrame([row])).iloc[0].status == "non_estimable"
