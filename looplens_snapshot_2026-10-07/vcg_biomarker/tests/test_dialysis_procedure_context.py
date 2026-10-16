import duckdb
import json
import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.within_subject import DIALYSIS_WINDOW, intervention_pair_query


def test_procedure_rows_are_not_claimed_as_independent_sessions(tmp_path):
    master = pd.DataFrame({"subject_id": [1, 1], "study_id": [10, 11],
                           "split": ["train", "train"], "hadm_id": [100, 100],
                           "ecg_time": pd.to_datetime(["2026-01-01", "2026-01-01 06:00"], format="mixed")})
    # Duplicate rows and different modalities are deliberately retained as
    # source context, not asserted to be three independent treatments.
    procedures = pd.DataFrame({"subject_id": [1, 1, 1], "hadm_id": [100]*3,
                               "itemid": [225441, 225441, 225802],
                               "starttime": pd.to_datetime(["2026-01-01 01:00"]*3)})
    features = pd.DataFrame({"subject_id": [1, 1], "study_id": [10, 11],
                             "extraction_status": ["ok", "ok"],
                             "signal_quality_score": [.9, .9], "marker": [1., 2.]})
    paths = [tmp_path / f"{name}.parquet" for name in ("master", "procedures", "features")]
    for frame, path in zip((master, procedures, features), paths):
        frame.to_parquet(path, index=False)
    m, p, f = [path.as_posix() for path in paths]
    sql = intervention_pair_query(m, p, f"read_parquet('{f}')", window=DIALYSIS_WINDOW,
                                 marker_columns="a.marker AS marker_first, b.marker AS marker_second")
    with duckdb.connect() as connection:
        result = connection.execute(sql).df()
    assert len(result) == 1
    row = result.iloc[0]
    assert row.n_procedure_rows == 3
    assert row.n_sessions == 3  # compatibility alias, explicitly qualified
    assert row.n_procedure_types == 2
    assert row.procedure_itemids == "225441,225802"
    assert row.intervention_count_semantics == "procedure_rows_not_independent_sessions"


@pytest.mark.parametrize("count", [np.nan, np.inf, -np.inf, None, pd.NA, 0, -1, 1.5, True])
def test_invalid_procedure_count_abstains_and_missing_context_is_json_safe(count):
    from vcg_biomarker.mimic.temporal_response import dialysis_temporal_response

    pairs = pd.DataFrame([{"subject_id": 1, "first_study_id": 10, "second_study_id": 11,
                           "split": "train", "hours_between": 4., "n_sessions": count,
                           "n_procedure_types": np.nan, "procedure_itemids": pd.NA}])
    result = dialysis_temporal_response(None, "unused", pairs, descriptor_available=False)
    row = result.iloc[0]
    assert row.status == "non_estimable"
    assert row.reason == "invalid_selected_dialysis_window"
    context = json.loads(row.details_json)["intervention_context"]
    assert context["n_procedure_rows"] is None
    assert context["n_procedure_types"] is None
    assert context["procedure_itemids"] is None
