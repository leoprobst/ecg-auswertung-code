import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.mechanism_echo_context import attach_echo_context


def test_context_preserves_index_and_rejects_wrong_patient_and_window():
    frame = pd.DataFrame({"subject_id": [1, 2, 3, 4, 5],
                          "nearest_echo_measurement_id": [10, 20, 30, 40, 99],
                          "nearest_echo_delta_hours": [-720, 720, 721, 0, 0]}, index=[8]*5)
    echo = pd.DataFrame({"measurement_id": [40, 30, 20, 10],
                         "subject_id": [999, 3, 2, 1], "tapse": [1., 2., 3., 4.],
                         "identity_conflict": [False]*4})
    frame["ecg_time"] = pd.Timestamp("2020-02-01")
    echo["echo_time"] = pd.Timestamp("2020-02-01") + pd.to_timedelta([0, 721, 720, -720], unit="h")
    result = attach_echo_context(frame, echo, columns=["tapse"])
    pd.testing.assert_frame_equal(result[frame.columns], frame)
    assert result.tapse.iloc[:2].tolist() == [4., 3.]
    assert result.tapse.iloc[2:].isna().all()
    assert result.mechanism_echo_context_eligible.tolist() == [True, True, False, False, False]


def test_ambiguous_echo_identity_and_overwrite_rejected():
    frame = pd.DataFrame({"subject_id": [1], "nearest_echo_measurement_id": [1],
                          "nearest_echo_delta_hours": [0]})
    echo = pd.DataFrame({"measurement_id": [1, 1], "subject_id": [1, 1], "tapse": [1., 2.],
                         "identity_conflict": [False, False]})
    frame["ecg_time"] = pd.Timestamp("2020-01-01")
    echo["echo_time"] = pd.Timestamp("2020-01-01")
    with pytest.raises(ValueError, match="unique"):
        attach_echo_context(frame, echo, columns=["tapse"])
    with pytest.raises(ValueError, match="overwrite"):
        attach_echo_context(frame, echo, columns=["subject_id"])
    with pytest.raises(ValueError, match="window"):
        attach_echo_context(frame, echo, columns=["tapse"], max_delta_hours=np.inf)


def test_real_parquet_loader_rejects_conflicting_or_missing_identity(tmp_path):
    from vcg_biomarker.mimic.mechanism_echo_context import load_echo_context
    wide = pd.DataFrame({"measurement_id": [1, 2, 3, 4], "subject_id": [1, 2, 3, 4],
                         "tapse": [1., 2., 3., 4.]})
    wide["echo_time"] = pd.Timestamp("2020-01-01")
    identities = pd.DataFrame({"measurement_id": [1, 2, 3], "subject_id": [1, 2, 99],
                               "identity_conflict": [False, True, False]})
    wide.to_parquet(tmp_path / "wide.parquet")
    identities.to_parquet(tmp_path / "index.parquet")
    loaded = load_echo_context(tmp_path / "wide.parquet", tmp_path / "index.parquet", columns=["tapse"])
    index = wide[["subject_id", "measurement_id"]].rename(columns={"measurement_id": "nearest_echo_measurement_id"})
    index["nearest_echo_delta_hours"] = 0.
    index["ecg_time"] = pd.Timestamp("2020-01-01")
    output = attach_echo_context(index, loaded, columns=["tapse"])
    assert output.tapse.iloc[0] == 1.
    assert output.tapse.iloc[1:].isna().all()


@pytest.mark.parametrize("stored,echo_time,expected", [
    (1., "2020-01-01 01:00:00", True),
    (-1., "2020-01-01 01:00:00", False),
    (0., "2020-01-02", False),
    (0., None, False),
])
def test_clock_mismatch_or_missing_clock_abstains(stored, echo_time, expected):
    frame = pd.DataFrame({"subject_id": [1], "ecg_time": ["2020-01-01"],
                          "nearest_echo_measurement_id": [10], "nearest_echo_delta_hours": [stored]})
    echo = pd.DataFrame({"subject_id": [1], "measurement_id": [10],
                         "echo_time": [echo_time], "identity_conflict": [False], "tapse": [2.]})
    output = attach_echo_context(frame, echo, columns=["tapse"])
    assert bool(output.mechanism_echo_context_eligible.iloc[0]) is expected
    assert bool(output.tapse.notna().iloc[0]) is expected
