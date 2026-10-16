import duckdb
import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.paired_clinical_context import attach_baseline_context


def test_context_is_patient_specific_first_ecg_and_development_only(tmp_path):
    master = pd.DataFrame(dict(subject_id=[1, 1, 2, 3], study_id=[10, 11, 20, 30],
        split=["train", "train", "locked_test", "validation"],
        age=[60., 61., 70., np.inf], sex=["F", "M", "M", None],
        machine_heart_rate_bpm=[80., 190., 100., -1.]))
    path = tmp_path / "master.parquet"
    master.to_parquet(path)
    pairs = pd.DataFrame(dict(subject_id=[1, 1, 2, 9, 3],
        first_study_id=[10, 10, 20, 10, 30], second_study_id=[11]*5), index=[9, 8, 7, 6, 5])
    original = pairs.copy(deep=True)
    with duckdb.connect() as con:
        result = attach_baseline_context(con, pairs, path)
    pd.testing.assert_frame_equal(pairs, original)
    assert result.index.tolist() == pairs.index.tolist()
    assert result.baseline_adjust_heart_rate.iloc[:2].tolist() == [80., 80.]
    assert result.baseline_adjust_sex_male.iloc[:2].tolist() == [0., 0.]
    assert result.baseline_context_status.tolist() == ["matched_development_index"]*2 + ["missing_development_index"]*2 + ["matched_development_index"]
    assert result.baseline_adjust_age.iloc[2:].isna().all()
    assert result.baseline_adjust_heart_rate.iloc[2:].isna().all()
    assert result.baseline_hadm_id.isna().all()


def test_duplicate_identity_abstains_and_registration_is_cleaned(tmp_path):
    path = tmp_path / "master.parquet"
    pd.DataFrame(dict(subject_id=[1, 1], study_id=[10, 10], split=["train"]*2)).to_parquet(path)
    pairs = pd.DataFrame(dict(subject_id=[1], first_study_id=[10]))
    with duckdb.connect() as con:
        with pytest.raises(ValueError, match="Ambiguous"):
            attach_baseline_context(con, pairs, path)
        assert "paired_baseline_keys" not in con.execute("SHOW TABLES").df()["name"].tolist()


def test_missing_columns_are_not_imputed_and_empty_skips_read(tmp_path):
    path = tmp_path / "master.parquet"
    pd.DataFrame(dict(subject_id=[1], study_id=[10], split=["train"])).to_parquet(path)
    pairs = pd.DataFrame(dict(subject_id=[1], first_study_id=[10]))
    with duckdb.connect() as con:
        result = attach_baseline_context(con, pairs, path)
        assert result.baseline_adjust_sex_male.isna().all()
        assert result.baseline_adjust_age.isna().all()
        assert attach_baseline_context(con, pd.DataFrame(), tmp_path/"absent").empty
        with pytest.raises(ValueError, match="overwrite"):
            attach_baseline_context(con, result, path)
