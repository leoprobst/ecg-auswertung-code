import duckdb
import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.within_subject import nearest_lab_query, nearest_labs_query


@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("variant", ["valid", "copy", "conflict", "different_unit",
    "different_assay", "missing", "infinite", "missing_unit", "wrong_admission", "unknown_admission", "equidistant",
    "hemolyzed", "not_hemolyzed"])
def test_nearest_actual_reference_never_silently_falls_back(tmp_path, single, variant):
    origin = pd.Timestamp("2020-01-01")
    master = pd.DataFrame(dict(subject_id=[1, 1], hadm_id=[10, 10],
        study_id=[100, 101], split=["train", "locked_test"], ecg_time=origin))
    nearest = dict(subject_id=1, hadm_id=10, itemid=50971,
        charttime=origin+pd.Timedelta(minutes=1), valuenum=4., valueuom="mmol/L",
        comments=None)
    far = {**nearest, "charttime": origin+pd.Timedelta(minutes=30), "valuenum": 5.}
    rows = [nearest, far]
    if variant == "copy": rows.append(dict(nearest))
    elif variant == "conflict": rows.append({**nearest, "valuenum": 6.})
    elif variant == "different_unit": rows.append({**nearest, "valueuom": "mg/dL"})
    elif variant == "different_assay": rows.append({**nearest, "itemid": 50822})
    elif variant == "missing": nearest["valuenum"] = None
    elif variant == "infinite": nearest["valuenum"] = np.inf
    elif variant == "missing_unit": nearest["valueuom"] = None
    elif variant == "wrong_admission": nearest["hadm_id"] = 20
    elif variant == "unknown_admission": nearest["hadm_id"] = None
    elif variant == "hemolyzed": nearest["comments"] = "HEMOLYSIS FALSELY ELEVATES K."
    elif variant == "not_hemolyzed": nearest["comments"] = "SPECIMEN NOT HEMOLYZED."
    elif variant == "equidistant": rows.append({**nearest, "charttime": origin-pd.Timedelta(minutes=1), "valuenum": 3.})
    master_path, lab_path = tmp_path/"master.parquet", tmp_path/"labs.parquet"
    master.to_parquet(master_path, index=False)
    pd.DataFrame(rows).to_parquet(lab_path, index=False)
    args = (master_path.as_posix(), lab_path.as_posix())
    query = nearest_lab_query(*args, itemids=(50971, 50822)) if single else nearest_labs_query(*args)
    with duckdb.connect() as con:
        result = con.execute(query).df()
    assert len(result) == 1 and result.study_id.iloc[0] == 100
    row = result.iloc[0]
    if variant in {"conflict", "different_unit", "different_assay"}:
        assert pd.isna(row.value)
        assert row.lab_reference_status == "conflicting_nearest_measurements"
    elif variant in {"missing", "infinite"}:
        assert pd.isna(row.value)
        assert row.lab_reference_status == "invalid_or_missing_nearest_value"
    elif variant == "missing_unit":
        assert pd.isna(row.value)
        assert row.lab_reference_status == "missing_nearest_unit"
    elif variant == "hemolyzed":
        # Audit D2: no fallback to the more distant, non-hemolyzed sample.
        assert pd.isna(row.value)
        assert row.lab_reference_status == "hemolysis_reported"
    else:
        expected = 5. if variant in {"wrong_admission", "unknown_admission"} else 3. if variant == "equidistant" else 4.
        assert row.value == expected and row.lab_reference_status == "eligible"
    if variant == "copy": assert row.lab_source_rows == 2


@pytest.mark.parametrize("window", [0, -1, np.inf, np.nan])
def test_invalid_windows_fail(window):
    with pytest.raises(ValueError): nearest_labs_query("m", "l", window_hours=window)


@pytest.mark.parametrize("variant", ["valid", "assay", "unit", "subject", "status", "no_metadata"])
def test_pair_requires_comparable_known_references(variant):
    from vcg_biomarker.mimic.within_subject import paired_change_table
    pairs = pd.DataFrame(dict(subject_id=[1], first_study_id=[100], second_study_id=[101],
        marker_first=[1.], marker_second=[2.]))
    labs = pd.DataFrame(dict(subject_id=[1, 1], study_id=[100, 101], value=[5., 4.],
        itemid=[50971, 50971], unit="mmol/l", lab_reference_status="eligible"))
    if variant == "assay": labs.loc[1, "itemid"] = 50822
    elif variant == "unit": labs.loc[1, "unit"] = "mg/dl"
    elif variant == "subject": labs.loc[1, "subject_id"] = 2
    elif variant == "status": labs.loc[1, "lab_reference_status"] = "conflicting_nearest_measurements"
    elif variant == "no_metadata": labs = labs[["study_id", "value"]]
    result = paired_change_table(pairs, labs, ("marker",)).iloc[0]
    assert result.delta_marker == 1.  # exclusion applies to the reference, not signal availability
    if variant == "valid":
        assert result.delta_lab == -1. and result.lab_pair_status == "eligible"
    else:
        assert pd.isna(result.delta_lab)
        assert result.lab_pair_status == (
            "different_assay_or_unit" if variant in {"assay", "unit"} else "reference_ineligible_or_missing")
