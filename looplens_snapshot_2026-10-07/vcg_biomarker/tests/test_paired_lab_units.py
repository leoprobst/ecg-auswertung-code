import duckdb
import pandas as pd
import pytest

from vcg_biomarker.mimic.within_subject import (
    DIALYSIS_LAB_ANALYTES, LabAnalyte, _analyte_units, nearest_labs_query,
)


@pytest.mark.parametrize("analyte_id,itemid,unit,value", [
    ("potassium", 50971, "mEq/L", 4.),
    ("potassium", 50971, "mmol/L", 4.),
    ("calcium_total", 50893, "mg/dL", 9.),
    ("calcium_ionized", 50808, "mmol/L", 1.2),
    ("magnesium", 50960, "mg/dL", 2.),
    ("ph", 50820, "units", 7.4),
])
def test_valid_numeric_value_in_wrong_unit_cannot_pass(tmp_path, analyte_id, itemid, unit, value):
    origin = pd.Timestamp("2020-01-01")
    master = pd.DataFrame(dict(subject_id=[1, 2], study_id=[1, 2], hadm_id=[1, 2],
        ecg_time=origin, split="train"))
    labs = pd.DataFrame(dict(subject_id=[1, 2, 2], hadm_id=[1, 2, 2], itemid=itemid,
        charttime=[origin, origin, origin+pd.Timedelta(hours=1)],
        valuenum=value, valueuom=[f" {unit.upper()} ", "unrecognized_scale", unit],
        comments=None))
    mp, lp = tmp_path/"master.parquet", tmp_path/"labs.parquet"
    master.to_parquet(mp, index=False)
    labs.to_parquet(lp, index=False)
    with duckdb.connect() as con:
        result = con.execute(nearest_labs_query(mp.as_posix(), lp.as_posix())).df().set_index("subject_id")
    assert result.analyte_id.eq(analyte_id).all()
    assert result.loc[1, "value"] == value
    assert result.loc[1, "lab_reference_status"] == "eligible"
    assert pd.isna(result.loc[2, "value"])
    assert result.loc[2, "lab_reference_status"] == "unit_not_supported_for_analyte"
    assert result.loc[2, "delta_hours"] == 0  # no fallback to farther valid unit


def test_uses_existing_evidence_policy_without_a_second_unit_dictionary():
    from vcg_biomarker.mimic.evidence import LAB_EXPECTED_UNITS
    aliases = {"calcium_total": "total_calcium", "calcium_ionized": "ionized_calcium", "ph": "blood_gas_ph"}
    for analyte in DIALYSIS_LAB_ANALYTES:
        assert _analyte_units(analyte) == LAB_EXPECTED_UNITS[aliases.get(analyte.analyte_id, analyte.analyte_id)]


def test_new_analyte_requires_explicit_policy():
    unknown = LabAnalyte("custom", "Custom", (1,), (0., 1.), "display-only")
    with pytest.raises(ValueError, match="accepted units"):
        nearest_labs_query("m", "l", analytes=(unknown,))
    explicit = LabAnalyte("custom", "Custom", (1,), (0., 1.), "display-only", (" mg/L ",))
    assert _analyte_units(explicit) == ("mg/l",)
