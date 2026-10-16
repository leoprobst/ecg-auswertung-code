import duckdb
import numpy as np
import pandas as pd
import pytest
from vcg_biomarker.mimic.within_subject import _dose_aggregate_columns, normalise_dose_within_drug


@pytest.mark.parametrize("items,units", [([1,2],["mg","mg"]),([1,1],["mg","mcg"]),
    ([1,1],["mg",None]),([1,1],["mg"," "])])
def test_sql_never_sums_incompatible_administrations(items,units):
    frame=pd.DataFrame(dict(itemid=items,amountuom=units,amount=[1.,1000.],label="drug"))
    with duckdb.connect() as con:
        con.register("doses",frame)
        row=con.execute("SELECT "+_dose_aggregate_columns()+" FROM doses x").df().iloc[0]
    assert pd.isna(row.dose_amount) and pd.isna(row.dose_itemid)
    assert row.dose_aggregation_status == "mixed_drug_or_unit_or_missing_unit"
    assert row.n_administrations == 2


def test_sql_valid_dose_and_cosmetic_unit_normalization():
    frame=pd.DataFrame(dict(itemid=[1,1],amountuom=[" mg ","MG"],amount=[1.,2.],label="drug"))
    with duckdb.connect() as con:
        con.register("doses",frame)
        row=con.execute("SELECT "+_dose_aggregate_columns()+" FROM doses x").df().iloc[0]
    assert row.dose_amount == 3. and row.dose_unit == "mg"


def test_rank_separates_units_and_excludes_invalid_doses_from_denominator():
    frame=pd.DataFrame(dict(dose_itemid=1,dose_unit=["mg"]*4+["mcg"]*2,
        dose_amount=[1.,2.,np.inf,0.,1000.,2000.]))
    result=normalise_dose_within_drug(frame,minimum_administrations=2)
    np.testing.assert_allclose(result.dose_rank.iloc[[0,1,4,5]],[.5,1.,.5,1.])
    assert result.dose_rank.iloc[[2,3]].isna().all()
    assert result.dose_drug_n.eq(2).all()
    missing=normalise_dose_within_drug(frame.drop(columns="dose_unit"),minimum_administrations=2)
    assert missing.dose_rank.isna().all()
