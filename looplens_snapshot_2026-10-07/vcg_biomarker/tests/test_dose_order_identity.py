import duckdb
import pandas as pd
import pytest
from vcg_biomarker.mimic.within_subject import _dose_identity_ctes, _dose_aggregate_columns


@pytest.mark.parametrize("classified", [False,True])
@pytest.mark.parametrize("variant", ["copy","conflict","segment","unknown"])
def test_order_identity_prevents_double_counting_without_merging_segments(classified,variant):
    origin=pd.Timestamp("2020-01-01")
    first=dict(subject_id=1,hadm_id=10,orderid=123,linkorderid=120,itemid=1,
        starttime=origin+pd.Timedelta(hours=1),endtime=origin+pd.Timedelta(hours=1.5),
        amount=5.,amountuom="mg",statusdescription="FinishedRunning",label="drug")
    second=dict(first)
    if variant=="conflict": second["amount"]=6.
    elif variant=="segment":
        second.update(starttime=first["endtime"],endtime=origin+pd.Timedelta(hours=1.75))
    elif variant=="unknown": first["orderid"]=second["orderid"]=0
    doses=pd.DataFrame([first,second])
    if classified: doses["drug_class"]="beta"
    candidate=pd.DataFrame(dict(first_ecg_time=[origin],second_ecg_time=[origin+pd.Timedelta(hours=2)]))
    ctes=_dose_identity_ctes(classified=classified).rstrip(",")
    with duckdb.connect() as con:
        con.register("dosed",doses)
        con.register("candidate",candidate)
        row=con.execute(f"WITH {ctes} SELECT {_dose_aggregate_columns(bounded=True)} FROM dose_identity x CROSS JOIN candidate c").df().iloc[0]
    assert row.dose_source_rows==2
    if variant in {"conflict","unknown"}:
        assert pd.isna(row.dose_amount)
        assert row.dose_aggregation_status=="unknown_or_conflicting_order_identity"
    else:
        assert row.dose_amount==(5. if variant=="copy" else 10.)
        assert row.n_administrations==(1 if variant=="copy" else 2)
        assert row.dose_aggregation_status=="eligible"
