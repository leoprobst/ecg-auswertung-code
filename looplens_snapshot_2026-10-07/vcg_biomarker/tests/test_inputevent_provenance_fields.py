import duckdb
import pandas as pd
import pytest

from vcg_biomarker.mimic import event_ingest
from vcg_biomarker.tests.test_provenance_integrity import _config


def input_source(tmp_path):
    config=_config(tmp_path)
    spec=next(s for s in event_ingest._clinical_event_specs(config) if s.table_id=="inputevents")
    values=dict(subject_id="1",hadm_id="10",stay_id="100",itemid="225158",
        starttime="2020-01-01 01:00",endtime="2020-01-01 02:00",
        amount="5",amountuom="mg",rate="5",rateuom="mg/hour",
        storetime="2020-01-01 03:00",orderid="123",linkorderid="120",
        statusdescription=" FinishedRunning ",ordercategoryname="Continuous IV",
        ordercategorydescription="Continuous",ordercomponenttypedescription="Main order parameter")
    source=pd.DataFrame([values,{**values,"orderid":"124","statusdescription":"Rewritten",
        "storetime":"","linkorderid":"invalid","ordercategorydescription":" "}])
    spec.source_path.parent.mkdir(parents=True,exist_ok=True)
    source.to_csv(spec.source_path,index=False,compression="gzip")
    return config,spec,source


def test_actual_ingest_preserves_status_orders_and_documentation_time(tmp_path,monkeypatch):
    config,spec,source=input_source(tmp_path)
    monkeypatch.setattr(event_ingest,"_clinical_event_specs",lambda _: (spec,))
    original=spec.source_path.read_bytes()
    manifest=event_ingest.ingest_clinical_event_tables(config)
    relation=event_ingest.clinical_event_relation(config,"inputevents")
    with duckdb.connect() as con:
        rows=con.execute(f"SELECT * FROM {relation} ORDER BY orderid").df()
    assert len(rows)==2  # keep audit records; consumers must decide eligibility
    assert rows.statusdescription.tolist()==["FinishedRunning","Rewritten"]
    assert rows.orderid.tolist()==[123,124]
    assert rows.storetime.iloc[0] > rows.endtime.iloc[0]
    assert pd.isna(rows.storetime.iloc[1]) and pd.isna(rows.linkorderid.iloc[1])
    assert pd.isna(rows.ordercategorydescription.iloc[1])
    assert spec.source_path.read_bytes()==original
    again=event_ingest.ingest_clinical_event_tables(config)
    assert again.input_hash==manifest.input_hash


def test_missing_status_column_rejected_not_invented(tmp_path,monkeypatch):
    config,spec,source=input_source(tmp_path)
    source.drop(columns="statusdescription").to_csv(spec.source_path,index=False,compression="gzip")
    monkeypatch.setattr(event_ingest,"_clinical_event_specs",lambda _: (spec,))
    with pytest.raises(RuntimeError,match="statusdescription"):
        event_ingest.ingest_clinical_event_tables(config)
