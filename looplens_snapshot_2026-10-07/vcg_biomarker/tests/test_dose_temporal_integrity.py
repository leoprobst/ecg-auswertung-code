import duckdb
import pandas as pd
import pytest
from vcg_biomarker.mimic.within_subject import dose_pair_query, dose_pairs_all_classes_query, DRUG_WINDOW


@pytest.mark.parametrize("combined", [False,True])
@pytest.mark.parametrize("status", ["FinishedRunning", "ChangeDose/Rate", " changedose/rate ", "Rewritten", "unknown", "Bolus", "Reported"])
def test_actual_queries_require_complete_same_admission_exposure(tmp_path,combined,status):
    start=pd.Timestamp("2020-01-01")
    master=[]
    inputs=[]
    for patient,(begin,end) in enumerate([(1,1.5),(1,3),(-1,1),(1,None),(1,.5),(1,2),(1,1.5),(2,3)],1):
        for index,hour in enumerate((0,2)):
            master.append(dict(subject_id=patient,hadm_id=100+patient,study_id=patient*10+index,
                split="train",ecg_time=start+pd.Timedelta(hours=hour)))
        inputs.append(dict(subject_id=patient,hadm_id=999 if patient==7 else 100+patient,
            starttime=start+pd.Timedelta(hours=begin),
            endtime=start+pd.Timedelta(hours=end) if end is not None else pd.NaT,
            itemid=1,amount=5.,amountuom="mg",statusdescription=status,orderid=patient,linkorderid=patient))
    frames={"master":pd.DataFrame(master),"inputs":pd.DataFrame(inputs),
            "items":pd.DataFrame(dict(itemid=[1],label=["Metoprolol"]))}
    frames["features"]=frames["master"][["subject_id","study_id"]].assign(extraction_status="ok",marker=1.,signal_quality_score=.9)
    paths={}
    for name,frame in frames.items():
        path=tmp_path/f"{name}.parquet"
        frame.to_parquet(path,index=False)
        paths[name]=path.as_posix()
    args=(paths["master"],paths["inputs"],paths["items"],f"read_parquet('{paths['features']}')")
    if combined:
        query=dose_pairs_all_classes_query(*args,window=DRUG_WINDOW,marker_ids=("marker",),class_patterns={"beta":"metoprolol"})
    else:
        query=dose_pair_query(*args,window=DRUG_WINDOW,marker_columns="b.marker-a.marker AS delta_marker",label_pattern="metoprolol")
    with duckdb.connect() as con:
        result=con.execute(query).df().set_index("subject_id")
    if status == "Rewritten":
        assert result.empty
        return
    assert set(result.index)==set(range(1,7))
    if status in {"unknown", "Bolus", "Reported"}:
        assert result.dose_amount.isna().all()
        assert result.dose_aggregation_status.eq("unknown_or_unusable_administration_status").all()
        return
    assert result.loc[1,"dose_amount"]==5.
    assert result.loc[1,"dose_aggregation_status"]=="eligible"
    # SQL without ORDER BY has no row-order contract. Label slicing on an
    # unsorted index could include patient 1 and omit invalid patients.
    assert result.loc[list(range(2, 7)), "dose_amount"].isna().all()
    assert result.loc[list(range(2, 7)), "dose_aggregation_status"].eq("unbounded_or_invalid_exposure_interval").all()
