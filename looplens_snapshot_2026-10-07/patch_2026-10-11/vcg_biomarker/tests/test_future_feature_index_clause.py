"""Missing read clause must preserve the intended Future-PheWAS index join."""
from types import SimpleNamespace
import duckdb
import pandas as pd
import pytest
from vcg_biomarker.mimic.future_phenome import _load_future_feature_index


def inputs(tmp_path,heart_rate=True,duplicate=False):
    idx=pd.DataFrame(dict(subject_id=[2,1,3],study_id=[20,10,30],split=['validation','train','test'],
        age=[50.,60.,70.],sex=['F','M','F'],anchor_year_group=['2017 - 2019']*3,
        machine_heart_rate_bpm=[75.,66.,70.],ed_stay_id=pd.array([1,None,2],dtype='Int64'),
        ecg_to_landmark_days=[0.,1.,2.],index_length_of_stay_days=[2.,3.,4.]))
    for c in ('ecg_time','index_end_time','last_observed_time'):
        idx[c]=pd.to_datetime(['2020-01-01']*3)
    idx['date_of_death']=pd.to_datetime([None,None,'2021-01-01'])
    features=pd.DataFrame(dict(subject_id=[2,1,3],study_id=[20,10,30],split=['validation','train','test'],
        extraction_status=['failed','ok','ok'],signal_quality_score=[None,.9,.8]))
    if heart_rate: features['heart_rate_bpm']=[80.,61.,70.]
    if duplicate:
        features=pd.concat([features,features.iloc[:1]],ignore_index=True)
    ip=tmp_path/'index.parquet'; fp=tmp_path/'features.parquet'
    idx.to_parquet(ip,index=False); features.to_parquet(fp,index=False)
    config=SimpleNamespace(catalog_path=tmp_path/'catalog.duckdb',duckdb_memory_limit_gb=1,workers=1)
    return config,ip,fp


@pytest.mark.parametrize('heart_rate',[True,False])
def test_feature_index_matches_explicit_join_with_and_without_comparator(tmp_path,heart_rate):
    config,ip,fp=inputs(tmp_path,heart_rate)
    actual=_load_future_feature_index(config,feature_glob=str(fp),index_path=ip)
    columns=['subject_id','study_id','split','ecg_time','index_end_time','last_observed_time','date_of_death',
             'age','sex','anchor_year_group','machine_heart_rate_bpm','ed_stay_id','ecg_to_landmark_days','index_length_of_stay_days']
    selection=', '.join('i.'+c for c in columns)+', f.extraction_status, f.signal_quality_score'
    if heart_rate: selection+=', f.heart_rate_bpm'
    with duckdb.connect(':memory:') as db:
        expected=db.execute(f"SELECT {selection} FROM read_parquet(?) i JOIN read_parquet(?) f "
            "USING(subject_id,study_id,split) WHERE i.split IN ('train','validation') "
            "ORDER BY i.split,i.subject_id,i.study_id",[str(ip),str(fp)]).fetchdf()
    pd.testing.assert_frame_equal(actual,expected,check_exact=True)
    assert actual.subject_id.tolist()==[1,2]  # Locked test is excluded; failed quality is retained for policy.


def test_duplicate_subject_guard_is_unchanged(tmp_path):
    config,ip,fp=inputs(tmp_path,duplicate=True)
    with pytest.raises(RuntimeError,match='repeated subjects'):
        _load_future_feature_index(config,feature_glob=str(fp),index_path=ip)
