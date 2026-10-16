"""Frank P95, unchanged adjudication labels, complete pair-flow counts."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
from repo_config import check_reference, reference
from aw149_common import *
import duckdb, pyarrow.dataset as ds, gc
def frank():
    p=MA/'frank/winkel_detail_privat.csv';f=pd.read_csv(bind(p));f=f[f['mode'].eq('integral')&f.group.eq('Healthy control')&f.method.isin(['kors','inverse_dower'])&np.isfinite(f.frank)&np.isfinite(f.reconstruction)]
    rows=[]
    for method,a in f.groupby('method'):
        assert len(a)==51 and a.patient.nunique()==51
        v=np.abs(a.difference.to_numpy());rng=np.random.default_rng(149007)
        bb=np.quantile(v[rng.integers(0,len(v),(10000,len(v)))],.95,axis=1);q=np.quantile(v,.95)
        rows.append({'method':method,'n_persons':len(v),'p95_abs_deviation_deg':q,'p95_ci_low':np.quantile(bb,.025),'p95_ci_high':np.quantile(bb,.975),'observations_above_p95':int((v>q).sum()),'observations_equal_p95':int((v==q).sum()),'draws':10000,'seed':149007,'quantile_method':'linear','source':str(p),'filter':'mode=integral; group=Healthy control','script':'werkzeuge/rv_a1/aw149_auxiliary.py'})
    pd.DataFrame(rows).to_csv(OUT/'frank_p95.csv',index=False)
def labels():
    p=MA/'p_ende/adjudikation_privat.csv';f=pd.read_csv(bind(p));d=json.loads(bind(MA/'p_ende/bildentscheidungen.json').read_text(encoding='utf8'))
    checks=[]
    for r in f.to_dict('records'):
        for field in ['primary_category','error','onset_error','end_error']:
            stored=d[str(r['audit_nr'])][field];current=r[field]
            equal=(stored is None and pd.isna(current)) or stored==current
            assert equal
    long=f[f.stratum.eq('lang')];both=long[long.onset_error.eq(True)&long.end_error.eq(True)]
    check_reference("aw149_auxiliary.n_double_label", len(both), what='number of cases carrying both boundary error labels');assert both.primary_category.eq('P_Ende_in_PR').all()
    result={'n_selected':len(f),'n_long':len(long),'primary_onset_category':int(long.primary_category.eq('Grundlinie_P_Beginn').sum()),'onset_error_true':int(long.onset_error.eq(True).sum()),'end_error_true':int(long.end_error.eq(True).sum()),'any_error_true':int(long.error.eq(True).sum()),'both_boundary_error':len(both),'decision_csv_identical':True,'additional_onset_in_end_category':reference("aw149_auxiliary.additional_onset_in_end_category"),'explanation':'Primary-category P-onset cases plus the P-end-in-PR cases carrying both boundary error labels make up the onset errors; each affected case is counted once, not twice.'}
    dump(OUT/'adjudication_reconciliation.json',result)
    both[['audit_nr','subject_id','study_id','primary_category','onset_error','end_error','comment']].to_csv(OUT/'double_label_privat.csv',index=False)
def flow():
    m=read(MA/'master_dev_privat.parquet',['subject_id','study_id','split','ecg_time','hadm_id','age','sex'])
    assert set(m.split)<={'train','validation'} and not m.duplicated(['subject_id','study_id']).any()
    # Read only explicit development split projection from shared master, never test values.
    p=Path(str(data_path("a.mimic_validation_master_ecg_master_index_parquet")));pf=pq.ParquetFile(p);bind(p)
    ed=ds.dataset(str(p),format='parquet').to_table(columns=['subject_id','study_id','split','ed_stay_id'],filter=ds.field('split').isin(['train','validation'])).to_pandas()
    assert set(ed.split)<={'train','validation'}
    m=m.merge(ed,on=['subject_id','study_id','split'],validate='one_to_one');del ed
    ff=[]
    for fold in ['train','validation']:
        p=next((RUN/'analysis_feature_cache').glob('*/split='+fold+'/features.parquet'));pf=pq.ParquetFile(p);bind(p)
        ff.append(pf.read(columns=['subject_id','study_id','extraction_status','signal_quality_score','signal_sha256']).to_pandas().assign(split=fold))
    features=pd.concat(ff,ignore_index=True);del ff
    assert not features.duplicated(['subject_id','study_id','split']).any()
    ok=features[features.extraction_status.eq('ok')];successful=set(ok.subject_id)
    assert len(successful)==128947 and len(ok)==632530
    m=m.sort_values(['subject_id','ecg_time','study_id']);mt=m[m.ecg_time.notna()].copy()
    nextrec=mt.groupby('subject_id').shift(-1)
    a=mt.rename(columns={c:'first_'+c for c in mt if c not in ['subject_id','split']})
    for c in ['study_id','ecg_time','hadm_id','ed_stay_id']:a['second_'+c]=nextrec[c]
    a=a[a.second_study_id.notna()].copy();a['hours_between']=(a.second_ecg_time-a.first_ecg_time).dt.total_seconds()/3600
    rows=[];previous=len(successful)
    def add(step,persons,pairs,reason):
        nonlocal previous
        n=len(persons);rows.append({'step':step,'n_persons':n,'n_pairs':pairs,'persons_lost_previous':previous-n,'reason':reason,'source':'development master + train/validation feature cache; release trajectory.py','script':'werkzeuge/rv_a1/aw149_auxiliary.py','run':RUN.name});previous=n
    add('successful_extracted_persons',successful,None,'at least one recording with extraction_status=ok')
    counts=m[m.hadm_id.notna()].groupby(['subject_id','hadm_id']).size();multi=set(counts[counts>=2].index.get_level_values('subject_id'))&successful
    add('at_least_two_same_admission',multi,None,'at least two master ECGs in the same non-missing admission; at least one successful extraction per person')
    a=a[a.subject_id.isin(multi)&a.first_hadm_id.notna()&a.first_hadm_id.eq(a.second_hadm_id)]
    add('adjacent_same_admission',set(a.subject_id),len(a),'adjacency in the complete time-ordered development master, not new adjacency within an admission')
    a=a[a.hours_between.between(0,2)]
    add('gap_0_to_2h',set(a.subject_id),len(a),'0 to 2 h including zero, as in the original pairing policy')
    a=a[a.first_ed_stay_id.fillna(-1).eq(a.second_ed_stay_id.fillna(-1))]
    add('same_setting',set(a.subject_id),len(a),'COALESCE(ed_stay_id,-1) identical')
    orig=read(MA/'repeatability/paargruppen_privat.parquet',['subject_id','split','first_study_id','second_study_id','intervening_intervention_checked'])
    checked=bool(orig.intervening_intervention_checked.all())
    if checked:
        regpath=Path(str(data_path("a.mimic_validation_standardized_clinical_events_versions_schema_1_2_0_922758943bbed5d1_clinical_event_registry_json")))
        registry=json.loads(bind(regpath).read_text(encoding='utf8'))
        for table in ['procedureevents','inputevents']:
            record=next(r for r in registry['tables'] if r['table_id']==table and r['status']=='ok')
            paths=sorted(regpath.parent.glob(record['relative_glob']))
            assert paths
            for p in paths:pq.ParquetFile(p);bind(p)
            events=ds.dataset([str(p) for p in paths],format='parquet').to_table(columns=['subject_id','starttime'],filter=ds.field('subject_id').isin(a.subject_id.unique().tolist())).to_pandas()
            assert set(events.subject_id)<=set(a.subject_id)
            eventdict={sub:g.starttime.dropna().sort_values().to_numpy(dtype='datetime64[ns]') for sub,g in events.groupby('subject_id')}
            keep=[]
            for r in a.itertuples():
                times=eventdict.get(r.subject_id,np.empty(0,dtype='datetime64[ns]'))
                left=np.searchsorted(times,np.datetime64(r.first_ecg_time),'right');right=np.searchsorted(times,np.datetime64(r.second_ecg_time),'left')
                keep.append(right<=left)
            a=a[np.array(keep)]
            add('no_'+table+'_start_between',set(a.subject_id),len(a),'no documented '+table+' start strictly between the two ECGs; projected to development persons only; no global event freedom')
            del events,eventdict;gc.collect()
    else:
        add('intervention_check_unavailable',set(a.subject_id),len(a),'intervening_intervention_checked=False; no documented event-freedom filter')
    h=features.set_index(['subject_id','study_id','split']);keys=['subject_id','split','first_study_id','second_study_id']
    for side in ['first','second']:
        ix=pd.MultiIndex.from_arrays([a.subject_id,a[side+'_study_id'],a.split]);v=h.reindex(ix)
        for c in ['extraction_status','signal_quality_score','signal_sha256']:a[side+'_'+c]=v[c].to_numpy()
    equal=a.first_signal_sha256.notna()&a.first_signal_sha256.eq(a.second_signal_sha256)
    before=len(a);a=a[~equal];removed=before-len(a)
    add('distinct_stored_signal',set(a.subject_id),len(a),'identical non-missing signal hashes excluded; before the extraction/quality gate')
    a=a[a.first_extraction_status.eq('ok')&a.second_extraction_status.eq('ok')]
    add('both_extraction_ok',set(a.subject_id),len(a),'both extraction_status=ok; missing feature row or error excluded')
    a=a[a.first_signal_quality_score.ge(.6)&a.second_signal_quality_score.ge(.6)]
    add('both_quality_ge_06',set(a.subject_id),len(a),'both signal qualities >=0.6; missing quality excluded')
    a.second_study_id=a.second_study_id.astype('int64')
    actual=set(map(tuple,a[keys].to_numpy()));expected=set(map(tuple,orig[keys].to_numpy()))
    comparison={'same_final_keys':actual==expected,'new_only':len(actual-expected),'original_only':len(expected-actual),'final_pairs':len(a),'final_subjects':a.subject_id.nunique(),'duplicate_signals_removed_in_successful_person_branch':removed,'intervention_checked':checked}
    dump(OUT/'flow_key_comparison.json',comparison)
    assert actual==expected and len(a)==15693 and a.subject_id.nunique()==10187
    a[keys].to_parquet(OUT/'flow_final_pairs_privat.parquet',index=False)
    pd.DataFrame(rows).to_csv(OUT/'cohort_flow.csv',index=False)
    # Compare persons selected for repeats vs successful extracted persons without repeats.
    index=m[m.study_id.isin(ok.study_id)].sort_values(['ecg_time','study_id']).drop_duplicates('subject_id')
    demo=[]
    for pop,mask in [('with_repeat',index.subject_id.isin(a.subject_id)),('without_repeat',~index.subject_id.isin(a.subject_id))]:
        b=index[mask];age=b.age.dropna();demo.append({'population':pop,'n_persons':len(b),'age_median':age.median(),'age_p25':age.quantile(.25),'age_p75':age.quantile(.75),'female_n':int(b.sex.eq('F').sum()),'female_percent':100*b.sex.eq('F').mean(),'age_missing':int(b.age.isna().sum()),'sex_missing':int(b.sex.isna().sum()),'index':'earliest successful recording; age at the first repeat differs from the earlier analysis','setting_comparison':'ward unavailable; ed_stay_id is not a ward identifier'})
    pd.DataFrame(demo).to_csv(OUT/'repeat_selection_demography.csv',index=False)
def main():
    frank();labels();flow()
    finish('auxiliary',[OUT/p for p in ['frank_p95.csv','adjudication_reconciliation.json','cohort_flow.csv','flow_key_comparison.json','repeat_selection_demography.csv']])
    print('AUXILIARY_DONE',flush=True)
if __name__=='__main__':main()
