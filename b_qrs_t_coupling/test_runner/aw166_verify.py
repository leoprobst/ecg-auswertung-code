"""Independent check of aggregate and key figures and identifier-leak protection for the measurement-artefact sensitivity analysis."""
def run():
    import json,re
    import numpy as np,pandas as pd,pyarrow.parquet as pq,statsmodels.api as sm
    from werkzeuge.b_test.aw166_run import OUT,ROOT,dump,sha
    from werkzeuge.b_test.statistik import raw_design,fixed_design
    p=OUT/'population_private.parquet';pq.ParquetFile(p);f=pd.read_parquet(p)
    assert set(f.split)=={'train','validation'} and not f.subject_id.duplicated().any()
    meta=pq.ParquetFile(OUT/'meta_private.parquet').read(columns=['subject_id','study_id','split']).to_pandas()
    assert len(meta.merge(f[['subject_id','study_id','split']],on=['subject_id','study_id','split'],validate='one_to_one'))==len(f)
    cal=json.loads((OUT/'calibration.json').read_text());assert round(cal['group_difference_ms']*.5)==0
    assert round(max(abs(v) for v in cal['group_difference_ci95_ms'])*.5)==0
    entry=json.loads((OUT/'prepare_manifest.json').read_text())['entry'];models=json.loads((OUT/'model_results.json').read_text())
    checks=[]
    gate=json.loads((OUT/'signal_gate.json').read_text());records={}
    if gate['state']=='passed':
        for cp in (OUT/'signal_records').glob('*.json'):
            r=json.loads(cp.read_text());assert r['equivalent'];records[r['study_id']]=r
        assert len(records)==len(f)
    for fold in ['train','validation']:
        ff=f[f.split.eq(fold)].copy()
        if records:
            v=np.array([records[s]['values']['corrected'] for s in ff.study_id],float);good=np.isfinite(v).all(1);ff=ff[good];v=v[good]
        y=ff.preferred_lvef.lt(40).astype(int).to_numpy()
        x,_=raw_design(ff,'qrs_t_normalized_discrete_frechet');z=fixed_design(x,entry['scaling_B3'])
        original=sm.GLM(y,z,family=sm.families.Binomial()).fit().params[-1]
        rr=next(r for r in models if r['fold']==fold and r['scenario']=='original')
        assert abs(original-rr['log_or'])<1e-5
        checks.append({'fold':fold,'scenario':'original','independent_GLM_beta':float(original),'solver_delta':float(original-rr['log_or'])})
        if records:
            x[:,-1]=v[:,0];x[:,-2]=v[:,1];z=fixed_design(x,entry['scaling_B3']);beta=sm.GLM(y,z,family=sm.families.Binomial()).fit().params[-1]
            rr=next(r for r in models if r['fold']==fold and r['scenario']=='corrected');assert abs(beta-rr['log_or'])<1e-5
            assert np.isclose(rr['remaining_log_or_fraction'],rr['log_or']/next(r['log_or'] for r in models if r['fold']==fold and r['scenario']=='original'))
            checks.append({'fold':fold,'scenario':'corrected','independent_GLM_beta':float(beta),'solver_delta':float(beta-rr['log_or'])})
    for r in models:
        assert r['n']==r['cases']+r['controls']
        if r['scenario'] in ['group_injected','worst_plus','worst_minus']:assert r['remaining_log_or_fraction']==1
        if r['bootstrap_requested']:assert (r['or_ci95'] is not None)==(r['bootstrap_valid']>=475)
    identifiers=set(map(str,f.subject_id))|set(map(str,f.study_id))
    project_files=list((ROOT/'werkzeuge/b_test').glob('aw166_*'))+[ROOT/'nachrechnung/B_TENDE_EF_AW166.md',ROOT/'nachrechnung/b_tende_ef.csv',ROOT/'koordination/abgaben/AW-166.md']
    leaks=[]
    for pp in project_files:
        if not pp.is_file():continue
        text=pp.read_text(encoding='utf8')
        found=set(re.findall(r'(?<![A-Za-z0-9_])\d{7,}(?![A-Za-z0-9_])',text))&identifiers
        if found:leaks.append({'path':str(pp),'identifiers':sorted(found)})
    dump('privacy_check_private.json',{'leaks':leaks});assert not leaks
    result={'state':'passed','independent_GLM':checks,'all_original_signal_equivalence':gate,'privacy_files_checked':len(project_files),'identifier_matches':0,'test_access':False}
    dump('verification.json',result)
    report_manifest=json.loads((OUT/'report_manifest.json').read_text());assert all(sha(p)==h for p,h in report_manifest['outputs'].items())
