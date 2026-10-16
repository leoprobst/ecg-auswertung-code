"""Measurement-artefact sensitivity analysis: development partition only; private logs go to the local output folder."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os, sys, json, hashlib, time, ctypes
from pathlib import Path
os.environ.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
OUT=Path(str(data_path("b.aw166"))); OUT.mkdir(exist_ok=True)
sys.stdout=(OUT/'stdout.log').open('a',encoding='utf8',buffering=1)
sys.stderr=(OUT/'stderr.log').open('a',encoding='utf8',buffering=1)
def dump(name,v): (OUT/name).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str),encoding='utf8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def guard():
    class M(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ['total','free','page','commit','vtotal','vfree','extended']]
    m=M();m.length=ctypes.sizeof(m);ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    if min(m.free,m.commit)<1.5*1024**3:raise RuntimeError('WAIT_MEMORY')
    return {'ram_free':m.free,'commit_free':m.commit}
def signals():
    from werkzeuge.b_test.aw166_signals import run
    run()
def profile():
    import cProfile,pstats
    from werkzeuge.b_test.aw166_signals import run
    p=cProfile.Profile();p.runcall(run,5);p.dump_stats(str(OUT/'profile.prof'))
    pstats.Stats(p,stream=sys.stdout).sort_stats('cumtime').print_stats(35)
def diagnose():
    from werkzeuge.b_test.aw166_signals import run
    run(300)
def difference():
    import numpy as np
    d=json.loads((OUT/'fast_grid_difference_private.json').read_text());a=d['old']['shift_grid'];b={str(k):v for k,v in d['new']['shift_grid'].items()}
    keys=set(a)|set(b);summary={'key_counts':[len(a),len(b)],'missing_keys':len(set(a)^set(b)),'changed':[]}
    for k in keys:
        if k not in a or k not in b:continue
        if a[k]!=b[k]:summary['changed'].append({'sample':k,'old_null':[v is None for v in a[k]],'new_null':[v is None for v in b[k]],'delta':[None if x is None or y is None else y-x for x,y in zip(a[k],b[k])]})
    print(json.dumps(summary))
def casegate():
    import pandas as pd
    from werkzeuge.varianten.adapter import prepare_record,release_import,context
    from werkzeuge.b_stoertest.run import OPTIONS,shifted
    fr=release_import()
    from vcg_biomarker.core.biomarkers.catalog_biomarkers import _feature_segment_dependencies,measurement_confidence_threshold
    d=json.loads((OUT/'fast_grid_difference_private.json').read_text());m=pd.read_parquet(OUT/'meta_private.parquet');row=m[m.study_id.eq(d['old']['study_id'])].iloc[0].to_dict()
    p=prepare_record(row,OPTIONS);dd=shifted(p['delineation'],'t_offset',13);c=context(p,delineation=dd)
    name='qrs_t_normalized_discrete_frechet';b=fr.BiomarkerRegistry.get(name);deps=b.spec.dependencies
    rv=fr.VCGTransformRegistry.transform(p['paths']['morphology'],'kors').xyz;bv=fr.VCGTransformRegistry.transform(p['robust'],'kors').xyz
    v=fr.BiomarkerRegistry.compute_all(delineation=dd,ecg_12lead=p['robust'],vcg_xyz=bv,fs=p['fs'],names=[name],record_ecg_12lead=p['paths']['morphology'],record_vcg_xyz=rv,record_raw_ecg_12lead=p['loaded'].signal_millivolts,record_acquisition=p['acquisition'])[0]
    dump('case_gate_private.json',{'dependency':deps,'confidence':{s:c.segment_confidence(s) for s in deps},'threshold':{s:measurement_confidence_threshold(s) for s in deps},'note':v.computation_notes})
    print(json.dumps({'dependency':deps,'failed_dependencies':[s for s in deps if c.segment_confidence(s)<measurement_confidence_threshold(s)],'note_is_error':v.computation_notes.startswith('error:')}))
def models():
    from werkzeuge.b_test.aw166_models import run
    run()
def report():
    from werkzeuge.b_test.aw166_report import run
    run()
def verify():
    from werkzeuge.b_test.aw166_verify import run
    run()
def inventory():
    protocol=ROOT/'werkzeuge/b_test/aw166_protokoll.md'
    assert sha(protocol)=='c3da93c3b8922526b456470e80071104f60b9fa2a467b46027d33cbae924c148'
    dump('protokoll_hash.json',{'sha256':sha(protocol),'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'before_calculation':True})
    import pyarrow.parquet as pq
    paths=[Path(str(data_path("b.aw131_bef_0063_entwicklung_parquet")))]
    cache=Path(str(data_path("b.mimic_validation_runs_a155_2026_10_04_d6e70e_analysis_feature_cache_cache_89d6aaf0a6b5957e")))
    paths += [cache/f'split={s}/features.parquet' for s in ['train','validation']]
    schemas={str(p):{'columns':pq.ParquetFile(p).schema_arrow.names,'rows':pq.ParquetFile(p).metadata.num_rows,'bytes':p.stat().st_size} for p in paths}
    dump('schemas.json',schemas)
    print(json.dumps({p:[c for c in v['columns'] if any(t in c.lower() for t in ['amplit','duration','t_wave','noise','snr','t_peak','magnitude'])] for p,v in schemas.items()}))
def prepare():
    import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.dataset as ds
    from werkzeuge.b_test.daten import masks,classify_lvef
    from werkzeuge.b_stoertest.run import META
    entry=json.loads((ROOT/'werkzeuge/b_test/parameter_entwicklung.json').read_text(encoding='utf8'))['hypotheses']['BEF-0063']
    p=Path(entry['input_development']['path']); assert sha(p)==entry['input_development']['sha256']
    cols=['subject_id','study_id','split','age','sex','heart_rate','qrs_duration_ms','qtc_fridericia_ms','spatial_qrs_t_angle_deg','qrs_t_normalized_discrete_frechet','preferred_lvef','lvef_conflict_gt_10','nearest_tte_measurement_id','nearest_tte_delta_hours','label_state','eligibility_state','index_selection_strategy','time_precision','time_delta_hours','evidence_ids_json','ehr_time_alignment_confirmatory_eligible']
    pq.ParquetFile(p);f=ds.dataset(str(p),format='parquet').to_table(columns=cols).to_pandas()
    assert set(f.split)=={'train','validation'}
    f,g,flow=masks(f,'reduced_lvef_strict','qrs_t_normalized_discrete_frechet','qrs_duration_ms');f=f[g].copy();f['y']=classify_lvef(f.preferred_lvef)
    sources={str(p):{'sha256':sha(p),'columns':cols}}
    a=[]
    cache=Path(str(data_path("b.mimic_validation_runs_a155_2026_10_04_d6e70e_analysis_feature_cache_cache_89d6aaf0a6b5957e")))
    for fold in ['train','validation']:
        guard();p=cache/f'split={fold}/features.parquet';pq.ParquetFile(p)
        c=['subject_id','study_id','t_vector_magnitude','t_duration_ms']
        z=ds.dataset(str(p),format='parquet').to_table(columns=c,filter=ds.field('study_id').isin(f.loc[f.split.eq(fold),'study_id'].tolist())).to_pandas();z['split']=fold;a.append(z)
        sources[str(p)]={'sha256':sha(p),'columns':c,'filter':'H1-B3 index study keys, matching split'}
    f=f.merge(pd.concat(a),on=['subject_id','study_id','split'],how='left',validate='one_to_one')
    f['amplitude_mv']=f.t_vector_magnitude/1000
    f['flatness_mv_ms']=f.amplitude_mv/f.t_duration_ms.where(f.t_duration_ms.gt(0))
    f.to_parquet(OUT/'population_private.parquet',index=False)
    master=Path(str(data_path("b.mimic_validation_master_ecg_master_index_parquet")));pq.ParquetFile(master)
    meta=ds.dataset(str(master),format='parquet').to_table(columns=META,filter=ds.field('split').isin(['train','validation']) & ds.field('study_id').isin(f.study_id.tolist())).to_pandas()
    assert len(meta)==len(f) and not meta.study_id.duplicated().any()
    meta.to_parquet(OUT/'meta_private.parquet',index=False)
    sources[str(master)]={'sha256':sha(master),'columns':META,'filter':'train/validation AND H1-B3 index study keys'}
    dump('prepare_manifest.json',{'sources':sources,'flow':flow,'entry':entry,'memory':guard(),'test_access':False})
    print('prepared',len(f))
def morphology():
    import numpy as np,pandas as pd
    p=OUT/'population_with_snr_private.parquet'
    if not p.exists():p=OUT/'population_private.parquet'
    f=pd.read_parquet(p);rows=[]
    for fold in ['train','validation','development']:
        z=f if fold=='development' else f[f.split.eq(fold)]
        for measure in ['amplitude_mv','t_duration_ms','flatness_mv_ms']+(['t_snr'] if 't_snr' in f else []):
            v=z[measure].to_numpy(float);y=z.y.to_numpy(int);ok=np.isfinite(v);a=v[ok&(y==1)];b=v[ok&(y==0)]
            def smd(a,b):return (a.mean()-b.mean())/np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2)
            rng=np.random.default_rng(16620261013);boot=[];means=[];quant=[]
            # One index per person: row resampling is person resampling, including missing values.
            for r in range(500):
                ix=rng.integers(0,len(z),len(z));vv=v[ix];yy=y[ix];valid=np.isfinite(vv);aa=vv[valid&(yy==1)];bb=vv[valid&(yy==0)]
                boot.append(smd(aa,bb));means.append([aa.mean(),bb.mean()]);quant.append([np.quantile(aa,[.05,.25,.5,.75,.95]),np.quantile(bb,[.05,.25,.5,.75,.95])])
            rows.append({'fold':fold,'measure':measure,'n_total':len(z),'n_low':len(a),'n_high':len(b),'missing_low':int(((y==1)&~ok).sum()),'missing_high':int(((y==0)&~ok).sum()),'mean_low':a.mean(),'mean_high':b.mean(),'means_ci95':np.quantile(means,[.025,.975],axis=0).tolist(),'smd':smd(a,b),'smd_ci_low':np.quantile(boot,.025),'smd_ci_high':np.quantile(boot,.975),'quantiles_low':np.quantile(a,[.05,.25,.5,.75,.95]).tolist(),'quantiles_high':np.quantile(b,[.05,.25,.5,.75,.95]).tolist(),'quantiles_ci95':np.quantile(quant,[.025,.975],axis=0).tolist(),'bootstrap_valid':len(boot)})
    dump('morphology.json',rows);print('morphology',json.dumps(rows))
def smoke():
    import numpy as np,pandas as pd
    from werkzeuge.varianten.adapter import prepare_record,context
    from werkzeuge.b_stoertest.run import OPTIONS,compute
    f=pd.read_parquet(OUT/'population_private.parquet');m=pd.read_parquet(OUT/'meta_private.parquet')
    row=m.merge(f[['study_id','qrs_t_normalized_discrete_frechet','spatial_qrs_t_angle_deg']],on='study_id',validate='one_to_one').sort_values('study_id').iloc[0].to_dict()
    package=prepare_record(row,OPTIONS);base=compute(package,package['delineation'])
    dump('smoke_private.json',{'baseline':base,'expected':[row.get(k) for k in OPTIONS['vcg_feature_ids']],'waveform_exists':Path(row['waveform_path']).exists()})
    ok=np.isclose(base[0],row['qrs_t_normalized_discrete_frechet'],rtol=1e-8,atol=0) and np.isclose(base[3],row['spatial_qrs_t_angle_deg'],rtol=1e-8,atol=0)
    dump('smoke.json',{'equivalent':bool(ok),'n':1,'memory':guard()})
def ludb():
    import numpy as np,pandas as pd,pyarrow.parquet as pq,wfdb
    from werkzeuge.varianten.adapter import release_import,context
    fr=release_import()
    from vcg_biomarker.core.preprocessing import prepare_signal_paths
    data=Path(str(data_path("b.lobachevsky_university_electrocardiography_database_1_0_1_data")))
    base=Path(str(data_path("b.mimic_validation_runs_a155_2026_10_04_d6e70e_publication_preparation_research_beat_boundaries_20261012_v1")))
    records=[x.split('/')[-1] for x in (data.parent/'RECORDS').read_text().splitlines() if x]
    rows=[];sources={};(OUT/'ludb_records').mkdir(exist_ok=True)
    for i,record in enumerate(records,1):
        guard();cp=OUT/'ludb_records'/f'{i:03d}.json'
        if cp.exists():rows.append(json.loads(cp.read_text(encoding='utf8')));continue
        rec=wfdb.rdrecord(str(data/record));lead=['i','ii','iii','avr','avl','avf','v1','v2','v3','v4','v5','v6']
        signal=rec.p_signal[:,[[s.lower() for s in rec.sig_name].index(s) for s in lead]]
        paths=prepare_signal_paths(signal,rec.fs,powerline=50.)
        ecg=paths['delineation'];xyz=fr.VCGTransformRegistry.transform(ecg,'kors').xyz
        det=fr.FusionDelineator().delineate_record(ecg_12lead=ecg,r_peaks=fr.detect_r_peaks_multilead(ecg,rec.fs),fs=rec.fs,ecg_id=int(record),vcg_xyz=xyz)
        saved=json.loads((Path(str(data_path("b.aw147")))/f'predictions-{i:03d}.json').read_text())
        expected=[b['t_offset'] for b in saved['beats']]
        actual=[b.t_offset.sample if b.t_offset else None for b in det.beats];assert actual==expected
        robust,diag=fr.construct_robust_morphology_record(paths['morphology'],det)
        rv=fr.VCGTransformRegistry.transform(paths['morphology'],'kors').xyz;bv=fr.VCGTransformRegistry.transform(robust,'kors').xyz
        result=fr.BiomarkerRegistry.compute_all(delineation=det,ecg_12lead=robust,vcg_xyz=bv,fs=rec.fs,names=['t_vector_magnitude','t_duration_ms'],record_ecg_12lead=paths['morphology'],record_vcg_xyz=rv,record_raw_ecg_12lead=signal,record_acquisition={})
        vals={};fr._store_biomarker_results(vals,result,[],metadata={})
        p=base/f'errors-{i:03d}.parquet';pq.ParquetFile(p)
        z=pq.read_table(p,columns=['scope','landmark','prediction_sample','reference_sample','error_ms']).to_pandas();z=z[z.scope.eq('fused_global')&z.landmark.eq('t_offset')]
        finite=z.error_ms.dropna();amp=vals.get('t_vector_magnitude');duration=vals.get('t_duration_ms')
        row={'record':record,'amplitude_mv':None if amp is None else amp/1000,'t_duration_ms':duration,'flatness_mv_ms':None if amp is None or not duration else amp/1000/duration,'n_reference':len(z),'n_paired':len(finite),'mean_error_ms':None if not len(finite) else finite.mean(),'sd_error_ms':None if len(finite)<2 else finite.std(ddof=1),'equivalent':True,'sources':{str(p):sha(p),str(data/f'{record}.dat'):sha(data/f'{record}.dat'),str(data/f'{record}.hea'):sha(data/f'{record}.hea'),str(Path(str(data_path("b.aw147")))/f'predictions-{i:03d}.json'):sha(Path(str(data_path("b.aw147")))/f'predictions-{i:03d}.json')},'code_sha256':sha(__file__)}
        dump('ludb_records/'+cp.name,row);rows.append(row);dump('checkpoint.json',{'phase':'ludb','done':i,'total':len(records),'pid':os.getpid()})
    pd.DataFrame(rows).to_parquet(OUT/'ludb_private.parquet',index=False)
    calibrate()
def calibrate():
    import numpy as np,pandas as pd,statsmodels.api as sm
    z=pd.read_parquet(OUT/'ludb_private.parquet');z=z.dropna(subset=['amplitude_mv','flatness_mv_ms','mean_error_ms']);x=sm.add_constant(z[['amplitude_mv','flatness_mv_ms']].to_numpy(float));y=z.mean_error_ms.to_numpy(float)
    # One record-level mean per independent person, HC3 = robust record-level SE.
    fit=sm.OLS(y,x).fit(cov_type='HC3');res=y-fit.predict(x)
    f=pd.read_parquet(OUT/'population_private.parquet');tx=sm.add_constant(f[['amplitude_mv','flatness_mv_ms']].to_numpy(float));pred=fit.predict(tx)
    rng=np.random.default_rng(16620261013);dif=[];coef=[]
    for b in range(500):
        ix=rng.integers(0,len(z),len(z));cb=np.linalg.lstsq(x[ix],y[ix],rcond=None)[0];coef.append(cb)
        ix=rng.integers(0,len(f),len(f));yy=f.y.to_numpy(int)[ix];pp=tx[ix]@cb;dif.append(pp[yy==1].mean()-pp[yy==0].mean())
    delta=pred[f.y.eq(1)].mean()-pred[f.y.eq(0)].mean();ci=np.quantile(dif,[.025,.975])
    result={'n_records':len(z),'n_beats':int(z.n_paired.sum()),'n_reference':int(z.n_reference.sum()),'params':fit.params.tolist(),'se_robust_HC3':fit.bse.tolist(),'ci95_robust_HC3':fit.conf_int().tolist(),'residual_sd_record_ms':res.std(ddof=3),'r_squared':fit.rsquared,'mean_shift_low':pred[f.y.eq(1)].mean(),'mean_shift_high':pred[f.y.eq(0)].mean(),'group_difference_ms':delta,'group_difference_ci95_ms':ci.tolist(),'bootstrap':500,'columns':['constant','amplitude_mv','flatness_mv_ms'],'ludb_range':{c:[float(z[c].min()),float(z[c].max())] for c in ['amplitude_mv','flatness_mv_ms']},'outside_range':{c:int((~f[c].between(z[c].min(),z[c].max())).sum()) for c in ['amplitude_mv','flatness_mv_ms']}}
    np.save(OUT/'calibration_bootstrap_private.npy',np.asarray(coef));dump('calibration.json',result);print('calibration',json.dumps(result))
if __name__=='__main__':
    phase=sys.argv[1];dump('status.json',{'state':'running','phase':phase,'pid':os.getpid(),'memory':guard()})
    try:
        globals()[phase]()
        dump('status.json',{'state':'complete','phase':phase,'pid':os.getpid(),'memory':guard()})
    except Exception:
        import traceback;traceback.print_exc();dump('status.json',{'state':'failed','phase':phase,'pid':os.getpid()});raise
