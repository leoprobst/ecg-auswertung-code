"""Frozen development partition only: serialized measurement and aggregates."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os, sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
sys.dont_write_bytecode=True
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
import json, hashlib, time, argparse, gc, copy, datetime
OUT=Path(str(data_path("a.aw150")))
VAR=Path(str(data_path("a.transform_katalog_fortsetzung_v3")))
CAT=PROJECT/'nachrechnung/transform_katalog.csv'
RC=Path(str(data_path("a.rv_a1_repeat_marker_metrics_csv")))
S2=PROJECT/'publikationen/PUB-01_messvaliditaet/supplementary_data_S2_marker_matrix.csv'
SEED=20261012
CLASSES={'stabil':0,'bedingt':1,'transformabhaengig':2}
ATTRS=('r_onset','r_offset','t_offset')
SD_MS=(13.5,11.6,32.8)
SOURCES={}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def bind(p):
    p=Path(p)
    assert not any(k in str(p).lower() for k in ('split=test','locked_test','holdout'))
    SOURCES[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size}
    return p

def save(name,obj):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=str),encoding='utf8')
    temp.replace(p)

def readj(p):
    p=Path(p);assert p.stat().st_size<20*1024**2
    return json.loads(p.read_text(encoding='utf8'))

def gate():
    from werkzeuge.nachrechnung.resources import available
    while min(available().values())<1.5*1024**3:
        print('WAIT_MEMORY',flush=True);time.sleep(5)
    return available()

def readp(p,cols):
    import pyarrow.parquet as pq
    p=Path(p);footer=pq.ParquetFile(p)
    assert set(cols)<=set(footer.schema_arrow.names)
    with (OUT/'access_private.jsonl').open('a',encoding='utf8') as f:
        f.write(json.dumps({'path':str(p),'columns':cols,'rows_footer':footer.metadata.num_rows})+'\n')
    # ParquetFile avoids unintended Hive partition inference.
    return footer.read(columns=cols).to_pandas()

def combine(a,b):
    if a in CLASSES and b in CLASSES:return max((a,b),key=CLASSES.get)
    if a==b=='konstant_identisch':return a
    return 'nicht_schaetzbar'

def counts(series):return {str(k):int(v) for k,v in series.value_counts().items()}

def calibration(x,y):
    """OLS y=Kors, x=Dower; centered and scaled, training fold only."""
    import numpy as np
    ok=np.isfinite(x)&np.isfinite(y);x=x[ok];y=y[ok]
    if len(x)<100 or np.std(x,ddof=1)==0:return None
    mx=float(x.mean());my=float(y.mean());sx=float(x.std(ddof=1));sy=float(y.std(ddof=1))
    if sy==0:return None
    slope=float(np.dot((x-mx)/sx,(y-my)/sy)/np.dot((x-mx)/sx,(x-mx)/sx)*sy/sx)
    return {'intercept':my-slope*mx,'slope':slope,'n_fit':len(x),'x_center':mx,'y_center':my}

def aggregate():
    import numpy as np,pandas as pd
    from werkzeuge.transform_katalog.metriken import vergleichen,selbsttest
    cat=pd.read_csv(bind(CAT));rep=pd.read_csv(bind(RC));assert len(cat)==2005 and cat.marker_id.is_unique
    blocks=sorted((VAR/'bloecke').glob('block_*.parquet'));assert len(blocks)==29
    seals=[]
    for p in blocks:
        seal=readj(p.with_suffix('.json'));assert sha(p)==seal['sha256'];bind(p)
        seals.append(seal)
    identity=pd.concat([readp(p,['subject_id','study_id','split']) for p in blocks],ignore_index=True)
    assert len(identity)==5648 and identity.subject_id.is_unique and identity.study_id.is_unique
    assert identity.split.value_counts().to_dict()=={'train':2824,'validation':2824}
    folds={f:identity.split.eq(f).to_numpy() for f in ('train','validation')}
    metrics=[];res=[];mismatch=[];angle={}
    admitted=cat.loc[cat.admitted.eq(True),'marker_id'].tolist()
    for start in range(0,len(admitted),48):
        gate();names=admitted[start:start+48]
        data=pd.concat([readp(p,[pre+n for pre in ('kors__','dower__') for n in names]) for p in blocks],ignore_index=True)
        for n in names:
            row=cat.loc[cat.marker_id.eq(n)].iloc[0];x=data['kors__'+n].to_numpy(float);y=data['dower__'+n].to_numpy(float)
            result={'marker_id':n};original={}
            for f in ('train','validation','pool'):
                take=np.ones(len(x),bool) if f=='pool' else folds[f]
                m=vergleichen(x[take],y[take]);original[f]=m
                for k in ('n_pairs','bias','difference_sd','loa_low','loa_high','kors_sd','spearman','relative_loa_width','quintile_switch'):
                    if k in m:
                        old=row[f+'_'+k]
                        if not np.isclose(m[k],old,rtol=1e-10,atol=1e-12,equal_nan=True):mismatch.append({'marker_id':n,'fold':f,'key':k})
                assert m['class']==row[f+'_class']
            assert combine(original['train']['class'],original['validation']['class'])==row['class']
            transformed=np.full_like(y,np.nan)
            for fitf,target in (('train','validation'),('validation','train')):
                fit=calibration(y[folds[fitf]],x[folds[fitf]])
                result[target+'_calibration_status']='berechnet' if fit else 'nicht_schaetzbar'
                if fit:
                    result.update({target+'_fit_'+k:v for k,v in fit.items()})
                    transformed[folds[target]]=fit['y_center']+fit['slope']*(y[folds[target]]-fit['x_center'])
                m=vergleichen(x[folds[target]],transformed[folds[target]])
                for k,v in m.items():result[target+'_calibrated_'+k]=v
            for k,v in vergleichen(x,transformed).items():result['pool_calibrated_'+k]=v
            result['calibrated_class']=combine(result['train_calibrated_class'],result['validation_calibrated_class'])
            result['original_reproduced']=True
            result['calibration_interpretability']='keine_kategoriale_Validierung' if row.pool_scale_type in ('binaer','diskret_oder_bindungen') else 'lineare_Sensitivitaet'
            res.append(result)
            if n=='spatial_qrs_t_angle_deg':
                good=np.isfinite(x)&np.isfinite(y)
                angle={'marker_id':n,'n_pairs':int(good.sum()),**original['pool']}
                angle['thresholds']={str(t):{'switch_n':int(((x[good]>t)!=(y[good]>t)).sum()),'switch_fraction':float(((x[good]>t)!=(y[good]>t)).mean()),'kors_above':int((x[good]>t).sum()),'dower_above':int((y[good]>t).sum())} for t in (90,105,120)}
        del data;gc.collect();print('aggregate_markers',min(start+48,len(admitted)),flush=True)
    assert not mismatch,mismatch[:10]
    result=cat.merge(pd.DataFrame(res),on='marker_id',how='left',validate='one_to_one')
    for pop in ('gesamt','proxy'):
        r=rep.loc[rep.population.eq(pop),['marker_id','n_pairs','n_subjects','rc','difference_sd','between_person_sd']].rename(columns={c:'repeat_'+pop+'_'+c for c in ('n_pairs','n_subjects','rc','difference_sd','between_person_sd')})
        result=result.merge(r,on='marker_id',how='left',validate='one_to_one')
        result['matrix_halfwidth_over_rc_'+pop]=1.96*result.pool_difference_sd/result['repeat_'+pop+'_rc'].where(result['repeat_'+pop+'_rc']>0)
        result['matrix_sd_over_repeat_sd_'+pop]=result.pool_difference_sd/result['repeat_'+pop+'_difference_sd'].where(result['repeat_'+pop+'_difference_sd']>0)
        result['matrix_sd_over_between_person_sd_'+pop]=result.pool_difference_sd/result['repeat_'+pop+'_between_person_sd'].where(result['repeat_'+pop+'_between_person_sd']>0)
    result['matrix_nonconstant_identical']=result['class'].eq('stabil') & result.pool_difference_sd.eq(0) & result.pool_bias.eq(0)
    assert int(result.matrix_nonconstant_identical.sum())==217
    result['matrix_primary_class']=result['class'].mask(result.matrix_nonconstant_identical,'matrixunabhaengig_identisch')
    primary=counts(result.matrix_primary_class)
    assert primary=={'transformabhaengig':1071,'bedingt':684,'matrixunabhaengig_identisch':217,'nicht_schaetzbar':31,'nicht_pruefbar':1,'konstant_identisch':1}
    main=result.loc[result.matrix_primary_class.isin(CLASSES)]
    ct=pd.crosstab(main.train_class,main.validation_class).reindex(index=CLASSES,columns=CLASSES,fill_value=0)
    ct.to_csv(OUT/'fold_cross_table.csv',index=True)
    n=len(main);po=float(np.trace(ct)/n);pe=float(np.dot(ct.sum(axis=1),ct.sum(axis=0))/n**2)
    dep=main.matrix_primary_class.eq('transformabhaengig');both=main.train_class.eq('transformabhaengig')&main.validation_class.eq('transformabhaengig')
    calibrated=counts(main.calibrated_class)
    summary={'n_ecgs':5648,'n_persons':5648,'folds':{'train':2824,'validation':2824},'catalog_n':2005,'primary_counts':primary,'reproduced_admitted':len(admitted),'metrics_mismatches':mismatch,'selftest':selbsttest(),
        'fold_concordance':{'n':n,'same':int(np.trace(ct)),'agreement':po,'kappa':(po-pe)/(1-pe),'dependent_both':int(both.sum()),'dependent_one_only':int((dep&~both).sum()),'train_counts':counts(main.train_class),'validation_counts':counts(main.validation_class),'pool_counts':counts(main.pool_class),'worse_dependent_pool_not_dependent':int((dep&~main.pool_class.eq('transformabhaengig')).sum())},
        'calibrated_counts_matrix_set':calibrated,'dependent_to_conditional':int((dep&main.calibrated_class.eq('bedingt')).sum()),'dependent_to_stable':int((dep&main.calibrated_class.eq('stabil')).sum()),'dependent_to_unestimable':int((dep&~main.calibrated_class.isin(CLASSES)).sum()),'angle_same_catalog_sample':angle,'repeat_source':'repeat-measurement metrics table; populations gesamt/proxy; earlier repeatability analysis not used', 'sources':SOURCES,'script_sha256':sha(__file__)}
    bind(PROJECT/'nachrechnung/b_stoertest.csv')
    sto=pd.read_csv(PROJECT/'nachrechnung/b_stoertest.csv');sto=sto.loc[sto.scenario.eq('S')].copy()
    sto=sto.merge(result[['marker_id','repeat_gesamt_rc','repeat_proxy_rc']],on='marker_id',validate='many_to_one')
    sto['mac_over_rc_gesamt']=sto.mean_absolute_change/sto.repeat_gesamt_rc.where(sto.repeat_gesamt_rc>0)
    sto['mac_over_rc_proxy']=sto.mean_absolute_change/sto.repeat_proxy_rc.where(sto.repeat_proxy_rc>0)
    sto.to_csv(OUT/'b_boundary_over_rc.csv',index=False)
    summary['sources']=SOURCES
    result.to_csv(OUT/'matrix_base_aggregate.csv',index=False)
    result.to_csv(S2,index=False)
    save('aggregate_summary.json',summary)
    print(json.dumps({k:summary[k] for k in ('primary_counts','fold_concordance','calibrated_counts_matrix_set','dependent_to_conditional','dependent_to_stable','angle_same_catalog_sample')},ensure_ascii=False),flush=True)

def prepare():
    import pandas as pd,pyarrow.parquet as pq,pyarrow as pa
    from werkzeuge.transform_katalog.fortsetzung import rows,binding
    import werkzeuge.varianten.adapter as adapter
    adapter.memory_guard=gate
    b=binding();cat=pd.read_csv(bind(CAT));allrows=rows()
    selected=[]
    for fold in ('train','validation'):
        rank=lambda r:hashlib.sha256(f'{SEED}|AW150|{fold}|{r["subject_id"]}|{r["study_id"]}'.encode()).hexdigest()
        selected+=sorted([r for r in allrows if r['split']==fold],key=rank)[:250]
    assert len(selected)==500 and len({r['subject_id'] for r in selected})==500
    pq.write_table(pa.Table.from_pylist(selected),OUT/'selection_private.parquet',compression='zstd')
    full=cat.loc[cat.admitted.eq(True),'marker_id'].tolist()
    panel=[]
    for family,g in cat.loc[cat.admitted.eq(True)&cat.implemented.eq(True)].groupby('familie'):
        order=sorted(g.marker_id,key=lambda n:hashlib.sha256(f'{SEED}|AW150|marker|{family}|{n}'.encode()).hexdigest())
        assert len(order)>=10
        panel+=order[:10]
    panel=sorted(set(panel)|{'qrs_t_normalized_discrete_frechet','qrs_t_frechet_beat_median','qrs_t_normalized_dtw_distance','spatial_qrs_t_angle_deg','qrs_duration_ms','qtc_fridericia_ms'})
    # Filter is defined and hashed before any new sensitivity result.
    from scipy.signal import firwin
    taps=firwin(81,112.5,fs=500,window=('kaiser',8.6))
    manifest={'seed':SEED,'selection_sha256':sha(OUT/'selection_private.parquet'),'source_binding':b,'cat_sha256':sha(CAT),'script_sha256':sha(__file__),
        'n':500,'folds':{'train':250,'validation':250},'full_marker_ids':full,'panel_marker_ids':panel,'panel_family_counts':cat.loc[cat.marker_id.isin(panel),'familie'].value_counts().to_dict(),
        'NS05':'Six separate shifts of every beat: r_onset/r_offset/t_offset, +/-4 ms at 500 Hz; identical signals and baseline delineation; worse direction and fold',
        'S':'Six separate +/-1 LUDB-SD shifts: r_onset 13.5 ms, r_offset 11.6 ms, t_offset 32.8 ms; round to sample',
        'NS20':'Physical raw 500 Hz signal; resample_poly up=1 down=2 axis=0, padtype=constant, custom 81-tap FIR Kaiser beta8.6 cutoff112.5Hz fs500; delay compensated by scipy; complete unchanged Release preprocessing/delineation/measurement at250; additional fixed-fiducial sensitivity',
        'filter_coefficients':taps.tolist(),'filter_sha256':hashlib.sha256(taps.tobytes()).hexdigest(),'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    save('measurement_manifest.json',manifest)
    print(json.dumps({'n':500,'panel_n':len(panel),'families':manifest['panel_family_counts']}),flush=True)

def values(package,row,names):
    import numpy as np
    from werkzeuge.varianten.adapter import release_import
    fr=release_import();available=set(fr.BiomarkerRegistry.available());reg=[n for n in names if n in available]
    record=fr.VCGTransformRegistry.transform(package['paths']['morphology'],'kors').xyz
    robust=fr.VCGTransformRegistry.transform(package['robust'],'kors').xyz
    wide=record if package['wideband_supported'] and 'wideband' in package['paths'] else None
    v=fr.BiomarkerRegistry.compute_all(delineation=package['delineation'],ecg_12lead=package['robust'],vcg_xyz=robust,fs=package['fs'],names=reg,record_ecg_12lead=package['paths']['morphology'],record_vcg_xyz=record,record_vcg_wideband_xyz=wide,record_raw_ecg_12lead=package['loaded'].signal_millivolts,record_acquisition=package['acquisition'])
    dest={};fr._store_biomarker_results(dest,v,[],metadata={})
    if any(n.startswith('classical_') for n in names):
        report=fr._REPORT_PARSER.parse(str(row.get('machine_report') or ''));positive=lambda k:report[k].state==fr.ReportMentionState.POSITIVE
        classic=fr.ClassicalECGCriteria().compute(package['paths']['diagnostic'],package['delineation'],age=fr._optional_int(row.get('age')),sex=row.get('sex'),machine_qrs_axis_deg=fr._optional_float(row.get('machine_qrs_axis_deg')),machine_t_axis_deg=fr._optional_float(row.get('machine_t_axis_deg')),paced_or_lbbb=positive('pacemaker') or positive('lbbb'),paced=positive('pacemaker'),measurement_sink={})
        for n in names:
            if n.startswith('classical_'):
                c=classic.get(n.removeprefix('classical_').removesuffix('_positive'))
                dest[n]=(c.positive if n.endswith('_positive') else c.value) if c is not None and c.evaluable else None
    return [float(dest[n]) if dest.get(n) is not None and np.isfinite(dest[n]) else None for n in names]

def shift(d,attr,samples):
    z=copy.deepcopy(d);seen=set()
    for beat in z.beats:
        p=getattr(beat,attr,None)
        if p is not None and id(p) not in seen:p.sample+=int(samples);seen.add(id(p))
    return z

def downsample(package,row,taps,fixed=False):
    """Only intentional change: FIR/raw fs250; all Release algorithms unchanged."""
    from dataclasses import replace
    from scipy.signal import resample_poly
    from werkzeuge.varianten.adapter import release_import
    fr=release_import();loaded=replace(package['loaded'],signal_millivolts=resample_poly(package['loaded'].signal_millivolts,1,2,axis=0,window=taps,padtype='constant'),sampling_rate=250)
    paths=fr.MimicWaveformLoader(target_sampling_rate=250).prepare_paths(loaded,powerline_hz=60.,device_filtering=row.get('machine_filtering'))
    if fixed:
        d=copy.deepcopy(package['delineation']);d.fs=250;d.n_samples=loaded.signal_millivolts.shape[0];seen=set()
        for beat in d.beats:
            for attr in ('p_onset','p_peak','p_offset','q_peak','r_onset','r_peak','r_offset','s_peak','j_point','t_onset','t_peak','t_offset','u_onset','u_peak','u_offset','retrograde_p_onset','retrograde_p_peak','retrograde_p_offset'):
                p=getattr(beat,attr,None)
                if p is not None and id(p) not in seen:p.sample=round(p.sample/2);seen.add(id(p))
        d.invalidate_beat_cache()
    else:
        peaks=fr.detect_r_peaks_multilead(paths['delineation'],250)
        if len(peaks)<2:raise ValueError('250Hz R-Peak-Gate')
        xyz=fr.VCGTransformRegistry.transform(paths['delineation'],'kors').xyz
        d=fr.FusionDelineator(weights={'neurokit':.4,'multilead':.35,'vcg':.25},conflict_threshold=20.).delineate_record(ecg_12lead=paths['delineation'],r_peaks=peaks,fs=250,ecg_id=int(row['study_id']),vcg_xyz=xyz)
        if not d.beats:raise ValueError('250Hz Delineationsgate')
        report=fr._REPORT_PARSER.parse(str(row.get('machine_report') or ''))
        if report['af'].state==fr.ReportMentionState.POSITIVE:d.suppress_discrete_p('discrete_p_suppressed_by_report_atrial_fibrillation_flutter')
    robust,diag=fr.construct_robust_morphology_record(paths['morphology'],d)
    return {**package,'loaded':loaded,'paths':paths,'robust':robust,'delineation':d,'fs':250,'diagnostics':diag}

def measure(limit=None):
    import numpy as np
    import werkzeuge.varianten.adapter as adapter
    adapter.memory_guard=gate
    manifest=readj(OUT/'measurement_manifest.json');assert sha(__file__)==manifest['script_sha256']
    assert sha(CAT)==manifest['cat_sha256'] and sha(OUT/'selection_private.parquet')==manifest['selection_sha256']
    import pyarrow.parquet as pq
    schema=pq.ParquetFile(OUT/'selection_private.parquet').schema_arrow
    rows=readp(OUT/'selection_private.parquet',schema.names).to_dict('records')
    names=manifest['full_marker_ids'];panel=manifest['panel_marker_ids'];ix=[names.index(n) for n in panel]
    # Pull the historical baseline only from sealed blocks of the earlier catalogue run, after footer/projection.
    expected={}
    for p in sorted((VAR/'bloecke').glob('block_*.parquet')):
        seal=readj(p.with_suffix('.json'));assert sha(p)==seal['sha256']
        f=readp(p,['study_id']+['kors__'+n for n in names]);wanted={r['study_id'] for r in rows}
        for r in f.loc[f.study_id.isin(wanted)].to_dict('records'):
            expected[r['study_id']]=[r['kors__'+n] for n in names]
    assert len(expected)==500
    mhash=sha(OUT/'measurement_manifest.json');start=time.monotonic()
    for i,row in enumerate(rows):
        if limit and i>=limit:break
        target=OUT/'measurements'/f'{i:04d}.json'
        if target.exists():
            old=readj(target);assert old['manifest_sha256']==mhash;continue
        gate();tic=time.monotonic();entry={'manifest_sha256':mhash,'row_index':i,'split':row['split'],'source_500_hz':False}
        try:package=adapter.prepare_record(row)
        except ValueError as exc:
            entry['status']='baseline_qc_abstention';entry['reason']=str(exc);save(str(target.relative_to(OUT)),entry);continue
        base=values(package,row,names);a=np.array(base,float);b=np.array(expected[row['study_id']],float)
        eq=np.isclose(a,b,rtol=1e-9,atol=0,equal_nan=True)
        if not eq.all():
            entry['status']='equivalence_failure';entry['mismatch_marker_ids']=[n for n,e in zip(names,eq) if not e]
            save('equivalence_failure_private.json',entry);raise RuntimeError('NS05 baseline does not reproduce the earlier catalogue run')
        entry['status']='computed';entry['baseline']=base;entry['baseline_equivalent']=True;entry['NS05']=[];entry['S']=[]
        for attr,sd in zip(ATTRS,SD_MS):
            for sign in (-1,1):
                z={**package,'delineation':shift(package['delineation'],attr,sign*2)}
                entry['NS05'].append({'boundary':attr,'sign':sign,'actual_ms':sign*4,'values':values(z,row,names)})
                samples=round(sign*sd*package['fs']/1000)
                z={**package,'delineation':shift(package['delineation'],attr,samples)}
                entry['S'].append({'boundary':attr,'sign':sign,'actual_ms':samples*1000/package['fs'],'values':values(z,row,panel)})
        entry['source_500_hz']=bool(package['loaded'].original_sampling_rate==500)
        if entry['source_500_hz']:
            for fixed,key in ((False,'fs250'),(True,'fs250_fixed')):
                try:entry[key]=values(downsample(package,row,np.array(manifest['filter_coefficients']),fixed),row,panel)
                except ValueError as exc:entry[key]=[None]*len(panel);entry[key+'_reason']=str(exc)
        else:entry['fs250_reason']='Originalaufnahme nicht 500Hz; ausgeschlossen'
        entry['seconds']=time.monotonic()-tic
        save(str(target.relative_to(OUT)),entry);gc.collect()
        save('measurement_status.json',{'status':'completed' if i==499 else 'running','done':i+1,'n':500,'seconds_this_invocation':time.monotonic()-start})
        if i%5==0:print(json.dumps({'done':i+1,'n':500,'record_seconds':entry['seconds'],'total_seconds':time.monotonic()-start}),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('aggregate','prepare','measure','report','verify'));parser.add_argument('--limit',type=int);a=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);memory=gate();save('resources_start_'+a.phase+'.json',memory)
    from werkzeuge.nachrechnung.resources import Budget
    with Budget(OUT) as budget:
        if a.phase=='aggregate':aggregate()
        elif a.phase=='prepare':prepare()
        elif a.phase=='measure':measure(a.limit)
        else:
            from werkzeuge.varianten.aw150_report import report,verify
            (report if a.phase=='report' else verify)()
    save('resources_end_'+a.phase+'.json',budget.summary())
if __name__=='__main__':main()
