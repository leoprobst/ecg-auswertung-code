"""AW-032 Pilotrunner: explizite Entwicklungsquellen, N-0-Gate, N-4, N-2."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os
os.environ.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
import argparse, hashlib, json, time, sys, gc
from pathlib import Path
from datetime import datetime, timezone
from .resources import guard, Budget

PROJECT=Path(__file__).resolve().parents[2]
DATA=Path(str(data_path("b.mimic_validation")))
DEV=DATA/'features/primary_core_sentinel_discovery_reserve_classical_criteria/development'
RUN=DATA/'runs/A155-2026-10-06-single-markers'
OUT=Path(str(data_path("b.nachrechnung")))
PILOTS={'BEF-0063':('reduced_lvef_strict','discovery_reserve','qrs_t_normalized_discrete_frechet'),
        'BEF-0032':('severe_aortic_stenosis_velocity','sentinel','qrs_loop_mean_distance_centroid')}
MODELS=('B0','B1','B2')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def write(path,obj):
    path=Path(path).resolve()
    if not path.is_relative_to(OUT.resolve()):raise ValueError('Ausgabe außerhalb freigegebenem E:-Bereich')
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    # Virenscanner/Windows können gerade geschlossene Checkpoints kurz sperren.
    for attempt in range(30):
        try:tmp.replace(path);break
        except PermissionError:
            if attempt==29:raise
            time.sleep(.1*(attempt+1))

def arrow_sha(table):
    import pyarrow as pa
    sink=pa.BufferOutputStream()
    with pa.ipc.new_stream(sink,table.schema) as w:w.write_table(table.combine_chunks())
    return hashlib.sha256(sink.getvalue()).hexdigest()

def freeze_binding():
    path=PROJECT/'nachrechnung/SPEZIFIKATION_N0_N2_N4.md'
    f=json.loads(path.with_suffix('.freeze.json').read_text(encoding='utf-8'))
    if sha(path)!=f['sha256']:raise ValueError('Spezifikationsfreeze verändert')
    files=[PROJECT/'schema/komparatoren.csv',PROJECT/'schema/endpunktgruppen.csv',PROJECT/'standards/02_pruefkette.md',PROJECT/'standards/03_skalen.md']
    # Nur tatsächlich rechnende Module; ein neuer Berichtshelfer ändert keine Fits.
    files += [Path(__file__).parent/n for n in ('__init__.py','core.py','resources.py','runner.py')]
    release=Path(str(data_path("b.release_sources_a155_survival_nullable_20261007_vcg_biomarker_mimic")))
    files += [release/n for n in ('validation.py','statistics.py','cross_fitting.py','labels.py')]
    return {'freeze':f,'files_sha256':{str(p):sha(p) for p in files}}

def comparator_for(endpoint):
    import pandas as pd
    groups=pd.read_csv(PROJECT/'schema/endpunktgruppen.csv',sep=';')
    group=groups.loc[groups.endpunkt_id==endpoint,'gruppe'].iloc[0]
    table=pd.read_csv(PROJECT/'schema/komparatoren.csv',sep=';')
    items=table.loc[(table.gruppe==group)&(table.form=='stetig')&(table.in_pipeline_umgesetzt=='ja')].sort_values(['staerke_rang','komparator_id'])
    if items.empty:raise ValueError('Kein umgesetzter stetiger Komparator: B2 offen')
    item=items.iloc[0].to_dict();feature=item['komparator_id']
    if feature=='classical_qrs_widening_ge_120':feature='qrs_duration_ms'
    item.update(feature_used=feature,selection='Registerrang vorab, keine Test-/Validation-AUC-Wahl',B2_equals_B1=feature=='qrs_duration_ms')
    return item

def load_inputs(identifier,base):
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.dataset as ds
    import duckdb
    from .core import assert_development
    pa.set_cpu_count(2);pa.set_io_thread_count(2)
    endpoint,panel,marker=PILOTS[identifier];comp=comparator_for(endpoint)
    labels=DATA/'labels/endpoint_index_cohorts.parquet';attpath=DATA/'labels/development_label_attestation.json'
    att=json.loads(attpath.read_text(encoding='utf-8'))
    if att['status']!='verified' or att['scope']!='train_validation_only' or set(att['source_splits'])!={'train','validation'}:raise ValueError('Labelattestierung ungültig')
    guard();labelhash=sha(labels)
    if labelhash!=att['development_artifacts']['endpoint_index_cohorts']['sha256']:raise ValueError('Labelhash nicht attestiert')
    lc=['subject_id','study_id','split','ecg_time','label_state','eligibility_state','index_policy','index_selection_strategy','time_delta_hours','time_precision','ehr_time_alignment_confirmatory_eligible']
    schema=pq.ParquetFile(labels).schema_arrow
    if set(lc+['label_id'])-set(schema.names):raise ValueError('Labelschema fehlt')
    con=duckdb.connect();con.execute("SET memory_limit='1GB'; SET threads=2")
    # Attestierte Kohorte enthält ausschließlich Entwicklung; vor engerer Fallwahl auditieren.
    counts=con.execute(f"SELECT split,label_state,eligibility_state,index_selection_strategy,count(*) AS n FROM read_parquet('{labels.as_posix()}') WHERE label_id=? GROUP BY ALL",[endpoint]).fetchdf()
    if not set(counts.split) <= {'train','validation'}:raise ValueError('Nichtentwicklungslabel')
    query=f"SELECT {','.join(lc)} FROM read_parquet('{labels.as_posix()}') WHERE label_id=? AND label_state IN ('positive','negative') AND eligibility_state='eligible'"
    frame=con.execute(query,[endpoint]).fetchdf()
    assert_development(frame)
    if set(frame.index_selection_strategy)!={'first_verified_ecg_reference_result_blind'}:raise ValueError('Indexstrategie nicht spezifiziert')
    if frame.subject_id.duplicated().any() or frame.study_id.duplicated().any():raise ValueError('Doppelter Index')
    wanted=frame[['subject_id','study_id','split','ecg_time']];con.register('wanted',wanted)
    master=DATA/'master/ecg_master_index.parquet';pq.ParquetFile(master)
    mq=f"SELECT subject_id,study_id,split,ecg_time,age,sex,CASE WHEN machine_metadata_usable_for_primary THEN machine_heart_rate_bpm ELSE NULL END AS heart_rate FROM read_parquet('{master.as_posix()}') WHERE study_id IN (SELECT study_id FROM wanted) AND split IN ('train','validation')"
    context=con.execute(mq).fetchdf();con.close()
    if len(context)!=len(frame) or context.study_id.duplicated().any():raise ValueError('Master-Identität fehlt/mehrdeutig')
    joined=wanted.merge(context,on=['subject_id','study_id','split','ecg_time'],validate='one_to_one')
    if len(joined)!=len(frame):raise ValueError('Master-Zeit-/Foldabweichung')
    masterhash=arrow_sha(pa.Table.from_pandas(context.sort_values('study_id'),preserve_index=False))
    frame=frame.merge(context,on=['subject_id','study_id','split','ecg_time'],validate='one_to_one')
    cols=list(dict.fromkeys(['subject_id','study_id','split',marker,'qrs_duration_ms','qtc_fridericia_ms',comp['feature_used']]))
    manifest=[];parts=[]
    for fold in ('train','validation'):
        ids=frame.loc[frame.split==fold,'study_id'].tolist()
        files=sorted((DEV/f'split={fold}').glob('part-*.parquet'))
        if not files:raise ValueError('Entwicklungspartitionen fehlen')
        for i,p in enumerate(files):
            guard()
            if p.resolve().parent!=(DEV/f'split={fold}').resolve():raise ValueError('Featurepfad unerwartet')
            s=pq.ParquetFile(p).schema_arrow;present=[c for c in cols if c in s.names]
            if set(['subject_id','study_id','split'])-set(present):raise ValueError('Featureidentität fehlt')
            t=ds.dataset(str(p),format='parquet').to_table(columns=present,filter=ds.field('study_id').isin(ids))
            if t.num_rows:
                f=t.to_pandas()
                if set(f.split)!={fold}:raise ValueError('Partitionssplit abweichend')
                for c in cols:
                    if c not in f:f[c]=np.nan
                parts.append(f[cols])
            manifest.append({'path':str(p),'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'schema_sha256':hashlib.sha256(str(s).encode()).hexdigest(),'selected_rows':t.num_rows,'projection_sha256':arrow_sha(t),'columns':present})
            if i%200==0:print(json.dumps({'phase':'features','befund':identifier,'fold':fold,'files_done':i,'files_total':len(files)}),flush=True)
    features=pd.concat(parts,ignore_index=True)
    if features.study_id.duplicated().any():raise ValueError('Doppelte Feature-ID')
    extra=features.merge(wanted[['subject_id','study_id','split']],on=['subject_id','study_id','split'],how='inner')
    if len(extra)!=len(features):raise ValueError('Featurepatient/Fold abweichend')
    frame=frame.merge(features,on=['subject_id','study_id','split'],how='left',validate='one_to_one',indicator=True)
    absent=int((frame['_merge']=='left_only').sum());frame=frame.drop(columns='_merge')
    frame['label']=(frame.label_state=='positive').astype('int8');frame['sex_numeric']=frame.sex.map({'F':0.,'M':1.})
    for c in ['age','sex_numeric','heart_rate',marker,'qrs_duration_ms','qtc_fridericia_ms',comp['feature_used']]:
        frame[c]=pd.to_numeric(frame[c],errors='coerce').replace([np.inf,-np.inf],np.nan)
    frame=frame.sort_values(['split','subject_id','study_id']).reset_index(drop=True)
    projection=pa.Table.from_pandas(frame,preserve_index=False)
    out=OUT/identifier;out.mkdir(parents=True,exist_ok=True)
    pq.write_table(projection,out/'development_projection.parquet')
    input_binding={**base,'labels_sha256':labelhash,'label_attestation_sha256':sha(attpath),'master_development_projection_sha256':masterhash,'cohort_feature_projection_sha256':arrow_sha(projection),'feature_partition_projection_manifest_sha256':digest(manifest),'feature_attestation_sha256':sha(DEV/'.feature_verification_attestation.json'),'results_sha256':sha(RUN/'cross_fit_results.parquet'),'confirmations_sha256':sha(RUN/'cross_fit_confirmations.parquet'),'comparator':comp,'key':{'endpoint_id':endpoint,'panel_id':panel,'feature_id':marker,'analysis_type':'single_marker'}}
    write(out/'input_manifest.json',{'binding':input_binding,'partition_manifest':manifest,'queries':{'labels':query,'labels_parameter':endpoint,'master':mq},'columns':cols,'missing_feature_records':absent,'label_flow':counts.to_dict('records'),'master_full_hash_computed':False,'sealed_read':False})
    return frame,comp,input_binding

def reference_rows(identifier):
    import pyarrow.parquet as pq
    import pyarrow.dataset as ds
    endpoint,panel,marker=PILOTS[identifier]
    filt=(ds.field('endpoint_id')==endpoint)&(ds.field('panel_id')==panel)&(ds.field('feature_id')==marker)&(ds.field('analysis_type')=='single_marker')
    def read(name,cols):
        p=RUN/(name+'.parquet');pq.ParquetFile(p)
        return ds.dataset(str(p),format='parquet').to_table(columns=cols,filter=filt).to_pandas()
    pooled=read('cross_fit_results',['fold_a_effect','fold_b_effect','pooled_effect','fold_a_direction','fold_b_direction','fold_a_n_confirming','fold_b_n_confirming'])
    folds=read('cross_fit_confirmations',['confirming_split','direction','n_total','n_positive','n_negative','roc_auc','roc_auc_ci_low','roc_auc_ci_high'])
    if len(pooled)!=1 or len(folds)!=2:raise ValueError('Quellschlüssel nicht eindeutig')
    return pooled.iloc[0].to_dict(),folds.set_index('confirming_split').loc[['validation','train']].reset_index().to_dict('records')

def n0(frame,identifier):
    import numpy as np
    from .core import auc
    marker=PILOTS[identifier][2];ref,foldref=reference_rows(identifier)
    f=frame.loc[frame[marker].notna()].copy()
    splits=[f.loc[f.split==s].copy() for s in ('validation','train')]
    raw=[auc(s.label.to_numpy(),s[marker].to_numpy()) for s in splits]
    directions=[1 if raw[1]>=.5 else -1,1 if raw[0]>=.5 else -1]
    effects=[a if d==1 else 1-a for a,d in zip(raw,directions)]
    se=np.array([(r['roc_auc_ci_high']-r['roc_auc_ci_low'])/(2*1.959963984540054) for r in foldref])
    weights=1/se**2;weights/=weights.sum()
    counts=[{'n':len(s),'positive':int(s.label.sum()),'negative':int((s.label==0).sum()),'subjects':int(s.subject_id.nunique())} for s in splits]
    differences=[float(effects[i]-ref['fold_'+letter+'_effect']) for i,letter in enumerate(('a','b'))]
    reproducing=float(np.dot(weights,effects));aggregate=float(np.dot(weights,[r['roc_auc'] for r in foldref]))
    matches=all(abs(d)<=1e-10 for d in differences+[reproducing-ref['pooled_effect'],aggregate-ref['pooled_effect']])
    matches=matches and all(c['n']==r['n_total'] and c['positive']==r['n_positive'] and c['negative']==r['n_negative'] and d==r['direction'] for c,r,d in zip(counts,foldref,directions))
    return f,{'status':'passed' if matches else 'stop_gate','tolerance_absolute':1e-10,'counts_A_B':counts,'directions_A_B':directions,'auc_A_B':effects,'auc_differences_A_B':differences,'historical_pool_weights_A_B':weights.tolist(),'pipeline_pool':ref['pooled_effect'],'P01b_aggregate_reproduced':aggregate,'P01c_individual_with_historical_weights':reproducing,'pool_difference':reproducing-ref['pooled_effect'],'reference_rows':foldref,'P01a_limit':'Aktuelle Projekthashes/Labelattestierung; vollständige historische Erzeugeridentität nicht bewiesen','P01c_pool_limit':'Gewichte historisch gebunden, individuelle Fold-AUC unabhängig'}

def bootstrap_phase(identifier,phase,binding,one,iterations=1000):
    import numpy as np
    path=OUT/identifier/(phase+'_checkpoint.json')
    rng=np.random.default_rng(32009+(63 if identifier=='BEF-0063' else 32)+(0 if phase=='N4' else 1000))
    signature=digest({'binding':binding,'phase':phase,'iterations':iterations})
    values=[];attempts=0;seconds=0.
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'))
        if old['signature']!=signature:raise ValueError('Checkpointbindung abweichend')
        values=old['values'];attempts=old['attempts'];seconds=old['seconds'];rng.bit_generator.state=old['rng_state']
    started=time.perf_counter();last=time.perf_counter();initial_count=len(values)
    while len(values)<iterations:
        if (len(values)-initial_count)%25==0:guard()
        attempts+=1
        if attempts>2*iterations:raise RuntimeError('Zu viele ungültige Replikate')
        try:values.append(one(rng).tolist())
        except ValueError as e:
            if str(e)!='Nur eine Klasse':raise
        if len(values)%25==0:
            now=time.perf_counter();seconds+=now-last;last=now
            write(path,{'signature':signature,'values':values,'attempts':attempts,'rng_state':rng.bit_generator.state,'seconds':seconds})
            print(json.dumps({'befund':identifier,'phase':phase,'valid_replicates':len(values),'seconds_accumulated':round(seconds,2)}),flush=True)
    return np.asarray(values),{'iterations':iterations,'attempts':attempts,'seed':32009+(63 if identifier=='BEF-0063' else 32)+(0 if phase=='N4' else 1000),'seconds':seconds,'seconds_this_invocation':time.perf_counter()-started,'resampling':'subject_id within original split; all records same multiplicity','interval':'percentile_95'}

def n4(frame,result,identifier,binding):
    import numpy as np
    from .core import RankedAUC,cluster_weights,pool_summary
    marker=PILOTS[identifier][2];splits=[frame.loc[frame.split==s] for s in ('validation','train')]
    ranked=[RankedAUC(s.label.to_numpy(),s[marker].to_numpy()) for s in splits]
    def one(rng):
        w=[cluster_weights(s.subject_id.to_numpy(),rng) for s in splits]
        raw=[ranked[i](w[i]) for i in range(2)]
        directions=[1 if raw[1]>=.5 else -1,1 if raw[0]>=.5 else -1]
        return np.array([a if d==1 else 1-a for a,d in zip(raw,directions)])
    samples,timing=bootstrap_phase(identifier,'N4',binding,one)
    countweights=np.array([len(s) for s in splits],dtype=float);countweights/=countweights.sum()
    return {'status':'completed','historical_weight_target':pool_summary(result['auc_A_B'],samples,result['historical_pool_weights_A_B']),
            'de_novo_count_weight_target':pool_summary(result['auc_A_B'],samples,countweights),'timing':timing,
            'pool_covariance_scope':'joint resampling; raw-marker directions refitted; historical weights fixed; no model fitting needed',
            'P08_limit':'Fixed historical weight conditional interval; hypothesis selection optimism not included'}

def n2(frame,comp,identifier,binding):
    import numpy as np
    from .core import auc,design,held_models,cluster_weights,pool_summary
    marker=PILOTS[identifier][2]
    required=list(dict.fromkeys(['age','sex_numeric','heart_rate','qrs_duration_ms','qtc_fridericia_ms',comp['feature_used'],marker]))
    complete=frame.dropna(subset=required).copy()
    splits=[complete.loc[complete.split==s] for s in ('validation','train')]
    if min(len(s) for s in splits)==0:raise ValueError('Keine gemeinsame Fallmenge')
    arrays={model:[design(s,model,comp['feature_used']) for s in splits] for model in MODELS}
    markers=[s[marker].to_numpy(float) for s in splits];labels=[s.label.to_numpy() for s in splits]
    def calc(w):
        res=[]
        for model in MODELS:
            if model=='B2' and comp['B2_equals_B1']:res.append(res[-1].copy());continue
            res.append(np.array([held_models(arrays[model][1-i],arrays[model][i],markers[1-i],markers[i],labels[1-i],labels[i],w[1-i],w[i]) for i in range(2)]))
        return np.array(res)
    started=time.perf_counter();point=calc([np.ones(len(s)) for s in splits]);fit_seconds=time.perf_counter()-started
    def one(rng):return calc([cluster_weights(s.subject_id.to_numpy(),rng) for s in splits])
    samples,timing=bootstrap_phase(identifier,'N2',binding,one)
    weights=np.array([len(s) for s in splits],dtype=float);weights/=weights.sum()
    metrics={}
    for i,model in enumerate(MODELS):
        metrics[model]={name:pool_summary(point[i,:,j],samples[:,i,:,j],weights) for j,name in enumerate(('baseline_auc','augmented_auc','delta_auc','residual_conditional_auc'))}
    missing={c:{split:{'n_missing':int(s[c].isna().sum()),'positive_missing':int(s.loc[s[c].isna(),'label'].sum()),'negative_missing':int((s.loc[s[c].isna(),'label']==0).sum())} for split,s in frame.groupby('split')} for c in required}
    return {'status':'completed','metrics':metrics,'timing':{**timing,'point_fit_seconds':fit_seconds},'B2_equals_B1':comp['B2_equals_B1'],'required_columns':required,'missingness':missing,
            'flow':{split:{'N0_n':len(frame.loc[frame.split==split]),'complete_n':len(s),'positive':int(s.label.sum()),'negative':int((s.label==0).sum()),'subjects':int(s.subject_id.nunique()),'raw_marker_auc_on_common_cases':auc(s.label.to_numpy(),result_direction(frame,marker,split)*s[marker].to_numpy())} for split,s in complete.groupby('split')},
            'bootstrap_refit':True,'conditional_auc_definition':'Held-out AUC of marker residualized against B basis in fitting split; direction trained there; descriptive adjusted rank diagnostic, not causal conditional ROC',
            'pool_definition':'sample-count weighted held-out fold AUC/difference, fixed weights; joint full refit bootstrap',
            'population_limit':'Complete cases in verified hospital cohort; no external/test confirmation; no selection optimism correction'}

def result_direction(frame,marker,split):
    from .core import auc
    s=frame.loc[frame.split!=split]
    return 1 if auc(s.label.to_numpy(),s[marker].to_numpy())>=.5 else -1

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--befund',choices=tuple(PILOTS),required=True)
    args=parser.parse_args();guard();OUT.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();base=freeze_binding();identifier=args.befund
    result={'auftrag':'AW-032','befund':identifier,'run':RUN.name,'started_at':datetime.now(timezone.utc).isoformat(),'recipe':f'python -m werkzeuge.nachrechnung.runner --befund {identifier}','sealed_read':False,'full_run_authorized':False}
    with Budget(OUT/identifier) as budget:
        tic=time.perf_counter();frame,comp,binding=load_inputs(identifier,base);result['input_seconds']=time.perf_counter()-tic
        result['input_binding']=binding;result['input_binding_sha256']=digest(binding)
        tic=time.perf_counter();observed,zero=n0(frame,identifier);result['N0']=zero;result['N0']['seconds']=time.perf_counter()-tic
        print(json.dumps({'befund':identifier,'phase':'N0','status':zero['status'],'auc_A_B':zero['auc_A_B'],'pool_difference':zero['pool_difference']}),flush=True)
        result['label_reference_time_summary']={str(s):{'n':len(f),'reference_time_delta_missing':int(f.time_delta_hours.isna().sum()),'delta_hours_min':None if f.time_delta_hours.isna().all() else float(f.time_delta_hours.min()),'delta_hours_max':None if f.time_delta_hours.isna().all() else float(f.time_delta_hours.max()),'alignment_confirmatory_eligible_n':int(f.ehr_time_alignment_confirmatory_eligible.fillna(False).sum()),'index_strategy':sorted(f.index_selection_strategy.unique()),'index_policy':sorted(f.index_policy.unique()),'time_precision':f.time_precision.value_counts(dropna=False).astype(int).to_dict()} for s,f in frame.groupby('split')}
        if zero['status']=='passed':
            result['N4']=n4(observed,zero,identifier,binding)
            guard();gc.collect();result['N2']=n2(observed,comp,identifier,binding);result['status']='completed'
        else:result['status']='stop_gate'
        result['resources']=budget.summary()
    result['seconds_this_invocation']=time.perf_counter()-started;result['finished_at']=datetime.now(timezone.utc).isoformat()
    result['resources']=budget.summary();path=OUT/identifier/(identifier+'.json');write(path,result)
    write(path.with_suffix('.sha256.json'),{'path':str(path),'sha256':sha(path),'created_at':result['finished_at']})
    print(json.dumps({'befund':identifier,'status':result['status'],'output':str(path),'seconds':result['seconds_this_invocation'],'resources':result['resources']}),flush=True)
    return 0 if result['status']=='completed' else 2

if __name__=='__main__':sys.exit(main())
