"""Technical extraction only; no association imports."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
import sys, json, hashlib, argparse
from pathlib import Path
from datetime import datetime, timezone
sys.dont_write_bytecode=True
from werkzeuge.nachrechnung.resources import guard, Budget
from . import testeingaben_aw137 as old
OUT=Path(str(data_path("b.aw138")))
ROOT=old.ROOT
RELEASE=old.RELEASE
PROJECT=old.PROJECT
SEED=1382026
SOURCES={}
VCG=list(dict.fromkeys(old.B3+[k for k in old.B4 if not k.startswith('classical_')]+['qt_interval_ms']+[h[2] for h in old.HYPOTHESES]))
CLASSICAL=['classical_cornell_product','classical_sokolow_lyon','classical_right_axis_deviation',
           'classical_st_elevation_contiguous_mm','classical_dominant_r_v1','classical_lbbb_pattern','classical_rbbb_pattern']
FIELDS=VCG+CLASSICAL+['classical_lbbb_pattern_positive','classical_rbbb_pattern_positive']
OPTIONS={'vcg_feature_ids':VCG,'classical_feature_ids':CLASSICAL,'sensitivity_feature_ids':[],
 'sensitivity_vcg_methods':[],'powerline_hz':60.,'delineation_weights':{'neurokit':.4,'multilead':.35,'vcg':.25},
 'delineation_conflict_threshold_ms':20.}
EXISTING_ONLY=['qrs_t_frechet_beat_median','qrs_t_loop_mirror_symmetry','qrs_t_normalized_dtw_distance']
LIMIT_OPTIONS={**OPTIONS,'vcg_feature_ids':[k for k in VCG if k not in EXISTING_ONLY]}
META=['subject_id','study_id','split','ecg_time','age','sex','waveform_path','machine_filtering','machine_bandwidth','machine_metadata_usable_for_primary',
      'machine_report','machine_qrs_axis_deg','machine_t_axis_deg','machine_qrs_duration_ms','machine_qt_interval_ms','machine_pr_interval_ms']

def sha(p): return old.sha(p)
def save(name,obj):
    p=OUT/name; p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=str),encoding='utf8')
def readj(p):
    p=Path(p)
    if p.stat().st_size>20*1024**2: raise ValueError('JSON >20MB')
    bind(p); return json.loads(p.read_text(encoding='utf8'))
def bind(p):
    p=Path(p)
    # No access to the research holdout; artifacts of the preceding stage are authorized separately.
    if 'research-holdout' in str(p).lower() or 'runtime_snapshots' in str(p).lower():
        raise PermissionError('holdout or runtime path blocked in the extraction stage')
    if str(p) not in SOURCES: SOURCES[str(p)]={'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def readp(p,cols=None,filt=None):
    import pyarrow.parquet as pq
    import pyarrow.dataset as ds
    guard(); p=Path(p); bind(p); pf=pq.ParquetFile(p)
    if cols is None:
        if p.stat().st_size>50*1024**2: raise ValueError('column selection required')
        cols=pf.schema_arrow.names
    cols=[c for c in cols if c in pf.schema_arrow.names]
    return ds.dataset(str(p),format='parquet').to_table(columns=cols,filter=filt).to_pandas()
def writep(name,f):
    import pyarrow as pa
    import pyarrow.parquet as pq
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    pq.write_table(pa.Table.from_pandas(f,preserve_index=False),p,row_group_size=8192,compression='zstd')
def inspect():
    import pyarrow.parquet as pq
    manifest=readj(old.OUT/'input_manifest_v2.json')
    package=readj(PROJECT/'werkzeuge/b_test/parameter_entwicklung.json')
    master=ROOT/'master/ecg_master_index.parquet';bind(master)
    schemas={'master':pq.ParquetFile(master).schema_arrow.names}
    caches=list((old.RUN/'analysis_feature_cache').glob('cache-*/cache.manifest.json'))
    if len(caches)!=1:raise ValueError('development cache not unique')
    cm=readj(caches[0]);save('cache_manifest.json',cm)
    for s in ('train','validation'):
        p=caches[0].parent/f'split={s}/features.parquet'
        schemas[s]={'path':str(p),'columns':pq.ParquetFile(p).schema_arrow.names,'bytes':p.stat().st_size}
    save('schemas.json',schemas)
    print(json.dumps({'master_columns':schemas['master'],'cache':str(caches[0]),'cache_keys':list(cm),
      'manifest_keys':list(manifest),'sources_n':len(manifest.get('source_candidates',[])),
      'development_inputs':{k:v['input_development'] for k,v in package['hypotheses'].items()},
      'train_columns_required': [x for x in schemas['train']['columns'] if x in old.B3+old.B4+old.EXTRA+[h[2] for h in old.HYPOTHESES]+[h[3] for h in old.HYPOTHESES]]}),flush=True)

def release():
    sys.path.insert(0,str(RELEASE.parent))
    from vcg_biomarker.mimic.feature_record import _extract_record_worker
    from vcg_biomarker.mimic.feature_extraction import feature_implementation_hash
    expected=readj(old.OUT/'provenance.json')['feature_implementation_hash_release']
    if feature_implementation_hash()!=expected:raise ValueError('release hash changed')
    for n,m in tuple(sys.modules.items()):
        if n.startswith('vcg_biomarker') and getattr(m,'__file__',None):
            if not Path(m.__file__).resolve().is_relative_to(RELEASE.resolve()):raise ValueError('wrong import state')
    return _extract_record_worker

def earlier_unknown(f):
    """All unknown technical candidates before the first known success."""
    import pandas as pd
    z=f.sort_values(['subject_id','ecg_time','study_id'],kind='stable').copy()
    z['__seen_ok']=z.extraction_status.eq('ok').groupby(z.subject_id).cumsum()
    return z.loc[z.extraction_status.isna() & z.__seen_ok.eq(0),['subject_id','study_id','ecg_time']]

def prepare():
    import pandas as pd
    import numpy as np
    import pyarrow.dataset as ds
    schemas=readj(str(data_path("b.aw137_schemas_json")))
    cp=ROOT/'evidence/clinical_context_features.parquet'
    cols=next(s['columns'] for p,s in schemas.items() if Path(p)==cp)
    # Same explicit context projection as in the preceding stage, no raw endpoints in the extraction request.
    template=readp(old.OUT/'b_test_eingaben/BEF-0063_test.parquet')
    cols=[k for k in template.columns if k in cols]
    ctx=readp(cp,cols,ds.field('split')=='locked_test')
    feat=readp(old.OUT/'features_projected_private.parquet',['subject_id','study_id','extraction_status'])
    f=ctx.merge(feat,on=['subject_id','study_id'],how='left',validate='one_to_one')
    labelcols=['subject_id','study_id','label_id','label_state','eligibility_state','time_delta_hours','time_precision','evidence_ids_json','rule_version']
    queues=[];flows={}
    for bef,ep,marker,comp in old.HYPOTHESES:
        lab=readp(old.OUT/'labels_reproduced_private.parquet',labelcols,ds.field('label_id')==ep)
        z=f.merge(lab,on=['subject_id','study_id'],validate='one_to_one')
        ref=z.eligibility_state.eq('eligible')&z.label_state.isin(['positive','negative'])
        if ep in ('reduced_lvef_strict','echo_rv_systolic_dysfunction'):
            ref &= z.time_precision.eq('timestamp')&z.time_delta_hours.abs().le(720)&np.isclose(z.time_delta_hours,z.nearest_tte_delta_hours,rtol=0,atol=1e-9,equal_nan=False)
            ref &= z.apply(old.echo_id_matches,axis=1)
            if ep=='reduced_lvef_strict':ref &= z.preferred_lvef.between(0,100)&np.isfinite(z.preferred_lvef)&z.lvef_conflict_gt_10.eq(False)
        elif ep in ('elevated_ntprobnp','acute_decompensated_heart_failure_proxy'):
            ref &= z.time_precision.eq('timestamp')&z.time_delta_hours.between(-6,12)
        z['reference_available']=ref.fillna(False)
        candidates=z.loc[z.reference_available&z.age.ge(18)].copy()
        flows[bef]=old.coverage(candidates)
        writep(f'candidates/{bef}_private.parquet',candidates)
        q=earlier_unknown(candidates);q['part']=bef;queues.append(q)
    t=f.loc[f.age.ge(18)].copy();writep('candidates/technical_private.parquet',t)
    flows['technical']=old.coverage(t);q=earlier_unknown(t);q['part']='technical';queues.append(q)
    queue=pd.concat(queues,ignore_index=True);writep('earlier_candidate_membership_private.parquet',queue)
    unique=queue.drop_duplicates('study_id')
    meta=readp(ROOT/'master/ecg_master_index.parquet',META,ds.field('study_id').isin(unique.study_id.tolist()))
    if len(meta)!=len(unique) or not meta.split.eq('locked_test').all():raise ValueError('candidate metadata not bound')
    writep('earlier_tasks_private.parquet',meta)
    baseline=readj(old.OUT/'index_coverage.json')
    for bef in flows:
        target=baseline['technical_universe'] if bef=='technical' else baseline['hypotheses'][bef]
        for k,v in flows[bef].items():
            if target.get(k)!=v:raise ValueError('candidate denominators differ from the preceding stage: '+bef+'/'+k)
    save('prepare_summary.json',{'baseline_coverage':flows,'earlier_candidate_ecgs_unique':len(unique),
      'earlier_candidate_memberships':len(queue),'by_part':{p:len(q) for p,q in queue.groupby('part')},
      'seed':SEED,'options':OPTIONS,'script_sha256':sha(__file__)})
    development_prepare()
    print(json.dumps({'phase':'prepare','earlier_unique':len(unique),'baseline_unresolved':{k:v['unresolved_persons'] for k,v in flows.items()}}),flush=True)

def development_prepare():
    import pandas as pd
    import numpy as np
    import pyarrow.dataset as ds
    # Uniform random sample from the stored development ECG universe,
    # without outcome, quality or marker filter; NumPy generator with fixed seed.
    schema=readj(OUT/'schemas.json');frames=[]
    for s in ('train','validation'):
        p=Path(schema[s]['path']);part=readp(p,['subject_id','study_id','split']);part['split']=s;frames.append(part)
    dev=pd.concat(frames,ignore_index=True).sort_values(['study_id'],kind='stable').reset_index(drop=True)
    if dev.study_id.duplicated().any() or not set(dev.split)<= {'train','validation'}:raise ValueError('development identity')
    sampled=dev.iloc[np.random.default_rng(SEED).choice(len(dev),200,replace=False)].sort_values('study_id')
    meta=readp(ROOT/'master/ecg_master_index.parquet',META,ds.field('study_id').isin(sampled.study_id.tolist()))
    writep('equivalence_tasks_private.parquet',meta)
    expected=[]
    for s in ('train','validation'):
        p=Path(schema[s]['path']);part=readp(p,['subject_id','study_id','split','extraction_status','signal_sha256']+FIELDS,ds.field('study_id').isin(sampled.study_id.tolist()));part['split']=s;expected.append(part)
    e=pd.concat(expected,ignore_index=True);writep('equivalence_expected_private.parquet',e)
    save('equivalence_selection.json',{'seed':SEED,'requested_n':200,'universe_ecgs':len(dev),
      'sampling':'NumPy default_rng choice without replacement; sorted study_id pool train+validation, no value, outcome or success selection',
      'request_sha256':sha(OUT/'equivalence_tasks_private.parquet'),'expected_sha256':sha(OUT/'equivalence_expected_private.parquet'),
      'by_split':sampled.split.value_counts().to_dict(),'options':OPTIONS})
    print(json.dumps({'phase':'development_prepare','equivalence_n':len(e),'universe':len(dev)}),flush=True)

def extract_tasks(taskfile,folder,limit=None,fast=False,stripe=None,reuse_full=False,initialized=False,max_new=None):
    import time
    import pandas as pd
    from collections import Counter
    worker=release();tasks=readp(OUT/taskfile).sort_values('study_id')
    if initialized:
        from vcg_biomarker.mimic import feature_record as fr
        fr._initialize_feature_worker(OPTIONS)
        worker=lambda task:fr._extract_record_worker(task[0])
    if fast:
        from .marker_adapter_aw138 import measure
        worker=lambda task:measure(*task)
    if stripe is not None:tasks=tasks.iloc[stripe::3]
    if limit: tasks=tasks.head(limit)
    d=OUT/folder;d.mkdir(exist_ok=True);counts=Counter();start=time.perf_counter()
    options_hash=hashlib.sha256(json.dumps(OPTIONS,sort_keys=True).encode()).hexdigest()
    request_hash=sha(OUT/taskfile)
    adapter_hash=sha(PROJECT/'werkzeuge/b_test/marker_adapter_aw138.py') if fast else None
    new_count=0;complete=True
    for i,row in enumerate(tasks.to_dict('records')):
        p=d/f"{int(row['study_id'])}.json"
        if p.exists():
            obj=json.loads(p.read_text(encoding='utf8'))
            if obj['options_sha256']!=options_hash or obj['request_sha256']!=request_hash or obj.get('adapter_sha256')!=adapter_hash:raise ValueError('resume binding')
            counts[obj['result']['extraction_status']]+=1;continue
        if reuse_full:
            prior=OUT/'marker_records'/p.name
            if prior.exists():
                try:obj=json.loads(prior.read_text(encoding='utf8'))
                except json.JSONDecodeError:obj=None
                if obj is not None:
                    if obj['request_sha256']!=request_hash:raise ValueError('reuse task changed')
                    obj.update(request_sha256=request_hash,options_sha256=options_hash,
                      reused_from={'path':str(prior),'sha256':sha(prior),'subset_equivalence_attestation_sha256':sha(OUT/'limited_equivalence_attestation.json')})
                    save(f'{folder}/{p.name}',obj);counts[obj['result']['extraction_status']]+=1;continue
        guard()
        import contextlib,io
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            result=worker((row,OPTIONS))
        kept={k:result.get(k) for k in ['subject_id','study_id','split','ecg_time','extraction_status','error_type','error_message','original_sampling_rate','signal_sha256','biomarker_calculation_failure_details','feature_unevaluable_reason_json','feature_confidence_json','classical_unevaluable_reason_json']+FIELDS}
        # NaN only as private JSON null, no marker values in the progress log.
        def clean(v):
            import math
            if isinstance(v,float) and not math.isfinite(v):return None
            return v
        kept={k:clean(v) for k,v in kept.items()}
        save(f'{folder}/{int(row["study_id"])}.json',{'request_sha256':request_hash,'options_sha256':options_hash,'adapter_sha256':adapter_hash,'result':kept})
        counts[kept['extraction_status']]+=1
        new_count+=1
        if (i+1)%(100 if stripe is not None else 10)==0 or i+1==len(tasks):print(json.dumps({'phase':folder,'stripe':stripe,'done':i+1,'total':len(tasks),'seconds':round(time.perf_counter()-start,1),'status_counts':dict(counts)}),flush=True)
        if max_new is not None and new_count>=max_new and i+1<len(tasks):
            complete=False;break
    suffix='' if stripe is None else '_'+str(stripe)
    summary={'requested':len(tasks),'completed_prefix':sum(counts.values()),'new_in_this_invocation':new_count,'complete':complete,
      'status_counts':dict(counts),'seconds':time.perf_counter()-start,'request_sha256':request_hash,'options_sha256':options_hash,'adapter_sha256':adapter_hash}
    save(folder+suffix+('_summary.json' if complete else '_partial_summary.json'),summary)
    print(json.dumps({'phase':'block_finished','stripe':stripe,'new':new_count,'completed_prefix':sum(counts.values()),'total':len(tasks),'complete':complete}),flush=True)

def equivalence():
    import pandas as pd
    import numpy as np
    extract_tasks('equivalence_tasks_private.parquet','equivalence_records')
    a=pd.DataFrame([json.loads(p.read_text(encoding='utf8'))['result'] for p in sorted((OUT/'equivalence_records').glob('*.json'))])
    e=readp(OUT/'equivalence_expected_private.parquet');z=e.merge(a,on=['subject_id','study_id','split'],suffixes=('_expected','_actual'),validate='one_to_one')
    checks={};maxdiff={}
    for k in FIELDS:
        x=pd.to_numeric(z[k+'_expected'],errors='coerce');y=pd.to_numeric(z[k+'_actual'],errors='coerce')
        match=np.isclose(x,y,rtol=1e-8,atol=0,equal_nan=True)
        checks[k]=int((~match).sum());delta=(x-y).abs().dropna();maxdiff[k]=None if delta.empty else float(delta.max())
    checks['extraction_status']=int(z.extraction_status_expected.ne(z.extraction_status_actual).sum())
    checks['signal_sha256']=int((~(z.signal_sha256_expected.eq(z.signal_sha256_actual)|(z.signal_sha256_expected.isna()&z.signal_sha256_actual.isna()))).sum())
    writep('equivalence_comparison_private.parquet',z)
    save('equivalence_attestation.json',{'passed':len(z)==200 and not any(checks.values()),'n':len(z),'seed':SEED,
      'rtol':1e-8,'atol':0,'missingness_exact':True,'mismatch_counts':checks,'max_absolute_difference':maxdiff,
      'selection':readj(OUT/'equivalence_selection.json'),'script_sha256':sha(__file__)})
    print(json.dumps({'phase':'equivalence','n':len(z),'mismatches':checks,'passed':len(z)==200 and not any(checks.values())}),flush=True)

def earlier():extract_tasks('earlier_tasks_private.parquet','earlier_records')

def pilot():extract_tasks('equivalence_tasks_private.parquet','equivalence_records',10)

def fast_equivalence():
    import pandas as pd
    import numpy as np
    if readj(OUT/'equivalence_attestation_v2.json')['n']!=200:raise ValueError('development equivalence check incomplete')
    extract_tasks('equivalence_tasks_private.parquet','adapter_equivalence_records',fast=True)
    a=pd.DataFrame([json.loads(p.read_text(encoding='utf8'))['result'] for p in sorted((OUT/'adapter_equivalence_records').glob('*.json'))])
    b=pd.DataFrame([json.loads(p.read_text(encoding='utf8'))['result'] for p in sorted((OUT/'equivalence_records').glob('*.json'))])
    z=b.merge(a,on=['subject_id','study_id','split'],suffixes=('_full','_adapter'),validate='one_to_one');checks={}
    for k in FIELDS:
        checks[k]=int((~np.isclose(pd.to_numeric(z[k+'_full'],errors='coerce'),pd.to_numeric(z[k+'_adapter'],errors='coerce'),rtol=1e-8,atol=0,equal_nan=True)).sum())
    for k in ['extraction_status','signal_sha256','feature_unevaluable_reason_json','feature_confidence_json']:
        checks[k]=int((~(z[k+'_full'].eq(z[k+'_adapter'])|(z[k+'_full'].isna()&z[k+'_adapter'].isna()))).sum())
    writep('adapter_equivalence_comparison_private.parquet',z)
    save('adapter_equivalence_attestation.json',{'passed':len(z)==200 and not any(checks.values()),'n':len(z),'seed':SEED,'rtol':1e-8,'atol':0,'mismatch_counts':checks,
      'adapter_sha256':sha(PROJECT/'werkzeuge/b_test/marker_adapter_aw138.py'),'script_sha256':sha(__file__),
      'scope':'Same release measurement functions without unneeded capsule outputs; historical technical statuses remain separate in the index contract.'})
    print(json.dumps({'phase':'adapter_equivalence','passed':not any(checks.values()),'mismatches':checks}),flush=True)

def index_prepare():
    import pandas as pd
    import pyarrow.dataset as ds
    if readj(OUT/'equivalence_attestation.json')['n']!=200:raise ValueError('development equivalence check incomplete')
    if not readj(OUT/'equivalence_attestation_v2.json')['passed']:raise ValueError('release equivalence gate')
    rows=[json.loads(p.read_text(encoding='utf8'))['result'] for p in sorted((OUT/'earlier_records_v2').glob('*.json'))]
    values=pd.DataFrame(rows);tasks=readp(OUT/'earlier_tasks_private.parquet',['subject_id','study_id'])
    if len(values)!=len(tasks) or set(values.study_id)!=set(tasks.study_id):raise ValueError('earlier candidates incomplete')
    writep('earlier_values_private.parquet',values)
    flows={};chosen=[]
    for part in [x[0] for x in old.HYPOTHESES]+['technical']:
        f=readp(OUT/f'candidates/{part}_private.parquet')
        status=values.set_index('study_id').extraction_status
        before=old.coverage(f)
        f['extraction_status']=f.extraction_status.combine_first(f.study_id.map(status))
        after=old.coverage(f)
        if after['unresolved_persons']!=0:raise ValueError('index gap remains: '+part)
        selected=f.loc[f.extraction_status.eq('ok')].sort_values(['ecg_time','study_id'],kind='stable').drop_duplicates('subject_id').copy()
        writep(f'indices/{part}_private.parquet',selected)
        chosen.append(selected[['subject_id','study_id']]);flows[part]={'before':before,'after':after}
    idx=pd.concat(chosen,ignore_index=True).drop_duplicates('study_id')
    meta=project_worker_inputs(readp(ROOT/'master/ecg_master_index.parquet',META,ds.field('study_id').isin(idx.study_id.tolist())))
    if len(meta)!=len(idx) or not meta.split.eq('locked_test').all():raise ValueError('index tasks incomplete')
    writep('marker_tasks_private.parquet',meta)
    save('index_coverage.json',{'parts':flows,'unique_index_ecgs':len(idx),'script_sha256':sha(__file__),
      'recipe':'adult + endpoint-specific known valid reference + release status ok; ecg_time/study_id; technical selection without marker missingness; time-of-day gate separate',
      'earlier_tasks_sha256':sha(OUT/'earlier_tasks_private.parquet'),'earlier_values_sha256':sha(OUT/'earlier_values_private.parquet')})
    print(json.dumps({'phase':'index_prepare','unique_index_ecgs':len(idx),'parts':{k:v['after'] for k,v in flows.items()}}),flush=True)

def markers(stripe):
    eq=readj(OUT/'adapter_equivalence_attestation.json')
    if not eq['passed'] or eq['adapter_sha256']!=sha(PROJECT/'werkzeuge/b_test/marker_adapter_aw138.py'):raise ValueError('adapter equivalence gate')
    if not readj(OUT/'equivalence_attestation_v2.json')['passed']:raise ValueError('release development equivalence gate')
    extract_tasks('marker_tasks_private.parquet','marker_records',fast=False,stripe=stripe)
def markers0():markers(0)
def markers1():markers(1)
def markers2():markers(2)

def tests():
    import pytest
    code=pytest.main(['-p','no:cacheprovider','werkzeuge/b_test/tests/test_aw138.py','-q','--junitxml='+str(OUT/'tests/junit.xml')])
    if code:raise SystemExit(code)

def diagnose():
    import numpy as np
    import pandas as pd
    import pyarrow.dataset as ds
    z=readp(OUT/'equivalence_comparison_private.parquet');bad=np.zeros(len(z),dtype=bool)
    for k in FIELDS:bad |= ~np.isclose(pd.to_numeric(z[k+'_expected'],errors='coerce'),pd.to_numeric(z[k+'_actual'],errors='coerce'),rtol=1e-8,atol=0,equal_nan=True)
    selected=z.loc[bad].copy();writep('equivalence_mismatches_private.parquet',selected)
    detail={}
    for k in FIELDS:
        x=pd.to_numeric(selected[k+'_expected'],errors='coerce').astype(float);y=pd.to_numeric(selected[k+'_actual'],errors='coerce').astype(float)
        detail[k]={'expected_missing':int(x.isna().sum()),'actual_missing':int(y.isna().sum()),'max_abs_diff':None if (x-y).dropna().empty else float((x-y).abs().max())}
    # Re-binding to original development fragments, only for the 200 requests.
    cm=readj(OUT/'cache_manifest.json');root=Path(cm['source_root']);source=[]
    for s in ('train','validation'):
        for p in sorted((root/f'split={s}').glob('part-*.json')):
            side=readj(p)
            if not set(side.get('study_ids',[]))&set(selected.study_id):continue
            q=p.with_suffix('.parquet');part=readp(q,['subject_id','study_id','split','extraction_status','signal_sha256']+FIELDS,ds.field('study_id').isin(selected.study_id.tolist()))
            part['split']=s;source.append(part)
    original=pd.concat(source,ignore_index=True);writep('equivalence_source_mismatches_private.parquet',original)
    comp=original.merge(selected,on=['subject_id','study_id','split'],validate='one_to_one')
    source_checks={}
    for k in FIELDS:
        source_checks[k]={'source_vs_cache_mismatches':int((~np.isclose(pd.to_numeric(comp[k],errors='coerce'),pd.to_numeric(comp[k+'_expected'],errors='coerce'),rtol=1e-8,atol=0,equal_nan=True)).sum()),
          'source_vs_recomputed_mismatches':int((~np.isclose(pd.to_numeric(comp[k],errors='coerce'),pd.to_numeric(comp[k+'_actual'],errors='coerce'),rtol=1e-8,atol=0,equal_nan=True)).sum())}
    save('equivalence_diagnosis.json',{'affected_ecgs':len(selected),'status_expected':selected.extraction_status_expected.value_counts().to_dict(),'status_actual':selected.extraction_status_actual.value_counts().to_dict(),'field_differences':detail,'source_fragment_checks':source_checks})
    print(json.dumps(readj(OUT/'equivalence_diagnosis.json')),flush=True)

def diagnose_repeat():
    import numpy as np
    import pandas as pd
    import pyarrow.dataset as ds
    affected=readp(OUT/'equivalence_mismatches_private.parquet');tasks=readp(OUT/'equivalence_tasks_private.parquet',filt=ds.field('study_id').isin(affected.study_id.tolist()))
    worker=release();values=[]
    for row in tasks.to_dict('records'):
        guard();values.append(worker((row,OPTIONS)))
    result=pd.DataFrame(values);writep('equivalence_repeat_private.parquet',result)
    z=affected.merge(result,on=['subject_id','study_id','split'],validate='one_to_one');check={}
    for k in FIELDS:
        check[k]=int((~np.isclose(pd.to_numeric(z[k+'_actual'],errors='coerce'),pd.to_numeric(z[k],errors='coerce'),rtol=1e-8,atol=0,equal_nan=True)).sum())
    save('equivalence_repeat_attestation.json',{'n':len(z),'original_recompute_vs_repeat_mismatches':check,'deterministic_in_this_repeat':not any(check.values()),
      'cause':'Not yet explained; no cache or source fragment difference, raw signal hash identical. No parameter change or case removal.'})
    print(json.dumps({'phase':'diagnose_repeat','deterministic':not any(check.values()),'n':len(z)}),flush=True)

def project_worker_inputs(f):
    """Exactly _load_feature_cohort: report, axes and intervals restricted to time-suitable sources.

    Bandwidth and filter remain unmasked, following optional_column_select.
    """
    f=f.copy();usable=f.machine_metadata_usable_for_primary.eq(True).fillna(False)
    for k in ['machine_report','machine_qrs_axis_deg','machine_t_axis_deg','machine_qrs_duration_ms','machine_qt_interval_ms','machine_pr_interval_ms']:
        f[k]=f[k].where(usable,None)
    return f

def correct_inputs():
    import pyarrow.dataset as ds
    import pandas as pd
    summaries={}
    for name in ['equivalence','earlier']:
        before=readp(OUT/f'{name}_tasks_private.parquet')
        after=project_worker_inputs(readp(ROOT/'master/ecg_master_index.parquet',META,ds.field('study_id').isin(before.study_id.tolist())))
        if set(after.study_id)!=set(before.study_id):raise ValueError('input correction changed the sample')
        writep(f'{name}_tasks_v2_private.parquet',after)
        summaries[name]={'rows':len(after),'metadata_not_usable':int((~after.machine_metadata_usable_for_primary.eq(True)).sum()),
          'v1_sha256':sha(OUT/f'{name}_tasks_private.parquet'),'v2_sha256':sha(OUT/f'{name}_tasks_v2_private.parquet')}
    save('input_projection_correction.json',{'cause':'v1 read device report, axes and intervals unmasked from the master index; the release _load_feature_cohort sets them to NULL when the metadata source is not time-suitable. Bandwidth and filter remain unmasked.',
      'source_code':str(RELEASE/'mimic/feature_extraction.py'),'source_sha256':sha(RELEASE/'mimic/feature_extraction.py'),
      'code_locations':'_machine_interval_select and _load_feature_cohort in the release source',
      'no_changed_sampling_seed_tolerance_or_measurement_recipe':True,'no_association_results_seen':True,'tasks':summaries})
    print(json.dumps({'phase':'input_projection_correction','tasks':summaries}),flush=True)

def corrected_extract(name):
    import pandas as pd
    # Unchanged effective worker inputs are taken over from v1 (hash-bound).
    # Every changed row is re-extracted with the same release measurement parameters.
    tasks=readp(OUT/f'{name}_tasks_v2_private.parquet');before=readp(OUT/f'{name}_tasks_private.parquet').set_index('study_id',drop=False)
    request_hash=sha(OUT/f'{name}_tasks_v2_private.parquet');option_hash=hashlib.sha256(json.dumps(OPTIONS,sort_keys=True).encode()).hexdigest()
    fields=[k for k in META if k!='machine_metadata_usable_for_primary'];reused=0;changed=0
    for row in tasks.to_dict('records'):
        p=OUT/f'{name}_records/{int(row["study_id"])}.json';target=OUT/f'{name}_records_v2/{int(row["study_id"])}.json'
        same=True;oldrow=before.loc[row['study_id']]
        for k in fields:
            x=oldrow[k];y=row[k]
            if not ((pd.isna(x) and pd.isna(y)) or x==y):same=False;break
        if same and p.exists():
            value=json.loads(p.read_text(encoding='utf8'))
            if value['options_sha256']!=option_hash:raise ValueError('options changed on reuse')
            value.update(request_sha256=request_hash,reused_from={'path':str(p),'sha256':sha(p),'effective_worker_inputs_equal':True})
            save(f'{name}_records_v2/{int(row["study_id"])}.json',value);reused+=1
        else:changed+=1
    extract_tasks(f'{name}_tasks_v2_private.parquet',f'{name}_records_v2')
    save(f'{name}_v2_reuse_attestation.json',{'effective_input_columns':fields,'same_inputs_reused':reused,'new_or_changed_inputs_recomputed':changed,'input_projection_correction_sha256':sha(OUT/'input_projection_correction.json')})

def equivalence_v2():
    import pandas as pd
    import numpy as np
    corrected_extract('equivalence')
    a=pd.DataFrame([json.loads(p.read_text(encoding='utf8'))['result'] for p in sorted((OUT/'equivalence_records_v2').glob('*.json'))])
    e=readp(OUT/'equivalence_expected_private.parquet');z=e.merge(a,on=['subject_id','study_id','split'],suffixes=('_expected','_actual'),validate='one_to_one')
    checks={};maxdiff={}
    for k in FIELDS:
        x=pd.to_numeric(z[k+'_expected'],errors='coerce').astype(float);y=pd.to_numeric(z[k+'_actual'],errors='coerce').astype(float)
        checks[k]=int((~np.isclose(x,y,rtol=1e-8,atol=0,equal_nan=True)).sum());d=(x-y).abs().dropna();maxdiff[k]=None if d.empty else float(d.max())
    for k in ['extraction_status','signal_sha256']:checks[k]=int((~(z[k+'_expected'].eq(z[k+'_actual'])|(z[k+'_expected'].isna()&z[k+'_actual'].isna()))).sum())
    writep('equivalence_comparison_v2_private.parquet',z)
    save('equivalence_attestation_v2.json',{'passed':len(z)==200 and not any(checks.values()),'n':len(z),'seed':SEED,'rtol':1e-8,'atol':0,'missingness_exact':True,
      'mismatch_counts':checks,'max_absolute_difference':maxdiff,'selection':readj(OUT/'equivalence_selection.json'),
      'request_sha256':sha(OUT/'equivalence_tasks_v2_private.parquet'),'correction_sha256':sha(OUT/'input_projection_correction.json'),'script_sha256':sha(__file__)})
    print(json.dumps({'phase':'equivalence_v2','n':len(z),'passed':not any(checks.values()),'mismatches':checks}),flush=True)

def earlier_v2():corrected_extract('earlier')

def limited_equivalence():
    import pandas as pd
    import numpy as np
    global OPTIONS
    original=OPTIONS;OPTIONS=LIMIT_OPTIONS
    try:extract_tasks('equivalence_tasks_v2_private.parquet','limited_equivalence_records',initialized=True)
    finally:OPTIONS=original
    a=pd.DataFrame([json.loads(p.read_text(encoding='utf8'))['result'] for p in sorted((OUT/'limited_equivalence_records').glob('*.json'))])
    expected=readp(OUT/'equivalence_expected_private.parquet');z=expected.merge(a,on=['subject_id','study_id','split'],suffixes=('_expected','_actual'),validate='one_to_one')
    checks={}
    for k in [k for k in FIELDS if k not in EXISTING_ONLY]:
        checks[k]=int((~np.isclose(pd.to_numeric(z[k+'_expected'],errors='coerce'),pd.to_numeric(z[k+'_actual'],errors='coerce'),rtol=1e-8,atol=0,equal_nan=True)).sum())
    for k in ['extraction_status','signal_sha256']:checks[k]=int((~(z[k+'_expected'].eq(z[k+'_actual'])|(z[k+'_expected'].isna()&z[k+'_actual'].isna()))).sum())
    save('limited_equivalence_attestation.json',{'passed':len(z)==200 and not any(checks.values()),'n':len(z),'seed':SEED,'rtol':1e-8,'atol':0,'mismatch_counts':checks,
      'options':LIMIT_OPTIONS,'initialized_release_worker':True,'script_sha256':sha(__file__),
      'reason':'Only markers that are already available are not requested again; all missing IDs and the B3/B4, ST and block measurement fields are unchanged. No change to signal, delineation, transform or normalization parameters.'})
    writep('limited_equivalence_comparison_private.parquet',z)
    print(json.dumps({'phase':'limited_equivalence','passed':not any(checks.values()),'mismatches':checks}),flush=True)

def limited_markers(stripe):
    global OPTIONS
    if not readj(OUT/'equivalence_attestation_v2.json')['passed'] or not readj(OUT/'limited_equivalence_attestation.json')['passed']:raise ValueError('equivalence gate')
    original=OPTIONS;OPTIONS=LIMIT_OPTIONS
    try:extract_tasks('marker_tasks_private.parquet','marker_records_v2',stripe=stripe,reuse_full=True,initialized=True)
    finally:OPTIONS=original
def limited0():limited_markers(0)
def limited1():limited_markers(1)
def limited2():limited_markers(2)

def batch(stripe):
    global OPTIONS
    if not readj(OUT/'equivalence_attestation_v2.json')['passed'] or not readj(OUT/'limited_equivalence_attestation.json')['passed']:raise ValueError('equivalence gate')
    original=OPTIONS;OPTIONS=LIMIT_OPTIONS
    try:extract_tasks('marker_tasks_private.parquet','marker_records_v2',stripe=stripe,reuse_full=True,initialized=True,max_new=400)
    finally:OPTIONS=original
def batch0():batch(0)
def batch1():batch(1)
def batch2():batch(2)

def profile():
    import cProfile,pstats,time
    from .marker_adapter_aw138 import measure
    tasks=readp(OUT/'equivalence_tasks_v2_private.parquet').sort_values('study_id').head(10).to_dict('records')
    worker=release();results={}
    for name,fn in [('full',lambda r:worker((r,LIMIT_OPTIONS))),('adapter',lambda r:measure(r,LIMIT_OPTIONS))]:
        profiler=cProfile.Profile();start=__import__('time').perf_counter();profiler.enable()
        for row in tasks:guard();fn(row)
        profiler.disable();elapsed=__import__('time').perf_counter()-start;profiler.dump_stats(str(OUT/f'profile_{name}.prof'))
        stat=pstats.Stats(profiler);top=sorted(stat.stats.items(),key=lambda item:item[1][3],reverse=True)[:25]
        results[name]={'seconds':elapsed,'top':[{'file':key[0],'line':key[1],'function':key[2],'calls':v[1],'self_seconds':v[2],'cumulative_seconds':v[3]} for key,v in top]}
    save('profile_summary.json',results);print(json.dumps(results),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode');args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);start=guard()
    snap=OUT/'executed_source'/sha(__file__)/Path(__file__).name;snap.parent.mkdir(parents=True,exist_ok=True);snap.write_bytes(Path(__file__).read_bytes())
    launch_sha=sha(__file__)
    with (OUT/'execution_history.jsonl').open('a',encoding='utf8') as log:
        log.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'phase':args.mode,'event':'start','script_sha256':launch_sha,'memory':start})+'\n')
    with Budget(OUT) as budget: globals()[args.mode]()
    resource={'start':start,**budget.summary(),'script_sha256':launch_sha,'utc_end':datetime.now(timezone.utc).isoformat()}
    save('resources_'+args.mode+'.json',resource)
    with (OUT/'execution_history.jsonl').open('a',encoding='utf8') as log:
        log.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'phase':args.mode,'event':'end',**resource})+'\n')
    save('sources_'+args.mode+'.json',list(SOURCES.values()))
