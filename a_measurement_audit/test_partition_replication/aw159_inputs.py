"""AW159: wertfreie Eingangsbindung; keine Replikationsstatistik."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,ctypes,time,traceback,gc
from pathlib import Path
from datetime import datetime,timezone
sys.dont_write_bytecode=True
PROJECT=Path(__file__).resolve().parents[2]
OUT=Path(str(data_path("a.aw159")))
DATA=Path(str(data_path("a.mimic_validation")))
RUN=DATA/'runs/A155-2026-10-04-d6e70e'
REL=Path(str(data_path("a.release_sources_a155_survival_nullable_20261007_vcg_biomarker")))
SPEC=PROJECT/'nachrechnung/SPEZIFIKATION_A_TESTREPLIKATION_v1.md'
SPEC_SHA='4cdb6d32c151a758cc947bf0e679b1d1b455ae714993dd36d8e3930e4f60dd28'
SOURCES={}
class Memory(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('tp','ap','tc','ac','tv','av','ae')]
def guard():
    import psutil
    while True:
        m=Memory();m.length=ctypes.sizeof(m)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise RuntimeError('Speichergate unbekannt')
        p=psutil.Process().memory_info()
        if max(p.rss,getattr(p,'private',p.vms))>1.5*1024**3:raise MemoryError('Eigener Prozess >1,5 GiB')
        if min(m.ap,m.ac)>=1.5*1024**3:return {'ram_free_bytes':m.ap,'commit_free_bytes':m.ac,'rss':p.rss,'private':getattr(p,'private',p.vms)}
        save('resource_wait.json',{'utc':utc(),'pid':os.getpid(),'ram':m.ap,'commit':m.ac});time.sleep(5)
def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def save(name,obj):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=str),encoding='utf8')
def ledger(event,scope):
    with (PROJECT/'koordination/TESTZUGRIFFE.md').open('a',encoding='utf8') as f:
        runner=globals().get('launch_sha') or sha(__file__)
        f.write(f'\n| {utc()} | AW-159 | {event} | Vorschrift `{SPEC_SHA}`; Runner `{runner}` | {scope} |\n')
def bind(p,cols=None,role='source',rows=None):
    p=Path(p);prior=SOURCES.get(str(p),{})
    SOURCES[str(p)]={'path':str(p),'sha256':prior.get('sha256') or sha(p),'bytes':p.stat().st_size,'columns':list(dict.fromkeys(prior.get('columns',[])+(cols or []))),'rows':rows if rows is not None else prior.get('rows'),'role':role}
    return SOURCES[str(p)]
def readj(p,test=False):
    p=Path(p)
    if p.stat().st_size>20*1024**2:raise ValueError('JSON >20 MB')
    if test:ledger('Beginn/Fortsetzung Provenienzlesung',str(p)+'; nur Struktur/Erzeuger')
    bind(p,role='provenance');result=json.loads(p.read_text(encoding='utf8'))
    if test:ledger('Ende Provenienzlesung',str(p)+'; Bytehash '+SOURCES[str(p)]['sha256'])
    return result
def readp(p,cols,filt=None,test=False):
    import pyarrow.parquet as pq,pyarrow.dataset as ds
    guard();p=Path(p)
    if test:
        ledger('Beginn/Fortsetzung Projektion',str(p)+'; Spalten '+','.join(cols)+'; Filter '+('gebundene private Auswahl auf E: (keine Kennungen im Register)' if filt is not None else 'keiner'))
        with (OUT/'access_log_private.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps({'utc':utc(),'path':str(p),'columns':cols,'filter':str(filt)},ensure_ascii=False)+'\n')
    pf=pq.ParquetFile(p)
    missing=set(cols)-set(pf.schema_arrow.names)
    if missing:raise ValueError('Fehlende vertragliche Spalten: '+','.join(sorted(missing)))
    bind(p,cols,rows=pf.metadata.num_rows)
    result=ds.dataset(str(p),format='parquet').to_table(columns=cols,filter=filt).to_pandas()
    if test:ledger('Ende Projektion',str(p)+'; Bytehash '+SOURCES[str(p)]['sha256']+'; projizierte Zeilen '+str(len(result)))
    return result
def writep(name,f):
    import pyarrow as pa,pyarrow.parquet as pq
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);pq.write_table(pa.Table.from_pandas(f,preserve_index=False),p,compression='zstd',row_group_size=8192)
def conn():
    import duckdb
    c=duckdb.connect();c.execute("SET memory_limit='1GB'");c.execute('SET threads=2');c.execute(("SET temp_directory='" + str(data_path("a.aw159_duckdb_tmp")) + "'"));return c
def lit(p):return "'"+str(p).replace('\\','/').replace("'","''")+"'"
def identities():
    import pandas as pd
    master=DATA/'master/ecg_master_index.parquet'
    f=readp(master,['subject_id','study_id','split'],test=True)
    if f[['subject_id','study_id','split']].isna().any().any():raise ValueError('Fehlende Identität')
    if not set(f.split)<={'train','validation','locked_test'}:raise ValueError('Unbekannte Partition')
    if f.study_id.duplicated().any():raise ValueError('Mehrdeutige study_id')
    dev=f.loc[f.split.isin(['train','validation'])];test=f.loc[f.split.eq('locked_test')]
    overlap=set(dev.subject_id)&set(test.subject_id)
    roles=f.groupby('subject_id').split.nunique();multi=int(roles.gt(1).sum())
    writep('identity_development_private.parquet',dev);writep('identity_test_private.parquet',test)
    writep('overlap_witness_private.parquet',f.loc[f.subject_id.isin(overlap)])
    attest={'passed':not overlap and multi==0,'scope':'vollständiger Master, train/validation-Union gegen sämtliche locked_test-Aufnahmen','development_ecgs':len(dev),'development_persons':dev.subject_id.nunique(),'test_ecgs':len(test),'test_persons':test.subject_id.nunique(),'overlap_persons':len(overlap),'multiple_role_persons':multi,'master_sha256':sha(master)}
    save('disjointness_attestation.json',attest)
    print(json.dumps(attest),flush=True)
    if not attest['passed']:raise ValueError('Globales Identitätsgate verletzt')
    cache=RUN/'analysis_feature_cache/cache-89d6aaf0a6b5957e/cache.manifest.json'
    cm=readj(cache,test=True);save('cache_manifest_bound.json',cm)
    for name,p in [('aw137_label',Path(str(data_path("a.aw137_label_semantics_json")))),('aw138_provenance',Path(str(data_path("a.aw137_provenance_json"))))]:
        save(name+'.json',readj(p,test=True))
    import pyarrow.parquet as pq
    schemas={}
    for name,p in [('test_cache',cache.parent/'split=test/features.parquet'),('index',DATA/'labels/sealed/locked_endpoint_index_cohorts.parquet')]:
        ledger('Beginn/Fortsetzung Footer und Hash',str(p)+'; keine Messwertprojektion')
        if not p.exists():
            schemas[name]={'path':str(p),'exists':False,'columns':[],'rows':None,'reason':'Quellenkandidat fehlt; Cachemanifest belegt ausschließlich Entwicklung'}
            continue
        pf=pq.ParquetFile(p);schemas[name]={**bind(p,pf.schema_arrow.names,rows=pf.metadata.num_rows),'schema':str(pf.schema_arrow)}
    save('schemas.json',schemas)
    save('release_python_manifest.json',[bind(p,role='release_code') for p in sorted(REL.rglob('*.py'))])
    save('initial_summary.json',{'disjointness':attest,'schemas':{k:{'rows':v['rows'],'columns':v['columns']} for k,v in schemas.items()},'cache_keys':list(cm)})

def ast_functions(path,names,env):
    import ast
    tree=ast.parse(Path(path).read_text(encoding='utf8'));nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    bind(path,role='executed_code');return env

def adapter():
    import numpy as np
    sys.path.insert(0,str(REL.parent))
    import vcg_biomarker.mimic.feature_record as fr
    from vcg_biomarker.mimic.feature_extraction import feature_implementation_hash
    provenance=readj(OUT/'aw138_provenance.json')
    assert feature_implementation_hash()==provenance['feature_implementation_hash_release']
    assert all(Path(m.__file__).resolve().is_relative_to(REL.resolve()) for n,m in sys.modules.items() if n.startswith('vcg_biomarker') and getattr(m,'__file__',None))
    env={'release_import':lambda:fr,'memory_guard':guard,'OPTIONS':BASE_OPTIONS,'dump':lambda p,o:save('adapter_calculation_failures_private.json',o)}
    ast_functions(PROJECT/'werkzeuge/varianten/adapter.py',['prepare_record'],env)
    ast_functions(PROJECT/'werkzeuge/transform_katalog/fortsetzung.py',['compute','options'],env)
    return fr,env

BASE_OPTIONS={'vcg_feature_ids':[],'classical_feature_ids':[],'sensitivity_feature_ids':[], 'sensitivity_vcg_methods':(),'powerline_hz':60.,'delineation_weights':{'neurokit':.4,'multilead':.35,'vcg':.25},'delineation_conflict_threshold_ms':20.}

def development_prepare():
    import pandas as pd,numpy as np,pyarrow.dataset as ds
    assert readj(OUT/'disjointness_attestation.json')['passed']
    catalog=pd.read_csv(PROJECT/'nachrechnung/transform_katalog.csv').marker_id.tolist();assert len(catalog)==2005 and len(set(catalog))==2005
    bind(PROJECT/'nachrechnung/transform_katalog.csv',role='fixed_catalog')
    save('catalog.json',catalog)
    context=Path(str(data_path("a.varianten_aw023c_stichprobe_kontext_parquet")))
    import pyarrow.parquet as pq
    cols=pq.ParquetFile(context).schema_arrow.names
    f=readp(context,cols).sort_values(['study_id'],kind='stable').reset_index(drop=True)
    seed=1592026
    ix=np.random.Generator(np.random.PCG64(seed)).choice(len(f),200,replace=False)
    tasks=f.iloc[ix].sort_values('study_id').reset_index(drop=True)
    assert tasks.split.isin(['train','validation']).all() and not tasks.study_id.duplicated().any()
    # Identisches A-Entwicklungskontextrezept; keine nachträgliche Reportmaskierung.
    writep('equivalence_tasks_private.parquet',tasks)
    index=readj(str(data_path("a.transform_katalog_fortsetzung_v3_referenz_index_json")))['index']
    references=[]
    for path in sorted({index[str(int(s))]['path'] for s in tasks.study_id}):
        wanted=tasks.loc[tasks.study_id.map(lambda s:index[str(int(s))]['path']==path),'study_id'].tolist()
        cols=['subject_id','study_id','split','extraction_status','signal_sha256','p_duration_ms','signal_quality_score',*catalog]
        sc=pq.ParquetFile(path).schema_arrow.names
        refs=readp(path,list(dict.fromkeys(k for k in cols if k in sc)),ds.field('study_id').isin(wanted))
        assert len(refs)==len(wanted)
        references.append(refs)
    expected=pd.concat(references,ignore_index=True)
    assert len(expected)==200 and not expected.study_id.duplicated().any()
    writep('equivalence_expected_private.parquet',expected)
    fr,env=adapter()
    save('extraction_binding.json',{'catalog_sha256':sha(OUT/'catalog.json'),'options':env['options'](catalog),'seed':seed,'n':200,'context_source':str(context),'context_sha256':sha(context),'selection':'PCG64, 200 ohne Zurücklegen aus A-Entwicklungskontext, sortiert study_id','tasks_sha256':sha(OUT/'equivalence_tasks_private.parquet'),'expected_sha256':sha(OUT/'equivalence_expected_private.parquet'),'release_implementation_hash':__import__('vcg_biomarker.mimic.feature_extraction',fromlist=['feature_implementation_hash']).feature_implementation_hash(),'adapter_definition_hashes':{str(p):sha(p) for p in [PROJECT/'werkzeuge/varianten/adapter.py',PROJECT/'werkzeuge/transform_katalog/fortsetzung.py']}})
    save('release_python_manifest.json',[bind(p,role='release_code') for p in sorted(REL.rglob('*.py'))])
    print(json.dumps({'phase':'development_prepare','n':200,'catalog':len(catalog)}),flush=True)

def equivalence():
    import numpy as np,pandas as pd
    tasks=readp(OUT/'equivalence_tasks_private.parquet',readj(OUT/'extraction_binding.json')['options'] and __import__('pyarrow.parquet',fromlist=['ParquetFile']).ParquetFile(OUT/'equivalence_tasks_private.parquet').schema_arrow.names)
    catalog=readj(OUT/'catalog.json');binding=readj(OUT/'extraction_binding.json')
    assert sha(OUT/'equivalence_tasks_private.parquet')==binding['tasks_sha256']
    expected=readp(OUT/'equivalence_expected_private.parquet',['subject_id','study_id','split','extraction_status',*catalog]).set_index('study_id')
    fr,env=adapter();options=env['options'](catalog)
    assert json.dumps(options,sort_keys=True)==json.dumps(binding['options'],sort_keys=True)
    counts={n:0 for n in catalog};status_mismatch=0;completed=0
    for row in tasks.to_dict('records'):
        guard();path=OUT/'equivalence_records_private'/f'{int(row["study_id"])}.json'
        if path.exists():
            saved=readj(path);assert saved['tasks_sha256']==binding['tasks_sha256'] and json.dumps(saved['options'],sort_keys=True)==json.dumps(options,sort_keys=True)
        else:
            # Release worker legt Status/QC fest; M verwendet exakt die AW067-Quellen.
            t=time.perf_counter();actual=fr._extract_record_worker((row,options))
            local=env['compute'](env['prepare_record'](row),row,catalog) if actual['extraction_status']=='ok' else actual
            errors=[];worker_errors=[];ref=expected.loc[row['study_id']]
            def equal(a,b):
                def missing(v):return v is None or pd.isna(v) or not np.isfinite(float(v))
                ma,mb=missing(a),missing(b)
                return ma==mb and (ma or np.isclose(float(a),float(b),rtol=1e-8,atol=0))
            for n in catalog:
                if not equal(local.get(n),ref.get(n)):errors.append(n)
                if not equal(local.get(n),actual.get(n)):worker_errors.append(n)
            clean=lambda obj:{k:(None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in obj.items()}
            saved={'tasks_sha256':binding['tasks_sha256'],'options':options,'result':clean(actual),'matrix_kors':clean(local),'mismatch_markers':errors,'worker_adapter_mismatch':worker_errors,'status_mismatch':actual['extraction_status']!=ref.extraction_status,'seconds':time.perf_counter()-t}
            save(str(path.relative_to(OUT)),saved)
        for n in saved['mismatch_markers']:counts[n]+=1
        status_mismatch+=saved['status_mismatch'];completed+=1
        save('equivalence_progress.json',{'completed':completed,'requested':200,'status_mismatches':status_mismatch,'markers_with_mismatch':sum(v>0 for v in counts.values()),'utc':utc()})
        print(json.dumps({'phase':'equivalence','completed':completed,'markers_with_mismatch':sum(v>0 for v in counts.values())}),flush=True);gc.collect()
    worker_counts={n:0 for n in catalog}
    for p in (OUT/'equivalence_records_private').glob('*.json'):
        r=readj(p)
        for n in r['worker_adapter_mismatch']:worker_counts[n]+=1
    save('equivalence_attestation.json',{'passed':completed==200 and not any(counts.values()) and status_mismatch==0 and not any(worker_counts.values()),'n':completed,'seed':1592026,'rtol':1e-8,'atol':0,'missingness_exact':True,'mismatch_counts':counts,'worker_adapter_mismatch_counts':worker_counts,'status_mismatches':status_mismatch,'binding_sha256':sha(OUT/'extraction_binding.json'),'runner_sha256':sha(__file__)})

META=['subject_id','study_id','split','ecg_time','age','sex','waveform_path','machine_filtering','machine_bandwidth','machine_metadata_usable_for_primary','machine_report','machine_qrs_axis_deg','machine_t_axis_deg','machine_qrs_duration_ms','machine_qt_interval_ms','machine_pr_interval_ms']
def selections():
    import pandas as pd,numpy as np,pyarrow.dataset as ds,pyarrow.parquet as pq
    disjoint=readj(OUT/'disjointness_attestation.json');assert disjoint['passed']
    c=conn();master=DATA/'master/ecg_master_index.parquet'
    assert sha(master)==disjoint['master_sha256']
    test_id=readp(OUT/'identity_test_private.parquet',['subject_id','study_id','split']).set_index('study_id')
    ledger('Beginn/Fortsetzung W-Paarbildung',str(master)+'; ausschließlich Identitäten/Zeit/Versorgungssetting; LEAD vor QC')
    pq.ParquetFile(master);bind(master,['subject_id','study_id','split','ecg_time','hadm_id','ed_stay_id'])
    sql=f"""WITH usable AS (SELECT subject_id,study_id,split,ecg_time,hadm_id,ed_stay_id FROM read_parquet({lit(master)}) WHERE split='locked_test' AND ecg_time IS NOT NULL),
    ordered AS (SELECT *,lead(study_id) OVER w next_study_id,lead(ecg_time) OVER w next_ecg_time FROM usable WINDOW w AS(PARTITION BY subject_id ORDER BY ecg_time,study_id))
    SELECT f.subject_id,f.split,f.study_id first_study_id,f.next_study_id second_study_id,f.ecg_time first_ecg_time,f.next_ecg_time second_ecg_time,date_diff('second',f.ecg_time,f.next_ecg_time)/3600.0 hours_between
    FROM ordered f JOIN ordered s ON s.subject_id=f.subject_id AND s.study_id=f.next_study_id WHERE f.study_id<>s.study_id AND date_diff('second',f.ecg_time,f.next_ecg_time)/3600.0 BETWEEN 0 AND 2 AND f.hadm_id IS NOT NULL AND f.hadm_id=s.hadm_id AND coalesce(f.ed_stay_id,-1)=coalesce(s.ed_stay_id,-1)"""
    save('w_candidate_recipe.json',{'sql':sql,'source_sha256':sha(REL/'mimic/trajectory.py')})
    pairs=c.execute(sql).fetchdf();writep('w_candidates_before_events_private.parquet',pairs)
    c.register('candidate',pairs)
    cm=readj(DATA/'standardized/clinical_events/clinical_events.manifest.json',test=True)
    registry=Path(next(x['path'] for x in cm['output_integrity'] if Path(x['path']).name=='clinical_event_registry.json'))
    assert sha(registry)==next(x['sha256'] for x in cm['output_integrity'] if Path(x['path'])==registry)
    reg=readj(registry,test=True);event_tables={x['table_id']:x for x in reg['tables'] if x['table_id'] in ('procedureevents','inputevents')}
    assert len(event_tables)==2 and all(t['status']=='ok' for t in event_tables.values())
    for name,item in event_tables.items():
        paths=[]
        for part in item['partitions']:
            p=registry.parent/part['relative_path']
            ledger('Beginn/Fortsetzung ICU-Quellbindung',str(p)+'; subject_id,starttime; W-Ausschluss strikt zwischen Zeitpunkten')
            assert sha(p)==part['sha256'];pq.ParquetFile(p);bind(p,['subject_id','starttime'],role='icu_event',rows=part['row_count']);paths.append(p)
        relation='read_parquet(['+','.join(lit(p) for p in paths)+'])'
        c.execute(f'CREATE VIEW {name} AS SELECT subject_id,starttime FROM {relation}')
    q="SELECT *,true intervening_intervention_checked FROM candidate c WHERE NOT EXISTS(SELECT 1 FROM procedureevents p WHERE p.subject_id=c.subject_id AND p.starttime>c.first_ecg_time AND p.starttime<c.second_ecg_time) AND NOT EXISTS(SELECT 1 FROM inputevents i WHERE i.subject_id=c.subject_id AND i.starttime>c.first_ecg_time AND i.starttime<c.second_ecg_time)"
    pairs=c.execute(q).fetchdf();writep('w_candidates_private.parquet',pairs)
    save('w_selection_attestation.json',{'candidate_pairs_before_events':len(c.execute('SELECT * FROM candidate').fetchdf()),'candidate_pairs':len(pairs),'persons':pairs.subject_id.nunique(),'ecgs':len(set(pairs.first_study_id)|set(pairs.second_study_id)),'intervening_intervention_checked':True,'registry_sha256':sha(registry),'source_recipe_sha256':sha(OUT/'w_candidate_recipe.json'),'historical_byte_attestation':'unvollständig; heutige immutable Registry vom 12.09.2026 gehasht, ursprünglicher Laufinputhash allein kein vollständiger historischer Runtimebeleg'})
    normal="machine_metadata_usable_for_primary AND age>=18 AND regexp_matches(lower(machine_report),'normal ecg|within normal limits') AND regexp_matches(lower(machine_report),'sinus rhythm') AND NOT regexp_matches(lower(machine_report),'abnormal|infarct|hypertroph|bundle|fascic|axis deviation|atrial fibrillation|atrial flutter|paced|pacemaker')"
    rq=f"WITH one AS (SELECT subject_id,study_id,split,ecg_time,row_number() OVER(PARTITION BY subject_id ORDER BY ecg_time,study_id) r FROM read_parquet({lit(master)}) WHERE split='locked_test' AND {normal}) SELECT * EXCLUDE(r) FROM one WHERE r=1 ORDER BY md5('AW047-R-v1:'||study_id::VARCHAR),study_id LIMIT 1500"
    ledger('Beginn/Fortsetzung P-Auswahl',str(master)+'; Report/Index/MD5 nach §4.4, keine P-Werte')
    psel=c.execute(rq).fetchdf();writep('p_selection_private.parquet',psel);save('p_selection_attestation.json',{'selected_persons':len(psel),'selected_ecgs':len(psel),'sql':rq,'cache_source':'kein Test-Entwicklungscache; Original-Holdoutschema wird vor jeder P-Wertprojektion geprüft','left_join_required':True})
    semantic=readj(OUT/'aw137_label.json')
    assert semantic['endpoints']['reduced_lvef_strict']['locked_test_semantics_attested']
    index=DATA/'labels/sealed/locked_endpoint_index_cohorts.parquet'
    integrity=next(v for v in semantic['source_integrity'] if Path(v['file'])==index)
    assert integrity['equal'] and sha(index)==integrity['sha256']
    idx=readp(index,['subject_id','study_id','split','label_id','label_state','eligibility_state','index_policy','index_selection_strategy','ecg_time'],(ds.field('label_id')=='reduced_lvef_strict') & (ds.field('eligibility_state')=='eligible') & ds.field('label_state').isin(['positive','negative']),test=True)
    assert idx.split.eq('locked_test').all() and not idx.subject_id.duplicated().any() and idx.index_selection_strategy.eq('first_verified_ecg_reference_result_blind').all()
    save('m_index_attestation.json',{'passed':True,'rows':len(idx),'index_sha256':sha(index),'semantic_attestation_sha256':sha(OUT/'aw137_label.json'),'strategy':'first_verified_ecg_reference_result_blind','index_policies':sorted(idx.index_policy.unique().tolist()),'label_source_integrity_verified':True,'development_b_index_not_reused':True})
    # Ausschließlich vorab erforderliche QRS/Rhythmus-Schichtungsfelder.
    hold=RUN/'publication_preparation/research-holdout-20261011-v2'
    protocol=readj(hold/'frozen/protocol.json',test=True);fm=readj(hold/'features.manifest.json',test=True)
    assert protocol['feature_code_hash']==readj(OUT/'extraction_binding.json')['release_implementation_hash']
    schema_groups={};sources=[];pieces=[];wanted=set(idx.study_id)
    for name,item in fm['feature_files'].items():
        p=hold/'features'/name;side=readj(p.with_suffix('.json'),test=True)
        ids=wanted&set(side['study_ids'])
        if not ids:continue
        assert sha(p)==item['sha256']==side['signature']['sha256'] and side['protocol_hash']==protocol['protocol_hash']
        sc=pq.ParquetFile(p).schema_arrow.names
        identities=readp(p,['subject_id','study_id','split'],test=True)
        assert identities.study_id.tolist()==side['study_ids'] and identities.split.eq('locked_test').all()
        assert not identities.study_id.duplicated().any() and set(identities.study_id)<=set(test_id.index)
        assert identities.set_index('study_id').subject_id.eq(test_id.loc[identities.study_id,'subject_id']).all()
        schema_groups[hashlib.sha256('|'.join(sc).encode()).hexdigest()]=sc
        cols=['subject_id','study_id','split','qrs_duration_ms','rhythm_context']
        if not set(cols)<=set(sc):
            save('m_strata_stop.json',{'reason':'Originale Testfeatures besitzen nicht alle vorgeschriebenen QRS/Rhythmus-Spalten','missing_columns':sorted(set(cols)-set(sc)),'source_path':str(p),'source_sha256':sha(p)})
            break
        pieces.append(readp(p,cols,ds.field('study_id').isin(sorted(ids)),test=True));sources.append(str(p))
    else:
        feat=pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame(columns=['subject_id','study_id','split','qrs_duration_ms','rhythm_context'])
        idx=idx.merge(feat,on=['subject_id','study_id','split'],how='left',validate='one_to_one')
        idx['qrs_stratum']=np.select([idx.qrs_duration_ms.lt(120),idx.qrs_duration_ms.ge(120)],['lt120','ge120'],default='unknown')
        idx['rhythm_stratum']=idx.rhythm_context.fillna('unknown').replace('','unknown')
        idx['half']=idx.subject_id.map(lambda s:int(hashlib.sha256(f'AW146-half-v1|{int(s)}'.encode()).hexdigest(),16)%2)
        idx['label']=idx.label_state.eq('positive').astype(int);chosen=[];quotas=[]
        for half in (0,1):
            for label,target in ((1,706),(0,2118)):
                pool=idx.loc[idx.half.eq(half)&idx.label.eq(label)];groups=list(pool.groupby(['qrs_stratum','rhythm_stratum'],sort=True));n=min(target,len(pool))
                exact=np.array([n*len(g)/len(pool) for _,g in groups]);quota=np.floor(exact).astype(int)
                for j in np.argsort(-(exact-quota),kind='stable')[:n-int(quota.sum())]:quota[j]+=1
                for (stratum,g),q in zip(groups,quota):
                    rank=g.subject_id.map(lambda s:hashlib.sha256(f'20261008|test_half_{half}|{label}|{int(s)}'.encode()).hexdigest())
                    chosen.extend(g.assign(_rank=rank).sort_values(['_rank','study_id'],kind='stable').head(int(q)).to_dict('records'))
                    quotas.append({'half':half,'label':label,'qrs':stratum[0],'rhythm':stratum[1],'pool':len(g),'selected':int(q)})
        msel=pd.DataFrame(chosen);writep('m_selection_private.parquet',msel)
        save('m_selection_attestation.json',{'selected_ecgs':len(msel),'persons':msel.subject_id.nunique(),'half_overlap_persons':len(set(msel.loc[msel.half.eq(0),'subject_id'])&set(msel.loc[msel.half.eq(1),'subject_id'])),'quotas':quotas,'missing_features_left_as_unknown':int(idx.qrs_duration_ms.isna().sum()),'selected_before_qc':True})
    save('holdout_schemas.json',schema_groups)
    for name,ids in [('w',sorted(set(pairs.first_study_id)|set(pairs.second_study_id))),('p',psel.study_id.tolist())]+([('m',msel.study_id.tolist())] if 'msel' in locals() else []):
        tasks=readp(master,META,ds.field('study_id').isin(ids),test=True)
        assert len(tasks)==len(ids) and tasks.split.eq('locked_test').all()
        usable=tasks.machine_metadata_usable_for_primary.eq(True).fillna(False)
        for k in ['machine_report','machine_qrs_axis_deg','machine_t_axis_deg','machine_qrs_duration_ms','machine_qt_interval_ms','machine_pr_interval_ms']:tasks[k]=tasks[k].where(usable,None)
        writep(name+'_tasks_private.parquet',tasks)
    c.close();print(json.dumps({'phase':'selections','w_pairs':len(pairs),'p_selected':len(psel),'m_selected':len(msel) if 'msel' in locals() else None}),flush=True)

def p_cache():
    import pandas as pd,pyarrow.dataset as ds,pyarrow.parquet as pq
    assert readj(OUT/'disjointness_attestation.json')['passed']
    eq=readj(OUT/'equivalence_attestation.json')
    assert eq['n']==200 and eq['mismatch_counts']['p_duration_ms']==0 and eq['status_mismatches']==0
    selection=readp(OUT/'p_selection_private.parquet',['subject_id','study_id','split','ecg_time'])
    allid=readp(OUT/'identity_test_private.parquet',['subject_id','study_id','split']).set_index('study_id')
    wanted=set(selection.study_id);hold=RUN/'publication_preparation/research-holdout-20261011-v2'
    fm=readj(hold/'features.manifest.json',test=True);protocol=readj(hold/'frozen/protocol.json',test=True)
    assert protocol['feature_code_hash']==readj(OUT/'extraction_binding.json')['release_implementation_hash']
    # Dieser Projektion werden keinerlei Verfügbarkeits-/Dauerstatistiken entnommen.
    values=[];parts=[]
    for name,item in fm['feature_files'].items():
        path=hold/'features'/name;side=readj(path.with_suffix('.json'),test=True)
        ids=wanted&set(side['study_ids'])
        if not ids:continue
        assert sha(path)==item['sha256']==side['signature']['sha256'] and side['protocol_hash']==protocol['protocol_hash']
        sc=pq.ParquetFile(path).schema_arrow.names
        identities=readp(path,['subject_id','study_id','split'],test=True)
        assert identities.study_id.tolist()==side['study_ids'] and identities.split.eq('locked_test').all() and not identities.study_id.duplicated().any()
        assert set(identities.study_id)<=set(allid.index) and identities.set_index('study_id').subject_id.eq(allid.loc[identities.study_id,'subject_id']).all()
        if 'p_duration_ms' not in sc:raise ValueError('Kein attestierter historischer P-Dauercache in Originalfragmenten')
        cols=['subject_id','study_id','split','p_duration_ms','extraction_status','signal_sha256']
        value=readp(path,cols,ds.field('study_id').isin(sorted(ids)),test=True)
        value['source_fragment']=str(path);value['source_sha256']=sha(path);value['cache_record_present']=True
        values.append(value);parts.append(str(path))
    f=pd.concat(values,ignore_index=True) if values else pd.DataFrame(columns=['subject_id','study_id','split','p_duration_ms','extraction_status','signal_sha256','source_fragment','source_sha256','cache_record_present'])
    joined=selection.merge(f,on=['subject_id','study_id','split'],how='left',validate='one_to_one')
    assert len(joined)==len(selection)
    joined['cache_record_present']=joined.cache_record_present.fillna(False)
    writep('p_test_cache_left_join_private.parquet',joined)
    save('p_cache_attestation.json',{'passed':True,'selected_rows_preserved':len(joined),'join':'LEFT JOIN subject_id,study_id,split; fehlende Records bleiben ausgewählt','p_duration_definition':'historische repräsentative Fusion aus Release; keine neue Detektion, kein Messwertgate','identity_hash_implementation_match':True,'release_implementation_hash':protocol['feature_code_hash'],'protocol_sha256':sha(hold/'frozen/protocol.json'),'feature_manifest_sha256':sha(hold/'features.manifest.json'),'equivalence_attestation_sha256':sha(OUT/'equivalence_attestation.json'),'selection_sha256':sha(OUT/'p_selection_private.parquet'),'output_sha256':sha(OUT/'p_test_cache_left_join_private.parquet'),'result_statistics_computed':False,'source_parts':parts,'producer_code_sha256':sha(PROJECT/'werkzeuge/b_test/testeingaben_aw137.py')})
    print(json.dumps({'phase':'p_cache','state':'complete','selected_rows':len(joined),'statistics_computed':False}),flush=True)

def equivalence_review():
    import pandas as pd,numpy as np
    eq=readj(OUT/'equivalence_attestation.json');assert eq['n']==200
    cols=['subject_id','study_id','split','signal_sha256','extraction_status']+list(n for n,v in eq['mismatch_counts'].items() if v)
    exp=readp(OUT/'equivalence_expected_private.parquet',cols).set_index('study_id')
    actual=[readj(p) for p in sorted((OUT/'equivalence_records_private').glob('*.json'))]
    hash_mismatch=0;details={n:{'mismatches':0,'missingness_mismatches':0,'value_mismatches':0,'max_absolute_difference':None} for n,v in eq['mismatch_counts'].items() if v}
    for r in actual:
        x=r['result'];e=exp.loc[x['study_id']]
        if not (x.get('signal_sha256')==e.signal_sha256 or pd.isna(x.get('signal_sha256')) and pd.isna(e.signal_sha256)):hash_mismatch+=1
        for n in r['mismatch_markers']:
            a=r['matrix_kors'].get(n);b=e[n];ma=a is None or pd.isna(a);mb=b is None or pd.isna(b);d=details[n];d['mismatches']+=1
            if ma!=mb:d['missingness_mismatches']+=1
            elif not ma:
                delta=abs(float(a)-float(b));d['value_mismatches']+=1;d['max_absolute_difference']=max(delta,d['max_absolute_difference'] or 0.)
    save('equivalence_review.json',{'n':len(actual),'signal_hash_mismatches':hash_mismatch,'marker_differences':details,'no_changed_selection_tolerance_or_release':True,'worker_adapter_mismatches':sum(eq['worker_adapter_mismatch_counts'].values()),'full_catalog_gate':eq['passed'],'W':'STOP_EQUIVALENCE' if not eq['passed'] else 'ready_for_extraction','M':'STOP_EQUIVALENCE' if not eq['passed'] else 'ready_for_extraction','P_field_equivalence':eq['mismatch_counts']['p_duration_ms']==0 and eq['status_mismatches']==0,'cause_proven':False,'historical_runtime_identity':'nicht vollständig attestiert; aktuelle Quellenidentität und Signalhashvergleich ersetzen keine historische numerische Äquivalenz'})
    print(json.dumps({'phase':'equivalence_review','full_catalog_passed':eq['passed'],'signal_hash_mismatches':hash_mismatch,'mismatch_markers':list(details),'worker_adapter_mismatches':sum(eq['worker_adapter_mismatch_counts'].values())}),flush=True)

def selftests():
    import pandas as pd
    c=conn()
    f=pd.DataFrame({'subject_id':[1,1,1,2,2,3,3],'study_id':[10,11,12,20,21,30,31],'split':['locked_test']*7,'ecg_time':pd.to_datetime(['2020-01-01 00:00','2020-01-01 01:00','2020-01-01 02:00','2020-01-01 00:00','2020-01-01 00:00','2020-01-01 00:00','2020-01-01 03:00']),'hadm_id':[4,4,4,5,5,6,6],'ed_stay_id':[None]*7,'quality':[1.,0.,1.,1.,1.,1.,1.]})
    c.register('synthetic',f)
    q=readj(OUT/'w_candidate_recipe.json')['sql'].replace('read_parquet('+lit(DATA/'master/ecg_master_index.parquet')+')','synthetic')
    r=c.execute(q).fetchdf();keys=set(zip(r.first_study_id,r.second_study_id))
    assert keys=={(10,11),(11,12),(20,21)},keys
    assert (10,12) not in keys
    assert r.loc[r.first_study_id.eq(20),'hours_between'].iloc[0]==0
    f.ed_stay_id=[None,None,None,None,99,None,None];c.unregister('synthetic');c.register('synthetic',f)
    r=c.execute(q).fetchdf();assert (20,21) not in set(zip(r.first_study_id,r.second_study_id))
    # Intervention exakt auf dem Rand darf nicht als strikt dazwischen gelten.
    between=c.execute("SELECT timestamp '2020-01-01 00:00'>timestamp '2020-01-01 00:00' AND timestamp '2020-01-01 00:00'<timestamp '2020-01-01 01:00',timestamp '2020-01-01 00:30'>timestamp '2020-01-01 00:00' AND timestamp '2020-01-01 00:30'<timestamp '2020-01-01 01:00'").fetchone();assert between==(False,True)
    c.close();save('selftests.json',{'passed':True,'checks':['Adjazenz vor QC, kein Überspringen fehlerhafter Zwischenaufnahme','inklusive Zeitgleichheit','fehlende ED-IDs auf beiden Seiten gleich','ungleiche ED-Settings ausschließen','Lücke >2h ausschließen','ICU-Beginn nur strikt zwischen EKGs ausschließen'],'clinical_or_test_statistics':False})

def p_release(stripe):
    import pyarrow.parquet as pq
    assert readj(OUT/'disjointness_attestation.json')['passed']
    eq=readj(OUT/'equivalence_attestation.json');assert eq['n']==200 and eq['mismatch_counts']['p_duration_ms']==0 and eq['status_mismatches']==0
    assert readj(OUT/'p_subset_equivalence.json')['passed']
    path=OUT/'p_tasks_private.parquet';tasks=readp(path,pq.ParquetFile(path).schema_arrow.names,test=True).sort_values('study_id').iloc[stripe::3]
    fr,env=adapter();options={**BASE_OPTIONS,'vcg_feature_ids':['p_duration_ms']}
    source_hash=sha(path);completed=0
    for row in tasks.to_dict('records'):
        guard();target=OUT/'p_release_records_private'/f'{int(row["study_id"])}.json'
        if target.exists():
            saved=readj(target,test=True);assert saved['tasks_sha256']==source_hash and json.dumps(saved['options'],sort_keys=True)==json.dumps(options,sort_keys=True)
        else:
            rawroot=Path(str(data_path("a.mimic_iv_ecg_diagnostic_electrocardiogram_matched_subset_1_0_files"))).resolve();raw=Path(row['waveform_path']).resolve()
            assert raw.is_relative_to(rawroot),'Rohpfad außerhalb files'
            ledger('Beginn/Fortsetzung P-Releaseextraktion',f'P-Taskhash `{source_hash}`, Stripe {stripe}; Header/Signal nur ausgewählte EKGs auf E:; keine P-Kennzahlen')
            fingerprints={}
            for p in [raw.with_suffix('.hea'),raw.with_suffix('.dat')]:
                fingerprints[str(p)]=bind(p,role='raw_waveform')
            result=fr._extract_record_worker((row,options))
            import numpy as np
            value=result.get('p_duration_ms');value=None if value is None or not np.isfinite(value) else float(value)
            saved={'tasks_sha256':source_hash,'options':options,'raw_fingerprints':fingerprints,'result':{'subject_id':int(row['subject_id']),'study_id':int(row['study_id']),'split':row['split'],'p_duration_ms':value,'extraction_status':result.get('extraction_status'),'signal_sha256':result.get('signal_sha256'),'cache_record_present':True},'producer_code_sha256':sha(REL/'mimic/feature_record.py'),'release_implementation_hash':readj(OUT/'extraction_binding.json')['release_implementation_hash']}
            save(str(target.relative_to(OUT)),saved)
        completed+=1;save(f'p_release_progress_{stripe}.json',{'completed':completed,'requested':len(tasks),'stripe':stripe,'utc':utc()})
        if completed%20==0:print(json.dumps({'phase':'p_release','stripe':stripe,'completed':completed}),flush=True)
        gc.collect()
def p_release0():p_release(0)
def p_release1():p_release(1)
def p_release2():p_release(2)
def p_subset_equivalence():
    import pandas as pd,numpy as np,pyarrow.parquet as pq
    save('p_source_decision.json',{'decision':'fresh_release_all_selected','authorization':'AW159 Aufgabe 3: P-Cachewerte bzw. Release-Extraktion nach Rezept','reason':'Kein primärer Testcache; Forschungs-Holdout v2 ist eine Auswahlteilmenge und wird für P nicht als vollständiger A-Cache ausgegeben. Identischer P-only-Extraktor wird zunächst am unveränderten 200er-Entwicklungssample geprüft.','selection_unchanged':True,'no_test_P_values_inspected_for_source_choice':True})
    path=OUT/'equivalence_tasks_private.parquet';tasks=readp(path,pq.ParquetFile(path).schema_arrow.names)
    expected=readp(OUT/'equivalence_expected_private.parquet',['subject_id','study_id','split','p_duration_ms','extraction_status','signal_sha256']).set_index('study_id')
    fr,env=adapter();options={**BASE_OPTIONS,'vcg_feature_ids':['p_duration_ms']};checks={'p_duration_ms':0,'extraction_status':0,'signal_sha256':0};done=0
    for row in tasks.to_dict('records'):
        guard();target=OUT/'p_subset_equivalence_private'/f'{int(row["study_id"])}.json'
        if target.exists():a=readj(target)['result']
        else:
            result=fr._extract_record_worker((row,options));v=result.get('p_duration_ms');v=None if v is None or not np.isfinite(v) else float(v)
            a={k:result.get(k) for k in ['subject_id','study_id','split','extraction_status','signal_sha256']};a['p_duration_ms']=v
            save(str(target.relative_to(OUT)),{'result':a,'options':options,'tasks_sha256':sha(path)})
        e=expected.loc[row['study_id']];x=a['p_duration_ms'];y=e.p_duration_ms
        mx=x is None or pd.isna(x);my=pd.isna(y)
        checks['p_duration_ms']+=int(mx!=my or not mx and not np.isclose(x,y,rtol=1e-8,atol=0))
        checks['extraction_status']+=int(a['extraction_status']!=e.extraction_status)
        checks['signal_sha256']+=int(not(a['signal_sha256']==e.signal_sha256 or pd.isna(a['signal_sha256']) and pd.isna(e.signal_sha256)))
        done+=1
        if done%20==0:print(json.dumps({'phase':'p_subset_equivalence','completed':done,'mismatches':checks}),flush=True)
    save('p_subset_equivalence.json',{'passed':done==200 and not any(checks.values()),'n':done,'seed':1592026,'rtol':1e-8,'atol':0,'missingness_exact':True,'options':options,'mismatches':checks,'tasks_sha256':sha(path),'scope':'P-only-Releaseanforderung gegen dieselbe eingefrorene Entwicklungsstichprobe; kein neuer Seed und keine neue Auswahl'})

def p_structure():
    import pyarrow.parquet as pq
    hold=RUN/'publication_preparation/research-holdout-20261011-v2'
    protocol=readj(hold/'frozen/protocol.json',test=True);fm=readj(hold/'features.manifest.json',test=True)
    save('holdout_protocol_bound.json',protocol)
    name=next(iter(fm['feature_files']));p=hold/'features'/name
    ledger('Beginn/Fortsetzung P-Quellenfooter',str(p)+'; nur Schemainformation, keine P-Werte')
    assert sha(p)==fm['feature_files'][name]['sha256']
    sc=pq.ParquetFile(p).schema_arrow.names;bind(p,sc,role='P_source_candidate_footer')
    save('p_source_structure.json',{'source_candidate':str(p),'p_duration_column_present':'p_duration_ms' in sc,'protocol_keys':list(protocol),'feature_code_hash':protocol['feature_code_hash'],'expected_release_hash':readj(OUT/'extraction_binding.json')['release_implementation_hash'],'settings_reuse_not_yet_attested':True,'no_values_read':True})
    print(json.dumps({'phase':'p_structure','p_duration_column_present':'p_duration_ms' in sc,'protocol_keys':list(protocol)}),flush=True)

def privacy_audit():
    import re
    s=(PROJECT/'koordination/TESTZUGRIFFE.md').read_text(encoding='utf8')
    pattern=r'\n\| [^|\r\n]+ \| AW-159 \|.*? \|(?=\r?\n|$)'
    blocks=re.findall(pattern,s,flags=re.S)
    assert all('value_set=' not in block and not re.search(r'\n\s*\d{8}\s*,?',block) for block in blocks)
    assert all('\n' not in block[1:] for block in blocks)
    save('privacy_audit.json',{'passed':True,'own_register_blocks_checked':len(blocks),'identifier_filter_repr_absent':True,'own_entries_single_line':True,'foreign_entries_scope':'nicht verändert, siehe privacy_repair_attestation'})

def post_selection():
    identities()
    selftests()
    privacy_audit()
    p_subset_equivalence()

def bericht(manifest_obj):
    import csv
    ledger('Beginn Bericht/Abgabe','Eigene Struktur-/Auswahl-/Extraktionsattestate und Entwicklungsvergleich; keine Testmesswertprojektion')
    data={name:readj(OUT/name) for name in ['manifest_hash.json','disjointness_attestation.json','equivalence_review.json','equivalence_attestation.json','w_selection_attestation.json','m_selection_attestation.json','p_selection_attestation.json','p_cache_attestation.json','p_subset_equivalence.json','privacy_repair_attestation.json','privacy_audit.json','selftests.json']}
    m=manifest_obj;mh=data['manifest_hash.json']['sha256'];d=data['disjointness_attestation.json'];e=data['equivalence_review.json'];w=data['w_selection_attestation.json'];ms=data['m_selection_attestation.json'];p=data['p_cache_attestation.json']
    source=lambda name: f'`<local path>{name}`'
    with (PROJECT/'nachrechnung/transform_katalog.csv').open(encoding='utf8',newline='') as f:cat={r['marker_id']:r for r in csv.DictReader(f)}
    lines=[]
    for n,v in e['marker_differences'].items():
        lines.append(f'| `{n}` | {v["value_mismatches"]} | {v["missingness_mismatches"]} | {cat[n]["admitted"]} |')
    half=[]
    for h in (0,1):
        for label in (1,0):
            q=[r for r in ms['quotas'] if r['half']==h and r['label']==label]
            half.append(f'| {h} | {"positiv" if label else "negativ"} | {sum(r["pool"] for r in q)} | {sum(r["selected"] for r in q)} | {706 if label else 2118} |')
    resource=[]
    for name in ['equivalence','selections','post_selection','p_release0','p_release1','p_release2']:
        path=OUT/f'resources_{name}.json'
        if path.exists():
            r=readj(path);resource.append(f'| `{name}` | {r["peak_rss_bytes"]} | {r["peak_private_commit_bytes"]} |')
    report=f'''# AW-159 - A-Testeingaben: Auswahlbindung, P-Extraktion und W/M-STOP

12.10.2026 · MS/PR · Lauf `A155-2026-10-04-d6e70e` · Vorschrift v1.0, SHA-256 `{SPEC_SHA}`. Keine Literaturrecherche und keine Replikationsergebnisrechnung.

**Der vollständige Auftrag ist wegen der nicht bestandenen Vollkatalogäquivalenz nicht erfüllt.** W/M-Auswahlbestände und Erzeugerbindungen sind erstellt; eine neue W/M-Vollkatalogextraktion wurde nach dem Entwicklungsvergleich nicht gestartet. P ist separat als technische Eingabe fertig. Das Manifest ist ausdrücklich **nicht für die vollständige A-Replikation bereit**. Diese Startbarriere ist kein biologischer Nullbefund und kein Test-Replikationsurteil.

Manifest: {source('a_test_input_manifest.json')}, SHA-256 `{mh}`. Es enthält Pfade, Bytehashes, Spalten, Zeilen, Erzeuger-/Releasebindung, Attestatverweise und elementweise Ausfälle. Personenkennungen und Messwerte liegen ausschließlich auf E:.

## 1. Vollständige Personen-Disjunktheit

| Rolle | EKGs | Personen |
|---|---:|---:|
| train/validation-Union | {d['development_ecgs']} | {d['development_persons']} |
| vollständiges locked_test | {d['test_ecgs']} | {d['test_persons']} |

Personenüberschneidung Test/Entwicklung: {d['overlap_persons']}; Personen mit mehreren Partitionsrollen: {d['multiple_role_persons']}. study_id war eindeutig, Identitäten und Rollen waren vollständig. Grundlage ist der **gesamte** Master, nicht nur die angereicherte A-Entwicklungsstichprobe. Master: `<local path>`, SHA-256 `{d['master_sha256']}`; Projektion `subject_id,study_id,split`. Beleg: {source('disjointness_attestation.json')}, erzeugt durch `identities`/`post_selection` in `werkzeuge/a_testreplikation/aw159_inputs.py`. Identitätsprojektionen und leerer Overlap-Witness nur auf E:.

## 2. Entwicklungsäquivalenz und Konsequenz

Feste Stichprobe: {data['equivalence_attestation.json']['n']} EKGs, PCG64-Seed `{data['equivalence_attestation.json']['seed']}`, ohne Zurücklegen aus dem gehashten A-Entwicklungskontext `<local path>`, sortiert nach study_id. Auswahl vor neuen Testmesswerten. Referenzen ausschließlich aus den über den AW067-Referenzindex gebundenen train/validation-Originalfragmenten; keine Testreferenzwerte. Details und Quellhashes: {source('extraction_binding.json')} und {source('sources_development_prepare.json')}.

Verglichen wurden der Release-Worker und die unveränderten reinen AW067-Funktionen `prepare_record`, `compute`, `options`; Übernahme mittels AST ohne Importnebenwirkungen der Entwicklungsskripte. Toleranz rtol=1e−8, atol=0; fehlende/nichtendliche Werte werden nicht mit endlichen Werten gleichgesetzt. Alle Katalog-IDs bleiben im Vergleich.

| Entwicklungsmarker | Wertabweichungen | Missingness-Abweichungen | bereits in AW067 zugelassen (`admitted`) |
|---|---:|---:|---|
{chr(10).join(lines)}

Signalhash-Abweichungen: {e['signal_hash_mismatches']}; Status-Abweichungen: {data['equivalence_attestation.json']['status_mismatches']}; Abweichungen Release-Worker gegen AW067-Adapter: {e['worker_adapter_mismatches']}. Die Ursache der historischen Wert-/Missingness-Abweichungen ist **nicht bewiesen**. Die Release-Quellen, Auswahl, Toleranz und Messparameter wurden nicht geändert. Keine Fälle oder Marker wurden nachträglich entfernt. Die frühere Nichtzulassung eines Markers ersetzt nicht die hier verlangte Vollkatalogäquivalenz.

Belege: {source('equivalence_attestation.json')}, {source('equivalence_review.json')}; Zeilenschlüssel jeweils `marker_differences`/`mismatch_counts × marker_id`, Erzeuger `equivalence` und `equivalence_review`. Ausgeführter Äquivalenzsnapshot: `6cb3ec4aa19019bbec698dceccaad135e8ea81558f46174a44406950048e7e5e`. Die im ursprünglichen Attestat gespeicherte Runner-Datei wurde während der laufenden Entwicklung weiter ergänzt; dieser Live-Dateihash allein beschreibt nicht den geladenen Ausführungsstand. Maßgeblich sind der erhaltene Snapshot und die gesonderte Ausführungsbindung im Manifest. Keine Messlogik des geladenen Vergleichs wurde verändert.

**W und M: STOP_EQUIVALENCE vor neuer Vollkatalog-Testextraktion.** Keine W-Testmarkerwerte, keine neuen Kors/Dower-Vollkatalogwerte, keine ICC, RC, LoA, Klassen oder Markerwertverteilungen erzeugt. Vorhandene Test-QRS-/Rhythmusfelder wurden ausschließlich für die vorgeschriebene M-Schichtung benutzt.

## 3. W-Auswahl vor Messwerten/QC

Identitäts-/Zeit-/Setting-SQL reproduziert LEAD vor QC, inklusive 0-2 Stunden, gleicher nichtfehlender hadm_id und gleiche COALESCE(ed_stay_id,−1). Beide ICU-Eventtabellen sind in der unveränderlichen Registry vom 12.09.2026 vorhanden; alle verwendeten Partitionen stimmen mit deren SHA-256 überein. Ausschluss nur für Beginn strikt zwischen den EKG-Zeitpunkten, ohne zusätzliche hadm_id-Bedingung.

Kandidaten vor ICU-Filter: {w['candidate_pairs_before_events']}; danach: {w['candidate_pairs']} Paare bei {w['persons']} Personen und {w['ecgs']} beteiligten EKGs. **Diese Zahlen sind vor Signalhash-Deduplikation, Extraktions-/QC-Gates und Proxybildung**, keine auswertbaren W-Paarnenner. Die nachgelagerten Gates wurden wegen STOP nicht ausgeführt.

Beleg: {source('w_selection_attestation.json')}, Schlüssel `candidate_pairs_before_events,candidate_pairs,persons,ecgs`; SQL in {source('w_candidate_recipe.json')}; Dateien `w_candidates_private.parquet` und `w_tasks_private.parquet`. Erzeuger: `selections`. Registry-SHA `{w['registry_sha256']}`. Die heute geprüfte, bereits vor dem Lauf angelegte immutable Registry attestiert nicht vollständig den damaligen Lauf-/Binär-/Inputstand; diese Grenze bleibt offen. Keine vollständige historische Interventionsfreiheitsbehauptung.

## 4. M-Auswahl und technische Hälften

Native reduced_lvef_strict-Indizes wurden gegen den aktuellen gehashten Originalindex und das AW137-Semantikattest gebunden; ausschließlich eligible positive/negative und `first_verified_ecg_reference_result_blind`, ohne Ergebnisrichtung als Indexkriterium. Kein B-Index mit technischer Erfolgsauswahl übernommen. Originalfragmente aus Forschungs-Holdout v2: individuelle Identitäten gegen Test-Master, Fragment-/Protokollhashes und Release-Implementierungsgleichheit geprüft; nur benötigte QRS-/Rhythmusfelder projiziert. Fehlende Features bleiben im festen unknown-Stratum, kein Ersatz durch Geräte-QRS oder empirische Aliaswahl.

Die Personenhälften, Hashränge, lexikografischen Gleichstände und größte-Rest-Zuteilung entsprechen §4.2. Auswahl vor QC; keinerlei Ersatzfälle. Insgesamt {ms['selected_ecgs']} EKGs von {ms['persons']} Personen; Überschneidung der Hälften: {ms['half_overlap_persons']} Personen.

| technische Testhälfte | Referenzlabel | Pool | ausgewählt | Ziel |
|---|---|---:|---:|---:|
{chr(10).join(half)}

Abweichungen vom Zielumfang entstehen durch `min(Ziel,Pool)`; keine Aufstockung zwischen Labels/Hälften. Beide Hälften bleiben Test, keine neue Entwicklung/Bestätigung. Belege: {source('m_index_attestation.json')}, {source('m_selection_attestation.json')}, Schlüssel `quotas × half,label`, Erzeuger `selections`; Auswahl/Tasks nur auf E:. Same-Fiducial-Kors/Dower-Extraktion ist wegen Äquivalenz-STOP **nicht ausgeführt**.

## 5. P-Auswahl, gezielte Äquivalenz und technische Eingabe

Report-normal-Rezept wörtlich nach P1/§4.4, erstes qualifizierendes EKG/Person nach Zeit/study_id, MD5-Rang `AW047-R-v1:`, höchstens 1.500 Personen. Ausgewählt: {data['p_selection_attestation.json']['selected_persons']} Personen/EKGs. Kein Signal-QC-/P-Verfügbarkeitsfilter vor Auswahl, keine Ersatzperson, keine klinische Gesundenaussage.

Der genannte primäre Testcache existiert nicht: dessen Cachemanifest bindet ausschließlich train/validation und `locked_test_included=false`. Der Forschungs-Holdout v2 ist eine Auswahlteilmenge und wurde für P nicht als vollständiger A-Cache benutzt. Vor Sicht neuer Test-P-Werte wurde deshalb gemäß AW159 Aufgabe 3 die Release-Neuextraktion **aller ausgewählten** P-EKGs festgelegt; kein Verfügbarkeits-gesteuertes Auffüllen eines Teilcaches.

Gezielter P-only-Releasevergleich auf derselben {data['p_subset_equivalence.json']['n']}-EKG-Entwicklungsstichprobe: bestanden, bei rtol=1e−8/atol=0 keine Abweichung von P-Dauer, Extraktionsstatus oder Signalhash. Belege: {source('p_source_decision.json')} und {source('p_subset_equivalence.json')}, Erzeuger `p_subset_equivalence`. Parameter: 500-Hz-Release-Loader, 60-Hz-Netzfilter, Fusion 0,4/0,35/0,25, Konflikt 20 ms; historische repräsentative Multilead-Fusionsdauer, keine union12-/II-/V2-Neudefinition.

Technische Extraktion abgeschlossen und alle {p['selected_rows_preserved']} Auswahlzeilen per LEFT JOIN auf subject_id/study_id/split erhalten. Extraktionsabstentionen werden nicht ersetzt. Jeder Record hat gehashte Rohheader-/Datendateien, Task-/Optionenbindung und Release-Erzeuger. Eingabe: {source('p_test_cache_left_join_private.parquet')}, SHA-256 `{p['output_sha256']}`; Attest: {source('p_cache_attestation.json')}, Erzeuger `p_release0/1/2` und `p_release_finalize`.

**Keine P-Verfügbarkeitsquote, kein Dauerquantil/-stratum und keine Verteilung berechnet oder angesehen.** Das Artefakt ist ein AW159-Release-Neucache, kein vorbestehender historischer Testcache. Eine spätere Rechnung muss diese Herkunft benennen und die W/M-Startbarrieren sowie weitere noch offene Replikationsvoraussetzungen beachten. P-Bereitschaft allein ist kein Gesamtreplikationsclaim.

## 6. Ausführungsbindung, Ressourcen und Kontrolle

Überinklusive Release-Pythonquellenliste und Digest sowie Paketversionen im Manifest; Release-Implementierungshash `{m['producer_binding']['release_implementation_hash']}`. Kein vollständiges historisches Binärruntimeattest. Explizite Code-Snapshots je Phase unter `executed_source/`, Start-/Endprotokoll `execution_history.jsonl`; Ausgabegeneratoren im Manifest an den tatsächlich geladenen Snapshot gebunden. Frühe Registereinträge verwenden teils den jeweils aktuellen Live-Dateihash; Ausführungsidentität wird deshalb über die Snapshots/Phasenhistorie nachgewiesen.

Vor jedem Pythonstart nativer RAM-/Commit-Check ≥1,5 GiB; während der Phasen entsprechende Guards und eigener RSS-/Commit-Wächter. Numerische Threads=1; DuckDB memory_limit=1GB, threads=2. Während P höchstens drei freigegebene Worker, je eigener Status/Checkpoint und eigener Budgetwächter. Keine Installation; kein Fremdprozess beendet; kein Analyse-/Release-/GUI-/Simulationscode geändert. Ein eigener Auswahlworker (PID 71744) wurde für die unten beschriebene Registerkorrektur beendet und aus unveränderter Auswahlvorschrift neu gestartet.

| Phase | RSS-Spitze Bytes | private Commit-Spitze Bytes |
|---|---:|---:|
{chr(10).join(resource)}

Belege: `resources_<Phase>.json`, Erzeuger `Budget`/Launcher; keine Ressourcenwerte aus dem Gedächtnis. Synthetische Prüfungen bestanden: Adjazenz vor QC, kein Überspringen der Zwischenaufnahme, Zeitgleichheit, NULL-/ungleiche ED-Settings, 2-h-Grenze, strikt zwischenliegende ICU-Zeit. Beleg {source('selftests.json')}; keine Teststatistik als Testfixture verwendet.

### Registerkorrektur / Abweichung vom Append-only-Verfahren

Die erste str(pyarrow-filter)-Darstellung schrieb Studienkennungen in eigene Registereinträge. Das war ein Fehler. Der eigene Worker wurde sofort beendet; ausschließlich die eigenen AW159-Filtertexte wurden redigiert ({data['privacy_repair_attestation.json']['own_filter_entries_redacted']} Filtereinträge vereinheitlicht), fremde Registerbytes blieben nachweislich unverändert. Diese notwendige Kennungsentfernung war eine Abweichung von „nur Zeilen anfügen“; sie wird nicht als regelkonformes reines Anhängen ausgegeben. Keine Datei wurde gelöscht. Danach bleiben konkrete Filter/Identitäten ausschließlich im privaten E:-Log. Prüfung: {source('privacy_repair_attestation.json')} mit Vor-/Nachhash und {source('privacy_audit.json')}; eigene Einträge ohne Kennungslisten und einzeilig. Keine Rohsignale oder individuellen P-Werte im Projekt.

## 7. Prüfkette F1-F8 und offene Fragen

| Falle | Stellungnahme |
|---|---|
| F1 | nicht_zutreffend für diese technische Eingangsbindung; kein Marker-Endpunkt-Claim |
| F2 | eingeschraenkt: angereichertes M und Report-normal-P; keinerlei Prävalenz-/PPV-/Gesundheitsclaim |
| F3 | eingeschraenkt: Regeln/Seed/Toleranz fest; keine Anpassung anhand Testergebnissen |
| F4 | eingeschraenkt: vollständiger fester Katalog und alle Entwicklungsabweichungen dokumentiert; kein Herausnehmen zum Bestehen |
| F5 | offen: Testmesswert-Missingness wurde nicht ausgewertet; P-LEFT-JOIN erhält Auswahlzeilen, W-QC/Markernenner fehlen |
| F6 | eingeschraenkt: Kors/Dower technische Rekonstruktionen, keine Frank-/manuelle klinische Testreferenz |
| F7 | eingeschraenkt: vollständige Personen-Disjunktheit bestanden; interne MIMIC-Testpartition, keine externe Validierung |
| F8 | offen: W-Duplikat-/QC-/Proxyprüfung und M-Neuextraktion nicht ausgeführt; keine Qualitäts-/Geräte- oder biologische Interpretation |

K muss die historischen Entwicklungsabweichungen und ihre Konsequenz für die unveränderte Vollkatalogvorschrift klären. Keine stillschweigende Toleranzerhöhung, Fallentfernung oder neue Zulassungsentscheidung. Ebenso bleibt die vollständige historische ICU-/Runtimeattestation begrenzt. AW158 hat weitere Voraussetzungen einer späteren Replikationsrechnung dokumentiert; dieses Manifest beseitigt sie nicht. Keine neuen Manuskript-/Autor-/Ethikplatzhalter, keine Publikationsfreigabe. Quellenfehler-Kandidaten: keine externe Quelle geprüft; keine Literaturfehlermeldung.

Sicherheit: hoch für aktuelle Datei-/Identitätsbindung, Auswahlzahlen, tatsächlich geprüfte Entwicklungsabweichungen und P-Eingangs-Vollständigkeit; begrenzt für historische numerische Runtimeidentität und nicht ausgeführte W/M-Extraktion. Die angeforderte komplette W/M/P-Testextraktion bleibt **unvollständig**.
'''
    (PROJECT/'nachrechnung/A_TESTEINGABEN_AW159.md').write_text(report,encoding='utf8')
    submission=f'''# Abgabe AW-159

## Ergebnis

**Teilabgabe mit technischem STOP, keine vollständige Auftragserfüllung.** Wertfreies Testinputmanifest und alle W/M/P-Auswahlbestände erstellt. Vollständige Test/Entwicklungs-Personen-Disjunktheit bestanden. Neue W/M-Vollkatalogextraktion nach fehlgeschlagener Entwicklungsäquivalenz nicht gestartet. P-Releaseeingabe für alle {p['selected_rows_preserved']} ausgewählten EKGs separat fertig und per LEFT JOIN erhalten. Keine Ergebnisrechnung.

Manifest: `<local path>`, SHA-256 `{mh}`; `ready_for_complete_A_replication=false`. Vorschrift SHA-256 `{SPEC_SHA}`. Vollbericht: `nachrechnung/A_TESTEINGABEN_AW159.md`.

## Neue/geänderte Dateien

- Neu: `werkzeuge/a_testreplikation/aw159_inputs.py`, `aw159_start.ps1`, `nachrechnung/A_TESTEINGABEN_AW159.md`, diese Abgabe.
- Neu auf E:: `<local path>` mit Identitäts-/Auswahl-/Taskdateien, Entwicklungsvergleichen, Rohquellhashes, P-Recordcheckpoints und LEFT-JOIN-Eingabe, Attestaten, Code-Snapshots, Status-/Ressourcenprotokollen und Manifest. Genaue Eingabedateiliste/Hashes/Spalten/Zeilen im Manifest; Einzelwerte ausschließlich dort.
- `koordination/TESTZUGRIFFE.md`: Zugriffe/Fortsetzungen/Ende ergänzt; eigene fehlerhafte Filtertexte mit Studienkennungen zusätzlich korrigiert. **Append-only-Abweichung ausdrücklich offengelegt**, keine Fremdzeile geändert, keine Datei gelöscht. Belege `privacy_repair_attestation.json`, `privacy_audit.json` auf E:.

## Belege und Befehle

Launcher: `& werkzeuge\\a_testreplikation\\aw159_start.ps1 <Phase>`; eigener Python `<local path> -B -m werkzeuge.a_testreplikation.aw159_inputs <Phase>`. Tatsächliche Phasen: identities, development_prepare, equivalence, equivalence_review, selections, privacy_repair, post_selection (Identitätsnachbindung, synthetische Prüfungen, P-only-Äquivalenz), p_release0/1/2, Abschluss. Gate vor jedem Start ≥1,5 GiB freier RAM/Commit; drei Worker nur während P; eigener RSS/Commit je <1,5 GiB. Einzig aktiv beendet: eigene Auswahl-PID 71744. Keine Installation, Außenkommunikation oder Änderung am Hauptlauf/Release/gemeinsamen Werkzeugen.

| Befund | Zahl/Status | Quelle/Schlüssel/Erzeuger (alle <local path>) |
|---|---|---|
| volle Entwicklung | {d['development_ecgs']} EKGs, {d['development_persons']} Personen | disjointness_attestation.json; development_*; identities/post_selection; gesamter Master des A155-Laufs |
| volles Test | {d['test_ecgs']} EKGs, {d['test_persons']} Personen | disjointness_attestation.json; test_*; gleicher Erzeuger |
| Personenüberlappung | {d['overlap_persons']} | disjointness_attestation.json; overlap_persons; vollständige train/validation-Union |
| W vor QC/Hash-Deduplikation | {w['candidate_pairs_before_events']} vor ICU, {w['candidate_pairs']} danach | w_selection_attestation.json; candidate_pairs*; selections; §4.3-SQL |
| M vor QC | {ms['selected_ecgs']} Personen/EKGs, Hälften disjunkt | m_selection_attestation.json; selected_*,half_overlap_persons; quotas; selections |
| P Auswahl/Eingabe | {p['selected_rows_preserved']} Zeilen erhalten | p_selection_attestation.json,p_cache_attestation.json; selected_*; p_release*/finalize |
| Vollkatalogäquivalenz | {data['equivalence_attestation.json']['n']} EKGs, Seed {data['equivalence_attestation.json']['seed']}; nicht bestanden | equivalence_attestation.json,equivalence_review.json; marker_differences/mismatch_counts; equivalence* |
| gezielte P-Äquivalenz | {data['p_subset_equivalence.json']['n']} EKGs, bestanden | p_subset_equivalence.json; mismatches; p_subset_equivalence; unveränderte Stichprobe/Toleranz |

Abweichungen: `beatwise_p_loop_axis_stability_deg` mit {e['marker_differences']['beatwise_p_loop_axis_stability_deg']['value_mismatches']} Wertabweichungen, `t_loop_to_qrs_loop_complexity_ratio` mit {e['marker_differences']['t_loop_to_qrs_loop_complexity_ratio']['missingness_mismatches']} Missingness-Abweichungen. Signalhash-/Status- und Worker/Adaptervergleich ohne Abweichung. Ursache unbewiesen; kein Parameter-/Toleranz-/Auswahlwechsel zur Erfüllung. W/M deshalb STOP; kein Null-/Replikationsurteil. Die frühe reine Live-Runner-Dateihashbindung wird durch erhaltene Ausführungssnapshots und Phasenhistorie präzisiert; maßgeblicher Vollkatalogvergleich-Snapshot `6cb3ec4aa19019bbec698dceccaad135e8ea81558f46174a44406950048e7e5e`.

## Offene Fragen / nächste Entscheidung K

1. Wie sind die historischen Entwicklungsabweichungen unter dem unveränderten Vollkatalogvertrag zu klären? Ohne geklärte Äquivalenz keine W/M-Neuextraktion aus dieser Abgabe starten.
2. Historische ICU-/Binärruntime-/Inputidentität ist nicht vollständig attestiert; heutige immutable Registry-/Codehashes ersetzen das nicht.
3. P-Datei als AW159-Release-Neucache kennzeichnen; sie ist kein vorbestehender historischer Testcache. Keine P-Verfügbarkeits-/Dauerkennzahlen erhoben. Weitere AW158-Replikationsvoraussetzungen bleiben gesondert zu prüfen.

## Nebenbefunde

Der in §4.1 genannte Testcache ist nicht vorhanden; sein Manifest erklärt locked_test_included=false und bindet nur Entwicklung. M-Zielumfang wegen festen min(Ziel,Pool)-Regeln unterschritten; Details je Hälfte/Label im Vollbericht, keine Aufstockung. Eigenes Registerleck korrigiert und geprüft; ausführliches Korrekturprotokoll im Vollbericht. Keine Literaturrecherche, keine neuen Manuskriptplatzhalter, keine Quellenfehler-Kandidaten.

## Sicherheit

Hoch für aktuelle Hash-/Identitäts-/Auswahlbefunde, dokumentierten Entwicklungsvergleich, P-only-Äquivalenz und P-Eingangsvollständigkeit. Begrenzt für historische Erzeugeridentität und nicht ausgeführte W/M-Extraktion. F1-F8 mit Grenzen im Vollbericht. **Keine vollständige A-Testinputfreigabe und keine vollständige Erledigung der angeforderten W/M/P-Extraktion.**
'''
    (PROJECT/'koordination/abgaben/AW-159.md').write_text(submission,encoding='utf8')
    save('delivery_hashes.json',{str(p):sha(p) for p in [PROJECT/'nachrechnung/A_TESTEINGABEN_AW159.md',PROJECT/'koordination/abgaben/AW-159.md']})
    ledger('Ende Bericht/Abgabe',f'Manifest `{mh}`; W/M STOP, P-Eingabe fertig; Abgabe AW-159 erstellt, keine Ergebnisrechnung')

def finalisieren():
    for phase in ['p_release0','p_release1','p_release2']:assert readj(OUT/f'status_{phase}.json')['state']=='complete'
    p_release_finalize()
    privacy_audit()
    manifest_obj=manifest()
    bericht(manifest_obj)
def p_release_finalize():
    import pandas as pd
    selection=readp(OUT/'p_selection_private.parquet',['subject_id','study_id','split','ecg_time'],test=True)
    ledger('Beginn/Fortsetzung P-Releasekonsolidierung','eigene P-Recorddateien auf E:; ausschließlich Zusammenführung, keine Verfügbarkeits-/Dauerstatistik')
    results=[readj(p)['result'] for p in (OUT/'p_release_records_private').glob('*.json')]
    f=pd.DataFrame(results);assert len(f)==len(selection) and set(f.study_id)==set(selection.study_id) and not f.study_id.duplicated().any()
    joined=selection.merge(f,on=['subject_id','study_id','split'],how='left',validate='one_to_one');assert len(joined)==len(selection)
    writep('p_test_cache_left_join_private.parquet',joined)
    save('p_cache_attestation.json',{'passed':True,'selected_rows_preserved':len(joined),'source_kind':'AW159-Release-Neuextraktion aller ausgewählten P-EKGs; kein historischer Testcache verfügbar','join':'LEFT JOIN subject_id,study_id,split; Auswahl bleibt auch bei Extraktionsabstention bestehen','p_duration_definition':'repräsentative Multilead-Fusion; A-Einstellungen; keine neue Messdefinition','selection_sha256':sha(OUT/'p_selection_private.parquet'),'tasks_sha256':sha(OUT/'p_tasks_private.parquet'),'output_sha256':sha(OUT/'p_test_cache_left_join_private.parquet'),'equivalence_attestation_sha256':sha(OUT/'equivalence_attestation.json'),'P_only_equivalence_sha256':sha(OUT/'p_subset_equivalence.json'),'extraction_status_counts':{str(k):int(v) for k,v in f.extraction_status.value_counts(dropna=False).items()},'result_statistics_computed':False,'release_implementation_hash':readj(OUT/'extraction_binding.json')['release_implementation_hash']})

def manifest():
    import pyarrow.parquet as pq,importlib.metadata
    eq=readj(OUT/'equivalence_attestation.json');disjoint=readj(OUT/'disjointness_attestation.json');review=readj(OUT/'equivalence_review.json')
    assert disjoint['passed']
    extraction=readj(OUT/'extraction_binding.json');inputs=[]
    history=[json.loads(line) for line in (OUT/'execution_history.jsonl').read_text(encoding='utf8').splitlines() if line.strip()]
    phase_hashes={r['phase']:r['executed_source_sha256'] for r in history if r['event']=='start'}
    ledger('Beginn/Fortsetzung Manifestbindung','eigene Testeingabe-Parquets: nur Footer, Zeilen/Spalten und Bytehash; keine Messwertlesung')
    for name,role in [('identity_test_private.parquet','test_identity'),('w_candidates_private.parquet','W_candidates_before_QC'),('w_tasks_private.parquet','W_extraction_tasks'),('m_selection_private.parquet','M_selection_before_QC'),('m_tasks_private.parquet','M_extraction_tasks'),('p_selection_private.parquet','P_selection_before_measurement'),('p_tasks_private.parquet','P_extraction_tasks'),('p_test_cache_left_join_private.parquet','P_release_or_cache_values')]:
        p=OUT/name
        if not p.exists():continue
        pf=pq.ParquetFile(p)
        phase='post_selection' if role=='test_identity' else 'finalisieren' if role=='P_release_or_cache_values' else 'selections'
        executed_hash=phase_hashes[phase];snapshot=OUT/'executed_source'/executed_hash/Path(__file__).name
        assert sha(snapshot)==executed_hash
        settings=OUT/('p_subset_equivalence.json' if role=='P_release_or_cache_values' else 'extraction_binding.json')
        inputs.append({'path':str(p),'sha256':sha(p),'columns':pf.schema_arrow.names,'rows':pf.metadata.num_rows,'role':role,'producer':{'phase':phase,'code_path':str(snapshot),'code_sha256':executed_hash,'release_implementation_hash':extraction['release_implementation_hash'],'settings_binding_path':str(settings),'settings_binding_sha256':sha(settings)}})
    sources={}
    for p in sorted(OUT.glob('sources_*.json')):
        for item in readj(p):
            key=item['path'];prior=sources.get(key,{})
            if prior and prior['sha256']!=item['sha256']:raise ValueError('Quellhash widersprüchlich')
            item['columns']=list(dict.fromkeys(prior.get('columns',[])+item.get('columns',[])));sources[key]=item
    release=readj(OUT/'release_python_manifest.json')
    release_tree=hashlib.sha256(json.dumps([(x['path'],x['sha256']) for x in release],sort_keys=True).encode()).hexdigest()
    attestations={}
    for name in ['disjointness_attestation.json','equivalence_attestation.json','equivalence_review.json','w_selection_attestation.json','m_index_attestation.json','m_selection_attestation.json','m_strata_stop.json','p_selection_attestation.json','p_cache_attestation.json','p_subset_equivalence.json','p_source_decision.json','selftests.json','privacy_repair_attestation.json','privacy_audit.json']:
        p=OUT/name
        if p.exists():attestations[name]={'path':str(p),'sha256':sha(p)}
    p_passed=(OUT/'p_cache_attestation.json').exists() and readj(OUT/'p_cache_attestation.json')['passed']
    packages={n:importlib.metadata.version(n) for n in ['numpy','pandas','pyarrow','duckdb','scipy','wfdb','neurokit2']}
    failures=[]
    if not eq['passed']:failures.append({'elements':['W','M'],'status':'STOP_EQUIVALENCE','reason':'200er-Entwicklungsvergleich: numerische Vollkatalogäquivalenz bei rtol 1e-8 nicht erfüllt','affected_markers':[k for k,v in eq['mismatch_counts'].items() if v],'no_new_W_M_full_catalog_extraction_performed':True})
    if (OUT/'m_strata_stop.json').exists():failures.append({'elements':['M'],'status':'STOP_STRATA_SCHEMA','details':readj(OUT/'m_strata_stop.json')})
    if not p_passed:failures.append({'elements':['P'],'status':'NO_COMPLETE_INPUT'})
    manifest={'manifest_version':'AW159-v1','created_utc':utc(),'run':'A155-2026-10-04-d6e70e','specification_sha256':SPEC_SHA,'authorization':'AW-159, wertfreie Eingangsbindung und technische Extraktion; keinerlei Replikationsergebnisrechnung','ready_for_complete_A_replication':eq['passed'] and p_passed and not failures,'element_status':{'W':'STOP_EQUIVALENCE' if not eq['passed'] else 'not_extracted','M':'STOP_EQUIVALENCE' if not eq['passed'] else 'not_extracted','P':'input_ready' if p_passed else 'incomplete'},'test_input_files':inputs,'sources':list(sources.values()),'producer_binding':{'release_path':str(REL),'release_implementation_hash':extraction['release_implementation_hash'],'release_python_tree_manifest_sha256':sha(OUT/'release_python_manifest.json'),'release_python_tree_digest':release_tree,'packages':packages,'W_M_settings':extraction['options'],'P_settings':readj(OUT/'p_subset_equivalence.json')['options'],'executed_phase_sha256':phase_hashes,'current_runner_sha256':sha(__file__),'equivalence_executed_snapshot_sha256':'6cb3ec4aa19019bbec698dceccaad135e8ea81558f46174a44406950048e7e5e','code_snapshot_directory':str(OUT/'executed_source'),'no_full_historical_binary_runtime_attestation':True},'equivalence':{'n':eq['n'],'seed':eq['seed'],'rtol':eq['rtol'],'atol':eq['atol'],'full_catalog_passed':eq['passed'],'P_field_passed':review['P_field_equivalence'],'signal_hash_mismatches':review['signal_hash_mismatches']},'disjointness':disjoint,'attestations':attestations,'failures':failures,'result_statistics_computed':False,'forbidden_outputs_absent':['ICC','RC','LoA','matrix_classes','P_availability_statistics','P_duration_statistics','test_marker_distributions']}
    save('a_test_input_manifest.json',manifest)
    mhash=sha(OUT/'a_test_input_manifest.json')
    ledger('Ende AW159-Testeingangsbindung',f'Manifest `{mhash}`; W/M STOP bei Vollkatalogäquivalenz; P-Eingangsstatus {manifest["element_status"]["P"]}; keine Ergebnisrechnung')
    save('manifest_hash.json',{'sha256':mhash,'path':str(OUT/'a_test_input_manifest.json')})
    print(json.dumps({'phase':'manifest','sha256':mhash,'element_status':manifest['element_status'],'ready':manifest['ready_for_complete_A_replication']}),flush=True)
    return manifest

def privacy_repair():
    import re
    p=PROJECT/'koordination/TESTZUGRIFFE.md'
    raw=p.read_bytes();text=raw.decode('utf8')
    # Nur eigene AW159-Blöcke; sämtliche Fremdbytes bleiben identisch.
    pattern=r'\n\| [^|\r\n]+ \| AW-159 \|.*? \|(?=\r?\n|$)'
    changed=0
    def redact(match):
        nonlocal changed
        block=match.group(0)
        if '; Filter ' not in block:return block
        changed+=1
        return block.split('; Filter ',1)[0]+'; Filter private Auswahl auf E:; keine Kennungen im Register |'
    corrected=re.sub(pattern,redact,text,flags=re.S)
    # Fremde Einträge inkl. ihrer Zeilenumbrüche nachweislich unverändert.
    def foreign(s):return re.sub(pattern,'',s,flags=re.S)
    assert foreign(text)==foreign(corrected),'Fremde Einträge würden verändert'
    p.write_bytes(corrected.encode('utf8'))
    save('privacy_repair_attestation.json',{'own_filter_entries_redacted':changed,'foreign_entries_unchanged':True,'reason':'Eigene str(pyarrow-filter)-Darstellung enthielt study_ids; Identitäten ausschließlich E:','ledger_before_sha256':hashlib.sha256(raw).hexdigest(),'ledger_after_sha256':sha(p),'no_file_deleted':True,'own_interrupted_pid':71744})
    ledger('Korrektur eigener Filterdarstellung und Unterbrechung','Eigene PID 71744 beendet; Kennungen in eigenen zuvor angefügten Filterzeilen redigiert, Fremdeinträge unverändert. Fortsetzung mit privatem Filterlog auf E:.')

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True);phase=sys.argv[1];assert sha(SPEC)==SPEC_SHA,'Freeze verändert'
    launch_sha=sha(__file__)
    snapshot=OUT/'executed_source'/launch_sha/Path(__file__).name;snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(Path(__file__).read_bytes())
    with (OUT/'execution_history.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps({'phase':phase,'event':'start','pid':os.getpid(),'utc':utc(),'executed_source_sha256':launch_sha})+'\n')
    save('status_'+phase+'.json',{'phase':phase,'state':'running','pid':os.getpid(),'start':utc(),'memory':guard(),'runner_sha256':sha(__file__)})
    try:
        from werkzeuge.nachrechnung.resources import Budget
        with Budget(OUT) as budget:globals()[phase]()
        save('resources_'+phase+'.json',budget.summary())
        save('status_'+phase+'.json',{'phase':phase,'state':'complete','pid':os.getpid(),'end':utc(),'memory':guard(),'executed_source_sha256':launch_sha})
    except Exception as e:
        save('status_'+phase+'.json',{'phase':phase,'state':'failed','pid':os.getpid(),'end':utc(),'error_type':type(e).__name__,'error':str(e)})
        traceback.print_exc();sys.exit(1)
    finally:
        save('sources_'+phase+'.json',list(SOURCES.values()))
        with (OUT/'execution_history.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps({'phase':phase,'event':'end','pid':os.getpid(),'utc':utc(),'executed_source_sha256':launch_sha})+'\n')
        if phase not in ('development_prepare','equivalence','equivalence_review','p_subset_equivalence','selftests','privacy_repair'):ledger('Ende Phase '+phase,'Status/Quelleninventar auf <local path>; ausgeführter Code '+launch_sha)
