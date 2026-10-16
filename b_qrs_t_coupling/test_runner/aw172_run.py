"""AW172: Vertrag zuerst; neue private Ausgaben ausschließlich im eigenen E:-Ordner."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
from repo_config import reference as reference_value  # 'reference' is a local variable name in run()
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
import sys, json, hashlib, time, ctypes, traceback
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
OUT=Path(str(data_path("b.aw172")))
OUT.mkdir(exist_ok=True)
sys.stdout=(OUT/'stdout.log').open('a',encoding='utf8',buffering=1)
sys.stderr=(OUT/'stderr.log').open('a',encoding='utf8',buffering=1)
PHASE=sys.argv[1]
PEAK={'rss':0,'private':0}
SPEC=ROOT/'nachrechnung/SPEZIFIKATION_B_TEST_EXPLORATIV_v1.md'
ADD=ROOT/'nachrechnung/SPEZIFIKATION_B_TEST_EXPLORATIV_v1_NACHTRAG1.md'
MAN=Path(str(data_path("b.b_freigabe_input_manifest_v5_json")))
F='qrs_t_normalized_discrete_frechet'
M='qrs_t_frechet_beat_median'
A='spatial_qrs_t_angle_deg'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()

def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def load(p):
    p=Path(p)
    if p.stat().st_size>20*1024**2:raise RuntimeError('JSON_20MB')
    return json.loads(p.read_text(encoding='utf8'))
def clean(v):
    if isinstance(v,dict):return {str(k):clean(a) for k,a in v.items()}
    if isinstance(v,(list,tuple)):return [clean(a) for a in v]
    if hasattr(v,'tolist'):return clean(v.tolist())
    if isinstance(v,float) and not __import__('math').isfinite(v):return None
    return v
def dump(name,v):
    target=OUT/name
    if not target.resolve().is_relative_to(OUT.resolve()):raise RuntimeError('Schreibziel')
    target.write_text(json.dumps(clean(v),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def status(state,**kw):dump('status.json',dict(phase=PHASE,state=state,pid=os.getpid(),utc=now(),**kw))
def now():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def memory():
    class Mem(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total','free','page','commit','vtotal','vfree','extended')]
    m=Mem();m.length=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise RuntimeError('Speichermessung')
    return dict(ram_free=m.free,commit_free=m.commit)
def guard():
    while min(memory().values())<1.5*1024**3:
        status('waiting_memory',memory=memory());time.sleep(30)
    import psutil
    info=psutil.Process().memory_info()
    PEAK['rss']=max(PEAK['rss'],info.rss);PEAK['private']=max(PEAK['private'],info.private)
    if max(PEAK.values())>1.5*1024**3:raise RuntimeError('Prozessspeicher_1_5GiB')

def prepare():
    expected={str(SPEC):'80cfb334c28a8c8834b39148ac7466281f3da8496ecf012ed493679d14437b2e',
              str(ADD):'ccb216c004f7ae8e4678059cad35c1e513f06480955e47f8c98c44a5873fe31f',
              str(data_path("b.aw171_parameters_json")):'17784e0a21811d2f61e9def653d7b1a5d9dfbb8cd1bf2088b508da0aa0938b72',
              str(data_path("b.aw171_mappings_json")):'e525b3493dd9b443272d6382b95cd3b402cead3f4beaa15699b24a1d4953646b',
              str(MAN):'5bfa03ae40fb812f6d037b618c8e7b5a76512b656a5bf3ab1f02265a8223b49e'}
    code=load(str(data_path("b.aw171_code_manifest_json")))
    if canonical(code['files'])!='543812c494f037dc8b185caef06dd34c25be08201b52827424f238c7f4e06686':raise RuntimeError('Gesamtcodebindung')
    expected.update({str(ROOT/p):h for p,h in code['files'].items()})
    checks=[]
    for p,h in expected.items():
        actual=sha(p);checks.append(dict(path=p,expected=h,actual=actual,match=actual==h))
    dump('preflight.json',dict(checks=checks,utc=now(),test_content_read=False))
    if not all(r['match'] for r in checks):raise RuntimeError('Hashabweichung_STOP_Gesamt')
    original=load(ROOT/'werkzeuge/b_test/parameter_entwicklung.json')
    import pyarrow.parquet as pq
    schemas={}
    for key in ('verification_train','technical_development'):
        item=original[key];guard()
        if sha(item['path'])!=item['sha256']:raise RuntimeError('Hashabweichung_Entwicklung')
        pf=pq.ParquetFile(item['path'])
        schemas[key]=dict(path=item['path'],sha256=item['sha256'],columns=pf.schema_arrow.names,rows=pf.metadata.num_rows)
    # Der vorhandene erste-technische Entwicklungsvertrag wird nicht durch C_e ersetzt.
    dump('development_schemas.json',schemas)
    import numpy as np
    import pyarrow.dataset as ds
    from werkzeuge.b_test import aw171_kernels as k
    from werkzeuge.b_test.statistik import raw_design,fixed_design
    cols=['subject_id','study_id','split','age','sex_numeric','heart_rate','qrs_duration_ms',
          'qtc_fridericia_ms',A,F,'classical_cornell_product','classical_sokolow_lyon',
          'magnitude_area_qrs','spatial_low_voltage_qrs_peak_magnitude','classical_right_axis_deviation','has_tte_30d']
    f=ds.dataset(schemas['technical_development']['path'],format='parquet').to_table(columns=cols,filter=ds.field('split')=='train').to_pandas()
    if f.subject_id.duplicated().any() or not f.split.eq('train').all():raise RuntimeError('W19_Trainuniversum')
    train_union=np.sort(f.subject_id.unique())
    raw,names=raw_design(f,F,strong=True)
    good=np.isfinite(raw).all(1)
    verified=f.has_tte_30d.eq(True).to_numpy(float)
    scale=original['variants']['verification_marker']
    if names!=scale['names']:raise RuntimeError('W19_Designvertrag')
    x=fixed_design(raw[good],scale);v=verified[good]
    b=k.solver(x,v)
    from scipy.special import expit
    p=expit(x@b)
    model=dict(beta=b.tolist(),scale=scale,p95=float(np.quantile(1/p[v==1],.95,method='linear')),
               p99=float(np.quantile(1/p[v==1],.99,method='linear')),n_universe=len(f),n_complete=int(good.sum()),
               n_verified=int(v.sum()),missing_marker=int(f[F].isna().sum()),fit_population='train first technical adult; B4+H1 marker complete',
               source=schemas['technical_development'],filter='split=train; no reference/marker filter at index selection')
    dump('w19_development.json',model)
    f.to_parquet(OUT/'w19_train_universe_private.parquet',index=False)
    dump('w19_train_union_private.json',dict(dtype=str(train_union.dtype),sorted='np.sort unique numeric',sha256=hashlib.sha256(train_union.tobytes()).hexdigest(),n=len(train_union)))
    # Keine scheinbare semantische Freigabe durch Vergleich zweier Regex-Parser.
    master=Path(str(data_path("b.mimic_validation_master_ecg_master_index_parquet")))
    guard();pf=pq.ParquetFile(master)
    report_columns=['subject_id','study_id','split','machine_report','machine_metadata_usable_for_primary']
    parser=dict(status='STOP_W24_semantische_100er_Pruefung_nicht_vorliegend',required_sample=100,seed=1722026,tolerance_errors=5,errors=None)
    if set(report_columns)<=set(pf.schema_arrow.names):
        reports=ds.dataset(str(master),format='parquet').to_table(columns=report_columns,
             filter=(ds.field('split').isin(['train','validation']))&ds.field('study_id').isin(f.study_id.tolist())).to_pandas()
        if not set(reports.split)<= {'train','validation'}:raise RuntimeError('Parser_Entwicklungsbarriere')
        from werkzeuge.b_test.aw171_mapping import pacing
        reports['pacing_aw171']=[pacing(text,bool(usable)) for text,usable in zip(reports.machine_report,reports.machine_metadata_usable_for_primary)]
        # Nachtrag-K-Regel explizit; keine Umdeutung dieses Regexvergleichs als
        # unabhängige menschliche Lesartprüfung.
        text=reports.machine_report.fillna('').str.lower()
        reports['pacing_K_positive']=text.str.contains('pace',regex=False)&~text.str.contains('no pacemaker',regex=False)&reports.machine_metadata_usable_for_primary.eq(True)
        sample=reports.sample(n=100,random_state=1722026)
        sample.to_parquet(OUT/'parser_sample_development_private.parquet',index=False)
        parser.update(source=str(master),source_sha256=sha(master),filter='train/validation and first-technical train study set',
                      n_reports=len(reports),category_counts=reports.pacing_aw171.value_counts().to_dict(),sample_n=len(sample),
                      rule_disagreements_n=int((sample.pacing_aw171.eq('positive')!=sample.pacing_K_positive).sum()),
                      reason='100er-Stichprobe mit Zuordnung auf E: gespeichert. Unabhaengige semantische Lesartpruefung fehlt; Regexvergleich ersetzt sie nicht.')
    else:parser['reason']='Maschinenbefundspalten im Entwicklungs-Master nicht verfuegbar.'
    dump('parser_development.json',parser)
    mapping=load(str(data_path("b.aw171_mappings_json")))
    mapping['w13_restricted_codes']={
        '9':['428*','40201','40211','40291','40401','40403','40411','40413','40491','40493','4148','422*','4293','6745*','2773*'],
        '10':['I50*','I110','I130','I132','I255','I40*','I41*','I517','O903','E85*','B572']}
    mapping['w13_time_rule']='admission up to and including index encounter; not merely index encounter'
    dump('mappings.json',mapping)
    manifest=load(MAN)
    inputs=[]
    # Ausschließlich explizit attestierte physische Projektionen; keine neuen Testquellen.
    for key in ('BEF-0063','BEF-0049'):
        entry=manifest['hypotheses'][key]
        inputs.append(dict(role=key,path=entry['path'],sha256=entry['sha256']))
    inputs.append(dict(role='technical',path=manifest['technical']['path'],sha256=manifest['technical']['sha256']))
    for entry in inputs:
        guard()
        if sha(entry['path'])!=entry['sha256']:raise RuntimeError('Hashabweichung_STOP_Gesamt')
    dump('input_bindings.json',dict(inputs=inputs,manifest_sha256=sha(MAN),partition='previously_opened_internal',sealed_excluded=True))
    print('PREPARE_COMPLETE W19_train_n='+str(model['n_complete']),flush=True)

def optimism():
    import numpy as np,pyarrow.parquet as pq,pyarrow.dataset as ds
    from werkzeuge.b_test import aw171_kernels as k
    from werkzeuge.b_test.statistik import raw_design,fixed_design
    from werkzeuge.nachrechnung.core import auc
    params=load(str(data_path("b.aw171_parameters_json")))
    contracts=load(str(data_path("b.aw171_input_contracts_json")))['development']
    for ep,bef,marker in [('H1','BEF-0063',F),('H4','BEF-0049',M)]:
        path=Path(str(data_path("b.aw168")))/(bef+'_private.parquet');guard();pf=pq.ParquetFile(path)
        if sha(path)!=contracts[str(path)]['sha256']:raise RuntimeError('Optimismus_Hashabweichung')
        cols=['subject_id','split','age','sex_numeric','heart_rate','qrs_duration_ms','qtc_fridericia_ms',A,marker,'label_state']
        cols+=['eligible_reference','preferred_lvef'] if ep=='H1' else ['eligibility_state','index_selection_strategy']
        f=ds.dataset(str(path),format='parquet').to_table(columns=cols,filter=ds.field('split').isin(['train','validation'])).to_pandas()
        ref=f.eligible_reference.eq(True) if ep=='H1' else f.age.ge(18)&f.label_state.isin(['positive','negative'])&f.eligibility_state.eq('eligible')&f.index_selection_strategy.eq('first_verified_ecg_reference_result_blind')
        f=f.loc[ref].copy();raw,_=raw_design(f,marker);good=np.isfinite(raw).all(1);f=f.loc[good]
        pack=params['endpoints'][ep]['markers'][marker]
        if len(f)!=pack['n'] or f.subject_id.duplicated().any():raise RuntimeError('Optimismus_Entwicklungsfallset')
        z=fixed_design(raw[good],pack['scale']);y=f.preferred_lvef.lt(40).to_numpy(float) if ep=='H1' else f.label_state.eq('positive').to_numpy(float)
        state=load(OUT/(ep+'_optimism_checkpoint.json')) if (OUT/(ep+'_optimism_checkpoint.json')).exists() else dict(next=0,rows=[],errors={})
        for r in range(state['next'],200):
            if r%25==0:guard();status('development_optimism',endpoint=ep,replicate=r)
            rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([1692026,20,r])))
            w=np.bincount(rng.integers(0,len(y),len(y)),minlength=len(y)).astype(float)
            try:
                row={}
                for model,size in [('B2',7),('B3',8),('B3_marker',9)]:
                    b=k.solver(z[:,:size],y,w);score=z[:,:size]@b
                    row[model]=auc(y,score,w)-auc(y,score)
                state['rows'].append(row)
            except Exception as ex:
                reason=type(ex).__name__+':'+str(ex);state['errors'][reason]=state['errors'].get(reason,0)+1
            state['next']=r+1
            if (r+1)%25==0:dump(ep+'_optimism_checkpoint.json',state)
        meanopt={model:float(np.mean([r[model] for r in state['rows']])) for model in pack['models']} if len(state['rows'])>=190 else None
        apparent={model:auc(y,z[:,:size]@np.asarray(pack['models'][model]['beta'])) for model,size in [('B2',7),('B3',8),('B3_marker',9)]}
        dump(ep+'_development_optimism.json',dict(endpoint=ep,planned=200,attempted=200,valid=len(state['rows']),minimum=190,
             n=len(y),cases=int(y.sum()),source=str(path),source_hash=sha(path),filter='train+validation C_e; original fixed scaling',
             seed='SeedSequence([1692026,20,r])',mean_optimism=meanopt,apparent_auc=apparent,
             optimism_corrected_auc={model:apparent[model]-meanopt[model] for model in apparent} if meanopt else None,failures=state['errors']))

def numeric(v):
    import numpy as np
    return {k:float(x) for k,x in v.items() if isinstance(x,(int,float,np.integer,np.floating)) and np.isfinite(x)}

def boot(key,fn,positions,union_n,planned,minimum):
    import numpy as np
    target=OUT/(key+'_bootstrap.json')
    state=load(target) if target.exists() else dict(rows=[],errors={},next=0,planned=planned,minimum=minimum)
    if state['planned']!=planned or state['minimum']!=minimum:raise RuntimeError('Bootstrap_Checkpointvertrag')
    for r in range(state['next'],planned):
        if r%25==0:guard();status('running_bootstrap',analysis_id=key,replicate=r,planned=planned)
        rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([1282026,r])))
        w=np.bincount(rng.integers(0,union_n,union_n),minlength=union_n)[positions].astype(float)
        try:state['rows'].append(dict(replicate=r,metrics=numeric(fn(w,r))))
        except Exception as ex:
            # Keine Personenwerte in Fehlern/Projektdateien; Detail ausschließlich E:.
            reason=type(ex).__name__+':'+str(ex)
            state['errors'][reason]=state['errors'].get(reason,0)+1
        state['next']=r+1
        if (r+1)%25==0 or r+1==planned:dump(target.name,state)
    metrics=set().union(*(r['metrics'] for r in state['rows'])) if state['rows'] else set()
    summary={}
    for metric in sorted(metrics):
        values=[r['metrics'][metric] for r in state['rows'] if metric in r['metrics']]
        summary[metric]=dict(valid=len(values),ci=np.quantile(values,[.025,.975],method='linear').tolist() if len(values)>=minimum else None)
    return dict(planned=planned,attempted=planned,minimum=minimum,metrics=summary,failures=state['errors'])

def analyze(key,W,ep,marker,pop,fn,positions=None,planned=0,minimum=950):
    target=OUT/(key+'.json')
    if target.exists():return load(target)
    status('running_analysis',analysis_id=key)
    record=dict(analysis_id=key,W=W,endpoint=ep,marker=marker,population=pop,status='schaetzbar',point={},bootstrap=None)
    try:
        guard();record['point']=clean(fn(None,-1))
        if planned:
            record['bootstrap']=boot(key,fn,positions,UNION_N,planned,minimum)
    except Exception as ex:
        record.update(status='STOP',reason=type(ex).__name__+':'+str(ex))
    dump(target.name,record);return record

def stop_record(key,W,ep,marker,reason):
    r=dict(analysis_id=key,W=W,endpoint=ep,marker=marker,population='nicht_erreichbar',status='STOP',reason=reason,point={},bootstrap=None)
    dump(key+'.json',r);return r

def association_rows(ep,df,marker,scale,key,W,selector=None,planned=500):
    import numpy as np
    from werkzeuge.b_test.statistik import raw_design,fixed_design
    from werkzeuge.b_test import aw171_kernels as k
    chosen=df if selector is None else df.loc[selector]
    raw,_=raw_design(chosen,marker);z=fixed_design(raw,scale);y=chosen.y.to_numpy(float)
    positions=np.searchsorted(UNION,chosen.subject_id.to_numpy())
    return analyze(key,W,ep,marker,'C_'+ep,lambda w,r:k.association(z,y,w),positions,planned)

def run():
    global UNION,UNION_N
    import numpy as np,pandas as pd,pyarrow.parquet as pq,pyarrow.dataset as ds
    from werkzeuge.b_test.daten import select_index,masks,echo_ids
    from werkzeuge.b_test.statistik import raw_design,fixed_design
    from werkzeuge.b_test import aw171_kernels as k
    binding=load(OUT/'bindung_vor_test.json')
    for p,h in binding['files'].items():
        if sha(p)!=h:raise RuntimeError('Hashabweichung_STOP_Gesamt')
    if (OUT/'access_finished.json').exists():raise RuntimeError('Einziger_Testzugriff_bereits_beendet')
    manifest=load(MAN);params=load(str(data_path("b.aw171_parameters_json")))
    original=load(ROOT/'werkzeuge/b_test/parameter_entwicklung.json')
    dev=load(OUT/'development_schemas.json')
    devpersons=set()
    for item in dev.values():
        guard();pf=pq.ParquetFile(item['path'])
        t=ds.dataset(item['path'],format='parquet').to_table(columns=['subject_id','split'],filter=ds.field('split').isin(['train','validation']))
        devpersons.update(t.column('subject_id').to_pylist())
    # Gegen sämtliche train/validation-Personen prüfen, nicht nur vollständige
    # Regressions-/Referenzsets. Physischer Master ist ausschließlich mit
    # Entwicklungsfilter freigegeben und vorab gehasht.
    master=Path(str(data_path("b.mimic_validation_master_ecg_master_index_parquet")))
    guard();pf=pq.ParquetFile(master)
    mt=ds.dataset(str(master),format='parquet').to_table(columns=['subject_id','split'],filter=ds.field('split').isin(['train','validation']))
    devpersons.update(mt.column('subject_id').to_pylist())
    dump('development_disjoint_universe.json',dict(n=len(devpersons),source=str(master),source_hash=sha(master),filter='split IN (train,validation); all persons, no eligibility/CC filter'))
    ledger=ROOT/'koordination/TESTZUGRIFFE.md'
    if not (OUT/'access_started.json').exists():
        line=f'\n| {now()} | AW-172 | Beginn gebündelter explorativer B-Zugriff | Bindung `{sha(OUT/"bindung_vor_test.json")}`; Spezifikation `{sha(SPEC)}`; Nachtrag `{sha(ADD)}` | §§3-13; vorbelastete interne Partition; gesperrte Testmenge ausgeschlossen |\n'
        with ledger.open('a',encoding='utf8') as f:f.write(line)
        dump('access_started.json',dict(utc=now(),binding_sha256=sha(OUT/'bindung_vor_test.json')))
    frames={};sources={};index_frames={};Ds={};Cs={};results=[]
    def read_entry(entry,cols):
        guard()
        if sha(entry['path'])!=entry['sha256']:raise RuntimeError('Hashabweichung_STOP_Gesamt')
        pf=pq.ParquetFile(entry['path'])
        if not set(cols)<=set(pf.schema_arrow.names):raise RuntimeError('Schema_STOP_Gesamt')
        f=ds.dataset(entry['path'],format='parquet').to_table(columns=cols).to_pandas()
        if set(f.split)!={'test'}:raise RuntimeError('Partitionsvertrag_STOP_Gesamt')
        if devpersons.intersection(f.subject_id.unique()):raise RuntimeError('Disjunktheit_STOP_Gesamt')
        sources[entry['path']]=dict(sha256=entry['sha256'],columns=cols,filter='attested test projection only',rows=len(f))
        return f
    # Leseadapter und Kernsets; Quellenvertrag vor Modellen abgleichen.
    try:
        common=['subject_id','study_id','split','ecg_time','age','sex','heart_rate','machine_heart_rate_bpm',
                'machine_metadata_usable_for_primary','qrs_duration_ms','qtc_fridericia_ms',A,F,M,
                'label_state','eligibility_state','index_selection_strategy','reference_available',
                'ehr_time_alignment_confirmatory_eligible','extraction_status','preferred_lvef',
                'lvef_conflict_gt_10','nearest_tte_measurement_id','nearest_tte_delta_hours','time_delta_hours',
                'time_precision','evidence_ids_json','has_tte_30d','preferred_lvef_method','sinus_selector',
                'classical_lbbb_pattern_positive','classical_rbbb_pattern_positive']
        expected={'H1':(9374,8371,800),'H4':(reference_value("aw172_run.h4_index_n"),18170,433)}
        # H4 Indexzahl wird aus dem eingefrorenen historischen Aggregat gebunden,
        # nicht aus Ergebnissicht angepasst.
        for ep,bef,marker,endpoint in [('H1','BEF-0063',F,'reduced_lvef_strict'),('H4','BEF-0049',M,'cardiomyopathy_icd')]:
            entry=manifest['hypotheses'][bef]
            f=read_entry(entry,common)
            frames[ep]=f
            idx=select_index(f,endpoint)
            index_frames[ep]=idx
            z,good,flow=masks(idx,endpoint,marker,'qrs_duration_ms')
            reference=z.age.ge(18)&z.label_state.isin(['positive','negative'])&z.eligibility_state.eq('eligible')
            if ep=='H1':
                reference &= z.time_precision.eq('timestamp')&z.time_delta_hours.abs().le(720)&np.isclose(z.time_delta_hours,z.nearest_tte_delta_hours,atol=1e-9,rtol=0)&z.apply(echo_ids,axis=1)&z.preferred_lvef.between(0,100)&z.lvef_conflict_gt_10.eq(False)
            z['y']=(z.preferred_lvef<40).astype(float) if ep=='H1' else z.label_state.map({'positive':1.,'negative':0.})
            D=z.loc[reference].copy();C=z.loc[good].copy()
            Ds[ep]=D;Cs[ep]=C
            # Gebundene historische Indexzahl aus Manifest, historische C_e aus AW142.
            check_n=entry['index_coverage']['proven_B_index_persons']
            matched=len(idx)==check_n and len(C)==expected[ep][1] and int(C.y.sum())==expected[ep][2]
            results.append(analyze(ep+'_index_match','W25',ep,marker,'index',lambda w,r,idx=idx,C=C,D=D,check_n=check_n,matched=matched:dict(index_n=len(idx),expected_index_n=check_n,reference_n=len(D),complete_n=len(C),cases=int(C.y.sum()),matched=matched)))
            if not matched:
                stop_record(ep+'_all_models','GLOBAL',ep,marker,'Referenz_Index_Abgleich; betroffener Endpunkt vor Fits gestoppt')
            C.attrs['matched']=matched
            f.to_parquet(OUT/(ep+'_candidate_private.parquet'),index=False)
            D.to_parquet(OUT/(ep+'_D_private.parquet'),index=False)
            C.to_parquet(OUT/(ep+'_C_private.parquet'),index=False)
            dump(ep+'_flow_private.json',flow)
        UNION=np.unique(np.concatenate([v.subject_id.to_numpy() for v in index_frames.values()]))
        UNION_N=len(UNION)
        # W19 benötigt technische Personen in derselben neuen R1-Union.
        technical_cols=common+[x for x in ['classical_cornell_product','classical_sokolow_lyon','magnitude_area_qrs','spatial_low_voltage_qrs_peak_magnitude','classical_right_axis_deviation'] if x not in common]
        tf=read_entry(manifest['technical'],technical_cols)
        tech=select_index(tf,'reduced_lvef_strict',technical=True)
        tech.to_parquet(OUT/'technical_index_private.parquet',index=False)
        UNION=np.unique(np.concatenate([UNION,tech.subject_id.to_numpy()]))
        UNION_N=len(UNION)
        dump('new_union_private.json',dict(n=UNION_N,dtype=str(UNION.dtype),sort='np.unique numeric ascending',sha256=hashlib.sha256(UNION.tobytes()).hexdigest()))
        dump('input_read_manifest.json',sources)
        # R0-Abschnitte, kleine Zellen werden erst im öffentlichen Export unterdrückt.
        for ep,marker in [('H1',F),('H4',M)]:
            f=frames[ep];idx=index_frames[ep];D=Ds[ep];C=Cs[ep]
            stages=[('candidate',pd.Series(True,index=f.index)),('adult',f.age.ge(18)),('technical',f.age.ge(18)&f.extraction_status.eq('ok')),('reference_available',f.age.ge(18)&f.extraction_status.eq('ok')&f.reference_available.eq(True))]
            counts=[];last_r=len(f);last_p=f.subject_id.nunique()
            for name,mask in stages:
                n=int(mask.sum());p=int(f.loc[mask].subject_id.nunique())
                counts.append(dict(stage=name,recordings=n,persons=p,excluded_recordings=last_r-n,excluded_persons=last_p-p));last_r=n;last_p=p
            for name,ff in [('index',idx),('reference_valid',D),('B3_complete',D.loc[np.isfinite(raw_design(D,marker)[0][:,:-1]).all(1)]),('C_e',C)]:
                n=len(ff);p=int(ff.subject_id.nunique());counts.append(dict(stage=name,recordings=n,persons=p,excluded_recordings=last_r-n,excluded_persons=last_p-p));last_r=n;last_p=p
            results.append(analyze(ep+'_flow','W25',ep,marker,'candidate→D_e→C_e',lambda w,r,counts=counts:dict(stages=counts,class_policy='unknown retained before reference; no backprojection of index labels')))
            class_counts=[]
            for name,mask in stages:
                for category,n in f.loc[mask,'label_state'].astype('object').where(f.loc[mask,'label_state'].notna(),'unknown').value_counts().items():
                    class_counts.append(dict(stage=name,category=str(category),recordings=int(n),persons=int(f.loc[mask&f.label_state.eq(category),'subject_id'].nunique())))
            overlaps=dict(not_adult_n=int((~f.age.ge(18)).sum()),technical_failed_n=int((~f.extraction_status.eq('ok')).sum()),reference_unavailable_n=int((~f.reference_available.eq(True)).sum()))
            echo_valid=idx.has_tte_30d.eq(True)&idx.nearest_tte_delta_hours.abs().le(720)&idx.preferred_lvef.between(0,100)&idx.lvef_conflict_gt_10.eq(False)
            results.append(analyze(ep+'_flow_classes','W25',ep,marker,'candidate per-recording labels; H4 echo parallel',lambda w,r,class_counts=class_counts,overlaps=overlaps,echo_valid=echo_valid:dict(rows=class_counts,overlapping_exclusions=overlaps,index_echo_parallel_n=int(echo_valid.sum()))))
            for population,df in [('D',D),('C',C)]:
                table=[]
                for cl,mask in [('all',pd.Series(True,index=df.index)),('case',df.y.eq(1)),('control',df.y.eq(0))]:
                    sub=df.loc[mask]
                    for col in ['age','heart_rate','qrs_duration_ms','qtc_fridericia_ms']:
                        values=pd.to_numeric(sub[col],errors='coerce').dropna()
                        table.append(dict(variable=col,category=cl,denominator=len(sub),observed_n=len(values),missing_n=len(sub)-len(values),median=values.median(),q1=values.quantile(.25),q3=values.quantile(.75),mean=values.mean(),sd=values.std(ddof=1)))
                    for col in ['sex','sinus_selector','classical_lbbb_pattern_positive','classical_rbbb_pattern_positive']:
                        for category,n in sub[col].astype('object').where(sub[col].notna(),'unknown').astype(str).value_counts().items():table.append(dict(variable=col,category=cl+':'+category,denominator=len(sub),observed_n=int(n),fraction=n/len(sub) if len(sub) else None))
                    for col in ['pacing','AF','setting','anchor_year_group','ICD_comorbidity']:
                        table.append(dict(variable=col,category=cl+':unknown',denominator=len(sub),observed_n=len(sub),status='nicht_verfuegbar'))
                results.append(analyze(ep+'_table1_'+population,'W30',ep,marker,population+'_e',lambda w,r,table=table:dict(rows=table)))
                hist=[]
                echo=df if ep=='H1' else df.loc[df.has_tte_30d.eq(True)&df.preferred_lvef.between(0,100)&df.lvef_conflict_gt_10.eq(False)&df.nearest_tte_delta_hours.abs().le(720)]
                values=pd.to_numeric(echo.preferred_lvef,errors='coerce');valid=np.isfinite(values)&values.between(0,100);lv=values[valid]
                for lower in range(100):hist.append(dict(bin_lower=lower,bin_upper=lower+1,bin_closed='left',n=int(((lv>=lower)&(lv<lower+1)).sum()),denominator=len(lv)))
                hist.append(dict(bin_lower=100,bin_upper=100,bin_closed='both',n=int(lv.eq(100).sum()),denominator=len(lv)))
                meas=[]
                for category,n in echo.preferred_lvef_method.fillna('unknown').value_counts().items():meas.append(dict(category=str(category),n=int(n),denominator=len(echo)))
                results.append(analyze(ep+'_lvef_'+population,'W12a',ep,marker,population+'_e; H4 selected echo subset',lambda w,r,hist=hist,lv=lv,values=values,valid=valid,meas=meas:dict(histogram=hist,measurement=meas,exact_40_n=int(lv.eq(40).sum()),lt40_n=int(lv.lt(40).sum()),ge40_n=int(lv.ge(40).sum()),le40_n=int(lv.le(40).sum()),gt40_n=int(lv.gt(40).sum()),invalid_n=int((~valid).sum()),denominator=len(lv))))
                missing=[]
                for cl in [0,1]:
                    sub=df.loc[df.y.eq(cl)]
                    for col in ['age','sex_numeric','heart_rate','qrs_duration_ms','qtc_fridericia_ms',A,marker]:missing.append(dict(variable=col,category=str(cl),n=int(sub[col].isna().sum()),denominator=len(sub)))
                if population=='D':results.append(analyze(ep+'_missingness','W18',ep,marker,'D_e',lambda w,r,missing=missing:dict(rows=missing)))
        results.append(analyze('overlap','W25','H1/H4','both','D_e/C_e',lambda w,r:dict(D_overlap_n=len(set(Ds['H1'].subject_id)&set(Ds['H4'].subject_id)),C_overlap_n=len(set(Cs['H1'].subject_id)&set(Cs['H4'].subject_id)))))
        results.append(stop_record('W10','W10','H1/H4','both','Historische Ergebnisdateien enthalten nur aggregierte valid/failed/ci95, keine gespeicherten gueltigen Einzelreplikate oder Originalunion-Inhaltshashes. Keine identische Fortsetzung belegt.'))
        for ep,marker in [('H1',F),('H4',M)]:
            D=Ds[ep];C=Cs[ep]
            if not C.attrs.get('matched'):
                for W in ['W20','W24','W13','W19','W18','W16','W26','W7']:results.append(stop_record(ep+'_'+W,W,ep,marker,'Referenz_Index_Abgleich'))
                continue
            pack=params['endpoints'][ep]['markers'][marker];scale=pack['scale'];raw,_=raw_design(C,marker);z=fixed_design(raw,scale);y=C.y.to_numpy(float)
            pos=np.searchsorted(UNION,C.subject_id.to_numpy())
            results.append(analyze(ep+'_association','reference',ep,marker,'C_e',lambda w,r,z=z,y=y:k.association(z,y,w)))
            results.append(analyze(ep+'_prediction','W20',ep,marker,'C_e',lambda w,r,z=z,y=y,models=pack['models']:{**k.prediction(z,y,models,w),'n':len(y),'cases':int(y.sum()),'controls':int((y==0).sum())},pos,1000))
            # W24 darf ohne vorgeschriebene semantische Parserprüfung nicht promotiert werden.
            results.append(stop_record(ep+'_W24_parser','W24',ep,marker,'Entwicklungspruefung der 100 Maschinenbefundaussagen nicht abgeschlossen; keine Stimulations-/AF-Stratenfreigabe'))
            # Unabhängig vom Befundtext attestierte QRS-/Blockflags sind verfügbar.
            for name,selector in [('qrs_ge120',C.qrs_duration_ms.ge(120)),('rbbb',C.classical_rbbb_pattern_positive.eq(True)),('lbbb',C.classical_lbbb_pattern_positive.eq(True)),('qrs_lt120_no_block',C.qrs_duration_ms.lt(120)&C.classical_lbbb_pattern_positive.eq(False)&C.classical_rbbb_pattern_positive.eq(False))]:
                results.append(association_rows(ep,C,marker,scale,ep+'_W24_'+name,'W24',selector,1000))
            if ep=='H4':
                results.append(stop_record('H4_W13','W13',ep,marker,'Freigegebene Projektionen enthalten keine vollstaendig attestierte ICD-Untercodeliste samt Aufnahmehistorie bis Index. Dx-Flag ist kein Ersatz.'))
            # W18 reduzierte/Indikatormodelle, unveränderlicher Index.
            train_start=params['endpoints'][ep]['mi']['start']
            base_raw,_=raw_design(D,marker)
            reduced_good=np.isfinite(np.delete(base_raw,5,axis=1)).all(1)
            reduced=D.loc[reduced_good].copy();rr=base_raw[reduced_good].copy();rr[:,5]=train_start['qtc_fridericia_ms']
            zr=np.delete(fixed_design(rr,scale),6,axis=1);yr=reduced.y.to_numpy(float);pr=np.searchsorted(UNION,reduced.subject_id.to_numpy())
            results.append(analyze(ep+'_no_qtc_extended','W18',ep,marker,'D_e reduced complete',lambda w,r,zr=zr,yr=yr:k.association(zr,yr,w),pr,500))
            results.append(analyze(ep+'_no_qtc_C','W18',ep,marker,'C_e',lambda w,r,z=z,y=y:k.association(np.delete(z,6,axis=1),y,w),pos,500))
            mi_good=np.isfinite(D[[marker,A]].to_numpy(float)).all(1)
            mf=D.loc[mi_good].copy();mf['marker']=mf[marker]
            mi_pos=np.searchsorted(UNION,mf.subject_id.to_numpy())
            indicator_good=np.isfinite(mf[['age','sex_numeric','qrs_duration_ms',A,marker]].to_numpy(float)).all(1)
            indicator=mf.loc[indicator_good].copy();extras=np.column_stack([indicator[c].isna().to_numpy(float) for c in ['heart_rate','qtc_fridericia_ms']])
            for c in ['heart_rate','qtc_fridericia_ms']:indicator[c]=indicator[c].fillna(train_start[c])
            ir,_=raw_design(indicator,marker);iz=fixed_design(ir,scale);iz=np.column_stack([iz[:,:-1],extras,iz[:,-1]]);iy=indicator.y.to_numpy(float)
            ipos=np.searchsorted(UNION,indicator.subject_id.to_numpy())
            results.append(analyze(ep+'_missing_indicator_extended','W18',ep,marker,'D_e median HF/QTc',lambda w,r,iz=iz,iy=iy:k.association(iz,iy,w),ipos,500))
            # Auf C_e sind beide Missing-Indikatoren strukturell Null: expliziter Rang-STOP.
            results.append(stop_record(ep+'_missing_indicator_C','W18',ep,marker,'Beide Missing-Indikatoren auf C_e konstant Null; vorgegebener Rang-STOP, keine Entfernung nach Ergebnissicht'))
            # PMM speichert die vollständige Distanzmatrix; Ressourcenabschätzung vor Aufruf.
            peak_est=max(int(mf[c].isna().sum())*int(mf[c].notna().sum())*16 for c in ['age','sex_numeric','heart_rate','qrs_duration_ms','qtc_fridericia_ms'])
            if peak_est>700*1024**2:
                results.append(stop_record(ep+'_MI','W18',ep,marker,'Ressource: gebundener PMM-Kern mit voller Distanzmatrix ueberschreitet konservatives Speicherbudget; kein Start'))
            else:
                results.append(analyze(ep+'_MI','W18',ep,marker,'D_e marker/angle known',lambda w,r,mf=mf,scale=scale,recipe=params['endpoints'][ep]['mi'],code=int(ep[-1]):k.mi_pool(mf,scale,recipe,code,0 if r<0 else r+1,w,guard),mi_pos,500,190))
            # Alle gebundenen Fehlerverfahren mit Nachtrag-H4-Einschränkung.
            variants=['rc','raw','clipped','logit','spline']+(['joint_rc','joint'] if ep=='H1' else [])
            for variant in variants:
                code=params['simex']['parameter_codes'][variant]
                def corr(w,r,z=z,y=y,scale=scale,variant=variant,code=code,ep=ep,spline=pack['spline']):
                    ww=np.ones(len(y)) if w is None else w
                    out=k.corrected(z,y,ww,scale,params['error'],variant,np.random.SeedSequence([1692026,16,int(ep[-1]),code,0 if r<0 else r+1]),20 if r<0 else 10,spline)
                    for name in ['beta','linear_beta','quadratic_beta']:
                        if name in out:out[name.replace('beta','or')]=float(np.exp(out[name]))
                    out.update(n=len(y),cases=int(y.sum()),controls=int((y==0).sum()))
                    if variant in ['rc','joint_rc']:
                        reference=k.association(z,y,ww)['beta'];out['uncorrected_beta']=reference;out['delta_beta']=out['beta']-reference
                        out['fraction']=out['beta']/reference if abs(reference)>=.01 else None
                    return out
                results.append(analyze(ep+'_error_'+variant,'W16',ep,marker,'C_e; development-fixed error',corr,pos,1000,950))
            if ep=='H4':results.append(stop_record('H4_joint_error','W16',ep,marker,'Gemeinsame H4-Verfahren ausgeschlossen durch Nachtrag 1; keine H1-Fehlerkovarianzuebertragung'))
            cuts=pack['quartiles']
            def risks(w,r,z=z,y=y,scale=scale,cuts=cuts):
                ww=np.ones(len(y)) if w is None else w;b=np.asarray(k.association(z,y,ww)['coefficients']);prob=[]
                from scipy.special import expit
                for median in cuts['medians']:
                    xx=z.copy();xx[:,-1]=(median-scale['mean'][-1])/scale['sd'][-1];prob.append(float(np.average(expit(xx@b),weights=ww)))
                return {**{f'q{i+1}':v for i,v in enumerate(prob)},'q4_minus_q1':prob[-1]-prob[0],'n':len(y),'cases':int(y.sum()),'controls':int((y==0).sum())}
            results.append(analyze(ep+'_standardized','W26',ep,marker,'C_e',risks,pos,500))
            dist=[]
            for cl in [0,1]:
                values=C.loc[C.y.eq(cl),marker];dist.append(dict(category=str(cl),n=len(values),median=values.median(),q1=values.quantile(.25),q3=values.quantile(.75)))
            results.append(analyze(ep+'_marker_distribution','W26',ep,marker,'C_e',lambda w,r,dist=dist,scale=scale,cuts=cuts:dict(distribution=dist,development_sd=scale['sd'][-1],development_mean=scale['mean'][-1],ddof=scale['ddof'],quartiles=cuts)))
            for cross in [F,M]:
                common_good=np.isfinite(C[[F,M]].to_numpy(float)).all(1)
                results.append(association_rows(ep,C.loc[common_good],cross,params['endpoints'][ep]['markers'][cross]['scale'],ep+'_cross_'+('F' if cross==F else 'M'),'W7',planned=500))
        # W19 auf eigenem technischen Indexuniversum, train-Fit gebunden.
        w19(tech,params,original,results)
        dump('result_index.json',[dict(analysis_id=r['analysis_id'],W=r['W'],endpoint=r['endpoint'],status=r['status']) for r in results])
        dump('last_read.json',dict(utc=now(),sealed_excluded=True))
        report()
    except Exception as ex:
        dump('global_stop.json',dict(status='STOP_Gesamt',error_class=type(ex).__name__,detail=str(ex),utc=now()))
        report()
        raise
    finally:
        hashes={str(p):sha(p) for p in [ROOT/'nachrechnung/B_TEST_EXPLORATIV_AW172.md',ROOT/'nachrechnung/b_test_explorativ.csv'] if p.exists()}
        with ledger.open('a',encoding='utf8') as f:f.write(f'\n| {now()} | AW-172 | Ende gebündelter explorativer B-Zugriff | Bindung `{sha(OUT/"bindung_vor_test.json")}` | Ausgabehashes `{json.dumps(hashes,ensure_ascii=False)}`; Status in Abgabe; keine Kennungen |\n')
        dump('access_finished.json',dict(utc=now(),output_hashes=hashes))

def w19(tech,params,original,results):
    import numpy as np,pyarrow.dataset as ds
    from werkzeuge.b_test.statistik import raw_design,fixed_design
    from werkzeuge.b_test.daten import masks
    from werkzeuge.b_test import aw171_kernels as k
    from scipy.special import expit
    model=load(OUT/'w19_development.json');scale=model['scale']
    train=ds.dataset(str(OUT/'w19_train_universe_private.parquet'),format='parquet').to_table().to_pandas()
    raw,_=raw_design(train,F,strong=True);good=np.isfinite(raw).all(1)
    tr=fixed_design(raw[good],scale);tv=train.loc[good,'has_tte_30d'].eq(True).to_numpy(float)
    train_union=np.sort(train.subject_id.unique());train_pos=np.searchsorted(train_union,train.loc[good,'subject_id'].to_numpy())
    # Nur technische Indexwahl wird auf den vorhandenen Referenzvertrag projiziert;
    # kein zweiter first-verified-Index.
    technical=tech.copy();technical['index_selection_strategy']='first_verified_ecg_reference_result_blind'
    z,valid,flow=masks(technical,'reduced_lvef_strict',F,'qrs_duration_ms')
    rawv,_=raw_design(z,F,strong=True);complete=np.isfinite(rawv).all(1)
    selected=valid&complete
    f=z.loc[selected].copy();y=f.preferred_lvef.lt(40).to_numpy(float)
    x=fixed_design(raw_design(f,F)[0],params['endpoints']['H1']['markers'][F]['scale'])
    vx=fixed_design(rawv[selected],scale);positions=np.searchsorted(UNION,f.subject_id.to_numpy())
    vbase=fixed_design(rawv[selected,:-1],original['verification'])
    trbase=fixed_design(raw[good,:-1],original['verification'])
    goodbase=np.isfinite(raw[:,:-1]).all(1)
    trbase=fixed_design(raw[goodbase,:-1],original['verification'])
    tvbase=train.loc[goodbase,'has_tte_30d'].eq(True).to_numpy(float)
    train_pos_base=np.searchsorted(train_union,train.loc[goodbase,'subject_id'].to_numpy())
    def compute(w,r):
        ow=np.ones(len(y)) if w is None else w
        if r<0:
            tw=np.ones(len(tv));b=np.asarray(model['beta']);caps=[model['p95'],model['p99']]
            oldb=np.asarray(original['verification']['beta']);oldcap=original['verification']['p99_train_verified']
        else:
            rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([1692026,19,r])))
            multiples=np.bincount(rng.integers(0,len(train_union),len(train_union)),minlength=len(train_union)).astype(float)
            tw=multiples[train_pos];twbase=multiples[train_pos_base]
            b=k.solver(tr,tv,tw);pp=expit(tr@b)
            def weighted_quantile(v,ww,q):
                take=np.repeat(v,ww.astype(int));return float(np.quantile(take,q,method='linear'))
            caps=[weighted_quantile(1/pp[tv==1],tw[tv==1],q) for q in [.95,.99]]
            oldb=k.solver(trbase,tvbase,twbase);oldpp=expit(trbase@oldb)
            oldcap=weighted_quantile(1/oldpp[tvbase==1],twbase[tvbase==1],.99)
        p=expit(vx@b)
        if not np.isfinite(p).all() or (p<=0).any():raise k.Stop('Propensity')
        out={}
        for name,weights in [('unweighted',np.ones(len(y))),('p95',np.minimum(1/p,caps[0])),('p99',np.minimum(1/p,caps[1])),('old_p99_common_set',np.minimum(1/expit(vbase@oldb),oldcap))]:
            ww=ow*weights;rr=k.association(x,y,ww,ess=name!='unweighted')
            for metric in ['beta','or','delta_auc','wald_low','wald_high']:out[name+'_'+metric]=rr[metric]
            for c in [0,1]:
                wc=ww[y==c];out[name+'_ess_'+str(c)]=float(wc.sum()**2/(wc@wc))
        out.update(n=len(y),cases=int(y.sum()),controls=int((y==0).sum()),p95=caps[0],p99=caps[1],marker_missing_technical_n=int(z[F].isna().sum()),verified_reference_complete_n=int(valid.sum()),B4_marker_complete_technical_n=int(complete.sum()))
        return out
    results.append(analyze('H1_W19','W19','H1',F,'first_technical; fixed train verification design',compute,positions,1000))

def bind():
    # Alle vorgeschriebenen Entwicklungsbindungen stehen vor jeglicher Rohdatenlesung
    # der Auswertungspartition; keine Veränderung nach Daten-/Ergebnissicht.
    if (OUT/'access_started.json').exists():raise RuntimeError('Bindung_nach_Testzugriff_verboten')
    import platform,numpy,pandas,scipy,pyarrow,psutil
    env=dict(python=platform.python_version(),platform=platform.platform(),numpy=numpy.__version__,pandas=pandas.__version__,scipy=scipy.__version__,pyarrow=pyarrow.__version__,psutil=psutil.__version__,threads=1,executable=sys.executable)
    dump('environment.json',dict(environment=env,sha256=canonical(env)))
    from werkzeuge.b_test.aw171_contracts import w7_gate
    w7_gate(load(str(data_path("b.aw171_parameters_json")))['w7'])
    targets=[SPEC,ADD,MAN,ROOT/'werkzeuge/b_test/FREIGABE_K.json',ROOT/'werkzeuge/b_test/parameter_entwicklung.json',
             Path(str(data_path("b.aw171_parameters_json"))),Path(str(data_path("b.aw171_mappings_json"))),
             OUT/'w19_development.json',OUT/'w19_train_universe_private.parquet',OUT/'w19_train_union_private.json',
             OUT/'mappings.json',OUT/'parser_development.json',OUT/'parser_sample_development_private.parquet',OUT/'input_bindings.json',OUT/'checks.json',OUT/'environment.json']
    code=load(str(data_path("b.aw171_code_manifest_json")))
    targets += [ROOT/p for p in code['files']]
    targets += list((ROOT/'werkzeuge/b_test').glob('aw172_*'))
    targets += list(OUT.glob('*_development_optimism.json'))
    targets.append(Path(str(data_path("b.mimic_validation_master_ecg_master_index_parquet"))))
    files={str(p):sha(p) for p in targets if p.is_file()}
    # Historische Ergebnisschemata haben keine Einzelreplikat-/Unionbindung;
    # W10 ist vor der neuen Testsicht geschlossen.
    dump('bindung_vor_test.json',dict(files=files,utc=now(),partition_role='previously_opened_internal',sealed_excluded=True,
         code_sha256=canonical({p:h for p,h in files.items() if p.endswith(('.py','.ps1'))}),
         W10='STOP_Originalunion_gueltige_Einzelreplikate_nicht_gebunden',W24='STOP_semantische_100er_Parserpruefung_nicht_vorliegend',
         W7='open; all eight development cells positive beta and nonnegative delta AUC',
         recipes={'main':1000,'other':500,'R1_minimum':950,'MI_minimum':190,'note':'Nachtrag 1 aendert Zahl, nicht explizit das absolute Minimum. Kein stilles Absenken des KI-Gates.'},
         inputs=load(OUT/'input_bindings.json')))

def check():
    import ast,numpy as np,pandas as pd
    from werkzeuge.b_test import aw171_kernels as k
    checks=[]
    for p in (ROOT/'werkzeuge/b_test').glob('aw172_*.py'):
        ast.parse(p.read_text(encoding='utf8'));checks.append('Syntax '+p.name)
    # Kein Auswertungsfooter/Personenlesen: ausschließlich vorab attestierte
    # Spaltenliste des Manifestes zur Adaptervertragsprüfung.
    manifest=load(MAN)
    required={'subject_id','study_id','split','ecg_time','age','sex','heart_rate','machine_heart_rate_bpm',
              'machine_metadata_usable_for_primary','qrs_duration_ms','qtc_fridericia_ms',A,F,M,'label_state',
              'eligibility_state','index_selection_strategy','reference_available','ehr_time_alignment_confirmatory_eligible',
              'extraction_status','preferred_lvef','lvef_conflict_gt_10','nearest_tte_measurement_id',
              'nearest_tte_delta_hours','time_delta_hours','time_precision','evidence_ids_json','has_tte_30d',
              'preferred_lvef_method','sinus_selector','classical_lbbb_pattern_positive','classical_rbbb_pattern_positive'}
    for key in ['BEF-0063','BEF-0049','technical']:
        entry=manifest['technical'] if key=='technical' else manifest['hypotheses'][key]
        missing=required-set(entry['columns'])
        checks.append(dict(check='Manifest_Adapterspalten_'+key,passed=not missing,missing=sorted(missing)))
    sample=pd.Series([True,False,None],dtype='boolean')
    values=sample.astype('object').where(sample.notna(),'unknown').astype(str)
    assert list(values)==['True','False','unknown'];checks.append('nullable bool unknown ohne Nullsetzung')
    rng=np.random.default_rng(1722026);x=np.column_stack([np.ones(400),rng.normal(size=(400,2))]);y=np.r_[np.zeros(200),np.ones(200)]
    result=k.association(x,y)
    assert result['n']==400 and result['cases']==200 and result['controls']==200;checks.append('Assoziationskern synthetisch')
    assert np.array_equal(np.bincount(np.random.Generator(np.random.PCG64(np.random.SeedSequence([1282026,0]))).integers(0,400,400),minlength=400),np.bincount(np.random.Generator(np.random.PCG64(np.random.SeedSequence([1282026,0]))).integers(0,400,400),minlength=400));checks.append('R1 deterministische Multiplizitaeten')
    dump('checks.json',dict(checks=checks,test_patient_content_read=False,all_passed=all(not isinstance(c,dict) or c['passed'] for c in checks)))
    if not all(not isinstance(c,dict) or c['passed'] for c in checks):raise RuntimeError('Adapterspaltenvertrag_vor_Test_STOP')

def report():
    import csv
    records=[]
    for path in OUT.glob('*.json'):
        if path.name.endswith('_bootstrap.json'):continue
        obj=load(path)
        if isinstance(obj,dict) and 'analysis_id' in obj and 'W' in obj:records.append(obj)
    records.sort(key=lambda r:r['analysis_id'])
    for ep in ['H1','H4']:
        path=OUT/(ep+'_development_optimism.json')
        if path.exists():
            obj=load(path);key=ep+'_optimism_result'
            r=dict(analysis_id=key,W='W20',endpoint=ep,marker=F if ep=='H1' else M,population='development train+validation C_e',status='schaetzbar' if obj['valid']>=190 else 'STOP',point=obj,bootstrap=None)
            dump(key+'.json',r)
            if not any(x['analysis_id']==key for x in records):records.append(r)
    gs=load(OUT/'global_stop.json') if (OUT/'global_stop.json').exists() else None
    bh=sha(OUT/'bindung_vor_test.json') if (OUT/'bindung_vor_test.json').exists() else ''
    columns=['analysis_id','W','endpoint','marker','population','stratum','model','scaling_hash','seed','planned','attempted','valid','status','error_class','metric','value','ci_low','ci_high','ci_type','N_persons','N_cases','N_controls','run','source','source_hash','filter','script_hash','specification_hash','addendum_hash','binding_hash']
    rows=[]
    def append(r,metric,value,extra=None):
        point=r.get('point') or {};bs=r.get('bootstrap') or {};bmetric=bs.get('metrics',{}).get(metric,{})
        ci=bmetric.get('ci') or [None,None]
        row=dict(analysis_id=r['analysis_id'],W=r['W'],endpoint=r['endpoint'],marker=r['marker'],population=r['population'],stratum='all',model='development-fixed prediction' if r['W']=='W20' else 'refit association' if r.get('bootstrap') else 'aggregate/STOP',
                 scaling_hash=sha(str(data_path("b.aw171_parameters_json"))),seed='SeedSequence([1282026,r])' if bs else 'not_applicable',planned=bs.get('planned',0),attempted=bs.get('attempted',0),valid=bmetric.get('valid',0),status=r['status'],error_class=r.get('reason',''),metric=metric,value=value,ci_low=ci[0],ci_high=ci[1],ci_type='person-bootstrap percentile 95%' if ci[0] is not None else 'not_available',N_persons=point.get('n'),N_cases=point.get('cases'),N_controls=point.get('controls'),run='A155-2026-10-04-d6e70e',source=str(OUT/(r['analysis_id']+'.json')),source_hash=sha(OUT/(r['analysis_id']+'.json')),filter=r['population'],script_hash=sha(__file__),specification_hash=sha(SPEC),addendum_hash=sha(ADD),binding_hash=bh)
        if bs and ci[0] is None and r['status']=='schaetzbar':row['status']='Punkt_schaetzbar_KI_nicht_schaetzbar'
        if extra:row.update(extra)
        # Kleinzellen einschließlich ergänzender Prozent-/Statistikwerte unterdrücken.
        small=(extra and any(isinstance(extra.get(k),(int,float)) and extra[k]<10 for k in ['N_persons','observed_n']))
        count_metric=metric.split('.')[-1] in ['n','persons','recordings','observed_n','missing_n','cases','controls','exact_40_n','lt40_n','ge40_n','le40_n','gt40_n','invalid_n','D_overlap_n','C_overlap_n','excluded_persons','excluded_recordings']
        if small or (count_metric and isinstance(value,(int,float)) and value<10):
            row.update(value=None,ci_low=None,ci_high=None,status='unterdrueckt_lt10')
        rows.append(row)
    for r in records:
        if r['status']=='STOP':append(r,'status',None);continue
        def flatten(v,prefix='',context=None):
            if isinstance(v,dict):
                context={**(context or {}),**{k:v[k] for k in ['variable','category','stage','bin_lower','bin_upper','bin_closed'] if k in v}}
                for k,x in v.items():flatten(x,prefix+'.'+k if prefix else k,context)
            elif isinstance(v,list):
                for i,x in enumerate(v):flatten(x,prefix+'.'+str(i),context)
            elif isinstance(v,(int,float)):
                append(r,prefix,v,{'stratum':json.dumps(context,ensure_ascii=False)} if context else None)
        flatten(r.get('point',{}))
        for collection in ['rows','histogram','distribution','measurement','stages']:
            for i,item in enumerate(r.get('point',{}).get(collection,[])):
                den=item.get('denominator',item.get('persons',item.get('n')))
                count=item.get('observed_n',item.get('n'))
                # Überschreibt die flachen Rohzeilen nicht: diese werden unten sicher gefiltert.
                if count is not None and count<10:
                    prefix=collection+'.'+str(i)+'.'
                    for row in rows:
                        if row['analysis_id']==r['analysis_id'] and row['metric'].startswith(prefix):
                            row.update(value=None,ci_low=None,ci_high=None,status='unterdrueckt_lt10')
                if 0<item.get('missing_n',0)<10:
                    prefix=collection+'.'+str(i)+'.'
                    for row in rows:
                        if row['analysis_id']==r['analysis_id'] and row['metric'] in [prefix+'missing_n',prefix+'observed_n']:
                            row.update(value=None,ci_low=None,ci_high=None,status='unterdrueckt_lt10_komplementaer')
    for sec,W in [('3','W25/W30'),('4','W24'),('5','W20'),('6','W16'),('7','W13'),('8','W10'),('9','W19'),('10','W18'),('11','W12a'),('12','W26'),('13','W7')]:
        if not any(r['W'] in W.split('/') for r in records):
            rows.append(dict(analysis_id='section_'+sec,W=W,endpoint='H1/H4',status='nicht_ausgefuehrt_GLOBAL_STOP' if gs else 'nicht_ausgefuehrt',error_class=gs['detail'] if gs else 'noch_nicht_abgeschlossen',metric='status',source=str(OUT/'global_stop.json'),binding_hash=bh,specification_hash=sha(SPEC),addendum_hash=sha(ADD)))
    target=ROOT/'nachrechnung/b_test_explorativ.csv'
    with target.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    lines=['# B - explorativer Testlauf AW-172', '',
           '**Ergebnis:** '+('Gesamt-STOP vor Fits: '+gs['detail'] if gs else 'Gebündelter Lauf; Status und Nichtschätzbarkeiten je Abschnitt unten. Keine neue Primärprüfung.'), '',
           'Nachträgliche explorative Analyse nach Kenntnis der H1/H4-Ergebnisse, vor der Zusatzrechnung festgelegt, auf derselben vorbelasteten internen Auswertungspartition. Die gesperrte Testmenge bleibt ausgeschlossen.', '',
           'Jede Zahl stammt aus den eigenen privaten Ergebnisdateien auf E:, siehe CSV-Spalten source/source_hash/filter/script_hash. Kleine Zellen (<10 Personen) sind unterdrückt. Fehlende Zahlen sind weder Nullbefunde noch Schätzwerte. 500 Ziehungen erfüllen das unveränderte R1-Minimum von 950 nicht; solche Bootstrap-KI werden nicht behauptet.', '',
           '| Abschnitt | Analyse | Status | Personen/Fälle/Kontrollen | OR bzw. Kennzahl | 95%-KI |', '|---|---|---|---|---|---|']
    for r in records:
        p=r.get('point') or {};bs=r.get('bootstrap') or {};metric='or' if 'or' in p else 'delta_marker' if 'delta_marker' in p else None
        ci=bs.get('metrics',{}).get(metric,{}).get('ci') if metric else None
        n='/'.join(str(p.get(k,'-')) for k in ['n','cases','controls'])
        st=r['status']+(': '+r['reason'] if r.get('reason') else '')
        if bs:st+=f'; {bs["attempted"]}/{bs["planned"]} versucht, Minimum {bs["minimum"]}'
        lines.append(f'| {r["W"]} | {r["analysis_id"]} | {st} | {n} | {p.get(metric,"-") if metric else "CSV"} | {ci or "nicht verfügbar / R0"} |')
    lines += ['', '## Englische Textbausteine für B v3', '',
              '“These post hoc exploratory analyses were specified after inspection of the H1/H4 results and evaluated in the same previously accessed internal evaluation partition. They do not constitute independent or unseen validation.”', '',
              'Table 1 and flow diagram: numerical cells and denominators are provided in the aggregate CSV for W25/W30; unavailable upstream information must be labelled unavailable. No Table 1 significance tests were performed.', '',
              '“Development-fixed predictions used the original development scaling, frozen coefficients and control-derived thresholds. Confidence intervals are conditional on the developed models.”', '',
              '“The historical bootstrap continuation was not performed where the original identity union and replicate-level records could not be verified. No primary results were replaced.”', '',
              '“Unavailable semantic parser verification, incomplete source contracts, model failures and resource-related non-estimability are reported explicitly. These limitations must not be interpreted as absence of confounding or robustness.”', '',
              'Supplement: vollständige numerische Aggregatzeilen mit Quellenhashes, Modell-/Fallset- und Replikatregime in `b_test_explorativ.csv`; keine Personenzeilen.', '',
              '## F1-F8', '',
              'F1 eingeschränkt: Winkel-/Marker-Geometrie und ICD-Proxy; kein Mechanismusbeweis. F2 eingeschränkt: ausgewählte Krankenhaus-/Referenzpopulation; keine Screening-PPV/NPV. F3 eingeschränkt: feste Entwicklungsschwellen, keine Optimierung auf der Auswertung. F4 eingeschränkt: post hoc und vorbelastete Partition, keine neue Erfolgsfamilie. F5 offen: informative Missingness/MAR nicht bewiesen. F6 eingeschränkt: feste B2/B3-Komparatoren, keine globale Optimalität. F7 offen: Verifikationsbias/IPW-Annahmen, keine Prognose- oder Innerhalb-Person-Aussage. F8 offen: Parserprüfung/Quellenqualität, Text nur für Strata.', '',
              '## Bindung und Abnahmegrenzen', '',f'Bindung vor Test: `{bh}`. Spezifikation: `{sha(SPEC)}`. Nachtrag: `{sha(ADD)}`. Skript: `{sha(__file__)}`.', '',
              'Keine klinische oder Publikationsfreigabe. Auftragsstatus und Fachstatus sind getrennt; STOP ist ein technischer/vertraglicher Befund, kein empirischer Nullbefund.']
    (ROOT/'nachrechnung/B_TEST_EXPLORATIV_AW172.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    dump('report_manifest.json',dict(utc=now(),aggregate_rows=len(rows),files={str(p):sha(p) for p in [target,ROOT/'nachrechnung/B_TEST_EXPLORATIV_AW172.md']}))

if __name__=='__main__':
    try:
        guard();status('running')
        source_hash=sha(__file__)
        snapshot=OUT/('executed_source_'+source_hash+'.py')
        if not snapshot.exists():snapshot.write_bytes(Path(__file__).read_bytes())
        dump(PHASE+'_executed.json',dict(utc=now(),pid=os.getpid(),source_hash=source_hash,memory_start=memory()))
        globals()[PHASE]()
        status('complete',peak=PEAK,memory_end=memory())
    except Exception as ex:
        status('STOP',error_class=type(ex).__name__)
        dump('failure.json',dict(phase=PHASE,error_class=type(ex).__name__,detail=str(ex),utc=now()))
        traceback.print_exc()
