"""AW067 / AE057: pilotbasiertes Tor und fortsetzbarer Matrixvergleich."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os
os.environ.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,json,time,hashlib,threading,argparse,gc
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
from .inventar import OUT,PROJECT,SAMPLE,CONTEXT,FREEZE,ROOT
from werkzeuge.varianten.audit import memory_guard
from werkzeuge.varianten.adapter import release_import,prepare_record,OPTIONS
from werkzeuge.varianten.lauf import sha
from werkzeuge.varianten.vertrag import development_file

RUN=OUT/'fortsetzung_v3'
SEED=20261009
def dump(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str),encoding='utf8');temp.replace(path)
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def wait_memory():
    while True:
        try:return memory_guard()
        except RuntimeError:
            print('{"phase":"resource_wait"}',flush=True);time.sleep(30)
class Budget:
    def __enter__(self):
        import psutil
        self.peak=0;self.done=threading.Event();self.process=psutil.Process()
        def watch():
            while not self.done.wait(.1):
                m=self.process.memory_info();self.peak=max(self.peak,m.rss,getattr(m,'peak_wset',0))
                if self.peak>1.5*1024**3:
                    dump(RUN/'rss_stop.json',{'peak':self.peak});os._exit(75)
        self.thread=threading.Thread(target=watch,daemon=True);self.thread.start();return self
    def __exit__(self,*args):self.done.set();self.thread.join()
def catalog():
    import pandas as pd
    return pd.read_csv(OUT/'inventar.csv').marker_id.tolist()
def binding():
    inv=read(OUT/'inventar.json')
    for p,key in ((SAMPLE,'sample'),(CONTEXT,'context'),(FREEZE,'freeze')):
        assert sha(p)==inv['binding'][key],key
    assert all(sha(ROOT/p)==h for p,h in inv['binding']['release'].items()),'Release verändert'
    spec=PROJECT/'nachrechnung/SPEZIFIKATION_AW067.md'
    section=spec.read_text(encoding='utf8').split('## Nachtrag zur Fortsetzung gemäß K / AE-057\n')[1]
    return {'sample':sha(SAMPLE),'context':sha(CONTEXT),'freeze':sha(FREEZE),'spec':sha(spec),
            'nachtrag':hashlib.sha256(section.encode()).hexdigest(),
            'code':{str(p):sha(p) for p in [Path(__file__),Path(__file__).parent/'metriken.py',PROJECT/'werkzeuge/varianten/adapter.py',PROJECT/'werkzeuge/varianten/audit.py',PROJECT/'werkzeuge/varianten/vertrag.py']},
            'release_binding':hashlib.sha256(json.dumps(inv['binding']['release'],sort_keys=True).encode()).hexdigest()}
def rows():
    import pyarrow.parquet as pq
    from werkzeuge.varianten.aw023b import MAPPING,MANIFEST,resolve_waveform
    import pyarrow.dataset as ds
    pq.ParquetFile(SAMPLE);ids=pq.read_table(SAMPLE,columns=['subject_id','study_id','split']).to_pylist()
    pq.ParquetFile(CONTEXT);ctx=pq.read_table(CONTEXT).to_pylist();by={r['study_id']:r for r in ctx}
    assert len(ids)==len(ctx)==5648 and len(set(r['subject_id'] for r in ids))==5648
    assert sha(MAPPING)==read(MANIFEST)['sha256']
    pq.ParquetFile(MAPPING)
    mapping=ds.dataset(str(MAPPING),format='parquet').to_table(columns=['subject_id','study_id','split','relative_waveform_path'],filter=ds.field('study_id').isin([r['study_id'] for r in ids])).to_pandas().drop_duplicates()
    assert not mapping.study_id.duplicated().any()
    paths=mapping.set_index('study_id').relative_waveform_path.to_dict()
    result=[]
    for a in ids:
        r=by[a['study_id']]
        assert a['split'] in ('train','validation') and all(r[k]==a[k] for k in a)
        assert Path(r['waveform_path']).resolve()==Path(resolve_waveform(paths[a['study_id']])).resolve()
        result.append(r)
    return result
def finite(v):
    import numpy as np
    return v is not None and np.isfinite(v)
def kinds(a,b,n):
    import numpy as np
    av=a.get(n);bv=b.get(n);af=finite(av);bf=finite(bv);out=[]
    if n not in b:out.append('reference_column_absent')
    elif af!=bf:out.append('missingness')
    elif af and not np.isclose(av,bv,rtol=1e-9,atol=0):out.append('value')
    # Metadaten nur einmal je Aufnahme parsen, nicht viermal je Marker.
    for rec in (a,b):
        if '_aw067_metadata' not in rec:
            rec['_aw067_metadata']={key:json.loads(rec.get(key) or '{}') for key in ('feature_confidence_json','feature_unevaluable_reason_json')}
    for key,kind in (('feature_confidence_json','confidence'),('feature_unevaluable_reason_json','abstention_reason')):
        if a['_aw067_metadata'][key].get(n)!=b['_aw067_metadata'][key].get(n):out.append(kind)
    return out
def compute(package,row,names,method='kors',historical=False):
    import numpy as np
    fr=release_import();available=set(fr.BiomarkerRegistry.available());reg=[n for n in names if n in available]
    record=fr.VCGTransformRegistry.transform(package['paths']['morphology'],method).xyz
    robust=fr.VCGTransformRegistry.transform(package['robust'],method).xyz
    wide=record if package['wideband_supported'] and 'wideband' in package['paths'] and not historical else None
    results=fr.BiomarkerRegistry.compute_all(delineation=package['delineation'],ecg_12lead=package['robust'],vcg_xyz=robust,fs=package['fs'],names=reg,
        record_ecg_12lead=package['paths']['morphology'],record_vcg_xyz=record,record_vcg_wideband_xyz=wide,
        record_raw_ecg_12lead=package['loaded'].signal_millivolts,record_acquisition=package['acquisition'])
    dest={};meta={};fail=[]
    fr._store_biomarker_results(dest,results,fail,metadata=meta);fr._store_feature_metadata(dest,meta)
    if fail:dump(RUN/'calculation_failures_last.json',{'failures':fail})
    if any(n.startswith('classical_') for n in names):
        report=fr._REPORT_PARSER.parse(str(row.get('machine_report') or ''))
        positive=lambda key:report[key].state==fr.ReportMentionState.POSITIVE
        classic=fr.ClassicalECGCriteria().compute(package['paths']['diagnostic'],package['delineation'],age=fr._optional_int(row.get('age')),sex=row.get('sex'),
            machine_qrs_axis_deg=fr._optional_float(row.get('machine_qrs_axis_deg')),machine_t_axis_deg=fr._optional_float(row.get('machine_t_axis_deg')),
            paced_or_lbbb=positive('pacemaker') or positive('lbbb'),paced=positive('pacemaker'),measurement_sink={})
        for n in names:
            if not n.startswith('classical_'):continue
            c=classic.get(n.removeprefix('classical_').removesuffix('_positive'))
            dest[n]=(c.positive if n.endswith('_positive') else c.value) if c is not None and c.evaluable else np.nan
    return dest
def options(names):
    available=set(release_import().BiomarkerRegistry.available())
    return {**OPTIONS,'vcg_feature_ids':[n for n in names if n in available],'sensitivity_feature_ids':[],
            'sensitivity_vcg_methods':(),'classical_feature_ids':sorted(set(n.removesuffix('_positive') for n in names if n.startswith('classical_')))}
def references(ids,names):
    """Nur gehashte Entwicklungswhitelist; einzeldateiweise ID-Filterprojektion."""
    import pyarrow.parquet as pq,pyarrow.dataset as ds
    wanted=[r['study_id'] for r in ids];out={};sources=[]
    indexpath=RUN/'referenz_index.json'
    indexbinding={'sample':sha(SAMPLE),'freeze':sha(FREEZE)}
    previous_index=OUT/'fortsetzung_v2/referenz_index.json'
    if not indexpath.exists() and previous_index.exists():
        oldindex=read(previous_index);assert oldindex['binding']==indexbinding
        dump(indexpath,oldindex)
    if indexpath.exists():
        cached=read(indexpath);assert cached['binding']==indexbinding
        index=cached['index']
    else:
        pq.ParquetFile(SAMPLE);sample=pq.read_table(SAMPLE,columns=['study_id','split']).to_pylist()
        index={};byfold={fold:[r['study_id'] for r in sample if r['split']==fold] for fold in ('train','validation')}
        for i,item in enumerate(read(FREEZE)['feature_sources']):
            p=Path(item['path']);fold=p.parent.name.split('=')[1];development_file(p,fold);pq.ParquetFile(p)
            identities=ds.dataset(str(p),format='parquet').to_table(columns=['study_id'],filter=ds.field('study_id').isin(byfold[fold]))
            for identity in identities.to_pylist():
                key=str(identity['study_id']);assert key not in index;index[key]=item
            if i%500==0:print(json.dumps({'phase':'reference_index','files':i+1,'matched':len(index)}),flush=True)
        assert len(index)==5648
        dump(indexpath,{'binding':indexbinding,'index':index})
    selected_sources={index[str(r['study_id'])]['path']:index[str(r['study_id'])] for r in ids}
    # Freeze enthält die bereits festgelegte Liste; keine Verzeichnisrekursion.
    for item in selected_sources.values():
        p=Path(item['path']);fold=p.parent.name.split('=')[1];development_file(p,fold)
        footer=pq.ParquetFile(p);sc=footer.schema_arrow
        # Zeilengruppenstatistik kann I/O sparen; sie ersetzt nie die Whitelist.
        idx=sc.get_field_index('study_id');possible=False
        selected=[r['study_id'] for r in ids if r['split']==fold]
        for j in range(footer.num_row_groups):
            stat=footer.metadata.row_group(j).column(idx).statistics
            if stat is None or not stat.has_min_max or any(stat.min<=v<=stat.max for v in selected):possible=True;break
        if not possible:continue
        identities=ds.dataset(str(p),format='parquet').to_table(columns=['study_id'],filter=ds.field('study_id').isin(selected))
        if not identities.num_rows:continue
        cols=['subject_id','study_id','split','extraction_status','feature_confidence_json','feature_unevaluable_reason_json',*names]
        cols=list(dict.fromkeys(c for c in cols if c in sc.names))
        table=ds.dataset(str(p),format='parquet').to_table(columns=cols,filter=ds.field('study_id').isin(selected))
        if not table.num_rows:continue
        # Voller Quellhash und gehashte Projektion dieser Fortsetzung festhalten.
        source={'path':str(p),'sha256':sha(p),'columns':cols,'rows':table.num_rows}
        if 'sha256' in item:assert source['sha256']==item['sha256']
        assert p.stat().st_size==item.get('file_size',p.stat().st_size),'Quellgröße verändert'
        sources.append(source)
        for r in table.to_pylist():
            assert r['split']==fold and r['study_id'] in wanted and r['study_id'] not in out
            out[r['study_id']]=r
    assert len(out)==len(ids),'Referenzdeckung'
    return out,sources
def diagnose():
    wait_memory();start=time.perf_counter();d=read(OUT/'gate_failure_privat.json');fr=release_import();n='t_loop_to_qrs_loop_complexity_ratio'
    package=prepare_record(d['row']);tests=[]
    for label,names in (('single',[n]),('sorted_catalog',catalog()),('reversed_catalog',list(reversed(catalog())))):
        wait_memory();a=compute(package,d['row'],names)
        tests.append({'panel':label,'value':a.get(n),'difference_types':kinds(a,d['expected'],n),
                      'reason':json.loads(a.get('feature_unevaluable_reason_json') or '{}').get(n)})
    del package
    report={'started_utc':datetime.now(timezone.utc).isoformat(),'seconds':time.perf_counter()-start,'tests':tests,
            'context_fields':list(d['row']),'source_path':'core/biomarkers/catalog_biomarkers.py:_loop_extra_feature/_kinematic_features',
            'source_conclusion':'Ratio verwendet QRS-/T-Kinematik; Registry setzt FeatureContext vor/nach jedem compute_all zurück. Historische Input-/Erzeugeridentität nicht aus Outputhash ableitbar.',
            'cause_proven':False,'release_changed':False}
    assert report['seconds']<7200;dump(RUN/'diagnose.json',report)
    print(json.dumps({'phase':'diagnose','seconds':report['seconds'],'panels':len(tests),'cause_proven':False}),flush=True)
def pilot():
    wait_memory();b=binding();allrows=rows();names=catalog();pilot=[]
    for fold in ('train','validation'):
        rank=lambda r:hashlib.sha256(f'{SEED}|AW067|pilot|{fold}|{r["subject_id"]}|{r["study_id"]}'.encode()).hexdigest()
        pilot.extend(sorted([r for r in allrows if r['split']==fold],key=rank)[:50])
    dump(RUN/'pilot_liste.json',pilot)
    manifest={'binding':b,'seed':SEED,'pilot_sha256':sha(RUN/'pilot_liste.json'),'n':100,'folds':{'train':50,'validation':50},'created_utc':datetime.now(timezone.utc).isoformat(),'dower_computed':False}
    if (RUN/'pilot_manifest.json').exists():assert read(RUN/'pilot_manifest.json')['binding']==b
    else:dump(RUN/'pilot_manifest.json',manifest)
    refs,sources=references(pilot,names);dump(RUN/'pilot_quellen.json',sources)
    opts=options(names);start=time.perf_counter()
    for i,row in enumerate(pilot):
        path=RUN/'pilot'/f'{i:03d}.json'
        if path.exists():
            saved=read(path);assert saved['study_id']==row['study_id'] and saved['binding']==b;continue
        wait_memory();tic=time.perf_counter();ref=refs[row['study_id']]
        actual=release_import()._extract_record_worker((row,opts))
        local=compute(prepare_record(row),row,names) if actual['extraction_status']=='ok' else actual
        errors={n:kinds(local,ref,n) for n in names};errors={n:k for n,k in errors.items() if k}
        # Bei QC-Abstention ist local dasselbe unveränderte Workerobjekt.
        # Gemeinsam nicht ausgegebene Felder sind keine Adapterabweichung.
        worker_errors={n:kinds(local,actual,n) for n in names} if actual['extraction_status']=='ok' else {}
        worker_errors={n:k for n,k in worker_errors.items() if k}
        dump(path,{'study_id':row['study_id'],'split':row['split'],'binding':b,'errors':errors,'worker_errors':worker_errors,
                   'finite_actual':[n for n in names if finite(local.get(n))],'finite_reference':[n for n in names if finite(ref.get(n))],
                   'status_actual':actual['extraction_status'],'status_reference':ref['extraction_status'],'seconds':time.perf_counter()-tic})
        print(json.dumps({'phase':'pilot','completed':i+1,'value_or_availability_errors':sum(any(k in e for k in ('value','missingness','reference_column_absent')) for e in errors.values()),'seconds':time.perf_counter()-tic}),flush=True)
        del actual,local;gc.collect()
    summaries=[]
    records=[read(RUN/'pilot'/f'{i:03d}.json') for i in range(100)]
    for n in names:
        bad=sum(any(k in r['errors'].get(n,[]) for k in ('value','missingness','reference_column_absent')) for r in records)
        local_bad=sum(bool(r['worker_errors'].get(n)) for r in records)
        summaries.append({'marker_id':n,'n_checked':100,'n_matches':100-bad,'admitted':bad<=1 and local_bad==0,
                          'difference_types':'|'.join(sorted(set(k for r in records for k in r['errors'].get(n,[])))),
                          'local_worker_mismatch':local_bad,'finite_pilot':sum(n in r['finite_actual'] for r in records),
                          'confidence_mismatch':sum('confidence' in r['errors'].get(n,[]) for r in records),
                          'reason_mismatch':sum('abstention_reason' in r['errors'].get(n,[]) for r in records)})
    import pandas as pd
    pd.DataFrame(summaries).to_csv(RUN/'pilot_marker.csv',index=False)
    dump(RUN/'pilot_ergebnis.json',{'binding':b,'status':'completed','n':100,'admitted':sum(r['admitted'] for r in summaries),
                                   'not_admitted':[r['marker_id'] for r in summaries if not r['admitted']],'seconds_this_invocation':time.perf_counter()-start,
                                   'total_compute_seconds':sum(read(RUN/'pilot'/f'{i:03d}.json')['seconds'] for i in range(100)),
                                   'marker_sha256':sha(RUN/'pilot_marker.csv'),'dower_computed':False})
    print(json.dumps({'phase':'pilot_finished','admitted':sum(r['admitted'] for r in summaries)}),flush=True)
def full():
    import pandas as pd,pyarrow as pa,pyarrow.parquet as pq
    wait_memory();b=binding();p=read(RUN/'pilot_ergebnis.json');assert p['status']=='completed' and p['binding']==b
    allrows=rows();names=catalog();gate=pd.read_csv(RUN/'pilot_marker.csv');allowed=gate.loc[gate.admitted,'marker_id'].tolist()
    assert allowed,'Keine pilotreproduzierten Marker: keine Dowerfreigabe'
    manifest={'binding':b,'pilot_result_sha256':sha(RUN/'pilot_ergebnis.json'),'pilot_marker_sha256':sha(RUN/'pilot_marker.csv'),'n':5648,'block_size':200,'admitted':allowed}
    if (RUN/'voll_manifest.json').exists():assert read(RUN/'voll_manifest.json')==manifest
    else:dump(RUN/'voll_manifest.json',manifest)
    twins=pd.read_csv(str(data_path("a.aw056_transform_final_csv"))).marker_id.tolist()
    start=time.perf_counter();completed=0
    for block,startidx in enumerate(range(0,len(allrows),200)):
        wait_memory();selected=allrows[startidx:startidx+200];blockpath=RUN/'bloecke'/f'block_{block:03d}.parquet';seal=blockpath.with_suffix('.json')
        if seal.exists():
            info=read(seal);assert sha(blockpath)==info['sha256'] and info['binding']==b;completed+=info['rows'];continue
        # 37 historische Sensitivitätswerte im selben Fold und derselben Liste.
        refs,sources=references(selected,['sensitivity_inverse_dower__'+n for n in twins]+allowed)
        dump(RUN/'quellen'/f'{block:03d}.json',sources)
        values=[]
        for j,row in enumerate(selected):
            index=startidx+j;target=RUN/'einzel'/f'{index:04d}.json'
            if target.exists():
                entry=read(target);assert entry['study_id']==row['study_id'] and entry['binding']==b
            else:
                memory=wait_memory();tic=time.perf_counter();ref=refs[row['study_id']]
                if ref['extraction_status']!='ok':
                    # QC-Verfügbarkeit durch unveränderten Worker erneut prüfen.
                    k=release_import()._extract_record_worker((row,options(allowed)));assert k['extraction_status']==ref['extraction_status']
                    d={};historical={}
                else:
                    package=prepare_record(row);k=compute(package,row,allowed,'kors');d=compute(package,row,allowed,'inverse_dower')
                    historical=compute(package,row,[n for n in twins if n in allowed],'inverse_dower',historical=True);del package
                kd={n:kinds(k,ref,n) for n in allowed};kd={n:errors for n,errors in kd.items() if errors}
                twinerrors={};twinchecked=[]
                for n in twins:
                    if n not in allowed:continue
                    pref='sensitivity_inverse_dower__'+n
                    if pref not in ref:twinerrors[n]=['reference_column_absent'];continue
                    twinchecked.append(n)
                    x=historical.get(n);y=ref.get(pref)
                    if finite(x)!=finite(y):twinerrors[n]=['missingness']
                    elif finite(x):
                        import numpy as np
                        if not np.isclose(x,y,rtol=1e-9,atol=0):twinerrors[n]=['value']
                entry={'study_id':row['study_id'],'subject_id':row['subject_id'],'split':row['split'],'binding':b,
                       'kors':{n:float(k[n]) if finite(k.get(n)) else None for n in allowed},
                       'dower':{n:float(d[n]) if finite(d.get(n)) else None for n in allowed},
                       'kors_reference_errors':kd,'twin_errors':twinerrors,'twin_checked':twinchecked,
                       'status_reference':ref['extraction_status'],'seconds':time.perf_counter()-tic,'memory_start':memory}
                dump(target,entry)
            values.append({'subject_id':entry['subject_id'],'study_id':entry['study_id'],'split':entry['split'],
                           **{'kors__'+n:v for n,v in entry['kors'].items()},**{'dower__'+n:v for n,v in entry['dower'].items()}})
            completed+=1
            if completed%5==0:print(json.dumps({'phase':'full','completed':completed,'n':5648,'seconds_this_invocation':time.perf_counter()-start}),flush=True)
            dump(RUN/'laufstatus.json',{'status':'running','completed':completed,'n':5648,'binding':b})
            gc.collect()
        blockpath.parent.mkdir(parents=True,exist_ok=True)
        schema=pa.schema([('subject_id',pa.int64()),('study_id',pa.int64()),('split',pa.string())]+[(prefix+n,pa.float64()) for prefix in ('kors__','dower__') for n in allowed])
        pq.write_table(pa.Table.from_pylist(values,schema=schema),blockpath,compression='zstd')
        dump(seal,{'binding':b,'sha256':sha(blockpath),'rows':len(values),'start':startidx})
        del refs,values;gc.collect()
    dump(RUN/'laufstatus.json',{'status':'completed','completed':completed,'n':5648,'binding':b})
def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['diagnose','pilot','full']);args=parser.parse_args()
    wait_memory();RUN.mkdir(parents=True,exist_ok=True)
    with Budget() as budget:
        globals()[args.mode]()
    dump(RUN/(args.mode+'_ressourcen.json'),{'peak_rss_bytes':budget.peak})
if __name__=='__main__':main()
