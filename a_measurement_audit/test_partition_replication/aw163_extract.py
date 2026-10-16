"""W/M extraction for the test-partition replication: bound selection, no replication statistics."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,threading,traceback,gc,importlib.metadata
from pathlib import Path
sys.dont_write_bytecode=True
from . import aw159_inputs as a
from .aw163_preflight import ROOT,OUT,REL,EXCLUDED,sha
OLD=Path(str(data_path("a.aw159")))
a.OUT=OUT
a.SOURCES={}
PHASE='binding'
def clean(x):
    import numpy as np
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.generic):x=x.item()
    return None if isinstance(x,float) and not np.isfinite(x) else x
def log(event,**kw):
    with (OUT/'access_private.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(clean({'utc':a.utc(),'pid':os.getpid(),'auftrag':'AW-163','phase':PHASE,'event':event,**kw}),ensure_ascii=False)+'\n')
def ledger(event,scope):
    # Calls of the earlier input module are redirected exclusively to the private ledger.
    log(event,scope=scope)
a.ledger=ledger
def public(phase,scope):
    with (ROOT/'koordination/TESTZUGRIFFE.md').open('a',encoding='utf8') as f:
        f.write(f'\n| {a.utc()} | AW-163 | {phase} | Runner `{sha(__file__)}`; Addendum `0d2a24a18fc8e0dcbb10e992800129c4cf87bf096b2aad7a6729bce1774e4ca7` | {scope}; details only in the private location |\n')
def readsmall(p):
    p=Path(p)
    if p.stat().st_size>20*1024**2:raise ValueError('Only small JSON inputs are allowed')
    return json.loads(p.read_text(encoding='utf8'))
def manifest_top(p):
    """Pretty-JSON stream of the earlier input module: skip the sources array, parse bounded top-level values."""
    result={};key=None;buf=[]
    with Path(p).open(encoding='utf8') as f:
        for line in f:
            if line.startswith('  "'):
                if key is not None and key!='sources':result[key]=json.loads(''.join(buf).strip().rstrip(','))
                key,value=line.strip().split(':',1);key=json.loads(key);buf=[value]
            elif key is not None:
                if line=='}' or line=='}\n':
                    if key!='sources':result[key]=json.loads(''.join(buf).strip().rstrip(','))
                    key=None
                elif key!='sources':buf.append(line)
    return result
class Budget:
    def __enter__(self):
        import psutil
        self.done=threading.Event();self.peak=0;self.private=0
        def watch():
            p=psutil.Process()
            while not self.done.wait(.2):
                m=p.memory_info();self.peak=max(self.peak,m.rss);self.private=max(self.private,getattr(m,'private',m.vms))
                if max(self.peak,self.private)>1.5*1024**3:
                    a.save(f'budget_stop_{PHASE}.json',{'pid':os.getpid(),'peak_rss':self.peak,'peak_private':self.private});os._exit(75)
        self.thread=threading.Thread(target=watch,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        self.done.set();self.thread.join();a.save(f'resources_{PHASE}.json',{'peak_rss':self.peak,'peak_private':self.private})
def binding():
    assert readsmall(OUT/'status_preflight.json')['status']=='DEVELOPMENT_BENCHMARKS_BOUND'
    public('Start of input binding','input manifest and bound selections; no new test measurements')
    manifest=OLD/'a_test_input_manifest.json';log('begin',type='bytehash_provenance',path=str(manifest))
    assert sha(manifest)=='a7e1320abbf243170ac18b2f8163203244f76195abfc08dbddc2e178d474215c'
    m=manifest_top(manifest)
    log('end',type='bytehash_provenance',path=str(manifest),sha256=sha(manifest),sources_array_parsed=False)
    assert m['disjointness']['passed']
    bound=[]
    for item in m['test_input_files']:
        log('begin',type='input_hash',path=item['path'],expected_sha=item['sha256'])
        assert sha(item['path'])==item['sha256']
        producer=item['producer']
        assert sha(producer['code_path'])==producer['code_sha256']
        assert sha(producer['settings_binding_path'])==producer['settings_binding_sha256']
        bound.append(item);log('end',type='input_hash',path=item['path'],sha256=item['sha256'])
    for item in m['attestations'].values():assert sha(item['path'])==item['sha256']
    release=readsmall(OLD/'release_python_manifest.json')
    assert sha(OLD/'release_python_manifest.json')==m['producer_binding']['release_python_tree_manifest_sha256']
    assert all(sha(x['path'])==x['sha256'] for x in release)
    assert all(importlib.metadata.version(k)==v for k,v in m['producer_binding']['packages'].items())
    # Recheck the complete person union without touching original sealed inputs.
    dev=a.readp(OLD/'identity_development_private.parquet',['subject_id','study_id','split'],test=True)
    test=a.readp(OLD/'identity_test_private.parquet',['subject_id','study_id','split'],test=True)
    assert dev.split.isin(['train','validation']).all() and test.split.eq('locked_test').all()
    assert not set(dev.subject_id)&set(test.subject_id)
    assert not dev.study_id.duplicated().any() and not test.study_id.duplicated().any()
    for x in (dev,test):assert not x.isna().any().any()
    a.save('identity_gate.json',{'passed':True,'overlap_persons':0,'development_persons':int(dev.subject_id.nunique()),'test_persons':int(test.subject_id.nunique()),'inputs':{str(OLD/n):sha(OLD/n) for n in ['identity_development_private.parquet','identity_test_private.parquet']}})
    a.save('aw138_provenance.json',readsmall(OLD/'aw138_provenance.json'))
    names=readsmall(OUT/'marker_universes.json')['catalog'];assert len(names)==2003
    fr,env=a.adapter();env['RUN']=OUT
    a.save('extraction_freeze.json',{'utc':a.utc(),'manifest_v1_sha256':sha(manifest),'inputs':bound,'development_benchmarks_sha256':sha(OUT/'development_benchmarks.json'),'marker_universes_sha256':sha(OUT/'marker_universes.json'),'exclusion':sorted(EXCLUDED),'options':env['options'](names),'release_implementation_hash':m['producer_binding']['release_implementation_hash'],'code':{str(p):sha(p) for p in [Path(__file__),ROOT/'werkzeuge/a_testreplikation/aw163_preflight.py',ROOT/'werkzeuge/a_testreplikation/aw159_inputs.py',ROOT/'werkzeuge/varianten/adapter.py',ROOT/'werkzeuge/transform_katalog/fortsetzung.py']},'packages':m['producer_binding']['packages'],'result_statistics_computed':False})
    public('Input binding completed',f'Extraction freeze `{sha(OUT/"extraction_freeze.json")}`; disjointness passed; 2003 markers')
def extract(kind,stripe):
    import numpy as np,pyarrow.parquet as pq
    freeze=readsmall(OUT/'extraction_freeze.json');freezehash=sha(OUT/'extraction_freeze.json')
    for p,h in freeze['code'].items():assert sha(p)==h
    task=OLD/f'{kind.lower()}_tasks_private.parquet'
    item=next(x for x in freeze['inputs'] if Path(x['path'])==task);assert sha(task)==item['sha256']
    log('begin',type='footer',path=str(task),sha256=item['sha256'],freeze=freezehash)
    names=readsmall(OUT/'marker_universes.json')['catalog']
    tasks=a.readp(task,pq.ParquetFile(task).schema_arrow.names,test=True).sort_values('study_id').iloc[stripe::3]
    fr,env=a.adapter();env['RUN']=OUT;options=env['options'](names)
    assert options==freeze['options']
    checkpoint=OUT/f'{kind}_records_private';checkpoint.mkdir(exist_ok=True)
    done=0
    for row in tasks.to_dict('records'):
        a.guard();dest=checkpoint/f'{int(row["study_id"])}.json'
        if dest.exists():
            saved=readsmall(dest);assert saved['freeze_sha256']==freezehash and saved['task_sha256']==item['sha256']
        else:
            raw=Path(row['waveform_path']).resolve();rawroot=Path(str(data_path("a.mimic_iv_ecg_diagnostic_electrocardiogram_matched_subset_1_0_files"))).resolve()
            assert raw.is_relative_to(rawroot)
            log('begin',type='waveform',path=str(raw),freeze=freezehash)
            rawhash={str(p):sha(p) for p in [raw.with_suffix('.hea'),raw.with_suffix('.dat')]}
            tic=time.perf_counter()
            # Exactly the release status, signal hash, QC and context gates.
            result=fr._extract_record_worker((row,options))
            saved={'freeze_sha256':freezehash,'task_sha256':item['sha256'],'raw_sha256':rawhash,'result':result}
            if kind=='M' and result.get('extraction_status')=='ok':
                try:
                    package=env['prepare_record'](row)
                    saved['kors']=env['compute'](package,row,names,'kors')
                    saved['inverse_dower']=env['compute'](package,row,names,'inverse_dower')
                    saved['same_fiducials']=True
                    del package
                except Exception as e:
                    # Technical failures remain in the fixed population, never replacement.
                    saved['matrix_technical_missing']=type(e).__name__;saved['error_private']=str(e)
            saved['seconds']=time.perf_counter()-tic
            a.save(str(dest.relative_to(OUT)),clean(saved))
            log('end',type='waveform',path=str(raw),freeze=freezehash,checkpoint_sha256=sha(dest),raw_sha256=rawhash)
        done+=1
        a.save(f'progress_{kind}{stripe}.json',{'completed':done,'requested':len(tasks),'utc':a.utc(),'pid':os.getpid()})
        if done%20==0:print(json.dumps({'phase':PHASE,'completed':done,'requested':len(tasks)}),flush=True)
        gc.collect()
def main():
    global PHASE
    PHASE=sys.argv[1] if len(sys.argv)>1 else 'binding'
    a.save(f'status_{PHASE}.json',{'state':'running','pid':os.getpid(),'utc':a.utc()})
    try:
        with Budget():
            a.guard()
            if PHASE=='binding':binding()
            else:extract(PHASE[0],int(PHASE[1]))
        a.save(f'sources_{PHASE}.json',list(a.SOURCES.values()))
        a.save(f'status_{PHASE}.json',{'state':'complete','pid':os.getpid(),'utc':a.utc()})
    except Exception:
        a.save(f'status_{PHASE}.json',{'state':'STOP','pid':os.getpid(),'utc':a.utc(),'traceback_private':traceback.format_exc()});raise
if __name__=='__main__':main()
