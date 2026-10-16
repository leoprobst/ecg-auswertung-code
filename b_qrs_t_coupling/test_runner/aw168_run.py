"""AW168: neue isolierte Entwicklungsrechnungen, kein Testadapter."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os,sys,time,json,hashlib,ctypes,threading
from pathlib import Path
os.environ.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=Path(str(data_path("b.aw168")));OUT.mkdir(exist_ok=True)
PHASE=sys.argv[1]
sys.stdout=(OUT/(PHASE+'_stdout.log')).open('a',encoding='utf8',buffering=1)
sys.stderr=(OUT/(PHASE+'_stderr.log')).open('a',encoding='utf8',buffering=1)
class Mem(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ['total','free','page','commit','vtotal','vfree','extended']]
def memory():
    m=Mem();m.length=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m));return {'ram_free':m.free,'commit_free':m.commit}
def guard():
    while min(memory().values())<1.5*1024**3:time.sleep(30)
guard()
import numpy as np,pandas as pd,pyarrow.parquet as pq,pyarrow.dataset as ds,psutil
from scipy.special import expit,logit
from scipy.stats import spearmanr
from werkzeuge.b_test.daten import masks,classify_lvef
from werkzeuge.b_test.statistik import raw_design,fixed_design,scaling,separation
from werkzeuge.nachrechnung.core import logistic_fit,auc
from werkzeuge.b_test.aw157_models import icc,rc,multirc
PARAM=ROOT/'werkzeuge/b_test/parameter_entwicklung.json'
A='spatial_qrs_t_angle_deg';F='qrs_t_normalized_discrete_frechet';M='qrs_t_frechet_beat_median'
SEED=1682026;INPUTS={};PEAK=0;PRIVATE=0
def watch():
    global PEAK,PRIVATE
    while True:
        v=psutil.Process().memory_info();PEAK=max(PEAK,v.rss);PRIVATE=max(PRIVATE,v.private)
        if max(PEAK,PRIVATE)>1.5*1024**3:os._exit(75)
        time.sleep(.5)
threading.Thread(target=watch,daemon=True).start()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def clean(v):
    if isinstance(v,dict):return {str(k):clean(a) for k,a in v.items()}
    if isinstance(v,(list,tuple)):return [clean(a) for a in v]
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    return v
def dump(p,v):
    Path(p).write_text(json.dumps(clean(v),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def bind(p):
    p=Path(p);assert 'test' not in p.name.lower() or p.parent.name=='b_test';assert 'sealed' not in p.parts
    INPUTS[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size};return p
def read(p,cols):
    guard();p=bind(p);pf=pq.ParquetFile(p);assert set(cols)<=set(pf.schema_arrow.names)
    flt=ds.field('split').isin(['train','validation']) if 'split' in pf.schema_arrow.names else None
    return ds.dataset(str(p),format='parquet').to_table(columns=cols,filter=flt).to_pandas()
def load(p):return json.loads(bind(p).read_text(encoding='utf8'))
def inventory():
    p=ROOT/'werkzeuge/b_test/aw168_protokoll.md'
    stamp=OUT/'protokoll_hash.json'
    if not stamp.exists():dump(stamp,{'sha256':sha(p),'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'before_new_computations':True})
    assert load(stamp)['sha256']==sha(p)
    e=load(PARAM)['hypotheses'];schemas={}
    for k in ['BEF-0063','BEF-0049']:
        s=Path(e[k]['input_development']['path']);assert sha(s)==e[k]['input_development']['sha256']
        pf=pq.ParquetFile(s);schemas[k]={'path':str(s),'columns':pf.schema_arrow.names,'rows':pf.metadata.num_rows,'bytes':s.stat().st_size}
    dump(OUT/'schemas.json',schemas)
    # Directory inventory only: no patient/test files opened.
    run=Path(str(data_path("b.mimic_validation_runs_a155_2026_10_04_d6e70e")))
    dump(OUT/'release_directories.json',[p.name for p in run.iterdir() if p.is_dir()])
    for k,v in schemas.items():print(k,v['path'],v['rows']);print('\n'.join(v['columns']))
def finish():
    for p in [__file__,ROOT/'werkzeuge/b_test/aw168_protokoll.md',ROOT/'werkzeuge/b_test/daten.py',ROOT/'werkzeuge/b_test/statistik.py',ROOT/'werkzeuge/b_test/aw157_models.py',ROOT/'werkzeuge/nachrechnung/core.py']:
        bind(p)
    for name in ['aw168_analysen.py','aw168_ordinal.py','aw168_nachtrag1.md','aw168_report.py','aw168_verify.py']:
        p=ROOT/'werkzeuge/b_test'/name
        if p.exists():bind(p)
    dump(OUT/(PHASE+'_manifest.json'),{'inputs':INPUTS,'peak_rss':PEAK,'peak_private':PRIVATE,'memory':memory(),'test_access':False,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
if __name__=='__main__':
    try:
        dump(OUT/'status.json',{'phase':PHASE,'state':'running','pid':os.getpid()})
        if PHASE in globals():globals()[PHASE]()
        else:
            import aw168_analysen
            getattr(aw168_analysen,PHASE)(sys.modules[__name__])
        finish();dump(OUT/'status.json',{'phase':PHASE,'state':'complete','pid':os.getpid()});print('COMPLETE',PHASE)
    except Exception:
        dump(OUT/'status.json',{'phase':PHASE,'state':'failed','pid':os.getpid()});raise
