import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os,sys,time,json,hashlib,ctypes,threading,atexit
from pathlib import Path
os.environ.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];OUT=Path(str(data_path("a.rv_a1")));OUT.mkdir(exist_ok=True)
RUN=Path(str(data_path("a.mimic_validation_runs_a155_2026_10_04_d6e70e")))
VAR=Path(str(data_path("a.transform_katalog_fortsetzung_v3")));MA=Path(str(data_path("a.messaudit")))
SPEC=ROOT/'nachrechnung/SPEZIFIKATION_AW077.md';SEED=77010;B=1000
class Mem(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ['total','free','page','commit','vtotal','vfree','extended']]
def memory():
    m=Mem();m.length=ctypes.sizeof(m);ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m));return {'ram_free':m.free,'commit_free':m.commit}
RESOURCE_LOG_LOCK=threading.Lock()
def guard(tag):
    while True:
        m=memory()
        with RESOURCE_LOG_LOCK:
            with (OUT/'resources.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps({'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'block':tag,**m})+'\n')
        if min(m.values())>=4*1024**3:return
        print('WAIT_MEMORY',tag,flush=True);time.sleep(30)
guard(Path(sys.argv[0]).stem)
if (OUT/'spec_lock.json').exists():
    assert hashlib.sha256(SPEC.read_bytes()).hexdigest()==json.loads((OUT/'spec_lock.json').read_text())['sha256'], 'specification binding changed'
import numpy as np,pandas as pd,pyarrow.parquet as pq,psutil
peak=0
def watch():
    global peak
    p=psutil.Process()
    while True:
        peak=max(peak,p.memory_info().rss)
        if peak>1.5*1024**3:os._exit(75)
        time.sleep(.2)
threading.Thread(target=watch,daemon=True).start()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str),encoding='utf8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()
def read(p,cols):
    p=Path(p);assert 'test' not in p.parts and 'split=test' not in str(p)
    pq.ParquetFile(p);return pq.read_table(p,columns=cols).to_pandas()
def ci(v):
    v=np.asarray(v);v=v[np.isfinite(v)];return np.quantile(v,[.025,.975]).tolist() if len(v)>=.95*B else [float('nan')]*2
def bind(name,inputs,outputs):
    dump(OUT/(name+'_manifest.json'),{'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'spec_sha256':sha(SPEC),'inputs':{str(p):{'sha256':sha(p),'bytes':Path(p).stat().st_size} for p in inputs},'outputs':{str(p):{'sha256':sha(p),'bytes':Path(p).stat().st_size} for p in outputs},'script':str(Path(sys.argv[0])),'script_sha256':sha(sys.argv[0]),'peak_rss':peak,'test_access':False})
@atexit.register
def finish():dump(OUT/(Path(sys.argv[0]).stem+'_resources.json'),{'peak_rss':peak,'end':memory()})
