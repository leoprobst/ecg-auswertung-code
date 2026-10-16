"""Isolated resource, read and provenance helpers."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os, sys, ctypes, time, json, hashlib, threading
from pathlib import Path
os.environ.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(str(data_path("a.aw149"))); OUT.mkdir(exist_ok=True)
if '--log' in sys.argv:
    sys.stdout=(OUT/(Path(sys.argv[0]).stem+'_stdout.log')).open('a',encoding='utf8',buffering=1)
    sys.stderr=(OUT/(Path(sys.argv[0]).stem+'_stderr.log')).open('a',encoding='utf8',buffering=1)
BASE=Path(str(data_path("a.auswertung"))); MA=BASE/'messaudit'
RUN=Path(str(data_path("a.mimic_validation_runs_a155_2026_10_04_d6e70e")))
class Mem(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ['total','free','page','commit','vtotal','vfree','extended']]
def memory():
    m=Mem(); m.length=ctypes.sizeof(m); assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return {'ram_free':m.free,'commit_free':m.commit}
def guard():
    while min(memory().values())<1.5*1024**3:
        print('WAIT_MEMORY',flush=True);time.sleep(30)
guard()
import numpy as np, pandas as pd, pyarrow.parquet as pq, psutil
PEAK=0
def watch():
    global PEAK
    while True:
        PEAK=max(PEAK,psutil.Process().memory_info().rss)
        if PEAK>1.5*1024**3:os._exit(75)
        time.sleep(.2)
threading.Thread(target=watch,daemon=True).start()
INPUTS={}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def bind(p):
    p=Path(p);assert 'split=test' not in str(p) and 'test' not in p.parts
    assert p.stat().st_size<=20*1024**2 or p.suffix!='.json'
    if str(p) not in INPUTS:INPUTS[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size}
    return p
def read(p,cols):
    guard();p=bind(p);pf=pq.ParquetFile(p);assert set(cols)<=set(pf.schema_arrow.names)
    return pf.read(columns=cols).to_pandas()
def dump(p,v):
    Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str),encoding='utf8')
def finish(name,outputs):
    bind(__file__);bind(Path(sys.argv[0]))
    dump(OUT/(name+'_manifest.json'),{'inputs':INPUTS,'outputs':{str(p):sha(p) for p in outputs},'peak_rss':PEAK,'memory_end':memory(),'test_access':False,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
def ci(a):
    a=np.asarray(a);a=a[np.isfinite(a)]
    return np.quantile(a,[.025,.975]) if len(a)>=950 else [np.nan,np.nan]
