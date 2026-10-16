import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os,sys,time,json,hashlib,threading,atexit
from pathlib import Path
os.environ.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
from inventar import memory,ROOT,OUT
RELEASE=Path(str(data_path("a.release_sources_a155_survival_nullable_20261007")))
RUN=Path(str(data_path("a.mimic_validation_runs_a155_2026_10_04_d6e70e")))
def jsave(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def guard(tag):
    while True:
        m=memory()
        with (OUT/'ressourcen.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'block':tag,**m})+'\n')
        if min(m.values())>=4*1024**3:return m
        print('WAIT_MEMORY',tag,flush=True);time.sleep(30)
guard(Path(sys.argv[0]).name)
SPEC_AT_START=sha(ROOT/'nachrechnung/SPEZIFIKATION_AW047.md')
archive=OUT/'spezifikationen';archive.mkdir(exist_ok=True)
snapshot=archive/(SPEC_AT_START+'.md')
if not snapshot.exists():snapshot.write_bytes((ROOT/'nachrechnung/SPEZIFIKATION_AW047.md').read_bytes())
import numpy as np,pandas as pd,pyarrow.parquet as pq,duckdb,psutil
peak=0
def watch():
    global peak
    p=psutil.Process()
    while True:
        peak=max(peak,p.memory_info().rss)
        if peak>1.5*1024**3:
            jsave(OUT/('speicherstopp_'+Path(sys.argv[0]).stem+'.json'),{'peak_rss':peak});os._exit(75)
        time.sleep(.2)
threading.Thread(target=watch,daemon=True).start()
@atexit.register
def final_resources():
    jsave(OUT/('rss_'+Path(sys.argv[0]).stem+'.json'),{'peak_rss_bytes':peak,'ram_commit_end':memory()})
def conn():
    c=duckdb.connect();c.execute("SET memory_limit='1GB'");c.execute('SET threads=2');c.execute(f"SET temp_directory='{(OUT/'duckdb_tmp').as_posix()}'");return c
def footer(p):return pq.ParquetFile(p).schema_arrow.names
def bind(paths,name):
    jsave(OUT/(name+'_inputs.json'),{'date':'2026-10-09','spec_sha256':SPEC_AT_START,'spec_at_end_sha256':sha(ROOT/'nachrechnung/SPEZIFIKATION_AW047.md'),'files':{str(p):{'sha256':sha(p),'size':Path(p).stat().st_size} for p in paths},'test_access':False,'rss':peak})
def ci(a):
    a=np.asarray(a);v=a[np.isfinite(a)];return np.quantile(v,[.025,.975]).tolist() if len(v)>=950 else [float('nan')]*2
def sqlp(p):return str(p).replace('\\','/').replace("'","''")
