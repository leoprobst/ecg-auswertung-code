"""AW-023: nur explizite Entwicklungsfooter; keine Patientenzeilen im Bericht."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os
os.environ.update({k: '1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
import sys, json, hashlib, ctypes
from pathlib import Path

ROOT = Path(str(data_path("b.release_sources_a155_survival_nullable_20261007")))
DEV = Path(str(data_path("b.mimic_validation_features_primary_core_sentinel_discovery_reserve_classical_criteria_development")))
OUT = Path(str(data_path("b.varianten")))

class Memory(ctypes.Structure):
    _fields_ = [('length',ctypes.c_ulong),('load',ctypes.c_ulong)] + [(k,ctypes.c_ulonglong) for k in ('totalPhys','availPhys','totalPage','availPage','totalVirtual','availVirtual','availExtended')]

def memory_guard():
    m=Memory(); m.length=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        raise RuntimeError('Speicherstatus nicht messbar')
    if min(m.availPhys,m.availPage) < 4*1024**3:
        raise RuntimeError('Warten: RAM oder freier Commit unter 4 GiB')
    return {'available_physical_bytes':m.availPhys,'available_commit_bytes':m.availPage}

def main():
    report={'memory':memory_guard(),'footers':{},'release':str(ROOT)}
    import pyarrow.parquet as pq
    for fold in ('train','validation'):
        files=sorted((DEV/f'split={fold}').glob('part-*.parquet'))
        schemas={}
        for file in files:
            schema=pq.ParquetFile(file).schema_arrow
            key=hashlib.sha256(str(schema).encode()).hexdigest()
            schemas.setdefault(key,{'columns':schema.names,'files':0})['files']+=1
        report['footers'][fold]={'partition_files':len(files),'schemas':schemas}
    sys.path.insert(0,str(ROOT))
    try:
        import vcg_biomarker.mimic.feature_record as record
        report['import']={'status':'ok','path':record.__file__,'sha256':hashlib.sha256(Path(record.__file__).read_bytes()).hexdigest()}
    except Exception as exc:
        report['import']={'status':'error','type':type(exc).__name__,'message':str(exc)}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'AW023_AUDIT.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'memory':report['memory'],'import':report['import'],'partition_files':{f:v['partition_files'] for f,v in report['footers'].items()}},indent=2))

if __name__=='__main__': main()
