import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os, ctypes, json
from pathlib import Path
os.environ['PYTHONDONTWRITEBYTECODE']='1'
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(str(data_path("a.messaudit"))); OUT.mkdir(parents=True,exist_ok=True)
class Mem(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ['total','free','page','commit','vtotal','vfree','extended']]
def memory():
    m=Mem(); m.length=ctypes.sizeof(m); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return {'ram_free':m.free,'commit_free':m.commit}
def main():
    mem=memory(); print(mem)
    if min(mem.values())<4*1024**3: raise SystemExit('Speicherwartepflicht')
    import pyarrow.parquet as pq
    run=Path(str(data_path("a.mimic_validation_runs_a155_2026_10_04_d6e70e")))
    paths=[Path(str(data_path("a.mimic_validation_master_ecg_master_index_parquet"))),Path(str(data_path("a.mimic_validation_labels_endpoint_index_cohorts_parquet")))]
    paths+=list((run/'repeatability/replicate_pairs').rglob('*.parquet'))[:1]
    paths+=list((run/'guideline').glob('*agreement*.parquet'))
    paths+=list((run/'analysis_feature_cache').glob('*/split=train/*.parquet'))
    data={}
    for p in paths:
        f=pq.ParquetFile(p); data[str(p)]={'rows':f.metadata.num_rows,'schema':str(f.schema_arrow),'columns':f.schema_arrow.names}
        print(str(p),f.metadata.num_rows, f.schema_arrow.names[:35])
    (OUT/'inventar.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    print('master',data[str(paths[0])]['schema'])
    print('pairs',[(p,v['schema']) for p,v in data.items() if 'replicate_pairs' in p])
    print('features_selected',[c for v in data.values() if 'spatial_qrs_t_angle_deg' in v['columns'] for c in v['columns'] if any(x in c.lower() for x in ['p_duration','rotation','loop','sinus','normal','rhythm','qrs_duration'])][:150])
if __name__=='__main__':main()
