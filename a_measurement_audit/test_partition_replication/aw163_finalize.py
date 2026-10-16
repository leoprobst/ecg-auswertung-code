"""W/M checkpoint validation and column-readable test inputs; no gates."""
import sys,json,gc
from pathlib import Path
from . import aw163_extract as e
from .aw163_preflight import ROOT,OUT,sha
OLD=e.OLD
def finalize(kind):
    import numpy as np,pyarrow as pa,pyarrow.parquet as pq
    for i in range(3):assert e.readsmall(OUT/f'status_{kind}{i}.json')['state']=='complete'
    freeze=e.readsmall(OUT/'extraction_freeze.json');fh=sha(OUT/'extraction_freeze.json');ch=sha(OUT/'continuation_binding_v2.json')
    item=next(x for x in freeze['inputs'] if Path(x['path'])==OLD/f'{kind.lower()}_tasks_private.parquet')
    task=e.a.readp(item['path'],['subject_id','study_id','split'],test=True).sort_values('study_id')
    names=e.readsmall(OUT/'marker_universes.json')['catalog']
    meta=['signal_quality_score','qrs_duration_ms','context_heart_rate_bpm','p_duration_ms']
    fields=[pa.field('subject_id',pa.int64()),pa.field('study_id',pa.int64()),pa.field('split',pa.string()),pa.field('extraction_status',pa.string()),pa.field('signal_sha256',pa.string()),pa.field('technical_matrix_status',pa.string())]
    numeric=list(dict.fromkeys(meta+names)) if kind=='W' else [n+s for n in names for s in ('_kors','_dower')]
    schema=pa.schema(fields+[pa.field(n,pa.float64()) for n in numeric])
    path=OUT/f'{kind}_features_private.parquet';assert not path.exists()
    bound=[];batch=[];statuses={};matrix={}
    with pq.ParquetWriter(path,schema,compression='zstd') as writer:
        for ident in task.to_dict('records'):
            e.a.guard();p=OUT/f'{kind}_records_private'/f'{ident["study_id"]}.json';r=e.readsmall(p)
            assert r['freeze_sha256']==fh and r['continuation_binding_sha256']==ch and r['task_sha256']==item['sha256']
            a=r['result'];assert all(a[k]==v for k,v in ident.items())
            status=a.get('extraction_status');statuses[status]=statuses.get(status,0)+1
            record={**ident,'extraction_status':status,'signal_sha256':a.get('signal_sha256'),'technical_matrix_status':r.get('matrix_technical_missing','ok' if r.get('same_fiducials') else 'extraction_abstention')}
            if kind=='W':record.update({n:a.get(n) for n in numeric})
            else:
                record.update({n+s:r.get(m,{}).get(n) for n in names for s,m in [('_kors','kors'),('_dower','inverse_dower')]})
                ms=record['technical_matrix_status'];matrix[ms]=matrix.get(ms,0)+1
            # Registry flags/classical positives are numeric markers; preserve 0/1
            # while supplying the explicitly bound common float schema to Arrow.
            for n in numeric:record[n]=None if record.get(n) is None else float(record[n])
            batch.append(record);bound.append({'path':str(p),'sha256':sha(p),'raw_sha256':r['raw_sha256']})
            if len(batch)==64:
                writer.write_table(pa.Table.from_pylist(batch,schema=schema),row_group_size=64);batch=[];gc.collect()
        if batch:writer.write_table(pa.Table.from_pylist(batch,schema=schema),row_group_size=64)
    sensitivity=[]
    if kind=='M':
        for study in task.study_id:
            p=OUT/'M_sensitivity_private'/f'{int(study)}.json'
            if p.exists():
                r=e.readsmall(p);assert r['freeze_sha256']==fh and r['continuation_binding_sha256']==ch
                sensitivity.append({'path':str(p),'sha256':sha(p)})
    e.a.save(f'{kind}_input_binding.json',{'task_sha256':item['sha256'],'freeze_sha256':fh,'continuation_binding_sha256':ch,'output':{'path':str(path),'sha256':sha(path),'rows':len(task)},'statuses':statuses,'matrix_technical_statuses':matrix,'checkpoints':bound,'sensitivity_checkpoints':sensitivity,'generator_sha256':sha(__file__)})
    e.public(f'{kind} extraction end',f'Input binding `{sha(OUT/f"{kind}_input_binding.json")}`; selection n={len(task)}; no replication gates evaluated')
def main():
    e.PHASE=sys.argv[1];kind=e.PHASE[0]
    e.a.save(f'status_{e.PHASE}.json',{'state':'running','pid':__import__('os').getpid()})
    with e.Budget():finalize(kind)
    e.a.save(f'status_{e.PHASE}.json',{'state':'complete'})
if __name__=='__main__':main()
