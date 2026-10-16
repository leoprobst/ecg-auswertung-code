"""AW-025: gebundene Vollrechnung, 200er-Commit, Stop bei jedem Äquivalenzfehler."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os
os.environ.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
import argparse, ctypes, json, time, hashlib
from pathlib import Path
from . import audit, adapter, lauf, aw023b, aw023c
from .vertrag import ACTIVE, development_file

ROUND=audit.OUT/'AW025'
PROJECT=Path(__file__).resolve().parents[2]
PLANS=[PROJECT/'verbesserung'/p/'VERBESSERUNG.md' for p in ('VF-01_qrs-t-kopplung','VF-02_zaehlproxy-reparatur')]

def wait_memory():
    while True:
        m=audit.Memory();m.length=ctypes.sizeof(m)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            raise RuntimeError('Speicherstatus nicht messbar')
        status={'available_physical_bytes':m.availPhys,'available_commit_bytes':m.availPage}
        if min(m.availPhys,m.availPage)>=4*1024**3:return status
        print(json.dumps({'phase':'memory_wait',**status}),flush=True)
        time.sleep(60)

def install_guard():
    # Nur dieser Runner erhält den expliziten AW-025-Wartevertrag.
    audit.memory_guard=adapter.memory_guard=lauf.memory_guard=wait_memory

def read_small(path,columns=None):
    import pyarrow.parquet as pq
    schema=pq.ParquetFile(path).schema_arrow
    return pq.read_table(path,columns=columns or schema.names).to_pandas()

def bindings():
    sample=aw023b.ROUND/'stichprobe/liste.parquet'
    context=aw023c.ROUND/'stichprobe_kontext.parquet'
    original=json.loads((aw023c.ROUND/'kontextbericht.json').read_text(encoding='utf-8'))
    if lauf.sha(sample)!=original['sample_sha256'] or lauf.sha(context)!=original['context_sha256']:
        raise ValueError('Stichprobe/Kontext verändert')
    for path in PLANS:
        lines=path.read_text(encoding='utf-8').splitlines(True)
        line=next(x for x in lines if x.startswith('Hash-SHA256-Inhalt:'))
        value=hashlib.sha256(''.join(x for x in lines if not x.startswith('Hash-SHA256-Inhalt:')).encode()).hexdigest()
        if value!=line.split(':',1)[1].strip():raise ValueError('Vergleichsplanhash verletzt')
    return {'sample_sha256':lauf.sha(sample),'context_sha256':lauf.sha(context),
            'plans':{str(p.relative_to(PROJECT)):lauf.sha(p) for p in PLANS},
            'sources':{str(p.relative_to(audit.ROOT)):lauf.sha(p) for p in (audit.ROOT/'vcg_biomarker').rglob('*.py')},
            'labor':{p.name:lauf.sha(p) for p in Path(__file__).parent.glob('*.py') if p.name!='aw025_auswerten.py'},
            'options':json.loads(json.dumps(adapter.OPTIONS)),'active':list(ACTIVE),'block_size':200}

def references(sample):
    """Nur explizite Entwicklungsdateien; JSON-Maps auf vier Marker begrenzen."""
    import duckdb, pyarrow.parquet as pq
    target=ROUND/'referenzen.parquet';manifest=ROUND/'referenzen.json'
    source_manifest=json.loads((aw023b.ROUND/'stichprobenmanifest.json').read_text(encoding='utf-8'))
    if target.exists():
        info=json.loads(manifest.read_text(encoding='utf-8'))
        if info['sha256']!=lauf.sha(target):raise ValueError('Referenzprojektion verändert')
        return read_small(target)
    con=duckdb.connect();con.execute("SET memory_limit='1GB'");con.execute('SET threads=2')
    con.register('wanted',sample[['subject_id','study_id','split']])
    results=[];sources=[]
    for j,item in enumerate(source_manifest['feature_sources']):
        wait_memory();path=Path(item['path']);fold=path.parent.name.split('=')[1];development_file(path,fold)
        names=pq.ParquetFile(path).schema_arrow.names
        columns=['subject_id','study_id','split','extraction_status',*adapter.ORIGINALS,*adapter.COMPARATORS]
        columns += [c for c in names if c in ('qrs_duration_ms','qt_interval_ms','qtc_bazett_ms','qtc_fridericia_ms','heart_rate_bpm','rhythm_context')]
        maps=[]
        for key in ('feature_confidence_json','feature_unevaluable_reason_json'):
            args=','.join("'"+m+"',json_extract("+key+", '$."+m+"')" for m in adapter.ORIGINALS)
            maps.append('json_object('+args+') AS '+key)
        query='SELECT '+','.join(columns+maps)+" FROM read_parquet('"+path.as_posix()+"') WHERE study_id IN (SELECT study_id FROM wanted) AND split='"+fold+"'"
        frame=con.execute(query).fetchdf()
        if len(frame):
            results.append(frame);sources.append({'path':str(path),'sha256':lauf.sha(path),'columns':columns+['feature_confidence_json','feature_unevaluable_reason_json'],'query':query,'rows':len(frame)})
        if j%200==0:print(json.dumps({'phase':'references','files':j+1}),flush=True)
    con.close()
    import pandas as pd, pyarrow as pa
    frame=pd.concat(results,ignore_index=True)
    if len(frame)!=len(sample) or frame.study_id.duplicated().any():raise ValueError('Referenzen fehlen/mehrdeutig')
    if len(frame.merge(sample,on=['subject_id','study_id','split'],validate='one_to_one'))!=len(sample):raise ValueError('Referenzidentität')
    # Nullschlüssel entfernen: Originalmetadaten verwenden fehlende Schlüssel.
    for key in ('feature_confidence_json','feature_unevaluable_reason_json'):
        frame[key]=frame[key].map(lambda x:json.dumps({k:v for k,v in json.loads(x).items() if v is not None}))
    pq.write_table(pa.Table.from_pandas(frame,preserve_index=False),target,compression='zstd')
    lauf.checkpoint(manifest,{'sha256':lauf.sha(target),'sources':sources,'rows':len(frame),'schema':list(frame.columns)})
    return frame

def run():
    install_guard();wait_memory();ROUND.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();binding=bindings()
    sample=read_small(aw023b.ROUND/'stichprobe/liste.parquet')
    if len(sample)!=5648 or sample.subject_id.duplicated().any() or not sample.split.isin(['train','validation']).all():raise ValueError('Stichprobenbarriere')
    context=read_small(aw023c.ROUND/'stichprobe_kontext.parquet')
    rows=sample.merge(context,on=['subject_id','study_id','split'],validate='one_to_one',sort=False)
    if len(rows)!=len(sample):raise ValueError('Kontext fehlt')
    rows=rows.astype(object).where(rows.notna(),None).to_dict('records')
    root=Path(str(data_path("b.mimic_iv_ecg_diagnostic_electrocardiogram_matched_subset_1_0_files"))).resolve()
    if any(not Path(r['waveform_path']).resolve().is_relative_to(root) for r in rows):raise ValueError('Rohpfadbarriere')
    state_path=ROUND/'checkpoint.json'
    state={'binding':binding,'status':'running','completed_blocks':[],'n_equivalent':0,'n_completed':0,'peak_rss_bytes':0,'seconds':0,'qc_abstentions':0}
    if state_path.exists():
        state=json.loads(state_path.read_text(encoding='utf-8'))
        if state['binding']!=binding:raise ValueError('Fortsetzungsbindung verletzt')
        if state['status']=='stop_gate':raise ValueError('Gate-Stopp darf nicht automatisch fortgesetzt werden')
        for block in state['completed_blocks']:
            for key in ('original','variant'):
                if lauf.sha(Path(block[key+'_path']))!=block[key+'_sha256']:raise ValueError('Blockhash verletzt')
    lauf.checkpoint(state_path,state)
    with lauf.Budget() as budget:
        try:
            ref_frame=references(sample);refs={int(r['study_id']):r for r in ref_frame.astype(object).where(ref_frame.notna(),None).to_dict('records')};del ref_frame
            adapter.release_import()
            for offset in range(0,len(rows),200):
                block_id=offset//200
                if block_id<len(state['completed_blocks']):continue
                memory=wait_memory();block_start=time.perf_counter();batch=rows[offset:offset+200]
                originals=[];variants=[];qc_abstentions=0
                # Vollständiges Worker- und Prepare-Gate des Blocks VOR Varianten.
                for row in batch:
                    actual=adapter.original_record(row);ref=refs[int(row['study_id'])]
                    if actual['extraction_status']!=ref['extraction_status']:raise RuntimeError('Extraktionsstatus abweichend')
                    adapter.equivalent(actual,ref)
                    if actual['extraction_status']=='ok':
                        package=adapter.prepare_record(row);adapter.equivalent(adapter.prepared_originals(package),ref);del package
                    else:qc_abstentions+=1
                    originals.extend([{k:row[k] for k in ('subject_id','study_id','split')}|{'marker':m,'value':actual.get(m)} for m in adapter.ORIGINALS])
                for row in batch:
                    if refs[int(row['study_id'])]['extraction_status']!='ok':
                        values={k:{'value':None,'abstention':'Originalextraktion nicht ok'} for k in ACTIVE};qc={};fs=None
                    else:
                        package=adapter.prepare_record(row);values,qc=adapter.variants(package);fs=package['fs'];del package
                    if tuple(values)!=ACTIVE:raise ValueError('Variantenliste verändert')
                    variants.extend([{k:row[k] for k in ('subject_id','study_id','split')}|{'variant':v,**result,'fs':fs,'qc_json':json.dumps(qc)} for v,result in values.items()])
                op=ROUND/'werte/originale'/f'block_{block_id:04d}.parquet';vp=ROUND/'werte/varianten'/f'block_{block_id:04d}.parquet'
                aw023c.write_block(op,originals);aw023c.write_block(vp,variants)
                block={'index':block_id,'n':len(batch),'original_path':str(op),'variant_path':str(vp),'original_sha256':lauf.sha(op),'variant_sha256':lauf.sha(vp),'seconds':time.perf_counter()-block_start,'memory_before':memory,'peak_rss_bytes':budget.peak}
                state['completed_blocks'].append(block);state['n_equivalent']+=len(batch);state['n_completed']+=len(batch);state['qc_abstentions']+=qc_abstentions
                state['peak_rss_bytes']=max(state['peak_rss_bytes'],budget.peak);state['status']='running'
                lauf.checkpoint(state_path,state)
                print(json.dumps({'phase':'block','completed':state['n_completed'],'seconds':block['seconds'],'peak_rss_bytes':budget.peak}),flush=True)
            state['status']='passed'
        except Exception as exc:
            state.update(status='stop_gate',error_type=type(exc).__name__,error=str(exc))
            # Einziger kleiner Fehlerbeleg nur unter E:, keine individuellen Massenlogs.
            if 'row' in locals():lauf.checkpoint(ROUND/'gate_failure.json',{'row':row,'actual':actual,'expected':ref,'error':str(exc)})
            raise
        finally:
            state['seconds']+=time.perf_counter()-started;state['peak_rss_bytes']=max(state['peak_rss_bytes'],budget.peak)
            lauf.checkpoint(state_path,state)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['run']);parser.parse_args();run()

if __name__=='__main__':main()
