"""AW-023c: ID-gefilterter Pipelinekontext, unverändertes Gate, Blockrauchtest."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os
os.environ.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
import argparse, json, time, hashlib
from pathlib import Path
from .audit import OUT, ROOT, memory_guard
from .lauf import sha, checkpoint, Budget
from . import aw023b as prior

ROUND=OUT/'AW023c'
MASTER=Path(str(data_path("b.mimic_validation_master_ecg_master_index_parquet")))
IDENTITY=['subject_id','study_id','split','ecg_time','waveform_path']
GATED=['machine_qrs_axis_deg','machine_t_axis_deg','machine_qrs_duration_ms','machine_qt_interval_ms','machine_pr_interval_ms','machine_report']
CONTEXT=['age','sex',*GATED,'machine_cart_id','machine_bandwidth','machine_filtering','machine_time_mismatch','machine_time_mismatch_seconds','machine_metadata_temporal_status','machine_metadata_usable_for_primary']

def context_sql(path):
    # Genau _load_feature_cohort; weder Labels noch Ergebnis-/Testspalten.
    selections=IDENTITY+[f'CASE WHEN machine_metadata_usable_for_primary THEN {c} ELSE NULL END AS {c}' if c in GATED else c for c in CONTEXT]
    return "SELECT "+', '.join(selections)+" FROM read_parquet('"+path.as_posix()+"') WHERE study_id IN (SELECT study_id FROM wanted) AND split IN ('train','validation')"

def collect():
    import pyarrow as pa, pyarrow.parquet as pq
    import duckdb
    memory_guard()
    freeze=json.loads((prior.ROUND/'stichprobenmanifest.json').read_text(encoding='utf-8'))
    sample=prior.ROUND/'stichprobe/liste.parquet'
    if sha(sample)!=freeze['list_sha256']:raise ValueError('Freeze geändert')
    pq.ParquetFile(sample);wanted=pq.read_table(sample,columns=['subject_id','study_id','split'])
    schema=pq.ParquetFile(MASTER).schema_arrow
    if set(IDENTITY+CONTEXT)-set(schema.names):raise ValueError('Worker-Kontextschema unvollständig')
    con=duckdb.connect();con.execute("SET memory_limit='1GB'");con.execute('SET threads=2');con.register('wanted',wanted)
    query=context_sql(MASTER)
    table=con.execute(query).fetch_arrow_table();con.close()
    frame=table.to_pandas();expected=wanted.to_pandas()
    if len(frame)!=len(expected) or frame.study_id.duplicated().any():raise ValueError('Kontext fehlt/mehrdeutig')
    if len(frame.merge(expected,on=['subject_id','study_id','split'],validate='one_to_one'))!=len(expected):raise ValueError('Kontextidentität')
    ROUND.mkdir(parents=True,exist_ok=True);target=ROUND/'stichprobe_kontext.parquet';pq.write_table(table,target)
    report={'path':str(MASTER),'columns':IDENTITY+CONTEXT,'filter':'study_id IN eingefrorene AW023b/liste.parquet; split IN (train,validation)','query':query,'rows':len(frame),'sample_sha256':sha(sample),'projection_sha256':hashlib.sha256(table.combine_chunks().serialize().to_pybytes()).hexdigest() if hasattr(table,'serialize') else sha(target),'context_sha256':sha(target),'missing_before':[c for c in CONTEXT if c!='machine_bandwidth'],'source_size':MASTER.stat().st_size,'source_mtime_ns':MASTER.stat().st_mtime_ns,'full_run_authorized':False}
    checkpoint(ROUND/'kontextbericht.json',report)
    print(json.dumps({'phase':'context','rows':len(frame),'context_sha256':sha(target)}),flush=True)

def write_block(path,rows):
    import pyarrow as pa, pyarrow.parquet as pq
    path.parent.mkdir(parents=True,exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows),path,compression='zstd')

def smoke():
    import pyarrow.parquet as pq, pyarrow.dataset as ds
    import numpy as np
    from .adapter import ORIGINALS,OPTIONS,release_import,original_record,prepare_record,prepared_originals,equivalent,variants
    from .vertrag import ACTIVE,development_file
    memory_guard();start=time.perf_counter()
    binding=json.loads((ROUND/'kontextbericht.json').read_text(encoding='utf-8'))
    source=ROUND/'stichprobe_kontext.parquet'
    if sha(source)!=binding['context_sha256'] or sha(prior.ROUND/'stichprobe/liste.parquet')!=binding['sample_sha256']:raise ValueError('Kontext-/Stichprobenfreeze')
    smoke_file=prior.ROUND/'stichprobe/rauchtest.parquet';pq.ParquetFile(smoke_file)
    ids=pq.read_table(smoke_file,columns=['subject_id','study_id','split']).to_pylist()
    if len(ids)!=20 or any(r['split'] not in ('train','validation') for r in ids):raise ValueError('20er-Barriere')
    wanted=[r['study_id'] for r in ids];pq.ParquetFile(source)
    context=ds.dataset(str(source),format='parquet').to_table(columns=IDENTITY+CONTEXT,filter=ds.field('study_id').isin(wanted)).to_pylist()
    by_id={r['study_id']:r for r in context};rows=[by_id[r['study_id']] for r in ids]
    for a,b in zip(rows,ids):
        if any(a[k]!=b[k] for k in ('subject_id','study_id','split')):raise ValueError('Rauchtestidentität')
        # Vor Rohzugriff Pfad gegen bereits attestiertes Mapping prüfen.
        oldfile=prior.ROUND/'stichprobe/rauchtest_kontext.parquet';pq.ParquetFile(oldfile)
        old=ds.dataset(str(oldfile),format='parquet').to_table(columns=['study_id','relative_waveform_path'],filter=ds.field('study_id')==a['study_id']).to_pylist()[0]
        if Path(a['waveform_path']).resolve()!=Path(prior.resolve_waveform(old['relative_waveform_path'])).resolve():raise ValueError('Rohpfad abweichend')
    old=json.loads((prior.ROUND/'rauchtestbericht.json').read_text(encoding='utf-8'))
    expected={};sources=[]
    cols=['subject_id','study_id','split','extraction_status',*ORIGINALS,'feature_confidence_json','feature_unevaluable_reason_json']
    for item in old['feature_sources']:
        path=Path(item['path']);fold=path.parent.name.split('=')[1];development_file(path,fold)
        if sha(path)!=item['sha256']:raise ValueError('Referenzhash geändert')
        pq.ParquetFile(path)
        table=ds.dataset(str(path),format='parquet').to_table(columns=cols,filter=ds.field('study_id').isin(wanted))
        for r in table.to_pylist():
            if r['study_id'] in expected:raise ValueError('Doppelte Referenz')
            expected[r['study_id']]=r
        sources.append({**item,'columns':cols,'filter':'study_id IN eingefrorene 20 Rauchtest-IDs','selected_rows':table.num_rows})
    if len(expected)!=20:raise ValueError('Referenzen fehlen')
    fr=release_import()
    report={'status':'running','n_requested':20,'n_original_attempted':0,'n_equivalent':0,'n_variants_completed':0,'original_seconds':[],'variant_seconds':[],'options':OPTIONS,'feature_sources':sources,'sample_sha256':binding['sample_sha256'],'smoke_sha256':sha(smoke_file),'context_sha256':sha(source),'memory_start':memory_guard(),'code_sha256':{n:sha(ROOT/'vcg_biomarker/mimic'/n) for n in ('feature_record.py','feature_extraction.py','signals.py')},'full_run_authorized':False,'tolerance':{'rtol':1e-8,'atol':0,'missingness':'exact','metadata':'exact'}}
    # Eine identisch reproduzierte Pipeline-QC-Abstention ist kein Gatefehler.
    # Bereits bestandene Worker-/Prepare-Gates nur bei identischer Bindung übernehmen.
    cached=None
    prior_report=ROUND/'rauchtestbericht.json'
    if prior_report.exists():
        candidate=json.loads(prior_report.read_text(encoding='utf-8'))
        if candidate.get('error')=='Original nicht vollständig extrahierbar' and all(candidate.get(k)==json.loads(json.dumps(report[k])) for k in ('sample_sha256','smoke_sha256','context_sha256','code_sha256','options','tolerance')):
            cached=candidate
            checkpoint(ROUND/'vorlauf_qc_abstention.json',candidate)
    report['cached_worker_records']=0;report['qc_abstentions']=0
    originals=[];variant_rows=[]
    with Budget() as budget:
        try:
            for index,row in enumerate(rows):
                budget.check();tic=time.perf_counter();report['n_original_attempted']+=1
                ref=expected[row['study_id']]
                if any(row[k]!=ref[k] for k in ('subject_id','study_id','split')):raise ValueError('Referenzidentität')
                reused=cached is not None and index<cached['n_original_attempted']
                if reused:
                    diagnostic=json.loads((ROUND/'diagnostik'/f'original_{index:02d}.json').read_text(encoding='utf-8'))
                    if diagnostic['row']!=json.loads(json.dumps(row,default=str)) or json.dumps(diagnostic['expected'],sort_keys=True)!=json.dumps(ref,sort_keys=True):raise ValueError('Cachebindung abweichend')
                    actual=diagnostic['actual'];report['cached_worker_records']+=1
                else:actual=original_record(row)
                checkpoint(ROUND/'diagnostik'/f'original_{index:02d}.json',json.loads(json.dumps({'row':row,'actual':actual,'expected':ref},default=str)))
                if actual['extraction_status']!=ref['extraction_status']:raise RuntimeError('Extraktionsstatus abweichend')
                equivalent(actual,ref)
                if actual['extraction_status']=='ok':
                    if not (reused and index<cached['n_equivalent']):
                        package=prepare_record(row);local=prepared_originals(package);equivalent(local,ref);del package
                else:report['qc_abstentions']+=1
                originals.extend([{**{k:row[k] for k in ('subject_id','study_id','split')},'marker':m,'value':actual.get(m)} for m in ORIGINALS])
                report['n_equivalent']+=1;report['original_seconds'].append(cached['original_seconds'][index] if reused else time.perf_counter()-tic)
                checkpoint(ROUND/'rauchtestbericht.json',report);print(json.dumps({'phase':'equivalence','completed':report['n_equivalent']}),flush=True)
            write_block(ROUND/'werte/originale/block_0000.parquet',originals)
            for row in rows:
                budget.check();tic=time.perf_counter()
                if expected[row['study_id']]['extraction_status']!='ok':
                    values={key:{'value':None,'abstention':'Originalextraktion nicht ok'} for key in ACTIVE};qc={'original_extraction_status':expected[row['study_id']]['extraction_status']};fs=None
                else:
                    package=prepare_record(row);values,qc=variants(package);fs=package['fs'];del package
                if tuple(values)!=ACTIVE:raise ValueError('Aktive Varianten geändert')
                variant_rows.extend([{**{k:row[k] for k in ('subject_id','study_id','split')},'variant':key,**v,'fs':fs,'qc_json':json.dumps(qc)} for key,v in values.items()])
                report['n_variants_completed']+=1;report['variant_seconds'].append(time.perf_counter()-tic)
                checkpoint(ROUND/'rauchtestbericht.json',report);print(json.dumps({'phase':'variants','completed':report['n_variants_completed']}),flush=True)
            write_block(ROUND/'werte/varianten/block_0000.parquet',variant_rows);report['status']='passed'
        except Exception as exc:
            report.update(status='stop_gate',error_type=type(exc).__name__,error=str(exc))
            if len(report['original_seconds'])<report['n_original_attempted']:report['original_seconds'].append(time.perf_counter()-tic)
        report['peak_rss_bytes']=budget.peak;report['seconds_total']=time.perf_counter()-start
        report['output_bytes']=sum(f.stat().st_size for f in (ROUND/'werte').rglob('*.parquet'))
        if report['status']=='passed':
            report['timing_summary']={phase:{'mean':float(np.mean(report[phase+'_seconds'])),'median':float(np.median(report[phase+'_seconds'])),'p95':float(np.percentile(report[phase+'_seconds'],95))} for phase in ('original','variant')}
            per=sum(report['timing_summary'][p]['mean'] for p in ('original','variant'))
            report['projection']={str(n):{'original_plus_variant_seconds':per*n,'bytes_linear':report['output_bytes']/20*n} for n in (4800,5648)}
        checkpoint(ROUND/'rauchtestbericht.json',report)
    print(json.dumps({k:report[k] for k in ('status','n_original_attempted','n_equivalent','n_variants_completed','peak_rss_bytes','seconds_total','output_bytes')}),flush=True)
    return report

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['context','smoke']);args=parser.parse_args();memory_guard()
    if args.mode=='context':
        with Budget() as budget:
            collect();checkpoint(ROUND/'kontextressourcen.json',{'peak_rss_bytes':budget.peak})
    elif smoke()['status']!='passed':raise SystemExit(2)

if __name__=='__main__':main()
