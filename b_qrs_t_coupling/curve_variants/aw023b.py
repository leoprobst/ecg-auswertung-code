"""AW-023b: attestierter Export, Entwicklungswhitelist, maximal 20 EKGs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os
os.environ.update({k: '1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
import argparse
import json
import time
from pathlib import Path
from .audit import OUT, DEV, ROOT, memory_guard
from .lauf import sha, checkpoint, Budget

MAPPING = OUT / 'export/dev_mapping_20261008T183015Z.parquet'
MANIFEST = MAPPING.with_suffix('.manifest.json')
ROUND = OUT / 'AW023b'

def validate_mapping(frame):
    if frame[['subject_id','study_id','split','relative_waveform_path']].isna().any().any():raise ValueError('Mappingidentität fehlt')
    if not frame.split.isin(['train','validation']).all() or frame.groupby('subject_id').split.nunique().max()>1 or frame.subject_id.duplicated().any():
        raise ValueError('Split-/Indexbarriere verletzt')
    if not frame.label_state.isin(['positive','negative']).all():raise ValueError('Unzulässiger Labelstatus')

def resolve_waveform(relative):
    root=Path(str(data_path("a.mimic_iv_ecg_diagnostic_electrocardiogram_matched_subset_1_0"))).resolve()
    path=(root/str(relative)).resolve()
    if not path.is_relative_to(root/'files'):raise ValueError('Rohpfadbarriere')
    return str(path)

def inspect():
    import pyarrow.parquet as pq
    report = {'memory': memory_guard(), 'mapping_hash': sha(MAPPING), 'manifest_hash': sha(MANIFEST)}
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if report['mapping_hash'] != manifest['sha256'] or manifest['splits'] != ['train','validation']:
        raise ValueError('Mappinghash/Splitattestation abweichend')
    report['mapping_schema'] = pq.ParquetFile(MAPPING).schema_arrow.names
    report['feature_schemas'] = []
    seen = set()
    for fold in ('train','validation'):
        for file in sorted((DEV/f'split={fold}').glob('part-*.parquet')):
            from .vertrag import development_file
            file = development_file(file, fold)
            names = pq.ParquetFile(file).schema_arrow.names
            key = tuple(names)
            if key in seen: continue
            seen.add(key)
            report['feature_schemas'].append({'example_file':str(file), 'relevant_columns':[c for c in names if any(s in c for s in ('machine','rhythm','qrs_duration','qrs_widening','report_af','pacing','extraction_status','subject_id','study_id','split'))]})
    checkpoint(ROUND/'inspection.json',report)
    print(json.dumps(report,indent=2))

def files(fold):
    from .vertrag import development_file
    return [development_file(f,fold) for f in sorted((DEV/f'split={fold}').glob('part-*.parquet'))]

def sample_draw():
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.dataset as ds
    from .vertrag import SEED
    memory_guard()
    manifest=json.loads(MANIFEST.read_text(encoding='utf-8'))
    if sha(MAPPING)!=manifest['sha256']:raise ValueError('Mappinghash abweichend')
    pq.ParquetFile(MAPPING)  # Footer vor Spaltenprojektion
    columns=['subject_id','study_id','ecg_time','split','label_id','label_state','eligibility_state','index_policy','index_selection_strategy','confidence','relative_waveform_path']
    table=ds.dataset(str(MAPPING),format='parquet').to_table(columns=columns,filter=(ds.field('label_id')=='reduced_lvef_strict') & ds.field('label_state').isin(['positive','negative']))
    frame=table.to_pandas();del table
    validate_mapping(frame)
    frame['label']=(frame.label_state=='positive').astype(int)
    feature_sources=[];pieces=[]
    for fold in ('train','validation'):
        wanted=list(map(int,frame.loc[frame.split==fold,'study_id']))
        for j,file in enumerate(files(fold)):
            memory_guard()
            schema=pq.ParquetFile(file).schema_arrow
            cols=['subject_id','study_id','split','qrs_duration_ms','rhythm_context']
            if set(cols)-set(schema.names):raise ValueError('QRS/Rhythmusschema fehlt')
            selected=ds.dataset(str(file),format='parquet').to_table(columns=cols,filter=ds.field('study_id').isin(wanted))
            if selected.num_rows:
                import hashlib
                pieces.append(selected.to_pandas());feature_sources.append({'path':str(file),'projected_rows':selected.num_rows,'projected_sha256':hashlib.sha256(selected.combine_chunks().to_batches()[0].serialize().to_pybytes()).hexdigest(),'file_size':file.stat().st_size})
            if j%200==0:print(json.dumps({'phase':'strata','fold':fold,'files_scanned':j+1}),flush=True)
    strata=pd.concat(pieces,ignore_index=True);del pieces
    if strata.study_id.duplicated().any():raise ValueError('Mehrdeutige Entwicklungsfeatures')
    frame=frame.merge(strata,on=['subject_id','study_id','split'],how='left',validate='one_to_one')
    frame['qrs_stratum']=np.select([frame.qrs_duration_ms<120,frame.qrs_duration_ms>=120],['lt120','ge120'],default='unknown')
    frame['rhythm_stratum']=frame.rhythm_context.fillna('unknown').replace('','unknown')
    import hashlib
    rows=[];quotas=[]
    for fold in ('train','validation'):
        for label,target in ((1,706),(0,2118)):
            pool=frame[(frame.split==fold)&(frame.label==label)]
            groups=list(pool.groupby(['qrs_stratum','rhythm_stratum'],sort=True));n=min(target,len(pool))
            exact=np.array([n*len(g)/len(pool) for _,g in groups]);quota=np.floor(exact).astype(int)
            for j in np.argsort(-(exact-quota),kind='stable')[:n-quota.sum()]:quota[j]+=1
            for (stratum,g),q in zip(groups,quota):
                rank=g.subject_id.map(lambda x:hashlib.sha256(f'{SEED}|{fold}|{label}|{int(x)}'.encode()).hexdigest())
                rows.extend(g.assign(_rank=rank).sort_values('_rank').head(q).to_dict('records'))
                quotas.append({'fold':fold,'label':label,'qrs':stratum[0],'rhythm':stratum[1],'pool':len(g),'selected':int(q)})
    sample=pd.DataFrame(rows).sort_values(['split','label','_rank']).reset_index(drop=True)
    smoke=sample.groupby(['split','label'],sort=True,group_keys=False).head(5)
    if len(smoke)!=20:raise ValueError('20 Rauchtestfälle nicht verfügbar')
    directory=ROUND/'stichprobe';directory.mkdir(parents=True,exist_ok=True)
    for name,data in [('liste',sample),('rauchtest',smoke)]:
        pq.write_table(pa.Table.from_pandas(data[['subject_id','study_id','split','label']],preserve_index=False),directory/f'{name}.parquet')
    # Kontext ausschließlich E:, kein Rohsignal; getrennt von der IDs/Fold/Label-Liste.
    pq.write_table(pa.Table.from_pandas(smoke,preserve_index=False),directory/'rauchtest_kontext.parquet')
    report={'seed':SEED,'mapping_sha256':sha(MAPPING),'manifest_sha256':sha(MANIFEST),'endpoint':'reduced_lvef_strict','pool_counts':frame.groupby(['split','label']).size().reset_index(name='n').to_dict('records'),'quotas':quotas,'n_sample':len(sample),'n_smoke':len(smoke),'list_sha256':sha(directory/'liste.parquet'),'smoke_sha256':sha(directory/'rauchtest.parquet'),'context_sha256':sha(directory/'rauchtest_kontext.parquet'),'feature_sources':feature_sources,'qrs_definition':'qrs_duration_ms aus gebundener Delineation; unknown bleibt eigenes Stratum','rhythm_definition':'rhythm_context aus gebundener Delineation; keine unabhängige AF/Pacing-Adjudikation','index_policies':sorted(frame.index_policy.unique().tolist()),'index_strategies':sorted(frame.index_selection_strategy.unique().tolist()),'reference_time':'nicht exportiert; klinischer Zeitbezug offen','full_run_authorized':False}
    checkpoint(ROUND/'stichprobenmanifest.json',report)
    print(json.dumps({k:report[k] for k in ('seed','pool_counts','n_sample','n_smoke','list_sha256')},indent=2),flush=True)

def smoke_run():
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.dataset as ds
    import numpy as np
    from .adapter import ORIGINALS, OPTIONS, original_record, prepare_record, prepared_originals, equivalent, variants
    from .vertrag import ACTIVE
    memory_guard();start=time.perf_counter()
    frozen=json.loads((ROUND/'stichprobenmanifest.json').read_text(encoding='utf-8'))
    context_file=ROUND/'stichprobe/rauchtest_kontext.parquet'
    if sha(context_file)!=frozen['context_sha256'] or sha(MAPPING)!=frozen['mapping_sha256']:raise ValueError('Freeze abweichend')
    pq.ParquetFile(context_file)
    smoke=pq.read_table(context_file).to_pandas()
    if len(smoke)!=20 or not smoke.split.isin(['train','validation']).all():raise ValueError('Rauchtestbarriere')
    expected={};sources=[]
    for fold in ('train','validation'):
        wanted=list(map(int,smoke.loc[smoke.split==fold,'study_id']))
        wanted_subjects=set(map(int,smoke.loc[smoke.split==fold,'subject_id']))
        for file in files(fold):
            footer=pq.ParquetFile(file);names=footer.schema_arrow.names
            cols=['subject_id','study_id','split','extraction_status',*ORIGINALS,'feature_confidence_json','feature_unevaluable_reason_json','machine_bandwidth']
            if set(cols)-set(names):continue
            subject_index=footer.schema.names.index('subject_id')
            ranges=[footer.metadata.row_group(i).column(subject_index).statistics for i in range(footer.num_row_groups)]
            if ranges and all(s is not None and s.has_min_max and not any(s.min<=p<=s.max for p in wanted_subjects) for s in ranges):continue
            import pyarrow.compute as pc
            identifiers=footer.read(columns=['study_id'])
            mask=pc.is_in(identifiers['study_id'],value_set=pa.array(wanted,type=identifiers['study_id'].type))
            if not pc.any(mask).as_py():continue
            selected=footer.read(columns=cols).filter(mask)
            if selected.num_rows:
                sources.append({'path':str(file),'sha256':sha(file)})
                for ref in selected.to_pylist():
                    if ref['study_id'] in expected:raise ValueError('Doppelte Referenz')
                    expected[ref['study_id']]=ref
    if len(expected)!=20:raise ValueError('Originalreferenzen fehlen')
    report={'status':'running','n_requested':20,'n_original_attempted':0,'n_equivalent':0,'n_variants_completed':0,'original_seconds':[],'variant_seconds':[],'sample_sha256':frozen['list_sha256'],'feature_sources':sources,'options':OPTIONS,'processing_source_sha256':sha(ROOT/'vcg_biomarker/mimic/feature_record.py'),'memory_start':memory_guard(),'full_run_authorized':False,'missing_context':['machine_filtering','machine_report'],'reference_time':'offen; nur technischer Rauchtest'}
    checkpoint(ROUND/'rauchtestbericht.json',report)
    rows=smoke.to_dict('records')
    for row in rows:
        row['waveform_path']=resolve_waveform(row['relative_waveform_path'])
        ref=expected[int(row['study_id'])]
        if ref['split']!=row['split'] or int(ref['subject_id'])!=int(row['subject_id']):raise ValueError('Referenzidentität')
        row['machine_bandwidth']=ref['machine_bandwidth']
    with Budget() as budget:
        try:
            for index,row in enumerate(rows):
                budget.check();tic=time.perf_counter();ref=expected[int(row['study_id'])]
                report['n_original_attempted']+=1
                actual=original_record(row)
                # Individuelle Diagnostik ausschließlich E:, nie Konsolenausgabe.
                checkpoint(ROUND/'diagnostik'/f'original_{index:02d}.json',json.loads(json.dumps({'row':row,'actual':actual,'expected':ref},default=str)))
                if actual['extraction_status']!=ref['extraction_status']:raise RuntimeError('Extraktionsstatus abweichend')
                equivalent(actual,ref)
                if actual['extraction_status']=='ok':
                    package=prepare_record(row);local=prepared_originals(package);equivalent(local,ref);del package
                original_target=ROUND/'werte/originale'/f'record_{index:02d}.parquet';original_target.parent.mkdir(parents=True,exist_ok=True)
                pq.write_table(pa.Table.from_pylist([{'subject_id':int(row['subject_id']),'study_id':int(row['study_id']),'split':row['split'],'marker':marker,'value':actual.get(marker)} for marker in ORIGINALS]),original_target)
                report['n_equivalent']+=1;report['original_seconds'].append(time.perf_counter()-tic)
                checkpoint(ROUND/'rauchtestbericht.json',report)
                print(json.dumps({'phase':'equivalence','completed':report['n_equivalent']}),flush=True)
            for index,row in enumerate(rows):
                budget.check();tic=time.perf_counter()
                if expected[int(row['study_id'])]['extraction_status']!='ok':
                    values={key:{'value':None,'abstention':'Originalextraktion nicht ok'} for key in ACTIVE};qc={};fs=None
                else:
                    package=prepare_record(row);values,qc=variants(package);fs=package['fs'];del package
                if tuple(values)!=ACTIVE:raise ValueError('Aktive Varianten abweichend')
                for key,value in values.items():
                    target=ROUND/'werte'/key/f'record_{index:02d}.parquet';target.parent.mkdir(parents=True,exist_ok=True)
                    pq.write_table(pa.Table.from_pylist([{'subject_id':int(row['subject_id']),'study_id':int(row['study_id']),'split':row['split'],'variant':key,**value,'fs':fs,'qc_json':json.dumps(qc)}]),target)
                report['n_variants_completed']+=1;report['variant_seconds'].append(time.perf_counter()-tic)
                checkpoint(ROUND/'rauchtestbericht.json',report)
                print(json.dumps({'phase':'variants','completed':report['n_variants_completed']}),flush=True)
            report['status']='passed'
        except Exception as exc:
            report['status']='stop_gate';report['error_type']=type(exc).__name__;report['error']=str(exc)
            if len(report['original_seconds'])<report['n_original_attempted']:report['original_seconds'].append(time.perf_counter()-tic)
        report['peak_rss_bytes']=budget.peak;report['seconds_total']=time.perf_counter()-start
        report['measured_output_bytes']=sum(f.stat().st_size for f in (ROUND/'werte').rglob('*.parquet'))
        if report['n_variants_completed']==20:
            mean=float(np.mean(report['variant_seconds']));report['projection_variant_seconds']={str(n):mean*n for n in (4800,frozen['n_sample'])}
            report['projection_value_bytes']={str(n):report['measured_output_bytes']/20*n for n in (4800,frozen['n_sample'])}
        checkpoint(ROUND/'rauchtestbericht.json',report)
        print(json.dumps({k:v for k,v in report.items() if k not in ('feature_sources','options')},indent=2),flush=True)
    return report

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['inspect','draw','smoke']);args=parser.parse_args(argv)
    memory_guard()
    if args.mode=='draw':
        with Budget() as budget:
            started=time.perf_counter();sample_draw()
            checkpoint(ROUND/'stichprobenressourcen.json',{'peak_rss_bytes':budget.peak,'seconds':time.perf_counter()-started})
    elif args.mode=='smoke':
        result=smoke_run()
        if result['status']!='passed':raise SystemExit(2)
    else:inspect()

if __name__ == '__main__':main()
