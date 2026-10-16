"""Entwicklungsrunner AW121. Explizite Eingänge; kein rekursives Suchen im Analyselauf."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os
os.environ.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
import argparse, json, hashlib, sys
from pathlib import Path
sys.dont_write_bytecode=True
from werkzeuge.nachrechnung.resources import guard, Budget, available
PROJECT=Path(__file__).resolve().parents[2]
DATA=Path(str(data_path("b.mimic_validation")))
ROOT=Path(str(data_path("b.release_sources_a155_survival_nullable_20261007_vcg_biomarker")))
OUTROOT=Path(str(data_path("b.b_vorbereitung_aw121")))

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def save(p,obj):
    p=Path(p)
    assert p.resolve().is_relative_to(OUTROOT.resolve())
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=str),encoding='utf8')

def inspect(run,out):
    import pyarrow.parquet as pq
    att=json.loads((DATA/'labels/development_label_attestation.json').read_text(encoding='utf8'))
    assert att['scope']=='train_validation_only' and att['status']=='verified'
    paths=[DATA/'evidence/clinical_context_features.parquet',DATA/'labels/endpoint_index_cohorts.parquet',DATA/'labels/clinical_label_evidence.parquet',
           DATA/'labels/clinical_labels_wide.parquet',DATA/'labels/endpoint_window_sensitivity_cohorts.parquet']
    cache=cache_path(run)
    paths += [cache/f'split={s}/features.parquet' for s in ('train','validation')]
    # Explicit existing development projections; no sealed/mixed master footer.
    paths += [Path(str(data_path("b.varianten_aw023c_stichprobe_kontext_parquet"))),Path(str(data_path("b.varianten_aw023b_stichprobe_liste_parquet")))]
    schema={}
    for p in paths:
        available()
        if p.exists():
            f=pq.ParquetFile(p);schema[str(p)]={'columns':f.schema_arrow.names,'rows':f.metadata.num_rows,'rowgroups':f.num_row_groups}
    save(out/'schemas.json',schema)
    print(json.dumps({'phase':'inspect','sources':len(schema),'schema_output':str(out/'schemas.json')},ensure_ascii=False),flush=True)

def cache_path(run):
    base=DATA/'runs'/run/'analysis_feature_cache'
    candidates=list(base.glob('cache-*/cache.manifest.json'))
    if len(candidates)!=1:raise ValueError('Genau ein Entwicklungscache erforderlich')
    manifest=json.loads(candidates[0].read_text(encoding='utf8'))
    assert manifest['locked_test_included'] is False
    assert {x['split'] for x in manifest['splits']}=={'train','validation'}
    return candidates[0].parent

def connect(out):
    import duckdb
    c=duckdb.connect();c.execute("SET memory_limit='256MB'; SET threads=2")
    c.execute(f"SET temp_directory='{(out/'duckdb_tmp').as_posix()}'")
    return c

def feature_projection(run,out):
    import pyarrow.parquet as pq
    from .models import BASE,EXTRA,MARKER,RV
    cache=cache_path(run);files=[];bindings=[]
    native=['lead_measurement_'+l+'_'+s for l in ('I','III','aVL','V1','V2','V3','V5','V6') for s in ('qrs_r_height_mv','qrs_s_depth_mv','qrs_q_duration_ms','qrs_r_peak_time_ms','qrs_terminal_s_depth_mv')]
    wanted=list(dict.fromkeys(['subject_id','study_id','split','extraction_status','qrs_duration_ms','qt_interval_ms','qtc_fridericia_ms','context_heart_rate_bpm',MARKER,RV]+EXTRA+['classical_cornell_voltage','classical_lbbb_pattern_positive','classical_rbbb_pattern_positive','rhythm_context']+native))
    c=connect(out)
    for split in ('train','validation'):
        available();p=cache/f'split={split}/features.parquet';f=pq.ParquetFile(p);cols=[x for x in wanted if x in f.schema_arrow.names]
        destination=out/f'features_{split}.parquet'
        projected=','.join(cols)+("" if 'split' in cols else f",'{split}' AS split")
        c.execute(f"COPY (SELECT {projected} FROM read_parquet('{p.as_posix()}') WHERE split='{split}') TO '{destination.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 4096)")
        with p.open('rb') as stream:
            stream.seek(-8,2);length=int.from_bytes(stream.read(4),'little');stream.seek(-8-length,2);footerhash=hashlib.sha256(stream.read(length+8)).hexdigest()
        bindings.append({'path':str(p),'columns':cols,'filter':f"split='{split}'",'footer_sha256':footerhash,'projection':str(destination),'projection_sha256':sha(destination),'missing_columns':sorted(set(wanted)-set(cols))})
        files.append(destination)
    c.close();save(out/'feature_binding.json',{'run_directory':run,'cache_manifest_sha256':sha(cache/'cache.manifest.json'),'partitions':bindings,'historical_full_cache_hash_checked':False})
    return files

def make_preindex(run,out):
    import pyarrow.parquet as pq
    files=feature_projection(run,out);c=connect(out)
    paths=','.join("'"+p.as_posix()+"'" for p in files)
    c.execute(f"CREATE VIEW features AS SELECT * FROM read_parquet([{paths}],union_by_name=true)")
    p=DATA/'evidence/clinical_context_features.parquet'
    att=json.loads((DATA/'labels/development_label_attestation.json').read_text(encoding='utf8'))
    assert sha(p)==att['clinical_context']['sha256']
    # Shared context: ONLY development rows returned; no master or sealed source.
    pq.ParquetFile(p)
    c.execute(f"CREATE VIEW context AS SELECT subject_id,study_id,split,ecg_time,age,sex,machine_metadata_usable_for_primary,machine_heart_rate_bpm,has_tte_30d,has_tte_7d,nearest_tte_delta_hours,nearest_tte_measurement_id,ehr_time_alignment_confirmatory_eligible FROM read_parquet('{p.as_posix()}') WHERE split IN ('train','validation')")
    q="""SELECT f.*,c.ecg_time,c.age,c.sex,CASE WHEN c.sex='M' THEN 1 WHEN c.sex='F' THEN 0 ELSE NULL END sex_numeric,
      CASE WHEN c.machine_metadata_usable_for_primary THEN c.machine_heart_rate_bpm END heart_rate,c.has_tte_30d,c.has_tte_7d,c.nearest_tte_delta_hours,c.nearest_tte_measurement_id,c.ehr_time_alignment_confirmatory_eligible
      FROM features f JOIN context c USING(subject_id,study_id,split) WHERE f.extraction_status='ok' AND c.age>=18
      QUALIFY row_number() OVER(PARTITION BY subject_id ORDER BY ecg_time,study_id)=1"""
    c.execute(f"COPY ({q}) TO '{(out/'preindex.parquet').as_posix()}' (FORMAT PARQUET, ROW_GROUP_SIZE 4096)")
    flow=c.execute("SELECT split,extraction_status,count(*) n,count(DISTINCT subject_id) persons FROM features GROUP BY ALL").fetchdf().to_dict('records')
    save(out/'index_flow.json',{'all_development_feature_records':flow,'index_policy':'erstes technisch auswertbares Erwachsenen-EKG, ecg_time/study_id; kein Echo-/Outcome-/Markerfilter','preindex_sha256':sha(out/'preindex.parquet')})
    c.close()

def small(p,columns=None):
    import pyarrow.parquet as pq
    f=pq.ParquetFile(p);cols=f.schema_arrow.names if columns is None else columns
    import pandas as pd
    return pd.concat([f.read_row_group(i,columns=cols).to_pandas() for i in range(f.num_row_groups)],ignore_index=True)

def verification(run,out):
    import numpy as np
    from .models import BASE,EXTRA,MARKER,b4,Fit,compare
    from werkzeuge.nachrechnung.core import assert_development,auc
    if (out/'weight_model_frozen.json').exists():
        import shutil
        archive=out/'superseded'/sha(out/'weight_model_frozen.json');archive.mkdir(parents=True,exist_ok=True)
        for name in ('weight_model_frozen.json','weight_model_frozen.sha256.json','verification_result.json','resources_verification.json','preindex.parquet','verification_private.parquet','clinical_binding.json'):
            source=out/name;target=archive/name
            if source.exists() and not target.exists():shutil.copyfile(source,target)
        save(archive/'status.json',{'status':'superseded_technical_mapping_correction','reason':'B4-HF-Vertrag gemäß AW054 zeitgeeignete Geräte-HF; Pipeline-HF nur VB-VF01-08; keine fertige klinische Ergebnisabgabe','old_weight_model_sha256':sha(out/'weight_model_frozen.json')})
    make_preindex(run,out);f=small(out/'preindex.parquet');assert_development(f);assert not f.subject_id.duplicated().any()
    # Release labels.py:_echo_rules setzt COALESCE(has_tte_30d,FALSE).
    # Null bedeutet hier fehlender TTE-Event-Join, keine unbekannte Echo-Diagnose.
    null=f.has_tte_30d.isna()
    assert not (null&f.nearest_tte_measurement_id.notna()).any()
    assert not (null&f.nearest_tte_delta_hours.notna()).any()
    f['verification']=f.has_tte_30d.eq(True).astype(float)
    f['verification7']=f.has_tte_7d.eq(True).astype(float)
    required=BASE+EXTRA+[MARKER,'verification']
    numeric=f[required].apply(__import__('pandas').to_numeric,errors='coerce');good=np.isfinite(numeric).all(axis=1)
    z=f.loc[good].copy();y=z.verification.to_numpy(float);x=b4(z);tr=z.split.eq('train').to_numpy();va=~tr
    fit=Fit().fit(x[tr],y[tr]);propensity=fit.prob(x);raw=1/propensity
    cap=float(np.quantile(raw[tr&(y==1)],.99))
    contract={'scope':'development_only_before_LVEF_calculation','marker_excluded':MARKER,'regressors':BASE+EXTRA,'design':'AW054 natural age spline (40/60/80), axis sin/cos','fit':fit.dump(),'cap_p99_train_verified':cap,
      'preindex_sha256':sha(out/'preindex.parquet'),'code':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
      'spec_sha256':sha(PROJECT/'luecken/bearbeitung/LU-078_spezifikation_B_entwurf_v2.md'),'run_id':run,'created_before_LVEF':True}
    save(out/'weight_model_frozen.json',contract)
    frozen=sha(out/'weight_model_frozen.json');save(out/'weight_model_frozen.sha256.json',{'sha256':frozen})
    z['ipw']=np.minimum(raw,cap);z['propensity']=propensity
    import pyarrow as pa,pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pandas(z,preserve_index=False),out/'verification_private.parquet',row_group_size=4096)
    # Modell mit Marker; Komparator Alter/Geschlecht/QRS, train -> validation.
    comp=np.column_stack([z.age,z.sex_numeric,z.qrs_duration_ms]);cf=Fit().fit(comp[tr],y[tr]);mf=Fit().fit(np.column_stack([x[tr],z[MARKER].to_numpy()[tr]]),y[tr])
    result=compare(z,y,z[MARKER].to_numpy(),replicates=1000)
    result.update({'comparator_age_sex_qrs_auc':auc(y[va],cf.score(comp[va])),'weight_model_sha256':frozen,'p99_cap':cap,
      'indicator':'has_tte_30d ohne Befund; bestehender TTE-Verfügbarkeitsvertrag, nearest_tte_delta_hours audit','all_index_n':len(f),'complete_n':len(z),
      'all_index_by_fold':f.groupby('split').agg(n=('study_id','size'),verified=('verification','sum'),verified7=('verification7','sum')).to_dict('index'),
      'complete_by_fold':z.groupby('split').agg(n=('study_id','size'),verified=('verification','sum')).to_dict('index'),
      'propensity_range':[float(propensity.min()),float(propensity.max())], 'weight_max_before':float(raw.max()),'weight_max_after':float(z.ipw.max()),
      'effective_n_verified':float(z.loc[y==1,'ipw'].sum()**2/(z.loc[y==1,'ipw']**2).sum()),
      'unknown_indicator_n':int(f.verification.isna().sum()),'sparse_no_tte_join_n':int(null.sum()),'verifiable_time_discordance_n':int(((f.has_tte_30d==True)&(f.nearest_tte_delta_hours.abs()>720)).sum()),
      'constraints':'V auf allen Indexpersonen; Fit Complete-case; B4 enthält Amplitude aber keinen Fréchet-Marker; MAR nicht attestiert'})
    for window in (168,720):result['window_'+str(window)+'_boundary_n']=int(f.nearest_tte_delta_hours.abs().eq(window).sum())
    save(out/'verification_result.json',result)
    print(json.dumps({'phase':'verification_completed','all_index':len(f),'complete':len(z),'weight_model_sha256':frozen,'result':{k:result[k] for k in ('delta_auc','or_per_train_sd','comparator_age_sex_qrs_auc')}}),flush=True)

def main():
    a=argparse.ArgumentParser();a.add_argument('--run-id',default='A155-2026-10-04-d6e70e');a.add_argument('--phase',default='all');args=a.parse_args()
    if not args.run_id.startswith('A155-') or Path(args.run_id).name!=args.run_id:raise ValueError('Lauf-ID')
    out=OUTROOT/args.run_id;out.mkdir(parents=True,exist_ok=True)
    assert sha(PROJECT/'luecken/bearbeitung/LU-078_spezifikation_B_entwurf_v2.md')=='74c3314fbe93fa62d428cc3b252b9fdc6c59f83535000db3e20c724520f77044'
    mem=guard()
    with Budget(out) as budget:
        if args.phase=='inspect':inspect(args.run_id,out)
        elif args.phase=='verification':verification(args.run_id,out)
        elif args.phase=='clinical':
            from .clinical import run
            run(out)
        elif args.phase=='lvh':
            from .lvh import run
            run(out)
        elif args.phase=='signals':
            from .signals import run
            run(out)
        elif args.phase=='ipw':
            from .clinical import ipw
            ipw(out)
        elif args.phase=='diagnose':
            from .diagnose import run
            run(out)
        elif args.phase=='audit':
            from .audit import run
            run(out)
        elif args.phase=='report':
            from .report import run
            run(out,args.run_id)
        elif args.phase=='all':
            from .kernels import synthetic
            from .lvh import synthetic as lvh_synthetic,run as lvh_run
            from .clinical import run as clinical_run,ipw
            from .signals import run as signals_run
            from .audit import run as audit_run
            from .report import run as report_run
            import gc
            result=synthetic();result['LVH_selection']=lvh_synthetic();save(out/'synthetic.json',result)
            for name,fn in [('verification',lambda:verification(args.run_id,out)),('clinical',lambda:clinical_run(out)),('ipw',lambda:ipw(out)),('lvh',lambda:lvh_run(out)),('signals',lambda:signals_run(out)),('audit',lambda:audit_run(out)),('report',lambda:report_run(out,args.run_id))]:
                phase_mem=available();fn();gc.collect();save(out/('resources_'+name+'.json'),{'start':phase_mem,**budget.summary()})
        elif args.phase=='synthetic':
            from .kernels import synthetic
            from .lvh import synthetic as lvh_synthetic
            result=synthetic();result['LVH_selection']=lvh_synthetic();save(out/'synthetic.json',result);print(json.dumps(result),flush=True)
        else:raise ValueError('Phase')
    save(out/('resources_'+args.phase+'.json'),{'start':mem,**budget.summary()})

if __name__=='__main__':main()
