"""AW137: technische Testprojektion; keine Assoziations- oder Fitimporte."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import hashlib
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

PROJECT=Path(__file__).resolve().parents[2]
OUT=Path(str(data_path("b.aw137")))
ROOT=Path(str(data_path("b.mimic_validation")))
RUN=ROOT/'runs/A155-2026-10-04-d6e70e'
HOLD=RUN/'publication_preparation/research-holdout-20261011-v2'
RELEASE=Path(str(data_path("b.release_sources_a155_survival_nullable_20261007_vcg_biomarker")))
HYPOTHESES=[('BEF-0063','reduced_lvef_strict','qrs_t_normalized_discrete_frechet','qrs_duration_ms'),
 ('BEF-0042','acute_decompensated_heart_failure_proxy','qrs_t_loop_mirror_symmetry','qrs_duration_ms'),
 ('BEF-0081','elevated_ntprobnp','qrs_t_frechet_beat_median','qrs_duration_ms'),
 ('BEF-0049','cardiomyopathy_icd','qrs_t_frechet_beat_median','qrs_duration_ms'),
 ('BEF-0075','troponin_positive_any','qrs_t_normalized_dtw_distance','classical_st_elevation_contiguous_mm'),
 ('BEF-0087','echo_rv_systolic_dysfunction','qrs_t_normalized_dtw_distance','classical_dominant_r_v1')]
B3=['qrs_duration_ms','qtc_fridericia_ms','spatial_qrs_t_angle_deg']
B4=['classical_cornell_product','classical_sokolow_lyon','magnitude_area_qrs',
    'spatial_low_voltage_qrs_peak_magnitude','classical_right_axis_deviation']
EXTRA=['qt_interval_ms','classical_lbbb_pattern_positive','classical_rbbb_pattern_positive']
SOURCES={}

def read_json(path):
    p=permitted(path)
    if p.stat().st_size>20*1024**2:raise ValueError('JSON zu groß')
    record(p,[],None,'json_provenienz')
    return json.loads(p.read_text(encoding='utf8'))

def record(path,columns,filter_text,action='projection'):
    p=permitted(path)
    if str(p) not in SOURCES:SOURCES[str(p)]={'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
    with (OUT/'access_log.jsonl').open('a',encoding='utf8') as f:
        f.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'path':str(p),
          'sha256':SOURCES[str(p)]['sha256'],'columns':columns,'filter':filter_text,'action':action})+'\n')

def projected(path,columns,filter_expr=None,filter_text=None):
    import pyarrow.dataset as ds
    from werkzeuge.nachrechnung.resources import guard
    guard();names=footer(path)['columns'];cols=[c for c in columns if c in names]
    record(path,cols,filter_text)
    return ds.dataset(str(path),format='parquet').to_table(columns=cols,filter=filter_expr).to_pandas()

def literal(path):return "'"+str(Path(path)).replace('\\','/').replace("'","''")+"'"

def connection():
    import duckdb
    c=duckdb.connect();c.execute("SET memory_limit='1GB'");c.execute('SET threads=2')
    c.execute('SET temp_directory='+literal(OUT/'duckdb_tmp'))
    return c

def provenance():
    sys.path.insert(0,str(RELEASE.parent))
    from vcg_biomarker.mimic import labels
    from vcg_biomarker.mimic.feature_extraction import feature_implementation_hash
    protocol=read_json(HOLD/'frozen/protocol.json')
    dev=read_json(ROOT/'labels/development_label_attestation.json')
    hashes={'label_implementation_hash':labels.label_implementation_hash(),
        'development_label_implementation_hash':dev['label_implementation_hash'],
        'feature_implementation_hash_release':feature_implementation_hash(),
        'feature_implementation_hash_holdout':protocol['feature_code_hash'],
        'label_stage_version':labels.LABEL_STAGE_VERSION,'label_rule_version':labels.LABEL_RULE_VERSION,
        'protocol_hash':protocol['protocol_hash'],'runtime':protocol['runtime']}
    for name in ('mimic/labels.py','mimic/label_registry.py','mimic/feature_record.py','mimic/signals.py',
                 'mimic/classical.py','core/biomarkers/discovery_v3_biomarkers.py'):
        record(RELEASE/name,[],None,'code_hash')
        other=Path(protocol['runtime'])/'vcg_biomarker'/name
        try:
            if other.exists():
                record(other,[],None,'code_hash')
                hashes[name]={'release':sha(RELEASE/name),'runtime':sha(other),'equal':sha(RELEASE/name)==sha(other)}
        except PermissionError:
            hashes[name]={'release':sha(RELEASE/name),'runtime_access':'Zugriff verweigert; keine Umgehung'}
    record(HOLD/'executed_source/audit_tools/a155_research_holdout.py',[],None,'code_hash')
    save('provenance.json',hashes);print(json.dumps(hashes),flush=True)
    return labels,protocol,dev

def labels_build():
    from werkzeuge.nachrechnung.resources import guard
    labels,protocol,dev=provenance()
    wanted=[x[1] for x in HYPOTHESES]+['echo_lvh_strict']
    definitions=tuple(x for x in labels.default_label_definitions() if x.label_id in wanted)
    rules={k:v for k,v in labels._label_rules().items() if k in wanted}
    query=labels._long_label_query(ROOT/'evidence/clinical_context_features.parquet',definitions,rules,split_values=('locked_test',))
    (OUT/'label_regeneration.sql').write_text(query,encoding='utf8')
    c=connection();target=OUT/'labels_reproduced_private.parquet'
    record(ROOT/'evidence/clinical_context_features.parquet',['release SQL references; see label_regeneration.sql'],"split='locked_test'",'sql_reference_projection')
    guard();c.execute(f'COPY ({query}) TO {literal(target)} (FORMAT PARQUET, ROW_GROUP_SIZE 32768)')
    record(target,[],None,'generated_labels');names=footer(target)['columns']
    sealed=ROOT/'labels/sealed/locked_clinical_label_evidence.parquet';footer(sealed)
    record(sealed,names,'label_id IN six B endpoints + echo_lvh_strict')
    c.execute(f'CREATE VIEW regenerated AS SELECT * FROM read_parquet({literal(target)})')
    in_list=','.join("'"+e+"'" for e in wanted)
    c.execute(f'CREATE VIEW sealed AS SELECT {",".join(names)} FROM read_parquet({literal(sealed)}) WHERE label_id IN ({in_list})')
    compare=[n for n in names if n not in ('subject_id','study_id','label_id')]
    mismatch=' OR '.join(f'a.{n} IS DISTINCT FROM b.{n}' for n in compare)
    audit=c.execute(f'''SELECT COALESCE(a.label_id,b.label_id) endpoint, count(*) compared,
       count(*) FILTER(WHERE a.subject_id IS NULL) absent_regenerated,
       count(*) FILTER(WHERE b.subject_id IS NULL) absent_sealed,
       count(*) FILTER(WHERE a.subject_id IS NOT NULL AND b.subject_id IS NOT NULL AND ({mismatch})) unequal
       FROM regenerated a FULL OUTER JOIN sealed b USING(subject_id,study_id,label_id) GROUP BY 1''').fetchdf().to_dict('records')
    native=OUT/'native_index_reproduced_private.parquet'
    iq=labels._index_cohort_query(target);(OUT/'index_regeneration.sql').write_text(iq,encoding='utf8')
    guard();c.execute(f'COPY ({iq}) TO {literal(native)} (FORMAT PARQUET)')
    original=ROOT/'labels/sealed/locked_endpoint_index_cohorts.parquet';orig_names=footer(original)['columns']
    record(original,orig_names,'label_id IN six B endpoints + echo_lvh_strict')
    neq=' OR '.join(f'a.{n} IS DISTINCT FROM b.{n}' for n in orig_names if n not in ('subject_id','label_id'))
    index_audit=c.execute(f'''SELECT COALESCE(a.label_id,b.label_id) endpoint,count(*) compared,
      count(*) FILTER(WHERE a.subject_id IS NULL OR b.subject_id IS NULL OR ({neq})) unequal
      FROM read_parquet({literal(native)}) a FULL OUTER JOIN
      (SELECT * FROM read_parquet({literal(original)}) WHERE label_id IN ({in_list})) b
      USING(subject_id,label_id) GROUP BY 1''').fetchdf().to_dict('records')
    lm=read_json(ROOT/'labels/labels.manifest.json');integrity=[]
    for p in (sealed,original,ROOT/'evidence/clinical_context_features.parquet',ROOT/'labels/development_label_attestation.json'):
        actual=sha(p)
        expected=dev['clinical_context']['sha256'] if 'context' in p.name else next((v['sha256'] for v in lm['output_integrity'] if Path(v['path'])==p),None)
        integrity.append({'file':str(p),'sha256':actual,'expected_sha256':expected,'equal':actual==expected})
    bound=all(x['equal'] for x in integrity) and labels.label_implementation_hash()==dev['label_implementation_hash']
    results={}
    for row in audit:
        idx=next(x for x in index_audit if x['endpoint']==row['endpoint'])
        results[row['endpoint']]={'locked_test_semantics_attested':bool(bound and row['compared']>0 and not any(row[k] for k in ('absent_regenerated','absent_sealed','unequal')) and idx['unequal']==0),
          'long_label_comparison':row,'native_index_comparison':idx,
          'reason':'Release-SQL auf derselben gebundenen Testreferenzquelle reproduziert; alle semantischen Felder und native Indizes nullsicher verglichen; technische B-Indexpolitik separat.'}
    save('label_semantics.json',{'endpoints':results,'source_integrity':integrity,'release_hash':labels.label_implementation_hash(),
      'sql_sha256':sha(OUT/'label_regeneration.sql'),'index_sql_sha256':sha(OUT/'index_regeneration.sql'),
      'script_sha256':sha(__file__),'development_attestation_changed':False})
    save('sources_labels.json',list(SOURCES.values()));c.close()
    print(json.dumps({'phase':'labels_complete','endpoints':results}),flush=True)

def features_build():
    import pandas as pd
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from werkzeuge.nachrechnung.resources import guard
    manifest=read_json(HOLD/'features.manifest.json');protocol=read_json(HOLD/'frozen/protocol.json')
    wanted=list(dict.fromkeys(['subject_id','study_id','split','ecg_time','extraction_status','original_sampling_rate','signal_sha256']+
       B3+B4+EXTRA+[x[2] for x in HYPOTHESES]+[x[3] for x in HYPOTHESES]))
    frames=[];availability={k:0 for k in wanted};proof=[];nulls={k:0 for k in wanted};row_count=0
    files=sorted((HOLD/'features').glob('part-*.parquet'))
    if set(p.name for p in files)!=set(manifest['feature_files']):raise ValueError('Fragmentinventar nicht identisch')
    for i,p in enumerate(files):
        if i%100==0:
            guard();print(json.dumps({'phase':'features','parts_done':i,'parts_total':len(files)}),flush=True)
        s=footer(p);actual=sha(p);expected=manifest['feature_files'][p.name]['sha256']
        if actual!=expected:raise ValueError('Featurehash geändert')
        side=read_json(p.with_suffix('.json'))
        if side['protocol_hash']!=protocol['protocol_hash'] or side['signature']['sha256']!=actual:raise ValueError('Fragmentprotokoll geändert')
        cols=[k for k in wanted if k in s['columns']];record(p,cols,None)
        f=pq.read_table(p,columns=cols).to_pandas()
        if f.study_id.tolist()!=side['study_ids']:raise ValueError('Fragment-Identitäten nicht identisch')
        if int(f.extraction_status.eq('ok').sum())!=side['successes']:raise ValueError('Fragmentstatus nicht identisch')
        for k in wanted:
            if k in f:availability[k]+=len(f)
        f=f.reindex(columns=wanted)
        for k in wanted:nulls[k]+=int(f[k].isna().sum())
        row_count+=len(f);frames.append(f)
        proof.append({'file':str(p),'sha256':actual,'rows':len(f),'columns_projected':cols,'protocol_hash':side['protocol_hash']})
    f=pd.concat(frames,ignore_index=True);del frames
    if set(f.split)!={'locked_test'} or f.study_id.duplicated().any() or f[['subject_id','study_id']].isna().any().any():raise ValueError('Featuresplit/-Identität ungültig')
    selected=projected(HOLD/'selected_metadata.parquet',['subject_id','study_id','split','ecg_time','age','sex'])
    cmp=f[['subject_id','study_id','split','ecg_time']].merge(selected[['subject_id','study_id','split','ecg_time']],on=['subject_id','study_id','split','ecg_time'],how='outer',indicator=True,validate='one_to_one')
    if not cmp._merge.eq('both').all():raise ValueError('Auswahl und Features nicht identisch')
    path=OUT/'features_projected_private.parquet';pq.write_table(pa.Table.from_pandas(f,preserve_index=False),path,row_group_size=32768)
    save('feature_inventory.json',{'rows':row_count,'subjects':int(f.subject_id.nunique()),'parts':len(files),
      'successes':int(f.extraction_status.eq('ok').sum()),'failures':int(f.extraction_status.ne('ok').sum()),
      'columns_present_rows':availability,'missingness_unconditional':nulls,
      'absent_columns':[k for k in wanted if availability[k]==0],
      'sampling_rate_unconditional':{str(k):int(v) for k,v in f.original_sampling_rate.value_counts(dropna=False).items()},
      'selection_identity_equal':True,'sha256':sha(path),'source_manifest_sha256':sha(HOLD/'features.manifest.json'),
      'script_sha256':sha(__file__),'no_marker_by_outcome_aggregation':True})
    save('feature_parts_hashes.json',proof);save('sources_features.json',list(SOURCES.values()))
    print(json.dumps({'phase':'features_complete','rows':row_count,'absent_columns':[k for k in wanted if availability[k]==0]}),flush=True)

def echo_id_matches(row):
    from decimal import Decimal,InvalidOperation
    try:
        key=Decimal(str(row.nearest_tte_measurement_id))
        return key in [Decimal(str(x)) for x in json.loads(row.evidence_ids_json)]
    except (ValueError,TypeError,InvalidOperation):return False

def coverage(frame):
    """Vollständige Referenzkandidaten; nur technischer Status, kein Markerfilter."""
    import pandas as pd
    if frame[['subject_id','study_id','ecg_time']].isna().any().any():raise ValueError('Indexidentität/Zeit fehlt')
    if frame.duplicated(['subject_id','study_id']).any():raise ValueError('Mehrfache Aufzeichnung')
    z=frame.sort_values(['subject_id','ecg_time','study_id'],kind='stable').copy()
    first=z.drop_duplicates('subject_id')
    z['__missing']=z.extraction_status.isna()
    z['__missing_before']=z.groupby('subject_id')['__missing'].cumsum()
    chosen=z.loc[z.extraction_status.eq('ok')].drop_duplicates('subject_id')
    proven=chosen.__missing_before.eq(0)
    missing_people=set(z.loc[z.__missing,'subject_id'])
    have=set(chosen.subject_id)
    counts={'reference_persons':len(first),'candidate_ecgs':len(z),
      'first_reference_ecg_selected':int(first.extraction_status.notna().sum()),
      'first_reference_ecg_technical_ok':int(first.extraction_status.eq('ok').sum()),
      'first_reference_ecg_failed':int((first.extraction_status.notna()&first.extraction_status.ne('ok')).sum()),
      'selected_first_technical_reference_persons':len(chosen),'proven_B_index_persons':int(proven.sum()),
      'selected_choice_with_unextracted_earlier_candidates':int((~proven).sum()),
      'no_selected_ok_with_unextracted_candidates':len(missing_people-have),
      'proven_no_technical_reference_persons':len(set(first.subject_id)-have-missing_people)}
    counts['unresolved_persons']=counts['selected_choice_with_unextracted_earlier_candidates']+counts['no_selected_ok_with_unextracted_candidates']
    counts['coverage_attested']=counts['unresolved_persons']==0
    counts['proven_B_index_fraction']=counts['proven_B_index_persons']/len(first) if len(first) else None
    counts['unresolved_fraction']=counts['unresolved_persons']/len(first) if len(first) else None
    return counts

def projection_build():
    import pandas as pd
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.dataset as ds
    from werkzeuge.nachrechnung.resources import guard
    features=projected(OUT/'features_projected_private.parquet',footer(OUT/'features_projected_private.parquet')['columns'])
    context_columns=['subject_id','study_id','split','ecg_time','age','sex','machine_heart_rate_bpm',
       'machine_metadata_usable_for_primary','machine_qrs_axis_deg','nearest_tte_measurement_id','nearest_tte_delta_hours',
       'nearest_tte_time','has_tte_30d','has_tte_7d','preferred_lvef','preferred_lvef_method','lvef_conflict_gt_10',
       'lvef_3d','lvef_biplane','lvef_generic','ntprobnp_nearest','ntprobnp_reference_upper','ntprobnp_delta_hours',
       'ntprobnp_nearest_censoring','ntprobnp_unit','ntprobnp_nearest_labevent_id','hadm_id',
       'troponin_interval_state_standard','troponin_max_url_ratio','troponin_max_url_labevent_id',
       'troponin_max_url_delta_hours','dx_cardiomyopathy','rv_function_category','ehr_time_alignment_confirmatory_eligible']
    ctx=projected(ROOT/'evidence/clinical_context_features.parquet',context_columns,
          ds.field('split')=='locked_test',"split='locked_test'")
    if ctx.study_id.duplicated().any() or set(ctx.split)!={'locked_test'}:raise ValueError('Kontextidentität ungültig')
    label_cols=['subject_id','study_id','label_id','label_state','eligibility_state','time_delta_hours','time_precision','evidence_ids_json','rule_version']
    native=projected(OUT/'native_index_reproduced_private.parquet',label_cols+['index_selection_strategy'])
    f=ctx.merge(features.drop(columns=['split','ecg_time']),on=['subject_id','study_id'],how='left',validate='one_to_one')
    # Geräte-HF nur bei gebundener zeitgeeigneter Metadatenregel; keine Signal-HF.
    f['heart_rate']=pd.to_numeric(f.machine_heart_rate_bpm,errors='coerce').where(f.machine_metadata_usable_for_primary.eq(True))
    f['sex_numeric']=f.sex.map({'F':0.,'M':1.})
    for ep in ('reduced_lvef_strict','echo_lvh_strict','echo_rv_systolic_dysfunction'):
        aux=projected(OUT/'labels_reproduced_private.parquet',label_cols,ds.field('label_id')==ep,'label_id='+ep)
        aux=aux.drop(columns='label_id').rename(columns={k:ep+'__'+k for k in label_cols if k not in ('subject_id','study_id','label_id')})
        f=f.merge(aux,on=['subject_id','study_id'],how='left',validate='one_to_one')
    del aux
    echo_cols=['measurement_id','subject_id','echo_time','preferred_lvef','preferred_lvef_method','lvef_conflict_gt_10','lvef_3d','lvef_biplane','lvef_generic']
    echo=projected(ROOT/'evidence/echo_measurements_wide.parquet',echo_cols,
         ds.field('subject_id').isin(ctx.subject_id.unique().tolist()),'subject_id IN Testpersonen (intern, keine IDs im Log)')
    e=ctx.loc[ctx.nearest_tte_measurement_id.notna()].merge(echo,left_on=['subject_id','nearest_tte_measurement_id'],
         right_on=['subject_id','measurement_id'],how='left',validate='many_to_one',suffixes=('_ctx','_echo'))
    checks={}
    for k in ['preferred_lvef','preferred_lvef_method','lvef_conflict_gt_10','lvef_3d','lvef_biplane','lvef_generic']:
        checks[k]=int((~(e[k+'_ctx'].eq(e[k+'_echo'])|(e[k+'_ctx'].isna()&e[k+'_echo'].isna()))).sum())
    preferred=echo.lvef_3d.combine_first(echo.lvef_biplane).combine_first(echo.lvef_generic)
    checks['preferred_method_order_value']=int((~(preferred.eq(echo.preferred_lvef)|(preferred.isna()&echo.preferred_lvef.isna()))).sum())
    checks['unmatched_echo_id']=int(e.measurement_id.isna().sum())
    expected_method=pd.Series(np.select([echo.lvef_3d.notna(),echo.lvef_biplane.notna(),echo.lvef_generic.notna()],['3d','biplane','generic'],default='missing'),index=echo.index)
    checks['preferred_method_order_name']=int(expected_method.ne(echo.preferred_lvef_method.fillna('missing')).sum())
    time_delta=(pd.to_datetime(e.echo_time)-pd.to_datetime(e.ecg_time)).dt.total_seconds()/3600
    checks['echo_time_delta']=int((~np.isclose(time_delta,e.nearest_tte_delta_hours,rtol=0,atol=1e-9,equal_nan=False)).sum())
    save('preferred_lvef_attestation.json',{'passed':not any(checks.values()),'test_context_echo_links':len(e),
       'test_person_echo_records':len(echo),'checks':checks,'method_order':'3d > biplane > generic; Release evidence.py 2709-2749',
       'conflict_rule':'Konfliktflag unverändert aus Echoquelle; im B-Referenzgate ausdrücklich false; fehlend ausschließen',
       'source_sha256':sha(ROOT/'evidence/echo_measurements_wide.parquet'),'script_sha256':sha(__file__)})
    projections={};flows={};class_counts={}
    dest=OUT/'b_test_eingaben';dest.mkdir(exist_ok=True)
    for bef,ep,marker,comp in HYPOTHESES:
        guard();ep_labels=projected(OUT/'labels_reproduced_private.parquet',label_cols,ds.field('label_id')==ep,'label_id='+ep)
        z=f.merge(ep_labels,on=['subject_id','study_id'],how='inner',validate='one_to_one')
        ref=z.eligibility_state.eq('eligible')&z.label_state.isin(['positive','negative'])
        if ep in ('reduced_lvef_strict','echo_rv_systolic_dysfunction'):
            ref &= z.time_precision.eq('timestamp')&z.time_delta_hours.abs().le(720)&np.isclose(z.time_delta_hours,z.nearest_tte_delta_hours,rtol=0,atol=1e-9,equal_nan=False)
            ref &= z.apply(echo_id_matches,axis=1)
            if ep=='reduced_lvef_strict':ref &= z.preferred_lvef.between(0,100)&np.isfinite(z.preferred_lvef)&z.lvef_conflict_gt_10.eq(False)
        elif ep in ('elevated_ntprobnp','acute_decompensated_heart_failure_proxy'):
            ref &= z.time_precision.eq('timestamp')&z.time_delta_hours.between(-6,12)
        z['reference_available']=ref.fillna(False)
        # Nur Referenzvalidität + Alter + technischer Status wählen, keine Markerwerte.
        candidates=z.loc[z.reference_available&z.age.ge(18)].copy()
        flow=coverage(candidates)
        nr=native.loc[native.label_id.eq(ep)]
        native_features=nr[['subject_id','study_id','index_selection_strategy']].merge(features[['subject_id','study_id','extraction_status']],on=['subject_id','study_id'],how='left',validate='one_to_one')
        flow.update(native_index_persons=len(nr),native_index_ecgs_selected=int(native_features.extraction_status.notna().sum()),
         reference_candidates_clock_eligible=int(candidates.ehr_time_alignment_confirmatory_eligible.eq(True).sum()),
         index_recipe='Erster adult/eligible/known/Referenzzeit-ID-valider Kandidat mit extraction_status=ok; ecg_time/study_id; kein Marker-/Kovariatenfilter')
        chosen=candidates.loc[candidates.extraction_status.eq('ok')].sort_values(['ecg_time','study_id'],kind='stable').drop_duplicates('subject_id')
        comparison=chosen[['subject_id','study_id']].merge(nr[['subject_id','study_id']],on='subject_id',suffixes=('_B','_native'))
        flow['selected_B_index_differs_from_native']=int(comparison.study_id_B.ne(comparison.study_id_native).sum())
        flows[bef]=flow
        # Outcomehäufigkeiten ausschließlich für sich, vor jeglichem Markerfilter.
        class_counts[bef]={str(k):int(v) for k,v in nr.label_state.value_counts(dropna=False).items()}
        output=z.loc[z.extraction_status.notna()].copy()
        output['source_split']=output['split'];output['split']='test'
        output['index_selection_strategy']='candidate_pool_reference_blind'
        # Ein unbekannter Marker bleibt NaN, nie berechnet oder durch Alias ersetzt.
        path=dest/f'{bef}_test.parquet';pq.write_table(pa.Table.from_pandas(output,preserve_index=False),path,row_group_size=16384)
        projections[bef]={'path':str(path),'sha256':sha(path),'columns':output.columns.tolist(),'rows':len(output),
          'endpoint':ep,'marker':marker,'comparator':comp,'role':'test','index_policy_attested':flow['coverage_attested'],
          'measurement_attestation':{'status':'STOP','reason':'Claimbezogene Messgateakte nicht vorhanden; fehlende exakte Marker/Kovariaten und gegebenenfalls Indexabdeckung.'}}
    technical=f.loc[f.extraction_status.notna()].copy()
    h1=projected(OUT/'labels_reproduced_private.parquet',label_cols,ds.field('label_id')=='reduced_lvef_strict','label_id=reduced_lvef_strict')
    technical=technical.merge(h1,on=['subject_id','study_id'],validate='one_to_one')
    technical['reference_available']=technical.eligibility_state.eq('eligible')&technical.label_state.isin(['positive','negative'])
    technical['source_split']=technical.split;technical['split']='test';technical['index_selection_strategy']='candidate_pool_first_technical'
    path=dest/'technical_test.parquet';pq.write_table(pa.Table.from_pandas(technical,preserve_index=False),path,row_group_size=16384)
    tech_cov=coverage(f.loc[f.age.ge(18)])
    save('index_coverage.json',{'hypotheses':flows,'technical_universe':tech_cov,
       'denominator':'Vollständige Testreferenzkandidaten aus allen Test-EKGs; extraction_status nur für die ausgewählten EKGs bekannt',
       'selection_bias':'Unextrahierte frühere Referenzkandidaten können echte B-Indizes sein. Ersatz durch späteres verfügbares EKG würde die Population ändern; keine vollständige Attestation bei unresolved_persons>0.',
       'script_sha256':sha(__file__)})
    save('endpoint_class_counts_only.json',{'population':'native_index_reproduced_private.parquet je label_id; kein technischer/Marker-/Kovariatenfilter','counts':class_counts})
    save('projections.json',{'hypotheses':projections,'technical':{'path':str(path),'sha256':sha(path),'columns':technical.columns.tolist(),'rows':len(technical)},'technical_attested':tech_cov['coverage_attested']})
    save('sources_projection.json',list(SOURCES.values()))
    print(json.dumps({'phase':'projection_complete','coverage':flows,'technical':tech_cov}),flush=True)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def save(name,value):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')

def permitted(path):
    p=Path(path)
    if any(v in str(p).lower() for v in ('associations','predictions','checkpoint','models.json')):
        raise PermissionError('Ergebnis-/Fitdatei ausgeschlossen')
    return p

def footer(path):
    import pyarrow.parquet as pq
    p=permitted(path);pf=pq.ParquetFile(p)
    return {'rows':pf.metadata.num_rows,'bytes':p.stat().st_size,'columns':pf.schema_arrow.names}

def inspect():
    from werkzeuge.nachrechnung.resources import guard
    paths=[HOLD/'selected_metadata.parquet',next((HOLD/'features').glob('*.parquet')),
           ROOT/'labels/sealed/locked_endpoint_index_cohorts.parquet',ROOT/'labels/endpoint_index_cohorts.parquet',
           ROOT/'evidence/clinical_context_features.parquet',ROOT/'master/ecg_master_index.parquet',
           ROOT/'evidence/echo_measurements_wide.parquet',ROOT/'evidence/ecg_lab_features.parquet']
    result={}
    for p in paths:
        guard();result[str(p)]=footer(p)
    save('schemas.json',result)
    print(json.dumps({p:{'rows':s['rows'],'columns_n':len(s['columns'])} for p,s in result.items()}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['inspect','provenance','labels','features','projection']);args=parser.parse_args()
    from werkzeuge.nachrechnung.resources import guard,Budget
    OUT.mkdir(parents=True,exist_ok=True);start=guard();save('resources_start.json',start)
    snapshot=OUT/'executed_source'/sha(__file__)/Path(__file__).name
    snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(Path(__file__).read_bytes())
    with Budget(OUT) as budget:
        if args.mode=='inspect':inspect()
        elif args.mode=='provenance':provenance()
        elif args.mode=='labels':labels_build()
        elif args.mode=='features':features_build()
        elif args.mode=='projection':projection_build()
    save('resources_end.json',budget.summary())
    save('resources_'+args.mode+'.json',{'start':start,**budget.summary(),'script_sha256':sha(__file__)})
