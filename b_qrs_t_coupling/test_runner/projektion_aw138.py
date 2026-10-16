"""Final technical projections and STOP manifest, without associations."""
import json, sys
from pathlib import Path
from datetime import datetime,timezone
from . import extraktion_aw138 as a
from . import testeingaben_aw137 as old
from werkzeuge.nachrechnung.resources import guard,Budget
OUT=a.OUT

def records(folder):
    import pandas as pd
    rows=[];expected=None;request=None;hashes=None
    if folder=='marker_records_v2':
        import hashlib
        expected=hashlib.sha256(json.dumps(a.LIMIT_OPTIONS,sort_keys=True).encode()).hexdigest()
        request=a.sha(OUT/'marker_tasks_private.parquet')
        hashes={v['file']:v['sha256'] for v in a.readj(OUT/'final_checkpoint_hashes_private.json')}
    elif folder=='earlier_records_v2':
        import hashlib
        expected=hashlib.sha256(json.dumps(a.OPTIONS,sort_keys=True).encode()).hexdigest()
        request=a.sha(OUT/'earlier_tasks_v2_private.parquet')
    for p in sorted((OUT/folder).glob('*.json')):
        value=json.loads(p.read_text(encoding='utf8'))
        if folder=='marker_records_v2':
            if p.name not in hashes or a.sha(p)!=hashes[p.name]:raise ValueError('Attested marker checkpoint changed')
        if folder in ('marker_records_v2','earlier_records_v2'):
            if value['request_sha256']!=request or value['options_sha256']!=expected or value.get('adapter_sha256') is not None:
                raise ValueError('Marker checkpoint binding differs')
        rows.append(value['result'])
    return pd.DataFrame(rows)

def build():
    import numpy as np
    import pandas as pd
    import pyarrow.dataset as ds
    from .daten import select_index
    eq=a.readj(OUT/'equivalence_attestation_v2.json')
    if not eq['passed']:raise ValueError('Release development equivalence gate not passed')
    audit=a.readj(OUT/'final_checkpoint_attestation.json')
    if not audit['passed'] or audit['remaining_records']!=0:raise ValueError('Final checkpoint check missing')
    if a.sha(OUT/'final_checkpoint_hashes_private.json')!=audit['inventory_sha256']:raise ValueError('Checkpoint inventory changed')
    coverage=a.readj(OUT/'index_coverage.json');parent=a.readj(old.OUT/'input_manifest_v2.json')
    if a.sha(OUT/'earlier_values_private.parquet')!=coverage['earlier_values_sha256']:raise ValueError('Index candidate values changed')
    if a.sha(OUT/'earlier_tasks_private.parquet')!=coverage['earlier_tasks_sha256']:raise ValueError('Index candidate request changed')
    for entry in list(parent['hypotheses'].values())+[parent['technical']]:
        if a.sha(entry['path'])!=entry['sha256']:raise ValueError('Projection of the preceding stage changed')
    pref=parent['attestations']['preferred_lvef_bound'];lab=parent['attestations']['label_mapping_bound']
    for proof in pref['evidence']+lab['evidence']:
        if a.sha(proof['file'])!=proof['sha256']:raise ValueError('Reference attestation of the preceding stage changed')
    package=a.readj(a.PROJECT/'werkzeuge/b_test/parameter_entwicklung.json')
    tasks=a.readp(OUT/'marker_tasks_private.parquet');values=records('marker_records_v2')
    if len(values)!=len(tasks) or set(values.study_id)!=set(tasks.study_id) or values.study_id.duplicated().any():raise ValueError('Marker extraction incomplete or ambiguous')
    if not values.split.eq('locked_test').all():raise ValueError('Wrong extraction partition')
    for k in a.FIELDS:
        if k not in values:raise ValueError('Exact ID missing: '+k)
    a.writep('marker_values_private.parquet',values)
    stored=a.readp(old.OUT/'features_projected_private.parquet')
    cmp=stored.merge(values,on=['subject_id','study_id'],suffixes=('_aw137','_aw138'),validate='one_to_one')
    mismatch={};maxdiff={}
    for k in a.FIELDS:
        if k in a.EXISTING_ONLY or k not in stored or stored[k].isna().all():continue
        x=pd.to_numeric(cmp[k+'_aw137'],errors='coerce').astype(float);y=pd.to_numeric(cmp[k+'_aw138'],errors='coerce').astype(float)
        mismatch[k]=int((~np.isclose(x,y,rtol=1e-8,atol=0,equal_nan=True)).sum())
        d=(x-y).abs().dropna();maxdiff[k]=None if d.empty else float(d.max())
    status_diff=int(cmp.extraction_status_aw137.ne(cmp.extraction_status_aw138).sum())
    signal_diff=int((~cmp.signal_sha256_aw137.eq(cmp.signal_sha256_aw138)).sum())
    a.writep('stored_vs_recomputed_private.parquet',cmp)
    a.save('stored_vs_recomputed_attestation.json',{'compared_ecgs':len(cmp),'field_mismatch_counts':mismatch,
      'max_absolute_difference':maxdiff,'status_mismatches':status_diff,'signal_sha256_mismatches':signal_diff,
      'scope':'Technical comparison without endpoints. Missing IDs and the complete B3/B4/ST/block measurement basis come from the recomputation; already existing H2/H3/DTW markers are taken unchanged from the preceding stage, new indices from the full candidate extraction.',
      'script_sha256':a.sha(__file__)})
    labels=a.readp(old.OUT/'labels_reproduced_private.parquet',
      ['subject_id','study_id','label_id','label_state','eligibility_state','time_delta_hours','time_precision','evidence_ids_json','rule_version'],
      ds.field('study_id').isin(tasks.study_id.tolist()))
    proposals={};inventory={}
    earlier=records('earlier_records_v2').set_index('study_id')
    vi=values.set_index('study_id')
    meta=tasks.set_index('study_id')
    report=meta.machine_report.fillna('').str.lower()
    sinus=meta.machine_metadata_usable_for_primary.eq(True).fillna(False)&report.str.contains(r'sinus (?:rhythm|bradycardia|tachycardia|arrhythmia)',regex=True)&~report.str.contains(r'atrial fibrillation|atrial flutter|paced|pacemaker',regex=True)
    for part,ep,marker,comp in old.HYPOTHESES+[('technical','reduced_lvef_strict',None,None)]:
        guard();base=a.readp(old.OUT/f'b_test_eingaben/{part}_test.parquet')
        idx=a.readp(OUT/f'indices/{part}_private.parquet')
        candidates=a.readp(OUT/f'candidates/{part}_private.parquet')
        extra=candidates.loc[candidates.study_id.isin(earlier.index)&~candidates.study_id.isin(base.study_id)].copy()
        if part=='technical':
            lab=labels.loc[labels.label_id.eq(ep)]
            extra=extra.merge(lab,on=['subject_id','study_id'],how='left',validate='one_to_one')
            extra['reference_available']=extra.eligibility_state.eq('eligible')&extra.label_state.isin(['positive','negative'])
        extra['extraction_status']=extra.study_id.map(earlier.extraction_status)
        for endpoint in ['reduced_lvef_strict','echo_lvh_strict','echo_rv_systolic_dysfunction']:
            aux=labels.loc[labels.label_id.eq(endpoint)].drop(columns='label_id').rename(columns={k:endpoint+'__'+k for k in labels.columns if k not in ['subject_id','study_id','label_id']})
            extra=extra.merge(aux,on=['subject_id','study_id'],how='left',validate='one_to_one')
        f=pd.concat([base,extra.reindex(columns=base.columns)],ignore_index=True)
        # Add technical status from the complete worker, not from marker values.
        f['extraction_status']=f.extraction_status.combine_first(f.study_id.map(earlier.extraction_status))
        is_index=f.study_id.isin(idx.study_id)
        for k in a.FIELDS+['original_sampling_rate','signal_sha256']:
            if k not in f:f[k]=np.nan
            # Lossless runner storage: boolean measurement fields as 0/1/NaN,
            # numeric measurement fields as float64, signal hash as text/null.
            if k=='signal_sha256':
                f[k]=f[k].astype(object)
            else:
                f[k]=pd.to_numeric(f[k],errors='raise').astype(float)
            if k in a.EXISTING_ONLY:
                # Deliberately not re-requested, already bound markers are retained.
                filled=f.loc[is_index,'study_id'].map(earlier[k]).combine_first(f.loc[is_index,k])
                f.loc[is_index,k]=pd.to_numeric(filled,errors='raise').astype(float).to_numpy()
            else:
                mapped=f.loc[is_index,'study_id'].map(vi[k])
                if k!='signal_sha256':mapped=pd.to_numeric(mapped,errors='raise').astype(float)
                f.loc[is_index,k]=mapped.to_numpy()
        f['aw138_reextraction_status']=f.study_id.map(vi.extraction_status)
        f['sinus_selector']=f.study_id.map(sinus).astype('boolean')
        f['heart_rate']=pd.to_numeric(f.machine_heart_rate_bpm,errors='coerce').where(f.machine_metadata_usable_for_primary.eq(True))
        f['sex_numeric']=f.sex.map({'F':0.,'M':1.})
        f['source_split']='locked_test';f['split']='test';f['index_selection_strategy']='candidate_pool_reference_blind' if part!='technical' else 'candidate_pool_first_technical'
        final_selected=select_index(f,ep,technical=part=='technical')
        expected=set(zip(idx.subject_id,idx.study_id));actual=set(zip(final_selected.subject_id,final_selected.study_id))
        if expected!=actual:raise ValueError('Runner selects a different index: '+part)
        a.writep(f'b_test_eingaben/{part}_test.parquet',f)
        p=OUT/f'b_test_eingaben/{part}_test.parquet'
        inv={'index_n':len(final_selected),'candidate_pool_rows':len(f),'added_earlier_candidates':len(extra),
          'index_choice_equal_to_full_candidate_proof':True,'missingness_unconditional_index':{},
          'reextraction_status_counts_index':final_selected.aw138_reextraction_status.value_counts(dropna=False).to_dict()}
        for k in a.FIELDS:inv['missingness_unconditional_index'][k]=int(final_selected[k].isna().sum())
        inventory[part]=inv
        oldentry=parent['technical'] if part=='technical' else parent['hypotheses'][part]
        entry=dict(oldentry);entry.update(path=str(p),sha256=a.sha(p),columns=f.columns.tolist(),rows=len(f),
          index_policy_attested=True,index_coverage=coverage['parts'][part]['after'],
          missing_required_B3_ids=[],missing_required_B4_ids=[],
          measurement_attestation={'status':'STOP','reason':'Claim-related measurement gate record open; complete technical extraction is not a clinical measurement release.'},
          sinus_selector_attested=True)
        proposals[part]=entry
    a.save('feature_inventory.json',{'parts':inventory,'unique_index_ecgs':len(values),'status_counts':values.extraction_status.value_counts().to_dict(),
      'original_sampling_rate_unconditional':{str(k):int(v) for k,v in values.original_sampling_rate.value_counts(dropna=False).items()},
      'present_exact_ids':a.FIELDS,'absent_exact_ids':[], 'no_marker_outcome_aggregation':True,'script_sha256':a.sha(__file__)})
    projections={'hypotheses':{k:v for k,v in proposals.items() if k!='technical'},'technical':proposals['technical'],'technical_attested':True}
    a.save('projections.json',projections)
    # Same verified identity recipe, only the output path rebound in this process.
    from .attestierung_aw137 import identities
    previous_out=old.OUT;old.OUT=OUT
    try:ids=identities(projections,package)
    finally:old.OUT=previous_out
    if not ids['passed']:raise ValueError('Person disjointness not passed')
    release_proof()
    measurement_proof()
    manifest=dict(parent);manifest.update(schema_version='AW138-3',utc=datetime.now(timezone.utc).isoformat(),
      hypotheses=projections['hypotheses'],technical=projections['technical'],technical_attested=True,
      test_access_scope='Technical signals/markers/references/indices of this stage; no associations and no research holdout result files',
      parent_manifest={'path':str(previous_out/'input_manifest_v2.json'),'sha256':a.sha(previous_out/'input_manifest_v2.json')},
      source_candidates=list(a.SOURCES.values()),projector_sha256=a.sha(__file__),attestation_script_sha256=a.sha(__file__),
      ready_for_association_run=False)
    manifest['attestations']=dict(parent['attestations'])
    def gate(name,value,reason,needed,evidence,affected=None):
        manifest[name]=bool(value);manifest['attestations'][name]={'value':bool(value),'reason':reason,'additional_evidence_needed':needed,
          'affected_parts':affected or list(projections['hypotheses']),'affected_analyses':'B3/B4 and associated sensitivity analyses',
          'evidence':[{'file':str(p),'sha256':a.sha(p),'code_location':loc} for p,loc in evidence]}
    gate('person_disjoint',True,'Checked again on the final projection hashes against the master and the bound development partition.',
      'Check again on every projection change.',[(OUT/'person_disjoint_attestation.json','identities')])
    technical_ok=values.extraction_status.eq('ok').all() and status_diff==0 and signal_diff==0 and not any(mismatch.values())
    gate('release_input_bound',technical_ok,'Bound release, correct value-blind metadata projection, complete technical indices and exact measurement IDs; development reproduction with n=200 passed. Runtime snapshot remains inaccessible.',
      'No complete historical runtime/library attestation; clarify status/signal deviations, if any.',
      [(OUT/'release_attestation.json','release_proof'),(OUT/'equivalence_attestation_v2.json','equivalence_v2'),
       (OUT/'input_projection_correction.json','project_worker_inputs'),(OUT/'index_coverage.json','index_prepare'),
       (OUT/'stored_vs_recomputed_attestation.json','build')])
    manifest['attestations']['release_input_bound']['evidence'].append({'file':str(OUT/'limited_equivalence_attestation.json'),'sha256':a.sha(OUT/'limited_equivalence_attestation.json'),'code_location':'limited_equivalence'})
    manifest['attestations']['release_input_bound']['evidence'].append({'file':str(OUT/'final_checkpoint_attestation.json'),'sha256':a.sha(OUT/'final_checkpoint_attestation.json'),'code_location':'checkpoints_aw138.main'})
    gate('measurement_gates_bound',False,'The measurement-audit analysis and the signal pilot provide partial evidence, but there is no fully accepted claim-related measurement gate record for primary markers and the model basis.',
      'Reviewer: assess the pre-specified measurement-gate items, absolute repeatability in a suitable population, and delineation, filter, device, transformation and pacing limits claim by claim; B3/B4 measurement basis separately.',
      [(OUT/'measurement_gates_attestation.json','measurement_proof')])
    manifest['source_release_equivalence']={**parent['source_release_equivalence'],'available_feature_ids':True,'development_reextraction_equivalent':True,
      'historical_full_runtime_tree_verified':False,'runtime_access':'Preceding stage: file access denied; this stage makes no attempt to circumvent it',
      'equivalence_evidence':str(OUT/'equivalence_attestation_v2.json')}
    manifest['independent_stops']=dict(parent['independent_stops']);manifest['independent_stops']['IPW']='Test index universe technically complete; unchanged development parameters and measurement gates remain a prerequisite.'
    manifest['independent_stops']['marker_substitution']='All exact IDs directly from the release; no alias or polarity change.'
    for part,entry in manifest['hypotheses'].items():entry['affected_analyses']='Measurement gates open; hour-based endpoints additionally under the unchanged time-of-day/development-parameter gate. No input ID or index gap.'
    from .vorbereitung_aw136 import endpoint_stops
    manifest['stopped_parts']=endpoint_stops(manifest,package)
    a.save('input_manifest_v3.json',manifest)
    draft=a.readj(previous_out/'FREIGABE_K_entwurf.json');draft.update(status='DRAFT; no release, measurement gates open',
      input_manifest_path=str(OUT/'input_manifest_v3.json'),input_manifest_sha256=a.sha(OUT/'input_manifest_v3.json'),
      utc_blindheitsattestation=None,not_estimable_parts=list(manifest['stopped_parts']))
    draft['ready_for_association_run']=False
    draft.pop('freigegeben_von',None);a.save('FREIGABE_K_entwurf.json',draft)
    stability=[]
    for source in a.SOURCES.values():stability.append({'file':source['path'],'sha256':source['sha256'],'stable':a.sha(source['path'])==source['sha256']})
    if not all(x['stable'] for x in stability):raise ValueError('Source changed during projection')
    a.save('source_stability.json',stability)
    a.save('final_verification.json',{'complete':True,'gates':{k:manifest[k] for k in manifest['attestations']},'stopped_parts':list(manifest['stopped_parts']),
      'manifest_sha256':a.sha(OUT/'input_manifest_v3.json'),'draft_sha256':a.sha(OUT/'FREIGABE_K_entwurf.json'),'no_association_run':True,'no_freigegeben_von':True})
    print(json.dumps(a.readj(OUT/'final_verification.json')),flush=True)

def release_proof():
    from importlib.metadata import version,PackageNotFoundError
    a.release()
    from vcg_biomarker.mimic.feature_extraction import feature_implementation_hash
    from vcg_biomarker.mimic import feature_record
    tree=[]
    for p in sorted(a.RELEASE.rglob('*.py')):tree.append({'relative':p.relative_to(a.RELEASE).as_posix(),'sha256':a.sha(p)})
    import hashlib
    treehash=hashlib.sha256(json.dumps(tree,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    versions={}
    for n in ['numpy','scipy','pandas','pyarrow','neurokit2','scikit-learn','wfdb']:
        try:versions[n]=version(n)
        except PackageNotFoundError:versions[n]=None
    a.save('release_source_hashes.json',tree)
    a.save('release_attestation.json',{'release':str(a.RELEASE),'source_tree_sha256':treehash,
      'source_tree_recipe':'SHA256 canonical JSON sorted relative .py paths and SHA256s, sort_keys=True/separators comma-colon',
      'feature_implementation_hash':feature_implementation_hash(),'feature_record_sha256':a.sha(feature_record.__file__),
      'options':a.OPTIONS,'python_executable':sys.executable,'python_version':sys.version,'current_dependencies':versions,
      'missing_marker_worker_options':a.LIMIT_OPTIONS,'limited_equivalence':a.readj(OUT/'limited_equivalence_attestation.json'),
      'equivalence_passed':a.readj(OUT/'equivalence_attestation_v2.json')['passed'],
      'runtime_snapshot_verified':False,'runtime_boundary':'Runtime snapshot of the preceding stage denied; neither opened nor circumvented. Present-day source/library binding does not prove complete historical runtime equality.',
      'extraction_script_sha256':a.sha(a.__file__),'projection_script_sha256':a.sha(__file__)})

def measurement_proof():
    import pandas as pd
    files=[a.PROJECT/'publikationen/PUB-01_messvaliditaet/T2_repeatability.csv',
      a.PROJECT/'publikationen/PUB-01_messvaliditaet/MANUSKRIPT_PMEA_v17.md',
      a.PROJECT/'nachrechnung/MESSAUDIT_VORARBEITEN.md',a.PROJECT/'nachrechnung/B_VORBEREITUNG_AW121.md',
      a.PROJECT/'standards/02_pruefkette.md',a.PROJECT/'nachrechnung/SPEZIFIKATION_B_FREEZE_v4.md']
    for p in files:a.bind(p)
    t=pd.read_csv(files[0]);row=t.loc[t.marker_id.eq('qrs_t_normalized_discrete_frechet')]
    if len(row)!=1:raise ValueError('Primary marker repeatability row not unique')
    rec=row.iloc[0].to_dict()
    import math
    rec={k:(None if isinstance(v,float) and not math.isfinite(v) else v) for k,v in rec.items()}
    a.save('measurement_gates_attestation.json',{'passed':False,'primary_marker':'qrs_t_normalized_discrete_frechet',
      'sources':[{'file':str(p),'sha256':a.sha(p)} for p in files],
      'repeatability_A':{'file':str(files[0]),'filter':'marker_id=qrs_t_normalized_discrete_frechet','row':rec,
        'limit':'Clinical short-term repeat; contains true state change, no isolated measurement error and no blanket gate for B complete cases.'},
      'AW121':{'sections':'transform/signal pilot; synthetic core tests; claim limits',
        'limit':'Stratified development signal pilot; neither a full-population measurement gate nor external validation.'},
      'P01a':'Scoped release hash, present-day source tree, input and output hashes documented; historical complete runtime tree open.',
      'P01b':'Technical index/missingness aggregates reproduced; no test association aggregates computed at all.',
      'P01c':'200 development ECGs individually re-extracted and numerically compared; no independent manual fiducial/measurement reference.',
      'missing':['Claim-related acceptance for the B primary marker and all B3/B4 covariates',
        'Suitable absolute repeatability/delineation error limits in the B target population',
        'Filter/device/transformation and pacing-remnant limits; 500 Hz and hash equality alone are not sufficient',
        'Full-population scaling for signal variants still unbound; EchoNext still excluded'],
      'no_new_literature_or_medical_validation':True,'script_sha256':a.sha(__file__)})

if __name__=='__main__':
    start=guard();launch=a.sha(__file__)
    snap=OUT/'executed_source'/launch/Path(__file__).name;snap.parent.mkdir(parents=True,exist_ok=True);snap.write_bytes(Path(__file__).read_bytes())
    with Budget(OUT) as budget:build()
    a.save('resources_projection.json',{'start':start,**budget.summary(),'script_sha256':launch})
    a.save('sources_projection.json',list(a.SOURCES.values()))
