"""AW156: gekapselte technische Fortsetzung; keine Änderung gebundener Rezepte.

Keine Einzelzeilen in Toolausgaben. Wiederaufnahme nur eigener Checkpoints.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import gc
import json
import os
from pathlib import Path
import traceback
from . import runner, adapter
from .daten import select_index, masks, classify_lvef, truth, echo_ids
from .statistik import raw_design, fixed_design
from .export import projected
from werkzeuge.nachrechnung.resources import guard, Budget
from werkzeuge.nachrechnung.core import auc, logistic_fit
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.dataset as ds

P=runner.PROJECT
OUT=Path(str(data_path("b.aw136_b_fortsetzung_aw156")))
AUX=Path(str(data_path("b.aw156")))
PARAM=P/'werkzeuge/b_test/parameter_entwicklung.json'
OLD=Path(str(data_path("b.aw136_b_testlauf_aw142_ergebnis_json")))
MAN=Path(str(data_path("b.b_freigabe_input_manifest_v5_json")))
MANHASH='5bfa03ae40fb812f6d037b618c8e7b5a76512b656a5bf3ab1f02265a8223b49e'

def save(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item()),encoding='utf8')

def ledger(action,scope):
    with (P/'koordination/TESTZUGRIFFE.md').open('a',encoding='utf8') as f:
        f.write(f'\n| {runner.now()} | AW-156 | {action} | Freeze v4 `{runner.SPEC_HASH}`; Manifest v5 `{MANHASH}` | {scope}; keine neuen Modelle/Parameter |\n')

def checked(source,role,columns=None):
    path=Path(source['path']);guard()
    cols=columns or source.get('columns')
    scope=f'{path}; SHA-256 {source["sha256"]}; Rolle {role}'
    ledger('Beginn Inputlesung/Hashprüfung',scope)
    try:
        if runner.sha(path)!=source['sha256']:raise ValueError('Inputhash verändert')
        pf=pq.ParquetFile(path)
        if cols is None:cols=pf.schema_arrow.names
        if set(cols)-set(pf.schema_arrow.names):raise ValueError('Spaltenvertrag fehlt')
        if role in ('train','development'):f=projected(path,cols)
        else:f=ds.dataset(str(path),format='parquet').to_table(columns=cols).to_pandas()
        if role=='development':
            if not set(f.split).issubset({'train','validation'}):raise ValueError('Entwicklungspartition falsch')
        elif set(f.split)!={role}:raise ValueError('Partition nicht physisch getrennt')
        with (OUT/'inputs.jsonl').open('a',encoding='utf8') as log:
            log.write(json.dumps({'path':str(path),'sha256':source['sha256'],'role':role,'columns':cols,'schema':str(pf.schema_arrow),'rows':len(f)})+'\n')
        return f
    finally:ledger('Ende Inputlesung/Hashprüfung',scope)

def missingness(f,entry):
    z,good,flow=masks(f,entry['endpoint'],entry['marker'],entry['comparator'])
    x,names=raw_design(z,entry['marker'],entry['comparator'])
    xb,nb=raw_design(z,entry['marker'],entry['comparator'],strong=True)
    adult=z.age.ge(18);known=z.label_state.isin(['positive','negative']);ref=pd.Series(True,index=z.index)
    if entry['endpoint']=='reduced_lvef_strict':
        delta=pd.to_numeric(z.time_delta_hours,errors='coerce');nearest=pd.to_numeric(z.nearest_tte_delta_hours,errors='coerce')
        v=pd.to_numeric(z.preferred_lvef,errors='coerce')
        ref=z.time_precision.eq('timestamp')&delta.abs().le(720)&np.isclose(delta,nearest,atol=1e-9,rtol=0)&z.apply(echo_ids,axis=1)&np.isfinite(v)&v.between(0,100)&z.lvef_conflict_gt_10.eq(False).fillna(False)
    stages={'adult':adult,'index_policy':z.index_selection_strategy.eq('first_verified_ecg_reference_result_blind'),'eligible':z.eligibility_state.eq('eligible'),'known_reference':known,'reference_time_id_value':ref,'confirmatory_time_alignment':pd.Series(True,index=z.index),'complete_B3_marker':pd.Series(np.isfinite(x).all(1),index=z.index)}
    cum=pd.Series(True,index=z.index);rows=[]
    cls=pd.Series(np.where(z.preferred_lvef.lt(40),'positive','negative'),index=z.index) if entry['endpoint']=='reduced_lvef_strict' else z.label_state
    # H1 class only meaningful if numeric reference is known; all remaining states separate.
    if entry['endpoint']=='reduced_lvef_strict':cls=cls.where(known&z.preferred_lvef.notna(),z.label_state)
    for stage,mask in [('index_universe',cum)]+list(stages.items()):
        cum &= mask.fillna(False)
        for c in sorted(cls.dropna().unique()):rows.append({'stage':stage,'class':c,'n':int((cum&cls.eq(c)).sum())})
    if int(cum.sum())!=int(good.sum()):raise AssertionError('Flussmasken ungleich')
    # Missingness denominator: reference-valid set before complete-case and B3 set for B4.
    refset=pd.Series(True,index=z.index)
    for name,mask in stages.items():
        if name!='complete_B3_marker':refset &= mask.fillna(False)
    miss=[]
    for denominator,dmask,xx,nn in [('reference_valid_before_CC',refset,x,names),('B3_complete_for_B4',good,xb,nb)]:
        for c in ('positive','negative'):
            chosen=(dmask&cls.eq(c)).to_numpy();n=int(chosen.sum())
            for i,name in enumerate(nn):
                count=int((~np.isfinite(xx[chosen,i])).sum())
                miss.append({'denominator':denominator,'class':c,'column':name,'n':n,'missing_n':count,'missing_fraction':count/n if n else None})
    b4=good&pd.Series(np.isfinite(xb).all(1),index=z.index)
    for c in ('positive','negative'):rows.append({'stage':'complete_B4_marker_on_B3','class':c,'n':int((b4&cls.eq(c)).sum())})
    return {'flow_by_class':rows,'missing_by_class':miss,'bound_flow':flow,'class_policy':'H1 numeric LVEF<40; H4 label_state; no reassignment of unknown/unavailable'}

def main():
    AUX.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'ergebnis.json').exists():raise FileExistsError('Fertiges Ergebnis nicht neu rechnen')
    guard();binding=json.loads((PARAM.parent/'bindung.json').read_text(encoding='utf8'))
    hashes={}
    for path,expected in binding['files'].items():
        actual=runner.sha(path);hashes[path]=actual
        if actual!=expected:raise ValueError('Gebundene Datei verändert: '+path)
    if runner.sha(MAN)!=MANHASH:raise ValueError('Manifest verändert')
    if runner.sha(P/'nachrechnung/SPEZIFIKATION_B_FREEZE_v4.md')!=runner.SPEC_HASH:raise ValueError('Freeze verändert')
    runner.authorize('test',PARAM)
    package=json.loads(PARAM.read_text(encoding='utf8'));manifest=json.loads(MAN.read_text(encoding='utf8'))
    if manifest.get('technical_attested') is not True:raise ValueError('Technischer Input nicht attestiert')
    runner.CHECKPOINT_ROOT=OUT/'checkpoints'
    numeric={k:v for k,v in binding['files'].items() if k.endswith('.py') and '/tests/' not in k.replace('\\','/') and not any(s in k for s in ('bericht_aw131','abgabe.py','migrate_aw131','export.py','extras.py'))}
    runner.NUMERIC_BINDING=__import__('hashlib').sha256(json.dumps(numeric,sort_keys=True).encode()).hexdigest()
    save(AUX/'bindung_pruefung.json',{'utc':runner.now(),'files':hashes,'wrapper_sha256':runner.sha(__file__),'parameter_sha256':runner.sha(PARAM),'manifest_sha256':MANHASH,'numeric_binding':runner.NUMERIC_BINDING,'pid':os.getpid()})
    ledger('Beginn AW-142-Ergebnislesung',str(OLD))
    try:
        if runner.sha(OLD)!='f17ef84b6533f71d3d7ed0099e81ed53b684e4e45f8539e3e0a17d0a0e3301a7':raise ValueError('AW142 Ergebnis verändert')
        old=json.loads(OLD.read_text(encoding='utf8'))
    finally:ledger('Ende AW-142-Ergebnislesung',str(OLD))
    result={'utc_start':runner.now(),'run':manifest['run'],'nt':{},'missingness':{},'sources':{},'sensitivities':[]}
    frames={}
    needed=set(['subject_id','study_id','split','ecg_time','age','sex','extraction_status','reference_available','index_selection_strategy','label_state','eligibility_state','ehr_time_alignment_confirmatory_eligible','preferred_lvef','lvef_conflict_gt_10','time_delta_hours','time_precision','nearest_tte_delta_hours','nearest_tte_measurement_id','evidence_ids_json','machine_metadata_usable_for_primary','machine_heart_rate_bpm','heart_rate','sex_numeric','qrs_duration_ms','qtc_fridericia_ms','spatial_qrs_t_angle_deg','classical_cornell_product','classical_sokolow_lyon','magnitude_area_qrs','spatial_low_voltage_qrs_peak_magnitude','classical_right_axis_deviation','qrs_t_normalized_discrete_frechet','qrs_t_frechet_beat_median'])
    for bef in ('BEF-0063','BEF-0049'):
        source=manifest['hypotheses'][bef];cols=[c for c in source['columns'] if c in needed]
        f=checked(source,'test',cols);frames[bef]=select_index(f,package['hypotheses'][bef]['endpoint']);del f
        result['sources'][bef]=source
        result['missingness'][bef]=missingness(frames[bef],package['hypotheses'][bef])
    # No H4 outcome refit. H1 pointfit solely the explicitly required equivalence check.
    h1=package['hypotheses']['BEF-0063'];h1['_require_measurement']=True;h1['measurement_attestation']=manifest['hypotheses']['BEF-0063']['measurement_attestation']
    original=old['results']['BEF-0063'];point=adapter.estimate(frames['BEF-0063'],h1,np.unique(frames['BEF-0063'].subject_id),bootstrap=False)
    keys=['beta','or','N','G','class_persons','p_raw','beta_ci95_wald','or_ci95_wald','auc_G','auc_G_marker','delta_auc','auc_marker_original_steigend']
    comparison={k:{'AW142':original[k],'AW156':point[k],'exact':point[k]==original[k]} for k in keys}
    save(OUT/'equivalence.json',comparison)
    if not all(v['exact'] for v in comparison.values()):raise AssertionError('H1 Äquivalenz nicht exakt; keine IPW')
    result['equivalence']=comparison
    z,good,_=masks(frames['BEF-0063'],h1['endpoint'],h1['marker'],h1['comparator']);x,names=raw_design(z.loc[good],h1['marker'],h1['comparator']);design=fixed_design(x,h1['scaling_B3']);y=classify_lvef(z.loc[good,'preferred_lvef'])
    base=design[:,:-1];b=logistic_fit(base,y,np.ones(len(y)));scores=base@b;marker=z.loc[good,h1['marker']].to_numpy(float)
    # Reconstruct predictions deterministically because AW142 did not persist point scores.
    pred=pd.DataFrame({'subject_id':z.loc[good,'subject_id'].to_numpy(),'y':y,'marker_original':marker,'base_linear_score':scores})
    pred.to_parquet(OUT/'h1_auc_predictions.parquet',index=False)
    from scipy.stats import rankdata
    def rank_auc(values):
        ranks=rankdata(values,method='average');n1=int(y.sum());n0=len(y)-n1;u=float(ranks[y==1].sum()-n1*(n1+1)/2)
        return {'auc':u/(n1*n0),'U':u,'pairs':n1*n0}
    result['auc_check']={'marker':rank_auc(marker),'base_B3':rank_auc(scores),'bound_marker_auc':original['auc_marker_original_steigend'],'bound_base_auc':original['auc_G'],'scores_origin':'deterministic reconstruction with frozen scaling/solver; no point predictions in AW142 checkpoints','ties_marker':int(len(marker)-len(np.unique(marker))),'ties_base':int(len(scores)-len(np.unique(scores)))}
    if result['auc_check']['marker']['auc']!=original['auc_marker_original_steigend'] or result['auc_check']['base_B3']['auc']!=original['auc_G']:raise AssertionError('AUC Rangnachrechnung abweichend')
    for name in ('H1_NT','RV_NT'):
        params=package['variants'][name]
        missing=[k for k in ('names','mean','sd','keep') if k not in params]
        if not missing:raise AssertionError('NT Parameter überraschend gebunden; separate Prüfung nötig')
        result['nt'][name]={'status':'nicht_schaetzbar','reason':'nicht schätzbar (Designparameter im Freeze nicht gebunden)','missing_parameters':missing,'package_entry':params,'technical_cause':'estimate_extra greift vor fixed_design auf params[names] zu; vorher kein STOP-Statuscheck; keine Schlüsselalias-Reparatur möglich'}
    save(OUT/'descriptive_checkpoint.json',result)
    h4_ids=frames['BEF-0049'].subject_id.to_numpy().copy()
    del pred,design,x,z,base,marker,scores,frames['BEF-0049'];gc.collect()
    # Union follows the frozen runner: H1/H4/RV index persons plus technical persons.
    rv=manifest['hypotheses']['BEF-0087'];rcols=['subject_id','study_id','split','ecg_time','age','extraction_status','reference_available']
    f=checked(rv,'test',rcols);rv_ids=select_index(f,'echo_rv_systolic_dysfunction').subject_id.to_numpy();del f
    techsource=manifest['technical'];cols=[c for c in techsource['columns'] if c in needed or c in ('has_tte_30d','has_tte_7d')]
    tech=select_index(checked(techsource,'test',cols),'reduced_lvef_strict',technical=True)
    devsource=package['verification_train'];dev=checked(devsource,'train',columns=['subject_id','split','age','sex_numeric','heart_rate','qrs_duration_ms','qtc_fridericia_ms','spatial_qrs_t_angle_deg','classical_cornell_product','classical_sokolow_lyon','magnitude_area_qrs','spatial_low_voltage_qrs_peak_magnitude','classical_right_axis_deviation','has_tte_30d'])
    # Manifest v5 points to train indirectly through the unchanged parameter package.
    if np.intersect1d(dev.subject_id,tech.subject_id).size:raise ValueError('train/test Identitäten überschneiden sich')
    for entry in package['hypotheses'].values():
        ids=checked(entry['input_development'],'development',['subject_id','split'])
        if np.intersect1d(ids.subject_id,tech.subject_id).size:raise ValueError('Entwicklung/technisch Test überschneidet sich')
    union=np.unique(np.concatenate([frames['BEF-0063'].subject_id.to_numpy(),h4_ids,rv_ids,tech.subject_id.to_numpy()]));gc.collect()
    save(OUT/'index_counts.json',{'technical_n':len(tech),'verification_train_n':len(dev),'union_n':len(union),'person_disjoint':True})
    for rank in ('3','3b'):
        cp=OUT/f'rank_{rank}.json'
        if cp.exists():res=json.loads(cp.read_text(encoding='utf8'))
        else:
            print(json.dumps({'phase':'rank_start','rank':rank}),flush=True)
            if rank=='3':res=adapter.ipw_analysis(dev,tech,h1,package['verification'],union,bootstrap=True)
            else:
                try:res=adapter.verification_associations(tech,h1,package['variants']['verification_marker'],union,bootstrap=True)
                except (ValueError,KeyError,AttributeError,RuntimeError,np.linalg.LinAlgError) as exc:res=adapter.stopped(exc)
            save(cp,res)
        result['sensitivities'].append({'rank':rank,**res})
    result.update(utc_end=runner.now(),input_manifest_sha256=MANHASH,parameter_sha256=runner.sha(PARAM),wrapper_sha256=runner.sha(__file__))
    for path,expected in hashes.items():
        if runner.sha(path)!=expected:raise ValueError('Gebundene Datei nach Lauf verändert')
    if runner.sha(OLD)!='f17ef84b6533f71d3d7ed0099e81ed53b684e4e45f8539e3e0a17d0a0e3301a7':raise ValueError('AW142 nach Lauf verändert')
    save(OUT/'ergebnis.json',result)
    print(json.dumps({'phase':'complete','ranks':[(r['rank'],r['status']) for r in result['sensitivities']]}),flush=True)

if __name__=='__main__':
    AUX.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    try:
        with Budget(OUT) as budget:main()
        save(OUT/'resources.json',budget.summary())
    except Exception:
        save(AUX/'error.json',{'utc':runner.now(),'traceback':traceback.format_exc(),'pid':os.getpid()})
        raise
