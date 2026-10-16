"""Echo-Gegenkontrollen erst nach Bindung des Verifikationsmodells öffnen."""
import json
import numpy as np
import pandas as pd
from .runner import DATA,ROOT,PROJECT,connect,sha,save,small,cache_path
from .models import BASE,EXTRA,MARKER,RV,b4,Fit,compare
from werkzeuge.nachrechnung.core import assert_development

CONTEXT=['subject_id','study_id','split','preferred_lvef','lvef_conflict_gt_10','nearest_tte_measurement_id','nearest_tte_delta_hours','lv_mass_index','lv_mass_index_devereux_g_m2','echo_body_surface_area_m2','rv_function_category','tapse_cm','current_report_pacemaker_state','current_report_af_state','bmi_before_ecg']

def extract(out):
    import pyarrow.parquet as pq
    frozen=json.loads((out/'weight_model_frozen.sha256.json').read_text(encoding='utf8'))['sha256']
    assert sha(out/'weight_model_frozen.json')==frozen
    c=connect(out);paths=','.join("'"+(out/f'features_{s}.parquet').as_posix()+"'" for s in ('train','validation'))
    cache=cache_path(json.loads((out/'weight_model_frozen.json').read_text(encoding='utf8'))['run_id'])
    rvpaths=','.join("'"+(cache/f'split={s}/features.parquet').as_posix()+"'" for s in ('train','validation'))
    # Exakter registergewählter RV-B2-Komparator zusätzlich zur B4-Liste.
    c.execute(f"CREATE VIEW features AS SELECT f.*,r.classical_dominant_r_v1 FROM read_parquet([{paths}],union_by_name=true) f LEFT JOIN (SELECT subject_id,study_id,classical_dominant_r_v1 FROM read_parquet([{rvpaths}]) WHERE split IN ('train','validation')) r USING(subject_id,study_id)")
    context=DATA/'evidence/clinical_context_features.parquet'
    # projected and development-filtered; LVEF first accessed here.
    c.execute(f"CREATE VIEW context AS SELECT {','.join(CONTEXT)},age,sex,ecg_time,machine_metadata_usable_for_primary,machine_heart_rate_bpm FROM read_parquet('{context.as_posix()}') WHERE split IN ('train','validation')")
    att=json.loads((DATA/'labels/development_label_attestation.json').read_text(encoding='utf8'))
    for key in ('clinical_label_evidence','endpoint_index_cohorts'):
        p=DATA/'labels'/f'{key}.parquet';pq.ParquetFile(p)
        assert sha(p)==att['development_artifacts'][key]['sha256']
    evidence=DATA/'labels/clinical_label_evidence.parquet';endpoint=DATA/'labels/endpoint_index_cohorts.parquet'
    c.execute(f"CREATE VIEW evidence AS SELECT subject_id,study_id,split,label_id,label_state,eligibility_state,time_delta_hours,time_precision,evidence_ids_json,ehr_time_alignment_confirmatory_eligible FROM read_parquet('{evidence.as_posix()}') WHERE split IN ('train','validation') AND label_id IN ('reduced_lvef_strict','echo_lvh_strict','echo_rv_systolic_dysfunction')")
    for ep in ('reduced_lvef_strict','echo_rv_systolic_dysfunction'):
        # Index definition from §1 retained; no outcome-specific re-indexing.
        q=f"""SELECT f.*,c.age,c.sex,CASE WHEN c.sex='M' THEN 1 WHEN c.sex='F' THEN 0 END sex_numeric,
          CASE WHEN c.machine_metadata_usable_for_primary THEN c.machine_heart_rate_bpm END heart_rate,{','.join('c.'+x for x in CONTEXT if x not in ('subject_id','study_id','split'))},
          idx.label_state,idx.eligibility_state,idx.index_selection_strategy,idx.time_precision,idx.time_delta_hours,idx.evidence_ids_json,idx.ehr_time_alignment_confirmatory_eligible
          FROM read_parquet('{endpoint.as_posix()}') idx JOIN features f USING(subject_id,study_id,split) JOIN context c USING(subject_id,study_id,split)
          WHERE idx.split IN ('train','validation') AND idx.label_id='{ep}' AND c.age>=18"""
        c.execute(f"COPY ({q}) TO '{(out/(ep+'_private.parquet')).as_posix()}' (FORMAT PARQUET, ROW_GROUP_SIZE 4096)")
    # First-technical index for coherent verification/IPW and LVH feasibility.
    q=f"SELECT p.*, {','.join('c.'+x for x in CONTEXT if x not in ('subject_id','study_id','split','nearest_tte_measurement_id','nearest_tte_delta_hours'))} FROM read_parquet('{(out/'preindex.parquet').as_posix()}') p JOIN context c USING(subject_id,study_id,split)"
    c.execute(f"COPY ({q}) TO '{(out/'clinical_first_technical_private.parquet').as_posix()}' (FORMAT PARQUET, ROW_GROUP_SIZE 4096)")
    q=f"""SELECT f.*,c.age,c.sex,CASE WHEN c.sex='M' THEN 1 WHEN c.sex='F' THEN 0 END sex_numeric,
      CASE WHEN c.machine_metadata_usable_for_primary THEN c.machine_heart_rate_bpm END heart_rate,c.ecg_time,{','.join('c.'+x for x in CONTEXT if x not in ('subject_id','study_id','split'))}
      FROM features f JOIN context c USING(subject_id,study_id,split)
      WHERE f.extraction_status='ok' AND c.age>=18 AND f.classical_lbbb_pattern_positive IS NOT NULL
      QUALIFY row_number() OVER(PARTITION BY subject_id ORDER BY ecg_time,study_id)=1"""
    c.execute(f"COPY ({q}) TO '{(out/'lvh_index_private.parquet').as_posix()}' (FORMAT PARQUET, ROW_GROUP_SIZE 4096)")
    wanted=f"SELECT study_id FROM read_parquet('{(out/'clinical_first_technical_private.parquet').as_posix()}') UNION SELECT study_id FROM read_parquet('{(out/'reduced_lvef_strict_private.parquet').as_posix()}') UNION SELECT study_id FROM read_parquet('{(out/'echo_rv_systolic_dysfunction_private.parquet').as_posix()}') UNION SELECT study_id FROM read_parquet('{(out/'lvh_index_private.parquet').as_posix()}')"
    c.execute(f"COPY (SELECT e.* FROM evidence e WHERE study_id IN ({wanted})) TO '{(out/'echo_evidence_private.parquet').as_posix()}' (FORMAT PARQUET, ROW_GROUP_SIZE 4096)")
    c.close();save(out/'clinical_binding.json',{'weight_model_before_LVEF_sha256':frozen,'attestation_sha256':sha(DATA/'labels/development_label_attestation.json'),
      'sources':{str(DATA/'labels'/f'{k}.parquet'):att['development_artifacts'][k]['sha256'] for k in ('clinical_label_evidence','endpoint_index_cohorts')},
      'projections_sha256':{p.name:sha(p) for p in out.glob('*private.parquet')},'filter':'train/validation; label_id IN reduced_lvef_strict,echo_lvh_strict,echo_rv_systolic_dysfunction; age>=18'})

def reference(f,e,endpoint):
    selected=e[e.label_id.eq(endpoint)].drop(columns=['subject_id','split','label_id']).rename(columns={c:endpoint+'__'+c for c in e.columns if c not in ('subject_id','study_id','split','label_id')})
    assert not selected.study_id.duplicated().any()
    return f.merge(selected,on='study_id',how='left',validate='one_to_one')

def eligible(f,ep,window=720):
    s=ep+'__'
    # Confirmatory alignment is FALSE throughout current development evidence.
    # Development pilot uses attested eligible timestamp labels; alignment remains a claim limitation.
    good=f[s+'label_state'].isin(['positive','negative'])&f[s+'eligibility_state'].eq('eligible')&f[s+'time_precision'].eq('timestamp')&f[s+'time_delta_hours'].abs().le(window)
    # Evidence ID and nearest TTE must match, no mixed echoes across fields.
    def match(row):
        def canonical(x):
            from decimal import Decimal,InvalidOperation
            try:return str(Decimal(str(x)).normalize())
            except InvalidOperation:return str(x)
        try:return canonical(row.nearest_tte_measurement_id) in [canonical(x) for x in json.loads(row[s+'evidence_ids_json'])]
        except (TypeError,ValueError):return False
    ids=f.apply(match,axis=1)
    same=np.isclose(pd.to_numeric(f[s+'time_delta_hours'],errors='coerce'),pd.to_numeric(f.nearest_tte_delta_hours,errors='coerce'),atol=1e-9,rtol=0,equal_nan=False)
    return good&ids&same

def complete(f,marker):
    return np.isfinite(f[BASE+EXTRA+[marker]].apply(pd.to_numeric,errors='coerce')).all(axis=1)

def run(out):
    extract(out);e=small(out/'echo_evidence_private.parquet');results={};flows={}
    f=small(out/'reduced_lvef_strict_private.parquet');assert_development(f);assert not f.subject_id.duplicated().any()
    f=reference(reference(f,e,'reduced_lvef_strict'),e,'echo_lvh_strict')
    ref=eligible(f,'reduced_lvef_strict');known=ref&~f.lvef_conflict_gt_10.fillna(True)&np.isfinite(f.preferred_lvef)
    good=complete(f,MARKER);z=f[known&good].copy()
    assert set(z.index_selection_strategy)=={'first_verified_ecg_reference_result_blind'}
    for mode in ('strict_lt40','inclusive_le40'):
        y=(z.preferred_lvef<40 if mode=='strict_lt40' else z.preferred_lvef<=40).to_numpy(float)
        if mode=='strict_lt40':assert np.array_equal(y,z.label_state.eq('positive').to_numpy(float))
        results[mode]=compare(z,y,z[MARKER].to_numpy())
    lvhref=eligible(f,'echo_lvh_strict')
    reduced=known&f.preferred_lvef.le(40);hypertrophy=known&lvhref&f['echo_lvh_strict__label_state'].eq('positive')&f.preferred_lvef.ge(50)
    z=f[(reduced|hypertrophy)&good].copy();y=z.preferred_lvef.le(40).to_numpy(float)
    results['control_LVEF40_vs_LVH50']=compare(z,y,z[MARKER].to_numpy())
    flows['LVEF']={s:{'index_n':int(m.sum()),'known_reference_n':int((m&known).sum()),'complete_n':int((m&known&good).sum()),'exactly40':int((m&known&f.preferred_lvef.eq(40)).sum()),'reduced40':int((m&reduced).sum()),'LVH_preserved50':int((m&hypertrophy).sum())} for s in ('train','validation') for m in [f.split.eq(s)]}
    rv=small(out/'echo_rv_systolic_dysfunction_private.parquet');assert_development(rv);assert not rv.subject_id.duplicated().any()
    rv=reference(reference(rv,e,'echo_rv_systolic_dysfunction'),e,'reduced_lvef_strict')
    valid=eligible(rv,'echo_rv_systolic_dysfunction')&eligible(rv,'reduced_lvef_strict')&~rv.lvef_conflict_gt_10.fillna(True)&rv.preferred_lvef.ge(50)
    goodrv=np.isfinite(rv.classical_dominant_r_v1)
    z=rv[valid&complete(rv,RV)&goodrv].copy();y=z['echo_rv_systolic_dysfunction__label_state'].eq('positive').to_numpy(float)
    results['control_RV_LVEF50']=compare(z,y,z[RV].to_numpy(),comparator='classical_dominant_r_v1')
    # Same RV sensitivity with primary coupling marker, separately descriptive.
    z=rv[valid&complete(rv,MARKER)&goodrv].copy();y=z['echo_rv_systolic_dysfunction__label_state'].eq('positive').to_numpy(float)
    results['control_RV_LVEF50_primary_frechet']=compare(z,y,z[MARKER].to_numpy(),comparator='classical_dominant_r_v1')
    flows['RV']={s:{'index_n':int(m.sum()),'preserved_eligible_n':int((m&valid).sum())} for s in ('train','validation') for m in [rv.split.eq(s)]}
    save(out/'clinical_flow.json',flows)
    save(out/'clinical_results.json',results)
    print(json.dumps({'phase':'clinical_completed','results':{k:{a:v.get(a) for a in ('status','counts','delta_auc','or_per_train_sd')} for k,v in results.items()}}),flush=True)

def ipw(out):
    from werkzeuge.nachrechnung.resources import available
    v=small(out/'verification_private.parquet');f=small(out/'clinical_first_technical_private.parquet');e=small(out/'echo_evidence_private.parquet')
    f=reference(f,e,'reduced_lvef_strict');good=eligible(f,'reduced_lvef_strict')&complete(f,MARKER)&~f.lvef_conflict_gt_10.fillna(True)&np.isfinite(f.preferred_lvef)
    f=f[good].merge(v[['subject_id','study_id','split','ipw']],on=['subject_id','study_id','split'],validate='one_to_one')
    x=b4(v);yver=v.verification.to_numpy(float);tr=v.split.eq('train').to_numpy();va=~tr;xf=b4(f)
    index=pd.Index(v.subject_id);positions=index.get_indexer(f.subject_id);assert (positions>=0).all()
    counter=[0]
    def boot(rng):
        counter[0]+=1
        if counter[0]%100==0:
            print(json.dumps({'phase':'ipw_bootstrap','replicate':counter[0],'memory':available()}),flush=True)
        # Shared subject draws in universe: outcome and verification use same multiplicities.
        w=np.zeros(len(v))
        for m in (tr,va):
            ids=np.flatnonzero(m);w+=np.bincount(rng.choice(ids,len(ids),replace=True),minlength=len(v))
        fit=Fit().fit(x[tr],yver[tr],w[tr]);p=fit.prob(x)
        raw=1/p;verified=tr&(yver==1)
        # Weighted empirical P99 equivalent to duplicated train-subject rows.
        take=np.repeat(raw[verified],w[verified].astype(int));cap=float(np.quantile(take,.99))
        propensity=fit.prob(xf);return w[positions]*np.minimum(1/propensity,cap)
    y=f.preferred_lvef.le(40).to_numpy(float)
    unweighted=compare(f,y,f[MARKER].to_numpy())
    weighted=compare(f,y,f[MARKER].to_numpy(),weights=f.ipw,bootstrap_weights=boot)
    weighted['ci_target']='Patienten-unstratifiziert train/validation; gemeinsame Multiplizitäten, Verifikationsfit/Skalierung/Train-P99 und LVEF-Fits in jedem Replikat neu geschätzt'
    weighted['index_policy']='erstes technisch auswertbares Erwachsenen-EKG, danach eligible Echo; kohärentes IPW-Fallset; kein Ersatz für primären first_verified-Index'
    weighted['weight_model_sha256']=sha(out/'weight_model_frozen.json')
    weighted['weight_sum']=float(f.ipw.sum());weighted['effective_n']=float(f.ipw.sum()**2/(f.ipw**2).sum())
    save(out/'ipw_result.json',{'unweighted_same_first_technical':unweighted,'IPW':weighted,'direction_reversed':weighted.get('direction')!=unweighted.get('direction'),'limitations':'MAR und ungemessene Echo-Indikation offen; Complete-case-Selektion zusätzlich; keine Repräsentativitätsbehauptung'})
    print(json.dumps({'phase':'ipw_completed','n':len(f),'delta_auc':weighted.get('delta_auc'),'OR':weighted.get('or_per_train_sd')}),flush=True)
