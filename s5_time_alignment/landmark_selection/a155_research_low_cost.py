"""Bounded, post-hoc research batch. Never alter native runs or sealed outputs."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '2'
import hashlib
import json
import time
import traceback
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import psutil
import duckdb
import statsmodels.api as sm
from audit_tools.onedrive_names import compatible_component

WORK = Path(__file__).resolve().parents[1]
MAIN = Path(str(data_path("s5.mimic_validation_runs_a155_2026_10_04_d6e70e")))
HOLD = MAIN/'publication_preparation/research-holdout-20261011-v2'
OUT = MAIN/'publication_preparation/research-low-cost-20261012-v1'
CONTEXT = Path(str(data_path("s5.mimic_validation_evidence_clinical_context_features_parquet")))

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    value = json.loads(json.dumps(value, default=lambda v:v.item() if isinstance(v,np.generic) else str(v)))
    def clean(v):
        if isinstance(v,float) and not np.isfinite(v): return None
        if isinstance(v,dict): return {k:clean(x) for k,x in v.items()}
        if isinstance(v,list): return [clean(x) for x in v]
        return v
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False),encoding='utf-8')
    for attempt in range(80):
        try: os.replace(tmp,path); return
        except PermissionError:
            if attempt == 79: raise
            time.sleep(.25)

def table(frame, name):
    assert compatible_component(name), name
    path = OUT/name
    tmp = path.with_suffix(path.suffix+'.tmp')
    frame.to_csv(tmp,index=False)
    os.replace(tmp,path)
    return dict(path=str(path),rows=len(frame),sha256=sha(path))

def bh(values):
    p=np.asarray(values,dtype=float); p=np.where(np.isfinite(p),p,1.)
    order=np.argsort(p,kind='stable'); q=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    result=np.empty_like(q); result[order]=np.clip(q,0,1); return result

def ols_cluster(frame, outcome, predictors, cluster='subject_id'):
    """HC cluster covariance, complete cases; no patient rows as independent replicas."""
    names=[outcome,*predictors,cluster]
    if any(c not in frame for c in names):
        return dict(status='unavailable_columns',missing=[c for c in names if c not in frame])
    d=frame[names].copy()
    for c in [outcome,*predictors]:d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.replace([np.inf,-np.inf],np.nan).dropna()
    n=len(d); g=d[cluster].nunique(); k=len(predictors)+1
    base=dict(n=n,n_patients=g,n_input=len(frame),n_removed=len(frame)-n,parameters=k)
    if n<max(100,20*k) or g<50:return dict(**base,status='underpowered')
    X=sm.add_constant(d[predictors],has_constant='add').astype(float)
    scales=X.std().replace(0,1);scales['const']=1
    means=X.mean();means['const']=0
    Z=(X-means)/scales
    if np.linalg.matrix_rank(Z)<k:return dict(**base,status='rank_deficient')
    condition=float(np.linalg.cond(Z))
    if condition>1e8:return dict(**base,status='ill_conditioned',condition=condition)
    fit=sm.OLS(d[outcome].to_numpy(),Z).fit(cov_type='cluster',cov_kwds={'groups':d[cluster].to_numpy(),'use_correction':True},use_t=True)
    ci=fit.conf_int()
    rows=[dict(term=c,beta_raw_units=float(fit.params[c]/scales[c]),standard_error=float(fit.bse[c]/scales[c]),p_value=float(fit.pvalues[c]),ci_low=float(ci.loc[c,0]/scales[c]),ci_high=float(ci.loc[c,1]/scales[c])) for c in predictors]
    return dict(**base,status='estimable',covariance='subject_cluster_small_sample_t',r_squared=float(fit.rsquared),condition=condition,coefficients=rows)

def plan():
    ideas=json.loads((WORK/'docs/betrieb/A155_FORSCHUNGSIDEEN_2026-10-12.json').read_text(encoding='utf-8'))['ideas']
    excluded={'F35':'No validated VT/VF or ICD therapy endpoints; public waveform labels do not supply the proposed preserved-EF outcome cohort.',
              'F36':'No matched pacing mode/programming and longitudinal references identified.',
              'F37':'No matched CMR fibrosis/genotype reference identified.',
              'F38':'Public long-term AF recordings exist, but do not supply the proposed 12-lead VCG/AF-burden pairing; a two-lead rhythm project would be a different question.'}
    ordered=sorted(ideas,key=lambda i:(i['compute_elapsed_hours']['min'],i['compute_elapsed_hours']['max'],i['id']))
    rows=[]
    for i in ordered:
        rows.append(dict(**i,execution_order=len(rows)+1,status='excluded_missing_matched_data' if i['id'] in excluded else 'queued',exclusion_reason=excluded.get(i['id']),estimate_measured=False))
    return dict(created_utc=datetime.now(timezone.utc).isoformat(),scope='Exploratory post-hoc; holdout already inspected; single markers; main remains paused',ordering='Ascending lower compute-hour estimate; upper estimate and ID break ties. Preparation effort is separate.',items=rows)

def atlas():
    c=pd.read_csv(HOLD/'clinical_holdout_adjusted_associations.csv')
    p=pd.read_csv(HOLD/'phewas_holdout_adjusted_associations.csv')
    f=pd.read_parquet(HOLD/'frozen/phewas_family.parquet',columns=['candidate_id','ecg_derived_class','ecg_circularity_claim_ceiling'])
    p=p.merge(f,on='candidate_id',validate='one_to_one')
    gate=p['holdout_association_confirmed'].eq(True)&p['marker_missingness_gate_passed'].eq(True)&p['model_complete_case_gate_passed'].eq(True)&p['ecg_derived_class'].eq('independent_of_ecg')
    coverage=pd.read_csv(HOLD/'holdout_reference_coverage.csv')
    return dict(status='completed',confirmed_clinical=int(c.three_split_effect_replicated.eq(True).sum()),strict_phewas=int(gate.sum()),outputs=[table(c,'F01-clinical-all.csv'),table(p,'F01-phewas-all.csv'),table(p.loc[gate],'F01-phewas-strict.csv'),table(coverage,'F01-reference-coverage.csv')],limitation='All 72/218 tests retained; variants are not independent discoveries; reference alignment remains research-only.')

def comparisons():
    models=[];paired=[];points=[];subgroups=[];windows=[]
    for path in sorted((HOLD/'clinical_checkpoints').glob('*.json')):
        d=json.loads(path.read_text(encoding='utf-8')); key={k:d[k] for k in ('endpoint_id','feature_id')}
        for field,dest in [('predictions',models),('paired_auc',paired),('operating_points',points),('subgroups',subgroups),('windows',windows)]:
            for row in d.get(field,[]):dest.append(dict(**key,**row))
    return dict(status='completed',outputs=[table(pd.json_normalize(rows),f'F02-{name}.csv') for name,rows in [('frozen-models',models),('paired-auc',paired),('train-thresholds',points),('subgroups',subgroups),('windows',windows)]],limitation='Existing paired AUC estimates reused; AP/Brier from different complete-case samples must not be subtracted as paired gains. Intervals are descriptive without multiplicity adjustment.')

def artifact_review(code,folder):
    rows=[];outputs=[]
    for path in sorted((MAIN/folder).glob('*.parquet')):
        if path.stat().st_size>100*1024*1024:continue
        d=pd.read_parquet(path)
        row=dict(source=str(path),source_sha256=sha(path),rows=len(d),n_patients=int(d.subject_id.nunique()) if 'subject_id' in d else None)
        for col in ['status','reason','clinical_claim_allowed','normalization_status','half_peak_crossing_status','abstention_reason','adjustment_status','population','split','drug_class','lab_pair_status']:
            if col in d:row[col+'_counts']=d[col].fillna('<missing>').astype(str).value_counts().to_dict()
        rows.append(row)
        # Aggregate existing scientific results only; no patient-level records in exports.
        if path.name in ['marker_reference_ranges.parquet','cohort_summary.parquet','within_subject_change.parquet','voltage_mass_discordance.parquet','time_to_event.parquet','marker_repeatability.parquet']:
            outputs.append(table(d,f'{code}-{path.stem}.csv'))
        if path.name=='recovery_episode_results.parquet':
            cols=[c for c in ['marker_id','status','normalization_status','half_peak_crossing_status'] if c in d]
            outputs.append(table(d.groupby(cols,dropna=False).size().reset_index(name='n_rows'),f'{code}-recovery-status-counts.csv'))
    return dict(status='preparatory_audit_completed',sources=rows,outputs=outputs,limitation='Inventory and existing results audit only. Additional proposed sensitivity models remain pending. Empty tables are not completed analyses; reference cohorts are not proven healthy.')

def matrix():
    marks=['tpeak_tend_ms','svg_qrs_area_ratio','qrs_t_hausdorff_distance','spatial_qrs_t_angle_deg','p_loop_width_major','p_loop_template_instability','qrs_duration_ms','qt_interval_ms','signal_quality_score','p_wave_coverage','p_fiducial_confidence','extraction_status','rhythm_context']
    ctx=pq.read_schema(CONTEXT).names
    wanted=['subject_id','study_id','age','sex','anchor_year_group','machine_heart_rate_bpm','machine_qtc_fridericia_ms','bmi_before_ecg','machine_cart_id','machine_bandwidth','machine_filtering','preferred_lvef','lv_mass_index_devereux_g_m2','la_volume_index','la_volume_ml','mitral_e_e_prime_ratio','tapse_cm','derived_rvsp_or_pasp_mmhg','nearest_echo_delta_hours','potassium_nearest','potassium_delta_hours','magnesium_nearest','magnesium_delta_hours','total_calcium_nearest','total_calcium_delta_hours','ionized_calcium_nearest','ionized_calcium_delta_hours','creatinine_nearest','creatinine_delta_hours','ntprobnp_nearest','ntprobnp_delta_hours','med_qt_prolonging_drug','med_antiarrhythmic_class_i_iii']
    wanted += [c for c in ctx if c.endswith('_censoring') or c.endswith('_unit')]
    wanted=list(dict.fromkeys(c for c in wanted if c in ctx))
    con=duckdb.connect();con.execute("SET memory_limit='1GB'");con.execute('SET threads=2')
    try:
        feat=con.execute('SELECT subject_id,study_id,ecg_time,'+','.join('"'+c+'"' for c in marks)+" FROM read_parquet(?,union_by_name=true)",[str(HOLD/'features/*.parquet')]).df()
        context=con.execute('SELECT '+','.join('c."'+c+'"' for c in wanted)+' FROM read_parquet(?) c INNER JOIN feat f USING(subject_id,study_id)',[str(CONTEXT)]).df()
    finally:con.close()
    assert not feat.duplicated(['subject_id','study_id']).any()
    d=feat.merge(context,on=['subject_id','study_id'],how='left',validate='one_to_one')
    # Exactly one deterministically selected ECG per patient, independent of values/outcomes.
    d=d.sort_values(['subject_id','ecg_time','study_id'],kind='stable').drop_duplicates('subject_id',keep='first').copy()
    d['sex_male']=d.sex.map({'M':1.,'F':0.})
    rows=[]
    for c in d.columns:
        v=pd.to_numeric(d[c],errors='coerce');finite=np.isfinite(v)
        row=dict(column=c,n_total=len(d),n_present=int(d[c].notna().sum()),n_numeric_finite=int(finite.sum()))
        if finite.any():row.update(dict(q01=float(v[finite].quantile(.01)),median=float(v[finite].median()),q99=float(v[finite].quantile(.99))))
        rows.append(row)
    table(pd.DataFrame(rows),'reference-and-feature-coverage.csv')
    atomic(OUT/'matrix-selection.json',dict(input_ecgs=len(feat),selected_patients=len(d),rule='First ECG by ecg_time/study_id in existing selected holdout feature union; no outcome-conditioned selection',context_sha256=sha(CONTEXT),context_join='subject_id and study_id, validated one-to-one',limitation='Selected holdout union, not full MIMIC population; no additional independent holdout. All exploratory.'))
    return d

def fixed_models(d, code):
    # Explicit models, no marker combinations, no threshold optimization.
    base=['age','sex_male','machine_heart_rate_bpm']
    definitions={
      'F03':[('tpeak_tend_ms',['potassium_nearest',*base], 'potassium'),('tpeak_tend_ms',['potassium_nearest','potassium_squared',*base],'potassium')],
      'F04':[('tpeak_tend_ms',['potassium_nearest','machine_qtc_fridericia_ms','med_qt_prolonging_drug','med_antiarrhythmic_class_i_iii',*base],'potassium')],
      'F05':[('tpeak_tend_ms',['potassium_nearest',c,'potassium_x_'+c,*base],'potassium+'+c.split('_')[0]) for c in ['magnesium_nearest','total_calcium_nearest','ionized_calcium_nearest']],
      'F10':[('lv_mass_index_devereux_g_m2',['svg_qrs_area_ratio',*base],'echo')],
      'F14':[('log_ntprobnp',['spatial_qrs_t_angle_deg','bmi_before_ecg','creatinine_nearest',*base],'ntprobnp+creatinine')],
      'F15':[('preferred_lvef',['qrs_t_hausdorff_distance',*base],'echo')],
      'F19':[('la_volume_index',['p_loop_width_major',*base],'echo'),('la_volume_ml',['p_loop_width_major',*base],'echo')],
      'F13':[('mitral_e_e_prime_ratio',['spatial_qrs_t_angle_deg',*base],'echo_preserved')],
      'F16':[('tapse_cm',['spatial_qrs_t_angle_deg',*base],'echo'),('derived_rvsp_or_pasp_mmhg',['spatial_qrs_t_angle_deg',*base],'echo')],
    }
    x=d.copy(); x['potassium_squared']=pd.to_numeric(x.potassium_nearest,errors='coerce')**2
    for c in ['magnesium_nearest','total_calcium_nearest','ionized_calcium_nearest']:
        x['potassium_x_'+c]=pd.to_numeric(x.potassium_nearest,errors='coerce')*pd.to_numeric(x[c],errors='coerce')
    v=pd.to_numeric(x.ntprobnp_nearest,errors='coerce');x['log_ntprobnp']=np.log(v.where(v>0))
    def gated(x,gate):
        keep=pd.Series(True,index=x.index)
        if gate.startswith('echo'):
            keep &= pd.to_numeric(x.nearest_echo_delta_hours,errors='coerce').abs().le(24*30)
            if gate=='echo_preserved':keep &= pd.to_numeric(x.preferred_lvef,errors='coerce').ge(50)
        else:
            labs={'potassium':'potassium','magnesium':'magnesium','total':'total_calcium','ionized':'ionized_calcium','ntprobnp':'ntprobnp','creatinine':'creatinine'}
            for token in gate.split('+'):
                lab=labs[token];keep &= pd.to_numeric(x[lab+'_delta_hours'],errors='coerce').abs().le(6)
                censor=lab+'_nearest_censoring'
                if censor in x:
                    # Censored measurements must not be treated as exact quantitative values.
                    keep &= x[censor].fillna('').astype(str).str.lower().isin(['','none','exact','uncensored','not_censored','nan'])
        return x.loc[keep.fillna(False)]
    fits=[]
    for idx,(y,p,gate) in enumerate(definitions[code]):
        g=gated(x,gate);fit=ols_cluster(g,y,p)
        fits.append(dict(model_id=f'{code}-{idx+1:02d}',outcome=y,predictors=p,reference_gate=gate,n_time_eligible=len(g),**fit))
    rows=[dict(model_id=f['model_id'],outcome=f['outcome'],n=f.get('n'),n_patients=f.get('n_patients'),status=f['status'],**r) for f in fits for r in f.get('coefficients',[])]
    if rows:
        df=pd.DataFrame(rows);df['q_bh_within_idea_all_slopes']=bh(df.p_value)
        outputs=[table(df,f'{code}-fixed-model-coefficients.csv')]
    else:outputs=[]
    return dict(status='bounded_models_completed',fits=fits,outputs=outputs,limitation='Fixed exploratory OLS with patient-cluster intervals; one index ECG per selected holdout patient. No causal or clinical release claim. BH applies only to reported prespecified slopes within this idea; rank-deficient/underpowered models abstain. Broader original idea, train/validation repetition and sensitivity work remain pending.')

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    import msvcrt
    with (OUT/'run.lock').open('a+b') as lock:
        lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        execute()

def execute():
    if not json.loads((MAIN/'user_pause.control.json').read_text())['paused']:raise RuntimeError('Main is not user-paused')
    p=plan();atomic(OUT/'execution-plan.json',p)
    source_archive=OUT/'executed_source';source_archive.mkdir(exist_ok=True)
    for name in ['a155_research_low_cost.py','onedrive_names.py']:
        target=source_archive/name;source=WORK/'audit_tools'/name
        if target.exists() and sha(target)!=sha(source):raise RuntimeError('Executed source changed: create new revision')
        if not target.exists():target.write_bytes(source.read_bytes())
    implementations={'F01':atlas,'F02':comparisons,'F27':lambda:artifact_review('F27','reference_ranges'),
      'F06':lambda:artifact_review('F06','within_subject'),'F07':lambda:artifact_review('F07','therapy_response'),
      'F08':lambda:artifact_review('F08','within_subject'),'F11':lambda:artifact_review('F11','voltage_mass'),
      'F20':lambda:artifact_review('F20','within_subject'),'F22':lambda:artifact_review('F22','repeatability'),
      'F33':lambda:artifact_review('F33','survival')}
    scalar={'F03','F04','F05','F10','F14','F15','F19','F13','F16'}
    matrix_cache=None
    for item in p['items']:
        code=item['id'];checkpoint=OUT/'checkpoints'/f'{code}.json'
        if checkpoint.exists():
            result=json.loads(checkpoint.read_text(encoding='utf-8'))
            for output in result.get('outputs',[]):
                if sha(Path(output['path']))!=output['sha256']:raise RuntimeError('Checkpoint output integrity mismatch')
            item['status']=result['status'];continue
        if code not in implementations and code not in scalar:continue
        while psutil.virtual_memory().available<4*1024**3:
            atomic(OUT/'status.json',dict(status='memory_pause',current_idea=code,required_free_gib=4));time.sleep(10)
        start=time.monotonic();atomic(OUT/'status.json',dict(status='running',current_idea=code,started_utc=datetime.now(timezone.utc).isoformat()))
        try:
            if code in scalar:
                if matrix_cache is None:matrix_cache=matrix()
                result=fixed_models(matrix_cache,code)
            else:result=implementations[code]()
            result.update(dict(id=code,title=item['title'],elapsed_seconds=time.monotonic()-start,posthoc=True,clinical_release_eligible=False,finished_utc=datetime.now(timezone.utc).isoformat()))
            atomic(checkpoint,result);item['status']=result['status'];item['measured_elapsed_seconds']=result['elapsed_seconds']
        except Exception:
            item['status']='failed';item['traceback']=traceback.format_exc()
            atomic(OUT/'failure.json',item)
            atomic(OUT/'execution-plan.json',p)
            atomic(OUT/'status.json',dict(status='failed',current_idea=code,traceback=item['traceback']))
            raise
        atomic(OUT/'execution-plan.json',p)
        print(code,item['status'],round(result['elapsed_seconds'],2),flush=True)
    atomic(OUT/'status.json',dict(status='implemented_batch_completed',completed_checkpoints=len(list((OUT/'checkpoints').glob('*.json'))),remaining_ideas=[i['id'] for i in p['items'] if i['status']=='queued'],full_38_idea_program_completed=False,finished_utc=datetime.now(timezone.utc).isoformat()))

if __name__=='__main__':run()
