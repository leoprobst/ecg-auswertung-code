"""Bounded post-discharge documented-AF association; broad Future arm untouched."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import sys,json,re,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
from audit_tools import a155_research_low_cost as b

OUT=b.MAIN/'publication_preparation/research-documented-af-20261012-v1'
RUNTIME=Path(str(data_path("s5.mimic_validation_jobs_runtime_snapshots_session_i6wl24th")))
POP=b.MAIN/'publication_preparation/research-rest-low-cost-20261012-v1/population-first-ecg.parquet'

def report_sinus(text):
    if text is None or text is pd.NA:return False
    text=str(text).lower()
    positive=re.search(r'\bsinus\s+(rhythm|tachycardia|bradycardia|arrhythmia)\b',text) is not None
    negative=re.search(r'\bno(?:\s+longer)?\s+sinus|sinus[^\n]{0,25}(?:has been replaced|not present)|\b(?:possible|probable|uncertain)[^\n]{0,20}\bsinus',text) is not None
    return positive and not negative

def risk_set(d,horizon):
    x=d.copy();landmark=(pd.to_datetime(x.dischtime)-pd.to_datetime(x.ecg_time)).dt.total_seconds()/3600
    first=pd.to_numeric(x.first_af_event_delta_hours,errors='coerce').astype(float)
    future=pd.to_numeric(x.next_af_event_delta_hours,errors='coerce').astype(float)
    follow=pd.to_numeric(x.followup_hours,errors='coerce').astype(float)
    death=(pd.to_datetime(x.date_of_death)-pd.to_datetime(x.ecg_time)).dt.total_seconds()/3600 if 'date_of_death' in x else pd.to_numeric(x.death_delta_days,errors='coerce').astype(float)*24
    stage=[('input',pd.Series(True,index=x.index))]
    gates=[('usable_metadata',x.machine_metadata_usable_for_primary.eq(True)),
           ('reported_sinus_regular',x.machine_report.map(report_sinus)&x.rhythm_context.eq('regular')),
           ('no_prior_or_index_coded_af',x.af_dx_prior_admission.eq(False)&x.af_dx_index_admission.eq(False)&x.current_report_af_state.eq('not_mentioned')),
           ('discharge_0_to_7_days',landmark.between(0,168)),
           ('no_known_af_through_discharge',first.isna()|first.gt(landmark)),
           ('consistent_documented_event_coverage',future.isna()|(future.le(follow)&(death.isna()|future.lt(death)))),
           ('positive_observed_postdischarge_time',follow.gt(landmark)&(death.isna()|death.gt(landmark)))]
    keep=pd.Series(True,index=x.index);flow=[]
    for name,gate in gates:
        keep &= gate.fillna(False);flow.append(dict(gate=name,n_retained=int(keep.sum())))
    # AF ascertainment ends at last observed contact, never the 365-day mortality registry window.
    censor_hours=np.minimum(follow-landmark,np.where(death.notna(),death-landmark,np.inf))
    event_hours=future-landmark
    event=event_hours.gt(0)&event_hours.le(censor_hours)&event_hours.le(horizon*24)
    time_hours=np.where(event,event_hours,np.minimum(censor_hours,horizon*24))
    x['time_days']=time_hours/24;x['event']=event.fillna(False).astype(int)
    keep &= pd.Series(np.isfinite(x.time_days)&x.time_days.gt(0),index=x.index)
    x=x.loc[keep.fillna(False)].copy()
    return x,flow

def run():
    OUT.mkdir(parents=True,exist_ok=True);b.OUT=OUT
    if not json.loads((b.MAIN/'user_pause.control.json').read_text())['paused']:raise RuntimeError('Main must remain paused')
    sys.path.insert(0,str(RUNTIME))
    from vcg_biomarker.mimic.survival import fit_cox
    protocol=dict(kind='Post-hoc association with subsequently documented AF after discharge; not adjudicated incident AF or calibrated probability prediction',horizons_days=[30,90,365],index='First master ECG per patient, reported sinus and waveform regular; metadata usable; ECG 0-7 days before discharge',history='No prior/index admission coded AF or current AF report; no first documented AF through discharge',time_origin='Index hospital discharge',censoring='Minimum last observed clinical contact, recorded death, horizon; no mortality-registry extension of AF follow-up',marker='p_loop_width_major',covariates=['age','sex_male','machine_heart_rate_bpm'],scaling='Train complete-case at-risk population, fixed across Validation/Test',minimum_events_per_parameter=20,broad_future_arm_enabled=False,clinical_release_eligible=False,holdout_already_seen=True)
    path=OUT/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise RuntimeError('Protocol changed')
    if not path.exists():b.atomic(path,protocol)
    d=pd.read_parquet(POP)
    extra=['dischtime','first_af_event_delta_hours','next_af_event_delta_hours','next_af_event_source','af_dx_prior_admission','af_dx_index_admission','current_report_af_state','followup_hours','death_delta_days','date_of_death']
    con=duckdb.connect();con.execute("SET memory_limit='1GB'");con.execute('SET threads=2')
    try:
        ctx=con.execute('SELECT c.subject_id,c.study_id,'+','.join('c."'+c+'"' for c in extra)+' FROM read_parquet(?) c INNER JOIN d USING(subject_id,study_id)',[str(b.CONTEXT)]).df()
        master=b.MAIN.parents[1]/'master/ecg_master_index.parquet'
        reports=con.execute('SELECT m.subject_id,m.study_id,m.machine_report FROM read_parquet(?) m INNER JOIN d USING(subject_id,study_id)',[str(master)]).df()
    finally:con.close()
    d=d.merge(ctx,on=['subject_id','study_id'],validate='one_to_one').merge(reports,on=['subject_id','study_id'],validate='one_to_one')
    results=[];rows=[];flowrows=[];folder=OUT/'checkpoints';folder.mkdir(exist_ok=True)
    predictors=['p_loop_width_major','age','sex_male','machine_heart_rate_bpm']
    for horizon in protocol['horizons_days']:
        risk,flow=risk_set(d,horizon);flowrows.extend(dict(horizon_days=horizon,**row) for row in flow)
        numeric=risk[predictors].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
        train=numeric.loc[risk.split.eq('train')].dropna();means=train.mean();scales=train.std(ddof=0);means['sex_male']=0;scales['sex_male']=1
        for split in ['train','validation','locked_test']:
            path=folder/f'{horizon:03d}-{split}.json'
            if path.exists():result=json.loads(path.read_text())
            else:
                x=risk.loc[risk.split.eq(split)].copy();x[predictors]=numeric.loc[x.index];x=x.dropna(subset=predictors)
                result=dict(horizon_days=horizon,split=split,n_input=int(risk.split.eq(split).sum()),n_complete=len(x),n_events=int(x.event.sum()),parameters=len(predictors),scaling_means=means.to_dict(),scaling_sd=scales.to_dict(),status='underpowered')
                if result['n_events']>=20*len(predictors) and len(x)>100 and scales.gt(0).all():
                    x[predictors]=(x[predictors]-means)/scales
                    b.atomic(OUT/'status.json',dict(state='fitting',horizon=horizon,split=split,completed=len(results),total=9))
                    try:
                        fit=fit_cox(x,predictors,robust=True);result.update(status='estimable',fit=fit.to_dict(),coefficients=fit.to_frame().to_dict('records'))
                    except (ValueError,RuntimeError,np.linalg.LinAlgError) as error:result.update(status='not_estimable',reason=str(error))
                b.atomic(path,result)
            results.append(result)
            for coefficient in result.get('coefficients',[]):rows.append(dict(horizon_days=horizon,split=split,n=result['n_complete'],n_events=result['n_events'],**coefficient))
    outputs=[b.table(pd.DataFrame(flowrows),'F21-F34-postdischarge-risk-flow.csv')]
    if rows:
        frame=pd.DataFrame(rows);frame['q_bh_complete_36_slope_family']=b.bh(list(frame.p_value)+[np.nan]*(36-len(frame)))[:len(frame)]
        outputs.append(b.table(frame,'F34-documented-af-cox-associations.csv'))
    census=pd.DataFrame([{k:v for k,v in result.items() if k not in ['coefficients','fit','scaling_means','scaling_sd']} for result in results]);outputs.append(b.table(census,'F34-documented-af-model-census.csv'))
    b.atomic(OUT/'summary.json',dict(models=len(results),estimable=sum(r['status']=='estimable' for r in results),outputs=outputs,population_source_sha256=b.sha(POP),context_source_sha256=b.sha(b.CONTEXT),clinical_release_eligible=False,scope='Observed AF documentation, not biological onset; discharge landmark, observation/censoring assumptions, device/EHR alignment unconfirmed; post-hoc Cox association only. Calibration and prospective independent validation remain absent.'))
    b.atomic(OUT/'status.json',dict(state='completed',completed=9,total=9))

if __name__=='__main__':
    import msvcrt
    OUT.mkdir(parents=True,exist_ok=True);archive=OUT/'executed_source';archive.mkdir(exist_ok=True)
    for name in ['a155_research_documented_af.py','a155_research_low_cost.py','onedrive_names.py']:
        source=b.WORK/'audit_tools'/name;target=archive/name
        if target.exists() and b.sha(source)!=b.sha(target):raise RuntimeError('Executed source changed')
        if not target.exists():target.write_bytes(source.read_bytes())
    with (OUT/'run.lock').open('a+b') as lock:
        lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:run()
        except Exception:
            b.atomic(OUT/'failure.json',dict(traceback=traceback.format_exc()));b.atomic(OUT/'status.json',dict(state='failed'));raise
