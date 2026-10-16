"""Feasibility design sensitivity: first eligible discharge, ECG latest before it."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import inspect,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
from audit_tools import a155_research_documented_af as original

ROOT=original.b.MAIN/'publication_preparation/research-documented-af-landmark-20261012-v2'
CACHE=Path(str(data_path("s5.vcg_analysis_cache_a155_2026_10_06_single_markers_cache_89d6aaf0a6b5957e")))

def select_landmarks(d):
    landmark=(pd.to_datetime(d.dischtime)-pd.to_datetime(d.ecg_time)).dt.total_seconds()/3600
    first=pd.to_numeric(d.first_af_event_delta_hours,errors='coerce').astype(float)
    keep=d.machine_metadata_usable_for_primary.eq(True)&d.machine_report.map(original.report_sinus)&d.rhythm_context.eq('regular')
    keep &= d.af_dx_prior_admission.eq(False)&d.af_dx_index_admission.eq(False)&d.current_report_af_state.eq('not_mentioned')
    keep &= landmark.between(0,168)&(first.isna()|first.gt(landmark))
    # Never use marker availability, future event status or future follow-up to choose the index.
    return d.loc[keep.fillna(False)].sort_values(['subject_id','dischtime','ecg_time','study_id'],ascending=[True,True,False,True],kind='stable').drop_duplicates('subject_id').reset_index(drop=True)

def cohort():
    path=ROOT/'first-eligible-landmarks.parquet'
    if path.exists():
        if original.b.sha(path)!=json.loads((ROOT/'first-eligible-landmarks.sha256.json').read_text())['sha256']:raise RuntimeError('Landmark cohort changed')
        return pd.read_parquet(path)
    con=duckdb.connect();con.execute("SET memory_limit='1GB'");con.execute('SET threads=2')
    columns=['age','sex','ecg_time','dischtime','machine_report','machine_metadata_usable_for_primary','machine_heart_rate_bpm','first_af_event_delta_hours','next_af_event_delta_hours','next_af_event_source','af_dx_prior_admission','af_dx_index_admission','current_report_af_state','followup_hours','death_delta_days','date_of_death']
    try:
        original.b.atomic(ROOT/'status.json',dict(state='building_first_eligible_discharge_cohort'))
        query='SELECT f.subject_id,f.study_id,f.split,f.p_loop_width_major,f.rhythm_context,'+','.join('c."'+v+'"' for v in columns)+" FROM read_parquet(?,hive_partitioning=true) f JOIN read_parquet(?) c USING(subject_id,study_id)"
        d=con.execute(query,[str(CACHE/'split=*/features.parquet'),str(original.b.CONTEXT)]).df()
    finally:con.close()
    if not set(d.split.unique())<={'train','validation'}:raise RuntimeError('Unexpected split')
    n=len(d);result=select_landmarks(d);result['sex_male']=result.sex.map({'M':1.,'F':0.})
    result.to_parquet(path,index=False)
    original.b.atomic(ROOT/'first-eligible-landmarks.sha256.json',dict(sha256=original.b.sha(path),n_candidate_ecgs=n,n_eligible_patient_landmarks=len(result),selection_uses_marker_value=False,selection_uses_postlandmark_outcome=False,selection_uses_postlandmark_followup=False,heldout_scope='Unavailable for equivalent complete candidate selection; selected holdout extraction is incomplete for all later eligible ECGs'))
    return result

def install():
    original.OUT=ROOT;original.POP=ROOT/'first-eligible-landmarks.parquet'
    source=inspect.getsource(original.run)
    start=source.index('    d=pd.read_parquet(POP)')
    end=source.index('    results=[];',start)
    source=source[:start]+'    d=cohort()\n'+source[end:]
    source=source.replace("'First master ECG per patient, reported sinus and waveform regular; metadata usable; ECG 0-7 days before discharge'","'First eligible discharge per patient; latest qualifying sinus/regular ECG 0-7 days before discharge; marker availability and subsequent outcomes unused in selection'")
    source=source.replace("['train','validation','locked_test']","['train','validation']").replace("total=9","total=6").replace("completed=9","completed=6").replace("*(36-len(frame))","*(24-len(frame))").replace('q_bh_complete_36_slope_family','q_bh_complete_24_slope_family')
    namespace=dict(original.__dict__);namespace['cohort']=cohort
    exec(compile(source,__file__,'exec'),namespace);return namespace['run']

if __name__=='__main__':
    import msvcrt
    ROOT.mkdir(parents=True,exist_ok=True);archive=ROOT/'executed_source';archive.mkdir(exist_ok=True)
    for name in ['a155_research_af_landmark_revision002.py','a155_research_documented_af.py','a155_research_low_cost.py','onedrive_names.py']:
        src=original.b.WORK/'audit_tools'/name;target=archive/name
        if target.exists() and original.b.sha(src)!=original.b.sha(target):raise RuntimeError('Executed design source changed')
        if not target.exists():target.write_bytes(src.read_bytes())
    original.b.atomic(ROOT/'design-revision002.json',dict(reason='First-master ECG design underpowered at all nine horizons/splits; separately evaluate outcome-blind first eligible discharge on complete existing development extraction. No event-per-parameter relaxation, no change to old results.',original_design=str(original.b.MAIN/'publication_preparation/research-documented-af-20261012-v1'),scope='Train/Validation design sensitivity, not new independent confirmation or broad Future arm'))
    execute=install()
    with (ROOT/'run.lock').open('a+b') as lock:
        lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1);execute()
