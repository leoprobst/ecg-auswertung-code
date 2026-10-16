"""Fail-closed document audit and sealed development-only AF census.

No historical generator is imported or run. No mixed patient source is opened.
All private details and checkpoints stay in the assigned private output directory.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import ast
import csv
import ctypes
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(str(data_path("s5.auswertung")))
OUT = Path(str(data_path("s5.aw178")))
PP = Path(str(data_path("s5.mimic_validation_runs_a155_2026_10_04_d6e70e_publication_preparation")))
PUB = ROOT / 'publikationen/PUB-08_zeitausrichtung'
SPEC_HASH = '21066abcec4572727448bda0133b8f4773835af06bcff987fd4ae71a21ee6b22'
RELEASE = Path(str(data_path("s5.release_sources_a155_survival_nullable_20261007_vcg_biomarker_mimic")))

class Memory(ctypes.Structure):
    _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [(k, ctypes.c_ulonglong) for k in ['total', 'free', 'commit_total', 'commit_free', 'virtual', 'virtual_free', 'extended']]

def memory():
    m = Memory(); m.length = ctypes.sizeof(m)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return {'free_ram_gib': m.free / 2**30, 'free_commit_gib': m.commit_free / 2**30}

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def read_small(p):
    assert p.stat().st_size < 20_000_000
    return p.read_text(encoding='utf-8-sig')

def read_csv(p):
    with p.open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))

def main():
    start = time.monotonic(); OUT.mkdir(parents=True, exist_ok=True)
    mem = memory()
    assert min(mem.values()) >= 1.5, 'Resource gate closed'
    assert sha(PUB / 'SPEZIFIKATION_S5_NACHRECHNUNG_ENTWURF.md') == SPEC_HASH
    save('status.json', {'phase': 'R0', 'pid': os.getpid(), 'state': 'running', 'memory': mem})
    files = [Path(__file__), ROOT/'werkzeuge/s5/aw178_start.ps1', ROOT/'AGENTS.md', ROOT/'koordination/auftraege/AW-178.md', PUB/'SPEZIFIKATION_S5_NACHRECHNUNG_ENTWURF.md', PUB/'DOSSIER_S5_ERWEITERUNG.md', PUB/'MANUSKRIPT_S5_v6.md', PUB/'MANUSKRIPT_S5_v7.md', PUB/'CLAIM_MATRIX_S5_v2.csv', ROOT/'publikationen/BLUEPRINT_ANGABEN.md', ROOT/'publikationen/AUTOR_ANGABEN.json', ROOT/'publikationen/ZIELJOURNALE.md', ROOT/'publikationen/STRATEGIE.md', ROOT/'standards/02_pruefkette.md']
    xroot = PP/'research-final-hypotheses-20261012-v1'
    aroot = PP/'research-documented-af-landmark-20261012-v2'
    rest = PP/'research-rest-low-cost-20261012-v1'
    for directory, names in [(xroot, ['protocol.json','completed-files.sha256.json','executed_source/a155_research_final_hypotheses.py','executed_source/a155_research_low_cost.py','interaction-contrasts.csv','frozen-validation-loss.csv','cohort-flow.csv']), (aroot, ['protocol.json','completed-files.sha256.json','first-eligible-landmarks.sha256.json','design-revision002.json','executed_source/a155_research_af_landmark_revision002.py','executed_source/a155_research_documented_af.py','F34-documented-af-model-census.csv']), (rest, ['protocol.json','population-first-ecg.sha256.json','executed_source/a155_research_rest_low_cost.py']), (RELEASE, ['evidence.py','event_ingest.py','lab_quality.py','mappings.py','labels.py'])]:
        files.extend(directory/name for name in names)
    manifest = {str(p): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in files}
    save('input_manifest.json', manifest)
    # Verify seals for each document actually consumed; never walk historic roots.
    sealchecks = {}
    for directory, names in [(xroot, ['protocol.json','executed_source/a155_research_final_hypotheses.py','executed_source/a155_research_low_cost.py','interaction-contrasts.csv','frozen-validation-loss.csv','cohort-flow.csv']), (aroot, ['protocol.json','executed_source/a155_research_af_landmark_revision002.py','executed_source/a155_research_documented_af.py','first-eligible-landmarks.parquet','first-eligible-landmarks.sha256.json','F34-documented-af-model-census.csv'])]:
        seal = {k.replace('\\','/'):v for k,v in json.loads(read_small(directory/'completed-files.sha256.json')).items()}
        for name in names:
            sealchecks[str(directory/name)] = seal.get(name) == sha(directory/name)
    save('seal_checks.json', sealchecks)
    assert all(sealchecks.values()), 'Generator seal does not match'
    xs = read_small(xroot/'executed_source/a155_research_final_hypotheses.py')
    rs = read_small(rest/'executed_source/a155_research_rest_low_cost.py')
    aps = read_small(aroot/'executed_source/a155_research_af_landmark_revision002.py')
    xp = json.loads(read_small(xroot/'protocol.json'))
    assert "d=pd.read_parquet(POP)" in xs and "d=d.loc[d.split.isin(['train','validation'])]" in xs
    assert 'held=[c for c in marks' in rs and 'UNION ALL BY NAME' in rs
    assert "if not set(d.split.unique())<={'train','validation'}" in aps
    assert 'not independently verified raw assay units' in xp['units']
    r0 = {'status':'abgebrochen', 'partition':'The X08 population contains the historical holdout; the filter is applied only after the read operation. Not opened, no footer of this input.', 'potassium':'The historical protocol confirms an unverified numeric convention. Raw assay, specimen, unit and reference range are not shown to be case-bound.', 'target':'X08: stored tpeak_tend_ms without target transformation; MSE on its square. Upstream raw-measurement binding not independently confirmed.', 'R1':'not started: R0 gate closed', 'R2':'not started: R0/R1 gate closed', 'R3':'aborted: no attested development-separated B01/M01/M03 input or historically bound episode selector found; historical numbers not re-read', 'R4':'aborted: no attested development-separated event-ECG pool with a frozen episode rule found; the historical summary stays closed', 'R5':'separate development gate: the sealed v2 selector enforces train/validation before storage; hash check before the footer; census only, no AF fits', 'R6':'AF v2 demographics executable; the remaining new demographic censuses are blocked by missing input binding; no public supplement released'}
    save('R0_checkpoint.json', r0)
    # Exact source anchors, in private output only; no source patient values exist.
    anchors = {}
    for p, patterns in [(xroot/'executed_source/a155_research_final_hypotheses.py', ['pd.read_parquet(POP)','d.split.isin','DEFS=','train[y]','fdr','units']), (rest/'executed_source/a155_research_rest_low_cost.py',['def population','held=[','UNION ALL BY NAME','first_index']), (RELEASE/'evidence.py',['nearest_order','distance_hours :=','before_ecg_rank :=','DATE_DIFF(\'second\', m.ecg_time, l.charttime)','followup_hours']), (RELEASE/'event_ingest.py',['TRY_CAST(charttime','TRY_CAST(storetime','TRY_CAST(starttime','TRY_CAST(chartdate']), (aroot/'executed_source/a155_research_documented_af.py',['def risk_set','censor_hours','event_hours','followup_hours'])]:
        anchors[str(p)] = [{'line':i,'text':line.strip()} for i,line in enumerate(read_small(p).splitlines(),1) if any(s in line for s in patterns)]
    save('source_anchors.json', anchors)
    # Existing development aggregate results: preserved, never relabelled as new fits.
    x = [r for r in read_csv(xroot/'interaction-contrasts.csv') if r['id']=='X08']
    loss = [r for r in read_csv(xroot/'frozen-validation-loss.csv') if r['id']=='X08']
    flow = [r for r in read_csv(xroot/'cohort-flow.csv') if r['id']=='X08']
    assert {r['split'] for r in x+flow} <= {'train','validation'}
    save('historical_x08.json', {'coefficients':x,'loss':loss,'flow':flow})
    save('status.json', {'phase':'R5/R6', 'pid':os.getpid(), 'state':'running'})
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    import pyarrow
    import psutil
    afpath = aroot/'first-eligible-landmarks.parquet'
    assert sha(afpath) == json.loads(read_small(aroot/'first-eligible-landmarks.sha256.json'))['sha256']
    footer = pq.ParquetFile(afpath)
    save('af_schema.json', {'rows':footer.metadata.num_rows,'row_groups':footer.num_row_groups,'schema':str(footer.schema_arrow),'sha256':sha(afpath)})
    cols = ['subject_id','split','age','sex','sex_male','ecg_time','dischtime','machine_report','machine_metadata_usable_for_primary','machine_heart_rate_bpm','first_af_event_delta_hours','next_af_event_delta_hours','next_af_event_source','af_dx_prior_admission','af_dx_index_admission','current_report_af_state','followup_hours','date_of_death','p_loop_width_major','rhythm_context']
    d = pq.read_table(afpath, columns=cols).to_pandas()
    assert set(d['split'].unique()) <= {'train','validation'}
    assert not d.subject_id.duplicated().any(), 'Duplicate person in sealed development cohort'
    # Execute only the two pure archived functions as AST nodes, never run/import generators.
    parsed = ast.parse(read_small(aroot/'executed_source/a155_research_documented_af.py'))
    nodes = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in {'risk_set','report_sinus'}]
    assert len(nodes)==2
    import re
    ns = {'np':np,'pd':pd,'re':re}
    exec(compile(ast.Module(body=nodes,type_ignores=[]), '<sealed AF pure functions>', 'exec'), ns)
    freeze = {'time_utc':datetime.now(timezone.utc).isoformat(),'partition':'sealed v2 only: train/validation','operation':'historical risk_set and complete-case reproduction, no model fit','horizons':[30,90,365],'complete_cases':['p_loop_width_major','age','sex_male','machine_heart_rate_bpm'],'ending':'minimum contact, recorded death, horizon; exact ties separately; AF at death not an event in archived recipe','unknown':'contact loss is not event-free follow-up; date-of-death may have day precision','R1_seed':178001,'R1_bootstrap_resamples':1000,'R1_family':'BH10 interactions, BH3 loss; NOT EXECUTED','new_hypothesis_tests':False}
    save('R5_freeze.json', freeze)
    census = []; demos = []; missing = []
    predictors = freeze['complete_cases']
    def demographic(frame, split, stage, horizon):
        age = pd.to_numeric(frame.age, errors='coerce').replace([np.inf,-np.inf],np.nan)
        quantiles = age.dropna().quantile([.25,.5,.75])
        return {'split':split,'stage':stage,'horizon_days':horizon,'n':len(frame),'sex_F':int(frame.sex.eq('F').sum()),'sex_M':int(frame.sex.eq('M').sum()),'sex_unknown':int((~frame.sex.isin(['F','M'])).sum()),'age_present':int(age.notna().sum()),'age_missing':int(age.isna().sum()),'age_q25':float(quantiles.iloc[0]) if len(quantiles) else None,'age_median':float(quantiles.iloc[1]) if len(quantiles) else None,'age_q75':float(quantiles.iloc[2]) if len(quantiles) else None,'age_definition':'stored context age; derivation not independently linked to this cached cohort','setting':'index hospital discharge; ED/ICU setting unavailable in projected sealed cohort'}
    stored = read_csv(aroot/'F34-documented-af-model-census.csv')
    for split in ['train','validation']:
        part = d.loc[d.split.eq(split)].copy()
        demos.append(demographic(part,split,'selected_landmarks',0))
        for col in cols:
            missing.append({'split':split,'field':col,'n':len(part),'missing':int(part[col].isna().sum())})
        for horizon in freeze['horizons']:
            risk, steps = ns['risk_set'](part, horizon)
            previous = len(part)
            for step in steps:
                census.append({'split':split,'horizon_days':horizon,'stage':step['gate'],'n':step['n_retained'],'excluded_exclusive':previous-step['n_retained']})
                previous = step['n_retained']
            numeric = risk[predictors].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
            cc = risk.loc[numeric.notna().all(axis=1)].copy()
            ref = next(r for r in stored if r['split']==split and int(r['horizon_days'])==horizon)
            assert len(cc)==int(ref['n_complete']) and int(cc.event.sum())==int(ref['n_events'])
            landmark = (pd.to_datetime(cc.dischtime)-pd.to_datetime(cc.ecg_time)).dt.total_seconds()/3600
            contact = pd.to_numeric(cc.followup_hours,errors='coerce')-landmark
            death = (pd.to_datetime(cc.date_of_death)-pd.to_datetime(cc.dischtime)).dt.total_seconds()/3600
            # Classify non-AF endings, separating exact timestamp ties; not a competing-risk model.
            noevent = cc.event.eq(0)
            ends = pd.DataFrame({'contact':contact,'death':death.fillna(np.inf),'horizon':horizon*24},index=cc.index)
            minima = ends.min(axis=1); ties = ends.eq(minima,axis=0).sum(axis=1).gt(1)
            rec = {'split':split,'horizon_days':horizon,'stage':'complete_cases','n':len(cc),'n_persons':int(cc.subject_id.nunique()),'excluded_exclusive':len(risk)-len(cc),'events':int(cc.event.sum()),'AF_machine_report':int((cc.event.eq(1)&cc.next_af_event_source.eq('machine_report')).sum()),'AF_icd_admission':int((cc.event.eq(1)&cc.next_af_event_source.eq('icd_admission')).sum()),'AF_other_or_unknown':int((cc.event.eq(1)&~cc.next_af_event_source.isin(['machine_report','icd_admission'])).sum()),'non_AF_contact_end':int((noevent & ~ties & contact.eq(minima)).sum()),'non_AF_death_end':int((noevent & ~ties & death.eq(minima)).sum()),'non_AF_horizon_end':int((noevent & ~ties & ends.horizon.eq(minima)).sum()),'non_AF_tied_end':int((noevent & ties).sum()),'death_missing':int(cc.date_of_death.isna().sum()),'death_recorded_before_horizon':int(death.between(0,horizon*24,inclusive='right').sum()),'source':'sealed v2 cohort; archived risk_set; reference census matched'}
            assert rec['events']+sum(rec[k] for k in ['non_AF_contact_end','non_AF_death_end','non_AF_horizon_end','non_AF_tied_end'])==len(cc)
            census.append(rec)
            demos.append(demographic(risk,split,'at_risk',horizon))
            demos.append(demographic(cc,split,'complete_cases',horizon))
    save('R5_census.json', census); save('R6_af_demographics.json', demos); save('R5_missingness.json', missing)
    save('execution.json', {'pid':os.getpid(),'start_memory':mem,'end_memory':memory(),'elapsed_seconds':time.monotonic()-start,'rss_bytes':psutil.Process().memory_info().rss,'peak_working_set_bytes':getattr(psutil.Process().memory_info(),'peak_wset',None),'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__,'pyarrow':pyarrow.__version__,'duckdb':'not used; no fits','test_access':False,'patient_sources_opened':['sealed AF-v2 development cohort only'],'R0':r0})
    save('status.json', {'phase':'audit complete', 'pid':os.getpid(), 'state':'completed_with_prespecified_stops','elapsed_seconds':time.monotonic()-start})
    print('Audit finished; R0 stop, R1/R2 blocked, R3/R4 stop, R5/R6 partial. See the assigned private output directory.')

if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT/'audit_stderr.txt').write_text(traceback.format_exc(), encoding='utf-8')
        save('status.json', {'pid':os.getpid(),'state':'failed','details':'audit_stderr.txt (private)'})
        raise
