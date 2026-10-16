"""Schema-/Quellen-/STOP-Verträge ohne Eingabelesung; Ledger mit injiziertem Ziel.

AW171 benutzt ausschließlich synthetische Ziele auf E:. Der spätere Auftrag
muss seine Lesequellen/Schreibziele attestieren und den Adapter separat binden.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
from contextlib import contextmanager
from pathlib import Path
import json
import numpy as np
from werkzeuge.b_test.aw171_kernels import require, Stop
from werkzeuge.b_test.daten import masks, select_index

REQUIRED_CONTRACTS=['partition_attested','previously_opened_internal','sealed_excluded',
                    'freeze_recorded','execution_order','input_hashes','code_hash','parameter_hash',
                    'physical_input_list','index_reference_attested']

def preflight(contract):
    # Vor Leseraufruf, Hash und Footer: keine Freigabe allein durch Splitnamen.
    for field in REQUIRED_CONTRACTS:require(bool(contract.get(field)),'Vertrag_'+field)
    require(contract.get('partition_role')=='previously_opened_internal','Vertrag_Partitionsrolle')
    require(contract.get('sealed') is False,'Sperrbarriere_gesperrte_Testmenge')
    require(contract.get('scaling_role')=='development_fixed','Vertrag_Skalierung')

def match_hash(actual,expected,kind):
    require(bool(expected) and actual==expected,'Hash_'+kind)

def split_guard(frame,mode):
    allowed={'synthetic_testschema'} if mode=='synthetic' else {'train','validation'} if mode=='development' else set()
    require(bool(allowed) and set(frame.split)<=allowed,'Sperrbarriere_Split')
    require(frame[['subject_id','study_id','split']].notna().all().all(),'Index_Identitaet')
    require(not frame.groupby('subject_id').split.nunique().gt(1).any(),'Index_Splitdisjunkt')

def case_sets(candidates,endpoint,marker,mode='synthetic',expected=None):
    """Vollständiges Testschema: wertblinder Index, D_e vor CC, H4 ohne Echo."""
    split_guard(candidates,mode)
    # H4-Referenzverfügbarkeit kommt aus Index-Encounter, kein Echoersatz.
    if endpoint=='cardiomyopathy_icd':
        require(candidates.reference_available.equals(candidates.encounter_reference_available),'H4_Referenzvertrag')
    index=select_index(candidates,endpoint)
    frame,complete,flow=masks(index,endpoint,marker,'qrs_duration_ms')
    # masks liefert Referenzstufen, die vor Vollständigkeit unverändert bleiben.
    # Zur Rekonstruktion D_e werden nur für den Rechenmaskenaufruf B3-Werte
    # temporär gefüllt; Originaldaten/Fallset bleiben unberührt.
    reference=index.copy()
    for col,value in [('sex','F'),('heart_rate',60),('qrs_duration_ms',100),('qtc_fridericia_ms',400),
                      ('spatial_qrs_t_angle_deg',90),(marker,1)]:
        reference[col]=reference[col].fillna(value)
    # age fehlt => keine Erwachsenenreferenz, Alter wird nie für D_e ergänzt.
    _,valid,_=masks(reference,endpoint,marker,'qrs_duration_ms')
    D=frame.loc[valid].copy();C=frame.loc[complete].copy()
    if expected is not None:
        require(len(index)==expected['index_n'] and len(C)==expected['complete_n'],'Referenz_Index_Abgleich')
    require(set(C.subject_id)<=set(D.subject_id),'Referenz_Fallset')
    return D,C,{'candidate_recordings':len(candidates),'candidate_persons':int(candidates.subject_id.nunique()),
               'adult_recordings':int(candidates.age.ge(18).sum()),
               'technical_recordings':int((candidates.age.ge(18)&candidates.extraction_status.eq('ok')).sum()),
               'index_n':len(index),'reference_n':len(D),'complete_n':len(C),'bound_flow':flow,
               'echo_parallel_n':int(index.echo_available.eq(True).sum()) if 'echo_available' in index else None}

@contextmanager
def access_phase(ledger,order,freeze_hash,code_hash,parameter_hash,clock):
    """Exactly one Beginn/Ende selbst bei Fehler, keine Input-/Kennungszeilen.

    Ziel wird vom späteren Auftrag zugewiesen. AW171-Testziel auf E: statt
    koordination/TESTZUGRIFFE.md, das in AW171 kein Schreibziel ist.
    """
    target=Path(ledger)
    line=lambda state:f'{clock()} | {order} | {state} gebuendelter explorativer B-Zugriff | {freeze_hash}/{code_hash}/{parameter_hash}\n'
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('a',encoding='utf8') as f:f.write(line('Beginn'))
    try:yield
    finally:
        with target.open('a',encoding='utf8') as f:f.write(line('Ende'))

def checkpoint(path,state,bindings):
    """Ein eigener Blockzustand auf E:, Wiederaufnahme nur bei identischen Bindungen."""
    target=Path(path)
    require(target.resolve().is_relative_to(Path(str(data_path("b.aw171"))).resolve()),'Checkpoint_Ziel')
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps({'bindings':bindings,'state':state},sort_keys=True,allow_nan=False),encoding='utf8')

def resume(path,bindings):
    record=json.loads(Path(path).read_text(encoding='utf8'))
    require(record['bindings']==bindings,'Checkpoint_Hashabweichung')
    return record['state']

def w7_gate(cells):
    require(len(cells)==8 and len({(r['endpoint'],r['marker'],r['fold']) for r in cells})==8,'W7_acht_Zellen')
    require(all(np.isfinite(r['beta']) and r['beta']>0 and np.isfinite(r['delta_auc']) and r['delta_auc']>=0 for r in cells),'W7_Gate_geschlossen')

def source_columns(actual,required):
    require(set(required)<=set(actual),'fehlende_Spalte')

def weight_gate(propensity):
    require(np.isfinite(propensity).all() and (np.asarray(propensity)>0).all(),'Propensity')
