"""Index-, Referenz- und Missingnessgates: keine Ersatzindizes/Imputation."""
import json
from decimal import Decimal, InvalidOperation
import numpy as np
import pandas as pd
from .statistik import raw_design

HYPOTHESES=[
    ('BEF-0063','reduced_lvef_strict','qrs_t_normalized_discrete_frechet','qrs_duration_ms',1),
    ('BEF-0042','acute_decompensated_heart_failure_proxy','qrs_t_loop_mirror_symmetry','qrs_duration_ms',-1),
    ('BEF-0081','elevated_ntprobnp','qrs_t_frechet_beat_median','qrs_duration_ms',1),
    ('BEF-0049','cardiomyopathy_icd','qrs_t_frechet_beat_median','qrs_duration_ms',1),
    ('BEF-0075','troponin_positive_any','qrs_t_normalized_dtw_distance','classical_st_elevation_contiguous_mm',1),
    ('BEF-0087','echo_rv_systolic_dysfunction','qrs_t_normalized_dtw_distance','classical_dominant_r_v1',1)]

def truth(series):return series.eq(True).fillna(False)

def classify_lvef(values,inclusive=False):
    a=np.asarray(values,float)
    if not np.isfinite(a).all() or (a<0).any() or (a>100).any():raise ValueError('ungültige LVEF')
    return (a<=40 if inclusive else a<40).astype(float)

def canonical(value):
    try:return str(Decimal(str(value)).normalize())
    except InvalidOperation:return str(value)

def echo_ids(row):
    try:
        ids=json.loads(row.evidence_ids_json)
        return canonical(row.nearest_tte_measurement_id) in [canonical(v) for v in ids]
    except (ValueError,TypeError,AttributeError):return False

def masks(f,ep,marker,comparator,window=720,strict_alignment=None,strong=False):
    required=['subject_id','study_id','split','age','sex','index_selection_strategy','label_state','eligibility_state','ehr_time_alignment_confirmatory_eligible']
    missing=set(required)-set(f.columns)
    if missing:raise ValueError('fehlender Inputvertrag: '+','.join(sorted(missing)))
    if f[['subject_id','study_id','split']].isna().any().any() or f.subject_id.duplicated().any():raise ValueError('fehlende/mehrfache Identität oder Index')
    if f.groupby('subject_id').split.nunique().max()>1:raise ValueError('Person nicht splitdisjunkt')
    index=f.index_selection_strategy.eq('first_verified_ecg_reference_result_blind')
    adult=pd.to_numeric(f.age,errors='coerce').ge(18)
    known=f.label_state.isin(['positive','negative'])
    eligible=f.eligibility_state.eq('eligible')
    if strict_alignment is None:
        strict_alignment=ep in ('elevated_ntprobnp','acute_decompensated_heart_failure_proxy','troponin_positive_any')
    aligned=truth(f.ehr_time_alignment_confirmatory_eligible) if strict_alignment else pd.Series(True,index=f.index)
    time=pd.Series(True,index=f.index);reference=pd.Series(True,index=f.index)
    if ep in ('reduced_lvef_strict','echo_rv_systolic_dysfunction'):
        fields=['time_delta_hours','time_precision','nearest_tte_delta_hours','nearest_tte_measurement_id','evidence_ids_json']
        if set(fields)-set(f):raise ValueError('Echo-ID-/Zeitvertrag fehlt')
        delta=pd.to_numeric(f.time_delta_hours,errors='coerce');nearest=pd.to_numeric(f.nearest_tte_delta_hours,errors='coerce')
        time=f.time_precision.eq('timestamp')&delta.abs().le(window)&np.isclose(delta,nearest,atol=1e-9,rtol=0,equal_nan=False)
        reference=f.apply(echo_ids,axis=1)
    elif ep in ('elevated_ntprobnp','acute_decompensated_heart_failure_proxy'):
        time=f.time_precision.eq('timestamp')&pd.to_numeric(f.time_delta_hours,errors='coerce').between(-6,12,inclusive='both')
    if ep=='reduced_lvef_strict':
        values=pd.to_numeric(f.preferred_lvef,errors='coerce')
        reference=reference&np.isfinite(values)&values.between(0,100)&f.lvef_conflict_gt_10.eq(False).fillna(False)
    # Kein unbekanntes Geschlecht als 0; zeitgeeignete Geräte-HF aus attestierter Projektion.
    f=f.copy();f['sex_numeric']=f.sex.map({'F':0.,'M':1.})
    if 'machine_metadata_usable_for_primary' in f:
        f['heart_rate']=pd.to_numeric(f.machine_heart_rate_bpm,errors='coerce').where(truth(f.machine_metadata_usable_for_primary))
    x,names=raw_design(f,marker,comparator,strong=strong)
    finite=pd.Series(np.isfinite(x).all(1),index=f.index)
    stages={k:v.fillna(False).astype(bool) for k,v in {'adult':adult,'index_policy':index,'eligible':eligible,'known_reference':known,'reference_time_id_value':time&reference,'confirmatory_time_alignment':aligned,'complete_B3_marker':finite}.items()}
    good=pd.Series(True,index=f.index);sequential={'index_universe':len(f)}
    for key,mask in stages.items():good=good&mask;sequential[key]=int(good.sum())
    overlapping={key:int((~mask).sum()) for key,mask in stages.items()}
    strata=[]
    agegroup=np.where(f.age.isna(),'unknown',np.where(f.age.lt(65).fillna(False),'<65','>=65'))
    tmp=pd.DataFrame({'split':f.split,'class':f.label_state,'age':agegroup,'sex':f.sex.fillna('unknown'),'good':good})
    for keys,z in tmp.groupby(['split','class','age','sex'],dropna=False):
        strata.append(dict(zip(['split','class','age','sex'],map(str,keys)))|{'index_n':len(z),'complete_n':int(z.good.sum())})
    return f,good,{'sequential':sequential,'overlapping_excluded':overlapping,'strata':strata,'missing_columns':{name:int(np.count_nonzero(~np.isfinite(x[:,i]))) for i,name in enumerate(names)}}

def select_index(f,ep,technical=False):
    """Wertblinde Indexwahl; Missingness darf kein Ersatz-EKG liefern.

    Eingangsvertrag: technische Auswertbarkeit und Referenzverfügbarkeit sind
    release-attestierte Flags, keine aus Markerwerten konstruierten Flags.
    """
    required={'subject_id','study_id','split','ecg_time','age','extraction_status'}
    if not technical:required.add('reference_available')
    if required-set(f):raise ValueError('Indexeingangsvertrag fehlt')
    if f[['subject_id','study_id','split','ecg_time']].isna().any().any():raise ValueError('Indexidentität/Zeit fehlt')
    if f.groupby('subject_id').split.nunique().gt(1).any():raise ValueError('Person nicht splitdisjunkt')
    if f.duplicated(['subject_id','study_id']).any():raise ValueError('mehrfache Aufzeichnung')
    good=f.age.ge(18)&f.extraction_status.eq('ok')
    if not technical:good &= truth(f.reference_available)
    z=f.loc[good].sort_values(['ecg_time','study_id'],kind='stable').drop_duplicates('subject_id').copy()
    z['index_selection_strategy']='first_technical_adult' if technical else 'first_verified_ecg_reference_result_blind'
    return z
