"""Vorbereiteter externer Adapter, ohne EchoNext-Zugriff im AW131-Auftrag."""
import numpy as np
from .statistik import raw_design,fixed_design,fit
from .adapter import stopped,FALLEN

STATUS={'status':'vorbereitet_gesperrt','reason':'labelblinde train/val-Vorstufe, Messvertrag und separate K-Lesefreigabe fehlen','primary':'lvef_lte_45_flag','sensitivity':'numerische LVEF <=40','processing_policy':'bandbegrenztes 2:1-Resampling 250→500 Hz; scipy.signal.resample_poly up=2 down=1 window=(kaiser,5.0) padtype=constant; vorstufig zu attestieren','technical_seed':123,'technical_replicates':1000}

def gates(attestation):
    if not attestation:raise PermissionError(STATUS['reason'])
    for key in ('labelblind','lead_order_bound','geometry_scaling_bound','filter_clipping_bound','row_mapping_bound','patient_disjoint','device_HR_QTcF_bound','processing_sha256','report_sha256','release_K'):
        if not attestation.get(key):raise PermissionError('EchoNext-Messgate fehlt: '+key)
    for split in ('train','val'):
        z=attestation['technical'][split]
        if z['measurability']<.8:raise PermissionError('Messbarkeit <80%')
    z=attestation['delineation']
    if not 1<=z['train_persons']<=200 or z['valid_boundary_fraction']<.9 or len(z['boundary_errors_ms'])!=4 or max(z['boundary_errors_ms'])>20:raise PermissionError('Delineationsgate verletzt')
    if attestation.get('processing_policy')!=STATUS['processing_policy']:raise PermissionError('250-Hz-Verarbeitungsbindung falsch')

def index(f):
    required={'patient_key','split','most_recent_ecg','age'}
    if required-set(f):raise ValueError('EchoNext-Indexvertrag fehlt')
    if f.patient_key.isna().any() or f.groupby('patient_key').split.nunique().gt(1).any():raise ValueError('EchoNext Identität/Split')
    chosen=f.loc[f.most_recent_ecg.eq(1)].copy()
    if chosen.patient_key.duplicated().any() or set(chosen.patient_key)!=set(f.patient_key):raise ValueError('fehlender/mehrfacher letzter Index')
    return chosen.loc[chosen.age.ge(18)]

def resample(signal):
    from scipy.signal import resample_poly
    signal=np.asarray(signal,float)
    if signal.ndim!=2 or signal.shape!=(2500,12) or not np.isfinite(signal).all():raise ValueError('EchoNext-Signalvertrag')
    return resample_poly(signal,2,1,axis=0,window=('kaiser',5.0),padtype='constant')

def run(f,entry,attestation):
    gates(attestation)  # Tor vor Index-/Labelauswertung.
    try:
        z=index(f)
        if set(z.split)!={'test'}:raise ValueError('externe Inferenz nur freigegebenes test')
        flag=z.lvef_lte_45_flag;value=z.lvef_value
        invalid=value.notna()&(~np.isfinite(value)|~value.between(0,100))
        conflict=flag.isin([0,1])&value.notna()&flag.ne(value.le(45).astype(int))
        good=flag.isin([0,1])&~invalid&~conflict
        x,names=raw_design(z,entry['marker'],entry['comparator']);good &= np.isfinite(x).all(axis=1)
        if names!=entry['scaling_B3']['names']:raise ValueError('externe B3-Struktur verändert')
        design=fixed_design(x[good],entry['scaling_B3']);ids=z.loc[good,'patient_key'].to_numpy()
        primary=fit(design,flag[good].to_numpy(float),ids)
        primary['success_external']=primary['beta']>0 and primary['beta_ci95_wald'][0]>0
        from .runner import bootstrap
        primary['bootstrap']=bootstrap(design,flag[good].to_numpy(float),ids,np.sort(z.patient_key.unique()))
        numeric=good&value.notna();sensitivity=stopped('numerische Referenz fehlt')
        if numeric.any():
            nd=fixed_design(x[numeric],entry['scaling_B3']);ni=z.loc[numeric,'patient_key'].to_numpy()
            sensitivity=fit(nd,value[numeric].le(40).to_numpy(float),ni)
            sensitivity['bootstrap']=bootstrap(nd,value[numeric].le(40).to_numpy(float),ni,np.sort(z.patient_key.unique()))
        return {'status':'schaetzbar','primary_LVEF45':primary,'descriptive_LVEF40':sensitivity,'fallen':FALLEN,'flow':{'index':len(z),'known_complete':int(good.sum()),'conflicts':int(conflict.sum()),'invalid_numeric':int(invalid.sum())},'B4':stopped('physikalischer Amplitudenvertrag fehlt'),'calibration':'nicht konfirmatorisch prüfbar unter K1'}
    except (ValueError,KeyError,AttributeError,RuntimeError,np.linalg.LinAlgError) as exc:return stopped(exc)
