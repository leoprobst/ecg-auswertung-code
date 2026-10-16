"""Eingabebarriere und eingefrorener Stichprobenvertrag."""
from pathlib import Path
import hashlib
import numpy as np
from .audit import DEV

SEED = 20261008
ACTIVE = tuple(f'VB-VF{family:02d}-{i:02d}' for family in (1,2) for i in range(1,9))
REQUIRED = ('subject_id','study_id','split','ecg_time','waveform_path','label','qrs_ms','rhythm_stratum','reference_time')

def development_file(path, fold):
    if fold not in ('train','validation'):
        raise ValueError('Nur train/validation erlaubt')
    p=Path(path).resolve(); root=(DEV/f'split={fold}').resolve()
    if p.parent != root or not p.name.startswith('part-') or p.suffix != '.parquet':
        raise ValueError('Datei außerhalb expliziter Entwicklungswhitelist')
    return p

def validate_rows(frame):
    missing=set(REQUIRED)-set(frame.columns)
    if missing: raise ValueError('Fehlende attestierte Felder: '+','.join(sorted(missing)))
    if frame[list(REQUIRED)].isna().any().any(): raise ValueError('Fehlende Eingabeattestation')
    if not frame['split'].isin(['train','validation']).all(): raise ValueError('Unerlaubter Fold')
    if frame.groupby('subject_id')['split'].nunique().max()>1: raise ValueError('Person in beiden Folds')
    if frame.duplicated(['subject_id','split']).any(): raise ValueError('Indexpolitik: ein EKG pro Person nötig')
    if not frame.label.isin([0,1]).all(): raise ValueError('Label muss 0/1 sein')
    return frame

def draw(frame):
    """Reserve 706/2118 je Fold; proportionale QRS/Rhythmus-Schichtung."""
    validate_rows(frame)
    rows=[]
    for fold in ('train','validation'):
        for label,target in ((1,706),(0,2118)):
            pool=frame[(frame.split==fold)&(frame.label==label)].copy()
            pool['qrs_stratum']=np.where(pool.qrs_ms<120,'lt120','ge120')
            groups=list(pool.groupby(['qrs_stratum','rhythm_stratum'],sort=True))
            n=min(target,len(pool))
            exact=np.array([n*len(g)/len(pool) for _,g in groups]) if len(pool) else np.array([])
            quotas=np.floor(exact).astype(int)
            for j in np.argsort(-(exact-quotas),kind='stable')[:n-quotas.sum()]: quotas[j]+=1
            for ((stratum,g),quota) in zip(groups,quotas):
                # Identitäts-Hashranking ist unabhängig von Eingangsreihenfolge und Markerwerten.
                rank=g.subject_id.map(lambda x:hashlib.sha256(f'{SEED}|{fold}|{label}|{int(x)}'.encode()).hexdigest())
                rows.extend(g.assign(_rank=rank).sort_values('_rank').head(quota).drop(columns='_rank').to_dict('records'))
    import pandas as pd
    return pd.DataFrame(rows).sort_values(['split','label','subject_id']).reset_index(drop=True)
