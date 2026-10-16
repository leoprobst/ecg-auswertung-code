import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import numpy as np, pandas as pd
src=open('aw231_qt.py',encoding='utf8').read().replace('\nmain()\n','\n'); ns={}; exec(compile(src,'q','exec'),ns)
fpt=ns['fpt']; nanrank=ns['nanrank']; LEADS=ns['LEADS']
d=pd.read_csv(str(data_path("pub12.benchmark_pub12_aw231_ecgdeli_qt_nachbau_aw231_csv")))
l=pd.read_csv(str(data_path("pub12.benchmark_pub12_aw231_ecgdeli_qt_ableitungen_aw231_csv")))
med_lead=l.groupby('ecg_id').feature.median()
d=d.set_index('ecg_id'); d['qt_leadmedian']=med_lead
print('Global QT (feature) minus median of the 12 lead QTs (feature): median',(d.feature_QT_global-d.qt_leadmedian).median(),'IQR',(d.feature_QT_global-d.qt_leadmedian).quantile([.25,.75]).tolist())
print('Median of the lead QTs over records',d.qt_leadmedian.median(),'| 12SL global',d.qt12.median(),'| UniG global',d.qtug.median(), '| ECGDeli global',d.feature_QT_global.median())
print('Lead-QT median minus 12SL global: median',(d.qt_leadmedian-d.qt12).median())
rows=[]
for e in d.index:
    M=[fpt(int(e),L)[0] for L in LEADS]; B=min(len(m) for m in M)
    A=np.stack([m[:B] for m in M],axis=2); A[A<5]=np.nan
    on=A[:,3,:]; toff=A[:,11,:]
    on_med=np.nanmedian(on,axis=1); toff_med=np.nanmedian(toff,axis=1)
    on_r2=nanrank(on,2,False); toff_r2=nanrank(toff,2,True)
    rows.append(dict(ecg_id=e, d_on=np.nanmedian((on_med-on_r2)*2), d_toff=np.nanmedian((toff_r2-toff_med)*2), qt_med_aggr=np.nanmedian((toff_med-on_med)*2),
        spread_toff=np.nanmedian((np.nanmax(toff,axis=1)-np.nanmin(toff,axis=1))*2), spread_on=np.nanmedian((np.nanmax(on,axis=1)-np.nanmin(on,axis=1))*2)))
r=pd.DataFrame(rows).set_index('ecg_id')
print(r.median())
print('Share of the difference global minus median per lead: onset extreme',r.d_on.median(),' T-end extreme',r.d_toff.median(), ' sum',r.d_on.median()+r.d_toff.median())
print('QT from the median over leads of onset and T end:',r.qt_med_aggr.median())
# 12SL global onset/T-end span
print('12SL: T_Off_Global-QRS_On_Global minus QT_Int_Global (Median)',((d.toff12-d.on12)-d.qt12).median())
