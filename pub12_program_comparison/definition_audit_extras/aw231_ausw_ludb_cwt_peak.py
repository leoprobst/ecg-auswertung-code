import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import pandas as pd, numpy as np
M=2.0
d=pd.read_csv(str(data_path("pub12.benchmark_pub12_aw231_schlagpaare_cwt_peak_ludb10_aw231_csv")))
f=pd.read_csv(str(data_path("pub12.benchmark_pub12_aw231_fehler_cwt_peak_ludb10_csv")))
print(f.groupby(['method','fehler']).size())
print(d.groupby('method').size())
x=d[d.method=='cwt'].copy()
x['e_on']=(x.on_nk-x.on_exp)*M; x['e_off']=(x.off_nk-x.off_exp)*M
print('cwt pairs',len(x),'onset set',x.on_nk.notna().sum(),'end set',x.off_nk.notna().sum())
for nm,c in (('Onset',x.e_on),('End',x.e_off)):
    v=c.dropna(); print(nm,'n',len(v),'median',v.median(),'IQR',v.quantile(.25),v.quantile(.75),'|e|<=20',(v.abs()<=20).mean().round(3),'<-50',(v<-50).mean().round(3),'>50',(v>50).mean().round(3))
v=((x.off_nk-x.on_nk)*M).dropna(); print('cwt duration median',v.median(),'n',len(v))
y=d[d.method=='peak'].copy()
print('peak pairs',len(y),'Q set',y.q_nk.notna().sum(),'S set',y.s_nk.notna().sum())
print('peak Q peak minus LUDB onset median',((y.q_nk-y.on_exp)*M).median(),'IQR',((y.q_nk-y.on_exp)*M).quantile([.25,.75]).tolist(),'; S peak minus LUDB end',((y.s_nk-y.off_exp)*M).median(),((y.s_nk-y.off_exp)*M).quantile([.25,.75]).tolist())
print('peak: method returns no R_Onsets/R_Offsets (keys) ->', 'onset/end NaN:', y.on_nk.isna().all(), y.off_nk.isna().all())
