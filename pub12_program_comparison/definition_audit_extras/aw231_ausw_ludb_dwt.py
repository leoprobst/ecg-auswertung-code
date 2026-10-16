import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import pandas as pd, numpy as np
M=2.0
d=pd.read_csv(str(data_path("pub12.benchmark_pub12_aw231_schlagpaare_ludb10_aw231_csv")))
x=d[d.method=='dwt'].copy()
x['e_on']=(x.on_nk-x.on_exp)*M; x['e_off']=(x.off_nk-x.off_exp)*M
x['q_err']=(x.q_nk-x.on_exp)*M   # Q peak minus LUDB onset (the Q peak lies after the onset: positive is normal)
x['exp_dur']=(x.off_exp-x.on_exp)*M
print('Pairs',len(x),'with onset',x.on_nk.notna().sum(),'with end',x.off_nk.notna().sum())
for nm,c in (('Onset',x.e_on),('End',x.e_off)):
    v=c.dropna(); print(nm,'median',v.median(),'IQR',v.quantile(.25),v.quantile(.75),'|e|<=10',(v.abs()<=10).mean().round(3),'|e|<=20',(v.abs()<=20).mean().round(3),'<-50',(v<-50).mean().round(3),'<-100',(v<-100).mean().round(3))
print('Q peak: set',x.q_nk.notna().sum(),'Q peak minus LUDB onset median',x.q_err.median())
x['q_nah']=x.q_err.between(0,60)
for k,g in x.groupby('q_nah'):
    print('Q peak 0..60 ms after LUDB onset =',k,'n',len(g),'onset error median',g.e_on.median(),'share <-50',(g.e_on.dropna()<-50).mean().round(3))
# Does the onset lie on the P wave?
p=x.dropna(subset=['on_nk','p_exp_on','p_exp_off'])
inP=(p.on_nk>=p.p_exp_on)&(p.on_nk<=p.p_exp_off)
print('NK onset within LUDB P wave (onset..end) :',inP.mean().round(3),'n',len(p))
print('NK onset before LUDB P end (incl. before P):',(p.on_nk<p.p_exp_off).mean().round(3))
# lead groups
gr={'Extremitaeten':['I','II','III','AVR','AVL','AVF'],'V1-V3':['V1','V2','V3'],'V4-V6':['V4','V5','V6']}
for k,l in gr.items():
    g=x[x.lead.isin(l)]; print(k,'onset median',g.e_on.median(),'end median',g.e_off.median(),'n',len(g))
# QRS duration per lead/beat from onset/end
x['nk_dur']=(x.off_nk-x.on_nk)*M
v=x.dropna(subset=['nk_dur']); print('Duration NK minus expert (same beat, lead) median',(v.nk_dur-v.exp_dur).median(),'n',len(v),'NK duration median',v.nk_dur.median(),'Exp',v.exp_dur.median())
# per record
print(x.groupby('rec').e_on.median().to_dict())
# R peak: NK R vs expert N peak
print('R_nk minus N_exp (ms) median',((x.r_nk-x.n_exp)*M).median(), 'IQR',((x.r_nk-x.n_exp)*M).quantile([.25,.75]).tolist())
print('Total number of expert QRS (lead, beat):', 'n_leadsbeats unique rec,lead,n_exp', x[['rec','lead','n_exp']].drop_duplicates().shape[0])
