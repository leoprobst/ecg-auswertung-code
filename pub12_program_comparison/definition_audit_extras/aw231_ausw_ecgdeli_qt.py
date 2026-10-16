import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import pandas as pd, numpy as np
d=pd.read_csv(str(data_path("pub12.benchmark_pub12_aw231_ecgdeli_qt_nachbau_aw231_csv")))
l=pd.read_csv(str(data_path("pub12.benchmark_pub12_aw231_ecgdeli_qt_ableitungen_aw231_csv")))
print('n',len(d),'Same beat count in all leads:',d.beats_gleich_alle_ableitungen.sum())
def cmp(name,a,b):
    x=(a-b).dropna(); print(f'{name:38s} n={len(x)} exakt={int((x.abs()<1e-9).sum())} |<=1ms|={int((x.abs()<=1).sum())} median={x.median():.2f} max|d|={x.abs().max():.1f}')
cmp('QT code (2nd extreme) vs feature QT_Int_Global',d.qt_code_median,d.feature_QT_global)
cmp('QT min/max vs feature',d.qt_minmax_median,d.feature_QT_global)
cmp('QT consensus vs feature',d.qt_consensus_median,d.feature_QT_global)
cmp('QT max of lead medians vs feature',d.qt_max_der_leadmediane,d.feature_QT_global)
cmp('QTc Framingham code vs feature',d.qtc_fram_code,d.feature_QTc_fram)
cmp('RR median (global) vs feature RR',d.rr_median_global,d.feature_RR)
print('Feature QT median',d.feature_QT_global.median(),' 12SL',d.qt12.median(),' UniG',d.qtug.median())
print('Paired difference feature-ECGDeli minus 12SL median',(d.feature_QT_global-d.qt12).median(),' minus UniG',(d.feature_QT_global-d.qtug).median())
# per lead
l['d']=l.rebuilt-l.feature
g=l.groupby('lead').d.agg(n='count',exakt=lambda s:int((s.abs()<1e-9).sum()),med='median',mx=lambda s:s.abs().max())
print(g.reindex(['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']))
# aVL/aVF swap
p=l.pivot(index='ecg_id',columns='lead',values='rebuilt'); q=l.pivot(index='ecg_id',columns='lead',values='feature')
print('aVL rebuilt vs aVF feature exact',int(((p.aVL-q.aVF).abs()<1e-9).sum()),' aVF rebuilt vs aVL feature',int(((p.aVF-q.aVL).abs()<1e-9).sum()))
# decomposition
print('Q-onset(rank2) Median ms',d.on_rank2_med.median(),'| QRS onset margin: Q peak minus onset (median over leads/beats)',d.q_minus_on_med.median())
print('T peak->T end median',d.tpeak_to_toff_med.median(),' R->T end',d.rpk_to_toff_med.median())
# difference decomposition via median
print('QT difference to 12SL (median):',(d.feature_QT_global-d.qt12).median())
# ECGDeli QT rebuild in the broad/narrow stratum
print(d.groupby('stratum')[['feature_QT_global','qt12','qtug']].median())
# QT distribution
print(d.feature_QT_global.describe())
# with RR: QT/RR
d['rrm']=d.feature_RR
print('RR median ECGDeli',d.feature_RR.median(),' 12SL',d.rr12.median(),' UniG',d.rrug.median())
# Fraction in which the ECGDeli global T end (rank-2 median value) lies behind the next R? -> toff - R(global) vs RR
print('T end (R->T end) median/RR median:',(d.rpk_to_toff_med/d.feature_RR).median())
# QT per lead: median over records
print('Lead QT median (feature):',q.median().round(0).to_dict())
