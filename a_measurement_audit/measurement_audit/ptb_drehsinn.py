import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
from common import *
import pickle
with Path(str(data_path("a.kompendium_musterseite_v2_frank_processed_privat_pkl"))).open('rb') as f:records=pickle.load(f)
rows=[]
for seg in ['qrs','t']:
    for plane,(i,j,sx,sy) in {'frontal':(0,1,1,1),'transversal':(0,2,1,-1),'rechts_sagittal':(2,1,-1,1)}.items():
        values=[]
        for r in records:
            if not r['primary'] or r['group']!='Healthy control':continue
            a=r['data']['segments']['kors'].get(seg)
            if a is None or len(a)<4:continue
            xy=a[:,[i,j]]*np.array([sx,sy]);xy-=xy[0];cr=xy[:-1,0]*xy[1:,1]-xy[1:,0]*xy[:-1,1];pos=cr[cr>0].sum();neg=-cr[cr<0].sum()
            if pos+neg<1e-6:continue
            values.append((min(pos,neg)/(pos+neg),'cw' if pos>neg else 'ccw' if neg>pos else 'tie'))
        for threshold in [.05,.1,.15,.25,'majority']:
            labels=np.array(['figure8' if threshold!='majority' and v[0]>=threshold else v[1] for v in values]);rng=np.random.default_rng(47009);idx=rng.integers(0,len(labels),(1000,len(labels)))
            for cls in ['cw','figure8','ccw','tie']:
                q=labels==cls;rows.append({'segment':seg,'plane':plane,'threshold':threshold,'class':cls,'n_candidate':52,'n_available':len(labels),'count':int(q.sum()),'fraction':float(q.mean()),'ci':ci(q[idx].mean(1)),'note':'PTB healthy controls, primary recording, Kors transform, no age matching; exploratory addition'})
jsave(OUT/'frank/achter_healthy.json',rows)
print('PTB_TURN',len(rows))
