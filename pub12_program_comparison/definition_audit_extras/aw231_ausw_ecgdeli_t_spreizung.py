import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import numpy as np, pandas as pd
src=open('aw231_qt.py',encoding='utf8').read().replace('\nmain()\n','\n'); ns={}; exec(compile(src,'q','exec'),ns)
fpt=ns['fpt']; nanrank=ns['nanrank']; LEADS=ns['LEADS']
d=pd.read_csv(str(data_path("pub12.benchmark_pub12_aw231_ecgdeli_qt_nachbau_aw231_csv")))
# Fraction of the lead T ends that lie > 100 ms behind the median over leads; median T peak->end
tot=0; late=0; q=[]
for e in d.ecg_id:
    M=[fpt(int(e),L)[0] for L in LEADS]; B=min(len(m) for m in M)
    A=np.stack([m[:B] for m in M],axis=2); A[A<5]=np.nan
    t=A[:,11,:]; med=np.nanmedian(t,axis=1)
    x=(t-med[:,None])*2; x=x[np.isfinite(x)]
    tot+=x.size; late+=(x>100).sum()
print('Lead T ends >100 ms behind the beat median of the 12 leads:',late,'of',tot,round(late/tot,3))
