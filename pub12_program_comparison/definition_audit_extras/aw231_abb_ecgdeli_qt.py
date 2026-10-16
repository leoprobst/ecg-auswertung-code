import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import numpy as np, pandas as pd, wfdb
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
src=open('aw231_qt.py',encoding='utf8').read().replace('\nmain()\n','\n'); ns={}; exec(compile(src,'q','exec'),ns)
fpt=ns['fpt']; nanrank=ns['nanrank']; LEADS=ns['LEADS']
OUT=(str(data_path("pub12.benchmark_pub12_aw231")) + "/")
smp=pd.read_csv(str(data_path("pub12.aw176_sample_csv")))
meta=pd.read_csv(str(data_path("pub12.research_diagnostic_identifiability_20261013_ptb_xl_1_0_3_ptbxl_database_csv")),usecols=['ecg_id','filename_hr']).set_index('ecg_id')
feat=pd.read_csv(OUT+'ecgdeli_qt_nachbau_aw231.csv').set_index('ecg_id')
sel=[]
for st in ('narrow','broad'):
    sel+=smp[smp.stratum==st].ecg_id.astype(int).tolist()[:2]
for e in sel:
    M=[fpt(e,L)[0] for L in LEADS]; B=min(len(m) for m in M)
    A=np.stack([m[:B] for m in M],axis=2); A[A<5]=np.nan
    sig=wfdb.rdrecord((str(data_path("pub12.aw176_signals")) + "/")+meta.loc[e,'filename_hr'])
    names=[{'AVR':'aVR','AVL':'aVL','AVF':'aVF'}.get(x,x) for x in sig.sig_name]
    k=B//2
    on=A[k,3,:]; toff=A[k,11,:]; r=np.nanmedian(A[k,5,:])
    on_r2=nanrank(on[None,:],2,False)[0]; toff_r2=nanrank(toff[None,:],2,True)[0]
    fig,axes=plt.subplots(4,3,figsize=(15,10),sharex=True)
    lo=int(r)-100; hi=int(r)+450
    for j,L in enumerate(LEADS):
        ax=axes.flat[j]; x=sig.p_signal[lo:hi,names.index(L)]; t=(np.arange(lo,hi)-r)*2
        ax.plot(t,x,color='k',lw=.8)
        if np.isfinite(on[j]): ax.axvline((on[j]-r)*2,color='tab:blue',lw=1.2)
        if np.isfinite(toff[j]): ax.axvline((toff[j]-r)*2,color='tab:red',lw=1.2)
        ax.axvline((on_r2-r)*2,color='tab:blue',ls=':',lw=1.5); ax.axvline((toff_r2-r)*2,color='tab:red',ls=':',lw=1.5)
        ax.set_title(L,fontsize=9); ax.grid(alpha=.2)
    fig.suptitle(f'PTB-XL+ ECGDeli (fiducial file): one beat, solid = lead (blue QRS onset, red T end), dotted = second earliest/second latest across leads\n(global QT of this beat {(toff_r2-on_r2)*2:.0f} ms; feature QT_Int_Global (median over beats) {feat.loc[e,"feature_QT_global"]:.0f} ms)',fontsize=9)
    fig.supxlabel('ms relative to the R peak')
    fig.tight_layout(rect=[0,0,1,.95]); fig.savefig(OUT+f'ecgdeli_qt_beispiel_{e:05d}.png',dpi=100); plt.close(fig)
print('done',len(sel))
