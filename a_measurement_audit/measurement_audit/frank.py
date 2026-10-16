import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
from common import *
import pickle
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
src=Path(str(data_path("a.kompendium_musterseite_v2_frank_processed_privat_pkl")));dout=OUT/'frank';dout.mkdir(exist_ok=True)
with src.open('rb') as f:rr=pickle.load(f)
rr=[r for r in rr if r['primary']]
def angle(a,b):
    if min(np.linalg.norm(a),np.linalg.norm(b))<1e-9:return np.nan
    return np.degrees(np.arccos(np.clip(a@b/np.linalg.norm(a)/np.linalg.norm(b),-1,1)))
def av(segs,mode):
    if not all(s in segs for s in ['qrs','t']):return np.nan
    def v(a):return np.trapezoid(a,dx=2,axis=0) if mode=='integral' else a[np.argmax(np.linalg.norm(a,axis=1))]
    return angle(v(segs['qrs']),v(segs['t']))
rows=[];detail=[];outliers=[]
for method in ['kors','inverse_dower']:
    for mode in ['integral','peak']:
        bag=[]
        for r in rr:
            x=av(r['data']['segments']['frank'],mode);y=av(r['data']['segments'][method],mode)
            q=r['data']['segments']['frank'].get('qrs');amp=np.max(np.linalg.norm(q,axis=1)) if q is not None else np.nan
            bag.append({'patient':r['patient'],'record':r['record'],'group':r['group'],'frank':x,'reconstruction':y,'difference':y-x,'qrs_amplitude_mv':amp,'method':method,'mode':mode})
        detail+=bag
        for group in ['Healthy control','Myocardial infarction','uebrige','gesamt']:
            data=[r for r in bag if group=='gesamt' or r['group']==group or (group=='uebrige' and r['group'] not in ['Healthy control','Myocardial infarction'])]
            a=np.array([r['difference'] for r in data]);a=a[np.isfinite(a)];n=len(a)
            rng=np.random.default_rng(47009);idx=rng.integers(0,n,(1000,n));b=a[idx]
            sd=a.std(ddof=1);bias=a.mean();row={'pruefung':3,'method':method,'mode':mode,'gruppe':group,'n_candidate':len(data),'n':n,'bias':bias,'sd':sd,'p95_abs':np.quantile(abs(a),.95),'loa_low':bias-1.96*sd,'loa_high':bias+1.96*sd,'bias_ci':ci(b.mean(1)),'sd_ci':ci(b.std(1,ddof=1)),'p95_ci':ci(np.quantile(abs(b),.95,axis=1)),'source':str(src),'filter':'primary/group/method/mode','script':'werkzeuge/messaudit/frank.py'};rows.append(row)
            for r in sorted([r for r in data if np.isfinite(r['difference'])],key=lambda r:-abs(r['difference']))[:5]:
                if group=='gesamt':continue
                row0=next(x for x in rr if x['patient']==r['patient']);outliers.append({**r,'stratum':group,'diagnostics':row0['data']['diagnostics'],'bounds':row0['data']['bounds'],'origin_mad':row0['data']['origin_mad']})
                if mode!='integral':continue
                fig,ax=plt.subplots(1,3,figsize=(12,4),layout='constrained')
                for j,(i,k,xsg,sg,label) in enumerate([(0,1,1,-1,'Frontal von vorn'),(0,2,1,1,'Transversal von oben'),(2,1,-1,-1,'Rechts-sagittal')]):
                    for met,col in [('frank','black'),(method,'tab:blue')]:
                        for seg,ls in [('qrs','-'),('t','--')]:
                            aa=row0['data']['segments'][met].get(seg)
                            if aa is not None:ax[j].plot(aa[:,i]*xsg,aa[:,k]*sg,ls,color=col,label=met+'/'+seg)
                    ax[j].set_title(label);ax[j].set_aspect('equal',adjustable='datalim');ax[j].set_xlabel('mV');ax[j].grid(alpha=.2)
                ax[0].legend(fontsize=7);fig.suptitle(f"{group} {method}: Abweichung {r['difference']:.1f} Grad; QRS {r['qrs_amplitude_mv']:.3f} mV")
                fig.savefig(dout/(group.replace(' ','_')+'_'+method+'_'+r['patient']+'.png'),dpi=120);plt.close(fig)
                fig,ax=plt.subplots(3,4,figsize=(14,7),layout='constrained')
                ecg=row0['data']['ecg'];grid=row0['data']['grid_ms'];qb=row0['data']['bounds']['qrs'];tb=row0['data']['bounds']['t']
                for lead,a0 in enumerate(ax.flat):
                    a0.plot(grid,ecg[:,lead],lw=.8,color='black');a0.set_title(['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'][lead],fontsize=9);a0.grid(alpha=.2)
                    for sample0,col in [(qb[0],'blue'),(qb[1],'blue'),(tb[0],'red'),(tb[1],'red')]:a0.axvline((sample0-qb[0])*2,color=col,ls='--',lw=.8)
                    a0.set_xlim(-100,700)
                fig.suptitle(f"{group} {method} gemeinsames EKG/Delineation: QRS {(qb[1]-qb[0])*2} ms, T {(tb[1]-tb[0])*2} ms")
                fig.savefig(dout/(group.replace(' ','_')+'_'+method+'_'+r['patient']+'_ekg.png'),dpi=120);plt.close(fig)
pd.DataFrame(detail).to_csv(dout/'winkel_detail_privat.csv',index=False)
pd.DataFrame(rows).to_csv(dout/'winkel_aggregate.csv',index=False);jsave(dout/'ausreisser_privat.json',outliers);jsave(dout/'aggregate.json',rows)
bind([src,Path(str(data_path("a.kompendium_musterseite_v2_frank_freeze_json"))),ROOT/'werkzeuge/messaudit/frank.py',dout/'winkel_aggregate.csv'],'frank')
print(pd.DataFrame(rows)[['method','mode','gruppe','n','bias','sd','p95_abs']].to_string(index=False))
