import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
from common import *
import pickle
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
with Path(str(data_path("a.kompendium_musterseite_v2_frank_processed_privat_pkl"))).open('rb') as f:records=pickle.load(f)
out=json.loads((OUT/'frank/ausreisser_privat.json').read_text());diagnostics=[]
for met in ['kors','inverse_dower']:
    for group in ['Healthy control','Myocardial infarction','uebrige']:
        sel=[r for r in out if r['method']==met and r['mode']=='integral' and r['stratum']==group]
        fig,axs=plt.subplots(5,4,figsize=(16,18),layout='constrained')
        for k,r in enumerate(sel):
            d=next(x['data'] for x in records if x['record']==r['record']);qb=d['bounds']['qrs'];tb=d['bounds']['t'];a=axs[k,0]
            for lead,col,off in [(1,'blue',0),(6,'green',.6)]:a.plot(d['grid_ms'],d['ecg'][:,lead]+off,lw=.8,color=col)
            for s,col in [(qb[0],'blue'),(qb[1],'blue'),(tb[0],'red'),(tb[1],'red')]:a.axvline((s-qb[0])*2,ls='--',color=col,lw=.8)
            a.set_xlim(-100,700);a.grid(alpha=.3);a.set_title(f"Rank{k+1}, Δ{r['difference']:.1f}°; II/V1")
            for j,(i,ii,sg1,sg2,label) in enumerate([(0,1,1,-1,'Frontal (anterior)'),(0,2,1,1,'Transverse (superior)'),(2,1,-1,-1,'Sagittal (right)')],1):
                for method,color in [('frank','black'),(met,'blue')]:
                    for seg,ls in [('qrs','-'),('t','--')]:
                        aa=d['segments'][method].get(seg)
                        if aa is not None:axs[k,j].plot(aa[:,i]*sg1,aa[:,ii]*sg2,ls,color=color,lw=.8)
                axs[k,j].set_aspect('equal',adjustable='datalim');axs[k,j].set_title(label);axs[k,j].grid(alpha=.3)
            diag={'method':met,'group':group,'rank':k+1,'patient':r['patient'],'record':r['record'],'difference':r['difference'],'qrs_duration_ms':(qb[1]-qb[0])*2,'t_duration_ms':(tb[1]-tb[0])*2}
            for method in ['frank',met]:
                for seg in ['qrs','t']:
                    aa=d['segments'][method][seg];diag[method+'_'+seg+'_peak_mv']=float(np.linalg.norm(aa,axis=1).max());diag[method+'_'+seg+'_integral_mvms']=float(np.linalg.norm(np.trapezoid(aa,dx=2,axis=0)))
            diagnostics.append(diag)
        fig.suptitle(group+' '+met+'; Frank black, reconstruction blue; QRS solid, T dashed')
        fig.savefig(OUT/'frank'/('atlas_'+group.replace(' ','_')+'_'+met+'.png'),dpi=100);plt.close(fig)
jsave(OUT/'frank/ausreisser_diagnostik_privat.json',diagnostics)
print('FRANK_ATLAS',len(diagnostics))
