import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
from statistik import *
from scipy.stats import spearmanr
import gc
dout=OUT/'transform';dout.mkdir(exist_ok=True)
src=Path(str(data_path("a.mimic_validation_features_primary_core_sentinel_discovery_reserve_classical_criteria_development_split_train")))
files=sorted(src.glob('part-*.parquet'));selected=files[::max(1,len(files)//60)][:60]
original=pq.read_table(RUN/'guideline/transform_agreement.parquet').to_pandas();markers=original.marker_id.tolist()
columns=['subject_id','study_id','extraction_status']+[x for m in markers for x in [m,'sensitivity_inverse_dower__'+m]]
for p in selected:assert set(columns)<=set(footer(p))
f=pd.concat([pd.read_parquet(p,columns=columns) for p in selected],ignore_index=True)
f=f.sort_values(['subject_id','study_id']).drop_duplicates('subject_id');n_candidate=len(f);f=f[f.extraction_status=='ok']
m=pd.read_parquet(OUT/'master_dev_privat.parquet',columns=['subject_id','study_id','machine_report','machine_qrs_duration_ms','machine_cart_id','ecg_time'])
f=f.merge(m,on=['subject_id','study_id'],how='left',validate='one_to_one');del m
s=f.machine_report.fillna('').str.lower();f['rhythm']=np.where(s.str.contains('atrial fibrillation|atrial flutter'),'AF',np.where(s.str.contains('sinus rhythm'),'Sinus','sonstige'))
f['qrs_class']=np.where(f.machine_qrs_duration_ms.isna(),'fehlend',np.where(f.machine_qrs_duration_ms>=120,'breit','schmal'))
f['pacing']=np.where(s.str.contains('paced|pacemaker'),'Reportpositiv','kein_Report')
f['block']=np.where(s.str.contains('left bundle|\\blbbb\\b'),'LBBB',np.where(s.str.contains('right bundle|\\brbbb\\b'),'RBBB','kein_Blockreport'))
f['cart']=f.machine_cart_id.fillna(-1).astype(int).astype(str);f['year']=pd.to_datetime(f.ecg_time).dt.year.astype('Int64').astype(str)
groups={'gesamt':np.ones(len(f),bool)}
for dim in ['rhythm','qrs_class','pacing','block','cart','year']:
    for val in sorted(f[dim].unique()):groups[dim+'/'+val]=(f[dim]==val).to_numpy()
jsave(dout/'nenner.json',{'n_candidate':n_candidate,'extraction_ok':len(f),'groups':{g:int(a.sum()) for g,a in groups.items()}})
def spearman_boot(x,y,B=1000):
    ux,ix=np.unique(x,return_inverse=True);uy,iy=np.unique(y,return_inverse=True);n=len(x)
    rng=np.random.default_rng(47009);values=[]
    for b in range(B):
        w=np.bincount(rng.integers(0,n,n),minlength=n);cx=np.bincount(ix,weights=w,minlength=len(ux));cy=np.bincount(iy,weights=w,minlength=len(uy))
        rx=(np.cumsum(cx)-cx/2+.5)[ix];ry=(np.cumsum(cy)-cy/2+.5)[iy]
        ax=rx-np.sum(w*rx)/n;ay=ry-np.sum(w*ry)/n
        den=np.sqrt(np.sum(w*ax*ax)*np.sum(w*ay*ay));values.append(np.sum(w*ax*ay)/den if den>0 else np.nan)
    return ci(values)
rows=[];audit=[]
for g,mask in groups.items():
    guard('transform_'+g)
    if mask.sum()<30:continue
    x=f.loc[mask,markers].to_numpy();y=f.loc[mask,['sensitivity_inverse_dower__'+m for m in markers]].to_numpy()
    point,boot,n,ns=pair_stats(x,y,f.loc[mask,'subject_id'].to_numpy())
    for j,marker in enumerate(markers):
        good=np.isfinite(x[:,j])&np.isfinite(y[:,j]);xx=x[good,j];yy=y[good,j];d=yy-xx
        rho=spearmanr(xx,yy).statistic if len(d)>1 and np.std(xx)>0 and np.std(yy)>0 else np.nan
        row={'marker_id':marker,'gruppe':g,'n':int(n[j]),'subjects':int(ns[j]),'bias':point[j,1],'sd':point[j,2],'loa_low':point[j,1]-point[j,3],'loa_high':point[j,1]+point[j,3],'spearman':rho,'paired_availability':float(n[j]/mask.sum()),'p95_abs':np.quantile(abs(d),.95) if len(d) else np.nan}
        for k,name in [(1,'bias'),(2,'sd')]:row[name+'_ci_low'],row[name+'_ci_high']=ci(boot[:,j,k]) if ns[j]>=30 else [np.nan,np.nan]
        row['loa_low_ci_low'],row['loa_low_ci_high']=ci(boot[:,j,1]-boot[:,j,3]) if ns[j]>=30 else [np.nan,np.nan]
        row['loa_high_ci_low'],row['loa_high_ci_high']=ci(boot[:,j,1]+boot[:,j,3]) if ns[j]>=30 else [np.nan,np.nan]
        # Correlation confidence interval for the rhythm/conduction/pacing strata and the total.
        row['spearman_ci_low'],row['spearman_ci_high']=spearman_boot(xx,yy) if ns[j]>=30 and np.isfinite(rho) else [np.nan,np.nan]
        rows.append(row)
        if g=='gesamt':
            o=original[original.marker_id==marker].iloc[0];audit.append({'marker_id':marker,'n_delta':int(n[j]-o.paired_count),'bias_delta':point[j,1]-o.bias,'rho_delta':rho-o.spearman_rho})
    pd.DataFrame(rows).to_csv(dout/'kennzahlen.csv',index=False);print('TRANSFORM',g,'n',int(mask.sum()),'rows',len(rows),flush=True);gc.collect()
jsave(dout/'reproduktion.json',audit)
bind([*selected,RUN/'guideline/transform_agreement.parquet',ROOT/'werkzeuge/messaudit/transformation.py',dout/'kennzahlen.csv'],'transform')
