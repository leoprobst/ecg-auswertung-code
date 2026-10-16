"""Independent recomputation of selected numbers (different code paths from aw211_teil1/2)."""
import os,sys,json
from pathlib import Path
os.environ.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from werkzeuge.b_test import aw211_common as C
import numpy as np,pandas as pd,statsmodels.api as sm
from scipy.special import expit
from werkzeuge.b_test.daten import classify_lvef
from werkzeuge.b_test.statistik import raw_design,fixed_design
from werkzeuge.b_test.aw157_models import multirc
from werkzeuge.nachrechnung.core import logistic_fit
A,F,AMP=C.A,C.F,C.AMP
out={'checks':[]}
def chk(name,a,b,tol):
    ok=bool(abs(a-b)<=tol);out['checks'].append({'name':name,'value':float(a),'reference':float(b),'tol':tol,'ok':ok});print(('OK  ' if ok else 'FAIL'),name,a,b,flush=True)

pt=C.load(C.OUT/'t1_point.json')['metrics']
f,_,scale=C.outcome_frame();q,_=C.pair_frame()
ref=f.loc[f.eligible_reference.eq(True)].drop_duplicates('subject_id')[['subject_id','study_id','preferred_lvef']].rename(columns={'study_id':'idx'})
ref['low']=(ref.preferred_lvef<40).astype(int)
m=q.merge(ref,on='subject_id',how='inner')
sdF=scale['sd'][-1]
def sd_icc(d,x1,x2):
    """SD of the differences and ICC(2,1) (absolute agreement, two-way) from the standard ANOVA, independent of aw157_models.icc."""
    n=len(d);k=2;sd=d.std(ddof=1);X=np.column_stack([x1,x2]);grand=X.mean()
    ssr=k*np.sum((X.mean(1)-grand)**2);ssc=n*np.sum((X.mean(0)-grand)**2);sst=np.sum((X-grand)**2);sse=sst-ssr-ssc
    msr=ssr/(n-1);msc=ssc/(k-1);mse=sse/((n-1)*(k-1))
    return sd,(msr-mse)/(msr+(k-1)*mse+k*(msc-mse)/n)
for st,mm in [('SB',m),('SA',m[m.first_study_id==m.idx])]:
    for g,gv in [('low',1),('pres',0)]:
        z=mm[mm.low==gv]
        a1,a2=z[A+'_first'].to_numpy(),z[A+'_second'].to_numpy();f1,f2=z[F+'_first'].to_numpy()/sdF,z[F+'_second'].to_numpy()/sdF
        s,i=sd_icc(a2-a1,a1,a2);chk(f'{st}_{g}_A_sd',s,pt[f'{st}_{g}_A_sd'],1e-9);chk(f'{st}_{g}_A_icc',i,pt[f'{st}_{g}_A_icc'],1e-6)
        s,i=sd_icc(f2-f1,f1,f2);chk(f'{st}_{g}_F_sd',s,pt[f'{st}_{g}_F_sd'],1e-9);chk(f'{st}_{g}_F_icc',i,pt[f'{st}_{g}_F_icc'],1e-6)
        chk(f'{st}_{g}_n',len(z),pt[f'{st}_{g}_n'],0)
amp=(q[AMP+'_first']+q[AMP+'_second'])/2
cuts=np.quantile(amp,[.2,.4,.6,.8]);qq=pd.cut(amp,[-np.inf,*cuts,np.inf],labels=False,right=False)+1
for k in (1,5):
    z=q[qq==k];a1,a2=z[A+'_first'].to_numpy(),z[A+'_second'].to_numpy();s,i=sd_icc(a2-a1,a1,a2)
    chk(f'Q{k}_A_sd',s,pt[f'Q{k}_A_sd'],1e-9);chk(f'Q{k}_A_icc',i,pt[f'Q{k}_A_icc'],1e-6)
# Quasi-Poisson crude = square root of the ratio of the mean delta^2
for st,mm in [('SB',m)]:
    d=mm[A+'_second']-mm[A+'_first'];r=np.sqrt((d[mm.low==1]**2).mean()/(d[mm.low==0]**2).mean());chk('SB_A_qp_crude',r,pt['SB_A_qp_crude_sdratio'],1e-9)

# Part 1b: EF_mix with the established core multirc
r1=C.load(C.OUT/'t1_results.json')
g=f.loc[f.eligible_B3.eq(True)].reset_index(drop=True);raw,names=raw_design(g,F);z=fixed_design(raw,scale);y=classify_lvef(g.preferred_lvef.to_numpy(float))
sdA=scale['sd'][-2]
mb=m.copy();dA=(mb[A+'_second']-mb[A+'_first']).to_numpy()/sdA;dF=((mb[F+'_second']-mb[F+'_first']).to_numpy())/sdF
dd=np.column_stack([dA,dF])
def Eg(mask):
    c=dd[mask]-dd[mask].mean(0);return c.T@c/(mask.sum()-1)/2
low=(mb.low==1).to_numpy();Ec=Eg(low);Ee=Eg(~low)
for fold in ['train','validation']:
    ix=g.split.eq(fold).to_numpy();zf,yf=z[ix],y[ix];w=np.ones(len(yf));b=logistic_fit(zf,yf,w);pi=yf.mean()
    E=(1-pi)*Ee+pi*Ec;res,_=multirc(zf,w,E,b)
    chk(f'{fold}_EFmix_logor',res['beta'],np.log(r1['boot2'][fold]['point']['logor_mix']),1e-8)
    k=r1['boot2'][fold]['point']['kappa_1.25']
    E1=(1-pi)*Ee+pi*k*Ec;E2=(1-pi)*Ee+pi*(k+0.01)*Ec
    a,_=multirc(zf,w,E1,b);bb,_=multirc(zf,w,E2,b)
    out['checks'].append({'name':f'{fold}_boundary_kappa','admissible_at_boundary':'beta' in a,'inadmissible_at_boundary_plus_0.01':'beta' not in bb});print(fold,'boundary',k,'beta' in a,'beta' not in bb,flush=True)

# Part 2: independent fit with statsmodels for one imputation
t2=C.load(C.OUT/'t2_results.json')
for fold in ['train','validation']:
    gg=f.loc[f.eligible_reference.eq(True)&f.split.eq(fold)].reset_index(drop=True);yy=classify_lvef(gg.preferred_lvef.to_numpy(float))
    zi=np.load(C.OUT/f't2_imp_{fold}.npz');k=0
    D=gg[['age','sex_numeric','heart_rate','qrs_duration_ms','qtc_fridericia_ms',A,F]].copy()
    cols={2:'heart_rate',3:'qrs_duration_ms',4:'qtc_fridericia_ms',5:A,6:F}
    miss={j:np.where(D[c].isna().to_numpy())[0] for j,c in cols.items()}
    for j,c in cols.items():D.loc[D.index[miss[j]],c]=zi[f'v{j}'][k]
    assert np.isfinite(D.to_numpy()).all()
    raw,_=raw_design(D,F);zz=fixed_design(raw,scale)
    sm_fit=sm.Logit(yy,zz).fit(disp=0,maxiter=100,method='newton')
    b=logistic_fit(zz,yy,np.ones(len(yy)));chk(f'{fold}_imp0_logOR_vs_statsmodels',b[-1],sm_fit.params[-1],1e-4)
    # Shift P2, S1, d=3: built independently
    Dm=D.copy();mF=np.zeros(len(D),bool);mF[miss[6]]=True
    shift=np.where(mF&(yy==0),3*0.76503231,0.0);Dm[F]=D[F]+shift
    raw2,_=raw_design(Dm,F);z2=fixed_design(raw2,scale);b2=logistic_fit(z2,yy,np.ones(len(yy)))
    out['checks'].append({'name':f'{fold}_imp0_P2_d3_logOR','value':float(b2[-1])});print(fold,'imp0 P2 d3 OR',np.exp(b2[-1]),flush=True)
    # imputed values against observed (plausibility)
    for cls,cv in [('cases',1),('controls',0)]:
        sel=(yy==cv);obsF=gg.loc[sel,F].dropna();imp=D.loc[sel&mF,F]
        out[f'{fold}_{cls}_F_obs_mean']=float(obsF.mean());out[f'{fold}_{cls}_F_imp_mean']=float(imp.mean());out[f'{fold}_{cls}_F_imp_n']=int(len(imp)) if len(imp)>=10 else '<10'
        print(fold,cls,'F obs',obsF.mean(),'imp',imp.mean(),flush=True)
# Tipping-point recheck P2/S1 and P3/S1 (train, imputations 0..9, coarse check of the direction)
C.dump(C.OUT/'verify.json',out);print('ALL OK' if all(c.get('ok',True) for c in out['checks']) else 'ERROR')
