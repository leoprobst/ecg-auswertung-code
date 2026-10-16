"""Part 2: MNAR sensitivity by delta-adjusted multiple imputation (development partition only).

Phases: describe | impute | analyze. Contract: the pre-registered protocol of this stage (section 2).
"""
import os,sys,time,json
from pathlib import Path
os.environ.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from werkzeuge.b_test import aw211_common as C
PHASE=sys.argv[1]
C.OUT.mkdir(exist_ok=True)
sys.stdout=(C.OUT/('t2_'+PHASE+'_stdout.log')).open('a',encoding='utf8',buffering=1)
sys.stderr=(C.OUT/('t2_'+PHASE+'_stderr.log')).open('a',encoding='utf8',buffering=1)
C.guard();C.start_watch(1.0)
import numpy as np,pandas as pd
from scipy.special import expit,logit
from scipy.optimize import brentq
from scipy import stats
from werkzeuge.b_test.daten import classify_lvef
from werkzeuge.b_test.statistik import raw_design,fixed_design
from werkzeuge.nachrechnung.core import logistic_fit
A,F=C.A,C.F
SEED=2112026;M_IMP=100;MIN_IMP=95;FOLDS=['train','validation']
VARS=['age','sex_numeric','heart_rate','qrs_duration_ms','qtc_fridericia_ms',A,F]
MISSABLE=[2,3,4,5,6]
SDA=46.84175;SDF=0.76503231
GRID=[0,.25,.5,.75,1,1.5,2,3,4,6,8]
DMAX=8.0

def load_fold(fold):
    f,info,scale=C.outcome_frame()
    g=f.loc[f.eligible_reference.eq(True)&f.split.eq(fold)].copy()
    ok=np.isfinite(g.preferred_lvef.to_numpy(float))&np.isfinite(g.age.to_numpy(float))&np.isfinite(g.sex_numeric.to_numpy(float))
    excluded=int((~ok).sum());g=g.loc[ok].reset_index(drop=True)
    g['y']=classify_lvef(g.preferred_lvef.to_numpy(float))
    return g,scale,excluded,info

def to_transformed(g):
    X=np.column_stack([g.age.to_numpy(float),g.sex_numeric.to_numpy(float)]+[g[v].to_numpy(float) for v in VARS[2:]])
    T=X.copy()
    for j in (2,3,4):
        v=X[:,j];assert np.all(v[np.isfinite(v)]>0),'non-positive value in '+VARS[j];T[:,j]=np.log(v)
    a=X[:,5];T[:,5]=logit((a+0.5)/181.0)
    f=X[:,6];assert np.all(f[np.isfinite(f)]>=0);T[:,6]=np.log1p(f)
    return X,T
def back_transform(T):
    X=T.copy();X[:,2:5]=np.exp(T[:,2:5]);X[:,5]=np.clip(181.0*expit(T[:,5])-0.5,0.0,180.0);X[:,6]=np.maximum(np.expm1(T[:,6]),0.0)
    return X

# ------------------------------------------------------------- EM for MVN with missing values
def patterns_of(X):
    obs=np.isfinite(X);keys,inv=np.unique(obs,axis=0,return_inverse=True);inv=inv.ravel()
    return [(keys[i],np.where(inv==i)[0]) for i in range(len(keys))]
def em_mvn(X,maxit=500,tol=1e-7,ridge=1e-8):
    n,p=X.shape;mu0=np.nanmean(X,axis=0);sd0=np.nanstd(X,axis=0);sd0[sd0==0]=1.0
    Z=(X-mu0)/sd0;pats=patterns_of(Z);mu=np.zeros(p);Sg=np.eye(p)
    for it in range(maxit):
        S1=np.zeros(p);S2=np.zeros((p,p))
        for obs,idx in pats:
            o=np.where(obs)[0];m=np.where(~obs)[0];Zo=Z[np.ix_(idx,o)]
            full=np.empty((len(idx),p));full[:,o]=Zo
            if len(m):
                Soo=Sg[np.ix_(o,o)];Smo=Sg[np.ix_(m,o)];B=np.linalg.solve(Soo,Smo.T).T
                full[:,m]=mu[m]+(Zo-mu[o])@B.T;cc=Sg[np.ix_(m,m)]-B@Smo.T
                S2[np.ix_(m,m)]+=len(idx)*cc
            S1+=full.sum(0);S2+=full.T@full
        mun=S1/n;Sn=S2/n-np.outer(mun,mun);Sn=(Sn+Sn.T)/2+ridge*np.trace(Sn)/p*np.eye(p)
        ch=max(np.max(np.abs(mun-mu)),np.max(np.abs(Sn-Sg))/max(1e-12,np.max(np.abs(Sg))))
        mu,Sg=mun,Sn
        if ch<tol:return mu*sd0+mu0,Sg*np.outer(sd0,sd0),it+1,True
    return mu*sd0+mu0,Sg*np.outer(sd0,sd0),maxit,False
def draw_missing(X,mu,Sg,rng):
    """Draws all missing entries of the rows from the conditional normal distribution; returns a completed copy."""
    Y=X.copy()
    for obs,idx in patterns_of(X):
        m=np.where(~obs)[0]
        if not len(m):continue
        o=np.where(obs)[0];Soo=Sg[np.ix_(o,o)];Smo=Sg[np.ix_(m,o)];B=np.linalg.solve(Soo,Smo.T).T
        cm=mu[m]+(X[np.ix_(idx,o)]-mu[o])@B.T;cc=Sg[np.ix_(m,m)]-B@Smo.T;cc=(cc+cc.T)/2
        L=np.linalg.cholesky(cc+1e-10*np.eye(len(m)));Y[np.ix_(idx,m)]=cm+rng.standard_normal((len(idx),len(m)))@L.T
    return Y

# ------------------------------------------------------------- Phases
def describe():
    C.check_stamp();out={'protokoll':C.check_stamp(),'script_sha256':C.sha(__file__),'folds':{}}
    for fold in FOLDS:
        g,scale,excl,info=load_fold(fold);y=g.y.to_numpy();cc=g.eligible_B3.eq(True).to_numpy()
        d={'excluded_age_sex_lvef':excl,'n':int(len(g)),'footer':info}
        for cls,cv in [('cases',1.),('controls',0.)]:
            m=y==cv;n=int(m.sum());mis=~cc[m]
            def sup(k):return int(k) if k>=10 else '<10'
            d[cls]={'n':sup(n),'b3_incomplete':sup(mis.sum()),'b3_incomplete_share':float(mis.mean()),
                    'missing_by_var':{v:{'n':sup(g.loc[m,v].isna().sum()),'share':float(g.loc[m,v].isna().mean())} for v in VARS[2:]}}
        fobs=g[F].to_numpy(float);ok=np.isfinite(fobs)
        a=fobs[(y==1)&cc].mean();b=fobs[(y==0)&cc].mean();d['d_CC']=float((a-b)/SDF);d['mean_F_cases_cc']=float(a);d['mean_F_controls_cc']=float(b)
        other_miss=g[[VARS[2],VARS[3],VARS[4],A]].isna().any(axis=1).to_numpy()
        d['mar_check']={}
        for cls,cv in [('cases',1.),('controls',0.)]:
            m=(y==cv)&ok;x1=fobs[m&other_miss];x0=fobs[m&~other_miss]
            d['mar_check'][cls]={'n_other_missing_F_observed':int(len(x1)) if len(x1)>=10 else '<10','smd':float((x1.mean()-x0.mean())/SDF) if len(x1)>=10 else None}
        out['folds'][fold]=d
    C.dump(C.OUT/'t2_describe.json',out);print('DESCRIBE done',flush=True)

def impute():
    C.check_stamp()
    for fi,fold in enumerate(FOLDS):
        path=C.OUT/f't2_imp_{fold}.npz'
        g,scale,excl,info=load_fold(fold);X,T=to_transformed(g);y=g.y.to_numpy()
        miss=~np.isfinite(X);cells=[np.where(miss[:,j])[0] for j in range(X.shape[1])]
        # load state
        if path.exists():
            z=np.load(path,allow_pickle=True);vals=[list(map(np.asarray,z[f'v{j}'])) for j in range(len(VARS))];done=int(z['done']);conv=list(z['conv']);fails=int(z['fails'])
            vals=[list(v) for v in vals]
        else:
            vals=[[] for _ in VARS];done=0;conv=[];fails=0
        for mi in range(done,M_IMP):
            if mi%10==0:
                C.guard();print('IMP',fold,mi,flush=True)
                if mi>done:save(path,vals,mi,conv,fails,cells)
            T2=T.copy();ok_imp=True;info_conv=[]
            rng=np.random.default_rng(np.random.SeedSequence([SEED,3,fi,mi]))
            try:
                for cv in (0.,1.):
                    ix=np.where(y==cv)[0];Tc=T[ix]
                    bs=rng.integers(0,len(ix),len(ix));mu,Sg,it,cv_ok=em_mvn(Tc[bs]);info_conv.append((it,cv_ok))
                    if not cv_ok:raise RuntimeError('EM not converged')
                    T2[ix]=draw_missing(Tc,mu,Sg,rng)
            except Exception as ex:
                ok_imp=False;fails+=1;print('IMP_FAIL',fold,mi,str(ex)[:80],flush=True)
            if ok_imp:
                Xb=back_transform(T2)
                for j in MISSABLE:vals[j].append(Xb[cells[j],j].astype(np.float64))
                conv.append(info_conv[0][0]*1000+info_conv[1][0])
            else:
                for j in MISSABLE:vals[j].append(np.full(len(cells[j]),np.nan))
                conv.append(-1)
        save(path,vals,M_IMP,conv,fails,cells)
        C.dump(C.OUT/f't2_imp_{fold}_meta.json',{'fold':fold,'attempted':M_IMP,'failed':fails,'valid':M_IMP-fails,'seed':[SEED,3,fi],'cells':{VARS[j]:int(len(cells[j])) for j in MISSABLE},'em_iterations_coded':conv,'peak_rss':C.PEAK[0]})
        print('IMP done',fold,flush=True)

def save(path,vals,done,conv,fails,cells):
    kw={f'v{j}':np.array(vals[j],dtype=object) if False else (np.vstack(vals[j]) if len(vals[j]) else np.zeros((0,len(cells[j])))) for j in range(len(VARS))}
    np.savez_compressed(path,done=done,conv=np.array(conv),fails=fails,**kw)

# ------------------------------------------------------------- Analysis
def rubin(q,u):
    q=np.asarray(q);u=np.asarray(u);m=len(q);qb=q.mean();ub=u.mean();b=q.var(ddof=1);t=ub+(1+1/m)*b
    r=(1+1/m)*b/ub if ub>0 else np.inf;df=(m-1)*(1+1/r)**2 if r>0 else 1e6;df=min(df,1e6)
    h=stats.t.ppf(.975,df)*np.sqrt(t);return qb,t,df,qb-h,qb+h

class FoldAnalysis:
    def __init__(self,fold):
        self.fold=fold;fi=FOLDS.index(fold)
        g,scale,excl,info=load_fold(fold);self.g=g;self.scale=scale;self.y=g.y.to_numpy();n=len(g)
        X,T=to_transformed(g);self.X=X;miss=~np.isfinite(X)
        self.cells=[np.where(miss[:,j])[0] for j in range(X.shape[1])]
        z=np.load(C.OUT/f't2_imp_{fold}.npz');self.ok=[mi for mi in range(M_IMP) if np.isfinite(z['v6'][mi]).all() and np.isfinite(z['v5'][mi]).all()]
        assert len(self.ok)>=MIN_IMP,'too few valid imputations'
        self.vals={j:z[f'v{j}'] for j in MISSABLE}
        self.mean=np.array(scale['mean']);self.sd=np.array(scale['sd']);self.keep=np.array(scale['keep'],bool)
        assert self.keep[-1] and self.keep[-2]
        self.mF=np.zeros(n,bool);self.mF[self.cells[6]]=True;self.mA=np.zeros(n,bool);self.mA[self.cells[5]]=True
        self.Z=[];self.F0=[];self.A0=[]
        for mi in self.ok:
            D=g[VARS].copy()
            for j in MISSABLE:D.iloc[self.cells[j],j]=self.vals[j][mi]
            raw,names=raw_design(D,F);zz=fixed_design(raw,scale);self.Z.append(zz);self.F0.append(D[F].to_numpy(float));self.A0.append(D[A].to_numpy(float))
        self.b0=[logistic_fit(zz,self.y,np.ones(n)) for zz in self.Z]
        self.n=n;self.w=np.ones(n);self.cache={}
        self.sd_log=None
    def set_sdlog(self,v):self.sd_log=v
    def design(self,k,scheme,dc,dk):
        zz=self.Z[k]
        if dc==0 and dk==0:return zz,0.0
        z=zz.copy();d=np.where(self.y==1,dc,dk);fl=0.0
        mF=self.mF
        f=self.F0[k].copy()
        if scheme in ('S1','S3'):
            fn=np.maximum(0.0,f+d*SDF);fn=np.where(mF,fn,f);fl=float(np.mean(fn[mF&((d!=0))]==0.0)) if (mF&(d!=0)).any() else 0.0
        else:
            fn=np.expm1(np.log1p(f)+d*self.sd_log);fn=np.where(mF,fn,f)
        z[:,-1]=(fn-self.mean[-1])/self.sd[-1]
        if scheme=='S3':
            a=self.A0[k];an=np.clip(a+d*SDA,0.0,180.0);an=np.where(self.mA,an,a);z[:,-2]=(an-self.mean[-2])/self.sd[-2]
        return z,fl
    def fit(self,k,scheme,dc,dk):
        z,fl=self.design(k,scheme,dc,dk);b=logistic_fit(z,self.y,self.w,initial=self.b0[k])
        p=expit(z@b);H=z.T@(z*(p*(1-p))[:,None])+np.diag(np.r_[0,np.full(z.shape[1]-1,1e-6)]);var=np.linalg.inv(H)[-1,-1]
        return b[-1],var,fl
    def evaluate(self,scheme,dc,dk):
        key=(scheme,round(dc,6),round(dk,6))
        if key in self.cache:return self.cache[key]
        q=[];u=[];fl=[]
        for k in range(len(self.Z)):
            a,b,c=self.fit(k,scheme,dc,dk);q.append(a);u.append(b);fl.append(c)
        r={'q':np.array(q),'u':np.array(u),'floor_share':float(np.mean(fl))};self.cache[key]=r;return r

def summarize(r,idx=None):
    q=r['q'] if idx is None else r['q'][idx];u=r['u'] if idx is None else r['u'][idx]
    qb,t,df,lo,hi=rubin(q,u);return {'or':float(np.exp(qb)),'lo':float(np.exp(lo)),'hi':float(np.exp(hi)),'df':float(df)}

PATHS={'P1':lambda d:(-d,0.0),'P2':lambda d:(0.0,d),'P3':lambda d:(-d,d)}
MIRROR={'P1':lambda d:(d,0.0),'P2':lambda d:(0.0,-d),'P3':lambda d:(d,-d)}

def tip(fa,scheme,path,target,what,idx=None,lo_d=0.0,hi_d=DMAX,mirror=False):
    """Smallest d at which the pooled OR (what='or') or the lower Rubin limit (what='lo') falls to target."""
    P=(MIRROR if mirror else PATHS)[path]
    def val(d):
        dc,dk=P(d);return np.log(summarize(fa.evaluate(scheme,dc,dk),idx)[what])-np.log(target)
    v0=val(0.0)
    if v0<=0:return {'status':'bei d=0 schon unter Ziel','d':0.0}
    # Bracketing on the grid
    prev=0.0
    for d in GRID[1:]:
        if val(d)<=0:
            root=brentq(val,prev,d,xtol=1e-4);return {'status':'Kreuzung','d':float(root)}
        prev=d
    return {'status':'nicht erreicht bis 8','d':None,'or_at_8':float(np.exp(val(DMAX)+np.log(target)))}

def analyze():
    C.check_stamp();desc=C.load(C.OUT/'t2_describe.json');res={'protokoll':C.check_stamp(),'script_sha256':C.sha(__file__),'folds':{}}
    gtrain=load_fold('train')[0];obs=gtrain.loc[gtrain.eligible_B3.eq(True),F].to_numpy(float);sd_log=float(np.log1p(obs).std(ddof=0));res['sd_log_train']=sd_log
    gate={}
    for fold in FOLDS:
        t0=time.time();fa=FoldAnalysis(fold);fa.set_sdlog(sd_log)
        # CC gate
        cc=fa.g.eligible_B3.eq(True).to_numpy();raw,_=raw_design(fa.g.loc[cc],F);zc=fixed_design(raw,fa.scale);bcc=logistic_fit(zc,fa.y[cc],np.ones(cc.sum()))
        pub={'train':2.480,'validation':2.392}[fold];gate[fold]={'cc_or':float(np.exp(bcc[-1])),'published':pub,'ok':abs(np.exp(bcc[-1])-pub)<=0.002}
        if not gate[fold]['ok']:C.dump(C.OUT/'t2_results.json',{'gate':gate});raise SystemExit('STOP: Gate '+fold)
        d={'n_valid_imputations':len(fa.ok),'cc_or':gate[fold]['cc_or'],'n_rows':int(fa.n)}
        base=summarize(fa.evaluate('S1',0.0,0.0));d['mi_delta0']=base;d['mi_over_cc']=base['or']/gate[fold]['cc_or']
        d['mi_delta0_halves']=[summarize(fa.evaluate('S1',0.0,0.0),np.arange(len(fa.ok))[h::2]) for h in (0,1)]
        d['imputed_F_cells']={'cases':int(fa.mF[fa.y==1].sum()) if fa.mF[fa.y==1].sum()>=10 else '<10','controls':int(fa.mF[fa.y==0].sum()) if fa.mF[fa.y==0].sum()>=10 else '<10'}
        # Table grid
        tab=[]
        for scheme in ['S1','S2','S3']:
            for path in ['P1','P2','P3']:
                for dd in GRID:
                    dc,dk=PATHS[path](dd);r=fa.evaluate(scheme,dc,dk);tab.append({'scheme':scheme,'path':path,'direction':'abschwaechend','d':dd,**summarize(r),'floor_share':r['floor_share']})
        for path in ['P1','P2','P3']:
            for dd in GRID[1:]:
                dc,dk=MIRROR[path](dd);r=fa.evaluate('S1',dc,dk);tab.append({'scheme':'S1','path':path,'direction':'verstaerkend','d':dd,**summarize(r),'floor_share':r['floor_share']})
        d['grid']=tab;print('GRID',fold,time.time()-t0,flush=True)
        # Tipping points
        tips=[];half=[np.arange(len(fa.ok))[h::2] for h in (0,1)]
        for scheme in ['S1','S2','S3']:
            for path in ['P1','P2','P3']:
                for target,what,label in [(1.25,'or','OR=1,25'),(1.0,'or','OR=1,0'),(1.0,'lo','untere Grenze=1,0')]:
                    full=tip(fa,scheme,path,target,what)
                    hs=[tip(fa,scheme,path,target,what,idx=h) for h in half]
                    tips.append({'scheme':scheme,'path':path,'target':label,**full,'half1':hs[0],'half2':hs[1]})
            print('TIP',fold,scheme,time.time()-t0,flush=True)
        d['tips']=tips
        res['folds'][fold]=d
        C.dump(C.OUT/'t2_results_partial.json',{'gate':gate,'folds':res['folds'],'sd_log_train':sd_log})
    res['gate']=gate;res['describe']=desc
    C.dump(C.OUT/'t2_results.json',res);print('ANALYZED',flush=True)

if __name__=='__main__':
    try:
        C.dump(C.OUT/('t2_status_'+PHASE+'.json'),{'phase':PHASE,'state':'running','pid':os.getpid(),'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
        globals()[PHASE]()
        C.dump(C.OUT/('t2_status_'+PHASE+'.json'),{'phase':PHASE,'state':'complete','pid':os.getpid(),'peak_rss':C.PEAK[0],'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())});print('COMPLETE',PHASE)
    except BaseException as ex:
        C.dump(C.OUT/('t2_status_'+PHASE+'.json'),{'phase':PHASE,'state':'failed','pid':os.getpid(),'reason':str(ex)[:300]});raise
