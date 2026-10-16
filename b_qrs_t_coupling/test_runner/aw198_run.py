"""Bivariate regression calibration with tipping curve, development partition only (train/validation).

Phases: stamp | run | simulate | aggregate. Contract: the pre-registered protocol of this stage.
All person-level data stay in memory; outputs contain aggregates only.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os,sys,time,json,hashlib,ctypes,threading
from pathlib import Path
os.environ.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=Path(str(data_path("b.aw198")));OUT.mkdir(exist_ok=True)
PROTOKOLL=ROOT/'werkzeuge/b_test/aw198_protokoll.md'
PHASE=sys.argv[1]
sys.stdout=(OUT/(PHASE+'_stdout.log')).open('a',encoding='utf8',buffering=1)
sys.stderr=(OUT/(PHASE+'_stderr.log')).open('a',encoding='utf8',buffering=1)

class Mem(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ['total','free','page','commit','vtotal','vfree','extended']]
def memory():
    m=Mem();m.length=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m));return {'ram_free':m.free,'commit_free':m.commit}
def guard():
    if min(memory().values())<1.5*1024**3:raise SystemExit('less than 1.5 GB free: aborting')
guard()
import numpy as np,pandas as pd,pyarrow.parquet as pq,pyarrow.dataset as ds,psutil
from scipy.special import expit
from scipy.optimize import brentq
from werkzeuge.b_test.daten import classify_lvef
from werkzeuge.b_test.statistik import raw_design,fixed_design
from werkzeuge.nachrechnung.core import logistic_fit
from werkzeuge.b_test.aw157_models import rc,multirc

A='spatial_qrs_t_angle_deg';F='qrs_t_normalized_discrete_frechet'
SEED=1982026;NBOOT=1000;MINVALID=950
POP=Path(str(data_path("b.aw168_bef_0063_private_parquet")))
ENTRY=Path(str(data_path("b.aw168_bef_0063_entry_json")))
PAIRS=Path(str(data_path("b.aw157_pairs_private_parquet")))
RA_SCAN=np.round(np.arange(1000,49,-1)/1000,3)          # 1.000 ... 0.050
RA_REPORT=np.round(np.arange(30,101,5)/100,2)
RF=np.round(np.arange(50,101,5)/100,2)
RHO=np.array([-.4,-.3,-.2,-.1,0.,.1,.2,.268,.3,.329,.4,.5,.6,.7,.8])
TARGETS=[1.25,1.0]
ANCHORS_I={'I_gesamt':(.617,.837,.268),'I_proxy_sinus':(.693,.892,.268),'I_technisch_rho0268':(.746,.977,.268),'I_technisch_rho0':(.746,.977,0.),
           'Stress_AF':(.461,.826,.268),'Stress_LSB':(.309,.798,.268),'Nur_Winkel_0617':(.617,1.,0.)}
PEAK=0
def watch():
    global PEAK
    while True:
        PEAK=max(PEAK,psutil.Process().memory_info().rss)
        if PEAK>0.9*1024**3:os._exit(75)
        time.sleep(.5)
threading.Thread(target=watch,daemon=True).start()

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def clean(v):
    if isinstance(v,dict):return {str(k):clean(a) for k,a in v.items()}
    if isinstance(v,(list,tuple)):return [clean(a) for a in v]
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    return v
def dump(p,v):Path(p).write_text(json.dumps(clean(v),ensure_ascii=False,indent=1,allow_nan=False),encoding='utf8')
def load(p):return json.loads(Path(p).read_text(encoding='utf8'))

def footer_ok(p):
    """Test-set lock before every read: the split statistics must lie within train..validation and contain no nulls."""
    pf=pq.ParquetFile(p);names=pf.schema_arrow.names
    if 'split' not in names:raise SystemExit('STOP: no split column in '+str(p))
    i=names.index('split');out=[]
    for g in range(pf.metadata.num_row_groups):
        st=pf.metadata.row_group(g).column(i).statistics
        if st is None or not st.has_min_max:raise SystemExit('STOP: no split statistics')
        mn=st.min.decode() if isinstance(st.min,bytes) else st.min;mx=st.max.decode() if isinstance(st.max,bytes) else st.max
        if mn<'train' or mx>'validation' or st.null_count:raise SystemExit('STOP: possible test rows in '+str(p))
        out.append([mn,mx,int(st.null_count)])
    return {'rows':pf.metadata.num_rows,'rowgroup_split_min_max_null':out}
def read(p,cols):
    guard();info=footer_ok(p)
    t=ds.dataset(str(p),format='parquet').to_table(columns=cols,filter=ds.field('split').isin(['train','validation'])).to_pandas()
    assert set(t.split.unique())<={'train','validation'} and len(t)==info['rows']
    return t,info

def data():
    entry=load(ENTRY);scale=entry['scaling_B3']
    cols=['subject_id','split','age','sex_numeric','heart_rate','qrs_duration_ms','qtc_fridericia_ms',A,F,'preferred_lvef','eligible_B3']
    f,info=read(POP,cols);f=f.loc[f.eligible_B3.eq(True)].reset_index(drop=True)
    raw,names=raw_design(f,F);z=fixed_design(raw,scale);y=classify_lvef(f.preferred_lvef.to_numpy())
    pcols=['subject_id','split','hours_between',A+'_first',A+'_second',F+'_first',F+'_second']
    p,pinfo=read(PAIRS,pcols)
    assert np.isfinite(p[pcols[3:]].to_numpy(float)).all()
    folds={}
    for fold in ['train','validation']:
        ix=f.split.eq(fold).to_numpy();folds[fold]={'z':z[ix],'y':y[ix],'ids':f.subject_id.to_numpy()[ix]}
    sd=np.array(scale['sd'][-2:])
    delta=np.column_stack([p[A+'_second']-p[A+'_first'],p[F+'_second']-p[F+'_first']]).astype(float)/sd
    pv={'delta':delta,'ids':p.subject_id.to_numpy(),'zeitgleich':p.hours_between.eq(0).to_numpy(),
        'x':np.column_stack([p[A+'_first'],p[A+'_second'],p[F+'_first'],p[F+'_second']]).astype(float)}
    meta={'population':{'path':str(POP),'sha256':sha(POP),**info,'eligible_B3_n':{k:int(len(v['y'])) for k,v in folds.items()},'cases':{k:int(v['y'].sum()) for k,v in folds.items()}},
          'pairs':{'path':str(PAIRS),'sha256':sha(PAIRS),**pinfo,'pairs':int(len(p)),'persons':int(p.subject_id.nunique()),'zeitgleich_pairs':int(pv['zeitgleich'].sum()),'zeitgleich_persons':int(p.loc[pv['zeitgleich']].subject_id.nunique())},
          'entry':{'path':str(ENTRY),'sha256':sha(ENTRY)},'names':['intercept']+names,'scaling_sd':scale['sd'],'test_access':False}
    return folds,pv,meta

def wcov(x,w):
    c=x-np.average(x,axis=0,weights=w);return c.T@(w[:,None]*c)/w.sum()
def pair_error(delta,w):
    c=delta-np.average(delta,axis=0,weights=w);return c.T@(w[:,None]*c)/(w.sum()-1)/2
def components(z,y,w,b=None):
    if b is None:b=logistic_fit(z,y,w)
    pr=expit(z@b);score=np.max(np.abs(z.T@(w*(pr-y))+np.r_[0,np.full(z.shape[1]-1,1e-6)]*b))/w.sum()
    if not np.isfinite(b).all() or score>=1e-7:raise ValueError('not converged')
    other=z[:,:-2];W=z[:,-2:]
    coef=np.linalg.solve(other.T@(w[:,None]*other),other.T@(w[:,None]*W));S=wcov(W-other@coef,w);V=wcov(W,w)
    return {'b':b,'S':S,'V':V,'coef':coef,'u':S@b[-2:]}

def grid(c,RA,RFv,RHOv):
    """Corrected marker log-OR and admissibility; axes (RF, rho, RA) are broadcast."""
    S=c['S'];V=c['V'];u=c['u']
    RA=np.asarray(RA,float)[None,None,:];RFv=np.asarray(RFv,float)[:,None,None];RHOv=np.asarray(RHOv,float)[None,:,None]
    eA=(1-RA)*V[0,0];eF=(1-RFv)*V[1,1];eAF=RHOv*np.sqrt(eA*eF)
    T11=S[0,0]-eA;T22=S[1,1]-eF;T12=S[0,1]-eAF
    T11,T22,T12=np.broadcast_arrays(T11,T22,T12)
    det=T11*T22-T12**2;lam=(T11+T22-np.sqrt((T11-T22)**2+4*T12**2))/2
    ok=lam>1e-10
    with np.errstate(divide='ignore',invalid='ignore'):
        cF=(-T12*u[0]+T11*u[1])/det;cA=(T22*u[0]-T12*u[1])/det
        pcorr=T12/np.sqrt(T11*T22)
        M11=V[0,0]-eA;M22=V[1,1]-eF;M12=V[0,1]-eAF;mcorr=M12/np.sqrt(M11*M22)
    cF=np.where(ok,cF,np.nan);cA=np.where(ok,cA,np.nan)
    return {'logor':cF,'angle_logor':cA,'ok':ok,'pcorr':np.where(ok,pcorr,np.nan),'mcorr':np.where(ok,mcorr,np.nan)}

def kipp(c,RFv,RHOv,target,refine=False):
    """First downward crossing when lowering R_A; returns (R_A*, status, bound, pcorr, mcorr)."""
    g=grid(c,RA_SCAN,RFv,RHOv);L=g['logor'];ok=g['ok'];lt=np.log(target)
    nF,nR,nA=L.shape;ra=np.full((nF,nR),np.nan);status=np.zeros((nF,nR),int);bound=np.full((nF,nR),np.nan)
    pc=np.full((nF,nR),np.nan);mc=np.full((nF,nR),np.nan)
    # status: 1 crossing, 2 bound before crossing, 3 inadmissible at R_A=1, 4 already below target at R_A=1, 5 no crossing down to 0.05
    for i in range(nF):
        for j in range(nR):
            bad=(~ok[i,j])|(L[i,j]<lt)
            if not bad.any():status[i,j]=5;continue
            k=int(np.argmax(bad))
            if not ok[i,j,k]:
                status[i,j]=3 if k==0 else 2;bound[i,j]=RA_SCAN[k-1] if k>0 else np.nan;continue
            if k==0:status[i,j]=4;ra[i,j]=1.;continue
            x0,x1=RA_SCAN[k-1],RA_SCAN[k];y0,y1=L[i,j,k-1],L[i,j,k]
            r=x0+(lt-y0)*(x1-x0)/(y1-y0)
            if refine:
                fun=lambda a:grid(c,[a],[RFv[i]],[RHOv[j]])['logor'][0,0,0]-lt
                r=brentq(fun,x1,x0,xtol=1e-12)
            ra[i,j]=r;status[i,j]=1
            gg=grid(c,[r],[RFv[i]],[RHOv[j]]);pc[i,j]=gg['pcorr'][0,0,0];mc[i,j]=gg['mcorr'][0,0,0]
            # admissibility bound below the crossing (for the distance)
            rest=~ok[i,j,k:]
            bound[i,j]=RA_SCAN[k+int(np.argmax(rest))-1] if rest.any() else np.nan
    return {'ra':ra,'status':status,'bound':bound,'pcorr':pc,'mcorr':mc}

def anchor_rel(c,E):
    V=c['V'];return {'R_A':1-E[0,0]/V[0,0],'R_F':1-E[1,1]/V[1,1],'rho':E[0,1]/np.sqrt(E[0,0]*E[1,1])}
def corrected_E(c,E):
    S=c['S'];T=S-E;lam=np.linalg.eigvalsh(T).min()
    if lam<=1e-10:return np.nan,np.nan,lam
    cw=np.linalg.solve(T,c['u']);return cw[1],cw[0],lam
def cross_repeat(x,w):
    def r(a,b):
        ma=np.average(a,weights=w);mb=np.average(b,weights=w)
        return np.average((a-ma)*(b-mb),weights=w)/np.sqrt(np.average((a-ma)**2,weights=w)*np.average((b-mb)**2,weights=w))
    a1,a2,f1,f2=x.T;rA=r(a1,a2);rF=r(f1,f2);rx=(r(a1,f2)+r(f1,a2))/2
    return {'r_A1A2':rA,'r_F1F2':rF,'r_cross':rx,'latent_corr':rx/np.sqrt(rA*rF)}

def evaluate(c,pv,pw):
    out={'b':c['b'][-2:],'naive_logor':c['b'][-1],'V':c['V'],'S':c['S']}
    out['grid']=grid(c,RA_REPORT,RF,RHO)
    out['kipp']={t:kipp(c,RF,RHO,t) for t in TARGETS}
    anchors={}
    for name,mask in [('P_gesamt',np.ones(len(pw),bool)),('P_zeitgleich',pv['zeitgleich'])]:
        E=pair_error(pv['delta'][mask],pw[mask]);rel=anchor_rel(c,E);lf,la,lam=corrected_E(c,E)
        k={t:kipp(c,[rel['R_F']],[rel['rho']],t) for t in TARGETS}
        anchors[name]={'E':E,**rel,'logor':lf,'angle_logor':la,'lam_min':lam,'kipp':{t:{q:v[0,0] for q,v in k[t].items()} for t in TARGETS}}
    for name,(ra,rf,rho) in ANCHORS_I.items():
        g=grid(c,[ra],[rf],[rho]);k={t:kipp(c,[rf],[rho],t) for t in TARGETS}
        anchors[name]={'R_A':ra,'R_F':rf,'rho':rho,'logor':g['logor'][0,0,0],'angle_logor':g['angle_logor'][0,0,0],'kipp':{t:{q:v[0,0] for q,v in k[t].items()} for t in TARGETS}}
    out['anchors']=anchors
    out['cross_repeat']=cross_repeat(pv['x'],pw)
    out['cross_repeat_zeitgleich']=cross_repeat(pv['x'][pv['zeitgleich']],pw[pv['zeitgleich']]) if pw[pv['zeitgleich']].sum()>2 else None
    return out

def stamp():
    p=OUT/'protokoll_hash.json'
    if p.exists():raise SystemExit('Hash already bound')
    dump(p,{'path':str(PROTOKOLL),'sha256':sha(PROTOKOLL),'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'before_computation':True})
    print('STAMP',sha(PROTOKOLL))

def check_stamp():
    s=load(OUT/'protokoll_hash.json');assert s['sha256']==sha(PROTOKOLL),'Protocol modified after binding';return s

def run():
    stamp_=check_stamp();folds,pv,meta=data();dump(OUT/'inputs_manifest.json',{**meta,'protokoll':stamp_,'script_sha256':sha(__file__)})
    point={};checks={}
    pub={'train':{'naive':2.480,'joint_all':8.737,'rc_0617':2.369},'validation':{'naive':2.392,'joint_all':9.259,'rc_0617':2.656}}
    for fi,fold in enumerate(['train','validation']):
        z,y=folds[fold]['z'],folds[fold]['y'];w=np.ones(len(y));pw=np.ones(len(pv['ids']))
        c=components(z,y,w);ev=evaluate(c,pv,pw);point[fold]=ev
        # Gate: closed form against multirc/rc and against the published values
        E=ev['anchors']['P_gesamt']['E'];m=multirc(z,w,E,c['b'])[0]
        V=c['V'];eA=(1-.617)*V[0,0];r1=rc(z,w,eA,c['b'])[0]
        diffs={'multirc_P_gesamt':abs(m['beta']-ev['anchors']['P_gesamt']['logor']),'rc_0617':abs(r1['beta']-ev['anchors']['Nur_Winkel_0617']['logor'])}
        for nm,(ra,rf,rho) in ANCHORS_I.items():
            g=ev['anchors'][nm]['logor'];eA=(1-ra)*V[0,0];eF=(1-rf)*V[1,1];EE=np.array([[eA,rho*np.sqrt(eA*eF)],[rho*np.sqrt(eA*eF),eF]])
            mm=multirc(z,w,EE,c['b'])[0]
            if 'beta' in mm:diffs['multirc_'+nm]=abs(mm['beta']-g) if np.isfinite(g) else np.inf
            else:diffs['multirc_'+nm]=0. if not np.isfinite(g) else np.inf
        repro={'naive':np.exp(c['b'][-1]),'joint_all':np.exp(ev['anchors']['P_gesamt']['logor']),'rc_0617':np.exp(ev['anchors']['Nur_Winkel_0617']['logor'])}
        gate=max(diffs.values())<1e-8 and all(abs(repro[k]-pub[fold][k])<=0.002 for k in repro)
        checks[fold]={'kernel_abs_logor_diff':diffs,'reproduced_or':repro,'published_or':pub[fold],'gate_passed':gate}
        # precise tipping points of the point estimate
        point[fold]['kipp_refined']={t:kipp(c,RF,RHO,t,refine=True) for t in TARGETS}
        print('POINT',fold,checks[fold],flush=True)
    dump(OUT/'checks.json',checks)
    if not all(v['gate_passed'] for v in checks.values()):dump(OUT/'point.json',point);raise SystemExit('STOP: reproduction gate not passed')
    dump(OUT/'point.json',point)
    # Bootstrap
    for fi,fold in enumerate(['train','validation']):
        path=OUT/('boot_'+fold+'.npz')
        if path.exists():continue
        z,y,ids=folds[fold]['z'],folds[fold]['y'],folds[fold]['ids']
        union=np.union1d(ids,pv['ids']);oi=np.searchsorted(union,ids);pi=np.searchsorted(union,pv['ids'])
        b0=point[fold]['b'];base=components(z,y,np.ones(len(y)))['b']
        keys=['naive','grid','kipp_ra','kipp_status','kipp_bound','kipp_pcorr','anc_logor','anc_kipp','anc_rel','cross']
        store={k:[] for k in keys};fails={};valid=0
        anames=['P_gesamt','P_zeitgleich']+list(ANCHORS_I)
        for r in range(NBOOT):
            if r%50==0:guard();print('BOOT',fold,r,valid,flush=True)
            rng=np.random.default_rng(np.random.SeedSequence([SEED,fi,r]));mult=np.bincount(rng.integers(0,len(union),len(union)),minlength=len(union)).astype(float)
            w=mult[oi];pw=mult[pi]
            try:
                c=components(z,y,w,logistic_fit(z,y,w,initial=base));ev=evaluate(c,pv,pw)
            except Exception as ex:fails[str(ex)]=fails.get(str(ex),0)+1;continue
            valid+=1
            store['naive'].append(ev['naive_logor']);store['grid'].append(ev['grid']['logor'].astype(np.float32))
            store['kipp_ra'].append(np.stack([ev['kipp'][t]['ra'] for t in TARGETS]).astype(np.float32))
            store['kipp_status'].append(np.stack([ev['kipp'][t]['status'] for t in TARGETS]).astype(np.int8))
            store['kipp_bound'].append(np.stack([ev['kipp'][t]['bound'] for t in TARGETS]).astype(np.float32))
            store['kipp_pcorr'].append(np.stack([ev['kipp'][t]['pcorr'] for t in TARGETS]).astype(np.float32))
            store['anc_logor'].append([ev['anchors'][a]['logor'] for a in anames])
            store['anc_kipp'].append([[ev['anchors'][a]['kipp'][t]['ra'] for t in TARGETS] for a in anames])
            store['anc_rel'].append([[ev['anchors'][a]['R_A'],ev['anchors'][a]['R_F'],ev['anchors'][a]['rho']] for a in anames])
            cr=ev['cross_repeat'];cz=ev['cross_repeat_zeitgleich'] or {}
            store['cross'].append([cr['latent_corr'],cr['r_A1A2'],cr['r_F1F2'],cz.get('latent_corr',np.nan)])
        np.savez_compressed(path,**{k:np.asarray(v,dtype=float if k in ('naive','anc_logor','anc_kipp','anc_rel','cross') else None) for k,v in store.items()},anchor_names=np.array(anames))
        dump(OUT/('boot_'+fold+'_meta.json'),{'fold':fold,'attempted':NBOOT,'valid':valid,'failures':fails,'union_persons':int(len(union)),'seed':[SEED,fi],'peak_rss':PEAK})
        print('BOOT_DONE',fold,valid,fails,flush=True)

def simulate():
    check_stamp();folds,pv,meta=data();point=load(OUT/'point.json')
    z,y=folds['train']['z'],folds['train']['y'];w=np.ones(len(y));c=components(z,y,w)
    E_all=pair_error(pv['delta'],np.ones(len(pv['ids'])));rel=anchor_rel(c,E_all)
    scen=[]
    for t in TARGETS:
        for nm,(rf,rho) in [('P_gesamt',(rel['R_F'],rel['rho'])),('Nur_Winkel',(1.,0.))]:
            k=kipp(c,[rf],[rho],t,refine=True)
            scen.append({'name':nm,'target':t,'R_A':float(k['ra'][0,0]),'R_F':float(rf),'rho':float(rho),'status':int(k['status'][0,0])})
    other=z[:,:-2];res=[]
    for si,s in enumerate(scen):
        if s['status']!=1:s['result']='kein Kipppunkt, keine Simulation';res.append(s);continue
        V=c['V'];eA=(1-s['R_A'])*V[0,0];eF=(1-s['R_F'])*V[1,1];E=np.array([[eA,s['rho']*np.sqrt(eA*eF)],[s['rho']*np.sqrt(eA*eF),eF]])
        T=c['S']-E;cw=np.linalg.solve(T,c['u']);co=c['b'][:-2]-c['coef']@(cw-c['b'][-2:]);truth=np.r_[co,cw]
        Lt=np.linalg.cholesky(T);LE=np.linalg.cholesky(E+np.eye(2)*1e-14) if np.linalg.eigvalsh(E).min()>0 else None
        rows=[]
        for j in range(200):
            if j%25==0:guard()
            rng=np.random.default_rng(np.random.SeedSequence([SEED,77,si,j]))
            Wl=other@c['coef']+rng.standard_normal((len(y),2))@Lt.T
            ys=rng.binomial(1,expit(other@co+Wl@cw)).astype(float)
            err=rng.standard_normal((len(y),2))@(LE.T if LE is not None else np.diag(np.sqrt(np.diag(E))))
            Wm=Wl+err;zm=np.column_stack([other,Wm]);zl=np.column_stack([other,Wl])
            try:
                cm=components(zm,ys,w);lf,la,lam=corrected_E(cm,E);bo=logistic_fit(zl,ys,w)
                rows.append([cm['b'][-1],lf,bo[-1],cm['b'][-2],la])
            except Exception as ex:rows.append([np.nan]*5)
        R=np.array(rows)
        def summ(v):
            v=v[np.isfinite(v)];return {'n':int(len(v)),'mean_logor':float(v.mean()),'gm_or':float(np.exp(v.mean())),'q':np.exp(np.quantile(v,[.025,.975])).tolist()} if len(v) else None
        s['true_marker_or']=float(np.exp(cw[1]));s['true_angle_or']=float(np.exp(cw[0]))
        s['naive']=summ(R[:,0]);s['rc']=summ(R[:,1]);s['oracle']=summ(R[:,2]);s['naive_angle']=summ(R[:,3])
        s['rc_inadmissible']=int(np.sum(~np.isfinite(R[:,1])));s['observed_naive_or_train']=float(np.exp(c['b'][-1]))
        s['naive_ge_observed_share']=float(np.mean(R[:,0]>=c['b'][-1]))
        s['rc_bias_logor']=float(np.nanmean(R[:,1])-cw[1]);res.append(s);print('SIM',s['name'],s['target'],s['naive'],s['rc'],flush=True)
    dump(OUT/'simulation.json',{'scenarios':res,'n_per_dataset':int(len(y)),'datasets':200,'seed':[SEED,77]})

STATUS={1:'Kreuzung',2:'Grenze vor Kreuzung',3:'unzulaessig bei R_A=1',4:'bei R_A=1 unter Ziel',5:'keine Kreuzung bis 0,05'}
PROFILES={'P_gesamt':None,'Nur_Winkel':(1.,0.),'technisch_0977_0268':(.977,.268),'RF085_rho06':(.85,.6),'RF09_rho0329':(.9,.329)}

def aggregate():
    check_stamp();import csv
    point=load(OUT/'point.json');res={'folds':{}};rows_k=[];rows_g=[];rows_a=[];rows_p=[]
    def ci(v):
        v=np.asarray(v,float);v=v[np.isfinite(v)];return [float(np.quantile(v,.025)),float(np.quantile(v,.975))] if len(v)>=MINVALID else None
    for fold in ['train','validation']:
        p=point[fold];bz=np.load(OUT/('boot_'+fold+'.npz'));bm=load(OUT/('boot_'+fold+'_meta.json'));nv=bm['valid']
        anames=list(bz['anchor_names']);res['folds'][fold]={'boot':bm,'naive_or':float(np.exp(p['naive_logor'])),'naive_or_ci':np.exp(ci(bz['naive'])).tolist() if ci(bz['naive']) else None}
        for ti,t in enumerate(TARGETS):
            k=p['kipp_refined'][str(t)];ra=np.array(k['ra'],float);st=np.array(k['status']);bd=np.array(k['bound'],float);pc=np.array(k['pcorr'],float);mc=np.array(k['mcorr'],float)
            for i,rf in enumerate(RF):
                for j,rho in enumerate(RHO):
                    bra=bz['kipp_ra'][:,ti,i,j];bst=bz['kipp_status'][:,ti,i,j];fin=np.isfinite(bra)&(bst==1)
                    counts={STATUS[s]:int((bst==s).sum()) for s in STATUS if (bst==s).sum()}
                    c=[float(np.quantile(bra[fin],.025)),float(np.quantile(bra[fin],.975))] if fin.sum()>=MINVALID else None
                    rows_k.append({'fold':fold,'target_or':t,'R_F':rf,'rho':rho,'R_A_kipp':ra[i,j] if st[i,j] in (1,4) else None,'status':STATUS[int(st[i,j])],
                        'R_A_zulaessig_min':bd[i,j],'abstand_zur_grenze':ra[i,j]-bd[i,j] if st[i,j]==1 else None,'partielle_latente_korr':pc[i,j] if st[i,j]==1 else None,'marginale_latente_korr':mc[i,j] if st[i,j]==1 else None,
                        'boot_gueltig':nv,'boot_kreuzungen':int(fin.sum()),'boot_ci_low':c[0] if c else None,'boot_ci_high':c[1] if c else None,'boot_status':json.dumps(counts,ensure_ascii=False)})
        L=np.array(p['grid']['logor'],float)
        for i,rf in enumerate(RF):
            for j,rho in enumerate(RHO):
                for a,ra in enumerate(RA_REPORT):
                    b=bz['grid'][:,i,j,a];c=ci(b)
                    rows_g.append({'fold':fold,'R_A':ra,'R_F':rf,'rho':rho,'or':float(np.exp(L[i,j,a])) if np.isfinite(L[i,j,a]) else None,'boot_zulaessig':int(np.isfinite(b).sum()),'boot_gueltig':nv,
                                   'or_ci_low':float(np.exp(c[0])) if c else None,'or_ci_high':float(np.exp(c[1])) if c else None})
        for ai,a in enumerate(anames):
            v=p['anchors'][a];lo=bz['anc_logor'][:,ai];c=ci(lo)
            row={'fold':fold,'anker':a,'R_A':v['R_A'],'R_F':v['R_F'],'rho':v['rho'],'or':float(np.exp(v['logor'])) if v['logor'] is not None else None,'boot_zulaessig':int(np.isfinite(lo).sum()),'boot_gueltig':nv,
                 'or_ci_low':float(np.exp(c[0])) if c else None,'or_ci_high':float(np.exp(c[1])) if c else None}
            if a.startswith('P_'):
                for q,nm in enumerate(['R_A','R_F','rho']):cc=ci(bz['anc_rel'][:,ai,q]);row[nm+'_ci']=cc
            for ti,t in enumerate(TARGETS):
                kk=v['kipp'][str(t)];row['kipp_'+str(t)]=kk['ra'];row['kipp_status_'+str(t)]=STATUS[int(kk['status'])];row['grenze']=kk['bound']
                bk=bz['anc_kipp'][:,ai,ti];row['boot_kreuzungen_'+str(t)]=int(np.isfinite(bk).sum())
            rows_a.append(row)
        cr=p['cross_repeat'];res['folds'][fold]['cross_repeat']={**cr,'latent_corr_ci':ci(bz['cross'][:,0]),'zeitgleich':p['cross_repeat_zeitgleich'],'zeitgleich_latent_corr_ci':ci(bz['cross'][:,3])}
        # fine profiles of the point estimate
        c={'S':np.array(p['S']),'V':np.array(p['V']),'u':np.array(p['S'])@np.array(p['b'])}
        for nm,val in PROFILES.items():
            rf,rho=(p['anchors']['P_gesamt']['R_F'],p['anchors']['P_gesamt']['rho']) if val is None else val
            g=grid(c,RA_SCAN,[rf],[rho])
            for a,ra in enumerate(RA_SCAN):
                if round(ra*1000)%5==0 or (g['ok'][0,0,a] and not g['ok'][0,0,min(a+1,len(RA_SCAN)-1)]):
                    rows_p.append({'fold':fold,'profil':nm,'R_F':rf,'rho':rho,'R_A':ra,'zulaessig':bool(g['ok'][0,0,a]),'or':float(np.exp(g['logor'][0,0,a])) if g['ok'][0,0,a] else None,
                                   'winkel_or':float(np.exp(g['angle_logor'][0,0,a])) if g['ok'][0,0,a] else None,'partielle_latente_korr':float(g['pcorr'][0,0,a]) if g['ok'][0,0,a] else None})
    for name,rows in [('kippkurve.csv',rows_k),('gitter_or.csv',rows_g),('anker.csv',rows_a),('profile.csv',rows_p)]:
        with (OUT/name).open('w',newline='',encoding='utf8') as fh:
            wr=csv.DictWriter(fh,fieldnames=list(rows[0].keys()));wr.writeheader()
            for r in rows:wr.writerow({k:('n/a' if v is None or (isinstance(v,float) and not np.isfinite(v)) else (round(v,6) if isinstance(v,float) else v)) for k,v in clean(r).items()} if True else r)
    res['files']={n:sha(OUT/n) for n in ['kippkurve.csv','gitter_or.csv','anker.csv','profile.csv']}
    dump(OUT/'results.json',res);figure(point,rows_k,rows_p);print('AGGREGATED')

def figure(point,rows_k,rows_p):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,2,figsize=(12,9))
    for c,fold in enumerate(['train','validation']):
        for r,t in enumerate([1.0]):
            a=ax[0,c];M=np.full((len(RF),len(RHO)),np.nan)
            for row in rows_k:
                if row['fold']==fold and row['target_or']==t and row['status']=='Kreuzung':
                    M[list(RF).index(row['R_F']),list(RHO).index(row['rho'])]=row['R_A_kipp']
            im=a.imshow(M,origin='lower',aspect='auto',cmap='viridis',vmin=.25,vmax=.6)
            a.set_xticks(range(len(RHO)));a.set_xticklabels([f'{v:g}' for v in RHO],rotation=60,fontsize=7);a.set_yticks(range(len(RF)));a.set_yticklabels([f'{v:.2f}' for v in RF],fontsize=7)
            a.set_xlabel('error correlation rho');a.set_ylabel('marker reliability R_F');a.set_title(f'{fold}: R_A at which corrected OR reaches 1.0 (blank: no crossing)',fontsize=9)
            for i in range(len(RF)):
                for j in range(len(RHO)):
                    if np.isfinite(M[i,j]):a.text(j,i,f'{M[i,j]:.2f}',ha='center',va='center',fontsize=6,color='w')
            fig.colorbar(im,ax=a,fraction=.04)
        a=ax[1,c]
        for nm in PROFILES:
            pr=[x for x in rows_p if x['fold']==fold and x['profil']==nm and x['zulaessig']]
            a.plot([x['R_A'] for x in pr],[x['or'] for x in pr],label=nm)
        a.axhline(1.25,ls='--',c='gray',lw=.8);a.axhline(1.0,ls=':',c='gray',lw=.8);a.axvspan(.59,.70,color='orange',alpha=.15)
        a.set_yscale('log');a.set_ylim(.3,30);a.set_xlim(.25,1);a.set_xlabel('angle reliability R_A (outcome set)');a.set_ylabel('corrected marker OR per development SD');a.legend(fontsize=7);a.set_title(f'{fold}: profiles (shaded: observed R_A range)',fontsize=9)
    fig.tight_layout();fig.savefig(OUT/'kippkurve.png',dpi=150);plt.close(fig)

if __name__=='__main__':
    try:
        dump(OUT/('status_'+PHASE+'.json'),{'phase':PHASE,'state':'running','pid':os.getpid(),'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
        globals()[PHASE]()
        dump(OUT/('status_'+PHASE+'.json'),{'phase':PHASE,'state':'complete','pid':os.getpid(),'peak_rss':PEAK,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())});print('COMPLETE',PHASE)
    except BaseException as ex:
        dump(OUT/('status_'+PHASE+'.json'),{'phase':PHASE,'state':'failed','pid':os.getpid(),'reason':str(ex)[:300]});raise
