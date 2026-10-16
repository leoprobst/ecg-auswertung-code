"""Part 1: differential measurement error (LVEF, amplitude quintiles) and kappa scenario analysis.

Phases: point | boot1 | boot2 | aggregate. Contract: the pre-registered protocol of this stage (section 1).
"""
import os,sys,time,json
from pathlib import Path
os.environ.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from werkzeuge.b_test import aw211_common as C
PHASE=sys.argv[1]
C.OUT.mkdir(exist_ok=True)
sys.stdout=(C.OUT/('t1_'+PHASE+'_stdout.log')).open('a',encoding='utf8',buffering=1)
sys.stderr=(C.OUT/('t1_'+PHASE+'_stderr.log')).open('a',encoding='utf8',buffering=1)
C.guard();C.start_watch(1.0)
import numpy as np,pandas as pd
from scipy.special import expit
from werkzeuge.b_test.daten import classify_lvef
from werkzeuge.b_test.statistik import raw_design,fixed_design
from werkzeuge.nachrechnung.core import logistic_fit
from werkzeuge.b_test.aw157_models import icc
A,F,AMP=C.A,C.F,C.AMP
SEED=2112026;NBOOT=1000;MINVALID=950;MINPAIRS=30
SCALES=['A','F','Fl']
KAPPA=np.round(np.arange(100,2001)/100,2)
TARGETS=[1.25,1.0]
STATUS={1:'Kreuzung',2:'Grenze vor Kreuzung',3:'unzulaessig bei kappa=1',4:'bei kappa=1 unter Ziel',5:'keine Kreuzung bis 20'}

# ---------------------------------------------------------------- Data
def load_all():
    f,finfo,scale=C.outcome_frame();q,qinfo=C.pair_frame()
    sdA,sdF=scale['sd'][-2],scale['sd'][-1]
    ref=f.loc[f.eligible_reference.eq(True)&np.isfinite(f.preferred_lvef.to_numpy(float))]
    ref=ref.drop_duplicates('subject_id')
    efmap=pd.Series(classify_lvef(ref.preferred_lvef.to_numpy(float)),index=ref.subject_id.to_numpy())
    idxmap=pd.Series(ref.study_id.to_numpy(),index=ref.subject_id.to_numpy())
    P={}
    P['sid']=q.subject_id.to_numpy()
    P['split']=q.split.to_numpy()
    P['x']={'A':(q[A+'_first'].to_numpy(float),q[A+'_second'].to_numpy(float)),
            'F':(q[F+'_first'].to_numpy(float)/sdF,q[F+'_second'].to_numpy(float)/sdF),
            'Fl':(np.log1p(q[F+'_first'].to_numpy(float)),np.log1p(q[F+'_second'].to_numpy(float)))}
    P['amp']=(q[AMP+'_first'].to_numpy(float)+q[AMP+'_second'].to_numpy(float))/2
    P['cuts']=np.quantile(P['amp'],[.2,.4,.6,.8])
    P['q']=np.searchsorted(P['cuts'],P['amp'],side='right')+1
    P['ef']=q.subject_id.map(efmap).to_numpy(float)
    P['SB']=np.isfinite(P['ef'])
    P['SA']=P['SB']&(q.first_study_id.to_numpy()==q.subject_id.map(idxmap).to_numpy())
    # Angle and Fréchet in development SD for the scenario analysis (delta, as in the preceding analysis stage)
    P['delta_dev']=np.column_stack([P['x']['A'][1]-P['x']['A'][0],(P['x']['F'][1]-P['x']['F'][0])])
    P['delta_dev'][:,0]=P['delta_dev'][:,0]/sdA
    P['info']={'pairs':qinfo,'population':finfo,'sdA':sdA,'sdF':sdF}
    # Outcome case set (eligible_B3), folds
    g=f.loc[f.eligible_B3.eq(True)].reset_index(drop=True)
    raw,names=raw_design(g,F);z=fixed_design(raw,scale);y=classify_lvef(g.preferred_lvef.to_numpy(float))
    folds={}
    for fold in ['train','validation']:
        ix=g.split.eq(fold).to_numpy();folds[fold]={'z':z[ix],'y':y[ix],'ids':g.subject_id.to_numpy()[ix]}
    return P,folds,scale

def wsd(d,w):
    m=np.average(d,weights=w);return float(np.sqrt(np.sum(w*(d-m)**2)/(w.sum()-1)))
def qp(y,X,w):
    """Quasi-Poisson, log link, Newton on the concave likelihood; weights as frequencies."""
    b=np.zeros(X.shape[1]);b[0]=np.log(max(np.average(y,weights=w),1e-12))
    for _ in range(100):
        mu=np.exp(np.clip(X@b,-40,40));g=X.T@(w*(y-mu));H=X.T@((w*mu)[:,None]*X)
        st=np.linalg.solve(H,g);b=b+st
        if np.max(np.abs(st))<1e-10:return b
    raise RuntimeError('QP not converged')

def group_stats(P,mask,w,out,prefix):
    n=w[mask].sum()
    if n<MINPAIRS:raise ValueError('too few pairs '+prefix)
    out[prefix+'_n']=float(n)
    dd={}
    for s in SCALES:
        x1,x2=P['x'][s][0][mask],P['x'][s][1][mask];d=x2-x1;ww=w[mask];dd[s]=d
        out[f'{prefix}_{s}_sd']=wsd(d,ww);out[f'{prefix}_{s}_mean']=float(np.average(d,weights=ww));out[f'{prefix}_{s}_icc']=icc(x1,x2,ww)
    ww=w[mask];ca=dd['A']-np.average(dd['A'],weights=ww);cf=dd['F']-np.average(dd['F'],weights=ww)
    out[prefix+'_rho']=float(np.sum(ww*ca*cf)/np.sqrt(np.sum(ww*ca**2)*np.sum(ww*cf**2)))

def compute_all(P,w):
    out={}
    for st in ['SA','SB']:
        m=P[st]&(w>0);y=P['ef']
        for gname,gv in [('low',1.),('pres',0.)]:group_stats(P,m&(y==gv),w,out,f'{st}_{gname}')
        for s in SCALES:
            out[f'{st}_{s}_sdratio']=out[f'{st}_low_{s}_sd']/out[f'{st}_pres_{s}_sd']
            out[f'{st}_{s}_iccdiff']=out[f'{st}_low_{s}_icc']-out[f'{st}_pres_{s}_icc']
            d=P['x'][s][1][m]-P['x'][s][0][m];yy=d**2
            X0=np.column_stack([np.ones(m.sum()),y[m]]);q=P['q'][m]
            X1=np.column_stack([X0]+[(q==k).astype(float) for k in (2,3,4,5)])
            b0=qp(yy,X0,w[m]);b1=qp(yy,X1,w[m])
            out[f'{st}_{s}_qp_crude_sdratio']=float(np.exp(b0[1]/2));out[f'{st}_{s}_qp_adj_sdratio']=float(np.exp(b1[1]/2))
            out[f'{st}_{s}_adj_over_crude']=out[f'{st}_{s}_qp_adj_sdratio']/out[f'{st}_{s}_qp_crude_sdratio']
        out[f'{st}_meanlogamp_low']=float(np.average(np.log(P['amp'][m&(y==1)]),weights=w[m&(y==1)]))
        out[f'{st}_meanlogamp_pres']=float(np.average(np.log(P['amp'][m&(y==0)]),weights=w[m&(y==0)]))
    allm=w>0
    for k in range(1,6):group_stats(P,allm&(P['q']==k),w,out,f'Q{k}')
    for s in SCALES:
        d=P['x'][s][1][allm]-P['x'][s][0][allm];yy=d**2;q=P['q'][allm]
        Xs=np.column_stack([np.ones(allm.sum())]+[(q==k).astype(float) for k in (2,3,4,5)]);bs=qp(yy,Xs,w[allm])
        Xt=np.column_stack([np.ones(allm.sum()),q-3.]);bt=qp(yy,Xt,w[allm])
        out[f'amp_{s}_q1_over_q5_sdratio']=float(np.exp(-bs[4]/2));out[f'amp_{s}_trend_sdratio_step']=float(np.exp(bt[1]/2))
    return out

# ---------------------------------------------------------------- Scenario analysis kappa
def kappa_scan(c,Ee,Ec,pi,target):
    S=c['S'];u=c['u'];k=KAPPA
    E11=(1-pi)*Ee[0,0]+pi*k*Ec[0,0];E22=(1-pi)*Ee[1,1]+pi*k*Ec[1,1];E12=(1-pi)*Ee[0,1]+pi*k*Ec[0,1]
    T11=S[0,0]-E11;T22=S[1,1]-E22;T12=S[0,1]-E12
    det=T11*T22-T12**2;lam=(T11+T22-np.sqrt((T11-T22)**2+4*T12**2))/2;ok=lam>1e-10
    with np.errstate(divide='ignore',invalid='ignore'):cF=(-T12*u[0]+T11*u[1])/det
    lt=np.log(target);bad=(~ok)|(cF<lt)
    if not bad.any():return 5,np.nan
    i=int(np.argmax(bad))
    if i==0:return (3 if not ok[0] else 4),1.0
    if not ok[i]:return 2,float(k[i-1])
    y0,y1=cF[i-1],cF[i];x0,x1=k[i-1],k[i]
    return 1,float(x0+(lt-y0)*(x1-x0)/(y1-y0))

def evaluate_ef(c,Ec,Ee,Ep,pi):
    E1=(1-pi)*Ee+pi*Ec;lf,la,lam,pc=corrected_pc(c,E1);rel=C.anchor_rel(c,E1)
    lp,lap,lamp,pcp=corrected_pc(c,Ep);relp=C.anchor_rel(c,Ep)
    out={'logor_mix':lf,'angle_logor_mix':la,'lam_mix':lam,'pcorr_mix':pc,'R_A_mix':rel['R_A'],'R_F_mix':rel['R_F'],'rho_mix':rel['rho'],
         'logor_pool':lp,'angle_logor_pool':lap,'lam_pool':lamp,'pcorr_pool':pcp,'R_A_pool':relp['R_A'],'R_F_pool':relp['R_F'],'rho_pool':relp['rho'],
         'ratio_A':Ec[0,0]/Ee[0,0],'ratio_F':Ec[1,1]/Ee[1,1],'pi':pi,'naive_logor':c['b'][-1],'naive_angle_logor':c['b'][-2]}
    for t in TARGETS:
        s,v=kappa_scan(c,Ee,Ec,pi,t);out[f'kstatus_{t}']=s;out[f'kappa_{t}']=v
    return out
def corrected_pc(c,E):
    return C.corrected_E(c,E)

def fold_inputs(P,folds,fold):
    d=folds[fold];sb=P['SB']
    union=np.union1d(d['ids'],P['sid'][sb]);oi=np.searchsorted(union,d['ids']);pi_=np.searchsorted(union,P['sid'][sb])
    return union,oi,pi_

def ef_eval_weights(P,folds,fold,w_o,w_p,base=None):
    d=folds[fold];z,y=d['z'],d['y'];sb=P['SB'];ef=P['ef'][sb];dd=P['delta_dev'][sb]
    b=logistic_fit(z,y,w_o,initial=base) if base is not None else logistic_fit(z,y,w_o)
    c=C.components(z,y,w_o,b,logistic_fit)
    if w_p[ef==1].sum()<MINPAIRS or w_p[ef==0].sum()<MINPAIRS:raise ValueError('too few pairs')
    Ec=C.pair_error(dd[ef==1],w_p[ef==1]);Ee=C.pair_error(dd[ef==0],w_p[ef==0]);Ep=C.pair_error(dd,w_p)
    pi=float(np.sum(w_o*y)/np.sum(w_o))
    return evaluate_ef(c,Ec,Ee,Ep,pi),c

# ---------------------------------------------------------------- Phases
def counts(P,folds):
    def sup(n):return int(n) if n>=10 else '<10'
    out={'pairs_all':sup(len(P['sid'])),'persons_all':sup(len(np.unique(P['sid'])))}
    for st in ['SA','SB']:
        m=P[st];out[st]={'pairs':sup(m.sum()),'persons':sup(len(np.unique(P['sid'][m])))}
        for g,gv in [('low',1.),('pres',0.)]:
            mm=m&(P['ef']==gv);out[st][g]={'pairs':sup(mm.sum()),'persons':sup(len(np.unique(P['sid'][mm])))}
    out['quintile_cuts_uV']=P['cuts'];out['quintile_pairs']={str(k):sup((P['q']==k).sum()) for k in range(1,6)}
    out['folds']={k:{'n':sup(len(v['y'])),'cases':sup(v['y'].sum())} for k,v in folds.items()}
    return out

def point():
    C.check_stamp();P,folds,scale=load_all()
    res={'counts':counts(P,folds),'info':P['info'],'script_sha256':C.sha(__file__),'common_sha256':C.sha(C.ROOT/'werkzeuge/b_test/aw211_common.py'),'protokoll':C.check_stamp()}
    # Gate: complete-case reference as in the preceding analysis stage
    gate={}
    pub={'train':2.480,'validation':2.392}
    for fold in folds:
        z,y=folds[fold]['z'],folds[fold]['y'];b=logistic_fit(z,y,np.ones(len(y)));gate[fold]={'naive_or':float(np.exp(b[-1])),'published':pub[fold],'ok':abs(np.exp(b[-1])-pub[fold])<=0.002}
    res['gate']=gate
    if not all(v['ok'] for v in gate.values()):C.dump(C.OUT/'t1_point.json',res);raise SystemExit('STOP: reproduction gate not passed')
    w=np.ones(len(P['sid']))
    res['metrics']=compute_all(P,w)
    # Cells quintile by LVEF (point estimate only)
    cells={}
    for st in ['SA','SB']:
        for gname,gv in [('low',1.),('pres',0.)]:
            for k in range(1,6):
                m=P[st]&(P['ef']==gv)&(P['q']==k);n=int(m.sum())
                if n<10:cells[f'{st}_{gname}_Q{k}']={'n':'<10'};continue
                cells[f'{st}_{gname}_Q{k}']={'n':n,**{f'sd_{s}':wsd(P['x'][s][1][m]-P['x'][s][0][m],np.ones(n)) for s in SCALES}}
    res['cells']=cells
    fp={}
    for fold in folds:
        u,oi,pi_=fold_inputs(P,folds,fold);w_o=np.ones(len(oi));w_p=np.ones(len(pi_))
        ev,c=ef_eval_weights(P,folds,fold,w_o,w_p);fp[fold]={k:v for k,v in ev.items()}
    res['kappa_point']=fp
    C.dump(C.OUT/'t1_point.json',res);print('POINT done',json.dumps(C.clean(gate)),flush=True)

def boot1():
    C.check_stamp();P,folds,scale=load_all();pt=C.load(C.OUT/'t1_point.json');names=list(pt['metrics'].keys())
    path=C.OUT/'t1_boot1.npz'
    if path.exists():
        z=np.load(path);M=z['M'];done=int(z['done'])
    else:M=np.full((NBOOT,len(names)),np.nan);done=0
    persons=np.unique(P['sid']);pidx=np.searchsorted(persons,P['sid']);fails={}
    for r in range(done,NBOOT):
        if r%50==0:
            C.guard();print('BOOT1',r,flush=True)
            if r>done:np.savez_compressed(path,M=M,done=r,names=np.array(names))
        rng=np.random.default_rng(np.random.SeedSequence([SEED,1,r]));mult=np.bincount(rng.integers(0,len(persons),len(persons)),minlength=len(persons)).astype(float)
        try:
            o=compute_all(P,mult[pidx]);M[r]=[o[k] for k in names]
        except Exception as ex:fails[str(ex)[:60]]=fails.get(str(ex)[:60],0)+1
    np.savez_compressed(path,M=M,done=NBOOT,names=np.array(names))
    C.dump(C.OUT/'t1_boot1_meta.json',{'attempted':NBOOT,'valid':int(np.isfinite(M).all(1).sum()),'failures':fails,'seed':[SEED,1],'peak_rss':C.PEAK[0]});print('BOOT1 done',flush=True)

def boot2():
    C.check_stamp();P,folds,scale=load_all()
    for fi,fold in enumerate(['train','validation']):
        path=C.OUT/f't1_boot2_{fold}.npz'
        u,oi,pi_=fold_inputs(P,folds,fold);base=logistic_fit(folds[fold]['z'],folds[fold]['y'],np.ones(len(oi)))
        pt=C.load(C.OUT/'t1_point.json')['kappa_point'][fold];names=list(pt.keys())
        if path.exists():
            z=np.load(path);M=z['M'];done=int(z['done'])
        else:M=np.full((NBOOT,len(names)),np.nan);done=0
        fails={}
        for r in range(done,NBOOT):
            if r%50==0:
                C.guard();print('BOOT2',fold,r,flush=True)
                if r>done:np.savez_compressed(path,M=M,done=r,names=np.array(names))
            rng=np.random.default_rng(np.random.SeedSequence([SEED,2,fi,r]));mult=np.bincount(rng.integers(0,len(u),len(u)),minlength=len(u)).astype(float)
            try:
                ev,_=ef_eval_weights(P,folds,fold,mult[oi],mult[pi_],base);M[r]=[ev[k] for k in names]
            except Exception as ex:fails[str(ex)[:60]]=fails.get(str(ex)[:60],0)+1
        np.savez_compressed(path,M=M,done=NBOOT,names=np.array(names))
        C.dump(C.OUT/f't1_boot2_{fold}_meta.json',{'fold':fold,'attempted':NBOOT,'valid':int(np.isfinite(M[:,names.index("naive_logor")]).sum()),'failures':fails,'seed':[SEED,2,fi],'union_persons':int(len(u)),'peak_rss':C.PEAK[0]})
        print('BOOT2 done',fold,flush=True)

def ci(v):
    v=np.asarray(v,float);v=v[np.isfinite(v)]
    return [float(np.quantile(v,.025)),float(np.quantile(v,.975))] if len(v)>=MINVALID else None

def aggregate():
    C.check_stamp();pt=C.load(C.OUT/'t1_point.json');z=np.load(C.OUT/'t1_boot1.npz');names=list(z['names']);M=z['M']
    valid=np.isfinite(M).all(1);res={'boot1':{'attempted':NBOOT,'valid':int(valid.sum())},'metrics':{}}
    for i,n in enumerate(names):
        res['metrics'][n]={'point':pt['metrics'][n],'ci':ci(M[valid,i]) if valid.sum()>=MINVALID else None}
    # kappa_plaus
    kp={}
    for s in ['A','F']:
        r=res['metrics'][f'SB_{s}_sdratio'];pointv=r['point']**2;up=(r['ci'][1]**2) if r['ci'] else np.nan
        kp[s]={'var_ratio_point':pointv,'var_ratio_upper':up,'kappa_plaus':max(1.0,up/pointv) if np.isfinite(up) else None}
    kplaus=max([v['kappa_plaus'] for v in kp.values() if v['kappa_plaus'] is not None],default=None)
    res['kappa_plaus']={'by_scale':kp,'kappa_plaus':kplaus}
    res['boot2']={}
    for fold in ['train','validation']:
        zz=np.load(C.OUT/f't1_boot2_{fold}.npz');nm=list(zz['names']);MM=zz['M'];meta=C.load(C.OUT/f't1_boot2_{fold}_meta.json');fp=pt['kappa_point'][fold]
        d={'meta':meta,'point':{k:(float(np.exp(v)) if (k.startswith('logor') or k=='naive_logor') and v is not None else v) for k,v in fp.items()},'ci':{}}
        for k in ['logor_mix','logor_pool','R_A_mix','R_F_mix','rho_mix','pcorr_mix','R_A_pool','R_F_pool','rho_pool','ratio_A','ratio_F','pi','naive_logor']:
            col=MM[:,nm.index(k)];d['ci'][k]=ci(np.exp(col) if k.startswith('logor') or k=='naive_logor' else col)
        d['admissible_mix']=int(np.isfinite(MM[:,nm.index('logor_mix')]).sum());d['admissible_pool']=int(np.isfinite(MM[:,nm.index('logor_pool')]).sum())
        for t in TARGETS:
            st=MM[:,nm.index(f'kstatus_{t}')];kv=MM[:,nm.index(f'kappa_{t}')]
            d[f'kappa_{t}']={'status_counts':{STATUS[int(s)]:int((st==s).sum()) for s in STATUS if (st==s).sum()},'ci_kreuzung':ci(kv[st==1]) if (st==1).sum()>=MINVALID else None,'n_kreuzung':int((st==1).sum()),
                             'point_status':STATUS[int(fp[f'kstatus_{t}'])],'point_kappa':fp[f'kappa_{t}'],
                             'boundary_ci':ci(kv[st==2]) if (st==2).sum()>=MINVALID else None}
        res['boot2'][fold]=d
    # Interpretation rule of the protocol
    ev=[]
    for fold in ['train','validation']:
        for t in TARGETS:
            ps=int(pt['kappa_point'][fold][f'kstatus_{t}']);pk=pt['kappa_point'][fold][f'kappa_{t}']
            if ps in (3,4):ev.append((fold,t,'relevant: '+STATUS[ps],None))
            elif ps==5:ev.append((fold,t,'nicht erreicht bis 20',np.inf))
            else:ev.append((fold,t,STATUS[ps],pk))
    rel=False
    for fold,t,lab,val in ev:
        if val is None:rel=True
        elif kplaus is not None and val<=kplaus:rel=True
    res['U4']={'events':[{'fold':a,'target':b,'label':c,'kappa':d} for a,b,c,d in ev],'kappa_plaus':kplaus,'decision':'relevant (nicht auszuschliessen)' if rel else 'nicht relevant'}
    C.dump(C.OUT/'t1_results.json',res);print('AGGREGATED',flush=True)

if __name__=='__main__':
    try:
        C.dump(C.OUT/('t1_status_'+PHASE+'.json'),{'phase':PHASE,'state':'running','pid':os.getpid(),'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
        globals()[PHASE]()
        C.dump(C.OUT/('t1_status_'+PHASE+'.json'),{'phase':PHASE,'state':'complete','pid':os.getpid(),'peak_rss':C.PEAK[0],'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())});print('COMPLETE',PHASE)
    except BaseException as ex:
        C.dump(C.OUT/('t1_status_'+PHASE+'.json'),{'phase':PHASE,'state':'failed','pid':os.getpid(),'reason':str(ex)[:300]});raise
