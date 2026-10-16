"""AW171: reine Rechenkerne, keine Daten-/Ledgerzugriffe, feste Rezepte."""
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import t
from werkzeuge.b_test.statistik import separation, fixed_design, raw_design
from werkzeuge.nachrechnung.core import auc, age_basis
from werkzeuge.b_test.aw157_models import rc, multirc

class Stop(RuntimeError):
    pass

def require(ok, reason):
    if not ok:
        raise Stop(reason)

def gates(x, y, w=None,ess=False):
    w=np.ones(len(y)) if w is None else np.asarray(w,float)
    require(w.shape==y.shape and np.isfinite(w).all() and (w>=0).all(), 'Gewichte')
    a=w>0; xx=x[a]; yy=y[a]; ww=w[a]
    require(np.isfinite(xx).all() and set(np.unique(yy))=={0,1}, 'Klasse/finite')
    require(min(np.count_nonzero(yy==c) for c in [0,1])>=100, 'kleine_Strata')
    if ess:require(min(ww[yy==c].sum()**2/(ww[yy==c]@ww[yy==c]) for c in [0,1])>=100, 'ESS')
    require(len(yy)>xx.shape[1] and np.linalg.matrix_rank(xx)==xx.shape[1], 'Rang')
    require(separation(xx,yy)=='keine', 'Separation')
    return xx,yy,ww

def solver(x,y,w=None,initial=None,diagnostics=False):
    """W10: identische Ridge-Zielfunktion, strengere gemeinsame Score-Gates."""
    x,y,w=gates(x,np.asarray(y,float),w)
    b=np.zeros(x.shape[1]) if initial is None else np.asarray(initial,float).copy()
    pen=np.r_[0.,np.full(len(b)-1,1e-6)]; n=len(y)
    loss=lambda v:float(w@(np.logaddexp(0,x@v)-y*(x@v))+.5*(pen*v)@v)
    q0=None; trail=[]
    for k in range(201):
        p=expit(x@b); g=x.T@(w*(p-y))+pen*b
        q=float(np.max(np.abs(g))/n); qw=float(np.max(np.abs(g))/w.sum())
        if q0 is None: q0=q
        h=x.T@((w*p*(1-p))[:,None]*x)+np.diag(pen)
        trail.append({'iteration':k,'score_n':q,'score_weight':qw,'initial_norm':q0,'hessian_condition':float(np.linalg.cond(h))})
        if q<=1e-7*max(1,q0) and q<1e-7 and qw<1e-7:
            require(np.isfinite(b).all(),'numerisch')
            return (b,trail) if diagnostics else b
        require(k<200,'Konvergenz_200')
        d=np.linalg.solve(h,g); step=1.; old=loss(b)
        for halves in range(31):
            if loss(b-step*d)<=old: break
            step*=.5
        else: raise Stop('Line_Search_30')
        trail[-1]['step']=step; trail[-1]['halvings']=halves
        b-=step*d

def association(x,y,w=None,ess=False):
    xx,yy,ww=gates(x,np.asarray(y,float),w,ess)
    b=solver(xx,yy,ww); b0=solver(xx[:,:-1],yy,ww)
    p=expit(xx@b); n,k=xx.shape
    h=xx.T@((ww*p*(1-p))[:,None]*xx)+np.diag(np.r_[0.,np.full(k-1,1e-6)])
    hi=np.linalg.inv(h); u=xx*(ww*(yy-p))[:,None]
    # Ein unveränderlicher Index pro Person. CR1 N=G, Nullmultiplizitäten entfernen.
    cov=hi@(u.T@u)@hi*n/(n-1)*(n-1)/(n-k)
    require(np.isfinite(cov).all() and cov[-1,-1]>0,'CR1')
    se=np.sqrt(cov[-1,-1]); d=auc(yy,xx@b,ww)-auc(yy,xx[:,:-1]@b0,ww)
    return {'beta':float(b[-1]),'or':float(np.exp(b[-1])), 'variance':float(cov[-1,-1]),
            'wald_low':float(np.exp(b[-1]-1.95996398454*se)), 'wald_high':float(np.exp(b[-1]+1.95996398454*se)),
            'auc_base':auc(yy,xx[:,:-1]@b0,ww),'auc_marker':auc(yy,xx@b,ww),'delta_auc':d,
            'n':n,'cases':int((yy==1).sum()),'controls':int((yy==0).sum()),'coefficients':b.tolist()}

def threshold90(scores,y):
    v=np.sort(np.asarray(scores)[np.asarray(y)==0])
    require(len(v)>0 and np.isfinite(v).all(),'Schwelle_Kontrollen')
    return float(v[int(np.ceil(.9*len(v)))-1])

def quartiles(values):
    v=np.asarray(values,float); require(np.isfinite(v).all() and len(v)>0,'Quartil_Input')
    q=np.quantile(v,[.25,.5,.75],method='linear')
    require((np.diff(q)>0).all(),'Quartil_Ties')
    g=np.searchsorted(q,v,side='left')
    require(all((g==k).any() for k in range(4)),'Quartil_leer')
    return {'bounds':q.tolist(),'medians':[float(np.median(v[g==k])) for k in range(4)],
            'counts':[int((g==k).sum()) for k in range(4)],'method':'linear'}

def prediction(x,y,models,w=None):
    w=np.ones(len(y)) if w is None else w; result={}; scores={}
    for model,k in [('B2',7),('B3',8),('B3_marker',9)]:
        m=models[model]; lp=x[:,:k]@np.asarray(m['beta']); scores[model]=lp
        p=expit(lp); pos=lp>m['threshold']; n0=w[y==0].sum(); n1=w[y==1].sum()
        require(n0>0 and n1>0,'Klasse')
        result[model+'_auc']=auc(y,lp,w); result[model+'_brier']=float(np.average((y-p)**2,weights=w))
        result[model+'_sensitivity']=float(w[(y==1)&pos].sum()/n1)
        result[model+'_specificity']=float(w[(y==0)&~pos].sum()/n0)
        # Offsetintercept Newton, Steigung bleibt 1.
        offset=0.
        for _ in range(200):
            pp=expit(lp+offset); den=float(w@(pp*(1-pp)))
            require(den>0,'Kalibrierungsseparation')
            grad=float(w@(pp-y))
            if abs(grad)/w.sum()<1e-7: break
            offset-=grad/den
        else: raise Stop('Kalibrierung_Konvergenz')
        cal=solver(np.column_stack([np.ones(len(y)),lp]),y,w)
        result[model+'_offset_intercept']=offset
        result[model+'_calibration_intercept']=float(cal[0]); result[model+'_calibration_slope']=float(cal[1])
    result['delta_marker']=result['B3_marker_auc']-result['B3_auc']
    result['delta_angle']=result['B3_auc']-result['B2_auc']
    return result

def rcs(x,knots):
    x=np.asarray(x,float); t=np.asarray(knots,float); h=[x]
    require(len(t)==4 and (np.diff(t)>0).all(),'Knoten')
    for tj in t[:-2]:
        h.append((np.maximum(x-tj,0)**3-(t[-1]-tj)/(t[-1]-t[-2])*np.maximum(x-t[-2],0)**3+(t[-2]-tj)/(t[-1]-t[-2])*np.maximum(x-t[-1],0)**3)/(t[-1]-t[0])**2)
    return np.column_stack(h)

def corrected(x,y,w,scale,error,variant,seed,draws,spline):
    require(draws>0 and draws%2==0,'antithetisch')
    b=solver(x,y,w); sd=np.asarray(scale['sd'][-2:]); mean=scale['mean'][-2]
    E=np.asarray(error['covariance'])/np.outer(sd,sd)
    require(np.linalg.eigvalsh(E).min()>=0,'Fehlerkovarianz_PSD')
    if variant=='rc':
        r,_=rc(x,w,error['angle_variance']/sd[0]**2,b)
        return r
    if variant=='joint_rc':
        r,_=multirc(x,w,E,b); require('reason' not in r,'latente_Kovarianz'); return r
    theta=x[:,-2]*sd[0]+mean
    def design(angle,marker):
        dd=x.copy(); dd[:,-1]=marker
        if variant=='logit': dd[:,-2]=logit(np.clip(angle,.5,179.5)/180)
        elif variant=='spline':
            basis=(rcs(angle,spline['knots'])-spline['mean'])/spline['sd']
            dd=np.column_stack([x[:,:-2],basis,marker])
        else: dd[:,-2]=(angle-mean)/sd[0]
        return dd
    base=design(theta,x[:,-1]); b=solver(base,y,w); curve=[b]; outside=[]; scatter=[]
    rng=np.random.Generator(np.random.PCG64(seed))
    normal=rng.normal(size=(draws//2,len(y),2))
    if variant=='logit':
        noise=normal[:,:,0]*np.sqrt(error['logit_variance']); em=np.zeros_like(noise)
    elif variant=='joint':
        e=normal@np.linalg.cholesky(np.asarray(error['covariance'])).T; noise=e[:,:,0]; em=e[:,:,1]/sd[1]
    else:
        noise=normal[:,:,0]*np.sqrt(error['angle_variance']); em=np.zeros_like(noise)
    for lam in [.5,1.,1.5,2.]:
        fits=[]; clips=[]
        for j in range(draws//2):
            for sign in [-1,1]:
                if variant=='logit':
                    dd=base.copy(); dd[:,-2]+=sign*np.sqrt(lam)*noise[j]; clips.append(0.)
                else:
                    a=theta+sign*np.sqrt(lam)*noise[j]
                    clips.append(float(np.average((a<0)|(a>180),weights=w)))
                    if variant=='clipped': a=np.clip(a,0,180)
                    dd=design(a,x[:,-1]+sign*np.sqrt(lam)*em[j])
                fits.append(solver(dd,y,w,initial=b))
        fits=np.asarray(fits); curve.append(fits.mean(0)); outside.append(float(np.mean(clips))); scatter.append(float(fits[:,-1].std(ddof=1)))
    lam=np.array([0,.5,1,1.5,2]); curve=np.asarray(curve)
    result={'uncorrected_beta':float(b[-1]),'outside':outside,'mc_scatter':scatter,'curve':curve[:,-1].tolist()}
    for order,name in [(1,'linear'),(2,'quadratic')]:
        design_lam=np.column_stack([lam**j for j in range(order+1)])
        extrap=np.array([(-1)**j for j in range(order+1)])@np.linalg.lstsq(design_lam,curve,rcond=None)[0]
        require(np.isfinite(extrap).all(),'Extrapolation')
        result[name+'_beta']=float(extrap[-1]); result[name+'_delta_beta']=float(extrap[-1]-b[-1])
        result[name+'_fraction']=float(extrap[-1]/b[-1]) if abs(b[-1])>=.01 else None
    return result

def impute(frame,scale,endpoint,replicate=0,m=0,weights=None):
    """20 Zyklen MICE: PMM-5/posterior draw, binäres Geschlecht; keine Ziel-/Markerimputation."""
    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([1692026,18,endpoint,replicate,m])))
    cols=['age','sex_numeric','heart_rate','qrs_duration_ms','qtc_fridericia_ms']
    aux=frame[['y','spatial_qrs_t_angle_deg','marker']].to_numpy(float)
    require(np.isfinite(aux).all(),'MI_Marker_Winkel_Klasse')
    raw=frame[cols].to_numpy(float); missing=~np.isfinite(raw); a=raw.copy(); trace=[]
    weights=np.ones(len(frame)) if weights is None else np.asarray(weights,float)
    require(weights.shape==(len(frame),) and np.isfinite(weights).all() and (weights>=0).all() and np.equal(weights,np.floor(weights)).all(),'MI_Frequenzgewichte')
    for j in range(5): a[missing[:,j],j]=scale['start'][cols[j]]
    for cycle in range(20):
        for j in range(5):
            obs=(~missing[:,j])&(weights>0); mis=missing[:,j]
            if not mis.any(): continue
            require(obs.sum()>=5,'MI_Donoren')
            cov=np.column_stack([aux,np.delete(a,j,axis=1)])
            ow=weights[obs]; mean=np.average(cov[obs],axis=0,weights=ow)
            sd=np.sqrt(np.average((cov[obs]-mean)**2,axis=0,weights=ow));keep=sd>0
            xx=np.column_stack([np.ones(len(a)),(cov[:,keep]-mean[keep])/sd[keep]])
            xo=xx[obs]; target=raw[obs,j]
            require(np.linalg.matrix_rank(xo)==xo.shape[1] and len(target)>xo.shape[1],'MI_Rang')
            if j==1:
                beta=solver(xo,target,ow); pp=expit(xo@beta)
                vc=np.linalg.inv(xo.T@((ow*pp*(1-pp))[:,None]*xo))
                beta=rng.multivariate_normal(beta,vc); a[mis,j]=rng.binomial(1,expit(xx[mis]@beta))
            else:
                inv=np.linalg.inv(xo.T@(ow[:,None]*xo)); beta=inv@xo.T@(ow*target)
                resid=target-xo@beta; df=ow.sum()-len(beta)
                sigma2=float((ow*resid)@resid/rng.chisquare(df))
                bdraw=rng.multivariate_normal(beta,sigma2*inv)
                # Frequenzdonoren entsprechend Personenbootstrap, wiederholte
                # Donorpositionen erhalten dieselbe Multiplizität wie Outcome.
                donor=np.repeat(xo@beta,ow.astype(int)); target=np.repeat(target,ow.astype(int))
                pred=xx[mis]@bdraw
                near=np.argpartition(np.abs(pred[:,None]-donor[None,:]),4,axis=1)[:,:5]
                a[mis,j]=target[near[np.arange(len(pred)),rng.integers(0,5,len(pred))]]
        am=np.average(a,axis=0,weights=weights)
        trace.append({'cycle':cycle,'mean':am.tolist(),'sd':np.sqrt(np.average((a-am)**2,axis=0,weights=weights)).tolist()})
    out=frame.copy(); out[cols]=a
    return out,trace

def mi_pool(frame,scale,mi_recipe,endpoint,replicate=0,weights=None,guard=lambda:None):
    results=[]; traces=[]
    for m in range(20):
        guard()
        f,trace=impute(frame,mi_recipe,endpoint,replicate,m,weights)
        x,_=raw_design(f,'marker'); z=fixed_design(x,scale)
        r=association(z,f.y.to_numpy(),weights); results.append(r); traces.append(trace)
    beta=np.array([r['beta'] for r in results]); within=float(np.mean([r['variance'] for r in results])); between=float(beta.var(ddof=1))
    total=within+1.05*between; df=19*(1+within/(1.05*between))**2 if between>0 else np.inf
    critical=float(t.ppf(.975,df)); b=float(beta.mean())
    return {'beta':b,'or':float(np.exp(b)),'rubin_low':float(np.exp(b-critical*np.sqrt(total))),
            'rubin_high':float(np.exp(b+critical*np.sqrt(total))),'within':within,'between':between,'total':total,'df':float(df),
            'delta_auc':float(np.mean([r['delta_auc'] for r in results])),'m':20,'trace':traces}

def verification(train_x,train_v,outcome_x,outcome_y,verification_x,train_w=None,outcome_w=None):
    """W19: ausschließlich train Fit/Kappung; identisches Outcome-Set.

    train_x/verification_x sind bereits mit dem gebundenen B4+Marker-Rezept
    skaliert; train_v umfasst ALLE technischen train-Personen, auch V=0.
    verification_x folgt genau den Outcome-Indizes. Kein Test-Rekalibrieren.
    """
    train_v=np.asarray(train_v,float);outcome_y=np.asarray(outcome_y,float)
    tw=np.ones(len(train_v)) if train_w is None else np.asarray(train_w,float)
    ow=np.ones(len(outcome_y)) if outcome_w is None else np.asarray(outcome_w,float)
    b=solver(train_x,train_v,tw)
    train_p=expit(train_x@b);p=expit(verification_x@b)
    require(np.isfinite(p).all() and (p>0).all(),'Propensity')
    active=(train_v==1)&(tw>0)
    require(active.any() and np.equal(tw,np.floor(tw)).all(),'train_Verifikationsgewichte')
    controls=np.repeat(1/train_p[active],tw[active].astype(int))
    result={'unweighted':association(outcome_x,outcome_y,ow),'train_coefficients':b.tolist()}
    for q,name in [(.95,'p95'),(.99,'p99')]:
        cap=float(np.quantile(controls,q,method='linear')); weights=np.minimum(1/p,cap)
        r=association(outcome_x,outcome_y,ow*weights,ess=True)
        # ESS der Auswertungspersonen unter IPW; Bootstrapmultiplikität und
        # IPW werden gemeinsam als Clustergewicht an association weitergegeben.
        r['cap']=cap;r['capped_fraction']=float(np.average(1/p>cap,weights=ow))
        r['ess_cases']=float((ow[outcome_y==1]*weights[outcome_y==1]).sum()**2/np.sum((ow[outcome_y==1]*weights[outcome_y==1])**2))
        r['ess_controls']=float((ow[outcome_y==0]*weights[outcome_y==0]).sum()**2/np.sum((ow[outcome_y==0]*weights[outcome_y==0])**2))
        result[name]=r
    return result

def verification_bootstrap(train_x,train_v,outcome_x,outcome_y,verification_x,train_positions,
                           train_union_n,outcome_positions,outcome_union_n,planned=1000,guard=lambda:None):
    def compute(ow,r):
        rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([1692026,19,r])))
        tw=np.bincount(rng.integers(0,train_union_n,train_union_n),minlength=train_union_n)[train_positions]
        result=verification(train_x,train_v,outcome_x,outcome_y,verification_x,tw,ow)
        return {name+'_'+metric:result[name][metric] for name in ['unweighted','p95','p99'] for metric in ['beta','or','delta_auc']}
    return bootstrap(compute,outcome_positions,outcome_union_n,planned,950,guard)

def repair_bootstrap(old_point,reproduced_point,union_hash,expected_union_hash,old_rows,failures,refit_failed,planned=1000):
    """Nur historische Fehler nachrechnen; gültige Replikate/Altpunkt unangetastet.

    refit_failed(r, diagnostics) muss historische Design-/Gewichts-/Union-
    Multiplizitäten liefern. Erste IPW-Solverdiagnose ist vor diesem Aufruf zu
    binden; nicht-Solverfehler bleiben Fehler, kein Methodenwechsel.
    """
    identity_gate(old_point,reproduced_point,union_hash,expected_union_hash)
    require(set(old_rows).isdisjoint(failures) and set(old_rows)|set(failures)==set(range(planned)),'W10_Replikatliste')
    solver_failures=[r for r in sorted(failures) if failures[r]['class']=='Konvergenz']
    require(bool(solver_failures),'W10_keine_Solverursache')
    first=solver_failures[0]; diagnostic=refit_failed(first,True)
    require(diagnostic.get('old_failure_is_solver') is True,'W10_keine_Solverursache')
    new={r:row.copy() for r,row in old_rows.items()};remaining={r:v.copy() for r,v in failures.items()}; rescued=[]
    for r in solver_failures:
        try:
            row=refit_failed(r,False)
            require(all(np.isfinite(v) for v in row.values()),'W10_numerisch')
            new[r]=row;del remaining[r];rescued.append(r)
        except (Stop,ValueError,RuntimeError,np.linalg.LinAlgError):pass
    require(all(new[r]==old_rows[r] for r in old_rows),'W10_gueltiges_Replikat_veraendert')
    ci={}
    if len(new)>=950:
        common=set.intersection(*(set(row) for row in new.values()))
        ci={key:np.quantile([row[key] for _,row in sorted(new.items())],[.025,.975],method='linear').tolist() for key in common}
    return {'old_point':old_point,'new_point':old_point,'planned':planned,'old_valid':len(old_rows),'new_valid':len(new),
            'diagnostic':diagnostic,'rescued':rescued,'remaining_failures':remaining,'ci':ci,'rows':new}

def identity_gate(old,reproduced,union_hash,expected_union_hash):
    import struct
    def same(a,b):
        if isinstance(a,dict): return isinstance(b,dict) and set(a)==set(b) and all(same(a[k],b[k]) for k in a)
        if isinstance(a,list): return isinstance(b,list) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
        if isinstance(a,(float,np.floating)): return isinstance(b,(float,np.floating)) and struct.pack('!d',a)==struct.pack('!d',b)
        return type(a)==type(b) and a==b
    require(expected_union_hash is not None and union_hash==expected_union_hash,'W10_Originalunion')
    require(same(old,reproduced),'W10_Punktwertabweichung')

def bootstrap(function,positions,union_n,planned=1000,min_valid=950,guard=lambda:None):
    rows=[]; errors={}
    for r in range(planned):
        if r%25==0: guard()
        rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([1282026,r])))
        mult=np.bincount(rng.integers(0,union_n,union_n),minlength=union_n)
        try:
            v=function(mult[positions].astype(float),r)
            numeric={k:float(x) for k,x in v.items() if isinstance(x,(int,float,np.floating)) and np.isfinite(x)}
            require(bool(numeric),'numerisch'); rows.append(numeric)
        except (Stop,ValueError,RuntimeError,np.linalg.LinAlgError) as ex:
            key=type(ex).__name__+':'+str(ex); errors[key]=errors.get(key,0)+1
    ci={}
    if len(rows)>=min_valid:
        keys=set.intersection(*(set(r) for r in rows))
        ci={k:np.quantile([r[k] for r in rows],[.025,.975],method='linear').tolist() for k in sorted(keys)}
    return {'planned':planned,'attempted':planned,'valid':len(rows),'minimum':min_valid,'ci':ci,'failures':errors,'status':'schaetzbar' if ci else 'KI_nicht_schaetzbar'}
