"""B-Freeze §§5-8: feste Skalierung, deterministische Separation, CR1, Holm."""
import numpy as np
from scipy.optimize import linprog
from scipy.special import expit
from scipy.stats import norm
from scipy.optimize import OptimizeWarning
import warnings
from werkzeuge.nachrechnung.core import age_basis, logistic_fit, auc
from werkzeuge.nachrechnung.aw054 import sandwich

Z95=1.95996398454
SEED=1282026

def _lp(*args,**kwargs):
    # HiGHS has its own thread pool; OMP_NUM_THREADS alone is insufficient.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',category=OptimizeWarning,message='Unrecognized options detected.*')
        return linprog(*args,**kwargs,options={'threads':1,'parallel':False})

def raw_design(f,marker,comparator='qrs_duration_ms',strong=False):
    names=['age_linear','age_natural','sex_numeric','heart_rate','qrs_duration_ms','qtc_fridericia_ms']
    parts=[age_basis(f.age),f[['sex_numeric','heart_rate','qrs_duration_ms','qtc_fridericia_ms']].to_numpy(float)]
    if comparator!='qrs_duration_ms':
        parts.append(f[[comparator]].to_numpy(float));names.append(comparator)
    parts.append(f[['spatial_qrs_t_angle_deg']].to_numpy(float));names.append('spatial_qrs_t_angle_deg')
    if strong:
        extra=['classical_cornell_product','classical_sokolow_lyon','magnitude_area_qrs','spatial_low_voltage_qrs_peak_magnitude']
        axis=np.deg2rad(f.classical_right_axis_deviation.to_numpy(float))
        parts.extend([f[extra].to_numpy(float),np.sin(axis)[:,None],np.cos(axis)[:,None]])
        names+=extra+['axis_sin','axis_cos']
    parts.append(f[[marker]].to_numpy(float));names.append(marker)
    return np.column_stack(parts),names

def scaling(x,names):
    x=np.asarray(x,float)
    if not len(x) or not np.isfinite(x).all():raise ValueError('kein vollständiges Entwicklungsfallset')
    mean=x.mean(0);sd=x.std(0,ddof=0);keep=sd>0
    if not keep[-1]:raise ValueError('Entwicklungsmarker konstant')
    return {'names':names,'mean':mean.tolist(),'sd':sd.tolist(),'keep':keep.tolist(),'ddof':0,'n':len(x)}

def fixed_design(x,params):
    if params.get('status')=='nicht_schaetzbar':raise ValueError(params['reason'])
    x=np.asarray(x,float);mean=np.asarray(params['mean']);sd=np.asarray(params['sd']);keep=np.asarray(params['keep'],bool)
    if x.ndim!=2 or x.shape[1]!=len(mean) or not np.isfinite(x).all():raise ValueError('Design/Parameter ungültig')
    if not keep[-1] or not np.isfinite(sd[keep]).all() or (sd[keep]<=0).any():raise ValueError('feste SD ungültig')
    # Nur vorab in der Entwicklung konstante Nuisance-Spalten entfernen.
    if np.any(x[:,~keep]!=mean[~keep]):raise ValueError('strukturelle Entwicklungskonstante verändert')
    return np.column_stack([np.ones(len(x)),(x[:,keep]-mean[keep])/sd[keep]])

def separation(z,y):
    """Albert-Anderson-Kegel: alle signed margins >=0, mindestens eine >0.

    Vollständig: LP signed margin >=1 machbar. Quasi: maximale Summe
    unter |beta|<=1 strikt positiv. Keine Regression/Polungswahl als Ersatz.
    """
    z=np.asarray(z,float);y=np.asarray(y,float)
    a=z*(2*y-1)[:,None]
    # Exact geometric sufficient certificate, not an inference subsample:
    # A_S' w=0 with every w>=1, and rank(A_S)=p implies that A_S b>=0
    # forces A_S b=0, hence b=0. Therefore the full cone is {0} as well.
    # Only the certificate search uses at most 1024 deterministic rows; every
    # regression/AUC still uses all selected persons. If not certified, fall
    # back to the complete full/quasi LP without changing any decision rule.
    if len(a)>1024:
        subset=a[np.linspace(0,len(a)-1,1024,dtype=int)]
        if np.linalg.matrix_rank(subset)==a.shape[1]:
            cert=_lp(np.zeros(len(subset)),A_eq=subset.T,b_eq=np.zeros(a.shape[1]),bounds=[(1,None)]*len(subset),method='highs')
            if cert.success and np.isfinite(cert.x).all() and (cert.x>=1).all():
                residual=np.abs(subset.T@cert.x);scale=np.maximum(1,np.abs(subset).T@cert.x)
                if np.max(residual/scale)<=1e-10:return 'keine'
    full=_lp(np.zeros(z.shape[1]),A_ub=-a,b_ub=-np.ones(len(y)),bounds=[(None,None)]*z.shape[1],method='highs')
    if full.success:return 'vollstaendig'
    if full.status!=2:raise ValueError('Separation-LP numerisch nicht schätzbar')
    quasi=_lp(-a.mean(0),A_ub=-a,b_ub=np.zeros(len(y)),bounds=[(-1,1)]*z.shape[1],method='highs')
    if not quasi.success:raise ValueError('Quasiseparation-LP numerisch nicht schätzbar')
    return 'quasi' if -quasi.fun>1e-8 else 'keine'

def patient_counts(subjects,y):
    return {str(v):int(len(np.unique(np.asarray(subjects)[np.asarray(y)==v]))) for v in (0,1)}

def verdict(beta,p,ci,direction=1):
    success=direction*beta>0 and p<=.05 and (ci[0]>0 if direction>0 else ci[1]<0)
    return {'urteil':'erfolg' if success else 'statistischer_misserfolg','gegenrichtung':direction*beta<0,
            'vollstaendig_gegengerichtetes_ki':ci[1]<0 if direction>0 else ci[0]>0}

def fit(z,y,subjects,direction=1,weights=None,wald=True):
    z=np.asarray(z,float);y=np.asarray(y,float);subjects=np.asarray(subjects)
    w=np.ones(len(y)) if weights is None else np.asarray(weights,float)
    if w.shape!=y.shape or not np.isfinite(w).all() or (w<0).any():raise ValueError('Gewichte ungültig')
    active=w>0;z=z[active];y=y[active];subjects=subjects[active];w=w[active]
    n,p=z.shape;g=len(np.unique(subjects));counts=patient_counts(subjects,y)
    if not np.isfinite(z).all() or not np.isfinite(w).all() or (w<0).any() or not set(np.unique(y))=={0.,1.}:raise ValueError('finite Design-/Klassen-/Gewichtregel')
    if min(counts.values())<100:raise ValueError('weniger als 100 verschiedene Personen je Klasse')
    if n<=p or g<=1:raise ValueError('N<=p oder G<=1')
    if np.linalg.matrix_rank(z)!=p:raise ValueError('Rangdefizit/erst hier konstante Kovariate')
    sep=separation(z,y)
    if sep!='keine':raise ValueError('Separation: '+sep)
    beta=logistic_fit(z,y,w);prob=expit(z@beta);pen=np.r_[0,np.full(p-1,1e-6)]
    if not np.isfinite(beta).all():raise ValueError('Koeffizienten nicht endlich')
    score_n=float(np.max(np.abs(z.T@(w*(prob-y))+pen*beta))/n)
    score=float(np.max(np.abs(z.T@(w*(prob-y))+pen*beta))/w.sum())
    if score_n>=1e-7:raise ValueError('Score/N-Konvergenz nicht erfüllt')
    if score>=1e-7:raise ValueError('Score-Konvergenz nicht erfüllt')
    result={'beta':float(beta[-1]),'or':float(np.exp(beta[-1])),'N':n,'G':g,'p_parameters':p,'class_persons':counts,
            'converged':True,'score_over_weight_sum':score,'score_over_N':score_n,'weight_sum':float(w.sum()),'separation':sep,'ridge':1e-6,'solver_max_steps':60}
    if wald:
        if np.array_equal(w,np.ones(len(w))):cov=sandwich(z,y,beta,subjects)
        else:
            h=z.T@((w*prob*(1-prob))[:,None]*z)+np.diag(pen)
            _,inv=np.unique(subjects,return_inverse=True);u=np.zeros((g,p));np.add.at(u,inv,z*(w*(y-prob))[:,None])
            hi=np.linalg.inv(h);cov=hi@(u.T@u)@hi*g/(g-1)*(n-1)/(n-p)
        var=float(cov[-1,-1])
        if not np.isfinite(var) or var<=0:raise ValueError('Marker-CR1-Varianz nicht positiv/endlich')
        se=np.sqrt(var);ci=[float(beta[-1]-Z95*se),float(beta[-1]+Z95*se)];pv=float(2*norm.sf(abs(beta[-1]/se)))
        result.update({'variance_cr1':var,'se_cr1':float(se),'beta_ci95_wald':ci,'or_ci95_wald':np.exp(ci).tolist(),
                       'p_raw':pv,'cr1_factor':g/(g-1)*(n-1)/(n-p),**verdict(beta[-1],pv,ci,direction)})
    # Gepaarter Basismodellfit auf denselben Personen und mit derselben Skalierung.
    base=z[:,:-1]
    if separation(base,y)!='keine':raise ValueError('Separation im Basismodell')
    b0=logistic_fit(base,y,w)
    a0=auc(y,base@b0,w);a1=auc(y,z@beta,w)
    result.update({'auc_G':a0,'auc_G_marker':a1,'delta_auc':a1-a0,'auc_marker_original_steigend':auc(y,z[:,-1],w),
                   'auc_typ':'apparente AUC im Assoziationsfit','calibration':'nicht konfirmatorisch prüfbar unter K1'})
    return result

def draws(union,subjects,r):
    union=np.asarray(union);subjects=np.asarray(subjects)
    if len(union)==0 or not np.array_equal(union,np.unique(union)):raise ValueError('Personenunion nicht sortiert/eindeutig')
    positions=np.searchsorted(union,subjects)
    if (positions>=len(union)).any() or not np.array_equal(union[positions],subjects):raise ValueError('Person fehlt in Indexunion')
    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([SEED,int(r)])))
    multiplicities=np.bincount(rng.integers(0,len(union),len(union)),minlength=len(union))
    return multiplicities[positions].astype(float)

def holm(results,h1_success):
    """Genau H2-H4; Nichtschätzbarkeit p=1 intern, kein kleinerer Nenner."""
    ids=['BEF-0042','BEF-0081','BEF-0049']
    order=sorted(ids,key=lambda k:(results[k].get('p_raw',1) if results[k].get('p_raw') is not None else 1,k))
    running=0.;continuing=True;out={}
    for j,k in enumerate(order):
        pv=results[k].get('p_raw');internal=1 if pv is None else pv
        running=max(running,(3-j)*internal);reject=continuing and internal<=.05/(3-j)
        continuing=reject
        beta=results[k].get('beta');direction=-1 if k=='BEF-0042' else 1
        out[k]={'p_raw':pv,'p_internal':internal,'p_holm':min(1.,running),'rank':j+1,'threshold':.05/(3-j),
                'H1_gate':bool(h1_success),'holm_reject':bool(reject),'confirmatory_success':bool(h1_success and reject and beta is not None and direction*beta>0)}
    return out
