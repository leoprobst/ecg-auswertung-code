"""AW157 kernels: RC affine Abbildung, Winkel-RCS und SIMEX."""
import numpy as np
from werkzeuge.nachrechnung.core import logistic_fit

def variance(x,w):
    m=np.average(x,weights=w);return float(w@((x-m)**2)/w.sum())
def cov(x,w):
    c=x-np.average(x,axis=0,weights=w);return c.T@(w[:,None]*c)/w.sum()
def icc(x,y,w):
    n=w.sum();sx=w@x;sy=w@y;sxx=w@(x*x);syy=w@(y*y);sxy=w@(x*y)
    if n<30:raise ValueError('pair_n<30')
    col=(sx-sy)**2/(2*n);msr=((sxx+2*sxy+syy)/2-(sx+sy)**2/(2*n))/(n-1)
    mse=((sxx+syy-2*sxy)/2-col)/(n-1);den=msr+mse+2*(col-mse)/n
    return float((msr-mse)/den)
def rc(z,w,error,b):
    ai=z.shape[1]-2;zi=[i for i in range(z.shape[1]) if i!=ai];other=z[:,zi]
    coef=np.linalg.solve(other.T@(w[:,None]*other),other.T@(w*z[:,ai]));res=z[:,ai]-other@coef
    v=float(w@(res**2)/w.sum());rel=1-error/v
    if rel<=0:raise ValueError('conditional_reliability<=0')
    mat=np.eye(z.shape[1]);mat[ai,ai]=rel;mat[zi,ai]=(1-rel)*coef
    corrected=np.linalg.solve(mat,b)
    return {'beta':float(corrected[-1]),'angle_beta':float(corrected[ai]),'conditional_reliability':float(rel),'residual_variance':v},mat
def multirc(z,w,error,b):
    k=z.shape[1];other=z[:,:-2];W=z[:,-2:];coef=np.linalg.solve(other.T@(w[:,None]*other),other.T@(w[:,None]*W))
    residual=W-other@coef;S=cov(residual,w);T=S-error;eig=np.linalg.eigvalsh(T)
    meta={'residual_covariance':S.tolist(),'error_covariance':error.tolist(),'latent_covariance_eigenvalues':eig.tolist(),'error_correlation':float(error[0,1]/np.sqrt(error[0,0]*error[1,1]))}
    if np.linalg.eigvalsh(error).min()<-1e-10:raise ValueError('error_covariance_not_psd')
    if eig.min()<=1e-10:return {'reason':'latent_covariance_not_positive_definite',**meta},None
    H=np.linalg.solve(S,T);mat=np.eye(k);mat[-2:,-2:]=H;mat[:-2,-2:]=coef@(np.eye(2)-H)
    corrected=np.linalg.solve(mat,b)
    return {'beta':float(corrected[-1]),'angle_beta':float(corrected[-2]),**meta},mat

def spline_design(z,angle,entry,rcs):
    basis=(rcs(angle,entry['aw157_knots_deg'])-entry['aw157_spline_mean'])/entry['aw157_spline_sd']
    return np.column_stack([z[:,:-2],basis,z[:,-1]])
def simex(z,y,w,error,entry,rcs,seed,draws=20,spline=False,b=None):
    sd=entry['scaling_B3']['sd'][-2];mean=entry['scaling_B3']['mean'][-2];angle=z[:,-2]*sd+mean
    design=spline_design(z,angle,entry,rcs) if spline else z.copy()
    b=logistic_fit(design,y,w) if b is None else b
    curve=[b.copy()];outside=[];scatter=[]
    noise=np.random.default_rng(seed).normal(size=(draws//2,len(y)))
    for lam in [.5,1.,1.5,2.]:
        fits=[];o=[]
        for e in noise:
            for sign in [-1,1]:
                perturbed=angle+sign*np.sqrt(lam*error)*e
                if spline:dd=spline_design(z,perturbed,entry,rcs)
                else:dd=design;dd[:,-2]=(perturbed-mean)/sd
                fits.append(logistic_fit(dd,y,w,initial=b))
                o.append(float(np.average((perturbed<0)|(perturbed>180),weights=w)))
        fits=np.asarray(fits);curve.append(fits.mean(0));outside.append(float(np.mean(o)));scatter.append(float(fits[:,-1].std(ddof=1)))
    curve=np.asarray(curve);lam=np.array([0,.5,1,1.5,2.]);q=np.array([1,-1,1])@np.linalg.lstsq(np.column_stack([np.ones(5),lam,lam**2]),curve,rcond=None)[0]
    linear=np.array([1,-1])@np.linalg.lstsq(np.column_stack([np.ones(5),lam]),curve,rcond=None)[0]
    return {'beta':float(q[-1]),'linear_beta':float(linear[-1]),'curve':curve[:,-1].tolist(),'outside_fraction':outside,'mc_scatter':scatter}

def scenarios():
    # Return tuples: group, basis, error source. All fixed before results.
    d=[]
    for rel in [.45,.55,.617,.70,.46,.31]:d.append(('gesamt','linear','icc_'+str(rel)))
    d.append(('gesamt','linear','pairs_gesamt'))
    for rel in [.45,.55,.617,.70]:d.append(('gesamt','spline','icc_'+str(rel)))
    d.append(('gesamt','spline','pairs_gesamt'))
    for group in ['Sinus','schmal_ohne_Block']:
        for err in ['icc_0.617','icc_0.7','pairs_stratum']:d.append((group,'linear',err))
    d.extend([('AF','linear','icc_0.46'),('LSB','linear','icc_0.31')])
    return d
