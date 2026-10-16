"""Proportional-odds logit with analytic gradient and frequency weights."""
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize

def objective(t,x,y,w,k):
    p=x.shape[1];b=t[:p];alpha=t[p:];cuts=np.r_[alpha[0],alpha[0]+np.cumsum(np.exp(alpha[1:]))]
    eta=x@b;lo=np.zeros(len(y));hi=np.ones(len(y));fl=np.zeros(len(y));fh=np.zeros(len(y))
    il=y>0;ih=y<k-1
    lo[il]=expit(cuts[y[il]-1]-eta[il]);hi[ih]=expit(cuts[y[ih]]-eta[ih]);fl=lo*(1-lo);fh=hi*(1-hi)
    prob=np.maximum(hi-lo,1e-300);loss=-np.sum(w*np.log(prob));gb=x.T@(w*(fh-fl)/prob)
    gc=np.zeros(k-1)
    np.add.at(gc,y[ih],-w[ih]*fh[ih]/prob[ih]);np.add.at(gc,y[il]-1,w[il]*fl[il]/prob[il])
    ga=np.r_[gc.sum(),np.exp(alpha[1:])*np.cumsum(gc[::-1])[::-1][1:]]
    return loss/w.sum(),np.r_[gb,ga]/w.sum()

def ordinal_fit(x,y,w,initial=None):
    k=int(y.max())+1;p=x.shape[1]
    if initial is None:
        probs=np.cumsum(np.bincount(y,weights=w,minlength=k))[:-1]/w.sum();cuts=np.log(probs/(1-probs));initial=np.r_[np.zeros(p),cuts[0],np.log(np.diff(cuts))]
    result=minimize(objective,initial,args=(x,y,w,k),method='BFGS',jac=True,options={'gtol':1e-6,'maxiter':200})
    grad=objective(result.x,x,y,w,k)[1]
    if not np.isfinite(result.x).all() or np.max(np.abs(grad))>1e-5:raise ValueError('ordinal_nonconvergence')
    return result.x,float(np.max(np.abs(grad)))
