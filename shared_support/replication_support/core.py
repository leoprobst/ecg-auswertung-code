"""Unabhängige AUC, Patientenbootstrap und gehaltene Modellrechnung (AW-032)."""
import numpy as np
from scipy.special import expit

def assert_development(frame):
    if not set(frame['split'].unique()) <= {'train','validation'}:
        raise ValueError('Nicht freigegebener Split')
    if frame[['subject_id','study_id','split']].isna().any().any():
        raise ValueError('Fehlende Identität')
    if frame.groupby('subject_id')['split'].nunique().max() > 1:
        raise ValueError('Patient in beiden Folds: Leakage')

class RankedAUC:
    """Gewichtetes Mann-Whitney-U, Ties halb; einmaliges Sortieren möglich."""
    def __init__(self,y,score):
        y=np.asarray(y);score=np.asarray(score,dtype=float)
        if not np.isfinite(score).all() or not set(np.unique(y)) <= {0,1}:
            raise ValueError('AUC-Eingang ungültig')
        self.order=np.argsort(score,kind='stable')
        self.y=y[self.order]
        z=score[self.order]
        self.starts=np.r_[0,np.flatnonzero(np.diff(z)!=0)+1]
        self.n=len(y)
    def __call__(self,weights=None):
        w=np.ones(self.n) if weights is None else np.asarray(weights)[self.order]
        if np.any(w<0):raise ValueError('Negative Gewichte')
        p=np.add.reduceat(w*self.y,self.starts)
        n=np.add.reduceat(w*(1-self.y),self.starts)
        if p.sum()==0 or n.sum()==0:raise ValueError('Nur eine Klasse')
        return float(np.sum(p*(np.cumsum(n)-0.5*n))/(p.sum()*n.sum()))

def auc(y,score,weights=None):return RankedAUC(y,score)(weights)

def cluster_weights(subjects,rng):
    unique,inverse=np.unique(subjects,return_inverse=True)
    counts=np.bincount(rng.integers(0,len(unique),size=len(unique)),minlength=len(unique))
    return counts[inverse].astype(float)

def interval(values):
    a=np.asarray(values,dtype=float)
    if not np.isfinite(a).all():raise ValueError('Nichtendlicher Bootstrap')
    return [float(x) for x in np.quantile(a,[.025,.975])]

def pool_summary(estimates,replicates,weights):
    a=np.asarray(replicates);w=np.asarray(weights,dtype=float)
    covariance=np.cov(a,rowvar=False,ddof=1)
    return {'fold_auc':list(map(float,estimates)),
            'fold_ci95':[interval(a[:,i]) for i in range(2)],
            'weights':w.tolist(),'pooled':float(w@estimates),
            'pooled_ci95':interval(a@w),'fold_covariance':covariance.tolist(),
            'pooled_variance_with_covariance':float(w@covariance@w),
            'pooled_variance_without_covariance':float(np.sum(w*w*np.diag(covariance)))}

def age_basis(age):
    """Natürlicher kubischer Spline, feste Knoten 40/60/80 Jahre."""
    a=np.asarray(age,dtype=float)/20
    def d(k):return (np.maximum(a-k,0)**3-np.maximum(a-4,0)**3)/(4-k)
    return np.column_stack([a,d(2)-d(3)])

def design(frame,model,comparator):
    cols=[age_basis(frame['age']),frame[['sex_numeric','heart_rate']].to_numpy(float)]
    if model in ('B1','B2'):cols.append(frame[['qrs_duration_ms','qtc_fridericia_ms']].to_numpy(float))
    if model=='B2' and comparator!='qrs_duration_ms':cols.append(frame[[comparator]].to_numpy(float))
    return np.column_stack(cols)

def standardize(train,test,w):
    mean=np.average(train,axis=0,weights=w)
    sd=np.sqrt(np.average((train-mean)**2,axis=0,weights=w));sd[sd<1e-12]=1
    return np.column_stack([np.ones(len(train)),(train-mean)/sd]),np.column_stack([np.ones(len(test)),(test-mean)/sd])

def logistic_fit(x,y,w,initial=None):
    """Newton/Line Search, feste winzige Ridge, keine Hyperparametersuche."""
    beta=np.zeros(x.shape[1]) if initial is None else initial.copy()
    penalty=np.full(len(beta),1e-6);penalty[0]=0
    def objective(b):
        z=x@b
        return float(np.sum(w*(np.logaddexp(0,z)-y*z))+.5*np.sum(penalty*b*b))
    for _ in range(60):
        probability=expit(x@beta)
        grad=x.T@(w*(probability-y))+penalty*beta
        if np.max(np.abs(grad))/w.sum()<1e-7:return beta
        h=x.T@((w*probability*(1-probability))[:,None]*x)+np.diag(penalty)
        step=np.linalg.solve(h,grad)
        loss=objective(beta);alpha=1.
        for _ in range(30):
            candidate=beta-alpha*step
            if objective(candidate)<=loss-1e-4*alpha*(grad@step):break
            alpha*=.5
        else:raise RuntimeError('Logistische Line Search gescheitert')
        beta=candidate
    raise RuntimeError('Logistischer Fit nicht konvergiert')

def held_models(train,test,marker_train,marker_test,y_train,y_test,w_train=None,w_test=None):
    """Fit nur Train; Testlabels ausschließlich in AUC. Rückgabe B/B+/Δ/Residual-AUC."""
    w_train=np.ones(len(y_train)) if w_train is None else w_train
    w_test=np.ones(len(y_test)) if w_test is None else w_test
    for y,w in ((y_train,w_train),(y_test,w_test)):
        if sum(w[y==0])<=0 or sum(w[y==1])<=0:raise ValueError('Nur eine Klasse')
    xt,xv=standardize(train,test,w_train)
    bt=logistic_fit(xt,y_train,w_train)
    at,av=standardize(np.column_stack([train,marker_train]),np.column_stack([test,marker_test]),w_train)
    ba=logistic_fit(at,y_train,w_train)
    baseline=auc(y_test,xv@bt,w_test);augmented=auc(y_test,av@ba,w_test)
    # Gleiche Basis und weighted least squares nur im fitting_split.
    coef=np.linalg.lstsq(xt*np.sqrt(w_train[:,None]),marker_train*np.sqrt(w_train),rcond=None)[0]
    rt=marker_train-xt@coef;rv=marker_test-xv@coef
    direction=1 if auc(y_train,rt,w_train)>=.5 else -1
    residual=auc(y_test,direction*rv,w_test)
    return np.array([baseline,augmented,augmented-baseline,residual])
