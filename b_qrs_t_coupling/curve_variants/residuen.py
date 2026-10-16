"""Train-fixe Huber-Residuen und patientenweise 5-fache OOF-Rechnung."""
import warnings
import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GroupKFold

class Residualizer:
    def fit(self,numeric,sex,target):
        x=np.asarray(numeric,float);y=np.asarray(target,float);sex=np.asarray(sex)
        if x.ndim!=2 or x.shape[1]!=4 or not np.isfinite(x).all() or not np.isfinite(y).all() or any(s is None or s!=s for s in sex):raise ValueError('Vollständige QRS/QT/HF/Alter/Geschlecht benötigt')
        self.mean=x.mean(0);self.sd=x.std(0,ddof=1)
        if (self.sd<=0).any() or not np.isfinite(self.sd).all():raise ValueError('Null-SD')
        self.categories=sorted(set(sex))
        design=self.design(x,sex)
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            self.model=HuberRegressor(epsilon=1.35,alpha=0,max_iter=200,tol=1e-5,fit_intercept=True).fit(design,y)
        return self
    def design(self,x,sex):
        sex=np.asarray(sex)
        if any(s not in self.categories for s in sex):raise ValueError('Unbekannte held-out Geschlechtskategorie')
        return np.column_stack([(np.asarray(x,float)-self.mean)/self.sd]+[(sex==c).astype(float) for c in self.categories])
    def residual(self,numeric,sex,target):
        return np.asarray(target,float)-self.model.predict(self.design(numeric,sex))

def oof_residual(numeric,sex,target,patients):
    x=np.asarray(numeric);sex=np.asarray(sex);y=np.asarray(target);out=np.full(len(y),np.nan)
    for train,held in GroupKFold(5).split(x,groups=patients):
        model=Residualizer().fit(x[train],sex[train],y[train])
        out[held]=model.residual(x[held],sex[held],y[held])
    return out
