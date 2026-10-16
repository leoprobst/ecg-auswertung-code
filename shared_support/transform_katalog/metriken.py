"""Vorab gebundene deskriptive Kennzahlen; keine klinischen Toleranzen."""
import numpy as np
from scipy.stats import spearmanr,rankdata

def vergleichen(kors,dower):
    x=np.asarray(kors,dtype=float);y=np.asarray(dower,dtype=float)
    if x.shape!=y.shape or x.ndim!=1:raise ValueError('Gepaarte Vektoren erforderlich')
    good=np.isfinite(x)&np.isfinite(y)
    out={'n_total':len(x),'n_kors':int(np.isfinite(x).sum()),'n_dower':int(np.isfinite(y).sum()),'n_pairs':int(good.sum()),'class':'nicht_schaetzbar'}
    x=x[good];y=y[good];n=len(x)
    if n<2:return out
    delta=y-x;sd=float(np.std(delta,ddof=1));bias=float(np.mean(delta));sx=float(np.std(x,ddof=1))
    out.update(bias=bias,difference_sd=sd,loa_low=bias-1.96*sd,loa_high=bias+1.96*sd,kors_sd=sx,quantile_low=float(np.quantile(delta,.025)),quantile_high=float(np.quantile(delta,.975)))
    if sx==0:
        if np.array_equal(x,y):out['class']='konstant_identisch'
        return out
    if n<100 or np.std(y,ddof=1)==0:return out
    rho=float(spearmanr(x,y).statistic)
    quintile=lambda v:np.clip(np.floor(5*(rankdata(v,method='average')-.5)/n),0,4)
    switch=float(np.mean(quintile(x)!=quintile(y)))
    width=3.92*sd/sx
    out.update(spearman=rho,relative_loa_width=width,quintile_switch=switch,distinct_kors=len(np.unique(x)),distinct_dower=len(np.unique(y)))
    if rho>=.95 and width<=.5 and switch<=.10:out['class']='stabil'
    elif rho<.80 or width>2 or switch>.40:out['class']='transformabhaengig'
    else:out['class']='bedingt'
    return out

def selbsttest():
    x=np.arange(200,dtype=float)
    assert vergleichen(x,x)['class']=='stabil'
    shift=vergleichen(x,x+1000)
    assert shift['bias']==1000 and shift['quintile_switch']==0
    assert vergleichen(x,-x)['class']=='transformabhaengig'
    assert vergleichen(np.ones(200),np.ones(200))['class']=='konstant_identisch'
    assert vergleichen(x[:99],x[:99])['class']=='nicht_schaetzbar'
    a=x.copy();a[0]=np.nan
    assert vergleichen(a,x)['n_pairs']==199
    binary=vergleichen(np.repeat([0.,1.],100),np.repeat([0.,1.],100))
    assert binary['distinct_kors']==2 and binary['quintile_switch']==0
    return {'checks':7,'status':'passed','clinical_tolerance':False}
