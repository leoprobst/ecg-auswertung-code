"""Pure pre-specified statistics functions for the test-partition replication; no file access."""
import numpy as np
from scipy.stats import rankdata,spearmanr
def ci(x):
    x=np.asarray(x);v=x[np.isfinite(x)];n=len(v)
    return {'low':float(np.quantile(v,.025)) if n>=950 else None,'high':float(np.quantile(v,.975)) if n>=950 else None,'valid':n,'invalid_fraction':float(1-n/1000)}
def weights(ids,seed):
    rng=np.random.Generator(np.random.PCG64(seed));u=len(ids)
    for lo in range(0,1000,20):
        draw=rng.integers(0,u,(20,u));w=np.empty((20,u))
        for j in range(20):w[j]=np.bincount(draw[j],minlength=u)
        yield lo,w
def weighted_metrics(x,y,w):
    """Repeated-person rank/quintile algebra, including average ties."""
    x=np.asarray(x,float);y=np.asarray(y,float);good=np.isfinite(x)&np.isfinite(y)
    x=x[good];y=y[good];w=np.asarray(w,dtype=float)[:,good];n=w.sum(1)
    def rank(v):
        order=np.argsort(v,kind='stable');sortedv=v[order]
        if not len(v):return np.empty_like(w)
        starts=np.r_[0,np.flatnonzero(np.diff(sortedv)!=0)+1]
        totals=np.add.reduceat(w[:,order],starts,axis=1)
        ends=np.cumsum(totals,axis=1);r=ends-totals/2+.5
        groups=np.repeat(np.arange(len(starts)),np.diff(np.r_[starts,len(v)]))
        out=np.empty_like(w);out[:,order]=r[:,groups];return out
    rx,ry=rank(x),rank(y)
    delta=y-x
    xc=x-np.mean(x) if len(x) else x
    yc=y-np.mean(y) if len(y) else y
    dc=delta-np.mean(delta) if len(delta) else delta
    def covariance(a,b):
        return np.sum(w*a*b,axis=1)-np.sum(w*a,axis=1)*np.sum(w*b,axis=1)/n
    with np.errstate(divide='ignore',invalid='ignore'):
        sx=np.sqrt(np.maximum(0,covariance(xc,xc)/(n-1)));sy=np.sqrt(np.maximum(0,covariance(yc,yc)/(n-1)))
        sd=np.sqrt(np.maximum(0,covariance(dc,dc)/(n-1)));bias=np.sum(w*delta,axis=1)/n
        rho=covariance(rx,ry)/np.sqrt(covariance(rx,rx)*covariance(ry,ry))
        width=3.92*sd/sx
        qx=np.clip(np.floor(5*(rx-.5)/n[:,None]),0,4);qy=np.clip(np.floor(5*(ry-.5)/n[:,None]),0,4)
        switch=np.sum(w*(qx!=qy),axis=1)/n
    bias[n<2]=np.nan
    cls=np.full(len(w),3,np.int8)
    finite=(n>=100)&(sx>0)&(sy>0)&np.isfinite(rho)
    cls[finite]=1
    cls[finite&(rho>=.95)&(width<=.5)&(switch<=.10)]=0
    cls[finite&((rho<.80)|(width>2)|(switch>.40))]=2
    # Constants retain a separate code, never estimable matrix-dependent agreement.
    cls[(sx==0)&(n>=2)&(np.sum(w*((x-y)!=0),axis=1)==0)]=4
    # M1 does not estimate rank/quintile/relative-width outputs outside its gate.
    rho[~finite]=np.nan;width[~finite]=np.nan;switch[~finite]=np.nan
    return cls,{'n_pairs':n,'bias':bias,'difference_sd':sd,'kors_sd':sx,'dower_sd':sy,'loa_low':bias-1.96*sd,'loa_high':bias+1.96*sd,'spearman':rho,'relative_loa_width':width,'relative_loa_width_symmetric_SD':3.92*sd/np.sqrt((sx*sx+sy*sy)/2),'quintile_switch':switch}
def combine(a,b):
    out=np.maximum(a,b);out[(a==3)|(b==3)]=3
    out[((a==4)|(b==4))&~((a==4)&(b==4))]=3
    return out
def agreement(dev,test):
    dev=np.asarray(dev);test=np.asarray(test);n=len(dev)
    if n==0:return np.nan,np.nan
    observed=float(np.mean(dev==test));cats=set(dev)|set(test)
    expected=sum(np.sum(dev==k)*np.sum(test==k) for k in cats)/n**2
    return observed,(observed-expected)/(1-expected) if expected<1 else np.nan
def selftest():
    from . import torlogik as t
    from ..transform_katalog.metriken import vergleichen,selbsttest
    rng=np.random.Generator(np.random.PCG64(1632026))
    for size in (99,100,205):
        x=np.round(rng.normal(size=size),1);y=x+rng.normal(size=size)*.2
        w=rng.integers(0,4,(20,size))
        cls,m=weighted_metrics(x,y,w)
        for j in range(20):
            rx=np.repeat(x,w[j]);ry=np.repeat(y,w[j]);r=vergleichen(rx,ry)
            codes={'stabil':0,'bedingt':1,'transformabhaengig':2,'nicht_schaetzbar':3,'konstant_identisch':4}
            assert cls[j]==codes[r['class']]
            for k in ('bias','difference_sd','spearman','relative_loa_width','quintile_switch'):
                if k in r:assert np.isclose(m[k][j],r[k],rtol=1e-10,atol=1e-10),k
    assert agreement([1,2],[1,3])[0]==.5
    assert not t.interval_inside(None,None,0,1,1000)
    assert ci(np.r_[np.zeros(949),np.full(51,np.nan)])['low'] is None
    insufficient,insufficient_metrics=weighted_metrics(np.arange(99.),np.arange(99.),np.ones((1,99)))
    assert insufficient[0]==3 and np.isnan(insufficient_metrics['spearman'][0])
    from . import aw163_extract as e
    scope={'np':np,'B':1000}
    e.a.ast_functions(e.ROOT/'werkzeuge/rv_a1/wiederholbarkeit.py',['estimate','moments'],scope)
    persons=np.repeat(np.arange(50),2);ids=np.arange(50)
    x=rng.normal(size=(100,3));y=x+rng.normal(size=(100,3))*.3
    x[:10,2]=np.nan
    pp,_,nn,ns,bsd,den,scale=scope['moments'](x,y,persons,ids)
    assert ns[2]==45 and np.isfinite(pp).all()
    z=np.ones((100,1));constant=scope['moments'](z,z,persons,ids)[0]
    assert np.isnan(constant[0,0])
    few=scope['moments'](x[:58],y[:58],persons[:58],ids)[0]
    assert np.isnan(few).all()
    a=np.arange(100.)[:,None];b=-a
    assert scope['moments'](a,b,persons,ids)[0][0,0]<0
    assert len(list(weights(ids,1461201)))==50
    return {'weighted_rank_vs_literal_repeated_person_draws':60,'original_M_checks':selbsttest(),'W_missing_constant_29persons_negativeICC_shared_chunks':'passed','missing_agreement_and_949_resamples':'passed','test_measurements_used':False}
