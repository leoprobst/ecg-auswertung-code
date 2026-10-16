"""Dimensionsunabhängige Fréchet-Kerne; gleiche 64-Punkte-Bogenlänge wie Release.

QRS-max ist der Wortlaut von §9.5, centered_rms der tatsächliche Release-Vertrag.
Diese beiden Messgrößen werden niemals unter derselben Marker-ID ausgegeben.
"""
import numpy as np
from scipy.spatial.distance import cdist
from werkzeuge.varianten.metriken import resample, frechet, form_distance
LEADS=('I','II','V1','V2','V3','V4','V5','V6')

def coupling(q,t,normalization='centered_rms'):
    q=resample(q);t=resample(t)
    if q is None or t is None or q.shape[1]!=t.shape[1]:return None
    q=q-q.mean(0);t=t-t.mean(0)
    if normalization=='centered_rms':
        qs=np.sqrt(np.mean(np.sum(q*q,1)));ts=np.sqrt(np.mean(np.sum(t*t,1)))
    elif normalization=='qrs_max':qs=ts=np.max(np.linalg.norm(q,axis=1))
    else:raise ValueError('Normvertrag')
    if min(qs,ts)<=1e-10 or np.linalg.norm(t)<=1e-10:return None
    return frechet(q/qs,t/ts)

def qrs_peak(signal,onset,offset,origin):
    x=np.asarray(signal,float);o=np.asarray(origin,float)
    if x.ndim!=2 or o.shape!=(x.shape[1],) or not (0<=onset<offset<len(x)):return None
    s=x[onset:offset+1]-o
    if not np.isfinite(s).all() or not np.isfinite(o).all():return None
    return float(np.max(np.linalg.norm(s,axis=1)))

def synthetic():
    from decimal import Decimal
    from werkzeuge.varianten.metriken import form_distance
    out=[]
    for dim in (3,8):
        for norm in ('centered_rms','qrs_max'):
            vals=[]
            for fs in (500,1000):
                u=np.linspace(0,1,round(.12*fs)+1);v=np.linspace(0,1,round(.25*fs)+1)
                def curve(u):
                    return np.column_stack([np.sin(np.pi*u),np.sin(2*np.pi*u),.3*np.cos(3*np.pi*u)]+[.1*np.sin((j+1)*np.pi*u) for j in range(dim-3)])
                q=curve(u);t=.6*curve(v);t[:,1]*=-1
                val=coupling(q,t,norm);assert val is not None
                assert np.isclose(val,coupling(q*5+3,t*5+3,norm),rtol=1e-10)
                assert np.isclose(val,coupling(q[:,::-1],t[:,::-1],norm),rtol=1e-10)
                assert coupling(q,np.zeros_like(t),norm) is None
                bad=t.copy();bad[3,0]=np.nan;assert coupling(q,bad,norm) is None
                vals.append(val)
            rel=abs(vals[1]-vals[0])/vals[0];assert rel<.01
            out.append({'dimensions':dim,'normalization':norm,'500_Hz':vals[0],'1000_Hz':vals[1],'relative_difference':rel,'gate_less_1_percent':True})
    # Independenter rekursiver Referenzalgorithmus für winzige Kostenmatrix.
    a=np.array([[0,0],[1,0],[2,0]],float);b=np.array([[0,1],[2,1]],float)
    def brute(i,j):
        d=float(np.linalg.norm(a[i]-b[j]))
        if i==j==0:return d
        p=[]
        if i:p.append(brute(i-1,j))
        if j:p.append(brute(i,j-1))
        if i and j:p.append(brute(i-1,j-1))
        return max(d,min(p))
    assert np.isclose(frechet(a,b),brute(2,1))
    x=np.zeros((80,3));x[15]=[2,1,0];p=qrs_peak(x,10,30,[0,0,0]);x[60]=[50,0,0]
    assert qrs_peak(x,10,30,[0,0,0])==p
    x[:,2]*=-1;assert qrs_peak(x,10,30,[0,0,0])==p
    assert qrs_peak(x,-1,30,[0,0,0]) is None
    for threshold in ('40','50','95','115','120','60','720','168'):
        d=Decimal(threshold)
        assert Decimal(str(float(threshold)))==d
        assert d-Decimal('.000001')<d<d+Decimal('.000001')
    return {'status':'passed','FE20':out,'frechet_brute_force':True,'qrs_peak_T_outside_invariance':True,'decimal_boundaries':True}
