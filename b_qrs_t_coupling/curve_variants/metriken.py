"""Reine Kurvenkerne der eingefrorenen Runde 1, keine Registrymutation."""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import connected_components
from scipy.ndimage import label

EPS=1e-10

def resample(curve,n=64,time=False):
    x=np.asarray(curve,float)
    if x.ndim!=2 or len(x)<2 or not np.isfinite(x).all(): return None
    if time:
        position=np.linspace(0,1,len(x))
    else:
        keep=np.r_[True,np.linalg.norm(np.diff(x,axis=0),axis=1)>0]
        x=x[keep]
        if len(x)<2: return None
        position=np.r_[0,np.cumsum(np.linalg.norm(np.diff(x,axis=0),axis=1))]
        if position[-1]<=EPS: return None
        position=position/position[-1]
    return np.stack([np.interp(np.linspace(0,1,n),position,x[:,j]) for j in range(x.shape[1])],axis=1)

def normalize(x):
    if x is None: return None
    x=x-x.mean(axis=0); scale=np.sqrt(np.mean(np.sum(x*x,axis=1)))
    return None if scale<=EPS else x/scale

def frechet(q,t):
    if q is None or t is None: return None
    costs=cdist(q,t); table=np.empty_like(costs)
    for i in range(len(q)):
        for j in range(len(t)):
            previous=0 if i==j==0 else table[i-1,j] if j==0 else table[i,j-1] if i==0 else min(table[i-1,j],table[i-1,j-1],table[i,j-1])
            table[i,j]=max(costs[i,j],previous)
    return float(table[-1,-1])

def form_distance(q,t,mode='arc'):
    q=resample(q,time=mode=='time'); t=resample(t,time=mode=='time')
    if q is None or t is None:return None
    if mode=='shared':
        q=q-q.mean(0); t=t-t.mean(0)
        scale=np.sqrt((np.mean(np.sum(q*q,1))+np.mean(np.sum(t*t,1)))/2)
        if scale<=EPS or min(np.linalg.norm(q),np.linalg.norm(t))<=EPS:return None
        q=q/scale;t=t/scale
    else:q=normalize(q);t=normalize(t)
    if q is None or t is None:return None
    if mode=='rotation':
        u,s,vt=np.linalg.svd(t.T@q)
        if s[1]<=EPS:return None
        correction=np.eye(q.shape[1]);correction[-1,-1]=np.linalg.det(u@vt)
        t=t@(u@correction@vt)
    return frechet(q,t)

def graph_statistics(adjacency):
    a=np.asarray(adjacency,bool)
    if a.ndim!=2 or a.shape[0]!=a.shape[1] or not np.array_equal(a,a.T) or np.diag(a).any():raise ValueError('Einfacher ungerichteter Graph erforderlich')
    v=len(a);e=int(a.sum()//2);c=int(connected_components(a,directed=False,return_labels=False))
    return {'edges':e,'vertices':v,'components':c,'beta':e-v+c}

def repaired_qrs(curve):
    if len(curve)<4:return None
    x=normalize(resample(curve))
    if x is None:return None
    d=cdist(x,x);i,j=np.triu_indices(64,8);nonlocal_d=d[i,j]
    contact=np.zeros((64,64),bool);contact[i,j]=nonlocal_d<.15
    components,count=label(contact,structure=np.ones((3,3),int))
    sizes=np.bincount(components.ravel());events=sum(sizes[1:]>=3)
    p10=graph_statistics((d>0)&(d<np.percentile(nonlocal_d,10)))
    fixed=graph_statistics((d>0)&(d<.15))
    return {'near':float(np.mean(nonlocal_d<.15)),'graph_p10':p10['beta'],'graph_fixed':fixed['beta']/2016,'events':int(events),'first_last_contact':bool(contact[0,-1]),'pair_count':len(i),'p10_components':p10['components'],'fixed_components':fixed['components'],'n':64}

def near_p5_baseline(n):
    m=max(0,(n-5)*(n-4)//2) if n>=5 else 0
    return {'pairs':m,'distinct_count':int(np.floor(.05*(m-1))+1) if m else 0}
