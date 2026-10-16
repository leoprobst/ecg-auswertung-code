"""Shared building blocks (test-set lock, data, core of the bivariate regression calibration).

Contract: the pre-registered protocol of this stage. No side effects on import.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os,sys,time,json,hashlib,ctypes,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
OUT=Path(str(data_path("b.aw211")))
PROTOKOLL=ROOT/'werkzeuge/b_test/aw211_protokoll.md'
A='spatial_qrs_t_angle_deg';F='qrs_t_normalized_discrete_frechet';AMP='spatial_qrs_peak_voltage_uv'
POP=Path(str(data_path("b.aw168_bef_0063_private_parquet")))
ENTRY=Path(str(data_path("b.aw168_bef_0063_entry_json")))
PAIRS=Path(str(data_path("b.aw157_pairs_private_parquet")))
AMPFILE=Path(str(data_path("b.mimic_validation_runs_a155_2026_10_04_d6e70e_repeatability_replicate_pairs_part_006_parquet")))
KEY=['subject_id','split','first_study_id','second_study_id']

class Mem(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ['total','free','page','commit','vtotal','vfree','extended']]
def memory():
    m=Mem();m.length=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m));return {'ram_free':m.free,'commit_free':m.commit}
def guard():
    if min(memory().values())<1.5*1024**3:raise SystemExit('less than 1.5 GB free: aborting')
PEAK=[0]
def start_watch(limit_gb=1.0):
    import psutil
    def watch():
        while True:
            PEAK[0]=max(PEAK[0],psutil.Process().memory_info().rss)
            if PEAK[0]>limit_gb*1024**3:os._exit(75)
            time.sleep(.5)
    threading.Thread(target=watch,daemon=True).start()

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def clean(v):
    import numpy as np
    if isinstance(v,dict):return {str(k):clean(a) for k,a in v.items()}
    if isinstance(v,(list,tuple)):return [clean(a) for a in v]
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    return v
def dump(p,v):Path(p).write_text(json.dumps(clean(v),ensure_ascii=False,indent=1,allow_nan=False),encoding='utf8')
def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def check_stamp():
    s=load(OUT/'protokoll_hash.json');assert s['sha256']==sha(PROTOKOLL),'Protocol modified after binding';return s

def footer_ok(p):
    import pyarrow.parquet as pq
    pf=pq.ParquetFile(p);names=pf.schema_arrow.names
    if 'split' not in names:raise SystemExit('STOP: no split column in '+str(p))
    i=names.index('split');out=[]
    for g in range(pf.metadata.num_row_groups):
        st=pf.metadata.row_group(g).column(i).statistics
        if st is None or not st.has_min_max:raise SystemExit('STOP: no split statistics')
        mn=st.min.decode() if isinstance(st.min,bytes) else st.min;mx=st.max.decode() if isinstance(st.max,bytes) else st.max
        if mn<'train' or mx>'validation' or st.null_count:raise SystemExit('STOP: possible test rows in '+str(p))
        out.append([mn,mx,int(st.null_count)])
    return {'rows':pf.metadata.num_rows,'rowgroup_split_min_max_null':out}
def read(p,cols):
    import pyarrow.dataset as ds
    guard();info=footer_ok(p)
    t=ds.dataset(str(p),format='parquet').to_table(columns=cols,filter=ds.field('split').isin(['train','validation'])).to_pandas()
    assert set(t.split.unique())<={'train','validation'} and len(t)==info['rows']
    return t,info

def outcome_frame():
    """All rows with eligible_reference (development) from the outcome file."""
    cols=['subject_id','study_id','split','age','sex_numeric','heart_rate','qrs_duration_ms','qtc_fridericia_ms',A,F,'preferred_lvef','eligible_reference','eligible_B3']
    f,info=read(POP,cols)
    return f,info,load(ENTRY)['scaling_B3']

def pair_frame():
    """Pairs with amplitude; returns the frame and information."""
    p,pinfo=read(PAIRS,['subject_id','split','first_study_id','second_study_id','hours_between',A+'_first',A+'_second',F+'_first',F+'_second'])
    a,ainfo=read(AMPFILE,KEY+[AMP+'_first',AMP+'_second'])
    n0=len(p);q=p.merge(a,on=KEY,how='left',validate='one_to_one')
    import numpy as np
    amp=q[[AMP+'_first',AMP+'_second']].to_numpy(float)
    ok=np.isfinite(amp).all(1)&(amp>0).all(1)
    return q.loc[ok].reset_index(drop=True),{'pairs_file':pinfo,'amp_file':ainfo,'pairs_before':int(n0),'pairs_with_valid_amplitude':int(ok.sum()),'pairs_excluded_amplitude':int((~ok).sum())}

import numpy as np
from scipy.special import expit

def wcov(x,w):
    c=x-np.average(x,axis=0,weights=w);return c.T@(w[:,None]*c)/w.sum()
def pair_error(delta,w):
    c=delta-np.average(delta,axis=0,weights=w);return c.T@(w[:,None]*c)/(w.sum()-1)/2
def components(z,y,w,b=None,logistic_fit=None):
    if b is None:b=logistic_fit(z,y,w)
    pr=expit(z@b);score=np.max(np.abs(z.T@(w*(pr-y))+np.r_[0,np.full(z.shape[1]-1,1e-6)]*b))/w.sum()
    if not np.isfinite(b).all() or score>=1e-7:raise ValueError('not converged')
    other=z[:,:-2];W=z[:,-2:]
    coef=np.linalg.solve(other.T@(w[:,None]*other),other.T@(w[:,None]*W));S=wcov(W-other@coef,w);V=wcov(W,w)
    return {'b':b,'S':S,'V':V,'coef':coef,'u':S@b[-2:]}
def anchor_rel(c,E):
    V=c['V'];return {'R_A':1-E[0,0]/V[0,0],'R_F':1-E[1,1]/V[1,1],'rho':E[0,1]/np.sqrt(E[0,0]*E[1,1])}
def corrected_E(c,E):
    """Corrected marker/angle log-OR, smallest eigenvalue of T, partial latent correlation."""
    S=c['S'];T=S-E;lam=np.linalg.eigvalsh(T).min()
    if lam<=1e-10:return np.nan,np.nan,lam,np.nan
    cw=np.linalg.solve(T,c['u']);return cw[1],cw[0],lam,T[0,1]/np.sqrt(T[0,0]*T[1,1])
