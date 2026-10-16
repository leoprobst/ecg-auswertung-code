"""AW155 isolated development-only measurement-error analysis."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os,sys,ctypes,time,json,hashlib,threading
from pathlib import Path
os.environ.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=Path(str(data_path("b.aw155")));OUT.mkdir(exist_ok=True)
phase=sys.argv[1]
sys.stdout=(OUT/(phase+'_stdout.log')).open('a',encoding='utf8',buffering=1)
sys.stderr=(OUT/(phase+'_stderr.log')).open('a',encoding='utf8',buffering=1)
class Mem(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ['total','free','page','commit','vtotal','vfree','extended']]
def memory():
    m=Mem();m.length=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m));return {'ram_free':m.free,'commit_free':m.commit}
def guard():
    while min(memory().values())<1.5*1024**3: print('WAIT_MEMORY',flush=True);time.sleep(30)
guard()
import numpy as np,pandas as pd,pyarrow.parquet as pq,pyarrow.dataset as ds,psutil
from scipy.special import expit
from werkzeuge.b_test.daten import masks,classify_lvef
from werkzeuge.b_test.statistik import raw_design,fixed_design,fit
from werkzeuge.nachrechnung.core import logistic_fit
PEAK=0;INPUTS={};SEED=1552026;B=500
def watch():
    global PEAK
    while True:
        PEAK=max(PEAK,psutil.Process().memory_info().rss)
        if PEAK>1.5*1024**3:os._exit(75)
        time.sleep(.2)
threading.Thread(target=watch,daemon=True).start()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str),encoding='utf8')
def bind(p):
    p=Path(p);assert 'split=test' not in str(p) and 'sealed' not in p.parts
    INPUTS[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size};return p
def read(p,cols):
    guard();p=bind(p);pf=pq.ParquetFile(p);assert set(cols)<=set(pf.schema_arrow.names)
    return ds.dataset(str(p),format='parquet').to_table(columns=cols).to_pandas()
def finish():
    for p in [__file__,ROOT/'werkzeuge/b_test/aw155_protokoll.md',ROOT/'werkzeuge/b_test/daten.py',ROOT/'werkzeuge/b_test/statistik.py',ROOT/'werkzeuge/nachrechnung/core.py']:bind(p)
    dump(OUT/(phase+'_manifest.json'),{'inputs':INPUTS,'peak_rss':PEAK,'memory':memory(),'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'test_access':False})
PARAM=ROOT/'werkzeuge/b_test/parameter_entwicklung.json'
def inventory():
    protocol=ROOT/'werkzeuge/b_test/aw155_protokoll.md'
    assert sha(protocol)=='f5c8a073adc7e90a4563d709aa0a39f46502f9c93294e2bae51f3ac8de3ddde5'
    dump(OUT/'protokoll_hash.json',{'sha256':sha(protocol),'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'before_results':True})
    entry=json.loads(bind(PARAM).read_text(encoding='utf8'))['hypotheses']['BEF-0063']
    paths=[Path(entry['input_development']['path']),Path(str(data_path("b.messaudit_repeatability_paargruppen_privat_parquet")))]
    paths+=list(Path(str(data_path("b.mimic_validation_runs_a155_2026_10_04_d6e70e_repeatability_replicate_pairs"))).glob('*.parquet'))
    schemas={str(p):{'columns':pq.ParquetFile(p).schema_arrow.names,'rows':pq.ParquetFile(p).metadata.num_rows} for p in paths}
    dump(OUT/'schemas.json',schemas);print(json.dumps(schemas,ensure_ascii=False))

KEY=['subject_id','split','first_study_id','second_study_id'];ANGLE='spatial_qrs_t_angle_deg';MARKER='qrs_t_normalized_discrete_frechet'
def prepare():
    assert (OUT/'protokoll_hash.json').exists()
    entry=json.loads(bind(PARAM).read_text(encoding='utf8'))['hypotheses']['BEF-0063']
    p=Path(entry['input_development']['path']);assert sha(p)==entry['input_development']['sha256']
    cols=['subject_id','study_id','split','age','sex','heart_rate','qrs_duration_ms','qtc_fridericia_ms',ANGLE,MARKER,'preferred_lvef','lvef_conflict_gt_10','nearest_tte_measurement_id','nearest_tte_delta_hours','label_state','eligibility_state','index_selection_strategy','time_precision','time_delta_hours','evidence_ids_json','ehr_time_alignment_confirmatory_eligible']
    f=read(p,cols);assert set(f.split)=={'train','validation'}
    z,good,flow=masks(f,'reduced_lvef_strict',MARKER,'qrs_duration_ms');flow['folds']={}
    for fold in ['train','validation']:
        a=good&z.split.eq(fold);ff=z.loc[a];x,names=raw_design(ff,MARKER);assert names==entry['scaling_B3']['names']
        design=fixed_design(x,entry['scaling_B3']);y=classify_lvef(ff.preferred_lvef)
        np.savez(OUT/(fold+'_private.npz'),z=design,y=y,ids=ff.subject_id.to_numpy(),study=ff.study_id.to_numpy())
        flow['folds'][fold]={'n':len(ff),'cases':int(y.sum()),'controls':int(len(y)-y.sum()),'subjects':int(ff.subject_id.nunique())}
    pair=read(str(data_path("b.messaudit_repeatability_paargruppen_privat_parquet")),KEY+['hours_between']);assert set(pair.split)<={'train','validation'}
    schemas=json.loads((OUT/'schemas.json').read_text());source=next(p for p,v in schemas.items() if p.endswith('.parquet') and ANGLE+'_first' in v['columns'])
    v=read(source,KEY+[ANGLE+'_first',ANGLE+'_second']);assert set(v.split)<={'train','validation'}
    pair=pair.merge(v,on=KEY,validate='one_to_one');pair=pair.loc[np.isfinite(pair[ANGLE+'_first'])&np.isfinite(pair[ANGLE+'_second'])]
    pair.to_parquet(OUT/'pairs_private.parquet',index=False)
    dump(OUT/'population.json',flow);dump(OUT/'entry.json',entry)
    print('PREPARED',flow['folds'])

def pair_estimate(x,y,w):
    n=w.sum();sx=w@x;sy=w@y;sxx=w@(x*x);syy=w@(y*y);sxy=w@(x*y)
    if n<30:raise ValueError('pair_n<30')
    col=(sx-sy)**2/(2*n);msr=((sxx+2*sxy+syy)/2-(sx+sy)**2/(2*n))/(n-1)
    mse=((sxx+syy-2*sxy)/2-col)/(n-1);den=msr+mse+2*(col-mse)/n
    error=(sxx+syy-2*sxy-(sy-sx)**2/n)/(n-1)/2
    if error<=0:raise ValueError('sigma_e²<=0')
    return {'icc':float((msr-mse)/den),'bias':float((sy-sx)/n),'error_variance_deg2':float(error),'n_pairs':int(n)}

def regression_calibration(z,y,w,error,beta=None,omit_marker=False):
    ai=z.shape[1]-2;zi=[i for i in range(z.shape[1]) if i!=ai and not (omit_marker and i==z.shape[1]-1)]
    cov=z[:,zi];angle=z[:,ai];coef=np.linalg.solve(cov.T@(w[:,None]*cov),cov.T@(w*angle))
    mean=cov@coef;residual=angle-mean;var=w@(residual**2)/w.sum();reliability=1-error/var
    if reliability<=0:raise ValueError('conditional_reliability<=0')
    cal=mean+reliability*residual
    design=z.copy();design[:,ai]=cal
    fitted=logistic_fit(design,y,w)
    return {'beta':float(fitted[-1]),'angle_beta':float(fitted[ai]),'conditional_reliability':float(reliability),'residual_variance_scaled':float(var),'outside_0_180':None},design

def simex(z,y,w,error,rng,beta=None):
    ai=z.shape[1]-2;beta=logistic_fit(z,y,w) if beta is None else beta
    curve=[beta.copy()];se=[];outside=[]
    noise=rng.normal(size=(10,len(y)))
    design=z.copy()
    for lam in [.5,1.,1.5,2.]:
        fits=[];o=[]
        for e in noise:
            for sign in [-1,1]:
                design[:,ai]=z[:,ai]+sign*np.sqrt(lam*error)*e
                fits.append(logistic_fit(design,y,w,initial=beta))
                # converted to physical angle in report using fixed scaling
                o.append(float(np.mean((design[:,ai]*46.84175177563957+84.5379612752467<0)|(design[:,ai]*46.84175177563957+84.5379612752467>180))))
        fits=np.asarray(fits);curve.append(fits.mean(0));se.append(float(fits[:,-1].std(ddof=1)/np.sqrt(20)));outside.append(float(np.mean(o)))
    curve=np.asarray(curve);lam=np.array([0,.5,1,1.5,2.]);quad=np.array([1,-1,1])@np.linalg.lstsq(np.column_stack([np.ones(5),lam,lam**2]),curve,rcond=None)[0]
    linear=np.array([1,-1])@np.linalg.lstsq(np.column_stack([np.ones(5),lam]),curve,rcond=None)[0]
    return {'beta':float(quad[-1]),'angle_beta':float(quad[ai]),'linear_beta':float(linear[-1]),'curve_beta':curve[:,-1].tolist(),'curve_angle_beta':curve[:,ai].tolist(),'mc_se_at_lambda':se,'simulated_outside_fraction':outside}

def data():
    entry=json.loads(bind(OUT/'entry.json').read_text());sd=entry['scaling_B3']['sd'][-2]
    pair=read(OUT/'pairs_private.parquet',KEY+['hours_between',ANGLE+'_first',ANGLE+'_second'])
    return entry,sd,pair
def points():
    entry,sd,pair=data();results=[]
    for group in ['gesamt','zeitgleich']:
        pp=pair if group=='gesamt' else pair.loc[pair.hours_between.eq(0)]
        pe=pair_estimate(pp[ANGLE+'_first'].to_numpy(),pp[ANGLE+'_second'].to_numpy(),np.ones(len(pp)))
        pe['n_subjects']=int(pp.subject_id.nunique())
        for fold in ['train','validation']:
            a=np.load(bind(OUT/(fold+'_private.npz')));z=a['z'];y=a['y'];ids=a['ids'];w=np.ones(len(y))
            for control in [False,True]:
                zz=z.copy()
                if control:zz[:,-2]=np.random.default_rng(SEED).permutation(z[:,-2])
                b=logistic_fit(zz,y,w);base=fit(zz,y,ids)
                row={'group':group,'fold':fold,'control':control,'pair':pe,'uncorrected':base,'angle_beta_uncorrected':float(b[-2])}
                try:
                    rc,dd=regression_calibration(zz,y,w,pe['error_variance_deg2']/sd**2)
                    rc['outside_0_180']=float(np.mean((dd[:,-2]*sd+entry['scaling_B3']['mean'][-2]<0)|(dd[:,-2]*sd+entry['scaling_B3']['mean'][-2]>180)))
                    row['rc']=rc
                except ValueError as exc:row['rc']={'status':'nicht_schaetzbar','reason':str(exc)}
                row['simex']=simex(zz,y,w,pe['error_variance_deg2']/sd**2,np.random.default_rng(SEED),b)
                # Nuisance-only RC is an invariance control, if feasible.
                try:
                    trivial,_=regression_calibration(zz,y,w,pe['error_variance_deg2']/sd**2,omit_marker=True)
                    row['nuisance_only_beta_difference']=trivial['beta']-float(b[-1])
                except ValueError as exc:row['nuisance_only_reason']=str(exc)
                results.append(row);dump(OUT/'points.json',results);print('POINT',group,fold,control,'sec',time.time(),flush=True)
    print('POINTS_COMPLETE')

def bootstrap():
    entry,sd,pair=data();points=json.loads(bind(OUT/'points.json').read_text())
    for fold in ['train','validation']:
        a=np.load(bind(OUT/(fold+'_private.npz')));z=a['z'];y=a['y'];ids=a['ids'];union=np.union1d(ids,pair.subject_id.to_numpy());oi=np.searchsorted(union,ids);pi=np.searchsorted(union,pair.subject_id)
        controls=[z.copy(),z.copy()];controls[1][:,-2]=np.random.default_rng(SEED).permutation(z[:,-2])
        path=OUT/(fold+'_bootstrap.json');seal=sha(__file__)+sha(OUT/'protokoll_hash.json')+sha(OUT/(fold+'_private.npz'))+sha(OUT/'pairs_private.parquet')
        if path.exists():
            state=json.loads(path.read_text());assert state['seal']==seal
        else:state={'seal':seal,'next_r':0,'rows':[],'failures':[]}
        for r in range(state['next_r'],B):
            if r%5==0:guard()
            start=time.time();rng=np.random.default_rng(np.random.SeedSequence([SEED,r,1 if fold=='train' else 2]));multiplicity=np.bincount(rng.integers(0,len(union),len(union)),minlength=len(union));ow=multiplicity[oi].astype(float);pw=multiplicity[pi].astype(float)
            active=ow>0;yy=y[active];ww=ow[active]
            for group in ['gesamt','zeitgleich']:
                select=np.ones(len(pair),bool) if group=='gesamt' else pair.hours_between.eq(0).to_numpy()
                try:pe=pair_estimate(pair.loc[select,ANGLE+'_first'].to_numpy(),pair.loc[select,ANGLE+'_second'].to_numpy(),pw[select])
                except Exception as exc:
                    state['failures'].append({'r':r,'group':group,'phase':'pair','reason':str(exc)});continue
                for control,zz in enumerate(controls):
                    row={'r':r,'fold':fold,'group':group,'control':bool(control),'pair':pe}
                    design=zz[active]
                    try:
                        b=logistic_fit(design,yy,ww);row['original_beta']=float(b[-1])
                        try:row['rc']=regression_calibration(design,yy,ww,pe['error_variance_deg2']/sd**2)[0]
                        except Exception as exc:row['rc']={'reason':str(exc)}
                        row['simex']=simex(design,yy,ww,pe['error_variance_deg2']/sd**2,np.random.default_rng(np.random.SeedSequence([SEED,r,10+control])),b)
                    except Exception as exc:row['reason']=str(exc)
                    state['rows'].append(row)
            state['next_r']=r+1;dump(path,state);print('BOOT',fold,r+1,'seconds',round(time.time()-start,2),flush=True)
    print('BOOTSTRAP_COMPLETE')
if __name__=='__main__':
    globals()[phase]();finish();print('COMPLETE',phase)
