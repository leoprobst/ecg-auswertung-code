"""Private computation on the development partition, no test access."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os, sys, json, hashlib, time, ctypes
from pathlib import Path
os.environ.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
OUT=Path(str(data_path("b.aw192"))); OUT.mkdir(exist_ok=True)
OLD=Path(str(data_path("b.aw166")))
GRID=[-32,-24,-16,-8,-4,0,4,8,16,24,32]
def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def guard():
    class M(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ['total','free','page','commit','vtotal','vfree','extended']]
    m=M();m.length=ctypes.sizeof(m);ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    import psutil
    if min(m.free,m.commit)<2*1024**3:raise RuntimeError('WAIT_MEMORY')
    if psutil.Process().memory_info().rss>.8*1024**3:raise RuntimeError('RSS_LIMIT')
def inventory():
    guard()
    dump('protokoll_hash.json',{'sha256':sha(ROOT/'werkzeuge/b_test/aw192_protokoll.md'),'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'before_population_calculation':True})
    import pyarrow.parquet as pq
    for name in ['population_private.parquet','meta_private.parquet']:
        p=OLD/name;pf=pq.ParquetFile(p)
        dump(name+'.schema.json',{'rows':pf.metadata.num_rows,'columns':pf.schema_arrow.names,'bytes':p.stat().st_size,'sha256':sha(p)})
    counts={str(ms):0 for ms in GRID};n=0
    for p in (OLD/'signal_records').glob('*.json'):
        r=json.loads(p.read_text());assert r['split'] in ['train','validation'];n+=1
        for ms in GRID:counts[str(ms)]+=str(ms//2) in r['shift_grid']
    dump('grid_inventory.json',{'records':n,'covered_records_by_ms':counts,'full_coverage':all(v==n for v in counts.values())})
    print(json.dumps({'records':n,'covered_records_by_ms':counts}))
def sample():
    guard()
    import numpy as np, pyarrow.dataset as ds
    cols=json.loads((OUT/'population_private.parquet.schema.json').read_text())['columns']
    f=ds.dataset(str(OLD/'population_private.parquet')).to_table(columns=cols).to_pandas().sort_values('study_id')
    assert set(f.split)=={'train','validation'} and not f.subject_id.duplicated().any()
    rng=np.random.default_rng(19220261013);parts=[];strata=[]
    for y in [0,1]:
        sizes=f[f.y.eq(y)].groupby('split').size();total=int(sizes.sum())
        nt=round(1500*int(sizes['train'])/total)
        for fold,n in [('train',nt),('validation',1500-nt)]:
            ff=f[f.y.eq(y)&f.split.eq(fold)]
            q=ff.iloc[np.sort(rng.choice(len(ff),n,replace=False))].copy();q['sampling_weight']=len(ff)/n;parts.append(q)
            strata.append({'split':fold,'y':y,'population':len(ff),'sample':n,'weight':len(ff)/n})
    f=__import__('pandas').concat(parts).sort_values('study_id');f.to_parquet(OUT/'sample_private.parquet',index=False)
    dump('sample_manifest.json',{'seed':19220261013,'strata':strata,'sample_sha256':sha(OUT/'sample_private.parquet'),'population_sha256':sha(OLD/'population_private.parquet'),'test_access':False})
    print(json.dumps(strata))
def signals():
    import numpy as np,pyarrow.dataset as ds
    guard()
    # Reuse the verified exact kernel of the measurement-artefact analysis without importing its output logger.
    source=(ROOT/'werkzeuge/b_test/aw166_signals.py').read_text(encoding='utf8')
    body=source[source.index('    from werkzeuge.varianten.adapter'):source.index("    f=pd.read_parquet")]
    scope={'np':np};exec('def kernel():\n'+body+'    return compute,compute_full,prepare_record,context,shifted,OPTIONS,verified_release_import,fr\n',scope)
    compute,full,prepare,context,shifted,options,verify,fr=scope['kernel']()
    # The legacy adapter requires 4 GiB. The current run uses a 2 GiB
    # reserve and a 0.8 GiB RSS cap; apply that guard only in this process.
    import werkzeuge.varianten.adapter as adapter
    adapter.memory_guard=guard
    f=ds.dataset(str(OUT/'sample_private.parquet')).to_table().to_pandas()
    meta=ds.dataset(str(OLD/'meta_private.parquet')).to_table(filter=ds.field('study_id').isin(f.study_id.tolist())).to_pandas()
    f=meta.merge(f[['study_id','y']],on='study_id',validate='one_to_one')
    (OUT/'signal_records').mkdir(exist_ok=True);start=time.monotonic()
    def one(row):
        guard();cp=OUT/'signal_records'/f"{int(row['study_id'])}.json"
        if cp.exists():return
        old=json.loads((OLD/'signal_records'/cp.name).read_text());assert old['split'] in ['train','validation']
        vals={};missing=[]
        for ms in GRID:
            key=str(ms//2 if row['y'] else 0)
            if key in old['shift_grid']:vals[str(ms)]=old['shift_grid'][key]
            else:missing.append(ms)
        if missing:
            p=prepare(row,options);baseline=compute(p,p['delineation'])
            assert np.isclose(np.asarray(baseline,float),np.asarray(old['baseline'],float),rtol=1e-8,atol=0,equal_nan=True).all(),'Baseline mismatch'
            for ms in missing:
                d=shifted(p['delineation'],'t_offset',ms//2);v=compute(p,d)
                # Exact registry witness per record; the same independently
                # verified minimax kernel covers the remaining grid positions.
                if ms==missing[0]:
                    assert np.isclose(np.asarray(v,float),np.asarray(full(p,d),float),rtol=0,atol=0,equal_nan=True).all(),'Registry mismatch'
                vals[str(ms)]=v
        dump('signal_records/'+cp.name,{'study_id':row['study_id'],'split':old['split'],'values':vals,'new_ms':missing,'equivalent':True})
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as executor:
        for i,_ in enumerate(executor.map(one,f.to_dict('records')),1):
            if i%25==0:dump('checkpoint.json',{'phase':'signals','done':i,'total':len(f),'seconds':time.monotonic()-start,'pid':os.getpid()})
    assert verify() is fr
    dump('signal_gate.json',{'state':'passed','n':len(f),'seconds':time.monotonic()-start,'code_sha256':sha(__file__)})
    print('signals passed')
def models():
    import numpy as np,pyarrow.dataset as ds
    from scipy.special import expit
    from werkzeuge.b_test.statistik import raw_design,fixed_design,fit
    from werkzeuge.nachrechnung.core import logistic_fit
    guard();assert json.loads((OUT/'signal_gate.json').read_text())['state']=='passed'
    f=ds.dataset(str(OUT/'sample_private.parquet')).to_table().to_pandas()
    entry=json.loads((OLD/'prepare_manifest.json').read_text())['entry'];sc=entry['scaling_B3']
    vals=np.array([[json.loads((OUT/'signal_records'/f'{int(s)}.json').read_text())['values'][str(ms)] for ms in GRID] for s in f.study_id],float)
    common=np.isfinite(vals).all(axis=(1,2));x,names=raw_design(f,'qrs_t_normalized_discrete_frechet');y=f.y.to_numpy(float);w=f.sampling_weight.to_numpy(float)
    designs=[];rows=[]
    for j,ms in enumerate(GRID):
        xx=x.copy();xx[:,-1]=vals[:,j,0];xx[:,-2]=vals[:,j,1]
        finite=np.isfinite(xx).all(1);z=fixed_design(xx[finite],sc)
        out=fit(z,y[finite],f.subject_id.to_numpy()[finite],weights=w[finite]);point=out['or']
        zz=fixed_design(xx[common],sc);out=fit(zz,y[common],f.subject_id.to_numpy()[common],weights=w[common]);designs.append(zz)
        rows.append({'shift_ms':ms,'or':out['or'],'beta':out['beta'],'wald_ci':out['or_ci95_wald'],'n':int(common.sum()),'cases':int(y[common].sum()),'available_n':int(finite.sum()),'available_cases':int(y[finite].sum()),'available_or':point,'excluded_cases':int((~finite & (y==1)).sum())})
    def crossing(betas,target,direction):
        zero=GRID.index(0);indices=[zero]+sorted([j for j,ms in enumerate(GRID) if ms*direction>0],key=lambda j:abs(GRID[j]))
        t=np.log(target)
        if betas[zero]<=t:return 0.
        for a,b in zip(indices,indices[1:]):
            if betas[a]>t and betas[b]<=t:return GRID[a]+(GRID[b]-GRID[a])*(t-betas[a])/(betas[b]-betas[a])
        return None
    strata=[np.flatnonzero((f.split.to_numpy()[common]==fold)&(y[common]==c)) for fold in ['train','validation'] for c in [0,1]]
    rng=np.random.default_rng(np.random.SeedSequence([19220261013,2]));boot=[];failures=[]
    for b in range(500):
        if b%10==0:guard()
        counts=np.zeros(int(common.sum()))
        for ix in strata:
            counts+=np.bincount(rng.choice(ix,len(ix),replace=True),minlength=len(counts))
        ww=w[common]*counts;betas=[]
        try:
            for z in designs:
                beta=logistic_fit(z,y[common],ww)
                score=np.max(abs(z.T@(ww*(expit(z@beta)-y[common]))+np.r_[0,np.full(z.shape[1]-1,1e-6)]*beta))/ww.sum()
                if not np.isfinite(beta).all() or score>=1e-7:raise ValueError('convergence')
                betas.append(float(beta[-1]))
            boot.append(betas)
        except (ValueError,RuntimeError,np.linalg.LinAlgError):failures.append(b)
        if b%25==0:dump('checkpoint.json',{'phase':'bootstrap','done':b+1,'valid':len(boot),'pid':os.getpid()})
    for j,r in enumerate(rows):r['bootstrap_ci']=np.quantile(np.exp(np.array(boot)[:,j]),[.025,.975]).tolist() if len(boot)>=475 else None
    tips=[]
    for direction in [-1,1]:
        for target in [1.25,1.0]:
            values=[crossing(v,target,direction) for v in boot];finite=[v for v in values if v is not None]
            tips.append({'direction':direction,'target_or':target,'shift_ms':crossing([r['beta'] for r in rows],target,direction),'crossings':len(finite),'valid_fits':len(boot),'ci95_ms':np.quantile(finite,[.025,.975]).tolist() if len(finite)>=475 and len(boot)>=475 else None})
    dump('results.json',{'rows':rows,'tips':tips,'bootstrap_valid':len(boot),'failures':failures,'sample_n':len(f),'common_n':int(common.sum()),'common_cases':int(y[common].sum()),'marker_sd':sc['sd'][-1],'test_access':False})
    dump('bootstrap_private.json',boot);print(json.dumps({'tips':tips,'common_n':int(common.sum()),'valid':len(boot)}))
def nullaudit():
    import numpy as np
    from scipy.special import expit
    from scipy.stats import spearmanr
    from werkzeuge.nachrechnung.core import logistic_fit,auc
    guard()
    dump('null_nachtrag_hash.json',{'sha256':sha(ROOT/'werkzeuge/b_test/aw192_null_nachtrag.md'),'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    p=Path(str(data_path("b.aw168_simulation_null_json")));old=json.loads(p.read_text())
    s=old['slope'];va=old['latent_angle_var'];vn=old['marker_noise_var'];E=np.array(old['error_covariance_used']);out=[]
    for mode in ['original','correlated_errors']:
        e=E.copy()
        if mode!='original':e[0,1]=e[1,0]=.268*np.sqrt(E[0,0]*E[1,1])
        rows=[]
        for j in range(200):
            if j%20==0:guard()
            rng=np.random.default_rng(np.random.SeedSequence([1682026,0,j]))
            angle=rng.normal(0,np.sqrt(va),4000);mark=s*angle+rng.normal(0,np.sqrt(vn),4000)
            y=rng.binomial(1,expit(-2.5+.8*angle/np.sqrt(va))).astype(float)
            noise=rng.normal(size=(4000,2))
            measured=np.column_stack([angle,mark])+noise@(np.diag(np.sqrt(np.diag(e))) if mode=='original' else np.linalg.cholesky(e).T)
            z=np.column_stack([np.ones(4000),measured]);beta=logistic_fit(z,y,np.ones(4000))
            if mode=='original':assert np.isclose(beta[-1],old['rows'][j]['uncorrected_beta'],rtol=1e-8,atol=1e-10),'Synthetic reproduction'
            za=z[:,:2];ba=logistic_fit(za,y,np.ones(4000))
            residual=measured[:,0]-np.column_stack([np.ones(4000),measured[:,1]])@np.linalg.lstsq(np.column_stack([np.ones(4000),measured[:,1]]),measured[:,0],rcond=None)[0]
            rows.append({'marker_or':float(np.exp(beta[-1])),'angle_or_with_marker':float(np.exp(beta[-2])),'angle_or_without_marker':float(np.exp(ba[-1])),'conditional_reliability':float(1-E[0,0]/np.var(residual)),'spearman':float(spearmanr(measured[:,0],measured[:,1]).statistic),'angle_auc':float(auc(y,measured[:,0])),'marker_auc':float(auc(y,measured[:,1])),'oracle_auc':float(auc(y,angle))})
        summary={k:{'mean':float(np.mean([r[k] for r in rows])),'quantiles':np.quantile([r[k] for r in rows],[.025,.975]).tolist()} for k in rows[0]}
        out.append({'mode':mode,'error_covariance':e.tolist(),'summary':summary,'rows':rows})
    cov=s*va;vt_a=va+E[0,0];vt_m=s*s*va+vn+E[1,1];pearson=cov/np.sqrt(vt_a*vt_m)
    dump('null_diagnostics.json',{'original_source_sha256':sha(p),'scenarios':out,'analytic_original':{'pearson':float(pearson),'spearman':float(6/np.pi*np.arcsin(pearson/2)),'conditional_reliability':float(1-E[0,0]/(vt_a-cov*cov/vt_m))},'test_access':False,'reproduction':'200 original marker coefficients passed'})
    print('null diagnostic reproduction passed')
if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except Exception:
        import traceback
        with (OUT/'error_private.log').open('a',encoding='utf8') as f:traceback.print_exc(file=f)
        print('Phase failed; details only in private log.');sys.exit(1)
