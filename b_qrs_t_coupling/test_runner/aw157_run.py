"""AW157: isolierte Entwicklungsanalyse; keine Testquelle zugelassen."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os,sys,ctypes,time,json,hashlib,threading
from pathlib import Path
os.environ.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=Path(str(data_path("b.aw157")));OUT.mkdir(exist_ok=True)
PHASE=sys.argv[1]
sys.stdout=(OUT/(PHASE+'_stdout.log')).open('a',encoding='utf8',buffering=1)
sys.stderr=(OUT/(PHASE+'_stderr.log')).open('a',encoding='utf8',buffering=1)
class Mem(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ['total','free','page','commit','vtotal','vfree','extended']]
def memory():
    m=Mem();m.length=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m));return {'ram_free':m.free,'commit_free':m.commit}
def guard():
    while min(memory().values())<1.5*1024**3:print('WAIT_MEMORY',flush=True);time.sleep(30)
guard()
import numpy as np,pandas as pd,pyarrow.parquet as pq,pyarrow.dataset as ds,psutil
from scipy.special import expit
from werkzeuge.b_test.daten import masks,classify_lvef
from werkzeuge.b_test.statistik import raw_design,fixed_design,separation
from werkzeuge.nachrechnung.core import logistic_fit
PEAK=0;PEAK_COMMIT=0;INPUTS={};SEED=1572026
PARAM=ROOT/'werkzeuge/b_test/parameter_entwicklung.json'
PAIR=Path(str(data_path("b.messaudit_repeatability_paargruppen_privat_parquet")))
REPEAT=Path(str(data_path("b.mimic_validation_runs_a155_2026_10_04_d6e70e_repeatability_replicate_pairs_part_000_parquet")))
KEY=['subject_id','split','first_study_id','second_study_id'];A='spatial_qrs_t_angle_deg';F='qrs_t_normalized_discrete_frechet'
def watch():
    global PEAK,PEAK_COMMIT
    while True:
        m=psutil.Process().memory_info();PEAK=max(PEAK,m.rss);PEAK_COMMIT=max(PEAK_COMMIT,m.private)
        if max(PEAK,PEAK_COMMIT)>1.5*1024**3:os._exit(75)
        time.sleep(.2)
threading.Thread(target=watch,daemon=True).start()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str,allow_nan=False),encoding='utf8')
def bind(p):
    p=Path(p);assert 'split=test' not in str(p) and 'sealed' not in p.parts
    INPUTS[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size};return p
def read(p,cols):
    guard();p=bind(p);pf=pq.ParquetFile(p);assert set(cols)<=set(pf.schema_arrow.names)
    # Explicit development split predicate whenever split exists.
    flt=ds.field('split').isin(['train','validation']) if 'split' in pf.schema_arrow.names else None
    return ds.dataset(str(p),format='parquet').to_table(columns=cols,filter=flt).to_pandas()
def finish():
    for p in [__file__,ROOT/'werkzeuge/b_test/aw157_models.py',ROOT/'werkzeuge/b_test/aw157_protokoll.md',ROOT/'werkzeuge/b_test/aw157_nachtrag1.md',ROOT/'werkzeuge/b_test/daten.py',ROOT/'werkzeuge/b_test/statistik.py',ROOT/'werkzeuge/nachrechnung/core.py']:bind(p)
    dump(OUT/(PHASE+'_manifest.json'),{'inputs':INPUTS,'peak_rss':PEAK,'peak_private_commit':PEAK_COMMIT,'memory':memory(),'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'test_access':False})
def inventory():
    assert sha(ROOT/'werkzeuge/b_test/aw157_protokoll.md')=='2d1056ecf65add8e9078b87220b167c8b817247652fb1dbfc619b547bdb67bcc'
    stamp={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'before_new_data':True,'hashes':{str(p):sha(p) for p in [ROOT/'werkzeuge/b_test/aw157_protokoll.md',ROOT/'werkzeuge/b_test/aw157_nachtrag1.md']}}
    if not (OUT/'protokoll_hash.json').exists():dump(OUT/'protokoll_hash.json',stamp)
    e=json.loads(bind(PARAM).read_text(encoding='utf8'))['hypotheses']['BEF-0063']
    paths=[Path(e['input_development']['path']),PAIR,REPEAT]
    schemas={str(p):{'columns':pq.ParquetFile(p).schema_arrow.names,'rows':pq.ParquetFile(p).metadata.num_rows,'bytes':p.stat().st_size} for p in paths}
    dump(OUT/'schemas.json',schemas);print(json.dumps(schemas,ensure_ascii=False))

def prepare():
    assert (OUT/'protokoll_hash.json').exists()
    entry=json.loads(bind(PARAM).read_text(encoding='utf8'))['hypotheses']['BEF-0063']
    source=Path(entry['input_development']['path']);assert sha(source)==entry['input_development']['sha256']
    cols=['subject_id','study_id','split','age','sex','heart_rate','qrs_duration_ms','qtc_fridericia_ms',A,F,'preferred_lvef','lvef_conflict_gt_10','nearest_tte_measurement_id','nearest_tte_delta_hours','label_state','eligibility_state','index_selection_strategy','time_precision','time_delta_hours','evidence_ids_json','ehr_time_alignment_confirmatory_eligible','sinus_selector','current_report_af_state','classical_lbbb_pattern_positive','classical_rbbb_pattern_positive']
    frame=read(source,cols);assert set(frame.split)=={'train','validation'}
    z,good,flow=masks(frame,'reduced_lvef_strict',F,'qrs_duration_ms');ff=z.loc[good].copy()
    l=ff.classical_lbbb_pattern_positive;r=ff.classical_rbbb_pattern_positive
    flags={'gesamt':np.ones(len(ff),bool),'Sinus':ff.sinus_selector.eq(True).fillna(False).to_numpy(),
           'schmal_ohne_Block':(ff.qrs_duration_ms.lt(120)&l.eq(False).fillna(False)&r.eq(False).fillna(False)).to_numpy(),
           'LSB':l.eq(True).fillna(False).to_numpy(), 'AF':ff.current_report_af_state.eq('positive').fillna(False).to_numpy()}
    # Store only aggregate value dictionaries for definitions diagnostics.
    diag={c:ff[c].value_counts(dropna=False).astype(int).to_dict() for c in cols[-4:]}
    diag={c:{str(k):v for k,v in d.items()} for c,d in diag.items()}
    flow['folds']={};flow['flag_values']=diag
    raw,names=raw_design(ff,F);assert names==entry['scaling_B3']['names']
    design=fixed_design(raw,entry['scaling_B3']);y=classify_lvef(ff.preferred_lvef)
    for fold in ['train','validation']:
        ix=ff.split.eq(fold).to_numpy();np.savez(OUT/(fold+'_private.npz'),z=design[ix],y=y[ix],ids=ff.subject_id.to_numpy()[ix],**{k:v[ix] for k,v in flags.items()})
        flow['folds'][fold]={k:{'n':int((ix&v).sum()),'cases':int(y[ix&v].sum()),'controls':int((ix&v).sum()-y[ix&v].sum())} for k,v in flags.items()}
    assert not np.intersect1d(ff.loc[ff.split.eq('train'),'subject_id'],ff.loc[ff.split.eq('validation'),'subject_id']).size
    group=read(PAIR,KEY+['hours_between','rhythm','block','qrs_class'])
    angle=read(REPEAT,KEY+[A+'_first',A+'_second'])
    parts=list(REPEAT.parent.glob('*.parquet'));fs=[p for p in parts if F+'_first' in pq.ParquetFile(p).schema_arrow.names]
    assert len(fs)==1
    frechet=read(fs[0],KEY+[F+'_first',F+'_second'])
    pair=group.merge(angle,on=KEY,validate='one_to_one').merge(frechet,on=KEY,validate='one_to_one')
    complete=np.isfinite(pair[[A+'_first',A+'_second',F+'_first',F+'_second']].to_numpy(float)).all(1)
    pair=pair.loc[complete].copy();assert (pair[[F+'_first',F+'_second']].to_numpy()>=0).all()
    pair.to_parquet(OUT/'pairs_private.parquet',index=False)
    flow['pairs']={'total_before_complete':len(group),'complete':len(pair),'subjects':int(pair.subject_id.nunique()),'F_source':str(fs[0]),'group_values':{c:pair[c].value_counts().astype(int).to_dict() for c in ['rhythm','block','qrs_class']}}
    train=ff.loc[ff.split.eq('train')];knots=np.quantile(train[A],[.05,.35,.65,.95]);assert (np.diff(knots)>0).all()
    entry['aw157_knots_deg']=knots.tolist()
    basis=rcs(train[A].to_numpy(),knots);entry['aw157_spline_mean']=basis.mean(0).tolist();entry['aw157_spline_sd']=basis.std(0).tolist()
    dump(OUT/'entry.json',entry);dump(OUT/'population.json',flow);print(json.dumps(flow,ensure_ascii=False));print('KNOTS',knots)

def rcs(x,knots):
    x=np.asarray(x,float);t=np.asarray(knots,float);den=(t[-1]-t[0])**2
    h=[x]
    for tj in t[:-2]:
        h.append((np.maximum(x-tj,0)**3-(t[-1]-tj)/(t[-1]-t[-2])*np.maximum(x-t[-2],0)**3+(t[-2]-tj)/(t[-1]-t[-2])*np.maximum(x-t[-1],0)**3)/den)
    return np.column_stack(h)

def pair_groups(pair,folds=True):
    d={'gesamt':np.ones(len(pair),bool),'zeitgleich':pair.hours_between.eq(0).to_numpy(),'Sinus':pair.rhythm.eq('Sinus').to_numpy(),'AF':pair.rhythm.eq('AF').to_numpy(),'LSB':pair.block.eq('LBBB').to_numpy(),'RSB':pair.block.eq('RBBB').to_numpy(),'QRS_lt120':pair.qrs_class.eq('schmal').to_numpy()}
    if folds:
        d.update({fold:pair.split.eq(fold).to_numpy() for fold in ['train','validation']})
    return d

def weighted_cov(x,w,ddof=1):
    n=w.sum()
    if n<=ddof:raise ValueError('n<=ddof')
    center=x-np.average(x,axis=0,weights=w);return center.T@(w[:,None]*center)/(n-ddof)
def wcor(x,w):
    cov=weighted_cov(x,w);s=np.sqrt(np.diag(cov));
    if (s<=0).any():raise ValueError('zero_variance')
    return cov/np.outer(s,s)
def wrank(x,w):
    order=np.argsort(x,kind='stable');v=x[order];ww=w[order];starts=np.r_[0,np.flatnonzero(np.diff(v)!=0)+1];total=np.add.reduceat(ww,starts)
    rank=np.repeat(np.cumsum(total)-total/2+.5,np.diff(np.r_[starts,len(x)]));out=np.empty(len(x));out[order]=rank;return out
def latent(x,w,spearman=False):
    if np.count_nonzero(w)>1 and w.sum()>2:
        xx=np.column_stack([wrank(x[:,i],w) for i in range(4)]) if spearman else x
        c=wcor(xx,w);rf=float(c[0,1]);ra=float(c[2,3])
        if rf<=0 or ra<=0:raise ValueError('nonpositive_repeat_correlation')
        den=np.sqrt(rf*ra);a=float(c[0,3]/den);b=float(c[2,1]/den)
        return {'rF':rf,'rA':ra,'cross_F1_A2':float(c[0,3]),'cross_A1_F2':float(c[2,1]),'r12':a,'r21':b,'latent':(a+b)/2,'same_first':float(c[0,2]),'same_second':float(c[1,3])}
    raise ValueError('too_few_pairs')
def correlations():
    pair=read(OUT/'pairs_private.parquet',KEY+['hours_between','rhythm','block','qrs_class',F+'_first',F+'_second',A+'_first',A+'_second']);rows=[]
    for gi,(group,select) in enumerate(pair_groups(pair).items()):
        p=pair.loc[select];ids,iv=np.unique(p.subject_id,return_inverse=True);n=len(p)
        x=p[[F+'_first',F+'_second',A+'_first',A+'_second']].to_numpy(float);pearson=x.copy();pearson[:,:2]=np.log1p(pearson[:,:2])
        for mode in ['pearson_log1p','spearman']:
            row={'group':group,'mode':mode,'n_pairs':n,'n_subjects':len(ids),'point':{},'boot':[],'failures':{}}
            if len(ids)<30:row['reason']='subjects<30';rows.append(row);continue
            xx=pearson if mode=='pearson_log1p' else x
            try:row['point']=latent(xx,np.ones(n),mode=='spearman')
            except Exception as e:row['reason']=str(e)
            for b in range(1000):
                rng=np.random.default_rng(np.random.SeedSequence([SEED,gi,b]));w=np.bincount(rng.integers(0,len(ids),len(ids)),minlength=len(ids))[iv].astype(float)
                try:row['boot'].append(latent(xx,w,mode=='spearman'))
                except Exception as e:row['failures'][str(e)]=row['failures'].get(str(e),0)+1
            rows.append(row);dump(OUT/'correlations.json',rows);print('CORRELATION',group,mode,'valid',len(row['boot']),flush=True)
    print('CORRELATIONS_COMPLETE')

def outcome_data():
    e=json.loads(bind(OUT/'entry.json').read_text());p=read(OUT/'pairs_private.parquet',KEY+['hours_between','rhythm','block','qrs_class',A+'_first',A+'_second',F+'_first',F+'_second'])
    return e,p

def error_source(name,group,z,w,p,pw,entry):
    from aw157_models import variance,icc
    sd=entry['scaling_B3']['sd'][-2];x=p[A+'_first'].to_numpy(float);y=p[A+'_second'].to_numpy(float)
    if name.startswith('icc_'):
        rel=float(name[4:]);v=(1-rel)*variance(z[:,-2]*sd,w)
    else:
        mask=np.ones(len(p),bool)
        if name=='pairs_stratum':mask=p.rhythm.eq('Sinus').to_numpy() if group=='Sinus' else p.qrs_class.eq('schmal').to_numpy()
        delta=y[mask]-x[mask];weights=pw[mask];v=float(weighted_cov(delta[:,None],weights)[0,0]/2);rel=icc(x[mask],y[mask],weights)
    return float(v),float(rel)

def evaluate(z,y,w,p,pw,entry,flags,seed,draws):
    from aw157_models import scenarios,rc,multirc,simex,spline_design
    sd=entry['scaling_B3']['sd'][-2];rows=[];bases={}
    for group,basis,source in scenarios():
        key=group+'|'+basis+'|'+source;mask=flags[group]&(w>0);zz=z[mask];yy=y[mask];ww=w[mask]
        row={'key':key,'group':group,'basis':basis,'source':source,'n':len(yy),'cases':int(np.count_nonzero(yy))}
        try:
            if min(np.count_nonzero(yy),np.count_nonzero(1-yy))<30:raise ValueError('class_n<30')
            bk=group+'|'+basis
            if bk not in bases:
                design=spline_design(zz,zz[:,-2]*sd+entry['scaling_B3']['mean'][-2],entry,rcs) if basis=='spline' else zz
                if np.linalg.matrix_rank(design)!=design.shape[1]:raise ValueError('rank_deficient')
                b=logistic_fit(design,yy,ww);bases[bk]=b
            b=bases[bk];row['original_beta']=float(b[-1]);error,rel=error_source(source,group,zz,ww,p,pw,entry);row['error_variance_deg2']=error;row['assumed_icc']=rel
            if basis=='linear':
                try:row['rc']=rc(zz,ww,error/sd**2,b)[0]
                except Exception as ex:row['rc']={'reason':str(ex)}
            row['simex']=simex(zz,yy,ww,error,entry,rcs,seed,draws,basis=='spline',b)
        except Exception as ex:row['reason']=str(ex)
        rows.append(row)
    for group in ['gesamt','zeitgleich']:
        row={'key':'gesamt|linear|multirc_'+group,'group':'gesamt','basis':'linear','source':'multirc_'+group,'original_beta':float(bases['gesamt|linear'][-1]),'n':len(y),'cases':int(y.sum())}
        try:
            mask=np.ones(len(p),bool) if group=='gesamt' else p.hours_between.eq(0).to_numpy()
            delta=np.column_stack([p[A+'_second']-p[A+'_first'],p[F+'_second']-p[F+'_first']])[mask]
            scale=np.array(entry['scaling_B3']['sd'][-2:]);error=weighted_cov(delta/scale,pw[mask])/2
            row['multirc']=multirc(z,w,error,bases['gesamt|linear'])[0]
        except Exception as ex:row['reason']=str(ex)
        rows.append(row)
    return rows

def points():
    entry,p=outcome_data();rows=[]
    for fi,fold in enumerate(['train','validation']):
        d=np.load(bind(OUT/(fold+'_private.npz')));z=d['z'];y=d['y'];w=np.ones(len(y));flags={k:d[k] for k in ['gesamt','Sinus','schmal_ohne_Block','AF','LSB']}
        from aw157_models import scenarios,spline_design
        gates={}
        for group in flags:
            zz=z[flags[group]];yy=y[flags[group]];gates[group]={'linear':separation(zz,yy)}
        gates['gesamt']['spline']=separation(spline_design(z,z[:,-2]*entry['scaling_B3']['sd'][-2]+entry['scaling_B3']['mean'][-2],entry,rcs),y)
        assert all(v=='keine' for g in gates.values() for v in g.values())
        dump(OUT/(fold+'_gates.json'),gates)
        start=time.time();result=evaluate(z,y,w,p,np.ones(len(p)),entry,flags,np.random.SeedSequence([SEED,fi,11]),20)
        for r in result:r['fold']=fold
        rows+=result;dump(OUT/'points.json',rows);print('POINTS',fold,'seconds',round(time.time()-start,2),flush=True)
    print('POINTS_COMPLETE')

def bootstrap():
    entry,p=outcome_data()
    for fi,fold in enumerate(['train','validation']):
        d=np.load(bind(OUT/(fold+'_private.npz')));z=d['z'];y=d['y'];ids=d['ids'];flags={k:d[k] for k in ['gesamt','Sinus','schmal_ohne_Block','AF','LSB']}
        union=np.union1d(ids,p.subject_id);oi=np.searchsorted(union,ids);pi=np.searchsorted(union,p.subject_id)
        path=OUT/(fold+'_bootstrap.json');seal=sha(ROOT/'werkzeuge/b_test/aw157_models.py')+sha(__file__)+sha(OUT/(fold+'_private.npz'))+sha(OUT/'pairs_private.parquet')
        if path.exists():state=json.loads(path.read_text());assert state['seal']==seal
        else:state={'seal':seal,'next_r':0,'rows':[]}
        for r in range(state['next_r'],200):
            guard();start=time.time();rng=np.random.default_rng(np.random.SeedSequence([SEED,fi,r,19]));mult=np.bincount(rng.integers(0,len(union),len(union)),minlength=len(union));w=mult[oi].astype(float);pw=mult[pi].astype(float)
            result=evaluate(z,y,w,p,pw,entry,flags,np.random.SeedSequence([SEED,fi,r,22]),10)
            for row in result:row.update({'fold':fold,'replicate':r})
            state['rows']+=result;state['next_r']=r+1;dump(path,state);print('BOOT',fold,r+1,'seconds',round(time.time()-start,2),flush=True)
    print('BOOTSTRAP_COMPLETE')

def verify():
    from aw157_verify import verify as check
    check(sys.modules[__name__])

def report():
    from aw157_report import report as generate
    generate(sys.modules[__name__])

if __name__=='__main__':
    globals()[PHASE]();finish();print('COMPLETE',PHASE)
