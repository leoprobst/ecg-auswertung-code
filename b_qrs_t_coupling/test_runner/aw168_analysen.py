"""AW168 phasenweise Kerne; alle Detaildateien privat."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
from pathlib import Path
import json,time
import numpy as np,pandas as pd
import pyarrow.parquet as pq,pyarrow.dataset as ds
from scipy.special import expit,logit
from scipy.stats import spearmanr
from werkzeuge.nachrechnung.core import logistic_fit,auc
from werkzeuge.b_test.statistik import raw_design,fixed_design,scaling,separation
from werkzeuge.b_test.daten import masks,classify_lvef
from werkzeuge.b_test.aw157_models import icc,rc,multirc

def sources(a):
    cache=Path(str(data_path("b.mimic_validation_runs_a155_2026_10_04_d6e70e_analysis_feature_cache_cache_89d6aaf0a6b5957e")))
    paths=[cache/('split='+s)/'features.parquet' for s in ['train','validation']]
    evidence=Path(str(data_path("b.mimic_validation_evidence")))
    paths += [evidence/(n+'.parquet') for n in ['clinical_context_features','ecg_lab_features','ecg_medication_features','ecg_medication_windows','ecg_medication_source_coverage','ecg_omr_features','encounter_care_setting_flags','patient_code_history','diagnosis_evidence','echo_measurements_wide','echo_measurements_normalized','medication_evidence','medication_order_administration_context']]
    paths += [Path(str(data_path("b.mimic_validation_master_ecg_master_index_parquet")))]
    schemas={str(p):{'columns':pq.ParquetFile(p).schema_arrow.names,'rows':pq.ParquetFile(p).metadata.num_rows,'bytes':p.stat().st_size} for p in paths if p.exists()}
    a.dump(a.OUT/'source_schemas.json',schemas)
    for p,v in schemas.items():
        print(p,v['rows']);print('\n'.join(c for c in v['columns'] if ('features.parquet' not in p or any(t in c for t in ['angle','azimuth','elevation','magnitude','frechet','subject_id','study_id']))))

def rcs(x,t):
    x=np.asarray(x,float);t=np.asarray(t);h=[x]
    for tj in t[:-2]:h.append((np.maximum(x-tj,0)**3-(t[-1]-tj)/(t[-1]-t[-2])*np.maximum(x-t[-2],0)**3+(t[-2]-tj)/(t[-1]-t[-2])*np.maximum(x-t[-1],0)**3)/(t[-1]-t[0])**2)
    return np.column_stack(h)

def subset(a,p,cols,ids,studies=None):
    # Footer first; predicate excludes all records outside development identity set.
    p=a.bind(p);names=pq.ParquetFile(p).schema_arrow.names
    assert 'subject_id' in names
    flt=ds.field('subject_id').isin(ids.tolist())
    if studies is not None and 'study_id' in names:flt=flt&ds.field('study_id').isin(studies.tolist())
    if 'split' in names:flt=flt&ds.field('split').isin(['train','validation'])
    return ds.dataset(str(p),format='parquet').to_table(columns=cols,filter=flt).to_pandas()

def prepare(a):
    params=a.load(a.PARAM)['hypotheses'];schemas=a.load(a.OUT/'source_schemas.json');flows={}
    cache=Path(str(data_path("b.mimic_validation_runs_a155_2026_10_04_d6e70e_analysis_feature_cache_cache_89d6aaf0a6b5957e")))
    featurecols=pq.ParquetFile(cache/'split=train/features.parquet').schema_arrow.names
    wanted=[a.F,a.M,'spatial_peak_qrs_t_angle_deg','qrs_area_to_t_area_angle_deg','qrs_area_azimuth_deg','qrs_area_elevation_deg','t_area_azimuth_deg','t_area_elevation_deg','spatial_t_peak_magnitude']
    # Candidate catalog spellings documented; exact semantic choice fixed from release code/schema.
    wanted += [c for c in featurecols if ('azimuth' in c or 'elevation' in c) and ('qrs' in c or c.startswith('t_') or 'st_t' in c)]
    wanted += [c for c in featurecols if c in ['magnitude_area_st_t','spatial_low_voltage_t_peak_magnitude','t_peak_magnitude','t_area_vector_magnitude_mv_ms']]
    wanted=list(dict.fromkeys(c for c in wanted if c in featurecols))
    for bef,ep in [('BEF-0063','reduced_lvef_strict'),('BEF-0049','cardiomyopathy_icd')]:
        entry=params[bef];p=Path(entry['input_development']['path']);assert a.sha(p)==entry['input_development']['sha256']
        names=pq.ParquetFile(p).schema_arrow.names
        cols=['subject_id','study_id','split','age','sex','heart_rate','qrs_duration_ms','qtc_fridericia_ms',a.A,'label_state','eligibility_state','index_selection_strategy','ehr_time_alignment_confirmatory_eligible']
        cols += [a.F,'preferred_lvef','lvef_conflict_gt_10','nearest_tte_measurement_id','nearest_tte_delta_hours','time_precision','time_delta_hours','evidence_ids_json','bmi_before_ecg'] if bef=='BEF-0063' else [a.M,'label','ecg_time']
        f=a.read(p,cols)
        additions=[c for c in wanted if c not in f]
        for fold in ['train','validation']:
            ix=f.split.eq(fold);ff=f.loc[ix]
            extra=subset(a,cache/('split='+fold)/'features.parquet',['subject_id','study_id']+additions,ff.subject_id.unique(),ff.study_id.unique())
            assert not extra.duplicated(['subject_id','study_id']).any()
            merged=ff[['subject_id','study_id']].merge(extra,on=['subject_id','study_id'],how='left',validate='one_to_one')
            for c in additions:f.loc[ix,c]=merged[c].to_numpy()
        marker=a.F if bef=='BEF-0063' else a.M
        ff,good,flow=masks(f,ep,marker,'qrs_duration_ms')
        ff['eligible_B3']=good
        if bef=='BEF-0063':
            # Complete B3 mask is varied only by marker availability. Reference-only mask removes marker and B3 missingness gate.
            ref=f.copy();ref['sex']=ref.sex.fillna('F');ref['heart_rate']=ref.heart_rate.fillna(60);ref['qrs_duration_ms']=ref.qrs_duration_ms.fillna(100);ref['qtc_fridericia_ms']=ref.qtc_fridericia_ms.fillna(400);ref[a.A]=ref[a.A].fillna(90);ref[marker]=ref[marker].fillna(1)
            _,r,_=masks(ref,ep,marker,'qrs_duration_ms');ff['eligible_reference']=r
        ff.to_parquet(a.OUT/(bef+'_private.parquet'),index=False);flows[bef]=flow
        a.dump(a.OUT/(bef+'_entry.json'),entry)
    a.dump(a.OUT/'population.json',flows)
    pairpath=Path(str(data_path("b.aw157_pairs_private_parquet")))
    cols=['subject_id','split','hours_between',a.A+'_first',a.A+'_second',a.F+'_first',a.F+'_second']
    pairs=a.read(pairpath,cols);pairs.to_parquet(a.OUT/'pairs_private.parquet',index=False)
    a.dump(a.OUT/'feature_selection.json',{'columns':wanted,'cache':str(cache),'sources':schemas})
    print('PREPARED',[(k,v['sequential']) for k,v in flows.items()])

def frame(a,bef):
    p=a.OUT/(bef+'_private.parquet');return a.read(p,pq.ParquetFile(p).schema_arrow.names)

def standard_extra(x,training):
    mu=training.mean(0);sd=training.std(0);keep=sd>1e-10
    return (x[:,keep]-mu[keep])/sd[keep],{'mean':mu.tolist(),'sd':sd.tolist(),'keep':keep.tolist()}

def logistic_stats(z,y,w=None):
    w=np.ones(len(y)) if w is None else w
    active=w>0;zz=z[active];yy=y[active];ww=w[active]
    if np.linalg.matrix_rank(zz)!=zz.shape[1]:raise ValueError('rank_deficient')
    b=logistic_fit(zz,yy,ww);pr=expit(zz@b)
    score=np.max(np.abs(zz.T@(ww*(pr-yy))+np.r_[0,np.ones(zz.shape[1]-1)*1e-6]*b))/ww.sum()
    if score>=1e-7:raise ValueError('nonconverged')
    hi=np.linalg.inv(zz.T@((ww*pr*(1-pr))[:,None]*zz)+np.diag(np.r_[0,np.ones(zz.shape[1]-1)*1e-6]))
    # Frequency bootstrap weights represent duplicated persons; point CR1 below is for unweighted one-index data.
    u=zz*(ww*(yy-pr))[:,None];cov=hi@(u.T@u)@hi*len(yy)/(len(yy)-1)*(len(yy)-1)/(len(yy)-zz.shape[1])
    se=np.sqrt(np.diag(cov));return b,np.column_stack([b-1.95996398454*se,b+1.95996398454*se])

def model_boot(a,key,z,y,meta):
    import inspect,hashlib
    path=a.OUT/('model_'+key+'.json');seal=hashlib.sha256((inspect.getsource(model_boot)+inspect.getsource(models)+inspect.getsource(logistic_stats)).encode()).hexdigest()
    np.savez(a.OUT/('design_'+key+'_private.npz'),z=z,y=y)
    if path.exists():
        r=a.load(path)
        if r.get('status'):
            assert separation(z,y)!='keine' or separation(z[:,:-1],y)!='keine'
            r['earlier_seal']=r['seal'];r['seal']=seal;a.dump(path,r);return r
        if r['seal']!=seal:
            b,_=logistic_stats(z,y);b0=logistic_fit(z[:,:-1],y,np.ones(len(y)))
            assert abs(b[-1]-r['beta'])<1e-9 and abs(auc(y,z@b)-auc(y,z[:,:-1]@b0)-r['delta_auc'])<1e-9
            r['earlier_seal']=r['seal'];r['seal']=seal
    else:
        sep=separation(z,y);sep0=separation(z[:,:-1],y)
        if sep!='keine' or sep0!='keine':
            r={'key':key,'seal':seal,**meta,'n':len(y),'cases':int(y.sum()),'status':'nicht_schaetzbar','reason':'Separation '+sep+' / Basismodell '+sep0,'next':0,'boot':[],'failures':{'point_separation':1}}
            a.dump(path,r);return r
        b,ci=logistic_stats(z,y);b0=logistic_fit(z[:,:-1],y,np.ones(len(y)))
        r={'key':key,'seal':seal,**meta,'n':len(y),'cases':int(y.sum()),'beta':b[-1],'or':np.exp(b[-1]),'or_wald_ci':np.exp(ci[-1]),'auc0':auc(y,z[:,:-1]@b0),'auc1':auc(y,z@b),'boot':[],'failures':{},'next':0}
        r['delta_auc']=r['auc1']-r['auc0']
    for i in range(r['next'],200):
        if i%20==0:a.guard()
        rng=np.random.default_rng(np.random.SeedSequence([a.SEED,int(hashlib_key(key)),i]))
        w=np.bincount(rng.integers(0,len(y),len(y)),minlength=len(y)).astype(float)
        try:
            b=logistic_fit(z,y,w);b0=logistic_fit(z[:,:-1],y,w)
            r['boot'].append([float(b[-1]),auc(y,z@b,w)-auc(y,z[:,:-1]@b0,w)])
        except Exception as ex:r['failures'][type(ex).__name__+':'+str(ex)]=r['failures'].get(type(ex).__name__+':'+str(ex),0)+1
        r['next']=i+1
        if (i+1)%20==0:a.dump(path,r)
    if len(r['boot'])>=190:
        q=np.quantile(r['boot'],[.025,.975],axis=0);r['or_ci']=np.exp(q[:,0]);r['delta_auc_ci']=q[:,1]
    a.dump(path,r);print('MODEL',key,'valid',len(r['boot']),flush=True)
    return r
def hashlib_key(key):
    import hashlib
    return int(hashlib.sha256(key.encode()).hexdigest()[:8],16)

def models(a):
    out=[]
    for bef,ep in [('BEF-0063','reduced_lvef_strict'),('BEF-0049','cardiomyopathy_icd')]:
        f=frame(a,bef);entry=a.load(a.OUT/(bef+'_entry.json'))
        for marker in [a.F,a.M]:
            ff,good,flow=masks(f,ep,marker,'qrs_duration_ms');ff=ff.loc[good]
            raw,names=raw_design(ff,marker)
            scale=entry['scaling_B3'] if marker==(a.F if bef=='BEF-0063' else a.M) else scaling(raw,names)
            z=fixed_design(raw,scale)
            for fold in ['train','validation']:
                ix=ff.split.eq(fold).to_numpy();yy=classify_lvef(ff.preferred_lvef.to_numpy()[ix]) if bef=='BEF-0063' else ff.label.to_numpy(float)[ix]
                meta={'point':'W7','endpoint':ep,'marker':marker,'fold':fold,'sd':scale['sd'][-1],'flow':flow['sequential'],'source':str(a.OUT/(bef+'_private.parquet'))}
                out.append(model_boot(a,bef+'_'+('F' if marker==a.F else 'M')+'_'+fold,z[ix],yy,meta))
        if bef!='BEF-0063':continue
        ff=f.loc[f.eligible_B3].copy();raw,names=raw_design(ff,a.F);z=fixed_design(raw,entry['scaling_B3']);tr=ff.split.eq('train').to_numpy()
        knots=np.quantile(ff.loc[tr,a.A],[.05,.35,.65,.95]);basis=rcs(ff[a.A],knots);spline,sp=standard_extra(basis,basis[tr])
        angles=['spatial_peak_qrs_t_angle_deg','qrs_area_to_t_area_angle_deg']
        # Axis spelling follows source schema; no clinical substitute.
        candidates=[c for c in ff if ('azimuth' in c or 'elevation' in c) and ('qrs' in c or c.startswith('t_') or 'st_t' in c)]
        a.dump(a.OUT/'comparator_candidates.json',{'angle':angles,'axes':candidates,'columns':list(ff.columns),'spline_knots':knots})
        for variant in ['angles','axes','spline_angles']:
            if variant=='axes':
                axes=['qrs_area_vector_azimuth_deg','qrs_area_vector_elevation_deg','t_area_vector_azimuth_deg','t_area_vector_elevation_deg']
                # Actual cache names, if different, are resolved in a pre-result implementation amendment.
                cols=axes+['t_area_vector_magnitude_mv_ms']
            else:cols=angles
            if set(cols)-set(ff):out.append({'point':'W17','variant':variant,'status':'missing_columns','missing':sorted(set(cols)-set(ff))});continue
            x=ff[cols].to_numpy(float);ok=np.isfinite(x).all(1)
            if variant=='axes':x=np.column_stack([np.sin(np.deg2rad(x[:,:4])),np.cos(np.deg2rad(x[:,:4])),x[:,4]])
            extra,es=standard_extra(x[ok],x[ok&tr]);base=z[ok,:-1]
            if variant=='spline_angles':base=np.column_stack([z[ok,:-2],spline[ok]])
            zz=np.column_stack([base,extra,z[ok,-1]])
            # Preserve column span, remove only exact training linear dependencies.
            keep=[];training=zz[ff.loc[ok].split.eq('train').to_numpy()]
            for j in range(zz.shape[1]-1):
                if np.linalg.matrix_rank(training[:,keep+[j]])>len(keep):keep.append(j)
            dropped=[j for j in range(zz.shape[1]-1) if j not in keep]
            assert np.linalg.matrix_rank(training[:,keep+[zz.shape[1]-1]])==len(keep)+1
            full=zz;zz=zz[:,keep+[zz.shape[1]-1]]
            if dropped:
                coef=np.linalg.lstsq(zz[:,:-1],full[:,dropped],rcond=None)[0]
                assert np.max(np.abs(zz[:,:-1]@coef-full[:,dropped]))<1e-7
            for fold in ['train','validation']:
                ix=ff.loc[ok].split.eq(fold).to_numpy();y=classify_lvef(ff.loc[ok].preferred_lvef.to_numpy()[ix])
                out.append(model_boot(a,'W17_'+variant+'_'+fold,zz[ix],y,{'point':'W17','variant':variant,'fold':fold,'columns':cols,'scaling':es,'redundant_column_positions_removed':dropped,'source':str(a.OUT/(bef+'_private.parquet'))}))
    a.dump(a.OUT/'models.json',out)

def reliability(a):
    f=frame(a,'BEF-0063');f=f.loc[f.eligible_B3];entry=a.load(a.OUT/'BEF-0063_entry.json');raw,names=raw_design(f,a.F);z=fixed_design(raw,entry['scaling_B3'])
    p=a.read(a.OUT/'pairs_private.parquet',['subject_id','split',a.A+'_first',a.A+'_second',a.F+'_first',a.F+'_second'])
    delta=p[[a.A+'_second',a.F+'_second']].to_numpy()-p[[a.A+'_first',a.F+'_first']].to_numpy();error=np.cov(delta.T,ddof=1)/2
    anglepairs=a.read(Path(str(data_path("b.aw155_pairs_private_parquet"))),['subject_id','split',a.A+'_first',a.A+'_second'])
    angle_error=np.var(anglepairs[a.A+'_second']-anglepairs[a.A+'_first'],ddof=1)/2
    rows=[];coefs=[]
    for fold in ['train','validation']:
        ix=f.split.eq(fold).to_numpy();zz=z[ix];y=classify_lvef(f.preferred_lvef.to_numpy()[ix]);w=np.ones(len(y))
        row={'fold':fold,'n':len(y),'cases':int(y.sum()),'source':str(a.OUT/'BEF-0063_private.parquet'),'error_variance':angle_error,'angle_sd':entry['scaling_B3']['sd'][-2]}
        for mode,design,ni in [('with_marker',zz,['intercept']+names),('without_marker',zz[:,:-1],['intercept']+names[:-1])]:
            b,ci=logistic_stats(design,y)
            for j,name in enumerate(ni):coefs.append({'fold':fold,'mode':mode,'coefficient':name,'beta':b[j],'or':np.exp(b[j]),'ci':np.exp(ci[j]).tolist(),'n':len(y)})
            ai=len(b)-2 if mode=='with_marker' else len(b)-1
            others=np.delete(design,ai,axis=1);res=design[:,ai]-others@np.linalg.lstsq(others,design[:,ai],rcond=None)[0];v=res.var()
            row[mode]={'conditional_reliability':1-angle_error/entry['scaling_B3']['sd'][-2]**2/v,'angle_or':np.exp(b[ai]),'angle_ci':np.exp(ci[ai]),'vif_angle':design[:,ai].var()/v}
        row.update({'pearson':np.corrcoef(zz[:,-2:].T)[0,1],'spearman':spearmanr(zz[:,-2],zz[:,-1]).statistic,'angle_auc':auc(y,zz[:,-2]),'marker_auc':auc(y,zz[:,-1])})
        # Joint person resampling of outcome and pooled repeat source estimates uncertainty.
        union=np.union1d(np.union1d(f.subject_id.to_numpy()[ix],p.subject_id),anglepairs.subject_id);oi=np.searchsorted(union,f.subject_id.to_numpy()[ix]);pi=np.searchsorted(union,p.subject_id);api=np.searchsorted(union,anglepairs.subject_id);adelta=(anglepairs[a.A+'_second']-anglepairs[a.A+'_first']).to_numpy()
        boot=[]
        for r in range(200):
            rng=np.random.default_rng(np.random.SeedSequence([a.SEED,15,int(fold=='validation'),r]));mult=np.bincount(rng.integers(0,len(union),len(union)),minlength=len(union));ww=mult[oi];pw=mult[pi]
            apw=mult[api];ev=np.average((adelta-np.average(adelta,weights=apw))**2,weights=apw)*apw.sum()/(apw.sum()-1)/2
            rel=[]
            for has in [False,True]:
                dd=zz if has else zz[:,:-1];ai=dd.shape[1]-2 if has else dd.shape[1]-1;other=np.delete(dd,ai,axis=1)
                coef=np.linalg.solve(other.T@(ww[:,None]*other),other.T@(ww*dd[:,ai]));v=np.average((dd[:,ai]-other@coef)**2,weights=ww)
                rel.append(1-ev/entry['scaling_B3']['sd'][-2]**2/v)
            # ICC fold-specific repeat population, not Pearson repeat correlation.
            pf=p.split.eq(fold).to_numpy();ic=[icc(p[c+'_first'].to_numpy()[pf],p[c+'_second'].to_numpy()[pf],pw[pf]) for c in [a.A,a.F]]
            # Spearman with frequency weights via expanded resampling positions, development only.
            take=np.repeat(np.arange(len(yy:=y)),ww)
            boot.append(rel+ic+[spearmanr(zz[take,-2],zz[take,-1]).statistic])
        q=np.quantile(boot,[.025,.975],axis=0)
        row['without_marker']['reliability_ci']=q[:,0];row['with_marker']['reliability_ci']=q[:,1]
        pf=p.split.eq(fold).to_numpy();row['repeat_n']=int(pf.sum());row['repeat_subjects']=int(p.loc[pf].subject_id.nunique())
        row['angle_icc']=icc(p.loc[pf,a.A+'_first'].to_numpy(),p.loc[pf,a.A+'_second'].to_numpy(),np.ones(pf.sum()));row['marker_icc']=icc(p.loc[pf,a.F+'_first'].to_numpy(),p.loc[pf,a.F+'_second'].to_numpy(),np.ones(pf.sum()))
        row['angle_icc_ci']=q[:,2];row['marker_icc_ci']=q[:,3];row['spearman_ci']=q[:,4];rows.append(row)
    a.dump(a.OUT/'reliability.json',{'rows':rows,'coefficients':coefs,'error_covariance':error,'error_pairs':len(p),'error_subjects':p.subject_id.nunique(),'angle_error':angle_error,'angle_error_pairs':len(anglepairs),'angle_error_subjects':anglepairs.subject_id.nunique()})
    print('RELIABILITY_COMPLETE')

def simex_variant(z,y,w,E,mode,seed,draws,mean,sd):
    zz=z.copy();err=E.copy();angle=zz[:,-2]*sd+mean
    if mode=='logit':
        zz[:,-2]=logit(np.clip(angle,.5,179.5)/180)
    b=logistic_fit(zz,y,w);curve=[b]
    rng=np.random.default_rng(seed);noise=rng.multivariate_normal(np.zeros(2),err,size=(draws//2,len(y)))
    if mode!='joint':noise[:,:,1]=0
    outside=[]
    for lam in [.5,1,1.5,2]:
        fits=[];o=[]
        for e in noise:
            for sign in [-1,1]:
                dd=zz.copy();dd[:,-2:]+=sign*np.sqrt(lam)*e
                if mode=='clipped':dd[:,-2]=(np.clip(dd[:,-2]*sd+mean,0,180)-mean)/sd
                o.append(float(np.average((dd[:,-2]*sd+mean<0)|(dd[:,-2]*sd+mean>180),weights=w)) if mode!='logit' else 0.)
                fits.append(logistic_fit(dd,y,w,initial=b))
        curve.append(np.mean(fits,axis=0));outside.append(np.mean(o))
    t=np.array([0,.5,1,1.5,2]);curve=np.array(curve)
    quad=np.array([1,-1,1])@np.linalg.lstsq(np.column_stack([np.ones(5),t,t*t]),curve,rcond=None)[0]
    lin=np.array([1,-1])@np.linalg.lstsq(np.column_stack([np.ones(5),t]),curve,rcond=None)[0]
    return {'uncorrected_beta':float(b[-1]),'quadratic_beta':float(quad[-1]),'linear_beta':float(lin[-1]),'curve':curve[:,-1].tolist(),'outside':outside}

def simex(a):
    f=frame(a,'BEF-0063');f=f.loc[f.eligible_B3];entry=a.load(a.OUT/'BEF-0063_entry.json');raw,_=raw_design(f,a.F);z=fixed_design(raw,entry['scaling_B3']);scale=np.array(entry['scaling_B3']['sd'][-2:]);mean=entry['scaling_B3']['mean'][-2]
    p=a.read(a.OUT/'pairs_private.parquet',['subject_id','split',a.A+'_first',a.A+'_second',a.F+'_first',a.F+'_second'])
    delta=(p[[a.A+'_second',a.F+'_second']].to_numpy()-p[[a.A+'_first',a.F+'_first']].to_numpy())/scale
    transformed=logit(np.clip(p[a.A+'_second'],.5,179.5)/180)-logit(np.clip(p[a.A+'_first'],.5,179.5)/180)
    rows=[]
    for fi,fold in enumerate(['train','validation']):
        ix=f.split.eq(fold).to_numpy();zz=z[ix];y=classify_lvef(f.preferred_lvef.to_numpy()[ix]);ids=f.subject_id.to_numpy()[ix]
        union=np.union1d(ids,p.subject_id);oi=np.searchsorted(union,ids);pi=np.searchsorted(union,p.subject_id)
        for mode in ['raw','clipped','logit','joint']:
            path=a.OUT/('simex_'+fold+'_'+mode+'.json')
            def getE(pw):
                d=delta.copy()
                if mode=='logit':d[:,0]=transformed
                center=d-np.average(d,axis=0,weights=pw);return center.T@(pw[:,None]*center)/(pw.sum()-1)/2
            E=getE(np.ones(len(p)))
            if path.exists():r=a.load(path)
            else:r={'fold':fold,'mode':mode,'point':'W16b','n':len(y),'cases':int(y.sum()),'error':E,'result':simex_variant(zz,y,np.ones(len(y)),E,mode,np.random.SeedSequence([a.SEED,16,fi]),20,mean,scale[0]),'boot':[],'failures':{},'next':0}
            for j in range(r['next'],200):
                if j%10==0:a.guard()
                rng=np.random.default_rng(np.random.SeedSequence([a.SEED,161,fi,j]));mul=np.bincount(rng.integers(0,len(union),len(union)),minlength=len(union));w=mul[oi];pw=mul[pi]
                try:
                    result=simex_variant(zz,y,w,getE(pw),mode,np.random.SeedSequence([a.SEED,162,fi,j]),10,mean,scale[0]);r['boot'].append([result['uncorrected_beta'],result['linear_beta'],result['quadratic_beta']])
                except Exception as ex:r['failures'][str(ex)]=r['failures'].get(str(ex),0)+1
                r['next']=j+1
                if (j+1)%10==0:a.dump(path,r);print('SIMEX',fold,mode,j+1,flush=True)
            if len(r['boot'])>=190:
                b=np.array(r['boot']);r['or_ci']=np.exp(np.quantile(b[:,1:],[.025,.975],axis=0));r['fraction_ci']=np.quantile(b[:,1:]/b[:,0,None],[.025,.975],axis=0)
            a.dump(path,r);rows.append(r)
    a.dump(a.OUT/'simex.json',rows)

def simulation(a):
    entry=a.load(a.OUT/'BEF-0063_entry.json');reli=a.load(a.OUT/'reliability.json');E=np.array(reli['error_covariance']);sd=np.array(entry['scaling_B3']['sd'][-2:]);scaled=E/np.outer(sd,sd)
    f=frame(a,'BEF-0063');f=f.loc[f.eligible_B3];theta=(f[a.A].to_numpy()-entry['scaling_B3']['mean'][-2])/sd[0];marker=(f[a.F].to_numpy()-entry['scaling_B3']['mean'][-1])/sd[1]
    slope=np.cov(theta,marker,ddof=0)[0,1]/theta.var();latent_var=theta.var()-scaled[0,0];latent_var_m=marker.var()-scaled[1,1]
    noisevar=latent_var_m-slope*slope*latent_var
    assert latent_var>0 and noisevar>0
    rows=[]
    for scenario,effect in [('null',0.),('true',.6)]:
        path=a.OUT/('simulation_'+scenario+'.json');r=a.load(path) if path.exists() else {'scenario':scenario,'effect':effect,'n_per_run':4000,'slope':slope,'latent_angle_var':latent_var,'marker_noise_var':noisevar,'error_covariance_used':np.diag(np.diag(scaled)).tolist(),'rows':[],'next':0}
        for j in range(r['next'],200):
            if j%20==0:a.guard()
            rng=np.random.default_rng(np.random.SeedSequence([a.SEED,int(effect>0),j]));angle=rng.normal(0,np.sqrt(latent_var),4000);mark=slope*angle+rng.normal(0,np.sqrt(noisevar),4000)
            latent=np.column_stack([np.ones(4000),angle,mark]);y=rng.binomial(1,expit(-2.5+.8*angle/np.sqrt(latent_var)+effect*mark/np.sqrt(latent_var_m))).astype(float)
            measured=latent.copy();measured[:,-2:]+=rng.normal(size=(4000,2))*np.sqrt(np.diag(scaled))
            w=np.ones(4000);b=logistic_fit(measured,y,w);truth=logistic_fit(latent,y,w)
            row={'replicate':j,'known_marker_beta':effect/np.sqrt(latent_var_m),'oracle_beta':float(truth[-1]),'uncorrected_beta':float(b[-1])}
            for name in ['rc','joint_rc']:
                try:
                    rr=rc(measured,w,scaled[0,0],b)[0] if name=='rc' else multirc(measured,w,np.diag(np.diag(scaled)),b)[0]
                    row[name+'_beta']=rr.get('beta');row[name+'_reason']=rr.get('reason')
                except Exception as ex:row[name+'_reason']=str(ex)
            ss=simex_variant(measured,y,w,scaled,'raw',np.random.SeedSequence([a.SEED,163,int(effect>0),j]),10,0,1)
            row['simex_beta']=ss['quadratic_beta'];r['rows'].append(row);r['next']=j+1
            if (j+1)%20==0:a.dump(path,r);print('SIMULATION',scenario,j+1,flush=True)
        a.dump(path,r);rows.append(r)
    a.dump(a.OUT/'simulation.json',rows)

def covariates(a):
    import ast,duckdb
    note=a.ROOT/'werkzeuge/b_test/aw168_nachtrag1.md';a.dump(a.OUT/'nachtrag1_hash.json',{'sha256':a.sha(note),'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'before_covariate_computation':True})
    f=frame(a,'BEF-0063');f=f.loc[f.eligible_B3].copy();ids=f.subject_id.unique();studies=f.study_id.unique();base=Path(str(data_path("b.mimic_validation_evidence")))
    ctx=base/'clinical_context_features.parquet';cols=['subject_id','study_id','ecg_time','anchor_year_group','ed_intime','ed_outtime','admittime','dischtime','ed_match_ambiguous','admission_match_ambiguous','potassium_nearest','potassium_delta_hours','potassium_nearest_censoring','preferred_lvef_method']
    c=subset(a,ctx,cols,ids,studies);assert not c.duplicated(['subject_id','study_id']).any();f=f.merge(c,on=['subject_id','study_id'],how='left',validate='one_to_one')
    covpath=base/'ecg_medication_source_coverage.parquet';cc=subset(a,covpath,['subject_id','study_id','has_emar_coverage','has_prescription_coverage','medication_assessed'],ids,studies);f=f.merge(cc,on=['subject_id','study_id'],how='left',validate='one_to_one')
    mapping=Path(str(data_path("b.release_sources_a155_survival_nullable_20261007_vcg_biomarker_mimic_mappings_py")));tree=ast.parse(a.bind(mapping).read_text(encoding='utf8'))
    patterns=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name) and n.target.id=='MEDICATION_PATTERNS')
    classes=['antiarrhythmic_class_i_iii','digoxin','qt_prolonging_drug'];a.dump(a.OUT/'medication_inventory.json',{'patterns':{k:patterns[k] for k in classes},'source_schemas':{str(p):pq.ParquetFile(p).schema_arrow.names for p in [base/'medication_evidence.parquet',covpath]},'operational':True,'literature_validated':False})
    con=duckdb.connect();con.execute("SET memory_limit='1GB'");con.execute('SET threads=2');con.execute(("SET temp_directory='" + str(data_path("b.aw168_duckdb_tmp")) + "'"))
    con.register('idx',f[['subject_id','study_id','ecg_time']]);med=a.bind(base/'medication_evidence.parquet');assert 'event_time' in pq.ParquetFile(med).schema_arrow.names
    for window in ['around24','before7d']:
        for cl in classes:
            # Same naming as release; raw names are necessary because Release CASE classification can overlap.
            pat=' OR '.join("lower(coalesce(m.medication_raw,'')) LIKE '%"+s+"%'" for s in patterns[cl])
            if window=='around24':cond="((m.source_table='emar' AND m.event_time BETWEEN i.ecg_time-INTERVAL 24 HOUR AND i.ecg_time+INTERVAL 24 HOUR) OR (m.source_table IN ('prescriptions','pharmacy') AND m.start_time<=i.ecg_time+INTERVAL 24 HOUR AND (m.end_time IS NULL OR m.end_time>=i.ecg_time-INTERVAL 24 HOUR)))"
            else:cond="m.source_table IN ('prescriptions','pharmacy') AND m.start_time BETWEEN i.ecg_time-INTERVAL 7 DAY AND i.ecg_time"
            q=f"SELECT DISTINCT i.subject_id,i.study_id FROM idx i JOIN read_parquet('{med.as_posix()}') m ON i.subject_id=m.subject_id WHERE ({pat}) AND {cond}"
            positive=con.execute(q).df();positive['positive']=True
            v=f[['subject_id','study_id']].merge(positive,on=['subject_id','study_id'],how='left',validate='one_to_one').positive.fillna(False).to_numpy()
            coverage=(f.has_emar_coverage.eq(True)|f.has_prescription_coverage.eq(True)).fillna(False).to_numpy() if window=='around24' else f.has_prescription_coverage.eq(True).fillna(False).to_numpy()
            f[window+'_'+cl]=np.where(v,'exposed',np.where(coverage,'observed_no_record','unknown'))
    # Exact elapsed setting windows, avoid an admission-linked ECG becoming automatically inpatient.
    ed=(f.ed_intime.le(f.ecg_time)&f.ed_outtime.ge(f.ecg_time)&f.ed_match_ambiguous.eq(False)).fillna(False)
    ip=(f.admittime.le(f.ecg_time)&f.dischtime.ge(f.ecg_time)&f.admission_match_ambiguous.eq(False)).fillna(False)
    ambiguous=f.ed_match_ambiguous.eq(True)|f.admission_match_ambiguous.eq(True)|f.ecg_time.isna()
    f['setting']=np.select([ambiguous,ed&ip,ed,ip],['unknown','both','ED','inpatient'],default='outside')
    f['potassium_clean']=pd.to_numeric(f.potassium_nearest,errors='coerce').where(f.potassium_delta_hours.abs().le(6)&f.potassium_nearest_censoring.isna())
    admissions=Path(str(data_path("b.mimic_validation_standardized_admissions_parquet")));schema=pq.ParquetFile(admissions).schema_arrow.names
    a.dump(a.OUT/'admissions_schema.json',{'source':str(admissions),'columns':schema})
    adm=subset(a,admissions,['subject_id','hadm_id','dischtime'],ids);con.register('adm_dev',adm)
    dx=a.bind(base/'diagnosis_evidence.parquet')
    cc=con.execute(f"SELECT i.subject_id,i.study_id,count(DISTINCT d.code_group) FILTER (WHERE d.code_group IS NOT NULL) as count_groups,count(DISTINCT h.hadm_id) as prior_admissions FROM idx i LEFT JOIN adm_dev h ON i.subject_id=h.subject_id AND h.dischtime<i.ecg_time LEFT JOIN read_parquet('{dx.as_posix()}') d ON d.subject_id=h.subject_id AND d.hadm_id=h.hadm_id GROUP BY i.subject_id,i.study_id").df()
    f=f.merge(cc,on=['subject_id','study_id'],how='left',validate='one_to_one');f['prior_code_group_count']=f.count_groups.where(f.prior_admissions.gt(0))
    con.close();f.to_parquet(a.OUT/'covariates_private.parquet',index=False)
    counts={fold:{col:f.loc[f.split.eq(fold),col].value_counts(dropna=False).to_dict() for col in ['setting','anchor_year_group','preferred_lvef_method']+[w+'_'+cl for w in ['around24','before7d'] for cl in classes]} for fold in ['train','validation']}
    counts['continuous']={fold:{col:{'observed':int(f.loc[f.split.eq(fold),col].notna().sum()),'missing':int(f.loc[f.split.eq(fold),col].isna().sum())} for col in ['bmi_before_ecg','potassium_clean','prior_code_group_count']} for fold in ['train','validation']}
    a.dump(a.OUT/'covariate_inventory.json',counts);print('COVARIATES_COMPLETE')

def extra_models(a):
    p=a.OUT/'covariates_private.parquet';f=a.read(p,pq.ParquetFile(p).schema_arrow.names);entry=a.load(a.OUT/'BEF-0063_entry.json');raw,names=raw_design(f,a.F);z=fixed_design(raw,entry['scaling_B3']);tr=f.split.eq('train').to_numpy();rows=[]
    def cat(cols):
        parts=[];ns=[]
        for col in cols:
            v=f[col].fillna('unknown').astype(str);levels=sorted(v[tr].unique());reference='observed_no_record' if 'observed_no_record' in levels else levels[0]
            for level in levels:
                if level!=reference:parts.append(v.eq(level).to_numpy(float));ns.append(col+'='+level)
        return np.column_stack(parts),ns
    def continuous(cols):
        parts=[];ns=[];meta={}
        for col in cols:
            v=pd.to_numeric(f[col],errors='coerce').to_numpy(float);missing=~np.isfinite(v);median=np.nanmedian(v[tr]);meta[col]={'train_median':float(median),'missing':int(missing.sum())}
            if not np.isfinite(median):continue
            parts.append(np.where(missing,median,v));ns.append(col)
            if missing[tr].any():parts.append(missing.astype(float));ns.append(col+'_missing')
        x=np.column_stack(parts);xx,sc=standard_extra(x,x[tr]);ns=[n for n,k in zip(ns,sc['keep']) if k];return xx,ns,meta
    for variant in ['around24','before7d','extended_partial','extended_prior_codes']:
        if variant in ['around24','before7d']:
            x,ns=cat([variant+'_'+c for c in ['antiarrhythmic_class_i_iii','digoxin','qt_prolonging_drug']]);point='W21';meta={}
        else:
            cc,cn=cat(['anchor_year_group','setting']);cols=['bmi_before_ecg','potassium_clean']+(['prior_code_group_count'] if variant=='extended_prior_codes' else [])
            xc,xn,meta=continuous(cols);x=np.column_stack([cc,xc]);ns=cn+xn;point='W22'
        zz=np.column_stack([z[:,:-1],x,z[:,-1]])
        for fold in ['train','validation']:
            ix=f.split.eq(fold).to_numpy();y=classify_lvef(f.preferred_lvef.to_numpy()[ix]);key=point+'_'+variant+('_potassium_fixed' if point=='W22' else '')+'_'+fold
            rows.append(model_boot(a,key,zz[ix],y,{'point':point,'variant':variant,'fold':fold,'columns':ns,'imputation':meta,'source':str(p),'full_requested_model':False if point=='W22' else True}))
    a.dump(a.OUT/'extra_models.json',rows)

def lvef(a):
    import statsmodels.api as sm
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    from aw168_ordinal import ordinal_fit,objective
    f=frame(a,'BEF-0063');entry=a.load(a.OUT/'BEF-0063_entry.json');rows=[];hist=[]
    for fold in ['train','validation']:
        for pop in ['eligible_reference','eligible_B3']:
            ff=f.loc[f.split.eq(fold)&f[pop]];v=ff.preferred_lvef.to_numpy(float);n=len(v)
            for j in range(101):hist.append({'point':'W12b','fold':fold,'population':pop,'bin_lower':j,'bin_upper':j+1 if j<100 else 100,'count':int(((v>=j)&(v<j+1)).sum()) if j<100 else int((v==100).sum()),'n':n})
            rows.append({'point':'W12b','fold':fold,'population':pop,'n':n,'exact39':int((v==39).sum()),'exact40':int((v==40).sum()),'exact41':int((v==41).sum()),'exact40_percent':100*float((v==40).mean()),'below40':int((v<40).sum()),'atmost40':int((v<=40).sum())})
    for marker in [a.F,a.M]:
        ff,good,_=masks(f,'reduced_lvef_strict',marker,'qrs_duration_ms');ff=ff.loc[good];raw,names=raw_design(ff,marker);scale=entry['scaling_B3'] if marker==a.F else scaling(raw,names);z=fixed_design(raw,scale)
        for fold in ['train','validation']:
            ix=ff.split.eq(fold).to_numpy();zz=z[ix];v=ff.preferred_lvef.to_numpy(float)[ix];n=len(v)
            # LVEF outcome is continuous, adjusted for B3 regressors, marker coefficient per fixed development SD.
            ols=sm.OLS(v,zz).fit(cov_type='HC1');rows.append({'point':'W12c','fold':fold,'marker':marker,'model':'linear','n':n,'beta_lvef_percentage_points':ols.params[-1],'ci':ols.conf_int()[-1],'sd':scale['sd'][-1]})
            for categories in [3,4]:
                y=np.select([v<40,v<50],[0,1],default=2) if categories==3 else np.select([v<35,v<40,v<50],[0,1,2],default=3)
                x=zz[:,1:];path=a.OUT/('ordinal_'+('F' if marker==a.F else 'M')+'_'+fold+'_'+str(categories)+'.json')
                if path.exists():r=a.load(path)
                else:
                    beta,grad=ordinal_fit(x,y,np.ones(n))
                    r={'point':'W12c','fold':fold,'marker':marker,'model':'ordinal'+str(categories),'n':n,'class_counts':np.bincount(y).tolist(),'beta':float(beta[x.shape[1]-1]),'higher_lvef_or':float(np.exp(beta[x.shape[1]-1])),'proportional_odds_assumed':True,'boot':[],'failures':{},'next':0,'initial':beta.tolist(),'gradient_max':grad}
                for j in range(r['next'],200):
                    if j%20==0:a.guard()
                    rng=np.random.default_rng(np.random.SeedSequence([a.SEED,12,int(fold=='validation'),j]));w=np.bincount(rng.integers(0,n,n),minlength=n)
                    try:
                        beta,_=ordinal_fit(x,y,w,initial=np.asarray(r['initial']))
                        r['boot'].append(float(beta[x.shape[1]-1]))
                    except Exception as ex:r['failures'][str(ex)]=r['failures'].get(str(ex),0)+1
                    r['next']=j+1
                    if (j+1)%20==0:a.dump(path,r);print('ORDINAL',marker,fold,categories,j+1,flush=True)
                if len(r['boot'])>=190:r['or_ci']=np.exp(np.quantile(r['boot'],[.025,.975]))
                a.dump(path,r);rows.append(r)
    a.dump(a.OUT/'lvef.json',{'rows':rows,'histogram':hist,'echo_text_validation':'nicht möglich: strukturierte Echo-Messwerte, keine Echo-Befundvolltexte im Releaseinventar'})

def pipeline(a):
    for phase in ['reliability','covariates','extra_models','simulation','simex','lvef']:
        if (a.OUT/(phase+'_manifest.json')).exists():continue
        a.PHASE=phase;a.dump(a.OUT/'status.json',{'phase':phase,'state':'running','pid':__import__('os').getpid()})
        print('PHASE',phase,flush=True);globals()[phase](a);a.finish();print('PHASE_COMPLETE',phase,flush=True)
    a.PHASE='pipeline'

def diagnose(a):
    p=a.OUT/'covariates_private.parquet';f=a.read(p,['split','preferred_lvef','potassium_nearest','potassium_delta_hours','potassium_nearest_censoring','setting','anchor_year_group'])
    print('CENSOR',f.potassium_nearest_censoring.value_counts(dropna=False).to_dict())
    for c in ['setting','anchor_year_group']:
        print(c);print(pd.crosstab([f.split,f[c].fillna('unknown')],classify_lvef(f.preferred_lvef)).to_string())

def refresh(a):
    # Correct null = uncensored under release lab_intervals.py, before reporting; old failed fits retained.
    p=a.OUT/'covariates_private.parquet';f=a.read(p,pq.ParquetFile(p).schema_arrow.names)
    f['potassium_clean']=pd.to_numeric(f.potassium_nearest,errors='coerce').where(f.potassium_delta_hours.abs().le(6)&f.potassium_nearest_censoring.isna())
    f.to_parquet(p,index=False);inv=a.load(a.OUT/'covariate_inventory.json')
    for fold in ['train','validation']:
        v=f.loc[f.split.eq(fold),'potassium_clean'];inv['continuous'][fold]['potassium_clean']={'observed':int(v.notna().sum()),'missing':int(v.isna().sum())}
    a.dump(a.OUT/'covariate_inventory.json',inv);reliability(a)
    print('REFRESH_COMPLETE')

def designs(a):
    models(a);extra_models(a)

def verify(a):
    from aw168_verify import verify as check
    check(a)

def report(a):
    from aw168_report import report as generate
    generate(a)

def finalcheck(a):
    import ast,re
    public=[a.ROOT/'nachrechnung/B_ENTWICKLUNG_W_AW168.md',a.ROOT/'nachrechnung/b_entwicklung_w.csv',a.ROOT/'koordination/abgaben/AW-168.md']
    for p in public:assert p.exists() and p.stat().st_size<1024**2
    for p in (a.ROOT/'werkzeuge/b_test').glob('aw168_*.py'):ast.parse(p.read_text(encoding='utf8'))
    csv=pd.read_csv(public[1],sep=';');assert set(csv.point)=={'W7','W8b','W12b','W12c','W15','W16a','W16b','W17','W21','W22'}
    assert not {'subject_id','study_id','hadm_id'}&set(csv.columns)
    assert not csv.duplicated(['point','fold','variant','metric']).any()
    for r in a.load(a.OUT/'models.json')+a.load(a.OUT/'extra_models.json'):
        if r.get('status'):continue
        rows=csv.loc[csv.source.str.contains('model_'+r['key']+'.json',regex=False)]
        target=rows.loc[rows.metric.eq('or')];assert len(target)>=1 and np.allclose(target.value,r['or'])
        assert np.allclose(target.ci_low,r['or_ci'][0]) and np.allclose(target.ci_high,r['or_ci'][1])
    f=frame(a,'BEF-0063');g=frame(a,'BEF-0049');ids=set(f.subject_id.astype(str))|set(g.subject_id.astype(str))|set(f.study_id.astype(str))|set(g.study_id.astype(str))
    for p in public:
        tokens=set(re.findall(r'(?<![\w.])\d{6,}(?![\w.])',p.read_text(encoding='utf-8-sig')));assert not ids&tokens
    report=public[0].read_text(encoding='utf8')
    for point in ['W15','W16a','W16b','W17','W7','W8b','W21','W22','W12b/c']:assert '## '+point in report
    assert report.count('**Methods (EN):**')==9 and report.count('**Results (EN):**')==9
    assert a.load(a.OUT/'verification.json')['all_checks_passed']
    # Immutable shared B3 implementations against first phase hashes.
    initial=a.load(a.OUT/'inventory_manifest.json')['inputs']
    for name in ['daten.py','statistik.py']:
        p=a.ROOT/'werkzeuge/b_test'/name;assert a.sha(p)==initial[str(p)]['sha256']
    manifests=[a.load(p) for p in a.OUT.glob('*_manifest.json') if p.name!='report_manifest.json']
    peak=max(m.get('peak_rss',0) for m in manifests);private=max(m.get('peak_private',0) for m in manifests);assert max(peak,private)<=1.5*1024**3
    result={'all_checks_passed':True,'public_files':{str(p):{'bytes':p.stat().st_size,'sha256':a.sha(p)} for p in public},'csv_rows':len(csv),'no_development_identifier_values_in_public_artifacts':True,'shared_B3_unchanged':True,'peak_rss_all_phases':peak,'peak_private_all_phases':private,'test_access':False}
    a.dump(a.OUT/'finalcheck.json',result);print('FINALCHECK_PASS',len(csv),'peak_rss',peak,'peak_private',private)
