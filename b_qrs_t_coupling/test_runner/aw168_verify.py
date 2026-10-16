"""Unabhängige Kernzahl-/Gradient-/Nennerprüfung AW168."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import json
from pathlib import Path
import numpy as np,pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import check_grad
from scipy.special import expit
from scipy.stats import spearmanr
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from aw168_ordinal import objective,ordinal_fit
from aw168_analysen import frame

def verify(a):
    result={'test_access':False,'all_checks_passed':False,'details':[]}
    import scipy,statsmodels,pyarrow,duckdb,sys
    result['versions']={'python':sys.version.split()[0],'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'statsmodels':statsmodels.__version__,'pyarrow':pyarrow.__version__,'duckdb':duckdb.__version__}
    models=a.load(a.OUT/'models.json')+a.load(a.OUT/'extra_models.json');maxbeta=0.;maxauc=0.;gates=[]
    for r in models:
        if r.get('status'):
            assert r['status']=='nicht_schaetzbar';gates.append({'key':r['key'],'reason':r['reason']});continue
        assert r['next']==200 and len(r['boot'])+sum(r['failures'].values())==200
        if len(r['boot'])<190:assert 'or_ci' not in r and 'delta_auc_ci' not in r
        if 'or_ci' in r:
            q=np.quantile(r['boot'],[.025,.975],axis=0);assert np.allclose(np.exp(q[:,0]),r['or_ci']) and np.allclose(q[:,1],r['delta_auc_ci'])
        d=np.load(a.bind(a.OUT/('design_'+r['key']+'_private.npz')));z=d['z'];y=d['y'];assert len(y)==r['n'] and y.sum()==r['cases']
        fit=sm.GLM(y,z,family=sm.families.Binomial()).fit(maxiter=100,tol=1e-10);base=sm.GLM(y,z[:,:-1],family=sm.families.Binomial()).fit(maxiter=100,tol=1e-10)
        betaerr=abs(float(fit.params[-1])-r['beta']);maxbeta=max(maxbeta,betaerr);assert betaerr<2e-5
        # AUC independently via SciPy average ranks (no core.auc reuse).
        from scipy.stats import rankdata
        def independent_auc(s):
            ranks=rankdata(s);n1=int(y.sum());n0=len(y)-n1;return (ranks[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
        delta=independent_auc(z@fit.params)-independent_auc(z[:,:-1]@base.params);err=abs(delta-r['delta_auc']);maxauc=max(maxauc,err);assert err<2e-5
        result['details'].append({'key':r['key'],'beta_error_glm':betaerr,'delta_auc_error':err,'bootstrap_valid':len(r['boot'])})
    result.update({'model_cells':len(models),'estimable_model_cells':len(models)-len(gates),'nonestimable_gates':gates,'max_beta_error_glm':maxbeta,'max_delta_auc_error_independent':maxauc})
    f=frame(a,'BEF-0063');g=frame(a,'BEF-0049');assert not np.intersect1d(f.loc[f.split.eq('train'),'subject_id'],f.loc[f.split.eq('validation'),'subject_id']).size;assert not np.intersect1d(g.loc[g.split.eq('train'),'subject_id'],g.loc[g.split.eq('validation'),'subject_id']).size
    assert not f.subject_id.duplicated().any() and not g.subject_id.duplicated().any();result['development_identity_disjoint']=True
    cf=a.read(a.OUT/'covariates_private.parquet',['split','ehr_time_alignment_confirmatory_eligible'])
    result['medication_clock_alignment_flags']={fold:{str(k):int(v) for k,v in cf.loc[cf.split.eq(fold),'ehr_time_alignment_confirmatory_eligible'].value_counts(dropna=False).items()} for fold in ['train','validation']}
    for r in a.load(a.OUT/'simulation.json'):
        assert r['next']==200 and len(r['rows'])==200 and sorted(v['replicate'] for v in r['rows'])==list(range(200))
        for v in r['rows']:assert v['known_marker_beta']==0 if r['scenario']=='null' else v['known_marker_beta']>0
    result['simulation_runs_per_scenario']=200
    sims=a.load(a.OUT/'simex.json');assert len(sims)==8
    for r in sims:
        assert r['next']==200 and len(r['boot'])+sum(r['failures'].values())==200
        b=np.asarray(r['boot']);assert len(b)>=190
        assert np.allclose(np.exp(np.quantile(b[:,1:],[.025,.975],axis=0)),r['or_ci'])
        assert np.allclose(np.quantile(b[:,1:]/b[:,0,None],[.025,.975],axis=0),r['fraction_ci'])
        curve=np.asarray(r['result']['curve']);x=np.array([0,.5,1,1.5,2])
        assert abs(np.polyval(np.polyfit(x,curve,2),-1)-r['result']['quadratic_beta'])<1e-10
        assert abs(np.polyval(np.polyfit(x,curve,1),-1)-r['result']['linear_beta'])<1e-10
    result['simex_cells_valid']=8
    rel=a.load(a.OUT/'reliability.json');entry=a.load(a.OUT/'BEF-0063_entry.json');rr=f.loc[f.eligible_B3]
    from werkzeuge.b_test.statistik import raw_design,fixed_design
    raw,names=raw_design(rr,a.F);design=fixed_design(raw,entry['scaling_B3'])
    pairs=a.read(a.OUT/'pairs_private.parquet',['split',a.A+'_first',a.A+'_second',a.F+'_first',a.F+'_second'])
    def independent_icc(x):
        n,k=x.shape;grand=x.mean();row=x.mean(1);col=x.mean(0)
        msr=k*np.sum((row-grand)**2)/(n-1);msc=n*np.sum((col-grand)**2)/(k-1);mse=np.sum((x-row[:,None]-col[None,:]+grand)**2)/((n-1)*(k-1))
        return (msr-mse)/(msr+(k-1)*mse+k*(msc-mse)/n)
    for r in rel['rows']:
        pp=pairs.loc[pairs.split.eq(r['fold'])]
        for col,metric in [(a.A,'angle_icc'),(a.F,'marker_icc')]:assert abs(independent_icc(pp[[col+'_first',col+'_second']].to_numpy())-r[metric])<1e-11
        ix=rr.split.eq(r['fold']).to_numpy();z=design[ix];y=(rr.preferred_lvef.to_numpy()[ix]<40).astype(float)
        for mode,zz in [('with_marker',z),('without_marker',z[:,:-1])]:
            fit=sm.GLM(y,zz,family=sm.families.Binomial()).fit(cov_type='HC1');angle=len(fit.params)-2 if mode=='with_marker' else len(fit.params)-1
            assert abs(np.exp(fit.params[angle])-r[mode]['angle_or'])<2e-5
            # Full VIFs independently via residual fits, no outcome information.
            for c in rel['coefficients']:
                if c['fold']!=r['fold'] or c['mode']!=mode:continue
                ni=['intercept']+names if mode=='with_marker' else ['intercept']+names[:-1];j=ni.index(c['coefficient'])
                if j==0:c['vif']=None;continue
                other=np.delete(zz,j,axis=1);res=zz[:,j]-other@np.linalg.lstsq(other,zz[:,j],rcond=None)[0];c['vif']=float(np.var(zz[:,j])/np.var(res))
    a.dump(a.OUT/'reliability.json',rel);result['angle_coefficients_glm_reproduced']=True
    result['icc_independent_anova_reproduced']=True
    # Analytic ordinal gradient and independent reference optimizer.
    rng=np.random.default_rng(168999);x=rng.normal(size=(700,3));cuts=np.array([-.8,.3,1.2]);latent=x@np.array([.4,-.2,.7]);u=rng.uniform(size=700);cum=expit(cuts[None,:]-latent[:,None]);y=(u[:,None]>cum).sum(1)
    initial=np.r_[np.zeros(3),cuts[0],np.log(np.diff(cuts))];w=np.ones(700)
    graderr=check_grad(lambda t:objective(t,x,y,w,4)[0],lambda t:objective(t,x,y,w,4)[1],initial);assert graderr<1e-6
    beta,_=ordinal_fit(x,y,w);reference=OrderedModel(y,x,distr='logit').fit(method='bfgs',disp=False,maxiter=300,gtol=1e-9);diff=np.max(np.abs(beta-reference.params));assert diff<2e-5
    # Frequency weights must equal actual row duplication.
    ww=rng.integers(0,4,700);bw,_=ordinal_fit(x,y,ww);take=np.repeat(np.arange(700),ww);be,_=ordinal_fit(x[take],y[take],np.ones(len(take)));assert np.max(np.abs(bw-be))<1e-8
    result.update({'ordinal_gradient_error':float(graderr),'ordinal_independent_parameter_error':float(diff),'ordinal_frequency_weights_equivalent':True})
    lv=a.load(a.OUT/'lvef.json');ordrows=[r for r in lv['rows'] if r.get('model','').startswith('ordinal')];assert len(ordrows)==8
    from werkzeuge.b_test.daten import masks
    from werkzeuge.b_test.statistik import scaling
    maxordinal=0.;maxlinear=0.
    for r in ordrows:
        assert sum(r['class_counts'])==r['n'] and r['next']==200 and len(r['boot'])+sum(r['failures'].values())==200
        if len(r['boot'])>=190:assert np.allclose(np.exp(np.quantile(r['boot'],[.025,.975])),r['or_ci'])
        ff,good,_=masks(f,'reduced_lvef_strict',r['marker'],'qrs_duration_ms');ff=ff.loc[good];raw,ns=raw_design(ff,r['marker']);sc=entry['scaling_B3'] if r['marker']==a.F else scaling(raw,ns);zz=fixed_design(raw,sc);ix=ff.split.eq(r['fold']).to_numpy();x=zz[ix,1:];v=ff.preferred_lvef.to_numpy()[ix]
        y=np.select([v<40,v<50],[0,1],default=2) if r['model']=='ordinal3' else np.select([v<35,v<40,v<50],[0,1,2],default=3)
        independent=OrderedModel(y,x,distr='logit').fit(start_params=np.asarray(r['initial']),method='bfgs',disp=False,maxiter=300,gtol=1e-8)
        err=abs(float(independent.params[x.shape[1]-1])-r['beta']);maxordinal=max(maxordinal,err);assert err<2e-4
        ols_beta=np.linalg.lstsq(zz[ix],v,rcond=None)[0][-1];lr=next(row for row in lv['rows'] if row.get('model')=='linear' and row['marker']==r['marker'] and row['fold']==r['fold']);err=abs(ols_beta-lr['beta_lvef_percentage_points']);maxlinear=max(maxlinear,err);assert err<1e-10
    result['real_ordinal_max_beta_error_statsmodels']=float(maxordinal);result['linear_lvef_max_beta_error_lstsq']=float(maxlinear)
    for r in lv['rows']:
        if r['point']=='W12b':
            hh=[h for h in lv['histogram'] if h['fold']==r['fold'] and h['population']==r['population']];assert len(hh)==101 and sum(h['count'] for h in hh)==r['n'];assert r['atmost40']-r['below40']==r['exact40']
    result['lvef_histogram_and_threshold_counts_passed']=True
    # Echo source schemas only: original report text must not be inferred from structured raw value fields.
    es={}
    for name in ['echo_measurements_all','echo_measurements_long','echo_study_index','echo_study_links']:
        p=Path(str(data_path("b.mimic_validation_standardized")))/(name+'.parquet')
        if p.exists():es[name]=pq.ParquetFile(p).schema_arrow.names
    a.dump(a.OUT/'echo_source_schemas.json',es)
    result['echo_original_text_columns']=[(n,c) for n,cs in es.items() for c in cs if c in ['report_text','note_text','text','echo_report','report']]
    assert not result['echo_original_text_columns']
    result['all_checks_passed']=True;a.dump(a.OUT/'verification.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['details']},ensure_ascii=False))
