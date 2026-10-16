"""Independent numerical checks of the measurement-error correction analysis; outputs aggregates only."""
import numpy as np
import statsmodels.api as sm
from scipy.stats import pearsonr,spearmanr
from aw157_models import rc,multirc,spline_design

def verify(a):
    e,p=a.outcome_data();points=a.json.loads(a.bind(a.OUT/'points.json').read_text());checks=[]
    rng=np.random.default_rng(157);x=rng.normal(size=(12,4));x[3]=x[2];w=rng.integers(0,4,12).astype(float);w[0]=2
    duplicated=np.repeat(x,w.astype(int),axis=0)
    ranks=np.column_stack([a.wrank(x[:,i],w) for i in range(4)])
    assert np.max(np.abs(a.wcor(ranks,w)-spearmanr(duplicated).statistic))<1e-12
    knots=e['aw157_knots_deg'];tail=a.rcs(np.array([200.,210.,220.]),knots);assert np.max(np.abs(np.diff(tail,n=2,axis=0)))<1e-10
    low=a.rcs(np.array([-20.,-10.,0.]),knots);assert np.max(np.abs(np.diff(low,n=2,axis=0)))<1e-10
    original=a.json.loads(a.bind(a.ROOT/'werkzeuge/b_test/parameter_entwicklung.json').read_text(encoding='utf8'))['hypotheses']['BEF-0063']
    assert e['scaling_B3']==original['scaling_B3']
    cf=a.json.loads(a.bind(a.OUT/'correlations.json').read_text())
    for mode in ['pearson_log1p','spearman']:
        v=p[[a.F+'_first',a.F+'_second',a.A+'_first',a.A+'_second']].to_numpy(float)
        if mode=='pearson_log1p':v[:,:2]=np.log1p(v[:,:2]);c=np.corrcoef(v,rowvar=False)
        else:c=spearmanr(v).statistic
        point=next(r for r in cf if r['group']=='gesamt' and r['mode']==mode)['point'];expected=(c[0,3]+c[2,1])/2/np.sqrt(c[0,1]*c[2,3])
        assert abs(expected-point['latent'])<1e-12
        checks.append({'check':'latent_independent_'+mode,'value':float(expected),'difference':float(expected-point['latent'])})
    allids=[]
    for fold in ['train','validation']:
        d=np.load(a.bind(a.OUT/(fold+'_private.npz')));z=d['z'];y=d['y'];w=np.ones(len(y));allids.append(d['ids']);assert len(np.unique(d['ids']))==len(y)
        glm=sm.GLM(y,z,family=sm.families.Binomial()).fit(maxiter=100,tol=1e-10)
        pt=next(r for r in points if r['fold']==fold and r['key']=='gesamt|linear|icc_0.617');assert abs(glm.params[-1]-pt['original_beta'])<1e-5
        check={'fold':fold,'GLM_beta':float(glm.params[-1]),'B3_difference':float(glm.params[-1]-pt['original_beta'])}
        for source in ['icc_0.617','icc_0.7','pairs_gesamt']:
            point=next(r for r in points if r['fold']==fold and r['key']=='gesamt|linear|'+source)
            result,M=rc(z,w,point['error_variance_deg2']/e['scaling_B3']['sd'][-2]**2,glm.params)
            corrected=sm.GLM(y,z@M,family=sm.families.Binomial()).fit(maxiter=100,tol=1e-10)
            assert abs(corrected.params[-1]-result['beta'])<1e-6
            assert abs(corrected.params[-1]-point['rc']['beta'])<1e-4
            check[source+'_RC_identity_difference']=float(corrected.params[-1]-result['beta'])
        for source in ['gesamt','zeitgleich']:
            point=next(r for r in points if r['fold']==fold and r['source']=='multirc_'+source)
            error=np.array(point['multirc']['error_covariance']);result,M=multirc(z,w,error,glm.params)
            if M is not None:
                corrected=sm.GLM(y,z@M,family=sm.families.Binomial()).fit(maxiter=100,tol=1e-10)
                assert abs(corrected.params[-1]-result['beta'])<1e-5
                assert abs(corrected.params[-1]-point['multirc']['beta'])<1e-4
                check['multirc_'+source+'_identity_difference']=float(corrected.params[-1]-result['beta'])
                # Independent difference covariance; raw degrees/F scaling.
                pp=p if source=='gesamt' else p.loc[p.hours_between.eq(0)]
                diff=np.column_stack([pp[a.A+'_second']-pp[a.A+'_first'],pp[a.F+'_second']-pp[a.F+'_first']])/np.array(e['scaling_B3']['sd'][-2:])
                assert np.max(np.abs(np.cov(diff,rowvar=False,ddof=1)/2-error))<1e-12
        rejected,_=multirc(z,w,np.eye(2)*100,glm.params);assert rejected['reason']=='latent_covariance_not_positive_definite'
        flexible=spline_design(z,z[:,-2]*e['scaling_B3']['sd'][-2]+e['scaling_B3']['mean'][-2],e,a.rcs)
        spline_fit=sm.GLM(y,flexible,family=sm.families.Binomial()).fit(maxiter=100,tol=1e-10)
        sp=next(r for r in points if r['fold']==fold and r['key']=='gesamt|spline|icc_0.617')
        assert abs(spline_fit.params[-1]-sp['original_beta'])<1e-4
        check['spline_GLM_difference']=float(spline_fit.params[-1]-sp['original_beta']);checks.append(check)
    assert not np.intersect1d(*allids).size
    for point in points:
        if 'simex' in point:
            expected=np.polyval(np.polyfit([0,.5,1,1.5,2],point['simex']['curve'],2),-1)
            assert abs(expected-point['simex']['beta'])<1e-10
    a.dump(a.OUT/'verification.json',{'status':'bestanden','checks':checks,'weighted_rank_duplicate_check':True,'spline_linear_tails':True,'PSD_rejection':True,'split_disjoint':True,'simex_polyfit_check':True,'utc':a.time.strftime('%Y-%m-%dT%H:%M:%SZ',a.time.gmtime())})
    print('VERIFY_COMPLETE')
