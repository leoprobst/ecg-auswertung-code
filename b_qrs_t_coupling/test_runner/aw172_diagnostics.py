"""Missing report outputs, produced from the run's own unchanged checkpoints.

R0: missingness strata and patterns, W19 weight diagnostics, quartile occupancy.
W16: identical bound point replay, producing only additional coefficient
and curve outputs; marker point values must agree bitwise with the main run.
No new models, parameters, test sources or bootstrap draws.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import sys,json,hashlib,time,ctypes,struct
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=Path(str(data_path("b.aw172")))
F='qrs_t_normalized_discrete_frechet';M='qrs_t_frechet_beat_median';A='spatial_qrs_t_angle_deg'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def now():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def load(p):
    p=Path(p);assert p.stat().st_size<=20*1024**2
    return json.loads(p.read_text(encoding='utf8'))
def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if hasattr(v,'tolist'):return clean(v.tolist())
    if isinstance(v,float) and not __import__('math').isfinite(v):return None
    return v
def dump(name,v):(OUT/name).write_text(json.dumps(clean(v),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def guard():
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total','free','page','commit','vtotal','vfree','extended')]
    import psutil
    m=Memory();m.length=ctypes.sizeof(m)
    while True:
        assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        if min(m.free,m.commit)>=1.5*1024**3:break
        dump('diagnostics_status.json',dict(state='waiting_memory',pid=os.getpid(),utc=now()));time.sleep(30)
    info=psutil.Process().memory_info()
    assert max(info.private,info.rss,getattr(info,'peak_pagefile',0),getattr(info,'peak_wset',0))<=1.5*1024**3,'Resource limit exceeded'

def main():
    assert (OUT/'access_finished.json').exists(),'main run still active'
    if (OUT/'diagnostics_manifest.json').exists():return
    guard()
    binding=load(OUT/'bindung_vor_test.json')
    for p,h in binding['files'].items():assert sha(p)==h,'binding changed'
    if (OUT/'global_stop.json').exists():
        dump('diagnostics_manifest.json',dict(status='STOP_wegen_Globalgate',new_original_test_source_opened=False));return
    import numpy as np,pyarrow.parquet as pq,pyarrow.dataset as ds
    from scipy.special import expit
    from werkzeuge.b_test import aw171_kernels as k
    from werkzeuge.b_test.statistik import raw_design,fixed_design
    from werkzeuge.b_test.daten import masks
    params=load(str(data_path("b.aw171_parameters_json")))
    original=load(ROOT/'werkzeuge/b_test/parameter_entwicklung.json')
    inputs={};records=[];identity=[]
    def read(name,columns):
        guard();path=OUT/name;pf=pq.ParquetFile(path)
        assert set(columns)<=set(pf.schema_arrow.names),'checkpoint columns'
        inputs[str(path)]=dict(sha256=sha(path),columns=columns,filter='own immutable selected checkpoint; no new index')
        return ds.dataset(str(path),format='parquet').to_table(columns=columns).to_pandas()
    def save(key,W,ep,marker,pop,point,status='schaetzbar',reason=None):
        r=dict(analysis_id=key,W=W,endpoint=ep,marker=marker,population=pop,status=status,point=point,bootstrap=None,
               origin='supplemental report diagnostics; original numeric recipe unchanged',script_hash=sha(__file__))
        if reason:r['reason']=reason
        dump(key+'.json',r);records.append(r)
    with (ROOT/'koordination/TESTZUGRIFFE.md').open('a',encoding='utf8') as f:
        f.write(f'\n| {now()} | AW-172 | Fortsetzung: Ausgabediagnostik aus eigenen Checkpoints | Bindung `{sha(OUT/"bindung_vor_test.json")}`; Berichtadapter `{sha(__file__)}` | R0 W18/W19/W26; identische W16-Punktreplays mit Bitabgleich; keine neuen Originalquellen/Modelle/Parameter |\n')
    for ep,marker in [('H1',F),('H4',M)]:
        match=load(OUT/(ep+'_index_match.json'))['point']
        if not match['matched']:continue
        cols=['subject_id','y','age','sex_numeric','heart_rate','qrs_duration_ms','qtc_fridericia_ms',A,F,M,'sinus_selector']
        D=read(ep+'_D_private.parquet',cols);C=read(ep+'_C_private.parquet',cols)
        assert D.subject_id.nunique()==len(D)==match['reference_n']
        assert C.subject_id.nunique()==len(C)==match['complete_n'] and int(C.y.sum())==match['cases']
        # The pipeline sinus flag is a reporting category, not an exclusion of AF/pacing.
        strata={'all':np.ones(len(D),bool),'qrs_lt120':D.qrs_duration_ms.lt(120).to_numpy(),
                'qrs_ge120':D.qrs_duration_ms.ge(120).to_numpy(),'qrs_unknown':D.qrs_duration_ms.isna().to_numpy(),
                'sinus_flag_positive':D.sinus_selector.eq(True).fillna(False).to_numpy(),
                'sinus_flag_nonpositive':D.sinus_selector.eq(False).fillna(False).to_numpy(),
                'sinus_flag_unknown':D.sinus_selector.isna().to_numpy(),
                'pacing_category_unknown':np.ones(len(D),bool),'AF_category_unavailable':np.ones(len(D),bool)}
        rows=[]
        for stratum,selection in strata.items():
            for cl in [0,1]:
                sub=D.loc[selection&D.y.eq(cl).to_numpy()]
                for variable in ['age','sex_numeric','heart_rate','qrs_duration_ms','qtc_fridericia_ms',A,marker]:
                    n=int((~np.isfinite(sub[variable].to_numpy(float))).sum())
                    rows.append(dict(variable=variable,category=str(cl),stratum=stratum,denominator=len(sub),n=n,fraction=n/len(sub) if len(sub) else None))
                hm=sub.heart_rate.isna();qm=sub.qtc_fridericia_ms.isna()
                for name,sel in [('both_observed',~hm&~qm),('HF_only_missing',hm&~qm),('QTc_only_missing',~hm&qm),('both_missing',hm&qm)]:
                    n=int(sel.sum());rows.append(dict(variable='HF_QTc_joint_pattern',category=str(cl)+':'+name,stratum=stratum,denominator=len(sub),n=n,fraction=n/len(sub) if len(sub) else None))
        save(ep+'_missingness_strata_export','W18',ep,marker,'D_e; fixed available strata; text-dependent categories unknown',dict(rows=rows))
        pack=params['endpoints'][ep]['markers'][marker];cuts=np.asarray(pack['quartiles']['bounds'])
        model_binding={name:dict(**model,model_hash=hashlib.sha256(json.dumps(model,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                      threshold_hash=hashlib.sha256(json.dumps(model['threshold'],separators=(',',':')).encode()).hexdigest()) for name,model in pack['models'].items()}
        save(ep+'_fixed_model_binding_export','W20',ep,marker,'development-fixed coefficients and control-derived thresholds',dict(models=model_binding))
        group=np.searchsorted(cuts,C[marker].to_numpy(float),side='left');qr=[]
        for q in range(4):
            for category,sel in [('all',np.ones(len(C),bool)),('case',C.y.eq(1).to_numpy()),('control',C.y.eq(0).to_numpy())]:
                qr.append(dict(category=category,stratum='Q'+str(q+1),n=int(((group==q)&sel).sum()),denominator=int(sel.sum())))
        save(ep+'_quartile_counts_export','W26',ep,marker,'C_e; developmental boundaries, side=left',dict(rows=qr,bounds=cuts,medians=pack['quartiles']['medians']))
        z=fixed_design(raw_design(C,marker)[0],pack['scale']);y=C.y.to_numpy(float)
        # Conditional variance / latent eigenvalues as an X diagnostic; also reported
        # when the original RC run stopped, without rescuing the stop.
        ww=np.ones(len(y));ai=z.shape[1]-2;other=z[:,[i for i in range(z.shape[1]) if i!=ai]]
        coef=np.linalg.solve(other.T@(ww[:,None]*other),other.T@(ww*z[:,ai]));res=z[:,ai]-other@coef
        v=float(ww@(res**2)/ww.sum());tau=params['error']['angle_variance']/pack['scale']['sd'][-2]**2
        error_diag=dict(conditional_variance_standardized=v,conditional_variance_degree2=v*pack['scale']['sd'][-2]**2,
                        conditional_reliability=1-tau/v,transported_univariate_angle_error_degree2=params['error']['angle_variance'])
        if ep=='H1':
            from werkzeuge.b_test.aw157_models import cov
            other=z[:,:-2];observed=z[:,-2:];co=np.linalg.solve(other.T@(ww[:,None]*other),other.T@(ww[:,None]*observed));res=observed-other@co
            S=cov(res,ww);sd=np.asarray(pack['scale']['sd'][-2:]);E=np.asarray(params['error']['covariance'])/np.outer(sd,sd)
            error_diag.update(residual_covariance_standardized=S,error_covariance_standardized=E,latent_covariance_eigenvalues=np.linalg.eigvalsh(S-E),error_correlation=float(E[0,1]/np.sqrt(E[0,0]*E[1,1])))
        save(ep+'_conditional_error_diagnostics_export','W16',ep,marker,'same C_e; original linear-angle basis; X diagnostics only',error_diag)
        for variant in ['raw','clipped','logit','spline']+(['joint'] if ep=='H1' else []):
            key=ep+'_error_'+variant;source=load(OUT/(key+'.json'))
            if source['status']!='schaetzbar':
                save(key+'_curve_export','W16',ep,marker,'C_e',{},'STOP','Original point fit not estimable; no rescue replay');continue
            guard();captured=[];baseline={};solver=k.solver
            def traced(x,yy,w=None,initial=None,diagnostics=False):
                if len(captured)%25==0:guard()
                b=solver(x,yy,w,initial,diagnostics)
                captured.append(np.asarray(b).copy())
                if len(captured)==2:baseline.update(x=np.asarray(x).copy(),b=np.asarray(b).copy())
                return b
            k.solver=traced
            try:
                out=k.corrected(z,y,np.ones(len(y)),pack['scale'],params['error'],variant,
                     np.random.SeedSequence([1692026,16,int(ep[-1]),params['simex']['parameter_codes'][variant],0]),20,pack['spline'])
            finally:k.solver=solver
            fields=['uncorrected_beta','linear_beta','quadratic_beta','linear_delta_beta','quadratic_delta_beta']
            equal=all(struct.pack('!d',float(out[name]))==struct.pack('!d',float(source['point'][name])) for name in fields)
            equal=equal and all(struct.pack('!d',float(a))==struct.pack('!d',float(b)) for a,b in zip(out['curve'],source['point']['curve']))
            identity.append(dict(analysis_id=key,marker_point_and_curve_bitwise_equal=equal))
            if not equal:
                save(key+'_curve_export','W16',ep,marker,'C_e',{},'STOP','Point replay bitwise comparison failed; original values unchanged');continue
            assert len(captured)==82
            curve=np.stack([captured[1]]+[np.mean(captured[2+i*20:2+(i+1)*20],axis=0) for i in range(4)])
            angle_indices=list(range(7,curve.shape[1]-1))
            result=dict(angle_basis='development-standardized RCS components' if variant=='spline' else 'unscaled logit(theta/180)' if variant=='logit' else 'angle per fixed developmental SD',
                        lambda_values=[0,.5,1,1.5,2],angle_curve=curve[:,angle_indices],point_draws=20,marker_bitwise_identity=True,
                        monte_carlo_seed=[1692026,16,int(ep[-1]),params['simex']['parameter_codes'][variant],0],
                        corrected_wald_status='nicht_verfuegbar; keine neue Composite-/RC-Varianzregel nach Testsicht')
            lam=np.array([0,.5,1,1.5,2])
            for order,name in [(1,'linear'),(2,'quadratic')]:
                poly=np.column_stack([lam**j for j in range(order+1)])
                extrap=np.array([(-1)**j for j in range(order+1)])@np.linalg.lstsq(poly,curve,rcond=None)[0]
                result[name+'_angle_coefficients']=extrap[angle_indices]
            # Uncorrected base fit: the existing CR1 recipe exactly, no new fit.
            bx=baseline['x'];b=baseline['b'];p=expit(bx@b);n,dimension=bx.shape
            h=bx.T@((p*(1-p))[:,None]*bx)+np.diag(np.r_[0.,np.full(dimension-1,1e-6)])
            hi=np.linalg.inv(h);u=bx*(y-p)[:,None];cov=hi@(u.T@u)@hi*n/(n-1)*(n-1)/(n-dimension)
            se=np.sqrt(cov[-1,-1]);result['uncorrected_marker_wald_low']=float(np.exp(b[-1]-1.95996398454*se));result['uncorrected_marker_wald_high']=float(np.exp(b[-1]+1.95996398454*se))
            save(key+'_curve_export','W16',ep,marker,'same C_e; identical point replay only',result)
    techcols=['subject_id','study_id','split','ecg_time','age','sex','heart_rate','machine_heart_rate_bpm','machine_metadata_usable_for_primary',
              'qrs_duration_ms','qtc_fridericia_ms',A,F,'label_state','eligibility_state','index_selection_strategy','ehr_time_alignment_confirmatory_eligible',
              'preferred_lvef','lvef_conflict_gt_10','nearest_tte_measurement_id','nearest_tte_delta_hours','time_delta_hours','time_precision','evidence_ids_json',
              'classical_cornell_product','classical_sokolow_lyon','magnitude_area_qrs','spatial_low_voltage_qrs_peak_magnitude','classical_right_axis_deviation']
    source=OUT/'H1_W19.json'
    if source.exists() and load(source)['status']=='schaetzbar':
        target=load(source)['point'];tech=read('technical_index_private.parquet',techcols)
        tech['index_selection_strategy']='first_verified_ecg_reference_result_blind'
        f,valid,_=masks(tech,'reduced_lvef_strict',F,'qrs_duration_ms')
        raw,_=raw_design(f,F,strong=True);good=valid&np.isfinite(raw).all(1);f=f.loc[good]
        assert len(f)==target['n'] and int(f.preferred_lvef.lt(40).sum())==target['cases']
        model=load(OUT/'w19_development.json');vx=fixed_design(raw[good],model['scale']);vold=fixed_design(raw[good,:-1],original['verification'])
        pn=expit(vx@np.asarray(model['beta']));po=expit(vold@np.asarray(original['verification']['beta']));assert (pn>0).all() and (po>0).all()
        y=f.preferred_lvef.lt(40).to_numpy();stats=[];q=[0,.01,.5,.99,1]
        for name,p,cap in [('p95',pn,model['p95']),('p99',pn,model['p99']),('old_p99_common_set',po,original['verification']['p99_train_verified'])]:
            raww=1/p;w=np.minimum(raww,cap)
            item=dict(model=name,cap=cap,capped_fraction=float(np.mean(raww>cap)),propensity_quantiles=np.quantile(p,q,method='linear'),weight_quantiles=np.quantile(w,q,method='linear'))
            for c in [0,1]:
                wc=w[y==bool(c)];ess=float(wc.sum()**2/(wc@wc));assert abs(ess-target[name+'_ess_'+str(c)])<1e-9
                item['ess_'+str(c)]=ess
            stats.append(item)
        save('H1_weight_diagnostics_export','W19','H1',F,'same first-technical marker/B4-complete outcome set; no refit',dict(rows=stats,quantile_probabilities=q,n=len(f),cases=int(y.sum()),controls=int((~y).sum())))
    dump('diagnostics_manifest.json',dict(status='completed',utc=now(),source_hash=sha(__file__),inputs=inputs,point_identity=identity,
         records=[r['analysis_id'] for r in records],new_original_test_source_opened=False,new_model_variant=False,new_bootstrap=False,
         existing_checkpoint_person_rows_read=True,corrected_wald_remains_unbound=True))
    print('AW172_REPORT_DIAGNOSTICS_COMPLETED',len(records))

if __name__=='__main__':
    sys.stdout=(OUT/'diagnostics_stdout.log').open('a',encoding='utf8',buffering=1)
    sys.stderr=(OUT/'diagnostics_stderr.log').open('a',encoding='utf8',buffering=1)
    try:main()
    except Exception as ex:
        dump('diagnostics_failure.json',dict(error_class=type(ex).__name__,detail=str(ex),utc=now()));raise
