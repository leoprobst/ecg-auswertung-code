"""Paired person-level bootstrap fits with renewed LUDB calibration."""
def run():
    import json
    import numpy as np,pandas as pd
    from werkzeuge.b_test.aw166_run import OUT,ROOT,dump,guard,sha
    from werkzeuge.b_test.statistik import raw_design,fixed_design,fit
    from werkzeuge.nachrechnung.core import logistic_fit
    from werkzeuge.b_test.daten import classify_lvef
    f=pd.read_parquet(OUT/'population_private.parquet')
    entry=json.loads((OUT/'prepare_manifest.json').read_text())['entry']
    gate=json.loads((OUT/'signal_gate.json').read_text())
    rows=[];snr=[];coefs=np.load(OUT/'calibration_bootstrap_private.npy')
    records={}
    if gate['state']=='passed':
        for p in (OUT/'signal_records').glob('*.json'):
            obj=json.loads(p.read_text());records[obj['study_id']]=obj
        assert len(records)==len(f)
        f['t_snr']=[records[s]['t_snr'] for s in f.study_id]
        f.to_parquet(OUT/'population_with_snr_private.parquet',index=False)
    cal=json.loads((OUT/'calibration.json').read_text());upper=max(abs(v) for v in cal['group_difference_ci95_ms'])
    assert round(cal['group_difference_ms']*.5)==0 and round(upper*.5)==0
    for fold in ['train','validation']:
        guard();ff=f[f.split.eq(fold)].copy();y=classify_lvef(ff.preferred_lvef)
        x,names=raw_design(ff,'qrs_t_normalized_discrete_frechet');z=fixed_design(x,entry['scaling_B3'])
        full=fit(z,y,ff.subject_id.to_numpy())
        rows.append({'fold':fold,'scenario':'original_full_reference','n':len(ff),'cases':int(y.sum()),'controls':int(len(y)-y.sum()),'excluded_corrected_missing':0,'or':full['or'],'or_ci95':full['or_ci95_wald'],'log_or':full['beta'],'remaining_log_or_fraction':1.0,'fraction_ci95':[1.0,1.0],'bootstrap_valid':0,'bootstrap_requested':0,'interval_type':'CR1 Wald, full H1 reference','classification':'unkorrigierte Gesamt-H1-Referenz','source':'population_private.parquet; fixed B3; aw166_models.py','failures':[]})
        scenarios=['original','group_injected','worst_plus','worst_minus']
        if gate['state']=='passed':scenarios+=['corrected']
        if gate['state']=='passed':
            vv=np.array([records[s]['values']['corrected'] for s in ff.study_id],dtype=float);common=np.isfinite(vv).all(1)
        else:common=np.ones(len(ff),bool)
        x=x[common];ff=ff[common];y=y[common];z=z[common]
        initial=fit(z,y,ff.subject_id.to_numpy());betas={'original':initial['beta']}
        for scenario in scenarios[1:]:
            zz=z.copy()
            if scenario=='corrected':
                xx=x.copy();v=np.array([records[s]['values']['corrected'] for s in ff.study_id],float);xx[:,-1]=v[:,0];xx[:,-2]=v[:,1];zz=fixed_design(xx,entry['scaling_B3'])
            betas[scenario]=fit(zz,y,ff.subject_id.to_numpy())['beta']
        boot={k:[] for k in scenarios};failures=[]
        rng=np.random.default_rng(np.random.SeedSequence([16620261013,2,0 if fold=='train' else 1]))
        ids=ff.study_id.to_numpy();morph=ff[['amplitude_mv','flatness_mv_ms']].to_numpy(float)
        for b in range(500):
            if b%25==0:guard()
            ix=rng.integers(0,len(ff),len(ff));weights=np.bincount(ix,minlength=len(ff)).astype(float)
            try:
                base=logistic_fit(z,y,weights)[-1]
                for k in scenarios:
                    if k!='corrected':boot[k].append([base,base])
                zz=z.copy()
                if gate['state']=='passed':
                    samples=np.rint(-(np.column_stack([np.ones(len(ff)),morph])@coefs[b])*.5).astype(int)
                    v=np.array([records[s]['shift_grid'][str(t)] for s,t in zip(ids,samples)],float)
                    if not np.isfinite(v[weights>0]).all():raise ValueError('bootstrap corrected marker missing')
                    v[weights==0]=x[weights==0][:,[-1,-2]]
                    xx=x.copy();xx[:,-1]=v[:,0];xx[:,-2]=v[:,1];zz=fixed_design(xx,entry['scaling_B3'])
                corrected=logistic_fit(zz,y,weights)[-1] if gate['state']=='passed' else None
                if 'corrected' in scenarios:boot['corrected'].append([base,corrected])
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as e:failures.append({'draw':b,'reason':str(e)})
        for scenario in scenarios:
            bb=np.array(boot[scenario]);ratio=betas[scenario]/betas['original'];valid=len(bb)
            ci=np.quantile(np.exp(bb[:,1]),[.025,.975]).tolist() if valid>=475 else None
            rci=np.quantile(bb[:,1]/bb[:,0],[.025,.975]).tolist() if valid>=475 else None
            rows.append({'fold':fold,'scenario':scenario,'n':len(ff),'cases':int(y.sum()),'controls':int(len(y)-y.sum()),'excluded_corrected_missing':int((~common).sum()),'or':float(np.exp(betas[scenario])),'or_ci95':ci,'log_or':betas[scenario],'remaining_log_or_fraction':ratio,'fraction_ci95':rci,'bootstrap_valid':valid,'bootstrap_requested':500,'classification':('erklärt Zusatz nicht' if ratio>=.8 else 'teilweise' if ratio>=.5 else 'wesentlicher Teil möglicherweise Messartefakt'),'source':'population_private.parquet + signal_records + calibration_bootstrap_private.npy; aw166_models.py','failures':failures})
            dump(f'bootstrap_{fold}_{scenario}.json',boot[scenario])
        dump('model_results.json',rows)
    dump('models_manifest.json',{'code_sha256':sha(__file__),'input_population_sha256':sha(OUT/'population_private.parquet'),'calibration_sha256':sha(OUT/'calibration.json'),'signal_gate':gate,'test_access':False})
