"""Signal recomputation identical to the production pipeline; checkpoints on local storage, no substitute indices."""
def run(limit=None):
    import os, json, time, hashlib
    from pathlib import Path
    import numpy as np,pandas as pd,psutil
    from werkzeuge.b_test.aw166_run import OUT,dump,sha,guard
    from werkzeuge.varianten.adapter import prepare_record,context
    from werkzeuge.b_stoertest.run import OPTIONS,shifted
    from werkzeuge.varianten.adapter import release_import
    fr=release_import()
    import werkzeuge.varianten.adapter as adapter
    # The helper otherwise re-resolves every loaded release module at every
    # shifted context, dominating runtime without changing an input or result.
    # Verify the import tree once, retain exactly that module object, verify at end.
    verified_release_import=adapter.release_import
    adapter.release_import=lambda:fr
    names=['qrs_t_normalized_discrete_frechet','spatial_qrs_t_angle_deg']
    def compute_full(p,d,selected=names):
        record=fr.VCGTransformRegistry.transform(p['paths']['morphology'],'kors').xyz
        robust=fr.VCGTransformRegistry.transform(p['robust'],'kors').xyz
        v=fr.BiomarkerRegistry.compute_all(delineation=d,ecg_12lead=p['robust'],vcg_xyz=robust,fs=p['fs'],names=selected,record_ecg_12lead=p['paths']['morphology'],record_vcg_xyz=record,record_raw_ecg_12lead=p['loaded'].signal_millivolts,record_acquisition=p['acquisition'])
        r={};fr._store_biomarker_results(r,v,[],metadata={});return [r.get(k) for k in selected]
    from vcg_biomarker.core.biomarkers.discovery_v3_biomarkers import _resampled,_unit_rows,EPS
    from vcg_biomarker.core.delineation.base import measurement_confidence_threshold
    dependencies=fr.BiomarkerRegistry.get(names[0]).spec.dependencies
    from scipy.spatial.distance import cdist
    def frechet(cost):
        if cost.ndim!=2 or not cost.size or not np.isfinite(cost).all():return float('nan')
        n,m=cost.shape;tab=np.full((n+1,m+1),np.inf);tab[0,0]=0
        for diagonal in range(2,n+m+1):
            rr=np.arange(max(1,diagonal-m),min(n,diagonal-1)+1);cc=diagonal-rr
            tab[rr,cc]=np.maximum(cost[rr-1,cc-1],np.minimum(np.minimum(tab[rr-1,cc],tab[rr,cc-1]),tab[rr-1,cc-1]))
        return float(tab[n,m])
    def compute(p,d):
        # Same production segment preparation, centring, scale, gates and cdist;
        # independent antidiagonal evaluation of the exact max/min recurrence.
        c=context(p,delineation=d);value=None
        eligible=all(c.segment_confidence(s)>=measurement_confidence_threshold(s) for s in dependencies)
        q=_resampled(c,'qrs',64) if eligible else None;t=_resampled(c,'t',64) if eligible else None
        if q is not None and t is not None:
            qc=q-np.mean(q,axis=0);tc=t-np.mean(t,axis=0)
            qs=float(np.sqrt(np.mean(np.sum(qc*qc,axis=1))));ts=float(np.sqrt(np.mean(np.sum(tc*tc,axis=1))))
            if qs>EPS and ts>EPS and len(_unit_rows(q))>=3 and len(_unit_rows(t))>=3:value=frechet(cdist(qc/qs,tc/ts))
        angle=compute_full(p,d,[names[1]])[0]
        return [value,angle]
    f=pd.read_parquet(OUT/'population_private.parquet');meta=pd.read_parquet(OUT/'meta_private.parquet')
    f=meta.merge(f[['study_id','y','amplitude_mv','flatness_mv_ms']+names],on='study_id',validate='one_to_one').sort_values('study_id')
    if limit is not None:f=f.head(limit)
    cal=json.loads((OUT/'calibration.json').read_text());coef=np.array(cal['params']);diff=cal['group_difference_ms'];upper=max(abs(v) for v in cal['group_difference_ci95_ms'])
    coef_boot=np.load(OUT/'calibration_bootstrap_private.npy')
    (OUT/'signal_records').mkdir(exist_ok=True)
    binding={'script':sha(__file__),'population':sha(OUT/'population_private.parquet'),'calibration':sha(OUT/'calibration.json')}
    start=time.monotonic();peak=0
    def one(row):
        guard()
        if psutil.Process().memory_info().rss>1.5*1024**3:raise RuntimeError('RSS limit')
        cp=OUT/'signal_records'/f"{int(row['study_id'])}.json"
        if cp.exists():
            try:old=json.loads(cp.read_text())
            except json.JSONDecodeError:old=None
            if old is not None and old['binding']==binding and limit is None:return True
        else:old=None
        p=prepare_record(row,OPTIONS);v=compute(p,p['delineation']);expected=[row[k] for k in names]
        assert np.isclose(np.asarray(v,float),np.asarray(compute_full(p,p['delineation']),float),rtol=0,atol=0,equal_nan=True).all(), 'Fast/original baseline'
        if not np.isclose(np.asarray(v,float),np.asarray(expected,float),rtol=1e-8,atol=0).all():
            dump('signal_equivalence_failure_private.json',{'row':row,'actual':v,'expected':expected})
            return False
        t=context(p).vcg_segment('t');snr=None
        if t is not None and len(t)>=5:
            residual=np.diff(t,n=2,axis=0).ravel()/np.sqrt(6)
            noise=1.4826*np.median(abs(residual-np.median(residual)))
            if noise>0:snr=row['amplitude_mv']/noise
        pred=np.array([1,row['amplitude_mv'],row['flatness_mv_ms']])@coef
        shifts={'corrected':-pred,'group_injected':diff*row['y'],'worst_plus':upper*row['y'],'worst_minus':-upper*row['y']}
        vals={};actual_ms={};grid={0:v}
        xx=np.array([1,row['amplitude_mv'],row['flatness_mv_ms']])
        # Exact production sample grid across all calibration draws, no interpolation.
        sample_grid=set(np.rint(-(coef_boot@xx)*p['fs']/1000).astype(int).tolist())
        sample_grid.update(round(ms*p['fs']/1000) for ms in shifts.values())
        for sample in sorted(sample_grid):
            if sample not in grid:grid[sample]=compute(p,shifted(p['delineation'],'t_offset',sample))
        for scenario,ms in shifts.items():
            samples=round(ms*p['fs']/1000);actual_ms[scenario]=samples*1000/p['fs']
            vals[scenario]=grid[samples]
        point_samples=round(shifts['corrected']*p['fs']/1000)
        assert np.isclose(np.asarray(vals['corrected'],float),np.asarray(compute_full(p,shifted(p['delineation'],'t_offset',point_samples)),float),rtol=0,atol=0,equal_nan=True).all(), 'Fast/original corrected'
        new={'binding':binding,'subject_id':row['subject_id'],'study_id':row['study_id'],'split':row['split'],'t_snr':snr,'baseline':v,'values':vals,'actual_shift_ms':actual_ms,'shift_grid':grid,'equivalent':True}
        if old is not None:
            for key in ['t_snr','baseline','values','actual_shift_ms']:
                assert new[key]==old[key], 'Thread/restart equivalence'
            if {str(k):v for k,v in new['shift_grid'].items()}!=old['shift_grid']:
                dump('fast_grid_difference_private.json',{'old':old,'new':new})
                changed=[]
                for k,value in new['shift_grid'].items():
                    if value!=old['shift_grid'].get(str(k)):
                        original=compute_full(p,shifted(p['delineation'],'t_offset',k))
                        assert np.isclose(np.asarray(value,float),np.asarray(original,float),rtol=0,atol=0,equal_nan=True).all(), 'Recomputed production grid'
                        changed.append(k)
                new['legacy_grid_recomputed_samples']=changed
            new['previous_binding']=old['binding']
        dump('signal_records/'+cp.name,new)
        return True
    # FeatureContext is threading.local in the bound release. One Python process,
    # one record worker, native BLAS/OMP one thread; original checks each record.
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import nullcontext
    with (ThreadPoolExecutor(max_workers=4) if limit is None else nullcontext()) as executor:
        results=executor.map(one,f.to_dict('records')) if executor is not None else map(one,f.to_dict('records'))
        for i,ok in enumerate(results,1):
            peak=max(peak,psutil.Process().memory_info().rss)
            if not ok:
                dump('signal_gate.json',{'state':'failed','reason':'production equivalence','verified_before_failure':i-1,'n_total':len(f)})
                return
            if i%25==0:
                dump('checkpoint.json',{'phase':'signals','done':i,'total':len(f),'seconds':time.monotonic()-start,'peak_rss':peak,'pid':os.getpid(),'record_threads':4})
    assert verified_release_import() is fr
    dump('signal_gate.json' if limit is None else 'profile_gate.json',{'state':'passed','n_total':len(f),'peak_rss':peak,'seconds':time.monotonic()-start})
